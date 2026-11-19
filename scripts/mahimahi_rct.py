#!/usr/bin/env python3
"""Compare download experience from the planned request schedule, including unfinished requests."""

import argparse
import csv
import math
import statistics
from dataclasses import dataclass, replace
from pathlib import Path

from mahimahi_eval import parse_meta_from_stem

DEADLINES_S = (1, 5, 15, 30)


@dataclass(frozen=True)
class Timing:
    id: int
    size: int
    scheduled_us: int
    status: str = "unobserved"
    started_us: int | None = None
    first_byte_us: int | None = None
    completed_us: int | None = None

    @property
    def rct_us(self) -> int | None:
        if self.completed_us is None:
            return None
        return max(0, self.completed_us - self.scheduled_us)

    @property
    def ttfb_us(self) -> int | None:
        if self.first_byte_us is None:
            return None
        return self.first_byte_us - self.scheduled_us

    @property
    def transfer_us(self) -> int | None:
        if self.completed_us is None or self.first_byte_us is None:
            return None
        return self.completed_us - self.first_byte_us


def load_schedule(path: Path) -> list[Timing]:
    planned = []
    scheduled_us = 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        gap_ms, size = map(int, line.split()[:2])
        scheduled_us += gap_ms * 1000
        planned.append(Timing(len(planned) + 1, size, scheduled_us))
    return planned


def csv_rows(path: Path) -> dict[int, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="") as source:
        return {int(row["id"]): row for row in csv.DictReader(source) if int(row["id"]) > 0}


def load_timings(plan: list[Timing], run: Path, stem: str, stack: str) -> tuple[str, list[Timing]]:
    path = run / f"{stem}.request-timings.csv"
    if stack == "mortise":
        path = run / f"{stem}.rct.csv"
    if not path.exists():
        return "missing", plan
    rows = csv_rows(path)
    if stack == "mortise":
        # The upstream client starts its workload clock before scheduling request 1.
        # Use request 1's recorded send as the origin, allowing for its initial delay.
        if not plan or 1 not in rows:
            return "missing anchor", plan
        origin_ns = int(rows[1]["client_send"]) - plan[0].scheduled_us * 1000
        observed = []
        for req in plan:
            row = rows.get(req.id)
            if row is None:
                observed.append(req)
                continue
            if int(row["size"]) != req.size:
                raise ValueError(f"{path}: request {req.id} differs from app.wk")
            observed.append(replace(
                req, status="completed",
                started_us=(int(row["client_send"]) - origin_ns) // 1000,
                completed_us=(int(row["client_recv"]) - origin_ns) // 1000,
            ))
        return "first send anchor", observed

    def timestamp(row: dict[str, str], field: str) -> int | None:
        return int(row[field]) if row[field] else None

    observed = []
    for req in plan:
        row = rows[req.id]
        if int(row["size"]) != req.size or int(row["scheduled_us"]) != req.scheduled_us:
            raise ValueError(f"{path}: request {req.id} differs from app.wk")
        observed.append(replace(
            req, status=row["status"], started_us=timestamp(row, "started_us"),
            first_byte_us=timestamp(row, "first_byte_us"), completed_us=timestamp(row, "completed_us"),
        ))
    return "replay clock", observed


def size_class(size: int) -> str:
    if size < 100 * 1024:
        return "<100 KiB"
    if size < 1024 * 1024:
        return "100 KiB–1 MiB"
    return "≥1 MiB"


def summarize(records: list[Timing], cutoff_us: int, source: str) -> dict:
    started = [r for r in records if r.scheduled_us <= cutoff_us]
    completed = sorted(r.rct_us for r in started if r.completed_us is not None and r.completed_us <= cutoff_us)
    n = len(records)
    stats = {"planned": n, "completed": len(completed), "completion_rate": len(completed) / n}
    valid = source in ("first send anchor", "replay clock")
    for seconds in DEADLINES_S:
        # An incomplete request is a miss. Report no rate if the common cutoff did not
        # observe all planned arrivals for the requested duration, unless all finished.
        horizon_us = seconds * 1_000_000
        covered = len(completed) == n or max(r.scheduled_us for r in records) + horizon_us <= cutoff_us
        stats[f"completed_within_{seconds}s"] = sum(t <= horizon_us for t in completed) / n if valid and covered else None
    for label, fraction in (("p50", 0.50), ("p90", 0.90), ("p95", 0.95)):
        index = math.ceil(n * fraction) - 1
        value = completed[index] if index < len(completed) else None
        covered = len(completed) == n or (
            value is not None and max(r.scheduled_us for r in records) + value <= cutoff_us
        )
        stats[f"rct_{label}_ms"] = value / 1000 if valid and covered and value is not None else None
    if valid and completed:
        censored = [
            r.rct_us if (r.completed_us is not None and r.completed_us <= cutoff_us) else max(0, cutoff_us - r.scheduled_us)
            for r in records
        ]
        stats["rct_mean_ms"] = (sum(censored) / n) / 1000.0
    else:
        stats["rct_mean_ms"] = None
    if not valid:
        stats["completed"] = stats["completion_rate"] = None
    return stats


def resolve_schedule_path(folder: Path, workload: str) -> Path:
    candidate = folder / f"{workload}.wk"
    if candidate.exists():
        return candidate
    return folder / "app.wk"


def median_or_none(vals: list[float | None]) -> float | None:
    clean = [v for v in vals if v is not None]
    return statistics.median(clean) if clean else None


def fmt_ms_agg(vals: list[float | None]) -> str:
    clean = [v for v in vals if v is not None]
    if not clean:
        return "–"
    med = statistics.median(clean)
    if len(clean) == 1:
        return f"{med:.1f}"
    return f"{med:.1f} ({min(clean):.0f}–{max(clean):.0f})"


def fmt_pct_agg(vals: list[float | None]) -> str:
    med = median_or_none(vals)
    return "–" if med is None else f"{med:.1%}"


def print_grouped_summary(summary: list[dict]) -> None:
    print("## Download experience from scheduled arrival")
    print(
        "\nRates include all planned requests. Mortise timing is aligned to request 1's send timestamp. "
        "Multiple repeats report median (min–max). A dash indicates missing data or an uncovered tail.\n"
    )
    by_run_class: dict[tuple, dict[str, dict]] = {}
    for row in summary:
        key = (
            row["workload"],
            row["trace"],
            row["queue_bytes"],
            row["stack"],
            row["cc"],
            row.get("repeat", "1"),
        )
        by_run_class.setdefault(key, {})[row["size_class"]] = row

    grouped: dict[tuple, list[dict[str, dict]]] = {}
    for (wk, trace, q, stack, cc, _rep), classes in sorted(by_run_class.items()):
        grouped.setdefault((wk, trace, q, stack, cc), []).append(classes)

    workloads = sorted({k[0] for k in grouped})
    for wk in workloads:
        print(f"\n### Workload: `{wk}`\n")
        print(
            "| Trace | Queue | Stack | CC | Done | ≤1 s | ≤5 s | Mean (ms) | P50 (ms) | P90 (ms) | P95 (ms) | <100 KiB P95 | 100 KiB–1 MiB P95 | ≥1 MiB P95 |"
        )
        print("|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for (row_wk, trace, q, stack, cc), runs in grouped.items():
            if row_wk != wk:
                continue
            all_rows = [r["all"] for r in runs if "all" in r]
            if not all_rows:
                continue
            planned = all_rows[0]["planned"]
            comp_vals = [r["completed"] for r in all_rows if r["completed"] is not None]
            done = "–" if not comp_vals else f"{int(round(statistics.median(comp_vals)))}/{planned}"
            r1 = fmt_pct_agg([r["completed_within_1s"] for r in all_rows])
            r5 = fmt_pct_agg([r["completed_within_5s"] for r in all_rows])
            mean_s = fmt_ms_agg([r["rct_mean_ms"] for r in all_rows])
            p50_s = fmt_ms_agg([r["rct_p50_ms"] for r in all_rows])
            p90_s = fmt_ms_agg([r["rct_p90_ms"] for r in all_rows])
            p95_s = fmt_ms_agg([r["rct_p95_ms"] for r in all_rows])
            s95 = fmt_ms_agg([r["<100 KiB"]["rct_p95_ms"] for r in runs if "<100 KiB" in r])
            m95 = fmt_ms_agg([r["100 KiB–1 MiB"]["rct_p95_ms"] for r in runs if "100 KiB–1 MiB" in r])
            l95 = fmt_ms_agg([r["≥1 MiB"]["rct_p95_ms"] for r in runs if "≥1 MiB" in r])
            print(
                "| "
                + " | ".join([trace, q, stack, cc, done, r1, r5, mean_s, p50_s, p90_s, p95_s, s95, m95, l95])
                + " |"
            )


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()
    requests = []
    summary = []
    for marker in sorted(args.input_dir.rglob("*.app-seconds")):
        folder = marker.parent
        stem = marker.stem
        meta = parse_meta_from_stem(stem)
        cutoff_us = int(marker.read_text()) * 1_000_000
        plan = load_schedule(resolve_schedule_path(folder, meta["workload"]))
        source, timings = load_timings(plan, folder, stem, meta["stack"])
        for r in timings:
            status = "after_cutoff" if r.completed_us is not None and r.completed_us > cutoff_us else r.status
            requests.append({
                **meta, "clock_source": source, "id": r.id, "size_bytes": r.size,
                "size_class": size_class(r.size), "status": status,
                "scheduled_us": r.scheduled_us, "started_us": r.started_us,
                "first_byte_us": r.first_byte_us, "completed_us": r.completed_us,
                "rct_us": r.rct_us, "ttfb_us": r.ttfb_us, "transfer_us": r.transfer_us,
            })
        for label in ("all", "<100 KiB", "100 KiB–1 MiB", "≥1 MiB"):
            subset = timings if label == "all" else [r for r in timings if size_class(r.size) == label]
            if subset:
                summary.append({
                    **meta, "clock_source": source, "cutoff_s": cutoff_us // 1_000_000,
                    "size_class": label, **summarize(subset, cutoff_us, source),
                })
    if not summary:
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "app_requests.csv", requests)
    write_csv(args.out_dir / "app_experience.csv", summary)
    if args.print_summary:
        print_grouped_summary(summary)


if __name__ == "__main__":
    main()
