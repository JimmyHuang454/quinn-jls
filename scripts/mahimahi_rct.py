#!/usr/bin/env python3
"""Compare download experience from the planned request schedule, including unfinished requests."""

import argparse
import csv
import math
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
    for label, fraction in (("p50", 0.50), ("p95", 0.95)):
        index = math.ceil(n * fraction) - 1
        value = completed[index] if index < len(completed) else None
        covered = len(completed) == n or (
            value is not None and max(r.scheduled_us for r in records) + value <= cutoff_us
        )
        stats[f"rct_{label}_ms"] = value / 1000 if valid and covered and value is not None else None
    if not valid:
        stats["completed"] = stats["completion_rate"] = None
    return stats


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
        plan = load_schedule(folder / "app.wk")
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
        print("## Download experience from scheduled arrival")
        print("\nRates include all planned requests. Mortise timing is aligned to request 1's send timestamp. A dash indicates missing data or a percentile/deadline the observation window cannot establish.\n")
        print("| Stack | CC | Size | Completed | ≤1 s | ≤5 s | ≤15 s | ≤30 s | P50 (s) | P95 (s) |")
        print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in summary:
            completed = "–" if row["completed"] is None else f'{row["completed"]}/{row["planned"]}'
            values = [row[f"completed_within_{s}s"] for s in DEADLINES_S]
            rates = ["–" if value is None else f"{value:.1%}" for value in values]
            times = ["–" if row[k] is None else f"{row[k] / 1000:.3f}" for k in ("rct_p50_ms", "rct_p95_ms")]
            print("| " + " | ".join([row["stack"], row["cc"], row["size_class"], completed, *rates, *times]) + " |")


if __name__ == "__main__":
    main()
