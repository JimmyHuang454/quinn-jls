#!/usr/bin/env python3
"""Unified Mahimahi downlink-log + RCT CSV analyzer for quinn-jls and Mortise."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def mortise_retrans_penalty(r: float) -> float:
    """Piecewise retransmission penalty P(r) from Mortise (NSDI '26, Eq. 6)."""
    if r < 0.05:
        return 0.0
    if r < 0.1:
        return 4.0 * r - 0.2
    if r < 0.4:
        return r + 0.1
    return 0.5


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return math.nan
    k = (len(sorted_vals) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def parse_downlink_log(
    path: Path,
    min_rtt_ms: float = 20.0,
    warmup_s: float = 0.0,
    max_s: float | None = None,
) -> dict[str, Any]:
    """Parse Mahimahi mm-link --downlink-log over [warmup_s, max_s] relative to first timestamp."""
    entries: list[tuple[float, str, int, float]] = []
    first_ts: float | None = None
    last_ts: float | None = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                ts = float(parts[0])
            except ValueError:
                continue

            if first_ts is None:
                first_ts = ts
            last_ts = ts

            kind = parts[1]
            if kind == "#" or kind == "-":
                nbytes = int(parts[2])
                qdelay = float(parts[3]) if (kind == "-" and len(parts) >= 4) else math.nan
                entries.append((ts, kind, nbytes, qdelay))
            elif kind == "d":
                nbytes = int(parts[3]) if len(parts) >= 4 else int(parts[2])
                entries.append((ts, kind, nbytes, math.nan))

    if first_ts is None or last_ts is None or last_ts <= first_ts:
        return {
            "duration_s": math.nan,
            "delivered_bytes": 0,
            "capacity_mbps": math.nan,
            "throughput_mbps": math.nan,
            "utilization": math.nan,
            "qdelay_avg_ms": math.nan,
            "qdelay_p95_ms": math.nan,
            "est_rtt_ms": math.nan,
            "util_power": math.nan,
            "norm_power": math.nan,
            "tput_power": math.nan,
            "drop_rate": math.nan,
        }

    total_span_s = (last_ts - first_ts) / 1000.0
    # If the run finished too early (e.g., short legacy trace), skip warmup filter so metrics stay valid.
    eff_warmup_s = warmup_s if total_span_s > warmup_s + 2.0 else 0.0
    win_start_ts = first_ts + eff_warmup_s * 1000.0
    win_end_ts = (first_ts + max_s * 1000.0) if (max_s is not None and total_span_s > eff_warmup_s + 2.0) else last_ts

    delivered_bytes = 0
    opportunity_bytes = 0
    dropped_bytes = 0
    qdelays: list[float] = []
    act_start: float | None = None
    act_end: float | None = None

    for ts, kind, nbytes, qdelay in entries:
        if ts < win_start_ts or ts > win_end_ts:
            continue
        if act_start is None:
            act_start = ts
        act_end = ts
        if kind == "#":
            opportunity_bytes += nbytes
        elif kind == "-":
            delivered_bytes += nbytes
            if not math.isnan(qdelay):
                qdelays.append(qdelay)
        elif kind == "d":
            dropped_bytes += nbytes

    duration_s = (act_end - act_start) / 1000.0 if (act_start is not None and act_end is not None and act_end > act_start) else math.nan
    tput_mbps = (delivered_bytes * 8.0 / 1e6 / duration_s) if duration_s and duration_s > 0 else math.nan
    cap_mbps = (opportunity_bytes * 8.0 / 1e6 / duration_s) if duration_s and duration_s > 0 else math.nan
    util = (delivered_bytes / opportunity_bytes) if opportunity_bytes > 0 else math.nan

    qdelays.sort()
    qdelay_avg = (sum(qdelays) / len(qdelays)) if qdelays else math.nan
    qdelay_p95 = percentile(qdelays, 0.95)
    est_rtt_ms = (min_rtt_ms + qdelay_avg) if not math.isnan(qdelay_avg) else math.nan
    util_power = (util / (est_rtt_ms / 1000.0)) if (est_rtt_ms and est_rtt_ms > 0 and not math.isnan(util)) else math.nan
    norm_power = (util / (est_rtt_ms / min_rtt_ms)) if (est_rtt_ms and est_rtt_ms > 0 and not math.isnan(util)) else math.nan
    tput_power = (tput_mbps / est_rtt_ms) if (est_rtt_ms and est_rtt_ms > 0 and not math.isnan(tput_mbps)) else math.nan

    total_offered = delivered_bytes + dropped_bytes
    drop_rate = (dropped_bytes / total_offered) if total_offered > 0 else 0.0

    return {
        "duration_s": duration_s,
        "delivered_bytes": delivered_bytes,
        "capacity_mbps": cap_mbps,
        "throughput_mbps": tput_mbps,
        "utilization": util,
        "qdelay_avg_ms": qdelay_avg,
        "qdelay_p95_ms": qdelay_p95,
        "est_rtt_ms": est_rtt_ms,
        "util_power": util_power,
        "norm_power": norm_power,
        "tput_power": tput_power,
        "drop_rate": drop_rate,
    }


def parse_rct_csv(path: Path, delivered_bytes: int = 0) -> dict[str, Any]:
    """Parse Mortise-compatible RCT CSV (`id,size,client_send,server_recv,client_recv,rct`)."""
    if not path.exists():
        return {}
    sizes: list[int] = []
    rcts_ms: list[float] = []
    retrans_pkts = 0

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return {}
        rows = list(reader)

    if not rows:
        return {}

    data_rows = rows
    if len(rows[-1]) >= 4 and rows[-1][0] == "0":
        try:
            retrans_pkts = int(float(rows[-1][3]))
        except ValueError:
            retrans_pkts = 0
        data_rows = rows[:-1]

    for row in data_rows:
        if len(row) < 6:
            continue
        try:
            sz = int(float(row[1]))
            rct = max(float(row[5]), 1.0)
        except ValueError:
            continue
        sizes.append(sz)
        rcts_ms.append(rct)

    if not rcts_ms:
        return {"retrans_pkts": retrans_pkts}

    sorted_rcts = sorted(rcts_ms)
    total_pkts = max(delivered_bytes / 1448.0, sum(sizes) / 1448.0, 1.0)
    retrans_ratio = retrans_pkts / total_pkts
    penalty = mortise_retrans_penalty(retrans_ratio)
    qoe_vals = [((sz / 1e6) / (rct / 1000.0)) * (1.0 - penalty) for sz, rct in zip(sizes, rcts_ms)]

    return {
        "requests": len(rcts_ms),
        "rct_mean_ms": sum(rcts_ms) / len(rcts_ms),
        "rct_p50_ms": percentile(sorted_rcts, 0.50),
        "rct_p95_ms": percentile(sorted_rcts, 0.95),
        "retrans_pkts": retrans_pkts,
        "retrans_ratio": retrans_ratio,
        "mortise_qoe": sum(qoe_vals) / len(qoe_vals),
    }


def parse_meta_from_stem(stem: str) -> dict[str, str]:
    """Extract metadata from `<stack>__<workload>__<trace>__q<bytes>__<cc>[__r<k>]`."""
    parts = stem.split("__")
    if len(parts) >= 5:
        repeat = "1"
        tail = parts[4:]
        if len(tail) >= 2 and tail[-1].startswith("r") and tail[-1][1:].isdigit():
            repeat = tail[-1][1:]
            tail = tail[:-1]
        return {
            "stack": parts[0],
            "workload": parts[1],
            "trace": parts[2],
            "queue_bytes": parts[3].lstrip("q"),
            "cc": "__".join(tail),
            "repeat": repeat,
        }
    return {
        "stack": "unknown",
        "workload": "unknown",
        "trace": stem,
        "queue_bytes": "",
        "cc": "",
        "repeat": "1",
    }


def fmt_num(val: Any, digits: int = 2) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "-"
    if isinstance(val, int):
        return str(val)
    return f"{val:.{digits}f}"


def trace_group(trace: str) -> str:
    if trace.startswith("leo-starlink-"):
        return "Starlink-" + trace.rsplit("-", 1)[-1]
    if trace.startswith("nyc2018-4g-subway-"):
        return trace
    if trace.startswith("5g_trace_"):
        return "Lumos5G-" + trace.rsplit("_", 1)[-1]
    if "LTE-driving" in trace:
        return "LTE-driving"
    if trace.endswith("-home1") or trace.endswith("-home3") or trace.endswith("-home") or trace in {
        "trace-3457194-timessquare",
        "trace-3458374-timessquare",
    }:
        return "NYC-stationary"
    if trace.endswith("-bus") or trace.endswith("-taxi1") or trace.endswith("-taxi3"):
        return "NYC-driving"
    if trace.startswith("trace-"):
        return "NYC-walking"
    if trace.startswith("synth-"):
        return trace
    return "other"


def mean_valid(rows: list[dict[str, Any]], key: str) -> float:
    vals = [r[key] for r in rows if key in r and isinstance(r[key], (int, float)) and not math.isnan(r[key])]
    return (sum(vals) / len(vals)) if vals else math.nan


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Mahimahi logs and RCT CSVs.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory with *.downlink.log and *.rct.csv")
    parser.add_argument("--out-csv", required=True, type=Path, help="Output summary CSV")
    parser.add_argument("--out-md", required=True, type=Path, help="Output summary Markdown")
    parser.add_argument("--min-rtt-ms", type=float, default=20.0, help="Base propagation RTT in ms")
    parser.add_argument("--bulk-warmup-s", type=float, default=5.0, help="Warmup seconds skipped for bulk runs")
    parser.add_argument("--bulk-end-s", type=float, default=None, help="Optional end timestamp (s) for bulk window")
    args = parser.parse_args()

    logs = sorted(args.input_dir.rglob("*.downlink.log"))
    records: list[dict[str, Any]] = []

    for log_path in logs:
        stem = log_path.name[: -len(".downlink.log")]
        meta = parse_meta_from_stem(stem)
        is_bulk = meta["workload"] == "bulk"
        warmup = args.bulk_warmup_s if is_bulk else 0.0
        max_s = args.bulk_end_s if is_bulk else None

        rtt_file = log_path.with_name(f"{meta['trace']}.base-rtt-ms")
        base_rtt_ms = float(rtt_file.read_text()) if rtt_file.exists() else args.min_rtt_ms
        row: dict[str, Any] = {**meta, "base_rtt_ms": base_rtt_ms}
        row.update(parse_downlink_log(log_path, min_rtt_ms=base_rtt_ms, warmup_s=warmup, max_s=max_s))

        rct_path = log_path.with_name(f"{stem}.rct.csv")
        row.update(parse_rct_csv(rct_path, delivered_bytes=row.get("delivered_bytes", 0)))

        sum_path = log_path.with_name(f"{stem}.summary.json")
        if sum_path.exists():
            with sum_path.open("r", encoding="utf-8") as sf:
                sdata = json.load(sf)
            if sdata.get("sent_packets", 0) > 0:
                row["quic_loss_ratio"] = sdata["lost_packets"] / sdata["sent_packets"]
                row["quic_rtt_ms"] = sdata.get("rtt_ms")

        records.append(row)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "stack",
        "workload",
        "trace",
        "queue_bytes",
        "cc",
        "repeat",
        "duration_s",
        "capacity_mbps",
        "throughput_mbps",
        "utilization",
        "qdelay_avg_ms",
        "qdelay_p95_ms",
        "base_rtt_ms",
        "est_rtt_ms",
        "util_power",
        "norm_power",
        "tput_power",
        "drop_rate",
        "requests",
        "rct_mean_ms",
        "rct_p50_ms",
        "rct_p95_ms",
        "retrans_pkts",
        "retrans_ratio",
        "mortise_qoe",
        "quic_loss_ratio",
        "quic_rtt_ms",
    ]
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    bulk_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    app_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for r in records:
        gk = (
            trace_group(r.get("trace", "")),
            r.get("queue_bytes", "-"),
            r.get("stack", "-"),
            r.get("cc", "-"),
        )
        if r.get("workload") == "bulk":
            bulk_groups.setdefault(gk, []).append(r)
        else:
            app_groups.setdefault(gk, []).append(r)

    lines = [
        "# Mahimahi Comparison Summary (`quinn-jls` vs `BobAnkh/Mortise`)",
        "",
        f"Bulk metrics skip the first `{args.bulk_warmup_s:.1f}s` warmup window.",
        "- Note: `copa-auto-nc-theta*` runs **Copa default mode only (`默认模式`, `mode_switching=false`)** with `--copa-tradeoff=<theta>`; `copa-auto-theta*` enables competitive mode (`mode_switching=true`).",
        "",
        "## 1. Bulk Saturated Flows (Kleinrock Power & Delay-Utilization Tradeoff)",
        "",
        "| Trace Group | Queue (B) | Stack | CC | N | Cap (Mbps) | Thr (Mbps) | Util | QDelay Avg (ms) | QDelay P95 (ms) | Power (Util/s) | Norm Power | Drop |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for gk in sorted(bulk_groups.keys()):
        grp, q, stk, cc = gk
        grows = bulk_groups[gk]
        lines.append(
            "| {grp} | {q} | {stk} | {cc} | {n} | {cap} | {thr} | {util} | {qd} | {qdp95} | {pwr} | {npwr} | {drop} |".format(
                grp=grp,
                q=q,
                stk=stk,
                cc=cc,
                n=len(grows),
                cap=fmt_num(mean_valid(grows, "capacity_mbps")),
                thr=fmt_num(mean_valid(grows, "throughput_mbps")),
                util=fmt_num(mean_valid(grows, "utilization"), 3),
                qd=fmt_num(mean_valid(grows, "qdelay_avg_ms")),
                qdp95=fmt_num(mean_valid(grows, "qdelay_p95_ms")),
                pwr=fmt_num(mean_valid(grows, "util_power")),
                npwr=fmt_num(mean_valid(grows, "norm_power"), 3),
                drop=fmt_num(mean_valid(grows, "drop_rate"), 4),
            )
        )

    lines.extend([
        "",
        "## 2. File Download Workload (`app.wk`: RCT & Mortise QoE)",
        "",
        "| Trace Group | Queue (B) | Stack | CC | N | Thr (Mbps) | QDelay Avg (ms) | Drop | Mean RCT (ms) | P95 RCT (ms) | Mortise QoE |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for gk in sorted(app_groups.keys()):
        grp, q, stk, cc = gk
        grows = app_groups[gk]
        lines.append(
            "| {grp} | {q} | {stk} | {cc} | {n} | {thr} | {qd} | {drop} | {rct} | {rct95} | {qoe} |".format(
                grp=grp,
                q=q,
                stk=stk,
                cc=cc,
                n=len(grows),
                thr=fmt_num(mean_valid(grows, "throughput_mbps")),
                qd=fmt_num(mean_valid(grows, "qdelay_avg_ms")),
                drop=fmt_num(mean_valid(grows, "drop_rate"), 4),
                rct=fmt_num(mean_valid(grows, "rct_mean_ms")),
                rct95=fmt_num(mean_valid(grows, "rct_p95_ms")),
                qoe=fmt_num(mean_valid(grows, "mortise_qoe")),
            )
        )

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Analyzed {len(records)} runs -> {args.out_csv}, {args.out_md}")


if __name__ == "__main__":
    main()
