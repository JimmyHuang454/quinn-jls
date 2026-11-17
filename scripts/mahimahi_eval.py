#!/usr/bin/env python3
"""Unified Mahimahi downlink-log + RCT CSV analyzer for quinn-jls and Mortise."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
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


def parse_downlink_log(path: Path, min_rtt_ms: float = 20.0) -> dict[str, Any]:
    """Parse Mahimahi mm-link --downlink-log into throughput, delay, loss, and power."""
    first_ts: float | None = None
    last_ts: float | None = None
    delivered_bytes = 0
    opportunity_bytes = 0
    dropped_bytes = 0
    qdelays: list[float] = []

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
            if kind == "#":
                opportunity_bytes += int(parts[2])
            elif kind == "-":
                delivered_bytes += int(parts[2])
                if len(parts) >= 4:
                    qdelays.append(float(parts[3]))
            elif kind == "d":
                # Format: `<ts> d <drops> <bytes>`
                dropped_bytes += int(parts[3]) if len(parts) >= 4 else int(parts[2])

    duration_s = (last_ts - first_ts) / 1000.0 if (first_ts is not None and last_ts is not None and last_ts > first_ts) else math.nan
    tput_mbps = (delivered_bytes * 8.0 / 1e6 / duration_s) if duration_s and duration_s > 0 else math.nan
    cap_mbps = (opportunity_bytes * 8.0 / 1e6 / duration_s) if duration_s and duration_s > 0 else math.nan
    util = (delivered_bytes / opportunity_bytes) if opportunity_bytes > 0 else math.nan

    qdelays.sort()
    qdelay_avg = (sum(qdelays) / len(qdelays)) if qdelays else math.nan
    qdelay_p95 = percentile(qdelays, 0.95)
    est_rtt_ms = (min_rtt_ms + qdelay_avg) if not math.isnan(qdelay_avg) else math.nan
    util_power = (util / (est_rtt_ms / 1000.0)) if (est_rtt_ms and est_rtt_ms > 0 and not math.isnan(util)) else math.nan
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

    # Last row in Mortise CSV is `0,0,0,<retrans_pkts>,0,0`
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

    # Mortise Eq. 6 normalized QoE: (R / RCT_s) * (1 - P(r)) in MB/s
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
    """Extract metadata from `<stack>__<workload>__<trace>__q<bytes>__<cc>`."""
    parts = stem.split("__")
    if len(parts) >= 5:
        return {
            "stack": parts[0],
            "workload": parts[1],
            "trace": parts[2],
            "queue_bytes": parts[3].lstrip("q"),
            "cc": "__".join(parts[4:]),
        }
    return {"stack": "unknown", "workload": "unknown", "trace": stem, "queue_bytes": "", "cc": ""}


def fmt_num(val: Any, digits: int = 2) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "-"
    if isinstance(val, int):
        return str(val)
    return f"{val:.{digits}f}"


def trace_group(trace: str) -> str:
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
    args = parser.parse_args()

    logs = sorted(args.input_dir.rglob("*.downlink.log"))
    records: list[dict[str, Any]] = []

    for log_path in logs:
        stem = log_path.name[: -len(".downlink.log")]
        meta = parse_meta_from_stem(stem)
        row: dict[str, Any] = {**meta}
        row.update(parse_downlink_log(log_path, min_rtt_ms=args.min_rtt_ms))

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
        "duration_s",
        "capacity_mbps",
        "throughput_mbps",
        "utilization",
        "qdelay_avg_ms",
        "qdelay_p95_ms",
        "est_rtt_ms",
        "util_power",
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

    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for r in records:
        gk = (
            r.get("workload", "-"),
            trace_group(r.get("trace", "")),
            r.get("queue_bytes", "-"),
            r.get("stack", "-"),
            r.get("cc", "-"),
        )
        groups.setdefault(gk, []).append(r)

    lines = [
        "# Mahimahi Comparison Summary (`quinn-jls` vs `BobAnkh/Mortise`)",
        "",
        "## Grouped Averages",
        "",
        "| Workload | Trace Group | Queue (B) | Stack | CC | N | Thr (Mbps) | Util | QDelay Avg (ms) | Power (Util/s) | Drop | Mean RCT (ms) | Mortise QoE |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for gk in sorted(groups.keys()):
        wk, grp, q, stk, cc = gk
        grows = groups[gk]
        lines.append(
            "| {wk} | {grp} | {q} | {stk} | {cc} | {n} | {thr} | {util} | {qd} | {pwr} | {drop} | {rct} | {qoe} |".format(
                wk=wk,
                grp=grp,
                q=q,
                stk=stk,
                cc=cc,
                n=len(grows),
                thr=fmt_num(mean_valid(grows, "throughput_mbps")),
                util=fmt_num(mean_valid(grows, "utilization"), 3),
                qd=fmt_num(mean_valid(grows, "qdelay_avg_ms")),
                pwr=fmt_num(mean_valid(grows, "util_power")),
                drop=fmt_num(mean_valid(grows, "drop_rate"), 4),
                rct=fmt_num(mean_valid(grows, "rct_mean_ms")),
                qoe=fmt_num(mean_valid(grows, "mortise_qoe")),
            )
        )

    if len(records) <= 80:
        lines.extend([
            "",
            "## Per-Run Details",
            "",
            "| Stack | Workload | Trace | Queue (B) | CC | Thr (Mbps) | Util | QDelay Avg (ms) | QDelay P95 (ms) | Power (Util/s) | Drop | Mean RCT (ms) | Mortise QoE |",
            "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for r in records:
            lines.append(
                "| {stack} | {workload} | {trace} | {q} | {cc} | {thr} | {util} | {qd} | {qdp95} | {pwr} | {drop} | {rct} | {qoe} |".format(
                    stack=r.get("stack", "-"),
                    workload=r.get("workload", "-"),
                    trace=r.get("trace", "-"),
                    q=r.get("queue_bytes", "-"),
                    cc=r.get("cc", "-"),
                    thr=fmt_num(r.get("throughput_mbps")),
                    util=fmt_num(r.get("utilization"), 3),
                    qd=fmt_num(r.get("qdelay_avg_ms")),
                    qdp95=fmt_num(r.get("qdelay_p95_ms")),
                    pwr=fmt_num(r.get("util_power")),
                    drop=fmt_num(r.get("drop_rate"), 4),
                    rct=fmt_num(r.get("rct_mean_ms")),
                    qoe=fmt_num(r.get("mortise_qoe")),
                )
            )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Analyzed {len(records)} runs -> {args.out_csv}, {args.out_md}")


if __name__ == "__main__":
    main()
