#!/usr/bin/env python3
"""Fetch Mahimahi/NYC/synthetic traces and run quinn-perf or Mortise suites in parallel."""

from __future__ import annotations

import argparse
import concurrent.futures
import queue
import random
import subprocess
import time
import urllib.request
from pathlib import Path

LTE_TRACES = [
    "ATT-LTE-driving-2016",
    "ATT-LTE-driving",
    "TMobile-LTE-driving",
    "Verizon-LTE-driving",
]

NYC_STATIONARY = [
    "trace-2925703-home1",
    "trace-3093662-home3",
    "trace-3219061-home",
    "trace-3457194-timessquare",
    "trace-3458374-timessquare",
]

NYC_WALKING = [
    "trace-3189663-timessquare",
    "trace-3199718-timessquare",
    "trace-3201037-timessquare",
    "trace-3201711-timessquare",
    "trace-3202253-timessquare",
    "trace-3203600-timessquare",
    "trace-3204803-timessquare",
    "trace-3205967-timessquare",
    "trace-3207521-timessquare",
    "trace-3208852-timessquare",
    "trace-3455408-timessquare",
    "trace-3453943-timessquare",
]

NYC_DRIVING = [
    "trace-3109898-bus",
    "trace-3114405-bus",
    "trace-3552192-bus",
    "trace-3555076-bus",
    "trace-2767958-taxi1",
    "trace-2768760-taxi3",
]

SYNTH_TRACES = ["synth-48mbps", "synth-step"]

SHARDS: dict[str, list[str]] = {
    "smoke": ["ATT-LTE-driving-2016"],
    "lte": LTE_TRACES,
    "nyc-stat": NYC_STATIONARY,
    "nyc-walk-1": NYC_WALKING[0:4],
    "nyc-walk-2": NYC_WALKING[4:8],
    "nyc-walk-3": NYC_WALKING[8:12],
    "nyc-drive-1": NYC_DRIVING[0:3],
    "nyc-drive-2": NYC_DRIVING[3:6],
    "synth": SYNTH_TRACES,
}


def download_file(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, dst)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def generate_rate_trace(segments: list[tuple[float, float, float]], dst: Path, seed: int = 42) -> None:
    """Generate Mahimahi packet-delivery-ms trace from (duration_s, mean_mbps, jitter_mbps) segments."""
    if dst.exists() and dst.stat().st_size > 0:
        return
    rng = random.Random(seed)
    lines: list[str] = []
    t_ms = 0.0
    for dur_s, mean_mbps, jitter_mbps in segments:
        end_ms = t_ms + dur_s * 1000.0
        # Update rate every 500 ms for realistic fluctuation
        while t_ms < end_ms:
            seg_end = min(t_ms + 500.0, end_ms)
            rate_mbps = max(1.0, mean_mbps + rng.uniform(-jitter_mbps, jitter_mbps))
            pkt_per_ms = rate_mbps * 1e6 / 8.0 / 1500.0 / 1000.0
            interval_ms = 1.0 / pkt_per_ms
            cur = t_ms
            while cur < seg_end:
                lines.append(str(int(cur)))
                cur += interval_ms
            t_ms = seg_end
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_trace_pair(trace_name: str, trace_dir: Path) -> tuple[Path, Path]:
    """Download/generate trace and return (up_trace, down_trace)."""
    if trace_name in LTE_TRACES:
        down = trace_dir / f"{trace_name}.down"
        raw_up = trace_dir / f"{trace_name}.up.raw"
        up = trace_dir / f"{trace_name}.up"
        base = "https://raw.githubusercontent.com/ravinet/mahimahi/master/traces"
        download_file(f"{base}/{trace_name}.down", down)
        download_file(f"{base}/{trace_name}.up", raw_up)
        if not up.exists():
            # Trim uplink to downlink duration so both wrap identically
            last_ts = int(down.read_text().split()[-1])
            with raw_up.open("r") as src, up.open("w") as dst:
                for line in src:
                    s = line.strip()
                    if s and int(s) <= last_ts:
                        dst.write(line)
        return up, down

    if trace_name in NYC_STATIONARY or trace_name in NYC_WALKING or trace_name in NYC_DRIVING:
        p = trace_dir / f"{trace_name}.trace"
        url = f"https://raw.githubusercontent.com/Soheil-ab/Cellular-Traces-NYC/master/{trace_name}"
        download_file(url, p)
        return p, p

    if trace_name == "synth-48mbps":
        p = trace_dir / "synth-48mbps.trace"
        generate_rate_trace([(60.0, 48.0, 0.0)], p, seed=42)
        return p, p

    if trace_name == "synth-step":
        p = trace_dir / "synth-step.trace"
        # Step-up at 20s and step-down at 45s (inspired by Mortise Fig. 9)
        generate_rate_trace([(20.0, 7.5, 1.5), (25.0, 13.0, 3.0), (15.0, 6.0, 1.5)], p, seed=42)
        return p, p

    raise ValueError(f"Unknown trace: {trace_name}")


def run_quinn_job(
    slot: int,
    bin_path: Path,
    cc: str,
    trace_name: str,
    up_trace: Path,
    down_trace: Path,
    q_bytes: str,
    workload_kind: str,
    workload_path: Path,
    out_dir: Path,
    bulk_secs: int,
) -> tuple[str, int]:
    port = 4433 + slot
    stem = f"quinn__{workload_kind}__{trace_name}__q{q_bytes}__{cc}"
    log_path = out_dir / f"{stem}.downlink.log"
    sum_path = out_dir / f"{stem}.summary.json"
    srv_log = out_dir / f"{stem}.server.log"

    srv = subprocess.Popen(
        [str(bin_path), "server", "--listen", f"0.0.0.0:{port}", "--congestion", cc],
        stdout=srv_log.open("w"),
        stderr=subprocess.STDOUT,
    )
    try:
        time.sleep(0.5)
        if workload_kind == "bulk":
            client_cmd = (
                f'timeout {bulk_secs + 60} "{bin_path}" client "${{MAHIMAHI_BASE}}:{port}" '
                f'--ip "${{MAHIMAHI_BASE}}" --congestion "{cc}" '
                f'--bi-requests 1 --download-size 2M --duration {bulk_secs} '
                f'--summary-json "{sum_path}" --json "{out_dir / (stem + ".iperf.json")}"'
            )
        else:
            rct_path = out_dir / f"{stem}.rct.csv"
            client_cmd = (
                f'timeout 220 "{bin_path}" client "${{MAHIMAHI_BASE}}:{port}" '
                f'--ip "${{MAHIMAHI_BASE}}" --congestion "{cc}" '
                f'--workload "{workload_path}" --duration 200 '
                f'--rct-csv "{rct_path}" --summary-json "{sum_path}"'
            )

        mm_cmd = [
            "mm-delay",
            "10",
            "mm-link",
            str(up_trace),
            str(down_trace),
            "--uplink-queue=droptail",
            f"--uplink-queue-args=bytes={q_bytes}",
            "--downlink-queue=droptail",
            f"--downlink-queue-args=bytes={q_bytes}",
            f"--downlink-log={log_path}",
            "--",
            "bash",
            "-c",
            client_cmd,
        ]
        res = subprocess.run(mm_cmd, capture_output=True, text=True)
        if res.returncode != 0:
            (out_dir / f"{stem}.err.log").write_text(res.stdout + "\n" + res.stderr, encoding="utf-8")
        return stem, res.returncode
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=5)
        except subprocess.TimeoutExpired:
            srv.kill()


def run_mortise_job(
    bin_path: Path,
    cc: str,
    trace_name: str,
    up_trace: Path,
    down_trace: Path,
    q_bytes: str,
    workload_kind: str,
    workload_path: Path,
    out_dir: Path,
) -> tuple[str, int]:
    stem = f"mortise__{workload_kind}__{trace_name}__q{q_bytes}__{cc}"
    log_path = out_dir / f"{stem}.downlink.log"
    rct_path = out_dir / f"{stem}.rct.csv"

    timeout_s = 150 if workload_kind == "bulk" else 220
    client_cmd = (
        f'timeout {timeout_s} "{bin_path}" --connect "${{MAHIMAHI_BASE}}" '
        f'--congestion "{cc}" --workload "{workload_path}" --output "{rct_path}"'
    )
    mm_cmd = [
        "mm-delay",
        "10",
        "mm-link",
        str(up_trace),
        str(down_trace),
        "--uplink-queue=droptail",
        f"--uplink-queue-args=bytes={q_bytes}",
        "--downlink-queue=droptail",
        f"--downlink-queue-args=bytes={q_bytes}",
        f"--downlink-log={log_path}",
        "--",
        "bash",
        "-c",
        client_cmd,
    ]
    res = subprocess.run(mm_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        (out_dir / f"{stem}.err.log").write_text(res.stdout + "\n" + res.stderr, encoding="utf-8")
    return stem, res.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Mahimahi trace suite for quinn-jls or Mortise.")
    parser.add_argument("--stack", required=True, choices=["quinn", "mortise"])
    parser.add_argument("--shard", required=True, choices=list(SHARDS.keys()))
    parser.add_argument("--ccs", required=True, help="Comma-separated congestion algorithms")
    parser.add_argument("--queues", default="150000,2000000", help="Comma-separated queue sizes in bytes")
    parser.add_argument("--workloads", default="bulk,app", help="Comma-separated workloads (bulk,app)")
    parser.add_argument("--bulk-secs", type=int, default=45)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--bin", required=True, type=Path, help="Path to quinn-perf or Mortise client binary")
    parser.add_argument("--trace-dir", required=True, type=Path)
    parser.add_argument("--app-workload", required=True, type=Path)
    parser.add_argument("--bulk-workload", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    traces = SHARDS[args.shard]
    ccs = [c.strip() for c in args.ccs.split(",") if c.strip()]
    queues = [q.strip() for q in args.queues.split(",") if q.strip()]
    workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]

    trace_pairs = {t: prepare_trace_pair(t, args.trace_dir) for t in traces}

    tasks = []
    for cc in ccs:
        for t in traces:
            up_trace, down_trace = trace_pairs[t]
            for q in queues:
                for wk in workloads:
                    wk_path = args.bulk_workload if wk == "bulk" else args.app_workload
                    tasks.append((cc, t, up_trace, down_trace, q, wk, wk_path))

    print(f"[{args.stack}] shard={args.shard}: {len(tasks)} runs across {len(traces)} traces (parallel={args.parallel})")

    slot_pool: queue.Queue[int] = queue.Queue()
    for s in range(args.parallel):
        slot_pool.put(s)

    def dispatch(item):
        cc, t, up_tr, down_tr, q, wk, wk_path = item
        slot = slot_pool.get()
        try:
            if args.stack == "quinn":
                return run_quinn_job(slot, args.bin, cc, t, up_tr, down_tr, q, wk, wk_path, args.out_dir, args.bulk_secs)
            return run_mortise_job(args.bin, cc, t, up_tr, down_tr, q, wk, wk_path, args.out_dir)
        finally:
            slot_pool.put(slot)

    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for stem, rc in pool.map(dispatch, tasks):
            status = "OK" if rc == 0 else f"WARN(rc={rc})"
            if rc != 0:
                failures += 1
            print(f"  {status}: {stem}", flush=True)

    print(f"[{args.stack}] shard={args.shard} finished: {len(tasks) - failures}/{len(tasks)} succeeded.")


if __name__ == "__main__":
    main()
