#!/usr/bin/env python3
"""Fetch cellular, Starlink, and synthetic traces and run Quinn or Mortise in parallel."""

from __future__ import annotations

import argparse
import concurrent.futures
import queue
import random
import re
import subprocess
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# Serialize mm-delay/mm-link startup to avoid Mahimahi 100.64.x address allocation races.
MM_LAUNCH_LOCK = threading.Lock()

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
NYC_SUBWAY = ["nyc2018-4g-subway-no-cross", "nyc2018-4g-subway-with-cross"]
LUMOS_5G = ["5g_trace_100_walking", "5g_trace_103_driving"]
LUMOS_REV = "35aaf2857550c7f4c736fe9b01e894253a777e4e"
LEO_TRACES = ["leo-starlink-example-fixed", "leo-starlink-example-dynamic"]
LEO_REV = "69df6406e9985a764b4d52b506cd6ee77596ffb2"

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
    "nyc-subway": NYC_SUBWAY,
    "lumos-5g": LUMOS_5G,
    "leo": LEO_TRACES,
}


@dataclass(frozen=True)
class Replay:
    up: Path
    down: Path
    delay: Path | None = None
    base_rtt_ms: float = 20.0


def prepare_replay(trace_name: str, trace_dir: Path, bulk_secs: int) -> Replay:
    if trace_name in LEO_TRACES:
        base = f"https://raw.githubusercontent.com/SpaceNetLab/LeoCC/{LEO_REV}/leoreplayer/replayer/example/Cubic"
        capacity = trace_dir / "leo-bw.txt"
        delay = trace_dir / "leo-delay.txt"
        download_file(f"{base}/bw_example.txt", capacity)
        download_file(f"{base}/delay_example.txt", delay)
        samples = [int(line) for line in delay.read_text().splitlines()]
        # Match the mean base RTT over the same nominal bulk measurement window.
        window = samples[500:bulk_secs * 100] or samples
        mean_delay = round(sum(window) / len(window))
        if trace_name.endswith("-fixed"):
            delay = trace_dir / "leo-delay-fixed.txt"
            delay.write_text(f"{mean_delay}\n", encoding="utf-8")
        # The upstream example uses the same delivery opportunities in both directions.
        return Replay(capacity, capacity, delay, 2.0 * mean_delay)
    return Replay(*prepare_trace_pair(trace_name, trace_dir))


def replay_command(replay: Replay, queue_bytes: str, log: Path, command: str) -> list[str]:
    delay = ["mm-delay", "10"]
    if replay.delay is not None:
        delay.append(str(replay.delay))
        delay.extend(["mm-loss", "uplink", "0.002", "mm-loss", "downlink", "0.0001"])
    return delay + [
        "mm-link", str(replay.up), str(replay.down),
        "--uplink-queue=droptail", f"--uplink-queue-args=bytes={queue_bytes}",
        "--downlink-queue=droptail", f"--downlink-queue-args=bytes={queue_bytes}",
        f"--downlink-log={log}", "--", "bash", "-c", command,
    ]


def download_file(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            partial = dst.with_suffix(dst.suffix + ".part")
            urllib.request.urlretrieve(url, partial)
            partial.replace(dst)
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

    if trace_name in NYC_SUBWAY:
        cross = "no-cross" if "-no-cross" in trace_name else "with-cross"
        base = "https://raw.githubusercontent.com/Soheil-ab/Cellular-Traces-2018/master"
        up = trace_dir / f"{trace_name}.up"
        down = trace_dir / f"{trace_name}.down"
        download_file(f"{base}/uplink-4g-{cross}-subway.pps", up)
        download_file(f"{base}/downlink-4g-{cross}-subway.pps", down)
        return up, down

    if trace_name in LUMOS_5G:
        directory = "Video-Streaming/Network-Traces/Lumous5G/5G"
        base = f"https://raw.githubusercontent.com/SIGCOMM21-5G/artifact/{LUMOS_REV}/{directory}"
        raw = trace_dir / f"{trace_name}.mbps"
        down = trace_dir / f"{trace_name}.down"
        up = trace_dir / "lumos-fixed-ack-100mbps.up"
        download_file(f"{base}/{trace_name}", raw)
        if not down.exists():
            # Convert cumulative byte budgets so fractional packets carry over between samples.
            last_time = 0.0
            packet_budget = 0.0
            partial = down.with_suffix(".part")
            with partial.open("w", encoding="utf-8") as out:
                for line in raw.read_text().splitlines():
                    if not line.strip():
                        continue
                    end_time, mbps = map(float, line.split())
                    interval = end_time - last_time
                    if interval < 0 or mbps < 0:
                        raise ValueError(f"Invalid Lumos5G sample: {line}")
                    next_budget = packet_budget + mbps * 1e6 * interval / 12000
                    for packet in range(int(packet_budget) + 1, int(next_budget) + 1):
                        timestamp = last_time + (packet - packet_budget) * interval / (next_budget - packet_budget)
                        out.write(f"{int(timestamp * 1000)}\n")
                    last_time, packet_budget = end_time, next_budget
            partial.replace(down)
        generate_rate_trace([(0.06, 100.0, 0.0)], up)
        return up, down

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


def run_mm_cmd(
    mm_cmd: list[str],
    expected_out: Path,
    link_log: Path,
    allow_timeout: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Launch mm-delay/mm-link with staggered startup and 1 retry on transient setup failure."""
    for attempt in range(2):
        with MM_LAUNCH_LOCK:
            proc = subprocess.Popen(mm_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            time.sleep(1.5)
        stdout, stderr = proc.communicate()
        res = subprocess.CompletedProcess(mm_cmd, proc.returncode, stdout, stderr)
        ok_code = res.returncode == 0 or (allow_timeout and res.returncode in (124, 130))
        if ok_code and expected_out.exists() and expected_out.stat().st_size > 0:
            return res
        # Once packet service started, a retry would rerun the experiment and could overwrite results.
        if link_log.exists() and link_log.stat().st_size > 0:
            break
        if attempt == 0:
            time.sleep(2.0)
    return res


def record_failure(path: Path, result: subprocess.CompletedProcess[str]) -> None:
    output = result.stdout + "\n" + result.stderr
    path.write_text(output, encoding="utf-8")
    if output.strip():
        detail = " | ".join(output.strip().splitlines()[-5:])[-1200:]
        # GitHub exposes annotations alongside the job status and preserves the full log artifact.
        detail = detail.replace("%", "%25").replace("\r", "%0D")
        print(f"::error::{path.stem} (exit {result.returncode}): {detail}", flush=True)
    else:
        print(f"::error::{path.stem} (exit {result.returncode}): required output missing or empty", flush=True)


def has_deliveries(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open(encoding="utf-8") as log:
        return any(line.split(maxsplit=2)[1:2] == ["-"] for line in log if not line.startswith("#"))


def parse_cc_flags(cc: str) -> tuple[str, list[str]]:
    """Map suite label like `copa-auto-theta0.98` or `copa-auto-nc-t70` to CLI flags."""
    m = re.match(r"^(copa-auto(?:-nc)?)-theta([0-9]*\.?[0-9]+)$", cc)
    if m:
        alg = m.group(1)
        theta = float(m.group(2))
        if not (0.01 <= theta <= 0.99):
            raise ValueError(f"theta out of [0.01, 0.99] in {cc!r}")
        return alg, ["--copa-tradeoff", f"{theta:g}"]
    m = re.match(r"^(copa-auto(?:-nc)?)-t(\d+)$", cc)
    if m:
        alg = m.group(1)
        theta = int(m.group(2)) / 100.0
        if not (0.01 <= theta <= 0.99):
            raise ValueError(f"theta out of [0.01, 0.99] in {cc!r}")
        return alg, ["--copa-tradeoff", f"{theta:.2f}"]
    return cc, []


def run_quinn_job(
    slot: int,
    bin_path: Path,
    cc: str,
    trace_name: str,
    replay: Replay,
    q_bytes: str,
    workload_kind: str,
    workload_path: Path,
    out_dir: Path,
    bulk_secs: int,
    send_fairness: str,
    stream_receive_window: str | None,
    connection_window: str | None,
) -> tuple[str, int]:
    port = 4433 + slot
    stem = f"quinn__{workload_kind}__{trace_name}__q{q_bytes}__{cc}"
    log_path = out_dir / f"{stem}.downlink.log"
    sum_path = out_dir / f"{stem}.summary.json"
    srv_log = out_dir / f"{stem}.server.log"
    alg, extra_cc_flags = parse_cc_flags(cc)

    srv_cmd = [
        str(bin_path),
        "server",
        "--listen",
        f"0.0.0.0:{port}",
        "--congestion",
        alg,
        *extra_cc_flags,
        "--initial-mtu",
        "1450",
        "--send-fairness",
        send_fairness,
    ]
    if stream_receive_window:
        srv_cmd.extend(["--stream-receive-window", stream_receive_window])
    if connection_window:
        srv_cmd.extend(["--receive-window", connection_window, "--send-window", connection_window])

    srv = subprocess.Popen(
        srv_cmd,
        stdout=srv_log.open("w"),
        stderr=subprocess.STDOUT,
    )
    try:
        time.sleep(0.8)
        extra_str = " ".join(extra_cc_flags)
        srw_str = f"--stream-receive-window {stream_receive_window}" if stream_receive_window else ""
        cw_str = f"--receive-window {connection_window} --send-window {connection_window}" if connection_window else ""
        common_cli = (
            f'--ip "${{MAHIMAHI_BASE}}" --congestion "{alg}" {extra_str} --initial-mtu 1450 '
            f'--send-fairness {send_fairness} {srw_str} {cw_str}'
        )
        if workload_kind == "bulk":
            download_size = "16G" if replay.delay is not None or trace_name in LUMOS_5G else "500M"
            client_cmd = (
                f'timeout {bulk_secs + 60} "{bin_path}" client "${{MAHIMAHI_BASE}}:{port}" '
                f'{common_cli} '
                f'--bi-requests 1 --upload-size 0 --download-size {download_size} --duration {bulk_secs} '
                f'--summary-json "{sum_path}" --json "{out_dir / (stem + ".iperf.json")}"'
            )
        else:
            rct_path = out_dir / f"{stem}.rct.csv"
            client_cmd = (
                f'timeout 220 "{bin_path}" client "${{MAHIMAHI_BASE}}:{port}" '
                f'{common_cli} '
                f'--workload "{workload_path}" --duration 200 '
                f'--rct-csv "{rct_path}" --summary-json "{sum_path}"'
            )

        res = run_mm_cmd(replay_command(replay, q_bytes, log_path, client_cmd), sum_path, log_path)
        if res.returncode != 0 or not sum_path.exists() or not has_deliveries(log_path):
            record_failure(out_dir / f"{stem}.err.log", res)
            return stem, res.returncode or 1
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
    replay: Replay,
    q_bytes: str,
    workload_kind: str,
    workload_path: Path,
    out_dir: Path,
    bulk_secs: int,
) -> tuple[str, int]:
    stem = f"mortise__{workload_kind}__{trace_name}__q{q_bytes}__{cc}"
    log_path = out_dir / f"{stem}.downlink.log"
    rct_path = out_dir / f"{stem}.rct.csv"

    if workload_kind == "bulk":
        # Bulk only needs the link log. Terminate at the deadline without waiting for an RCT report.
        client_cmd = (
            f'timeout -k 5 {bulk_secs} "{bin_path}" --connect "${{MAHIMAHI_BASE}}" '
            f'--congestion "{cc}" --workload "{workload_path}" --output "{rct_path}"; '
            'status=$?; if [ "$status" -eq 124 ]; then exit 0; fi; exit "$status"'
        )
    else:
        client_cmd = (
            f'timeout 220 "{bin_path}" --connect "${{MAHIMAHI_BASE}}" '
            f'--congestion "{cc}" --workload "{workload_path}" --output "{rct_path}"'
        )
    # For bulk runs, timeout exit code 124 is expected when stopping cleanly at bulk_secs.
    expected = log_path if workload_kind == "bulk" else rct_path
    allow_timeout = workload_kind == "bulk"
    res = run_mm_cmd(replay_command(replay, q_bytes, log_path, client_cmd), expected, log_path, allow_timeout=allow_timeout)
    ok_code = res.returncode == 0 or (allow_timeout and res.returncode in (124, 130))
    ok = ok_code and expected.exists() and has_deliveries(log_path)
    if not ok:
        record_failure(out_dir / f"{stem}.err.log", res)
        return stem, res.returncode or 1
    return stem, 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Mahimahi trace suite for quinn-jls or Mortise.")
    parser.add_argument("--stack", required=True, choices=["quinn", "mortise"])
    parser.add_argument("--shard", required=True, choices=list(SHARDS.keys()))
    parser.add_argument("--ccs", required=True, help="Comma-separated congestion algorithms")
    parser.add_argument("--queues", default="150000,2000000", help="Comma-separated queue sizes in bytes")
    parser.add_argument("--workloads", default="bulk,app", help="Comma-separated workloads (bulk,app)")
    parser.add_argument("--bulk-secs", type=int, default=45)
    parser.add_argument("--send-fairness", choices=["true", "false"], default="true")
    parser.add_argument("--stream-receive-window", default=None, help="Optional quinn --stream-receive-window (e.g. 8M)")
    parser.add_argument("--connection-window", help="Quinn connection receive and send windows (e.g. 64M)")
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

    replays = {t: prepare_replay(t, args.trace_dir, args.bulk_secs) for t in traces}
    for trace, replay in replays.items():
        if replay.delay is not None:
            (args.out_dir / f"{trace}.base-rtt-ms").write_text(f"{replay.base_rtt_ms}\n", encoding="utf-8")

    tasks = []
    for cc in ccs:
        for t in traces:
            for q in queues:
                for wk in workloads:
                    wk_path = args.bulk_workload if wk == "bulk" else args.app_workload
                    tasks.append((cc, t, q, wk, wk_path))

    print(f"[{args.stack}] shard={args.shard}: {len(tasks)} runs across {len(traces)} traces (parallel={args.parallel})")

    slot_pool: queue.Queue[int] = queue.Queue()
    for s in range(args.parallel):
        slot_pool.put(s)

    def dispatch(item):
        cc, t, q, wk, wk_path = item
        slot = slot_pool.get()
        try:
            if args.stack == "quinn":
                return run_quinn_job(
                    slot,
                    args.bin,
                    cc,
                    t,
                    replays[t],
                    q,
                    wk,
                    wk_path,
                    args.out_dir,
                    args.bulk_secs,
                    args.send_fairness,
                    args.stream_receive_window or None,
                    args.connection_window,
                )
            return run_mortise_job(args.bin, cc, t, replays[t], q, wk, wk_path, args.out_dir, args.bulk_secs)
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
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
