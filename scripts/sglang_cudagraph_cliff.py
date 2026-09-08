"""Reproduce SGLang issue #33483: the decode CUDA-graph coverage cliff.

https://github.com/sgl-project/sglang/issues/33483

The claim, as filed against 1x L40 (46GB) with Qwen2.5-0.5B-Instruct: the default
decode `max_bs` is 32, so every decode step at a running batch above 32 falls back
to eager, and median TPOT goes from 3.38 ms to 13.00 ms.

Since the issue was filed the ladder has moved once, which is the thing this
harness now tests. PR #37898 (commit f3b2725, merged 2026-09-04 by BBuf, a
collaborator) raised the <35GB rung from max_bs 24 to 48 and chunked_prefill_size
from 2048 to 4096, citing an RTX 5090 (32GB) serving Qwen3.5-9B. The <60GB rung
the original reporter sits on is untouched, and model size still never enters the
ladder. So the open question is no longer "is 24 too low" but "is 48 still too low
for a small model", which is exactly the original thesis re-aimed at a fresh,
maintainer-authored constant.

Where the ladder lives, in SGLang main as of 2026-09-08 (commit ccfa120):

    python/sglang/srt/arg_groups/memory_hook.py, handle_gpu_memory_settings (line 27)

It moved out of server_args.py in the arg_groups refactor. It is a ladder over
device memory alone, split once on tp_size:

    <20GB  -> max_bs 8
    <35GB  -> max_bs 48/160    (A10, 4090, 5090)   <- raised from 24/80 by #37898
    <60GB  -> max_bs 32/160    (A100 40GB, L40)    <- the reporter's rung
    <90GB  -> max_bs 256/512
    <160GB -> max_bs 256/512
    else   -> max_bs 512

The effective ceiling is not the ladder value alone. In
model_executor/runner/base_cuda_graph_runner.py, get_batch_sizes_to_capture (line
64) clamps the capture list to req_to_token_pool.size, which is derived from
max_running_requests. So:

    effective ceiling = min(ladder(device_memory), req_to_token_pool.size)

That clamp is why a commenter on the issue saw bs=[1, 2, 4, 8, 12, 14] on a 5090:
the ladder gave them a larger number and the request pool cut it to 14. Both
quantities are therefore read back from the server log, never assumed.

Two rules this harness follows, both learned the hard way in docs/profiling.md:

- Every configuration is measured on the same physical box in the same session.
  Two nominally identical cards differed by 1.22x in that experiment, so a
  before/after split across boxes would be meaningless.
- The bimodality claim is a run-length effect, so it is tested by varying run
  length at a fixed rate and repeating, not by a single pair of runs. A single
  pair cannot separate a real bistability from ordinary variance. Pick the rate
  from the sweep, near where the two configs actually diverge. A rate chosen
  naively as "just above the ceiling" can land past saturation, where both
  configs are queue-bound and the test measures queue growth instead.

Usage (on the GPU box, inside the SGLang environment):

    python scripts/sglang_cudagraph_cliff.py --phase sweep
    python scripts/sglang_cudagraph_cliff.py --phase bimodality

Results append to results/sglang_cliff.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_common import device_used_mib, print_env  # noqa: E402


DEFAULT_CSV = os.path.join("results", "sglang_cliff.csv")


@dataclass
class CliffResult:
    """One bench_serving run against one server configuration."""

    config: str            # "default" | "wide"
    cuda_graph_max_bs: int  # what the server actually captured up to
    capture_bs_list: str    # the full capture list, as logged
    max_running_requests: int  # the other half of the effective ceiling
    max_total_num_tokens: int  # KV pool size, where widening is actually paid for
    graph_capture_mib: float   # graph buffers, from the capture end line
    graph_capture_seconds: float  # capture time, the other half of the cost
    request_rate: float
    num_prompts: int
    repeat: int

    median_tpot_ms: float
    p99_tpot_ms: float
    median_ttft_ms: float
    output_throughput: float
    completed: int

    # Per-server-launch costs, repeated on every row of that launch so the CSV
    # stays flat and each row is self-describing.
    startup_seconds: float
    vram_after_ready_mib: float

    model: str
    attention_backend: str
    input_len: int
    output_len: int

    gpu_name: str = ""
    sglang_version: str = ""
    note: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


def write_result(result: CliffResult, csv_path: str) -> None:
    """Append a result row, writing the header if the file is new."""
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    row = asdict(result)
    is_new = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def port_is_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


class Server:
    """A launched sglang server, timed from exec to first successful health check.

    The startup number matters here: the whole argument for widening graph
    coverage is that the extra capture time is small, so it has to be measured
    rather than asserted. Capture happens before the server reports ready, so
    time-to-ready includes it.
    """

    def __init__(self, args, max_bs: int | None):
        self.args = args
        self.max_bs = max_bs
        self.proc: subprocess.Popen | None = None
        self.startup_seconds = 0.0
        self.vram_after_ready_mib = 0.0
        self.log_path = os.path.join(
            args.log_dir, f"server_{'default' if max_bs is None else max_bs}.log"
        )

    def cmd(self) -> list[str]:
        c = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", self.args.model,
            "--attention-backend", self.args.attention_backend,
            "--host", "127.0.0.1",
            "--port", str(self.args.port),
        ]
        if self.max_bs is not None:
            # Current flag name. --cuda-graph-max-bs still works as a deprecated
            # alias, which is what the issue reporter used.
            c += ["--cuda-graph-max-bs-decode", str(self.max_bs)]
        c += self.args.extra_server_args
        return c

    def __enter__(self) -> "Server":
        if port_is_open(self.args.port):
            raise RuntimeError(
                f"port {self.args.port} is already in use. A previous server did "
                f"not shut down, and benchmarking against it would silently "
                f"measure the wrong configuration."
            )
        os.makedirs(self.args.log_dir, exist_ok=True)
        print(f"[server] launching: {' '.join(self.cmd())}")
        self.log = open(self.log_path, "w")
        start = time.perf_counter()
        self.proc = subprocess.Popen(
            self.cmd(), stdout=self.log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._wait_ready(start)
        return self

    def _wait_ready(self, start: float) -> None:
        import urllib.error
        import urllib.request

        url = f"http://127.0.0.1:{self.args.port}/health_generate"
        deadline = start + self.args.startup_timeout
        while time.perf_counter() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"server exited with code {self.proc.returncode} before becoming "
                    f"ready. See {self.log_path}."
                )
            try:
                with urllib.request.urlopen(url, timeout=5) as r:
                    if r.status == 200:
                        self.startup_seconds = time.perf_counter() - start
                        # Read VRAM only after ready, so the weights, the KV pool,
                        # and the captured graphs are all resident.
                        self.vram_after_ready_mib = device_used_mib(self.args.device)
                        print(
                            f"[server] ready in {self.startup_seconds:.1f}s, "
                            f"{self.vram_after_ready_mib:.0f} MiB used"
                        )
                        return
            except (urllib.error.URLError, OSError, TimeoutError):
                pass
            time.sleep(1.0)
        raise RuntimeError(f"server not ready within {self.args.startup_timeout}s")

    def _log_text(self) -> str:
        try:
            with open(self.log_path, errors="replace") as f:
                return f.read()
        except OSError:
            return ""

    def capture_bs(self) -> list[int]:
        """The decode capture list the server actually built, read from its log.

        Never trust the flag. The ladder value is only one half of the ceiling:
        get_batch_sizes_to_capture also clamps the list to req_to_token_pool.size,
        so the flag can be silently reduced. The log line is emitted by
        model_runner_components/cuda_graph_setup.py and looks like:

            Capture target decode CUDA graph begin. backend=full,
            num_tokens_per_req=1, bs=[1, 2, 4, ...], avail mem=8.12 GB

        The draft-worker line is skipped: speculative decoding is off here, but a
        future run with it on would otherwise pick up the wrong list.
        """
        for line in self._log_text().splitlines():
            if "Capture" not in line or "decode" not in line:
                continue
            if "draft" in line:
                continue
            m = re.search(r"bs=\[([0-9,\s]*)\]", line)
            if m:
                return [int(t) for t in m.group(1).split(",") if t.strip()]
        return []

    def resolved_max_bs(self) -> int:
        """The top of the capture list, or -1 if the log did not report one."""
        bs = self.capture_bs()
        return max(bs) if bs else -1

    def resolved_max_total_num_tokens(self) -> int:
        """KV pool size in tokens, logged by managers/scheduler.py.

        This is where widening the graph coverage is actually paid for, and
        leaving it out of the CSV is what made the first run of this experiment
        report "widening was free". It is not free: mem_fraction_static derives
        from reserved_mem = chunked_prefill_size * 1.5 + max_bs * 2, so a larger
        max_bs shrinks the pool by roughly the amount the graph buffers grow and
        the total footprint stays flat. Diffing total VRAM alone sees nothing.
        """
        m = re.search(r"max_total_num_tokens=(\d+)", self._log_text())
        return int(m.group(1)) if m else -1

    def graph_capture_cost(self) -> tuple[float, float]:
        """(MiB, seconds) spent capturing decode graphs, from the capture end line:

            Capture target decode CUDA graph end. elapsed=2.19 s,
            mem usage=0.13 GB, avail mem=5.07 GB.

        This is the direct measurement of what wider coverage costs, and it is
        the number to check the tree's own (max_bs * 2) MB prediction against.
        """
        for line in self._log_text().splitlines():
            if "Capture" not in line or "decode" not in line or "end" not in line:
                continue
            if "draft" in line:
                continue
            mem = re.search(r"mem usage=([0-9.]+) GB", line)
            sec = re.search(r"elapsed=([0-9.]+) s", line)
            if mem and sec:
                return float(mem.group(1)) * 1024.0, float(sec.group(1))
        return float("nan"), float("nan")

    def resolved_max_running_requests(self) -> int:
        """max_running_requests as the scheduler resolved it.

        Logged by managers/scheduler.py alongside max_total_num_tokens. It is the
        other input to the effective ceiling, so a row that records only max_bs
        cannot distinguish "the ladder chose this" from "the request pool cut it".
        """
        m = re.search(r"max_running_requests=(\d+)", self._log_text())
        return int(m.group(1)) if m else -1

    def __exit__(self, *exc) -> None:
        if self.proc is not None and self.proc.poll() is None:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                self.proc.wait(timeout=30)
        self.log.close()
        # The port takes a moment to free, and launching the next server against
        # a half-closed port produces a confusing failure.
        for _ in range(30):
            if not port_is_open(self.args.port):
                return
            time.sleep(1.0)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_bench(args, rate: float, num_prompts: int) -> dict:
    """One bench_serving run. Returns the parsed JSONL record."""
    out_file = os.path.join(args.log_dir, "bench.jsonl")
    if os.path.exists(out_file):
        os.remove(out_file)
    cmd = [
        sys.executable, "-m", "sglang.benchmark.serving",
        "--backend", "sglang",
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "--dataset-name", "random",
        "--random-input-len", str(args.input_len),
        "--random-output-len", str(args.output_len),
        "--random-range-ratio", "1.0",
        "--num-prompts", str(num_prompts),
        "--request-rate", str(rate),
        "--warmup-requests", str(args.warmup_requests),
        "--output-file", out_file,
    ]
    # --flush-cache is deliberately never passed. SGLang issue #3050 traced a
    # decode-throughput discrepancy to flush_cache releasing preallocated CUDA
    # memory and inflicting a cold start on the requests that follow, which is
    # exactly the kind of contamination this measurement cannot afford.
    print(f"[bench] rate={rate} n={num_prompts}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.bench_timeout)
    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        print(proc.stderr[-4000:])
        raise RuntimeError(f"bench_serving failed with code {proc.returncode}")
    with open(out_file) as f:
        lines = [line for line in f.read().splitlines() if line.strip()]
    return json.loads(lines[-1])


def record(args, server: Server, config: str, rate: float, n: int, repeat: int,
           note: str = "") -> CliffResult:
    rec = run_bench(args, rate, n)
    result = CliffResult(
        config=config,
        cuda_graph_max_bs=server.resolved_max_bs(),
        capture_bs_list=" ".join(str(b) for b in server.capture_bs()),
        max_running_requests=server.resolved_max_running_requests(),
        max_total_num_tokens=server.resolved_max_total_num_tokens(),
        graph_capture_mib=server.graph_capture_cost()[0],
        graph_capture_seconds=server.graph_capture_cost()[1],
        request_rate=rate,
        num_prompts=n,
        repeat=repeat,
        median_tpot_ms=rec.get("median_tpot_ms", float("nan")),
        p99_tpot_ms=rec.get("p99_tpot_ms", float("nan")),
        median_ttft_ms=rec.get("median_ttft_ms", float("nan")),
        output_throughput=rec.get("output_throughput", float("nan")),
        completed=rec.get("completed", -1),
        startup_seconds=server.startup_seconds,
        vram_after_ready_mib=server.vram_after_ready_mib,
        model=args.model,
        attention_backend=args.attention_backend,
        input_len=args.input_len,
        output_len=args.output_len,
        gpu_name=_gpu_name(),
        sglang_version=_sglang_version(),
        note=note,
    )
    write_result(result, args.csv)
    print(
        f"  -> median TPOT {result.median_tpot_ms:.2f} ms, "
        f"output {result.output_throughput:.1f} tok/s"
    )
    return result


def _gpu_name() -> str:
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if smi.returncode == 0:
            return smi.stdout.strip().splitlines()[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def _sglang_version() -> str:
    try:
        import sglang

        return getattr(sglang, "__version__", "unknown")
    except ImportError:
        return "unknown"


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def phase_sweep(args) -> None:
    """Rate sweep at fixed run length, default coverage against widened coverage.

    Both configurations see identical rates in the same session on the same card.
    """
    for config, max_bs in (("default", None), ("wide", args.wide_max_bs)):
        with Server(args, max_bs) as server:
            print(f"[phase] sweep, config={config}, captured up to "
                  f"{server.resolved_max_bs()}, max_running_requests="
                  f"{server.resolved_max_running_requests()}")
            for rate in args.rates:
                for repeat in range(args.repeats):
                    record(args, server, config, rate, args.num_prompts, repeat)


def phase_bimodality(args) -> None:
    """Fixed rate, three run lengths, repeated, on the default configuration.

    The issue reports 3.68 ms at n=500 and 12-13 ms at n>=1000 at the same rate,
    measured on the 32 rung. The rate here is scaled to the 48 rung instead.
    If that is real bistability, the short runs stay fast across repeats and the
    long runs stay slow. If it is a warmup transient being averaged away by short
    runs, the long-run median will sit between the two and the spread across
    repeats will be wide. One pair of runs cannot tell those apart, which is why
    this phase exists separately from the sweep.
    """
    with Server(args, None) as server:
        print(f"[phase] bimodality at rate {args.bimodal_rate}, captured up to "
              f"{server.resolved_max_bs()}, max_running_requests="
              f"{server.resolved_max_running_requests()}")
        for n in args.bimodal_prompts:
            for repeat in range(args.bimodal_repeats):
                record(args, server, "default", args.bimodal_rate, n, repeat,
                       note="bimodality")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=["sweep", "bimodality", "both"], default="both")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--attention-backend", default="flashinfer")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--csv", default=DEFAULT_CSV)
    p.add_argument("--log-dir", default=os.path.join("logs", "sglang_cliff"))

    p.add_argument("--input-len", type=int, default=1024)
    p.add_argument("--output-len", type=int, default=256)
    p.add_argument("--warmup-requests", type=int, default=16)

    # These bracket a ceiling of 48, not the 24 the <35GB rung used to give.
    # The cliff shows up when the equilibrium running batch crosses the capture
    # ceiling, so the rates have to push past 48 on a 24GB card, not past 32.
    p.add_argument("--rates", type=float, nargs="+",
                   default=[16, 32, 48, 56, 64, 72],
                   help="request rates bracketing the cliff")
    p.add_argument("--num-prompts", type=int, default=1000)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--wide-max-bs", type=int, default=128)

    # Just above the 48 ceiling, mirroring the reporter's choice of a rate just
    # above their own 32. Recheck this if the sweep puts the cliff elsewhere.
    p.add_argument("--bimodal-rate", type=float, default=62)
    p.add_argument("--bimodal-prompts", type=int, nargs="+", default=[500, 1000, 2000])
    p.add_argument("--bimodal-repeats", type=int, default=3)

    p.add_argument("--startup-timeout", type=float, default=900)
    p.add_argument("--bench-timeout", type=float, default=3600)
    p.add_argument("--extra-server-args", nargs=argparse.REMAINDER, default=[])
    args = p.parse_args()

    print_env()
    if args.phase in ("sweep", "both"):
        phase_sweep(args)
    if args.phase in ("bimodality", "both"):
        phase_bimodality(args)
    print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
