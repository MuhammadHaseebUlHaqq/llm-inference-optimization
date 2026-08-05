"""Reproduce SGLang issue #33483: the decode CUDA-graph coverage cliff.

https://github.com/sgl-project/sglang/issues/33483

The claim, as filed against 1x L40 (46GB) with Qwen2.5-0.5B-Instruct: the default
decode `max_bs` is 32, so every decode step at a running batch above 32 falls back
to eager, and median TPOT goes from 3.38 ms to 13.00 ms. The reporter also saw
bimodality, where the same request rate produced two different operating points
depending only on how many prompts the run sent.

This script reproduces both claims on a different card, which is the part of the
issue nobody has contributed yet.

Where the default comes from, in SGLang main as of 2026-08-05:

    python/sglang/srt/server_args.py, _handle_gpu_memory_settings (line ~4602)

It is a ladder over device memory alone, split once on tp_size:

    <20GB -> 8      <35GB -> 24/80     <60GB -> 32/160
    <90GB -> 256/512    <160GB -> 256/512    else -> 512

Model size never enters. A 0.5B model and a 70B model on the same L40 both get 32,
even though the 0.5B leaves tens of GB of KV pool and can therefore reach a running
batch in the hundreds. That is the mechanism behind the cliff.

Two rules this harness follows, both learned the hard way in docs/profiling.md:

- Every configuration is measured on the same physical box in the same session.
  Two nominally identical cards differed by 1.22x in that experiment, so a
  before/after split across boxes would be meaningless.
- The bimodality claim is a run-length effect, so it is tested by varying run
  length at a fixed rate and repeating, not by a single pair of runs. A single
  pair cannot separate a real bistability from ordinary variance.

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

    def resolved_max_bs(self) -> int:
        """The capture ceiling the server actually chose, read back from its log.

        Never trust the flag. When max_bs is left to the heuristic the whole point
        is that its value is not obvious, and an unsupported combination can be
        silently downgraded rather than rejected.
        """
        try:
            with open(self.log_path, errors="replace") as f:
                text = f.read()
        except OSError:
            return -1
        marker = "max_bs"
        for line in text.splitlines():
            if marker in line and ("cuda_graph" in line or "CudaGraph" in line):
                for token in line.replace("=", " ").replace(",", " ").split():
                    if token.isdigit():
                        return int(token)
        return -1

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
                  f"{server.resolved_max_bs()}")
            for rate in args.rates:
                for repeat in range(args.repeats):
                    record(args, server, config, rate, args.num_prompts, repeat)


def phase_bimodality(args) -> None:
    """Fixed rate, three run lengths, repeated, on the default configuration.

    The issue reports 3.68 ms at n=500 and 12-13 ms at n>=1000 at the same rate.
    If that is real bistability, the short runs stay fast across repeats and the
    long runs stay slow. If it is a warmup transient being averaged away by short
    runs, the long-run median will sit between the two and the spread across
    repeats will be wide. One pair of runs cannot tell those apart, which is why
    this phase exists separately from the sweep.
    """
    with Server(args, None) as server:
        print(f"[phase] bimodality at rate {args.bimodal_rate}, captured up to "
              f"{server.resolved_max_bs()}")
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

    p.add_argument("--rates", type=float, nargs="+",
                   default=[8, 16, 20, 24, 28, 32],
                   help="request rates bracketing the cliff")
    p.add_argument("--num-prompts", type=int, default=1000)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--wide-max-bs", type=int, default=128)

    p.add_argument("--bimodal-rate", type=float, default=31)
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
