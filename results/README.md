# Results provenance

Every file here was produced by a script in `scripts/`, on the Vast.ai RTX 3060
box described in `docs/vastai.md`. Nothing in this directory is hand-edited. If a
prose document in `docs/`, the blog draft, or a lecture PDF disagrees with a CSV
here, the CSV wins.

| file | produced by | what it holds |
| --- | --- | --- |
| `baseline_hf.csv` | `baseline_hf.py` | Task 1, the HF transformers control (fp16 and NF4 rows) |
| `baseline_vllm.csv` | `bench_vllm.py` | Task 2, vLLM on the same model and workload |
| `oom_sweep.csv` | `oom_sweep.py` | Task 3, per-checkpoint VRAM against context length |
| `oom_curve.png` | `oom_sweep.py` | the headline plot: measured VRAM vs the analytical prediction |
| `vllm_batch_sweep.csv` | `bench_vllm_batch.py` | Week 3 batching sweep, realistic and constrained pools |
| `vllm_batch_throughput.png` | `plot_batch_sweep.py` | output throughput against batch size |
| `vllm_batch_latency.png` | `plot_batch_sweep.py` | per-token latency (TPOT) against batch size |
| `vllm_batch_tradeoff.png` | `plot_batch_sweep.py` | the throughput/latency tradeoff curve |
| `trace_hf_decode.json.gz` | `profile_decode.py` | Week 6, PyTorch profiler trace of one HF decode window |
| `trace_vllm-*.json.gz` | `profile_vllm.py` | the same window under vLLM, one file per engine mode |
| `profile_summary.csv` | `analyze_trace.py` | traces reduced to kernels, launches, and idle gap per step |
| `decode_timeline.png` | `analyze_trace.py` | the reduced traces drawn on a common time axis |

## Reproducing

The exact commands are in the README, split by environment: `baseline_hf.py` and
`oom_sweep.py` run in Environment A (the image's torch 2.12), everything with
`vllm` in the name runs in Environment B (`/workspace/vllm-env`). Both write with
`--csv`, appending a row per run rather than overwriting, so a re-run adds history
instead of destroying it.

## Reading the benchmark CSVs

`baseline_hf.csv`, `baseline_vllm.csv`, and `oom_sweep.csv` share one schema,
written by `bench_common.py`, so they can be concatenated. The columns that carry
the claims:

- `prefill_seconds` and `decode_seconds` are always separate. Prefill is the one
  forward pass over the prompt; decode is the token-at-a-time loop. Never blend
  them into a single latency number.
- `decode_tokens_per_sec` is the headline number: new tokens divided by decode
  wall time, excluding the prompt and excluding prefill.
- `weights_vram_mib`, `peak_allocated_mib`, and `peak_reserved_mib` are tracked
  separately on purpose. Allocated is what tensors hold; reserved is what the
  caching allocator took from the driver. The gap between them is real and is not
  a leak.
- The vLLM rows report NVML device-used, not allocator numbers, because the v1
  engine reserves its KV pool up front in a child process. That figure is not
  comparable to the HF rows, which grow organically. The `note` column says so on
  every affected row.
- `oom` marks the row where the run actually died. In `oom_sweep.csv` the rows
  before it are `checkpoint` rows from one continuous decode, not separate runs.
