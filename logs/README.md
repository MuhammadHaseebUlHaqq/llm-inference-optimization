# Raw engine logs

Kept because some claims in the writeup are only visible in engine stdout, never
in the CSVs. The benchmark schema in `results/` has one VRAM number per run; the
vLLM init log is where that number splits into its parts.

| file | produced by | why it is kept |
| --- | --- | --- |
| `vllm_run.log` | `scripts/bench_vllm.py` | the vLLM v1 init sequence for the Task 2 batch-1 run |

## What `vllm_run.log` is load-bearing for

`results/baseline_vllm.csv` reports 11237 MiB, which is NVML device-used, so it
folds weights, the reserved KV pool, and the captured CUDA graphs into one
figure. That is not comparable to the HF rows, which grow organically. The split
only exists in this log:

- `Available KV cache memory: 6.91 GiB` and `GPU KV cache size: 258,799 tokens`,
  which is the reserved pool, sized once at startup rather than grown per token.
- `Graph capturing finished in 5 secs, took 0.47 GiB`, the CUDA-graph cost, in
  two passes (51 PIECEWISE sizes then 35 FULL).
- `Casting torch.bfloat16 to torch.float16`, confirming the fp16-only rule
  actually held through the engine rather than being silently upcast.
- The resolved `max_model_len=784` and `gpu_memory_utilization=0.9`, which are
  what make the pool that size and are therefore part of the measurement.

Note the version skew in this file: it was captured under vLLM 0.25.1, while the
pinned Task 2 run in the README is 0.23.0. The numbers quoted above are stable
across both, but do not read any 0.25-only field here as if it described the
pinned environment.

## Convention

Logs are committed only when they carry something the CSVs cannot. Routine run
output is not archived here. `nohup.out` is gitignored for that reason.
