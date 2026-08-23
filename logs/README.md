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
