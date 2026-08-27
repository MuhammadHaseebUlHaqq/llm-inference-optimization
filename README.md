# llm-inference-optimization

Phase 1 of an LLM inference optimization roadmap: run one small model two ways on
a single GPU, measure both with the same harness, then deliberately OOM the KV
cache and plot the crash against the analytical prediction.

The goal is a reproducible, measured artifact, not code that ran once. See
`docs/phase1.md` for the full plan, `docs/vastai.md` for the compute record, and
`CLAUDE.md` for persistent context.

## What is here

```
scripts/
  bench_common.py            shared measurement harness (timing, VRAM, logging, KV math)
  baseline_hf.py             Task 1: HuggingFace transformers baseline (the control)
  bench_vllm.py              Task 2: vLLM, same model and workload
  oom_sweep.py               Task 3: the KV-cache OOM experiment
  bench_vllm_batch.py        Week 3: vLLM batching throughput/latency sweep
  plot_batch_sweep.py        plots for the batching sweep
  analyze_vram_deviations.py per-step VRAM delta reader for the OOM CSV
  profile_decode.py          Week 6: PyTorch profiler trace of the HF decode step
  profile_vllm.py            Week 6: the same trace for vLLM (via the engine hook)
  analyze_trace.py           Week 6: one parser reducing either trace to 3 numbers
  measure_bandwidth.py       measured vs spec memory bandwidth for this exact card
  sglang_cudagraph_cliff.py  SGLang issue #33483 repro (decode CUDA-graph cliff)
  render_lecture_pdfs.sh     re-renders the lecture PDFs from docs/lecture-src/
test/
  probe_weights_vram.py      breaks resident weights down by dtype, pins VRAM gaps
results/                     raw CSV logs, plots, and profiler traces (see results/README.md)
logs/                        raw engine stdout kept for claims the CSVs do not carry
colab/run.ipynb              superseded, kept as the record of the original T4 runner
docs/                        per-task writeups (baseline, vLLM, OOM) and the compute record
blog/draft.md                writeup, grown alongside the measurements
```

### Documents

The repo also carries the written side of the phase, which is half the artifact:

| file | what it is |
| --- | --- |
| `blog/draft.md` | the writeup, grown alongside the measurements |
| `docs/phase1.md` | full scope, methodology, and deliverables for the phase |
| `docs/vastai.md` | the compute record for the rented box |
| `docs/sglang-contribution-runbook.md` | step by step plan for the first SGLang PR |
| `ROADMAP_3week_interview_prep.pdf` | the 3-week roadmap these notes follow |
| `WEEK1_DAY1-4_condensed.pdf` | condensed lecture notes, days 1-4 |
| `WEEK1_DAY5-8_condensed.pdf` | condensed lecture notes, days 5-8 |
| `DAY9-12_condensed.pdf` | condensed lecture notes, days 9-12 |

HTML sources for the PDFs live in `docs/lecture-src/` so they can be re-rendered
rather than rewritten.

## Hardware

Phase 1 runs on a rented Vast.ai box, not Colab. The full record is in
`docs/vastai.md`.

- GPU: one NVIDIA RTX 3060, 12GB (12,288 MiB), Ampere, compute capability 8.6
  (sm_86). On-demand, about $0.045/hr.
- Image: NVIDIA NGC PyTorch, driver 580.126.09, CUDA 13.0, torch 2.12.0+cu130
  preinstalled. transformers, vLLM, and accelerate are not preinstalled.
- Persistent working directory is `/workspace`. The repo and the weights cache
  (`HF_HOME=/workspace/hf-cache`) live there, never `/root` or `/tmp`, because
  storage is billed while the instance is stopped. Destroy the instance when done.

The small card is intentional: at 12GB the KV cache hits the wall at a realistic
context length, which is the headline experiment, not something to avoid.

## How to run

Two environments on purpose, so the vLLM install cannot disturb the baseline
torch.

### Environment A: HuggingFace baseline (Task 1)

Uses the preinstalled torch 2.12. Just add the baseline pins:

```bash
export HF_HOME=/workspace/hf-cache          # cache weights on persistent storage
pip install -r requirements.txt             # transformers, accelerate, bitsandbytes, matplotlib, pandas
python scripts/baseline_hf.py --model Qwen/Qwen2.5-1.5B --prompt-tokens 512 --new-tokens 256
```

The same script runs the 4-bit NF4 comparison through the identical prefill/decode
loop, so the only change is how the weights are stored:

```bash
python scripts/baseline_hf.py --model Qwen/Qwen2.5-1.5B --prompt-tokens 512 --new-tokens 256 --quant nf4
```

The lesson is that 4-bit is a memory lever, not a speed lever at batch 1: weights
drop from ~2945 to ~1099 MiB, but decode slows (bitsandbytes dequantizes NF4 to
fp16 every layer every step, and decode was already memory-bound). The freed VRAM
is the win, and it moves the OOM cliff out.

The OOM sweep (Task 3) also runs in Environment A, because HF grows the cache
organically so the crash is a real memory event:

```bash
python scripts/oom_sweep.py --model Qwen/Qwen2.5-1.5B --prompt-tokens 32 \
    --checkpoint-every 1000 --csv results/oom_sweep.csv --plot results/oom_curve.png
```

The full sweep is a single continuous decode and takes about six hours of wall
time on the 3060.

### Environment B: vLLM (Task 2)

vLLM is binary-tied to a specific torch build, so a naive `pip install vllm` into
Environment A would clobber the box's torch 2.12 with vLLM's bundled 2.11. Install
it into a fresh, isolated venv and let the wheel bring its own torch for CUDA
13.0:

```bash
python -m venv /workspace/vllm-env
source /workspace/vllm-env/bin/activate
# uv is the recommended installer; it resolves the CUDA 13.0 wheel cleanly
uv pip install vllm --torch-backend=cu130
python scripts/bench_vllm.py --model Qwen/Qwen2.5-1.5B --prompt-tokens 512 --new-tokens 256
```

The batching throughput sweep (Week 3) also runs in Environment B. It sends a
growing number of concurrent sequences to vLLM and measures throughput and
latency at each batch size. Run it twice: a realistic pool to find where compute
saturates, and a deliberately constrained pool to force the KV-cache wall at a low
batch size. Then plot:

```bash
# realistic pool: throughput climbs then flattens from the compute ceiling
python scripts/bench_vllm_batch.py --model Qwen/Qwen2.5-1.5B --prompt-tokens 512 \
    --new-tokens 256 --gpu-mem-util 0.9 --sweep 1,2,4,8,16,32,48,64,96,128,192,256 \
    --repeats 2 --tag realistic
# constrained pool: small KV pool so the wall bites early and requests queue
python scripts/bench_vllm_batch.py --model Qwen/Qwen2.5-1.5B --prompt-tokens 512 \
    --new-tokens 256 --gpu-mem-util 0.4 --sweep 1,2,4,8,16,32,48,64 \
    --repeats 2 --tag constrained
python scripts/plot_batch_sweep.py
```

Throughput is total output tokens across the batch divided by wall time, never
blended with the batch-1 decode rate. `ignore_eos` forces exactly the requested
output length per sequence, so total output equals batch x output length.

### Profiling (Week 6)

Explains the gap rather than restating it: capture a trace of one decode window
per engine, then reduce both through the same parser so "idle gap" means the
identical thing on each side.

```bash
# Environment A
python scripts/profile_decode.py --model Qwen/Qwen2.5-1.5B --prompt-tokens 512 \
    --trace results/trace_hf_decode.json.gz
# Environment B, once per engine mode
python scripts/profile_vllm.py --label vllm-eager     --enforce-eager
python scripts/profile_vllm.py --label vllm-compile   --no-cudagraph
python scripts/profile_vllm.py --label vllm-graphonly --cudagraph-only
python scripts/profile_vllm.py --label vllm-graph
# reduce every trace through one parser
python scripts/analyze_trace.py --trace hf=results/trace_hf_decode.json.gz \
    --trace vllm-graph=results/trace_vllm-graph.json.gz \
    --clean-step-ms hf=41.2 --figure results/decode_timeline.png
```

Every "% of peak bandwidth" claim divides by a peak, and which peak is not
obvious. Measure the card rather than trusting the model name:

```bash
python scripts/measure_bandwidth.py
```

On this card that is 360 GB/s theoretical against 291.5 GB/s achievable. It
matters: two nominally identical 3060s ran the same decode work 1.22x apart, so a
before/after comparison split across boxes is meaningless. See `docs/profiling.md`.

## Constraints

- fp16 only, never bf16. The 3060 supports bf16, but we target fp16 for
  comparability with the later NUST HPC V100/T4 runs (which do not support bf16).
  It does not change the KV-cache OOM math (fp16 and bf16 are both 2 bytes).
- Single GPU, 12GB. The small GPU is intentional: the point is to hit the
  KV-cache wall at a realistic context length.
- Small model (Qwen2.5-1.5B, ~3GB of fp16 weights), leaving roughly 9GB for the
  cache to grow into.

## Working dependency versions

The vLLM install is the friction point of this phase. The combinations that
actually ran on the 3060:

- GPU: RTX 3060 12GB (Ampere, sm_86), driver 580.126.09, CUDA 13.0.
- Environment A (baseline + OOM): torch 2.12.0+cu130 (preinstalled),
  transformers 4.46.3, accelerate 0.34.2. Pins in `requirements.txt`.
- Environment B (vLLM): vLLM 0.23.0 (v1 engine) with torch 2.11.0+cu130, in the
  separate `/workspace/vllm-env`.

The transformers pin matters: 4.46+ moved rotary embeddings to a single
model-level module, dropping a fixed ~1.79 GB of duplicated cos/sin buffers that
4.44.x kept across all 28 layers. See the comment in `requirements.txt`.
