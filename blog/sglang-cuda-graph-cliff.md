# A serving engine that sizes its CUDA graphs by GPU model

SGLang decides how large a decode batch it will capture CUDA graphs for by
looking at one number: how much memory the GPU has. Model size never enters the
decision. A 0.5B model and a 70B model on the same card get the same coverage,
even though the small one leaves far more KV pool free and can therefore run a
much larger batch.

Above that coverage, decode falls back to eager execution. This post measures
what that costs on an RTX 4090.

The headline: at 32 requests per second, median time per output token was
**30.57 ms on the default configuration and 5.31 ms with the coverage widened,
a 5.76x gap**, on the same card in the same session. The cost of widening was
not extra VRAM. It was **14,110 KV tokens, about 1.1% of the pool**, and half a
second of startup.

Two things I got wrong on the first reading are in here too, because both were
more interesting than the headline.

## The mechanism, from the source

`handle_gpu_memory_settings` in `python/sglang/srt/arg_groups/memory_hook.py`
picks the decode capture ceiling from a ladder over device memory:

| device memory | decode max_bs (tp<4) | cards |
| --- | --- | --- |
| < 20 GB | 8 | T4, 4080 |
| < 35 GB | 48 | A10, 4090, 5090 |
| < 60 GB | 32 | A100 40GB, L40 |
| < 90 GB | 256 | H100, A100 |
| else | 512 | B200, MI300 |

The ladder is not the whole ceiling. In
`model_executor/runner/base_cuda_graph_runner.py`, `get_batch_sizes_to_capture`
clamps the capture list to `req_to_token_pool.size`, which comes from
`max_running_requests`. The effective ceiling is:

    min(ladder(device_memory), req_to_token_pool.size)

That distinction matters. A commenter on the upstream issue reported a capture
list of `[1, 2, 4, 8, 12, 14]` on a 32 GB card and attributed the 14 to the
ladder. The ladder had given them a larger number and the request pool had cut
it. Both quantities have to be read back from the server log or the measurement
is ambiguous.

On this box the ladder resolved to `max_bs=48` with `max_running_requests=4096`,
so the ceiling was the ladder, and the captured shapes were
`[1, 2, 4, 8, 12, 16, 24, 32, 40, 48]`.

## Why 48 and not 24

The `<35GB` rung was raised from 24 to 48 four days before this run, by PR
#37898. The reason given in the source comment:

> 32GB Blackwell (RTX 5090) can hold decode cuda graphs well past bs=24; the
> previous cap forced eager decode at bs>=32 and collapsed high-concurrency
> throughput vs vLLM.

That PR was justified on Qwen3.5-9B, Qwen3.5-4B, Qwen2.5-VL-7B and gpt-oss-20b.
Every model in its validation set is 4B or larger.

So the question this run asks is not "is 24 too low." It is whether 48, chosen
against 9B-class models, still leaves a 0.5B model falling off the edge. Model
size still does not enter the ladder, so a better constant is not a fix.

## Method

One rented RTX 4090 (24 GB), driver 580.65.06, CUDA 13.0. SGLang `main` at
`141febf3`, which includes #37898. `Qwen/Qwen2.5-0.5B-Instruct`, tp=1, flashinfer
backend, no tuning flags beyond the one under test.

Both configurations were measured **on the same physical box in the same
session**. This is not fussiness. An earlier experiment in this repo found two
nominally identical cards differing by 1.22x, so a before-and-after split across
two rentals would have been meaningless.

Default coverage (`max_bs=48`) against `--cuda-graph-max-bs-decode 128`. Request
rates 16 through 72, 1000 prompts per run, input 1024 tokens, output 256, two
repeats per point. Harness: `scripts/sglang_cudagraph_cliff.py`. Raw data:
`results/sglang_cliff.csv`.

## Results

Median time per output token, mean of two repeats:

| rate | default (48) | wide (128) | ratio | default tok/s | wide tok/s |
| --- | --- | --- | --- | --- | --- |
| 16 | 2.64 ms | 2.63 ms | 1.00 | 4142 | 4141 |
| 32 | **30.57 ms** | **5.31 ms** | **5.76** | 6979 | 7887 |
| 48 | 40.99 | 37.81 | 1.08 | 9001 | 9088 |
| 56 | 50.48 | 47.41 | 1.06 | 9222 | 9548 |
| 64 | 56.21 | 53.93 | 1.04 | 9495 | 9668 |
| 72 | 58.70 | 59.41 | 0.99 | 9675 | 9750 |

p99 at rate 32 moves with the median, 36.76 ms against 7.32 ms, so this is the
whole distribution shifting rather than a tail effect.

The cliff reproduces. A 0.5B model on the raised rung still falls off the
captured-shape edge, which is the point: 48 is a better constant chosen against
larger models, and the ladder still ignores model size.

## The first thing I got wrong: the divergence is bounded above, not below

I expected the gap to open at a rate that drives the running batch past 48 and
then stay open. It does not. It opens at 32 and closes again by 48.

Reading the throughput column explains it. The card tops out near 9.7k output
tokens per second. From rate 48 upward both configurations sit at that ceiling,
so time per token is dominated by queueing delay, which is identical because
throughput is identical. Graph coverage cannot show through once offered load
exceeds what either configuration can serve.

At rate 32 the default is both slower per token and lower throughput, 6979
against 7887 tok/s. It fell off the captured-shape edge, decode slowed, the
running batch grew, and it stayed there. That is the absorbing state the upstream
issue describes, reproduced.

So the cliff bites **below saturation**, in the load region where a server is
actually doing useful work. That is a stronger result than a gap that persists
everywhere, not a weaker one. A gap that only appears past saturation would be a
curiosity. A gap that appears at the load you would actually run at is a bug.

## The second thing I got wrong: widening looked free

My first pass compared total VRAM after the server reported ready, saw
19392.8 MiB against 19390.8 MiB, and concluded the widening cost nothing. The
widened run was 2 MiB *lower*, which should have been the tell.

That diff can only ever be about zero. `mem_fraction_static` is derived from

    reserved_mem = chunked_prefill_size * 1.5 + max_bs * 2

so raising `max_bs` raises the reserve, which shrinks the KV pool by roughly what
the graph buffers gain. Total footprint stays flat by construction. I was
diffing the one quantity the mechanism holds constant.

The variables that move:

| | default (48) | wide (128) | delta |
| --- | --- | --- | --- |
| `max_total_num_tokens` | 1,330,288 | 1,316,178 | **-14,110 tokens** |
| graph capture `mem usage` | 0.13 GB | 0.27 GB | **+143.4 MiB** |
| graph capture `elapsed` | 2.19 s | 2.69 s | +0.50 s |
| total VRAM after ready | 19392.8 MiB | 19390.8 MiB | -2.0 MiB |

KV cell size here is 12,288 bytes per token, from the allocation log: K at
7.61 GB and V at 7.61 GB over 1,330,288 tokens. So 14,110 tokens is 165.4 MiB of
pool given up against 143.4 MiB of graph buffers gained.

The tree's own prediction is `(128 - 48) * 2 = 160 MB`. Both measurements bracket
it. So `reserve_for_graph_mb` is accurate on this card, and the honest cost of
widening 48 to 128 is about 1.1% of the KV pool plus half a second of startup.

This is a better argument for widening than "it is free," which is not true, and
better than "it costs 0.22 GB of VRAM," which is what the total-footprint framing
suggests and which overstates it.

## The bimodality did not reproduce, and the test was not fair

The upstream issue reports bistability: at a fixed rate, short runs stay fast and
long runs stay slow, with no recovery. I tested it at a fixed rate of 62 across
run lengths of 500, 1000 and 2000 prompts, three repeats each.

| prompts | median TPOT | runs | p99/median |
| --- | --- | --- | --- |
| 500 | 31.68 ms | 31.2, 33.8, 30.0 | 1.18-1.32 |
| 1000 | 53.89 ms | 49.3, 57.0, 55.3 | 1.08-1.15 |
| 2000 | 81.87 ms | 85.0, 79.6, 81.0 | 1.15-1.15 |

Monotonic in run length, tight within each length, no long-tail split. That is
not two states. That is one state getting steadily worse.

But the test was badly designed, and the sweep above says why. I picked rate 62
as "just above the ceiling of 48" before I had the sweep results. On this card 62
is well past saturation, so all three run lengths were measuring queue growth
under sustained overload. A fair test here would fix the rate near 32, where the
two configurations actually diverge.

So: null result, reported, with the caveat that it does not falsify the upstream
claim. It tests the wrong operating point.

## Limitations

- **One card, one model.** Everything here is a 4090 running Qwen2.5-0.5B. The
  claim that the ladder ignores model size is read from the source and is
  general; the magnitudes are not.
- **The bimodality test targets the wrong rate**, as above. It should be rerun
  near 32 before anyone treats the null as meaningful.
- **The saturation ceiling is not independently characterised.** I infer roughly
  9.7k tok/s from the throughput column plateauing in both configurations, but I
  did not measure it directly with a separate saturation sweep.
- **`max_total_num_tokens` was not recorded per run** in the first pass, only in
  the server logs. That is why the cost analysis needed a second reading. The
  harness now records it, along with graph capture size and time.
- **fp16 versus bf16.** This repo targets fp16 for comparability with later V100
  and T4 runs, but SGLang allocated the KV cache in bf16 here by default. Both
  are 2 bytes per element so none of the memory arithmetic changes.

## Upstream

Reported on sgl-project/sglang issue #33483, which is where the original L40
measurement was filed. Two notes on what that reporting is worth.

The startup warning that would make this visible to operators already exists as
PR #33900, opened by the issue's author and unreviewed for a month. Writing a
second one would be wasted work. As of this run SGLang had 4,318 open pull
requests, 2,446 of them older than a month, so an unreviewed PR is the normal
outcome rather than a slight.

That reframes what an outside measurement is for. It is not a step toward a merge
you control. It is a public, reproducible data point that costs a maintainer
about a minute to check and gives them a second card to reason about. Whether it
turns into code is not up to the person who measured it.

## Reproducing

    git clone https://github.com/MuhammadHaseebUlHaqq/llm-inference-optimization
    python scripts/sglang_cudagraph_cliff.py --phase sweep
    python scripts/sglang_cudagraph_cliff.py --phase bimodality

Full setup, including which card to rent and why the rung matters, is in
`docs/sglang-contribution-runbook.md`. Raw data is `results/sglang_cliff.csv`,
and the server logs both phases parsed are committed under `logs/sglang_cliff/`.

Total cost of the measurement: about three hours on a rented 4090, under $1.50.
