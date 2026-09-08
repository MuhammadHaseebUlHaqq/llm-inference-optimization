# Runbook: first SGLang contribution, issue #33483

Target: https://github.com/sgl-project/sglang/issues/33483

Verified against SGLang `main` at commit `ccfa120` (2026-09-08). The code moved
since the first draft of this runbook, so if you are reading it much later,
re-check the two file paths below before spending money on a box.

The claim under test is that SGLang's default decode CUDA-graph coverage
(`max_bs`) is chosen from device memory alone, so a small model on a large card
gets graph coverage far below the batch it actually reaches, and decode falls off
a cliff into eager execution above that batch.

Budget: about 2 hours of GPU time on a rented 4090, plus 30 to 60 minutes of
billed setup (image pull, source install, weight download). At the typical
on-demand 4090 rate of $0.35 to $0.50/hr that is roughly $1.00 to $1.75 for one
clean pass. Storage is negligible while the instance runs and is the main way to
waste money afterwards: 60 GB left stopped for a week costs about $2.

---

## What changed since this runbook was first written

Two things, and both matter.

**1. The ladder moved file.** It used to be `ServerArgs._handle_gpu_memory_settings`
in `python/sglang/srt/server_args.py`. The `arg_groups` refactor moved it to:

    python/sglang/srt/arg_groups/memory_hook.py, handle_gpu_memory_settings (line 27)

**2. The rung this experiment targets was raised, five weeks after the issue was
filed.** PR #37898 (commit `f3b2725`, merged 2026-09-04 by BBuf, a collaborator)
raised the `<35GB` rung from `max_bs` 24 to 48, and `chunked_prefill_size` from
2048 to 4096. Its stated reason, quoted from the source comment:

> 32GB Blackwell (RTX 5090) can hold decode cuda graphs well past bs=24; the
> previous cap forced eager decode at bs>=32 and collapsed high-concurrency
> throughput vs vLLM.

That PR does not reference issue #33483. It was justified entirely on Qwen3.5-9B,
Qwen3.5-4B, Qwen2.5-VL-7B and gpt-oss-20b benchmarks against vLLM.

This is good news, not bad. It means:

- A 4090 now gets **48**, not 24. Any instruction telling you to expect 24 is stale.
- The reporter's own rung (`<60GB`, the L40) is **untouched**. The filed bug is unfixed.
- The raise was derived from 9B-and-larger models. **Model size still never enters
  the ladder.** 48 is a better constant, not a fix.
- There is now a named, active maintainer on this exact code path, which solves the
  original problem with this plan: nobody had replied to the issue in five weeks.

So the contribution is sharper than it was. It is no longer "24 is too low." It is:

> You raised the `<35GB` rung to 48 using 9B-class models. Here is a 0.5B on that
> same rung, on a 4090, still falling off the cliff, because the ladder is indexed
> on device memory alone.

---

## The diagnosis (read this before you start)

`handle_gpu_memory_settings` in `python/sglang/srt/arg_groups/memory_hook.py`
(line 27) picks decode `max_bs` from a ladder over device memory:

| device memory | chunked_prefill_size | decode max_bs (tp<4 / tp>=4) | cards |
| --- | --- | --- | --- |
| < 20 GB | 2048 | 8 | T4, 4080 |
| < 35 GB | 4096 | **48 / 160** | A10, 4090, 5090. Raised from 24/80 by #37898 |
| < 60 GB | 4096 | **32 / 160** | A100 40GB, L40. The reporter's rung, unchanged |
| < 90 GB | 8192 | 256 / 512 | H100, A100 |
| < 160 GB | 8192 | 256 / 512 | H20, H200 |
| else | 16384 | 512 | B200, MI300 |

Model size never enters. A 0.5B model and a 70B model on the same card get the
same number, even though the 0.5B leaves far more KV pool and can therefore reach
a much larger running batch.

**The ladder is only half the ceiling.** In
`python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py`,
`get_batch_sizes_to_capture` (line 64) clamps the capture list to
`req_to_token_pool.size`, which comes from `max_running_requests`. So:

    effective ceiling = min(ladder(device_memory), req_to_token_pool.size)

This is worth knowing because a commenter on the issue reported
`bs=[1, 2, 4, 8, 12, 14]` on a 5090 and attributed it to the ladder. It was not
the ladder: their request pool cut the list to 14. If you conflate those two
mechanisms in your issue comment, a maintainer who knows this code will discount
everything else you wrote. The harness therefore reads **both** numbers back out
of the server log and records them on every CSV row.

One more supporting fact from `memory_hook.py`: the reserve model charges
`chunked_prefill_size * 1.5 + max_bs * 2` MB for activations plus graph buffers.
Widening 48 to 128 predicts `(128 - 48) * 2 = 160` MB. The reporter measured
0.22 GB for their own widening. The tree's own cost model is already roughly
accurate here and already says the cost is small. Checking that prediction on a
second card is a cheap, genuinely useful result.

---

## Step 1: rent the box

Vast.ai, on-demand, one GPU. Add your SSH key at https://cloud.vast.ai/manage-keys/
**before** you rent: Vast only injects account keys into new instances, so adding
it afterwards locks you out of the box you just paid for.

- **GPU: 1x RTX 4090 (24 GB).** It sits on the `<35GB` rung, the one #37898 just
  changed. That is now the point: you are testing a fresh maintainer-authored
  constant with a model class its author did not test.
- **Disk: at least 60 GB.** SGLang from source plus its wheels plus the model.
  Disk is permanent on Vast and cannot be resized after creation.
- **On-demand, not interruptible.** A preemption mid-sweep silently corrupts a
  latency measurement.
- **Image: an NVIDIA NGC PyTorch image**, same as the Phase 1 box.
- Prefer high reliability score and high download bandwidth over the last cent of
  price. A slow host turns a 15-minute setup into an hour of billed time, and a
  host that drops you at minute 50 costs the whole sweep.
- Note the exact GPU name, VRAM, driver, and CUDA version from `nvidia-smi`. You
  will quote them in the issue comment, and `docs/profiling.md` is the reminder of
  why: two nominally identical cards differed by 1.22x.

Raw DLPerf does not matter much here. Both configurations are measured on the same
box in the same session, so absolute card speed cancels out of the comparison.

Storage is billed while the instance is stopped. Destroy it when you are done, do
not just stop it.

## Step 2: set up

Everything lives in `/workspace` so it survives a stop.

```bash
export HF_HOME=/workspace/hf-cache
mkdir -p /workspace/hf-cache

python --version          # must be 3.10 or higher
nvidia-smi                # record GPU name, VRAM, driver, CUDA
```

Fork `sgl-project/sglang` on GitHub first (button on the repo page), then clone
**your fork**, not the upstream. You need the fork to open a PR later, and cloning
upstream now means redoing this.

```bash
cd /workspace
git clone https://github.com/<your-username>/sglang.git
cd sglang
git remote add upstream https://github.com/sgl-project/sglang.git
```

Stay on `main`. The issue was filed against a `main` dev build, and the install
docs pin an older release tag, which would put you on different code from the
reporter. It would also put you *before* #37898, which is the change you are
testing.

Install from source. This is slower than `uv pip install sglang` but you need the
source tree anyway to change `memory_hook.py`, and having two SGLangs installed is
a debugging trap you do not want.

```bash
pip install --upgrade pip
pip install uv
uv pip install -e "python"
```

CUDA 13 is the default for the current wheels, which matches the Phase 1 box. If
the box turns out to be CUDA 12, follow the CUDA 12 override block in
https://docs.sglang.io/get_started/install.html rather than fighting it.

Verify before benchmarking anything:

```bash
python -c "import sglang; print(sglang.__version__)"
git log -1 --format=%H                     # record the commit you are testing
git merge-base --is-ancestor f3b2725 HEAD && echo "has #37898"
python -m sglang.launch_server --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --attention-backend flashinfer --port 30000
```

Watch the startup log for two lines:

```
max_total_num_tokens=..., chunked_prefill_size=4096, ..., max_running_requests=..., ...
Capture target decode CUDA graph begin. backend=full, num_tokens_per_req=1, bs=[..., 48], avail mem=... GB
```

**On a 4090 the capture list should top out at 48.** If it tops out at 24, your
tree predates #37898 and you are on the wrong commit. If it tops out at something
small and odd, read `max_running_requests` on the line above: the request pool is
clamping you, not the ladder, and that changes what you are measuring. Either way,
stop and work out which before running the sweep. Ctrl-C once you have seen it.

## Step 3: get this repo onto the box

```bash
cd /workspace
git clone https://github.com/MuhammadHaseebUlHaqq/llm-inference-optimization.git
cd llm-inference-optimization
```

The reproduction harness is `scripts/sglang_cudagraph_cliff.py`. Run it from the
environment where SGLang is installed.

## Step 4: run the reproduction

Two phases, run them separately so a failure in one does not cost you the other.

```bash
cd /workspace/llm-inference-optimization
python scripts/sglang_cudagraph_cliff.py --phase sweep
```

This launches two servers in turn (default coverage, then
`--cuda-graph-max-bs-decode 128`) and sweeps request rates 16 through 72 against
each, twice. About one hour.

The rates bracket a ceiling of **48**, not the 24 the old rung gave. What you are
looking for: median TPOT roughly flat across rates on the wide config, and a sharp
rise on the default config once the equilibrium running batch passes 48.

```bash
python scripts/sglang_cudagraph_cliff.py --phase bimodality
```

Fixed rate (62, just above the ceiling), three run lengths (500, 1000, 2000
prompts), three repeats each, all on the default configuration. About 20 minutes.

**Do not shrink `--bimodal-prompts` to save time.** Run length is the independent
variable of this phase. Shrinking it deletes the experiment.

Both phases append to `results/sglang_cliff.csv`. Each row carries its own server
startup time, VRAM after ready, the full capture list, and the resolved
`max_running_requests`, so no row depends on you remembering which server it came
from, and no row can confuse a ladder ceiling with a request-pool clamp.

If a run dies, check `logs/sglang_cliff/server_*.log`. The most common failures are
the port not being free from a previous run, and the model download timing out.

## Step 5: read the results honestly

Four questions, in order:

1. **Does the cliff still reproduce at 48?** Compare median TPOT, default against
   wide, at each rate. Report the rate at which they diverge. If they never
   diverge, #37898 accidentally covered the small-model case on this rung, and
   that is a real finding you report as-is.
2. **Was the ceiling the ladder or the request pool?** Check `cuda_graph_max_bs`
   against `max_running_requests` on every row. If they are equal and small, you
   measured the clamp, not the ladder, and the run needs redoing with a larger
   pool.
3. **What did the wider coverage cost?** Do **not** answer this with
   `vram_after_ready_mib` alone. It will show roughly zero and you will conclude
   the widening was free. It is not: `mem_fraction_static` derives from
   `reserved_mem = chunked_prefill_size * 1.5 + max_bs * 2`, so a larger `max_bs`
   shrinks the KV pool by about what the graph buffers gain, and the total
   footprint stays flat by construction.

   Diff `max_total_num_tokens` and `graph_capture_mib` instead. On the 4090 run
   that gave a KV pool shrinking by 14,110 tokens (165.4 MiB at 12,288 B/token)
   against graph buffers growing 143.4 MiB, versus the tree's predicted
   `(128 - 48) * 2 = 160` MB. The prediction is accurate; the cost is just paid
   in KV capacity and 0.5 s of capture time rather than VRAM.
4. **Is the bimodality real?** Pick the rate from the sweep first, near where the
   two configs actually diverge. Choosing it naively as "just above the ceiling"
   can land past saturation, where both configs are queue-bound and you measure
   queue growth instead of bistability. That is what happened on the 4090 run:
   rate 62 was chosen before the sweep showed divergence at 32.

   Real bistability means short runs stay fast and long
   runs stay slow across all three repeats. A warmup transient means the long-run
   medians land between the two extremes with wide spread across repeats. Say which
   one you saw. "The bimodality did not reproduce" is a perfectly good finding and
   you report it either way.

Do not smooth over a result that disagrees with the reporter. `docs/profiling.md`
reports a 1.4% anomaly it could not fully explain, and that is the standard here.

## Step 6: comment on the issue

Write the comment fresh, against your actual numbers. It should do four things:

1. State the mechanism from the code, with the current file and line, and get the
   ladder-versus-request-pool distinction right.
2. Report the 4090 measurement on the post-#37898 rung.
3. Note that #37898 raised this rung using 9B-class models only, and that model
   size still does not enter the ladder.
4. Ask which of the reporter's three options the maintainers want first.

That last question is the point. It converts a cold PR into an invited one, and it
is the step most first-time contributors skip.

Address it to the thread, but be aware BBuf is the one who just touched this code.
Then wait for a reply before writing code. If nobody answers in about a week, ping
once, politely, with the reproduction attached.

## Step 7: the PR

**The startup warning is already taken.** PR #33900, "Warn once when decode batch
exceeds the largest captured CUDA graph shape", was opened by KeMaSF on
2026-08-06 and has sat with zero comments and zero reviews since. Do not write a
competing one. If you want that change to land, the useful contribution is
getting eyes on #33900, which is what the 4090 comment asks for.

That leaves the ladder heuristic itself, direction 3 in the issue: make the choice
account for the reachable batch, or at least model size, rather than device memory
alone. It is a bigger change than the warning and it has the circularity problem
(`max_bs` is fixed before the KV pool exists, so it cannot simply be derived from
the pool). Do not start it without a maintainer saying they want it. The 4090
measurement is the argument for it, and #37898 shows the ladder is actively
maintained, so there is someone to ask.

If a maintainer does invite the change, scope the first PR narrowly and keep the
heuristic change separate from any refactor.

```bash
cd /workspace/sglang
git checkout -b warn-cuda-graph-coverage
pip install pre-commit
pre-commit install
```

Make the change, then before pushing:

```bash
pre-commit run --all-files      # re-run if it fails the first time, it applies fixes
pytest test/registered/unit/ -v
```

Add tests under `test/registered/unit/` mirroring the path of whatever you changed
under `python/sglang/srt/`. This is expected of any PR touching `srt`, and a PR
without them sits.

PR body: the before/after table, the card, the model, the exact commands, and the
VRAM and startup cost. You already write this way. The PR body is the same artifact
as a section of `docs/profiling.md`, just shorter.

```bash
git push origin warn-cuda-graph-coverage
```

Open the PR against `sgl-project/sglang` `main` and link the issue with
"Closes #33483" only if the PR fully closes it, which the warning alone does not.
Use "Refs #33483" instead.

## What the 4090 run actually found

Run of 2026-09-08, sglang `main` @ `141febf3` (includes #37898), RTX 4090 24GB,
Qwen2.5-0.5B-Instruct, tp=1. Raw data in `results/sglang_cliff.csv`, server logs
in `logs/sglang_cliff/`.

The ladder resolved to `max_bs=48` with `max_running_requests=4096`, so the
ceiling was the ladder and not the request pool. **The cliff reproduces on the
raised rung**: at rate 32, median TPOT was 30.6 ms on the default against 5.3 ms
with `--cuda-graph-max-bs-decode 128`, a 5.8x gap.

Two results needed a second pass to read correctly, and both are the reason this
runbook now says what it says:

- **Above rate 48 the two configs reconverge.** Throughput plateaus near
  9.7k tok/s in both, so past saturation TPOT is dominated by queueing and graph
  coverage is masked. The cliff bites *below* saturation, which is the load region
  operators actually run in. This is a stronger framing than "it diverges
  everywhere", not a weaker one.
- **Widening looked free and is not.** See Step 5 question 3.

The bimodality did not reproduce: at rate 62, median TPOT rose monotonically with
run length (31 -> 54 -> 82 ms) and was tight within each length. Given the sweep,
62 was past saturation, so this measured queue growth rather than a second
attractor. Reported as a null result.

## Step 8: capture it

Whatever the outcome, the reproduction becomes a short post in `blog/` in the style
of `docs/profiling.md`: the claim, the method, the number, the limitation. Commit
`results/sglang_cliff.csv` alongside it. A merged PR plus a public measurement is
the artifact; the measurement is the half an admissions committee can read.

## Step 9: destroy the instance

Storage bills while stopped. Push everything first.
