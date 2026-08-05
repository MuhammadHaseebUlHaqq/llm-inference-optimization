# Runbook: first SGLang contribution, issue #33483

Target: https://github.com/sgl-project/sglang/issues/33483

The claim under test is that SGLang's default decode CUDA-graph coverage (`max_bs`)
is chosen from device memory alone, so a small model on a large card gets graph
coverage far below the batch it actually reaches, and decode falls off a cliff into
eager execution above that batch.

The mechanism is already confirmed by reading the code (see "The diagnosis" below).
What is missing from the issue, and what this runbook produces, is an independent
reproduction on a different memory tier.

Budget: about 2 hours of GPU time, under $1 on a rented 4090.

---

## The diagnosis (already done, read this before you start)

`ServerArgs._handle_gpu_memory_settings` in `python/sglang/srt/server_args.py`
(around line 4602) picks decode `max_bs` from a ladder over device memory:

| device memory | chunked_prefill_size | decode max_bs (tp<4 / tp>=4) |
| --- | --- | --- |
| < 20 GB | 2048 | 8 |
| < 35 GB | 2048 | **24 / 80** |
| < 60 GB | 4096 | **32 / 160** |
| < 90 GB | 8192 | 256 / 512 |
| < 160 GB | 8192 | 256 / 512 |
| else | 16384 | 512 |

Model size never enters. The reporter's L40 (46 GB) lands on the 32 rung. A 4090
(24 GB) lands on the 24 rung. Both run a 0.5B model whose KV pool leaves room for a
running batch in the hundreds.

Two supporting facts, both from the same file:

- `reserve_for_graph_mb()` (line ~4877) charges `max_bs * 2` MB for graph buffers.
  Widening 32 to 128 predicts 192 MB. The reporter measured 0.22 GB. The tree's own
  cost model is already accurate here and already says the cost is small.
- Deriving `max_bs` from the KV pool is circular as written:
  `max_bs -> reserve_for_graph_mb -> mem_fraction_static -> KV pool -> reachable batch`.
  `max_bs` is fixed before the pool exists. This is the reason the obvious fix is
  not already in place, and any PR has to deal with it.

---

## Step 1: rent the box

Vast.ai, on-demand, one GPU.

- **GPU: 1x RTX 4090 (24 GB).** Chosen deliberately: it sits on a *different* rung
  (24) than the reporter's L40 (32). Reproducing on the same rung would only confirm
  their number. Reproducing on a different rung shows the problem is the ladder
  itself, not one badly chosen constant. That is a stronger contribution.
- **Disk: at least 60 GB.** SGLang from source plus its wheels plus the model.
- **Image: an NVIDIA NGC PyTorch image**, same as the Phase 1 box.
- Note the exact GPU name, VRAM, driver, and CUDA version from `nvidia-smi`. You
  will quote them in the issue comment, and `docs/profiling.md` is the reminder of
  why: two nominally identical cards differed by 1.22x.

Storage is billed while the instance is stopped. Destroy it when you are done.

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
reporter.

Install from source. This is slower than `uv pip install sglang` but you need the
source tree anyway to change `server_args.py`, and having two SGLangs installed is
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
python -m sglang.launch_server --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --attention-backend flashinfer --port 30000
```

Watch the startup log for the line reporting the CUDA-graph capture batch sizes.
**On a 4090 it should show a ceiling of 24.** If it does not, the ladder has moved
and every number below needs rechecking. Ctrl-C once you have seen it.

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

This launches two servers in turn (default coverage, then `--cuda-graph-max-bs-decode 128`)
and sweeps request rates 8 through 32 against each, twice. About one hour.

What you are looking for: median TPOT roughly flat across rates on the wide config,
and a sharp rise on the default config once the equilibrium running batch passes 24.

```bash
python scripts/sglang_cudagraph_cliff.py --phase bimodality
```

Fixed rate, three run lengths (500, 1000, 2000 prompts), three repeats each, all on
the default configuration. About 20 minutes.

**Do not shrink `--bimodal-prompts` to save time.** Run length is the independent
variable of this phase. Shrinking it deletes the experiment.

Both phases append to `results/sglang_cliff.csv`. Each row carries its own server
startup time, VRAM after ready, and the capture ceiling read back out of the server
log, so no row depends on you remembering which server it came from.

If a run dies, check `logs/sglang_cliff/server_*.log`. The most common failures are
the port not being free from a previous run, and the model download timing out.

## Step 5: read the results honestly

Three questions, in order:

1. **Does the cliff reproduce on the 24 rung?** Compare median TPOT, default against
   wide, at each rate. Report the rate at which they diverge.
2. **What did the wider coverage cost?** `vram_after_ready_mib` and
   `startup_seconds`, wide minus default. Check it against the tree's own prediction
   of `(128 - 24) * 2 = 208` MB. If the prediction holds on a second card too, that
   is a genuinely useful result for the PR.
3. **Is the bimodality real?** Real bistability means short runs stay fast and long
   runs stay slow across all three repeats. A warmup transient means the long-run
   medians land between the two extremes with wide spread across repeats. Say which
   one you saw. "The bimodality did not reproduce" is a perfectly good finding and
   you report it either way.

Do not smooth over a result that disagrees with the reporter. `docs/profiling.md`
reports a 1.4% anomaly it could not fully explain, and that is the standard here.

## Step 6: comment on the issue

Draft is in the scratchpad (`issue-33483-comment.md`); fill in the card name and
your numbers. Post it from your own account.

The comment does three things: confirms the mechanism from the code, adds the cost
number from a second card, and asks the maintainers which of the reporter's three
options they want first. That last question is the point. It converts a cold PR
into an invited one, and it is the step most first-time contributors skip.

Then wait for a maintainer reply before writing code. If nobody answers in about a
week, ping once, politely, with the reproduction attached.

## Step 7: the PR

Scope it to the **startup warning** first. It runs after pool allocation, where the
reachable batch is actually known, so it sidesteps the circularity entirely, and it
is small enough to review in one pass. The heuristic change is a second PR.

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

## Step 8: capture it

Whatever the outcome, the reproduction becomes a short post in `blog/` in the style
of `docs/profiling.md`: the claim, the method, the number, the limitation. Commit
`results/sglang_cliff.csv` alongside it. A merged PR plus a public measurement is
the artifact; the measurement is the half an admissions committee can read.

## Step 9: destroy the instance

Storage bills while stopped. Push everything first.
