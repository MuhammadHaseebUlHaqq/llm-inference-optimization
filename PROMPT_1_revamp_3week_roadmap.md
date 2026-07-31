# PROMPT 1 — Rebuild the 3-Week LLM Inference Interview Roadmap, grounded in this repo

> **How to use this file:** paste the whole thing as your first message to Claude Code in the
> `llm-inference-optimization` repo. It is written as a self-contained brief — the agent there has
> no memory of the sessions where this material was originally produced, so everything it needs is
> restated below.

---

## 0. Who I am and what I am doing

I am preparing for **research master's admissions interviews in LLM inference optimization**
(systems side: kernels, serving, memory, quantization). The bar I am preparing to is not
recognition, it is **defence** — I need to be able to derive the key equation on a whiteboard and
name the one experiment that proves each claim, unaided, out loud, in under 90 seconds per topic.

I have **three weeks**. That is a hard constraint. This is a triage document, not a curriculum.

**This repo (`llm-inference-optimization`) is my single most important asset in those interviews.**
It contains my own measured results. Routing an abstract question back through my own measurement
beats a recited paper every time. The roadmap you write must be built *around* this repo, not
alongside it.

## 1. What I want you to produce

A single document: **a revised, re-grounded 3-week interview roadmap.**

Deliver it as a **detailed PDF** (see §5 for the exact format contract), saved into this repo's
root, plus the substance summarised in chat so I can react to it without opening the file.

## 2. Do this first — read the repo before you write anything

Before drafting a single line, go through this repository and build an inventory of **what I have
actually measured**. Specifically, look for and extract concrete numbers from:

- Any benchmark result files, `*results*.md`, CSVs, logs, notebooks, or plots.
- Profiling traces or profiler summaries (kernel-level timings, idle-time breakdowns).
- Batching sweeps (throughput/latency vs batch size).
- Memory / OOM experiments (KV-cache growth, the sequence length where it dies).
- HuggingFace-vs-vLLM comparisons.
- Any quantization experiments (NF4/int8/int4) and their measured effect.
- Hardware and model config actually used (GPU model, VRAM, bandwidth; model, dtype, context).
- README, notes, or TODO files describing what is done vs half-done.

Then **report the inventory back to me before or alongside the roadmap** as a short table:
`experiment → the headline number → the interview question it answers`.

This matters because the previous version of this roadmap was written from memory, in a different
repo, and cited results second-hand. From memory, the numbers I believe are in here are:

- Model **Qwen2.5-1.5B**, fp16, on an **RTX 3060** (12 GB).
- A **KV-cache OOM cliff at ~123,565 tokens** of total context.
- A **batching sweep with a throughput knee around batch 64**.
- **HF vs vLLM profiling: ~90% of vLLM's win came from removing idle/launch-bound time**, with
  only ~9% from faster kernels; baseline decode was ~62% idle.
- An **NF4 quantization result where weights halved but batch-1 decode got *slower*** (dequant
  cost ate the bandwidth win).

**Treat every one of those as unverified.** Confirm each against the actual files, correct the ones
that are wrong, and tell me explicitly which ones you could not find. Do not silently carry forward
a number you did not verify — I would quote it in an interview.

## 3. The syllabus this roadmap triages from

The full field was previously mapped into 11 layers. Priority markers: **★** = asked almost every
time, **◆** = strong differentiator, **○** = frontier awareness only.

| Layer | Topic | Tier |
|---|---|---|
| 1 | Transformer architecture: attention, MHA/MQA/GQA/MLA, RoPE, RMSNorm, SwiGLU, residual stream | ★ |
| 2 | Inference arithmetic: param counting, 2×params FLOPs, prefill vs decode, roofline, arithmetic intensity, MBU/MFU, latency-throughput Pareto | ★ |
| 3 | KV cache: size formula, fragmentation, PagedAttention, RadixAttention/prefix caching, eviction (StreamingLLM/H2O), KV quantization, offload | ★ |
| 4 | GPU architecture: SMs/warps, memory hierarchy, coalescing, occupancy, bank conflicts, tensor cores, latency hiding | ★ |
| 5 | CUDA + GEMM optimization ladder, Triton, Nsight Compute, CUDA graphs | ★ |
| 6 | Landmark kernels: FlashAttention 1/2, online softmax, flash-decoding/split-KV, PagedAttention as a kernel | ★ |
| 7 | Serving + scheduling: continuous batching (Orca), chunked prefill (Sarathi), P/D disaggregation, preemption, TTFT/TPOT/p99/goodput | ★ |
| 8 | Quantization: GPTQ, AWQ, SmoothQuant, LLM.int8() outliers, NF4/QLoRA, KV quant | ◆ |
| 9 | Speculative decoding: draft-verify, lossless rejection sampling, acceptance rate, Medusa/EAGLE | ◆ |
| 10 | Parallelism: tensor parallel (Megatron), pipeline, expert, sequence/ring | ◆ |
| 11 | Synthesis + staying current: Lilian Weng survey, gpt-fast, MoE basics, structured decoding, multi-LoRA | mixed |

**All ○ items are cut from the 3-week plan entirely** — MoE, Ring Attention, multi-LoRA, structured
decoding, FP8, MLA math beyond one sentence, FlashAttention-3, DistServe/Mooncake. If asked about
those in an interview, "aware of it, haven't gone deep" is a correct and well-calibrated answer.

## 4. The previous roadmap — the baseline you are revising

This is the plan as it currently stands. **Do not just reformat it.** Revise it against what you
actually find in the repo (§2): move things earlier or later, cut what my measurements already
cover, and add anything the repo shows I am weak on. Say explicitly what you changed and why.

**Week 1 — finish the foundations already in progress**
- *Days 1–2:* Attention cold from memory (incl. the √d variance argument), RoPE mechanism, MHA vs
  MQA vs GQA and why GQA shrinks KV but not weights. Build nanoGPT by hand if not already done
  (~3–4 hrs, highest ROI single item on the list).
- *Days 3–4:* Close out the in-progress GPU architecture pass — SMs/warps, memory hierarchy,
  coalescing, occupancy. Reading fluency is the goal, not a from-scratch kernel this week.
- *Days 5–6:* Roofline, arithmetic intensity, MBU/MFU. **Pope et al., "Efficiently Scaling
  Transformer Inference"** — worth more than everything else in this area combined for an academic
  interview. Read it twice.
- *Day 7:* Buffer. (No Sundays — that was the agreed schedule.)
- *Exit bar:* whiteboard attention shapes through one decode step, the roofline ridge point, and
  why decode is memory-bound while prefill is compute-bound — derived, not recited.

**Week 2 — KV cache, FlashAttention, serving (non-negotiable in full)**
- *Days 1–2:* KV cache size formula; PagedAttention (block tables, why fragmentation dies);
  RadixAttention/prefix caching in one paragraph. Rehearse the OOM experiment **out loud** — it is
  the best answer I have to "walk me through a memory problem you've debugged."
- *Days 3–4:* FlashAttention properly derived — the online-softmax running-max/rescale update, why
  it is IO-awareness and not a FLOPs reduction, and why it does nothing for batch-1 decode
  (split-KV / flash-decoding is the answer). Single most likely deep-dive question in the interview.
- *Days 5–6:* Continuous batching (Orca) and chunked prefill (Sarathi) — mechanism and the problem
  each solves. Metric vocabulary: TTFT, TPOT/ITL, p50/p95/p99, goodput. Map my batching-sweep
  results onto this and practise explaining the knee in Orca/Sarathi language.
- *Day 7:* Buffer.

**Week 3 — quantization, speculative decoding (light), synthesis**
- *Days 1–2:* Quantization, one paragraph of mechanism each: GPTQ (Hessian-based rounding) vs AWQ
  (activation-aware scaling), SmoothQuant's outlier migration, LLM.int8()'s outlier story, and why
  weight-only quant is fundamentally a *bandwidth* play. Anchor on my own NF4 result.
- *Day 3:* Speculative decoding — draft-verify, why rejection sampling stays lossless (plain
  language is enough, skip the proof), one sentence each on Medusa and EAGLE.
- *Day 4:* Tensor parallelism only — split heads/FFN, two all-reduces per layer, why it wants
  NVLink. Pipeline parallelism gets one sentence ("throughput not latency, has bubbles").
- *Days 5–6:* Synthesis. For every layer: the concept, the equation, and the one experiment or
  paper result that proves it. Rehearse the four anchor answers.
- *Day 7:* Buffer.

**The four anchor answers** (almost any question routes back to one of these):
1. The FlashAttention derivation.
2. The KV-cache OOM story.
3. The batching knee.
4. The HF-vs-vLLM profiling result.

→ **Verify all four against the repo and rewrite them as quotable, number-bearing paragraphs.**

**The habit that matters most:** end prep on each topic by explaining it out loud, unaided, in under
90 seconds. If I can't, re-derive it from scratch rather than rereading. That is a far better signal
than "did I finish the reading list."

## 5. Format contract — this is not optional

Deliver as a **PDF** written the way a researcher or professor writes lecture notes:

- **Every claim carries a real number**, and the arithmetic is carried through visibly — not just
  formulas and prose. Wherever a number can come from *this repo's measurements*, use mine rather
  than a textbook's.
- **Clickable links per section**, inline, to the actual papers/blogs/videos (arXiv links, author
  names, titles). The previous version did this well — keep that.
- Clear day-by-day structure with a stated **goal per week** and an **exit bar per week** ("by end
  of week N I can whiteboard X").
- An explicit **"what to deprioritize"** section — naming what I am deliberately not learning and
  why that is the right call in a 3-week window.
- Tables where tables are clearer than prose.

**Method:** write the content as HTML, then render with headless Chrome —
`chrome.exe --headless --disable-gpu --no-pdf-header-footer --print-to-pdf=...` — and save the PDF
into this repo's root. Also give me the substance in chat; the PDF is a reference artifact, not a
replacement for explaining it to me.

## 6. What I explicitly do not want

- A reformat of §4 with nicer headings. If the repo shows a gap, change the plan.
- Numbers you did not verify against the files in this repo.
- Topics from the ○ tier smuggled back in "for completeness."
- A reading list without a per-item statement of *what interview question it lets me answer*.

## 7. Start here

1. Inventory the repo (§2) and report the `experiment → number → question` table.
2. Tell me which of the five remembered results (§2) you confirmed, corrected, or could not find.
3. Then produce the revised 3-week roadmap PDF per §4–§5, with a short changelog of what you moved,
   cut, or added relative to the baseline, and why.
