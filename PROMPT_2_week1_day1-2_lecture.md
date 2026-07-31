# PROMPT 2 — Week 1, Days 1–2 lecture notes: Attention, RoPE, and MHA→MQA→GQA

> **How to use this file:** paste the whole thing as a message to Claude Code in the
> `llm-inference-optimization` repo, *after* Prompt 1 has been run (so the agent already has the
> repo inventory in context). It is self-contained enough to work standalone if needed.

---

## 0. Context

I am preparing for **research master's admissions interviews in LLM inference optimization**, on a
three-week timeline. This is the **Week 1, Days 1–2** unit of that plan: the transformer
architecture layer — attention derived from scratch, RoPE, and the MHA → MQA → GQA progression as
an *inference-motivated* story (shrinking KV bytes per token).

The bar is **defence, not recognition**. I must be able to reproduce every derivation on a
whiteboard and quote concrete numbers from memory. So the notes must be built out of arithmetic I
can redo by hand, not summaries I can only nod along to.

## 1. What to produce

One document: **detailed lecture notes for Days 1–2**, delivered as a **PDF** in this repo's root,
with the substance also given in chat.

Target depth: this should be a substantial document (the previous version ran ~29 PDF pages).
Do not compress it. Length is not the goal, but completeness of derivation is, and that costs pages.

## 2. Ground it in this repository

Before writing, read this repo and pull out the **actual model config and measured numbers** to use
as the running example throughout: the model and its config (hidden size, layers, heads, KV heads,
head dim, vocab, intermediate size, dtype), the GPU (name, VRAM, memory bandwidth, peak FLOPS), and
any profiling/latency/memory measurements.

Use **my measured numbers, not textbook ones**, everywhere they exist — that way the notes double
as interview ammunition. Every architectural claim should terminate in a number I have personally
measured or can derive from my own config.

From memory the running example is **Qwen2.5-1.5B in fp16 on an RTX 3060**, with
`d_model = 1536`, `28 layers`, `12 query heads`, `2 KV heads`, `head_dim = 128`,
`vocab = 151,936`, `intermediate = 8960`, total params `1,543,714,304`, weights ≈ `2,944 MiB`.
**Verify all of this against the repo's config before using it.** Correct anything that is wrong and
tell me what you corrected.

## 3. Required structure

Follow this outline. It worked well; keep it unless the repo contents give you a reason to change
something, in which case say so.

### PART 0 — Fundamentals from the ground up (assumes nothing)

Every term used in Part 1 must be defined here first, so nothing in Part 1 is unexplained.

1. What a language model actually does — next-token distribution, autoregressive loop, and the two
   consequences: generation is inherently sequential, and each step re-runs over a sequence that
   grows by one (which is *why the KV cache exists*).
2. Tokens — subword/BPE, ids as meaningless indices, the ~4 chars/token rule of thumb.
3. Embeddings — the `[V, d_model]` lookup table; **worked example: how many params and MiB that
   table costs, and what fraction of the whole model it is**. Note the lookup is pure memory
   traffic with zero arithmetic — the first instance of the theme of the entire field.
4. The dot product — `a·b = |a||b|cosθ`, worked with unit vectors to show alignment → large
   positive, perpendicular → zero, opposite → negative. This is *why* attention uses dot products.
5. Matrix multiplication — `[M,K] × [K,N] → [M,N]`, worked fully by hand on a small example.
   Define **GEMM vs GEMV** and flag that the entire performance story reduces to which one you are
   running.
6. Linear layers — `y = xW + b`; weights are fixed and *shared across every token, request and
   batch element*, so weight memory is a constant. That constancy is exactly why batching helps.
7. Counting FLOPs — `2·M·K·N`; **derive the "2 × params per token" rule numerically** on one real
   projection and show the ratio comes out to exactly 2.0, then explain *why* (each weight touched
   once, one multiply + one add). State the caveat: linear layers only, ignores the O(seq²)
   attention term.
8. Nonlinearity — why stacked linears collapse; SiLU evaluated at several points; SwiGLU's gated
   three-matrix form.
9. RMSNorm — the formula, **worked on a small vector, then verified by recomputing the RMS of the
   output and showing it equals 1**. Contrast with LayerNorm and say why modern models dropped the
   mean subtraction.
10. Residual connections and the residual stream — the gradient-path reason and the
    interpretability reason (blocks accumulate into a shared running representation).
11. The transformer block assembled — an ASCII diagram, plus the one-line division of labour:
    **attention moves information *between* tokens, the MLP processes each token independently**.
    Include the per-layer parameter split (attention vs MLP) and draw the consequence: most weight
    traffic is MLP, most systems complexity is attention.
12. Number formats — fp32/fp16/bf16/int8/int4 table with bytes and bit layout; fp16 vs bf16 in one
    sentence; and the **MB vs MiB trap** (~7% discrepancy that looks like a measurement bug),
    shown by expressing the same weight size three ways.
13. What a GPU is, in the three terms that matter — compute throughput, memory bandwidth, memory
    capacity, with this GPU's real spec numbers.
14. **Compute-bound vs memory-bound** — arithmetic intensity, the ridge point = FLOPS/bandwidth,
    and where decode sits (~1–2 FLOP/byte). This is the most important result in Part 0.
15. Prefill vs decode — the two phases, why prefill is a compute-bound GEMM and decode is a
    memory-bound GEMV.
16. Vocabulary checkpoint — a compact glossary of every term introduced.

### PART 1 — Attention, RoPE, and GQA

1. The model we will compute with — the config table, plus a **parameter-count verification that
   sums to the published total** (this is a favourite interview exercise).
2. What attention is actually for — the problem it solves, in plain language.
3. Q, K, V — the three projections, with shapes and the query/key/value intuition.
4. **The attention formula worked fully by hand** — a small concrete example (e.g. 3 tokens,
   d = 4), carrying real numbers through `QKᵀ`, the scaling, the softmax, and the `V` weighting.
5. **Why √d — the variance derivation.** Show that the dot product of two d-dimensional unit-variance
   vectors has variance d, so scores grow like √d, and softmax saturates. Then show numerically what
   saturation does to the distribution.
6. Softmax numerical stability — the max-subtraction trick, worked; and explicitly flag this as
   **the seed of FlashAttention's online softmax**, which Week 2 Days 3–4 will build on.
7. Causal masking — the −∞ trick and why.
8. Multi-head attention — shapes through the reshape/transpose, and the key point that
   **multi-head is free** (`n_heads × d_head = d_model`, same FLOPs and params, but the model can
   represent several relational patterns at once). Add the second-order benefit: smaller heads mean
   a smaller √d and per-head score matrices small enough to tile into SRAM — the property
   FlashAttention depends on.
9. **RoPE, derived and verified numerically.** Start from the problem: attention is
   permutation-equivariant, so position must be injected. Table of approaches (learned absolute,
   sinusoidal, RoPE) with the weakness of each. Then the mechanism: split `head_dim` into pairs,
   rotate each pair by `m·θᵢ`. **Then verify numerically** that the dot product between a rotated
   query at position m and a rotated key at position n depends only on `m − n` — actually compute
   both sides and show they match. One line on YaRN/NTK scaling for long-context extrapolation.
10. **MHA vs MQA vs GQA — the KV-cache arithmetic.** The size formula
    `2 × layers × kv_heads × head_dim × bytes × seq × batch`, then compute the per-token KV bytes
    under all three schemes for my actual model, and show the ratio. Make the punchline explicit:
    **GQA shrinks the KV cache but not the weights**, and state why (KV heads shrink; Q heads and
    all the projection matrices do not).
11. **A full decode-step trace** for my model — every operation in order with tensor shapes at
    batch 1, from token id to logits.
12. **Tying it to the roofline** — total bytes read per decode step, divide by this GPU's
    bandwidth, get the theoretical floor in ms/token and tok/s, then **compare against my actual
    measured decode rate** from the repo and explain the gap (this gap is where the rest of the
    three weeks lives).
13. **Self-test with answers** — a set of questions I should be able to answer unaided, answers
    given separately below them.

## 4. Format contract — this is not optional

- **Every concept backed by an actual worked numerical example**, with real numbers carried through
  the arithmetic step by step. Not formulas and prose. This is the single most important
  requirement — I have asked for it repeatedly.
- Worked examples should be visually set apart and labelled (the previous version used
  `WORKED EXAMPLE 0-A`, `0-B`, … and that worked well).
- **Clickable links per section** to the underlying sources.
- Monospace blocks for shape traces, arithmetic, and ASCII diagrams.
- Tables where a table beats prose.
- Written in the register of a researcher or professor writing lecture notes for themselves —
  precise, unhedged, no filler, no "in this section we will."

**Method:** write the content as HTML, render via headless Chrome —
`chrome.exe --headless --disable-gpu --no-pdf-header-footer --print-to-pdf=...` — and save the PDF
into this repo's root, named for the week/day and topic. Also deliver the substance in chat; the PDF
is a reference artifact, not a substitute for teaching it to me.

## 5. Source links to include

- Karpathy, *Let's build GPT from scratch* + the nanoGPT repo
- 3Blue1Brown, *Attention in transformers, visually explained*
- Vaswani et al., *Attention Is All You Need* — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Su et al., *RoFormer* (RoPE) — [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- EleutherAI, *Rotary Embeddings: A Relative Revolution*
- Ainslie et al., *GQA* — [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
- Shazeer, *Fast Transformer Decoding* (MQA) — [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)
- kipply, *Transformer Inference Arithmetic*
- Pope et al., *Efficiently Scaling Transformer Inference* — [arXiv:2211.05102](https://arxiv.org/abs/2211.05102)
- Milakov & Gimelshein, *Online normalizer calculation for softmax* —
  [arXiv:1805.02867](https://arxiv.org/abs/1805.02867) (forward reference for §Part 1.6)

## 6. What I explicitly do not want

- Formulas without numbers plugged in.
- Generic textbook constants where this repo has a real measurement.
- Skipping Part 0 because it "looks basic" — it is what makes Part 1 defensible rather than
  memorised.
- A summary of RoPE that does not numerically demonstrate the relative-position property.

## 7. After the document

End with a short **"can you answer this?"** block — the three or four questions an interviewer is
most likely to ask from this material, so I can self-check before moving to Days 3–4 (GPU
architecture). Suggested set:

- Walk through one decode step, layer by layer, with tensor shapes at batch 1.
- Why does GQA shrink the KV cache but not the weights?
- Why RoPE rather than learned absolute positions?
- Softmax needs the whole row for its denominator — so how can attention ever be tiled?
