# Correspondence on SGLang issue #33483

https://github.com/sgl-project/sglang/issues/33483

Kept in the repo on purpose. The first draft of the opening comment lived in a
scratchpad directory and was gone by the next session, so the record of what was
actually said upstream now lives with the data it refers to.

---

## Comment 1, posted 2026-09-08

Reported the 4090 reproduction on the raised `<35GB` rung: `max_bs=48`,
`max_running_requests=4096`, cliff reproduces at rate 32 with a 5.8x TPOT gap.
Flagged three things as unexplained: the divergence appearing below the ceiling,
the reconvergence above rate 48, and a VRAM cost that matched neither the tree's
prediction nor the reporter's L40 measurement. Also flagged PR #33900 as already
proposing the startup warning and asked whether the ladder heuristic itself
(direction 3) is still wanted.

Full text is in the issue thread.

---

## Comment 2, follow-up (draft, not yet posted)

Corrects two of the three flagged items. Both turned out to be measurement or
framing errors rather than real anomalies. Post as-is; it is plain markdown.

---

Follow-up on my numbers above: both things I flagged as unexplained have
explanations, and one of them means I reported the cost wrong. Correcting the
record.

**1. The widening is not free. I measured the wrong variable.**

I compared `vram_after_ready` between configs, saw a 2 MiB difference, and said
the cost didn't match the prediction. That diff can only ever be about zero.
`mem_fraction_static` derives from
`reserved_mem = chunked_prefill_size * 1.5 + max_bs * 2`, so raising `max_bs`
shrinks the KV pool by roughly what the graph buffers gain and the total
footprint stays flat by construction.

From the same server logs, the right variables:

| | default (48) | wide (128) | delta |
|---|---|---|---|
| `max_total_num_tokens` | 1,330,288 | 1,316,178 | **-14,110 tokens** |
| graph capture `mem usage` | 0.13 GB | 0.27 GB | **+143.4 MiB** |
| graph capture `elapsed` | 2.19 s | 2.69 s | +0.50 s |
| total VRAM after ready | 19392.8 MiB | 19390.8 MiB | -2.0 MiB |

KV cell size on this model is 12,288 B/token (`K size: 7.61 GB` + `V size: 7.61
GB` over 1,330,288 tokens), so 14,110 tokens is **165.4 MiB** of pool given up
against **143.4 MiB** of graph buffers gained.

Predicted by the tree's own model: `(128 - 48) * 2 = 160 MB`.

So `reserve_for_graph_mb` is accurate on this card too. The cost of widening
48 -> 128 is about 1.1% of the KV pool and half a second of startup, not extra
VRAM. That is a cheaper and more defensible cost than "0.22 GB of VRAM" makes it
sound.

**2. The reconvergence above rate 48 is saturation, not a limit on the effect.**

Throughput from the same runs:

| rate | default tok/s | wide tok/s |
|---|---|---|
| 16 | 4142 | 4141 |
| 32 | **6979** | **7887** |
| 48 | 9001 | 9088 |
| 72 | 9675 | 9750 |

The card tops out near 9.7k tok/s. At rate >= 48 both configs sit at that
ceiling, so TPOT is dominated by queueing delay and is identical in both. Graph
coverage is masked once offered load exceeds what either config can serve.

At rate 32 the default is both slower per token *and* lower throughput (6979 vs
7887 tok/s). It fell off the captured-shape edge, decode slowed, the running
batch grew, and it stayed there. That is the absorbing state from the issue body,
reproduced on the raised rung.

So the cliff bites **below saturation**, which is the load region operators
actually run in.

**3. That also explains my bimodality null.**

I picked rate 62 as "just above the 48 ceiling" before the sweep showed
divergence at 32. On this card 62 is well past saturation, so all three run
lengths were measuring queue growth under sustained overload, not a second
attractor. My null result stands as reported but it was not a fair test of
@KeMaSF's claim. A proper test here would fix the rate near 32.

Harness and raw data updated accordingly:
https://github.com/MuhammadHaseebUlHaqq/llm-inference-optimization

---

## Open threads

- Waiting on a maintainer reply about direction 3 (the ladder heuristic).
  Ping once, politely, after about a week of silence.
- PR #33900 (KeMaSF's startup warning) is open with no review since 2026-08-06.
  Do not write a competing PR. Getting eyes on that one is the useful move.
