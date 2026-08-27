# Docs index

Read in this order. Each writeup states the method before the numbers, and every
number traces back to a CSV in `results/` (mapping in `results/README.md`). Where
a doc here and a CSV disagree, the CSV wins.

## Plan and environment

| file | what it settles |
| --- | --- |
| `phase1.md` | the ordered task breakdown and the shared measurement definitions |
| `vastai.md` | the compute actually used. Beats `CLAUDE.md` and `phase1.md` on hardware |
| `notes-cuda-memory-and-timing.md` | why the harness synchronizes before every timer and tracks allocated separately from reserved |

## Results, in the order they were measured

| file | the experiment |
| --- | --- |
| `baseline-hf-results.md` | Task 1, the HF transformers control every later number is measured against |
| `baseline-vllm-results.md` | Task 2, the same workload through vLLM. The gap against Task 1 is the point |
| `oom-results.md` | Task 3, the headline: push context length until the 3060 dies, measured crash point against the analytical prediction |
| `batching-results.md` | Week 3, the first non-batch-1 experiment: throughput against batch size, and the two different plateaus |
| `profiling.md` | Week 6, why the HF-to-vLLM gap exists. Traces both engines through one parser instead of reasoning about it |

Read `profiling.md` last. It is the one that turns the earlier gap from an
observation into a mechanism, and it also carries the two-box result: two
nominally identical 3060s ran the same decode work 1.22x apart, which is the
reason every later comparison is pinned to one physical machine.

## Next phase

| file | what it is |
| --- | --- |
| `sglang-contribution-runbook.md` | step by step plan for the first SGLang PR, against issue #33483. The repro harness is `scripts/sglang_cudagraph_cliff.py` |

## Sources

`lecture-src/` holds the HTML for the condensed lecture PDFs in the repo root.
Re-render them with `scripts/render_lecture_pdfs.sh` rather than editing the PDFs.
