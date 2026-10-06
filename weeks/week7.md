# Week 7: the prerequisites I actually skipped, then close the Phase 2 gate

Written 2026-09-19. This file replaces the Phase 2 plan in `weeks/week4.md` and
`weeks/week5.md`, which assumed input work that did not happen. See the
correction notes at the top of both files.

## Why this plan is shorter than weeks 4 and 5

Weeks 4 and 5 budgeted about 40 hours of lectures (CS149 5 and 6, GPU MODE 2, 3,
4, 5 and 8, PMPP chapters 4 to 6). That was written before the 21-day roadmap
notes existed. Reading Day 3 of those notes changes the estimate.

Day 3 (`docs/lecture-src/week1_day1-4_condensed.html`) already covers, with
worked examples on this exact card:

- SM, thread, warp, block, tensor core, warp divergence
- the memory hierarchy with latencies (registers ~1 cycle, shared/L1 ~30,
  L2 ~200, VRAM ~400 to 600)
- coalescing, including the stride-64 example: 128 bytes wanted, 1024 fetched,
  12.5% useful, an 8x effective-bandwidth loss
- occupancy, including the correction that hiding latency is the goal and high
  occupancy is only a means to it
- arithmetic intensity and the ridge point, 142 FLOP/byte on the 3060

Two of the gate's three lenses are therefore already held at the concept level.
Only two real gaps remain:

1. **CUDA code has never been read.** Day 3 is prose about hardware. The gate
   asks for a diagnosis of an actual `__global__` function with `threadIdx`,
   `__shared__` and `__syncthreads()`.
2. **Shared-memory tiling as a technique.** Day 3 says shared memory is ~30
   cycles instead of ~400. It does not walk through loading a tile
   cooperatively, syncing, computing, and moving on. That walkthrough is the
   whole of the naive-vs-tiled comparison the gate asks for.

So the input is about 6 hours, not 40. Lecture 5 below covers the tiling
ground that PMPP chapter 5 was budgeted 4 hours for, which is where the rest of
the saving comes from.

## Block 1: the prerequisites, about 6 hours, 2026-09-19 to 2026-09-27

> **Video links verified 2026-10-06.** The GPU MODE README carries no YouTube
> links, so each video below was matched by fetching it and checking its title.
> The lecture numbering comes from that README. Durations are from secondary
> write-ups, not from YouTube, so treat them as approximate.

Do these in order. Each one ends with notes committed to
`docs/notes-phase2-gpu.md`, per the "no checkbox without a commit" rule.

- [ ] **Step 0, read CUDA code, about 1.5 hours.** GPU MODE Lecture 3,
      "Getting Started With CUDA for Python Programmers", Jeremy Howard.
      https://www.youtube.com/watch?v=nOxKexn3iBo (about 1h 17m)
      Notebook: the `lecture_003` folder in https://github.com/gpu-mode/lectures
      Starts with RGB to grayscale in pure Python, then the same kernel in CUDA,
      so `threadIdx`, `blockIdx` and the launch config arrive attached to
      something concrete. The book alternative is PMPP chapters 2 and 3, slower
      and more precise.
      The goal is reading a kernel, not writing one.

- [ ] **Step 1, tiling, about 1.5 hours.** GPU MODE Lecture 5, "Going Further
      with CUDA for Python Programmers", Jeremy Howard.
      https://www.youtube.com/watch?v=eUuGdh3nBGo (about 1h 5m)
      Notebook in `lecture_005`. It assumes lecture 3 first, so keep the order.
      Shared memory at 0:49, shared memory from Python at 12:00, dynamic shared
      memory at 18:41.
      This is the shared-memory tiled matmul built step by step, the same ground
      as PMPP chapter 5 (memory architecture and data locality). Video first is
      the better order with no prior CUDA reading. Read the chapter afterwards
      only if the video version stays slippery.

- [ ] **Step 2, the measured version, about 3 hours.** Simon Boehm, "How to
      Optimize a CUDA Matmul Kernel for cuBLAS-like Performance":
      https://siboehm.com/articles/22/CUDA-MMM
      Read kernels 1 through 3 carefully, skim the rest. Kernel 1 to kernel 2 is
      coalescing isolated, with a speedup attached. Kernel 3 is shared-memory
      cache-blocking. `CLAUDE.md` lists this as Phase 0 material, so this may be
      a re-read rather than new ground.
      PMPP gives the mechanism, Boehm gives the numbers. The gate wants both.

- [ ] **Reference, as needed.** NVIDIA CUDA C++ Best Practices Guide, "Coalesced
      Access to Global Memory", for the exact transaction rules when a specific
      claim needs checking. Look-up, not reading.

### Deliberately skipped

PMPP chapters 4 and 6 (Day 3 already covers occupancy and coalescing to gate
standard), CS149 entirely (general parallel computing, good context, not what
the gate asks), and the GPU MODE performance checklist lecture:
GPU MODE Lecture 8, "CUDA Performance Checklist", Mark Saroufim,
https://www.youtube.com/watch?v=SGhfUhlowB4. Save that one for after the gate is
written, where it works better as a self-check than as input.

## Block 2: write the gate, about 7 hours, 2026-09-28 to 2026-10-04

One new file, `docs/gate-phase2.md`, two sections. No GPU needed: the traces are
already committed and this runs on the laptop.

- [ ] **Section 1, a kernel I did not write, about 4 hours.** Naive against
      tiled matmul. Numbers, not adjectives:
      how many times each element of B is read from global memory in the naive
      version (N times, once per output row) against the tiled version (once per
      tile, so a 32x32 tile is 32x fewer global reads); the arithmetic intensity
      of each in FLOP/byte; and where each lands against the 3060 ridge point of
      142 FLOP/byte from Day 3.

- [ ] **Section 2, a kernel from my own trace, about 3 hours.** The subject is
      already picked. Top kernels in `results/trace_hf_decode.json.gz`, over 5
      decode steps:

      | kernel | calls | total |
      | --- | --- | --- |
      | `gemv2T_kernel_val<..., 128, 16, 4, 4, ...>` | 285 | 35.4 ms |
      | `cutlass_80_wmma_tensorop_f16_s161616gemm_f16_16x16_128x2_tn_align8` | 140 | 15.9 ms |
      | `pytorch_flash::flash_fwd_splitkv_kernel` | 140 | 2.1 ms |
      | `at::native::elementwise_kernel` (binary) | 560 | 1.8 ms |

      `gemv2T_kernel_val` is cuBLAS's matrix-vector kernel and it costs more
      than every other kernel combined. It is there because at batch 1 the
      decode matmuls degenerate to matrix-vector: [1, 1536] against
      [1536, 8960]. Each weight is read once and used once, so arithmetic
      intensity is about 1 FLOP/byte against a ridge point of 142. There is no
      tile to hoist into shared memory because there is no reuse to exploit.
      Cross-check against `docs/profiling.md`: while busy, HF hits 202.7 GB/s of
      291.5 GB/s achievable, 70%, which is close to the best a memory-bound
      kernel can do. The waste is not in the kernel, it is the 62.3% of wall
      time the GPU spends idle waiting on CPU launches. The contrast worth
      naming is the cutlass tensor-core GEMM: a real matmul with reuse, which is
      why cuBLAS picked a tiled tensor-core kernel for it instead of a GEMV.

- [ ] **Housekeeping, about 1 hour.** Add the "no checkbox without a commit"
      rule to `CLAUDE.md` (owed since week 6 Part D), and make sure
      `docs/notes-phase2-gpu.md` from Block 1 is committed.

## Running in parallel, not in the repo

- [ ] **Book the IELTS date.** Not research it, book it. This is more urgent
      than the gate. Supervisor outreach is due now, the HAT is mid-October, and
      the Canada, Finland and Switzerland deadlines cluster December to January.
      It is the one item where a slipped week cannot be recovered later.
- [ ] **One IELTS Writing Task 2 under timed conditions.** Forty minutes, no
      edits after. Tells you whether Writing needs weekly hours or almost none.

## Checkpoint: done on 2026-10-04

Phase 2 closes, and Phase 3 (writing Triton kernels) opens, when all three hold:

1. `docs/notes-phase2-gpu.md` is committed and covers steps 0 to 2.
2. `docs/gate-phase2.md` is committed with both sections, and every claim in it
   carries a number.
3. `docs/profiling.md` is committed. Already true since 2026-08-27.

If Block 1 slips, Block 2 slips with it and the checkpoint moves to 2026-10-11.
Do not write the gate on top of skipped reading: a shaky gate makes Phase 3
slower, not faster, which is the entire reason the gate exists.
