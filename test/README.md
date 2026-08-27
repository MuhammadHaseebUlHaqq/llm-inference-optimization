# Diagnostics

Not a test suite. There are no assertions here and nothing runs in CI. These are
one-off probes written to answer a specific question about a measurement, kept
because the question will come back on different hardware.

| file | the question it answered |
| --- | --- |
| `probe_weights_vram.py` | why did the HF baseline report 4737 MiB of resident weights when the fp16 floor for Qwen2.5-1.5B is ~2944 MiB? |

`probe_weights_vram.py` loads the model exactly the way `scripts/baseline_hf.py`
does, then breaks the live GPU allocation down by (param or buffer, dtype). A
large float32 row means a cast did not land; a failed tied-embeddings check means
a second copy of the head. On this repo it found neither: the gap was
transformers 4.44.x registering per-layer rotary cos/sin tables sized to
`max_position_embeddings`, 1.79 GB duplicated across 28 layers and never read
past the current sequence length. That is why `requirements.txt` pins 4.46.3.

Run it before trusting a weights number on any new box. A resident-weights figure
that does not match parameters times bytes is the measurement telling you
something, not rounding.
