# Ai_inference

A hands-on curriculum for **LLM inference internals**: not training, not prompt
engineering — the actual mechanics of running a trained transformer efficiently.

You already know what attention and a forward pass are. This project is about
the layer below that: how real inference engines (vLLM, llama.cpp, TensorRT-LLM,
HF `generate()`) turn "run the model" into something fast and memory-efficient.

## How it works

There's one tiny GPT-2-style transformer (`common/reference_model.py`, PyTorch,
random deterministic weights — output quality is irrelevant, only the math is)
that acts as ground truth for every stage. Each stage is a self-contained folder:

- `README.md` — the math for that stage
- `exercise.py` — function stubs for you to implement (pure NumPy)
- `test_stage*.py` — checks your implementation against the PyTorch reference

Run a stage's tests with:

```
pip install -r requirements.txt
pytest stage1_forward_pass/ -v
```

Nothing is graded by me reading your code — the tests either pass (numerically
matches the reference to float tolerance) or they don't.

## Roadmap

| Stage | Topic | Status |
|---|---|---|
| 1 | [Naive forward pass](stage1_forward_pass/README.md) — embeddings, attention, LayerNorm, GELU, causal masking, FLOPs | ready |
| 2 | KV cache — incremental decoding, memory footprint math | next |
| 3 | Weight quantization (int8) — scale/zero-point math, error bounds | planned |
| 4 | Batching — padding masks, why continuous batching exists | planned |
| 5 | Sampling — temperature, top-k, top-p (nucleus) | planned |
| 6+ | Stretch: speculative decoding, paged attention, PyTorch port for perf comparison | ideas |

Work through stage 1 first (`stage1_forward_pass/exercise.py`) — once its tests
pass, say so and we'll build stage 2 (KV cache), which reuses your stage 1 code.
