import numpy as np
import torch

from common.config import GPTConfig
from common.reference_model import build_reference, export_weights
from stage1_forward_pass.exercise import causal_self_attention, gelu, gpt_forward, layer_norm, linear, softmax


def test_layer_norm_matches_torch():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 5, 8)).astype(np.float32)
    weight = rng.normal(size=(8,)).astype(np.float32)
    bias = rng.normal(size=(8,)).astype(np.float32)

    ref = torch.nn.functional.layer_norm(
        torch.tensor(x), (8,), torch.tensor(weight), torch.tensor(bias), eps=1e-5
    ).numpy()
    out = layer_norm(x, weight, bias)
    np.testing.assert_allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_linear_matches_torch():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, 4)).astype(np.float32)
    W = rng.normal(size=(4, 6)).astype(np.float32)
    b = rng.normal(size=(6,)).astype(np.float32)

    ref = (x @ W) + b
    out = linear(x, W, b)
    np.testing.assert_allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_gelu_matches_torch():
    x = np.linspace(-5, 5, 101).astype(np.float32)
    ref = torch.nn.functional.gelu(torch.tensor(x), approximate="tanh").numpy()
    out = gelu(x)
    np.testing.assert_allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_softmax_sums_to_one_and_matches_torch():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(4, 7)).astype(np.float32)
    ref = torch.softmax(torch.tensor(x), dim=-1).numpy()
    out = softmax(x, axis=-1)
    np.testing.assert_allclose(out.sum(axis=-1), np.ones(4), atol=1e-6)
    np.testing.assert_allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_causal_attention_matches_torch_reference():
    config = GPTConfig(vocab_size=65, block_size=32, n_layer=2, n_head=4, n_embd=32)
    model = build_reference(config, seed=0)
    weights = export_weights(model)

    rng = np.random.default_rng(3)
    x_np = rng.normal(size=(2, 6, config.n_embd)).astype(np.float32)
    x_t = torch.tensor(x_np)

    with torch.no_grad():
        ref = model.h[0].attn(x_t).numpy()
    out = causal_self_attention(x_np, weights, "h.0.attn.", config.n_head)
    np.testing.assert_allclose(out, ref, atol=1e-4, rtol=1e-4)


def test_full_forward_matches_reference():
    config = GPTConfig(vocab_size=65, block_size=32, n_layer=2, n_head=4, n_embd=32)
    model = build_reference(config, seed=0)
    weights = export_weights(model)

    rng = np.random.default_rng(4)
    idx = rng.integers(0, config.vocab_size, size=(2, 16))

    with torch.no_grad():
        ref_logits = model(torch.tensor(idx, dtype=torch.long)).numpy()

    logits = gpt_forward(idx, weights, config)
    assert logits.shape == ref_logits.shape == (2, 16, config.vocab_size)
    np.testing.assert_allclose(logits, ref_logits, atol=1e-4, rtol=1e-4)
