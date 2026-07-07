"""Stage 1: naive NumPy forward pass for a GPT-2-style decoder-only transformer.

Fill in every function below. See README.md in this folder for the math.
Check yourself with:  pytest stage1_forward_pass/ -v
"""
import numpy as np

from common.config import GPTConfig


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Normalize over the last axis, then scale/shift.

    x:      (..., C)
    weight: (C,)
    bias:   (C,)
    returns (..., C)
    """
    raise NotImplementedError


def linear(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """y = x @ W + b

    x: (..., in_features)
    W: (in_features, out_features)
    b: (out_features,)
    returns (..., out_features)
    """
    raise NotImplementedError


def gelu(x: np.ndarray) -> np.ndarray:
    """Tanh-approximation GELU (see README)."""
    raise NotImplementedError


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically-stable softmax along `axis`."""
    raise NotImplementedError


def causal_self_attention(x: np.ndarray, w: dict, prefix: str, n_head: int) -> np.ndarray:
    """Multi-head causal self-attention for one block.

    x: (B, T, C)
    w: full weight dict (index into it with keys like f"{prefix}c_attn.weight")
    prefix: e.g. "h.0.attn."
    returns (B, T, C)
    """
    raise NotImplementedError


def mlp(x: np.ndarray, w: dict, prefix: str) -> np.ndarray:
    """Feed-forward block: Linear -> GELU -> Linear.

    x: (B, T, C)
    prefix: e.g. "h.0.mlp."
    returns (B, T, C)
    """
    raise NotImplementedError


def transformer_block(x: np.ndarray, w: dict, prefix: str, n_head: int) -> np.ndarray:
    """One pre-norm transformer block: x + Attn(LN(x)), then x + MLP(LN(x)).

    x: (B, T, C)
    prefix: e.g. "h.0."
    returns (B, T, C)
    """
    raise NotImplementedError


def gpt_forward(idx: np.ndarray, w: dict, config: GPTConfig) -> np.ndarray:
    """Full forward pass.

    idx: (B, T) int array of token ids
    w:   weight dict from common.reference_model.export_weights
    returns logits: (B, T, vocab_size)
    """
    raise NotImplementedError
