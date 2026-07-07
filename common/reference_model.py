"""A tiny GPT-2-style transformer in PyTorch, used purely as ground truth.

These exercises are about *inference mechanics* (attention, KV caching, quantization,
batching, sampling) -- not about training a good language model. Weights are random
and deterministic (seeded), so output text is meaningless, but every intermediate
tensor is well-defined and can be checked numerically.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig


def new_gelu(x: torch.Tensor) -> torch.Tensor:
    """Tanh approximation of GELU, as used in GPT-2 / nanoGPT."""
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(causal_mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(new_gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.h = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.config.block_size
        pos = torch.arange(T)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)


def build_reference(config: GPTConfig, seed: int = 0) -> GPT:
    """Deterministic random weights. Output quality doesn't matter -- only forward-pass
    correctness does, so any implementation that reproduces these numbers exactly (up to
    float tolerance) has the mechanics right."""
    torch.manual_seed(seed)
    model = GPT(config)
    for p in model.parameters():
        if p.dim() >= 2:
            nn.init.normal_(p, mean=0.0, std=0.02)
        else:
            nn.init.zeros_(p)
    for block in model.h:
        nn.init.ones_(block.ln1.weight)
        nn.init.ones_(block.ln2.weight)
    nn.init.ones_(model.ln_f.weight)
    model.eval()
    return model


def export_weights(model: GPT) -> dict:
    """Flatten a GPT's parameters into plain numpy arrays, oriented so every linear layer
    can be applied as `y = x @ W + b` (W is (in_features, out_features), the transpose of
    PyTorch's nn.Linear.weight)."""
    import numpy as np

    def np_(t: torch.Tensor) -> "np.ndarray":
        return t.detach().numpy().copy()

    w = {
        "wte": np_(model.wte.weight),                     # (vocab_size, n_embd)
        "wpe": np_(model.wpe.weight),                      # (block_size, n_embd)
        "ln_f.weight": np_(model.ln_f.weight),
        "ln_f.bias": np_(model.ln_f.bias),
        "lm_head.weight": np_(model.lm_head.weight.t()),   # (n_embd, vocab_size)
        "lm_head.bias": np_(model.lm_head.bias),
    }
    for i, block in enumerate(model.h):
        p = f"h.{i}."
        w[p + "ln1.weight"] = np_(block.ln1.weight)
        w[p + "ln1.bias"] = np_(block.ln1.bias)
        w[p + "attn.c_attn.weight"] = np_(block.attn.c_attn.weight.t())    # (n_embd, 3*n_embd)
        w[p + "attn.c_attn.bias"] = np_(block.attn.c_attn.bias)
        w[p + "attn.c_proj.weight"] = np_(block.attn.c_proj.weight.t())    # (n_embd, n_embd)
        w[p + "attn.c_proj.bias"] = np_(block.attn.c_proj.bias)
        w[p + "ln2.weight"] = np_(block.ln2.weight)
        w[p + "ln2.bias"] = np_(block.ln2.bias)
        w[p + "mlp.c_fc.weight"] = np_(block.mlp.c_fc.weight.t())          # (n_embd, 4*n_embd)
        w[p + "mlp.c_fc.bias"] = np_(block.mlp.c_fc.bias)
        w[p + "mlp.c_proj.weight"] = np_(block.mlp.c_proj.weight.t())      # (4*n_embd, n_embd)
        w[p + "mlp.c_proj.bias"] = np_(block.mlp.c_proj.bias)
    return w
