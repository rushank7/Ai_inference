# Stage 1 — Naive forward pass (no cache)

Goal: implement the forward pass of a GPT-2-style decoder-only transformer in
**pure NumPy**, given a dict of weights exported from a PyTorch reference model.
No caching, no batching tricks yet — just one full forward pass over the whole
sequence at once, exactly like `model(idx)` does in PyTorch.

Your code lives in `exercise.py`. Run `pytest stage1_forward_pass/ -v` to check
yourself against the PyTorch reference — it must match to `1e-4`.

## The model

Decoder-only transformer, `n_layer` identical blocks:

```
x = wte[idx] + wpe[positions]                     # (B, T, C)
for each block:
    x = x + Attention(LayerNorm(x))                # pre-norm residual
    x = x + MLP(LayerNorm(x))                      # pre-norm residual
x = LayerNorm(x)
logits = x @ W_head + b_head                       # (B, T, vocab_size)
```

`B` = batch size, `T` = sequence length, `C` = `n_embd`.

## Math you need

**Linear layer.** Weights are already stored as `(in_features, out_features)`
(the transpose of PyTorch's `nn.Linear.weight`), so it's just:

```
y = x @ W + b
```

**LayerNorm.** Per token, normalize across the last axis (`C`), then scale/shift:

```
mu    = mean(x, axis=-1)
var   = mean((x - mu)^2, axis=-1)          # biased variance (population, not sample)
x_hat = (x - mu) / sqrt(var + eps)
y     = x_hat * weight + bias
```

`eps = 1e-5`.

**GELU (tanh approximation, as used in GPT-2/nanoGPT — not the exact erf version):**

```
gelu(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
```

**Scaled dot-product causal self-attention**, per head:

```
Q, K, V = split(x @ W_c_attn + b_c_attn, 3, axis=-1)     # each (B, T, C)
# reshape each into (B, n_head, T, head_dim)

scores = Q @ K^T / sqrt(head_dim)                        # (B, n_head, T, T)
scores[..., i, j] = -inf   for all j > i                 # causal mask: can't see the future
weights = softmax(scores, axis=-1)
out = weights @ V                                         # (B, n_head, T, head_dim)
# merge heads back to (B, T, C)
out = out @ W_c_proj + b_c_proj
```

Why divide by `sqrt(head_dim)`? Without it, dot products of `head_dim`-dim
vectors grow with `head_dim`, pushing softmax into a saturated (near one-hot)
regime and killing gradients/making attention overly peaked. Scaling keeps
`Q·K` roughly unit-variance regardless of `head_dim`.

**Softmax** (numerically stable — always subtract the max first):

```
softmax(x)_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
```

## Weight dict keys (from `common/reference_model.export_weights`)

```
wte                         (vocab_size, n_embd)
wpe                         (block_size, n_embd)
h.{i}.ln1.weight/bias       (n_embd,)
h.{i}.attn.c_attn.weight    (n_embd, 3*n_embd)
h.{i}.attn.c_attn.bias      (3*n_embd,)
h.{i}.attn.c_proj.weight    (n_embd, n_embd)
h.{i}.attn.c_proj.bias      (n_embd,)
h.{i}.ln2.weight/bias       (n_embd,)
h.{i}.mlp.c_fc.weight       (n_embd, 4*n_embd)
h.{i}.mlp.c_fc.bias         (4*n_embd,)
h.{i}.mlp.c_proj.weight     (4*n_embd, n_embd)
h.{i}.mlp.c_proj.bias       (n_embd,)
ln_f.weight/bias            (n_embd,)
lm_head.weight              (n_embd, vocab_size)
lm_head.bias                (vocab_size,)
```

## Bonus (not tested): count the FLOPs

A matmul of `(m,k) @ (k,n)` costs `2*m*k*n` FLOPs (multiply + add per term).
For one forward pass over a batch of `B` sequences of length `T`, the transformer
blocks cost approximately:

```
FLOPs ≈ 2 * B * T * n_layer * (
             4 * n_embd^2              # qkv proj (3x) + output proj combined ≈ 4*C^2
           + 8 * n_embd^2               # MLP: C -> 4C -> C is 2 * (C * 4C) = 8*C^2
         )
       + 2 * B * T^2 * n_layer * n_embd   # attention score + weighted-sum matmuls
```

This "roughly 2 * n_params_active_per_token * tokens" rule of thumb is the same
one used to estimate compute for real LLMs (e.g. the Chinchilla/OpenAI scaling-law
papers). Try writing a `count_flops(config, T)` function and sanity-check it
against the formula above — it's useful intuition for reasoning about inference
cost and where compute actually goes (spoiler: for long contexts, the `T^2`
attention term eventually dominates).
