"""
GPT Language Model in JAX - aligned with nanoGPT architecture.
References:
1) https://github.com/karpathy/nanoGPT
2) GPT-2 paper: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
"""

import math
from dataclasses import dataclass
from typing import Any, Optional
from functools import partial
import itertools

import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from jax.sharding import PartitionSpec as P, Mesh, NamedSharding
from jax.nn import silu, gelu, softmax

PyTree = Any


@dataclass(kw_only=True, frozen=True)
@jax.tree_util.register_static
class GPTConfig:
    """Configuration for GPT model, aligned with nanoGPT."""
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64
    n_layer: int = 12
    embd_dim: int = 768
    head_dim: int = 64
    dropout: float = 0.0  # dropout rate (0.0 = no dropout for pretraining)
    rope_base: float = 10000.0  # RoPE base frequency
    weight_sharding: Optional[str] = None  # sharding spec for weights

    @property
    def n_head(self) -> int:
        return self.embd_dim // self.head_dim

def precompute_rope(config: GPTConfig, mesh: Mesh = None) -> PyTree:
    if mesh is not None:
        weight_sharding = NamedSharding(mesh, P(config.weight_sharding))
    else:
        weight_sharding = None
    dim = config.head_dim
    seq_len = config.block_size
    inv_freq = 1.0 / (
        config.rope_base ** (jnp.arange(0, dim // 4, dtype=jnp.float32) / (dim // 4))
    )
    inv_freq = jnp.concatenate([inv_freq, jnp.zeros_like(inv_freq)])
    t = jnp.arange(seq_len)
    freqs = jnp.outer(t, inv_freq)
    cos = jnp.cos(freqs).astype(jnp.bfloat16)
    sin = jnp.sin(freqs).astype(jnp.bfloat16)
    precomputed_params = {}
    precomputed_params["cos"] = jax.device_put(cos[:, None, :], weight_sharding)
    precomputed_params["sin"] = jax.device_put(sin[:, None, :], weight_sharding)
    return precomputed_params

def apply_rotary_emb(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return jnp.concatenate([y1, y2], axis=-1).astype(x.dtype)

def layer_norm(x: jax.Array, weight: jax.Array, eps: float = 1e-5) -> jax.Array:
    """Layer normalization without bias."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / jnp.sqrt(var + eps)
    return x_norm * weight


def linear(x: jax.Array, weight: jax.Array) -> jax.Array:
    """Linear layer without bias: y = x @ W"""
    return x @ weight.astype(x.dtype)


def causal_self_attention(
    x: jax.Array,
    c_attn_weight: jax.Array,
    c_proj_weight: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    config: GPTConfig,
    dropout_key: Optional[PRNGKey] = None,
    training: bool = False,
) -> jax.Array:
    """
    Causal self-attention with RoPE positional encoding.
    Uses flash attention via jax.nn.dot_product_attention.
    """
    B, T, C = x.shape
    n_head = config.n_head
    head_dim = config.head_dim

    # Compute Q, K, V projections
    qkv = linear(x, c_attn_weight)  # (B, T, 3*C)
    q, k, v = jnp.split(qkv, 3, axis=-1)  # each (B, T, C)

    # Reshape for multi-head attention: (B, T, n_head, head_dim)
    q = q.reshape(B, T, n_head, head_dim)
    k = k.reshape(B, T, n_head, head_dim)
    v = v.reshape(B, T, n_head, head_dim)

    # Apply RoPE to Q and K
    q = apply_rotary_emb(q, cos[:T], sin[:T])
    k = apply_rotary_emb(k, cos[:T], sin[:T])

    # Use JAX's efficient attention implementation
    scale = config.embd_dim ** -0.5
    y = jax.nn.dot_product_attention(q, k, v, scale=scale, is_causal=True)

    # Reshape back: (B, T, n_head, head_dim) -> (B, T, C)
    y = y.reshape(B, T, C)

    # Output projection
    y = linear(y, c_proj_weight)

    # Dropout (only during training)
    if training and config.dropout > 0 and dropout_key is not None:
        mask = jax.random.bernoulli(dropout_key, 1 - config.dropout, y.shape)
        y = jnp.where(mask, y / (1 - config.dropout), 0)

    return y


def mlp(
    x: jax.Array,
    c_fc_weight: jax.Array,
    c_fc2_weight: jax.Array,
    c_proj_weight: jax.Array,
    config: GPTConfig,
    dropout_key: Optional[PRNGKey] = None,
    training: bool = False,
) -> jax.Array:
    """MLP block: (silu(Linear) * Linear) -> Linear -> Dropout"""
    gate = linear(x, c_fc_weight)
    up = linear(x, c_fc2_weight)
    x = silu(gate) * up
    x = linear(x, c_proj_weight)

    # Dropout (only during training)
    if training and config.dropout > 0 and dropout_key is not None:
        mask = jax.random.bernoulli(dropout_key, 1 - config.dropout, x.shape)
        x = jnp.where(mask, x / (1 - config.dropout), 0)

    return x


def block_forward(
    x: jax.Array,
    block_params: dict,
    cos: jax.Array,
    sin: jax.Array,
    config: GPTConfig,
    dropout_key: Optional[PRNGKey] = None,
    training: bool = False,
) -> jax.Array:
    """
    Transformer block: x = x + attn(ln_1(x)); x = x + mlp(ln_2(x))
    """
    if training and dropout_key is not None:
        attn_key, mlp_key = jax.random.split(dropout_key)
    else:
        attn_key, mlp_key = None, None

    # Attention with pre-norm
    ln1_out = layer_norm(x, block_params["ln_1"])
    attn_out = causal_self_attention(
        ln1_out,
        block_params["attn"]["c_attn"],
        block_params["attn"]["c_proj"],
        cos,
        sin,
        config,
        attn_key,
        training,
    )
    x = x + attn_out

    # MLP with pre-norm
    ln2_out = layer_norm(x, block_params["ln_2"])
    mlp_out = mlp(
        ln2_out,
        block_params["mlp"]["c_fc"],
        block_params["mlp"]["c_fc2"],
        block_params["mlp"]["c_proj"],
        config,
        mlp_key,
        training,
    )
    x = x + mlp_out

    return x


def gpt_forward(
    params: PyTree,
    rope_params: PyTree,
    idx: jax.Array,
    config: GPTConfig,
    dropout_key: Optional[PRNGKey] = None,
    training: bool = False,
) -> jax.Array:
    """
    GPT forward pass:
    1. Token embeddings (RoPE applied in attention)
    2. Dropout
    3. N transformer blocks
    4. Final layer norm
    5. Language model head (weight-tied with token embeddings)
    """
    B, T = idx.shape
    assert T <= config.block_size, f"Cannot forward sequence of length {T}, block size is only {config.block_size}"

    # Token embeddings (no position embeddings - using RoPE)
    x = params["wte"][idx]

    # Dropout on embeddings
    if training and config.dropout > 0 and dropout_key is not None:
        dropout_key, drop_key = jax.random.split(dropout_key)
        mask = jax.random.bernoulli(drop_key, 1 - config.dropout, x.shape)
        x = jnp.where(mask, x / (1 - config.dropout), 0)

    # Get RoPE cos/sin
    cos = rope_params["cos"]
    sin = rope_params["sin"]

    # Transformer blocks
    for block_params in params["h"]:
        if training and dropout_key is not None:
            dropout_key, block_key = jax.random.split(dropout_key)
        else:
            block_key = None
        x = block_forward(x, block_params, cos, sin, config, block_key, training)

    # Final layer norm
    x = layer_norm(x, params["ln_f"])

    # Language model head (weight-tied with wte): logits = x @ wte.T
    logits = jnp.einsum("...d,vd->...v", x, params["wte"].astype(x.dtype))

    return logits.astype(jnp.float32)


def init_params(config: GPTConfig, mesh: Mesh, key: PRNGKey) -> tuple[PyTree, PyTree]:
    """
    Initialize GPT parameters following nanoGPT initialization:
    - Linear weights: normal(0, 0.02)
    - Embeddings: normal(0, 0.02)
    - LayerNorm weights: ones
    - c_proj weights: scaled by 1/sqrt(2*n_layer) for residual connections
    
    Returns:
        params: Learnable model parameters
        rope_params: Precomputed RoPE cos/sin embeddings (not learned)
    """
    weight_sharding = NamedSharding(mesh, P(config.weight_sharding))
    dtype = jnp.bfloat16

    def sharded_normal(key, shape, std=0.02):
        arr = jax.random.normal(key, shape, dtype=dtype) * std
        return jax.device_put(arr, weight_sharding)

    def sharded_ones(shape):
        arr = jnp.ones(shape, dtype=dtype)
        return jax.device_put(arr, weight_sharding)

    keys = map(partial(jax.random.fold_in, key), itertools.count())

    params = {}

    # Token embeddings: (vocab_size, embd_dim)
    params["wte"] = sharded_normal(next(keys), (config.vocab_size, config.embd_dim))

    # Transformer blocks
    params["h"] = []
    residual_scale = 0.02 / math.sqrt(2 * config.n_layer)

    for _ in range(config.n_layer):
        block_params = {
            "ln_1": sharded_ones((config.embd_dim,)),
            "ln_2": sharded_ones((config.embd_dim,)),
            "attn": {
                "c_attn": sharded_normal(next(keys), (config.embd_dim, 3 * config.embd_dim)),
                "c_proj": sharded_normal(next(keys), (config.embd_dim, config.embd_dim), std=residual_scale),
            },
            "mlp": {
                "c_fc": sharded_normal(next(keys), (config.embd_dim, int(8/3.0 * config.embd_dim))),
                "c_fc2": sharded_normal(next(keys), (config.embd_dim, int(8/3.0 * config.embd_dim))),
                "c_proj": sharded_normal(next(keys), (int(8/3.0 * config.embd_dim), config.embd_dim), std=residual_scale),
            },
        }
        params["h"].append(block_params)

    # Final layer norm
    params["ln_f"] = sharded_ones((config.embd_dim,))

    # Precompute RoPE embeddings
    rope_params = precompute_rope(config, mesh)

    return params, rope_params


def loss_fn(
    params: PyTree,
    batch: tuple[jax.Array, jax.Array],
    config: GPTConfig,
    rope_params: PyTree,
    dropout_key: Optional[PRNGKey] = None,
    training: bool = False,
) -> jax.Array:
    """Cross-entropy loss for language modeling."""
    idx, targets = batch
    logits = gpt_forward(params, rope_params, idx, config, dropout_key, training)

    # Cross-entropy loss
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_log_probs = jnp.take_along_axis(
        log_probs, jnp.expand_dims(targets, axis=-1), axis=-1
    ).squeeze(-1)

    return -jnp.mean(target_log_probs)


def get_num_params(params: PyTree) -> int:
    """
    Return the number of parameters in the model.
    Token embeddings are counted because they are weight-tied with lm_head.
    RoPE parameters are precomputed and not learned, so not included.
    """
    return sum(p.size for p in jax.tree_util.tree_leaves(params))
