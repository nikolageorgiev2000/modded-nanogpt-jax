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
    pos_encoding_base: float = 10000.0  # positional embeddings base frequency
    max_seq_len: int = 2048  # maximum sequence length for positional embeddings precomputation
    use_pope: bool = False  # if True, use PoPE; otherwise default to RoPE
    use_mlp: bool = True  # whether to use MLP layers in transformer blocks
    off_by_one_attn: bool = False  # whether to add 1.0 to attention softmax denominator

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
    seq_len = config.max_seq_len
    inv_freq = 1.0 / (
        config.pos_encoding_base ** (jnp.arange(0, dim // 4, dtype=jnp.float32) / (dim // 4))
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

def precompute_pope(config: GPTConfig, mesh: Mesh = None) -> PyTree:
    """
    Precompute PoPE position tables.

    We implement PoPE scores with:
      a_{t,s} = sum_c softplus(q_{t,c}) * softplus(k_{s,c}) * cos((s - t) * theta_c)
    (delta term ignored / assumed 0).

    Using the identity:
      cos((s-t)θ) = cos(sθ)cos(tθ) + sin(sθ)sin(tθ)
    we only need per-position cos/sin tables (T x head_dim), not a (T x T x head_dim) tensor.
    """
    if mesh is not None:
        weight_sharding = NamedSharding(mesh, P(config.weight_sharding))
    else:
        weight_sharding = None

    dim = config.head_dim
    seq_len = config.max_seq_len

    # PoPE uses one frequency per scalar feature (d frequencies instead of d/2).
    # This matches "doubling the number of frequencies" relative to standard RoPE.
    inv_freq = 1.0 / (config.pos_encoding_base ** (jnp.arange(0, dim, dtype=jnp.float32) / dim))
    t = jnp.arange(seq_len, dtype=jnp.float32)
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


def dot_product_attention_custom(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    config: GPTConfig,
    scale: float = 1.0,
    is_causal: bool = False,
    return_attn_weights: bool = False,
) -> tuple[jax.Array, Optional[jax.Array]]:
    """
    Custom attention supporting:
    - RoPE (default) vs PoPE (config.use_pope)
    - Standard softmax vs off-by-one denominator tweak (config.off_by_one_attn)

    PoPE implements (delta ignored):
      scores[t,s] = sum_c softplus(q[t,c]) * softplus(k[s,c]) * cos((s-t)*theta_c)

    Using:
      cos((s-t)θ) = cos(sθ)cos(tθ) + sin(sθ)sin(tθ)
    we compute PoPE logits via two matmuls without materializing a (T x T x d) tensor.
    """
    T = query.shape[1]
    if config.use_pope:
        cos_t = cos[:T]  # (T, 1, head_dim)
        sin_t = sin[:T]  # (T, 1, head_dim)

        qmag = jax.nn.softplus(query)
        kmag = jax.nn.softplus(key)

        q_cos = qmag * cos_t
        q_sin = qmag * sin_t
        k_cos = kmag * cos_t
        k_sin = kmag * sin_t

        scores = (
            jnp.einsum("bqhd,bkhd->bhqk", q_cos, k_cos)
            + jnp.einsum("bqhd,bkhd->bhqk", q_sin, k_sin)
        ) * scale
    else:
        # RoPE path: rotate Q/K, then standard dot-product logits
        q = apply_rotary_emb(query, cos[:T], sin[:T])
        k = apply_rotary_emb(key, cos[:T], sin[:T])
        scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) * scale

    if is_causal:
        mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
        scores = jnp.where(mask, scores, -1e10)

    if config.off_by_one_attn:
        max_score = jnp.max(scores, axis=-1, keepdims=True)
        exp_scores = jnp.exp(scores - max_score)
        # The added 1.0 needs to be scaled as exp(-max_score) to account for the max subtraction
        attn_weights = exp_scores / (jnp.sum(exp_scores, axis=-1, keepdims=True) + jnp.exp(-max_score))
    else:
        attn_weights = softmax(scores, axis=-1)
    output = jnp.einsum("bhqk,bkhd->bqhd", attn_weights, value)
    return output, attn_weights if return_attn_weights else None


def causal_self_attention(
    x: jax.Array,
    c_attn_weight: jax.Array,
    c_proj_weight: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    config: GPTConfig,
    dropout_key: Optional[PRNGKey] = None,
    training: bool = False,
    return_attn_weights: bool = False,
) -> jax.Array:
    """
    Causal self-attention with positional embeddings.
    Uses either custom off-by-one attention or JAX's optimized dot_product_attention.
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

    # Choose attention implementation based on config and positional embeddings
    scale = config.embd_dim ** -0.5
    y, attn_weights = dot_product_attention_custom(q, k, v, cos=cos, sin=sin, config=config, scale=scale, is_causal=True, return_attn_weights=return_attn_weights)

    # Reshape back: (B, T, n_head, head_dim) -> (B, T, C)
    y = y.reshape(B, T, C)

    # Output projection
    y = linear(y, c_proj_weight)

    # Dropout (only during training)
    if training and config.dropout > 0 and dropout_key is not None:
        mask = jax.random.bernoulli(dropout_key, 1 - config.dropout, y.shape)
        y = jnp.where(mask, y / (1 - config.dropout), 0)

    return y, attn_weights


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
    return_attn_weights: bool = False,
) -> tuple[jax.Array, Optional[jax.Array]]:
    """
    Transformer block: x = x + attn(ln_1(x)); x = x + mlp(ln_2(x)) (if use_mlp=True)
    """
    if training and dropout_key is not None:
        attn_key, mlp_key = jax.random.split(dropout_key)
    else:
        attn_key, mlp_key = None, None

    # Attention with pre-norm
    ln1_out = layer_norm(x, block_params["ln_1"])
    attn_out, attn_weights = causal_self_attention(
        ln1_out,
        block_params["attn"]["c_attn"],
        block_params["attn"]["c_proj"],
        cos,
        sin,
        config,
        attn_key,
        training,
        return_attn_weights=return_attn_weights,
    )
    x = x + attn_out

    # MLP with pre-norm (only if use_mlp=True)
    if config.use_mlp:
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

    return x, attn_weights if return_attn_weights else None


def gpt_forward(
    params: PyTree,
    pos_params: PyTree,
    idx: jax.Array,
    config: GPTConfig,
    dropout_key: Optional[PRNGKey] = None,
    training: bool = False,
    return_attn_weights: bool = False,
) -> tuple[jax.Array, Optional[jax.Array]]:
    """
    GPT forward pass:
    1. Token embeddings (positional embeddings applied in attention)
    2. Dropout
    3. N transformer blocks
    4. Final layer norm
    5. Language model head (weight-tied with token embeddings)
    """
    B, T = idx.shape
    assert T <= config.max_seq_len, f"Cannot forward sequence of length {T}, max sequence length is only {config.max_seq_len}"

    # Token embeddings (no position embeddings - using positional embeddings)
    x = params["wte"][idx]

    # Dropout on embeddings
    if training and config.dropout > 0 and dropout_key is not None:
        dropout_key, drop_key = jax.random.split(dropout_key)
        mask = jax.random.bernoulli(drop_key, 1 - config.dropout, x.shape)
        x = jnp.where(mask, x / (1 - config.dropout), 0)

    # Get positional embeddings cos/sin
    cos = pos_params["cos"]
    sin = pos_params["sin"]

    # Transformer blocks
    attn_weights_list = []
    for block_params in params["h"]:
        if training and dropout_key is not None:
            dropout_key, block_key = jax.random.split(dropout_key)
        else:
            block_key = None
        x, attn_weights = block_forward(x, block_params, cos, sin, config, block_key, training, return_attn_weights=return_attn_weights)
        if return_attn_weights:
            attn_weights_list.append(attn_weights)

    # Final layer norm
    x = layer_norm(x, params["ln_f"])

    # Language model head (weight-tied with wte): logits = x @ wte.T
    logits = jnp.einsum("...d,vd->...v", x, params["wte"].astype(x.dtype))

    if return_attn_weights:
        return logits.astype(jnp.float32), attn_weights_list
    else:
        return logits.astype(jnp.float32)


def newtonschulz5(G: jax.Array, steps: int = 5, eps: float = 1e-7) -> jax.Array:
    """
    JAX implementation of Newton-Schulz iteration for matrix function.
    Source: https://kellerjordan.github.io/posts/muon/
    Args:
        G: [n, m] matrix
        steps: number of Newton-Schulz steps
        eps: small numerical constant
    Returns:
        [n, m] or [m, n] matrix depending on input shape
    """
    assert G.ndim == 2
    a, b, c = 2.0, -1.5, 0.5

    # Ensure bfloat16 dtype
    X = G.astype(jnp.bfloat16)

    # Normalize
    X = X / (jnp.linalg.norm(X) + eps)

    swapped = X.shape[0] > X.shape[1]
    if swapped:
        X = jnp.transpose(X)

    def body_fun(_, X):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X_new = a * X + B @ X
        return X_new

    X = jax.lax.fori_loop(0, steps, body_fun, X)

    if swapped:
        X = jnp.transpose(X)

    return X


def init_params(config: GPTConfig, mesh: Mesh, key: PRNGKey) -> tuple[PyTree, PyTree]:
    """
    Initialize GPT parameters following nanoGPT initialization:
    - Linear weights: normal(0, 0.02)
    - Embeddings: normal(0, 0.02)
    - LayerNorm weights: ones
    - c_proj weights: scaled by 1/sqrt(2*n_layer) for residual connections
    
    Returns:
        params: Learnable model parameters
        pos_params: Precomputed positional embeddings cos/sin embeddings (not learned)
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
    # params["wte"] = newtonschulz5(params["wte"])

    # Transformer blocks
    params["h"] = []
    residual_scale = 0.02 / math.sqrt(2 * config.n_layer)

    for _ in range(config.n_layer):
        block_params = {
            "ln_1": sharded_ones((config.embd_dim,)),
            "attn": {
                "c_attn": sharded_normal(next(keys), (config.embd_dim, 3 * config.embd_dim)),
                "c_proj": sharded_normal(next(keys), (config.embd_dim, config.embd_dim), std=residual_scale),
            },
        }
        
        # Only initialize MLP parameters if use_mlp=True
        if config.use_mlp:
            block_params["ln_2"] = sharded_ones((config.embd_dim,))
            block_params["mlp"] = {
                "c_fc": sharded_normal(next(keys), (config.embd_dim, int(8/3.0 * config.embd_dim))),
                "c_fc2": sharded_normal(next(keys), (config.embd_dim, int(8/3.0 * config.embd_dim))),
                "c_proj": sharded_normal(next(keys), (int(8/3.0 * config.embd_dim), config.embd_dim), std=residual_scale),
            }
        
        params["h"].append(block_params)

    # Final layer norm
    params["ln_f"] = sharded_ones((config.embd_dim,))

    # Precompute positional embeddings/tables (not learned)
    pos_params = precompute_pope(config, mesh) if config.use_pope else precompute_rope(config, mesh)

    return params, pos_params


def loss_fn(
    params: PyTree,
    batch: tuple[jax.Array, ...],
    config: GPTConfig,
    pos_params: PyTree,
    dropout_key: Optional[PRNGKey] = None,
    training: bool = False,
    num_loss_groups: Optional[int] = None,
    loss_combiner: Optional[callable] = None,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Cross-entropy loss for language modeling with optional mask.
    
    Args:
        batch: Either (idx, targets) or (idx, targets, mask) where:
               - For binary masks: mask is {0,1} array (0=ignore, 1=include)
               - For grouped masks: mask contains consecutive positive integers (1, 2, ..., num_loss_groups)
                 where 0 means ignore and each positive integer represents a loss group.
        num_loss_groups: Maximum mask value for grouped loss computation. If provided along with
                        loss_combiner, enables segment-based loss computation where each mask value
                        (1 to num_loss_groups) represents a separate loss group.
        loss_combiner: Callable that takes (group_sums, group_counts) arrays (each shape: num_loss_groups,)
                      and returns a scalar combined loss for gradient computation.
                      If None with num_loss_groups, defaults to total_sum / total_count.
    
    Returns:
        If num_loss_groups is provided: (combined_loss, group_sums, group_counts) where
            group_sums and group_counts are shape (num_loss_groups,).
        Otherwise: scalar loss value.
    """
    if len(batch) == 3:
        idx, targets, mask = batch
    else:
        idx, targets = batch
        mask = None

    logits = gpt_forward(params, pos_params, idx, config, dropout_key, training)

    # Cross-entropy loss
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_log_probs = jnp.take_along_axis(
        log_probs, jnp.expand_dims(targets, axis=-1), axis=-1
    ).squeeze(-1)
    
    # Per-token loss (positive, since we negate later or use directly)
    token_losses = -target_log_probs

    if mask is not None and num_loss_groups is not None:
        # Grouped loss computation using segment_sum
        # mask values: 0=ignore, 1..num_loss_groups = different loss groups
        assert target_log_probs.shape == mask.shape
        
        # Flatten for segment operations
        mask_flat = mask.flatten().astype(jnp.uint16)
        losses_flat = token_losses.flatten()
        
        # Compute sum of losses per segment (index 0 = ignored positions)
        # num_segments = num_loss_groups + 1 to include segment 0
        loss_sums_all = jax.ops.segment_sum(
            losses_flat, mask_flat, num_segments=num_loss_groups + 1
        )
        
        # Compute count per segment
        counts_all = jax.ops.segment_sum(
            jnp.ones_like(losses_flat), mask_flat, num_segments=num_loss_groups + 1
        )
        
        # Extract sums and counts for groups 1..num_loss_groups (exclude segment 0 which is ignored)
        group_sums = loss_sums_all[1:]  # shape (num_loss_groups,)
        group_counts = counts_all[1:]   # shape (num_loss_groups,)
        
        # Combine losses for gradient computation
        if loss_combiner is not None:
            # loss_combiner receives (sums, counts) to allow custom weighted combinations
            combined_loss = loss_combiner(group_sums, group_counts)
        else:
            # Default: total sum / total count (weighted mean across all groups)
            total_sum = jnp.sum(group_sums)
            total_count = jnp.sum(group_counts)
            combined_loss = total_sum / jnp.maximum(total_count, 1.0)
        
        return combined_loss, (group_sums, group_counts)
    
    elif mask is not None:
        # Binary mask: sum of masked losses / sum of active mask elements
        assert target_log_probs.shape == mask.shape
        masked_loss = token_losses * mask
        return jnp.sum(masked_loss) / jnp.sum(mask)
    else:
        return jnp.mean(token_losses)


def get_num_params(params: PyTree) -> int:
    """
    Return the number of parameters in the model.
    Token embeddings are counted because they are weight-tied with lm_head.
    Positional embeddings are precomputed and not learned, so not included.
    """
    return sum(p.size for p in jax.tree_util.tree_leaves(params))
