# currently based on https://github.com/KellerJordan/modded-nanogpt/blob/master/records/111024_UNetDoubleLr/c87bb826-797b-4f37-98c7-d3a5dad2de74.txt

import os
import sys
import glob
import time
import uuid
import dataclasses
import datetime

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
from jax.lax import fori_loop, rsqrt, cond, scan
from jax.random import PRNGKey, split, normal
from jax.tree_util import (
    tree_map,
    tree_leaves,
    tree_map_with_path,
    tree_flatten,
    tree_unflatten,
    DictKey,
)
from jax.sharding import PartitionSpec as P, Mesh, NamedSharding, AxisType
from jax.nn import initializers, relu, log_softmax
from jax.nn import dot_product_attention

import einops

import pickle

import numpy as np

from functools import reduce, partial
import itertools

from dataclasses import dataclass, field
from typing import Any, Dict, List, Iterator, NamedTuple, Callable, Union
import collections.abc

PyTree = Any


# ======================== utils ========================


class Logger:
    def __init__(self):
        self.run_id = None
        self.logdir = None
        self.logfile = None
        self.is_master = jax.process_index() == 0
        if not self.is_master:
            return
        self.run_id = str(uuid.uuid4())
        self.logdir = f"logs/{self.run_id}/"
        os.makedirs(self.logdir, exist_ok=True)
        self.logfile = f"logs/{self.run_id}.txt"
        self.prev_metrics = None
        with open(self.logfile, "w") as f:
            with open(sys.argv[0]) as f2:
                code = f2.read()
            f.write("=" * 100 + "\n" + code + "\n" + "=" * 100 + "\n")

    def msg(self, msg: str):
        if not self.is_master:
            return
        print(msg)
        with open(self.logfile, "a") as f:
            f.write("[MESSAGE] " + str(msg) + "\n")

    def log(self, metrics: dict):
        if not self.is_master:
            return
        metrics, self.prev_metrics = self.prev_metrics, metrics
        if metrics is None:
            return
        metrics = "  |  ".join(
            list(itertools.starmap("{}: {}".format, metrics.items()))
        )
        print(metrics)
        with open(self.logfile, "a") as f:
            f.write("[METRICS (1 step stale)] " + str(metrics) + "\n")

    def flush(self):
        if not self.is_master:
            return
        metrics = self.prev_metrics
        self.prev_metrics = None
        if metrics is None:
            return
        metrics = "  |  ".join(
            list(itertools.starmap("{}: {}".format, metrics.items()))
        )
        print(metrics)
        with open(self.logfile, "a") as f:
            f.write("[METRICS (latest)] " + str(metrics) + "\n")

    def dump(self, step: int, params: PyTree, opt_state: PyTree, config):
        if not self.is_master:
            return
        params_host = jax.device_get(params)
        opt_state_host = jax.device_get(opt_state)
        state_to_save = {
            "step": step,
            "params": params_host,
            "opt_state": opt_state_host,
            "config": config,
        }
        save_path = f"{self.logdir}/state_step{step:06d}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(state_to_save, f)

        self.msg(f"Saved checkpoint to {save_path}")


def filter_pytree(pytree: PyTree, condition_map: Any) -> PyTree | None:
    if condition_map is True:
        return pytree
    if not condition_map:
        return None
    if isinstance(pytree, collections.abc.Mapping) and isinstance(
        condition_map, collections.abc.Mapping
    ):
        filtered_dict = {
            k: subtree
            for k, sub_map in condition_map.items()
            if k in pytree
            and (subtree := filter_pytree(pytree[k], sub_map)) is not None
        }
        return pytree.__class__(filtered_dict)
    if isinstance(pytree, (list, tuple)) and isinstance(condition_map, (list, tuple)):
        filtered_list = [
            subtree
            for item, sub_map in zip(pytree, condition_map)
            if (subtree := filter_pytree(item, sub_map)) is not None
        ]
        return type(pytree)(filtered_list)
    return None


# ====================== training config =========================


@dataclass(kw_only=True, frozen=True)
@jax.tree_util.register_static
class Config:
    # mesh
    mesh_axis_names: tuple[str, ...] = ("dp",)
    mesh_shape: tuple[int, ...] = ()

    # paths
    input_bin: str = "fineweb10B/fineweb_train_*.bin"
    input_val_bin: str = "fineweb10B/fineweb_val_*.bin"

    # iteration handling
    n_train_iters: int = 3000
    n_warmup_iters: int = 0
    n_warmdown_iters: int = 900
    val_loss_every: int = 125
    val_tokens: int = 10485760
    save_every: int = 0
    n_grad_acc_steps: int = 0

    # input sizes
    batch_size: int = 8 * 64
    micro_batch_size: int = 16
    sequence_length: int = 1024

    # init
    seed: int = 42

    # eps
    adam_eps: float = 1e-8

    # adam for embeddings
    adam_embed_base_lr: float = 0.6
    adam_embed_beta1: float = 0.9
    adam_embed_beta2: float = 0.95

    # adam for lm head
    adam_lm_head_base_lr: float = 0.008
    adam_lm_head_beta1: float = 0.9
    adam_lm_head_beta2: float = 0.95

    # muon for matrices
    muon_base_lr: float = 0.04
    muon_momentum_warmup_steps: int = 500
    muon_warmup_momentum_init: float = 0.85
    muon_warmup_momentum_final: float = 0.95
    muon_ns_iters: int = 5
    muon_eps: float = 1e-7

    # adam for non-matrices
    adam_nonmat_base_lr: float = 0.04
    adam_nonmat_beta1: float = 0.9
    adam_nonmat_beta2: float = 0.95

    # arch
    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 6
    d_head: int = 0
    logit_softcap: float = 30.0
    rope_base: float = 10000
    vocab_size: int = 50304
    dtype: str = "bfloat16"

    # sharding
    weight_sharding = None
    activation_sharding = (
        None,
        "dp",
    )  # gradient accumulation axis, micro-batch axis, remaining are automatically None
    attention_sharding = ("dp", None, None, None)

    def __post_init__(self):
        object.__setattr__(self, "mesh_shape", (jax.device_count(),))
        assert self.batch_size % self.micro_batch_size == 0
        object.__setattr__(
            self, "n_grad_acc_steps", self.batch_size // self.micro_batch_size
        )
        assert self.d_model % self.n_heads == 0
        object.__setattr__(self, "d_head", self.d_model // self.n_heads)
        assert self.n_layers % 2 == 0


def get_mesh(config: Config):
    mesh = jax.make_mesh(config.mesh_shape, config.mesh_axis_names)
    return mesh


# ====================== optimizers ==================


class Optimizer(NamedTuple):
    init: Callable
    update: Callable


def get_lr(it, n_warmup_iters, n_warmdown_iters, n_train_iters):
    warmup_lr = (it + 1) / n_warmup_iters
    constant_lr = 1.0
    warmdown_lr = (n_train_iters - it) / n_warmdown_iters
    lr = jnp.where(
        it < n_warmup_iters,
        warmup_lr,
        jnp.where(it < n_train_iters - n_warmdown_iters, constant_lr, warmdown_lr),
    )
    return lr


def adam(
    base_lr: float,
    b1: float,
    b2: float,
    n_warmup_iters: int,
    n_warmdown_iters: int,
    n_train_iters: int,
    adam_eps: float,
):

    def init(params):
        m = tree_map(jnp.zeros_like, params)
        v = tree_map(jnp.zeros_like, params)
        step = jnp.array(0, dtype=jnp.int32)
        return {"m": m, "v": v, "step": step}

    def update(grads, params, state):
        step = state["step"] + 1
        lr = base_lr * get_lr(
            state["step"], n_warmup_iters, n_warmdown_iters, n_train_iters
        )
        m = tree_map(lambda m, g: b1 * m + (1 - b1) * g, state["m"], grads)
        v = tree_map(lambda v, g: b2 * v + (1 - b2) * g**2, state["v"], grads)
        m_hat = tree_map(lambda m: m / (1 - b1**step), m)
        v_hat = tree_map(lambda v: v / (1 - b2**step), v)
        updates = tree_map(lambda m, v: lr * m / (jnp.sqrt(v) + adam_eps), m_hat, v_hat)
        new_params = tree_map(lambda p, u: p - u, params, updates)
        new_state = {"m": m, "v": v, "step": step}
        return new_params, new_state

    return Optimizer(init, update)


def zeropower_via_newtonschulz5(G, steps, eps):

    assert len(G.shape) == 2
    transpose = G.shape[0] > G.shape[1]

    def _update_loop(X):
        a, b, c = (3.4445, -4.7750, 2.0315)
        for i in range(steps):
            A = X @ X.T
            B = b * A + c * (A @ A)
            X = a * X + B @ X
        return X

    def tall_case(g):
        X = g.T.astype(jnp.bfloat16)
        X /= jnp.linalg.norm(X) + eps
        X_final = _update_loop(X)
        return X_final.T.astype(g.dtype)

    def wide_case(g):
        X = g.astype(jnp.bfloat16)
        X /= jnp.linalg.norm(X) + eps
        X_final = _update_loop(X)
        return X_final.astype(g.dtype)

    return cond(transpose, tall_case, wide_case, G)


def muon(
    base_lr: float,
    momentum_warmup_steps: int,
    warmup_momentum_init: float,
    warmup_momentum_final: float,
    n_warmup_iters: int,
    n_warmdown_iters: int,
    n_train_iters: int,
    ns_iters: int,
    eps: float,
):

    def init(params):
        m = tree_map(jnp.zeros_like, params)
        step = jnp.array(0, dtype=jnp.int32)
        return {"m": m, "step": step}

    def update(grads, params, state):
        step = state["step"]
        lr = base_lr * get_lr(step, n_warmup_iters, n_warmdown_iters, n_train_iters)
        frac = jnp.minimum(step / momentum_warmup_steps, 1.0)
        momentum = (1 - frac) * warmup_momentum_init + frac * warmup_momentum_final
        new_m = tree_map(lambda m, g: momentum * m + g, state["m"], grads)

        def _update_leaf(g, p, m):
            g_nesterov = g + m * momentum
            update = (
                lr
                * zeropower_via_newtonschulz5(g_nesterov, ns_iters, eps)
                * jnp.sqrt(jnp.maximum(1.0, g.shape[0] / g.shape[1]))
            )
            return p - update

        new_params = tree_map(_update_leaf, grads, params, new_m)
        new_state = {"m": new_m, "step": step + 1}
        return new_params, new_state

    return Optimizer(init, update)


def multi_optimizer(optimizer_map: Any, **optimizers: Optimizer):
    optimizer_names = list(optimizers.keys())

    def init(params):
        states = {}
        for name, opt in optimizers.items():
            is_relevant_map = tree_map(lambda label: label == name, optimizer_map)
            params_subset = filter_pytree(params, is_relevant_map)
            states[name] = opt.init(params_subset)
        return states

    def update(grads, params, states):
        leaves, treedef = tree_flatten(params)
        map_leaves, _ = tree_flatten(optimizer_map)
        new_leaves_list = list(leaves)
        new_states = {}
        for name, opt in optimizers.items():
            is_relevant_map = tree_map(lambda label: label == name, optimizer_map)
            grads_subset = filter_pytree(grads, is_relevant_map)
            params_subset = filter_pytree(params, is_relevant_map)
            if not grads_subset or not tree_leaves(grads_subset):
                new_states[name] = states[name]
                continue
            new_params_subset, new_states[name] = opt.update(
                grads_subset, params_subset, states[name]
            )
            subset_leaves, _ = tree_flatten(new_params_subset)
            original_indices = [
                i for i, label in enumerate(map_leaves) if label == name
            ]
            assert len(subset_leaves) == len(
                original_indices
            ), f"Mismatch for optimizer {name}"
            for idx, leaf_val in zip(original_indices, subset_leaves):
                new_leaves_list[idx] = leaf_val
        new_params = tree_unflatten(treedef, new_leaves_list)
        return new_params, new_states

    return Optimizer(init, update)


def create_optimizer_map(params):
    def get_label(path, leaf):
        is_adam_embed = any(isinstance(k, DictKey) and k.key == "wte" for k in path)
        is_adam_lm_head = any(
            isinstance(k, DictKey) and k.key == "lm_head" for k in path
        )
        is_muon = (
            any(isinstance(k, DictKey) and k.key == "h" for k in path)
            and leaf.ndim == 2
        )
        is_adam_nonmat = any(
            isinstance(k, DictKey) and k.key == "skip_weights" for k in path
        ) or (
            leaf.ndim < 2 and any(isinstance(k, DictKey) and k.key == "h" for k in path)
        )
        assert (
            int(is_adam_embed)
            + int(is_adam_lm_head)
            + int(is_muon)
            + int(is_adam_nonmat)
            == 1
        )
        if is_adam_embed:
            return "adam_embed"
        elif is_adam_lm_head:
            return "adam_lm_head"
        elif is_muon:
            return "muon"
        else:
            return "adam_nonmat"

    return tree_map_with_path(get_label, params)


# ======================== dataset =============================


# synchronous as of now, some free gains to be had here
def load_dataset(
    config: Config, logger: Logger, mesh: Mesh
) -> Iterator[tuple[jax.Array, jax.Array]]:
    def _load_data_shard(filename):
        with open(filename, "rb") as f:
            header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
            assert header[0] == 20240520, f"Magic number mismatch in {filename}"
            assert header[1] == 1, f"Unsupported version in {filename}"
            ntok = header[2]
            tokens = np.frombuffer(f.read(), dtype=np.uint16)
        assert len(tokens) == ntok, f"Token count mismatch in {filename}"
        return tokens

    process_rank = jax.process_index()
    num_processes = jax.process_count()
    files = sorted(glob.glob(config.input_bin))
    if not files:
        raise RuntimeError(f"No files found for pattern {config.input_bin}")
    logger.msg(
        f"Process {process_rank}/{num_processes} starting data loading from {len(files)} shards."
    )
    tokens_per_device_batch = config.batch_size * config.sequence_length
    stride = tokens_per_device_batch * num_processes
    while True:
        for shard_filename in files:
            tokens = _load_data_shard(shard_filename)
            current_position = tokens_per_device_batch * process_rank
            while current_position + tokens_per_device_batch + 1 <= len(tokens):
                buf = tokens[
                    current_position : current_position + tokens_per_device_batch + 1
                ]
                x = np.array(buf[:-1], dtype=np.int32).reshape(
                    config.batch_size, config.sequence_length
                )
                y = np.array(buf[1:], dtype=np.int32).reshape(
                    config.batch_size, config.sequence_length
                )
                batched_x = einops.rearrange(
                    x, "(a b) ... -> a b ...", a=config.n_grad_acc_steps
                )
                batched_y = einops.rearrange(
                    y, "(a b) ... -> a b ...", a=config.n_grad_acc_steps
                )
                activation_sharding = NamedSharding(
                    mesh, P(*config.activation_sharding)
                )
                batched_x = jax.device_put(batched_x, activation_sharding)
                batched_y = jax.device_put(batched_y, activation_sharding)
                yield batched_x, batched_y
                current_position += stride


# ======================== inits =============================


def init_params(config: Config, mesh: Mesh) -> PyTree:

    weight_sharding = NamedSharding(mesh, P(config.weight_sharding))

    def sharded_zeros(shape):
        arr = jnp.zeros(shape, dtype=config.dtype)
        return jax.device_put(arr, weight_sharding)

    def sharded_ones(shape):
        arr = jnp.ones(shape, dtype=config.dtype)
        return jax.device_put(arr, weight_sharding)

    def sharded_normal(key, shape, std):
        arr = jax.random.normal(key, shape, dtype=config.dtype) * std
        return jax.device_put(arr, weight_sharding)

    def sharded_uniform(key, shape, bound):
        arr = jax.random.uniform(
            key, shape, dtype=config.dtype, minval=-bound, maxval=bound
        )
        return jax.device_put(arr, weight_sharding)

    root_key = jax.random.key(seed=config.seed)
    key = map(partial(jax.random.fold_in, root_key), itertools.count())

    params = dict()
    params["wte"] = sharded_normal(next(key), (config.vocab_size, config.d_model), 1.0)
    params["h"] = []
    params["skip_weights"] = sharded_ones(config.n_layers // 2)
    params["lm_head"] = sharded_zeros((config.d_model, config.vocab_size))

    for i in range(config.n_layers):
        block_params = dict()
        block_params["attn"] = dict()
        block_params["attn"]["c_qkv"] = sharded_uniform(
            next(key),
            (3 * config.d_model, config.d_model),
            (1.0 / config.d_model) ** 0.5,
        )
        block_params["attn"]["c_proj"] = sharded_zeros((config.d_model, config.d_model))
        block_params["attn"]["lamb"] = jnp.array(0.5, dtype=config.dtype)
        block_params["mlp"] = dict()
        block_params["mlp"]["c_fc"] = sharded_uniform(
            next(key),
            (config.d_model, 4 * config.d_model),
            (1.0 / config.d_model) ** 0.5,
        )
        block_params["mlp"]["c_proj"] = sharded_zeros(
            (4 * config.d_model, config.d_model)
        )
        lambdas_arr = jnp.array([1.0, 0.0], dtype=config.dtype)
        block_params["lambdas"] = jax.device_put(lambdas_arr, weight_sharding)
        params["h"].append(block_params)

    # precompute rope
    dim = config.d_head
    seq_len = config.sequence_length
    inv_freq = 1.0 / (
        config.rope_base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
    )
    t = jnp.arange(seq_len)
    freqs = jnp.outer(t, inv_freq)
    cos = jnp.cos(freqs).astype(config.dtype)
    sin = jnp.sin(freqs).astype(config.dtype)

    precomputed_params = {}
    precomputed_params["cos"] = jax.device_put(cos[:, None, :], weight_sharding)
    precomputed_params["sin"] = jax.device_put(sin[:, None, :], weight_sharding)

    return params, precomputed_params


def init_optimizer(config: Config, params: PyTree, mesh: Mesh):

    optimizer_map = create_optimizer_map(params)

    adam_embed = adam(
        config.adam_embed_base_lr,
        config.adam_embed_beta1,
        config.adam_embed_beta2,
        config.n_warmup_iters,
        config.n_warmdown_iters,
        config.n_train_iters,
        config.adam_eps,
    )

    adam_lm_head = adam(
        config.adam_lm_head_base_lr,
        config.adam_lm_head_beta1,
        config.adam_lm_head_beta2,
        config.n_warmup_iters,
        config.n_warmdown_iters,
        config.n_train_iters,
        config.adam_eps,
    )

    adam_nonmat = adam(
        config.adam_nonmat_base_lr,
        config.adam_nonmat_beta1,
        config.adam_nonmat_beta2,
        config.n_warmup_iters,
        config.n_warmdown_iters,
        config.n_train_iters,
        config.adam_eps,
    )

    muon_opt = muon(
        config.muon_base_lr,
        config.muon_momentum_warmup_steps,
        config.muon_warmup_momentum_init,
        config.muon_warmup_momentum_final,
        config.n_warmup_iters,
        config.n_warmdown_iters,
        config.n_train_iters,
        config.muon_ns_iters,
        config.muon_eps,
    )

    optimizer = multi_optimizer(
        optimizer_map,
        adam_embed=adam_embed,
        adam_lm_head=adam_lm_head,
        muon=muon_opt,
        adam_nonmat=adam_nonmat,
    )
    opt_state = optimizer.init(params)

    return optimizer, opt_state


# ======================= model and loss =======================


def rms_norm(x, config):
    return x * rsqrt(
        jnp.mean(jnp.square(x), axis=-1, keepdims=True) + jnp.finfo(x.dtype).eps
    )


def apply_rotary_emb(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return jnp.concatenate([y1, y2], axis=-1).astype(x.dtype)


def linear(x, weight):
    return jnp.einsum("...i,io->...o", x, weight)


def attention_forward(params, x, v1, cos, sin, config):
    B, T, C = x.shape
    params_qkv = einops.rearrange(
        params["c_qkv"], "(three h d) c -> three h d c", d=config.d_head, three=3
    )
    q, k, v = einops.einsum(x, params_qkv, "b t c, three h d c -> three b t h d")

    if v1 is None:
        v1 = v
    v = (1 - params["lamb"]) * v + params["lamb"] * v1.reshape(v.shape)
    q = apply_rotary_emb(rms_norm(q, config), cos, sin)
    k = apply_rotary_emb(rms_norm(k, config), cos, sin)
    y = dot_product_attention(q, k, v, is_causal=True).reshape(B, T, C)
    y = linear(y, params["c_proj"])
    return y, v1


def mlp_forward(params, x):
    x = linear(x, params["c_fc"])
    x = relu(x) ** 2
    x = linear(x, params["c_proj"])
    return x


def block_forward(params, x, v1, x0, cos, sin, config):
    x = params["lambdas"][0] * x + params["lambdas"][1] * x0
    x1, v1 = attention_forward(
        params["attn"], rms_norm(x, config), v1, cos, sin, config
    )
    x = x + x1
    x = x + mlp_forward(params["mlp"], rms_norm(x, config))
    return x, v1


def gpt_forward(params, idx, precomputed_params, config):
    x = params["wte"][idx]
    x = rms_norm(x, config)
    x0 = x
    v1 = None
    skip_connections = []
    n_encoder_layers = config.n_layers // 2
    n_decoder_layers = config.n_layers - n_encoder_layers
    cos = precomputed_params["cos"]
    sin = precomputed_params["sin"]
    for i in range(n_encoder_layers):
        x, v1 = block_forward(params["h"][i], x, v1, x0, cos, sin, config)
        skip_connections.append(x)
    for i in range(n_decoder_layers):
        x = x + params["skip_weights"][i] * skip_connections.pop()
        x, v1 = block_forward(
            params["h"][n_encoder_layers + i], x, v1, x0, cos, sin, config
        )
    x = rms_norm(x, config)
    logits = linear(x, params["lm_head"])
    logits = config.logit_softcap * jnp.tanh(logits / config.logit_softcap)
    return logits.astype(jnp.float32)


def loss_fn(params, batch, precomputed_params, config):
    idx, labels = batch
    logits = gpt_forward(params, idx, precomputed_params, config)
    axis = logits.ndim - 1
    label_logits = jnp.take_along_axis(
        logits, jnp.expand_dims(labels, axis), axis=axis
    ).take(0, axis=axis)
    log_normalizers = jax.nn.logsumexp(logits, axis=axis)
    return jnp.mean(log_normalizers - label_logits)


# ======================== training ============================


@partial(jit, static_argnames=("config", "optimizer"), donate_argnums=(1, 3))
def train_step(
    config: Config,
    params: PyTree,
    precomputed_params: PyTree,
    opt_state: PyTree,
    optimizer: Optimizer,
    batched_x: jax.Array,
    batched_y: jax.Array,
) -> tuple[PyTree, PyTree, dict]:

    def loss_and_grad_fn(p, micro_batch):
        return value_and_grad(loss_fn)(p, micro_batch, precomputed_params, config)

    def micro_step(carry, micro_batch):
        accum_grads, total_loss = carry
        loss, grads = loss_and_grad_fn(params, micro_batch)
        new_accum_grads = tree_map(jnp.add, accum_grads, grads)
        return (new_accum_grads, total_loss + loss), None

    zero_grads = tree_map(jnp.zeros_like, params)
    init_carry = (zero_grads, 0.0)
    (final_grads_accum, total_loss), _ = scan(
        micro_step, init_carry, (batched_x, batched_y)
    )
    avg_loss = total_loss / config.n_grad_acc_steps
    final_grads = tree_map(lambda g: g / config.n_grad_acc_steps, final_grads_accum)
    new_params, new_opt_state = optimizer.update(final_grads, params, opt_state)
    return new_params, new_opt_state, {"loss": avg_loss}


@partial(jit, static_argnames=("config",))
def eval_step(
    params: PyTree,
    batched_x: jax.Array,
    batched_y: jax.Array,
    precomputed_params: PyTree,
    config: Config,
) -> jax.Array:
    def loss_loop_body(i, accumulated_loss):
        micro_batch = (batched_x[i], batched_y[i])
        loss = loss_fn(params, micro_batch, precomputed_params, config)
        return accumulated_loss + loss

    total_loss = fori_loop(0, config.n_grad_acc_steps, loss_loop_body, 0.0)
    avg_loss = total_loss / config.n_grad_acc_steps
    return avg_loss


def run_evaluation(
    step: int,
    config: Config,
    params: PyTree,
    val_loader: Iterator,
    precomputed_params: PyTree,
    mesh: Mesh,
    logger: Logger,
):
    val_steps = config.val_tokens // (config.batch_size * config.sequence_length)
    if val_steps == 0:
        if step == config.val_loss_every:
            logger.msg(
                "Warning: val_tokens is smaller than a single batch, no validation will be run."
            )
        return
    logger.msg(f"Running validation for step {step}...")
    val_loss_accum = 0.0
    for _ in range(val_steps):
        batched_x, batched_y = next(val_loader)
        loss = eval_step(params, batched_x, batched_y, precomputed_params, config)
        val_loss_accum += loss
    final_val_loss = val_loss_accum / val_steps
    logger.log({"step": step, "val_loss": final_val_loss})
    logger.msg(f"Validation finished for step {step}.")


def train_loop(config: Config):
    logger = Logger()
    mesh = get_mesh(config)

    # profiler
    # tb_logdir = f"{logger.logdir}/tensorboard"
    # if logger.is_master:
    #     os.makedirs(tb_logdir, exist_ok=True)

    with mesh:
        params, precomputed_params = init_params(config, mesh)
        optimizer, opt_state = init_optimizer(config, params, mesh)
        train_loader = load_dataset(config, logger, mesh)
        val_config = dataclasses.replace(config, input_bin=config.input_val_bin)
        logger.msg("Starting training...")
        for step in range(config.n_train_iters):

            # profiler
            # if step == 10:
            #     logger.msg("Starting profiler trace...")
            #     jax.profiler.start_trace(tb_logdir)

            batched_x, batched_y = next(train_loader)
            params, opt_state, metrics = train_step(
                config,
                params,
                precomputed_params,
                opt_state,
                optimizer,
                batched_x,
                batched_y,
            )

            # profiler
            # jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
            # if step == 13:
            #     logger.msg("Stopping profiler trace...")
            #     jax.profiler.stop_trace()

            if step % 10 == 9:
                logger.log({"step": step, "time": datetime.datetime.now()} | metrics)
            else:
                logger.log({"step": step, "time": None} | metrics)
            if step > 0 and (step % config.val_loss_every == 0):
                val_loader = load_dataset(val_config, logger, mesh)
                run_evaluation(
                    step, config, params, val_loader, precomputed_params, mesh, logger
                )
            if config.save_every > 0 and step > 0 and (step % config.save_every == 0):
                logger.dump(step, params, opt_state, config)
        logger.flush()
        logger.msg("Final validation")
        val_loader = iter(load_dataset(val_config, logger, mesh))
        run_evaluation(
            step, config, params, val_loader, precomputed_params, mesh, logger
        )
        logger.flush()
        logger.msg("Training finished.")
        logger.dump(step, params, opt_state, config)


if __name__ == "__main__":
    config = Config()
    train_loop(config)
