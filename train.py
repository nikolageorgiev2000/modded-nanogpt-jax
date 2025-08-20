import os
import sys
import glob
import time
import uuid
import dataclasses
import datetime

import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "all")

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
    n_train_iters: int = 1675
    n_warmup_iters: int = 0
    f_warmdown_iters: float = 0.4
    n_warmdown_iters: int = 0
    val_loss_every: int = 125
    val_tokens: int = 10485760
    save_every: int = 0

    # input sizes
    batch_size: int = 8 * 64  # batch size for min_sequence_length
    micro_batch_size: int = 16
    min_sequence_length: int = 1024
    max_sequence_length: int = 2048
    sequence_warmup_intervals: int = 1024

    # init
    seed: int = 42

    # eps
    adam_eps: float = 1e-10

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
    d_model: int = 1024
    n_heads: int = 4
    d_head: int = 0
    logit_softcap: float = 15.0
    rope_base: float = 1024
    vocab_size: int = 50304
    dtype: str = "bfloat16"

    # sharding
    weight_sharding = None
    activation_sharding = (None, "dp")
    attention_sharding = ("dp", None, None, None)

    def __post_init__(self):
        object.__setattr__(self, "mesh_shape", (jax.device_count(),))
        assert self.batch_size % self.micro_batch_size == 0

        object.__setattr__(
            self, "n_warmdown_iters", int(self.n_train_iters * self.f_warmdown_iters)
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
    warmdown_lr = (n_train_iters - it) / n_warmdown_iters * (1.0 - 0.1) + 0.1
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
        m = tree_map(
            lambda m, g: (b1 * m + (1 - b1) * g).astype(m.dtype), state["m"], grads
        )
        v = tree_map(
            lambda v, g: (b2 * v + (1 - b2) * g**2).astype(v.dtype), state["v"], grads
        )
        m_hat = tree_map(lambda m: (m / (1 - b1**step)).astype(m.dtype), m)
        v_hat = tree_map(lambda v: (v / (1 - b2**step)).astype(v.dtype), v)
        updates = tree_map(lambda m, v: lr * m / (jnp.sqrt(v) + adam_eps), m_hat, v_hat)
        new_params = tree_map(lambda p, u: p - u.astype(p.dtype), params, updates)
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
        momentum = warmup_momentum_init + frac * (
            warmup_momentum_final - warmup_momentum_init
        )
        new_m = tree_map(
            lambda m, g: (m + (1 - momentum).astype(m.dtype) * (g - m)).astype(m.dtype),
            state["m"],
            grads,
        )

        def _update_leaf(g, p, m):
            g_nesterov = g + momentum.astype(m.dtype) * (m - g)
            update = (
                lr.astype(p.dtype)
                * zeropower_via_newtonschulz5(g_nesterov, ns_iters, eps).astype(p.dtype)
                * jnp.sqrt(jnp.maximum(1.0, g.shape[0] / g.shape[1])).astype(p.dtype)
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


def _get_shape_for_step(step: int, config: Config):
    available_seq_lens = np.arange(
        config.min_sequence_length,
        config.max_sequence_length + 1,
        config.sequence_warmup_intervals,
    )
    if config.max_sequence_length not in available_seq_lens:
        available_seq_lens = np.append(available_seq_lens, config.max_sequence_length)
    n_lens = len(available_seq_lens)
    idx = int((step / config.n_train_iters) * n_lens)
    current_seq_len = available_seq_lens[idx]
    total_tokens = config.batch_size * config.min_sequence_length
    current_B = total_tokens // current_seq_len
    current_B = (current_B // config.micro_batch_size) * config.micro_batch_size
    current_B = max(current_B, config.micro_batch_size)
    current_n_grad_acc = current_B // config.micro_batch_size
    return int(current_seq_len), int(current_B), int(current_n_grad_acc)


def load_dataset(
    config: Config, logger: Logger, mesh: Mesh, is_training: bool
) -> List[tuple[jax.Array, jax.Array]]:
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
        f"Process {process_rank}/{num_processes} starting data pre-loading into RAM..."
    )
    all_tokens_list = [_load_data_shard(f) for f in files]
    all_tokens = np.concatenate(all_tokens_list)
    logger.msg(
        f"Process {process_rank}/{num_processes} finished loading {all_tokens.nbytes / 1e9:.2f} GB of tokens."
    )
    shape_schedule = []
    if is_training:
        for step in range(config.n_train_iters):
            seq_len, B, _ = _get_shape_for_step(step, config)
            shape_schedule.append({"seq_len": seq_len, "B": B})
        num_global_batches = config.n_train_iters
    else:
        seq_len = config.max_sequence_length
        total_tokens = config.batch_size * config.min_sequence_length
        batch_size = (
            total_tokens // seq_len // config.micro_batch_size
        ) * config.micro_batch_size
        batch_size = max(batch_size, config.micro_batch_size)
        total_tokens_per_batch = batch_size * seq_len
        num_global_batches = config.val_tokens // total_tokens_per_batch
        for _ in range(num_global_batches):
            shape_schedule.append({"seq_len": seq_len, "B": batch_size})

    precomputed_batches = []
    token_cursor = 0
    activation_sharding = NamedSharding(mesh, P(*config.activation_sharding))
    for global_step_idx in range(num_global_batches):
        shape_info = shape_schedule[global_step_idx]
        tokens_for_this_batch = shape_info["B"] * shape_info["seq_len"]
        if global_step_idx % num_processes == process_rank:
            seq_len = shape_info["seq_len"]
            batch_size = shape_info["B"]
            n_grad_acc_steps = batch_size // config.micro_batch_size
            start_idx = token_cursor
            end_idx = start_idx + tokens_for_this_batch + 1
            if end_idx > len(all_tokens):
                if process_rank == 0:
                    logger.msg("Cycling dataset...")
                token_cursor = 0
                start_idx = 0
                end_idx = tokens_for_this_batch + 1
                if end_idx > len(all_tokens):
                    raise RuntimeError(
                        f"Not enough tokens ({len(all_tokens)}) to form even one batch of size {tokens_for_this_batch+1}."
                    )
            buf = all_tokens[start_idx:end_idx]
            x = np.array(buf[:-1], dtype=np.int32).reshape(batch_size, seq_len)
            y = np.array(buf[1:], dtype=np.int32).reshape(batch_size, seq_len)
            batched_x = einops.rearrange(x, "(a b) ... -> a b ...", a=n_grad_acc_steps)
            batched_y = einops.rearrange(y, "(a b) ... -> a b ...", a=n_grad_acc_steps)
            batched_x = jax.device_put(batched_x, activation_sharding)
            batched_y = jax.device_put(batched_y, activation_sharding)
            precomputed_batches.append((batched_x, batched_y))
        token_cursor += tokens_for_this_batch
    logger.msg(
        f"Process {process_rank}/{num_processes} pre-computed {len(precomputed_batches)} batches."
    )
    if num_global_batches > 0 and not precomputed_batches:
        raise RuntimeError(
            f"Process {process_rank} could not create any batches. "
            "Check data size, batch configuration, and number of processes."
        )
    return precomputed_batches


# ======================== inits =============================


def precompute_rope(config: Config, mesh: Mesh) -> PyTree:
    weight_sharding = NamedSharding(mesh, P(config.weight_sharding))
    dim = config.d_head
    seq_len = config.max_sequence_length
    inv_freq = 1.0 / (
        config.rope_base ** (jnp.arange(0, dim // 4, dtype=jnp.float32) / (dim // 4))
    )
    inv_freq = jnp.concatenate([inv_freq, jnp.zeros_like(inv_freq)])
    t = jnp.arange(seq_len)
    freqs = jnp.outer(t, inv_freq)
    cos = jnp.cos(freqs).astype(config.dtype)
    sin = jnp.sin(freqs).astype(config.dtype)
    precomputed_params = {}
    precomputed_params["cos"] = jax.device_put(cos[:, None, :], weight_sharding)
    precomputed_params["sin"] = jax.device_put(sin[:, None, :], weight_sharding)
    return precomputed_params


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
            (0.75 / config.d_model) ** 0.5,
        )
        block_params["attn"]["c_proj"] = sharded_zeros((config.d_model, config.d_model))
        block_params["attn"]["lamb"] = jnp.array(0.5, dtype=config.dtype)
        block_params["attn"]["scale"] = jnp.array(0.12, dtype=config.dtype)
        block_params["mlp"] = dict()
        block_params["mlp"]["c_fc"] = sharded_uniform(
            next(key),
            (config.d_model, 4 * config.d_model),
            (0.75 / config.d_model) ** 0.5,
        )
        block_params["mlp"]["c_proj"] = sharded_zeros(
            (4 * config.d_model, config.d_model)
        )
        lambdas_arr = jnp.array([1.0, 0.0], dtype=config.dtype)
        block_params["lambdas"] = jax.device_put(lambdas_arr, weight_sharding)
        params["h"].append(block_params)

    precomputed_params = precompute_rope(config, mesh)

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
    return jnp.einsum("...i,io->...o", x, weight.astype(x.dtype))


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
    y = dot_product_attention(q, k, v, scale=params["scale"], is_causal=True).reshape(
        B, T, C
    )
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
    _, T = idx.shape
    x = params["wte"][idx]
    x = rms_norm(x, config)
    x0 = x
    v1 = None
    skip_connections = []
    n_encoder_layers = config.n_layers // 2
    n_decoder_layers = config.n_layers - n_encoder_layers
    cos = precomputed_params["cos"][:T, :, :]
    sin = precomputed_params["sin"][:T, :, :]
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
    logits = (2.0 * config.logit_softcap) * jax.nn.sigmoid(
        logits / (config.logit_softcap / 2.0)
    )
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


def train_step(
    config: Config,
    params: PyTree,
    precomputed_params: PyTree,
    opt_state: PyTree,
    optimizer: Optimizer,
    batched_x: jax.Array,
    batched_y: jax.Array,
) -> tuple[PyTree, PyTree, dict]:
    n_grad_acc_steps = batched_x.shape[0]

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
    avg_loss = total_loss / n_grad_acc_steps
    final_grads = tree_map(
        lambda g: (g / n_grad_acc_steps).astype(g.dtype), final_grads_accum
    )
    new_params, new_opt_state = optimizer.update(final_grads, params, opt_state)
    return new_params, new_opt_state, {"loss": avg_loss}


def eval_step(
    params: PyTree,
    batched_x: jax.Array,
    batched_y: jax.Array,
    precomputed_params: PyTree,
    config: Config,
) -> jax.Array:
    n_grad_acc_steps = batched_x.shape[0]

    def loss_loop_body(i, accumulated_loss):
        micro_batch = (batched_x[i], batched_y[i])
        loss = loss_fn(params, micro_batch, precomputed_params, config)
        return accumulated_loss + loss

    total_loss = fori_loop(0, n_grad_acc_steps, loss_loop_body, 0.0)
    avg_loss = total_loss / n_grad_acc_steps
    return avg_loss


def run_evaluation(
    step: int,
    config: Config,
    params: PyTree,
    val_loader: Iterator,
    precomputed_params: PyTree,
    mesh: Mesh,
    logger: Logger,
    compiled_eval_fn: Callable,
):
    logger.msg(f"Running validation for step {step}...")
    val_loss_accum = 0.0
    val_steps = 0
    for batched_x, batched_y in val_loader:
        loss = compiled_eval_fn(params, batched_x, batched_y, precomputed_params)
        val_loss_accum += loss
        val_steps += 1
    if val_steps == 0:
        if step == config.val_loss_every or step >= config.n_train_iters - 1:
            logger.msg("Warning: Validation loader was empty, no validation was run.")
        return
    final_val_loss = val_loss_accum / val_steps
    logger.log({"step": step, "val_loss": final_val_loss})
    logger.msg(f"Validation finished for step {step}.")


def train_loop(config: Config):
    logger = Logger()
    mesh = get_mesh(config)

    with mesh:
        params, precomputed_params = init_params(config, mesh)
        optimizer, opt_state = init_optimizer(config, params, mesh)

        # ahead of time compilation
        jitted_train_step = jit(
            train_step, static_argnames=("config", "optimizer"), donate_argnums=(1, 3)
        )
        jitted_eval_step = jit(eval_step, static_argnames=("config",))
        logger.msg("Determining all unique training shapes...")
        train_shapes = {
            _get_shape_for_step(s, config) for s in range(config.n_train_iters)
        }
        val_config = dataclasses.replace(config, input_bin=config.input_val_bin)
        val_seq_len = val_config.max_sequence_length
        total_tokens = val_config.batch_size * val_config.min_sequence_length
        val_B = (
            total_tokens // val_seq_len // val_config.micro_batch_size
        ) * val_config.micro_batch_size
        val_B = max(val_B, val_config.micro_batch_size)
        val_n_grad_acc = val_B // val_config.micro_batch_size
        val_shape = (val_seq_len, val_B, val_n_grad_acc)
        train_shapes.add(val_shape)
        logger.msg("Starting Ahead-of-Time (AOT) compilation for all shapes...")
        compiled_train_steps = {}
        compiled_eval_fn = None
        activation_sharding = NamedSharding(mesh, P(*config.activation_sharding))
        for seq_len, batch_size, n_grad_acc_steps in sorted(list(train_shapes)):
            shape_key = (seq_len, batch_size, n_grad_acc_steps)
            logger.msg(
                f"AOT compiling for seq_len={seq_len}, B={batch_size}, grad_acc={n_grad_acc_steps}..."
            )
            dummy_x_shape = (n_grad_acc_steps, config.micro_batch_size, seq_len)
            dummy_x = jnp.zeros(dummy_x_shape, dtype=jnp.int32)
            dummy_y = jnp.zeros_like(dummy_x)
            dummy_x = jax.device_put(dummy_x, activation_sharding)
            dummy_y = jax.device_put(dummy_y, activation_sharding)
            compiled_fn = jitted_train_step.lower(
                config,
                params,
                precomputed_params,
                opt_state,
                optimizer,
                dummy_x,
                dummy_y,
            ).compile()
            compiled_train_steps[shape_key] = compiled_fn
            if shape_key == val_shape:
                compiled_eval_fn = jitted_eval_step.lower(
                    params, dummy_x, dummy_y, precomputed_params, config
                ).compile()
        logger.msg("AOT compilation finished for all function variants.")

        # data loading
        logger.msg("Pre-computing and loading all training batches...")
        train_batches = load_dataset(config, logger, mesh, is_training=True)
        train_loader = iter(train_batches)
        logger.msg(f"Loaded {len(train_batches)} training batches for this process.")
        logger.msg("Pre-computing and loading all validation batches...")
        val_batches = load_dataset(val_config, logger, mesh, is_training=False)
        logger.msg(f"Loaded {len(val_batches)} validation batches for this process.")

        logger.msg("Starting training...")
        for step in range(config.n_train_iters):
            batched_x, batched_y = next(train_loader)
            n_grad_acc, _, seq_len = batched_x.shape
            batch_size = n_grad_acc * config.micro_batch_size
            current_shape_key = (seq_len, batch_size, n_grad_acc)
            aot_train_fn = compiled_train_steps[current_shape_key]
            params, opt_state, metrics = aot_train_fn(
                params,
                precomputed_params,
                opt_state,
                batched_x,
                batched_y,
            )

            if step % 10 == 9:
                logger.log({"step": step, "time": datetime.datetime.now()} | metrics)
            if step > 0 and (step % config.val_loss_every == 0):
                run_evaluation(
                    step,
                    config,
                    params,
                    iter(val_batches),
                    precomputed_params,
                    mesh,
                    logger,
                    compiled_eval_fn,
                )
            if config.save_every > 0 and step > 0 and (step % config.save_every == 0):
                logger.dump(step, params, opt_state, config)
        logger.flush()
        logger.msg("Final validation")
        run_evaluation(
            step,
            config,
            params,
            iter(val_batches),
            precomputed_params,
            mesh,
            logger,
            compiled_eval_fn,
        )
        logger.flush()
        logger.msg("Training finished.")
        logger.dump(step, params, opt_state, config)


if __name__ == "__main__":
    config = Config()
    train_loop(config)
