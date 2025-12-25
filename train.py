"""
Training script for GPT in JAX - aligned with nanoGPT.
Uses Optax AdamW optimizer with cosine learning rate schedule.

Parameter Freezing:
-------------------
This script supports selective parameter freezing via the `freeze_params` config option.
Frozen parameters will have their gradients zeroed out during training, effectively
preventing them from being updated.

To freeze parameters, specify string patterns in the `freeze_params` tuple. Any parameter
whose path contains one of these patterns will be frozen.

Examples:
    # Freeze token embeddings
    config = TrainConfig(freeze_params=("wte",))
    
    # Freeze first two transformer layers
    config = TrainConfig(freeze_params=("h.0.", "h.1."))
    
    # Freeze all attention layers
    config = TrainConfig(freeze_params=("attn",))
    
    # Freeze all layer norms
    config = TrainConfig(freeze_params=("ln_",))
    
    # Freeze multiple components
    config = TrainConfig(freeze_params=("wte", "h.0.", "h.1."))

Parameter paths follow the structure: "wte", "h.0.attn.c_attn.weight", "h.1.mlp.c_fc.bias", etc.
"""

import os
import sys
import glob
import uuid
import dataclasses
import datetime
import pickle
import itertools
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "all")

import jax.numpy as jnp
from jax import jit, value_and_grad
from jax.lax import scan
from jax.tree_util import tree_map, tree_map_with_path
from jax.sharding import PartitionSpec as P, Mesh, NamedSharding

import optax

import einops
import numpy as np

from model import GPTConfig, loss_fn, init_params, get_num_params

PyTree = Any


# ======================== utils ========================


try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None


class Logger:
    """Unified logger that handles both local printing and optional W&B logging."""

    def __init__(self, config):
        self.is_master = jax.process_index() == 0
        self.use_wandb = config.wandb_mode != "disabled" and self.is_master
        self.logfile: Optional[str] = None
        self.logdir: Optional[str] = None
        self.run_id: Optional[str] = None
        self._wandb_run: Optional[Any] = None

        if not self.is_master:
            return

        if self.use_wandb:
            if wandb is None:
                print("[wandb] wandb not installed. Disabling W&B.")
                self.use_wandb = False
            else:
                mode = config.wandb_mode
                has_env_key = bool(os.getenv("WANDB_API_KEY"))
                has_netrc = os.path.exists(os.path.expanduser("~/.netrc"))
                if mode == "online" and not (has_env_key or has_netrc):
                    print("[wandb] No credentials found. Falling back to offline mode.")
                    mode = "offline"

                try:
                    self._wandb_run = wandb.init(
                        project=config.wandb_project,
                        entity=config.wandb_entity,
                        name=config.wandb_run_name,
                        group=config.wandb_group,
                        job_type=config.wandb_job_type,
                        notes=config.wandb_notes,
                        tags=list(config.wandb_tags),
                        mode=mode,
                        config=dataclasses.asdict(config),
                    )
                    self.run_id = getattr(self._wandb_run, "id", None) or str(uuid.uuid4())
                except Exception as e:
                    print(f"[wandb] init failed ({type(e).__name__}: {e}). Disabling W&B.")
                    self.use_wandb = False

        if self.run_id is None:
            self.run_id = str(uuid.uuid4())

        self.logdir = f"logs/{self.run_id}/"
        os.makedirs(self.logdir, exist_ok=True)
        self.logfile = f"logs/{self.run_id}.txt"

        # Local logfile header
        with open(self.logfile, "w") as f:
            with open(sys.argv[0]) as f2:
                f.write("=" * 100 + "\n" + f2.read() + "\n" + "=" * 100 + "\n")

        if self.use_wandb and getattr(config, "wandb_log_code", True):
            try:
                wandb.run.log_code(".", include_fn=lambda p: p.endswith(".py"))
            except Exception:
                pass

    def msg(self, msg: str):
        if not self.is_master:
            return
        print(msg)
        if self.use_wandb:
            try:
                wandb.termlog(str(msg))
            except Exception:
                pass
        with open(self.logfile, "a") as f:
            f.write(f"[MESSAGE] {msg}\n")

    def log(self, metrics: dict):
        if not self.is_master or not metrics:
            return
        
        # Sanitize metrics
        safe = {}
        for k, v in metrics.items():
            if isinstance(v, datetime.datetime):
                safe[k] = v.isoformat()
            else:
                safe[k] = v

        step = safe.get("step")
        if self.use_wandb:
            try:
                if step is not None:
                    wandb.log(safe, step=int(step))
                else:
                    wandb.log(safe)
            except Exception:
                pass

        parts = []
        for k, v in safe.items():
            if isinstance(v, (float, jnp.ndarray, np.ndarray)) and np.ndim(v) == 0:
                parts.append(f"{k}: {float(v):.4g}")
            else:
                parts.append(f"{k}: {v}")
        line = " | ".join(parts)
        print(line)
        with open(self.logfile, "a") as f:
            f.write(f"[METRICS] {line}\n")

    def dump(self, step: int, params: PyTree, opt_state: PyTree, config):
        if not self.is_master:
            return
        state = {
            "step": step,
            "params": jax.device_get(params),
            "opt_state": jax.device_get(opt_state),
            "config": config,
        }
        path = f"{self.logdir}/state_step{step:06d}.pkl"
        with open(path, "wb") as f:
            pickle.dump(state, f)

        if self.use_wandb:
            try:
                wandb.save(path, base_path=self.logdir)
            except Exception:
                pass
        self.msg(f"Saved checkpoint to {path}")

    def flush(self):
        pass  # Simplified Logger logs immediately now

    def finish(self):
        if self.use_wandb:
            try:
                wandb.finish()
            except Exception:
                pass


# ====================== training config =========================


def is_masked(config: Any) -> bool:
    """Check if a config indicates a masked dataset (both train and val contain 'mask')."""
    return (
        hasattr(config, "input_bin")
        and "mask" in config.input_bin.lower()
        and hasattr(config, "input_val_bin")
        and "mask" in config.input_val_bin.lower()
    )


@dataclass(kw_only=True, frozen=True)
@jax.tree_util.register_static
class TrainConfig:
    """Training configuration aligned with nanoGPT defaults."""

    # Mesh / sharding
    mesh_axis_names: tuple[str, ...] = ("dp",)
    mesh_shape: tuple[int, ...] = ()

    # Data paths
    input_bin: str = "data/openwebtext/train.bin"
    input_val_bin: str = "data/openwebtext/val.bin"
    # Note: OWT2 has 9.0B train and 4.4M validation tokens

    # W&B logging
    wandb_project: str = "gpt-jax"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_job_type: Optional[str] = None
    wandb_tags: tuple[str, ...] = ()
    wandb_notes: Optional[str] = None
    wandb_mode: str = "online"  # "online", "offline", or "disabled"
    wandb_log_code: bool = True

    # Training iterations
    max_iters: int = 20_000  # total number of training iterations
    warmup_iters: int = 2_000  # linear warmup steps
    lr_decay_iters: int = max_iters - warmup_iters  # should be ~= max_iters per Chinchilla
    eval_interval: int = 100  # evaluate every N steps
    eval_iters: int = 8  # number of batches for evaluation (if it exceeds the dataset size, it will cycle through the dataset)
    log_interval: int = 10  # log every N steps
    save_every: int = 0  # save checkpoint every N steps (0 = disabled)

    # Batch sizes (aligned with nanoGPT, but adjusted for 16-device mesh)
    batch_size: int = 64  # micro batch size (must be divisible by num devices for this sharding)
    gradient_accumulation_steps: int = 8  # adjusted to keep total batch size 512

    # AdamW optimizer (aligned with nanoGPT)
    learning_rate: float = 3e-3  # max learning rate
    min_lr: float = 1e-2 * learning_rate # min learning rate
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0  # gradient clipping (0 = disabled)

    # Model config (GPT-2 124M)
    n_layer: int = 12
    embd_dim: int = 768
    head_dim: int = 64
    block_size: int = 512  # sequence length
    vocab_size: int = 50304
    dropout: float = 0.0
    max_seq_len: int = 2048  # maximum sequence length for RoPE precomputation
    rope_base: float = 10000.0  # RoPE base frequency
    use_mlp: bool = True  # whether to use MLP layers in transformer blocks
    off_by_one_attn: bool = False  # whether to add 1.0 to attention softmax denominator

    # Random seed
    seed: int = 1337

    # Sharding
    weight_sharding = None
    activation_sharding = (None, "dp")

    # Masked loss (dataset format: [chunk1][mask1][chunk2][mask2]...)
    use_masked_loss: bool = False

    # Parameter freezing - specify parameter name patterns to freeze
    # Examples: ("wte",) to freeze embeddings, ("h.0.", "h.1.") to freeze first two layers
    freeze_params: tuple[str, ...] = ()

    @property
    def data_has_masks(self) -> bool:
        # needed to load data with the right shape (2, ntok)
        return is_masked(self)

    def __post_init__(self):
        object.__setattr__(self, "mesh_shape", (jax.device_count(),))
        if self.use_masked_loss:
            assert self.data_has_masks, (
                f"use_masked_loss=True requires data with masks (filenames containing 'mask'), "
                f"but input paths are '{self.input_bin}' and '{self.input_val_bin}'"
            )

    def get_model_config(self) -> GPTConfig:
        """Create GPTConfig from training config."""
        return GPTConfig(
            block_size=self.block_size,
            vocab_size=self.vocab_size,
            n_layer=self.n_layer,
            embd_dim=self.embd_dim,
            head_dim=self.head_dim,
            dropout=self.dropout,
            max_seq_len=self.max_seq_len,
            rope_base=self.rope_base,
            weight_sharding=self.weight_sharding,
            use_mlp=self.use_mlp,
            off_by_one_attn=self.off_by_one_attn,
        )


def get_mesh(config: TrainConfig) -> Mesh:
    return jax.make_mesh(config.mesh_shape, config.mesh_axis_names)


# ====================== optimizer ==================


def get_lr_schedule(config: TrainConfig) -> optax.Schedule:
    """Cosine decay learning rate schedule with linear warmup."""
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=config.warmup_iters,
        decay_steps=config.lr_decay_iters,
        end_value=config.min_lr,
    )


def create_freeze_mask(params: PyTree, freeze_patterns: tuple[str, ...]) -> PyTree:
    """
    Create a boolean mask tree for frozen parameters.
    
    Args:
        params: Parameter tree
        freeze_patterns: Tuple of string patterns to match against parameter paths.
                        If a parameter path contains any of these patterns, it will be frozen.
    
    Returns:
        Boolean mask tree where True = update (train), False = freeze (no gradient)
    
    Example patterns:
        ("wte",) - freeze token embeddings
        ("h.0.", "h.1.") - freeze first two transformer layers
        ("ln_",) - freeze all layer norms
    """
    if not freeze_patterns:
        # No freezing - all parameters trainable
        return tree_map(lambda _: True, params)
    
    def should_train(path, leaf):
        # Convert path to string for pattern matching
        # path is a tuple of keys like (('h', 0), ('attn',), ('c_attn',), ('weight',))
        path_str = ".".join(str(key) for key_tuple in path for key in (key_tuple if isinstance(key_tuple, tuple) else (key_tuple,)))
        
        # Check if any freeze pattern matches
        for pattern in freeze_patterns:
            if pattern in path_str:
                return False  # Freeze this parameter
        return True  # Train this parameter
    
    return tree_map_with_path(should_train, params)


def create_optimizer(config: TrainConfig) -> optax.GradientTransformation:
    """
    Create AdamW optimizer with cosine learning rate schedule and warmup.
    Aligned with nanoGPT optimizer configuration.
    """
    lr_schedule = get_lr_schedule(config)

    # Build optimizer chain
    components = []

    # Gradient clipping (if enabled) - uses global norm clipping like nanoGPT
    if config.grad_clip > 0:
        components.append(optax.clip_by_global_norm(config.grad_clip))

    # AdamW with weight decay applied only to 2D+ parameters (weight matrices)
    # LayerNorm scales (1D) don't get weight decay, matching nanoGPT behavior
    def weight_decay_mask(params):
        def should_decay(path, leaf):
            return leaf.ndim >= 2
        return tree_map_with_path(should_decay, params)

    components.append(
        optax.adamw(
            learning_rate=lr_schedule,
            b1=config.beta1,
            b2=config.beta2,
            weight_decay=config.weight_decay,
            mask=weight_decay_mask,
        )
    )

    return optax.chain(*components)


# ======================== dataset =============================


def load_dataset(
    config: TrainConfig, logger, mesh: Mesh, is_training: bool
) -> Iterable[tuple[jax.Array, ...]]:
    """Load dataset from binary files.
    
    Returns:
        loader: Iterable yielding batches as either (x, y) or (x, y, mask) depending on 
                whether the data has masks.
    
    Data is detected as having masks if the config indicates it (config.data_has_masks).
    For masked data, the format is 2D array (2, ntok) where row 0 is tokens, row 1 is masks.
    """
    process_rank = jax.process_index()
    num_processes = jax.process_count()

    input_pattern = config.input_bin if is_training else config.input_val_bin
    files = sorted(glob.glob(input_pattern))
    if not files:
        raise RuntimeError(f"No files found for pattern {input_pattern}")
    
    # Detect if data has masks
    has_mask = config.data_has_masks

    def _describe_token_file(filename: str) -> dict[str, Any]:
        """Assumes `.bin`: raw uint16 tokens (no header)."""
        filesize = os.path.getsize(filename)
        if filesize < 2:
            raise RuntimeError(f"File too small to contain tokens: {filename}")
        if filesize % 2 != 0:
            raise RuntimeError(
                f"Expected even number of bytes for raw uint16 tokens in {filename}, got {filesize}."
            )
        total_elements = filesize // 2
        
        if has_mask:
            # 2D array: (2, ntok) where row 0 is tokens, row 1 is masks
            assert total_elements % 2 == 0, (
                f"Masked data file {filename} must have even number of uint16 elements "
                f"to form (2, N) array, got {total_elements}"
            )
            ntok = total_elements // 2
        else:
            ntok = total_elements
        
        return {
            "filename": filename,
            "kind": "bin_raw",
            "ntok": int(ntok),
        }

    def _open_tokens_memmap(file_info: dict[str, Any]) -> np.ndarray:
        filename = file_info["filename"]
        kind = file_info["kind"]
        if kind == "bin_raw":
            ntok = int(file_info["ntok"])
            if has_mask:
                # 2D array: (2, ntok) - row 0 is tokens, row 1 is masks
                return np.memmap(
                    filename,
                    dtype=np.uint16,
                    mode="r",
                    shape=(2, ntok),
                )
            else:
                return np.memmap(
                    filename,
                    dtype=np.uint16,
                    mode="r",
                    shape=(ntok,),
                )
        raise RuntimeError(f"Unknown token file kind={kind} for {filename}")

    file_infos = [_describe_token_file(f) for f in files]
    shard_lengths = np.array([fi["ntok"] for fi in file_infos], dtype=np.int64)
    shard_ends = np.cumsum(shard_lengths)
    total_ntok = int(shard_ends[-1])

    total_bytes_on_disk = sum(os.path.getsize(f) for f in files)
    logger.msg(
        f"Process {process_rank}/{num_processes} prepared dataset "
        f"from {len(files)} file(s): {total_ntok:,} tokens, {total_bytes_on_disk / 1e9:.2f} GB on disk."
    )

    def _read_token_window(start: int, end: int) -> np.ndarray:
        """Read tokens in [start, end) from the virtual concatenation of shards.
        
        For masked data (2D): returns shape (2, end - start)
        For non-masked data (1D): returns shape (end - start,)
        """
        if start < 0 or end < 0 or end < start:
            raise RuntimeError(f"Invalid token window: start={start}, end={end}")
        if end > total_ntok:
            raise RuntimeError(
                f"Token window out of range: end={end} > total_ntok={total_ntok}"
            )
        if start == end:
            if has_mask:
                return np.empty((2, 0), dtype=np.uint16)
            else:
                return np.empty((0,), dtype=np.uint16)

        parts: list[np.ndarray] = []
        pos = start
        while pos < end:
            shard_idx = int(np.searchsorted(shard_ends, pos, side="right"))
            shard_start = 0 if shard_idx == 0 else int(shard_ends[shard_idx - 1])
            local_start = pos - shard_start
            local_end = min(int(shard_lengths[shard_idx]), local_start + (end - pos))
            mm = _open_tokens_memmap(file_infos[shard_idx])
            if has_mask:
                # mm shape: (2, shard_ntok), slice along token dimension
                parts.append(mm[:, local_start:local_end])
            else:
                parts.append(mm[local_start:local_end])
            pos += local_end - local_start
        if len(parts) == 1:
            return parts[0]
        # Concatenate along token dimension (axis 1 for 2D, axis 0 for 1D)
        return np.concatenate(parts, axis=1 if has_mask else 0)

    batch_size = config.batch_size
    n_grad_acc = config.gradient_accumulation_steps
    num_samples = batch_size * n_grad_acc

    if is_training:
        num_batches = config.max_iters
    else:
        num_batches = config.eval_iters

    activation_sharding = NamedSharding(mesh, P(*config.activation_sharding))
    
    # We read tokens_per_batch + 1 to create shifted x/y (targets)
    seq_len = config.block_size
    tokens_per_batch = num_samples * seq_len

    class _BatchLoader:
        def __len__(self) -> int:
            return num_batches

        def __iter__(self):
            cursor = 0
            for _ in range(num_batches):
                start_idx = cursor
                # Need +1 token for the shifted target
                end_idx = start_idx + tokens_per_batch + 1

                if end_idx > total_ntok:
                    if process_rank == 0:
                        logger.msg("Cycling dataset...")
                    cursor = 0
                    start_idx = 0
                    end_idx = tokens_per_batch + 1
                    if end_idx > total_ntok:
                        raise RuntimeError(
                            f"Not enough tokens ({total_ntok}) to form even one batch of size {end_idx}."
                        )

                buf = _read_token_window(start_idx, end_idx)

                if has_mask:
                    # buf shape: (2, tokens_per_batch + 1)
                    chunks = buf[0]  # tokens
                    masks = buf[1]   # masks
                    
                    # Create x, y, and loss_mask with length = block_size
                    x = np.array(chunks[:-1], dtype=np.int32).reshape(num_samples, seq_len)
                    y = np.array(chunks[1:], dtype=np.int32).reshape(num_samples, seq_len)
                    # Normalize mask to {0, 1}: any non-zero value becomes 1
                    m = (masks[1:] > 0).astype(np.float32).reshape(num_samples, seq_len)
                    
                    # If use_masked_loss=False, replace mask with ones to disable masking
                    if not config.use_masked_loss:
                        m = np.ones_like(m)

                    # Reshape and shard
                    batched_x = jax.device_put(einops.rearrange(x, "(a b) s -> a b s", a=n_grad_acc), activation_sharding)
                    batched_y = jax.device_put(einops.rearrange(y, "(a b) s -> a b s", a=n_grad_acc), activation_sharding)
                    batched_m = jax.device_put(einops.rearrange(m, "(a b) s -> a b s", a=n_grad_acc), activation_sharding)
                    yield batched_x, batched_y, batched_m
                else:
                    # buf shape: (tokens_per_batch + 1,)
                    x = np.array(buf[:-1], dtype=np.int32).reshape(num_samples, seq_len)
                    y = np.array(buf[1:], dtype=np.int32).reshape(num_samples, seq_len)

                    # Reshape and shard
                    batched_x = jax.device_put(einops.rearrange(x, "(a b) s -> a b s", a=n_grad_acc), activation_sharding)
                    batched_y = jax.device_put(einops.rearrange(y, "(a b) s -> a b s", a=n_grad_acc), activation_sharding)
                    yield batched_x, batched_y

                cursor += tokens_per_batch

    loader = _BatchLoader()

    logger.msg(
        f"Process {process_rank}/{num_processes} prepared loader with {len(loader)} batches."
    )
    return loader


# ======================== training ============================


def train_step(
    model_config: GPTConfig,
    params: PyTree,
    precomputed_params: PyTree,
    opt_state: PyTree,
    optimizer: optax.GradientTransformation,
    batched_x: jax.Array,
    batched_y: jax.Array,
    batched_mask: Optional[jax.Array] = None,
    freeze_mask: Optional[PyTree] = None,
) -> tuple[PyTree, PyTree, dict]:
    """Single training step with gradient accumulation.
    
    Args:
        batched_mask: Optional mask array of shape (n_grad_acc, batch_size, seq_len).
                      If provided, the loss will be computed only on masked positions.
        freeze_mask: Optional boolean mask tree where True = trainable, False = frozen.
                     Frozen parameters will have their gradients zeroed out.
    """
    n_grad_acc_steps = batched_x.shape[0]

    def loss_and_grad_fn(p, micro_batch):
        return value_and_grad(loss_fn)(p, micro_batch, model_config, precomputed_params, training=False)

    def micro_step(carry, micro_batch):
        accum_grads, total_loss = carry
        loss, grads = loss_and_grad_fn(params, micro_batch)
        new_accum_grads = tree_map(jnp.add, accum_grads, grads)
        return (new_accum_grads, total_loss + loss), None

    zero_grads = tree_map(jnp.zeros_like, params)
    init_carry = (zero_grads, 0.0)
    
    # Build scan input tuple: (x, y) or (x, y, mask)
    if batched_mask is not None:
        scan_input = (batched_x, batched_y, batched_mask)
    else:
        scan_input = (batched_x, batched_y)
    
    (final_grads_accum, total_loss), _ = scan(
        micro_step, init_carry, scan_input
    )

    avg_loss = total_loss / n_grad_acc_steps
    final_grads = tree_map(
        lambda g: (g / n_grad_acc_steps).astype(g.dtype), final_grads_accum
    )

    # Apply freeze mask to gradients (zero out frozen parameters)
    if freeze_mask is not None:
        final_grads = tree_map(
            lambda grad, mask: jnp.where(mask, grad, jnp.zeros_like(grad)),
            final_grads,
            freeze_mask,
        )

    # Apply optimizer update
    updates, new_opt_state = optimizer.update(final_grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state, {"loss": avg_loss}


def eval_step(
    params: PyTree,
    batched_x: jax.Array,
    batched_y: jax.Array,
    model_config: GPTConfig,
    precomputed_params: PyTree,
    batched_mask: Optional[jax.Array] = None,
) -> jax.Array:
    """Evaluation step.
    
    Args:
        batched_mask: Optional mask array of shape (n_grad_acc, batch_size, seq_len).
                      If provided, the loss will be computed only on masked positions.
    """
    n_grad_acc_steps = batched_x.shape[0]

    def loss_loop_body(i, accumulated_loss):
        if batched_mask is not None:
            micro_batch = (batched_x[i], batched_y[i], batched_mask[i])
        else:
            micro_batch = (batched_x[i], batched_y[i])
        loss = loss_fn(params, micro_batch, model_config, precomputed_params, training=False)
        return accumulated_loss + loss

    total_loss = jax.lax.fori_loop(0, n_grad_acc_steps, loss_loop_body, 0.0)
    avg_loss = total_loss / n_grad_acc_steps
    return avg_loss


def run_evaluation(
    step: int,
    params: PyTree,
    precomputed_params: PyTree,
    val_loader,
    logger,
    compiled_eval_fn: Callable,
    config: TrainConfig,
):
    """Run validation loop."""
    logger.msg(f"Running validation for step {step}...")
    val_loss_accum = 0.0
    val_steps = 0
    
    for batch in val_loader:
        # Unpack x, y, (mask) and call with precomputed_params
        loss = compiled_eval_fn(params, *batch[:2], precomputed_params, *batch[2:])
        val_loss_accum += loss
        val_steps += 1
    if val_steps == 0:
        logger.msg("Warning: Validation loader was empty, no validation was run.")
        return
    final_val_loss = val_loss_accum / val_steps
    return final_val_loss


def count_params(params: PyTree) -> dict[str, int]:
    """Granular parameter counting for the model."""
    counts = {"total": 0, "attn": 0, "mlp": 0, "embed": 0}
    
    # Token embeddings
    counts["embed"] = params["wte"].size
    
    # Transformer blocks
    for block in params["h"]:
        # Attention
        attn_params = jax.tree_util.tree_leaves(block["attn"])
        counts["attn"] += sum(p.size for p in attn_params)
        
        # MLP
        if "mlp" in block:
            mlp_params = jax.tree_util.tree_leaves(block["mlp"])
            counts["mlp"] += sum(p.size for p in mlp_params)
        
            # LayerNorms in blocks
            counts["total"] += block["ln_1"].size + block["ln_2"].size

    counts["total"] += counts["attn"] + counts["mlp"] + counts["embed"] + params["ln_f"].size
    return counts


def train_loop(config: TrainConfig):
    """Main training loop."""
    # Initialize logger
    logger = Logger(config)

    mesh = get_mesh(config)
    model_config = config.get_model_config()

    with mesh:
        # Initialize model parameters
        key = jax.random.key(config.seed)
        params, precomputed_params = init_params(model_config, mesh, key)

        # Report number of parameters
        param_counts = count_params(params)
        logger.msg(f"Number of parameters: {param_counts['total'] / 1e6:.2f}M")
        
        # Initial logging of model info
        log_dict = {
            "model/total_params": param_counts["total"],
            "model/attn_params": param_counts["attn"],
            "model/mlp_params": param_counts["mlp"],
            "model/embed_params": param_counts["embed"],
            "model/vocab_size": model_config.vocab_size,
        }

        # Initialize optimizer
        optimizer = create_optimizer(config)
        opt_state = optimizer.init(params)
        
        # Get LR schedule for logging
        lr_schedule = get_lr_schedule(config)
        
        # Create freeze mask for parameter freezing
        freeze_mask = create_freeze_mask(params, config.freeze_params)
        if config.freeze_params:
            # Count frozen parameters
            frozen_count = 0
            trainable_count = 0
            def count_params_fn(path, leaf, mask_leaf):
                nonlocal frozen_count, trainable_count
                if mask_leaf:
                    trainable_count += leaf.size
                else:
                    frozen_count += leaf.size
            tree_map_with_path(count_params_fn, params, freeze_mask)
            logger.msg(f"Parameter freezing enabled: {frozen_count:,} frozen, {trainable_count:,} trainable")
            logger.msg(f"Freeze patterns: {config.freeze_params}")
        else:
            freeze_mask = None
            logger.msg("No parameter freezing applied")

        # JIT compile training and evaluation functions
        jitted_train_step = jit(
            train_step,
            static_argnames=("model_config", "optimizer"),
            donate_argnums=(1, 3), # Updated donate_argnums because params is now arg 1 and opt_state is arg 3
        )
        jitted_eval_step = jit(eval_step, static_argnames=("model_config",))

        # AOT compile for the fixed batch shape
        # Load datasets
        logger.msg("Loading training data...")
        train_batches = load_dataset(config, logger, mesh, is_training=True)
        train_loader = iter(train_batches)
        logger.msg(f"Loaded {len(train_batches)} training batches.")

        val_config = dataclasses.replace(config, input_bin=config.input_val_bin)
        logger.msg("Loading validation data...")
        val_batches = load_dataset(val_config, logger, mesh, is_training=False)
        logger.msg(f"Loaded {len(val_batches)} validation batches.")
        
        has_mask = config.data_has_masks

        # AOT compile training and evaluation functions
        logger.msg("Starting Ahead-of-Time (AOT) compilation...")
        activation_sharding = NamedSharding(mesh, P(*config.activation_sharding))
        n_grad_acc = config.gradient_accumulation_steps
        
        seq_len = config.block_size
        
        dummy_x = jnp.zeros((n_grad_acc, config.batch_size, seq_len), dtype=jnp.int32)
        dummy_y = jnp.zeros_like(dummy_x)
        dummy_x = jax.device_put(dummy_x, activation_sharding)
        dummy_y = jax.device_put(dummy_y, activation_sharding)
        
        if has_mask:
            dummy_mask = jnp.ones((n_grad_acc, config.batch_size, seq_len), dtype=jnp.float32)
            dummy_mask = jax.device_put(dummy_mask, activation_sharding)
            
            compiled_train_step = jitted_train_step.lower(
                model_config,
                params,
                precomputed_params,
                opt_state,
                optimizer,
                dummy_x,
                dummy_y,
                dummy_mask,
                freeze_mask,
            ).compile()

            compiled_eval_step = jitted_eval_step.lower(
                params, dummy_x, dummy_y, model_config, precomputed_params, dummy_mask
            ).compile()
        else:
            compiled_train_step = jitted_train_step.lower(
                model_config,
                params,
                precomputed_params,
                opt_state,
                optimizer,
                dummy_x,
                dummy_y,
                None,  # batched_mask
                freeze_mask,
            ).compile()

            compiled_eval_step = jitted_eval_step.lower(
                params, dummy_x, dummy_y, model_config, precomputed_params
            ).compile()
        logger.msg("AOT compilation finished.")

        # Training loop
        logger.msg("Starting training...")

        for step in range(config.max_iters):
            batch = next(train_loader)

            # First step we have logs from the initialization we want to keep
            if step > 0:
                log_dict = {}

            # Evaluation
            if step % config.eval_interval == 0:
                val_loss = run_evaluation(
                    step,
                    params,
                    precomputed_params,
                    iter(val_batches),
                    logger,
                    compiled_eval_step,
                    config,
                )
                log_dict["val_loss"] = val_loss
            
            # Training: compiled_train_step(params, precomputed, opt_state, x, y, [mask], freeze_mask)
            params, opt_state, metrics = compiled_train_step(
                params,
                precomputed_params,
                opt_state,
                *batch[:2],
                *batch[2:],
                freeze_mask,
            )

            # Logging at the end of every step
            if step % config.log_interval == 0:
                current_lr = lr_schedule(step)
                log_dict.update({
                    "step": step,
                    "lr": current_lr,
                    **metrics,
                })
            
            logger.log(log_dict)

            # Checkpointing
            if config.save_every > 0 and step > 0 and step % config.save_every == 0:
                logger.dump(step, params, opt_state, config)

        # Final evaluation
        logger.msg("Final validation...")
        final_val_loss = run_evaluation(
            step,
            params,
            precomputed_params,
            iter(val_batches),
            logger,
            compiled_eval_step,
            config,
        )
        logger.log({"step": step, "val_loss": final_val_loss})
        logger.msg("Training finished.")
        logger.dump(step, params, opt_state, config)
        logger.finish()
    return params


if __name__ == "__main__":
    jax.distributed.initialize()
    print("Training starting...")
    print("Found", len(jax.devices()), "devices")
    config = TrainConfig()
    print("Config:", config)
    train_loop(config)
    print("Training finished.")
