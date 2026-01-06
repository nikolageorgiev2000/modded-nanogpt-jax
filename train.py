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

Custom targets:
---------------
This script optionally supports supervised / non-autoregressive targets stored in the dataset.
When enabled via `TrainConfig.use_custom_target=True`, the loader will read `y` directly from the
dataset instead of constructing it via a 1-token shift (autoregressive next-token prediction).

Dataset layout (uint16 `.bin`, no header):
  - default (no masks, no custom targets): shape (N,) tokens
  - masked loss (masks only): shape (2, N) where row0=tokens, row1=masks
  - custom targets without masks: shape (2, N) where row0=tokens, row1=targets
  - custom targets with masks: shape (3, N) where row0=tokens, row1=targets, row2=masks

Note: for `use_custom_target=True`, the loader reads exactly `tokens_per_batch` elements (no +1),
and expects targets/masks to align position-wise with `x` (same length).
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
from typing import Any, Callable, Iterable, Optional, Union

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

def is_targeted(config: Any) -> bool:
    """Check if a config indicates a dataset with targets (both train and val contain 'target')."""
    return (
        hasattr(config, "input_bin")
        and "target" in config.input_bin.lower()
        and hasattr(config, "input_val_bin")
        and "target" in config.input_val_bin.lower()
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
    pos_encoding_base: float = 10000.0  # RoPE base frequency
    use_mlp: bool = True  # whether to use MLP layers in transformer blocks
    off_by_one_attn: bool = False  # whether to add 1.0 to attention softmax denominator
    use_pope: bool = False  # whether to use PoPE

    # Random seed
    seed: int = 1337

    # Sharding
    weight_sharding = None
    activation_sharding = (None, "dp")

    # Masked loss (dataset format: [chunk1][mask1][chunk2][mask2]...)
    use_masked_loss: bool = False

    # Custom targets: read `y` directly from the dataset instead of autoregressive shifting.
    # Note: datasets may contain targets even if this is False; those targets will be ignored.
    use_custom_target: bool = False

    # Parameter freezing - specify parameter name patterns to freeze
    # Examples: ("wte",) to freeze embeddings, ("h.0.", "h.1.") to freeze first two layers
    freeze_params: tuple[str, ...] = ()

    # Grouped loss computation - for masks with consecutive positive integer values
    # num_loss_groups: max mask value (e.g., 3 means mask values 1, 2, 3 are valid groups)
    # loss_combiner: callable that takes (group_sums, group_counts) arrays of shape (num_loss_groups,)
    #                and returns a scalar combined loss for gradient computation.
    #                If None, defaults to total_sum / total_count (weighted mean).
    # Example: loss_combiner=lambda sums, counts: sums.sum() / counts.sum()  # weighted mean
    # Example: loss_combiner=lambda sums, counts: (sums / jnp.maximum(counts, 1)).mean()  # mean of means
    num_loss_groups: Optional[int] = None
    loss_combiner: Optional[Callable[[Any, Any], Any]] = None

    @property
    def data_has_masks(self) -> bool:
        # needed to load data with the right shape (2, ntok)
        return is_masked(self)

    @property
    def data_has_targets(self) -> bool:
        # needed to load data with the right shape (2 or 3, ntok)
        return is_targeted(self)

    def __post_init__(self):
        object.__setattr__(self, "mesh_shape", (jax.device_count(),))
        if self.use_masked_loss:
            assert self.data_has_masks, (
                f"use_masked_loss=True requires data with masks (filenames containing 'mask'), "
                f"but input paths are '{self.input_bin}' and '{self.input_val_bin}'"
            )
        if self.use_custom_target:
            assert self.data_has_targets, (
                f"use_custom_target=True requires data with targets (filenames containing 'target'), "
                f"but input paths are '{self.input_bin}' and '{self.input_val_bin}'"
            )
        if self.num_loss_groups is not None:
            assert self.data_has_masks, (
                f"num_loss_groups={self.num_loss_groups} requires data with masks (filenames containing 'mask'), "
                f"but input paths are '{self.input_bin}' and '{self.input_val_bin}'"
            )
            assert self.num_loss_groups > 0, f"num_loss_groups must be positive, got {self.num_loss_groups}"
        if self.loss_combiner is not None:
            assert self.num_loss_groups is not None and self.num_loss_groups > 0, (
                f"loss_combiner requires num_loss_groups to be set to a positive value, "
                f"but num_loss_groups={self.num_loss_groups}"
            )
            # Validate that loss_combiner can process (sums, counts) arrays of the expected length
            dummy_sums = np.ones(self.num_loss_groups, dtype=np.float32)
            dummy_counts = np.ones(self.num_loss_groups, dtype=np.float32)
            try:
                result = self.loss_combiner(dummy_sums, dummy_counts)
                # Check result is scalar-like
                assert np.ndim(result) == 0, (
                    f"loss_combiner must return a scalar, but returned array with shape {np.shape(result)}"
                )
            except Exception as e:
                raise AssertionError(
                    f"loss_combiner failed to process dummy (sums, counts) arrays of length {self.num_loss_groups}: {e}"
                ) from e
        assert self.max_seq_len >= self.block_size, f"max_seq_len must be greater than or equal to block_size, got {self.max_seq_len} and {self.block_size}"

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
            pos_encoding_base=self.pos_encoding_base,
            off_by_one_attn=self.off_by_one_attn,
            use_mlp=self.use_mlp,
            use_pope=self.use_pope,
            weight_sharding=self.weight_sharding,
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
        loader: Iterable yielding batches as either (x, y) or (x, y, mask).
                - If `config.use_masked_loss=True`, yields (x, y, mask)
                - Else yields (x, y)
                - If `config.use_custom_target=True`, `y` is read from the dataset; otherwise
                  `y` is built autoregressively as a 1-token shift of `x`.
    
    Dataset capabilities are inferred from filenames:
      - masks present if `config.data_has_masks` (filenames contain 'mask')
      - targets present if `config.data_has_targets` (filenames contain 'target')

    Row order when both are present: [tokens, targets, masks].
    """
    process_rank = jax.process_index()
    num_processes = jax.process_count()

    input_pattern = config.input_bin if is_training else config.input_val_bin
    files = sorted(glob.glob(input_pattern))
    if not files:
        raise RuntimeError(f"No files found for pattern {input_pattern}")
    
    # Detect dataset contents (by convention: filenames contain 'mask' and/or 'target')
    data_has_masks = config.data_has_masks
    data_has_targets = config.data_has_targets

    # Whether training will *use* these fields
    use_masked_loss = bool(getattr(config, "use_masked_loss", False))
    use_custom_target = bool(getattr(config, "use_custom_target", False))
    num_loss_groups = getattr(config, "num_loss_groups", None)
    use_grouped_loss = num_loss_groups is not None

    n_rows = 1 + (1 if data_has_targets else 0) + (1 if data_has_masks else 0)

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
        
        if n_rows > 1:
            assert total_elements % n_rows == 0, (
                f"Expected {n_rows} rows in data file {filename} but number of uint16 elements "
                f"({total_elements}) is not divisible by {n_rows}."
            )
            ntok = total_elements // n_rows
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
            if n_rows > 1:
                # Multi-row array: see header comment for row semantics.
                return np.memmap(
                    filename,
                    dtype=np.uint16,
                    mode="r",
                    shape=(n_rows, ntok),
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
        
        For multi-row data: returns shape (n_rows, end - start)
        For non-masked data (1D): returns shape (end - start,)
        """
        if start < 0 or end < 0 or end < start:
            raise RuntimeError(f"Invalid token window: start={start}, end={end}")
        if end > total_ntok:
            raise RuntimeError(
                f"Token window out of range: end={end} > total_ntok={total_ntok}"
            )
        if start == end:
            if n_rows > 1:
                return np.empty((n_rows, 0), dtype=np.uint16)
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
            if n_rows > 1:
                # mm shape: (n_rows, shard_ntok), slice along token dimension
                parts.append(mm[:, local_start:local_end])
            else:
                parts.append(mm[local_start:local_end])
            pos += local_end - local_start
        if len(parts) == 1:
            return parts[0]
        # Concatenate along token dimension (axis 1 for 2D, axis 0 for 1D)
        return np.concatenate(parts, axis=1 if n_rows > 1 else 0)

    batch_size = config.batch_size
    n_grad_acc = config.gradient_accumulation_steps
    num_samples = batch_size * n_grad_acc

    if is_training:
        num_batches = config.max_iters
    else:
        num_batches = config.eval_iters

    activation_sharding = NamedSharding(mesh, P(*config.activation_sharding))
    
    # For autoregressive targets we read tokens_per_batch + 1 to create shifted x/y (targets).
    # For custom targets we read exactly tokens_per_batch and expect y (and optional masks)
    # to align position-wise with x.
    seq_len = config.block_size
    tokens_per_batch = num_samples * seq_len

    class _BatchLoader:
        def __len__(self) -> int:
            return num_batches

        def __iter__(self):
            cursor = 0
            for _ in range(num_batches):
                start_idx = cursor
                if use_custom_target:
                    end_idx = start_idx + tokens_per_batch
                else:
                    # Need +1 token for the shifted target
                    end_idx = start_idx + tokens_per_batch + 1

                if end_idx > total_ntok:
                    if process_rank == 0:
                        logger.msg("Cycling dataset...")
                    cursor = 0
                    start_idx = 0
                    end_idx = tokens_per_batch if use_custom_target else (tokens_per_batch + 1)
                    if end_idx > total_ntok:
                        raise RuntimeError(
                            f"Not enough tokens ({total_ntok}) to form even one batch of size {end_idx}."
                        )

                buf = _read_token_window(start_idx, end_idx)

                if n_rows > 1:
                    # Multi-row layouts:
                    # - masks only: (2, N) => [tokens, masks]
                    # - targets only: (2, N) => [tokens, targets]
                    # - targets + masks: (3, N) => [tokens, targets, masks]
                    chunks = buf[0]  # tokens
                    if data_has_targets:
                        targets = buf[1]
                    if data_has_masks:
                        masks = buf[2] if data_has_targets else buf[1]

                    if use_custom_target:
                        # Create x, y (and optional loss_mask) with length = block_size
                        x = np.array(chunks, dtype=np.int32).reshape(num_samples, seq_len)
                        y = np.array(targets, dtype=np.int32).reshape(num_samples, seq_len)
                        if data_has_masks:
                            if use_grouped_loss:
                                # Keep original mask values (1, 2, ..., num_loss_groups) as uint16
                                m = np.array(masks, dtype=np.uint16).reshape(num_samples, seq_len)
                            else:
                                # Normalize mask to {0, 1}: any non-zero value becomes 1
                                m = (masks > 0).astype(np.float32).reshape(num_samples, seq_len)
                    else:
                        # Autoregressive targets from shifted tokens
                        x = np.array(chunks[:-1], dtype=np.int32).reshape(num_samples, seq_len)
                        y = np.array(chunks[1:], dtype=np.int32).reshape(num_samples, seq_len)
                        if data_has_masks:
                            if use_grouped_loss:
                                # Keep original mask values (1, 2, ..., num_loss_groups) as uint16
                                m = np.array(masks[1:], dtype=np.uint16).reshape(num_samples, seq_len)
                            else:
                                # Normalize mask to {0, 1}: any non-zero value becomes 1
                                m = (masks[1:] > 0).astype(np.float32).reshape(num_samples, seq_len)

                    # If use_masked_loss=False or no masks available, disable masking by using ones.
                    if data_has_masks and not use_masked_loss and not use_grouped_loss:
                            m = np.ones_like(m)

                    # Reshape and shard
                    batched_x = jax.device_put(
                        einops.rearrange(x, "(a b) s -> a b s", a=n_grad_acc), activation_sharding
                    )
                    batched_y = jax.device_put(
                        einops.rearrange(y, "(a b) s -> a b s", a=n_grad_acc), activation_sharding
                    )
                    if data_has_masks:
                        batched_m = jax.device_put(
                            einops.rearrange(m, "(a b) s -> a b s", a=n_grad_acc), activation_sharding
                        )
                        yield batched_x, batched_y, batched_m
                    else:
                        yield batched_x, batched_y
                else:
                    # 1D layout: raw tokens only (autoregressive)
                    x = np.array(buf[:-1], dtype=np.int32).reshape(num_samples, seq_len)
                    y = np.array(buf[1:], dtype=np.int32).reshape(num_samples, seq_len)

                    # Reshape and shard
                    batched_x = jax.device_put(
                        einops.rearrange(x, "(a b) s -> a b s", a=n_grad_acc), activation_sharding
                    )
                    batched_y = jax.device_put(
                        einops.rearrange(y, "(a b) s -> a b s", a=n_grad_acc), activation_sharding
                    )
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
    num_loss_groups: Optional[int] = None,
    loss_combiner: Optional[Callable] = None,
) -> tuple[PyTree, PyTree, dict]:
    """Single training step with gradient accumulation.
    
    Args:
        batched_mask: Optional mask array of shape (n_grad_acc, batch_size, seq_len).
                      If provided, the loss will be computed only on masked positions.
                      For grouped loss: contains consecutive positive integers (1..num_loss_groups).
        freeze_mask: Optional boolean mask tree where True = trainable, False = frozen.
                     Frozen parameters will have their gradients zeroed out.
        num_loss_groups: Maximum mask value for grouped loss computation.
        loss_combiner: Callable to combine individual group losses into a scalar.
    """
    n_grad_acc_steps = batched_x.shape[0]
    use_grouped_loss = num_loss_groups is not None

    def loss_and_grad_fn(p, micro_batch):
        def _loss_fn(params):
            return loss_fn(
                params, micro_batch, model_config, precomputed_params,
                training=False, num_loss_groups=num_loss_groups, loss_combiner=loss_combiner
            )
        if use_grouped_loss:
            # Returns (combined_loss, individual_losses), need has_aux=True
            return value_and_grad(_loss_fn, has_aux=True)(p)
        else:
            return value_and_grad(_loss_fn)(p)

    if use_grouped_loss:
        def micro_step(carry, micro_batch):
            accum_grads, total_loss, total_sums, total_counts = carry
            (loss, (group_sums, group_counts)), grads = loss_and_grad_fn(params, micro_batch)
            new_accum_grads = tree_map(jnp.add, accum_grads, grads)
            return (new_accum_grads, total_loss + loss, total_sums + group_sums, total_counts + group_counts), None

        zero_grads = tree_map(jnp.zeros_like, params)
        init_carry = (zero_grads, 0.0, jnp.zeros(num_loss_groups, dtype=jnp.float32), jnp.zeros(num_loss_groups, dtype=jnp.float32))
    else:
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
    
    if use_grouped_loss:
        (final_grads_accum, total_loss, total_sums, total_counts), _ = scan(
            micro_step, init_carry, scan_input
        )
    else:
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

    metrics = {"loss": avg_loss}

    return new_params, new_opt_state, metrics


def eval_step(
    model_config: GPTConfig,
    params: PyTree,
    precomputed_params: PyTree,
    batched_x: jax.Array,
    batched_y: jax.Array,
    batched_mask: Optional[jax.Array] = None,
    num_loss_groups: Optional[int] = None,
    loss_combiner: Optional[Callable] = None,
) -> Union[jax.Array, tuple[jax.Array, jax.Array]]:
    """Evaluation step.
    
    Args:
        model_config: Model configuration (static).
        params: Model parameters.
        precomputed_params: Precomputed positional embeddings.
        batched_x: Input token IDs of shape (n_grad_acc, batch_size, seq_len).
        batched_y: Target token IDs of shape (n_grad_acc, batch_size, seq_len).
        batched_mask: Optional mask array of shape (n_grad_acc, batch_size, seq_len).
                      If provided, the loss will be computed only on masked positions.
        num_loss_groups: Maximum mask value for grouped loss computation (static).
        loss_combiner: Callable to combine (group_sums, group_counts) into a scalar (static).
    
    Returns:
        If num_loss_groups is provided: (avg_loss, total_sums, total_counts)
        Otherwise: avg_loss (scalar)
    """
    n_grad_acc_steps = batched_x.shape[0]
    use_grouped_loss = num_loss_groups is not None

    # Build scan input tuple: (x, y) or (x, y, mask)
    if batched_mask is not None:
        scan_input = (batched_x, batched_y, batched_mask)
    else:
        scan_input = (batched_x, batched_y)

    if use_grouped_loss:
        def micro_step(carry, micro_batch):
            accumulated_loss, accumulated_sums, accumulated_counts = carry
            loss, (group_sums, group_counts) = loss_fn(
                params, micro_batch, model_config, precomputed_params, 
                training=False, num_loss_groups=num_loss_groups, loss_combiner=loss_combiner
            )
            return (accumulated_loss + loss, accumulated_sums + group_sums, accumulated_counts + group_counts), None

        init_carry = (0.0, jnp.zeros(num_loss_groups, dtype=jnp.float32), jnp.zeros(num_loss_groups, dtype=jnp.float32))
        (total_loss, total_sums, total_counts), _ = scan(micro_step, init_carry, scan_input)
        avg_loss = total_loss / n_grad_acc_steps
        return avg_loss, total_sums, total_counts
    else:
        def micro_step(accumulated_loss, micro_batch):
            loss = loss_fn(params, micro_batch, model_config, precomputed_params, training=False)
            return accumulated_loss + loss, None

        total_loss, _ = scan(micro_step, 0.0, scan_input)
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
) -> Union[float, tuple[float, jax.Array, jax.Array]]:
    """Run validation loop.
    
    Returns:
        If config.num_loss_groups is set: (final_val_loss, total_sums, total_counts)
        Otherwise: final_val_loss (scalar)
    """
    logger.msg(f"Running validation for step {step}...")
    val_loss_accum = 0.0
    val_steps = 0
    has_mask = config.data_has_masks
    use_grouped_loss = config.num_loss_groups is not None
    
    if use_grouped_loss:
        val_sums_accum = jnp.zeros(config.num_loss_groups, dtype=jnp.float32)
        val_counts_accum = jnp.zeros(config.num_loss_groups, dtype=jnp.float32)
    
    for batch in val_loader:
        # Call compiled_eval_fn with args matching eval_step signature (minus static args)
        # Non-static args: params, precomputed_params, batched_x, batched_y, batched_mask
        result = compiled_eval_fn(params, precomputed_params, batch[0], batch[1], batch[2] if has_mask else None)
        
        if use_grouped_loss:
            loss, group_sums, group_counts = result
            val_sums_accum = val_sums_accum + group_sums
            val_counts_accum = val_counts_accum + group_counts
        else:
            loss = result
        
        val_loss_accum += loss
        val_steps += 1
    
    if val_steps == 0:
        logger.msg("Warning: Validation loader was empty, no validation was run.")
        return
    
    final_val_loss = val_loss_accum / val_steps
    
    if use_grouped_loss:
        return final_val_loss, val_sums_accum, val_counts_accum
    
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
            static_argnames=("model_config", "optimizer", "num_loss_groups", "loss_combiner"),
            donate_argnums=(1, 3),  # Donate params (arg 1) and opt_state (arg 3) buffers
        )
        jitted_eval_step = jit(eval_step, static_argnames=("model_config", "num_loss_groups", "loss_combiner"))

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
        
        # Get grouped loss settings
        num_loss_groups = config.num_loss_groups
        loss_combiner = config.loss_combiner
        use_grouped_loss = num_loss_groups is not None
        
        if use_grouped_loss:
            logger.msg(f"Using grouped loss with {num_loss_groups} groups")
            if loss_combiner is not None:
                logger.msg("Custom loss combiner function configured")
        
        # Create dummy mask if data has masks (None otherwise)
        dummy_mask = None
        if has_mask:
            mask_dtype = jnp.uint16 if use_grouped_loss else jnp.float32
            dummy_mask = jax.device_put(
                jnp.ones((n_grad_acc, config.batch_size, seq_len), dtype=mask_dtype),
                activation_sharding
            )
        
        compiled_train_step = jitted_train_step.lower(
            model_config, params, precomputed_params, opt_state, optimizer,
            dummy_x, dummy_y, dummy_mask, freeze_mask,
            num_loss_groups, loss_combiner,
        ).compile()

        compiled_eval_step = jitted_eval_step.lower(
            model_config, params, precomputed_params,
            dummy_x, dummy_y, dummy_mask,
            num_loss_groups, loss_combiner
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
                val_result = run_evaluation(
                    step,
                    params,
                    precomputed_params,
                    iter(val_batches),
                    logger,
                    compiled_eval_step,
                    config,
                )
                if use_grouped_loss:
                    val_loss, val_sums, val_counts = val_result
                    log_dict["val_loss"] = val_loss
                    # Log individual validation losses per group (mean = sum / count)
                    val_means = val_sums / jnp.maximum(val_counts, 1.0)
                    for i in range(num_loss_groups):
                        log_dict[f"val_loss_group_{i+1}"] = float(val_means[i])
                else:
                    log_dict["val_loss"] = val_result
            
            # Training: compiled_train_step(params, precomputed, opt_state, x, y, [mask], freeze_mask)
            params, opt_state, metrics = compiled_train_step(
                params,
                precomputed_params,
                opt_state,
                batch[0], batch[1], # x, y
                batch[2] if has_mask else None, # mask
                freeze_mask,
            )

            # Logging at the end of every step
            if step % config.log_interval == 0:
                current_lr = lr_schedule(step)
                log_dict.update({
                    "step": step,
                    "lr": current_lr,
                    "loss": metrics["loss"],
                })
                
                # Log individual training losses per group (mean = sum / count)
                if use_grouped_loss and "group_sums" in metrics:
                    train_means = metrics["group_sums"] / jnp.maximum(metrics["group_counts"], 1.0)
                    for i in range(num_loss_groups):
                        log_dict[f"loss_group_{i+1}"] = float(train_means[i])
                        log_dict[f"count_group_{i+1}"] = float(metrics["group_counts"][i])
            
            logger.log(log_dict)

            # Checkpointing
            if config.save_every > 0 and step > 0 and step % config.save_every == 0:
                logger.dump(step, params, opt_state, config)

        # Final evaluation
        logger.msg("Final validation...")
        final_val_result = run_evaluation(
            step,
            params,
            precomputed_params,
            iter(val_batches),
            logger,
            compiled_eval_step,
            config,
        )
        if use_grouped_loss:
            final_val_loss, final_sums, final_counts = final_val_result
            final_log = {"step": step, "val_loss": final_val_loss}
            final_means = final_sums / jnp.maximum(final_counts, 1.0)
            for i in range(num_loss_groups):
                final_log[f"val_loss_group_{i+1}"] = float(final_means[i])
            logger.log(final_log)
        else:
            logger.log({"step": step, "val_loss": final_val_result})
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
