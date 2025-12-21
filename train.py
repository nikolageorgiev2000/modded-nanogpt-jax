"""
Training script for GPT in JAX - aligned with nanoGPT.
Uses Optax AdamW optimizer with cosine learning rate schedule.
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
        if not self.is_master:
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

        line = "  |  ".join(f"{k}: {v}" for k, v in safe.items())
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

    # Random seed
    seed: int = 1337

    # Sharding
    weight_sharding = None
    activation_sharding = (None, "dp")

    def __post_init__(self):
        object.__setattr__(self, "mesh_shape", (jax.device_count(),))

    def get_model_config(self) -> GPTConfig:
        """Create GPTConfig from training config."""
        return GPTConfig(
            block_size=self.block_size,
            vocab_size=self.vocab_size,
            n_layer=self.n_layer,
            embd_dim=self.embd_dim,
            head_dim=self.head_dim,
            dropout=self.dropout,
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
) -> Iterable[tuple[jax.Array, jax.Array]]:
    """Load dataset from binary files."""

    def _describe_token_file(filename: str) -> dict[str, Any]:
        """Assumes `.bin`: raw uint16 tokens (no header)."""
        filesize = os.path.getsize(filename)
        if filesize < 2:
            raise RuntimeError(f"File too small to contain tokens: {filename}")
        if filesize % 2 != 0:
            raise RuntimeError(
                f"Expected even number of bytes for raw uint16 tokens in {filename}, got {filesize}."
            )
        ntok = filesize // 2
        return {
            "filename": filename,
            "kind": "bin_raw",
            "ntok": int(ntok),
        }

    def _open_tokens_memmap(file_info: dict[str, Any]) -> np.ndarray:
        filename = file_info["filename"]
        kind = file_info["kind"]
        if kind == "bin_raw":
            return np.memmap(
                filename,
                dtype=np.uint16,
                mode="r",
                shape=(int(file_info["ntok"]),),
            )
        raise RuntimeError(f"Unknown token file kind={kind} for {filename}")

    process_rank = jax.process_index()
    num_processes = jax.process_count()

    input_pattern = config.input_bin if is_training else config.input_val_bin
    files = sorted(glob.glob(input_pattern))
    if not files:
        raise RuntimeError(f"No files found for pattern {input_pattern}")

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
        """Read tokens in [start, end) from the virtual concatenation of shards."""
        if start < 0 or end < 0 or end < start:
            raise RuntimeError(f"Invalid token window: start={start}, end={end}")
        if end > total_ntok:
            raise RuntimeError(
                f"Token window out of range: end={end} > total_ntok={total_ntok}"
            )
        if start == end:
            return np.empty((0,), dtype=np.uint16)

        parts: list[np.ndarray] = []
        pos = start
        while pos < end:
            shard_idx = int(np.searchsorted(shard_ends, pos, side="right"))
            shard_start = 0 if shard_idx == 0 else int(shard_ends[shard_idx - 1])
            local_start = pos - shard_start
            local_end = min(int(shard_lengths[shard_idx]), local_start + (end - pos))
            mm = _open_tokens_memmap(file_infos[shard_idx])
            parts.append(mm[local_start:local_end])
            pos += local_end - local_start
        if len(parts) == 1:
            return parts[0]
        return np.concatenate(parts, axis=0)

    # Fixed batch shape (no sequence warmup like original)
    seq_len = config.block_size
    batch_size = config.batch_size
    n_grad_acc = config.gradient_accumulation_steps
    tokens_per_batch = batch_size * n_grad_acc * seq_len

    if is_training:
        num_batches = config.max_iters
    else:
        num_batches = config.eval_iters

    activation_sharding = NamedSharding(mesh, P(*config.activation_sharding))

    class _BatchLoader:
        def __len__(self) -> int:
            return num_batches

        def __iter__(self):
            token_cursor = 0
            for _ in range(num_batches):
                start_idx = token_cursor
                end_idx = start_idx + tokens_per_batch + 1

                if end_idx > total_ntok:
                    if process_rank == 0:
                        logger.msg("Cycling dataset...")
                    token_cursor = 0
                    start_idx = 0
                    end_idx = tokens_per_batch + 1
                    if end_idx > total_ntok:
                        raise RuntimeError(
                            f"Not enough tokens ({total_ntok}) to form even one batch of size {tokens_per_batch+1}."
                        )

                buf = _read_token_window(start_idx, end_idx)
                x = np.array(buf[:-1], dtype=np.int32).reshape(n_grad_acc * batch_size, seq_len)
                y = np.array(buf[1:], dtype=np.int32).reshape(n_grad_acc * batch_size, seq_len)

                # Reshape for gradient accumulation: (n_grad_acc, batch_size, seq_len)
                batched_x = einops.rearrange(x, "(a b) s -> a b s", a=n_grad_acc)
                batched_y = einops.rearrange(y, "(a b) s -> a b s", a=n_grad_acc)

                # Shard across mesh
                batched_x = jax.device_put(batched_x, activation_sharding)
                batched_y = jax.device_put(batched_y, activation_sharding)
                yield batched_x, batched_y

                token_cursor += tokens_per_batch

    loader = _BatchLoader()
    logger.msg(
        f"Process {process_rank}/{num_processes} prepared loader with {len(loader)} batches."
    )
    return loader


# ======================== training ============================


def train_step(
    model_config: GPTConfig,
    params: PyTree,
    opt_state: PyTree,
    optimizer: optax.GradientTransformation,
    batched_x: jax.Array,
    batched_y: jax.Array,
) -> tuple[PyTree, PyTree, dict]:
    """Single training step with gradient accumulation."""
    n_grad_acc_steps = batched_x.shape[0]

    def loss_and_grad_fn(p, micro_batch):
        return value_and_grad(loss_fn)(p, micro_batch, model_config, training=False)

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

    # Apply optimizer update
    updates, new_opt_state = optimizer.update(final_grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state, {"loss": avg_loss}


def eval_step(
    params: PyTree,
    batched_x: jax.Array,
    batched_y: jax.Array,
    model_config: GPTConfig,
) -> jax.Array:
    """Evaluation step."""
    n_grad_acc_steps = batched_x.shape[0]

    def loss_loop_body(i, accumulated_loss):
        micro_batch = (batched_x[i], batched_y[i])
        loss = loss_fn(params, micro_batch, model_config, training=False)
        return accumulated_loss + loss

    total_loss = jax.lax.fori_loop(0, n_grad_acc_steps, loss_loop_body, 0.0)
    avg_loss = total_loss / n_grad_acc_steps
    return avg_loss


def run_evaluation(
    step: int,
    params: PyTree,
    val_loader,
    logger,
    compiled_eval_fn: Callable,
):
    """Run validation loop."""
    logger.msg(f"Running validation for step {step}...")
    val_loss_accum = 0.0
    val_steps = 0
    for batched_x, batched_y in val_loader:
        loss = compiled_eval_fn(params, batched_x, batched_y)
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
    
    # Token and position embeddings
    counts["embed"] = params["wte"].size + params["wpe"].size
    
    # Transformer blocks
    for block in params["h"]:
        # Attention
        attn_params = jax.tree_util.tree_leaves(block["attn"])
        counts["attn"] += sum(p.size for p in attn_params)
        
        # MLP
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
        params = init_params(model_config, mesh, key)

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

        # JIT compile training and evaluation functions
        jitted_train_step = jit(
            train_step,
            static_argnames=("model_config", "optimizer"),
            donate_argnums=(1, 2),
        )
        jitted_eval_step = jit(eval_step, static_argnames=("model_config",))

        # AOT compile for the fixed batch shape
        logger.msg("Starting Ahead-of-Time (AOT) compilation...")
        activation_sharding = NamedSharding(mesh, P(*config.activation_sharding))
        n_grad_acc = config.gradient_accumulation_steps
        dummy_x = jnp.zeros((n_grad_acc, config.batch_size, config.block_size), dtype=jnp.int32)
        dummy_y = jnp.zeros_like(dummy_x)
        dummy_x = jax.device_put(dummy_x, activation_sharding)
        dummy_y = jax.device_put(dummy_y, activation_sharding)

        compiled_train_step = jitted_train_step.lower(
            model_config,
            params,
            opt_state,
            optimizer,
            dummy_x,
            dummy_y,
        ).compile()

        compiled_eval_step = jitted_eval_step.lower(
            params, dummy_x, dummy_y, model_config
        ).compile()
        logger.msg("AOT compilation finished.")

        # Load datasets
        logger.msg("Loading training data...")
        train_batches = load_dataset(config, logger, mesh, is_training=True)
        train_loader = iter(train_batches)
        logger.msg(f"Loaded {len(train_batches)} training batches.")

        val_config = dataclasses.replace(config, input_bin=config.input_val_bin)
        logger.msg("Loading validation data...")
        val_batches = load_dataset(val_config, logger, mesh, is_training=False)
        logger.msg(f"Loaded {len(val_batches)} validation batches.")

        # Training loop
        logger.msg("Starting training...")

        for step in range(config.max_iters):
            batched_x, batched_y = next(train_loader)

            # First step we have logs from the initialization we want to keep
            if step > 0:
                log_dict = {}

            # Evaluation
            if step % config.eval_interval == 0:
                val_loss = run_evaluation(
                    step,
                    params,
                    iter(val_batches),
                    logger,
                    compiled_eval_step,
                )
                log_dict["val_loss"] = val_loss
            
            # Training
            params, opt_state, metrics = compiled_train_step(
                params,
                opt_state,
                batched_x,
                batched_y,
            )

            # # Normalize weights
            # params["h"][0]["mlp"]["c_fc"] = params["h"][0]["mlp"]["c_fc"] / jnp.linalg.norm(params["h"][0]["mlp"]["c_fc"], axis=0, keepdims=True)
            # params["h"][0]["mlp"]["c_proj"] = params["h"][0]["mlp"]["c_proj"] / jnp.linalg.norm(params["h"][0]["mlp"]["c_proj"], axis=-1, keepdims=True)

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
        run_evaluation(
            step,
            params,
            iter(val_batches),
            logger,
            compiled_eval_step,
        )
        logger.msg("Training finished.")
        logger.dump(step, params, opt_state, config)
        logger.finish()


if __name__ == "__main__":
    jax.distributed.initialize()
    print("Training starting...")
    print("Found", len(jax.devices()), "devices")
    config = TrainConfig()
    print("Config:", config)
    train_loop(config)
    print("Training finished.")
