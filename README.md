# Modded-NanoGPT-JAX

This repository contains a pure JAX selective port (plus other optimizations) of the [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) speedrun, optimized for Google TPUs. The goal is to train a GPT-style language model to a validation loss of **≤ 3.28** on a FineWeb subset in the shortest time possible.

This implementation achieves the target in approximately **10 minutes on a TPU v6e-8**.

For a detailed write-up on the porting process, optimizations, and performance analysis of the initial run, please see the accompanying blog post: [The modded nanogpt speedrun, but in JAX and on TPUs](https://nor-blog.pages.dev/posts/2025-08-21-modded-nanogpt-jax/).

The training script is self-contained in `train.py` and is written using core JAX APIs, without high-level libraries like Flax, Optax, or Orbax, in the spirit of the original speedrun.

## Performance

The current implementation successfully reaches the target validation loss of ≤ 3.28 in approximately ~10 minutes on a single TPU v6e-8 node.

Key optimizations enabling this performance include selected changes from the modded-nanogpt repository, as well as other changes on top that make it better for TPUs.

## How to Run

### 1. Setup

Clone the repository and install the necessary dependencies. You will need a JAX installation configured for your accelerator (e.g., TPU or GPU).

```bash
git clone <repository_url>
cd <repository_name>

# Install dependencies (example for TPU)
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install numpy huggingface_hub einops
```

### 2. Download Data

The training script requires the FineWeb10B dataset, pre-tokenized with the GPT-2 tokenizer. The `cached_fineweb10B.py` script downloads the necessary data shards from the Hugging Face Hub, saving significant data preparation time.

```bash
python cached_fineweb10B.py 10

# Or, for a quicker test, download a subset of the data (e.g., 1 chunk)
python cached_fineweb10B.py 1
```

The data will be saved to a `fineweb10B/` directory.

### 3. Start Training

Run the training script. The script will first perform AOT compilation for all batch shapes, which may take a few minutes but will be cached to a directory. After compilation, the training loop will begin.

```bash
python train.py
```

Logs, metrics, and model checkpoints will be saved to a uniquely identified directory inside `logs/`.

## Code Overview

  - `train.py`: A single, self-contained script for training the model. It includes:
      - The model architecture (GPT-2 style with modern enhancements).
      - Data loading and batching logic with sequence length warmup.
      - A multi-optimizer implementation featuring Adam and the Muon optimizer.
      - The main training and evaluation loop with AOT compilation.
      - Custom logging and checkpointing utilities.
  - `cached_fineweb10B.py`: A utility script to download pre-processed FineWeb10B data shards.
  - `records/sub10m.txt`: A log file from a successful training run on a TPU v6e-8, demonstrating that the 3.28 validation loss target is met. Note that for statistical rigor, multiple runs are required, and this might be considered for future runs.

## Acknowledgments

This project is a direct port and adaptation of the incredible work done by the open-source community on the original NanoGPT speedrun.

  - **Andrej Karpathy** for [NanoGPT](https://github.com/karpathy/nanogpt), which started it all.
  - **Keller Jordan and all contributors** to the [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) repository for pioneering architecture and optimizer improvements.
