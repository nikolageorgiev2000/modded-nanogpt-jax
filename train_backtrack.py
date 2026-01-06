# # Imports and visualizers

import jax
import jax.numpy as jnp
import numpy as np
import numba as nb
from jax import vmap
from dataclasses import dataclass

import matplotlib
from matplotlib import pyplot as plt
import os

import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from tqdm import tqdm


import model
import train


@dataclass
class DatasetConfig:
    """Configuration for backtrack task dataset generation."""
    dataset_name: str = 'backtracktask'
    n_edges: int = 30  # number of edges per DAG
    n_interleaved: int = 2  # number of DAGs interleaved together
    branching_factor: int = 2  # branching factor of the tree
    height: int = 5  # tree height (starting from level 1, ending at level `height`)
    n_data: int = 2**19 * 3  # total number of data samples
    n_data_batch: int = 2**18  # batch size for data generation
    seed: int = 123  # base seed for random generation

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the un-sampled balanced tree."""
        if self.branching_factor == 1:
            return self.height
        return (self.branching_factor ** self.height - 1) // (self.branching_factor - 1)

    @property
    def adj_list_len(self) -> int:
        """Length of adjacency list: 3 tokens per edge pair, +1 for root self-loops."""
        return 3 * (self.n_edges + 1)

    @property
    def sample_len(self) -> int:
        """Total sample length."""
        return self.adj_list_len * self.n_interleaved

    @property
    def token_arr(self) -> np.ndarray:
        """Token array including separator token."""
        return np.arange((self.n_edges + 1) * self.n_interleaved + 1, dtype=np.uint16)

    @property
    def num_passes(self) -> int:
        """Number of passes for data generation."""
        return self.n_data // self.n_data_batch

    def validate(self) -> None:
        """Validate configuration parameters."""
        assert self.n_nodes >= self.n_edges + 1, \
            f"Balanced tree too small: n_nodes={self.n_nodes} < n_edges+1={self.n_edges + 1}"

jax.distributed.initialize()


def heatmap(data, row_labels, col_labels, ax=None, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(range(data.shape[1]), labels=col_labels,
                  rotation=0, ha="right", rotation_mode="anchor", fontsize=6)
    ax.set_yticks(range(data.shape[0]), labels=row_labels, fontsize=6)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    for ticklabel in ax.get_yticklabels():
        ticklabel.set_horizontalalignment("right")
    for ticklabel in ax.get_xticklabels():
        ticklabel.set_horizontalalignment("center")

    return im


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def dag_pos_dot(G):
    return graphviz_layout(G, prog="dot")

# ---
# 
# # Backtrack Task
# 
# Given the adjacency list of a DAG consisting of several disconnected components and a start node (e.g. a leaf), return the root of its component. This is a global feature while the adjacency list only provides local features. Backtracking takes O(d) time where d is the max depth of a component, but this alone makes the problem difficult for a Transformer, which has fixed depth.
# There are two approaches that would work but are just circumventing the interesting question:
# 1. Return the entire backtracking-path rather than just the root. So the compute scales over tokens rather than model depth.
# 2. Use a positional encoding for the DAG, creating a geometry which the Transformer can learn to query, essentially encoding distances between nodes in the input. This is kinda cheating, since it now makes querying for the root node constant time, since each node can query the farthest node from it that is reachable – the problem is solved implicitly in the definition of the positional encoding.
# In particular, if the model architecture is parametric, so it adapts based on the graph depth, can we learn the right circuit for backtracking to the root of a component?
# 
# We can start by fixing the depth to a low value. We know depth 1 can be solved with a single induction head, since it merely reads the parent of each node in the adjacency list, which is the root. At depth 2, the model needs to query for the parent of each node's parent. The depth will then be logarithmic in the depth of the tree, since the distance the pointer covers doubles with each step.
# 
# Implementation:
# Take as input EDGE PARENT CHILD EDGE PARENT CHILD ... EDGE PARENT CHILD.
# Use causal attention but enforce the DAG – ensure a node can be a parent only if it has already been a child in the sequence thus far.
# Exclude outputs for SEP and PARENT from the loss.
# For all CHILD tokens have the root token as the target.
# Each edge creates a local constraint: the nodes must be in the same component.
# The root of a component is identified by the fact it has an edge to itself.
# Also it is the first node to appear in an edge pair for its component.
# To generate the data, create a few DAGs, create the edge pairs, then interleave them randomly.
# 
# 
# 

# ---------- RNG: xorshift32 (fast, njit-friendly) ----------

@nb.njit(inline="always")
def xorshift32(state: np.uint32) -> np.uint32:
    state ^= (state << np.uint32(13))
    state ^= (state >> np.uint32(17))
    state ^= (state << np.uint32(5))
    return state

@nb.njit(inline="always")
def rand_below(state: np.uint32, high: int):
    state = xorshift32(state)
    return state, int(state % np.uint32(high))

# ---------- Tree size (nodes) ----------

@nb.njit(inline="always")
def balanced_tree_nnodes(r: int, h: int) -> int:
    # levels 0..h inclusive
    if r == 1:
        return h
    return (r ** h - 1) // (r - 1)

# ---------- Generate ONE DAG edge list (subtree sample) into a provided array ----------

@nb.njit(inline="always")
def sample_one_edges_into(r: int, h: int, n_edges: int,
                          node_offset: int, state: np.uint32,
                          out_edges: np.ndarray) -> np.uint32:
    """
    Writes (n_edges, 2) into out_edges using heap/BFS r-ary indexing.
    Node labels are offset by node_offset to avoid collisions between graphs.
    Assumes n_edges <= n_nodes-1 (otherwise you can't have that many tree edges).
    """
    n_nodes = balanced_tree_nnodes(r, h)
    frontier = np.empty(n_nodes, dtype=np.int32)
    next_child = np.zeros(n_nodes, dtype=np.int16)

    frontier[0] = 0
    frontier_size = 1
    m = 0

    while m < n_edges and frontier_size > 0:
        state, idx = rand_below(state, frontier_size)
        u = frontier[idx]

        base = r * u + 1
        j = int(next_child[u])

        # exhausted parent or out of bounds (implicit height limit) => remove from frontier (swap-pop)
        if j >= r or (base + j) >= n_nodes:
            frontier_size -= 1
            frontier[idx] = frontier[frontier_size]
            continue

        v = base + j
        next_child[u] = np.int16(j + 1)

        out_edges[m, 0] = u + node_offset
        out_edges[m, 1] = v + node_offset
        m += 1

        # child becomes available as a future parent
        frontier[frontier_size] = v
        frontier_size += 1

    # If you ever request too many edges, pad with -1 (shouldn't happen if n_edges <= n_nodes-1)
    while m < n_edges:
        out_edges[m, 0] = -1
        out_edges[m, 1] = -1
        m += 1

    return state

# ---------- Generate k edge lists and interleave them uniformly ----------

@nb.njit(parallel=True, fastmath=True)
def sample_interleaved_edges_many(r: int, h: int, n_edges: int, k: int,
                                  n_out: int, seed0: int) -> np.ndarray:
    """
    r: branching factor
    h: height (starts at 1)
    n_edges: number of edges per list
    k: number of lists
    n_out: number of output samples
    seed0: seed

    Returns: out shape (n_out, k*n_edges, 2) int32
    Each output is a uniform random interleaving of k DAG edge lists,
    preserving within-list order, with disjoint node IDs via offsets.
    """
    n_nodes = balanced_tree_nnodes(r, h)
    out = np.empty((n_out, k * n_edges, 2), dtype=np.int32)

    for g in nb.prange(n_out):
        # independent RNG stream per output
        state = (np.uint32(seed0)
                 ^ np.uint32(0x9E3779B9)
                 ^ np.uint32(g * 747796405))

        # Generate k lists of edges (local buffer)
        lists = np.empty((k, n_edges, 2), dtype=np.int32)
        for t in range(k):
            offset = t * n_nodes
            state = sample_one_edges_into(r, h, n_edges, offset, state, lists[t])

        # Interleave them uniformly:
        # remaining counts r[i] start at n_edges
        rem = np.empty(k, dtype=np.int32)
        pos = np.zeros(k, dtype=np.int32)
        for i in range(k):
            rem[i] = n_edges

        remaining_total = k * n_edges

        for m in range(k * n_edges):
            # draw u in [0, remaining_total)
            state, u = rand_below(state, remaining_total)

            # choose list index by walking cumulative remaining
            cum = 0
            chosen = 0
            for i in range(k):
                cum += rem[i]
                if u < cum:
                    chosen = i
                    break

            p = pos[chosen]
            out[g, m, 0] = lists[chosen, p, 0]
            out[g, m, 1] = lists[chosen, p, 1]

            pos[chosen] = p + 1
            rem[chosen] -= 1
            remaining_total -= 1

    return out


@nb.njit(inline="always")
def lower_bound(sorted_arr: np.ndarray, x: int) -> int:
    lo = 0
    hi = sorted_arr.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_arr[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo

@nb.njit(parallel=True)
def compact_nodes_many(interleaved_edges: np.ndarray):
    """
    interleaved_edges: (n_out, k*n_edges, 2) int32
    n_edges: int
    k: int
    Returns:
      - out_edges: (n_out, k*n_edges, 2) int32
    """
    n_out, m, _ = interleaved_edges.shape

    out = np.empty_like(interleaved_edges)

    for g in nb.prange(n_out):
        # collect node IDs (2*m endpoints)
        nodes = np.empty(2 * m, dtype=np.int32)
        for i in range(m):
            nodes[2 * i]     = interleaved_edges[g, i, 0]
            nodes[2 * i + 1] = interleaved_edges[g, i, 1]

        # sort + unique
        nodes.sort()
        uniq = np.empty(2 * m, dtype=np.int32)
        u = 0
        last = nodes[0]
        uniq[u] = last
        u += 1
        for i in range(1, nodes.shape[0]):
            v = nodes[i]
            if v != last:
                uniq[u] = v
                u += 1
                last = v

        compact_inds = np.arange(u, dtype=np.int32)

        # relabel edges
        for i in range(m):
            a = interleaved_edges[g, i, 0]
            b = interleaved_edges[g, i, 1]
            ia = lower_bound(uniq[:u], a)
            ib = lower_bound(uniq[:u], b)
            out[g, i, 0] = compact_inds[ia]
            out[g, i, 1] = compact_inds[ib]

    return out



# Create dataset configuration
data_cfg = DatasetConfig(
    dataset_name='backtracktask',
    n_edges=30,
    n_interleaved=2,
    branching_factor=2,
    height=5,
    n_data=2**19 * 3,
    n_data_batch=2**18,
    seed=123,
)
data_cfg.validate()

print(f"Token count: {data_cfg.token_arr.shape}")

if not os.path.exists(data_cfg.dataset_name):
    os.makedirs(data_cfg.dataset_name)

def make_memmap(path, shape, dtype):
    print(f"Creating memmap for {path} with shape {shape} and dtype {dtype}")
    return np.memmap(path, mode='w+', dtype=dtype, shape=shape)

total_len = data_cfg.n_data * data_cfg.sample_len
n_rows = 3
val_len = total_len // 16

# Total shapes for memmaps (we'll simply append pass-by-pass)
train_shape = (n_rows, total_len - val_len)
val_shape   = (n_rows, val_len)

print(f"Opening train memmap: {os.path.join(data_cfg.dataset_name, 'train_with_mask_n_targets.bin')}")
train_mm = make_memmap(os.path.join(data_cfg.dataset_name, 'train_with_mask_n_targets.bin'), train_shape, np.uint16)
print(f"Opening val memmap: {os.path.join(data_cfg.dataset_name, 'val_with_mask_n_targets.bin')}")
val_mm   = make_memmap(os.path.join(data_cfg.dataset_name, 'val_with_mask_n_targets.bin'),   val_shape,   np.uint16)

train_pos = 0
val_pos = 0

for rep in tqdm(range(data_cfg.num_passes)):
    edges = sample_interleaved_edges_many(data_cfg.branching_factor, data_cfg.height, data_cfg.n_edges, data_cfg.n_interleaved, data_cfg.n_data_batch, seed0=data_cfg.seed+rep)
    levels = np.ceil(np.log((((edges % data_cfg.n_nodes)+1) * (data_cfg.branching_factor - 1))+1) / np.log(data_cfg.branching_factor))
    compacted = compact_nodes_many(edges)
    roots_tiled = np.tile(np.arange(data_cfg.n_interleaved)*(data_cfg.n_edges+1), (data_cfg.n_data_batch, 1))
    root_self_loops = np.repeat(roots_tiled, 2, axis=-1).reshape(compacted.shape[0], data_cfg.n_interleaved, 2)
    compacted_with_root_self_loops = np.concatenate([root_self_loops, compacted], axis=-2)
    data = np.concatenate([data_cfg.token_arr[-1]*np.ones_like(compacted_with_root_self_loops), compacted_with_root_self_loops], axis=-1)[:,:,1:]
    targets = np.zeros_like(data)
    targets[..., 2] = (data[..., 2] // (data_cfg.n_edges+1)) * (data_cfg.n_edges+1)
    labels = np.concatenate([np.ones((len(levels), data_cfg.n_interleaved, 2)), levels], axis=1) # concatenate self-loops for the roots
    labels = np.concatenate([np.zeros_like(labels), labels], axis=2)[:,:,1:] # result has last dim like [0, depth, depth]
    mask = np.array(labels)
    mask[..., :2] = 0 # only keep child tokens in the loss
    # print(f"levels: {levels.shape}", f"data: {data.shape}", f"targets: {targets.shape}", f"mask: {mask.shape}")

    data, targets, mask = (
        jnp.array(data, dtype=jnp.uint16),
        jnp.array(targets, dtype=jnp.uint16),
        jnp.array(mask, dtype=jnp.uint16),
    )
    key_d2 = jax.random.PRNGKey(2 + rep)
    key_perms = jax.random.split(key_d2, data_cfg.n_data_batch)
    print("Generating token permutations (this can be slow)...")
    tok_permutations = vmap(lambda k : jnp.concatenate(
        [jax.random.permutation(k, len(data_cfg.token_arr)-1), data_cfg.token_arr[-1:]]
    ))(key_perms)
    print("Applying permutations to data and targets (this can be slow)...")
    data_from_perms = vmap(lambda i : tok_permutations[i][data.reshape(data_cfg.n_data_batch, -1)[i]])(jnp.arange(data_cfg.n_data_batch))
    targets_from_perms = vmap(lambda i : tok_permutations[i][targets.reshape(data_cfg.n_data_batch, -1)[i]])(jnp.arange(data_cfg.n_data_batch))
    
    data_from_perms_flat, targets_from_perms_flat = data_from_perms.flatten(), targets_from_perms.flatten()
    mask_flat = mask.flatten()
    data_with_mask_n_targets = jnp.stack([
        data_from_perms_flat, targets_from_perms_flat, mask_flat
    ], axis=0)
    val_data_len = data_with_mask_n_targets.shape[1] // 16
    assert val_data_len == val_len // data_cfg.num_passes
    train_ids = data_with_mask_n_targets[:, :-val_data_len].astype(np.uint16)
    val_ids = data_with_mask_n_targets[:, -val_data_len:].astype(np.uint16)

    print(f"Writing train chunk [{train_pos}:{train_pos+train_ids.shape[1]}] and val chunk [{val_pos}:{val_pos+val_ids.shape[1]}] to disk...")
    # Write this chunk to the train and val memmaps
    train_mm[:, train_pos:train_pos+train_ids.shape[1]] = np.array(train_ids, dtype=np.uint16)
    val_mm[:, val_pos:val_pos+val_ids.shape[1]] = np.array(val_ids, dtype=np.uint16)
    train_pos += train_ids.shape[1]
    val_pos += val_ids.shape[1]

    print(f"Pass {rep+1}/{data_cfg.num_passes} done. train_pos={train_pos}, val_pos={val_pos}")

print("Flushing memmaps to disk...")
train_mm.flush()
val_mm.flush()
print("Done!")


def extract_interleaved_edges(x, sample_ind = 0):
    return x[sample_ind*data_cfg.sample_len : (sample_ind+1)*data_cfg.sample_len]

i = 0
G = nx.DiGraph()
G.add_edges_from(list(map(tuple, extract_interleaved_edges(val_ids[0], i).reshape(-1, 3)[:, 1:].tolist())))
pos = dag_pos_dot(G)
plt.figure(figsize=(8, 6))
nx.draw(
    G, pos=pos,
    with_labels=True, node_size=120,
    node_color="lightblue", edge_color="gray",
    width=0.8, arrows=True, arrowsize=10
)
plt.savefig(f"{data_cfg.dataset_name}/example_input.png", bbox_inches="tight")
plt.close()



config = train.TrainConfig(
    input_bin=f"{data_cfg.dataset_name}/train_with_mask_n_targets.bin",
    input_val_bin=f"{data_cfg.dataset_name}/val_with_mask_n_targets.bin",
    embd_dim = 512,
    head_dim = 256,
    n_layer = 5,
    block_size = data_cfg.sample_len, # should match the task sequence length so tasks are independently trained on
    batch_size = 256,
    gradient_accumulation_steps = 1,
    max_iters = 100_000,
    eval_iters = 25, # val_data_len // 64, # number of examples // batch_size
    learning_rate = 6e-4,
    min_lr = 0,
    warmup_iters = 5_000,
    lr_decay_iters = 100_000,
    vocab_size = len(data_cfg.token_arr),
    use_masked_loss = True,
    use_custom_target=True,
    use_mlp = False,
    off_by_one_attn = False,
    use_pope = True,
    # freeze_params=("wte",),
    max_seq_len = 2*data_cfg.sample_len,

    num_loss_groups = data_cfg.height,
    loss_combiner = lambda sums, counts: sums.sum() / counts.sum(),

    pos_encoding_base = 2*data_cfg.sample_len,
    
    log_interval = 10_000,
    eval_interval = 1_000,

    # seed = 15,
)

params = train.train_loop(config)

print("FINISHED TRAINING")
print("Saving config and params")
# save config and params using numpy
np.savez(f"{data_cfg.dataset_name}/config.npz", config=config)
np.savez(f"{data_cfg.dataset_name}/params.npz", params=params)



# load config and params using numpy
config = np.load(f"{data_cfg.dataset_name}/config.npz", allow_pickle=True)["config"].item()
params = np.load(f"{data_cfg.dataset_name}/params.npz", allow_pickle=True)["params"].item()

print(config)
print(params)


rope_params = model.precompute_pope(config.get_model_config(), None) if config.use_pope else model.precompute_rope(config.get_model_config(), None)

error_pct = jnp.empty(100)
for i in tqdm(range(len(error_pct))):
    test_input_ids = extract_interleaved_edges(val_ids[0], i)
    test_target_ids = extract_interleaved_edges(val_ids[1], i)
    test_mask_ids = extract_interleaved_edges(val_ids[2], i)
    res = model.gpt_forward(params, rope_params, test_input_ids[None,:], config.get_model_config())[0].argmax(axis=-1)

    diff = jnp.count_nonzero(test_mask_ids * (test_target_ids - res))
    error_pct = error_pct.at[i].set(diff / test_mask_ids.sum())
print(f"Error count: {error_pct.mean()}")


i = 0
test_input_ids = extract_interleaved_edges(val_ids[0], i)
test_target_ids = extract_interleaved_edges(val_ids[1], i)
test_mask_ids = extract_interleaved_edges(val_ids[2], i)
test_input_ids * test_mask_ids, res * test_mask_ids


preds, attn_weights = model.gpt_forward(params, rope_params, test_input_ids[None,:], config.get_model_config(), return_attn_weights=True)
res = preds[0].argmax(axis=-1)


query_subset = jnp.arange(2, data_cfg.sample_len, 3)
key_subset = jnp.concatenate([jnp.arange(1, data_cfg.sample_len, 3)[None, :], jnp.arange(2, data_cfg.sample_len, 3)[None, :]]).T.flatten()

# Determine num_layers and num_heads from attn_weights structure
num_layers = len(attn_weights)
num_heads = attn_weights[0][0].shape[0]  # Assuming shape: [num_heads, seq, seq]

fig, axes = plt.subplots(num_layers, num_heads, figsize=(6 * num_heads, 6 * num_layers), squeeze=False)

for l in range(num_layers):
    for h in range(num_heads):
        # sum over all heads because we only need two heads in the first layer
        # remaining layers are splitting the single head into two (seems arbitrary)
        selected_attn = attn_weights[l][0][h][query_subset[:, None], key_subset[None, :]].astype(np.float32)
        ax = axes[l, h]
        im = heatmap(
            selected_attn,
            test_input_ids[query_subset], test_input_ids[key_subset], ax=ax,
            cmap="YlGnBu"
        )
        ax.set_title(f"Layer {l}, Head {h}")

fig.tight_layout()
plt.show()
