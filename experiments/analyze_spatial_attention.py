"""Analyze spatial attention weights for dead/intermittent/functional nodes.

Loads the baseline STAEformer, hooks into spatial attention layers,
and examines how much attention dead/noisy nodes receive and give.
"""
import os, sys
sys.path.insert(0, '/data/pretrainingbasicts')

import torch
import numpy as np
from collections import defaultdict

# ---- Model setup ----
from baselines.STAEformer.arch.staeformer_arch import STAEformer

MODEL_PARAM = {
    "num_nodes": 893, "in_steps": 12, "out_steps": 12,
    "steps_per_day": 288, "input_dim": 3, "output_dim": 1,
    "input_embedding_dim": 24, "tod_embedding_dim": 24, "dow_embedding_dim": 24,
    "spatial_embedding_dim": 0, "adaptive_embedding_dim": 24,
    "feed_forward_dim": 256, "num_heads": 4, "num_layers": 1,
    "dropout": 0.1, "use_mixed_proj": True,
}

CKPT_PATH = 'checkpoints/ContextContrastive_baseline_3mo/xtraffic_SAN_BERNARDINO_30_12_12/040af4f5bcb37097bc5263d7f350174b/STAEformer_Baseline_best_val_MAE.pt'

device = torch.device('cuda:1')

# ---- Load model ----
model = STAEformer(**MODEL_PARAM).to(device)
ckpt = torch.load(CKPT_PATH, map_location=device)
state = ckpt['model_state_dict']
# Checkpoint uses modular format (encoder./spatial./decoder. prefixes)
# Map to monolithic STAEformer keys
new_state = {}
for k, v in state.items():
    if k.startswith('encoder.'):
        new_state[k[len('encoder.'):]] = v
    elif k.startswith('spatial.'):
        new_state[k[len('spatial.'):]] = v
    elif k.startswith('decoder.'):
        new_state[k[len('decoder.'):]] = v
    else:
        new_state[k] = v
model.load_state_dict(new_state)
model.eval()
print("Model loaded successfully")

# ---- Hook into spatial attention to capture attention scores ----
attn_scores_spatial = []

def hook_spatial_attn(module, input, output):
    """Hook on AttentionLayer inside spatial SelfAttentionLayer.
    Captures attn_score after softmax."""
    pass  # We need to modify the forward to capture scores

# Instead of hooks (which can't easily capture intermediate tensors),
# let's monkey-patch the AttentionLayer.forward to store attn_scores
original_forward = model.attn_layers_s[0].attn.__class__.forward

def patched_forward(self, query, key, value):
    batch_size = query.shape[0]
    tgt_length = query.shape[-2]
    src_length = key.shape[-2]

    query = self.FC_Q(query)
    key = self.FC_K(key)
    value = self.FC_V(value)

    query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
    key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
    value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

    key = key.transpose(-1, -2)
    attn_score = (query @ key) / self.head_dim**0.5

    if self.mask:
        mask = torch.ones(tgt_length, src_length, dtype=torch.bool, device=query.device).tril()
        attn_score.masked_fill_(~mask, -torch.inf)

    attn_score = torch.softmax(attn_score, dim=-1)

    # Store attention scores: shape (num_heads*batch_size, in_steps, num_nodes, num_nodes)
    attn_scores_spatial.append(attn_score.detach().cpu())

    out = attn_score @ value
    out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)
    out = self.out_proj(out)
    return out

# Apply monkey-patch to ALL spatial attention layers
for layer in model.attn_layers_s:
    layer.attn.forward = lambda q, k, v, _self=layer.attn: patched_forward(_self, q, k, v)

print("Spatial attention hooks installed")

# ---- Load data and scaler ----
from basicts.scaler import ZScoreScaler

data = np.memmap('datasets/xtraffic/SAN_BERNARDINO/data.dat', dtype='float32', mode='r',
                 shape=(105120, 893, 5))
# data_range = (0, 26280), 3 months
data_3mo = np.array(data[:26280])  # load into memory

# Compute zero rates for node categorization
flow = data_3mo[:, :, 0]
zero_rates = (flow == 0).mean(axis=0)
print(f"Zero rates computed: min={zero_rates.min():.3f}, max={zero_rates.max():.3f}")

# Node categories
dead_nodes = np.where(zero_rates > 0.9)[0]
major_fail = np.where((zero_rates > 0.5) & (zero_rates <= 0.9))[0]
partial_fail = np.where((zero_rates > 0.05) & (zero_rates <= 0.5))[0]
functional = np.where(zero_rates <= 0.05)[0]
print(f"Dead: {len(dead_nodes)}, Major fail: {len(major_fail)}, Partial: {len(partial_fail)}, Functional: {len(functional)}")

# Find intermittent nodes (high transition count)
transitions = np.diff((flow > 0).astype(int), axis=0)
transition_counts = np.abs(transitions).sum(axis=0)
top_intermittent = np.argsort(transition_counts)[-20:]  # top 20 most intermittent
print(f"Top intermittent nodes (by transition count): {top_intermittent}")
print(f"Their transition counts: {transition_counts[top_intermittent]}")
print(f"Their zero rates: {zero_rates[top_intermittent]}")

# ---- Scaler ----
scaler = ZScoreScaler(
    dataset_name='xtraffic/SAN_BERNARDINO',
    train_ratio=0.6,
    norm_each_channel=True,
    rescale=True,
)

# ---- Build test samples ----
# Use test split: train=60%, val=20%, test=20%
# data_range = (0, 26280), test starts at 80% = 21024
test_start = int(26280 * 0.8)
INPUT_LEN = 12
OUTPUT_LEN = 12

# Sample some test windows
num_samples = 200
np.random.seed(42)
sample_starts = np.random.randint(test_start, 26280 - INPUT_LEN - OUTPUT_LEN, size=num_samples)

print(f"\nRunning {num_samples} samples through model...")

# Process in batches
batch_size = 16
all_attn_received = []  # attention received by each node (as target of attention)
all_attn_given = []     # attention given by each node (as source)

for batch_idx in range(0, num_samples, batch_size):
    batch_starts = sample_starts[batch_idx:batch_idx + batch_size]
    B = len(batch_starts)

    # Build input: [B, 12, 893, 3] (flow, tod, dow)
    history = np.zeros((B, INPUT_LEN, 893, 3), dtype=np.float32)
    for i, s in enumerate(batch_starts):
        raw = data_3mo[s:s+INPUT_LEN, :, :]  # [12, 893, 5]
        history[i, :, :, 0] = raw[:, :, 0]  # flow
        history[i, :, :, 1] = raw[:, :, 3]  # tod
        history[i, :, :, 2] = raw[:, :, 4]  # dow

    h_tensor = torch.tensor(history, dtype=torch.float32).to(device)

    # Apply scaler to flow channel (channel 0)
    h_tensor[..., 0:1] = scaler.transform(h_tensor[..., 0:1])

    # Clear stored attention scores
    attn_scores_spatial.clear()

    with torch.no_grad():
        _ = model(h_tensor, None, batch_seen=0, epoch=0, train=False)

    # Process attention scores
    # attn_scores_spatial[0] shape: (num_heads*B, in_steps, num_nodes, num_nodes)
    # After softmax: attn[..., i, j] = how much node i attends to node j
    for attn in attn_scores_spatial:
        num_heads = 4
        # Reshape: (num_heads, B, in_steps, N, N)
        attn_reshaped = attn.reshape(num_heads, B, 12, 893, 893)
        # Average over heads and timesteps: (B, N, N)
        attn_avg = attn_reshaped.mean(dim=(0, 2))  # (B, N, N)

        # Attention RECEIVED by each node: sum over source dim (how much others attend TO this node)
        # attn_avg[b, i, j] = node i attends to node j
        # So attn received by node j = sum over i of attn_avg[b, i, j]
        attn_received = attn_avg.sum(dim=1)  # (B, N) - sum over query nodes

        # Attention GIVEN by each node: this is just the row sum which is always 1 (softmax)
        # More interesting: self-attention weight (diagonal)
        self_attn = torch.diagonal(attn_avg, dim1=1, dim2=2)  # (B, N)

        all_attn_received.append(attn_received.numpy())
        all_attn_given.append(self_attn.numpy())

all_attn_received = np.concatenate(all_attn_received, axis=0)  # (num_samples, 893)
all_attn_given = np.concatenate(all_attn_given, axis=0)  # (num_samples, 893) - self-attn

print(f"Attention collected: received shape={all_attn_received.shape}, self-attn shape={all_attn_given.shape}")

# ---- Analysis ----
print("\n" + "="*70)
print("SPATIAL ATTENTION ANALYSIS")
print("="*70)

# 1. Average attention RECEIVED by category
print("\n--- Attention RECEIVED (how much other nodes attend to this node) ---")
print("(Higher = more influence on neighbors' predictions)")
for name, nodes in [("Dead", dead_nodes), ("Major fail", major_fail),
                     ("Partial fail", partial_fail), ("Functional", functional)]:
    if len(nodes) == 0:
        continue
    recv = all_attn_received[:, nodes].mean()
    recv_std = all_attn_received[:, nodes].mean(axis=0).std()
    # Uniform attention would be 893/893 = 1.0 per node
    # But softmax sums to 1 per query node, so total received is N=893
    # Uniform per node = 1.0
    print(f"  {name:15s} ({len(nodes):3d} nodes): mean={recv:.4f}, std={recv_std:.4f}, "
          f"vs uniform={1.0:.4f} (ratio={recv/1.0:.3f})")

# 2. Self-attention by category
print("\n--- Self-Attention (diagonal of attention matrix) ---")
print("(Higher = node relies more on itself, less on neighbors)")
for name, nodes in [("Dead", dead_nodes), ("Major fail", major_fail),
                     ("Partial fail", partial_fail), ("Functional", functional)]:
    if len(nodes) == 0:
        continue
    self_attn = all_attn_given[:, nodes].mean()
    self_attn_std = all_attn_given[:, nodes].mean(axis=0).std()
    uniform_self = 1.0 / 893
    print(f"  {name:15s} ({len(nodes):3d} nodes): mean={self_attn:.6f}, std={self_attn_std:.6f}, "
          f"vs uniform={uniform_self:.6f} (ratio={self_attn/uniform_self:.2f})")

# 3. Top intermittent nodes analysis
print("\n--- Top 20 Intermittent Nodes (most on/off transitions) ---")
print(f"{'Node':>6s} {'Transitions':>12s} {'ZeroRate':>10s} {'AttnRecv':>10s} {'SelfAttn':>10s}")
for node in top_intermittent:
    recv = all_attn_received[:, node].mean()
    self_a = all_attn_given[:, node].mean()
    print(f"{node:6d} {transition_counts[node]:12.0f} {zero_rates[node]:10.3f} {recv:10.4f} {self_a:10.6f}")

# 4. Attention from functional nodes TO dead nodes
print("\n--- Cross-category Attention Flow ---")
print("(How much do functional nodes attend to different categories?)")
# For each functional query node, what fraction of attention goes to each category?
# attn_avg[b, i, j] = node i attends to node j
# We want: for i in functional, sum of attn to j in dead/major/partial/functional
# Need to recompute from raw data
attn_scores_spatial.clear()

# Run a small batch to get raw attention matrix
batch_starts_small = sample_starts[:32]
B_small = len(batch_starts_small)
history_small = np.zeros((B_small, INPUT_LEN, 893, 3), dtype=np.float32)
for i, s in enumerate(batch_starts_small):
    raw = data_3mo[s:s+INPUT_LEN, :, :]
    history_small[i, :, :, 0] = raw[:, :, 0]
    history_small[i, :, :, 1] = raw[:, :, 3]
    history_small[i, :, :, 2] = raw[:, :, 4]

h_small = torch.tensor(history_small, dtype=torch.float32).to(device)
h_small[..., 0:1] = scaler.transform(h_small[..., 0:1])

with torch.no_grad():
    _ = model(h_small, None, batch_seen=0, epoch=0, train=False)

attn = attn_scores_spatial[0]  # (num_heads*B, 12, 893, 893)
attn_reshaped = attn.reshape(4, B_small, 12, 893, 893)
attn_avg = attn_reshaped.mean(dim=(0, 2)).numpy()  # (B, N, N)

# For functional query nodes, compute attention to each category
for query_cat, query_name, query_nodes in [
    (functional, "Functional", functional),
    (partial_fail, "Partial fail", partial_fail),
]:
    if len(query_nodes) == 0:
        continue
    print(f"\n  From {query_name} nodes attend to:")
    for target_name, target_nodes in [("Dead", dead_nodes), ("Major fail", major_fail),
                                       ("Partial fail", partial_fail), ("Functional", functional)]:
        if len(target_nodes) == 0:
            continue
        # attn_avg[:, query_nodes, :][:, :, target_nodes] -> (B, |query|, |target|)
        flow_to_target = attn_avg[:, query_nodes][:, :, target_nodes].sum(axis=-1).mean()
        # Normalize: what fraction of total attention goes to this category?
        frac = flow_to_target  # softmax already normalized to sum=1 per query
        expected_frac = len(target_nodes) / 893
        print(f"    → {target_name:15s}: {frac:.4f} (expected uniform: {expected_frac:.4f}, ratio: {frac/expected_frac:.3f})")

# 5. Attention during "failure" vs "working" timesteps for intermittent nodes
print("\n--- Intermittent Node: Attention when WORKING vs FAILING ---")
# Pick node 149 (high transition count) or top intermittent
target_node = top_intermittent[-1]  # highest transition count
print(f"Analyzing node {target_node} (transitions={transition_counts[target_node]:.0f}, zero_rate={zero_rates[target_node]:.3f})")

# Find timesteps where this node has zero flow vs non-zero flow in test set
test_flow = data_3mo[test_start:, :, 0]  # (remaining, 893)
node_flow = test_flow[:, target_node]

working_times = np.where(node_flow > 0)[0]
failing_times = np.where(node_flow == 0)[0]
print(f"  Working timesteps: {len(working_times)}, Failing timesteps: {len(failing_times)}")

# Sample windows from each
n_each = min(50, len(working_times) - INPUT_LEN - OUTPUT_LEN, len(failing_times) - INPUT_LEN - OUTPUT_LEN)
if n_each > 10:
    np.random.seed(123)

    def get_attn_for_starts(starts):
        B = len(starts)
        h = np.zeros((B, INPUT_LEN, 893, 3), dtype=np.float32)
        for i, s in enumerate(starts):
            actual_s = test_start + s
            raw = data_3mo[actual_s:actual_s+INPUT_LEN, :, :]
            h[i, :, :, 0] = raw[:, :, 0]
            h[i, :, :, 1] = raw[:, :, 3]
            h[i, :, :, 2] = raw[:, :, 4]
        ht = torch.tensor(h, dtype=torch.float32).to(device)
        ht[..., 0:1] = scaler.transform(ht[..., 0:1])
        attn_scores_spatial.clear()
        with torch.no_grad():
            _ = model(ht, None, batch_seen=0, epoch=0, train=False)
        a = attn_scores_spatial[0].reshape(4, B, 12, 893, 893)
        return a.mean(dim=(0, 2)).numpy()  # (B, N, N)

    # Working samples
    working_starts = working_times[working_times < len(test_flow) - INPUT_LEN - OUTPUT_LEN]
    failing_starts = failing_times[failing_times < len(test_flow) - INPUT_LEN - OUTPUT_LEN]

    working_sample = np.random.choice(working_starts, size=n_each, replace=False)
    failing_sample = np.random.choice(failing_starts, size=n_each, replace=False)

    attn_working = get_attn_for_starts(working_sample)  # (n_each, 893, 893)
    attn_failing = get_attn_for_starts(failing_sample)

    # How much attention does target_node RECEIVE from ALL other nodes?
    recv_working = attn_working[:, :, target_node].sum(axis=1).mean()  # sum over query nodes
    recv_failing = attn_failing[:, :, target_node].sum(axis=1).mean()

    # Self-attention of target_node
    self_working = attn_working[:, target_node, target_node].mean()
    self_failing = attn_failing[:, target_node, target_node].mean()

    # How much do NEIGHBORS attend to target_node?
    # Find top-5 geographic neighbors (by attention when working)
    mean_attn_working = attn_working.mean(axis=0)  # (893, 893)
    top_attendees = np.argsort(mean_attn_working[:, target_node])[-10:]  # nodes that attend most to target

    print(f"\n  Attention RECEIVED by node {target_node}:")
    print(f"    When WORKING: {recv_working:.4f}")
    print(f"    When FAILING: {recv_failing:.4f}")
    print(f"    Ratio (failing/working): {recv_failing/recv_working:.3f}")

    print(f"\n  Self-attention of node {target_node}:")
    print(f"    When WORKING: {self_working:.6f}")
    print(f"    When FAILING: {self_failing:.6f}")

    print(f"\n  Top attendees to node {target_node} (when working):")
    for n in top_attendees:
        a_w = attn_working[:, n, target_node].mean()
        a_f = attn_failing[:, n, target_node].mean()
        print(f"    Node {n:4d}: working={a_w:.6f}, failing={a_f:.6f}, ratio={a_f/a_w:.3f}")
else:
    print("  Not enough samples for working/failing comparison")

print("\n" + "="*70)
print("DONE")
