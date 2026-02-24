"""Timestep-level spatial attention analysis for intermittent nodes.

For each timestep within the 12-step window, check if the target node
has zero flow at THAT specific timestep, and compare attention received.
"""
import os, sys
sys.path.insert(0, '/data/pretrainingbasicts')

import torch
import numpy as np
from baselines.STAEformer.arch.staeformer_arch import STAEformer
from basicts.scaler import ZScoreScaler

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

model = STAEformer(**MODEL_PARAM).to(device)
ckpt = torch.load(CKPT_PATH, map_location=device)
state = ckpt['model_state_dict']
new_state = {}
for k, v in state.items():
    for prefix in ('encoder.', 'spatial.', 'decoder.'):
        if k.startswith(prefix):
            k = k[len(prefix):]
            break
    new_state[k] = v
model.load_state_dict(new_state)
model.eval()

# Monkey-patch to capture per-timestep, per-head attention
attn_scores_spatial = []

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
    attn_scores_spatial.append(attn_score.detach().cpu())
    out = attn_score @ value
    out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)
    out = self.out_proj(out)
    return out

for layer in model.attn_layers_s:
    layer.attn.forward = lambda q, k, v, _self=layer.attn: patched_forward(_self, q, k, v)

# Load data
data = np.memmap('datasets/xtraffic/SAN_BERNARDINO/data.dat', dtype='float32', mode='r', shape=(105120, 893, 5))
data_3mo = np.array(data[:26280])

flow = data_3mo[:, :, 0]
zero_rates = (flow == 0).mean(axis=0)
transitions = np.diff((flow > 0).astype(int), axis=0)
transition_counts = np.abs(transitions).sum(axis=0)
top_intermittent = np.argsort(transition_counts)[-10:]

# Categories
dead_nodes = set(np.where(zero_rates > 0.9)[0])
major_fail = set(np.where((zero_rates > 0.5) & (zero_rates <= 0.9))[0])
partial_fail = set(np.where((zero_rates > 0.05) & (zero_rates <= 0.5))[0])
functional = set(np.where(zero_rates <= 0.05)[0])

scaler = ZScoreScaler(dataset_name='xtraffic/SAN_BERNARDINO', train_ratio=0.6, norm_each_channel=True, rescale=True)

# Test set
test_start = int(26280 * 0.8)
INPUT_LEN = 12

# Collect per-timestep attention for many samples
num_samples = 300
np.random.seed(42)
sample_starts = np.random.randint(test_start, 26280 - INPUT_LEN - 12, size=num_samples)

# For each target node, collect: (timestep_is_zero, attn_received_at_that_timestep)
target_nodes = list(top_intermittent[-5:])  # top 5 most intermittent
print(f"Target nodes: {target_nodes}")
print(f"Transition counts: {transition_counts[target_nodes]}")
print(f"Zero rates: {zero_rates[target_nodes]}")

# Store per-timestep results: {node: {0: [attn_values], 1: [attn_values]}}
# key 0 = failing (zero), key 1 = working (non-zero)
node_timestep_attn = {n: {0: [], 1: []} for n in target_nodes}
# Also per-head
node_timestep_attn_perhead = {n: {h: {0: [], 1: []} for h in range(4)} for n in target_nodes}

batch_size = 16
for batch_idx in range(0, num_samples, batch_size):
    batch_starts = sample_starts[batch_idx:batch_idx + batch_size]
    B = len(batch_starts)

    history = np.zeros((B, INPUT_LEN, 893, 3), dtype=np.float32)
    raw_flow = np.zeros((B, INPUT_LEN, 893), dtype=np.float32)
    for i, s in enumerate(batch_starts):
        raw = data_3mo[s:s+INPUT_LEN, :, :]
        history[i, :, :, 0] = raw[:, :, 0]
        history[i, :, :, 1] = raw[:, :, 3]
        history[i, :, :, 2] = raw[:, :, 4]
        raw_flow[i] = raw[:, :, 0]

    h_tensor = torch.tensor(history, dtype=torch.float32).to(device)
    h_tensor[..., 0:1] = scaler.transform(h_tensor[..., 0:1])

    attn_scores_spatial.clear()
    with torch.no_grad():
        _ = model(h_tensor, None, batch_seen=0, epoch=0, train=False)

    # attn shape: (num_heads*B, 12, 893, 893)
    attn = attn_scores_spatial[0]
    num_heads = 4
    # Reshape to (num_heads, B, 12, 893, 893)
    attn = attn.reshape(num_heads, B, 12, 893, 893).numpy()

    for node in target_nodes:
        for b in range(B):
            for t in range(INPUT_LEN):
                is_zero = raw_flow[b, t, node] == 0
                # Attention received by this node at this timestep (avg over heads)
                recv = attn[:, b, t, :, node].mean(axis=0).sum()  # sum over all query nodes
                node_timestep_attn[node][0 if is_zero else 1].append(recv)
                # Per-head
                for h in range(num_heads):
                    recv_h = attn[h, b, t, :, node].sum()
                    node_timestep_attn_perhead[node][h][0 if is_zero else 1].append(recv_h)

print("\n" + "="*70)
print("TIMESTEP-LEVEL ATTENTION ANALYSIS")
print("="*70)

for node in target_nodes:
    failing = node_timestep_attn[node][0]
    working = node_timestep_attn[node][1]
    print(f"\nNode {node} (transitions={transition_counts[node]:.0f}, zero_rate={zero_rates[node]:.3f})")
    print(f"  Timesteps: {len(failing)} failing, {len(working)} working")
    if len(failing) > 0 and len(working) > 0:
        f_mean = np.mean(failing)
        w_mean = np.mean(working)
        print(f"  Avg attention received:")
        print(f"    When WORKING: {w_mean:.4f}")
        print(f"    When FAILING: {f_mean:.4f}")
        print(f"    Ratio (failing/working): {f_mean/w_mean:.4f}")
        print(f"    Difference: {(f_mean - w_mean):.4f} ({(f_mean - w_mean)/w_mean*100:+.2f}%)")

        # Per-head analysis
        print(f"  Per-head breakdown:")
        for h in range(4):
            f_h = np.mean(node_timestep_attn_perhead[node][h][0])
            w_h = np.mean(node_timestep_attn_perhead[node][h][1])
            print(f"    Head {h}: working={w_h:.4f}, failing={f_h:.4f}, ratio={f_h/w_h:.4f}")

# Also check: do FUNCTIONAL neighbors change their attention to the failing node?
print("\n" + "="*70)
print("NEIGHBOR-LEVEL ANALYSIS (functional neighbors → intermittent node)")
print("="*70)

# Re-run with a focused batch to get full matrices
node = target_nodes[-1]  # most intermittent
print(f"\nFocusing on node {node}")

# Collect per-timestep, per-neighbor attention
func_nodes = sorted(list(functional))[:100]  # first 100 functional nodes
neighbor_attn_working = []  # (timestep_count, 100)
neighbor_attn_failing = []

for batch_idx in range(0, num_samples, batch_size):
    batch_starts = sample_starts[batch_idx:batch_idx + batch_size]
    B = len(batch_starts)

    history = np.zeros((B, INPUT_LEN, 893, 3), dtype=np.float32)
    raw_flow = np.zeros((B, INPUT_LEN, 893), dtype=np.float32)
    for i, s in enumerate(batch_starts):
        raw = data_3mo[s:s+INPUT_LEN, :, :]
        history[i, :, :, 0] = raw[:, :, 0]
        history[i, :, :, 1] = raw[:, :, 3]
        history[i, :, :, 2] = raw[:, :, 4]
        raw_flow[i] = raw[:, :, 0]

    h_tensor = torch.tensor(history, dtype=torch.float32).to(device)
    h_tensor[..., 0:1] = scaler.transform(h_tensor[..., 0:1])

    attn_scores_spatial.clear()
    with torch.no_grad():
        _ = model(h_tensor, None, batch_seen=0, epoch=0, train=False)

    attn = attn_scores_spatial[0].reshape(4, B, 12, 893, 893).numpy()
    attn_head_avg = attn.mean(axis=0)  # (B, 12, 893, 893)

    for b in range(B):
        for t in range(INPUT_LEN):
            is_zero = raw_flow[b, t, node] == 0
            # Attention from each functional neighbor to target node
            neighbor_attn = attn_head_avg[b, t, func_nodes, node]  # (100,)
            if is_zero:
                neighbor_attn_failing.append(neighbor_attn)
            else:
                neighbor_attn_working.append(neighbor_attn)

neighbor_attn_working = np.array(neighbor_attn_working)  # (n_working, 100)
neighbor_attn_failing = np.array(neighbor_attn_failing)  # (n_failing, 100)

print(f"  Working timesteps: {len(neighbor_attn_working)}, Failing: {len(neighbor_attn_failing)}")
w_avg = neighbor_attn_working.mean(axis=0)  # (100,) per-neighbor avg when working
f_avg = neighbor_attn_failing.mean(axis=0)  # (100,) per-neighbor avg when failing
ratios = f_avg / np.clip(w_avg, 1e-10, None)

print(f"\n  Across 100 functional neighbors:")
print(f"    Mean attention to node {node} when WORKING: {w_avg.mean():.6f}")
print(f"    Mean attention to node {node} when FAILING: {f_avg.mean():.6f}")
print(f"    Mean ratio (failing/working): {ratios.mean():.4f}")
print(f"    Min ratio: {ratios.min():.4f}, Max ratio: {ratios.max():.4f}")
print(f"    Neighbors with ratio < 0.9 (>10% reduction): {(ratios < 0.9).sum()}/100")
print(f"    Neighbors with ratio < 0.8 (>20% reduction): {(ratios < 0.8).sum()}/100")
print(f"    Neighbors with ratio > 1.1 (>10% increase): {(ratios > 1.1).sum()}/100")

print("\n" + "="*70)
print("DONE")
