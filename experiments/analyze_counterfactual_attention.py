"""Counterfactual test: replace failing node's zero values with mean,
check if attention normalizes.

Hypothesis: z-score normalized zeros are distinctive → high attention.
If we replace zeros with mean (=0 in z-score space), attention should normalize.
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

# Monkey-patch
attn_scores_spatial = []
def patched_forward(self, query, key, value):
    batch_size = query.shape[0]
    query = self.FC_Q(query)
    key = self.FC_K(key)
    value = self.FC_V(value)
    query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
    key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
    value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)
    key = key.transpose(-1, -2)
    attn_score = (query @ key) / self.head_dim**0.5
    if self.mask:
        tgt_length, src_length = query.shape[-2], key.shape[-1]
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
scaler = ZScoreScaler(dataset_name='xtraffic/SAN_BERNARDINO', train_ratio=0.6, norm_each_channel=True, rescale=True)

# Check z-score value for zero
print("=== Z-score value check ===")
# scaler stores mean/std per node
mean_val = scaler.mean  # shape depends on implementation
std_val = scaler.std
if hasattr(mean_val, 'shape'):
    print(f"Scaler mean shape: {mean_val.shape}, std shape: {std_val.shape}")
    # For node 382
    if len(mean_val.shape) > 1:
        m382 = mean_val[382] if mean_val.shape[0] > 382 else mean_val[0, 382]
        s382 = std_val[382] if std_val.shape[0] > 382 else std_val[0, 382]
    else:
        m382 = mean_val.item() if mean_val.shape == () else mean_val[0]
        s382 = std_val.item() if std_val.shape == () else std_val[0]
    print(f"Node 382: mean={m382}, std={s382}")
    print(f"Zero → z-score = (0 - {m382}) / {s382} = {-m382/s382}")
else:
    print(f"Scaler mean: {mean_val}, std: {std_val}")

# Target nodes
flow = data_3mo[:, :, 0]
zero_rates = (flow == 0).mean(axis=0)
transitions = np.diff((flow > 0).astype(int), axis=0)
transition_counts = np.abs(transitions).sum(axis=0)

target_nodes = [382, 562, 616]  # top intermittent
test_start = int(26280 * 0.8)
INPUT_LEN = 12

# Find samples where target node has failing timesteps
num_samples = 200
np.random.seed(42)
sample_starts = np.random.randint(test_start, 26280 - INPUT_LEN - 12, size=num_samples)

def run_model_get_attn(h_tensor):
    """Run model, return per-timestep attention: (num_heads, B, 12, 893, 893)"""
    attn_scores_spatial.clear()
    with torch.no_grad():
        _ = model(h_tensor, None, batch_seen=0, epoch=0, train=False)
    return attn_scores_spatial[0].reshape(4, h_tensor.shape[0], 12, 893, 893).numpy()

print("\n=== COUNTERFACTUAL TEST ===")
print("For each target node, compare attention at failing timesteps:")
print("  (A) Original: zero values (distinctive z-score)")
print("  (B) Counterfactual: zeros replaced with per-node mean (z-score = 0)")
print()

batch_size = 16

for node in target_nodes:
    print(f"--- Node {node} (transitions={transition_counts[node]:.0f}, zero_rate={zero_rates[node]:.3f}) ---")

    attn_original_failing = []
    attn_replaced_failing = []
    attn_original_working = []

    for batch_idx in range(0, num_samples, batch_size):
        batch_starts = sample_starts[batch_idx:batch_idx + batch_size]
        B = len(batch_starts)

        # Build original input
        history = np.zeros((B, INPUT_LEN, 893, 3), dtype=np.float32)
        raw_flow = np.zeros((B, INPUT_LEN, 893), dtype=np.float32)
        for i, s in enumerate(batch_starts):
            raw = data_3mo[s:s+INPUT_LEN, :, :]
            history[i, :, :, 0] = raw[:, :, 0]
            history[i, :, :, 1] = raw[:, :, 3]
            history[i, :, :, 2] = raw[:, :, 4]
            raw_flow[i] = raw[:, :, 0]

        # (A) Original
        h_orig = torch.tensor(history, dtype=torch.float32).to(device)
        h_orig[..., 0:1] = scaler.transform(h_orig[..., 0:1])
        attn_orig = run_model_get_attn(h_orig)  # (4, B, 12, 893, 893)

        # (B) Replace zeros with mean for target node only
        history_replaced = history.copy()
        for i in range(B):
            for t in range(INPUT_LEN):
                if raw_flow[i, t, node] == 0:
                    # Replace with the raw mean of that node (so z-score becomes 0)
                    # We need the raw mean. scaler.transform does (x - mean) / std
                    # So to get z-score = 0, raw value should be mean
                    # We set raw to mean before transform
                    history_replaced[i, t, node, 0] = float(scaler.mean[0, node].item())

        h_replaced = torch.tensor(history_replaced, dtype=torch.float32).to(device)
        h_replaced[..., 0:1] = scaler.transform(h_replaced[..., 0:1])
        attn_repl = run_model_get_attn(h_replaced)

        # Collect per-timestep attention received by target node
        for b in range(B):
            for t in range(INPUT_LEN):
                is_zero = raw_flow[b, t, node] == 0
                recv_orig = attn_orig[:, b, t, :, node].mean(axis=0).sum()
                recv_repl = attn_repl[:, b, t, :, node].mean(axis=0).sum()
                if is_zero:
                    attn_original_failing.append(recv_orig)
                    attn_replaced_failing.append(recv_repl)
                else:
                    attn_original_working.append(recv_orig)

    orig_fail = np.mean(attn_original_failing) if attn_original_failing else 0
    repl_fail = np.mean(attn_replaced_failing) if attn_replaced_failing else 0
    orig_work = np.mean(attn_original_working) if attn_original_working else 0

    print(f"  Failing timesteps: {len(attn_original_failing)}")
    print(f"  Working timesteps: {len(attn_original_working)}")
    print(f"")
    print(f"  Attention received at FAILING timesteps:")
    print(f"    (A) Original (zero):      {orig_fail:.4f}")
    print(f"    (B) Replaced with mean:   {repl_fail:.4f}")
    print(f"    Working reference:         {orig_work:.4f}")
    print(f"")
    print(f"  Original failing / working:   {orig_fail/orig_work:.3f}x  (the problem)")
    print(f"  Replaced failing / working:   {repl_fail/orig_work:.3f}x  (after fix)")
    print(f"  Reduction: {orig_fail:.4f} → {repl_fail:.4f} ({(repl_fail-orig_fail)/orig_fail*100:+.1f}%)")
    print()

print("=== DONE ===")
