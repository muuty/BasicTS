"""Compare spatial attention patterns between baseline and zero-mean models.

For each model, check: when a node has zero flow at a specific timestep,
how much attention does it receive vs when it's working?
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

BASELINE_CKPT = 'checkpoints/ContextContrastive_baseline_3mo/xtraffic_SAN_BERNARDINO_30_12_12/040af4f5bcb37097bc5263d7f350174b/STAEformer_Baseline_best_val_MAE.pt'
ZEROMEAN_CKPT = 'checkpoints/STAEformer_ZeroMean/SAN_BERNARDINO_30_12_12/9ca69d61cefbf81c0fc1b4cde8f6d4aa/STAEformer_best_val_MAE.pt'

device = torch.device('cuda:1')

def load_model(ckpt_path):
    model = STAEformer(**MODEL_PARAM).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
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
    return model

# Monkey-patch setup
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

def patch_model(model):
    for layer in model.attn_layers_s:
        layer.attn.forward = lambda q, k, v, _self=layer.attn: patched_forward(_self, q, k, v)

# Load data
data = np.memmap('datasets/xtraffic/SAN_BERNARDINO/data.dat', dtype='float32', mode='r', shape=(105120, 893, 5))
data_3mo = np.array(data[:26280])

scaler = ZScoreScaler(dataset_name='SAN_BERNARDINO', train_ratio=0.6, norm_each_channel=False, rescale=True)
flow_mean = scaler.mean.clone()

flow = data_3mo[:, :, 0]
zero_rates = (flow == 0).mean(axis=0)
transitions = np.diff((flow > 0).astype(int), axis=0)
transition_counts = np.abs(transitions).sum(axis=0)
top_intermittent = np.argsort(transition_counts)[-5:]

test_start = int(26280 * 0.8)
INPUT_LEN = 12
num_samples = 300
np.random.seed(42)
sample_starts = np.random.randint(test_start, 26280 - INPUT_LEN - 12, size=num_samples)

target_nodes = list(top_intermittent[-5:])
print(f"Target nodes: {target_nodes}")
print(f"Transition counts: {transition_counts[target_nodes]}")
print(f"Zero rates: {zero_rates[target_nodes]}")

batch_size = 16

def analyze_model(model, model_name, apply_zero_mean_replacement=False):
    """Run timestep-level attention analysis for a model."""
    patch_model(model)

    node_timestep_attn = {n: {0: [], 1: []} for n in target_nodes}

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

        # Apply zero-to-mean replacement if needed (matching the Runner behavior)
        if apply_zero_mean_replacement:
            flow_ch = h_tensor[..., 0]
            zero_mask = (flow_ch == 0)
            if zero_mask.any():
                mean_val = flow_mean.to(device)
                h_tensor[..., 0] = torch.where(zero_mask, mean_val, flow_ch)

        # Z-score normalize flow channel
        h_tensor[..., 0:1] = scaler.transform(h_tensor[..., 0:1])

        attn_scores_spatial.clear()
        with torch.no_grad():
            _ = model(h_tensor, None, batch_seen=0, epoch=0, train=False)

        attn = attn_scores_spatial[0]
        num_heads = 4
        attn = attn.reshape(num_heads, B, 12, 893, 893).numpy()

        for node in target_nodes:
            for b in range(B):
                for t in range(INPUT_LEN):
                    is_zero = raw_flow[b, t, node] == 0
                    recv = attn[:, b, t, :, node].mean(axis=0).sum()
                    node_timestep_attn[node][0 if is_zero else 1].append(recv)

    print(f"\n{'='*60}")
    print(f"MODEL: {model_name}")
    print(f"{'='*60}")

    for node in target_nodes:
        failing = node_timestep_attn[node][0]
        working = node_timestep_attn[node][1]
        if len(failing) > 0 and len(working) > 0:
            f_mean = np.mean(failing)
            w_mean = np.mean(working)
            print(f"\nNode {node} (transitions={transition_counts[node]:.0f}, zero_rate={zero_rates[node]:.3f})")
            print(f"  Timesteps: {len(failing)} failing, {len(working)} working")
            print(f"  Avg attention received:")
            print(f"    When WORKING: {w_mean:.4f}")
            print(f"    When FAILING: {f_mean:.4f}")
            print(f"    Ratio (failing/working): {f_mean/w_mean:.4f} ({(f_mean/w_mean - 1)*100:+.1f}%)")

# Run analysis
print("\n>>> Loading BASELINE model...")
model_baseline = load_model(BASELINE_CKPT)
analyze_model(model_baseline, "BASELINE (no replacement)", apply_zero_mean_replacement=False)
del model_baseline
torch.cuda.empty_cache()

print("\n>>> Loading ZERO-MEAN model...")
model_zeromean = load_model(ZEROMEAN_CKPT)
analyze_model(model_zeromean, "ZERO-MEAN (zeros replaced with mean)", apply_zero_mean_replacement=True)
del model_zeromean
torch.cuda.empty_cache()

print("\n\n" + "="*60)
print("DONE")
