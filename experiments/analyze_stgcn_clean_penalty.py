"""
Analyze why STGCN + v2 combo has +13% clean MAE penalty.

Hypotheses:
1. Encoder correction is non-zero on clean data (identity fallback imperfect)
2. Flow channel correction distortion is large
3. Distortion is concentrated on specific nodes
4. STGCN (1ch) absorbs all distortion vs STAEformer (3ch) dilutes it
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

import numpy as np
import torch
from basicts.utils import load_adj

# ============================================================
# 1. Load encoder and measure correction on clean data
# ============================================================
print("=" * 60)
print("1. ENCODER CORRECTION ON CLEAN DATA")
print("=" * 60)

from baselines.ContextContrastive.arch.denoising_encoder import DenoisingEncoder

# Load v2 encoder
encoder = DenoisingEncoder(
    input_dim=5, d_model=5, hidden_dim=64,
    temporal_layers=4, spatial_layers=1,
    k_neighbors=10, dropout=0.1,
    adj_path='datasets/SAN_BERNARDINO/adj_mx.pkl',
    physical_channels=[0, 1, 2],
    residual_connection=True,
)

# Load pretrained weights
import glob
ckpt_pattern = 'checkpoints/DenoisingPretrainV2/SAN_BERNARDINO_30_12_12/0ac3bff3*/DenoisingPretrainV2_best_val_MAE.pt'
ckpt_files = glob.glob(ckpt_pattern)
assert len(ckpt_files) == 1, f"Expected 1 checkpoint, found {len(ckpt_files)}: {ckpt_files}"
ckpt = torch.load(ckpt_files[0], map_location='cpu')

# Extract encoder weights
encoder_state = {}
for k, v in ckpt.items():
    if k.startswith('encoder.'):
        encoder_state[k[len('encoder.'):]] = v
encoder.load_state_dict(encoder_state, strict=False)
encoder.eval()
print(f"Loaded encoder from {ckpt_files[0]}")

# Load test data (clean)
data = np.memmap('datasets/SAN_BERNARDINO/data.dat', dtype='float32', mode='r',
                 shape=(105120, 893, 5))
# Use a sample of test data (last portion)
test_start = int(26280 * 0.8)  # 80% of data_range for test
test_data = np.array(data[test_start:test_start+288])  # 1 day of test data

# Create batched input: [B, T, N, 5]
# Use sliding windows of length 12
windows = []
for i in range(0, 288 - 12, 12):
    windows.append(test_data[i:i+12])
batch = np.stack(windows)  # [B, 12, 893, 5]
batch_tensor = torch.FloatTensor(batch)

print(f"Input shape: {batch_tensor.shape}")

# Run encoder on clean data
with torch.no_grad():
    encoded = encoder.encode(batch_tensor)

# Measure correction per channel
correction = encoded - batch_tensor
physical_channels = [0, 1, 2]
ch_names = ['flow', 'occupancy', 'speed']

print("\n--- Correction magnitude on CLEAN data ---")
for i, (ch, name) in enumerate(zip(physical_channels, ch_names)):
    corr_ch = correction[..., ch]
    input_ch = batch_tensor[..., ch]

    abs_corr = corr_ch.abs()
    rel_corr = (abs_corr / (input_ch.abs() + 1e-8))

    print(f"\n  {name} (ch{ch}):")
    print(f"    Mean abs correction:  {abs_corr.mean():.4f}")
    print(f"    Std abs correction:   {abs_corr.std():.4f}")
    print(f"    Max abs correction:   {abs_corr.max():.4f}")
    print(f"    Mean rel correction:  {rel_corr.mean():.4f} ({rel_corr.mean()*100:.2f}%)")
    print(f"    Mean input magnitude: {input_ch.abs().mean():.4f}")

# Per-node flow correction
flow_corr_per_node = correction[..., 0].abs().mean(dim=(0, 1))  # [N]
print(f"\n--- Per-node flow correction distribution ---")
print(f"  Mean: {flow_corr_per_node.mean():.4f}")
print(f"  Median: {flow_corr_per_node.median():.4f}")
print(f"  Std: {flow_corr_per_node.std():.4f}")
print(f"  Max: {flow_corr_per_node.max():.4f} (node {flow_corr_per_node.argmax().item()})")
print(f"  Min: {flow_corr_per_node.min():.4f} (node {flow_corr_per_node.argmin().item()})")

# Nodes with largest correction
top_k = 20
top_nodes = flow_corr_per_node.argsort(descending=True)[:top_k]
print(f"\n  Top-{top_k} nodes by flow correction magnitude:")
for idx in top_nodes:
    node_flow_mean = batch_tensor[..., idx.item(), 0].abs().mean()
    print(f"    Node {idx.item():3d}: correction={flow_corr_per_node[idx]:.4f}, "
          f"mean_flow={node_flow_mean:.2f}, "
          f"rel_correction={flow_corr_per_node[idx]/(node_flow_mean+1e-8)*100:.1f}%")

# ============================================================
# 2. Load per-node MAE from test results
# ============================================================
print("\n" + "=" * 60)
print("2. PER-NODE MAE COMPARISON")
print("=" * 60)

# Load node categories
dead_idx = np.load('datasets/SAN_BERNARDINO/dead_indices.npy')
major_idx = np.load('datasets/SAN_BERNARDINO/major_fail_indices.npy')
exclude = set(dead_idx.tolist()) | set(major_idx.tolist())
eval_nodes = [i for i in range(893) if i not in exclude]
print(f"Eval nodes: {len(eval_nodes)}")

# Load test predictions for STGCN baseline and v2 combo
stgcn_base_pattern = 'checkpoints/STGCN/SAN_BERNARDINO_30_12_12/*/test_results/*'
stgcn_v2_pattern = 'checkpoints/STGCN_denoising_v2_noisy/SAN_BERNARDINO_30_12_12/*/test_results/*'

# Try to find test_results directories
import json

def find_test_metrics(pattern):
    """Find test_metrics.json matching pattern."""
    base = pattern.rsplit('/test_results', 1)[0]
    matches = glob.glob(base)
    for m in matches:
        metrics_file = os.path.join(m, 'test_metrics.json')
        if os.path.exists(metrics_file):
            return metrics_file
    return None

# Find prediction files (memmap format)
def find_prediction_files(model_dir_pattern):
    """Find prediction and target memmap files."""
    dirs = glob.glob(model_dir_pattern)
    for d in dirs:
        test_dir = os.path.join(d, 'test_results')
        if os.path.exists(test_dir):
            pred_file = os.path.join(test_dir, 'prediction.dat')
            target_file = os.path.join(test_dir, 'target.dat')
            if os.path.exists(pred_file) and os.path.exists(target_file):
                return pred_file, target_file
    return None, None

stgcn_base_pred, stgcn_base_target = find_prediction_files('checkpoints/STGCN/SAN_BERNARDINO_30_12_12/*')
stgcn_v2_pred, stgcn_v2_target = find_prediction_files('checkpoints/STGCN_denoising_v2_noisy/SAN_BERNARDINO_30_12_12/*')
stae_base_pred, stae_base_target = find_prediction_files('checkpoints/STAEformer_5ch/SAN_BERNARDINO_30_12_12/*')
stae_v2_pred, stae_v2_target = find_prediction_files('checkpoints/STAEformer_5ch_denoising_v2_noisy/SAN_BERNARDINO_30_12_12/*')

n_test_samples = 5233  # known from prior work

def load_memmap(path, n_samples=5233, n_horizon=12, n_nodes=893):
    return np.memmap(path, dtype='float32', mode='r', shape=(n_samples, n_horizon, n_nodes))

def compute_per_node_mae(pred_path, target_path, eval_nodes):
    pred = load_memmap(pred_path)
    target = load_memmap(target_path)

    # Per-node MAE (masked, null_val=0)
    per_node_mae = []
    for n in eval_nodes:
        p = pred[:, :, n].flatten()
        t = target[:, :, n].flatten()
        mask = t != 0
        if mask.sum() > 0:
            mae = np.abs(p[mask] - t[mask]).mean()
        else:
            mae = 0.0
        per_node_mae.append(mae)
    return np.array(per_node_mae)

results = {}
for name, pred_path, target_path in [
    ('STGCN_baseline', stgcn_base_pred, stgcn_base_target),
    ('STGCN_v2_combo', stgcn_v2_pred, stgcn_v2_target),
    ('STAEformer_baseline', stae_base_pred, stae_base_target),
    ('STAEformer_v2_combo', stae_v2_pred, stae_v2_target),
]:
    if pred_path and target_path:
        mae = compute_per_node_mae(pred_path, target_path, eval_nodes)
        results[name] = mae
        print(f"  {name}: mean MAE = {mae.mean():.4f}, median = {np.median(mae):.4f}")
    else:
        print(f"  {name}: prediction files not found")

# ============================================================
# 3. Correlate correction magnitude with MAE increase
# ============================================================
if 'STGCN_baseline' in results and 'STGCN_v2_combo' in results:
    print("\n" + "=" * 60)
    print("3. CORRELATION: ENCODER CORRECTION vs MAE PENALTY")
    print("=" * 60)

    stgcn_delta = results['STGCN_v2_combo'] - results['STGCN_baseline']  # per-node MAE diff
    flow_corr_eval = flow_corr_per_node[eval_nodes].numpy()

    # Correlation
    corr = np.corrcoef(flow_corr_eval, stgcn_delta)[0, 1]
    print(f"  Correlation(flow_correction, MAE_increase): r = {corr:.4f}")

    # Nodes where v2 combo is WORSE
    worse_mask = stgcn_delta > 0
    better_mask = stgcn_delta < 0
    print(f"\n  Nodes where v2 combo is WORSE: {worse_mask.sum()}/{len(eval_nodes)} ({worse_mask.mean()*100:.1f}%)")
    print(f"  Nodes where v2 combo is BETTER: {better_mask.sum()}/{len(eval_nodes)} ({better_mask.mean()*100:.1f}%)")

    print(f"\n  Mean MAE increase (worse nodes): +{stgcn_delta[worse_mask].mean():.4f}")
    print(f"  Mean MAE decrease (better nodes): {stgcn_delta[better_mask].mean():.4f}")

    # Correction on worse vs better nodes
    print(f"\n  Mean flow correction (worse nodes): {flow_corr_eval[worse_mask].mean():.4f}")
    print(f"  Mean flow correction (better nodes): {flow_corr_eval[better_mask].mean():.4f}")

    # Breakdown by node category
    partial_idx = set()
    functional_idx = set()
    for i, node in enumerate(eval_nodes):
        flow_data = data[:26280, node, 0]
        zero_rate = (flow_data == 0).mean()
        if zero_rate >= 0.05:
            partial_idx.add(i)
        else:
            functional_idx.add(i)

    partial_list = sorted(partial_idx)
    functional_list = sorted(functional_idx)

    print(f"\n  --- By sensor category ---")
    for cat_name, cat_idx in [('Partial fail', partial_list), ('Functional', functional_list)]:
        if len(cat_idx) > 0:
            cat_delta = stgcn_delta[cat_idx]
            cat_corr = flow_corr_eval[cat_idx]
            cat_worse = (cat_delta > 0).sum()
            print(f"\n  {cat_name} ({len(cat_idx)} nodes):")
            print(f"    Mean MAE delta: {cat_delta.mean():+.4f}")
            print(f"    Worse: {cat_worse}/{len(cat_idx)} ({cat_worse/len(cat_idx)*100:.1f}%)")
            print(f"    Mean flow correction: {cat_corr.mean():.4f}")

if 'STAEformer_baseline' in results and 'STAEformer_v2_combo' in results:
    print("\n" + "=" * 60)
    print("4. STAEformer COMPARISON (for reference)")
    print("=" * 60)

    stae_delta = results['STAEformer_v2_combo'] - results['STAEformer_baseline']
    flow_corr_eval = flow_corr_per_node[eval_nodes].numpy()

    worse_mask = stae_delta > 0
    better_mask = stae_delta < 0
    print(f"  Nodes where v2 combo is WORSE: {worse_mask.sum()}/{len(eval_nodes)} ({worse_mask.mean()*100:.1f}%)")
    print(f"  Nodes where v2 combo is BETTER: {better_mask.sum()}/{len(eval_nodes)} ({better_mask.mean()*100:.1f}%)")
    print(f"  Mean MAE increase (worse): +{stae_delta[worse_mask].mean():.4f}")
    print(f"  Mean MAE decrease (better): {stae_delta[better_mask].mean():.4f}")

# ============================================================
# 5. Encoder output analysis: how much does flow change?
# ============================================================
print("\n" + "=" * 60)
print("5. FLOW CHANNEL DISTORTION DETAIL")
print("=" * 60)

# Compare encoded vs original flow on eval nodes
orig_flow = batch_tensor[..., eval_nodes, 0]  # [B, T, 745]
encoded_flow = encoded[..., eval_nodes, 0]

# Relative change
rel_change = (encoded_flow - orig_flow) / (orig_flow.abs() + 1e-8)
print(f"  Flow relative change (eval nodes):")
print(f"    Mean: {rel_change.mean():.6f} ({rel_change.mean()*100:.4f}%)")
print(f"    Std:  {rel_change.std():.4f}")
print(f"    Abs mean: {rel_change.abs().mean():.4f} ({rel_change.abs().mean()*100:.2f}%)")

# Is the correction biased (systematic shift)?
mean_correction_flow = correction[..., eval_nodes, 0].mean()
print(f"\n  Mean signed correction (flow, eval): {mean_correction_flow:.6f}")
print(f"  → {'Systematic positive shift' if mean_correction_flow > 0 else 'Systematic negative shift'}")

# Per-node signed correction
signed_corr = correction[..., 0].mean(dim=(0, 1))  # [893]
print(f"\n  Per-node signed correction distribution:")
print(f"    Mean: {signed_corr[eval_nodes].mean():.6f}")
print(f"    Std:  {signed_corr[eval_nodes].std():.4f}")
print(f"    Positive (encoder adds): {(signed_corr[eval_nodes] > 0).sum()}/{len(eval_nodes)}")
print(f"    Negative (encoder removes): {(signed_corr[eval_nodes] < 0).sum()}/{len(eval_nodes)}")

print("\nDone!")
