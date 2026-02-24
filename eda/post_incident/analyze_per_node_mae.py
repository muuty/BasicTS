"""Per-node MAE distribution analysis for robust_prediction research.

Analyzes:
1. Per-node MAE distribution (histogram, percentiles, long-tail)
2. Worst node characteristics (traffic volume, variance, zero_rate)
3. Correlation between node difficulty and node features
4. Loss contribution by node difficulty groups
"""

import numpy as np
import json
import os

# Paths
BASELINE_CKPT = "checkpoints/STAEformer_5ch_unmasked/SAN_BERNARDINO_30_12_12/9827aaaa1c185035775032f481e3f339"
DATA_PATH = "datasets/xtraffic/SAN_BERNARDINO/data.dat"
OUTPUT_DIR = "eda/robust_prediction"

# Load raw data for node characteristics
print("=" * 70)
print("Loading data...")
data = np.memmap(DATA_PATH, dtype=np.float32, mode='r').reshape(105120, 893, 5)
# Use training portion only (first 60% of 26280 = 15768)
train_data = data[:15768]
test_data_range = data[:26280]  # full 3 months

# Load predictions and targets from test results
pred_path = os.path.join(BASELINE_CKPT, "test_results", "predictions.npy")
target_path = os.path.join(BASELINE_CKPT, "test_results", "targets.npy")

num_samples = 5233
num_horizons = 12
num_nodes = 893

# Raw memmap files (no npy header, despite .npy extension)
pred = np.memmap(pred_path, dtype=np.float32, mode='r').reshape(num_samples, num_horizons, num_nodes)
target = np.memmap(target_path, dtype=np.float32, mode='r').reshape(num_samples, num_horizons, num_nodes)

print(f"Predictions shape: {pred.shape}")
print(f"Targets shape: {target.shape}")

# ============================================================
# 1. Per-node MAE distribution
# ============================================================
print("\n" + "=" * 70)
print("1. PER-NODE MAE DISTRIBUTION")
print("=" * 70)

# Compute per-node MAE (unmasked - as used in training)
per_node_mae = np.mean(np.abs(pred - target), axis=(0, 1))  # (893,)

# Also compute masked MAE (exclude zero targets)
per_node_masked_mae = np.zeros(num_nodes)
for n in range(num_nodes):
    node_pred = pred[:, :, n].flatten()
    node_target = target[:, :, n].flatten()
    mask = node_target != 0
    if mask.sum() > 0:
        per_node_masked_mae[n] = np.mean(np.abs(node_pred[mask] - node_target[mask]))
    else:
        per_node_masked_mae[n] = 0.0

print(f"\nUnmasked MAE statistics:")
print(f"  Mean:   {per_node_mae.mean():.3f}")
print(f"  Median: {np.median(per_node_mae):.3f}")
print(f"  Std:    {per_node_mae.std():.3f}")
print(f"  Min:    {per_node_mae.min():.3f} (node {per_node_mae.argmin()})")
print(f"  Max:    {per_node_mae.max():.3f} (node {per_node_mae.argmax()})")

percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
print(f"\nPercentiles:")
for p in percentiles:
    val = np.percentile(per_node_mae, p)
    print(f"  P{p:2d}: {val:.3f}")

print(f"\nMasked MAE statistics:")
print(f"  Mean:   {per_node_masked_mae.mean():.3f}")
print(f"  Median: {np.median(per_node_masked_mae):.3f}")
print(f"  Std:    {per_node_masked_mae.std():.3f}")
print(f"  Min:    {per_node_masked_mae.min():.3f} (node {per_node_masked_mae.argmin()})")
print(f"  Max:    {per_node_masked_mae.max():.3f} (node {per_node_masked_mae.argmax()})")

# Distribution shape
print(f"\nDistribution shape:")
print(f"  Skewness: {((per_node_mae - per_node_mae.mean())**3).mean() / per_node_mae.std()**3:.3f}")
print(f"  Kurtosis: {((per_node_mae - per_node_mae.mean())**4).mean() / per_node_mae.std()**4:.3f}")
q75 = np.percentile(per_node_mae, 75)
q25 = np.percentile(per_node_mae, 25)
iqr = q75 - q25
print(f"  IQR: {iqr:.3f}")
print(f"  Outlier threshold (Q75 + 1.5*IQR): {q75 + 1.5 * iqr:.3f}")
outlier_count = (per_node_mae > q75 + 1.5 * iqr).sum()
print(f"  Outlier nodes: {outlier_count}")

# ============================================================
# 2. Node characteristics
# ============================================================
print("\n" + "=" * 70)
print("2. NODE CHARACTERISTICS")
print("=" * 70)

# Compute per-node features from training data
flow = train_data[:, :, 0]  # (15768, 893)
occ = train_data[:, :, 1]
speed = train_data[:, :, 2]

per_node_mean_flow = flow.mean(axis=0)         # (893,)
per_node_std_flow = flow.std(axis=0)            # (893,)
per_node_max_flow = flow.max(axis=0)            # (893,)
per_node_zero_rate = (flow == 0).mean(axis=0)   # (893,)
per_node_mean_occ = occ.mean(axis=0)
per_node_mean_speed = speed.mean(axis=0)

# 3-channel zero rate (stricter)
three_ch_zero = ((flow == 0) & (occ == 0) & (speed == 0)).mean(axis=0)

# Coefficient of variation (for non-zero flow)
per_node_cv = np.zeros(num_nodes)
for n in range(num_nodes):
    nonzero_flow = flow[:, n][flow[:, n] > 0]
    if len(nonzero_flow) > 10:
        per_node_cv[n] = nonzero_flow.std() / nonzero_flow.mean()
    else:
        per_node_cv[n] = 0.0

# ============================================================
# 3. Worst nodes analysis
# ============================================================
print("\n" + "=" * 70)
print("3. WORST NODES ANALYSIS")
print("=" * 70)

# Sort by unmasked MAE
sorted_indices = np.argsort(per_node_mae)[::-1]  # descending

print(f"\nTop 20 worst nodes (by unmasked MAE):")
print(f"{'Rank':>4} {'Node':>5} {'MAE':>8} {'mMAE':>8} {'MeanFlow':>9} {'StdFlow':>9} {'MaxFlow':>8} {'ZeroRate':>9} {'CV':>6}")
print("-" * 80)
for rank, idx in enumerate(sorted_indices[:20]):
    print(f"{rank+1:4d} {idx:5d} {per_node_mae[idx]:8.2f} {per_node_masked_mae[idx]:8.2f} "
          f"{per_node_mean_flow[idx]:9.1f} {per_node_std_flow[idx]:9.1f} {per_node_max_flow[idx]:8.0f} "
          f"{per_node_zero_rate[idx]:9.3f} {per_node_cv[idx]:6.2f}")

print(f"\nTop 20 best nodes (by unmasked MAE):")
print(f"{'Rank':>4} {'Node':>5} {'MAE':>8} {'mMAE':>8} {'MeanFlow':>9} {'StdFlow':>9} {'MaxFlow':>8} {'ZeroRate':>9} {'CV':>6}")
print("-" * 80)
for rank, idx in enumerate(sorted_indices[-20:][::-1]):
    print(f"{rank+1:4d} {idx:5d} {per_node_mae[idx]:8.2f} {per_node_masked_mae[idx]:8.2f} "
          f"{per_node_mean_flow[idx]:9.1f} {per_node_std_flow[idx]:9.1f} {per_node_max_flow[idx]:8.0f} "
          f"{per_node_zero_rate[idx]:9.3f} {per_node_cv[idx]:6.2f}")

# ============================================================
# 4. Correlation analysis
# ============================================================
print("\n" + "=" * 70)
print("4. CORRELATION: NODE MAE vs CHARACTERISTICS")
print("=" * 70)

from numpy import corrcoef

# For all nodes
features = {
    'mean_flow': per_node_mean_flow,
    'std_flow': per_node_std_flow,
    'max_flow': per_node_max_flow,
    'zero_rate': per_node_zero_rate,
    'cv': per_node_cv,
    'mean_occ': per_node_mean_occ,
    'mean_speed': per_node_mean_speed,
    '3ch_zero_rate': three_ch_zero,
}

print(f"\nCorrelation with UNMASKED per-node MAE (all 893 nodes):")
for name, feat in features.items():
    r = corrcoef(per_node_mae, feat)[0, 1]
    print(f"  r({name:15s}) = {r:+.4f}")

# For functional nodes only (zero_rate < 5%)
func_mask = per_node_zero_rate < 0.05
func_indices = np.where(func_mask)[0]
print(f"\nCorrelation with UNMASKED per-node MAE (functional only, N={func_mask.sum()}):")
for name, feat in features.items():
    r = corrcoef(per_node_mae[func_mask], feat[func_mask])[0, 1]
    print(f"  r({name:15s}) = {r:+.4f}")

print(f"\nCorrelation with MASKED per-node MAE (functional only, N={func_mask.sum()}):")
for name, feat in features.items():
    r = corrcoef(per_node_masked_mae[func_mask], feat[func_mask])[0, 1]
    print(f"  r({name:15s}) = {r:+.4f}")

# ============================================================
# 5. Difficulty groups and loss contribution
# ============================================================
print("\n" + "=" * 70)
print("5. DIFFICULTY GROUPS & LOSS CONTRIBUTION")
print("=" * 70)

# Group by MAE quintiles (using unmasked MAE)
quintile_labels = ['Q1 (easiest)', 'Q2', 'Q3', 'Q4', 'Q5 (hardest)']
quintile_boundaries = np.percentile(per_node_mae, [0, 20, 40, 60, 80, 100])

print(f"\nQuintile boundaries (unmasked MAE):")
for i in range(5):
    print(f"  {quintile_labels[i]}: [{quintile_boundaries[i]:.2f}, {quintile_boundaries[i+1]:.2f}]")

print(f"\n{'Group':<15} {'N':>4} {'Mean MAE':>9} {'Mean mMAE':>10} {'Mean Flow':>10} {'Mean ZR':>8} {'Loss Share':>11} {'Fair Share':>10}")
print("-" * 85)

total_loss = per_node_mae.sum()
for i in range(5):
    mask = (per_node_mae >= quintile_boundaries[i]) & (per_node_mae <= quintile_boundaries[i+1])
    if i < 4:
        mask = (per_node_mae >= quintile_boundaries[i]) & (per_node_mae < quintile_boundaries[i+1])
    n = mask.sum()
    group_loss = per_node_mae[mask].sum()
    loss_share = group_loss / total_loss * 100
    fair_share = n / num_nodes * 100
    print(f"{quintile_labels[i]:<15} {n:4d} {per_node_mae[mask].mean():9.2f} {per_node_masked_mae[mask].mean():10.2f} "
          f"{per_node_mean_flow[mask].mean():10.1f} {per_node_zero_rate[mask].mean():8.3f} "
          f"{loss_share:10.1f}% {fair_share:9.1f}%")

# Top 10% vs bottom 90%
top10_mask = per_node_mae >= np.percentile(per_node_mae, 90)
bottom90_mask = ~top10_mask
print(f"\nTop 10% hardest nodes:")
print(f"  N = {top10_mask.sum()}")
print(f"  Mean MAE = {per_node_mae[top10_mask].mean():.2f}")
print(f"  Loss contribution = {per_node_mae[top10_mask].sum() / total_loss * 100:.1f}%")
print(f"  Mean flow = {per_node_mean_flow[top10_mask].mean():.1f}")
print(f"  Mean zero_rate = {per_node_zero_rate[top10_mask].mean():.3f}")

print(f"\nBottom 90% (easier nodes):")
print(f"  N = {bottom90_mask.sum()}")
print(f"  Mean MAE = {per_node_mae[bottom90_mask].mean():.2f}")
print(f"  Loss contribution = {per_node_mae[bottom90_mask].sum() / total_loss * 100:.1f}%")

# ============================================================
# 6. Functional-only difficulty analysis
# ============================================================
print("\n" + "=" * 70)
print("6. FUNCTIONAL NODES DIFFICULTY ANALYSIS")
print("=" * 70)

func_mae = per_node_mae[func_mask]
func_mmae = per_node_masked_mae[func_mask]
func_flow = per_node_mean_flow[func_mask]
func_std = per_node_std_flow[func_mask]
func_cv_vals = per_node_cv[func_mask]
func_node_ids = func_indices

print(f"Functional nodes: {func_mask.sum()}")
print(f"\nUnmasked MAE distribution (functional only):")
print(f"  Mean:   {func_mae.mean():.3f}")
print(f"  Median: {np.median(func_mae):.3f}")
print(f"  Std:    {func_mae.std():.3f}")
print(f"  P90:    {np.percentile(func_mae, 90):.3f}")
print(f"  P95:    {np.percentile(func_mae, 95):.3f}")
print(f"  P99:    {np.percentile(func_mae, 99):.3f}")
print(f"  Max:    {func_mae.max():.3f}")

# Within functional, worst vs best
func_sorted = np.argsort(func_mae)[::-1]
print(f"\nTop 10 hardest FUNCTIONAL nodes:")
print(f"{'Rank':>4} {'Node':>5} {'MAE':>8} {'mMAE':>8} {'MeanFlow':>9} {'StdFlow':>9} {'CV':>6}")
print("-" * 55)
for rank in range(10):
    fi = func_sorted[rank]
    ni = func_node_ids[fi]
    print(f"{rank+1:4d} {ni:5d} {func_mae[fi]:8.2f} {func_mmae[fi]:8.2f} "
          f"{per_node_mean_flow[ni]:9.1f} {per_node_std_flow[ni]:9.1f} {per_node_cv[ni]:6.2f}")

# Is it just high-traffic nodes?
print(f"\n\nFunctional nodes: MAE vs Mean Flow quartiles")
func_flow_q = np.percentile(func_flow, [0, 25, 50, 75, 100])
flow_labels = ['Low flow', 'Mid-low', 'Mid-high', 'High flow']
for i in range(4):
    if i < 3:
        fm = (func_flow >= func_flow_q[i]) & (func_flow < func_flow_q[i+1])
    else:
        fm = (func_flow >= func_flow_q[i]) & (func_flow <= func_flow_q[i+1])
    print(f"  {flow_labels[i]:10s} (flow {func_flow_q[i]:6.0f}-{func_flow_q[i+1]:6.0f}): "
          f"N={fm.sum():3d}, mean MAE={func_mae[fm].mean():.2f}, std MAE={func_mae[fm].std():.2f}")

# ============================================================
# 7. How much could we gain?
# ============================================================
print("\n" + "=" * 70)
print("7. POTENTIAL GAINS FROM ROBUST OPTIMIZATION")
print("=" * 70)

# If we could bring worst 10% nodes down to P90 level
p90_mae = np.percentile(per_node_mae, 90)
worst10_excess = np.maximum(per_node_mae - p90_mae, 0)
total_excess = worst10_excess.sum()
print(f"\nIf worst 10% nodes improved to P90 level ({p90_mae:.2f}):")
print(f"  Excess MAE to redistribute: {total_excess:.2f}")
print(f"  Overall MAE reduction: {total_excess / num_nodes:.3f}")
print(f"  New overall MAE: {per_node_mae.mean() - total_excess / num_nodes:.3f}")

# If worst 10% improved by 20%
improve_pct = 0.20
improved = per_node_mae.copy()
improved[top10_mask] *= (1 - improve_pct)
print(f"\nIf worst 10% nodes improved by {improve_pct*100:.0f}%:")
print(f"  New overall MAE: {improved.mean():.3f} (was {per_node_mae.mean():.3f})")
print(f"  Reduction: {(per_node_mae.mean() - improved.mean()):.3f} ({(per_node_mae.mean() - improved.mean()) / per_node_mae.mean() * 100:.2f}%)")

# Worst 1% (top 9 nodes)
top1_mask = per_node_mae >= np.percentile(per_node_mae, 99)
print(f"\nWorst 1% nodes ({top1_mask.sum()} nodes):")
print(f"  Mean MAE: {per_node_mae[top1_mask].mean():.2f}")
print(f"  These are {per_node_mae[top1_mask].mean() / per_node_mae.mean():.1f}x the overall mean")
print(f"  Loss contribution: {per_node_mae[top1_mask].sum() / total_loss * 100:.1f}%")

# Save per-node MAE for later use
np.save(os.path.join(OUTPUT_DIR, "per_node_mae_unmasked.npy"), per_node_mae)
np.save(os.path.join(OUTPUT_DIR, "per_node_mae_masked.npy"), per_node_masked_mae)
np.save(os.path.join(OUTPUT_DIR, "per_node_mean_flow.npy"), per_node_mean_flow)
np.save(os.path.join(OUTPUT_DIR, "per_node_zero_rate.npy"), per_node_zero_rate)

# Save summary as JSON
summary = {
    "per_node_mae": {
        "mean": float(per_node_mae.mean()),
        "median": float(np.median(per_node_mae)),
        "std": float(per_node_mae.std()),
        "min": float(per_node_mae.min()),
        "max": float(per_node_mae.max()),
        "p90": float(np.percentile(per_node_mae, 90)),
        "p95": float(np.percentile(per_node_mae, 95)),
        "p99": float(np.percentile(per_node_mae, 99)),
        "skewness": float(((per_node_mae - per_node_mae.mean())**3).mean() / per_node_mae.std()**3),
    },
    "worst_10pct": {
        "n_nodes": int(top10_mask.sum()),
        "mean_mae": float(per_node_mae[top10_mask].mean()),
        "loss_share_pct": float(per_node_mae[top10_mask].sum() / total_loss * 100),
        "mean_flow": float(per_node_mean_flow[top10_mask].mean()),
        "mean_zero_rate": float(per_node_zero_rate[top10_mask].mean()),
    },
    "worst_nodes": [int(i) for i in sorted_indices[:20]],
    "worst_nodes_mae": [float(per_node_mae[i]) for i in sorted_indices[:20]],
}

with open(os.path.join(OUTPUT_DIR, "analysis_summary.json"), 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nResults saved to {OUTPUT_DIR}/")
print("Done.")
