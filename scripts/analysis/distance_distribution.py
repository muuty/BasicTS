#!/usr/bin/env python3
"""
1.0 Distance Distribution Analysis + Normalization Effect

Computes pairwise distance distributions for all 8 distance types,
measures concentration (coefficient of variation), and compares with MAE rankings.
Also analyzes normalization effect by comparing combined vs normalized single-view.

Output: experiments/result/analysis/distance_distribution.png
        experiments/result/analysis/distance_stats.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from basicts.data.simple_tsf_dataset import TimeSeriesForecastingDataset
from coreset.distance import (
    extract_features, compute_distance_matrix,
    get_temporal_features, get_spatial_features, get_flat_features,
    _gpu_pairwise_l2, _gpu_pairwise_cosine_dist,
)

OUTPUT_DIR = 'experiments/result/analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Dataset setup ──
class DummyModelConfig:
    FORWARD_FEATURES = [0, 1, 2]  # STGCN uses flow, occupancy, speed
    TARGET_FEATURES = [0]

dataset = TimeSeriesForecastingDataset(
    dataset_name='SAN_BERNARDINO',
    train_val_test_ratio=[0.6, 0.2, 0.2],
    mode='train',
    input_len=12,
    output_len=12,
    data_range=(0, 24192),
)
print(f"Dataset size: {len(dataset)}")

model_config = DummyModelConfig()
inputs, targets = extract_features(dataset, model_config)
print(f"Inputs shape: {inputs.shape}, Targets shape: {targets.shape}")

# ── Compute features ──
feat_temporal = get_temporal_features(inputs, targets)
feat_spatial = get_spatial_features(inputs, targets)
feat_raw = get_flat_features(inputs, targets)

DIM_RAW = feat_raw.shape[1]
DIM_TEMPORAL = feat_temporal.shape[1]
DIM_SPATIAL = feat_spatial.shape[1]
print(f"Feature dims — temporal: {DIM_TEMPORAL}, spatial: {DIM_SPATIAL}, raw: {DIM_RAW}")

# ── Compute distance matrices ──
# We sample upper triangle for statistics (no need for full N×N histogram)
N = inputs.shape[0]
n_pairs_sample = min(500000, N * (N - 1) // 2)

def sample_upper_triangle(dist_mat, n_sample):
    """Sample from upper triangle of distance matrix."""
    N = dist_mat.shape[0]
    rng = np.random.default_rng(42)
    idx_i = rng.integers(0, N, n_sample)
    idx_j = rng.integers(0, N, n_sample)
    mask = idx_i < idx_j
    return dist_mat[idx_i[mask], idx_j[mask]]

print("\nComputing distance matrices (one at a time to save memory)...")
distance_types = ['euclidean', 'temporal', 'spatial', 'combined',
                  'cosine_raw', 'cosine_temporal', 'cosine_spatial', 'cosine_combined']

import gc, torch

dist_samples = {}
dist_stats = []

def get_dim_label(dt):
    if 'raw' in dt or dt == 'euclidean':
        return DIM_RAW
    elif 'temporal' in dt:
        return DIM_TEMPORAL
    elif 'spatial' in dt:
        return DIM_SPATIAL
    else:
        return f"T{DIM_TEMPORAL}+S{DIM_SPATIAL}"

def record_stats(name, samples, dim_label):
    cv = samples.std() / samples.mean() if samples.mean() > 0 else 0
    dist_stats.append({
        'distance_type': name, 'dim': dim_label,
        'mean': float(samples.mean()), 'std': float(samples.std()),
        'min': float(samples.min()), 'max': float(samples.max()),
        'cv': float(cv),
        'iqr': float(np.percentile(samples, 75) - np.percentile(samples, 25)),
    })

# Free raw features early — only needed for dim label
del feat_raw; gc.collect()

for dt in distance_types:
    print(f"  Computing {dt}...")
    dm = compute_distance_matrix(inputs, targets, dt)
    samples = sample_upper_triangle(dm, n_pairs_sample)
    dist_samples[dt] = samples
    record_stats(dt, samples, get_dim_label(dt))
    del dm; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ── Normalized single-view distributions ──
print("  Computing normalized temporal L2...")
dm_temporal = _gpu_pairwise_l2(feat_temporal)
t_max = dm_temporal.max()
samples_nt = sample_upper_triangle(dm_temporal / t_max if t_max > 0 else dm_temporal, n_pairs_sample)
dist_samples['norm_temporal_only'] = samples_nt
record_stats('norm_temporal_only', samples_nt, 'normalized')
del dm_temporal; gc.collect()

print("  Computing normalized spatial L2...")
dm_spatial = _gpu_pairwise_l2(feat_spatial)
s_max = dm_spatial.max()
samples_ns = sample_upper_triangle(dm_spatial / s_max if s_max > 0 else dm_spatial, n_pairs_sample)
dist_samples['norm_spatial_only'] = samples_ns
record_stats('norm_spatial_only', samples_ns, 'normalized')
del dm_spatial; gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ── MAE rankings from results ──
results = pd.read_csv('experiments/result/phase_a_distance_screening.csv')
stgcn = results[(results.model == 'STGCNChebGraphConv') & (results.MAE_mean > 0)]
mae_by_dist = stgcn.groupby('coreset_distance_type')['MAE_mean'].mean().sort_values()

# ── Save stats ──
stats_df = pd.DataFrame(dist_stats).sort_values('cv', ascending=False)
stats_df.to_csv(f'{OUTPUT_DIR}/distance_stats.csv', index=False)
print("\n=== Distance Statistics (sorted by CV, higher = better discrimination) ===")
print(stats_df.to_string(index=False, float_format='{:.4f}'.format))

# ── Rank correlation: CV vs MAE ──
cv_by_dist = stats_df[stats_df.distance_type.isin(distance_types)].set_index('distance_type')['cv']
common = cv_by_dist.index.intersection(mae_by_dist.index)
if len(common) >= 4:
    r, p = stats.spearmanr(-cv_by_dist[common], mae_by_dist[common])
    print(f"\nSpearman correlation (CV rank vs MAE rank): r={r:.3f}, p={p:.3f}")
    print("(Negative CV → lower MAE expected, so negative r means CV predicts MAE)")

# ── Plot ──
fig, axes = plt.subplots(2, 5, figsize=(22, 8))
all_keys = distance_types + ['norm_temporal_only', 'norm_spatial_only']

for ax, key in zip(axes.flat, all_keys):
    samples = dist_samples[key]
    ax.hist(samples, bins=80, density=True, alpha=0.7, color='steelblue')
    cv = samples.std() / samples.mean() if samples.mean() > 0 else 0
    ax.set_title(f'{key}\nCV={cv:.3f}', fontsize=10)
    ax.axvline(samples.mean(), color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('Distance')
    ax.set_ylabel('Density')

fig.suptitle('Pairwise Distance Distributions (SAN_BERNARDINO, train set)\nHigher CV = better sample discrimination',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/distance_distribution.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved: {OUTPUT_DIR}/distance_distribution.png")

# ── CV vs MAE scatter ──
fig2, ax2 = plt.subplots(figsize=(8, 6))
for dt in distance_types:
    if dt in mae_by_dist.index and dt in cv_by_dist.index:
        ax2.scatter(cv_by_dist[dt], mae_by_dist[dt], s=100, zorder=5)
        ax2.annotate(dt, (cv_by_dist[dt], mae_by_dist[dt]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)

ax2.set_xlabel('Coefficient of Variation (higher = less concentrated)')
ax2.set_ylabel('Mean MAE (STGCN, all methods/ratios/seeds)')
ax2.set_title('Distance Concentration vs Model Performance')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/cv_vs_mae.png', dpi=150, bbox_inches='tight')
print(f"Plot saved: {OUTPUT_DIR}/cv_vs_mae.png")
