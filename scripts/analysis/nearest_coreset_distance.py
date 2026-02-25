#!/usr/bin/env python
"""
Nearest coreset point distance distribution.

For each data point, compute distance to its nearest coreset point.
Sort from nearest → farthest to reveal coverage inequality.

Usage:
    conda activate cuda && python scripts/analysis/nearest_coreset_distance.py
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append('.')
outdir = 'experiments/result/analysis'

# ── Load features ────────────────────────────────────────────────────────────
print("Loading features...")
from easytorch.config import import_config
from coreset.distance import extract_features, get_features_by_type
from experiments.select_coreset import get_dataset_from_config

cfg = import_config('baselines/STGCN/SAN_BERNARDINO/SAN_BERNARDINO.py', verbose=False)
dataset = get_dataset_from_config(cfg)
inputs, targets = extract_features(dataset, cfg['MODEL'])
features = get_features_by_type(inputs, targets, 'combined')

pca = PCA(n_components=50, random_state=42)
feat_pca = pca.fit_transform(features)
print(f"  PCA: {features.shape[1]} → 50 dims, var={pca.explained_variance_ratio_.sum():.3f}")

dataset_size = len(dataset)
steps_per_day = 288
tod_all = np.array([(i % steps_per_day) // (steps_per_day // 24) for i in range(dataset_size)])

# ── Load indices ─────────────────────────────────────────────────────────────
index_dir = 'coreset_indices/SAN_BERNARDINO'
methods = {
    'graph_cut+combined': f'{index_dir}/graph_cut_combined_030_seed42.json',
    'graph_cut+euclidean': f'{index_dir}/graph_cut_euclidean_030_seed42.json',
    'k_medoids+combined': f'{index_dir}/k_medoids_combined_030_seed42.json',
    'k_center+combined': f'{index_dir}/k_center_combined_030_seed42.json',
    'random+euclidean': f'{index_dir}/random_euclidean_030_seed42.json',
}

# ── Compute nearest distances ────────────────────────────────────────────────
print("Computing nearest coreset distances...")

def nearest_distances(feat, indices, batch_size=1000):
    """For each point, distance to nearest coreset point."""
    coreset_feat = feat[indices]  # (|S|, D)
    N = len(feat)
    dists = np.zeros(N)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        # (batch, D) vs (|S|, D) → (batch, |S|)
        diff = feat[start:end, None, :] - coreset_feat[None, :, :]
        d = np.sqrt(np.sum(diff ** 2, axis=-1))  # (batch, |S|)
        dists[start:end] = d.min(axis=1)
    return dists

results = {}
for name, path in methods.items():
    with open(path) as f:
        idx = json.load(f)
    print(f"  {name}...", end=" ", flush=True)
    dists = nearest_distances(feat_pca, idx)
    results[name] = dists
    print(f"mean={dists.mean():.2f}, max={dists.max():.2f}, "
          f"p95={np.percentile(dists, 95):.2f}, p99={np.percentile(dists, 99):.2f}")

# ── Plot 1: Sorted distance curves ──────────────────────────────────────────
print("\nPlot 1: Sorted distance curves...")

colors = {
    'graph_cut+combined': '#3498db',
    'graph_cut+euclidean': '#2980b9',
    'k_medoids+combined': '#2ecc71',
    'k_center+combined': '#e74c3c',
    'random+euclidean': '#95a5a6',
}

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: full sorted curve
ax = axes[0]
x_pct = np.linspace(0, 100, dataset_size)
for name, dists in results.items():
    sorted_d = np.sort(dists)
    ax.plot(x_pct, sorted_d, color=colors[name], linewidth=2, label=name)

ax.set_xlabel('Data Points (sorted by distance, %)', fontsize=12)
ax.set_ylabel('Distance to Nearest Coreset Point', fontsize=12)
ax.set_title('Coverage Inequality: Sorted Distance Curves', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Right: zoom into top 20% (worst-covered points)
ax = axes[1]
cutoff = int(dataset_size * 0.8)
x_pct_tail = np.linspace(80, 100, dataset_size - cutoff)
for name, dists in results.items():
    sorted_d = np.sort(dists)[cutoff:]
    ax.plot(x_pct_tail, sorted_d, color=colors[name], linewidth=2, label=name)

ax.set_xlabel('Data Points (percentile)', fontsize=12)
ax.set_ylabel('Distance to Nearest Coreset Point', fontsize=12)
ax.set_title('Zoom: Worst-Covered 20% of Data', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle('How Far is Each Data Point from Its Nearest Coreset Representative? (ratio=0.3)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{outdir}/nearest_coreset_distance_sorted.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved sorted curves.")

# ── Plot 2: Distance by hour-of-day (per method) ────────────────────────────
print("Plot 2: Distance by hour-of-day...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
plot_order = ['graph_cut+combined', 'graph_cut+euclidean', 'k_center+combined',
              'k_medoids+combined', 'random+euclidean']

for i, (ax, name) in enumerate(zip(axes.flat, plot_order)):
    dists = results[name]
    # Box plot per hour
    hourly_dists = [dists[tod_all == h] for h in range(24)]
    bp = ax.boxplot(hourly_dists, positions=range(24), widths=0.6,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='black', linewidth=1.5))
    for patch in bp['boxes']:
        patch.set_facecolor(colors[name])
        patch.set_alpha(0.6)

    # Mean line
    means = [np.mean(d) for d in hourly_dists]
    ax.plot(range(24), means, 'k-o', markersize=3, linewidth=1, label='mean')

    # Highlight worst hours
    worst_hour = np.argmax(means)
    ax.axvspan(worst_hour - 0.5, worst_hour + 0.5, color='red', alpha=0.1)

    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Distance to Nearest Coreset Point')
    ax.set_title(f'{name}\nmean_dist={np.mean(dists):.1f}, worst_hour={worst_hour}:00',
                 fontsize=10, fontweight='bold')
    ax.set_xticks([0, 4, 8, 12, 16, 20])
    ax.grid(axis='y', alpha=0.3)

# Hide last subplot
axes[1, 2].set_visible(False)

plt.suptitle('Distance to Nearest Coreset Point by Hour-of-Day (ratio=0.3)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{outdir}/nearest_coreset_distance_by_hour.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved hourly boxplots.")

# ── Plot 3: Summary statistics table ─────────────────────────────────────────
print("\nSummary Statistics:")
print(f"{'Method':25s} {'mean':>7s} {'median':>7s} {'p90':>7s} {'p95':>7s} {'p99':>7s} {'max':>7s} {'std':>7s} {'Gini':>6s}")
print("-" * 90)
for name, dists in results.items():
    # Gini coefficient (measure of inequality)
    sorted_d = np.sort(dists)
    n = len(sorted_d)
    gini = (2 * np.sum((np.arange(1, n+1) * sorted_d)) / (n * np.sum(sorted_d))) - (n + 1) / n

    print(f"{name:25s} {dists.mean():7.2f} {np.median(dists):7.2f} "
          f"{np.percentile(dists, 90):7.2f} {np.percentile(dists, 95):7.2f} "
          f"{np.percentile(dists, 99):7.2f} {dists.max():7.2f} {dists.std():7.2f} {gini:6.3f}")

print("\n(Gini: 0=perfect equality, 1=max inequality)")
