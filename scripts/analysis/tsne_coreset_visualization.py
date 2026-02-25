#!/usr/bin/env python
"""
t-SNE visualization of coreset selections.

Full dataset in gray, coreset selections colored by ToD / DoW.
Compares graph_cut vs k_medoids to show temporal coverage difference.

Usage:
    conda activate cuda && python scripts/analysis/tsne_coreset_visualization.py
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append('.')
outdir = 'experiments/result/analysis'

# ── Load features ────────────────────────────────────────────────────────────
print("Loading dataset and extracting features...")
from easytorch.config import import_config
from coreset.distance import extract_features, get_features_by_type
from experiments.select_coreset import get_dataset_from_config

cfg = import_config('baselines/STGCN/SAN_BERNARDINO/SAN_BERNARDINO.py', verbose=False)
dataset = get_dataset_from_config(cfg)
model_config = cfg['MODEL']
inputs, targets = extract_features(dataset, model_config)
features = get_features_by_type(inputs, targets, 'combined')
print(f"  Features: {features.shape}")  # (14493, 3620)

dataset_size = len(dataset)
steps_per_day = 288
bins_per_hour = steps_per_day // 24
tod_all = np.array([(i % steps_per_day) // bins_per_hour for i in range(dataset_size)])  # 0-23
dow_all = np.array([(i // steps_per_day) % 7 for i in range(dataset_size)])  # 0-6

# ── PCA → t-SNE ─────────────────────────────────────────────────────────────
print("PCA(50) → t-SNE(2)...")
pca = PCA(n_components=50, random_state=42)
features_pca = pca.fit_transform(features)
print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum():.3f}")

tsne = TSNE(n_components=2, random_state=42, perplexity=50, max_iter=1000, init='pca', learning_rate='auto')
coords = tsne.fit_transform(features_pca)
print(f"  t-SNE done: {coords.shape}")

# ── Load coreset indices ─────────────────────────────────────────────────────
index_dir = 'coreset_indices/SAN_BERNARDINO'
configs = {
    'graph_cut+combined': f'{index_dir}/graph_cut_combined_030_seed42.json',
    'graph_cut+euclidean': f'{index_dir}/graph_cut_euclidean_030_seed42.json',
    'k_medoids+combined': f'{index_dir}/k_medoids_combined_030_seed42.json',
    'k_medoids+euclidean': f'{index_dir}/k_medoids_euclidean_030_seed42.json',
    'k_center+combined': f'{index_dir}/k_center_combined_030_seed42.json',
    'random+euclidean': f'{index_dir}/random_euclidean_030_seed42.json',
}

indices = {}
for name, path in configs.items():
    with open(path) as f:
        indices[name] = json.load(f)
    print(f"  {name}: {len(indices[name])} samples")

# ── Plot 1: ToD coloring (2×3 grid) ─────────────────────────────────────────
print("\nGenerating ToD visualization...")

fig, axes = plt.subplots(2, 3, figsize=(20, 13))
plot_configs = [
    ('graph_cut+combined', 'Graph Cut + Combined'),
    ('graph_cut+euclidean', 'Graph Cut + Euclidean'),
    ('k_center+combined', 'K-Center + Combined'),
    ('k_medoids+combined', 'K-Medoids + Combined'),
    ('k_medoids+euclidean', 'K-Medoids + Euclidean'),
    ('random+euclidean', 'Random'),
]

tod_cmap = cm.get_cmap('hsv', 24)

for ax, (key, title) in zip(axes.flat, plot_configs):
    # Full dataset in gray
    ax.scatter(coords[:, 0], coords[:, 1], c='lightgray', s=1, alpha=0.3, rasterized=True)

    # Coreset colored by ToD
    idx = indices[key]
    tod_sel = tod_all[idx]
    sc = ax.scatter(coords[idx, 0], coords[idx, 1],
                    c=tod_sel, cmap='hsv', vmin=0, vmax=24,
                    s=8, alpha=0.7, edgecolors='none', rasterized=True)

    # Compute KL for title
    full_dist = np.bincount(tod_all, minlength=24).astype(float)
    full_dist /= full_dist.sum()
    sel_dist = np.bincount(tod_sel, minlength=24).astype(float)
    sel_dist /= sel_dist.sum()
    kl = np.sum((sel_dist + 1e-10) * np.log((sel_dist + 1e-10) / (full_dist + 1e-10)))

    ax.set_title(f'{title}\nKL(ToD)={kl:.4f}, n={len(idx)}', fontsize=11, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

# Colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
norm = matplotlib.colors.Normalize(vmin=0, vmax=24)
sm = cm.ScalarMappable(cmap='hsv', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Hour of Day', fontsize=12)
cbar.set_ticks([0, 4, 8, 12, 16, 20, 24])

fig.suptitle('t-SNE: Coreset Selections Colored by Time-of-Day (ratio=0.3, seed=42)\nGray = Full Dataset',
             fontsize=14, fontweight='bold', y=0.98)
plt.savefig(f'{outdir}/tsne_coreset_tod.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved tsne_coreset_tod.png")

# ── Plot 2: DoW coloring (2×3 grid) ─────────────────────────────────────────
print("Generating DoW visualization...")

fig, axes = plt.subplots(2, 3, figsize=(20, 13))
dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
dow_cmap = cm.get_cmap('Set1', 7)

for ax, (key, title) in zip(axes.flat, plot_configs):
    ax.scatter(coords[:, 0], coords[:, 1], c='lightgray', s=1, alpha=0.3, rasterized=True)

    idx = indices[key]
    dow_sel = dow_all[idx]
    sc = ax.scatter(coords[idx, 0], coords[idx, 1],
                    c=dow_sel, cmap='Set1', vmin=0, vmax=6,
                    s=8, alpha=0.7, edgecolors='none', rasterized=True)

    ax.set_title(f'{title}\nn={len(idx)}', fontsize=11, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
norm = matplotlib.colors.Normalize(vmin=0, vmax=6)
sm = cm.ScalarMappable(cmap='Set1', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_ticks(range(7))
cbar.set_ticklabels(dow_labels)
cbar.set_label('Day of Week', fontsize=12)

fig.suptitle('t-SNE: Coreset Selections Colored by Day-of-Week (ratio=0.3, seed=42)\nGray = Full Dataset',
             fontsize=14, fontweight='bold', y=0.98)
plt.savefig(f'{outdir}/tsne_coreset_dow.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved tsne_coreset_dow.png")

# ── Plot 3: Reference — full dataset colored by ToD ─────────────────────────
print("Generating reference plot (full dataset ToD)...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Full dataset by ToD
ax = axes[0]
sc = ax.scatter(coords[:, 0], coords[:, 1], c=tod_all, cmap='hsv',
                vmin=0, vmax=24, s=2, alpha=0.5, rasterized=True)
ax.set_title('Full Dataset colored by Hour-of-Day', fontsize=12, fontweight='bold')
ax.set_xticks([])
ax.set_yticks([])
plt.colorbar(sc, ax=ax, label='Hour')

# Full dataset by DoW
ax = axes[1]
sc = ax.scatter(coords[:, 0], coords[:, 1], c=dow_all, cmap='Set1',
                vmin=0, vmax=6, s=2, alpha=0.5, rasterized=True)
ax.set_title('Full Dataset colored by Day-of-Week', fontsize=12, fontweight='bold')
ax.set_xticks([])
ax.set_yticks([])
cbar = plt.colorbar(sc, ax=ax, label='Day')
cbar.set_ticks(range(7))
cbar.set_ticklabels(dow_labels)

plt.suptitle('t-SNE Reference: Full Dataset Temporal Structure', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{outdir}/tsne_full_dataset_temporal.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved tsne_full_dataset_temporal.png")

print(f"\nAll t-SNE plots saved to {outdir}/tsne_*.png")
