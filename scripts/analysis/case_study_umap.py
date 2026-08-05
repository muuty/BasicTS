#!/usr/bin/env python3
"""
Case Study (feature-space coverage) — UMAP overlay of full training windows
versus the selections from each coreset method.

Representation: per-window mean across 893 sensors, over the three traffic
features (flow, speed, occupancy) and the 12 input time-steps. This gives
a 36-dim summary per training window that is small enough to project with
UMAP in a single pass while still reflecting the network-level dynamics
that distinguish rush hour, off-peak, and weekend traffic.

We deliberately bypass easytorch and memmap the raw data file so the script
does not depend on the training stack.
"""
import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import umap

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)

OUT = ROOT / 'experiments/result/analysis'
FIG = ROOT / 'writing/CoresetSelection-paper/figures'

DATASET = 'SAN_BERNARDINO'
DATA_SHAPE = (105120, 893, 5)   # from desc.json
TRAIN_END = 14516               # integer 6:2:2 split of range (0, 24192)
T_IN = 12

RATIO = 10  # use 10% so selected dots are sparse enough to show clustering
DISTANCE = 'euclidean'
SEED = 42

METHODS = ['k_medoids', 'stride', 'k_center', 'graph_cut']
METHOD_LABELS = {
    'k_medoids': 'K-medoids',
    'stride': 'Stride',
    'k_center': 'K-center',
    'graph_cut': 'Graph Cut',
}
GROUP_LABEL = {
    'k_medoids': 'density-matching',
    'stride': 'uniform sampling',
    'k_center': 'diversity-maximising',
    'graph_cut': 'diversity-maximising',
}


def load_training_slice():
    """Sequentially read the first (TRAIN_END + T_IN) timesteps into RAM.

    The data file lives on a parallel filesystem where strided memmap
    access is slow; a single sequential read is far faster than letting
    np.memmap fetch pages on demand.
    """
    n_load = TRAIN_END + T_IN
    n_feat = DATA_SHAPE[2]              # 5
    n_nodes = DATA_SHAPE[1]             # 893
    count = n_load * n_nodes * n_feat   # total float32 elements
    arr = np.fromfile(f'datasets/{DATASET}/data.dat',
                      dtype=np.float32, count=count)
    arr = arr.reshape(n_load, n_nodes, n_feat)
    return np.ascontiguousarray(arr[:, :, :3])  # drop time/dow channels


def extract_window_features(train_arr, n_windows):
    """Per-window sensor-mean: (n_windows, 36) — vectorised, in-memory."""
    # sensor-mean per timestep: (n_load, 3)
    per_step = train_arr.mean(axis=1)
    feats = np.zeros((n_windows, T_IN * 3), dtype=np.float32)
    for i in range(n_windows):
        feats[i] = per_step[i:i + T_IN].reshape(-1)
    return feats


print(f"Loading training slice of {DATASET}/data.dat into RAM...", flush=True)
train_arr = load_training_slice()
print(f"  Loaded: {train_arr.shape}  ({train_arr.nbytes / 1e6:.1f} MB)",
      flush=True)

valid_max = TRAIN_END - T_IN - T_IN + 1
n_windows = valid_max
print(f"  Training windows to embed: {n_windows}", flush=True)

print(f"Extracting sensor-mean features ({T_IN}*3 = {T_IN*3} dims per window)...",
      flush=True)
feats = extract_window_features(train_arr, n_windows)
print(f"  Feature matrix: {feats.shape}  ({feats.nbytes / 1e6:.1f} MB)",
      flush=True)

# Z-score
feats = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-8)

print("Running UMAP(2)...")
reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                    random_state=42, metric='euclidean', verbose=True)
coords = reducer.fit_transform(feats)
print(f"  UMAP done: {coords.shape}")

# Cache
np.savez(OUT / 'case_study_umap_coords.npz',
         coords=coords, n_windows=n_windows)
print(f"→ Coords cached: {OUT}/case_study_umap_coords.npz")

# Load indices for each method
indices = {}
for m in METHODS:
    p = f'coreset_indices/{DATASET}/{m}_{DISTANCE}_{RATIO:03d}_seed{SEED}.json'
    with open(p) as f:
        idx = np.array(json.load(f), dtype=int)
    # Keep only indices within our valid-window range
    indices[m] = idx[idx < n_windows]
    print(f"  {m}: {len(indices[m])} of {len(idx)} selected (in-range)")


# ── Figure: 1×4 panels with marginal KDEs ────────────────────────────────
n_panels = len(METHODS)
fig = plt.figure(figsize=(3.6 * n_panels, 4.2))
outer = GridSpec(1, n_panels, figure=fig, wspace=0.35,
                 left=0.05, right=0.99, top=0.85, bottom=0.10)

x_min, x_max = coords[:, 0].min() - 0.5, coords[:, 0].max() + 0.5
y_min, y_max = coords[:, 1].min() - 0.5, coords[:, 1].max() + 0.5
GRAY = '#bcbcbc'
RED = '#c0392b'

for i, method in enumerate(METHODS):
    inner = outer[i].subgridspec(5, 5, hspace=0.05, wspace=0.05)
    ax_top = fig.add_subplot(inner[0, :4])
    ax_main = fig.add_subplot(inner[1:, :4])
    ax_right = fig.add_subplot(inner[1:, 4])

    # Full data: solid gray cloud
    ax_main.scatter(coords[:, 0], coords[:, 1], s=4, c=GRAY,
                    alpha=0.55, rasterized=True, edgecolors='none')
    sel = indices[method]
    # Selected: smaller, less opaque so gray remains visible underneath
    ax_main.scatter(coords[sel, 0], coords[sel, 1], s=2, c=RED,
                    alpha=0.35, rasterized=True, edgecolors='none')
    ax_main.set_xlim(x_min, x_max)
    ax_main.set_ylim(y_min, y_max)
    ax_main.set_xticks([])
    ax_main.set_yticks([])
    if i == 0:
        ax_main.set_xlabel('UMAP-1', fontsize=9)
        ax_main.set_ylabel('UMAP-2', fontsize=9)

    bins = np.linspace(x_min, x_max, 50)
    ax_top.hist(coords[:, 0], bins=bins, color=GRAY, density=True, alpha=0.8)
    ax_top.hist(coords[sel, 0], bins=bins, color=RED, density=True, alpha=0.55,
                histtype='step', lw=1.4)
    ax_top.set_xlim(x_min, x_max)
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    ax_top.spines[['top', 'right', 'left']].set_visible(False)
    ax_top.set_title(f"{METHOD_LABELS[method]}\n({GROUP_LABEL[method]})",
                     fontsize=10)

    bins_y = np.linspace(y_min, y_max, 50)
    ax_right.hist(coords[:, 1], bins=bins_y, color=GRAY, density=True,
                  alpha=0.8, orientation='horizontal')
    ax_right.hist(coords[sel, 1], bins=bins_y, color=RED, density=True,
                  alpha=0.55, orientation='horizontal', histtype='step', lw=1.4)
    ax_right.set_ylim(y_min, y_max)
    ax_right.set_xticks([])
    ax_right.set_yticks([])
    ax_right.spines[['top', 'right', 'bottom']].set_visible(False)

fig.legend([plt.Line2D([0], [0], marker='o', linestyle='none',
                       color=GRAY, markersize=6),
            plt.Line2D([0], [0], marker='o', linestyle='none',
                       color=RED, markersize=6)],
           ['Full training', 'Selected coreset'],
           loc='upper center', ncol=2, frameon=False, fontsize=9,
           bbox_to_anchor=(0.5, 1.0))

fig.savefig(FIG / 'case_study_umap.pdf', dpi=150, bbox_inches='tight')
fig.savefig(FIG / 'case_study_umap.png', dpi=150, bbox_inches='tight')
print(f"\n→ Figure saved: {FIG}/case_study_umap.{{pdf,png}}")
