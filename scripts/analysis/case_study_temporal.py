#!/usr/bin/env python3
"""
Case Study A — What does each selection method preserve, temporally?

For each (hour, day-of-week) cell, compute the *selection ratio*: the
fraction of full-training windows in that cell that the method chose.

  selection_ratio[h, d]
      = (# selected windows in (h, d)) / (# full-training windows in (h, d))

A method that samples uniformly over time has selection_ratio ≈ r in every
cell (the global sampling ratio). Higher cells are over-sampled; lower
cells are under-sampled. The colormap is sequential so that high-density
cells stand out without a centred reference.

Dataset metadata:
  - SAN_BERNARDINO / CONTRA_COSTA: 2023, 5-min resolution, 288 steps/day
  - 2023-01-01 was a Sunday, so day-of-week(i) = (i // 288) % 7, Sun=0
  - Training spans first 14516 steps after integer 6:2:2 splitting
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)

OUT = ROOT / 'experiments/result/analysis'
FIG = ROOT / 'writing/CoresetSelection-paper/figures'

DATASETS = {
    'SAN_BERNARDINO': {'train_end': 14516, 'start_dow_sun0': 0},
    'CONTRA_COSTA':   {'train_end': 14516, 'start_dow_sun0': 0},
}
DOW_LABELS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

# All six main methods, but only four are plotted (the most contrasting).
ALL_METHODS = ['k_medoids', 'random', 'stride', 'recent', 'k_center', 'graph_cut']
PLOT_METHODS = ['k_medoids', 'stride', 'k_center', 'graph_cut']
METHOD_LABELS = {
    'k_medoids': 'K-medoids',
    'random': 'Random',
    'stride': 'Stride',
    'recent': 'Recent',
    'k_center': 'K-center',
    'graph_cut': 'Graph Cut',
}
GROUP_LABEL = {
    'k_medoids': 'average distance',
    'random':    'uniform random',
    'stride':    'fixed interval',
    'recent':    'most recent',
    'k_center':  'maximum radius',
    'graph_cut': 'similarity - redundancy',
}
SEEDS = [42, 123, 456]
RATIO = 30
DISTANCE = 'euclidean'


def load_indices(dataset, method, ratio, distance, seed):
    p = f'coreset_indices/{dataset}/{method}_{distance}_{ratio:03d}_seed{seed}.json'
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return np.array(json.load(f), dtype=int)


def hour_dow_counts(indices, start_dow_sun0=0):
    """Return a 7 x 24 (dow, hour) count matrix."""
    hour = (indices % 288) // 12
    dow = (indices // 288 + start_dow_sun0) % 7
    counts = np.zeros((7, 24), dtype=float)
    np.add.at(counts, (dow, hour), 1)
    return counts


def total_variation(p, q):
    p = p / p.sum()
    q = q / q.sum()
    return 0.5 * np.abs(p - q).sum()


def selection_ratio_per_cell(dataset, method):
    """Mean (across seeds) per-cell selection ratio."""
    cfg = DATASETS[dataset]
    train_end = cfg['train_end']
    valid_max = train_end - 12 - 12 + 1
    full_idx = np.arange(0, valid_max)
    full_counts = hour_dow_counts(full_idx, cfg['start_dow_sun0'])

    rates = []
    for seed in SEEDS:
        idx = load_indices(dataset, method, RATIO, DISTANCE, seed)
        if idx is None:
            continue
        sel_counts = hour_dow_counts(idx, cfg['start_dow_sun0'])
        with np.errstate(divide='ignore', invalid='ignore'):
            rate = np.where(full_counts > 0, sel_counts / full_counts, 0.0)
        rates.append(rate)
    if not rates:
        return None, None
    return np.mean(rates, axis=0), full_counts


# ── Compute ──────────────────────────────────────────────────────────────
all_rates = {}      # {dataset: {method: rate_matrix}}
all_full = {}       # {dataset: full_counts}
for ds in DATASETS:
    all_rates[ds] = {}
    for m in ALL_METHODS:
        rate, full = selection_ratio_per_cell(ds, m)
        if rate is not None:
            all_rates[ds][m] = rate
            all_full[ds] = full

# Common color scale across all plotted panels
vmax = 0.0
for ds in DATASETS:
    for m in PLOT_METHODS:
        if m in all_rates[ds]:
            vmax = max(vmax, all_rates[ds][m].max())
vmax = float(np.ceil(vmax * 10) / 10)  # round up to 1 d.p.
print(f"Selection-ratio colour range: 0 to {vmax:.2f}")

# ── Summary CSV (all six methods) ────────────────────────────────────────
rows = []
for ds in DATASETS:
    full_dens = all_full[ds] / all_full[ds].sum()
    rush_mask = np.zeros_like(full_dens, dtype=bool)
    rush_mask[:, 7:10] = True
    rush_mask[:, 16:19] = True
    weekend_mask = np.zeros_like(full_dens, dtype=bool)
    weekend_mask[[0, 6], :] = True
    rows.append({
        'dataset': ds, 'method': 'Full', 'group': 'reference',
        'tv_joint': 0.0,
        'rush_share': float(full_dens[rush_mask].sum()),
        'weekend_share': float(full_dens[weekend_mask].sum()),
    })
    for m in ALL_METHODS:
        if m not in all_rates[ds]:
            continue
        # Approximate per-cell density: rate × full_count
        sel_counts = all_rates[ds][m] * all_full[ds]
        sel_dens = sel_counts / sel_counts.sum()
        rows.append({
            'dataset': ds, 'method': METHOD_LABELS[m], 'group': GROUP_LABEL[m],
            'tv_joint': float(total_variation(sel_dens, full_dens)),
            'rush_share': float(sel_dens[rush_mask].sum()),
            'weekend_share': float(sel_dens[weekend_mask].sum()),
        })
summary = pd.DataFrame(rows)
summary.to_csv(OUT / 'case_study_temporal.csv', index=False)
print("\nSummary (joint TV in (hour, dow) space, plus rush/weekend share):")
print(summary.to_string(index=False))

# ── Figure: 2 datasets × 4 methods, selection-ratio heatmaps ─────────────
fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), constrained_layout=True)

for row, dataset in enumerate(DATASETS):
    for col, method in enumerate(PLOT_METHODS):
        ax = axes[row, col]
        rate = all_rates[dataset][method]
        im = ax.imshow(rate, aspect='auto', cmap='YlGnBu', origin='lower',
                       vmin=0.0, vmax=vmax,
                       extent=[-0.5, 23.5, -0.5, 6.5])
        # cell borders
        for h in range(25):
            ax.axvline(h - 0.5, color='white', lw=0.5, alpha=0.7)
        for d in range(8):
            ax.axhline(d - 0.5, color='white', lw=0.5, alpha=0.7)

        ax.set_xticks(range(0, 24, 3))
        ax.set_yticks(range(7))
        ax.set_yticklabels(DOW_LABELS if col == 0 else [])
        ax.set_xlabel('Hour of day' if row == 1 else '')
        if col == 0:
            ax.set_ylabel(f"{dataset.replace('_', ' ')}\nDay of week",
                          fontsize=9)
        if row == 0:
            ax.set_title(f"{METHOD_LABELS[method]}\n"
                         f"({GROUP_LABEL[method]})",
                         fontsize=10)
        ax.tick_params(labelsize=8)

cb = fig.colorbar(im, ax=axes, location='right', shrink=0.7, pad=0.02,
                  label=f'Selection ratio (target = {RATIO/100:.1f})')
cb.ax.tick_params(labelsize=8)

fig.savefig(FIG / 'case_study_temporal.pdf', dpi=150, bbox_inches='tight')
fig.savefig(FIG / 'case_study_temporal.png', dpi=150, bbox_inches='tight')
print(f"\n→ Figure saved: {FIG}/case_study_temporal.{{pdf,png}}")
