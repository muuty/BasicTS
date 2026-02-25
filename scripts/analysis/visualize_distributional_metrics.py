#!/usr/bin/env python
"""
Distributional metrics visualization suite.

Generates 5 plots:
  A1. Cross-metric correlation heatmap
  A2. Method comparison radar chart
  A3. Method × metric grouped bar chart
  A4. KL_tod vs KL_feature scatter plot
  A5. Temporal distribution histogram per method

Usage:
    conda activate cuda && python scripts/analysis/visualize_distributional_metrics.py
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
outdir = 'experiments/result/analysis'
os.makedirs(outdir, exist_ok=True)

# ── Load Data ────────────────────────────────────────────────────────────────
df_all = pd.read_csv(f'{outdir}/full_proxy_metrics_table.csv')
df_avg = pd.read_csv(f'{outdir}/full_proxy_metrics_seed_averaged.csv')

smart_methods = ['k_center', 'k_medoids', 'graph_cut']
baseline_methods = ['random', 'stride', 'recent']
method_colors = {
    'k_center': '#e74c3c',
    'k_medoids': '#2ecc71',
    'graph_cut': '#3498db',
    'random': '#95a5a6',
    'stride': '#7f8c8d',
    'recent': '#bdc3c7',
}
distance_markers = {
    'euclidean': 'o',
    'temporal': 's',
    'spatial': '^',
    'combined': 'D',
}

# ═══════════════════════════════════════════════════════════════════════════════
# A1. Cross-Metric Correlation Heatmap
# ═══════════════════════════════════════════════════════════════════════════════
print("A1. Correlation Heatmap...")

r03 = df_avg[df_avg['ratio'] == 0.3]
corr_cols = ['ot_cost', 'sinkhorn_div', 'kl_tod', 'js_tod', 'kl_feature', 'js_feature',
             'fl_objective', 'redundancy', 'h_tod', 'h_dow']
# Use nicer labels
label_map = {
    'ot_cost': 'OT Cost',
    'sinkhorn_div': 'Sinkhorn Div',
    'kl_tod': 'KL (ToD)',
    'js_tod': 'JS (ToD)',
    'kl_feature': 'KL (Feature)',
    'js_feature': 'JS (Feature)',
    'fl_objective': 'FL Objective',
    'redundancy': 'Redundancy',
    'h_tod': 'H (ToD)',
    'h_dow': 'H (DoW)',
}

corr = r03[corr_cols].corr()
corr.index = [label_map[c] for c in corr.index]
corr.columns = [label_map[c] for c in corr.columns]

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, square=True, linewidths=0.5,
            cbar_kws={'label': 'Pearson Correlation'}, ax=ax)
ax.set_title('Cross-Metric Correlations (ratio=0.3, seed-averaged)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{outdir}/dist_metrics_A1_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved A1.")

# ═══════════════════════════════════════════════════════════════════════════════
# A2. Method Comparison Radar Chart
# ═══════════════════════════════════════════════════════════════════════════════
print("A2. Radar Chart...")

radar_metrics = ['fl_objective', 'sinkhorn_div', 'kl_tod', 'redundancy', 'h_tod', 'js_feature']
radar_labels = ['FL Obj ↑', '1-Sinkhorn ↑', '1-KL(ToD) ↑', '1-Redundancy ↑', 'H(ToD) ↑', '1-JS(Feat) ↑']
# Higher is better for all (we invert where needed)

r03_data = df_avg[df_avg['ratio'] == 0.3].copy()

# Normalize all to [0, 1] then invert for "lower is better" metrics
radar_vals = {}
for col in radar_metrics:
    vals = r03_data[col].values
    min_v, max_v = vals.min(), vals.max()
    if max_v == min_v:
        normed = np.ones_like(vals) * 0.5
    else:
        normed = (vals - min_v) / (max_v - min_v)
    # Invert metrics where lower is better
    if col in ['sinkhorn_div', 'kl_tod', 'redundancy', 'js_feature']:
        normed = 1.0 - normed
    radar_vals[col] = normed

# Select representative configs for radar
representatives = [
    ('k_medoids', 'combined', '#2ecc71', '-'),
    ('k_medoids', 'temporal', '#27ae60', '--'),
    ('k_center', 'combined', '#e74c3c', '-'),
    ('graph_cut', 'combined', '#3498db', '-'),
    ('graph_cut', 'euclidean', '#2980b9', '--'),
    ('random', 'euclidean', '#95a5a6', ':'),
    ('recent', 'euclidean', '#bdc3c7', ':'),
    ('stride', 'euclidean', '#7f8c8d', ':'),
]

n_metrics = len(radar_metrics)
angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
angles += angles[:1]  # close the plot

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

for method, dist, color, ls in representatives:
    mask = (r03_data['method'] == method) & (r03_data['distance'] == dist)
    if mask.sum() == 0:
        continue
    idx = r03_data[mask].index[0]
    row_idx = np.where(r03_data.index == idx)[0][0]
    values = [radar_vals[col][row_idx] for col in radar_metrics]
    values += values[:1]
    label = f"{method}+{dist}"
    ax.plot(angles, values, color=color, linestyle=ls, linewidth=2, label=label)
    ax.fill(angles, values, alpha=0.05, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_labels, fontsize=10)
ax.set_ylim(0, 1.05)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8, color='gray')
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)
ax.set_title('Method Profiles (ratio=0.3, normalized)', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{outdir}/dist_metrics_A2_radar_chart.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved A2.")

# ═══════════════════════════════════════════════════════════════════════════════
# A3. Method × Metric Grouped Bar Chart
# ═══════════════════════════════════════════════════════════════════════════════
print("A3. Grouped Bar Chart...")

bar_metrics = [
    ('sinkhorn_div', 'Sinkhorn Divergence (↓)', True),
    ('kl_tod', 'KL Divergence ToD (↓)', True),
    ('js_feature', 'JS Divergence Feature (↓)', True),
    ('fl_objective', 'FL Objective (↑)', False),
    ('redundancy', 'Redundancy (↓)', True),
    ('h_tod', 'H(ToD) Entropy (↑)', False),
]

r03_plot = df_avg[df_avg['ratio'] == 0.3].copy()
r03_plot['label'] = r03_plot['method'] + '\n' + r03_plot['distance']
r03_plot = r03_plot.sort_values(['method', 'distance'])

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, (col, title, lower_better) in zip(axes.flat, bar_metrics):
    colors = [method_colors[m] for m in r03_plot['method']]
    bars = ax.bar(range(len(r03_plot)), r03_plot[col], color=colors, width=0.7, edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(r03_plot)))
    ax.set_xticklabels(r03_plot['label'], rotation=45, ha='right', fontsize=7)
    ax.set_title(title, fontsize=11, fontweight='bold')
    if lower_better:
        ax.invert_yaxis()
    ax.grid(axis='y', alpha=0.3)

# Legend
legend_patches = [Patch(facecolor=c, label=m) for m, c in method_colors.items()]
fig.legend(handles=legend_patches, loc='upper center', ncol=6, fontsize=10,
           bbox_to_anchor=(0.5, 1.02))
plt.suptitle('Distributional Metrics by Method+Distance (ratio=0.3)', fontsize=14, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig(f'{outdir}/dist_metrics_A3_bar_chart.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved A3.")

# ═══════════════════════════════════════════════════════════════════════════════
# A4. KL_tod vs KL_feature Scatter Plot
# ═══════════════════════════════════════════════════════════════════════════════
print("A4. Scatter Plot...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, ratio in zip(axes, [0.3, 0.7]):
    sub = df_avg[df_avg['ratio'] == ratio]
    for _, row in sub.iterrows():
        m = row['method']
        d = row['distance']
        ax.scatter(row['kl_tod'], row['kl_feature'],
                   c=method_colors[m], marker=distance_markers.get(d, 'x'),
                   s=100, edgecolors='black', linewidths=0.5, zorder=5)
        # Label
        ax.annotate(f"{m[:2]}+{d[:3]}", (row['kl_tod'], row['kl_feature']),
                    fontsize=6, ha='center', va='bottom', textcoords='offset points',
                    xytext=(0, 5))

    ax.set_xlabel('KL Divergence (Time-of-Day)', fontsize=11)
    ax.set_ylabel('KL Divergence (Feature Space)', fontsize=11)
    ax.set_title(f'Temporal vs Feature Coverage (ratio={ratio})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    # Add diagonal reference
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.2, linewidth=1)

# Legend
method_patches = [Patch(facecolor=c, label=m) for m, c in method_colors.items() if m in sub['method'].values]
from matplotlib.lines import Line2D
dist_handles = [Line2D([0], [0], marker=v, color='gray', linestyle='', markersize=8, label=k)
                for k, v in distance_markers.items()]
fig.legend(handles=method_patches + dist_handles, loc='upper center', ncol=5,
           fontsize=9, bbox_to_anchor=(0.5, 1.05))
plt.tight_layout()
plt.savefig(f'{outdir}/dist_metrics_A4_scatter_kl.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved A4.")

# ═══════════════════════════════════════════════════════════════════════════════
# A5. Temporal Distribution Histogram (per method)
# ═══════════════════════════════════════════════════════════════════════════════
print("A5. Temporal Histograms...")

# Load index files for representative configs (ratio=0.3, distance=combined, seed=42)
index_dir = 'coreset_indices/SAN_BERNARDINO'
dataset_size = 14493
steps_per_day = 288
bins_per_hour = steps_per_day // 24
tod_all = np.array([i % steps_per_day for i in range(dataset_size)])
hour_all = tod_all // bins_per_hour

full_hist = np.bincount(hour_all, minlength=24).astype(float)
full_hist_norm = full_hist / full_hist.sum()

configs_to_plot = [
    ('k_medoids', 'combined', 42),
    ('k_center', 'combined', 42),
    ('graph_cut', 'combined', 42),
    ('random', 'euclidean', 42),
    ('recent', 'euclidean', 42),
    ('stride', 'euclidean', 42),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
hours = np.arange(24)

for ax, (method, dist, seed) in zip(axes.flat, configs_to_plot):
    fname = f"{method}_{dist}_030_seed{seed}.json"
    fpath = os.path.join(index_dir, fname)
    if not os.path.exists(fpath):
        ax.set_title(f"{method}+{dist} (not found)")
        continue

    with open(fpath) as f:
        indices = json.load(f)

    hour_sub = (tod_all[indices] // bins_per_hour).astype(int)
    sub_hist = np.bincount(hour_sub, minlength=24).astype(float)
    sub_hist_norm = sub_hist / sub_hist.sum()

    # Compute coverage gap
    gap = np.abs(sub_hist_norm - full_hist_norm) / (full_hist_norm + 1e-10)

    # Bar chart
    width = 0.35
    ax.bar(hours - width/2, full_hist_norm, width, label='Full Dataset', color='lightgray', edgecolor='gray')
    ax.bar(hours + width/2, sub_hist_norm, width, label='Coreset',
           color=method_colors[method], edgecolor='white', alpha=0.8)

    # Highlight hours with >20% gap
    for h in range(24):
        if gap[h] > 0.2:
            ax.axvspan(h - 0.5, h + 0.5, color='red', alpha=0.08)

    kl = np.sum((sub_hist_norm + 1e-10) * np.log((sub_hist_norm + 1e-10) / (full_hist_norm + 1e-10)))
    max_gap_hour = np.argmax(gap)

    ax.set_title(f"{method}+{dist}\nKL={kl:.4f}, max_gap=h{max_gap_hour}({gap[max_gap_hour]:.0%})",
                 fontsize=10, fontweight='bold')
    ax.set_xticks([0, 4, 8, 12, 16, 20])
    ax.set_xticklabels(['0', '4', '8', '12', '16', '20'])
    ax.set_xlabel('Hour')
    ax.set_ylabel('Proportion')
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Temporal (Hour-of-Day) Distribution: Coreset vs Full Dataset (ratio=0.3)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{outdir}/dist_metrics_A5_temporal_hist.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved A5.")

print(f"\nAll visualizations saved to {outdir}/dist_metrics_A*.png")
