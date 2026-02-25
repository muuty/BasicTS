#!/usr/bin/env python3
"""
1.1 Proxy Metrics vs Training MAE

Joins proxy metrics (OT cost, Sinkhorn divergence, FL objective, redundancy,
information gain, temporal diversity) with Phase A MAE results.
Computes Pearson/Spearman correlations and generates scatter plots.

Only L2 distance types (euclidean, temporal, spatial, combined) have proxy metrics.

Output: experiments/result/analysis/proxy_vs_mae.png
        experiments/result/analysis/proxy_correlations.csv
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

OUTPUT_DIR = 'experiments/result/analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load proxy metrics ──
with open('coreset_indices/SAN_BERNARDINO/proxy_metrics.json') as f:
    proxy_raw = json.load(f)

# Parse filename → (method, distance, ratio, seed)
proxy_rows = []
for fname, metrics in proxy_raw.items():
    # e.g. k_medoids_combined_070_seed42.json
    stem = fname.replace('.json', '')
    parts = stem.split('_')

    # Extract seed
    seed_part = [p for p in parts if p.startswith('seed')]
    if not seed_part:
        continue
    seed = int(seed_part[0].replace('seed', ''))

    # Extract ratio (3-digit number)
    ratio_part = [p for p in parts if p.isdigit() and len(p) == 3]
    if not ratio_part:
        continue
    ratio = int(ratio_part[0]) / 100.0

    # Known distances and methods
    known_distances = {'euclidean', 'temporal', 'spatial', 'combined'}
    known_methods = {'k_medoids', 'k_center', 'graph_cut'}

    # Find distance
    distance = None
    for d in known_distances:
        if d in parts:
            distance = d
            break
    if distance is None:
        continue

    # Find method
    method = None
    if 'k_medoids' in stem:
        method = 'k_medoids'
    elif 'k_center' in stem:
        method = 'k_center'
    elif 'graph_cut' in stem:
        method = 'graph_cut'
    if method is None:
        continue

    # Round ratio to avoid floating point mismatch (0.2999 vs 0.3)
    ratio = round(ratio, 1)

    row = {
        'method': method,
        'distance': distance,
        'ratio': ratio,
        'seed': seed,
        **{k: v for k, v in metrics.items() if k not in ['num_indices', 'total_samples', 'distance_type', 'compute_time_s', 'ratio']},
    }
    proxy_rows.append(row)

proxy_df = pd.DataFrame(proxy_rows)
print(f"Proxy metrics: {len(proxy_df)} entries")
print(proxy_df.head())

# ── Load Phase A results ──
results = pd.read_csv('experiments/result/phase_a_distance_screening.csv')
# Filter to L2 distances only (matching proxy metrics)
l2_distances = ['euclidean', 'temporal', 'spatial', 'combined']
results_l2 = results[results.coreset_distance_type.isin(l2_distances)].copy()
results_l2.rename(columns={
    'coreset_distance_type': 'distance',
    'coreset_selection_strategy': 'method',
    'coreset_selection_ratio': 'ratio',
    'coreset_seed': 'seed',
}, inplace=True)
print(f"\nPhase A results (L2 only): {len(results_l2)} rows")

# ── Join: proxy metrics × results ──
# For STGCN (most data), join on method + distance + ratio + seed
stgcn = results_l2[results_l2.model == 'STGCNChebGraphConv'].copy()
merged = stgcn.merge(proxy_df, on=['method', 'distance', 'ratio', 'seed'], how='inner')
print(f"Merged (STGCN): {len(merged)} rows")

# ── Also aggregate: average over seeds for method×distance×ratio ──
proxy_agg = proxy_df.groupby(['method', 'distance', 'ratio']).mean(numeric_only=True).reset_index()
mae_agg = stgcn.groupby(['method', 'distance', 'ratio'])['MAE_mean'].mean().reset_index()
merged_agg = mae_agg.merge(proxy_agg, on=['method', 'distance', 'ratio'], how='inner')
print(f"Merged (aggregated): {len(merged_agg)} rows")

# ── Compute correlations ──
proxy_metrics = ['ot_cost', 'sinkhorn_divergence', 'fl_objective', 'redundancy',
                 'information_gain', 'h_tod', 'h_dow']

corr_rows = []
for pm in proxy_metrics:
    if pm not in merged.columns:
        continue

    # Per-run correlation
    valid = merged[[pm, 'MAE_mean']].dropna()
    if len(valid) >= 5:
        r_pearson, p_pearson = stats.pearsonr(valid[pm], valid['MAE_mean'])
        r_spearman, p_spearman = stats.spearmanr(valid[pm], valid['MAE_mean'])
        corr_rows.append({
            'proxy_metric': pm,
            'level': 'per_run',
            'n': len(valid),
            'pearson_r': round(r_pearson, 4),
            'pearson_p': round(p_pearson, 4),
            'spearman_r': round(r_spearman, 4),
            'spearman_p': round(p_spearman, 4),
        })

    # Aggregated correlation
    valid_agg = merged_agg[[pm, 'MAE_mean']].dropna()
    if len(valid_agg) >= 5:
        r_pearson, p_pearson = stats.pearsonr(valid_agg[pm], valid_agg['MAE_mean'])
        r_spearman, p_spearman = stats.spearmanr(valid_agg[pm], valid_agg['MAE_mean'])
        corr_rows.append({
            'proxy_metric': pm,
            'level': 'aggregated',
            'n': len(valid_agg),
            'pearson_r': round(r_pearson, 4),
            'pearson_p': round(p_pearson, 4),
            'spearman_r': round(r_spearman, 4),
            'spearman_p': round(p_spearman, 4),
        })

    # Per-ratio correlation
    for r in [0.3, 0.7]:
        subset = merged[merged.ratio == r][[pm, 'MAE_mean']].dropna()
        if len(subset) >= 5:
            r_pearson, p_pearson = stats.pearsonr(subset[pm], subset['MAE_mean'])
            r_spearman, p_spearman = stats.spearmanr(subset[pm], subset['MAE_mean'])
            corr_rows.append({
                'proxy_metric': pm,
                'level': f'ratio_{r}',
                'n': len(subset),
                'pearson_r': round(r_pearson, 4),
                'pearson_p': round(p_pearson, 4),
                'spearman_r': round(r_spearman, 4),
                'spearman_p': round(p_spearman, 4),
            })

corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv(f'{OUTPUT_DIR}/proxy_correlations.csv', index=False)
print("\n=== Proxy Metric Correlations with MAE ===")
print(corr_df.to_string(index=False, float_format='{:.4f}'.format))

# ── Plot: scatter grid ──
# Select most interesting metrics
plot_metrics = ['sinkhorn_divergence', 'fl_objective', 'information_gain', 'h_tod']
method_colors = {'k_medoids': 'tab:blue', 'k_center': 'tab:orange', 'graph_cut': 'tab:green'}
distance_markers = {'euclidean': 'o', 'temporal': 's', 'spatial': '^', 'combined': 'D'}

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

for ax, pm in zip(axes.flat, plot_metrics):
    for _, row in merged.iterrows():
        ax.scatter(row[pm], row['MAE_mean'],
                  color=method_colors[row['method']],
                  marker=distance_markers[row['distance']],
                  s=80, alpha=0.7, edgecolors='black', linewidths=0.5)

    # Correlation annotation
    valid = merged[[pm, 'MAE_mean']].dropna()
    r, p = stats.spearmanr(valid[pm], valid['MAE_mean'])
    ax.set_title(f'{pm}\nSpearman r={r:.3f} (p={p:.3f})', fontsize=11)
    ax.set_xlabel(pm)
    ax.set_ylabel('MAE')

# Add legends
from matplotlib.lines import Line2D
method_legend = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10, label=m)
                 for m, c in method_colors.items()]
dist_legend = [Line2D([0], [0], marker=mk, color='w', markerfacecolor='gray', markersize=10, label=d)
               for d, mk in distance_markers.items()]

fig.legend(handles=method_legend, loc='upper left', bbox_to_anchor=(0.01, 0.99),
          title='Method', fontsize=9, title_fontsize=10)
fig.legend(handles=dist_legend, loc='upper right', bbox_to_anchor=(0.99, 0.99),
          title='Distance', fontsize=9, title_fontsize=10)

fig.suptitle('Proxy Metrics vs MAE (STGCN, L2 distances, per-run)',
            fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(f'{OUTPUT_DIR}/proxy_vs_mae.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved: {OUTPUT_DIR}/proxy_vs_mae.png")

# ── Additional: proxy metrics summary by distance (averaged over methods/seeds) ──
print("\n=== Proxy Metrics by Distance (averaged over methods, seeds, ratios) ===")
by_dist = proxy_df.groupby('distance')[proxy_metrics].mean()
print(by_dist.to_string(float_format='{:.4f}'.format))

print("\n=== Proxy Metrics by Method (averaged over distances, seeds, ratios) ===")
by_method = proxy_df.groupby('method')[proxy_metrics].mean()
print(by_method.to_string(float_format='{:.4f}'.format))

# ── Per-method correlation (does proxy predict MAE *within* a method?) ──
print("\n=== Per-Method Spearman Correlations ===")
for method in ['k_medoids', 'k_center', 'graph_cut']:
    subset = merged[merged.method == method]
    if len(subset) < 5:
        continue
    print(f"\n{method} (n={len(subset)}):")
    for pm in proxy_metrics:
        if pm not in subset.columns:
            continue
        valid = subset[[pm, 'MAE_mean']].dropna()
        if len(valid) >= 4:
            r, p = stats.spearmanr(valid[pm], valid['MAE_mean'])
            sig = '*' if p < 0.05 else ''
            print(f"  {pm:25s}: r={r:+.3f} p={p:.3f} {sig}")
