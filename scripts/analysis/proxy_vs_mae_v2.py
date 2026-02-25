#!/usr/bin/env python
"""
Proxy Metrics vs MAE Correlation (v2) — includes all distributional metrics.

Merges full_proxy_metrics_table.csv with phase_a_distance_screening.csv
to find which proxy metric best predicts downstream MAE.

Usage:
    conda activate cuda && python scripts/analysis/proxy_vs_mae_v2.py
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
outdir = 'experiments/result/analysis'

# ── Load proxy metrics ──────────────────────────────────────────────────────
df_proxy = pd.read_csv(f'{outdir}/full_proxy_metrics_table.csv')
print(f"Proxy metrics: {len(df_proxy)} rows, columns: {list(df_proxy.columns)}")

# ── Load MAE results ────────────────────────────────────────────────────────
df_mae = pd.read_csv('experiments/result/phase_a_distance_screening.csv')
# Filter to STGCN only (most data)
df_mae = df_mae[df_mae['model'] == 'STGCNChebGraphConv'].copy()
# Rename columns for merge
df_mae = df_mae.rename(columns={
    'coreset_selection_strategy': 'method',
    'coreset_distance_type': 'distance',
    'coreset_selection_ratio': 'ratio',
    'coreset_seed': 'seed',
})
# Round ratio for matching
df_mae['ratio'] = df_mae['ratio'].round(1)
print(f"MAE results (STGCN): {len(df_mae)} rows")

# Filter to smart methods only (baselines don't have varying distance)
smart = ['k_center', 'k_medoids', 'graph_cut']
df_mae_smart = df_mae[df_mae['method'].isin(smart)]
df_proxy_smart = df_proxy[df_proxy['method'].isin(smart)]

# Merge
merged = pd.merge(
    df_proxy_smart,
    df_mae_smart[['method', 'distance', 'ratio', 'seed', 'MAE_mean']],
    on=['method', 'distance', 'ratio', 'seed'],
    how='inner'
)
print(f"Merged: {len(merged)} rows")

if len(merged) == 0:
    print("ERROR: No merge matches. Checking keys...")
    print("Proxy distances:", sorted(df_proxy_smart['distance'].unique()))
    print("MAE distances:", sorted(df_mae_smart['distance'].unique()))
    print("Proxy seeds:", sorted(df_proxy_smart['seed'].unique()))
    print("MAE seeds:", sorted(df_mae_smart['seed'].unique()))
    sys.exit(1)

# Also merge baselines (distance = 'none' or similar)
baseline_methods = ['random', 'stride', 'recent']
df_mae_base = df_mae[df_mae['method'].isin(baseline_methods)]
df_proxy_base = df_proxy[df_proxy['method'].isin(baseline_methods)]

if len(df_mae_base) > 0 and len(df_proxy_base) > 0:
    # Baselines have distance='euclidean' in proxy, may differ in MAE
    merged_base = pd.merge(
        df_proxy_base,
        df_mae_base[['method', 'ratio', 'seed', 'MAE_mean']],
        on=['method', 'ratio', 'seed'],
        how='inner'
    )
    merged_all = pd.concat([merged, merged_base], ignore_index=True)
    print(f"With baselines: {len(merged_all)} rows")
else:
    merged_all = merged

# ── Correlation Analysis ────────────────────────────────────────────────────
print("\n" + "="*80)
print("PROXY METRIC vs MAE CORRELATION RANKING")
print("="*80)

proxy_cols = [c for c in merged_all.columns if c not in
              ['method', 'distance', 'ratio', 'seed', 'MAE_mean', 'num_indices',
               'coverage_gap_max_hour', 'underrep_hours']]

results = []
for col in proxy_cols:
    valid = merged_all[[col, 'MAE_mean']].dropna()
    if len(valid) < 5:
        continue
    r_pearson, p_pearson = stats.pearsonr(valid[col], valid['MAE_mean'])
    r_spearman, p_spearman = stats.spearmanr(valid[col], valid['MAE_mean'])
    results.append({
        'metric': col,
        'pearson_r': r_pearson,
        'pearson_p': p_pearson,
        'spearman_r': r_spearman,
        'spearman_p': p_spearman,
        'abs_spearman': abs(r_spearman),
        'n': len(valid),
    })

df_corr = pd.DataFrame(results).sort_values('abs_spearman', ascending=False)

print(f"\n{'Metric':22s} {'Pearson r':>10s} {'p':>8s} {'Spearman r':>10s} {'p':>8s} {'n':>4s}")
print("-" * 70)
for _, row in df_corr.iterrows():
    sig = '***' if row['spearman_p'] < 0.001 else '**' if row['spearman_p'] < 0.01 else '*' if row['spearman_p'] < 0.05 else ''
    print(f"{row['metric']:22s} {row['pearson_r']:10.4f} {row['pearson_p']:8.4f} "
          f"{row['spearman_r']:10.4f} {row['spearman_p']:8.4f} {int(row['n']):4d} {sig}")

# ── Per-method correlation ──────────────────────────────────────────────────
print("\n" + "="*80)
print("PER-METHOD CORRELATION (top metrics)")
print("="*80)

top_metrics = df_corr.head(6)['metric'].tolist()

for method in smart:
    sub = merged[merged['method'] == method]
    if len(sub) < 5:
        continue
    print(f"\n  {method} (n={len(sub)}):")
    for col in top_metrics:
        valid = sub[[col, 'MAE_mean']].dropna()
        if len(valid) < 4:
            continue
        r, p = stats.spearmanr(valid[col], valid['MAE_mean'])
        sig = '*' if p < 0.05 else ''
        print(f"    {col:22s}: r={r:7.4f} p={p:.4f} {sig}")

# ── Visualization: Top 6 metrics scatter plots ──────────────────────────────
print("\nGenerating scatter plots...")

method_colors = {
    'k_center': '#e74c3c',
    'k_medoids': '#2ecc71',
    'graph_cut': '#3498db',
    'random': '#95a5a6',
    'stride': '#7f8c8d',
    'recent': '#bdc3c7',
}

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for ax, metric in zip(axes.flat, top_metrics):
    for method in merged_all['method'].unique():
        sub = merged_all[merged_all['method'] == method]
        ax.scatter(sub[metric], sub['MAE_mean'],
                   c=method_colors.get(method, 'gray'), label=method,
                   s=40, alpha=0.7, edgecolors='white', linewidths=0.5)
    # Trend line
    valid = merged_all[[metric, 'MAE_mean']].dropna()
    z = np.polyfit(valid[metric], valid['MAE_mean'], 1)
    x_line = np.linspace(valid[metric].min(), valid[metric].max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), 'k--', alpha=0.3, linewidth=1)

    r, p = stats.spearmanr(valid[metric], valid['MAE_mean'])
    ax.set_xlabel(metric, fontsize=10)
    ax.set_ylabel('MAE', fontsize=10)
    ax.set_title(f'ρ={r:.3f} (p={p:.3f})', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

# Legend
from matplotlib.patches import Patch
handles = [Patch(facecolor=c, label=m) for m, c in method_colors.items()
           if m in merged_all['method'].unique()]
fig.legend(handles=handles, loc='upper center', ncol=6, fontsize=10,
           bbox_to_anchor=(0.5, 1.02))
plt.suptitle('Proxy Metrics vs MAE (STGCN, SAN_BERNARDINO)', fontsize=14,
             fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig(f'{outdir}/dist_metrics_B3_proxy_vs_mae.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {outdir}/dist_metrics_B3_proxy_vs_mae.png")
