#!/usr/bin/env python3
"""
Phase C: Proxy Metrics vs Model MAE — Model-wise Correlation Analysis

Global correlation is meaningless (different MAE scales across models).
All analysis is done per-model (or per model×dataset).

Includes quantization_cost (k-medoids objective in PCA space) if available.

Output:
  experiments/result/analysis/phase_c_proxy_correlations.csv
  experiments/result/analysis/phase_c_proxy_vs_mae.png
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

PROXY_METRICS = ['ot_cost', 'sinkhorn_divergence', 'fl_objective', 'redundancy',
                 'information_gain', 'h_tod', 'h_dow',
                 'quantization_cost', 'quantization_median', 'quantization_max']

KNOWN_METHODS = {'k_medoids', 'k_center', 'graph_cut', 'random', 'stride', 'recent'}

MODEL_SHORT = {
    'AGCRN': 'AGCRN',
    'DCRNN': 'DCRNN',
    'STAEformer': 'STAEformer',
    'STGCNChebGraphConv': 'STGCN',
    'STID': 'STID',
}


def parse_proxy_filename(fname):
    """Parse coreset index filename → (method, distance, ratio, seed)."""
    stem = fname.replace('.json', '')
    parts = stem.split('_')

    seed_part = [p for p in parts if p.startswith('seed')]
    if not seed_part:
        return None
    seed = int(seed_part[0].replace('seed', ''))

    ratio_part = [p for p in parts if p.isdigit() and len(p) == 3]
    if not ratio_part:
        return None
    ratio = round(int(ratio_part[0]) / 100.0, 2)

    known_distances = {'euclidean', 'temporal', 'spatial', 'combined'}
    distance = next((p for p in parts if p in known_distances), None)
    if distance is None:
        return None

    # Accept only canonical METHOD_DISTANCE_RATIO_seed filenames.  The index
    # directories also retain variants such as k_medoids_old_* and
    # k_medoids_v2_*; treating those aliases as canonical k_medoids silently
    # duplicates rows in the MAE/proxy join.
    method = None
    for m in sorted(KNOWN_METHODS, key=len, reverse=True):
        prefix = m + '_'
        if stem.startswith(prefix):
            remainder = stem[len(prefix):]
            if remainder.split('_', 1)[0] in known_distances:
                method = m
            break
    if method is None:
        return None

    return {'method': method, 'distance': distance, 'ratio': ratio, 'seed': seed}


def load_proxy_metrics(dataset_name):
    """Load proxy metrics for a dataset, return DataFrame."""
    pm_path = f'coreset_indices/{dataset_name}/proxy_metrics.json'
    if not os.path.exists(pm_path):
        print(f"  WARNING: {pm_path} not found")
        return pd.DataFrame()

    with open(pm_path) as f:
        proxy_raw = json.load(f)

    rows = []
    for fname, metrics in proxy_raw.items():
        parsed = parse_proxy_filename(fname)
        if parsed is None:
            continue
        if parsed['distance'] != 'euclidean':
            continue

        row = {
            'dataset': dataset_name,
            'method': parsed['method'],
            'ratio': parsed['ratio'],
            'seed': parsed['seed'],
            **{k: v for k, v in metrics.items() if k in PROXY_METRICS},
        }
        rows.append(row)

    return pd.DataFrame(rows)


def compute_correlations(df, proxy_col, mae_col='MAE_mean'):
    """Compute Pearson and Spearman correlation."""
    valid = df[[proxy_col, mae_col]].dropna()
    if len(valid) < 5:
        return None
    r_pearson, p_pearson = stats.pearsonr(valid[proxy_col], valid[mae_col])
    r_spearman, p_spearman = stats.spearmanr(valid[proxy_col], valid[mae_col])
    return {
        'pearson_r': round(r_pearson, 4),
        'pearson_p': round(p_pearson, 6),
        'spearman_r': round(r_spearman, 4),
        'spearman_p': round(p_spearman, 6),
        'n': len(valid),
    }


# ── Load Phase C results ──
print("Loading Phase C results...")
df1 = pd.read_csv('experiments/result/phase_c_method_comparison.csv')
df2 = pd.read_csv('experiments/result/phase_c_extra_ratios.csv')
results = pd.concat([df1, df2], ignore_index=True)

# Replace old DCRNN (with CL) with new DCRNN (no CL) results
dcrnn_no_cl_path = 'experiments/result/phase_c_dcrnn_no_cl.csv'
if os.path.exists(dcrnn_no_cl_path):
    print("  Replacing DCRNN results with phase_c_dcrnn_no_cl (CL disabled)...")
    results = results[results.model != 'DCRNN']
    df_dcrnn = pd.read_csv(dcrnn_no_cl_path)
    results = pd.concat([results, df_dcrnn], ignore_index=True)

# Match the main paper grid: its final K-medoids cells come from the Phase D
# joint-PCA/L1 rerun, including DCRNN.  This replacement must happen after the
# DCRNN replacement above.
phase_d_path = 'experiments/result/phase_d_kmedoids_rerun.csv'
if os.path.exists(phase_d_path):
    print("  Replacing K-medoids results with phase_d_kmedoids_rerun...")
    results = results[
        results.coreset_selection_strategy != 'k_medoids'
    ]
    results = pd.concat([results, pd.read_csv(phase_d_path)], ignore_index=True)

results.rename(columns={
    'coreset_selection_strategy': 'method',
    'coreset_selection_ratio': 'ratio',
    'coreset_seed': 'seed',
}, inplace=True)
results = results[results.ratio < 1.0].copy()
results['ratio'] = results['ratio'].round(2)
print(f"  Results: {len(results)} rows (excl. full data)")

# ── Load proxy metrics ──
print("Loading proxy metrics...")
proxy_sb = load_proxy_metrics('SAN_BERNARDINO')
proxy_cc = load_proxy_metrics('CONTRA_COSTA')
proxy_all = pd.concat([proxy_sb, proxy_cc], ignore_index=True)
print(f"  Proxy metrics: {len(proxy_all)} entries (SB={len(proxy_sb)}, CC={len(proxy_cc)})")

# Filter PROXY_METRICS to only those actually present
available_metrics = [pm for pm in PROXY_METRICS if pm in proxy_all.columns]
print(f"  Available proxy metrics: {available_metrics}")

# ── Join ──
merged = results.merge(proxy_all, on=['dataset', 'method', 'ratio', 'seed'], how='inner')
print(f"  Merged: {len(merged)} rows")

# ── 1. Per-Model Correlation (main analysis) ──
print("\n" + "="*80)
print("PER-MODEL Spearman ρ (proxy metric vs MAE)")
print("="*80)

corr_rows = []
for model in sorted(merged.model.unique()):
    subset = merged[merged.model == model]
    for pm in available_metrics:
        c = compute_correlations(subset, pm)
        if c:
            corr_rows.append({
                'model': model,
                'level': 'per_model',
                'proxy_metric': pm,
                **c,
            })

corr_df = pd.DataFrame(corr_rows)

# Pivot for display
pivot = corr_df.pivot(index='proxy_metric', columns='model', values='spearman_r')
pivot = pivot.reindex(columns=sorted(merged.model.unique()))
pivot['mean_abs'] = pivot.abs().mean(axis=1)
pivot = pivot.sort_values('mean_abs', ascending=False)
print(pivot.to_string(float_format='{:+.3f}'.format))

# ── 2. Per-Model×Dataset ──
print("\n" + "="*80)
print("PER-MODEL×DATASET Spearman ρ")
print("="*80)

for model in sorted(merged.model.unique()):
    short = MODEL_SHORT.get(model, model)
    for ds in sorted(merged.dataset.unique()):
        subset = merged[(merged.model == model) & (merged.dataset == ds)]
        row_str = f"  {short:12s} × {ds:16s} (n={len(subset):3d}): "
        vals = []
        for pm in available_metrics:
            c = compute_correlations(subset, pm)
            if c:
                corr_rows.append({
                    'model': model, 'dataset': ds,
                    'level': 'per_model_dataset',
                    'proxy_metric': pm, **c,
                })
                sig = '*' if c['spearman_p'] < 0.05 else ' '
                vals.append(f"{pm}={c['spearman_r']:+.2f}{sig}")
        print(row_str + '  '.join(vals))

# ── 3. Within-setting: fixed (model, dataset, ratio), vary method ──
print("\n" + "="*80)
print("WITHIN-SETTING Mean Spearman ρ (fixed model+dataset+ratio, vary method)")
print("="*80)

ws_rows = []
for pm in available_metrics:
    setting_rhos = []
    for (model, ds, r), grp in merged.groupby(['model', 'dataset', 'ratio']):
        valid = grp[[pm, 'MAE_mean']].dropna()
        if len(valid) >= 6:  # 6 methods × seeds
            rho, p = stats.spearmanr(valid[pm], valid['MAE_mean'])
            setting_rhos.append({'model': model, 'dataset': ds, 'ratio': r,
                                 'rho': rho, 'p': p, 'n': len(valid)})

    if setting_rhos:
        wdf = pd.DataFrame(setting_rhos)
        n_sig = (wdf.p < 0.05).sum()
        ws_rows.append({
            'proxy_metric': pm,
            'mean_rho': wdf['rho'].mean(),
            'median_p': wdf['p'].median(),
            'n_settings': len(wdf),
            'n_significant': n_sig,
        })
        corr_rows.append({
            'model': 'ALL', 'level': 'within_setting',
            'proxy_metric': pm,
            'spearman_r': round(wdf['rho'].mean(), 4),
            'spearman_p': round(wdf['p'].median(), 6),
            'pearson_r': np.nan, 'pearson_p': np.nan,
            'n': len(wdf),
        })

ws_df = pd.DataFrame(ws_rows).sort_values('mean_rho', key=abs, ascending=False)
for _, row in ws_df.iterrows():
    print(f"  {row['proxy_metric']:25s}  mean ρ={row['mean_rho']:+.3f}  "
          f"median p={row['median_p']:.4f}  "
          f"sig={row['n_significant']}/{row['n_settings']}")

# ── Save all correlations ──
all_corr = pd.DataFrame(corr_rows)
all_corr.to_csv(f'{OUTPUT_DIR}/phase_c_proxy_correlations.csv', index=False)

# ── Plot: per-model scatter for top metrics ──
# Pick top 4 by mean |ρ| across models
top4 = pivot.drop(columns='mean_abs').index[:4].tolist()

model_colors = {
    'AGCRN': 'tab:blue', 'DCRNN': 'tab:orange',
    'STAEformer': 'tab:green', 'STGCNChebGraphConv': 'tab:red', 'STID': 'tab:purple',
}
method_markers = {
    'k_medoids': 'o', 'k_center': 's', 'graph_cut': '^',
    'random': 'D', 'stride': 'v', 'recent': 'P',
}

n_models = len(merged.model.unique())
fig, axes = plt.subplots(n_models, len(top4), figsize=(5*len(top4), 4*n_models))

for row_idx, model in enumerate(sorted(merged.model.unique())):
    model_data = merged[merged.model == model]
    short = MODEL_SHORT.get(model, model)

    for col_idx, pm in enumerate(top4):
        ax = axes[row_idx, col_idx]

        for _, r in model_data.iterrows():
            color = model_colors.get(r['model'], 'gray')
            marker = method_markers.get(r['method'], 'x')
            ax.scatter(r[pm], r['MAE_mean'],
                       color=color, marker=marker,
                       s=25, alpha=0.5, edgecolors='none')

        valid = model_data[[pm, 'MAE_mean']].dropna()
        if len(valid) >= 5:
            rho, p = stats.spearmanr(valid[pm], valid['MAE_mean'])
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            ax.set_title(f'{short}: ρ={rho:+.3f}{sig}', fontsize=9)
        if row_idx == n_models - 1:
            ax.set_xlabel(pm, fontsize=8)
        if col_idx == 0:
            ax.set_ylabel(f'{short}\nMAE', fontsize=8)
        ax.tick_params(labelsize=7)

from matplotlib.lines import Line2D
method_legend = [Line2D([0], [0], marker=mk, color='w', markerfacecolor='gray',
                         markersize=7, label=m) for m, mk in method_markers.items()]
fig.legend(handles=method_legend, loc='upper right', bbox_to_anchor=(0.99, 0.99),
           title='Method', fontsize=7, title_fontsize=8)

fig.suptitle('Phase C: Proxy Metrics vs MAE (per-model)', fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 0.95, 0.97])
plt.savefig(f'{OUTPUT_DIR}/phase_c_proxy_vs_mae.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved: {OUTPUT_DIR}/phase_c_proxy_vs_mae.png")
print(f"CSV saved: {OUTPUT_DIR}/phase_c_proxy_correlations.csv")
