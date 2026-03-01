#!/usr/bin/env python
"""
Per-setting method effect analysis.

Controls for (model, distance, ratio) and isolates method effect.
Outputs actual values (not rankings) for all proxy metrics + MAE.

Usage:
    conda activate cuda && python scripts/analysis/per_setting_method_analysis.py
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

outdir = Path('experiments/result/analysis')

# ── Step 1: Build unified proxy table ─────────────────────────────────────

# Base metrics (original space)
base = pd.read_csv(outdir / 'full_metrics_with_mae.csv')
base_cols = ['method', 'distance', 'ratio', 'seed',
             'fl_objective', 'redundancy', 'information_gain',
             'kl_feature', 'js_feature', 'mmd_rbf']
base = base[base_cols].copy()
base['ratio'] = base['ratio'].round(4)

# PCA-space metrics
pca = pd.read_csv(outdir / 'pca_space_metrics.csv')
pca_cols = ['method', 'distance', 'ratio', 'seed',
            'fl_pca', 'redundancy_pca', 'ig_pca', 'ot_pca', 'sinkhorn_pca',
            'tcg_mean', 'tcg_95', 'tcg_99',
            'h_hod', 'h_dow', 'kl_hod',
            'coverage_gap_hod_mean', 'coverage_gap_hod_max']
pca = pca[[c for c in pca_cols if c in pca.columns]].copy()
pca['ratio'] = pca['ratio'].round(4)

# Density-proportional metrics
dp = pd.read_csv(outdir / 'density_proportional_metrics.csv')
dp['ratio'] = dp['ratio'].round(4)

# Merge all proxy metrics
keys = ['method', 'distance', 'ratio', 'seed']
proxy = base.merge(pca, on=keys, how='outer')
proxy = proxy.merge(dp, on=keys, how='outer')

# Only smart methods
proxy = proxy[proxy['method'].isin(['k_medoids', 'k_center', 'graph_cut'])]
print(f"Unified proxy table: {len(proxy)} rows")

# ── Step 2: Merge Phase B MAE ─────────────────────────────────────────────

phase_b = pd.read_csv(outdir.parent / 'phase_b_deterministic.csv')
phase_b = phase_b[phase_b['model'].isin(['STGCNChebGraphConv', 'AGCRN'])]
phase_b['ratio'] = phase_b['coreset_selection_ratio'].round(4)

# Pivot to get MAE_stgcn, MAE_agcrn as separate columns
mae_wide = phase_b.pivot_table(
    index=['coreset_selection_strategy', 'coreset_distance_type', 'ratio', 'coreset_seed'],
    columns='model',
    values='MAE_mean',
).reset_index()
mae_wide.columns.name = None
mae_wide = mae_wide.rename(columns={
    'coreset_selection_strategy': 'method',
    'coreset_distance_type': 'distance',
    'coreset_seed': 'seed',
    'STGCNChebGraphConv': 'MAE_stgcn',
    'AGCRN': 'MAE_agcrn',
})

unified = proxy.merge(mae_wide, on=keys, how='inner')
print(f"Unified with MAE: {len(unified)} rows")
print(f"  MAE_stgcn non-null: {unified['MAE_stgcn'].notna().sum()}")
print(f"  MAE_agcrn non-null: {unified['MAE_agcrn'].notna().sum()}")

# Save unified table
unified.to_csv(outdir / 'unified_proxy_with_mae.csv', index=False)
print(f"  Saved: {outdir / 'unified_proxy_with_mae.csv'}")

# ── Step 3: Per-setting tables ─────────────────────────────────────────────

# Select key metrics to display (keep table readable)
display_metrics = [
    # Density-proportional
    'voronoi_cv', 'ess_ratio',
    # PCA-space coverage
    'tcg_mean', 'fl_pca', 'redundancy_pca',
    # Distributional
    'ot_pca', 'sinkhorn_pca',
    # Temporal
    'h_hod', 'h_dow', 'coverage_gap_hod_max',
    # Divergence
    'js_feature',
    # Original space (reference)
    'redundancy',
]

print(f"\n{'='*100}")
print("STEP 3: Per-Setting Method Comparison (seed-averaged values)")
print(f"{'='*100}")

for model_col, model_name in [('MAE_stgcn', 'STGCN'), ('MAE_agcrn', 'AGCRN')]:
    for ratio in [0.3, 0.7]:
        print(f"\n{'='*100}")
        print(f"  Model: {model_name}, Ratio: {ratio}")
        print(f"{'='*100}")

        for dist in ['euclidean', 'temporal', 'spatial', 'combined']:
            subset = unified[(unified['ratio'] == ratio) & (unified['distance'] == dist)]
            if len(subset) == 0:
                continue

            agg = subset.groupby('method').agg(
                **{f'{model_col}_mean': (model_col, 'mean')},
                **{f'{model_col}_std': (model_col, 'std')},
                **{m: (m, 'mean') for m in display_metrics if m in subset.columns},
            ).sort_values(f'{model_col}_mean')

            print(f"\n  --- distance={dist} ---")
            # Header
            header = f"  {'method':12s} {'MAE':>14s}"
            for m in display_metrics:
                if m in agg.columns:
                    header += f" {m:>12s}"
            print(header)
            print(f"  {'-'*12} {'-'*14}" + "".join(f" {'-'*12}" for m in display_metrics if m in agg.columns))

            for method in agg.index:
                mae_val = agg.loc[method, f'{model_col}_mean']
                mae_std = agg.loc[method, f'{model_col}_std']
                mae_str = f"{mae_val:.2f}±{mae_std:.2f}" if not pd.isna(mae_std) else f"{mae_val:.2f}"
                line = f"  {method:12s} {mae_str:>14s}"
                for m in display_metrics:
                    if m in agg.columns:
                        val = agg.loc[method, m]
                        if abs(val) > 1000:
                            line += f" {val:12.0f}"
                        elif abs(val) > 1:
                            line += f" {val:12.3f}"
                        else:
                            line += f" {val:12.4f}"
                print(line)

# ── Step 4: Within-setting Spearman ρ ─────────────────────────────────────

print(f"\n\n{'='*100}")
print("STEP 4: Within-Setting Spearman ρ (method effect only)")
print(f"{'='*100}")
print("(Each setting = fixed model×distance×ratio, n=6 points: 3 methods × 2 seeds)")

all_proxy_cols = [c for c in display_metrics if c in unified.columns]

for model_col, model_name in [('MAE_stgcn', 'STGCN'), ('MAE_agcrn', 'AGCRN')]:
    print(f"\n  === {model_name} ===")

    rho_records = {m: [] for m in all_proxy_cols}

    for ratio in [0.3, 0.7]:
        for dist in ['euclidean', 'temporal', 'spatial', 'combined']:
            subset = unified[(unified['ratio'] == ratio) & (unified['distance'] == dist)]
            if len(subset) < 4:
                continue

            for m in all_proxy_cols:
                valid = subset[[m, model_col]].dropna()
                if len(valid) >= 4:
                    r, p = stats.spearmanr(valid[m], valid[model_col])
                    if not np.isnan(r):
                        rho_records[m].append({'ratio': ratio, 'dist': dist, 'rho': r, 'p': p})

    # Summary: mean ρ across settings
    print(f"\n  {'metric':25s} {'mean_ρ':>8s} {'median_ρ':>9s} {'n_settings':>10s} {'sig_rate':>9s}")
    print(f"  {'-'*25} {'-'*8} {'-'*9} {'-'*10} {'-'*9}")

    summary_rows = []
    for m in all_proxy_cols:
        records = rho_records[m]
        if not records:
            continue
        rhos = [r['rho'] for r in records]
        sig_count = sum(1 for r in records if r['p'] < 0.1)
        summary_rows.append({
            'metric': m,
            'mean_rho': np.mean(rhos),
            'median_rho': np.median(rhos),
            'n': len(rhos),
            'sig_rate': sig_count / len(rhos),
        })

    summary_rows.sort(key=lambda x: x['mean_rho'])
    for row in summary_rows:
        print(f"  {row['metric']:25s} {row['mean_rho']:+8.4f} {row['median_rho']:+9.4f} {row['n']:10d} {row['sig_rate']:9.1%}")

# ── Step 4b: Cross-model consistency ──────────────────────────────────────

print(f"\n\n{'='*100}")
print("STEP 4b: Cross-Model Consistency")
print(f"{'='*100}")
print("(Does the same metric work well for both STGCN and AGCRN?)")

combined_summary = []
for m in all_proxy_cols:
    rhos_all = []
    for model_col in ['MAE_stgcn', 'MAE_agcrn']:
        for ratio in [0.3, 0.7]:
            for dist in ['euclidean', 'temporal', 'spatial', 'combined']:
                subset = unified[(unified['ratio'] == ratio) & (unified['distance'] == dist)]
                valid = subset[[m, model_col]].dropna()
                if len(valid) >= 4:
                    r, _ = stats.spearmanr(valid[m], valid[model_col])
                    if not np.isnan(r):
                        rhos_all.append(r)
    if rhos_all:
        combined_summary.append({
            'metric': m,
            'mean_rho': np.mean(rhos_all),
            'std_rho': np.std(rhos_all),
            'n': len(rhos_all),
        })

combined_summary.sort(key=lambda x: x['mean_rho'])
print(f"\n  {'metric':25s} {'mean_ρ':>8s} {'std_ρ':>8s} {'n':>4s}")
print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*4}")
for row in combined_summary:
    print(f"  {row['metric']:25s} {row['mean_rho']:+8.4f} {row['std_rho']:8.4f} {row['n']:4d}")
