#!/usr/bin/env python3
"""
Proxy metric diagnostics — defend against ratio-confound critique.

Three diagnostics:
  (1) Seed noise floor vs across-method spread per (model, dataset, ratio).
      If across-method spread <= seed noise, within-ratio task is ill-posed.
  (2) Per-ratio Spearman of quant_cost vs MAE.
      Shows whether the strong global correlation collapses at fixed ratio.
  (3) Family-binary AUC: can quant_cost classify
      {k_medoids, random, stride} vs {graph_cut, k_center, recent}
      within fixed (model, dataset, ratio)?

All inputs already exist; no GPU needed.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)

OUT = ROOT / 'experiments/result/analysis'
OUT.mkdir(parents=True, exist_ok=True)

PROXY_METRICS = [
    'ot_cost', 'sinkhorn_divergence', 'fl_objective', 'redundancy',
    'information_gain', 'h_tod', 'h_dow',
    'quantization_cost', 'quantization_median', 'quantization_max',
]
KNOWN_METHODS = {'k_medoids', 'k_center', 'graph_cut', 'random', 'stride', 'recent'}
PRESERVING = {'k_medoids', 'random', 'stride'}
AGGRESSIVE = {'k_center', 'graph_cut', 'recent'}


def parse_proxy_filename(fname):
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
    pm_path = f'coreset_indices/{dataset_name}/proxy_metrics.json'
    if not os.path.exists(pm_path):
        return pd.DataFrame()
    with open(pm_path) as f:
        proxy_raw = json.load(f)
    rows = []
    for fname, metrics in proxy_raw.items():
        parsed = parse_proxy_filename(fname)
        if parsed is None or parsed['distance'] != 'euclidean':
            continue
        row = {
            'dataset': dataset_name, 'method': parsed['method'],
            'ratio': parsed['ratio'], 'seed': parsed['seed'],
            **{k: v for k, v in metrics.items() if k in PROXY_METRICS},
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_merged():
    df1 = pd.read_csv('experiments/result/phase_c_method_comparison.csv')
    df2 = pd.read_csv('experiments/result/phase_c_extra_ratios.csv')
    results = pd.concat([df1, df2], ignore_index=True)
    dcrnn_no_cl = 'experiments/result/phase_c_dcrnn_no_cl.csv'
    if os.path.exists(dcrnn_no_cl):
        results = results[results.model != 'DCRNN']
        results = pd.concat([results, pd.read_csv(dcrnn_no_cl)], ignore_index=True)
    phase_d = 'experiments/result/phase_d_kmedoids_rerun.csv'
    if os.path.exists(phase_d):
        results = results[
            results.coreset_selection_strategy != 'k_medoids'
        ]
        results = pd.concat([results, pd.read_csv(phase_d)], ignore_index=True)
    results.rename(columns={
        'coreset_selection_strategy': 'method',
        'coreset_selection_ratio': 'ratio',
        'coreset_seed': 'seed',
    }, inplace=True)
    results = results[results.ratio < 1.0].copy()
    results['ratio'] = results['ratio'].round(2)
    proxy = pd.concat([load_proxy_metrics('SAN_BERNARDINO'),
                       load_proxy_metrics('CONTRA_COSTA')], ignore_index=True)
    merged = results.merge(proxy, on=['dataset', 'method', 'ratio', 'seed'], how='inner')
    return merged


# ── Build data ─────────────────────────────────────────────────────────────
merged = build_merged()
print(f"merged: {len(merged)} rows")
print(f"  models   : {sorted(merged.model.unique())}")
print(f"  datasets : {sorted(merged.dataset.unique())}")
print(f"  methods  : {sorted(merged.method.unique())}")
print(f"  ratios   : {sorted(merged.ratio.unique())}")
print(f"  seeds    : {sorted(merged.seed.unique())}")
print()

# ── Diagnostic 1: selection-instantiation vs across-method spread ──────────
print("=" * 78)
print("DIAGNOSTIC 1: Within-(model, dataset, ratio) noise floor analysis")
print("=" * 78)
print()
print("  σ_sel    = std of MAE across selection seeds for fixed (model, dataset, method, ratio)")
print("  σ_method = std of MAE across 6 methods (selection-seed averaged)")
print("  ratio    = σ_method / σ_sel   (descriptive separation ratio)")
print()

# Seed-level std per (model, dataset, method, ratio)
seed_std = (merged.groupby(['model', 'dataset', 'method', 'ratio'])['MAE_mean']
            .std(ddof=1).reset_index(name='sigma_selection'))

# Pooled seed std per (model, dataset, ratio): RMS over methods
seed_std['sigma_selection_sq'] = seed_std['sigma_selection'] ** 2
pooled_seed = (
    seed_std.groupby(['model', 'dataset', 'ratio'], as_index=False)
    .agg(sigma_selection_sq_mean=('sigma_selection_sq', 'mean'))
)
pooled_seed['sigma_selection_pooled'] = np.sqrt(
    pooled_seed.pop('sigma_selection_sq_mean')
)

# Method-level std per (model, dataset, ratio) on seed-mean MAE
method_mean = (merged.groupby(['model', 'dataset', 'method', 'ratio'])['MAE_mean']
               .mean().reset_index())
method_std = (method_mean.groupby(['model', 'dataset', 'ratio'])['MAE_mean']
              .std(ddof=1).reset_index(name='sigma_method'))

noise_tbl = method_std.merge(pooled_seed, on=['model', 'dataset', 'ratio'])
noise_tbl['separation_ratio'] = (
    noise_tbl['sigma_method'] / noise_tbl['sigma_selection_pooled']
)
# Temporary compatibility alias for downstream plotting code.
noise_tbl['snr'] = noise_tbl['separation_ratio']

# Per-ratio summary
print("  By ratio (median over 10 (model, dataset) combinations):")
print(f"  {'ratio':>6}  {'σ_method':>10}  {'σ_sel':>10}  {'sep.':>6}  {'#separated (>2)':>20}")
for r in sorted(noise_tbl.ratio.unique()):
    sub = noise_tbl[noise_tbl.ratio == r]
    well = (sub['snr'] > 2).sum()
    print(f"  {r:>6.2f}  "
          f"{sub['sigma_method'].median():>10.4f}  "
          f"{sub['sigma_selection_pooled'].median():>10.4f}  "
          f"{sub['separation_ratio'].median():>6.2f}  "
          f"{well:>10d}/{len(sub):<3d}")
print()

noise_tbl.to_csv(OUT / 'proxy_diag1_noise_floor.csv', index=False)
print(f"  → saved: {OUT}/proxy_diag1_noise_floor.csv")
print()

# ── Diagnostic 2: Per-ratio Spearman of quant_cost vs MAE ──────────────────
print("=" * 78)
print("DIAGNOSTIC 2: Within-ratio Spearman ρ (proxy vs MAE), by ratio")
print("=" * 78)
print()
print("  For each ratio: pool 6 methods × 3 seeds × 5 models × 2 datasets = 180 rows.")
print("  Then compute Spearman of proxy vs MAE *within each model* (since MAE scales differ),")
print("  and take median across models.")
print()

target_metrics = ['quantization_cost', 'quantization_median',
                  'ot_cost', 'sinkhorn_divergence', 'fl_objective',
                  'redundancy', 'h_tod', 'h_dow']

rows = []
for r in sorted(merged.ratio.unique()):
    sub_r = merged[merged.ratio == r]
    for pm in target_metrics:
        per_model_rhos = []
        for model in sub_r.model.unique():
            ssub = sub_r[sub_r.model == model][[pm, 'MAE_mean']].dropna()
            if len(ssub) < 5:
                continue
            rho, p = stats.spearmanr(ssub[pm], ssub['MAE_mean'])
            per_model_rhos.append(rho)
        if per_model_rhos:
            rows.append({
                'ratio': r, 'proxy_metric': pm,
                'mean_rho': float(np.mean(per_model_rhos)),
                'median_rho': float(np.median(per_model_rhos)),
                'min_rho': float(np.min(per_model_rhos)),
                'max_rho': float(np.max(per_model_rhos)),
                'n_models': len(per_model_rhos),
            })

per_ratio_df = pd.DataFrame(rows)
print(f"  {'ratio':>6}  {'metric':28s}  {'median ρ':>10}  {'[min, max]':>20}")
for r in sorted(per_ratio_df.ratio.unique()):
    for _, row in (per_ratio_df[per_ratio_df.ratio == r]
                   .sort_values('median_rho', key=lambda s: s.abs(),
                                ascending=False).iterrows()):
        rng = f"[{row['min_rho']:+.2f}, {row['max_rho']:+.2f}]"
        print(f"  {row['ratio']:>6.2f}  {row['proxy_metric']:28s}  "
              f"{row['median_rho']:>+10.3f}  {rng:>20}")
    print()

per_ratio_df.to_csv(OUT / 'proxy_diag2_per_ratio_spearman.csv', index=False)
print(f"  → saved: {OUT}/proxy_diag2_per_ratio_spearman.csv")
print()

# ── Diagnostic 3: Family AUC at fixed ratio ────────────────────────────────
print("=" * 78)
print("DIAGNOSTIC 3: Family-binary classification AUC at fixed ratio")
print("=" * 78)
print()
print("  Within each (model, dataset, ratio) cell, label rows by family:")
print(f"    preserving = {sorted(PRESERVING)}")
print(f"    aggressive = {sorted(AGGRESSIVE)}")
print("  Can a single proxy metric separate the two families?  (AUC by ratio)")
print("  Direction: higher proxy ↔ aggressive (label=1); flip if needed.")
print()

merged['family'] = merged['method'].apply(
    lambda m: 1 if m in AGGRESSIVE else (0 if m in PRESERVING else np.nan))

family_rows = []
for r in sorted(merged.ratio.unique()):
    for pm in target_metrics:
        per_cell = []
        for (model, ds), grp in merged[merged.ratio == r].groupby(['model', 'dataset']):
            sub = grp[[pm, 'family']].dropna()
            if sub['family'].nunique() < 2 or len(sub) < 6:
                continue
            try:
                auc = roc_auc_score(sub['family'], sub[pm])
            except ValueError:
                continue
            auc = max(auc, 1 - auc)
            per_cell.append(auc)
        if per_cell:
            family_rows.append({
                'ratio': r, 'proxy_metric': pm,
                'mean_auc': float(np.mean(per_cell)),
                'median_auc': float(np.median(per_cell)),
                'min_auc': float(np.min(per_cell)),
                'n_cells': len(per_cell),
            })

family_df = pd.DataFrame(family_rows)
print(f"  {'ratio':>6}  {'metric':28s}  {'median AUC':>10}  {'min':>6}  {'n':>4}")
for r in sorted(family_df.ratio.unique()):
    for _, row in (family_df[family_df.ratio == r]
                   .sort_values('median_auc', ascending=False).iterrows()):
        print(f"  {row['ratio']:>6.2f}  {row['proxy_metric']:28s}  "
              f"{row['median_auc']:>10.3f}  {row['min_auc']:>6.2f}  "
              f"{row['n_cells']:>4d}")
    print()

family_df.to_csv(OUT / 'proxy_diag3_family_auc.csv', index=False)
print(f"  → saved: {OUT}/proxy_diag3_family_auc.csv")
print()

# ── Headline summary ───────────────────────────────────────────────────────
print("=" * 78)
print("HEADLINE SUMMARY")
print("=" * 78)
qm_per_ratio = per_ratio_df[per_ratio_df.proxy_metric == 'quantization_median']
qm_family = family_df[family_df.proxy_metric == 'quantization_median']
print()
print("  quantization_median:")
print(f"    Per-ratio median Spearman ρ:  "
      f"{qm_per_ratio.set_index('ratio')['median_rho'].round(2).to_dict()}")
print(f"    Per-ratio median family AUC:  "
      f"{qm_family.set_index('ratio')['median_auc'].round(2).to_dict()}")
print()
snr_by_ratio = noise_tbl.groupby('ratio')['separation_ratio'].median().round(2).to_dict()
print(f"  separation ratio (σ_method / σ_sel) median by ratio: {snr_by_ratio}")
print()
