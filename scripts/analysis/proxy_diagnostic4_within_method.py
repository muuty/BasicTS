#!/usr/bin/env python3
"""
Diagnostic 4 — practitioner use case:
  "I will use method M. How small a ratio can I go down to before MAE degrades?"

For each (model, dataset, method) cell, compute Spearman of (proxy, MAE) across
{ratio × seed}. A reliable proxy should track MAE as the practitioner sweeps
ratio with a fixed method.
"""
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore', category=stats.ConstantInputWarning)

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT = ROOT / 'experiments/result/analysis'

PROXY_METRICS = [
    'ot_cost', 'sinkhorn_divergence', 'fl_objective', 'redundancy',
    'information_gain', 'h_tod', 'h_dow',
    'quantization_cost', 'quantization_median', 'quantization_max',
]
KNOWN_METHODS = {'k_medoids', 'k_center', 'graph_cut', 'random', 'stride', 'recent'}


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


def load_proxy(ds):
    p = f'coreset_indices/{ds}/proxy_metrics.json'
    if not os.path.exists(p):
        return pd.DataFrame()
    with open(p) as f:
        raw = json.load(f)
    rows = []
    for fname, metrics in raw.items():
        parsed = parse_proxy_filename(fname)
        if parsed is None or parsed['distance'] != 'euclidean':
            continue
        rows.append({'dataset': ds, **{k: parsed[k] for k in ['method', 'ratio', 'seed']},
                     **{k: v for k, v in metrics.items() if k in PROXY_METRICS}})
    return pd.DataFrame(rows)


def build_merged():
    df1 = pd.read_csv('experiments/result/phase_c_method_comparison.csv')
    df2 = pd.read_csv('experiments/result/phase_c_extra_ratios.csv')
    results = pd.concat([df1, df2], ignore_index=True)
    no_cl = 'experiments/result/phase_c_dcrnn_no_cl.csv'
    if os.path.exists(no_cl):
        results = results[results.model != 'DCRNN']
        results = pd.concat([results, pd.read_csv(no_cl)], ignore_index=True)
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
    proxy = pd.concat([load_proxy('SAN_BERNARDINO'), load_proxy('CONTRA_COSTA')],
                      ignore_index=True)
    return results.merge(proxy, on=['dataset', 'method', 'ratio', 'seed'], how='inner')


merged = build_merged()
print(f"merged: {len(merged)} rows")

target = ['quantization_cost', 'quantization_median',
          'ot_cost', 'sinkhorn_divergence', 'fl_objective',
          'redundancy', 'h_tod', 'h_dow']

# ── Per (model, dataset, method) Spearman of proxy vs MAE across {ratio × seed} ──
cell_rows = []
for (model, ds, method), grp in merged.groupby(['model', 'dataset', 'method']):
    if grp['ratio'].nunique() < 3:
        continue
    for pm in target:
        sub = grp[[pm, 'MAE_mean']].dropna()
        if len(sub) < 5 or sub[pm].nunique() < 3:
            continue
        rho, p = stats.spearmanr(sub[pm], sub['MAE_mean'])
        cell_rows.append({
            'model': model, 'dataset': ds, 'method': method, 'proxy_metric': pm,
            'spearman': float(rho), 'p': float(p), 'n': len(sub),
        })

cells = pd.DataFrame(cell_rows)
cells['abs_rho'] = cells['spearman'].abs()
cells.to_csv(OUT / 'proxy_diag4_within_method.csv', index=False)
print(f"  → saved cells: {OUT}/proxy_diag4_within_method.csv  ({len(cells)} rows)")
print()

# ── Aggregate across the 60 cells (5 models × 2 datasets × 6 methods) ──
print("=" * 78)
print("DIAGNOSTIC 4: Within-(model, dataset, method), proxy vs MAE across {ratio × seed}")
print("=" * 78)
print()
print("  Practitioner use case: 'I picked a method; can the proxy tell me how low to go on ratio?'")
print(f"  Each cell = ~15 points (5 ratios × 3 seeds).  Total cells per proxy: up to 60.")
print()

print(f"  {'metric':28s}  {'median |ρ|':>10}  {'mean |ρ|':>9}  "
      f"{'%|ρ|>0.7':>9}  {'%|ρ|>0.9':>9}  {'n':>4}")
agg_rows = []
for pm in target:
    sub = cells[cells['proxy_metric'] == pm]
    if len(sub) == 0:
        continue
    row = {
        'proxy_metric': pm,
        'median_abs_rho': float(sub['abs_rho'].median()),
        'mean_abs_rho': float(sub['abs_rho'].mean()),
        'pct_strong': float((sub['abs_rho'] > 0.7).mean() * 100),
        'pct_near_perfect': float((sub['abs_rho'] > 0.9).mean() * 100),
        'n_cells': len(sub),
    }
    agg_rows.append(row)

agg = pd.DataFrame(agg_rows).sort_values('median_abs_rho', ascending=False)
for _, r in agg.iterrows():
    print(f"  {r['proxy_metric']:28s}  {r['median_abs_rho']:>10.3f}  "
          f"{r['mean_abs_rho']:>9.3f}  {r['pct_strong']:>8.1f}%  "
          f"{r['pct_near_perfect']:>8.1f}%  {r['n_cells']:>4d}")
agg.to_csv(OUT / 'proxy_diag4_within_method_agg.csv', index=False)
print()
print(f"  → saved aggregate: {OUT}/proxy_diag4_within_method_agg.csv")
print()

# ── Breakdown by method (does the proxy work for some methods but not others?) ──
print("=" * 78)
print("DIAGNOSTIC 4b: Quant_cost reliability by method (median |ρ| across 10 (model, dataset) cells)")
print("=" * 78)
print()
qc_only = cells[cells['proxy_metric'] == 'quantization_cost']
print(f"  {'method':12s}  {'median |ρ|':>10}  {'mean |ρ|':>9}  "
      f"{'min':>6}  {'max':>6}  {'n':>4}")
by_method = (qc_only.groupby('method')['abs_rho']
             .agg(['median', 'mean', 'min', 'max', 'count'])
             .sort_values('median', ascending=False))
for method, row in by_method.iterrows():
    print(f"  {method:12s}  {row['median']:>10.3f}  {row['mean']:>9.3f}  "
          f"{row['min']:>6.2f}  {row['max']:>6.2f}  {int(row['count']):>4d}")
print()

# ── Headline ──
print("=" * 78)
print("HEADLINE")
print("=" * 78)
qc_med = agg[agg.proxy_metric == 'quantization_cost'].iloc[0]
next_best = agg[agg.proxy_metric != 'quantization_cost'].iloc[0]
print(f"  quantization_cost: median |ρ| = {qc_med['median_abs_rho']:.2f}, "
      f"{qc_med['pct_strong']:.0f}% of cells have |ρ| > 0.7")
print(f"  next best ({next_best['proxy_metric']}): median |ρ| = {next_best['median_abs_rho']:.2f}, "
      f"{next_best['pct_strong']:.0f}% of cells with |ρ| > 0.7")
