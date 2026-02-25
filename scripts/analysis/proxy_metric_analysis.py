#!/usr/bin/env python
"""
Comprehensive proxy metric analysis (model-independent).

Analyzes:
  1. Seed stability — how stable are proxy metrics across seeds?
  2. Method ranking — which methods produce best coverage?
  3. Distance effect — does distance type affect coverage quality?
  4. Ratio effect — how does selection ratio affect coverage?
  5. Cross-metric correlation — do FL, Sinkhorn, redundancy agree?

Uses: coreset_indices/SAN_BERNARDINO/proxy_metrics.json (62 entries)
"""

import json
import os
import sys
import re
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# ── Load Data ────────────────────────────────────────────────────────────────
with open('coreset_indices/SAN_BERNARDINO/proxy_metrics.json') as f:
    raw = json.load(f)

rows = []
for fname, metrics in raw.items():
    stem = fname.replace('.json', '')
    # Parse: method_distance_ratio_seedN
    # Methods: k_center, k_medoids, graph_cut, random, stride, recent
    # Distances: euclidean, temporal, spatial, combined

    # Extract seed
    seed_match = re.search(r'seed(\d+)', stem)
    seed = int(seed_match.group(1)) if seed_match else None

    # Extract ratio (3-digit number: 030, 070, 100)
    ratio_match = re.search(r'_(\d{3})_seed', stem)
    ratio_str = ratio_match.group(1) if ratio_match else None
    ratio = int(ratio_str) / 100.0 if ratio_str else None

    # Extract method and distance
    prefix = stem[:stem.index(f'_{ratio_str}_seed')]
    # Known distances
    distances = ['euclidean', 'temporal', 'spatial', 'combined']
    distance = 'euclidean'  # default
    method = prefix
    for d in distances:
        if prefix.endswith(f'_{d}'):
            distance = d
            method = prefix[:-(len(d)+1)]
            break

    row = {
        'method': method,
        'distance': distance,
        'ratio': round(ratio, 1) if ratio else None,
        'seed': seed,
        'fl_objective': metrics['fl_objective'],
        'sinkhorn_divergence': metrics['sinkhorn_divergence'],
        'ot_cost': metrics['ot_cost'],
        'redundancy': metrics['redundancy'],
        'information_gain': metrics['information_gain'],
        'h_tod': metrics['h_tod'],
        'h_dow': metrics['h_dow'],
        'num_indices': metrics['num_indices'],
    }
    rows.append(row)

df = pd.DataFrame(rows)
print(f"Total entries: {len(df)}")
print(f"Methods: {sorted(df['method'].unique())}")
print(f"Distances: {sorted(df['distance'].unique())}")
print(f"Ratios: {sorted(df['ratio'].unique())}")
print(f"Seeds: {sorted(df['seed'].unique())}")

# Filter out full data (ratio=1.0) for meaningful comparisons
df_sel = df[df['ratio'] < 1.0].copy()
# Separate distance-dependent methods from baselines
smart_methods = ['k_center', 'k_medoids', 'graph_cut']
baseline_methods = ['random', 'stride', 'recent']
df_smart = df_sel[df_sel['method'].isin(smart_methods)]
df_base = df_sel[df_sel['method'].isin(baseline_methods)]

print(f"\nSmart methods: {len(df_smart)} rows")
print(f"Baseline methods: {len(df_base)} rows")

# ── 1. Seed Stability of Proxy Metrics ──────────────────────────────────────
print("\n" + "="*80)
print("1. SEED STABILITY OF PROXY METRICS")
print("="*80)
print("(How much do proxy metrics change when only the coreset seed differs?)\n")

proxy_cols = ['fl_objective', 'sinkhorn_divergence', 'redundancy', 'h_tod', 'h_dow']

# For smart methods: group by (method, distance, ratio)
seed_stability = []
for (method, dist, ratio), grp in df_smart.groupby(['method', 'distance', 'ratio']):
    if len(grp) == 2:
        row = {'method': method, 'distance': dist, 'ratio': ratio}
        for col in proxy_cols:
            vals = grp[col].values
            mean_val = np.mean(vals)
            abs_diff = abs(vals[0] - vals[1])
            pct_diff = abs_diff / abs(mean_val) * 100 if mean_val != 0 else 0
            row[f'{col}_pct_diff'] = pct_diff
        seed_stability.append(row)

df_seed = pd.DataFrame(seed_stability)

print("Mean % difference between seeds (smart methods):")
print("-" * 70)
for col in proxy_cols:
    pct_col = f'{col}_pct_diff'
    by_method = df_seed.groupby('method')[pct_col].agg(['mean', 'max'])
    print(f"\n  {col}:")
    for method in smart_methods:
        if method in by_method.index:
            print(f"    {method:12s}: mean={by_method.loc[method, 'mean']:6.2f}%, max={by_method.loc[method, 'max']:6.2f}%")

# Also for baselines
print("\nBaseline methods (seed diff):")
for (method, ratio), grp in df_base.groupby(['method', 'ratio']):
    if len(grp) == 2:
        for col in ['fl_objective', 'sinkhorn_divergence', 'h_tod']:
            vals = grp[col].values
            mean_val = np.mean(vals)
            pct = abs(vals[0] - vals[1]) / abs(mean_val) * 100 if mean_val != 0 else 0
            if col == 'fl_objective':
                print(f"  {method} ratio={ratio}: FL pct_diff={pct:.2f}%, ", end="")
            elif col == 'h_tod':
                print(f"H_tod pct_diff={pct:.2f}%")

# ── 2. Method Ranking by Proxy Metrics ──────────────────────────────────────
print("\n" + "="*80)
print("2. METHOD RANKING (averaged over seeds)")
print("="*80)

# Average over seeds first
df_avg = df_sel.groupby(['method', 'distance', 'ratio'])[proxy_cols].mean().reset_index()

# For ratio=0.3 (harder task)
print("\n--- Ratio = 0.3 ---")
r03 = df_avg[df_avg['ratio'] == 0.3]

# FL: higher is better (more coverage)
print("\nFL Objective (higher = better coverage):")
for _, row in r03.sort_values('fl_objective', ascending=False).head(10).iterrows():
    print(f"  {row['method']:12s} {row['distance']:10s}: {row['fl_objective']:10.1f}")

print("\n  Baselines:")
for _, row in r03[r03['method'].isin(baseline_methods)].sort_values('fl_objective', ascending=False).iterrows():
    print(f"  {row['method']:12s} {row['distance']:10s}: {row['fl_objective']:10.1f}")

# Sinkhorn: lower is better (closer to full distribution)
print("\nSinkhorn Divergence (lower = better distributional match):")
for _, row in r03.sort_values('sinkhorn_divergence', ascending=True).head(10).iterrows():
    print(f"  {row['method']:12s} {row['distance']:10s}: {row['sinkhorn_divergence']:.6f}")

print("\n  Baselines:")
for _, row in r03[r03['method'].isin(baseline_methods)].sort_values('sinkhorn_divergence', ascending=True).iterrows():
    print(f"  {row['method']:12s} {row['distance']:10s}: {row['sinkhorn_divergence']:.6f}")

# H_tod: higher is better (more temporal diversity)
print("\nTemporal Diversity H_tod (higher = better):")
for _, row in r03.sort_values('h_tod', ascending=False).head(10).iterrows():
    print(f"  {row['method']:12s} {row['distance']:10s}: {row['h_tod']:.4f}")

# Redundancy: lower is better
print("\nRedundancy (lower = less redundant):")
for _, row in r03.sort_values('redundancy', ascending=True).head(10).iterrows():
    print(f"  {row['method']:12s} {row['distance']:10s}: {row['redundancy']:12.1f}")

# --- Ratio = 0.7 ---
print("\n--- Ratio = 0.7 ---")
r07 = df_avg[df_avg['ratio'] == 0.7]

print("\nFL Objective (higher = better coverage):")
for _, row in r07.sort_values('fl_objective', ascending=False).head(10).iterrows():
    print(f"  {row['method']:12s} {row['distance']:10s}: {row['fl_objective']:10.1f}")

print("\nSinkhorn Divergence (lower = better):")
for _, row in r07.sort_values('sinkhorn_divergence', ascending=True).head(10).iterrows():
    print(f"  {row['method']:12s} {row['distance']:10s}: {row['sinkhorn_divergence']:.6f}")

# ── 3. Distance Type Effect ─────────────────────────────────────────────────
print("\n" + "="*80)
print("3. DISTANCE TYPE EFFECT")
print("="*80)
print("(Do different distance types lead to different coverage qualities?)\n")

# Average over seeds, then compare distances within each method
for method in smart_methods:
    print(f"\n  {method}:")
    sub = df_avg[(df_avg['method'] == method) & (df_avg['ratio'] == 0.3)]
    for _, row in sub.sort_values('sinkhorn_divergence').iterrows():
        print(f"    {row['distance']:10s}: SD={row['sinkhorn_divergence']:.6f}  FL={row['fl_objective']:10.1f}  H_tod={row['h_tod']:.4f}")

# ── 4. Cross-Metric Correlations ────────────────────────────────────────────
print("\n" + "="*80)
print("4. CROSS-METRIC CORRELATIONS")
print("="*80)
print("(Do FL, Sinkhorn, redundancy, H_tod agree on what makes a good coreset?)\n")

corr_cols = ['fl_objective', 'sinkhorn_divergence', 'redundancy', 'h_tod', 'h_dow']

# Use seed-averaged values for cleaner signal
df_corr = df_avg[df_avg['method'].isin(smart_methods)]

print("Pearson correlations (smart methods, ratio=0.3):")
r03_corr = df_corr[df_corr['ratio'] == 0.3][corr_cols]
corr_matrix = r03_corr.corr()
print(corr_matrix.round(3).to_string())

print("\nPearson correlations (all methods, both ratios):")
all_corr = df_avg[corr_cols]
corr_matrix2 = all_corr.corr()
print(corr_matrix2.round(3).to_string())

# ── 5. Smart Methods vs Baselines ───────────────────────────────────────────
print("\n" + "="*80)
print("5. SMART METHODS vs BASELINES (ratio=0.3)")
print("="*80)

r03_all = df_avg[df_avg['ratio'] == 0.3]

# Best smart method per metric
best_smart = {}
for col in ['sinkhorn_divergence', 'fl_objective', 'h_tod']:
    smart_sub = r03_all[r03_all['method'].isin(smart_methods)]
    base_sub = r03_all[r03_all['method'].isin(baseline_methods)]

    if col == 'sinkhorn_divergence':
        best = smart_sub.loc[smart_sub[col].idxmin()]
        worst_base = base_sub[col].max()
        best_base = base_sub[col].min()
        print(f"\n{col}:")
        print(f"  Best smart:    {best['method']}+{best['distance']} = {best[col]:.6f}")
        print(f"  Best baseline: {best_base:.6f}")
        print(f"  Worst baseline: {worst_base:.6f}")
    elif col == 'fl_objective':
        best = smart_sub.loc[smart_sub[col].idxmax()]
        best_base_val = base_sub[col].max()
        worst_base_val = base_sub[col].min()
        print(f"\n{col}:")
        print(f"  Best smart:    {best['method']}+{best['distance']} = {best[col]:.1f}")
        print(f"  Best baseline: {best_base_val:.1f}")
        print(f"  Worst baseline: {worst_base_val:.1f}")
    elif col == 'h_tod':
        best = smart_sub.loc[smart_sub[col].idxmax()]
        best_base_val = base_sub[col].max()
        print(f"\n{col}:")
        print(f"  Best smart:    {best['method']}+{best['distance']} = {best[col]:.4f}")
        print(f"  Best baseline: {best_base_val:.4f}")

# ── 6. Summary Table ────────────────────────────────────────────────────────
print("\n" + "="*80)
print("6. SUMMARY TABLE (seed-averaged, ratio=0.3)")
print("="*80)

r03_summary = df_avg[df_avg['ratio'] == 0.3][['method', 'distance', 'fl_objective', 'sinkhorn_divergence', 'redundancy', 'h_tod']].copy()
r03_summary = r03_summary.sort_values(['method', 'distance'])
print(f"\n{'method':12s} {'distance':10s} {'FL':>10s} {'Sinkhorn':>10s} {'Redundancy':>12s} {'H_tod':>6s}")
print("-" * 66)
for _, row in r03_summary.iterrows():
    print(f"{row['method']:12s} {row['distance']:10s} {row['fl_objective']:10.1f} {row['sinkhorn_divergence']:10.6f} {row['redundancy']:12.1f} {row['h_tod']:6.4f}")

print("\n" + "="*80)
print("7. KEY FINDINGS")
print("="*80)

# Automatically detect key findings
r03_smart = df_avg[(df_avg['ratio'] == 0.3) & (df_avg['method'].isin(smart_methods))]
r03_base = df_avg[(df_avg['ratio'] == 0.3) & (df_avg['method'].isin(baseline_methods))]

# Best/worst by Sinkhorn
best_sd = r03_smart.loc[r03_smart['sinkhorn_divergence'].idxmin()]
worst_sd = r03_smart.loc[r03_smart['sinkhorn_divergence'].idxmax()]
print(f"\n- Best Sinkhorn Divergence: {best_sd['method']}+{best_sd['distance']} ({best_sd['sinkhorn_divergence']:.6f})")
print(f"- Worst Sinkhorn Divergence: {worst_sd['method']}+{worst_sd['distance']} ({worst_sd['sinkhorn_divergence']:.6f})")

# Best/worst by FL
best_fl = r03_smart.loc[r03_smart['fl_objective'].idxmax()]
worst_fl = r03_smart.loc[r03_smart['fl_objective'].idxmin()]
print(f"- Best FL Objective: {best_fl['method']}+{best_fl['distance']} ({best_fl['fl_objective']:.1f})")
print(f"- Worst FL Objective: {worst_fl['method']}+{worst_fl['distance']} ({worst_fl['fl_objective']:.1f})")

# Do smart methods beat baselines?
base_best_sd = r03_base['sinkhorn_divergence'].min()
smart_better_count = (r03_smart['sinkhorn_divergence'] < base_best_sd).sum()
print(f"\n- Smart methods with better Sinkhorn than best baseline: {smart_better_count}/{len(r03_smart)}")

base_best_fl = r03_base['fl_objective'].max()
smart_fl_better = (r03_smart['fl_objective'] > base_best_fl).sum()
print(f"- Smart methods with better FL than best baseline: {smart_fl_better}/{len(r03_smart)}")

# Seed stability summary
print(f"\n- Proxy metric seed stability (mean %diff):")
for col in ['fl_objective', 'sinkhorn_divergence', 'h_tod']:
    mean_pct = df_seed[f'{col}_pct_diff'].mean()
    max_pct = df_seed[f'{col}_pct_diff'].max()
    print(f"    {col}: mean={mean_pct:.2f}%, max={max_pct:.2f}%")
