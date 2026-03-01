#!/usr/bin/env python
"""
Additional coverage metrics: Coverage Gap (per-hour) + MMD + TCG.

B1. Coverage Gap: per-hour relative deviation from full dataset distribution
B2. MMD: Maximum Mean Discrepancy in PCA feature space
B3. TCG (Tail Coverage Gap): 95th percentile of nearest coreset distance

Usage:
    conda activate cuda && python scripts/analysis/additional_coverage_metrics.py
"""

import json
import os
import sys
import re
import time
import numpy as np
import pandas as pd
from pathlib import Path

os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append('.')

# ── Setup ────────────────────────────────────────────────────────────────────
outdir = 'experiments/result/analysis'
index_dir = Path('coreset_indices/SAN_BERNARDINO')

# Load existing table
df = pd.read_csv(f'{outdir}/full_proxy_metrics_table.csv')

# Dataset temporal info
dataset_size = 14493
steps_per_day = 288
bins_per_hour = steps_per_day // 24
tod_all = np.array([i % steps_per_day for i in range(dataset_size)])
hour_all = (tod_all // bins_per_hour).astype(int)
full_hour_dist = np.bincount(hour_all, minlength=24).astype(float)
full_hour_dist_norm = full_hour_dist / full_hour_dist.sum()

# ── B1. Coverage Gap ────────────────────────────────────────────────────────
print("B1. Computing Coverage Gap (per-hour)...")

def compute_coverage_gap(indices):
    """Per-hour coverage gap: |p_coreset(h) - p_full(h)| / p_full(h)."""
    hour_sub = hour_all[indices]
    sub_dist = np.bincount(hour_sub, minlength=24).astype(float)
    sub_dist_norm = sub_dist / sub_dist.sum()
    gap = np.abs(sub_dist_norm - full_hour_dist_norm) / (full_hour_dist_norm + 1e-10)
    return {
        'coverage_gap_mean': float(np.mean(gap)),
        'coverage_gap_max': float(np.max(gap)),
        'coverage_gap_max_hour': int(np.argmax(gap)),
        'underrep_hours': int(np.sum(gap > 0.2)),  # hours with >20% deviation
    }

# ── B2. MMD ──────────────────────────────────────────────────────────────────
print("B2. Computing MMD (feature space)...")

from easytorch.config import import_config
from coreset.distance import extract_features, get_features_by_type
from sklearn.decomposition import PCA

cfg = import_config('baselines/STGCN/SAN_BERNARDINO/SAN_BERNARDINO.py', verbose=False)
from experiments.select_coreset import get_dataset_from_config
dataset = get_dataset_from_config(cfg)
model_config = cfg['MODEL']
inputs, targets = extract_features(dataset, model_config)
features_combined = get_features_by_type(inputs, targets, 'combined')

pca10 = PCA(n_components=10, random_state=42)
features_pca = pca10.fit_transform(features_combined)
print(f"  PCA(10): {features_combined.shape[1]} -> 10 dims, var_explained={pca10.explained_variance_ratio_.sum():.3f}")

# TCG uses same PCA-10 features (avoid OOM with PCA-50 on large coresets)

def compute_mmd_rbf(indices, features, sigma=None, subsample=2000):
    """Compute MMD^2 with RBF kernel between coreset and full dataset.

    MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
    where x ~ coreset, y ~ full dataset.
    """
    rng = np.random.RandomState(42)

    X = features[indices]
    Y = features

    # Subsample for efficiency
    if len(X) > subsample:
        X = X[rng.choice(len(X), subsample, replace=False)]
    if len(Y) > subsample:
        Y = Y[rng.choice(len(Y), subsample, replace=False)]

    # Compute sigma via median heuristic if not given
    if sigma is None:
        # Sample pairwise distances
        sample = features[rng.choice(len(features), min(1000, len(features)), replace=False)]
        dists = np.sum((sample[:, None] - sample[None, :]) ** 2, axis=-1)
        sigma = np.sqrt(np.median(dists[np.triu_indices(len(sample), k=1)]))
        if sigma == 0:
            sigma = 1.0

    gamma = 1.0 / (2 * sigma ** 2)

    # k(X, X)
    XX = np.sum((X[:, None] - X[None, :]) ** 2, axis=-1)
    kXX = np.exp(-gamma * XX)

    # k(Y, Y)
    YY = np.sum((Y[:, None] - Y[None, :]) ** 2, axis=-1)
    kYY = np.exp(-gamma * YY)

    # k(X, Y)
    XY = np.sum((X[:, None] - Y[None, :]) ** 2, axis=-1)
    kXY = np.exp(-gamma * XY)

    # MMD^2 (unbiased estimate)
    n, m = len(X), len(Y)
    mmd2 = (kXX.sum() - np.trace(kXX)) / (n * (n-1)) + \
           (kYY.sum() - np.trace(kYY)) / (m * (m-1)) - \
           2 * kXY.mean()

    return max(0.0, mmd2), sigma

# ── B3. TCG (Tail Coverage Gap) ───────────────────────────────────────────────
print("B3. Computing TCG (95th percentile nearest coreset distance)...")

def compute_tcg(indices, features, batch_size=200):
    """TCG_95 = Q_0.95 of {d_i} where d_i = min_{s in S} ||x_i - x_s||."""
    from scipy.spatial.distance import cdist
    coreset_feat = features[indices]
    N = len(features)
    dists = np.zeros(N)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        d = cdist(features[start:end], coreset_feat, metric='euclidean')
        dists[start:end] = d.min(axis=1)
    return {
        'tcg_95': float(np.percentile(dists, 95)),
        'tcg_99': float(np.percentile(dists, 99)),
        'tcg_mean': float(np.mean(dists)),
    }

# ── Compute for all index files ─────────────────────────────────────────────
print("\nComputing metrics for all index files...")

# Pre-compute sigma once
rng = np.random.RandomState(42)
sample = features_pca[rng.choice(len(features_pca), 1000, replace=False)]
dists = np.sum((sample[:, None] - sample[None, :]) ** 2, axis=-1)
global_sigma = np.sqrt(np.median(dists[np.triu_indices(len(sample), k=1)]))
print(f"  Global RBF sigma: {global_sigma:.4f}")

results = {}
for i, row in df.iterrows():
    method, dist, ratio, seed = row['method'], row['distance'], row['ratio'], int(row['seed'])
    ratio_str = f"{int(ratio * 100):03d}"
    fname = f"{method}_{dist}_{ratio_str}_seed{seed}.json"
    fpath = index_dir / fname

    if not fpath.exists():
        continue

    with open(fpath) as f:
        indices = json.load(f)

    # Coverage Gap
    cg = compute_coverage_gap(indices)

    # MMD
    t0 = time.time()
    mmd2, _ = compute_mmd_rbf(indices, features_pca, sigma=global_sigma)
    mmd_time = time.time() - t0

    # TCG (using PCA-10 features)
    tcg = compute_tcg(indices, features_pca)

    results[i] = {**cg, 'mmd_rbf': mmd2, **tcg}

    if (i + 1) % 10 == 0:
        print(f"  [{i+1}/{len(df)}] {fname}: gap_mean={cg['coverage_gap_mean']:.4f}, "
              f"TCG95={tcg['tcg_95']:.2f}, "
              f"MMD={mmd2:.6f} ({mmd_time:.1f}s)")

# Add to dataframe
for col in ['coverage_gap_mean', 'coverage_gap_max', 'coverage_gap_max_hour', 'underrep_hours', 'mmd_rbf', 'tcg_95', 'tcg_99', 'tcg_mean']:
    df[col] = df.index.map(lambda i: results.get(i, {}).get(col, np.nan))

# Save updated table
df.to_csv(f'{outdir}/full_proxy_metrics_table.csv', index=False, float_format='%.6f')
print(f"\nUpdated table with {len(results)} entries → {outdir}/full_proxy_metrics_table.csv")

# ── Analysis ─────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("COVERAGE GAP ANALYSIS (ratio=0.3)")
print("="*80)

r03 = df[df['ratio'] == 0.3]
r03_avg = r03.groupby(['method', 'distance'])[['coverage_gap_mean', 'coverage_gap_max', 'underrep_hours', 'mmd_rbf', 'tcg_95']].mean()
r03_avg = r03_avg.sort_values('coverage_gap_mean')

print(f"\n{'method':12s} {'distance':10s} {'gap_mean':>9s} {'gap_max':>8s} {'#under':>6s} {'MMD':>10s} {'TCG95':>8s}")
print("-" * 70)
for (method, dist), row in r03_avg.iterrows():
    print(f"{method:12s} {dist:10s} {row['coverage_gap_mean']:9.4f} {row['coverage_gap_max']:8.2f} "
          f"{row['underrep_hours']:6.1f} {row['mmd_rbf']:10.6f} {row['tcg_95']:8.2f}")

print("\n" + "="*80)
print("MMD RANKING (ratio=0.3, lower=better)")
print("="*80)
r03_mmd = r03_avg.sort_values('mmd_rbf')
for rank, ((method, dist), row) in enumerate(r03_mmd.iterrows(), 1):
    print(f"  {rank:2d}. {method:12s}+{dist:10s}: MMD={row['mmd_rbf']:.6f}")

# Cross-metric correlation with new metrics
print("\n" + "="*80)
print("CORRELATION WITH NEW METRICS (ratio=0.3)")
print("="*80)
corr_cols = ['ot_cost', 'sinkhorn_div', 'kl_tod', 'kl_feature', 'fl_objective',
             'redundancy', 'h_tod', 'coverage_gap_mean', 'mmd_rbf', 'tcg_95']
r03_corr = r03.groupby(['method', 'distance'])[corr_cols].mean()
corr = r03_corr.corr()
print("\nCorrelations with coverage_gap_mean:")
for col in corr_cols:
    if col != 'coverage_gap_mean':
        print(f"  {col:20s}: r={corr.loc['coverage_gap_mean', col]:.3f}")
print("\nCorrelations with mmd_rbf:")
for col in corr_cols:
    if col != 'mmd_rbf':
        print(f"  {col:20s}: r={corr.loc['mmd_rbf', col]:.3f}")
