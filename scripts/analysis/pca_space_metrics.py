#!/usr/bin/env python
"""
Compute all proxy metrics in a unified PCA feature space.

Replaces the original pipeline where FL/OT/Sinkhorn/Redundancy used
per-distance-type matrices. Now everything uses PCA-10 of combined features.

Metrics:
  Feature-space (PCA-10 Euclidean):
    - fl_pca:           Facility Location objective
    - redundancy_pca:   Intra-coreset redundancy (RBF similarity)
    - ig_pca:           Information Gain (FL - λ·Redundancy)
    - ot_pca:           OT cost (Sinkhorn)
    - sinkhorn_pca:     Sinkhorn divergence (debiased)
    - mmd_rbf:          Maximum Mean Discrepancy (RBF kernel)
    - tcg_mean/95/99:   Tail Coverage Gap (nearest coreset distance)
  Temporal (hour-of-day):
    - h_hod:            Normalized entropy (24 bins)
    - kl_hod:           KL divergence vs full dataset
    - coverage_gap_hod_mean/max: Per-hour relative deviation
  Temporal (day-of-week):
    - h_dow:            Normalized entropy (7 bins)

Usage:
    conda activate cuda && python scripts/analysis/pca_space_metrics.py
"""

import json
import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist

os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append('.')

outdir = 'experiments/result/analysis'
index_dir = Path('coreset_indices/SAN_BERNARDINO')

# ── Load dataset & build PCA feature space ────────────────────────────────
print("Loading features...")
from easytorch.config import import_config
from coreset.distance import extract_features, get_features_by_type
from experiments.select_coreset import get_dataset_from_config
from sklearn.decomposition import PCA

cfg = import_config('baselines/STGCN/SAN_BERNARDINO/SAN_BERNARDINO.py', verbose=False)
dataset = get_dataset_from_config(cfg)
inputs, targets = extract_features(dataset, cfg['MODEL'])
features_combined = get_features_by_type(inputs, targets, 'combined')

PCA_DIM = 10
pca = PCA(n_components=PCA_DIM, random_state=42)
feat_pca = pca.fit_transform(features_combined)
print(f"  PCA: {features_combined.shape[1]} → {PCA_DIM} dims, "
      f"var_explained={pca.explained_variance_ratio_.sum():.3f}")

dataset_size = len(dataset)
steps_per_day = 288
samples_per_hour = steps_per_day // 24

# ── Pre-compute shared matrices ───────────────────────────────────────────
print("Building distance & similarity matrices in PCA space...")
t0 = time.time()

# Pairwise Euclidean distance (PCA-10, feasible: 14493 × 14493 × float32 ≈ 800MB)
# Use batched computation to avoid OOM
N = len(feat_pca)

# RBF sigma via median heuristic (sample-based)
rng = np.random.RandomState(42)
sample_idx = rng.choice(N, min(2000, N), replace=False)
sample = feat_pca[sample_idx]
sample_dists = cdist(sample, sample, metric='euclidean')
sigma = float(np.median(sample_dists[np.triu_indices(len(sample), k=1)]))
if sigma == 0:
    sigma = 1.0
print(f"  RBF sigma (median heuristic): {sigma:.4f}")

# Build similarity matrix in batches (avoid full N×N distance matrix)
print(f"  Building RBF similarity matrix ({N}×{N})...", end=" ", flush=True)
sim_pca = np.zeros((N, N), dtype=np.float32)
BATCH = 500
for start in range(0, N, BATCH):
    end = min(start + BATCH, N)
    d_batch = cdist(feat_pca[start:end], feat_pca, metric='sqeuclidean')
    sim_pca[start:end] = np.exp(-d_batch / (2 * sigma ** 2))
print(f"done ({time.time() - t0:.1f}s)")

# Full dataset temporal distributions
tod_all = np.array([(i % steps_per_day) // samples_per_hour for i in range(dataset_size)])
dow_all = np.array([(i // steps_per_day) % 7 for i in range(dataset_size)])
full_hod_dist = np.bincount(tod_all, minlength=24).astype(float)
full_hod_dist_norm = full_hod_dist / full_hod_dist.sum()


# ── Metric functions ──────────────────────────────────────────────────────

def compute_fl_objective(sim, indices, batch_size=2048):
    """Facility Location: Σ_i max_{j∈S} sim(i,j)."""
    S = np.array(indices)
    fl_sum = 0.0
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        fl_sum += sim[start:end][:, S].max(axis=1).sum()
    return float(fl_sum)


def compute_redundancy(sim, indices):
    """Intra-coreset redundancy: Σ_{i,j∈S} sim(i,j)."""
    S = np.array(indices)
    return float(sim[np.ix_(S, S)].sum())


def compute_ot_metrics(feat, indices, epsilon=0.1, max_iter=100, subsample=3000):
    """OT cost and Sinkhorn divergence in PCA space."""
    from coreset.ot_distance import _sinkhorn_cost

    X = feat[indices]
    Y = feat

    # Subsample for efficiency
    if len(X) > subsample:
        X = X[rng.choice(len(X), subsample, replace=False)]
    if len(Y) > subsample:
        Y = Y[rng.choice(len(Y), subsample, replace=False)]

    # No PCA needed - already in PCA space
    X_t = __import__('torch').from_numpy(X).float()
    Y_t = __import__('torch').from_numpy(Y).float()
    if __import__('torch').cuda.is_available():
        X_t, Y_t = X_t.cuda(), Y_t.cuda()

    ot_cost = _sinkhorn_cost(X_t, Y_t, epsilon, max_iter)
    ot_pp = _sinkhorn_cost(X_t, X_t, epsilon, max_iter)
    ot_qq = _sinkhorn_cost(Y_t, Y_t, epsilon, max_iter)
    sinkhorn_div = max(0.0, ot_cost - 0.5 * ot_pp - 0.5 * ot_qq)
    return ot_cost, sinkhorn_div


def compute_mmd(feat, indices, sigma_val, subsample=2000):
    """MMD^2 with RBF kernel."""
    X = feat[indices]
    Y = feat
    if len(X) > subsample:
        X = X[rng.choice(len(X), subsample, replace=False)]
    if len(Y) > subsample:
        Y = Y[rng.choice(len(Y), subsample, replace=False)]

    gamma = 1.0 / (2 * sigma_val ** 2)
    kXX = np.exp(-gamma * cdist(X, X, 'sqeuclidean'))
    kYY = np.exp(-gamma * cdist(Y, Y, 'sqeuclidean'))
    kXY = np.exp(-gamma * cdist(X, Y, 'sqeuclidean'))

    n, m = len(X), len(Y)
    mmd2 = ((kXX.sum() - np.trace(kXX)) / (n * (n - 1)) +
            (kYY.sum() - np.trace(kYY)) / (m * (m - 1)) -
            2 * kXY.mean())
    return max(0.0, mmd2)


def compute_tcg(feat, indices, batch_size=200):
    """Tail Coverage Gap: percentiles of nearest coreset distance."""
    coreset_feat = feat[indices]
    dists = np.zeros(len(feat))
    for start in range(0, len(feat), batch_size):
        end = min(start + batch_size, len(feat))
        d = cdist(feat[start:end], coreset_feat, metric='euclidean')
        dists[start:end] = d.min(axis=1)
    return {
        'tcg_mean': float(np.mean(dists)),
        'tcg_95': float(np.percentile(dists, 95)),
        'tcg_99': float(np.percentile(dists, 99)),
    }


def compute_temporal_metrics(indices):
    """All temporal metrics (hour-of-day and day-of-week)."""
    idx = np.array(indices)
    tod = (idx % steps_per_day) // samples_per_hour  # 0-23
    dow = (idx // steps_per_day) % 7  # 0-6

    # Entropy
    def norm_entropy(bins, n_bins):
        counts = np.bincount(bins, minlength=n_bins).astype(float)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        H = -np.sum(probs * np.log(probs))
        return float(H / np.log(n_bins)) if np.log(n_bins) > 0 else 0.0

    h_hod = norm_entropy(tod, 24)
    h_dow = norm_entropy(dow, 7)

    # KL divergence (hour-of-day)
    sub_hod_dist = np.bincount(tod, minlength=24).astype(float)
    sub_hod_norm = sub_hod_dist / sub_hod_dist.sum()
    eps = 1e-10
    kl_hod = float(np.sum(full_hod_dist_norm * np.log((full_hod_dist_norm + eps) / (sub_hod_norm + eps))))

    # Coverage gap (hour-of-day)
    gap = np.abs(sub_hod_norm - full_hod_dist_norm) / (full_hod_dist_norm + eps)

    return {
        'h_hod': round(h_hod, 4),
        'h_dow': round(h_dow, 4),
        'kl_hod': float(kl_hod),
        'coverage_gap_hod_mean': float(np.mean(gap)),
        'coverage_gap_hod_max': float(np.max(gap)),
    }


# ── Compute for all index files ──────────────────────────────────────────
print(f"\nComputing metrics for all index files in {index_dir}...")

index_files = sorted(index_dir.glob("*.json"))
index_files = [f for f in index_files if f.name != "proxy_metrics.json"]
print(f"  Found {len(index_files)} index files")

rows = []
for i, fpath in enumerate(index_files, 1):
    name = fpath.stem
    # Parse: method_distance_ratio_seed
    parts = name.split('_')

    # Find distance type
    KNOWN = {'euclidean', 'temporal', 'spatial', 'combined'}
    distance = next((p for p in parts if p in KNOWN), None)
    if distance is None:
        continue  # skip cosine variants for now

    # Parse method, ratio, seed
    dist_idx = parts.index(distance)
    method = '_'.join(parts[:dist_idx])
    ratio_str = parts[dist_idx + 1]  # e.g. "030"
    seed = int(parts[dist_idx + 2].replace('seed', ''))
    ratio = int(ratio_str) / 100.0

    with open(fpath) as f:
        indices = json.load(f)

    t0 = time.time()

    # Feature-space metrics (PCA)
    fl = compute_fl_objective(sim_pca, indices)
    red = compute_redundancy(sim_pca, indices)
    ig = fl - red
    ot_cost, sinkhorn = compute_ot_metrics(feat_pca, indices)
    mmd = compute_mmd(feat_pca, indices, sigma)
    tcg = compute_tcg(feat_pca, indices)

    # Temporal metrics
    temp = compute_temporal_metrics(indices)

    elapsed = time.time() - t0

    row = {
        'method': method, 'distance': distance, 'ratio': ratio, 'seed': seed,
        'num_indices': len(indices),
        # Feature-space (PCA)
        'fl_pca': fl, 'redundancy_pca': red, 'ig_pca': ig,
        'ot_pca': ot_cost, 'sinkhorn_pca': sinkhorn,
        'mmd_rbf': mmd,
        **tcg,
        # Temporal
        **temp,
    }
    rows.append(row)

    if i % 10 == 0 or i == len(index_files):
        print(f"  [{i}/{len(index_files)}] {name}: "
              f"FL={fl:.1f} SD={sinkhorn:.4f} TCG={tcg['tcg_mean']:.2f} ({elapsed:.1f}s)")

df = pd.DataFrame(rows)
out_path = f'{outdir}/pca_space_metrics.csv'
df.to_csv(out_path, index=False, float_format='%.6f')
print(f"\nSaved {len(df)} rows → {out_path}")

# ── Quick comparison: PCA metrics vs MAE ─────────────────────────────────
mae_df = pd.read_csv(f'{outdir}/full_metrics_with_mae.csv')
merged = df.merge(mae_df[['method', 'distance', 'ratio', 'seed', 'MAE_mean']],
                  on=['method', 'distance', 'ratio', 'seed'], how='inner')
merged = merged.dropna(subset=['MAE_mean'])

if len(merged) > 10:
    from scipy.stats import spearmanr

    print(f"\n{'='*70}")
    print(f"PCA-SPACE METRICS vs MAE (N={len(merged)})")
    print(f"{'='*70}")

    metric_cols = [c for c in df.columns
                   if c not in ['method', 'distance', 'ratio', 'seed', 'num_indices']]

    results = []
    for col in metric_cols:
        valid = merged[[col, 'MAE_mean']].dropna()
        if len(valid) < 5:
            continue
        rho, p = spearmanr(valid[col], valid['MAE_mean'])
        results.append((col, rho, p))

    results.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"\n{'Metric':25s} {'ρ':>8s} {'p':>8s}  {'|ρ|':>5s}")
    print('-' * 55)
    for col, rho, p in results:
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f'{col:25s} {rho:8.3f} {p:8.4f}  {abs(rho):5.3f} {sig}')

    # Compare with original metrics
    print(f"\n{'='*70}")
    print("COMPARISON: Original distance vs PCA space")
    print(f"{'='*70}")

    orig = pd.read_csv(f'{outdir}/full_proxy_metrics_table.csv')
    orig_merged = orig.merge(mae_df[['method', 'distance', 'ratio', 'seed', 'MAE_mean']],
                             on=['method', 'distance', 'ratio', 'seed'], how='inner')
    orig_merged = orig_merged.dropna(subset=['MAE_mean'])

    pairs = [
        ('fl_objective', 'fl_pca'),
        ('redundancy', 'redundancy_pca'),
        ('information_gain', 'ig_pca'),
        ('ot_cost', 'ot_pca'),
        ('sinkhorn_div', 'sinkhorn_pca'),
    ]
    print(f"\n{'Original':25s} {'ρ_orig':>8s}  {'PCA version':25s} {'ρ_pca':>8s}  {'Δ':>6s}")
    print('-' * 80)
    for orig_col, pca_col in pairs:
        if orig_col in orig_merged.columns:
            rho_o, _ = spearmanr(orig_merged[orig_col], orig_merged['MAE_mean'])
        else:
            rho_o = float('nan')
        rho_p, _ = spearmanr(merged[pca_col], merged['MAE_mean'])
        delta = abs(rho_p) - abs(rho_o)
        print(f'{orig_col:25s} {rho_o:8.3f}  {pca_col:25s} {rho_p:8.3f}  {delta:+6.3f}')
