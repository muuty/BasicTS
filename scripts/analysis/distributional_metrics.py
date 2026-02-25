#!/usr/bin/env python
"""
Distributional distance analysis between coreset and full dataset.

Metrics computed:
  1. OT Cost (vanilla Sinkhorn) — already in proxy_metrics.json
  2. Sinkhorn Divergence (debiased) — already in proxy_metrics.json
  3. KL Divergence (temporal) — new: KL on time-of-day distribution
  4. KL Divergence (feature) — new: KL on PCA-binned feature distribution
  5. Jensen-Shannon Divergence — new: symmetric version of KL

Usage:
    conda activate cuda && python scripts/analysis/distributional_metrics.py
"""

import json
import os
import sys
import re
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.special import rel_entr
from pathlib import Path

os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append('.')

# ── 1. Load existing proxy metrics ──────────────────────────────────────────
print("Loading existing proxy metrics...")
with open('coreset_indices/SAN_BERNARDINO/proxy_metrics.json') as f:
    raw = json.load(f)

def parse_filename(fname):
    """Parse index filename into (method, distance, ratio, seed)."""
    stem = fname.replace('.json', '')
    seed_match = re.search(r'seed(\d+)', stem)
    seed = int(seed_match.group(1)) if seed_match else None
    ratio_match = re.search(r'_(\d{3})_seed', stem)
    ratio_str = ratio_match.group(1) if ratio_match else None
    ratio = int(ratio_str) / 100.0 if ratio_str else None
    prefix = stem[:stem.index(f'_{ratio_str}_seed')]
    distances = ['euclidean', 'temporal', 'spatial', 'combined']
    distance = 'euclidean'
    method = prefix
    for d in distances:
        if prefix.endswith(f'_{d}'):
            distance = d
            method = prefix[:-(len(d)+1)]
            break
    return method, distance, round(ratio, 1) if ratio else None, seed

# ── 2. Compute KL & JS divergence on temporal distributions ─────────────────
print("Computing KL/JS divergence on temporal distributions...")

# Load dataset to get total sample count and time info
from easytorch.config import import_config
from coreset.distance import extract_features

cfg_path = 'baselines/STGCN/SAN_BERNARDINO/SAN_BERNARDINO.py'
cfg = import_config(cfg_path, verbose=False)
from experiments.select_coreset import get_dataset_from_config
dataset = get_dataset_from_config(cfg)
dataset_size = len(dataset)
print(f"  Dataset size: {dataset_size}")

# Extract time-of-day for all samples
# Each sample has input shape [T, N, F]. Feature index 3 = tod, 4 = dow
# But STGCN uses FORWARD_FEATURES=[0,1,2], so we need raw data
# Time of day can be inferred from sample index
steps_per_day = 288  # 5-min intervals
tod_all = np.array([i % steps_per_day for i in range(dataset_size)])
dow_all = np.array([(i // steps_per_day) % 7 for i in range(dataset_size)])

# Full dataset temporal distribution (24 hour bins)
bins_per_hour = steps_per_day // 24  # 12 steps per hour
hour_all = tod_all // bins_per_hour
full_hour_dist = np.bincount(hour_all, minlength=24).astype(float)
full_hour_dist /= full_hour_dist.sum()

full_dow_dist = np.bincount(dow_all, minlength=7).astype(float)
full_dow_dist /= full_dow_dist.sum()


def kl_divergence(p, q, epsilon=1e-10):
    """KL(P || Q) with smoothing."""
    p = np.asarray(p, dtype=float) + epsilon
    q = np.asarray(q, dtype=float) + epsilon
    p /= p.sum()
    q /= q.sum()
    return np.sum(rel_entr(p, q))


def js_divergence(p, q, epsilon=1e-10):
    """Jensen-Shannon divergence (symmetric)."""
    p = np.asarray(p, dtype=float) + epsilon
    q = np.asarray(q, dtype=float) + epsilon
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * np.sum(rel_entr(p, m)) + 0.5 * np.sum(rel_entr(q, m))


# ── 3. Compute feature-space KL using PCA histograms ────────────────────────
print("Extracting features for PCA-based KL...")
model_config = cfg['MODEL']
inputs, targets = extract_features(dataset, model_config)

# Get combined features (most general)
from coreset.distance import get_features_by_type
features_combined = get_features_by_type(inputs, targets, 'combined')
print(f"  Combined features shape: {features_combined.shape}")

# PCA to low-dim
from sklearn.decomposition import PCA
pca = PCA(n_components=10, random_state=42)
features_pca = pca.fit_transform(features_combined)
print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum():.3f}")

# For KL in feature space: histogram each PCA component separately, then average KL
n_bins = 50

def feature_kl(indices, features_pca, n_bins=50):
    """Compute mean KL divergence across PCA components."""
    full = features_pca
    subset = features_pca[indices]
    kls = []
    for d in range(features_pca.shape[1]):
        # Use same bin edges for both
        edges = np.histogram_bin_edges(full[:, d], bins=n_bins)
        p_full, _ = np.histogram(full[:, d], bins=edges, density=True)
        p_sub, _ = np.histogram(subset[:, d], bins=edges, density=True)
        kls.append(kl_divergence(p_sub, p_full))
    return np.mean(kls)

def feature_js(indices, features_pca, n_bins=50):
    """Compute mean JS divergence across PCA components."""
    full = features_pca
    subset = features_pca[indices]
    jss = []
    for d in range(features_pca.shape[1]):
        edges = np.histogram_bin_edges(full[:, d], bins=n_bins)
        p_full, _ = np.histogram(full[:, d], bins=edges, density=True)
        p_sub, _ = np.histogram(subset[:, d], bins=edges, density=True)
        jss.append(js_divergence(p_sub, p_full))
    return np.mean(jss)


# ── 4. Compute all metrics for each index file ─────────────────────────────
print("\nComputing distributional metrics for all index files...")
index_dir = Path('coreset_indices/SAN_BERNARDINO')

rows = []
index_files = sorted(index_dir.glob("*.json"))
index_files = [f for f in index_files if f.name != "proxy_metrics.json"]

for i, idx_file in enumerate(index_files):
    fname = idx_file.name
    # Skip cosine files for now (separate analysis)
    if 'cosine' in fname:
        continue

    method, distance, ratio, seed = parse_filename(fname)
    if ratio is None or ratio > 0.99:
        # Skip full-data files for ratio=1.0 (trivial)
        if ratio == 1.0:
            continue

    with open(idx_file) as f:
        indices = json.load(f)

    # Temporal KL/JS
    hour_sub = (tod_all[indices] // bins_per_hour).astype(int)
    sub_hour_dist = np.bincount(hour_sub, minlength=24).astype(float)
    sub_hour_dist /= sub_hour_dist.sum()

    dow_sub = dow_all[indices]
    sub_dow_dist = np.bincount(dow_sub, minlength=7).astype(float)
    sub_dow_dist /= sub_dow_dist.sum()

    kl_tod = kl_divergence(sub_hour_dist, full_hour_dist)
    js_tod = js_divergence(sub_hour_dist, full_hour_dist)
    kl_dow = kl_divergence(sub_dow_dist, full_dow_dist)

    # Feature-space KL/JS
    kl_feat = feature_kl(indices, features_pca)
    js_feat = feature_js(indices, features_pca)

    # Existing metrics
    existing = raw.get(fname, {})

    row = {
        'method': method,
        'distance': distance,
        'ratio': ratio,
        'seed': seed,
        # Existing
        'ot_cost': existing.get('ot_cost', np.nan),
        'sinkhorn_div': existing.get('sinkhorn_divergence', np.nan),
        'fl_objective': existing.get('fl_objective', np.nan),
        'redundancy': existing.get('redundancy', np.nan),
        'h_tod': existing.get('h_tod', np.nan),
        # New temporal
        'kl_tod': kl_tod,
        'js_tod': js_tod,
        'kl_dow': kl_dow,
        # New feature-space
        'kl_feature': kl_feat,
        'js_feature': js_feat,
    }
    rows.append(row)
    if (i + 1) % 10 == 0:
        print(f"  [{i+1}/{len(index_files)}]")

df = pd.DataFrame(rows)
print(f"\nTotal entries: {len(df)}")

# ── 5. Analysis ─────────────────────────────────────────────────────────────
smart_methods = ['k_center', 'k_medoids', 'graph_cut']
baseline_methods = ['random', 'stride', 'recent']

# Average over seeds
dist_metrics = ['ot_cost', 'sinkhorn_div', 'kl_tod', 'js_tod', 'kl_dow', 'kl_feature', 'js_feature']
all_metrics = dist_metrics + ['fl_objective', 'redundancy', 'h_tod']
df_avg = df.groupby(['method', 'distance', 'ratio'])[all_metrics].mean().reset_index()

print("\n" + "="*80)
print("DISTRIBUTIONAL METRICS COMPARISON (ratio=0.3, seed-averaged)")
print("="*80)

r03 = df_avg[df_avg['ratio'] == 0.3].sort_values('js_feature')

print(f"\n{'method':12s} {'distance':10s} {'OT_cost':>9s} {'Sinkhorn':>9s} {'KL_tod':>8s} {'JS_tod':>8s} {'KL_feat':>8s} {'JS_feat':>8s}")
print("-" * 80)
for _, row in r03.iterrows():
    print(f"{row['method']:12s} {row['distance']:10s} "
          f"{row['ot_cost']:9.4f} {row['sinkhorn_div']:9.6f} "
          f"{row['kl_tod']:8.4f} {row['js_tod']:8.6f} "
          f"{row['kl_feature']:8.4f} {row['js_feature']:8.6f}")

print("\n" + "="*80)
print("DISTRIBUTIONAL METRICS COMPARISON (ratio=0.7, seed-averaged)")
print("="*80)

r07 = df_avg[df_avg['ratio'] == 0.7].sort_values('js_feature')

print(f"\n{'method':12s} {'distance':10s} {'OT_cost':>9s} {'Sinkhorn':>9s} {'KL_tod':>8s} {'JS_tod':>8s} {'KL_feat':>8s} {'JS_feat':>8s}")
print("-" * 80)
for _, row in r07.iterrows():
    print(f"{row['method']:12s} {row['distance']:10s} "
          f"{row['ot_cost']:9.4f} {row['sinkhorn_div']:9.6f} "
          f"{row['kl_tod']:8.4f} {row['js_tod']:8.6f} "
          f"{row['kl_feature']:8.4f} {row['js_feature']:8.6f}")

# ── 6. Seed Stability ──────────────────────────────────────────────────────
print("\n" + "="*80)
print("SEED STABILITY OF DISTRIBUTIONAL METRICS")
print("="*80)

for col in dist_metrics:
    diffs = []
    for (method, dist, ratio), grp in df.groupby(['method', 'distance', 'ratio']):
        if len(grp) == 2 and ratio < 1.0:
            vals = grp[col].values
            mean_val = np.mean(vals)
            pct = abs(vals[0] - vals[1]) / abs(mean_val) * 100 if abs(mean_val) > 1e-10 else 0
            diffs.append(pct)
    if diffs:
        print(f"  {col:16s}: mean_pct_diff={np.mean(diffs):6.2f}%, max={np.max(diffs):6.2f}%")

# ── 7. Cross-Metric Correlations ───────────────────────────────────────────
print("\n" + "="*80)
print("CROSS-METRIC CORRELATIONS (seed-averaged, ratio=0.3)")
print("="*80)

r03_all = df_avg[df_avg['ratio'] == 0.3]
corr_cols = ['ot_cost', 'sinkhorn_div', 'kl_tod', 'js_tod', 'kl_feature', 'js_feature', 'fl_objective', 'redundancy', 'h_tod']
corr = r03_all[corr_cols].corr()
print("\n" + corr.round(3).to_string())

# ── 8. Method Rankings Summary ──────────────────────────────────────────────
print("\n" + "="*80)
print("METHOD RANKINGS (ratio=0.3, lower=better for distributional metrics)")
print("="*80)

r03_smart = r03[r03['method'].isin(smart_methods)]
r03_base = r03[r03['method'].isin(baseline_methods)]

for metric in ['ot_cost', 'sinkhorn_div', 'kl_tod', 'js_tod', 'kl_feature', 'js_feature']:
    sorted_df = r03.sort_values(metric)
    best = sorted_df.iloc[0]
    print(f"\n  {metric}:")
    for rank, (_, row) in enumerate(sorted_df.iterrows(), 1):
        marker = " *" if row['method'] in baseline_methods else ""
        print(f"    {rank:2d}. {row['method']:12s}+{row['distance']:10s} = {row[metric]:.6f}{marker}")

# ── 9. Key Insight: Does graph_cut's low H_tod explain its MAE instability? ─
print("\n" + "="*80)
print("GRAPH_CUT TEMPORAL COVERAGE ANALYSIS")
print("="*80)

gc_r03 = df[(df['method'] == 'graph_cut') & (df['ratio'] == 0.3)]
km_r03 = df[(df['method'] == 'k_medoids') & (df['ratio'] == 0.3)]

print("\nGraph_cut vs K_medoids (ratio=0.3, all runs):")
print(f"  {'':20s} {'KL_tod':>8s} {'JS_tod':>8s} {'KL_feat':>8s} {'JS_feat':>8s} {'H_tod':>6s}")
for _, row in gc_r03.iterrows():
    print(f"  gc+{row['distance']:10s} s{row['seed']:3d}: "
          f"{row['kl_tod']:8.4f} {row['js_tod']:8.6f} "
          f"{row['kl_feature']:8.4f} {row['js_feature']:8.6f} {row['h_tod']:6.4f}")
print()
for _, row in km_r03.iterrows():
    print(f"  km+{row['distance']:10s} s{row['seed']:3d}: "
          f"{row['kl_tod']:8.4f} {row['js_tod']:8.6f} "
          f"{row['kl_feature']:8.4f} {row['js_feature']:8.6f} {row['h_tod']:6.4f}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("""
Distributional metrics tell us how well the coreset approximates the full dataset's distribution.

1. OT Cost & Sinkhorn: measure transport cost in feature space (geometric)
2. KL/JS Divergence (temporal): measure time-of-day distribution match
3. KL/JS Divergence (feature): measure feature-space distribution match

Key question: Which metric best predicts downstream MAE?
→ Requires Phase B deterministic experiments to answer definitively.
""")

# Save augmented metrics
output_path = 'experiments/result/analysis/distributional_metrics.csv'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)
print(f"Saved to {output_path}")
