#!/usr/bin/env python3
"""
1.5 Seed Stability Analysis

Quantifies seed sensitivity of k_medoids vs k_center vs graph_cut:
- Index Jaccard between seed=42 vs seed=123 for same (method, distance, ratio)
- MAE difference between seeds
- Graph cut RBF sigma analysis

Output: experiments/result/analysis/seed_stability.png
        experiments/result/analysis/seed_stability.csv
"""

import os
import sys
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

OUTPUT_DIR = 'experiments/result/analysis'
INDEX_DIR = 'coreset_indices/SAN_BERNARDINO'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Parse helpers ──
def parse_filename(fname):
    name = fname.replace('.json', '')
    m = re.search(r'_seed(\d+)$', name)
    if not m:
        return None
    seed = int(m.group(1))
    name = name[:m.start()]
    m = re.search(r'_(\d{3})$', name)
    if not m:
        return None
    ratio = int(m.group(1)) / 100.0
    name = name[:m.start()]
    methods = ['k_medoids', 'k_center', 'graph_cut', 'random', 'stride', 'recent', 'full']
    for method in methods:
        if name.startswith(method + '_'):
            distance = name[len(method) + 1:]
            return method, distance, ratio, seed
        elif name == method:
            return method, 'none', ratio, seed
    return None

def jaccard(a, b):
    if len(a) == 0 and len(b) == 0:
        return 1.0
    return len(a & b) / len(a | b)

# ── Load index files ──
index_data = {}
for f in sorted(os.listdir(INDEX_DIR)):
    if not f.endswith('.json'):
        continue
    parsed = parse_filename(f)
    if parsed is None:
        continue
    method, distance, ratio, seed = parsed
    with open(os.path.join(INDEX_DIR, f)) as fp:
        indices = set(json.load(fp))
    index_data[(method, distance, ratio, seed)] = indices

# ── Load MAE results ──
results = pd.read_csv('experiments/result/phase_a_distance_screening.csv')
stgcn = results[(results.model == 'STGCNChebGraphConv') & (results.MAE_mean > 0)].copy()

# ── Compute seed-pair metrics ──
rows = []
distances = ['euclidean', 'temporal', 'spatial', 'combined',
             'cosine_raw', 'cosine_temporal', 'cosine_spatial', 'cosine_combined']
methods = ['k_medoids', 'k_center', 'graph_cut']

for method in methods:
    for distance in distances:
        for ratio in [0.3, 0.7]:
            k42 = (method, distance, ratio, 42)
            k123 = (method, distance, ratio, 123)

            # Index Jaccard
            idx_jaccard = None
            if k42 in index_data and k123 in index_data:
                idx_jaccard = jaccard(index_data[k42], index_data[k123])

            # MAE difference
            mae_42 = stgcn[(stgcn.coreset_selection_strategy == method) &
                           (stgcn.coreset_distance_type == distance) &
                           (stgcn.coreset_selection_ratio == ratio) &
                           (stgcn.coreset_seed == 42)]['MAE_mean']
            mae_123 = stgcn[(stgcn.coreset_selection_strategy == method) &
                            (stgcn.coreset_distance_type == distance) &
                            (stgcn.coreset_selection_ratio == ratio) &
                            (stgcn.coreset_seed == 123)]['MAE_mean']

            mae_diff = None
            if len(mae_42) > 0 and len(mae_123) > 0:
                mae_diff = abs(mae_42.values[0] - mae_123.values[0])

            if idx_jaccard is not None or mae_diff is not None:
                rows.append({
                    'method': method,
                    'distance': distance,
                    'ratio': ratio,
                    'idx_jaccard': idx_jaccard,
                    'mae_diff': mae_diff,
                })

df = pd.DataFrame(rows)
df.to_csv(f'{OUTPUT_DIR}/seed_stability.csv', index=False)

# ── Summary by method ──
print("=== Seed Stability by Method ===")
for method in methods:
    sub = df[df.method == method]
    print(f"\n{method}:")
    jac = sub['idx_jaccard'].dropna()
    mae = sub['mae_diff'].dropna()
    if len(jac) > 0:
        print(f"  Index Jaccard: mean={jac.mean():.3f}, min={jac.min():.3f}, max={jac.max():.3f}")
    if len(mae) > 0:
        print(f"  MAE diff:      mean={mae.mean():.3f}, min={mae.min():.3f}, max={mae.max():.3f}")

# ── Summary by method × ratio ──
print("\n=== Seed Stability by Method × Ratio ===")
summary = df.groupby(['method', 'ratio']).agg(
    jaccard_mean=('idx_jaccard', 'mean'),
    jaccard_min=('idx_jaccard', 'min'),
    mae_diff_mean=('mae_diff', 'mean'),
    mae_diff_max=('mae_diff', 'max'),
    n=('idx_jaccard', 'count'),
).reset_index()
print(summary.to_string(index=False, float_format='{:.3f}'.format))

# ── Plot: 3-panel figure ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: Box plot of Index Jaccard by method
data_jac = [df[df.method == m]['idx_jaccard'].dropna().values for m in methods]
bp1 = axes[0].boxplot(data_jac, labels=methods, patch_artist=True,
                       boxprops=dict(facecolor='lightblue'))
axes[0].set_ylabel('Index Jaccard (seed42 vs seed123)')
axes[0].set_title('Index Reproducibility')
axes[0].axhline(0.8, color='red', linestyle='--', alpha=0.5, label='Good threshold')
axes[0].legend()

# Panel 2: Box plot of MAE diff by method
data_mae = [df[df.method == m]['mae_diff'].dropna().values for m in methods]
bp2 = axes[1].boxplot(data_mae, labels=methods, patch_artist=True,
                       boxprops=dict(facecolor='lightyellow'))
axes[1].set_ylabel('|MAE_seed42 - MAE_seed123|')
axes[1].set_title('MAE Sensitivity to Seed')

# Panel 3: Scatter of Jaccard vs MAE diff
colors = {'k_medoids': 'blue', 'k_center': 'orange', 'graph_cut': 'red'}
for method in methods:
    sub = df[(df.method == method) & df.idx_jaccard.notna() & df.mae_diff.notna()]
    if len(sub) > 0:
        axes[2].scatter(sub.idx_jaccard, sub.mae_diff, c=colors[method],
                       label=method, alpha=0.7, s=60)

axes[2].set_xlabel('Index Jaccard (higher = more stable index)')
axes[2].set_ylabel('|MAE diff| (lower = more stable performance)')
axes[2].set_title('Index Stability vs Performance Stability')
axes[2].legend()

plt.suptitle('Seed Sensitivity Analysis (STGCN, SAN_BERNARDINO)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/seed_stability.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved: {OUTPUT_DIR}/seed_stability.png")
