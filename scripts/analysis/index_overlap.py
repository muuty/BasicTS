#!/usr/bin/env python3
"""
1.2 Index Overlap Analysis (Jaccard Similarity)

Compares which samples are selected by different distance types and methods.
Key question: Is combined's selection an intersection, union, or something else
relative to temporal and spatial?

Output: experiments/result/analysis/jaccard_heatmap.png
        experiments/result/analysis/set_operations.csv
"""

import os
import sys
import json
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

OUTPUT_DIR = 'experiments/result/analysis'
INDEX_DIR = 'coreset_indices/SAN_BERNARDINO'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load all index files ──
def parse_filename(fname):
    """Parse index filename into (method, distance, ratio, seed)."""
    # e.g. k_medoids_combined_070_seed42.json
    name = fname.replace('.json', '')
    # Extract seed
    m = re.search(r'_seed(\d+)$', name)
    if not m:
        return None
    seed = int(m.group(1))
    name = name[:m.start()]

    # Extract ratio (3 digits)
    m = re.search(r'_(\d{3})$', name)
    if not m:
        return None
    ratio = int(m.group(1)) / 100.0
    name = name[:m.start()]

    # Method is first part before distance
    methods = ['k_medoids', 'k_center', 'graph_cut', 'random', 'stride', 'recent', 'full']
    for method in methods:
        if name.startswith(method + '_'):
            distance = name[len(method) + 1:]
            return method, distance, ratio, seed
        elif name == method:
            return method, 'none', ratio, seed
    return None

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

print(f"Loaded {len(index_data)} index files")

# ── Jaccard similarity ──
def jaccard(a, b):
    if len(a) == 0 and len(b) == 0:
        return 1.0
    return len(a & b) / len(a | b)

# ── Distance × Distance heatmap for each (method, ratio, seed) ──
distances_l2 = ['euclidean', 'temporal', 'spatial', 'combined']
distances_cos = ['cosine_raw', 'cosine_temporal', 'cosine_spatial', 'cosine_combined']
all_distances = distances_l2 + distances_cos

# Focus on k_medoids and k_center (most complete data)
for method in ['k_medoids', 'k_center', 'graph_cut']:
    for ratio in [0.3, 0.7]:
        # Average Jaccard across seeds
        jac_sum = np.zeros((len(all_distances), len(all_distances)))
        n_seeds = 0

        for seed in [42, 123]:
            available = []
            for d in all_distances:
                key = (method, d, ratio, seed)
                if key in index_data:
                    available.append(d)

            if len(available) < 3:
                continue
            n_seeds += 1

            for i, d1 in enumerate(all_distances):
                for j, d2 in enumerate(all_distances):
                    k1, k2 = (method, d1, ratio, seed), (method, d2, ratio, seed)
                    if k1 in index_data and k2 in index_data:
                        jac_sum[i, j] += jaccard(index_data[k1], index_data[k2])

        if n_seeds == 0:
            continue

        jac_avg = jac_sum / n_seeds

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(jac_avg, cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_xticks(range(len(all_distances)))
        ax.set_yticks(range(len(all_distances)))
        ax.set_xticklabels(all_distances, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(all_distances, fontsize=9)

        for i in range(len(all_distances)):
            for j in range(len(all_distances)):
                if jac_avg[i, j] > 0:
                    ax.text(j, i, f'{jac_avg[i, j]:.2f}', ha='center', va='center', fontsize=8)

        plt.colorbar(im, label='Jaccard Similarity')
        ax.set_title(f'Index Overlap: {method}, ratio={ratio} (avg over seeds)', fontsize=12)
        plt.tight_layout()
        fname = f'{OUTPUT_DIR}/jaccard_{method}_r{int(ratio*100):02d}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {fname}")

# ── Set operation analysis: combined vs temporal, spatial ──
print("\n=== Set Operations: combined vs temporal/spatial ===")
set_ops_rows = []

for method in ['k_medoids', 'k_center', 'graph_cut']:
    for ratio in [0.3, 0.7]:
        for seed in [42, 123]:
            keys = {
                'combined': (method, 'combined', ratio, seed),
                'temporal': (method, 'temporal', ratio, seed),
                'spatial': (method, 'spatial', ratio, seed),
            }
            if not all(k in index_data for k in keys.values()):
                continue

            c = index_data[keys['combined']]
            t = index_data[keys['temporal']]
            s = index_data[keys['spatial']]

            total = len(c)
            c_and_t = len(c & t)
            c_and_s = len(c & s)
            t_and_s = len(t & s)
            c_and_t_and_s = len(c & t & s)
            c_only = len(c - t - s)  # in combined but not in either
            t_union_s = t | s
            c_in_union = len(c & t_union_s)

            set_ops_rows.append({
                'method': method, 'ratio': ratio, 'seed': seed,
                'coreset_size': total,
                'combined∩temporal': c_and_t,
                'combined∩spatial': c_and_s,
                'temporal∩spatial': t_and_s,
                'combined∩temporal∩spatial': c_and_t_and_s,
                'combined_only': c_only,
                'combined_in_t∪s': c_in_union,
                'jaccard_c_t': jaccard(c, t),
                'jaccard_c_s': jaccard(c, s),
                'jaccard_t_s': jaccard(t, s),
                'pct_combined_unique': c_only / total * 100 if total > 0 else 0,
            })

set_ops_df = pd.DataFrame(set_ops_rows)
if not set_ops_df.empty:
    set_ops_df.to_csv(f'{OUTPUT_DIR}/set_operations.csv', index=False)
    print(set_ops_df.to_string(index=False, float_format='{:.3f}'.format))
else:
    print("No complete (combined, temporal, spatial) triplets found.")

# ── Method × Method overlap for same distance ──
print("\n=== Method Overlap (same distance, ratio, seed) ===")
method_overlap_rows = []
method_list = ['k_medoids', 'k_center', 'graph_cut']

for distance in all_distances:
    for ratio in [0.3, 0.7]:
        for seed in [42, 123]:
            for m1, m2 in combinations(method_list, 2):
                k1 = (m1, distance, ratio, seed)
                k2 = (m2, distance, ratio, seed)
                if k1 in index_data and k2 in index_data:
                    method_overlap_rows.append({
                        'distance': distance, 'ratio': ratio, 'seed': seed,
                        'method1': m1, 'method2': m2,
                        'jaccard': jaccard(index_data[k1], index_data[k2]),
                    })

method_overlap_df = pd.DataFrame(method_overlap_rows)
if not method_overlap_df.empty:
    summary = method_overlap_df.groupby(['method1', 'method2'])['jaccard'].agg(['mean', 'std', 'min', 'max'])
    print(summary.to_string(float_format='{:.3f}'.format))
    method_overlap_df.to_csv(f'{OUTPUT_DIR}/method_overlap.csv', index=False)

print("\nDone!")
