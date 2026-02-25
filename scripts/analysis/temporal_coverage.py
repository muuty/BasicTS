#!/usr/bin/env python3
"""
1.3 Temporal Distribution Analysis + 1.4 Incident Proximity (lightweight)

Analyzes how evenly each coreset covers time-of-day and day-of-week.
Also checks incident-adjacent sample inclusion rate.

Output: experiments/result/analysis/temporal_coverage.png
        experiments/result/analysis/temporal_stats.csv
        experiments/result/analysis/incident_coverage.csv
"""

import os
import sys
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy

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

# ── Dataset time info ──
# SAN_BERNARDINO: 5-min intervals, 288 per day
# data_range (0, 24192) → 24192 / 288 = 84 days
# Training: 60% of 24192 = 14515 timesteps, samples start from index 0
# Each sample starts at timestep = sample_index (sliding window, stride=1)
# input_len=12, output_len=12 → sample covers timesteps [idx, idx+23]

INTERVAL_PER_DAY = 288  # 5-min intervals
INPUT_LEN = 12
OUTPUT_LEN = 12

# ── Compute time-of-day and day-of-week for each sample index ──
# Sample index i → starts at timestep i
# Time of day = (i % 288) / 288 * 24 → hour of day
# Day of week = (i // 288) % 7

def sample_tod(idx):
    """Time-of-day bin (0-23 hours) for sample starting at idx."""
    return (idx % INTERVAL_PER_DAY) * 24 // INTERVAL_PER_DAY

def sample_dow(idx):
    """Day-of-week (0-6) for sample starting at idx."""
    return (idx // INTERVAL_PER_DAY) % 7

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
        indices = json.load(fp)
    index_data[(method, distance, ratio, seed)] = indices

# ── Load incident metadata ──
incident_file = 'datasets/SAN_BERNARDINO/incident_metadata_2023.csv'
incident_slots = set()
if os.path.exists(incident_file):
    inc_df = pd.read_csv(incident_file)
    if 'incident_slot' in inc_df.columns:
        incident_slots = set(inc_df['incident_slot'].dropna().astype(int).values)
    elif 'input_start_slot' in inc_df.columns:
        # Sample is incident-adjacent if any of its timesteps overlap with incident
        for _, row in inc_df.iterrows():
            start = int(row.get('input_start_slot', 0))
            end = int(row.get('output_end_slot', start + INPUT_LEN + OUTPUT_LEN))
            for s in range(start, end + 1):
                incident_slots.add(s)
    print(f"Loaded {len(incident_slots)} incident-related timesteps")
else:
    print(f"WARNING: {incident_file} not found, skipping incident analysis")

# ── Compute temporal stats ──
distances = ['euclidean', 'temporal', 'spatial', 'combined',
             'cosine_raw', 'cosine_temporal', 'cosine_spatial', 'cosine_combined']
methods = ['k_medoids', 'k_center', 'graph_cut']
temporal_rows = []
incident_rows = []

for method in methods:
    for distance in distances:
        for ratio in [0.3, 0.7]:
            tod_counts_sum = np.zeros(24)
            dow_counts_sum = np.zeros(7)
            inc_coverage_sum = 0
            n_seeds = 0

            for seed in [42, 123]:
                key = (method, distance, ratio, seed)
                if key not in index_data:
                    continue

                indices = index_data[key]
                n_seeds += 1

                # Time-of-day distribution
                tod = [sample_tod(i) for i in indices]
                tod_hist, _ = np.histogram(tod, bins=np.arange(25))
                tod_counts_sum += tod_hist

                # Day-of-week distribution
                dow = [sample_dow(i) for i in indices]
                dow_hist, _ = np.histogram(dow, bins=np.arange(8))
                dow_counts_sum += dow_hist

                # Incident coverage
                if incident_slots:
                    sample_slots = set()
                    for idx in indices:
                        for t in range(idx, idx + INPUT_LEN + OUTPUT_LEN):
                            sample_slots.add(t)
                    inc_covered = len(incident_slots & sample_slots)
                    inc_coverage_sum += inc_covered / len(incident_slots) if incident_slots else 0

            if n_seeds == 0:
                continue

            tod_avg = tod_counts_sum / n_seeds
            dow_avg = dow_counts_sum / n_seeds

            # Entropy (higher = more uniform)
            tod_probs = tod_avg / tod_avg.sum() if tod_avg.sum() > 0 else np.ones(24) / 24
            dow_probs = dow_avg / dow_avg.sum() if dow_avg.sum() > 0 else np.ones(7) / 7
            h_tod = entropy(tod_probs) / np.log(24)  # normalized to [0,1]
            h_dow = entropy(dow_probs) / np.log(7)

            temporal_rows.append({
                'method': method, 'distance': distance, 'ratio': ratio,
                'h_tod': h_tod, 'h_dow': h_dow,
                'n_seeds': n_seeds,
            })

            if incident_slots:
                incident_rows.append({
                    'method': method, 'distance': distance, 'ratio': ratio,
                    'incident_coverage': inc_coverage_sum / n_seeds,
                })

temporal_df = pd.DataFrame(temporal_rows)
temporal_df.to_csv(f'{OUTPUT_DIR}/temporal_stats.csv', index=False)

# ── Summary ──
print("=== Temporal Entropy by Distance (avg over methods, seeds) ===")
dist_summary = temporal_df.groupby('distance').agg(
    h_tod_mean=('h_tod', 'mean'),
    h_dow_mean=('h_dow', 'mean'),
).sort_values('h_tod_mean', ascending=False)
print(dist_summary.to_string(float_format='{:.4f}'.format))

print("\n=== Temporal Entropy by Method ===")
method_summary = temporal_df.groupby('method').agg(
    h_tod_mean=('h_tod', 'mean'),
    h_dow_mean=('h_dow', 'mean'),
).sort_values('h_tod_mean', ascending=False)
print(method_summary.to_string(float_format='{:.4f}'.format))

# ── Incident coverage ──
if incident_rows:
    inc_df = pd.DataFrame(incident_rows)
    inc_df.to_csv(f'{OUTPUT_DIR}/incident_coverage.csv', index=False)
    print("\n=== Incident Coverage by Distance ===")
    inc_summary = inc_df.groupby('distance')['incident_coverage'].agg(['mean', 'std'])
    print(inc_summary.sort_values('mean', ascending=False).to_string(float_format='{:.4f}'.format))

# ── Plot: Temporal entropy comparison ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: h_tod by distance × method
for method in methods:
    sub = temporal_df[(temporal_df.method == method) & (temporal_df.ratio == 0.7)]
    sub = sub.sort_values('distance')
    axes[0].plot(sub.distance, sub.h_tod, 'o-', label=method, markersize=6)

axes[0].set_xlabel('Distance Type')
axes[0].set_ylabel('Normalized ToD Entropy (1=uniform)')
axes[0].set_title('Time-of-Day Coverage (ratio=0.7)')
axes[0].tick_params(axis='x', rotation=45)
axes[0].legend()
axes[0].set_ylim(0.9, 1.01)

# Panel 2: h_dow by distance × method
for method in methods:
    sub = temporal_df[(temporal_df.method == method) & (temporal_df.ratio == 0.7)]
    sub = sub.sort_values('distance')
    axes[1].plot(sub.distance, sub.h_dow, 'o-', label=method, markersize=6)

axes[1].set_xlabel('Distance Type')
axes[1].set_ylabel('Normalized DoW Entropy (1=uniform)')
axes[1].set_title('Day-of-Week Coverage (ratio=0.7)')
axes[1].tick_params(axis='x', rotation=45)
axes[1].legend()
axes[1].set_ylim(0.9, 1.01)

plt.suptitle('Temporal Coverage Analysis (SAN_BERNARDINO)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/temporal_coverage.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved: {OUTPUT_DIR}/temporal_coverage.png")
