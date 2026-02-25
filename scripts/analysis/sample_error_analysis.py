#!/usr/bin/env python3
"""
3.3 Per-Sample Error Analysis

Compares per-sample (per-timestep) MAE between best and worst coreset configurations.
Identifies which time-of-day patterns show the biggest quality gaps.

Output: experiments/result/analysis/sample_error_analysis.png
        experiments/result/analysis/sample_error_by_tod.csv
"""

import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

OUTPUT_DIR = 'experiments/result/analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Map checkpoint dirs to configs ──
CKPT_BASE = 'checkpoints/phase_a_distance_screening/STGCNChebGraphConv/xtraffic/SAN_BERNARDINO_100_12_12/1'

ckpt_configs = {}
for d in os.listdir(CKPT_BASE):
    cfg_path = os.path.join(CKPT_BASE, d, 'cfg.txt')
    if not os.path.exists(cfg_path):
        continue
    with open(cfg_path) as f:
        content = f.read()

    # Parse CORESET section
    config = {}
    in_coreset = False
    for line in content.split('\n'):
        if line.strip() == 'CORESET:':
            in_coreset = True
            continue
        if in_coreset:
            if line.startswith('  ') and ':' in line:
                key, val = line.strip().split(':', 1)
                config[key.strip()] = val.strip()
            else:
                break

    if config:
        key = (config.get('SELECTION_STRATEGY', ''),
               config.get('DISTANCE_TYPE', ''),
               float(config.get('SELECTION_RATIO', '0')),
               int(config.get('SEED', '0')))
        ckpt_configs[key] = d

print(f"Mapped {len(ckpt_configs)} checkpoint dirs to configs")

# ── Load Phase A results to identify best/worst ──
results = pd.read_csv('experiments/result/phase_a_distance_screening.csv')
stgcn = results[results.model == 'STGCNChebGraphConv'].sort_values('MAE_mean')

# Select comparison pairs
# 1. Best overall vs worst overall (ratio 0.7)
best_07 = stgcn[stgcn.coreset_selection_ratio == 0.7].iloc[0]
worst_07 = stgcn[stgcn.coreset_selection_ratio == 0.7].iloc[-1]

# 2. Best combined vs best euclidean (same method/ratio)
best_combined = stgcn[(stgcn.coreset_distance_type == 'combined') &
                      (stgcn.coreset_selection_ratio == 0.7)].iloc[0]
best_euclidean = stgcn[(stgcn.coreset_distance_type == 'euclidean') &
                       (stgcn.coreset_selection_ratio == 0.7)].iloc[0]

# 3. Best k_medoids vs best graph_cut (ratio 0.7)
best_kmedoids = stgcn[(stgcn.coreset_selection_strategy == 'k_medoids') &
                      (stgcn.coreset_selection_ratio == 0.7)].iloc[0]
best_graphcut = stgcn[(stgcn.coreset_selection_strategy == 'graph_cut') &
                      (stgcn.coreset_selection_ratio == 0.7)].iloc[0]

def load_test_results(row):
    """Load test_results.npz for a given experiment config."""
    key = (row['coreset_selection_strategy'],
           row['coreset_distance_type'],
           float(row['coreset_selection_ratio']),
           int(row['coreset_seed']))
    if key not in ckpt_configs:
        print(f"  WARNING: No checkpoint for {key}")
        return None
    d = ckpt_configs[key]
    npz_path = os.path.join(CKPT_BASE, d, 'test_results.npz')
    if not os.path.exists(npz_path):
        print(f"  WARNING: No test_results.npz in {d}")
        return None
    data = np.load(npz_path)
    return data

def compute_sample_mae(data):
    """Compute per-sample MAE: (n_samples,) averaging over time, nodes, features."""
    pred = data['prediction']  # (n_samples, T, N, F)
    tgt = data['target']
    return np.abs(pred - tgt).mean(axis=(1, 2, 3))

def compute_tod_mae(data, n_steps_per_day=288):
    """Compute MAE by time-of-day (5-min intervals, 288 per day)."""
    pred = data['prediction']
    tgt = data['target']
    mae_per_sample = np.abs(pred - tgt).mean(axis=(1, 2, 3))  # (n_samples,)

    # Test set starts at 80% of 24192 = 19354
    n_total = 24192
    test_start = int(n_total * 0.8)  # actually it's train+val = 0.6+0.2 = 0.8
    n_test = len(mae_per_sample)

    # Each sample's start timestep
    sample_starts = np.arange(test_start, test_start + n_test)
    tod = sample_starts % n_steps_per_day  # time of day index

    # Group by hour (aggregate 5-min into hourly)
    hour = (tod * 24) // n_steps_per_day

    tod_mae = pd.DataFrame({'hour': hour, 'mae': mae_per_sample})
    return tod_mae.groupby('hour')['mae'].mean()

# ── Load and compare ──
comparisons = [
    ('Best vs Worst (ratio=0.7)', best_07, worst_07),
    ('Combined vs Euclidean', best_combined, best_euclidean),
    ('k_medoids vs graph_cut', best_kmedoids, best_graphcut),
]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

all_tod_data = []

for idx, (title, row_a, row_b) in enumerate(comparisons):
    label_a = f"{row_a['coreset_selection_strategy']}_{row_a['coreset_distance_type']}_{row_a['coreset_selection_ratio']}_s{row_a['coreset_seed']}"
    label_b = f"{row_b['coreset_selection_strategy']}_{row_b['coreset_distance_type']}_{row_b['coreset_selection_ratio']}_s{row_b['coreset_seed']}"

    print(f"\n=== {title} ===")
    print(f"  A: {label_a} (MAE={row_a['MAE_mean']:.3f})")
    print(f"  B: {label_b} (MAE={row_b['MAE_mean']:.3f})")

    data_a = load_test_results(row_a)
    data_b = load_test_results(row_b)

    if data_a is None or data_b is None:
        continue

    mae_a = compute_sample_mae(data_a)
    mae_b = compute_sample_mae(data_b)

    # Top row: per-sample MAE difference histogram
    ax = axes[0, idx]
    diff = mae_b - mae_a  # positive = B is worse
    ax.hist(diff, bins=80, density=True, alpha=0.7, color='steelblue')
    ax.axvline(0, color='red', linestyle='--', alpha=0.7)
    ax.axvline(diff.mean(), color='orange', linestyle='-', alpha=0.7, label=f'mean={diff.mean():.3f}')
    pct_worse = (diff > 0).mean() * 100
    ax.set_title(f'{title}\n{pct_worse:.0f}% samples worse in B', fontsize=10)
    ax.set_xlabel('MAE(B) - MAE(A)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)

    # Bottom row: time-of-day MAE
    ax2 = axes[1, idx]
    tod_a = compute_tod_mae(data_a)
    tod_b = compute_tod_mae(data_b)
    ax2.plot(tod_a.index, tod_a.values, 'b-o', markersize=3, label=label_a.split('_')[0])
    ax2.plot(tod_b.index, tod_b.values, 'r-s', markersize=3, label=label_b.split('_')[0])
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('MAE')
    ax2.set_title(f'MAE by Time of Day', fontsize=10)
    ax2.legend(fontsize=7)
    ax2.set_xticks(range(0, 24, 3))

    # Save ToD data
    for h in tod_a.index:
        all_tod_data.append({
            'comparison': title,
            'config': label_a,
            'hour': h,
            'mae': tod_a[h],
        })
    for h in tod_b.index:
        all_tod_data.append({
            'comparison': title,
            'config': label_b,
            'hour': h,
            'mae': tod_b[h],
        })

fig.suptitle('Per-Sample Error Analysis (STGCN, SAN_BERNARDINO)',
            fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f'{OUTPUT_DIR}/sample_error_analysis.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved: {OUTPUT_DIR}/sample_error_analysis.png")

# Save ToD data
if all_tod_data:
    tod_df = pd.DataFrame(all_tod_data)
    tod_df.to_csv(f'{OUTPUT_DIR}/sample_error_by_tod.csv', index=False)
    print(f"Saved: {OUTPUT_DIR}/sample_error_by_tod.csv")

# ── Additional: incident vs non-incident error comparison ──
print("\n=== Incident vs Non-Incident Error ===")
# Load incident data
try:
    raw = np.load('datasets/SAN_BERNARDINO/data.npz')
    full_data = raw['data']  # (T, N, F)
    n_total = 24192
    test_start = int(n_total * 0.8)
    input_len = 12

    # Check if incident feature exists (index 3 or 4)
    if full_data.shape[2] >= 4:
        incident_feat = full_data[:, :, 3]  # incident indicator
        print(f"Incident feature shape: {incident_feat.shape}")

        for title, row_a, row_b in comparisons:
            data_a = load_test_results(row_a)
            data_b = load_test_results(row_b)
            if data_a is None or data_b is None:
                continue

            n_test = data_a['prediction'].shape[0]
            mae_a = compute_sample_mae(data_a)
            mae_b = compute_sample_mae(data_b)

            # For each test sample, check if any incident in its window
            has_incident = []
            for i in range(n_test):
                t_start = test_start + i
                window = incident_feat[t_start:t_start + input_len]
                has_incident.append(window.sum() > 0)
            has_incident = np.array(has_incident)

            n_inc = has_incident.sum()
            n_non = (~has_incident).sum()
            label_a = f"{row_a['coreset_selection_strategy']}_{row_a['coreset_distance_type']}"
            label_b = f"{row_b['coreset_selection_strategy']}_{row_b['coreset_distance_type']}"

            print(f"\n{title}:")
            print(f"  Incident samples: {n_inc}/{n_test} ({100*n_inc/n_test:.1f}%)")
            print(f"  A ({label_a}): inc_MAE={mae_a[has_incident].mean():.3f}, non_inc_MAE={mae_a[~has_incident].mean():.3f}")
            print(f"  B ({label_b}): inc_MAE={mae_b[has_incident].mean():.3f}, non_inc_MAE={mae_b[~has_incident].mean():.3f}")
            diff_inc = mae_b[has_incident].mean() - mae_a[has_incident].mean()
            diff_non = mae_b[~has_incident].mean() - mae_a[~has_incident].mean()
            print(f"  Gap (B-A): incident={diff_inc:+.3f}, non-incident={diff_non:+.3f}")
    else:
        print("No incident feature found in data")
except Exception as e:
    print(f"Could not load incident data: {e}")
