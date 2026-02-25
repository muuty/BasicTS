#!/usr/bin/env python3
"""
Graph_cut seed 간 미세 차이 분석

graph_cut은 seed 간 Jaccard ~0.998인데 MAE는 크게 다르다.
그 아주 작은 차이가 어떤 샘플인지 구체적으로 분석한다.

1. seed42 vs seed123에서 다른 샘플들 식별
2. 그 샘플들의 특성 분석 (시간대, traffic 패턴)
3. 그 샘플들이 학습에 미치는 영향 추정
"""

import os, sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

OUTPUT_DIR = 'experiments/result/analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

INDEX_DIR = 'coreset_indices/SAN_BERNARDINO'

# ── Dataset metadata ──
STEPS_PER_DAY = 288
DATA_RANGE = 24192  # 3 months
INPUT_LEN = 12
N_NODES = 893
N_TRAIN = int(DATA_RANGE * 0.6)  # train samples start from 0

# Load raw data for characterizing samples
data = np.memmap('datasets/SAN_BERNARDINO/data.dat', dtype='float32', mode='r')
data = data.reshape(105120, N_NODES, 5)[:DATA_RANGE]  # flow, occ, speed, tod, dow

def sample_to_timestep(idx):
    """Sample index → starting timestep in training set."""
    return idx  # direct mapping in BasicTS

def get_tod_hour(timestep):
    """Timestep → hour of day."""
    tod_idx = timestep % STEPS_PER_DAY
    return tod_idx * 24.0 / STEPS_PER_DAY

def get_dow(timestep):
    """Timestep → day of week."""
    return (timestep // STEPS_PER_DAY) % 7

def characterize_samples(indices, label=""):
    """Compute characteristics of a set of sample indices."""
    hours = np.array([get_tod_hour(i) for i in indices])
    dows = np.array([get_dow(i) for i in indices])

    # Traffic characteristics: mean flow, speed at each sample's timestep
    flows = []
    speeds = []
    flow_stds = []
    for idx in indices:
        t = sample_to_timestep(idx)
        if t + INPUT_LEN <= len(data):
            window = data[t:t+INPUT_LEN]  # (12, 893, 5)
            flows.append(window[:, :, 0].mean())   # mean flow
            speeds.append(window[:, :, 2].mean())   # mean speed
            flow_stds.append(window[:, :, 0].std())  # flow variability

    return {
        'label': label,
        'n': len(indices),
        'hours': hours,
        'dows': dows,
        'mean_flow': np.mean(flows) if flows else 0,
        'mean_speed': np.mean(speeds) if speeds else 0,
        'mean_flow_std': np.mean(flow_stds) if flow_stds else 0,
        'flows': np.array(flows),
        'speeds': np.array(speeds),
        'flow_stds': np.array(flow_stds),
    }


# ── 1. Graph_cut seed 차이 분석 ──
print("=" * 70)
print("1. GRAPH_CUT SEED DIFFERENCE ANALYSIS")
print("=" * 70)

distances = ['combined', 'euclidean', 'temporal', 'spatial']
ratios = ['030', '070']

all_diff_stats = []

for dist in distances:
    for ratio in ratios:
        f42 = os.path.join(INDEX_DIR, f'graph_cut_{dist}_{ratio}_seed42.json')
        f123 = os.path.join(INDEX_DIR, f'graph_cut_{dist}_{ratio}_seed123.json')

        if not (os.path.exists(f42) and os.path.exists(f123)):
            continue

        with open(f42) as f:
            idx42 = set(json.load(f))
        with open(f123) as f:
            idx123 = set(json.load(f))

        common = idx42 & idx123
        only_42 = idx42 - idx123
        only_123 = idx123 - idx42
        jaccard = len(common) / len(idx42 | idx123)

        print(f"\n--- graph_cut_{dist}_{ratio} ---")
        print(f"  seed42: {len(idx42)}, seed123: {len(idx123)}")
        print(f"  Common: {len(common)}, Only42: {len(only_42)}, Only123: {len(only_123)}")
        print(f"  Jaccard: {jaccard:.4f}")

        if len(only_42) > 0:
            chars_42 = characterize_samples(list(only_42), f"only_seed42_{dist}_{ratio}")
            chars_123 = characterize_samples(list(only_123), f"only_seed123_{dist}_{ratio}")
            chars_common = characterize_samples(list(common)[:1000], f"common_{dist}_{ratio}")  # sample 1000 from common

            print(f"\n  Only-42 samples ({len(only_42)}):")
            print(f"    Hours: mean={chars_42['hours'].mean():.1f}, std={chars_42['hours'].std():.1f}")
            print(f"    Mean flow: {chars_42['mean_flow']:.2f}, Mean speed: {chars_42['mean_speed']:.2f}")
            print(f"    Flow variability: {chars_42['mean_flow_std']:.2f}")

            print(f"  Only-123 samples ({len(only_123)}):")
            print(f"    Hours: mean={chars_123['hours'].mean():.1f}, std={chars_123['hours'].std():.1f}")
            print(f"    Mean flow: {chars_123['mean_flow']:.2f}, Mean speed: {chars_123['mean_speed']:.2f}")
            print(f"    Flow variability: {chars_123['mean_flow_std']:.2f}")

            print(f"  Common samples (sampled 1000):")
            print(f"    Hours: mean={chars_common['hours'].mean():.1f}, std={chars_common['hours'].std():.1f}")
            print(f"    Mean flow: {chars_common['mean_flow']:.2f}, Mean speed: {chars_common['mean_speed']:.2f}")
            print(f"    Flow variability: {chars_common['mean_flow_std']:.2f}")

            all_diff_stats.append({
                'distance': dist, 'ratio': ratio,
                'n_diff': len(only_42) + len(only_123),
                'n_total': len(idx42),
                'pct_diff': 100 * (len(only_42) + len(only_123)) / (2 * len(idx42)),
                'jaccard': jaccard,
                'only42_mean_hour': chars_42['hours'].mean(),
                'only123_mean_hour': chars_123['hours'].mean(),
                'common_mean_hour': chars_common['hours'].mean(),
                'only42_mean_flow': chars_42['mean_flow'],
                'only123_mean_flow': chars_123['mean_flow'],
                'common_mean_flow': chars_common['mean_flow'],
                'only42_flow_std': chars_42['mean_flow_std'],
                'only123_flow_std': chars_123['mean_flow_std'],
                'common_flow_std': chars_common['mean_flow_std'],
            })

# ── Detailed plot for combined_070 (most important setting) ──
print("\n\n--- DETAILED: graph_cut_combined_070 ---")
f42 = os.path.join(INDEX_DIR, 'graph_cut_combined_070_seed42.json')
f123 = os.path.join(INDEX_DIR, 'graph_cut_combined_070_seed123.json')

with open(f42) as f:
    idx42 = set(json.load(f))
with open(f123) as f:
    idx123 = set(json.load(f))

only_42 = sorted(idx42 - idx123)
only_123 = sorted(idx123 - idx42)
common = sorted(idx42 & idx123)

print(f"  Diff samples: only42={len(only_42)}, only123={len(only_123)}")

# Characterize ALL diff samples
chars_42 = characterize_samples(only_42, "only_42")
chars_123 = characterize_samples(only_123, "only_123")
# Random sample from common for comparison
np.random.seed(0)
common_sample = list(np.random.choice(common, min(2000, len(common)), replace=False))
chars_common = characterize_samples(common_sample, "common")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Row 1: Hour distribution of diff samples vs common
ax = axes[0, 0]
if len(chars_42['hours']) > 0:
    ax.hist(chars_42['hours'], bins=24, range=(0, 24), alpha=0.5, density=True, label=f'Only seed42 (n={len(only_42)})', color='tab:blue')
if len(chars_123['hours']) > 0:
    ax.hist(chars_123['hours'], bins=24, range=(0, 24), alpha=0.5, density=True, label=f'Only seed123 (n={len(only_123)})', color='tab:orange')
ax.hist(chars_common['hours'], bins=24, range=(0, 24), alpha=0.3, density=True, label=f'Common (n={len(common_sample)})', color='gray')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Density')
ax.set_title('Time-of-Day: Diff vs Common Samples')
ax.legend(fontsize=8)

# Row 1: Day of week
ax = axes[0, 1]
if len(chars_42['dows']) > 0:
    ax.hist(chars_42['dows'], bins=7, range=(-0.5, 6.5), alpha=0.5, density=True, label='Only seed42', color='tab:blue')
if len(chars_123['dows']) > 0:
    ax.hist(chars_123['dows'], bins=7, range=(-0.5, 6.5), alpha=0.5, density=True, label='Only seed123', color='tab:orange')
ax.hist(chars_common['dows'], bins=7, range=(-0.5, 6.5), alpha=0.3, density=True, label='Common', color='gray')
ax.set_xlabel('Day of Week (0=Mon)')
ax.set_ylabel('Density')
ax.set_title('Day-of-Week: Diff vs Common')
ax.legend(fontsize=8)

# Row 1: Flow distribution
ax = axes[0, 2]
if len(chars_42['flows']) > 0:
    ax.hist(chars_42['flows'], bins=30, alpha=0.5, density=True, label='Only seed42', color='tab:blue')
if len(chars_123['flows']) > 0:
    ax.hist(chars_123['flows'], bins=30, alpha=0.5, density=True, label='Only seed123', color='tab:orange')
ax.hist(chars_common['flows'], bins=30, alpha=0.3, density=True, label='Common', color='gray')
ax.set_xlabel('Mean Flow')
ax.set_ylabel('Density')
ax.set_title('Mean Flow: Diff vs Common')
ax.legend(fontsize=8)

# Row 2: Speed distribution
ax = axes[1, 0]
if len(chars_42['speeds']) > 0:
    ax.hist(chars_42['speeds'], bins=30, alpha=0.5, density=True, label='Only seed42', color='tab:blue')
if len(chars_123['speeds']) > 0:
    ax.hist(chars_123['speeds'], bins=30, alpha=0.5, density=True, label='Only seed123', color='tab:orange')
ax.hist(chars_common['speeds'], bins=30, alpha=0.3, density=True, label='Common', color='gray')
ax.set_xlabel('Mean Speed')
ax.set_ylabel('Density')
ax.set_title('Mean Speed: Diff vs Common')
ax.legend(fontsize=8)

# Row 2: Flow variability
ax = axes[1, 1]
if len(chars_42['flow_stds']) > 0:
    ax.hist(chars_42['flow_stds'], bins=30, alpha=0.5, density=True, label='Only seed42', color='tab:blue')
if len(chars_123['flow_stds']) > 0:
    ax.hist(chars_123['flow_stds'], bins=30, alpha=0.5, density=True, label='Only seed123', color='tab:orange')
ax.hist(chars_common['flow_stds'], bins=30, alpha=0.3, density=True, label='Common', color='gray')
ax.set_xlabel('Flow Std (variability)')
ax.set_ylabel('Density')
ax.set_title('Flow Variability: Diff vs Common')
ax.legend(fontsize=8)

# Row 2: Timeline - where are the diff samples?
ax = axes[1, 2]
if len(only_42) > 0:
    ax.scatter(only_42, [1]*len(only_42), alpha=0.3, s=2, color='tab:blue', label='Only seed42')
if len(only_123) > 0:
    ax.scatter(only_123, [0]*len(only_123), alpha=0.3, s=2, color='tab:orange', label='Only seed123')
ax.set_xlabel('Sample Index (timestep)')
ax.set_ylabel('')
ax.set_yticks([0, 1])
ax.set_yticklabels(['Only seed123', 'Only seed42'])
ax.set_title('Timeline Position of Diff Samples')
ax.legend(fontsize=8)

fig.suptitle('Graph Cut Seed Difference Analysis (combined, ratio=0.7)\n'
             f'Only {len(only_42)+len(only_123)} samples differ out of ~{len(idx42)}',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(f'{OUTPUT_DIR}/graphcut_seed_diff.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/graphcut_seed_diff.png")


# ── 2. Graph_cut sigma 차이가 원인인지 확인 ──
print("\n\n" + "=" * 70)
print("2. GRAPH_CUT RBF SIGMA ANALYSIS")
print("=" * 70)

# Reproduce sigma computation for different seeds
from basicts.data import TimeSeriesForecastingDataset

class DummyModel:
    class FORWARD_FEATURES:
        pass
    class TARGET_FEATURES:
        pass
DummyModel.FORWARD_FEATURES = [0, 1, 2]
DummyModel.TARGET_FEATURES = [0]

dataset = TimeSeriesForecastingDataset(
    dataset_name='SAN_BERNARDINO',
    train_val_test_ratio=[0.6, 0.2, 0.2],
    mode='train', input_len=12, output_len=12,
    data_range=(0, 24192),
)

from coreset.distance import extract_features, compute_distance_matrix
import torch

print("Extracting features...")
inputs, targets = extract_features(dataset, DummyModel)

for dist_type in ['combined']:
    print(f"\n  Computing {dist_type} distance matrix...")
    dm = compute_distance_matrix(inputs, targets, dist_type)
    dm_np = dm if isinstance(dm, np.ndarray) else dm.numpy()
    N = dm_np.shape[0]

    # Replicate graph_cut's sigma computation for different seeds
    n_sample = 1000
    for seed in [42, 123, 456, 789]:
        np.random.seed(seed)
        idx_i = np.random.randint(0, N, n_sample)
        idx_j = np.random.randint(0, N, n_sample)
        mask = idx_i != idx_j
        sampled_dists = dm_np[idx_i[mask], idx_j[mask]]
        sigma = float(np.median(sampled_dists))
        print(f"    seed={seed}: sigma={sigma:.6f} (from {mask.sum()} pairs)")

    del dm, dm_np
    import gc; gc.collect()

del inputs, targets
gc.collect()
torch.cuda.empty_cache()


# ── 3. Graph_cut의 greedy 순서에서 diff 샘플의 위치 ──
print("\n\n" + "=" * 70)
print("3. WHERE DO DIFF SAMPLES APPEAR IN GREEDY ORDER?")
print("=" * 70)

# The graph_cut greedy algorithm adds samples in order of marginal gain.
# If diff samples are added late (low marginal gain), they're "borderline" samples.
# Let's check if the diff samples are near the selection boundary.

# Load both index files as ordered lists (they're stored in selection order!)
with open(os.path.join(INDEX_DIR, 'graph_cut_combined_070_seed42.json')) as f:
    ordered_42 = json.load(f)
with open(os.path.join(INDEX_DIR, 'graph_cut_combined_070_seed123.json')) as f:
    ordered_123 = json.load(f)

only_42_set = set(ordered_42) - set(ordered_123)
only_123_set = set(ordered_123) - set(ordered_42)

# Find position of diff samples in selection order
pos_42 = [i for i, s in enumerate(ordered_42) if s in only_42_set]
pos_123 = [i for i, s in enumerate(ordered_123) if s in only_123_set]

total = len(ordered_42)
print(f"\nDiff samples in seed42's selection order:")
print(f"  Positions: min={min(pos_42) if pos_42 else 'N/A'}, max={max(pos_42) if pos_42 else 'N/A'}, "
      f"mean={np.mean(pos_42) if pos_42 else 'N/A':.0f}")
print(f"  Total selected: {total}")
if pos_42:
    print(f"  % in last 10%: {100*sum(1 for p in pos_42 if p > 0.9*total)/len(pos_42):.1f}%")
    print(f"  % in last 1%: {100*sum(1 for p in pos_42 if p > 0.99*total)/len(pos_42):.1f}%")

print(f"\nDiff samples in seed123's selection order:")
print(f"  Positions: min={min(pos_123) if pos_123 else 'N/A'}, max={max(pos_123) if pos_123 else 'N/A'}, "
      f"mean={np.mean(pos_123) if pos_123 else 'N/A':.0f}")
if pos_123:
    print(f"  % in last 10%: {100*sum(1 for p in pos_123 if p > 0.9*total)/len(pos_123):.1f}%")
    print(f"  % in last 1%: {100*sum(1 for p in pos_123 if p > 0.99*total)/len(pos_123):.1f}%")

# Plot: selection order histogram
fig, ax = plt.subplots(1, 1, figsize=(10, 4))
if pos_42:
    ax.hist(pos_42, bins=50, alpha=0.5, label=f'Only-42 diffs (n={len(pos_42)})', color='tab:blue')
if pos_123:
    ax.hist(pos_123, bins=50, alpha=0.5, label=f'Only-123 diffs (n={len(pos_123)})', color='tab:orange')
ax.axvline(0.9 * total, color='red', linestyle='--', alpha=0.5, label='Last 10% boundary')
ax.set_xlabel('Selection Order Position')
ax.set_ylabel('Count')
ax.set_title('Where Diff Samples Appear in Graph Cut Greedy Order')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/graphcut_diff_position.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: {OUTPUT_DIR}/graphcut_diff_position.png")

# Save summary
if all_diff_stats:
    pd.DataFrame(all_diff_stats).to_csv(f'{OUTPUT_DIR}/graphcut_seed_diff.csv', index=False)
    print(f"Saved: {OUTPUT_DIR}/graphcut_seed_diff.csv")

print("\nDone.")
