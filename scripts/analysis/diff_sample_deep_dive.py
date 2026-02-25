#!/usr/bin/env python3
"""
Deep dive: diff samples between seeds for graph_cut and k_medoids

1. graph_cut combined_070: 22 diff samples → per-sample error, characteristics
2. k_medoids combined_070: ~6000 diff samples → common core vs diff analysis
3. Are diff samples "hard" or "easy" to predict?
"""

import os, sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

OUTPUT_DIR = 'experiments/result/analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Constants ──
INDEX_DIR = 'coreset_indices/SAN_BERNARDINO'
CKPT_BASE = 'checkpoints/phase_a_distance_screening/STGCNChebGraphConv/xtraffic/SAN_BERNARDINO_100_12_12/1'
STEPS_PER_DAY = 288
DATA_RANGE = 24192
N_NODES = 893
N_TRAIN = int(DATA_RANGE * 0.6)  # 14515... actually let's compute
INPUT_LEN = 12

# Load raw data
data = np.memmap('datasets/SAN_BERNARDINO/data.dat', dtype='float32', mode='r')
data = data.reshape(105120, N_NODES, 5)[:DATA_RANGE]  # flow, occ, speed, tod, dow

def get_hour(idx):
    return (idx % STEPS_PER_DAY) * 24.0 / STEPS_PER_DAY

def get_dow(idx):
    return (idx // STEPS_PER_DAY) % 7

def get_day_name(dow):
    return ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][int(dow)]

def get_traffic_stats(idx):
    """Get traffic characteristics for a sample."""
    t = idx
    if t + INPUT_LEN > len(data):
        return None
    window = data[t:t+INPUT_LEN]  # (12, 893, 5)
    return {
        'mean_flow': window[:, :, 0].mean(),
        'std_flow': window[:, :, 0].std(),
        'mean_speed': window[:, :, 2].mean(),
        'std_speed': window[:, :, 2].std(),
        'min_speed': window[:, :, 2].mean(axis=1).min(),  # worst time step avg speed
        'flow_range': window[:, :, 0].mean(axis=1).max() - window[:, :, 0].mean(axis=1).min(),
    }


# ── Map checkpoint dirs to configs ──
def find_checkpoint(method, distance, ratio, seed):
    for d in os.listdir(CKPT_BASE):
        cfg_path = os.path.join(CKPT_BASE, d, 'cfg.txt')
        if not os.path.exists(cfg_path):
            continue
        with open(cfg_path) as f:
            content = f.read()
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
        if (config.get('SELECTION_STRATEGY') == method and
            config.get('DISTANCE_TYPE') == distance and
            abs(float(config.get('SELECTION_RATIO', 0)) - ratio) < 0.01 and
            int(config.get('SEED', 0)) == seed):
            return d
    return None

def load_test_mae(ckpt_dir):
    """Load per-sample MAE from test_results.npz."""
    npz_path = os.path.join(CKPT_BASE, ckpt_dir, 'test_results.npz')
    data = np.load(npz_path)
    pred = data['prediction']  # (n_test, 12, 893, 1)
    tgt = data['target']
    mae_per_sample = np.abs(pred - tgt).mean(axis=(1, 2, 3))
    return mae_per_sample


# ══════════════════════════════════════════════════════════════════════
# 1. GRAPH_CUT: 22 diff samples deep dive
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("1. GRAPH_CUT COMBINED_070: 22 DIFF SAMPLES")
print("=" * 70)

with open(os.path.join(INDEX_DIR, 'graph_cut_combined_070_seed42.json')) as f:
    gc_idx42 = json.load(f)
with open(os.path.join(INDEX_DIR, 'graph_cut_combined_070_seed123.json')) as f:
    gc_idx123 = json.load(f)

gc_only42 = sorted(set(gc_idx42) - set(gc_idx123))
gc_only123 = sorted(set(gc_idx123) - set(gc_idx42))
gc_common = sorted(set(gc_idx42) & set(gc_idx123))

print(f"only_seed42: {len(gc_only42)}, only_seed123: {len(gc_only123)}, common: {len(gc_common)}")

# Load per-sample test MAE for both models
gc_ckpt42 = find_checkpoint('graph_cut', 'combined', 0.7, 42)
gc_ckpt123 = find_checkpoint('graph_cut', 'combined', 0.7, 123)
print(f"Checkpoint seed42: {gc_ckpt42}")
print(f"Checkpoint seed123: {gc_ckpt123}")

if gc_ckpt42 and gc_ckpt123:
    gc_mae42 = load_test_mae(gc_ckpt42)
    gc_mae123 = load_test_mae(gc_ckpt123)
    n_test = len(gc_mae42)
    test_start = N_TRAIN  # test set starts after train

    # Actually, test set offset: train=0.6*24192=14515, val=0.2*24192=4838, test starts at 14515+4838=19353
    # But the dataset handles input_len offset, so test sample i corresponds to timestep (train+val+i)
    # Let's compute correctly
    n_total_samples = DATA_RANGE - INPUT_LEN - INPUT_LEN + 1  # accounting for input+output
    # Actually BasicTS: train ratio 0.6 of total timesteps = 0.6 * 24192 = 14515 timesteps
    # train samples: 0 to 14515-input_len-output_len+1
    # test sample i -> timestep = test_start_timestep + i

    train_steps = int(DATA_RANGE * 0.6)
    val_steps = int(DATA_RANGE * 0.2)
    test_start_step = train_steps + val_steps  # 19354

    print(f"\nTest samples: {n_test}, test starts at step {test_start_step}")
    print(f"Overall MAE: seed42={gc_mae42.mean():.4f}, seed123={gc_mae123.mean():.4f}")
    print(f"MAE diff: {gc_mae42.mean() - gc_mae123.mean():.4f}")

    # ── Characterize the 22 diff samples as TRAINING samples ──
    print(f"\n--- The 22 diff training samples ---")
    diff_data = []
    for label, samples in [('only_42', gc_only42), ('only_123', gc_only123)]:
        for idx in samples:
            h = get_hour(idx)
            d = get_dow(idx)
            stats = get_traffic_stats(idx)
            diff_data.append({
                'group': label,
                'sample_idx': idx,
                'hour': round(h, 1),
                'dow': get_day_name(d),
                'mean_flow': round(stats['mean_flow'], 1) if stats else None,
                'std_flow': round(stats['std_flow'], 1) if stats else None,
                'mean_speed': round(stats['mean_speed'], 1) if stats else None,
                'min_speed': round(stats['min_speed'], 1) if stats else None,
            })

    diff_df = pd.DataFrame(diff_data)
    print(diff_df.to_string(index=False))

    # ── Are these in test set? If so, check their error ──
    # Training samples are indices 0..N_TRAIN-1, test samples start later
    # The diff samples are training indices, so we can't directly look at test error for them
    # BUT we can ask: do models trained with/without these samples predict differently?

    # ── Per-sample error DIFFERENCE between two models ──
    mae_diff = gc_mae42 - gc_mae123  # positive = seed42 model worse
    print(f"\n--- Per-sample MAE difference (seed42 - seed123) ---")
    print(f"  Mean: {mae_diff.mean():+.4f}")
    print(f"  Std:  {mae_diff.std():.4f}")
    print(f"  Max absolute: {np.abs(mae_diff).max():.4f}")
    print(f"  % samples where seed42 worse: {(mae_diff > 0).mean()*100:.1f}%")

    # Error by time of day
    test_hours = np.array([get_hour(test_start_step + i) for i in range(n_test)])
    hour_bins = np.floor(test_hours).astype(int)

    print(f"\n--- MAE difference by hour of day ---")
    for h in range(0, 24, 3):
        mask = (hour_bins >= h) & (hour_bins < h + 3)
        if mask.sum() > 0:
            d = mae_diff[mask]
            print(f"  {h:02d}-{h+3:02d}h: mean_diff={d.mean():+.4f}, "
                  f"|diff|={np.abs(d).mean():.4f}, n={mask.sum()}")

    # Top 20 test samples with biggest error difference
    top_diff_idx = np.argsort(np.abs(mae_diff))[-20:][::-1]
    print(f"\n--- Top 20 test samples with biggest |MAE diff| ---")
    print(f"{'test_idx':>8} {'hour':>5} {'dow':>4} {'mae42':>8} {'mae123':>8} {'diff':>8}")
    for ti in top_diff_idx:
        h = get_hour(test_start_step + ti)
        d = get_day_name(get_dow(test_start_step + ti))
        print(f"{ti:8d} {h:5.1f} {d:>4} {gc_mae42[ti]:8.3f} {gc_mae123[ti]:8.3f} {mae_diff[ti]:+8.3f}")


# ══════════════════════════════════════════════════════════════════════
# 2. K_MEDOIDS: Common core analysis
# ══════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("2. K_MEDOIDS COMBINED_070: COMMON CORE vs DIFF")
print("=" * 70)

with open(os.path.join(INDEX_DIR, 'k_medoids_combined_070_seed42.json')) as f:
    km_idx42 = set(json.load(f))
with open(os.path.join(INDEX_DIR, 'k_medoids_combined_070_seed123.json')) as f:
    km_idx123 = set(json.load(f))

km_common = sorted(km_idx42 & km_idx123)
km_only42 = sorted(km_idx42 - km_idx123)
km_only123 = sorted(km_idx123 - km_idx42)
km_neither = sorted(set(range(N_TRAIN)) - km_idx42 - km_idx123)

# Actually N_TRAIN is wrong - let me get the actual number of training samples
# from index file length
total_train = max(max(km_idx42), max(km_idx123)) + 1
# ratio=0.7 means 10145 out of ~14493
print(f"seed42: {len(km_idx42)}, seed123: {len(km_idx123)}")
print(f"Common: {len(km_common)} ({100*len(km_common)/len(km_idx42):.1f}%)")
print(f"Only seed42: {len(km_only42)}")
print(f"Only seed123: {len(km_only123)}")

# ── Characterize the groups ──
def group_stats(indices, name, max_n=2000):
    """Compute aggregate statistics for a group of sample indices."""
    if len(indices) > max_n:
        np.random.seed(0)
        indices = list(np.random.choice(indices, max_n, replace=False))

    hours = [get_hour(i) for i in indices]
    dows = [get_dow(i) for i in indices]
    flows, speeds, stds = [], [], []
    for i in indices:
        s = get_traffic_stats(i)
        if s:
            flows.append(s['mean_flow'])
            speeds.append(s['mean_speed'])
            stds.append(s['std_flow'])

    hour_counts = Counter([int(h) for h in hours])
    # Peak hour ratio (7-9, 16-19)
    peak = sum(hour_counts.get(h, 0) for h in [7, 8, 16, 17, 18])
    offpeak = sum(hour_counts.get(h, 0) for h in [0, 1, 2, 3, 4, 5, 22, 23])
    total = len(hours)

    print(f"\n  [{name}] n={len(indices)}")
    print(f"    Hour: mean={np.mean(hours):.1f}, std={np.std(hours):.1f}")
    print(f"    Peak(7-9,16-19): {100*peak/total:.1f}%, Off-peak(22-5): {100*offpeak/total:.1f}%")
    print(f"    Weekday: {100*sum(1 for d in dows if d < 5)/total:.1f}%")
    print(f"    Flow: mean={np.mean(flows):.1f}, std_of_means={np.std(flows):.1f}")
    print(f"    Speed: mean={np.mean(speeds):.1f}")
    print(f"    Flow variability: mean={np.mean(stds):.1f}")

    return {
        'name': name, 'n': len(indices),
        'mean_hour': np.mean(hours),
        'peak_pct': 100*peak/total,
        'offpeak_pct': 100*offpeak/total,
        'weekday_pct': 100*sum(1 for d in dows if d < 5)/total,
        'mean_flow': np.mean(flows),
        'mean_speed': np.mean(speeds),
        'flow_variability': np.mean(stds),
    }

stats = []
stats.append(group_stats(km_common, "Common (always selected)"))
stats.append(group_stats(km_only42, "Only seed42"))
stats.append(group_stats(km_only123, "Only seed123"))

# What about never-selected samples?
all_selected = km_idx42 | km_idx123
n_total_train = 14493  # from proxy metrics json
never_selected = sorted(set(range(n_total_train)) - all_selected)
if never_selected:
    stats.append(group_stats(never_selected, "Never selected"))

# ── K_medoids per-sample error ──
km_ckpt42 = find_checkpoint('k_medoids', 'combined', 0.7, 42)
km_ckpt123 = find_checkpoint('k_medoids', 'combined', 0.7, 123)

if km_ckpt42 and km_ckpt123:
    km_mae42 = load_test_mae(km_ckpt42)
    km_mae123 = load_test_mae(km_ckpt123)
    km_mae_diff = km_mae42 - km_mae123

    print(f"\n--- K_medoids per-sample MAE difference ---")
    print(f"  Overall MAE: seed42={km_mae42.mean():.4f}, seed123={km_mae123.mean():.4f}")
    print(f"  Mean diff: {km_mae_diff.mean():+.4f}")
    print(f"  Std diff: {km_mae_diff.std():.4f}")
    print(f"  % worse for seed42: {(km_mae_diff > 0).mean()*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════
# 3. VISUALIZATION
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Row 1: Graph_cut analysis
if gc_ckpt42 and gc_ckpt123:
    # 1a: Per-sample MAE diff histogram
    ax = axes[0, 0]
    ax.hist(mae_diff, bins=80, density=True, alpha=0.7, color='steelblue')
    ax.axvline(0, color='red', linestyle='--', alpha=0.7)
    ax.axvline(mae_diff.mean(), color='orange', linestyle='-', label=f'mean={mae_diff.mean():+.3f}')
    ax.set_xlabel('MAE(seed42) - MAE(seed123)')
    ax.set_ylabel('Density')
    ax.set_title('Graph Cut: Per-sample MAE Difference')
    ax.legend(fontsize=8)

    # 1b: MAE diff by hour
    ax = axes[0, 1]
    hourly_diff = pd.DataFrame({'hour': hour_bins, 'diff': mae_diff}).groupby('hour')['diff'].mean()
    ax.bar(hourly_diff.index, hourly_diff.values, color=['tab:red' if v > 0 else 'tab:blue' for v in hourly_diff.values], alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Mean MAE diff (seed42 - seed123)')
    ax.set_title('Graph Cut: Error Diff by Hour')
    ax.set_xticks(range(0, 24, 3))

    # 1c: The 22 diff training samples - hour distribution
    ax = axes[0, 2]
    h42 = [get_hour(i) for i in gc_only42]
    h123 = [get_hour(i) for i in gc_only123]
    ax.scatter(h42, [1]*len(h42), s=100, marker='|', color='tab:blue', linewidths=2, label=f'Only seed42 ({len(gc_only42)})')
    ax.scatter(h123, [0]*len(h123), s=100, marker='|', color='tab:orange', linewidths=2, label=f'Only seed123 ({len(gc_only123)})')
    ax.set_xlim(0, 24)
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['seed123\nonly', 'seed42\nonly'])
    ax.set_xlabel('Hour of Day')
    ax.set_title(f'Graph Cut: 22 Diff Training Samples')
    ax.legend(fontsize=8)

# Row 2: K_medoids analysis
# 2a: Common core temporal distribution
ax = axes[1, 0]
common_hours = [get_hour(i) for i in km_common[:2000]]
only42_hours = [get_hour(i) for i in km_only42[:2000]]
only123_hours = [get_hour(i) for i in km_only123[:2000]]
ax.hist(common_hours, bins=24, range=(0, 24), alpha=0.5, density=True, label=f'Common ({len(km_common)})', color='gray')
ax.hist(only42_hours, bins=24, range=(0, 24), alpha=0.5, density=True, label=f'Only42 ({len(km_only42)})', color='tab:blue')
ax.hist(only123_hours, bins=24, range=(0, 24), alpha=0.5, density=True, label=f'Only123 ({len(km_only123)})', color='tab:orange')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Density')
ax.set_title('K_medoids: Common vs Diff Samples')
ax.legend(fontsize=8)

# 2b: Common core flow distribution
ax = axes[1, 1]
np.random.seed(0)
common_flows = [get_traffic_stats(i)['mean_flow'] for i in np.random.choice(km_common, 2000, replace=False) if get_traffic_stats(i)]
only42_flows = [get_traffic_stats(i)['mean_flow'] for i in np.random.choice(km_only42, min(2000, len(km_only42)), replace=False) if get_traffic_stats(i)]
only123_flows = [get_traffic_stats(i)['mean_flow'] for i in np.random.choice(km_only123, min(2000, len(km_only123)), replace=False) if get_traffic_stats(i)]
if never_selected:
    never_flows = [get_traffic_stats(i)['mean_flow'] for i in np.random.choice(never_selected, min(2000, len(never_selected)), replace=False) if get_traffic_stats(i)]

ax.hist(common_flows, bins=30, alpha=0.4, density=True, label='Common', color='gray')
ax.hist(only42_flows, bins=30, alpha=0.4, density=True, label='Only42', color='tab:blue')
if never_selected:
    ax.hist(never_flows, bins=30, alpha=0.4, density=True, label='Never selected', color='tab:red')
ax.set_xlabel('Mean Flow')
ax.set_ylabel('Density')
ax.set_title('K_medoids: Flow Distribution by Group')
ax.legend(fontsize=8)

# 2c: K_medoids per-sample MAE diff
if km_ckpt42 and km_ckpt123:
    ax = axes[1, 2]
    ax.hist(km_mae_diff, bins=80, density=True, alpha=0.7, color='steelblue')
    ax.axvline(0, color='red', linestyle='--', alpha=0.7)
    ax.axvline(km_mae_diff.mean(), color='orange', linestyle='-', label=f'mean={km_mae_diff.mean():+.3f}')
    ax.set_xlabel('MAE(seed42) - MAE(seed123)')
    ax.set_ylabel('Density')
    ax.set_title('K_medoids: Per-sample MAE Difference')
    ax.legend(fontsize=8)

fig.suptitle('Diff Sample Deep Dive: Graph Cut (22 diff) vs K_medoids (6000 diff)',
            fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(f'{OUTPUT_DIR}/diff_sample_deep_dive.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: {OUTPUT_DIR}/diff_sample_deep_dive.png")

# ── Summary table ──
print("\n\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
if gc_ckpt42 and gc_ckpt123 and km_ckpt42 and km_ckpt123:
    print(f"\n{'':30s} {'graph_cut':>15s} {'k_medoids':>15s}")
    print(f"{'Diff training samples':30s} {len(gc_only42)*2:>15d} {len(km_only42)+len(km_only123):>15d}")
    print(f"{'% diff':30s} {100*len(gc_only42)*2/len(gc_idx42):>14.1f}% {100*(len(km_only42)+len(km_only123))/(2*len(km_idx42)):>14.1f}%")
    print(f"{'Overall MAE seed42':30s} {gc_mae42.mean():>15.4f} {km_mae42.mean():>15.4f}")
    print(f"{'Overall MAE seed123':30s} {gc_mae123.mean():>15.4f} {km_mae123.mean():>15.4f}")
    print(f"{'MAE diff':30s} {gc_mae42.mean()-gc_mae123.mean():>+15.4f} {km_mae42.mean()-km_mae123.mean():>+15.4f}")
    print(f"{'Per-sample MAE std':30s} {mae_diff.std():>15.4f} {km_mae_diff.std():>15.4f}")
