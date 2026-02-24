"""
Why is incident/non-incident MAE difference meaningful at only 0.87%?
=====================================================================
Tests 6 hypotheses for why the model performs worse during incidents
despite incidents affecting only 0.87% of (sample, node) pairs.

Hypotheses:
H1: Distribution shift - incidents create patterns unseen in training
H2: Flow magnitude - incidents cause congestion → higher absolute errors
H3: Input contamination - incidents in input window feed anomalous data
H4: Temporal confounding - incidents cluster at hard-to-predict times
H5: Heavy-tail errors - a few extreme errors drive the mean
H6: Consistent direction - effect is always worse, never better → significant
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

sys.path.append('/data/pretrainingbasicts')

# ============================================================================
# Configuration
# ============================================================================
PRED_DIR = 'checkpoints/ContextContrastive_baseline_3mo/xtraffic_SAN_BERNARDINO_30_12_12/040af4f5bcb37097bc5263d7f350174b/test_results'
DATA_PATH = 'datasets/xtraffic/SAN_BERNARDINO/data.dat'
INCIDENT_PATH = 'datasets/xtraffic/SAN_BERNARDINO/incident_metadata_2023.csv'
OUTPUT_DIR = 'eda/post_incident'

NUM_NODES = 893
INPUT_LEN = 12
OUTPUT_LEN = 12
N_TEST_SAMPLES = 5233
DATA_RANGE = (0, 26280)
TRAIN_VAL_TEST_RATIO = [0.6, 0.2, 0.2]
STEPS_PER_DAY = 288
RECOVERY_BUFFER = 12

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("WHY IS INCIDENT MAE DIFFERENCE MEANINGFUL AT 0.87%?")
print("=" * 70)

# ============================================================================
# Load data
# ============================================================================
print("\n[LOAD] Loading data...")

predictions = np.memmap(f'{PRED_DIR}/predictions.npy', dtype='float32', mode='r').reshape(N_TEST_SAMPLES, OUTPUT_LEN, NUM_NODES)
targets = np.memmap(f'{PRED_DIR}/targets.npy', dtype='float32', mode='r').reshape(N_TEST_SAMPLES, OUTPUT_LEN, NUM_NODES)
data = np.memmap(DATA_PATH, dtype='float32', mode='r').reshape(-1, NUM_NODES, 5)[:DATA_RANGE[1]]

total_len = DATA_RANGE[1] - DATA_RANGE[0]
test_start = int(total_len * (TRAIN_VAL_TEST_RATIO[0] + TRAIN_VAL_TEST_RATIO[1]))

incidents = pd.read_csv(INCIDENT_PATH)
incidents = incidents[incidents['original_start_slot'] < DATA_RANGE[1]].copy()

pred_window_start = test_start + INPUT_LEN
pred_window_end = test_start + N_TEST_SAMPLES - 1 + INPUT_LEN + OUTPUT_LEN - 1
incidents['incident_end'] = incidents['original_start_slot'] + incidents['duration_slots'] + RECOVERY_BUFFER

test_incidents = incidents[
    (incidents['incident_end'] >= pred_window_start) &
    (incidents['original_start_slot'] <= pred_window_end)
].copy()

# Build incident mask
incident_mask = np.zeros((N_TEST_SAMPLES, NUM_NODES), dtype=bool)
# Also track per-incident info for each (sample, node) pair
incident_start_in_input = np.zeros((N_TEST_SAMPLES, NUM_NODES), dtype=bool)

for _, inc in test_incidents.iterrows():
    s = int(inc['sensor_idx'])
    inc_start = int(inc['original_start_slot'])
    inc_end = int(inc['incident_end'])
    t_min = max(0, inc_start - test_start - INPUT_LEN - OUTPUT_LEN + 1)
    t_max = min(N_TEST_SAMPLES - 1, inc_end - test_start - INPUT_LEN)
    if t_min <= t_max:
        incident_mask[t_min:t_max+1, s] = True
        # Check if incident starts within input window of each sample
        for t in range(t_min, t_max + 1):
            input_start_slot = test_start + t
            input_end_slot = input_start_slot + INPUT_LEN - 1
            if input_start_slot <= inc_start <= input_end_slot:
                incident_start_in_input[t, s] = True

n_affected = incident_mask.sum()
print(f"  Incident-affected pairs: {n_affected:,} / {incident_mask.size:,} ({n_affected/incident_mask.size*100:.3f}%)")

# Compute errors
abs_errors = np.abs(predictions - targets)  # (5233, 12, 893)
nonzero_mask = (targets != 0)
nonzero_count = nonzero_mask.sum(axis=1)  # (5233, 893)
masked_errors = (abs_errors * nonzero_mask).sum(axis=1)
sample_node_masked_mae = np.where(nonzero_count > 0, masked_errors / nonzero_count, np.nan)

# Load sensor categories
dead_indices = set(np.load('datasets/xtraffic/SAN_BERNARDINO/dead_indices.npy'))
major_fail_indices = set(np.load('datasets/xtraffic/SAN_BERNARDINO/major_fail_indices.npy'))

# Filter to meaningful nodes (not dead/major_fail)
meaningful_nodes = [n for n in range(NUM_NODES) if n not in dead_indices and n not in major_fail_indices]
meaningful_set = set(meaningful_nodes)

print(f"  Meaningful nodes: {len(meaningful_nodes)}")

# ============================================================================
# Baseline: confirm the MAE difference
# ============================================================================
print("\n" + "=" * 70)
print("BASELINE CONFIRMATION")
print("=" * 70)

inc_mae_vals = sample_node_masked_mae[incident_mask]
normal_mae_vals = sample_node_masked_mae[~incident_mask]
inc_mae_vals = inc_mae_vals[~np.isnan(inc_mae_vals)]
normal_mae_vals = normal_mae_vals[~np.isnan(normal_mae_vals)]

print(f"  Incident pairs MAE: {inc_mae_vals.mean():.4f} (n={len(inc_mae_vals):,})")
print(f"  Normal pairs MAE:   {normal_mae_vals.mean():.4f} (n={len(normal_mae_vals):,})")
print(f"  Difference: +{inc_mae_vals.mean() - normal_mae_vals.mean():.4f}")

# Also for meaningful nodes only
inc_mask_meaningful = incident_mask[:, meaningful_nodes]
mae_meaningful = sample_node_masked_mae[:, meaningful_nodes]
inc_vals_m = mae_meaningful[inc_mask_meaningful]
normal_vals_m = mae_meaningful[~inc_mask_meaningful]
inc_vals_m = inc_vals_m[~np.isnan(inc_vals_m)]
normal_vals_m = normal_vals_m[~np.isnan(normal_vals_m)]

print(f"\n  [Meaningful nodes only]")
print(f"  Incident pairs MAE: {inc_vals_m.mean():.4f} (n={len(inc_vals_m):,})")
print(f"  Normal pairs MAE:   {normal_vals_m.mean():.4f} (n={len(normal_vals_m):,})")
print(f"  Difference: +{inc_vals_m.mean() - normal_vals_m.mean():.4f}")

# Welch's t-test
t_stat, p_val = stats.ttest_ind(inc_vals_m, normal_vals_m, equal_var=False)
cohen_d = (inc_vals_m.mean() - normal_vals_m.mean()) / np.sqrt((inc_vals_m.var() + normal_vals_m.var()) / 2)
print(f"  Welch's t-test: t={t_stat:.3f}, p={p_val:.2e}")
print(f"  Cohen's d: {cohen_d:.4f}")

# ============================================================================
# H1: Distribution shift - flow patterns during incidents differ
# ============================================================================
print("\n" + "=" * 70)
print("H1: DISTRIBUTION SHIFT - incident traffic patterns differ from training")
print("=" * 70)

# Compare target flow values during incident vs normal
target_flow_h1 = targets[:, 0, :]  # h=1 target flow (first horizon)
inc_flow = target_flow_h1[incident_mask]
normal_flow = target_flow_h1[~incident_mask]
inc_flow_nz = inc_flow[inc_flow != 0]
normal_flow_nz = normal_flow[normal_flow != 0]

print(f"  Target flow (non-zero) during incidents: mean={inc_flow_nz.mean():.2f}, std={inc_flow_nz.std():.2f}, median={np.median(inc_flow_nz):.2f}")
print(f"  Target flow (non-zero) during normal:    mean={normal_flow_nz.mean():.2f}, std={normal_flow_nz.std():.2f}, median={np.median(normal_flow_nz):.2f}")
print(f"  Flow ratio (incident/normal): {inc_flow_nz.mean() / normal_flow_nz.mean():.3f}")

# Compare zero-rate
inc_zero_rate = (inc_flow == 0).mean()
normal_zero_rate = (normal_flow == 0).mean()
print(f"  Zero-flow rate during incidents: {inc_zero_rate:.4f}")
print(f"  Zero-flow rate during normal:    {normal_zero_rate:.4f}")

# Compare flow variability (std within each group)
# Per-node comparison: how different is incident-time flow from training mean?
train_end = int(total_len * TRAIN_VAL_TEST_RATIO[0])
train_flow = data[:train_end, :, 0]
train_mean_flow = train_flow.mean(axis=0)
train_std_flow = train_flow.std(axis=0)

# Z-scores of target flow relative to training distribution
nodes_with_incidents = [n for n in range(NUM_NODES) if incident_mask[:, n].any() and n in meaningful_set]
flow_zscore_incident = []
flow_zscore_normal = []

for n in nodes_with_incidents:
    if train_std_flow[n] < 0.01:
        continue
    inc_mask_n = incident_mask[:, n]
    inc_targets = target_flow_h1[inc_mask_n, n]
    normal_targets = target_flow_h1[~inc_mask_n, n]

    z_inc = np.abs((inc_targets[inc_targets != 0] - train_mean_flow[n]) / train_std_flow[n])
    z_normal = np.abs((normal_targets[normal_targets != 0] - train_mean_flow[n]) / train_std_flow[n])

    if len(z_inc) > 0:
        flow_zscore_incident.extend(z_inc.tolist())
    if len(z_normal) > 0:
        flow_zscore_normal.extend(z_normal.tolist())

flow_zscore_incident = np.array(flow_zscore_incident)
flow_zscore_normal = np.array(flow_zscore_normal)
print(f"\n  |Z-score| of target flow vs training distribution:")
print(f"  Incident: mean={flow_zscore_incident.mean():.3f}, median={np.median(flow_zscore_incident):.3f}")
print(f"  Normal:   mean={flow_zscore_normal.mean():.3f}, median={np.median(flow_zscore_normal):.3f}")
print(f"  → Incident targets are {flow_zscore_incident.mean()/flow_zscore_normal.mean():.2f}x more deviant from training mean")

# ============================================================================
# H2: Flow magnitude effect - higher flow → higher absolute error
# ============================================================================
print("\n" + "=" * 70)
print("H2: FLOW MAGNITUDE - incidents have higher flow → higher absolute error")
print("=" * 70)

# Bin by target flow magnitude and compare MAE
all_target_h1 = targets[:, 0, :].flatten()
all_error_h1 = abs_errors[:, 0, :].flatten()
all_inc_mask = incident_mask.flatten()

# Only non-zero targets
nz = all_target_h1 != 0
flow_vals = all_target_h1[nz]
error_vals = all_error_h1[nz]
inc_vals_flat = all_inc_mask[nz]

# Flow bins
flow_bins = [0, 10, 30, 60, 100, 150, 200, 300, 500, 1000, float('inf')]
bin_labels = ['0-10', '10-30', '30-60', '60-100', '100-150', '150-200', '200-300', '300-500', '500-1k', '1k+']

print(f"\n  {'Flow Bin':<12} {'N_inc':>8} {'N_norm':>10} {'Inc MAE':>10} {'Norm MAE':>10} {'Diff':>8} {'Ratio':>8}")
print("  " + "-" * 70)

h2_results = []
for i in range(len(flow_bins) - 1):
    in_bin = (flow_vals >= flow_bins[i]) & (flow_vals < flow_bins[i+1])
    inc_in_bin = in_bin & inc_vals_flat
    norm_in_bin = in_bin & ~inc_vals_flat

    n_inc = inc_in_bin.sum()
    n_norm = norm_in_bin.sum()

    if n_inc > 10 and n_norm > 100:
        inc_mae = error_vals[inc_in_bin].mean()
        norm_mae = error_vals[norm_in_bin].mean()
        diff = inc_mae - norm_mae
        ratio = inc_mae / norm_mae
        print(f"  {bin_labels[i]:<12} {n_inc:>8,} {n_norm:>10,} {inc_mae:>10.2f} {norm_mae:>10.2f} {diff:>+8.2f} {ratio:>8.3f}")
        h2_results.append({
            'bin': bin_labels[i], 'n_inc': int(n_inc), 'n_norm': int(n_norm),
            'inc_mae': float(inc_mae), 'norm_mae': float(norm_mae),
            'diff': float(diff), 'ratio': float(ratio)
        })

# Flow-magnitude-controlled comparison
# Match incident pairs with normal pairs of SAME flow magnitude
print(f"\n  Flow-controlled comparison (matching flow magnitudes):")
rng = np.random.RandomState(42)

inc_flows = flow_vals[inc_vals_flat]
inc_errors_matched = error_vals[inc_vals_flat]

norm_flows = flow_vals[~inc_vals_flat]
norm_errors = error_vals[~inc_vals_flat]

# Bin-based matching
matched_inc_errors = []
matched_norm_errors = []
for i in range(len(flow_bins) - 1):
    inc_in_bin = (inc_flows >= flow_bins[i]) & (inc_flows < flow_bins[i+1])
    norm_in_bin = (norm_flows >= flow_bins[i]) & (norm_flows < flow_bins[i+1])

    n_inc_b = inc_in_bin.sum()
    n_norm_b = norm_in_bin.sum()

    if n_inc_b > 0 and n_norm_b > 0:
        matched_inc_errors.extend(inc_errors_matched[inc_in_bin].tolist())
        # Sample same number of normal pairs from this bin
        sampled = rng.choice(norm_errors[norm_in_bin], size=n_inc_b, replace=(n_inc_b > n_norm_b))
        matched_norm_errors.extend(sampled.tolist())

matched_inc_errors = np.array(matched_inc_errors)
matched_norm_errors = np.array(matched_norm_errors)
print(f"  Matched incident MAE: {matched_inc_errors.mean():.4f}")
print(f"  Matched normal MAE:   {matched_norm_errors.mean():.4f}")
print(f"  Difference (flow-controlled): {matched_inc_errors.mean() - matched_norm_errors.mean():.4f}")
print(f"  → If diff ≈ 0, flow magnitude is the sole driver")
print(f"  → If diff > 0, there's additional incident-specific degradation beyond flow magnitude")

# ============================================================================
# H3: Input contamination - incident in input window → worse prediction
# ============================================================================
print("\n" + "=" * 70)
print("H3: INPUT CONTAMINATION - incident starts in input window")
print("=" * 70)

# For each incident-affected (sample, node), check if incident starts within input window
inc_in_input_mask = incident_start_in_input & incident_mask
inc_not_in_input_mask = incident_mask & ~incident_start_in_input

n_in_input = inc_in_input_mask.sum()
n_not_in_input = inc_not_in_input_mask.sum()

mae_in_input = sample_node_masked_mae[inc_in_input_mask]
mae_not_in_input = sample_node_masked_mae[inc_not_in_input_mask]
mae_in_input = mae_in_input[~np.isnan(mae_in_input)]
mae_not_in_input = mae_not_in_input[~np.isnan(mae_not_in_input)]

print(f"  Incident starts in input window: {n_in_input:,} pairs ({n_in_input/incident_mask.sum()*100:.1f}%)")
print(f"  Incident NOT in input window:    {n_not_in_input:,} pairs ({n_not_in_input/incident_mask.sum()*100:.1f}%)")

if len(mae_in_input) > 0 and len(mae_not_in_input) > 0:
    print(f"\n  MAE when incident in input:     {mae_in_input.mean():.4f}")
    print(f"  MAE when incident NOT in input: {mae_not_in_input.mean():.4f}")
    print(f"  Normal (no incident):           {normal_vals_m.mean():.4f}")
    print(f"  → Input contamination effect: {mae_in_input.mean() - mae_not_in_input.mean():.4f}")

# ============================================================================
# H4: Temporal confounding - incidents cluster at hard-to-predict times
# ============================================================================
print("\n" + "=" * 70)
print("H4: TEMPORAL CONFOUNDING - incidents at rush hours?")
print("=" * 70)

# Time-of-day for each test sample
test_timesteps = np.arange(N_TEST_SAMPLES) + test_start + INPUT_LEN  # prediction start timestep
tod = test_timesteps % STEPS_PER_DAY  # 0-287 (5-min slots within day)
hour = tod / 12.0  # Convert to hours (0-24)

# For each sample, is it incident-affected for ANY meaningful node?
any_incident_per_sample = incident_mask[:, meaningful_nodes].any(axis=1)

# Hour distribution
inc_hours = hour[any_incident_per_sample]
normal_hours = hour[~any_incident_per_sample]

print(f"  Samples with any incident: {any_incident_per_sample.sum()} ({any_incident_per_sample.mean()*100:.1f}%)")
print(f"  Samples without incidents: {(~any_incident_per_sample).sum()} ({(~any_incident_per_sample).mean()*100:.1f}%)")
print(f"\n  Hour distribution:")
print(f"  Incident samples mean hour: {inc_hours.mean():.1f}")
print(f"  Normal samples mean hour:   {normal_hours.mean():.1f}")

# Per-hour MAE comparison (for meaningful nodes)
hour_bins = list(range(0, 25, 3))  # 0, 3, 6, 9, 12, 15, 18, 21, 24
hour_labels = [f"{h:02d}-{h+3:02d}" for h in range(0, 24, 3)]

print(f"\n  {'Hour':<10} {'Inc MAE':>10} {'Norm MAE':>10} {'Diff':>8} {'N_inc':>8} {'N_norm':>10}")
print("  " + "-" * 56)

h4_results = []
for i in range(len(hour_bins) - 1):
    in_hour = (hour >= hour_bins[i]) & (hour < hour_bins[i+1])

    # All (sample, node) pairs in this hour range
    inc_in_hour = incident_mask[in_hour][:, meaningful_nodes]
    mae_in_hour = sample_node_masked_mae[in_hour][:, meaningful_nodes]

    inc_vals_h = mae_in_hour[inc_in_hour]
    norm_vals_h = mae_in_hour[~inc_in_hour]
    inc_vals_h = inc_vals_h[~np.isnan(inc_vals_h)]
    norm_vals_h = norm_vals_h[~np.isnan(norm_vals_h)]

    if len(inc_vals_h) > 10:
        diff = inc_vals_h.mean() - norm_vals_h.mean()
        print(f"  {hour_labels[i]:<10} {inc_vals_h.mean():>10.3f} {norm_vals_h.mean():>10.3f} {diff:>+8.3f} {len(inc_vals_h):>8,} {len(norm_vals_h):>10,}")
        h4_results.append({
            'hour': hour_labels[i], 'inc_mae': float(inc_vals_h.mean()),
            'norm_mae': float(norm_vals_h.mean()), 'diff': float(diff),
            'n_inc': len(inc_vals_h), 'n_norm': len(norm_vals_h)
        })

# Time-controlled comparison: within each hour block, compute inc vs norm MAE
time_controlled_inc = []
time_controlled_norm = []
for i in range(len(hour_bins) - 1):
    in_hour = (hour >= hour_bins[i]) & (hour < hour_bins[i+1])
    inc_in_hour = incident_mask[in_hour][:, meaningful_nodes]
    mae_in_hour = sample_node_masked_mae[in_hour][:, meaningful_nodes]

    inc_vals_h = mae_in_hour[inc_in_hour]
    norm_vals_h = mae_in_hour[~inc_in_hour]
    inc_vals_h = inc_vals_h[~np.isnan(inc_vals_h)]
    norm_vals_h = norm_vals_h[~np.isnan(norm_vals_h)]

    if len(inc_vals_h) > 0:
        time_controlled_inc.extend(inc_vals_h.tolist())
        # Weighted sample to match incident count
        sampled = np.random.RandomState(42).choice(norm_vals_h, size=len(inc_vals_h), replace=len(inc_vals_h) > len(norm_vals_h))
        time_controlled_norm.extend(sampled.tolist())

time_controlled_inc = np.array(time_controlled_inc)
time_controlled_norm = np.array(time_controlled_norm)
print(f"\n  Time-controlled comparison:")
print(f"  Incident MAE: {time_controlled_inc.mean():.4f}")
print(f"  Normal MAE:   {time_controlled_norm.mean():.4f}")
print(f"  Diff: {time_controlled_inc.mean() - time_controlled_norm.mean():.4f}")

# ============================================================================
# H5: Heavy-tail errors - extreme errors drive the mean
# ============================================================================
print("\n" + "=" * 70)
print("H5: HEAVY-TAIL - extreme errors drive the incident mean")
print("=" * 70)

print(f"  Incident MAE distribution:")
print(f"    Mean:   {inc_vals_m.mean():.4f}")
print(f"    Median: {np.median(inc_vals_m):.4f}")
print(f"    P90:    {np.percentile(inc_vals_m, 90):.4f}")
print(f"    P95:    {np.percentile(inc_vals_m, 95):.4f}")
print(f"    P99:    {np.percentile(inc_vals_m, 99):.4f}")
print(f"    Max:    {inc_vals_m.max():.4f}")

print(f"\n  Normal MAE distribution:")
print(f"    Mean:   {normal_vals_m.mean():.4f}")
print(f"    Median: {np.median(normal_vals_m):.4f}")
print(f"    P90:    {np.percentile(normal_vals_m, 90):.4f}")
print(f"    P95:    {np.percentile(normal_vals_m, 95):.4f}")
print(f"    P99:    {np.percentile(normal_vals_m, 99):.4f}")
print(f"    Max:    {normal_vals_m.max():.4f}")

# Mean vs Median shift
mean_diff = inc_vals_m.mean() - normal_vals_m.mean()
median_diff = np.median(inc_vals_m) - np.median(normal_vals_m)
print(f"\n  Mean shift:   {mean_diff:.4f}")
print(f"  Median shift: {median_diff:.4f}")
print(f"  → If median shift << mean shift: heavy tail drives the difference")
print(f"  → If median shift ≈ mean shift: genuine distribution shift")

# Trimmed mean (exclude top 5%)
trim_pct = 0.05
inc_trimmed = np.sort(inc_vals_m)[:int(len(inc_vals_m) * (1 - trim_pct))]
norm_trimmed = np.sort(normal_vals_m)[:int(len(normal_vals_m) * (1 - trim_pct))]
print(f"\n  Trimmed mean (top 5% excluded):")
print(f"  Incident: {inc_trimmed.mean():.4f}")
print(f"  Normal:   {norm_trimmed.mean():.4f}")
print(f"  Diff:     {inc_trimmed.mean() - norm_trimmed.mean():.4f}")

# ============================================================================
# H6: Consistent direction - per-node sign test
# ============================================================================
print("\n" + "=" * 70)
print("H6: CONSISTENT DIRECTION - is the effect always worse?")
print("=" * 70)

# Per-node: does MAE increase for MOST nodes during incidents?
n_worse = 0
n_better = 0
n_same = 0
per_node_diffs = []

for n in nodes_with_incidents:
    inc_mask_n = incident_mask[:, n]
    if inc_mask_n.sum() < 5:
        continue

    inc_mae_n = sample_node_masked_mae[inc_mask_n, n]
    norm_mae_n = sample_node_masked_mae[~inc_mask_n, n]
    inc_mae_n = inc_mae_n[~np.isnan(inc_mae_n)]
    norm_mae_n = norm_mae_n[~np.isnan(norm_mae_n)]

    if len(inc_mae_n) > 0 and len(norm_mae_n) > 0:
        diff = inc_mae_n.mean() - norm_mae_n.mean()
        per_node_diffs.append(diff)
        if diff > 0.01:
            n_worse += 1
        elif diff < -0.01:
            n_better += 1
        else:
            n_same += 1

per_node_diffs = np.array(per_node_diffs)
print(f"  Nodes where incident MAE is WORSE:  {n_worse} ({n_worse/(n_worse+n_better+n_same)*100:.1f}%)")
print(f"  Nodes where incident MAE is BETTER: {n_better} ({n_better/(n_worse+n_better+n_same)*100:.1f}%)")
print(f"  Nodes where no meaningful change:   {n_same} ({n_same/(n_worse+n_better+n_same)*100:.1f}%)")

# Sign test
n_positive = (per_node_diffs > 0).sum()
n_total = len(per_node_diffs)
sign_test_p = stats.binomtest(n_positive, n_total, p=0.5).pvalue
print(f"\n  Sign test: {n_positive}/{n_total} nodes have positive diff")
print(f"  p-value (H0: random 50/50): {sign_test_p:.2e}")
print(f"  Mean per-node diff: {per_node_diffs.mean():.4f}")
print(f"  Median per-node diff: {np.median(per_node_diffs):.4f}")

# Distribution of per-node diffs
print(f"\n  Per-node MAE increase distribution:")
for pct in [10, 25, 50, 75, 90]:
    print(f"    P{pct}: {np.percentile(per_node_diffs, pct):.4f}")

# ============================================================================
# BONUS: Node-type decomposition
# ============================================================================
print("\n" + "=" * 70)
print("BONUS: NODE-TYPE DECOMPOSITION")
print("=" * 70)

# Load functional indices
functional_indices = set(np.load('datasets/xtraffic/SAN_BERNARDINO/functional_node_indices.npy'))

# Which types of nodes have incidents?
for label, idx_set in [('Dead', dead_indices), ('Major fail', major_fail_indices),
                        ('Partial fail', meaningful_set - functional_indices),
                        ('Functional', functional_indices)]:
    nodes_in_cat = [n for n in range(NUM_NODES) if n in idx_set]
    inc_pairs = incident_mask[:, nodes_in_cat].sum()
    total_pairs = len(nodes_in_cat) * N_TEST_SAMPLES

    if inc_pairs > 0:
        inc_mae_cat = sample_node_masked_mae[:, nodes_in_cat][incident_mask[:, nodes_in_cat]]
        norm_mae_cat = sample_node_masked_mae[:, nodes_in_cat][~incident_mask[:, nodes_in_cat]]
        inc_mae_cat = inc_mae_cat[~np.isnan(inc_mae_cat)]
        norm_mae_cat = norm_mae_cat[~np.isnan(norm_mae_cat)]

        diff = inc_mae_cat.mean() - norm_mae_cat.mean() if len(inc_mae_cat) > 0 else float('nan')
        print(f"  {label:<15}: {inc_pairs:>7,} pairs ({inc_pairs/total_pairs*100:.2f}%), "
              f"inc_MAE={inc_mae_cat.mean():.3f}, norm_MAE={norm_mae_cat.mean():.3f}, diff={diff:+.3f}")

# ============================================================================
# BONUS: Decomposition of 0.87% → overall MAE effect
# ============================================================================
print("\n" + "=" * 70)
print("BONUS: ARITHMETIC DECOMPOSITION")
print("=" * 70)

# If we compute overall MAE as weighted average of incident and normal:
# overall = (1-p) * normal + p * incident
# where p = fraction of incident pairs
p = n_affected / incident_mask.size
overall_expected = (1 - p) * normal_mae_vals.mean() + p * inc_mae_vals.mean()
print(f"  p (incident fraction): {p:.6f} ({p*100:.3f}%)")
print(f"  Normal MAE: {normal_mae_vals.mean():.6f}")
print(f"  Incident MAE: {inc_mae_vals.mean():.6f}")
print(f"  Expected overall: (1-{p:.4f})*{normal_mae_vals.mean():.4f} + {p:.4f}*{inc_mae_vals.mean():.4f} = {overall_expected:.6f}")
print(f"  Actual overall: {np.nanmean(sample_node_masked_mae):.6f}")
print(f"\n  Impact of incidents on overall MAE: {p * (inc_mae_vals.mean() - normal_mae_vals.mean()):.6f}")
print(f"  → Incidents shift overall MAE by only {p * (inc_mae_vals.mean() - normal_mae_vals.mean()):.4f}")
print(f"  → But WITHIN affected pairs, the increase is {inc_mae_vals.mean() - normal_mae_vals.mean():.4f} ({(inc_mae_vals.mean() - normal_mae_vals.mean())/normal_mae_vals.mean()*100:.2f}%)")

# ============================================================================
# BONUS: What fraction of overall error comes from incidents?
# ============================================================================
print("\n" + "=" * 70)
print("BONUS: INCIDENT CONTRIBUTION TO OVERALL ERROR")
print("=" * 70)

total_error = np.nansum(sample_node_masked_mae)
inc_error = np.nansum(sample_node_masked_mae[incident_mask])
norm_error = total_error - inc_error

# Count non-NaN pairs
total_nonnan = (~np.isnan(sample_node_masked_mae)).sum()
inc_nonnan = (~np.isnan(sample_node_masked_mae[incident_mask])).sum()
norm_nonnan = total_nonnan - inc_nonnan

print(f"  Total error (sum): {total_error:,.0f}")
print(f"  Incident error:    {inc_error:,.0f} ({inc_error/total_error*100:.2f}%)")
print(f"  Normal error:      {norm_error:,.0f} ({norm_error/total_error*100:.2f}%)")
print(f"\n  Incident pairs: {inc_nonnan/total_nonnan*100:.3f}% of data, {inc_error/total_error*100:.3f}% of total error")
print(f"  → Error contribution / data share = {(inc_error/total_error) / (inc_nonnan/total_nonnan):.3f}x")

# ============================================================================
# Visualization
# ============================================================================
print("\n[VIZ] Creating visualizations...")

fig = plt.figure(figsize=(24, 20))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

# 1: Error distribution comparison (H5)
ax = fig.add_subplot(gs[0, 0])
bins = np.linspace(0, 60, 80)
ax.hist(normal_vals_m, bins=bins, alpha=0.6, color='blue', density=True, label=f'Normal (n={len(normal_vals_m):,})')
ax.hist(inc_vals_m, bins=bins, alpha=0.6, color='red', density=True, label=f'Incident (n={len(inc_vals_m):,})')
ax.axvline(normal_vals_m.mean(), color='blue', linestyle='--', linewidth=2)
ax.axvline(inc_vals_m.mean(), color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Masked MAE per (sample, node)')
ax.set_ylabel('Density')
ax.set_title('H5: Error Distribution\nIncident vs Normal')
ax.legend(fontsize=8)
ax.set_xlim(0, 60)

# 2: Flow-binned MAE comparison (H2)
ax = fig.add_subplot(gs[0, 1])
if h2_results:
    x = range(len(h2_results))
    ax.bar([i-0.15 for i in x], [r['norm_mae'] for r in h2_results], width=0.3,
           color='blue', alpha=0.7, label='Normal')
    ax.bar([i+0.15 for i in x], [r['inc_mae'] for r in h2_results], width=0.3,
           color='red', alpha=0.7, label='Incident')
    ax.set_xticks(x)
    ax.set_xticklabels([r['bin'] for r in h2_results], rotation=45, ha='right')
    ax.set_xlabel('Target Flow Range')
    ax.set_ylabel('MAE')
    ax.set_title('H2: MAE by Flow Magnitude\nIncident vs Normal')
    ax.legend()

# 3: Per-hour MAE comparison (H4)
ax = fig.add_subplot(gs[0, 2])
if h4_results:
    x = range(len(h4_results))
    ax.bar([i-0.15 for i in x], [r['norm_mae'] for r in h4_results], width=0.3,
           color='blue', alpha=0.7, label='Normal')
    ax.bar([i+0.15 for i in x], [r['inc_mae'] for r in h4_results], width=0.3,
           color='red', alpha=0.7, label='Incident')
    ax.set_xticks(x)
    ax.set_xticklabels([r['hour'] for r in h4_results], rotation=45, ha='right')
    ax.set_xlabel('Hour Block')
    ax.set_ylabel('MAE')
    ax.set_title('H4: MAE by Time of Day\nIncident vs Normal')
    ax.legend()

# 4: Per-node diff distribution (H6)
ax = fig.add_subplot(gs[1, 0])
ax.hist(per_node_diffs, bins=50, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No change')
ax.axvline(per_node_diffs.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean={per_node_diffs.mean():.2f}')
ax.set_xlabel('Per-node MAE Increase (incident - normal)')
ax.set_ylabel('Number of Nodes')
ax.set_title(f'H6: Per-Node Consistency\n{n_worse} worse / {n_better} better / {n_same} neutral')
ax.legend()

# 5: CDF comparison (H5)
ax = fig.add_subplot(gs[1, 1])
sorted_inc = np.sort(inc_vals_m)
sorted_norm = np.sort(normal_vals_m)
ax.plot(sorted_norm, np.linspace(0, 1, len(sorted_norm)), 'b-', linewidth=2, label='Normal')
ax.plot(sorted_inc, np.linspace(0, 1, len(sorted_inc)), 'r-', linewidth=2, label='Incident')
ax.set_xlabel('Masked MAE')
ax.set_ylabel('Cumulative Probability')
ax.set_title('H5: CDF of MAE\nIncident vs Normal')
ax.legend()
ax.set_xlim(0, 60)
ax.grid(True, alpha=0.3)

# 6: Input contamination (H3)
ax = fig.add_subplot(gs[1, 2])
if len(mae_in_input) > 0 and len(mae_not_in_input) > 0:
    categories = ['Normal', 'Inc (not in input)', 'Inc (in input)']
    values = [normal_vals_m.mean(), mae_not_in_input.mean(), mae_in_input.mean()]
    colors = ['blue', 'orange', 'red']
    bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Masked MAE')
    ax.set_title('H3: Input Window Contamination\nEffect on Prediction')
    ax.set_ylim(0, max(values) * 1.15)

# 7: Flow z-score comparison (H1)
ax = fig.add_subplot(gs[2, 0])
bins_z = np.linspace(0, 5, 50)
ax.hist(flow_zscore_normal[:100000], bins=bins_z, alpha=0.6, color='blue', density=True, label='Normal')
ax.hist(flow_zscore_incident, bins=bins_z, alpha=0.6, color='red', density=True, label='Incident')
ax.set_xlabel('|Z-score| of Target Flow vs Training Mean')
ax.set_ylabel('Density')
ax.set_title('H1: Distribution Shift\nFlow Deviation from Training')
ax.legend()

# 8: Scatter - per-node incident count vs MAE increase
ax = fig.add_subplot(gs[2, 1])
per_node_df = pd.DataFrame({
    'node': nodes_with_incidents[:len(per_node_diffs)],
    'diff': per_node_diffs,
})
# Get incident counts per node
inc_counts = {}
for n in nodes_with_incidents:
    inc_counts[n] = incident_mask[:, n].sum()
per_node_df['n_inc_samples'] = per_node_df['node'].map(inc_counts)
ax.scatter(per_node_df['n_inc_samples'], per_node_df['diff'], alpha=0.5, s=15, c='steelblue')
ax.axhline(0, color='red', linestyle='--', alpha=0.7)
ax.set_xlabel('Number of Incident-Affected Samples')
ax.set_ylabel('MAE Increase')
ax.set_title('Incident Frequency vs MAE Increase')
corr, pval = stats.spearmanr(per_node_df['n_inc_samples'], per_node_df['diff'])
ax.annotate(f'Spearman r={corr:.3f}\np={pval:.4f}', xy=(0.05, 0.95), xycoords='axes fraction', va='top')

# 9: Summary text
ax = fig.add_subplot(gs[2, 2])
ax.axis('off')

summary_lines = [
    "HYPOTHESIS TEST RESULTS",
    "=" * 50,
    "",
    f"Incident pairs: {n_affected:,} ({n_affected/incident_mask.size*100:.3f}%)",
    f"Inc MAE: {inc_vals_m.mean():.3f}, Normal MAE: {normal_vals_m.mean():.3f}",
    f"Diff: +{inc_vals_m.mean()-normal_vals_m.mean():.3f}, Cohen's d: {cohen_d:.4f}",
    "",
    "H1 (Distribution shift):",
    f"  Flow |z| inc={flow_zscore_incident.mean():.3f} vs norm={flow_zscore_normal.mean():.3f}",
    "",
    "H2 (Flow magnitude):",
    f"  Flow-controlled diff: {matched_inc_errors.mean()-matched_norm_errors.mean():.3f}",
    f"  (vs raw diff: {inc_vals_m.mean()-normal_vals_m.mean():.3f})",
    "",
    "H3 (Input contamination):",
    f"  In input: {mae_in_input.mean():.3f}",
    f"  Not in input: {mae_not_in_input.mean():.3f}",
    "",
    f"H5 (Heavy tail): median shift={median_diff:.3f} vs mean shift={mean_diff:.3f}",
    "",
    f"H6 (Consistency): {n_worse} worse / {n_better} better",
    f"  Sign test p={sign_test_p:.2e}",
]

ax.text(0.05, 0.95, '\n'.join(summary_lines), transform=ax.transAxes,
        fontsize=8, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.savefig(f'{OUTPUT_DIR}/why_incident_mae_matters.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}/why_incident_mae_matters.png")

# Save detailed results
results_json = {
    'baseline': {
        'inc_mae': float(inc_vals_m.mean()),
        'normal_mae': float(normal_vals_m.mean()),
        'diff': float(inc_vals_m.mean() - normal_vals_m.mean()),
        'cohens_d': float(cohen_d),
        't_stat': float(t_stat),
        'p_value': float(p_val),
    },
    'H1_distribution_shift': {
        'flow_zscore_incident': float(flow_zscore_incident.mean()),
        'flow_zscore_normal': float(flow_zscore_normal.mean()),
        'flow_mean_incident': float(inc_flow_nz.mean()),
        'flow_mean_normal': float(normal_flow_nz.mean()),
    },
    'H2_flow_magnitude': {
        'flow_controlled_diff': float(matched_inc_errors.mean() - matched_norm_errors.mean()),
        'raw_diff': float(inc_vals_m.mean() - normal_vals_m.mean()),
        'per_bin': h2_results,
    },
    'H3_input_contamination': {
        'mae_in_input': float(mae_in_input.mean()) if len(mae_in_input) > 0 else None,
        'mae_not_in_input': float(mae_not_in_input.mean()) if len(mae_not_in_input) > 0 else None,
        'n_in_input': int(n_in_input),
        'n_not_in_input': int(n_not_in_input),
    },
    'H4_temporal_confounding': {
        'time_controlled_diff': float(time_controlled_inc.mean() - time_controlled_norm.mean()),
        'per_hour': h4_results,
    },
    'H5_heavy_tail': {
        'mean_shift': float(mean_diff),
        'median_shift': float(median_diff),
        'trimmed_mean_diff': float(inc_trimmed.mean() - norm_trimmed.mean()),
    },
    'H6_consistency': {
        'n_worse': int(n_worse),
        'n_better': int(n_better),
        'n_same': int(n_same),
        'sign_test_p': float(sign_test_p),
        'mean_per_node_diff': float(per_node_diffs.mean()),
        'median_per_node_diff': float(np.median(per_node_diffs)),
    },
}

with open(f'{OUTPUT_DIR}/why_incident_mae_results.json', 'w') as f:
    json.dump(results_json, f, indent=2)
print(f"  Saved: {OUTPUT_DIR}/why_incident_mae_results.json")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
