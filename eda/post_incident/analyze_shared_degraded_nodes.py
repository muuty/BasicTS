"""
Shared Degraded Nodes Deep Analysis
====================================
16 nodes that are in top-20 degraded for BOTH STAEformer and STGCN.
Analyze their data patterns and model prediction behavior.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

OUTPUT_DIR = 'eda/post_incident'
DATA_PATH = 'datasets/xtraffic/SAN_BERNARDINO/data.dat'
INCIDENT_PATH = 'datasets/xtraffic/SAN_BERNARDINO/incident_metadata_2023.csv'
ADJ_PATH = 'datasets/xtraffic/SAN_BERNARDINO/adj_mx.pkl'

STAE_PRED_DIR = 'checkpoints/ContextContrastive_baseline_3mo/xtraffic_SAN_BERNARDINO_30_12_12/040af4f5bcb37097bc5263d7f350174b/test_results'
STGCN_PRED_DIR = sorted([d for d in __import__('glob').glob('checkpoints/STGCN/SAN_BERNARDINO_*/*/test_results') if os.path.exists(d)], key=os.path.getmtime)[-1]

NUM_NODES = 893
INPUT_LEN = 12
OUTPUT_LEN = 12
N_TEST_SAMPLES = 5233
DATA_RANGE = (0, 26280)
TRAIN_VAL_TEST_RATIO = [0.6, 0.2, 0.2]
STEPS_PER_DAY = 288
RECOVERY_BUFFER = 12

# Shared top-20 degraded nodes
SHARED_NODES = sorted([5, 12, 214, 249, 303, 330, 352, 363, 372, 443, 495, 557, 576, 653, 663, 699])

print("=" * 70)
print(f"Deep Analysis of {len(SHARED_NODES)} Shared Degraded Nodes")
print(f"Nodes: {SHARED_NODES}")
print("=" * 70)

# ============================================================================
# 1. Load everything
# ============================================================================
print("\n[1] Loading data...")

data = np.memmap(DATA_PATH, dtype='float32', mode='r').reshape(-1, NUM_NODES, 5)[:DATA_RANGE[1]]
flow, occ, speed, tod, dow = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3], data[:,:,4]

total_len = DATA_RANGE[1]
test_start = int(total_len * (TRAIN_VAL_TEST_RATIO[0] + TRAIN_VAL_TEST_RATIO[1]))
train_end = int(total_len * TRAIN_VAL_TEST_RATIO[0])

# Predictions
stae_pred = np.memmap(f'{STAE_PRED_DIR}/predictions.npy', dtype='float32', mode='r').reshape(N_TEST_SAMPLES, OUTPUT_LEN, NUM_NODES)
stae_tgt = np.memmap(f'{STAE_PRED_DIR}/targets.npy', dtype='float32', mode='r').reshape(N_TEST_SAMPLES, OUTPUT_LEN, NUM_NODES)
stgcn_pred = np.memmap(f'{STGCN_PRED_DIR}/predictions.npy', dtype='float32', mode='r').reshape(N_TEST_SAMPLES, OUTPUT_LEN, NUM_NODES)
stgcn_tgt = np.memmap(f'{STGCN_PRED_DIR}/targets.npy', dtype='float32', mode='r').reshape(N_TEST_SAMPLES, OUTPUT_LEN, NUM_NODES)

# Incidents
incidents = pd.read_csv(INCIDENT_PATH)
incidents = incidents[incidents['original_start_slot'] < DATA_RANGE[1]].copy()
pred_window_start = test_start + INPUT_LEN
pred_window_end = test_start + N_TEST_SAMPLES - 1 + INPUT_LEN + OUTPUT_LEN - 1
incidents['incident_end'] = incidents['original_start_slot'] + incidents['duration_slots'] + RECOVERY_BUFFER
test_incidents = incidents[
    (incidents['incident_end'] >= pred_window_start) &
    (incidents['original_start_slot'] <= pred_window_end)
].copy()

# Build per-node incident mask
incident_mask = np.zeros((N_TEST_SAMPLES, NUM_NODES), dtype=bool)
for _, inc in test_incidents.iterrows():
    s = int(inc['sensor_idx'])
    inc_start = int(inc['original_start_slot'])
    inc_end = int(inc['incident_end'])
    t_min = max(0, inc_start - test_start - INPUT_LEN - OUTPUT_LEN + 1)
    t_max = min(N_TEST_SAMPLES - 1, inc_end - test_start - INPUT_LEN)
    if t_min <= t_max:
        incident_mask[t_min:t_max+1, s] = True

# Adjacency
with open(ADJ_PATH, 'rb') as f:
    adj_data = pickle.load(f, encoding='latin1')
adj_mx = adj_data[-1] if isinstance(adj_data, list) else np.array(adj_data)

print(f"  STGCN pred dir: {STGCN_PRED_DIR}")

# ============================================================================
# 2. Per-node data pattern summary
# ============================================================================
print("\n[2] Data pattern analysis for shared degraded nodes...")

# Also pick 16 non-degraded functional nodes for comparison
all_functional = [n for n in range(NUM_NODES) if n not in set(np.load('datasets/xtraffic/SAN_BERNARDINO/dead_indices.npy')) and n not in set(np.load('datasets/xtraffic/SAN_BERNARDINO/major_fail_indices.npy'))]
non_degraded_candidates = [n for n in all_functional if n not in SHARED_NODES and incident_mask[:, n].sum() > 0]
# Pick nodes with similar incident counts but low degradation
stae_errors = np.abs(stae_pred - stae_tgt)
node_inc_mae = {}
for n in non_degraded_candidates:
    inc_mask_n = incident_mask[:, n]
    if inc_mask_n.sum() < 5:
        continue
    inc_mae = stae_errors[:, :, n][inc_mask_n].mean()
    norm_mae = stae_errors[:, :, n][~inc_mask_n].mean()
    node_inc_mae[n] = inc_mae - norm_mae
# Pick 16 with lowest degradation
resilient_nodes = sorted(node_inc_mae, key=node_inc_mae.get)[:16]

print(f"  Comparison resilient nodes: {resilient_nodes}")

node_profiles = []
for node_idx in SHARED_NODES + resilient_nodes:
    group = 'degraded' if node_idx in SHARED_NODES else 'resilient'
    train_flow = flow[:train_end, node_idx]
    test_flow = flow[test_start:, node_idx]

    # Daily pattern (average flow by time-of-day)
    daily_pattern = np.zeros(STEPS_PER_DAY)
    for t in range(train_end):
        slot = int(tod[t, node_idx] * STEPS_PER_DAY) % STEPS_PER_DAY
        daily_pattern[slot] += flow[t, node_idx]
    counts = np.zeros(STEPS_PER_DAY)
    for t in range(train_end):
        slot = int(tod[t, node_idx] * STEPS_PER_DAY) % STEPS_PER_DAY
        counts[slot] += 1
    daily_pattern = np.where(counts > 0, daily_pattern / counts, 0)

    # Flow statistics
    peak_flow = daily_pattern.max()
    offpeak_flow = daily_pattern[daily_pattern > 0].min() if (daily_pattern > 0).any() else 0
    peak_offpeak_ratio = peak_flow / offpeak_flow if offpeak_flow > 0 else 0

    # Variability
    flow_cv = train_flow.std() / train_flow.mean() if train_flow.mean() > 0 else 0

    # Incident info for this node
    node_incs = test_incidents[test_incidents['sensor_idx'] == node_idx]

    node_profiles.append({
        'node_idx': node_idx,
        'group': group,
        'mean_flow': train_flow.mean(),
        'std_flow': train_flow.std(),
        'cv_flow': flow_cv,
        'max_flow': train_flow.max(),
        'mean_occ': occ[:train_end, node_idx].mean(),
        'mean_speed': speed[:train_end, node_idx].mean(),
        'zero_rate': (train_flow == 0).mean(),
        'peak_flow': peak_flow,
        'offpeak_flow': offpeak_flow,
        'peak_ratio': peak_offpeak_ratio,
        'n_incidents': len(node_incs),
        'n_incident_samples': int(incident_mask[:, node_idx].sum()),
        'incident_types': ','.join(node_incs['incident_type'].value_counts().head(3).index.tolist()) if len(node_incs) > 0 else '',
        'mean_duration': node_incs['duration_slots'].mean() if len(node_incs) > 0 else 0,
        'degree': int((adj_mx[node_idx] > 0).sum()),
    })

df_profiles = pd.DataFrame(node_profiles)
print("\n  Degraded nodes summary:")
print(df_profiles[df_profiles['group']=='degraded'][['node_idx','mean_flow','cv_flow','mean_speed','peak_ratio','n_incidents','mean_duration','incident_types']].to_string(index=False))

# ============================================================================
# 3. Prediction behavior analysis
# ============================================================================
print("\n[3] Prediction behavior during incidents...")

pred_analysis = []
for node_idx in SHARED_NODES:
    inc_mask_n = incident_mask[:, node_idx]
    norm_mask_n = ~inc_mask_n

    for model_name, preds, tgts in [('STAEformer', stae_pred, stae_tgt), ('STGCN', stgcn_pred, stgcn_tgt)]:
        # Normal period
        norm_preds = preds[norm_mask_n, :, node_idx]  # (n_normal, 12)
        norm_tgts = tgts[norm_mask_n, :, node_idx]
        # Incident period
        inc_preds = preds[inc_mask_n, :, node_idx]
        inc_tgts = tgts[inc_mask_n, :, node_idx]

        # Masked (exclude zero targets)
        norm_nz = norm_tgts != 0
        inc_nz = inc_tgts != 0

        norm_mae = np.abs(norm_preds[norm_nz] - norm_tgts[norm_nz]).mean() if norm_nz.any() else np.nan
        inc_mae = np.abs(inc_preds[inc_nz] - inc_tgts[inc_nz]).mean() if inc_nz.any() else np.nan

        # Prediction statistics
        norm_pred_mean = norm_preds[norm_nz].mean() if norm_nz.any() else np.nan
        inc_pred_mean = inc_preds[inc_nz].mean() if inc_nz.any() else np.nan
        norm_tgt_mean = norm_tgts[norm_nz].mean() if norm_nz.any() else np.nan
        inc_tgt_mean = inc_tgts[inc_nz].mean() if inc_nz.any() else np.nan

        # Prediction bias (over/under-prediction)
        norm_bias = (norm_preds[norm_nz] - norm_tgts[norm_nz]).mean() if norm_nz.any() else np.nan
        inc_bias = (inc_preds[inc_nz] - inc_tgts[inc_nz]).mean() if inc_nz.any() else np.nan

        # Does model over-predict or under-predict during incidents?
        pred_analysis.append({
            'node_idx': node_idx,
            'model': model_name,
            'normal_mae': norm_mae,
            'incident_mae': inc_mae,
            'increase': inc_mae - norm_mae if not (np.isnan(inc_mae) or np.isnan(norm_mae)) else np.nan,
            'normal_pred_mean': norm_pred_mean,
            'incident_pred_mean': inc_pred_mean,
            'normal_tgt_mean': norm_tgt_mean,
            'incident_tgt_mean': inc_tgt_mean,
            'normal_bias': norm_bias,
            'incident_bias': inc_bias,
            'bias_shift': inc_bias - norm_bias if not (np.isnan(inc_bias) or np.isnan(norm_bias)) else np.nan,
        })

df_pred = pd.DataFrame(pred_analysis)

print("\n  STAEformer prediction behavior:")
stae_df = df_pred[df_pred['model']=='STAEformer']
print(stae_df[['node_idx','normal_mae','incident_mae','increase','normal_bias','incident_bias','bias_shift']].to_string(index=False))

print("\n  STGCN prediction behavior:")
stgcn_df = df_pred[df_pred['model']=='STGCN']
print(stgcn_df[['node_idx','normal_mae','incident_mae','increase','normal_bias','incident_bias','bias_shift']].to_string(index=False))

# Key question: are models over-predicting (predicting normal traffic when incident drops flow)?
print("\n  Bias analysis summary:")
print(f"  STAEformer - Normal bias: {stae_df['normal_bias'].mean():.2f}, Incident bias: {stae_df['incident_bias'].mean():.2f}, Shift: {stae_df['bias_shift'].mean():.2f}")
print(f"  STGCN      - Normal bias: {stgcn_df['normal_bias'].mean():.2f}, Incident bias: {stgcn_df['incident_bias'].mean():.2f}, Shift: {stgcn_df['bias_shift'].mean():.2f}")

# ============================================================================
# 4. Temporal zoom: actual predictions around specific incidents
# ============================================================================
print("\n[4] Temporal prediction profiles around incidents...")

# For each shared node, find incidents and plot actual pred vs target
window_before = 24  # 2 hours before
window_after = 36   # 3 hours after

temporal_data = []
for node_idx in SHARED_NODES:
    node_incs = test_incidents[test_incidents['sensor_idx'] == node_idx]
    for _, inc in node_incs.iterrows():
        inc_start = int(inc['original_start_slot'])
        inc_dur = int(inc['duration_slots'])
        inc_type = inc['incident_type']

        for rel_t in range(-window_before, window_after):
            target_timestep = inc_start + rel_t
            sample_t = target_timestep - test_start - INPUT_LEN
            if sample_t < 0 or sample_t >= N_TEST_SAMPLES:
                continue

            # Use horizon 0 (first prediction step)
            temporal_data.append({
                'node_idx': node_idx,
                'relative_time': rel_t,
                'target': float(stae_tgt[sample_t, 0, node_idx]),
                'stae_pred': float(stae_pred[sample_t, 0, node_idx]),
                'stgcn_pred': float(stgcn_pred[sample_t, 0, node_idx]),
                'stae_error': float(np.abs(stae_pred[sample_t, 0, node_idx] - stae_tgt[sample_t, 0, node_idx])),
                'stgcn_error': float(np.abs(stgcn_pred[sample_t, 0, node_idx] - stgcn_tgt[sample_t, 0, node_idx])),
                'incident_type': inc_type,
                'incident_duration': inc_dur,
            })

df_temporal = pd.DataFrame(temporal_data)
print(f"  Temporal data points: {len(df_temporal)}")

# Average temporal profile
avg_temporal = df_temporal.groupby('relative_time').agg(
    mean_target=('target', 'mean'),
    mean_stae_pred=('stae_pred', 'mean'),
    mean_stgcn_pred=('stgcn_pred', 'mean'),
    mean_stae_error=('stae_error', 'mean'),
    mean_stgcn_error=('stgcn_error', 'mean'),
    count=('target', 'count'),
).reset_index()

# ============================================================================
# 5. What happens to the actual flow during incidents?
# ============================================================================
print("\n[5] Raw flow behavior during incidents...")

flow_temporal = []
for node_idx in SHARED_NODES:
    node_incs = test_incidents[test_incidents['sensor_idx'] == node_idx]
    # Also get train-period average daily flow for this node
    train_daily = np.zeros(STEPS_PER_DAY)
    train_counts = np.zeros(STEPS_PER_DAY)
    for t in range(train_end):
        slot = int(tod[t, node_idx] * STEPS_PER_DAY) % STEPS_PER_DAY
        train_daily[slot] += flow[t, node_idx]
        train_counts[slot] += 1
    train_daily = np.where(train_counts > 0, train_daily / train_counts, 0)

    for _, inc in node_incs.iterrows():
        inc_start = int(inc['original_start_slot'])
        for rel_t in range(-window_before, window_after):
            abs_t = inc_start + rel_t
            if abs_t < 0 or abs_t >= DATA_RANGE[1]:
                continue
            slot = int(tod[abs_t, node_idx] * STEPS_PER_DAY) % STEPS_PER_DAY
            expected_flow = train_daily[slot]
            actual_flow = flow[abs_t, node_idx]
            flow_temporal.append({
                'node_idx': node_idx,
                'relative_time': rel_t,
                'actual_flow': actual_flow,
                'expected_flow': expected_flow,
                'flow_deviation': actual_flow - expected_flow,
                'flow_deviation_pct': (actual_flow - expected_flow) / expected_flow * 100 if expected_flow > 0 else 0,
            })

df_flow = pd.DataFrame(flow_temporal)
avg_flow = df_flow.groupby('relative_time').agg(
    mean_actual=('actual_flow', 'mean'),
    mean_expected=('expected_flow', 'mean'),
    mean_deviation=('flow_deviation', 'mean'),
    mean_deviation_pct=('flow_deviation_pct', 'mean'),
).reset_index()

print(f"  Peak deviation: {avg_flow['mean_deviation'].min():.1f} vehicles at t={avg_flow.loc[avg_flow['mean_deviation'].idxmin(), 'relative_time']}")
print(f"  Peak deviation %: {avg_flow['mean_deviation_pct'].min():.1f}% at t={avg_flow.loc[avg_flow['mean_deviation_pct'].idxmin(), 'relative_time']}")

# ============================================================================
# 6. Visualizations
# ============================================================================
print("\n[6] Creating visualizations...")

fig = plt.figure(figsize=(28, 36))
gs = gridspec.GridSpec(7, 3, figure=fig, hspace=0.35, wspace=0.3)

# --- Row 1: Data patterns ---

# 6.1: Degraded vs Resilient node characteristics
ax = fig.add_subplot(gs[0, 0])
features = ['mean_flow', 'cv_flow', 'mean_speed', 'peak_ratio', 'mean_occ']
deg_vals = [df_profiles[df_profiles['group']=='degraded'][f].mean() for f in features]
res_vals = [df_profiles[df_profiles['group']=='resilient'][f].mean() for f in features]
# Normalize for comparison
max_vals = [max(d, r) if max(d, r) > 0 else 1 for d, r in zip(deg_vals, res_vals)]
x = np.arange(len(features))
width = 0.35
ax.bar(x - width/2, [d/m for d,m in zip(deg_vals, max_vals)], width, label=f'Degraded (n={len(SHARED_NODES)})', color='red', alpha=0.7)
ax.bar(x + width/2, [r/m for r,m in zip(res_vals, max_vals)], width, label=f'Resilient (n={len(resilient_nodes)})', color='green', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(features, rotation=30, ha='right')
ax.set_ylabel('Normalized Value')
ax.set_title('Degraded vs Resilient: Traffic Characteristics')
ax.legend(fontsize=8)

# 6.2: Incident frequency comparison
ax = fig.add_subplot(gs[0, 1])
deg_inc = df_profiles[df_profiles['group']=='degraded']['n_incidents'].values
res_inc = df_profiles[df_profiles['group']=='resilient']['n_incidents'].values
ax.hist(deg_inc, bins=range(0, max(deg_inc.max(), res_inc.max())+2), alpha=0.6, color='red', label='Degraded')
ax.hist(res_inc, bins=range(0, max(deg_inc.max(), res_inc.max())+2), alpha=0.6, color='green', label='Resilient')
ax.set_xlabel('Number of Incidents')
ax.set_ylabel('Number of Nodes')
ax.set_title(f'Incident Frequency: Degraded={deg_inc.mean():.1f} vs Resilient={res_inc.mean():.1f}')
ax.legend()

# 6.3: Per-node degradation bar chart (both models side by side)
ax = fig.add_subplot(gs[0, 2])
stae_inc = stae_df.set_index('node_idx')['increase']
stgcn_inc = stgcn_df.set_index('node_idx')['increase']
x = np.arange(len(SHARED_NODES))
ax.bar(x - 0.2, [stae_inc.get(n, 0) for n in SHARED_NODES], 0.4, label='STAEformer', color='blue', alpha=0.7)
ax.bar(x + 0.2, [stgcn_inc.get(n, 0) for n in SHARED_NODES], 0.4, label='STGCN', color='red', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels([f'N{n}' for n in SHARED_NODES], rotation=45, ha='right', fontsize=7)
ax.set_ylabel('Masked MAE Increase')
ax.set_title('Per-Node Degradation: Both Models')
ax.legend(fontsize=8)

# --- Row 2: Prediction behavior ---

# 6.4: Prediction bias comparison
ax = fig.add_subplot(gs[1, 0])
ax.scatter(stae_df['normal_bias'], stae_df['incident_bias'], c='blue', s=50, alpha=0.7, label='STAEformer', zorder=5)
ax.scatter(stgcn_df['normal_bias'], stgcn_df['incident_bias'], c='red', s=50, alpha=0.7, label='STGCN', zorder=5)
max_b = max(abs(df_pred['normal_bias']).max(), abs(df_pred['incident_bias']).max()) * 1.1
ax.plot([-max_b, max_b], [-max_b, max_b], 'k--', alpha=0.3)
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
ax.axvline(x=0, color='gray', linestyle=':', alpha=0.3)
ax.set_xlabel('Normal Period Bias (pred - target)')
ax.set_ylabel('Incident Period Bias (pred - target)')
ax.set_title('Prediction Bias: Normal vs Incident')
ax.legend()

# 6.5: Target flow change during incidents
ax = fig.add_subplot(gs[1, 1])
ax.scatter(stae_df['normal_tgt_mean'], stae_df['incident_tgt_mean'], c='blue', s=50, alpha=0.7, label='STAEformer target', zorder=5)
max_t = max(stae_df['normal_tgt_mean'].max(), stae_df['incident_tgt_mean'].max()) * 1.1
ax.plot([0, max_t], [0, max_t], 'k--', alpha=0.3, label='No change')
ax.set_xlabel('Normal Period Mean Target')
ax.set_ylabel('Incident Period Mean Target')
ax.set_title('Target Flow: Normal vs Incident')
ax.legend()

# 6.6: Prediction vs Target during incidents
ax = fig.add_subplot(gs[1, 2])
ax.scatter(stae_df['incident_tgt_mean'], stae_df['incident_pred_mean'], c='blue', s=50, alpha=0.7, label='STAEformer')
ax.scatter(stgcn_df['incident_tgt_mean'], stgcn_df['incident_pred_mean'], c='red', s=50, alpha=0.7, label='STGCN')
max_v = max(df_pred['incident_tgt_mean'].max(), df_pred['incident_pred_mean'].max()) * 1.1
ax.plot([0, max_v], [0, max_v], 'k--', alpha=0.3, label='Perfect prediction')
ax.set_xlabel('Incident Period Mean Target')
ax.set_ylabel('Incident Period Mean Prediction')
ax.set_title('Predictions vs Targets During Incidents')
ax.legend()

# --- Row 3: Temporal profiles ---

# 6.7: Average prediction trajectory around incidents
ax = fig.add_subplot(gs[2, :2])
ax.plot(avg_temporal['relative_time'], avg_temporal['mean_target'], 'k-', linewidth=2.5, label='Target (actual flow)')
ax.plot(avg_temporal['relative_time'], avg_temporal['mean_stae_pred'], 'b--', linewidth=2, label='STAEformer prediction')
ax.plot(avg_temporal['relative_time'], avg_temporal['mean_stgcn_pred'], 'r--', linewidth=2, label='STGCN prediction')
ax.axvline(x=0, color='red', linestyle=':', linewidth=2, alpha=0.7, label='Incident start')
ax.fill_betweenx([ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0, ax.get_ylim()[1]], 0, 12, alpha=0.1, color='red', label='Avg incident duration')
ax.set_xlabel('Relative Time (5-min slots from incident start)')
ax.set_ylabel('Flow Value')
ax.set_title('Average Prediction Trajectory Around Incidents (16 Shared Degraded Nodes)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 6.8: Error trajectory
ax = fig.add_subplot(gs[2, 2])
ax.plot(avg_temporal['relative_time'], avg_temporal['mean_stae_error'], 'b-', linewidth=2, label='STAEformer error')
ax.plot(avg_temporal['relative_time'], avg_temporal['mean_stgcn_error'], 'r-', linewidth=2, label='STGCN error')
ax.axvline(x=0, color='red', linestyle=':', linewidth=2, alpha=0.7)
ax.set_xlabel('Relative Time')
ax.set_ylabel('Absolute Error')
ax.set_title('Error Trajectory Around Incidents')
ax.legend()
ax.grid(True, alpha=0.3)

# --- Row 4: Raw flow analysis ---

# 6.9: Expected vs actual flow around incidents
ax = fig.add_subplot(gs[3, :2])
ax.plot(avg_flow['relative_time'], avg_flow['mean_expected'], 'g--', linewidth=2, label='Expected (train avg)')
ax.plot(avg_flow['relative_time'], avg_flow['mean_actual'], 'k-', linewidth=2.5, label='Actual flow')
ax.fill_between(avg_flow['relative_time'], avg_flow['mean_expected'], avg_flow['mean_actual'],
                where=avg_flow['mean_actual'] < avg_flow['mean_expected'], alpha=0.3, color='red', label='Flow drop')
ax.fill_between(avg_flow['relative_time'], avg_flow['mean_expected'], avg_flow['mean_actual'],
                where=avg_flow['mean_actual'] > avg_flow['mean_expected'], alpha=0.3, color='blue', label='Flow surge')
ax.axvline(x=0, color='red', linestyle=':', linewidth=2, alpha=0.7, label='Incident start')
ax.set_xlabel('Relative Time (5-min slots from incident start)')
ax.set_ylabel('Flow')
ax.set_title('Expected vs Actual Flow Around Incidents (16 Degraded Nodes)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 6.10: Flow deviation %
ax = fig.add_subplot(gs[3, 2])
ax.plot(avg_flow['relative_time'], avg_flow['mean_deviation_pct'], 'purple', linewidth=2)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='red', linestyle=':', linewidth=2, alpha=0.7)
ax.fill_between(avg_flow['relative_time'], 0, avg_flow['mean_deviation_pct'],
                where=avg_flow['mean_deviation_pct'] < 0, alpha=0.3, color='red')
ax.set_xlabel('Relative Time')
ax.set_ylabel('Flow Deviation (%)')
ax.set_title('Flow Deviation from Expected Pattern')
ax.grid(True, alpha=0.3)

# --- Row 5: Individual node examples ---

# Pick 4 most interesting nodes (highest degradation)
example_nodes = [5, 576, 249, 303]
for i, node_idx in enumerate(example_nodes):
    ax = fig.add_subplot(gs[4, i] if i < 3 else gs[5, 0])
    node_temporal = df_temporal[df_temporal['node_idx'] == node_idx]
    if len(node_temporal) == 0:
        continue
    node_avg = node_temporal.groupby('relative_time').mean(numeric_only=True)
    ax.plot(node_avg.index, node_avg['target'], 'k-', linewidth=2, label='Target')
    ax.plot(node_avg.index, node_avg['stae_pred'], 'b--', linewidth=1.5, label='STAEformer')
    ax.plot(node_avg.index, node_avg['stgcn_pred'], 'r--', linewidth=1.5, label='STGCN')
    ax.axvline(x=0, color='red', linestyle=':', alpha=0.7)
    node_incs = test_incidents[test_incidents['sensor_idx'] == node_idx]
    inc_info = f"{len(node_incs)} incidents, types: {','.join(node_incs['incident_type'].unique()[:3])}"
    stae_inc_val = stae_df[stae_df['node_idx']==node_idx]['increase'].values[0]
    stgcn_inc_val = stgcn_df[stgcn_df['node_idx']==node_idx]['increase'].values[0]
    ax.set_title(f'Node {node_idx}: STAE+{stae_inc_val:.1f}, STGCN+{stgcn_inc_val:.1f}\n({inc_info})', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

# --- Row 6: Horizon-wise comparison ---

ax = fig.add_subplot(gs[5, 1])
stae_norm_h, stae_inc_h, stgcn_norm_h, stgcn_inc_h = [], [], [], []
for h in range(OUTPUT_LEN):
    stae_n, stae_i, stgcn_n, stgcn_i = [], [], [], []
    for nid in SHARED_NODES:
        inc_m = incident_mask[:, nid]
        stae_n.append(np.abs(stae_pred[~inc_m, h, nid] - stae_tgt[~inc_m, h, nid]).mean())
        stae_i.append(np.abs(stae_pred[inc_m, h, nid] - stae_tgt[inc_m, h, nid]).mean())
        stgcn_n.append(np.abs(stgcn_pred[~inc_m, h, nid] - stgcn_tgt[~inc_m, h, nid]).mean())
        stgcn_i.append(np.abs(stgcn_pred[inc_m, h, nid] - stgcn_tgt[inc_m, h, nid]).mean())
    stae_norm_h.append(np.mean(stae_n))
    stae_inc_h.append(np.mean(stae_i))
    stgcn_norm_h.append(np.mean(stgcn_n))
    stgcn_inc_h.append(np.mean(stgcn_i))

horizons = range(1, OUTPUT_LEN + 1)
ax.plot(horizons, stae_norm_h, 'b-o', linewidth=2, markersize=4, label='STAE normal')
ax.plot(horizons, stae_inc_h, 'b--s', linewidth=2, markersize=4, label='STAE incident')
ax.plot(horizons, stgcn_norm_h, 'r-o', linewidth=2, markersize=4, label='STGCN normal')
ax.plot(horizons, stgcn_inc_h, 'r--s', linewidth=2, markersize=4, label='STGCN incident')
ax.set_xlabel('Prediction Horizon')
ax.set_ylabel('MAE')
ax.set_title('Horizon-wise: Normal vs Incident\n(16 Shared Degraded Nodes)')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# --- Summary panel ---
ax = fig.add_subplot(gs[5, 2])
ax.axis('off')
deg_prof = df_profiles[df_profiles['group']=='degraded']
res_prof = df_profiles[df_profiles['group']=='resilient']
summary = [
    "SHARED DEGRADED NODES SUMMARY",
    "=" * 42,
    f"Nodes analyzed: {len(SHARED_NODES)}",
    f"Comparison resilient nodes: {len(resilient_nodes)}",
    "",
    "TRAFFIC CHARACTERISTICS (degraded/resilient):",
    f"  Mean flow:   {deg_prof['mean_flow'].mean():.0f} / {res_prof['mean_flow'].mean():.0f}",
    f"  CV flow:     {deg_prof['cv_flow'].mean():.3f} / {res_prof['cv_flow'].mean():.3f}",
    f"  Peak ratio:  {deg_prof['peak_ratio'].mean():.1f} / {res_prof['peak_ratio'].mean():.1f}",
    f"  Mean speed:  {deg_prof['mean_speed'].mean():.1f} / {res_prof['mean_speed'].mean():.1f}",
    "",
    "INCIDENT CHARACTERISTICS:",
    f"  Incidents:   {deg_prof['n_incidents'].mean():.1f} / {res_prof['n_incidents'].mean():.1f}",
    f"  Duration:    {deg_prof['mean_duration'].mean():.1f} / {res_prof['mean_duration'].mean():.1f}",
    "",
    "PREDICTION BIAS (normal → incident):",
    f"  STAE: {stae_df['normal_bias'].mean():.1f} → {stae_df['incident_bias'].mean():.1f}",
    f"  STGCN: {stgcn_df['normal_bias'].mean():.1f} → {stgcn_df['incident_bias'].mean():.1f}",
    "",
    "KEY: Both models OVER-PREDICT during",
    "incidents (predict normal traffic level",
    "when actual flow drops).",
]
ax.text(0.02, 0.98, '\n'.join(summary), transform=ax.transAxes,
        fontsize=8.5, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# --- Row 7: Individual incident case studies ---
# Find 3 specific large incidents on shared nodes
big_incidents = test_incidents[test_incidents['sensor_idx'].isin(SHARED_NODES)].nlargest(3, 'duration_slots')
for i, (_, inc) in enumerate(big_incidents.iterrows()):
    ax = fig.add_subplot(gs[6, i])
    nid = int(inc['sensor_idx'])
    inc_start = int(inc['original_start_slot'])
    inc_dur = int(inc['duration_slots'])

    times, targets_list, stae_list, stgcn_list, raw_flow_list = [], [], [], [], []
    for rel_t in range(-window_before, window_after):
        target_ts = inc_start + rel_t
        sample_t = target_ts - test_start - INPUT_LEN
        if sample_t < 0 or sample_t >= N_TEST_SAMPLES:
            continue
        times.append(rel_t)
        targets_list.append(float(stae_tgt[sample_t, 0, nid]))
        stae_list.append(float(stae_pred[sample_t, 0, nid]))
        stgcn_list.append(float(stgcn_pred[sample_t, 0, nid]))
        if target_ts < DATA_RANGE[1]:
            raw_flow_list.append(float(flow[target_ts, nid]))
        else:
            raw_flow_list.append(np.nan)

    ax.plot(times, targets_list, 'k-', linewidth=2, label='Target')
    ax.plot(times, stae_list, 'b--', linewidth=1.5, label='STAEformer')
    ax.plot(times, stgcn_list, 'r--', linewidth=1.5, label='STGCN')
    ax.axvline(x=0, color='red', linestyle=':', linewidth=2, alpha=0.7)
    ax.axvspan(0, inc_dur, alpha=0.15, color='red', label=f'Incident ({inc_dur*5}min)')
    ax.set_xlabel('Relative Time (5-min slots)')
    ax.set_title(f'Case Study: Node {nid}, {inc["incident_type"]}\nDuration: {inc_dur*5}min ({inc_dur} slots)', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.savefig(f'{OUTPUT_DIR}/shared_degraded_deep_analysis.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: {OUTPUT_DIR}/shared_degraded_deep_analysis.png")

# Save data
df_pred.to_csv(f'{OUTPUT_DIR}/shared_nodes_prediction_behavior.csv', index=False)
df_profiles.to_csv(f'{OUTPUT_DIR}/shared_nodes_profiles.csv', index=False)
avg_temporal.to_csv(f'{OUTPUT_DIR}/shared_nodes_temporal_profile.csv', index=False)
print(f"Saved: shared_nodes_prediction_behavior.csv, shared_nodes_profiles.csv, shared_nodes_temporal_profile.csv")

print("\n" + "=" * 70)
print("DEEP ANALYSIS COMPLETE")
print("=" * 70)
