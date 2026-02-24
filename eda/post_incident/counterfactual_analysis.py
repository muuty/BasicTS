"""
Counterfactual Analysis: Incident vs Same-TOD/DOW Non-Incident
==============================================================
For each incident on shared degraded nodes:
  - Find matched non-incident samples at same TOD + DOW
  - Compare raw data (flow, occ, speed) and model predictions
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUTPUT_DIR = 'eda/post_incident'
DATA_PATH = 'datasets/xtraffic/SAN_BERNARDINO/data.dat'
INCIDENT_PATH = 'datasets/xtraffic/SAN_BERNARDINO/incident_metadata_2023.csv'

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

SHARED_NODES = sorted([5, 12, 214, 249, 303, 330, 352, 363, 372, 443, 495, 557, 576, 653, 663, 699])

print("=" * 70)
print("Counterfactual Analysis: Incident vs Same-TOD/DOW Non-Incident")
print("=" * 70)

# ============================================================================
# Load data
# ============================================================================
print("\n[1] Loading data...")

data = np.memmap(DATA_PATH, dtype='float32', mode='r').reshape(-1, NUM_NODES, 5)[:DATA_RANGE[1]]
flow, occ, speed, tod, dow = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3], data[:,:,4]

total_len = DATA_RANGE[1]
test_start = int(total_len * (TRAIN_VAL_TEST_RATIO[0] + TRAIN_VAL_TEST_RATIO[1]))

stae_pred = np.memmap(f'{STAE_PRED_DIR}/predictions.npy', dtype='float32', mode='r').reshape(N_TEST_SAMPLES, OUTPUT_LEN, NUM_NODES)
stae_tgt = np.memmap(f'{STAE_PRED_DIR}/targets.npy', dtype='float32', mode='r').reshape(N_TEST_SAMPLES, OUTPUT_LEN, NUM_NODES)
stgcn_pred = np.memmap(f'{STGCN_PRED_DIR}/predictions.npy', dtype='float32', mode='r').reshape(N_TEST_SAMPLES, OUTPUT_LEN, NUM_NODES)
stgcn_tgt = np.memmap(f'{STGCN_PRED_DIR}/targets.npy', dtype='float32', mode='r').reshape(N_TEST_SAMPLES, OUTPUT_LEN, NUM_NODES)

# Incidents
incidents = pd.read_csv(INCIDENT_PATH)
incidents = incidents[incidents['original_start_slot'] < DATA_RANGE[1]].copy()
incidents['incident_end'] = incidents['original_start_slot'] + incidents['duration_slots'] + RECOVERY_BUFFER
pred_window_start = test_start + INPUT_LEN
pred_window_end = test_start + N_TEST_SAMPLES - 1 + INPUT_LEN + OUTPUT_LEN - 1
test_incidents = incidents[
    (incidents['incident_end'] >= pred_window_start) &
    (incidents['original_start_slot'] <= pred_window_end)
].copy()

# Build incident mask
incident_mask = np.zeros((N_TEST_SAMPLES, NUM_NODES), dtype=bool)
for _, inc in test_incidents.iterrows():
    s = int(inc['sensor_idx'])
    inc_start = int(inc['original_start_slot'])
    inc_end = int(inc['incident_end'])
    t_min = max(0, inc_start - test_start - INPUT_LEN - OUTPUT_LEN + 1)
    t_max = min(N_TEST_SAMPLES - 1, inc_end - test_start - INPUT_LEN)
    if t_min <= t_max:
        incident_mask[t_min:t_max+1, s] = True

# Pre-compute TOD/DOW for each test sample's prediction window start
# test sample t -> prediction covers timesteps [test_start + t + INPUT_LEN, ..., test_start + t + INPUT_LEN + OUTPUT_LEN - 1]
test_pred_start_ts = np.arange(N_TEST_SAMPLES) + test_start + INPUT_LEN
test_tod_slot = np.array([int(tod[ts, 0] * STEPS_PER_DAY) % STEPS_PER_DAY if ts < DATA_RANGE[1] else -1 for ts in test_pred_start_ts])
test_dow_val = np.array([int(round(dow[ts, 0] * 7)) % 7 if ts < DATA_RANGE[1] else -1 for ts in test_pred_start_ts])

print(f"  Test samples with valid TOD/DOW: {(test_tod_slot >= 0).sum()}")

# ============================================================================
# For each incident, find matched non-incident samples
# ============================================================================
print("\n[2] Building matched pairs (incident vs same TOD/DOW non-incident)...")

TOD_TOLERANCE = 1  # +-1 slot = +-5 min matching window

matched_records = []

for node_idx in SHARED_NODES:
    node_incs = test_incidents[test_incidents['sensor_idx'] == node_idx]
    inc_mask_node = incident_mask[:, node_idx]

    for _, inc in node_incs.iterrows():
        inc_start = int(inc['original_start_slot'])
        inc_dur = int(inc['duration_slots'])
        inc_type = inc['incident_type']

        # The test sample where prediction window starts at incident
        inc_sample_t = inc_start - test_start - INPUT_LEN
        if inc_sample_t < 0 or inc_sample_t >= N_TEST_SAMPLES:
            continue

        inc_tod = test_tod_slot[inc_sample_t]
        inc_dow = test_dow_val[inc_sample_t]
        if inc_tod < 0 or inc_dow < 0:
            continue

        # Find non-incident samples at same TOD +- tolerance and same DOW
        matched_indices = []
        for t in range(N_TEST_SAMPLES):
            if inc_mask_node[t]:
                continue  # skip incident-affected samples
            if test_dow_val[t] != inc_dow:
                continue
            if abs(test_tod_slot[t] - inc_tod) > TOD_TOLERANCE:
                continue
            matched_indices.append(t)

        if len(matched_indices) < 3:
            continue

        # --- Raw data comparison (input window: 12 steps before prediction) ---
        # Input window for incident sample
        inc_input_start = test_start + inc_sample_t  # absolute timestep
        inc_input_end = inc_input_start + INPUT_LEN

        # Raw channels during input window
        inc_input_flow = flow[inc_input_start:inc_input_end, node_idx]
        inc_input_occ = occ[inc_input_start:inc_input_end, node_idx]
        inc_input_speed = speed[inc_input_start:inc_input_end, node_idx]

        # Matched non-incident input windows (average)
        match_input_flows, match_input_occs, match_input_speeds = [], [], []
        for mt in matched_indices:
            mt_start = test_start + mt
            mt_end = mt_start + INPUT_LEN
            if mt_end <= DATA_RANGE[1]:
                match_input_flows.append(flow[mt_start:mt_end, node_idx])
                match_input_occs.append(occ[mt_start:mt_end, node_idx])
                match_input_speeds.append(speed[mt_start:mt_end, node_idx])
        match_input_flow = np.mean(match_input_flows, axis=0)
        match_input_occ = np.mean(match_input_occs, axis=0)
        match_input_speed = np.mean(match_input_speeds, axis=0)

        # --- Target window (prediction: 12 steps) ---
        inc_target = stae_tgt[inc_sample_t, :, node_idx]
        match_targets = stae_tgt[matched_indices, :, node_idx]  # (n_matched, 12)
        match_target_mean = match_targets.mean(axis=0)

        # --- Predictions ---
        inc_stae_pred = stae_pred[inc_sample_t, :, node_idx]
        inc_stgcn_pred = stgcn_pred[inc_sample_t, :, node_idx]
        match_stae_preds = stae_pred[matched_indices, :, node_idx].mean(axis=0)
        match_stgcn_preds = stgcn_pred[matched_indices, :, node_idx].mean(axis=0)

        # --- Aggregate stats ---
        for h in range(OUTPUT_LEN):
            matched_records.append({
                'node_idx': node_idx,
                'incident_type': inc_type,
                'incident_duration': inc_dur,
                'inc_tod': inc_tod,
                'inc_dow': inc_dow,
                'n_matched': len(matched_indices),
                'horizon': h + 1,
                # Input window (last slot)
                'inc_input_flow_last': float(inc_input_flow[-1]),
                'match_input_flow_last': float(match_input_flow[-1]),
                'inc_input_occ_last': float(inc_input_occ[-1]),
                'match_input_occ_last': float(match_input_occ[-1]),
                'inc_input_speed_last': float(inc_input_speed[-1]),
                'match_input_speed_last': float(match_input_speed[-1]),
                # Input window mean
                'inc_input_flow_mean': float(inc_input_flow.mean()),
                'match_input_flow_mean': float(match_input_flow.mean()),
                'inc_input_occ_mean': float(inc_input_occ.mean()),
                'match_input_occ_mean': float(match_input_occ.mean()),
                'inc_input_speed_mean': float(inc_input_speed.mean()),
                'match_input_speed_mean': float(match_input_speed.mean()),
                # Target
                'inc_target': float(inc_target[h]),
                'match_target': float(match_target_mean[h]),
                'target_diff': float(inc_target[h] - match_target_mean[h]),
                'target_diff_pct': float((inc_target[h] - match_target_mean[h]) / match_target_mean[h] * 100) if match_target_mean[h] > 0 else 0,
                # Predictions
                'inc_stae_pred': float(inc_stae_pred[h]),
                'match_stae_pred': float(match_stae_preds[h]),
                'inc_stgcn_pred': float(inc_stgcn_pred[h]),
                'match_stgcn_pred': float(match_stgcn_preds[h]),
                # Errors
                'inc_stae_error': float(abs(inc_stae_pred[h] - inc_target[h])),
                'match_stae_error': float(abs(match_stae_preds[h] - match_target_mean[h])),
                'inc_stgcn_error': float(abs(inc_stgcn_pred[h] - inc_target[h])),
                'match_stgcn_error': float(abs(match_stgcn_preds[h] - match_target_mean[h])),
                # Bias
                'inc_stae_bias': float(inc_stae_pred[h] - inc_target[h]),
                'match_stae_bias': float(match_stae_preds[h] - match_target_mean[h]),
                'inc_stgcn_bias': float(inc_stgcn_pred[h] - inc_target[h]),
                'match_stgcn_bias': float(match_stgcn_preds[h] - match_target_mean[h]),
            })

df = pd.DataFrame(matched_records)
n_incidents_matched = df.groupby(['node_idx', 'inc_tod', 'inc_dow']).ngroups
print(f"  Matched incidents: {n_incidents_matched}, total records: {len(df)}")

# ============================================================================
# Analysis
# ============================================================================
print("\n[3] Analysis: Incident vs Matched Non-Incident...")

# Average across all incidents and horizons
agg = df.groupby('horizon').agg(
    inc_target=('inc_target', 'mean'),
    match_target=('match_target', 'mean'),
    target_diff=('target_diff', 'mean'),
    target_diff_pct=('target_diff_pct', 'mean'),
    inc_stae_pred=('inc_stae_pred', 'mean'),
    match_stae_pred=('match_stae_pred', 'mean'),
    inc_stgcn_pred=('inc_stgcn_pred', 'mean'),
    match_stgcn_pred=('match_stgcn_pred', 'mean'),
    inc_stae_error=('inc_stae_error', 'mean'),
    match_stae_error=('match_stae_error', 'mean'),
    inc_stgcn_error=('inc_stgcn_error', 'mean'),
    match_stgcn_error=('match_stgcn_error', 'mean'),
    inc_stae_bias=('inc_stae_bias', 'mean'),
    match_stae_bias=('match_stae_bias', 'mean'),
    inc_stgcn_bias=('inc_stgcn_bias', 'mean'),
    match_stgcn_bias=('match_stgcn_bias', 'mean'),
).reset_index()

print("\n  Per-horizon comparison (averaged over all matched incidents):")
print(f"  {'H':>3} | {'Inc Target':>10} {'Match Tgt':>10} {'Diff%':>7} | {'STAE Inc':>10} {'STAE Match':>10} | {'STGCN Inc':>10} {'STGCN Match':>10}")
print("  " + "-" * 95)
for _, r in agg.iterrows():
    print(f"  {int(r['horizon']):3d} | {r['inc_target']:10.1f} {r['match_target']:10.1f} {r['target_diff_pct']:6.1f}% | {r['inc_stae_pred']:10.1f} {r['match_stae_pred']:10.1f} | {r['inc_stgcn_pred']:10.1f} {r['match_stgcn_pred']:10.1f}")

# Input window comparison
input_agg = df.drop_duplicates(subset=['node_idx', 'inc_tod', 'inc_dow']).agg({
    'inc_input_flow_mean': 'mean', 'match_input_flow_mean': 'mean',
    'inc_input_occ_mean': 'mean', 'match_input_occ_mean': 'mean',
    'inc_input_speed_mean': 'mean', 'match_input_speed_mean': 'mean',
    'inc_input_flow_last': 'mean', 'match_input_flow_last': 'mean',
    'inc_input_occ_last': 'mean', 'match_input_occ_last': 'mean',
    'inc_input_speed_last': 'mean', 'match_input_speed_last': 'mean',
})

print(f"\n  Input Window Comparison (12 steps before prediction):")
print(f"  {'Channel':<15} {'Incident':>12} {'Matched':>12} {'Diff':>10} {'Diff%':>8}")
print("  " + "-" * 60)
for ch, label in [('flow', 'Flow'), ('occ', 'Occupancy'), ('speed', 'Speed')]:
    inc_v = input_agg[f'inc_input_{ch}_mean']
    mat_v = input_agg[f'match_input_{ch}_mean']
    diff_pct = (inc_v - mat_v) / mat_v * 100 if mat_v > 0 else 0
    print(f"  {label + ' (mean)':<15} {inc_v:12.2f} {mat_v:12.2f} {inc_v-mat_v:10.2f} {diff_pct:7.1f}%")
    inc_l = input_agg[f'inc_input_{ch}_last']
    mat_l = input_agg[f'match_input_{ch}_last']
    diff_pct_l = (inc_l - mat_l) / mat_l * 100 if mat_l > 0 else 0
    print(f"  {label + ' (last)':<15} {inc_l:12.2f} {mat_l:12.2f} {inc_l-mat_l:10.2f} {diff_pct_l:7.1f}%")

# Bias analysis
overall = df.mean(numeric_only=True)
print(f"\n  Overall Bias (pred - target):")
print(f"  STAEformer: incident={overall['inc_stae_bias']:.2f}, matched={overall['match_stae_bias']:.2f}")
print(f"  STGCN:      incident={overall['inc_stgcn_bias']:.2f}, matched={overall['match_stgcn_bias']:.2f}")

print(f"\n  Overall MAE:")
print(f"  STAEformer: incident={overall['inc_stae_error']:.2f}, matched={overall['match_stae_error']:.2f}, increase={overall['inc_stae_error']-overall['match_stae_error']:.2f}")
print(f"  STGCN:      incident={overall['inc_stgcn_error']:.2f}, matched={overall['match_stgcn_error']:.2f}, increase={overall['inc_stgcn_error']-overall['match_stgcn_error']:.2f}")

# Per-node analysis
print(f"\n  Per-Node Summary (horizon-averaged):")
node_agg = df.groupby('node_idx').agg(
    inc_target=('inc_target', 'mean'),
    match_target=('match_target', 'mean'),
    target_diff_pct=('target_diff_pct', 'mean'),
    inc_stae_error=('inc_stae_error', 'mean'),
    match_stae_error=('match_stae_error', 'mean'),
    inc_stgcn_error=('inc_stgcn_error', 'mean'),
    match_stgcn_error=('match_stgcn_error', 'mean'),
    inc_stae_bias=('inc_stae_bias', 'mean'),
    inc_stgcn_bias=('inc_stgcn_bias', 'mean'),
    n_matched=('n_matched', 'first'),
    incident_type=('incident_type', 'first'),
).reset_index()
node_agg['stae_error_increase'] = node_agg['inc_stae_error'] - node_agg['match_stae_error']
node_agg['stgcn_error_increase'] = node_agg['inc_stgcn_error'] - node_agg['match_stgcn_error']

print(node_agg[['node_idx','incident_type','n_matched','inc_target','match_target','target_diff_pct',
                'stae_error_increase','stgcn_error_increase','inc_stae_bias','inc_stgcn_bias']].sort_values('stae_error_increase', ascending=False).to_string(index=False))

# ============================================================================
# Visualizations
# ============================================================================
print("\n[4] Creating visualizations...")

fig = plt.figure(figsize=(28, 32))
gs = gridspec.GridSpec(6, 3, figure=fig, hspace=0.35, wspace=0.3)

# 1: Target comparison (incident vs matched) by horizon
ax = fig.add_subplot(gs[0, 0])
ax.plot(agg['horizon'], agg['match_target'], 'g-o', linewidth=2, markersize=6, label='Matched (same TOD/DOW, no incident)')
ax.plot(agg['horizon'], agg['inc_target'], 'r-s', linewidth=2, markersize=6, label='Incident')
ax.fill_between(agg['horizon'], agg['match_target'], agg['inc_target'], alpha=0.2, color='red')
ax.set_xlabel('Prediction Horizon')
ax.set_ylabel('Target Flow')
ax.set_title('Target: Incident vs Matched Non-Incident')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 2: Target difference %
ax = fig.add_subplot(gs[0, 1])
ax.bar(agg['horizon'], agg['target_diff_pct'], color='red', alpha=0.7)
ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax.set_xlabel('Prediction Horizon')
ax.set_ylabel('Target Difference (%)')
ax.set_title('How Much Does Target Change During Incidents?')
ax.grid(True, alpha=0.3)

# 3: Input window comparison
ax = fig.add_subplot(gs[0, 2])
channels = ['Flow (mean)', 'Flow (last)', 'Occ (mean)', 'Occ (last)', 'Speed (mean)', 'Speed (last)']
inc_vals = [input_agg[f'inc_input_{ch}_{stat}'] for ch, stat in
            [('flow','mean'),('flow','last'),('occ','mean'),('occ','last'),('speed','mean'),('speed','last')]]
mat_vals = [input_agg[f'match_input_{ch}_{stat}'] for ch, stat in
            [('flow','mean'),('flow','last'),('occ','mean'),('occ','last'),('speed','mean'),('speed','last')]]
# Normalize for display
max_v = [max(abs(i), abs(m)) if max(abs(i), abs(m)) > 0 else 1 for i, m in zip(inc_vals, mat_vals)]
x = np.arange(len(channels))
width = 0.35
ax.bar(x - width/2, [i/m for i,m in zip(inc_vals, max_v)], width, label='Incident', color='red', alpha=0.7)
ax.bar(x + width/2, [m/mv for m,mv in zip(mat_vals, max_v)], width, label='Matched', color='green', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(channels, rotation=30, ha='right', fontsize=8)
ax.set_ylabel('Normalized Value')
ax.set_title('Input Window: Incident vs Matched')
ax.legend(fontsize=8)

# 4: STAEformer predictions
ax = fig.add_subplot(gs[1, 0])
ax.plot(agg['horizon'], agg['match_target'], 'g-o', linewidth=2, label='Matched target')
ax.plot(agg['horizon'], agg['match_stae_pred'], 'g--^', linewidth=1.5, label='Matched STAE pred')
ax.plot(agg['horizon'], agg['inc_target'], 'r-o', linewidth=2, label='Incident target')
ax.plot(agg['horizon'], agg['inc_stae_pred'], 'r--^', linewidth=1.5, label='Incident STAE pred')
ax.set_xlabel('Prediction Horizon')
ax.set_ylabel('Value')
ax.set_title('STAEformer: Predictions vs Targets')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# 5: STGCN predictions
ax = fig.add_subplot(gs[1, 1])
ax.plot(agg['horizon'], agg['match_target'], 'g-o', linewidth=2, label='Matched target')
ax.plot(agg['horizon'], agg['match_stgcn_pred'], 'g--^', linewidth=1.5, label='Matched STGCN pred')
ax.plot(agg['horizon'], agg['inc_target'], 'r-o', linewidth=2, label='Incident target')
ax.plot(agg['horizon'], agg['inc_stgcn_pred'], 'r--^', linewidth=1.5, label='Incident STGCN pred')
ax.set_xlabel('Prediction Horizon')
ax.set_ylabel('Value')
ax.set_title('STGCN: Predictions vs Targets')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# 6: Error comparison by horizon
ax = fig.add_subplot(gs[1, 2])
ax.plot(agg['horizon'], agg['match_stae_error'], 'b-o', linewidth=2, label='STAE matched')
ax.plot(agg['horizon'], agg['inc_stae_error'], 'b--s', linewidth=2, label='STAE incident')
ax.plot(agg['horizon'], agg['match_stgcn_error'], 'r-o', linewidth=2, label='STGCN matched')
ax.plot(agg['horizon'], agg['inc_stgcn_error'], 'r--s', linewidth=2, label='STGCN incident')
ax.set_xlabel('Prediction Horizon')
ax.set_ylabel('MAE')
ax.set_title('Error: Incident vs Matched')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# 7: Bias shift
ax = fig.add_subplot(gs[2, 0])
ax.plot(agg['horizon'], agg['match_stae_bias'], 'b-o', linewidth=2, label='STAE matched')
ax.plot(agg['horizon'], agg['inc_stae_bias'], 'b--s', linewidth=2, label='STAE incident')
ax.plot(agg['horizon'], agg['match_stgcn_bias'], 'r-o', linewidth=2, label='STGCN matched')
ax.plot(agg['horizon'], agg['inc_stgcn_bias'], 'r--s', linewidth=2, label='STGCN incident')
ax.axhline(y=0, color='black', linestyle=':', alpha=0.5)
ax.set_xlabel('Prediction Horizon')
ax.set_ylabel('Bias (pred - target)')
ax.set_title('Prediction Bias: Incident vs Matched')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# 8: Per-node target diff %
ax = fig.add_subplot(gs[2, 1])
node_agg_sorted = node_agg.sort_values('target_diff_pct')
colors = ['red' if v < 0 else 'blue' for v in node_agg_sorted['target_diff_pct']]
ax.barh(range(len(node_agg_sorted)), node_agg_sorted['target_diff_pct'], color=colors, alpha=0.7)
ax.set_yticks(range(len(node_agg_sorted)))
ax.set_yticklabels([f"N{n}" for n in node_agg_sorted['node_idx']], fontsize=7)
ax.set_xlabel('Target Diff % (incident vs matched)')
ax.set_title('Per-Node: How Much Does Flow Change?')
ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)

# 9: Scatter - target change vs error increase
ax = fig.add_subplot(gs[2, 2])
ax.scatter(node_agg['target_diff_pct'], node_agg['stae_error_increase'], c='blue', s=60, alpha=0.7, label='STAEformer')
ax.scatter(node_agg['target_diff_pct'], node_agg['stgcn_error_increase'], c='red', s=60, alpha=0.7, label='STGCN')
for _, row in node_agg.iterrows():
    ax.annotate(f"N{int(row['node_idx'])}", (row['target_diff_pct'], row['stae_error_increase']), fontsize=6, alpha=0.7)
ax.set_xlabel('Target Flow Change % (incident vs matched)')
ax.set_ylabel('MAE Increase (incident vs matched)')
ax.set_title('Does Larger Flow Change → Larger Error?')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 10-15: Individual node case studies (6 nodes)
case_nodes = node_agg.nlargest(6, 'stae_error_increase')['node_idx'].values
for i, nid in enumerate(case_nodes):
    row_idx = 3 + i // 3
    col_idx = i % 3
    ax = fig.add_subplot(gs[row_idx, col_idx])

    node_df = df[df['node_idx'] == nid]
    # Average over incidents for this node
    h_data = node_df.groupby('horizon').mean(numeric_only=True)

    ax.plot(h_data.index, h_data['match_target'], 'g-o', linewidth=2, markersize=5, label='Matched target')
    ax.plot(h_data.index, h_data['match_stae_pred'], 'g--', linewidth=1, alpha=0.7, label='Matched STAE')
    ax.plot(h_data.index, h_data['inc_target'], 'r-o', linewidth=2, markersize=5, label='Inc target')
    ax.plot(h_data.index, h_data['inc_stae_pred'], 'b--^', linewidth=1.5, label='Inc STAE pred')
    ax.plot(h_data.index, h_data['inc_stgcn_pred'], 'r--^', linewidth=1.5, label='Inc STGCN pred')

    info = node_agg[node_agg['node_idx']==nid].iloc[0]
    ax.set_xlabel('Horizon')
    ax.set_title(f"Node {nid}: target diff {info['target_diff_pct']:.0f}%\n"
                 f"STAE err +{info['stae_error_increase']:.1f}, STGCN err +{info['stgcn_error_increase']:.1f}\n"
                 f"({info['incident_type']}, {int(info['n_matched'])} matched)", fontsize=8)
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

# Summary panel
ax = fig.add_subplot(gs[4, 2])
ax.axis('off')
summary = [
    "COUNTERFACTUAL ANALYSIS SUMMARY",
    "=" * 40,
    f"Incidents analyzed: {n_incidents_matched}",
    f"Nodes: {len(SHARED_NODES)}",
    "",
    "INPUT WINDOW (what model sees):",
    f"  Flow:  inc={input_agg['inc_input_flow_mean']:.1f} vs match={input_agg['match_input_flow_mean']:.1f}",
    f"  Occ:   inc={input_agg['inc_input_occ_mean']:.3f} vs match={input_agg['match_input_occ_mean']:.3f}",
    f"  Speed: inc={input_agg['inc_input_speed_mean']:.1f} vs match={input_agg['match_input_speed_mean']:.1f}",
    "",
    "TARGET (what should be predicted):",
    f"  Inc: {overall['inc_target']:.1f} vs Match: {overall['match_target']:.1f}",
    f"  Diff: {overall['target_diff_pct']:.1f}%",
    "",
    "ERROR INCREASE:",
    f"  STAE: {overall['inc_stae_error']:.1f} vs {overall['match_stae_error']:.1f} (+{overall['inc_stae_error']-overall['match_stae_error']:.1f})",
    f"  STGCN: {overall['inc_stgcn_error']:.1f} vs {overall['match_stgcn_error']:.1f} (+{overall['inc_stgcn_error']-overall['match_stgcn_error']:.1f})",
    "",
    "BIAS DURING INCIDENTS:",
    f"  STAE: {overall['inc_stae_bias']:.1f} (match: {overall['match_stae_bias']:.1f})",
    f"  STGCN: {overall['inc_stgcn_bias']:.1f} (match: {overall['match_stgcn_bias']:.1f})",
]
ax.text(0.02, 0.98, '\n'.join(summary), transform=ax.transAxes,
        fontsize=8.5, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.savefig(f'{OUTPUT_DIR}/counterfactual_analysis.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: {OUTPUT_DIR}/counterfactual_analysis.png")

df.to_csv(f'{OUTPUT_DIR}/counterfactual_matched_data.csv', index=False)
node_agg.to_csv(f'{OUTPUT_DIR}/counterfactual_per_node.csv', index=False)
print(f"Saved: counterfactual_matched_data.csv, counterfactual_per_node.csv")

print("\n" + "=" * 70)
print("COUNTERFACTUAL ANALYSIS COMPLETE")
print("=" * 70)
