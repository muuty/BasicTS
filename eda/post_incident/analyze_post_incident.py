"""
Post-Incident Node-wise Prediction Error Analysis
==================================================
Analyzes which nodes degrade most during/after traffic incidents,
and characterizes what makes them vulnerable.

Key questions:
1. Which nodes predict well normally but degrade sharply during/after incidents?
2. What are the characteristics of these vulnerable nodes?
3. How does prediction error evolve temporally around incidents?
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
ADJ_PATH = 'datasets/xtraffic/SAN_BERNARDINO/adj_mx.pkl'
OUTPUT_DIR = 'eda/post_incident'

NUM_NODES = 893
INPUT_LEN = 12
OUTPUT_LEN = 12
N_TEST_SAMPLES = 5233
DATA_RANGE = (0, 26280)
TRAIN_VAL_TEST_RATIO = [0.6, 0.2, 0.2]
STEPS_PER_DAY = 288

# Post-incident recovery buffer (slots after incident to still consider "affected")
RECOVERY_BUFFER = 12  # 1 hour

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Helper
def paired_spearmanr(s1, s2):
    mask = s1.notna() & s2.notna()
    return stats.spearmanr(s1[mask], s2[mask])

print("=" * 70)
print("Post-Incident Node-wise Prediction Error Analysis")
print("=" * 70)

# ============================================================================
# 1. Load predictions and data
# ============================================================================
print("\n[1] Loading data...")

# Saved predictions (raw scale, already inverse-transformed)
predictions = np.memmap(f'{PRED_DIR}/predictions.npy', dtype='float32', mode='r').reshape(N_TEST_SAMPLES, OUTPUT_LEN, NUM_NODES)
targets = np.memmap(f'{PRED_DIR}/targets.npy', dtype='float32', mode='r').reshape(N_TEST_SAMPLES, OUTPUT_LEN, NUM_NODES)

# Verify
mask_nz = targets != 0
mae_check = np.abs(predictions[mask_nz] - targets[mask_nz]).mean()
print(f"  Predictions shape: {predictions.shape}")
print(f"  Masked MAE check: {mae_check:.4f} (should be ~12.10)")

# Raw data (for node characteristics)
data = np.memmap(DATA_PATH, dtype='float32', mode='r').reshape(-1, NUM_NODES, 5)[:DATA_RANGE[1]]

# Test set boundaries
total_len = DATA_RANGE[1] - DATA_RANGE[0]
test_start = int(total_len * (TRAIN_VAL_TEST_RATIO[0] + TRAIN_VAL_TEST_RATIO[1]))
print(f"  Test start timestep: {test_start}, Test samples: {N_TEST_SAMPLES}")

# Load incidents
incidents = pd.read_csv(INCIDENT_PATH)
incidents = incidents[incidents['original_start_slot'] < DATA_RANGE[1]].copy()

# Filter to incidents affecting test predictions
pred_window_start = test_start + INPUT_LEN
pred_window_end = test_start + N_TEST_SAMPLES - 1 + INPUT_LEN + OUTPUT_LEN - 1
incidents['incident_end'] = incidents['original_start_slot'] + incidents['duration_slots'] + RECOVERY_BUFFER

test_incidents = incidents[
    (incidents['incident_end'] >= pred_window_start) &
    (incidents['original_start_slot'] <= pred_window_end)
].copy()
print(f"  Incidents affecting test: {len(test_incidents)}, unique sensors: {test_incidents['sensor_idx'].nunique()}")

# ============================================================================
# 2. Build incident mask: (N_TEST_SAMPLES, NUM_NODES) boolean
# ============================================================================
print("\n[2] Building incident mask...")

incident_mask = np.zeros((N_TEST_SAMPLES, NUM_NODES), dtype=bool)

for _, inc in test_incidents.iterrows():
    s = int(inc['sensor_idx'])
    inc_start = int(inc['original_start_slot'])
    inc_end = int(inc['incident_end'])
    t_min = max(0, inc_start - test_start - INPUT_LEN - OUTPUT_LEN + 1)
    t_max = min(N_TEST_SAMPLES - 1, inc_end - test_start - INPUT_LEN)
    if t_min <= t_max:
        incident_mask[t_min:t_max+1, s] = True

n_affected = incident_mask.sum()
print(f"  Incident-affected pairs: {n_affected:,} / {incident_mask.size:,} ({n_affected/incident_mask.size*100:.2f}%)")

# ============================================================================
# 3. Compute per-sample per-node errors
# ============================================================================
print("\n[3] Computing errors...")

abs_errors = np.abs(predictions - targets)  # (5233, 12, 893)

# Per-sample per-node MAE (average over 12 horizons)
sample_node_mae = abs_errors.mean(axis=1)  # (5233, 893)

# Masked version (exclude zero targets)
nonzero_mask = (targets != 0)
nonzero_count = nonzero_mask.sum(axis=1)  # (5233, 893)
masked_errors = (abs_errors * nonzero_mask).sum(axis=1)
sample_node_masked_mae = np.where(nonzero_count > 0, masked_errors / nonzero_count, np.nan)

print(f"  Overall unmasked MAE: {sample_node_mae.mean():.4f}")
print(f"  Overall masked MAE: {np.nanmean(sample_node_masked_mae):.4f}")

# ============================================================================
# 4. Compare normal vs incident-affected per node
# ============================================================================
print("\n[4] Comparing normal vs incident-affected per node...")

results = []
for node_idx in range(NUM_NODES):
    inc_samples = incident_mask[:, node_idx]
    normal_samples = ~inc_samples
    n_inc = inc_samples.sum()
    n_normal = normal_samples.sum()
    if n_inc == 0 or n_normal == 0:
        continue

    normal_mae = sample_node_mae[normal_samples, node_idx].mean()
    incident_mae = sample_node_mae[inc_samples, node_idx].mean()

    normal_masked = sample_node_masked_mae[normal_samples, node_idx]
    incident_masked = sample_node_masked_mae[inc_samples, node_idx]
    normal_masked_mae = np.nanmean(normal_masked)
    incident_masked_mae = np.nanmean(incident_masked)

    mae_increase = incident_mae - normal_mae
    masked_increase = incident_masked_mae - normal_masked_mae if not (np.isnan(incident_masked_mae) or np.isnan(normal_masked_mae)) else np.nan
    masked_ratio = incident_masked_mae / normal_masked_mae if normal_masked_mae > 0 and not np.isnan(normal_masked_mae) else np.nan

    results.append({
        'node_idx': node_idx,
        'n_incident_samples': n_inc,
        'n_normal_samples': n_normal,
        'normal_mae': normal_mae,
        'incident_mae': incident_mae,
        'mae_increase': mae_increase,
        'normal_masked_mae': normal_masked_mae,
        'incident_masked_mae': incident_masked_mae,
        'masked_increase': masked_increase,
        'masked_ratio': masked_ratio,
    })

df_results = pd.DataFrame(results)
print(f"  Nodes with both normal & incident samples: {len(df_results)}")

# ============================================================================
# 5. Add node characteristics
# ============================================================================
print("\n[5] Adding node characteristics...")

train_end = int(total_len * TRAIN_VAL_TEST_RATIO[0])
train_data = data[:train_end]
flow = train_data[:, :, 0]
occ = train_data[:, :, 1]
speed = train_data[:, :, 2]

node_features = pd.DataFrame({
    'node_idx': range(NUM_NODES),
    'mean_flow': flow.mean(axis=0),
    'std_flow': flow.std(axis=0),
    'cv_flow': np.where(flow.mean(axis=0) > 0, flow.std(axis=0) / flow.mean(axis=0), 0),
    'mean_occ': occ.mean(axis=0),
    'mean_speed': speed.mean(axis=0),
    'zero_rate_flow': (flow == 0).mean(axis=0),
    'zero_rate_3ch': ((flow == 0) & (occ == 0) & (speed == 0)).mean(axis=0),
    'max_flow': flow.max(axis=0),
})

# Sensor categories
dead_indices = np.load('datasets/xtraffic/SAN_BERNARDINO/dead_indices.npy')
major_fail_indices = np.load('datasets/xtraffic/SAN_BERNARDINO/major_fail_indices.npy')
functional_indices = np.load('datasets/xtraffic/SAN_BERNARDINO/functional_node_indices.npy')
node_features['sensor_category'] = 'partial_fail'
node_features.loc[node_features['node_idx'].isin(dead_indices), 'sensor_category'] = 'dead'
node_features.loc[node_features['node_idx'].isin(major_fail_indices), 'sensor_category'] = 'major_fail'
node_features.loc[node_features['node_idx'].isin(functional_indices), 'sensor_category'] = 'functional'

# Adjacency matrix
with open(ADJ_PATH, 'rb') as f:
    adj_data = pickle.load(f, encoding='latin1')
adj_mx = adj_data[-1] if isinstance(adj_data, list) else np.array(adj_data)
node_features['degree'] = (adj_mx > 0).sum(axis=1)

# Merge
df_results = df_results.merge(node_features, on='node_idx', how='left')

# Incident counts per node
incident_counts = test_incidents.groupby('sensor_idx').agg(
    n_incidents=('sensor_idx', 'count'),
    mean_duration=('duration_slots', 'mean'),
    max_duration=('duration_slots', 'max'),
    incident_types=('incident_type', lambda x: ','.join(x.value_counts().head(3).index)),
).reset_index().rename(columns={'sensor_idx': 'node_idx'})
df_results = df_results.merge(incident_counts, on='node_idx', how='left')

# ============================================================================
# 6. Identify most degraded nodes
# ============================================================================
print("\n[6] Identifying most degraded nodes...")

# Focus on functional/partial nodes
df_meaningful = df_results[df_results['sensor_category'].isin(['functional', 'partial_fail'])].copy()
df_meaningful = df_meaningful.sort_values('masked_increase', ascending=False)

top_cols = ['node_idx', 'sensor_category', 'normal_masked_mae', 'incident_masked_mae',
            'masked_increase', 'masked_ratio', 'mean_flow', 'cv_flow', 'n_incidents',
            'mean_duration', 'degree']

print(f"\n  Top 20 most degraded nodes (by masked MAE increase):")
print(df_meaningful[top_cols].head(20).to_string(index=False))

print(f"\n  Top 10 most resilient nodes:")
df_resilient = df_meaningful[df_meaningful['n_incident_samples'] >= 10].sort_values('masked_increase')
print(df_resilient[top_cols].head(10).to_string(index=False))

# ============================================================================
# 7. Statistical analysis
# ============================================================================
print("\n[7] Statistical analysis...")

q75 = df_meaningful['masked_increase'].quantile(0.75)
q25 = df_meaningful['masked_increase'].quantile(0.25)
degraded = df_meaningful[df_meaningful['masked_increase'] >= q75]
resilient = df_meaningful[df_meaningful['masked_increase'] <= q25]

print(f"\n  Degraded (top 25%, increase >= {q75:.2f}): {len(degraded)}")
print(f"  Resilient (bottom 25%, increase <= {q25:.2f}): {len(resilient)}")

compare_features = ['mean_flow', 'std_flow', 'cv_flow', 'mean_occ', 'mean_speed',
                    'zero_rate_flow', 'max_flow', 'degree', 'n_incidents', 'mean_duration']

print(f"\n  {'Feature':<20} {'Degraded':<15} {'Resilient':<15} {'t-stat':<10} {'p-value':<10}")
print("  " + "-" * 70)
for feat in compare_features:
    d_vals = degraded[feat].dropna()
    r_vals = resilient[feat].dropna()
    if len(d_vals) > 1 and len(r_vals) > 1:
        t_stat, p_val = stats.ttest_ind(d_vals, r_vals)
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
        print(f"  {feat:<20} {d_vals.mean():<15.4f} {r_vals.mean():<15.4f} {t_stat:<10.3f} {p_val:<10.4f} {sig}")

# ============================================================================
# 8. Temporal profile around incidents
# ============================================================================
print("\n[8] Temporal profile around incidents...")

window_before = 24
window_after = 36
top_degraded_nodes = df_meaningful[df_meaningful['sensor_category'] == 'functional'].nlargest(50, 'masked_increase')['node_idx'].values

temporal_profiles = []
for _, inc in test_incidents.iterrows():
    s = int(inc['sensor_idx'])
    if s not in top_degraded_nodes:
        continue
    inc_start = int(inc['original_start_slot'])

    for rel_t in range(-window_before, window_after):
        target_timestep = inc_start + rel_t
        sample_t = target_timestep - test_start - INPUT_LEN
        if sample_t < 0 or sample_t >= N_TEST_SAMPLES:
            continue
        err = abs_errors[sample_t, 0, s]
        tgt = targets[sample_t, 0, s]
        temporal_profiles.append({
            'node_idx': s, 'relative_time': rel_t,
            'error': err, 'target': tgt, 'incident_type': inc['incident_type'],
        })

df_temporal = pd.DataFrame(temporal_profiles)
if len(df_temporal) > 0:
    avg_profile = df_temporal.groupby('relative_time').agg(
        mean_error=('error', 'mean'), median_error=('error', 'median'),
        mean_target=('target', 'mean'), count=('error', 'count'),
    ).reset_index()
    print(f"  Temporal profile: {len(top_degraded_nodes)} nodes, {len(df_temporal)} data points")
else:
    avg_profile = pd.DataFrame()

# ============================================================================
# 9. Per incident-type analysis
# ============================================================================
print("\n[9] Per incident-type analysis...")

normal_mae_all = sample_node_mae[~incident_mask].mean()
type_results = []
for inc_type in test_incidents['incident_type'].unique():
    type_incs = test_incidents[test_incidents['incident_type'] == inc_type]
    type_mask = np.zeros((N_TEST_SAMPLES, NUM_NODES), dtype=bool)
    for _, inc in type_incs.iterrows():
        s = int(inc['sensor_idx'])
        inc_start = int(inc['original_start_slot'])
        inc_end = int(inc['incident_end'])
        t_min = max(0, inc_start - test_start - INPUT_LEN - OUTPUT_LEN + 1)
        t_max = min(N_TEST_SAMPLES - 1, inc_end - test_start - INPUT_LEN)
        if t_min <= t_max:
            type_mask[t_min:t_max+1, s] = True

    affected = sample_node_mae[type_mask]
    type_results.append({
        'incident_type': inc_type, 'n_incidents': len(type_incs),
        'n_affected_pairs': type_mask.sum(),
        'affected_mae': affected.mean() if len(affected) > 0 else np.nan,
        'normal_mae': normal_mae_all,
        'mae_increase': affected.mean() - normal_mae_all if len(affected) > 0 else np.nan,
    })

df_types = pd.DataFrame(type_results).sort_values('mae_increase', ascending=False)
print(df_types.to_string(index=False))

# ============================================================================
# 10. Spatial contagion analysis
# ============================================================================
print("\n[10] Spatial contagion analysis...")

degraded_set = set(degraded['node_idx'].values)
for is_deg, label in [(True, 'Degraded'), (False, 'Non-degraded')]:
    nodes = degraded['node_idx'].values if is_deg else df_meaningful[~df_meaningful['node_idx'].isin(degraded_set)]['node_idx'].values
    fracs = []
    for n in nodes:
        neighbors = np.where(adj_mx[n] > 0)[0]
        if len(neighbors) > 0:
            fracs.append(sum(1 for nb in neighbors if nb in degraded_set) / len(neighbors))
    if fracs:
        print(f"  {label} nodes: mean frac_degraded_neighbors = {np.mean(fracs):.3f}")

# ============================================================================
# 11. Visualizations
# ============================================================================
print("\n[11] Creating visualizations...")

fig = plt.figure(figsize=(24, 28))
gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.35, wspace=0.3)

# 11.1: Normal vs Incident MAE scatter
ax = fig.add_subplot(gs[0, 0])
colors = df_meaningful['sensor_category'].map({'functional': 'blue', 'partial_fail': 'orange'})
ax.scatter(df_meaningful['normal_masked_mae'], df_meaningful['incident_masked_mae'],
           c=colors, alpha=0.5, s=15, edgecolors='none')
max_val = max(df_meaningful['normal_masked_mae'].max(), df_meaningful['incident_masked_mae'].max())
ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='y=x (no change)')
ax.set_xlabel('Normal MAE')
ax.set_ylabel('Incident MAE')
ax.set_title('Normal vs Incident MAE per Node\n(above diagonal = degraded)')
ax.legend()

# 11.2: MAE increase distribution
ax = fig.add_subplot(gs[0, 1])
for cat, color in [('functional', 'blue'), ('partial_fail', 'orange')]:
    subset = df_meaningful[df_meaningful['sensor_category'] == cat]
    ax.hist(subset['masked_increase'].dropna(), bins=40, alpha=0.6, color=color, label=cat, edgecolor='black', linewidth=0.5)
ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No change')
ax.set_xlabel('Masked MAE Increase (incident - normal)')
ax.set_ylabel('Number of Nodes')
ax.set_title('Distribution of Post-Incident MAE Increase')
ax.legend()

# 11.3: Top 30 most degraded
ax = fig.add_subplot(gs[0, 2])
top30 = df_meaningful.nlargest(30, 'masked_increase')
ax.barh(range(len(top30)), top30['masked_increase'], color='red', alpha=0.7)
ax.set_yticks(range(len(top30)))
ax.set_yticklabels([f"N{idx}" for idx in top30['node_idx']], fontsize=7)
ax.set_xlabel('Masked MAE Increase')
ax.set_title('Top 30 Most Degraded Nodes')
ax.invert_yaxis()

# 11.4: Characteristics comparison
ax = fig.add_subplot(gs[1, 0])
feat_labels = ['mean_flow', 'cv_flow', 'mean_occ', 'mean_speed', 'degree']
degraded_means = [degraded[f].mean() for f in feat_labels]
resilient_means = [resilient[f].mean() for f in feat_labels]
deg_norm = [d / max(d, r) if max(d, r) > 0 else 0 for d, r in zip(degraded_means, resilient_means)]
res_norm = [r / max(d, r) if max(d, r) > 0 else 0 for d, r in zip(degraded_means, resilient_means)]
x = np.arange(len(feat_labels))
width = 0.35
ax.bar(x - width/2, deg_norm, width, label='Degraded (top 25%)', color='red', alpha=0.7)
ax.bar(x + width/2, res_norm, width, label='Resilient (bottom 25%)', color='green', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(feat_labels, rotation=45, ha='right')
ax.set_ylabel('Normalized Value')
ax.set_title('Degraded vs Resilient Node Characteristics')
ax.legend()

# 11.5-11.6: Scatter plots with correlations
for idx, (col, title) in enumerate([
    ('mean_flow', 'Traffic Volume vs Post-Incident Degradation'),
    ('degree', 'Graph Connectivity vs Post-Incident Degradation'),
]):
    ax = fig.add_subplot(gs[1, 1+idx])
    ax.scatter(df_meaningful[col], df_meaningful['masked_increase'], alpha=0.4, s=10, c='steelblue')
    ax.set_xlabel(col)
    ax.set_ylabel('Masked MAE Increase')
    ax.set_title(title)
    corr, pval = paired_spearmanr(df_meaningful[col], df_meaningful['masked_increase'])
    ax.annotate(f'Spearman r={corr:.3f}\np={pval:.4f}', xy=(0.05, 0.95), xycoords='axes fraction', va='top', fontsize=10)

# 11.7: Temporal profile
ax = fig.add_subplot(gs[2, :2])
if len(avg_profile) > 0:
    ax.plot(avg_profile['relative_time'], avg_profile['mean_error'], 'b-', linewidth=2, label='Mean Error')
    ax.fill_between(avg_profile['relative_time'],
                    avg_profile['mean_error'] - avg_profile['median_error'],
                    avg_profile['mean_error'] + avg_profile['median_error'], alpha=0.2, color='blue')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Incident Start')
    ax.set_xlabel('Relative Time (5-min slots from incident start)')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Error Trajectory Around Incidents (Top 50 Degraded Functional Nodes)')
    ax.legend()
    ax.grid(True, alpha=0.3)

# 11.8: Target flow around incidents
ax = fig.add_subplot(gs[2, 2])
if len(avg_profile) > 0:
    ax.plot(avg_profile['relative_time'], avg_profile['mean_target'], 'g-', linewidth=2)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Incident Start')
    ax.set_xlabel('Relative Time (5-min slots from incident start)')
    ax.set_ylabel('Target Flow')
    ax.set_title('Target Flow Around Incidents')
    ax.legend()
    ax.grid(True, alpha=0.3)

# 11.9: Per incident-type
ax = fig.add_subplot(gs[3, 0])
df_types_sorted = df_types.sort_values('mae_increase', ascending=True)
ax.barh(range(len(df_types_sorted)), df_types_sorted['mae_increase'].fillna(0), color='coral', alpha=0.7)
ax.set_yticks(range(len(df_types_sorted)))
ax.set_yticklabels(df_types_sorted['incident_type'])
ax.set_xlabel('MAE Increase vs Normal')
ax.set_title('Post-Incident MAE Increase by Incident Type')

# 11.10-11.11: More scatter plots
for idx, (col, title) in enumerate([
    ('n_incidents', 'Incident Frequency vs Degradation'),
    ('mean_duration', 'Incident Duration vs Degradation'),
]):
    ax = fig.add_subplot(gs[3, 1+idx])
    ax.scatter(df_meaningful[col], df_meaningful['masked_increase'], alpha=0.4, s=10, c='steelblue')
    ax.set_xlabel(col)
    ax.set_ylabel('Masked MAE Increase')
    ax.set_title(title)
    corr, pval = paired_spearmanr(df_meaningful[col], df_meaningful['masked_increase'])
    ax.annotate(f'Spearman r={corr:.3f}\np={pval:.4f}', xy=(0.05, 0.95), xycoords='axes fraction', va='top', fontsize=10)

# 11.12: Horizon-wise error
ax = fig.add_subplot(gs[4, 0])
top50_nodes = df_meaningful.nlargest(50, 'masked_increase')['node_idx'].values
horizon_normal, horizon_incident = [], []
for h in range(OUTPUT_LEN):
    h_errors = abs_errors[:, h, :]
    n_errs, i_errs = [], []
    for nid in top50_nodes:
        inc_mask_node = incident_mask[:, nid]
        n_errs.append(h_errors[~inc_mask_node, nid].mean())
        i_errs.append(h_errors[inc_mask_node, nid].mean())
    horizon_normal.append(np.mean(n_errs))
    horizon_incident.append(np.mean(i_errs))
ax.plot(range(1, OUTPUT_LEN+1), horizon_normal, 'b-o', label='Normal', linewidth=2)
ax.plot(range(1, OUTPUT_LEN+1), horizon_incident, 'r-o', label='Incident', linewidth=2)
ax.set_xlabel('Prediction Horizon')
ax.set_ylabel('MAE')
ax.set_title('Horizon-wise Error: Normal vs Incident\n(Top 50 Degraded Nodes)')
ax.legend()
ax.grid(True, alpha=0.3)

# 11.13: MAE increase vs zero_rate
ax = fig.add_subplot(gs[4, 1])
ax.scatter(df_meaningful['zero_rate_flow'], df_meaningful['masked_increase'], alpha=0.4, s=10, c='steelblue')
ax.set_xlabel('Zero-flow Rate')
ax.set_ylabel('Masked MAE Increase')
ax.set_title('Sensor Health vs Post-Incident Degradation')
corr, pval = paired_spearmanr(df_meaningful['zero_rate_flow'], df_meaningful['masked_increase'])
ax.annotate(f'Spearman r={corr:.3f}\np={pval:.4f}', xy=(0.05, 0.95), xycoords='axes fraction', va='top', fontsize=10)

# 11.14: Summary
ax = fig.add_subplot(gs[4, 2])
ax.axis('off')
summary_text = [
    "SUMMARY",
    "=" * 40,
    f"Total nodes analyzed: {len(df_meaningful)}",
    f"Nodes with incidents: {(df_meaningful['n_incident_samples'] > 0).sum()}",
    f"",
    f"Overall normal MAE: {df_meaningful['normal_masked_mae'].mean():.3f}",
    f"Overall incident MAE: {df_meaningful['incident_masked_mae'].mean():.3f}",
    f"Overall increase: {df_meaningful['masked_increase'].mean():.3f}",
    f"",
    f"Degraded nodes (top 25%): {len(degraded)}",
    f"  Mean increase: {degraded['masked_increase'].mean():.3f}",
    f"  Mean flow: {degraded['mean_flow'].mean():.1f}",
    f"",
    f"Resilient nodes (bottom 25%): {len(resilient)}",
    f"  Mean increase: {resilient['masked_increase'].mean():.3f}",
    f"  Mean flow: {resilient['mean_flow'].mean():.1f}",
]
ax.text(0.05, 0.95, '\n'.join(summary_text), transform=ax.transAxes,
        fontsize=9, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.savefig(f'{OUTPUT_DIR}/post_incident_analysis.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}/post_incident_analysis.png")

# ============================================================================
# 12. Save results
# ============================================================================
print("\n[12] Saving results...")

df_results.to_csv(f'{OUTPUT_DIR}/per_node_incident_impact.csv', index=False)
print(f"  Saved: per_node_incident_impact.csv")

df_meaningful.nlargest(50, 'masked_increase').to_csv(f'{OUTPUT_DIR}/top50_degraded_nodes.csv', index=False)
print(f"  Saved: top50_degraded_nodes.csv")

if len(df_temporal) > 0:
    avg_profile.to_csv(f'{OUTPUT_DIR}/temporal_profile_around_incidents.csv', index=False)
    print(f"  Saved: temporal_profile_around_incidents.csv")

df_types.to_csv(f'{OUTPUT_DIR}/per_incident_type_impact.csv', index=False)
print(f"  Saved: per_incident_type_impact.csv")

summary = {
    'model': 'STAEformer baseline (ContextContrastive_baseline_3mo)',
    'overall_masked_mae': float(mae_check),
    'n_test_samples': N_TEST_SAMPLES,
    'n_test_incidents': int(len(test_incidents)),
    'n_nodes_with_incidents': int(test_incidents['sensor_idx'].nunique()),
    'incident_affected_fraction': float(n_affected / incident_mask.size),
    'recovery_buffer_slots': RECOVERY_BUFFER,
    'overall_normal_masked_mae': float(df_meaningful['normal_masked_mae'].mean()),
    'overall_incident_masked_mae': float(df_meaningful['incident_masked_mae'].mean()),
    'overall_masked_increase': float(df_meaningful['masked_increase'].mean()),
    'n_degraded_nodes': int(len(degraded)),
    'n_resilient_nodes': int(len(resilient)),
    'degraded_mean_increase': float(degraded['masked_increase'].mean()),
    'resilient_mean_increase': float(resilient['masked_increase'].mean()),
}
with open(f'{OUTPUT_DIR}/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  Saved: summary.json")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
