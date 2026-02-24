"""
Cross-Model Post-Incident Degradation Comparison
=================================================
Compares STAEformer vs STGCN on which nodes degrade most during incidents.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# ============================================================================
# Configuration
# ============================================================================
OUTPUT_DIR = 'eda/post_incident'
INCIDENT_PATH = 'datasets/xtraffic/SAN_BERNARDINO/incident_metadata_2023.csv'

NUM_NODES = 893
INPUT_LEN = 12
OUTPUT_LEN = 12
N_TEST_SAMPLES = 5233
DATA_RANGE = (0, 26280)
TRAIN_VAL_TEST_RATIO = [0.6, 0.2, 0.2]
RECOVERY_BUFFER = 12

MODELS = {
    'STAEformer': {
        'pred_dir': 'checkpoints/ContextContrastive_baseline_3mo/xtraffic_SAN_BERNARDINO_30_12_12/040af4f5bcb37097bc5263d7f350174b/test_results',
    },
    'STGCN': {
        'pred_dir': None,  # Will be auto-detected
    },
}

# Auto-detect STGCN pred_dir
import glob
stgcn_dirs = glob.glob('checkpoints/STGCN/SAN_BERNARDINO_*/*/test_results')
if stgcn_dirs:
    # Pick the most recent
    stgcn_dirs.sort(key=os.path.getmtime, reverse=True)
    MODELS['STGCN']['pred_dir'] = stgcn_dirs[0]
    print(f"Auto-detected STGCN: {stgcn_dirs[0]}")
else:
    print("ERROR: No STGCN test_results found!")
    exit(1)

# Helper
def paired_spearmanr(s1, s2):
    mask = s1.notna() & s2.notna()
    if mask.sum() < 3:
        return 0, 1
    return stats.spearmanr(s1[mask], s2[mask])

# ============================================================================
# Build incident mask (same as main analysis)
# ============================================================================
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

incident_mask = np.zeros((N_TEST_SAMPLES, NUM_NODES), dtype=bool)
for _, inc in test_incidents.iterrows():
    s = int(inc['sensor_idx'])
    inc_start = int(inc['original_start_slot'])
    inc_end = int(inc['incident_end'])
    t_min = max(0, inc_start - test_start - INPUT_LEN - OUTPUT_LEN + 1)
    t_max = min(N_TEST_SAMPLES - 1, inc_end - test_start - INPUT_LEN)
    if t_min <= t_max:
        incident_mask[t_min:t_max+1, s] = True

# Load sensor categories
dead_indices = set(np.load('datasets/xtraffic/SAN_BERNARDINO/dead_indices.npy'))
major_fail_indices = set(np.load('datasets/xtraffic/SAN_BERNARDINO/major_fail_indices.npy'))

def get_category(n):
    if n in dead_indices: return 'dead'
    if n in major_fail_indices: return 'major_fail'
    return 'functional_or_partial'

# ============================================================================
# Compute per-node incident impact for each model
# ============================================================================
print("=" * 70)
print("Cross-Model Post-Incident Comparison: STAEformer vs STGCN")
print("=" * 70)

model_results = {}

for model_name, cfg in MODELS.items():
    print(f"\n--- {model_name} ---")
    pred_dir = cfg['pred_dir']

    predictions = np.memmap(f'{pred_dir}/predictions.npy', dtype='float32', mode='r').reshape(N_TEST_SAMPLES, OUTPUT_LEN, NUM_NODES)
    targets = np.memmap(f'{pred_dir}/targets.npy', dtype='float32', mode='r').reshape(N_TEST_SAMPLES, OUTPUT_LEN, NUM_NODES)

    # Verify MAE
    mask_nz = targets != 0
    overall_mae = np.abs(predictions[mask_nz] - targets[mask_nz]).mean()
    print(f"  Overall masked MAE: {overall_mae:.4f}")

    abs_errors = np.abs(predictions - targets)
    nonzero_mask = (targets != 0)
    nonzero_count = nonzero_mask.sum(axis=1)
    masked_errors = (abs_errors * nonzero_mask).sum(axis=1)
    sample_node_masked_mae = np.where(nonzero_count > 0, masked_errors / nonzero_count, np.nan)

    node_data = []
    for node_idx in range(NUM_NODES):
        inc_samples = incident_mask[:, node_idx]
        normal_samples = ~inc_samples
        n_inc = inc_samples.sum()
        n_normal = normal_samples.sum()
        if n_inc == 0 or n_normal == 0:
            continue

        normal_masked = sample_node_masked_mae[normal_samples, node_idx]
        incident_masked = sample_node_masked_mae[inc_samples, node_idx]
        normal_masked_mae = np.nanmean(normal_masked)
        incident_masked_mae = np.nanmean(incident_masked)
        masked_increase = incident_masked_mae - normal_masked_mae if not (np.isnan(incident_masked_mae) or np.isnan(normal_masked_mae)) else np.nan

        node_data.append({
            'node_idx': node_idx,
            'category': get_category(node_idx),
            'n_incident_samples': n_inc,
            'normal_masked_mae': normal_masked_mae,
            'incident_masked_mae': incident_masked_mae,
            'masked_increase': masked_increase,
        })

    df = pd.DataFrame(node_data)
    # Filter to meaningful nodes (not dead/major_fail)
    df = df[df['category'] == 'functional_or_partial'].copy()
    model_results[model_name] = df
    print(f"  Nodes analyzed: {len(df)}")
    print(f"  Mean masked increase: {df['masked_increase'].mean():.4f}")
    print(f"  Top 10 degraded:")
    print(df.nlargest(10, 'masked_increase')[['node_idx', 'normal_masked_mae', 'incident_masked_mae', 'masked_increase']].to_string(index=False))

# ============================================================================
# Merge and compare
# ============================================================================
print("\n" + "=" * 70)
print("CROSS-MODEL COMPARISON")
print("=" * 70)

df_stae = model_results['STAEformer'].rename(columns={
    'masked_increase': 'stae_increase', 'normal_masked_mae': 'stae_normal', 'incident_masked_mae': 'stae_incident'
})[['node_idx', 'stae_normal', 'stae_incident', 'stae_increase']]

df_stgcn = model_results['STGCN'].rename(columns={
    'masked_increase': 'stgcn_increase', 'normal_masked_mae': 'stgcn_normal', 'incident_masked_mae': 'stgcn_incident'
})[['node_idx', 'stgcn_normal', 'stgcn_incident', 'stgcn_increase']]

df_comp = df_stae.merge(df_stgcn, on='node_idx', how='inner')
print(f"\nNodes in both models: {len(df_comp)}")

# Correlation of degradation patterns
corr, pval = paired_spearmanr(df_comp['stae_increase'], df_comp['stgcn_increase'])
print(f"\nSpearman correlation of masked_increase: r={corr:.4f}, p={pval:.6f}")

pearson_r, pearson_p = stats.pearsonr(df_comp['stae_increase'].dropna(), df_comp['stgcn_increase'].dropna())
print(f"Pearson correlation of masked_increase: r={pearson_r:.4f}, p={pearson_p:.6f}")

# Top degraded overlap
for k in [20, 50]:
    top_stae = set(df_comp.nlargest(k, 'stae_increase')['node_idx'])
    top_stgcn = set(df_comp.nlargest(k, 'stgcn_increase')['node_idx'])
    overlap = top_stae & top_stgcn
    print(f"\nTop {k} degraded overlap: {len(overlap)}/{k} ({len(overlap)/k*100:.0f}%)")
    if overlap:
        print(f"  Shared nodes: {sorted(overlap)}")

# Bottom (resilient) overlap
for k in [20, 50]:
    bot_stae = set(df_comp.nsmallest(k, 'stae_increase')['node_idx'])
    bot_stgcn = set(df_comp.nsmallest(k, 'stgcn_increase')['node_idx'])
    overlap = bot_stae & bot_stgcn
    print(f"\nTop {k} resilient overlap: {len(overlap)}/{k} ({len(overlap)/k*100:.0f}%)")

# Which nodes are consistently degraded?
stae_q75 = df_comp['stae_increase'].quantile(0.75)
stgcn_q75 = df_comp['stgcn_increase'].quantile(0.75)
both_degraded = df_comp[(df_comp['stae_increase'] >= stae_q75) & (df_comp['stgcn_increase'] >= stgcn_q75)]
print(f"\nNodes degraded in BOTH models (top 25%): {len(both_degraded)}")
if len(both_degraded) > 0:
    print(both_degraded[['node_idx', 'stae_increase', 'stgcn_increase']].sort_values('stae_increase', ascending=False).head(20).to_string(index=False))

# ============================================================================
# Visualizations
# ============================================================================
print("\n\nCreating comparison visualizations...")

fig = plt.figure(figsize=(20, 16))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

# 1: Scatter of degradation across models
ax = fig.add_subplot(gs[0, 0])
ax.scatter(df_comp['stae_increase'], df_comp['stgcn_increase'], alpha=0.5, s=15, c='steelblue', edgecolors='none')
max_val = max(df_comp['stae_increase'].abs().max(), df_comp['stgcn_increase'].abs().max())
ax.plot([-max_val, max_val], [-max_val, max_val], 'k--', alpha=0.5, label='y=x')
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('STAEformer Masked MAE Increase')
ax.set_ylabel('STGCN Masked MAE Increase')
ax.set_title(f'Post-Incident Degradation Correlation\nSpearman r={corr:.3f}, p={pval:.4f}')
ax.legend()

# 2: Distribution comparison
ax = fig.add_subplot(gs[0, 1])
bins = np.linspace(min(df_comp['stae_increase'].min(), df_comp['stgcn_increase'].min()),
                   max(df_comp['stae_increase'].max(), df_comp['stgcn_increase'].max()), 50)
ax.hist(df_comp['stae_increase'], bins=bins, alpha=0.6, color='blue', label='STAEformer', edgecolor='black', linewidth=0.3)
ax.hist(df_comp['stgcn_increase'], bins=bins, alpha=0.6, color='red', label='STGCN', edgecolor='black', linewidth=0.3)
ax.axvline(x=0, color='black', linestyle='--', alpha=0.7)
ax.set_xlabel('Masked MAE Increase')
ax.set_ylabel('Number of Nodes')
ax.set_title('Distribution of Post-Incident Degradation')
ax.legend()

# 3: Top 20 degraded comparison (side by side)
ax = fig.add_subplot(gs[0, 2])
top20_stae = df_comp.nlargest(20, 'stae_increase')[['node_idx', 'stae_increase', 'stgcn_increase']]
x = np.arange(len(top20_stae))
width = 0.35
ax.barh(x - width/2, top20_stae['stae_increase'], width, label='STAEformer', color='blue', alpha=0.7)
ax.barh(x + width/2, top20_stae['stgcn_increase'], width, label='STGCN', color='red', alpha=0.7)
ax.set_yticks(x)
ax.set_yticklabels([f"N{n}" for n in top20_stae['node_idx']], fontsize=7)
ax.set_xlabel('Masked MAE Increase')
ax.set_title('Top 20 STAEformer-Degraded Nodes:\nDo They Also Degrade in STGCN?')
ax.invert_yaxis()
ax.legend(fontsize=8)

# 4: Top 20 STGCN-degraded
ax = fig.add_subplot(gs[1, 0])
top20_stgcn = df_comp.nlargest(20, 'stgcn_increase')[['node_idx', 'stae_increase', 'stgcn_increase']]
x = np.arange(len(top20_stgcn))
ax.barh(x - width/2, top20_stgcn['stae_increase'], width, label='STAEformer', color='blue', alpha=0.7)
ax.barh(x + width/2, top20_stgcn['stgcn_increase'], width, label='STGCN', color='red', alpha=0.7)
ax.set_yticks(x)
ax.set_yticklabels([f"N{n}" for n in top20_stgcn['node_idx']], fontsize=7)
ax.set_xlabel('Masked MAE Increase')
ax.set_title('Top 20 STGCN-Degraded Nodes:\nDo They Also Degrade in STAEformer?')
ax.invert_yaxis()
ax.legend(fontsize=8)

# 5: Normal MAE comparison
ax = fig.add_subplot(gs[1, 1])
ax.scatter(df_comp['stae_normal'], df_comp['stgcn_normal'], alpha=0.4, s=10, c='green', edgecolors='none')
max_n = max(df_comp['stae_normal'].max(), df_comp['stgcn_normal'].max())
ax.plot([0, max_n], [0, max_n], 'k--', alpha=0.5)
ax.set_xlabel('STAEformer Normal MAE')
ax.set_ylabel('STGCN Normal MAE')
corr_n, _ = paired_spearmanr(df_comp['stae_normal'], df_comp['stgcn_normal'])
ax.set_title(f'Normal MAE per Node (r={corr_n:.3f})')

# 6: Incident MAE comparison
ax = fig.add_subplot(gs[1, 2])
ax.scatter(df_comp['stae_incident'], df_comp['stgcn_incident'], alpha=0.4, s=10, c='orange', edgecolors='none')
max_i = max(df_comp['stae_incident'].max(), df_comp['stgcn_incident'].max())
ax.plot([0, max_i], [0, max_i], 'k--', alpha=0.5)
ax.set_xlabel('STAEformer Incident MAE')
ax.set_ylabel('STGCN Incident MAE')
corr_i, _ = paired_spearmanr(df_comp['stae_incident'], df_comp['stgcn_incident'])
ax.set_title(f'Incident MAE per Node (r={corr_i:.3f})')

# 7: Quadrant analysis
ax = fig.add_subplot(gs[2, 0])
stae_med = df_comp['stae_increase'].median()
stgcn_med = df_comp['stgcn_increase'].median()
colors = []
labels = {'both_high': 0, 'stae_only': 0, 'stgcn_only': 0, 'both_low': 0}
for _, row in df_comp.iterrows():
    if row['stae_increase'] >= stae_med and row['stgcn_increase'] >= stgcn_med:
        colors.append('red'); labels['both_high'] += 1
    elif row['stae_increase'] >= stae_med:
        colors.append('blue'); labels['stae_only'] += 1
    elif row['stgcn_increase'] >= stgcn_med:
        colors.append('orange'); labels['stgcn_only'] += 1
    else:
        colors.append('green'); labels['both_low'] += 1
ax.scatter(df_comp['stae_increase'], df_comp['stgcn_increase'], c=colors, alpha=0.5, s=15, edgecolors='none')
ax.axhline(y=stgcn_med, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=stae_med, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('STAEformer Increase')
ax.set_ylabel('STGCN Increase')
ax.set_title(f'Quadrant: Both-high={labels["both_high"]}, STAEonly={labels["stae_only"]}\nSTGCNonly={labels["stgcn_only"]}, Both-low={labels["both_low"]}')

# 8: Rank correlation
ax = fig.add_subplot(gs[2, 1])
df_comp['stae_rank'] = df_comp['stae_increase'].rank(ascending=False)
df_comp['stgcn_rank'] = df_comp['stgcn_increase'].rank(ascending=False)
ax.scatter(df_comp['stae_rank'], df_comp['stgcn_rank'], alpha=0.4, s=10, c='purple', edgecolors='none')
ax.plot([0, len(df_comp)], [0, len(df_comp)], 'k--', alpha=0.3)
ax.set_xlabel('STAEformer Degradation Rank')
ax.set_ylabel('STGCN Degradation Rank')
ax.set_title(f'Rank Comparison (Spearman r={corr:.3f})')

# 9: Summary text
ax = fig.add_subplot(gs[2, 2])
ax.axis('off')
summary_text = [
    "CROSS-MODEL COMPARISON SUMMARY",
    "=" * 40,
    f"STAEformer overall masked MAE: {df_comp['stae_normal'].mean():.2f}",
    f"STGCN overall masked MAE: {df_comp['stgcn_normal'].mean():.2f}",
    f"",
    f"STAEformer mean increase: {df_comp['stae_increase'].mean():.3f}",
    f"STGCN mean increase: {df_comp['stgcn_increase'].mean():.3f}",
    f"",
    f"Spearman correlation: {corr:.3f} (p={pval:.4f})",
    f"Pearson correlation: {pearson_r:.3f} (p={pearson_p:.4f})",
    f"",
    f"Top 20 overlap: {len(set(df_comp.nlargest(20, 'stae_increase')['node_idx']) & set(df_comp.nlargest(20, 'stgcn_increase')['node_idx']))}/20",
    f"Top 50 overlap: {len(set(df_comp.nlargest(50, 'stae_increase')['node_idx']) & set(df_comp.nlargest(50, 'stgcn_increase')['node_idx']))}/50",
    f"",
    f"Both-degraded (top 25%): {len(both_degraded)}",
]
ax.text(0.05, 0.95, '\n'.join(summary_text), transform=ax.transAxes,
        fontsize=9, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.savefig(f'{OUTPUT_DIR}/cross_model_comparison.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/cross_model_comparison.png")

# Save comparison CSV
df_comp.to_csv(f'{OUTPUT_DIR}/cross_model_node_comparison.csv', index=False)
print(f"Saved: {OUTPUT_DIR}/cross_model_node_comparison.csv")

# Save comparison summary
comp_summary = {
    'models': ['STAEformer', 'STGCN'],
    'n_nodes_compared': int(len(df_comp)),
    'stae_mean_increase': float(df_comp['stae_increase'].mean()),
    'stgcn_mean_increase': float(df_comp['stgcn_increase'].mean()),
    'spearman_r': float(corr),
    'spearman_p': float(pval),
    'pearson_r': float(pearson_r),
    'pearson_p': float(pearson_p),
    'top20_overlap': int(len(set(df_comp.nlargest(20, 'stae_increase')['node_idx']) & set(df_comp.nlargest(20, 'stgcn_increase')['node_idx']))),
    'top50_overlap': int(len(set(df_comp.nlargest(50, 'stae_increase')['node_idx']) & set(df_comp.nlargest(50, 'stgcn_increase')['node_idx']))),
    'both_degraded_top25pct': int(len(both_degraded)),
}
with open(f'{OUTPUT_DIR}/cross_model_summary.json', 'w') as f:
    json.dump(comp_summary, f, indent=2)
print(f"Saved: {OUTPUT_DIR}/cross_model_summary.json")

print("\n" + "=" * 70)
print("COMPARISON COMPLETE")
print("=" * 70)
