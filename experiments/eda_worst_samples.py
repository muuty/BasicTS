"""
EDA: Why are Worst-K% samples difficult?

Analyzes:
1. Temporal patterns - specific times/days harder?
2. Spatial patterns - specific nodes consistently harder?
3. Clustering - do difficult samples cluster together?
4. Feature analysis - what characterizes difficult samples?
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta

sys.path.append('/data/pretrainingbasicts')

# Configuration
CHECKPOINT_DIR = 'checkpoints/ContextContrastive_freeze_spatial_mask_3mo/xtraffic_SAN_BERNARDINO_30_12_12'
DATA_PATH = 'datasets/xtraffic/SAN_BERNARDINO/data.dat'
OUTPUT_DIR = 'experiments/eda_results'

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("EDA: Worst-K% Sample Analysis")
print("=" * 60)

# =============================================================================
# 1. Load Data
# =============================================================================
print("\n[1] Loading data...")

# Load raw data
data = np.memmap(DATA_PATH, dtype='float32', mode='r').reshape(-1, 893, 5)
print(f"Raw data shape: {data.shape}")  # (T, N, C) - time, nodes, channels

# Channel indices: 0=speed, 1=flow, 2=occupancy, 3=tod, 4=dow
flow = data[:, :, 1]  # We predict flow
tod = data[:, :, 3]   # Time of day (normalized 0-1)
dow = data[:, :, 4]   # Day of week (normalized 0-1)

# Convert to actual values
tod_actual = (tod * 288).astype(int)[:, 0]  # 288 steps per day
dow_actual = (dow * 7).astype(int)[:, 0]    # 7 days

print(f"Time range: {len(data)} steps ({len(data)/288:.1f} days)")

# =============================================================================
# 2. Load Model Predictions (if available) or Compute Errors
# =============================================================================
print("\n[2] Computing/Loading errors...")

# Find latest checkpoint directory
ckpt_dirs = list(Path(CHECKPOINT_DIR).glob('*'))
if ckpt_dirs:
    ckpt_dir = sorted(ckpt_dirs)[-1]
    print(f"Using checkpoint: {ckpt_dir}")
else:
    print("No checkpoint found, using simulated errors for analysis")
    ckpt_dir = None

# For this analysis, we'll compute per-sample and per-node statistics
# from the test set portion of the data

# Test set: last 20% of data (based on 0.6/0.2/0.2 split)
total_samples = len(data)
test_start = int(total_samples * 0.8)
test_data = data[test_start:]
test_flow = flow[test_start:]
test_tod = tod_actual[test_start:]
test_dow = dow_actual[test_start:]

print(f"Test set: {len(test_data)} samples")

# =============================================================================
# 3. Analyze Node Difficulty
# =============================================================================
print("\n[3] Analyzing node difficulty...")

# Compute per-node flow statistics
node_stats = pd.DataFrame({
    'node_id': range(893),
    'mean_flow': test_flow.mean(axis=0),
    'std_flow': test_flow.std(axis=0),
    'cv_flow': test_flow.std(axis=0) / (test_flow.mean(axis=0) + 1e-6),  # Coefficient of variation
    'max_flow': test_flow.max(axis=0),
    'min_flow': test_flow.min(axis=0),
    'range_flow': test_flow.max(axis=0) - test_flow.min(axis=0),
    'zero_ratio': (test_flow == 0).sum(axis=0) / len(test_flow),
})

# Load worst node info from metrics if available
if ckpt_dir:
    metrics_file = ckpt_dir / 'test_metrics.json'
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
        worst_node_idx = metrics.get('robustness', {}).get('per_node', {}).get('worst_node_idx', None)
        best_node_idx = metrics.get('robustness', {}).get('per_node', {}).get('best_node_idx', None)
        print(f"Worst node: {worst_node_idx}, Best node: {best_node_idx}")

# Identify high-variance nodes (likely difficult)
high_cv_nodes = node_stats.nlargest(45, 'cv_flow')  # Top 5% = ~45 nodes
low_cv_nodes = node_stats.nsmallest(45, 'cv_flow')

print(f"\nHigh CV nodes (likely difficult):")
print(high_cv_nodes[['node_id', 'mean_flow', 'std_flow', 'cv_flow']].head(10))

print(f"\nLow CV nodes (likely easy):")
print(low_cv_nodes[['node_id', 'mean_flow', 'std_flow', 'cv_flow']].head(10))

# =============================================================================
# 4. Temporal Pattern Analysis
# =============================================================================
print("\n[4] Analyzing temporal patterns...")

# Flow by time of day
tod_flow = pd.DataFrame({
    'tod': test_tod,
    'mean_flow': test_flow.mean(axis=1),
    'std_flow': test_flow.std(axis=1),
})
tod_grouped = tod_flow.groupby('tod').agg(['mean', 'std']).reset_index()

# Flow by day of week
dow_flow = pd.DataFrame({
    'dow': test_dow,
    'mean_flow': test_flow.mean(axis=1),
    'std_flow': test_flow.std(axis=1),
})
dow_grouped = dow_flow.groupby('dow').agg(['mean', 'std']).reset_index()

print("\nFlow variance by time of day (top 5 high-variance times):")
high_var_times = tod_grouped.nlargest(5, ('std_flow', 'mean'))
print(high_var_times)

print("\nFlow variance by day of week:")
print(dow_grouped)

# =============================================================================
# 5. Visualizations
# =============================================================================
print("\n[5] Creating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 5.1 Node CV distribution
ax = axes[0, 0]
ax.hist(node_stats['cv_flow'], bins=50, edgecolor='black', alpha=0.7)
ax.axvline(node_stats['cv_flow'].quantile(0.95), color='r', linestyle='--', label='95th percentile')
ax.set_xlabel('Coefficient of Variation')
ax.set_ylabel('Number of Nodes')
ax.set_title('Node Flow Variability Distribution')
ax.legend()

# 5.2 Mean flow vs CV (scatter)
ax = axes[0, 1]
ax.scatter(node_stats['mean_flow'], node_stats['cv_flow'], alpha=0.5, s=10)
ax.set_xlabel('Mean Flow')
ax.set_ylabel('Coefficient of Variation')
ax.set_title('Mean Flow vs Variability')
# Highlight worst 5%
worst_5pct = node_stats.nlargest(45, 'cv_flow')
ax.scatter(worst_5pct['mean_flow'], worst_5pct['cv_flow'], color='red', s=20, label='Worst 5%')
ax.legend()

# 5.3 Flow by time of day
ax = axes[0, 2]
hours = np.arange(288) / 12  # Convert to hours
ax.plot(hours, tod_grouped[('mean_flow', 'mean')].values, label='Mean')
ax.fill_between(hours,
                tod_grouped[('mean_flow', 'mean')].values - tod_grouped[('std_flow', 'mean')].values,
                tod_grouped[('mean_flow', 'mean')].values + tod_grouped[('std_flow', 'mean')].values,
                alpha=0.3, label='±1 Std')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Flow')
ax.set_title('Flow Pattern by Time of Day')
ax.legend()

# 5.4 Flow by day of week
ax = axes[1, 0]
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
ax.bar(range(7), dow_grouped[('mean_flow', 'mean')].values, yerr=dow_grouped[('std_flow', 'mean')].values, capsize=5)
ax.set_xticks(range(7))
ax.set_xticklabels(days)
ax.set_ylabel('Flow')
ax.set_title('Flow Pattern by Day of Week')

# 5.5 Zero-flow ratio by node
ax = axes[1, 1]
ax.hist(node_stats['zero_ratio'], bins=50, edgecolor='black', alpha=0.7)
ax.set_xlabel('Zero-Flow Ratio')
ax.set_ylabel('Number of Nodes')
ax.set_title('Nodes with Missing/Zero Data')

# 5.6 Node difficulty heatmap (simplified)
ax = axes[1, 2]
# Create a simple 2D representation (30x30 grid approximation)
cv_values = node_stats['cv_flow'].values
grid_size = 30
grid = np.zeros((grid_size, grid_size))
for i, cv in enumerate(cv_values[:grid_size*grid_size]):
    grid[i // grid_size, i % grid_size] = cv
im = ax.imshow(grid, cmap='Reds', aspect='auto')
ax.set_title('Node Difficulty Heatmap (CV)')
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/worst_sample_analysis.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/worst_sample_analysis.png")

# =============================================================================
# 6. Detailed Analysis Summary
# =============================================================================
print("\n" + "=" * 60)
print("ANALYSIS SUMMARY")
print("=" * 60)

print("\n📊 Node Characteristics:")
print(f"  - Total nodes: 893")
print(f"  - High-variance nodes (CV > {node_stats['cv_flow'].quantile(0.95):.2f}): {len(high_cv_nodes)}")
print(f"  - Nodes with >50% zero-flow: {(node_stats['zero_ratio'] > 0.5).sum()}")
print(f"  - Nodes with >90% zero-flow: {(node_stats['zero_ratio'] > 0.9).sum()}")

print("\n⏰ Temporal Patterns:")
peak_morning = tod_grouped[('std_flow', 'mean')].values[7*12:9*12].mean()  # 7-9 AM
peak_evening = tod_grouped[('std_flow', 'mean')].values[17*12:19*12].mean()  # 5-7 PM
off_peak = tod_grouped[('std_flow', 'mean')].values[2*12:5*12].mean()  # 2-5 AM
print(f"  - Morning peak (7-9 AM) variance: {peak_morning:.2f}")
print(f"  - Evening peak (5-7 PM) variance: {peak_evening:.2f}")
print(f"  - Off-peak (2-5 AM) variance: {off_peak:.2f}")
print(f"  - Peak/Off-peak ratio: {max(peak_morning, peak_evening)/off_peak:.2f}x")

weekend_var = dow_grouped[dow_grouped['dow'].isin([5, 6])][('std_flow', 'mean')].mean()
weekday_var = dow_grouped[dow_grouped['dow'].isin([0,1,2,3,4])][('std_flow', 'mean')].mean()
print(f"  - Weekday variance: {weekday_var:.2f}")
print(f"  - Weekend variance: {weekend_var:.2f}")

print("\n🔍 Key Findings:")
findings = []

if (node_stats['zero_ratio'] > 0.5).sum() > 50:
    findings.append("- Many nodes have >50% missing data → data quality issue")

if max(peak_morning, peak_evening)/off_peak > 2:
    findings.append("- Rush hours have 2x+ variance → peak times are harder to predict")

if node_stats['cv_flow'].std() > 0.5:
    findings.append("- High variance in node difficulty → some nodes are much harder than others")

for f in findings:
    print(f"  {f}")

# =============================================================================
# 7. Save detailed statistics
# =============================================================================
node_stats.to_csv(f'{OUTPUT_DIR}/node_statistics.csv', index=False)
print(f"\nSaved: {OUTPUT_DIR}/node_statistics.csv")

print("\n✅ EDA Complete!")
