"""
EDA: Prediction Error Analysis with Node/Temporal Characteristics

Correlates actual prediction errors with:
1. Node characteristics (CV, zero_ratio, mean_flow)
2. Temporal patterns (time of day, day of week)
3. Identifies what makes samples/nodes difficult
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from scipy import stats

sys.path.append('/data/pretrainingbasicts')

# Configuration - Use baseline freeze model
CHECKPOINT_DIR = Path('checkpoints/ContextContrastive_freeze_spatial_mask_3mo/xtraffic_SAN_BERNARDINO_30_12_12')
DATA_PATH = 'datasets/xtraffic/SAN_BERNARDINO/data.dat'
OUTPUT_DIR = 'experiments/eda_results'

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("Prediction Error Analysis")
print("=" * 60)

# =============================================================================
# 1. Load Data and Node Statistics
# =============================================================================
print("\n[1] Loading data...")

# Load raw data
data = np.memmap(DATA_PATH, dtype='float32', mode='r').reshape(-1, 893, 5)
print(f"Raw data shape: {data.shape}")

# Load node statistics from previous EDA
node_stats = pd.read_csv(f'{OUTPUT_DIR}/node_statistics.csv')
print(f"Node statistics loaded: {len(node_stats)} nodes")

# Test set indices
total_samples = len(data)
test_start = int(total_samples * 0.8)
INPUT_LEN, OUTPUT_LEN = 12, 12

# =============================================================================
# 2. Run Inference to Get Predictions
# =============================================================================
print("\n[2] Running inference to get predictions...")

from basicts.data import TimeSeriesForecastingDataset
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings
from baselines.STAEformer.arch import STAEformer
from baselines.ContextContrastive.runner import RepresentationLearningRunner
from torch.utils.data import DataLoader

# Find checkpoint
ckpt_dirs = list(CHECKPOINT_DIR.glob('*'))
if not ckpt_dirs:
    raise FileNotFoundError(f"No checkpoint found in {CHECKPOINT_DIR}")
ckpt_dir = sorted(ckpt_dirs)[-1]
print(f"Using checkpoint: {ckpt_dir}")

# Find best model checkpoint
best_ckpt = list(ckpt_dir.glob('*best*.pt'))
if best_ckpt:
    ckpt_path = best_ckpt[0]
else:
    ckpt_path = sorted(ckpt_dir.glob('*.pt'))[-1]
print(f"Loading model from: {ckpt_path}")

# Load config
DATA_NAME = 'xtraffic/SAN_BERNARDINO'
regular_settings = get_regular_settings(DATA_NAME)

# Create test dataset
test_dataset = TimeSeriesForecastingDataset(
    dataset_name=DATA_NAME,
    train_val_test_ratio=[0.6, 0.2, 0.2],
    input_len=INPUT_LEN,
    output_len=OUTPUT_LEN,
    mode='test'
)

# Create scaler
scaler = ZScoreScaler(
    dataset_name=DATA_NAME,
    train_ratio=0.6,
    norm_each_channel=regular_settings['NORM_EACH_CHANNEL'],
    rescale=regular_settings['RESCALE']
)

# Load model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(ckpt_path, map_location=device)

# Build model
D_MODEL = 64
model = STAEformer(
    num_nodes=893,
    in_steps=INPUT_LEN,
    out_steps=OUTPUT_LEN,
    input_dim=D_MODEL + 2,
    output_dim=1,
    steps_per_day=288,
    input_embedding_dim=24,
    tod_embedding_dim=24,
    dow_embedding_dim=24,
    spatial_embedding_dim=0,
    adaptive_embedding_dim=24,
    feed_forward_dim=256,
    num_heads=4,
    num_layers=1,
    dropout=0.1,
    use_mixed_proj=True,
)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Build encoder
from baselines.ContextContrastive.arch import build_encoder
encoder_cfg = {
    'type': 'TransformerEncoder',
    'input_dim': 3,
    'd_model': D_MODEL,
    'num_layers': 2,
    'nhead': 4,
    'dropout': 0.1,
}
encoder = build_encoder(encoder_cfg)
encoder.load_state_dict(checkpoint['encoder_state_dict'])
encoder = encoder.to(device)
encoder.eval()

print(f"Model loaded successfully")

# =============================================================================
# 3. Compute Predictions and Errors
# =============================================================================
print("\n[3] Computing predictions and errors...")

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

all_predictions = []
all_targets = []
all_inputs = []
all_indices = []

FORWARD_FEATURES = [0, 3, 4]  # flow, tod, dow
TARGET_FEATURES = [0]

sample_idx = 0
with torch.no_grad():
    for batch in test_loader:
        # batch: dict with 'inputs', 'target', 'index'
        inputs = batch['inputs'].to(device)  # (B, T, N, C) - history
        target_data = batch['target'].to(device)  # (B, T, N, C) - future

        # Select features
        history_data = inputs[..., FORWARD_FEATURES]

        # Encode
        encoded = encoder.encode(history_data)

        # Add tod/dow
        tod = history_data[..., 1:2]
        dow = history_data[..., 2:3]
        placeholder = encoded[..., 0:1]
        encoded_rest = encoded[..., 1:]
        encoded_history = torch.cat([placeholder, tod, dow, encoded_rest], dim=-1)

        # Predict
        future_data_dec = target_data[..., FORWARD_FEATURES]
        output = model(
            history_data=encoded_history,
            future_data=future_data_dec,
            batch_seen=0,
            epoch=0,
            train=False
        )

        pred = output['prediction']  # (B, T, N, 1)
        target = target_data[..., TARGET_FEATURES]  # (B, T, N, 1)

        # Rescale (scaler expects tensor)
        pred_rescaled = scaler.inverse_transform(pred.cpu()).numpy()
        target_rescaled = scaler.inverse_transform(target.cpu()).numpy()

        all_predictions.append(pred_rescaled)
        all_targets.append(target_rescaled)
        all_indices.extend(range(sample_idx, sample_idx + len(pred)))
        sample_idx += len(pred)

predictions = np.concatenate(all_predictions, axis=0)  # (N_samples, T, N_nodes, 1)
targets = np.concatenate(all_targets, axis=0)

print(f"Predictions shape: {predictions.shape}")
print(f"Targets shape: {targets.shape}")

# Compute errors
errors = np.abs(predictions - targets)  # (N_samples, T, N_nodes, 1)
errors = errors.squeeze(-1)  # (N_samples, T, N_nodes)

# =============================================================================
# 4. Per-Node Error Analysis
# =============================================================================
print("\n[4] Analyzing per-node errors...")

# Per-node MAE (average over samples and time)
per_node_mae = errors.mean(axis=(0, 1))  # (N_nodes,)

# Add to node_stats
node_stats['pred_mae'] = per_node_mae

# Correlations
print("\nCorrelations with prediction MAE:")
for col in ['mean_flow', 'std_flow', 'cv_flow', 'zero_ratio', 'range_flow']:
    corr, pval = stats.spearmanr(node_stats[col], node_stats['pred_mae'])
    print(f"  {col}: r={corr:.3f} (p={pval:.4f})")

# Top worst nodes by actual prediction error
print("\nTop 10 worst nodes by prediction MAE:")
worst_nodes = node_stats.nlargest(10, 'pred_mae')[['node_id', 'pred_mae', 'mean_flow', 'cv_flow', 'zero_ratio']]
print(worst_nodes.to_string())

print("\nTop 10 best nodes by prediction MAE:")
best_nodes = node_stats.nsmallest(10, 'pred_mae')[['node_id', 'pred_mae', 'mean_flow', 'cv_flow', 'zero_ratio']]
print(best_nodes.to_string())

# =============================================================================
# 5. Per-Sample Error Analysis
# =============================================================================
print("\n[5] Analyzing per-sample errors...")

# Per-sample MAE (average over time and nodes)
per_sample_mae = errors.mean(axis=(1, 2))  # (N_samples,)

# Get temporal info for each sample
test_tod = data[test_start:test_start+len(per_sample_mae), 0, 3]  # (N_samples,)
test_dow = data[test_start:test_start+len(per_sample_mae), 0, 4]

test_tod_hour = (test_tod * 288 / 12).astype(int)  # Convert to hour
test_dow_int = (test_dow * 7).astype(int)

sample_df = pd.DataFrame({
    'sample_idx': range(len(per_sample_mae)),
    'mae': per_sample_mae,
    'hour': test_tod_hour,
    'dow': test_dow_int,
})

# Error by hour
hourly_error = sample_df.groupby('hour')['mae'].agg(['mean', 'std', 'count']).reset_index()
print("\nMAE by hour of day (top 5 worst hours):")
print(hourly_error.nlargest(5, 'mean').to_string())

# Error by day of week
daily_error = sample_df.groupby('dow')['mae'].agg(['mean', 'std', 'count']).reset_index()
print("\nMAE by day of week:")
print(daily_error.to_string())

# =============================================================================
# 6. Worst-K% Sample Analysis
# =============================================================================
print("\n[6] Analyzing worst-K% samples...")

# Worst 5% samples
worst_5pct_threshold = np.percentile(per_sample_mae, 95)
worst_5pct_mask = per_sample_mae >= worst_5pct_threshold
worst_5pct_samples = sample_df[worst_5pct_mask]

print(f"\nWorst 5% samples (MAE >= {worst_5pct_threshold:.2f}):")
print(f"  Count: {len(worst_5pct_samples)}")
print(f"  Hour distribution:")
print(worst_5pct_samples.groupby('hour').size().nlargest(5).to_string())
print(f"\n  Day distribution:")
print(worst_5pct_samples.groupby('dow').size().to_string())

# Compare worst 5% vs rest
rest_samples = sample_df[~worst_5pct_mask]
print(f"\nComparison - Hour distribution:")
print(f"  Worst 5% peak hours: {worst_5pct_samples['hour'].mode().values}")
print(f"  Rest peak hours: {rest_samples['hour'].mode().values}")

# =============================================================================
# 7. Visualizations
# =============================================================================
print("\n[7] Creating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 7.1 Prediction MAE vs CV
ax = axes[0, 0]
ax.scatter(node_stats['cv_flow'], node_stats['pred_mae'], alpha=0.5, s=10)
ax.set_xlabel('Coefficient of Variation')
ax.set_ylabel('Prediction MAE')
ax.set_title('Node CV vs Prediction Error')
# Add trend line
z = np.polyfit(node_stats['cv_flow'].fillna(0), node_stats['pred_mae'], 1)
p = np.poly1d(z)
x_line = np.linspace(0, node_stats['cv_flow'].max(), 100)
ax.plot(x_line, p(x_line), 'r--', alpha=0.8, label=f'Trend')
ax.legend()

# 7.2 Prediction MAE vs Zero Ratio
ax = axes[0, 1]
ax.scatter(node_stats['zero_ratio'], node_stats['pred_mae'], alpha=0.5, s=10)
ax.set_xlabel('Zero-Flow Ratio')
ax.set_ylabel('Prediction MAE')
ax.set_title('Missing Data vs Prediction Error')

# 7.3 MAE by hour
ax = axes[0, 2]
ax.bar(hourly_error['hour'], hourly_error['mean'], yerr=hourly_error['std']/np.sqrt(hourly_error['count']), capsize=2, alpha=0.7)
ax.set_xlabel('Hour of Day')
ax.set_ylabel('MAE')
ax.set_title('Prediction Error by Hour')
ax.axhline(y=per_sample_mae.mean(), color='r', linestyle='--', label='Overall Mean')
ax.legend()

# 7.4 MAE by day of week
ax = axes[1, 0]
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
ax.bar(range(7), daily_error['mean'], yerr=daily_error['std']/np.sqrt(daily_error['count']), capsize=3)
ax.set_xticks(range(7))
ax.set_xticklabels(days)
ax.set_ylabel('MAE')
ax.set_title('Prediction Error by Day of Week')

# 7.5 Worst 5% hour distribution
ax = axes[1, 1]
worst_hour_dist = worst_5pct_samples.groupby('hour').size()
rest_hour_dist = rest_samples.groupby('hour').size()
# Normalize
worst_hour_pct = worst_hour_dist / worst_hour_dist.sum() * 100
rest_hour_pct = rest_hour_dist / rest_hour_dist.sum() * 100
ax.plot(worst_hour_pct.index, worst_hour_pct.values, 'r-', label='Worst 5%', linewidth=2)
ax.plot(rest_hour_pct.index, rest_hour_pct.values, 'b-', label='Rest 95%', linewidth=2)
ax.set_xlabel('Hour of Day')
ax.set_ylabel('% of Samples')
ax.set_title('Worst 5% vs Rest: Hour Distribution')
ax.legend()

# 7.6 Node MAE distribution with worst highlighted
ax = axes[1, 2]
ax.hist(node_stats['pred_mae'], bins=50, alpha=0.7, edgecolor='black')
worst_mae_threshold = node_stats['pred_mae'].quantile(0.95)
ax.axvline(x=worst_mae_threshold, color='r', linestyle='--', label=f'95th percentile')
ax.set_xlabel('Prediction MAE')
ax.set_ylabel('Number of Nodes')
ax.set_title('Distribution of Node-wise Prediction MAE')
ax.legend()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/prediction_error_analysis.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/prediction_error_analysis.png")

# =============================================================================
# 8. Summary
# =============================================================================
print("\n" + "=" * 60)
print("ANALYSIS SUMMARY")
print("=" * 60)

print("\n📊 Key Correlations with Prediction Error:")
corr_cv, _ = stats.spearmanr(node_stats['cv_flow'].fillna(0), node_stats['pred_mae'])
corr_zero, _ = stats.spearmanr(node_stats['zero_ratio'], node_stats['pred_mae'])
corr_flow, _ = stats.spearmanr(node_stats['mean_flow'], node_stats['pred_mae'])
print(f"  - CV (variability) ↔ MAE: r={corr_cv:.3f}")
print(f"  - Zero ratio ↔ MAE: r={corr_zero:.3f}")
print(f"  - Mean flow ↔ MAE: r={corr_flow:.3f}")

print("\n⏰ Temporal Patterns:")
worst_hour = hourly_error.loc[hourly_error['mean'].idxmax()]
best_hour = hourly_error.loc[hourly_error['mean'].idxmin()]
print(f"  - Worst hour: {int(worst_hour['hour'])}:00 (MAE={worst_hour['mean']:.2f})")
print(f"  - Best hour: {int(best_hour['hour'])}:00 (MAE={best_hour['mean']:.2f})")
print(f"  - Ratio: {worst_hour['mean']/best_hour['mean']:.2f}x")

worst_day = daily_error.loc[daily_error['mean'].idxmax()]
best_day = daily_error.loc[daily_error['mean'].idxmin()]
print(f"  - Worst day: {days[int(worst_day['dow'])]} (MAE={worst_day['mean']:.2f})")
print(f"  - Best day: {days[int(best_day['dow'])]} (MAE={best_day['mean']:.2f})")

print("\n🎯 Worst 5% Sample Characteristics:")
print(f"  - Concentrated in hours: {list(worst_5pct_samples.groupby('hour').size().nlargest(3).index)}")
print(f"  - Most common day: {days[worst_5pct_samples['dow'].mode().values[0]]}")

print("\n🔍 Actionable Insights:")
if corr_cv > 0.3:
    print("  ✓ High CV nodes need special attention (strong correlation with error)")
if abs(corr_zero) > 0.3:
    print("  ✓ Zero-flow ratio affects prediction quality")
if worst_hour['mean']/best_hour['mean'] > 1.5:
    print("  ✓ Time-aware loss weighting could help (large hourly variance)")

# Save updated node stats
node_stats.to_csv(f'{OUTPUT_DIR}/node_statistics_with_errors.csv', index=False)
print(f"\nSaved: {OUTPUT_DIR}/node_statistics_with_errors.csv")

print("\n✅ Analysis Complete!")
