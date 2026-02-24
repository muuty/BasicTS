"""
Cross-Year Contrastive Learning: False Positive Analysis

Analyzes whether same-node same-time-of-day/day-of-week pairs from different years
are truly similar (valid positive pairs) or "false positives" due to pattern changes.
"""

import numpy as np
import json
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

print("[STAGE:begin:data_loading]")
print("[OBJECTIVE] Analyze cross-year positive pair false positive rates for contrastive learning")

# Load Q1 data from pre-split datasets
dataset_base = Path("/data/pretrainingbasicts/datasets")
num_nodes = 893
num_features = 5
q1_length = 25920  # 90 days × 288 steps/day

print(f"[DATA] Loading Q1 datasets from {dataset_base}")

year_2022_q1 = np.array(np.memmap(dataset_base / "SAN_BERNARDINO_2022_Q1/data.dat",
                                   dtype='float32', mode='r', shape=(q1_length, num_nodes, num_features)))
year_2023_q1 = np.array(np.memmap(dataset_base / "SAN_BERNARDINO_2023_Q1/data.dat",
                                   dtype='float32', mode='r', shape=(q1_length, num_nodes, num_features)))
year_2024_q1 = np.array(np.memmap(dataset_base / "SAN_BERNARDINO_2024_Q1/data.dat",
                                   dtype='float32', mode='r', shape=(q1_length, num_nodes, num_features))[:q1_length])

print(f"[DATA] Loaded Q1 data: 2022 {year_2022_q1.shape}, 2023 {year_2023_q1.shape}, 2024 {year_2024_q1.shape}")
print(f"[STAGE:status:success]")
print(f"[STAGE:end:data_loading]")

# Load sensor health indices
print("\n[STAGE:begin:sensor_health]")
dataset_dir = Path("/data/pretrainingbasicts/datasets/xtraffic/SAN_BERNARDINO")
dead_indices = np.load(dataset_dir / "dead_indices.npy")
major_fail_indices = np.load(dataset_dir / "major_fail_indices.npy")

print(f"[DATA] Dead sensors: {len(dead_indices)}, Major fail: {len(major_fail_indices)}")

# Create sensor category mapping
sensor_category = {}
for idx in dead_indices:
    sensor_category[idx] = 'dead'
for idx in major_fail_indices:
    sensor_category[idx] = 'major_fail'
for idx in range(num_nodes):
    if idx not in sensor_category:
        sensor_category[idx] = 'functional'

print(f"[STAT:dead] {len(dead_indices)}")
print(f"[STAT:major_fail] {len(major_fail_indices)}")
print(f"[STAT:functional] {num_nodes - len(dead_indices) - len(major_fail_indices)}")
print(f"[STAGE:status:success]")
print(f"[STAGE:end:sensor_health]")

# Compute average daily profiles
print("\n[STAGE:begin:daily_profiles]")
steps_per_day = 288

def compute_daily_profile(data_q1):
    """Compute average daily profile (288 timesteps) from Q1 data."""
    # data_q1: (25920, num_nodes, num_features)
    # Reshape to (90 days, 288 steps, num_nodes, num_features)
    num_days = data_q1.shape[0] // steps_per_day
    reshaped = data_q1.reshape(num_days, steps_per_day, num_nodes, num_features)

    # Average over days: (288, num_nodes, num_features)
    daily_profile = reshaped.mean(axis=0)
    return daily_profile

profile_2022 = compute_daily_profile(year_2022_q1)  # (288, 893, 5)
profile_2023 = compute_daily_profile(year_2023_q1)
profile_2024 = compute_daily_profile(year_2024_q1)

print(f"[DATA] Daily profiles computed: {profile_2022.shape}")
print(f"[STAGE:status:success]")
print(f"[STAGE:end:daily_profiles]")

# Extract flow channel (channel 0)
print("\n[STAGE:begin:cross_year_correlation]")
flow_2022 = profile_2022[:, :, 0]  # (288, 893)
flow_2023 = profile_2023[:, :, 0]
flow_2024 = profile_2024[:, :, 0]

# Compute correlations and cosine similarity for each node
correlations_2022_2023 = []
correlations_2022_2024 = []
correlations_2023_2024 = []

cosine_2022_2023 = []
cosine_2022_2024 = []
cosine_2023_2024 = []

for node_idx in range(num_nodes):
    # Pearson correlation
    if np.std(flow_2022[:, node_idx]) > 0 and np.std(flow_2023[:, node_idx]) > 0:
        r_22_23, _ = pearsonr(flow_2022[:, node_idx], flow_2023[:, node_idx])
    else:
        r_22_23 = np.nan

    if np.std(flow_2022[:, node_idx]) > 0 and np.std(flow_2024[:, node_idx]) > 0:
        r_22_24, _ = pearsonr(flow_2022[:, node_idx], flow_2024[:, node_idx])
    else:
        r_22_24 = np.nan

    if np.std(flow_2023[:, node_idx]) > 0 and np.std(flow_2024[:, node_idx]) > 0:
        r_23_24, _ = pearsonr(flow_2023[:, node_idx], flow_2024[:, node_idx])
    else:
        r_23_24 = np.nan

    correlations_2022_2023.append(r_22_23)
    correlations_2022_2024.append(r_22_24)
    correlations_2023_2024.append(r_23_24)

    # Cosine similarity
    cos_22_23 = cosine_similarity([flow_2022[:, node_idx]], [flow_2023[:, node_idx]])[0, 0]
    cos_22_24 = cosine_similarity([flow_2022[:, node_idx]], [flow_2024[:, node_idx]])[0, 0]
    cos_23_24 = cosine_similarity([flow_2023[:, node_idx]], [flow_2024[:, node_idx]])[0, 0]

    cosine_2022_2023.append(cos_22_23)
    cosine_2022_2024.append(cos_22_24)
    cosine_2023_2024.append(cos_23_24)

correlations_2022_2023 = np.array(correlations_2022_2023)
correlations_2022_2024 = np.array(correlations_2022_2024)
correlations_2023_2024 = np.array(correlations_2023_2024)

cosine_2022_2023 = np.array(cosine_2022_2023)
cosine_2022_2024 = np.array(cosine_2022_2024)
cosine_2023_2024 = np.array(cosine_2023_2024)

print(f"[FINDING] Cross-year correlations computed for {num_nodes} nodes")
print(f"[STAT:2022_2023_mean_r] {np.nanmean(correlations_2022_2023):.3f}")
print(f"[STAT:2022_2024_mean_r] {np.nanmean(correlations_2022_2024):.3f}")
print(f"[STAT:2023_2024_mean_r] {np.nanmean(correlations_2023_2024):.3f}")
print(f"[STAGE:status:success]")
print(f"[STAGE:end:cross_year_correlation]")

# Classify nodes by similarity
print("\n[STAGE:begin:similarity_classification]")

def classify_similarity(r):
    """Classify correlation into similarity categories."""
    if np.isnan(r):
        return 'invalid'
    elif r > 0.9:
        return 'high'  # Safe positive pair
    elif r > 0.7:
        return 'medium'  # Borderline
    elif r > 0:
        return 'low'  # Likely false positive
    else:
        return 'negative'  # Definitely false positive

# Average correlation across year pairs for each node
avg_correlation = np.nanmean([correlations_2022_2023, correlations_2022_2024, correlations_2023_2024], axis=0)

similarity_classes = [classify_similarity(r) for r in avg_correlation]

# Count by class
from collections import Counter
class_counts = Counter(similarity_classes)

print("[FINDING] Cross-year similarity classification:")
for cls in ['high', 'medium', 'low', 'negative', 'invalid']:
    count = class_counts[cls]
    pct = 100 * count / num_nodes
    print(f"  {cls}: {count} ({pct:.1f}%)")
    print(f"[STAT:similarity_{cls}] {count}")

print(f"[STAGE:status:success]")
print(f"[STAGE:end:similarity_classification]")

# Sensor state change analysis
print("\n[STAGE:begin:sensor_state_change]")

def compute_zero_rate(data_q1, channel=0):
    """Compute zero rate for each node in Q1 data."""
    # data_q1: (25920, 893, 5)
    flow_data = data_q1[:, :, channel]  # (25920, 893)
    zero_rate = (flow_data == 0).sum(axis=0) / data_q1.shape[0]
    return zero_rate

zero_rate_2022 = compute_zero_rate(year_2022_q1)
zero_rate_2023 = compute_zero_rate(year_2023_q1)
zero_rate_2024 = compute_zero_rate(year_2024_q1)

# Define dead as >90% zero
dead_threshold = 0.9
dead_2022 = zero_rate_2022 > dead_threshold
dead_2023 = zero_rate_2023 > dead_threshold
dead_2024 = zero_rate_2024 > dead_threshold

# Count state changes
state_change_22_23 = np.sum(dead_2022 != dead_2023)
state_change_22_24 = np.sum(dead_2022 != dead_2024)
state_change_23_24 = np.sum(dead_2023 != dead_2024)

print(f"[FINDING] Sensor state changes (dead <-> alive):")
print(f"  2022-2023: {state_change_22_23} nodes changed")
print(f"  2022-2024: {state_change_22_24} nodes changed")
print(f"  2023-2024: {state_change_23_24} nodes changed")
print(f"[STAT:state_change_22_23] {state_change_22_23}")
print(f"[STAT:state_change_22_24] {state_change_22_24}")
print(f"[STAT:state_change_23_24] {state_change_23_24}")

# Identify nodes with state changes
state_changed_any = (dead_2022 != dead_2023) | (dead_2022 != dead_2024) | (dead_2023 != dead_2024)
num_state_changed = np.sum(state_changed_any)
print(f"[FINDING] Total nodes with ANY state change: {num_state_changed}")
print(f"[STAT:total_state_changed] {num_state_changed}")

print(f"[STAGE:status:success]")
print(f"[STAGE:end:sensor_state_change]")

# False positive analysis by sensor category
print("\n[STAGE:begin:false_positive_by_category]")

# Define false positive as low or negative similarity
def is_false_positive(r):
    return r < 0.7 or np.isnan(r)

fp_by_category = {
    'dead': [],
    'major_fail': [],
    'functional': []
}

for node_idx in range(num_nodes):
    category = sensor_category[node_idx]
    is_fp = is_false_positive(avg_correlation[node_idx])
    fp_by_category[category].append(is_fp)

print("[FINDING] False positive rates by sensor category:")
for category in ['dead', 'major_fail', 'functional']:
    fp_list = fp_by_category[category]
    if len(fp_list) > 0:
        fp_rate = np.mean(fp_list)
        fp_count = np.sum(fp_list)
        total = len(fp_list)
        print(f"  {category}: {fp_count}/{total} = {fp_rate:.1%}")
        print(f"[STAT:fp_rate_{category}] {fp_rate:.3f}")
    else:
        print(f"  {category}: N/A")

print(f"[STAGE:status:success]")
print(f"[STAGE:end:false_positive_by_category]")

# Detailed correlation distribution
print("\n[STAGE:begin:correlation_distribution]")

valid_corr_22_23 = correlations_2022_2023[~np.isnan(correlations_2022_2023)]
valid_corr_22_24 = correlations_2022_2024[~np.isnan(correlations_2022_2024)]
valid_corr_23_24 = correlations_2023_2024[~np.isnan(correlations_2023_2024)]

print("[FINDING] Correlation distribution statistics:")
print("\n2022-2023:")
print(f"  Mean: {np.mean(valid_corr_22_23):.3f}")
print(f"  Median: {np.median(valid_corr_22_23):.3f}")
print(f"  Std: {np.std(valid_corr_22_23):.3f}")
print(f"  Min: {np.min(valid_corr_22_23):.3f}")
print(f"  Max: {np.max(valid_corr_22_23):.3f}")
print(f"  Q1: {np.percentile(valid_corr_22_23, 25):.3f}")
print(f"  Q3: {np.percentile(valid_corr_22_23, 75):.3f}")

print("\n2022-2024:")
print(f"  Mean: {np.mean(valid_corr_22_24):.3f}")
print(f"  Median: {np.median(valid_corr_22_24):.3f}")
print(f"  Std: {np.std(valid_corr_22_24):.3f}")
print(f"  Min: {np.min(valid_corr_22_24):.3f}")
print(f"  Max: {np.max(valid_corr_22_24):.3f}")

print("\n2023-2024:")
print(f"  Mean: {np.mean(valid_corr_23_24):.3f}")
print(f"  Median: {np.median(valid_corr_23_24):.3f}")
print(f"  Std: {np.std(valid_corr_23_24):.3f}")
print(f"  Min: {np.min(valid_corr_23_24):.3f}")
print(f"  Max: {np.max(valid_corr_23_24):.3f}")

print(f"[STAGE:status:success]")
print(f"[STAGE:end:correlation_distribution]")

# Save detailed results
print("\n[STAGE:begin:save_results]")

output_dir = Path("/data/pretrainingbasicts/eda/concept_drift")
output_dir.mkdir(parents=True, exist_ok=True)

# Save per-node results
results = {
    'correlations_2022_2023': correlations_2022_2023.tolist(),
    'correlations_2022_2024': correlations_2022_2024.tolist(),
    'correlations_2023_2024': correlations_2023_2024.tolist(),
    'cosine_2022_2023': cosine_2022_2023.tolist(),
    'cosine_2022_2024': cosine_2022_2024.tolist(),
    'cosine_2023_2024': cosine_2023_2024.tolist(),
    'avg_correlation': avg_correlation.tolist(),
    'similarity_class': similarity_classes,
    'sensor_category': [sensor_category[i] for i in range(num_nodes)],
    'state_changed': state_changed_any.tolist(),
    'zero_rate_2022': zero_rate_2022.tolist(),
    'zero_rate_2023': zero_rate_2023.tolist(),
    'zero_rate_2024': zero_rate_2024.tolist(),
}

results_path = output_dir / "cross_year_correlations.json"
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"[FINDING] Detailed results saved to {results_path}")

# Save numpy arrays for further analysis
np.save(output_dir / "avg_correlation.npy", avg_correlation)
np.save(output_dir / "flow_profile_2022.npy", flow_2022)
np.save(output_dir / "flow_profile_2023.npy", flow_2023)
np.save(output_dir / "flow_profile_2024.npy", flow_2024)

print(f"[FINDING] Numpy arrays saved for further analysis")
print(f"[STAGE:status:success]")
print(f"[STAGE:end:save_results]")

# Summary and implications
print("\n" + "="*80)
print("SUMMARY: Cross-Year Contrastive Learning Positive Pair Analysis")
print("="*80)

print("\n1. OVERALL FALSE POSITIVE RATE:")
overall_fp_rate = np.mean([is_false_positive(r) for r in avg_correlation])
print(f"   {overall_fp_rate:.1%} of nodes have r < 0.7 (likely false positives)")
print(f"   [STAT:overall_fp_rate] {overall_fp_rate:.3f}")

print("\n2. FALSE POSITIVE BY SENSOR HEALTH:")
for category in ['functional', 'major_fail', 'dead']:
    fp_list = fp_by_category[category]
    if len(fp_list) > 0:
        fp_rate = np.mean(fp_list)
        print(f"   {category.upper()}: {fp_rate:.1%}")

print("\n3. SENSOR STATE CHANGES:")
print(f"   {num_state_changed} nodes ({100*num_state_changed/num_nodes:.1f}%) changed dead/alive status")
print(f"   These are GUARANTEED false positives")

print("\n4. IMPLICATIONS FOR CONTRASTIVE LEARNING:")
if overall_fp_rate < 0.3:
    print("   ✓ LOW false positive rate - cross-year pairs are generally valid")
    print("   ✓ Recommendation: Use cross-year CL with confidence")
elif overall_fp_rate < 0.5:
    print("   ⚠ MEDIUM false positive rate - some noise expected")
    print("   ⚠ Recommendation: Filter by correlation threshold (r > 0.7) or sensor health")
else:
    print("   ✗ HIGH false positive rate - cross-year pairs unreliable")
    print("   ✗ Recommendation: Avoid cross-year CL or use strict filtering")

print("\n5. FILTERING STRATEGIES:")
print(f"   Option A: Exclude nodes with state changes → keep {num_nodes - num_state_changed} nodes")
print(f"   Option B: Only functional sensors → keep {len(fp_by_category['functional'])} nodes")
functional_good = len([x for x in fp_by_category['functional'] if not x])
print(f"   Option C: Functional + r>0.7 → keep {functional_good} nodes")

print("\n6. CORRELATION METRICS:")
print(f"   2022-2023: mean r={np.nanmean(correlations_2022_2023):.3f}, median r={np.nanmedian(correlations_2022_2023):.3f}")
print(f"   2022-2024: mean r={np.nanmean(correlations_2022_2024):.3f}, median r={np.nanmedian(correlations_2022_2024):.3f}")
print(f"   2023-2024: mean r={np.nanmean(correlations_2023_2024):.3f}, median r={np.nanmedian(correlations_2023_2024):.3f}")

print("\n[LIMITATION] Analysis based on Q1 data only (90 days) - full-year patterns may differ")
print("[LIMITATION] Dead sensor definition (>90% zero) may not capture all malfunction types")
print("[LIMITATION] Correlation measures temporal pattern similarity, not scale similarity")

print("\n" + "="*80)
print("Analysis complete. Results saved to:")
print(f"  {results_path}")
print(f"  {output_dir}/avg_correlation.npy")
print(f"  {output_dir}/flow_profile_*.npy")
print("="*80)
