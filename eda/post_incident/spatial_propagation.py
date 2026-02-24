"""
Spatial Propagation Analysis: How Far Do Incidents Propagate?
=============================================================
For each incident:
  - Find top-K closest neighbors (by adjacency weight)
  - Check if those neighbors show anomalous patterns around the incident
  - Use broadened time window (not just incident start)
  - Recompute incident-affected fraction including spatial propagation
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUTPUT_DIR = 'eda/post_incident'
DATA_PATH = 'datasets/xtraffic/SAN_BERNARDINO/data.dat'
ADJ_PATH = 'datasets/xtraffic/SAN_BERNARDINO/adj_mx.pkl'
INCIDENT_PATH = 'datasets/xtraffic/SAN_BERNARDINO/incident_metadata_2023.csv'

NUM_NODES = 893
DATA_RANGE = (0, 26280)
TRAIN_VAL_TEST_RATIO = [0.6, 0.2, 0.2]
STEPS_PER_DAY = 288
RECOVERY_BUFFER = 12
INPUT_LEN = 12
OUTPUT_LEN = 12
Z_THRESHOLD = 2.0

# Time window around incident to check neighbors
# Broader than exact incident time to capture delayed propagation
CHECK_WINDOW_BEFORE = 6   # 30 min before incident start
CHECK_WINDOW_AFTER = 24   # 2 hours after incident start
NEIGHBOR_K_VALUES = [5, 10, 20, 50]  # test different K

print("=" * 70)
print("Spatial Propagation Analysis")
print("=" * 70)

# ============================================================================
# Load data
# ============================================================================
data = np.memmap(DATA_PATH, dtype='float32', mode='r').reshape(-1, NUM_NODES, 5)[:DATA_RANGE[1]]
flow = data[:, :, 0]
occ = data[:, :, 1]
speed = data[:, :, 2]
tod = data[:, :, 3]
dow = data[:, :, 4]

with open(ADJ_PATH, 'rb') as f:
    adj_mx = pickle.load(f)
if isinstance(adj_mx, list):
    adj_mx = adj_mx[-1]
np.fill_diagonal(adj_mx, 0)

total_len = DATA_RANGE[1]
test_start = int(total_len * (TRAIN_VAL_TEST_RATIO[0] + TRAIN_VAL_TEST_RATIO[1]))

incidents = pd.read_csv(INCIDENT_PATH)
incidents = incidents[incidents['original_start_slot'] < DATA_RANGE[1]].copy()
incidents['incident_end'] = incidents['original_start_slot'] + incidents['duration_slots'] + RECOVERY_BUFFER

all_incidents = incidents.copy()
test_incidents = incidents[
    (incidents['incident_end'] >= test_start) &
    (incidents['original_start_slot'] <= DATA_RANGE[1])
].copy()

print(f"  Test incidents: {len(test_incidents)}")

# ============================================================================
# Pre-compute per-node top-K neighbors
# ============================================================================
print("\n[1] Computing neighbor rankings...")
# For each node, sort neighbors by adjacency weight (descending)
neighbor_rankings = {}
for node in range(NUM_NODES):
    weights = adj_mx[node, :]
    sorted_indices = np.argsort(-weights)  # descending
    # Exclude self (already 0 on diagonal but just in case)
    sorted_indices = sorted_indices[sorted_indices != node]
    neighbor_rankings[node] = sorted_indices

max_k = max(NEIGHBOR_K_VALUES)
print(f"  Pre-computed neighbor rankings for {NUM_NODES} nodes (max K={max_k})")

# Show sample weights for a typical node
sample_node = test_incidents.iloc[0]['sensor_idx']
sample_neighbors = neighbor_rankings[sample_node][:50]
sample_weights = adj_mx[sample_node, sample_neighbors]
print(f"\n  Sample node {sample_node}: top-50 neighbor weights:")
print(f"    Top 5:  {sample_weights[:5]}")
print(f"    Top 10: {sample_weights[5:10]}")
print(f"    Top 20: {sample_weights[10:20]}")
print(f"    Top 50: {sample_weights[40:50]}")

# ============================================================================
# Build per-node incident slot index (ALL incidents, for matching exclusion)
# ============================================================================
print("\n[2] Building incident slot index...")
# For matching, we need to know when ANY incident is active near a node
# Include the incident node AND its neighbors
node_incident_slots = {}
for node_idx in range(NUM_NODES):
    node_incs = all_incidents[all_incidents['sensor_idx'] == node_idx]
    slots = set()
    for _, inc in node_incs.iterrows():
        start = int(inc['original_start_slot'])
        end = int(inc['incident_end'])
        for s in range(max(0, start - 36), min(total_len, end + 36)):
            slots.add(s)
    node_incident_slots[node_idx] = slots

# ============================================================================
# For each incident + each neighbor, check for anomalous patterns
# ============================================================================
print(f"\n[3] Analyzing spatial propagation ({len(test_incidents)} incidents)...")

results = []  # per (incident, neighbor_node) results

for inc_idx, (_, inc) in enumerate(test_incidents.iterrows()):
    inc_node = int(inc['sensor_idx'])
    inc_start = int(inc['original_start_slot'])
    inc_dur = int(inc['duration_slots'])
    inc_type = inc['incident_type']
    inc_end = int(inc['incident_end'])

    if inc_start >= DATA_RANGE[1]:
        continue

    inc_dow_val = int(round(dow[inc_start, inc_node] * 7)) % 7

    # Check window: [inc_start - CHECK_WINDOW_BEFORE, inc_start + CHECK_WINDOW_AFTER]
    check_start = max(0, inc_start - CHECK_WINDOW_BEFORE)
    check_end = min(total_len, inc_start + CHECK_WINDOW_AFTER)

    # Get top-K neighbors (use max K)
    neighbors = neighbor_rankings[inc_node][:max_k]

    for nb_rank, nb_node in enumerate(neighbors):
        nb_node = int(nb_node)

        # Find matched non-incident days for this NEIGHBOR node
        nb_slots = node_incident_slots.get(nb_node, set())
        matched_starts = []
        for day_offset in range(-90, 90):
            if day_offset == 0:
                continue
            cs = inc_start + day_offset * STEPS_PER_DAY
            if cs - CHECK_WINDOW_BEFORE < 0 or cs + CHECK_WINDOW_AFTER >= DATA_RANGE[1]:
                continue
            cd = int(round(dow[cs, nb_node] * 7)) % 7
            if cd != inc_dow_val:
                continue
            # Check no incident on neighbor at this time
            if cs in nb_slots:
                continue
            matched_starts.append(cs)

        if len(matched_starts) < 3:
            continue

        # Check each timestep in the window
        max_flow_z = 0
        max_speed_z = 0
        max_occ_z = 0
        max_any_z = 0
        first_sig_t = None

        for rel_t in range(-CHECK_WINDOW_BEFORE, CHECK_WINDOW_AFTER):
            abs_t = inc_start + rel_t
            if abs_t < 0 or abs_t >= DATA_RANGE[1]:
                continue

            # Get neighbor's actual values
            nb_f = float(flow[abs_t, nb_node])
            nb_o = float(occ[abs_t, nb_node])
            nb_s = float(speed[abs_t, nb_node])

            # Get matched values for neighbor
            match_f = [float(flow[ms + rel_t, nb_node]) for ms in matched_starts
                       if 0 <= ms + rel_t < DATA_RANGE[1]]
            match_o = [float(occ[ms + rel_t, nb_node]) for ms in matched_starts
                       if 0 <= ms + rel_t < DATA_RANGE[1]]
            match_s = [float(speed[ms + rel_t, nb_node]) for ms in matched_starts
                       if 0 <= ms + rel_t < DATA_RANGE[1]]

            if len(match_f) < 3:
                continue

            m_f, s_f = np.mean(match_f), np.std(match_f)
            m_o, s_o = np.mean(match_o), np.std(match_o)
            m_s, s_s = np.mean(match_s), np.std(match_s)

            fz = abs((nb_f - m_f) / s_f) if s_f > 0 else 0
            oz = abs((nb_o - m_o) / s_o) if s_o > 0 else 0
            sz = abs((nb_s - m_s) / s_s) if s_s > 0 else 0
            any_z = max(fz, oz, sz)

            max_flow_z = max(max_flow_z, fz)
            max_occ_z = max(max_occ_z, oz)
            max_speed_z = max(max_speed_z, sz)
            max_any_z = max(max_any_z, any_z)

            if first_sig_t is None and any_z > Z_THRESHOLD:
                first_sig_t = rel_t

        results.append({
            'inc_node': inc_node,
            'inc_type': inc_type,
            'inc_duration': inc_dur,
            'inc_start': inc_start,
            'neighbor_node': nb_node,
            'neighbor_rank': nb_rank + 1,  # 1-indexed
            'adj_weight': float(adj_mx[inc_node, nb_node]),
            'max_flow_z': max_flow_z,
            'max_occ_z': max_occ_z,
            'max_speed_z': max_speed_z,
            'max_any_z': max_any_z,
            'affected_flow': max_flow_z > Z_THRESHOLD,
            'affected_occ': max_occ_z > Z_THRESHOLD,
            'affected_speed': max_speed_z > Z_THRESHOLD,
            'affected_any': max_any_z > Z_THRESHOLD,
            'first_sig_t': first_sig_t,
        })

    if (inc_idx + 1) % 200 == 0:
        print(f"    Processed {inc_idx + 1}/{len(test_incidents)} incidents...")

print(f"  Total (incident, neighbor) pairs analyzed: {len(results)}")

df = pd.DataFrame(results)

# ============================================================================
# Analysis
# ============================================================================
print("\n[4] Spatial propagation statistics...")

# For each K value, compute statistics
print(f"\n  === Affected Neighbor Rate by K ===")
print(f"  {'K':>4} | {'Affected/Total':>16} {'Rate':>8} | {'Avg affected':>14} | {'Flow':>8} {'Occ':>8} {'Speed':>8}")
print("  " + "-" * 90)

k_stats = []
for k in NEIGHBOR_K_VALUES:
    sub = df[df['neighbor_rank'] <= k]
    n_total = len(sub)
    n_affected = sub['affected_any'].sum()
    rate = n_affected / n_total * 100 if n_total > 0 else 0

    # Average affected neighbors per incident
    per_inc = sub.groupby(['inc_node', 'inc_start'])['affected_any'].sum()
    avg_affected = per_inc.mean()

    flow_rate = sub['affected_flow'].mean() * 100
    occ_rate = sub['affected_occ'].mean() * 100
    speed_rate = sub['affected_speed'].mean() * 100

    print(f"  {k:4d} | {n_affected:7d}/{n_total:<7d} {rate:7.1f}% | {avg_affected:13.1f} | {flow_rate:7.1f}% {occ_rate:7.1f}% {speed_rate:7.1f}%")

    k_stats.append({
        'K': k, 'total': n_total, 'affected': n_affected, 'rate': rate,
        'avg_affected_per_inc': avg_affected,
        'flow_rate': flow_rate, 'occ_rate': occ_rate, 'speed_rate': speed_rate,
    })

# Propagation by neighbor rank (distance from incident)
print(f"\n  === Affected Rate by Neighbor Rank (closeness) ===")
print(f"  {'Rank':>8} | {'Affected/Total':>16} {'Rate':>8} | {'Avg Adj Weight':>16}")
print("  " + "-" * 60)
for rank_lo, rank_hi, label in [(1, 5, '1-5'), (6, 10, '6-10'), (11, 20, '11-20'), (21, 50, '21-50')]:
    sub = df[(df['neighbor_rank'] >= rank_lo) & (df['neighbor_rank'] <= rank_hi)]
    n_total = len(sub)
    n_affected = sub['affected_any'].sum()
    rate = n_affected / n_total * 100 if n_total > 0 else 0
    avg_w = sub['adj_weight'].mean()
    print(f"  {label:>8} | {n_affected:7d}/{n_total:<7d} {rate:7.1f}% | {avg_w:15.4f}")

# Propagation timing
print(f"\n  === When Do Neighbors Get Affected? ===")
affected_df = df[df['affected_any'] & df['first_sig_t'].notna()].copy()
if len(affected_df) > 0:
    print(f"  Total affected neighbor pairs: {len(affected_df)}")
    print(f"  Mean first significant t: {affected_df['first_sig_t'].mean():.1f}")
    print(f"  Median first significant t: {affected_df['first_sig_t'].median():.0f}")

    # Distribution
    bins = [(-CHECK_WINDOW_BEFORE, -1, 'before incident'),
            (0, 6, '0-30min after'),
            (7, 12, '30-60min after'),
            (13, CHECK_WINDOW_AFTER, '60min+ after')]
    for lo, hi, label in bins:
        cnt = ((affected_df['first_sig_t'] >= lo) & (affected_df['first_sig_t'] <= hi)).sum()
        print(f"    {label:>20}: {cnt:5d} ({cnt/len(affected_df)*100:.1f}%)")

# Updated incident-affected fraction
print(f"\n  === Updated Incident-Affected Fraction (K=20) ===")
k20 = df[df['neighbor_rank'] <= 20]
n_inc = test_incidents.shape[0]

# Original: only incident node
original_affected = n_inc  # 1 node per incident
# With neighbors: incident node + affected neighbors
affected_per_inc = k20.groupby(['inc_node', 'inc_start'])['affected_any'].sum()
avg_additional = affected_per_inc.mean()
print(f"  Original: 1 node per incident")
print(f"  With K=20 neighbors: 1 + {avg_additional:.1f} = {1 + avg_additional:.1f} nodes per incident on avg")

# Recalculate (sample, node) pair fraction
n_test_samples = total_len - test_start - INPUT_LEN - OUTPUT_LEN + 1
total_pairs = n_test_samples * NUM_NODES
# Rough estimate of affected pairs including propagation
avg_duration = test_incidents['duration_slots'].mean() + RECOVERY_BUFFER
original_affected_pairs = n_inc * avg_duration
propagated_pairs = n_inc * avg_additional * avg_duration
new_total = original_affected_pairs + propagated_pairs

print(f"\n  Original affected (sample,node) pairs: ~{int(original_affected_pairs):,} ({original_affected_pairs/total_pairs*100:.2f}%)")
print(f"  Additional propagated pairs: ~{int(propagated_pairs):,} ({propagated_pairs/total_pairs*100:.2f}%)")
print(f"  Total with propagation: ~{int(new_total):,} ({new_total/total_pairs*100:.2f}%)")

# By incident duration
print(f"\n  === Propagation by Incident Duration ===")
dur_bins = [(0, 2, 'short(0-10min)'), (2, 6, 'medium(10-30min)'),
            (6, 20, 'long(30-100min)'), (20, 999, 'very_long(100min+)')]
k20_per_inc = k20.groupby(['inc_node', 'inc_start', 'inc_duration']).agg(
    n_affected=('affected_any', 'sum'),
    n_checked=('affected_any', 'count'),
).reset_index()

for lo, hi, label in dur_bins:
    sub = k20_per_inc[(k20_per_inc['inc_duration'] >= lo) & (k20_per_inc['inc_duration'] < hi)]
    if len(sub) > 0:
        print(f"  {label:>25}: avg {sub['n_affected'].mean():.1f} affected neighbors "
              f"(out of 20), {len(sub)} incidents")

# ============================================================================
# Visualizations
# ============================================================================
print("\n[5] Creating visualizations...")

fig = plt.figure(figsize=(24, 20))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
fig.suptitle('Spatial Propagation of Incidents', fontsize=16, fontweight='bold')

# Row 1: Affected rate by K / by rank / by timing
ax = fig.add_subplot(gs[0, 0])
ks = [s['K'] for s in k_stats]
rates = [s['rate'] for s in k_stats]
ax.bar(range(len(ks)), rates, color='steelblue', alpha=0.8)
ax.set_xticks(range(len(ks)))
ax.set_xticklabels([f'K={k}' for k in ks])
for i, (k, r) in enumerate(zip(ks, rates)):
    ax.text(i, r + 0.3, f'{r:.1f}%', ha='center', fontsize=10)
ax.set_ylabel('% of neighbors affected')
ax.set_title('Neighbor Affected Rate by K')
ax.grid(True, alpha=0.3, axis='y')

# By rank bin
ax = fig.add_subplot(gs[0, 1])
rank_bins = [(1, 5), (6, 10), (11, 20), (21, 50)]
rank_rates = []
rank_labels = []
for lo, hi in rank_bins:
    sub = df[(df['neighbor_rank'] >= lo) & (df['neighbor_rank'] <= hi)]
    rank_rates.append(sub['affected_any'].mean() * 100)
    rank_labels.append(f'{lo}-{hi}')
bars = ax.bar(range(len(rank_bins)), rank_rates,
              color=['#e74c3c', '#e67e22', '#f1c40f', '#95a5a6'], alpha=0.8)
ax.set_xticks(range(len(rank_bins)))
ax.set_xticklabels([f'Rank {l}' for l in rank_labels])
for i, r in enumerate(rank_rates):
    ax.text(i, r + 0.3, f'{r:.1f}%', ha='center', fontsize=10)
ax.set_ylabel('% affected')
ax.set_title('Affected Rate by Neighbor Closeness')
ax.grid(True, alpha=0.3, axis='y')

# Timing histogram
ax = fig.add_subplot(gs[0, 2])
if len(affected_df) > 0:
    ax.hist(affected_df['first_sig_t'], bins=range(-CHECK_WINDOW_BEFORE, CHECK_WINDOW_AFTER + 1),
            color='darkorange', alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Incident start')
    ax.set_xlabel('Relative time of first significant deviation')
    ax.set_ylabel('Count')
    ax.set_title('When Do Neighbors Get Affected?')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Row 2: Affected per incident distribution / by variable / by adj weight
ax = fig.add_subplot(gs[1, 0])
per_inc_k20 = k20.groupby(['inc_node', 'inc_start'])['affected_any'].sum()
ax.hist(per_inc_k20, bins=range(0, 22), color='steelblue', alpha=0.7, edgecolor='black')
ax.set_xlabel('Number of affected neighbors (out of K=20)')
ax.set_ylabel('Number of incidents')
ax.set_title(f'Affected Neighbors per Incident (K=20)\nMean={per_inc_k20.mean():.1f}')
ax.grid(True, alpha=0.3)

# By variable
ax = fig.add_subplot(gs[1, 1])
k20_sub = df[df['neighbor_rank'] <= 20]
var_rates = {
    'Flow': k20_sub['affected_flow'].mean() * 100,
    'Occupancy': k20_sub['affected_occ'].mean() * 100,
    'Speed': k20_sub['affected_speed'].mean() * 100,
    'Any': k20_sub['affected_any'].mean() * 100,
}
ax.bar(var_rates.keys(), var_rates.values(), color=['blue', 'orange', 'green', 'red'], alpha=0.7)
for i, (k, v) in enumerate(var_rates.items()):
    ax.text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=10)
ax.set_ylabel('% of K=20 neighbors affected')
ax.set_title('Which Variable Detects Propagation?')
ax.grid(True, alpha=0.3, axis='y')

# Affected rate vs adjacency weight
ax = fig.add_subplot(gs[1, 2])
# Bin by weight
weight_bins = pd.qcut(df['adj_weight'], q=10, duplicates='drop')
weight_rates = df.groupby(weight_bins)['affected_any'].mean() * 100
ax.bar(range(len(weight_rates)), weight_rates.values, color='teal', alpha=0.7)
ax.set_xticks(range(len(weight_rates)))
ax.set_xticklabels([f'{b.left:.2f}-{b.right:.2f}' for b in weight_rates.index], rotation=45, fontsize=7)
ax.set_xlabel('Adjacency Weight Range')
ax.set_ylabel('% affected')
ax.set_title('Affected Rate vs Adjacency Weight')
ax.grid(True, alpha=0.3, axis='y')

# Row 3: Duration impact / Summary / Adj weight vs max z
ax = fig.add_subplot(gs[2, 0])
dur_data = []
for lo, hi, label in dur_bins:
    sub = k20_per_inc[(k20_per_inc['inc_duration'] >= lo) & (k20_per_inc['inc_duration'] < hi)]
    if len(sub) > 0:
        dur_data.append((label.split('(')[0], sub['n_affected'].mean(), len(sub)))
if dur_data:
    labels, means, counts = zip(*dur_data)
    bars = ax.bar(range(len(labels)), means, color=['#4CAF50', '#2196F3', '#FF9800', '#f44336'], alpha=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    for i, (m, c) in enumerate(zip(means, counts)):
        ax.text(i, m + 0.1, f'{m:.1f}\n(n={c})', ha='center', fontsize=8)
    ax.set_ylabel('Avg affected neighbors (K=20)')
    ax.set_title('Propagation by Incident Duration')
    ax.grid(True, alpha=0.3, axis='y')

# Scatter: adj weight vs max z-score (sample)
ax = fig.add_subplot(gs[2, 1])
sample = df.sample(min(5000, len(df)), random_state=42)
ax.scatter(sample['adj_weight'], sample['max_any_z'], alpha=0.1, s=5, c='steelblue')
ax.axhline(y=Z_THRESHOLD, color='red', linestyle='--', label=f'z={Z_THRESHOLD}')
ax.set_xlabel('Adjacency Weight (closeness)')
ax.set_ylabel('Max |z-score| during incident')
ax.set_title('Closeness vs Impact Magnitude')
ax.set_ylim(0, 10)
ax.legend()
ax.grid(True, alpha=0.3)

# Summary text
ax = fig.add_subplot(gs[2, 2])
ax.axis('off')
summary = [
    "SPATIAL PROPAGATION SUMMARY",
    "=" * 40,
    f"Incidents analyzed: {n_inc}",
    f"Check window: [{-CHECK_WINDOW_BEFORE}, +{CHECK_WINDOW_AFTER}] slots",
    f"Z threshold: {Z_THRESHOLD}",
    "",
    "PROPAGATION EXTENT (K=20):",
    f"  Avg affected neighbors: {per_inc_k20.mean():.1f}/20",
    f"  % incidents with 0 affected: {(per_inc_k20==0).sum()}/{len(per_inc_k20)} ({(per_inc_k20==0).mean()*100:.0f}%)",
    f"  % incidents with 5+ affected: {(per_inc_k20>=5).sum()}/{len(per_inc_k20)} ({(per_inc_k20>=5).mean()*100:.0f}%)",
    "",
    "UPDATED AFFECTED FRACTION:",
    f"  Original: {original_affected_pairs/total_pairs*100:.2f}%",
    f"  With propagation: {new_total/total_pairs*100:.2f}%",
    f"  Increase: {new_total/original_affected_pairs:.1f}x",
]
ax.text(0.02, 0.98, '\n'.join(summary), transform=ax.transAxes,
        fontsize=9.5, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.savefig(f'{OUTPUT_DIR}/spatial_propagation.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: {OUTPUT_DIR}/spatial_propagation.png")

df.to_csv(f'{OUTPUT_DIR}/spatial_propagation_detail.csv', index=False)
pd.DataFrame(k_stats).to_csv(f'{OUTPUT_DIR}/spatial_propagation_k_stats.csv', index=False)
print(f"Saved: spatial_propagation_detail.csv, spatial_propagation_k_stats.csv")

print("\n" + "=" * 70)
print("SPATIAL PROPAGATION ANALYSIS COMPLETE")
print("=" * 70)
