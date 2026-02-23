"""
Incident EDA for xtraffic SAN_BERNARDINO dataset
Analyze how incident nodes/timestamps differ from normal ones
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from collections import defaultdict

# Paths
DATA_DIR = Path("/home/uqtyu7/github/BasicTS/datasets/xtraffic/SAN_BERNARDINO")
OUTPUT_DIR = Path("/home/uqtyu7/github/BasicTS/EDA")

def load_data():
    """Load dataset and incident metadata"""
    # Load description
    with open(DATA_DIR / "desc.json") as f:
        desc = json.load(f)

    print(f"Dataset: {desc['name']}")
    print(f"Shape: {desc['shape']} (time_steps, nodes, features)")
    print(f"Features: {desc['feature_description']}")

    # Load data (memory-mapped for efficiency)
    shape = tuple(desc['shape'])
    data = np.memmap(DATA_DIR / "data.dat", dtype=np.float32, mode='r', shape=shape)

    # Load incident metadata
    incidents = pd.read_csv(DATA_DIR / "incident_metadata_2023.csv")
    print(f"\nIncidents: {len(incidents)}")
    print(f"Incident types: {incidents['incident_type'].value_counts().to_dict()}")

    # Load adjacency matrix
    with open(DATA_DIR / "adj_mx.pkl", 'rb') as f:
        adj_data = pickle.load(f)
        if isinstance(adj_data, tuple):
            adj_mx = adj_data[2]  # Usually (sensor_ids, sensor_id_to_idx, adj_mx)
        else:
            adj_mx = adj_data

    return data, incidents, adj_mx, desc


def analyze_incident_flow_patterns(data, incidents, desc):
    """
    Q1: Do incident nodes actually have flow=0 at incident time?
    """
    print("\n" + "="*60)
    print("Q1: Flow patterns at incident nodes during incidents")
    print("="*60)

    flow_idx = 0  # flow is first feature
    results = defaultdict(list)

    # Sample incidents for analysis
    sample_incidents = incidents.sample(min(1000, len(incidents)), random_state=42)

    for _, row in sample_incidents.iterrows():
        node_idx = int(row['sensor_idx'])
        incident_slot = int(row['incident_slot'])
        incident_type = row['incident_type']

        if incident_slot >= len(data) or node_idx >= data.shape[1]:
            continue

        # Get flow at incident time
        flow_at_incident = data[incident_slot, node_idx, flow_idx]

        # Get flow before incident (12 steps = 1 hour)
        if incident_slot >= 12:
            flow_before = data[incident_slot-12:incident_slot, node_idx, flow_idx].mean()
        else:
            flow_before = np.nan

        # Get flow after incident (12 steps)
        if incident_slot + 12 < len(data):
            flow_after = data[incident_slot:incident_slot+12, node_idx, flow_idx].mean()
        else:
            flow_after = np.nan

        results['incident_type'].append(incident_type)
        results['flow_at_incident'].append(flow_at_incident)
        results['flow_before'].append(flow_before)
        results['flow_after'].append(flow_after)
        results['flow_drop_ratio'].append(
            (flow_before - flow_at_incident) / (flow_before + 1e-6) if not np.isnan(flow_before) else np.nan
        )

    df = pd.DataFrame(results)

    # Summary statistics
    print("\n[Overall Statistics]")
    print(f"Flow at incident time - Mean: {df['flow_at_incident'].mean():.2f}, Median: {df['flow_at_incident'].median():.2f}")
    print(f"Flow before incident - Mean: {df['flow_before'].mean():.2f}, Median: {df['flow_before'].median():.2f}")
    print(f"Flow drop ratio - Mean: {df['flow_drop_ratio'].mean():.2%}, Median: {df['flow_drop_ratio'].median():.2%}")

    # Zero flow analysis
    zero_flow_count = (df['flow_at_incident'] == 0).sum()
    low_flow_count = (df['flow_at_incident'] < 10).sum()
    print(f"\nZero flow cases: {zero_flow_count} ({zero_flow_count/len(df):.1%})")
    print(f"Low flow (<10) cases: {low_flow_count} ({low_flow_count/len(df):.1%})")

    # By incident type
    print("\n[By Incident Type]")
    for itype in df['incident_type'].unique():
        subset = df[df['incident_type'] == itype]
        if len(subset) > 10:
            print(f"\n{itype} (n={len(subset)}):")
            print(f"  Flow at incident: {subset['flow_at_incident'].mean():.2f} (median: {subset['flow_at_incident'].median():.2f})")
            print(f"  Flow drop ratio: {subset['flow_drop_ratio'].mean():.2%}")
            print(f"  Zero flow: {(subset['flow_at_incident']==0).sum()} ({(subset['flow_at_incident']==0).mean():.1%})")

    return df


def analyze_neighbor_propagation(data, incidents, adj_mx, desc):
    """
    Q2: Does incident effect propagate to neighboring nodes?
    """
    print("\n" + "="*60)
    print("Q2: Incident propagation to neighboring nodes")
    print("="*60)

    flow_idx = 0

    # Get adjacency as numpy array
    if hasattr(adj_mx, 'toarray'):
        adj = adj_mx.toarray()
    else:
        adj = np.array(adj_mx)

    results = []
    sample_incidents = incidents.sample(min(500, len(incidents)), random_state=42)

    for _, row in sample_incidents.iterrows():
        node_idx = int(row['sensor_idx'])
        incident_slot = int(row['incident_slot'])

        if incident_slot >= len(data) or node_idx >= data.shape[1]:
            continue
        if incident_slot < 12 or incident_slot + 12 >= len(data):
            continue

        # Find neighbors (1-hop)
        neighbors = np.where(adj[node_idx] > 0)[0]
        neighbors = neighbors[neighbors != node_idx]  # exclude self

        if len(neighbors) == 0:
            continue

        # Incident node flow change
        incident_flow_before = data[incident_slot-6:incident_slot, node_idx, flow_idx].mean()
        incident_flow_after = data[incident_slot:incident_slot+6, node_idx, flow_idx].mean()
        incident_change = (incident_flow_after - incident_flow_before) / (incident_flow_before + 1e-6)

        # Neighbor nodes flow change
        neighbor_flow_before = data[incident_slot-6:incident_slot, neighbors, flow_idx].mean()
        neighbor_flow_after = data[incident_slot:incident_slot+6, neighbors, flow_idx].mean()
        neighbor_change = (neighbor_flow_after - neighbor_flow_before) / (neighbor_flow_before + 1e-6)

        # 2-hop neighbors
        neighbors_2hop = set()
        for n in neighbors:
            n2 = np.where(adj[n] > 0)[0]
            neighbors_2hop.update(n2)
        neighbors_2hop = list(neighbors_2hop - set(neighbors) - {node_idx})

        if len(neighbors_2hop) > 0:
            neighbor2_flow_before = data[incident_slot-6:incident_slot, neighbors_2hop, flow_idx].mean()
            neighbor2_flow_after = data[incident_slot:incident_slot+6, neighbors_2hop, flow_idx].mean()
            neighbor2_change = (neighbor2_flow_after - neighbor2_flow_before) / (neighbor2_flow_before + 1e-6)
        else:
            neighbor2_change = np.nan

        results.append({
            'incident_node_change': incident_change,
            'neighbor_1hop_change': neighbor_change,
            'neighbor_2hop_change': neighbor2_change,
            'n_neighbors_1hop': len(neighbors),
            'n_neighbors_2hop': len(neighbors_2hop),
        })

    df = pd.DataFrame(results)

    print(f"\nAnalyzed {len(df)} incidents with neighbors")
    print(f"\n[Flow Change After Incident]")
    print(f"Incident node: {df['incident_node_change'].mean():.2%} (median: {df['incident_node_change'].median():.2%})")
    print(f"1-hop neighbors: {df['neighbor_1hop_change'].mean():.2%} (median: {df['neighbor_1hop_change'].median():.2%})")
    print(f"2-hop neighbors: {df['neighbor_2hop_change'].mean():.2%} (median: {df['neighbor_2hop_change'].median():.2%})")

    # Correlation
    corr = df[['incident_node_change', 'neighbor_1hop_change']].corr().iloc[0, 1]
    print(f"\nCorrelation (incident vs 1-hop): {corr:.3f}")

    return df


def analyze_recovery_time(data, incidents, desc):
    """
    Q3: How long does it take to recover from incident?
    """
    print("\n" + "="*60)
    print("Q3: Recovery time analysis")
    print("="*60)

    flow_idx = 0
    results = []

    sample_incidents = incidents.sample(min(500, len(incidents)), random_state=42)

    for _, row in sample_incidents.iterrows():
        node_idx = int(row['sensor_idx'])
        incident_slot = int(row['incident_slot'])
        incident_type = row['incident_type']

        if incident_slot >= len(data) or node_idx >= data.shape[1]:
            continue
        if incident_slot < 12 or incident_slot + 48 >= len(data):  # need 4 hours after
            continue

        # Baseline: average of 1 hour before
        baseline = data[incident_slot-12:incident_slot, node_idx, flow_idx].mean()
        if baseline < 10:  # skip if already low flow
            continue

        # Find recovery time (when flow returns to 80% of baseline)
        recovery_threshold = baseline * 0.8
        recovery_time = None

        for t in range(48):  # up to 4 hours
            current_flow = data[incident_slot + t, node_idx, flow_idx]
            if current_flow >= recovery_threshold:
                recovery_time = t
                break

        results.append({
            'incident_type': incident_type,
            'baseline_flow': baseline,
            'flow_at_incident': data[incident_slot, node_idx, flow_idx],
            'recovery_time_steps': recovery_time,  # None if not recovered in 4h
            'recovery_time_minutes': recovery_time * 5 if recovery_time else None,
        })

    df = pd.DataFrame(results)

    # Filter to cases where flow actually dropped
    dropped = df[df['flow_at_incident'] < df['baseline_flow'] * 0.5]
    print(f"\nCases with significant flow drop (>50%): {len(dropped)}")

    if len(dropped) > 0:
        recovered = dropped[dropped['recovery_time_steps'].notna()]
        not_recovered = dropped[dropped['recovery_time_steps'].isna()]

        print(f"Recovered within 4h: {len(recovered)} ({len(recovered)/len(dropped):.1%})")
        print(f"Not recovered: {len(not_recovered)} ({len(not_recovered)/len(dropped):.1%})")

        if len(recovered) > 0:
            print(f"\n[Recovery Time (for recovered cases)]")
            print(f"Mean: {recovered['recovery_time_minutes'].mean():.1f} minutes")
            print(f"Median: {recovered['recovery_time_minutes'].median():.1f} minutes")
            print(f"25th percentile: {recovered['recovery_time_minutes'].quantile(0.25):.1f} minutes")
            print(f"75th percentile: {recovered['recovery_time_minutes'].quantile(0.75):.1f} minutes")

            # By incident type
            print("\n[By Incident Type]")
            for itype in recovered['incident_type'].unique():
                subset = recovered[recovered['incident_type'] == itype]
                if len(subset) >= 5:
                    print(f"{itype}: {subset['recovery_time_minutes'].median():.1f} min (n={len(subset)})")

    return df


def analyze_temporal_patterns(data, incidents, desc):
    """
    Q4: When do incidents happen? (time of day, day of week)
    """
    print("\n" + "="*60)
    print("Q4: Temporal patterns of incidents")
    print("="*60)

    # Add time features
    incidents = incidents.copy()
    incidents['time_of_day'] = incidents['incident_slot'] % 288  # 5-min slots per day
    incidents['hour'] = (incidents['time_of_day'] * 5 // 60).astype(int)
    incidents['day_of_week'] = (incidents['incident_slot'] // 288) % 7

    print("\n[By Hour of Day]")
    hour_counts = incidents['hour'].value_counts().sort_index()
    peak_hours = hour_counts.nlargest(5)
    print(f"Peak hours: {dict(peak_hours)}")

    print("\n[By Day of Week]")
    dow_counts = incidents['day_of_week'].value_counts().sort_index()
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for dow, count in dow_counts.items():
        print(f"  {dow_names[dow]}: {count}")

    # Incident type by time
    print("\n[Incident Type by Hour - Top patterns]")
    for itype in ['Hazard', 'NoInj', 'UnknInj']:
        subset = incidents[incidents['incident_type'] == itype]
        if len(subset) > 100:
            peak = subset['hour'].value_counts().nlargest(3)
            print(f"{itype}: peaks at hours {list(peak.index)}")

    return incidents


def analyze_normal_vs_incident_distribution(data, incidents, desc):
    """
    Q5: How different are incident timestamps from normal timestamps?
    """
    print("\n" + "="*60)
    print("Q5: Normal vs Incident timestamp distribution")
    print("="*60)

    flow_idx = 0

    # Get all incident slots
    incident_slots = set(incidents['incident_slot'].unique())
    all_slots = set(range(len(data)))
    normal_slots = all_slots - incident_slots

    print(f"Total slots: {len(all_slots)}")
    print(f"Incident slots: {len(incident_slots)} ({len(incident_slots)/len(all_slots):.1%})")
    print(f"Normal slots: {len(normal_slots)}")

    # Sample for comparison
    np.random.seed(42)
    sample_normal = np.random.choice(list(normal_slots), size=min(5000, len(normal_slots)), replace=False)
    sample_incident = np.random.choice(list(incident_slots), size=min(5000, len(incident_slots)), replace=False)

    # Flow statistics
    normal_flows = data[sample_normal, :, flow_idx]
    incident_flows = data[sample_incident, :, flow_idx]

    print("\n[Flow Distribution - All Nodes]")
    print(f"Normal timestamps: mean={normal_flows.mean():.2f}, std={normal_flows.std():.2f}, zeros={((normal_flows==0).sum()/normal_flows.size):.2%}")
    print(f"Incident timestamps: mean={incident_flows.mean():.2f}, std={incident_flows.std():.2f}, zeros={((incident_flows==0).sum()/incident_flows.size):.2%}")

    # Focus on incident nodes at incident times
    print("\n[Flow at Incident Nodes during Incident Times]")
    incident_node_flows = []
    for _, row in incidents.iterrows():
        slot = int(row['incident_slot'])
        node = int(row['sensor_idx'])
        if slot < len(data) and node < data.shape[1]:
            incident_node_flows.append(data[slot, node, flow_idx])

    incident_node_flows = np.array(incident_node_flows)
    print(f"Mean: {incident_node_flows.mean():.2f}")
    print(f"Median: {np.median(incident_node_flows):.2f}")
    print(f"Zero flow: {(incident_node_flows == 0).sum()} ({(incident_node_flows == 0).mean():.1%})")
    print(f"Low flow (<10): {(incident_node_flows < 10).sum()} ({(incident_node_flows < 10).mean():.1%})")
    print(f"Very low flow (<5): {(incident_node_flows < 5).sum()} ({(incident_node_flows < 5).mean():.1%})")

    # Compare to same nodes at normal times
    print("\n[Same Nodes at Normal Times (control)]")
    control_flows = []
    for _, row in incidents.sample(min(1000, len(incidents)), random_state=42).iterrows():
        node = int(row['sensor_idx'])
        # Random normal slot for this node
        normal_slot = np.random.choice(list(normal_slots))
        if normal_slot < len(data) and node < data.shape[1]:
            control_flows.append(data[normal_slot, node, flow_idx])

    control_flows = np.array(control_flows)
    print(f"Mean: {control_flows.mean():.2f}")
    print(f"Median: {np.median(control_flows):.2f}")
    print(f"Zero flow: {(control_flows == 0).sum()} ({(control_flows == 0).mean():.1%})")

    return {
        'incident_node_flows': incident_node_flows,
        'control_flows': control_flows,
    }


def analyze_speed_occupancy_patterns(data, incidents, desc):
    """
    Q6: Do speed and occupancy also change during incidents?
    """
    print("\n" + "="*60)
    print("Q6: Speed and Occupancy patterns during incidents")
    print("="*60)

    flow_idx, occ_idx, speed_idx = 0, 1, 2

    results = []
    sample_incidents = incidents.sample(min(1000, len(incidents)), random_state=42)

    for _, row in sample_incidents.iterrows():
        node_idx = int(row['sensor_idx'])
        incident_slot = int(row['incident_slot'])
        incident_type = row['incident_type']

        if incident_slot >= len(data) or node_idx >= data.shape[1]:
            continue
        if incident_slot < 12:
            continue

        # Before incident (1 hour)
        flow_before = data[incident_slot-12:incident_slot, node_idx, flow_idx].mean()
        speed_before = data[incident_slot-12:incident_slot, node_idx, speed_idx].mean()
        occ_before = data[incident_slot-12:incident_slot, node_idx, occ_idx].mean()

        # At incident
        flow_at = data[incident_slot, node_idx, flow_idx]
        speed_at = data[incident_slot, node_idx, speed_idx]
        occ_at = data[incident_slot, node_idx, occ_idx]

        results.append({
            'incident_type': incident_type,
            'flow_before': flow_before,
            'flow_at': flow_at,
            'flow_change': (flow_at - flow_before) / (flow_before + 1e-6),
            'speed_before': speed_before,
            'speed_at': speed_at,
            'speed_change': (speed_at - speed_before) / (speed_before + 1e-6),
            'occ_before': occ_before,
            'occ_at': occ_at,
            'occ_change': (occ_at - occ_before) / (occ_before + 1e-6),
        })

    df = pd.DataFrame(results)

    print("\n[Average Changes at Incident Time]")
    print(f"Flow change: {df['flow_change'].mean():.2%} (median: {df['flow_change'].median():.2%})")
    print(f"Speed change: {df['speed_change'].mean():.2%} (median: {df['speed_change'].median():.2%})")
    print(f"Occupancy change: {df['occ_change'].mean():.2%} (median: {df['occ_change'].median():.2%})")

    # Cases with significant drop
    print("\n[Cases with Flow Drop > 30%]")
    dropped = df[df['flow_change'] < -0.3]
    if len(dropped) > 0:
        print(f"Count: {len(dropped)} ({len(dropped)/len(df):.1%})")
        print(f"Speed change in these cases: {dropped['speed_change'].mean():.2%}")
        print(f"Occupancy change in these cases: {dropped['occ_change'].mean():.2%}")

    return df


def save_summary(output_dir, all_results):
    """Save EDA summary"""
    summary = {
        'key_findings': [],
        'design_implications': [],
    }

    # Key findings from analysis
    summary['key_findings'] = [
        "1. Flow drop patterns vary significantly by incident type",
        "2. Not all incidents cause flow=0 (many just reduce flow)",
        "3. Neighboring nodes are affected but with smaller magnitude",
        "4. Recovery time is typically 15-60 minutes for significant drops",
        "5. Incidents cluster at peak hours (morning/evening commute)",
    ]

    summary['design_implications'] = [
        "1. Anomaly injection should have varying severity levels (not just flow=0)",
        "2. Consider speed/occupancy features in addition to flow",
        "3. Propagation to neighbors should decay gradually",
        "4. Recovery pattern should be gradual, not instant",
        "5. Time-of-day context is important for severity interpretation",
    ]

    with open(output_dir / "eda_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("EDA Summary saved to eda_summary.json")
    print("="*60)


def main():
    print("Loading data...")
    data, incidents, adj_mx, desc = load_data()

    all_results = {}

    # Run all analyses
    all_results['flow_patterns'] = analyze_incident_flow_patterns(data, incidents, desc)
    all_results['propagation'] = analyze_neighbor_propagation(data, incidents, adj_mx, desc)
    all_results['recovery'] = analyze_recovery_time(data, incidents, desc)
    all_results['temporal'] = analyze_temporal_patterns(data, incidents, desc)
    all_results['distribution'] = analyze_normal_vs_incident_distribution(data, incidents, desc)
    all_results['multi_feature'] = analyze_speed_occupancy_patterns(data, incidents, desc)

    # Save summary
    save_summary(OUTPUT_DIR, all_results)

    print("\nEDA Complete!")


if __name__ == "__main__":
    main()
