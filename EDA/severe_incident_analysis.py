"""
Focus analysis on SEVERE incidents only (z-score < -2)
These are the actual "problem" cases
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path

DATA_DIR = Path("/home/uqtyu7/github/BasicTS/datasets/xtraffic/SAN_BERNARDINO")
OUTPUT_DIR = Path("/home/uqtyu7/github/BasicTS/EDA")


def load_data():
    with open(DATA_DIR / "desc.json") as f:
        desc = json.load(f)
    shape = tuple(desc['shape'])
    data = np.memmap(DATA_DIR / "data.dat", dtype=np.float32, mode='r', shape=shape)
    incidents = pd.read_csv(DATA_DIR / "incident_metadata_2023.csv")
    return data, incidents, desc


def identify_severe_incidents(data, incidents):
    """
    Identify incidents where flow actually dropped significantly
    """
    print("="*60)
    print("Identifying SEVERE Incidents (z-score < -2)")
    print("="*60)

    flow_idx = 0
    severe_incidents = []

    for _, row in incidents.iterrows():
        node_idx = int(row['sensor_idx'])
        incident_slot = int(row['incident_slot'])

        if incident_slot >= len(data) or node_idx >= data.shape[1]:
            continue

        # Get time-of-day baseline
        tod = incident_slot % 288
        incident_day = incident_slot // 288

        baseline_slots = []
        for day in range(365):
            if abs(day - incident_day) <= 1:
                continue
            slot = day * 288 + tod
            if 0 <= slot < len(data):
                baseline_slots.append(slot)

        if len(baseline_slots) < 10:
            continue

        flow_at_incident = data[incident_slot, node_idx, flow_idx]
        baseline_flows = data[baseline_slots, node_idx, flow_idx]
        baseline_mean = baseline_flows.mean()
        baseline_std = baseline_flows.std()

        if baseline_std > 0:
            z_score = (flow_at_incident - baseline_mean) / baseline_std
        else:
            z_score = 0

        # Identify severe cases
        if z_score < -2 and baseline_mean > 20:  # significant drop AND reasonable baseline
            severe_incidents.append({
                **row.to_dict(),
                'flow_at_incident': flow_at_incident,
                'baseline_mean': baseline_mean,
                'baseline_std': baseline_std,
                'z_score': z_score,
                'drop_ratio': (baseline_mean - flow_at_incident) / baseline_mean,
            })

    severe_df = pd.DataFrame(severe_incidents)
    print(f"\nTotal severe incidents: {len(severe_df)}")

    return severe_df


def analyze_severe_patterns(data, severe_df):
    """
    Detailed analysis of severe incidents
    """
    print("\n" + "="*60)
    print("Severe Incident Patterns")
    print("="*60)

    if len(severe_df) == 0:
        print("No severe incidents found!")
        return

    print(f"\n[Basic Statistics]")
    print(f"Count: {len(severe_df)}")
    print(f"Average z-score: {severe_df['z_score'].mean():.2f}")
    print(f"Average drop ratio: {severe_df['drop_ratio'].mean():.1%}")
    print(f"Zero flow cases: {(severe_df['flow_at_incident']==0).sum()}")

    print(f"\n[By Incident Type]")
    type_counts = severe_df['incident_type'].value_counts()
    for itype, count in type_counts.items():
        pct = count / len(severe_df) * 100
        print(f"  {itype}: {count} ({pct:.1f}%)")

    print(f"\n[Temporal Distribution]")
    severe_df['hour'] = ((severe_df['incident_slot'] % 288) * 5 // 60).astype(int)
    hour_counts = severe_df['hour'].value_counts().sort_index()
    peak_hours = hour_counts.nlargest(5)
    print(f"Peak hours: {dict(peak_hours)}")

    print(f"\n[Node Distribution]")
    node_counts = severe_df['sensor_idx'].value_counts()
    print(f"Unique nodes with severe incidents: {len(node_counts)}")
    print(f"Top 5 nodes: {dict(node_counts.head())}")

    # Flow trajectory for severe cases
    print(f"\n[Flow Trajectory for Severe Incidents]")
    flow_idx = 0
    trajectories = []

    for _, row in severe_df.iterrows():
        node_idx = int(row['sensor_idx'])
        incident_slot = int(row['incident_slot'])

        if incident_slot < 12 or incident_slot + 12 >= len(data):
            continue

        traj = data[incident_slot-12:incident_slot+12, node_idx, flow_idx]
        trajectories.append(traj)

    if len(trajectories) > 0:
        trajectories = np.array(trajectories)
        avg_traj = trajectories.mean(axis=0)

        time_points = [-12, -6, -3, 0, 3, 6, 12]
        print("Time (steps from incident) | Avg Flow")
        for t in time_points:
            idx = 12 + t
            if idx < len(avg_traj):
                print(f"  t={t:+3d} ({t*5:+4d} min): {avg_traj[idx]:.2f}")

    return severe_df


def analyze_train_test_split(severe_df, total_slots=105120):
    """
    How many severe incidents are in train/val/test?
    """
    print("\n" + "="*60)
    print("Train/Val/Test Distribution of Severe Incidents")
    print("="*60)

    # 60/20/20 split
    train_end = int(total_slots * 0.6)
    val_end = int(total_slots * 0.8)

    train_severe = severe_df[severe_df['incident_slot'] < train_end]
    val_severe = severe_df[(severe_df['incident_slot'] >= train_end) & (severe_df['incident_slot'] < val_end)]
    test_severe = severe_df[severe_df['incident_slot'] >= val_end]

    print(f"Train: {len(train_severe)} severe incidents")
    print(f"Val: {len(val_severe)} severe incidents")
    print(f"Test: {len(test_severe)} severe incidents")

    return train_severe, val_severe, test_severe


def compare_severe_vs_mild(data, incidents, severe_df):
    """
    Compare severe incidents to mild incidents (for contrastive learning insight)
    """
    print("\n" + "="*60)
    print("Severe vs Mild Incidents Comparison")
    print("="*60)

    severe_nodes = set(severe_df['sensor_idx'].unique())
    severe_slots = set(severe_df['incident_slot'].unique())

    # Get mild incidents (same incident_type but no significant drop)
    mild_incidents = incidents[
        ~incidents['incident_slot'].isin(severe_slots) &
        incidents['sensor_idx'].isin(severe_nodes)  # same nodes
    ]

    print(f"Severe incidents: {len(severe_df)}")
    print(f"Mild incidents (same nodes): {len(mild_incidents)}")

    # Could use this for contrastive: severe vs mild on SAME node
    print(f"\n[Potential for Contrastive Learning]")
    print(f"Same node, different severity → natural positive/negative pairs")
    print(f"Nodes with both severe and mild: {len(severe_nodes)}")

    return mild_incidents


def main():
    print("Loading data...")
    data, incidents, desc = load_data()

    # Identify severe incidents
    severe_df = identify_severe_incidents(data, incidents)

    if len(severe_df) > 0:
        # Detailed analysis
        analyze_severe_patterns(data, severe_df)

        # Train/test split
        train_severe, val_severe, test_severe = analyze_train_test_split(severe_df)

        # Comparison
        mild_incidents = compare_severe_vs_mild(data, incidents, severe_df)

        # Save severe incidents for later use
        severe_df.to_csv(OUTPUT_DIR / "severe_incidents.csv", index=False)
        print(f"\nSaved severe incidents to {OUTPUT_DIR / 'severe_incidents.csv'}")

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
