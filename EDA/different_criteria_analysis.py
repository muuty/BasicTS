"""
Analyze incidents with different "significant drop" criteria
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


def analyze_with_multiple_criteria(data, incidents):
    """
    Analyze incidents with different criteria for "significant drop"
    """
    print("="*70)
    print("Analyzing with Multiple Criteria")
    print("="*70)

    flow_idx = 0
    results = []

    for _, row in incidents.iterrows():
        node_idx = int(row['sensor_idx'])
        incident_slot = int(row['incident_slot'])

        if incident_slot >= len(data) or node_idx >= data.shape[1]:
            continue
        if incident_slot < 12:
            continue

        # Flow at incident
        flow_at_incident = data[incident_slot, node_idx, flow_idx]

        # Criterion 1: Compare to 1-hour before (same day)
        flow_1h_before = data[incident_slot-12:incident_slot, node_idx, flow_idx].mean()

        # Criterion 2: Compare to same time yesterday
        if incident_slot >= 288:
            flow_yesterday = data[incident_slot - 288, node_idx, flow_idx]
        else:
            flow_yesterday = np.nan

        # Criterion 3: Compare to same time-of-day baseline (used before)
        tod = incident_slot % 288
        incident_day = incident_slot // 288
        baseline_slots = []
        for day in range(365):
            if abs(day - incident_day) <= 1:
                continue
            slot = day * 288 + tod
            if 0 <= slot < len(data):
                baseline_slots.append(slot)

        if len(baseline_slots) >= 10:
            baseline_flows = data[baseline_slots, node_idx, flow_idx]
            baseline_mean = baseline_flows.mean()
            baseline_std = baseline_flows.std()
            z_score = (flow_at_incident - baseline_mean) / (baseline_std + 1e-6)
        else:
            baseline_mean, baseline_std, z_score = np.nan, np.nan, np.nan

        results.append({
            'incident_type': row['incident_type'],
            'flow_at_incident': flow_at_incident,
            'flow_1h_before': flow_1h_before,
            'flow_yesterday': flow_yesterday,
            'baseline_mean': baseline_mean,
            'baseline_std': baseline_std,
            'z_score': z_score,

            # Different drop calculations
            'drop_vs_1h_before': (flow_1h_before - flow_at_incident) / (flow_1h_before + 1e-6),
            'drop_vs_yesterday': (flow_yesterday - flow_at_incident) / (flow_yesterday + 1e-6) if not np.isnan(flow_yesterday) else np.nan,
            'drop_vs_baseline': (baseline_mean - flow_at_incident) / (baseline_mean + 1e-6) if not np.isnan(baseline_mean) else np.nan,

            # Absolute drop
            'abs_drop_vs_1h_before': flow_1h_before - flow_at_incident,
        })

    df = pd.DataFrame(results)

    # =========================================
    # Criterion 1: Z-score based (original)
    # =========================================
    print("\n" + "="*70)
    print("Criterion 1: Z-score (incident vs same-time-of-day baseline)")
    print("="*70)

    for threshold in [-1, -1.5, -2, -2.5, -3]:
        count = (df['z_score'] < threshold).sum()
        pct = count / len(df) * 100
        print(f"  z < {threshold}: {count:,} incidents ({pct:.1f}%)")

    # =========================================
    # Criterion 2: Percentage drop vs 1h before
    # =========================================
    print("\n" + "="*70)
    print("Criterion 2: % Drop vs 1-hour before (same day)")
    print("="*70)

    for threshold in [0.1, 0.2, 0.3, 0.5, 0.7]:
        # Only count where baseline was reasonable (>20)
        mask = (df['flow_1h_before'] > 20) & (df['drop_vs_1h_before'] > threshold)
        count = mask.sum()
        pct = count / len(df) * 100
        print(f"  >{threshold*100:.0f}% drop (baseline>20): {count:,} incidents ({pct:.1f}%)")

    # =========================================
    # Criterion 3: Absolute drop vs 1h before
    # =========================================
    print("\n" + "="*70)
    print("Criterion 3: Absolute Drop vs 1-hour before")
    print("="*70)

    for threshold in [10, 20, 50, 100, 200]:
        count = (df['abs_drop_vs_1h_before'] > threshold).sum()
        pct = count / len(df) * 100
        print(f"  Drop > {threshold}: {count:,} incidents ({pct:.1f}%)")

    # =========================================
    # Criterion 4: Flow near zero at incident
    # =========================================
    print("\n" + "="*70)
    print("Criterion 4: Low/Zero flow at incident time")
    print("="*70)

    for threshold in [0, 5, 10, 20, 50]:
        count = (df['flow_at_incident'] <= threshold).sum()
        pct = count / len(df) * 100
        print(f"  Flow <= {threshold}: {count:,} incidents ({pct:.1f}%)")

    # =========================================
    # Combined: Practical "severe" definition
    # =========================================
    print("\n" + "="*70)
    print("Criterion 5: Practical 'Severe' Definition")
    print("="*70)

    # Severe = significant drop from reasonable baseline
    practical_severe = (
        (df['flow_1h_before'] > 30) &  # had decent flow before
        (df['drop_vs_1h_before'] > 0.3)  # dropped by >30%
    )
    print(f"\n  Definition: baseline > 30 AND >30% drop")
    print(f"  Count: {practical_severe.sum():,} incidents ({practical_severe.mean()*100:.1f}%)")

    if practical_severe.sum() > 0:
        severe_df = df[practical_severe]
        print(f"\n  [Severe incidents details]")
        print(f"  Average flow before: {severe_df['flow_1h_before'].mean():.1f}")
        print(f"  Average flow at incident: {severe_df['flow_at_incident'].mean():.1f}")
        print(f"  Average drop: {severe_df['drop_vs_1h_before'].mean()*100:.1f}%")
        print(f"  Zero flow cases: {(severe_df['flow_at_incident']==0).sum()}")

        print(f"\n  [By incident type]")
        for itype in severe_df['incident_type'].value_counts().head(5).index:
            subset = severe_df[severe_df['incident_type'] == itype]
            print(f"    {itype}: {len(subset)} ({len(subset)/len(severe_df)*100:.1f}%)")

    # =========================================
    # Summary Table
    # =========================================
    print("\n" + "="*70)
    print("Summary: Incidents with Flow Drop by Different Criteria")
    print("="*70)

    summary = {
        'z-score < -2': (df['z_score'] < -2).sum(),
        'z-score < -1.5': (df['z_score'] < -1.5).sum(),
        '>30% drop (baseline>30)': ((df['flow_1h_before'] > 30) & (df['drop_vs_1h_before'] > 0.3)).sum(),
        '>50% drop (baseline>30)': ((df['flow_1h_before'] > 30) & (df['drop_vs_1h_before'] > 0.5)).sum(),
        'Absolute drop > 50': (df['abs_drop_vs_1h_before'] > 50).sum(),
        'Flow at incident <= 10': (df['flow_at_incident'] <= 10).sum(),
        'Flow at incident = 0': (df['flow_at_incident'] == 0).sum(),
    }

    print(f"\n{'Criterion':<35} {'Count':>10} {'Percentage':>12}")
    print("-" * 60)
    for criterion, count in summary.items():
        pct = count / len(df) * 100
        print(f"{criterion:<35} {count:>10,} {pct:>11.1f}%")

    return df


def analyze_what_model_should_predict(data, incidents):
    """
    From prediction perspective: when is it HARD to predict?
    """
    print("\n" + "="*70)
    print("Prediction Difficulty: Input vs Output Analysis")
    print("="*70)

    flow_idx = 0
    results = []

    for _, row in incidents.iterrows():
        node_idx = int(row['sensor_idx'])
        input_start = int(row['input_start_slot'])
        output_start = int(row['output_start_slot'])
        output_end = int(row['output_end_slot'])

        if output_end >= len(data) or node_idx >= data.shape[1]:
            continue
        if input_start < 0:
            continue

        # Input: what model sees (12 steps)
        input_flow = data[input_start:input_start+12, node_idx, flow_idx]
        input_mean = input_flow.mean()
        input_last = input_flow[-1]  # most recent

        # Output: what model should predict (typically starts at incident)
        output_flow = data[output_start:output_end, node_idx, flow_idx]
        output_mean = output_flow.mean()
        output_first = output_flow[0]  # first step to predict
        output_min = output_flow.min()

        results.append({
            'incident_type': row['incident_type'],
            'input_mean': input_mean,
            'input_last': input_last,
            'output_mean': output_mean,
            'output_first': output_first,
            'output_min': output_min,

            # Prediction difficulty: input looks normal but output drops
            'input_to_output_drop': (input_last - output_first) / (input_last + 1e-6),
            'input_to_output_min_drop': (input_last - output_min) / (input_last + 1e-6),
        })

    df = pd.DataFrame(results)

    print(f"\nTotal samples: {len(df)}")

    # Hard cases: input looks normal, output drops significantly
    print("\n[Hard Cases: Input normal (>30), Output drops significantly]")

    for threshold in [0.3, 0.5, 0.7]:
        hard = (df['input_last'] > 30) & (df['input_to_output_drop'] > threshold)
        print(f"  >{threshold*100:.0f}% drop from input_last to output_first: {hard.sum():,} ({hard.mean()*100:.1f}%)")

    # These are the ACTUALLY hard cases for the model
    hard_cases = (df['input_last'] > 30) & (df['input_to_output_drop'] > 0.3)
    if hard_cases.sum() > 0:
        hard_df = df[hard_cases]
        print(f"\n[Hard cases details (input>30, >30% drop)]")
        print(f"  Count: {len(hard_df)}")
        print(f"  Input last avg: {hard_df['input_last'].mean():.1f}")
        print(f"  Output first avg: {hard_df['output_first'].mean():.1f}")
        print(f"  Avg drop: {hard_df['input_to_output_drop'].mean()*100:.1f}%")

    return df


def main():
    print("Loading data...")
    data, incidents, desc = load_data()

    # Multiple criteria analysis
    df = analyze_with_multiple_criteria(data, incidents)

    # Prediction difficulty analysis
    pred_df = analyze_what_model_should_predict(data, incidents)

    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
