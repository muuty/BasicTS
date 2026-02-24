#!/usr/bin/env python3
"""
Case Study: Nodes with SUSTAINED concept drift (not one-off spikes).
Focus on nodes where the daily profile has structurally changed across years,
consistently observed over many days.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import json

OUTPUT_DIR = Path("/data/pretrainingbasicts/eda/concept_drift")
SENSOR_META_PATH = "/data/XTraffic/process/data/sensor_meta_feature.csv"
STEPS_PER_DAY = 288

# ──────────────────────────────────────────
# 1. Load SAN_BERNARDINO 3-year data
# ──────────────────────────────────────────
sensor_meta = pd.read_csv(SENSOR_META_PATH, sep="\t")
sb_df = sensor_meta[sensor_meta["County"] == "San Bernardino"].copy()
valid = ~(sb_df["Lat"].isna() | sb_df["Lng"].isna())
sb_df = sb_df[valid].reset_index(drop=True)
sb_indices = sensor_meta[sensor_meta["County"] == "San Bernardino"].index.values
sb_indices = sb_indices[valid.values]
n_sensors = len(sb_indices)

print(f"Loading 3-year data for {n_sensors} sensors...")

def load_year(year):
    monthly = []
    for m in range(1, 13):
        if year == 2023:
            arr = np.load(f"/data/XTraffic/process/data/p{m:02d}_done.npy")[:, sb_indices, :]
        else:
            arr = np.load(f"/data/XTraffic/process/data/year_{year}/year_{year}/{year}_p{m:02d}.npy")[sb_indices, :, :].transpose(1, 0, 2)
        monthly.append(arr)
    data = np.concatenate(monthly, axis=0).astype(np.float32)
    np.nan_to_num(data, copy=False, nan=0.0)
    return data

data_2022 = load_year(2022)
data_2023 = load_year(2023)
data_2024 = load_year(2024)
print(f"  2022: {data_2022.shape}, 2023: {data_2023.shape}, 2024: {data_2024.shape}")

# ──────────────────────────────────────────
# 2. Compute per-node WEEKLY mean flow
#    (weekly smoothing removes day-of-week noise)
# ──────────────────────────────────────────
def weekly_means(data):
    """Compute weekly mean flow per node. Returns (n_weeks, N)."""
    flow = data[:, :, 0]
    steps_per_week = STEPS_PER_DAY * 7
    n_weeks = flow.shape[0] // steps_per_week
    trimmed = flow[:n_weeks * steps_per_week]
    return trimmed.reshape(n_weeks, steps_per_week, -1).mean(axis=1)

wk_2022 = weekly_means(data_2022)  # (52, N)
wk_2023 = weekly_means(data_2023)
wk_2024 = weekly_means(data_2024)

# ──────────────────────────────────────────
# 3. Identify nodes with SUSTAINED drift
#    Criteria: NOT one-off spikes, but consistent shift
# ──────────────────────────────────────────
# Strategy: compare Q1(Jan-Mar) and Q3(Jul-Sep) across years
# A node has sustained drift if BOTH quarters show consistent shift

def quarter_mean(data, quarter):
    """Get mean flow per node for a calendar quarter."""
    flow = data[:, :, 0]
    if quarter == 'Q1':
        # Jan-Mar: first ~90 days = 25920 steps
        return flow[:25920].mean(axis=0)
    elif quarter == 'Q3':
        # Jul-Sep: steps ~52128 to ~78624
        return flow[52128:78624].mean(axis=0)
    elif quarter == 'Q4':
        # Oct-Dec
        return flow[78624:].mean(axis=0)

q1_22 = quarter_mean(data_2022, 'Q1')
q1_24 = quarter_mean(data_2024, 'Q1')
q3_22 = quarter_mean(data_2022, 'Q3')
q3_24 = quarter_mean(data_2024, 'Q3')

# Drift in both quarters (same direction)
drift_q1 = q1_24 - q1_22
drift_q3 = q3_24 - q3_22

# Sustained: same direction in both quarters AND magnitude > threshold
same_direction = (drift_q1 * drift_q3) > 0  # same sign
min_drift = np.minimum(np.abs(drift_q1), np.abs(drift_q3))  # minimum of both
avg_drift = (drift_q1 + drift_q3) / 2

# Filter: functional in both years
zr_22 = ((data_2022[:, :, 0] == 0) & (data_2022[:, :, 1] == 0) & (data_2022[:, :, 2] == 0)).mean(axis=0)
zr_24 = ((data_2024[:, :, 0] == 0) & (data_2024[:, :, 1] == 0) & (data_2024[:, :, 2] == 0)).mean(axis=0)
zr_23 = ((data_2023[:, :, 0] == 0) & (data_2023[:, :, 1] == 0) & (data_2023[:, :, 2] == 0)).mean(axis=0)
func_all = (zr_22 < 0.05) & (zr_23 < 0.05) & (zr_24 < 0.05)

# Sustained drift score: min(|Q1 drift|, |Q3 drift|) * same_direction * functional
sustained_score = min_drift * same_direction * func_all

# Also check weekly consistency: what fraction of weeks show the same direction?
def weekly_consistency(wk_a, wk_b, node):
    """What fraction of weeks show same-direction change?"""
    diff = wk_b[:, node] - wk_a[:, node]
    n_weeks = min(len(diff), 52)
    sign = np.sign(np.mean(diff))
    return (np.sign(diff[:n_weeks]) == sign).sum() / n_weeks

# Top candidates by sustained score
top_indices = np.argsort(sustained_score)[::-1]

print("\n" + "="*80)
print("TOP SUSTAINED DRIFT CANDIDATES")
print("="*80)

case_studies = []
seen_patterns = set()  # avoid duplicates of same pattern type

for rank, node in enumerate(top_indices[:50]):
    if not func_all[node]:
        continue
    if sustained_score[node] < 10:
        break

    consistency = weekly_consistency(wk_2022, wk_2024, node)
    if consistency < 0.7:  # at least 70% of weeks in same direction
        continue

    direction = "INCREASE" if avg_drift[node] > 0 else "DECREASE"
    mean_22 = data_2022[:, node, 0].mean()
    mean_23 = data_2023[:, node, 0].mean()
    mean_24 = data_2024[:, node, 0].mean()
    pct_change = (mean_24 - mean_22) / mean_22 * 100

    # Classify pattern type
    if mean_23 > mean_22 and mean_24 > mean_23:
        pattern = "monotonic_increase"
    elif mean_23 < mean_22 and mean_24 < mean_23:
        pattern = "monotonic_decrease"
    elif (mean_23 - mean_22) * (mean_24 - mean_23) < 0:
        # Direction changed
        if abs(mean_24 - mean_22) > abs(mean_23 - mean_22):
            pattern = "accelerating_after_reversal"
        else:
            pattern = "partial_recovery"
    else:
        pattern = "gradual_" + direction.lower()

    station_id = int(sb_df.iloc[node]["station_id"]) if node < len(sb_df) else -1

    case = {
        "node_idx": int(node),
        "station_id": station_id,
        "direction": direction,
        "pattern": pattern,
        "mean_flow_2022": round(float(mean_22), 1),
        "mean_flow_2023": round(float(mean_23), 1),
        "mean_flow_2024": round(float(mean_24), 1),
        "pct_change_22_24": round(float(pct_change), 1),
        "drift_q1": round(float(drift_q1[node]), 1),
        "drift_q3": round(float(drift_q3[node]), 1),
        "weekly_consistency": round(float(consistency), 3),
        "sustained_score": round(float(sustained_score[node]), 1),
    }
    case_studies.append(case)
    print(f"  Node {node} (station {station_id}): {direction} {pattern}")
    print(f"    Flow: {mean_22:.0f} → {mean_23:.0f} → {mean_24:.0f} ({pct_change:+.1f}%)")
    print(f"    Q1 drift: {drift_q1[node]:+.1f}, Q3 drift: {drift_q3[node]:+.1f}, consistency: {consistency:.0%}")

    if len(case_studies) >= 20:
        break

print(f"\nFound {len(case_studies)} sustained drift cases")

# ──────────────────────────────────────────
# 4. Select diverse cases for visualization
# ──────────────────────────────────────────
# Pick representative cases: biggest increase, biggest decrease,
# monotonic increase, monotonic decrease, gradual, etc.

def select_diverse_cases(cases, n=8):
    """Select diverse cases covering different patterns."""
    selected = []
    # Sort by absolute pct change
    by_increase = sorted([c for c in cases if c["direction"] == "INCREASE"],
                         key=lambda x: -x["pct_change_22_24"])
    by_decrease = sorted([c for c in cases if c["direction"] == "DECREASE"],
                         key=lambda x: x["pct_change_22_24"])

    # Top increases
    for c in by_increase[:3]:
        if c not in selected:
            selected.append(c)
    # Top decreases
    for c in by_decrease[:3]:
        if c not in selected:
            selected.append(c)
    # Fill remaining with highest consistency
    remaining = sorted([c for c in cases if c not in selected],
                       key=lambda x: -x["weekly_consistency"])
    for c in remaining:
        if len(selected) >= n:
            break
        selected.append(c)
    return selected

selected = select_diverse_cases(case_studies, n=8)
print(f"\nSelected {len(selected)} diverse cases for visualization")

# ──────────────────────────────────────────
# 5. Visualize: Daily profiles + weekly time series
# ──────────────────────────────────────────
n_cases = len(selected)
fig, axes = plt.subplots(n_cases, 3, figsize=(24, 4 * n_cases))
if n_cases == 1:
    axes = axes[None, :]

for row, case in enumerate(selected):
    node = case["node_idx"]

    # ── Col 0: Weekday daily profile (averaged over all weekdays) ──
    ax = axes[row, 0]
    first_dow = {2022: 5, 2023: 6, 2024: 0}
    for year, data, color in [(2022, data_2022, "#1565c0"), (2023, data_2023, "#ff8f00"), (2024, data_2024, "#2e7d32")]:
        flow = data[:, node, 0]
        n_days = len(flow) // STEPS_PER_DAY
        daily = flow[:n_days * STEPS_PER_DAY].reshape(n_days, STEPS_PER_DAY)
        dows = np.array([(first_dow[year] + d) % 7 for d in range(n_days)])
        weekday_daily = daily[dows < 5]  # weekdays only
        profile = weekday_daily.mean(axis=0)
        std = weekday_daily.std(axis=0)
        x = np.arange(STEPS_PER_DAY) / 12
        ax.plot(x, profile, color=color, label=str(year), linewidth=1.5)
        ax.fill_between(x, profile - std, profile + std, alpha=0.1, color=color)
    ax.set_title(f"Node {node} (stn {case['station_id']}): Weekday Profile", fontsize=10)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Flow")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Col 1: Weekend daily profile ──
    ax = axes[row, 1]
    for year, data, color in [(2022, data_2022, "#1565c0"), (2023, data_2023, "#ff8f00"), (2024, data_2024, "#2e7d32")]:
        flow = data[:, node, 0]
        n_days = len(flow) // STEPS_PER_DAY
        daily = flow[:n_days * STEPS_PER_DAY].reshape(n_days, STEPS_PER_DAY)
        dows = np.array([(first_dow[year] + d) % 7 for d in range(n_days)])
        weekend_daily = daily[dows >= 5]
        profile = weekend_daily.mean(axis=0)
        std = weekend_daily.std(axis=0)
        x = np.arange(STEPS_PER_DAY) / 12
        ax.plot(x, profile, color=color, label=str(year), linewidth=1.5)
        ax.fill_between(x, profile - std, profile + std, alpha=0.1, color=color)
    ax.set_title(f"Weekend Profile | {case['direction']} {case['pct_change_22_24']:+.0f}%", fontsize=10)
    ax.set_xlabel("Hour")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Col 2: Weekly mean time series (156 weeks) ──
    ax = axes[row, 2]
    all_weeks = np.concatenate([wk_2022[:, node], wk_2023[:, node], wk_2024[:, node]])
    n_total = len(all_weeks)
    x = np.arange(n_total)
    ax.plot(x, all_weeks, color="#333", linewidth=0.8, alpha=0.7)
    # Moving average (8-week)
    if n_total > 8:
        ma = np.convolve(all_weeks, np.ones(8)/8, mode='valid')
        ax.plot(np.arange(len(ma)) + 3.5, ma, color="#d32f2f", linewidth=2, label="8-wk MA")
    # Year boundaries
    n22 = len(wk_2022[:, node])
    n23 = len(wk_2023[:, node])
    ax.axvline(n22, color="gray", linestyle="--", linewidth=1)
    ax.axvline(n22 + n23, color="gray", linestyle="--", linewidth=1)
    ax.text(n22/2, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else all_weeks.max(), "2022", ha="center", fontsize=8, color="gray")
    ax.text(n22 + n23/2, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else all_weeks.max(), "2023", ha="center", fontsize=8, color="gray")
    ax.text(n22 + n23 + (n_total - n22 - n23)/2, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else all_weeks.max(), "2024", ha="center", fontsize=8, color="gray")
    ax.set_title(f"Weekly Mean Flow | consistency={case['weekly_consistency']:.0%}", fontsize=10)
    ax.set_xlabel("Week")
    ax.set_ylabel("Flow")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle("Concept Drift Case Studies: Sustained Multi-Day Pattern Changes\nSAN_BERNARDINO County", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "case_study_sustained_drift.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: {OUTPUT_DIR / 'case_study_sustained_drift.png'}")

# ──────────────────────────────────────────
# 6. Deep dive: characterize the nature of each change
# ──────────────────────────────────────────
print("\n" + "="*80)
print("DETAILED CASE STUDY ANALYSIS")
print("="*80)

for i, case in enumerate(selected):
    node = case["node_idx"]
    print(f"\n{'─'*80}")
    print(f"CASE {i+1}: Node {node} (Station {case['station_id']})")
    print(f"{'─'*80}")
    print(f"  Pattern: {case['pattern']}")
    print(f"  Direction: {case['direction']}")
    print(f"  Mean flow: {case['mean_flow_2022']:.0f} → {case['mean_flow_2023']:.0f} → {case['mean_flow_2024']:.0f} ({case['pct_change_22_24']:+.1f}%)")
    print(f"  Q1 drift: {case['drift_q1']:+.1f}, Q3 drift: {case['drift_q3']:+.1f}")
    print(f"  Weekly consistency: {case['weekly_consistency']:.0%}")

    # Analyze what specifically changed
    for year, data in [(2022, data_2022), (2023, data_2023), (2024, data_2024)]:
        flow = data[:, node, 0]
        occ = data[:, node, 1]
        speed = data[:, node, 2]
        n_days = len(flow) // STEPS_PER_DAY
        daily_flow = flow[:n_days * STEPS_PER_DAY].reshape(n_days, STEPS_PER_DAY)
        dows = np.array([({2022: 5, 2023: 6, 2024: 0}[year] + d) % 7 for d in range(n_days)])

        wd_mean = daily_flow[dows < 5].mean()
        we_mean = daily_flow[dows >= 5].mean()
        hourly = daily_flow.mean(axis=0).reshape(24, 12).mean(axis=1)
        am_peak_hr = np.argmax(hourly[5:12]) + 5
        pm_peak_hr = np.argmax(hourly[14:21]) + 14
        am_peak_val = hourly[am_peak_hr]
        pm_peak_val = hourly[pm_peak_hr]
        zero_rate = (flow == 0).mean()

        print(f"  {year}: mean_flow={flow.mean():.1f}, WD={wd_mean:.1f}, WE={we_mean:.1f}, "
              f"AM_peak={am_peak_hr}h({am_peak_val:.0f}), PM_peak={pm_peak_hr}h({pm_peak_val:.0f}), "
              f"speed={speed.mean():.1f}, occ={occ.mean():.4f}, zero={zero_rate:.1%}")

    # What type of change is it?
    m22 = case["mean_flow_2022"]
    m24 = case["mean_flow_2024"]

    # Check if it's a capacity change (proportional scaling) or pattern change
    n_days_22 = data_2022.shape[0] // STEPS_PER_DAY
    n_days_24 = data_2024.shape[0] // STEPS_PER_DAY
    profile_22 = data_2022[:n_days_22*STEPS_PER_DAY, node, 0].reshape(n_days_22, STEPS_PER_DAY).mean(axis=0)
    profile_24 = data_2024[:n_days_24*STEPS_PER_DAY, node, 0].reshape(n_days_24, STEPS_PER_DAY).mean(axis=0)
    # Normalize profiles to [0,1]
    p22_norm = profile_22 / (profile_22.max() + 1e-6)
    p24_norm = profile_24 / (profile_24.max() + 1e-6)
    shape_corr = np.corrcoef(p22_norm, p24_norm)[0, 1]

    if shape_corr > 0.95:
        change_type = "SCALE CHANGE (same shape, different magnitude)"
    elif shape_corr > 0.8:
        change_type = "MILD SHAPE CHANGE (mostly scaled, some pattern shift)"
    else:
        change_type = "STRUCTURAL CHANGE (different daily pattern)"

    case["shape_correlation"] = round(float(shape_corr), 3)
    case["change_type"] = change_type
    print(f"  → Profile shape correlation: {shape_corr:.3f} → {change_type}")

    # Check occupancy/speed changes too
    occ_22 = data_2022[:, node, 1].mean()
    occ_24 = data_2024[:, node, 1].mean()
    spd_22 = data_2022[:, node, 2].mean()
    spd_24 = data_2024[:, node, 2].mean()

    occ_change = (occ_24 - occ_22) / (occ_22 + 1e-6) * 100
    spd_change = (spd_24 - spd_22) / (spd_22 + 1e-6) * 100

    if abs(occ_change) > 20 or abs(spd_change) > 10:
        print(f"  → Occupancy: {occ_22:.4f}→{occ_24:.4f} ({occ_change:+.1f}%), Speed: {spd_22:.1f}→{spd_24:.1f} ({spd_change:+.1f}%)")

    # Hypothesis about cause
    if case["direction"] == "INCREASE" and occ_change > 10 and spd_change < -5:
        hypothesis = "Increased demand + congestion (more vehicles, slower)"
    elif case["direction"] == "INCREASE" and spd_change > 0:
        hypothesis = "Increased demand without congestion (capacity added or new route)"
    elif case["direction"] == "DECREASE" and occ_change < -10:
        hypothesis = "Reduced demand (route diversion or area decline)"
    elif case["direction"] == "DECREASE" and shape_corr < 0.8:
        hypothesis = "Fundamental usage pattern change (road function changed)"
    elif shape_corr < 0.8:
        hypothesis = "Pattern restructuring (e.g., new traffic signal, lane reconfiguration)"
    else:
        hypothesis = "Gradual demand shift (population/commute pattern change)"

    case["hypothesis"] = hypothesis
    print(f"  → Likely cause: {hypothesis}")

# ──────────────────────────────────────────
# 7. Save case studies
# ──────────────────────────────────────────
with open(OUTPUT_DIR / "case_studies_sustained_drift.json", "w") as f:
    json.dump(selected, f, indent=2)
print(f"\nSaved: {OUTPUT_DIR / 'case_studies_sustained_drift.json'}")

# ──────────────────────────────────────────
# 8. Summary statistics of all sustained drift nodes
# ──────────────────────────────────────────
print("\n" + "="*80)
print("SUMMARY OF ALL SUSTAINED DRIFT NODES")
print("="*80)

n_increase = sum(1 for c in case_studies if c["direction"] == "INCREASE")
n_decrease = sum(1 for c in case_studies if c["direction"] == "DECREASE")
print(f"Total sustained drift nodes: {len(case_studies)}")
print(f"  Increasing: {n_increase}")
print(f"  Decreasing: {n_decrease}")
print(f"  Mean |change|: {np.mean([abs(c['pct_change_22_24']) for c in case_studies]):.1f}%")
print(f"  Mean weekly consistency: {np.mean([c['weekly_consistency'] for c in case_studies]):.1%}")

print("\n✅ Case study analysis complete!")
