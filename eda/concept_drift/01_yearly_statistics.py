#!/usr/bin/env python3
"""
Concept Drift EDA - Step 1: Yearly Statistics Comparison
Compare basic statistics of SAN_BERNARDINO traffic data across 2022, 2023, 2024.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path

OUTPUT_DIR = Path("/data/pretrainingbasicts/eda/concept_drift")
SENSOR_META_PATH = "/data/XTraffic/process/data/sensor_meta_feature.csv"
CHANNEL_NAMES = ["flow", "occupancy", "speed"]
STEPS_PER_DAY = 288

# ──────────────────────────────────────────
# 1. Get SAN_BERNARDINO sensor indices
# ──────────────────────────────────────────
sensor_meta = pd.read_csv(SENSOR_META_PATH, sep="\t")
sb_mask = sensor_meta["County"] == "San Bernardino"
# Drop NaN coords (same logic as build_xtraffic_basicts.py)
sb_df = sensor_meta[sb_mask].copy()
valid_coord = ~(sb_df["Lat"].isna() | sb_df["Lng"].isna())
sb_df = sb_df[valid_coord]
sb_indices = sb_df.index.values  # indices in the global sensor array
n_sensors = len(sb_indices)
print(f"SAN_BERNARDINO: {n_sensors} sensors (after dropping NaN coords)")

# ──────────────────────────────────────────
# 2. Load 3 years of data
# ──────────────────────────────────────────
def load_year(year):
    """Load all 12 months for a year, return (T, N_sb, 3)"""
    monthly = []
    for m in range(1, 13):
        if year == 2023:
            path = f"/data/XTraffic/process/data/p{m:02d}_done.npy"
            arr = np.load(path)  # (T, N_all, C)
            monthly.append(arr[:, sb_indices, :])
        else:
            path = f"/data/XTraffic/process/data/year_{year}/year_{year}/{year}_p{m:02d}.npy"
            arr = np.load(path)  # (N_all, T, C)
            monthly.append(arr[sb_indices, :, :].transpose(1, 0, 2))  # -> (T, N_sb, C)
    return np.concatenate(monthly, axis=0).astype(np.float32)

print("Loading 2022...")
data_2022 = load_year(2022)
print(f"  Shape: {data_2022.shape}")

print("Loading 2023...")
data_2023 = load_year(2023)
print(f"  Shape: {data_2023.shape}")

print("Loading 2024...")
data_2024 = load_year(2024)
print(f"  Shape: {data_2024.shape}")

# Replace NaN with 0
for d in [data_2022, data_2023, data_2024]:
    np.nan_to_num(d, copy=False, nan=0.0)

years_data = {2022: data_2022, 2023: data_2023, 2024: data_2024}

# ──────────────────────────────────────────
# 3. Global statistics per year per channel
# ──────────────────────────────────────────
print("\n" + "="*70)
print("GLOBAL STATISTICS PER YEAR")
print("="*70)

global_stats = {}
for year, data in years_data.items():
    stats = {}
    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        ch_data = data[:, :, ch_idx]
        nonzero = ch_data[ch_data > 0]
        stats[ch_name] = {
            "mean": float(ch_data.mean()),
            "std": float(ch_data.std()),
            "median": float(np.median(ch_data)),
            "mean_nonzero": float(nonzero.mean()) if len(nonzero) > 0 else 0,
            "zero_rate": float((ch_data == 0).mean()),
            "p25": float(np.percentile(ch_data, 25)),
            "p75": float(np.percentile(ch_data, 75)),
            "p95": float(np.percentile(ch_data, 95)),
            "max": float(ch_data.max()),
        }
    global_stats[year] = stats

for ch_name in CHANNEL_NAMES:
    print(f"\n--- {ch_name.upper()} ---")
    print(f"{'Metric':<20} {'2022':>12} {'2023':>12} {'2024':>12} {'Δ(22→24)':>12}")
    for metric in ["mean", "std", "mean_nonzero", "zero_rate", "p25", "median", "p75", "p95", "max"]:
        v22 = global_stats[2022][ch_name][metric]
        v23 = global_stats[2023][ch_name][metric]
        v24 = global_stats[2024][ch_name][metric]
        delta = v24 - v22
        pct = f"({delta/v22*100:+.1f}%)" if v22 != 0 else ""
        print(f"{metric:<20} {v22:>12.4f} {v23:>12.4f} {v24:>12.4f} {delta:>+8.4f} {pct}")

# ──────────────────────────────────────────
# 4. Per-node yearly means
# ──────────────────────────────────────────
print("\n" + "="*70)
print("PER-NODE YEARLY MEAN COMPARISON")
print("="*70)

node_means = {}
for year, data in years_data.items():
    means = {}
    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        means[ch_name] = data[:, :, ch_idx].mean(axis=0)  # (N,)
    node_means[year] = means

# Compute per-node drift: |mean_2024 - mean_2022|
for ch_name in CHANNEL_NAMES:
    drift = np.abs(node_means[2024][ch_name] - node_means[2022][ch_name])
    rel_drift = drift / (node_means[2022][ch_name] + 1e-6)

    print(f"\n--- {ch_name.upper()} node-level mean drift (2022→2024) ---")
    print(f"  Absolute drift: mean={drift.mean():.4f}, std={drift.std():.4f}, max={drift.max():.4f} (node {drift.argmax()})")
    print(f"  Relative drift: mean={rel_drift.mean():.4f}, std={rel_drift.std():.4f}, max={rel_drift.max():.4f}")

    # How many nodes changed by >10%, >20%, >50%?
    for thresh in [0.05, 0.10, 0.20, 0.50]:
        n = (rel_drift > thresh).sum()
        print(f"  Nodes with >{thresh*100:.0f}% drift: {n}/{n_sensors} ({n/n_sensors*100:.1f}%)")

# ──────────────────────────────────────────
# 5. Dead sensor evolution across years
# ──────────────────────────────────────────
print("\n" + "="*70)
print("SENSOR HEALTH EVOLUTION")
print("="*70)

# 3-channel detection: all 3 channels == 0
for year, data in years_data.items():
    all_zero = (data[:, :, 0] == 0) & (data[:, :, 1] == 0) & (data[:, :, 2] == 0)
    zero_rate_per_node = all_zero.mean(axis=0)  # (N,)

    dead = (zero_rate_per_node > 0.9).sum()
    major_fail = ((zero_rate_per_node > 0.5) & (zero_rate_per_node <= 0.9)).sum()
    partial = ((zero_rate_per_node > 0.05) & (zero_rate_per_node <= 0.5)).sum()
    functional = (zero_rate_per_node <= 0.05).sum()

    print(f"\n{year}:")
    print(f"  Dead (>90% zero):       {dead:>4}")
    print(f"  Major fail (50-90%):    {major_fail:>4}")
    print(f"  Partial fail (5-50%):   {partial:>4}")
    print(f"  Functional (<5% zero):  {functional:>4}")

    if year == 2022:
        zero_rates_2022 = zero_rate_per_node
    elif year == 2024:
        zero_rates_2024 = zero_rate_per_node

# Sensor state transitions
print("\n--- Sensor State Transitions (2022→2024) ---")
cats_2022 = np.where(zero_rates_2022 > 0.9, "dead",
            np.where(zero_rates_2022 > 0.5, "major_fail",
            np.where(zero_rates_2022 > 0.05, "partial", "functional")))
cats_2024 = np.where(zero_rates_2024 > 0.9, "dead",
            np.where(zero_rates_2024 > 0.5, "major_fail",
            np.where(zero_rates_2024 > 0.05, "partial", "functional")))

transitions = {}
for c22, c24 in zip(cats_2022, cats_2024):
    key = f"{c22} → {c24}"
    transitions[key] = transitions.get(key, 0) + 1

for k, v in sorted(transitions.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")

# ──────────────────────────────────────────
# 6. Hourly profile comparison (functional nodes only)
# ──────────────────────────────────────────
print("\n" + "="*70)
print("HOURLY FLOW PROFILE (Functional nodes)")
print("="*70)

# Use nodes that are functional in all 3 years
func_all = (zero_rates_2022 <= 0.05) & (zero_rates_2024 <= 0.05)
for year, data in years_data.items():
    all_zero = (data[:, :, 0] == 0) & (data[:, :, 1] == 0) & (data[:, :, 2] == 0)
    zr = all_zero.mean(axis=0)
    func_all = func_all & (zr <= 0.05)

n_func = func_all.sum()
print(f"Nodes functional in ALL 3 years: {n_func}")

hourly_profiles = {}
for year, data in years_data.items():
    flow = data[:, func_all, 0]  # (T, N_func)
    n_days = flow.shape[0] // STEPS_PER_DAY
    # Reshape to (days, steps_per_day, N_func)
    flow_trimmed = flow[:n_days * STEPS_PER_DAY]
    flow_daily = flow_trimmed.reshape(n_days, STEPS_PER_DAY, -1)
    # Average over days and nodes -> hourly profile
    profile = flow_daily.mean(axis=(0, 2))  # (288,)
    hourly_profiles[year] = profile

# Convert 5-min to hourly for cleaner display
for year in [2022, 2023, 2024]:
    p = hourly_profiles[year]
    hourly = p.reshape(24, 12).mean(axis=1)
    print(f"\n{year} hourly avg flow:")
    for h in range(24):
        bar = "█" * int(hourly[h] / 2)
        print(f"  {h:02d}:00  {hourly[h]:>7.2f}  {bar}")

# Peak hour shift analysis
for year in [2022, 2023, 2024]:
    p = hourly_profiles[year]
    hourly = p.reshape(24, 12).mean(axis=1)
    am_peak = np.argmax(hourly[5:12]) + 5
    pm_peak = np.argmax(hourly[14:21]) + 14
    print(f"{year}: AM peak={am_peak}:00 ({hourly[am_peak]:.2f}), PM peak={pm_peak}:00 ({hourly[pm_peak]:.2f})")

# ──────────────────────────────────────────
# 7. Weekday vs Weekend ratio evolution
# ──────────────────────────────────────────
print("\n" + "="*70)
print("WEEKDAY vs WEEKEND FLOW RATIO")
print("="*70)

for year, data in years_data.items():
    flow = data[:, func_all, 0]
    n_days = flow.shape[0] // STEPS_PER_DAY
    flow_trimmed = flow[:n_days * STEPS_PER_DAY]
    daily_flow = flow_trimmed.reshape(n_days, STEPS_PER_DAY, -1).mean(axis=(1, 2))  # (n_days,)

    # day_of_week: 0=first day of year. We need actual DOW.
    # 2022-01-01 = Saturday (5), 2023-01-01 = Sunday (6), 2024-01-01 = Monday (0)
    first_dow = {2022: 5, 2023: 6, 2024: 0}  # 0=Monday
    dows = np.array([(first_dow[year] + d) % 7 for d in range(n_days)])

    weekday_flow = daily_flow[dows < 5].mean()
    weekend_flow = daily_flow[dows >= 5].mean()
    ratio = weekday_flow / weekend_flow if weekend_flow > 0 else 0

    print(f"{year}: weekday={weekday_flow:.2f}, weekend={weekend_flow:.2f}, ratio={ratio:.3f}")

# ──────────────────────────────────────────
# 8. Save plots
# ──────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Row 1: Hourly profiles per year (flow, occ, speed)
for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
    ax = axes[0, ch_idx]
    for year, data in years_data.items():
        ch = data[:, func_all, ch_idx]
        n_days = ch.shape[0] // STEPS_PER_DAY
        ch_daily = ch[:n_days*STEPS_PER_DAY].reshape(n_days, STEPS_PER_DAY, -1).mean(axis=(0, 2))
        x = np.arange(STEPS_PER_DAY) / 12  # hours
        ax.plot(x, ch_daily, label=str(year), alpha=0.8)
    ax.set_title(f"{ch_name} - Daily Profile")
    ax.set_xlabel("Hour of Day")
    ax.legend()
    ax.grid(True, alpha=0.3)

# Row 2: Per-node mean scatter (2022 vs 2024)
for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
    ax = axes[1, ch_idx]
    m22 = node_means[2022][ch_name]
    m24 = node_means[2024][ch_name]

    colors = np.where(func_all, "steelblue", "red")
    ax.scatter(m22, m24, c=colors, s=5, alpha=0.5)
    lim = max(m22.max(), m24.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, label="y=x (no drift)")
    ax.set_xlabel(f"2022 mean {ch_name}")
    ax.set_ylabel(f"2024 mean {ch_name}")
    ax.set_title(f"{ch_name} - Node Mean Drift")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle("Concept Drift EDA: SAN_BERNARDINO 2022 vs 2023 vs 2024", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "yearly_comparison.png", dpi=150, bbox_inches="tight")
print(f"\nPlot saved: {OUTPUT_DIR / 'yearly_comparison.png'}")

# ──────────────────────────────────────────
# 9. Distribution shift: flow histogram per year
# ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
    ax = axes[ch_idx]
    for year, data in years_data.items():
        ch = data[:, func_all, ch_idx].flatten()
        # Subsample for histogram
        rng = np.random.RandomState(42)
        sample = rng.choice(ch[ch > 0], size=min(1_000_000, (ch > 0).sum()), replace=False)
        ax.hist(sample, bins=100, alpha=0.4, density=True, label=str(year))
    ax.set_title(f"{ch_name} Distribution (nonzero, functional)")
    ax.set_xlabel(ch_name)
    ax.set_ylabel("Density")
    ax.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "distribution_comparison.png", dpi=150, bbox_inches="tight")
print(f"Plot saved: {OUTPUT_DIR / 'distribution_comparison.png'}")

# ──────────────────────────────────────────
# 10. Save summary JSON
# ──────────────────────────────────────────
summary = {
    "n_sensors": int(n_sensors),
    "n_functional_all_years": int(n_func),
    "global_stats": {str(k): v for k, v in global_stats.items()},
    "transitions": transitions,
}
with open(OUTPUT_DIR / "yearly_statistics_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved: {OUTPUT_DIR / 'yearly_statistics_summary.json'}")

print("\n✅ Done!")
