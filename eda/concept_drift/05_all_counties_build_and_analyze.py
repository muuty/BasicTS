#!/usr/bin/env python3
"""
Build 3-year datasets and run concept drift analysis for ALL major counties.
Counties with 500+ sensors: LA, Orange, Santa Clara, San Bernardino, Alameda,
Sacramento, Riverside, Contra Costa, San Joaquin
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import pickle
import shutil
from pathlib import Path
from scipy import stats

# ──────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────
BASE_DIR = Path("/data/pretrainingbasicts")
OUTPUT_DIR = Path("/data/pretrainingbasicts/eda/concept_drift")
SENSOR_META_PATH = "/data/XTraffic/process/data/sensor_meta_feature.csv"
CHANNEL_NAMES = ["flow", "occupancy", "speed"]
STEPS_PER_DAY = 288

COUNTIES = [
    ("Alameda", "ALAMEDA"),
    ("Contra Costa", "CONTRA_COSTA"),
    ("Los Angeles", "LOS_ANGELES"),
    ("Orange", "ORANGE"),
    ("Riverside", "RIVERSIDE"),
    ("Sacramento", "SACRAMENTO"),
    ("San Bernardino", "SAN_BERNARDINO"),
    ("San Joaquin", "SAN_JOAQUIN"),
    ("Santa Clara", "SANTA_CLARA"),
]

# ──────────────────────────────────────────
# Load sensor metadata
# ──────────────────────────────────────────
sensor_meta = pd.read_csv(SENSOR_META_PATH, sep="\t")

def get_county_indices(county_name):
    df = sensor_meta[sensor_meta["County"] == county_name].copy()
    valid = ~(df["Lat"].isna() | df["Lng"].isna())
    df = df[valid]
    return df.index.values, len(df)

def add_temporal_features(data, steps_per_day=288):
    l, n, c = data.shape
    tod = np.array([i % steps_per_day / steps_per_day for i in range(l)])
    tod_tiled = np.tile(tod, [1, n, 1]).transpose((2, 1, 0))
    dow = np.array([(i // steps_per_day) % 7 / 7 for i in range(l)])
    dow_tiled = np.tile(dow, [1, n, 1]).transpose((2, 1, 0))
    return np.concatenate([data, tod_tiled, dow_tiled], axis=-1)

def load_year_county(year, indices):
    """Load one year of data for given sensor indices. Returns (T, N, 3)."""
    monthly = []
    for m in range(1, 13):
        if year == 2023:
            path = f"/data/XTraffic/process/data/p{m:02d}_done.npy"
            arr = np.load(path)[:, indices, :]
        else:
            path = f"/data/XTraffic/process/data/year_{year}/year_{year}/{year}_p{m:02d}.npy"
            arr = np.load(path)[indices, :, :].transpose(1, 0, 2)
        monthly.append(arr)
    data = np.concatenate(monthly, axis=0).astype(np.float32)
    np.nan_to_num(data, copy=False, nan=0.0)
    return data

# ──────────────────────────────────────────
# Process each county
# ──────────────────────────────────────────
all_county_results = {}

for county_name, folder_name in COUNTIES:
    print(f"\n{'='*70}")
    print(f"PROCESSING: {county_name} ({folder_name})")
    print(f"{'='*70}")

    indices, n_sensors = get_county_indices(county_name)
    print(f"  Sensors: {n_sensors}")

    # ── Load 3 years ──
    years_data = {}
    year_boundaries = {}
    cumulative = 0
    for year in [2022, 2023, 2024]:
        print(f"  Loading {year}...", end=" ", flush=True)
        data = load_year_county(year, indices)
        years_data[year] = data
        year_boundaries[year] = (cumulative, cumulative + data.shape[0])
        cumulative += data.shape[0]
        print(f"shape={data.shape}")

    # ── Build datasets ──
    # 3-year dataset
    ds_3y_dir = BASE_DIR / "datasets" / f"{folder_name}_3Y"
    ds_3y_dir.mkdir(parents=True, exist_ok=True)

    all_3ch = np.concatenate([years_data[y] for y in [2022, 2023, 2024]], axis=0)
    all_5ch = add_temporal_features(all_3ch, STEPS_PER_DAY)

    fp = np.memmap(ds_3y_dir / "data.dat", dtype="float32", mode="w+", shape=all_5ch.shape)
    fp[:] = all_5ch[:]
    fp.flush()
    del fp

    # Copy adj if exists
    src_adj = BASE_DIR / "datasets" / folder_name / "adj_mx.pkl"
    if not src_adj.exists():
        src_adj = BASE_DIR / "datasets" / "xtraffic" / folder_name / "adj_mx.pkl"
    if src_adj.exists():
        shutil.copy(src_adj, ds_3y_dir / "adj_mx.pkl")

    # desc.json for 3Y
    desc_3y = {
        "name": f"{folder_name}_3Y",
        "domain": "traffic flow",
        "shape": list(all_5ch.shape),
        "num_time_steps": int(all_5ch.shape[0]),
        "num_nodes": int(n_sensors),
        "num_features": 5,
        "feature_description": ["flow", "occupancy", "speed", "time of day", "day of week"],
        "has_graph": False,
        "frequency (minutes)": 5,
        "steps_per_day": STEPS_PER_DAY,
        "county": county_name,
        "years": [2022, 2023, 2024],
        "year_boundaries": {str(k): list(v) for k, v in year_boundaries.items()},
        "regular_settings": {
            "INPUT_LEN": 12, "OUTPUT_LEN": 12,
            "TRAIN_VAL_TEST_RATIO": [0.6, 0.2, 0.2],
            "NORM_EACH_CHANNEL": True, "RESCALE": True,
            "METRICS": ["MAE", "RMSE", "MAPE"], "NULL_VAL": 0.0
        }
    }
    with open(ds_3y_dir / "desc.json", "w") as f:
        json.dump(desc_3y, f, indent=4)

    # Per-year datasets
    for year in [2022, 2023, 2024]:
        ds_dir = BASE_DIR / "datasets" / f"{folder_name}_{year}"
        ds_dir.mkdir(parents=True, exist_ok=True)
        start, end = year_boundaries[year]
        year_5ch = all_5ch[start:end]
        fp = np.memmap(ds_dir / "data.dat", dtype="float32", mode="w+", shape=year_5ch.shape)
        fp[:] = year_5ch[:]
        fp.flush()
        del fp
        if (ds_3y_dir / "adj_mx.pkl").exists():
            shutil.copy(ds_3y_dir / "adj_mx.pkl", ds_dir / "adj_mx.pkl")
        desc_yr = {
            "name": f"{folder_name}_{year}",
            "domain": "traffic flow",
            "shape": list(year_5ch.shape),
            "num_time_steps": int(year_5ch.shape[0]),
            "num_nodes": int(n_sensors),
            "num_features": 5,
            "feature_description": ["flow", "occupancy", "speed", "time of day", "day of week"],
            "has_graph": False,
            "frequency (minutes)": 5,
            "steps_per_day": STEPS_PER_DAY,
            "county": county_name,
            "year": year,
            "regular_settings": {
                "INPUT_LEN": 12, "OUTPUT_LEN": 12,
                "TRAIN_VAL_TEST_RATIO": [0.6, 0.2, 0.2],
                "NORM_EACH_CHANNEL": True, "RESCALE": True,
                "METRICS": ["MAE", "RMSE", "MAPE"], "NULL_VAL": 0.0
            }
        }
        with open(ds_dir / "desc.json", "w") as f:
            json.dump(desc_yr, f, indent=4)

    del all_3ch, all_5ch  # free memory

    print(f"  Datasets created: {folder_name}_2022, _2023, _2024, _3Y")

    # ──────────────────────────────────────
    # ANALYSIS
    # ──────────────────────────────────────
    result = {
        "county": county_name,
        "folder": folder_name,
        "n_sensors": n_sensors,
    }

    # ── Global stats per year per channel ──
    global_stats = {}
    for year, data in years_data.items():
        stats_yr = {}
        for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
            ch = data[:, :, ch_idx]
            nonzero = ch[ch > 0]
            stats_yr[ch_name] = {
                "mean": float(ch.mean()),
                "mean_nonzero": float(nonzero.mean()) if len(nonzero) > 0 else 0,
                "zero_rate": float((ch == 0).mean()),
            }
        global_stats[year] = stats_yr

    # YoY change
    for ch_name in CHANNEL_NAMES:
        m22 = global_stats[2022][ch_name]["mean"]
        m24 = global_stats[2024][ch_name]["mean"]
        pct = (m24 - m22) / m22 * 100 if m22 != 0 else 0
        result[f"{ch_name}_yoy_pct"] = round(pct, 2)
        result[f"{ch_name}_mean_2022"] = round(m22, 2)
        result[f"{ch_name}_mean_2024"] = round(m24, 2)

    # ── Sensor health per year ──
    health = {}
    zero_rates_by_year = {}
    for year, data in years_data.items():
        all_zero = (data[:, :, 0] == 0) & (data[:, :, 1] == 0) & (data[:, :, 2] == 0)
        zr = all_zero.mean(axis=0)
        zero_rates_by_year[year] = zr
        dead = int((zr > 0.9).sum())
        major = int(((zr > 0.5) & (zr <= 0.9)).sum())
        partial = int(((zr > 0.05) & (zr <= 0.5)).sum())
        func = int((zr <= 0.05).sum())
        health[year] = {"dead": dead, "major_fail": major, "partial": partial, "functional": func}

    result["health"] = health

    # Sensor transitions
    cats_22 = np.where(zero_rates_by_year[2022] > 0.9, "dead",
              np.where(zero_rates_by_year[2022] > 0.5, "major",
              np.where(zero_rates_by_year[2022] > 0.05, "partial", "func")))
    cats_24 = np.where(zero_rates_by_year[2024] > 0.9, "dead",
              np.where(zero_rates_by_year[2024] > 0.5, "major",
              np.where(zero_rates_by_year[2024] > 0.05, "partial", "func")))

    degraded = int(((cats_22 == "func") & (cats_24 != "func")).sum() +
                   ((cats_22 == "partial") & ((cats_24 == "major") | (cats_24 == "dead"))).sum() +
                   ((cats_22 == "major") & (cats_24 == "dead")).sum())
    improved = int(((cats_24 == "func") & (cats_22 != "func")).sum())
    result["sensors_degraded"] = degraded
    result["sensors_improved"] = improved

    # ── Node-level drift (flow) ──
    node_mean_22 = years_data[2022][:, :, 0].mean(axis=0)
    node_mean_24 = years_data[2024][:, :, 0].mean(axis=0)
    abs_drift = np.abs(node_mean_24 - node_mean_22)

    func_both = (zero_rates_by_year[2022] <= 0.05) & (zero_rates_by_year[2024] <= 0.05)
    n_func_both = int(func_both.sum())
    result["n_functional_both_years"] = n_func_both

    if n_func_both > 0:
        func_drift = abs_drift[func_both]
        base = node_mean_22[func_both]
        rel_drift = func_drift / (base + 1e-6)
        result["flow_drift_mean_abs"] = round(float(func_drift.mean()), 2)
        result["flow_drift_pct_gt10"] = round(float((rel_drift > 0.1).sum() / n_func_both * 100), 1)
        result["flow_drift_pct_gt20"] = round(float((rel_drift > 0.2).sum() / n_func_both * 100), 1)
        result["flow_drift_pct_gt50"] = round(float((rel_drift > 0.5).sum() / n_func_both * 100), 1)

    # ── KS test (subsample for speed) ──
    n_ks_test = min(200, n_func_both)  # test up to 200 functional nodes
    func_indices = np.where(func_both)[0]
    if n_ks_test > 0:
        rng = np.random.RandomState(42)
        test_nodes = rng.choice(func_indices, size=n_ks_test, replace=False)
        flow_22 = years_data[2022][:, :, 0]
        flow_24 = years_data[2024][:, :, 0]
        sig_count = 0
        ks_vals = []
        for node in test_nodes:
            s22 = rng.choice(flow_22[:, node], size=3000, replace=False)
            s24 = rng.choice(flow_24[:, node], size=3000, replace=False)
            ks_stat, pval = stats.ks_2samp(s22, s24)
            ks_vals.append(ks_stat)
            if pval < 0.001:
                sig_count += 1
        result["ks_significant_pct"] = round(sig_count / n_ks_test * 100, 1)
        result["ks_mean_stat"] = round(float(np.mean(ks_vals)), 4)

    # ── Persistence MAE per year ──
    for year, data in years_data.items():
        flow = data[:, :, 0]
        n_steps = flow.shape[0]
        test_start = int(n_steps * 0.8)
        test_data = flow[test_start:]

        maes = []
        for i in range(0, len(test_data) - 24, 12):  # stride for speed
            last_val = test_data[i + 11]
            future = test_data[i + 12: i + 24]
            mask = future != 0
            if mask.sum() > 0:
                maes.append(float((np.abs(future - last_val[None, :]) * mask).sum() / mask.sum()))
        result[f"persistence_mae_{year}"] = round(float(np.mean(maes)), 2)

    # ── Hourly profile comparison ──
    # Check if peak hours shifted
    func_all_3 = func_both.copy()
    for year in [2023]:
        zr = zero_rates_by_year.get(year, zero_rates_by_year[2022])
        func_all_3 = func_all_3 & (zr <= 0.05)

    # Recalculate with 2023 data
    all_zero_23 = (years_data[2023][:, :, 0] == 0) & (years_data[2023][:, :, 1] == 0) & (years_data[2023][:, :, 2] == 0)
    zr_23 = all_zero_23.mean(axis=0)
    func_all_3 = func_both & (zr_23 <= 0.05)
    n_func_3 = func_all_3.sum()

    if n_func_3 > 10:
        profiles = {}
        for year, data in years_data.items():
            flow = data[:, func_all_3, 0]
            n_days = flow.shape[0] // STEPS_PER_DAY
            flow_daily = flow[:n_days * STEPS_PER_DAY].reshape(n_days, STEPS_PER_DAY, -1).mean(axis=(0, 2))
            hourly = flow_daily.reshape(24, 12).mean(axis=1)
            profiles[year] = hourly

        # Weekday/weekend ratio
        first_dow = {2022: 5, 2023: 6, 2024: 0}
        for year, data in years_data.items():
            flow = data[:, func_all_3, 0]
            n_days = flow.shape[0] // STEPS_PER_DAY
            daily = flow[:n_days * STEPS_PER_DAY].reshape(n_days, STEPS_PER_DAY, -1).mean(axis=(1, 2))
            dows = np.array([(first_dow[year] + d) % 7 for d in range(n_days)])
            wd = float(daily[dows < 5].mean())
            we = float(daily[dows >= 5].mean())
            result[f"wd_we_ratio_{year}"] = round(wd / we, 3) if we > 0 else 0

    all_county_results[folder_name] = result

    # Free memory
    del years_data
    print(f"  ✅ {county_name} analysis complete")

# ──────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────
print("\n\n" + "="*120)
print("CROSS-COUNTY CONCEPT DRIFT SUMMARY")
print("="*120)

print(f"\n{'County':<18} {'Sensors':>7} {'Func(both)':>10} | {'Flow Δ%':>8} {'Occ Δ%':>8} {'Spd Δ%':>8} | {'Dead 22':>7} {'Dead 24':>7} {'Degraded':>8} | {'KS sig%':>7} {'Drift>10%':>9}")
print("-" * 120)

for folder_name in [c[1] for c in COUNTIES]:
    r = all_county_results[folder_name]
    h22 = r["health"][2022]
    h24 = r["health"][2024]
    print(f"{r['county']:<18} {r['n_sensors']:>7} {r.get('n_functional_both_years', 0):>10} | "
          f"{r.get('flow_yoy_pct', 0):>+7.1f}% {r.get('occupancy_yoy_pct', 0):>+7.1f}% {r.get('speed_yoy_pct', 0):>+7.1f}% | "
          f"{h22['dead']:>7} {h24['dead']:>7} {r.get('sensors_degraded', 0):>8} | "
          f"{r.get('ks_significant_pct', 0):>6.1f}% {r.get('flow_drift_pct_gt10', 0):>8.1f}%")

print("\n\n{'County':<18} {'Persist22':>10} {'Persist23':>10} {'Persist24':>10} | {'WD/WE 22':>9} {'WD/WE 23':>9} {'WD/WE 24':>9}")

# Persistence & weekday/weekend table
print(f"\n{'County':<18} {'Persist22':>10} {'Persist23':>10} {'Persist24':>10} | {'WD/WE 22':>9} {'WD/WE 23':>9} {'WD/WE 24':>9}")
print("-" * 90)
for folder_name in [c[1] for c in COUNTIES]:
    r = all_county_results[folder_name]
    print(f"{r['county']:<18} {r.get('persistence_mae_2022', 0):>10.2f} {r.get('persistence_mae_2023', 0):>10.2f} {r.get('persistence_mae_2024', 0):>10.2f} | "
          f"{r.get('wd_we_ratio_2022', 0):>9.3f} {r.get('wd_we_ratio_2023', 0):>9.3f} {r.get('wd_we_ratio_2024', 0):>9.3f}")

# ──────────────────────────────────────────
# Save results
# ──────────────────────────────────────────
# Convert numpy types for JSON
def convert_for_json(obj):
    if isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

with open(OUTPUT_DIR / "all_counties_drift_summary.json", "w") as f:
    json.dump(convert_for_json(all_county_results), f, indent=2)
print(f"\nSaved: {OUTPUT_DIR / 'all_counties_drift_summary.json'}")

# ──────────────────────────────────────────
# Comparison plot
# ──────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

counties_sorted = sorted(all_county_results.keys(), key=lambda x: all_county_results[x]["n_sensors"], reverse=True)
county_labels = [all_county_results[c]["county"] for c in counties_sorted]

# 1. Flow YoY change
ax = axes[0, 0]
vals = [all_county_results[c].get("flow_yoy_pct", 0) for c in counties_sorted]
colors = ["#d32f2f" if v > 2 else "#1565c0" if v < -2 else "#666666" for v in vals]
ax.barh(county_labels, vals, color=colors)
ax.set_xlabel("Flow Change 2022→2024 (%)")
ax.set_title("Global Flow Drift by County")
ax.axvline(0, color="black", linewidth=0.5)
ax.grid(True, alpha=0.3, axis="x")

# 2. KS test significance
ax = axes[0, 1]
vals = [all_county_results[c].get("ks_significant_pct", 0) for c in counties_sorted]
ax.barh(county_labels, vals, color="steelblue")
ax.set_xlabel("% Nodes with Significant Shift (KS p<0.001)")
ax.set_title("Distribution Shift Prevalence")
ax.grid(True, alpha=0.3, axis="x")

# 3. Dead sensor growth
ax = axes[1, 0]
x = np.arange(len(counties_sorted))
w = 0.25
for i, year in enumerate([2022, 2023, 2024]):
    vals = [all_county_results[c]["health"][year]["dead"] for c in counties_sorted]
    ax.barh(x + i*w, vals, height=w, label=str(year), alpha=0.8)
ax.set_yticks(x + w)
ax.set_yticklabels(county_labels)
ax.set_xlabel("Dead Sensors (>90% zero)")
ax.set_title("Sensor Health Degradation")
ax.legend()
ax.grid(True, alpha=0.3, axis="x")

# 4. Persistence MAE
ax = axes[1, 1]
for i, year in enumerate([2022, 2023, 2024]):
    vals = [all_county_results[c].get(f"persistence_mae_{year}", 0) for c in counties_sorted]
    ax.barh(x + i*w, vals, height=w, label=str(year), alpha=0.8)
ax.set_yticks(x + w)
ax.set_yticklabels(county_labels)
ax.set_xlabel("Persistence MAE (flow)")
ax.set_title("Prediction Difficulty by Year")
ax.legend()
ax.grid(True, alpha=0.3, axis="x")

plt.suptitle("Concept Drift Analysis: All Counties (2022-2024)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "all_counties_comparison.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR / 'all_counties_comparison.png'}")

print("\n✅ All counties processed!")
