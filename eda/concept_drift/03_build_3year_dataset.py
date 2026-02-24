#!/usr/bin/env python3
"""
Build a 3-year SAN_BERNARDINO dataset (2022-2024) for cross-year experiments.
Output: datasets/SAN_BERNARDINO_3Y/data.dat (T_total, 893, 5)
"""

import numpy as np
import pandas as pd
import pickle
import json
import os
from pathlib import Path

SENSOR_META_PATH = "/data/XTraffic/process/data/sensor_meta_feature.csv"
OUTPUT_DIR = Path("/data/pretrainingbasicts/datasets/SAN_BERNARDINO_3Y")
STEPS_PER_DAY = 288

# ──────────────────────────────────────────
# 1. Get SAN_BERNARDINO sensor indices
# ──────────────────────────────────────────
sensor_meta = pd.read_csv(SENSOR_META_PATH, sep="\t")
sb_df = sensor_meta[sensor_meta["County"] == "San Bernardino"].copy()
valid_coord = ~(sb_df["Lat"].isna() | sb_df["Lng"].isna())
sb_df = sb_df[valid_coord]
sb_indices = sb_df.index.values
n_sensors = len(sb_indices)
print(f"SAN_BERNARDINO: {n_sensors} sensors")

# ──────────────────────────────────────────
# 2. Load and concatenate 3 years
# ──────────────────────────────────────────
def add_temporal_features(data, steps_per_day=288):
    """Add time-of-day and day-of-week features."""
    l, n, c = data.shape
    tod = np.array([i % steps_per_day / steps_per_day for i in range(l)])
    tod_tiled = np.tile(tod, [1, n, 1]).transpose((2, 1, 0))
    dow = np.array([(i // steps_per_day) % 7 / 7 for i in range(l)])
    dow_tiled = np.tile(dow, [1, n, 1]).transpose((2, 1, 0))
    return np.concatenate([data, tod_tiled, dow_tiled], axis=-1)

all_monthly = []
year_boundaries = {}
cumulative_t = 0

for year in [2022, 2023, 2024]:
    year_start = cumulative_t
    for m in range(1, 13):
        if year == 2023:
            path = f"/data/XTraffic/process/data/p{m:02d}_done.npy"
            arr = np.load(path)[:, sb_indices, :]  # (T, N, 3)
        else:
            path = f"/data/XTraffic/process/data/year_{year}/year_{year}/{year}_p{m:02d}.npy"
            arr = np.load(path)[sb_indices, :, :].transpose(1, 0, 2)
        all_monthly.append(arr)
        cumulative_t += arr.shape[0]
        print(f"  {year}-{m:02d}: {arr.shape[0]} steps (cumulative: {cumulative_t})")

    year_boundaries[year] = (year_start, cumulative_t)
    print(f"  → {year}: steps [{year_start}, {cumulative_t}) = {cumulative_t - year_start} steps = {(cumulative_t - year_start)/288:.0f} days")

# Concatenate
raw_3ch = np.concatenate(all_monthly, axis=0).astype(np.float32)
np.nan_to_num(raw_3ch, copy=False, nan=0.0)
print(f"\nRaw 3-channel shape: {raw_3ch.shape}")

# Add temporal features
data_5ch = add_temporal_features(raw_3ch, steps_per_day=STEPS_PER_DAY)
print(f"5-channel shape: {data_5ch.shape}")

# ──────────────────────────────────────────
# 3. Save dataset
# ──────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

fp = np.memmap(OUTPUT_DIR / "data.dat", dtype="float32", mode="w+", shape=data_5ch.shape)
fp[:] = data_5ch[:]
fp.flush()
del fp
print(f"Saved: {OUTPUT_DIR / 'data.dat'} ({os.path.getsize(OUTPUT_DIR / 'data.dat') / 1024**3:.2f} GB)")

# Copy adjacency matrix from existing dataset
import shutil
src_adj = Path("/data/pretrainingbasicts/datasets/xtraffic/SAN_BERNARDINO/adj_mx.pkl")
if src_adj.exists():
    shutil.copy(src_adj, OUTPUT_DIR / "adj_mx.pkl")
    print(f"Copied adj_mx.pkl")

# Save desc.json
total_steps = data_5ch.shape[0]
desc = {
    "name": "SAN_BERNARDINO_3Y",
    "domain": "traffic flow",
    "shape": list(data_5ch.shape),
    "num_time_steps": int(total_steps),
    "num_nodes": int(n_sensors),
    "num_features": 5,
    "feature_description": ["flow", "occupancy", "speed", "time of day", "day of week"],
    "has_graph": False,
    "frequency (minutes)": 5,
    "steps_per_day": STEPS_PER_DAY,
    "county": "San Bernardino",
    "years": [2022, 2023, 2024],
    "year_boundaries": {str(k): list(v) for k, v in year_boundaries.items()},
    "regular_settings": {
        "INPUT_LEN": 12,
        "OUTPUT_LEN": 12,
        "TRAIN_VAL_TEST_RATIO": [0.6, 0.2, 0.2],
        "NORM_EACH_CHANNEL": True,
        "RESCALE": True,
        "METRICS": ["MAE", "RMSE", "MAPE"],
        "NULL_VAL": 0.0
    }
}

with open(OUTPUT_DIR / "desc.json", "w") as f:
    json.dump(desc, f, indent=4)
print(f"Saved: {OUTPUT_DIR / 'desc.json'}")

# ──────────────────────────────────────────
# 4. Print year ranges for config files
# ──────────────────────────────────────────
print(f"\n{'='*70}")
print("YEAR BOUNDARIES (for data_range in config files)")
print(f"{'='*70}")
for year, (start, end) in year_boundaries.items():
    print(f"  {year}: data_range=({start}, {end})  [{(end-start)/STEPS_PER_DAY:.0f} days]")

# Useful splits for experiments
print(f"\nSUGGESTED EXPERIMENT CONFIGS:")
print(f"  Train 2022, Test 2024:")
print(f"    train data_range = (0, {year_boundaries[2022][1]})")
print(f"    test  data_range = ({year_boundaries[2024][0]}, {year_boundaries[2024][1]})")
print(f"  Train 2023, Test 2024:")
print(f"    train data_range = ({year_boundaries[2023][0]}, {year_boundaries[2023][1]})")
print(f"    test  data_range = ({year_boundaries[2024][0]}, {year_boundaries[2024][1]})")
print(f"  Train 2022+2023, Test 2024:")
print(f"    train data_range = (0, {year_boundaries[2023][1]})")
print(f"    test  data_range = ({year_boundaries[2024][0]}, {year_boundaries[2024][1]})")
print(f"  Train 2024 only (baseline):")
print(f"    data_range = ({year_boundaries[2024][0]}, {year_boundaries[2024][1]})")

print("\n✅ 3-year dataset built successfully!")
