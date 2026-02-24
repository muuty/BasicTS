"""Build ALAMEDA Q1 datasets (2022, 2023, 2024) for cross-year experiments."""
import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path

STEPS_PER_DAY = 288
BASE_DIR = Path("/data/pretrainingbasicts")

# ========== 1. Get ALAMEDA sensor indices ==========
sensor_meta = pd.read_csv("/data/XTraffic/process/data/sensor_meta_feature.csv", sep="\t")
alameda_df = sensor_meta[sensor_meta["County"] == "Alameda"].copy()
alameda_indices = alameda_df.index.values
n_sensors = len(alameda_indices)
print(f"ALAMEDA: {n_sensors} sensors")

# ========== 2. Build Q1 data for each year ==========
for year in [2022, 2023, 2024]:
    monthly_data = []
    for m in range(1, 4):  # Jan, Feb, Mar only
        if year == 2023:
            path = f"/data/XTraffic/process/data/p{m:02d}_done.npy"
            arr = np.load(path)[:, alameda_indices, :]  # (T, N, 3)
        else:
            path = f"/data/XTraffic/process/data/year_{year}/year_{year}/{year}_p{m:02d}.npy"
            arr = np.load(path)[alameda_indices, :, :].transpose(1, 0, 2)  # (N,T,3) -> (T,N,3)
        monthly_data.append(arr)
        print(f"  {year}-{m:02d}: {arr.shape}")

    data_3ch = np.concatenate(monthly_data, axis=0)  # (T, N, 3)
    n_steps = data_3ch.shape[0]
    n_days = n_steps // STEPS_PER_DAY
    print(f"  {year} Q1: {data_3ch.shape}, {n_days} days")

    # Add temporal features (tod, dow)
    tod = np.array([i % STEPS_PER_DAY / STEPS_PER_DAY for i in range(n_steps)])
    tod_tiled = np.tile(tod[:, None, None], (1, n_sensors, 1))
    dow = np.array([(i // STEPS_PER_DAY) % 7 / 7 for i in range(n_steps)])
    dow_tiled = np.tile(dow[:, None, None], (1, n_sensors, 1))

    data_5ch = np.concatenate([data_3ch, tod_tiled, dow_tiled], axis=-1).astype(np.float32)
    print(f"  With temporal: {data_5ch.shape}")

    # Save dataset
    out_dir = BASE_DIR / "datasets" / f"ALAMEDA_{year}_Q1"
    out_dir.mkdir(parents=True, exist_ok=True)

    fp = np.memmap(out_dir / "data.dat", dtype="float32", mode="w+", shape=data_5ch.shape)
    fp[:] = data_5ch[:]
    fp.flush()
    del fp

    # Build adjacency matrix from coordinates
    lats = alameda_df["Lat"].values.astype(float)
    lngs = alameda_df["Lng"].values.astype(float)

    # Save adj_mx.pkl (copy from existing ALAMEDA if available, else build simple distance-based)
    alameda_adj_path = Path("/data/XTraffic/process/data/counties/ALAMEDA/adj_mx.pkl")
    if alameda_adj_path.exists():
        import shutil
        shutil.copy(alameda_adj_path, out_dir / "adj_mx.pkl")
        print(f"  Copied adj_mx from XTraffic")
    else:
        print(f"  WARNING: No adj_mx found for ALAMEDA")

    # Save desc.json
    desc = {
        "shape": list(data_5ch.shape),
        "num_nodes": n_sensors,
        "num_features": 5,
        "feature_description": ["flow", "occupancy", "speed", "time of day", "day of week"],
        "year": year,
        "period": "Q1 (Jan-Mar)",
        "days": n_days,
        "county": "Alameda"
    }
    with open(out_dir / "desc.json", "w") as f:
        json.dump(desc, f, indent=2)

    print(f"  Saved to {out_dir}")

print("\nDone! All ALAMEDA Q1 datasets created.")
