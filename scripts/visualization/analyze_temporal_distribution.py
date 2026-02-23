"""
Analyze temporal distribution (DOW/TOD) of coreset selections.

Compares how each selection method samples across day-of-week and time-of-day
bins relative to the full training set distribution.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_dataset_temporal(dataset_name="SAN_BERNARDINO"):
    """Load TOD/DOW features from dataset."""
    base = Path(f"datasets/xtraffic/{dataset_name}")
    desc = json.load(open(base / "desc.json"))
    shape = tuple(desc["shape"])
    data = np.memmap(base / "data.dat", dtype="float32", mode="r", shape=shape)

    # Features: [flow, occ, speed, tod, dow] — use node 0
    tod = data[:, 0, 3]  # normalized [0, 1)
    dow = data[:, 0, 4]  # normalized [0, 6/7]

    # Convert to integer bins
    steps_per_day = desc["steps_per_day"]  # 288
    tod_bins = np.round(tod * steps_per_day).astype(int) % steps_per_day
    dow_bins = np.round(dow * 7).astype(int) % 7

    return tod_bins, dow_bins, desc


def compute_distributions(indices, tod_bins, dow_bins):
    """Compute DOW and TOD distributions for given indices."""
    dow_hist = np.bincount(dow_bins[indices], minlength=7).astype(float)
    dow_hist /= dow_hist.sum()

    # TOD: bin into hourly (24 bins) for readability
    tod_hourly = tod_bins[indices] // 12  # 288 steps / 12 = 24 hours
    tod_hist = np.bincount(tod_hourly, minlength=24).astype(float)
    tod_hist /= tod_hist.sum()

    return dow_hist, tod_hist


def kl_divergence(p, q, eps=1e-10):
    """KL(P || Q) — how different P is from Q."""
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    return np.sum(p * np.log(p / q))


def main():
    dataset_name = "SAN_BERNARDINO"
    tod_bins, dow_bins, desc = load_dataset_temporal(dataset_name)

    # Training range: data_range=(0, 24192), 60% train
    data_range_end = 24192
    input_len, output_len = 12, 12
    total_samples = data_range_end - input_len - output_len + 1
    train_samples = int(total_samples * 0.6)

    # Full training distribution
    train_indices = list(range(train_samples))
    full_dow, full_tod = compute_distributions(train_indices, tod_bins, dow_bins)

    dow_names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

    # Load all coreset index files
    index_dir = Path(f"coreset_indices/{dataset_name}")
    index_files = sorted(index_dir.glob("*.json"))
    index_files = [f for f in index_files if f.name != "proxy_metrics.json"]

    print(f"Dataset: {dataset_name}")
    print(f"Train samples: {train_samples}")
    print(f"Index files: {len(index_files)}")
    print()

    # Full train reference
    print("=" * 80)
    print("FULL TRAINING SET (reference)")
    print("-" * 80)
    print("DOW: ", end="")
    for i, name in enumerate(dow_names):
        print(f"{name}={full_dow[i]*100:5.1f}%", end="  ")
    print()
    print("TOD (hourly): ", end="")
    for h in range(24):
        print(f"{full_tod[h]*100:4.1f}", end=" ")
    print()
    print()

    # Group by method and distance
    results = []
    for f in index_files:
        indices = json.load(open(f))
        name = f.stem
        dow_hist, tod_hist = compute_distributions(indices, tod_bins, dow_bins)
        dow_kl = kl_divergence(dow_hist, full_dow)
        tod_kl = kl_divergence(tod_hist, full_tod)
        results.append({
            "name": name,
            "n": len(indices),
            "dow_hist": dow_hist,
            "tod_hist": tod_hist,
            "dow_kl": dow_kl,
            "tod_kl": tod_kl,
        })

    # Sort by DOW KL divergence
    results.sort(key=lambda x: x["dow_kl"])

    # Print summary table
    print("=" * 80)
    print(f"{'Selection':<50s} {'N':>6s} {'DOW_KL':>8s} {'TOD_KL':>8s}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<50s} {r['n']:>6d} {r['dow_kl']:>8.4f} {r['tod_kl']:>8.4f}")

    print()
    print("=" * 80)
    print("DOW DISTRIBUTION BY SELECTION")
    print("-" * 80)
    print(f"{'Selection':<45s}", end="")
    for name in dow_names:
        print(f" {name:>6s}", end="")
    print(f" {'KL':>7s}")
    print("-" * 80)

    # Print full train as reference
    print(f"{'[FULL TRAIN]':<45s}", end="")
    for val in full_dow:
        print(f" {val*100:5.1f}%", end="")
    print(f" {'0.0000':>7s}")

    for r in results:
        print(f"{r['name']:<45s}", end="")
        for val in r["dow_hist"]:
            print(f" {val*100:5.1f}%", end="")
        print(f" {r['dow_kl']:>7.4f}")

    # Aggregate by method
    print()
    print("=" * 80)
    print("AVERAGE KL BY METHOD (lower = more uniform)")
    print("-" * 80)
    method_stats = defaultdict(list)
    for r in results:
        parts = r["name"].split("_")
        # Extract method: first 1-2 parts (k_center, k_medoids, graph_cut, random, stride, recent)
        if parts[0] in ("k", "graph"):
            method = "_".join(parts[:2])
        else:
            method = parts[0]
        method_stats[method].append((r["dow_kl"], r["tod_kl"]))

    print(f"{'Method':<20s} {'Avg DOW_KL':>12s} {'Avg TOD_KL':>12s} {'Count':>6s}")
    print("-" * 55)
    for method in sorted(method_stats.keys()):
        vals = method_stats[method]
        avg_dow = np.mean([v[0] for v in vals])
        avg_tod = np.mean([v[1] for v in vals])
        print(f"{method:<20s} {avg_dow:>12.4f} {avg_tod:>12.4f} {len(vals):>6d}")

    # Aggregate by distance type
    print()
    print("=" * 80)
    print("AVERAGE KL BY DISTANCE TYPE")
    print("-" * 80)
    dist_stats = defaultdict(list)
    for r in results:
        parts = r["name"].split("_")
        # Extract distance: after method, before ratio
        if parts[0] in ("k", "graph"):
            rest = parts[2:]
        else:
            rest = parts[1:]
        # rest = [dist_parts..., ratio, seedX]
        # ratio is like "030", "070", "100"
        ratio_idx = next(i for i, p in enumerate(rest) if p in ("030", "070", "100"))
        dist_type = "_".join(rest[:ratio_idx])
        dist_stats[dist_type].append((r["dow_kl"], r["tod_kl"]))

    print(f"{'Distance':<20s} {'Avg DOW_KL':>12s} {'Avg TOD_KL':>12s} {'Count':>6s}")
    print("-" * 55)
    for dist in sorted(dist_stats.keys()):
        vals = dist_stats[dist]
        avg_dow = np.mean([v[0] for v in vals])
        avg_tod = np.mean([v[1] for v in vals])
        print(f"{dist:<20s} {avg_dow:>12.4f} {avg_tod:>12.4f} {len(vals):>6d}")


if __name__ == "__main__":
    main()
