"""Analyze concept drift on consistently functional nodes only.

Exclude sensor availability effects by filtering to nodes that are
functional (<5% zero rate) across ALL 3 years (2022, 2023, 2024).
This isolates true concept drift from sensor lifecycle issues.
"""
import sys
import os
import json
import numpy as np
import torch
sys.path.append("/data/pretrainingbasicts")

from baselines.STAEformer.arch import STAEformer

MODEL_PARAM = {
    "num_nodes": 893,
    "in_steps": 12,
    "out_steps": 12,
    "steps_per_day": 288,
    "input_dim": 3,
    "output_dim": 1,
    "input_embedding_dim": 24,
    "tod_embedding_dim": 24,
    "dow_embedding_dim": 24,
    "spatial_embedding_dim": 0,
    "adaptive_embedding_dim": 24,
    "feed_forward_dim": 256,
    "num_heads": 4,
    "num_layers": 1,
    "dropout": 0.1,
    "use_mixed_proj": True,
}

PATTERN_CHECKPOINTS = {
    2022: "checkpoints/ConceptDrift_PatternOnly/SAN_BERNARDINO_2022_Q1_30_12_12/0d9ffc16d4684f45f28491272cff154b/STAEformer_best_val_MAE.pt",
    2023: "checkpoints/ConceptDrift_PatternOnly/SAN_BERNARDINO_2023_Q1_30_12_12/283c0f33e9f29c3f7e261fefefdd2098/STAEformer_best_val_MAE.pt",
    2024: "checkpoints/ConceptDrift_PatternOnly/SAN_BERNARDINO_2024_Q1_30_12_12/a605ff3555a04c278c5ead83189fc15b/STAEformer_best_val_MAE.pt",
}

BASELINE_CHECKPOINTS = {
    2022: "checkpoints/ConceptDrift_Q1/SAN_BERNARDINO_2022_Q1_30_12_12/6d33ff60f58f5fa9e14b8d42bfdda7a8/STAEformer_best_val_MAE.pt",
    2023: "checkpoints/ConceptDrift_Q1/SAN_BERNARDINO_2023_Q1_30_12_12/0de30023c9399c9d238c0c7bbbfba3d6/STAEformer_best_val_MAE.pt",
    2024: "checkpoints/ConceptDrift_Q1/SAN_BERNARDINO_2024_Q1_30_12_12/a00dabf5d051255d7be14e2acb8c7945/STAEformer_best_val_MAE.pt",
}

DATASETS = {
    2022: "datasets/SAN_BERNARDINO_2022_Q1",
    2023: "datasets/SAN_BERNARDINO_2023_Q1",
    2024: "datasets/SAN_BERNARDINO_2024_Q1",
}

INPUT_LEN = 12
OUTPUT_LEN = 12
TRAIN_RATIO = 0.6
DEVICE = "cuda:1"


def load_model(ckpt_path):
    model = STAEformer(**MODEL_PARAM)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE).eval()
    return model


def load_year_data(dataset_dir):
    desc = json.load(open(os.path.join(dataset_dir, "desc.json")))
    shape = desc["shape"]
    data = np.memmap(os.path.join(dataset_dir, "data.dat"), dtype="float32", mode="r").reshape(shape)
    n_total = shape[0]
    n_train = int(n_total * TRAIN_RATIO)
    train_data = data[:n_train]
    n_val = int(n_total * 0.2)
    test_data = data[n_train + n_val:]

    # Global scaler (channel 0)
    mean = float(np.mean(train_data[:, :, 0]))
    std = float(np.std(train_data[:, :, 0]))

    # Per-node statistics (all data, for drift analysis)
    all_ch0 = data[:, :, 0]
    train_ch0 = train_data[:, :, 0]
    test_ch0 = test_data[:, :, 0]

    # 3-channel zero rate on full data
    all_zero = (data[:, :, 0] == 0) & (data[:, :, 1] == 0) & (data[:, :, 2] == 0)
    zero_rate = np.mean(all_zero, axis=0)

    # Per-node flow stats
    per_node_stats = {
        "train_mean": np.mean(train_ch0, axis=0),
        "train_std": np.std(train_ch0, axis=0),
        "train_median": np.median(train_ch0, axis=0),
        "test_mean": np.mean(test_ch0, axis=0),
        "test_std": np.std(test_ch0, axis=0),
        "all_mean": np.mean(all_ch0, axis=0),
        "all_std": np.std(all_ch0, axis=0),
    }

    # Temporal pattern stats: average daily profile per node (train)
    steps_per_day = 288
    n_days_train = len(train_ch0) // steps_per_day
    daily_profiles = train_ch0[:n_days_train * steps_per_day].reshape(n_days_train, steps_per_day, -1)
    avg_daily_profile = np.mean(daily_profiles, axis=0)  # (288, N)

    # Test daily profile
    n_days_test = len(test_ch0) // steps_per_day
    daily_profiles_test = test_ch0[:n_days_test * steps_per_day].reshape(n_days_test, steps_per_day, -1)
    avg_daily_profile_test = np.mean(daily_profiles_test, axis=0)  # (288, N)

    return {
        "data": data,
        "train_data": train_data,
        "test_data": test_data,
        "mean": mean, "std": std,
        "zero_rate": zero_rate,
        "per_node_stats": per_node_stats,
        "avg_daily_profile_train": avg_daily_profile,
        "avg_daily_profile_test": avg_daily_profile_test,
    }


def create_samples(data, input_len, output_len):
    samples_x, samples_y = [], []
    for i in range(len(data) - input_len - output_len + 1):
        samples_x.append(data[i:i + input_len])
        samples_y.append(data[i + input_len:i + input_len + output_len, :, 0:1])
    return np.array(samples_x), np.array(samples_y)


def normalize_input(x, mean, std):
    x_norm = x.copy()
    x_norm[:, :, :, 0] = (x[:, :, :, 0] - mean) / std
    return x_norm


def predict_pattern_only(model, test_x, mean, std, batch_size=64):
    test_x_norm = normalize_input(test_x, mean, std)
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(test_x_norm), batch_size):
            bx = torch.FloatTensor(test_x_norm[i:i + batch_size]).to(DEVICE)
            input_flow_mean = bx[:, :, :, 0].mean(dim=1, keepdim=True)
            bx_centered = bx.clone()
            bx_centered[:, :, :, 0] = bx[:, :, :, 0] - input_flow_mean
            out = model(bx_centered, None, 0, 0, False)["prediction"]
            out = out + input_flow_mean.unsqueeze(-1)
            pred = out.cpu().numpy() * std + mean
            all_preds.append(pred)
    return np.concatenate(all_preds, axis=0)


def predict_baseline(model, test_x, mean, std, batch_size=64):
    test_x_norm = normalize_input(test_x, mean, std)
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(test_x_norm), batch_size):
            bx = torch.FloatTensor(test_x_norm[i:i + batch_size]).to(DEVICE)
            out = model(bx, None, 0, 0, False)["prediction"]
            pred = out.cpu().numpy() * std + mean
            all_preds.append(pred)
    return np.concatenate(all_preds, axis=0)


def per_node_mae(preds, targets):
    return np.mean(np.abs(preds - targets), axis=(0, 1, 3))


def daily_profile_similarity(profile_a, profile_b):
    """Cosine similarity between daily profiles per node."""
    # profile_a, profile_b: (288, N)
    dot = np.sum(profile_a * profile_b, axis=0)
    norm_a = np.sqrt(np.sum(profile_a ** 2, axis=0))
    norm_b = np.sqrt(np.sum(profile_b ** 2, axis=0))
    return dot / (norm_a * norm_b + 1e-8)


def normalized_daily_profile(profile):
    """Normalize daily profile per node: subtract mean, divide by std."""
    # profile: (288, N)
    mean = profile.mean(axis=0, keepdims=True)
    std = profile.std(axis=0, keepdims=True) + 1e-8
    return (profile - mean) / std


def main():
    years = [2022, 2023, 2024]

    print("=" * 70)
    print("CONCEPT DRIFT ON CONSISTENTLY FUNCTIONAL NODES")
    print("=" * 70)

    # Load all year data
    print("\nLoading data...")
    data_cache = {}
    for year in years:
        data_cache[year] = load_year_data(DATASETS[year])
        print(f"  {year}: zero_rate <5%: {(data_cache[year]['zero_rate'] <= 0.05).sum()} nodes")

    # ===================================================================
    # 1. Find consistently functional nodes
    # ===================================================================
    print("\n" + "=" * 70)
    print("1. CONSISTENTLY FUNCTIONAL NODES")
    print("=" * 70)

    func_masks = {}
    for year in years:
        func_masks[year] = data_cache[year]["zero_rate"] <= 0.05

    # Nodes functional in ALL 3 years
    stable_mask = func_masks[2022] & func_masks[2023] & func_masks[2024]
    stable_indices = np.where(stable_mask)[0]
    n_stable = stable_mask.sum()

    print(f"\n  Functional per year: 2022={func_masks[2022].sum()}, "
          f"2023={func_masks[2023].sum()}, 2024={func_masks[2024].sum()}")
    print(f"  Consistently functional (all 3 years): {n_stable}")
    print(f"  Lost due to filtering: {max(func_masks[y].sum() for y in years) - n_stable}")

    # Also identify nodes that changed status
    became_dead = func_masks[2022] & ~func_masks[2023]  # functional 2022 → dead 2023
    became_alive = ~func_masks[2022] & func_masks[2023]  # dead 2022 → functional 2023
    print(f"\n  Status changes 2022→2023:")
    print(f"    Functional→Dead: {became_dead.sum()}")
    print(f"    Dead→Functional: {became_alive.sum()}")

    became_dead_24 = func_masks[2022] & ~func_masks[2024]
    became_alive_24 = ~func_masks[2022] & func_masks[2024]
    print(f"  Status changes 2022→2024:")
    print(f"    Functional→Dead: {became_dead_24.sum()}")
    print(f"    Dead→Functional: {became_alive_24.sum()}")

    # Save stable indices
    np.save("datasets/SAN_BERNARDINO_2022_Q1/stable_functional_indices.npy", stable_indices)
    print(f"\n  Stable indices saved ({n_stable} nodes)")

    # ===================================================================
    # 2. Scale drift analysis on stable nodes
    # ===================================================================
    print("\n" + "=" * 70)
    print("2. SCALE DRIFT ON STABLE NODES")
    print("=" * 70)

    for y1, y2 in [(2022, 2023), (2022, 2024), (2023, 2024)]:
        mean1 = data_cache[y1]["per_node_stats"]["all_mean"][stable_mask]
        mean2 = data_cache[y2]["per_node_stats"]["all_mean"][stable_mask]
        std1 = data_cache[y1]["per_node_stats"]["all_std"][stable_mask]
        std2 = data_cache[y2]["per_node_stats"]["all_std"][stable_mask]

        mean_ratio = mean2 / (mean1 + 1e-6)
        std_ratio = std2 / (std1 + 1e-6)
        additive_shift = mean2 - mean1
        additive_shift_pct = additive_shift / (mean1 + 1e-6) * 100

        print(f"\n--- {y1} → {y2} (n={n_stable} stable nodes) ---")
        print(f"  Mean flow ratio (test/train):")
        print(f"    mean={np.mean(mean_ratio):.3f}, median={np.median(mean_ratio):.3f}, "
              f"std={np.std(mean_ratio):.3f}")
        print(f"    range=[{np.min(mean_ratio):.3f}, {np.max(mean_ratio):.3f}]")
        print(f"  Additive shift:")
        print(f"    mean={np.mean(additive_shift):.1f}, median={np.median(additive_shift):.1f}")
        print(f"    pct: mean={np.mean(additive_shift_pct):.1f}%, "
              f"median={np.median(additive_shift_pct):.1f}%")
        print(f"  Std ratio:")
        print(f"    mean={np.mean(std_ratio):.3f}, median={np.median(std_ratio):.3f}")

        # Distribution of scale ratio
        r = mean_ratio
        print(f"  Scale ratio distribution:")
        print(f"    <0.8 (shrunk >20%):  {(r < 0.8).sum():4d} ({(r < 0.8).sum()/n_stable*100:5.1f}%)")
        print(f"    0.8-0.9:             {((r >= 0.8) & (r < 0.9)).sum():4d}")
        print(f"    0.9-1.1 (stable):    {((r >= 0.9) & (r < 1.1)).sum():4d} "
              f"({((r >= 0.9) & (r < 1.1)).sum()/n_stable*100:5.1f}%)")
        print(f"    1.1-1.2:             {((r >= 1.1) & (r < 1.2)).sum():4d}")
        print(f"    >1.2 (grew >20%):    {(r >= 1.2).sum():4d} ({(r >= 1.2).sum()/n_stable*100:5.1f}%)")

    # ===================================================================
    # 3. Temporal pattern drift on stable nodes
    # ===================================================================
    print("\n" + "=" * 70)
    print("3. TEMPORAL PATTERN DRIFT (daily profile similarity)")
    print("=" * 70)

    for y1, y2 in [(2022, 2023), (2022, 2024), (2023, 2024)]:
        prof1 = data_cache[y1]["avg_daily_profile_train"][:, stable_mask]  # (288, n_stable)
        prof2 = data_cache[y2]["avg_daily_profile_train"][:, stable_mask]

        # Raw cosine similarity
        raw_sim = daily_profile_similarity(prof1, prof2)

        # Normalized cosine similarity (shape-only, no scale)
        norm_prof1 = normalized_daily_profile(prof1)
        norm_prof2 = normalized_daily_profile(prof2)
        shape_sim = daily_profile_similarity(norm_prof1, norm_prof2)

        print(f"\n--- {y1} → {y2} ---")
        print(f"  Raw profile cosine similarity:")
        print(f"    mean={np.mean(raw_sim):.4f}, median={np.median(raw_sim):.4f}, "
              f"min={np.min(raw_sim):.4f}")
        print(f"  Shape-only (normalized) cosine similarity:")
        print(f"    mean={np.mean(shape_sim):.4f}, median={np.median(shape_sim):.4f}, "
              f"min={np.min(shape_sim):.4f}")

        # Distribution
        s = shape_sim
        print(f"  Shape similarity distribution:")
        print(f"    <0.90 (pattern changed): {(s < 0.90).sum():4d} ({(s < 0.90).sum()/n_stable*100:5.1f}%)")
        print(f"    0.90-0.95:               {((s >= 0.90) & (s < 0.95)).sum():4d}")
        print(f"    0.95-0.99:               {((s >= 0.95) & (s < 0.99)).sum():4d}")
        print(f"    >0.99 (pattern stable):  {(s >= 0.99).sum():4d} ({(s >= 0.99).sum()/n_stable*100:5.1f}%)")

    # ===================================================================
    # 4. Cross-year evaluation on STABLE nodes only
    # ===================================================================
    print("\n" + "=" * 70)
    print("4. CROSS-YEAR MAE (stable functional nodes only)")
    print("=" * 70)

    # Prepare test samples for each year
    test_samples = {}
    for year in years:
        test_x, test_y = create_samples(data_cache[year]["test_data"], INPUT_LEN, OUTPUT_LEN)
        test_samples[year] = {"x": test_x, "y": test_y}

    results = {}

    # Pattern-only models
    print("\n--- Pattern-Only Models ---")
    for train_year in years:
        model = load_model(PATTERN_CHECKPOINTS[train_year])
        train_info = data_cache[train_year]

        for test_year in years:
            test_info = data_cache[test_year]
            ts = test_samples[test_year]

            preds = predict_pattern_only(model, ts["x"], train_info["mean"], train_info["std"])
            node_mae = per_node_mae(preds, ts["y"])

            # Stable nodes only
            stable_mae = float(np.mean(node_mae[stable_mask]))
            all_mae = float(np.mean(node_mae))

            key = f"pattern_only_{train_year}_{test_year}"
            results[key] = {"all_MAE": all_mae, "stable_MAE": stable_mae}

            marker = " (self)" if train_year == test_year else ""
            print(f"  Train {train_year} → Test {test_year}: "
                  f"all={all_mae:.4f}, stable={stable_mae:.4f}{marker}")

        del model
        torch.cuda.empty_cache()

    # Baseline models
    print("\n--- Baseline Models ---")
    for train_year in years:
        if not os.path.exists(BASELINE_CHECKPOINTS[train_year]):
            continue
        model = load_model(BASELINE_CHECKPOINTS[train_year])
        train_info = data_cache[train_year]

        for test_year in years:
            test_info = data_cache[test_year]
            ts = test_samples[test_year]

            preds = predict_baseline(model, ts["x"], train_info["mean"], train_info["std"])
            node_mae = per_node_mae(preds, ts["y"])

            stable_mae = float(np.mean(node_mae[stable_mask]))
            all_mae = float(np.mean(node_mae))

            key = f"baseline_{train_year}_{test_year}"
            results[key] = {"all_MAE": all_mae, "stable_MAE": stable_mae}

            marker = " (self)" if train_year == test_year else ""
            print(f"  Train {train_year} → Test {test_year}: "
                  f"all={all_mae:.4f}, stable={stable_mae:.4f}{marker}")

        del model
        torch.cuda.empty_cache()

    # ===================================================================
    # 5. Summary tables
    # ===================================================================
    print("\n" + "=" * 70)
    print("5. SUMMARY: Stable nodes vs All nodes")
    print("=" * 70)

    for model_type in ["baseline", "pattern_only"]:
        print(f"\n--- {model_type} ---")
        print(f"{'Pair':<20} {'All MAE':>10} {'Stable MAE':>12} {'Diff':>8}")
        print("-" * 52)

        self_all, self_stable = [], []
        cross_all, cross_stable = [], []

        for train_year in years:
            for test_year in years:
                key = f"{model_type}_{train_year}_{test_year}"
                if key not in results:
                    continue
                r = results[key]
                marker = " *" if train_year == test_year else ""
                print(f"  {train_year}→{test_year}{marker:<14} "
                      f"{r['all_MAE']:>10.4f} {r['stable_MAE']:>12.4f} "
                      f"{r['stable_MAE'] - r['all_MAE']:>+8.4f}")
                if train_year == test_year:
                    self_all.append(r["all_MAE"])
                    self_stable.append(r["stable_MAE"])
                else:
                    cross_all.append(r["all_MAE"])
                    cross_stable.append(r["stable_MAE"])

        avg_self_all = np.mean(self_all)
        avg_self_stable = np.mean(self_stable)
        avg_cross_all = np.mean(cross_all)
        avg_cross_stable = np.mean(cross_stable)
        degrade_all = (avg_cross_all - avg_self_all) / avg_self_all * 100
        degrade_stable = (avg_cross_stable - avg_self_stable) / avg_self_stable * 100

        print(f"\n  {'Avg Self':<20} {avg_self_all:>10.4f} {avg_self_stable:>12.4f}")
        print(f"  {'Avg Cross':<20} {avg_cross_all:>10.4f} {avg_cross_stable:>12.4f}")
        print(f"  {'Degradation':<20} {degrade_all:>9.1f}% {degrade_stable:>11.1f}%")

    # ===================================================================
    # 6. Correlation: per-node drift predictors vs degradation (stable only)
    # ===================================================================
    print("\n" + "=" * 70)
    print("6. WHAT PREDICTS DEGRADATION ON STABLE NODES?")
    print("=" * 70)

    for train_year in years:
        model = load_model(PATTERN_CHECKPOINTS[train_year])
        train_info = data_cache[train_year]

        for test_year in years:
            if train_year == test_year:
                continue

            ts_self = test_samples[train_year]
            ts_cross = test_samples[test_year]

            self_preds = predict_pattern_only(model, ts_self["x"], train_info["mean"], train_info["std"])
            cross_preds = predict_pattern_only(model, ts_cross["x"], train_info["mean"], train_info["std"])

            self_node_mae = per_node_mae(self_preds, ts_self["y"])[stable_mask]
            cross_node_mae = per_node_mae(cross_preds, ts_cross["y"])[stable_mask]
            degradation = cross_node_mae - self_node_mae

            # Predictors
            train_mean = data_cache[train_year]["per_node_stats"]["all_mean"][stable_mask]
            test_mean = data_cache[test_year]["per_node_stats"]["all_mean"][stable_mask]
            train_std = data_cache[train_year]["per_node_stats"]["all_std"][stable_mask]
            test_std = data_cache[test_year]["per_node_stats"]["all_std"][stable_mask]

            scale_ratio = test_mean / (train_mean + 1e-6)
            std_ratio = test_std / (train_std + 1e-6)

            # Shape similarity
            prof1 = data_cache[train_year]["avg_daily_profile_train"][:, stable_mask]
            prof2 = data_cache[test_year]["avg_daily_profile_train"][:, stable_mask]
            norm_prof1 = normalized_daily_profile(prof1)
            norm_prof2 = normalized_daily_profile(prof2)
            shape_sim = daily_profile_similarity(norm_prof1, norm_prof2)

            predictors = {
                "scale_ratio": scale_ratio,
                "std_ratio": std_ratio,
                "abs_scale_change": np.abs(scale_ratio - 1),
                "shape_dissimilarity": 1 - shape_sim,
                "train_mean_flow": train_mean,
                "self_mae": self_node_mae,
            }

            print(f"\n--- {train_year} → {test_year} (n={n_stable} stable nodes) ---")
            print(f"  Mean degradation: {np.mean(degradation):+.2f}")
            for name, pred in predictors.items():
                corr = np.corrcoef(pred, degradation)[0, 1]
                print(f"  r({name:25s}, degradation) = {corr:+.4f}")

        del model
        torch.cuda.empty_cache()

    # Save results
    output = {
        "n_stable_nodes": int(n_stable),
        "stable_indices": stable_indices.tolist(),
        "results": results,
    }
    output_path = "eda/concept_drift/stable_node_drift_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
