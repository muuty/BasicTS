"""Analyze the remaining 44% degradation after pattern-only centering.

Per-node analysis to understand:
1. Which nodes contribute most to cross-year degradation?
2. Is degradation correlated with sensor health, scale, or location?
3. What types of drift remain after scale removal?
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


def load_data_and_scaler(dataset_dir):
    desc = json.load(open(os.path.join(dataset_dir, "desc.json")))
    shape = desc["shape"]
    data = np.memmap(os.path.join(dataset_dir, "data.dat"), dtype="float32", mode="r").reshape(shape)
    n_total = shape[0]
    n_train = int(n_total * TRAIN_RATIO)
    train_data_ch0 = data[:n_train, :, 0]
    mean = float(np.mean(train_data_ch0))
    std = float(np.std(train_data_ch0))
    # Per-node train statistics
    per_node_train_mean = np.mean(train_data_ch0, axis=0)  # (N,)
    per_node_train_std = np.std(train_data_ch0, axis=0)    # (N,)
    n_val = int(n_total * 0.2)
    test_start = n_train + n_val
    test_data = data[test_start:]
    # Per-node test statistics
    test_data_ch0 = test_data[:, :, 0]
    per_node_test_mean = np.mean(test_data_ch0, axis=0)  # (N,)
    per_node_test_std = np.std(test_data_ch0, axis=0)    # (N,)
    # Zero rate (sensor health)
    all_zero = (data[:n_train, :, 0] == 0) & (data[:n_train, :, 1] == 0) & (data[:n_train, :, 2] == 0)
    zero_rate = np.mean(all_zero, axis=0)  # (N,)
    return {
        "test_data": test_data,
        "mean": mean, "std": std,
        "per_node_train_mean": per_node_train_mean,
        "per_node_train_std": per_node_train_std,
        "per_node_test_mean": per_node_test_mean,
        "per_node_test_std": per_node_test_std,
        "zero_rate": zero_rate,
    }


def create_samples(data, input_len, output_len):
    samples_x, samples_y = [], []
    for i in range(len(data) - input_len - output_len + 1):
        samples_x.append(data[i:i+input_len])
        samples_y.append(data[i+input_len:i+input_len+output_len, :, 0:1])
    return np.array(samples_x), np.array(samples_y)


def normalize_input(x, mean, std):
    x_norm = x.copy()
    x_norm[:, :, :, 0] = (x[:, :, :, 0] - mean) / std
    return x_norm


def predict_pattern_only(model, test_x, mean, std, batch_size=64):
    """Pattern-only prediction, return per-sample predictions."""
    test_x_norm = normalize_input(test_x, mean, std)
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(test_x_norm), batch_size):
            bx = torch.FloatTensor(test_x_norm[i:i+batch_size]).to(DEVICE)
            input_flow_mean = bx[:, :, :, 0].mean(dim=1, keepdim=True)
            bx_centered = bx.clone()
            bx_centered[:, :, :, 0] = bx[:, :, :, 0] - input_flow_mean
            out = model(bx_centered, None, 0, 0, False)["prediction"]
            out = out + input_flow_mean.unsqueeze(-1)
            pred = out.cpu().numpy() * std + mean
            all_preds.append(pred)
    return np.concatenate(all_preds, axis=0)


def predict_baseline(model, test_x, mean, std, batch_size=64):
    """Standard prediction."""
    test_x_norm = normalize_input(test_x, mean, std)
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(test_x_norm), batch_size):
            bx = torch.FloatTensor(test_x_norm[i:i+batch_size]).to(DEVICE)
            out = model(bx, None, 0, 0, False)["prediction"]
            pred = out.cpu().numpy() * std + mean
            all_preds.append(pred)
    return np.concatenate(all_preds, axis=0)


def per_node_mae(preds, targets):
    """Compute MAE per node. preds/targets: (S, T, N, 1)"""
    return np.mean(np.abs(preds - targets), axis=(0, 1, 3))  # (N,)


def main():
    years = [2022, 2023, 2024]

    print("=" * 70)
    print("RESIDUAL DRIFT ANALYSIS (after pattern-only centering)")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    data_cache = {}
    for year in years:
        info = load_data_and_scaler(DATASETS[year])
        test_x, test_y = create_samples(info["test_data"], INPUT_LEN, OUTPUT_LEN)
        info["test_x"] = test_x
        info["test_y"] = test_y
        data_cache[year] = info
        print(f"  {year}: {len(test_x)} samples, mean={info['mean']:.1f}, std={info['std']:.1f}")

    # ===================================================================
    # 1. Per-node cross-year MAE for pattern-only models
    # ===================================================================
    print("\n" + "=" * 70)
    print("1. PER-NODE CROSS-YEAR MAE (pattern-only)")
    print("=" * 70)

    # We'll focus on the most informative pairs
    analysis_pairs = [
        (2022, 2024, "normal→normal"),
        (2022, 2023, "normal→anomaly"),
        (2024, 2022, "normal→normal (reverse)"),
        (2024, 2023, "normal→anomaly (reverse)"),
    ]

    per_node_results = {}

    for train_year, test_year, label in analysis_pairs:
        print(f"\n--- Train {train_year} → Test {test_year} ({label}) ---")
        model = load_model(PATTERN_CHECKPOINTS[train_year])
        train_info = data_cache[train_year]
        test_info = data_cache[test_year]

        # Pattern-only prediction using TRAIN year's scaler
        preds = predict_pattern_only(
            model, test_info["test_x"], train_info["mean"], train_info["std"]
        )
        node_mae = per_node_mae(preds, test_info["test_y"])  # (893,)

        # Self-year for comparison
        self_preds = predict_pattern_only(
            model, train_info["test_x"], train_info["mean"], train_info["std"]
        )
        self_node_mae = per_node_mae(self_preds, train_info["test_y"])  # (893,)

        # Per-node degradation
        node_degradation = node_mae - self_node_mae  # absolute increase
        node_degradation_pct = np.where(
            self_node_mae > 0.1,
            (node_mae - self_node_mae) / self_node_mae * 100,
            0
        )

        key = f"{train_year}_{test_year}"
        per_node_results[key] = {
            "cross_mae": node_mae,
            "self_mae": self_node_mae,
            "degradation": node_degradation,
            "degradation_pct": node_degradation_pct,
        }

        # Summary stats
        print(f"  Overall: self={np.mean(self_node_mae):.2f}, cross={np.mean(node_mae):.2f}, "
              f"degrade={np.mean(node_degradation):.2f}")

        # By sensor category
        zero_rate = train_info["zero_rate"]
        categories = {
            "dead (>90%)": zero_rate > 0.9,
            "major_fail (50-90%)": (zero_rate > 0.5) & (zero_rate <= 0.9),
            "partial (5-50%)": (zero_rate > 0.05) & (zero_rate <= 0.5),
            "functional (<5%)": zero_rate <= 0.05,
        }

        for cat_name, mask in categories.items():
            n = mask.sum()
            if n > 0:
                cat_self = np.mean(self_node_mae[mask])
                cat_cross = np.mean(node_mae[mask])
                cat_degrade = np.mean(node_degradation[mask])
                print(f"  {cat_name:25s} (n={n:3d}): self={cat_self:6.2f}, "
                      f"cross={cat_cross:6.2f}, degrade={cat_degrade:+6.2f}")

        del model
        torch.cuda.empty_cache()

    # ===================================================================
    # 2. Worst degradation nodes analysis
    # ===================================================================
    print("\n" + "=" * 70)
    print("2. WORST DEGRADED NODES (pattern-only, train 2022 → test 2023)")
    print("=" * 70)

    key = "2022_2023"
    r = per_node_results[key]
    func_mask = data_cache[2022]["zero_rate"] <= 0.05

    # Sort functional nodes by degradation
    func_indices = np.where(func_mask)[0]
    func_degrade = r["degradation"][func_mask]
    sorted_idx = np.argsort(-func_degrade)  # descending

    print(f"\nTop 20 worst-degraded functional nodes (train 2022 → test 2023):")
    print(f"{'Node':>6} {'Self MAE':>10} {'Cross MAE':>10} {'Degrade':>10} {'Degrade%':>10} "
          f"{'TrainMean':>10} {'TestMean':>10} {'MeanShift%':>12}")

    train_node_mean = data_cache[2022]["per_node_train_mean"]
    test_node_mean = data_cache[2023]["per_node_test_mean"]

    for i in range(20):
        idx = func_indices[sorted_idx[i]]
        s_mae = r["self_mae"][idx]
        c_mae = r["cross_mae"][idx]
        deg = r["degradation"][idx]
        deg_pct = r["degradation_pct"][idx]
        tr_mean = train_node_mean[idx]
        te_mean = test_node_mean[idx]
        mean_shift = (te_mean - tr_mean) / (tr_mean + 1e-6) * 100
        print(f"{idx:6d} {s_mae:10.2f} {c_mae:10.2f} {deg:+10.2f} {deg_pct:+9.1f}% "
              f"{tr_mean:10.1f} {te_mean:10.1f} {mean_shift:+11.1f}%")

    # ===================================================================
    # 3. Correlation analysis: what predicts per-node degradation?
    # ===================================================================
    print("\n" + "=" * 70)
    print("3. CORRELATION ANALYSIS: What predicts per-node degradation?")
    print("=" * 70)

    for key_pair, (train_yr, test_yr) in [("2022_2023", (2022, 2023)),
                                           ("2022_2024", (2022, 2024)),
                                           ("2024_2023", (2024, 2023))]:
        r = per_node_results[key_pair]
        func_mask = data_cache[train_yr]["zero_rate"] <= 0.05
        degrade = r["degradation"][func_mask]

        # Candidate predictors (functional nodes only)
        train_mean = data_cache[train_yr]["per_node_train_mean"][func_mask]
        test_mean = data_cache[test_yr]["per_node_test_mean"][func_mask]
        train_std = data_cache[train_yr]["per_node_train_std"][func_mask]
        test_std = data_cache[test_yr]["per_node_test_std"][func_mask]

        mean_shift = np.abs(test_mean - train_mean)
        mean_shift_pct = mean_shift / (train_mean + 1e-6)
        std_shift = np.abs(test_std - train_std)
        std_shift_pct = std_shift / (train_std + 1e-6)
        scale_ratio = test_mean / (train_mean + 1e-6)

        predictors = {
            "abs_mean_shift": mean_shift,
            "pct_mean_shift": mean_shift_pct,
            "abs_std_shift": std_shift,
            "pct_std_shift": std_shift_pct,
            "train_mean_flow": train_mean,
            "scale_ratio": scale_ratio,
            "self_mae": r["self_mae"][func_mask],
        }

        print(f"\n--- {train_yr} → {test_yr} (functional nodes, n={func_mask.sum()}) ---")
        for name, pred in predictors.items():
            corr = np.corrcoef(pred, degrade)[0, 1]
            print(f"  r({name:20s}, degradation) = {corr:+.4f}")

    # ===================================================================
    # 4. Degradation distribution
    # ===================================================================
    print("\n" + "=" * 70)
    print("4. DEGRADATION DISTRIBUTION (functional nodes)")
    print("=" * 70)

    for key_pair, (train_yr, test_yr) in [("2022_2023", (2022, 2023)),
                                           ("2022_2024", (2022, 2024))]:
        r = per_node_results[key_pair]
        func_mask = data_cache[train_yr]["zero_rate"] <= 0.05
        degrade = r["degradation"][func_mask]

        improved = (degrade < -0.5).sum()
        stable = (np.abs(degrade) <= 0.5).sum()
        mild = ((degrade > 0.5) & (degrade <= 2)).sum()
        moderate = ((degrade > 2) & (degrade <= 5)).sum()
        severe = ((degrade > 5) & (degrade <= 10)).sum()
        extreme = (degrade > 10).sum()

        total = func_mask.sum()
        print(f"\n--- {train_yr} → {test_yr} ---")
        print(f"  Improved (<-0.5):     {improved:4d} ({improved/total*100:5.1f}%)")
        print(f"  Stable (±0.5):        {stable:4d} ({stable/total*100:5.1f}%)")
        print(f"  Mild (0.5-2):         {mild:4d} ({mild/total*100:5.1f}%)")
        print(f"  Moderate (2-5):       {moderate:4d} ({moderate/total*100:5.1f}%)")
        print(f"  Severe (5-10):        {severe:4d} ({severe/total*100:5.1f}%)")
        print(f"  Extreme (>10):        {extreme:4d} ({extreme/total*100:5.1f}%)")
        print(f"  Mean degrade: {np.mean(degrade):+.2f}, Median: {np.median(degrade):+.2f}")

        # Contribution to total degradation
        total_degrade = np.sum(np.maximum(degrade, 0))
        top10_degrade = np.sum(np.sort(degrade)[-int(total*0.1):])
        top20_degrade = np.sum(np.sort(degrade)[-int(total*0.2):])
        print(f"  Top 10% nodes contribute: {top10_degrade/total_degrade*100:.1f}% of total degradation")
        print(f"  Top 20% nodes contribute: {top20_degrade/total_degrade*100:.1f}% of total degradation")

    # ===================================================================
    # 5. Compare with baseline degradation
    # ===================================================================
    print("\n" + "=" * 70)
    print("5. PATTERN-ONLY vs BASELINE: Per-node degradation comparison")
    print("=" * 70)

    # Load baseline model for 2022→2023
    print("\nLoading baseline model (2022)...")
    baseline_model = load_model(BASELINE_CHECKPOINTS[2022])
    train_info = data_cache[2022]
    test_info = data_cache[2023]

    baseline_cross_preds = predict_baseline(
        baseline_model, test_info["test_x"], train_info["mean"], train_info["std"]
    )
    baseline_cross_mae = per_node_mae(baseline_cross_preds, test_info["test_y"])

    baseline_self_preds = predict_baseline(
        baseline_model, train_info["test_x"], train_info["mean"], train_info["std"]
    )
    baseline_self_mae = per_node_mae(baseline_self_preds, train_info["test_y"])
    baseline_degrade = baseline_cross_mae - baseline_self_mae

    del baseline_model
    torch.cuda.empty_cache()

    func_mask = data_cache[2022]["zero_rate"] <= 0.05
    po_degrade = per_node_results["2022_2023"]["degradation"]

    print(f"\nFunctional nodes (n={func_mask.sum()}), Train 2022 → Test 2023:")
    print(f"  Baseline avg degradation:      {np.mean(baseline_degrade[func_mask]):+.2f}")
    print(f"  Pattern-only avg degradation:   {np.mean(po_degrade[func_mask]):+.2f}")
    print(f"  Correlation(baseline_deg, po_deg): {np.corrcoef(baseline_degrade[func_mask], po_degrade[func_mask])[0,1]:.4f}")

    # Nodes where pattern-only still has high degradation
    po_still_bad = func_mask & (po_degrade > 5)
    baseline_also_bad = func_mask & (baseline_degrade > 5)
    both_bad = po_still_bad & baseline_also_bad
    po_only_bad = po_still_bad & ~baseline_also_bad
    baseline_only_bad = baseline_also_bad & ~po_still_bad

    print(f"\n  Nodes with >5 MAE degradation:")
    print(f"    Both bad:          {both_bad.sum()}")
    print(f"    Pattern-only bad only: {po_only_bad.sum()}")
    print(f"    Baseline bad only:    {baseline_only_bad.sum()}")

    # ===================================================================
    # 6. Save per-node results
    # ===================================================================
    save_data = {}
    for key, r in per_node_results.items():
        save_data[key] = {
            "cross_mae": r["cross_mae"].tolist(),
            "self_mae": r["self_mae"].tolist(),
            "degradation": r["degradation"].tolist(),
        }
    save_data["baseline_2022_2023"] = {
        "cross_mae": baseline_cross_mae.tolist(),
        "self_mae": baseline_self_mae.tolist(),
        "degradation": baseline_degrade.tolist(),
    }
    save_data["zero_rate_2022"] = data_cache[2022]["zero_rate"].tolist()
    save_data["per_node_train_mean"] = {
        str(y): data_cache[y]["per_node_train_mean"].tolist() for y in years
    }
    save_data["per_node_test_mean"] = {
        str(y): data_cache[y]["per_node_test_mean"].tolist() for y in years
    }

    output_path = "eda/concept_drift/residual_drift_analysis.json"
    with open(output_path, "w") as f:
        json.dump(save_data, f)
    print(f"\nPer-node results saved to {output_path}")


if __name__ == "__main__":
    main()
