"""Cross-year evaluation for Pattern-Only trained models.

Same protocol as cross_year_eval_v3.py, but with per-node additive centering:
1. Z-score normalize input with TRAIN year's scaler
2. Subtract per-node mean of input flow (channel 0) — pattern-only centering
3. Model predicts centered pattern
4. Add back per-node mean, then inverse Z-score
5. Compute MAE/RMSE in raw space

Comparison: baseline (standard) vs pattern-only vs oracle
"""
import sys
import os
import json
import glob
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

# Baseline checkpoints (standard training)
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


def find_pattern_only_checkpoints():
    """Find pattern-only model checkpoints (auto-detect hash dirs)."""
    ckpts = {}
    for year in [2022, 2023, 2024]:
        base_dir = f"checkpoints/ConceptDrift_PatternOnly/SAN_BERNARDINO_{year}_Q1_30_12_12"
        if not os.path.isdir(base_dir):
            print(f"  [WARN] No checkpoint dir for {year}: {base_dir}")
            continue
        # Find hash subdirectory
        subdirs = [d for d in os.listdir(base_dir)
                    if os.path.isdir(os.path.join(base_dir, d))]
        if not subdirs:
            print(f"  [WARN] No hash subdir for {year}")
            continue
        hash_dir = subdirs[0]
        ckpt_path = os.path.join(base_dir, hash_dir, "STAEformer_best_val_MAE.pt")
        if os.path.exists(ckpt_path):
            ckpts[year] = ckpt_path
            print(f"  Found {year}: {ckpt_path}")
        else:
            print(f"  [WARN] No checkpoint file for {year}: {ckpt_path}")
    return ckpts


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

    n_val = int(n_total * 0.2)
    test_start = n_train + n_val
    test_data = data[test_start:]

    return test_data, mean, std


def create_samples(data, input_len, output_len):
    samples_x, samples_y = [], []
    for i in range(len(data) - input_len - output_len + 1):
        samples_x.append(data[i:i+input_len])
        samples_y.append(data[i+input_len:i+input_len+output_len, :, 0:1])
    return np.array(samples_x), np.array(samples_y)


def normalize_input(x, mean, std):
    """Z-score normalize flow channel (channel 0 only)."""
    x_norm = x.copy()
    x_norm[:, :, :, 0] = (x[:, :, :, 0] - mean) / std
    return x_norm


def evaluate_baseline(model, test_x, test_y, mean, std, batch_size=64):
    """Standard evaluation (no centering)."""
    test_x_norm = normalize_input(test_x, mean, std)

    all_preds = []
    with torch.no_grad():
        for i in range(0, len(test_x_norm), batch_size):
            bx = torch.FloatTensor(test_x_norm[i:i+batch_size]).to(DEVICE)
            out = model(bx, None, 0, 0, False)["prediction"]
            pred = out.cpu().numpy() * std + mean
            all_preds.append(pred)

    preds = np.concatenate(all_preds, axis=0)
    mae = float(np.mean(np.abs(preds - test_y)))
    return mae, preds


def evaluate_pattern_only(model, test_x, test_y, mean, std, batch_size=64):
    """Pattern-only evaluation: center input flow, predict, de-center."""
    test_x_norm = normalize_input(test_x, mean, std)

    all_preds = []
    with torch.no_grad():
        for i in range(0, len(test_x_norm), batch_size):
            bx = torch.FloatTensor(test_x_norm[i:i+batch_size]).to(DEVICE)

            # Per-node mean of input flow (in Z-score space)
            input_flow_mean = bx[:, :, :, 0].mean(dim=1, keepdim=True)  # (B, 1, N)

            # Center input flow
            bx_centered = bx.clone()
            bx_centered[:, :, :, 0] = bx[:, :, :, 0] - input_flow_mean

            # Model predicts centered pattern
            out = model(bx_centered, None, 0, 0, False)["prediction"]  # (B, T, N, 1)

            # De-center: add back per-node mean
            out = out + input_flow_mean.unsqueeze(-1)  # (B, 1, N, 1) broadcasts

            # Inverse Z-score
            pred = out.cpu().numpy() * std + mean
            all_preds.append(pred)

    preds = np.concatenate(all_preds, axis=0)
    mae = float(np.mean(np.abs(preds - test_y)))
    return mae, preds


def compute_oracle(preds, test_y):
    """Target oracle: per-node additive re-centering."""
    pred_node_mean = preds.mean(axis=1, keepdims=True)
    target_node_mean = test_y.mean(axis=1, keepdims=True)
    preds_oracle = preds - pred_node_mean + target_node_mean
    mae = float(np.mean(np.abs(preds_oracle - test_y)))
    return mae


def main():
    years = [2022, 2023, 2024]

    print("=" * 70)
    print("PATTERN-ONLY CROSS-YEAR EVALUATION")
    print("=" * 70)

    # Find pattern-only checkpoints
    print("\nSearching for pattern-only checkpoints...")
    pattern_ckpts = find_pattern_only_checkpoints()

    if not pattern_ckpts:
        print("\nNo pattern-only checkpoints found yet. Training may still be running.")
        return

    # Load test data
    print("\nLoading test data...")
    test_data_cache = {}
    for year in years:
        test_data, mean, std = load_data_and_scaler(DATASETS[year])
        test_x, test_y = create_samples(test_data, INPUT_LEN, OUTPUT_LEN)
        test_data_cache[year] = {"test_x": test_x, "test_y": test_y, "mean": mean, "std": std}
        print(f"  {year} Q1: {len(test_x)} samples, scaler mean={mean:.2f} std={std:.2f}")

    results = {}

    # === Pattern-Only Models ===
    print("\n" + "=" * 70)
    print("PATTERN-ONLY MODELS")
    print("=" * 70)

    for train_year in sorted(pattern_ckpts.keys()):
        print(f"\n--- Pattern-Only model trained on {train_year} Q1 ---")
        model = load_model(pattern_ckpts[train_year])
        train_cache = test_data_cache[train_year]

        for test_year in years:
            cache = test_data_cache[test_year]
            # Use TRAIN year's scaler
            mae, preds = evaluate_pattern_only(
                model, cache["test_x"], cache["test_y"],
                train_cache["mean"], train_cache["std"]
            )
            oracle_mae = compute_oracle(preds, cache["test_y"])

            key = f"train_{train_year}_test_{test_year}"
            results[f"pattern_only_{key}"] = {"MAE": mae, "oracle_MAE": oracle_mae}
            marker = " (self)" if train_year == test_year else ""
            print(f"  Test {test_year}: MAE={mae:.4f}, Oracle={oracle_mae:.4f}{marker}")

        del model
        torch.cuda.empty_cache()

    # === Baseline Models (for comparison) ===
    print("\n" + "=" * 70)
    print("BASELINE MODELS (standard training)")
    print("=" * 70)

    for train_year in years:
        if not os.path.exists(BASELINE_CHECKPOINTS[train_year]):
            print(f"  [SKIP] No baseline checkpoint for {train_year}")
            continue
        print(f"\n--- Baseline model trained on {train_year} Q1 ---")
        model = load_model(BASELINE_CHECKPOINTS[train_year])
        train_cache = test_data_cache[train_year]

        for test_year in years:
            cache = test_data_cache[test_year]
            mae, preds = evaluate_baseline(
                model, cache["test_x"], cache["test_y"],
                train_cache["mean"], train_cache["std"]
            )
            oracle_mae = compute_oracle(preds, cache["test_y"])

            key = f"train_{train_year}_test_{test_year}"
            results[f"baseline_{key}"] = {"MAE": mae, "oracle_MAE": oracle_mae}
            marker = " (self)" if train_year == test_year else ""
            print(f"  Test {test_year}: MAE={mae:.4f}, Oracle={oracle_mae:.4f}{marker}")

        del model
        torch.cuda.empty_cache()

    # === Summary Table ===
    print("\n" + "=" * 70)
    print("SUMMARY: Self-year MAE")
    print("=" * 70)
    print(f"{'Model':<20} {'2022':>10} {'2023':>10} {'2024':>10} {'Avg':>10}")
    print("-" * 60)

    for model_type in ["baseline", "pattern_only"]:
        row = f"{model_type:<20}"
        maes = []
        for year in years:
            key = f"{model_type}_train_{year}_test_{year}"
            if key in results:
                mae = results[key]["MAE"]
                row += f"{mae:>10.4f}"
                maes.append(mae)
            else:
                row += f"{'N/A':>10}"
        if maes:
            row += f"{np.mean(maes):>10.4f}"
        print(row)

    print("\n" + "=" * 70)
    print("SUMMARY: Cross-year MAE (avg of 6 cross pairs)")
    print("=" * 70)
    print(f"{'Model':<20} {'Avg Cross':>12} {'Avg Self':>12} {'Degradation':>14}")
    print("-" * 60)

    for model_type in ["baseline", "pattern_only"]:
        cross_maes = []
        self_maes = []
        for train_year in years:
            for test_year in years:
                key = f"{model_type}_train_{train_year}_test_{test_year}"
                if key in results:
                    if train_year == test_year:
                        self_maes.append(results[key]["MAE"])
                    else:
                        cross_maes.append(results[key]["MAE"])
        if cross_maes and self_maes:
            avg_cross = np.mean(cross_maes)
            avg_self = np.mean(self_maes)
            degrade = (avg_cross - avg_self) / avg_self * 100
            print(f"{model_type:<20} {avg_cross:>12.4f} {avg_self:>12.4f} {degrade:>13.1f}%")

    # Save
    output_path = "eda/concept_drift/cross_year_pattern_only_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
