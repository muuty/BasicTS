"""Cross-year evaluation for DecomposedSTAEformer.

Compares decomposed prediction (learned scale head + instance-normed backbone)
against baseline and test-time instance normalization.
"""
import sys
import os
import json
import numpy as np
import torch
sys.path.append("/data/pretrainingbasicts")

from baselines.STAEformer.arch.decomposed_staeformer import DecomposedSTAEformer
from baselines.STAEformer.arch import STAEformer

BACKBONE_PARAMS = {
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

MODEL_PARAM = {
    "backbone_params": BACKBONE_PARAMS,
    "scale_head_hidden": 32,
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
DEVICE = "cuda:0"


def find_decomposed_checkpoints():
    """Find DecomposedSTAEformer checkpoints."""
    ckpts = {}
    for year in [2022, 2023, 2024]:
        base_dir = f"checkpoints/ConceptDrift_Decomposed/SAN_BERNARDINO_{year}_Q1_30_12_12"
        if not os.path.isdir(base_dir):
            continue
        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if not subdirs:
            continue
        hash_dir = subdirs[0]
        ckpt_path = os.path.join(base_dir, hash_dir, "DecomposedSTAEformer_best_val_MAE.pt")
        if os.path.exists(ckpt_path):
            ckpts[year] = ckpt_path
        else:
            print(f"  Warning: checkpoint not found at {ckpt_path}")
    return ckpts


def load_decomposed_model(ckpt_path):
    model = DecomposedSTAEformer(**MODEL_PARAM)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE).eval()
    return model


def load_baseline_model(ckpt_path):
    model = STAEformer(**BACKBONE_PARAMS)
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
    x_norm = x.copy()
    x_norm[:, :, :, 0] = (x[:, :, :, 0] - mean) / std
    return x_norm


def evaluate_decomposed(model, test_x, test_y, mean, std, batch_size=64):
    """Evaluate decomposed model. Model handles instance norm + scale prediction internally."""
    test_x_norm = normalize_input(test_x, mean, std)
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(test_x_norm), batch_size):
            bx = torch.FloatTensor(test_x_norm[i:i+batch_size]).to(DEVICE)
            out = model(bx, None, 0, 0, False)
            # Model output is in Z-score space (prediction = mu_y + sigma_y * r_y in Z space)
            pred = out['prediction'].cpu().numpy() * std + mean
            all_preds.append(pred)
    preds = np.concatenate(all_preds, axis=0)
    mae = float(np.mean(np.abs(preds - test_y)))
    per_node_mae = np.mean(np.abs(preds - test_y), axis=(0, 1, 3))
    return mae, preds, per_node_mae


def evaluate_baseline(model, test_x, test_y, mean, std, batch_size=64):
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
    per_node_mae = np.mean(np.abs(preds - test_y), axis=(0, 1, 3))
    return mae, preds, per_node_mae


def evaluate_instance_norm(model, test_x, test_y, mean, std, batch_size=64):
    """Test-time instance norm on baseline model (for comparison)."""
    test_x_norm = normalize_input(test_x, mean, std)
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(test_x_norm), batch_size):
            bx = torch.FloatTensor(test_x_norm[i:i+batch_size]).to(DEVICE)
            flow = bx[:, :, :, 0]
            input_flow_mean = flow.mean(dim=1, keepdim=True)
            input_flow_std = flow.std(dim=1, keepdim=True) + 1e-5
            bx_normed = bx.clone()
            bx_normed[:, :, :, 0] = (flow - input_flow_mean) / input_flow_std
            out = model(bx_normed, None, 0, 0, False)["prediction"]
            out = out * input_flow_std.unsqueeze(-1) + input_flow_mean.unsqueeze(-1)
            pred = out.cpu().numpy() * std + mean
            all_preds.append(pred)
    preds = np.concatenate(all_preds, axis=0)
    mae = float(np.mean(np.abs(preds - test_y)))
    per_node_mae = np.mean(np.abs(preds - test_y), axis=(0, 1, 3))
    return mae, preds, per_node_mae


def main():
    years = [2022, 2023, 2024]

    print("=" * 70)
    print("CROSS-YEAR EVALUATION: DECOMPOSED vs BASELINE vs INSTANCE NORM")
    print("=" * 70)

    # Find decomposed checkpoints
    decomposed_ckpts = find_decomposed_checkpoints()
    print(f"\nDecomposed checkpoints: {list(decomposed_ckpts.keys())}")
    print(f"Baseline checkpoints: {list(BASELINE_CHECKPOINTS.keys())}")

    if not decomposed_ckpts:
        print("\nNo decomposed checkpoints found. Training may still be running.")
        return

    # Load test data
    print("\nLoading test data...")
    test_data_cache = {}
    for year in years:
        test_data, mean, std = load_data_and_scaler(DATASETS[year])
        test_x, test_y = create_samples(test_data, INPUT_LEN, OUTPUT_LEN)
        test_data_cache[year] = {"test_x": test_x, "test_y": test_y, "mean": mean, "std": std}
        print(f"  {year} Q1: {len(test_x)} samples, scaler mean={mean:.2f} std={std:.2f}")

    # Load stable functional indices
    stable_indices = None
    for path in ["datasets/SAN_BERNARDINO_2022_Q1/stable_functional_indices.npy",
                 "eda/concept_drift/functional_indices.npy"]:
        if os.path.exists(path):
            stable_indices = np.load(path)
            print(f"\nLoaded {len(stable_indices)} stable functional node indices from {path}")
            break

    results = {}

    # === Evaluate Decomposed Model ===
    print(f"\n{'=' * 70}")
    print("DECOMPOSED MODEL (learned scale head + instance-normed backbone)")
    print(f"{'=' * 70}")

    for train_year in sorted(decomposed_ckpts.keys()):
        print(f"\n--- Decomposed trained on {train_year} Q1 ---")
        model = load_decomposed_model(decomposed_ckpts[train_year])
        train_cache = test_data_cache[train_year]

        for test_year in years:
            cache = test_data_cache[test_year]
            mae, preds, per_node_mae = evaluate_decomposed(
                model, cache["test_x"], cache["test_y"],
                train_cache["mean"], train_cache["std"]
            )

            result_entry = {"MAE": mae}
            if stable_indices is not None:
                stable_mae = float(per_node_mae[stable_indices].mean())
                result_entry["stable_MAE"] = stable_mae

            key = f"decomposed_train_{train_year}_test_{test_year}"
            results[key] = result_entry
            marker = " (self)" if train_year == test_year else ""
            stable_str = f", stable={result_entry.get('stable_MAE', 'N/A'):.4f}" if stable_indices is not None else ""
            print(f"  Test {test_year}: MAE={mae:.4f}{stable_str}{marker}")

        del model
        torch.cuda.empty_cache()

    # === Evaluate Baseline + Instance Norm (from existing results or re-compute) ===
    existing_results_path = "eda/concept_drift/cross_year_all_methods_results.json"
    if os.path.exists(existing_results_path):
        print(f"\nLoading existing baseline/instance_norm results from {existing_results_path}")
        with open(existing_results_path) as f:
            existing = json.load(f)
        # Copy baseline and instance_norm results
        for k, v in existing.items():
            if k.startswith("baseline_") or k.startswith("instance_norm_"):
                results[k] = v
    else:
        print("\nRe-computing baseline and instance_norm results...")
        for train_year in years:
            if not os.path.exists(BASELINE_CHECKPOINTS[train_year]):
                continue
            model = load_baseline_model(BASELINE_CHECKPOINTS[train_year])
            train_cache = test_data_cache[train_year]

            for test_year in years:
                cache = test_data_cache[test_year]

                # Baseline
                mae, _, per_node_mae = evaluate_baseline(
                    model, cache["test_x"], cache["test_y"],
                    train_cache["mean"], train_cache["std"]
                )
                entry = {"MAE": mae}
                if stable_indices is not None:
                    entry["stable_MAE"] = float(per_node_mae[stable_indices].mean())
                results[f"baseline_train_{train_year}_test_{test_year}"] = entry

                # Instance norm
                mae, _, per_node_mae = evaluate_instance_norm(
                    model, cache["test_x"], cache["test_y"],
                    train_cache["mean"], train_cache["std"]
                )
                entry = {"MAE": mae}
                if stable_indices is not None:
                    entry["stable_MAE"] = float(per_node_mae[stable_indices].mean())
                results[f"instance_norm_train_{train_year}_test_{test_year}"] = entry

            del model
            torch.cuda.empty_cache()

    # === Summary Table ===
    print("\n" + "=" * 70)
    print("SUMMARY: Cross-Year Degradation")
    print("=" * 70)

    for metric_key, metric_label in [("MAE", "All Nodes"), ("stable_MAE", "Stable Functional")]:
        print(f"\n--- {metric_label} ---")
        print(f"{'Method':<20} {'Avg Self':>10} {'Avg Cross':>10} {'Degrade':>10} {'Degrade%':>10}")
        print("-" * 62)

        for method_name in ["baseline", "instance_norm", "decomposed"]:
            cross_vals, self_vals = [], []
            for train_year in years:
                for test_year in years:
                    key = f"{method_name}_train_{train_year}_test_{test_year}"
                    if key not in results or metric_key not in results[key]:
                        continue
                    val = results[key][metric_key]
                    if train_year == test_year:
                        self_vals.append(val)
                    else:
                        cross_vals.append(val)

            if cross_vals and self_vals:
                avg_self = np.mean(self_vals)
                avg_cross = np.mean(cross_vals)
                degrade = avg_cross - avg_self
                degrade_pct = degrade / avg_self * 100
                print(f"{method_name:<20} {avg_self:>10.4f} {avg_cross:>10.4f} {degrade:>+10.4f} {degrade_pct:>+9.1f}%")

    # === Detailed Cross-Year Matrix ===
    print("\n" + "=" * 70)
    print("DETAILED CROSS-YEAR MATRIX (MAE)")
    print("=" * 70)

    for method_name in ["baseline", "instance_norm", "decomposed"]:
        print(f"\n{method_name.upper()}:")
        header = "Train\\Test"
        print(f"{header:<12}", end="")
        for y in years:
            print(f"  {y:>10}", end="")
        print()
        print("-" * 44)

        for train_year in years:
            print(f"{train_year:<12}", end="")
            for test_year in years:
                key = f"{method_name}_train_{train_year}_test_{test_year}"
                if key in results:
                    mae = results[key]["MAE"]
                    marker = "*" if train_year == test_year else " "
                    print(f"  {mae:>9.4f}{marker}", end="")
                else:
                    print(f"  {'N/A':>10}", end="")
            print()

    # === Improvement over baseline ===
    print("\n" + "=" * 70)
    print("IMPROVEMENT: Decomposed vs Baseline (cross-year MAE reduction)")
    print("=" * 70)

    for train_year in years:
        for test_year in years:
            if train_year == test_year:
                continue
            bl_key = f"baseline_train_{train_year}_test_{test_year}"
            dc_key = f"decomposed_train_{train_year}_test_{test_year}"
            in_key = f"instance_norm_train_{train_year}_test_{test_year}"
            if all(k in results for k in [bl_key, dc_key]):
                bl_mae = results[bl_key]["MAE"]
                dc_mae = results[dc_key]["MAE"]
                in_mae = results[in_key]["MAE"] if in_key in results else None
                improvement = (bl_mae - dc_mae) / bl_mae * 100
                in_str = f", inst_norm={in_mae:.4f}" if in_mae else ""
                print(f"  Train {train_year} -> Test {test_year}: baseline={bl_mae:.4f}, decomposed={dc_mae:.4f} ({improvement:+.1f}%){in_str}")

    # Save results
    output_path = "eda/concept_drift/cross_year_decomposed_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
