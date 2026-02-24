"""Cross-year evaluation comparing ALL centering methods.

Methods:
1. Baseline: standard training, no centering
2. Pattern-only: subtract per-node mean (additive centering)
3. Instance-norm: subtract mean + divide std (full scale removal)

This isolates:
- Additive scale contribution: baseline - pattern_only
- Multiplicative scale contribution: pattern_only - instance_norm
- Pure pattern drift: instance_norm degradation (what remains after removing ALL scale)
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


def find_checkpoints(ckpt_dir_prefix):
    """Find model checkpoints (auto-detect hash dirs)."""
    ckpts = {}
    for year in [2022, 2023, 2024]:
        base_dir = f"checkpoints/{ckpt_dir_prefix}/SAN_BERNARDINO_{year}_Q1_30_12_12"
        if not os.path.isdir(base_dir):
            continue
        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if not subdirs:
            continue
        hash_dir = subdirs[0]
        ckpt_path = os.path.join(base_dir, hash_dir, "STAEformer_best_val_MAE.pt")
        if os.path.exists(ckpt_path):
            ckpts[year] = ckpt_path
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
    x_norm = x.copy()
    x_norm[:, :, :, 0] = (x[:, :, :, 0] - mean) / std
    return x_norm


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
    # Per-node MAE
    per_node_mae = np.mean(np.abs(preds - test_y), axis=(0, 1, 3))  # (N,)
    return mae, preds, per_node_mae


def evaluate_pattern_only(model, test_x, test_y, mean, std, batch_size=64):
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
    preds = np.concatenate(all_preds, axis=0)
    mae = float(np.mean(np.abs(preds - test_y)))
    per_node_mae = np.mean(np.abs(preds - test_y), axis=(0, 1, 3))
    return mae, preds, per_node_mae


def evaluate_instance_norm(model, test_x, test_y, mean, std, batch_size=64):
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
    print("CROSS-YEAR EVALUATION: ALL CENTERING METHODS")
    print("=" * 70)

    # Find checkpoints
    pattern_ckpts = find_checkpoints("ConceptDrift_PatternOnly")
    instance_norm_ckpts = find_checkpoints("ConceptDrift_InstanceNorm")

    print(f"\nBaseline checkpoints: {list(BASELINE_CHECKPOINTS.keys())}")
    print(f"Pattern-only checkpoints: {list(pattern_ckpts.keys())}")
    print(f"Instance-norm checkpoints: {list(instance_norm_ckpts.keys())}")

    if not instance_norm_ckpts:
        print("\nNo instance-norm checkpoints found. Training may still be running.")
        return

    # Load test data
    print("\nLoading test data...")
    test_data_cache = {}
    for year in years:
        test_data, mean, std = load_data_and_scaler(DATASETS[year])
        test_x, test_y = create_samples(test_data, INPUT_LEN, OUTPUT_LEN)
        test_data_cache[year] = {"test_x": test_x, "test_y": test_y, "mean": mean, "std": std}
        print(f"  {year} Q1: {len(test_x)} samples, scaler mean={mean:.2f} std={std:.2f}")

    # Load stable functional indices for per-node analysis
    stable_indices_path = "datasets/SAN_BERNARDINO_2022_Q1/stable_functional_indices.npy"
    stable_indices = None
    if os.path.exists(stable_indices_path):
        stable_indices = np.load(stable_indices_path)
        print(f"\nLoaded {len(stable_indices)} stable functional node indices")

    results = {}

    methods = [
        ("baseline", BASELINE_CHECKPOINTS, evaluate_baseline),
        ("pattern_only", pattern_ckpts, evaluate_pattern_only),
        ("instance_norm", instance_norm_ckpts, evaluate_instance_norm),
    ]

    for method_name, ckpts, eval_fn in methods:
        print(f"\n{'=' * 70}")
        print(f"{method_name.upper()} MODELS")
        print(f"{'=' * 70}")

        for train_year in sorted(ckpts.keys()):
            if not os.path.exists(ckpts[train_year]):
                continue
            print(f"\n--- {method_name} trained on {train_year} Q1 ---")
            model = load_model(ckpts[train_year])
            train_cache = test_data_cache[train_year]

            for test_year in years:
                cache = test_data_cache[test_year]
                mae, preds, per_node_mae = eval_fn(
                    model, cache["test_x"], cache["test_y"],
                    train_cache["mean"], train_cache["std"]
                )

                result_entry = {"MAE": mae}

                # Stable-only MAE
                if stable_indices is not None:
                    stable_mae = float(per_node_mae[stable_indices].mean())
                    result_entry["stable_MAE"] = stable_mae

                key = f"{method_name}_train_{train_year}_test_{test_year}"
                results[key] = result_entry
                marker = " (self)" if train_year == test_year else ""
                stable_str = f", stable={result_entry.get('stable_MAE', 'N/A'):.4f}" if stable_indices is not None else ""
                print(f"  Test {test_year}: MAE={mae:.4f}{stable_str}{marker}")

            del model
            torch.cuda.empty_cache()

    # === Summary Table ===
    print("\n" + "=" * 70)
    print("SUMMARY: Degradation Analysis")
    print("=" * 70)
    print(f"\n{'Method':<20} {'Avg Self':>10} {'Avg Cross':>10} {'Degrade%':>10} {'Stable Self':>12} {'Stable Cross':>12} {'Stable Deg%':>12}")
    print("-" * 86)

    for method_name in ["baseline", "pattern_only", "instance_norm"]:
        cross_maes, self_maes = [], []
        cross_stable, self_stable = [], []
        for train_year in years:
            for test_year in years:
                key = f"{method_name}_train_{train_year}_test_{test_year}"
                if key not in results:
                    continue
                if train_year == test_year:
                    self_maes.append(results[key]["MAE"])
                    if "stable_MAE" in results[key]:
                        self_stable.append(results[key]["stable_MAE"])
                else:
                    cross_maes.append(results[key]["MAE"])
                    if "stable_MAE" in results[key]:
                        cross_stable.append(results[key]["stable_MAE"])

        if cross_maes and self_maes:
            avg_cross = np.mean(cross_maes)
            avg_self = np.mean(self_maes)
            degrade = (avg_cross - avg_self) / avg_self * 100

            row = f"{method_name:<20} {avg_self:>10.4f} {avg_cross:>10.4f} {degrade:>9.1f}%"

            if cross_stable and self_stable:
                avg_cross_s = np.mean(cross_stable)
                avg_self_s = np.mean(self_stable)
                degrade_s = (avg_cross_s - avg_self_s) / avg_self_s * 100
                row += f" {avg_self_s:>12.4f} {avg_cross_s:>12.4f} {degrade_s:>11.1f}%"

            print(row)

    # === Decomposition ===
    print("\n" + "=" * 70)
    print("DRIFT DECOMPOSITION")
    print("=" * 70)

    for metric_key, metric_label in [("MAE", "All Nodes"), ("stable_MAE", "Stable Functional Only")]:
        baseline_cross, baseline_self = [], []
        pattern_cross, pattern_self = [], []
        instnorm_cross, instnorm_self = [], []

        for train_year in years:
            for test_year in years:
                for method, cross_list, self_list in [
                    ("baseline", baseline_cross, baseline_self),
                    ("pattern_only", pattern_cross, pattern_self),
                    ("instance_norm", instnorm_cross, instnorm_self),
                ]:
                    key = f"{method}_train_{train_year}_test_{test_year}"
                    if key in results and metric_key in results[key]:
                        val = results[key][metric_key]
                        if train_year == test_year:
                            self_list.append(val)
                        else:
                            cross_list.append(val)

        if not (baseline_cross and pattern_cross and instnorm_cross):
            continue

        bl_deg = np.mean(baseline_cross) - np.mean(baseline_self)
        po_deg = np.mean(pattern_cross) - np.mean(pattern_self)
        in_deg = np.mean(instnorm_cross) - np.mean(instnorm_self)

        additive_contrib = bl_deg - po_deg
        multiplicative_contrib = po_deg - in_deg
        pattern_contrib = in_deg

        total = bl_deg
        print(f"\n{metric_label}:")
        print(f"  Total degradation (baseline):     {bl_deg:>8.4f} MAE (100%)")
        print(f"  After additive centering:          {po_deg:>8.4f} MAE ({po_deg/total*100:>5.1f}%)")
        print(f"  After full scale removal:          {in_deg:>8.4f} MAE ({in_deg/total*100:>5.1f}%)")
        print(f"  ---")
        print(f"  Additive scale contribution:       {additive_contrib:>8.4f} MAE ({additive_contrib/total*100:>5.1f}%)")
        print(f"  Multiplicative scale contribution: {multiplicative_contrib:>8.4f} MAE ({multiplicative_contrib/total*100:>5.1f}%)")
        print(f"  Pure pattern drift:                {pattern_contrib:>8.4f} MAE ({pattern_contrib/total*100:>5.1f}%)")

    # Save
    output_path = "eda/concept_drift/cross_year_all_methods_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
