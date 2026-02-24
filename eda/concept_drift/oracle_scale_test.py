"""Oracle Scale Test: Is scale mismatch the main cause of cross-year degradation?

For each (train_year, test_year) pair, compare:
1. Baseline: denormalize with train year's mean/std (standard cross-year eval)
2. Global Oracle: denormalize with test year's global mean/std
3. Per-Node Oracle: denormalize with test year's per-node mean/std
4. Per-Sample Oracle: denormalize with each sample's target mean/std (perfect RevIN)

If Oracle MAE << Baseline MAE, scale mismatch is the dominant cause of drift.
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

CHECKPOINTS = {
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


def load_data_and_stats(dataset_dir):
    """Load dataset and compute various statistics."""
    desc = json.load(open(os.path.join(dataset_dir, "desc.json")))
    shape = desc["shape"]
    data = np.memmap(os.path.join(dataset_dir, "data.dat"), dtype="float32", mode="r").reshape(shape)

    n_total = shape[0]
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * 0.2)
    test_start = n_train + n_val
    test_data = data[test_start:]

    # Global mean/std from train split (channel 0 = flow)
    train_ch0 = data[:n_train, :, 0]
    global_mean = float(np.mean(train_ch0))
    global_std = float(np.std(train_ch0))

    # Per-node mean/std from train split
    per_node_mean = np.mean(train_ch0, axis=0)  # (893,)
    per_node_std = np.std(train_ch0, axis=0)     # (893,)
    per_node_std[per_node_std == 0] = 1.0

    # Test split statistics (for oracle)
    test_ch0 = test_data[:, :, 0]
    test_global_mean = float(np.mean(test_ch0))
    test_global_std = float(np.std(test_ch0))
    test_per_node_mean = np.mean(test_ch0, axis=0)  # (893,)
    test_per_node_std = np.std(test_ch0, axis=0)     # (893,)
    test_per_node_std[test_per_node_std == 0] = 1.0

    return {
        "test_data": test_data,
        "global_mean": global_mean,
        "global_std": global_std,
        "per_node_mean": per_node_mean,
        "per_node_std": per_node_std,
        "test_global_mean": test_global_mean,
        "test_global_std": test_global_std,
        "test_per_node_mean": test_per_node_mean,
        "test_per_node_std": test_per_node_std,
    }


def create_samples(data, input_len, output_len):
    samples_x, samples_y = [], []
    total_len = input_len + output_len
    for i in range(len(data) - total_len + 1):
        samples_x.append(data[i:i+input_len])
        samples_y.append(data[i+input_len:i+total_len, :, 0:1])  # flow only
    return np.array(samples_x), np.array(samples_y)


def get_normalized_predictions(model, test_x, train_mean, train_std, batch_size=64):
    """Run model and return raw normalized predictions."""
    x_norm = test_x.copy()
    x_norm[:, :, :, 0] = (test_x[:, :, :, 0] - train_mean) / train_std

    all_preds_norm = []
    with torch.no_grad():
        for i in range(0, len(x_norm), batch_size):
            bx = torch.FloatTensor(x_norm[i:i+batch_size]).to(DEVICE)
            out = model(bx, None, 0, 0, False)["prediction"]
            all_preds_norm.append(out.cpu().numpy())

    return np.concatenate(all_preds_norm, axis=0)  # (N, 12, 893, 1) in normalized space


def compute_mae(preds, targets):
    return float(np.mean(np.abs(preds - targets)))


def main():
    years = [2022, 2023, 2024]

    # Preload all data
    data_cache = {}
    for year in years:
        stats = load_data_and_stats(DATASETS[year])
        test_x, test_y = create_samples(stats["test_data"], INPUT_LEN, OUTPUT_LEN)
        stats["test_x"] = test_x
        stats["test_y"] = test_y
        data_cache[year] = stats
        print(f"{year} Q1: {len(test_x)} samples, "
              f"train_mean={stats['global_mean']:.2f}, test_mean={stats['test_global_mean']:.2f}, "
              f"train_std={stats['global_std']:.2f}, test_std={stats['test_global_std']:.2f}")

    results = {}

    for train_year in years:
        print(f"\n{'='*70}")
        print(f"Model trained on {train_year} Q1")
        print(f"{'='*70}")

        model = load_model(CHECKPOINTS[train_year])
        train_stats = data_cache[train_year]
        train_mean = train_stats["global_mean"]
        train_std = train_stats["global_std"]

        for test_year in years:
            test_stats = data_cache[test_year]
            test_x = test_stats["test_x"]
            test_y = test_stats["test_y"]  # (N, 12, 893, 1)

            # Get normalized predictions
            preds_norm = get_normalized_predictions(model, test_x, train_mean, train_std)

            # === 1. Baseline: denormalize with train year's global stats ===
            preds_baseline = preds_norm * train_std + train_mean
            mae_baseline = compute_mae(preds_baseline, test_y)

            # === 2. Global Oracle: denormalize with test year's global stats ===
            test_mean = test_stats["test_global_mean"]
            test_std_val = test_stats["test_global_std"]
            preds_global_oracle = preds_norm * test_std_val + test_mean
            mae_global_oracle = compute_mae(preds_global_oracle, test_y)

            # === 3. Per-Node Oracle: denormalize with test year's per-node stats ===
            # preds_norm: (N, 12, 893, 1), per_node_mean: (893,)
            pn_mean = test_stats["test_per_node_mean"][None, None, :, None]  # (1, 1, 893, 1)
            pn_std = test_stats["test_per_node_std"][None, None, :, None]
            preds_pernode_oracle = preds_norm * pn_std + pn_mean
            mae_pernode_oracle = compute_mae(preds_pernode_oracle, test_y)

            # === 4. Per-Sample Oracle: denormalize with each sample's target stats ===
            # For each sample, use the target's mean/std as the "perfect future knowledge"
            sample_mean = test_y.mean(axis=1, keepdims=True)  # (N, 1, 893, 1)
            sample_std = test_y.std(axis=1, keepdims=True)    # (N, 1, 893, 1)
            sample_std[sample_std == 0] = 1.0
            preds_sample_oracle = preds_norm * sample_std + sample_mean
            mae_sample_oracle = compute_mae(preds_sample_oracle, test_y)

            # === 5. Simple Rescale: baseline * (test_mean / train_mean) ===
            if train_mean != 0:
                scale_ratio = test_mean / train_mean
                preds_rescaled = preds_baseline * scale_ratio
                mae_rescaled = compute_mae(preds_rescaled, test_y)
            else:
                mae_rescaled = float('nan')

            key = f"train_{train_year}_test_{test_year}"
            is_self = train_year == test_year
            results[key] = {
                "baseline_MAE": mae_baseline,
                "global_oracle_MAE": mae_global_oracle,
                "pernode_oracle_MAE": mae_pernode_oracle,
                "persample_oracle_MAE": mae_sample_oracle,
                "rescale_MAE": mae_rescaled,
            }

            # Improvement percentages
            if not is_self:
                self_mae = results.get(f"train_{train_year}_test_{train_year}", {}).get("baseline_MAE", mae_baseline)
                degrad_baseline = (mae_baseline - self_mae) / self_mae * 100
                degrad_global = (mae_global_oracle - self_mae) / self_mae * 100
                degrad_pernode = (mae_pernode_oracle - self_mae) / self_mae * 100
                degrad_sample = (mae_sample_oracle - self_mae) / self_mae * 100
                improvement_global = (mae_baseline - mae_global_oracle) / mae_baseline * 100
                improvement_pernode = (mae_baseline - mae_pernode_oracle) / mae_baseline * 100
                improvement_sample = (mae_baseline - mae_sample_oracle) / mae_baseline * 100
                improvement_rescale = (mae_baseline - mae_rescaled) / mae_baseline * 100

                results[key].update({
                    "degrad_baseline_pct": degrad_baseline,
                    "degrad_global_oracle_pct": degrad_global,
                    "improvement_global_pct": improvement_global,
                    "improvement_pernode_pct": improvement_pernode,
                    "improvement_sample_pct": improvement_sample,
                    "improvement_rescale_pct": improvement_rescale,
                })

            marker = " (SELF)" if is_self else ""
            print(f"\n  Test on {test_year} Q1{marker}:")
            print(f"    Baseline (train stats):    MAE = {mae_baseline:.4f}")
            print(f"    Global Oracle (test stats): MAE = {mae_global_oracle:.4f}"
                  f"  ({(mae_baseline-mae_global_oracle)/mae_baseline*100:+.1f}%)")
            print(f"    Per-Node Oracle:           MAE = {mae_pernode_oracle:.4f}"
                  f"  ({(mae_baseline-mae_pernode_oracle)/mae_baseline*100:+.1f}%)")
            print(f"    Per-Sample Oracle:         MAE = {mae_sample_oracle:.4f}"
                  f"  ({(mae_baseline-mae_sample_oracle)/mae_baseline*100:+.1f}%)")
            print(f"    Simple Rescale:            MAE = {mae_rescaled:.4f}"
                  f"  ({(mae_baseline-mae_rescaled)/mae_baseline*100:+.1f}%)")

        del model
        torch.cuda.empty_cache()

    # Save results
    output_path = "eda/concept_drift/oracle_scale_test_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("ORACLE SCALE TEST SUMMARY")
    print("=" * 80)
    print("\nQuestion: Is scale mismatch the main cause of cross-year MAE degradation?")
    print()

    # Cross-year pairs only
    cross_pairs = [(ty, tey) for ty in years for tey in years if ty != tey]

    print(f"{'Train→Test':<15} {'Baseline':>10} {'Global':>10} {'PerNode':>10} {'PerSample':>10} {'Rescale':>10}")
    print("-" * 70)
    for train_year, test_year in cross_pairs:
        key = f"train_{train_year}_test_{test_year}"
        r = results[key]
        print(f"{train_year}→{test_year}      "
              f"{r['baseline_MAE']:>10.2f}"
              f"{r['global_oracle_MAE']:>10.2f}"
              f"{r['pernode_oracle_MAE']:>10.2f}"
              f"{r['persample_oracle_MAE']:>10.2f}"
              f"{r['rescale_MAE']:>10.2f}")

    print("\n--- Improvement over Baseline (%) ---")
    print(f"{'Train→Test':<15} {'Global':>10} {'PerNode':>10} {'PerSample':>10} {'Rescale':>10}")
    print("-" * 55)
    improvements = {"global": [], "pernode": [], "sample": [], "rescale": []}
    for train_year, test_year in cross_pairs:
        key = f"train_{train_year}_test_{test_year}"
        r = results[key]
        ig = r.get("improvement_global_pct", 0)
        ip = r.get("improvement_pernode_pct", 0)
        is_ = r.get("improvement_sample_pct", 0)
        ir = r.get("improvement_rescale_pct", 0)
        improvements["global"].append(ig)
        improvements["pernode"].append(ip)
        improvements["sample"].append(is_)
        improvements["rescale"].append(ir)
        print(f"{train_year}→{test_year}      {ig:>+10.1f}%{ip:>+10.1f}%{is_:>+10.1f}%{ir:>+10.1f}%")

    print("-" * 55)
    print(f"{'AVERAGE':<15}"
          f"{np.mean(improvements['global']):>+10.1f}%"
          f"{np.mean(improvements['pernode']):>+10.1f}%"
          f"{np.mean(improvements['sample']):>+10.1f}%"
          f"{np.mean(improvements['rescale']):>+10.1f}%")

    # Interpretation
    avg_global = np.mean(improvements["global"])
    avg_pernode = np.mean(improvements["pernode"])

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    if avg_global > 50:
        print(">>> Scale is the DOMINANT cause of drift. Global scale correction alone")
        print("    recovers >50% of degradation. SICPL approach is strongly justified.")
    elif avg_global > 20:
        print(">>> Scale is a SIGNIFICANT cause of drift. Global scale correction")
        print("    recovers 20-50%. Hybrid approach (SPD + pattern) recommended.")
    elif avg_global > 10:
        print(">>> Scale contributes MODERATELY to drift. Other factors also important.")
        print("    Scale correction helps but is not sufficient alone.")
    else:
        print(">>> Scale is NOT the main cause of drift (<10% improvement).")
        print("    Look for other sources: structural pattern changes, sensor issues, etc.")

    if avg_pernode > avg_global * 1.5:
        print(f"\n>>> Per-Node oracle ({avg_pernode:.1f}%) >> Global oracle ({avg_global:.1f}%):")
        print("    Node-level scale variation matters. Per-node scale estimation is valuable.")
    else:
        print(f"\n>>> Per-Node oracle ({avg_pernode:.1f}%) ~ Global oracle ({avg_global:.1f}%):")
        print("    Global scale correction is sufficient. Per-node adds limited value.")


if __name__ == "__main__":
    main()
