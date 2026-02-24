"""RevIN Test: Can input-window statistics correct cross-year drift?

RevIN (Kim et al., ICLR 2022) key idea:
  1. Normalize input by per-instance stats: x_norm = (x - mean(x)) / std(x)
  2. Model predicts in normalized space: z = model(x_norm)
  3. Denormalize with SAME input stats: y = z * std(x) + mean(x)

Since our models are trained with global Z-score, we apply RevIN post-hoc:
  - baseline_pred = model_output * train_std + train_mean
  - z = (baseline_pred - train_mean) / train_std  (back to normalized space)
  - revin_pred = z * input_std + input_mean       (denorm with input stats)

Variants tested:
1. Baseline: standard cross-year eval (denorm with train year's global stats)
2. RevIN-additive: shift per-node by (input_mean - train_mean)
3. RevIN-full: rescale + re-center with per-node input stats
4. RevIN-global: rescale + re-center with per-sample global stats
5. Target Oracle (upper bound): re-center with target mean (from oracle_test_v2)
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
    desc = json.load(open(os.path.join(dataset_dir, "desc.json")))
    shape = desc["shape"]
    data = np.memmap(os.path.join(dataset_dir, "data.dat"), dtype="float32", mode="r").reshape(shape)

    n_total = shape[0]
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * 0.2)
    test_start = n_train + n_val
    test_data = data[test_start:]

    train_ch0 = data[:n_train, :, 0]
    global_mean = float(np.mean(train_ch0))
    global_std = float(np.std(train_ch0))

    return {
        "test_data": test_data,
        "global_mean": global_mean,
        "global_std": global_std,
    }


def create_samples(data, input_len, output_len):
    samples_x, samples_y = [], []
    total_len = input_len + output_len
    for i in range(len(data) - total_len + 1):
        samples_x.append(data[i:i+input_len])
        samples_y.append(data[i+input_len:i+total_len, :, 0:1])
    return np.array(samples_x), np.array(samples_y)


def get_baseline_predictions(model, test_x, train_mean, train_std, batch_size=64):
    """Get denormalized baseline predictions."""
    x_norm = test_x.copy()
    x_norm[:, :, :, 0] = (test_x[:, :, :, 0] - train_mean) / train_std

    all_preds = []
    with torch.no_grad():
        for i in range(0, len(x_norm), batch_size):
            bx = torch.FloatTensor(x_norm[i:i+batch_size]).to(DEVICE)
            out = model(bx, None, 0, 0, False)["prediction"]
            pred = out.cpu().numpy() * train_std + train_mean
            all_preds.append(pred)

    return np.concatenate(all_preds, axis=0)  # (N, 12, 893, 1)


def compute_mae(preds, targets):
    return float(np.mean(np.abs(preds - targets)))


def main():
    years = [2022, 2023, 2024]

    # Load sensor categories
    dataset_dir = "datasets/xtraffic/SAN_BERNARDINO"
    dead_indices = set(np.load(os.path.join(dataset_dir, "dead_indices.npy")).tolist())
    major_fail = set(np.load(os.path.join(dataset_dir, "major_fail_indices.npy")).tolist())
    functional = [i for i in range(893) if i not in dead_indices and i not in major_fail]
    print(f"Nodes: {len(functional)} functional, {len(dead_indices)} dead, {len(major_fail)} major_fail")

    # Preload all data
    data_cache = {}
    for year in years:
        stats = load_data_and_stats(DATASETS[year])
        test_x, test_y = create_samples(stats["test_data"], INPUT_LEN, OUTPUT_LEN)
        stats["test_x"] = test_x
        stats["test_y"] = test_y
        data_cache[year] = stats
        print(f"{year} Q1: {len(test_x)} samples, global_mean={stats['global_mean']:.2f}, "
              f"global_std={stats['global_std']:.2f}")

    results = {}

    for train_year in years:
        print(f"\n{'='*80}")
        print(f"Model trained on {train_year} Q1")
        print(f"{'='*80}")

        model = load_model(CHECKPOINTS[train_year])
        train_mean = data_cache[train_year]["global_mean"]
        train_std = data_cache[train_year]["global_std"]

        for test_year in years:
            test_stats = data_cache[test_year]
            test_x = test_stats["test_x"]  # (N, 12, 893, C)
            test_y = test_stats["test_y"]  # (N, 12, 893, 1)

            # Get baseline predictions (denormalized with train year's global stats)
            preds = get_baseline_predictions(model, test_x, train_mean, train_std)

            # Per-sample input flow statistics
            input_flow = test_x[:, :, :, 0]  # (N, 12, 893)
            input_pn_mean = input_flow.mean(axis=1)  # (N, 893) per-node mean over 12 steps
            input_pn_std = input_flow.std(axis=1)     # (N, 893) per-node std over 12 steps
            input_pn_std[input_pn_std == 0] = 1.0     # avoid div/0

            input_global_mean = input_flow.mean(axis=(1, 2))  # (N,) per-sample global mean
            input_global_std = input_flow.std(axis=(1, 2))    # (N,) per-sample global std
            input_global_std[input_global_std == 0] = 1.0

            # Per-sample target flow statistics (for oracle comparison)
            target_flow = test_y[:, :, :, 0]  # (N, 12, 893)
            target_pn_mean = target_flow.mean(axis=1)  # (N, 893)

            # === 1. Baseline ===
            mae_baseline = compute_mae(preds, test_y)
            mae_baseline_func = compute_mae(preds[:, :, functional], test_y[:, :, functional])

            # === 2. RevIN-additive (per-node): shift by (input_mean - train_mean) ===
            # Logic: predictions are centered around train_mean, re-center to input_mean
            shift = input_pn_mean - train_mean  # (N, 893)
            preds_add = preds + shift[:, None, :, None]  # (N, 12, 893, 1)
            mae_add = compute_mae(preds_add, test_y)
            mae_add_func = compute_mae(preds_add[:, :, functional], test_y[:, :, functional])

            # === 3. RevIN-full (per-node): rescale + re-center ===
            # Go back to normalized space, then denorm with input stats
            z = (preds - train_mean) / train_std  # (N, 12, 893, 1) normalized
            preds_full = z * input_pn_std[:, None, :, None] + input_pn_mean[:, None, :, None]
            mae_full = compute_mae(preds_full, test_y)
            mae_full_func = compute_mae(preds_full[:, :, functional], test_y[:, :, functional])

            # === 4. RevIN-global (per-sample): rescale + re-center with global stats ===
            preds_global = z * input_global_std[:, None, None, None] + input_global_mean[:, None, None, None]
            mae_global = compute_mae(preds_global, test_y)
            mae_global_func = compute_mae(preds_global[:, :, functional], test_y[:, :, functional])

            # === 5. Target Oracle: additive re-center with target mean (upper bound) ===
            pred_pn_mean = preds[:, :, :, 0].mean(axis=1)  # (N, 893)
            preds_oracle = preds + (target_pn_mean - pred_pn_mean)[:, None, :, None]
            mae_oracle = compute_mae(preds_oracle, test_y)
            mae_oracle_func = compute_mae(preds_oracle[:, :, functional], test_y[:, :, functional])

            # === 6. Diagnostic: How well does input_mean predict target_mean? ===
            # Per-node correlation between input and target mean
            corr_pn = []
            for n in functional:
                c = np.corrcoef(input_pn_mean[:, n], target_pn_mean[:, n])[0, 1]
                if not np.isnan(c):
                    corr_pn.append(c)
            mean_corr = np.mean(corr_pn)

            # Mean absolute gap between input_mean and target_mean
            gap = np.abs(input_pn_mean[:, functional] - target_pn_mean[:, functional])
            mean_gap = float(np.mean(gap))

            is_self = train_year == test_year
            key = f"train_{train_year}_test_{test_year}"

            results[key] = {
                "baseline_MAE": mae_baseline,
                "baseline_func_MAE": mae_baseline_func,
                "revin_additive_MAE": mae_add,
                "revin_additive_func_MAE": mae_add_func,
                "revin_full_MAE": mae_full,
                "revin_full_func_MAE": mae_full_func,
                "revin_global_MAE": mae_global,
                "revin_global_func_MAE": mae_global_func,
                "target_oracle_MAE": mae_oracle,
                "target_oracle_func_MAE": mae_oracle_func,
                "input_target_corr_functional": mean_corr,
                "input_target_gap_functional": mean_gap,
            }

            def pct(corrected, baseline):
                return (baseline - corrected) / baseline * 100

            print(f"\n  Test on {test_year} Q1{' (SELF)' if is_self else ''}:")
            print(f"    {'Method':<35} {'All MAE':>10} {'Func MAE':>10} {'Improv%':>10}")
            print(f"    {'-'*65}")
            print(f"    {'Baseline':<35} {mae_baseline:>10.2f} {mae_baseline_func:>10.2f} {'---':>10}")
            print(f"    {'RevIN-additive (per-node)':<35} {mae_add:>10.2f} {mae_add_func:>10.2f} {pct(mae_add, mae_baseline):>+10.1f}%")
            print(f"    {'RevIN-full (per-node)':<35} {mae_full:>10.2f} {mae_full_func:>10.2f} {pct(mae_full, mae_baseline):>+10.1f}%")
            print(f"    {'RevIN-global (per-sample)':<35} {mae_global:>10.2f} {mae_global_func:>10.2f} {pct(mae_global, mae_baseline):>+10.1f}%")
            print(f"    {'Target Oracle (upper bound)':<35} {mae_oracle:>10.2f} {mae_oracle_func:>10.2f} {pct(mae_oracle, mae_baseline):>+10.1f}%")
            print(f"    Input→Target corr (func): {mean_corr:.4f}, mean gap: {mean_gap:.2f}")

        del model
        torch.cuda.empty_cache()

    # Save results
    output_path = "eda/concept_drift/revin_test_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # ===== SUMMARY =====
    print("\n" + "=" * 90)
    print("SUMMARY: All Pairs (Improvement over Baseline %)")
    print("=" * 90)

    all_pairs = [(ty, tey) for ty in years for tey in years]
    cross_pairs = [(ty, tey) for ty in years for tey in years if ty != tey]

    print(f"\n{'Train→Test':<12} {'Base':>8} {'RevAdd':>8} {'RevFull':>8} {'RevGlob':>8} {'Oracle':>8} {'Corr':>7} {'Gap':>7}")
    print("-" * 80)

    avg = {"add": [], "full": [], "glob": [], "oracle": []}
    avg_self = {"add": [], "full": [], "glob": [], "oracle": []}

    for ty, tey in all_pairs:
        key = f"train_{ty}_test_{tey}"
        r = results[key]
        b = r["baseline_MAE"]
        ia = (b - r["revin_additive_MAE"]) / b * 100
        if_ = (b - r["revin_full_MAE"]) / b * 100
        ig = (b - r["revin_global_MAE"]) / b * 100
        io = (b - r["target_oracle_MAE"]) / b * 100

        is_self = ty == tey
        marker = " *" if is_self else ""

        print(f"{ty}→{tey}{marker:<4} {b:>8.2f} {ia:>+8.1f}% {if_:>+8.1f}% {ig:>+8.1f}% {io:>+8.1f}% "
              f"{r['input_target_corr_functional']:>7.3f} {r['input_target_gap_functional']:>7.1f}")

        if is_self:
            avg_self["add"].append(ia); avg_self["full"].append(if_)
            avg_self["glob"].append(ig); avg_self["oracle"].append(io)
        else:
            avg["add"].append(ia); avg["full"].append(if_)
            avg["glob"].append(ig); avg["oracle"].append(io)

    print("-" * 80)
    print(f"{'AVG cross':<12} {'':>8} "
          f"{np.mean(avg['add']):>+8.1f}% {np.mean(avg['full']):>+8.1f}% "
          f"{np.mean(avg['glob']):>+8.1f}% {np.mean(avg['oracle']):>+8.1f}%")
    print(f"{'AVG self':<12} {'':>8} "
          f"{np.mean(avg_self['add']):>+8.1f}% {np.mean(avg_self['full']):>+8.1f}% "
          f"{np.mean(avg_self['glob']):>+8.1f}% {np.mean(avg_self['oracle']):>+8.1f}%")
    print("* = self-evaluation")

    # Functional-only summary
    print(f"\n{'--- Functional Nodes Only ---'}")
    print(f"{'Train→Test':<12} {'Base':>8} {'RevAdd':>8} {'RevFull':>8} {'Oracle':>8}")
    print("-" * 55)

    avg_func = {"add": [], "full": [], "oracle": []}
    for ty, tey in cross_pairs:
        key = f"train_{ty}_test_{tey}"
        r = results[key]
        b = r["baseline_func_MAE"]
        ia = (b - r["revin_additive_func_MAE"]) / b * 100
        if_ = (b - r["revin_full_func_MAE"]) / b * 100
        io = (b - r["target_oracle_func_MAE"]) / b * 100
        avg_func["add"].append(ia); avg_func["full"].append(if_); avg_func["oracle"].append(io)
        print(f"{ty}→{tey}     {b:>8.2f} {ia:>+8.1f}% {if_:>+8.1f}% {io:>+8.1f}%")

    print("-" * 55)
    print(f"{'AVG cross':<12} {'':>8} "
          f"{np.mean(avg_func['add']):>+8.1f}% {np.mean(avg_func['full']):>+8.1f}% "
          f"{np.mean(avg_func['oracle']):>+8.1f}%")

    # Interpretation
    avg_add_cross = np.mean(avg['add'])
    avg_full_cross = np.mean(avg['full'])
    avg_oracle_cross = np.mean(avg['oracle'])

    print(f"\n{'='*90}")
    print("INTERPRETATION")
    print(f"{'='*90}")

    print(f"\n1. RevIN-additive (cross-year): {avg_add_cross:+.1f}%")
    print(f"   RevIN-full (cross-year): {avg_full_cross:+.1f}%")
    print(f"   Target Oracle (cross-year): {avg_oracle_cross:+.1f}%")

    if avg_add_cross > 5:
        recovery_pct = avg_add_cross / avg_oracle_cross * 100 if avg_oracle_cross > 0 else 0
        print(f"\n   → RevIN recovers {recovery_pct:.0f}% of Oracle's improvement.")
        print(f"   → RevIN IS a strong baseline. SICPL must outperform this.")
    elif avg_add_cross > 0:
        print(f"\n   → RevIN provides modest improvement.")
        print(f"   → Input statistics partially predict output scale, but not well enough.")
    else:
        print(f"\n   → RevIN HURTS performance!")
        print(f"   → Input scale ≠ output scale. RevIN assumption is violated.")
        print(f"   → This justifies more sophisticated approaches like SICPL.")

    # RevIN assumption check: input_mean ≈ target_mean?
    avg_corr = np.mean([results[f"train_{ty}_test_{tey}"]["input_target_corr_functional"]
                       for ty, tey in cross_pairs])
    avg_gap = np.mean([results[f"train_{ty}_test_{tey}"]["input_target_gap_functional"]
                      for ty, tey in cross_pairs])
    print(f"\n2. RevIN Assumption Check (input_mean ≈ target_mean?):")
    print(f"   Avg correlation (functional, cross-year): {avg_corr:.4f}")
    print(f"   Avg absolute gap (functional, cross-year): {avg_gap:.2f}")
    if avg_corr > 0.9:
        print(f"   → Strong correlation. RevIN assumption holds reasonably well.")
    elif avg_corr > 0.7:
        print(f"   → Moderate correlation. Some information in input stats.")
    else:
        print(f"   → Weak correlation. 12-step input window is too short for stable estimates.")


if __name__ == "__main__":
    main()
