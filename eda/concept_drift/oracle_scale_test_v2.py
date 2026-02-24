"""Oracle Scale Test v2: Post-hoc correction approach.

v1 had a flaw: changing denormalization stats for a globally-normalized model
is semantically incorrect. Instead, v2 applies post-hoc corrections to
already-denormalized baseline predictions.

Oracle variants:
1. Baseline: standard cross-year eval (train year scaler)
2. Additive shift: pred + (test_mean - train_mean)  [global]
3. Multiplicative scale: pred * (test_mean / train_mean)  [global]
4. Per-node multiplicative: pred[n] * (test_node_mean[n] / train_node_mean[n])
5. Input-based RevIN: use input window's actual per-node mean to adjust
6. Per-sample target oracle: use target window's mean to re-center (upper bound)
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

    # Train split statistics
    train_ch0 = data[:n_train, :, 0]
    global_mean = float(np.mean(train_ch0))
    global_std = float(np.std(train_ch0))
    per_node_mean = np.mean(train_ch0, axis=0)  # (893,)
    per_node_mean_safe = per_node_mean.copy()
    per_node_mean_safe[per_node_mean_safe == 0] = 1.0  # avoid div by 0 for dead nodes

    return {
        "test_data": test_data,
        "global_mean": global_mean,
        "global_std": global_std,
        "per_node_mean": per_node_mean,
        "per_node_mean_safe": per_node_mean_safe,
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

    return np.concatenate(all_preds, axis=0)  # (N, 12, 893, 1) in real flow units


def compute_mae(preds, targets):
    return float(np.mean(np.abs(preds - targets)))


def compute_per_node_mae(preds, targets):
    """Return per-node MAE array."""
    return np.mean(np.abs(preds - targets), axis=(0, 1)).squeeze()  # (893,)


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
        print(f"  Per-node mean flow (functional): "
              f"mean={stats['per_node_mean'][functional].mean():.2f}, "
              f"median={np.median(stats['per_node_mean'][functional]):.2f}")

    results = {}

    for train_year in years:
        print(f"\n{'='*70}")
        print(f"Model trained on {train_year} Q1")
        print(f"{'='*70}")

        model = load_model(CHECKPOINTS[train_year])
        train_stats = data_cache[train_year]
        train_mean = train_stats["global_mean"]
        train_std = train_stats["global_std"]
        train_pn_mean = train_stats["per_node_mean"]
        train_pn_safe = train_stats["per_node_mean_safe"]

        for test_year in years:
            test_stats = data_cache[test_year]
            test_x = test_stats["test_x"]
            test_y = test_stats["test_y"]  # (N, 12, 893, 1)
            test_pn_mean = test_stats["per_node_mean"]

            # Get baseline predictions (already denormalized)
            preds = get_baseline_predictions(model, test_x, train_mean, train_std)

            # === 1. Baseline ===
            mae_baseline = compute_mae(preds, test_y)
            mae_baseline_func = compute_mae(preds[:, :, functional], test_y[:, :, functional])

            # === 2. Additive global shift ===
            test_global_mean = test_stats["global_mean"]
            shift = test_global_mean - train_mean
            preds_shift = preds + shift
            mae_shift = compute_mae(preds_shift, test_y)

            # === 3. Multiplicative global scale ===
            ratio = test_global_mean / train_mean
            preds_scale = preds * ratio
            mae_scale = compute_mae(preds_scale, test_y)

            # === 4. Per-node multiplicative (using train-split per-node means) ===
            # Correct the per-node scale: pred[n] * test_node_mean[n] / train_node_mean[n]
            pn_ratio = test_pn_mean / train_pn_safe  # (893,)
            preds_pernode = preds * pn_ratio[None, None, :, None]
            mae_pernode = compute_mae(preds_pernode, test_y)
            mae_pernode_func = compute_mae(preds_pernode[:, :, functional], test_y[:, :, functional])

            # === 5. Input-based RevIN ===
            # Use each sample's input flow mean (per-node) to re-center prediction
            # Logic: model predicted based on train-scale input, correct to actual input scale
            input_flow = test_x[:, :, :, 0]  # (N, 12, 893)
            input_pn_mean = input_flow.mean(axis=1)  # (N, 893) - actual input per-node mean
            train_expected_pn_mean = train_pn_mean[None, :]  # (1, 893) - expected per-node mean

            # Ratio: actual input scale / expected scale
            safe_expected = train_expected_pn_mean.copy()
            safe_expected[safe_expected == 0] = 1.0
            revin_ratio = input_pn_mean / safe_expected  # (N, 893)
            preds_revin = preds * revin_ratio[:, None, :, None]
            mae_revin = compute_mae(preds_revin, test_y)
            mae_revin_func = compute_mae(preds_revin[:, :, functional], test_y[:, :, functional])

            # === 6. Per-sample target oracle (upper bound) ===
            # Use target's per-node mean to re-center: "if we knew the future mean flow"
            target_pn_mean = test_y.mean(axis=1)  # (N, 893, 1)
            # Re-center: shift prediction so its per-node mean matches target per-node mean
            pred_pn_mean = preds.mean(axis=1)  # (N, 893, 1)
            preds_target_oracle = preds - pred_pn_mean[:, None, :, :] + target_pn_mean[:, None, :, :]
            mae_target_oracle = compute_mae(preds_target_oracle, test_y)
            mae_target_func = compute_mae(preds_target_oracle[:, :, functional], test_y[:, :, functional])

            is_self = train_year == test_year
            key = f"train_{train_year}_test_{test_year}"

            results[key] = {
                "baseline_MAE": mae_baseline,
                "baseline_func_MAE": mae_baseline_func,
                "additive_shift_MAE": mae_shift,
                "multiplicative_scale_MAE": mae_scale,
                "pernode_scale_MAE": mae_pernode,
                "pernode_func_MAE": mae_pernode_func,
                "revin_MAE": mae_revin,
                "revin_func_MAE": mae_revin_func,
                "target_oracle_MAE": mae_target_oracle,
                "target_oracle_func_MAE": mae_target_func,
            }

            def pct(corrected, baseline):
                return (baseline - corrected) / baseline * 100

            print(f"\n  Test on {test_year} Q1{' (SELF)' if is_self else ''}:")
            print(f"    {'Method':<30} {'All MAE':>10} {'Func MAE':>10} {'Improv%':>10}")
            print(f"    {'-'*60}")
            print(f"    {'Baseline':<30} {mae_baseline:>10.2f} {mae_baseline_func:>10.2f} {'---':>10}")
            print(f"    {'+ Additive shift':<30} {mae_shift:>10.2f} {'':>10} {pct(mae_shift, mae_baseline):>+10.1f}%")
            print(f"    {'+ Multiplicative scale':<30} {mae_scale:>10.2f} {'':>10} {pct(mae_scale, mae_baseline):>+10.1f}%")
            print(f"    {'+ Per-node scale':<30} {mae_pernode:>10.2f} {mae_pernode_func:>10.2f} {pct(mae_pernode, mae_baseline):>+10.1f}%")
            print(f"    {'+ Input-based RevIN':<30} {mae_revin:>10.2f} {mae_revin_func:>10.2f} {pct(mae_revin, mae_baseline):>+10.1f}%")
            print(f"    {'+ Target oracle (upper bound)':<30} {mae_target_oracle:>10.2f} {mae_target_func:>10.2f} {pct(mae_target_oracle, mae_baseline):>+10.1f}%")

        del model
        torch.cuda.empty_cache()

    # Save
    output_path = "eda/concept_drift/oracle_scale_test_v2_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary table (cross-year pairs only)
    print("\n" + "=" * 80)
    print("SUMMARY: Cross-Year Pairs Only (Improvement over Baseline %)")
    print("=" * 80)
    cross_pairs = [(ty, tey) for ty in years for tey in years if ty != tey]

    print(f"\n{'Train→Test':<12} {'Base MAE':>10} {'Additive':>10} {'Multip':>10} {'PerNode':>10} {'RevIN':>10} {'Oracle':>10}")
    print("-" * 75)

    avg = {"add": [], "mult": [], "pn": [], "revin": [], "oracle": []}
    for ty, tey in cross_pairs:
        key = f"train_{ty}_test_{tey}"
        r = results[key]
        b = r["baseline_MAE"]
        ia = pct(r["additive_shift_MAE"], b)
        im = pct(r["multiplicative_scale_MAE"], b)
        ip = pct(r["pernode_scale_MAE"], b)
        ir = pct(r["revin_MAE"], b)
        io = pct(r["target_oracle_MAE"], b)
        avg["add"].append(ia); avg["mult"].append(im); avg["pn"].append(ip)
        avg["revin"].append(ir); avg["oracle"].append(io)
        print(f"{ty}→{tey}     {b:>10.2f} {ia:>+10.1f}% {im:>+10.1f}% {ip:>+10.1f}% {ir:>+10.1f}% {io:>+10.1f}%")

    print("-" * 75)
    print(f"{'AVERAGE':<12} {'':>10} "
          f"{np.mean(avg['add']):>+10.1f}% "
          f"{np.mean(avg['mult']):>+10.1f}% "
          f"{np.mean(avg['pn']):>+10.1f}% "
          f"{np.mean(avg['revin']):>+10.1f}% "
          f"{np.mean(avg['oracle']):>+10.1f}%")

    # Self-year impact
    print(f"\n{'--- Self-Year Impact (should be ~0% or slightly negative) ---'}")
    for year in years:
        key = f"train_{year}_test_{year}"
        r = results[key]
        b = r["baseline_MAE"]
        print(f"  {year}: Base={b:.2f}, "
              f"PerNode={pct(r['pernode_scale_MAE'], b):+.1f}%, "
              f"RevIN={pct(r['revin_MAE'], b):+.1f}%, "
              f"Oracle={pct(r['target_oracle_MAE'], b):+.1f}%")

    # Interpretation
    avg_pn = np.mean(avg['pn'])
    avg_revin = np.mean(avg['revin'])
    avg_oracle = np.mean(avg['oracle'])

    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print(f"{'='*80}")

    print(f"\n1. Global corrections (Additive/Multiplicative): {np.mean(avg['add']):+.1f}% / {np.mean(avg['mult']):+.1f}%")
    print(f"   → Global mean/std barely differ across years → global scale correction is irrelevant")

    print(f"\n2. Per-Node scale correction: {avg_pn:+.1f}%")
    if avg_pn > 10:
        print(f"   → Node-level scale changes ARE significant. Per-node scale estimation is valuable.")
    elif avg_pn > 0:
        print(f"   → Moderate per-node scale effect. Some nodes shift more than others.")
    else:
        print(f"   → Per-node scale correction doesn't help or hurts.")

    print(f"\n3. Input-based RevIN: {avg_revin:+.1f}%")
    if avg_revin > 10:
        print(f"   → Input-window statistics can correct scale at test time.")
        print(f"   → RevIN-like approach is promising for drift robustness.")
    elif avg_revin > 0:
        print(f"   → Modest improvement. Input statistics provide some scale signal.")
    else:
        print(f"   → Input-based correction doesn't help. Scale at input ≠ scale at output.")

    print(f"\n4. Target oracle (upper bound): {avg_oracle:+.1f}%")
    if avg_oracle > 30:
        print(f"   → Knowing the true future mean could recover >{avg_oracle:.0f}% of degradation.")
        print(f"   → Scale prediction IS the key challenge. SICPL approach justified.")
    elif avg_oracle > 10:
        print(f"   → Moderate upper bound. Scale matters but isn't everything.")
    else:
        print(f"   → Even perfect scale knowledge helps little. Drift is NOT mainly scale.")


if __name__ == "__main__":
    main()
