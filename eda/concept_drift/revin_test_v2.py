"""RevIN Test v2: Correct post-hoc RevIN as additive re-centering.

v1 failed because it tried to replace denormalization stats directly,
which doesn't work with globally-normalized models.

The correct approach mirrors the Target Oracle:
  Target Oracle: pred_corrected = pred - mean(pred_per_node) + mean(target_per_node)
  RevIN analog:  pred_corrected = pred - mean(pred_per_node) + mean(input_per_node)

This re-centers each node's prediction to match the input window's mean level,
which is the practical approximation of the oracle (since we know input but not target).

Additional variant: use input window's per-node mean directly as the anchor,
without subtracting prediction mean (simpler shift).

Variants:
1. Baseline: standard cross-year eval
2. RevIN re-center: pred - pred_node_mean + input_node_mean (analogous to oracle)
3. RevIN re-center + std: also rescale by input_std/pred_std
4. Target Oracle: pred - pred_node_mean + target_node_mean (upper bound)
5. Diagnostic: How much of oracle does RevIN recover?
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

            preds = get_baseline_predictions(model, test_x, train_mean, train_std)
            # preds: (N, 12, 893, 1) denormalized predictions

            # Input flow statistics (per sample, per node)
            input_flow = test_x[:, :, :, 0]  # (N, 12, 893)
            input_pn_mean = input_flow.mean(axis=1)  # (N, 893)
            input_pn_std = input_flow.std(axis=1)     # (N, 893)
            input_pn_std[input_pn_std == 0] = 1.0

            # Target flow statistics
            target_flow = test_y[:, :, :, 0]  # (N, 12, 893)
            target_pn_mean = target_flow.mean(axis=1)  # (N, 893)

            # Prediction statistics (per sample, per node)
            pred_flow = preds[:, :, :, 0]  # (N, 12, 893)
            pred_pn_mean = pred_flow.mean(axis=1)  # (N, 893)
            pred_pn_std = pred_flow.std(axis=1)     # (N, 893)
            pred_pn_std[pred_pn_std == 0] = 1.0

            # === 1. Baseline ===
            mae_baseline = compute_mae(preds, test_y)
            mae_baseline_func = compute_mae(preds[:, :, functional], test_y[:, :, functional])

            # === 2. RevIN re-center (additive only) ===
            # Same logic as Target Oracle, but using INPUT mean instead of TARGET mean
            # pred_corrected[i, t, n] = pred[i, t, n] - pred_mean[i, n] + input_mean[i, n]
            preds_revin_add = preds.copy()
            preds_revin_add[:, :, :, 0] = pred_flow - pred_pn_mean[:, None, :] + input_pn_mean[:, None, :]
            mae_revin_add = compute_mae(preds_revin_add, test_y)
            mae_revin_add_func = compute_mae(preds_revin_add[:, :, functional], test_y[:, :, functional])

            # === 3. RevIN re-center + rescale ===
            # Normalize predictions, then denormalize with input stats
            # z[i, t, n] = (pred[i, t, n] - pred_mean[i, n]) / pred_std[i, n]
            # pred_corrected[i, t, n] = z * input_std[i, n] + input_mean[i, n]
            z_norm = (pred_flow - pred_pn_mean[:, None, :]) / pred_pn_std[:, None, :]
            pred_revin_full_flow = z_norm * input_pn_std[:, None, :] + input_pn_mean[:, None, :]
            preds_revin_full = preds.copy()
            preds_revin_full[:, :, :, 0] = pred_revin_full_flow
            mae_revin_full = compute_mae(preds_revin_full, test_y)
            mae_revin_full_func = compute_mae(preds_revin_full[:, :, functional], test_y[:, :, functional])

            # === 4. Target Oracle (upper bound) ===
            preds_oracle = preds.copy()
            preds_oracle[:, :, :, 0] = pred_flow - pred_pn_mean[:, None, :] + target_pn_mean[:, None, :]
            mae_oracle = compute_mae(preds_oracle, test_y)
            mae_oracle_func = compute_mae(preds_oracle[:, :, functional], test_y[:, :, functional])

            # === Diagnostics ===
            # How well does input_mean predict target_mean? (per node, across samples)
            corrs = []
            for n in functional:
                c = np.corrcoef(input_pn_mean[:, n], target_pn_mean[:, n])[0, 1]
                if not np.isnan(c):
                    corrs.append(c)
            mean_corr = np.mean(corrs)

            # Mean absolute gap between input and target per-node means
            gap = np.abs(input_pn_mean[:, functional] - target_pn_mean[:, functional])
            mean_gap = float(np.mean(gap))

            # How well does input_mean predict pred_mean? (sanity check)
            corrs_pred_input = []
            for n in functional:
                c = np.corrcoef(input_pn_mean[:, n], pred_pn_mean[:, n])[0, 1]
                if not np.isnan(c):
                    corrs_pred_input.append(c)
            mean_corr_pred_input = np.mean(corrs_pred_input)

            is_self = train_year == test_year
            key = f"train_{train_year}_test_{test_year}"

            results[key] = {
                "baseline_MAE": mae_baseline,
                "baseline_func_MAE": mae_baseline_func,
                "revin_recenter_MAE": mae_revin_add,
                "revin_recenter_func_MAE": mae_revin_add_func,
                "revin_full_MAE": mae_revin_full,
                "revin_full_func_MAE": mae_revin_full_func,
                "target_oracle_MAE": mae_oracle,
                "target_oracle_func_MAE": mae_oracle_func,
                "input_target_corr": mean_corr,
                "input_target_gap": mean_gap,
                "pred_input_corr": mean_corr_pred_input,
            }

            def pct(corrected, baseline):
                return (baseline - corrected) / baseline * 100

            print(f"\n  Test on {test_year} Q1{' (SELF)' if is_self else ''}:")
            print(f"    {'Method':<35} {'All MAE':>10} {'Func MAE':>10} {'Improv%':>10}")
            print(f"    {'-'*65}")
            print(f"    {'Baseline':<35} {mae_baseline:>10.2f} {mae_baseline_func:>10.2f} {'---':>10}")
            print(f"    {'RevIN re-center (mean only)':<35} {mae_revin_add:>10.2f} {mae_revin_add_func:>10.2f} {pct(mae_revin_add, mae_baseline):>+10.1f}%")
            print(f"    {'RevIN full (mean + std)':<35} {mae_revin_full:>10.2f} {mae_revin_full_func:>10.2f} {pct(mae_revin_full, mae_baseline):>+10.1f}%")
            print(f"    {'Target Oracle (upper bound)':<35} {mae_oracle:>10.2f} {mae_oracle_func:>10.2f} {pct(mae_oracle, mae_baseline):>+10.1f}%")
            print(f"    Diagnostics: input→target corr={mean_corr:.4f}, gap={mean_gap:.2f}, "
                  f"pred→input corr={mean_corr_pred_input:.4f}")

            # Recovery analysis
            if not is_self:
                self_key = f"train_{train_year}_test_{train_year}"
                self_mae = results.get(self_key, {}).get("baseline_MAE", mae_baseline)
                degradation = mae_baseline - self_mae
                if degradation > 0:
                    revin_recovery = (mae_baseline - mae_revin_add) / degradation * 100
                    oracle_recovery = (mae_baseline - mae_oracle) / degradation * 100
                    print(f"    Degradation: {degradation:.2f}")
                    print(f"    RevIN recovers: {revin_recovery:+.1f}% of degradation")
                    print(f"    Oracle recovers: {oracle_recovery:+.1f}% of degradation")

        del model
        torch.cuda.empty_cache()

    # Save results
    output_path = "eda/concept_drift/revin_test_v2_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # ===== FINAL SUMMARY =====
    print("\n" + "=" * 90)
    print("FINAL SUMMARY")
    print("=" * 90)

    cross_pairs = [(ty, tey) for ty in years for tey in years if ty != tey]
    self_pairs = [(ty, ty) for ty in years]

    # --- Cross-year performance ---
    print(f"\n{'--- Cross-Year Pairs (Improvement over Baseline %) ---'}")
    print(f"{'Train→Test':<12} {'Base':>8} {'RevIN':>8} {'RevFull':>8} {'Oracle':>8} {'RevIN/Oracle':>12}")
    print("-" * 65)

    avg = {"revin": [], "full": [], "oracle": [], "recovery": []}
    for ty, tey in cross_pairs:
        key = f"train_{ty}_test_{tey}"
        r = results[key]
        b = r["baseline_MAE"]
        ir = (b - r["revin_recenter_MAE"]) / b * 100
        if_ = (b - r["revin_full_MAE"]) / b * 100
        io = (b - r["target_oracle_MAE"]) / b * 100
        recovery = ir / io * 100 if io > 0 else 0
        avg["revin"].append(ir); avg["full"].append(if_)
        avg["oracle"].append(io); avg["recovery"].append(recovery)
        print(f"{ty}→{tey}     {b:>8.2f} {ir:>+8.1f}% {if_:>+8.1f}% {io:>+8.1f}% {recovery:>10.1f}%")

    print("-" * 65)
    print(f"{'AVERAGE':<12} {'':>8} "
          f"{np.mean(avg['revin']):>+8.1f}% {np.mean(avg['full']):>+8.1f}% "
          f"{np.mean(avg['oracle']):>+8.1f}% {np.mean(avg['recovery']):>10.1f}%")

    # --- Self-year impact ---
    print(f"\n{'--- Self-Year Impact (should be ~0%) ---'}")
    for ty in years:
        key = f"train_{ty}_test_{ty}"
        r = results[key]
        b = r["baseline_MAE"]
        ir = (b - r["revin_recenter_MAE"]) / b * 100
        if_ = (b - r["revin_full_MAE"]) / b * 100
        io = (b - r["target_oracle_MAE"]) / b * 100
        print(f"  {ty}: Base={b:.2f}, RevIN={ir:+.1f}%, RevFull={if_:+.1f}%, Oracle={io:+.1f}%")

    # --- Functional-only cross-year ---
    print(f"\n{'--- Functional Nodes Only (Cross-Year) ---'}")
    print(f"{'Train→Test':<12} {'Base':>8} {'RevIN':>8} {'RevFull':>8} {'Oracle':>8} {'RevIN/Oracle':>12}")
    print("-" * 65)

    avg_f = {"revin": [], "full": [], "oracle": [], "recovery": []}
    for ty, tey in cross_pairs:
        key = f"train_{ty}_test_{tey}"
        r = results[key]
        b = r["baseline_func_MAE"]
        ir = (b - r["revin_recenter_func_MAE"]) / b * 100
        if_ = (b - r["revin_full_func_MAE"]) / b * 100
        io = (b - r["target_oracle_func_MAE"]) / b * 100
        recovery = ir / io * 100 if io > 0 else 0
        avg_f["revin"].append(ir); avg_f["full"].append(if_)
        avg_f["oracle"].append(io); avg_f["recovery"].append(recovery)
        print(f"{ty}→{tey}     {b:>8.2f} {ir:>+8.1f}% {if_:>+8.1f}% {io:>+8.1f}% {recovery:>10.1f}%")

    print("-" * 65)
    print(f"{'AVERAGE':<12} {'':>8} "
          f"{np.mean(avg_f['revin']):>+8.1f}% {np.mean(avg_f['full']):>+8.1f}% "
          f"{np.mean(avg_f['oracle']):>+8.1f}% {np.mean(avg_f['recovery']):>10.1f}%")

    # --- Degradation recovery analysis ---
    print(f"\n{'--- Degradation Recovery (how much of cross-year degradation is recovered) ---'}")
    for ty, tey in cross_pairs:
        key = f"train_{ty}_test_{tey}"
        self_key = f"train_{ty}_test_{ty}"
        r = results[key]
        s = results[self_key]
        degrad = r["baseline_MAE"] - s["baseline_MAE"]
        revin_recov = (r["baseline_MAE"] - r["revin_recenter_MAE"]) / degrad * 100 if degrad > 0 else 0
        oracle_recov = (r["baseline_MAE"] - r["target_oracle_MAE"]) / degrad * 100 if degrad > 0 else 0
        print(f"  {ty}→{tey}: degradation={degrad:.2f}, "
              f"RevIN recovers {revin_recov:+.1f}%, Oracle recovers {oracle_recov:+.1f}%")

    # Interpretation
    avg_revin = np.mean(avg['revin'])
    avg_oracle = np.mean(avg['oracle'])
    avg_recovery = np.mean(avg['recovery'])

    print(f"\n{'='*90}")
    print("INTERPRETATION")
    print(f"{'='*90}")
    print(f"\nRevIN re-center (cross-year): {avg_revin:+.1f}%")
    print(f"Target Oracle (cross-year):   {avg_oracle:+.1f}%")
    print(f"RevIN / Oracle recovery:      {avg_recovery:.1f}%")

    if avg_revin > 5 and avg_recovery > 30:
        print(f"\n→ RevIN IS a strong baseline! It recovers {avg_recovery:.0f}% of Oracle.")
        print(f"  SICPL must significantly outperform RevIN to justify its complexity.")
    elif avg_revin > 0:
        print(f"\n→ RevIN provides modest improvement ({avg_revin:+.1f}%), but only recovers")
        print(f"  {avg_recovery:.0f}% of Oracle. Significant room for SICPL to add value.")
    else:
        print(f"\n→ RevIN does NOT help or HURTS ({avg_revin:+.1f}%).")
        print(f"  Input window statistics cannot predict output scale.")
        print(f"  This SUPPORTS the need for external information (weather, metadata).")

    # Check WHY RevIN might fail: input_mean vs target_mean gap
    avg_corr = np.mean([results[f"train_{ty}_test_{tey}"]["input_target_corr"]
                       for ty, tey in cross_pairs])
    avg_gap = np.mean([results[f"train_{ty}_test_{tey}"]["input_target_gap"]
                      for ty, tey in cross_pairs])
    print(f"\nInput→Target mean correlation: {avg_corr:.4f}")
    print(f"Input→Target mean absolute gap: {avg_gap:.2f}")

    if avg_corr > 0.9 and avg_revin <= 0:
        print(f"\n→ PARADOX: High correlation ({avg_corr:.3f}) but RevIN doesn't help!")
        print(f"  Likely cause: model already incorporates input-level information well.")
        print(f"  The re-centering to input mean doesn't add new information.")
    elif avg_corr < 0.7:
        print(f"\n→ Weak input→target correlation. 12-step window too noisy for mean estimation.")


if __name__ == "__main__":
    main()
