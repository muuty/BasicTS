"""RevIN Diagnosis: Why does self-year RevIN destroy performance?

Hypothesis: The model uses global Z-score (single mean/std for all 893 nodes),
but RevIN re-centers per-node. Let's quantify:
1. How different are pred_pn_mean and input_pn_mean?
2. How different are input_pn_mean and target_pn_mean?
3. What's the model's own mean-prediction quality vs RevIN's input-mean estimate?
4. Does the 12-step window cause temporal-trend mismatch?
"""
import sys
import os
import json
import numpy as np
import torch
sys.path.append("/data/pretrainingbasicts")

from baselines.STAEformer.arch import STAEformer

MODEL_PARAM = {
    "num_nodes": 893, "in_steps": 12, "out_steps": 12, "steps_per_day": 288,
    "input_dim": 3, "output_dim": 1, "input_embedding_dim": 24,
    "tod_embedding_dim": 24, "dow_embedding_dim": 24, "spatial_embedding_dim": 0,
    "adaptive_embedding_dim": 24, "feed_forward_dim": 256, "num_heads": 4,
    "num_layers": 1, "dropout": 0.1, "use_mixed_proj": True,
}

CHECKPOINTS = {
    2022: "checkpoints/ConceptDrift_Q1/SAN_BERNARDINO_2022_Q1_30_12_12/6d33ff60f58f5fa9e14b8d42bfdda7a8/STAEformer_best_val_MAE.pt",
}

DATASETS = {
    2022: "datasets/SAN_BERNARDINO_2022_Q1",
}

DEVICE = "cuda:1"


def load_model(ckpt_path):
    model = STAEformer(**MODEL_PARAM)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE).eval()
    return model


def load_data(dataset_dir):
    desc = json.load(open(os.path.join(dataset_dir, "desc.json")))
    shape = desc["shape"]
    data = np.memmap(os.path.join(dataset_dir, "data.dat"), dtype="float32", mode="r").reshape(shape)
    n_total = shape[0]
    n_train = int(n_total * 0.6)
    n_val = int(n_total * 0.2)
    test_data = data[n_train + n_val:]
    train_ch0 = data[:n_train, :, 0]
    return test_data, float(np.mean(train_ch0)), float(np.std(train_ch0))


def create_samples(data, input_len=12, output_len=12):
    samples_x, samples_y = [], []
    for i in range(len(data) - input_len - output_len + 1):
        samples_x.append(data[i:i+input_len])
        samples_y.append(data[i+input_len:i+input_len+output_len, :, 0:1])
    return np.array(samples_x), np.array(samples_y)


def get_predictions(model, test_x, mean, std, batch_size=64):
    x_norm = test_x.copy()
    x_norm[:, :, :, 0] = (test_x[:, :, :, 0] - mean) / std
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(x_norm), batch_size):
            bx = torch.FloatTensor(x_norm[i:i+batch_size]).to(DEVICE)
            out = model(bx, None, 0, 0, False)["prediction"]
            pred = out.cpu().numpy() * std + mean
            all_preds.append(pred)
    return np.concatenate(all_preds, axis=0)


def main():
    # Load sensor categories
    dead_indices = set(np.load("datasets/xtraffic/SAN_BERNARDINO/dead_indices.npy").tolist())
    major_fail = set(np.load("datasets/xtraffic/SAN_BERNARDINO/major_fail_indices.npy").tolist())
    functional = [i for i in range(893) if i not in dead_indices and i not in major_fail]

    # Self-year case: train=2022, test=2022
    test_data, train_mean, train_std = load_data(DATASETS[2022])
    test_x, test_y = create_samples(test_data)
    model = load_model(CHECKPOINTS[2022])
    preds = get_predictions(model, test_x, train_mean, train_std)

    print(f"Train scaler: mean={train_mean:.2f}, std={train_std:.2f}")
    print(f"Samples: {len(test_x)}, functional nodes: {len(functional)}")

    # Per-sample, per-node means
    input_flow = test_x[:, :, :, 0]  # (N, 12, 893)
    target_flow = test_y[:, :, :, 0]  # (N, 12, 893)
    pred_flow = preds[:, :, :, 0]     # (N, 12, 893)

    input_pn_mean = input_flow.mean(axis=1)   # (N, 893)
    target_pn_mean = target_flow.mean(axis=1) # (N, 893)
    pred_pn_mean = pred_flow.mean(axis=1)     # (N, 893)

    # === Analysis 1: Quality of per-node mean estimates ===
    print("\n" + "="*70)
    print("1. PER-NODE MEAN ESTIMATE QUALITY (functional nodes only)")
    print("="*70)

    # How well does the model predict per-node mean?
    model_mean_error = np.abs(pred_pn_mean[:, functional] - target_pn_mean[:, functional])
    input_mean_error = np.abs(input_pn_mean[:, functional] - target_pn_mean[:, functional])
    shift_error = np.abs(input_pn_mean[:, functional] - pred_pn_mean[:, functional])

    print(f"\n  |pred_mean - target_mean| (model quality):   {model_mean_error.mean():.2f}")
    print(f"  |input_mean - target_mean| (RevIN estimate):  {input_mean_error.mean():.2f}")
    print(f"  |input_mean - pred_mean| (RevIN shift size):  {shift_error.mean():.2f}")

    print(f"\n  → Model's mean prediction is {input_mean_error.mean()/model_mean_error.mean():.1f}x "
          f"better than input mean estimate")
    print(f"  → RevIN replaces good estimate ({model_mean_error.mean():.2f} error) "
          f"with bad estimate ({input_mean_error.mean():.2f} error)")

    # === Analysis 2: Why is input_mean so different from target_mean? ===
    print("\n" + "="*70)
    print("2. WHY IS INPUT MEAN ≠ TARGET MEAN? (12-step = 1 hour window)")
    print("="*70)

    # Check if the gap is due to temporal trends
    # Input = t-12 to t-1, Target = t to t+11
    # If flow is trending up/down, input_mean ≠ target_mean

    # Compute per-sample flow trend (slope)
    # Use the full 24-step window (input + target) to estimate trend
    full_flow = np.concatenate([input_flow, target_flow], axis=1)  # (N, 24, 893)
    first_half_mean = full_flow[:, :12, :].mean(axis=1)   # input mean
    second_half_mean = full_flow[:, 12:, :].mean(axis=1)  # target mean
    trend = second_half_mean - first_half_mean  # (N, 893) positive = increasing

    print(f"\n  Mean trend (target_mean - input_mean) across all samples:")
    print(f"    Functional nodes: mean={trend[:, functional].mean():.3f}, "
          f"std={trend[:, functional].std():.2f}")
    print(f"    Absolute: {np.abs(trend[:, functional]).mean():.2f}")

    # Is the gap related to time-of-day?
    # ToD should be in channel 3 (normalized to 0-1)
    tod = test_x[:, 0, 0, -2]  # time-of-day of first input step
    if tod.max() <= 1:
        tod_hour = tod * 24  # convert to hours
    else:
        tod_hour = tod

    # Bin by hour and compute mean gap
    print(f"\n  Mean |input_mean - target_mean| by time of day (functional):")
    for h in [0, 6, 7, 8, 9, 12, 17, 18, 19, 22]:
        mask = (tod_hour >= h) & (tod_hour < h+1)
        if mask.sum() > 0:
            gap_h = np.abs(input_pn_mean[mask][:, functional] - target_pn_mean[mask][:, functional]).mean()
            trend_h = (target_pn_mean[mask][:, functional] - input_pn_mean[mask][:, functional]).mean()
            print(f"    Hour {h:02d}: gap={gap_h:.2f}, trend={trend_h:+.2f} "
                  f"(n={mask.sum()} samples)")

    # === Analysis 3: What if we use global input mean instead of per-node? ===
    print("\n" + "="*70)
    print("3. GLOBAL vs PER-NODE RevIN (functional nodes)")
    print("="*70)

    # Global input mean (same shift for all nodes) - matches our scaler's granularity
    input_global_mean = input_flow[:, :, functional].mean(axis=(1, 2))  # (N,)
    pred_global_mean = pred_flow[:, :, functional].mean(axis=(1, 2))   # (N,)
    target_global_mean = target_flow[:, :, functional].mean(axis=(1, 2)) # (N,)

    print(f"\n  Global means (averaged over samples):")
    print(f"    input:  {input_global_mean.mean():.2f}")
    print(f"    pred:   {pred_global_mean.mean():.2f}")
    print(f"    target: {target_global_mean.mean():.2f}")

    print(f"\n  Global mean estimation errors:")
    print(f"    |pred_global - target_global|:  {np.abs(pred_global_mean - target_global_mean).mean():.2f}")
    print(f"    |input_global - target_global|: {np.abs(input_global_mean - target_global_mean).mean():.2f}")

    # Test global RevIN (shift all nodes by same amount)
    global_shift = input_global_mean - pred_global_mean  # (N,)
    preds_global_revin = pred_flow.copy()
    preds_global_revin[:, :, functional] += global_shift[:, None, None]

    mae_baseline = np.abs(pred_flow[:, :, functional] - target_flow[:, :, functional]).mean()
    mae_global_revin = np.abs(preds_global_revin[:, :, functional] - target_flow[:, :, functional]).mean()

    # Test per-node RevIN
    preds_pernode_revin = pred_flow.copy()
    preds_pernode_revin = pred_flow - pred_pn_mean[:, None, :] + input_pn_mean[:, None, :]

    mae_pernode_revin = np.abs(preds_pernode_revin[:, :, functional] - target_flow[:, :, functional]).mean()

    # Target Oracle
    preds_oracle = pred_flow - pred_pn_mean[:, None, :] + target_pn_mean[:, None, :]
    mae_oracle = np.abs(preds_oracle[:, :, functional] - target_flow[:, :, functional]).mean()

    print(f"\n  MAE comparison (functional, self-year 2022):")
    print(f"    Baseline:            {mae_baseline:.4f}")
    print(f"    Global RevIN:        {mae_global_revin:.4f} ({(mae_baseline-mae_global_revin)/mae_baseline*100:+.1f}%)")
    print(f"    Per-Node RevIN:      {mae_pernode_revin:.4f} ({(mae_baseline-mae_pernode_revin)/mae_baseline*100:+.1f}%)")
    print(f"    Target Oracle:       {mae_oracle:.4f} ({(mae_baseline-mae_oracle)/mae_baseline*100:+.1f}%)")

    # === Analysis 4: Node-level breakdown ===
    print("\n" + "="*70)
    print("4. NODE-LEVEL BREAKDOWN: Where does RevIN help/hurt?")
    print("="*70)

    # Per-node MAE for each method
    baseline_pn_mae = np.abs(pred_flow - target_flow).mean(axis=(0, 1))  # (893,)
    revin_pn_mae = np.abs(preds_pernode_revin - target_flow).mean(axis=(0, 1))
    oracle_pn_mae = np.abs(preds_oracle - target_flow).mean(axis=(0, 1))

    # For functional nodes
    helped = 0
    hurt = 0
    for n in functional:
        if revin_pn_mae[n] < baseline_pn_mae[n]:
            helped += 1
        else:
            hurt += 1

    print(f"\n  Per-node RevIN effect (functional):")
    print(f"    Helped: {helped}/{len(functional)} ({helped/len(functional)*100:.1f}%)")
    print(f"    Hurt:   {hurt}/{len(functional)} ({hurt/len(functional)*100:.1f}%)")

    # Nodes where RevIN helps most vs hurts most
    revin_change = revin_pn_mae[functional] - baseline_pn_mae[functional]
    sorted_idx = np.argsort(revin_change)

    print(f"\n  Top 5 nodes where RevIN HELPS most:")
    for i in sorted_idx[:5]:
        n = functional[i]
        print(f"    Node {n}: baseline={baseline_pn_mae[n]:.2f} → RevIN={revin_pn_mae[n]:.2f} "
              f"({revin_change[i]:+.2f}), mean_flow={input_pn_mean[:, n].mean():.1f}")

    print(f"\n  Top 5 nodes where RevIN HURTS most:")
    for i in sorted_idx[-5:]:
        n = functional[i]
        print(f"    Node {n}: baseline={baseline_pn_mae[n]:.2f} → RevIN={revin_pn_mae[n]:.2f} "
              f"({revin_change[i]:+.2f}), mean_flow={input_pn_mean[:, n].mean():.1f}")

    # === Analysis 5: Decompose MAE into mean-error and pattern-error ===
    print("\n" + "="*70)
    print("5. MAE DECOMPOSITION: Mean Error vs Pattern Error")
    print("="*70)

    # MAE = |pred - target| can be decomposed:
    # pred[t] = pred_mean + pred_deviation[t]
    # target[t] = target_mean + target_deviation[t]
    # error[t] = (pred_mean - target_mean) + (pred_deviation[t] - target_deviation[t])

    # Mean error component (constant offset)
    mean_error = np.abs(pred_pn_mean[:, functional] - target_pn_mean[:, functional])
    mean_error_avg = mean_error.mean()

    # Total MAE
    total_mae = np.abs(pred_flow[:, :, functional] - target_flow[:, :, functional]).mean()

    # After perfect re-centering (pattern-only error)
    pattern_error = np.abs(preds_oracle[:, :, functional] - target_flow[:, :, functional]).mean()

    print(f"\n  Total MAE:     {total_mae:.4f}")
    print(f"  Mean error:    {mean_error_avg:.4f} (offset between pred and target per-node means)")
    print(f"  Pattern error: {pattern_error:.4f} (MAE after perfect re-centering)")
    print(f"  Sum check:     {mean_error_avg + pattern_error:.4f} (≈ total if errors align)")

    print(f"\n  For RevIN:")
    revin_mean_error = np.abs(input_pn_mean[:, functional] - target_pn_mean[:, functional]).mean()
    print(f"  RevIN mean error: {revin_mean_error:.4f} (gap between input and target means)")
    print(f"  RevIN total:      {mae_pernode_revin:.4f}")

    print(f"\n  Summary:")
    print(f"    Model's mean prediction error: {mean_error_avg:.2f}")
    print(f"    RevIN's mean estimate error:   {revin_mean_error:.2f}")
    print(f"    → RevIN's estimate is {revin_mean_error/mean_error_avg:.1f}x WORSE than model's")


if __name__ == "__main__":
    main()
