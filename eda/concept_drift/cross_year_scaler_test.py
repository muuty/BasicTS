"""Scaler mismatch vs true drift: Is the 56% "scale drift" just wrong scaler?

Comparison:
1. Baseline (train scaler): model_A + scaler_A → test_B  (current protocol)
2. Corrected scaler:        model_A + scaler_B → test_B  (fix scaler only)
3. Instance norm (train scaler): instnorm_model_A + scaler_A → test_B

If (2) ≈ (3): the 56% improvement is mostly scaler mismatch (trivially fixable)
If (2) >> (3): instance norm captures real per-sample scale dynamics beyond global scaler
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


def find_checkpoints(prefix):
    ckpts = {}
    for year in [2022, 2023, 2024]:
        base_dir = f"checkpoints/{prefix}/SAN_BERNARDINO_{year}_Q1_30_12_12"
        if not os.path.isdir(base_dir):
            continue
        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if not subdirs:
            continue
        ckpt_path = os.path.join(base_dir, subdirs[0], "STAEformer_best_val_MAE.pt")
        if os.path.exists(ckpt_path):
            ckpts[year] = ckpt_path
    return ckpts


def load_model(ckpt_path):
    model = STAEformer(**MODEL_PARAM)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(DEVICE).eval()


def load_data_and_scaler(dataset_dir):
    desc = json.load(open(os.path.join(dataset_dir, "desc.json")))
    shape = desc["shape"]
    data = np.memmap(os.path.join(dataset_dir, "data.dat"), dtype="float32", mode="r").reshape(shape)
    n_total = shape[0]
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * 0.2)
    test_data = data[n_train + n_val:]
    train_ch0 = data[:n_train, :, 0]
    mean, std = float(np.mean(train_ch0)), float(np.std(train_ch0))
    return test_data, mean, std


def create_samples(data):
    sx, sy = [], []
    for i in range(len(data) - INPUT_LEN - OUTPUT_LEN + 1):
        sx.append(data[i:i+INPUT_LEN])
        sy.append(data[i+INPUT_LEN:i+INPUT_LEN+OUTPUT_LEN, :, 0:1])
    return np.array(sx), np.array(sy)


def evaluate(model, test_x, test_y, norm_mean, norm_std, denorm_mean, denorm_std,
             method="baseline", batch_size=64):
    """Evaluate with potentially different norm/denorm scalers.

    Args:
        norm_mean/std: used to Z-score normalize the INPUT
        denorm_mean/std: used to reverse Z-score the OUTPUT
        method: "baseline" or "instance_norm"
    """
    test_x_norm = test_x.copy()
    test_x_norm[:, :, :, 0] = (test_x[:, :, :, 0] - norm_mean) / norm_std

    all_preds = []
    with torch.no_grad():
        for i in range(0, len(test_x_norm), batch_size):
            bx = torch.FloatTensor(test_x_norm[i:i+batch_size]).to(DEVICE)

            if method == "baseline":
                out = model(bx, None, 0, 0, False)["prediction"]

            elif method == "instance_norm":
                flow = bx[:, :, :, 0]
                fm = flow.mean(dim=1, keepdim=True)
                fs = flow.std(dim=1, keepdim=True) + 1e-5
                bx_n = bx.clone()
                bx_n[:, :, :, 0] = (flow - fm) / fs
                out = model(bx_n, None, 0, 0, False)["prediction"]
                out = out * fs.unsqueeze(-1) + fm.unsqueeze(-1)

            pred = out.cpu().numpy() * denorm_std + denorm_mean
            all_preds.append(pred)

    preds = np.concatenate(all_preds, axis=0)
    per_node_mae = np.mean(np.abs(preds - test_y), axis=(0, 1, 3))  # (N,)
    return float(per_node_mae.mean()), per_node_mae


def main():
    years = [2022, 2023, 2024]

    # Load data
    cache = {}
    for year in years:
        td, m, s = load_data_and_scaler(DATASETS[year])
        tx, ty = create_samples(td)
        cache[year] = {"x": tx, "y": ty, "mean": m, "std": s}
        print(f"{year} Q1: scaler mean={m:.2f}, std={s:.2f}, samples={len(tx)}")

    # Stable functional indices
    stable_idx = np.load("datasets/SAN_BERNARDINO_2022_Q1/stable_functional_indices.npy")
    print(f"Stable functional nodes: {len(stable_idx)}")

    # Instance norm checkpoints
    instnorm_ckpts = find_checkpoints("ConceptDrift_InstanceNorm")

    # Cross-year pairs
    cross_pairs = [(2022, 2023), (2022, 2024), (2023, 2022), (2023, 2024), (2024, 2022), (2024, 2023)]

    results = {}

    print("\n" + "=" * 80)
    print("SCALER MISMATCH TEST")
    print("=" * 80)

    for train_year, test_year in cross_pairs:
        c_test = cache[test_year]
        c_train = cache[train_year]

        print(f"\n--- {train_year} → {test_year} ---")
        print(f"  Train scaler: mean={c_train['mean']:.2f}, std={c_train['std']:.2f}")
        print(f"  Test scaler:  mean={c_test['mean']:.2f}, std={c_test['std']:.2f}")

        # Load baseline model (trained on train_year)
        model = load_model(BASELINE_CHECKPOINTS[train_year])

        # (A) Baseline: train scaler for both norm and denorm
        mae_a, pn_a = evaluate(model, c_test["x"], c_test["y"],
                               c_train["mean"], c_train["std"],
                               c_train["mean"], c_train["std"],
                               method="baseline")

        # (B) Corrected scaler: test scaler for both norm and denorm
        mae_b, pn_b = evaluate(model, c_test["x"], c_test["y"],
                               c_test["mean"], c_test["std"],
                               c_test["mean"], c_test["std"],
                               method="baseline")

        del model; torch.cuda.empty_cache()

        # (C) Instance norm model: train scaler (as originally evaluated)
        if train_year in instnorm_ckpts:
            model_in = load_model(instnorm_ckpts[train_year])
            mae_c, pn_c = evaluate(model_in, c_test["x"], c_test["y"],
                                   c_train["mean"], c_train["std"],
                                   c_train["mean"], c_train["std"],
                                   method="instance_norm")
            del model_in; torch.cuda.empty_cache()
        else:
            mae_c, pn_c = None, None

        # Stable MAE
        stable_a = float(pn_a[stable_idx].mean())
        stable_b = float(pn_b[stable_idx].mean())
        stable_c = float(pn_c[stable_idx].mean()) if pn_c is not None else None

        print(f"  (A) Train scaler:     MAE={mae_a:.4f}, stable={stable_a:.4f}")
        print(f"  (B) Corrected scaler: MAE={mae_b:.4f}, stable={stable_b:.4f}")
        if mae_c is not None:
            print(f"  (C) Instance norm:    MAE={mae_c:.4f}, stable={stable_c:.4f}")

        key = f"{train_year}_{test_year}"
        results[key] = {
            "train_scaler": {"MAE": mae_a, "stable_MAE": stable_a},
            "corrected_scaler": {"MAE": mae_b, "stable_MAE": stable_b},
        }
        if mae_c is not None:
            results[key]["instance_norm"] = {"MAE": mae_c, "stable_MAE": stable_c}

    # Self-year baselines (for degradation calculation)
    print("\n--- Self-year baselines ---")
    self_results = {}
    for year in years:
        c = cache[year]
        model = load_model(BASELINE_CHECKPOINTS[year])
        mae_self, pn_self = evaluate(model, c["x"], c["y"],
                                     c["mean"], c["std"], c["mean"], c["std"],
                                     method="baseline")
        stable_self = float(pn_self[stable_idx].mean())
        self_results[year] = {"MAE": mae_self, "stable_MAE": stable_self}
        print(f"  {year} self: MAE={mae_self:.4f}, stable={stable_self:.4f}")
        del model; torch.cuda.empty_cache()

    # Instance norm self-year
    instnorm_self = {}
    for year in years:
        if year not in instnorm_ckpts:
            continue
        c = cache[year]
        model = load_model(instnorm_ckpts[year])
        mae_self, pn_self = evaluate(model, c["x"], c["y"],
                                     c["mean"], c["std"], c["mean"], c["std"],
                                     method="instance_norm")
        stable_self = float(pn_self[stable_idx].mean())
        instnorm_self[year] = {"MAE": mae_self, "stable_MAE": stable_self}
        print(f"  {year} self (instnorm): MAE={mae_self:.4f}, stable={stable_self:.4f}")
        del model; torch.cuda.empty_cache()

    # === SUMMARY ===
    print("\n" + "=" * 80)
    print("SUMMARY: Avg Cross-Year Degradation")
    print("=" * 80)

    avg_self_mae = np.mean([self_results[y]["MAE"] for y in years])
    avg_self_stable = np.mean([self_results[y]["stable_MAE"] for y in years])

    for metric, label in [("MAE", "All Nodes"), ("stable_MAE", "Stable Functional")]:
        print(f"\n{label}:")

        avg_self = np.mean([self_results[y][metric] for y in years])

        # (A) Train scaler cross
        cross_a = [results[f"{ty}_{tey}"]["train_scaler"][metric] for ty, tey in cross_pairs]
        avg_a = np.mean(cross_a)
        deg_a = avg_a - avg_self

        # (B) Corrected scaler cross
        cross_b = [results[f"{ty}_{tey}"]["corrected_scaler"][metric] for ty, tey in cross_pairs]
        avg_b = np.mean(cross_b)
        deg_b = avg_b - avg_self

        # (C) Instance norm cross
        cross_c = [results[f"{ty}_{tey}"]["instance_norm"][metric]
                   for ty, tey in cross_pairs
                   if "instance_norm" in results[f"{ty}_{tey}"]]
        if cross_c and instnorm_self:
            avg_self_in = np.mean([instnorm_self[y][metric] for y in years if y in instnorm_self])
            avg_c = np.mean(cross_c)
            deg_c = avg_c - avg_self_in

        print(f"  Avg self-year MAE:            {avg_self:.4f}")
        print(f"  (A) Train scaler:             {avg_a:.4f}  (degrade: +{deg_a:.4f}, {deg_a/avg_self*100:.1f}%)")
        print(f"  (B) Corrected scaler:         {avg_b:.4f}  (degrade: +{deg_b:.4f}, {deg_b/avg_self*100:.1f}%)")
        if cross_c and instnorm_self:
            print(f"  (C) Instance norm:            {avg_c:.4f}  (degrade: +{deg_c:.4f}, {deg_c/avg_self_in*100:.1f}%)")

        print(f"\n  Scaler mismatch contribution: {deg_a - deg_b:.4f} MAE ({(deg_a - deg_b)/deg_a*100:.1f}% of total)")
        print(f"  True model drift (corrected):  {deg_b:.4f} MAE ({deg_b/deg_a*100:.1f}% of total)")
        if cross_c and instnorm_self:
            print(f"  Instance norm drift:           {deg_c:.4f} MAE")
            print(f"  Instance norm vs corrected:    {deg_b - deg_c:+.4f} MAE "
                  f"({'instance norm better' if deg_c < deg_b else 'corrected scaler better'})")

    # Save
    output = {"cross_pairs": results, "self_year": self_results, "instnorm_self": instnorm_self}
    output_path = "eda/concept_drift/scaler_mismatch_test_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
