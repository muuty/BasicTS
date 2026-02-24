"""Per-node pattern drift analysis after instance normalization.

After removing ALL scale information (additive + multiplicative),
43.5% of degradation remains. Is this pattern drift:
  (A) Uniform across all nodes? → End-to-end approach sufficient
  (B) Concentrated in specific nodes? → Node-level adaptation needed

Also compares baseline vs instance-norm per-node degradation to see
which nodes benefit from scale removal and which don't.
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


def load_data(dataset_dir):
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


def get_per_node_mae(model, test_x, test_y, mean, std, method, batch_size=64):
    """Returns per-node MAE array (N,)."""
    test_x_norm = test_x.copy()
    test_x_norm[:, :, :, 0] = (test_x[:, :, :, 0] - mean) / std

    all_preds = []
    with torch.no_grad():
        for i in range(0, len(test_x_norm), batch_size):
            bx = torch.FloatTensor(test_x_norm[i:i+batch_size]).to(DEVICE)

            if method == "baseline":
                out = model(bx, None, 0, 0, False)["prediction"]
                pred = out.cpu().numpy() * std + mean

            elif method == "instance_norm":
                flow = bx[:, :, :, 0]
                fm = flow.mean(dim=1, keepdim=True)
                fs = flow.std(dim=1, keepdim=True) + 1e-5
                bx_n = bx.clone()
                bx_n[:, :, :, 0] = (flow - fm) / fs
                out = model(bx_n, None, 0, 0, False)["prediction"]
                out = out * fs.unsqueeze(-1) + fm.unsqueeze(-1)
                pred = out.cpu().numpy() * std + mean

            all_preds.append(pred)

    preds = np.concatenate(all_preds, axis=0)
    per_node_mae = np.mean(np.abs(preds - test_y), axis=(0, 1, 3))  # (N,)
    return per_node_mae


def main():
    years = [2022, 2023, 2024]
    instance_norm_ckpts = find_checkpoints("ConceptDrift_InstanceNorm")

    # Load data
    cache = {}
    for year in years:
        td, m, s = load_data(DATASETS[year])
        tx, ty = create_samples(td)
        cache[year] = {"x": tx, "y": ty, "mean": m, "std": s}

    # Load stable indices
    stable_idx = np.load("datasets/SAN_BERNARDINO_2022_Q1/stable_functional_indices.npy")

    # Compute per-node MAE for all (train, test) pairs
    # Focus on cross-year pairs
    cross_pairs = [(2022, 2023), (2022, 2024), (2023, 2022), (2023, 2024), (2024, 2022), (2024, 2023)]

    results = {}

    for method_name, ckpts in [("baseline", BASELINE_CHECKPOINTS), ("instance_norm", instance_norm_ckpts)]:
        print(f"\n{'='*60}")
        print(f"  {method_name.upper()}")
        print(f"{'='*60}")

        # Self-year per-node MAE
        self_per_node = {}
        for year in years:
            model = load_model(ckpts[year])
            c = cache[year]
            pn_mae = get_per_node_mae(model, c["x"], c["y"], c["mean"], c["std"], method_name)
            self_per_node[year] = pn_mae
            print(f"  Self {year}: overall MAE={pn_mae.mean():.4f}, stable MAE={pn_mae[stable_idx].mean():.4f}")
            del model; torch.cuda.empty_cache()

        # Cross-year per-node MAE and degradation
        cross_degradations = []  # per-node degradation across all cross pairs
        for train_year, test_year in cross_pairs:
            model = load_model(ckpts[train_year])
            c_test = cache[test_year]
            c_train = cache[train_year]
            pn_mae = get_per_node_mae(model, c_test["x"], c_test["y"], c_train["mean"], c_train["std"], method_name)

            # Per-node degradation: cross MAE - self MAE of the train year model on its own data
            pn_degrade = pn_mae - self_per_node[train_year]
            cross_degradations.append(pn_degrade)

            print(f"  Cross {train_year}→{test_year}: MAE={pn_mae.mean():.4f}, "
                  f"stable MAE={pn_mae[stable_idx].mean():.4f}, "
                  f"degrade={pn_degrade.mean():.4f}")
            del model; torch.cuda.empty_cache()

        # Average per-node degradation across all cross pairs
        avg_degrade = np.mean(cross_degradations, axis=0)  # (N,)
        results[method_name] = {
            "avg_per_node_degradation": avg_degrade,
            "self_per_node": {y: self_per_node[y] for y in years},
        }

    # ========================================
    # ANALYSIS
    # ========================================
    print("\n" + "=" * 60)
    print("  PER-NODE PATTERN DRIFT ANALYSIS (stable functional only)")
    print("=" * 60)

    bl_deg = results["baseline"]["avg_per_node_degradation"][stable_idx]
    in_deg = results["instance_norm"]["avg_per_node_degradation"][stable_idx]

    print(f"\n--- Baseline degradation (stable, N={len(stable_idx)}) ---")
    print(f"  Mean:   {bl_deg.mean():.4f}")
    print(f"  Median: {np.median(bl_deg):.4f}")
    print(f"  Std:    {bl_deg.std():.4f}")
    print(f"  Min:    {bl_deg.min():.4f}")
    print(f"  Max:    {bl_deg.max():.4f}")
    print(f"  >0 (degraded): {(bl_deg > 0).sum()}/{len(bl_deg)} ({(bl_deg > 0).mean()*100:.1f}%)")
    print(f"  <=0 (improved): {(bl_deg <= 0).sum()}/{len(bl_deg)} ({(bl_deg <= 0).mean()*100:.1f}%)")

    print(f"\n--- Instance-norm degradation (stable, N={len(stable_idx)}) ---")
    print(f"  Mean:   {in_deg.mean():.4f}")
    print(f"  Median: {np.median(in_deg):.4f}")
    print(f"  Std:    {in_deg.std():.4f}")
    print(f"  Min:    {in_deg.min():.4f}")
    print(f"  Max:    {in_deg.max():.4f}")
    print(f"  >0 (degraded): {(in_deg > 0).sum()}/{len(in_deg)} ({(in_deg > 0).mean()*100:.1f}%)")
    print(f"  <=0 (improved): {(in_deg <= 0).sum()}/{len(in_deg)} ({(in_deg <= 0).mean()*100:.1f}%)")

    # Concentration analysis
    print(f"\n--- Pattern drift concentration (instance-norm, stable) ---")
    sorted_deg = np.sort(in_deg)[::-1]  # descending
    total_pos_deg = np.sum(sorted_deg[sorted_deg > 0])
    for pct in [5, 10, 20, 50]:
        k = max(1, int(len(stable_idx) * pct / 100))
        top_k_sum = sorted_deg[:k].sum()
        if total_pos_deg > 0:
            contrib = top_k_sum / total_pos_deg * 100
        else:
            contrib = 0
        print(f"  Top {pct}% nodes ({k}): sum_degrade={top_k_sum:.2f}, "
              f"contrib={contrib:.1f}% of total positive degradation")

    # Percentile distribution
    print(f"\n--- Percentile distribution of pattern drift (instance-norm, stable) ---")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  P{p}: {np.percentile(in_deg, p):.4f}")

    # Improvement from baseline to instance-norm per node
    improvement = bl_deg - in_deg  # positive = instance norm helped
    print(f"\n--- Scale removal benefit per node (baseline_deg - instnorm_deg) ---")
    print(f"  Mean improvement: {improvement.mean():.4f}")
    print(f"  Median improvement: {np.median(improvement):.4f}")
    print(f"  Nodes where instance-norm helped: {(improvement > 0).sum()}/{len(improvement)} ({(improvement > 0).mean()*100:.1f}%)")
    print(f"  Nodes where instance-norm hurt: {(improvement < 0).sum()}/{len(improvement)} ({(improvement < 0).mean()*100:.1f}%)")

    # Categorize nodes by drift type
    print(f"\n--- Node categorization (stable functional) ---")
    scale_threshold = 1.0  # MAE improvement threshold for "scale helped"
    pattern_threshold = 2.0  # remaining degradation threshold for "has pattern drift"

    scale_helped = improvement > scale_threshold
    has_pattern_drift = in_deg > pattern_threshold

    n_scale_only = (scale_helped & ~has_pattern_drift).sum()
    n_pattern_only = (~scale_helped & has_pattern_drift).sum()
    n_both = (scale_helped & has_pattern_drift).sum()
    n_neither = (~scale_helped & ~has_pattern_drift).sum()

    print(f"  Scale-only (scale helped, no pattern drift): {n_scale_only} ({n_scale_only/len(stable_idx)*100:.1f}%)")
    print(f"  Pattern-only (no scale benefit, has pattern drift): {n_pattern_only} ({n_pattern_only/len(stable_idx)*100:.1f}%)")
    print(f"  Both (scale helped AND pattern drift): {n_both} ({n_both/len(stable_idx)*100:.1f}%)")
    print(f"  Neither (stable across years): {n_neither} ({n_neither/len(stable_idx)*100:.1f}%)")

    # Save per-node data for further analysis
    save_dir = "eda/concept_drift"
    np.save(os.path.join(save_dir, "per_node_baseline_degradation.npy"),
            results["baseline"]["avg_per_node_degradation"])
    np.save(os.path.join(save_dir, "per_node_instnorm_degradation.npy"),
            results["instance_norm"]["avg_per_node_degradation"])

    print(f"\nPer-node degradation arrays saved to {save_dir}/")


if __name__ == "__main__":
    main()
