"""Per-node PB Delta Analysis

Questions:
1. Which nodes does PB change the most? (L2 norm of delta = softmax(W) @ P)
2. Does PB delta magnitude correlate with per-node drift severity?
3. Does PB delta correlate with node categories (dead/functional)?
4. Does PB automatically focus adaptation on drifting nodes?

For each (source, target) pair × time budget:
- Train PB adapter
- Extract delta per node
- Compute per-node zero-shot MAE (drift severity proxy)
- Correlate delta magnitude with drift severity
"""
import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
sys.path.append("/data/pretrainingbasicts")

from baselines.STAEformer.arch import STAEformer

MODEL_PARAM = {
    "num_nodes": 893, "in_steps": 12, "out_steps": 12, "steps_per_day": 288,
    "input_dim": 3, "output_dim": 1, "input_embedding_dim": 24,
    "tod_embedding_dim": 24, "dow_embedding_dim": 24, "spatial_embedding_dim": 0,
    "adaptive_embedding_dim": 24, "feed_forward_dim": 256, "num_heads": 4,
    "num_layers": 1, "dropout": 0.1, "use_mixed_proj": True,
}

INSTNORM_CHECKPOINTS = {
    2022: "checkpoints/ConceptDrift_InstanceNorm/SAN_BERNARDINO_2022_Q1_30_12_12/ee3765011653c94baa0ab81045fb239e/STAEformer_best_val_MAE.pt",
    2023: "checkpoints/ConceptDrift_InstanceNorm/SAN_BERNARDINO_2023_Q1_30_12_12/d2940bc326d7f29eedeedb573e026973/STAEformer_best_val_MAE.pt",
    2024: "checkpoints/ConceptDrift_InstanceNorm/SAN_BERNARDINO_2024_Q1_30_12_12/16c8094c3cf1ef5013e0384708d2e954/STAEformer_best_val_MAE.pt",
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
EPOCHS = 10
BATCH_SIZE = 16
K = 8
NUM_NODES = 893

FINETUNE_HOURS = [3, 6, 12, 24, 72, 168]

# Sensor categories
DEAD_INDICES = np.load("datasets/xtraffic/SAN_BERNARDINO/dead_indices.npy")
MAJOR_FAIL_INDICES = np.load("datasets/xtraffic/SAN_BERNARDINO/major_fail_indices.npy")


class STAEformerWithPB(nn.Module):
    def __init__(self, backbone, num_nodes=893, adp_dim=24, K=8):
        super().__init__()
        self.backbone = backbone
        self.pattern_bank = nn.Parameter(torch.randn(K, adp_dim) * 0.01)
        self.node_weights = nn.Parameter(torch.zeros(num_nodes, K))

    def get_delta(self):
        """Extract per-node delta: (N, adp_dim)"""
        with torch.no_grad():
            return (F.softmax(self.node_weights, dim=-1) @ self.pattern_bank).cpu().numpy()

    def forward(self, history_data, future_data, batch_seen, epoch, train, **kwargs):
        x = history_data
        batch_size = x.shape[0]
        enc = self.backbone.encoder
        if enc.tod_embedding_dim > 0:
            tod = x[..., enc.tod_index] * enc.steps_per_day
        if enc.dow_embedding_dim > 0:
            dow = x[..., enc.dow_index] * 7
        x = x[..., :enc.input_dim]
        x = enc.input_proj(x)
        features = [x]
        if enc.tod_embedding_dim > 0:
            features.append(enc.tod_embedding(tod.long()))
        if enc.dow_embedding_dim > 0:
            features.append(enc.dow_embedding(dow.long()))
        if enc.adaptive_embedding_dim > 0:
            adp_emb = enc.adaptive_embedding.expand(batch_size, *enc.adaptive_embedding.shape)
            delta = F.softmax(self.node_weights, dim=-1) @ self.pattern_bank
            adp_emb = adp_emb + delta.unsqueeze(0).unsqueeze(0)
            features.append(adp_emb)
        x = torch.cat(features, dim=-1)
        for attn in enc.attn_layers_t:
            x = attn(x, dim=1)
        x = self.backbone.spatial(x, None)
        out = self.backbone.decoder(x)
        return {"prediction": out}


def load_backbone(ckpt_path):
    backbone = STAEformer(**MODEL_PARAM)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    backbone.load_state_dict(ckpt["model_state_dict"])
    return backbone


def load_data_and_scaler(dataset_dir):
    desc = json.load(open(os.path.join(dataset_dir, "desc.json")))
    shape = desc["shape"]
    data = np.memmap(os.path.join(dataset_dir, "data.dat"), dtype="float32", mode="r").reshape(shape)
    n_total = shape[0]
    n_train = int(n_total * TRAIN_RATIO)
    mean = float(np.mean(data[:n_train, :, 0]))
    std = float(np.std(data[:n_train, :, 0]))
    return data, mean, std, n_total


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


def apply_instance_norm(bx):
    flow = bx[:, :, :, 0]
    inst_mean = flow.mean(dim=1, keepdim=True)
    inst_std = flow.std(dim=1, keepdim=True) + 1e-5
    bx_normed = bx.clone()
    bx_normed[:, :, :, 0] = (flow - inst_mean) / inst_std
    return bx_normed, inst_mean, inst_std


def denorm_instance_norm(pred, inst_mean, inst_std):
    return pred * inst_std.unsqueeze(-1) + inst_mean.unsqueeze(-1)


def per_node_mae(model, test_x, test_y, mean, std, batch_size=64):
    """Returns per-node MAE array of shape (N,)."""
    model.eval()
    test_x_norm = normalize_input(test_x, mean, std)
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(test_x_norm), batch_size):
            bx = torch.FloatTensor(test_x_norm[i:i+batch_size]).to(DEVICE)
            bx_normed, inst_mean, inst_std = apply_instance_norm(bx)
            out = model(bx_normed, None, 0, 0, False)["prediction"]
            out = denorm_instance_norm(out, inst_mean, inst_std)
            pred = out.cpu().numpy() * std + mean
            all_preds.append(pred)
    preds = np.concatenate(all_preds, axis=0)  # (S, T, N, 1)
    # per-node MAE: mean over samples and timesteps
    return np.mean(np.abs(preds - test_y), axis=(0, 1, 3))  # (N,)


def train_pb(model, train_x, train_y, mean, std, epochs=EPOCHS, lr=0.001):
    model = model.to(DEVICE)
    model.train()
    for name, param in model.named_parameters():
        param.requires_grad = ("pattern_bank" in name or "node_weights" in name)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    train_x_norm = normalize_input(train_x, mean, std)
    for epoch in range(epochs):
        indices = np.random.permutation(len(train_x_norm))
        for i in range(0, len(indices), BATCH_SIZE):
            batch_idx = indices[i:i+BATCH_SIZE]
            bx = torch.FloatTensor(train_x_norm[batch_idx]).to(DEVICE)
            by = torch.FloatTensor(train_y[batch_idx]).to(DEVICE)
            bx_normed, inst_mean, inst_std = apply_instance_norm(bx)
            pred = model(bx_normed, None, 0, 0, True)["prediction"]
            pred = denorm_instance_norm(pred, inst_mean, inst_std)
            loss = nn.L1Loss()(pred * std + mean, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def make_category_labels():
    """Returns array (N,) with category: 0=dead, 1=major_fail, 2=functional."""
    cats = np.full(NUM_NODES, 2, dtype=int)  # default functional
    cats[DEAD_INDICES] = 0
    cats[MAJOR_FAIL_INDICES] = 1
    return cats


def main():
    years = [2022, 2023, 2024]
    pairs = [(s, t) for s in years for t in years if s != t]
    categories = make_category_labels()
    cat_names = {0: "dead", 1: "major_fail", 2: "functional"}

    print("=" * 70)
    print("PER-NODE PB DELTA ANALYSIS")
    print("=" * 70)

    # Load data
    data_cache = {}
    for year in years:
        data, mean, std, n_total = load_data_and_scaler(DATASETS[year])
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * 0.2)
        test_data = data[n_train + n_val:]
        test_x, test_y = create_samples(test_data, INPUT_LEN, OUTPUT_LEN)
        data_cache[year] = {"full_data": data, "test_x": test_x, "test_y": test_y,
                            "mean": mean, "std": std}
        print(f"  {year}: test={len(test_x)}")

    all_results = {}

    for source_year, target_year in pairs:
        ckpt_path = INSTNORM_CHECKPOINTS[source_year]
        if not os.path.exists(ckpt_path):
            print(f"  Skipping {source_year}->{target_year}: checkpoint missing")
            continue
        src = data_cache[source_year]
        tgt = data_cache[target_year]
        pair_key = f"{source_year}_{target_year}"

        print(f"\n{'='*60}")
        print(f"  {source_year} -> {target_year}")
        print(f"{'='*60}")

        # Step 1: Compute per-node zero-shot MAE on target (drift severity proxy)
        print("  Computing zero-shot per-node MAE (drift severity)...")
        backbone = load_backbone(ckpt_path).to(DEVICE)
        zs_node_mae = per_node_mae(backbone, tgt["test_x"], tgt["test_y"],
                                   src["mean"], src["std"])
        print(f"    Zero-shot overall MAE: {np.mean(zs_node_mae):.2f}")
        print(f"    Per-category zero-shot MAE:")
        for cat_id, cat_name in cat_names.items():
            mask = categories == cat_id
            print(f"      {cat_name}: {np.mean(zs_node_mae[mask]):.2f} (n={mask.sum()})")
        del backbone
        torch.cuda.empty_cache()

        pair_results = {
            "zeroshot_node_mae": zs_node_mae.tolist(),
        }

        for hours in FINETUNE_HOURS:
            n_steps = hours * 12
            ft_data = tgt["full_data"][:n_steps]
            ft_x, ft_y = create_samples(ft_data, INPUT_LEN, OUTPUT_LEN)
            h_label = f"{hours}h" if hours < 24 else f"{hours//24}d"

            if len(ft_x) < 2:
                print(f"    {h_label}: insufficient samples, skipping")
                continue

            print(f"\n  --- {h_label} ({len(ft_x)} samples) ---")

            # Train PB
            backbone = load_backbone(ckpt_path)
            model = STAEformerWithPB(backbone)
            model = train_pb(model, ft_x, ft_y, src["mean"], src["std"])

            # Extract delta
            delta = model.get_delta()  # (N, 24)
            delta_norm = np.linalg.norm(delta, axis=1)  # (N,)

            # Per-node MAE after PB
            pb_node_mae = per_node_mae(model, tgt["test_x"], tgt["test_y"],
                                       src["mean"], src["std"])

            # MAE improvement per node
            mae_improvement = zs_node_mae - pb_node_mae  # positive = PB helped

            print(f"    PB overall MAE: {np.mean(pb_node_mae):.2f}")
            print(f"    Delta L2 norm: mean={np.mean(delta_norm):.4f}, "
                  f"std={np.std(delta_norm):.4f}, "
                  f"max={np.max(delta_norm):.4f}")

            # Correlations
            # 1. Delta magnitude vs drift severity (zero-shot MAE)
            r_drift, p_drift = stats.spearmanr(delta_norm, zs_node_mae)
            print(f"    Corr(delta_norm, zeroshot_MAE): r={r_drift:.3f}, p={p_drift:.2e}")

            # 2. Delta magnitude vs MAE improvement
            r_improv, p_improv = stats.spearmanr(delta_norm, mae_improvement)
            print(f"    Corr(delta_norm, MAE_improvement): r={r_improv:.3f}, p={p_improv:.2e}")

            # 3. Per-category delta stats
            print(f"    Per-category delta norm:")
            for cat_id, cat_name in cat_names.items():
                mask = categories == cat_id
                print(f"      {cat_name}: mean={np.mean(delta_norm[mask]):.4f}, "
                      f"std={np.std(delta_norm[mask]):.4f}")

            # 4. Top-10 nodes by delta magnitude
            top10_idx = np.argsort(delta_norm)[-10:][::-1]
            print(f"    Top-10 nodes by delta: {top10_idx.tolist()}")
            print(f"      delta_norms: {delta_norm[top10_idx].tolist()}")
            print(f"      categories:  {[cat_names[categories[i]] for i in top10_idx]}")
            print(f"      zs_MAE:      {[f'{zs_node_mae[i]:.1f}' for i in top10_idx]}")

            # 5. Quartile analysis: split nodes by drift severity, check delta in each
            quartiles = np.percentile(zs_node_mae, [25, 50, 75])
            q_labels = ["Q1(low drift)", "Q2", "Q3", "Q4(high drift)"]
            q_bounds = [(-np.inf, quartiles[0]), (quartiles[0], quartiles[1]),
                        (quartiles[1], quartiles[2]), (quartiles[2], np.inf)]
            print(f"    Drift quartile analysis:")
            for ql, (lo, hi) in zip(q_labels, q_bounds):
                mask = (zs_node_mae > lo) & (zs_node_mae <= hi)
                if mask.sum() > 0:
                    print(f"      {ql}: n={mask.sum()}, "
                          f"mean_delta={np.mean(delta_norm[mask]):.4f}, "
                          f"mean_improvement={np.mean(mae_improvement[mask]):.2f}")

            # Store results
            pair_results[h_label] = {
                "delta_norm": delta_norm.tolist(),
                "pb_node_mae": pb_node_mae.tolist(),
                "mae_improvement": mae_improvement.tolist(),
                "corr_delta_vs_drift": {"r": float(r_drift), "p": float(p_drift)},
                "corr_delta_vs_improvement": {"r": float(r_improv), "p": float(p_improv)},
                "overall_pb_mae": float(np.mean(pb_node_mae)),
                "top10_nodes": top10_idx.tolist(),
            }

            del model, backbone
            torch.cuda.empty_cache()

        all_results[pair_key] = pair_results

    # ===== AGGREGATE SUMMARY =====
    print("\n" + "=" * 70)
    print("AGGREGATE SUMMARY")
    print("=" * 70)

    print("\n--- Correlation: delta_norm vs zeroshot_MAE (Spearman) ---")
    header = f"{'':>5}"
    for s, t in pairs:
        header += f" {s}->{t:>5}"
    header += "    mean"
    print(header)
    for hours in FINETUNE_HOURS:
        h = f"{hours}h" if hours < 24 else f"{hours//24}d"
        row = f"{h:>5}"
        vals = []
        for s, t in pairs:
            pk = f"{s}_{t}"
            if pk in all_results and h in all_results[pk]:
                r = all_results[pk][h]["corr_delta_vs_drift"]["r"]
                row += f"  {r:>7.3f}"
                vals.append(r)
            else:
                row += f"  {'N/A':>7}"
        if vals:
            row += f"  {np.mean(vals):>7.3f}"
        print(row)

    print("\n--- Correlation: delta_norm vs MAE_improvement (Spearman) ---")
    print(header)
    for hours in FINETUNE_HOURS:
        h = f"{hours}h" if hours < 24 else f"{hours//24}d"
        row = f"{h:>5}"
        vals = []
        for s, t in pairs:
            pk = f"{s}_{t}"
            if pk in all_results and h in all_results[pk]:
                r = all_results[pk][h]["corr_delta_vs_improvement"]["r"]
                row += f"  {r:>7.3f}"
                vals.append(r)
            else:
                row += f"  {'N/A':>7}"
        if vals:
            row += f"  {np.mean(vals):>7.3f}"
        print(row)

    print("\n--- Mean delta_norm by category ---")
    for hours in [24, 168]:  # 1d and 7d for brevity
        h = f"{hours}h" if hours < 24 else f"{hours//24}d"
        print(f"\n  {h}:")
        for cat_id, cat_name in cat_names.items():
            mask = categories == cat_id
            vals = []
            for s, t in pairs:
                pk = f"{s}_{t}"
                if pk in all_results and h in all_results[pk]:
                    dn = np.array(all_results[pk][h]["delta_norm"])
                    vals.append(np.mean(dn[mask]))
            if vals:
                print(f"    {cat_name:>12}: {np.mean(vals):.4f} (avg across pairs)")

    # Save results
    output_path = "eda/concept_drift/pb_node_delta_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
