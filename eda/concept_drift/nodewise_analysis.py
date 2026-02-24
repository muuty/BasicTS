"""Node-wise analysis: Where does PB win/lose vs Full FT?

For representative pairs and time budgets, extracts per-node MAE
and correlates with node characteristics (traffic volume, drift, sensor health).
"""
import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
STEPS_PER_DAY = 288
DEVICE = "cuda:1"
FINETUNE_EPOCHS = 10
BATCH_SIZE = 16
K = 8


# ============================================================
# Model (PB)
# ============================================================

class STAEformerWithPB(nn.Module):
    def __init__(self, backbone, num_nodes, adp_dim, K):
        super().__init__()
        self.backbone = backbone
        self.pattern_bank = nn.Parameter(torch.randn(K, adp_dim) * 0.01)
        self.node_weights = nn.Parameter(torch.zeros(num_nodes, K))

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


# ============================================================
# Utilities
# ============================================================

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


def evaluate_per_node(model, test_x, test_y, mean, std, batch_size=64):
    """Returns per-node MAE array (shape: num_nodes)."""
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
    preds = np.concatenate(all_preds, axis=0)  # (samples, 12, 893, 1)
    # Per-node MAE: average over samples and horizons
    per_node_mae = np.mean(np.abs(preds - test_y), axis=(0, 1, 3))  # (893,)
    return per_node_mae


def finetune_model(model, trainable_names, train_x, train_y, mean, std,
                   epochs=FINETUNE_EPOCHS, lr=0.001):
    model = model.to(DEVICE)
    model.train()
    for name, param in model.named_parameters():
        param.requires_grad = any(t in name for t in trainable_names)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
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
    return model, n_trainable


# ============================================================
# Node characteristics
# ============================================================

def compute_node_characteristics(data_cache, source_year, target_year):
    """Compute per-node characteristics for analysis."""
    src_data = data_cache[source_year]["full_data"]
    tgt_data = data_cache[target_year]["full_data"]

    # Use train portion for statistics
    n_src = int(src_data.shape[0] * TRAIN_RATIO)
    n_tgt = int(tgt_data.shape[0] * TRAIN_RATIO)

    src_flow = src_data[:n_src, :, 0]  # (T, N)
    tgt_flow = tgt_data[:n_tgt, :, 0]

    chars = {}
    # Mean flow per node
    chars["src_mean_flow"] = np.mean(src_flow, axis=0)  # (N,)
    chars["tgt_mean_flow"] = np.mean(tgt_flow, axis=0)
    # Drift magnitude (absolute change in mean flow)
    chars["drift_magnitude"] = np.abs(chars["tgt_mean_flow"] - chars["src_mean_flow"])
    chars["drift_signed"] = chars["tgt_mean_flow"] - chars["src_mean_flow"]
    # Flow variability
    chars["src_std_flow"] = np.std(src_flow, axis=0)
    chars["tgt_std_flow"] = np.std(tgt_flow, axis=0)
    # Zero rate (sensor health proxy)
    chars["src_zero_rate"] = np.mean(src_flow == 0, axis=0)
    chars["tgt_zero_rate"] = np.mean(tgt_flow == 0, axis=0)

    # Sensor categories
    dead_idx = np.load("datasets/xtraffic/SAN_BERNARDINO/dead_indices.npy") \
        if os.path.exists("datasets/xtraffic/SAN_BERNARDINO/dead_indices.npy") else np.array([])
    major_fail_idx = np.load("datasets/xtraffic/SAN_BERNARDINO/major_fail_indices.npy") \
        if os.path.exists("datasets/xtraffic/SAN_BERNARDINO/major_fail_indices.npy") else np.array([])

    category = np.full(893, "functional", dtype=object)
    for i in dead_idx:
        if i < 893:
            category[i] = "dead"
    for i in major_fail_idx:
        if i < 893:
            category[i] = "major_fail"
    chars["category"] = category

    return chars


# ============================================================
# Main analysis
# ============================================================

def main():
    years = [2022, 2023, 2024]
    # Analyze representative pairs: one high-drift, one low-drift
    analysis_pairs = [(2022, 2023), (2022, 2024), (2023, 2022)]
    analysis_hours = [12, 24, 168]  # 12h, 1d, 7d

    print("=" * 70)
    print("NODE-WISE ANALYSIS: PB vs Full FT")
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

    all_results = {}

    for source_year, target_year in analysis_pairs:
        ckpt_path = INSTNORM_CHECKPOINTS[source_year]
        src = data_cache[source_year]
        tgt = data_cache[target_year]
        pair = f"{source_year}_{target_year}"

        print(f"\n{'='*50} {source_year}->{target_year} {'='*10}")

        # Node characteristics
        chars = compute_node_characteristics(data_cache, source_year, target_year)
        print(f"  Drift magnitude: mean={chars['drift_magnitude'].mean():.1f}, "
              f"max={chars['drift_magnitude'].max():.1f}")

        # RevIN baseline (per-node)
        backbone = load_backbone(ckpt_path).to(DEVICE)
        revin_node_mae = evaluate_per_node(backbone, tgt["test_x"], tgt["test_y"],
                                            src["mean"], src["std"])
        print(f"  RevIN per-node MAE: mean={revin_node_mae.mean():.2f}, "
              f"median={np.median(revin_node_mae):.2f}, max={revin_node_mae.max():.2f}")
        del backbone; torch.cuda.empty_cache()

        for hours in analysis_hours:
            n_steps = hours * 12
            ft_data = tgt["full_data"][:n_steps]
            ft_x, ft_y = create_samples(ft_data, INPUT_LEN, OUTPUT_LEN)
            h_label = f"{hours}h" if hours < 24 else f"{hours//24}d"
            print(f"\n  --- {h_label} ({len(ft_x)} samples) ---")

            # PB
            backbone = load_backbone(ckpt_path)
            pb_model = STAEformerWithPB(backbone, 893, 24, K)
            pb_model, _ = finetune_model(pb_model, ["pattern_bank", "node_weights"],
                                          ft_x, ft_y, src["mean"], src["std"])
            pb_node_mae = evaluate_per_node(pb_model, tgt["test_x"], tgt["test_y"],
                                             src["mean"], src["std"])
            # Extract PB weights for analysis
            pb_weights = F.softmax(pb_model.node_weights.data, dim=-1).cpu().numpy()
            pb_assignment = np.argmax(pb_weights, axis=-1)
            del pb_model, backbone; torch.cuda.empty_cache()

            # Full FT
            backbone = load_backbone(ckpt_path)
            backbone, _ = finetune_model(backbone, [""], ft_x, ft_y,
                                          src["mean"], src["std"], lr=0.0001)
            full_node_mae = evaluate_per_node(backbone, tgt["test_x"], tgt["test_y"],
                                               src["mean"], src["std"])
            del backbone; torch.cuda.empty_cache()

            # ============================================================
            # Analysis
            # ============================================================

            diff = pb_node_mae - full_node_mae  # positive = Full FT better
            pb_better = diff < 0
            full_better = diff > 0

            print(f"    Overall: PB={pb_node_mae.mean():.2f}, Full={full_node_mae.mean():.2f}")
            print(f"    PB wins at {pb_better.sum()}/893 nodes, Full wins at {full_better.sum()}/893")
            print(f"    Mean diff (PB-Full): {diff.mean():+.2f}, median: {np.median(diff):+.2f}")

            # --- Analysis 1: By sensor category ---
            print(f"\n    [By Sensor Category]")
            for cat in ["dead", "major_fail", "functional"]:
                mask = chars["category"] == cat
                n = mask.sum()
                if n == 0:
                    continue
                pb_cat = pb_node_mae[mask].mean()
                full_cat = full_node_mae[mask].mean()
                revin_cat = revin_node_mae[mask].mean()
                pb_wins_cat = (diff[mask] < 0).sum()
                print(f"      {cat:>12} ({n:>3}): RevIN={revin_cat:.2f}  "
                      f"PB={pb_cat:.2f}  Full={full_cat:.2f}  "
                      f"diff={full_cat-pb_cat:+.2f}  PB wins {pb_wins_cat}/{n}")

            # --- Analysis 2: Correlation with drift magnitude ---
            from scipy import stats
            r_pb_drift, p_pb = stats.pearsonr(chars["drift_magnitude"], pb_node_mae)
            r_full_drift, p_full = stats.pearsonr(chars["drift_magnitude"], full_node_mae)
            r_diff_drift, p_diff = stats.pearsonr(chars["drift_magnitude"], diff)
            print(f"\n    [Correlation with Drift Magnitude]")
            print(f"      PB MAE vs drift:   r={r_pb_drift:.3f} (p={p_pb:.1e})")
            print(f"      Full MAE vs drift: r={r_full_drift:.3f} (p={p_full:.1e})")
            print(f"      Diff vs drift:     r={r_diff_drift:.3f} (p={p_diff:.1e})")
            if r_diff_drift > 0:
                print(f"      -> High-drift nodes: Full FT relatively better")
            else:
                print(f"      -> High-drift nodes: PB relatively better")

            # --- Analysis 3: Correlation with traffic volume ---
            r_pb_vol, _ = stats.pearsonr(chars["src_mean_flow"], pb_node_mae)
            r_full_vol, _ = stats.pearsonr(chars["src_mean_flow"], full_node_mae)
            r_diff_vol, p_vol = stats.pearsonr(chars["src_mean_flow"], diff)
            print(f"\n    [Correlation with Traffic Volume]")
            print(f"      PB MAE vs volume:   r={r_pb_vol:.3f}")
            print(f"      Full MAE vs volume: r={r_full_vol:.3f}")
            print(f"      Diff vs volume:     r={r_diff_vol:.3f} (p={p_vol:.1e})")

            # --- Analysis 4: Worst 10% nodes ---
            n_worst = 89  # ~10%
            pb_worst_idx = np.argsort(pb_node_mae)[-n_worst:]
            full_worst_idx = np.argsort(full_node_mae)[-n_worst:]
            overlap = len(set(pb_worst_idx) & set(full_worst_idx))

            print(f"\n    [Worst 10% Nodes (n={n_worst})]")
            print(f"      Overlap: {overlap}/{n_worst} ({overlap/n_worst*100:.0f}%)")
            print(f"      PB worst10% avg MAE:   {pb_node_mae[pb_worst_idx].mean():.2f}")
            print(f"      Full worst10% avg MAE: {full_node_mae[full_worst_idx].mean():.2f}")

            # Characteristics of PB-worst nodes
            pb_worst_drift = chars["drift_magnitude"][pb_worst_idx].mean()
            pb_worst_vol = chars["src_mean_flow"][pb_worst_idx].mean()
            full_worst_drift = chars["drift_magnitude"][full_worst_idx].mean()
            full_worst_vol = chars["src_mean_flow"][full_worst_idx].mean()
            all_drift = chars["drift_magnitude"].mean()
            all_vol = chars["src_mean_flow"].mean()

            print(f"      PB worst10%:   drift={pb_worst_drift:.1f} (avg={all_drift:.1f}), "
                  f"volume={pb_worst_vol:.1f} (avg={all_vol:.1f})")
            print(f"      Full worst10%: drift={full_worst_drift:.1f} (avg={all_drift:.1f}), "
                  f"volume={full_worst_vol:.1f} (avg={all_vol:.1f})")

            # Category breakdown of worst nodes
            for method_name, worst_idx in [("PB", pb_worst_idx), ("Full", full_worst_idx)]:
                cats = chars["category"][worst_idx]
                dead_n = (cats == "dead").sum()
                major_n = (cats == "major_fail").sum()
                func_n = (cats == "functional").sum()
                print(f"      {method_name} worst10% category: "
                      f"dead={dead_n}, major={major_n}, func={func_n}")

            # --- Analysis 5: Where PB beats Full FT ---
            pb_wins_mask = diff < -0.5  # PB better by >0.5 MAE
            full_wins_mask = diff > 0.5
            print(f"\n    [Where PB beats Full FT by >0.5 MAE]")
            print(f"      PB wins: {pb_wins_mask.sum()} nodes")
            if pb_wins_mask.sum() > 0:
                print(f"        Avg drift: {chars['drift_magnitude'][pb_wins_mask].mean():.1f} "
                      f"(vs all: {all_drift:.1f})")
                print(f"        Avg volume: {chars['src_mean_flow'][pb_wins_mask].mean():.1f}")
                pb_win_cats = chars["category"][pb_wins_mask]
                print(f"        Category: dead={sum(pb_win_cats=='dead')}, "
                      f"major={sum(pb_win_cats=='major_fail')}, "
                      f"func={sum(pb_win_cats=='functional')}")

            print(f"      Full FT wins: {full_wins_mask.sum()} nodes")
            if full_wins_mask.sum() > 0:
                print(f"        Avg drift: {chars['drift_magnitude'][full_wins_mask].mean():.1f}")
                print(f"        Avg volume: {chars['src_mean_flow'][full_wins_mask].mean():.1f}")
                full_win_cats = chars["category"][full_wins_mask]
                print(f"        Category: dead={sum(full_win_cats=='dead')}, "
                      f"major={sum(full_win_cats=='major_fail')}, "
                      f"func={sum(full_win_cats=='functional')}")

            # --- Analysis 6: PB prototype assignment pattern ---
            print(f"\n    [PB Prototype Assignment]")
            for proto_id in range(K):
                mask = pb_assignment == proto_id
                n = mask.sum()
                if n == 0:
                    continue
                proto_mae = pb_node_mae[mask].mean()
                proto_drift = chars["drift_magnitude"][mask].mean()
                proto_vol = chars["src_mean_flow"][mask].mean()
                proto_dead = sum(chars["category"][mask] == "dead")
                print(f"      Proto {proto_id}: {n:>3} nodes, "
                      f"MAE={proto_mae:.2f}, drift={proto_drift:.1f}, "
                      f"vol={proto_vol:.1f}, dead={proto_dead}")

            # Save per-node data
            all_results[f"{pair}_{h_label}"] = {
                "pb_node_mae": pb_node_mae.tolist(),
                "full_node_mae": full_node_mae.tolist(),
                "revin_node_mae": revin_node_mae.tolist(),
                "pb_assignment": pb_assignment.tolist(),
                "drift_magnitude": chars["drift_magnitude"].tolist(),
                "src_mean_flow": chars["src_mean_flow"].tolist(),
                "category": chars["category"].tolist(),
            }

    output_path = "eda/concept_drift/nodewise_analysis_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
