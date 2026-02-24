"""CKA (Centered Kernel Alignment) Analysis for Concept Drift Adaptation.

Measures representation similarity between source model and adapted models
at each layer of STAEformer (decomposed: encoder/spatial/decoder).

Hook points:
- input_proj: after input projection (before embedding concat)
- after_embedding: input_proj + tod/dow/adaptive concat (before attention)
- after_temporal_0: after temporal self-attention
- after_spatial_0: after spatial self-attention
- before_output: final representation before decoder

Analysis:
1. Source vs each adapted method (PB, emb_only, PB+head, full_ft)
2. Layer-wise CKA scores
3. Cross-method comparison
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


# ========================= Linear CKA =========================

def linear_cka(X, Y):
    """Linear CKA: ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)"""
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    num = torch.norm(Y.T @ X, p='fro') ** 2
    denom = torch.norm(X.T @ X, p='fro') * torch.norm(Y.T @ Y, p='fro')
    if denom < 1e-10:
        return 0.0
    return float(num / denom)


# ========================= Representation Extraction =========================

def extract_reps_encoder(model, x):
    """Extract representations from decomposed STAEformer (encoder/spatial/decoder)."""
    model.eval()
    enc = model.encoder
    reps = {}

    with torch.no_grad():
        batch_size = x.shape[0]
        if enc.tod_embedding_dim > 0:
            tod = x[..., enc.tod_index] * enc.steps_per_day
        if enc.dow_embedding_dim > 0:
            dow = x[..., enc.dow_index] * 7
        x_in = x[..., :enc.input_dim]

        x_emb = enc.input_proj(x_in)
        reps['input_proj'] = x_emb.clone()

        features = [x_emb]
        if enc.tod_embedding_dim > 0:
            features.append(enc.tod_embedding(tod.long()))
        if enc.dow_embedding_dim > 0:
            features.append(enc.dow_embedding(dow.long()))
        if enc.adaptive_embedding_dim > 0:
            adp_emb = enc.adaptive_embedding.expand(batch_size, *enc.adaptive_embedding.shape)
            features.append(adp_emb)

        h = torch.cat(features, dim=-1)
        reps['after_embedding'] = h.clone()

        for i, attn in enumerate(enc.attn_layers_t):
            h = attn(h, dim=1)
            reps[f'after_temporal_{i}'] = h.clone()

        for i, attn in enumerate(model.spatial.attn_layers_s):
            h = attn(h, dim=2)
            reps[f'after_spatial_{i}'] = h.clone()

        reps['before_output'] = h.clone()

    return reps


def extract_reps_pb(model_pb, x):
    """Extract representations from PB-wrapped STAEformer."""
    model_pb.eval()
    bb = model_pb.backbone
    enc = bb.encoder
    reps = {}

    with torch.no_grad():
        batch_size = x.shape[0]
        if enc.tod_embedding_dim > 0:
            tod = x[..., enc.tod_index] * enc.steps_per_day
        if enc.dow_embedding_dim > 0:
            dow = x[..., enc.dow_index] * 7
        x_in = x[..., :enc.input_dim]

        x_emb = enc.input_proj(x_in)
        reps['input_proj'] = x_emb.clone()

        features = [x_emb]
        if enc.tod_embedding_dim > 0:
            features.append(enc.tod_embedding(tod.long()))
        if enc.dow_embedding_dim > 0:
            features.append(enc.dow_embedding(dow.long()))
        if enc.adaptive_embedding_dim > 0:
            adp_emb = enc.adaptive_embedding.expand(batch_size, *enc.adaptive_embedding.shape)
            delta = F.softmax(model_pb.node_weights, dim=-1) @ model_pb.pattern_bank
            adp_emb = adp_emb + delta.unsqueeze(0).unsqueeze(0)
            features.append(adp_emb)

        h = torch.cat(features, dim=-1)
        reps['after_embedding'] = h.clone()

        for i, attn in enumerate(enc.attn_layers_t):
            h = attn(h, dim=1)
            reps[f'after_temporal_{i}'] = h.clone()

        for i, attn in enumerate(bb.spatial.attn_layers_s):
            h = attn(h, dim=2)
            reps[f'after_spatial_{i}'] = h.clone()

        reps['before_output'] = h.clone()

    return reps


# ========================= Data & Model Utils =========================

def load_backbone(ckpt_path):
    model = STAEformer(**MODEL_PARAM)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    return model


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


# ========================= PB Wrapper =========================

class STAEformerWithPB(nn.Module):
    def __init__(self, backbone, num_nodes=893, adp_dim=24, K=8):
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


# ========================= Fine-tuning =========================

def train_model(model, trainable_names, train_x, train_y, mean, std,
                epochs=EPOCHS, lr=0.001):
    model = model.to(DEVICE)
    model.train()
    for name, param in model.named_parameters():
        param.requires_grad = any(t in name for t in trainable_names)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    train_x_norm = normalize_input(train_x, mean, std)
    for epoch in range(epochs):
        indices = np.random.permutation(len(train_x_norm))
        for i in range(0, len(indices), BATCH_SIZE):
            batch_idx = indices[i:i+BATCH_SIZE]
            bx = torch.FloatTensor(train_x_norm[batch_idx]).to(DEVICE)
            by = torch.FloatTensor(train_y[batch_idx]).to(DEVICE)
            bx_normed, inst_mean, inst_std = apply_instance_norm(bx)
            pred = model(bx_normed, None, 0, 0, True)["prediction"]
            pred_denorm = pred * inst_std.unsqueeze(-1) + inst_mean.unsqueeze(-1)
            loss = nn.L1Loss()(pred_denorm * std + mean, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    return model


# ========================= CKA Computation =========================

def compute_layerwise_cka(model_a, model_b, test_x, mean, std,
                          extract_fn_a, extract_fn_b, n_samples=200,
                          n_positions=5000):
    """Compute CKA between two models at each layer.

    Representations are (B, T, N, D). We treat each (t, n) position as a sample
    and D as the feature dimension. This gives manageable CKA computation:
    X ∈ (n_positions, D), Y ∈ (n_positions, D), CKA matrices are (D, D).
    """
    idx = np.random.choice(len(test_x), min(n_samples, len(test_x)), replace=False)
    x_subset = test_x[idx]
    x_norm = normalize_input(x_subset, mean, std)

    reps_a, reps_b = {}, {}

    for start in range(0, len(x_norm), 32):
        bx = torch.FloatTensor(x_norm[start:start+32]).to(DEVICE)
        bx_normed, _, _ = apply_instance_norm(bx)

        batch_reps_a = extract_fn_a(model_a, bx_normed)
        batch_reps_b = extract_fn_b(model_b, bx_normed)

        for key in batch_reps_a:
            # (B, T, N, D) -> (B*T*N, D)
            r = batch_reps_a[key]
            if r.dim() == 4:
                r = r.reshape(-1, r.shape[-1])
            else:
                r = r.reshape(-1, r.shape[-1])
            reps_a.setdefault(key, []).append(r.cpu())
        for key in batch_reps_b:
            r = batch_reps_b[key]
            if r.dim() == 4:
                r = r.reshape(-1, r.shape[-1])
            else:
                r = r.reshape(-1, r.shape[-1])
            reps_b.setdefault(key, []).append(r.cpu())

    cka_scores = {}
    for key in sorted(reps_a.keys()):
        if key in reps_b:
            X = torch.cat(reps_a[key], dim=0)  # (total_positions, D)
            Y = torch.cat(reps_b[key], dim=0)
            # Subsample positions for efficiency
            if X.shape[0] > n_positions:
                perm = torch.randperm(X.shape[0])[:n_positions]
                X = X[perm]
                Y = Y[perm]
            cka_scores[key] = linear_cka(X, Y)
    return cka_scores


# ========================= Main =========================

def main():
    print("=" * 70)
    print("CKA Analysis: How PEFT Methods Change Internal Representations")
    print("=" * 70)

    PAIRS = [(2022, 2023), (2023, 2024), (2022, 2024)]
    TIME_BUDGETS = [24, 168]  # 1d and 7d

    data_cache = {}
    for year in [2022, 2023, 2024]:
        data, mean, std, n_total = load_data_and_scaler(DATASETS[year])
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * 0.2)
        test_data = data[n_train + n_val:]
        test_x, test_y = create_samples(test_data, INPUT_LEN, OUTPUT_LEN)
        data_cache[year] = {"full_data": data, "test_x": test_x, "test_y": test_y,
                            "mean": mean, "std": std}
        print(f"  {year}: test={len(test_x)}")

    all_results = {}

    for hours in TIME_BUDGETS:
        h_label = f"{hours}h" if hours < 24 else f"{hours//24}d"
        print(f"\n{'#'*70}")
        print(f"  TIME BUDGET: {h_label}")
        print(f"{'#'*70}")

        for source_year, target_year in PAIRS:
            pair_key = f"{source_year}_{target_year}_{h_label}"
            print(f"\n{'='*60}")
            print(f"  {source_year} -> {target_year} ({h_label})")
            print(f"{'='*60}")

            src = data_cache[source_year]
            tgt = data_cache[target_year]

            n_steps = hours * 12
            ft_data = tgt["full_data"][:n_steps]
            ft_x, ft_y = create_samples(ft_data, INPUT_LEN, OUTPUT_LEN)
            print(f"  Fine-tune samples: {len(ft_x)}")

            source_model = load_backbone(INSTNORM_CHECKPOINTS[source_year]).to(DEVICE)
            source_model.eval()

            # === Adapt models ===
            adapted = {}

            # PB only
            print("  Training PB only...")
            bb = load_backbone(INSTNORM_CHECKPOINTS[source_year])
            m = STAEformerWithPB(bb)
            m = train_model(m, ["pattern_bank", "node_weights"],
                           ft_x, ft_y, src["mean"], src["std"])
            adapted["pb_only"] = ("pb", m)

            # PB+head
            print("  Training PB+head...")
            bb = load_backbone(INSTNORM_CHECKPOINTS[source_year])
            m = STAEformerWithPB(bb)
            m = train_model(m, ["pattern_bank", "node_weights", "output_proj"],
                           ft_x, ft_y, src["mean"], src["std"])
            adapted["pb+head"] = ("pb", m)

            # emb_only
            print("  Training emb_only...")
            m = load_backbone(INSTNORM_CHECKPOINTS[source_year])
            m = train_model(m, ["adaptive_embedding"],
                           ft_x, ft_y, src["mean"], src["std"])
            adapted["emb_only"] = ("plain", m)

            # full_ft
            print("  Training full_ft...")
            m = load_backbone(INSTNORM_CHECKPOINTS[source_year])
            m = train_model(m, ["encoder", "spatial", "decoder"],
                           ft_x, ft_y, src["mean"], src["std"], lr=0.0005)
            adapted["full_ft"] = ("plain", m)

            # === Compute CKA: source vs each adapted ===
            pair_cka = {}
            for method_name, (model_type, adapted_model) in adapted.items():
                print(f"  CKA source vs {method_name}...", end=" ")
                extract_fn_b = extract_reps_pb if model_type == "pb" else extract_reps_encoder
                cka = compute_layerwise_cka(
                    source_model, adapted_model,
                    tgt["test_x"], src["mean"], src["std"],
                    extract_fn_a=extract_reps_encoder,
                    extract_fn_b=extract_fn_b,
                    n_samples=500
                )
                pair_cka[method_name] = cka
                # One-line summary
                layers = sorted(cka.keys())
                vals = [f"{cka[l]:.3f}" for l in layers]
                print(" | ".join(vals))

            # Cross-method: PB+head vs emb_only
            print(f"  CKA pb+head vs emb_only...", end=" ")
            cross_cka = compute_layerwise_cka(
                adapted["pb+head"][1], adapted["emb_only"][1],
                tgt["test_x"], src["mean"], src["std"],
                extract_fn_a=extract_reps_pb,
                extract_fn_b=extract_reps_encoder,
                n_samples=500
            )
            pair_cka["cross_pb_emb"] = cross_cka
            vals = [f"{cross_cka[l]:.3f}" for l in sorted(cross_cka.keys())]
            print(" | ".join(vals))

            all_results[pair_key] = pair_cka

            del source_model
            for _, (_, m) in adapted.items():
                del m
            torch.cuda.empty_cache()

    # ========================= Summary =========================
    print("\n" + "=" * 70)
    print("SUMMARY: Average CKA across 3 pairs (source vs adapted)")
    print("=" * 70)

    layer_order = ['input_proj', 'after_embedding', 'after_temporal_0', 'after_spatial_0', 'before_output']
    method_names = ["pb_only", "pb+head", "emb_only", "full_ft"]

    for hours in TIME_BUDGETS:
        h_label = f"{hours}h" if hours < 24 else f"{hours//24}d"
        print(f"\n--- {h_label} ---")
        header = f"{'Layer':>20s}"
        for m in method_names:
            header += f" {m:>10s}"
        print(header)
        print("-" * (20 + 11 * len(method_names)))

        for layer in layer_order:
            row = f"{layer:>20s}"
            for method in method_names:
                vals = []
                for s, t in PAIRS:
                    pk = f"{s}_{t}_{h_label}"
                    if pk in all_results and method in all_results[pk]:
                        if layer in all_results[pk][method]:
                            vals.append(all_results[pk][method][layer])
                avg = np.mean(vals) if vals else float('nan')
                row += f" {avg:>10.4f}"
            print(row)

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("CKA ≈ 1.0: representations unchanged from source")
    print("CKA < 1.0: representations changed (lower = more change)")
    print()

    for hours in TIME_BUDGETS:
        h_label = f"{hours}h" if hours < 24 else f"{hours//24}d"
        print(f"--- {h_label} ---")
        for method in method_names:
            min_cka, min_layer = 1.0, ""
            for layer in layer_order:
                vals = []
                for s, t in PAIRS:
                    pk = f"{s}_{t}_{h_label}"
                    if pk in all_results and method in all_results[pk]:
                        if layer in all_results[pk][method]:
                            vals.append(all_results[pk][method][layer])
                if vals:
                    avg = np.mean(vals)
                    if avg < min_cka:
                        min_cka = avg
                        min_layer = layer
            print(f"  {method:>10s}: most change at {min_layer} (CKA={min_cka:.4f})")
        print()

    # Save
    output_path = "eda/concept_drift/cka_analysis_results.json"
    serializable = {}
    for k, v in all_results.items():
        serializable[k] = {}
        for k2, v2 in v.items():
            serializable[k][k2] = {k3: float(v3) for k3, v3 in v2.items()}
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
