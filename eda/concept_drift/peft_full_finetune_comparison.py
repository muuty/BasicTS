"""Full fine-tuning vs PB: Does parameter efficiency matter in few-shot?

Compares:
1. PB (ours): 7,336 params — prototype-constrained
2. Full fine-tune: 452K params — all model parameters
3. RevIN only: 0 params — zero-shot baseline

All use RevIN (Instance Norm) trained backbone.
Key question: With only 3h~1d of data, does full fine-tuning overfit?
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

FINETUNE_HOURS = [3, 6, 12, 24, 72, 168]


# ============================================================
# Model wrapper for PB
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


def evaluate(model, test_x, test_y, mean, std, batch_size=64):
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
    preds = np.concatenate(all_preds, axis=0)
    return float(np.mean(np.abs(preds - test_y)))


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
        epoch_loss, n_batches = 0, 0
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
            epoch_loss += loss.item()
            n_batches += 1
    model.eval()
    return model, n_trainable


# ============================================================
# Main
# ============================================================

def main():
    years = [2022, 2023, 2024]
    pairs = [(s, t) for s in years for t in years if s != t]

    print("=" * 70)
    print("FULL FINE-TUNING vs PB: Parameter efficiency in few-shot")
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
        print(f"  {year}: test={len(test_x)}, mean={mean:.2f}, std={std:.2f}")

    results = {}

    for source_year, target_year in pairs:
        ckpt_path = INSTNORM_CHECKPOINTS[source_year]
        if not os.path.exists(ckpt_path):
            continue
        src = data_cache[source_year]
        tgt = data_cache[target_year]
        pair = f"{source_year}_{target_year}"
        print(f"\n{'='*50} {source_year}->{target_year} {'='*10}")

        # RevIN only (zero-shot)
        backbone = load_backbone(ckpt_path).to(DEVICE)
        mae_revin = evaluate(backbone, tgt["test_x"], tgt["test_y"], src["mean"], src["std"])
        results[f"revin_{pair}"] = mae_revin
        print(f"  RevIN only: {mae_revin:.2f}")
        del backbone; torch.cuda.empty_cache()

        for hours in FINETUNE_HOURS:
            n_steps = hours * 12
            ft_data = tgt["full_data"][:n_steps]
            ft_x, ft_y = create_samples(ft_data, INPUT_LEN, OUTPUT_LEN)
            h_label = f"{hours}h" if hours < 24 else f"{hours//24}d"
            n_samples = len(ft_x)
            print(f"\n  --- {h_label} ({n_samples} samples) ---")

            # (1) PB (ours)
            backbone = load_backbone(ckpt_path)
            model = STAEformerWithPB(backbone, MODEL_PARAM["num_nodes"],
                                      MODEL_PARAM["adaptive_embedding_dim"], K)
            model, np_pb = finetune_model(model, ["pattern_bank", "node_weights"],
                                           ft_x, ft_y, src["mean"], src["std"])
            mae_pb = evaluate(model, tgt["test_x"], tgt["test_y"], src["mean"], src["std"])
            results[f"pb_{h_label}_{pair}"] = mae_pb
            print(f"    PB(K={K}):        {mae_pb:.2f}  ({np_pb:,} params)")
            del model, backbone; torch.cuda.empty_cache()

            # (2) Full fine-tune (all params, lower LR to mitigate overfitting)
            backbone = load_backbone(ckpt_path)
            backbone, np_full = finetune_model(backbone, [""],  # "" matches all names
                                                ft_x, ft_y, src["mean"], src["std"],
                                                lr=0.0001)  # lower LR for full model
            mae_full = evaluate(backbone, tgt["test_x"], tgt["test_y"], src["mean"], src["std"])
            results[f"full_{h_label}_{pair}"] = mae_full
            print(f"    Full FT:         {mae_full:.2f}  ({np_full:,} params)")
            del backbone; torch.cuda.empty_cache()

            # Gap
            diff = mae_full - mae_pb
            ratio = np_full / np_pb
            print(f"    Full-PB gap: {diff:+.2f}  (Full has {ratio:.0f}x more params)")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Average MAE across 6 cross-year pairs")
    print("=" * 70)

    revin_avg = np.mean([results[f"revin_{s}_{t}"] for s, t in pairs
                         if f"revin_{s}_{t}" in results])
    print(f"\n  RevIN only (zero-shot): {revin_avg:.2f}")

    print(f"\n{'':>8} {'PB(7.3K)':>10} {'Full(452K)':>12} │ {'Full-PB':>8} {'PB wins':>8}")
    print("-" * 55)
    for hours in FINETUNE_HOURS:
        h_label = f"{hours}h" if hours < 24 else f"{hours//24}d"
        pb_maes = [results.get(f"pb_{h_label}_{s}_{t}", float('nan')) for s, t in pairs]
        full_maes = [results.get(f"full_{h_label}_{s}_{t}", float('nan')) for s, t in pairs]
        pb_avg = np.nanmean(pb_maes)
        full_avg = np.nanmean(full_maes)
        wins = sum(1 for p, f in zip(pb_maes, full_maes) if p < f)
        print(f"{h_label:>8} {pb_avg:>10.2f} {full_avg:>12.2f} │ {full_avg-pb_avg:>+8.2f} {wins}/6")

    # Per-pair detail
    print(f"\n{'='*70}")
    print("PER-PAIR DETAIL")
    print(f"{'='*70}")
    print(f"\n{'Pair':<12} {'RevIN':>7} │", end="")
    for hours in FINETUNE_HOURS:
        h = f"{hours}h" if hours < 24 else f"{hours//24}d"
        print(f"  PB{h:>3} Full{h:>3}", end="")
    print()
    print("-" * (22 + 14 * len(FINETUNE_HOURS)))

    for s, t in pairs:
        pair = f"{s}_{t}"
        rev = results.get(f"revin_{pair}", float('nan'))
        print(f"{s}->{t:<5} {rev:>7.2f} │", end="")
        for hours in FINETUNE_HOURS:
            h_label = f"{hours}h" if hours < 24 else f"{hours//24}d"
            pb = results.get(f"pb_{h_label}_{pair}", float('nan'))
            full = results.get(f"full_{h_label}_{pair}", float('nan'))
            print(f" {pb:>6.2f} {full:>6.2f}", end="")
        print()

    output_path = "eda/concept_drift/peft_full_finetune_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
