"""Emb-only LR sweep: Fair comparison with PB.

PB is best at lr=0.001. What's Emb-only's best LR?
Tests lr = 0.0001, 0.0003, 0.001, 0.003

All use RevIN (Instance Norm) trained backbone.
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
DEVICE = "cuda:1"
FINETUNE_EPOCHS = 10
BATCH_SIZE = 16
K = 8

FINETUNE_HOURS = [3, 6, 12, 24, 72, 168]
LR_LIST = [0.0001, 0.0003, 0.001, 0.003]


# ============================================================
# Models
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


class STAEformerEmbeddingOnly(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        N = backbone.encoder.adaptive_embedding.shape[-2]
        d = backbone.encoder.adaptive_embedding.shape[-1]
        self.adp_emb_delta = nn.Parameter(torch.zeros(N, d))

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
            adp_emb = adp_emb + self.adp_emb_delta.unsqueeze(0).unsqueeze(0)
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
# Main
# ============================================================

def main():
    years = [2022, 2023, 2024]
    pairs = [(s, t) for s in years for t in years if s != t]

    print("=" * 70)
    print("EMB-ONLY LR SWEEP: Fair comparison with PB")
    print(f"LRs: {LR_LIST}")
    print("=" * 70)

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

    results = {}

    for source_year, target_year in pairs:
        ckpt_path = INSTNORM_CHECKPOINTS[source_year]
        if not os.path.exists(ckpt_path):
            continue
        src = data_cache[source_year]
        tgt = data_cache[target_year]
        pair = f"{source_year}_{target_year}"
        print(f"\n{'='*50} {source_year}->{target_year} {'='*10}")

        for hours in FINETUNE_HOURS:
            n_steps = hours * 12
            ft_data = tgt["full_data"][:n_steps]
            ft_x, ft_y = create_samples(ft_data, INPUT_LEN, OUTPUT_LEN)
            h_label = f"{hours}h" if hours < 24 else f"{hours//24}d"
            print(f"\n  --- {h_label} ({len(ft_x)} samples) ---")

            # PB with best LR (0.001)
            backbone = load_backbone(ckpt_path)
            model = STAEformerWithPB(backbone, 893, 24, K)
            model, np_pb = finetune_model(model, ["pattern_bank", "node_weights"],
                                           ft_x, ft_y, src["mean"], src["std"], lr=0.001)
            mae_pb = evaluate(model, tgt["test_x"], tgt["test_y"], src["mean"], src["std"])
            results[f"pb_{h_label}_{pair}"] = mae_pb
            print(f"    PB(lr=1e-3):     {mae_pb:.2f}  ({np_pb:,}p)")
            del model, backbone; torch.cuda.empty_cache()

            # Emb-only with each LR
            for lr in LR_LIST:
                backbone = load_backbone(ckpt_path)
                model = STAEformerEmbeddingOnly(backbone)
                model, np_emb = finetune_model(model, ["adp_emb_delta"],
                                                ft_x, ft_y, src["mean"], src["std"], lr=lr)
                mae_emb = evaluate(model, tgt["test_x"], tgt["test_y"], src["mean"], src["std"])
                lr_str = f"{lr:.0e}".replace("+", "").replace("0", "")
                results[f"emb_lr{lr_str}_{h_label}_{pair}"] = mae_emb
                diff = mae_emb - mae_pb
                print(f"    Emb(lr={lr:.0e}): {mae_emb:.2f}  ({np_emb:,}p)  vs PB: {diff:+.2f}")
                del model, backbone; torch.cuda.empty_cache()

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Average MAE across 6 pairs")
    print("=" * 70)

    header = f"{'':>8} {'PB(1e-3)':>10}"
    for lr in LR_LIST:
        header += f" {'Emb('+f'{lr:.0e}'+')':>12}"
    print(header)
    print("-" * (10 + 10 + 13 * len(LR_LIST)))

    for hours in FINETUNE_HOURS:
        h_label = f"{hours}h" if hours < 24 else f"{hours//24}d"
        pb_avg = np.nanmean([results.get(f"pb_{h_label}_{s}_{t}", float('nan'))
                             for s, t in pairs])
        row = f"{h_label:>8} {pb_avg:>10.2f}"
        for lr in LR_LIST:
            lr_str = f"{lr:.0e}".replace("+", "").replace("0", "")
            emb_avg = np.nanmean([results.get(f"emb_lr{lr_str}_{h_label}_{s}_{t}", float('nan'))
                                  for s, t in pairs])
            row += f" {emb_avg:>12.2f}"
        print(row)

    # Best Emb-only LR per time budget
    print(f"\n{'='*70}")
    print("BEST EMB-ONLY vs PB")
    print(f"{'='*70}")
    print(f"{'':>8} {'PB':>8} {'BestEmb':>8} {'BestLR':>10} {'Gap':>8} {'PB wins':>8}")
    print("-" * 55)

    for hours in FINETUNE_HOURS:
        h_label = f"{hours}h" if hours < 24 else f"{hours//24}d"
        pb_avg = np.nanmean([results.get(f"pb_{h_label}_{s}_{t}", float('nan'))
                             for s, t in pairs])
        best_emb = float('inf')
        best_lr = None
        for lr in LR_LIST:
            lr_str = f"{lr:.0e}".replace("+", "").replace("0", "")
            emb_avg = np.nanmean([results.get(f"emb_lr{lr_str}_{h_label}_{s}_{t}", float('nan'))
                                  for s, t in pairs])
            if emb_avg < best_emb:
                best_emb = emb_avg
                best_lr = lr

        # Win rate at best LR
        lr_str = f"{best_lr:.0e}".replace("+", "").replace("0", "")
        wins = sum(1 for s, t in pairs
                   if results.get(f"pb_{h_label}_{s}_{t}", float('inf')) <
                      results.get(f"emb_lr{lr_str}_{h_label}_{s}_{t}", float('inf')))
        print(f"{h_label:>8} {pb_avg:>8.2f} {best_emb:>8.2f} {best_lr:>10.0e} "
              f"{best_emb-pb_avg:>+8.2f} {wins}/6")

    output_path = "eda/concept_drift/peft_emb_lr_sweep_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
