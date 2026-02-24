"""K sensitivity analysis for Pattern Bank vs LoRA.

Tests K=4,8,16 across all time budgets: 3h, 6h, 12h, 1d, 3d, 7d.
Also runs LoRA with matching rank for fair comparison.

Param counts:
  K=4:  893*4 + 4*24 = 3,668
  K=8:  893*8 + 8*24 = 7,336
  K=16: 893*16 + 16*24 = 14,672
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

K_VALUES = [4, 8, 16]
# hours: 3h, 6h, 12h, 24h(1d), 72h(3d), 168h(7d)
FINETUNE_HOURS = [3, 6, 12, 24, 72, 168]


class STAEformerWithAdapter(nn.Module):
    """Unified adapter: PB (softmax) or LoRA (unconstrained)."""

    def __init__(self, backbone, num_nodes, adaptive_embedding_dim, rank, use_softmax=True):
        super().__init__()
        self.backbone = backbone
        self.use_softmax = use_softmax
        self.A = nn.Parameter(torch.zeros(num_nodes, rank))
        self.B = nn.Parameter(torch.randn(rank, adaptive_embedding_dim) * 0.01)

    def get_adapter_output(self):
        if self.use_softmax:
            return F.softmax(self.A, dim=-1) @ self.B
        return self.A @ self.B

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
        if enc.spatial_embedding_dim > 0:
            spatial_emb = enc.node_emb.expand(batch_size, enc.in_steps, *enc.node_emb.shape)
            features.append(spatial_emb)
        if enc.adaptive_embedding_dim > 0:
            adp_emb = enc.adaptive_embedding.expand(batch_size, *enc.adaptive_embedding.shape)
            adapter_out = self.get_adapter_output()
            adp_emb = adp_emb + adapter_out.unsqueeze(0).unsqueeze(0)
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


def finetune(model, train_x, train_y, mean, std, epochs=10, lr=0.001):
    model = model.to(DEVICE)
    model.train()
    for name, param in model.named_parameters():
        param.requires_grad = (name in ("A", "B"))
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
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
    return model


def run_one(ckpt_path, src_cache, tgt_cache, ft_x, ft_y, K, use_softmax):
    backbone = load_backbone(ckpt_path)
    model = STAEformerWithAdapter(
        backbone, MODEL_PARAM["num_nodes"],
        MODEL_PARAM["adaptive_embedding_dim"], rank=K, use_softmax=use_softmax,
    )
    model = finetune(model, ft_x, ft_y, src_cache["mean"], src_cache["std"], epochs=FINETUNE_EPOCHS)
    mae = evaluate(model, tgt_cache["test_x"], tgt_cache["test_y"], src_cache["mean"], src_cache["std"])
    del model, backbone; torch.cuda.empty_cache()
    return mae


def main():
    years = [2022, 2023, 2024]

    print("=" * 70)
    print("K SENSITIVITY: Pattern Bank vs LoRA")
    print(f"K values: {K_VALUES}")
    print(f"Time budgets: {FINETUNE_HOURS}")
    print("=" * 70)

    # Load data
    data_cache = {}
    for year in years:
        data, mean, std, n_total = load_data_and_scaler(DATASETS[year])
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * 0.2)
        test_data = data[n_train + n_val:]
        test_x, test_y = create_samples(test_data, INPUT_LEN, OUTPUT_LEN)
        data_cache[year] = {"full_data": data, "test_x": test_x, "test_y": test_y, "mean": mean, "std": std}
        print(f"  {year}: test={len(test_x)}, mean={mean:.2f}, std={std:.2f}")

    results = {}

    for source_year in years:
        ckpt_path = INSTNORM_CHECKPOINTS[source_year]
        if not os.path.exists(ckpt_path):
            continue
        src = data_cache[source_year]

        for target_year in years:
            if target_year == source_year:
                continue
            tgt = data_cache[target_year]
            pair = f"{source_year}_{target_year}"
            print(f"\n{'='*50} {source_year}->{target_year} {'='*10}")

            for hours in FINETUNE_HOURS:
                n_steps = hours * 12
                ft_data = tgt["full_data"][:n_steps]
                ft_x, ft_y = create_samples(ft_data, INPUT_LEN, OUTPUT_LEN)
                h_label = f"{hours}h" if hours < 24 else f"{hours//24}d"
                print(f"\n  {h_label} ({len(ft_x)} samples):")

                for K in K_VALUES:
                    n_params = 893 * K + K * 24

                    # PB
                    mae_pb = run_one(ckpt_path, src, tgt, ft_x, ft_y, K, use_softmax=True)
                    results[f"pb_K{K}_{h_label}_{pair}"] = mae_pb

                    # LoRA
                    mae_lo = run_one(ckpt_path, src, tgt, ft_x, ft_y, K, use_softmax=False)
                    results[f"lora_K{K}_{h_label}_{pair}"] = mae_lo

                    diff = mae_lo - mae_pb
                    winner = "PB" if diff > 0 else "LoRA"
                    print(f"    K={K:2d} ({n_params:,}p): PB={mae_pb:.2f}  LoRA={mae_lo:.2f}  "
                          f"diff={diff:+.2f} ({winner})")

    # === Summary tables ===
    print("\n" + "=" * 70)
    print("SUMMARY: Average MAE across 6 cross-year pairs")
    print("=" * 70)

    pairs = [(s, t) for s in years for t in years if s != t]

    # Table: PB by K and hours
    print(f"\n{'':>6}", end="")
    for hours in FINETUNE_HOURS:
        h_label = f"{hours}h" if hours < 24 else f"{hours//24}d"
        print(f" {h_label:>8}", end="")
    print()
    print("-" * (6 + 9 * len(FINETUNE_HOURS)))

    for K in K_VALUES:
        print(f"PB K={K:<2}", end="")
        for hours in FINETUNE_HOURS:
            h_label = f"{hours}h" if hours < 24 else f"{hours//24}d"
            maes = [results.get(f"pb_K{K}_{h_label}_{s}_{t}", float('nan'))
                    for s, t in pairs]
            print(f" {np.nanmean(maes):>8.2f}", end="")
        print()

    print()
    for K in K_VALUES:
        print(f"LR K={K:<2}", end="")
        for hours in FINETUNE_HOURS:
            h_label = f"{hours}h" if hours < 24 else f"{hours//24}d"
            maes = [results.get(f"lora_K{K}_{h_label}_{s}_{t}", float('nan'))
                    for s, t in pairs]
            print(f" {np.nanmean(maes):>8.2f}", end="")
        print()

    # PB-LoRA gap
    print(f"\n{'Gap (LoRA-PB), positive = PB wins':>40}")
    print(f"{'':>6}", end="")
    for hours in FINETUNE_HOURS:
        h_label = f"{hours}h" if hours < 24 else f"{hours//24}d"
        print(f" {h_label:>8}", end="")
    print()
    print("-" * (6 + 9 * len(FINETUNE_HOURS)))

    for K in K_VALUES:
        print(f"K={K:<4}", end="")
        for hours in FINETUNE_HOURS:
            h_label = f"{hours}h" if hours < 24 else f"{hours//24}d"
            pb_maes = [results.get(f"pb_K{K}_{h_label}_{s}_{t}", float('nan')) for s, t in pairs]
            lo_maes = [results.get(f"lora_K{K}_{h_label}_{s}_{t}", float('nan')) for s, t in pairs]
            gap = np.nanmean(lo_maes) - np.nanmean(pb_maes)
            print(f" {gap:>+8.3f}", end="")
        print()

    # PB win rate
    print(f"\n{'PB win rate (out of 6 pairs)':>35}")
    print(f"{'':>6}", end="")
    for hours in FINETUNE_HOURS:
        h_label = f"{hours}h" if hours < 24 else f"{hours//24}d"
        print(f" {h_label:>8}", end="")
    print()
    print("-" * (6 + 9 * len(FINETUNE_HOURS)))

    for K in K_VALUES:
        print(f"K={K:<4}", end="")
        for hours in FINETUNE_HOURS:
            h_label = f"{hours}h" if hours < 24 else f"{hours//24}d"
            wins = 0
            for s, t in pairs:
                pb = results.get(f"pb_K{K}_{h_label}_{s}_{t}", float('nan'))
                lo = results.get(f"lora_K{K}_{h_label}_{s}_{t}", float('nan'))
                if pb < lo:
                    wins += 1
            print(f" {wins:>5}/6  ", end="")
        print()

    output_path = "eda/concept_drift/peft_k_sensitivity_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
