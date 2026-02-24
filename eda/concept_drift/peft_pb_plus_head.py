"""PB + pred_head: Can combining PB with prediction head get best of both worlds?

Methods:
1. PB only (7.3K) — baseline
2. pred_head only (14K) — baseline
3. PB + pred_head joint (21K) — train both simultaneously
4. Sequential: pred_head first → PB (21K) — avoid gradient conflict
5. Sequential: PB first → pred_head (21K) — reverse order

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

FINETUNE_HOURS = [3, 6, 12, 24, 72, 168]


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


def train_loop(model, trainable_names, train_x, train_y, mean, std,
               epochs=EPOCHS, lr=0.001):
    """One round of training. Returns (model, n_trainable)."""
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


def main():
    years = [2022, 2023, 2024]
    pairs = [(s, t) for s in years for t in years if s != t]

    print("=" * 70)
    print("PB + PRED_HEAD: Joint vs Sequential")
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

            # 1) PB only
            backbone = load_backbone(ckpt_path)
            model = STAEformerWithPB(backbone)
            model, np1 = train_loop(model, ["pattern_bank", "node_weights"],
                                    ft_x, ft_y, src["mean"], src["std"], lr=0.001)
            mae = evaluate(model, tgt["test_x"], tgt["test_y"], src["mean"], src["std"])
            results[f"pb_{h_label}_{pair}"] = mae
            print(f"    PB only          {mae:.2f}  ({np1:,}p)")
            del model, backbone; torch.cuda.empty_cache()

            # 2) pred_head only
            backbone = load_backbone(ckpt_path)
            model = STAEformerWithPB(backbone)  # use PB wrapper but only train decoder
            model, np2 = train_loop(model, ["decoder"],
                                    ft_x, ft_y, src["mean"], src["std"], lr=0.001)
            mae = evaluate(model, tgt["test_x"], tgt["test_y"], src["mean"], src["std"])
            results[f"head_{h_label}_{pair}"] = mae
            print(f"    pred_head only   {mae:.2f}  ({np2:,}p)")
            del model, backbone; torch.cuda.empty_cache()

            # 3) PB + pred_head joint
            backbone = load_backbone(ckpt_path)
            model = STAEformerWithPB(backbone)
            model, np3 = train_loop(model, ["pattern_bank", "node_weights", "decoder"],
                                    ft_x, ft_y, src["mean"], src["std"], lr=0.001)
            mae = evaluate(model, tgt["test_x"], tgt["test_y"], src["mean"], src["std"])
            results[f"pb+head_{h_label}_{pair}"] = mae
            print(f"    PB+head joint    {mae:.2f}  ({np3:,}p)")
            del model, backbone; torch.cuda.empty_cache()

            # 4) Sequential: head first (5ep) → PB (5ep)
            backbone = load_backbone(ckpt_path)
            model = STAEformerWithPB(backbone)
            model, _ = train_loop(model, ["decoder"],
                                  ft_x, ft_y, src["mean"], src["std"],
                                  epochs=5, lr=0.001)
            model, np4 = train_loop(model, ["pattern_bank", "node_weights"],
                                    ft_x, ft_y, src["mean"], src["std"],
                                    epochs=5, lr=0.001)
            mae = evaluate(model, tgt["test_x"], tgt["test_y"], src["mean"], src["std"])
            results[f"seq_h2pb_{h_label}_{pair}"] = mae
            print(f"    seq head→PB      {mae:.2f}  ({np4:,}p)")
            del model, backbone; torch.cuda.empty_cache()

            # 5) Sequential: PB first (5ep) → head (5ep)
            backbone = load_backbone(ckpt_path)
            model = STAEformerWithPB(backbone)
            model, _ = train_loop(model, ["pattern_bank", "node_weights"],
                                  ft_x, ft_y, src["mean"], src["std"],
                                  epochs=5, lr=0.001)
            model, np5 = train_loop(model, ["decoder"],
                                    ft_x, ft_y, src["mean"], src["std"],
                                    epochs=5, lr=0.001)
            mae = evaluate(model, tgt["test_x"], tgt["test_y"], src["mean"], src["std"])
            results[f"seq_pb2h_{h_label}_{pair}"] = mae
            print(f"    seq PB→head      {mae:.2f}  ({np5:,}p)")
            del model, backbone; torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Average MAE across 6 pairs")
    print("=" * 70)
    methods = ["pb", "head", "pb+head", "seq_h2pb", "seq_pb2h"]
    labels = ["PB only", "head only", "PB+head joint", "seq head→PB", "seq PB→head"]

    header = f"{'':>5}"
    for l in labels:
        header += f" {l:>14}"
    print(header)
    print("-" * (5 + 15 * len(labels)))

    for hours in FINETUNE_HOURS:
        h = f"{hours}h" if hours < 24 else f"{hours//24}d"
        row = f"{h:>5}"
        for m in methods:
            avg = np.nanmean([results.get(f"{m}_{h}_{s}_{t}", float('nan'))
                              for s, t in pairs])
            row += f" {avg:>14.2f}"
        print(row)

    # Best method per time budget
    print(f"\n{'='*70}")
    print("BEST METHOD per time budget")
    print(f"{'='*70}")
    for hours in FINETUNE_HOURS:
        h = f"{hours}h" if hours < 24 else f"{hours//24}d"
        best_mae = float('inf')
        best_m = ""
        for m, l in zip(methods, labels):
            avg = np.nanmean([results.get(f"{m}_{h}_{s}_{t}", float('nan'))
                              for s, t in pairs])
            if avg < best_mae:
                best_mae = avg
                best_m = l
        # Win counts
        print(f"  {h}: {best_m} ({best_mae:.2f})")

    output_path = "eda/concept_drift/peft_pb_plus_head_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
