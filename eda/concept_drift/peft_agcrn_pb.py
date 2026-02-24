"""PB + pred_head on AGCRN: Model-agnostic verification of Pattern Bank adapter.

AGCRN (NeurIPS 2020) is a GRU-based model with node_embeddings (N, embed_dim).
Completely different architecture from STAEformer (Transformer-based).

Methods:
1. PB only — Pattern Bank on node_embeddings
2. pred_head only — end_conv (prediction head)
3. PB + pred_head joint — both simultaneously
4. emb_only — full node_embeddings
5. full_ft — all parameters

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

from baselines.AGCRN.arch import AGCRN

MODEL_PARAM = {
    "num_nodes": 893, "input_dim": 1, "rnn_units": 64, "output_dim": 1,
    "horizon": 12, "num_layers": 2, "default_graph": True,
    "embed_dim": 10, "cheb_k": 2,
}

INSTNORM_CHECKPOINTS = {}  # Will be filled after training completes

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
K = 8  # Pattern bank size

FINETUNE_HOURS = [3, 6, 12, 24, 72, 168]


class AGCRNWithPB(nn.Module):
    """AGCRN with Pattern Bank adapter on node_embeddings."""
    def __init__(self, backbone, num_nodes=893, embed_dim=10, K=8):
        super().__init__()
        self.backbone = backbone
        self.pattern_bank = nn.Parameter(torch.randn(K, embed_dim) * 0.01)
        self.node_weights = nn.Parameter(torch.zeros(num_nodes, K))

    def forward(self, history_data, future_data, batch_seen, epoch, train, **kwargs):
        # Compute PB delta
        delta = F.softmax(self.node_weights, dim=-1) @ self.pattern_bank  # (N, embed_dim)
        modified_emb = self.backbone.node_embeddings + delta

        # Run AGCRN forward with modified embeddings
        init_state = self.backbone.encoder.init_hidden(history_data.shape[0])
        output, _ = self.backbone.encoder(history_data, init_state, modified_emb)
        output = output[:, -1:, :, :]
        output = self.backbone.end_conv(output)
        output = output.squeeze(-1).reshape(
            -1, self.backbone.horizon, self.backbone.output_dim, self.backbone.num_node)
        output = output.permute(0, 1, 3, 2)
        return {"prediction": output}


def find_checkpoints():
    """Auto-discover AGCRN InstanceNorm checkpoints (only completed trainings with test_metrics.json)."""
    import glob
    ckpts = {}
    for year in [2022, 2023, 2024]:
        pattern = f"checkpoints/ConceptDrift_InstanceNorm_AGCRN/SAN_BERNARDINO_{year}_Q1_*/**/AGCRN_best_val_MAE.pt"
        matches = glob.glob(pattern, recursive=True)
        for m in matches:
            # Only use checkpoints from completed training runs
            ckpt_dir = os.path.dirname(m)
            if os.path.exists(os.path.join(ckpt_dir, "test_metrics.json")):
                ckpts[year] = m
                print(f"  Found {year}: {m}")
                break
        else:
            print(f"  Skipping {year}: no completed training found")
    return ckpts


def load_backbone(ckpt_path):
    backbone = AGCRN(**MODEL_PARAM)
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
        samples_x.append(data[i:i+input_len, :, 0:1])  # AGCRN uses only flow (input_dim=1)
        samples_y.append(data[i+input_len:i+input_len+output_len, :, 0:1])
    return np.array(samples_x), np.array(samples_y)


def apply_instance_norm(bx):
    """Apply instance normalization to flow channel."""
    flow = bx[:, :, :, 0]  # (B, T, N)
    inst_mean = flow.mean(dim=1, keepdim=True)  # (B, 1, N)
    inst_std = flow.std(dim=1, keepdim=True) + 1e-5
    bx_normed = bx.clone()
    bx_normed[:, :, :, 0] = (flow - inst_mean) / inst_std
    return bx_normed, inst_mean, inst_std


def denorm_instance_norm(pred, inst_mean, inst_std):
    return pred * inst_std.unsqueeze(-1) + inst_mean.unsqueeze(-1)


def normalize_input(x, mean, std):
    x_norm = x.copy()
    x_norm[:, :, :, 0] = (x[:, :, :, 0] - mean) / std
    return x_norm


def evaluate(model, test_x, test_y, mean, std, batch_size=64):
    model = model.to(DEVICE)
    model.eval()
    test_x_norm = normalize_input(test_x, mean, std)
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(test_x_norm), batch_size):
            bx = torch.FloatTensor(test_x_norm[i:i+batch_size]).to(DEVICE)
            bx_normed, inst_mean, inst_std = apply_instance_norm(bx)
            out = model(bx_normed, None, 0, 0, False)
            if isinstance(out, dict):
                out = out["prediction"]
            out = denorm_instance_norm(out, inst_mean, inst_std)
            pred = out.cpu().numpy() * std + mean
            all_preds.append(pred)
    preds = np.concatenate(all_preds, axis=0)
    return float(np.mean(np.abs(preds - test_y)))


def train_loop(model, trainable_names, train_x, train_y, mean, std,
               epochs=EPOCHS, lr=0.003):
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
            pred = model(bx_normed, None, 0, 0, True)
            if isinstance(pred, dict):
                pred = pred["prediction"]
            pred = denorm_instance_norm(pred, inst_mean, inst_std)
            loss = nn.L1Loss()(pred * std + mean, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    return model, n_trainable


def main():
    global INSTNORM_CHECKPOINTS
    print("=" * 70)
    print("AGCRN: PB + PRED_HEAD (Model-Agnostic Verification)")
    print("=" * 70)

    # Find checkpoints
    INSTNORM_CHECKPOINTS = find_checkpoints()
    years = sorted(INSTNORM_CHECKPOINTS.keys())
    if len(years) < 2:
        print(f"ERROR: Need at least 2 year checkpoints, found {len(years)}: {years}")
        return

    pairs = [(s, t) for s in years for t in years if s != t]
    print(f"Years: {years}, Pairs: {len(pairs)}")

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

    # Count params
    backbone_tmp = AGCRN(**MODEL_PARAM)
    pb_params = K * MODEL_PARAM["embed_dim"] + MODEL_PARAM["num_nodes"] * K  # P + W
    head_params = sum(p.numel() for n, p in backbone_tmp.named_parameters() if "end_conv" in n)
    emb_params = MODEL_PARAM["num_nodes"] * MODEL_PARAM["embed_dim"]
    total_params = sum(p.numel() for p in backbone_tmp.parameters())
    print(f"\nParam counts: PB={pb_params:,}, head={head_params:,}, emb={emb_params:,}, total={total_params:,}")
    del backbone_tmp

    results = {}

    for source_year, target_year in pairs:
        ckpt_path = INSTNORM_CHECKPOINTS[source_year]
        src = data_cache[source_year]
        tgt = data_cache[target_year]
        pair = f"{source_year}_{target_year}"
        print(f"\n{'='*50} {source_year}->{target_year} {'='*10}")

        # Zero-shot baseline
        backbone = load_backbone(ckpt_path)
        model = AGCRNWithPB(backbone)
        mae = evaluate(model, tgt["test_x"], tgt["test_y"], src["mean"], src["std"])
        results[f"zero_shot_{pair}"] = mae
        print(f"  Zero-shot: {mae:.2f}")
        del model, backbone; torch.cuda.empty_cache()

        for hours in FINETUNE_HOURS:
            n_steps = hours * 12
            ft_data = tgt["full_data"][:n_steps]
            ft_x, ft_y = create_samples(ft_data, INPUT_LEN, OUTPUT_LEN)
            if len(ft_x) < 2:
                print(f"  {hours}h: skip (only {len(ft_x)} samples)")
                continue
            h_label = f"{hours}h" if hours < 24 else f"{hours//24}d"
            print(f"\n  --- {h_label} ({len(ft_x)} samples) ---")

            # 1) PB only
            backbone = load_backbone(ckpt_path)
            model = AGCRNWithPB(backbone)
            model, np1 = train_loop(model, ["pattern_bank", "node_weights"],
                                    ft_x, ft_y, src["mean"], src["std"], lr=0.003)
            mae = evaluate(model, tgt["test_x"], tgt["test_y"], src["mean"], src["std"])
            results[f"pb_{h_label}_{pair}"] = mae
            print(f"    PB only          {mae:.2f}  ({np1:,}p)")
            del model, backbone; torch.cuda.empty_cache()

            # 2) pred_head only
            backbone = load_backbone(ckpt_path)
            model = AGCRNWithPB(backbone)
            model, np2 = train_loop(model, ["end_conv"],
                                    ft_x, ft_y, src["mean"], src["std"], lr=0.003)
            mae = evaluate(model, tgt["test_x"], tgt["test_y"], src["mean"], src["std"])
            results[f"head_{h_label}_{pair}"] = mae
            print(f"    pred_head only   {mae:.2f}  ({np2:,}p)")
            del model, backbone; torch.cuda.empty_cache()

            # 3) PB + pred_head joint
            backbone = load_backbone(ckpt_path)
            model = AGCRNWithPB(backbone)
            model, np3 = train_loop(model, ["pattern_bank", "node_weights", "end_conv"],
                                    ft_x, ft_y, src["mean"], src["std"], lr=0.003)
            mae = evaluate(model, tgt["test_x"], tgt["test_y"], src["mean"], src["std"])
            results[f"pb+head_{h_label}_{pair}"] = mae
            print(f"    PB+head joint    {mae:.2f}  ({np3:,}p)")
            del model, backbone; torch.cuda.empty_cache()

            # 4) emb_only (full node_embeddings)
            backbone = load_backbone(ckpt_path)
            model = AGCRNWithPB(backbone)
            model, np4 = train_loop(model, ["node_embeddings"],
                                    ft_x, ft_y, src["mean"], src["std"], lr=0.003)
            mae = evaluate(model, tgt["test_x"], tgt["test_y"], src["mean"], src["std"])
            results[f"emb_{h_label}_{pair}"] = mae
            print(f"    emb_only         {mae:.2f}  ({np4:,}p)")
            del model, backbone; torch.cuda.empty_cache()

            # 5) full_ft (all parameters)
            backbone = load_backbone(ckpt_path)
            model = AGCRNWithPB(backbone)
            model, np5 = train_loop(model, ["backbone", "pattern_bank", "node_weights"],
                                    ft_x, ft_y, src["mean"], src["std"], lr=0.001)
            mae = evaluate(model, tgt["test_x"], tgt["test_y"], src["mean"], src["std"])
            results[f"full_{h_label}_{pair}"] = mae
            print(f"    full_ft          {mae:.2f}  ({np5:,}p)")
            del model, backbone; torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Average MAE across all pairs")
    print("=" * 70)
    methods = ["pb", "head", "pb+head", "emb", "full"]
    labels = ["PB only", "head only", "PB+head", "emb_only", "full_ft"]

    header = f"{'':>5}"
    for l in labels:
        header += f" {l:>12}"
    print(header)
    print("-" * (5 + 13 * len(labels)))

    for hours in FINETUNE_HOURS:
        h = f"{hours}h" if hours < 24 else f"{hours//24}d"
        row = f"{h:>5}"
        for m in methods:
            vals = [results.get(f"{m}_{h}_{s}_{t}", float('nan')) for s, t in pairs]
            avg = np.nanmean(vals)
            row += f" {avg:>12.2f}"
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
        print(f"  {h}: {best_m} ({best_mae:.2f})")

    # Compare with STAEformer pattern
    print(f"\n{'='*70}")
    print("PB+head vs alternatives: win rate")
    print(f"{'='*70}")
    for hours in FINETUNE_HOURS:
        h = f"{hours}h" if hours < 24 else f"{hours//24}d"
        wins = 0
        total = 0
        for s, t in pairs:
            pb_head = results.get(f"pb+head_{h}_{s}_{t}")
            pb = results.get(f"pb_{h}_{s}_{t}")
            head = results.get(f"head_{h}_{s}_{t}")
            if pb_head is not None and pb is not None and head is not None:
                total += 1
                if pb_head <= min(pb, head):
                    wins += 1
        print(f"  {h}: PB+head wins {wins}/{total} pairs vs max(PB,head)")

    output_path = "eda/concept_drift/peft_agcrn_pb_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
