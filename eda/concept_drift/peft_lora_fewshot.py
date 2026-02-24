"""LoRA vs Pattern Bank: Extreme few-shot comparison (sub-1-day).

Tests 3h, 6h, 12h adaptation data — where softmax regularization should matter most.
At 5-min intervals (288 steps/day):
  3h  = 36 steps  → 13 samples
  6h  = 72 steps  → 49 samples
  12h = 144 steps → 121 samples
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
FINETUNE_HOURS = [3, 6, 12]  # sub-1-day
FINETUNE_EPOCHS = 10
BATCH_SIZE = 16
NUM_PATTERNS = 8


class STAEformerWithLoRA(nn.Module):
    def __init__(self, backbone, num_nodes, adaptive_embedding_dim, rank=8):
        super().__init__()
        self.backbone = backbone
        self.lora_A = nn.Parameter(torch.zeros(num_nodes, rank))
        self.lora_B = nn.Parameter(torch.randn(rank, adaptive_embedding_dim) * 0.01)

    def get_adapter_output(self):
        return self.lora_A @ self.lora_B

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


class STAEformerWithPatternBank(nn.Module):
    def __init__(self, backbone, num_nodes, adaptive_embedding_dim, num_patterns=8):
        super().__init__()
        self.backbone = backbone
        self.pattern_bank = nn.Parameter(torch.randn(num_patterns, adaptive_embedding_dim) * 0.01)
        self.node_weights = nn.Parameter(torch.zeros(num_nodes, num_patterns))

    def get_adapter_output(self):
        weights = F.softmax(self.node_weights, dim=-1)
        return weights @ self.pattern_bank

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
    train_data_ch0 = data[:n_train, :, 0]
    mean = float(np.mean(train_data_ch0))
    std = float(np.std(train_data_ch0))
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


def evaluate_with_instnorm(model, test_x, test_y, mean, std, batch_size=64):
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
    mae = float(np.mean(np.abs(preds - test_y)))
    return {"MAE": mae}


def finetune_adapter(model, train_x, train_y, mean, std, adapter_param_names, epochs=10, lr=0.001):
    model = model.to(DEVICE)
    model.train()

    for name, param in model.named_parameters():
        param.requires_grad = any(n in name for n in adapter_param_names)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"    Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    train_x_norm = normalize_input(train_x, mean, std)

    for epoch in range(epochs):
        indices = np.random.permutation(len(train_x_norm))
        epoch_loss = 0
        n_batches = 0
        for i in range(0, len(indices), BATCH_SIZE):
            batch_idx = indices[i:i+BATCH_SIZE]
            bx = torch.FloatTensor(train_x_norm[batch_idx]).to(DEVICE)
            by = torch.FloatTensor(train_y[batch_idx]).to(DEVICE)

            bx_normed, inst_mean, inst_std = apply_instance_norm(bx)
            pred = model(bx_normed, None, 0, 0, True)["prediction"]
            pred = denorm_instance_norm(pred, inst_mean, inst_std)

            pred_raw = pred * std + mean
            loss = nn.L1Loss()(pred_raw, by)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/n_batches:.4f}")

    model.eval()
    return model


def main():
    years = [2022, 2023, 2024]

    print("=" * 70)
    print("LoRA vs PATTERN BANK: EXTREME FEW-SHOT (sub-1-day)")
    print(f"Rank/K={NUM_PATTERNS}, adaptive_dim={MODEL_PARAM['adaptive_embedding_dim']}")
    print("=" * 70)

    for h in FINETUNE_HOURS:
        n_steps = h * 12  # 5-min intervals → 12 per hour
        n_samples = n_steps - INPUT_LEN - OUTPUT_LEN + 1
        print(f"  {h}h = {n_steps} steps → {n_samples} samples")

    print("\nLoading data...")
    data_cache = {}
    for year in years:
        data, mean, std, n_total = load_data_and_scaler(DATASETS[year])
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * 0.2)
        test_start = n_train + n_val
        test_data = data[test_start:]
        test_x, test_y = create_samples(test_data, INPUT_LEN, OUTPUT_LEN)
        data_cache[year] = {
            "full_data": data,
            "test_x": test_x,
            "test_y": test_y,
            "mean": mean,
            "std": std,
        }
        print(f"  {year}: test={len(test_x)} samples, mean={mean:.2f}, std={std:.2f}")

    results = {}

    for source_year in years:
        ckpt_path = INSTNORM_CHECKPOINTS[source_year]
        if not os.path.exists(ckpt_path):
            print(f"  WARNING: checkpoint not found for {source_year}")
            continue
        source_cache = data_cache[source_year]

        for target_year in years:
            if target_year == source_year:
                continue

            print(f"\n{'='*70}")
            print(f"SOURCE: {source_year} -> TARGET: {target_year}")
            print(f"{'='*70}")

            target_cache = data_cache[target_year]

            for n_hours in FINETUNE_HOURS:
                n_steps = n_hours * 12
                ft_data = target_cache["full_data"][:n_steps]
                ft_x, ft_y = create_samples(ft_data, INPUT_LEN, OUTPUT_LEN)
                print(f"\n  --- {n_hours}h: {len(ft_x)} samples ---")

                # === LoRA ===
                print(f"\n  [LoRA, {n_hours}h]")
                backbone = load_backbone(ckpt_path)
                lora_model = STAEformerWithLoRA(
                    backbone,
                    num_nodes=MODEL_PARAM["num_nodes"],
                    adaptive_embedding_dim=MODEL_PARAM["adaptive_embedding_dim"],
                    rank=NUM_PATTERNS,
                )
                lora_model = finetune_adapter(
                    lora_model, ft_x, ft_y,
                    source_cache["mean"], source_cache["std"],
                    adapter_param_names=["lora_A", "lora_B"],
                    epochs=FINETUNE_EPOCHS,
                )
                res = evaluate_with_instnorm(
                    lora_model, target_cache["test_x"], target_cache["test_y"],
                    source_cache["mean"], source_cache["std"],
                )
                results[f"lora_{n_hours}h_{source_year}_{target_year}"] = res
                print(f"    MAE={res['MAE']:.4f}")
                del lora_model, backbone; torch.cuda.empty_cache()

                # === Pattern Bank ===
                print(f"\n  [PB, {n_hours}h]")
                backbone = load_backbone(ckpt_path)
                pb_model = STAEformerWithPatternBank(
                    backbone,
                    num_nodes=MODEL_PARAM["num_nodes"],
                    adaptive_embedding_dim=MODEL_PARAM["adaptive_embedding_dim"],
                    num_patterns=NUM_PATTERNS,
                )
                pb_model = finetune_adapter(
                    pb_model, ft_x, ft_y,
                    source_cache["mean"], source_cache["std"],
                    adapter_param_names=["pattern_bank", "node_weights"],
                    epochs=FINETUNE_EPOCHS,
                )
                res = evaluate_with_instnorm(
                    pb_model, target_cache["test_x"], target_cache["test_y"],
                    source_cache["mean"], source_cache["std"],
                )
                results[f"pb_{n_hours}h_{source_year}_{target_year}"] = res
                print(f"    MAE={res['MAE']:.4f}")
                del pb_model, backbone; torch.cuda.empty_cache()

    # === Summary ===
    print("\n" + "=" * 70)
    print("SUMMARY: LoRA vs Pattern Bank (Few-Shot)")
    print("=" * 70)

    # Load previous results for 1d, 3d, 7d
    prev_results = {}
    if os.path.exists("eda/concept_drift/peft_lora_comparison_results.json"):
        with open("eda/concept_drift/peft_lora_comparison_results.json") as f:
            prev_results = json.load(f)

    prev_in = {}
    if os.path.exists("eda/concept_drift/peft_pb_on_instnorm_results.json"):
        with open("eda/concept_drift/peft_pb_on_instnorm_results.json") as f:
            prev_in = json.load(f)

    all_hours = FINETUNE_HOURS + [24]  # include 1d from prev
    print(f"\n{'Source->Target':<14} {'IN only':>9}", end="")
    for h in all_hours:
        label = f"{h}h" if h < 24 else "1d"
        print(f" {'L'+label:>7} {'P'+label:>7}", end="")
    print()
    print("-" * (14 + 9 + len(all_hours) * 16))

    lora_avgs = {h: [] for h in all_hours}
    pb_avgs = {h: [] for h in all_hours}

    for s in years:
        for t in years:
            if s == t:
                continue
            row = f"{s}->{t}"
            in_only = prev_in.get(f"in_only_{s}_{t}", {}).get("MAE", float('nan'))
            print(f"{row:<14} {in_only:>9.2f}", end="")

            for h in all_hours:
                if h < 24:
                    lora_mae = results.get(f"lora_{h}h_{s}_{t}", {}).get("MAE", float('nan'))
                    pb_mae = results.get(f"pb_{h}h_{s}_{t}", {}).get("MAE", float('nan'))
                else:
                    lora_mae = prev_results.get(f"lora_1d_{s}_{t}", {}).get("MAE", float('nan'))
                    pb_mae = prev_results.get(f"pb_1d_{s}_{t}", {}).get("MAE", float('nan'))
                lora_avgs[h].append(lora_mae)
                pb_avgs[h].append(pb_mae)
                winner = "<" if pb_mae < lora_mae else ">" if lora_mae < pb_mae else "="
                print(f" {lora_mae:>7.2f} {pb_mae:>7.2f}", end="")
            print()

    print("-" * (14 + 9 + len(all_hours) * 16))
    print(f"{'Average':<14} {np.nanmean([prev_in.get(f'in_only_{s}_{t}', {}).get('MAE', float('nan')) for s in years for t in years if s != t]):>9.2f}", end="")
    for h in all_hours:
        print(f" {np.nanmean(lora_avgs[h]):>7.2f} {np.nanmean(pb_avgs[h]):>7.2f}", end="")
    print()

    # === Win rate ===
    print(f"\n{'Hours':<6} {'PB wins':>9} {'LoRA wins':>11} {'Avg LoRA-PB':>13}")
    print("-" * 42)
    for h in all_hours:
        label = f"{h}h" if h < 24 else "1d"
        pb_w = sum(1 for l, p in zip(lora_avgs[h], pb_avgs[h]) if p < l)
        lo_w = sum(1 for l, p in zip(lora_avgs[h], pb_avgs[h]) if l < p)
        diff = np.nanmean(lora_avgs[h]) - np.nanmean(pb_avgs[h])
        print(f"{label:<6} {pb_w:>5}/6    {lo_w:>5}/6     {diff:>+.4f}")

    output_path = "eda/concept_drift/peft_lora_fewshot_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
