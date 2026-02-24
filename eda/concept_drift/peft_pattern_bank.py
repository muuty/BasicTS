"""PEFT with Pattern Bank Adapter for cross-year adaptation.

Pattern Bank: K learnable pattern prototypes (K × d_adaptive)
Node Weights: per-node mixing weights (N × K) with softmax
Adapter output = softmax(node_weights) @ pattern_bank → (N, d_adaptive)
Added as residual to adaptive_embedding before attention layers.

Compares:
  (a) No adaptation (baseline cross-year)
  (b) Pattern Bank adapter (node_weights + pattern_bank learned)
  (c) Pattern Bank adapter (node_weights only, pattern_bank frozen after init)
  (d) Previous PEFT (embedding only) - from saved results
  (e) Instance norm - from saved results
"""
import sys
import os
import json
import copy
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
STEPS_PER_DAY = 288
DEVICE = "cuda:1"
FINETUNE_DAYS = [1, 3, 7]
FINETUNE_EPOCHS = 10
BATCH_SIZE = 16
NUM_PATTERNS = 8


class STAEformerWithPatternBank(nn.Module):
    """Wraps a pre-trained STAEformer with a Pattern Bank adapter.

    The adapter output is added as a residual to the adaptive_embedding
    before attention computation.
    """

    def __init__(self, backbone, num_nodes, adaptive_embedding_dim, num_patterns=8):
        super().__init__()
        self.backbone = backbone

        # Pattern Bank adapter
        self.pattern_bank = nn.Parameter(torch.randn(num_patterns, adaptive_embedding_dim) * 0.01)
        self.node_weights = nn.Parameter(torch.zeros(num_nodes, num_patterns))

    def get_adapter_output(self):
        """(N, adaptive_dim) residual to add to adaptive_embedding."""
        weights = F.softmax(self.node_weights, dim=-1)  # (N, K)
        return weights @ self.pattern_bank               # (N, adaptive_dim)

    def forward(self, history_data, future_data, batch_seen, epoch, train, **kwargs):
        x = history_data
        batch_size = x.shape[0]
        enc = self.backbone.encoder

        # === Embedding (same as encoder but with adapter residual) ===
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
            # Add pattern bank adapter residual
            adapter_out = self.get_adapter_output()  # (N, adaptive_dim)
            adp_emb = adp_emb + adapter_out.unsqueeze(0).unsqueeze(0)  # broadcast (B, T, N, d)
            features.append(adp_emb)

        x = torch.cat(features, dim=-1)

        # === Temporal Attention (from encoder) ===
        for attn in enc.attn_layers_t:
            x = attn(x, dim=1)

        # === Spatial Attention ===
        x = self.backbone.spatial(x, None)

        # === Decoder (output projection) ===
        out = self.backbone.decoder(x)

        return {"prediction": out}


def load_model_with_adapter(ckpt_path, num_patterns=NUM_PATTERNS):
    backbone = STAEformer(**MODEL_PARAM)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    backbone.load_state_dict(ckpt["model_state_dict"])
    model = STAEformerWithPatternBank(
        backbone,
        num_nodes=MODEL_PARAM["num_nodes"],
        adaptive_embedding_dim=MODEL_PARAM["adaptive_embedding_dim"],
        num_patterns=num_patterns,
    )
    return model


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


def evaluate(model, test_x, test_y, mean, std, stable_indices=None, batch_size=64):
    model.eval()
    test_x_norm = normalize_input(test_x, mean, std)
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(test_x_norm), batch_size):
            bx = torch.FloatTensor(test_x_norm[i:i+batch_size]).to(DEVICE)
            out = model(bx, None, 0, 0, False)["prediction"]
            pred = out.cpu().numpy() * std + mean
            all_preds.append(pred)
    preds = np.concatenate(all_preds, axis=0)
    mae = float(np.mean(np.abs(preds - test_y)))
    result = {"MAE": mae}
    if stable_indices is not None:
        per_node_mae = np.mean(np.abs(preds - test_y), axis=(0, 1, 3))
        result["stable_MAE"] = float(per_node_mae[stable_indices].mean())
    return result


def finetune_adapter(model, train_x, train_y, mean, std, mode="both", epochs=10, lr=0.001):
    """Fine-tune pattern bank adapter.

    mode:
        'both': update pattern_bank + node_weights
        'weights_only': update node_weights only (pattern_bank frozen)
    """
    model = model.to(DEVICE)
    model.train()

    # Freeze backbone entirely
    for name, param in model.named_parameters():
        if "pattern_bank" in name or "node_weights" in name:
            if mode == "weights_only" and "pattern_bank" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        else:
            param.requires_grad = False

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

            pred = model(bx, None, 0, 0, True)["prediction"]
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
    print("PEFT with PATTERN BANK ADAPTER")
    print(f"K={NUM_PATTERNS} patterns, adaptive_dim={MODEL_PARAM['adaptive_embedding_dim']}")
    print("=" * 70)

    # Load stable functional indices
    stable_indices = None
    for path in ["datasets/SAN_BERNARDINO_2022_Q1/stable_functional_indices.npy",
                 "eda/concept_drift/functional_indices.npy"]:
        if os.path.exists(path):
            stable_indices = np.load(path)
            print(f"Loaded {len(stable_indices)} stable functional node indices")
            break

    # Pre-load all data
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
        ckpt_path = BASELINE_CHECKPOINTS[source_year]
        if not os.path.exists(ckpt_path):
            continue
        source_cache = data_cache[source_year]

        for target_year in years:
            if target_year == source_year:
                continue

            print(f"\n{'='*70}")
            print(f"SOURCE: {source_year} -> TARGET: {target_year}")
            print(f"{'='*70}")

            target_cache = data_cache[target_year]

            # (a) No adaptation
            print("\n  [No Adaptation]")
            model = load_model_with_adapter(ckpt_path)
            model = model.to(DEVICE)
            res = evaluate(model, target_cache["test_x"], target_cache["test_y"],
                          source_cache["mean"], source_cache["std"], stable_indices)
            results[f"no_adapt_{source_year}_{target_year}"] = res
            print(f"    MAE={res['MAE']:.4f}, stable={res.get('stable_MAE','N/A'):.4f}")
            del model; torch.cuda.empty_cache()

            for n_days in FINETUNE_DAYS:
                n_steps = n_days * STEPS_PER_DAY
                ft_data = target_cache["full_data"][:n_steps]
                ft_x, ft_y = create_samples(ft_data, INPUT_LEN, OUTPUT_LEN)
                print(f"\n  --- {n_days} day(s): {len(ft_x)} samples ---")

                # (b) Pattern Bank: both pattern_bank + node_weights
                print(f"\n  [PatternBank Both, {n_days}d]")
                model = load_model_with_adapter(ckpt_path)
                model = finetune_adapter(model, ft_x, ft_y,
                                        source_cache["mean"], source_cache["std"],
                                        mode="both", epochs=FINETUNE_EPOCHS)
                res = evaluate(model, target_cache["test_x"], target_cache["test_y"],
                              source_cache["mean"], source_cache["std"], stable_indices)
                results[f"pb_both_{n_days}d_{source_year}_{target_year}"] = res
                print(f"    MAE={res['MAE']:.4f}, stable={res.get('stable_MAE','N/A'):.4f}")
                del model; torch.cuda.empty_cache()

                # (c) Pattern Bank: node_weights only
                print(f"\n  [PatternBank WeightsOnly, {n_days}d]")
                model = load_model_with_adapter(ckpt_path)
                model = finetune_adapter(model, ft_x, ft_y,
                                        source_cache["mean"], source_cache["std"],
                                        mode="weights_only", epochs=FINETUNE_EPOCHS)
                res = evaluate(model, target_cache["test_x"], target_cache["test_y"],
                              source_cache["mean"], source_cache["std"], stable_indices)
                results[f"pb_weights_{n_days}d_{source_year}_{target_year}"] = res
                print(f"    MAE={res['MAE']:.4f}, stable={res.get('stable_MAE','N/A'):.4f}")
                del model; torch.cuda.empty_cache()

    # === Summary ===
    print("\n" + "=" * 70)
    print("SUMMARY: Average Cross-Year MAE")
    print("=" * 70)

    # Load reference results
    prev_peft = {}
    if os.path.exists("eda/concept_drift/peft_results.json"):
        with open("eda/concept_drift/peft_results.json") as f:
            prev_peft = json.load(f)

    instnorm_results = {}
    if os.path.exists("eda/concept_drift/cross_year_all_methods_results.json"):
        with open("eda/concept_drift/cross_year_all_methods_results.json") as f:
            existing = json.load(f)
        for k, v in existing.items():
            instnorm_results[k] = v

    print(f"\n{'Method':<30} {'Params':>10} {'Avg MAE':>10} {'Avg Stable':>12}")
    print("-" * 64)

    methods = [
        ("No Adaptation", "no_adapt", "-", results),
        ("Instance Norm", "instance_norm_train_{s}_test_{t}", "-", instnorm_results),
        ("Prev PEFT Emb 1d", "peft_emb_1d", "257K(56.8%)", prev_peft),
        ("Prev PEFT Emb 3d", "peft_emb_3d", "257K(56.8%)", prev_peft),
        ("Prev PEFT Emb 7d", "peft_emb_7d", "257K(56.8%)", prev_peft),
        ("PatternBank Both 1d", "pb_both_1d", "7.3K(1.6%)", results),
        ("PatternBank Both 3d", "pb_both_3d", "7.3K(1.6%)", results),
        ("PatternBank Both 7d", "pb_both_7d", "7.3K(1.6%)", results),
        ("PatternBank Weights 1d", "pb_weights_1d", "7.1K(1.6%)", results),
        ("PatternBank Weights 3d", "pb_weights_3d", "7.1K(1.6%)", results),
        ("PatternBank Weights 7d", "pb_weights_7d", "7.1K(1.6%)", results),
    ]

    for label, prefix, params, data_src in methods:
        maes, stable_maes = [], []
        for s in years:
            for t in years:
                if s == t:
                    continue
                if "{s}" in prefix:
                    key = prefix.format(s=s, t=t)
                else:
                    key = f"{prefix}_{s}_{t}"
                if key in data_src:
                    maes.append(data_src[key]["MAE"])
                    if "stable_MAE" in data_src[key]:
                        stable_maes.append(data_src[key]["stable_MAE"])
        if maes:
            avg_mae = np.mean(maes)
            avg_stable = np.mean(stable_maes) if stable_maes else float('nan')
            print(f"{label:<30} {params:>10} {avg_mae:>10.4f} {avg_stable:>12.4f}")

    # Self-year reference
    self_maes = []
    for y in years:
        key = f"baseline_train_{y}_test_{y}"
        if key in instnorm_results:
            self_maes.append(instnorm_results[key]["MAE"])
    if self_maes:
        print(f"{'Self-Year (oracle)':<30} {'--':>10} {np.mean(self_maes):>10.4f}")

    # === Per-pair comparison table ===
    print(f"\n{'='*70}")
    print("PER-PAIR: PatternBank Both vs Previous PEFT Emb vs Instance Norm")
    print(f"{'='*70}")
    hdr = "Train->Test"
    print(f"\n{hdr:<14} {'NoAdapt':>9} {'InstNorm':>9} {'PB 1d':>9} {'PB 3d':>9} {'PB 7d':>9} {'OldPEFT1d':>9} {'OldPEFT3d':>9} {'OldPEFT7d':>9}")
    print("-" * 95)

    for s in years:
        for t in years:
            if s == t:
                continue
            row = f"{s}->{t}"
            no_a = results.get(f"no_adapt_{s}_{t}", {}).get("MAE", float('nan'))
            instn = instnorm_results.get(f"instance_norm_train_{s}_test_{t}", {}).get("MAE", float('nan'))
            pb1 = results.get(f"pb_both_1d_{s}_{t}", {}).get("MAE", float('nan'))
            pb3 = results.get(f"pb_both_3d_{s}_{t}", {}).get("MAE", float('nan'))
            pb7 = results.get(f"pb_both_7d_{s}_{t}", {}).get("MAE", float('nan'))
            pe1 = prev_peft.get(f"peft_emb_1d_{s}_{t}", {}).get("MAE", float('nan'))
            pe3 = prev_peft.get(f"peft_emb_3d_{s}_{t}", {}).get("MAE", float('nan'))
            pe7 = prev_peft.get(f"peft_emb_7d_{s}_{t}", {}).get("MAE", float('nan'))
            print(f"{row:<14} {no_a:>9.4f} {instn:>9.4f} {pb1:>9.4f} {pb3:>9.4f} {pb7:>9.4f} {pe1:>9.4f} {pe3:>9.4f} {pe7:>9.4f}")

    # Save
    output_path = "eda/concept_drift/peft_pattern_bank_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
