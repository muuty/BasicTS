"""Pattern Bank Adapter on Instance Norm-trained models.

Correct experiment design:
- Backbone: models trained WITH instance norm (InstanceNormRunner)
- PB adapter: fine-tuned with instance norm applied (same as training)
- Comparison: InstanceNorm model alone vs InstanceNorm model + PB

If IN model + PB > IN model alone → PB captures real pattern drift
If IN model + PB ≈ IN model alone → PB was only doing scale correction
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

# InstanceNorm-trained model checkpoints
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
FINETUNE_DAYS = [1, 3, 7]
FINETUNE_EPOCHS = 10
BATCH_SIZE = 16
NUM_PATTERNS = 8


class STAEformerWithPatternBank(nn.Module):
    """Wraps a pre-trained STAEformer with a Pattern Bank adapter."""

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
    """Per-sample instance norm on flow channel (ch0)."""
    flow = bx[:, :, :, 0]
    inst_mean = flow.mean(dim=1, keepdim=True)
    inst_std = flow.std(dim=1, keepdim=True) + 1e-5
    bx_normed = bx.clone()
    bx_normed[:, :, :, 0] = (flow - inst_mean) / inst_std
    return bx_normed, inst_mean, inst_std


def denorm_instance_norm(pred, inst_mean, inst_std):
    """Denormalize from instance norm space."""
    return pred * inst_std.unsqueeze(-1) + inst_mean.unsqueeze(-1)


def evaluate_with_instnorm(model, test_x, test_y, mean, std, stable_indices=None, batch_size=64):
    """Evaluate with instance norm (matching InstanceNormRunner behavior)."""
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
    result = {"MAE": mae}
    if stable_indices is not None:
        per_node_mae = np.mean(np.abs(preds - test_y), axis=(0, 1, 3))
        result["stable_MAE"] = float(per_node_mae[stable_indices].mean())
    return result


def finetune_adapter_with_instnorm(model, train_x, train_y, mean, std, epochs=10, lr=0.001):
    """Fine-tune PB adapter with instance norm applied (matching training)."""
    model = model.to(DEVICE)
    model.train()

    for name, param in model.named_parameters():
        param.requires_grad = ("pattern_bank" in name or "node_weights" in name)

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
    print("PATTERN BANK on INSTANCE NORM-TRAINED MODELS")
    print(f"K={NUM_PATTERNS}, adaptive_dim={MODEL_PARAM['adaptive_embedding_dim']}")
    print("=" * 70)
    print("\nQuestion: Does PB capture real PATTERN drift beyond scale?")

    stable_indices = None
    for path in ["datasets/SAN_BERNARDINO_2022_Q1/stable_functional_indices.npy",
                 "eda/concept_drift/functional_indices.npy"]:
        if os.path.exists(path):
            stable_indices = np.load(path)
            print(f"Loaded {len(stable_indices)} stable functional node indices")
            break

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
            print(f"  WARNING: checkpoint not found for {source_year}: {ckpt_path}")
            continue
        source_cache = data_cache[source_year]

        for target_year in years:
            if target_year == source_year:
                continue

            print(f"\n{'='*70}")
            print(f"SOURCE: {source_year} -> TARGET: {target_year}")
            print(f"{'='*70}")

            target_cache = data_cache[target_year]

            # (a) IN model only (no PB, no fine-tuning)
            print("\n  [IN Model Only - zero-shot]")
            backbone = load_backbone(ckpt_path)
            backbone = backbone.to(DEVICE)
            res = evaluate_with_instnorm(backbone, target_cache["test_x"], target_cache["test_y"],
                                         source_cache["mean"], source_cache["std"], stable_indices)
            results[f"in_only_{source_year}_{target_year}"] = res
            print(f"    MAE={res['MAE']:.4f}, stable={res.get('stable_MAE','N/A'):.4f}")
            del backbone; torch.cuda.empty_cache()

            # Self-year for reference
            if source_year == years[0] and target_year == years[1]:
                print("\n  [IN Model Self-Year Reference]")
                backbone = load_backbone(ckpt_path)
                backbone = backbone.to(DEVICE)
                res_self = evaluate_with_instnorm(backbone, source_cache["test_x"], source_cache["test_y"],
                                                   source_cache["mean"], source_cache["std"], stable_indices)
                results[f"in_self_{source_year}"] = res_self
                print(f"    Self MAE={res_self['MAE']:.4f}")
                del backbone; torch.cuda.empty_cache()

            for n_days in FINETUNE_DAYS:
                n_steps = n_days * STEPS_PER_DAY
                ft_data = target_cache["full_data"][:n_steps]
                ft_x, ft_y = create_samples(ft_data, INPUT_LEN, OUTPUT_LEN)
                print(f"\n  --- {n_days} day(s): {len(ft_x)} samples ---")

                # (b) IN model + PB
                print(f"\n  [IN Model + PB, {n_days}d]")
                model = load_model_with_adapter(ckpt_path)
                model = finetune_adapter_with_instnorm(model, ft_x, ft_y,
                                                       source_cache["mean"], source_cache["std"],
                                                       epochs=FINETUNE_EPOCHS)
                res = evaluate_with_instnorm(model, target_cache["test_x"], target_cache["test_y"],
                                             source_cache["mean"], source_cache["std"], stable_indices)
                results[f"in_pb_{n_days}d_{source_year}_{target_year}"] = res
                print(f"    MAE={res['MAE']:.4f}, stable={res.get('stable_MAE','N/A'):.4f}")
                del model; torch.cuda.empty_cache()

    # === Summary ===
    print("\n" + "=" * 70)
    print("SUMMARY: Does PB improve over Instance Norm?")
    print("=" * 70)

    # Load previous results for comparison
    prev_pb = {}
    if os.path.exists("eda/concept_drift/peft_pattern_bank_results.json"):
        with open("eda/concept_drift/peft_pattern_bank_results.json") as f:
            prev_pb = json.load(f)

    prev_peft = {}
    if os.path.exists("eda/concept_drift/peft_results.json"):
        with open("eda/concept_drift/peft_results.json") as f:
            prev_peft = json.load(f)

    print(f"\n{'Method':<35} {'Params':>12} {'Data':>6} {'Avg MAE':>10} {'Avg Stable':>12}")
    print("-" * 77)

    methods = [
        ("No Adaptation (baseline)", "no_adapt", "-", "0", prev_pb),
        ("IN Model (zero-shot)", "in_only", "0", "0", results),
        ("IN Model + PB 1d", "in_pb_1d", "7.3K(1.6%)", "1d", results),
        ("IN Model + PB 3d", "in_pb_3d", "7.3K(1.6%)", "3d", results),
        ("IN Model + PB 7d", "in_pb_7d", "7.3K(1.6%)", "7d", results),
        ("PB Only (baseline model) 1d", "pb_both_1d", "7.3K(1.6%)", "1d", prev_pb),
        ("PB Only (baseline model) 3d", "pb_both_3d", "7.3K(1.6%)", "3d", prev_pb),
        ("PB Only (baseline model) 7d", "pb_both_7d", "7.3K(1.6%)", "7d", prev_pb),
        ("Old PEFT Emb (baseline) 1d", "peft_emb_1d", "257K(56.8%)", "1d", prev_peft),
        ("Old PEFT Emb (baseline) 3d", "peft_emb_3d", "257K(56.8%)", "3d", prev_peft),
        ("Old PEFT Emb (baseline) 7d", "peft_emb_7d", "257K(56.8%)", "7d", prev_peft),
    ]

    for label, prefix, params, data_req, data_src in methods:
        maes, stable_maes = [], []
        for s in years:
            for t in years:
                if s == t:
                    continue
                key = f"{prefix}_{s}_{t}"
                if key in data_src:
                    maes.append(data_src[key]["MAE"])
                    if "stable_MAE" in data_src[key]:
                        stable_maes.append(data_src[key]["stable_MAE"])
        if maes:
            avg_mae = np.mean(maes)
            avg_stable = np.mean(stable_maes) if stable_maes else float('nan')
            print(f"{label:<35} {params:>12} {data_req:>6} {avg_mae:>10.4f} {avg_stable:>12.4f}")

    # === Per-pair detail ===
    print(f"\n{'='*70}")
    print("PER-PAIR: IN Model + PB vs IN Model Only")
    print(f"{'='*70}")
    hdr = "Train->Test"
    print(f"\n{hdr:<14} {'NoAdapt':>9} {'IN only':>9} {'IN+PB1d':>9} {'IN+PB3d':>9} {'IN+PB7d':>9} {'PBonly7d':>9} {'PEFT7d':>9}")
    print("-" * 80)

    for s in years:
        for t in years:
            if s == t:
                continue
            row = f"{s}->{t}"
            no_a = prev_pb.get(f"no_adapt_{s}_{t}", {}).get("MAE", float('nan'))
            in_only = results.get(f"in_only_{s}_{t}", {}).get("MAE", float('nan'))
            ip1 = results.get(f"in_pb_1d_{s}_{t}", {}).get("MAE", float('nan'))
            ip3 = results.get(f"in_pb_3d_{s}_{t}", {}).get("MAE", float('nan'))
            ip7 = results.get(f"in_pb_7d_{s}_{t}", {}).get("MAE", float('nan'))
            pb7 = prev_pb.get(f"pb_both_7d_{s}_{t}", {}).get("MAE", float('nan'))
            pe7 = prev_peft.get(f"peft_emb_7d_{s}_{t}", {}).get("MAE", float('nan'))
            print(f"{row:<14} {no_a:>9.4f} {in_only:>9.4f} {ip1:>9.4f} {ip3:>9.4f} {ip7:>9.4f} {pb7:>9.4f} {pe7:>9.4f}")

    # === Key analysis ===
    print(f"\n{'='*70}")
    print("KEY ANALYSIS: PB improvement over IN Model alone")
    print(f"{'='*70}")

    in_only_maes = []
    in_pb_maes = {d: [] for d in FINETUNE_DAYS}
    for s in years:
        for t in years:
            if s == t:
                continue
            k = f"in_only_{s}_{t}"
            if k in results:
                in_only_maes.append(results[k]["MAE"])
            for d in FINETUNE_DAYS:
                k = f"in_pb_{d}d_{s}_{t}"
                if k in results:
                    in_pb_maes[d].append(results[k]["MAE"])

    if in_only_maes:
        in_avg = np.mean(in_only_maes)
        print(f"\nIN Model only (zero-shot): {in_avg:.4f}")
        for d in FINETUNE_DAYS:
            if in_pb_maes[d]:
                pb_avg = np.mean(in_pb_maes[d])
                improvement = (in_avg - pb_avg) / in_avg * 100
                print(f"IN + PB {d}d:               {pb_avg:.4f}  ({improvement:+.1f}%)")
        print(f"\nIf improvement is significant → PB captures REAL pattern drift")
        print(f"If improvement is negligible → PB was only doing scale correction")

    output_path = "eda/concept_drift/peft_pb_on_instnorm_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
