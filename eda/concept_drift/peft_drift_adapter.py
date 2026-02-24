"""DriftAdapter: Model-agnostic representation-level Pattern Bank.

Instead of attaching PB to STAEformer's adaptive_embedding (model-specific),
DriftAdapter operates at the input level before any backbone:

    x_physical (B,T,N,3) -> repr_encoder(frozen) -> +PB(trainable) -> repr_decoder(frozen) -> residual + x_physical

This makes the adapter model-agnostic: works with STAEformer, STGCN, etc.

Experiment: random init encoder as baseline.
Later: compare with SSL-trained encoder.
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
FINETUNE_DAYS = [1, 3, 7]
FINETUNE_EPOCHS = 10
BATCH_SIZE = 16
NUM_PATTERNS = 8
PHYSICAL_DIM = 3  # flow, occ, speed
REPR_DIMS = [8, 16, 32]


class DriftAdapter(nn.Module):
    """Model-agnostic representation-level Pattern Bank.

    Architecture:
        repr_encoder (frozen): physical_dim -> repr_dim
        pattern_bank (trainable): K prototypes of dim repr_dim
        node_weights (trainable): N -> K mixing weights
        repr_decoder (frozen): repr_dim -> physical_dim
        Output: x_physical + repr_decoder(softmax(node_weights) @ pattern_bank)
    """

    def __init__(self, num_nodes, physical_dim=3, repr_dim=16, num_patterns=8):
        super().__init__()
        self.repr_encoder = nn.Linear(physical_dim, repr_dim, bias=False)
        self.repr_decoder = nn.Linear(repr_dim, physical_dim, bias=False)
        # Freeze encoder/decoder
        self.repr_encoder.weight.requires_grad = False
        self.repr_decoder.weight.requires_grad = False

        self.pattern_bank = nn.Parameter(torch.randn(num_patterns, repr_dim) * 0.01)
        self.node_weights = nn.Parameter(torch.zeros(num_nodes, num_patterns))

    def get_adapter_output(self):
        """Compute per-node drift delta in repr space, then decode to physical space."""
        weights = F.softmax(self.node_weights, dim=-1)  # (N, K)
        adapter_repr = weights @ self.pattern_bank       # (N, repr_dim)
        delta = self.repr_decoder(adapter_repr)           # (N, physical_dim)
        return delta  # (N, physical_dim)

    def forward(self, x_physical):
        """
        Args:
            x_physical: (B, T, N, physical_dim) - flow, occ, speed channels
        Returns:
            x_adapted: (B, T, N, physical_dim) - adapted physical features
        """
        delta = self.get_adapter_output()  # (N, physical_dim)
        return x_physical + delta  # broadcast over B, T


class ModelWithDriftAdapter(nn.Module):
    """Wraps any backbone with a DriftAdapter at the input level."""

    def __init__(self, backbone, drift_adapter, physical_dim=3):
        super().__init__()
        self.backbone = backbone
        self.drift_adapter = drift_adapter
        self.physical_dim = physical_dim

    def forward(self, history_data, future_data, batch_seen, epoch, train, **kwargs):
        x_physical = history_data[..., :self.physical_dim]   # (B, T, N, 3)
        x_rest = history_data[..., self.physical_dim:]        # (B, T, N, 2) tod, dow

        x_adapted = self.drift_adapter(x_physical)
        history_adapted = torch.cat([x_adapted, x_rest], dim=-1)

        out = self.backbone(history_adapted, future_data, batch_seen, epoch, train, **kwargs)
        if isinstance(out, torch.Tensor):
            return {"prediction": out}
        return out


def load_model_with_drift_adapter(ckpt_path, repr_dim, num_patterns=NUM_PATTERNS):
    backbone = STAEformer(**MODEL_PARAM)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    backbone.load_state_dict(ckpt["model_state_dict"])

    adapter = DriftAdapter(
        num_nodes=MODEL_PARAM["num_nodes"],
        physical_dim=PHYSICAL_DIM,
        repr_dim=repr_dim,
        num_patterns=num_patterns,
    )
    model = ModelWithDriftAdapter(backbone, adapter, physical_dim=PHYSICAL_DIM)
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


def finetune_drift_adapter(model, train_x, train_y, mean, std, epochs=10, lr=0.001):
    """Fine-tune only DriftAdapter params (pattern_bank + node_weights)."""
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
    print("DRIFT ADAPTER: Model-Agnostic Representation-Level Pattern Bank")
    print(f"K={NUM_PATTERNS}, physical_dim={PHYSICAL_DIM}, repr_dims={REPR_DIMS}")
    print("Encoder: random init (frozen)")
    print("=" * 70)

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

    # Load previous results for comparison
    prev_instnorm_pb = {}
    if os.path.exists("eda/concept_drift/peft_pb_on_instnorm_results.json"):
        with open("eda/concept_drift/peft_pb_on_instnorm_results.json") as f:
            prev_instnorm_pb = json.load(f)

    results = {}

    for repr_dim in REPR_DIMS:
        print(f"\n{'#'*70}")
        print(f"# REPR_DIM = {repr_dim}")
        n_params = NUM_PATTERNS * repr_dim + 893 * NUM_PATTERNS
        print(f"# Trainable: pattern_bank({NUM_PATTERNS}x{repr_dim}) + node_weights(893x{NUM_PATTERNS}) = {n_params:,}")
        print(f"{'#'*70}")

        for source_year in years:
            ckpt_path = INSTNORM_CHECKPOINTS[source_year]
            if not os.path.exists(ckpt_path):
                print(f"  WARNING: checkpoint not found for {source_year}")
                continue
            source_cache = data_cache[source_year]

            for target_year in years:
                if target_year == source_year:
                    continue

                print(f"\n{'='*60}")
                print(f"repr_dim={repr_dim} | SOURCE: {source_year} -> TARGET: {target_year}")
                print(f"{'='*60}")

                target_cache = data_cache[target_year]

                for n_days in FINETUNE_DAYS:
                    n_steps = n_days * STEPS_PER_DAY
                    ft_data = target_cache["full_data"][:n_steps]
                    ft_x, ft_y = create_samples(ft_data, INPUT_LEN, OUTPUT_LEN)
                    print(f"\n  [DriftAdapter repr_dim={repr_dim}, {n_days}d, {len(ft_x)} samples]")

                    model = load_model_with_drift_adapter(ckpt_path, repr_dim)
                    model = finetune_drift_adapter(
                        model, ft_x, ft_y,
                        source_cache["mean"], source_cache["std"],
                        epochs=FINETUNE_EPOCHS,
                    )
                    res = evaluate_with_instnorm(
                        model, target_cache["test_x"], target_cache["test_y"],
                        source_cache["mean"], source_cache["std"],
                    )
                    key = f"da_r{repr_dim}_{n_days}d_{source_year}_{target_year}"
                    results[key] = res
                    print(f"    MAE={res['MAE']:.4f}")
                    del model; torch.cuda.empty_cache()

    # === Summary ===
    print("\n" + "=" * 70)
    print("SUMMARY: DriftAdapter vs Adaptive-Embedding PB")
    print("=" * 70)

    print(f"\n{'Method':<40} {'Params':>10} {'Avg MAE':>10}")
    print("-" * 62)

    # IN only baseline
    in_only_maes = []
    for s in years:
        for t in years:
            if s == t:
                continue
            k = f"in_only_{s}_{t}"
            if k in prev_instnorm_pb:
                in_only_maes.append(prev_instnorm_pb[k]["MAE"])
    if in_only_maes:
        print(f"{'IN Model (zero-shot)':<40} {'0':>10} {np.mean(in_only_maes):>10.4f}")

    # Previous adaptive-emb PB results
    for n_days in FINETUNE_DAYS:
        maes = []
        for s in years:
            for t in years:
                if s == t:
                    continue
                k = f"in_pb_{n_days}d_{s}_{t}"
                if k in prev_instnorm_pb:
                    maes.append(prev_instnorm_pb[k]["MAE"])
        if maes:
            print(f"{'Adaptive-Emb PB ' + str(n_days) + 'd':<40} {'7,336':>10} {np.mean(maes):>10.4f}")

    # DriftAdapter results
    for repr_dim in REPR_DIMS:
        n_params = NUM_PATTERNS * repr_dim + 893 * NUM_PATTERNS
        for n_days in FINETUNE_DAYS:
            maes = []
            for s in years:
                for t in years:
                    if s == t:
                        continue
                    k = f"da_r{repr_dim}_{n_days}d_{s}_{t}"
                    if k in results:
                        maes.append(results[k]["MAE"])
            if maes:
                label = f"DriftAdapter r={repr_dim} {n_days}d"
                print(f"{label:<40} {n_params:>10,} {np.mean(maes):>10.4f}")

    # Per-pair detail
    print(f"\n{'='*70}")
    print("PER-PAIR DETAIL (7d)")
    print(f"{'='*70}")

    hdr = "Train->Test"
    cols = ["IN only"]
    for rd in REPR_DIMS:
        cols.append(f"DA r={rd}")
    cols.append("AE-PB")
    header = f"{hdr:<14}" + "".join(f"{c:>12}" for c in cols)
    print(f"\n{header}")
    print("-" * (14 + 12 * len(cols)))

    for s in years:
        for t in years:
            if s == t:
                continue
            row = f"{s}->{t}"
            vals = []
            k = f"in_only_{s}_{t}"
            vals.append(prev_instnorm_pb.get(k, {}).get("MAE", float('nan')))
            for rd in REPR_DIMS:
                k = f"da_r{rd}_7d_{s}_{t}"
                vals.append(results.get(k, {}).get("MAE", float('nan')))
            k = f"in_pb_7d_{s}_{t}"
            vals.append(prev_instnorm_pb.get(k, {}).get("MAE", float('nan')))
            print(f"{row:<14}" + "".join(f"{v:>12.4f}" for v in vals))

    # Save
    output_path = "eda/concept_drift/peft_drift_adapter_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
