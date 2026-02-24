"""DriftAdapter with Learned (Nonlinear Autoencoder) repr encoder/decoder.

Compared to peft_drift_adapter.py (random init frozen encoder/decoder),
this version pre-trains the encoder/decoder as a nonlinear autoencoder
on source year data, then freezes them for PB fine-tuning.

The hypothesis: learned decoder can meaningfully interpret PB deltas
in repr space, whereas random decoder cannot.
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
PHYSICAL_DIM = 3
REPR_DIM = 16  # Focus on single dim, random showed dim doesn't matter much
AE_PRETRAIN_EPOCHS = 30
AE_PRETRAIN_LR = 0.001
AE_PRETRAIN_BATCH = 4096
AE_DATA_RANGE = 26280  # 3 months (not full year)


class NonlinearAE(nn.Module):
    """Nonlinear autoencoder for physical features."""

    def __init__(self, physical_dim=3, repr_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(physical_dim, repr_dim),
            nn.ReLU(),
            nn.Linear(repr_dim, repr_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(repr_dim, repr_dim),
            nn.ReLU(),
            nn.Linear(repr_dim, physical_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


class DriftAdapterLearned(nn.Module):
    """DriftAdapter with pre-trained encoder/decoder (frozen)."""

    def __init__(self, encoder, decoder, num_nodes, repr_dim=16, num_patterns=8):
        super().__init__()
        self.encoder = encoder  # frozen
        self.decoder = decoder  # frozen
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False

        self.pattern_bank = nn.Parameter(torch.randn(num_patterns, repr_dim) * 0.01)
        self.node_weights = nn.Parameter(torch.zeros(num_nodes, num_patterns))

    def get_adapter_output(self):
        weights = F.softmax(self.node_weights, dim=-1)  # (N, K)
        adapter_repr = weights @ self.pattern_bank       # (N, repr_dim)
        delta = self.decoder(adapter_repr)                # (N, physical_dim)
        return delta

    def forward(self, x_physical):
        delta = self.get_adapter_output()  # (N, physical_dim)
        return x_physical + delta


class ModelWithDriftAdapter(nn.Module):
    def __init__(self, backbone, drift_adapter, physical_dim=3):
        super().__init__()
        self.backbone = backbone
        self.drift_adapter = drift_adapter
        self.physical_dim = physical_dim

    def forward(self, history_data, future_data, batch_seen, epoch, train, **kwargs):
        x_physical = history_data[..., :self.physical_dim]
        x_rest = history_data[..., self.physical_dim:]
        x_adapted = self.drift_adapter(x_physical)
        history_adapted = torch.cat([x_adapted, x_rest], dim=-1)
        out = self.backbone(history_adapted, future_data, batch_seen, epoch, train, **kwargs)
        if isinstance(out, torch.Tensor):
            return {"prediction": out}
        return out


def pretrain_autoencoder(data, mean, std, repr_dim=REPR_DIM):
    """Pre-train autoencoder on source year physical features."""
    ae = NonlinearAE(PHYSICAL_DIM, repr_dim).to(DEVICE)
    optimizer = torch.optim.Adam(ae.parameters(), lr=AE_PRETRAIN_LR)

    # Prepare data: use limited range, normalize flow, subsample
    n_steps = min(AE_DATA_RANGE, data.shape[0])
    train_data = data[:n_steps].copy()  # (T, N, 5)
    physical = train_data[:, :, :PHYSICAL_DIM]  # (T, N, 3)
    physical[:, :, 0] = (physical[:, :, 0] - mean) / std
    flat = physical.reshape(-1, PHYSICAL_DIM)  # (T*N, 3)
    # Subsample: 3-dim AE doesn't need millions of samples
    max_samples = 500000
    if len(flat) > max_samples:
        idx = np.random.choice(len(flat), max_samples, replace=False)
        flat = flat[idx]
    flat_tensor = torch.FloatTensor(flat)
    print(f"    AE training samples: {len(flat_tensor):,}")

    dataset = torch.utils.data.TensorDataset(flat_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=AE_PRETRAIN_BATCH, shuffle=True)

    ae.train()
    for epoch in range(AE_PRETRAIN_EPOCHS):
        total_loss = 0
        n_batches = 0
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            x_hat, _ = ae(batch)
            loss = F.mse_loss(x_hat, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    AE Epoch {epoch+1}/{AE_PRETRAIN_EPOCHS}, Recon Loss: {total_loss/n_batches:.6f}")

    ae.eval()
    return ae


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


def finetune_drift_adapter(model, train_x, train_y, mean, std, epochs=10, lr=0.001):
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
    print("DRIFT ADAPTER with LEARNED AUTOENCODER")
    print(f"K={NUM_PATTERNS}, physical_dim={PHYSICAL_DIM}, repr_dim={REPR_DIM}")
    print(f"AE pre-train: {AE_PRETRAIN_EPOCHS} epochs, lr={AE_PRETRAIN_LR}")
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
    prev_random = {}
    if os.path.exists("eda/concept_drift/peft_drift_adapter_results.json"):
        with open("eda/concept_drift/peft_drift_adapter_results.json") as f:
            prev_random = json.load(f)

    prev_instnorm_pb = {}
    if os.path.exists("eda/concept_drift/peft_pb_on_instnorm_results.json"):
        with open("eda/concept_drift/peft_pb_on_instnorm_results.json") as f:
            prev_instnorm_pb = json.load(f)

    results = {}

    # Pre-train autoencoder per source year
    print("\n" + "=" * 70)
    print("PHASE 1: Pre-train Autoencoders on Source Year Data")
    print("=" * 70)

    ae_models = {}
    for year in years:
        print(f"\n  Pre-training AE for {year}...")
        ae = pretrain_autoencoder(
            data_cache[year]["full_data"],
            data_cache[year]["mean"],
            data_cache[year]["std"],
            repr_dim=REPR_DIM,
        )
        ae_models[year] = ae

        # Evaluate reconstruction quality
        test_physical = data_cache[year]["test_x"][:100, :, :, :PHYSICAL_DIM].copy()
        test_physical[:, :, :, 0] = (test_physical[:, :, :, 0] - data_cache[year]["mean"]) / data_cache[year]["std"]
        flat = torch.FloatTensor(test_physical.reshape(-1, PHYSICAL_DIM)).to(DEVICE)
        with torch.no_grad():
            x_hat, _ = ae(flat)
            recon_mse = F.mse_loss(x_hat, flat).item()
        print(f"    Test recon MSE: {recon_mse:.6f}")

    # Phase 2: Cross-year experiments
    print("\n" + "=" * 70)
    print("PHASE 2: Cross-Year DriftAdapter with Learned AE")
    print("=" * 70)

    for source_year in years:
        ckpt_path = INSTNORM_CHECKPOINTS[source_year]
        if not os.path.exists(ckpt_path):
            print(f"  WARNING: checkpoint not found for {source_year}")
            continue
        source_cache = data_cache[source_year]
        ae = ae_models[source_year]

        for target_year in years:
            if target_year == source_year:
                continue

            print(f"\n{'='*60}")
            print(f"SOURCE: {source_year} -> TARGET: {target_year}")
            print(f"{'='*60}")

            target_cache = data_cache[target_year]

            for n_days in FINETUNE_DAYS:
                n_steps = n_days * STEPS_PER_DAY
                ft_data = target_cache["full_data"][:n_steps]
                ft_x, ft_y = create_samples(ft_data, INPUT_LEN, OUTPUT_LEN)
                print(f"\n  [Learned DA, {n_days}d, {len(ft_x)} samples]")

                # Build model with learned AE
                backbone = STAEformer(**MODEL_PARAM)
                ckpt = torch.load(ckpt_path, map_location="cpu")
                backbone.load_state_dict(ckpt["model_state_dict"])

                # Clone AE encoder/decoder for this run
                import copy
                encoder_copy = copy.deepcopy(ae.encoder)
                decoder_copy = copy.deepcopy(ae.decoder)

                adapter = DriftAdapterLearned(
                    encoder=encoder_copy,
                    decoder=decoder_copy,
                    num_nodes=MODEL_PARAM["num_nodes"],
                    repr_dim=REPR_DIM,
                    num_patterns=NUM_PATTERNS,
                )
                model = ModelWithDriftAdapter(backbone, adapter, PHYSICAL_DIM)

                model = finetune_drift_adapter(
                    model, ft_x, ft_y,
                    source_cache["mean"], source_cache["std"],
                    epochs=FINETUNE_EPOCHS,
                )
                res = evaluate_with_instnorm(
                    model, target_cache["test_x"], target_cache["test_y"],
                    source_cache["mean"], source_cache["std"],
                )
                key = f"learned_{n_days}d_{source_year}_{target_year}"
                results[key] = res
                print(f"    MAE={res['MAE']:.4f}")
                del model, backbone; torch.cuda.empty_cache()

    # === Summary ===
    print("\n" + "=" * 70)
    print("SUMMARY: Learned AE vs Random Init vs Adaptive-Emb PB")
    print("=" * 70)

    print(f"\n{'Method':<40} {'Params':>10} {'Avg MAE':>10}")
    print("-" * 62)

    # IN only
    in_only_maes = []
    for s in years:
        for t in years:
            if s == t: continue
            k = f"in_only_{s}_{t}"
            if k in prev_instnorm_pb:
                in_only_maes.append(prev_instnorm_pb[k]["MAE"])
    if in_only_maes:
        print(f"{'IN Model (zero-shot)':<40} {'0':>10} {np.mean(in_only_maes):>10.4f}")

    # Random DA r=16
    for n_days in FINETUNE_DAYS:
        maes = []
        for s in years:
            for t in years:
                if s == t: continue
                k = f"da_r16_{n_days}d_{s}_{t}"
                if k in prev_random:
                    maes.append(prev_random[k]["MAE"])
        if maes:
            print(f"{'Random DA r=16 ' + str(n_days) + 'd':<40} {'7,272':>10} {np.mean(maes):>10.4f}")

    # Learned DA
    for n_days in FINETUNE_DAYS:
        maes = []
        for s in years:
            for t in years:
                if s == t: continue
                k = f"learned_{n_days}d_{s}_{t}"
                if k in results:
                    maes.append(results[k]["MAE"])
        if maes:
            n_pb = NUM_PATTERNS * REPR_DIM + 893 * NUM_PATTERNS
            print(f"{'Learned DA r=16 ' + str(n_days) + 'd':<40} {n_pb:>10,} {np.mean(maes):>10.4f}")

    # AE-PB
    for n_days in FINETUNE_DAYS:
        maes = []
        for s in years:
            for t in years:
                if s == t: continue
                k = f"in_pb_{n_days}d_{s}_{t}"
                if k in prev_instnorm_pb:
                    maes.append(prev_instnorm_pb[k]["MAE"])
        if maes:
            print(f"{'Adaptive-Emb PB ' + str(n_days) + 'd':<40} {'7,336':>10} {np.mean(maes):>10.4f}")

    # Per-pair detail (7d)
    print(f"\n{'='*70}")
    print("PER-PAIR DETAIL (7d)")
    print(f"{'='*70}")

    hdr = "Train->Test"
    cols = ["IN only", "Random DA", "Learned DA", "AE-PB"]
    header = f"{hdr:<14}" + "".join(f"{c:>12}" for c in cols)
    print(f"\n{header}")
    print("-" * (14 + 12 * len(cols)))

    for s in years:
        for t in years:
            if s == t: continue
            row = f"{s}->{t}"
            vals = [
                prev_instnorm_pb.get(f"in_only_{s}_{t}", {}).get("MAE", float('nan')),
                prev_random.get(f"da_r16_7d_{s}_{t}", {}).get("MAE", float('nan')),
                results.get(f"learned_7d_{s}_{t}", {}).get("MAE", float('nan')),
                prev_instnorm_pb.get(f"in_pb_7d_{s}_{t}", {}).get("MAE", float('nan')),
            ]
            print(f"{row:<14}" + "".join(f"{v:>12.4f}" for v in vals))

    # Save
    output_path = "eda/concept_drift/peft_drift_adapter_learned_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
