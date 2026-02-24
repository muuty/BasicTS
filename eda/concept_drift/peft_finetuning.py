"""PEFT (Parameter-Efficient Fine-Tuning) for cross-year adaptation.

Compares:
  (a) No adaptation: source model on target test set
  (b) Full fine-tune: all parameters updated
  (c) PEFT (embedding only): only adaptive_embedding updated
  (d) Instance norm: reference from existing results

Fine-tuning data: first N days of target year (simulating real deployment).
"""
import sys
import os
import json
import copy
import numpy as np
import torch
import torch.nn as nn
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
DEVICE = "cuda:0"
FINETUNE_DAYS = [1, 3, 7]
FINETUNE_EPOCHS = 10
BATCH_SIZE = 16


def load_model(ckpt_path):
    model = STAEformer(**MODEL_PARAM)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
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
    """Evaluate model on test set. Returns overall MAE and stable MAE."""
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


def finetune(model, train_x, train_y, mean, std, mode="full", epochs=10, lr=None):
    """Fine-tune model.

    mode:
        'full': all parameters
        'embedding': only adaptive_embedding
    """
    model = model.to(DEVICE)
    model.train()

    if mode == "embedding":
        # Freeze everything except adaptive_embedding
        for name, param in model.named_parameters():
            param.requires_grad = "adaptive_embedding" in name
        if lr is None:
            lr = 0.001
    else:
        for param in model.parameters():
            param.requires_grad = True
        if lr is None:
            lr = 0.0001  # Lower lr for full fine-tune to avoid catastrophic forgetting

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"    Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
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
            # De-normalize prediction for MAE loss in raw space
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
    print("PEFT EXPERIMENT: Parameter-Efficient Fine-Tuning for Cross-Year")
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
        print(f"  {year}: {n_total} steps, test={len(test_x)} samples, mean={mean:.2f}, std={std:.2f}")

    results = {}

    for source_year in years:
        ckpt_path = BASELINE_CHECKPOINTS[source_year]
        if not os.path.exists(ckpt_path):
            print(f"\nSkipping source {source_year}: checkpoint not found")
            continue

        source_cache = data_cache[source_year]

        for target_year in years:
            if target_year == source_year:
                continue

            print(f"\n{'='*70}")
            print(f"SOURCE: {source_year} → TARGET: {target_year}")
            print(f"{'='*70}")

            target_cache = data_cache[target_year]

            # (a) No adaptation
            print("\n  [No Adaptation]")
            model = load_model(ckpt_path).to(DEVICE)
            res = evaluate(model, target_cache["test_x"], target_cache["test_y"],
                          source_cache["mean"], source_cache["std"], stable_indices)
            results[f"no_adapt_{source_year}_{target_year}"] = res
            print(f"    MAE={res['MAE']:.4f}, stable={res.get('stable_MAE', 'N/A'):.4f}")
            del model; torch.cuda.empty_cache()

            for n_days in FINETUNE_DAYS:
                n_steps = n_days * STEPS_PER_DAY
                ft_data = target_cache["full_data"][:n_steps]
                ft_x, ft_y = create_samples(ft_data, INPUT_LEN, OUTPUT_LEN)
                print(f"\n  --- {n_days} day(s) fine-tune data: {len(ft_x)} samples ---")

                # (b) Full fine-tune
                print(f"\n  [Full Fine-Tune, {n_days}d]")
                model = load_model(ckpt_path)
                model = finetune(model, ft_x, ft_y,
                                source_cache["mean"], source_cache["std"],
                                mode="full", epochs=FINETUNE_EPOCHS)
                res = evaluate(model, target_cache["test_x"], target_cache["test_y"],
                              source_cache["mean"], source_cache["std"], stable_indices)
                results[f"full_ft_{n_days}d_{source_year}_{target_year}"] = res
                print(f"    MAE={res['MAE']:.4f}, stable={res.get('stable_MAE', 'N/A'):.4f}")
                del model; torch.cuda.empty_cache()

                # (c) PEFT: embedding only
                print(f"\n  [PEFT Embedding, {n_days}d]")
                model = load_model(ckpt_path)
                model = finetune(model, ft_x, ft_y,
                                source_cache["mean"], source_cache["std"],
                                mode="embedding", epochs=FINETUNE_EPOCHS)
                res = evaluate(model, target_cache["test_x"], target_cache["test_y"],
                              source_cache["mean"], source_cache["std"], stable_indices)
                results[f"peft_emb_{n_days}d_{source_year}_{target_year}"] = res
                print(f"    MAE={res['MAE']:.4f}, stable={res.get('stable_MAE', 'N/A'):.4f}")
                del model; torch.cuda.empty_cache()

    # === Summary ===
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Load instance norm reference
    instnorm_results = {}
    existing_path = "eda/concept_drift/cross_year_all_methods_results.json"
    if os.path.exists(existing_path):
        with open(existing_path) as f:
            existing = json.load(f)
        for k, v in existing.items():
            if k.startswith("instance_norm_"):
                instnorm_results[k] = v

    # Print comparison table
    print(f"\n{'Source→Target':<16} {'No Adapt':>10} {'InstNorm':>10} {'PEFT 1d':>10} {'PEFT 3d':>10} {'PEFT 7d':>10} {'Full 1d':>10} {'Full 3d':>10} {'Full 7d':>10}")
    print("-" * 106)

    for source_year in years:
        for target_year in years:
            if source_year == target_year:
                continue

            row = f"{source_year}→{target_year}"
            no_adapt = results.get(f"no_adapt_{source_year}_{target_year}", {}).get("MAE", float('nan'))

            in_key = f"instance_norm_train_{source_year}_test_{target_year}"
            instnorm = instnorm_results.get(in_key, {}).get("MAE", float('nan'))

            vals = [f"{no_adapt:>10.4f}", f"{instnorm:>10.4f}"]
            for n_days in FINETUNE_DAYS:
                peft = results.get(f"peft_emb_{n_days}d_{source_year}_{target_year}", {}).get("MAE", float('nan'))
                vals.append(f"{peft:>10.4f}")
            for n_days in FINETUNE_DAYS:
                full = results.get(f"full_ft_{n_days}d_{source_year}_{target_year}", {}).get("MAE", float('nan'))
                vals.append(f"{full:>10.4f}")

            print(f"{row:<16} {'  '.join(vals)}")

    # Average degradation comparison
    print(f"\n{'Method':<20} {'Avg Cross MAE':>15}")
    print("-" * 37)

    for method_prefix, label in [
        ("no_adapt", "No Adaptation"),
        ("peft_emb_1d", "PEFT 1 day"),
        ("peft_emb_3d", "PEFT 3 days"),
        ("peft_emb_7d", "PEFT 7 days"),
        ("full_ft_1d", "Full FT 1 day"),
        ("full_ft_3d", "Full FT 3 days"),
        ("full_ft_7d", "Full FT 7 days"),
    ]:
        maes = []
        for s in years:
            for t in years:
                if s == t:
                    continue
                key = f"{method_prefix}_{s}_{t}"
                if key in results:
                    maes.append(results[key]["MAE"])
        if maes:
            print(f"{label:<20} {np.mean(maes):>15.4f}")

    # Instance norm average
    in_maes = []
    for s in years:
        for t in years:
            if s == t:
                continue
            key = f"instance_norm_train_{s}_test_{t}"
            if key in instnorm_results:
                in_maes.append(instnorm_results[key]["MAE"])
    if in_maes:
        print(f"{'Instance Norm':<20} {np.mean(in_maes):>15.4f}")

    # Self-year reference
    self_maes = []
    for y in years:
        key = f"baseline_train_{y}_test_{y}"
        if key in existing:
            self_maes.append(existing[key]["MAE"])
    if self_maes:
        print(f"{'Self-Year (oracle)':<20} {np.mean(self_maes):>15.4f}")

    # Save
    output_path = "eda/concept_drift/peft_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
