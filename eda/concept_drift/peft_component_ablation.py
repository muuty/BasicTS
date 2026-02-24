"""Component-wise ablation: Is node embedding the key to drift adaptation?

Model-agnostic comparison:
1. Node embedding only — fine-tune only adaptive_embedding
2. Without node embedding — fine-tune everything EXCEPT adaptive_embedding
3. Full model — fine-tune all parameters

All use RevIN (Instance Norm) trained backbone.
Key question: Does node embedding capture the majority of drift?
"""
import sys
import os
import json
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

FINETUNE_HOURS = [3, 6, 12, 24, 72, 168]

# Five conditions: (name, trainable_filter_fn, lr)
# trainable_filter_fn: given param name, returns True if should be trained
METHODS = {
    "emb_only": {
        "desc": "Node embedding only",
        "filter": lambda name: "adaptive_embedding" in name,
        "lr": 0.001,
    },
    "pred_head": {
        "desc": "Prediction head only",
        "filter": lambda name: "decoder" in name,
        "lr": 0.001,
    },
    "emb+head": {
        "desc": "Node embedding + prediction head",
        "filter": lambda name: "adaptive_embedding" in name or "decoder" in name,
        "lr": 0.001,
    },
    "without_emb": {
        "desc": "Without node embedding",
        "filter": lambda name: "adaptive_embedding" not in name,
        "lr": 0.0001,
    },
    "full_ft": {
        "desc": "Full model",
        "filter": lambda name: True,
        "lr": 0.0001,
    },
}


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


def finetune_model(model, filter_fn, train_x, train_y, mean, std,
                   epochs=FINETUNE_EPOCHS, lr=0.001):
    model = model.to(DEVICE)
    model.train()
    for name, param in model.named_parameters():
        param.requires_grad = filter_fn(name)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_trainable == 0:
        model.eval()
        return model, 0
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
    print("NODE EMBEDDING ABLATION: Is embedding the key to drift?")
    print("=" * 70)

    # Count params per method
    backbone_tmp = STAEformer(**MODEL_PARAM)
    print("\nMethod parameter counts:")
    for mname, minfo in METHODS.items():
        n = sum(p.numel() for name, p in backbone_tmp.named_parameters()
                if minfo["filter"](name))
        print(f"  {mname} ({minfo['desc']}): {n:,} params (lr={minfo['lr']})")
    del backbone_tmp

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

        # Zero-shot baseline
        backbone = load_backbone(ckpt_path).to(DEVICE)
        mae_zero = evaluate(backbone, tgt["test_x"], tgt["test_y"], src["mean"], src["std"])
        results[f"zero_shot_{pair}"] = mae_zero
        print(f"  Zero-shot: {mae_zero:.2f}")
        del backbone; torch.cuda.empty_cache()

        for hours in FINETUNE_HOURS:
            n_steps = hours * 12
            ft_data = tgt["full_data"][:n_steps]
            ft_x, ft_y = create_samples(ft_data, INPUT_LEN, OUTPUT_LEN)
            h_label = f"{hours}h" if hours < 24 else f"{hours//24}d"
            print(f"\n  --- {h_label} ({len(ft_x)} samples) ---")

            for mname, minfo in METHODS.items():
                backbone = load_backbone(ckpt_path)
                backbone, n_params = finetune_model(
                    backbone, minfo["filter"], ft_x, ft_y,
                    src["mean"], src["std"], lr=minfo["lr"]
                )
                mae = evaluate(backbone, tgt["test_x"], tgt["test_y"], src["mean"], src["std"])
                results[f"{mname}_{h_label}_{pair}"] = mae
                improve = mae_zero - mae
                print(f"    {mname:<15} {mae:.2f}  ({n_params:>7,}p)  improve: {improve:+.2f}")
                del backbone; torch.cuda.empty_cache()

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Average MAE across 6 pairs")
    print("=" * 70)

    zero_avg = np.nanmean([results.get(f"zero_shot_{s}_{t}", float('nan'))
                           for s, t in pairs])
    print(f"\nZero-shot (RevIN only): {zero_avg:.2f}")

    method_names = list(METHODS.keys())
    header = f"{'':>8}"
    for m in method_names:
        header += f" {m:>15}"
    print(header)
    print("-" * (8 + 16 * len(method_names)))

    for hours in FINETUNE_HOURS:
        h_label = f"{hours}h" if hours < 24 else f"{hours//24}d"
        row = f"{h_label:>8}"
        for mname in method_names:
            avg = np.nanmean([results.get(f"{mname}_{h_label}_{s}_{t}", float('nan'))
                              for s, t in pairs])
            row += f" {avg:>15.2f}"
        print(row)

    # Relative contribution
    print(f"\n{'='*70}")
    print("EMBEDDING CONTRIBUTION (% of Full FT improvement over zero-shot)")
    print(f"{'='*70}")
    print(f"{'':>8} {'Emb%':>10} {'w/oEmb%':>10} {'Full%':>10}")
    print("-" * 45)

    for hours in FINETUNE_HOURS:
        h_label = f"{hours}h" if hours < 24 else f"{hours//24}d"
        full_avg = np.nanmean([results.get(f"full_ft_{h_label}_{s}_{t}", float('nan'))
                               for s, t in pairs])
        full_improve = zero_avg - full_avg

        row = f"{h_label:>8}"
        for mname in method_names:
            avg = np.nanmean([results.get(f"{mname}_{h_label}_{s}_{t}", float('nan'))
                              for s, t in pairs])
            improve = zero_avg - avg
            pct = (improve / full_improve * 100) if full_improve > 0 else 0
            row += f" {pct:>9.1f}%"
        print(row)

    # Per-pair detail
    print(f"\n{'='*70}")
    print("PER-PAIR: Embedding improvement as % of Full FT improvement")
    print(f"{'='*70}")

    for hours in FINETUNE_HOURS:
        h_label = f"{hours}h" if hours < 24 else f"{hours//24}d"
        print(f"\n  {h_label}:")
        for s, t in pairs:
            pair = f"{s}_{t}"
            zero = results.get(f"zero_shot_{pair}", float('nan'))
            emb = results.get(f"emb_only_{h_label}_{pair}", float('nan'))
            full = results.get(f"full_ft_{h_label}_{pair}", float('nan'))
            full_imp = zero - full
            emb_imp = zero - emb
            pct = (emb_imp / full_imp * 100) if full_imp > 0 else 0
            print(f"    {s}->{t}: zero={zero:.2f} emb={emb:.2f} full={full:.2f} "
                  f"emb_contrib={pct:.0f}%")

    output_path = "eda/concept_drift/peft_component_ablation_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
