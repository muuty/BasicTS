"""Evaluate Masked Node Pre-training + Proxy FT for Node Expansion.

Ablation experiments:
1. Normal PT + Vanilla FT (baseline)
2. Masked PT + Vanilla FT (masked PT only)
3. Normal PT + Proxy FT (proxy only)
4. Masked PT + Proxy FT (full)

For each: cold_start (no FT) and FT versions across budgets.
"""

import os
import sys
import json
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from baselines.STAEformer.arch import STAEformer

# ============================================================
# Config
# ============================================================
DEVICE = "cuda:1"
DATA_DIR = "datasets/expanding_experiment"

# Source model checkpoints
CKPT_NORMAL = "checkpoints/Expanding_Source/SAN_BERNARDINO_2023_Q1_30_12_12"
CKPT_MASKED = "checkpoints/Expanding_Source_MaskedPT/SAN_BERNARDINO_2023_Q1_30_12_12"
CKPT_LEARNABLE = "checkpoints/Expanding_Source_MaskedPT_Learnable/SAN_BERNARDINO_2023_Q1_30_12_12"

RESULTS_DIR = "project/few-shot-adaptation/results/masked_pt"

INPUT_LEN = 12
OUTPUT_LEN = 12
NUM_CHANNELS = 5
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
BATCH_SIZE = 16
MAX_EPOCHS = 200
PATIENCE = 10
FT_LR = 1e-4

# Proxy FT config
PROXY_TEMPORAL_MASK_RATIO = 0.25  # mask 25% of input timesteps
PROXY_LAMBDA = 1.0  # weight of proxy loss (equal to forecast loss)
PROXY_PROB = 0.5  # probability of applying masking per batch

BUDGETS = {
    "3h": 36,
    "12h": 144,
    "1d": 288,
    "7d": 2016,
    "full": None,
}

MODEL_PARAM = {
    "in_steps": INPUT_LEN,
    "out_steps": OUTPUT_LEN,
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

QUARTERS = {
    "Q2": {"data": f"{DATA_DIR}/SAN_BERNARDINO_2023_Q2", "nodes": f"{DATA_DIR}/nodes_q2.npy", "new": f"{DATA_DIR}/new_in_q2.npy"},
    "Q3": {"data": f"{DATA_DIR}/SAN_BERNARDINO_2023_Q3", "nodes": f"{DATA_DIR}/nodes_q3.npy", "new": f"{DATA_DIR}/new_in_q3.npy"},
    "Q4": {"data": f"{DATA_DIR}/SAN_BERNARDINO_2023_Q4", "nodes": f"{DATA_DIR}/nodes_q4.npy", "new": f"{DATA_DIR}/new_in_q4.npy"},
}


# ============================================================
# Data utilities
# ============================================================
def load_quarter_raw(quarter_name):
    cfg = QUARTERS[quarter_name]
    with open(f"{cfg['data']}/desc.json") as f:
        desc = json.load(f)
    data = np.memmap(f"{cfg['data']}/data.dat", dtype="float32", mode="r",
                     shape=tuple(desc["shape"]))
    return np.array(data)


def create_windows(data, node_indices, input_len=12, output_len=12):
    total_len = input_len + output_len
    num_windows = len(data) - total_len + 1
    data_filtered = data[:, node_indices, :]
    inputs = np.stack([data_filtered[i:i+input_len] for i in range(num_windows)])
    targets = np.stack([data_filtered[i+input_len:i+total_len] for i in range(num_windows)])
    return inputs, targets


def compute_zscore_stats(data, train_ratio=0.6):
    train_size = int(len(data) * train_ratio)
    train_flow = data[:train_size, :, 0]
    mean = float(np.mean(train_flow))
    std = float(np.std(train_flow))
    if std == 0:
        std = 1.0
    return mean, std


def zscore_transform(tensor, mean, std):
    out = tensor.clone()
    out[..., 0] = (out[..., 0] - mean) / std
    return out


def instance_norm_forward(model, history_data, mean, std):
    flow = history_data[:, :, :, 0]
    flow_mean = flow.mean(dim=1, keepdim=True)
    flow_std = flow.std(dim=1, keepdim=True) + 1e-5
    history_normed = history_data.clone()
    history_normed[:, :, :, 0] = (flow - flow_mean) / flow_std
    model_input = history_normed[..., :5]
    output = model(history_data=model_input, future_data=None,
                   batch_seen=0, epoch=0, train=False)
    pred = output["prediction"]
    pred = pred * flow_std.unsqueeze(-1) + flow_mean.unsqueeze(-1)
    pred = pred * std + mean
    return pred


# ============================================================
# Model loading and expansion
# ============================================================
def load_source_model(ckpt_dir):
    hash_dirs = [d for d in os.listdir(ckpt_dir)
                 if os.path.isdir(os.path.join(ckpt_dir, d)) and not d.startswith('.')]
    ckpt_path = os.path.join(ckpt_dir, hash_dirs[0], "STAEformer_best_val_MAE.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"]
    return state_dict


def expand_model(state_dict, old_nodes, new_nodes, init_mode="mean"):
    """Expand model with specified init for new nodes.

    Args:
        init_mode: "mean" (mean of existing, ≈ zero), "learned" (use default_embedding from checkpoint)
    """
    all_nodes = np.sort(np.concatenate([old_nodes, new_nodes]))
    new_num_nodes = len(all_nodes)
    all_list = all_nodes.tolist()

    param = {**MODEL_PARAM, "num_nodes": new_num_nodes}
    new_model = STAEformer(**param).to(DEVICE)
    new_state = new_model.state_dict()

    skip_keys = {"encoder.adaptive_embedding", "encoder.default_embedding"}
    for key in state_dict:
        if key in skip_keys:
            continue
        if key in new_state and state_dict[key].shape == new_state[key].shape:
            new_state[key] = state_dict[key]

    # Determine fill value for new nodes
    old_emb = state_dict["encoder.adaptive_embedding"]  # (T, N_old, D)
    T, N_old, D = old_emb.shape

    if init_mode == "learned" and "encoder.default_embedding" in state_dict:
        fill_emb = state_dict["encoder.default_embedding"].squeeze(1)  # (T, D)
    else:
        fill_emb = old_emb.mean(dim=1)  # (T, D)

    new_emb = torch.zeros(T, new_num_nodes, D)

    for i, node_id in enumerate(old_nodes):
        new_pos = all_list.index(node_id)
        new_emb[:, new_pos, :] = old_emb[:, i, :]

    for node_id in new_nodes:
        new_pos = all_list.index(node_id)
        new_emb[:, new_pos, :] = fill_emb

    new_state["encoder.adaptive_embedding"] = new_emb
    new_model.load_state_dict(new_state, strict=False)
    return new_model, all_nodes


# ============================================================
# Fine-tuning methods
# ============================================================
def finetune_vanilla(model, train_loader, val_loader, old_nodes, new_nodes,
                     all_nodes, mean, std):
    """Standard fine-tuning: only update new node embeddings."""
    all_list = all_nodes.tolist()
    old_local = [all_list.index(n) for n in old_nodes]

    for name, param in model.named_parameters():
        param.requires_grad = (name == "encoder.adaptive_embedding")
    optimizer = torch.optim.Adam([model.encoder.adaptive_embedding], lr=FT_LR)

    best_val = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(MAX_EPOCHS):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            inputs_z = zscore_transform(inputs, mean, std)
            pred = instance_norm_forward(model, inputs_z, mean, std).squeeze(-1)
            loss = (pred - targets[..., 0]).abs().mean()
            optimizer.zero_grad()
            loss.backward()
            if model.encoder.adaptive_embedding.grad is not None:
                model.encoder.adaptive_embedding.grad[:, old_local, :] = 0
            optimizer.step()

        # Validate
        model.eval()
        val_loss = 0
        val_count = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                inputs_z = zscore_transform(inputs, mean, std)
                pred = instance_norm_forward(model, inputs_z, mean, std).squeeze(-1)
                val_loss += (pred - targets[..., 0]).abs().sum().item()
                val_count += targets[..., 0].numel()
        val_loss /= val_count

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)
        model = model.to(DEVICE)
    return model, epoch + 1


def finetune_proxy(model, train_loader, val_loader, old_nodes, new_nodes,
                   all_nodes, mean, std):
    """Proxy FT: temporal masking augmentation during fine-tuning."""
    all_list = all_nodes.tolist()
    old_local = [all_list.index(n) for n in old_nodes]
    new_local = [all_list.index(n) for n in new_nodes]

    for name, param in model.named_parameters():
        param.requires_grad = (name == "encoder.adaptive_embedding")
    optimizer = torch.optim.Adam([model.encoder.adaptive_embedding], lr=FT_LR)

    best_val = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(MAX_EPOCHS):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            inputs_z = zscore_transform(inputs, mean, std)

            # Standard forecasting loss
            pred = instance_norm_forward(model, inputs_z, mean, std).squeeze(-1)
            loss_forecast = (pred - targets[..., 0]).abs().mean()

            # Proxy: temporal masking on new nodes' input
            inputs_masked = inputs_z.clone()
            B, T, N, C = inputs_masked.shape
            # Random temporal mask per sample: mask PROXY_TEMPORAL_MASK_RATIO of timesteps
            tmask = torch.rand(B, T, device=DEVICE) < PROXY_TEMPORAL_MASK_RATIO
            # Apply mask only to new nodes
            for ni in new_local:
                # Replace masked timesteps with 0 (after z-score, 0 ≈ mean)
                inputs_masked[:, :, ni, :][tmask] = 0.0

            pred_masked = instance_norm_forward(model, inputs_masked, mean, std).squeeze(-1)
            loss_proxy = (pred_masked - targets[..., 0]).abs().mean()

            loss = loss_forecast + PROXY_LAMBDA * loss_proxy

            optimizer.zero_grad()
            loss.backward()
            if model.encoder.adaptive_embedding.grad is not None:
                model.encoder.adaptive_embedding.grad[:, old_local, :] = 0
            optimizer.step()

        # Validate (no masking)
        model.eval()
        val_loss = 0
        val_count = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                inputs_z = zscore_transform(inputs, mean, std)
                pred = instance_norm_forward(model, inputs_z, mean, std).squeeze(-1)
                val_loss += (pred - targets[..., 0]).abs().sum().item()
                val_count += targets[..., 0].numel()
        val_loss /= val_count

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)
        model = model.to(DEVICE)
    return model, epoch + 1


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def evaluate(model, test_loader, all_nodes, old_nodes, new_nodes, mean, std):
    model.eval()
    all_list = list(all_nodes)
    old_local = [all_list.index(n) for n in old_nodes if n in all_list]
    new_local = [all_list.index(n) for n in new_nodes if n in all_list]

    all_errors = []
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        inputs_z = zscore_transform(inputs, mean, std)
        pred = instance_norm_forward(model, inputs_z, mean, std).squeeze(-1)
        errors = (pred - targets[..., 0]).abs()
        all_errors.append(errors.cpu())

    all_errors = torch.cat(all_errors, dim=0)
    return {
        "mae_all": all_errors.mean().item(),
        "mae_existing": all_errors[:, :, old_local].mean().item(),
        "mae_new": all_errors[:, :, new_local].mean().item(),
    }


# ============================================================
# Main
# ============================================================
def run_experiment(pt_name, ckpt_dir, ft_methods, init_mode="mean"):
    """Run expansion experiment for one PT variant."""
    print(f"\n{'='*70}")
    print(f"Source: {pt_name} ({ckpt_dir}) [init={init_mode}]")
    print(f"{'='*70}")

    state_dict = load_source_model(ckpt_dir)
    q1_nodes = np.load(f"{DATA_DIR}/nodes_q1.npy")
    results = {}

    for q_name, q_cfg in QUARTERS.items():
        print(f"\n  Quarter: {q_name}")
        q_nodes = np.load(q_cfg["nodes"])
        # Cumulative new nodes vs Q1 (not incremental vs previous quarter)
        new_nodes = np.setdiff1d(q_nodes, q1_nodes)
        print(f"    Nodes: {len(q1_nodes)} -> {len(q_nodes)} (+{len(new_nodes)} new)")

        raw_data = load_quarter_raw(q_name)
        mean, std = compute_zscore_stats(raw_data[:, q_nodes, :])

        inputs, targets = create_windows(raw_data, q_nodes)
        num_total = len(inputs)
        train_end = int(num_total * TRAIN_RATIO)
        val_end = int(num_total * (TRAIN_RATIO + VAL_RATIO))

        def to_loader(start, end, shuffle=False):
            inp = torch.tensor(inputs[start:end], dtype=torch.float32)
            tgt = torch.tensor(targets[start:end], dtype=torch.float32)
            return DataLoader(TensorDataset(inp, tgt), batch_size=BATCH_SIZE, shuffle=shuffle)

        val_size = val_end - train_end
        val_loader = to_loader(train_end, min(train_end + 1000, val_end))
        test_loader = to_loader(val_end, num_total)

        results[q_name] = {}

        for budget_name, budget_steps in BUDGETS.items():
            print(f"\n    Budget: {budget_name}")

            if budget_steps is not None:
                max_windows = max(1, budget_steps - INPUT_LEN - OUTPUT_LEN + 1)
                end = min(train_end, max_windows)
            else:
                end = train_end
            train_loader = to_loader(0, end, shuffle=True)

            budget_results = {}

            for method_name, ft_fn in ft_methods:
                # Expand model (fresh copy each time)
                model, all_nodes = expand_model(state_dict, q1_nodes, new_nodes, init_mode=init_mode)

                if ft_fn is None:
                    # Cold start - no fine-tuning
                    pass
                else:
                    model, stopped = ft_fn(
                        model, train_loader, val_loader,
                        q1_nodes, new_nodes, all_nodes, mean, std
                    )

                res = evaluate(model, test_loader, all_nodes, q1_nodes, new_nodes, mean, std)
                budget_results[method_name] = res
                print(f"      {method_name:20s}: all={res['mae_all']:.4f}, "
                      f"new={res['mae_new']:.4f}, exist={res['mae_existing']:.4f}")

                del model
                torch.cuda.empty_cache()

            results[q_name][budget_name] = budget_results

        del raw_data, inputs, targets

    return results


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = {}

    # Define FT methods
    ft_cold = ("cold_start", None)
    ft_vanilla = ("vanilla_ft", finetune_vanilla)
    ft_proxy = ("proxy_ft", finetune_proxy)

    # Experiment 1: Normal PT (baseline + proxy only)
    if os.path.exists(CKPT_NORMAL):
        print("\n" + "#"*70)
        print("EXPERIMENT: Normal PT")
        print("#"*70)
        all_results["normal_pt"] = run_experiment(
            "Normal PT", CKPT_NORMAL,
            [ft_cold, ft_vanilla, ft_proxy]
        )

    # Experiment 2: Masked PT (masked PT only + full)
    if os.path.exists(CKPT_MASKED):
        print("\n" + "#"*70)
        print("EXPERIMENT: Masked PT")
        print("#"*70)
        all_results["masked_pt"] = run_experiment(
            "Masked PT", CKPT_MASKED,
            [ft_cold, ft_vanilla, ft_proxy]
        )
    else:
        print(f"\nMasked PT checkpoint not found: {CKPT_MASKED}")
        print("Train it first with: expanding_q1_source_masked.py")

    # Experiment 3: Masked PT + Learnable Default
    if os.path.exists(CKPT_LEARNABLE):
        print("\n" + "#"*70)
        print("EXPERIMENT: Masked PT + Learnable Default")
        print("#"*70)
        all_results["learnable_default"] = run_experiment(
            "Learnable Default", CKPT_LEARNABLE,
            [ft_cold, ft_vanilla, ft_proxy],
            init_mode="learned"
        )
    else:
        print(f"\nLearnable Default checkpoint not found: {CKPT_LEARNABLE}")
        print("Train it first with: expanding_q1_source_masked_learnable.py")

    # Save all results
    out_path = os.path.join(RESULTS_DIR, "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY: New Node MAE")
    print(f"{'='*80}")
    print(f"{'Method':<30s} {'3h':>8s} {'12h':>8s} {'1d':>8s} {'7d':>8s} {'full':>8s}")
    print("-" * 70)
    for pt_name, pt_results in all_results.items():
        if "Q2" not in pt_results:
            continue
        q2 = pt_results["Q2"]
        for method in ["cold_start", "vanilla_ft", "proxy_ft"]:
            label = f"{pt_name}/{method}"
            vals = []
            for b in ["3h", "12h", "1d", "7d", "full"]:
                if b in q2 and method in q2[b]:
                    vals.append(f"{q2[b][method]['mae_new']:.2f}")
                else:
                    vals.append("--")
            print(f"{label:<30s} {'  '.join(f'{v:>6s}' for v in vals)}")


if __name__ == "__main__":
    main()
