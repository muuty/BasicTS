"""Evaluate EPN on Real Expansion Scenarios

Uses trained EPNs to predict NSP for new nodes in Q2/Q3/Q4 expansion.
Compares EPN initialization vs baselines (mean init, graph-only, etc.).

Pipeline:
1. Load source model (Q1, 558 nodes)
2. For each quarter (Q2, Q3, Q4):
   a. Expand model to new node count
   b. For each budget (3h, 12h, 1d, 7d):
      - EPN: predict new node NSPs from traffic + graph context
      - Baselines: mean init, graph-only init
      - (Optional) Fine-tune with few-shot data
      - Evaluate MAE on test set
"""

import os
import sys
import json
import copy
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.dirname(__file__))

from epn_model import EmbeddingPredictorNetwork
from baselines.STAEformer.arch import STAEformer

# ============================================================
# Config (shared with expanding_adaptation.py)
# ============================================================
DEVICE = "cuda:1"
DATA_DIR = "datasets/expanding_experiment"
ADJ_PATH = "datasets/xtraffic/SAN_BERNARDINO/adj_mx.pkl"
CKPT_DIR = "checkpoints/Expanding_Source/SAN_BERNARDINO_2023_Q1_30_12_12"
EPN_DIR = "project/few-shot-adaptation/results"
RESULTS_DIR = "project/few-shot-adaptation/results"

INPUT_LEN = 12
OUTPUT_LEN = 12
NUM_CHANNELS = 5
NSP_T = 12
NSP_D = 24
NSP_DIM = NSP_T * NSP_D
MAX_NEIGHBORS = 10

TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
BATCH_SIZE = 16
MAX_EPOCHS = 200
PATIENCE = 10  # early stopping patience (epochs)
VAL_EVERY = 1  # validate every epoch
FT_LR = 1e-4

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

INIT_METHODS = ["mean_init", "graph_only", "epn", "epn_ft", "mean_ft"]


# ============================================================
# Data utilities (same as expanding_adaptation.py)
# ============================================================
def load_adjacency_full(adj_path):
    """Load full 893x893 adjacency matrix."""
    with open(adj_path, "rb") as f:
        adj_data = pickle.load(f, encoding="latin1")
    if isinstance(adj_data, list):
        adj_mx = adj_data[2]
    else:
        adj_mx = adj_data
    adj_mx = np.array(adj_mx, dtype=np.float32)
    np.fill_diagonal(adj_mx, 0)
    return adj_mx


def load_quarter_raw(quarter_name):
    """Load raw data for a quarter."""
    cfg = QUARTERS[quarter_name]
    with open(f"{cfg['data']}/desc.json") as f:
        desc = json.load(f)
    data = np.memmap(f"{cfg['data']}/data.dat", dtype="float32", mode="r",
                     shape=tuple(desc["shape"]))
    return np.array(data)


def create_windows(data, node_indices, input_len=12, output_len=12):
    """Create sliding windows."""
    total_len = input_len + output_len
    num_windows = len(data) - total_len + 1
    data_filtered = data[:, node_indices, :]
    inputs = np.stack([data_filtered[i:i+input_len] for i in range(num_windows)])
    targets = np.stack([data_filtered[i+input_len:i+total_len] for i in range(num_windows)])
    return inputs, targets


def compute_zscore_stats(data, train_ratio=0.6):
    """Compute global mean/std from training portion of flow."""
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
    """Forward pass with instance norm."""
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
def load_source_model():
    """Load trained Q1 source model."""
    hash_dirs = [d for d in os.listdir(CKPT_DIR)
                 if os.path.isdir(os.path.join(CKPT_DIR, d)) and not d.startswith('.')]
    ckpt_path = os.path.join(CKPT_DIR, hash_dirs[0], "STAEformer_best_val_MAE.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"]

    model = STAEformer(**{**MODEL_PARAM, "num_nodes": 558}).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model, state_dict


def expand_model_with_init(state_dict, old_nodes, new_nodes, new_nsp_values):
    """Expand model and set new node NSPs to given values.

    Args:
        state_dict: source model state dict
        old_nodes: array of old node global indices
        new_nodes: array of new node global indices
        new_nsp_values: dict {global_node_id: tensor (T, D)} for new nodes
    """
    all_nodes = np.sort(np.concatenate([old_nodes, new_nodes]))
    new_num_nodes = len(all_nodes)
    all_list = all_nodes.tolist()

    param = {**MODEL_PARAM, "num_nodes": new_num_nodes}
    new_model = STAEformer(**param).to(DEVICE)
    new_state = new_model.state_dict()

    for key in state_dict:
        if key == "encoder.adaptive_embedding":
            continue
        if key in new_state and state_dict[key].shape == new_state[key].shape:
            new_state[key] = state_dict[key]

    # Build expanded NSP
    old_emb = state_dict["encoder.adaptive_embedding"]  # (T, N_old, D)
    T, N_old, D = old_emb.shape
    new_emb = torch.zeros(T, new_num_nodes, D, device=DEVICE)

    # Copy old node NSPs
    for i, node_id in enumerate(old_nodes):
        new_pos = all_list.index(node_id)
        new_emb[:, new_pos, :] = old_emb[:, i, :]

    # Set new node NSPs
    for node_id in new_nodes:
        new_pos = all_list.index(node_id)
        if node_id in new_nsp_values:
            new_emb[:, new_pos, :] = new_nsp_values[node_id]
        # else: stays at zero

    new_state["encoder.adaptive_embedding"] = new_emb
    new_model.load_state_dict(new_state)
    return new_model, all_nodes


# ============================================================
# NSP initialization methods
# ============================================================
def mean_init_nsps(old_emb, new_nodes):
    """Initialize new nodes with mean of existing NSPs."""
    mean_emb = old_emb.mean(dim=1)  # (T, D)
    return {nid: mean_emb.clone() for nid in new_nodes}


def graph_only_init_nsps(old_emb, old_nodes, new_nodes, adj_full):
    """Initialize new nodes from weighted mean of neighbor NSPs."""
    old_list = old_nodes.tolist()
    result = {}
    for nid in new_nodes:
        # Get adjacency weights to old nodes
        weights = []
        neighbor_embs = []
        for i, old_nid in enumerate(old_nodes):
            w = adj_full[nid, old_nid]
            if w > 0:
                weights.append(w)
                neighbor_embs.append(old_emb[:, i, :])  # (T, D)

        if neighbor_embs:
            # Top-K neighbors
            if len(weights) > MAX_NEIGHBORS:
                topk_idx = np.argsort(weights)[-MAX_NEIGHBORS:]
                weights = [weights[j] for j in topk_idx]
                neighbor_embs = [neighbor_embs[j] for j in topk_idx]

            weights_t = torch.tensor(weights, dtype=torch.float32)
            weights_t = weights_t / weights_t.sum()
            stacked = torch.stack(neighbor_embs)  # (K, T, D)
            weighted = (stacked * weights_t.view(-1, 1, 1)).sum(dim=0)  # (T, D)
            result[nid] = weighted
        else:
            # Fallback to mean
            result[nid] = old_emb.mean(dim=1)
    return result


def epn_init_nsps(epn, old_emb, old_nodes, new_nodes, adj_full, raw_data_quarter,
                  all_quarter_nodes, budget_steps):
    """Use EPN to predict new node NSPs."""
    old_list = old_nodes.tolist()
    old_nsp_flat = old_emb.permute(1, 0, 2).reshape(len(old_nodes), -1)  # (N_old, T*D)
    result = {}

    for nid in new_nodes:
        # 1. Traffic data for this new node
        nid_local = list(all_quarter_nodes).index(nid)
        train_end = int(len(raw_data_quarter) * TRAIN_RATIO)
        node_traffic = raw_data_quarter[:train_end, nid_local, :]
        actual_steps = min(budget_steps, len(node_traffic)) if budget_steps else len(node_traffic)
        traffic = torch.tensor(node_traffic[:actual_steps], dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # 2. Neighbor NSPs from old nodes
        weights = []
        neighbor_nsps = []
        for i, old_nid in enumerate(old_nodes):
            w = adj_full[nid, old_nid]
            if w > 0:
                weights.append(w)
                neighbor_nsps.append(old_nsp_flat[i])

        if len(weights) > MAX_NEIGHBORS:
            topk_idx = np.argsort(weights)[-MAX_NEIGHBORS:]
            weights = [weights[j] for j in topk_idx]
            neighbor_nsps = [neighbor_nsps[j] for j in topk_idx]

        # Pad
        pad_len = MAX_NEIGHBORS - len(weights)
        if neighbor_nsps:
            n_nsps = torch.stack(neighbor_nsps)
        else:
            n_nsps = torch.zeros(0, NSP_DIM)
        if pad_len > 0:
            n_nsps = torch.cat([n_nsps, torch.zeros(pad_len, NSP_DIM)], dim=0)
        adj_w = torch.tensor(weights + [0.0] * pad_len, dtype=torch.float32)

        n_nsps = n_nsps.unsqueeze(0).to(DEVICE)
        adj_w = adj_w.unsqueeze(0).to(DEVICE)

        # 3. EPN prediction (cosine-loss trained → rescale to match existing norm)
        with torch.no_grad():
            pred_nsp = epn(traffic, n_nsps, adj_w).squeeze(0).cpu()  # (T*D,)

        # Rescale: cosine loss only learns direction, so match mean existing norm
        mean_norm = old_nsp_flat.norm(dim=1).mean()
        pred_norm = pred_nsp.norm()
        if pred_norm > 1e-8:
            pred_nsp = pred_nsp * (mean_norm / pred_norm)

        result[nid] = pred_nsp.view(NSP_T, NSP_D)  # (T, D)

    return result


# ============================================================
# Fine-tuning (new embeddings only, gradient mask)
# ============================================================
def finetune_new_only(model, train_loader, val_loader, old_nodes, new_nodes,
                      all_nodes, mean, std):
    """Fine-tune only new node NSPs with early stopping."""
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
            # Gradient mask: zero out old node gradients
            if model.encoder.adaptive_embedding.grad is not None:
                model.encoder.adaptive_embedding.grad[:, old_local, :] = 0
            optimizer.step()

        # Validate every VAL_EVERY epochs
        if (epoch + 1) % VAL_EVERY != 0 and epoch < MAX_EPOCHS - 1:
            continue

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
    """Evaluate and return MAE for all/existing/new nodes."""
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
        "mae_existing": all_errors[:, :, old_local].mean().item() if old_local else float("nan"),
        "mae_new": all_errors[:, :, new_local].mean().item() if new_local else float("nan"),
    }


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load source model and state dict
    _, source_state = load_source_model()
    old_emb = source_state["encoder.adaptive_embedding"]  # (T, 558, D)
    print(f"Source model NSP: {old_emb.shape}")

    # Load full adjacency
    adj_full = load_adjacency_full(ADJ_PATH)
    print(f"Full adjacency: {adj_full.shape}")

    # Q1 nodes (source)
    q1_nodes = np.load(f"{DATA_DIR}/nodes_q1.npy")

    # Load EPNs
    epns = {}
    for budget_name in BUDGETS:
        epn_path = os.path.join(EPN_DIR, f"epn_{budget_name}.pt")
        if os.path.exists(epn_path):
            epn = EmbeddingPredictorNetwork(nsp_dim=NSP_DIM, num_channels=NUM_CHANNELS, d_hidden=64).to(DEVICE)
            epn.load_state_dict(torch.load(epn_path, map_location=DEVICE))
            epn.eval()
            epns[budget_name] = epn
            print(f"Loaded EPN: {budget_name}")
        else:
            print(f"EPN not found: {epn_path}")

    results = {}

    for q_name, q_cfg in QUARTERS.items():
        print(f"\n{'='*60}")
        print(f"Quarter: {q_name}")
        print(f"{'='*60}")

        q_nodes = np.load(q_cfg["nodes"])
        new_nodes = np.load(q_cfg["new"])
        print(f"  Nodes: {len(q1_nodes)} → {len(q_nodes)} (+{len(new_nodes)} new)")

        # Load quarter data
        raw_data = load_quarter_raw(q_name)
        mean, std = compute_zscore_stats(raw_data[:, q_nodes, :])

        # Create windows and loaders
        inputs, targets = create_windows(raw_data, q_nodes)
        num_total = len(inputs)
        train_end = int(num_total * TRAIN_RATIO)
        val_end = int(num_total * (TRAIN_RATIO + VAL_RATIO))

        def to_loader(start, end, shuffle=False):
            inp = torch.tensor(inputs[start:end], dtype=torch.float32)
            tgt = torch.tensor(targets[start:end], dtype=torch.float32)
            return DataLoader(TensorDataset(inp, tgt), batch_size=BATCH_SIZE, shuffle=shuffle)

        # Subsample validation for faster fine-tuning (max 1000 windows)
        val_size = val_end - train_end
        if val_size > 1000:
            val_subset_end = train_end + 1000
        else:
            val_subset_end = val_end
        val_loader = to_loader(train_end, val_subset_end)
        test_loader = to_loader(val_end, num_total)

        results[q_name] = {}

        for budget_name, budget_steps in BUDGETS.items():
            print(f"\n  --- Budget: {budget_name} ---")

            # Train loader for FT
            if budget_steps is not None:
                max_windows = max(1, budget_steps - INPUT_LEN - OUTPUT_LEN + 1)
                end = min(train_end, max_windows)
            else:
                end = train_end
            train_loader = to_loader(0, end, shuffle=True)

            budget_results = {}

            for method in INIT_METHODS:
                print(f"  Method: {method}")

                # Determine NSP values for new nodes
                if method in ("mean_init", "mean_ft"):
                    new_nsps = mean_init_nsps(old_emb, new_nodes)
                elif method == "graph_only":
                    new_nsps = graph_only_init_nsps(old_emb, q1_nodes, new_nodes, adj_full)
                elif method in ("epn", "epn_ft"):
                    if budget_name not in epns:
                        print(f"    Skipping (no EPN for {budget_name})")
                        continue
                    new_nsps = epn_init_nsps(
                        epns[budget_name], old_emb, q1_nodes, new_nodes,
                        adj_full, raw_data, q_nodes, budget_steps or len(raw_data)
                    )
                else:
                    continue

                # Build expanded model
                model, all_nodes = expand_model_with_init(
                    source_state, q1_nodes, new_nodes, new_nsps
                )

                # Fine-tune if needed
                stopped_epoch = 0
                if method in ("mean_ft", "epn_ft"):
                    model, stopped_epoch = finetune_new_only(
                        model, train_loader, val_loader,
                        q1_nodes, new_nodes, all_nodes, mean, std
                    )

                # Evaluate
                res = evaluate(model, test_loader, all_nodes, q1_nodes, new_nodes, mean, std)
                res["stopped_epoch"] = stopped_epoch
                budget_results[method] = res
                print(f"    MAE: all={res['mae_all']:.4f}, existing={res['mae_existing']:.4f}, "
                      f"new={res['mae_new']:.4f}")

                del model
                torch.cuda.empty_cache()

            results[q_name][budget_name] = budget_results

        del raw_data, inputs, targets

    # Save results
    out_path = os.path.join(RESULTS_DIR, "epn_evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for q_name in results:
        print(f"\n{q_name}:")
        for budget in results[q_name]:
            print(f"  {budget}:")
            for method, res in results[q_name][budget].items():
                print(f"    {method:15s}: all={res['mae_all']:.4f}, "
                      f"new={res['mae_new']:.4f}, exist={res['mae_existing']:.4f}")


if __name__ == "__main__":
    main()
