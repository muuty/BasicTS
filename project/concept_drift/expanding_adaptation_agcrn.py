"""Continual Node Expansion Experiment - AGCRN

Q1(558) → Q2(669) → Q3(781) → Q4(893)
Methods: emb_all, emb_new_only, full_model
Budgets: 3h, 12h, 1d, 7d, full (time-based)

AGCRN has node_embeddings (N, embed_dim=10) as its NSP.
No instance norm. Input: flow only (1 channel).
"""

import os
import sys
import copy
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from baselines.AGCRN.arch import AGCRN

# ============================================================
# Config
# ============================================================
DEVICE = "cuda:1"
DATA_DIR = "datasets/expanding_experiment"
NODE_DIR = "datasets/expanding_experiment"

QUARTERS = {
    "Q1": {"data": f"{DATA_DIR}/SAN_BERNARDINO_2023_Q1", "nodes": f"{NODE_DIR}/nodes_q1.npy"},
    "Q2": {"data": f"{DATA_DIR}/SAN_BERNARDINO_2023_Q2", "nodes": f"{NODE_DIR}/nodes_q2.npy", "new": f"{NODE_DIR}/new_in_q2.npy"},
    "Q3": {"data": f"{DATA_DIR}/SAN_BERNARDINO_2023_Q3", "nodes": f"{NODE_DIR}/nodes_q3.npy", "new": f"{NODE_DIR}/new_in_q3.npy"},
    "Q4": {"data": f"{DATA_DIR}/SAN_BERNARDINO_2023_Q4", "nodes": f"{NODE_DIR}/nodes_q4.npy", "new": f"{NODE_DIR}/new_in_q4.npy"},
}

INPUT_LEN = 12
OUTPUT_LEN = 12
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
BATCH_SIZE = 16
MAX_EPOCHS = 200
PATIENCE = 10
FT_LR = 1e-4

BUDGETS = {
    "3h": 36,
    "12h": 144,
    "1d": 288,
    "7d": 2016,
    "full": None,
}

MODEL_PARAM = {
    "input_dim": 1,
    "rnn_units": 64,
    "output_dim": 1,
    "horizon": OUTPUT_LEN,
    "num_layers": 2,
    "default_graph": True,
    "embed_dim": 10,
    "cheb_k": 2,
}

NSP_KEY = "node_embeddings"  # (N, embed_dim)
METHODS = ["emb_all", "emb_new_only", "full_model"]


# ============================================================
# Data utilities
# ============================================================
def load_quarter_data(quarter_name):
    cfg = QUARTERS[quarter_name]
    with open(f"{cfg['data']}/desc.json") as f:
        desc = json.load(f)
    data = np.memmap(f"{cfg['data']}/data.dat", dtype="float32", mode="r",
                     shape=tuple(desc["shape"]))
    data = np.array(data)
    node_indices = np.load(cfg["nodes"])
    new_indices = np.load(cfg["new"]) if "new" in cfg else np.array([], dtype=int)
    return data, node_indices, new_indices


def compute_zscore_stats(data, train_ratio=0.6):
    train_size = int(len(data) * train_ratio)
    train_flow = data[:train_size, :, 0]
    mean = float(np.mean(train_flow))
    std = float(np.std(train_flow))
    if std == 0:
        std = 1.0
    return mean, std


def create_windows(data, node_indices, input_len=12, output_len=12):
    """Create sliding windows. Returns flow-only inputs and targets."""
    total_len = input_len + output_len
    num_windows = len(data) - total_len + 1
    data_filtered = data[:, node_indices, :]

    # AGCRN uses flow only (channel 0)
    flow = data_filtered[:, :, 0:1]  # (T_total, N, 1)
    inputs = np.stack([flow[i:i+input_len] for i in range(num_windows)])  # (W, T_in, N, 1)
    targets = np.stack([flow[i+input_len:i+total_len] for i in range(num_windows)])  # (W, T_out, N, 1)
    return inputs, targets


def precompute_quarter_data(data, node_indices, batch_size=16):
    inputs, targets = create_windows(data, node_indices, INPUT_LEN, OUTPUT_LEN)
    num_total = len(inputs)
    train_end = int(num_total * TRAIN_RATIO)
    val_end = int(num_total * (TRAIN_RATIO + VAL_RATIO))

    train_inputs = inputs[:train_end]
    train_targets = targets[:train_end]

    def to_loader(start, end, shuffle=False):
        inp = torch.tensor(inputs[start:end], dtype=torch.float32)
        tgt = torch.tensor(targets[start:end], dtype=torch.float32)
        return DataLoader(TensorDataset(inp, tgt), batch_size=batch_size, shuffle=shuffle)

    val_loader = to_loader(train_end, val_end)
    test_loader = to_loader(val_end, num_total)

    print(f"  Windows: total={num_total}, train={train_end}, val={val_end-train_end}, test={num_total-val_end}")
    del inputs, targets

    return {
        "train_inputs": train_inputs,
        "train_targets": train_targets,
        "train_end": train_end,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }


def make_train_loader(qdata, budget_steps=None, batch_size=16):
    train_inputs = qdata["train_inputs"]
    train_targets = qdata["train_targets"]
    train_end = qdata["train_end"]

    if budget_steps is not None:
        max_windows = max(1, budget_steps - INPUT_LEN - OUTPUT_LEN + 1)
        end = min(train_end, max_windows)
    else:
        end = train_end

    inp = torch.tensor(train_inputs[:end], dtype=torch.float32)
    tgt = torch.tensor(train_targets[:end], dtype=torch.float32)
    print(f"  Train loader: {end} windows (budget={'full' if budget_steps is None else budget_steps})")
    return DataLoader(TensorDataset(inp, tgt), batch_size=batch_size, shuffle=True)


# ============================================================
# ZScore utilities (AGCRN: flow only, no instance norm)
# ============================================================
def model_forward(model, inputs_z, mean, std):
    """Forward pass for AGCRN (no instance norm).

    Args:
        model: AGCRN
        inputs_z: (B, T, N, 1) ZScore'd flow
        mean, std: ZScore stats

    Returns:
        prediction: (B, T_out, N) in raw space
    """
    output = model(
        history_data=inputs_z,
        future_data=None,
        batch_seen=0, epoch=0, train=False
    )  # (B, T_out, N, 1)
    pred_z = output.squeeze(-1)  # (B, T_out, N)
    pred_raw = pred_z * std + mean
    return pred_raw


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def evaluate(model, test_loader, node_indices, existing_indices, new_indices, mean, std):
    model.eval()
    node_list = list(node_indices)
    existing_local = [node_list.index(n) for n in existing_indices if n in node_list]
    new_local = [node_list.index(n) for n in new_indices if n in node_list]

    all_errors = []
    for inputs, targets in test_loader:
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        # ZScore input
        inputs_z = (inputs - mean) / std  # (B, T, N, 1)
        targets_raw = targets.squeeze(-1)  # (B, T, N)

        pred_raw = model_forward(model, inputs_z, mean, std)
        errors = (pred_raw - targets_raw).abs()
        all_errors.append(errors.cpu())

    all_errors = torch.cat(all_errors, dim=0)
    mae_all = all_errors.mean().item()
    mae_existing = all_errors[:, :, existing_local].mean().item() if existing_local else float("nan")
    mae_new = all_errors[:, :, new_local].mean().item() if new_local else float("nan")

    return {"mae_all": mae_all, "mae_existing": mae_existing, "mae_new": mae_new}


# ============================================================
# Model expansion
# ============================================================
def expand_model(model, old_node_indices, new_node_indices, init="mean"):
    """Expand AGCRN to include new nodes.

    Only node_embeddings (N, embed_dim) is N-dependent.
    end_conv kernel is (1, hidden_dim) - not N-dependent.
    """
    all_node_indices = np.sort(np.concatenate([old_node_indices, new_node_indices]))
    new_num_nodes = len(all_node_indices)

    param = {**MODEL_PARAM, "num_nodes": new_num_nodes}
    new_model = AGCRN(**param).to(DEVICE)

    old_state = model.state_dict()
    new_state = new_model.state_dict()

    # Copy all weights except node_embeddings
    for key in old_state:
        if key == NSP_KEY:
            continue
        if key in new_state and old_state[key].shape == new_state[key].shape:
            new_state[key] = old_state[key]

    # Expand node_embeddings: (N_old, D) → (N_new, D)
    old_emb = old_state[NSP_KEY]  # (N_old, D)
    N_old, D = old_emb.shape
    new_emb = torch.zeros(new_num_nodes, D, device=DEVICE)

    all_list = all_node_indices.tolist()
    for i, node_id in enumerate(old_node_indices):
        new_pos = all_list.index(node_id)
        new_emb[new_pos, :] = old_emb[i, :]

    if init == "mean":
        mean_emb = old_emb.mean(dim=0)  # (D,)
        for node_id in new_node_indices:
            new_pos = all_list.index(node_id)
            new_emb[new_pos, :] = mean_emb

    new_state[NSP_KEY] = new_emb
    new_model.load_state_dict(new_state)

    print(f"  Expanded model: {N_old} → {new_num_nodes} nodes (init={init})")
    return new_model, all_node_indices


# ============================================================
# Fine-tuning
# ============================================================
def get_ft_params(model, method, old_node_indices, new_node_indices, all_node_indices):
    all_list = all_node_indices.tolist()
    old_local = [all_list.index(n) for n in old_node_indices]

    if method == "emb_all":
        for name, param in model.named_parameters():
            param.requires_grad = (name == NSP_KEY)
        optimizer = torch.optim.Adam([model.node_embeddings], lr=FT_LR)

        def cleanup():
            pass

    elif method == "emb_new_only":
        for name, param in model.named_parameters():
            param.requires_grad = (name == NSP_KEY)
        optimizer = torch.optim.Adam([model.node_embeddings], lr=FT_LR)

        def cleanup():
            if model.node_embeddings.grad is not None:
                model.node_embeddings.grad[old_local, :] = 0

    elif method == "full_model":
        for param in model.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(model.parameters(), lr=FT_LR)

        def cleanup():
            pass

    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Method={method}, trainable params={num_trainable:,}")
    return optimizer, cleanup


@torch.no_grad()
def compute_val_loss(model, val_loader, mean, std):
    model.eval()
    total_loss = 0
    count = 0
    for inputs, targets in val_loader:
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        inputs_z = (inputs - mean) / std
        targets_z = (targets.squeeze(-1) - mean) / std

        output = model(
            history_data=inputs_z, future_data=None,
            batch_seen=0, epoch=0, train=False
        )
        pred_z = output.squeeze(-1)
        loss = (pred_z - targets_z).abs().mean()
        total_loss += loss.item() * inputs.size(0)
        count += inputs.size(0)

    return total_loss / count


def finetune(model, train_loader, val_loader, method, old_node_indices, new_node_indices,
             all_node_indices, mean, std):
    optimizer, cleanup = get_ft_params(model, method, old_node_indices,
                                       new_node_indices, all_node_indices)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        total_loss = 0
        count = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            inputs_z = (inputs - mean) / std
            targets_z = (targets.squeeze(-1) - mean) / std

            output = model(
                history_data=inputs_z, future_data=None,
                batch_seen=0, epoch=epoch, train=True
            )
            pred_z = output.squeeze(-1)
            loss = (pred_z - targets_z).abs().mean()

            optimizer.zero_grad()
            loss.backward()
            cleanup()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            count += inputs.size(0)

        train_loss = total_loss / count

        val_loss = compute_val_loss(model, val_loader, mean, std)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{MAX_EPOCHS}, train={train_loss:.4f}, val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"    Early stop at epoch {epoch+1}, best_val={best_val_loss:.4f}")
                model.load_state_dict(best_state)
                return model, epoch + 1

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"    Max epochs reached ({MAX_EPOCHS}), best_val={best_val_loss:.4f}")
    return model, MAX_EPOCHS


# ============================================================
# Main
# ============================================================
def find_checkpoint(ckpt_dir):
    for root, dirs, files in os.walk(ckpt_dir):
        for f in files:
            if "best_val_MAE" in f:
                return os.path.join(root, f)
    for root, dirs, files in os.walk(ckpt_dir):
        pt_files = sorted([f for f in files if f.endswith(".pt") and "best" not in f])
        if pt_files:
            return os.path.join(root, pt_files[-1])
    return None


def run_experiment():
    print("=" * 60)
    print("Continual Node Expansion Experiment - AGCRN")
    print("=" * 60)

    # Load Q1 source model
    ckpt_dir = "checkpoints/Expanding_Source_AGCRN/SAN_BERNARDINO_2023_Q1_30_12_12"
    ckpt_path = find_checkpoint(ckpt_dir)
    if ckpt_path is None:
        print(f"ERROR: No checkpoint found in {ckpt_dir}")
        return

    print(f"\nLoading Q1 source model: {ckpt_path}")
    q1_nodes = np.load(f"{NODE_DIR}/nodes_q1.npy")
    source_model = AGCRN(num_nodes=len(q1_nodes), **MODEL_PARAM).to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    source_model.load_state_dict(state_dict)
    source_model.eval()
    print(f"  Loaded: {len(q1_nodes)} nodes, NSP shape: {source_model.node_embeddings.shape}")

    results = {}
    carry_forward = {method: source_model.state_dict() for method in METHODS}
    prev_nodes = {method: q1_nodes for method in METHODS}
    transitions = [("Q1", "Q2"), ("Q2", "Q3"), ("Q3", "Q4")]

    for src_q, tgt_q in transitions:
        print(f"\n{'='*60}")
        print(f"Transition: {src_q} → {tgt_q}")
        print(f"{'='*60}")

        tgt_data, tgt_nodes, new_nodes = load_quarter_data(tgt_q)
        mean, std = compute_zscore_stats(tgt_data)
        print(f"Target: {tgt_q}, nodes={len(tgt_nodes)}, new={len(new_nodes)}, zscore mean={mean:.2f} std={std:.2f}")

        qdata = precompute_quarter_data(tgt_data, tgt_nodes, batch_size=BATCH_SIZE)
        test_loader = qdata["test_loader"]
        del tgt_data

        for method in METHODS:
            print(f"\n--- Method: {method} ---")
            results_key = f"{method}/{tgt_q}"
            results[results_key] = {}

            old_nodes = prev_nodes[method]
            base_model = AGCRN(num_nodes=len(old_nodes), **MODEL_PARAM).to(DEVICE)
            base_model.load_state_dict(carry_forward[method])

            existing_nodes = old_nodes
            expanded_model, all_nodes = expand_model(base_model, old_nodes, new_nodes, init="mean")

            cold_start = evaluate(expanded_model, test_loader, tgt_nodes, existing_nodes, new_nodes, mean, std)
            results[results_key]["cold_start"] = cold_start
            print(f"  Cold-start: all={cold_start['mae_all']:.4f}, existing={cold_start['mae_existing']:.4f}, new={cold_start['mae_new']:.4f}")

            val_loader = qdata["val_loader"]
            for budget_name, budget_steps in BUDGETS.items():
                print(f"\n  Budget: {budget_name}")
                ft_model = copy.deepcopy(expanded_model)

                train_loader = make_train_loader(qdata, budget_steps=budget_steps, batch_size=BATCH_SIZE)

                ft_model, stopped_epoch = finetune(
                    ft_model, train_loader, val_loader, method, old_nodes, new_nodes,
                    all_nodes, mean, std
                )

                eval_result = evaluate(ft_model, test_loader, tgt_nodes, existing_nodes, new_nodes, mean, std)
                eval_result["forgetting"] = eval_result["mae_existing"] - cold_start["mae_existing"]
                eval_result["stopped_epoch"] = stopped_epoch
                results[results_key][budget_name] = eval_result
                print(f"  Result: all={eval_result['mae_all']:.4f}, existing={eval_result['mae_existing']:.4f}, "
                      f"new={eval_result['mae_new']:.4f}, forgetting={eval_result['forgetting']:+.4f}, epoch={stopped_epoch}")

                if budget_name == "full":
                    carry_forward[method] = ft_model.state_dict()
                    prev_nodes[method] = all_nodes

    # Save results
    save_dir = "project/concept_drift/results"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/results_agcrn.json"

    serializable = {}
    for k, v in results.items():
        serializable[k] = {}
        for kk, vv in v.items():
            serializable[k][kk] = {kkk: float(vvv) for kkk, vvv in vv.items()}

    with open(save_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {save_path}")

    print_summary(results)


def print_summary(results):
    print("\n" + "=" * 80)
    print("SUMMARY - AGCRN")
    print("=" * 80)

    for tgt_q in ["Q2", "Q3", "Q4"]:
        print(f"\n--- {tgt_q} ---")
        print(f"{'Method':<15} {'Budget':<8} {'MAE_all':<10} {'MAE_exist':<10} {'MAE_new':<10} {'Forget':<10} {'Epoch':<6}")
        print("-" * 69)

        for method in METHODS:
            key = f"{method}/{tgt_q}"
            if key not in results:
                continue

            cs = results[key]["cold_start"]
            print(f"{method:<15} {'cold':<8} {cs['mae_all']:<10.4f} {cs['mae_existing']:<10.4f} {cs['mae_new']:<10.4f} {'--':<10} {'--':<6}")

            for budget_name in BUDGETS:
                if budget_name in results[key]:
                    r = results[key][budget_name]
                    ep = int(r.get('stopped_epoch', 0))
                    print(f"{'':<15} {budget_name:<8} {r['mae_all']:<10.4f} {r['mae_existing']:<10.4f} "
                          f"{r['mae_new']:<10.4f} {r['forgetting']:+.4f}   {ep}")


if __name__ == "__main__":
    run_experiment()
