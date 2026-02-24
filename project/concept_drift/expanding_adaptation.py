"""Continual Node Expansion Experiment

Q1(558) → Q2(669) → Q3(781) → Q4(893)
Methods: emb_all, emb_new_only, full_model, pb_adapt
Budgets: 3h, 12h, 1d, 7d, full (time-based)
Evaluation: MAE (all/existing/new), forgetting, cold-start
"""

import os
import sys
import copy
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.cluster import KMeans

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from baselines.STAEformer.arch import STAEformer

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
PATIENCE = 10  # early stopping patience
FT_LR = 1e-4
PB_K = 8  # number of prototypes for pb_adapt

# Budget in hours → number of 5-min steps
BUDGETS = {
    "3h": 36,
    "12h": 144,
    "1d": 288,
    "7d": 2016,
    "full": None,  # use all training data
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

METHODS = ["emb_all", "emb_new_only", "full_model", "pb_adapt"]


# ============================================================
# Data utilities
# ============================================================
def load_quarter_data(quarter_name):
    """Load raw data and compute ZScore stats for a quarter."""
    cfg = QUARTERS[quarter_name]
    with open(f"{cfg['data']}/desc.json") as f:
        desc = json.load(f)
    data = np.memmap(f"{cfg['data']}/data.dat", dtype="float32", mode="r",
                     shape=tuple(desc["shape"]))
    data = np.array(data)  # load into memory

    node_indices = np.load(cfg["nodes"])
    new_indices = np.load(cfg["new"]) if "new" in cfg else np.array([], dtype=int)

    return data, node_indices, new_indices


def compute_zscore_stats(data, train_ratio=0.6):
    """Compute global mean/std from training portion of flow (channel 0)."""
    train_size = int(len(data) * train_ratio)
    train_flow = data[:train_size, :, 0]
    mean = float(np.mean(train_flow))
    std = float(np.std(train_flow))
    if std == 0:
        std = 1.0
    return mean, std


def create_windows(data, node_indices, input_len=12, output_len=12):
    """Create sliding windows from time series data.

    Returns:
        inputs: (num_windows, input_len, num_nodes, 5)
        targets: (num_windows, output_len, num_nodes, 5)
    """
    total_len = input_len + output_len
    num_windows = len(data) - total_len + 1
    # Filter nodes
    data_filtered = data[:, node_indices, :]

    inputs = np.stack([data_filtered[i:i+input_len] for i in range(num_windows)])
    targets = np.stack([data_filtered[i+input_len:i+total_len] for i in range(num_windows)])
    return inputs, targets


def precompute_quarter_data(data, node_indices, batch_size=16):
    """Precompute windows and split into train/val/test tensors once per quarter.

    Returns dict with train_inputs, train_targets, test_loader, val_loader, train_end.
    """
    inputs, targets = create_windows(data, node_indices, INPUT_LEN, OUTPUT_LEN)
    num_total = len(inputs)
    train_end = int(num_total * TRAIN_RATIO)
    val_end = int(num_total * (TRAIN_RATIO + VAL_RATIO))

    # Keep train data as numpy (will slice per budget), convert val/test to loaders
    train_inputs = inputs[:train_end]
    train_targets = targets[:train_end]

    def to_loader(start, end, shuffle=False):
        inp = torch.tensor(inputs[start:end], dtype=torch.float32)
        tgt = torch.tensor(targets[start:end], dtype=torch.float32)
        return DataLoader(TensorDataset(inp, tgt), batch_size=batch_size, shuffle=shuffle)

    val_loader = to_loader(train_end, val_end)
    test_loader = to_loader(val_end, num_total)

    print(f"  Windows: total={num_total}, train={train_end}, val={val_end-train_end}, test={num_total-val_end}")

    # Free full arrays (val/test now in torch tensors)
    del inputs, targets

    return {
        "train_inputs": train_inputs,
        "train_targets": train_targets,
        "train_end": train_end,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }


def make_train_loader(qdata, budget_steps=None, batch_size=16):
    """Create train DataLoader with budget limiting from precomputed data."""
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
# ZScore + Instance Norm utilities
# ============================================================
def zscore_transform(tensor, mean, std):
    """Apply ZScore to flow channel (0)."""
    out = tensor.clone()
    out[..., 0] = (out[..., 0] - mean) / std
    return out


def zscore_inverse(tensor, mean, std):
    """Inverse ZScore for flow channel (0)."""
    out = tensor.clone()
    out[..., 0] = out[..., 0] * std + mean
    return out


def instance_norm_forward(model, history_data, mean, std):
    """Forward pass with instance norm (replicates InstanceNormRunner).

    Args:
        model: STAEformer
        history_data: (B, T, N, 5) raw features (already ZScore'd)
        mean, std: ZScore stats for inverse transform

    Returns:
        prediction: (B, T_out, N, 1) in raw space
    """
    # Instance norm on flow (channel 0)
    flow = history_data[:, :, :, 0]  # (B, T, N)
    flow_mean = flow.mean(dim=1, keepdim=True)  # (B, 1, N)
    flow_std = flow.std(dim=1, keepdim=True) + 1e-5  # (B, 1, N)

    history_normed = history_data.clone()
    history_normed[:, :, :, 0] = (flow - flow_mean) / flow_std

    # Select input features (first 3 channels: flow, occ, speed)
    # tod and dow are at index 3, 4
    model_input = history_normed[..., :5]  # all 5 features passed

    # Forward
    output = model(
        history_data=model_input,
        future_data=None,
        batch_seen=0, epoch=0, train=False
    )
    pred = output["prediction"]  # (B, T_out, N, 1)

    # De-instance-norm
    pred = pred * flow_std.unsqueeze(-1) + flow_mean.unsqueeze(-1)

    # De-ZScore
    pred = pred * std + mean

    return pred


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def evaluate(model, test_loader, node_indices, existing_indices, new_indices,
             mean, std):
    """Evaluate model on test set.

    Returns dict with mae_all, mae_existing, mae_new.
    """
    model.eval()

    # Map global node indices to local positions
    node_list = list(node_indices)
    existing_local = [node_list.index(n) for n in existing_indices if n in node_list]
    new_local = [node_list.index(n) for n in new_indices if n in node_list]

    all_errors = []
    for inputs, targets in test_loader:
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        # ZScore transform
        inputs_z = zscore_transform(inputs, mean, std)
        targets_raw = targets[..., 0]  # (B, T, N) raw flow

        # Forward with instance norm
        pred_raw = instance_norm_forward(model, inputs_z, mean, std)  # (B, T, N, 1)
        pred_raw = pred_raw.squeeze(-1)  # (B, T, N)

        errors = (pred_raw - targets_raw).abs()  # (B, T, N)
        all_errors.append(errors.cpu())

    all_errors = torch.cat(all_errors, dim=0)  # (num_test, T, N)
    mae_all = all_errors.mean().item()
    mae_existing = all_errors[:, :, existing_local].mean().item() if existing_local else float("nan")
    mae_new = all_errors[:, :, new_local].mean().item() if new_local else float("nan")

    return {"mae_all": mae_all, "mae_existing": mae_existing, "mae_new": mae_new}


# ============================================================
# Model expansion
# ============================================================
def expand_model(model, old_node_indices, new_node_indices, init="mean"):
    """Expand model to include new nodes.

    Args:
        model: trained STAEformer with old_node_indices
        old_node_indices: array of node indices the model was trained on
        new_node_indices: array of new node indices to add
        init: initialization for new embeddings ("mean", "zero", "random")

    Returns:
        new_model: STAEformer with expanded nodes
        all_node_indices: combined sorted node indices
    """
    all_node_indices = np.sort(np.concatenate([old_node_indices, new_node_indices]))
    new_num_nodes = len(all_node_indices)

    # Create new model
    param = {**MODEL_PARAM, "num_nodes": new_num_nodes}
    new_model = STAEformer(**param).to(DEVICE)

    # Copy all weights except adaptive_embedding and output_proj
    old_state = model.state_dict()
    new_state = new_model.state_dict()

    for key in old_state:
        if key == "encoder.adaptive_embedding":
            continue
        if key in new_state and old_state[key].shape == new_state[key].shape:
            new_state[key] = old_state[key]

    # Expand adaptive_embedding: (T, N_old, D) → (T, N_new, D)
    old_emb = old_state["encoder.adaptive_embedding"]  # (T, N_old, D)
    T, N_old, D = old_emb.shape
    new_emb = torch.zeros(T, new_num_nodes, D, device=DEVICE)

    # Map old nodes to new positions
    all_list = all_node_indices.tolist()
    for i, node_id in enumerate(old_node_indices):
        new_pos = all_list.index(node_id)
        new_emb[:, new_pos, :] = old_emb[:, i, :]

    # Initialize new node embeddings
    if init == "mean":
        mean_emb = old_emb.mean(dim=1, keepdim=True)  # (T, 1, D)
        for node_id in new_node_indices:
            new_pos = all_list.index(node_id)
            new_emb[:, new_pos, :] = mean_emb.squeeze(1)
    elif init == "zero":
        pass  # already zeros
    elif init == "random":
        for node_id in new_node_indices:
            new_pos = all_list.index(node_id)
            nn.init.xavier_uniform_(new_emb[:, new_pos:new_pos+1, :])

    new_state["encoder.adaptive_embedding"] = new_emb
    new_model.load_state_dict(new_state)
    new_model.num_nodes = new_num_nodes

    print(f"  Expanded model: {N_old} → {new_num_nodes} nodes (init={init})")
    return new_model, all_node_indices


# ============================================================
# PB Adapt: post-hoc PB factorization
# ============================================================
class PBEmbedding(nn.Module):
    """Drop-in replacement for adaptive_embedding that computes from PB.

    Mimics nn.Parameter interface: has .shape and .expand() so STAEformer
    forward code works without modification.
    """
    def __init__(self, prototypes, weights, T, N, D):
        super().__init__()
        self.register_buffer("prototypes", prototypes)  # (K, T*D), frozen
        self.weights = nn.Parameter(weights)  # (N, K), trainable
        self._T = T
        self._N = N
        self._D = D
        self._shape = torch.Size([T, N, D])

    @property
    def shape(self):
        return self._shape

    def expand(self, *size):
        w = torch.softmax(self.weights, dim=-1)  # (N, K)
        emb = w @ self.prototypes  # (N, T*D)
        emb = emb.view(self._N, self._T, self._D).permute(1, 0, 2)  # (T, N, D)
        return emb.unsqueeze(0).expand(*size)


def factorize_to_pb(model, old_node_indices, new_node_indices, all_node_indices, K=8):
    """Convert adaptive_embedding to PB factorization.

    1. K-means on existing embeddings → prototypes
    2. Compute mixing weights for existing nodes
    3. Uniform weights for new nodes
    4. Replace adaptive_embedding with PBEmbedding

    Returns modified model (in-place).
    """
    emb = model.encoder.adaptive_embedding  # nn.Parameter (T, N, D)
    T, N, D = emb.shape

    all_list = all_node_indices.tolist()
    old_local = [all_list.index(n) for n in old_node_indices]
    new_local = [all_list.index(n) for n in new_node_indices]

    # Reshape existing embeddings for K-means: (N_old, T*D)
    old_emb = emb.data[:, old_local, :].permute(1, 0, 2).reshape(len(old_local), T * D)
    old_emb_np = old_emb.cpu().numpy()

    # K-means
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    kmeans.fit(old_emb_np)
    prototypes = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=DEVICE)  # (K, T*D)

    # Compute soft mixing weights for existing nodes (inverse distance → softmax)
    dists = torch.cdist(old_emb.to(DEVICE), prototypes)  # (N_old, K)
    weights_init = torch.zeros(N, K, device=DEVICE)
    weights_init[old_local] = -dists  # negative distance → softmax gives higher weight to closer
    # New nodes: all zeros → softmax = uniform (1/K)

    # Replace nn.Parameter with nn.Module (must remove from _parameters first)
    pb_emb = PBEmbedding(prototypes, weights_init, T, N, D)
    del model.encoder._parameters['adaptive_embedding']
    model.encoder.adaptive_embedding = pb_emb

    print(f"  PB factorization: K={K}, prototypes={prototypes.shape}, weights={weights_init.shape}")
    return model


# ============================================================
# Fine-tuning methods
# ============================================================
def get_ft_params(model, method, old_node_indices, new_node_indices, all_node_indices):
    """Get trainable parameters and optimizer for a given FT method.

    Returns (optimizer, cleanup_fn).
    cleanup_fn is called after each backward to handle gradient masking.
    """
    all_list = all_node_indices.tolist()
    new_local = [all_list.index(n) for n in new_node_indices]
    old_local = [all_list.index(n) for n in old_node_indices]

    if method == "emb_all":
        # Core frozen, all embeddings trainable
        for name, param in model.named_parameters():
            param.requires_grad = (name == "encoder.adaptive_embedding")
        optimizer = torch.optim.Adam([model.encoder.adaptive_embedding], lr=FT_LR)

        def cleanup():
            pass

    elif method == "emb_new_only":
        # Core frozen, only new node embeddings trainable (gradient masking)
        for name, param in model.named_parameters():
            param.requires_grad = (name == "encoder.adaptive_embedding")
        optimizer = torch.optim.Adam([model.encoder.adaptive_embedding], lr=FT_LR)

        def cleanup():
            if model.encoder.adaptive_embedding.grad is not None:
                model.encoder.adaptive_embedding.grad[:, old_local, :] = 0

    elif method == "full_model":
        # Everything trainable
        for param in model.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(model.parameters(), lr=FT_LR)

        def cleanup():
            pass

    elif method == "pb_adapt":
        # Only PB weights trainable (prototypes frozen)
        for name, param in model.named_parameters():
            param.requires_grad = False
        model.encoder.adaptive_embedding.weights.requires_grad = True
        optimizer = torch.optim.Adam([model.encoder.adaptive_embedding.weights], lr=FT_LR)

        def cleanup():
            pass

    else:
        raise ValueError(f"Unknown method: {method}")

    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Method={method}, trainable params={num_trainable:,}")
    return optimizer, cleanup


@torch.no_grad()
def compute_val_loss(model, val_loader, mean, std):
    """Compute validation MAE in ZScore space."""
    model.eval()
    total_loss = 0
    count = 0
    for inputs, targets in val_loader:
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        inputs_z = zscore_transform(inputs, mean, std)
        targets_z = (targets[..., 0] - mean) / std

        flow = inputs_z[:, :, :, 0]
        flow_mean = flow.mean(dim=1, keepdim=True)
        flow_std = flow.std(dim=1, keepdim=True) + 1e-5

        inputs_normed = inputs_z.clone()
        inputs_normed[:, :, :, 0] = (flow - flow_mean) / flow_std

        output = model(
            history_data=inputs_normed, future_data=None,
            batch_seen=0, epoch=0, train=False
        )
        pred = output["prediction"]
        pred = pred * flow_std.unsqueeze(-1) + flow_mean.unsqueeze(-1)
        pred_z = pred.squeeze(-1)

        loss = (pred_z - targets_z).abs().mean()
        total_loss += loss.item() * inputs.size(0)
        count += inputs.size(0)

    return total_loss / count


def finetune(model, train_loader, val_loader, method, old_node_indices, new_node_indices,
             all_node_indices, mean, std):
    """Fine-tune model with early stopping."""
    optimizer, cleanup = get_ft_params(model, method, old_node_indices,
                                       new_node_indices, all_node_indices)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        # ---- Train ----
        model.train()
        total_loss = 0
        count = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            # ZScore
            inputs_z = zscore_transform(inputs, mean, std)
            targets_z = (targets[..., 0] - mean) / std  # (B, T, N) in ZScore space

            # Instance norm on flow
            flow = inputs_z[:, :, :, 0]
            flow_mean = flow.mean(dim=1, keepdim=True)
            flow_std = flow.std(dim=1, keepdim=True) + 1e-5

            inputs_normed = inputs_z.clone()
            inputs_normed[:, :, :, 0] = (flow - flow_mean) / flow_std

            # Forward
            output = model(
                history_data=inputs_normed, future_data=None,
                batch_seen=0, epoch=epoch, train=True
            )
            pred = output["prediction"]  # (B, T_out, N, 1)

            # De-instance-norm (prediction now in ZScore'd space)
            pred = pred * flow_std.unsqueeze(-1) + flow_mean.unsqueeze(-1)
            pred_z = pred.squeeze(-1)  # (B, T, N)

            # Loss: MAE in ZScore space (consistent with source training)
            loss = (pred_z - targets_z).abs().mean()

            optimizer.zero_grad()
            loss.backward()
            cleanup()  # gradient masking for emb_new_only
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            count += inputs.size(0)

        train_loss = total_loss / count

        # ---- Validation ----
        val_loss = compute_val_loss(model, val_loader, mean, std)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{MAX_EPOCHS}, train={train_loss:.4f}, val={val_loss:.4f}")

        # ---- Early stopping ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                stopped_epoch = epoch + 1
                print(f"    Early stop at epoch {stopped_epoch}, best_val={best_val_loss:.4f}")
                model.load_state_dict(best_state)
                return model, stopped_epoch

    # Ran all epochs without early stop
    stopped_epoch = MAX_EPOCHS
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"    Max epochs reached ({MAX_EPOCHS}), best_val={best_val_loss:.4f}")
    return model, stopped_epoch


# ============================================================
# Main experiment
# ============================================================
def find_checkpoint(ckpt_dir):
    """Find the best model checkpoint in a directory."""
    for root, dirs, files in os.walk(ckpt_dir):
        for f in files:
            if "best_val_MAE" in f:
                return os.path.join(root, f)
    # Fallback: find latest .pt file
    for root, dirs, files in os.walk(ckpt_dir):
        pt_files = sorted([f for f in files if f.endswith(".pt") and "best" not in f])
        if pt_files:
            return os.path.join(root, pt_files[-1])
    return None


def run_experiment():
    print("=" * 60)
    print("Continual Node Expansion Experiment")
    print("=" * 60)

    # ---- Load Q1 source model ----
    ckpt_dir = "checkpoints/Expanding_Source/SAN_BERNARDINO_2023_Q1_30_12_12"
    ckpt_path = find_checkpoint(ckpt_dir)
    if ckpt_path is None:
        print(f"ERROR: No checkpoint found in {ckpt_dir}")
        print("Run the Q1 source training first.")
        return

    print(f"\nLoading Q1 source model: {ckpt_path}")
    q1_nodes = np.load(f"{NODE_DIR}/nodes_q1.npy")
    source_model = STAEformer(num_nodes=len(q1_nodes), **MODEL_PARAM).to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    source_model.load_state_dict(state_dict)
    source_model.eval()
    print(f"  Loaded: {len(q1_nodes)} nodes")

    # ---- Results storage ----
    results = {}

    # ---- Continual loop ----
    # Each method has its own chain
    carry_forward = {}  # method → model state_dict after full-budget FT
    for method in METHODS:
        carry_forward[method] = source_model.state_dict()

    prev_nodes = {method: q1_nodes for method in METHODS}
    transitions = [("Q1", "Q2"), ("Q2", "Q3"), ("Q3", "Q4")]

    for src_q, tgt_q in transitions:
        print(f"\n{'='*60}")
        print(f"Transition: {src_q} → {tgt_q}")
        print(f"{'='*60}")

        tgt_data, tgt_nodes, new_nodes = load_quarter_data(tgt_q)
        mean, std = compute_zscore_stats(tgt_data)
        print(f"Target: {tgt_q}, nodes={len(tgt_nodes)}, new={len(new_nodes)}, zscore mean={mean:.2f} std={std:.2f}")

        # Precompute windows once per quarter
        qdata = precompute_quarter_data(tgt_data, tgt_nodes, batch_size=BATCH_SIZE)
        test_loader = qdata["test_loader"]
        del tgt_data  # free raw data

        for method in METHODS:
            print(f"\n--- Method: {method} ---")
            results_key = f"{method}/{tgt_q}"
            results[results_key] = {}

            # Load carry-forward model
            old_nodes = prev_nodes[method]
            base_model = STAEformer(num_nodes=len(old_nodes), **MODEL_PARAM).to(DEVICE)
            base_model.load_state_dict(carry_forward[method])

            # Expand model
            existing_nodes = old_nodes
            expanded_model, all_nodes = expand_model(base_model, old_nodes, new_nodes, init="mean")

            # For pb_adapt: factorize before FT
            if method == "pb_adapt":
                expanded_model = factorize_to_pb(expanded_model, old_nodes, new_nodes, all_nodes, K=PB_K)

            # Cold-start evaluation (before any FT)
            cold_start = evaluate(expanded_model, test_loader, tgt_nodes, existing_nodes, new_nodes, mean, std)
            results[results_key]["cold_start"] = cold_start
            print(f"  Cold-start: all={cold_start['mae_all']:.4f}, existing={cold_start['mae_existing']:.4f}, new={cold_start['mae_new']:.4f}")

            # Fine-tune with different budgets
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

                # Carry forward the full-budget model
                if budget_name == "full":
                    # For pb_adapt, need to convert back to standard embedding for next expansion
                    if method == "pb_adapt":
                        # Compute final embedding from PB and store as nn.Parameter
                        with torch.no_grad():
                            pb = ft_model.encoder.adaptive_embedding
                            w = torch.softmax(pb.weights, dim=-1)
                            emb = w @ pb.prototypes
                            emb = emb.view(pb._N, pb._T, pb._D).permute(1, 0, 2)
                        # Create a clean model with standard embedding
                        clean_model = STAEformer(num_nodes=len(all_nodes), **MODEL_PARAM).to(DEVICE)
                        state = ft_model.state_dict()
                        # Remove PB keys, add adaptive_embedding
                        clean_state = {k: v for k, v in state.items()
                                      if not k.startswith("encoder.adaptive_embedding")}
                        clean_state["encoder.adaptive_embedding"] = emb
                        clean_model.load_state_dict(clean_state)
                        carry_forward[method] = clean_model.state_dict()
                    else:
                        carry_forward[method] = ft_model.state_dict()
                    prev_nodes[method] = all_nodes

    # ---- Save results ----
    save_dir = "project/concept_drift/results"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/results_v2_earlystop.json"

    # Convert to serializable
    serializable = {}
    for k, v in results.items():
        serializable[k] = {}
        for kk, vv in v.items():
            serializable[k][kk] = {kkk: float(vvv) for kkk, vvv in vv.items()}

    with open(save_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {save_path}")

    # ---- Print summary table ----
    print_summary(results)


def print_summary(results):
    """Print a formatted summary table."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for tgt_q in ["Q2", "Q3", "Q4"]:
        print(f"\n--- {tgt_q} ---")
        print(f"{'Method':<15} {'Budget':<8} {'MAE_all':<10} {'MAE_exist':<10} {'MAE_new':<10} {'Forget':<10} {'Epoch':<6}")
        print("-" * 69)

        for method in METHODS:
            key = f"{method}/{tgt_q}"
            if key not in results:
                continue

            # Cold start
            cs = results[key]["cold_start"]
            print(f"{method:<15} {'cold':<8} {cs['mae_all']:<10.4f} {cs['mae_existing']:<10.4f} {cs['mae_new']:<10.4f} {'--':<10} {'--':<6}")

            # Budgets
            for budget_name in BUDGETS:
                if budget_name in results[key]:
                    r = results[key][budget_name]
                    ep = int(r.get('stopped_epoch', 0))
                    print(f"{'':<15} {budget_name:<8} {r['mae_all']:<10.4f} {r['mae_existing']:<10.4f} "
                          f"{r['mae_new']:<10.4f} {r['forgetting']:+.4f}   {ep}")


if __name__ == "__main__":
    run_experiment()
