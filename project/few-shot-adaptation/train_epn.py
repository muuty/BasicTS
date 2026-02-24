"""Train EPN via Synthetic Episodes

For each episode:
1. Pick a random existing node i
2. Mask its NSP (adaptive_embedding)
3. Input: node i's traffic data (budget-sized) + neighbor NSPs
4. Target: node i's learned NSP
5. Loss: MSE

Trains one EPN per budget (3h, 12h, 1d, 7d).
"""

import os
import sys
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.dirname(__file__))

from epn_model import EmbeddingPredictorNetwork

# ============================================================
# Config
# ============================================================
DEVICE = "cuda:1"
DATA_DIR = "datasets/expanding_experiment"
ADJ_PATH = "datasets/xtraffic/SAN_BERNARDINO/adj_mx.pkl"
CKPT_DIR = "checkpoints/Expanding_Source/SAN_BERNARDINO_2023_Q1_30_12_12"

INPUT_LEN = 12
OUTPUT_LEN = 12
NUM_CHANNELS = 5  # flow, occ, speed, tod, dow
NSP_T = 12  # adaptive_embedding T dimension
NSP_D = 24  # adaptive_embedding D dimension
NSP_DIM = NSP_T * NSP_D  # 288, flattened per-node NSP

MAX_NEIGHBORS = 10  # max neighbors to consider per node

BUDGETS = {
    "3h": 36,
    "12h": 144,
    "1d": 288,
    "7d": 2016,
}

# EPN training hyperparams
EPN_EPOCHS = 100
EPN_LR = 1e-3
EPN_BATCH = 32
EPN_PATIENCE = 15
EPN_D_HIDDEN = 64

RESULTS_DIR = "project/few-shot-adaptation/results"


# ============================================================
# Data loading
# ============================================================
def load_adjacency(adj_path, node_indices):
    """Load adjacency matrix and filter to node_indices.

    Note: This is a Gaussian kernel distance matrix (dense).
    Sparsification is handled by top-K selection in get_neighbor_info.
    """
    with open(adj_path, "rb") as f:
        adj_data = pickle.load(f, encoding="latin1")
    if isinstance(adj_data, list):
        adj_mx = adj_data[2]
    else:
        adj_mx = adj_data
    adj_mx = np.array(adj_mx, dtype=np.float32)
    adj_filtered = adj_mx[np.ix_(node_indices, node_indices)]
    np.fill_diagonal(adj_filtered, 0)
    return adj_filtered  # (N, N)


def load_source_model_nsp(ckpt_dir):
    """Load the learned NSP (adaptive_embedding) from source model checkpoint."""
    # Find the hash dir
    hash_dirs = [d for d in os.listdir(ckpt_dir)
                 if os.path.isdir(os.path.join(ckpt_dir, d)) and not d.startswith('.')]
    ckpt_path = os.path.join(ckpt_dir, hash_dirs[0], "STAEformer_best_val_MAE.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # Handle wrapped checkpoints (epoch, model_state_dict, ...)
    state_dict = ckpt.get("model_state_dict", ckpt)
    for key in ["encoder.adaptive_embedding", "adaptive_embedding"]:
        if key in state_dict:
            return state_dict[key]  # (T=12, N=558, D=24)
    raise KeyError(f"adaptive_embedding not found. Keys: {list(state_dict.keys())[:10]}")


def load_quarter_data(quarter_name):
    """Load raw data for a quarter."""
    data_dir = f"{DATA_DIR}/{quarter_name}"
    with open(f"{data_dir}/desc.json") as f:
        desc = json.load(f)
    data = np.memmap(f"{data_dir}/data.dat", dtype="float32", mode="r",
                     shape=tuple(desc["shape"]))
    return np.array(data)


def get_node_traffic_data(raw_data, node_local_idx, budget_steps, train_ratio=0.6):
    """Extract traffic data for a specific node within the training portion.

    Returns: (budget_steps, num_channels) array.
    If budget_steps > available training data, returns all training data.
    """
    train_end = int(len(raw_data) * train_ratio)
    node_data = raw_data[:train_end, node_local_idx, :]  # (train_steps, C)
    # Take first budget_steps (chronological)
    actual_steps = min(budget_steps, len(node_data))
    return node_data[:actual_steps]  # (actual_steps, C)


def get_neighbor_info(adj_mx, node_local_idx, max_neighbors=MAX_NEIGHBORS):
    """Get top-K neighbors and their adjacency weights for a node.

    Returns: neighbor_indices (list), adj_weights (list)
    """
    row = adj_mx[node_local_idx].copy()
    row[node_local_idx] = 0  # exclude self
    # Top-K by adjacency weight
    if (row > 0).sum() <= max_neighbors:
        neighbor_idx = np.where(row > 0)[0]
    else:
        neighbor_idx = np.argsort(row)[-max_neighbors:]
    weights = row[neighbor_idx]
    return neighbor_idx.tolist(), weights.tolist()


# ============================================================
# Dataset for synthetic episodes
# ============================================================
class SyntheticEpisodeDataset(Dataset):
    """Dataset of synthetic episodes for EPN training.

    Each item = one node: (traffic_data, neighbor_nsps, adj_weights, target_nsp)
    """

    def __init__(self, raw_data, nsp_tensor, adj_mx, node_indices, budget_steps,
                 num_episodes_per_node=1):
        """
        raw_data: (time_steps, N, C) - raw traffic data for Q1
        nsp_tensor: (T, N, D) - learned adaptive_embedding
        adj_mx: (N, N) - adjacency matrix (filtered to these nodes)
        node_indices: list of global node indices
        budget_steps: number of time steps for traffic data
        """
        self.raw_data = raw_data
        self.nsp_flat = nsp_tensor.permute(1, 0, 2).reshape(nsp_tensor.shape[1], -1)  # (N, T*D)
        self.adj_mx = adj_mx
        self.budget_steps = budget_steps
        self.num_nodes = len(node_indices)
        self.num_episodes = self.num_nodes * num_episodes_per_node

        # Precompute neighbor info for all nodes
        self.neighbor_info = []
        for i in range(self.num_nodes):
            n_idx, n_wts = get_neighbor_info(adj_mx, i)
            self.neighbor_info.append((n_idx, n_wts))

    def __len__(self):
        return self.num_episodes

    def __getitem__(self, idx):
        node_idx = idx % self.num_nodes

        # 1. Traffic data for this node (budget-sized)
        traffic = get_node_traffic_data(self.raw_data, node_idx, self.budget_steps)
        traffic = torch.tensor(traffic, dtype=torch.float32)  # (steps, C)

        # 2. Neighbor NSPs (masking this node's NSP)
        n_indices, n_weights = self.neighbor_info[node_idx]
        # Gather neighbor NSPs (these are NOT masked - they keep their learned values)
        neighbor_nsps = []
        adj_weights = []
        for ni, nw in zip(n_indices, n_weights):
            neighbor_nsps.append(self.nsp_flat[ni])
            adj_weights.append(nw)

        # Pad to max_neighbors
        pad_len = MAX_NEIGHBORS - len(neighbor_nsps)
        if neighbor_nsps:
            neighbor_nsps = torch.stack(neighbor_nsps)  # (K, nsp_dim)
        else:
            neighbor_nsps = torch.zeros(0, NSP_DIM)
        if pad_len > 0:
            neighbor_nsps = torch.cat([
                neighbor_nsps,
                torch.zeros(pad_len, NSP_DIM)
            ], dim=0)
        adj_weights = torch.tensor(adj_weights + [0.0] * pad_len, dtype=torch.float32)

        # 3. Target: this node's learned NSP
        target_nsp = self.nsp_flat[node_idx]  # (nsp_dim,)

        return traffic, neighbor_nsps, adj_weights, target_nsp


def collate_variable_length(batch):
    """Custom collate to handle variable-length traffic data."""
    traffics, neighbor_nsps, adj_weights, targets = zip(*batch)

    # Pad traffic to max length in batch
    max_len = max(t.shape[0] for t in traffics)
    padded_traffics = []
    for t in traffics:
        if t.shape[0] < max_len:
            pad = torch.zeros(max_len - t.shape[0], t.shape[1])
            t = torch.cat([t, pad], dim=0)
        padded_traffics.append(t)

    return (
        torch.stack(padded_traffics),       # (B, max_len, C)
        torch.stack(neighbor_nsps),          # (B, max_neighbors, nsp_dim)
        torch.stack(adj_weights),            # (B, max_neighbors)
        torch.stack(targets),                # (B, nsp_dim)
    )


# ============================================================
# Training
# ============================================================
def train_epn(budget_name, budget_steps, raw_data, nsp_tensor, adj_mx, node_indices):
    """Train one EPN for a specific budget."""
    print(f"\n{'='*60}")
    print(f"Training EPN for budget: {budget_name} ({budget_steps} steps)")
    print(f"{'='*60}")

    # Create dataset
    dataset = SyntheticEpisodeDataset(
        raw_data=raw_data,
        nsp_tensor=nsp_tensor,
        adj_mx=adj_mx,
        node_indices=node_indices,
        budget_steps=budget_steps,
        num_episodes_per_node=5,  # 5 episodes per node = 2790 total
    )

    # Train/val split (80/20 by node)
    num_nodes = len(node_indices)
    perm = np.random.RandomState(42).permutation(num_nodes)
    train_nodes = set(perm[:int(0.8 * num_nodes)])

    train_indices = [i for i in range(len(dataset)) if (i % num_nodes) in train_nodes]
    val_indices = [i for i in range(len(dataset)) if (i % num_nodes) not in train_nodes]

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=EPN_BATCH, shuffle=True,
                              collate_fn=collate_variable_length)
    val_loader = DataLoader(val_subset, batch_size=EPN_BATCH, shuffle=False,
                            collate_fn=collate_variable_length)

    print(f"  Train episodes: {len(train_indices)}, Val episodes: {len(val_indices)}")

    # Create EPN
    epn = EmbeddingPredictorNetwork(
        nsp_dim=NSP_DIM,
        num_channels=NUM_CHANNELS,
        d_hidden=EPN_D_HIDDEN,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(epn.parameters(), lr=EPN_LR)

    # Cosine similarity loss: learns direction, not magnitude
    # MSE fails because target std=0.051 → zero prediction ≈ actual prediction loss
    def cosine_loss(pred, target):
        return 1 - nn.functional.cosine_similarity(pred, target, dim=1).mean()

    # Training loop with early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(EPN_EPOCHS):
        # Train
        epn.train()
        train_loss = 0
        for traffic, n_nsps, adj_w, target in train_loader:
            traffic = traffic.to(DEVICE)
            n_nsps = n_nsps.to(DEVICE)
            adj_w = adj_w.to(DEVICE)
            target = target.to(DEVICE)

            pred = epn(traffic, n_nsps, adj_w)
            loss = cosine_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * traffic.shape[0]
        train_loss /= len(train_indices)

        # Validate
        epn.eval()
        val_loss = 0
        with torch.no_grad():
            for traffic, n_nsps, adj_w, target in val_loader:
                traffic = traffic.to(DEVICE)
                n_nsps = n_nsps.to(DEVICE)
                adj_w = adj_w.to(DEVICE)
                target = target.to(DEVICE)

                pred = epn(traffic, n_nsps, adj_w)
                loss = cosine_loss(pred, target)
                val_loss += loss.item() * traffic.shape[0]
        val_loss /= len(val_indices)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{EPN_EPOCHS}, train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in epn.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= EPN_PATIENCE:
                print(f"  Early stop at epoch {epoch+1}, best_val_loss={best_val_loss:.6f}")
                break

    if best_state is not None:
        epn.load_state_dict(best_state)
    epn = epn.to(DEVICE)

    print(f"  Final best val_loss: {best_val_loss:.6f}")
    return epn, best_val_loss


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load node indices for Q1
    node_indices = np.load(f"{DATA_DIR}/nodes_q1.npy")
    print(f"Q1 nodes: {len(node_indices)}")

    # Load adjacency
    adj_mx = load_adjacency(ADJ_PATH, node_indices)
    print(f"Adjacency: {adj_mx.shape}, non-zero: {(adj_mx > 0).sum()}")

    # Load learned NSP from source model
    nsp_tensor = load_source_model_nsp(CKPT_DIR)
    print(f"NSP shape: {nsp_tensor.shape}")  # (12, 558, 24)

    # Load Q1 raw data
    raw_data_full = load_quarter_data("SAN_BERNARDINO_2023_Q1")
    # Filter to Q1 nodes
    raw_data = raw_data_full[:, node_indices, :]
    print(f"Raw data: {raw_data.shape}")  # (T, 558, 5)

    # Train EPNs for each budget
    results = {}
    for budget_name, budget_steps in BUDGETS.items():
        epn, val_loss = train_epn(
            budget_name, budget_steps, raw_data, nsp_tensor, adj_mx, node_indices
        )

        # Save model
        save_path = os.path.join(RESULTS_DIR, f"epn_{budget_name}.pt")
        torch.save(epn.state_dict(), save_path)
        print(f"  Saved: {save_path}")

        results[budget_name] = {"val_mse": val_loss}

    # Save results summary
    with open(os.path.join(RESULTS_DIR, "epn_training_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll EPN training complete. Results: {results}")


if __name__ == "__main__":
    main()
