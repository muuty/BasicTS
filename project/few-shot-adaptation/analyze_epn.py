"""Analyze EPN prediction quality vs mean baseline."""
import torch
import numpy as np
import os
import sys
import json
import pickle

sys.path.append(os.path.dirname(__file__))
from epn_model import EmbeddingPredictorNetwork

# Load source NSP
ckpt_dir = "checkpoints/Expanding_Source/SAN_BERNARDINO_2023_Q1_30_12_12"
hash_dirs = [d for d in os.listdir(ckpt_dir)
             if os.path.isdir(os.path.join(ckpt_dir, d)) and not d.startswith('.')]
ckpt = torch.load(os.path.join(ckpt_dir, hash_dirs[0], "STAEformer_best_val_MAE.pt"),
                  map_location="cpu")
nsp = ckpt["model_state_dict"]["encoder.adaptive_embedding"]  # (12, 558, 24)
nsp_flat = nsp.permute(1, 0, 2).reshape(558, -1)  # (558, 288)
mean_nsp = nsp_flat.mean(dim=0)

# Load EPN 3h
epn = EmbeddingPredictorNetwork(nsp_dim=288, num_channels=5, d_hidden=64)
epn.load_state_dict(torch.load("project/few-shot-adaptation/results/epn_3h.pt", map_location="cpu"))
epn.eval()

# Load data
node_indices = np.load("datasets/expanding_experiment/nodes_q1.npy")
data_dir = "datasets/expanding_experiment/SAN_BERNARDINO_2023_Q1"
with open(f"{data_dir}/desc.json") as f:
    desc = json.load(f)
raw = np.memmap(f"{data_dir}/data.dat", dtype="float32", mode="r",
                shape=tuple(desc["shape"]))
raw_filtered = np.array(raw[:, node_indices, :])
print(f"raw_filtered shape: {raw_filtered.shape}")

# Load adj
with open("datasets/xtraffic/SAN_BERNARDINO/adj_mx.pkl", "rb") as f:
    adj_data = pickle.load(f, encoding="latin1")
if isinstance(adj_data, list):
    adj_full = np.array(adj_data[2], dtype=np.float32)
else:
    adj_full = np.array(adj_data, dtype=np.float32)
print(f"adj_full shape: {adj_full.shape}")
adj_filtered = adj_full[np.ix_(node_indices, node_indices)]
np.fill_diagonal(adj_filtered, 0)
print(f"adj_filtered shape: {adj_filtered.shape}")

# Check predictions for all nodes
F = torch.nn.functional
epn_mses = []
mean_mses = []
cos_pred_trues = []
cos_pred_means = []

for node_idx in range(558):
    traffic = torch.tensor(raw_filtered[:36, node_idx, :], dtype=torch.float32).unsqueeze(0)  # (1, 36, 5)

    row = adj_filtered[node_idx].copy()
    row[node_idx] = 0
    topk = np.argsort(row)[-10:]
    weights = row[topk]

    n_nsps = nsp_flat[topk].unsqueeze(0)
    adj_w = torch.tensor(weights, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred = epn(traffic, n_nsps, adj_w).squeeze(0)

    true_nsp = nsp_flat[node_idx]
    mse_epn = ((pred - true_nsp) ** 2).mean().item()
    mse_mean = ((mean_nsp - true_nsp) ** 2).mean().item()
    cos_pt = F.cosine_similarity(pred.unsqueeze(0), true_nsp.unsqueeze(0)).item()
    cos_pm = F.cosine_similarity(pred.unsqueeze(0), mean_nsp.unsqueeze(0)).item()

    epn_mses.append(mse_epn)
    mean_mses.append(mse_mean)
    cos_pred_trues.append(cos_pt)
    cos_pred_means.append(cos_pm)

    if node_idx in [0, 100, 200, 300, 500]:
        print(f"  Node {node_idx:3d}: MSE(EPN)={mse_epn:.6f}, MSE(mean)={mse_mean:.6f}, "
              f"cos(pred,true)={cos_pt:.4f}, cos(pred,mean)={cos_pm:.4f}")

epn_mses = np.array(epn_mses)
mean_mses = np.array(mean_mses)
cos_pred_trues = np.array(cos_pred_trues)
cos_pred_means = np.array(cos_pred_means)

print(f"\n{'='*60}")
print(f"Summary over all 558 nodes")
print(f"{'='*60}")
print(f"MSE(EPN, true):  mean={epn_mses.mean():.6f}, std={epn_mses.std():.6f}")
print(f"MSE(mean, true): mean={mean_mses.mean():.6f}, std={mean_mses.std():.6f}")
print(f"EPN better than mean: {(epn_mses < mean_mses).sum()}/558 nodes")
print(f"cos(pred, true):  mean={cos_pred_trues.mean():.4f}, std={cos_pred_trues.std():.4f}")
print(f"cos(pred, mean):  mean={cos_pred_means.mean():.4f}, std={cos_pred_means.std():.4f}")
print(f"  (cos(pred,mean) close to 1.0 means EPN is just predicting the mean)")

# Check EPN prediction norms vs true
sample_preds = []
for i in range(10):
    traffic = torch.tensor(raw_filtered[:36, i, :], dtype=torch.float32).unsqueeze(0)
    row = adj_filtered[i].copy()
    row[i] = 0
    topk = np.argsort(row)[-10:]
    weights = row[topk]
    n_nsps = nsp_flat[topk].unsqueeze(0)
    adj_w = torch.tensor(weights, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        pred = epn(traffic, n_nsps, adj_w).squeeze(0)
    sample_preds.append(pred)

sample_preds = torch.stack(sample_preds)
print(f"\nEPN pred L2 norm: {sample_preds.norm(dim=1).mean():.4f}")
print(f"Mean NSP L2 norm: {mean_nsp.norm():.4f}")
print(f"True NSP avg L2 norm: {nsp_flat.norm(dim=1).mean():.4f}")

# Check: are all EPN predictions similar to each other?
pred_var = sample_preds.var(dim=0).mean().item()
true_var = nsp_flat[:10].var(dim=0).mean().item()
print(f"\nVariance across EPN preds (10 nodes): {pred_var:.6f}")
print(f"Variance across true NSPs (10 nodes): {true_var:.6f}")
print(f"Ratio: {pred_var/true_var:.4f} (<<1 means EPN predictions collapse to similar values)")
