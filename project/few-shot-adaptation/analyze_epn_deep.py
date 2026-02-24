"""Deep node-wise analysis of EPN embedding quality.

Key questions:
1. How similar are actual NSPs across nodes? (Is mean_init already near-optimal?)
2. Per-node: how well did EPN predict each node's NSP?
3. What node properties correlate with EPN prediction quality?
4. What does the NSP space look like? (PCA structure, clusters)
5. Does NSP similarity correlate with graph adjacency?
6. EPN collapse analysis: diversity of predicted vs actual NSPs
"""

import os
import sys
import json
import pickle
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

DEVICE = "cuda:1"
DATA_DIR = "datasets/expanding_experiment"
FULL_DATA_DIR = "datasets/xtraffic/SAN_BERNARDINO"
CKPT_DIR = "checkpoints/Expanding_Source/SAN_BERNARDINO_2023_Q1_30_12_12"
EPN_DIR = "project/few-shot-adaptation/results"
RESULTS_DIR = "project/few-shot-adaptation/results/epn_analysis"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# 1. Load actual NSPs from source model
# ============================================================
print("=" * 70)
print("1. Loading actual NSPs from source model")
print("=" * 70)

hash_dirs = [d for d in os.listdir(CKPT_DIR)
             if os.path.isdir(os.path.join(CKPT_DIR, d)) and not d.startswith('.')]
ckpt_path = os.path.join(CKPT_DIR, hash_dirs[0], "STAEformer_best_val_MAE.pt")
ckpt = torch.load(ckpt_path, map_location="cpu")
state_dict = ckpt["model_state_dict"]

# adaptive_embedding: (T=12, N=558, D=24)
actual_nsp_3d = state_dict["encoder.adaptive_embedding"].numpy()
T, N, D = actual_nsp_3d.shape
print(f"  NSP shape: ({T}, {N}, {D})")

# Flatten to (N, T*D=288)
actual_nsp = actual_nsp_3d.reshape(N, -1)  # (558, 288)
nsp_dim = actual_nsp.shape[1]
print(f"  Flattened: ({N}, {nsp_dim})")

# ============================================================
# 2. NSP space analysis: how similar are nodes' NSPs?
# ============================================================
print("\n" + "=" * 70)
print("2. NSP space analysis")
print("=" * 70)

# Mean NSP and distance from mean
mean_nsp = actual_nsp.mean(axis=0)  # (288,)
dist_from_mean = np.linalg.norm(actual_nsp - mean_nsp, axis=1)  # (558,)
nsp_norms = np.linalg.norm(actual_nsp, axis=1)  # (558,)

print(f"  Mean NSP norm: {np.linalg.norm(mean_nsp):.4f}")
print(f"  Per-node NSP norm: mean={nsp_norms.mean():.4f}, std={nsp_norms.std():.4f}, "
      f"min={nsp_norms.min():.4f}, max={nsp_norms.max():.4f}")
print(f"  Distance from mean: mean={dist_from_mean.mean():.4f}, std={dist_from_mean.std():.4f}, "
      f"min={dist_from_mean.min():.4f}, max={dist_from_mean.max():.4f}")

# Pairwise cosine similarity
from numpy.linalg import norm
nsp_normed = actual_nsp / (norm(actual_nsp, axis=1, keepdims=True) + 1e-8)
cos_sim_matrix = nsp_normed @ nsp_normed.T  # (N, N)
# Upper triangle (exclude diagonal)
triu_idx = np.triu_indices(N, k=1)
pairwise_cos = cos_sim_matrix[triu_idx]
print(f"\n  Pairwise cosine similarity:")
print(f"    mean={pairwise_cos.mean():.4f}, std={pairwise_cos.std():.4f}")
print(f"    min={pairwise_cos.min():.4f}, max={pairwise_cos.max():.4f}")
print(f"    median={np.median(pairwise_cos):.4f}")

# PCA analysis: how many dimensions explain most variance?
from numpy.linalg import svd
nsp_centered = actual_nsp - mean_nsp
U, S, Vt = svd(nsp_centered, full_matrices=False)
explained_var = (S ** 2) / (S ** 2).sum()
cumvar = np.cumsum(explained_var)
for threshold in [0.5, 0.8, 0.9, 0.95, 0.99]:
    n_comp = np.searchsorted(cumvar, threshold) + 1
    print(f"  PCA: {n_comp} components explain {threshold*100:.0f}% variance")

# ============================================================
# 3. Load EPN predictions and per-node analysis
# ============================================================
print("\n" + "=" * 70)
print("3. EPN per-node prediction quality")
print("=" * 70)

sys.path.append(os.path.dirname(__file__))
from epn_model import EmbeddingPredictorNetwork

# Load Q1 data for traffic patterns
q1_nodes = np.load(f"{DATA_DIR}/nodes_q1.npy")
with open(f"{DATA_DIR}/SAN_BERNARDINO_2023_Q1/desc.json") as f:
    desc = json.load(f)
q1_data = np.array(np.memmap(f"{DATA_DIR}/SAN_BERNARDINO_2023_Q1/data.dat",
                              dtype="float32", mode="r", shape=tuple(desc["shape"])))

# Load adjacency
with open(f"{FULL_DATA_DIR}/adj_mx.pkl", "rb") as f:
    adj_data = pickle.load(f)
if isinstance(adj_data, list):
    adj_full = adj_data[2] if len(adj_data) > 2 else adj_data[0]
else:
    adj_full = adj_data
adj_q1 = adj_full[np.ix_(q1_nodes, q1_nodes)]

# Compute per-node traffic stats
q1_flow = q1_data[:, q1_nodes, 0]  # (time, 558)
node_mean_flow = q1_flow.mean(axis=0)  # (558,)
node_std_flow = q1_flow.std(axis=0)
node_zero_rate = (q1_flow == 0).mean(axis=0)

# Node connectivity (degree from adj)
node_degree = (adj_q1 > 0).sum(axis=1)

print(f"  Q1 nodes: {len(q1_nodes)}")
print(f"  Mean flow: mean={node_mean_flow.mean():.2f}, std={node_mean_flow.std():.2f}")

# For each budget, load EPN and predict NSPs
MAX_NEIGHBORS = 10

def get_node_context(node_idx, all_nsp, adj, max_neighbors=10):
    """Get neighbor NSPs and adj weights for a node."""
    weights = adj[node_idx].copy()
    weights[node_idx] = 0  # exclude self
    top_k = np.argsort(weights)[-max_neighbors:]
    neighbor_nsps = all_nsp[top_k]  # (k, nsp_dim)
    neighbor_weights = weights[top_k]
    # Pad
    pad_n = max_neighbors - len(top_k)
    if pad_n > 0:
        neighbor_nsps = np.pad(neighbor_nsps, ((0, pad_n), (0, 0)))
        neighbor_weights = np.pad(neighbor_weights, (0, pad_n))
    return neighbor_nsps, neighbor_weights


def predict_all_nodes(epn, budget_steps):
    """Run EPN on all Q1 nodes to get predicted NSPs."""
    epn.eval()
    predicted = np.zeros((N, nsp_dim))

    with torch.no_grad():
        for i in range(N):
            # Traffic pattern: last budget_steps of training data
            train_end = int(len(q1_data) * 0.6)
            start = max(0, train_end - budget_steps)
            traffic = q1_data[start:train_end, q1_nodes[i], :]  # (steps, 5)
            traffic_t = torch.tensor(traffic, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            # Graph context (leave-one-out: use actual NSP of neighbors)
            neighbor_nsps, neighbor_weights = get_node_context(i, actual_nsp, adj_q1)
            n_t = torch.tensor(neighbor_nsps, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            w_t = torch.tensor(neighbor_weights, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            pred = epn(traffic_t, n_t, w_t).cpu().numpy()[0]
            predicted[i] = pred

    return predicted


budgets = {"3h": 36, "12h": 144, "1d": 288, "7d": 2016}
epn_results = {}

for budget_name, budget_steps in budgets.items():
    epn_path = os.path.join(EPN_DIR, f"epn_{budget_name}.pt")
    if not os.path.exists(epn_path):
        print(f"  EPN {budget_name} not found, skipping")
        continue

    epn = EmbeddingPredictorNetwork(nsp_dim=nsp_dim, num_channels=5, d_hidden=64).to(DEVICE)
    epn.load_state_dict(torch.load(epn_path, map_location=DEVICE))

    predicted_nsp = predict_all_nodes(epn, budget_steps)

    # Per-node metrics
    per_node_mse = ((predicted_nsp - actual_nsp) ** 2).mean(axis=1)
    per_node_cos = np.array([
        np.dot(predicted_nsp[i], actual_nsp[i]) /
        (norm(predicted_nsp[i]) * norm(actual_nsp[i]) + 1e-8)
        for i in range(N)
    ])
    per_node_l2 = norm(predicted_nsp - actual_nsp, axis=1)
    pred_norms = norm(predicted_nsp, axis=1)

    # Mean init baseline: distance from mean
    mean_init_l2 = dist_from_mean  # already computed
    mean_init_cos = np.array([
        np.dot(mean_nsp, actual_nsp[i]) /
        (norm(mean_nsp) * norm(actual_nsp[i]) + 1e-8)
        for i in range(N)
    ])

    print(f"\n  --- Budget: {budget_name} ({budget_steps} steps) ---")
    print(f"  EPN predicted norm: mean={pred_norms.mean():.4f}, std={pred_norms.std():.4f}")
    print(f"  Actual norm:        mean={nsp_norms.mean():.4f}, std={nsp_norms.std():.4f}")
    print(f"  Norm ratio (pred/actual): mean={np.mean(pred_norms/nsp_norms):.4f}")
    print(f"  ")
    print(f"  Per-node cosine sim (EPN vs actual): mean={per_node_cos.mean():.4f}, "
          f"std={per_node_cos.std():.4f}")
    print(f"  Per-node cosine sim (mean vs actual): mean={mean_init_cos.mean():.4f}, "
          f"std={mean_init_cos.std():.4f}")
    print(f"  ")
    print(f"  Per-node L2 (EPN vs actual):  mean={per_node_l2.mean():.4f}, std={per_node_l2.std():.4f}")
    print(f"  Per-node L2 (mean vs actual): mean={mean_init_l2.mean():.4f}, std={mean_init_l2.std():.4f}")

    # How many nodes does EPN beat mean_init?
    epn_better = (per_node_l2 < mean_init_l2).sum()
    epn_better_cos = (per_node_cos > mean_init_cos).sum()
    print(f"  ")
    print(f"  Nodes where EPN beats mean_init (L2): {epn_better}/{N} ({epn_better/N*100:.1f}%)")
    print(f"  Nodes where EPN beats mean_init (cos): {epn_better_cos}/{N} ({epn_better_cos/N*100:.1f}%)")

    # Correlation with node properties
    from scipy.stats import pearsonr, spearmanr

    # What predicts EPN quality?
    epn_improvement = mean_init_l2 - per_node_l2  # positive = EPN better

    corrs = {}
    for prop_name, prop_val in [
        ("mean_flow", node_mean_flow),
        ("std_flow", node_std_flow),
        ("zero_rate", node_zero_rate),
        ("degree", node_degree.astype(float)),
        ("nsp_norm", nsp_norms),
        ("dist_from_mean", dist_from_mean),
    ]:
        r_pear, p_pear = pearsonr(prop_val, epn_improvement)
        r_spear, p_spear = spearmanr(prop_val, epn_improvement)
        corrs[prop_name] = {"pearson": r_pear, "p_pear": p_pear,
                           "spearman": r_spear, "p_spear": p_spear}

    print(f"\n  Correlation: node property → EPN improvement over mean_init (L2)")
    for prop_name, c in corrs.items():
        sig = "***" if c["p_pear"] < 0.001 else "**" if c["p_pear"] < 0.01 else "*" if c["p_pear"] < 0.05 else ""
        print(f"    {prop_name:20s}: r={c['pearson']:+.3f} (p={c['p_pear']:.4f}) {sig}")

    # Predicted NSP diversity (are they all the same?)
    pred_normed = predicted_nsp / (norm(predicted_nsp, axis=1, keepdims=True) + 1e-8)
    pred_pairwise_cos = (pred_normed @ pred_normed.T)[triu_idx]
    print(f"\n  Predicted NSP pairwise cosine: mean={pred_pairwise_cos.mean():.4f}, "
          f"std={pred_pairwise_cos.std():.4f}")
    print(f"  Actual NSP pairwise cosine:    mean={pairwise_cos.mean():.4f}, "
          f"std={pairwise_cos.std():.4f}")

    epn_results[budget_name] = {
        "per_node_cos": per_node_cos,
        "per_node_l2": per_node_l2,
        "mean_init_cos": mean_init_cos,
        "mean_init_l2": mean_init_l2,
        "pred_norms": pred_norms,
        "predicted_nsp": predicted_nsp,
        "corrs": corrs,
    }

    del epn
    torch.cuda.empty_cache()

# ============================================================
# 4. Adjacency vs NSP similarity
# ============================================================
print("\n" + "=" * 70)
print("4. Does graph adjacency predict NSP similarity?")
print("=" * 70)

adj_flat = adj_q1[triu_idx]
connected = adj_flat > 0
print(f"  Connected pairs: {connected.sum()} / {len(connected)} ({connected.mean()*100:.1f}%)")
print(f"  Cosine sim (connected pairs):     mean={pairwise_cos[connected].mean():.4f}")
print(f"  Cosine sim (unconnected pairs):   mean={pairwise_cos[~connected].mean():.4f}")

r, p = pearsonr(adj_flat[connected], pairwise_cos[connected])
print(f"  Correlation (adj weight vs cosine, connected only): r={r:.4f}, p={p:.4f}")

# ============================================================
# 5. Is NSP even necessary? Mean-init quality analysis
# ============================================================
print("\n" + "=" * 70)
print("5. How good is mean_init? (upper/lower bounds)")
print("=" * 70)

# Best possible: actual NSP (L2=0, cos=1)
# Worst: random NSP
# Mean init: our baseline
print(f"  Mean init cosine sim with actual: mean={mean_init_cos.mean():.4f}, "
      f"std={mean_init_cos.std():.4f}")
print(f"  Mean init L2 from actual: mean={dist_from_mean.mean():.4f}")

# Random baseline
rng = np.random.RandomState(42)
random_nsp = rng.randn(N, nsp_dim).astype(np.float32)
random_nsp = random_nsp / norm(random_nsp, axis=1, keepdims=True) * nsp_norms.mean()
random_cos = np.array([
    np.dot(random_nsp[i], actual_nsp[i]) / (norm(random_nsp[i]) * norm(actual_nsp[i]) + 1e-8)
    for i in range(N)
])
random_l2 = norm(random_nsp - actual_nsp, axis=1)
print(f"  Random init cosine sim with actual: mean={random_cos.mean():.4f}")
print(f"  Random init L2 from actual: mean={random_l2.mean():.4f}")

# NN copy: use nearest neighbor's NSP
nn_cos = np.zeros(N)
nn_l2 = np.zeros(N)
for i in range(N):
    # Find nearest connected neighbor by adj weight
    weights = adj_q1[i].copy()
    weights[i] = 0
    nn_idx = np.argmax(weights)
    nn_cos[i] = np.dot(actual_nsp[nn_idx], actual_nsp[i]) / (
        norm(actual_nsp[nn_idx]) * norm(actual_nsp[i]) + 1e-8)
    nn_l2[i] = norm(actual_nsp[nn_idx] - actual_nsp[i])

print(f"  NN-copy cosine sim with actual: mean={nn_cos.mean():.4f}, std={nn_cos.std():.4f}")
print(f"  NN-copy L2 from actual: mean={nn_l2.mean():.4f}")

# ============================================================
# 6. Node categories: dead/functional
# ============================================================
print("\n" + "=" * 70)
print("6. NSP analysis by sensor category")
print("=" * 70)

categories = {}
categories["dead"] = node_zero_rate > 0.9
categories["major_fail"] = (node_zero_rate > 0.5) & (node_zero_rate <= 0.9)
categories["partial"] = (node_zero_rate > 0.05) & (node_zero_rate <= 0.5)
categories["functional"] = node_zero_rate <= 0.05

for cat_name, mask in categories.items():
    cnt = mask.sum()
    if cnt == 0:
        continue
    print(f"\n  {cat_name} ({cnt} nodes):")
    print(f"    NSP norm: mean={nsp_norms[mask].mean():.4f}, std={nsp_norms[mask].std():.4f}")
    print(f"    Dist from mean: mean={dist_from_mean[mask].mean():.4f}")
    print(f"    Mean init cosine: mean={mean_init_cos[mask].mean():.4f}")
    if "7d" in epn_results:
        er = epn_results["7d"]
        print(f"    EPN cosine (7d): mean={er['per_node_cos'][mask].mean():.4f}")
        epn_better = (er['per_node_l2'][mask] < er['mean_init_l2'][mask]).sum()
        print(f"    EPN beats mean (L2): {epn_better}/{cnt} ({epn_better/cnt*100:.1f}%)")

# ============================================================
# 7. Key insight: what makes NSPs different?
# ============================================================
print("\n" + "=" * 70)
print("7. What makes NSPs different across nodes?")
print("=" * 70)

# Project NSPs onto top PCA components
pca_coords = nsp_centered @ Vt[:10].T  # (N, 10) - top 10 PCA components

print("  Correlation: PCA component → node property")
for pc_i in range(5):
    pc = pca_coords[:, pc_i]
    var_pct = explained_var[pc_i] * 100
    print(f"\n  PC{pc_i+1} ({var_pct:.1f}% variance):")
    for prop_name, prop_val in [
        ("mean_flow", node_mean_flow),
        ("zero_rate", node_zero_rate),
        ("degree", node_degree.astype(float)),
    ]:
        r, p = pearsonr(pc, prop_val)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"    {prop_name:15s}: r={r:+.3f} {sig}")

# ============================================================
# Save results
# ============================================================
print("\n" + "=" * 70)
print("Summary saved to", RESULTS_DIR)
print("=" * 70)

summary = {
    "nsp_shape": {"T": T, "N": N, "D": D, "flat": nsp_dim},
    "nsp_norms": {"mean": float(nsp_norms.mean()), "std": float(nsp_norms.std())},
    "mean_nsp_norm": float(np.linalg.norm(mean_nsp)),
    "dist_from_mean": {"mean": float(dist_from_mean.mean()), "std": float(dist_from_mean.std())},
    "pairwise_cos": {"mean": float(pairwise_cos.mean()), "std": float(pairwise_cos.std())},
    "pca_cumvar": {f"pc{i+1}": float(cumvar[i]) for i in range(min(20, len(cumvar)))},
    "mean_init_cos": {"mean": float(mean_init_cos.mean()), "std": float(mean_init_cos.std())},
    "nn_copy_cos": {"mean": float(nn_cos.mean()), "std": float(nn_cos.std())},
    "random_cos": {"mean": float(random_cos.mean())},
}

for budget_name, er in epn_results.items():
    summary[f"epn_{budget_name}"] = {
        "cos_mean": float(er["per_node_cos"].mean()),
        "cos_std": float(er["per_node_cos"].std()),
        "l2_mean": float(er["per_node_l2"].mean()),
        "pred_norm_mean": float(er["pred_norms"].mean()),
        "pred_norm_std": float(er["pred_norms"].std()),
        "beats_mean_l2_pct": float((er["per_node_l2"] < er["mean_init_l2"]).mean()),
        "beats_mean_cos_pct": float((er["per_node_cos"] > er["mean_init_cos"]).mean()),
        "pred_pairwise_cos_mean": float(
            ((er["predicted_nsp"] / (norm(er["predicted_nsp"], axis=1, keepdims=True) + 1e-8)) @
             (er["predicted_nsp"] / (norm(er["predicted_nsp"], axis=1, keepdims=True) + 1e-8)).T
            )[triu_idx].mean()
        ),
    }

with open(os.path.join(RESULTS_DIR, "epn_deep_analysis.json"), "w") as f:
    json.dump(summary, f, indent=2)

# Save per-node data for later visualization
np.savez(os.path.join(RESULTS_DIR, "per_node_data.npz"),
         actual_nsp=actual_nsp,
         nsp_norms=nsp_norms,
         dist_from_mean=dist_from_mean,
         mean_init_cos=mean_init_cos,
         nn_cos=nn_cos,
         node_mean_flow=node_mean_flow,
         node_zero_rate=node_zero_rate,
         node_degree=node_degree,
         pca_coords=pca_coords,
         explained_var=explained_var,
         **{f"epn_{b}_cos": er["per_node_cos"] for b, er in epn_results.items()},
         **{f"epn_{b}_l2": er["per_node_l2"] for b, er in epn_results.items()},
         **{f"epn_{b}_pred": er["predicted_nsp"] for b, er in epn_results.items()},
)

print("\nDone!")
