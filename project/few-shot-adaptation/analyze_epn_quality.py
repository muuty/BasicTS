"""Analyze EPN prediction quality and embedding distances."""
import torch
import numpy as np
import json, pickle, sys, os
from torch.nn.functional import cosine_similarity

sys.path.append(os.path.dirname(__file__))
from epn_model import EmbeddingPredictorNetwork

# 1. Source embeddings
ckpt = torch.load("checkpoints/Expanding_Source/SAN_BERNARDINO_2023_Q1_30_12_12/df169dea7f0928b8a58b0e3aa9f91031/STAEformer_best_val_MAE.pt",
                   map_location="cpu", weights_only=False)
source_emb = ckpt["model_state_dict"]["encoder.adaptive_embedding"]
T, N_old, D = source_emb.shape
NSP_DIM = T * D
emb_flat = source_emb.permute(1, 0, 2).reshape(N_old, -1)
norms = emb_flat.norm(dim=1)
mean_emb_flat = emb_flat.mean(dim=0)
existing_var = emb_flat.var(dim=0).mean().item()

cos_mat = cosine_similarity(emb_flat.unsqueeze(1), emb_flat.unsqueeze(0), dim=2)
mask_tri = torch.triu(torch.ones(N_old, N_old), diagonal=1).bool()
existing_pairwise = cos_mat[mask_tri]

print(f"Source: {source_emb.shape}, norm={norms.mean():.4f}±{norms.std():.4f}")
print(f"Existing pairwise cos: {existing_pairwise.mean():.4f}±{existing_pairwise.std():.4f}")
print(f"Existing per-dim var: {existing_var:.6f}")

# 2. Q2 data
q1_nodes = np.load("datasets/expanding_experiment/nodes_q1.npy")
q2_nodes = np.load("datasets/expanding_experiment/nodes_q2.npy")
new_in_q2 = np.load("datasets/expanding_experiment/new_in_q2.npy")

with open("datasets/expanding_experiment/SAN_BERNARDINO_2023_Q2/desc.json") as f:
    desc = json.load(f)
raw_mm = np.memmap("datasets/expanding_experiment/SAN_BERNARDINO_2023_Q2/data.dat",
                   dtype="float32", mode="r", shape=tuple(desc["shape"]))
raw_data = np.array(raw_mm)

with open("datasets/xtraffic/SAN_BERNARDINO/adj_mx.pkl", "rb") as f:
    adj_full = pickle.load(f)

MAX_NEIGHBORS = 10
train_end = int(len(raw_data) * 0.6)
budgets = {"3h": 36, "12h": 144, "1d": 288, "7d": 2016}
print(f"Q2: {len(new_in_q2)} new nodes, raw={raw_data.shape}")


def pw_cos(X):
    c = cosine_similarity(X.unsqueeze(1), X.unsqueeze(0), dim=2)
    m = torch.triu(torch.ones(len(X), len(X)), diagonal=1).bool()
    return c[m]


# 3. Analyze per budget
print("\n" + "=" * 75)
print("EPN PREDICTION QUALITY")
print("=" * 75)

for bname, bsteps in budgets.items():
    epn = EmbeddingPredictorNetwork(nsp_dim=NSP_DIM, num_channels=5, d_hidden=64)
    epn.load_state_dict(torch.load(f"project/few-shot-adaptation/results/epn_{bname}.pt", map_location="cpu"))
    epn.eval()

    epn_preds, graph_inits = [], []
    n_neighbors_list = []

    for nid in new_in_q2:
        nid = int(nid)
        node_traffic = raw_data[:train_end, nid, :]
        actual = min(bsteps, len(node_traffic))
        traffic = torch.tensor(node_traffic[:actual].copy(), dtype=torch.float32).unsqueeze(0)

        weights, neighbor_nsps = [], []
        for i, old_nid in enumerate(q1_nodes):
            w = adj_full[nid, int(old_nid)]
            if w > 0:
                weights.append(float(w))
                neighbor_nsps.append(emb_flat[i])
        n_neighbors_list.append(len(weights))

        if len(weights) > MAX_NEIGHBORS:
            topk = np.argsort(weights)[-MAX_NEIGHBORS:]
            weights = [weights[j] for j in topk]
            neighbor_nsps = [neighbor_nsps[j] for j in topk]

        if neighbor_nsps:
            w_t = torch.tensor(weights)
            w_t = w_t / w_t.sum()
            graph_init = (torch.stack(neighbor_nsps) * w_t.unsqueeze(1)).sum(0)
        else:
            graph_init = mean_emb_flat.clone()
        graph_inits.append(graph_init)

        pad_len = MAX_NEIGHBORS - len(weights)
        n_nsps = torch.stack(neighbor_nsps) if neighbor_nsps else torch.zeros(0, NSP_DIM)
        if pad_len > 0:
            n_nsps = torch.cat([n_nsps, torch.zeros(pad_len, NSP_DIM)])
        adj_w = torch.tensor(weights + [0.0] * pad_len)

        with torch.no_grad():
            pred = epn(traffic, n_nsps.unsqueeze(0), adj_w.unsqueeze(0)).squeeze(0)
        epn_preds.append(pred)

    epn_preds = torch.stack(epn_preds)
    graph_inits = torch.stack(graph_inits)
    mean_inits = mean_emb_flat.unsqueeze(0).expand(len(new_in_q2), -1)

    epn_var = epn_preds.var(dim=0).mean().item()
    graph_var = graph_inits.var(dim=0).mean().item()
    epn_norms = epn_preds.norm(dim=1)
    graph_norms = graph_inits.norm(dim=1)
    epn_pw = pw_cos(epn_preds)
    graph_pw = pw_cos(graph_inits)
    cos_em = cosine_similarity(epn_preds, mean_inits, dim=1)
    cos_eg = cosine_similarity(epn_preds, graph_inits, dim=1)

    no_nb = sum(1 for n in n_neighbors_list if n == 0)

    print(f"\n--- {bname} ---")
    print(f"  Variance ratio:  EPN={epn_var/existing_var:.1%}  graph={graph_var/existing_var:.1%}")
    print(f"  Norm:            EPN={epn_norms.mean():.3f}±{epn_norms.std():.3f}  graph={graph_norms.mean():.3f}±{graph_norms.std():.3f}  existing={norms.mean():.3f}±{norms.std():.3f}")
    print(f"  Pairwise cos:    EPN={epn_pw.mean():.4f}  graph={graph_pw.mean():.4f}  existing={existing_pairwise.mean():.4f}")
    print(f"  Cos EPN<->mean={cos_em.mean():.4f}  EPN<->graph={cos_eg.mean():.4f}")
    print(f"  Neighbors: avg={np.mean(n_neighbors_list):.1f}, w/o={no_nb}/{len(new_in_q2)}")

    # L2 distance between init methods
    l2_em = (epn_preds - mean_inits).norm(dim=1)
    l2_eg = (epn_preds - graph_inits).norm(dim=1)
    l2_gm = (graph_inits - mean_inits).norm(dim=1)
    print(f"  L2: EPN<->mean={l2_em.mean():.4f}  EPN<->graph={l2_eg.mean():.4f}  graph<->mean={l2_gm.mean():.4f}")
