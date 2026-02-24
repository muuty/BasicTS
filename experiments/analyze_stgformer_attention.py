"""Analyze attention collapse in STGformer.

Two analysis targets:
1. Adaptive graph weights: softmax(relu(emb @ emb.T)) — data-independent, extract from model weights
2. FastAttention scores: switch to normal attention mode to get explicit N×N matrices

Compare with STAEformer findings: dead nodes receive 3.5-12x more spatial attention.
"""
import os, sys, glob
sys.path.insert(0, '/data/pretrainingbasicts')

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

# ---- Model setup ----
from baselines.STGformer.arch import STGformer
from basicts.utils import load_adj

device = torch.device('cuda:1')

# Load adjacency matrix
adj_mx, _ = load_adj("datasets/SAN_BERNARDINO/adj_mx.pkl", "normlap")
supports = [torch.Tensor(i) for i in adj_mx]

MODEL_PARAM = {
    "num_nodes": 893, "in_steps": 12, "out_steps": 12,
    "steps_per_day": 288, "input_dim": 5, "output_dim": 1,
    "input_embedding_dim": 24, "tod_embedding_dim": 24, "dow_embedding_dim": 24,
    "spatial_embedding_dim": 0, "adaptive_embedding_dim": 80,
    "supports": supports,
    "num_heads": 4, "num_layers": 3,
    "dropout": 0.1, "use_mixed_proj": True,
}

# ---- Find checkpoint ----
ckpt_dir = glob.glob('checkpoints/STGformer_baseline/SAN_BERNARDINO_30_12_12/*/')[0]
ckpt_path = os.path.join(ckpt_dir, 'STGformer_best_val_MAE.pt')
print(f"Loading checkpoint: {ckpt_path}")

model = STGformer(**MODEL_PARAM).to(device)
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print("Model loaded successfully")

# ---- Node categories ----
data = np.memmap('datasets/xtraffic/SAN_BERNARDINO/data.dat', dtype='float32', mode='r',
                 shape=(105120, 893, 5))
data_3mo = np.array(data[:26280])

# 3-channel detection: flow=0 AND occ=0 AND speed=0
all_zero = (data_3mo[:, :, 0] == 0) & (data_3mo[:, :, 1] == 0) & (data_3mo[:, :, 2] == 0)
zero_rates = all_zero.mean(axis=0)

dead_nodes = np.where(zero_rates > 0.9)[0]
major_fail = np.where((zero_rates > 0.5) & (zero_rates <= 0.9))[0]
partial_fail = np.where((zero_rates > 0.05) & (zero_rates <= 0.5))[0]
functional = np.where(zero_rates <= 0.05)[0]
print(f"Dead: {len(dead_nodes)}, Major fail: {len(major_fail)}, Partial: {len(partial_fail)}, Functional: {len(functional)}")

# ============================================================
# PART 1: Adaptive Graph Analysis (data-independent)
# ============================================================
print("\n" + "="*70)
print("PART 1: ADAPTIVE GRAPH WEIGHTS ANALYSIS")
print("="*70)

# Extract adaptive embeddings from both encoder and spatial modules
for module_name, module in [("Encoder", model.encoder), ("Spatial", model.spatial)]:
    emb = module.adaptive_embedding.data.cpu()  # (T, N, D)
    T, N, D = emb.shape
    print(f"\n--- {module_name} adaptive_embedding: shape=({T}, {N}, {D}) ---")

    # Compute adaptive graph: softmax(relu(emb @ emb.T))
    graph = torch.matmul(emb, emb.transpose(1, 2))  # (T, N, N)
    graph = F.softmax(F.relu(graph), dim=-1)  # (T, N, N)

    # Average over timesteps
    graph_avg = graph.mean(dim=0).numpy()  # (N, N)

    # graph[i, j] = weight from node i to node j (row-normalized by softmax)
    # Attention RECEIVED by node j = sum over i of graph[i, j] = column sum
    received = graph_avg.sum(axis=0)  # (N,)

    # Uniform expected: each node receives N * (1/N) = 1.0
    print(f"  Graph weight stats: min={graph_avg.min():.6f}, max={graph_avg.max():.6f}")
    print(f"  Received weight stats: min={received.min():.4f}, max={received.max():.4f}, mean={received.mean():.4f}")

    print(f"\n  --- Graph Weight RECEIVED by Category ---")
    print(f"  (Uniform expected = 1.0)")
    for cat_name, cat_nodes in [("Dead", dead_nodes), ("Major fail", major_fail),
                                 ("Partial fail", partial_fail), ("Functional", functional)]:
        if len(cat_nodes) == 0:
            continue
        recv = received[cat_nodes]
        print(f"  {cat_name:15s} ({len(cat_nodes):3d} nodes): mean={recv.mean():.4f}, "
              f"std={recv.std():.4f}, ratio_vs_uniform={recv.mean()/1.0:.3f}x")

    # Self-weight (diagonal)
    self_weight = np.diag(graph_avg)  # (N,)
    uniform_self = 1.0 / N
    print(f"\n  --- Self-Weight (diagonal) ---")
    print(f"  (Uniform expected = {uniform_self:.6f})")
    for cat_name, cat_nodes in [("Dead", dead_nodes), ("Major fail", major_fail),
                                 ("Partial fail", partial_fail), ("Functional", functional)]:
        if len(cat_nodes) == 0:
            continue
        sw = self_weight[cat_nodes]
        print(f"  {cat_name:15s} ({len(cat_nodes):3d} nodes): mean={sw.mean():.6f}, "
              f"ratio={sw.mean()/uniform_self:.2f}x")

    # Cross-category flow: from functional TO each category
    print(f"\n  --- Cross-Category Graph Flow (from Functional nodes) ---")
    for target_name, target_nodes in [("Dead", dead_nodes), ("Major fail", major_fail),
                                       ("Partial fail", partial_fail), ("Functional", functional)]:
        if len(target_nodes) == 0:
            continue
        # graph_avg[func_nodes, target_nodes] -> mean weight
        flow = graph_avg[np.ix_(functional, target_nodes)].mean()
        expected = len(target_nodes) / N  # expected fraction under uniform
        actual_frac = flow * len(target_nodes)  # total fraction going to this category
        expected_frac = len(target_nodes) / N
        ratio = flow / (1.0 / N)
        print(f"  Functional → {target_name:15s}: per-node weight={flow:.6f}, "
              f"vs uniform={1.0/N:.6f}, ratio={ratio:.3f}x")

    # Per-timestep analysis: does collapse vary across timesteps?
    print(f"\n  --- Per-Timestep Collapse Ratio (Dead recv / Functional recv) ---")
    for t in range(T):
        g_t = graph[t].numpy()  # (N, N)
        recv_t = g_t.sum(axis=0)
        dead_recv_t = recv_t[dead_nodes].mean()
        func_recv_t = recv_t[functional].mean()
        ratio_t = dead_recv_t / func_recv_t if func_recv_t > 0 else float('inf')
        print(f"  t={t:2d}: dead={dead_recv_t:.4f}, func={func_recv_t:.4f}, ratio={ratio_t:.3f}x")

# ============================================================
# PART 2: Linear (Fast) Attention - Effective Weight Analysis
# ============================================================
print("\n" + "="*70)
print("PART 2: LINEAR ATTENTION (ORIGINAL MODE) EFFECTIVE WEIGHT ANALYSIS")
print("="*70)
print("""
Linear attention formula:
  w_ij = φ(q_i)·φ(k_j) / normalizer_i       (i ≠ j)
  w_ii = (φ(q_i)·φ(k_i) + N) / normalizer_i  (self-connection)
  normalizer_i = Σ_l φ(q_i)·φ(k_l) + N
  φ = L2 normalization

Received by node j = Σ_i w_ij
  = φ(k_j) · [Σ_i φ(q_i)/normalizer_i] + N/normalizer_j
""")

# Keep fast mode (self.fast = 1) — original linear attention
# Hook into fast_attention to capture normalized qs, ks
linear_attn_data = []

def patched_fast_attention(self, x, qs, ks, vs, dim):
    """Patched fast_attention: same computation, but captures qs/ks for analysis."""
    qs_norm = torch.nn.functional.normalize(qs, dim=-1)
    ks_norm = torch.nn.functional.normalize(ks, dim=-1)
    N = qs_norm.shape[1]
    b, l = x.shape[dim : dim + 2]

    # Capture for spatial attention only (N=893)
    if qs_norm.shape[1] == 893:
        linear_attn_data.append({
            'qs': qs_norm.detach().cpu(),  # (B*T, N, num_heads, head_dim)
            'ks': ks_norm.detach().cpu(),
            'N': N,
        })

    # Original computation (unchanged)
    kvs = torch.einsum("blhm,blhd->bhmd", ks_norm, vs)
    attention_num = torch.einsum("bnhm,bhmd->bnhd", qs_norm, kvs)
    attention_num += N * vs

    all_ones = torch.ones([ks_norm.shape[1]], device=ks_norm.device)
    ks_sum = torch.einsum("blhm,l->bhm", ks_norm, all_ones)
    attention_normalizer = torch.einsum("bnhm,bhm->bnh", qs_norm, ks_sum)
    attention_normalizer = torch.unsqueeze(
        attention_normalizer, len(attention_normalizer.shape)
    )
    attention_normalizer += torch.ones_like(attention_normalizer) * N
    out = attention_num / attention_normalizer
    out = torch.unflatten(out, dim, (b, l)).flatten(start_dim=3)
    return out

# Monkey-patch fast_attention on all FastAttentionLayer instances
for layer in model.spatial.attn_layers:
    for attn_module in layer.attn:
        attn_module.fast_attention = lambda x, qs, ks, vs, dim, _self=attn_module: \
            patched_fast_attention(_self, x, qs, ks, vs, dim)

print("Linear attention hooks installed (original fast mode)")

# ---- Load scaler and prepare test data ----
from basicts.scaler import ZScoreScaler

scaler = ZScoreScaler(
    dataset_name='xtraffic/SAN_BERNARDINO',
    train_ratio=0.6,
    norm_each_channel=False,
    rescale=True,
)

test_start = int(26280 * 0.8)
INPUT_LEN = 12
OUTPUT_LEN = 12

num_samples = 100
np.random.seed(42)
sample_starts = np.random.randint(test_start, 26280 - INPUT_LEN - OUTPUT_LEN, size=num_samples)

batch_size = 8
all_received = []  # per-batch effective attention received

print(f"Running {num_samples} test samples...")

for batch_idx in range(0, num_samples, batch_size):
    batch_starts = sample_starts[batch_idx:batch_idx + batch_size]
    B = len(batch_starts)

    history = np.zeros((B, INPUT_LEN, 893, 5), dtype=np.float32)
    for i, s in enumerate(batch_starts):
        history[i] = data_3mo[s:s+INPUT_LEN, :, :]

    h_tensor = torch.tensor(history, dtype=torch.float32).to(device)
    h_tensor[..., 0:1] = scaler.transform(h_tensor[..., 0:1])

    future = torch.zeros(B, OUTPUT_LEN, 893, 5, dtype=torch.float32, device=device)

    linear_attn_data.clear()

    with torch.no_grad():
        _ = model(h_tensor, future, batch_seen=0, epoch=0, train=False)

    # Compute effective attention received from captured qs, ks
    for entry in linear_attn_data:
        qs = entry['qs']  # (B*T, N, num_heads, head_dim)
        ks = entry['ks']  # (B*T, N, num_heads, head_dim)
        N = entry['N']    # 893

        # For each head separately, compute received weight per node
        # qs shape: (BT, N, H, D), ks shape: (BT, N, H, D)
        BT, NN, H, D = qs.shape

        # Compute normalizer_i = Σ_l φ(q_i)·φ(k_l) + N
        # For each (bt, h): qs[bt, :, h, :] @ ks[bt, :, h, :].T → (N, N), sum over columns → (N,)
        # Efficient: normalizer_i = q_i · ks_sum + N
        ks_sum = ks.sum(dim=1)  # (BT, H, D) — sum over all key nodes
        # q_i · ks_sum: for each node i
        # qs: (BT, N, H, D), ks_sum: (BT, H, D) → need (BT, N, H)
        normalizer = torch.einsum('bnhd,bhd->bnh', qs, ks_sum) + N  # (BT, N, H)

        # Effective received by node j:
        # received_j = φ(k_j) · [Σ_i φ(q_i)/normalizer_i] + N/normalizer_j
        # Step 1: weighted_q = Σ_i φ(q_i)/normalizer_i  → shape (BT, H, D)
        # normalizer: (BT, N, H) → expand to (BT, N, H, 1)
        weighted_q = (qs / normalizer.unsqueeze(-1)).sum(dim=1)  # (BT, H, D)

        # Step 2: k_j · weighted_q → (BT, N, H)
        cross_recv = torch.einsum('bnhd,bhd->bnh', ks, weighted_q)  # (BT, N, H)

        # Step 3: self-connection: N / normalizer_j → (BT, N, H)
        self_recv = N / normalizer  # (BT, N, H)

        # Total received (per head, averaged over BT)
        total_recv = cross_recv + self_recv  # (BT, N, H)
        # Average over batch*time and heads → (N,)
        recv_avg = total_recv.mean(dim=(0, 2)).numpy()  # (N,)
        all_received.append(recv_avg)

if len(all_received) > 0:
    all_received = np.stack(all_received)  # (num_captures, 893)
    avg_recv = all_received.mean(axis=0)  # (893,)

    print(f"\nCaptured {len(all_received)} spatial linear attention measurements")
    print(f"\n  --- Linear Attention Effective Weight RECEIVED by Category ---")
    print(f"  (Uniform expected = 1.0)")
    for cat_name, cat_nodes in [("Dead", dead_nodes), ("Major fail", major_fail),
                                 ("Partial fail", partial_fail), ("Functional", functional)]:
        if len(cat_nodes) == 0:
            continue
        recv = avg_recv[cat_nodes]
        print(f"  {cat_name:15s} ({len(cat_nodes):3d} nodes): mean={recv.mean():.4f}, "
              f"std={recv.std():.4f}, ratio={recv.mean()/1.0:.3f}x")

    dead_recv = avg_recv[dead_nodes].mean()
    func_recv = avg_recv[functional].mean()
    print(f"\n  Dead/Functional ratio: {dead_recv/func_recv:.3f}x")

    # Also compute per-head breakdown
    print(f"\n  --- Per-Head Breakdown (last batch) ---")
    if linear_attn_data:
        entry = linear_attn_data[-1] if linear_attn_data else None
    # Use the last captured data for per-head analysis
    if len(all_received) > 0:
        # Recompute for last batch with per-head detail
        last_entry = None
        # Run one more batch for detailed analysis
        batch_starts = sample_starts[:8]
        B = len(batch_starts)
        history = np.zeros((B, INPUT_LEN, 893, 5), dtype=np.float32)
        for i, s in enumerate(batch_starts):
            history[i] = data_3mo[s:s+INPUT_LEN, :, :]
        h_tensor = torch.tensor(history, dtype=torch.float32).to(device)
        h_tensor[..., 0:1] = scaler.transform(h_tensor[..., 0:1])
        future = torch.zeros(B, OUTPUT_LEN, 893, 5, dtype=torch.float32, device=device)
        linear_attn_data.clear()
        with torch.no_grad():
            _ = model(h_tensor, future, batch_seen=0, epoch=0, train=False)

        if linear_attn_data:
            entry = linear_attn_data[0]
            qs = entry['qs']
            ks = entry['ks']
            N = entry['N']
            BT, NN, H, D = qs.shape
            ks_sum = ks.sum(dim=1)
            normalizer = torch.einsum('bnhd,bhd->bnh', qs, ks_sum) + N
            weighted_q = (qs / normalizer.unsqueeze(-1)).sum(dim=1)
            cross_recv = torch.einsum('bnhd,bhd->bnh', ks, weighted_q)
            self_recv = N / normalizer
            total_recv = (cross_recv + self_recv).mean(dim=0).numpy()  # (N, H)

            for h in range(H):
                dead_h = total_recv[dead_nodes, h].mean()
                func_h = total_recv[functional, h].mean()
                print(f"  Head {h}: dead={dead_h:.4f}, func={func_h:.4f}, ratio={dead_h/func_h:.3f}x")
else:
    print("\nNo linear attention data captured. Check hooks.")

# ============================================================
# PART 3: Correlation Analysis (using linear attention results)
# ============================================================
print("\n" + "="*70)
print("PART 3: CORRELATION WITH ZERO RATE (Linear Attention)")
print("="*70)

from scipy.stats import pearsonr, spearmanr

if len(all_received) > 0:
    received = avg_recv  # from Part 2 linear attention

    r_pearson, p_pearson = pearsonr(zero_rates, received)
    r_spearman, p_spearman = spearmanr(zero_rates, received)

    print(f"Zero rate vs Linear attention weight received:")
    print(f"  Pearson r={r_pearson:.4f} (p={p_pearson:.2e})")
    print(f"  Spearman r={r_spearman:.4f} (p={p_spearman:.2e})")

    # Top/bottom 20 nodes by received weight
    top20 = np.argsort(received)[-20:][::-1]
    bot20 = np.argsort(received)[:20]

    print(f"\nTop 20 nodes by linear attn weight received:")
    print(f"{'Node':>6s} {'ZeroRate':>10s} {'Received':>10s} {'Category':>15s}")
    for n in top20:
        cat = "Dead" if zero_rates[n] > 0.9 else "Major" if zero_rates[n] > 0.5 else \
              "Partial" if zero_rates[n] > 0.05 else "Functional"
        print(f"{n:6d} {zero_rates[n]:10.3f} {received[n]:10.4f} {cat:>15s}")

    print(f"\nBottom 20 nodes by linear attn weight received:")
    for n in bot20:
        cat = "Dead" if zero_rates[n] > 0.9 else "Major" if zero_rates[n] > 0.5 else \
              "Partial" if zero_rates[n] > 0.05 else "Functional"
        print(f"{n:6d} {zero_rates[n]:10.3f} {received[n]:10.4f} {cat:>15s}")
else:
    print("No linear attention data available for correlation.")

# ============================================================
# Summary comparison with STAEformer
# ============================================================
print("\n" + "="*70)
print("SUMMARY: STGformer vs STAEformer Attention Collapse")
print("="*70)
print("""
STAEformer findings (from docs/attention_collapse_missing_values.md):
  - Dead nodes receive 3.5-12x more spatial attention than functional
  - Cause: softmax on consistent (zero) keys → peaked attention

STGformer has TWO mechanisms:
  1. Adaptive graph: softmax(relu(emb @ emb.T)) — learned, data-independent
  2. FastAttention: linear attention (Q·K^T replaced with kernel trick)

Key question: Does the adaptive graph (which IS explicit softmax)
show the same collapse pattern?
""")

# Print final ratios
# 1. Adaptive graph
emb = model.spatial.adaptive_embedding.data.cpu()
graph = torch.matmul(emb, emb.transpose(1, 2))
graph = F.softmax(F.relu(graph), dim=-1)
graph_avg = graph.mean(dim=0).numpy()
graph_received = graph_avg.sum(axis=0)

dead_graph = graph_received[dead_nodes].mean()
func_graph = graph_received[functional].mean()
print(f"1. Adaptive Graph Collapse Ratio (Dead/Functional): {dead_graph/func_graph:.3f}x")
print(f"   Dead: {dead_graph:.4f}, Functional: {func_graph:.4f}")

# 2. Linear attention
if len(all_received) > 0:
    dead_linear = avg_recv[dead_nodes].mean()
    func_linear = avg_recv[functional].mean()
    print(f"2. Linear Attention Collapse Ratio (Dead/Functional): {dead_linear/func_linear:.3f}x")
    print(f"   Dead: {dead_linear:.4f}, Functional: {func_linear:.4f}")

print(f"""
3. Reference - STAEformer (from prior analysis):
   Softmax Attention Collapse Ratio (Dead/Functional): 3.5-12x
""")
print("Done!")
