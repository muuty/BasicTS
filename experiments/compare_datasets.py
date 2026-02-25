"""Compare CONTRA_COSTA vs SAN_BERNARDINO dataset structural differences."""
import numpy as np
import pickle

# Load adjacency matrices
with open('datasets/CONTRA_COSTA/adj_mx.pkl', 'rb') as f:
    cc_adj_data = pickle.load(f, encoding='latin1')
with open('datasets/xtraffic/SAN_BERNARDINO/adj_mx.pkl', 'rb') as f:
    sb_adj_data = pickle.load(f, encoding='latin1')

# Extract adj matrices
if isinstance(cc_adj_data, list):
    cc_adj = cc_adj_data[-1] if len(cc_adj_data) > 1 else cc_adj_data[0]
else:
    cc_adj = cc_adj_data
if isinstance(sb_adj_data, list):
    sb_adj = sb_adj_data[-1] if len(sb_adj_data) > 1 else sb_adj_data[0]
else:
    sb_adj = sb_adj_data

if hasattr(cc_adj, 'toarray'):
    cc_adj = cc_adj.toarray()
if hasattr(sb_adj, 'toarray'):
    sb_adj = sb_adj.toarray()

cc_adj = np.array(cc_adj, dtype=float)
sb_adj = np.array(sb_adj, dtype=float)

print("=" * 60)
print("PART 1: ADJACENCY MATRIX COMPARISON")
print("=" * 60)
print(f"CC shape: {cc_adj.shape}, SB shape: {sb_adj.shape}")

# Density
cc_no_diag = cc_adj.copy(); np.fill_diagonal(cc_no_diag, 0)
sb_no_diag = sb_adj.copy(); np.fill_diagonal(sb_no_diag, 0)

cc_edges = (cc_no_diag > 0.01).sum()
sb_edges = (sb_no_diag > 0.01).sum()
cc_n, sb_n = cc_adj.shape[0], sb_adj.shape[0]
print(f"\nCC edges (>0.01): {cc_edges}, density: {cc_edges / (cc_n*(cc_n-1)):.6f}")
print(f"SB edges (>0.01): {sb_edges}, density: {sb_edges / (sb_n*(sb_n-1)):.6f}")

# Degree distribution
cc_degree = (cc_no_diag > 0.01).sum(axis=1)
sb_degree = (sb_no_diag > 0.01).sum(axis=1)
print(f"\nCC degree: mean={cc_degree.mean():.1f}, median={np.median(cc_degree):.0f}, max={cc_degree.max()}, min={cc_degree.min()}, std={cc_degree.std():.1f}")
print(f"SB degree: mean={sb_degree.mean():.1f}, median={np.median(sb_degree):.0f}, max={sb_degree.max()}, min={sb_degree.min()}, std={sb_degree.std():.1f}")

# Edge weight distribution
cc_weights = cc_no_diag[cc_no_diag > 0.01]
sb_weights = sb_no_diag[sb_no_diag > 0.01]
print(f"\nCC edge weights: mean={cc_weights.mean():.4f}, median={np.median(cc_weights):.4f}, std={cc_weights.std():.4f}")
print(f"SB edge weights: mean={sb_weights.mean():.4f}, median={np.median(sb_weights):.4f}, std={sb_weights.std():.4f}")

# Strong connections
for thresh in [0.3, 0.5, 0.7]:
    cc_s = (cc_no_diag > thresh).sum()
    sb_s = (sb_no_diag > thresh).sum()
    print(f"Edges > {thresh}: CC={cc_s} ({cc_s/max(cc_edges,1)*100:.1f}%), SB={sb_s} ({sb_s/max(sb_edges,1)*100:.1f}%)")

# Isolated nodes
cc_isolated = (cc_degree == 0).sum()
sb_isolated = (sb_degree == 0).sum()
print(f"\nCC isolated nodes (degree=0): {cc_isolated} ({cc_isolated/cc_n*100:.1f}%)")
print(f"SB isolated nodes (degree=0): {sb_isolated} ({sb_isolated/sb_n*100:.1f}%)")

# ========== PART 2: DATA COMPARISON ==========
print("\n" + "=" * 60)
print("PART 2: DATA COMPARISON")
print("=" * 60)

cc_data = np.memmap('datasets/CONTRA_COSTA/data.dat', dtype='float32', mode='r').reshape(-1, cc_n, 5)
sb_data = np.memmap('datasets/xtraffic/SAN_BERNARDINO/data.dat', dtype='float32', mode='r').reshape(-1, sb_n, 5)

print(f"CC data shape: {cc_data.shape}")
print(f"SB data shape: {sb_data.shape}")

# Use first 26280 timesteps (training period, 3 months)
cc_train = cc_data[:26280]
sb_train = sb_data[:26280]

# Flow statistics
cc_flow = cc_train[:, :, 0]
sb_flow = sb_train[:, :, 0]
print(f"\nFlow (ch0):")
print(f"  CC: mean={cc_flow.mean():.2f}, std={cc_flow.std():.2f}, max={cc_flow.max():.2f}")
print(f"  SB: mean={sb_flow.mean():.2f}, std={sb_flow.std():.2f}, max={sb_flow.max():.2f}")

# Per-node mean flow
cc_node_mean_flow = cc_flow.mean(axis=0)
sb_node_mean_flow = sb_flow.mean(axis=0)
print(f"\n  CC per-node mean flow: mean={cc_node_mean_flow.mean():.2f}, median={np.median(cc_node_mean_flow):.2f}, std={cc_node_mean_flow.std():.2f}")
print(f"  SB per-node mean flow: mean={sb_node_mean_flow.mean():.2f}, median={np.median(sb_node_mean_flow):.2f}, std={sb_node_mean_flow.std():.2f}")

# ========== PART 3: SENSOR HEALTH ==========
print("\n" + "=" * 60)
print("PART 3: SENSOR HEALTH COMPARISON")
print("=" * 60)

# 3-channel zero detection
cc_zero = (cc_train[:, :, 0] == 0) & (cc_train[:, :, 1] == 0) & (cc_train[:, :, 2] == 0)
sb_zero = (sb_train[:, :, 0] == 0) & (sb_train[:, :, 1] == 0) & (sb_train[:, :, 2] == 0)

cc_zero_rate = cc_zero.mean(axis=0)
sb_zero_rate = sb_zero.mean(axis=0)

for name, rates, n in [("CC", cc_zero_rate, cc_n), ("SB", sb_zero_rate, sb_n)]:
    dead = (rates > 0.9).sum()
    major = ((rates > 0.5) & (rates <= 0.9)).sum()
    partial = ((rates > 0.05) & (rates <= 0.5)).sum()
    func = (rates <= 0.05).sum()
    print(f"\n{name} ({n} nodes):")
    print(f"  Dead (>90%): {dead} ({dead/n*100:.1f}%)")
    print(f"  Major fail (50-90%): {major} ({major/n*100:.1f}%)")
    print(f"  Partial (5-50%): {partial} ({partial/n*100:.1f}%)")
    print(f"  Functional (<5%): {func} ({func/n*100:.1f}%)")

# ========== PART 4: SPATIAL CORRELATION ==========
print("\n" + "=" * 60)
print("PART 4: SPATIAL CORRELATION (KEY FOR SPILLOVER)")
print("=" * 60)

# For connected node pairs, compute flow correlation
def compute_neighbor_correlation(data_flow, adj_no_diag, threshold=0.01):
    """Compute average flow correlation between connected nodes."""
    n = adj_no_diag.shape[0]
    edges = np.argwhere(adj_no_diag > threshold)

    if len(edges) == 0:
        return 0, 0, []

    # Sample edges for efficiency
    if len(edges) > 5000:
        idx = np.random.choice(len(edges), 5000, replace=False)
        edges = edges[idx]

    corrs = []
    for i, j in edges:
        f_i = data_flow[:, i]
        f_j = data_flow[:, j]
        # Skip if either is dead
        if f_i.std() < 0.01 or f_j.std() < 0.01:
            continue
        c = np.corrcoef(f_i, f_j)[0, 1]
        if not np.isnan(c):
            corrs.append(c)

    corrs = np.array(corrs)
    return corrs.mean(), np.median(corrs), corrs

np.random.seed(42)
cc_corr_mean, cc_corr_med, cc_corrs = compute_neighbor_correlation(cc_flow, cc_no_diag)
sb_corr_mean, sb_corr_med, sb_corrs = compute_neighbor_correlation(sb_flow, sb_no_diag)

print(f"\nNeighbor flow correlation (connected pairs, excl dead):")
print(f"  CC: mean={cc_corr_mean:.4f}, median={cc_corr_med:.4f}, n_pairs={len(cc_corrs)}")
print(f"  SB: mean={sb_corr_mean:.4f}, median={sb_corr_med:.4f}, n_pairs={len(sb_corrs)}")

# Correlation distribution
if len(cc_corrs) > 0 and len(sb_corrs) > 0:
    for thresh in [0.3, 0.5, 0.7, 0.9]:
        cc_pct = (cc_corrs > thresh).mean() * 100
        sb_pct = (sb_corrs > thresh).mean() * 100
        print(f"  Corr > {thresh}: CC={cc_pct:.1f}%, SB={sb_pct:.1f}%")

# ========== PART 5: WEIGHTED SPATIAL COUPLING ==========
print("\n" + "=" * 60)
print("PART 5: WEIGHTED SPATIAL COUPLING")
print("=" * 60)

# How much does each node's prediction depend on neighbors?
# Measure: for each node, weighted avg of neighbor flow vs own flow correlation
def compute_weighted_coupling(data_flow, adj_no_diag, threshold=0.01):
    """For each node, compute flow correlation with its weighted neighbor average."""
    n = adj_no_diag.shape[0]
    couplings = []

    for i in range(n):
        weights = adj_no_diag[i, :]
        neighbors = np.where(weights > threshold)[0]
        if len(neighbors) == 0:
            continue

        f_i = data_flow[:, i]
        if f_i.std() < 0.01:
            continue

        # Weighted average of neighbor flows
        w = weights[neighbors]
        w = w / w.sum()
        neighbor_avg = (data_flow[:, neighbors] * w[np.newaxis, :]).sum(axis=1)

        if neighbor_avg.std() < 0.01:
            continue

        c = np.corrcoef(f_i, neighbor_avg)[0, 1]
        if not np.isnan(c):
            couplings.append(c)

    return np.array(couplings)

cc_coupling = compute_weighted_coupling(cc_flow, cc_no_diag)
sb_coupling = compute_weighted_coupling(sb_flow, sb_no_diag)

print(f"\nNode-to-weighted-neighbor coupling:")
print(f"  CC: mean={cc_coupling.mean():.4f}, median={np.median(cc_coupling):.4f}, std={cc_coupling.std():.4f}, n={len(cc_coupling)}")
print(f"  SB: mean={sb_coupling.mean():.4f}, median={np.median(sb_coupling):.4f}, std={sb_coupling.std():.4f}, n={len(sb_coupling)}")

for thresh in [0.5, 0.7, 0.9]:
    cc_pct = (cc_coupling > thresh).mean() * 100
    sb_pct = (sb_coupling > thresh).mean() * 100
    print(f"  Coupling > {thresh}: CC={cc_pct:.1f}%, SB={sb_pct:.1f}%")

# ========== PART 6: DEAD NODE NEIGHBOR ANALYSIS ==========
print("\n" + "=" * 60)
print("PART 6: DEAD NODE NEIGHBOR ANALYSIS")
print("=" * 60)

for name, zero_rate, adj_nd, n, thresh in [("CC", cc_zero_rate, cc_no_diag, cc_n, 0.01),
                                              ("SB", sb_zero_rate, sb_no_diag, sb_n, 0.01)]:
    dead_idx = np.where(zero_rate > 0.9)[0]
    func_idx = np.where(zero_rate <= 0.05)[0]

    # How many functional nodes have dead neighbors?
    func_with_dead_neighbor = 0
    func_dead_neighbor_count = []
    func_dead_neighbor_weight = []

    for f in func_idx:
        neighbors = np.where(adj_nd[f, :] > thresh)[0]
        dead_neighbors = np.intersect1d(neighbors, dead_idx)
        if len(dead_neighbors) > 0:
            func_with_dead_neighbor += 1
            func_dead_neighbor_count.append(len(dead_neighbors))
            func_dead_neighbor_weight.append(adj_nd[f, dead_neighbors].sum())

    print(f"\n{name}:")
    print(f"  Functional nodes: {len(func_idx)}")
    print(f"  Functional with dead neighbor: {func_with_dead_neighbor} ({func_with_dead_neighbor/len(func_idx)*100:.1f}%)")
    if func_dead_neighbor_count:
        print(f"  Avg dead neighbors per affected node: {np.mean(func_dead_neighbor_count):.1f}")
        print(f"  Avg dead neighbor total weight: {np.mean(func_dead_neighbor_weight):.4f}")

print("\n" + "=" * 60)
print("SUMMARY: KEY DIFFERENCES")
print("=" * 60)
print(f"""
1. Graph density: CC={cc_edges/(cc_n*(cc_n-1)):.6f} vs SB={sb_edges/(sb_n*(sb_n-1)):.6f}
2. Avg degree: CC={cc_degree.mean():.1f} vs SB={sb_degree.mean():.1f}
3. Neighbor correlation: CC={cc_corr_mean:.4f} vs SB={sb_corr_mean:.4f}
4. Spatial coupling: CC={cc_coupling.mean():.4f} vs SB={sb_coupling.mean():.4f}
5. Isolated nodes: CC={cc_isolated} vs SB={sb_isolated}
""")
