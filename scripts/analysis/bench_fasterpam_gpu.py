#!/usr/bin/env python
"""
Verify correctness of GPU FasterPAM:
  1. Tiny case (N=50, k=5): compare against brute-force global optimum
  2. Small case (N=500): compare loss vs Rust kmedoids
  3. Real data ratio 0.1: just confirm GPU finishes quickly and cost is finite

Cross-check: Rust kmedoids is the reference implementation.
If our GPU version gets same or better loss on same dist matrix, implementation is correct.
"""
import os, sys, time, json
import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from itertools import combinations
import kmedoids as km_rust

sys.path.append(os.path.abspath(__file__ + '/../../..'))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from coreset.fasterpam_gpu import fasterpam_gpu, _swap_fasterpam_gpu

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}\n")


# =============================================================
# Test 1: Tiny case (N=30, k=4) — brute-force global optimum
# =============================================================
print("="*60)
print("  TEST 1: N=30, k=4  vs brute-force global optimum")
print("="*60)

np.random.seed(0)
N, k = 30, 4
X = np.random.randn(N, 5).astype(np.float32)
D_np = cdist(X, X, metric='cityblock')
D = torch.from_numpy(D_np).to(device).float()

t0 = time.time()
best_cost_brute = float('inf')
for combo in combinations(range(N), k):
    cost = D_np[:, list(combo)].min(axis=1).sum()
    if cost < best_cost_brute:
        best_cost_brute = cost
t_brute = time.time() - t0
print(f"Brute-force optimum: cost={best_cost_brute:.4f}, time={t_brute:.2f}s")

t0 = time.time()
medoids_gpu = fasterpam_gpu(D, k, init='build', max_iter=100, verbose=False)
cost_gpu = D[:, medoids_gpu].min(dim=1).values.sum().item()
t_gpu = time.time() - t0
print(f"GPU FasterPAM:       cost={cost_gpu:.4f}, time={t_gpu:.3f}s")

result = km_rust.fasterpam(D_np.astype(np.float64), k, max_iter=100, init='build')
print(f"Rust FasterPAM:      cost={result.loss:.4f}")

match_brute = abs(cost_gpu - best_cost_brute) < 1e-3
print(f"\nGPU == brute-force: {match_brute}  (diff={cost_gpu - best_cost_brute:.6f})")

# =============================================================
# Test 2: N=500, k=30 — compare GPU loss vs Rust loss
# =============================================================
print("\n" + "="*60)
print("  TEST 2: N=500, k=30  vs Rust reference")
print("="*60)

np.random.seed(1)
N, k = 500, 30
X = np.random.randn(N, 10).astype(np.float32)
D_np = cdist(X, X, metric='cityblock')
D = torch.from_numpy(D_np).to(device).float()

t0 = time.time()
medoids_gpu = fasterpam_gpu(D, k, init='build', max_iter=200, verbose=False)
cost_gpu = D[:, medoids_gpu].min(dim=1).values.sum().item()
t_gpu = time.time() - t0
print(f"GPU FasterPAM:  cost={cost_gpu:.2f}, time={t_gpu:.2f}s")

t0 = time.time()
result = km_rust.fasterpam(D_np.astype(np.float64), k, max_iter=200, init='build')
t_rust = time.time() - t0
print(f"Rust FasterPAM: loss={result.loss:.2f}, n_iter={result.n_iter}, n_swap={result.n_swap}, time={t_rust:.2f}s")

ratio = cost_gpu / result.loss
match = abs(ratio - 1.0) < 0.01
print(f"\ncost ratio (GPU/Rust): {ratio:.4f}")
print(f"Within 1%: {match}")

# =============================================================
# Test 3: Monotonic cost decrease check (SWAP correctness)
# =============================================================
print("\n" + "="*60)
print("  TEST 3: SWAP monotonicity check (cost must decrease)")
print("="*60)

np.random.seed(2)
N, k = 300, 20
X = np.random.randn(N, 8).astype(np.float32)
D_np = cdist(X, X, metric='cityblock')
D = torch.from_numpy(D_np).to(device).float()

# Random init then SWAP
torch.manual_seed(42)
medoids = torch.randperm(N, device=device)[:k].long()
prev = D[:, medoids].min(dim=1).values.sum().item()
init_cost = prev
print(f"Initial (random) cost: {prev:.2f}")

costs = [prev]
for step in range(20):
    medoids = _swap_fasterpam_gpu(D, medoids.clone(), max_iter=1, batch_size=128, verbose=False)
    cur = D[:, medoids].min(dim=1).values.sum().item()
    costs.append(cur)
    if cur > prev + 1e-4:
        print(f"  FAIL at step {step}: {prev:.2f} -> {cur:.2f}")
    prev = cur

print(f"Final cost: {prev:.2f}  (decreased: {init_cost - prev:.2f})")
monotonic = all(costs[i+1] <= costs[i] + 1e-4 for i in range(len(costs)-1))
print(f"Monotonic decrease: {monotonic}")

# =============================================================
# Summary
# =============================================================
print("\n" + "="*60)
print("  SUMMARY")
print("="*60)
print(f"  Test 1 (brute-force match):   {'PASS' if match_brute else 'FAIL'}")
print(f"  Test 2 (within 1% of Rust):   {'PASS' if match else 'FAIL'}")
print(f"  Test 3 (SWAP monotonic):      {'PASS' if monotonic else 'FAIL'}")
