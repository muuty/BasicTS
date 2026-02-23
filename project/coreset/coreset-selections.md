# Coreset Selection Methods: Implementation Review

Last updated: 2026-02-21

## Overview

All selection methods share a unified interface via `coreset/distance.py`:
- **Feature extraction**: `extract_features(dataset, model_config)` -> `(N, T, Nodes, F)` structured arrays
- **Distance types**: `euclidean`, `temporal`, `spatial`, `combined` — configurable via `CFG.CORESET.DISTANCE_TYPE`
- **Factory**: `coreset/factory.py` — `get_selection(type, ratio, dataset, model_config, distance_type, seed)`

### Available Methods (8 total)

| # | Factory Key | Category | Distance-aware | GPU |
|---|-------------|----------|----------------|-----|
| 1 | `random` | Baseline | No | N/A |
| 2 | `recent` | Baseline | No | N/A |
| 3 | `stride` | Baseline | No | N/A |
| 4 | `k_center` | Coverage (min-max) | Yes | Full |
| 5 | `k_medoids` | Representativeness | Yes | Partial |
| 6 | `total_similarity` | Facility Location | Yes | Full |
| 7 | `graph_cut` | FL + Redundancy penalty | Yes | Full |
| 8 | `herding` | Distribution Matching | Yes | No |

---

## 1. RandomSelection (`coreset/random.py`)

**Category**: Baseline
**Algorithm**: Uniform random sampling without replacement.

| Item | Status |
|------|--------|
| Correctness | OK |
| GPU | N/A |
| Complexity | O(N) |
| Distance-aware | No (distance_type ignored) |

**Notes**: Simplest baseline. Useful as lower bound reference.

---

## 2. RecentSelection (`coreset/recent.py`)

**Category**: Baseline
**Algorithm**: Select the last N samples (most recent in time).

| Item | Status |
|------|--------|
| Correctness | OK |
| GPU | N/A |
| Complexity | O(1) |
| Distance-aware | No |

**Notes**: Tests the "recency bias" hypothesis — recent traffic patterns may be more relevant for prediction.

---

## 3. StrideSelection (`coreset/stride.py`)

**Category**: Baseline (uniform temporal stratification)
**Algorithm**: Select samples at evenly spaced intervals via `np.linspace`.

| Item | Status |
|------|--------|
| Correctness | OK |
| GPU | N/A |
| Complexity | O(1) |
| Distance-aware | No |

**Notes**: Ensures uniform coverage across the entire time period. Complementary to `recent` (which only covers the tail). Acts as a simple stratified sampling baseline.

---

## 4. KCenterGreedySelection (`coreset/k_center.py`)

**Category**: Coverage Maximization (min-max)
**Algorithm**: Greedy 2-approximation. Iteratively select the point farthest from the current selected set.
**Objective**: min_S max_i min_{j in S} d(i, j)

| Item | Status |
|------|--------|
| Correctness | OK |
| GPU | Full (distance matrix + selection loop on GPU) |
| Complexity | O(N^2) distance + O(N*k) selection |
| Distance-aware | Yes |

**Notes**: Incremental `min_dist` update avoids recomputation. Clean and efficient implementation.

---

## 5. KMedoidsSelection (`coreset/k_medoids.py`)

**Category**: Representativeness (Facility Location variant)
**Algorithm**: FasterPAM-inspired swap optimization. Random initialization + iterative swap.
**Objective**: min_S sum_i min_{j in S} d(i, j)

| Item | Status |
|------|--------|
| Correctness | OK (swap logic correct) |
| GPU | Partial (distance on GPU, swap loop has Python overhead) |
| Complexity | O(N^2) distance + O(N*H*k) per swap iteration |
| Distance-aware | Yes |

**Known Issues**:

1. **Memory**: `compute_gain_matrix` creates (H, N) tensors where H ~ N-k. For N=24K, this is ~2.2GB per tensor. Combined with the distance matrix (~2.3GB), total GPU memory ~ 6-7GB. May OOM on smaller GPUs.

2. **Convergence**: `max_iter=5` means at most 5 swaps. For k=2400 (ratio=0.1) this is very few. Random initialization without BUILD phase means starting from a poor solution.

3. **Single swap per iteration**: True FasterPAM performs ALL improving swaps per iteration. This implementation only does the best single swap, making it closer to standard PAM.

4. **`non_medoids` computation**: Python list comprehension `[i for i in range(N) if i not in medoids]` is O(N*k) — slow for large k.

**Recommendation**: Works for current experiments, but may need optimization for ratio > 0.5 or datasets > 20K samples.

---

## 6. TotalSimilaritySelection (`coreset/total_similarity.py`)

**Category**: Representativeness (Facility Location)
**Algorithm**: Greedy submodular maximization with (1-1/e) approximation.
**Objective**: max_S sum_i max_{j in S} sim(x_i, x_j), where sim = -distance

| Item | Status |
|------|--------|
| Correctness | OK |
| GPU | Full (similarity matrix + marginal gain on GPU) |
| Complexity | O(N^2) distance + O(N^2 * k) selection |
| Distance-aware | Yes |

**Known Issues**:

1. **Memory**: Marginal gain computation broadcasts (N, N) tensor: `similarity_matrix - current_max_sim.unsqueeze(1)`. For N=24K this is ~2.3GB. Combined with similarity_matrix, ~4.6GB GPU needed.

2. **Masking loop**: `for idx in selected_indices: marginal_gains[idx] = -inf` grows linearly. For ratio=0.9 and N=24K, the inner mask loop runs up to 21K times per selection step. Could use a boolean mask tensor instead.

3. **sim = -distance**: All similarities are negative. Mathematically correct for the greedy algorithm, but RBF kernel would be more standard.

---

## 7. GraphCutSelection (`coreset/graph_cut.py`)

**Category**: Facility Location + Redundancy Penalty (submodular)
**Algorithm**: Greedy submodular maximization.
**Objective**: f(S) = sum_i max_{j in S} sim(i,j) - lambda * sum_{i,j in S} sim(i,j)

| Item | Status |
|------|--------|
| Correctness | OK |
| GPU | Full (RBF similarity + marginal gain on GPU) |
| Complexity | O(N^2) distance + O(N^2 * k) selection |
| Distance-aware | Yes |

**Key Design Decisions**:

1. **RBF kernel similarity**: Uses `sim = exp(-d^2 / 2*sigma^2)` with median heuristic for sigma. This is critical — with `sim = -distance` (as in FL), the redundancy penalty becomes a diversity bonus since all sim values are negative. RBF gives sim in [0,1], so the penalty properly penalizes selecting similar points.

2. **Lambda parameter**: `lam=1.0` by default. Higher lambda = stronger diversity pressure. Set via factory or config.

3. **Marginal gain**: `delta(e|S) = FL_gain(e) - 2*lambda * sum_{i in S} sim(i,e)`. The constant `sim(e,e)=1` is dropped since it doesn't affect argmax.

**Memory**: Same as FL (~7GB peak for N=24K).

**Reference**: Iyer & Bilmes (2021), DeepCore (Guo et al., 2022)

---

## 8. HerdingSelection (`coreset/herding.py`)

**Category**: Distribution Matching (moment matching)
**Algorithm**: Greedy mean matching. At each step, select the sample that keeps the running mean closest to the global mean.
**Objective**: min ||mean(S) - mean(X)||

| Item | Status |
|------|--------|
| Correctness | OK |
| GPU | No (pure numpy + Python loop) |
| Complexity | O(N * k * D) where D = feature dimension |
| Distance-aware | Yes (via feature type, not distance matrix) |

**Known Issues**:

1. **Performance**: Inner Python loop iterates over all remaining indices. For N=24K, ratio=0.5, D=42864 (euclidean features with 893 nodes): ~24K * 12K * 42K floating point ops. Extremely slow (hours).

2. **Temporal/spatial features help**: With `distance_type='temporal'`, D drops from ~42K to ~48. This makes herding ~900x faster. Spatial: D ~ 3572. Strong practical argument for ST-aware features.

**Recommendation**: Vectorize the inner loop with numpy broadcasting:
```python
remaining_arr = np.array(list(remaining))
candidates = selected_sum + features[remaining_arr]  # (|remaining|, D)
scores = -np.linalg.norm(candidates - target_sum, axis=1)
best_idx = remaining_arr[np.argmax(scores)]
```

---

## Feature Dimension by Distance Type (SAN_BERNARDINO, 893 nodes)

| Distance Type | Feature Dim D | Notes |
|--------------|---------------|-------|
| euclidean | (12*893*3 + 12*893*1) = 42,864 | Very high — expensive for pairwise computation |
| temporal | (12*3 + 12*1) = 48 | Very compact — fast computation |
| spatial | (893*3 + 893*1) = 3,572 | Moderate |
| combined | 48 + 3,572 = 3,620 (concat) | For feature-based methods (herding) |

---

## Distance Matrix Memory (N=24,192 samples)

| Component | Size |
|-----------|------|
| Distance matrix (N x N, float32) | 24192^2 * 4B = 2.3 GB |
| Similarity matrix (if needed) | +2.3 GB |
| Marginal gain broadcast (FL, GraphCut) | +2.3 GB |
| **Total peak (FL, GraphCut)** | **~7 GB** |
| **Total peak (k-Center, k-Medoids)** | **~4.6 GB** |

---

## Summary

| Method | Correct | GPU | Scalability (N=24K) | Priority Fix |
|--------|---------|-----|---------------------|-------------|
| Random | OK | N/A | OK | - |
| Recent | OK | N/A | OK | - |
| Stride | OK | N/A | OK | - |
| k-Center | OK | Full | OK | - |
| k-Medoids | OK | Partial | Risky (memory, convergence) | Memory optimization |
| Facility Location | OK | Full | OK (needs ~7GB VRAM) | Mask optimization |
| GraphCut | OK | Full | OK (needs ~7GB VRAM) | - |
| Herding | OK | No | Very slow (euclidean) | Vectorize inner loop |

---

## Config Example

```python
CFG.CORESET = {
    'SELECTION_STRATEGY': 'graph_cut',  # any of the 8 methods above
    'SELECTION_RATIO': 0.3,
    'DISTANCE_TYPE': 'temporal',        # euclidean | temporal | spatial | combined
    'SEED': 42
}
```
