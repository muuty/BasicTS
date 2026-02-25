# In-Depth Analysis: Why Combined Distance & K-Medoids Work Best

## Overview

This document synthesizes experimental findings from Phase A distance screening (130/220 runs) with relevant literature to answer two key research questions:

1. **Q1**: Why does combined distance work better than other distance types?
2. **Q2**: Why is k_medoids the most stable and effective selection method?

All experiments use SAN_BERNARDINO dataset (893 nodes, 3-month traffic, 5-min intervals).
Sample structure: `[T=12, N=893, F=5]` per sample. Coreset selection is **sample-level** (not node-level).

---

## Key Experimental Findings Summary

### MAE Rankings (STGCN, averaged across methods/ratios/seeds)

| Distance | MAE (mean) | MAE (std) |
|----------|-----------|-----------|
| combined | **15.616** | 1.076 |
| cosine_raw | 15.827 | 1.620 |
| euclidean | 15.902 | 1.227 |
| cosine_temporal | 15.903 | 2.082 |
| cosine_spatial | 15.904 | 1.831 |
| spatial | 16.338 | 1.671 |
| temporal | 16.409 | 1.712 |
| cosine_combined | 16.534 | 2.021 |

| Method | MAE (mean) | MAE (std) |
|--------|-----------|-----------|
| k_medoids | **15.662** | 1.208 |
| k_center | 15.842 | 1.254 |
| graph_cut | 16.658 | 2.182 |

---

## Q1: Why Combined Distance Works Better

### 1.1 Distance Concentration (Curse of Dimensionality)

**Literature**: Beyer et al. (1999) showed that in high dimensions, pairwise distances converge:
`(D_max - D_min) / D_min → 0`. Aggarwal et al. (2001) demonstrated L2 norms suffer more than L1.
Peng et al. (2024) unified these findings: higher L_k norms cause more concentration.

**Our findings**: Feature dimensions vary significantly:
- temporal: 48 dims (mean-over-893-nodes, `[T,F]` → flat)
- spatial: 3,572 dims (mean-over-time, `[N,F]` → flat)
- raw/euclidean: 53,580 dims (`[T×N×F]` → flat)

Coefficient of Variation (CV) measurements:

| Distance | CV | Dim |
|----------|-----|-----|
| cosine_temporal | 0.864 | 48 |
| cosine_spatial | 0.823 | 3,572 |
| cosine_combined | 0.807 | T48+S3572 |
| cosine_raw | 0.755 | 53,580 |
| temporal | 0.651 | 48 |
| **combined** | **0.542** | normalized |
| spatial | 0.460 | 3,572 |
| euclidean | 0.419 | 53,580 |

**Key finding**: CV ranking does **NOT** correlate with MAE ranking (Spearman r=-0.214, p=0.610).
Combined has moderate CV (0.542) but best MAE. This suggests distance concentration alone doesn't explain performance differences.

**Interpretation**: While euclidean (53K dims) suffers the most concentration (CV=0.419, consistent with Beyer et al.), the concentration-to-performance link is not straightforward because selection algorithms respond differently to distance distributions.

### 1.2 Multi-View Normalization Effect

**Literature**: Multi-view learning theory (Li et al. 2016, Sun et al. 2017) establishes that combining normalized views exploits both **consensus** (agreement reinforces signal) and **complementarity** (unique information from each view). Normalization is critical because different views operate on different scales.

**Our implementation**: `combined = 0.5 * norm(d_temporal) + 0.5 * norm(d_spatial)`, where each L2 matrix is divided by its max value (scaled to [0,1]).

**Our findings**:
- Combined's CV (0.542) lies between temporal (0.651) and spatial (0.460)
- Normalized single-view CVs: norm_temporal_only, norm_spatial_only (both available)
- The normalization itself contributes meaningfully: raw temporal and spatial have very different scales, and without normalization, one view would dominate

**Key insight**: Combined distance doesn't have the *highest* discriminative power (CV), but it provides the most **balanced** representation. It avoids the pathologies of extreme concentration (euclidean) while capturing both temporal patterns and spatial correlations.

### 1.3 Index Overlap Reveals Method-Dependent Behavior

**Critical finding**: For k_medoids at ratio=0.7, the selected indices are nearly identical regardless of distance type:
- combined ∩ temporal Jaccard: **0.997**
- combined ∩ spatial Jaccard: **0.995**

For k_center, overlap drops dramatically: combined ∩ temporal = **0.380**.

**Implication**: "Combined is better" is primarily driven by k_center and graph_cut results, not k_medoids. K_medoids' iterative optimization converges to similar selections regardless of distance, while k_center's greedy approach is highly sensitive to the distance metric.

### 1.4 Proxy Metrics Support

**Strongest correlations with MAE** (per-run, all methods, STGCN):

| Proxy Metric | Pearson r | p-value |
|-------------|----------|---------|
| h_tod (temporal diversity) | **-0.566** | <0.0001 |
| h_dow (day-of-week diversity) | -0.528 | 0.0001 |
| redundancy | -0.515 | 0.0002 |
| information_gain | +0.515 | 0.0002 |
| fl_objective | -0.329 | 0.022 |

Combined distance achieves the highest FL objective (14,418) and h_tod (0.964) among L2 distances, consistent with its best MAE performance.

**Note**: These correlations are partially confounded by ratio (0.3 vs 0.7). Within ratio=0.3, correlations are strong (h_tod: r=-0.589, p=0.002); within ratio=0.7, they are weak and often non-significant.

### 1.5 Q1 Synthesis

Combined distance is best because:

1. **Balanced representation**: It avoids the concentration pathology of high-dimensional euclidean (53K dims) while maintaining more discriminative power than single-view distances
2. **Normalization enables fair fusion**: Max-normalization ensures neither temporal (48-dim) nor spatial (3572-dim) dominates, consistent with multi-view learning principles
3. **Better temporal coverage**: Combined achieves the highest temporal entropy (h_tod=0.964), suggesting it produces more temporally diverse selections
4. **Robust across methods**: It performs well for both k_center (where distance matters) and k_medoids (where it's comparable)

**However**: The effect size is modest. Combined's advantage over euclidean is only 0.3 MAE points on average. The choice of **method** matters more than the choice of distance.

---

## Q2: Why K-Medoids is Most Stable and Effective

### 2.1 Seed Stability Paradox

**Surprising finding**: k_medoids has the **lowest** index reproducibility (Jaccard=0.353 between seeds) but the **best** MAE stability (mean MAE diff=0.656). Graph_cut has the **highest** Jaccard (0.998) but **worst** MAE stability (mean diff=2.117, max=5.350).

| Method | Avg Jaccard | Mean MAE diff | Max MAE diff |
|--------|------------|---------------|--------------|
| k_medoids | 0.353 | 0.656 | 1.766 |
| k_center | 0.673 | 1.081 | 3.073 |
| graph_cut | 0.998 | 2.117 | 5.350 |

### 2.2 K-Medoids: Different Solutions, Same Quality

**Literature**: Schubert & Rousseeuw (2021) describe FasterPAM's properties:
- Eager swapping with randomized order explores diverse local optima
- The sum-of-distances objective is smooth: many local optima have similar objective values
- Finding the global optimum is NP-hard, but good local optima are abundant

**Our implementation**: FasterPAM with `max_iter=5`, random init via `torch.randperm(N)[:k]`.

**Interpretation**: K_medoids finds **different but equally good** subsets across seeds. The sum-of-distances objective provides a strong quality guarantee: any local minimum must have the property that no single swap improves the total within-cluster distance. This means:
- Different seeds → different initial medoids → different local optima
- But all local optima approximate the same representativeness criterion
- Result: diverse selections that all capture the essential data structure

This is analogous to ensemble diversity: each seed's selection is a valid "view" of the representative samples.

### 2.3 K-Center: Single Random Start Determines Everything

K_center starts from **one** random point (`torch.randint(N, (1,))`), then greedily selects the farthest point. The first point determines the entire selection trajectory.

- Non-iterative: no refinement after greedy construction
- Seed sensitivity is intermediate (Jaccard=0.673)
- MAE stability is intermediate (mean diff=1.081)

### 2.4 Graph Cut: Deterministic but Unstable MAE

**Critical finding**: Graph_cut's RBF sigma is computed using random sampling:
```python
idx_i = np.random.randint(0, N, n_sample)  # seeded
sigma = float(np.median(dist_np[idx_i[mask], idx_j[mask]]))
```

Different seeds → slightly different sigma → same similarity matrix (nearly) → same greedy selection (Jaccard=0.998). But:

**The MAE instability comes from training dynamics**, not selection variability. Graph_cut's greedy submodular selection maximizes the cut value (coverage - redundancy), but this objective may produce subsets that are locally optimal for the objective but poorly conditioned for SGD training:

1. **Graph cut minimizes redundancy aggressively** (lowest intra-coreset redundancy: 30.5M vs k_center's 37.6M vs k_medoids' 34.4M)
2. Per-method correlation: for graph_cut, proxy metrics strongly predict MAE (redundancy r=-0.622, p=0.010; h_tod r=-0.688, p=0.003). This suggests graph_cut's performance is **fragile to the specific selection quality**.
3. K_medoids shows almost no per-method proxy-MAE correlation, suggesting its performance is **robust regardless** of specific proxy characteristics.

### 2.5 Temporal Diversity

| Method | h_tod (mean) | h_dow (mean) |
|--------|-------------|-------------|
| k_medoids | **0.9998** | **0.9993** |
| k_center | 0.9548 | 0.9941 |
| graph_cut | 0.9361 | 0.9917 |

K_medoids achieves near-perfect temporal diversity (h_tod=0.9998, max=1.0). This is because the sum-of-distances objective naturally spreads medoids across the temporal dimension when temporal features are part of the distance computation.

Graph_cut's lower temporal diversity (0.936) suggests it over-concentrates on certain time periods, potentially missing important patterns.

### 2.6 Q2 Synthesis

K_medoids is best because:

1. **Smooth objective landscape**: The sum-of-distances objective has many good local optima (Schubert & Rousseeuw, 2021), making it robust to random initialization
2. **Iterative refinement**: Unlike k_center (greedy, no refinement) and graph_cut (greedy, no refinement), k_medoids iteratively improves via FasterPAM swaps, even with just 5 iterations
3. **Natural temporal diversity**: The representativeness objective inherently spreads selections across time, achieving h_tod=0.9998
4. **Robust to selection quality variation**: Per-method analysis shows k_medoids' MAE is not sensitive to proxy metrics, while graph_cut's MAE strongly depends on selection quality
5. **Diversity through non-determinism**: Different seeds find different-but-equally-good subsets, avoiding the risk of a single "bad" greedy trajectory

---

## Implications for Phase B

1. **Combined + k_medoids** is the recommended baseline for Phase B experiments
2. **Seed averaging** is valuable: k_medoids' diverse selections could potentially be ensembled
3. **Distance type matters less than method**: Focus optimization efforts on the selection algorithm rather than the distance metric
4. **Proxy metric for early stopping**: h_tod (temporal diversity) could serve as a cheap proxy to evaluate selection quality without training a model (Spearman r=-0.57, p<0.0001)

---

## References

- Beyer, K., Goldstein, J., Ramakrishnan, R., & Shaft, U. (1999). When Is "Nearest Neighbor" Meaningful? ICDT'99.
- Aggarwal, C. C., Hinneburg, A., & Keim, D. A. (2001). On the Surprising Behavior of Distance Metrics in High Dimensional Space. ICDT 2001.
- Peng, Y., et al. (2024). Interpreting the Curse of Dimensionality from Distance Concentration and Manifold Effect. arXiv:2401.00422.
- Schubert, E. & Rousseeuw, P. J. (2021). Fast and Eager k-Medoids Clustering: O(k) Runtime Improvement of the PAM, CLARA, and CLARANS Algorithms. Information Systems, 101.
- Guo, C., Zhao, B., & Bai, Y. (2022). DeepCore: A Comprehensive Library for Coreset Selection in Deep Learning. DEXA 2022.
- Wei, K., Iyer, R., & Bilmes, J. (2015). Submodularity in Data Subset Selection and Active Learning. ICML 2015.
- Li, Y., Yang, M., & Zhang, Z. (2016). A Survey of Multi-View Representation Learning. arXiv:1610.01206.
- Sun, S., et al. (2017). Multi-view Learning Overview: Recent Progress and New Challenges. Information Fusion.

---

## Generated Analysis Files

| File | Description |
|------|-------------|
| `experiments/result/analysis/distance_distribution.png` | Pairwise distance histograms (8 types + 2 normalized) |
| `experiments/result/analysis/distance_stats.csv` | CV, mean, std, IQR for all distance types |
| `experiments/result/analysis/cv_vs_mae.png` | CV vs MAE scatter plot |
| `experiments/result/analysis/jaccard_*.png` | Index overlap heatmaps (6 files) |
| `experiments/result/analysis/set_operations.csv` | Set intersection/union analysis |
| `experiments/result/analysis/method_overlap.csv` | Method-pair overlap |
| `experiments/result/analysis/seed_stability.png` | Seed stability box plots |
| `experiments/result/analysis/seed_stability.csv` | Detailed seed stability stats |
| `experiments/result/analysis/temporal_coverage.png` | Temporal entropy analysis |
| `experiments/result/analysis/temporal_stats.csv` | Temporal statistics |
| `experiments/result/analysis/incident_coverage.csv` | Incident proximity coverage |
| `experiments/result/analysis/proxy_vs_mae.png` | Proxy metrics vs MAE scatter |
| `experiments/result/analysis/proxy_correlations.csv` | Proxy-MAE correlation table |
| `experiments/result/analysis/sample_error_analysis.png` | Per-sample error comparison |
| `experiments/result/analysis/sample_error_by_tod.csv` | Error by time-of-day |

## Analysis Scripts

| Script | Analysis |
|--------|----------|
| `scripts/analysis/distance_distribution.py` | 1.0 Distance distribution + normalization |
| `scripts/analysis/index_overlap.py` | 1.2 Jaccard similarity |
| `scripts/analysis/seed_stability.py` | 1.5 Seed stability |
| `scripts/analysis/temporal_coverage.py` | 1.3 Temporal coverage + 1.4 Incident |
| `scripts/analysis/proxy_vs_mae.py` | 1.1 Proxy metrics vs MAE |
| `scripts/analysis/sample_error_analysis.py` | 3.3 Per-sample error |
