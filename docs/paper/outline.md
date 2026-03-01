# T-ITS Paper Outline

## Working Title
**Quantization Cost as a Distribution-Aware Proxy for Coreset Selection in Traffic Forecasting**

Alternative: "Why k-Medoids Works Best: A Wasserstein-Based Analysis of Coreset Selection for Traffic Forecasting"

---

## Abstract (~200 words)
- **Problem**: Coreset selection reduces training data while maintaining performance, but no pre-training metric reliably predicts which coreset will yield the best model
- **Approach**: We propose quantization cost (average distance to nearest coreset point) as a proxy metric, grounded in a Wasserstein-1 bound on the test-set performance gap
- **Theory**: The bound decomposes G_test into distributional shift + model discrepancy; k-medoids directly minimizes the controllable term via its connection to W₁
- **Experiments**: SAN_BERNARDINO (893 nodes), 3 selection methods × 4 distance types × 2 ratios × 2 seeds × 2 models (STGCN, AGCRN). Quantization cost achieves r > 0.94 within-setting correlation with test MAE
- **Key finding**: k-medoids is the only method achieving ΔW₁ > 0 (coreset closer to test than full training set)

---

## 1. Introduction (~1.5 pages)

### Hook
- Traffic forecasting 모델의 데이터 효율성 → 대규모 센서 네트워크에서 학습 비용 문제
- Coreset selection이 해결책이 될 수 있으나, "어떤 coreset이 좋은 coreset인가?"에 대한 답이 없음

### Gap
- 기존 coreset 연구는 classification 중심 (Craig, GradMatch 등)
- Traffic forecasting의 특성 (continuous regression, temporal shift, spatial structure)이 반영되지 않음
- 학습 전에 coreset 품질을 예측할 수 있는 metric이 부재

### Contribution (3가지)
1. **Empirical**: Traffic forecasting에서 다양한 coreset selection method를 체계적으로 비교 (k-medoids, k-center, graph-cut × 4 distances × multiple models)
2. **Proxy Metric**: Quantization cost를 학습-전(pre-training) 품질 예측 지표로 제안. Within-setting r > 0.94 달성
3. **Theoretical**: Wasserstein-1 기반 bound로 quantization cost가 왜 좋은 proxy인지 이론적 정당화. k-medoids가 이 bound의 제어 가능한 항을 직접 최소화함을 증명

---

## 2. Related Work (~1 page)

### 2.1 Spatio-Temporal Traffic Forecasting
- GCN 기반: STGCN, DCRNN, GWNet, MTGNN
- Transformer 기반: STAEformer, STGformer
- Adaptive graph: AGCRN, DGCRN
- **Gap**: 데이터 효율성/선택에 대한 연구 부족

### 2.2 Coreset Selection and Data Pruning
- Geometry-based: k-medoids, k-center, facility location
- Gradient-based: Craig, GradMatch
- Classification 벤치마크: DeepCore (Guo et al., 2022)
- **Gap**: Regression task, 특히 시공간 예측에 대한 coreset 연구 거의 없음

### 2.3 Distribution Metrics for Data Quality
- Optimal transport: Wasserstein distance, Sinkhorn divergence
- Maximum Mean Discrepancy (MMD)
- **Gap**: Coreset quality prediction에 활용된 사례 부족

---

## 3. Problem Formulation (~0.5 page)

### 3.1 Traffic Forecasting Setup
- Input: X ∈ R^{T×N×F}, Output: Y ∈ R^{T'×N×F}
- Dataset split: D_train, D_val, D_test

### 3.2 Coreset Selection
- D_core ⊂ D_train, |D_core| = k = ⌊r·n⌋
- Goal: MAE_{D_test}(f_core) ≈ MAE_{D_test}(f_train)
- Methods compared: k-medoids, k-center, graph-cut

### 3.3 Distance Metrics
- Feature space: PCA-10 embedding (82.2% variance explained)
- 4 distance types: euclidean, temporal, spatial, combined

---

## 4. Theoretical Analysis (~2 pages)

### 4.1 Performance Gap Bound
- Setup: finite datasets, MAE, ground distance d
- Triangle inequality → (A)(B)(C) decomposition
- KR inequality for (A) and (C)
- Weighted MAE for (B)
- **Final bound**: G_test ≤ (L_train + L_core)[W₁(test,train) + W₁(train,C;w)] + ε_disc

### 4.2 Connection to k-Medoids
- Claim: W₁(X_train, C; w) ≤ J_kmed(C)
- Proof: nearest-assignment plan is feasible for W₁
- Design principle: k-medoids directly minimizes the controllable term

### 4.3 Why Not Minimize W₁ Directly?
- Computational complexity of exact W₁ (LP solver, O(n³))
- J_kmed as practical surrogate
- Empirically J_kmed ≈ W₁ (ε_disc dominates anyway)

### 4.4 When Can the Coreset Model Win?
- Intuitive argument: Δ_test ≈ L·[W₁(test,core) - W₁(test,train)]
- k-medoids makes W₁(train,core) small → W₁(test,core) ≈ W₁(test,train)
- Outlier dropping mechanism → possible Δ_test < 0
- Rigorous sufficient condition

### 4.5 Why Traffic Forecasting?
- MAE is Lipschitz (unlike 0-1 loss in classification)
- Near-deterministic Y|X → conditional noise ≈ 0
- Density-proportional coverage optimal (vs. decision-boundary in classification)
- Temporal distribution shift makes W₁-based analysis natural

---

## 5. Proposed Framework (~1 page)

### 5.1 Coreset Selection Pipeline
- Feature extraction → PCA-10 → distance computation → selection → training
- Offline computation (selection indices saved to JSON)

### 5.2 Quantization Cost as Proxy Metric
- Definition: Quant(S) = (1/n) Σ min_{s∈S} d(x_i, s)
- Connection to bound: Quant = J_kmed ≥ W₁(train, C; w)
- Pre-training computation: O(nk) after PCA

---

## 6. Experiments (~3 pages)

### 6.1 Setup
- Dataset: SAN_BERNARDINO (893 nodes, 3 months, 5-min interval)
  - Second dataset: CONTRA_COSTA (773 nodes) for generalization
- Models: STGCN, AGCRN (+ potentially STAEformer, DCRNN)
- Baselines: k-medoids, k-center, graph-cut × euclidean/temporal/spatial/combined
- Ratios: 0.3, 0.7
- Seeds: 42, 123 (+ 456 for key settings)
- Metric: MAE, RMSE

### 6.2 Finding 1: Quantization Cost Predicts MAE
- **Table**: Method × distance별 quant cost vs MAE
- **Figure**: Scatter plot (quant_pca vs MAE) with correlation r > 0.94
- graph_cut의 quant cost 6~7x → MAE +2.5
- 다른 proxy metric (sinkhorn, voronoi_cv, ess_ratio)와 비교: quant가 최강

### 6.3 Finding 2: Coreset Advantage (ΔW₁ > 0)
- **Table**: Method별 SW₁(test,core), SW₁(core,train), ΔW₁
- k-medoids만 ΔW₁ > 0 달성 (8/96 cases)
- k-medoids의 SW₁(core,train) 극도로 작음 (0.17~0.52)
- Mechanism: J_kmed 최소화 → 삼각부등식 → coreset ≈ train from test's perspective

### 6.4 Finding 3: Bound Verification
- **Table**: (A)(B)(C) decomposition + sub-decomposition results
- L values 매우 작음 (~0.001)
- ε_disc dominates (~95%)
- Bound holds: L95 ~77%, Lmax ~82%
- graph_cut ratio=0.3: 0% (bound too loose for poor coresets)

### 6.5 Ablation Studies
- Distance type effect (combined vs single)
- Ratio sensitivity (0.3 vs 0.7)
- Seed stability

---

## 7. Discussion (~0.5 page)

- ε_disc가 bound의 dominant term → 향후 연구 방향 (tighter bound)
- graph_cut의 실패 이유: W₁(train,core) 과도하게 큼
- Limitations: single-county dataset, 2 models, PCA-10 embedding 선택

---

## 8. Conclusion (~0.5 page)
- 요약: 3 contributions
- Practical implication: 학습 전 quant cost로 coreset 품질 예측 가능
- Future work: 더 많은 데이터셋, gradient-based methods, tighter bound

---

## Figures & Tables Plan

### Tables
1. **Table I**: Experiment setup summary (dataset, models, methods, etc.)
2. **Table II**: Quantization cost vs MAE by method (ratio=0.3, 0.7)
3. **Table III**: Proxy metric correlation comparison
4. **Table IV**: SW₁ coreset advantage
5. **Table V**: Bound verification (A)(B)(C) decomposition
6. **Table VI**: Bound holds rate by method/ratio

### Figures
1. **Fig 1**: Overview diagram (coreset pipeline + theory connection)
2. **Fig 2**: Scatter: quantization cost vs MAE (color by method)
3. **Fig 3**: Bar chart: ΔW₁ by method
4. **Fig 4**: Bound decomposition stacked bar

---

## Data Sources (existing files)

| Content | Source File |
|---------|------------|
| Quant cost, W₁ | `experiments/result/analysis/bound_terms_cpu.csv` |
| SW₁ advantage | `experiments/result/analysis/coreset_advantage.csv` |
| Bound verification | `experiments/result/analysis/bound_verification.csv` |
| Proxy correlations | `experiments/result/analysis/proxy_correlations.csv` |
| Existing scatter plot | `experiments/result/analysis/quantization_vs_mae.png` |
| Existing advantage plot | `experiments/result/analysis/coreset_advantage.png` |
