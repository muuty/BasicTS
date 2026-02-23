# Literature Review: Adjacency Matrix Methods in GNN Traffic Prediction

## 1. Overview

이 문서는 GNN 기반 교통 예측에서 adjacency matrix 방법론과 관련된 기존 연구를 정리한다.

## 2. Key Finding: Research Gap

**동일한 adjacency method set을 여러 GNN architecture에 걸쳐 체계적으로 비교한 연구는 존재하지 않는다.**

기존 연구 유형:
| 유형 | 하는 것 | 우리와의 차이 |
|------|--------|-------------|
| Survey/분류 | Graph construction 방법론 분류 | 실험적 비교 없음 |
| 새 방법 제안 | 자기 방법의 우수성 입증 | Cross-method 비교 아님 |
| 개별 모델 ablation | 모델 내 predefined vs. adaptive | Model-specific, 체계적이지 않음 |

---

## 3. Core References

### 3.1 Survey & Taxonomy Papers

#### STG4Traffic (arXiv 2023)
- **Citation**: Liang et al., "STG4Traffic: A Survey and Benchmark of Spatial-Temporal Graph Neural Networks for Traffic Prediction"
- **URL**: https://arxiv.org/abs/2307.00495
- **Graph Construction Taxonomy**:
  - **Predefined Graph**: Distance, Connectivity, Semantic (DTW), Functionality (POI), Distribution (JS divergence)
  - **Adaptive Graph**: Parameter-based (learnable embeddings), Gumbel-softmax sampled
- **Benchmark**: 18 models on METR-LA, PEMS-BAY, PEMSD4, PEMSD8
- **Key Gap Identified**: *"No golden measure of learned graph quality exists, other than prediction accuracy"*
- **Table 2**: 30+ 모델이 어떤 graph type을 사용하는지 정리 (distance, connectivity, multi-graph, adaptive, dynamic, sampled)
- **Limitation**: 방법론 분류만 수행, **graph construction method를 cross-model로 비교하는 실험은 없음**
- **Relevance**: 우리 연구의 taxonomy 기반으로 활용 가능

#### Emerging Trends in GNNs for Traffic Flow Prediction (2025)
- **URL**: https://link.springer.com/article/10.1007/s11831-025-10286-9
- ~40 SOTA 모델을 5개 데이터셋에서 평가
- Adjacency 방법론에 대한 체계적 비교는 없음

#### GNN4Traffic Repository
- **URL**: https://github.com/jwwthu/GNN4Traffic
- GNN 교통 예측 논문 모음 (지속 업데이트)
- 논문 목록 참고용

---

### 3.2 Representation Learning & Bootstrapping

#### CLEAR (TKDE 2025)
- **Citation**: Yu et al., "CLEAR: Spatial-Temporal Traffic Data Representation Learning for Traffic Prediction"
- **Venue**: IEEE Transactions on Knowledge and Data Engineering, Vol. 37, No. 4, April 2025
- **PDF**: `docs/ref/CLEAR_Spatial-Temporal_Traffic_Data_Representation_Learning.pdf`
- **Key Contributions**:
  - Contrastive learning으로 time-series + graph data representation 학습
  - C1-C4 모델 카테고리 정의 (Section IV-E)
  - 10개 모델 × 4개 데이터셋 bootstrapping 실험
- **C1-C4 Categories**:
  - C1: Static adjacency (geo-distance) — ASTGCN, STSGCN, STGODE
  - C2: Adaptive adjacency (E₁E₂ᵀ) — GWNet, AGCRN
  - C3: Spatial-temporal attention — GMAN, STWave+
  - C4: Spatial-temporal representation — STAEFormer, GMSDR, DGCRN
- **Key Result (TABLE III)**:
  - C1 improvement: +11.4% (가장 큰 개선)
  - C2: +10.4%
  - C3: +9.6%
  - C4: +4.9% (가장 작은 개선)
  - → Graph가 단순할수록 더 좋은 representation 교체 시 이득이 큼
- **Datasets**: Beijing (3126 nodes), NI-SH (1830), PeMS04 (307), METR-LA (207)
- **Relevance**: 모델 카테고리 분류, 실험 규모/scope 참고, 상호보완적 연구

---

### 3.3 Graph Structure Learning

#### RAGL (arXiv 2025)
- **Citation**: Wu et al., "Regularized Adaptive Graph Learning for Large-Scale Traffic Forecasting"
- **URL**: https://arxiv.org/abs/2506.07179
- Adaptive graph의 regularization + scalability 문제 해결
- Stochastic Shared Embedding (SSE) + Efficient Cosine Operator (ECO)
- Large-scale dataset에서 SOTA
- **Relevance**: Adaptive graph의 한계와 개선 방향 참고

#### GTS (ICLR 2022)
- Graph structure를 Gumbel-Softmax로 discrete하게 학습
- DCGRUCell이 forward()에서 adj를 받는 가장 유연한 설계
- **Relevance**: Learnable discrete graph 방법론

---

### 3.4 "Do We Need Graph?" Challenge

#### STID (CIKM 2022)
- **Citation**: Shao et al., "Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting"
- **URL**: https://arxiv.org/abs/2208.05233
- MLP + spatial/temporal identity embedding만으로 GNN과 경쟁
- Pre-defined graph 불필요, 모든 데이터셋에서 best 또는 near-best
- **Key Insight**: Graph structure보다 node identity 정보가 더 중요할 수 있음
- **Relevance**: "No graph" baseline으로 포함. Graph의 실제 가치를 검증하는 대조군

---

### 3.5 Attention + Graph Hybrid Approaches

#### STGAFormer (Information Fusion 2024)
- **URL**: https://dl.acm.org/doi/10.1016/j.inffus.2024.102228
- Spatial-Temporal Gated Attention Transformer + GNN
- Gated fusion으로 temporal/spatial attention 결합
- **Relevance**: Dynamic adjacency 방법론 참고 (C3 모델)

#### DynamicChebNet (Scientific Reports 2025)
- **URL**: https://www.nature.com/articles/s41598-025-12598-z
- Multi-graph adjacency fusion: physical connectivity + DTW similarity + dynamic data-driven
- **Relevance**: 여러 adjacency를 결합하는 방법론

#### MLCAFormer (PLOS One 2025)
- Multi-level causal attention + node-identity-aware spatial attention
- Node별 unique identity embedding으로 spatial correlation 학습
- **Relevance**: Node identity + attention 결합 방법론

---

### 3.6 Individual Model Papers (Adjacency Method Origin)

| Model | Paper | Adj Method | Key Innovation |
|-------|-------|-----------|----------------|
| DCRNN | Li et al., ICLR 2018 | Distance-based diffusion | Bidirectional random walk on distance graph |
| STGCN | Yu et al., IJCAI 2018 | Distance Chebyshev | Chebyshev polynomial on Laplacian |
| GWNet | Wu et al., IJCAI 2019 | Adaptive + predefined | Learnable E₁E₂ᵀ + predefined supports |
| AGCRN | Bai et al., NeurIPS 2020 | Fully adaptive | Node-adaptive weights + learned graph |
| MTGNN | Wu et al., KDD 2020 | Mixprop + adaptive | Sparse directed graph from embeddings |
| DGCRN | Li et al., TKDD 2023 | Dynamic + predefined | Hyper-network generates per-step adj |
| STAEFormer | Liu et al., CIKM 2023 | Full attention | Spatio-temporal adaptive embedding |
| STGODE | Fang et al., KDD 2021 | ODE + dual graph | Neural ODE with spatial + semantic adj |

---

## 4. Adjacency Method Taxonomy (Synthesized)

기존 연구들을 종합하면 adjacency method는 다음과 같이 분류됨:

### Level 1: Static (Training/inference에서 고정)

#### 1a. Prior Knowledge-based
- **Distance (Gaussian kernel)**: A_ij = exp(-d²/σ²) [DCRNN, STGCN]
- **Binary connectivity**: A_ij = 1 if connected [간단한 baseline]
- **k-NN (distance)**: k nearest neighbors [STWave]

#### 1b. Data-driven (학습 데이터에서 계산, 이후 고정)
- **Correlation**: Pearson correlation of traffic patterns
- **DTW similarity**: Dynamic Time Warping [STGODE semantic graph]
- **Functionality**: POI distribution similarity [STG4Traffic 분류]
- **Distribution**: JS divergence of traffic flow distribution [STG4Traffic 분류]

### Level 2: Semi-static (학습 중 변하지만, 추론 시 고정)

- **Learnable embedding (E·Eᵀ)**: softmax(ReLU(E₁E₂ᵀ)) [GWNet, AGCRN]
- **Gumbel-softmax discrete**: Differentiable discrete graph [GTS]

### Level 3: Dynamic (매 sample/timestep마다 변함)

- **Full attention**: (N,N) attention per timestep [STAEFormer]
- **Hyper-network generated**: Per-step adj from hyper-GCN [DGCRN]
- **Input-conditioned**: Adj computed from current input features [D2STGNN]

### Level 4: Hybrid (여러 레벨 결합)
- **Predefined + adaptive**: GWNet, DGCRN
- **Multi-graph fusion**: DynamicChebNet (physical + semantic + dynamic)
- **Attention + graph mask**: Sparse attention guided by predefined graph

---

## 5. Research Positioning

### What exists:
- ✅ Taxonomy/classification of graph construction methods (STG4Traffic, surveys)
- ✅ New adjacency methods proposed yearly (RAGL, GTS, etc.)
- ✅ Model-specific ablations (predefined vs. adaptive within one model)
- ✅ CLEAR: representation bootstrapping across models (but for their specific method)

### What does NOT exist (= Our contribution):
- ❌ Cross-model controlled experiment: same adjacency methods tested across multiple architectures
- ❌ Full spectrum benchmark: static → semi-static → dynamic 통합 비교
- ❌ Parameter sensitivity analysis for key adjacency methods
- ❌ Practical guidelines: which adjacency for which model/dataset/scenario

### Proposed Positioning:
> "While existing research has categorized graph construction methods (surveys) or proposed new methods (individual papers), no study has systematically evaluated the same set of adjacency methods across multiple GNN architectures in a controlled experiment. We present the first comprehensive benchmark spanning static to dynamic adjacency methods across C1-C4 model categories, providing parameter sensitivity analysis and practical guidelines for adjacency method selection."
