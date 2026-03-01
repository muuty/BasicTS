# Literature Reference Base: Traffic Denoising & Augmentation

> Noise-resilient spatio-temporal traffic forecasting 관련 논문 정리.
> 우리 연구와의 관계, 차별점 포함.

*Last updated: 2026-02-25*

---

## A. 직접 경쟁/관련 논문 (Must-cite)

### SAFER-Predictor (2025)
- **Venue**: Communications in Transportation Research
- **Core Idea**: Sparse adversarial training for robust traffic prediction under missing+noisy data. Two-phase: pretrain on clean → adversarial fine-tuning with worst-case perturbations at subset of nodes.
- **Noise Handling**: Min-max optimization으로 worst-case noise pattern 탐색. Random noise가 아닌 adversarial perturbation.
- **우리와의 관계**: 가장 직접적 경쟁자. 우리 noise aug와 유사하나 adversarial (worst-case) vs empirical (realistic failure modes).
- **차별점**: SAFER는 robustness만 → 우리는 robustness + denoising encoder + cross-dataset analysis + spillover measurement.
- **Links**: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2772424725000320) | [ResearchGate](https://www.researchgate.net/publication/393510269)

### RT-GCN (2024)
- **Venue**: Information Fusion, Vol. 102
- **Core Idea**: Gaussian node representation (mean + variance) + variance-based attention + Batch Random Noise injection during training.
- **Noise Handling**: 노드를 점 추정이 아닌 분포(평균+분산)로 모델링. 분산 기반 attention이 high-uncertainty 센서를 down-weight.
- **우리와의 관계**: Credibility bias와 유사 개념. Gaussian representation = 우리 quality-aware attention의 대안적 구현.
- **차별점**: RT-GCN은 사전 정의된 Gaussian → 우리는 multi-channel signal statistics에서 동적 추론 가능.
- **Links**: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1566253523003949) | [PDF (TU/e)](https://pure.tue.nl/ws/files/309332629/1-s2.0-S1566253523003949-main.pdf)

### STD-MAE (2024)
- **Venue**: IJCAI 2024
- **Authors**: Haotian Gao, Renhe Jiang, Zheng Dong, Jinliang Deng, Yuxin Ma, Xuan Song
- **Core Idea**: Decoupled spatial+temporal masked autoencoder. Spatial MAE hides entire sensor streams, temporal MAE hides time patches.
- **Noise Handling**: Spatial masking = sensor failure 시뮬레이션. Optimal masking ratio 0.25.
- **우리와의 관계**: 우리 denoising encoder의 대안적 pretrain 방식. 둘 다 encoder pretrain → downstream forecasting.
- **차별점**: STD-MAE는 random masking → 우리는 realistic noise types (gaussian, bias, stuck, drift, dead). Residual design 분석 없음. Spillover 측정 없음.
- **Links**: [arXiv:2312.00516](https://arxiv.org/abs/2312.00516) | [GitHub](https://github.com/Jimmy-7664/STD-MAE)

### STMAE (2023-2024)
- **Venue**: arXiv / Springer
- **Core Idea**: Biased random walk masking (topologically important sensors 우선 masking) + patch temporal masking. Plug-and-play pretrained encoder.
- **Noise Handling**: Hub 센서를 우선적으로 마스킹하여 중요 센서 고장 시나리오 시뮬레이션.
- **우리와의 관계**: 동일한 목표 (plug-in denoising encoder), 다른 접근 (masking vs denoising).
- **Links**: [arXiv:2309.15169](https://arxiv.org/abs/2309.15169)

### DiffSTG (2023)
- **Venue**: ACM SIGSPATIAL 2023
- **Authors**: Wen et al.
- **Core Idea**: DDPM for spatio-temporal graph forecasting. UGnet (U-Net + GNN) as denoising backbone.
- **Noise Handling**: Diffusion의 iterative denoising을 forecasting 메커니즘으로 사용. Probabilistic forecasts with uncertainty.
- **우리와의 관계**: Diffusion-as-forecaster 패러다임. 우리 denoising encoder 개념과 철학적 유사성.
- **Links**: [arXiv:2301.13629](https://arxiv.org/abs/2301.13629) | [ACM](https://dl.acm.org/doi/10.1145/3589132.3625614)

### Reinforced Dynamic Adversarial Training (2023)
- **Venue**: KDD 2023
- **Core Idea**: RL로 동적으로 어떤 노드를 adversarial perturbation할지 선택. Self-knowledge distillation으로 catastrophic forgetting 방지.
- **Noise Handling**: Static adversarial과 달리 RL 기반 노드 선택이 취약한 위치를 자동 발견.
- **우리와의 관계**: 우리 random noise aug의 상위 호환. Dynamic, spatially-aware adversarial approach.
- **Links**: [arXiv:2306.14126](https://arxiv.org/abs/2306.14126)

---

## B. 관련 기법 논문 (Imputation / SSL / Pretrain)

### PriSTI (2023)
- **Venue**: ICDE 2023
- **Core Idea**: Conditional diffusion for spatiotemporal imputation. CFEM extracts global prior from observations, NEM uses prior for denoising.
- **우리와의 관계**: Prior extraction + noise estimation = 우리 encoder + predictor pipeline과 구조적 유사.
- **Links**: [arXiv:2302.09746](https://arxiv.org/abs/2302.09746)

### GRIN (2022)
- **Venue**: ICLR 2022
- **Core Idea**: Bidirectional recurrent GNN for multivariate time series imputation. Healthy sensors → graph propagation → faulty sensor recovery.
- **우리와의 관계**: Graph-based imputation 기반선. 우리 denoising encoder는 이 개념의 learned, end-to-end 버전.
- **Links**: [arXiv:2108.00298](https://arxiv.org/abs/2108.00298) | [GitHub](https://github.com/Graph-Machine-Learning-Group/grin)

### ST-SSL (2023)
- **Venue**: AAAI 2023
- **Core Idea**: Attribute-level + structure-level augmentation with self-supervised auxiliary tasks. Spatial heterogeneity + temporal heterogeneity 고려.
- **우리와의 관계**: Attribute augmentation ≈ noise injection on sensor readings. SSL objective로 noise-invariant representation 학습.
- **Links**: [arXiv:2212.04475](https://arxiv.org/abs/2212.04475)

### MagiNet (2024)
- **Venue**: ACM TKDD 2024
- **Core Idea**: Mask-aware graph imputation. Zero-filling 회피하여 adaptive mask-aware ST encoder 사용.
- **우리와의 관계**: Zero-filling이 "inevitable noise" 도입한다는 발견이 우리 denoising encoder 동기와 일치.
- **Links**: [arXiv:2406.03511](https://arxiv.org/abs/2406.03511) | [ACM](https://dl.acm.org/doi/10.1145/3743141)

### STS-CCL (2024)
- **Venue**: ICASSP 2024
- **Core Idea**: Basic+strong graph augmentation + hard mutual-view contrastive. Graph-level + node-level contrast.
- **우리와의 관계**: Two-level contrastive design. 우리 contrastive pretrain 실패 → 이 방식의 더 정교한 augmentation이 필요했을 수 있음.
- **Links**: [arXiv:2307.02507](https://arxiv.org/abs/2307.02507)

### SCPT (2023)
- **Venue**: ECML-PKDD 2023
- **Authors**: Arian Prabowo et al.
- **Core Idea**: Spatial contrastive pre-training for traffic forecasting on new roads. 2일 local data로 unseen roads 예측.
- **우리와의 관계**: Cold-start generalization ≈ dead/faulty sensor problem. Pre-training strategy가 우리 encoder approach의 선례.
- **Links**: [arXiv:2305.05237](https://arxiv.org/abs/2305.05237) | [GitHub](https://github.com/cruiseresearchgroup/forecasting-on-new-roads)

### GT-TDI (2023)
- **Venue**: arXiv
- **Core Idea**: GNN+Transformer with road network semantic context for large-scale traffic imputation.
- **Links**: [arXiv:2301.11691](https://arxiv.org/abs/2301.11691)

### DEGNN (2024)
- **Venue**: arXiv
- **Core Idea**: Dual experts GNN — edge expert (graph structure denoising) + node feature expert (feature denoising). Self-supervised.
- **우리와의 관계**: Node feature expert = 우리 denoising encoder. Edge expert = graph structure 측면 (미탐색).
- **Links**: [arXiv:2404.09207](https://arxiv.org/html/2404.09207)

### VMGCN (2025)
- **Venue**: arXiv
- **Core Idea**: Variational mode decomposition + learnable soft-thresholding + 3D attention (spatial+temporal+channel) for noise-resilient long-term prediction.
- **우리와의 관계**: Signal decomposition + attention 기반 noise suppression. Architectural noise resilience의 대안적 접근.
- **Links**: [arXiv:2504.06660](https://arxiv.org/abs/2504.06660)

---

## C. General Techniques (Cross-domain, Transferable Ideas)

### DECL — Denoising-Aware Contrastive Learning (2024)
- **Venue**: IJCAI 2024
- **Core Idea**: Positive samples = denoised version of noisy input. Negative = over-noised. Adaptive denoiser selection per sample.
- **Traffic 적용**: Spatial contrastive + adaptive denoiser → noise-invariant representation.
- **Links**: [arXiv:2406.04627](https://arxiv.org/abs/2406.04627)

### RobustTSF (2024)
- **Venue**: ICLR 2024
- **Core Idea**: Anomaly-aware curriculum (variance/trend ratio로 difficulty 정의) + robust loss. Detect-impute-retrain pipeline 회피.
- **Traffic 적용**: Sensor health 기반 curriculum scheduling.
- **Links**: [arXiv:2402.02032](https://arxiv.org/abs/2402.02032)

### MoE Robustness Theory (2025)
- **Venue**: arXiv
- **Core Idea**: Sparse MoE의 routing이 noise filter 역할 (이론적 증명). Corrupted input은 clean input과 다른 expert 활성화.
- **Traffic 적용**: Sensor quality → expert routing signal.
- **Links**: [arXiv:2601.14792](https://arxiv.org/html/2601.14792v1)

### Soft Contrastive Learning for Time Series (2024)
- **Venue**: ICLR 2024
- **Core Idea**: Binary positive/negative 대신 continuous similarity score. Time series의 자연적 temporal correlation 반영.
- **Links**: [ICLR 2024 PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/ccc48eade8845cbc0b44384e8c49889a-Paper-Conference.pdf)

### CSDI (2021)
- **Venue**: NeurIPS 2021
- **Core Idea**: Conditional score-based diffusion for time series imputation. Foundation for DiffSTG, PriSTI 등.
- **Links**: [NeurIPS PDF](https://papers.neurips.cc/paper_files/paper/2021/file/cfe8504bda37b575c70ee1a8276f3486-Paper.pdf)

---

## D. Surveys & Collections

| Resource | Link |
|---|---|
| STG4Traffic: Survey of ST-GNNs | [arXiv:2307.00495](https://arxiv.org/abs/2307.00495) |
| Survey of ST Traffic Data Imputation | [arXiv:2412.04733](https://arxiv.org/html/2412.04733) |
| Awesome TimeSeries ST Diffusion Model | [GitHub](https://github.com/yyysjz1997/Awesome-TimeSeries-SpatioTemporal-Diffusion-Model) |
| GNN4Traffic Collection | [GitHub](https://github.com/jwwthu/GNN4Traffic) |
| Uncertainty in Traffic Forecasting (Survey) | [IEEE TKDE 2023](https://dl.acm.org/doi/abs/10.1109/TKDE.2023.3312261) |

---

## E. 우리 연구의 Positioning

### 기존 연구 대비 차별점

| 측면 | 기존 연구 | 우리 연구 |
|---|---|---|
| Noise type | Random masking (STD-MAE, STMAE) 또는 adversarial (SAFER, KDD'23) | **5가지 realistic failure modes** (gaussian, bias, stuck, drift, dead) |
| Evaluation | 단일 dataset, 단일 noise type | **Cross-dataset** (SB vs CC) + **spatial coupling analysis** |
| Spillover | 측정 안 함 | **Healthy node spillover** 명시적 측정 |
| Encoder design | Random masking pretrain | **Residual denoising** (v1 vs v2 비교, residual의 중요성 분석) |
| Architecture agnostic | 특정 모델에 종속 | **STAEformer + STGCN** 둘 다 검증 |

### 미탐색 Gap (우리가 기여할 수 있는 영역)

1. **Dynamic quality-aware attention**: RT-GCN의 static Gaussian vs 우리 input-dependent quality bias
2. **Input-quality-conditioned uncertainty**: 기존 traffic uncertainty 논문은 미래 불확실성만 다룸, 센서 품질 → 예측 불확실성 연결은 미탐색
3. **Cross-dataset noise robustness analysis**: Spatial coupling (adj weight)과 defense effectiveness의 관계 분석은 우리가 유일
4. **Failure-mode-aware augmentation**: Empirical sensor failure 패턴 기반 augmentation (vs random/adversarial)

---

## F. Attention-Based Denoising with Interpretable Reliability (신규 방향)

*Added: 2026-02-27*

### 배경: 왜 이 방향이 필요한가

#### Spillover Correction 실험에서 얻은 교훈

Output-level SpilloverCorrector (backbone 후단 보정)를 구현하여 실험한 결과:

1. **실패 원인 — 신호 부재**: Backbone의 spatial attention이 이미 노드 간 정보를 섞은 후라, hidden space에서 corrupt/clean 노드의 분리도가 **0.9%**에 불과. 어떤 아키텍처를 붙여도 per-node anomaly 탐지 불가.

2. **Global shift만 학습**: Corrector는 per-node correction이 아닌 **global bias shift**만 학습 (모든 노드 δ ≈ 동일). Gaussian noise처럼 전체 hidden 분포를 바꾸는 noise에만 효과 (healthy MAE 9.9-60.7% 감소), stuck/dead/drift에는 무효 (1-6% 변화).

3. **Input level에서는 신호가 명확**: Stuck = 시계열 불변, Dead = 전 채널 0, Drift = 추세, Gaussian = 분산 증가. Raw signal에서는 noise 패턴이 직접 관찰 가능.

4. **핵심 딜레마 발견**:
   - Input level: noise 패턴은 보이지만 **context 부족** (이 값이 이 시간대에 정상인가?)
   - Hidden level: context는 풍부하지만 **noise 신호 희석** (attention mixing)

#### 기존 Denoising Encoder의 한계

DenoisingEncoder v2 (dilated conv + GCN)는 robustness 70% 개선으로 잘 작동하지만:
- **Black box**: 어떤 노드가 noisy인지, 얼마나 보정했는지 해석 불가
- **Reliability head 추가 실험 실패**: auxiliary output으로만 사용 → 성능 12.05→12.57 악화
- **Linear attention encoder 실험 실패**: conv+GCN을 attention으로 단순 교체 → robustness 약화 (Avg Deg +24.0% vs +22.0%), 특히 bias noise에 취약 (+45.0% vs +24.3%)

#### 이전 attention encoder 실패의 근본 원인

LinearAttentionDenoisingEncoder는 `temporal self-attention → spatial self-attention → correction` 구조였으나:
1. **Anomaly-aware 구조 아님**: 표준 transformer 패턴, noise propagation 메커니즘에 대한 inductive bias 없음
2. **Spatial self-attention**: 모든 노드가 모든 노드를 동등하게 참조 → noisy 노드의 영향이 global하게 전파
3. **Interpretable 중간값 없음**: reliability, anomaly signal 등 해석 가능한 출력 부재

### 새 접근: Attention-Based Denoising with Interpretable Reliability

#### 핵심 아이디어

SpilloverCorrector의 3-stage mechanistic 구조 (reliability → anomaly propagation → correction)를 input level에서 attention 기반으로 구현. 이전 실패를 회피하기 위해:

- **Reliability-gated cross-attention**: 이전 self-attention(모든 노드 동등)과 달리, reliability score로 noisy 노드의 영향을 명시적으로 감쇠
- **Input level 작동**: hidden level의 0.9% 분리도 문제 회피
- **Interpretable 중간값**: per-node reliability r, attention weights, correction δ 모두 해석 가능

#### 관련 검증된 접근

**GDN (Graph Deviation Network, AAAI 2021)** — 가장 직접적 선례
- 4-stage: sensor embedding → graph structure learning → **graph attention forecasting** → deviation scoring
- **Attention weights로 어떤 센서가 이상인지 해석** — 검증된 패턴
- Root cause analysis: attention weight visualization으로 anomaly 원인 추적
- 우리와의 차이: GDN은 anomaly **detection** (사후), 우리는 **correction** (preprocessing)
- [arXiv:2106.06947](https://arxiv.org/abs/2106.06947)

**CG-DGAE (Expert Systems with Applications, 2024)** — Denoising 측면 선례
- Cluster-guided denoising graph autoencoder로 contaminated traffic data → healthy data 복원
- Diffusion GCN + sensor clustering → fault detection 99% accuracy
- 우리와 동일한 목적 (traffic sensor denoising + fault detection)이지만 GCN 기반, attention 해석성 없음
- [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417424023984)

**MTAD-GAT / TopoGDN (2024)** — Temporal+Spatial attention 조합 선례
- Graph attention으로 temporal+spatial 차원 동시 모델링
- Anomaly detection에서 높은 성능
- [arXiv:2408.13082](https://arxiv.org/abs/2408.13082)

#### 제안 아키텍처 개요

```
Input: (B, T, N, C) — raw traffic data
  │
  ├─ Temporal Self-Attention (per-node, T=12 tokens)
  │   → contextual temporal features (B, N, d_model)
  │   → "이 시간대에 이 값이 정상인가?" 파악
  │
  ├─ Stage 1: Reliability Estimation
  │   → r ∈ [0,1] per node (B, N, 1)
  │   → interpretable: 어떤 센서가 신뢰할 수 없는가
  │
  ├─ Stage 2: Reliability-Gated Spatial Cross-Attention
  │   → anomaly_signal = (1-r) * h
  │   → cross_attn(query=h, key=anomaly_signal, value=anomaly_signal)
  │   → noisy 노드의 영향 명시적 감쇠 (이전 self-attention 실패 회피)
  │   → interpretable: attention weights = spatial anomaly propagation 패턴
  │
  ├─ Stage 3: Correction
  │   → δ per node (B, N, C_physical), zero-init
  │   → broadcast to (B, T, N, C_physical)
  │
  └─ Output: input + δ (residual, replace strategy)
```

#### 이전 실패를 반복하지 않기 위한 설계 원칙

| 이전 실패 | 교훈 | 이번 설계의 대응 |
|---|---|---|
| SpilloverCorrector: hidden에서 신호 없음 (0.9%) | 모듈의 입력에 신호가 있는지 먼저 확인 | Input level 작동 — raw signal에서 noise 패턴 직접 관찰 가능 |
| SpilloverCorrector: global shift만 학습 | Loss를 줄이는 shortcut 방지 | Reliability supervision (BCE) 옵션 + denoising objective (node별 reconstruction) |
| LinearAttention: noise가 attention으로 전파 | Spatial self-attention의 global 전파 문제 | **Reliability-gated cross-attention** — r이 낮은 노드의 key/value를 감쇠 |
| LinearAttention: interpretable하지 않음 | Attention만으로는 부족, 구조적 해석성 필요 | 3-stage 구조: r(reliability) + attention weights + δ(correction) 모두 출력 |
| Reliability head: auxiliary output만 → 효과 없음 | Reliability가 구조적으로 영향을 줘야 함 | r이 cross-attention의 anomaly signal을 직접 제어 |
| Training 파이프라인 버그 (scale 불일치 등) | Normalization/scale 상호작용 주의 | 기존 DenoisingEncoder의 검증된 pretrain pipeline 재사용 |

#### 예상 기여

1. **성능**: Denoising v2 + aug (Avg Deg +12.6%) 대비 개선 또는 동등
2. **해석 가능성**: per-node reliability r → 센서 고장 자동 탐지 (실용적 가치)
3. **Attention weight 분석**: spatial anomaly propagation 패턴 시각화 (GDN 스타일)
4. **구조적 차별화**: reliability-gated cross-attention = 기존 denoising encoder의 black-box GCN과 본질적으로 다른 메커니즘

#### 불확실성 및 리스크

- **Temporal self-attention이 context를 충분히 잡는가?**: T=12 tokens은 짧음. tod/dow 정보를 임베딩으로 포함하면 도움될 수 있음.
- **Reliability가 실제로 의미 있는 값을 학습하는가?**: Input level에서는 신호가 있지만, reconstruction loss만으로 r이 자연스럽게 학습되는지는 실험 필요.
- **성능이 conv+GCN보다 나을 근거가 약함**: Attention의 장점은 해석 가능성이지, denoising 성능은 conv+GCN이 더 적합한 inductive bias를 가질 수 있음. 성능 동등 + 해석 가능성 추가가 현실적 목표.
