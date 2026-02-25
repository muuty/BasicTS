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
