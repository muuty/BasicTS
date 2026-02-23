# Project Background: Incident-Aware Traffic Forecasting

## 1. 연구 목표

교통 예측 모델이 **사고(incident) 상황에서 성능이 크게 저하**되는 문제를 해결하는 것이 핵심 목표이다.

- 전체 평균 MAE는 양호하나, node/timestamp 간 성능 편차가 큼
- 사고 발생 시 특정 노드의 flow가 0으로 급감하지만, 모델은 평소와 유사한 값을 예측
- Message passing에서 주변 정상 노드 정보가 사고 노드의 이상 signal을 덮어씀
- **목표**: overall MAE를 유지하면서 incident 상황의 worst-case 성능 개선

## 2. 프레임워크: BasicTS

[BasicTS](https://github.com/zezhishao/BasicTS)는 시공간 예측(Spatio-Temporal Forecasting) 벤치마크 라이브러리이다. EasyTorch 기반으로 60+개 모델을 지원하며, config 파일로 실험을 제어한다.

### 주요 모델 (본 프로젝트에서 사용)
| 모델 | 유형 | 특징 |
|------|------|------|
| **STGCN** | GCN + Temporal Conv | 가벼움, graph structure에 민감 |
| **STAEformer** | Transformer | 안정적, 높은 baseline 성능 |
| **AGCRN** | Adaptive GCN + GRU | 적응형 graph learning |
| **STGformer** | Spatial-Temporal Transformer | Transformer 기반 |
| **GWNet** | Graph WaveNet | Dilated causal conv + adaptive graph |
| **MTGNN** | Multi-graph + Temporal | Mix-hop propagation |
| **DGCRN** | Dynamic GCN + GRU | Dynamic graph generation |

### 데이터셋
| 데이터셋 | 노드 수 | 출처 | 비고 |
|----------|---------|------|------|
| **SAN_BERNARDINO** | 893 | XTraffic (California) | 주요 실험 데이터셋 |
| **CONTRA_COSTA** | 773 | XTraffic (California) | 보조 데이터셋 |
| **ALAMEDA** | - | XTraffic (California) | 보조 데이터셋 |
| **SACRAMENTO** | - | XTraffic (California) | 보조 데이터셋 |

- 3개월 데이터 사용: `data_range = (0, 24192)` (5분 간격)
- Incident metadata CSV 제공 (사고 유형, 시간, 노드 정보)
- 데이터는 `datasets/` 디렉토리에 symlink로 연결

---

## 3. 실험 히스토리 및 접근법

프로젝트는 여러 접근법을 순차적으로 실험해왔다.

### 3.1 Adjacency Matrix 실험
**목적**: 그래프 구조(인접 행렬)가 예측 성능에 미치는 영향 분석

- 4가지 인접 행렬 방법: distance, distance_sparse, distance_dense, identity
- 3개 모델 × 2개 데이터셋 × 3개 seed = 총 72 실험
- 결과: `experiments/config/adjacency_phase1a_*.yaml`

### 3.2 Node Identity / Weakened Message Passing 실험
**목적**: Message passing이 사고 signal을 희석시키는지 검증

| 변형 | MAE | Inc MAE | 설명 |
|------|-----|---------|------|
| STGCN (baseline) | 14.23 | 13.56 | 표준 ChebGraphConv |
| WeakenedMP | 14.70 | 16.20 | Message passing 약화 |
| NodeIdentity | 17.18 | 19.15 | Identity matrix (MP 제거) |

**결론**: Message passing 제거/약화는 전반적 성능 저하를 초래함

### 3.3 Experience Replay 실험
**목적**: 어려운 샘플을 반복 학습하여 incident 성능 개선

- **구현**: `experience_replay/` (FIFO buffer, MAE 기반 가중 샘플링)
- **전략**: Difficult, Representative (K-Medoids), Random, Adaptive
- **Gating module**: `gating/gate.py` - 학습 가능한 비대칭 인접 행렬 생성

| 변형 | 설명 |
|------|------|
| ONLY_GATE | Gating만 적용 |
| ONLY_REPLAY_001 / 0001 | Replay만 적용 (capacity ratio 0.01 / 0.001) |
| REPLAY_FULL_UPDATE | Gate + Replay, 전체 모델 업데이트 |
| REPLAY_GATE_UPDATE | Gate + Replay, gate만 업데이트 |

**핵심 결과**:
- STGCN에서 replay 매우 효과적: MAE 17.91 → 13.84 (-22.7%)
- STAEformer/STGformer에서는 효과 없음 (baseline 이미 우수)
- Random sampling > Representative sampling
- Gate + Replay 조합은 추가 개선 미미

### 3.4 Contrastive Learning 실험
**목적**: Anti-smoothing loss로 사고 노드의 representation을 보존

- `contrastive/contrastive_loss.py`: Euclidean, Cosine, Prediction-Guided loss
- `contrastive/augmentation.py`: Edge dropout을 통한 GSO augmentation
- **결과**: 사고 노드 개선 시도 → overall MAE 1~10% 악화 (공통 패턴)

### 3.5 Severity-aware Pre-training (설계 단계)
**목적**: Self-supervised pre-training으로 anomaly-aware representation 학습

- `docs/design.md`에 상세 설계 문서
- 2-stage: Contrastive pre-training → Prediction fine-tuning
- Traffic-specific anomaly injection으로 synthetic negative sample 생성
- `baselines/Encoder/` 에 ContextAwareSTEncoder 구현
- 현재 테스트 단계 (pretrain_test.yaml)

---

## 4. Coreset Selection (핵심 실험)

### 4.1 개요
학습 데이터에서 대표적인 부분집합(coreset)을 선택하여 효율적으로 모델을 학습시키는 실험이다.

### 4.2 Selection 방법 (10가지)

**Similarity-based (거리 의존)**:
| 방법 | 파일 | 알고리즘 |
|------|------|----------|
| K-Medoids | `coreset/k_medoids.py` | FasterPAM (GPU) |
| K-Center Greedy | `coreset/k_center.py` | 최대 거리 최소화 |
| Herding | `coreset/herding.py` | Mean matching |
| Facility Location | `coreset/facility_location.py` | 최대 커버리지 (1-1/e 보장) |
| Graph Cut | `coreset/graph_cut.py` | Cross-partition similarity 최대화 |
| Total Similarity | `coreset/total_similarity.py` | Facility location 변형 |

**Temporal (거리 비의존)**:
| 방법 | 파일 | 알고리즘 |
|------|------|----------|
| Recent | `coreset/recent.py` | 최근 n개 샘플 |
| Stride | `coreset/stride.py` | 균등 간격 샘플링 |

**Baseline**:
| 방법 | 파일 | 알고리즘 |
|------|------|----------|
| Random | `coreset/random.py` | 균일 랜덤 |
| Full (default) | `coreset/factory.py` | 전체 데이터 |

### 4.3 Distance Metrics (8가지)
`coreset/distance.py`에서 정의:

| Pipeline | Metric | Feature |
|----------|--------|---------|
| L2 | euclidean | Flattened [T×N×F] |
| L2 | temporal | Mean-over-nodes [T, F] |
| L2 | spatial | Mean-over-time [N, F] |
| L2 | combined | Normalized temporal + spatial |
| Cosine | cosine_raw | Flattened features |
| Cosine | cosine_temporal | Temporal features |
| Cosine | cosine_spatial | Spatial features |
| Cosine | cosine_combined | Temporal + spatial |

### 4.4 파이프라인

```
1. Offline Selection (experiments/select_coreset.py)
   Dataset → Feature 추출 → Distance 계산 → Method 적용 → Index 저장
   → coreset_indices/{dataset}/{method}_{distance}_{ratio}_seed{seed}.json

2. Quality Evaluation (experiments/compute_proxy_metrics.py)
   Index → OT Divergence, FL Objective, Redundancy, Temporal Entropy 계산
   → proxy_metrics.json

3. Training Integration (basicts/runners/base_tsf_runner.py)
   Config의 CORESET 파라미터 → Index 로드 → Subset(dataset, indices) → 학습

4. Results Collection (experiments/get_results.py)
   Checkpoints → CSV (MAE, RMSE, MAPE + incident/non-incident split)
```

### 4.5 Phase A: Distance Screening 실험
**목적**: Coreset selection에 최적인 distance function 찾기

- **Grid**: 4 representation × 2 distance metric × 6 selection method × 2 ratio × 2 seed × 2 model
- **총 220 training runs**
- Config: `experiments/config/phase_a_distance_screening.yaml`
- 체크리스트: `docs/phase_a_experiment_checklist.md`

**초기 결과** (SAN_BERNARDINO, STGCN):
- K-Center + combined distance @ ratio=0.7이 가장 안정적 (MAE ~14.0-14.6)
- Ratio 0.3에서는 성능 편차가 크고, 0.7에서 안정적
- Graph Cut은 seed에 민감 (MAE 14.7~18.7)
- K-Medoids는 중간 수준의 성능

### 4.6 이전 Coreset 실험 결과
**K-Medoids 중심 실험** (coreset.yaml):
- 4개 모델 × 10개 ratio (0.1~1.0) × SAN_BERNARDINO

| 모델 | 최적 Ratio | MAE |
|------|-----------|-----|
| STAEformer | 0.8 | 11.85 |
| STGformer | 0.9 | 12.00 |
| AGCRN | 0.7 | 13.21 |
| STGCN | 0.6 | 14.20 |

**핵심 발견**: 전체 데이터(ratio=1.0)보다 60~80% coreset이 오히려 성능이 좋거나 비슷함

### 4.7 Proxy Metrics
학습 없이 coreset 품질을 평가하는 지표:
- **Sinkhorn Divergence**: Debiased OT distance (분포 유사성)
- **Facility Location Objective**: 커버리지 측정
- **Intra-coreset Redundancy**: 내부 유사도
- **Information Gain**: FL - λ×Redundancy
- **Temporal Diversity**: H_tod (시간대 엔트로피), H_dow (요일 엔트로피)

---

## 5. 인프라

### 실험 실행
```bash
# 단일 실험
python experiments/train.py -c baselines/STGCN/SAN_BERNARDINO/SAN_BERNARDINO.py -g 0

# YAML config 기반 배치 실행 (SLURM)
python experiments/run_experiments.py --cfg experiments/config/coreset.yaml --arch=cuda --batch-size 5

# Coreset index 사전 계산
python experiments/select_coreset.py --cfg experiments/config/coreset.yaml --gpus 0

# 결과 수집
python experiments/get_results.py --config experiments/config/coreset.yaml --metrics MAE RMSE
```

### SLURM 설정
- Partition: `gpu_cuda` (1 GPU, 32GB 메모리)
- Script: `experiments/scripts/submit_job_cuda.sh`
- 환경: `conda activate cuda`

### 디렉토리 구조
```
BasicTS/
├── baselines/          # 모델 구현 + config 파일
├── basicts/            # Core 프레임워크 (runners, data, metrics)
├── coreset/            # Coreset selection 알고리즘
├── coreset_indices/    # 사전 계산된 index 파일
├── contrastive/        # Contrastive learning 모듈
├── experience_replay/  # Experience replay 모듈
├── gating/             # Dynamic graph gating 모듈
├── experiments/        # 실험 실행/설정/결과
│   ├── config/         # YAML 실험 설정
│   ├── result/         # 결과 CSV/이미지
│   └── scripts/        # SLURM 스크립트
├── datasets/           # 데이터 (symlinks → xtraffic/)
├── docs/               # 문서
├── project/            # 부가 실험 코드
└── scripts/            # 분석/시각화 스크립트
```

---

## 6. 주요 참고 논문

- **CLEAR** (TKDE 2025): Contrastive spatio-temporal representation learning
- **STD-MAE** (IJCAI 2024): Spatio-temporal masked autoencoder
- **SCPT** (DMKD 2023): Spatial contrastive pre-training
- **DeepCore** (Guo et al., 2022): Coreset selection 벤치마크
- **DCDetector** (KDD 2023): Anomaly detection with contrastive learning
