# Experiment Plan: Comprehensive Evaluation Framework for Coreset Selection in Spatio-Temporal Traffic Prediction

## 1. Paper Positioning

**Conference → Journal 확장 전략**: 기존 IEEE-ITS 컨퍼런스 논문(STGCN 1개 모델, 8개 방법, 2개 데이터셋 벤치마크)을 T-ITS 저널 수준으로 확장.

**핵심 전환**: "어떤 방법이 좋은가" → **"왜 좋고, 왜 안 좋은가"**에 대한 deep analysis

**핵심 기여 3개**:
1. **ST-aware distance function + evaluation framework**: 기존 coreset selection이 Euclidean distance로 시공간 구조를 무시하는 문제를 지적하고, temporal/spatial distance를 분리·결합하는 distance function을 제안. 이 distance 위에서 proxy metric(Wasserstein, FL, redundancy, information gain)을 평가
2. **ST 특화 인사이트**: inductive bias별 최적 ratio 차이, temporal/spatial distance의 상대적 중요도, proxy metric과 성능의 관계
3. **실무 가이드라인**: 모델 교체 시 coreset 재사용 가능 범위, noise 환경에서의 robustness, 최적 selection ratio 가이드

**Novelty 근거**:
- Image 분야에서는 pre-trained representation space에서의 coreset selection이 효과적이나, ST traffic 데이터는 (1) 원본 차원이 이미 낮고 (2) spatial structure가 feature가 아닌 graph에 인코딩되어 있어서 동일한 접근이 적용되지 않음 → raw feature space에서 ST-aware distance를 정의하는 것이 핵심
- Distance function 자체가 ST 도메인 특화 contribution (방법론 제안이 아닌 evaluation methodology의 novelty)

---

## 2. Experimental Setup

### 2.1 Data Structure
- 하나의 sample = `[T_in, N, F]` (e.g., `[12, 307, 3]`)
- Coreset selection은 **시간축에서** sample을 선택하는 문제 (노드는 항상 전체 포함)
- Features: flow, speed, occupancy (3개)
- T_in = 12 (1시간, 5분 간격)

### 2.2 Datasets (xtraffic)

| Dataset | Nodes | Samples (approx.) | Period | Role |
|---------|-------|--------------------|--------|------|
| SAN_BERNARDINO | 893 | ~14,500 (3개월) | 2023 | **Primary** (모든 분석) |
| ALAMEDA | 897 | ~14,500 (3개월) | 2023 | Secondary (검증) |
| SACRAMENTO | 836 | ~14,500 (3개월) | 2023 | Secondary (검증) |

**xtraffic 선택 이유** (PEMS 대비):
- 센서 메타데이터 (위치, 도로 정보) → 공간 분석 가능
- 사고 라벨 (incident metadata) → incident-aware evaluation 축 추가 가능
- 3년치 데이터 → 장기 패턴 분석 가능 (현재는 3개월로 제한)
- 3개 county가 유사한 규모 (800~900 노드)로 비교 적합

### 2.3 Models

**Primary 모델 (4개)**

| Model | Type | Inductive Bias |
|-------|------|----------------|
| STGCN | GNN (ChebConv + Conv1D) | 강함 |
| AGCRN | GNN (Adaptive Graph + GRU) | 강함 |
| STAEformer | Transformer | 약함 |
| STGformer | Transformer (학습 시간 짧음) | 약함 |

**Secondary 모델 (3개 — Cross-model transferability에서만)**

| Model | Type | Inductive Bias |
|-------|------|----------------|
| DCRNN | GNN (Diffusion Conv + GRU) | 강함 |
| Graph WaveNet | GNN (Adaptive Adj + Dilated Conv) | 강함 |
| ASTGCN | GNN + Attention | 중간 |

### 2.4 Coreset Selection Methods

**Category 1: Coverage Maximization (min-max)**
- k-Center Greedy — 최악 거리를 최소화

**Category 2: Representativeness Maximization (facility location)**
- k-Medoids — swap 기반 facility location 최적화
- GraphCut (**신규**) — submodular facility location + redundancy 패널티

**Category 3: Distribution Sequential Matching**
- Herding — greedy moment matching

**Category 4: Baseline**
- Random Sampling
- Stratified Sampling (시간 기반)

*(기존 논문의 나머지 방법들도 포함하되, 분석은 카테고리 대표 방법 중심)*

### 2.5 Selection Ratios
- **5개**: 10%, 30%, 50%, 70%, 90%

### 2.6 Distance Functions

**ST-aware distance 분해**:

```
d(xᵢ, xⱼ) = α · d_temporal(xᵢ, xⱼ) + β · d_spatial(xᵢ, xⱼ)
```

| Distance | 정의 | 설명 |
|----------|------|------|
| **Euclidean** | L2 on flattened [T×N×F] | 현재 기본값, 시공간 구조 무시 |
| **Temporal-only** | L2 on mean-over-nodes [T, F] | 시간 변화 패턴 유사도 |
| **Spatial-only** | L2 on mean-over-time [N, F] | 공간 분포 유사도 (Graph Laplacian 가중치 옵션) |
| **Combined** | α·temporal + β·spatial | 시공간 결합 (α, β는 Phase A에서 탐색) |

### 2.7 Implementation Note: Batch + GPU 기반 Coreset Selection

**필수 요구사항**: 모든 coreset selection의 distance 계산 및 selection 과정은 **batch 처리 + GPU 가속**으로 구현해야 함.

- **Distance matrix 계산**: pairwise distance를 CPU에서 for-loop으로 계산하면 25,000 × 25,000에서 병목 발생. PyTorch의 `torch.cdist` 등을 활용하여 GPU에서 batch로 계산
- **Selection 알고리즘**: k-Medoids swap, k-Center greedy, GraphCut의 submodular greedy 등도 가능한 한 vectorized + GPU 연산으로 구현
- **메모리 관리**: full distance matrix가 GPU 메모리에 안 들어갈 경우 mini-batch chunking으로 분할 계산
- **이유**: Phase A~D 전체에서 distance 계산이 반복적으로 수행되므로 (4 distances × 9 methods × 5 ratios), selection 속도가 전체 실험 일정의 병목이 될 수 있음. 특히 ST-aware distance는 temporal/spatial 각각 계산 후 합산하므로 Euclidean 대비 연산량이 증가함

---

## 3. ST-aware Proxy Metrics

### 3.1 후보 지표 (6개)

세 가지 관점에서 coreset 품질을 측정:
- **분포적(distributional)**: OT 기반 — coreset과 원본의 분포 차이
- **조합적(combinatorial)**: 유사도 기반 — 커버리지, 중복도, 순 정보량
- **시간적(temporal)**: 선택된 샘플의 시간 다양성

| # | Metric | 관점 | 수식 | 방향 |
|---|--------|------|------|------|
| 1 | **OT Cost** | distributional | OT_ε(P,Q) | ↓ 낮을수록 좋음 |
| 2 | **Sinkhorn Divergence** | distributional | OT_ε(P,Q) − 0.5·OT_ε(P,P) − 0.5·OT_ε(Q,Q) | ↓ 낮을수록 좋음 |
| 3 | **Facility Location Obj.** | combinatorial | Σᵢ maxⱼ∈S sim(i,j) | ↑ 높을수록 좋음 |
| 4 | **Intra-coreset Redundancy** | combinatorial | Σᵢ,ⱼ∈S sim(i,j) | ↓ 낮을수록 좋음 |
| 5 | **Information Gain** | combinatorial | FL(S) − λ·Redundancy(S) | ↑ 높을수록 좋음 |
| 6 | **Temporal Diversity** | temporal | H_tod(S), H_dow(S) (normalized entropy) | ↑ 높을수록 좋음 |

**설계 근거**:

- **#1 OT Cost**: 정규화된 최적 수송 비용 (vanilla Sinkhorn). Coreset에서 원본으로의 raw transport cost. Entropic bias가 포함되어 있어 ε 값에 민감할 수 있음.
- **#2 Sinkhorn Divergence**: OT Cost에서 자기수송 비용(OT(P,P), OT(Q,Q))을 제거한 debiased 버전 (Feydy et al., NeurIPS 2019). SD(P,P)=0이 보장되는 proper divergence. ε→0이면 Wasserstein, ε→∞이면 MMD로 수렴. **#1 vs #2 비교로 debiasing의 실질적 효과를 측정**.
- **#3 FL Objective**: 각 데이터 포인트가 coreset 내 가장 가까운 대표점과 얼마나 유사한지의 합. 높을수록 좋음 (coreset이 전체를 잘 대표).
- **#4 Redundancy**: coreset 내부 점들 간 유사도 합. 높을수록 나쁨 (선택된 점들이 서로 비슷함 = 중복).
- **#5 Information Gain = #3 − λ·#4**: GraphCut objective와 동일 구조. FL vs IG 비교를 통해 redundancy penalty의 가치를 ablation 가능.
- **#6 Temporal Diversity**: 선택된 샘플이 시간적으로 얼마나 다양한지 측정. 두 가지 축으로 분해:
  - **H_tod** (time-of-day): 24개 bin (hour), normalized entropy. 모든 시간대를 균등하게 커버하는가?
  - **H_dow** (day-of-week): 7개 bin, normalized entropy. 모든 요일을 균등하게 커버하는가?
  - H = 1.0이면 완벽히 균등, H → 0이면 특정 시간/요일에 집중.
  - Traffic 도메인에서 tod/dow는 자연스러운 분할 (rush hour, weekend 등).
  - `recent` 방법은 H_dow ≈ 0 (마지막 며칠에 집중), `stride`는 H ≈ 1.0.
  - 계산 비용 ≈ 0 (index만 보면 됨). MAE 예측력은 약하겠지만 분석 도구로 유용.

**교차 비교 포인트**:

1. **OT Cost vs Sinkhorn Divergence (#1 vs #2)**: debiasing의 효과. 같은 ratio 내에서 ranking이 바뀌는가?
2. **SD vs IG (#2 vs #5)**: 둘 다 "cross-term − self-term" 구조이지만 다른 공간에서 동작 (distributional vs combinatorial). 모델 유형에 따라 어느 관점이 MAE를 더 잘 예측하는가?
3. **FL vs IG (#3 vs #5)**: redundancy ablation. IG가 FL보다 나은 proxy → GraphCut이 k-Medoids보다 나은 이유 설명.
4. **Temporal Diversity vs 성능**: H_tod/H_dow가 낮은데 성능이 좋다면 → 최신 데이터가 중요 (recency bias). H가 높은데 성능이 좋다면 → 전체 패턴 커버가 중요.

**핵심**: #1~#5 지표는 distance function 위에서 계산됨 → distance를 ST-aware로 바꾸면 proxy metric이 ST-specific이 됨. #6은 distance-independent.

**구현 상태**:
- ✅ OT Cost: `coreset/ot_distance.py` — `_sinkhorn_cost()` / `compute_distance()`
- ✅ Sinkhorn Divergence: `coreset/ot_distance.py` — `sinkhorn_divergence()` / `compute_divergence()`
- ⬜ FL Objective, Redundancy, Information Gain
- ⬜ Temporal Diversity (H_tod, H_dow)

### 3.2 분석 방법
- 각 (method × ratio) 조합에서 5개 proxy metric + 실제 prediction MAE 계산
- **Proxy vs MAE의 Spearman correlation** 측정 → 어떤 proxy가 성능을 가장 잘 예측하는가
- **모델별 correlation 차이** 확인 (GNN vs Transformer)
- **Distance별 correlation 차이** 확인 (Euclidean vs ST-aware)

### 3.3 기대 인사이트

**OT Cost vs Sinkhorn Divergence** (debiasing 효과):
- SD가 더 나은 proxy → self-transport 제거가 유의미 → proper divergence의 가치
- 차이 미미 → debiasing 불필요, vanilla OT로 충분

**SD vs IG** (distributional vs combinatorial):
- SD가 더 좋은 proxy → 분포 전체의 매칭이 중요 (global influence)
- IG가 더 좋은 proxy → 개별 커버리지 − 중복이 핵심 (local influence)
- GNN(local message passing) vs Transformer(global attention)에서 다를 수 있음

**FL vs Information Gain** (redundancy ablation):
- IG > FL → redundancy 최소화가 추가 가치 → GraphCut이 k-Medoids보다 나은 이유 설명
- FL ≈ IG → redundancy 영향 미미

**Euclidean vs ST-aware distance에서의 proxy correlation**:
- ST-aware distance로 계산한 proxy가 더 높은 correlation → distance 정의가 중요
- 차이 없음 → Euclidean이면 충분 (이것도 practical guideline)

---

## 4. Toy Example Analysis

### 4.1 설계
- PEMS04 전체 ~25,000 샘플에서 **greedy하게 10개만** 선택하는 과정 추적
- **3개 대표 방법**: k-Center vs k-Medoids vs Herding
- Random은 reference baseline

### 4.2 시각화
- **3×10 grid**: 가로 = 3개 방법, 세로 = selection 순서 1~10
- 각 칸에 해당 sample의 traffic temporal profile 또는 heatmap
- 각 step에서의 marginal proxy metric 변화 curve

### 4.3 분석 포인트
- 각 방법이 어떤 traffic pattern을 먼저 고르는가 (rush hour? night? transition?)
- Saturation point: 몇 개부터 marginal gain이 거의 없어지는가
- 카테고리 분류 정당화: 같은 카테고리 내 방법들의 behavior가 유사한지

---

## 5. Evaluation Axes

### Axis 1: Prediction Performance (Decomposed)
**목적**: 전체 성능 + 세부 조건별 분해

- 전체 MAE / RMSE / MAPE
- **시간대별**: rush hour (7-9AM, 5-7PM) / off-peak / night (11PM-5AM)
- **Horizon별**: 15분 / 30분 / 60분
- **노드 특성별**: high-degree hub vs leaf nodes

### Axis 2: Scaling Curve
**목적**: ratio별 성능 변화 + critical point 식별

- 5개 ratio에서 성능 curve (Beyond Neural Scaling Laws 스타일)
- 모델별 optimal ratio 차이 분석 → inductive bias와 연결

### Axis 3: Noise Robustness (test-time, 추가 training 없음)
**목적**: coreset 모델의 noise 환경 내성

- **Test-time evaluation**: Phase B에서 학습된 모델 재활용 → noisy test data로 평가
- Noise types: sensor dropout, Gaussian noise, temporal corruption
- 필요시 train-time robustness 추가 (별도 training 필요)

### Axis 4: Cross-model Transferability
**목적**: coreset의 모델 간 재사용 가능성

- Phase B의 coreset을 secondary 모델 3개에 추가 학습
- 총 7개 모델 간 성능 rank correlation (Spearman)

---

## 6. Key Research Questions

### RQ1: ST-aware distance가 coreset selection에 유의미한 차이를 만드는가?
- **Phase A에서 답변**
- temporal vs spatial vs combined 중 어떤 distance가 효과적인가
- 모델 유형(GNN/Transformer)에 따라 최적 distance가 다른가

### RQ2: 학습 전 coreset 품질을 예측할 수 있는가?
- 4개 proxy metric 중 어떤 게 prediction MAE와 가장 correlate하는가
- correlation 있으면 practical tool, 없으면 open problem

### RQ3: 모델의 inductive bias가 최적 coreset 전략에 미치는 영향
- Transformer는 90%에서도 성능 유지 vs GNN은 40-70%에서 optimal
- inductive bias 강도 → minimum viable data threshold

### RQ4: Coreset의 cross-model transferability
- 모델 바꿀 때 coreset 재사용 가능한가
- optimal ratio는 모델에 따라 재조정 필요한가

---

## 7. Experiment Phases & Run Budget

### Phase A: Distance Screening (소규모 탐색)

**목적**: 4개 distance 중 best 1~2개 선택 (lock)

| 항목 | 설정 |
|------|------|
| Methods | k-Center, k-Medoids, GraphCut (distance 의존 3개만) |
| Distances | Euclidean, temporal, spatial, combined |
| Dataset | SAN_BERNARDINO |
| Models | STGCN, STGformer (GNN 1 + Transformer 1) |
| Ratios | 30%, 70% |
| Seeds | 2 |

**Runs**: 3 methods × 4 distances × 2 models × 2 ratios × 2 seeds = **96 runs**

**선택 기준**:
- 평균 rank 개선 (Euclidean 대비)
- Selection/computation overhead (+20% 이내)
- Proxy-MAE correlation 안정성
- **Method 간 일관성**: distance 순위가 3개 method에 걸쳐 일관적인가

**Phase A 결과 → Lock**: best distance 1개 (필요시 +Euclidean baseline = 2개)

---

### Phase B: Main Benchmark (본실험)

**목적**: 논문 본문 핵심 결과 생성

| 항목 | 설정 |
|------|------|
| Methods | 전체 (9개) |
| Distances | **Locked best distance + Euclidean** (2개) |
| Dataset | SAN_BERNARDINO |
| Models | Primary 4개 (STGCN, AGCRN, STAEformer, STGformer) |
| Ratios | 10%, 30%, 50%, 70%, 90% |
| Seeds | 3 |

**Runs**: 9 methods × 2 distances × 4 models × 5 ratios × 3 seeds = **1,080 runs**

**이 Phase에서 생성되는 결과**:
- Axis 1: Decomposed prediction performance (시간대별/horizon별/노드별 분해)
- Axis 2: Scaling curve (5-point)
- Proxy metric analysis: 4 metrics × 2 distances의 MAE correlation
- Toy example: 3 methods의 greedy selection 추적 + 시각화

**추가 training 없는 분석** (Phase B 모델 재활용):
- Axis 3: Noise robustness (test-time evaluation only)

---

### Phase C: Verification (축소 검증)

**목적**: Phase B 발견의 데이터셋 일반성 확인

| 항목 | 설정 |
|------|------|
| Methods | 상위 3~4개 + Random baseline |
| Distance | **Locked best distance만** (1개) |
| Datasets | ALAMEDA, SACRAMENTO |
| Models | STGCN, STGformer (2개) |
| Ratios | 30%, 50%, 70% |
| Seeds | 3 |

**Runs**: ~4 methods × 1 distance × 2 models × 3 ratios × 3 seeds × 2 datasets = **~144 runs**

**확인 사항**: "PEMS04에서 발견한 패턴이 재현되는가?" (full re-benchmark 아님)

---

### Phase D: Transfer & Robustness (보완)

**목적**: Cross-model transferability + 추가 robustness 분석

**Transferability**:
- Phase B의 coreset → secondary 모델 3개에 학습
- 상위 method 4개, ratio 3개 (30%, 50%, 70%)

| 항목 | 설정 |
|------|------|
| Methods | 상위 4개 |
| Distance | Locked best |
| Models | DCRNN, Graph WaveNet, ASTGCN (secondary 3개) |
| Ratios | 30%, 50%, 70% |
| Seeds | 3 |

**Runs**: 4 methods × 3 models × 3 ratios × 3 seeds = **108 runs**

**Noise Robustness (train-time, optional)**:
- 필요 시에만: noisy data로 coreset selection + 학습
- 대표 method 3개 × noise 2종 × 모델 2개 = ~36 runs

---

### Total Run Budget Summary

| Phase | 목적 | Runs | 누적 |
|-------|------|------|------|
| **A** | Distance screening | 96 | 96 |
| **B** | Main benchmark | 1,080 | 1,176 |
| **C** | Dataset verification | ~144 | ~1,320 |
| **D** | Transfer + robustness | ~108 | **~1,428** |
| (D opt.) | Train-time noise | ~36 | ~1,464 |

**기존 계획 대비 ~25% 절감** (1,971 → 1,428), Phase A의 early lock이 핵심

---

## 8. Paper Structure (Draft)

1. **Introduction** — motivation + positioning (image와의 차이, ST-aware distance 필요성)
2. **Related Work** — traffic prediction, coreset selection, evaluation in image domain
3. **Preliminaries** — problem formulation, method categorization (3 categories), distance function 정의
4. **ST-aware Evaluation Framework**
   - Distance function 분해 (temporal / spatial / combined)
   - Proxy metrics 정의 (Wasserstein, FL, redundancy, information gain)
5. **Toy Example** — motivating illustration (3 methods × 10 samples)
6. **Experiments**
   - Phase A: Distance screening 결과
   - Phase B: Main benchmark (Axis 1-2 + proxy analysis)
   - Phase C: Verification
   - Phase D: Transferability + robustness
7. **Discussion** — RQ1-4 종합 답변 + practical guidelines
8. **Conclusion & Future Work**

---

## 9. Key References

- **Beyond Neural Scaling Laws** (NeurIPS 2022) — scaling curve 분석 영감
- **Coverage-centric Coreset Selection (CCS)** (ICLR 2023) — coverage가 failure mode를 설명
- **InfoMax** (ICLR 2025) — informativeness + redundancy 분해
- **D3** (IJCAI 2025) — diversity, difficulty, dependability 프레임워크
- **Robust Data Pruning** (2024) — noise robustness 평가
- **ELFS** (ICLR 2025) — cross-architecture transferability
- **OpenCity** (ACM TIST 2024) — traffic foundation model (representation 비교 논의용)