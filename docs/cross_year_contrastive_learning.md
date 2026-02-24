# Cross-Year Contrastive Learning for Drift-Robust Traffic Forecasting

## 1. Motivation

### 1.1 Problem
교통 데이터는 기상 재해, 도로 공사, 이벤트 등으로 인해 **sudden distribution shift**가 발생할 수 있다.
이러한 이상 기간이 **training data에 포함**되면, 모델이 이상 패턴에 과적합하거나, 정상/이상 데이터가 혼재하여 학습이 혼란될 수 있다.

### 1.2 Key Observation
SAN_BERNARDINO Q1 데이터 (2022-2024)에서 발견한 핵심 관찰:

| 지표 | 2022-2023 | 2022-2024 | 2023-2024 |
|------|-----------|-----------|-----------|
| Daily profile correlation (r) | 0.894 | 0.981 | 0.894 |
| Flow scale (평균) | 165.75 | 169.43 | 151.82→169.43 |
| Cross-year MAE 증가율 | +125% | +29% | +143% |

**핵심 발견**: 2023년 대기강 이벤트(flow -14~17%)에도 불구하고, **일일 교통 패턴(daily profile shape)의 상관관계는 r=0.89로 매우 높다.** 즉:

> **Scale은 변하지만, Pattern(시간적 변동 형태)은 보존된다.**

이는 교통 예측에서의 concept drift가 주로 **scale shift**이며, **temporal pattern**은 상대적으로 invariant함을 의미한다.

### 1.3 Insight
기존 STAEformer는 raw flow 값을 직접 입력으로 받아 학습하므로, scale shift에 취약하다.
만약 모델이 **scale-invariant한 representation**을 학습할 수 있다면, training data에 이상 기간이 포함되어도 robust한 예측이 가능할 것이다.

**Cross-Year Contrastive Learning**: "같은 장소, 같은 시간 맥락(ToD/DoW), 다른 연도"의 데이터를 positive pair로 사용하여, 연도에 불변한 representation을 학습한다. 서로 다른 연도의 데이터는 **자연스러운 augmentation**이 된다.

---

## 2. Story

### 2.1 Problem Statement
Spatio-temporal traffic forecasting 모델의 실제 배포에서, training data에 이상 기간(extreme weather, major incidents)이 포함되는 것은 불가피하다. 기존 방법론들은:
- 이상 데이터를 **제거**하거나 (데이터 낭비, 이상 기간 정의 어려움)
- 이상에 **강건한 loss function**을 사용하거나 (reweighting, Group DRO)
- **Test-time adaptation**으로 대응한다 (test time에만 해결, training은 방치)

이 방법들의 한계: 이상 데이터를 "제거할 대상" 또는 "견뎌야 할 noise"로 취급한다.

### 2.2 Proposed Perspective
우리는 반대로, **이상 기간의 데이터를 적극적으로 활용**한다.

핵심 가설: 동일 센서의 정상/이상 기간 데이터는 **scale은 다르지만 pattern은 공유**한다. 이 자연스러운 변동(natural variation)을 contrastive learning의 positive pair로 활용하면, 모델은 자연스럽게 **scale-invariant, pattern-preserving representation**을 학습한다.

### 2.3 Novelty
기존 교통 contrastive learning과의 차별점:

| 방법 | Positive Pair 정의 | Augmentation |
|------|-------------------|-------------|
| **ST-SSL** (AAAI 2023) | 같은 그래프의 augmented views | 인위적 (edge drop, feature mask) |
| **STGCL** (SIGSPATIAL 2022) | 같은 입력의 augmented views | 인위적 (interpolation, frequency) |
| **CL4ST** (CIKM 2023) | Meta-learned views | 학습된 augmentation |
| **CoST** (ICLR 2022) | 같은 시계열의 time/freq views | Domain-specific |
| **Ours** | 같은 node의 **다른 연도** 데이터 | **자연적 변동 (natural augmentation)** |

**우리의 장점**:
1. 인위적 augmentation이 아닌 **real-world variation** 활용 → 더 의미있는 invariance 학습
2. Multi-year 데이터가 있는 교통 도메인에 자연스럽게 적용
3. Scale shift vs Pattern preservation를 명시적으로 분리하는 inductive bias

### 2.4 Key References

**Contrastive Learning in Traffic:**
- ST-SSL (AAAI 2023): Spatio-temporal self-supervised learning, adaptive graph augmentation
- STGCL (SIGSPATIAL 2022): Multi-domain augmentation (time, frequency, graph structure)
- CL4ST (CIKM 2023): Meta-learning으로 augmentation view 자동 생성

**Scale-Invariant Representation:**
- CoST (ICLR 2022): Seasonal-trend disentanglement via contrastive learning
- RevIN (ICLR 2022): Instance normalization for non-stationary time series

**Invariant Learning:**
- DIDA (NeurIPS 2022): Spatio-temporal intervention for invariant pattern extraction
- CaST (NeurIPS 2023): Causal backdoor adjustment for spurious correlation removal

**Soft Contrastive:**
- SoftCLT (ICLR 2024): Continuous similarity (0-1) instead of hard positive/negative

---

## 3. Methodology

### 3.1 Overall Framework

```
Multi-Year Data ──→ Cross-Year Pair Sampler ──→ Shared Encoder ──→ Projection Head ──→ Contrastive Loss
    (2022, 2023, 2024)     (positive/negative pairs)      (STAEformer backbone)   (MLP)          (InfoNCE)
                                                                │
                                                                ├──→ Prediction Head ──→ Forecasting Loss
                                                                │       (downstream)
                                                                └──→ [Optional] Scale Predictor ──→ Scale Loss
```

**Phase 1: Pre-training** - Cross-year contrastive learning으로 drift-invariant representation 학습
**Phase 2: Fine-tuning** - Learned encoder + prediction head로 forecasting task 학습

### 3.2 Positive/Negative Pair Selection

#### Positive Pairs (같은 semantic context, 다른 연도)

| Level | Positive Pair 정의 | 설명 |
|-------|-------------------|------|
| **Strict** | 같은 node, 같은 ToD, 같은 DoW, 다른 year | 월요일 8am, Node A, 2022 ↔ 2023 |
| **Relaxed** | 같은 node, 같은 ToD (±1h), 같은 DoW group, 다른 year | DoW group: 평일/주말 |
| **Spatial** | 이웃 node group, 같은 ToD, 다른 year | 인접 도로 센서 포함 |

**권장**: Strict로 시작, ablation으로 Relaxed/Spatial 비교

#### Negative Pairs

| Strategy | 설명 | 장단점 |
|----------|------|--------|
| **Random** | Batch 내 다른 모든 sample | 간단, false negative 위험 |
| **Hard negative** | 같은 node, 다른 ToD, 같은 year | ToD가 다르면 패턴이 다름 → meaningful |
| **Cross-node** | 다른 node, 같은 ToD, 같은 year | 공간적 차이 학습 |
| **Soft** (SoftCLT) | 연속적 유사도 (0-1) | False negative 문제 완화 |

**권장**: Hard negative (같은 node, 다른 ToD) + Random의 혼합

#### Sampling Strategy

```python
# Pseudo-code for cross-year pair sampling
def sample_pairs(data_2022, data_2023, data_2024, node_idx, batch_size):
    pairs = []
    for _ in range(batch_size):
        n = random.choice(node_idx)          # 같은 node
        tod = random.randint(0, 287)          # 같은 time-of-day
        dow = random.randint(0, 6)            # 같은 day-of-week

        # Positive: 다른 연도의 같은 context
        y1, y2 = random.sample([2022, 2023, 2024], 2)
        x_anchor = get_window(data[y1], n, tod, dow)  # (T_in, C)
        x_positive = get_window(data[y2], n, tod, dow)  # (T_in, C)

        # Hard negative: 같은 node, 다른 ToD
        tod_neg = (tod + random.randint(72, 216)) % 288  # 4-15시간 차이
        x_negative = get_window(data[y1], n, tod_neg, dow)

        pairs.append((x_anchor, x_positive, x_negative))
    return pairs
```

### 3.3 Encoder Architecture

기존 STAEformer를 backbone으로 사용:
- Input: (B, T_in, N, C) → 각 노드별 temporal embedding 추출
- Encoder output: (B, N, D) where D = hidden dimension

**Projection Head** (contrastive loss용):
```
z = MLP(encoder_output)
  = Linear(D, 512) → ReLU → Linear(512, 128)
```

### 3.4 Loss Functions

#### Contrastive Loss (Pre-training)

**InfoNCE Loss** (node-level):
```
L_CL = -log( exp(sim(z_i, z_i+) / τ) / Σ_k exp(sim(z_i, z_k) / τ) )
```
- `z_i`: anchor node representation
- `z_i+`: positive pair (같은 node, 다른 year)
- `z_k`: all samples in batch (negatives + positive)
- `sim(·,·)`: cosine similarity
- `τ`: temperature (0.07 ~ 0.1)

#### Joint Loss (Fine-tuning)

```
L_total = L_forecast + λ * L_CL
```
- `L_forecast`: MAE loss on flow prediction
- `λ`: contrastive loss weight (0.01 ~ 0.1)
- **Annealing**: λ를 학습 초기에 크게, 후반에 줄여서 forecasting에 집중

### 3.5 추가 아이디어

#### 3.5.1 Multi-Scale Contrastive

단일 시점이 아닌 **다양한 시간 scale**에서 contrastive:
- **Point-level**: 개별 timestep의 representation 비교
- **Patch-level** (1시간): 12 timestep window의 representation 비교
- **Daily-level**: 전체 일일 패턴의 representation 비교

```
L_CL = α * L_point + β * L_patch + γ * L_daily
```

각 scale에서 서로 다른 invariance를 학습:
- Point: 순간적 교통 상태의 invariance
- Patch: 출퇴근 피크 등 단기 패턴의 invariance
- Daily: 전체 일일 리듬의 invariance

#### 3.5.2 Spatial-Aware Contrastive

이웃 노드를 **soft positive**로 활용:
```
w_spatial(i, j) = exp(-d(i,j) / σ)  # 거리 기반 가중치
L_spatial = -Σ_j w(i,j) * log(exp(sim(z_i, z_j+) / τ) / ...)
```

인접 센서는 비슷한 교통 패턴 → soft positive pair

#### 3.5.3 Scale-Explicit Disentanglement (Cross-Year CL + Disentanglement 결합)

Encoder output을 두 component로 분리:
- **z_pattern**: 시간 변동 패턴 (contrastive로 year-invariant하게 학습)
- **z_scale**: 절대 유량 크기 (year-specific, scale prediction으로 학습)

```
z = Encoder(x)
z_pattern = Linear(z)  → contrastive loss (cross-year invariant)
z_scale = Linear(z)    → scale prediction loss (predict mean flow)
z_combined = concat(z_pattern, z_scale)  → forecasting head
```

이 방법은 아이디어 #7 (Disentangled Representation)과 자연스럽게 결합된다.

#### 3.5.4 Curriculum Contrastive Learning

쉬운 pair에서 어려운 pair로 점진적 학습:
1. **Stage 1**: 2022↔2024 (정상↔정상, r=0.98) → 쉬운 alignment
2. **Stage 2**: 2022↔2023 (정상↔이상, r=0.89) → 어려운 alignment
3. **Stage 3**: All pairs → 통합 학습

---

## 4. 모델 학습 이전 검증 (Pre-experiment Validation)

모델 학습 없이도 아이디어의 근거를 확인할 수 있는 분석들:

### 4.1 Pattern Preservation 정량화 ✅ (완료)

이미 확인된 결과:
- Daily profile correlation: r=0.89 (2022-2023), r=0.98 (2022-2024)
- **결론**: Pattern은 drift 기간에도 잘 보존됨

### 4.1b Cross-Year Positive Pair False Positive 분석 ✅ (완료)

**목적**: "같은 node + 같은 ToD/DoW + 다른 year = positive pair" 가정의 위험도 정량화

**방법**: 각 노드의 평균 daily profile (288 steps)을 연도 간 Pearson correlation으로 비교

#### 결과 1: 연도 간 Daily Profile Correlation

| 연도 쌍 | Mean r | Median r | Std | Min |
|---------|--------|----------|-----|-----|
| 2022-2023 | 0.877 | 0.909 | 0.116 | 0.093 |
| 2022-2024 | **0.976** | **0.993** | 0.059 | 0.261 |
| 2023-2024 | 0.879 | 0.912 | 0.111 | 0.093 |

→ 2023 anomaly year 포함 쌍이 낮고, 정상 연도 간(2022↔2024)은 매우 높음

#### 결과 2: Similarity 분류 (893 nodes)

| 등급 | 기준 | 노드 수 | 비율 |
|------|------|---------|------|
| High | r > 0.9 | 568 | 63.6% |
| Medium | 0.7 < r < 0.9 | 200 | 22.4% |
| Low | 0 < r < 0.7 | 24 | 2.7% |
| Negative | r < 0 | 0 | 0.0% |
| Invalid | NaN (std=0) | 101 | 11.3% |

→ **전체 false positive rate (r < 0.7) = 14.0%**, 대부분 dead/major sensor

#### 결과 3: 센서 상태 변화 (dead↔alive)

- 2022↔2023: **243개** 노드 상태 변화 (2023 anomaly로 다수 고장)
- 2022↔2024: 103개 변화
- 2023↔2024: 212개 변화
- 전체: **279개 (31.2%)** 어떤 연도 쌍에서든 상태 변화

#### 결과 4: False Positive Rate by Category

| 카테고리 | FP Rate (r < 0.7) | 해석 |
|----------|-------------------|------|
| Functional | **12.9%** | 대부분 안전 |
| Major fail | 17.9% | 약간 높음 |
| Dead | 20.0% | 높지만 CL에서 제외 대상 |

#### 시사점: Pair Selection 전략

| 전략 | 사용 가능 노드 | 예상 FP |
|------|---------------|---------|
| 필터 없음 | 893 | ~14% |
| Functional만 | 745 | ~12.9% |
| **Functional + r>0.7** | **649** | **~0%** |

**결론**: Cross-Year CL은 viable. Functional 센서 기준 FP 12.9%이며, correlation-based filtering으로 거의 0%까지 줄일 수 있음. 단, **센서 상태 변화(31.2%)가 가장 큰 위험 요인**이므로, pair selection 시 sensor health filtering이 필수.

> 분석 코드: `eda/concept_drift/cross_year_pair_analysis.py`
> 결과 파일: `eda/concept_drift/cross_year_correlations.json`

### 4.2 Scale vs Pattern Decomposition

```python
# 각 노드의 daily profile을 scale과 pattern으로 분해
for each node n:
    profile_y = mean daily flow profile for year y  # (288,)
    scale_y = mean(profile_y)                        # scalar
    pattern_y = profile_y / scale_y                  # normalized shape (288,)

# Cross-year 비교
pattern_similarity = corr(pattern_2022, pattern_2023)  # 높을수록 좋음
scale_ratio = scale_2023 / scale_2022                   # 1에서 벗어날수록 drift
```

**기대**: pattern_similarity >> scale_similarity → contrastive learning의 target이 명확함

### 4.3 기존 STAEformer Hidden Representation 분석

학습된 STAEformer의 내부 representation을 분석:
- 2022 데이터와 2023 데이터를 forward pass
- 같은 node의 representation이 연도 간에 얼마나 유사한지 측정
- t-SNE/UMAP으로 시각화: node별, year별 clustering 확인

**기대**: 현재 모델은 scale에 민감 → 같은 node라도 연도가 다르면 representation이 떨어져 있을 것

### 4.4 Oracle Experiment: RevIN의 효과

가장 간단한 scale-invariant 방법인 RevIN을 STAEformer에 적용:
- Instance normalization → prediction → de-normalization
- 만약 RevIN만으로 cross-year MAE가 크게 개선되면, scale이 주요 drift 원인임을 확인
- 이후 contrastive learning은 더 정교한 scale-invariance를 제공

### 4.5 Flow Magnitude vs Prediction Error 관계

```python
# 각 sample의 평균 flow magnitude와 prediction error 관계
for each test sample:
    magnitude = mean(input_flow)
    error = MAE(prediction, target)

# 높은 상관관계 → scale-dependent prediction → contrastive의 잠재적 효과 큼
```

---

## 5. 모델 학습 이후 분석 (Post-experiment Analysis)

### 5.1 Cross-Year Generalization 비교

| Experiment | Train | Test | MAE |
|-----------|-------|------|-----|
| Baseline (no CL) | 2022 | 2022 | ? |
| Baseline (no CL) | 2022 | 2023 | ? |
| Cross-Year CL | 2022 | 2022 | ? |
| Cross-Year CL | 2022 | 2023 | ? |

**핵심 metric**: Cross-year MAE 증가율 (낮을수록 drift-robust)

### 5.2 Representation Quality 분석

#### 5.2.1 Alignment & Uniformity
- **Alignment**: Positive pair representation의 평균 거리 (낮을수록 좋음)
- **Uniformity**: 전체 representation의 균일 분포 정도 (높을수록 좋음)
- Wang & Isola (2020)의 framework 적용

#### 5.2.2 CKA (Centered Kernel Alignment)
- Baseline vs CL 모델의 layer별 representation 비교
- 어느 layer에서 가장 큰 차이가 나는지 확인

#### 5.2.3 t-SNE/UMAP Visualization
- Representation을 시각화하여:
  - CL 전: year별로 clustering (scale에 의존)
  - CL 후: node/ToD별로 clustering (pattern에 의존)
  - → year-invariant representation 학습 확인

### 5.3 Scale Sensitivity 분석

- Test data의 flow를 인위적으로 scaling (0.5x, 0.8x, 1.2x, 1.5x)
- Baseline vs CL 모델의 성능 변화 비교
- CL 모델이 scaling에 덜 민감하면 → scale-invariant representation 학습 확인

### 5.4 Per-Node Improvement 분석

- Node별로 CL의 개선 효과를 측정
- drift가 심한 node (flow 변화 큰 node)에서 더 큰 개선이 있는지 확인
- 상관관계: flow_change_rate vs MAE_improvement

### 5.5 Ablation Study

| Variant | 설명 |
|---------|------|
| No CL (baseline) | Contrastive learning 없이 학습 |
| CL (strict pairs) | 같은 node, 같은 ToD/DoW |
| CL (relaxed pairs) | ToD ±1h, DoW group |
| CL (spatial pairs) | 이웃 node 포함 |
| CL + multi-scale | Point + Patch + Daily |
| CL + scale disentangle | Pattern/Scale 분리 |
| CL + curriculum | Easy → Hard pair progression |

### 5.6 Practical Impact

- **Same-year performance**: CL이 same-year 예측도 개선하는가? (representation quality 향상)
- **Mixed-year training**: 2022+2023 혼합 데이터로 학습 시, CL 유무에 따른 차이
- **Transfer**: County A에서 학습한 CL representation이 County B에서도 유효한가?

---

## 6. Implementation Roadmap

### Phase 0: Pre-experiment Validation (1-2일)
- [ ] Scale vs Pattern decomposition 분석
- [ ] 기존 STAEformer representation 분석 (t-SNE)
- [ ] RevIN oracle experiment
- [ ] Flow magnitude vs error 상관관계

### Phase 1: Basic Cross-Year CL (3-5일)
- [ ] Cross-year pair sampler 구현
- [ ] Projection head + InfoNCE loss 구현
- [ ] Pre-training loop 구현
- [ ] Fine-tuning + evaluation
- [ ] Cross-year MAE 비교

### Phase 2: Extensions (3-5일)
- [ ] Multi-scale contrastive
- [ ] Scale disentanglement
- [ ] Ablation studies

### Phase 3: Analysis (2-3일)
- [ ] Representation visualization
- [ ] Alignment & Uniformity 측정
- [ ] Scale sensitivity test
- [ ] Per-node improvement 분석

---

## 7. Hyperparameters

| Parameter | Range | Default |
|-----------|-------|---------|
| Temperature τ | 0.05 - 0.2 | 0.07 |
| Contrastive weight λ | 0.01 - 0.1 | 0.05 |
| Projection dim | 64 - 256 | 128 |
| Projection hidden | 256 - 512 | 512 |
| Batch size (contrastive) | 128 - 512 | 256 |
| Pre-training epochs | 10 - 50 | 30 |
| ToD tolerance (relaxed) | 0 - 12 steps | 0 (strict) |
| Spatial radius (spatial pairs) | 0 - 3 hops | 0 (node only) |

---

## 8. Risk & Mitigation

| Risk | Mitigation |
|------|-----------|
| Contrastive collapse (모든 representation 동일) | Temperature tuning, uniformity monitoring |
| False positive pairs (센서 상태 변화) | Sensor health filtering (functional only) + correlation threshold (r>0.7) → FP ~0% |
| False negative (같은 패턴을 negative로) | Soft contrastive (SoftCLT) 또는 hard negative 전략 |
| Pre-training/fine-tuning gap | Joint training (CL + forecasting 동시) 옵션 |
| 3년 데이터만으로 pair 부족 | Relaxed pair + temporal augmentation |
| Same-year performance 저하 | λ annealing, forecasting loss 우선 |
