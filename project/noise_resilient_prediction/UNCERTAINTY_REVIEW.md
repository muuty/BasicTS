# Uncertainty-Aware Traffic Prediction: 실험 리뷰 문서

**Date**: 2026-02-26
**Status**: 탐색 단계 (두 가지 접근 모두 한계 확인, 방향 재설정 필요)

---

## 1. 연구 동기

Traffic prediction에서 센서 노드의 **입력 품질이 불균일**한 문제:
- 893개 노드 중 120개 dead(>90% zero), 28개 major fail(50-90% zero), 745개 functional
- STAEformer의 spatial attention은 dead 센서에 3.5-12x 더 많은 attention 부여 (softmax bias)
- 모델이 노이즈/결측 입력을 얼마나 신뢰해야 하는지 모름

**목표**: 예측과 함께 **node-level uncertainty σ**를 출력 → "이 노드의 예측은 불확실하다"는 신호 제공

---

## 2. 접근법 A: Heteroscedastic NLL Loss

### 2.1 설계

모델이 각 노드에 대해 μ (예측값)과 σ (불확실성)를 동시 출력:

```
STAEformer(output_dim=2) → [μ, raw_σ]
σ = softplus(raw_σ) + σ_min
```

**Loss**: MAE와 Gaussian NLL의 가중합

```
L = α · MAE(μ, y) + (1-α) · 0.5 · [log(σ²) + (y-μ)²/σ²]
```

- α=0.5 고정 (Exp A), curriculum 스케줄링 예정이었으나 Exp A 결과로 중단
- NLL의 역할: σ가 크면 에러 패널티 감소, σ가 작으면 에러 패널티 증가 → 모델이 "확신 없는 곳에 큰 σ"를 배움

### 2.2 구현

| 파일 | 설명 |
|------|------|
| `baselines/STAEformer/arch/staeformer_uncertainty.py` | STAEformerUncertainty (STAEformer + σ head) |
| `basicts/losses/gaussian_nll.py` | hybrid_nll_loss (MAE + NLL) |
| `baselines/STAEformer/SAN_BERNARDINO_5ch_uncertainty.py` | Exp A config (α=0.5) |

### 2.3 실험 결과 (Exp A)

**학습**: 50 epochs, GPU 1, SAN_BERNARDINO 5ch

| Metric | Baseline (STAEformer) | Exp A (α=0.5) |
|--------|----------------------|----------------|
| Overall MAE | 12.10 | 13.18 (+8.9%) |

**σ 분석 결과**:
- σ ordering은 센서 건강도와 일치: dead(σ=15.2) > major_fail(σ=8.7) > functional(σ=3.1) ✓
- **BUT** Spearman correlation (σ vs per-node MAE) = **-0.775 (음수!)**
  - σ가 높은 노드 = dead 센서 = MAE가 낮음 (zero 예측 → 낮은 에러)
  - σ가 낮은 노드 = functional = MAE가 높음 (실제 flow가 크니까)

### 2.4 진단

**σ가 학습한 것: "입력 품질" (input quality), "예측 에러" (prediction error)가 아님**

```
Dead sensor: 입력 불확실 → σ 높음, BUT 예측 쉬움 (zero) → MAE 낮음
Functional:  입력 확실 → σ 낮음, BUT 예측 어려움 (변동성) → MAE 높음
```

NLL loss의 본질적 한계:
- NLL은 `(y-μ)²/σ²`를 최소화 → 에러가 큰 곳에 σ를 키움
- Robust model은 dead sensor를 잘 보상함 (spatial attention으로 이웃 정보 활용) → dead의 MAE가 이미 낮음
- 결과적으로 σ는 "prediction error"가 아닌 "input variability"를 반영

**결론**: NLL만으로는 "입력 품질" 기반 uncertainty를 얻을 수 없음. σ와 에러가 반대 방향.

---

## 3. 접근법 B: Perturbation-based Δ (Usefulness Signal)

### 3.1 동기

NLL의 한계를 극복하기 위해 **"이 노드가 다른 노드의 예측에 얼마나 도움이 되는가"**를 측정:

```
Δ(v) = MAE_masked(neighbors) - MAE_clean(neighbors)
```

- 노드 v의 물리적 채널(flow, occ, speed)을 0으로 마스킹
- v의 이웃 노드들의 MAE 변화 측정
- Δ > 0: v는 이웃에게 유용 (masking하면 이웃 성능 저하)
- Δ ≈ 0: v는 이웃에게 무관
- Δ < 0: v는 이웃에게 해로움

**장기 목표**: Δ를 supervision signal로 사용해 σ head를 학습 (amortized estimation)
- 학습 시: 비싼 Δ 계산 → σ의 target으로 사용
- 추론 시: 학습된 σ head만 사용 (O(1))

### 3.2 구현

| 파일 | 설명 |
|------|------|
| `experiments/validate_delta.py` | 기본 Δ 검증 (top-K 이웃, 카테고리별 비교) |
| `experiments/validate_delta_deep.py` | 심화 분석 (collective masking, attention 분석 등) |

### 3.3 첫 실행 (버그 수정 전)

- adj_mx가 거의 fully connected (평균 847/893 이웃) → Δ 신호 희석 → 모든 카테고리 Δ≈0
- Normalization 버그: scaler가 3D tensor를 받아 node 0만 정규화 → clean MAE=130 (예상 ~12)
- 카테고리 인덱스 float→int 캐스팅 에러

### 3.4 버그 수정 후 결과

**수정 사항**:
1. Top-K=10 이웃 사용 (adj weight 기준 상위 10개)
2. Scaler에 채널 차원 유지하여 전달 (`[..., :1]`)
3. 카테고리 인덱스 `.astype(int)`

**결과**:

| Category | n | mean Δ | std Δ | 해석 |
|----------|---|--------|-------|------|
| functional | 20 | +0.0009 | 0.0044 | 사실상 무의미 |
| major_fail | 20 | -0.0003 | 0.0014 | 사실상 무의미 |
| dead | 20 | +0.0000 | 0.0000 | 완전히 0 |

- Clean MAE = 14.286 (baseline ~12, subset이라 약간 차이)
- Mann-Whitney U test: 아직 수행하지 않았으나, 차이가 너무 작음
- **방향은 맞음** (functional > dead) 하지만 **magnitude가 negligible** (0.0009 / 14 = 0.006%)

### 3.5 진단

**왜 Δ가 이렇게 작은가?**

1. **STAEformer의 spatial attention은 매우 분산적**: 단일 노드 제거가 이웃 예측에 거의 영향 없음
2. **Temporal pattern이 dominant**: 교통 데이터는 시간적 패턴이 강해서, 공간 정보 하나 빠져도 시간 패턴으로 보상 가능
3. **Top-K=10도 불충분할 수 있음**: 실제 spatial dependency는 adj_mx weight와 다를 수 있음 (model은 adaptive embedding으로 자체 spatial representation 학습)

---

## 4. 현재 상태 및 판단이 필요한 지점

### 확인된 사실

1. **NLL 기반 σ**는 "input quality"를 학습하지만, 에러와 **반대 방향** (dead=높은σ+낮은MAE)
2. **Perturbation Δ**는 방향은 맞으나 signal이 너무 약해서 supervision에 부적합
3. STAEformer의 spatial attention은 단일 노드에 대한 의존도가 극히 낮음
4. 모델은 이미 dead sensor를 잘 보상하고 있음 (robust)

### 미수행 분석 (validate_delta_deep.py)

심화 분석 스크립트를 준비했으나 아직 실행하지 않음:

| 분석 | 목적 |
|------|------|
| **Collective masking** | dead 120개를 한꺼번에 마스킹 → 집단 효과는? |
| **Top-K sensitivity** | K=3,5,10,20,50에서 Δ 변화 패턴 |
| **Spatial attention 추출** | 모델이 실제로 dead를 얼마나 attend하는지 |
| **Per-sample Δ variance** | 평균은 작지만 특정 시점에서는 큰 Δ가 있는지 |
| **Input stats vs Δ 상관** | flow 평균/분산과 Δ의 관계 |

### 열린 질문들

1. **Δ가 약한 이유가 "spatial attention이 분산적"이라면**: 이건 좋은 것인가 나쁜 것인가?
   - 좋은 점: 모델이 이미 robust
   - 나쁜 점: uncertainty signal을 구할 수 없음

2. **Collective masking이 유의미하다면**: 개별 노드 Δ는 약하지만 집단적 영향은 있을 수 있음
   → Dead 노드 집단의 collective 영향을 supervision signal로 사용 가능?

3. **NLL의 σ가 "input quality"를 잘 반영한다면**: 에러 correlation이 아닌 **input quality 자체가 유용한 output**이 될 수 있는가?
   - "이 센서의 데이터를 신뢰할 수 있는가"라는 질문에는 답이 됨
   - 다만 "이 예측이 맞는가"라는 질문에는 답이 안 됨

4. **대안적 접근**: Ensemble disagreement, MC Dropout, 또는 category-supervised σ?

---

## 5. 코드 구조

```
baselines/STAEformer/
├── arch/
│   ├── staeformer_arch.py          # 원본 STAEformer
│   ├── staeformer_uncertainty.py   # STAEformerUncertainty (μ+σ output)
│   └── __init__.py                 # STAEformerUncertainty 등록됨
├── SAN_BERNARDINO_5ch.py           # Baseline config
└── SAN_BERNARDINO_5ch_uncertainty.py # Exp A config

basicts/losses/
└── gaussian_nll.py                 # hybrid_nll_loss

experiments/
├── validate_delta.py               # Δ 기본 검증 (수정 완료)
└── validate_delta_deep.py          # Δ 심화 분석 (미실행)
```

---

## 6. 재현 방법

```bash
# Exp A 학습
python -c "from basicts import launch_training; launch_training('baselines/STAEformer/SAN_BERNARDINO_5ch_uncertainty.py', gpus='1')"

# Δ 검증 실험
python experiments/validate_delta.py --gpu 1 --n-samples 500 --n-nodes 20

# 심화 분석 (미실행)
python experiments/validate_delta_deep.py --gpu 1 --n-samples 500 --n-nodes 20
```
