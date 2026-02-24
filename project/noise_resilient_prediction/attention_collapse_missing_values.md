# Attention Collapse with Missing Values in Spatio-Temporal Forecasting

## Problem Statement

Attention 기반 시공간 예측 모델들이 temporal/spatial aggregation을 수행할 때, 불규칙하게 분포한 missing values (zero-filled sensor failures)가 일관된 key representation을 생성하여 softmax attention에서 비정상적으로 높은 weight를 받게 된다. 이로 인해 정상적인 관측값의 정보가 희석되고, 모델이 uninformative한 입력에 과도하게 의존하는 **attention collapse** 현상이 발생한다.

---

## 1. Background

### 1.1 Dataset: SAN_BERNARDINO

- **893 sensors**, 5 features: flow(ch0), occupancy(ch1), speed(ch2), tod(ch3), dow(ch4)
- Shape: (105120, 893, 5), 5분 간격, 288 steps/day
- 학습에 사용한 범위: data_range=(0, 26280) = 약 3개월
- Train/Val/Test split: 60/20/20

### 1.2 Sensor Quality Categories

데이터 내 센서의 품질이 매우 불균일함. 3개월 기준 flow=0 비율로 분류:

| Category | 기준 | 노드 수 | 비율 | 특성 |
|----------|------|---------|------|------|
| Dead | >90% zero | 121 | 13.5% | 항상 0, 완전 고장 |
| Major fail | 50-90% zero | 28 | 3.1% | 간헐적 정상, 대부분 0 |
| Partial fail | 5-50% zero | 161 | 18.0% | 불규칙하게 0 섞임 |
| Functional | <5% zero | 583 | 65.3% | 정상 작동 |

**핵심**: Major/Partial fail 노드는 같은 노드라도 시간대에 따라 정상(flow>0)일 때도 있고 결측(flow=0)일 때도 있음. Static per-node 처리로는 해결 불가.

### 1.3 Base Model: STAEformer

- Paper: "STAEformer: Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting" (CIKM 2023)
- 구조: Input Embedding → Temporal Self-Attention → Spatial Self-Attention → Output Projection
- 우리 설정: num_layers=1, model_dim=96 (24+24+24+0+24), num_heads=4
- Forward features: [flow, tod, dow] (3 channels)
- Scaler: Z-score on flow only (mean=124.03, std=157.83)
- **Baseline MAE: 12.10**

---

## 2. 발견: Attention Collapse

### 2.1 Spatial Attention 분석

Baseline STAEformer의 spatial attention weight를 분석한 결과, **dead/failing 노드가 정상 노드보다 3.5~12배 높은 attention을 받음**.

#### Functional 노드가 각 카테고리에 주는 attention weight (per-node 평균)

| Time | → Dead | → Major | → Partial | → Func | Dead/Uniform | Func/Uniform |
|------|--------|---------|-----------|--------|--------------|--------------|
| 3am  | 0.002091 | 0.001653 | 0.001439 | 0.000804 | 1.87x | 0.72x |
| 9am  | 0.003868 | 0.003178 | 0.001501 | 0.000345 | 3.45x | 0.31x |
| 3pm  | 0.003932 | 0.003098 | 0.001559 | 0.000320 | 3.51x | 0.29x |
| 9pm  | 0.003169 | 0.002593 | 0.001554 | 0.000504 | 2.83x | 0.45x |

Uniform attention = 1/893 = 0.001120

#### Func→Dead vs Func→Func 비율

| Time | Attn to Dead | Attn to Func | Dead/Func 비율 |
|------|-------------|-------------|----------------|
| 3am  | 0.002091 | 0.000804 | **2.6x** |
| 9am  | 0.003868 | 0.000345 | **11.2x** |
| 3pm  | 0.003932 | 0.000320 | **12.3x** |
| 9pm  | 0.003169 | 0.000504 | **6.3x** |

**패턴**: 낮 시간(9am, 3pm)에 효과가 더 강함 — 이 시간대에 dead=0, functional=high traffic으로 차이가 극대화.

#### 가장 많이/적게 attend 받는 노드

- **Top 10 most attended**: 8개 dead (zero_rate=1.0), 2개 major fail
- **Top 10 least attended**: 10개 모두 functional (zero_rate≈0)

### 2.2 원인: Consistent Key Representation

Dead/failing 노드가 과도한 attention을 받는 메커니즘:

1. **일관된 입력**: Dead 노드는 항상 flow≈0 → z-score 정규화 후 항상 ≈ -0.785
2. **일관된 key embedding**: 같은 입력 → input_proj, tod/dow/adaptive embedding을 거쳐도 key가 항상 비슷
3. **Softmax 편향**: Key의 분산이 작을수록 (일관적일수록) 다양한 query와의 dot product가 비슷 → softmax에서 상대적으로 높은 weight
4. **Functional 노드의 key 분산**: 시간대별로 다양한 flow → key가 분산 → 특정 query에만 높은 attention, 평균적으로 낮음

이것은 softmax attention의 **구조적 취약점**임: constant input이 attention을 독점.

### 2.3 Temporal Attention에서도 동일한 문제

Major/Partial fail 노드의 12 timestep 중 flow=0인 timestep들은:
- Spatial의 dead 노드와 동일한 메커니즘으로 과도한 temporal attention을 받음
- 유의미한 flow>0 timestep의 시간적 패턴 정보가 희석됨

```
Major fail 노드 예시 (12 timesteps):
flow:    [0,    0,    150,  200,  0,    0,    0,    180,  210,  0,    0,    0   ]
z-score: [-0.79,-0.79, 0.16, 0.48,-0.79,-0.79,-0.79, 0.35, 0.54,-0.79,-0.79,-0.79]
                                    ↑ 이 timestep들이 temporal attention을 독점
```

---

## 3. Gate 실험 (실패 분석)

Dead 노드의 spatial attention 영향을 줄이기 위해 value gating 메커니즘을 시도함.

### 3.1 Gating 메커니즘 설계

```python
# GatedAttentionLayer의 forward에서:
gate = sigmoid(gate_net(value) * temperature)  # [B, T, N, 1]
value = FC_V(value) * gate  # gate가 value를 스케일링
attn_score = softmax(Q @ K^T / sqrt(d))
output = attn_score @ (value * gate)
```

Gate는 **value (outgoing info)에만 작용**. Query/Key는 변경 없음.
Gate 입력: temporal attention 이후의 96-dim embedding.

### 3.2 실험 결과

| Model | 설정 | MAE | Gate 패턴 |
|-------|------|-----|-----------|
| Baseline (no gate) | - | **12.10** | N/A |
| Soft gate | τ=1, bias=0 | **12.01** | Uniform ~0.23 (99% in [0.1, 0.3)) |
| Sharp gate | τ=10, bias=-2 | 12.04 | All near 0 (81% < 0.01) |
| MLP gate | τ=10, bias=-2, h=24 | 12.62 | All exactly 0 (100% < 0.01) |

### 3.3 Gate 분석: 왜 실패했는가

#### Soft gate (τ=1): Uniform scaling, 차별 없음
- 모든 노드, 모든 시간대에서 gate ≈ 0.23
- Dead와 functional 노드 간 gate 차이 없음
- MAE 12.01 (baseline 대비 -0.09)은 단순히 spatial attention 전체를 23%로 축소한 효과
- Attention 패턴 분석: baseline과 동일한 dead-biased 패턴 유지 (3am: 2.6x, 9am: 9.4x)

#### Sharp gate (τ=10, bias=-2): 전부 닫힘
- sigmoid(-2 * 10) = sigmoid(-20) ≈ 0 에서 시작
- 학습 중에도 gate가 열리지 못함 (gradient 부족)
- Spatial attention이 사실상 비활성화 → 완전 uniform attention (1.0x everywhere)
- 모델이 temporal attention + output projection만으로 예측

#### MLP gate (τ=10, bias=-2, h=24): 완전 사망
- Sharp보다 더 심각: 100% gate = 0.0000
- MLP의 추가 nonlinearity가 gradient vanishing을 악화
- Attention 완전 uniform, 모든 노드 정확히 1/893

### 3.4 Gate 실패의 근본 원인

1. **작용 위치**: Gate는 value(softmax 이후)에 작용 → attention weight 자체는 변경 못함
2. **입력 신호 희석**: Gate 입력이 temporal attention 이후의 96-dim hidden → flow 정보가 24/96 dim으로 희석, temporal attention이 추가 혼합
3. **Gradient 부족**: Dead 노드는 target도 0이므로 masked_mae loss에서 직접 gradient 없음. Functional 노드로부터의 간접 gradient는 893-node softmax로 희석
4. **초기화 문제** (sharp/MLP): bias=-2 + temperature=10으로 sigmoid 포화 영역에서 시작 → gradient vanishing

---

## 4. Attention 패턴 전체 비교

### 4.1 Spatial Attention: Func→Dead / Func→Func 비율

| Model | MAE | 3am | 9am | 3pm | 9pm | 의미 |
|-------|-----|-----|-----|-----|-----|------|
| Baseline | 12.10 | 2.6x | 11.2x | 12.3x | 6.3x | Dead에 강한 편향 |
| Soft gate | 12.01 | 2.6x | 9.4x | 9.2x | 6.5x | Dead 편향 유지, 약간 완화 |
| Sharp gate | 12.04 | 1.0x | 1.0x | 1.0x | 1.0x | 완전 uniform (spatial OFF) |
| MLP gate | 12.62 | 1.0x | 1.0x | 1.0x | 1.0x | 완전 uniform (spatial OFF) |

### 4.2 해석

- Soft gate의 MAE 개선(12.01)은 dead-biased attention을 uniform scaling(23%)으로 **약간 희석**한 효과
- Sharp/MLP gate는 spatial attention을 완전히 제거 → baseline보다 나쁨 (spatial info 손실)
- **최적해는 dead 편향을 교정하되 spatial attention 자체는 유지하는 것**

---

## 5. 문제의 보편성

이 문제는 STAEformer에 국한되지 않음. Softmax attention을 사용하는 모든 시공간 모델에 해당:

### 5.1 영향받는 모델 유형
- **Attention 기반**: STAEformer, Informer, Autoformer, PatchTST, iTransformer 등
- **GNN + Attention**: STGCN (spatial graph + temporal attention), D2STGNN, STWave 등
- **Message passing GNN**: 이웃 노드의 feature를 aggregation할 때 동일한 문제 (softmax 대신 mean/sum이라도 constant value가 편향 생성)

### 5.2 영향받는 도메인
- Traffic forecasting (sensor failure → zero readings)
- Air quality monitoring (equipment malfunction)
- IoT sensor networks (intermittent connectivity)
- 모든 시공간 데이터에서 irregular missing values가 있는 경우

---

## 6. 해결 방향: Input-Conditioned Attention Bias

### 6.1 핵심 아이디어

Softmax 이전에 input-conditioned bias를 추가하여 missing value의 attention weight를 직접 조절:

```python
# 기존 attention
attn_score = Q @ K^T / sqrt(d)
attn_weight = softmax(attn_score)

# 제안: Input-Conditioned Attention Bias
credibility = f(raw_embedding)  # [B, T, N, 1] - 각 노드/시점의 신뢰도
attn_score = Q @ K^T / sqrt(d) + credibility.transpose(-1, -2)  # key-side bias
attn_weight = softmax(attn_score)
```

### 6.2 이전 Gate 대비 장점

| | Gate (실패) | Attention Bias (제안) |
|---|---|---|
| 작용 위치 | Value (softmax **이후**) | Attention logit (softmax **이전**) |
| 효과 | Value 크기만 조절 | **어떤 노드를 볼지** 직접 조절 |
| 입력 | 96-dim hidden (temporal attention 이후, flow 희석) | Raw embedding (temporal attention 이전, flow 신호 명확) |
| Gradient | `∂L/∂gate` = attn*V에 의존, 약함 | `∂L/∂bias` = softmax gradient, 직접적 |

### 6.3 설계 요구사항

1. **Input-conditioned (동적)**: 같은 노드라도 시간에 따라 다른 bias (major/partial fail 대응)
2. **Temporal + Spatial 양쪽에 적용**: 두 attention 모두 같은 문제를 가짐
3. **Bias 입력은 temporal attention 이전의 raw embedding**: Temporal attention 이후에는 flow 신호 희석

### 6.4 대안으로 고려했지만 탈락한 방법들

| 방법 | 탈락 이유 |
|------|-----------|
| Learnable per-node bias (A1) | Static — major/partial fail의 시간별 변화 대응 불가 |
| Historical variance feature (B4) | Static per-node 통계 |
| Graph-guided attention (C8) | Static adjacency |
| Dead node imputation (E14) | Static 전처리 |
| Entropy regularization (D11) | 시간별 차별화 불가 |

---

## 7. 분석 도구

### 7.1 `eda/analyze_attention.py`

Spatial attention weight 분석 도구. Baseline과 gated 모델 모두 지원.

```bash
# Baseline
python eda/analyze_attention.py checkpoints/.../STAEformer_best_val_MAE.pt \
    --num_layers 1 --adaptive_dim 24

# Gated model
python eda/analyze_attention.py checkpoints/.../STAEformerGated_best_val_MAE.pt \
    --gated --temperature 10 --bias -2 --hidden_dim 24
```

출력: 카테고리별 attention weight, per-node attention received, top/bottom attended nodes

### 7.2 `eda/analyze_gate.py`

Gate value 분석 도구. Forward hook으로 실제 gate 값 캡처.

```bash
python eda/analyze_gate.py checkpoints/.../STAEformerGated_best_val_MAE.pt \
    --temperature 10 --bias -2 --hidden_dim 24
```

출력: 카테고리별/시간대별/flow상태별 gate 값, gate 분포 히스토그램

---

## 8. 실험 체크포인트 위치

| Model | Checkpoint | MAE |
|-------|-----------|-----|
| Baseline | `checkpoints/ContextContrastive_baseline_3mo/xtraffic_SAN_BERNARDINO_30_12_12/040af4f5bcb37097bc5263d7f350174b/` | 12.10 |
| Soft gate (τ=1) | `checkpoints/STAEformer_Gated/SAN_BERNARDINO_30_12_12/666de8a4e62bfebb2dd670b2acde8075/` | 12.01 |
| Sharp gate (τ=10) | `checkpoints/STAEformer_Gated_Sharp/SAN_BERNARDINO_30_12_12/535b8aa7d6dd926a0b72b539a4751d5e/` | 12.04 |
| MLP gate (τ=10,h=24) | `checkpoints/STAEformer_Gated_MLP/SAN_BERNARDINO_30_12_12/5286b25259635796bb5a05a95890b25f/` | 12.62 |

## 9. 분석 결과 파일

- `experiments/eda_results/baseline_attention_analysis.txt` — Baseline attention 전체 결과
- `experiments/eda_results/soft_gate_attention_analysis.txt` — Soft gate attention 결과
- `experiments/eda_results/sharp_gate_attention_analysis.txt` — Sharp gate attention 결과
- `experiments/eda_results/mlp_gate_attention_analysis.txt` — MLP gate attention 결과

---

## 10. 핵심 Takeaways

1. **Missing values는 attention을 독점한다**: Dead 노드가 functional 노드보다 3.5~12x 높은 attention을 받음
2. **이것은 softmax의 구조적 취약점**: Constant input → consistent key → softmax bias
3. **Value gating으로는 해결 불가**: Gate가 value에 작용하므로 attention distribution 자체를 교정하지 못함
4. **Attention logit에 직접 개입해야 함**: Input-conditioned bias를 softmax 이전에 추가
5. **Temporal과 Spatial 양쪽 모두 문제**: 두 attention에 모두 적용 필요
6. **Dynamic (input-conditioned) 해야 함**: Major/Partial fail 노드의 시간별 변화 대응 필요
