# Noise-Resilient Traffic Prediction: Experiment Log

> Chronological record of experiments and findings.
> For structured results, see [RESULTS.md](RESULTS.md) and [EXPERIMENTS.md](EXPERIMENTS.md).
> For archived early experiments (contrastive pre-training, MICL), see [experiment_log_archive.md](experiment_log_archive.md).

---

## 12. Per-Node MAE Analysis by Sensor Quality (2026-02-09)

> **목적**: Encoder가 실제로 traffic prediction을 개선하는지, 아니면 dead sensor detection artifact인지 확인

### 12.1 Background

이전 분석에서 contrastive encoder의 pn_w1% (worst 1% node MAE) 개선이 flow_zero_rate와 r=0.88로 상관됨을 발견. 이번 분석은 **masked MAE** (zero target 제외)를 사용하여 실제 prediction 품질을 평가.

### 12.2 Sensor Categories

| Category | N | flow_zero_rate |
|---|---|---|
| Dead (>90%) | 121 | 센서 고장, 거의 항상 0 |
| Major fail (50-90%) | 28 | 심각한 데이터 손실 |
| Partial fail (5-50%) | 161 | 간헐적 데이터 손실 |
| Functional (<5%) | 583 | 정상 센서 |

### 12.3 Results: Masked MAE (proper metric, zero targets excluded)

| Category | N | Baseline | Contrastive_only | Context_only | Self_only |
|---|---|---|---|---|---|
| Dead (>90%) | 121 | 0.01 | 0.06 (+503%) | 0.08 (+646%) | 0.06 (+527%) |
| Major fail | 28 | 4.88 | 7.79 (+60%) | 8.70 (+78%) | 7.66 (+57%) |
| Partial fail | 161 | 5.60 | 5.94 (+6%) | 6.02 (+7%) | 5.82 (+4%) |
| Functional | 583 | 13.54 | 13.72 (+1%) | 14.02 (+4%) | 13.98 (+3%) |
| **ALL** | 893 | **10.00** | **10.28 (+3%)** | **10.52 (+5%)** | **10.42 (+4%)** |

### 12.4 Results: Raw (unmasked) MAE

| Category | N | Baseline | Contrastive_only | Context_only | Self_only |
|---|---|---|---|---|---|
| Dead (>90%) | 121 | 23.29 | 0.39 (**-98%**) | 0.46 (-98%) | 0.41 (-98%) |
| Major fail | 28 | 14.61 | 11.14 (-24%) | 15.01 (+3%) | 12.03 (-18%) |
| Partial fail | 161 | 7.98 | 7.91 (-1%) | 8.40 (+5%) | 8.01 (+0.3%) |
| Functional | 583 | 13.83 | 13.90 (+0.5%) | 14.23 (+3%) | 14.14 (+2%) |

### 12.5 Correlation Analysis

| Metric | Contrastive_only | Context_only | Self_only |
|---|---|---|---|
| r(unmasked improvement, flow_zero_rate) | 0.88 | 0.86 | 0.89 |
| r(masked improvement, flow_zero_rate) - all | -0.02 | -0.0001 | 0.003 |
| r(masked improvement, flow_zero_rate) - alive only | -0.13 | -0.11 | -0.12 |

### 12.6 Key Conclusions

1. **Masked MAE 기준, 모든 encoder 실험이 baseline보다 나쁨** (ALL categories에서)
2. **Unmasked MAE "개선"은 100% dead sensor detection 효과**: encoder가 zero-flow 센서에 zero 예측
3. **pn_w1% 개선도 동일 artifact**: pn_w1%가 unmasked per-node MAE 사용 → dead sensor가 worst에서 빠짐
4. **Masked MAE vs flow_zero_rate 상관관계 ≈ 0**: encoder가 어떤 카테고리도 체계적으로 개선하지 않음
5. **결론: contrastive encoder는 traffic prediction을 개선하지 않음. "robustness 개선"은 dead sensor detection artifact**

---

## 2026-02-10: Cross-Variable Consistency Pre-training

### 목적
Contrastive SSL (SimCLR)의 invariance objective가 forecasting에 misaligned됨을 확인 후, 새로운 SSL 접근:
물리적 변수 간 관계(flow/occupancy/speed)를 학습하는 cross-variable reconstruction objective.

### 방법
1. **Pre-training**: 3개 물리변수 중 1개를 랜덤 마스킹, 나머지로 복원 (MAE loss)
2. **Encoder**: TransformerEncoder(d_model=32, num_layers=2, nhead=4)
3. **Downstream**: encoder(5feat) → [B,T,N,32] + tod,dow → [B,T,N,34] → STAEformer

### 결과

| Experiment | Overall MAE | Masked MAE |
|---|---|---|
| Baseline (no encoder) | 12.10 | ~12.10 |
| Cross-variable pretrained (frozen) | 12.15 | 12.22 |
| Scratch encoder (random init) | 12.26 | 12.31 |

### 분석
1. Pretrained(12.15) > Scratch(12.26): pre-training이 0.9% 개선 → 뭔가 학습됨
2. Pretrained(12.15) < Baseline(12.10): 하지만 encoder 없는 것보다 나쁨
3. **핵심 문제**: 5feat → 32dim은 차원 확장이지만(bottleneck 아님), encoder representation이 STAEformer가 이미 학습하는 것 이상의 정보를 제공하지 못함
4. dim_feedforward 주의: ContextAwareEncoder=d_model*4, TransformerEncoder 기본값=256 (불일치 주의)

### 인사이트
- 모든 encoder 실험(contrastive, predictive, cross-variable)이 baseline 대비 개선 실패
- 문제는 SSL objective가 아닌 **별도 encoder 추가 자체** (5ch→32dim으로 차원은 늘어났으나 bottleneck이 아님)
- STAEformer가 이미 raw features에서 충분한 표현을 학습하므로, encoder representation이 추가 정보를 제공하지 못함
- 다음 방향: 별도 encoder 없이 SSL을 적용하는 방법 고민 필요 (e.g., multi-task, auxiliary loss)

---

## Attention Collapse & Node Removal Ablation (2026-02-11)

### 배경
Dead/major-fail 센서가 spatial attention에서 비정상적으로 높은 가중치를 받는 "attention collapse" 현상 발견.
Dead 노드는 일관된 zero 입력 → 일관된 key representation → softmax에서 높은 가중치.
해결 가설: (1) 문제 노드 제거, (2) credibility bias를 attention에 주입.

### 실험 설계
- **Masked loss**: `masked_mae` (target=0 제외) - 기존 baseline과 동일
- **Unmasked loss**: `unmasked_mae` (모든 값 포함) - 현실적 세팅
- **Node filtering**: `node_indices` 파라미터로 dead/major-fail 노드 제외
- 5ch 입력: flow, occupancy, speed, tod, dow

### Masked Loss 결과 (5ch)

| Experiment | Nodes | Overall MAE | Functional masked_mae |
|---|---|---|---|
| **5ch masked baseline** | **893** | **12.263** | **13.974** |
| 5ch masked no_dead | 773 | 12.285 (+0.2%) | 13.988 (+0.1%) |
| 5ch masked no_dead_major | 745 | 12.456 (+1.6%) | 14.037 (+0.5%) |

### Unmasked Loss 결과 (5ch)

| Experiment | Nodes | Overall MAE | Functional masked_mae |
|---|---|---|---|
| **5ch unmasked baseline** | **893** | **12.178** | **13.974** |
| 5ch unmasked no_dead | 773 | 12.361 (+1.5%) | 14.194 (+1.6%) |
| 5ch unmasked no_dead_major | 745 | 12.322 (+1.2%) | 14.021 (+0.3%) |

### 3ch vs 5ch 비교 (unmasked, no_dead)

> **채널 구성**: 3ch = flow + tod + dow (FORWARD_FEATURES=[0,3,4], input_dim=3). 5ch = flow + occ + speed + tod + dow (FORWARD_FEATURES=[0,1,2,3,4], input_dim=3). 5ch는 input_proj에 flow/occ/speed가 들어가고, 3ch는 flow/tod/dow가 들어감.

| Setting | Baseline functional | no_dead functional | Effect |
|---|---|---|---|
| 3ch unmasked (flow+tod+dow) | 14.464 | 13.795 | **-4.6% (개선)** |
| 5ch unmasked (flow+occ+speed+tod+dow) | 13.974 | 14.194 | **+1.6% (악화)** |

### Robustness 지표 (5ch unmasked)

| Metric | Baseline (893) | no_dead (773) | no_dead_major (745) |
|---|---|---|---|
| per_node std_MAE | 9.833 | 9.773 | 9.615 |
| worst_1pct_MAE (node) | 55.564 | 58.671 | 56.382 |
| worst_5pct_MAE (node) | 34.706 | 36.379 | 36.024 |

### 핵심 인사이트

1. **노드 제거는 해답이 아님**: Masked/unmasked 모두에서 dead/major 노드 제거 시 functional 예측 성능 악화
2. **5ch가 implicit credibility 제공**: 3ch(flow+tod+dow)에서는 노드 제거가 functional 예측 개선(-4.6%), 5ch(flow+occ+speed+tod+dow)에서는 악화(+1.6%). 5ch의 occupancy/speed 채널이 센서 health 판별 정보를 추가로 제공 (dead 센서는 3채널 모두 0)
3. **Unmasked가 masked보다 약간 나음**: 동일 functional masked_mae(13.974)이지만 overall MAE 기준으로 unmasked(12.178) < masked(12.263)
4. **Dead 노드가 학습에 도움**: 제거 시 학습 데이터 다양성 감소 → 일반화 저하. Dead 노드가 spatial pattern의 "anchor" 역할
5. **다음 방향**: 노드 제거 대신 credibility-aware attention (pre-softmax bias) 실험

---

## Credibility-Aware Spatial Attention (2026-02-12)

### 배경
노드 제거 실험에서 dead/major-fail 노드를 제거하면 오히려 functional 예측이 악화됨을 확인.
대신 **pre-softmax credibility bias**를 spatial attention에 추가하여 attention collapse를 직접 교정하는 접근.

### 메커니즘
```
credibility = linear(raw_embedding)  # (B, T, N, num_heads)
attn_score = Q @ K^T / sqrt(d) + credibility_bias  # key-side bias
attn_weight = softmax(attn_score)
```

- Raw embedding (temporal attention 이전)에서 credibility 계산 → flow 신호 보존
- Key-side bias: 각 노드가 얼마나 attend 받을지 직접 제어
- Per-head 출력: 각 head가 다른 credibility 기준 학습 가능
- 추가 파라미터: 단 388개 (model_dim * num_heads + num_heads = 96*4+4)

### 결과 (5ch unmasked, 893 nodes)

| Experiment | Overall MAE | Functional masked_mae | Dead raw_mae |
|---|---|---|---|
| **5ch unmasked baseline** | **12.178** | **13.974** | 0.659 |
| **5ch credibility bias** | **12.151 (-0.22%)** | **13.947 (-0.19%)** | **0.124 (-81%)** |

### Per-Category 비교 (masked_mae)

| Category | Baseline | Credibility | Change |
|---|---|---|---|
| Dead (133n) | 0.326 | 0.312 | -4.3% |
| Major fail (21n) | 11.957 | 11.987 | +0.2% |
| Partial fail (190n) | 5.794 | 5.762 | -0.6% |
| Functional (549n) | 13.974 | 13.947 | **-0.19%** |

### Robustness 비교

| Metric | Baseline | Credibility |
|---|---|---|
| median_MAE (sample) | 11.040 | 10.890 (-1.4%) |
| worst_1pct_MAE (sample) | 26.320 | 27.109 (+3.0%) |
| per_node std_MAE | 9.833 | 9.892 (+0.6%) |

### 핵심 인사이트

1. **Functional 예측 개선을 달성한 첫 번째 기법**: 노드 제거, 별도 encoder 추가, value gating 모두 functional 예측을 악화시켰으나, credibility bias는 13.974→13.947로 개선
2. **Dead 노드 raw_mae 극적 감소**: 0.659→0.124 (-81%). 모델이 dead 노드의 출력을 거의 0으로 학습
3. **최소 개입, 최대 효과**: 단 388개 파라미터 추가로 전체적 개선
4. **개선폭은 modest**: ~0.2% 개선은 noise 범위일 수 있음. 추가 실험으로 확인 필요

### Per-Sample Credibility 분석 (Node 184, major-fail, zero_rate=0.714)

Credibility bias가 의도대로 동적으로 작동하는지 검증하기 위해, major-fail 노드 184에 대해 sample별 분석 수행.

#### Credibility Bias 값 분석
| 노드 상태 | Credibility Bias 평균 | Std |
|---|---|---|
| Node 184 Working (flow > 0) | -0.1128 | ~0.001 |
| Node 184 Failing (flow = 0) | -0.1122 | ~0.001 |
| **Working/Failing Gap** | **0.0006** | - |
| Functional 참조 노드 | +0.0333 | 0.048 |
| Dead 노드 | ~-0.11 | 0.0007 |

#### 핵심 발견: Credibility Bias는 사실상 STATIC
1. **Dynamic 의도 실패**: Working/Failing 간 bias 차이가 0.0006으로, 사실상 상수
2. **Node-level bias만 학습**: Functional 노드(+0.033) vs Dead/Major-fail 노드(-0.112)의 차이는 크지만, 같은 노드 내 시간에 따른 변화는 거의 없음
3. **Attention 반전 현상**: Baseline에서는 Working→high attention, Failing→low attention이지만, Credibility 모델에서는 반전됨

#### 원인 분석
`credibility_net = nn.Linear(model_dim, num_heads)`의 input인 `x`에 **adaptive_embedding** (static per-node, per-timestep 고정 학습 파라미터)이 포함됨. 이 static 신호가 flow 값의 동적 신호를 지배하여 credibility_net이 node identity만 학습.

#### 5ch에서 Attention Collapse 자체가 약화
| 시간대 | 3ch Dead/Func Ratio | 5ch Dead/Func Ratio |
|---|---|---|
| 3am | 2.6x | 2.0x |
| 9am | 11.2x | 0.2x |
| 3pm | 12.3x | 0.0x |
| 9pm | 6.3x | 0.9x |

5ch에서는 occupancy/speed 채널이 implicit sensor health signal 역할을 하여 attention collapse가 이미 크게 완화됨. 따라서 credibility bias의 효과가 제한적.

#### Mixed On/Off 패턴 분석 (Node 151, zero_rate=50.8%)

센서가 [464,0,524,645,0,353,0,0]처럼 on/off를 반복하는 상황에서 spatial attention 분석.

**Attention TO node 151:**
| 조건 | Baseline | Credibility |
|---|---|---|
| All-Working windows | 0.61x uniform | 0.98x uniform |
| All-Failing windows | 0.91x uniform | 1.21x uniform |
| Mixed windows | 1.07x uniform | 1.73x uniform |

**핵심 발견: Credibility bias가 attention collapse를 2배 악화:**
- Functional→zero-flow node attention ratio (50개 functional node 평균):
  - Baseline: **1.83x** (이미 zero 노드에 bias)
  - Credibility: **3.66x** (2배 악화!)

**원인 - Credibility net이 반대로 학습:**
| 노드 유형 | Credibility Bias |
|---|---|
| Dead node | -0.040 (약한 패널티) |
| Major-fail node 151 | -0.024 (더 약한 패널티) |
| Functional node | **-0.438** (강한 패널티!) |

의도: dead에 negative bias → 결과: functional에 더 큰 negative bias. 완전히 반대.

**MAE가 살짝 개선된 이유**: attention 개선이 아닌, dead 노드를 0으로 예측하는 부수효과 (dead raw_mae: 0.659→0.124).

#### 다음 방향
1. **Unmasked loss가 원인**: dead 노드 target=0 예측이 쉬우므로, dead에 attention 주는 게 loss 감소에 유리. Credibility net도 이 방향으로 학습됨
2. **3ch credibility 실험**: Attention collapse가 심한 3ch에서 테스트
3. **Loss 설계 재고**: masked loss 사용 시 credibility의 학습 방향이 달라질 수 있음
4. **Credibility input 개선**: adaptive_embedding 제외하고 flow signal만 사용

---

### 11.5 Context-Aware Self-Supervised Credibility (2026-02-12)

#### 목적
이전 credibility bias가 static한 per-node bias만 학습하고 실제로는 attention collapse를 악화시킨 문제 해결.
Self-supervised masked temporal convolution으로 context-dependent credibility를 학습하여 timestep별로 동적인 attention bias 생성.

#### 설계
- **MaskedTemporalConv**: 1D conv에서 center weight를 0으로 마스킹 → "leave-one-out" context prediction
- **Dual-head architecture**: shared latent → reconstruction head (self-supervised) + credibility head (end-to-end)
- **핵심 아이디어**: `[1,0,1,2,0,0]`에서 0은 high credibility (정상 저교통), `[1110,0,500,0,0,454]`에서 0은 low credibility (센서 고장)
- Reconstruction loss (0.1 weight) + main prediction loss
- Raw input (3ch: flow, occ, speed)에서 직접 계산 (adaptive embedding의 static bias 회피)
- Context module: 677 params (전체 453K의 0.15%)

#### 결과

| Metric | Baseline (5ch) | Credibility Bias | Context Cred |
|---|---|---|---|
| **Overall MAE** | **12.178** | 12.151 | 14.844 (+22%) |
| Functional masked MAE | 13.974 | 13.947 | 17.389 (+24%) |
| Dead raw MAE | - | - | 0.148 |
| Major fail masked MAE | - | - | 12.249 |

#### 상세 분석

**Credibility가 의도와 완전히 반전됨:**

| Category | Credibility (avg) | Recon MAE | Attn Received |
|---|---|---|---|
| Dead | -0.017 (≈0, penalty 없음) | 0.039 (trivial) | 4.47x uniform |
| Major fail | -0.407 | 3.83 | 2.46x uniform |
| Functional | **-8.155** (극심한 penalty) | 71.03 (very hard) | 0.39x uniform |

- **recon_error vs credibility 상관: r = -0.997** → cred head가 "재구성 용이성 = credibility"로 학습
- **zero_rate vs credibility 상관: r = +0.505** → zeros가 많을수록 높은 credibility (의도와 반대)
- **Attention collapse 6x 악화**: Dead/Func ratio 1.93x (baseline) → 11.55x (context cred)

**Per-head 분석:**
- Head 0: Dead=-0.04, Func=-9.39 (잘못된 방향)
- Head 1: Dead=-0.03, Func=-8.56 (잘못된 방향)
- Head 2: Dead=-0.01, Func=+8.82 (**올바른 방향**, 유일)
- Head 3: Dead=+0.01, Func=-23.49 (최악)
- 3:1로 잘못된 방향 압도

**Mixed on/off node (Node 27, 37% zero):**
- Credibility가 timestep별로 변함 (masked conv 작동 확인)
- 하지만 값 범위 -0.12~-0.004 → Functional의 -8.15에 비해 무의미
- Zero/non-zero timestep 간 차이 미미

**Dead node:**
- 완전히 static (모든 timestep -0.0169)
- 재구성도 static (모든 timestep -0.039)

#### 근본 원인
**Shared latent의 구조적 결함**: Reconstruction head와 credibility head가 같은 latent features를 공유.
Reconstruction latent는 "context로부터의 예측 가능성"을 encode하는데, **재구성 용이성 ∝ 1/센서건강도**.
Dead 센서 = trivially predictable (항상 0) → 높은 credibility.
Functional 센서 = 높은 분산, 예측 어려움 → 낮은 credibility.
결과적으로 main task gradient와 결합되어 "dead에 attend하라"는 방향으로 학습.

#### 교훈
1. **Reconstruction quality ≠ sensor credibility**: 재구성이 쉬운 것(dead)과 attention을 받아야 하는 것(functional)은 반대
2. **Shared latent의 한계**: Recon head가 지배하는 latent space에서 cred head가 독립적인 의미를 학습 불가
3. **Pre-softmax bias 위험**: 작은 값(-8)도 softmax 분포를 극적으로 왜곡
4. **Unmasked loss와의 상호작용**: Dead 노드 target=0 예측이 쉬우므로, 모든 학습 신호가 "dead에 attend"를 강화

---

## Sensor Failure 기반 Credibility 접근의 종합 결론 (2026-02-12)

### 왜 이 접근을 시도했는가

**관찰된 문제**: Dead/major-fail 센서가 spatial attention에서 비정상적으로 높은 가중치를 받음 (attention collapse).
Dead 노드는 일관된 zero 입력 → 일관된 key representation → softmax에서 높은 attention weight.

**가설**: 센서 고장 여부를 자동으로 감지하여, 고장 시점에는 해당 노드의 attention을 줄이면 functional 노드의 예측이 개선될 것이다.

**상상한 시나리오**:
```
[1110, 0, 500, 0, 0, 454]  → 주변이 높은데 0 → 센서 고장 → low credibility
[1, 0, 1, 2, 0, 0]         → 주변도 낮음 → 자연스러운 저교통량 → high credibility
```

이 시나리오에서는 temporal context를 보면 "고장인 zero"와 "자연스러운 zero"를 구별할 수 있다고 기대했다.

### 시도한 접근들과 결과

| # | 접근 | 결과 | 실패 원인 |
|---|---|---|---|
| 1 | 노드 제거 (dead/major 삭제) | Functional MAE 악화 (+1.6%) | Dead 노드가 spatial anchor 역할, 제거 시 정보 손실 |
| 2 | Static credibility bias (nn.Linear on embedding) | MAE 12.151 (-0.22%) | 실제로는 static per-node bias만 학습, attention 반전 |
| 3 | Value gating | 실패 | Value gating으로는 attention distribution 수정 불가 |
| 4 | Context-aware SSL credibility (MaskedTemporalConv) | MAE 14.844 (+22%) | Shared latent 결함 + 데이터 구조적 문제 |

### 데이터 분석에서 밝혀진 근본적 문제

#### 1. Zero의 94.8%가 temporal context로 구별 불가능

전체 데이터의 zero 값 중 temporal context(전후 12 timestep)를 분석한 결과:

| Zero 유형 | 비율 | 설명 |
|---|---|---|
| All-zero context (주변도 전부 0) | **94.8%** | Temporal context에 정보 없음 |
| Mixed context (주변에 non-zero 존재) | 5.0% | 구별 가능하나 소수 |
| High-context (주변 평균 > 100) | **0.2%** | 우리가 상상한 시나리오 |

**우리가 상상한 `[1110, 0, 500, 0, 0, 454]` 패턴은 전체 zero의 0.2%에 불과하다.**

#### 2. Major-fail 노드의 실제 failure 패턴

Major-fail 노드(zero_rate 50-90%)가 "intermittent on/off"를 반복할 것이라 예상했으나, 실제로는:

| 패턴 유형 | 예시 | 특징 |
|---|---|---|
| **Block failure** (Node 64) | 수일간 연속 off → 수일간 on | Zero run 중앙값 26 steps (2시간+) |
| **Intermittent** (Node 39) | 짧은 on/off 반복 | Zero run 99.5%가 ≤12 steps |

- Major-fail 노드의 19.3% window가 mixed (on+off 공존)
- **그러나 mixed window의 flow 값이 매우 낮음 (1-6)**, 고교통량 failure가 아닌 저교통량 시간대의 noise
- Block failure의 경우, 전후 context도 모두 zero → temporal reconstruction으로 구별 불가

#### 3. Z-score 정규화가 discriminative signal 압축

| Space | Dead recon error | Functional recon error | 차이 |
|---|---|---|---|
| Raw space | 0.04 | 71.03 | **1776x** |
| Normalized space | 0.157 | 0.140 | **1.12x** (거의 동일) |

Z-score (mean=124.03, std=157.83)로 정규화하면 dead와 functional의 reconstruction error 차이가 사실상 사라진다.

#### 4. Reconstruction ease ∝ 1/sensor health (구조적 역전)

| 센서 유형 | Reconstruction 난이도 | 의도한 credibility | 실제 학습 방향 |
|---|---|---|---|
| Dead (항상 0) | ⭐ 극히 쉬움 | Low (attend 받지 마라) | **High** (잘 예측됨 → 신뢰) |
| Functional (높은 분산) | 😰 어려움 | High (attend 받아라) | **Low** (예측 어려움 → 불신) |

Self-supervised reconstruction objective는 본질적으로 "예측 가능성"을 학습하며, 이는 sensor health와 **정반대 방향**이다.

### 왜 Sensor Failure 기반 접근 자체가 잘못되었는가

1. **Temporal context가 존재하지 않음**: Zero의 94.8%가 all-zero context. Temporal reconstruction으로 "고장 zero"와 "자연스러운 zero"를 구별하는 것 자체가 불가능.

2. **5ch가 이미 implicit credibility 제공**: Occupancy/speed 채널이 sensor health 정보를 암시적으로 전달. Dead 센서는 3채널(flow, occ, speed) 모두 0이므로 모델이 이미 학습 가능. 실제로 5ch에서는 attention collapse가 이미 크게 완화됨 (Dead/Func ratio: 3ch 12.3x → 5ch 0.0x at 3pm).

3. **Reconstruction ≠ Credibility**: Self-supervised reconstruction의 학습 방향이 credibility와 구조적으로 반대. 어떤 reconstruction objective를 사용하든 이 문제를 피할 수 없음.

4. **남은 개선 여지가 극히 작음**: 5ch baseline의 functional MAE가 13.974이고, 5ch에서의 attention collapse가 이미 미미하므로, attention 교정으로 얻을 수 있는 이득이 매우 제한적.

### 결론

> **Temporal reconstruction 기반 sensor credibility는 이 데이터셋에서 구조적으로 작동할 수 없다.**
>
> 1. 데이터 분포 문제: 구별해야 할 zero의 대부분(94.8%)이 temporal context로 구별 불가능
> 2. Objective 역전: Reconstruction ease와 sensor health가 정반대
> 3. 이미 해결된 문제: 5ch 입력이 implicit sensor health signal을 이미 제공
>
> **Sensor failure detection은 이 연구의 올바른 방향이 아니다.**

### 향후 방향 제안 (미확정)

Attention collapse 문제는 5ch에서 이미 크게 완화되었으므로, 다른 각도의 연구가 필요:
- **Spatial context**: Cross-node 관계 활용 (dead 노드 주변의 functional 노드 정보)
- **다른 데이터셋**: Attention collapse가 더 심각한 3ch 데이터셋에서 실험
- **아예 다른 문제**: Sensor failure 대신, data augmentation이나 multi-scale temporal modeling 등

---

## Robust Prediction: Per-Node MAE Distribution Analysis (2026-02-12)

### 목적
Attention collapse / credibility 접근이 실패한 후, "robust prediction" 관점에서 per-node MAE 분포를 분석.
어떤 노드가 왜 예측이 어려운지, task(flow vs speed)에 따라 어려움이 달라지는지 이해.

### Per-Node Flow MAE 분포 (5ch unmasked baseline)

| 통계 | 값 |
|------|-----|
| Mean | 12.18 |
| Median | 6.86 |
| Std | 9.83 |
| Max | 108.53 (node 630) |
| Min | 0.45 (node 230) |
| Skewness | 3.38 |

**Quintile 분석 (loss 기여도):**

| Quintile | MAE 범위 | Loss 기여% |
|----------|----------|-----------|
| Q5 (worst 20%) | 19.5 - 108.5 | 54.9% |
| Q4 | 8.2 - 19.5 | 28.3% |
| Q3 | 3.2 - 8.2 | 14.6% |
| Q2 | 0.7 - 3.2 | 2.1% |
| Q1 (best 20%) | 0.4 - 0.7 | 0.1% |

**핵심**: Worst 20% 노드가 전체 loss의 55%를 차지. Loss imbalance가 극심.

### Scale Effect 분석

Flow prediction에서 MAE가 높은 이유가 단순히 flow 값이 크기 때문인지 검증.

| 노드 그룹 | Mean Flow | MAE | nMAE (MAE/Mean) | MAPE |
|-----------|-----------|-----|-----------------|------|
| Q5 (worst) | 318.2 | 33.5 | 0.105 | 0.144 |
| Q4 | 168.9 | 13.4 | 0.079 | 0.124 |
| Q3 | 74.2 | 5.3 | 0.072 | 0.192 |
| Functional 전체 | 196.5 | 14.0 | **0.599** | 0.187 |

- MAE vs mean_flow 상관: r=0.82 (강한 양의 상관)
- nMAE vs mean_flow 상관: r=0.04 (상관 없음)
- **결론**: Flow scale이 MAE의 주요 원인. nMAE 기준으로는 노드 간 예측 난이도가 균등.

### Speed Prediction 실험

#### ZScoreScaler target_channel 이슈
Speed config에서 `target_channel`을 설정하지 않아 scaler가 flow의 mean/std(124.03/157.83)로 역변환.
모델은 정상 학습되지만 보고 MAE가 flow_std 배수로 부풀려짐.

- 보고 MAE: 165.55 → **실제 MAE: 1.05 mph** (÷157.83)
- MAPE: 0.024 (비율이므로 영향 없음)

#### Per-Node Speed MAE 분포

| 통계 | Flow MAE | Speed MAE (mph) |
|------|----------|-----------------|
| Mean | 12.18 | 1.04 |
| Median | 6.86 | 0.77 |
| Std | 9.83 | 1.14 |
| Max | 108.53 (node 630) | 8.40 (node 630) |
| nMAE (functional) | **0.5988** | **0.0265** |

Speed prediction이 상대적으로 **24배 쉬움** (nMAE 기준).

#### Speed Quintile 분석

| Quintile | Speed Loss 기여% | Flow Loss 기여% |
|----------|-----------------|-----------------|
| Q5 (worst 20%) | 55.3% | 54.9% |
| Q4 | 29.5% | 28.3% |
| Q3 | 14.0% | 14.6% |
| Q2 | 1.2% | 2.1% |
| Q1 (best 20%) | 0.0% | 0.1% |

두 task 모두 **비슷한 loss 불균형** 패턴.

#### Cross-Task 상관관계

- Flow MAE vs Speed MAE: **r=0.76** (강한 양의 상관)
- Worst-10 노드 overlap: **2/10** (node 630, 149)
- 어려운 노드가 부분적으로 공유되지만 task별 특성 존재

### Checkpoint 정보

| 실험 | Path |
|------|------|
| Flow baseline | `checkpoints/STAEformer_5ch_unmasked/SAN_BERNARDINO_30_12_12/9827aaaa.../` |
| Speed prediction | `checkpoints/STAEformer_5ch_speed/SAN_BERNARDINO_30_12_12/4ddb4b5b.../` |

### 저장된 분석 파일

| 파일 | 내용 |
|------|------|
| `eda/robust_prediction/per_node_mae_speed.npy` | Speed per-node MAE (893,), true mph |

### 인사이트

1. **Loss imbalance는 task-agnostic**: Flow와 speed 모두 worst 20%가 ~55% loss 차지
2. **Flow MAE의 주요 원인은 scale**: nMAE 기준으로는 노드 간 난이도 균등
3. **Speed는 상대적으로 쉬운 task**: nMAE 2.65% (speed) vs 59.88% (flow)
4. **Cross-task 난이도 부분 공유**: r=0.76이지만 worst 노드는 대부분 다름
5. **Node 630이 두 task 모두 worst**: 이 노드의 특성 조사 가치 있음

---

## Changelog (Updated)

| Date | Experiment | Result | Notes |
|------|------------|--------|-------|
| 2026-02-12 | Robust prediction per-node analysis | Flow nMAE=0.60, Speed nMAE=0.03 | Loss imbalance ~55% from worst 20%, cross-task r=0.76 |
| 2026-02-12 | Speed prediction (5ch) | True MAE 1.05 mph | ZScoreScaler target_channel bug inflates reported MAE |
| 2026-02-12 | Sensor failure credibility 종합 결론 | 접근 자체 기각 | Temporal context 부재 (94.8%), reconstruction ≠ credibility |
| 2026-02-12 | Context-aware self-supervised credibility | MAE 14.844 (+22%) | Self-supervised cred FAILS badly, corrupts spatial attention |
| 2026-02-12 | Mixed on/off attention analysis | Cred WORSENS collapse 2x | zero/nonzero ratio: baseline 1.83x → cred 3.66x |
| 2026-02-12 | Credibility per-sample analysis | Bias is STATIC | Working/Failing gap=0.0006, node-level only |
| 2026-02-12 | 5ch credibility bias (spatial) | MAE 12.151 | First improvement on functional (13.947, -0.19%) |
| 2026-02-11 | 5ch unmasked no_dead_major (745n) | MAE 12.322 | Functional 14.021 (+0.3% vs baseline) |
| 2026-02-11 | 5ch unmasked no_dead (773n) | MAE 12.361 | Functional 14.194 (+1.6% vs baseline) |
| 2026-02-11 | 5ch unmasked baseline (893n) | MAE 12.178 | Functional 13.974, best unmasked |
| 2026-02-11 | 5ch masked no_dead_major (745n) | MAE 12.456 | Node removal hurts |
| 2026-02-11 | 5ch masked no_dead (773n) | MAE 12.285 | Node removal slightly hurts |
| 2026-02-11 | 5ch masked baseline (893n) | MAE 12.263 | 5ch masked baseline |
| 2026-02-10 | Cross-variable pretrained frozen | MAE 12.15 | Pre-training helps vs scratch but worse than baseline |
| 2026-02-10 | Cross-variable scratch encoder | MAE 12.26 | Control: random init encoder |
| 2026-02-09 | Per-node masked MAE analysis | ALL encoders worse | Dead sensor artifact confirmed |
| 2026-02-09 | Random projection downstream | MAE 11.94, pn_w1% 90.07 | |
| 2026-02-09 | Predictive contrastive downstream | MAE 13.10, pn_w1% 90.19 | |
| 2026-02-01 | TSFormer NaN Fix | Fixed | Added pre_encoder_norm |
| 2026-01-31 | GPTST Pretrain | MAE 117.85 (recon) | |
| 2026-01-30 | Feature masking | MAE 11.71 | Best overall MAE |
| 2026-01-30 | Baseline STAEformer | MAE 12.10 | |

*Last updated: 2026-02-12 (robust prediction analysis added)*

---

## 2026-02-15: Noise Vulnerability Evaluation

### Purpose
Evaluate how robust STAEformer 5ch and STGCN 1ch are to noisy sensor data where noise is invisible (mask channels stay 1). This is the key distinction from missing data: interpolation can fix missing values (mask=0), but cannot fix noisy readings that are trusted by the model (mask=1).

### Noise Types (from FHWA Traffic Detector Handbook)
1. **Gaussian**: Additive noise, severity = noise_std/channel_std
2. **Bias**: Systematic multiplicative under/over-counting (+/-30% or +/-50%)
3. **Stuck**: Sensor frozen at t=0 value for entire window
4. **Drift**: Gradual calibration shift over time window

### Key Results (functional node masked MAE)

| Noise Config | STAEformer 5ch | STGCN 1ch |
|---|---|---|
| Clean baseline | 12.13 | 13.34 |
| Gaussian s=0.3 r=30% | 16.22 (+33.8%) | 15.10 (+13.2%) |
| Gaussian s=1.0 r=30% | 77.32 (+537.6%) | 27.77 (+108.2%) |
| Bias s=0.3 r=30% | 21.16 (+74.5%) | 19.30 (+44.7%) |
| Stuck r=30% | 14.48 (+19.4%) | 16.43 (+23.2%) |
| Drift s=0.3 r=30% | 16.73 (+38.0%) | 17.99 (+34.9%) |

### Critical Insight: Spatial Spillover
STAEformer's attention mechanism aggressively propagates noise to healthy nodes:
- Gaussian s=1.0 r=30%: healthy nodes +449.3% (STAEformer) vs +70.7% (STGCN)
- Bias s=0.3 r=10%: healthy nodes +43.6% (STAEformer) vs +0.4% (STGCN)

### Noise vs Missing Data
- Missing data with interpolation: +1.8% at p=30%
- Gaussian noise s=0.3 r=30%: +33.8% (STAEformer), +13.2% (STGCN)
- **Noise is 10-20x more damaging than missing data**, confirming the pivot from missing-robust to noise-robust representation learning

### Conclusion
- STAEformer is more vulnerable than STGCN due to spatial attention spillover
- Noise is fundamentally harder than missing data because the model cannot identify which inputs are corrupted
- A denoising encoder that can detect and correct noisy inputs BEFORE they reach the forecasting model would be valuable
- Script: `experiments/eval_noise_vulnerability.py`
- Results: `experiments/noise_vulnerability_results/results.json`

---

## Denoising Encoder: Noise-Robust Representation Learning (2026-02-15)

### Motivation
Noise vulnerability experiments showed noise is 10-20x more damaging than missing data, and STAEformer is especially vulnerable due to spatial attention spillover. A denoising encoder placed BEFORE the forecasting model can clean noisy inputs transparently.

### Architecture: DenoisingEncoder
- **Two-stage**: Temporal Dilated Conv → Spatial Graph Conv
- **Replace strategy**: D_out = D_in = 5 (transparent filter, not bottleneck)
- **Physical channel passthrough**: Only denoises channels [0,1,2] (flow, occ, speed); tod/dow pass through unchanged
- **Temporal**: 4 layers of dilated causal convolutions (dilation 1,2,4,8) with residual connections
- **Spatial**: 1 layer of k-nearest-neighbor graph convolution (k=10)
- **Parameters**: ~17K (lightweight)
- **Implementation**: `baselines/ContextContrastive/arch/denoising_encoder.py`

### Pre-training
- **Objective**: Inject noise (gaussian/bias/stuck/drift) on physical channels, reconstruct clean signal
- **Loss**: L_recon (MSE on noisy→clean) + L_passthrough (MSE on clean→clean, prevents degradation)
- **Training**: 30 epochs, val loss 0.88 → 0.18
- **Config**: `baselines/ContextContrastive/DenoisingPretrain_SAN_BERNARDINO.py`
- **Checkpoint**: `checkpoints/DenoisingPretrain/SAN_BERNARDINO_30_12_12/`

### Downstream Training
| Model | Config | Clean MAE | Baseline MAE | Delta |
|---|---|---|---|---|
| STAEformer + Denoising | `STAEformer/SAN_BERNARDINO_5ch_denoising.py` | **12.05** | 12.13 | **-0.08 (improved!)** |
| STGCN + Denoising | `STGCN/SAN_BERNARDINO_denoising.py` | 13.38 | 13.34 | +0.04 (neutral) |

**Key**: Clean performance preserved (no degradation from encoder).

### Noise Robustness Results (ALL FUNCTIONAL nodes, % degradation)

| Noise Config | STAEformer baseline | STAEformer+Denoising | Improvement | STGCN baseline | STGCN+Denoising | Improvement |
|---|---|---|---|---|---|---|
| **Gaussian s=0.3 r=30%** | +33.8% | **+9.6%** | **3.5x** | +13.2% | **+9.6%** | **1.4x** |
| **Gaussian s=0.5 r=30%** | +191.8% | **+31.7%** | **6.0x** | +37.9% | **+24.2%** | **1.6x** |
| **Gaussian s=1.0 r=30%** | +537.6% | **+133.5%** | **4.0x** | +108.2% | **+63.0%** | **1.7x** |
| **Bias s=0.3 r=30%** | +74.5% | **+38.9%** | **1.9x** | +44.7% | **+28.9%** | **1.5x** |
| **Bias s=0.5 r=30%** | +155.5% | **+88.6%** | **1.8x** | +85.7% | **+78.7%** | **1.1x** |
| **Stuck r=30%** | +19.4% | **+18.0%** | **1.1x** | +23.2% | **+22.1%** | **1.1x** |
| **Drift s=0.3 r=30%** | +38.0% | **+21.2%** | **1.8x** | +34.9% | **+20.9%** | **1.7x** |
| **Drift s=0.5 r=30%** | +74.0% | **+46.8%** | **1.6x** | +73.9% | **+59.6%** | **1.2x** |

### Spatial Spillover Reduction (HEALTHY nodes only)

| Noise Config | STAEformer baseline | STAEformer+Denoising | Improvement |
|---|---|---|---|
| Gaussian s=0.3 r=30% | +20.7% | **+0.3%** | **69x** |
| Gaussian s=0.5 r=30% | +155.1% | **+3.2%** | **48x** |
| Gaussian s=1.0 r=30% | +449.3% | **+17.5%** | **26x** |
| Bias s=0.5 r=30% | +84.0% | **+16.4%** | **5x** |
| Drift s=0.5 r=30% | +11.0% | **+8.1%** | **1.4x** |

### Key Findings

1. **Clean performance preserved**: STAEformer denoising actually improves clean MAE by 0.08. STGCN essentially unchanged (+0.04).

2. **Massive noise robustness gain for STAEformer**: 1.8-6.0x reduction in degradation across all noise types. Greatest improvement for Gaussian noise (most common sensor fault).

3. **STGCN also benefits**: 1.1-1.7x improvement despite already being more inherently robust.

4. **Spatial spillover nearly eliminated**: STAEformer's biggest weakness (noise spreading through spatial attention) is reduced by 26-69x for Gaussian noise. Healthy node degradation drops from +155% to +3.2% at s=0.5, r=30%.

5. **Stuck noise hardest to denoise**: Only 1.1x improvement - makes sense because stuck sensors repeat a plausible value (first timestep), making them hard to distinguish from clean data.

6. **Denoising encoder benefits STAEformer more than STGCN**: STAEformer was more vulnerable → more room for improvement. The encoder effectively compensates for spatial attention's vulnerability to noise.

### Conclusion
The denoising encoder successfully achieves noise-robust prediction without sacrificing clean performance. It is most effective for the most vulnerable model (STAEformer) and against the most common noise types (Gaussian, bias, drift). The lightweight design (~17K params) adds negligible computational overhead.

---

*Last updated: 2026-02-15 (denoising encoder experiment results added)*

---

## 2026-02-18: Paper RQ Experiments (Comprehensive)

### Research Questions

| RQ | Question | Experiments |
|---|---|---|
| RQ1 | How does sensor noise impact prediction across channel configurations? | 1ch vs 3ch vs 5ch baseline + noise perturbation |
| RQ2 | Does the denoising encoder improve robustness? | STAEformer + denoising encoder, multi-county |
| RQ3 | What are the key components of the encoder? (Ablation) | A2/A3/A7/B2 |
| RQ4 | Does the encoder generalize to unseen noise types? | Cross-noise generalization |
| RQ5 | Is the approach model-agnostic? | STAEformer vs STGCN |

### RQ1: Channel Configurations
- **STAEformer 5ch** (baseline): MAE=12.2626 (existing)
- **STAEformer 1ch** (flow+tod+dow): TRAINING (epoch 1/30)
- **STGCN 3ch** (flow+occ+speed): TRAINING (epoch 1/30)
- **STGCN 1ch** (existing): MAE=14.0864

### RQ2: Denoising Encoder
- **STAEformer 5ch + denoising**: MAE=12.1261 (existing, improves baseline)
- **CONTRA_COSTA 5ch baseline**: PENDING
- **CONTRA_COSTA 5ch + denoising**: PENDING (pretrain in progress)

### RQ3: Ablation Study

All ablations use v2 architecture (hidden_dim=64, residual=True) except:
- A7: residual_connection=False
- B2: hidden_dim=32

| Variant | Description | Pretrain Status | Downstream Status |
|---|---|---|---|
| **Full model** | temporal(4L) + spatial(1L) + residual, h=64 | Done (existing denoising v2) | MAE=12.9196 (noisy training) |
| **A2: MLP only** | Per-node MLP, no temporal/spatial conv | Done | Queued (auto-launch after A7) |
| **A3: Temporal only** | spatial_layers=0 | Done | Queued (auto-launch after A7) |
| **A7: No residual** | residual_connection=False | Done | Epoch 23/30 (ETA ~23:00) |
| **B2: Hidden=32** | Smaller hidden dim (32 vs 64) | Done | Queued (auto-launch after A7) |

### RQ4: Cross-Noise Generalization

Pre-training with subset of noise types, evaluate on all:
- **Common noise** (gaussian/bias/drift): Pretrain epoch 4/30 (ETA Feb 19 ~20:00)
- **Structural noise** (stuck/dead): Pretrain epoch 17/30 (ETA ~23:30 today)
- Downstream: Auto-launch script waiting for pretrains to finish

### RQ5: Model Agnosticity

| Model | Baseline MAE | + Denoising MAE | Delta |
|---|---|---|---|
| STAEformer 5ch | 12.2626 | 12.1261 | -0.14 |
| STGCN 1ch | 14.0864 | 14.1063 | +0.02 |

Channel baselines in progress:
- **1ch STAEformer**: Epoch 5/30 (ETA Feb 19 ~18:00)
- **3ch STGCN**: Epoch 4/30 (ETA Feb 19 ~22:00)

Multi-county generalization:
- **CONTRA_COSTA pretrain**: Epoch 4/30 (ETA Feb 19 ~21:00)
- CONTRA_COSTA baseline + downstream: Auto-launch waiting

### Reliability Estimation (Completed)

Dual-output encoder (denoised + reliability score) to signal downstream model about untrustworthy nodes:
- Pretrain: reliability-weighted loss with β=1.0
- Downstream: STAEformerReliability uses reliability as spatial attention pre-softmax bias
- **Result: MAE=12.72** (worse than denoising-only 12.13, and baseline 12.26)
- Conclusion: Reliability estimation adds noise to the signal rather than helping

### Comprehensive Noise Vulnerability Results

Results from `experiments/noise_vulnerability_results/results.json`:

| Model | Clean MAE | gauss s=0.3 r=30% | gauss s=0.5 r=30% | stuck r=30% | dead r=30% |
|---|---|---|---|---|---|
| STAEformer 5ch (baseline) | 12.26 | 15.45 | 19.31 | 13.47 | 13.75 |
| + denoising (frozen enc) | 12.13 | 12.78 | 13.45 | 12.47 | 12.73 |
| + denoi + noisy training | 12.40 | 12.37 | 13.79 | 12.34 | 12.67 |
| STGCN 1ch (baseline) | 14.09 | 17.48 | 22.51 | 15.15 | 15.51 |
| + denoising | 14.11 | 14.64 | 15.23 | 14.18 | 14.28 |
| + denoi + noisy training | 14.17 | 14.23 | 15.38 | 14.15 | 14.28 |

Key findings:
- Denoising encoder provides **up to 14x reduction** in noise-induced degradation
- Noisy training improves Gaussian robustness but slightly hurts clean MAE
- STGCN also benefits from denoising (model-agnostic)

### Experiment Execution Status

8 experiments running on GPU 1 (~95 GB used):
- A7 downstream: epoch 23/30
- 1ch STAEformer baseline: epoch 5/30
- 3ch STGCN baseline: epoch 4/30
- Cross-noise common pretrain: epoch 4/30
- Cross-noise structural pretrain: epoch 17/30
- CONTRA_COSTA pretrain: epoch 4/30
- SAN_BERNARDINO_2022_Q1 (from ray queue): running
- Auto-launch scripts: 2 waiting (A2/A3/B2 downstream + cross-noise downstream)

---

## Cross-Dataset Generalization: CONTRA_COSTA (2026-02-25)

### Dataset Comparison

| | SAN_BERNARDINO | CONTRA_COSTA |
|---|---|---|
| Nodes | 893 | 773 |
| Functional | 745 (83.4%) | 634 (82.0%) |
| Dead | 120 (13.4%) | 104 (13.5%) |
| Major fail | 28 (3.1%) | 35 (4.5%) |
| Mean flow | 120.27 | 121.13 |
| **Adj mean weight** | **0.739** | **0.396** |
| Adj edges > 0.5 | 81.5% | 35.1% |
| Neighbor correlation | 0.673 | 0.639 |
| Spatial coupling | 0.818 | 0.803 |
| Dead neighbor weight | 84.9 | 36.7 |

**Key structural difference**: CC has ~half the adjacency weight of SB (0.40 vs 0.74), despite similar flow statistics and sensor health distributions.

### CONTRA_COSTA 2×2 Factorial Results (30% corruption rate, excl. dead)

| Model | Clean MAE | Gaussian | Bias | Stuck | Drift | **Avg Deg** |
|---|---|---|---|---|---|---|
| Baseline (E0+N0) | 12.39 | +19.2% | +74.0% | +20.8% | +43.3% | **39.3%** |
| Denoising (E1+N0) | 12.32 | +17.1% | +75.2% | +19.8% | +42.7% | **38.7%** |
| Noisy (E0+N1) | 12.47 | +6.5% | +36.3% | +14.5% | +17.9% | **18.8%** |
| Denoising+Noisy (E1+N1) | 13.49 | +5.4% | +34.0% | +11.1% | +16.4% | **16.7%** |

### SAN_BERNARDINO 2×2 for comparison

| Model | Clean MAE | **Avg Deg** |
|---|---|---|
| Baseline | 12.26 | **41.8%** |
| Denoising | 12.13 | **22.0%** |
| Noisy | 12.40 | **16.7%** |
| Denoising+Noisy | 12.40 | **16.7%** |

### Cross-Dataset Comparison Summary

| Strategy | SB Reduction | CC Reduction |
|---|---|---|
| Denoising encoder | -47% (41.8→22.0) | **-1.5%** (39.3→38.7) |
| Noisy training | -60% (41.8→16.7) | **-52%** (39.3→18.8) |
| Combination | -60% (41.8→16.7) | **-58%** (39.3→16.7) |

### Key Insights

1. **Noisy training is universally effective**: Both datasets achieve ~50-60% reduction in avg degradation. This is the robust, dataset-agnostic component.

2. **Denoising encoder effectiveness depends on spatial coupling**: SB's strong adjacency (mean 0.74) enables spatial denoising — corrupted nodes' signals are diluted by strong neighbor connections. CC's weak adjacency (mean 0.40) limits this mechanism, making the encoder nearly useless alone (-1.5% vs -47%).

3. **Both datasets converge to ~16.7% avg degradation**: With the best configuration, the final noise vulnerability is similar regardless of dataset structure. The paths differ (SB: encoder helps, CC: encoder doesn't help alone) but the destination is the same.

4. **CC denoising+noisy has a clean MAE penalty**: 13.49 vs 12.47 for noisy-only (+1.02 MAE). This is because the frozen v1 encoder (residual_connection=False) hurts clean performance when combined with noise augmentation. The noisy-only model (12.47) may be the better practical choice for CC.

5. **Spillover mechanism confirmed**: CC baseline spillover is 4x lower than SB (Gaussian: 5.4% vs 20.3%), directly explained by the 2x weaker adjacency weights. The denoising encoder's primary mechanism — spatial denoising via neighbor information — requires strong spatial connections to be effective.

### Files
- Dataset comparison: `experiments/compare_datasets.py`
- CC noise eval: `experiments/eval_contra_costa.py`
- Results: `experiments/noise_vulnerability_results/contra_costa_results.json`
- Configs: `baselines/STAEformer/CONTRA_COSTA_5ch_{noisy,denoising_noisy}.py`

---

## Denoising Encoder v2 on CONTRA_COSTA (2026-02-25)

### 목적
CC에서는 v1 encoder(hidden=32, no residual)만 테스트됨. v1은 clean MAE penalty가 크고 (+1.02 in combo), encoder 단독 robustness 효과도 없었음 (-1.5%). SB에서 v2(residual+hidden=64)가 v1 대비 큰 개선을 보였으므로, CC에서도 v2 테스트.

### v2 vs v1 차이점
- `hidden_dim`: 64 (v1: 32)
- `residual_connection`: True (v1: False)
- `noise_severity_range`: (0.1, 1.5) (v1: (0.1, 0.5))
- `noise_rate_range`: (0.1, 0.7) (v1: (0.1, 0.5))

### Pretrain
- Val loss: 0.168 → 0.040 (SB v2는 ~0.05까지)
- Checkpoint: `checkpoints/DenoisingPretrainV2/CONTRA_COSTA_30_12_12/`

### Results

| Model | Clean MAE | Gaussian | Bias | Stuck | Drift | Avg Deg |
|---|---|---|---|---|---|---|
| Baseline | 12.39 | +19.2% | +74.0% | +20.8% | +43.3% | **39.3%** |
| v1 encoder only | 12.32 | +17.1% | +75.2% | +19.8% | +42.7% | **38.7%** |
| **v2 encoder only** | **11.98** | +27.8% | +72.2% | +22.6% | +42.5% | **41.3%** |
| Noisy only | 12.47 | +6.5% | +36.3% | +14.5% | +17.9% | **18.8%** |
| v1 encoder+noisy | 13.49 | +5.4% | +34.0% | +11.1% | +16.4% | **16.7%** |
| **v2 encoder+noisy** | **13.21** | +4.8% | +30.5% | +11.2% | +16.8% | **15.8%** |

### Key Insights

1. **v2 encoder only가 CC 최고 clean MAE 달성 (11.98)**: baseline 12.39 대비 -0.41. Residual architecture가 clean input을 보존하면서 약간의 regularization 효과.

2. **v2 encoder만으로 robustness 개선은 안됨 (41.3% vs baseline 39.3%)**: CC의 약한 adjacency(0.40)에서는 spatial denoising이 작동하지 않음. SB(0.74)와 동일한 패턴.

3. **v2+noisy가 CC 최고 robustness (15.8%)**: v1+noisy(16.7%) 대비 0.9pp 개선, clean MAE도 13.21 vs 13.49로 개선(-0.28).

4. **Residual이 clean MAE penalty 완화**: v2+noisy 13.21 vs v1+noisy 13.49 (-0.28). 여전히 noisy-only(12.47) 대비 penalty 있지만 v1보다 줄어듦.

5. **Gaussian spillover 증가 주의**: v2 encoder only에서 healthy spillover +13.8% (baseline +5.4%, v1 +5.9%). v2의 stronger pretrain noise가 encoder 행동을 변화시켜 일부 noise를 오히려 증폭? v2+noisy에서는 -0.2%로 해결됨.

### Configs
- Pretrain: `baselines/ContextContrastive/CONTRA_COSTA/pretrain_denoising_v2.py`
- Downstream E2+N0: `baselines/STAEformer/CONTRA_COSTA_5ch_denoising_v2.py`
- Downstream E2+N1: `baselines/STAEformer/CONTRA_COSTA_5ch_denoising_v2_noisy.py`

*Last updated: 2026-02-25*

