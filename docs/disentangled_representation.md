# SICPL: Scale-Invariant Contrastive Pattern Learning for Drift-Robust Traffic Forecasting

> **Research Hypothesis**: "교통 데이터의 Concept Drift는 주로 저주파수(Scale) 성분에서 발생하며, 고주파수(Pattern) 성분은 상대적으로 불변한다. 따라서 Contrastive Learning으로 패턴의 Invariance를 학습하고, 별도의 Scale 예측기를 통해 Drift에 대응한다."

## 1. Motivation

### 1.1 Problem
교통 데이터의 concept drift는 크게 두 가지 요소로 분해할 수 있다:

1. **Scale drift (저주파)**: 교통량의 절대 크기 변화 (e.g., 폭우로 전체 유량 -15%)
2. **Pattern drift (고주파)**: 시공간 변동 패턴 변화 (e.g., 출퇴근 피크 시간대 변경, 경로 변경)

### 1.0 Frequency Perspective (문헌 근거)
이 가설은 최근 연구들에 의해 뒷받침된다:
- **Deep Frequency Derivative Learning (IJCAI 2024)**: 기존 normalization은 zero-frequency (DC) component를 조작 → 저주파가 분포 정보의 주 담체
- **Concept Drift in Atmospheric Data (2024)**: 기상 시계열에서 drift는 저주파 변동이 지배적 (arXiv:2511.19638)
- **Low-High Frequency Network (2025)**: 교통의 저주파 = 안정적 주기성(일/주간), 고주파 = 외부 요인에 의한 단기 변동
- **Robformer (2024)**: Trend component(저주파)가 distribution shift에 더 취약
- **TrafficTL (2023)**: Periodicity(주파수) 기반 feature가 도시 간 transfer에 효과적 → 주파수 영역의 domain invariance

### 1.2 Key Observation
SAN_BERNARDINO 2022-2024 분석에서:

| 지표 | 2022 | 2023 (대기강) | 2024 |
|------|------|-------------|------|
| Mean flow (functional) | 165.75 | 151.82 (-8.4%) | 169.43 (+2.2%) |
| Daily profile correlation (vs 2022) | 1.00 | 0.894 | 0.981 |
| Cross-year MAE 증가율 (vs self) | - | +125% | +29% |

**핵심 발견**:
- Scale은 2023에서 크게 변함 (-8.4%)
- 그러나 Pattern 상관은 여전히 높음 (r=0.894)
- 현재 STAEformer의 MAE 증가(+125%)는 scale shift에 과도하게 민감하기 때문

> **Scale과 Pattern이 분리되면, Pattern-only로 예측하고 Scale은 별도로 조정하여 drift-robust 예측이 가능하다.**

### 1.3 왜 기존 모델이 Scale에 취약한가?

STAEformer 구조:
```
Input (B, T, N, C) → Embedding → Temporal Attn → Spatial Attn → FFN → Output
```

- Raw flow 값이 embedding에 직접 들어감
- Attention score가 flow magnitude에 영향받음
- Z-score normalization이 **전체 training set 기준** → test 분포 다르면 normalize 실패
- **결과**: scale이 다르면 전체 representation이 왜곡

---

## 2. Story

### 2.1 Problem Statement
기존 traffic forecasting 모델은 time series의 **scale과 pattern을 entangled하게** 학습한다.
이로 인해 scale shift (기상 재해, 계절 변동, 정책 변화 등)가 발생하면 예측 성능이 급격히 저하된다.

### 2.2 Proposed Method
**Scale-Pattern Disentanglement (SPD)**: 모델 내부에서 scale과 pattern을 명시적으로 분리하여 학습한다.

1. **Scale Extractor**: 입력 데이터에서 scale 정보 추출 (노드별/시간별 mean, variance)
2. **Pattern Encoder**: Scale-normalized 데이터에서 temporal/spatial pattern 학습
3. **Scale-Conditioned Decoder**: Pattern representation에 scale 정보를 conditioning하여 최종 예측

### 2.3 Key Insight: RevIN과의 관계

RevIN (Reversible Instance Normalization, ICLR 2022)은 가장 간단한 scale-pattern 분리:
```
x_norm = (x - μ_x) / σ_x          # Scale 제거
ŷ_norm = Model(x_norm)             # Pattern 예측
ŷ = ŷ_norm * σ_x + μ_x           # Scale 복원 (입력과 동일한 scale 가정)
```

**RevIN의 한계**: 출력의 scale을 입력의 scale과 동일하다고 가정한다.
하지만 실제로는 출력 시점의 scale이 입력과 다를 수 있다 (비정상 시계열의 본질).

**우리의 개선**: Scale을 단순 복사가 아니라 **예측**하고, Pattern과 독립적으로 조합한다.

### 2.4 Novelty

| Method | Scale 처리 | Pattern 처리 | Scale-Pattern 관계 |
|--------|-----------|-------------|-------------------|
| **Standard** | Entangled | Entangled | 구분 없음 |
| **RevIN** | Remove & restore | 중간에만 분리 | 입력=출력 scale 가정 |
| **SAN** (NeurIPS 2023) | Future statistics 예측 | 별도 처리 | 선형 결합 |
| **CoST** (ICLR 2022) | Trend component | Seasonal component | Additive 분해 |
| **Ours (SPD)** | 명시적 scale encoding | Scale-free pattern learning | 조건부 생성 (곱셈) |

### 2.5 Key References

**Non-stationary Time Series:**
- RevIN (ICLR 2022): Reversible instance normalization
- Non-stationary Transformers (NeurIPS 2022): De-stationary attention
- SAN (NeurIPS 2023): Slice-and-normalize, future statistics prediction
- Dish-TS (AAAI 2023): Distribution shift handling

**Disentangled Representation:**
- β-VAE: Disentangled latent factors
- CoST (ICLR 2022): Seasonal-trend disentanglement via contrastive
- TDRL (NeurIPS 2023): Temporal disentangled representation learning

**Traffic-Specific:**
- STNorm: Spatial-temporal normalization for traffic
- DIDA (NeurIPS 2022): Variant/invariant pattern disentanglement in dynamic graphs

---

## 3. Methodology

### 3.1 Overall Architecture

```
Input x (B, T_in, N, C)
    │
    ├──→ Scale Extractor ──→ s = (μ, σ, trend) per node  ──→ Scale Embedding
    │                                                              │
    ├──→ Instance Normalize ──→ x_norm (B, T_in, N, C)            │
    │                              │                               │
    │                    Pattern Encoder (STAEformer)               │
    │                              │                               │
    │                    z_pattern (B, N, D)                        │
    │                              │                               │
    │                    Scale-Conditioned Decoder ←────────────────┘
    │                              │
    └──→ Prediction ŷ (B, T_out, N, 1)
```

### 3.2 Scale Extractor (Metadata-Guided)

각 노드별로 입력 window에서 scale 정보 추출. **핵심 개선**: 단순 flow 통계뿐 아니라 **외부 변수(Weather, Calendar)**를 함께 활용.

#### 3.2.1 Basic Scale Extractor (flow statistics only)

```python
class ScaleExtractor(nn.Module):
    def __init__(self, d_scale, use_metadata=False, n_weather_features=0):
        n_features = 4 + n_weather_features  # base: mean, std, slope, last_value
        self.scale_encoder = nn.Sequential(
            nn.Linear(n_features, d_scale),
            nn.ReLU(),
            nn.Linear(d_scale, d_scale)
        )
        self.use_metadata = use_metadata

    def forward(self, x, weather=None):
        # x: (B, T, N, C) - raw input
        flow = x[:, :, :, 0]  # (B, T, N)

        # Per-node scale statistics
        mu = flow.mean(dim=1)    # (B, N)
        sigma = flow.std(dim=1)  # (B, N)

        # Trend slope
        t = torch.arange(flow.shape[1], device=flow.device).float()
        t_norm = (t - t.mean()) / (t.std() + 1e-8)
        slope = (flow * t_norm[None, :, None]).mean(dim=1)  # (B, N)

        last_val = flow[:, -1, :]  # (B, N)

        features = [mu, sigma, slope, last_val]

        # Metadata-guided: weather features
        if self.use_metadata and weather is not None:
            features.append(weather)  # (B, N, n_weather)

        scale_features = torch.cat([f.unsqueeze(-1) if f.dim()==2 else f for f in features], dim=-1)
        s = self.scale_encoder(scale_features)  # (B, N, d_scale)

        return s, mu, sigma
```

#### 3.2.2 Metadata-Guided Scale Estimation ⭐

**핵심 아이디어**: 교통량의 Scale은 **기상 데이터(강수량), 요일, 공휴일**에 매우 민감하다.
"비가 오면 Scale이 낮아질 것"이라는 **inductive bias**를 모델에 학습시킨다.

**문헌 근거**:
- Heavy rain → 10-17% capacity 감소, 4-7% speed 감소
- Heavy snow → 19-27% capacity 감소, 11-15% speed 감소
- 2023 대기강 → SAN_BERNARDINO 14-17% flow 감소 (우리 데이터 확인)

**Weather Features**:
| Feature | Source | Resolution | 설명 |
|---------|--------|-----------|------|
| Precipitation (mm/hr) | NOAA API | Hourly → 5min interpolation | 강수량 (가장 중요) |
| Temperature (°C) | NOAA API | Hourly | 기온 |
| Wind speed (m/s) | NOAA API | Hourly | 풍속 |
| Visibility (km) | NOAA API | Hourly | 가시거리 |

**Calendar Features** (이미 tod/dow로 부분 포함):
| Feature | 설명 |
|---------|------|
| Holiday flag | 공휴일 여부 |
| School session | 학기 중 여부 |
| Special event | 대규모 이벤트 (optional) |

**California Weather Data Source**:
- **NOAA Weather API**: `api.weather.gov/gridpoints/{office}/{gridX},{gridY}/forecast/hourly`
- ~2.5km grid resolution, 무료/open data
- SAN_BERNARDINO 893 센서 → 가장 가까운 NOAA grid point 매칭
- **NOAA Climate Data Online (CDO)**: 과거 데이터 (ISD, ASOS, METAR)

```python
# Weather data integration pipeline
class MetadataGuidedScaleEstimator(nn.Module):
    def __init__(self, d_scale=32, n_weather=4, n_calendar=3):
        self.flow_encoder = nn.Linear(4, d_scale // 2)      # flow statistics
        self.weather_encoder = nn.Linear(n_weather, d_scale // 4)  # weather
        self.calendar_encoder = nn.Linear(n_calendar, d_scale // 4) # calendar
        self.fusion = nn.Sequential(
            nn.Linear(d_scale, d_scale),
            nn.ReLU(),
            nn.Linear(d_scale, d_scale)
        )

    def forward(self, flow_stats, weather_features, calendar_features):
        z_flow = self.flow_encoder(flow_stats)          # (B, N, d/2)
        z_weather = self.weather_encoder(weather_features)  # (B, N, d/4)
        z_calendar = self.calendar_encoder(calendar_features) # (B, N, d/4)
        z_scale = self.fusion(torch.cat([z_flow, z_weather, z_calendar], dim=-1))
        return z_scale
```

**장점**: 2023년 대기강 같은 상황에서, 강수량 정보가 Scale 예측을 직접 도와줌.
**실험 전략**: Weather 없는 버전 → Weather 포함 → ablation으로 기여도 확인.

### 3.3 Pattern Encoder

Instance-normalized 데이터로 scale-free pattern 학습:

```python
class PatternEncoder(nn.Module):
    # 기존 STAEformer를 그대로 사용하되, normalized input을 받음
    def forward(self, x_norm):
        # x_norm: (B, T, N, C) - instance normalized
        z_pattern = self.staeformer(x_norm)  # (B, N, D)
        return z_pattern
```

### 3.4 Scale-Conditioned Decoder

Pattern과 Scale을 결합하여 최종 예측:

```python
class ScaleConditionedDecoder(nn.Module):
    def __init__(self, d_pattern, d_scale, T_out):
        self.pattern_proj = nn.Linear(d_pattern, T_out)
        self.scale_gate = nn.Sequential(
            nn.Linear(d_scale, d_pattern),
            nn.Sigmoid()
        )
        self.scale_bias = nn.Linear(d_scale, T_out)

    def forward(self, z_pattern, z_scale, mu, sigma):
        # Modulate pattern with scale
        gate = self.scale_gate(z_scale)        # (B, N, D)
        z_modulated = z_pattern * gate          # Scale-aware pattern

        # Predict normalized output
        y_norm = self.pattern_proj(z_modulated) # (B, N, T_out)

        # Scale-conditioned de-normalization
        scale_offset = self.scale_bias(z_scale) # (B, N, T_out)
        y = y_norm * sigma.unsqueeze(-1) + mu.unsqueeze(-1) + scale_offset

        return y.transpose(1, 2).unsqueeze(-1)  # (B, T_out, N, 1)
```

### 3.5 Loss Functions

#### Main Loss
```
L_forecast = MAE(ŷ, y_true)
```

#### Disentanglement Losses (optional, 더 강한 분리를 위해)

**Scale Prediction Loss**: Scale embedding으로 평균 유량 예측
```
L_scale = MAE(ScalePredictor(z_scale), mean_flow_target)
```

**Pattern Independence Loss**: Pattern representation이 scale에 무관하도록
```
L_indep = |corr(z_pattern, log(mu))|  # Pattern-Scale 상관 최소화
```

**Total Loss**:
```
L_total = L_forecast + α * L_scale + β * L_indep
```
- α = 0.1, β = 0.01 (hyperparameters)

### 3.6 추가 아이디어

#### 3.6.1 Multi-Channel Scale Decomposition

Flow뿐만 아니라 occupancy, speed 채널도 별도 scale 추출:
- Flow scale: 유량 크기
- Occupancy scale: 도로 점유율 기준
- Speed scale: 평균 속도 기준

각 채널의 scale이 독립적으로 변할 수 있으므로, multi-channel scale을 유지.

#### 3.6.2 Temporal Scale Dynamics

입력 window 내에서도 scale이 변할 수 있음 (점진적 악화 등):
- Sliding window scale: 매 1시간마다 local scale 추출
- Temporal scale transition 예측: 현재 → 미래 scale 변화 방향

#### 3.6.3 Cross-Year CL과의 결합 (핵심!)

Cross-Year Contrastive Learning과 자연스럽게 결합:
```
z_pattern에 contrastive loss 적용:
  - 같은 node, 다른 year의 z_pattern은 가까워야 함 (pattern은 year-invariant)

z_scale은 contrastive에 포함하지 않음:
  - scale은 year마다 달라도 됨
```

이렇게 하면:
- Contrastive loss가 pattern의 year-invariance를 강화
- Scale embedding은 자유롭게 year-specific 정보를 인코딩
- **두 아이디어가 상호 보완적으로 작동**

#### 3.6.4 Adaptive Scale at Test Time

Test time에 scale이 변할 때:
- Scale Extractor는 입력 데이터에서 직접 scale을 추출하므로, 별도 adaptation 불필요
- Pattern은 학습된 것을 재사용, Scale만 새 데이터에서 추출
- → **자연스러운 Test-Time Adaptation 효과**

---

## 4. 모델 학습 이전 검증 (Pre-experiment Validation)

실제 모델 전체를 학습하기 전에 아이디어의 타당성을 증명할 수 있는 **Quick & Dirty** 실험들.

### 4.1 Upper Bound 확인 (Oracle Test) ⭐ 가장 중요 ✅ 완료

**방법**: 모델의 baseline 예측에 post-hoc additive re-centering 적용.
각 노드의 예측 평균을 target의 실제 per-node 평균으로 교체.

```python
# Target Oracle: re-center prediction to match true future per-node mean
pred_node_mean = preds.mean(axis=1)     # (N, 893, 1) - predicted mean
target_node_mean = test_y.mean(axis=1)  # (N, 893, 1) - actual future mean
preds_oracle = preds - pred_node_mean + target_node_mean
```

#### Oracle Test 결과 (SAN_BERNARDINO)

| 평가 유형 | Baseline MAE | Oracle MAE | 개선율 |
|----------|-------------|------------|--------|
| Self (동일 연도) 평균 | ~10.4 | ~7.8 | **+24.7%** |
| Cross (다른 연도) 평균 | ~20.2 | ~9.8 | **+50.4%** |

| Train→Test | Baseline | Oracle | 개선율 |
|------------|----------|--------|--------|
| 2022→2023 | 22.43 | 9.82 | **+56.2%** |
| 2022→2024 | 12.83 | 8.10 | **+36.9%** |
| 2023→2022 | 25.12 | 10.45 | **+58.4%** |
| 2023→2024 | 24.20 | 9.89 | **+59.2%** |
| 2024→2022 | 13.15 | 8.60 | **+34.6%** |
| 2024→2023 | 23.33 | 10.00 | **+57.1%** |

**기타 oracle 변형 결과**:
- Global Additive/Multiplicative: -2.7% / -1.3% (연도 간 global mean/std가 거의 동일하므로 무효)
- Per-Node Multiplicative: 폭주 (dead node의 0→non-zero 전환으로 ratio 폭발)
- Input-based RevIN: 구조적으로 부적합 (12-step window에서 input scale ≠ output scale, 상세: `docs/sub/revin_analysis.md`)

> 분석 코드: `eda/concept_drift/revin_test_v2.py`
> 결과 파일: `eda/concept_drift/revin_test_v2_results.json`

**핵심 발견**:
1. **Global scale은 변하지 않는다**: 연도 간 global mean 차이 <4 (119.48 vs 119.56 vs 123.23)
2. **Per-node scale이 핵심**: 노드별 re-centering으로 50%+ 회복
3. **Self에서도 +25%**: 모델 자체가 per-node mean 예측에 취약 → Scale Extractor는 same-year에도 도움
4. **50% 넘는 잔여 에러**: pattern level의 cross-year 차이도 존재 → CL 필요

**분해**:
- ~25%: 모델의 일반적 mean prediction bias (같은 연도에서도 존재)
- ~25%: 연도 간 scale drift 보정 (cross-year에서만 추가)
- ~50%: scale 외 요인 (temporal pattern 차이, 센서 상태 변화 등)

**해석 기준** (SAN/Dish-TS 문헌 기반):
| Oracle MAE 개선율 | 의미 | 다음 행동 |
|---|---|---|
| **>50%** ← 우리 결과 | Scale이 절대적 → **이 연구는 무조건 된다** | Scale 예측만 잘하면 됨 |
| **20-50%** | Scale이 상당히 중요 | Hybrid approach (SPD + pattern) |
| **10-20%** | Scale 기여 보통 | 구조적 변화도 함께 다뤄야 |
| **<10%** | Scale이 아닌 다른 원인 | Normalization보다 transfer learning |

> 분석 코드: `eda/concept_drift/oracle_scale_test_v2.py`
> 결과 파일: `eda/concept_drift/oracle_scale_test_v2_results.json`

### 4.2 Baseline Scale Contamination 측정 (Linear Probing)

**방법**: 학습된 STAEformer의 **중간 Layer Activation**과 Raw Flow 데이터 간의 상관을 측정.

```python
# 1. 학습된 모델에서 hidden states 추출
hidden_states = []
def hook_fn(module, input, output):
    hidden_states.append(output.detach())
model.temporal_transformer.register_forward_hook(hook_fn)
model(test_x)

# 2. Linear probing: hidden state → mean_flow 예측
from sklearn.linear_model import Ridge
h = hidden_states[0].reshape(-1, D)  # (B*N, D)
mean_flow = test_x[:, :, :, 0].mean(dim=1).reshape(-1)  # (B*N,)

probe = Ridge().fit(h[:train_size], mean_flow[:train_size])
r2 = probe.score(h[train_size:], mean_flow[train_size:])
print(f"Scale R²: {r2:.3f}")  # 높을수록 scale에 오염됨

# 3. Layer별 분석: 어느 depth에서 scale 정보가 강한가?
```

**해석**:
- R² > 0.8 → representation이 scale에 심하게 오염 → disentanglement 필요성 입증
- R² < 0.3 → scale 정보가 이미 적음 → 다른 접근 필요

**더 정교한 방법**: MINE (Mutual Information Neural Estimation)으로 I(hidden_state; mean_flow) 추정.
- MINE 구현: [mine-pytorch](https://github.com/gtegner/mine-pytorch)
- 장점: 비선형 관계도 포착, 이론적 근거 강함

### 4.3 Pattern-only Training ⭐

**방법**: 모든 데이터를 `x / mean(x)`로 스케일링하여 **패턴만 남긴 뒤** 기존 모델을 학습.

```python
# Pattern-only normalization (per-sample, per-node)
def pattern_normalize(x):
    """x: (B, T, N, C)"""
    flow = x[:, :, :, 0:1]
    node_mean = flow.mean(dim=1, keepdim=True)  # (B, 1, N, 1)
    flow_pattern = flow / (node_mean + 1e-8)     # scale-free pattern
    x_pattern = x.clone()
    x_pattern[:, :, :, 0:1] = flow_pattern
    return x_pattern, node_mean

# Train: pattern-only model
x_pattern, scale = pattern_normalize(train_x)
model_pattern.train(x_pattern, train_y / scale)  # target도 pattern-only

# Test: cross-year transfer
x_test_pattern, test_scale = pattern_normalize(test_x_2023)
pred_pattern = model_pattern(x_test_pattern)
pred = pred_pattern * test_scale  # scale 복원 (input scale 사용)
```

**검증 포인트**:
1. Same-year 성능: pattern-only로도 합리적 예측이 가능한가?
2. Cross-year 전이: pattern-only 모델의 MAE 증가율이 baseline보다 낮은가?
3. **만약 cross-year 전이가 크게 개선되면** → Scale이 drift의 주범 → SICPL의 근거

### 4.4 Frequency Spectrum 분석 (저주파 vs 고주파)

**방법**: 2022 vs 2023 데이터를 FFT하여 주파수 대역별 변화를 비교.

```python
import numpy as np
from numpy.fft import fft, fftfreq

# Per-node FFT
for node in functional_nodes:
    flow_2022 = data_2022[:, node, 0]
    flow_2023 = data_2023[:, node, 0]

    spec_2022 = np.abs(fft(flow_2022))
    spec_2023 = np.abs(fft(flow_2023))

    freqs = fftfreq(len(flow_2022), d=5*60)  # 5-min intervals

    # 저주파 (< 1/day) vs 고주파 (> 1/day)
    low_freq_mask = np.abs(freqs) < 1/(24*3600)
    high_freq_mask = np.abs(freqs) >= 1/(24*3600)

    low_change = np.mean(np.abs(spec_2023[low_freq_mask] - spec_2022[low_freq_mask]))
    high_change = np.mean(np.abs(spec_2023[high_freq_mask] - spec_2022[high_freq_mask]))

    # 기대: low_change >> high_change → 가설 확인
```

**기대**: 저주파 변화량 >> 고주파 변화량 → "drift는 주로 저주파(Scale)에서 발생" 가설 확인

---

## 5. 모델 학습 이후 분석 (Post-experiment Analysis)

### 5.1 Core Metrics

| Metric | 설명 | 기대 |
|--------|------|------|
| Same-year MAE | 같은 연도 test | SPD ≈ Baseline |
| Cross-year MAE | 다른 연도 test | SPD << Baseline |
| MAE degradation rate | (cross - same) / same | SPD의 비율이 낮음 |
| nMAE stability | 연도 간 nMAE 변동 | SPD가 더 안정적 |

### 5.2 Disentanglement Quality

#### 5.2.1 Scale-Pattern Correlation
```
corr(z_pattern, z_scale) → 0에 가까울수록 잘 분리됨
```

#### 5.2.2 Scale Probe
```
Linear classifier: z_pattern → year 분류
- 정확도 ≈ chance level (33%) → pattern은 year-invariant
Linear classifier: z_scale → year 분류
- 정확도 > chance → scale은 year 정보를 인코딩
```

#### 5.2.3 Intervention Test
```
# Scale swap: node A의 pattern + node B의 scale
y_swap = Decoder(z_pattern_A, z_scale_B)
# 결과: A의 temporal 패턴 + B의 flow 크기 → disentanglement 직접 확인
```

### 5.3 Visualization

#### 5.3.1 t-SNE of z_pattern
- Year별로 색칠: 분리되면 안 됨 (year-invariant이므로 섞여야 함)
- Node/ToD별로 색칠: 이것으로 분리되어야 함

#### 5.3.2 t-SNE of z_scale
- Year별로 색칠: 연도별 cluster가 있어야 함

#### 5.3.3 Scale Trajectory
- 동일 node의 z_scale을 시간에 따라 plot
- 폭풍 기간에 z_scale이 변하고, z_pattern은 안정적인지 확인

### 5.4 Robustness Test

#### 5.4.1 Synthetic Scale Shift
Test data flow를 인위적으로 0.5x, 0.8x, 1.2x, 1.5x scaling:
```
SPD: scale extractor가 새 scale을 추출 → robust
Baseline: 전체 prediction 왜곡 → fragile
```

#### 5.4.2 Partial Drift
일부 node만 scale shift (실제 시나리오: 특정 도로만 공사/사고):
```
# 20% node에 0.7x scale → 나머지 80%에 영향 미치는가?
SPD: 각 node 독립적 scale → 영향 없음
Baseline: spatial attention으로 전파 → 영향 있음
```

### 5.5 Ablation Study

| Variant | Scale 처리 | Pattern 처리 | 특이사항 |
|---------|-----------|-------------|---------|
| Baseline | 없음 | 없음 | 현재 STAEformer |
| RevIN only | Remove/restore | 묵시적 | 가장 간단한 baseline |
| SPD (ours) | Scale encoder | Pattern encoder | Full model |
| SPD - L_indep | Scale encoder | Pattern encoder | Independence loss 제거 |
| SPD - Scale pred | Scale encoder | Pattern encoder | Scale prediction loss 제거 |
| SPD + Cross-Year CL | Scale encoder | Contrastive pattern | 두 아이디어 결합 |

---

## 6. Implementation Roadmap

### Phase 0: Pre-experiment Validation
- [x] Cross-year evaluation: SAN_BERNARDINO + ALAMEDA (2022-2024 Q1)
- [x] Daily profile correlation: r=0.89 (drift), r=0.98 (normal)
- [x] Cross-year pair false positive: FP 14% → filtering으로 ~0%
- [x] Weather API 가용성 확인: IEM ASOS, Open-Meteo 등
- [x] **Oracle Test**: Target re-centering으로 cross-year MAE **50.4% 회복**
- [ ] Pattern-only Training (x/mean(x))
- [ ] FFT Spectrum 분석 (저주파 vs 고주파)
- [ ] Linear Probing (scale R²)
- [x] **RevIN baseline experiment**: Post-hoc RevIN re-centering 평균 -13.4% (실패). 모델이 이미 input scale 활용. 외부 정보 필요성 확인

### Phase 1: Basic SPD Model (3-5일)
- [ ] ScaleExtractor 구현
- [ ] PatternEncoder (normalized STAEformer) 구현
- [ ] ScaleConditionedDecoder 구현
- [ ] L_forecast + L_scale + L_indep 구현
- [ ] Same-year + Cross-year evaluation

### Phase 2: Extensions (3-5일)
- [ ] Cross-Year CL + SPD 결합
- [ ] Ablation study
- [ ] Multi-channel scale

### Phase 3: Analysis (2-3일)
- [ ] Disentanglement quality 측정
- [ ] t-SNE visualization
- [ ] Robustness test (synthetic scale shift)
- [ ] Intervention test (scale swap)

---

## 7. Hyperparameters

| Parameter | Range | Default |
|-----------|-------|---------|
| d_scale (scale embedding dim) | 16 - 64 | 32 |
| d_pattern (pattern dim) | Same as STAEformer hidden | 64 |
| α (scale prediction weight) | 0.01 - 0.5 | 0.1 |
| β (independence weight) | 0.001 - 0.1 | 0.01 |
| Scale features | {mean, std} or {mean, std, slope, last} | 4 features |

---

## 8. Risk & Mitigation

| Risk | Mitigation |
|------|-----------|
| Pattern이 scale 정보 없이 부족한 경우 | Scale gate로 pattern에 조건부 scale 정보 주입 |
| Scale prediction 오류가 전파 | Residual connection: base prediction + scale adjustment |
| 분리가 잘 안 되는 경우 | L_indep 가중치 증가, gradient reversal layer 추가 |
| 학습 불안정 | 2-phase: (1) Pattern encoder만 학습, (2) Scale 추가 학습 |
| RevIN과 성능 차이 없음 | ✅ RevIN이 -13.4%로 실패 확인. 이 리스크는 해소됨 |

---

## 9. SICPL: 최종 통합 Framework

### 9.1 Architecture Overview

**Scale-Invariant Contrastive Pattern Learning (SICPL)** — 논문의 최종 형태:

```
Multi-Year Traffic Data (2022, 2023, 2024)
    │
    ├──→ Metadata-Guided Scale Extractor ──→ z_scale (year-specific)
    │         ↑                                    │
    │    [Weather: precip, temp, wind]             │
    │    [Calendar: holiday, school]               │
    │                                              │
    ├──→ Instance Normalize ──→ Pattern Encoder ──→ z_pattern (year-invariant)
    │         (x / mean(x))        (STAEformer)        │
    │                                              Cross-Year Contrastive Loss
    │                                              (같은 node, 다른 year = positive)
    │                                                   │
    └──→ Scale-Conditioned Decoder(z_pattern, z_scale) ──→ Prediction
              │
         L_forecast + λ₁·L_contrastive + λ₂·L_scale + λ₃·L_independence
```

### 9.2 Three Pillars

| Pillar | Component | 역할 | Loss |
|--------|-----------|------|------|
| **Pattern Invariance** | Cross-Year CL | z_pattern이 연도에 불변하도록 | InfoNCE |
| **Scale Awareness** | Metadata-Guided Scale Extractor | Weather/Calendar로 scale 변화 예측 | Scale prediction |
| **Explicit Disentanglement** | Independence constraint | z_pattern ⊥ z_scale 강제 | Correlation penalty |

### 9.3 Why Each Pillar Needs the Others

- **CL만**: Pattern은 invariant하지만, scale 예측이 없으면 최종 output scale이 부정확
- **SPD만**: Scale과 pattern을 분리하지만, pattern의 year-invariance에 대한 명시적 학습 신호 없음
- **Weather만**: Scale 예측은 가능하지만, pattern encoder가 여전히 scale에 오염될 수 있음
- **SICPL**: 세 요소가 상호 보완 — CL이 pattern의 invariance를 보장하고, SPD가 분리를 강제하고, Weather가 scale 예측의 정확도를 높임

### 9.4 Experimental Progression

논문에서의 실험 순서 (점진적으로 component 추가):

| Step | Model | 검증 포인트 |
|------|-------|-----------|
| 0 | Baseline STAEformer | Cross-year MAE 확인 (현재: +93~175%) |
| 1 | + RevIN (Instance Norm) | Scale 제거만으로 얼마나 개선? |
| 2 | + Pattern-only training | Scale-free 모델의 cross-year transfer |
| 3 | + Scale Extractor (flow stats) | 학습된 scale 분리의 효과 |
| 4 | + Cross-Year CL | Pattern invariance 학습 추가 효과 |
| 5 | + Weather metadata | Metadata-guided scale의 추가 효과 |
| 6 | Full SICPL | 모든 component 통합 |

### 9.5 Expected Contribution

1. **Empirical Finding**: 교통 concept drift의 주파수 분해 분석 — 저주파(scale) vs 고주파(pattern)
2. **Framework**: SICPL — contrastive pattern learning + metadata-guided scale estimation
3. **Cross-County Validation**: SAN_BERNARDINO(남부) + ALAMEDA(북부)에서 일관된 효과
4. **Practical Impact**: Multi-year 데이터가 있는 교통 시스템에 즉시 적용 가능
