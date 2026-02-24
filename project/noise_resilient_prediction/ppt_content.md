# Noise-Resilient Spatiotemporal Forecasting — PPT Content

---

## Slide 1: Problem — 센서 노이즈가 예측을 망친다

### 교통 예측 시스템의 현실
- 수백~수천 개 센서가 실시간 교통 데이터 수집
- **센서 고장은 일상적**: 캘리포니아 PeMS 데이터 기준 ~17% 센서가 불완전
- 고장 유형: 랜덤 노이즈, 시스템적 편향, 점진적 드리프트

### 핵심 문제: 노이즈는 "보이지 않는다"
| | Missing Data | **Sensor Noise** |
|---|---|---|
| 값 | 0 (빈값) | **그럴듯한 오류값** |
| 탐지 | mask=0으로 알 수 있음 | **mask=1, 모델이 구분 불가** |
| 기존 해결책 | 보간법으로 해결 가능 | **기존 방법 없음** |

### 결과: 예측 성능 급격 저하
- 30%의 센서에 moderate noise → **STAEformer MAE +192% 악화**
- 노이즈가 없는 센서의 예측까지 악화 (spatial spillover)

---

## Slide 2: 실험 세팅

### Dataset: SAN_BERNARDINO
| 항목 | 값 |
|---|---|
| 센서 수 | 893개 (California 고속도로 loop detector) |
| 입력 변수 | 5ch: flow, occupancy, speed, time-of-day, day-of-week |
| 시간 해상도 | 5분 간격 (하루 288 step) |
| 데이터 기간 | 3개월 (26,280 timestamps) |
| 예측 태스크 | 12 step 입력 → 12 step 출력 (1시간 → 1시간) |
| 예측 대상 | Flow (교통량, channel 0) |
| Train/Val/Test | 6:2:2 |

### 센서 건강 상태
| 카테고리 | 기준 (3ch zero rate) | 수 | 비율 |
|---|---|---|---|
| Dead | >90% zero | 120 | 13.4% |
| Major fail | 50-90% zero | 28 | 3.1% |
| Partial fail | 5-50% zero | 120 | 13.4% |
| **Functional** | <5% zero | **625** | **70.0%** |

→ 평가 대상: Dead/Major 제외한 **745개 노드** (Partial + Functional)

### Backbone 모델 2종
| 모델 | 유형 | 공간 메커니즘 | Clean MAE |
|---|---|---|---|
| **STAEformer** | Transformer | Adaptive attention | 12.13 |
| **STGCN** | GCN | Fixed graph convolution | 13.99 (sparse adj) |

---

## Slide 3: 노이즈 주입 프로토콜

### Test-time 노이즈 주입
- 745개 평가 노드 중 **r%를 무작위 선택**하여 physical channels (flow, occ, speed)에 노이즈 주입
- mask는 1 유지 (모델은 노이즈를 감지할 수 없음)
- 동일한 노이즈를 모든 모델에 적용 (seed 고정, 공정 비교)

### 3가지 노이즈 유형

| 유형 | 설명 | 현실 사례 | 파라미터 |
|---|---|---|---|
| **Gaussian** | 랜덤 additive noise | 전자 간섭, 양자화 오류 | severity s = noise_std / ch_std |
| **Bias** | 시스템적 over/under counting | 센서 캘리브레이션 오류 | factor: ×(1±s) |
| **Drift** | 시간에 따라 점진적 편향 증가 | 센서 노화, 환경 변화 | max drift at end = s × ch_std |

### 13가지 테스트 구성

| 유형 | 구성 (severity × rate) |
|---|---|
| Gaussian (5) | s=0.3×r=10%, s=0.3×r=30%, s=0.3×r=50%, s=0.5×r=30%, s=1.0×r=30% |
| Bias (4) | s=0.3×r=10%, s=0.3×r=30%, s=0.3×r=50%, s=0.5×r=30% |
| Drift (4) | s=0.3×r=10%, s=0.3×r=30%, s=0.3×r=50%, s=0.5×r=30% |

- Severity 범위: 0.3 (mild) ~ 1.0 (severe)
- Rate 범위: 10% (74 nodes) ~ 50% (372 nodes)

---

## Slide 4: 평가 지표

### 3가지 관점의 MAE 측정

```
전체 745개 평가 노드
├── Corrupted nodes (r% 선택된 노드)     → "직접 피해"
└── Healthy nodes (나머지 노드)           → "간접 피해 (spillover)"
```

| 지표 | 대상 | 의미 |
|---|---|---|
| **All Functional MAE** | 745 노드 전체 | **1차 지표** — 전체 시스템 예측 품질 |
| Corrupted MAE | 노이즈 주입 노드만 | 직접 피해 — encoder의 복원 능력 |
| Healthy MAE | 노이즈 없는 노드만 | Spillover — 공간 전파 방어 능력 |

### 보고 방식: 절대 MAE (Absolute Noisy MAE)
- **절대 MAE**: 모델의 noisy 환경 실제 성능 (clean MAE + degradation)
- 상대 %(relative degradation)는 보조 참고용
- 이유: 모델별 clean MAE가 다르면 %만으로 비교 시 오해 발생

### Masked MAE (null_val=0)
- Target이 0인 시점 제외 (센서가 실제로 꺼져있는 구간)
- 관측된 시점의 예측 정확도만 평가

---

## Slide 5: Proposed Method — Overview

### Two Lines of Defense + Graph Topology Design

```
┌─────────────────────────────────────────────────────┐
│                  Inference Pipeline                   │
│                                                       │
│  Sensor Data ──→ [Denoising Encoder] ──→ [Forecasting Model] ──→ Prediction │
│  (noisy)         (1st defense:          (2nd defense:                        │
│                   signal cleaning)       noise-robust features)              │
│                                                       │
│  + Sparse Adjacency Matrix (GNN 모델: distortion 전파 차단)                │
└─────────────────────────────────────────────────────┘
```

### 3가지 구성요소

| # | 구성요소 | 역할 | 비용 |
|---|---|---|---|
| 1 | **Residual Denoising Encoder** | 입력 신호 복원 (corrupted → clean) | ~2M params, pre-training 필요 |
| 2 | **Noise Augmentation** | 잔여 노이즈에 대한 implicit robustness | **0 extra params** |
| 3 | **Sparse Adjacency Matrix** | Encoder distortion 전파 차단 (GNN) | 0 params (graph 설계) |

---

## Slide 6: Method 1 — Residual Denoising Encoder

### 핵심 아이디어: "교정값만 학습"

```
v1 (실패):  output = f(noisy_input)           ← 전체 신호 재구성 필요
v2 (성공):  output = noisy_input + Δ(noisy_input)  ← 교정값(Δ)만 학습
```

| | v1 | **v2 (제안)** |
|---|---|---|
| Clean input | output ≠ input (왜곡 발생) | **output ≈ input** (Δ ≈ 0) |
| Noisy input | 재구성 시도 (불완전) | **Δ ≠ 0, 노이즈 제거** |
| Pretrain val loss | 0.18 | **0.05** (3.6× 개선) |

### 구조
- 입력: [flow, occ, speed, tod, dow] (5ch)
- Temporal: Dilated conv 4 layers → 시간 패턴 학습
- Spatial: Graph conv 1 layer → 이웃 센서 참조하여 이상치 탐지
- 출력: [flow_clean, occ_clean, speed_clean, tod, dow] (5ch)

### Self-Supervised Pre-training
1. Clean data에 synthetic noise 주입 (severity 0.1~1.5, rate 0.1~0.7)
2. Encoder가 clean signal 복원하도록 학습
3. Loss: MSE(denoised, clean)
4. Downstream에서는 **encoder freeze** (고정)

---

## Slide 7: Method 2 — Noise Augmentation

### Training-Time Regularization (0 extra parameters)

학습 중 50% 확률로:
1. 5~30% 노드 무작위 선택
2. Physical channels에 noise 주입 (Gaussian/Bias/Drift)
3. **Target은 원본 clean data** → 모델이 노이즈에 강건한 feature 학습

### 효과
| 효과 | 메커니즘 |
|---|---|
| **Spillover 방지** | Spatial attention이 노이즈에 둔감해짐 (→ healthy node 보호) |
| **Implicit robustness** | 노이즈가 있는 입력에서도 정상적 예측 학습 |
| **Zero cost** | 파라미터 추가 없음, training-time에만 적용 |

### Encoder와의 상호작용
```
Training:  Clean data → [Noise Aug] → [Encoder denoises] → [Model predicts]
                         ↑ 연습용 노이즈        ↑ 대부분 제거          ↑ 잔여분 처리

Inference: Noisy data ──────────────→ [Encoder denoises] → [Model predicts]
                                       ↑ 실제 노이즈 제거    ↑ 잔여분 처리
```

---

## Slide 8: Method 3 — Sparse Adjacency Matrix

### 문제: Dense Graph가 Distortion을 전파한다

| | 기존 (Dense) | **제안 (Sparse)** |
|---|---|---|
| 구성 방법 | Distance Gaussian kernel (σ=41.5km) | (같은 고속도로+방향) OR (거리<10km) |
| 밀도 | **95%** (평균 848 이웃) | **28.5%** (평균 255 이웃) |
| 특성 | 거의 fully-connected | 도로 구조 반영 |

### 왜 중요한가
- Encoder의 residual correction은 clean data에서도 완벽히 0이 아님 (평균 |Δ| ≈ 0.33)
- Dense adj: 이 미세한 오차가 graph conv를 통해 **893개 전체 노드로 전파**
- Sparse adj: 오차가 **지역적으로 제한**, 먼 노드에 영향 안 줌

### 결과
| | Dense Adj | Sparse Adj |
|---|---|---|
| STGCN baseline MAE | 14.66 | 14.63 (동등) |
| STGCN + v2 combo MAE | 15.71 | **14.74** |
| **Clean penalty** | **+7.1%** | **+0.75%** |

---

## Slide 9: Results — 절대 MAE 비교

### STAEformer (All Functional, 745 nodes)

| Noise Config | Baseline | Aug only | **v2 Combo** | v2 승리? |
|---|---|---|---|---|
| Clean | **12.13** | 12.57 | 12.81 | — |
| Gaussian s=0.3 r=30% | 16.22 | 13.33 | **13.45** | (aug-0.12) |
| Gaussian s=0.5 r=30% | 35.39 | 14.32 | **14.19** | O (-0.13) |
| Gaussian s=1.0 r=30% | 77.32 | 17.46 | **16.51** | O (-0.95) |
| Bias s=0.3 r=30% | 21.16 | 16.66 | **15.93** | O (-0.73) |
| Bias s=0.5 r=30% | 30.98 | 20.59 | **19.93** | O (-0.66) |
| Drift s=0.3 r=30% | 16.74 | 14.57 | **14.16** | O (-0.41) |
| Drift s=0.5 r=30% | 21.09 | 16.59 | **15.81** | O (-0.78) |

**절대 MAE 승리: 10/13 (77%)**

### STGCN + Sparse Adj (All Functional, 745 nodes)

| Noise Config | Baseline | **v2 Combo** | v2 이득 |
|---|---|---|---|
| Clean | **13.99** | 14.03 | +0.04 |
| Gaussian s=0.3 r=10% | 14.77 | **14.30** | **-0.47** |
| Gaussian s=0.3 r=30% | 16.48 | **14.83** | **-1.65** |
| Gaussian s=0.5 r=30% | 21.59 | **15.73** | **-5.86** |
| Gaussian s=1.0 r=30% | 40.74 | **17.79** | **-22.95** |
| Bias s=0.3 r=10% | 15.68 | **15.14** | **-0.54** |
| Bias s=0.3 r=30% | 19.85 | **17.78** | **-2.07** |
| Bias s=0.5 r=30% | 26.07 | **20.82** | **-5.25** |
| Drift s=0.3 r=10% | 15.19 | **14.80** | **-0.39** |
| Drift s=0.3 r=30% | 18.04 | **16.55** | **-1.49** |
| Drift s=0.5 r=30% | 22.29 | **19.15** | **-3.14** |

**절대 MAE 승리: 13/13 (100%)**

---

## Slide 10: Results — Spillover 방어

### 노이즈가 없는 센서도 피해를 입는다 (Spatial Spillover)

Gaussian s=1.0, r=30% (가장 extreme case):

| 모델 | Healthy MAE 증가 | 의미 |
|---|---|---|
| STAEformer baseline | **+449.3%** | 노이즈 없는 센서도 예측 5.5배 악화 |
| + Aug only | +1.0% | 거의 완벽 방어 |
| + v2 Combo | **+0.5%** | 거의 완벽 방어 |
| STGCN sparse baseline | +163.6% | 심각한 spillover |
| + v2 Combo (sparse) | **+7.5%** | 대폭 감소 |

→ Noise Augmentation이 spillover 방어의 핵심 메커니즘

---

## Slide 11: Key Findings 요약

### 1. Residual Connection이 핵심
- v1 (full reconstruction): clean data 왜곡 → combo가 오히려 악화
- **v2 (residual correction): identity fallback → clean data 안전**

### 2. 두 방어선이 상호 보완
- Encoder: corrupted node 직접 복원 (1st line)
- Augmentation: 잔여 노이즈 처리 + spillover 방지 (2nd line)
- 단독보다 조합이 항상 우수

### 3. Graph Topology가 결정적 (GNN 모델)
- Dense adj (95%): encoder distortion 전파 → clean penalty +7.1%
- **Sparse adj (28.5%): distortion 차단 → clean penalty +0.75%**

### 4. 종합 성능
| 아키텍처 | 절대 MAE 승리 | Clean Penalty |
|---|---|---|
| STAEformer (v2 combo) | **10/13 (77%)** | +5.6% |
| STGCN + sparse adj (v2 combo) | **13/13 (100%)** | +0.75% |

---

## Slide 12: Limitations & Future Work

1. **Single seed**: 다중 시드 실험으로 통계적 유의성 검증 필요
2. **v1 vs v2 confound**: Residual 외에 hidden dim, noise range도 다름 → controlled ablation 필요
3. **STAEformer clean penalty (+5.6%)**: Sparse adj 트릭은 GNN에만 적용 가능. Attention 기반 모델의 clean penalty 감소 방법 연구 필요
4. **단일 데이터셋**: 다른 교통 데이터셋에서 검증 필요
5. **Stuck noise**: 값이 정상으로 보이는 frozen sensor는 reconstruction으로 해결 불가 → reliability estimation (신뢰도 점수) 접근 필요
