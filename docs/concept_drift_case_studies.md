# Concept Drift Case Studies: Sustained Multi-Day Pattern Changes

**Dataset**: SAN_BERNARDINO county (893 sensors, 2022-2024)
**Date**: 2026-02-12

## Methodology

"Concept drift"를 일시적 변동(장비 이상, 사고 등)과 구분하기 위해 **sustained drift**를 다음과 같이 정의:

1. **Q1(1-3월)과 Q3(7-9월) 모두에서 같은 방향의 변화** - 계절 효과가 아닌 구조적 변화
2. **Weekly consistency ≥ 70%** - 전체 주의 70% 이상에서 같은 방향 변화 관찰
3. **Functional in all 3 years** - 센서 고장이 아닌 실제 교통 패턴 변화

893개 센서 중 **20개가 sustained drift** 기준 충족 (증가 9, 감소 11).
평균 변화율 37.3%, 평균 weekly consistency 91.6%.

## Drift 유형 분류

| 유형 | 설명 | 해당 수 |
|------|------|---------|
| **Scale Change** (shape corr > 0.95) | 일일 패턴 형태는 유지, 전체 스케일만 변화 | 다수 |
| **Mild Shape Change** (0.8-0.95) | 대부분 스케일 변화지만 일부 시간대 패턴도 변화 | 소수 |
| **Structural Change** (< 0.8) | 일일 패턴 자체가 근본적으로 변화 | 희귀 |

핵심 발견: 대부분의 sustained drift는 **Scale Change** — daily profile의 형태는 보존되면서 전체 크기만 변화. 이는 도로의 기능(출퇴근 도로, 간선 도로 등)은 유지되지만 **수요량이 구조적으로 바뀌었음**을 의미.

---

## Case 1: Node 487 (Station 801349) — Monotonic Increase + Congestion

**패턴**: 3년간 꾸준히 증가 (283 → 302 → 420, **+48%**)
**Weekly consistency**: 98%
**Profile shape correlation**: 0.981 (Scale Change)

| | 2022 | 2023 | 2024 |
|---|---|---|---|
| Mean flow | 283 | 302 | **420** |
| Occupancy | 0.074 | 0.062 | **0.129** |
| Speed | 60.8 | 61.7 | **54.6** |
| WD/WE | 284/280 | 309/284 | 415/431 |

**해석**: 전형적인 **수요 증가 + 혼잡 악화** 패턴.
- Flow가 +48% 증가하면서 occupancy는 +74%, speed는 -10% 감소
- 2024년에는 주말 flow(431)가 주중(415)을 초과 — 상업/레저 수요 증가 시사
- Daily profile 형태는 보존 (AM 11시, PM 14시 피크 유지)
- **가설**: 인근 상업 개발 또는 인구 유입으로 인한 구조적 수요 증가

---

## Case 2: Node 731 (Station 814564) — Monotonic Decrease

**패턴**: 3년간 꾸준히 감소 (276 → 216 → 83, **-70%**)
**Weekly consistency**: 79%
**Profile shape correlation**: 0.993 (Scale Change)

| | 2022 | 2023 | 2024 |
|---|---|---|---|
| Mean flow | 276 | 216 | **83** |
| Occupancy | 0.076 | 0.031 | 0.046 |
| Speed | 60.2 | 66.1 | 63.6 |
| WD/WE | 286/252 | 222/201 | 85/79 |

**해석**: **수요 급감 + 속도 증가** — 차량이 줄면서 오히려 속도가 빨라짐.
- 3년간 거의 1/3로 감소, 그러나 daily profile 형태는 완벽히 보존 (r=0.993)
- 주중/주말 비율도 유지 — 도로 기능은 동일, 순수 수요만 감소
- **가설**: 대체 경로 개통으로 인한 우회 (route diversion), 또는 인근 시설 폐쇄

---

## Case 3: Node 557 (Station 812822) — Gradual Monotonic Decrease

**패턴**: 점진적 감소 (344 → 310 → 225, **-34%**)
**Weekly consistency**: **100%** (모든 주에서 감소)
**Profile shape correlation**: 0.995 (Scale Change)

| | 2022 | 2023 | 2024 |
|---|---|---|---|
| Mean flow | 344 | 310 | **225** |
| Occupancy | 0.114 | 0.052 | 0.089 |
| Speed | 55.5 | 65.3 | **50.7** |
| WD/WE | 347/334 | 299/338 | 222/233 |

**해석**: 가장 일관된 감소 사례 (consistency 100%).
- 2023→2024: flow 감소(-27%)에도 speed 감소(65→51) — 역설적
- Occupancy도 상당히 높은 수준 유지 — 차량 수는 줄었지만 혼잡은 악화?
- 2023년에 주말(338) > 주중(299), 2024년에도 주말(233) > 주중(222) — 비통근 도로
- **가설**: 도로 용량 감소 (차선 축소, 공사 등)로 flow와 speed 모두 감소

---

## Case 4: Node 289 (Station 827093) — Accelerating Increase

**패턴**: 2022→2023 약간 감소 후 2024 급증 (89 → 74 → 180, **+102%**)
**Weekly consistency**: 96%
**Profile shape correlation**: 0.866 (Mild Shape Change)

| | 2022 | 2023 | 2024 |
|---|---|---|---|
| Mean flow | 89 | 74 | **180** |
| Occupancy | 0.018 | 0.019 | **0.042** |
| Speed | 69.3 | 66.4 | 66.1 |
| AM peak | 7h (234) | 8h (147) | 8h (304) |
| PM peak | 16h (123) | 15h (146) | 15h (265) |

**해석**: 가장 큰 변화율을 보인 케이스. Shape도 일부 변화 (r=0.866).
- 2022년에는 AM peak가 7시(출근)에 뚜렷, PM은 약함 → 편향된 출근 도로
- 2024년에는 AM(8시, 304), PM(15시, 265) 모두 강해짐 → 양방향 통근 도로로 전환
- Occupancy +135%로 급증, speed는 유지 — 아직 용량 여유 있음
- **가설**: 인근 주거/상업 개발로 새로운 통근 수요 발생, 도로 기능 자체가 변화

---

## Case 5: Node 728 (Station 818723) — Non-Monotonic (Spike then Crash)

**패턴**: 2023년 급증 후 2024년 급감 (286 → 461 → 182, **-36%**)
**Weekly consistency**: 100%
**Profile shape correlation**: 0.949 (Mild Shape Change)

| | 2022 | 2023 | 2024 |
|---|---|---|---|
| Mean flow | 286 | **461** | 182 |
| Occupancy | 0.072 | 0.100 | 0.065 |
| PM peak hour | 15h | 14h | **18h** |

**해석**: 가장 복잡한 패턴. 2023년에 +61% 급증 후 2024년에 원래 수준 이하로 급감.
- 2024년 PM peak가 15시→18시로 이동 — 퇴근 시간 자체가 변화
- **가설**: 2023년 우회 도로 공사로 임시 수요 증가 → 2024년 공사 완료 후 정상화 + 새 경로 패턴

---

## Case 6: Node 535 (Station 812609) — Crash and Partial Recovery

**패턴**: 2023년 급감 후 부분 회복 (289 → 45 → 156, **-46%**)
**Weekly consistency**: 73%
**Profile shape correlation**: 0.948 (Mild Shape Change)

| | 2022 | 2023 | 2024 |
|---|---|---|---|
| Mean flow | 289 | **45** | 156 |
| Occupancy | 0.050 | 0.025 | 0.026 |
| Speed | 65.2 | 63.8 | 68.0 |

**해석**: 2023년에 85% 급감 (거의 dead sensor 수준) 후 2024년 부분 회복.
- 그러나 zero_rate는 낮음 (3.9%) — 센서 고장이 아니라 실제로 차량이 안 다님
- 2024년에 speed 증가 (65→68), occ 감소 — 회복은 되었지만 이전보다 훨씬 한산
- **가설**: 2023년 도로 폐쇄/대규모 공사 → 2024년 재개통되었지만 일부 수요가 영구 이탈

---

## Summary: Concept Drift의 유형화

893개 노드 중 sustained drift 기준 충족 노드의 패턴:

| 변화 유형 | 특징 | 예측 영향 |
|-----------|------|-----------|
| **Scale Change** (대다수) | Daily profile 보존, 크기만 변화 | 학습된 패턴은 유효, mean shift adaptation으로 대응 가능 |
| **Shape Change** (소수) | Peak hour 이동, WD/WE 비율 변화 | 학습된 패턴 자체가 무효, re-training 필요 |
| **Non-monotonic** (다수) | 2023년 이상 후 2024년 부분 회복 | 가장 예측 어려움, adaptive learning 필요 |

### 예측 모델에 대한 함의

1. **Scale drift** (Case 1, 2, 3): normalization이 year-specific이면 문제 없음. 그러나 2022 scaler로 2024를 normalize하면 체계적 bias 발생
2. **Peak shift** (Case 4, 5): 시간대별 attention pattern이 달라져야 함 → fine-tuning 또는 re-training 필수
3. **Non-monotonic** (Case 5, 6): 어느 시점의 데이터로 학습해도 미래를 완벽히 반영 못함 → continual learning이 가장 적합

### Files

- Plot: `eda/concept_drift/case_study_sustained_drift.png`
- Data: `eda/concept_drift/case_studies_sustained_drift.json`
- Script: `eda/concept_drift/06_case_study_sustained_drift.py`
