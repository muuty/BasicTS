# SAN_BERNARDINO 실제 Noise 패턴 분석

Dataset: 105,120 timesteps (365일), 893 nodes, 5-min intervals
분석 기준: functional nodes = dead/major_fail 제외한 745개

## 1. Dead (센서 완전 사망)

All 3 physical channels (flow, occ, speed) = 0인 비율로 판단.

| 카테고리 | 노드 수 | 비율 |
|---|---|---|
| Dead (>90% zero) | 120 | 13.4% |
| Major fail (50-90%) | 28 | 3.1% |
| Partial fail (5-50%) | 150 | 16.8% |
| Functional (<5%) | 595 | 66.6% |

- **전체 노드의 33%가 일정 수준 이상의 dead 패턴 보유**
- Dead 센서는 장기간 (수일~수개월) 지속
- 인덱스 파일: `datasets/xtraffic/SAN_BERNARDINO/{dead_indices,major_fail_indices}.npy`

## 2. Stuck (값 고착)

같은 non-zero 값이 6 timestep (30분) 이상 연속 반복.

| 지표 | 값 |
|---|---|
| 영향 노드 (functional 중) | 579 / 745 (78%) |
| 총 에피소드 수 | 87,391 |
| 노드당 중앙값 | ~수십 회 |
| 최대 지속시간 | 1,190분 (≈20시간) |

**예시:**
- Node 0: flow=13이 **1,170분 (19.5시간)** 고착
  - `...26, 24, 19 → [13]*234 → 24, 30, 21...`
- Node 11: flow=10이 1,190분 고착
  - `...5, 2, 5 → [10]*238 → 1, 0, 0...`

**원인 추정:** 센서 통신 장애, 마지막 정상값을 계속 보고

## 3. Momentary Zero (순간 0 스파이크)

이전 timestep > 5, 현재 = 0, 다음 timestep > 5.

| 지표 | 값 |
|---|---|
| 영향 노드 (functional 중) | 495 / 745 (66%) |
| 총 에피소드 수 | 26,700 |

**예시:**
- Node 3: `8 → 0 → 8`
- Node 6: `26 → 0 → 13`
- Node 10: `40 → 0 → 80`

**원인 추정:** 통신 에러, 센서 순간 리셋, 데이터 전송 누락

## 4. Large Jump Spike (급등/급락)

|delta| > 5 * node_std (해당 노드의 non-zero flow 기준).

| 지표 | 값 |
|---|---|
| 영향 노드 (functional 중) | 133 / 745 (18%) |
| 총 에피소드 수 | 2,053 |

**예시:**
- Node 36: `6 → 148` (mean=24.4, std=21.4)
- Node 63: `46 → 565` (mean=35.2, std=53.1)
- Node 75: `85 → 422` (mean=81.2, std=56.5)

## 5. Extreme High (비정상 고값)

flow > mean + 10*std.

| 지표 | 값 |
|---|---|
| 영향 노드 (functional 중) | 27 / 745 (4%) |
| 총 에피소드 수 | 969 |

**예시:**
- Node 63: flow=786 (mean=35.2, std=53.1, threshold=566.0)

## 6. Drift (점진적 편향)

1개월차 vs 3개월차 non-zero flow 평균 비교.

| 지표 | 값 |
|---|---|
| >20% drift | 105 nodes |
| >50% drift | 47 nodes |
| 평균 drift ratio (functional) | 17.2% |

**Top drift nodes:**
- Node 838: 5.7 → 111.6 (1,851% drift!) — 센서 교체/신설 가능성
- Node 152: 100.6 → 546.2 (443%)
- Node 98: 6.4 → 30.4 (374%)

**주의:** drift는 실제 교통량 변화와 센서 이상을 구분하기 어려움.
극단적 drift (>200%)는 센서 이상 가능성 높음, 경미한 drift (<20%)는 자연적 변화일 수 있음.

## Noise 유형별 빈도 순위

| 순위 | Noise 유형 | 영향 노드 | 에피소드 수 | 현실성 |
|---|---|---|---|---|
| 1 | **Stuck** | 579 (78%) | 87,391 | ★★★★★ |
| 2 | **Momentary Zero** | 495 (66%) | 26,700 | ★★★★★ |
| 3 | **Dead** | 298 (33% 전체) | - (지속적) | ★★★★★ |
| 4 | **Drift** | 105 (14%, >20%) | - (지속적) | ★★★★ |
| 5 | **Large Jump** | 133 (18%) | 2,053 | ★★★ |
| 6 | **Extreme High** | 27 (4%) | 969 | ★★ |

## Noise Robustness Eval 결과 (encoder 비교)

각 encoder + noise augmentation 조합의 functional node % degradation.

| Noise | baseline | noisy_only | MLP+aug | denoise_v2+aug | lin_attn+aug | Best |
|---|---|---|---|---|---|---|
| **Dead** | +259% | +262% | **+28%** | +267% | +160% | MLP |
| **Stuck** | +20% | +13% | +14% | +11% | **+11%** | lin_attn |
| **Drift** | +53% | +21% | +16% | **+14%** | +16% | denoise_v2 |
| **Gauss(low)** | +43% | +6% | +5% | +5% | **+5%** | lin_attn |
| **Gauss(high)** | +365% | +26% | +25% | +20% | **+16%** | lin_attn |
| **Bias** | +104% | +42% | **+26%** | +33% | +34% | MLP |

### 핵심 인사이트
1. **Dead noise: MLP 압도적 승리** — ST encoder는 dead 신호를 spatial aggregation으로 전파 (spillover)
   - MLP healthy_deg = +0.8%, denoise_v2 healthy_deg = +44.1% (dead_r50 기준)
2. **Stuck/Drift: ST encoder 우위** — temporal/spatial context가 고착/편향 감지에 유리
3. **Dead+Stuck이 가장 흔한데, 최적 encoder가 정반대** → adaptive/hybrid 접근 필요
4. **Noise augmentation만으로는 dead에 무력** (noisy_only ≈ baseline)
