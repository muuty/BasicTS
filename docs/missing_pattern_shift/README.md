# Missing-Aware Spatio-Temporal Prediction

> 최종 업데이트: 2026-02-15
> 연구 방향: Missing Pattern Shift에 Robust한 Imputation-Free Forecasting

---

## Table of Contents

1. [연구 요약 (TL;DR)](#1-연구-요약-tldr)
2. [Problem Statement](#2-problem-statement)
3. [Key Findings (Prior Research)](#3-key-findings-prior-research)
4. [Layer 1: Missing Indicator Channels](#4-layer-1-missing-indicator-channels)
5. [Layer 1.5: Mask-Aware Loss](#5-layer-15-mask-aware-loss)
6. [Phase 2 Results: Artificial Missing Pattern Shift](#6-phase-2-results-artificial-missing-pattern-shift)
7. [Related Work Survey](#7-related-work-survey)
8. [Proposed Method: Missing-Invariant Contrastive Learning (MICL)](#8-proposed-method-missing-invariant-contrastive-learning-micl)
9. [Experiment Plan](#9-experiment-plan)
10. [File Directory](#10-file-directory)

---

## 1. 연구 요약 (TL;DR)

**문제**: 교통 센서 forecasting에서 missing data의 패턴(어떤 센서가 missing인지)이 시간에 따라 변함.
기존 방법들은 train/test의 missing pattern이 동일하다고 가정하지만, 현실에서는 매년 100+개 센서의 상태가 전환됨.

**핵심 발견**:
1. Flow=0의 89.1%는 missing data (real zero가 아님). NULL_VAL=0으로 이를 구분 못함
2. Mask channel 추가 → STAEformer -1.7% (도움), STGCN +21.4% (해로움). Architecture-dependent
3. Mask-aware loss → STGCN에서 17.80→14.43 (-19% 회복). Loss alignment이 핵심
4. **Missing pattern shift를 다룬 기존 연구가 없음** (Related work gap 확인)
5. **Tradeoff 발견**: mask-aware는 intermittent shift에 강하고(+64% vs +101%) node death에 약함(+19% vs +7%) → 두 shift type 모두에 robust한 방법 필요

**제안**: Missing-Invariant Contrastive Learning (MICL)
- SSL로 missing pattern에 invariant한 representation 학습
- 같은 데이터에 다른 missing mask를 적용 → positive pair
- Model-agnostic, imputation-free

---

## 2. Problem Statement

### 2.1 핵심 문제: NaN→0 대체로 인한 정보 손실

| 구분 | 설명 |
|---|---|
| 현재 관행 | NaN을 0으로 채운 후 모델에 입력 |
| 문제점 | 모델이 "교통량 0" vs "센서 고장"을 구분 불가 |
| 영향 | Attention collapse, 잘못된 message passing, 노드간 성능 편차 |

### 2.2 데이터 근거 (SAN_BERNARDINO, 893 nodes, 3개월)

**Flow=0 Decomposition:**
- 전체 flow=0 이벤트: 18,482,565개
- NaN→0 (missing data): 16,463,468개 (**89.1%**)
- Real zero (센서 작동 중 실제 교통량 0): 722,440개 (3.9%)
- Ambiguous (모든 채널 0): 1,296,657개 (7.0%)

**Speed=0:** 34,205,592개 중 **100%가 missing** (실제 속도 0은 존재하지 않음)

**센서 건강 분포:**
| Category | Nodes | Missing Rate |
|---|---|---|
| Functional (<5%) | 633 | ~0.1% |
| Partial (5-50%) | 118 | variable |
| Major fail (50-90%) | 34 | variable |
| Dead (>90%) | 108 | ~100% |
| **Total** | **893** | **mean 17.5%** |

### 2.3 NULL_VAL=0 Confusion

`masked_mae`는 target=0을 loss에서 제외. 테스트 기간:
- flow=0 이벤트 3,765,341개 중:
  - Missing (NaN→0): 3,334,230개 (88.6%) → 평가에서 제외
  - Real zero: 431,111개 (11.4%) → **역시 제외됨 (오류)**
- → Missing 때문에 target이 0인 샘플이 평가에서 빠지므로,
  모델이 missing 데이터를 잘못 처리해도 metric에 반영되지 않음

### 2.4 Missing Pattern Shift: 진짜 문제

**기존 연구의 암묵적 가정:**
- Train과 test의 missing pattern이 동일 (같은 센서가 같은 비율로 missing)
- Missing mechanism이 stationary (MCAR/MAR 가정)

**현실 (SAN_BERNARDINO cross-year 데이터):**
- 2022→2023: 123개 센서 부활, 112개 사망 (매년 100+ 상태 전환)
- 재난(대기강), 공사, 장비 교체 등으로 missing pattern이 비정상적으로 변함
- Train 시점에 functional이던 센서가 test 시점에 dead (또는 그 반대)

**왜 기존 방법이 취약한가:**

```
Imputation 기반:
  Train: 센서 A=dead, B=alive → impute A from B 학습
  Test:  센서 A=alive, B=dead → B를 impute할 근거 없음
  → Imputation 품질 급락 → Prediction 연쇄 실패

Imputation-free (mask channel):
  Train: mask_A=0, mask_B=1 → "mask=0이면 무시" 학습
  Test:  mask_A=1, mask_B=0 → mask만 바뀌고 모델은 동일하게 작동
  → Missing pattern 변화에 적응 가능하나, shift type별 tradeoff 존재 (Section 6)
```

**Phase 2 실험에서 밝혀진 tradeoff (Section 6 참조):**
- Mask input만 추가 → node death에 robust (+6.8%), 그러나 intermittent에 취약 (+115%)
- Mask-aware loss 추가 → intermittent에 robust (+64%), 그러나 node death에 취약 (+19%)
- → 단일 전략으로 모든 shift type에 robust할 수 없음 → MICL 필요 (Section 8)

---

## 3. Key Findings (Prior Research)

### 3.1 Attention Collapse on Missing Nodes
- Dead 노드가 functional 대비 **3.5-12x 과도한 spatial attention** 수신 (softmax bias)
- STGformer (linear attention)에서는 반대 현상 (0.949x anti-collapse)
- → Missing에 대한 반응이 **architecture-dependent**
- 상세: `docs/attention_collapse_missing_values.md`

### 3.2 모델별 Missing 대응 차이

```
Per-Category Masked MAE:
Model              Overall   Dead   Major   Partial   Functional
STAEformer_5ch      12.26    0.47   10.44     6.31      13.97
STGCN_1ch           14.09    0.30   11.71     8.94      15.60
STGCN_5ch           14.66    0.24   12.24     8.36      16.50

Per-Category Raw MAE (target=0 포함):
Model              Overall   Dead   Major   Partial   Functional
STAEformer_5ch      12.26   13.16   28.17     7.64      14.07
STGCN_1ch           14.09  120.15   25.36    10.22      15.74
STGCN_5ch           14.66  103.73   26.71     9.78      16.64
```

**Key insights:**
1. STAEformer는 implicit missing awareness 보유 (dead raw_mae 13 vs STGCN 120)
   - Attention이 occ/speed 채널로부터 센서 health를 암묵적으로 학습
2. STGCN 5ch 역설: occ/speed 추가가 dead에는 도움 (120→104), functional에는 해로움 (15.6→16.5)
   - Graph convolution은 extra channel을 효과적으로 활용 못함

### 3.3 Encoder Pretraining 실패 (Negative Result)

Missing 정보 없이 general representation learning 시도 → 전부 baseline 이하:

| Experiment | MAE (masked) |
|---|---|
| Baseline (no encoder) | ~12.10 |
| Contrastive only | 12.19 |
| Context fusion | 12.19 |
| Random projection | 11.94 |
| Predictive contrastive | 13.10 |
| Cross-variable pretrained | 12.22 |

→ **Missing 정보 없이는 representation으로 개선 불가**

### 3.4 Credibility Bias 효과 (Positive Signal)
- Pre-softmax bias를 raw embedding에서 학습 → attention을 missing node에서 away
- MAE: 12.178 → 12.151, functional: 13.974 → 13.947
- → Missing-aware mechanism이 prediction 개선 가능하다는 **proof of concept**
- 구현: `baselines/STAEformer/arch/staeformer_credibility.py`

---

## 4. Layer 1: Missing Indicator Channels

**Idea**: 3개의 binary mask channel (flow/occ/speed) 추가 → 5ch + 3mask = 8ch
**Dataset**: `SAN_BERNARDINO_MASK` (8ch: flow, occ, speed, mask_flow, mask_occ, mask_speed, tod, dow)

### 4.1 실험 결과

| Model | Config | Channels | MAE | vs Baseline |
|---|---|---|---|---|
| STAEformer 5ch (baseline) | `STAEformer/SAN_BERNARDINO_5ch.py` | flow,occ,speed,tod,dow | 12.26 | - |
| **STAEformer 8ch mask** | `STAEformer/SAN_BERNARDINO_5ch_mask.py` | +mask_f,mask_o,mask_s | **12.05** | **-1.7%** |
| STGCN 1ch (baseline) | `STGCN/SAN_BERNARDINO.py` | flow | 14.09 | - |
| STGCN 5ch | `STGCN/SAN_BERNARDINO_5ch.py` | flow,occ,speed,tod,dow | 14.66 | +4.0% |
| STGCN 8ch mask | `STGCN/SAN_BERNARDINO_5ch_mask.py` | +mask_f,mask_o,mask_s | 17.80 | +26.3% |

### 4.2 Per-Category 분석 (STAEformer)

| Category | n | 5ch Baseline | 8ch Mask | Change |
|---|---|---|---|---|
| Functional (<5% missing) | 632 | 13.97 | 13.86 | -0.8% |
| **Partial fail (5-50%)** | **113** | **6.31** | **5.75** | **-8.9%** |
| Major fail (50-90%) | 28 | 10.44 | 10.00 | -4.2% |
| Dead (>90%) | 120 | 0.47 | 0.22 | -53.6% |

### 4.3 결론

- **STAEformer**: mask channel 활용 가능 (attention이 mask로부터 sensor health 학습)
- **STGCN**: mask channel이 오히려 해로움 (graph conv가 extra features를 활용 못함)
- → Architecture-dependent. Model-agnostic approach 필요 (Layer 2 motivation)

---

## 5. Layer 1.5: Mask-Aware Loss

### 5.1 동기

Layer 1 실험의 한계: mask channel이 input에만 사용되고, loss에는 반영 안 됨.
- `masked_mae(null_val=0)`: target=0이면 **모두** 제외 (missing AND real zero 모두)
- **mask-aware loss**: target_mask=0인 것**만** 제외 (real zero는 loss에 포함)

### 5.2 구현

- `basicts/metrics/mask_aware.py`: `mask_aware_mae`, `mask_aware_mape`, `mask_aware_rmse`
- `basicts/runners/runner_zoo/mask_aware_runner.py`: `MaskAwareRunner`
  - `CFG.MODEL.MASK_CHANNEL_IDX = 3` (8ch 데이터에서 mask_flow의 인덱스)
  - forward에서 target_mask 추출 → return dict에 포함 → loss에 자동 전달

### 5.3 실험 결과

| Model | Config | Loss | MAE | vs mask_input_only | vs baseline |
|---|---|---|---|---|---|
| STAEformer 8ch mask | mask input | masked_mae | **12.05** | - | -1.7% |
| STAEformer 8ch mask-aware | mask input + loss | mask_aware_mae | 12.12 | +0.6% | -1.1% |
| STGCN 8ch mask | mask input | masked_mae | 17.80 | - | +26.3% |
| **STGCN 8ch mask-aware** | mask input + loss | mask_aware_mae | **14.43** | **-18.9%** | +2.4% |

### 5.4 Per-Category 분석

**STAEformer:**
| Category | n | 8ch mask (masked_mae) | 8ch mask-aware | Change |
|---|---|---|---|---|
| Functional | 632 | 13.86 | 13.85 | -0.1% |
| Partial fail | 113 | 5.75 | 6.09 | +5.9% |
| Major fail | 28 | 10.00 | 10.38 | +3.8% |

**STGCN:**
| Category | n | 8ch mask (masked_mae) | 8ch mask-aware | Change |
|---|---|---|---|---|
| Functional | 632 | 20.47 | 15.97 | **-22.0%** |
| Partial fail | 113 | 8.62 | 9.19 | +6.6% |
| Major fail | 28 | 13.48 | 12.10 | **-10.2%** |

### 5.5 결론

- **STGCN**: mask-aware loss가 극적으로 도움 (17.80→14.43). Mask channel을 input으로만 주면 해로웠지만, loss도 함께 align하면 baseline 수준으로 회복
- **STAEformer**: mask-aware loss가 약간 손해. Attention이 이미 implicit하게 missing을 잘 처리하고 있어서, explicit mask loss가 train-eval misalignment 유발
- **Train-eval misalignment 주의**: mask_aware_mae로 train하지만 evaluation은 masked_mae(null_val=0)로 수행. 학습 목표와 평가 기준이 다름

### 5.6 전체 결과 요약

| Model | Config | Loss | Overall (893) | Functional (632) | Partial (113) | Major (28) |
|---|---|---|---|---|---|---|
| **STAEformer** 5ch | baseline | masked_mae | 12.26 | 13.97 | 6.31 | 10.44 |
| **STAEformer** 8ch mask | mask input | masked_mae | **12.05** | 13.86 | **5.75** | 10.00 |
| **STAEformer** 8ch mask-aware | mask input+loss | mask_aware_mae | 12.12 | 13.85 | 6.09 | 10.38 |
| **STGCN** 1ch | baseline | masked_mae | **14.09** | 15.60 | 8.94 | 11.71 |
| **STGCN** 5ch | baseline | masked_mae | 14.66 | 16.50 | 8.36 | 12.24 |
| **STGCN** 8ch mask | mask input | masked_mae | 17.80 | 20.47 | 8.62 | 13.48 |
| **STGCN** 8ch mask-aware | mask input+loss | mask_aware_mae | 14.43 | 15.97 | 9.19 | 12.10 |

### 5.7 Normalization & Scaler Pipeline

Mask channel이 normalization에 어떻게 영향받는지 (또는 받지 않는지) 문서화.

#### 데이터 흐름

```
Raw 8ch data: [flow, occ, speed, mask_f, mask_o, mask_s, tod, dow]
                                                    ↓
              ┌──────────────────────────────────────┤
              │ Dataset → DataLoader batch           │
              │ shape: (B, T, N, 8)                  │
              ↓                                      │
    ┌─────────────────────┐                          │
    │ SimpleRunner:       │                          │
    │ preprocessing()     │                          │
    │   scaler.transform  │ ← ZScoreScaler:          │
    │   (channel 0 only)  │   target_channel=0       │
    └─────────────────────┘   → flow만 z-score 정규화│
              │               → mask/occ/speed/tod/dow│
              │                 는 변환 없이 통과     │
              ↓                                      │
    ┌─────────────────────┐                          │
    │ Feature selection:  │                          │
    │ FORWARD_FEATURES    │ = [0,1,2,3,4,5,6,7]      │
    │ TARGET_FEATURES     │ = [0] (flow만 예측)      │
    └─────────────────────┘                          │
              │                                      │
              ↓                                      │
    ┌─────────────────────┐                          │
    │ MaskAwareRunner:    │ (mask-aware 실험만)      │
    │ target_mask 추출    │ target[:,:,:,3] (mask_f) │
    │ (feature selection  │ ← 스케일링 영향 없음     │
    │  전에 추출)         │   (binary 0/1 그대로)    │
    └─────────────────────┘                          │
              │                                      │
              ↓                                      │
    ┌─────────────────────┐                          │
    │ postprocessing()    │                          │
    │ scaler.inverse_     │ ← prediction, target을   │
    │ transform           │   원래 스케일로 복원     │
    └─────────────────────┘                          │
```

#### ZScoreScaler 핵심 설정

| 파라미터 | 값 | 의미 |
|---|---|---|
| `target_channel` | `0` (default) | flow 채널만 z-score 정규화 |
| `norm_each_channel` | `False` | 전체 노드에 대해 global mean/std 계산 |
| `rescale` | `True` | postprocessing에서 inverse_transform 수행 |

**`norm_each_channel=False`가 중요한 이유:**
- `True`로 설정하면 per-node mean/std 계산
- Dead 센서 (std≈0) → std=1로 대체되어 scaler=(0, 1) 적용
- Train에서 dead인 센서가 test에서 살아나면 scaler 불일치 → 치명적 오류 (MAE ~90)
- `False`는 전체 노드의 global 통계 사용 → dead 센서 영향 최소화

#### 구현 파일

| File | 역할 |
|---|---|
| `basicts/scaler/z_score_scaler.py` | ZScoreScaler: channel별 독립 mean/std, `norm_each_channel` 분기 |
| `basicts/runners/runner_zoo/simple_tsf_runner.py` | `preprocessing()`: scaler.transform, `postprocessing()`: inverse_transform |
| `basicts/runners/runner_zoo/mask_aware_runner.py` | `MaskAwareRunner`: target_mask 추출 (scaler 영향 없는 binary mask) |

---

## 6. Phase 2 Results: Artificial Missing Pattern Shift

> 실험 날짜: 2026-02-14

### 6.1 실험 설계

5개 모델을 동일한 조건에서 비교. 학습된 모델의 가중치를 고정한 채,
test-time에 missing pattern을 인공적으로 변경하여 healthy 노드의 MAE 변화 측정.

**Shift types:**
| Type | 설명 | 현실 시나리오 |
|---|---|---|
| Node death (r%) | functional 노드 r%를 전 시간 zero화 | 영구 센서 고장 발생 |
| Intermittent (p%) | (timestep, functional_node)의 p%를 random zero화 | 랜덤 데이터 손실 증가 |

**Corruption 방식:**
- 5ch models: flow, occ, speed → 0 (모델이 missing vs real zero 구분 불가)
- 8ch models: flow, occ, speed → 0 AND mask_flow, mask_occ, mask_speed → 0 (모델이 missing 인지)

**Metric:** healthy (non-corrupted) functional 노드의 masked MAE (target_mask == 1, 8ch dataset의 binary mask 사용)

> **Metric 검증 (2026-02-15):** 기존 `target > 0`과 `target_mask == 1` 비교 결과, functional 노드의 observed rate가 96.37%로 높아 두 metric의 차이가 <1%임을 확인. 모든 결론과 모델 순위 동일. 아래 결과는 `target_mask == 1` (correct metric) 기준.

### 6.2 Node Death 결과 (영구 센서 고장)

| r% | n_healthy | STAEformer 5ch | 8ch mask | 8ch mask-aware | STGCN 1ch | STGCN 8ch mask-aware |
|---|---|---|---|---|---|---|
| Clean | 745 | 12.126 | 11.869 | 11.900 | 13.337 | 13.509 |
| 10% | 671 | +0.054 (+0.4%) | +0.060 (+0.5%) | +0.202 (+1.7%) | +2.983 (+22.4%) | +2.361 (+17.4%) |
| 20% | 596 | +0.141 (+1.2%) | +0.132 (+1.1%) | +0.561 (+4.7%) | +10.801 (+80.5%) | +9.624 (+70.8%) |
| 30% | 522 | +0.332 (+2.7%) | **+0.238 (+2.0%)** | +0.939 (+7.9%) | +14.323 (+107.4%) | +17.250 (+127.9%) |
| 50% | 373 | +1.058 (+8.7%) | **+0.814 (+6.8%)** | +2.308 (+19.3%) | +19.404 (+145.7%) | **+40.345 (+297.5%)** |

**핵심 발견 — Node Death:**
1. **STAEformer >> STGCN**: Attention은 dead node 무시 가능 (r=50%: +9% vs +146%). GCN은 zero를 이웃에 전파.
2. **8ch_mask가 가장 robust**: mask 채널이 "이 노드는 missing" 정보를 attention에 제공 → dead node를 효과적으로 무시.
3. **mask_aware는 node death에 오히려 취약** (+19.3% vs 8ch_mask의 +6.8%): 학습 시 mask=0 entry의 loss 무시 → mask=0 노드에 대한 의미 있는 representation 미학습 → unpredictable output이 attention 통해 전파.
4. **STGCN 8ch mask-aware는 catastrophic** (r=50%: +298%): GCN이 zero value + zero mask를 모두 전파, corruption 증폭.

### 6.3 Intermittent Missing 결과 (랜덤 데이터 손실)

| p% | n_healthy | STAEformer 5ch | 8ch mask | **8ch mask-aware** | STGCN 1ch | STGCN 8ch mask-aware |
|---|---|---|---|---|---|---|
| Clean | 745 | 12.126 | 11.869 | 11.900 | 13.337 | 13.509 |
| 10% | 745 | +3.904 (+32.2%) | +8.855 (+74.6%) | **+1.964 (+16.5%)** | +10.915 (+81.8%) | +7.986 (+59.1%) |
| 20% | 745 | +7.881 (+65.0%) | +12.522 (+105.5%) | **+4.524 (+38.0%)** | +23.186 (+173.8%) | +14.436 (+106.9%) |
| 30% | 745 | +12.266 (+101.2%) | +13.600 (+114.6%) | **+7.628 (+64.1%)** | +29.999 (+224.9%) | +19.608 (+145.1%) |

**핵심 발견 — Intermittent:**
1. **mask-aware가 압도적 best**: p=30%에서 +64.1% vs 5ch의 +101.2%. Mask-aware loss가 개별 missing entry 처리를 학습.
2. **8ch_mask (mask input만, 일반 loss)가 STAEformer 중 WORST**: 5ch보다도 나쁨! Mask 채널이 있으나 학습 시 이를 제대로 활용 못함 → 오히려 confusion.
3. **Loss alignment이 핵심**: 같은 8ch 구조인데 loss만 다름 → 8ch_mask(+115%) vs 8ch_mask_aware(+64%).

### 6.4 Tradeoff 발견: Node Death vs Intermittent

| Shift Type | Best STAEformer | Worst STAEformer | 이유 |
|---|---|---|---|
| **Node death** | 8ch_mask (+6.8%) | 8ch_mask_aware (+19.3%) | mask_aware가 대규모 mask=0에 과민 반응 |
| **Intermittent** | 8ch_mask_aware (+16.5%) | 8ch_mask (+74.6%) | mask_aware가 개별 missing을 적절히 무시 |

**→ 두 가지 shift type 모두에 robust한 방법이 필요 → MICL의 motivation!**

MICL은 다양한 missing pattern (node death + intermittent + spatial cluster 등)으로 augmentation하여
missing-invariant representation을 학습 → 모든 shift type에 robust.

### 6.5 결과 파일

| File | 설명 |
|---|---|
| `experiments/eval_missing_pattern_shift.py` | 실험 스크립트 (target_mask==1 및 target>0 dual metric) |
| `experiments/missing_pattern_shift_results/results.json` | 전체 결과 JSON (observed/nonzero 양쪽 포함) |
| `experiments/missing_pattern_shift_results/*_clean_mae_observed.npy` | 모델별 clean per-node MAE (target_mask==1) |
| `experiments/missing_pattern_shift_results/*_clean_mae_nonzero.npy` | 모델별 clean per-node MAE (target>0, legacy) |

---

## 7. Related Work Survey

> 조사 시점: 2026-02-14

### 7.1 핵심 결론: Missing Pattern Shift는 미개척 영역

**기존 연구가 하는 것:**
- Missing **rate** 변화에 대한 robustness (10%→50%)
- 같은 pattern type 내에서 imputation 품질 비교
- MCAR/MAR/MNAR 각각의 성능 분석

**기존 연구가 안 하는 것:**
- Train pattern A → Test pattern B (cross-pattern generalization)
- Imputation-based vs imputation-free 비교 under pattern shift
- 센서 health distribution 변화에 대한 robustness

### 7.2 가장 관련 있는 기존 연구

#### (1) Revisiting MTSF with Missing Values (2024)
- **발견**: Imputation 없이 직접 예측하는 것이 imputation-then-forecast보다 오히려 나음
- **한계**: Fixed missing pattern 가정. Pattern shift 미평가
- **우리와의 관계**: 그들의 finding을 pattern shift 상황에서 확장

#### (2) TSI-Bench (2024)
- **기여**: 28개 imputation 알고리즘 벤치마크 (point, subsequence, block missing)
- **한계**: 각 pattern을 따로 평가. Cross-pattern generalization 없음
- **우리와의 관계**: Cross-pattern evaluation protocol 제안으로 확장 가능

#### (3) Modeling Information Blackouts in MNAR (2025)
- **기여**: Traffic sensor blackout을 MNAR로 모델링
- **한계**: 같은 데이터 내에서 MCAR vs MNAR 비교. Train-test shift 아님
- **우리와의 관계**: 우리는 train-test 간 missing mechanism 자체가 변하는 상황

#### (4) Masking the Gaps: Imputation-Free Approach (NeurIPS 2024 Workshop)
- **기여**: (timestep, feature)를 토큰으로 처리하는 imputation-free framework
- **한계**: Pattern shift robustness 미평가
- **우리와의 관계**: 방법론적으로 유사하나, SSL을 통한 missing-invariance 학습은 다름

#### (5) S4M: S4 with Missing Values (2025)
- **기여**: S4에 missing data를 end-to-end 통합, mask feature의 중요성 입증
- **한계**: Cross-pattern robustness 미평가
- **우리와의 관계**: Mask feature 중요성은 우리 Layer 1 결과와 일치

#### (6) Imputation-free methods (GRU-D, BRITS, mTAN, SAITS)
- Mask를 input feature로 사용하는 접근
- 모두 **fixed missing pattern** 가정. Pattern shift robustness 미평가

### 7.3 우리의 Positioning

| Dimension | 기존 연구 | 우리 연구 |
|---|---|---|
| Evaluation | Same pattern for train/test | **Train pattern A → Test pattern B** |
| Robustness | Missing rate 변화 | **Missing pattern 변화 (어떤 센서가 missing)** |
| Motivation | 시뮬레이션 (MCAR) | **Real-world evidence** (cross-year sensor transitions) |
| 방법론 | Imputation 품질 개선 | **Imputation-free + SSL for missing-invariance** |

### 7.4 주요 참고문헌

- Revisiting MTSF with Missing Values (2024): https://arxiv.org/html/2509.23494
- TSI-Bench (2024): https://arxiv.org/abs/2406.12747
- MNAR Blackout Modeling (2025): https://arxiv.org/html/2601.01480
- Masking the Gaps (NeurIPS WS 2024): https://openreview.net/forum?id=XBtDrlK1Qc
- S4M (2025): https://arxiv.org/abs/2503.00900
- GRU-D (2018): https://arxiv.org/abs/1606.01865
- SAITS (2023): https://arxiv.org/abs/2202.08516
- PriSTI (ICML 2023): https://arxiv.org/abs/2302.09746
- ImputeFormer (KDD 2024): https://arxiv.org/abs/2312.01728
- CSDI (NeurIPS 2021): https://arxiv.org/abs/2107.03502

---

## 8. Proposed Method: Missing-Invariant Contrastive Learning (MICL)

### 8.1 Core Idea

**같은 데이터에 다른 missing pattern을 적용해도 representation이 같아야 한다.**

```
같은 시간대 데이터 x:
  Augmentation 1: 센서 {A,B} missing → repr_1
  Augmentation 2: 센서 {C,D} missing → repr_2

  SSL objective: repr_1 ≈ repr_2  (positive pair)

→ Missing pattern이 바뀌어도 representation 불변
→ Test time에 새로운 missing pattern이 와도 robust
```

### 8.2 이전 실패한 SSL과의 차이

| | 이전 실험 (실패) | MICL (제안) |
|---|---|---|
| Augmentation | Temporal crop, noise 등 | **Missing pattern 자체** |
| 학습 목표 | General representation | **Missing-invariant representation** |
| Missing 정보 | 없음 (0 채움) | Mask channel로 명시적 제공 |
| SSL이 뭘 배우는지 | 불명확 | **"missing pattern에 무관한 데이터의 본질"** |
| 평가 조건 | 같은 missing pattern | **다른 missing pattern** (pattern shift) |

핵심 차이: 이전 SSL은 "뭘 invariant하게 할지" 불명확했음. MICL은 **missing pattern에 invariant**하게 하는 것이 명시적 목표.

### 8.3 방법론 스케치

#### Pretraining Phase

```python
# Input: 관측값 x, 원본 mask m
# (functional 노드의 데이터를 사용하여 ground truth 확보)

# Step 1: Random Missing Augmentation
mask_1 = random_missing_mask(x, pattern='block', rate=0.3)
mask_2 = random_missing_mask(x, pattern='spatial_cluster', rate=0.4)

view_1 = x * mask_1  # mask_1이 0인 곳은 0으로 채움
view_2 = x * mask_2

# Step 2: Encoder (shared weights)
z_1 = encoder(view_1, mask_1)  # [raw + mask] → representation
z_2 = encoder(view_2, mask_2)

# Step 3: Contrastive Loss
# positive: (z_1, z_2) from same x with different masks
# negative: (z_1, z_j) from different x
loss = contrastive_loss(z_1, z_2)
```

#### Augmentation Strategy (Missing Pattern Types)

| Pattern | 설명 | 현실 대응 |
|---|---|---|
| Random point | 개별 (t, node) 랜덤 masking | 간헐적 센서 오류 |
| Block missing | 연속 시간대 masking | 정전, 장비 교체 |
| Spatial cluster | 인접 노드 그룹 masking | 도로 공사, 지역 정전 |
| Channel dropout | 특정 채널 전체 masking | 특정 측정값 센서 고장 |
| Node death | 특정 노드 전 시간 masking | 영구 센서 고장 |

#### Downstream Phase

```python
# Pretrained encoder를 downstream forecasting model 앞에 부착
z = encoder(x_observed, mask)     # pretrained, frozen or fine-tuned
prediction = forecaster(z)        # any backbone (STAEformer, STGCN, ...)

# 또는 end-to-end:
# encoder + forecaster를 jointly fine-tune
# contrastive loss + prediction loss jointly optimize
```

### 8.4 Design Choices (미결정, 실험으로 결정)

| Choice | Options | Notes |
|---|---|---|
| Encoder architecture | Transformer, GNN, MLP | Model-agnostic이면 simple이 좋을 수 있음 |
| Contrastive loss | InfoNCE, BYOL, SimSiam | Negative sample 필요 여부 |
| Augmentation diversity | 2 views vs N views | More views = stronger invariance but slower |
| Downstream integration | Frozen encoder vs fine-tune | Frozen = 더 model-agnostic |
| Pretraining data | 같은 dataset vs cross-dataset | Cross-dataset이면 더 general |
| Missing rate range | 10-50% vs 10-90% | 높은 rate에서의 robustness 중요 |

### 8.5 기대 효과

**Standard setting (같은 missing pattern):**
- Baseline과 comparable (약간 개선 또는 동등)
- 이전 SSL 실패와 달리, missing-aware이므로 추가 정보 제공 가능

**Pattern shift setting (다른 missing pattern):**
- **MICL이 우위** ← 핵심 contribution
- Imputation-based: pattern shift에서 급격한 성능 저하
- Baseline (NULL_VAL=0): 새로운 missing을 구분 못함
- MICL: missing-invariant representation으로 자연스럽게 적응

---

## 9. Experiment Plan

### Phase 1: Cross-Year Stress Test (인프라 있음, 바로 실행 가능)

Cross-year 데이터를 활용한 missing pattern shift evaluation.

**Setup:**
- Train: 2022 Q1 데이터 (dead 센서 set A)
- Test: 2023 Q1 데이터 (dead 센서 set B, 100+개 상태 전환)
- 기존 cross-year 인프라: `eda/concept_drift/cross_year_eval_v3.py`

**비교 대상:**
| Method | 설명 |
|---|---|
| Baseline (NULL_VAL=0) | 기존 방식, missing 구분 없음 |
| Mask-input (8ch) | Mask channel as input feature |
| Mask-aware (8ch + mask loss) | Mask in input + loss |
| Imputation + Baseline | KNN/mean imputation 후 baseline 모델 |
| **MICL (proposed)** | Missing-invariant SSL + downstream |

**Metric:**
- Overall MAE (standard)
- Per-category MAE (functional, partial, major, dead)
- **Healthy-only MAE** (train에서 functional이고 test에서도 functional인 노드만)
- **Pattern-shifted MAE** (train에서 functional → test에서 dead, 또는 그 반대)

### Phase 2: Artificial Missing Pattern Shift (Controlled Experiment)

**Setup:**
- 같은 데이터 (SAN_BERNARDINO, 3개월)
- Train: original missing pattern
- Test: functional 노드에 artificial missing 추가

**실험 변수:**
| Variable | Values |
|---|---|
| Missing rate | 10%, 20%, 30%, 50% |
| Missing type | Random point, Block, Spatial cluster |
| Shift type | Rate increase, Pattern type change, Both |

**기대 결과:**
- Imputation-based: missing rate/type 변화에 따라 급격한 성능 저하
- MICL: graceful degradation (pretraining에서 다양한 pattern을 봤으므로)

### Phase 3: MICL 구현 및 평가

1. Encoder 구현 (Transformer-based, mask-aware)
2. Contrastive pretraining (random missing augmentation)
3. Downstream fine-tuning (STAEformer, STGCN)
4. Phase 1, 2의 stress test에서 평가

### Phase 4: Ablation & Analysis

- Augmentation strategy ablation (어떤 missing pattern이 가장 유용?)
- Encoder size/depth ablation
- Frozen vs fine-tuned encoder
- Representation 시각화 (t-SNE: missing-invariance 확인)

---

## 10. File Directory

### 데이터셋

| Path | 설명 |
|---|---|
| `datasets/xtraffic/SAN_BERNARDINO/` | 원본 5ch 데이터 (105120, 893, 5) |
| `datasets/SAN_BERNARDINO_MASK/` | 8ch 데이터 (flow, occ, speed, mask_f, mask_o, mask_s, tod, dow) |
| `datasets/SAN_BERNARDINO_2022_Q1/` | 2022 Q1 cross-year 데이터 |
| `datasets/SAN_BERNARDINO_2023_Q1/` | 2023 Q1 cross-year 데이터 |
| `datasets/SAN_BERNARDINO_2024_Q1/` | 2024 Q1 cross-year 데이터 |

### 실험 Config 파일

| Config | 설명 | MAE |
|---|---|---|
| `baselines/STAEformer/SAN_BERNARDINO_5ch.py` | STAEformer 5ch baseline | 12.26 |
| `baselines/STAEformer/SAN_BERNARDINO_5ch_mask.py` | STAEformer 8ch mask input | 12.05 |
| `baselines/STAEformer/SAN_BERNARDINO_5ch_mask_aware.py` | STAEformer 8ch mask-aware loss | 12.12 |
| `baselines/STGCN/SAN_BERNARDINO.py` | STGCN 1ch baseline | 14.09 |
| `baselines/STGCN/SAN_BERNARDINO_5ch.py` | STGCN 5ch baseline | 14.66 |
| `baselines/STGCN/SAN_BERNARDINO_5ch_mask.py` | STGCN 8ch mask input | 17.80 |
| `baselines/STGCN/SAN_BERNARDINO_5ch_mask_aware.py` | STGCN 8ch mask-aware loss | 14.43 |

### 구현 파일

| File | 설명 |
|---|---|
| `basicts/metrics/mask_aware.py` | mask_aware_mae, mask_aware_mape, mask_aware_rmse |
| `basicts/runners/runner_zoo/mask_aware_runner.py` | MaskAwareRunner (target_mask 추출) |
| `baselines/STAEformer/arch/staeformer_credibility.py` | STAEformerCredibility (credibility bias) |

### 분석 데이터

| File | 설명 |
|---|---|
| `eda/robust_prediction/per_node_flow_nan_rate.npy` | Per-node flow NaN rate (893,) |
| `datasets/xtraffic/SAN_BERNARDINO/dead_indices.npy` | Dead sensor indices |
| `datasets/xtraffic/SAN_BERNARDINO/major_fail_indices.npy` | Major fail indices |
| `datasets/xtraffic/SAN_BERNARDINO/keep_no_dead.npy` | Dead 제외 노드 인덱스 |
| `datasets/xtraffic/SAN_BERNARDINO/keep_no_dead_major.npy` | Dead+Major 제외 노드 인덱스 |

### EDA 분석 스크립트 및 결과

#### `eda/robustness/` — Test-time Perturbation Analysis
불량 센서가 정상 센서 예측에 미치는 영향 분석. **STGCN이 zero-type corruption에 20-27x 더 취약** 발견.

| File | 설명 |
|---|---|
| `run_perturbation.py` | 통합 실험 스크립트 (STAEformer/STGCN/AGCRN) |
| `README.md` | 실험 설계, 결과, 핵심 발견 |
| `staeformer_summary.json` | STAEformer 24개 config별 요약 통계 |
| `stgcn_summary.json` | STGCN 24개 config별 요약 통계 |
| `agcrn_summary.json` | AGCRN 24개 config별 요약 통계 |
| `*_per_node_degradation.npy` | (24, 893) config별 per-node MAE degradation |
| `*_clean_mae.npy` | (893,) 노드별 clean MAE |

**핵심 발견**: STGCN은 GCN message passing이 zero 값을 직접 전파 → zero-type에 취약. STAEformer는 attention이 zero 패턴을 감지하여 무시 가능하나 noise/spike에는 취약.

#### `eda/post_incident/` — Per-Node MAE & Robustness Analysis
Per-node 에러 분포, incident 영향, counterfactual 분석.

| File | 설명 |
|---|---|
| `analyze_per_node_mae.py` | Per-node MAE 분석 (masked/unmasked/speed) |
| `counterfactual_analysis.py` | Dead node 제거 시 counterfactual 분석 |
| `input_signal_analysis.py` | Worst node 입력 신호 분석 |
| `spatial_propagation.py` | 에러 전파 공간 분석 |
| `FINAL_REPORT.md` | 종합 리포트 |
| `per_node_mae_masked.npy` | (893,) 노드별 masked MAE |
| `per_node_zero_rate.npy` | (893,) 노드별 zero rate |
| `per_node_mean_flow.npy` | (893,) 노드별 평균 flow |

**핵심 발견**: MAE는 scale 의존적 (MAE vs mean_flow r=0.82), nMAE로 보면 균일. Loss imbalance: worst 20% 노드 = 전체 loss의 55%.

#### `eda/robust_prediction/` — Missing Rate 분석

| File | 설명 |
|---|---|
| `per_node_flow_nan_rate.npy` | (893,) 노드별 flow NaN rate |

#### `eda/analyze_attention.py` — Attention Collapse 분석
Dead 노드가 functional 대비 3.5-12x 과도한 spatial attention 수신하는 현상 분석.

#### `eda/concept_drift/` — Cross-Year Concept Drift (인프라 공유)
Cross-year evaluation 인프라. Missing pattern shift의 Phase 1 (cross-year stress test)에서 활용.

| File | 설명 |
|---|---|
| `cross_year_eval_v3.py` | Cross-year evaluation (train year scaler 사용) |
| `README.md` | 전체 concept drift 연구 문서 |
| `functional_indices.npy` | (508,) 3년 모두 functional인 노드 |

### Corruption 실험 (experiments/)

| File | 설명 |
|---|---|
| `experiments/run_corruption_baseline.py` | Flow stuck, noisy, intermittent missing corruption |
| `experiments/analyze_corruption_results.py` | Corruption 결과 분석 |
| `experiments/corruption_analysis_results.json` | 분석 결과 JSON |

### 관련 문서

| 문서 | 설명 |
|---|---|
| `docs/attention_collapse_missing_values.md` | Attention collapse 상세 분석 |
| `docs/representation_learning_experiment_log.md` | 이전 SSL 실험 로그 (negative results) |
| `eda/concept_drift/README.md` | Cross-year concept drift 연구 |
| `eda/robustness/README.md` | Test-time perturbation 실험 설계 및 결과 |
| `eda/post_incident/FINAL_REPORT.md` | Per-node robustness 종합 리포트 |

---

## Quick Start: 이어서 연구하려면?

### 1. 환경 설정
```bash
source ~/.conda/etc/profile.d/conda.sh && conda activate basicts
cd /data/pretrainingbasicts
```

### 2. 현재 상태 확인
- Layer 1 (mask input) ✅ 완료
- Layer 1.5 (mask-aware loss) ✅ 완료
- Related work survey ✅ 완료 (Section 7)
- Phase 2: Artificial missing pattern shift ✅ 완료 (Section 6)
  - **핵심 발견**: mask-aware는 intermittent에 강하고 node death에 약함 (tradeoff)
  - **→ MICL motivation 확립**: 모든 shift type에 robust한 방법 필요
- **MICL 구현 → 다음 단계**

### 3. 다음 우선순위

1. **Phase 3: MICL encoder 구현**
   - Random missing augmentation 모듈
   - Contrastive pretraining pipeline
   - Downstream integration
   - Phase 2 실험에서 MICL의 robustness 검증

2. **Phase 1: Cross-year stress test 구현**
   - 기존 cross-year 인프라 활용 (`eda/concept_drift/cross_year_eval_v3.py`)
   - Mask-aware 모델을 cross-year로 평가
   - Real-world missing pattern shift에서의 성능 저하 정량화
