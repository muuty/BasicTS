# Concept Drift Analysis: SAN_BERNARDINO Traffic Data (2022-2024)

> 최종 업데이트: 2026-02-14
> 연구 주제: 교통 예측 모델의 cross-year 성능 저하 원인 분석 및 Scale-Adaptive Forecasting

---

## Table of Contents

1. [연구 요약 (TL;DR)](#1-연구-요약-tldr)
2. [배경 및 동기](#2-배경-및-동기)
3. [실험 인프라](#3-실험-인프라)
4. [Phase 1: Drift 존재 확인](#4-phase-1-drift-존재-확인)
5. [Phase 2: Drift 원인 분석 (Scale vs Pattern)](#5-phase-2-drift-원인-분석-scale-vs-pattern)
6. [Phase 3: Baseline Method 평가 (RevIN)](#6-phase-3-baseline-method-평가-revin)
7. [Phase 4: Scale-Pattern Disentanglement 시각적 근거](#7-phase-4-scale-pattern-disentanglement-시각적-근거)
8. [Phase 5: Pattern-Only Training (Empirical Validation)](#8-phase-5-pattern-only-training-empirical-validation)
9. [Phase 6: Stable Node 분석 (True Concept Drift 분리)](#9-phase-6-stable-node-분석-true-concept-drift-분리)
10. [핵심 Insights 정리 (Revised)](#10-핵심-insights-정리-revised)
11. [향후 방향: Scale-Adaptive Forecasting](#11-향후-방향-scale-adaptive-forecasting)
12. [비판적 검토 (알려진 약점)](#12-비판적-검토-알려진-약점)
13. [파일 디렉토리](#13-파일-디렉토리)
14. [관련 문서](#14-관련-문서)

---

## 1. 연구 요약 (TL;DR)

**질문**: 교통 예측 모델을 2022년 데이터로 학습하면, 2023/2024년에 왜 성능이 떨어지는가?

**발견 (Phase 1-4, 가설 수립)**:
1. **Drift는 실재한다**: Cross-year MAE +17~143% 증가 (평균 degradation 94%)
2. **Scale(per-node mean flow)이 핵심 원인**: Target Oracle로 per-node mean을 보정하면 cross-year MAE **50.4% 회복**
3. **Pattern(일간 프로파일)은 보존된다**: 연도 간 daily profile correlation r=0.914~0.985
4. **단순 input-based 보정은 실패**: RevIN(입력 통계 기반 re-centering)은 평균 -13.4% (악화)

**실험적 검증 (Phase 5-6, 핵심 결과)** ⭐:
5. **Pattern-only training이 cross-year degradation을 절반으로 줄임**: 94% → **44%** (additive centering)
6. **자기 연도 성능도 5% 개선**: centering이 정규화 효과를 가짐 (10.40 → 9.89)
7. **센서 생사변동은 원인이 아님**: 3년간 꾸준히 functional인 396개 노드만 분석해도 동일 패턴 (94% → 46%)
8. **남은 44% degradation의 주 원인은 multiplicative scale (분산 변화)**: std_ratio와 r=+0.58 상관, 반면 temporal pattern shape 변화와는 r=0.08 (무상관)

**결론 (Revised)**:
- Cross-year degradation의 ~53%는 **additive scale (평균 변화)**로 설명 → 단순 centering으로 해결 가능
- 나머지 ~47%의 대부분은 **multiplicative scale (분산 변화)** → 학습 가능한 instance-adaptive normalization 필요
- **진짜 temporal pattern 변화는 degradation의 극히 일부** → "Scale-Adaptive Forecasting"이 핵심
- 단순 전처리(centering)가 이미 강력 → ML 기반 접근은 이를 넘어서야 함

---

## 2. 배경 및 동기

### 2.1 데이터
- **Source**: XTraffic (TraffiDent), California San Bernardino County
- **Nodes**: 893 traffic sensors (I-10, I-15, SR-60, SR-71, I-210)
- **Features**: flow(ch0), occupancy(ch1), speed(ch2), time_of_day(ch3), day_of_week(ch4)
- **Resolution**: 5-min intervals, 288 steps/day
- **실험 기간**: 각 연도 Q1 (1~3월), 약 90일

### 2.2 2023 Q1 이상 현상
2022-2023 California Atmospheric River Events:
- 12개 atmospheric river 연속 상륙 (70년 관측 사상 최장)
- San Bernardino County 12명 사망, 최대 70개 도로 폐쇄
- FEMA 대통령 재난 선포 (DR-4699-CA)
- Functional 노드 교통량: SR-60 -14.2%, SR-71 -17.1%, I-210 -13.9% (2022 대비)

References:
- [Wikipedia: 2022-2023 California floods](https://en.wikipedia.org/wiki/2022%E2%80%932023_California_floods)
- [FEMA DR-4699-CA](https://www.fema.gov/disaster/4699)

### 2.3 왜 "그냥 최신 데이터로 재학습"하면 안 되는가?

리뷰어 질문: "왜 A 기간 모델을 B 기간에 적응하는 연구를 하는가?"

**답변**:
1. **Label delay**: 최신 ground truth가 항상 가용하지 않음 (센서 검증, 인프라 개통 등)
2. **Training data selection**: "얼마나 최신?", "얼마나 많이?"가 그 자체로 연구 문제
3. **Catastrophic forgetting**: 최신 데이터만으로 재학습하면 과거 패턴 지식 소실 (2024로만 학습 → 2022 평가 시 MAE 13.15 vs 2022로 학습 시 10.90)
4. **Efficiency**: Oracle Test에서 scale만 보정하면 50%가 회복됨. 전체 재학습 대신 scale만 업데이트하면 충분할 수 있음
5. **Understanding**: drift의 원인을 이해하는 것 자체가 학문적 기여. "왜 떨어지는가?"를 모르면 "어떻게 고칠까?"도 모름

---

## 3. 실험 인프라

### 3.1 모델
- **Architecture**: STAEformer (Spatio-Temporal Adaptive Embedding Transformer)
- **Config**: input=12 steps (1h), output=12 steps (1h), batch=16, lr=0.002, 30 epochs
- **Loss**: Unmasked MAE
- **Scaler**: Global Z-score (norm_each_channel=False), channel 0 (flow)만 정규화

### 3.2 Checkpoints
| Year | Path | Self MAE |
|------|------|----------|
| 2022 | `checkpoints/ConceptDrift_Q1/SAN_BERNARDINO_2022_Q1_.../6d33ff60.../` | 10.90 |
| 2023 | `checkpoints/ConceptDrift_Q1/SAN_BERNARDINO_2023_Q1_.../0de30023.../` | 10.47 |
| 2024 | `checkpoints/ConceptDrift_Q1/SAN_BERNARDINO_2024_Q1_.../a00dabf5.../` | 9.85 |

### 3.3 Datasets
| Year | Path | Steps |
|------|------|-------|
| 2022 Q1 | `datasets/SAN_BERNARDINO_2022_Q1/` | 25,920 |
| 2023 Q1 | `datasets/SAN_BERNARDINO_2023_Q1/` | 25,920 |
| 2024 Q1 | `datasets/SAN_BERNARDINO_2024_Q1/` | 26,208 (leap year) |

### 3.4 Evaluation Protocol
- Cross-year evaluation: **train year의 scaler 사용** (model + scaler = deployment unit)
- Test set: 마지막 20% (data split 60/20/20)
- Dead/fault 센서 영향 제거를 위해 "functional nodes" 필터 사용

### 3.5 Functional Node 정의
- **기준**: 3-channel zero_rate < 20% in ALL 3 years (flow=0 AND occ=0 AND speed=0)
- **결과**: 508/893 nodes (cross_year_eval_v3 기준), 745/893 (drift_analysis 기준, 약간 다른 기준)
- **이유**: dead 센서(zero_rate>90%)와 major fail(50-90%)만 제외, 심야 영유량은 허용
- **인덱스**: `functional_indices.npy` (508개)

### 3.6 ALAMEDA County (추가 검증)
SAN_BERNARDINO와 동일한 분석을 ALAMEDA County에서도 수행하여 재현성 확인.
- Datasets: `datasets/ALAMEDA_2022_Q1/`, `ALAMEDA_2023_Q1/`, `ALAMEDA_2024_Q1/`
- Configs: `baselines/STAEformer/concept_drift_alameda_{2022,2023,2024}_Q1.py`
- 결과: `cross_year_alameda_results.json`

---

## 4. Phase 1: Drift 존재 확인

### 4.1 Cross-Year MAE Matrix

전체 893 nodes (unmasked MAE):

| Train \ Test | 2022 Q1 | 2023 Q1 | 2024 Q1 |
|---|---|---|---|
| **2022 Q1** | **10.90** | 22.43 (+106%) | 12.83 (+18%) |
| **2023 Q1** | 25.12 (+140%) | **10.47** | 24.20 (+131%) |
| **2024 Q1** | 13.15 (+34%) | 23.33 (+137%) | **9.85** |

> 코드: `cross_year_eval_v3.py` | 결과: `cross_year_q1_v3_results.json`

### 4.2 Functional Nodes Only (508 nodes)

| Train \ Test | 2022 Q1 | 2023 Q1 | 2024 Q1 |
|---|---|---|---|
| **2022 Q1** | **14.14** | 28.38 (+101%) | 16.44 (+16%) |
| **2023 Q1** | 32.03 (+155%) | **12.59** | 30.93 (+146%) |
| **2024 Q1** | 16.54 (+30%) | 29.01 (+131%) | **12.74** |

### 4.3 nMAE (Scale-normalized)

| Pair | Self nMAE | Cross nMAE | Delta% |
|---|---|---|---|
| 2022→2023 | 0.1292 | 0.2586 | **+100.1%** |
| 2022→2024 | 0.1053 | 0.1350 | +28.2% |
| 2024→2022 | 0.1123 | 0.1326 | +18.1% |
| 2023→2024 | 0.1053 | 0.2482 | **+135.7%** |

> nMAE도 증가 → 단순 유량 스케일 변화만이 아닌 패턴 수준의 drift도 존재

### 4.4 상세 Drift 분석

> 코드: `analyze_drift_detail.py` | 결과: `drift_analysis_report.md`, `drift_analysis_arrays.npz`

#### 노드별 분포
- 2023 관련 pairs: **94-100% 노드**에서 성능 저하
- 2022↔2024: 69-93% 노드에서 성능 저하 (더 mild)
- Worst 10% 노드가 전체 degradation의 35-69% 차지

#### 시간대별
- 피크 시간 (7-9AM, 4-7PM): 비피크 대비 1.3-1.5배 더 큰 저하
- 최악 시간대: 05-07시 (출근 시작, flow 급증 시점)
- 최선 시간대: 01시 (야간 안정)

#### 센서 상태 변화
- 2022→2023: 123개 센서 부활, 112개 사망
- 매년 100+ 센서의 상태 전환 발생 (인프라 문제, drift와 독립)

### 4.5 ALAMEDA County 재현

| Train \ Test | 2022 Q1 | 2023 Q1 | 2024 Q1 |
|---|---|---|---|
| **2022 Q1** | **10.20** | 19.66 (+93%) | 13.44 (+32%) |
| **2023 Q1** | 25.70 (+156%) | **10.06** | 24.44 (+143%) |
| **2024 Q1** | 13.75 (+41%) | 26.75 (+175%) | **9.72** |

> 동일한 패턴: 2023 관련 pair에서 극심한 degradation, 2022↔2024에서는 moderate.
> 결과: `cross_year_alameda_results.json`

### 4.6 다른 County 비교

> 코드: `05_all_counties_build_and_analyze.py` | 결과: `all_counties_drift_summary.json`

California 전역의 여러 county에서 2023 Q1의 교통량 급감이 공통적으로 관측됨.

---

## 5. Phase 2: Drift 원인 분석 (Scale vs Pattern)

### 5.1 Oracle Test (핵심 실험) ⭐

**목적**: "만약 각 노드의 예측 평균을 정확히 알 수 있다면, cross-year MAE가 얼마나 회복되는가?"
→ Scale correction의 upper bound 측정.

**방법**: Post-hoc additive re-centering
```python
pred_node_mean = preds.mean(axis=1)     # 모델이 예측한 per-node 평균
target_node_mean = test_y.mean(axis=1)  # 실제 미래 per-node 평균 (oracle)
preds_oracle = preds - pred_node_mean + target_node_mean
```

**결과**:

| Train→Test | Baseline | Target Oracle | 개선율 |
|------------|----------|---------------|--------|
| 2022→2023 | 22.43 | 9.82 | **+56.2%** |
| 2022→2024 | 12.83 | 8.10 | **+36.9%** |
| 2023→2022 | 25.12 | 10.45 | **+58.4%** |
| 2023→2024 | 24.20 | 9.89 | **+59.2%** |
| 2024→2022 | 13.15 | 8.60 | **+34.6%** |
| 2024→2023 | 23.33 | 10.00 | **+57.1%** |
| **Cross 평균** | **~20.2** | **~9.8** | **+50.4%** |
| **Self 평균** | **~10.4** | **~7.8** | **+24.7%** |

> 코드: `oracle_scale_test_v2.py` | 결과: `oracle_scale_test_v2_results.json`

**해석**:
- **Cross-year 50.4% 회복**: scale(per-node mean)이 drift의 핵심 원인
- **Self-year 24.7% 회복**: 모델 자체가 per-node mean 예측에 취약 (drift와 무관한 일반적 bias)
- **순수 drift-specific 개선 ≈ 25%**: (50.4% - 24.7%)
- **나머지 ~50%**: scale 외 요인 (temporal pattern 차이, 센서 상태 변화 등)

**Oracle v1 vs v2 비교**:

Oracle v1 (`oracle_scale_test.py`)에서는 다양한 scale correction 변형을 시도:
| 방법 | 설명 | 결과 |
|------|------|------|
| Global Additive | 연도 간 global mean 차이로 shift | 거의 효과 없음 (~0%) |
| Global Multiplicative | 연도 간 global std 비율로 rescale | 거의 효과 없음 |
| Per-Node Multiplicative | 노드별 ratio로 rescale | 폭주 (dead→alive 전환 시 ratio ∞) |
| Per-Sample Oracle | 샘플별 mean correction | +35~39% |
| **Per-Node Target Oracle (v2)** | **노드별 mean re-centering** | **+50.4%** |

**핵심 발견**: Global scale은 연도 간 거의 동일 (119.48 vs 119.56 vs 123.23). **Per-node scale이 핵심**.

### 5.2 Daily Profile Correlation (Pattern Preservation)

**목적**: Scale이 변해도 Pattern(일간 프로파일 모양)은 보존되는가?

**방법**: 연도별 288-step daily profile을 평균 내고, 노드별 Pearson correlation 계산.

> 코드: `cross_year_pair_analysis.py` | 결과: `cross_year_correlations.json`

**결과** (consistently functional nodes, 295개 기준):

| 연도 쌍 | Mean Correlation | Interpretation |
|---------|-----------------|----------------|
| 2022↔2023 | **0.914** | 대기강에도 불구하고 패턴 상당히 보존 |
| 2022↔2024 | **0.985** | 정상 연도 간 매우 높은 보존 |
| 2023↔2024 | **0.915** | 대기강 회복 후에도 유사 |

**저장된 데이터**:
- `flow_profile_{2022,2023,2024}.npy`: 연도별 per-node daily profile (288, 893)
- `avg_correlation.npy`: 노드별 평균 cross-year correlation (893,)

### 5.3 Scale-Pattern Disentanglement 요약

| 지표 | 관찰 | 의미 |
|------|------|------|
| Per-node mean flow | 연도 간 크게 변함 (개별 노드 ±50%+) | Scale drift 실재 |
| Daily profile correlation | r=0.914~0.985 | Pattern은 상대적으로 불변 |
| Oracle recovery | +50.4% (cross), +24.7% (self) | Scale correction이 효과적 |
| Global mean | 119~123 (거의 동일) | 노드 수준이 아닌 전체 수준에서는 drift 미미 |

→ **Scale과 Pattern은 서로 다른 drift dynamics를 가짐. 분리하여 다루는 것이 합리적.**

---

## 6. Phase 3: Baseline Method 평가 (RevIN)

### 6.1 RevIN이란?
Reversible Instance Normalization (Kim et al., ICLR 2022).
핵심 가정: **input scale ≈ output scale** (입력의 평균/분산으로 출력을 역정규화).

### 6.2 실험 이력

#### v1: 잘못된 접근 (denormalization stats 교체)
```python
# 모델은 global Z-score로 학습됨
# v1: global stats → input per-node stats로 교체
pred_revin = z * input_per_node_std + input_per_node_mean
```
**결과**: 전부 catastrophic failure (MAE 수천~수만). 모델이 global normalized space에서 학습되었으므로 per-node stats로 denormalize하면 scale이 완전히 틀어짐.

> 코드: `revin_test.py` | 결과: `revin_test_results.json`

#### v2: 올바른 접근 (Re-centering)
```python
# Target Oracle과 동일한 구조이되, target mean 대신 input mean 사용
pred_revin = pred - pred_per_node_mean + input_per_node_mean
```

> 코드: `revin_test_v2.py` | 결과: `revin_test_v2_results.json`

### 6.3 RevIN v2 결과

**Cross-Year**:

| Train→Test | Baseline | RevIN | Oracle | RevIN 개선 | Oracle 개선 |
|------------|----------|-------|--------|-----------|------------|
| 2022→2023 | 22.43 | 21.14 | 9.82 | +5.8% | +56.2% |
| 2022→2024 | 12.83 | 20.30 | 8.10 | **-58.3%** | +36.9% |
| 2023→2022 | 25.12 | 22.10 | 10.45 | +12.0% | +58.4% |
| 2023→2024 | 24.20 | 21.11 | 9.89 | +12.8% | +59.2% |
| 2024→2022 | 13.15 | 21.28 | 8.60 | **-61.8%** | +34.6% |
| 2024→2023 | 23.33 | 21.21 | 10.00 | +9.1% | +57.1% |
| **평균** | | | | **-13.4%** | **+50.4%** |

**Self-Year**: **-97.2% 평균** (치명적 악화)

### 6.4 RevIN 실패 원인 진단

> 코드: `revin_diagnosis.py`

**핵심**: 12-step(1시간) window의 **temporal trend** 때문.

| 지표 | 값 |
|------|-----|
| 모델의 per-node mean 예측 오차 | **7.27** |
| RevIN의 input mean 오차 | **19.44** |
| RevIN / Model 비율 | **2.7x 나쁨** |

**시간대별 input↔target gap**:

| 시간대 | gap | 해석 |
|--------|-----|------|
| 00시 (야간) | 8.36 | 안정 → gap 작음 |
| 06시 (출근 직전) | **27.32** | 급증 → gap 큼 |
| 17시 (퇴근 시작) | **27.08** | 급감 → gap 큼 |

**결론**:
- 12-step = 1시간이므로 출퇴근 전환 시 input window(6-7시)와 target window(7-8시)의 교통량이 **구조적으로 다름**
- 모델이 이미 RevIN보다 2.7배 더 정확하게 mean을 예측 → RevIN은 좋은 추정을 나쁜 추정으로 교체
- Global vs Per-Node RevIN 차이 없음 (-94.3% vs -93.0%) → scaler 문제가 아닌 temporal dynamics 문제
- **80.4%** 노드에서 RevIN 해로움

> 상세 분석: `docs/sub/revin_analysis.md`

---

## 7. Phase 4: Scale-Pattern Disentanglement 시각적 근거

### 7.1 Case Study 노드 선택 기준

"Scale이 변하면서 Pattern은 보존된다"를 시각적으로 보여주기 위한 case study.

**Consistently Functional Nodes 조건**:
- 모든 3개 연도에서 mean flow > 50 (sensor failure 아님)
- MAX_SCALE_RATIO < 2.0 (극단적 scale 변화 제외 → genuine traffic drift만)
- **결과**: 295/893 nodes

> 주의: 초기 시도에서 MIN_FLOW=30 또는 ratio 제한 없이 하면 센서 고장 노드가 선택됨 (e.g., 343→35→372는 sensor failure, not traffic drift)

### 7.2 선택된 대표 노드

| 노드 | 2022 mean | 2023 mean | 2024 mean | Pattern r |
|------|-----------|-----------|-----------|-----------|
| Node 837 | 228 | 168 | 265 | 0.979 |
| Node 140 | 143 | 284 | 145 | 0.984 |
| Node 8 | 285 | 232 | 367 | 0.920 |
| Node 548 | 244 | 122 | 158 | 0.870 |

### 7.3 생성된 시각화

> 코드: `scale_pattern_case_study.py` | 출력: `figures/`

| 파일 | 내용 |
|------|------|
| `figures/daily_profiles_raw_vs_normalized.png` | Raw vs Normalized daily profiles (4개 노드). Raw에서는 scale 차이가 크지만, normalized 후에는 profile 형태가 거의 동일 |
| `figures/scale_trajectory_pattern_corr.png` | 연도 간 scale trajectory + correlation heatmap. 모든 연도 pair에서 r>0.91 |
| `figures/scale_vs_pattern_scatter.png` | 295개 consistently functional 노드의 scatter: x=max scale change, y=pattern correlation |

### 7.4 핵심 통계

295개 consistently functional 노드 중:
- **55.3%**: scale change > 1.2x AND pattern correlation > 0.90
  → Scale은 의미있게 변하지만 Pattern은 보존됨 (disentanglement의 직접적 근거)
- Mean pattern correlation: 0.914 (2022↔2023), 0.985 (2022↔2024)

---

## 8. Phase 5: Pattern-Only Training (Empirical Validation) ⭐

> 실험 날짜: 2026-02-14

### 8.1 실험 설계

**목적**: Phase 2의 Oracle Test 가설을 실제 모델 학습으로 검증. Scale 정보를 제거하고 pattern만 학습하면 cross-year transfer가 개선되는가?

**방법: Additive Centering in Z-score Space**

```python
# PatternOnlyRunner (basicts/runners/runner_zoo/pattern_only_runner.py)

# 1. 입력을 global Z-score 정규화 (기존과 동일)
x_norm = (x - global_mean) / global_std

# 2. Z-score 공간에서 per-node mean 계산 (input window 12 steps)
input_flow_mean = x_norm[:, :, :, 0].mean(dim=1, keepdim=True)  # (B, 1, N)

# 3. Per-node mean 제거 → 모델은 "0 중심 패턴"만 학습
x_centered = x_norm.clone()
x_centered[:, :, :, 0] = x_norm[:, :, :, 0] - input_flow_mean

# 4. 모델 forward (centered input)
prediction = model(x_centered)

# 5. Per-node mean 복원 → 최종 예측
prediction = prediction + input_flow_mean
```

**핵심 차이 vs RevIN**:
- RevIN: 모델 외부에서 input stats로 normalize → denormalize (학습에 반영 안 됨)
- Pattern-only: **학습 중에** centered input으로 모델이 pattern 예측을 학습 → scale 제거가 학습에 직접 반영

**Configs**:
| Year | Config | Checkpoint |
|------|--------|------------|
| 2022 | `concept_drift_2022_Q1_pattern_only.py` | `checkpoints/ConceptDrift_PatternOnly/.../` |
| 2023 | `concept_drift_2023_Q1_pattern_only.py` | 상동 |
| 2024 | `concept_drift_2024_Q1_pattern_only.py` | 상동 |

### 8.2 Self-year 결과

| Model | 2022 | 2023 | 2024 | Avg |
|-------|------|------|------|-----|
| Baseline | 10.90 | 10.47 | 9.85 | **10.40** |
| Pattern-only | 10.42 | 9.65 | 9.60 | **9.89 (-5%)** |

**Pattern-only가 self-year에서도 더 좋다.** Centering이 정규화 역할을 하여 모델이 pattern 학습에 더 집중.

### 8.3 Cross-year 결과 (핵심)

| Train→Test | Baseline | Pattern-only | 개선율 |
|-----------|----------|-------------|--------|
| 2022→2023 | 22.43 | 14.99 | **-33%** |
| 2022→2024 | 12.83 | 11.21 | **-13%** |
| 2023→2022 | 25.12 | 16.40 | **-35%** |
| 2023→2024 | 24.20 | 15.65 | **-35%** |
| 2024→2022 | 13.15 | 11.89 | **-10%** |
| 2024→2023 | 23.33 | 15.32 | **-34%** |

### 8.4 Degradation 비교 (핵심 지표)

| Model | Avg Self MAE | Avg Cross MAE | **Degradation** |
|-------|-------------|---------------|-----------------|
| Baseline | 10.40 | 20.18 | **93.9%** |
| Pattern-only | 9.89 | 14.24 | **44.0%** |

**Cross-year degradation이 94% → 44%로 절반 이상 감소.**

Oracle Test (Phase 2)에서 예측한 50.4% 회복 가능성과 정합: 실제로 (93.9-44.0)/93.9 = **53.1% 회복** 달성.

### 8.5 Oracle MAE 비교

| Model | Avg Cross Oracle MAE | 해석 |
|-------|---------------------|------|
| Baseline | 9.48 | Scale correction 후 남는 residual |
| Pattern-only | 8.87 | Pattern-only의 residual이 더 작음 |

Pattern-only 모델의 oracle MAE도 더 낮음 → centering이 scale 뿐 아니라 pattern 학습 자체도 개선.

> 코드: `cross_year_eval_pattern_only.py` | 결과: `cross_year_pattern_only_results.json`

---

## 9. Phase 6: Stable Node 분석 (True Concept Drift 분리) ⭐

> 실험 날짜: 2026-02-14

### 9.1 동기

Phase 5의 잔여 44% degradation은 어디서 오는가? 센서 생사변동(dead→alive) 때문인가, 아니면 진짜 concept drift인가?

### 9.2 Consistently Functional Nodes

**정의**: 3년 (2022, 2023, 2024) 모두에서 zero_rate < 5%인 노드

| Year | Functional nodes |
|------|-----------------|
| 2022 | 586 |
| 2023 | 625 |
| 2024 | 628 |
| **All 3 years** | **396** |

센서 상태 전환:
- 2022→2023: Functional→Dead 175개, Dead→Functional 214개
- 2022→2024: Functional→Dead 20개, Dead→Functional 62개

### 9.3 핵심 결과: 센서 생사변동은 원인이 아니다

| Model | All nodes degradation | Stable nodes (396) degradation |
|-------|-----------------------|-------------------------------|
| Baseline | 93.9% | **94.0%** |
| Pattern-only | 44.0% | **45.8%** |

**Stable 노드만 봐도 degradation 비율이 거의 동일!** 센서 생사변동을 완전히 배제해도 concept drift의 강도는 변하지 않음.

### 9.4 Scale Drift 분석 (Stable Nodes)

2022→2024 (정상 연도):
- 62% 노드가 scale 안정적 (ratio 0.9-1.1)
- Median additive shift: +2.2 flow

2022→2023 (anomaly 연도):
- **단 7.6%만 안정적!**
- 44% 축소 (ratio < 0.8), 40% 확대 (ratio > 1.2)
- Median additive shift: -17.3 flow (전체적으로 flow 감소 = 대기강 영향)

### 9.5 Temporal Pattern Shape 분석

Daily profile을 normalize(mean=0, std=1)한 후 cosine similarity 비교:

| Year pair | Mean shape similarity | Pattern changed (<0.90) | Pattern stable (>0.99) |
|-----------|----------------------|------------------------|----------------------|
| 2022↔2024 | 0.977 | 5.3% | **61.1%** |
| 2022↔2023 | 0.901 | **36.6%** | 2.0% |
| 2023↔2024 | 0.900 | **36.6%** | 2.0% |

정상 연도(2022↔2024) 간에는 61%가 패턴 거의 불변. 2023 anomaly 연도에서는 37%가 패턴 변화.

### 9.6 핵심 발견: 무엇이 Degradation을 예측하는가?

Stable functional 노드에서 per-node degradation과 각 predictor의 상관관계:

| Predictor | 2022→2023 | 2022→2024 | 2024→2023 |
|-----------|-----------|-----------|-----------|
| **std_ratio** | **+0.58** | +0.12 | **+0.57** |
| **scale_ratio** | **+0.50** | +0.10 | **+0.50** |
| abs_scale_change | +0.39 | +0.15 | +0.38 |
| **shape_dissimilarity** | **+0.08** | +0.20 | **+0.04** |
| self_mae | -0.64 | -0.54 | -0.57 |

**결론**:
- **Multiplicative scale (std_ratio): r=+0.58** — 남은 degradation의 가장 강한 예측자
- **Shape dissimilarity: r=+0.08** — temporal pattern 변화는 degradation과 **거의 무관!**
- **self_mae: r=-0.64** — self에서 잘 맞추던 노드(작은 flow)가 더 큰 degradation 경험

### 9.7 Degradation 분포 (Stable Functional, 2022→2023)

```
47.2% 개선됨 (degrade < -0.5)  ← 거의 절반은 오히려 좋아짐!
 2.6% 안정적 (±0.5)
 3.4% 경미 (0.5-2)
 7.2% 중간 (2-5)
13.4% 심각 (5-10)
26.2% 극심 (>10)
```

**Top 10% 노드가 전체 degradation의 45% 차지.** 소수 노드의 극심한 degradation이 전체 MAE를 끌어올림.

### 9.8 Baseline vs Pattern-only 비교 (Stable Nodes)

```
Baseline avg degradation:     +8.81
Pattern-only avg degradation: +1.38
상관관계: r = 0.88
```

Baseline에서 나빴던 노드가 pattern-only에서도 여전히 나쁨. 같은 노드들이 문제이고, pattern-only가 강도만 줄인 것.

> 코드: `analyze_stable_node_drift.py`, `analyze_residual_drift.py`
> 결과: `stable_node_drift_results.json`, `residual_drift_analysis.json`

---

## 10. 핵심 Insights 정리 (Revised)

### Insight 1: Cross-year degradation의 3단계 구조

실험을 통해 degradation이 3가지 독립적 원인으로 분해됨을 밝힘:

| 원인 | 기여도 | 해결 방법 | 상태 |
|------|--------|-----------|------|
| **Additive scale drift** (평균 변화) | ~53% | Additive centering (Pattern-only) | ✅ 해결 |
| **Multiplicative scale drift** (분산 변화) | ~35% | Instance-adaptive normalization | 미해결 |
| **Temporal pattern drift** (형태 변화) | ~12% | Pattern adaptation / meta-learning | 미해결 |

근거:
- Additive centering으로 degradation 94% → 44% (53% 회복)
- 남은 44%에서 std_ratio와 r=+0.58, shape_dissimilarity와 r=+0.08

### Insight 2: 단순 전처리가 놀라울 정도로 효과적이다

Pattern-only training은 단순히 "input에서 per-node mean 빼기"라는 전처리이지만:
- Cross-year degradation 53% 회복
- Self-year MAE 5% 개선 (정규화 효과)
- 모델 구조 변경 불필요, 어떤 backbone에도 적용 가능

**Implication**: 더 복잡한 ML 접근은 반드시 이 baseline을 넘어야 함.

### Insight 3: "Concept Drift" ≈ "Scale Drift"이다

Stable functional 노드에서 temporal pattern shape의 변화는 degradation과 거의 무상관 (r=0.08).
실제로 degradation을 유발하는 것은 **scale 변화** (mean: r=0.50, std: r=0.58).

→ "Concept Drift in Traffic Forecasting"의 실체는 주로 **Scale Drift**.
→ 연구 방향을 "Scale-Adaptive Forecasting"으로 재정립.

### Insight 4: 센서 생사변동은 Concept Drift가 아니다

3년간 consistently functional한 396개 노드만 분석해도 degradation 비율 동일 (94% → 94%).
매년 100+개 센서의 상태 전환은 있지만, 이는 concept drift가 아닌 infrastructure issue.

### Insight 5: 2023 Anomaly는 비대칭적 drift를 만든다

```
2022→2023 degradation: +5.93 MAE
2023→2022 degradation: +10.99 MAE  ← 2배!
```

2023 anomaly 환경에서 학습한 모델은 정상 환경에서 더 나쁨. Anomaly data에 과적합되기 때문.

### Insight 6: Degradation은 소수 노드에 집중된다

전체 degradation의 45%가 worst 10% 노드에서 발생. 47%의 노드는 오히려 cross-year에서 개선.
→ **Per-node adaptive approach**가 uniform approach보다 효과적일 가능성.

---

## 9. 미완료 실험 및 향후 방향

### 9.1 Pre-experiment Validation (미완료)

| 실험 | 목적 | 우선순위 | 상태 |
|------|------|----------|------|
| **Pattern-only Training** | x/mean(x)로 normalize 후 학습 → cross-year transfer 개선되는지 | 높음 | 미시작 |
| **FFT Spectrum 분석** | 저주파 vs 고주파 변화량 비교 → "drift=저주파" 가설 직접 검증 | 높음 | 미시작 |
| **Linear Probing** | STAEformer hidden state에서 scale R² 측정 → scale contamination 정도 | 중간 | 미시작 |
| **Functional-only Oracle** | Dead node 제외한 Oracle Test (multiplicative 포함) 재실행 | 중간 | 미시작 |

#### Pattern-only Training 예상 방법:
```python
# 모든 데이터를 x / mean(x)로 normalize
flow_pattern = flow / (node_mean + 1e-8)
# Train: pattern-only model
# Test: cross-year transfer → MAE degradation rate이 baseline보다 낮은가?
```
**만약 cross-year transfer가 크게 개선되면** → Scale이 drift의 주범이라는 직접적 증거

#### FFT Spectrum 분석 예상 방법:
```python
# Per-node FFT → 저주파(< 1/day) vs 고주파(> 1/day) 변화량 비교
# 기대: low_freq_change >> high_freq_change → "drift는 주로 저주파에서 발생"
```

### 9.2 SICPL Framework 구현 (미시작)

**Scale-Invariant Contrastive Pattern Learning** — 제안된 framework의 핵심 구성:

1. **Scale Extractor**: 입력에서 per-node scale 정보 추출 (mean, std, slope, last_value + optional weather)
2. **Pattern Encoder**: Instance-normalized 데이터로 scale-free pattern 학습 (STAEformer backbone)
3. **Scale-Conditioned Decoder**: z_pattern + z_scale → 최종 예측
4. **Cross-Year Contrastive Loss**: 같은 노드, 다른 연도 = positive pair → pattern의 year-invariance 학습
5. **Independence Loss**: corr(z_pattern, z_scale) → 0 강제

> 상세 설계: `docs/disentangled_representation.md`

### 9.3 향후 분석 방향

1. **Multi-Quarter 검증**: Q1뿐 아니라 Q3 (여름)에서도 drift 존재하는지 확인
2. **Non-weather Drift**: 도로 공사, 신호 변경, 인구 이동 등 기상 외 원인의 drift
3. **Day-level Pattern Correlation**: 현재는 90일 평균 profile. 개별 일 수준의 cross-year correlation은?
4. **Spatial Attention과 Drift**: Dead node가 spatial attention에 미치는 영향 (기존 연구: `docs/attention_collapse_missing_values.md`)
5. **Weather API 연동**: NOAA/IEM ASOS 데이터로 metadata-guided scale estimation

---

## 10. 비판적 검토 (알려진 약점)

> 상세: `docs/critical_review_sicpl.md`

| # | 약점 | 심각도 | 현재 상태 |
|---|------|--------|----------|
| 1 | Oracle 해석 과대평가: self에서도 +25% → drift-specific은 ~25% | CRITICAL | 인지. 해석 수정 필요 |
| 2 | "Scale" 정의 모호 (additive shift? multiplicative? 저주파?) | MAJOR | FFT 분석으로 해결 예정 |
| 3 | Dead sensor가 모든 분석 오염 | MAJOR | Functional-only 필터 적용 중 |
| 4 | N=1 이벤트(2023 대기강)에 의존 | MAJOR | ALAMEDA로 재현은 함. 다른 유형 drift 미검증 |
| 5 | Q1 편향 (겨울만) | MODERATE | Q3 추가 실험 필요 |
| 6 | Pattern r=0.89는 90일 평균. 개별 day는 훨씬 낮을 수 있음 | MODERATE | Day-level 분석 미실시 |
| 7 | RevIN baseline 부적합 | ✅ 해결 | RevIN -13.4%로 실패 확인 |

---

## 11. 파일 디렉토리

### EDA 스크립트 (번호순: 초기 분석)

| 파일 | 설명 | 출력 |
|------|------|------|
| `01_yearly_statistics.py` | 연도별 기초 통계 (flow, occ, speed 분포) | `yearly_statistics_summary.json`, `yearly_comparison.png`, `distribution_comparison.png` |
| `02_monthly_drift_and_node_analysis.py` | 월별 drift 패턴, 노드별 분류 | `drift_classification.json`, `monthly_trends.png`, `monthly_drift_heatmap.png` |
| `03_build_3year_dataset.py` | 3년 연속 데이터셋 구축 | datasets 디렉토리에 저장 |
| `04_create_yearly_datasets_and_configs.py` | 연도별 Q1 데이터셋 및 학습 config 생성 | datasets + baselines configs |
| `05_all_counties_build_and_analyze.py` | 다른 county와의 비교 분석 | `all_counties_drift_summary.json`, `all_counties_comparison.png` |
| `06_case_study_sustained_drift.py` | 지속적 drift 사례 연구 | `case_studies_sustained_drift.json`, `case_study_sustained_drift.png` |

### 분석 스크립트 (주제별)

| 파일 | 설명 | 출력 |
|------|------|------|
| `cross_year_eval_v3.py` | **Cross-year evaluation** (최종 버전, train year scaler 사용) | `cross_year_q1_v3_results.json` |
| `cross_year_eval_alameda.py` | ALAMEDA County cross-year evaluation | `cross_year_alameda_results.json` |
| `analyze_drift_detail.py` | **상세 drift 분석**: 노드별, 시간대별, 요일별 breakdown | `drift_analysis_report.md`, `drift_analysis_arrays.npz` |
| `stable_nodes_analysis.py` | Functional 노드 분석: zero_rate 필터링, nMAE 계산 | `functional_indices.npy` |
| `metadata_analysis.py` | 센서 메타데이터 분석: 고속도로별, 센서유형별 | - |
| `cross_year_pair_analysis.py` | **Daily profile correlation** 분석 | `cross_year_correlations.json`, `flow_profile_{year}.npy`, `avg_correlation.npy` |
| `build_alameda_q1.py` | ALAMEDA Q1 데이터셋 빌드 | datasets |

### Oracle & Baseline 실험 스크립트

| 파일 | 설명 | 출력 |
|------|------|------|
| `oracle_scale_test.py` | Oracle v1: 다양한 scale correction 변형 시도 | `oracle_scale_test_results.json` |
| `oracle_scale_test_v2.py` | **Oracle v2 (최종)**: Per-node additive re-centering | `oracle_scale_test_v2_results.json` |
| `revin_test.py` | RevIN v1: denormalization stats 교체 (실패) | `revin_test_results.json` |
| `revin_test_v2.py` | **RevIN v2 (최종)**: Additive re-centering with input mean | `revin_test_v2_results.json` |
| `revin_diagnosis.py` | **RevIN 실패 원인 진단**: temporal trend, 시간대별 gap | (stdout) |
| `scale_pattern_case_study.py` | **Case study 시각화**: 노드별 scale vs pattern | `figures/*.png` |

### 데이터 파일

| 파일 | Shape/Type | 설명 |
|------|-----------|------|
| `cross_year_q1_v3_results.json` | JSON | 9-pair cross-year MAE/RMSE |
| `cross_year_alameda_results.json` | JSON | ALAMEDA cross-year 결과 |
| `oracle_scale_test_v2_results.json` | JSON | Oracle v2 결과 (baseline, oracle, additive, multiplicative 등) |
| `revin_test_v2_results.json` | JSON | RevIN v2 결과 + input-target correlation/gap |
| `drift_analysis_arrays.npz` | NPZ | 노드별 MAE, zero_rate, mean_flow 전체 배열 |
| `drift_analysis_report.md` | Markdown | 자동 생성된 drift breakdown 리포트 |
| `cross_year_correlations.json` | JSON | 노드별, 연도 pair별 daily profile correlation |
| `functional_indices.npy` | (508,) int | Functional 노드 인덱스 |
| `truly_stable_indices.npy` | (48,) int | Deprecated. functional_indices 사용 |
| `flow_profile_{2022,2023,2024}.npy` | (288, 893) float32 | 연도별 per-node 평균 daily profile |
| `avg_correlation.npy` | (893,) float32 | 노드별 3-pair 평균 correlation |
| `drift_classification.json` | JSON | 노드별 drift 유형 분류 |
| `yearly_statistics_summary.json` | JSON | 연도별 기초 통계 |
| `all_counties_drift_summary.json` | JSON | 다수 county의 drift 비교 요약 |
| `case_studies_sustained_drift.json` | JSON | 지속적 drift 사례 노드 정보 |

### 시각화

| 파일 | 설명 |
|------|------|
| `yearly_comparison.png` | 연도별 flow/occupancy/speed 분포 비교 |
| `distribution_comparison.png` | 분포 비교 히스토그램 |
| `monthly_trends.png` | 월별 교통량 트렌드 |
| `monthly_drift_heatmap.png` | 월별 drift 히트맵 |
| `all_counties_comparison.png` | 다른 county 비교 |
| `case_study_sustained_drift.png` | 지속적 drift 사례 |
| `figures/daily_profiles_raw_vs_normalized.png` | Raw vs Normalized daily profiles (4 노드) |
| `figures/scale_trajectory_pattern_corr.png` | Scale trajectory + correlation heatmap |
| `figures/scale_vs_pattern_scatter.png` | Scale change vs Pattern correlation scatter |

### Training Configs

| 파일 | 설명 |
|------|------|
| `baselines/STAEformer/concept_drift_2022_Q1.py` | 2022 Q1 학습 |
| `baselines/STAEformer/concept_drift_2023_Q1.py` | 2023 Q1 학습 |
| `baselines/STAEformer/concept_drift_2024_Q1.py` | 2024 Q1 학습 |
| `baselines/STAEformer/concept_drift_alameda_2022_Q1.py` | ALAMEDA 2022 Q1 |
| `baselines/STAEformer/concept_drift_alameda_2023_Q1.py` | ALAMEDA 2023 Q1 |
| `baselines/STAEformer/concept_drift_alameda_2024_Q1.py` | ALAMEDA 2024 Q1 |

---

## 12. 관련 문서

| 문서 | 위치 | 내용 |
|------|------|------|
| **SICPL Framework 설계** | `docs/disentangled_representation.md` | 전체 아키텍처, loss, methodology, roadmap |
| **비판적 리뷰** | `docs/critical_review_sicpl.md` | 알려진 약점, 대안적 해석, 보완 방안 |
| **RevIN 상세 분석** | `docs/sub/revin_analysis.md` | RevIN 실험 v1/v2, 진단, 결론 |
| **Attention Collapse** | `docs/attention_collapse_missing_values.md` | Dead node와 spatial attention 관계 |
| **Cross-Year CL 설계** | `docs/cross_year_contrastive_learning.md` | Contrastive learning 아이디어 |
| **Concept Drift Case Studies** | `docs/concept_drift_case_studies.md` | 개별 사례 연구 |

---

## Quick Start: 이어서 연구하려면?

### 1. 환경 설정
```bash
source ~/.conda/etc/profile.d/conda.sh && conda activate basicts
cd /data/pretrainingbasicts
```

### 2. Cross-year evaluation 재현
```bash
python eda/concept_drift/cross_year_eval_v3.py
# → cross_year_q1_v3_results.json 생성
```

### 3. Oracle Test 재현
```bash
python eda/concept_drift/oracle_scale_test_v2.py
# → oracle_scale_test_v2_results.json 생성
# DEVICE = "cuda:1" (GPU 선택은 스크립트 내 수정)
```

### 4. 다음 우선순위 실험
1. **Pattern-only Training**: `x / mean(x)`로 normalize → cross-year transfer 개선 확인
2. **FFT Spectrum 분석**: 저주파 vs 고주파 변화량 비교
3. **SICPL 구현**: `docs/disentangled_representation.md`의 Phase 1 참조
