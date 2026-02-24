# RevIN Post-hoc Test: 교통 데이터 단기 예측에서의 부적합성 분석

> 작성일: 2026-02-14
> 결론: RevIN은 단기 교통 예측(12-step, 1시간)에 구조적으로 부적합. Baseline으로 부적절.

---

## 1. 배경

RevIN (Reversible Instance Normalization, Kim et al., ICLR 2022)은 시계열 예측에서 distribution shift를 다루는 대표적 방법.
핵심 가정: **input scale ≈ output scale** (입력의 평균/분산으로 출력을 역정규화).

SICPL 연구에서 RevIN을 baseline으로 검토한 이유:
- Scale correction의 가장 단순한 접근
- Zero parameter 추가로 구현 간단
- RevIN만으로 cross-year degradation이 해결되면 SICPL의 복잡도가 정당화되지 않음

---

## 2. 실험 설계

### 2.1 Post-hoc RevIN (v1 — 잘못된 접근)
기존 학습된 모델의 denormalization stats를 input window stats로 교체.

```python
# 모델은 global Z-score로 학습됨: output = z * global_std + global_mean
# v1: global stats 대신 input window stats 사용
pred_revin = z * input_per_node_std + input_per_node_mean
```

**결과**: 전부 실패 (MAE 수천~수만). 원인: 모델이 global normalized space에서 학습되었으므로, per-node stats로 denormalize하면 scale이 완전히 틀어짐.

> 코드: `eda/concept_drift/revin_test.py`

### 2.2 Post-hoc RevIN (v2 — 올바른 접근: Re-centering)
Target Oracle과 동일한 구조이되, target mean 대신 input mean 사용.

```python
# Target Oracle (upper bound):
pred_oracle = pred - pred_per_node_mean + target_per_node_mean

# RevIN re-centering (input mean 사용):
pred_revin = pred - pred_per_node_mean + input_per_node_mean
```

> 코드: `eda/concept_drift/revin_test_v2.py`
> 결과: `eda/concept_drift/revin_test_v2_results.json`

---

## 3. 결과

### 3.1 Cross-Year 성능

| Train→Test | Baseline | RevIN | Oracle | RevIN 개선 | Oracle 개선 |
|------------|----------|-------|--------|-----------|------------|
| 2022→2023 | 22.43 | 21.14 | 9.82 | +5.8% | +56.2% |
| 2022→2024 | 12.83 | 20.30 | 8.10 | -58.3% | +36.9% |
| 2023→2022 | 25.12 | 22.10 | 10.45 | +12.0% | +58.4% |
| 2023→2024 | 24.20 | 21.11 | 9.89 | +12.8% | +59.2% |
| 2024→2022 | 13.15 | 21.28 | 8.60 | -61.8% | +34.6% |
| 2024→2023 | 23.33 | 21.21 | 10.00 | +9.1% | +57.1% |
| **평균** | | | | **-13.4%** | **+50.4%** |

### 3.2 Self-Year 성능

| Year | Baseline | RevIN | Oracle | RevIN 개선 |
|------|----------|-------|--------|-----------|
| 2022 | 10.90 | 21.13 | 8.23 | **-94.0%** |
| 2023 | 10.47 | 20.20 | 7.61 | **-93.0%** |
| 2024 | 9.85 | 20.15 | 7.65 | **-104.6%** |
| **평균** | | | | **-97.2%** |

### 3.3 패턴 분석
- Degradation이 큰 경우 (>13 MAE): RevIN이 부분적 도움 (+9~13%)
- Degradation이 작은 경우 (<4 MAE): RevIN이 치명적 (-58~-62%)
- Self-year: 항상 치명적 (-93~-105%)

---

## 4. 원인 분석 (진단)

> 코드: `eda/concept_drift/revin_diagnosis.py`

### 4.1 핵심: 12-step window의 temporal trend

| 지표 | 값 |
|------|-----|
| 모델의 per-node mean 예측 오차 | **7.27** |
| RevIN의 input mean 오차 | **19.44** |
| RevIN / Model 비율 | **2.7x 나쁨** |

모델의 mean 추정이 RevIN의 input mean보다 2.7배 정확함.

### 4.2 시간대별 input↔target gap

| 시간대 | gap | trend | 해석 |
|--------|-----|-------|------|
| 00시 (야간) | 8.36 | -7.51 | 안정 → gap 작음 |
| 06시 (출근 직전) | 27.32 | +20.14 | 급증 → gap 큼 |
| 09시 (출근 완료) | 12.78 | +3.78 | 안정화 |
| 12시 (점심) | 12.39 | +7.24 | 안정 |
| 17시 (퇴근 시작) | 27.08 | -23.98 | 급감 → gap 큼 |
| 22시 (야간 진입) | 23.43 | -22.99 | 급감 |

**12 step = 1시간**이므로 출퇴근 전환 시 input window (6-7시)와 target window (7-8시)의 교통량이 구조적으로 다름.

### 4.3 Global vs Per-Node: 차이 없음

| Method | Self-year MAE | 개선율 |
|--------|-------------|--------|
| Baseline | 11.25 | - |
| Global RevIN | 21.87 | -94.3% |
| Per-Node RevIN | 21.72 | -93.0% |
| Target Oracle | 8.45 | +24.9% |

Global/Per-Node scaler 차이가 아닌, **temporal dynamics**가 원인.

### 4.4 노드별 영향

- **80.4% 노드에서 RevIN 해로움**, 19.6%만 도움
- 고유량 노드 (mean 300-633): MAE +40~50 증가 (절대 gap이 큼)
- 저유량 노드 (mean 11-27): MAE -2~-6 감소 (변동 자체가 작음)

---

## 5. 결론

### RevIN이 이 문제에 부적합한 이유
1. **설계 의도 불일치**: RevIN은 LTSF (96-720 step)용. 긴 window에서 mean/std가 안정적
2. **가정 위반**: 12-step (1시간) window에서 input scale ≠ output scale이 구조적으로 보장됨
3. **모델이 이미 우수**: STAEformer의 adaptive embedding + attention이 input으로부터 per-node scale을 이미 잘 추출

### Baseline으로 부적절한 이유
- Self-year에서 -97%는 "method가 task에 맞지 않음"을 의미
- Retrain해도 12-step window의 temporal trend 문제는 해결 불가
- 논문에서 한 줄 언급으로 충분: "RevIN assumes input-output scale consistency, which is violated in short-horizon traffic forecasting (1-hour windows, mean gap = 19.4)"

### SICPL에 대한 시사점
- 단순 input statistics 기반 scale correction은 불충분
- Scale Extractor는 instantaneous mean이 아닌 broader context (일간 패턴, 기상 정보 등) 필요
- Time-of-day conditioning으로 "6시의 100 → 7시의 300" 학습 필요
