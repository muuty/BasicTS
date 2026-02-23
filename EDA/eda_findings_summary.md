# EDA Findings Summary

## Critical Discovery: Our Assumption Was Wrong

### Original Assumption (design.md)
> "사고 발생 시 특정 노드의 flow가 0으로 급감하지만, 모델은 평소와 유사한 값을 예측"

### Reality from Data

| Metric | Value | Implication |
|--------|-------|-------------|
| Significantly below baseline (z < -2) | **1.4%** | 대부분의 incident에서 flow가 유의미하게 떨어지지 않음 |
| Below baseline at all | 39.2% | 절반 이상은 baseline과 같거나 더 높음 |
| Zero flow cases | 17.5% | 센서 데이터 품질 문제 가능성 |
| Average z-score | 0.08 | 평균적으로 incident 시점 = 정상 시점 |

### Flow Trajectory Around Incident
```
t=-120 min: 148.17
t= -60 min: 154.06
t= -30 min: 156.69
t=   0 min: 159.04  ← incident 발생
t= +30 min: 158.06
t= +60 min: 155.97
```
**Flow가 거의 변하지 않음!**

---

## Possible Explanations

### 1. Incident Metadata Quality Issue
- Incident report가 실제 traffic에 영향을 준 사건만 포함하지 않음
- "Hazard" (12,899건) 대부분은 경미하여 traffic에 영향 없을 수 있음
- 실제 심각한 사고는 극소수

### 2. Sensor Location Mismatch
- Incident 위치와 sensor 위치가 정확히 일치하지 않을 수 있음
- `distance` 필드가 있지만, 영향 범위가 다를 수 있음

### 3. Time Alignment Issue
- Incident report 시간과 실제 traffic 영향 시간에 lag이 있을 수 있음
- 사고 보고가 늦거나, 영향이 늦게 나타날 수 있음

### 4. Traffic Already Adapted
- 사고 발생 시 우회로로 traffic이 분산
- Incident node만 보면 변화가 없어 보일 수 있음

---

## Revised Problem Definition

### Before (Incorrect)
"모델이 incident 시점에서 flow=0을 예측하지 못함"

### After (Corrected)
"**대부분의 incident는 traffic에 유의미한 영향을 미치지 않음**
→ 소수의 severe incident (1-5%)에서만 예측 실패가 발생"

---

## Implications for Design

### 1. Anomaly Injection Strategy 재검토

**기존 설계:**
```python
severity_levels = {
    1: (0.3, 0.5),   # 30-50% 감소
    2: (0.5, 0.8),   # 50-80% 감소
    3: (0.8, 1.0),   # 80-100% 감소
}
```

**현실 반영 수정:**
```python
severity_distribution = {
    'no_change': 0.60,      # 60% - incident이지만 변화 없음
    'mild': (0.05, 0.15),   # 25% - 5-15% 감소
    'moderate': (0.15, 0.40), # 10% - 15-40% 감소
    'severe': (0.40, 0.80),  # 4% - 40-80% 감소
    'critical': (0.80, 1.0), # 1% - 80-100% 감소
}
```

### 2. Evaluation 재정의

**기존:** 모든 incident timestamp에서 MAE 측정

**수정:**
- **Severe incident만 따로 평가** (z-score < -2인 354건)
- Flow drop이 실제로 발생한 case만 타겟팅

### 3. Pre-training Objective 재검토

**기존:** Anomaly injection으로 negative sample 생성

**문제:** 실제 데이터에서 "anomaly"가 거의 없음

**대안 고려:**
1. **Synthetic severity만으로 학습** - 실제 incident data 의존 X
2. **Zero-flow prediction task** - flow=0인 경우 감지하는 task 추가
3. **Sudden change detection** - input→output 급변 예측

### 4. Neighbor Propagation 제거

**이유:**
- Real data에서 correlation: -0.02 (거의 무관)
- Propagation 가정이 데이터로 지지되지 않음

---

## Recommended Next Steps

### Option A: Refocus on Severe Cases Only
1. z-score < -2인 354건만 추출
2. 이 case들의 특성 분석
3. 이 case들만 타겟팅하는 모델 설계

### Option B: Reframe as Rare Event Prediction
1. Incident prediction이 아닌 **extreme value prediction**으로 재정의
2. 전체 data에서 상위 1% outlier 예측 문제
3. Incident metadata 의존성 제거

### Option C: Data Quality Improvement
1. 실제로 traffic에 영향을 준 incident만 필터링
2. Collision (NoInj, UnknInj)만 사용
3. Flow drop이 관찰된 case만 학습에 활용

---

## Questions to Resolve

1. **심각한 사고만 필터링하면 sample 수가 너무 적어지는 문제**
   - 354건으로 학습/평가가 가능한가?

2. **Synthetic anomaly injection이 여전히 유효한가?**
   - Real data에 anomaly가 없어도, synthetic anomaly로 학습하면 효과가 있을 수 있음

3. **문제 정의 자체를 바꿔야 하나?**
   - "Incident prediction" → "Sudden drop prediction"으로 전환?
