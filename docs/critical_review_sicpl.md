# SICPL 비판적 리뷰 (Reviewer/Supervisor 관점)

> 작성일: 2026-02-14
> 목적: 제안된 SICPL framework의 약점, 대안적 해석, 보완 필요 사항을 비판적으로 검토

---

## 1. Oracle Test 해석의 과대평가 (CRITICAL)

### 문제
"Target Oracle로 50.4% 회복 → Scale이 절대적"이라고 해석했으나, 이는 **과대평가**다.

### 근거
- **Self-evaluation에서도 +25% 개선됨**. 이것은 drift와 무관한, 모델 자체의 per-node mean prediction bias.
- 따라서 **drift-specific scale correction = 50% - 25% ≈ 25%**. 이는 "Scale이 상당히 중요" (20-50% 범위)이지, "절대적" (>50%)이 아님.
- 우리가 정의한 기준표에 의하면 "Hybrid approach" 카테고리.

### 보완 방안
- Self-year oracle 개선율을 baseline으로 빼고 순수 drift-specific 개선율 계산
- Cross-year에서만 나타나는 추가 개선을 정확히 분리
- 해석을 "50% 회복" → "25% drift-specific + 25% general bias correction"으로 수정

---

## 2. "Scale"의 정의가 모호 (MAJOR)

### 문제
논문 전체에서 "scale"을 일관되게 정의하지 않았다.

### 구체적 혼동
| 우리가 혼용하는 "Scale" | 실제 수학적 의미 | Oracle에서 테스트한 것 |
|----------------------|----------------|---------------------|
| 평균 유량 크기 | Mean flow level | Additive shift (mean correction) |
| 분포의 scale parameter | Std/variance | 테스트 안 함 (폭주) |
| 저주파 성분 | DC + low-freq | FFT 분석 아직 안 함 |
| 곱셈적 변환 | y = a*x + b | Multiplicative oracle 실패 |

- **Additive re-centering이 작동하고 multiplicative가 실패**한다면, 문제는 "scale"이 아니라 **"bias/mean shift"**.
- "Scale-Invariant"라는 이름이 부적절할 수 있음. "Mean-Adaptive" 또는 "Level-Aware"가 더 정확.

### 보완 방안
- Scale을 명확히 정의: "per-node mean flow level"
- FFT 분석으로 저주파 vs 고주파 변화를 실제 확인 (아직 미완)
- Multiplicative oracle을 dead node 제외 후 재실행

---

## 3. Dead Sensor 오염 — 모든 분석에 영향 (MAJOR)

### 문제
893개 노드 중 120개(13.4%)가 dead, 28개(3.1%)가 major fail. 이들이 거의 모든 분석을 오염시킨다.

### 영향
| 분석 | Dead sensor 영향 |
|------|----------------|
| Global mean (119 vs 120 vs 123) | Dead(=0)가 mean을 크게 낮춤 → 연도 간 차이를 과소평가 |
| Per-node multiplicative oracle | Dead→alive 전환으로 ratio 폭발 → 분석 불가 |
| Cross-year pair FP | Dead 노드가 FP의 주요 원인 |
| Daily profile correlation | Dead 노드 제외 시 r이 달라질 수 있음 |

### 핵심 질문
Functional 노드(745개)만으로 모든 분석을 재실행하면 결과가 크게 달라지는가?
- **Global mean (functional only)**: 2022=122.93, 2023=142.99, 2024=127.60 → 차이가 훨씬 크다!
  - 2022→2023: **+16.3%** 증가 (전체 global로는 +0.07%)
  - 이 차이가 왜 "global mean이 거의 동일"이라고 해석됐는지 재검토 필요

### 보완 방안
- **모든 분석을 functional-only로 재실행**: Oracle test, correlation, FP 분석
- Dead node 처리 전략을 명시적으로 기술
- Multiplicative oracle을 functional 노드만으로 재실행

---

## 4. 단일 이벤트 의존성 (MAJOR)

### 문제
우리의 "concept drift"는 실질적으로 **2023 California 대기강** 하나의 이벤트에 의존한다.

| 연도 쌍 | 의미 | 우리 데이터에서 |
|---------|------|--------------|
| 2022↔2023 | 정상→대기강 | 큰 degradation |
| 2022↔2024 | 정상→정상 | 작은 degradation |
| 2023↔2024 | 대기강→정상 | 큰 degradation |

- N=1 이벤트로 "Scale이 drift의 주범"이라고 일반화할 수 있는가?
- 대기강은 **기상 이벤트** → 당연히 scale(유량 감소)이 주 영향
- 다른 종류의 drift (도로 공사, 신호 변경, 인구 이동)에서는 pattern 변화가 더 클 수 있음

### 보완 방안
- "기상 이벤트 기반 drift"로 scope를 한정하는 것이 정직한 접근
- 또는 비기상 drift 데이터 확보 (난이도 높음)
- ALAMEDA 결과가 "재현"이라고 하지만, 같은 이벤트에 영향받은 같은 지역

---

## 5. Pattern Preservation의 과대 해석 (MODERATE)

### 문제
"r=0.89로 pattern이 보존된다"고 주장하지만:

- r=0.89 → **R²=0.79, 즉 21% 분산이 설명 안 됨**
- 이것은 **90일 평균** daily profile의 상관. 개별 날의 상관은 훨씬 낮을 것
- Contrastive learning에서 positive pair는 개별 sample. 평균이 아닌 개별 day 수준에서의 similarity가 중요
- 만약 개별 day 수준의 r이 0.5 이하라면, "같은 node + 다른 year = positive pair"는 매우 noisy한 supervision

### 보완 방안
- 개별 day 수준의 cross-year correlation 계산 (평균이 아닌)
- Day-to-day 변동의 크기 대비 year-to-year 변동의 크기 비교
- SoftCLT (continuous similarity) 적용 필요성의 근거로 활용

---

## 6. RevIN을 실제로 테스트하지 않았다 (MODERATE) ✅ 해결됨

### 문제
RevIN을 "입력=출력 scale 가정의 한계"로 비판하지만, **실제로 실험하지 않았다**.

### ✅ 실험 결과 (2026-02-14)
Post-hoc RevIN re-centering (pred - pred_mean + input_mean) 테스트 완료:
- **Cross-year 평균: -13.4%** (오히려 악화)
- **Self-year 평균: -97.2%** (치명적 악화)
- Degradation이 큰 경우에만 부분적 도움 (2023→2022: +12%, 2023→2024: +12.8%)
- Input→Target correlation=0.91이지만 absolute gap=19.92로 re-centering이 해로움

**결론**: RevIN은 이 task에서 강력한 baseline이 아님. SICPL의 복잡도 정당화 부담이 줄어들었으나,
동시에 "단순한 input statistics로는 scale correction 불가"를 확인 → 외부 정보 또는 학습된 predictor 필요.

> 코드: `eda/concept_drift/revin_test_v2.py`, 결과: `eda/concept_drift/revin_test_v2_results.json`

---

## 7. SICPL 복잡도 대비 효과 의문 (MODERATE)

### 문제
SICPL = Scale Extractor + Pattern Encoder + Scale-Conditioned Decoder + Cross-Year CL + Independence Loss + Weather Pipeline

| Component | 추가 파라미터 | 추가 하이퍼파라미터 | 추가 인프라 |
|-----------|------------|-----------------|-----------|
| Scale Extractor | ~수천 | d_scale | - |
| Independence Loss | 0 | β | - |
| Scale Prediction Loss | ~수백 | α | - |
| Cross-Year CL | Projection head ~수만 | τ, λ₁ | Multi-year 데이터 |
| Weather Pipeline | Weather encoder ~수천 | - | NOAA API, 전처리 |

총 추가 HP: τ, λ₁, λ₂, λ₃, d_scale, d_projection, plus weather features...

### 핵심 질문
- RevIN (0 parameter) 대비 이 모든 복잡도가 정당화되는 얼마큼의 추가 개선이 있어야 하는가?
- "50% oracle recovery" 중 RevIN이 이미 30%를 잡으면, SICPL은 나머지 20%를 위해 이 복잡도를 감수하는가?

### 보완 방안
- Step-by-step ablation을 통해 각 component의 marginal contribution 정확히 측정
- RevIN → +Scale Extractor → +CL → +Weather 순서로 점진적 추가
- 각 단계의 MAE 개선 vs 복잡도 증가 trade-off 명시

---

## 8. Q1 편향 (MODERATE)

### 문제
모든 실험이 Q1 (1-3월)만 사용. 2023 대기강은 겨울 이벤트.

- 여름(Q3)에는 교통 패턴이 근본적으로 다름 (학교 방학, 바캉스 등)
- 겨울 기상 이벤트가 drift의 유일한 원인이라면, 여름에는 drift 자체가 없을 수 있음
- Q1 결과를 "교통 데이터의 concept drift"로 일반화하기 어려움

### 보완 방안
- 최소 Q3 (7-9월) 추가 실험으로 계절 효과 확인
- 또는 "겨울 기상 이벤트에 의한 concept drift"로 scope 한정

---

## 9. 대안적 가설 미검증 (MINOR-MODERATE)

### 대안 1: Scaler Mismatch가 주범
- Z-score scaler가 2022 통계로 2023 데이터를 normalize → 입력 자체가 왜곡
- Oracle은 출력만 교정. 입력 왜곡은 교정 안 됨
- **실험**: test year scaler로 normalize + train year scaler로 denormalize → 비교

### 대안 2: Spatial Attention 교란
- 센서 상태 변화 (279개 = 31.2%)로 spatial attention이 교란
- Dead node가 갑자기 살아나면 attention 분포가 변함
- **이전 연구 (attention collapse)에서 이미 확인**: dead node가 3.5-12x 더 많은 attention을 받음

### 대안 3: 단순 Data Volume 효과
- Q1 = 90일 = ~5000 test samples. 통계적 파워가 충분한가?
- Cross-year eval에서 scaler mismatch로 인한 systematic bias가 MAE를 지배할 수 있음

---

## 10. 강점 (인정하는 부분)

반드시 기록해야 할 강점들:

1. **Cross-county 재현성**: SAN_BERNARDINO + ALAMEDA에서 동일한 drift 패턴 확인
2. **풍부한 pre-experiment validation**: 모델 학습 전에 아이디어 타당성을 데이터로 검증
3. **Oracle test 설계**: upper bound를 먼저 확인하는 방법론적 접근이 우수
4. **실용적 관점**: Weather API 가용성까지 확인 — 실제 배포 가능성 고려
5. **Natural augmentation**: 인위적 augmentation이 아닌 real-world variation 활용 아이디어 자체는 novel

---

## 11. 다음 단계 우선순위 (제안)

비판적 검토 결과를 바탕으로 한 우선순위:

### 즉시 실행 (Before any model training)
1. ~~**RevIN baseline 실험**~~ ✅ 완료 — RevIN은 -13.4%로 실패. 강력한 baseline 아님 확인
2. **Functional-only 재분석** — Oracle test, correlation 등 모든 분석을 745개 functional 노드만으로 재실행
3. **Multiplicative oracle (functional only)** — dead node 제외 후 재실행
4. **FFT Spectrum 분석** — "저주파=scale" 가설의 직접 증거

### 가설 수정 필요 시
- "Scale-Invariant" → "Level-Adaptive" 또는 "Mean-Aware" 리프레이밍 고려
- Additive vs multiplicative drift의 구분을 명확히

### 장기적
- Q3 데이터 추가
- Non-weather drift 데이터 확보
- RevIN이 잘 되면 "RevIN + CL"이라는 더 간결한 프레임워크 고려
