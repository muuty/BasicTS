# Pattern Bank: Parameter-Efficient Concept Drift Adaptation via Prototype-Constrained Embedding Perturbation

---

## 1. Motivation

### 1-1. The Problem: Temporal Concept Drift

Spatio-temporal forecasting 모델은 학습 시점의 데이터에 최적화된다. 시간이 지나면 교통 패턴이 변화하여 성능이 급격히 하락한다.

| | MAE |
|---|---|
| Same-year (oracle) | 10.40 |
| Cross-year (no adaptation) | 20.17 |
| **Performance gap** | **+94%** |

- 설정: STAEformer, SAN_BERNARDINO 893 nodes, 2022/2023/2024 Q1, 6개 cross-year pair 평균
- 이 gap은 모델 재학습 없이는 해결되지 않음

### 1-2. Why Existing Approaches Fall Short

**Full retraining**: 새 연도 데이터 수 개월 필요 + GPU 비용. 빠른 배포 불가능.

**Transfer learning / fine-tuning**: 전체 모델 fine-tune 시 수일 데이터로는 overfitting 불가피 (452K params vs ~2000 samples).

**LoRA (Low-Rank Adaptation)**: NLP에서 성공했지만, few-shot regime (수 시간~1일)에서는 unconstrained update가 오히려 불안정.

**RevIN (Instance Normalization)**: Scale drift는 해결하지만 pattern drift는 남김.

### 1-3. Our Insight: Drift Decomposition

Concept drift를 두 독립 요소로 분해한다:

```
Total Drift = Scale Drift + Pattern Drift
               (RevIN)       (Pattern Bank)
```

- **Scale drift**: 전반적 교통량 규모 변화 → RevIN으로 zero-shot 해결 (62.2% gap closed)
- **Pattern drift**: 노드별 구조적 패턴 변화 → 잔여 gap 3.69 MAE → **Pattern Bank로 해결**

---

## 2. Method: Pattern Bank Adapter

### 2-1. Key Observation — Node Embedding as "Spatial Memory"

최신 ST 모델들은 learnable node embedding을 사용한다 (STAEformer, GWNET, MTGNN, AGCRN 등). 이 embedding은 각 노드의 고유 교통 패턴을 인코딩하는 **spatial memory** 역할을 한다.

Concept drift 시 이 memory가 outdated → 모델이 잘못된 prior로 예측 → **embedding을 적응시키면 drift를 해결**할 수 있다.

### 2-2. Pattern Bank Structure

Node embedding을 직접 fine-tune하면 파라미터가 너무 많다 (N×d). Pattern Bank은 변화량을 **prototype-constrained low-rank**로 제약한다:

```
P ∈ R^{K×d}     — K개의 drift prototype vector
W ∈ R^{N×K}     — 노드별 prototype mixing weight

delta = softmax(W) · P           # (N, d)
embedding' = embedding + delta    # residual adaptation
```

핵심 설계:
- **Low-rank**: K=8개의 공유 prototype으로 893개 노드의 drift를 표현 → 7,336 params (1.6%)
- **Softmax constraint**: mixing weight 합 = 1 → prototype의 convex combination으로 제약
- **Residual**: 원래 embedding에 delta를 더함 → 학습된 지식 보존

### 2-3. Pattern Bank vs LoRA

|  | Pattern Bank | LoRA |
|---|---|---|
| 수식 | δ = **softmax**(W) · P | δ = A · B |
| 제약 | mixing weight 합 = 1 | 없음 (unconstrained) |
| 해석 | prototype의 가중 평균 | 불가 |
| Params | 동일 (N·K + K·d) | 동일 (N·K + K·d) |

유일한 차이는 **softmax**. 이 한 줄의 제약이 few-shot에서 implicit regularizer로 작동한다 (Section 4 실험으로 검증).

---

## 3. Main Results

### 3-1. RevIN + Pattern Bank — Gap Closure

| Method | Params | Adaptation Data | Avg MAE | Gap Closed |
|---|---|---|---|---|
| No Adaptation | 0 | 0 | 20.17 | 0% |
| RevIN | 0 | 0 | 14.09 | 62.2% |
| RevIN + PB (12h) | 7.3K (1.6%) | 12 hours | 13.08 | 72.6% |
| RevIN + PB (1d) | 7.3K (1.6%) | 1 day | 12.71 | 76.4% |
| RevIN + PB (7d) | 7.3K (1.6%) | 7 days | 11.46 | 89.2% |
| Full PEFT (7d) | 257K (56.8%) | 7 days | 11.86 | 85.1% |
| Same-year oracle | 452K (100%) | 3 months | 10.40 | 100% |

- **35x fewer parameters** than full PEFT, yet **better performance** (11.46 vs 11.86)
- **12시간** 데이터(전체의 0.5%)만으로 gap의 72.6% 해소
- RevIN (scale) + PB (pattern) = 분리된 drift 처리의 효과

### 3-2. Per-Pair Results

| Source→Target | No Adapt | RevIN | +PB 12h | +PB 1d | +PB 7d | Same-year |
|---|---|---|---|---|---|---|
| 2022→2023 | 22.44 | 14.87 | 13.49 | 12.81 | 11.47 | 10.40 |
| 2022→2024 | 12.86 | 11.13 | 11.82 | 11.98 | 10.40 | 10.40 |
| 2023→2022 | 25.04 | 16.13 | 14.33 | 13.65 | 12.37 | 10.40 |
| 2023→2024 | 24.21 | 15.24 | 13.51 | 13.01 | 11.48 | 10.40 |
| 2024→2022 | 13.15 | 11.98 | 11.73 | 11.74 | 11.33 | 10.40 |
| 2024→2023 | 23.35 | 15.22 | 13.56 | 13.05 | 11.73 | 10.40 |
| **Average** | **20.17** | **14.09** | **13.08** | **12.71** | **11.46** | **10.40** |

---

## 4. Key Ablation: Pattern Bank vs LoRA

**질문**: Softmax constraint가 정말 필요한가? 제거하면 (= LoRA) 어떻게 되는가?

동일 파라미터 수, 동일 학습 프로토콜, 유일한 차이는 softmax.

### 4-1. Few-shot Regime — PB Wins

| Adaptation Data | Samples | PB (K=8) | LoRA (K=8) | Gap (LoRA−PB) | PB Win Rate |
|---|---|---|---|---|---|
| 3h | 13 | **13.90** | 14.08 | +0.18 | **6/6** |
| 6h | 49 | **13.53** | 13.96 | +0.43 | **5/6** |
| **12h** | **121** | **13.07** | **13.65** | **+0.57** | **5/6** |
| 1d | 265 | **12.69** | 12.84 | +0.15 | **4/6** |

12h에서 **+0.57 MAE** 차이, 6/6 pairs 중 5~6개에서 PB 승리. Softmax가 극단적 few-shot에서 overfitting을 방지하는 implicit regularizer 역할.

### 4-2. Data-Rich Regime — LoRA Wins

| Adaptation Data | Samples | PB (K=8) | LoRA (K=8) | Gap (LoRA−PB) | PB Win Rate |
|---|---|---|---|---|---|
| 3d | 841 | 11.83 | **11.72** | -0.11 | 2/6 |
| 7d | 1993 | 11.47 | **11.30** | -0.17 | 1/6 |

데이터가 충분하면 unconstrained LoRA가 더 자유로운 적응 가능. 하지만 **실제 배포 시나리오에서는 수 시간~1일 적응이 현실적**이므로 PB가 실용적 우위.

### 4-3. Crossover Point

```
           PB wins                    LoRA wins
    ←────────────────────┼──────────────────────→
    3h   6h   12h   1d   │   3d   7d
                         ~1-3d
```

Crossover는 약 1~3일 사이. **빠른 적응이 필요한 현실 시나리오에서 PB가 LoRA를 압도.**

---

## 5. K Sensitivity Analysis

**질문**: Prototype 수 K에 민감한가?

| K | Params | 3h | 6h | 12h | 1d | 3d | 7d |
|---|---|---|---|---|---|---|---|
| **4** | 3.7K | 13.94 | 13.49 | 13.08 | 12.70 | 11.85 | 11.54 |
| **8** | 7.3K | 13.90 | 13.53 | 13.07 | 12.69 | 11.83 | 11.47 |
| **16** | 14.7K | 13.91 | 13.53 | 13.08 | 12.70 | 11.81 | 11.44 |

**PB는 K에 극도로 robust** — K=4 (3,668 params)도 K=16과 거의 동일한 성능.

반면 LoRA는 K에 민감 (data-rich에서 K=16이 K=4보다 0.26 MAE 개선).

→ PB의 softmax constraint가 K 선택에 대한 robustness를 제공. **K=4~8이면 충분.**

---

## 6. Prototype Interpretability — "The Killer Slide"

7일 데이터로 학습한 PB의 prototype 분석 (2022→2023).

### 6-1. Spatial Clustering

학습된 node-prototype assignment를 지도 위에 시각화하면, prototype이 **지리적으로 군집**됨:
- I-15 (남북 축) vs I-10/I-210 (동서 축)에서 다른 prototype 분포
- 시각화: `eda/concept_drift/pb_analysis/spatial_assignment_*_168h.png`

### 6-2. Sensor Health Discovery

PB가 명시적 label 없이도 **센서 건강 상태를 자동으로 분리**:

| Prototype | Nodes | Dead | Major Fail | Functional | Dead Rate |
|---|---|---|---|---|---|
| Proto 5 | 131 | 57 | 11 | 63 | **52%** |
| Proto 3 | 126 | 28 | 5 | 93 | 26% |
| Proto 0 | 102 | 0 | 1 | 101 | **0%** |
| Proto 7 | 108 | 0 | 0 | 108 | **0%** |

Proto 5 = "dead sensor drift pattern", Proto 0/7 = "functional sensor drift pattern". PB는 dead sensor와 functional sensor에 서로 다른 prototype을 할당하여 각각에 맞는 적응 수행.

### 6-3. Flow Change by Prototype

각 prototype에 할당된 노드들의 실제 traffic flow 변화량이 다름 → prototype이 **물리적으로 의미있는 drift 방향을 포착**:
- 일부 prototype: 교통량 증가 패턴 (+100 vehicles/5min)
- 일부 prototype: 교통량 감소 패턴 (-90 vehicles/5min)
- 시각화: `eda/concept_drift/pb_analysis/flow_change_*_168h.png`

---

## 7. Summary of Evidence

| Claim | Evidence | Section |
|---|---|---|
| Drift는 scale + pattern으로 분해 가능 | RevIN(62.2%) + PB(추가 27%) = 89.2% | §3 |
| PB는 극소 파라미터로 적응 가능 | 7.3K params (1.6%), full PEFT 257K 대비 35x 적음 | §3 |
| Softmax가 few-shot regularizer | PB vs LoRA: 3h~12h에서 PB 5~6/6 승리 | §4 |
| K에 robust | K=4와 K=16 성능 차이 < 0.1 MAE | §5 |
| Prototype이 해석 가능 | 지리적 군집, 센서 건강 분리, drift 방향 포착 | §6 |

---

## 8. Remaining Work

### Main paper에 필요
1. **Multi-model generalization**: STGCN 등 node embedding 없는 모델에 PB 적용 (embedding 추가 후)
2. **추가 dataset**: METR-LA, PEMS-BAY 등 다른 교통 데이터셋에서 검증

### Appendix / Supplementary
3. **DriftAdapter (input-level PB) 실험**: embedding-level이 input-level보다 1.1 MAE 우월 → 개입 위치의 중요성
4. **Learned AE vs Random encoder**: representation 품질이 아닌 개입 위치가 핵심이라는 추가 증거
5. **PB weights-only 실패**: prototype vector 학습이 필수라는 ablation

---

## Appendix A: DriftAdapter Experiments (Input-Level Intervention)

Input 앞에서 PB를 적용하는 model-agnostic 대안. Random 또는 learned encoder/decoder로 representation space에서 PB 작동.

| Method | Params | 7d MAE |
|---|---|---|
| DriftAdapter (random, r=32) | 7.4K | 12.54 |
| DriftAdapter (learned AE) | 7.3K | 13.78 |
| **Pattern Bank (embedding)** | **7.3K** | **11.46** |

- Input-level은 delta가 여러 비선형 변환(input_proj → attn → decoder)을 통과하며 효과 희석
- Learned AE가 random보다 **오히려 나쁨** (catastrophic failure on some pairs) → representation 품질이 아닌 **개입 위치**가 핵심

## Appendix B: Experimental Files

| File | Description |
|---|---|
| `eda/concept_drift/peft_pb_on_instnorm_model.py` | Main result: RevIN + PB |
| `eda/concept_drift/peft_lora_comparison.py` | LoRA vs PB (1d, 3d, 7d) |
| `eda/concept_drift/peft_lora_fewshot.py` | LoRA vs PB (3h, 6h, 12h) |
| `eda/concept_drift/peft_k_sensitivity.py` | K=4,8,16 sensitivity |
| `eda/concept_drift/analyze_pattern_bank.py` | Prototype analysis & visualization |
| `eda/concept_drift/peft_drift_adapter.py` | DriftAdapter (random) |
| `eda/concept_drift/peft_drift_adapter_learned.py` | DriftAdapter (learned AE) |
| `eda/concept_drift/pb_analysis/` | Visualization outputs |
