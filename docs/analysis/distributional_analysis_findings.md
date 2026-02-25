# Distributional Analysis Findings

## Context
Coreset selection에서 "좋은 coreset"이란 무엇인가? 모델 학습 없이 proxy metrics만으로 coreset 품질을 평가하고, 실제 MAE와의 상관관계를 분석.

**데이터**: SAN_BERNARDINO, 3개 method (k_center, k_medoids, graph_cut) × 4 distance (euclidean, temporal, spatial, combined) × 2 ratio (0.3, 0.7) × 2 seed

---

## 1. Metric Taxonomy

### 1.1 측정된 Metrics (17개)

| Category | Metric | 방향 | 설명 |
|----------|--------|------|------|
| **OT-based** | ot_cost | ↓ | Vanilla Sinkhorn transport cost |
| | sinkhorn_div | ↓ | Debiased Sinkhorn divergence |
| **KL/JS** | kl_tod | ↓ | KL divergence on time-of-day distribution |
| | js_tod | ↓ | Jensen-Shannon on time-of-day |
| | kl_dow | ↓ | KL on day-of-week |
| | kl_feature | ↓ | KL on PCA(10) feature distribution |
| | js_feature | ↓ | JS on PCA(10) feature distribution |
| **Combinatorial** | fl_objective | ↑ | Facility location coverage |
| | redundancy | ↓ | Intra-coreset redundancy |
| | information_gain | ↑ | FL - λ·Redundancy |
| **Temporal** | h_tod | ↑ | Time-of-day entropy |
| | h_dow | ↑ | Day-of-week entropy |
| **Coverage** | coverage_gap_mean | ↓ | Mean per-hour relative deviation |
| | coverage_gap_max | ↓ | Max per-hour deviation |
| **Kernel** | mmd_rbf | ↓ | Maximum Mean Discrepancy (RBF) |

### 1.2 Metric Clusters (from correlation heatmap)

Three nearly independent clusters emerged:

1. **Temporal Cluster** (r > 0.88): KL_tod ↔ JS_tod ↔ KL_feature ↔ JS_feature ↔ H_tod ↔ coverage_gap
   - 모두 "시간/feature 분포 매칭"을 측정
   - H_tod와 KL_tod: r = -0.98 → 사실상 동일 정보

2. **OT Cluster** (r = 0.98): OT_cost ↔ Sinkhorn_div
   - Temporal cluster와 r ≈ -0.2 → 거의 독립!
   - OT는 "geometric transport cost", KL은 "probability mass 차이"

3. **Combinatorial Cluster**: FL_objective, redundancy
   - 다른 cluster와 약한 상관 (r = 0.3~0.5)

**핵심**: OT와 KL은 같은 것을 측정하지 않는다. 둘 다 필요.

---

## 2. Method Profiles

### 2.1 Radar Chart Summary (ratio=0.3)

| Method | Temporal Coverage | Feature Coverage | OT Distance | FL Objective | Redundancy |
|--------|:---:|:---:|:---:|:---:|:---:|
| **k_medoids** | ★★★★★ | ★★★★★ | ★★★ | ★★★★ | ★★★ |
| **k_center** | ★★~★★★★ | ★★~★★★ | ★★★ | ★★★★★ | ★★ |
| **graph_cut** | ★ | ★ | ★★★★ | ★★★ | ★★★★★ |

### 2.2 구체적 수치 (ratio=0.3, seed-averaged)

**K_medoids** — 분포 보존의 왕
- KL_tod = 0.001 (거의 0), coverage_gap_mean = 0.04, MMD = 0.00005
- Distance type에 관계없이 동일한 분포 품질
- H_tod = 0.9995 → 24시간 거의 균등 커버

**Graph_cut** — 비중복의 왕, 커버리지의 최하위
- KL_tod = 0.31~0.40 (k_medoids의 300배)
- coverage_gap_max = 1.8~2.0 (특정 시간대 2배 과소대표)
- 하지만 redundancy = 8.5M (k_medoids 10.5M의 80%)

**K_center** — distance type에 극도로 민감
- euclidean: KL_tod = 0.41, coverage_gap = 0.73 (graph_cut보다 나쁨!)
- spatial/combined: KL_tod = 0.05~0.07 (중간)
- temporal: KL_tod = 0.29 (graph_cut 수준)

### 2.3 Graph_cut의 시간대별 커버리지 (A5 histogram)

- **가장 과소대표**: 저녁~야간 (h20, 180% 편차)
- **과대대표**: 새벽~오전
- 원인: Greedy submodular selection이 spatial diversity를 우선시 → 비슷한 시간대 패턴은 중복으로 판단하여 스킵

---

## 3. Proxy Metrics vs MAE (핵심 결과)

### 3.1 전체 상관 (48 merged rows, STGCN)

| Rank | Metric | Spearman ρ | p-value | 의미 |
|------|--------|:---:|:---:|------|
| 1 | **information_gain** | **0.502** | 0.0003 | IG 높을수록 MAE 높음 (반직관적?) |
| 2 | **redundancy** | **-0.500** | 0.0003 | redundancy 낮을수록 MAE 높음! |
| 3 | h_dow | -0.443 | 0.002 | 요일 다양성 ↑ → MAE ↓ |
| 4 | coverage_gap_max | 0.437 | 0.002 | 최대 시간 편차 ↑ → MAE ↑ |
| 5 | js_feature | 0.396 | 0.005 | feature 분포 괴리 ↑ → MAE ↑ |
| 6 | kl_feature | 0.393 | 0.006 | |
| 7 | coverage_gap_mean | 0.379 | 0.008 | |
| 8 | h_tod | -0.377 | 0.008 | 시간 다양성 ↑ → MAE ↓ |
| 9 | kl_tod | 0.372 | 0.009 | |
| 10 | fl_objective | -0.287 | 0.048 | FL 높을수록 MAE 약간 낮음 |
| 11 | mmd_rbf | 0.257 | 0.078 | 유의하지 않음 |
| 12 | sinkhorn_div | -0.246 | 0.092 | 유의하지 않음! |
| 13 | ot_cost | -0.210 | 0.152 | 유의하지 않음! |

### 3.2 핵심 인사이트

1. **OT/Sinkhorn은 MAE를 예측하지 못한다** (ρ=-0.25, p=0.09)
   - Recent baseline의 Sinkhorn=0이지만 MAE가 좋지 않음
   - OT는 geometric proximity를 측정하지만, 학습 성능은 distributional coverage가 더 중요

2. **Redundancy의 역설**: redundancy가 낮은 coreset이 오히려 MAE가 높다 (ρ=-0.50)
   - Graph_cut은 redundancy 최저이지만 temporal coverage 부족 → MAE 높음
   - 어느 정도의 redundancy가 학습에 필요함 (regularization 효과?)

3. **Temporal coverage가 MAE의 가장 강한 predictor**: h_dow, coverage_gap_max, kl_feature, h_tod 모두 유의

4. **Per-method 패턴**:
   - **graph_cut에서 모든 metric이 유의** (ρ=0.62~0.66, p<0.01) → proxy metric이 graph_cut 내부 변동을 잘 포착
   - **k_medoids에서 거의 모든 metric이 무의미** (p>0.05) → 분포가 이미 완벽해서 변동 여지 없음
   - **k_center에서 coverage_gap_max만 유의** (ρ=0.50, p=0.05) → distance type 선택이 중요

---

## 4. Discussion

### C1. Graph_cut의 Temporal Coverage 부족 원인
Graph_cut은 greedy submodular (facility location)으로 "가장 다양한" 포인트를 선택. 하지만 feature space (53K dim)에서 spatial pattern이 temporal pattern보다 훨씬 큰 variance를 차지 → 시간적 다양성 무시.

evidence: PCA의 상위 10 components가 전체 variance의 97%를 설명 → spatial features가 지배적.

### C2. K_medoids의 분포 보존 메커니즘
K_medoids는 "sum of distances to nearest medoid"를 최소화하는 Vector Quantization. 이는 자연스럽게 데이터 분포의 density를 보존. 밀도가 높은 영역에 더 많은 medoid가 배치됨 → 시간대별 샘플 수가 원본과 동일.

### C3. Redundancy의 역설적 역할
직관적으로는 "redundancy 낮음 = 더 많은 정보 = 더 좋은 학습"이지만, 실제로는:
- 어느 정도의 redundancy가 gradient signal의 안정성(noise reduction)에 기여
- Graph_cut의 낮은 redundancy = 모든 샘플이 unique → 각 gradient update의 variance가 높음
- K_medoids의 적당한 redundancy = 비슷한 샘플이 gradient를 reinforcing → 더 안정적 학습

### C4. "좋은 coreset" 정의
Phase A 결과 기반으로, **좋은 coreset = 원본 데이터의 분포를 최대한 보존하는 coreset**.
- Temporal coverage (KL_tod, coverage_gap) > Feature coverage (KL_feature) > Redundancy > OT distance
- K_medoids가 이 기준에서 최적

---

## 5. Implications for Phase B

1. **Deterministic training**: training randomness 제거 후 metric-MAE 상관이 더 강해질 것으로 예상
2. **Multi-model validation**: AGCRN, DCRNN, STID, STAEformer에서도 같은 패턴인지 확인
3. **Distance type 선택 기준**: temporal coverage를 해치지 않는 distance (temporal, combined) 선호
4. **Method 추천**: k_medoids + combined/temporal distance가 가장 안전한 선택

---

## 6. Files Generated

### Scripts
- `scripts/analysis/distributional_metrics.py` — KL/JS divergence 계산
- `scripts/analysis/proxy_metric_analysis.py` — 종합 proxy metric 분석
- `scripts/analysis/additional_coverage_metrics.py` — Coverage Gap + MMD
- `scripts/analysis/proxy_vs_mae_v2.py` — Metric vs MAE correlation (v2)
- `scripts/analysis/visualize_distributional_metrics.py` — 5개 시각화

### Data
- `experiments/result/analysis/full_proxy_metrics_table.csv` — 전체 결과 (60 rows × 22 cols)
- `experiments/result/analysis/full_proxy_metrics_seed_averaged.csv` — Seed 평균

### Visualizations
- `experiments/result/analysis/dist_metrics_A1_correlation_heatmap.png`
- `experiments/result/analysis/dist_metrics_A2_radar_chart.png`
- `experiments/result/analysis/dist_metrics_A3_bar_chart.png`
- `experiments/result/analysis/dist_metrics_A4_scatter_kl.png`
- `experiments/result/analysis/dist_metrics_A5_temporal_hist.png`
- `experiments/result/analysis/dist_metrics_B3_proxy_vs_mae.png`
