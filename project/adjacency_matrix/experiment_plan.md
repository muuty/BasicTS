# Adjacency Matrix Methods for GNN Traffic Prediction: Experiment Plan

## 1. Research Overview

### Research Question
GNN 기반 교통 예측 모델들이 사용하는 adjacency matrix 방법론에 대한 best practice가 확립되어 있지 않다. 동일한 adjacency method set을 여러 GNN architecture에 걸쳐 체계적으로 비교하고, 실용적 가이드라인을 도출한다.

### Research Gap
- 기존 survey (STG4Traffic 등): 방법론 분류만 수행, 실험적 비교 없음
- 개별 논문: 자기 모델 내 ablation만 수행 (model-specific)
- CLEAR (TKDE 2025): 자체 representation bootstrapping만 테스트
- **Cross-model controlled experiment는 부재**

### Positioning
> To our knowledge, one of the first large-scale controlled cross-architecture studies on adjacency matrix methods for GNN traffic prediction. 통제 수준, 재현성, 실용적 가이드라인에 novelty를 둔다.

### Contributions (예상)
1. **Cross-model controlled experiment** — 동일 프로토콜 하에서 같은 adjacency를 여러 모델에 적용
2. **Full spectrum coverage** — static predefined ~ dynamic attention 통합 비교
3. **Parameter sensitivity** — threshold, k, σ 등의 영향 분석
4. **Practical guidelines** — 모델/데이터 특성에 따른 adjacency 선택 가이드
5. **Reproducibility** — 공정 비교 프로토콜, 코드, 실험 로그 공개

---

## 2. Fair Comparison Protocol (공정 비교 프로토콜)

### 2.1 고정 변수 (모든 실험에서 동일)

| 항목 | 설정 | 비고 |
|------|------|------|
| Data split | Chronological 7/1/2 (train/val/test) | CLEAR과 동일 |
| Lookback (T) | 12 steps (1시간) | 표준 설정 |
| Horizon (L) | 12 steps (1시간) | 표준 설정 |
| Early stopping | Patience 15 epochs on val MAE | 모든 모델 동일 |
| Max epochs | 100 | |
| Seeds | 3개 (42, 123, 456) | 평균±표준편차 보고 |
| Optimizer | Adam, lr=1e-3 | 모델별 기본 설정 유지 |
| Batch size | 모델별 기본값 유지 | 메모리 제약으로 통일 어려움 |
| Normalization | Z-score (norm_each_channel=False) | |

### 2.2 Data Leakage 방지 규칙

**원칙: Adjacency matrix는 반드시 train split에서만 생성**

| Adjacency Method | Leakage 위험 | 대응 |
|------------------|-------------|------|
| Distance (Gaussian) | 없음 (static) | 도로 거리이므로 split 무관 |
| Binary connectivity | 없음 (static) | 도로 연결이므로 split 무관 |
| k-NN (distance) | 없음 (static) | 도로 거리 기반 |
| **Correlation-based** | **높음** | **train 구간의 traffic data로만 correlation 계산** |
| **DTW Similarity** | **높음** | **train 구간의 traffic data로만 DTW 계산** |
| Adaptive (E·Eᵀ) | 없음 | 학습 과정에서 자동 생성 |
| Identity / Fully Connected | 없음 | 데이터 무관 |

```python
# 구현 시 반드시 적용
def generate_data_driven_adjacency(data, train_ratio=0.7):
    """Correlation, DTW 등 데이터 기반 adjacency는 train 구간만 사용"""
    train_end = int(len(data) * train_ratio)
    train_data = data[:train_end]
    return compute_adjacency(train_data)  # test 데이터 절대 사용 금지
```

### 2.3 Adjacency 정규화 통일 규칙

모델마다 요구하는 adjacency 형태가 다르므로, **공통 변환 레이어**를 정의:

```
Raw Adjacency (method output)
    ↓
[공통 변환 레이어]
    ├── symmetric: (A + Aᵀ) / 2  (대칭화, 필요 시)
    ├── row_normalize: D⁻¹A
    ├── doubletransition: [D_o⁻¹A, D_i⁻¹Aᵀ]  (DCRNN/GWNet용)
    ├── normlap: I - D⁻¹/²AD⁻¹/²  (STGCN용)
    └── raw: 변환 없음
```

| Model | 요구 형태 | 변환 |
|-------|----------|------|
| GWNet | doubletransition (list of 2) | `[D_o⁻¹A, D_i⁻¹Aᵀ]` |
| MTGNN | raw (single matrix) | None |
| DGCRN | doubletransition (list of 2) | `[D_o⁻¹A, D_i⁻¹Aᵀ]` |
| STGCN | normlap | `I - D⁻¹/²AD⁻¹/²` |

**핵심**: 모든 method의 output은 동일한 raw adjacency → 모델별 변환 함수를 거쳐 입력. 이렇게 하면 method 차이만 비교 가능.

### 2.4 통계 검정

| 항목 | 방법 |
|------|------|
| 중심 경향 | Mean ± Std (3 seeds) |
| 모델간 비교 | Paired t-test (동일 seed 매칭) |
| 전체 순위 | Friedman test + Nemenyi post-hoc |
| 효과 크기 | Wins/Ties/Losses 테이블 |
| 유의 수준 | α = 0.05 |

---

## 3. Model Categories & Selection

CLEAR (TKDE 2025)의 C1-C4 분류 체계를 기반으로 모델 선정.

### Category Definitions (from CLEAR paper, Section IV-E)
| Cat | Definition | Adjacency Handling |
|-----|-----------|-------------------|
| **C1** | Static adjacency (geo-distance 기반 고정) | 사전정의된 A를 graph conv에 직접 사용 |
| **C2** | Adaptive adjacency (learnable embedding E₁E₂ᵀ) | 학습 가능한 node embedding에서 adjacency 생성 |
| **C3** | Spatial-temporal attention | Attention mechanism으로 spatial dependency 표현 |
| **C4** | Spatial-temporal representation learning | 학습된 embedding을 직접 사용하여 예측 |

### Selected Models (Phase별)

#### Pilot & Core (Phase 1a, 1b)
| Cat | Model | Graph Conv Type | Adj Swap 난이도 | Phase |
|-----|-------|----------------|---------------|-------|
| C2 | **GWNet** | Diffusion + adaptive | **Easy** | 1a (pilot) |
| C2 | **MTGNN** | Mixprop diffusion | **Easy** | 1a (pilot) |
| C4 | **DGCRN** | Diffusion (dynamic + predefined) | Medium | 1b |
| - | **STID** | MLP (no graph) | N/A (baseline) | 1a (pilot) |

#### Extension (Phase 확장, 신호 확인 후)
| Cat | Model | Graph Conv Type | Adj Swap 난이도 | Phase |
|-----|-------|----------------|---------------|-------|
| C1 | **STGCN** | Chebyshev spectral | Hard | 확장 |
| C1 | **STGODE** | ODE-based diffusion | Hard | 확장 |
| C2 | **AGCRN** | Adaptive Chebyshev | N/A (fully learned) | 확장 |
| C3 | **STWave+** | Sparse attention + wavelet | Medium | 확장 |
| C4 | **STAEFormer** | Full attention | N/A (별도 section) | Phase 2 |

---

## 4. Adjacency Methods (Independent Variables)

### Part 1: Static Adjacency Methods (C1/C2 모델 대상)

**Core methods (pilot에 포함):**

| Method | 설명 | Leakage 위험 |
|--------|------|-------------|
| **Distance (Gaussian)** | exp(-d²/σ²), threshold 적용 | 없음 |
| **k-NN (distance)** | 가장 가까운 k개 노드만 연결 | 없음 |
| **Correlation-based** | train data의 Pearson correlation | **train-only** |
| **Identity** | 단위행렬 (no spatial info) | 없음 |

**Extended methods (확장 실험):**

| Method | 설명 | Leakage 위험 |
|--------|------|-------------|
| Binary Connectivity | 도로 연결 여부만 0/1 | 없음 |
| DTW Similarity | Dynamic Time Warping 유사도 | **train-only** |
| Adaptive (E·Eᵀ) | Learnable node embedding | 없음 |
| Fully Connected | 모든 노드 연결 | 없음 |

### Part 2: Dynamic Adjacency Methods (C3/C4 모델 대상)

**Note**: 이 실험들은 "adjacency method 비교"가 아니라 "architecture intervention 비교"로 분리 보고.

| Method | 설명 | 적용 방식 |
|--------|------|----------|
| **Attention bias** | Attention score에 distance 기반 bias 추가 | α_ij = softmax(QKᵀ/√d + B_ij) |
| **Attention mask** | Graph neighbor만 attend 허용 | α_ij = softmax(QKᵀ/√d) * M_ij |
| **Relative spatial encoding** | Key/Query에 spatial encoding 추가 | 별도 spatial embedding |
| **Graph-regularized attention** | Attention이 graph와 유사하도록 loss 추가 | L_reg = \|\|α - A\|\|² |

---

## 5. Datasets

| Dataset | Nodes | Time Range | Metric | Interval | Phase |
|---------|-------|-----------|--------|----------|-------|
| **METR-LA** | 207 | 2012.03-06 | Speed | 5min | Pilot |
| **PeMS04** | 307 | 2018.01-02 | Flow | 5min | Pilot |
| **PeMS-BAY** | 325 | 2017.01-05 | Speed | 5min | 확장 |
| **PeMS08** | 170 | 2016.07-08 | Flow | 5min | 확장 |

Pilot에서는 METR-LA (speed) + PeMS04 (flow) 2개만 사용하여 speed/flow 모두 커버.

---

## 6. Phased Experiment Design

### Phase 1a: Pilot (신호 확인)
```
4 core adj methods × 2 models (GWNet, MTGNN) × 2 datasets × 3 seeds
+ STID baseline × 2 datasets × 3 seeds
+ Identity/FullyConnected baseline × 2 models × 2 datasets × 3 seeds
= (4×2×2×3) + (1×2×3) + (2×2×2×3) = 48 + 6 + 24 = 78 runs
```
- 핵심 질문: "adjacency method에 따라 유의미한 성능 차이가 있는가?"
- Go/No-go 기준: method 간 MAE 차이가 seed 변동보다 유의미하게 클 때 확장

### Phase 1b: Core Extension (신호 확인 시)
```
4 core adj methods × 1 model (DGCRN) × 2 datasets × 3 seeds
+ extended adj methods × 3 models × 2 datasets × 3 seeds
= 24 + (4×3×2×3) = 24 + 72 = 96 runs
```

### Phase 2: Dynamic Adjacency (별도 section)
```
4 dynamic methods × 2 models (STAEFormer, STWave+) × 2 datasets × 3 seeds
= 48 runs
```
- **별도 section으로 보고**: "adjacency 비교"가 아닌 "architecture intervention 비교"

### Phase 3: Parameter Sensitivity (핵심 method만)
```
Distance σ: 5 values × 1 model × 2 datasets × 3 seeds = 30
k-NN k: 5 values × 1 model × 2 datasets × 3 seeds = 30
Correlation threshold: 4 values × 1 model × 2 datasets × 3 seeds = 24
= 84 runs
```

### Phase 4: Full Extension (논문 완성 시)
```
확장 데이터셋 (PeMS-BAY, PeMS08) + Hard 모델 (STGCN, STGODE)
```

### 총 실험 수

| Phase | Runs | 우선순위 |
|-------|------|---------|
| 1a (Pilot) | 78 | **최우선** |
| 1b (Core) | 96 | 신호 확인 후 |
| 2 (Dynamic) | 48 | 별도 track |
| 3 (Sensitivity) | 84 | Phase 1 완료 후 |
| 4 (Extension) | TBD | 논문 완성 시 |
| **Total (core)** | **~306** | |

### Evaluation Metrics
- **MAE** (primary), **RMSE**, **MAPE**
- Horizon별: 15min (h3), 30min (h6), 60min (h12)
- 학습 시간 (training wall-clock time)
- Mean ± Std (3 seeds), Wins/Ties/Losses

---

## 7. Baseline System

"그래프가 실제로 필요한가?" 질문에 답하기 위한 baseline:

| Baseline | 설명 | 역할 |
|----------|------|------|
| **STID** | MLP + spatial-temporal identity (no graph) | Graph 필요성 검증 |
| **Identity adj** | A = I (자기 자신만 참조) | Spatial info 없음 baseline |
| **Fully Connected** | A = 11ᵀ (모든 노드 연결) | Graph structure 없음 baseline |
| **Original adj** | 각 모델의 원래 adjacency method | 재현 baseline |

이 4개는 모든 Phase에서 동일 프로토콜로 포함.

---

## 8. Implementation Plan

### Step 1: Adjacency Matrix Generation Module
```python
# project/adjacency_matrix/adjacency_generator.py
def generate_adjacency(method, dataset_path, **params):
    """
    method: 'distance_gaussian', 'binary', 'knn', 'correlation', 'dtw', 'identity', 'fully_connected'
    Returns: np.ndarray (N, N) — raw adjacency (정규화 전)
    """

def normalize_adjacency(adj, norm_type):
    """
    norm_type: 'raw', 'symmetric', 'row_normalize', 'doubletransition', 'normlap'
    Returns: normalized adjacency (or list for doubletransition)
    """
```

### Step 2: Model Wrapper (Easy 모델만 먼저)
```python
# project/adjacency_matrix/model_wrapper.py
def inject_adjacency(config, adj_matrix, model_name, norm_type):
    """
    model_name에 따라 config의 adjacency 설정을 교체
    - GWNet: config MODEL_PARAM['supports'] 교체
    - MTGNN: config MODEL_PARAM['predefined_A'] 교체
    - DGCRN: config MODEL_PARAM['predefined_A'] 교체
    """
```

### Step 3: Config Generator
실험 조합에 따른 config 파일 자동 생성.
```python
# project/adjacency_matrix/generate_configs.py
# Phase × Model × Adj Method × Dataset × Seed → config file
```

### Step 4: Experiment Execution
Ray queue 시스템으로 실험 관리.

---

## 9. Reproducibility & Logging

### 9.1 Checkpoint Naming Convention
```
checkpoints/{model}_{adj_method}/{dataset}_{lookback}_{horizon}/{seed}/
```
예시: `checkpoints/GWNet_distance_gaussian/METR-LA_12_12/seed42/`

### 9.2 Result Schema
```json
{
    "model": "GWNet",
    "adj_method": "distance_gaussian",
    "adj_params": {"sigma": 1.0, "threshold": 0.1},
    "dataset": "METR-LA",
    "seed": 42,
    "metrics": {
        "MAE": 2.69, "RMSE": 5.15, "MAPE": 6.86,
        "h3_MAE": 2.35, "h6_MAE": 2.65, "h12_MAE": 3.08
    },
    "training_time_sec": 3600,
    "epochs_trained": 87,
    "adj_norm_type": "doubletransition",
    "adj_leakage_safe": true
}
```

### 9.3 실패 처리 규칙
- OOM: batch size 절반으로 재시도 (1회)
- NaN loss: seed 변경 후 재시도 (1회)
- 2회 연속 실패: 해당 조합 기록 후 skip

### 9.4 결과 집계
```python
# project/adjacency_matrix/aggregate_results.py
# checkpoints/*/test_metrics.json 자동 수집 → CSV → 분석
```

---

## 10. Expected Analysis & Visualization

1. **Heatmap**: Adjacency Method × Model → MAE (각 데이터셋별)
2. **Bar chart**: 카테고리별 평균 성능 비교
3. **Line plot**: Parameter sensitivity curves
4. **Radar chart**: 각 adjacency method의 다차원 비교 (정확도, 학습속도, 파라미터수)
5. **Statistical test**: Paired t-test + Friedman/Nemenyi + Wins/Ties/Losses
6. **Critical difference diagram**: Nemenyi post-hoc 결과 시각화

---

## 11. Key References

| Paper | Venue | Relevance |
|-------|-------|-----------|
| STG4Traffic | arXiv 2023 | Graph construction taxonomy, benchmark |
| CLEAR | TKDE 2025 | C1-C4 model categorization, bootstrapping |
| STID | CIKM 2022 | MLP baseline (no graph needed?) |
| RAGL | arXiv 2025 | Adaptive graph regularization, scalability |
| Graph WaveNet | IJCAI 2019 | Adaptive + predefined adjacency |
| DCRNN | ICLR 2018 | Diffusion convolution, distance-based adj |
| AGCRN | NeurIPS 2020 | Fully adaptive graph |
| STAEFormer | CIKM 2023 | Attention-based spatial modeling |
| DGCRN | TKDD 2023 | Dynamic graph + predefined hybrid |

---

## 12. Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| "Adaptive가 항상 최고" 결론 | Contribution 약화 | Parameter sensitivity, computational cost, leakage 분석으로 차별화 |
| Adj format 불일치 | 잘못된 결론 | 공통 변환 레이어(Section 2.3) 적용 |
| Data leakage (corr/DTW) | 논문 신뢰도 하락 | Train-only 규칙 프로토콜화 (Section 2.2) |
| 실험 수 과다 | GPU 시간, 관리 부담 | Phase 분리, pilot 먼저 |
| Seed 변동이 method 차이보다 큼 | 유의미한 결론 불가 | 3 seeds + paired test |
| C3/C4 인과 해석 어려움 | Claim 약화 | 별도 section으로 분리 보고 |

---

## Changelog

- **v1 (2026-02-15)**: 초기 계획 작성
- **v2 (2026-02-15)**: 동료 피드백 반영
  - Novelty claim 보수적 재정의
  - 공정 비교 프로토콜 추가 (Section 2)
  - Data leakage 방지 규칙 명시
  - Adjacency 정규화 통일 규칙 추가
  - Phase 1을 1a(pilot)/1b(core)로 분리, 스코프 축소
  - C3/C4 dynamic method를 별도 section으로 분리
  - Baseline 체계 강화 (STID, Identity, FC)
  - Hard model (STGCN, STGODE) 확장 실험으로 후순위화
  - Reproducibility 규칙 추가 (naming, schema, 실패 처리)
  - 통계 검정 보완 (paired test, wins/ties/losses)
