# Coreset Selection 실험 계획

## 실험 개요

Spatio-temporal traffic forecasting에서 coreset selection 방법론 비교 실험.
ITSC 논문 → T-ITS 저널 확장을 위한 종합 실험 설계.

---

## Phase A: Adjacency Matrix (완료)

목적: Graph 기반 모델의 인접행렬 방법론 비교
- 모델: GWNet, MTGNN, STGCN
- 데이터셋: SAN_BERNARDINO, CONTRA_COSTA
- 인접행렬: distance, distance_sparse, distance_dense, identity
- Seed: 42, 123, 456

---

## Phase B: Proxy Metric vs MAE (완료 — SAN_BERNARDINO)

목적: Coreset selection method별 proxy metric과 실제 MAE의 상관관계 분석
- 모델: STGCN, AGCRN (완료), DCRNN/STID/STAEformer (일부)
- 데이터셋: SAN_BERNARDINO
- Distance: euclidean, temporal, spatial, combined (4종)
- Ratio: 0.3, 0.7
- Seed: 42, 123

**핵심 발견**:
- Distance type이 k-medoids 성능에 미치는 영향 미미
- → Phase C에서 distance를 `combined` 하나로 고정

---

## Phase C: 종합 실험 (진행 중)

### 설계 원칙
- **Distance 축 고정**: `combined` 1개 (Phase B에서 영향 미미 확인)
- **Method × Ratio 집중**: 방법론 비교 + k-medoids ratio 최적화
- **통계적 견고성**: Seed 3개 (42, 123, 456)
- **일반화 검증**: 데이터셋 2개 (SAN_BERNARDINO, CONTRA_COSTA)

### Group 1: Method Comparison
`experiments/config/phase_c_method_comparison.yaml`

| 항목 | 값 |
|---|---|
| 모델 | STGCN, AGCRN, DCRNN, STID, STAEformer (5) |
| 데이터셋 | SAN_BERNARDINO (893 nodes), CONTRA_COSTA (773 nodes) |
| Smart methods | k_medoids, k_center, graph_cut |
| Baseline methods | random, stride, recent |
| Distance | **combined** (고정) |
| Ratio | 0.3, 0.5, 0.7 |
| Seed | 42, 123, 456 |
| Full data | ratio=1.0 (별도) |

**Run count**: 270 (smart) + 270 (baseline) + 30 (full data) = **570**

### Group 2: k-medoids Ratio Sweep
`experiments/config/phase_c_ratio_sweep.yaml`

| 항목 | 값 |
|---|---|
| 모델 | 동일 5개 |
| 데이터셋 | 동일 2개 |
| Method | **k_medoids만** |
| Distance | **combined** (고정) |
| Ratio | 0.1, 0.2, 0.4, 0.6, 0.8, 0.9 (Group 1과 비겹침) |
| Seed | 42, 123, 456 |

**Run count**: **180**

### 전체 총계: **750 runs**

Group 1 + 2를 합치면 k-medoids는 ratio 0.1~0.9 + 1.0 = 10단계 전부 커버.

---

## 전체 실험 매트릭스 (Phase C)

```
모델 5 × 데이터셋 2 = 10 config

Group 1 (Method Comparison):
  Smart:    10 configs × 3 methods × 1 dist × 3 ratios × 3 seeds = 270
  Baseline: 10 configs × 3 methods × 1 dist × 3 ratios × 3 seeds = 270
  Full:     10 configs × 1 method  × 1 dist × 1 ratio  × 3 seeds =  30
  Subtotal: 570

Group 2 (Ratio Sweep):
  k-medoids: 10 configs × 1 method × 1 dist × 6 ratios × 3 seeds = 180
  Subtotal: 180

Grand Total: 750
```

---

## 향후 가능한 확장 (Optional)

필요 시 별도 config로 추가:
- **Distance 비교**: k-medoids × 4 dist × ratio=0.3 × 3 seeds × 10 configs = 120 runs
- **추가 데이터셋**: ALAMEDA, SACRAMENTO (datasets/xtraffic/ 에 존재)

---

## Config 파일 위치

### 모델 Config
| 모델 | SAN_BERNARDINO | CONTRA_COSTA |
|---|---|---|
| STGCN | `baselines/STGCN/SAN_BERNARDINO/SAN_BERNARDINO_no_dropout.py` | `baselines/STGCN/CONTRA_COSTA/CONTRA_COSTA.py` |
| AGCRN | `baselines/AGCRN/SAN_BERNARDINO/SAN_BERNARDINO.py` | `baselines/AGCRN/CONTRA_COSTA/CONTRA_COSTA.py` |
| DCRNN | `baselines/DCRNN/SAN_BERNARDINO/SAN_BERNARDINO.py` | `baselines/DCRNN/CONTRA_COSTA/CONTRA_COSTA.py` |
| STID | `baselines/STID/SAN_BERNARDINO/SAN_BERNARDINO.py` | `baselines/STID/CONTRA_COSTA/CONTRA_COSTA.py` |
| STAEformer | `baselines/STAEformer/SAN_BERNARDINO/SAN_BERNARDINO.py` | `baselines/STAEformer/CONTRA_COSTA/CONTRA_COSTA.py` |

### 실험 Config
- `experiments/config/phase_c_method_comparison.yaml` — Group 1
- `experiments/config/phase_c_ratio_sweep.yaml` — Group 2

### 실행 방법
```bash
# 1. Coreset index 생성
python experiments/select_coreset.py --cfg experiments/config/phase_c_method_comparison.yaml --gpus 0
python experiments/select_coreset.py --cfg experiments/config/phase_c_ratio_sweep.yaml --gpus 0

# 2. Training 제출
python experiments/run_experiments.py --cfg experiments/config/phase_c_method_comparison.yaml --arch=cuda
python experiments/run_experiments.py --cfg experiments/config/phase_c_ratio_sweep.yaml --arch=cuda

# 3. 결과 수집
python experiments/get_results.py --config experiments/config/phase_c_method_comparison.yaml --metrics MAE RMSE
```
