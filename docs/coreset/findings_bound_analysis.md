# Coreset Selection: 이론적 Bound와 실험적 검증

**최종 수정**: 2026-03-01
**이론 증명**: [`theoritical_bound.latex`](theoritical_bound.latex)
**데이터셋**: SAN_BERNARDINO (893 nodes, 3개월)

---

## 목차

1. [이론적 Bound 요약](#1-이론적-bound-요약)
2. [k-medoids 연결](#2-k-medoids-연결)
3. [실험 설정](#3-실험-설정)
4. [Finding 1: Quantization Cost — 가장 강력한 Proxy](#4-finding-1-quantization-cost--가장-강력한-proxy)
5. [Finding 2: Coreset Advantage](#5-finding-2-coreset-advantage)
6. [Finding 3: Bound Verification Table](#6-finding-3-bound-verification-table)
7. [Traffic Forecasting에서의 고유한 이점](#7-traffic-forecasting에서의-고유한-이점)
8. [논문 서사 구조](#8-논문-서사-구조)
9. [데이터 파일 및 재현 코드](#9-데이터-파일-및-재현-코드)

---

## 1. 이론적 Bound 요약

전체 증명은 `theoritical_bound.latex` 참조. 여기서는 핵심만 요약한다.

### 1.1 Goal

Full-data 모델 $f_{\text{train}}$과 coreset 모델 $f_{\text{core}}$의 test MAE gap:

$$
G_{\text{test}} := \left| \text{MAE}_{D_{\text{test}}}(f_{\text{train}}) - \text{MAE}_{D_{\text{test}}}(f_{\text{core}}) \right|
$$

### 1.2 Final Bound

삼각부등식 + Kantorovich-Rubinstein(KR) 부등식 + weighted MAE를 거쳐:

$$
\boxed{
G_{\text{test}} \le (L_{\text{train}} + L_{\text{core}}) \left[ W_1(X_{\text{test}}, X_{\text{train}}) + W_1(X_{\text{train}}, C; \mathbf{w}) \right] + \varepsilon_{\text{disc}}
}
$$

### 1.3 각 항의 의미

| 항 | 의미 | Coreset으로 제어? |
|---|---|---|
| $W_1(X_{\text{test}}, X_{\text{train}})$ | test-train 분포 거리 (고정) | 불가 |
| $W_1(X_{\text{train}}, C; \mathbf{w})$ | train-coreset mismatch | **직접 제어 가능** |
| $L_{\text{train}}, L_{\text{core}}$ | 각 모델의 pointwise loss $\phi_f(x) = \|f(x)-y(x)\|$의 Lipschitz 상수 | 모델에 의존 |
| $\varepsilon_{\text{disc}}$ | $\|\text{MAE}^w_C(f_{\text{train}}) - \text{MAE}^w_C(f_{\text{core}})\|$ — 모델 불일치 | 간접적 |

### 1.4 Weighted MAE와 Weighted W₁

Cluster-size weighted MAE:

$$
\text{MAE}^w_C(f) := \sum_{\ell=1}^{k} w_\ell \|f(c_\ell) - y(c_\ell)\|, \quad w_\ell = n_\ell / n
$$

여기서 $n_\ell$은 medoid $c_\ell$에 배정된 training point 수. 이 weighting이 k-medoids의 nearest-assignment와 정확히 대응하여, W₁과 J_kmed의 연결을 exact하게 만든다.

---

## 2. k-medoids 연결

### 2.1 핵심 부등식

k-medoids 목적함수:

$$
J_{\text{kmed}}(C) := \frac{1}{n} \sum_{i=1}^{n} \min_{c \in C} d(x_i, c)
$$

**Claim**: $W_1(X_{\text{train}}, C; \mathbf{w}) \le J_{\text{kmed}}(C)$

**Proof**: Nearest-assignment transport plan $\Gamma_{i\ell} = \frac{1}{n} \cdot \mathbf{1}[\pi(x_i) = c_\ell]$이 $W_1(\cdot; \mathbf{w})$에 대해 feasible하고, 그 cost가 정확히 $J_{\text{kmed}}$. W₁은 모든 feasible plan 중 minimum이므로 $W_1 \le J_{\text{kmed}}$.

### 2.2 설계 원리

- k-medoids는 $J_{\text{kmed}}$을 최소화 → bound의 제어 가능한 항 $W_1(X_{\text{train}}, C; \mathbf{w})$의 upper bound를 직접 줄임
- k-center, graph_cut은 다른 목적함수를 최소화 → W₁ 최소화 보장 없음

---

## 3. 실험 설정

| 항목 | 값 |
|---|---|
| 데이터셋 | SAN_BERNARDINO (893 nodes, 100% coverage, 12-step in/out) |
| 모델 (Phase B) | STGCN, AGCRN |
| Coreset methods | k_medoids, k_center, graph_cut |
| Distance types | euclidean, temporal, spatial, combined |
| Ratios | 0.3, 0.7 |
| Seeds | 42, 123 |
| 총 checkpoints | 96 (3 methods × 4 distances × 2 ratios × 2 seeds × 2 models) |
| Feature 공간 | PCA-10 (82.2% explained variance) |
| Train set | 14,493 samples |

---

## 4. Finding 1: Quantization Cost — 가장 강력한 Proxy

### 4.1 Quantization Cost = $J_{\text{kmed}}$

Quantization cost는 각 train sample이 가장 가까운 coreset sample까지의 평균 거리:

$$
\text{Quant}(S) = \frac{1}{n} \sum_{i=1}^{n} \min_{s \in S} d_{\text{PCA}}(x_i, s)
$$

### 4.2 Method별 Quantization Cost vs MAE

**Ratio = 0.3**:

| Method | Quant Cost (PCA) | MAE (STGCN) | MAE (AGCRN) |
|---|---|---|---|
| **k_medoids** | **3.06 ~ 3.09** | **14.08** | **13.40** |
| k_center | 2.27 ~ 4.68 | 14.12 | 13.60 |
| graph_cut | **18.01 ~ 22.04** | **16.83** | **15.53** |

**Ratio = 0.7**:

| Method | Quant Cost (PCA) | MAE (STGCN) | MAE (AGCRN) |
|---|---|---|---|
| **k_medoids** | **0.92 ~ 0.93** | **13.35** | **13.41** |
| k_center | 0.58 ~ 1.11 | 13.49 | 13.54 |
| graph_cut | 2.87 ~ 3.33 | 13.58 | 13.45 |

graph_cut의 quant cost는 ratio=0.3에서 k_medoids 대비 **약 6~7배** → MAE +2.5 (STGCN) 증가.

### 4.3 상관 분석

**Within-setting Pearson r (ratio=0.3, model+distance 고정)**:

| Distance | STGCN r | AGCRN r |
|---|---|---|
| euclidean | **+0.996** | +0.943 |
| temporal | **+0.995** | **+0.992** |
| spatial | **+0.996** | **+0.991** |
| combined | **+0.998** | **+0.997** |

**모든 within-setting에서 r > 0.94** — quantization cost가 MAE의 거의 완벽한 predictor.

**다른 proxy metric과 비교**:

| Metric | Global r (STGCN) | 의미 |
|---|---|---|
| **quant_pca** | **+0.983** | W₁ proxy (최강) |
| sinkhorn_pca | -0.855 | Sinkhorn divergence |
| voronoi_cv | +0.755 | 밀도 균일성 |
| ess_ratio | -0.733 | Effective sample size |

---

## 5. Finding 2: Coreset Advantage

### 5.1 정의

$$
\Delta W_1 := SW_1(P_{\text{test}}, P_{\text{train}}) - SW_1(P_{\text{test}}, P_{\text{core}})
$$

양수 = coreset이 test에 더 가까움 = coreset 모델이 full-data 모델보다 좋을 수 있음.

### 5.2 Sliced Wasserstein Distance 실측

Baseline: $SW_1(P_{\text{test}}, P_{\text{train}}) = 3.0354$

| Method | Ratio | $SW_1$(test, core) | $SW_1$(core, train) | $\Delta W_1$ |
|---|---|---|---|---|
| **k_medoids** | 0.3 | **2.89 ~ 3.14** | **0.38 ~ 0.52** | **-0.10 ~ +0.15** |
| **k_medoids** | 0.7 | **2.99 ~ 3.09** | **0.17 ~ 0.24** | **-0.05 ~ +0.04** |
| graph_cut | 0.3 | 7.6 ~ 9.7 | 8.0 ~ 10.4 | -4.6 ~ -6.7 |
| graph_cut | 0.7 | 4.3 ~ 4.8 | 2.3 ~ 2.7 | -1.2 ~ -1.7 |
| k_center | 0.3 | 7.6 ~ 22.9 | 7.0 ~ 22.5 | -4.6 ~ -19.9 |
| k_center | 0.7 | 7.9 ~ 12.7 | 7.3 ~ 12.2 | -4.9 ~ -9.7 |

### 5.3 핵심 관찰

1. **k_medoids만이 $\Delta W_1 > 0$ 달성** (96개 중 8개, 전부 k_medoids)
2. **k_medoids의 $SW_1$(core, train)이 극도로 작음** (0.17~0.52 vs baseline 3.04의 6~17%)
3. graph_cut/k_center는 train에서도 test에서도 먼 분포 생성

### 5.4 메커니즘

```
k_medoids: J_kmed 최소화 → W₁(train, core) 작음
    → 삼각부등식 → W₁(test, core) ≈ W₁(test, train)
    → 미세한 차이로 ΔW₁ > 0 가능 → coreset advantage

graph_cut/k_center: W₁(train, core) 큼
    → W₁(test, core) >> W₁(test, train) → ΔW₁ << 0
```

### 5.5 $\Delta W_1 > 0$ 상세 데이터

| Distance | Ratio | Seed | $SW_1$(test,core) | $\Delta W_1$ | MAE (STGCN) | MAE (AGCRN) |
|---|---|---|---|---|---|---|
| spatial | 0.3 | 123 | 2.8905 | **+0.145** | 14.26 | 13.39 |
| combined | 0.3 | 123 | 2.8886 | **+0.147** | 14.13 | 13.41 |
| temporal | 0.3 | 123 | 2.8926 | **+0.143** | 14.07 | 13.40 |
| euclidean | 0.3 | 123 | 2.8977 | **+0.138** | 14.18 | 13.37 |
| spatial | 0.7 | 42 | 3.0010 | **+0.034** | 13.33 | 13.43 |
| combined | 0.7 | 42 | 3.0015 | **+0.034** | 13.33 | 13.45 |
| euclidean | 0.7 | 42 | 3.0028 | **+0.033** | 13.36 | 13.44 |
| temporal | 0.7 | 42 | 2.9981 | **+0.037** | 13.36 | 13.34 |

---

## 6. Finding 3: Bound Verification Table

> **Status**: 스크립트 작성 완료, GPU 실행 대기중

### 6.1 목적

이론적 bound의 각 항을 실측하여 bound가 tight한지 검증:

$$
G_{\text{test}} \le \underbrace{(L_{\text{train}} + L_{\text{core}})}_{\text{sensitivity}} \cdot \underbrace{\left[ W_1(X_{\text{test}}, X_{\text{train}}) + W_1(X_{\text{train}}, C; \mathbf{w}) \right]}_{\text{distributional shifts}} + \underbrace{\varepsilon_{\text{disc}}}_{\text{model discrepancy}}
$$

### 6.2 측정 항목

| 항 | 측정 방법 | 계산 위치 |
|---|---|---|
| $G_{\text{test}}$ | 실제 MAE gap (full-data vs coreset 모델) | GPU |
| $L_{\text{train}}, L_{\text{core}}$ | $\|\phi_f(x_i) - \phi_f(x_j)\| / d_{\text{PCA}}(x_i, x_j)$ — 2000 random pairs의 95th percentile / max | GPU |
| $\varepsilon_{\text{disc}}$ | $\|\text{MAE}^w_C(f_{\text{train}}) - \text{MAE}^w_C(f_{\text{core}})\|$ | GPU |
| $W_1(X_{\text{test}}, X_{\text{train}})$ | NN-based 근사, PCA-10 | CPU (선계산: ~13.02) |
| $W_1(X_{\text{train}}, C; \mathbf{w})$ | Quantization cost (= J_kmed) | CPU (선계산) |
| Bound / $G_{\text{test}}$ | Bound tightness ratio | - |

### 6.3 초기 관측

STGCN full-data 모델 기준:
- $L_{\text{train}}$: max=0.003773, 95th=0.000828, median=0.000366
- **L이 매우 작음** → bound가 tight할 가능성 높음
- $L \approx 0.004$, $W_1 \approx 13$ → $L \cdot W_1 \approx 0.05$ (MAE gap 대비 의미있는 수준)

### 6.4 Full-data Model Checkpoints

| 모델 | MAE_test | Checkpoint |
|---|---|---|
| STGCN | 13.409 | `checkpoints/phase_b_full_data/STGCNChebGraphConv/.../best_val_MAE.pt` |
| AGCRN | 13.348 | `checkpoints/phase_b_full_data/AGCRN/.../best_val_MAE.pt` |

### 6.5 결과 (TODO)

GPU 실행 후 `experiments/result/analysis/bound_verification.csv`에 저장 예정.

---

## 7. Traffic Forecasting에서의 고유한 이점

이 bound가 traffic forecasting에서 특히 tight한 이유:

| 성질 | Traffic Forecasting | Classification |
|---|---|---|
| Loss function | MAE — Lipschitz continuous | 0-1 loss — not Lipschitz |
| $Y\|X$ | Near-deterministic → $\Delta_{\text{cond}} \approx 0$ | Stochastic (boundary 근처) |
| 성능 결정 영역 | 전체 분포 (밀도 비례) | Decision boundary 근처 |
| 최적 coreset 전략 | 밀도 비례 (k-medoids) | Boundary-aware |
| KR duality | Tight (L 작음) | Loose 또는 불가 |

**핵심**: Traffic data의 near-deterministic $Y|X$ + MAE의 Lipschitz 구조 + temporal distribution shift → k-medoids의 W₁ 최소화가 이론적으로 최적에 가까운 전략.

---

## 8. 논문 서사 구조

1. **관찰**: k-medoids가 모든 setting에서 최저 MAE
2. **Bound 유도**: $G_{\text{test}} \le (L_{\text{train}} + L_{\text{core}}) [W_1(\text{test,train}) + W_1(\text{train,core}; \mathbf{w})] + \varepsilon_{\text{disc}}$
3. **핵심 항 식별**: 제어 가능한 유일한 항 = $W_1(X_{\text{train}}, C; \mathbf{w})$
4. **k-medoids 연결**: $J_{\text{kmed}} \ge W_1(\cdot; \mathbf{w})$ → k-medoids가 이 항을 직접 줄임
5. **실험 검증**: quant cost vs MAE: r > 0.94 (within-setting)
6. **Coreset advantage**: k-medoids만 $\Delta W_1 > 0$ 달성
7. **Bound verification**: L 실측, bound tightness 확인 (TODO)

---

## 9. 데이터 파일 및 재현 코드

### 결과 데이터

| 파일 | 설명 |
|---|---|
| `experiments/result/analysis/bound_terms_cpu.csv` | CPU bound terms (96 rows) |
| `experiments/result/analysis/coreset_advantage.csv` | SW₁ coreset advantage (96 rows) |
| `experiments/result/analysis/bound_verification.csv` | Bound verification (TODO) |
| `experiments/result/analysis/coreset_advantage.png` | 2D trade-off plot |
| `experiments/result/analysis/quantization_vs_mae.png` | Quant cost vs MAE scatter |

### 분석 스크립트

| 파일 | 설명 |
|---|---|
| `scripts/analysis/compute_bound_terms_cpu.py` | CPU terms (quant cost, W₁(test,train)) |
| `scripts/analysis/coreset_advantage_analysis.py` | SW₁ coreset advantage |
| `scripts/analysis/compute_bound_verification.py` | GPU bound verification (L, ε_disc, G_test) |

### 재현

```bash
# CPU terms
conda activate cuda && python scripts/analysis/compute_bound_terms_cpu.py

# Coreset advantage (SW₁)
conda activate cuda && python scripts/analysis/coreset_advantage_analysis.py

# Bound verification (GPU 필요)
conda activate cuda && python scripts/analysis/compute_bound_verification.py
```

---

## Archived Documents

이전에 별도 파일로 존재하던 내용이 이 문서에 통합됨:
- `theory_bound_derivation.md` → Section 1-2 (이론 요약) + `theoritical_bound.latex` (전체 증명)
- `bound_empirical_analysis.md` → Section 4-6 (실험 결과)
- `experiment_plan.md` → Section 3 (실험 설정) + `experiments/config/` (YAML configs)
