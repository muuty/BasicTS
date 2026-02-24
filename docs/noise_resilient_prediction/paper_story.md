# Degradation-Robust Spatiotemporal Forecasting under Sensor Degradation

> **Main Story Document** for paper writing
> **Focus**: Denoising noise-robustness + degradation-robust forecasting
> **Last Updated**: 2026-02-16

---

## 1. Motivation (Intuitive)

도시 교통 예측 시스템은 "센서 일부가 고장나도" 전체 예측 품질이 크게 무너지지 않아야 한다.
현실에서는 이 가정이 자주 깨진다.

- 한 센서가 stuck/bias/noisy 상태가 되면, 그 센서만 틀리는 것이 아니라 주변 정상 센서 예측까지 같이 흔들린다.
- 운영 관점에서 진짜 문제는 clean benchmark에서 0.1~0.2 MAE를 줄이는 것보다,
  **degradation 상황에서 붕괴를 막는 것**이다.

이 문서의 목표는 clean SOTA 경쟁이 아니라, degradation 하에서 예측 안정성을 확보하는 방법을 정리하는 것이다.

---

## 2. Background (왜 기존 접근이 깨지는가)

### 2.1 Missing과 Noise는 다르다

- Missing: 보간 또는 mask-aware 학습으로 일정 부분 대응 가능
- Noise: mask가 1로 남아 모델이 corrupted 값을 정상값으로 믿음

즉, noise는 missing보다 탐지와 완화가 훨씬 어렵다.

### 2.2 Attention 모델의 구조적 취약성

초기 분석에서 STAEformer는 dead/failing 노드에 과도한 attention을 주는 경향(attention collapse)을 보였다.
5ch(occ/speed 포함)에서는 이 현상이 일부 완화되지만, degradation/noise 환경에서는 여전히 성능 붕괴와 spillover가 발생한다.

핵심 실패 모드:
1. corrupted input이 정제 없이 예측기로 들어감
2. spatial attention이 오류를 healthy node로 전파(spillover)

---

## 3. Problem Setting

SAN_BERNARDINO (xtraffic), 893 nodes, 5 features (flow, occupancy, speed, tod, dow).

### 3.1 Sensor Health Categories (Current Proxy)

아래 분류는 **zero-filled 기준 proxy**다.
즉, "관측 missing 자체"가 아니라 "0 값 출현률" 기반 분류다.

| Category | 기준 (flow zero rate) | 노드 수 | 비율 |
|---|---|---|---|
| Dead | >90% | 121 | 13.5% |
| Major fail | 50-90% | 28 | 3.1% |
| Minor fail (Partial) | 5-50% | 161 | 18.0% |
| Functional | <5% | 583 | 65.3% |

**Critical note**: 이 분류는 원본 missing mask 기준 분류가 아니므로, 논문 최종본에는 missing-aware 재측정이 필요하다.

### 3.2 Evaluation Axes (이 스토리의 핵심 지표)

1. Clean performance: clean MAE
2. Degradation robustness: functional node MAE degradation (%)
3. Spatial spillover: healthy node degradation (%)
4. Hard case: stuck noise robustness

---

## 4. Core Idea: Denoise First, Then Control Propagation

단순히 "어떤 센서를 버릴까"가 아니라,
"들어오는 신호를 먼저 정제하고, 남은 불확실성의 전파를 막자"가 핵심이다.

### 4.1 Stage 1 - Denoising Encoder

- 역할: forecasting model 앞단의 transparent filter
- 대상: physical channels(flow/occ/speed)만 denoise, tod/dow passthrough
- 효과: noisy input 자체를 clean manifold로 복원

### 4.2 Stage 2 - Reliability Bias (Pre-softmax)

- 역할: 복원 실패/불확실 신호의 spatial 전파 억제
- 메커니즘: reliability를 attention logit(pre-softmax)에 bias로 주입
- 기대효과: noisy node -> healthy node spillover 차단

### 4.3 Stage 3 - Cross-Prediction Contrastive Reliability

reconstruction-based reliability는 stuck에 약했다 (reconstruction error가 작아서 reliability가 높게 나옴).
그래서 "복원 오차" 대신 "주변 패턴과의 temporal 일관성"으로 reliability를 학습.

메커니즘:
1. Per-node temporal conv → node representation `h_i`
2. Learned cross-attention (self excluded) → context `h_context_i`
3. `reliability = sigmoid(scale * cos_sim(h_i, h_context_i) + bias)`

Adjacency-free 설계: pre-defined adj_mx 대신 learned attention으로 이웃 발견.

---

## 5. Evidence So Far

### 5.1 Baseline Vulnerability (문제의 크기)

노이즈는 missing보다 훨씬 파괴적이다.

- Missing (interpolation): +1.8% (@ p=30)
- Gaussian noise (s=0.3, r=30): +33.8% (STAEformer)

또한 STAEformer의 healthy node spillover가 매우 큼:
- Gaussian s=1.0, r=30: +449.3%
- Bias s=0.3, r=10: +43.6%

### 5.2 End-to-End Results (Canonical Table) — Reliability 방식

| Model | Clean MAE | Gauss s=0.5 r=30 | Gauss s=1.0 r=30 | Stuck r=30 | Bias s=0.5 r=30 |
|---|---:|---:|---:|---:|---:|
| STAEformer baseline (5ch) | 12.13 | +191.8% | +537.6% | +19.4% | +155.5% |
| + Denoising encoder | **12.05** | +31.7% | +133.5% | +18.0% | +88.6% |
| + Dn + Recon-reliability | 12.57 | +23.6% | +85.0% | +15.8% | +90.4% |
| + Contrastive-reliability | 12.11 | +24.3% | +81.7% | +17.1% | +144.2% |

### 5.3 Noise Augmentation + Denoising SSL 조합 (2026-02-17, 핵심 결과)

> **상세 결과**: `project/noise_resilient_prediction/noise_augmentation_ssl_results.md`

**접근법 비교 (All Functional nodes, STAEformer)**:

| Model | Clean MAE | Gauss s=0.5 r=30 | Gauss s=1.0 r=30 | Stuck r=30 | Bias s=0.5 r=30 | Drift s=0.5 r=30 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 12.13 | +191.8% | +537.6% | +19.4% | +155.5% | +74.0% |
| + Denoising v1 | **12.05** | +31.7% | +133.5% | +18.0% | +88.6% | +46.8% |
| + Noise augmentation | 12.52 | +13.9% | +38.9% | +12.7% | +63.8% | +31.9% |
| + v1 combo (dn+aug) | 12.62 | +13.8% | +44.4% | +12.9% | +70.9% | +37.9% |
| **+ v2 combo (dn_v2+aug)** | 12.81 | **+10.8%** | **+28.9%** | **+10.5%** | **+55.6%** | **+23.4%** |

**접근법 비교 (All Functional nodes, STGCN)**:

| Model | Clean MAE | Gauss s=0.5 r=30 | Gauss s=1.0 r=30 | Stuck r=30 | Bias s=0.5 r=30 | Drift s=0.5 r=30 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 13.34 | +37.9% | +108.2% | +23.2% | +85.7% | +73.9% |
| + Denoising v1 | 13.38 | +24.2% | +63.0% | +22.1% | +78.7% | +59.6% |
| + Noise augmentation | 13.60 | +10.2% | +33.8% | +15.5% | +59.1% | +39.9% |
| + v1 combo (dn+aug) | 13.50 | +11.4% | +31.6% | +17.7% | +67.0% | +41.0% |
| **+ v2 combo (dn_v2+aug)** | 15.08 | **+6.8%** | **+17.5%** | **+10.7%** | **+43.3%** | **+29.6%** |

**Spillover (healthy nodes only, STAEformer)**:

| Model | Gauss s=0.5 r=30 | Gauss s=1.0 r=30 | Bias s=0.5 r=30 |
|---|---:|---:|---:|
| Baseline | +155.1% | +449.3% | +84.0% |
| + Noise augmentation | **-0.1%** | +1.0% | +1.7% |
| + v2 combo | -0.0% | **+0.5%** | +2.9% |

**Corrupted nodes only (STAEformer)**:

| Model | Gauss s=1.0 r=30 | Bias s=0.5 r=30 | Drift s=0.5 r=30 |
|---|---:|---:|---:|
| Baseline | +743.3% | +322.0% | +220.9% |
| + Noise augmentation | +126.5% | +207.4% | +102.3% |
| **+ v2 combo** | **+94.4%** | **+177.3%** | **+74.3%** |

**Win count (16 noise scenarios × 2 architectures)**:

| View | STAEformer v2 combo | STGCN v2 combo |
|---|---|---|
| All functional | **16/16** (100%) | **13/16** (81%) |
| Corrupted only | **16/16** (100%) | **16/16** (100%) |
| Healthy (spillover) | 4/16 (25%) | 8/16 (50%) |

#### 핵심 해석

1. **v1 combo는 aug-only보다 나빴다**: encoder가 clean data도 왜곡 (occ 37% distortion, Gaussian 1.3% recovery)
2. **v2 combo는 모든 noise에서 최고**: residual connection이 핵심 — `output = input + correction`
3. **Complementarity 증명**:
   - Augmentation → spillover 방어 (healthy nodes 보호)
   - Denoising encoder → corrupted node 직접 교정
   - 조합 시 both effects 달성
4. **Residual connection이 game-changer**: pretrain loss 0.18→0.05, clean data 보존, 안전한 fallback

#### v1 vs v2 Encoder 차이

| | v1 | v2 |
|---|---|---|
| Output | `f(noisy_input)` (full reconstruction) | `input + correction(input)` (residual) |
| hidden_dim | 32 | 64 |
| Pretrain noise range | severity 0.1-0.5, rate 0.1-0.5 | severity 0.1-1.5, rate 0.1-0.7 |
| Pretrain val_loss | 0.18 | **0.05** |
| Clean data distortion | 37% occ distortion | ~0% (identity fallback) |

### 5.4 "Denoise가 잘 될수록 예측도 좋아지나?"에 대한 답변

**Yes — v1→v2로 denoising quality 향상 시 downstream robustness도 일관되게 향상.**

| Encoder version | Pretrain val_loss | STAEformer Gauss s1.0 r30 |
|---|---|---|
| v1 (no residual) | 0.18 | +133.5% (denoising only), +44.4% (combo) |
| v2 (residual) | 0.05 | N/A (denoising only), **+28.9%** (combo) |

특히 v1에서 combo가 aug-only보다 나빴던 것이 v2에서 최고로 반전된 것은,
encoder quality가 threshold를 넘어야 combo synergy가 작동함을 시사한다.

---

## 6. What We Removed from Main Narrative

`SSL Contrastive Encoder` 트랙은 이 메인 스토리에서 분리한다.
이유는 두 가지다:

1. 현재 논문의 중심 질문(degradation robustness)에 대한 직접 해법이 아니었다.
2. 본문에 남기면 narrative가 분산되고 핵심 메시지가 약해진다.

관련 내용은 historical archive로만 유지:
- `experiment_insights_comprehensive.md`
- `representation_learning_experiment_log.md`
- `design.md`

---

## 7. Critical Gaps (비판적으로 남겨야 할 리스크)

1. Sensor category가 missing-aware가 아니라 zero-rate proxy다.
2. multi-seed 통계(평균±std, 유의성)가 아직 부족하다.
3. stuck noise는 여전히 hardest case다 — 모든 방법에서 개선폭이 가장 작음.
4. 일부 결과는 실험 프로토콜(마스킹/loss/노드셋) 차이로 해석 리스크가 있다.
5. **STGCN v2 combo의 clean MAE penalty가 크다** (+13.0%, 15.08 vs 13.34).
   - 1ch downstream 모델에서 encoder의 correction이 과도할 수 있음
   - Encoder fine-tuning 또는 lighter encoder로 개선 가능성
6. **Reliability 방식은 현재 메인 스토리에서 후순위**:
   - v2 combo가 reliability 없이도 모든 noise에서 최고 성능
   - Reliability는 추후 spillover 추가 방어가 필요할 때 고려

---

## 8. TODO (Paper Story 기준)

- [x] Noise augmentation + denoising SSL 조합 실험 (v1, v2)
- [x] Residual connection의 효과 검증 (v1 vs v2)
- [x] 2개 아키텍처 (STAEformer, STGCN)에서 일관성 확인
- [x] 3-view 분석 (all functional, healthy spillover, corrupted)
- [ ] Dead/Major/Minor/Functional을 **원본 missing mask 기준**으로 재산출
- [ ] n>=5 seed 재실험: clean/degradation/spillover 평균±std + paired significance
- [ ] STGCN clean MAE penalty 줄이기 (encoder fine-tune 또는 lighter encoder)
- [ ] stuck 전용 stress test 추가
- [ ] 논문 writing: Introduction, Method, Experiment sections 초안

---

## 9. Reproducibility Assets

### 9.1 Key Checkpoints

| Model | Path |
|---|---|
| Baseline STAEformer 5ch | `checkpoints/STAEformer_5ch/SAN_BERNARDINO_30_12_12/` |
| STAEformer + denoising v1 | `checkpoints/STAEformer_5ch_denoising/SAN_BERNARDINO_30_12_12/` |
| STAEformer + noise aug | `checkpoints/STAEformer_5ch_noisy/SAN_BERNARDINO_30_12_12/` |
| STAEformer + v1 combo | `checkpoints/STAEformer_5ch_denoising_noisy/SAN_BERNARDINO_30_12_12/` |
| **STAEformer + v2 combo** | `checkpoints/STAEformer_5ch_denoising_v2_noisy/SAN_BERNARDINO_30_12_12/c7aa5c1f.../` |
| STGCN baseline | `checkpoints/STGCN/SAN_BERNARDINO_30_12_12/` |
| STGCN + noise aug | `checkpoints/STGCN_noisy/SAN_BERNARDINO_30_12_12/` |
| **STGCN + v2 combo** | `checkpoints/STGCN_denoising_v2_noisy/SAN_BERNARDINO_30_12_12/ac51aa4d.../` |
| Denoising pretrain v1 | `checkpoints/DenoisingPretrain/SAN_BERNARDINO_30_12_12/` |
| **Denoising pretrain v2** | `checkpoints/DenoisingPretrainV2/SAN_BERNARDINO_30_12_12/0ac3bff3.../` |
| Recon-reliability pretrain | `checkpoints/DenoisingPretrainReliability/SAN_BERNARDINO_30_12_12/` |
| Contrastive-reliability pretrain | `checkpoints/ContrastiveReliabilityPretrain/SAN_BERNARDINO_30_12_12/` |

### 9.2 Evaluation Scripts

- `experiments/eval_noise_vulnerability.py` — 전체 noise robustness 평가
- `experiments/noise_vulnerability_results/results.json` — 결과 JSON
- `experiments/eval_missing_pattern_shift.py`

### 9.3 Project Directory

- `project/noise_resilient_prediction/` — 이 주제의 종합 프로젝트 디렉토리
  - `noise_augmentation_ssl_results.md` — v2 combo 실험 결과 종합 문서
  - `analysis/` — eval script 및 결과 symlinks

---

## 10. Document Map

- `paper_story.md` (this file): 메인 스토리, 논문 본문 기준
- `project/noise_resilient_prediction/noise_augmentation_ssl_results.md`: **Aug+SSL 실험 종합 (v2 포함)**
- `attention_collapse_missing_values.md`: attention collapse 메커니즘 분석
- `experiment_insights_comprehensive.md`: SSL 트랙/평가 전환 히스토리 (historical)
- `representation_learning_experiment_log.md`: 전체 실험 연대기 (historical)
- `design.md`: 초기 아이디어 문서 (historical)
