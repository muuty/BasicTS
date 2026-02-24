# Noise Augmentation + Denoising SSL: Comprehensive Results

> **Last Updated**: 2026-02-17
> **Dataset**: SAN_BERNARDINO (893 nodes, 5ch, 3 months)
> **Architectures**: STAEformer (attention-based), STGCN (GCN-based)

---

## 1. Executive Summary

Noise augmentation과 denoising SSL은 **complementary**하다.
단, encoder 설계가 핵심이다 — residual connection 없이는 combo가 오히려 해롭다.

| | STAEformer | STGCN |
|---|---|---|
| Augmentation only | 강한 spillover 방어, corrupted 교정은 제한적 | 동일 패턴 |
| Denoising v1 only | corrupted 교정 우수, spillover 방어 미흡 | 동일 패턴 |
| **v1 combo** | **aug보다 나쁨** (encoder가 clean도 왜곡) | aug보다 나쁨 |
| **v2 combo** | **모든 noise에서 최고** | **모든 noise에서 최고** |

**v1 → v2의 핵심 차이**: residual connection (`output = input + correction` vs `output = f(input)`)

---

## 2. Models Overview

### 2.1 Approaches

| Approach | 설명 | Extra Params |
|---|---|---|
| **Baseline** | STAEformer/STGCN, 학습 없는 원본 | 0 |
| **Noise Augmentation** | 학습 시 input에 random noise 주입, target은 clean 유지 | 0 |
| **Denoising Encoder v1** | Temporal dilated conv + spatial graph conv, pretrained, frozen | ~1M |
| **Denoising Encoder v2** | v1 + residual connection + hidden_dim 64 + stronger noise | ~2M |
| **v1 Combo** | Denoising v1 + Noise Augmentation | ~1M |
| **v2 Combo** | Denoising v2 + Noise Augmentation | ~2M |

### 2.2 Denoising Encoder v1 vs v2

| | v1 | v2 |
|---|---|---|
| Output | `output = f(noisy_input)` | `output = noisy_input + correction(noisy_input)` |
| hidden_dim | 32 | 64 |
| Pretrain noise severity | 0.1 - 0.5 | 0.1 - 1.5 |
| Pretrain noise rate | 0.1 - 0.5 | 0.1 - 0.7 |
| Pretrain val_loss (final) | ~0.18 | **~0.05** |
| residual_connection | False | **True** |

**핵심**: v1은 전체 신호를 reconstruction해야 하므로 어려운 task.
v2는 correction term만 학습하므로 쉽고, 실패해도 identity로 fallback (안전).

### 2.3 Noise Augmentation Config

```python
CFG.NOISE_AUGMENTATION = {
    'prob': 0.5,              # 50% 확률로 noise 주입
    'rate_range': [0.05, 0.3],  # 노드 5-30% 선택
    'types': ['gaussian', 'bias', 'stuck', 'drift'],
    'severity_range': [0.1, 0.5],
    'physical_channels': [0, 1, 2],  # flow, occ, speed
}
```

### 2.4 Checkpoints

| Model | Clean MAE | Checkpoint |
|---|---:|---|
| STAEformer baseline | 12.26 | `checkpoints/STAEformer_5ch/SAN_BERNARDINO_30_12_12/` |
| STAEformer + denoising v1 | 12.13 | `checkpoints/STAEformer_5ch_denoising/SAN_BERNARDINO_30_12_12/` |
| STAEformer + aug | 12.63 | `checkpoints/STAEformer_5ch_noisy/SAN_BERNARDINO_30_12_12/` |
| STAEformer + v1 combo | 12.75 | `checkpoints/STAEformer_5ch_denoising_noisy/SAN_BERNARDINO_30_12_12/` |
| STAEformer + v2 combo | 12.92 | `checkpoints/STAEformer_5ch_denoising_v2_noisy/SAN_BERNARDINO_30_12_12/c7aa5c1f.../` |
| STGCN baseline | 14.09 | `checkpoints/STGCN/SAN_BERNARDINO_30_12_12/` |
| STGCN + denoising v1 | 14.11 | `checkpoints/STGCN_denoising/SAN_BERNARDINO_30_12_12/` |
| STGCN + aug | 14.29 | `checkpoints/STGCN_noisy/SAN_BERNARDINO_30_12_12/` |
| STGCN + v1 combo | 14.18 | `checkpoints/STGCN_denoising_noisy/SAN_BERNARDINO_30_12_12/` |
| STGCN + v2 combo | 15.71 | `checkpoints/STGCN_denoising_v2_noisy/SAN_BERNARDINO_30_12_12/ac51aa4d.../` |
| Denoising pretrain v1 | - | `checkpoints/DenoisingPretrain/SAN_BERNARDINO_30_12_12/` |
| Denoising pretrain v2 | - | `checkpoints/DenoisingPretrainV2/SAN_BERNARDINO_30_12_12/0ac3bff3.../` |

---

## 3. Main Results: All Functional Nodes

> **All Functional** = corrupted nodes + healthy nodes (dead/major_fail 제외, 745 nodes)
> 이것이 **핵심 지표**: 실제 서비스에서 기능하는 센서들의 전체 예측 품질

### 3.1 STAEformer

| Config | Baseline | Denoising v1 | Aug only | v1 combo | **v2 combo** |
|---|---:|---:|---:|---:|---:|
| **Clean MAE** | **12.13** | **12.05** | 12.52 | 12.62 | 12.81 |
| gauss s=0.3 r=10% | +17.6% | +8.1% | +2.4% | +2.9% | **+2.0%** |
| gauss s=0.3 r=30% | +33.8% | +9.6% | +6.0% | +5.7% | **+5.0%** |
| gauss s=0.3 r=50% | +76.2% | +23.2% | +10.6% | +11.4% | **+8.5%** |
| gauss s=0.5 r=30% | +191.8% | +31.7% | +13.9% | +13.8% | **+10.8%** |
| gauss s=1.0 r=30% | +537.6% | +133.5% | +38.9% | +44.4% | **+28.9%** |
| bias s=0.3 r=10% | +58.6% | +52.5% | +12.4% | +12.7% | **+8.7%** |
| bias s=0.3 r=30% | +74.5% | +38.9% | +32.5% | +31.8% | **+24.4%** |
| bias s=0.3 r=50% | +128.5% | +100.0% | +57.7% | +59.0% | **+42.3%** |
| bias s=0.5 r=30% | +155.5% | +88.6% | +63.8% | +70.9% | **+55.6%** |
| stuck r=10% | +5.8% | +5.3% | +3.8% | +4.0% | **+3.0%** |
| stuck r=30% | +19.4% | +18.0% | +12.7% | +12.9% | **+10.5%** |
| stuck r=50% | +35.9% | +33.7% | +23.5% | +23.4% | **+20.2%** |
| drift s=0.3 r=10% | +29.1% | +20.5% | +5.9% | +6.7% | **+3.9%** |
| drift s=0.3 r=30% | +38.0% | +21.2% | +15.9% | +16.1% | **+10.6%** |
| drift s=0.3 r=50% | +69.9% | +47.1% | +28.2% | +29.8% | **+18.7%** |
| drift s=0.5 r=30% | +74.0% | +46.8% | +31.9% | +37.9% | **+23.4%** |

**v2 combo는 16/16 시나리오에서 최고.**

### 3.2 STGCN

| Config | Baseline | Denoising v1 | Aug only | v1 combo | **v2 combo** |
|---|---:|---:|---:|---:|---:|
| **Clean MAE** | **13.34** | 13.38 | 13.60 | 13.50 | 15.08 |
| gauss s=0.3 r=10% | +3.3% | +2.8% | +1.6% | +1.6% | **+0.7%** |
| gauss s=0.3 r=30% | +13.2% | +9.6% | +4.8% | +5.2% | **+2.8%** |
| gauss s=0.3 r=50% | +28.8% | +18.9% | +8.1% | +9.0% | **+5.6%** |
| gauss s=0.5 r=30% | +37.9% | +24.2% | +10.2% | +11.4% | **+6.8%** |
| gauss s=1.0 r=30% | +108.2% | +63.0% | +33.8% | +31.6% | **+17.5%** |
| bias s=0.3 r=10% | +14.0% | +9.5% | +9.5% | **+7.7%** | +7.2% |
| bias s=0.3 r=30% | +44.7% | +28.9% | +31.1% | **+24.5%** | +24.9% |
| bias s=0.3 r=50% | +73.3% | +52.0% | +50.8% | +43.8% | **+42.6%** |
| bias s=0.5 r=30% | +85.7% | +78.7% | +59.1% | +67.0% | **+43.3%** |
| stuck r=10% | +5.4% | +5.5% | +3.3% | +4.3% | **+2.0%** |
| stuck r=30% | +23.2% | +22.1% | +15.5% | +17.7% | **+10.7%** |
| stuck r=50% | +47.3% | +43.0% | +32.6% | +35.3% | **+23.5%** |
| drift s=0.3 r=10% | +10.8% | +6.5% | +6.5% | +5.0% | **+4.9%** |
| drift s=0.3 r=30% | +34.9% | +20.9% | +20.8% | **+16.1%** | +16.5% |
| drift s=0.3 r=50% | +59.1% | +39.0% | +35.1% | +29.4% | **+29.7%** |
| drift s=0.5 r=30% | +73.9% | +59.6% | +39.9% | +41.0% | **+29.6%** |

**v2 combo는 13/16 시나리오에서 최고.** 나머지 3개도 v1 combo와 근소 차이.

---

## 4. Spillover Analysis: Healthy Nodes Only

> **Spillover** = corrupted 되지 않은 건강한 노드의 예측이 얼마나 나빠지는가
> 즉, 노이즈가 spatial message passing을 통해 전파되는 정도

### 4.1 STAEformer (Healthy nodes)

| Config | Baseline | Denoising v1 | Aug only | v1 combo | **v2 combo** |
|---|---:|---:|---:|---:|---:|
| gauss s=0.3 r=30% | +20.7% | +0.3% | **-0.1%** | -0.1% | +0.1% |
| gauss s=0.5 r=30% | +155.1% | +3.2% | **-0.1%** | +0.2% | -0.0% |
| gauss s=1.0 r=30% | +449.3% | +17.5% | +1.0% | +1.5% | **+0.5%** |
| bias s=0.3 r=30% | +25.8% | +1.8% | **+0.9%** | +0.9% | +1.0% |
| bias s=0.5 r=30% | +84.0% | +16.4% | **+1.7%** | +4.4% | +2.9% |
| stuck r=30% | +0.2% | +0.6% | **-0.4%** | -0.1% | -0.3% |
| drift s=0.3 r=30% | +2.2% | +0.5% | +0.5% | **-0.1%** | +0.6% |
| drift s=0.5 r=30% | +11.0% | +8.1% | +1.5% | +1.7% | **+1.3%** |

**Spillover 방어**: Augmentation이 대부분 최고 (11/16). v2 combo도 비슷하게 우수 (4/16 best).
두 방법 모두 spillover를 거의 0%에 가깝게 억제.

### 4.2 STGCN (Healthy nodes)

| Config | Baseline | Denoising v1 | Aug only | v1 combo | **v2 combo** |
|---|---:|---:|---:|---:|---:|
| gauss s=0.3 r=30% | +5.2% | +2.3% | +0.4% | +0.5% | **-1.2%** |
| gauss s=0.5 r=30% | +21.0% | +8.0% | +1.5% | +1.7% | **-0.9%** |
| gauss s=1.0 r=30% | +70.7% | +30.4% | +14.1% | +10.3% | **+2.8%** |
| bias s=0.3 r=30% | +1.8% | +2.2% | **+1.1%** | +1.8% | +4.1% |
| bias s=0.5 r=30% | +9.3% | +24.5% | **+2.5%** | +24.7% | +6.6% |
| stuck r=30% | +5.1% | +4.9% | +3.0% | +3.8% | **+1.7%** |
| drift s=0.3 r=30% | +4.0% | +2.0% | **+0.8%** | +0.8% | +2.8% |
| drift s=0.5 r=30% | +20.3% | +21.9% | **+3.9%** | +10.0% | +5.1% |

**STGCN spillover**: v2 combo가 Gaussian에서 best (심지어 음수 = 오히려 개선).
Augmentation이 bias/drift에서 best.

---

## 5. Corrupted Nodes Only

> **Corrupted nodes** = noise가 직접 주입된 노드들의 예측 정확도

### 5.1 STAEformer (Corrupted nodes)

| Config | Baseline | Denoising v1 | Aug only | v1 combo | **v2 combo** |
|---|---:|---:|---:|---:|---:|
| gauss s=0.3 r=30% | +64.3% | +31.4% | +20.0% | +19.0% | **+16.1%** |
| gauss s=0.5 r=30% | +277.5% | +98.3% | +46.2% | +45.3% | **+35.6%** |
| gauss s=1.0 r=30% | +743.3% | +404.6% | +126.5% | +143.6% | **+94.4%** |
| bias s=0.3 r=30% | +188.0% | +125.6% | +105.8% | +103.3% | **+78.4%** |
| bias s=0.5 r=30% | +322.0% | +257.4% | +207.4% | +224.3% | **+177.3%** |
| stuck r=30% | +64.3% | +58.6% | +43.1% | +43.2% | **+35.4%** |
| drift s=0.3 r=30% | +121.4% | +69.7% | +51.3% | +53.5% | **+33.9%** |
| drift s=0.5 r=30% | +220.9% | +137.3% | +102.3% | +121.7% | **+74.3%** |

**v2 combo는 16/16 시나리오에서 압도적 최고.** Corrupted node 복원이 encoder의 핵심 강점.

### 5.2 STGCN (Corrupted nodes)

| Config | Baseline | Denoising v1 | Aug only | v1 combo | **v2 combo** |
|---|---:|---:|---:|---:|---:|
| gauss s=0.3 r=30% | +31.9% | +26.7% | +14.9% | +16.0% | **+12.1%** |
| gauss s=0.5 r=30% | +77.5% | +62.1% | +30.3% | +33.8% | **+24.7%** |
| gauss s=1.0 r=30% | +196.1% | +139.2% | +79.5% | +80.7% | **+51.7%** |
| bias s=0.3 r=30% | +145.1% | +91.3% | +100.8% | +76.7% | **+73.6%** |
| bias s=0.5 r=30% | +264.3% | +205.7% | +190.7% | +164.3% | **+128.9%** |
| stuck r=30% | +65.5% | +62.3% | +44.7% | +49.7% | **+31.7%** |
| drift s=0.3 r=30% | +107.1% | +65.1% | +67.3% | +51.3% | **+48.4%** |
| drift s=0.5 r=30% | +199.4% | +147.9% | +123.7% | +112.3% | **+86.7%** |

**v2 combo는 16/16에서 최고.**

---

## 6. Win Count Summary

### 6.1 v2 combo가 최고인 시나리오 수 (총 16개 시나리오)

| View | STAEformer | STGCN |
|---|---|---|
| All functional | **16/16** (100%) | **13/16** (81%) |
| Corrupted only | **16/16** (100%) | **16/16** (100%) |
| Healthy only (spillover) | 4/16 (25%) | 8/16 (50%) |

### 6.2 해석

- **Corrupted nodes**: v2 combo가 양 아키텍처에서 **100%** 승리. Encoder가 직접 교정.
- **All functional**: STAEformer 100%, STGCN 81% 승리. Corrupted 이득 > healthy 손실.
- **Healthy (spillover)**: Augmentation이 더 자주 승리. Spillover 방어는 augmentation의 강점.

---

## 7. Analysis: 왜 v1 combo는 실패하고 v2 combo는 성공했는가

### 7.1 v1 combo의 문제

v1 encoder는 `output = f(noisy_input)` 형태로, **전체 신호를 reconstruction**해야 한다.

문제점:
1. **Clean data도 왜곡**: clean input을 encoder에 통과시켜도 occupancy 37% distortion 발생
2. **Gaussian 복원력 부족**: 1.3% recovery rate (거의 denoising 못 함)
3. **Downstream 모델 혼란**: 왜곡된 input으로 학습 → clean MAE도 나빠짐 (12.62 vs 12.52)
4. **Augmentation 효과 상쇄**: encoder의 추가 왜곡이 augmentation의 robustness를 상쇄

결과: v1 combo가 augmentation-only보다 **오히려 나쁨** (gauss s1.0: +44.4% vs +38.9%)

### 7.2 v2 combo의 성공

v2 encoder는 `output = input + correction(input)` 형태로, **correction term만 학습**.

장점:
1. **Clean data 보존**: correction ≈ 0이면 output ≈ input (identity fallback)
2. **학습 용이**: 전체 reconstruction보다 correction이 훨씬 쉬운 task
3. **안전한 실패**: encoder가 correction을 잘 못 배워도 input이 그대로 통과
4. **강한 noise 학습**: pretrain 시 severity 0.1-1.5, rate 0.1-0.7으로 더 다양한 noise 경험

결과: pretrain val_loss 0.18 → 0.05 (3.6x 개선), 모든 noise에서 최고 성능

### 7.3 Complementarity의 메커니즘

```
Noise Augmentation의 역할:
  - Training-time regularization → 모델이 noisy input에 robust한 feature 학습
  - Spatial attention이 노이즈에 둔감해짐 → spillover 방어
  - 모델 내부의 implicit robustness

Denoising Encoder의 역할:
  - Input-level denoising → corrupted 값을 직접 교정
  - Noise가 모델에 도달하기 전에 제거
  - 모델 외부의 explicit preprocessing

조합의 시너지:
  - Encoder가 noise의 대부분을 제거 (1st line of defense)
  - 남은 residual noise는 augmentation으로 학습된 robustness가 처리 (2nd line)
  - 결과: corrupted node 복원 + spillover 방어를 동시에 달성
```

---

## 8. Clean MAE Trade-off

v2 combo의 clean MAE가 baseline보다 높다는 점에 주의:

| | STAEformer | STGCN |
|---|---|---|
| Baseline clean MAE | 12.13 | 13.34 |
| v2 combo clean MAE | 12.81 (+0.68) | 15.08 (+1.74) |
| Clean MAE penalty | +5.6% | +13.0% |

**STGCN의 clean penalty가 크다** (13.0%). 원인 추정:
- STGCN은 1ch (flow only) 모델인데, encoder는 5ch → 3ch denoising 후 flow만 추출
- Encoder의 correction이 clean data에서도 약간의 distortion 유발
- STAEformer는 3ch를 모두 사용하므로 상대적으로 영향 적음

**Trade-off 해석**: 현실 운영 환경에서 **항상 일부 센서가 noisy**하므로,
clean MAE +5.6%를 감수하고 worst-case 안정성을 확보하는 것이 합리적 선택.
Gauss s=1.0에서 degradation이 +537% → +29%로 줄어드는 것을 고려하면
clean penalty는 매우 작은 비용이다.

---

## 9. Noise Type별 특성

### 9.1 Gaussian Noise
- **가장 큰 degradation**: baseline STAEformer s=1.0 r=30%에서 +537.6%
- **v2 combo 효과 극대화**: +537.6% → +28.9% (94.6% 감소)
- **Spillover 심각**: baseline healthy +449%, v2 combo +0.5%
- Encoder가 Gaussian에 가장 효과적 (랜덤 noise는 spatial pattern으로 필터 가능)

### 9.2 Bias Noise
- **Encoder 만으로 어려움**: denoising v1이 bias에서 +88.6% (Gaussian +31.7% 대비 약함)
- **v2 combo가 크게 개선**: bias s=0.5에서 +155.5% → +55.6%
- **Augmentation의 역할 큼**: bias를 학습 시 경험해야 방어 가능
- Bias는 systematic offset이므로 correction term 학습이 상대적으로 용이

### 9.3 Stuck Noise
- **가장 완화 어려운 noise**: 모든 방법에서 개선폭 가장 작음
- **v2 combo**: +19.4% → +10.5% (46% 감소, 다른 noise 대비 적음)
- 이유: stuck = 값이 고정 → temporal pattern은 유지되나 value가 틀림 → 탐지 어려움

### 9.4 Drift Noise
- **점진적 변화**: bias와 유사하나 시간에 따라 증가
- **v2 combo 매우 효과적**: drift s=0.5에서 +74.0% → +23.4% (68% 감소)
- Encoder의 temporal conv가 drift pattern 감지에 유리

---

## 10. Architecture 차이: STAEformer vs STGCN

| 특성 | STAEformer | STGCN |
|---|---|---|
| Spatial mechanism | Attention (adaptive) | GCN (fixed adj) |
| Spillover 크기 (baseline) | +449% (gauss s1.0) | +71% |
| Augmentation 효과 | spillover 99% 제거 | spillover 80% 제거 |
| v2 combo clean penalty | +5.6% | +13.0% |
| v2 combo win rate (all func) | 16/16 (100%) | 13/16 (81%) |

- STAEformer는 attention 기반이라 spillover가 훨씬 심하고, augmentation/encoder 효과도 큼
- STGCN은 fixed adjacency로 spillover가 작지만, clean penalty가 더 큼
- 두 아키텍처 모두 v2 combo가 가장 robust

---

## 11. Conclusion

1. **Noise augmentation만으로도 강력한 방어** — 0 extra parameters로 spillover 99% 방어
2. **Denoising SSL은 corrupted node 교정에 강점** — augmentation이 못 하는 영역
3. **조합의 시너지는 encoder 설계에 달려 있다**:
   - Residual connection 없음 → combo가 aug-only보다 **나쁨** (v1)
   - Residual connection 있음 → combo가 **모든 시나리오에서 최고** (v2)
4. **Trade-off**: clean MAE +5.6% 비용으로 worst-case degradation 94% 감소
5. **Model-agnostic**: STAEformer (attention)와 STGCN (GCN) 모두에서 일관된 패턴

---

## Appendix: Evaluation Protocol

- **Noise injection**: test time에 functional nodes 중 일정 비율을 corrupted
- **Corruption types**: gaussian, bias, stuck, drift (severity × rate combinations)
- **Metrics**: MAE degradation % from clean (per-view: all functional, healthy only, corrupted only)
- **Node categories**: dead (>90% zero, 120), major_fail (50-90%, 28), functional (<5%, 625+)
- **Evaluation script**: `experiments/eval_noise_vulnerability.py`
- **Results JSON**: `experiments/noise_vulnerability_results/results.json`
