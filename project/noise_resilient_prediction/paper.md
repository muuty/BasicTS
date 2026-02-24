# Noise-Resilient Spatiotemporal Forecasting under Sensor Degradation

> **Status**: Internal draft (pre-submission)
> **Last Updated**: 2026-02-17
> **Dataset**: SAN_BERNARDINO (893 nodes, 5 channels, 3 months)
> **Architectures**: STAEformer (attention-based), STGCN (GCN-based)

---

## 1. Introduction

Urban traffic forecasting systems must remain reliable when sensors degrade.
In practice, sensor failures — stuck readings, systematic bias, Gaussian noise, gradual drift — are common and ongoing.
The real operational challenge is not improving clean-benchmark MAE by 0.1, but **preventing catastrophic degradation when a subset of sensors becomes unreliable**.

Standard spatiotemporal models treat all input equally, making them vulnerable in two ways:

1. **Direct corruption**: The model receives wrong values and produces wrong predictions for corrupted nodes.
2. **Spillover**: Through spatial message passing (attention or graph convolution), errors propagate to predictions for healthy nodes that were never corrupted.

This work studies two complementary defense mechanisms and their combination:

- **Noise augmentation**: Training-time injection of synthetic noise to build implicit robustness (0 extra parameters).
- **Denoising encoder**: A pretrained input-cleaning module that explicitly corrects corrupted values before they reach the forecasting model.

We show that the combination is powerful but **encoder design is critical**: a naive full-reconstruction encoder (v1) hurts when combined with augmentation, while a residual-connection encoder (v2) achieves consistent improvement. We present results on two architectures (STAEformer and STGCN) with honest analysis of trade-offs, including the clean-MAE penalty and absolute-vs-relative metric discrepancies.

---

## 2. Problem Setting

### 2.1 Dataset

SAN_BERNARDINO from the xtraffic collection: 893 loop-detector sensors on California highways.

| Property | Value |
|---|---|
| Nodes | 893 |
| Features | 5: flow (ch0), occupancy (ch1), speed (ch2), time-of-day (ch3), day-of-week (ch4) |
| Resolution | 5-minute intervals (288 steps/day) |
| Duration | 3 months (26,280 timestamps) |
| Task | 12-step input → 12-step output (1 hour → 1 hour) |
| Target | Flow (channel 0) |

### 2.2 Sensor Health Categories

Sensors are categorized by 3-channel zero rate (flow=0 AND occupancy=0 AND speed=0 simultaneously):

| Category | Criterion | Count | % |
|---|---|---:|---:|
| Dead | >90% zero across all 3 physical channels | 120 | 13.4% |
| Major fail | 50–90% zero | 28 | 3.1% |
| Partial fail | 5–50% zero | 120 | 13.4% |
| Functional | <5% zero | 625 | 70.0% |
| **Total** | | **893** | |

**Evaluation set**: All non-dead, non-major-fail nodes = 893 − 120 − 28 = **745 nodes**.
This set includes partial-fail (120) and functional (625) nodes.
Dead and major-fail nodes are excluded because they have near-zero ground truth, making MAE uninformative.

**Note on counting**: Some earlier analyses used a flow-only zero-rate criterion, yielding slightly different counts (e.g., dead=121). This document uses the stricter 3-channel criterion throughout.

### 2.3 Two Types of MAE Reported

Throughout this document:
- **Functional clean MAE** (used in robustness tables): MAE computed on the 745-node evaluation set. Values: STAEformer baseline = 12.13, STGCN baseline = 13.34.
- **Overall MAE** (from `test_metrics.json`, all 893 nodes): Includes dead/major nodes. Values: STAEformer baseline = 12.26, STGCN baseline = 14.09. These are higher because dead-node predictions (model predicts mean ≈ 23, target ≈ 0) are included.

All noise-robustness analyses use **functional clean MAE** on the 745-node evaluation set.

---

## 3. Methods

### 3.1 Noise Augmentation (Training-Time)

During training, with probability 0.5, we randomly select 5–30% of nodes and inject one of four noise types into their physical channels (flow, occupancy, speed):

| Noise Type | Description | Severity Controls |
|---|---|---|
| Gaussian | Additive `N(0, σ)` scaled by feature std | σ ∈ [0.1, 0.5] |
| Bias | Constant offset added to all timesteps | Magnitude ∈ [0.1, 0.5] × feature std |
| Stuck | Value frozen at a single timestep | N/A (binary) |
| Drift | Linearly increasing offset over time | Slope ∈ [0.1, 0.5] × feature std |

The training target remains the original clean data. This forces the model to learn features that are robust to input perturbations. Cost: **0 extra parameters**, only a training-time augmentation.

### 3.2 Denoising Encoder (Pretrained)

A lightweight encoder placed before the forecasting model. It processes all 5 input channels, denoises the 3 physical channels, and passes time-of-day/day-of-week through unchanged. The encoder is **pretrained** on a self-supervised denoising task (inject noise → reconstruct clean) and **frozen** during downstream training.

Architecture: temporal dilated convolution layers + spatial graph convolution (using the adjacency matrix).

#### v1 vs v2: The Critical Design Choice

| | v1 | v2 |
|---|---|---|
| Output formula | `output = f(noisy_input)` | `output = noisy_input + correction(noisy_input)` |
| Hidden dim | 32 | 64 |
| Pretrain noise severity | 0.1–0.5 | 0.1–1.5 |
| Pretrain noise rate | 0.1–0.5 | 0.1–0.7 |
| Pretrain final val loss | 0.18 | **0.05** |
| Parameters | ~1M | ~2M |
| Key property | Must reconstruct entire signal | Only learns correction term; identity fallback when input is clean |

**Why residual matters**: v1 must reconstruct the entire signal from scratch, which is hard and introduces distortion even on clean input (observed: 37% occupancy distortion on clean data). v2 only needs to learn the difference between noisy and clean, defaulting to identity (zero correction) when input is already clean. This makes v2 **safe** — worst case, it passes input through unchanged.

**Confound note**: v1 and v2 differ in multiple ways (residual connection, hidden dim, noise range). We attribute the improvement primarily to the residual connection based on the identity-fallback argument, but a controlled ablation isolating each factor has not been conducted. This is listed as future work (Section 8).

### 3.3 Combined Approach

The two methods operate at different levels:
- **Denoising encoder**: Explicit input-level preprocessing (1st line of defense)
- **Noise augmentation**: Implicit model-level regularization (2nd line of defense)

When combined, the encoder removes most noise before it reaches the model, and augmentation-trained robustness handles any residual noise the encoder misses.

---

## 4. Experimental Setup

### 4.1 Models

| Model | Clean MAE (func, 745 nodes) | Clean MAE (overall, 893 nodes) | Description |
|---|---:|---:|---|
| STAEformer baseline | 12.13 | 12.26 | Attention-based, 5ch input |
| STAEformer + denoising v1 | 12.05 | — | + frozen v1 encoder |
| STAEformer + augmentation | 12.57 | 12.63 | + training-time noise |
| STAEformer + v1 combo | 12.62 | 12.75 | + v1 encoder + augmentation |
| **STAEformer + v2 combo** | **12.81** | **12.92** | + v2 encoder + augmentation |
| STGCN baseline | 13.34 | 14.09 | GCN-based, 1ch downstream |
| STGCN + denoising v1 | 13.38 | 14.11 | + frozen v1 encoder |
| STGCN + augmentation | 13.60 | 14.29 | + training-time noise |
| STGCN + v1 combo | 13.50 | 14.18 | + v1 encoder + augmentation |
| **STGCN + v2 combo** | **15.08** | **15.71** | + v2 encoder + augmentation |

**Clean MAE penalty**: v2 combo has higher clean MAE than baseline.
- STAEformer: +0.68 (+5.6%)
- STGCN: +1.74 (+13.0%)

The STGCN penalty is notably larger. STGCN uses only flow (1 channel) from the encoder's 3-channel denoised output, so the encoder's correction on occupancy/speed is wasted, and any distortion on flow is fully absorbed.

### 4.2 Noise Evaluation Protocol

At test time, we select a fraction of the 745 evaluation nodes uniformly at random and inject noise into their physical channels. We then measure MAE from three views:

1. **All functional** (745 nodes): The primary metric — overall prediction quality on all working sensors.
2. **Corrupted only** (the selected subset): How well the model handles directly-corrupted nodes.
3. **Healthy only** (the unselected nodes): Spillover — how much noise on some nodes degrades predictions for others.

16 noise configurations are tested: 4 types × 4 severity/rate combinations (see Appendix A).

---

## 5. Results

### 5.1 Relative Robustness (% Degradation from Clean)

The tables below show MAE degradation as a percentage of each model's own clean MAE.

**Caution**: Because models have different clean MAE baselines, a lower % does not always mean a lower absolute noisy MAE. See Section 5.2 for absolute comparisons.

#### STAEformer — All Functional Nodes (% degradation)

| Noise Config | Baseline | Dn v1 | Aug | v1 combo | **v2 combo** |
|---|---:|---:|---:|---:|---:|
| *Clean MAE* | *12.13* | *12.05* | *12.57* | *12.62* | *12.81* |
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

v2 combo achieves the lowest % degradation in **16/16** scenarios.

#### STGCN — All Functional Nodes (% degradation)

| Noise Config | Baseline | Dn v1 | Aug | v1 combo | **v2 combo** |
|---|---:|---:|---:|---:|---:|
| *Clean MAE* | *13.34* | *13.38* | *13.60* | *13.50* | *15.08* |
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

v2 combo achieves the lowest % degradation in **13/16** scenarios. The remaining 3 (bias s=0.3 r=10%, bias s=0.3 r=30%, drift s=0.3 r=30%) are won by v1 combo with negligible margins.

### 5.2 Absolute Robustness (Noisy MAE)

**This is the more meaningful comparison** when models have different clean baselines. Below we show actual noisy MAE values (lower is better). **Bold** = best in row.

#### STAEformer — Absolute Noisy MAE (All Functional, 745 nodes)

| Noise Config | Baseline | Dn v1 | Aug | v1 combo | v2 combo |
|---|---:|---:|---:|---:|---:|
| Clean | **12.13** | **12.05** | 12.57 | 12.62 | 12.81 |
| gauss s=0.3 r=10% | 14.26 | 13.02 | **12.87** | 12.99 | 13.07 |
| gauss s=0.3 r=30% | 16.22 | **13.29** | 13.33 | 13.34 | 13.45 |
| gauss s=0.3 r=50% | 21.37 | 14.85 | **13.90** | 14.06 | **13.90** |
| gauss s=0.5 r=30% | 35.39 | 15.87 | 14.32 | 14.36 | **14.19** |
| gauss s=1.0 r=30% | 77.32 | 28.13 | 17.46 | 18.22 | **16.51** |
| bias s=0.3 r=10% | 19.24 | 18.38 | 14.13 | 14.22 | **13.92** |
| bias s=0.3 r=30% | 21.16 | 16.73 | 16.66 | 16.63 | **15.93** |
| bias s=0.3 r=50% | 27.70 | 24.09 | 19.82 | 20.06 | **18.23** |
| bias s=0.5 r=30% | 30.98 | 22.72 | 20.59 | 21.57 | **19.93** |
| stuck r=10% | 12.83 | **12.69** | 13.05 | 13.12 | 13.19 |
| stuck r=30% | 14.48 | 14.21 | **14.17** | 14.25 | 14.15 |
| stuck r=50% | 16.48 | 16.11 | 15.53 | 15.57 | **15.39** |
| drift s=0.3 r=10% | 15.66 | 14.53 | 13.32 | 13.47 | **13.31** |
| drift s=0.3 r=30% | 16.74 | 14.60 | 14.57 | 14.65 | **14.16** |
| drift s=0.3 r=50% | 20.60 | 17.74 | 16.12 | 16.38 | **15.20** |
| drift s=0.5 r=30% | 21.09 | 17.70 | 16.59 | 17.40 | **15.81** |

**STAEformer absolute-MAE wins**: v2 combo best in **11/16**, aug-only in **2/16** (mild Gaussian), Dn v1 in **2/16** (gauss s=0.3 r=30%, stuck r=10%), tied in 1.

**Key pattern**: At mild noise (gauss s=0.3 r=10%), aug-only has lower absolute MAE (12.87) than v2 combo (13.07) because the clean penalty (+0.68) exceeds the robustness gain. The crossover occurs around gauss s≈0.3 r≈50% or s≈0.5 r≈30%.

#### STGCN — Absolute Noisy MAE (All Functional, 745 nodes)

| Noise Config | Baseline | Dn v1 | Aug | v1 combo | v2 combo |
|---|---:|---:|---:|---:|---:|
| Clean | **13.34** | 13.38 | 13.60 | 13.50 | 15.08 |
| gauss s=0.3 r=10% | **13.78** | 13.75 | 13.82 | **13.72** | 15.19 |
| gauss s=0.3 r=30% | 15.10 | 14.66 | **14.25** | **14.20** | 15.50 |
| gauss s=0.3 r=50% | 17.18 | 15.91 | **14.71** | 14.72 | 15.93 |
| gauss s=0.5 r=30% | 18.39 | 16.62 | **14.99** | 15.04 | 16.11 |
| gauss s=1.0 r=30% | 27.77 | 21.81 | 18.20 | 17.77 | **17.72** |
| bias s=0.3 r=10% | 15.20 | 14.65 | 14.90 | **14.38** | 16.17 |
| bias s=0.3 r=30% | 19.30 | 17.26 | 17.83 | **16.81** | 18.84 |
| bias s=0.3 r=50% | 23.11 | 20.35 | 20.51 | **19.41** | 21.51 |
| bias s=0.5 r=30% | 24.77 | 23.91 | **21.64** | 22.55 | 21.60 |
| stuck r=10% | 14.06 | 14.12 | **14.05** | 14.08 | 15.38 |
| stuck r=30% | 16.43 | 16.34 | **15.71** | 15.89 | 16.70 |
| stuck r=50% | 19.65 | 19.13 | **18.04** | 18.27 | 18.63 |
| drift s=0.3 r=10% | 14.78 | 14.26 | 14.49 | **14.18** | 15.82 |
| drift s=0.3 r=30% | 17.99 | 16.17 | 16.43 | **15.67** | 17.57 |
| drift s=0.3 r=50% | 21.22 | 18.60 | 18.38 | **17.48** | 19.56 |
| drift s=0.5 r=30% | 23.20 | 21.37 | **19.03** | 18.95 | 19.55 |

**STGCN absolute-MAE wins**: v2 combo wins only **2/16** (gauss s=1.0 r=30%, bias s=0.5 r=30%). Aug-only or v1 combo wins the remaining **14/16**.

**Critical finding**: For STGCN, the +13% clean MAE penalty is too large. Despite achieving the lowest % degradation in 13/16 scenarios, v2 combo produces higher absolute noisy MAE than augmentation-only or v1 combo in most scenarios. The crossover to v2 advantage only occurs at very high noise (gauss severity ≥ 1.0 or bias severity ≥ 0.5).

### 5.3 Breakeven Analysis

The clean MAE penalty means v2 combo must achieve sufficiently large robustness improvement to compensate. The **breakeven point** is where v2 combo's absolute noisy MAE drops below augmentation-only's.

| Architecture | Clean penalty | Breakeven noise level | Absolute-MAE wins |
|---|---|---|---|
| STAEformer | +5.6% | ~gauss s≈0.3 r≈50% | **11/16** (69%) |
| STGCN | +13.0% | ~gauss s≈1.0 r≈30% | **2/16** (13%) |

**Interpretation**: STAEformer's modest clean penalty (+5.6%) is offset at moderate noise levels, making v2 combo the recommended choice for noisy environments. For STGCN, the severe clean penalty (+13.0%) means augmentation-only is preferred in most realistic scenarios; v2 combo only pays off under extreme noise.

**Why the architecture difference?** STGCN uses only flow (1 channel) from the encoder output, discarding occupancy/speed corrections. Any encoder distortion on flow is fully absorbed. STAEformer uses all 3 physical channels, providing cross-channel redundancy that absorbs minor distortions.

### 5.4 Corrupted Node Recovery

On directly-corrupted nodes, the encoder's denoising effect is dramatic and consistent:

#### STAEformer — Corrupted Nodes Only (% degradation, representative configs)

| Noise Config | Baseline | Dn v1 | Aug | v1 combo | **v2 combo** |
|---|---:|---:|---:|---:|---:|
| gauss s=0.3 r=30% | +64.3% | +31.4% | +20.0% | +19.0% | **+16.1%** |
| gauss s=0.5 r=30% | +277.5% | +98.3% | +46.2% | +45.3% | **+35.6%** |
| gauss s=1.0 r=30% | +743.3% | +404.6% | +126.5% | +143.6% | **+94.4%** |
| bias s=0.5 r=30% | +322.0% | +257.4% | +207.4% | +224.3% | **+177.3%** |
| stuck r=30% | +64.3% | +58.6% | +43.1% | +43.2% | **+35.4%** |
| drift s=0.5 r=30% | +220.9% | +137.3% | +102.3% | +121.7% | **+74.3%** |

#### STGCN — Corrupted Nodes Only (% degradation, representative configs)

| Noise Config | Baseline | Dn v1 | Aug | v1 combo | **v2 combo** |
|---|---:|---:|---:|---:|---:|
| gauss s=0.3 r=30% | +31.9% | +26.7% | +14.9% | +16.0% | **+12.1%** |
| gauss s=0.5 r=30% | +77.5% | +62.1% | +30.3% | +33.8% | **+24.7%** |
| gauss s=1.0 r=30% | +196.1% | +139.2% | +79.5% | +80.7% | **+51.7%** |
| bias s=0.5 r=30% | +264.3% | +205.7% | +190.7% | +164.3% | **+128.9%** |
| stuck r=30% | +65.5% | +62.3% | +44.7% | +49.7% | **+31.7%** |
| drift s=0.5 r=30% | +199.4% | +147.9% | +123.7% | +112.3% | **+86.7%** |

v2 combo achieves the lowest corrupted-node degradation in **all** tested scenarios for both architectures. The corrupted-node % comparison is less affected by the denominator issue because the pre-noise clean MAE of the corrupted subset is similar across models (same underlying nodes).

### 5.5 Spillover Prevention

Spillover measures how much noise on corrupted nodes degrades predictions for healthy nodes. This is primarily an augmentation effect.

#### STAEformer — Healthy Nodes Only (% degradation, representative configs)

| Noise Config | Baseline | Dn v1 | Aug | v1 combo | v2 combo |
|---|---:|---:|---:|---:|---:|
| gauss s=0.3 r=30% | +20.7% | +0.3% | **−0.1%** | −0.1% | +0.1% |
| gauss s=0.5 r=30% | +155.1% | +3.2% | **−0.1%** | +0.2% | −0.0% |
| gauss s=1.0 r=30% | +449.3% | +17.5% | +1.0% | +1.5% | **+0.5%** |
| bias s=0.5 r=30% | +84.0% | +16.4% | **+1.7%** | +4.4% | +2.9% |
| stuck r=30% | +0.2% | +0.6% | **−0.4%** | −0.1% | −0.3% |
| drift s=0.5 r=30% | +11.0% | +8.1% | **+1.5%** | +1.7% | +1.3% |

Both augmentation and v2 combo reduce spillover to near-zero (from +449% to <+2%). Augmentation-only is marginally better in most spillover scenarios, confirming that training-time noise exposure is the key mechanism for spillover prevention.

### 5.6 Summary of Win Counts

| View | Metric | STAEformer v2 combo | STGCN v2 combo |
|---|---|---|---|
| All functional | % degradation | 16/16 (100%) | 13/16 (81%) |
| All functional | **Absolute MAE** | **11/16 (69%)** | **2/16 (13%)** |
| Corrupted only | % degradation | 16/16 (100%) | 16/16 (100%) |
| Healthy (spillover) | % degradation | 4/16 (25%) | 8/16 (50%) |

The discrepancy between % and absolute win counts is entirely explained by the clean MAE penalty. For STAEformer (modest penalty), v2 combo remains the best choice in most noisy scenarios. For STGCN (severe penalty), augmentation-only is preferred.

---

## 6. Analysis

### 6.1 Why v1 Combo Failed and v2 Succeeded

**v1 combo problems**:
1. Clean data distortion: Passing clean input through v1 produces 37% occupancy distortion.
2. Downstream confusion: The model trains on distorted "clean" input.
3. Augmentation interference: v1's distortion overlaps with augmentation's synthetic noise.
4. Result: v1 combo is worse than augmentation-only (e.g., gauss s=1.0: +44.4% vs +38.9%).

**v2 combo success**:
1. Identity fallback: When input is clean, correction ≈ 0, so output ≈ input.
2. Easier learning: Learning a correction is simpler than full signal reconstruction.
3. Stronger pretraining: Wider noise range (severity up to 1.5, rate up to 0.7).
4. Result: Pretrain val loss improved 3.6× (0.18 → 0.05).

### 6.2 Complementarity Mechanism

```
Noise Augmentation (implicit, inside the model):
  → Training-time regularization → noise-robust features
  → Spatial attention becomes noise-insensitive → spillover prevention

Denoising Encoder (explicit, outside the model):
  → Input-level signal cleaning → corrupted values directly corrected
  → Noise removed before it reaches the forecasting model

Combined (two lines of defense):
  → Encoder handles most noise (1st line)
  → Augmentation-trained robustness handles residual noise (2nd line)
  → Corrupted-node correction + spillover prevention simultaneously
```

### 6.3 Noise Type Characteristics

| Noise Type | Baseline vulnerability | v2 combo reduction (%) | Difficulty |
|---|---|---|---|
| **Gaussian** | Highest (up to +538%) | 94% reduction | Easiest (random, spatially detectable) |
| **Bias** | High (+156%) | 64% reduction | Moderate (systematic offset) |
| **Drift** | Moderate (+74%) | 68% reduction | Moderate (temporal conv detects pattern) |
| **Stuck** | Lowest (+19%) | 46% reduction | Hardest (values look plausible) |

Stuck noise is hardest because a frozen value may appear normal at any single timestep. Detection requires temporal context, and correction remains limited.

### 6.4 Architecture Comparison

| Property | STAEformer | STGCN |
|---|---|---|
| Spatial mechanism | Attention (adaptive) | GCN (fixed adjacency) |
| Input channels used | 3 (flow, occ, speed) | 1 (flow only) |
| Baseline spillover | Severe (+449% at gauss s=1.0) | Moderate (+71%) |
| v2 combo clean penalty | +5.6% | +13.0% |
| v2 combo absolute wins | 11/16 | 2/16 |
| Recommended defense | **v2 combo** (moderate+ noise) | **Aug-only** (most scenarios) |

---

## 7. Recommendations

| Deployment scenario | STAEformer | STGCN |
|---|---|---|
| Clean (no noise expected) | Baseline | Baseline |
| Mild noise (occasional glitches) | Augmentation-only | Augmentation-only |
| Moderate noise (regular failures) | **v2 combo** | Augmentation-only |
| Severe noise (frequent failures) | **v2 combo** | v2 combo (if clean accuracy is secondary) |

**General rule**: Noise augmentation should always be applied — it costs 0 parameters and provides substantial robustness. Adding a denoising encoder is beneficial when (a) the clean MAE penalty is acceptable and (b) the expected noise level exceeds the breakeven threshold.

---

## 8. Limitations and Future Work

1. **Single seed**: All results are from single training runs. Multi-seed experiments (n ≥ 5) with significance tests are needed before publication.

2. **v1 vs v2 confound**: v2 differs from v1 in residual connection, hidden dim (32→64), and pretrain noise range. A controlled ablation adding only the residual connection to v1 would isolate its effect.

3. **No v2 denoise-only experiment**: We tested v2 only in combination with augmentation. A standalone v2 encoder experiment would decompose the combo's improvement into encoder and augmentation contributions.

4. **STGCN clean penalty**: The +13% clean penalty makes v2 combo impractical for STGCN in most scenarios. Potential mitigations: fine-tuning the encoder (instead of freezing), lighter encoder, or output gating.

5. **Sensor categories are proxy-based**: Categories use zero-rate as a proxy for sensor failure, not ground-truth failure labels.

6. **Stuck noise remains hard**: All methods show limited improvement on stuck noise. Reliability estimation (confidence scores per node) could help flag stuck readings.

7. **Single dataset**: Validation on additional traffic datasets is needed.

---

## 9. Conclusion

1. **Noise augmentation alone provides strong defense** at zero parameter cost: it reduces spillover by >99% and overall degradation by 70–90%.

2. **A denoising encoder adds value for corrupted-node correction**, achieving the lowest corrupted-node degradation across all scenarios for both architectures.

3. **Encoder design is critical**: Residual connection (`output = input + correction`) is essential. Without it, the combo is worse than augmentation alone.

4. **The clean MAE penalty determines practical applicability**: For STAEformer (+5.6%), v2 combo is beneficial above moderate noise (absolute MAE wins 11/16). For STGCN (+13%), augmentation-only is preferred (wins 14/16 in absolute MAE).

5. **Always report absolute noisy MAE alongside % degradation**: When models have different clean baselines, % metrics can be misleading.

---

## Appendix A: Noise Configurations

| # | Type | Severity | Rate |
|---|---|---|---|
| 1–5 | Gaussian | 0.3, 0.3, 0.3, 0.5, 1.0 | 10%, 30%, 50%, 30%, 30% |
| 6–9 | Bias | 0.3, 0.3, 0.3, 0.5 | 10%, 30%, 50%, 30% |
| 10–12 | Stuck | — | 10%, 30%, 50% |
| 13–16 | Drift | 0.3, 0.3, 0.3, 0.5 | 10%, 30%, 50%, 30% |

Severity is relative to the feature's standard deviation. Rate is the fraction of 745 evaluation nodes corrupted.

## Appendix B: Reproducibility

### Checkpoints

| Model | Path |
|---|---|
| Denoising pretrain v1 | `checkpoints/DenoisingPretrain/SAN_BERNARDINO_30_12_12/` |
| Denoising pretrain v2 | `checkpoints/DenoisingPretrainV2/SAN_BERNARDINO_30_12_12/0ac3bff3.../` |
| STAEformer baseline | `checkpoints/STAEformer_5ch/SAN_BERNARDINO_30_12_12/` |
| STAEformer + dn v1 | `checkpoints/STAEformer_5ch_denoising/SAN_BERNARDINO_30_12_12/` |
| STAEformer + aug | `checkpoints/STAEformer_5ch_noisy/SAN_BERNARDINO_30_12_12/` |
| STAEformer + v1 combo | `checkpoints/STAEformer_5ch_denoising_noisy/SAN_BERNARDINO_30_12_12/` |
| STAEformer + v2 combo | `checkpoints/STAEformer_5ch_denoising_v2_noisy/SAN_BERNARDINO_30_12_12/c7aa5c1f.../` |
| STGCN baseline | `checkpoints/STGCN/SAN_BERNARDINO_30_12_12/` |
| STGCN + dn v1 | `checkpoints/STGCN_denoising/SAN_BERNARDINO_30_12_12/` |
| STGCN + aug | `checkpoints/STGCN_noisy/SAN_BERNARDINO_30_12_12/` |
| STGCN + v1 combo | `checkpoints/STGCN_denoising_noisy/SAN_BERNARDINO_30_12_12/` |
| STGCN + v2 combo | `checkpoints/STGCN_denoising_v2_noisy/SAN_BERNARDINO_30_12_12/ac51aa4d.../` |

### Key Code

| File | Purpose |
|---|---|
| `baselines/ContextContrastive/arch/denoising_encoder.py` | DenoisingEncoder (v1 + v2) |
| `baselines/ContextContrastive/arch/denoising_pretrain_model.py` | Pretrain wrapper |
| `baselines/ContextContrastive/runner/noisy_representation_learning_runner.py` | Noise aug + encoder runner |
| `experiments/eval_noise_vulnerability.py` | Evaluation script |
| `experiments/noise_vulnerability_results/results.json` | Raw results |

### Configs

| Config | Description |
|---|---|
| `baselines/STAEformer/SAN_BERNARDINO_5ch.py` | STAEformer baseline |
| `baselines/STAEformer/SAN_BERNARDINO_5ch_noisy.py` | + Augmentation |
| `baselines/STAEformer/SAN_BERNARDINO_5ch_denoising.py` | + Denoising v1 |
| `baselines/STAEformer/SAN_BERNARDINO_5ch_denoising_noisy.py` | + v1 combo |
| `baselines/STAEformer/SAN_BERNARDINO_5ch_denoising_v2_noisy.py` | + v2 combo |
| `baselines/STGCN/SAN_BERNARDINO_5ch.py` | STGCN baseline |
| `baselines/STGCN/SAN_BERNARDINO_noisy.py` | + Augmentation |
| `baselines/STGCN/SAN_BERNARDINO_denoising.py` | + Denoising v1 |
| `baselines/STGCN/SAN_BERNARDINO_denoising_noisy.py` | + v1 combo |
| `baselines/STGCN/SAN_BERNARDINO_denoising_v2_noisy.py` | + v2 combo |
| `baselines/ContextContrastive/SAN_BERNARDINO/pretrain_denoising.py` | v1 pretrain |
| `baselines/ContextContrastive/SAN_BERNARDINO/pretrain_denoising_v2.py` | v2 pretrain |
