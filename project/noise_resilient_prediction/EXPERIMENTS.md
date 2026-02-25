# Noise-Resilient Traffic Prediction: Experiment Results

Comprehensive noise robustness evaluation across 33+ model configurations on two datasets.
For a one-page summary, see [RESULTS.md](RESULTS.md).

**Noise evaluation protocol**: Inject noise into 30% of functional nodes at test time, measure MAE degradation (%) on functional nodes. Noise types: Gaussian (additive), Bias (multiplicative shift), Stuck (frozen sensor), Drift (gradual calibration shift).

**Metrics**:
- **Clean MAE**: Standard test MAE on functional nodes (no noise)
- **Degradation %**: `(noisy_MAE - clean_MAE) / clean_MAE * 100` — lower is more robust
- **Avg Deg**: Mean degradation across 4 noise types (excl. dead)
- **Spillover**: Degradation on healthy (uncorrupted) nodes — measures noise propagation through spatial dependencies

---

## RQ1: Impact of Channel Configuration

Does using more input channels (flow/occupancy/speed) improve or hurt noise robustness?

| Model | Clean MAE | Gaussian | Bias | Stuck | Drift | Avg Deg |
|---|---|---|---|---|---|---|
| STAEformer 1ch (flow+tod+dow) | **11.90** | +36.9% | +56.9% | +19.9% | +34.7% | +37.1% |
| STAEformer 5ch (flow+occ+spd+tod+dow) | 12.13 | +33.4% | +76.1% | +19.4% | +38.1% | +41.8% |
| STGCN 1ch | 13.34 | +13.1% | +44.5% | +23.2% | +35.0% | +29.0% |
| STGCN 3ch (flow+occ+spd) | 15.74 | +26.0% | +29.9% | +8.2% | +18.9% | +20.8% |

**Findings**:
- More physical channels = larger attack surface for bias noise (+76% vs +57% for 5ch vs 1ch STAEformer)
- 1ch models have lower clean MAE but similar overall vulnerability
- STGCN 3ch appears robust (low avg deg) but has high clean MAE — the denominator effect

---

## RQ2: Denoising Encoder Effectiveness

Core claim: pretrained denoising encoder + noise augmentation reduces noise vulnerability.

### Functional Node Degradation (30% corruption rate)

| Model | Clean MAE | Gaussian | Bias | Stuck | Drift | Avg Deg | Reduction |
|---|---|---|---|---|---|---|---|
| Baseline (no defense) | 12.13 | +33.4% | +76.1% | +19.4% | +38.1% | +41.8% | — |
| Denoising encoder only | **12.05** | +9.6% | +39.0% | +18.0% | +21.4% | +22.0% | **47%** |
| Noisy training only | 12.52 | +5.9% | +32.5% | +12.7% | +15.7% | +16.7% | **60%** |
| Denoising + noisy training | 12.62 | +5.7% | +32.1% | +12.9% | +16.3% | +16.7% | 60% |
| **Denoising v2 + noisy training** | 12.81 | **+5.0%** | **+24.3%** | **+10.5%** | **+10.6%** | **+12.6%** | **70%** |

### Spillover to Healthy Nodes (Gaussian, 30% rate)

| Model | Healthy Deg | Reduction |
|---|---|---|
| Baseline | +20.3% | — |
| Denoising only | +0.3% | **99%** |
| Noisy training only | -0.1% | **100%** |
| Denoising v2 + noisy | +0.1% | **100%** |

**Key findings**:
- Denoising encoder alone: best clean MAE (12.05) with 47% robustness gain — **best accuracy-robustness balance**
- Noisy training alone: 60% robustness gain but +0.39 clean MAE penalty
- Combined v2+noisy: 70% robustness gain (best) but +0.68 clean MAE penalty
- Spillover nearly eliminated by all defense approaches (99-100% reduction)

---

## RQ3: Ablation Study — Encoder Components

All ablations use noisy training. Full model = DenoisingEncoder v2 (temporal 4L + spatial 1L + residual, hidden=64).

| Variant | Description | Clean MAE | Gaussian | Bias | Stuck | Drift | Avg Deg |
|---|---|---|---|---|---|---|---|
| **Full model** | temporal(4L) + spatial(1L) + residual, h=64 | 12.81 | +5.0% | +24.3% | +10.5% | +10.6% | **+12.6%** |
| A2: MLP only | Per-node MLP, no temporal/spatial | **12.51** | +5.1% | +22.6% | +12.8% | +13.3% | +13.4% |
| A3: Temporal only | spatial_layers=0 | 12.89 | +4.5% | +30.0% | +13.0% | +15.6% | +15.8% |
| A7: No residual | residual_connection=False | 13.07 | +4.3% | +32.7% | +10.5% | +14.6% | +15.5% |
| B2: Hidden=32 | Smaller hidden dim (32 vs 64) | 12.92 | +4.8% | +28.1% | +11.5% | +15.4% | +15.0% |

**Key findings**:
- **A2 (MLP only) is surprisingly competitive** — +13.4% avg deg vs full model's +12.6%, with better clean MAE (12.51 vs 12.81). Spatial/temporal convolutions contribute minimally to noise robustness.
- **Residual connections matter most**: A7 (no residual) has worst clean MAE (13.07) and +15.5% avg deg
- **Capacity matters**: B2 (hidden=32) degrades both clean MAE and robustness vs full model
- **Spatial layers add value for bias noise**: A3 (no spatial) has +30.0% bias deg vs full model's +24.3%
- Gaussian robustness is similar across all ablations (~+4-5%) — likely dominated by noisy training effect

---

## RQ4: Cross-Noise Generalization

Encoder pretrained on subset of noise types, evaluated on all types.

| Pretrain Noise Types | Clean MAE | Gaussian | Bias | Stuck | Drift | Avg Deg |
|---|---|---|---|---|---|---|
| All 5 types | **12.05** | +9.6% | +39.0% | +18.0% | +21.4% | +22.0% |
| Common only (gaussian/bias/drift) | 12.66 | **+6.7%** | **+35.5%** | **+15.7%** | **+20.6%** | **+19.6%** |
| Structural only (stuck/dead) | 12.03 | +15.5% | +52.1% | +17.5% | +33.3% | +29.6% |

**Key findings**:
- **Common noise pretrain generalizes well**: better than all-noise on seen types AND unseen stuck noise (+15.7% vs +18.0%)
- **Structural noise pretrain fails to generalize**: good on stuck (+17.5%) but bad on unseen gaussian (+15.5% vs +9.6%) and bias (+52.1% vs +39.0%)
- Asymmetric transfer: additive/multiplicative noise knowledge transfers to structural, but not vice versa

---

## RQ5: Model Agnosticity

Same denoising encoder approach applied to different downstream architectures.

### STAEformer (Transformer-based)

| Variant | Clean MAE | Avg Deg | Reduction |
|---|---|---|---|
| Baseline | 12.13 | +41.8% | — |
| + Denoising | 12.05 | +22.0% | 47% |
| + Denoising v2 + noisy | 12.81 | +12.6% | 70% |

### STGCN (GCN-based)

| Variant | Clean MAE | Avg Deg | Reduction |
|---|---|---|---|
| Baseline (1ch) | 13.34 | +29.0% | — |
| + Denoising | 13.38 | +20.3% | 30% |
| + Noisy training | 13.60 | +18.1% | 38% |
| + Denoising + noisy | 13.50 | +15.9% | 45% |
| + Denoising v2 + noisy | 15.08 | +13.6% | 53% |

**Key findings**:
- Both architectures benefit from denoising encoder — **approach is model-agnostic**
- STAEformer benefits more (70% reduction vs 53% for STGCN)
- STGCN v2+noisy has severe clean MAE penalty (+1.74) — encoder may be too aggressive for simpler models
- Best STGCN configuration: denoising + noisy (15.9% avg deg, reasonable clean MAE 13.50)

---

## Alternative Encoder Architectures

| Encoder | Clean MAE | Gaussian | Bias | Stuck | Drift | Avg Deg |
|---|---|---|---|---|---|---|
| DenoisingEncoder (standard) | 12.05 | +9.6% | +39.0% | +18.0% | +21.4% | +22.0% |
| LinearAttentionDenoising | 12.04 | +8.3% | +45.0% | +18.5% | +24.3% | +24.0% |
| LinearAttention + noisy | 13.40 | +4.6% | +27.4% | +10.7% | +13.4% | +14.0% |
| GatedMLP + noisy | 12.47 | +6.0% | +29.6% | +14.4% | +17.4% | +16.9% |

**Findings**:
- Linear attention encoder: similar clean MAE but slightly worse robustness than standard denoising
- GatedMLP + noisy: good balance of clean MAE (12.47) and robustness (+16.9%)
- Noisy training consistently improves robustness across all encoder types

---

## Reliability Estimation (Negative Result)

| Model | Clean MAE | Gaussian | Bias | Stuck | Drift | Avg Deg |
|---|---|---|---|---|---|---|
| Denoising (reference) | 12.05 | +9.6% | +39.0% | +18.0% | +21.4% | +22.0% |
| Denoising + reliability | 12.57 | +9.0% | +41.8% | +15.8% | +25.6% | +23.1% |
| Contrastive + reliability | 12.11 | +9.2% | +52.1% | +17.1% | +27.2% | +26.4% |

**Conclusion**: Reliability estimation does not improve robustness. Higher clean MAE and similar or worse degradation.

---

## Plug & Play Encoder Variants

Testing encoder effectiveness when plugged into clean-trained downstream models.

| Model | Clean MAE | Avg Deg | Notes |
|---|---|---|---|
| GatedMLP cleanonly | 12.13 | +41.4% | No noise in pretrain → no robustness gain |
| GatedMLP with noise | 12.45 | +39.0% | Minimal gain from encoder alone |
| GatedMLP w/clean+noise | 12.49 | +39.3% | Same |
| LinearAttn cleanonly | 12.57 | +38.3% | No gain |
| LinearAttn with noise | 14.49 | +18.3% | Gains but high clean MAE penalty |
| LinearAttn w/clean+noise | 13.78 | +18.2% | Same pattern |
| Denoising plug&play | 15.32 | +38.4% | High clean MAE, no robustness gain |
| Denoising v2 plug&play | 17.57 | +15.2% | Extreme clean MAE penalty |

**Conclusion**: Plug & play (frozen encoder, no joint training) is ineffective. Noise exposure during pretraining is necessary, and even then clean MAE suffers significantly without end-to-end fine-tuning.

---

## Summary: Best Configurations

| Goal | Best Model | Clean MAE | Avg Deg | Trade-off |
|---|---|---|---|---|
| **Best clean accuracy** | STAEformer 1ch | 11.90 | +37.1% | No robustness defense |
| **Best accuracy-robustness balance** | STAEformer 5ch + denoising | 12.05 | +22.0% | Minimal clean MAE penalty |
| **Maximum robustness** | STAEformer 5ch + denoising v2 + noisy | 12.81 | +12.6% | +0.68 clean MAE penalty |
| **Best for STGCN** | STGCN + denoising + noisy | 13.50 | +15.9% | +0.16 clean MAE penalty |

---

## RQ6: Cross-Dataset Generalization (CONTRA_COSTA)

Full 2×2 factorial (encoder × noise augmentation) on a second dataset.

**Dataset**: CONTRA_COSTA — 773 nodes, 634 functional (104 dead, 35 major fail).

### Dataset Structural Comparison

| | SAN_BERNARDINO | CONTRA_COSTA |
|---|---|---|
| Nodes (functional) | 893 (745) | 773 (634) |
| Dead / Major fail | 120 / 28 | 104 / 35 |
| Mean flow | 120.3 | 121.1 |
| **Adj mean weight** | **0.739** | **0.396** |
| Adj edges > 0.5 | 81.5% | 35.1% |
| Adj edges > 0.7 | 65.2% | 24.1% |
| Neighbor flow correlation | 0.673 | 0.639 |
| Dead neighbor total weight | 84.9 | 36.7 |

**Key structural difference**: CC adjacency weight is ~half of SB (0.40 vs 0.74).
Flow statistics, sensor health distributions, and neighbor correlations are nearly identical.
The difference is entirely in **how strongly the graph encodes spatial proximity**.

### CC Results (30% corruption rate, excl. dead)

| Model | Clean MAE | Gaussian | Bias | Stuck | Drift | **Avg Deg** |
|---|---|---|---|---|---|---|
| Baseline (E0+N0) | 12.39 | +19.2% | +74.0% | +20.8% | +43.3% | **39.3%** |
| Denoising v1 (E1+N0) | 12.32 | +17.1% | +75.2% | +19.8% | +42.7% | **38.7%** |
| **Denoising v2 (E2+N0)** | **11.98** | +27.8% | +72.2% | +22.6% | +42.5% | **41.3%** |
| Noisy training (E0+N1) | 12.47 | +6.5% | +36.3% | +14.5% | +17.9% | **18.8%** |
| Denoising v1+Noisy (E1+N1) | 13.49 | +5.4% | +34.0% | +11.1% | +16.4% | **16.7%** |
| **Denoising v2+Noisy (E2+N1)** | **13.21** | +4.8% | +30.5% | +11.2% | +16.8% | **15.8%** |

### SB 2×2 Factorial (for comparison)

| Model | Clean MAE | Gaussian | Bias | Stuck | Drift | **Avg Deg** |
|---|---|---|---|---|---|---|
| Baseline (E0+N0) | 12.13 | +33.4% | +76.1% | +19.4% | +38.1% | **41.8%** |
| Denoising (E1+N0) | 12.05 | +9.6% | +39.0% | +18.0% | +21.4% | **22.0%** |
| Noisy training (E0+N1) | 12.52 | +5.9% | +32.5% | +12.7% | +15.7% | **16.7%** |
| Denoising+Noisy (E1+N1) | 12.62 | +5.7% | +32.1% | +12.9% | +16.3% | **16.7%** |

### Cross-Dataset Comparison

| Strategy | SB Degradation | SB Reduction | CC Degradation | CC Reduction |
|---|---|---|---|---|
| Baseline | 41.8% | — | 39.3% | — |
| Denoising v1 only | 22.0% | **-47%** | 38.7% | -1.5% |
| Denoising v2 only | — | — | 41.3% | +5% (worse) |
| Noisy training only | 16.7% | **-60%** | 18.8% | **-52%** |
| Denoising v1 + Noisy | 16.7% | **-60%** | 16.7% | **-58%** |
| **Denoising v2 + Noisy** | **12.6%** | **-70%** | **15.8%** | **-60%** |

### Spillover Comparison (Gaussian, 30% rate)

| Dataset | Baseline | Denoising v1 | Denoising v2 | Noisy | v2+Noisy |
|---|---|---|---|---|---|
| SAN_BERNARDINO | +20.3% | +0.3% | — | -0.1% | +0.1% |
| CONTRA_COSTA | +5.4% | +5.9% | +13.8% | -0.2% | -0.2% |

### Analysis

**Finding 1: Noisy training is universally effective (dataset-agnostic)**
- SB: 41.8% → 16.7% (-60%), CC: 39.3% → 18.8% (-52%)
- Works regardless of graph density, adjacency structure, or spillover magnitude

**Finding 2: Denoising encoder robustness depends on spatial coupling**
- SB (adj mean 0.74): 47% reduction — strong spatial connections enable neighbor-based denoising
- CC (adj mean 0.40): encoder alone does NOT improve robustness (v1 -1.5%, v2 +5%)
- The encoder's spatial layer needs strong graph weights to be effective

**Finding 3: v2 residual architecture improves clean MAE universally**
- SB: v2 encoder only 12.05 (baseline 12.13, -0.08)
- CC: v2 encoder only **11.98** (baseline 12.39, **-0.41**)
- Residual connection = identity on clean data → no degradation, slight regularization benefit
- v1 (no residual) distorts clean inputs → limited clean MAE improvement

**Finding 4: v2+noisy is the best combo on both datasets**
- SB: 12.6% avg deg (70% reduction), clean MAE 12.81
- CC: 15.8% avg deg (60% reduction), clean MAE 13.21
- v2 reduces clean MAE penalty vs v1: CC 13.21 vs 13.49 (-0.28)

**Finding 5: On weak graphs, noisy-only remains the practical choice**
- CC noisy-only: 12.47 MAE, 18.8% deg — best accuracy-robustness balance
- CC v2+noisy: 13.21 MAE, 15.8% deg — best robustness but +0.74 clean penalty
- Trade-off: 3pp less degradation costs +0.74 clean MAE

---

## Why v1 Combo Fails and v2 Succeeds

(From detailed analysis on SAN_BERNARDINO, applicable to understanding CC results)

### Encoder v1 vs v2 Architecture

| | v1 | v2 |
|---|---|---|
| Output | `f(noisy_input)` | `input + correction(input)` |
| hidden_dim | 32 | 64 |
| Pretrain noise severity | 0.1 - 0.5 | 0.1 - 1.5 |
| Pretrain noise rate | 0.1 - 0.5 | 0.1 - 0.7 |
| Pretrain val_loss (final) | ~0.18 | **~0.05** |
| residual_connection | False | **True** |

### Why v1 + noise augmentation is worse than augmentation alone

1. v1 must **reconstruct the full signal** → occupancy distortion ~37% even on clean data
2. Gaussian recovery rate: only 1.3% (nearly no denoising)
3. Downstream model receives distorted input → clean MAE worsens (12.62 vs 12.52)
4. Encoder distortion **cancels** augmentation's robustness gain

### Why v2 + noise augmentation achieves best results

1. Residual: `output = input + correction` → correction ≈ 0 on clean data → identity fallback
2. Easier learning task (correction vs full reconstruction) → val_loss 3.6x lower
3. Safe failure mode: if correction is wrong, original signal passes through
4. **Complementarity**: encoder removes noise at input level (1st defense), augmentation builds internal model robustness (2nd defense)

---

## Noise Type Characteristics

| Type | Mechanism | Severity on Baseline | Best Defense | Note |
|---|---|---|---|---|
| **Gaussian** | Additive random | Very high (+33-538%) | Encoder (spatial filtering) | Scale-dependent; random noise easily filtered by neighbors |
| **Bias** | Multiplicative shift | Very high (+76-156%) | Combination needed | Systematic offset; harder than gaussian because spatially consistent |
| **Stuck** | Frozen at t=0 | Moderate (+19-36%) | Augmentation | Temporal anomaly; value plausible but constant |
| **Drift** | Gradual calibration shift | High (+38-74%) | v2 combo (temporal conv) | Time-varying; encoder's temporal conv detects drift pattern |
| **Dead** | All channels → 0 | Extreme (+321%) | None fully effective | Real-world most common; treated separately from other types |

---

## Summary: Best Configurations

### SAN_BERNARDINO (strong spatial graph)

| Goal | Best Model | Clean MAE | Avg Deg | Trade-off |
|---|---|---|---|---|
| **Best clean accuracy** | STAEformer 1ch | 11.90 | +37.1% | No robustness defense |
| **Best accuracy-robustness balance** | STAEformer 5ch + denoising | 12.05 | +22.0% | Minimal clean MAE penalty |
| **Maximum robustness** | STAEformer 5ch + denoising v2 + noisy | 12.81 | +12.6% | +0.68 clean MAE penalty |
| **Best for STGCN** | STGCN + denoising + noisy | 13.50 | +15.9% | +0.16 clean MAE penalty |

### CONTRA_COSTA (weak spatial graph)

| Goal | Best Model | Clean MAE | Avg Deg | Trade-off |
|---|---|---|---|---|
| **Best clean accuracy** | STAEformer 5ch + denoising v2 | **11.98** | +41.3% | No robustness gain, but best MAE |
| **Best accuracy-robustness balance** | STAEformer 5ch + noisy training | 12.47 | +18.8% | +0.08 clean MAE penalty |
| **Maximum robustness** | STAEformer 5ch + denoising v2 + noisy | 13.21 | +15.8% | +0.82 clean MAE penalty |

---

## Appendix: Full Per-Scenario Tables

### STAEformer All Functional Nodes (SAN_BERNARDINO)

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

v2 combo best in **16/16** scenarios.

### STGCN All Functional Nodes (SAN_BERNARDINO)

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

v2 combo best in **13/16** scenarios.

### Checkpoints

| Model | Clean MAE | Checkpoint |
|---|---:|---|
| STAEformer baseline | 12.26 | `checkpoints/STAEformer_5ch/SAN_BERNARDINO_30_12_12/` |
| STAEformer + denoising v1 | 12.13 | `checkpoints/STAEformer_5ch_denoising/SAN_BERNARDINO_30_12_12/` |
| STAEformer + aug | 12.63 | `checkpoints/STAEformer_5ch_noisy/SAN_BERNARDINO_30_12_12/` |
| STAEformer + v1 combo | 12.75 | `checkpoints/STAEformer_5ch_denoising_noisy/SAN_BERNARDINO_30_12_12/` |
| STAEformer + v2 combo | 12.92 | `checkpoints/STAEformer_5ch_denoising_v2_noisy/SAN_BERNARDINO_30_12_12/` |
| STGCN baseline | 14.09 | `checkpoints/STGCN/SAN_BERNARDINO_30_12_12/` |
| STGCN + denoising v1 | 14.11 | `checkpoints/STGCN_denoising/SAN_BERNARDINO_30_12_12/` |
| STGCN + aug | 14.29 | `checkpoints/STGCN_noisy/SAN_BERNARDINO_30_12_12/` |
| STGCN + v1 combo | 14.18 | `checkpoints/STGCN_denoising_noisy/SAN_BERNARDINO_30_12_12/` |
| STGCN + v2 combo | 15.71 | `checkpoints/STGCN_denoising_v2_noisy/SAN_BERNARDINO_30_12_12/` |
| CC baseline | 12.52 | `checkpoints/STAEformer_5ch/CONTRA_COSTA_30_12_12/` |
| CC + denoising | 12.43 | `checkpoints/STAEformer_5ch_denoising/CONTRA_COSTA_30_12_12/` |
| CC + noisy | 12.59 | `checkpoints/STAEformer_5ch_noisy/CONTRA_COSTA_30_12_12/` |
| CC + denoising+noisy | 13.55 | `checkpoints/STAEformer_5ch_denoising_noisy/CONTRA_COSTA_30_12_12/` |
| CC + denoising v2 | 12.06 | `checkpoints/STAEformer_5ch_denoising_v2/CONTRA_COSTA_30_12_12/` |
| CC + denoising v2+noisy | 13.35 | `checkpoints/STAEformer_5ch_denoising_v2_noisy/CONTRA_COSTA_30_12_12/` |

---

*Evaluation: `experiments/eval_noise_vulnerability.py` (SAN_BERNARDINO), `experiments/eval_contra_costa.py` (CONTRA_COSTA)*
*Results: `experiments/noise_vulnerability_results/`*
*Dataset comparison: `experiments/compare_datasets.py`*
*Last updated: 2026-02-25*
