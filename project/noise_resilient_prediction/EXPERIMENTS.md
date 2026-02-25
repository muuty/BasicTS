# Noise-Resilient Traffic Prediction: Experiment Results

Comprehensive noise robustness evaluation across 33 model configurations.
All results on SAN_BERNARDINO dataset (893 nodes, 745 functional).

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

## Cross-Dataset Generalization: CONTRA_COSTA

Same approach evaluated on a second dataset to test generalizability.

**Dataset**: CONTRA_COSTA — 773 nodes, 634 functional (104 dead, 35 major fail).

| Model | Clean MAE | Gaussian | Bias | Stuck | Drift | Dead | Avg Deg (excl. dead) |
|---|---|---|---|---|---|---|---|
| STAEformer 5ch Baseline | 12.39 | +19.2% | +74.0% | +20.8% | +43.3% | +320.9% | +39.3% |
| STAEformer 5ch + Denoising | **12.32** | **+17.1%** | +75.2% | **+19.8%** | **+42.7%** | +338.3% | **+38.7%** |

### Spillover (Gaussian, 30% rate)

| Model | Healthy Deg |
|---|---|
| Baseline | +5.4% |
| Denoising | +5.9% |

**Key findings**:
- Denoising encoder provides **marginal improvement** on CONTRA_COSTA (avg deg 38.7% vs 39.3%) — much less than SAN_BERNARDINO (22.0% vs 41.8%)
- Clean MAE slightly improved (12.32 vs 12.39)
- Bias noise remains equally devastating with denoising (+75.2% ≈ +74.0%)
- Spillover NOT reduced (5.9% vs 5.4%) — contrasts sharply with SAN_BERNARDINO (0.3% vs 20.3%)
- **Possible causes**: (1) encoder pretrained with `residual_connection=False` and smaller hidden_dim=32 (vs 64 in SAN_BERNARDINO), (2) frozen encoder without noisy training limits effectiveness, (3) different noise characteristics between datasets

---

*Evaluation: `experiments/eval_noise_vulnerability.py` (SAN_BERNARDINO), `experiments/eval_contra_costa.py` (CONTRA_COSTA)*
*Results: `experiments/noise_vulnerability_results/`*
*Last updated: 2026-02-25*
