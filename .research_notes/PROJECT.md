# Robust Spatio-Temporal Traffic Forecasting

## Overview
Investigating robustness of spatio-temporal graph neural networks (STGNNs) for traffic prediction under real-world sensor noise and failure. The SAN_BERNARDINO dataset (893 sensors) has ~34% dead/failing sensors that cause attention collapse and prediction degradation. This project develops denoising encoders, credibility-aware attention, and noise augmentation strategies to improve robustness.

## Goals
- [ ] Short-term: Complete RQ1-5 paper experiments (ablations, cross-noise generalization, multi-model validation)
- [ ] Mid-term: Submit paper on noise-resilient spatio-temporal forecasting
- [ ] Long-term: Generalize to multi-city deployment and combine with few-shot node onboarding (EPN)

## Background
- **Attention collapse**: Dead/failing sensors (zero-filled) receive 3.5-12x more spatial attention than functional sensors due to consistent key representations in softmax
- **Encoder paradox**: Pre-trained encoders (contrastive, masking) don't improve clean prediction — "improvements" in unmasked MAE are dead sensor artifacts (r=0.88 with zero_flow_rate)
- **Credibility bias**: Pre-softmax attention bias from raw embeddings reduces dead sensor attention, MAE 12.178→12.151
- **Denoising encoder**: Reconstruction-based encoder with residual connections achieves up to 14x reduction in noise-induced degradation
- **6 real noise types identified**: Dead (33% nodes), Stuck (78%), Momentary Zero (66%), Drift (14%), Large Jump (18%), Extreme High (4%)

## Key Resources
- **Repo:** `/data/pretrainingbasicts/` (branch: hotfix/deepar)
- **Dataset:** SAN_BERNARDINO — 893 nodes, 5 features (flow, occ, speed, tod, dow), 105120 timesteps, 5-min intervals
- **Sensor categories:** 121 dead (>90% zero), 28 major fail (50-90%), 161 partial fail (5-50%), 583 functional (<5%)
- **Key papers:** STAEformer (Liu et al.), GPT-ST, STEP, STMAE — all implemented as baselines in this repo
- **Compute:** Single GPU (GPU 1 only; GPU 0 reserved for other users)
- **Experiment log:** `project/noise_resilient_prediction/representation_learning_experiment_log.md` (91KB)
- **Attention analysis:** `project/noise_resilient_prediction/attention_collapse_missing_values.md`

## Current Status
RQ1-5 paper experiments are running (ablation A2/A3/A7/B2, cross-noise pretraining, CONTRA_COSTA generalization, 1ch/3ch baselines). Denoising encoder v2 shows strong robustness gains but modest clean-data improvement. Need to compile all results into paper-ready tables.

## Collaborators / Advisors
- Solo research project within lab group

---

# Secondary Project: Few-Shot Node Onboarding (EPN)

**Status: Paused** — will resume after noise-resilient paper

## Overview
When traffic networks expand (new sensors installed quarterly), existing STGNN models need to adapt with minimal data (3h-7d). EPN (Embedding Predictor Network) predicts learnable node embeddings for new nodes from their traffic pattern + neighbor context.

## Key Insight
Across all STGNNs (AGCRN, STAEformer, STGCN, GWNet), the ONLY parameters that scale with node count N are Node-Specific Parameters (NSPs). All other weights transfer directly. This means network expansion requires adapting only NSPs, with zero forgetting guaranteed via gradient masking.

## Results So Far
- EPN achieves 1-5% MAE improvement at 7d budget with zero forgetting on existing nodes
- Mean initialization is surprisingly competitive — EPN only beats it on 52-58% of nodes
- NSP space is highly structured: PCA compresses 11x (28 of 288 components for 80% variance)
- Positioned against TrafficStream (IJCAI'21), PECPM (KDD'23), EAC (ICLR'25)

## Key Resources
- **Code:** `project/few-shot-adaptation/` (epn_model.py, train_epn.py, evaluate_*.py)
- **Concept drift:** `project/concept_drift/expanding_adaptation*.py`
- **Related work:** `project/few-shot-adaptation/related_work.md`
- **Results:** `project/few-shot-adaptation/results/`
