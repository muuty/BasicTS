# Tasks — Robust Spatio-Temporal Traffic Forecasting

## 🔥 In Progress (Noise-Resilient)
(none currently)

## 📋 Backlog (Noise-Resilient — Novelty Enhancement)
- [ ] Implement dynamic quality-aware attention (extend STAEformerCredibility with window-adaptive bias)
- [ ] Implement sensor-health-conditioned uncertainty output (heteroscedastic NLL loss)
- [ ] Implement noise curriculum learning (epoch-dependent noise_prob in NoisyRunner)
- [ ] Implement failure-mode-aware augmentation (empirical sensor failure patterns from SB/CC data)

## 📋 Backlog (Noise-Resilient — Paper)
- [ ] Compile RQ1-6 results into unified comparison table (paper Table 1)
- [ ] Write paper Section 4 (Experiments) with compiled results
- [ ] Generate attention visualization figures for paper (dead vs functional sensor attention maps)
- [ ] Evaluate on additional datasets: METR-LA, PEMS-BAY (generalization beyond xtraffic)
- [ ] Run combined approach: denoising encoder + credibility bias + noise augmentation

## 📋 Backlog (EPN Few-Shot — paused, lower priority)
- [ ] Evaluate EPN on real new nodes (current eval uses synthetic masking of existing nodes)
- [ ] Cross-architecture EPN transfer: train on STAEformer NSPs, test on AGCRN
- [ ] Test larger budget regime (14d, 30d) — currently only 3h-7d
- [ ] Investigate why EPN only beats mean_init on 52-58% of nodes
- [ ] Compare EPN against simple transfer learning baselines (e.g., nearest-neighbor NSP copy)

## ✅ Recently Done (last 2 weeks)
- [x] **CC v2 denoising encoder** — best clean MAE on CC (11.98), v2+noisy best robustness (15.8% avg deg)
- [x] CC v1 2×2 factorial (encoder × noisy) — noisy training universally effective (-52%)
- [x] RQ1-6 all experiments completed (all GPUs idle)
- [x] RQ3 ablation A2/A3/A7/B2 — MLP surprisingly competitive, residual matters most
- [x] RQ4 cross-noise generalization — common→structural transfers, not vice versa
- [x] RQ5 STGCN model agnosticity — both architectures benefit (70%/53%)
- [x] Implemented STAEformerCredibility with pre-softmax attention bias (MAE 12.178→12.151)
- [x] Completed attention collapse analysis — dead sensors get 3.5-12x more spatial attention
- [x] Built denoising encoder v2 (residual + stronger noise + hidden=64): up to 14x noise degradation reduction
- [x] Cross-dataset analysis: encoder robustness depends on adj weight (SB 0.74 vs CC 0.40)

## 📅 Deadlines
- No hard deadlines currently — paper submission target TBD
