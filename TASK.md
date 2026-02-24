# Tasks — Robust Spatio-Temporal Traffic Forecasting

## 🔥 In Progress (Noise-Resilient)
- [ ] RQ3 ablation A7 downstream (no residual connection) — training ~epoch 23/30
- [ ] RQ1 1ch STAEformer baseline — training ~epoch 5/30
- [ ] RQ5 3ch STGCN baseline — training ~epoch 4/30
- [ ] RQ4 cross-noise pretraining: common noise (~epoch 4/30), structural noise (~epoch 17/30)
- [ ] RQ2 CONTRA_COSTA denoising pretrain (~epoch 4/30)

## 📋 Backlog (Noise-Resilient)
- [ ] Check all running experiments for completion, collect test_metrics.json results
- [ ] Queue ablation A2 (MLP-only encoder), A3 (temporal-only), B2 (hidden=32) downstream runs after A7 completes
- [ ] Auto-launch cross-noise downstream experiments after pretraining completes
- [ ] Compile RQ1-5 results into unified comparison table (paper Table 1)
- [ ] Run combined approach: denoising encoder + credibility bias + noise augmentation
- [ ] Evaluate on additional datasets: METR-LA, PEMS-BAY (generalization beyond xtraffic)
- [ ] Write paper Section 4 (Experiments) with compiled results
- [ ] Generate attention visualization figures for paper (dead vs functional sensor attention maps)

## 📋 Backlog (EPN Few-Shot — paused, lower priority)
- [ ] Evaluate EPN on real new nodes (current eval uses synthetic masking of existing nodes)
- [ ] Cross-architecture EPN transfer: train on STAEformer NSPs, test on AGCRN
- [ ] Test larger budget regime (14d, 30d) — currently only 3h-7d
- [ ] Investigate why EPN only beats mean_init on 52-58% of nodes
- [ ] Compare EPN against simple transfer learning baselines (e.g., nearest-neighbor NSP copy)

## 🚧 Blocked
- [ ] RQ3 ablations A2/A3/B2 — blocked by: GPU occupied with A7 + other RQ experiments

## ✅ Recently Done (last 2 weeks)
- [x] Implemented STAEformerCredibility with pre-softmax attention bias (MAE 12.178→12.151)
- [x] Completed attention collapse analysis — dead sensors get 3.5-12x more spatial attention
- [x] Identified masked vs unmasked MAE artifact: unmasked "improvement" driven by dead sensors (r=0.88)
- [x] Documented 6 real noise types in SAN_BERNARDINO (dead, stuck, momentary zero, drift, jump, extreme)
- [x] Built denoising encoder v2 (residual + stronger noise + hidden=64): up to 14x noise degradation reduction
- [x] Fixed DeepAR runner bug (hotfix/deepar branch)
- [x] EPN deep analysis: NSP space structure (PCA 11x compression, pairwise cosine 0.38)
- [x] EPN evaluation on Q2/Q3/Q4 expansion: 1-5% MAE improvement at 7d, zero forgetting confirmed
- [x] Masked pre-training ablation: learnable default embedding provides modest gains at 7d budget
- [x] N-dependency analysis across 5 STGNN architectures (AGCRN, STAEformer, STGCN, GWNet, D2STGNN)

## 📅 Deadlines
- No hard deadlines currently — paper submission target TBD
