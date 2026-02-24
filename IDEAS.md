# Ideas — Robust Spatio-Temporal Traffic Forecasting

## Experiment Ideas

### Noise-Resilient
- Joint approach: combine denoising encoder + credibility bias + noise augmentation in single model — each addresses different failure mode, may be complementary
- Combined masking pre-training: temporal + feature + spatial simultaneously — individual masking each gives ~3% improvement, synergy unexplored
- Test-Time Adaptation (TTA): online encoder adaptation at inference when noise patterns shift — framework exists in `baselines/STAEformer/arch/` (TTA configs)
- Adaptive/hybrid encoder: MLP path for dead sensors, ST encoder for stuck sensors — noise pattern analysis shows opposite optimal encoders for different failure types
- Curriculum learning: train on easy (functional) nodes first, gradually introduce noisy nodes — may stabilize early training
- Per-noise-type evaluation: break down RQ results by the 6 noise types rather than aggregate — may reveal which approach works for which noise

### EPN Few-Shot
- Cross-architecture EPN: train EPN on STAEformer NSPs, predict AGCRN node_embeddings — tests universality of traffic pattern → NSP mapping
- Universal EPN: single EPN that outputs NSPs for any architecture given architecture descriptor — ambitious but high impact
- Temporal EPN: predict NSP evolution over time (not just static) — addresses concept drift + expansion simultaneously
- Proxy fine-tuning with noise augmentation: add synthetic noise during EPN training episodes for robustness

## Research Questions
- Why don't better SSL representations improve clean prediction? STAEformer already learns sufficient patterns from raw features — is this architecture-specific or fundamental?
- Can credibility bias scale to STGCN/AGCRN or is it transformer-specific? Graph convolution doesn't have softmax attention
- What determines whether EPN beats mean_init for a given node? (only 52-58% benefit) — traffic pattern complexity? graph centrality? sensor reliability?
- Cross-city transfer: can denoising encoder trained on SAN_BERNARDINO improve METR-LA/PEMS-BAY robustness without retraining?
- Is the attention collapse problem unique to traffic or present in any graph transformer with missing data?

## Future Directions
- **Continual learning for concept drift**: quarterly model updates that handle both sensor expansion and temporal pattern shift — bridge noise-resilient + EPN projects
- **Multimodal fusion**: incorporate weather, incident reports, construction data — external signals may explain "unexplainable" noise patterns
- **Domain adaptation**: pre-train on large multi-city corpus, fine-tune on target city with few-shot — foundation model for traffic
- **Interpretable node health scoring**: unsupervised detection of sensor degradation from prediction residuals — practical deployment value

## Uncertain / Risky Ideas
- Diffusion-based sensor imputation: generate realistic traffic data for dead sensors conditioned on neighbors — may be overkill vs simple interpolation
- Causal attention: use intervention theory to identify true causal neighbors vs spurious correlations — theoretically appealing but computationally expensive
- GAN-based noise augmentation: learn realistic noise distributions from real failures rather than synthetic gaussian/stuck/drift — may not generalize to unseen failure modes
- Noise-aware NSP initialization: bridge both projects — when new sensor is installed in noisy area, initialize its embedding accounting for neighbor sensor health
- Self-supervised anomaly detection as auxiliary task: jointly predict traffic + detect noise — multi-task signal may regularize both
