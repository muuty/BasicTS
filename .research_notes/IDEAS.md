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

## Literature-Backed Novelty Ideas (2026-02-25)
> See `project/noise_resilient_prediction/LITERATURE.md` for full reference base.

### Priority 1: Dynamic Quality-Aware Attention (LOW effort)
- Extend `STAEformerCredibility` static bias → window-adaptive, input-dependent quality bias
- `quality_score(v, t) = MLP(flow_std, zero_rate, occ_mean, speed_cv)` per window
- Differentiates from RT-GCN (static Gaussian) — ours is multi-channel, dynamic
- ~100-200 lines modification of existing code

### Priority 2: Sensor-Health-Conditioned Uncertainty (LOW-MED effort)
- Add 2nd output head: `mu, log_sigma = model(input)` with heteroscedastic NLL loss
- sigma explicitly conditioned on sensor health features → "input quality → output uncertainty"
- Gap: existing traffic uncertainty papers (DeepSTUQ, ICDM'23) only model future uncertainty, not input quality
- Practical value: downstream systems discount uncertain predictions automatically

### Priority 3: Noise Curriculum Learning (LOW effort)
- Progressive noise injection: clean → mild → full noise during training
- Based on RobustTSF (ICLR'24) curriculum concept applied to sensor reliability
- No architectural changes — just modify training loop's noise scheduling
- Unexplored in traffic forecasting

### Priority 4: Failure-Mode-Aware Augmentation (LOW effort)
- Extract empirical sensor failure patterns from SB/CC data (temporal structure of failures)
- Real dead sensors: sudden 0 → intermittent recovery → permanent death
- Current synthetic noise lacks this temporal structure
- Differentiates from SAFER-Predictor (adversarial) — ours is empirical/physics-informed

### Priority 5: Spatial Contrastive Denoising (MED effort)
- Fix why cross-variable pretrain failed: contrastive loss > reconstruction loss for noise-invariant repr
- DECL (IJCAI'24) + spatial graph structure = unexplored combination
- Positive pairs: same sensor {clean vs noisy}, Negative: different cluster

### Longer-Term: MoE with Sensor-Quality Routing (HIGH effort)
- Expert 1: healthy sensors, Expert 2: degraded, Expert 3: dead
- Router conditioned on real-time quality features
- Theoretical backing: MoE sparse activation = noise filter (arXiv 2025)
- Very novel but requires significant architecture changes

## Uncertain / Risky Ideas
- Diffusion-based sensor imputation: generate realistic traffic data for dead sensors conditioned on neighbors — may be overkill vs simple interpolation
- Causal attention: use intervention theory to identify true causal neighbors vs spurious correlations — theoretically appealing but computationally expensive
- GAN-based noise augmentation: learn realistic noise distributions from real failures rather than synthetic gaussian/stuck/drift — may not generalize to unseen failure modes
- Noise-aware NSP initialization: bridge both projects — when new sensor is installed in noisy area, initialize its embedding accounting for neighbor sensor health
- Self-supervised anomaly detection as auxiliary task: jointly predict traffic + detect noise — multi-task signal may regularize both
