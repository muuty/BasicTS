# Weekly Log — Incident-Aware Traffic Forecasting

---

## 2026-02-24

### This Week's Goal
Organize the codebase (months of accumulated code/configs) and advance Phase A distance screening toward completion.

### Accomplished
- Major codebase cleanup: ~220 files organized into 7 logical commits on `feature/incident-aware`
- Removed dead code (gate integration, vendored easytorch, legacy selection)
- Set up structured project documentation (CLAUDE.md, project background, design docs)
- Phase A training progressed from ~87 to ~135/220 runs
- Pre-computed 88/110 coreset index files for SAN_BERNARDINO

### Not Accomplished (and why)
- Phase A not complete — SLURM queue bottleneck (max 4 concurrent jobs, each 1.5-7h)
- Cosine pipeline graph_cut indices not yet computed — waiting for k_medoids jobs to finish
- No progress on severity-aware pre-training — intentionally deferred until Phase A completes

### Key Insights
- K-Center + combined distance is consistently the best performer at ratio=0.7
- Graph Cut is too seed-sensitive to be reliable without more seeds
- The codebase cleanup revealed the project has tested 5 distinct approaches, with coreset selection being the most promising

### Next Week Plan
- [ ] Complete all 220 Phase A training runs
- [ ] Run full Phase A analysis: distance ranking, method comparison, ratio effect
- [ ] Investigate STGformer zero-MAE runs
- [ ] Start severity-aware pre-training implementation if Phase A analysis is clear

### Health Check
- Momentum: 🟡 slow — productive on code organization, but experiment throughput limited by SLURM
- Confidence in direction: 🟢 good — coreset selection shows clear promise, pre-training design is solid
