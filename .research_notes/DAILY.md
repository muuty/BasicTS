# Daily Log — Incident-Aware Traffic Forecasting

---

## 2026-02-24

### Done
- Reorganized entire codebase into 7 logical git commits on `feature/incident-aware` branch
  - Removed vendored easytorch and legacy selection module
  - Cleaned up model architectures (removed gate integration, added STGformer)
  - Added incident-aware training infrastructure
  - Added experiment configs, coreset framework, orchestration, and documentation
- Set up project documentation system (CLAUDE.md, PROJECT.md, TASK.md, etc.)
- Phase A distance screening: ~135/220 training runs complete (up from ~87 earlier today)

### Observations / Results
- K-Center + combined distance @ ratio=0.7 remains most stable (MAE ~14.0-14.6)
- Graph Cut shows high seed sensitivity (MAE range 14.7-18.7) — may need more seeds
- Cosine pipeline k_medoids still running; graph_cut indices pending
- Some STGformer runs showing MAE=0.0 — needs investigation (possible early termination)

### Blockers
- SLURM QOSMaxJobsPerUserLimit caps at 4 concurrent jobs — Phase A completion is slow

### Tomorrow
- Monitor Phase A training progress, check for failed/zero-result runs
- Start preliminary analysis on completed L2 pipeline results
- Begin cosine graph_cut index computation if queue opens up
