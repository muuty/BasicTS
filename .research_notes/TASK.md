# Tasks — Incident-Aware Traffic Forecasting

## 🔥 In Progress
- [ ] Complete Phase A distance screening training (135/220 runs done, SLURM bottleneck)
- [ ] Monitor cosine pipeline: k_medoids running, graph_cut pending for cosine distances

## 📋 Backlog
- [ ] Analyze Phase A results: rank distance functions across models, methods, ratios (after 220 runs complete)
- [ ] Visualize Phase A: heatmap of MAE by distance × method, seed variance analysis
- [ ] Debug STGformer zero-MAE runs in Phase A (some combos produce MAE=0.0 — likely early termination)
- [ ] Implement severity-aware pre-training Stage 1 (contrastive, design in `docs/design.md`)
- [ ] Integrate severity-aware encoder with STGCN and STAEformer backbones (Stage 2)
- [ ] Run Phase A best settings on ALAMEDA and SACRAMENTO datasets (generalization check)
- [ ] Add CLEAR, STD-MAE, SCPT baselines for comparison
- [ ] Phase B: main benchmark (1,080 runs) with locked distance + expanded method/ratio grid
- [ ] Phase C: dataset verification (~144 runs across ALAMEDA, SACRAMENTO)
- [ ] Phase D: cross-model coreset transferability (~108 runs)

## 🚧 Blocked
- [ ] Cosine graph_cut index computation — blocked by: SLURM job queue (QOSMaxJobsPerUserLimit, max 4 concurrent)
- [ ] Phase A full analysis — blocked by: training runs not yet complete (~85 remaining)

## ✅ Recently Done
- [x] Reorganize codebase into 7 logical git commits (834aeae → aec777b)
- [x] Set up project documentation (CLAUDE.md, PROJECT_BACKGROUND.md, design.md)
- [x] Pre-compute 88/110 coreset index files for SAN_BERNARDINO
- [x] Complete adjacency matrix experiment (4 methods × 3 models × 2 datasets × 3 seeds)
- [x] Complete experience replay experiment (STGCN: -22.7% incident MAE; STAEformer/STGformer: no effect)
- [x] Complete node identity experiment (weakened MP hurts overall more than helps incidents)
- [x] Complete contrastive learning experiment (incident improvement but 1-10% overall MAE cost)
- [x] Complete coreset K-Medoids sweep (60-80% ratio optimal across 4 models)
- [x] Consolidate Phase A config into single `phase_a_distance_screening.yaml`
- [x] Design severity-aware contrastive pre-training framework (`docs/design.md`)

## 📅 Deadlines
- No hard deadlines currently. Paper submission target TBD.
