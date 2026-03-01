# Tasks — Incident-Aware Traffic Forecasting

## 🔥 In Progress
- [ ] Phase C method comparison: 400/570 runs submitted, ~116 completed, remaining 43 batches pending QOS limit
- [ ] Phase C ratio sweep: 180 runs ready to submit (after method comparison has queue room)

## 📋 Backlog
- [ ] Resubmit remaining 43 batches of phase_c_method_comparison
- [ ] Submit phase_c_ratio_sweep.yaml (180 runs)
- [ ] Collect Phase C results: `get_results.py --config phase_c_*.yaml`
- [ ] Begin T-ITS paper writing (3 contributions: theory, k-medoids justification, proxy metric)
- [ ] Implement severity-aware pre-training Stage 1 (contrastive, design in `docs/design.md`)
- [ ] Integrate severity-aware encoder with STGCN and STAEformer backbones (Stage 2)

## 🚧 Blocked
- [ ] Phase C full analysis — blocked by: training runs in progress (~750 total)

## ✅ Recently Done
- [x] Fix STID channel index mismatch for multi-feature datasets
- [x] Fix IncidentAwareRunner test hang (double forward pass + .item() sync bottleneck)
- [x] Create Phase C experiment configs (euclidean, 5 models, 2 datasets, 750 runs)
- [x] Submit Phase C method comparison (400/570 runs)
- [x] Create CONTRA_COSTA model configs (STGCN, AGCRN, DCRNN, STID, STAEformer)
- [x] T-ITS paper readiness assessment — confirmed 3 contributions sufficient
- [x] Phase A + Phase B training completed (SAN_BERNARDINO, STGCN/AGCRN/DCRNN)
- [x] Theoretical bound analysis + quantization cost proxy metric development
- [x] Complete all 110 coreset index files for SAN_BERNARDINO
- [x] Implement Facility Location selection method (greedy submodular, (1-1/e) guarantee)
- [x] Temporal distribution analysis (DOW/TOD KL divergence across all selections)
- [x] Submit Phase A training: 220 runs in 44 batch jobs (batch-size=5)
- [x] Move Discord tokens to env vars + git history rewrite
- [x] Reorganize codebase into 7 logical git commits (834aeae → aec777b)
- [x] Set up project documentation (CLAUDE.md, PROJECT_BACKGROUND.md, design.md)
- [x] Pre-compute 110/110 coreset index files for SAN_BERNARDINO
- [x] Complete adjacency matrix experiment (4 methods × 3 models × 2 datasets × 3 seeds)
- [x] Complete experience replay experiment (STGCN: -22.7% incident MAE; STAEformer/STGformer: no effect)
- [x] Complete node identity experiment (weakened MP hurts overall more than helps incidents)
- [x] Complete contrastive learning experiment (incident improvement but 1-10% overall MAE cost)
- [x] Complete coreset K-Medoids sweep (60-80% ratio optimal across 4 models)
- [x] Consolidate Phase A config into single `phase_a_distance_screening.yaml`
- [x] Design severity-aware contrastive pre-training framework (`docs/design.md`)

## 📅 Deadlines
- T-ITS paper submission: target after Phase C experiments complete
