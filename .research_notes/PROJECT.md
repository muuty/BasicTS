# Incident-Aware Traffic Forecasting

## Overview
Traffic prediction models (STGCN, STAEformer, etc.) achieve good overall MAE but fail badly during traffic incidents — when a node's flow drops to zero, GCN message passing from neighboring normal nodes suppresses the anomaly signal. This project develops methods to improve incident-specific prediction while preserving overall accuracy.

## Goals
- [ ] Complete Phase A distance screening and identify optimal coreset selection strategy (this month)
- [ ] Implement and evaluate severity-aware contrastive pre-training (this quarter)
- [ ] Submit paper demonstrating incident-aware forecasting with coreset selection (long-term)

## Background
Traffic forecasting is well-studied for average-case performance, but worst-case (incident) performance is neglected. We've systematically tested multiple approaches:
1. **Adjacency matrix variations** — limited impact on incident performance
2. **Weakened message passing** — hurts overall MAE more than it helps incidents
3. **Experience replay** — effective for STGCN (-22.7% incident MAE) but not transformer models
4. **Contrastive learning** — improves incident nodes but degrades overall MAE 1-10%
5. **Coreset selection** (current focus) — 60-80% subset matches or beats full data; now screening distance functions

Key insight: the problem is fundamentally about representation — models can't distinguish incident from normal states in their learned embeddings.

## Key Resources
- **Repo:** `/home/uqtyu7/github/BasicTS` (branch: `feature/incident-aware`)
- **Dataset:** SAN_BERNARDINO (893 nodes, XTraffic California, 3-month/5-min interval), CONTRA_COSTA (773 nodes)
- **Key papers:** CLEAR (TKDE 2025), STD-MAE (IJCAI 2024), SCPT (DMKD 2023), DeepCore (Guo et al. 2022)
- **Compute:** UQ HPC — `gpu_cuda` partition, 1 GPU, 32GB memory, 4 concurrent job limit

## Current Status
Phase A distance screening is ~60% complete (135/220 training runs), bottlenecked by SLURM queue limits. Early results show K-Center + combined distance at 70% ratio is most stable. Severity-aware pre-training design is complete but not yet integrated.

## Collaborators / Advisors
<!-- Add advisor/collaborator info here -->
