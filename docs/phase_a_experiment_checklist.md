# Phase A: Distance Screening — Experiment Checklist

## Experiment Design

**Goal**: Find the best distance function for coreset selection in spatio-temporal traffic forecasting.

**Grid**:
- Representation (4): raw, temporal, spatial, combined
- Distance metric (2): L2 (euclidean + RBF for graph_cut), cosine
- Selection method (6): k_center, k_medoids, graph_cut, random, stride, recent
- Ratio (2): 0.3, 0.7
- Seed (2): 42, 123
- Model (2): STGCN, STAEformer
- Dataset: SAN_BERNARDINO (893 nodes, 3-month data)

**Total**: 192 distance-dependent + 24 baseline + 4 full = **220 training runs**

---

## 1. Coreset Selection (Index Files)

Index files: `coreset_indices/SAN_BERNARDINO/{method}_{distance}_{ratio}_seed{seed}.json`

### Selection count breakdown

| Group | Formula | Files |
|-------|---------|-------|
| Distance-dependent | 3 methods × 8 distances × 2 ratios × 2 seeds | **96** |
| Baseline (distance-independent) | 3 methods × 1 distance × 2 ratios × 2 seeds | **12** |
| Full data (ratio=1.0) | 1 method × 1 ratio × 2 seeds | **2** |
| **Total** | | **110** |

### L2 Pipeline (euclidean + RBF)
| Method | euclidean | temporal | spatial | combined |
|--------|-----------|----------|---------|----------|
| k_center | done | done | done | done |
| k_medoids | done | done | done | done |
| graph_cut | done | done | done | done |
| random | done | — | — | — |
| stride | done | — | — | — |
| recent | done | — | — | — |
| full (1.0) | done | — | — | — |

### Cosine Pipeline
| Method | cosine_raw | cosine_temporal | cosine_spatial | cosine_combined |
|--------|------------|-----------------|----------------|-----------------|
| k_center | done | done | done | done |
| k_medoids | running | running | running | running |
| graph_cut | pending | pending | pending | pending |

**Selection status**: 88/110 done, Job 21488604 RUNNING

---

## 2. Training Runs

### Training count breakdown

Selection files are model-independent. Each selection file is used by both STGCN and STAEformer.

| Group | Formula | Runs |
|-------|---------|------|
| Distance-dependent | 96 selections × 2 models | **192** |
| Baseline | 12 selections × 2 models | **24** |
| Full data | 2 selections × 2 models | **4** |
| **Total** | | **220** |

Results stored under `checkpoints/{experiment_name}/`.
Config: `experiments/config/phase_a_distance_screening.yaml` (consolidated)

All training runs: pending (waiting for selection to complete)

---

## 3. Job Summary

| Job | Description | Status |
|-----|-------------|--------|
| Selection (Job 21488604) | 110 index files | **running** (88/110) |
| Training (batch-size=5) | 220 runs = 44 SLURM jobs | **pending** (after selection) |

**Command to submit training:**
```bash
python experiments/run_experiments.py \
  --cfg experiments/config/phase_a_distance_screening.yaml \
  --arch=cuda --batch-size 5
```

---

## 4. Results Collection

```bash
conda activate cuda
python experiments/get_results.py \
  --config experiments/config/phase_a_distance_screening.yaml \
  --metrics MAE RMSE
```

---

## 5. Key Analysis Questions

1. **L2 vs Cosine**: Does cosine distance outperform L2 across representations?
2. **Representation ranking**: raw vs temporal vs spatial vs combined — consistent across metrics?
3. **Method × distance interaction**: Does the best distance depend on the selection method?
4. **Model consistency**: Do STGCN and STAEformer agree on the best distance?
5. **Ratio effect**: Is the distance gap larger at ratio=0.3 (less data)?

---

## 6. Notes

- STGformer was replaced by STAEformer (more numerically stable)
- GraphCut was reimplemented to match DeepCore's standard formulation (modular sum, not FL)
- Combined distance: temporal and spatial computed separately, normalized to [0,1], then averaged (avoids dimension imbalance)
- Cosine pipeline for graph_cut: cosine similarity computed directly on features (no RBF conversion)
- STGCN uses FORWARD_FEATURES=[0,1,2], STAEformer uses [0,3,4]
- Training has implicit skip logic via EasyTorch checkpoint resume (no explicit skip needed)
- Experiment configs consolidated into single `phase_a_distance_screening.yaml`
