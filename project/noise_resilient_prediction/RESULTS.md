# Noise-Resilient Traffic Prediction: Key Results

> One-page summary of what we know, what works, and what doesn't.
> For detailed tables and analysis, see [EXPERIMENTS.md](EXPERIMENTS.md).

## The Problem

Traffic prediction models are highly vulnerable to noisy sensors.
At 30% corruption rate, STAEformer degrades **+42%** and STGCN **+29%** in MAE.
Noise also **spills over** to healthy nodes via spatial attention/convolution (+20% on uncorrupted nodes in STAEformer).

## What Works

### 1. Noise Augmentation (Training-Time)
- Inject random noise during training; no extra parameters
- **Universally effective**: 52-60% degradation reduction on both SAN_BERNARDINO and CONTRA_COSTA
- Eliminates spillover almost completely (~0%)
- Small clean MAE penalty (+0.3-0.4)

### 2. Denoising Encoder v2 (Pre-trained, Frozen, Residual)
- Pre-trained to reconstruct clean signals from noisy inputs; residual architecture
- **Robustness is dataset-dependent**: 47% reduction on SB, ~0% on CC (encoder alone)
- **Clean MAE universally improved**: SB 12.05 (-0.08), CC 11.98 (-0.41)
- Effectiveness proportional to spatial coupling strength (adjacency weight)

### 3. Combination (v2 Encoder + Noise Augmentation)
- v2 encoder (residual + hidden=64) + noise augmentation = best overall
- **70% reduction on SB** (avg deg 41.8% -> 12.6%), **60% on CC** (39.3% -> 15.8%)
- Model-agnostic: works on both STAEformer and STGCN
- Cross-dataset validated on SAN_BERNARDINO and CONTRA_COSTA

## What Doesn't Work

| Approach | Why It Failed |
|---|---|
| Contrastive pre-training | Encoder representation adds no information beyond raw features |
| Cross-variable pre-training | Same — STAEformer already learns from raw 5ch |
| MICL (missing-invariant CL) | v1-v3 all failed; mask-aware loss doesn't improve |
| Reliability estimation | Higher clean MAE, no robustness gain |
| Plug & play encoder (no fine-tuning) | Severe clean MAE penalty, minimal robustness gain |
| v1 encoder + noise aug combo | v1 (no residual) distorts clean data -> combo worse than aug alone |

## Critical Design Choice: Residual Connection

| | v1 (no residual) | v2 (residual) |
|---|---|---|
| Output | `f(noisy_input)` | `input + correction(input)` |
| Clean data behavior | Distorts (~37% occ error) | Preserves (correction -> 0) |
| Combo with noise aug | **Worse** than aug alone | **Best** of all methods |
| Pretrain val loss | 0.18 | 0.05 |

## Cross-Dataset: SAN_BERNARDINO vs CONTRA_COSTA

| | SB adj weight | CC adj weight |
|---|---|---|
| Mean edge weight | **0.74** | **0.40** |
| Edges > 0.5 | 81.5% | 35.1% |

| Strategy | SB Reduction | CC Reduction |
|---|---|---|
| Denoising v2 only | **-47%** | ~0% |
| Noisy training only | **-60%** | **-52%** |
| v2 Encoder + Noisy | **-70%** | **-60%** |

**Takeaway**: Noise augmentation is the robust, dataset-agnostic defense.
Denoising encoder's robustness gain depends on spatial coupling, but v2 residual architecture universally improves clean MAE (CC: 12.39→11.98).

## Practical Recommendation

| Scenario | Recommended Config | SB MAE | SB Deg | CC MAE | CC Deg |
|---|---|---|---|---|---|
| **Max robustness** | Denoising v2 + noise aug | 12.81 | +12.6% | 13.21 | +15.8% |
| **Balance (strong graph)** | Denoising v2 encoder only | 12.05 | +22.0% | 11.98 | +41.3% |
| **Balance (weak graph)** | Noise augmentation only | 12.52 | +16.7% | 12.47 | +18.8% |
| **No modification** | Baseline | 12.13 | +41.8% | 12.39 | +39.3% |

## Files

| What | Where |
|---|---|
| Detailed results | [EXPERIMENTS.md](EXPERIMENTS.md) |
| Experiment history | [experiment_log.md](experiment_log.md) |
| Eval script (SB) | `experiments/eval_noise_vulnerability.py` |
| Eval script (CC) | `experiments/eval_contra_costa.py` |
| Results JSON | `experiments/noise_vulnerability_results/` |
| Dataset comparison | `experiments/compare_datasets.py` |

*Last updated: 2026-02-25*
