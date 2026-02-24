#!/bin/bash
# Run comprehensive noise vulnerability evaluation on all models
# Usage: bash experiments/run_full_noise_eval.sh [gpu_id]
# Evaluates all models with all 16 noise configs

set -e
cd /data/pretrainingbasicts
source ~/.conda/etc/profile.d/conda.sh && conda activate basicts

GPU=${1:-1}
LOGDIR=/tmp/noise_eval_logs
mkdir -p $LOGDIR

echo "[$(date)] Starting comprehensive noise vulnerability evaluation on GPU $GPU"

# Phase 1: Re-evaluate existing models with full 16 noise configs
echo ""
echo "=== Phase 1: Existing models (full noise configs) ==="
EXISTING_MODELS=(
    staeformer_5ch
    stgcn_1ch
    staeformer_5ch_denoising
    stgcn_denoising
    staeformer_5ch_noisy
    stgcn_noisy
    staeformer_5ch_denoising_noisy
    stgcn_denoising_noisy
    staeformer_5ch_denoising_v2_noisy
    stgcn_denoising_v2_noisy
)
for model in "${EXISTING_MODELS[@]}"; do
    echo "[$(date)] Evaluating $model..."
    python experiments/eval_noise_vulnerability.py --gpu $GPU --models $model > $LOGDIR/${model}.log 2>&1
    echo "[$(date)]   Done: $model"
done

# Phase 2: New channel baseline models
echo ""
echo "=== Phase 2: Channel baselines ==="
CHANNEL_MODELS=(
    staeformer_1ch
    stgcn_3ch
)
for model in "${CHANNEL_MODELS[@]}"; do
    echo "[$(date)] Evaluating $model..."
    python experiments/eval_noise_vulnerability.py --gpu $GPU --models $model > $LOGDIR/${model}.log 2>&1
    echo "[$(date)]   Done: $model"
done

# Phase 3: Ablation models (require downstream training to be complete)
echo ""
echo "=== Phase 3: Ablation models ==="
ABLATION_MODELS=(
    ablation_a2_mlp
    ablation_a3_temporal_only
    ablation_a7_no_residual
    ablation_b2_hidden32
)
for model in "${ABLATION_MODELS[@]}"; do
    echo "[$(date)] Evaluating $model..."
    python experiments/eval_noise_vulnerability.py --gpu $GPU --models $model > $LOGDIR/${model}.log 2>&1 || echo "  FAILED: $model (checkpoint may not exist yet)"
    echo "[$(date)]   Done: $model"
done

# Phase 4: Cross-noise generalization models
echo ""
echo "=== Phase 4: Cross-noise generalization ==="
CROSSNOISE_MODELS=(
    staeformer_5ch_denoising_common_noise
    staeformer_5ch_denoising_structural_noise
)
for model in "${CROSSNOISE_MODELS[@]}"; do
    echo "[$(date)] Evaluating $model..."
    python experiments/eval_noise_vulnerability.py --gpu $GPU --models $model > $LOGDIR/${model}.log 2>&1 || echo "  FAILED: $model (checkpoint may not exist yet)"
    echo "[$(date)]   Done: $model"
done

echo ""
echo "[$(date)] All evaluations complete!"
echo "Results saved to: experiments/noise_vulnerability_results/results.json"
