#!/bin/bash
# Re-run evaluation for all completed experiments to update robustness metrics
# Usage: ./experiments/rerun_all_evaluations.sh [GPU_ID]

GPU_ID=${1:-0}
PROJECT_DIR="/data/pretrainingbasicts"
CONDA_ENV="basicts"
CONDA_PATH="/home/cau_lab/.conda"

cd "$PROJECT_DIR"

# Activate conda environment
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "=== Re-running evaluations to update robustness metrics ==="
echo "GPU: $GPU_ID"
echo "Python: $(which python)"
echo ""

# Find all completed experiments with best checkpoints
find checkpoints -name "*best*.pt" | while read CKPT; do
    CKPT_DIR=$(dirname "$CKPT")

    # Find config file in the same directory
    CONFIG=$(find "$CKPT_DIR" -maxdepth 1 -name "*.py" | head -1)

    if [ -z "$CONFIG" ]; then
        echo "Skipping $CKPT_DIR - no config file found"
        continue
    fi

    # Check if robustness metrics already exist
    METRICS_FILE="$CKPT_DIR/test_metrics.json"
    if [ -f "$METRICS_FILE" ]; then
        if grep -q '"robustness"' "$METRICS_FILE"; then
            echo "Skipping $CKPT_DIR - already has robustness metrics"
            continue
        fi
    fi

    echo ""
    echo "=== Evaluating: $CKPT_DIR ==="
    echo "Config: $CONFIG"
    echo "Checkpoint: $CKPT"

    python experiments/evaluate.py -cfg "$CONFIG" -ckpt "$CKPT" -g "$GPU_ID"

    echo "Done: $CKPT_DIR"
done

echo ""
echo "=== All evaluations completed ==="
