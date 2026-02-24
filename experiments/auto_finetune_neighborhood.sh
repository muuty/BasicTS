#!/bin/bash
# Auto-finetune script: Monitors pretrain and starts finetune when done

source ~/.conda/etc/profile.d/conda.sh
conda activate basicts

PRETRAIN_LOG="/tmp/pretrain_neighborhood_mask.log"
FINETUNE_CONFIG="baselines/ContextContrastive/SAN_BERNARDINO/STAEformer_finetune_neighborhood_mask.py"
FINETUNE_LOG="/tmp/finetune_neighborhood_mask.log"
GPU="0"

echo "[$(date)] Waiting for Neighborhood Masking pretrain to complete..."

# Wait for pretrain to complete (check for "Epoch 30 / 30" followed by test results)
while true; do
    if grep -q "Loading Checkpoint from" "$PRETRAIN_LOG" 2>/dev/null; then
        # Check if final test was run (indicates completion)
        if tail -50 "$PRETRAIN_LOG" | grep -q "Result <test>.*test/MAE"; then
            echo "[$(date)] Pretrain completed! Starting fine-tuning..."
            break
        fi
    fi
    sleep 60
done

# Small delay to ensure checkpoint is fully saved
sleep 10

# Start fine-tuning
echo "[$(date)] Launching fine-tuning on GPU $GPU..."
python -c "from basicts import launch_training; launch_training('$FINETUNE_CONFIG', gpus='$GPU')" > "$FINETUNE_LOG" 2>&1

echo "[$(date)] Fine-tuning completed!"
echo "Results saved to: $FINETUNE_LOG"
