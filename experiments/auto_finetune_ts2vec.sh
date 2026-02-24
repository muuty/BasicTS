#!/bin/bash
# Auto-finetune script: Monitors TS2Vec pretrain and starts finetune when done

source ~/.conda/etc/profile.d/conda.sh
conda activate basicts

PRETRAIN_LOG="/tmp/ts2vec_pretrain.log"
FINETUNE_CONFIG="baselines/TS2Vec/SAN_BERNARDINO/STAEformer_finetune.py"
FINETUNE_LOG="/tmp/finetune_ts2vec.log"
GPU="1"

echo "[$(date)] Waiting for TS2Vec pretrain to complete..."

# Wait for pretrain to complete (check for "Loading Checkpoint from" which indicates final eval)
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
