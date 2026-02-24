#!/bin/bash
# Chain training: wait for 2022 (already running), then run 2023 and 2024 on GPU 1
source ~/.conda/etc/profile.d/conda.sh && conda activate basicts

echo "[$(date)] Waiting for 2022 instance norm training to complete..."
# Wait until checkpoint directory appears (training completion indicator)
while true; do
    CKPT_DIR="checkpoints/ConceptDrift_InstanceNorm/SAN_BERNARDINO_2022_Q1_30_12_12"
    if [ -d "$CKPT_DIR" ]; then
        # Check if best model exists in any hash subdir
        BEST=$(find "$CKPT_DIR" -name "STAEformer_best_val_MAE.pt" 2>/dev/null | head -1)
        if [ -n "$BEST" ]; then
            # Also check test_metrics.json to confirm training finished
            METRICS=$(find "$CKPT_DIR" -name "test_metrics.json" 2>/dev/null | head -1)
            if [ -n "$METRICS" ]; then
                echo "[$(date)] 2022 training complete! Found: $METRICS"
                break
            fi
        fi
    fi
    sleep 30
done

echo "[$(date)] Starting 2023 Q1 instance norm training..."
python -c "
from basicts import launch_training
launch_training('baselines/STAEformer/concept_drift_2023_Q1_instance_norm.py', gpus='1')
"
echo "[$(date)] 2023 done!"

echo "[$(date)] Starting 2024 Q1 instance norm training..."
python -c "
from basicts import launch_training
launch_training('baselines/STAEformer/concept_drift_2024_Q1_instance_norm.py', gpus='1')
"
echo "[$(date)] All instance norm training complete!"
