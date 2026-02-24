#!/bin/bash
# Chain remaining training jobs on GPU 1
set -e
source ~/.conda/etc/profile.d/conda.sh && conda activate basicts
cd /data/pretrainingbasicts

# Wait for 2022 PB to finish
echo "[$(date)] Waiting for 2022 PB training to complete..."
while true; do
    if ls checkpoints/Expanding_PB/SAN_BERNARDINO_2022_Q1_30_12_12/*/test_metrics.json 2>/dev/null | head -1 | grep -q .; then
        echo "[$(date)] 2022 PB DONE"
        break
    fi
    sleep 120
done

sleep 30

# Train 2023 Baseline
echo "[$(date)] Starting 2023 Baseline on GPU 1..."
python -c "from basicts import launch_training; launch_training('baselines/STAEformer/expanding_2023_baseline.py', gpus='1')"
echo "[$(date)] 2023 Baseline DONE"

sleep 10

# Train 2023 PB
echo "[$(date)] Starting 2023 PB on GPU 1..."
python -c "from basicts import launch_training; launch_training('baselines/STAEformer/expanding_2023_pb.py', gpus='1')"
echo "[$(date)] 2023 PB DONE"

echo "[$(date)] ALL TRAINING COMPLETE"
