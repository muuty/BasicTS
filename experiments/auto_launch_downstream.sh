#!/bin/bash
# Auto-launch downstream experiments sequentially after A7 finishes
# Usage: nohup bash experiments/auto_launch_downstream.sh > /tmp/auto_launch.log 2>&1 &

set -e
cd /data/pretrainingbasicts
source ~/.conda/etc/profile.d/conda.sh && conda activate basicts

A7_PID=1591364

echo "[$(date)] Waiting for A7 downstream (PID=$A7_PID) to finish..."
while kill -0 $A7_PID 2>/dev/null; do
    sleep 60
done
echo "[$(date)] A7 downstream finished. Waiting 30s for GPU memory to free..."
sleep 30

# Launch A2 downstream
echo "[$(date)] Launching A2 downstream..."
python -c "from basicts import launch_training; launch_training('baselines/STAEformer/SAN_BERNARDINO_5ch_ablation_a2_mlp.py', gpus='1')" > /tmp/ablation_a2_downstream_v2.log 2>&1
echo "[$(date)] A2 downstream finished."

# Launch A3 downstream
echo "[$(date)] Launching A3 downstream..."
python -c "from basicts import launch_training; launch_training('baselines/STAEformer/SAN_BERNARDINO_5ch_ablation_a3_temporal_only.py', gpus='1')" > /tmp/ablation_a3_downstream_v2.log 2>&1
echo "[$(date)] A3 downstream finished."

# Launch B2 downstream
echo "[$(date)] Launching B2 downstream..."
python -c "from basicts import launch_training; launch_training('baselines/STAEformer/SAN_BERNARDINO_5ch_ablation_b2_hidden32.py', gpus='1')" > /tmp/ablation_b2_downstream_v2.log 2>&1
echo "[$(date)] B2 downstream finished."

echo "[$(date)] All ablation downstream experiments complete!"
