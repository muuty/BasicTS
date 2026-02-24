#!/bin/bash
# Auto-launch reliability downstream after pretrain completes
# Usage: nohup bash experiments/auto_launch_reliability_downstream.sh > /tmp/auto_launch_reliability.log 2>&1 &

set -e
cd /data/pretrainingbasicts
source ~/.conda/etc/profile.d/conda.sh && conda activate basicts

RELIABILITY_PRETRAIN_PID=2191732

wait_for_gpu_memory() {
    local needed=$1
    while true; do
        local free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 2>/dev/null)
        if [ "$free" -ge "$needed" ] 2>/dev/null; then
            echo "[$(date)] GPU 1 has ${free}MB free (need ${needed}MB). Proceeding."
            return
        fi
        echo "[$(date)] GPU 1 has ${free}MB free, need ${needed}MB. Waiting 60s..."
        sleep 60
    done
}

# Wait for reliability pretrain to finish
echo "[$(date)] Waiting for reliability pretrain (PID=$RELIABILITY_PRETRAIN_PID)..."
while kill -0 $RELIABILITY_PRETRAIN_PID 2>/dev/null; do sleep 60; done
echo "[$(date)] Reliability pretrain finished."

# Wait for enough GPU memory (~35GB for STAEformer downstream)
echo "[$(date)] Waiting for GPU memory (35GB)..."
wait_for_gpu_memory 35000

# Launch reliability downstream
echo "[$(date)] Launching reliability downstream..."
python -c "from basicts import launch_training; launch_training('baselines/STAEformer/SAN_BERNARDINO_5ch_denoising_reliability.py', gpus='1')" > /tmp/reliability_downstream.log 2>&1
echo "[$(date)] Reliability downstream finished."
