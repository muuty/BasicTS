#!/bin/bash
# Auto-launch cross-noise and CONTRA_COSTA downstream experiments
# Waits for pretrains to finish, then waits for enough GPU memory
# Usage: nohup bash experiments/auto_launch_crossnoise_downstream.sh > /tmp/auto_launch_crossnoise.log 2>&1 &

set -e
cd /data/pretrainingbasicts
source ~/.conda/etc/profile.d/conda.sh && conda activate basicts

COMMON_PID=1798207
STRUCTURAL_PID=1798414
CONTRACOSTA_PID=1799207
ONECH_PID=1787983

wait_for_gpu_memory() {
    # Wait until at least $1 GB free on GPU 1
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

# Wait for all three pretrains to finish
echo "[$(date)] Waiting for common noise pretrain (PID=$COMMON_PID)..."
while kill -0 $COMMON_PID 2>/dev/null; do sleep 60; done
echo "[$(date)] Common noise pretrain finished."

echo "[$(date)] Waiting for structural noise pretrain (PID=$STRUCTURAL_PID)..."
while kill -0 $STRUCTURAL_PID 2>/dev/null; do sleep 60; done
echo "[$(date)] Structural noise pretrain finished."

# Wait for enough GPU memory (~35GB needed for STAEformer downstream)
echo "[$(date)] Waiting for GPU memory (35GB)..."
wait_for_gpu_memory 35000

# Launch common noise downstream
echo "[$(date)] Launching common noise downstream..."
python -c "from basicts import launch_training; launch_training('baselines/STAEformer/SAN_BERNARDINO_5ch_denoising_common_noise.py', gpus='1')" > /tmp/crossnoise_common_downstream.log 2>&1
echo "[$(date)] Common noise downstream finished."

# Launch structural noise downstream
echo "[$(date)] Launching structural noise downstream..."
python -c "from basicts import launch_training; launch_training('baselines/STAEformer/SAN_BERNARDINO_5ch_denoising_structural_noise.py', gpus='1')" > /tmp/crossnoise_structural_downstream.log 2>&1
echo "[$(date)] Structural noise downstream finished."

# Wait for CONTRA_COSTA pretrain
echo "[$(date)] Waiting for CONTRA_COSTA pretrain (PID=$CONTRACOSTA_PID)..."
while kill -0 $CONTRACOSTA_PID 2>/dev/null; do sleep 60; done
echo "[$(date)] CONTRA_COSTA pretrain finished."

# Launch CONTRA_COSTA baseline
echo "[$(date)] Waiting for GPU memory for CONTRA_COSTA baseline..."
wait_for_gpu_memory 35000
echo "[$(date)] Launching CONTRA_COSTA 5ch baseline..."
python -c "from basicts import launch_training; launch_training('baselines/STAEformer/CONTRA_COSTA_5ch.py', gpus='1')" > /tmp/contra_costa_baseline.log 2>&1
echo "[$(date)] CONTRA_COSTA baseline finished."

# Launch CONTRA_COSTA downstream
echo "[$(date)] Launching CONTRA_COSTA denoising downstream..."
python -c "from basicts import launch_training; launch_training('baselines/STAEformer/CONTRA_COSTA_5ch_denoising.py', gpus='1')" > /tmp/contra_costa_downstream.log 2>&1
echo "[$(date)] CONTRA_COSTA denoising downstream finished."

echo "[$(date)] All cross-noise and CONTRA_COSTA experiments complete!"
