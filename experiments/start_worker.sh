#!/bin/bash
# Start RQ worker for a specific GPU
# Usage: ./experiments/start_worker.sh [GPU_ID]

GPU_ID=${1:-0}
cd /data/pretrainingbasicts

source /home/cau_lab/.conda/etc/profile.d/conda.sh
conda activate basicts

# GPU is passed via gpus parameter in launch_training, not CUDA_VISIBLE_DEVICES
echo "Worker GPU $GPU_ID"
python -m rq.cli worker gpu$GPU_ID --path /data/pretrainingbasicts/experiments
