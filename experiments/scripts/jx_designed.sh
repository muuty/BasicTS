#!/bin/bash
#SBATCH --job-name=jx_designed
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --account=a_civil_eng
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/uqtyu7/github/BasicTS/logs/jx_designed_%j.out
source ~/.bashrc
conda activate cuda
cd /home/uqtyu7/github/BasicTS
python3 scripts/analysis/fd_sensor_resolved.py \
  --methods random fd_hyb0604 fd_front10 fd_front3g \
  --ratios 0.1 --seeds 42 --out-suffix _designed
