#!/bin/bash
#SBATCH --job-name=proxy_families_raw
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --account=a_civil_eng
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%j.log
source ~/.bashrc
conda activate cuda
cd /home/uqtyu7/github/BasicTS
python -u scripts/analysis/compute_proxy_families_raw.py
