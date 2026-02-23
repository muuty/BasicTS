#!/bin/bash
CONFIG_FILE=$1

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=coreset_sel
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%j.log

echo "=== Offline Coreset Selection ==="
echo "Config: $CONFIG_FILE"
echo "Start: \$(date)"
echo ""

eval "\$(conda shell.bash hook)"
conda activate cuda

python experiments/select_coreset.py --cfg "$CONFIG_FILE" --gpus 0

echo ""
echo "End: \$(date)"
EOF
