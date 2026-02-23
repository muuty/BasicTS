#!/bin/bash
CONFIG_FILE=$1

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=proxy_metrics
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.log

echo "=== Proxy Metric Computation ==="
echo "Config: $CONFIG_FILE"
echo "Start: \$(date)"
echo ""

eval "\$(conda shell.bash hook)"
conda activate cuda

python experiments/compute_proxy_metrics.py --cfg "$CONFIG_FILE" --gpus 0

echo ""
echo "End: \$(date)"
EOF
