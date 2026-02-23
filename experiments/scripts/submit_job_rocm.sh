#!/bin/bash
CONFIG_FILE=$1
EXP_NAME=$2
RUN_IDX=$3

# Create run name for job identification
if [ -n "$RUN_IDX" ]; then
    RUN_NAME="${EXP_NAME}_run${RUN_IDX}"
else
    RUN_NAME="$EXP_NAME"
fi

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$RUN_NAME
#SBATCH --partition=gpu_rocm
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=logs/%j.log

echo "Run: $RUN_NAME"
echo "Config: $CONFIG_FILE"
echo "Exp: $EXP_NAME"
if [ -n "$RUN_IDX" ]; then
    echo "Run Index: $RUN_IDX"
fi
echo "Start: \$(date)"

eval "\$(conda shell.bash hook)"
conda activate rocm

if [ -n "$RUN_IDX" ]; then
    python experiments/train.py --cfg=$CONFIG_FILE --exp=$EXP_NAME --run=$RUN_IDX
else
    python experiments/train.py --cfg=$CONFIG_FILE --exp=$EXP_NAME
fi

echo "End: \$(date)"
EOF