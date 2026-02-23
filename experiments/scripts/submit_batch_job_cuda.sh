#!/bin/bash
# Submit a batch of training configs as a single SLURM job.
# Usage: bash submit_batch_job_cuda.sh <batch_file> <exp_name> <run_idx>
# batch_file: text file with one config path per line

BATCH_FILE=$1
EXP_NAME=$2
RUN_IDX=$3
BATCH_ID=$(basename "$BATCH_FILE" .txt)

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${EXP_NAME}_${BATCH_ID}
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=logs/%j.log

echo "Batch: $BATCH_FILE"
echo "Exp: $EXP_NAME"
echo "Start: \$(date)"
echo ""

eval "\$(conda shell.bash hook)"
conda activate cuda

TOTAL=\$(wc -l < "$BATCH_FILE")
CURRENT=0

while IFS= read -r CONFIG_FILE; do
    CURRENT=\$((CURRENT + 1))
    echo "=== [\${CURRENT}/\${TOTAL}] \$(basename \$CONFIG_FILE) ==="
    echo "Start: \$(date)"

    if [ -n "$RUN_IDX" ]; then
        python experiments/train.py --cfg="\$CONFIG_FILE" --exp="$EXP_NAME" --run="$RUN_IDX"
    else
        python experiments/train.py --cfg="\$CONFIG_FILE" --exp="$EXP_NAME"
    fi

    echo "End: \$(date)"
    echo ""
done < "$BATCH_FILE"

echo "=== All done: \$(date) ==="
EOF
