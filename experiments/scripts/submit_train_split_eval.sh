#!/bin/bash
# Evaluate every completed phase_c checkpoint on the full training split.
# Sharded across an array so the grid finishes in a few hours.
SHARDS=${1:-10}

mkdir -p logs
sbatch <<SBATCH
#!/bin/bash
#SBATCH --job-name=train_split_eval
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-$((SHARDS-1))
#SBATCH --output=logs/train_split_eval_%A_%a.log

echo "Shard \$SLURM_ARRAY_TASK_ID of $SHARDS"
echo "Start: \$(date)"

eval "\$(conda shell.bash hook)"
conda activate cuda

python scripts/analysis/eval_on_train_split.py \
    --shard \$SLURM_ARRAY_TASK_ID --shards $SHARDS \
    --out train_split_mae_shard\${SLURM_ARRAY_TASK_ID}.csv --overwrite

echo "End: \$(date)"
SBATCH
