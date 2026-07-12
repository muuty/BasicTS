#!/bin/bash
# Durable (connection-independent) SLURM job for the state-conditional MAE
# bridge: seed-averaged per-state forecasting MAE + follow-up analyses
# (difficulty decomposition, persistence baseline, paired detector test).
# CPU-only (I/O bound on archived test_results.npz); no GPU required.

MAX_SEEDS=${1:-3}

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=fd_state_mae
#SBATCH --account=a_civil_eng
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/fd_state_mae_%j.log

echo "=== state-conditional MAE bridge ==="
echo "Start: \$(date)  max_seeds=$MAX_SEEDS"

eval "\$(conda shell.bash hook)"
conda activate cuda
cd $(pwd)

echo ""
echo "--- [1/2] seed-averaged state-conditional MAE ---"
python -u scripts/analysis/fd_state_conditional_mae.py --max-seeds $MAX_SEEDS

echo ""
echo "--- [2/2] follow-ups (decomposition, persistence, paired test) ---"
python -u scripts/analysis/fd_state_conditional_followups.py

echo ""
echo "End: \$(date)"
EOF
