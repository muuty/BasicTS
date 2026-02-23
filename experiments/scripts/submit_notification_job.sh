#!/bin/bash
# Submit a notification job that runs after all experiment jobs complete.
#
# Usage:
#   bash submit_notification_job.sh <exp_name> <job_ids> <start_time> <conda_env>
#
# Arguments:
#   exp_name   - Experiment name for the notification
#   job_ids    - Comma-separated list of job IDs to wait for
#   start_time - Unix timestamp when experiments started
#   conda_env  - Conda environment to use (cuda or rocm)

EXP_NAME=$1
JOB_IDS=$2
START_TIME=$3
CONDA_ENV=${4:-cuda}

# Convert comma-separated job IDs to colon-separated for SLURM dependency
DEPENDENCY_STR=$(echo "$JOB_IDS" | tr ',' ':')

sbatch --dependency=afterany:${DEPENDENCY_STR} <<EOF
#!/bin/bash
#SBATCH --job-name=${EXP_NAME}_notify
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/notify_%j.log

echo "Notification job started: \$(date)"
echo "Experiment: $EXP_NAME"
echo "Monitored jobs: $JOB_IDS"

eval "\$(conda shell.bash hook)"
conda activate $CONDA_ENV

python experiments/scripts/send_notification.py \\
    --exp-name="$EXP_NAME" \\
    --job-ids="$JOB_IDS" \\
    --start-time=$START_TIME

echo "Notification job completed: \$(date)"
EOF
