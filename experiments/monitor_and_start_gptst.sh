#!/bin/bash
# Monitor STMAE finetune and start GPTST finetune when it completes

STMAE_PID=2415041
GPTST_CONFIG="baselines/GPTST/SAN_BERNARDINO_finetune.py"

echo "Monitoring STMAE finetune (PID: $STMAE_PID)..."

while kill -0 $STMAE_PID 2>/dev/null; do
    sleep 60
    echo "$(date): STMAE still running..."
done

echo "$(date): STMAE finetune completed!"
echo "Starting GPTST finetune..."

cd /data/pretrainingbasicts
source ~/.conda/etc/profile.d/conda.sh
conda activate basicts

python -c "from basicts import launch_training; launch_training('$GPTST_CONFIG', gpus='0')" > /tmp/gptst_finetune.log 2>&1

echo "GPTST finetune started. Check /tmp/gptst_finetune.log for progress."
