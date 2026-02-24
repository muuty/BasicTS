#!/bin/bash
source ~/.conda/etc/profile.d/conda.sh
conda activate basicts
cd /data/pretrainingbasicts

configs=(
    "baselines/STAEformer/concept_drift_2022_Q1.py"
    "baselines/STAEformer/concept_drift_2023_Q1.py"
    "baselines/STAEformer/concept_drift_2024_Q1.py"
)

for cfg in "${configs[@]}"; do
    echo "========== Starting: $cfg =========="
    python -c "from basicts import launch_training; launch_training('$cfg', gpus='1')"
    echo "========== Finished: $cfg =========="
done
echo "ALL Q1 V3 (UNMASKED MAE) EXPERIMENTS DONE"
