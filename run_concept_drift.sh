#!/bin/bash
source ~/.conda/etc/profile.d/conda.sh
conda activate basicts
cd /data/pretrainingbasicts

configs=(
    "baselines/STAEformer/concept_drift_2022.py"
    "baselines/STAEformer/concept_drift_2023.py"
    "baselines/STAEformer/concept_drift_2024.py"
    "baselines/STAEformer/concept_drift_3Y.py"
)

for cfg in "${configs[@]}"; do
    echo "========== Starting: $cfg =========="
    python -c "from basicts import launch_training; launch_training('$cfg', gpus='1')"
    echo "========== Finished: $cfg =========="
done
echo "ALL EXPERIMENTS DONE"
