#!/bin/bash
# Overnight ablation experiments — sequential on GPU 1
# Run: nohup bash experiments/run_overnight_ablation.sh > /tmp/overnight_ablation.log 2>&1 &

set -e
source ~/.conda/etc/profile.d/conda.sh
conda activate basicts
cd /data/pretrainingbasicts

CONFIGS=(
  "baselines/STAEformer/SAN_BERNARDINO_5ch_input_corrector_only.py"
  "baselines/STAEformer/SAN_BERNARDINO_5ch_input_corrector_hybrid_noisy.py"
  "baselines/STAEformer/SAN_BERNARDINO_5ch_input_corrector_no_reliability.py"
  "baselines/STAEformer/SAN_BERNARDINO_5ch_input_corrector_no_crossattn.py"
  "baselines/STAEformer/SAN_BERNARDINO_5ch_input_corrector_unfrozen_clean.py"
  "baselines/STAEformer/SAN_BERNARDINO_5ch_input_corrector_v2_clean.py"
)

echo "=== Overnight Ablation: ${#CONFIGS[@]} experiments ==="
echo "Start: $(date)"

for i in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$i]}"
  name=$(basename "$cfg" .py)
  echo ""
  echo "=== [$((i+1))/${#CONFIGS[@]}] $name ==="
  echo "Start: $(date)"

  python -c "
from basicts import launch_training
launch_training('$cfg', gpus='1')
"

  echo "Done: $(date)"
done

echo ""
echo "=== ALL DONE ==="
echo "End: $(date)"
