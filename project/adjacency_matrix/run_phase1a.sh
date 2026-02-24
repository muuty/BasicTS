#!/bin/bash
# Phase 1a: Run all adj experiments sequentially on GPU 1
# Usage: nohup bash project/adjacency_matrix/run_phase1a.sh > /tmp/adj_phase1a.log 2>&1 &

set -e
source ~/.conda/etc/profile.d/conda.sh
conda activate basicts
cd /data/pretrainingbasicts

CONFIG_DIR="project/adjacency_matrix/configs"
COMPLETED=0
FAILED=0
TOTAL=$(ls "$CONFIG_DIR"/*.py | wc -l)

echo "=== Phase 1a Adjacency Experiment ==="
echo "Total configs: $TOTAL"
echo "GPU: 1"
echo "Started: $(date)"
echo "=================================="

for config in "$CONFIG_DIR"/*.py; do
    name=$(basename "$config" .py)

    # Skip if test_metrics.json already exists (already completed)
    ckpt_pattern="checkpoints/adj_experiment/*/${name}/"
    found_metrics=$(find $ckpt_pattern -name "test_metrics.json" 2>/dev/null | head -1)
    if [ -n "$found_metrics" ]; then
        echo "[SKIP] $name (already completed)"
        COMPLETED=$((COMPLETED + 1))
        continue
    fi

    echo ""
    echo "[START] $name ($(date '+%H:%M:%S')) [$((COMPLETED + FAILED + 1))/$TOTAL]"
    if python -c "from basicts import launch_training; launch_training('$config', gpus='1')" 2>&1; then
        echo "[DONE] $name ($(date '+%H:%M:%S'))"
        COMPLETED=$((COMPLETED + 1))
    else
        echo "[FAIL] $name ($(date '+%H:%M:%S'))"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=== Phase 1a Complete ==="
echo "Completed: $COMPLETED, Failed: $FAILED, Total: $TOTAL"
echo "Finished: $(date)"
