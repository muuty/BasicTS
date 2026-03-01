#!/bin/bash
# Monitor training completion and send Discord notification.
# Usage: bash experiments/notify_on_complete.sh <log_file> [poll_interval_sec]

LOG_FILE="$1"
POLL=${2:-30}

# Load webhook URL from .env
source /data/pretrainingbasicts/.env

if [ -z "$DISCORD_WEBHOOK_URL" ]; then
    echo "ERROR: DISCORD_WEBHOOK_URL not set in .env"
    exit 1
fi

echo "Monitoring: $LOG_FILE (poll every ${POLL}s)"

while true; do
    if grep -q "Training finished" "$LOG_FILE" 2>/dev/null || \
       grep -q "Epoch 30 / 30" "$LOG_FILE" 2>/dev/null && \
       grep -c "Result <test>" "$LOG_FILE" 2>/dev/null | grep -q "^3[0-9]"; then

        # Extract final results
        BEST_VAL=$(grep "best_val_MAE.pt saved" "$LOG_FILE" | tail -1)
        FINAL_TEST=$(grep "Result <test>:" "$LOG_FILE" | tail -1)
        FINAL_TRAIN=$(grep "Result <train>:" "$LOG_FILE" | tail -1)

        # Check for test_metrics.json
        CKPT_DIR=$(dirname "$LOG_FILE")
        METRICS_FILE=$(find "$CKPT_DIR" -name "test_metrics.json" 2>/dev/null | head -1)
        if [ -n "$METRICS_FILE" ]; then
            OVERALL_MAE=$(python3 -c "import json; d=json.load(open('$METRICS_FILE')); print(f\"MAE={d['overall']['MAE']:.4f}\")" 2>/dev/null || echo "N/A")
        else
            OVERALL_MAE="test_metrics.json not found yet"
        fi

        MSG="**🔬 STAEformerUncertainty Exp A 학습 완료!**\n\n"
        MSG+="**Final Test**: ${FINAL_TEST}\n"
        MSG+="**Overall**: ${OVERALL_MAE}\n"
        MSG+="**Last Best**: ${BEST_VAL}\n\n"
        MSG+="eval_uncertainty.py 실행 준비 완료."

        curl -s -H "Content-Type: application/json" \
             -d "{\"content\": \"${MSG}\"}" \
             "$DISCORD_WEBHOOK_URL"

        echo ""
        echo "Discord notification sent!"
        exit 0
    fi

    sleep "$POLL"
done
