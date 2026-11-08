#!/bin/bash
# Monitor training progress

cd "$(dirname "$0")"

LOG_FILE="training_log.txt"

if [ ! -f "$LOG_FILE" ]; then
    echo "Training log not found. Is training running?"
    exit 1
fi

echo "=========================================="
echo "Training Monitor"
echo "=========================================="
echo "Press Ctrl+C to exit"
echo ""

while true; do
    clear
    echo "=========================================="
    echo "AlphaZero Training Progress"
    echo "=========================================="
    echo ""
    
    # Show last 30 lines
    tail -30 "$LOG_FILE" 2>/dev/null || echo "Waiting for log..."
    
    echo ""
    echo "=========================================="
    echo "Updated: $(date '+%H:%M:%S')"
    echo "=========================================="
    
    sleep 10
done

