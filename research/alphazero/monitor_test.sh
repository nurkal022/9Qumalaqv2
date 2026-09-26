#!/bin/bash
# Monitor test progress

cd "$(dirname "$0")"

LOG_FILE="test_final_results.txt"

if [ ! -f "$LOG_FILE" ]; then
    echo "Test log not found. Is test running?"
    exit 1
fi

echo "=========================================="
echo "AlphaZero Test Monitor"
echo "=========================================="
echo "Press Ctrl+C to exit"
echo ""

while true; do
    clear
    echo "=========================================="
    echo "AlphaZero vs AI Levels - Test Progress"
    echo "=========================================="
    echo ""
    
    # Show last 40 lines
    tail -40 "$LOG_FILE" 2>/dev/null || echo "Waiting for log..."
    
    echo ""
    echo "=========================================="
    echo "Updated: $(date '+%H:%M:%S')"
    echo "=========================================="
    
    # Check if test is complete
    if grep -q "SUMMARY" "$LOG_FILE" 2>/dev/null; then
        echo ""
        echo "✅ Test completed! Showing final results..."
        sleep 2
        break
    fi
    
    sleep 5
done

# Show final summary
echo ""
echo "=========================================="
echo "FINAL RESULTS"
echo "=========================================="
grep -A 50 "SUMMARY" "$LOG_FILE" 2>/dev/null || tail -30 "$LOG_FILE"

