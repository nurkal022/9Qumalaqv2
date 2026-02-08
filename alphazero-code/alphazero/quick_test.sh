#!/bin/bash
# Quick test of trained model against all AI levels

CHECKPOINT="${1:-checkpoints/model_iter150.pt}"
GAMES="${2:-20}"

echo "=========================================="
echo "Testing AlphaZero Model"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Games per level: $GAMES"
echo ""

cd "$(dirname "$0")"

if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo "Running comprehensive tests..."
echo ""

# Test against all levels
~/miniconda3/envs/togyz-alphazero/bin/python test_alphazero_vs_levels.py \
    --checkpoint "$CHECKPOINT" \
    --games "$GAMES" \
    2>&1 | tee test_results_$(date +%Y%m%d_%H%M%S).txt

echo ""
echo "=========================================="
echo "Testing completed!"
echo "Check test_results_*.txt for details"
echo "=========================================="

