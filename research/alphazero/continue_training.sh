#!/bin/bash
# Continue AlphaZero training from checkpoint (OPTIMIZED VERSION)

CHECKPOINT="checkpoints/model_iter50.pt"
MODEL_SIZE="medium"
GAMES=200
SIMULATIONS=200
BATCH_GAMES=32  # Play 32 games in parallel
ITERATIONS=100  # Total iterations (will continue from checkpoint)
BATCH_SIZE=1024  # Training batch size

echo "=========================================="
echo "ULTRA-FAST AlphaZero Training"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Model: $MODEL_SIZE"
echo "Games per iteration: $GAMES"
echo "MCTS simulations: $SIMULATIONS"
echo "Batch games (parallel): $BATCH_GAMES"
echo "Training batch size: $BATCH_SIZE"
echo "Target iterations: $ITERATIONS"
echo ""
echo "Optimizations:"
echo "  ✅ Batch MCTS (32 games parallel)"
echo "  ✅ AMP with BF16 (2x speedup)"
echo "  ✅ Optimized GPU utilization"
echo "=========================================="
echo ""

cd "$(dirname "$0")"

~/miniconda3/envs/togyz-alphazero/bin/python train_fast.py \
    --resume "$CHECKPOINT" \
    --model-size "$MODEL_SIZE" \
    --games "$GAMES" \
    --simulations "$SIMULATIONS" \
    --batch-games "$BATCH_GAMES" \
    --iterations "$ITERATIONS" \
    --batch-size "$BATCH_SIZE"

