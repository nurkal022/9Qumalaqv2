#!/bin/bash
# Maximum strength AlphaZero training from scratch
# Optimized for RTX 5080 with all optimizations

MODEL_SIZE="medium"  # Можно использовать "large" для еще большей силы, но медленнее
GAMES=300            # Больше игр = больше разнообразия данных
SIMULATIONS=200       # Баланс между скоростью и качеством
BATCH_GAMES=32        # Параллельные игры (оптимально для RTX 5080)
ITERATIONS=150       # Достаточно для сильной модели
BATCH_SIZE=1024       # Большой batch для лучшей GPU утилизации

echo "=========================================="
echo "MAXIMUM STRENGTH AlphaZero Training"
echo "=========================================="
echo "Starting from SCRATCH"
echo "Model: $MODEL_SIZE"
echo "Games per iteration: $GAMES"
echo "MCTS simulations: $SIMULATIONS"
echo "Batch games (parallel): $BATCH_GAMES"
echo "Training batch size: $BATCH_SIZE"
echo "Total iterations: $ITERATIONS"
echo ""
echo "Expected time: ~12-15 hours"
echo "Expected strength: Champion level"
echo ""
echo "Optimizations enabled:"
echo "  ✅ Batch MCTS (32 games parallel)"
echo "  ✅ AMP with BF16 (2x speedup)"
echo "  ✅ TF32 matrix operations"
echo "  ✅ torch.compile()"
echo "  ✅ Optimized GPU utilization"
echo "=========================================="
echo ""
echo "Starting training in 3 seconds..."
sleep 3

cd "$(dirname "$0")"

~/miniconda3/envs/togyz-alphazero/bin/python train_fast.py \
    --model-size "$MODEL_SIZE" \
    --games "$GAMES" \
    --simulations "$SIMULATIONS" \
    --batch-games "$BATCH_GAMES" \
    --iterations "$ITERATIONS" \
    --batch-size "$BATCH_SIZE" \
    2>&1 | tee training_log.txt

echo ""
echo "=========================================="
echo "Training completed!"
echo "Check checkpoints/ for saved models"
echo "=========================================="

