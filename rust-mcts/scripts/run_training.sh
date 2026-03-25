#!/bin/bash
# AlphaZero training loop: Rust MCTS selfplay → Python train → ONNX export → repeat
set -e

# ── Config ──────────────────────────────────────────────
ITERATIONS=200
GAMES_PER_ITER=100
SIMS=800
WORKERS=20
BATCH_SIZE=256
EPOCHS=5
LR=0.0003
MODEL_SIZE="medium"
EXPERT_DIR="../../game-pars/games"
EXPERT_RATIO=0.3
EVAL_INTERVAL=10
EVAL_GAMES=20
EVAL_ENGINE_TIME=1000

# ── Paths ───────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECKPOINT_DIR="$ROOT_DIR/checkpoints"
MODEL_ONNX="$ROOT_DIR/model.onnx"
MODEL_PT="$ROOT_DIR/model.pt"
REPLAY_BUF="$ROOT_DIR/replay_buffer.bin"
ENGINE_BIN="$ROOT_DIR/../engine/target/release/togyzkumalaq-engine"
LOG_FILE="/tmp/rust_mcts_training.log"

# NVIDIA libs for ONNX Runtime CUDA
export ORT_DYLIB_PATH="/home/nurlykhan/.local/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.24.4"
export LD_LIBRARY_PATH="/home/nurlykhan/.local/lib/python3.12/site-packages/nvidia/cublas/lib:/home/nurlykhan/.local/lib/python3.12/site-packages/nvidia/cudnn/lib:/home/nurlykhan/.local/lib/python3.12/site-packages/nvidia/curand/lib:/home/nurlykhan/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/nurlykhan/.local/lib/python3.12/site-packages/nvidia/cufft/lib:/home/nurlykhan/.local/lib/python3.12/site-packages/nvidia/cusparse/lib:/home/nurlykhan/.local/lib/python3.12/site-packages/onnxruntime/capi:$LD_LIBRARY_PATH"

mkdir -p "$CHECKPOINT_DIR"
cd "$ROOT_DIR"

# ── Resume support ──────────────────────────────────────
START_ITER=1
if [ -f "$CHECKPOINT_DIR/iteration.txt" ]; then
    START_ITER=$(cat "$CHECKPOINT_DIR/iteration.txt")
    echo "Resuming from iteration $START_ITER"
fi

# Check initial model exists
if [ ! -f "$MODEL_ONNX" ]; then
    echo "ERROR: No model.onnx found. Run export_onnx.py first."
    exit 1
fi

echo "============================================================"
echo "Rust MCTS AlphaZero Training"
echo "============================================================"
echo "Iterations: $START_ITER → $ITERATIONS"
echo "Games/iter: $GAMES_PER_ITER, Sims: $SIMS, Workers: $WORKERS"
echo "Epochs: $EPOCHS, LR: $LR, Expert ratio: $EXPERT_RATIO"
echo "Eval: every $EVAL_INTERVAL iters, $EVAL_GAMES games vs Gen7"
echo "Log: $LOG_FILE"
echo "============================================================"
echo ""

BEST_WINRATE=0

for ITER in $(seq $START_ITER $ITERATIONS); do
    ITER_START=$(date +%s)
    echo "============================================================"
    echo "Iteration $ITER/$ITERATIONS  ($(date '+%H:%M:%S'))"
    echo "============================================================"

    # ── Phase 1: Rust MCTS Self-Play ────────────────────
    echo "[Self-Play: $GAMES_PER_ITER games, $SIMS sims]"
    SP_START=$(date +%s)

    ITER_BUF="$ROOT_DIR/replay_iter${ITER}.bin"
    "$ROOT_DIR/target/release/rust-mcts" \
        --model "$MODEL_ONNX" \
        --games $GAMES_PER_ITER \
        --sims $SIMS \
        --workers $WORKERS \
        --batch-size $BATCH_SIZE \
        --output "$ITER_BUF" \
        2>&1 | grep -E "games|Done|Positions"

    # Append to accumulated buffer (keep last 500K positions = ~31MB)
    cat "$ITER_BUF" >> "$REPLAY_BUF"
    # Trim to max size (500K positions × 63 bytes = 31.5MB)
    MAX_BYTES=$((500000 * 63))
    CURRENT_BYTES=$(wc -c < "$REPLAY_BUF")
    if [ "$CURRENT_BYTES" -gt "$MAX_BYTES" ]; then
        TRIM_BYTES=$((CURRENT_BYTES - MAX_BYTES))
        # Align to record boundary
        TRIM_BYTES=$(( (TRIM_BYTES / 63 + 1) * 63 ))
        tail -c +$((TRIM_BYTES + 1)) "$REPLAY_BUF" > "${REPLAY_BUF}.tmp"
        mv "${REPLAY_BUF}.tmp" "$REPLAY_BUF"
    fi
    rm -f "$ITER_BUF"

    SP_END=$(date +%s)
    SP_TIME=$((SP_END - SP_START))
    POSITIONS=$(($(wc -c < "$REPLAY_BUF") / 63))
    echo "  Self-play: ${SP_TIME}s, buffer: $POSITIONS positions"

    # ── Phase 2: Python Training ────────────────────────
    echo "[Training: $EPOCHS epochs]"
    TRAIN_START=$(date +%s)

    python3 "$SCRIPT_DIR/train_alphazero.py" \
        --replay "$REPLAY_BUF" \
        --checkpoint "$MODEL_PT" \
        --output-onnx "$MODEL_ONNX" \
        --output-pt "$MODEL_PT" \
        --model-size "$MODEL_SIZE" \
        --epochs $EPOCHS \
        --batch-size 512 \
        --lr $LR \
        --expert-dir "$EXPERT_DIR" \
        --expert-ratio $EXPERT_RATIO \
        2>&1 | grep -E "Epoch|loss=|Exported|positions"

    TRAIN_END=$(date +%s)
    TRAIN_TIME=$((TRAIN_END - TRAIN_START))
    echo "  Training: ${TRAIN_TIME}s"

    # ── Save checkpoint ─────────────────────────────────
    if [ $((ITER % 5)) -eq 0 ]; then
        cp "$MODEL_PT" "$CHECKPOINT_DIR/iter${ITER}.pt"
        cp "$MODEL_ONNX" "$CHECKPOINT_DIR/iter${ITER}.onnx"
        echo "  Saved: iter${ITER}.pt"
    fi
    echo "$((ITER + 1))" > "$CHECKPOINT_DIR/iteration.txt"

    # ── Phase 3: Evaluation vs Gen7 ─────────────────────
    if [ $((ITER % EVAL_INTERVAL)) -eq 0 ] && [ -f "$ENGINE_BIN" ]; then
        echo "[Eval vs Gen7: $EVAL_GAMES games]"
        EVAL_RESULT=$(python3 "$SCRIPT_DIR/eval_vs_engine.py" \
            --model "$MODEL_ONNX" \
            --model-size "$MODEL_SIZE" \
            --games $EVAL_GAMES \
            --sims 200 \
            --engine "$ENGINE_BIN" \
            --time $EVAL_ENGINE_TIME \
            2>&1 | tail -1)
        echo "  $EVAL_RESULT"

        # Save best
        WINRATE=$(echo "$EVAL_RESULT" | grep -oP '[\d.]+(?=%)' | head -1)
        if [ -n "$WINRATE" ]; then
            IS_BETTER=$(python3 -c "print(1 if float('$WINRATE') > float('$BEST_WINRATE') else 0)")
            if [ "$IS_BETTER" = "1" ]; then
                BEST_WINRATE=$WINRATE
                cp "$MODEL_PT" "$CHECKPOINT_DIR/best.pt"
                cp "$MODEL_ONNX" "$CHECKPOINT_DIR/best.onnx"
                echo "  NEW BEST: ${WINRATE}% vs Gen7"
            fi
        fi
    fi

    ITER_END=$(date +%s)
    ITER_TIME=$((ITER_END - ITER_START))
    echo "  Total: ${ITER_TIME}s (selfplay ${SP_TIME}s + train ${TRAIN_TIME}s)"
    echo ""
done

echo "============================================================"
echo "Training complete! Best winrate: ${BEST_WINRATE}% vs Gen7"
echo "============================================================"
