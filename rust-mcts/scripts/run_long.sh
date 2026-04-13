#!/bin/bash
# Long-running training with auto-restart and game collection
# Run: nohup bash scripts/run_long.sh > /dev/null 2>&1 &

cd /home/nurlykhan/9QumalaqV2/rust-mcts

# CUDA/ORT paths
export NVIDIA_LIBS=/home/nurlykhan/.local/lib/python3.12/site-packages/nvidia
export ORT_DYLIB_PATH=/home/nurlykhan/.local/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.24.4
export LD_LIBRARY_PATH=$NVIDIA_LIBS/cublas/lib:$NVIDIA_LIBS/cuda_runtime/lib:$NVIDIA_LIBS/curand/lib:$NVIDIA_LIBS/cudnn/lib:$NVIDIA_LIBS/cufft/lib:${LD_LIBRARY_PATH:-}

LOG=/home/nurlykhan/9QumalaqV2/rust-mcts/training.log
GAME_LOG=/home/nurlykhan/9QumalaqV2/rust-mcts/game_collection.log

echo "$(date): === Long training started ===" >> $LOG

# Collect master games from server every 6 hours in background
collect_and_clean() {
    while true; do
        echo "$(date): Collecting master games..." >> $GAME_LOG
        python3 scripts/collect_master_games.py \
            --output master_games.bin \
            --games-dir master_games \
            >> $GAME_LOG 2>&1

        # Clean old checkpoints (keep every 100th + latest/best)
        cd /home/nurlykhan/9QumalaqV2/rust-mcts/checkpoints_v3
        for f in iter_*.pt; do
            [ -f "$f" ] || continue
            num=$(echo $f | grep -oP '\d+')
            if [ $((num % 100)) -ne 0 ]; then
                rm "$f"
            fi
        done
        echo "$(date): Cleanup done, disk: $(df -h / | tail -1 | awk '{print $4}') free" >> $GAME_LOG
        cd /home/nurlykhan/9QumalaqV2/rust-mcts

        sleep 21600  # 6 hours
    done
}
collect_and_clean &
COLLECT_PID=$!

# Auto-restart training loop
MAX_RESTARTS=50
RESTART_COUNT=0

while [ $RESTART_COUNT -lt $MAX_RESTARTS ]; do
    echo "$(date): Starting training (restart #$RESTART_COUNT)..." >> $LOG

    python3 -u scripts/train_loop.py \
        --iterations 5000 \
        --games 100 \
        --sims 200 \
        --workers 10 \
        --model-size large2m \
        --resume checkpoints_v3/latest.pt \
        --lr 0.0001 \
        --train-epochs 2 \
        --eval-interval 50 \
        --eval-pairs 5 \
        --eval-sims 200 \
        --max-buffer 500000 \
        --expert-ratio 0.15 \
        --checkpoint-dir checkpoints_v3 \
        --log /home/nurlykhan/9QumalaqV2/rust-mcts/training.log \
        >> /home/nurlykhan/9QumalaqV2/rust-mcts/training_stdout.log 2>&1

    EXIT_CODE=$?
    echo "$(date): Training exited with code $EXIT_CODE" >> $LOG

    if [ $EXIT_CODE -eq 0 ]; then
        echo "$(date): Training completed normally" >> $LOG
        break
    fi

    RESTART_COUNT=$((RESTART_COUNT + 1))
    echo "$(date): Restarting in 30 seconds..." >> $LOG
    sleep 30
done

# Cleanup
kill $COLLECT_PID 2>/dev/null
echo "$(date): === Long training finished ===" >> $LOG
