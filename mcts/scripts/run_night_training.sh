#!/bin/bash
# Night training run — continuous selfplay + train for ~3.5h.
# Args: $1 = init checkpoint .pt path (e.g. checkpoints_v3/iter_500.pt)
#       $2 = duration seconds (default 12600 = 3.5h)
set -u
cd /home/nurlykhan/9QumalaqV2/mcts

INIT_CKPT="${1:-checkpoints_v3/iter_500.pt}"
DURATION="${2:-12600}"

NVIDIA_LIBS=/home/nurlykhan/.local/lib/python3.12/site-packages/nvidia
export ORT_DYLIB_PATH=/home/nurlykhan/.local/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.24.4
export LD_LIBRARY_PATH=$NVIDIA_LIBS/cublas/lib:$NVIDIA_LIBS/cuda_runtime/lib:$NVIDIA_LIBS/curand/lib:$NVIDIA_LIBS/cudnn/lib:$NVIDIA_LIBS/cufft/lib:${LD_LIBRARY_PATH:-}

CKPT_DIR=checkpoints_night
mkdir -p $CKPT_DIR
LOG=$CKPT_DIR/night_train.log

# Seed checkpoint into night dir as latest.pt so train_loop will resume from it
cp -f "$INIT_CKPT" $CKPT_DIR/latest.pt
echo "[$(date)] Seeded $CKPT_DIR/latest.pt from $INIT_CKPT" >> $LOG

# Run with timeout
timeout $DURATION python3 -u scripts/train_loop.py \
    --iterations 5000 \
    --games 100 \
    --sims 200 \
    --workers 10 \
    --model-size large2m \
    --resume $CKPT_DIR/latest.pt \
    --lr 0.0001 \
    --train-epochs 2 \
    --eval-interval 25 \
    --eval-pairs 5 \
    --eval-sims 1 \
    --max-buffer 500000 \
    --expert-ratio 0.20 \
    --checkpoint-dir $CKPT_DIR \
    --log $LOG \
    >> $CKPT_DIR/train_stdout.log 2>&1

EXIT=$?
echo "[$(date)] Training run finished with exit $EXIT" >> $LOG
