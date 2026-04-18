#!/bin/bash
# Maximum results pipeline on current hardware (RTX 5080, 24 cores).
#
# Phase 1: Generate distillation data from Gen7 engine
# Phase 2: Train distillation model (policy + value from engine)
# Phase 3: Evaluate distilled model vs Gen7 engine
# Phase 4: (optional) Fine-tune with MCTS selfplay starting from distilled
#
# Run: bash scripts/run_max.sh

set -e
cd /home/nurlykhan/9QumalaqV2/rust-mcts

# Paths
ENGINE=/home/nurlykhan/9QumalaqV2/engine/target/release/togyzkumalaq-engine
ENGINE_DIR=/home/nurlykhan/9QumalaqV2/engine
DATA_DIR=/home/nurlykhan/9QumalaqV2/rust-mcts/distill_data
CKPT_DIR=/home/nurlykhan/9QumalaqV2/rust-mcts/checkpoints_distill
LOG=/home/nurlykhan/9QumalaqV2/rust-mcts/max_pipeline.log

mkdir -p $DATA_DIR $CKPT_DIR

# CUDA env
NVIDIA_LIBS=/home/nurlykhan/.local/lib/python3.12/site-packages/nvidia
export ORT_DYLIB_PATH=/home/nurlykhan/.local/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.24.4
export LD_LIBRARY_PATH=$NVIDIA_LIBS/cublas/lib:$NVIDIA_LIBS/cuda_runtime/lib:$NVIDIA_LIBS/curand/lib:$NVIDIA_LIBS/cudnn/lib:$NVIDIA_LIBS/cufft/lib:${LD_LIBRARY_PATH:-}

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }

# Phase 1: Datagen (10000 games, depth 12, ~2-3 hours)
if [ $(ls $DATA_DIR/*.bin 2>/dev/null | wc -l) -lt 16 ]; then
    log "=== Phase 1: Generating distillation data ==="
    cd $ENGINE_DIR
    # 10000 games, depth 12, 16 threads
    # At ~125 games/min on depth 8, depth 12 will be ~40 games/min → ~4 hours for 10000
    ./target/release/togyzkumalaq-engine datagen 10000 12 16 $DATA_DIR/gen7 2>&1 | tee -a $LOG
    cd /home/nurlykhan/9QumalaqV2/rust-mcts
fi

log "=== Data ready: $(du -sh $DATA_DIR | cut -f1) ==="

# Phase 2: Distillation training
log "=== Phase 2: Distillation training ==="
python3 -u scripts/train_distillation.py \
    --data "$DATA_DIR/*.bin" \
    --output $CKPT_DIR/distilled.pt \
    --model-size large2m \
    --init-checkpoint checkpoints_v3/supervised_fresh.pt \
    --epochs 40 \
    --batch-size 1024 \
    --lr 0.0005 \
    2>&1 | tee -a $LOG

log "=== Phase 3: Export ONNX and evaluate ==="
python3 scripts/export_onnx.py $CKPT_DIR/distilled.pt \
    -o $CKPT_DIR/distilled.onnx \
    --model-size large2m 2>&1 | tee -a $LOG

# Deep MCTS eval vs engine (20 pairs)
log "=== Phase 4: Deep MCTS eval vs Gen7 ==="
./target/release/rust-mcts --eval \
    --model $CKPT_DIR/distilled.onnx \
    --games 20 --eval-sims 200 \
    --engine $ENGINE \
    --engine-time 200 2>&1 | tee -a $LOG

log "=== Pipeline complete ==="
