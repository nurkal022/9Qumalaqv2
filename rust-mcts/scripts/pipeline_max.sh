#!/bin/bash
# Maximum improvement pipeline after book bug fix.
# Steps:
# 1. Wait for clean datagen to complete
# 2. Train 3 models: distillation-only, playok-only, hybrid
# 3. Export all to ONNX
# 4. Eval each vs fixed engine
# 5. Compare with previous results

set -e
cd /home/nurlykhan/9QumalaqV2/rust-mcts

NVIDIA_LIBS=/home/nurlykhan/.local/lib/python3.12/site-packages/nvidia
export ORT_DYLIB_PATH=/home/nurlykhan/.local/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.24.4
export LD_LIBRARY_PATH=$NVIDIA_LIBS/cublas/lib:$NVIDIA_LIBS/cuda_runtime/lib:$NVIDIA_LIBS/curand/lib:$NVIDIA_LIBS/cudnn/lib:$NVIDIA_LIBS/cufft/lib:${LD_LIBRARY_PATH:-}

CKPT=checkpoints_max
mkdir -p $CKPT
LOG=$CKPT/pipeline.log

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }

log "=== Pipeline start ==="

# Step 1: Wait for datagen
while pgrep -f "togyzkumalaq-engine datagen" > /dev/null; do
    line=$(tail -c 300 /tmp/datagen_clean.log 2>/dev/null | tr '\r' '\n' | tail -1)
    log "datagen: $(echo "$line" | grep -oE '[0-9]+%')"
    sleep 300
done

log "Datagen complete. Data: $(ls -lh distill_data/clean_gen7d10_training_data.bin | awk '{print $5}')"

# Step 2a: Train distillation-only from supervised init
log "=== Training: distillation-only ==="
python3 -u scripts/train_master.py \
    --engine-data "distill_data/clean_gen7d10_training_data.bin" \
    --engine-weight 1.0 \
    --output $CKPT/dist_only.pt \
    --model-size large2m \
    --init-checkpoint checkpoints_v3/supervised_fresh.pt \
    --epochs 40 --batch-size 1024 --lr 0.0003 \
    2>&1 | tee -a $LOG

# Step 2b: Train hybrid (engine + playok)
log "=== Training: hybrid (engine + playok) ==="
python3 -u scripts/train_master.py \
    --engine-data "distill_data/clean_gen7d10_training_data.bin" \
    --engine-weight 1.5 \
    --playok-dir ../game-pars/games \
    --playok-min-elo 1500 \
    --playok-max 500000 \
    --playok-weight 1.0 \
    --output $CKPT/hybrid.pt \
    --model-size large2m \
    --init-checkpoint checkpoints_v3/supervised_fresh.pt \
    --epochs 60 --batch-size 1024 --lr 0.0003 \
    --label-smooth 0.05 \
    2>&1 | tee -a $LOG

# Step 3: Export ONNX
log "=== Exporting ONNX models ==="
for name in dist_only hybrid; do
    python3 scripts/export_onnx.py $CKPT/$name.pt \
        -o $CKPT/$name.onnx --model-size large2m 2>&1 | tail -1 | tee -a $LOG
done

# Also export supervised_fresh for comparison
if [ ! -f $CKPT/supervised.onnx ]; then
    python3 scripts/export_onnx.py checkpoints_v3/supervised_fresh.pt \
        -o $CKPT/supervised.onnx --model-size large2m 2>&1 | tail -1 | tee -a $LOG
fi

# Step 4: Eval vs fixed engine (1-ply, fast)
log "=== Eval vs fixed engine (1-ply, 10 pairs) ==="
for name in supervised dist_only hybrid; do
    log "Model: $name"
    timeout 180 ./target/release/rust-mcts --eval \
        --model $CKPT/$name.onnx \
        --games 10 --eval-sims 1 \
        --engine /home/nurlykhan/9QumalaqV2/engine/target/release/togyzkumalaq-engine \
        --engine-time 200 2>&1 | grep '{' | tee -a $LOG
done

log "=== Pipeline complete ==="
