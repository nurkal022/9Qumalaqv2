#!/bin/bash
# Tournament-quality test: night_best vs both engines at 1000ms/move
set -u
cd /home/nurlykhan/9QumalaqV2/rust-mcts

NVIDIA_LIBS=/home/nurlykhan/.local/lib/python3.12/site-packages/nvidia
export ORT_DYLIB_PATH=/home/nurlykhan/.local/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.24.4
export LD_LIBRARY_PATH=$NVIDIA_LIBS/cublas/lib:$NVIDIA_LIBS/cuda_runtime/lib:$NVIDIA_LIBS/curand/lib:$NVIDIA_LIBS/cudnn/lib:$NVIDIA_LIBS/cufft/lib:${LD_LIBRARY_PATH:-}

ENGINE_CUR=/home/nurlykhan/9QumalaqV2/engine/target/release/togyzkumalaq-engine
ENGINE_BASE=/home/nurlykhan/9QumalaqV2/engine/target/release/togyzkumalaq-engine-baseline
RESULTS=/home/nurlykhan/9QumalaqV2/rust-mcts/champion_test.txt

echo "=== Tournament-quality eval (1000ms/move) ===" > $RESULTS
echo "Date: $(date)" >> $RESULTS
echo "Format: 20 color-pairs (40 games), 1-ply Rust (raw policy)" >> $RESULTS
echo "" >> $RESULTS

for engine_name in current baseline; do
    if [ "$engine_name" = "current" ]; then
        ENG=$ENGINE_CUR
    else
        ENG=$ENGINE_BASE
    fi
    echo "" >> $RESULTS
    echo "=== night_best vs Gen7-$engine_name (1000ms) ===" >> $RESULTS
    t0=$(date +%s)
    timeout 2400 ./target/release/rust-mcts --eval \
        --model eval_onnx_final/night_best.onnx \
        --games 20 --eval-sims 1 \
        --engine $ENG \
        --engine-time 1000 \
        --batch-size 64 \
        --workers 1 2>&1 | grep -E '\{|pair_wins|pairs |^==' | tee -a $RESULTS
    t1=$(date +%s)
    echo "Elapsed: $((t1-t0))s" >> $RESULTS
done

echo "" >> $RESULTS
echo "=== Tournament test done $(date) ===" >> $RESULTS
