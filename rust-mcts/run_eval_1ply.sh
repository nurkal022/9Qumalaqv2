#!/bin/bash
# 1-ply (raw policy) eval of each candidate vs Gen7. Fast — same eval as train_loop uses.
set -u
cd /home/nurlykhan/9QumalaqV2/rust-mcts

NVIDIA_LIBS=/home/nurlykhan/.local/lib/python3.12/site-packages/nvidia
export ORT_DYLIB_PATH=/home/nurlykhan/.local/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.24.4
export LD_LIBRARY_PATH=$NVIDIA_LIBS/cublas/lib:$NVIDIA_LIBS/cuda_runtime/lib:$NVIDIA_LIBS/curand/lib:$NVIDIA_LIBS/cudnn/lib:$NVIDIA_LIBS/cufft/lib:${LD_LIBRARY_PATH:-}

ENGINE=/home/nurlykhan/9QumalaqV2/engine/target/release/togyzkumalaq-engine
RESULTS=/home/nurlykhan/9QumalaqV2/rust-mcts/eval_results_1ply.txt

echo "=== 1-ply eval vs Gen7 engine ===" > $RESULTS
echo "Date: $(date)" >> $RESULTS
echo "Format: 25 color-pairs (50 games), eval-sims=1, engine=200ms" >> $RESULTS
echo "" >> $RESULTS

for ckpt in iter_500 iter_1000 iter_1500 iter_2000 iter_2500 iter_2645; do
    echo "" >> $RESULTS
    echo "--- $ckpt ---" >> $RESULTS
    t0=$(date +%s)
    timeout 600 ./target/release/rust-mcts --eval \
        --model eval_onnx/$ckpt.onnx \
        --games 25 --eval-sims 1 \
        --engine $ENGINE \
        --engine-time 200 \
        --batch-size 64 \
        --workers 1 2>&1 | grep -E '\{|pair_wins|pairs |^==' | tee -a $RESULTS
    t1=$(date +%s)
    echo "Elapsed: $((t1-t0))s" >> $RESULTS
done

echo "" >> $RESULTS
echo "=== Eval done $(date) ===" >> $RESULTS
