#!/bin/bash
# Eval each candidate checkpoint vs Gen7 engine.
# Sequential runs; saves results to eval_results.txt
set -e
cd /home/nurlykhan/9QumalaqV2/rust-mcts

NVIDIA_LIBS=/home/nurlykhan/.local/lib/python3.12/site-packages/nvidia
export ORT_DYLIB_PATH=/home/nurlykhan/.local/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.24.4
export LD_LIBRARY_PATH=$NVIDIA_LIBS/cublas/lib:$NVIDIA_LIBS/cuda_runtime/lib:$NVIDIA_LIBS/curand/lib:$NVIDIA_LIBS/cudnn/lib:$NVIDIA_LIBS/cufft/lib:${LD_LIBRARY_PATH:-}

ENGINE=/home/nurlykhan/9QumalaqV2/engine/target/release/togyzkumalaq-engine
RESULTS=/home/nurlykhan/9QumalaqV2/rust-mcts/eval_results.txt

echo "=== Checkpoint vs Gen7 engine eval ===" > $RESULTS
echo "Date: $(date)" >> $RESULTS
echo "Format: 10 color-pairs (20 games), MCTS sims=200, engine=200ms" >> $RESULTS
echo "" >> $RESULTS

for ckpt in iter_500 iter_1000 iter_2000 iter_2645; do
    echo "" >> $RESULTS
    echo "--- $ckpt ---" >> $RESULTS
    t0=$(date +%s)
    timeout 900 ./target/release/rust-mcts --eval \
        --model eval_onnx/$ckpt.onnx \
        --games 10 --eval-sims 200 \
        --engine $ENGINE \
        --engine-time 200 \
        --batch-size 64 \
        --workers 1 2>&1 | grep -E '\[|wins|losses|=' | tee -a $RESULTS
    t1=$(date +%s)
    echo "Elapsed: $((t1-t0))s" >> $RESULTS
done

echo "" >> $RESULTS
echo "=== Eval done $(date) ===" >> $RESULTS
