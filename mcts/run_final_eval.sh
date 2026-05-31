#!/bin/bash
# Final eval: 3-way comparison after night training.
# Compares: starting iter_2645, training "best.pt", and final "latest.pt".
set -u
cd /home/nurlykhan/9QumalaqV2/mcts

NVIDIA_LIBS=/home/nurlykhan/.local/lib/python3.12/site-packages/nvidia
export ORT_DYLIB_PATH=/home/nurlykhan/.local/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.24.4
export LD_LIBRARY_PATH=$NVIDIA_LIBS/cublas/lib:$NVIDIA_LIBS/cuda_runtime/lib:$NVIDIA_LIBS/curand/lib:$NVIDIA_LIBS/cudnn/lib:$NVIDIA_LIBS/cufft/lib:${LD_LIBRARY_PATH:-}

ENGINE=/home/nurlykhan/9QumalaqV2/engine/target/release/togyzkumalaq-engine
RESULTS=/home/nurlykhan/9QumalaqV2/mcts/final_eval.txt
ONNX_DIR=eval_onnx_final
mkdir -p $ONNX_DIR

# Export 3 candidates to ONNX
echo "=== Final eval ($(date)) ===" > $RESULTS
echo "Format: 20 color-pairs (40 games), eval-sims=1, engine=100ms" >> $RESULTS
echo "Pre-training baseline: iter_2645 = 15% (10 pairs)" >> $RESULTS
echo "" >> $RESULTS

declare -A CKPTS=(
    [start_iter_2645]="checkpoints_v3/iter_2645.pt"
    [night_best]="checkpoints_night/best.pt"
    [night_latest]="checkpoints_night/latest.pt"
)

echo "[Exporting ONNX]" >> $RESULTS
for name in start_iter_2645 night_best night_latest; do
    pt="${CKPTS[$name]}"
    if [ ! -f "$pt" ]; then
        echo "  $name: $pt NOT FOUND, skipping" >> $RESULTS
        continue
    fi
    python3 scripts/export_onnx.py $pt -o $ONNX_DIR/$name.onnx --model-size large2m 2>&1 | grep -E "Exported|Error" | head -1 >> $RESULTS
done
echo "" >> $RESULTS

echo "[Eval vs Gen7-current (1-ply, 20 pairs, engine=100ms)]" >> $RESULTS
for name in start_iter_2645 night_best night_latest; do
    onnx=$ONNX_DIR/$name.onnx
    if [ ! -f "$onnx" ]; then continue; fi
    echo "" >> $RESULTS
    echo "--- $name vs Gen7-current ---" >> $RESULTS
    t0=$(date +%s)
    timeout 600 ./target/release/mcts --eval \
        --model $onnx \
        --games 20 --eval-sims 1 \
        --engine $ENGINE \
        --engine-time 100 \
        --batch-size 64 \
        --workers 1 2>&1 | grep -E '\{|pair_wins|pairs |^==' | tee -a $RESULTS
    t1=$(date +%s)
    echo "Elapsed: $((t1-t0))s" >> $RESULTS
done

# Sanity check: also vs baseline engine (Mar 18 build, supposedly weaker)
ENGINE_BASE=/home/nurlykhan/9QumalaqV2/engine/target/release/togyzkumalaq-engine-baseline
if [ -x "$ENGINE_BASE" ]; then
    echo "" >> $RESULTS
    echo "[Eval vs Gen7-baseline (older binary, Mar 18)]" >> $RESULTS
    for name in start_iter_2645 night_latest; do
        onnx=$ONNX_DIR/$name.onnx
        if [ ! -f "$onnx" ]; then continue; fi
        echo "" >> $RESULTS
        echo "--- $name vs Gen7-baseline ---" >> $RESULTS
        timeout 600 ./target/release/mcts --eval \
            --model $onnx \
            --games 20 --eval-sims 1 \
            --engine $ENGINE_BASE \
            --engine-time 100 \
            --batch-size 64 \
            --workers 1 2>&1 | grep -E '\{|pair_wins|pairs |^==' | tee -a $RESULTS
    done
fi

# Head-to-head: night_latest vs start_iter_2645 (no engine, would need different harness)
# Skipped — Rust eval mode only supports model-vs-engine.

echo "" >> $RESULTS
echo "=== Final eval done $(date) ===" >> $RESULTS
