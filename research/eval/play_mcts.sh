#!/bin/bash
# Programmatic serve mode for the strongest neural-net checkpoint we have.
# This is the AlphaZero-style Rust MCTS using the night_best ONNX model
# (iter 2650, 15.0% vs Gen7-current at 1-ply / raw policy).
#
# Note: full search (eval-sims=200) currently regresses (value head unreliable),
# so this serve mode uses Gumbel 1-ply selection. Strength is policy-only.

cd /home/nurlykhan/9QumalaqV2

NVIDIA_LIBS=/home/nurlykhan/.local/lib/python3.12/site-packages/nvidia
export ORT_DYLIB_PATH=/home/nurlykhan/.local/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.24.4
export LD_LIBRARY_PATH=$NVIDIA_LIBS/cublas/lib:$NVIDIA_LIBS/cuda_runtime/lib:$NVIDIA_LIBS/curand/lib:$NVIDIA_LIBS/cudnn/lib:$NVIDIA_LIBS/cufft/lib:${LD_LIBRARY_PATH:-}

MODEL=eval_onnx_final/night_best.onnx
if [ ! -f "$MODEL" ]; then
    echo "Model missing: $MODEL"
    echo "Run: python3 ../data/export_onnx.py /home/nurlykhan/9QumalaqV2/research/runs/_legacy/checkpoints_night/best.pt -o $MODEL --model-size large2m"
    exit 1
fi

echo "============================================================"
echo "  Rust MCTS serve — neural model night_best (iter 2650)"
echo "============================================================"
echo "Protocol (line-based stdin/stdout):"
echo "  newgame                                  -> ready"
echo "  position <state>                          -> ready"
echo "  go pos <state> time <ms>                 -> bestmove <pit> ..."
echo "  quit"
echo ""
echo "  Position format: w0,w1,...,w8/b0,...,b8/kw,kb/tw,tb/side"
echo "    (side: 0=white, 1=black; tw/tb tuzdyk pit -1 if none)"
echo ""

exec ./target/release/mcts --serve --model "$MODEL"
