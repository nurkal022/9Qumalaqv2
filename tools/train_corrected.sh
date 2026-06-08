#!/bin/bash
# Corrected-label AlphaZero training (2026-06-08).
#
# Continues from iter_2645 but with the END-GAME SWEEP RULE fixed everywhere
# (core/board.rs game_result, game.py referee, swept value magnitude in
# self_play.rs + league.rs). This relabels the ~59% of decisive endgames that the
# old pipeline scored with the WRONG winner — the root cause of "value head
# unreliable". Goal: a reliable value head so MCTS search >1-ply helps instead of
# hurting.
#
# Usage:  bash tools/train_corrected.sh [iterations]
# Resumable: re-run; it picks up checkpoints_dir/latest.pt via --resume.
set -u
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

export NVIDIA_LIBS=/home/nurlykhan/.local/lib/python3.12/site-packages/nvidia
export ORT_DYLIB_PATH=/home/nurlykhan/.local/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.24.4
export LD_LIBRARY_PATH=$NVIDIA_LIBS/cublas/lib:$NVIDIA_LIBS/cuda_runtime/lib:$NVIDIA_LIBS/curand/lib:$NVIDIA_LIBS/cudnn/lib:$NVIDIA_LIBS/cufft/lib:${LD_LIBRARY_PATH:-}

ITERS="${1:-60}"
CKPT_DIR="$ROOT/research/runs/2026-06-08-corrected/checkpoints"
LOG="$ROOT/research/runs/2026-06-08-corrected/train.log"
INIT="$ROOT/research/runs/_legacy/checkpoints_v3/iter_2645.pt"
mkdir -p "$CKPT_DIR"

# Resume from this run's own latest.pt if present, else seed from iter_2645.
RESUME_ARG=()
if [ -f "$CKPT_DIR/latest.pt" ]; then
  RESUME_ARG=(--resume "$CKPT_DIR/latest.pt")
else
  RESUME_ARG=(--init-checkpoint "$INIT")
fi

cd "$ROOT/research/training"
exec python3.12 -u train_loop.py \
  --iterations "$ITERS" \
  --games 100 \
  --sims 200 \
  --workers 10 \
  --model-size large2m \
  "${RESUME_ARG[@]}" \
  --lr 0.0001 \
  --train-epochs 2 \
  --eval-interval 20 \
  --eval-pairs 10 \
  --eval-sims 1 \
  --max-buffer 500000 \
  --expert-ratio 0.20 \
  --checkpoint-dir "$CKPT_DIR" \
  --log "$LOG"
