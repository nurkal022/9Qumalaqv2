#!/bin/bash
# Prepare a fresh clone of the `alphazero` branch for a training run on a GPU server.
#
#   HF_REPO=<user>/<dataset> bash tools/az_server_setup.sh
#
# Builds the Rust workspace, pulls the nets + expert games + EGTB from the private
# Hugging Face dataset, wires the engine resources the way train_loop.py expects, and
# prints the env exports for the run. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
: "${HF_REPO:?set HF_REPO to the Hugging Face dataset (see README, section Данные)}"
PY="${PY:-python3.12}"
DL="${DL_DIR:-$ROOT/.hf-data}"

echo "[1/5] Python deps"
"$PY" -m pip install -q -U torch --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install -q -U onnxruntime-gpu onnx numpy huggingface_hub \
    nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-curand-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12

echo "[2/5] Rust build (engine + mcts)"
cargo build --release -p togyzkumalaq-engine -p mcts
# mcts resolves nnue/egtb/book from engine_dir/../.. = engine/, so the engine binary
# must be reachable as engine/target/release/togyzkumalaq-engine.
mkdir -p engine/target/release
ln -sf "$ROOT/target/release/togyzkumalaq-engine" engine/target/release/togyzkumalaq-engine

echo "[3/5] Data from $HF_REPO"
mkdir -p "$DL"
for a in models playok-games-current research-runs; do
    [ -f "$DL/$a.tar.zst" ] || hf download "$HF_REPO" "$a.tar.zst" --repo-type dataset --local-dir "$DL"
    tar -I zstd -xf "$DL/$a.tar.zst" -C "$ROOT"
done

echo "[4/5] Engine resources"
# This branch's core/ uses the either-side-empty terminal rule -> EGTB format TKEGTB01.
# (models/engine/egtb.bin is TKEGTB02, built for the endgame-rules branch.)
[ -f engine/egtb.bin ] || cp runs/egtb_old_rule.bin engine/egtb.bin

echo "[5/5] Env for the run"
SITE="$("$PY" -c 'import site; print(site.getusersitepackages())')"
[ -d "$SITE/nvidia" ] || SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
ORT="$(ls "$SITE"/onnxruntime/capi/libonnxruntime.so.* | head -1)"
cat <<EOF

export NVIDIA_LIBS=$SITE/nvidia
export ORT_DYLIB_PATH=$ORT
export EXPERT_DIR=$ROOT/game-pars/games

Smoke test (few minutes), then the real run: see docs/ALPHAZERO_3000.md
EOF
