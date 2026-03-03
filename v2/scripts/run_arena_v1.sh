#!/bin/bash
# Match V2 (Gumbel AlphaZero) vs V1 (Alpha-Beta)
set -e

V2_MODEL=${1:-v2/models/gen_20.onnx}
SIMULATIONS=${2:-100}
GAMES=${3:-200}

echo "=== V2 vs V1 Arena ==="
echo "V2 Model: $V2_MODEL"
echo "Games: $GAMES, Simulations: $SIMULATIONS"

# Build both engines
cargo build --release --manifest-path v2/rust/Cargo.toml
cargo build --release --manifest-path engine/Cargo.toml

V2_BIN=./v2/rust/target/release/v2_engine
V1_BIN=./engine/target/release/togyzkumalaq-engine

echo "Running arena..."
$V2_BIN arena \
    --model-new "$V2_MODEL" \
    --model-old "$V2_MODEL" \
    --games "$GAMES" \
    --simulations "$SIMULATIONS" \
    --candidates 8 \
    --sigma-scale 50.0

echo "Done."
