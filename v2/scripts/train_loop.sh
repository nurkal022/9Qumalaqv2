#!/bin/bash
# Full Gumbel AlphaZero training loop: selfplay -> train -> export -> arena
set -e

GENERATIONS=${1:-20}
GAMES_PER_GEN=3000
SIMULATIONS=100
CANDIDATES=8
EPOCHS=10
BATCH_SIZE=512
HIDDEN=256
BLOCKS=6
SIGMA_SCALE=50.0
THREADS=$(nproc)

echo "=== Gumbel AlphaZero Training Loop ==="
echo "Generations: $GENERATIONS"
echo "Games/gen: $GAMES_PER_GEN"
echo "Simulations: $SIMULATIONS"
echo "Threads: $THREADS"

# Build Rust engine
echo "Building V2 engine..."
cargo build --release --manifest-path v2/rust/Cargo.toml
V2_BIN=./v2/rust/target/release/v2_engine

# Create initial random model if needed
if [ ! -f v2/models/gen_0.pt ]; then
    echo "Creating initial random model..."
    python v2/python/training/model.py --save
    python v2/python/training/export_onnx.py \
        --model-in v2/models/gen_0.pt \
        --output v2/models/gen_0.onnx \
        --hidden-size $HIDDEN --num-blocks $BLOCKS
fi

for gen in $(seq 1 $GENERATIONS); do
    prev=$((gen - 1))
    echo ""
    echo "========================================="
    echo "  Generation $gen / $GENERATIONS"
    echo "========================================="

    # 1. Self-Play
    echo "[1/4] Generating $GAMES_PER_GEN games..."
    $V2_BIN selfplay \
        --model v2/models/gen_${prev}.onnx \
        --games $GAMES_PER_GEN \
        --simulations $SIMULATIONS \
        --candidates $CANDIDATES \
        --threads $THREADS \
        --sigma-scale $SIGMA_SCALE \
        --output v2/data/gen_${gen}.bin

    # 2. Train
    echo "[2/4] Training model..."
    python v2/python/training/train.py \
        --data v2/data/gen_${gen}.bin \
        --model-in v2/models/gen_${prev}.pt \
        --model-out v2/models/gen_${gen}.pt \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --hidden-size $HIDDEN \
        --num-blocks $BLOCKS

    # 3. Export ONNX
    echo "[3/4] Exporting to ONNX..."
    python v2/python/training/export_onnx.py \
        --model-in v2/models/gen_${gen}.pt \
        --output v2/models/gen_${gen}.onnx \
        --hidden-size $HIDDEN --num-blocks $BLOCKS

    # 4. Arena: new vs old generation
    echo "[4/4] Arena: Gen $gen vs Gen $prev..."
    $V2_BIN arena \
        --model-new v2/models/gen_${gen}.onnx \
        --model-old v2/models/gen_${prev}.onnx \
        --games 200 \
        --simulations $SIMULATIONS \
        --candidates $CANDIDATES \
        --sigma-scale $SIGMA_SCALE

    echo "Generation $gen complete."
done

echo ""
echo "=== Training complete! ==="
echo "Final model: v2/models/gen_${GENERATIONS}.onnx"
