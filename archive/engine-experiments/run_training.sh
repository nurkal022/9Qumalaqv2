#!/bin/bash
# Training pipeline for NNUE improvement
# Combines self-play data with human game data

set -e
cd "$(dirname "$0")"

echo "==================================================="
echo "  NNUE Training Pipeline"
echo "==================================================="

# Check data files
SELFPLAY_DATA="gen1_training_data.bin"
HUMAN_DATA="human_all_training.bin"
HUMAN_ELO2000="human_elo2000_training.bin"

if [ ! -f "$SELFPLAY_DATA" ]; then
    echo "Warning: $SELFPLAY_DATA not found, using human data only"
    DATA="$HUMAN_DATA"
else
    DATA="$SELFPLAY_DATA,$HUMAN_DATA"
fi

echo "Training data: $DATA"
echo ""

# Phase 1: Train standard architecture (256->32->1)
echo "=== Phase 1: Standard architecture (256->32->1) ==="
python3 train_nnue_v2.py \
    --data "$DATA" \
    --epochs 100 \
    --batch-size 4096 \
    --hidden1 256 \
    --hidden2 32 \
    --lr 0.001 \
    --lam 0.75 \
    --output-pt nnue_gen1_256.pt \
    --output-bin nnue_weights_gen1_256.bin

echo ""
echo "=== Phase 2: Scaled architecture (512->32->1) ==="
python3 train_nnue_v2.py \
    --data "$DATA" \
    --epochs 100 \
    --batch-size 4096 \
    --hidden1 512 \
    --hidden2 32 \
    --lr 0.001 \
    --lam 0.75 \
    --output-pt nnue_gen1_512.pt \
    --output-bin nnue_weights_gen1_512.bin

echo ""
echo "=== Training complete! ==="
echo "Standard (256): nnue_weights_gen1_256.bin"
echo "Scaled (512):   nnue_weights_gen1_512.bin"
echo ""
echo "Test with:"
echo "  cp nnue_weights_gen1_256.bin nnue_weights.bin"
echo "  ./target/release/togyzkumalaq-engine match 100 500"
