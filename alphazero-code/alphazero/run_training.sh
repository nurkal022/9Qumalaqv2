#!/bin/bash
# AlphaZero Training Script for Тоғызқұмалақ

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate togyz-alphazero

# Change to alphazero directory
cd "$(dirname "$0")"

echo "=========================================="
echo "  AlphaZero Тоғызқұмалақ Training"
echo "=========================================="
echo ""

# Check CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "Starting training..."
echo ""

# Default training parameters
# Adjust these for your needs:
# - Quick test: --games 20 --simulations 200 --iterations 10
# - Normal: --games 100 --simulations 800 --iterations 100
# - Strong: --games 200 --simulations 1600 --iterations 500

python train.py \
    --model-size medium \
    --games 100 \
    --simulations 800 \
    --iterations 100 \
    --batch-size 256 \
    --lr 0.001 \
    "$@"

echo ""
echo "Training completed!"

