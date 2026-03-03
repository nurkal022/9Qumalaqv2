# V2: Gumbel AlphaZero Engine for Тоғызқұмалақ

A neural network-based game engine using Gumbel AlphaZero with Sequential Halving for the Kazakh board game Togyz Kumalak.

## Architecture

- **Rust** (`v2/rust/`): MCTS + Gumbel search, self-play generation, ONNX inference, arena
- **Python** (`v2/python/`): Neural network training (PyTorch), ONNX export, visualization

### Neural Network (TogyzNetV2)
- Input: 70 features (board state, strategic indicators)
- Trunk: 6 residual blocks, 256 hidden units (~170K parameters)
- Output: Policy logits (9 moves) + Value head ([-1, +1])

### Search: Gumbel AlphaZero
- Gumbel noise for stochastic exploration
- Sequential Halving for efficient candidate selection
- PUCT-based MCTS with neural network guidance

## Quick Start

```bash
# Setup
bash v2/scripts/setup.sh

# Run smoke test
./v2/rust/target/release/v2_engine test

# Full training loop (20 generations)
bash v2/scripts/train_loop.sh 20
```

## Training Pipeline

1. **Self-Play**: Rust engine generates games using current model
2. **Training**: Python trains neural network on self-play data
3. **Export**: PyTorch model exported to ONNX for Rust inference
4. **Arena**: New model vs old model to verify improvement

## Configuration

See `v2/configs/default.toml` for all hyperparameters.

## File Structure

```
v2/
├── rust/src/          # Rust engine
│   ├── board.rs       # Game rules
│   ├── mcts.rs        # MCTS search
│   ├── gumbel.rs      # Gumbel + Sequential Halving
│   ├── network.rs     # ONNX inference
│   ├── selfplay.rs    # Game generation
│   ├── arena.rs       # Model comparison
│   └── main.rs        # CLI
├── python/
│   ├── training/      # Model, training, export
│   └── utils/         # Elo, metrics, visualization
├── configs/           # Hyperparameters
├── scripts/           # Automation
├── models/            # Saved weights (by generation)
└── data/              # Self-play data
```
