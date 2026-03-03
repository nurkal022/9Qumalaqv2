#!/bin/bash
# Setup script: install dependencies for V2 training
set -e

echo "=== V2 Setup ==="

# Python dependencies
echo "Installing Python dependencies..."
pip install torch numpy onnx onnxruntime matplotlib

# Build Rust engine
echo "Building Rust engine..."
cargo build --release --manifest-path v2/rust/Cargo.toml

# Create directories
mkdir -p v2/models v2/data

# Run smoke test
echo "Running smoke test..."
./v2/rust/target/release/v2_engine test

echo "=== Setup complete! ==="
