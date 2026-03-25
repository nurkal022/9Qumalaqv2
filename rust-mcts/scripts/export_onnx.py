#!/usr/bin/env python3
"""Export TogyzNet PyTorch model to ONNX format for Rust MCTS inference."""

import sys
import os
import argparse
import torch

# Add alphazero module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../alphazero-code/alphazero'))

from model import create_model


def export_onnx(checkpoint_path: str, output_path: str, model_size: str = "medium"):
    """Export model to ONNX with dynamic batch size."""

    # Create model
    model = create_model(model_size, device='cpu')

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Handle torch.compile _orig_mod. prefix
    cleaned = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(cleaned)
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_size} ({params:,} params)")

    # Dummy input: [batch=1, channels=7, positions=9]
    dummy = torch.randn(1, 7, 9)

    # Test forward pass
    with torch.no_grad():
        log_policy, value = model(dummy)
        print(f"Test output: log_policy shape={log_policy.shape}, value shape={value.shape}")
        print(f"  policy sum (exp): {torch.exp(log_policy).sum().item():.4f}")
        print(f"  value: {value.item():.4f}")

    # Export
    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        opset_version=17,
        input_names=['state'],
        output_names=['log_policy', 'value'],
        dynamic_axes={
            'state': {0: 'batch'},
            'log_policy': {0: 'batch'},
            'value': {0: 'batch'},
        },
    )

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"\nExported: {output_path} ({size_mb:.1f} MB)")

    # Verify with different batch sizes
    import onnxruntime as ort
    sess = ort.InferenceSession(output_path)

    for batch_size in [1, 16, 64, 128]:
        test_input = torch.randn(batch_size, 7, 9).numpy()
        outputs = sess.run(None, {'state': test_input})
        print(f"  Batch {batch_size:3d}: policy {outputs[0].shape}, value {outputs[1].shape} ✓")


def main():
    parser = argparse.ArgumentParser(description="Export TogyzNet to ONNX")
    parser.add_argument("checkpoint", help="Path to PyTorch checkpoint (.pt)")
    parser.add_argument("--output", "-o", default="model.onnx", help="Output ONNX file")
    parser.add_argument("--model-size", default="medium", choices=["small", "medium", "large"])
    args = parser.parse_args()

    export_onnx(args.checkpoint, args.output, args.model_size)


if __name__ == "__main__":
    main()
