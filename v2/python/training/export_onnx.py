"""
Export trained PyTorch model to ONNX format for Rust inference.
"""

import argparse

import torch

from model import TogyzNetV2, FEATURE_SIZE


def export(args):
    model = TogyzNetV2(
        input_size=FEATURE_SIZE,
        hidden_size=args.hidden_size,
        num_blocks=args.num_blocks,
    )
    model.load_state_dict(torch.load(args.model_in, map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, FEATURE_SIZE)

    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["features"],
        output_names=["policy_logits", "value"],
        dynamic_axes={
            "features": {0: "batch"},
            "policy_logits": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"Exported ONNX model to {args.output}")

    # Verify
    import onnxruntime as ort
    import numpy as np

    session = ort.InferenceSession(args.output)
    test_input = np.random.randn(1, FEATURE_SIZE).astype(np.float32)
    outputs = session.run(None, {"features": test_input})
    print(f"  Policy logits shape: {outputs[0].shape}")
    print(f"  Value shape: {outputs[1].shape}")
    print(f"  Verification passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export TogyzNet V2 to ONNX")
    parser.add_argument("--model-in", required=True, help="Input PyTorch model (.pt)")
    parser.add_argument("--output", required=True, help="Output ONNX model (.onnx)")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=6)
    export(parser.parse_args())
