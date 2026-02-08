"""
Export trained model to ONNX for browser deployment
"""

import torch
import numpy as np
import os
import json
from model import create_model


def export_to_onnx(checkpoint_path: str, output_path: str, model_size: str = "medium"):
    """
    Export PyTorch model to ONNX format
    
    Args:
        checkpoint_path: Path to PyTorch checkpoint
        output_path: Output ONNX file path
        model_size: Model size used during training
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model
    model = create_model(model_size, device='cpu')
    
    # Handle torch.compile() prefix
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        # Remove _orig_mod. prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                new_state_dict[k[10:]] = v  # Remove '_orig_mod.' prefix
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    model.eval()
    
    # Dummy input for tracing
    dummy_input = torch.randn(1, 7, 9)
    
    # Export to ONNX
    print(f"Exporting to ONNX: {output_path}")
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['state'],
        output_names=['policy', 'value'],
        dynamic_axes={
            'state': {0: 'batch_size'},
            'policy': {0: 'batch_size'},
            'value': {0: 'batch_size'}
        }
    )
    
    # Verify export
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified successfully!")
    
    # Print model info
    print(f"\nModel info:")
    print(f"  Iteration: {checkpoint.get('iteration', 'unknown')}")
    print(f"  Total games: {checkpoint.get('total_games', 'unknown')}")
    print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    return output_path


def export_metadata(checkpoint_path: str, output_path: str):
    """Export model metadata as JSON for browser"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    metadata = {
        'iteration': checkpoint.get('iteration', 0),
        'total_games': checkpoint.get('total_games', 0),
        'config': checkpoint.get('config', {}),
        'input_shape': [7, 9],
        'output_policy_size': 9,
        'normalization': {
            'pit_factor': 50.0,
            'kazan_factor': 82.0
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to: {output_path}")
    return metadata


def create_browser_bundle(checkpoint_path: str, output_dir: str, model_size: str = "medium"):
    """
    Create complete browser deployment bundle
    
    Creates:
    - model.onnx: ONNX model
    - metadata.json: Model metadata
    - inference.js: JavaScript inference code
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Export ONNX
    onnx_path = os.path.join(output_dir, "model.onnx")
    export_to_onnx(checkpoint_path, onnx_path, model_size)
    
    # Export metadata
    meta_path = os.path.join(output_dir, "metadata.json")
    export_metadata(checkpoint_path, meta_path)
    
    # Create JavaScript inference wrapper
    js_code = '''/**
 * AlphaZero Neural Network inference for Тоғызқұмалақ
 * Uses ONNX Runtime Web for browser inference
 */

class AlphaZeroNN {
    constructor() {
        this.session = null;
        this.metadata = null;
    }
    
    async load(modelPath = 'model.onnx', metadataPath = 'metadata.json') {
        // Load ONNX Runtime
        if (typeof ort === 'undefined') {
            throw new Error('ONNX Runtime Web not loaded. Add: <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>');
        }
        
        // Load model
        this.session = await ort.InferenceSession.create(modelPath);
        console.log('[AlphaZeroNN] Model loaded');
        
        // Load metadata
        const metaResponse = await fetch(metadataPath);
        this.metadata = await metaResponse.json();
        console.log('[AlphaZeroNN] Metadata loaded:', this.metadata);
    }
    
    /**
     * Encode game state for neural network
     * @param {Object} state - Game state {pits, kazan, tuzdyk, currentPlayer}
     * @returns {Float32Array} - Encoded state tensor
     */
    encodeState(state) {
        const PIT_NORM = 50.0;
        const KAZAN_NORM = 82.0;
        
        const player = state.currentPlayer === 'white' ? 0 : 1;
        const opponent = 1 - player;
        
        const playerPits = player === 0 ? state.pits.white : state.pits.black;
        const opponentPits = player === 0 ? state.pits.black : state.pits.white;
        const playerKazan = player === 0 ? state.kazan.white : state.kazan.black;
        const opponentKazan = player === 0 ? state.kazan.black : state.kazan.white;
        const playerTuzdyk = player === 0 ? state.tuzdyk.white : state.tuzdyk.black;
        const opponentTuzdyk = player === 0 ? state.tuzdyk.black : state.tuzdyk.white;
        
        // 7 channels x 9 positions
        const tensor = new Float32Array(7 * 9);
        
        // Channel 0: Player's pits (normalized)
        for (let i = 0; i < 9; i++) {
            tensor[0 * 9 + i] = playerPits[i] / PIT_NORM;
        }
        
        // Channel 1: Opponent's pits (normalized)
        for (let i = 0; i < 9; i++) {
            tensor[1 * 9 + i] = opponentPits[i] / PIT_NORM;
        }
        
        // Channel 2: Player's kazan (broadcast)
        for (let i = 0; i < 9; i++) {
            tensor[2 * 9 + i] = playerKazan / KAZAN_NORM;
        }
        
        // Channel 3: Opponent's kazan (broadcast)
        for (let i = 0; i < 9; i++) {
            tensor[3 * 9 + i] = opponentKazan / KAZAN_NORM;
        }
        
        // Channel 4: Player's tuzdyk (one-hot)
        if (playerTuzdyk >= 0) {
            tensor[4 * 9 + playerTuzdyk] = 1.0;
        }
        
        // Channel 5: Opponent's tuzdyk (one-hot)
        if (opponentTuzdyk >= 0) {
            tensor[5 * 9 + opponentTuzdyk] = 1.0;
        }
        
        // Channel 6: Current player indicator (white=1, black=0)
        const playerIndicator = player === 0 ? 1.0 : 0.0;
        for (let i = 0; i < 9; i++) {
            tensor[6 * 9 + i] = playerIndicator;
        }
        
        return tensor;
    }
    
    /**
     * Run inference on game state
     * @param {Object} state - Game state
     * @returns {Object} - {policy: Float32Array, value: number}
     */
    async predict(state) {
        if (!this.session) {
            throw new Error('Model not loaded. Call load() first.');
        }
        
        // Encode state
        const encoded = this.encodeState(state);
        
        // Create tensor [1, 7, 9]
        const inputTensor = new ort.Tensor('float32', encoded, [1, 7, 9]);
        
        // Run inference
        const results = await this.session.run({ state: inputTensor });
        
        // Extract outputs
        const logPolicy = results.policy.data;
        const value = results.value.data[0];
        
        // Convert log policy to probabilities
        const policy = new Float32Array(9);
        let maxLogP = -Infinity;
        for (let i = 0; i < 9; i++) {
            if (logPolicy[i] > maxLogP) maxLogP = logPolicy[i];
        }
        
        let sumExp = 0;
        for (let i = 0; i < 9; i++) {
            policy[i] = Math.exp(logPolicy[i] - maxLogP);
            sumExp += policy[i];
        }
        for (let i = 0; i < 9; i++) {
            policy[i] /= sumExp;
        }
        
        return { policy, value };
    }
    
    /**
     * Get best move from policy
     * @param {Float32Array} policy - Policy distribution
     * @param {Array<number>} validMoves - List of valid move indices
     * @returns {number} - Best move index
     */
    getBestMove(policy, validMoves) {
        let bestMove = validMoves[0];
        let bestProb = -Infinity;
        
        for (const move of validMoves) {
            if (policy[move] > bestProb) {
                bestProb = policy[move];
                bestMove = move;
            }
        }
        
        return bestMove;
    }
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { AlphaZeroNN };
}
'''
    
    js_path = os.path.join(output_dir, "alphazero-inference.js")
    with open(js_path, 'w') as f:
        f.write(js_code)
    
    print(f"\nBrowser bundle created in: {output_dir}")
    print(f"  - model.onnx")
    print(f"  - metadata.json")
    print(f"  - alphazero-inference.js")
    print("\nTo use in browser:")
    print('  1. Add: <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>')
    print('  2. Add: <script src="alphazero-inference.js"></script>')
    print('  3. Use:')
    print('     const nn = new AlphaZeroNN();')
    print('     await nn.load();')
    print('     const {policy, value} = await nn.predict(gameState);')


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Export AlphaZero model")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--output", type=str, default="browser_model",
                        help="Output directory")
    parser.add_argument("--model-size", type=str, default="medium",
                        choices=["small", "medium", "large"])
    
    args = parser.parse_args()
    
    create_browser_bundle(args.checkpoint, args.output, args.model_size)


if __name__ == "__main__":
    main()

