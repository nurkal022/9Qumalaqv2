#!/usr/bin/env python
"""
Test script to verify AlphaZero setup is working correctly
"""

import sys
import time


def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        import numpy as np
        print(f"  ✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"  ✗ NumPy: {e}")
        return False
    
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"    CUDA: {torch.version.cuda}")
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
            print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("    ⚠ CUDA not available")
    except ImportError as e:
        print(f"  ✗ PyTorch: {e}")
        return False
    
    try:
        from tqdm import tqdm
        print("  ✓ tqdm")
    except ImportError as e:
        print(f"  ✗ tqdm: {e}")
        return False
    
    return True


def test_game():
    """Test game logic"""
    print("\nTesting game logic...")
    
    try:
        from game import TogyzQumalaq
        
        game = TogyzQumalaq()
        print(f"  ✓ Game initialized")
        
        # Make some moves
        moves = 0
        while not game.is_terminal() and moves < 20:
            valid = game.get_valid_moves_list()
            if not valid:
                break
            import numpy as np
            move = np.random.choice(valid)
            game.make_move(move)
            moves += 1
        
        print(f"  ✓ Made {moves} moves successfully")
        print(f"  ✓ Board encoding shape: {game.encode_state().shape}")
        return True
    
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_model():
    """Test neural network"""
    print("\nTesting neural network...")
    
    try:
        import torch
        from model import create_model, count_parameters
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = create_model("small", device)
        params = count_parameters(model)
        print(f"  ✓ Small model created ({params:,} params)")
        
        # Test forward pass
        x = torch.randn(4, 7, 9).to(device)
        policy, value = model(x)
        print(f"  ✓ Forward pass: policy {policy.shape}, value {value.shape}")
        
        # Test predict
        import numpy as np
        state = np.random.randn(7, 9).astype(np.float32)
        p, v = model.predict(state)
        print(f"  ✓ Predict: policy sum={p.sum():.3f}, value={v:.3f}")
        
        return True
    
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mcts():
    """Test MCTS"""
    print("\nTesting MCTS...")
    
    try:
        import torch
        from game import TogyzQumalaq
        from model import create_model
        from mcts import MCTS, MCTSConfig
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = create_model("small", device)
        
        config = MCTSConfig(num_simulations=50)  # Small for testing
        mcts = MCTS(model, config)
        
        game = TogyzQumalaq()
        state = game.get_state()
        
        print("  Running MCTS (50 simulations)...")
        start = time.time()
        policy = mcts.search(state)
        elapsed = time.time() - start
        
        print(f"  ✓ MCTS completed in {elapsed:.2f}s")
        print(f"  ✓ Policy: {policy}")
        print(f"  ✓ Best move: pit {policy.argmax() + 1}")
        
        return True
    
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_self_play():
    """Test self-play"""
    print("\nTesting self-play...")
    
    try:
        import torch
        from model import create_model
        from self_play import SelfPlayWorker, SelfPlayConfig
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = create_model("small", device)
        
        config = SelfPlayConfig(
            num_games=1,
            num_simulations=30  # Very small for testing
        )
        
        worker = SelfPlayWorker(model, config)
        
        print("  Playing 1 test game...")
        start = time.time()
        examples = worker.play_game()
        elapsed = time.time() - start
        
        print(f"  ✓ Generated {len(examples)} examples in {elapsed:.1f}s")
        print(f"  ✓ First example: state shape={examples[0].state.shape}, value={examples[0].value}")
        
        return True
    
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 50)
    print("AlphaZero Тоғызқұмалақ - Setup Test")
    print("=" * 50)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Game Logic", test_game()))
    results.append(("Neural Network", test_model()))
    results.append(("MCTS", test_mcts()))
    results.append(("Self-Play", test_self_play()))
    
    print("\n" + "=" * 50)
    print("Results:")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    
    if all_passed:
        print("\n🎉 All tests passed! Ready for training.")
        print("\nRun training with:")
        print("  python train.py --model-size medium --games 100 --simulations 800")
        return 0
    else:
        print("\n❌ Some tests failed. Please fix issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

