"""
Test checkpoint loading to ensure everything works correctly
"""

import torch
from train_fast import FastTrainer, FastConfig

def test_checkpoint_load():
    """Test loading checkpoint"""
    print("=" * 60)
    print("Testing Checkpoint Loading")
    print("=" * 60)
    
    checkpoint_path = "checkpoints/model_iter50.pt"
    
    # Create config
    config = FastConfig(
        model_size="medium",
        games_per_iteration=100,
        num_simulations=200,
        num_iterations=100,
        batch_size=512
    )
    
    # Create trainer
    print("\n1. Creating trainer...")
    trainer = FastTrainer(config)
    
    # Load checkpoint
    print("\n2. Loading checkpoint...")
    try:
        trainer.load_checkpoint(checkpoint_path)
        print("✅ Checkpoint loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Verify loaded state
    print("\n3. Verifying loaded state...")
    print(f"   Iteration: {trainer.iteration} (expected: 50)")
    print(f"   Total games: {trainer.total_games} (expected: 5000)")
    
    if trainer.iteration == 50:
        print("   ✅ Iteration matches")
    else:
        print(f"   ⚠️ Iteration mismatch (got {trainer.iteration}, expected 50)")
    
    # Test model forward pass
    print("\n4. Testing model forward pass...")
    try:
        from game import TogyzQumalaq
        game = TogyzQumalaq()
        encoded = game.encode_state()
        x = torch.FloatTensor(encoded).unsqueeze(0).to(trainer.device)
        
        trainer.model.eval()
        with torch.no_grad():
            log_policy, value = trainer.model(x)
        
        print(f"   ✅ Model forward pass works")
        print(f"   Policy shape: {log_policy.shape}, Value: {value.item():.4f}")
    except Exception as e:
        print(f"   ❌ Model forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Ready to continue training.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_checkpoint_load()
    exit(0 if success else 1)

