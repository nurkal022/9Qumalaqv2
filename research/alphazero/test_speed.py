"""
Test speed improvements of optimized training
"""

import torch
import time
import numpy as np
from train_fast import FastTrainer, FastConfig, TrueBatchMCTS, ParallelSelfPlay
from game import TogyzQumalaq

def test_batch_mcts_speed():
    """Test batch MCTS speed"""
    print("=" * 60)
    print("Testing Batch MCTS Speed")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Create model
    from model import create_model
    model = create_model("medium", device)
    model.eval()
    
    # Test configurations
    configs = [
        {"batch_size": 8, "simulations": 100},
        {"batch_size": 16, "simulations": 100},
        {"batch_size": 32, "simulations": 100},
        {"batch_size": 32, "simulations": 200},
    ]
    
    for config in configs:
        batch_size = config["batch_size"]
        simulations = config["simulations"]
        
        print(f"\nTesting: batch_size={batch_size}, simulations={simulations}")
        
        mcts = TrueBatchMCTS(model, num_simulations=simulations, device=device, use_amp=True)
        
        # Create test games
        games = [TogyzQumalaq() for _ in range(batch_size)]
        
        # Warmup
        _ = mcts.search_batch(games[:min(4, batch_size)])
        torch.cuda.synchronize() if device == "cuda" else None
        
        # Time it
        start = time.time()
        for _ in range(10):
            policies = mcts.search_batch(games)
        torch.cuda.synchronize() if device == "cuda" else None
        elapsed = time.time() - start
        
        avg_time = elapsed / 10
        searches_per_sec = batch_size / avg_time
        
        print(f"  Time: {avg_time*1000:.1f}ms per batch")
        print(f"  Throughput: {searches_per_sec:.1f} searches/sec")
        print(f"  Per game: {avg_time/batch_size*1000:.1f}ms")


def test_self_play_speed():
    """Test self-play speed"""
    print("\n" + "=" * 60)
    print("Testing Self-Play Speed")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = FastConfig(
        games_per_iteration=32,
        num_simulations=100,
        batch_size_games=16
    )
    
    from model import create_model
    model = create_model("medium", device)
    model.eval()
    
    player = ParallelSelfPlay(model, config, device)
    
    print(f"Playing {config.games_per_iteration} games with batch_size={config.batch_size_games}")
    
    start = time.time()
    examples = player.play_games(config.games_per_iteration)
    elapsed = time.time() - start
    
    games_per_sec = config.games_per_iteration / elapsed
    examples_per_sec = len(examples) / elapsed
    
    print(f"\nResults:")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Games/sec: {games_per_sec:.2f}")
    print(f"  Examples/sec: {examples_per_sec:.1f}")
    print(f"  Total examples: {len(examples)}")


def test_training_speed():
    """Test training speed"""
    print("\n" + "=" * 60)
    print("Testing Training Speed")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = FastConfig(
        batch_size=1024,
        num_epochs=1
    )
    
    trainer = FastTrainer(config)
    
    # Fill buffer
    print("Filling buffer...")
    for _ in range(5):
        trainer.self_play()
    
    buffer_size = len(trainer.buffer_states)
    print(f"Buffer size: {buffer_size}")
    
    # Time training
    print("\nTraining one epoch...")
    start = time.time()
    metrics = trainer.train_epoch()
    elapsed = time.time() - start
    
    samples_per_sec = buffer_size / elapsed
    
    print(f"\nResults:")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Samples/sec: {samples_per_sec:.0f}")
    print(f"  Loss: {metrics['loss']:.4f}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SPEED TEST - Optimized AlphaZero")
    print("=" * 60)
    
    test_batch_mcts_speed()
    test_self_play_speed()
    test_training_speed()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)

