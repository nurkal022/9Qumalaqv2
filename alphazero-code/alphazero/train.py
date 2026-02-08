"""
AlphaZero Training Pipeline for Тоғызқұмалақ
Main training loop with self-play and neural network updates
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import json
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm

from game import TogyzQumalaq
from model import create_model, TogyzNet, TogyzNetLarge, count_parameters
from mcts import MCTS, MCTSConfig
from self_play import SelfPlayWorker, SelfPlayConfig, SelfPlayManager, evaluate_model


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Model
    model_size: str = "medium"  # "small", "medium", "large"
    
    # Self-play
    games_per_iteration: int = 100
    num_simulations: int = 800
    temperature_threshold: int = 30
    
    # Training
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    num_epochs: int = 10
    
    # Buffer
    buffer_size: int = 200000
    min_buffer_size: int = 5000  # Min examples before training
    
    # Iterations
    num_iterations: int = 100
    eval_interval: int = 5
    save_interval: int = 10
    
    # Evaluation
    eval_games: int = 50
    
    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


class AlphaZeroTrainer:
    """
    AlphaZero training manager
    Coordinates self-play and neural network training
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Using device: {self.device}")
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Create model
        self.model = create_model(config.model_size, self.device)
        print(f"Model size: {config.model_size}")
        print(f"Parameters: {count_parameters(self.model):,}")
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_iterations,
            eta_min=config.learning_rate * 0.01
        )
        
        # Self-play config
        self.self_play_config = SelfPlayConfig(
            num_games=config.games_per_iteration,
            num_simulations=config.num_simulations,
            temperature_threshold=config.temperature_threshold
        )
        
        # Training buffer
        self.buffer_states = []
        self.buffer_policies = []
        self.buffer_values = []
        
        # Statistics
        self.iteration = 0
        self.total_games = 0
        self.training_history = []
        
        # Create directories
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
    
    def self_play(self) -> int:
        """
        Generate training data through self-play
        
        Returns:
            Number of examples generated
        """
        self.model.eval()
        
        worker = SelfPlayWorker(self.model, self.self_play_config)
        examples = worker.play_games(self.config.games_per_iteration, progress_bar=True)
        
        # Add to buffer
        for ex in examples:
            self.buffer_states.append(ex.state)
            self.buffer_policies.append(ex.policy)
            self.buffer_values.append(ex.value)
        
        # Limit buffer size
        if len(self.buffer_states) > self.config.buffer_size:
            excess = len(self.buffer_states) - self.config.buffer_size
            self.buffer_states = self.buffer_states[excess:]
            self.buffer_policies = self.buffer_policies[excess:]
            self.buffer_values = self.buffer_values[excess:]
        
        self.total_games += self.config.games_per_iteration
        
        return len(examples)
    
    def train_epoch(self) -> dict:
        """
        Train for one epoch on buffer data
        
        Returns:
            Training metrics
        """
        self.model.train()
        
        n = len(self.buffer_states)
        indices = np.random.permutation(n)
        
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        num_batches = 0
        
        for start in range(0, n, self.config.batch_size):
            end = min(start + self.config.batch_size, n)
            batch_indices = indices[start:end]
            
            # Prepare batch
            states = torch.FloatTensor(
                np.array([self.buffer_states[i] for i in batch_indices])
            ).to(self.device)
            
            target_policies = torch.FloatTensor(
                np.array([self.buffer_policies[i] for i in batch_indices])
            ).to(self.device)
            
            target_values = torch.FloatTensor(
                np.array([self.buffer_values[i] for i in batch_indices])
            ).unsqueeze(1).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            log_policies, values = self.model(states)
            
            # Policy loss (cross-entropy)
            policy_loss = -torch.mean(torch.sum(target_policies * log_policies, dim=1))
            
            # Value loss (MSE)
            value_loss = torch.mean((values - target_values) ** 2)
            
            # Combined loss
            loss = policy_loss + value_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'policy_loss': total_policy_loss / num_batches,
            'value_loss': total_value_loss / num_batches
        }
    
    def train_iteration(self) -> dict:
        """
        Full training iteration: self-play + training epochs
        
        Returns:
            Iteration metrics
        """
        self.iteration += 1
        print(f"\n{'='*60}")
        print(f"Iteration {self.iteration}/{self.config.num_iterations}")
        print(f"{'='*60}")
        
        # Self-play
        print("\n[Self-Play]")
        start_time = time.time()
        num_examples = self.self_play()
        self_play_time = time.time() - start_time
        
        print(f"Generated {num_examples} examples in {self_play_time:.1f}s")
        print(f"Buffer size: {len(self.buffer_states)}")
        print(f"Total games: {self.total_games}")
        
        # Wait for minimum buffer size
        if len(self.buffer_states) < self.config.min_buffer_size:
            print(f"Waiting for buffer ({len(self.buffer_states)}/{self.config.min_buffer_size})...")
            return {'self_play_time': self_play_time, 'examples': num_examples}
        
        # Training
        print("\n[Training]")
        start_time = time.time()
        
        epoch_metrics = []
        for epoch in range(self.config.num_epochs):
            metrics = self.train_epoch()
            epoch_metrics.append(metrics)
            print(f"  Epoch {epoch+1}: loss={metrics['loss']:.4f} "
                  f"(policy={metrics['policy_loss']:.4f}, value={metrics['value_loss']:.4f})")
        
        training_time = time.time() - start_time
        
        # Update learning rate
        self.scheduler.step()
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # Average metrics
        avg_metrics = {
            'loss': np.mean([m['loss'] for m in epoch_metrics]),
            'policy_loss': np.mean([m['policy_loss'] for m in epoch_metrics]),
            'value_loss': np.mean([m['value_loss'] for m in epoch_metrics]),
            'self_play_time': self_play_time,
            'training_time': training_time,
            'examples': num_examples,
            'buffer_size': len(self.buffer_states),
            'learning_rate': current_lr,
            'iteration': self.iteration
        }
        
        print(f"\nTraining time: {training_time:.1f}s")
        print(f"Learning rate: {current_lr:.6f}")
        
        # Evaluation
        if self.iteration % self.config.eval_interval == 0:
            print("\n[Evaluation vs Random]")
            eval_results = evaluate_model(self.model, num_games=self.config.eval_games)
            avg_metrics['eval_win_rate'] = eval_results['win_rate']
            print(f"Win rate: {eval_results['win_rate']*100:.1f}%")
            print(f"Model wins: {eval_results['model_wins']}, "
                  f"Random wins: {eval_results['random_wins']}, "
                  f"Draws: {eval_results['draws']}")
        
        # Save checkpoint
        if self.iteration % self.config.save_interval == 0:
            self.save_checkpoint()
        
        self.training_history.append(avg_metrics)
        
        return avg_metrics
    
    def train(self):
        """Run full training loop"""
        print("\n" + "="*60)
        print("Starting AlphaZero Training for Тоғызқұмалақ")
        print("="*60)
        print(f"\nConfig:")
        for key, value in asdict(self.config).items():
            print(f"  {key}: {value}")
        
        start_time = time.time()
        
        try:
            for _ in range(self.config.num_iterations):
                self.train_iteration()
                
                # Auto-save training history
                self.save_history()
        
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
        
        finally:
            total_time = time.time() - start_time
            print(f"\n{'='*60}")
            print(f"Training completed!")
            print(f"Total time: {total_time/3600:.1f} hours")
            print(f"Total iterations: {self.iteration}")
            print(f"Total games: {self.total_games}")
            print(f"{'='*60}")
            
            # Save final checkpoint
            self.save_checkpoint(final=True)
            self.save_history()
    
    def save_checkpoint(self, final: bool = False):
        """Save model checkpoint"""
        suffix = "final" if final else f"iter{self.iteration}"
        path = os.path.join(self.config.checkpoint_dir, f"model_{suffix}.pt")
        
        checkpoint = {
            'iteration': self.iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': asdict(self.config),
            'total_games': self.total_games
        }
        
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
        
        # Also save latest
        latest_path = os.path.join(self.config.checkpoint_dir, "model_latest.pt")
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.iteration = checkpoint['iteration']
        self.total_games = checkpoint.get('total_games', 0)
        
        print(f"Loaded checkpoint: {path}")
        print(f"Resuming from iteration {self.iteration}")
    
    def save_history(self):
        """Save training history to JSON"""
        path = os.path.join(self.config.log_dir, "training_history.json")
        with open(path, 'w') as f:
            json.dump(self.training_history, f, indent=2)


def main():
    """Main training entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train AlphaZero for Тоғызқұмалақ")
    parser.add_argument("--model-size", type=str, default="medium",
                        choices=["small", "medium", "large"])
    parser.add_argument("--games", type=int, default=100,
                        help="Games per iteration")
    parser.add_argument("--simulations", type=int, default=800,
                        help="MCTS simulations per move")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of training iterations")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        model_size=args.model_size,
        games_per_iteration=args.games,
        num_simulations=args.simulations,
        num_iterations=args.iterations,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir
    )
    
    trainer = AlphaZeroTrainer(config)
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    trainer.train()


if __name__ == "__main__":
    main()

