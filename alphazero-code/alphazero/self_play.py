"""
Self-Play for AlphaZero training
Generates training data through self-play games
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import pickle
import os
import time
from tqdm import tqdm

from game import TogyzQumalaq, GameState, Player
from mcts import MCTS, MCTSConfig, select_action


@dataclass
class TrainingExample:
    """Single training example"""
    state: np.ndarray  # Encoded board state
    policy: np.ndarray  # MCTS policy
    value: float  # Game outcome from this player's perspective


@dataclass 
class SelfPlayConfig:
    """Self-play configuration"""
    num_games: int = 100
    num_simulations: int = 800
    temperature_threshold: int = 30  # After this move, use temp=0
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    num_workers: int = 4


class SelfPlayWorker:
    """Worker for generating self-play games"""
    
    def __init__(self, model, config: SelfPlayConfig):
        self.model = model
        self.config = config
        self.mcts_config = MCTSConfig(
            num_simulations=config.num_simulations,
            c_puct=config.c_puct,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_epsilon=config.dirichlet_epsilon
        )
    
    def play_game(self) -> List[TrainingExample]:
        """
        Play a single self-play game
        
        Returns:
            List of training examples from the game
        """
        game = TogyzQumalaq()
        mcts = MCTS(self.model, self.mcts_config)
        
        examples = []
        move_count = 0
        max_moves = 300  # Prevent infinite games
        
        while not game.is_terminal() and move_count < max_moves:
            # Get canonical state for network
            state = game.get_state()
            encoded_state = game.encode_state()
            
            # Temperature: exploration early, exploitation late
            temperature = 1.0 if move_count < self.config.temperature_threshold else 0.0
            
            # MCTS search
            self.mcts_config.temperature = temperature
            policy = mcts.search(state, add_noise=True)
            
            # Store example (without value yet)
            examples.append({
                'state': encoded_state.copy(),
                'policy': policy.copy(),
                'current_player': state.current_player
            })
            
            # Select and make move
            if temperature > 0:
                action = select_action(policy, temperature)
            else:
                action = int(np.argmax(policy))
            
            game.make_move(action)
            move_count += 1
        
        # Get game result
        winner = game.get_winner()
        
        # Assign values based on game outcome
        training_examples = []
        for ex in examples:
            if winner == 2:  # Draw
                value = 0.0
            elif winner == ex['current_player']:
                value = 1.0  # Win
            else:
                value = -1.0  # Loss
            
            training_examples.append(TrainingExample(
                state=ex['state'],
                policy=ex['policy'],
                value=value
            ))
        
        return training_examples
    
    def play_games(self, num_games: int, progress_bar: bool = True) -> List[TrainingExample]:
        """
        Play multiple self-play games
        
        Args:
            num_games: Number of games to play
            progress_bar: Show progress bar
        
        Returns:
            Combined training examples from all games
        """
        all_examples = []
        
        iterator = range(num_games)
        if progress_bar:
            iterator = tqdm(iterator, desc="Self-play games")
        
        for _ in iterator:
            examples = self.play_game()
            all_examples.extend(examples)
        
        return all_examples


class SelfPlayManager:
    """
    Manages self-play data generation
    Can use multiple processes for speed
    """
    
    def __init__(self, model_factory, config: SelfPlayConfig):
        """
        Args:
            model_factory: Function that creates a model instance
            config: Self-play configuration
        """
        self.model_factory = model_factory
        self.config = config
        self.examples_buffer: List[TrainingExample] = []
        self.buffer_max_size: int = 500000  # Max examples in buffer
    
    def generate_examples(self, model_state_dict: dict) -> List[TrainingExample]:
        """
        Generate training examples through self-play
        
        Args:
            model_state_dict: Model weights to use
        
        Returns:
            List of training examples
        """
        import torch
        
        # Create model and load weights
        model = self.model_factory()
        model.load_state_dict(model_state_dict)
        model.eval()
        
        # Single-threaded for GPU model
        worker = SelfPlayWorker(model, self.config)
        examples = worker.play_games(self.config.num_games)
        
        return examples
    
    def add_examples(self, examples: List[TrainingExample]):
        """Add examples to buffer, removing old ones if necessary"""
        self.examples_buffer.extend(examples)
        
        # Remove oldest examples if buffer is full
        if len(self.examples_buffer) > self.buffer_max_size:
            excess = len(self.examples_buffer) - self.buffer_max_size
            self.examples_buffer = self.examples_buffer[excess:]
    
    def get_training_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get a random batch of training examples
        
        Returns:
            states, policies, values as numpy arrays
        """
        if len(self.examples_buffer) < batch_size:
            batch_size = len(self.examples_buffer)
        
        indices = np.random.choice(len(self.examples_buffer), batch_size, replace=False)
        
        states = np.array([self.examples_buffer[i].state for i in indices])
        policies = np.array([self.examples_buffer[i].policy for i in indices])
        values = np.array([self.examples_buffer[i].value for i in indices])
        
        return states, policies, values
    
    def save_buffer(self, path: str):
        """Save examples buffer to file"""
        with open(path, 'wb') as f:
            pickle.dump(self.examples_buffer, f)
    
    def load_buffer(self, path: str):
        """Load examples buffer from file"""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.examples_buffer = pickle.load(f)
    
    def buffer_size(self) -> int:
        return len(self.examples_buffer)


def play_game_vs_random(model, mcts_config: MCTSConfig = None, model_plays_white: bool = True) -> Tuple[int, int]:
    """
    Play a game: model vs random player
    
    Returns:
        (winner, num_moves)
    """
    if mcts_config is None:
        mcts_config = MCTSConfig(num_simulations=400)
    
    game = TogyzQumalaq()
    mcts = MCTS(model, mcts_config)
    
    move_count = 0
    model_player = Player.WHITE if model_plays_white else Player.BLACK
    
    while not game.is_terminal() and move_count < 300:
        state = game.get_state()
        
        if state.current_player == model_player:
            # Model's turn
            mcts_config.temperature = 0  # Deterministic for evaluation
            policy = mcts.search(state, add_noise=False)
            action = int(np.argmax(policy))
        else:
            # Random player's turn
            valid_moves = game.get_valid_moves_list()
            action = np.random.choice(valid_moves)
        
        game.make_move(action)
        move_count += 1
    
    winner = game.get_winner()
    return winner, move_count


def evaluate_model(model, num_games: int = 50) -> dict:
    """
    Evaluate model strength vs random player
    
    Returns:
        Dictionary with win/loss/draw statistics
    """
    config = MCTSConfig(num_simulations=200)  # Faster for evaluation
    
    results = {'model_wins': 0, 'random_wins': 0, 'draws': 0, 'total_moves': 0}
    
    for i in tqdm(range(num_games), desc="Evaluation"):
        # Alternate colors
        model_plays_white = (i % 2 == 0)
        model_player = Player.WHITE if model_plays_white else Player.BLACK
        
        winner, moves = play_game_vs_random(model, config, model_plays_white)
        results['total_moves'] += moves
        
        if winner == 2:
            results['draws'] += 1
        elif winner == model_player:
            results['model_wins'] += 1
        else:
            results['random_wins'] += 1
    
    results['win_rate'] = results['model_wins'] / num_games
    results['avg_moves'] = results['total_moves'] / num_games
    
    return results


if __name__ == "__main__":
    from model import create_model
    import torch
    
    print("Testing Self-Play...")
    
    # Create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = create_model("small", device=device)
    
    # Quick self-play test
    config = SelfPlayConfig(
        num_games=3,
        num_simulations=50  # Small for testing
    )
    
    worker = SelfPlayWorker(model, config)
    
    print("\nPlaying 3 test games...")
    start = time.time()
    examples = worker.play_games(3, progress_bar=True)
    elapsed = time.time() - start
    
    print(f"\nGenerated {len(examples)} training examples in {elapsed:.1f}s")
    print(f"Examples per second: {len(examples)/elapsed:.1f}")
    
    # Show example
    ex = examples[0]
    print(f"\nExample 0:")
    print(f"  State shape: {ex.state.shape}")
    print(f"  Policy: {ex.policy}")
    print(f"  Value: {ex.value}")
    
    # Quick evaluation
    print("\nEvaluating vs random (5 games)...")
    results = evaluate_model(model, num_games=5)
    print(f"Results: {results}")
    
    print("\nSelf-play tests passed!")

