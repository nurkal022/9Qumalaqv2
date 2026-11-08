"""
Play against trained AlphaZero model or watch it play
"""

import torch
import numpy as np
from game import TogyzQumalaq, Player
from model import create_model
from mcts import MCTS, MCTSConfig


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load trained model from checkpoint"""
    print(f"Loading model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model size from checkpoint or default
    model_size = checkpoint.get('config', {}).get('model_size', 'medium')
    if isinstance(model_size, dict):
        model_size = model_size.get('model_size', 'medium')
    
    print(f"Model size: {model_size}")
    
    model = create_model(model_size, device)
    
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
    
    print(f"Model loaded! Iteration: {checkpoint.get('iteration', 'unknown')}")
    print(f"Total games trained: {checkpoint.get('total_games', 'unknown')}")
    
    return model


def play_ai_vs_ai(model, num_games: int = 5, simulations: int = 400):
    """Watch AI play against itself"""
    print(f"\n{'='*60}")
    print(f"AI vs AI - {num_games} games")
    print(f"{'='*60}\n")
    
    config = MCTSConfig(num_simulations=simulations, temperature=0)
    mcts = MCTS(model, config)
    
    results = {'white_wins': 0, 'black_wins': 0, 'draws': 0}
    
    for game_num in range(1, num_games + 1):
        print(f"\n--- Game {game_num} ---")
        game = TogyzQumalaq()
        move_count = 0
        max_moves = 200
        
        while not game.is_terminal() and move_count < max_moves:
            state = game.get_state()
            player_name = "White" if state.current_player == Player.WHITE else "Black"
            
            # Get AI move
            policy = mcts.search(state, add_noise=False)
            action = int(np.argmax(policy))
            
            print(f"Move {move_count + 1}: {player_name} plays pit {action + 1} "
                  f"(policy: {policy[action]:.3f})")
            
            game.make_move(action)
            move_count += 1
        
        winner = game.get_winner()
        white_score = game.state.kazan[Player.WHITE]
        black_score = game.state.kazan[Player.BLACK]
        
        if winner == Player.WHITE:
            results['white_wins'] += 1
            print(f"\n✅ White wins! ({white_score} - {black_score})")
        elif winner == Player.BLACK:
            results['black_wins'] += 1
            print(f"\n✅ Black wins! ({white_score} - {black_score})")
        else:
            results['draws'] += 1
            print(f"\n🤝 Draw! ({white_score} - {black_score})")
        
        print(f"Total moves: {move_count}")
    
    print(f"\n{'='*60}")
    print("Results:")
    print(f"  White wins: {results['white_wins']}")
    print(f"  Black wins: {results['black_wins']}")
    print(f"  Draws: {results['draws']}")
    print(f"{'='*60}")


def play_human_vs_ai(model, human_plays_white: bool = True, simulations: int = 400):
    """Play against AI"""
    print(f"\n{'='*60}")
    print("Human vs AI")
    print(f"You play: {'White' if human_plays_white else 'Black'}")
    print(f"{'='*60}\n")
    
    config = MCTSConfig(num_simulations=simulations, temperature=0)
    mcts = MCTS(model, config)
    
    game = TogyzQumalaq()
    move_count = 0
    
    while not game.is_terminal() and move_count < 200:
        state = game.get_state()
        is_human_turn = (state.current_player == Player.WHITE) == human_plays_white
        
        # Print board
        print(f"\n--- Move {move_count + 1} ---")
        print(game)
        
        if is_human_turn:
            # Human move
            valid_moves = game.get_valid_moves_list()
            print(f"\nYour turn! Valid pits: {[m+1 for m in valid_moves]}")
            
            while True:
                try:
                    pit = int(input("Enter pit number (1-9): ")) - 1
                    if pit in valid_moves:
                        break
                    else:
                        print(f"Invalid! Valid pits: {[m+1 for m in valid_moves]}")
                except ValueError:
                    print("Please enter a number!")
            
            game.make_move(pit)
            print(f"You played pit {pit + 1}")
        else:
            # AI move
            print("\nAI thinking...")
            policy = mcts.search(state, add_noise=False)
            action = int(np.argmax(policy))
            
            print(f"AI plays pit {action + 1} (confidence: {policy[action]:.3f})")
            game.make_move(action)
        
        move_count += 1
    
    # Game over
    winner = game.get_winner()
    white_score = game.state.kazan[Player.WHITE]
    black_score = game.state.kazan[Player.BLACK]
    
    print(f"\n{'='*60}")
    print("Game Over!")
    print(f"Final score: White {white_score} - {black_score} Black")
    
    if winner == 2:
        print("🤝 Draw!")
    elif (winner == Player.WHITE and human_plays_white) or \
         (winner == Player.BLACK and not human_plays_white):
        print("🎉 You win!")
    else:
        print("😢 AI wins!")
    print(f"{'='*60}")


def quick_demo(model, simulations: int = 200):
    """Quick demo - show a few moves"""
    print(f"\n{'='*60}")
    print("Quick Demo - AI playing")
    print(f"{'='*60}\n")
    
    config = MCTSConfig(num_simulations=simulations, temperature=0)
    mcts = MCTS(model, config)
    
    game = TogyzQumalaq()
    
    print("Initial position:")
    print(game)
    
    for move in range(10):
        if game.is_terminal():
            break
        
        state = game.get_state()
        player_name = "White" if state.current_player == Player.WHITE else "Black"
        
        policy = mcts.search(state, add_noise=False)
        action = int(np.argmax(policy))
        
        print(f"\nMove {move + 1}: {player_name} plays pit {action + 1}")
        game.make_move(action)
        
        if move < 5:  # Show first 5 moves
            print(game)
    
    print(f"\nAfter {move + 1} moves:")
    print(game)
    
    if game.is_terminal():
        winner = game.get_winner()
        if winner == 2:
            print("Game ended in draw!")
        else:
            print(f"Winner: {'White' if winner == Player.WHITE else 'Black'}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Play with trained AlphaZero model")
    parser.add_argument("--checkpoint", type=str, 
                       default="checkpoints/model_iter50.pt",
                       help="Path to model checkpoint")
    parser.add_argument("--mode", type=str, 
                       choices=["demo", "ai-vs-ai", "human-vs-ai"],
                       default="demo",
                       help="Play mode")
    parser.add_argument("--simulations", type=int, default=200,
                       help="MCTS simulations per move")
    parser.add_argument("--games", type=int, default=5,
                       help="Number of games (for ai-vs-ai)")
    parser.add_argument("--human-white", action="store_true",
                       help="Human plays white (for human-vs-ai)")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Play
    if args.mode == "demo":
        quick_demo(model, args.simulations)
    elif args.mode == "ai-vs-ai":
        play_ai_vs_ai(model, args.games, args.simulations)
    elif args.mode == "human-vs-ai":
        play_human_vs_ai(model, args.human_white, args.simulations)


if __name__ == "__main__":
    main()

