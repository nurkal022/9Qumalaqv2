"""
Quick test: AlphaZero vs Random - full games
"""

import torch
import numpy as np
from game import TogyzQumalaq, Player
from model import create_model
from mcts import MCTS, MCTSConfig
from play import load_model


def play_full_game(model, mcts, verbose=False):
    """Play a full game, AlphaZero (White) vs Random (Black)"""
    game = TogyzQumalaq()
    move_count = 0
    max_moves = 300
    
    while not game.is_terminal() and move_count < max_moves:
        state = game.get_state()
        current_player = state.current_player
        
        if current_player == Player.WHITE:
            # AlphaZero
            policy = mcts.search(state, add_noise=False)
            move = int(np.argmax(policy))
            if verbose:
                print(f"Move {move_count+1}: White (AZ) pit {move+1}, policy max: {policy[move]:.2f}")
        else:
            # Random
            valid_moves = game.get_valid_moves_list()
            if not valid_moves:
                break
            move = np.random.choice(valid_moves)
            if verbose:
                print(f"Move {move_count+1}: Black (Rand) pit {move+1}")
        
        success, winner = game.make_move(move)
        if not success:
            print(f"Invalid move: {move}")
            break
        
        move_count += 1
        
        if verbose and move_count % 20 == 0:
            print(f"  Score: White {game.state.kazan[0]} - {game.state.kazan[1]} Black")
    
    winner = game.get_winner()
    return winner, move_count, game.state.kazan[0], game.state.kazan[1]


def main():
    print("Loading AlphaZero model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("checkpoints/model_iter50.pt", device)
    
    # Test with different simulation counts
    for sims in [100, 200, 400, 800]:
        print(f"\n{'='*60}")
        print(f"Testing with {sims} MCTS simulations")
        print(f"{'='*60}")
        
        config = MCTSConfig(num_simulations=sims, temperature=0)
        mcts = MCTS(model, config)
        
        az_wins = 0
        rand_wins = 0
        draws = 0
        
        for game_num in range(5):
            winner, moves, az_score, rand_score = play_full_game(model, mcts)
            
            if winner == Player.WHITE:
                az_wins += 1
                result = "AZ wins"
            elif winner == Player.BLACK:
                rand_wins += 1
                result = "Random wins"
            else:
                draws += 1
                result = "Draw"
            
            print(f"Game {game_num+1}: {result} ({moves} moves, score {az_score}-{rand_score})")
        
        print(f"\nResults: AZ {az_wins}/5, Random {rand_wins}/5, Draws {draws}/5")
    
    # Play one verbose game
    print(f"\n{'='*60}")
    print("Detailed game (800 simulations)")
    print(f"{'='*60}")
    config = MCTSConfig(num_simulations=800, temperature=0)
    mcts = MCTS(model, config)
    winner, moves, az_score, rand_score = play_full_game(model, mcts, verbose=True)
    print(f"\nFinal: Winner = {'White (AZ)' if winner == 0 else 'Black (Rand)' if winner == 1 else 'Draw'}")
    print(f"Score: {az_score} - {rand_score}")


if __name__ == "__main__":
    main()

