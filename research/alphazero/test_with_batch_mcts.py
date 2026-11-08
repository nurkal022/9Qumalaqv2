"""
Test AlphaZero using the same BatchMCTS as training
"""

import torch
import numpy as np
from game import TogyzQumalaq, Player
from play import load_model

# Import BatchMCTS from train_fast
import sys
sys.path.insert(0, '.')
from train_fast import BatchMCTS


def play_game(model, device, num_simulations=200, verbose=False):
    """Play AlphaZero (White) vs Random (Black)"""
    game = TogyzQumalaq()
    mcts = BatchMCTS(model, num_simulations=num_simulations, device=device)
    
    move_count = 0
    
    while not game.is_terminal() and move_count < 300:
        if game.state.current_player == Player.WHITE:
            # AlphaZero
            policy = mcts.search_single(game)
            move = int(np.argmax(policy))
            if verbose:
                print(f"Move {move_count+1}: White (AZ) pit {move+1}")
        else:
            # Random
            valid = game.get_valid_moves_list()
            if not valid:
                break
            move = np.random.choice(valid)
            if verbose:
                print(f"Move {move_count+1}: Black (Rand) pit {move+1}")
        
        game.make_move(move)
        move_count += 1
        
        if verbose and move_count % 50 == 0:
            print(f"  Score: {game.state.kazan[0]}-{game.state.kazan[1]}")
    
    winner = game.get_winner()
    return winner, move_count, game.state.kazan[0], game.state.kazan[1]


def main():
    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("checkpoints/model_iter50.pt", device)
    
    print(f"\nTesting with BatchMCTS (same as training)")
    
    for sims in [100, 200, 400]:
        print(f"\n{'='*50}")
        print(f"Simulations: {sims}")
        print(f"{'='*50}")
        
        az_wins = 0
        rand_wins = 0
        draws = 0
        
        for game_num in range(10):
            winner, moves, az_score, rand_score = play_game(model, device, sims)
            
            if winner == Player.WHITE:
                az_wins += 1
                result = "AZ wins"
            elif winner == Player.BLACK:
                rand_wins += 1
                result = "Rand wins"
            else:
                draws += 1
                result = "Draw"
            
            print(f"Game {game_num+1}: {result} ({moves} moves, {az_score}-{rand_score})")
        
        print(f"\nResults: AZ {az_wins}/10, Random {rand_wins}/10, Draws {draws}/10")
        win_rate = (az_wins + 0.5 * draws) / 10 * 100
        print(f"Win rate: {win_rate:.1f}%")


if __name__ == "__main__":
    main()

