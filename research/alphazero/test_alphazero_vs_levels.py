"""
Test AlphaZero against different AI levels
Uses correct BatchMCTS for AlphaZero
"""

import torch
import numpy as np
from game import TogyzQumalaq, GameState, Player
from model import create_model
from play import load_model
from train_fast import TrueBatchMCTS


class SimpleMinimaxAI:
    """Simple minimax for comparison"""
    def __init__(self, depth=6):
        self.depth = depth
    
    def evaluate(self, state, player):
        """Simple evaluation function"""
        opponent = 1 - player
        score = 0
        
        # Kazan difference
        score += (state.kazan[player] - state.kazan[opponent]) * 10
        
        # Tuzdyk bonus
        if state.tuzdyk[player] != -1:
            score += 30
        if state.tuzdyk[opponent] != -1:
            score -= 30
        
        # Material (stones in pits)
        for i in range(9):
            score += state.pits[player][i] * 0.5
            score -= state.pits[opponent][i] * 0.5
        
        return score
    
    def minimax(self, game, depth, alpha, beta, maximizing, player):
        state = game.get_state()
        if depth == 0 or game.is_terminal():
            return self.evaluate(state, player)
        
        moves = game.get_valid_moves_list()
        if not moves:
            return self.evaluate(state, player)
        
        if maximizing:
            max_score = -float('inf')
            for move in moves:
                new_game = TogyzQumalaq()
                new_game.set_state(state.copy())
                new_game.make_move(move)
                score = self.minimax(new_game, depth - 1, alpha, beta, False, player)
                max_score = max(max_score, score)
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
            return max_score
        else:
            min_score = float('inf')
            for move in moves:
                new_game = TogyzQumalaq()
                new_game.set_state(state.copy())
                new_game.make_move(move)
                score = self.minimax(new_game, depth - 1, alpha, beta, True, player)
                min_score = min(min_score, score)
                beta = min(beta, score)
                if beta <= alpha:
                    break
            return min_score
    
    def get_best_move(self, game, player):
        state = game.get_state()
        moves = game.get_valid_moves_list()
        if not moves:
            return moves[0] if moves else 0
        
        best_move = moves[0]
        best_score = -float('inf')
        
        for move in moves:
            new_game = TogyzQumalaq()
            new_game.set_state(state.copy())
            new_game.make_move(move)
            score = self.minimax(new_game, self.depth - 1, -float('inf'), float('inf'), False, player)
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move


class RandomMCTSAI:
    """Simple MCTS with random playouts (no neural network)"""
    def __init__(self, simulations=10000):
        self.simulations = simulations
    
    def simulate_random(self, game):
        """Random playout"""
        sim_game = TogyzQumalaq()
        sim_game.set_state(game.get_state().copy())
        move_count = 0
        max_moves = 200
        original_player = game.get_state().current_player
        
        while not sim_game.is_terminal() and move_count < max_moves:
            moves = sim_game.get_valid_moves_list()
            if not moves:
                break
            move = np.random.choice(moves)
            sim_game.make_move(move)
            move_count += 1
        
        winner = sim_game.get_winner()
        if winner == 2:
            return 0.0
        elif winner == original_player:
            return 1.0
        else:
            return -1.0
    
    def get_best_move(self, game, player):
        """MCTS with random playouts"""
        state = game.get_state()
        visit_counts = np.zeros(9, dtype=np.int32)
        value_sums = np.zeros(9, dtype=np.float32)
        valid_moves = game.get_valid_moves_list()
        
        # Uniform prior
        prior = np.ones(9) / len(valid_moves)
        for i in range(9):
            if i not in valid_moves:
                prior[i] = 0
        
        c_puct = 1.5
        
        for _ in range(self.simulations):
            # Selection
            sqrt_total = np.sqrt(visit_counts.sum() + 1)
            q_values = np.divide(value_sums, visit_counts + 1e-8, 
                                where=visit_counts > 0, out=np.zeros(9))
            ucb = q_values + c_puct * prior * sqrt_total / (1 + visit_counts)
            ucb = np.where(np.array([i in valid_moves for i in range(9)]), ucb, -np.inf)
            action = int(np.argmax(ucb))
            
            # Simulate
            test_game = TogyzQumalaq()
            test_game.set_state(state.copy())
            test_game.make_move(action)
            
            if test_game.is_terminal():
                winner = test_game.get_winner()
                if winner == player:
                    value = 1.0
                elif winner == 2:
                    value = 0.0
                else:
                    value = -1.0
            else:
                value = -self.simulate_random(test_game)
            
            # Update
            visit_counts[action] += 1
            value_sums[action] += value
        
        # Return best move
        return int(np.argmax(visit_counts))


def play_game(ai1, ai2, ai1_name, ai2_name, ai1_plays_white=True):
    """Play a game between two AIs"""
    game = TogyzQumalaq()
    move_count = 0
    max_moves = 300
    
    while not game.is_terminal() and move_count < max_moves:
        state = game.get_state()
        current_player = state.current_player
        
        if current_player == Player.WHITE:
            if ai1_plays_white:
                move = ai1.get_best_move(game, Player.WHITE)
            else:
                move = ai2.get_best_move(game, Player.WHITE)
        else:
            if ai1_plays_white:
                move = ai2.get_best_move(game, Player.BLACK)
            else:
                move = ai1.get_best_move(game, Player.BLACK)
        
        success, _ = game.make_move(move)
        if not success:
            break
        move_count += 1
    
    winner = game.get_winner()
    
    # Determine result from ai1's perspective
    if winner == 2:  # Draw
        result = 'draw'
    elif ai1_plays_white:
        result = 'win' if winner == Player.WHITE else 'loss'
    else:
        result = 'win' if winner == Player.BLACK else 'loss'
    
    return result, move_count, game.state.kazan[0], game.state.kazan[1]


def test_alphazero_vs_levels(checkpoint_path, num_games=3):
    """Test AlphaZero against different AI levels"""
    print("=" * 70)
    print("AlphaZero vs AI Levels - Objective Test")
    print("=" * 70)
    
    # Load AlphaZero
    print(f"\nLoading AlphaZero model from {checkpoint_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(checkpoint_path, device)
    
    # Create AlphaZero AI with TrueBatchMCTS
    # Use MORE simulations for testing (vs strong opponents)
    class AlphaZeroWrapper:
        def __init__(self, model, device, simulations=800):
            self.model = model
            self.device = device
            # Create MCTS without Dirichlet noise for testing
            self.mcts = TrueBatchMCTS(model, num_simulations=simulations, device=device, use_amp=True)
            # Disable noise for testing
            self.mcts.dirichlet_eps = 0.0  # No exploration noise during testing
        
        def get_best_move(self, game, player):
            # Use search_batch with single game
            policies = self.mcts.search_batch([game])
            return int(np.argmax(policies[0]))
    
    # Use 800 simulations for stronger play (vs 200 during training)
    az_ai = AlphaZeroWrapper(model, device, simulations=800)
    
    # Define opponents matching browser AI levels
    opponents = {
        'Hard (Minimax 6)': {
            'type': 'minimax',
            'depth': 6
        },
        'Expert (MCTS 5K)': {
            'type': 'mcts',
            'simulations': 5000
        },
        'Master (MCTS 15K)': {
            'type': 'mcts',
            'simulations': 15000
        },
        'Grandmaster (MCTS 30K)': {
            'type': 'mcts',
            'simulations': 30000
        },
        'Super (MCTS 100K)': {
            'type': 'mcts',
            'simulations': 100000
        }
    }
    
    results = {}
    
    for opponent_name, opponent_config in opponents.items():
        print(f"\n{'='*70}")
        print(f"Testing AlphaZero vs {opponent_name}")
        print(f"{'='*70}")
        
        wins = 0
        losses = 0
        draws = 0
        total_moves = 0
        az_scores = []
        opp_scores = []
        
        for game_num in range(1, num_games + 1):
            # Alternate colors
            az_plays_white = (game_num % 2 == 1)
            
            # Create opponent
            if opponent_config['type'] == 'minimax':
                opponent = SimpleMinimaxAI(depth=opponent_config['depth'])
            elif opponent_config['type'] == 'mcts':
                opponent = RandomMCTSAI(simulations=opponent_config['simulations'])
            else:
                print(f"Unknown opponent type: {opponent_config['type']}")
                continue
            
            print(f"\nGame {game_num} ({'AlphaZero White' if az_plays_white else 'AlphaZero Black'})...", end=' ', flush=True)
            
            result, moves, white_score, black_score = play_game(
                az_ai, opponent, "AlphaZero", opponent_name, az_plays_white
            )
            
            total_moves += moves
            
            if az_plays_white:
                az_score = white_score
                opp_score = black_score
            else:
                az_score = black_score
                opp_score = white_score
            
            az_scores.append(az_score)
            opp_scores.append(opp_score)
            
            if result == 'win':
                wins += 1
                print(f"✅ AlphaZero wins ({moves} moves, {az_score}-{opp_score})")
            elif result == 'loss':
                losses += 1
                print(f"❌ {opponent_name} wins ({moves} moves, {az_score}-{opp_score})")
            else:
                draws += 1
                print(f"🤝 Draw ({moves} moves, {az_score}-{opp_score})")
        
        win_rate = wins / num_games * 100
        avg_moves = total_moves / num_games
        avg_az_score = np.mean(az_scores)
        avg_opp_score = np.mean(opp_scores)
        
        results[opponent_name] = {
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_rate': win_rate,
            'avg_moves': avg_moves,
            'avg_az_score': avg_az_score,
            'avg_opp_score': avg_opp_score
        }
        
        print(f"\nResults vs {opponent_name}:")
        print(f"  Wins: {wins}/{num_games} ({win_rate:.1f}%)")
        print(f"  Losses: {losses}/{num_games}")
        print(f"  Draws: {draws}/{num_games}")
        print(f"  Avg moves: {avg_moves:.1f}")
        print(f"  Avg score: AlphaZero {avg_az_score:.1f} - {avg_opp_score:.1f} Opponent")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Opponent':<25} {'Win Rate':<12} {'W-L-D':<15} {'Avg Score':<20}")
    print("-" * 70)
    
    for name, stats in results.items():
        print(f"{name:<25} {stats['win_rate']:>6.1f}%      "
              f"{stats['wins']}-{stats['losses']}-{stats['draws']:<10} "
              f"{stats['avg_az_score']:.1f}-{stats['avg_opp_score']:.1f}")
    
    print(f"\n{'='*70}")
    print("Analysis:")
    
    # Determine strength level
    if results['Hard (Minimax 6)']['win_rate'] >= 66:
        print("✅ AlphaZero beats Hard (Minimax 6) - Strong!")
    else:
        print("⚠️ AlphaZero struggles vs Hard (Minimax 6)")
    
    if results['Expert (MCTS 5K)']['win_rate'] >= 50:
        print("✅ AlphaZero competitive vs Expert (MCTS 5K)")
    else:
        print("⚠️ AlphaZero weaker than Expert (MCTS 5K)")
    
    if results['Master (MCTS 15K)']['win_rate'] >= 50:
        print("✅ AlphaZero competitive vs Master (MCTS 15K)")
    else:
        print("⚠️ AlphaZero weaker than Master (MCTS 15K)")
    
    if results['Grandmaster (MCTS 30K)']['win_rate'] >= 50:
        print("✅ AlphaZero competitive vs Grandmaster (MCTS 30K) - Very Strong!")
    else:
        print("⚠️ AlphaZero weaker than Grandmaster (MCTS 30K)")
    
    if results['Super (MCTS 100K)']['win_rate'] >= 50:
        print("✅ AlphaZero competitive vs Super (MCTS 100K) - Exceptional!")
    else:
        print("⚠️ AlphaZero weaker than Super (MCTS 100K)")
    
    print(f"{'='*70}")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test AlphaZero vs AI levels")
    parser.add_argument("--checkpoint", default="checkpoints/model_iter50.pt",
                       help="AlphaZero checkpoint path")
    parser.add_argument("--games", type=int, default=3,
                       help="Games per opponent")
    
    args = parser.parse_args()
    
    test_alphazero_vs_levels(args.checkpoint, args.games)


if __name__ == "__main__":
    main()

