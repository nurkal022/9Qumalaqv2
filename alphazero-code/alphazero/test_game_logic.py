"""
Test game logic step by step
"""

import numpy as np
from game import TogyzQumalaq, GameState, Player


def test_single_stone_move():
    """Test moving a single stone"""
    print("=" * 60)
    print("Test: Single stone move from pit 1")
    print("=" * 60)
    
    game = TogyzQumalaq()
    # Set up: only 1 stone in pit 1, rest empty
    game.state.pits[Player.WHITE] = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
    game.state.pits[Player.BLACK] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
    
    print(f"Before move:")
    print(f"  White pits: {game.state.pits[Player.WHITE]}")
    print(f"  Black pits: {game.state.pits[Player.BLACK]}")
    print(f"  Current player: {'White' if game.state.current_player == 0 else 'Black'}")
    
    game.make_move(0)  # Play pit 1 (index 0)
    
    print(f"\nAfter move (pit 1):")
    print(f"  White pits: {game.state.pits[Player.WHITE]}")
    print(f"  Black pits: {game.state.pits[Player.BLACK]}")
    print(f"  White kazan: {game.state.kazan[Player.WHITE]}")
    print(f"  Current player: {'White' if game.state.current_player == 0 else 'Black'}")
    
    # With 1 stone from pit 1, it should go to pit 2 (index 1) on White's side
    expected_white = [0, 1, 0, 0, 0, 0, 0, 0, 0]
    if list(game.state.pits[Player.WHITE]) == expected_white:
        print("  ✅ Correct!")
    else:
        print(f"  ❌ Expected white pits: {expected_white}")


def test_multi_stone_move():
    """Test moving multiple stones"""
    print("\n" + "=" * 60)
    print("Test: 9 stones from pit 1 crossing to opponent's side")
    print("=" * 60)
    
    game = TogyzQumalaq()
    # Standard starting position but only pit 1 has stones
    game.state.pits[Player.WHITE] = np.array([9, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
    game.state.pits[Player.BLACK] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
    
    print(f"Before move:")
    print(f"  White pits: {game.state.pits[Player.WHITE]}")
    
    game.make_move(0)  # Play pit 1
    
    print(f"\nAfter move (pit 1 with 9 stones):")
    print(f"  White pits: {game.state.pits[Player.WHITE]}")
    print(f"  Black pits: {game.state.pits[Player.BLACK]}")
    print(f"  White kazan: {game.state.kazan[Player.WHITE]}")
    
    # 9 stones from pit 1:
    # First stone stays in pit 1, then distribute to pits 2-9
    # That's only 8 more pits on white side, so last stone goes to black's pit 1
    expected_white = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    expected_black = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # No stones should reach black yet
    
    # Wait, let me recalculate:
    # Start: pit 1 has 9 stones
    # Rule: first stone stays in starting pit, then distribute
    # pit 1 (idx 0) -> pit 2 (idx 1) -> ... -> pit 9 (idx 8) -> black pit 1 (idx 0)
    # That's 9 positions total, starting from pit 1 itself
    # So: 1 stays at pit 1, 8 go to pits 2-9
    # No, wait - the code says first stone stays: pits[player, pit_index] += 1, stones -= 1
    # Then while stones > 0, move to next pit
    
    # Actually looking at code (line 124-143):
    # If stones > 1:
    #   - First stone stays at starting pit
    #   - Then distribute remaining stones counterclockwise
    # So 9 stones: 1 stays, 8 distributed
    # From pit 1 (idx 0): pit 2, 3, 4, 5, 6, 7, 8, 9 = 8 pits = perfect fit
    print(f"  Expected white: [1, 1, 1, 1, 1, 1, 1, 1, 1]")


def test_capture():
    """Test capture mechanism"""
    print("\n" + "=" * 60)
    print("Test: Capture (landing on opponent's pit with even number)")
    print("=" * 60)
    
    game = TogyzQumalaq()
    # Set up a capture scenario:
    # White plays pit 9 (index 8) with 1 stone
    # It should land on Black's pit 1 (index 0)
    # If Black's pit 1 has odd number, it becomes even -> capture
    game.state.pits[Player.WHITE] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.int32)
    game.state.pits[Player.BLACK] = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)  # 1 stone in pit 1
    
    print(f"Before move:")
    print(f"  White pits: {game.state.pits[Player.WHITE]}")
    print(f"  Black pits: {game.state.pits[Player.BLACK]}")
    print(f"  Black pit 1 has 1 stone, White plays pit 9 (1 stone)")
    print(f"  Stone should land on Black pit 1, making it 2 (even) -> capture")
    
    game.make_move(8)  # Play pit 9 (index 8)
    
    print(f"\nAfter move:")
    print(f"  White pits: {game.state.pits[Player.WHITE]}")
    print(f"  Black pits: {game.state.pits[Player.BLACK]}")
    print(f"  White kazan: {game.state.kazan[Player.WHITE]}")
    
    if game.state.kazan[Player.WHITE] == 2:
        print("  ✅ Capture worked!")
    else:
        print("  ❌ Capture failed!")


def test_tuzdyk():
    """Test tuzdyk creation"""
    print("\n" + "=" * 60)
    print("Test: Tuzdyk creation (landing with exactly 3 stones)")
    print("=" * 60)
    
    game = TogyzQumalaq()
    # Set up: White plays pit 9 (index 8) with 1 stone
    # Black pit 1 has 2 stones, so 2 + 1 = 3 -> tuzdyk
    game.state.pits[Player.WHITE] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.int32)
    game.state.pits[Player.BLACK] = np.array([2, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
    
    print(f"Before move:")
    print(f"  White pits: {game.state.pits[Player.WHITE]}")
    print(f"  Black pits: {game.state.pits[Player.BLACK]}")
    print(f"  White tuzdyk: {game.state.tuzdyk[Player.WHITE]}")
    print(f"  Black pit 1 has 2, White plays pit 9 -> 2+1=3 -> tuzdyk")
    
    game.make_move(8)  # Play pit 9
    
    print(f"\nAfter move:")
    print(f"  White pits: {game.state.pits[Player.WHITE]}")
    print(f"  Black pits: {game.state.pits[Player.BLACK]}")
    print(f"  White tuzdyk: {game.state.tuzdyk[Player.WHITE]}")
    print(f"  White kazan: {game.state.kazan[Player.WHITE]}")
    
    if game.state.tuzdyk[Player.WHITE] == 0:  # index 0 = pit 1
        print("  ✅ Tuzdyk created at Black's pit 1!")
    else:
        print("  ❌ Tuzdyk not created!")


def test_sample_game():
    """Play a sample game step by step"""
    print("\n" + "=" * 60)
    print("Test: Sample game (first 5 moves)")
    print("=" * 60)
    
    game = TogyzQumalaq()
    
    moves = [2, 2, 0, 0, 4]  # Some moves
    
    for i, move in enumerate(moves):
        player = "White" if game.state.current_player == 0 else "Black"
        print(f"\nMove {i+1}: {player} plays pit {move + 1}")
        
        valid_moves = game.get_valid_moves_list()
        if move not in valid_moves:
            print(f"  ⚠️ Invalid move! Valid moves: {[m+1 for m in valid_moves]}")
            break
        
        success, winner = game.make_move(move)
        
        print(f"  White: pits={list(game.state.pits[0])}, kazan={game.state.kazan[0]}")
        print(f"  Black: pits={list(game.state.pits[1])}, kazan={game.state.kazan[1]}")
        
        if winner is not None:
            print(f"  Game over! Winner: {winner}")
            break


def check_alphazero_moves():
    """Check what moves AlphaZero is making"""
    print("\n" + "=" * 60)
    print("Check: AlphaZero move selection")
    print("=" * 60)
    
    import torch
    from model import create_model
    from mcts import MCTS, MCTSConfig
    from play import load_model
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("checkpoints/model_iter50.pt", device)
    
    config = MCTSConfig(num_simulations=200, temperature=0)
    mcts = MCTS(model, config)
    
    game = TogyzQumalaq()
    
    print("Playing 10 moves with AlphaZero...")
    for i in range(10):
        state = game.get_state()
        player = "White" if state.current_player == 0 else "Black"
        
        if state.current_player == Player.WHITE:
            policy = mcts.search(state, add_noise=False)
            move = int(np.argmax(policy))
            print(f"Move {i+1}: {player} (AlphaZero) -> pit {move + 1}, policy: {policy}")
        else:
            # Black plays random
            valid = game.get_valid_moves_list()
            move = np.random.choice(valid)
            print(f"Move {i+1}: {player} (Random) -> pit {move + 1}")
        
        game.make_move(move)
        
        if game.is_terminal():
            print(f"Game over! Winner: {game.get_winner()}")
            break
    
    print(f"\nFinal state:")
    print(f"  White: kazan={game.state.kazan[0]}")
    print(f"  Black: kazan={game.state.kazan[1]}")


if __name__ == "__main__":
    test_single_stone_move()
    test_multi_stone_move()
    test_capture()
    test_tuzdyk()
    test_sample_game()
    check_alphazero_moves()

