"""
Diagnostic script to check if AlphaZero model was trained correctly
"""

import torch
import numpy as np
from game import TogyzQumalaq, GameState, Player
from model import create_model
from mcts import MCTS, MCTSConfig
from play import load_model


def diagnose_model(checkpoint_path: str = "checkpoints/model_iter50.pt"):
    """Run comprehensive diagnostics on the model"""
    
    print("=" * 70)
    print("AlphaZero Model Diagnostics")
    print("=" * 70)
    
    # 1. Load model
    print("\n1. Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")
    
    try:
        model = load_model(checkpoint_path, device)
        print("   ✅ Model loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        return
    
    # 2. Check model outputs on initial position
    print("\n2. Testing model on initial position...")
    game = TogyzQumalaq()
    encoded = game.encode_state()
    
    print(f"   Encoded state shape: {encoded.shape}")
    print(f"   Encoded state sample (channel 0, player pits): {encoded[0]}")
    
    policy, value = model.predict(encoded)
    
    print(f"   Policy: {policy}")
    print(f"   Policy sum: {policy.sum():.4f}")
    print(f"   Value: {value:.4f}")
    
    if np.isnan(policy).any() or np.isnan(value):
        print("   ❌ NaN detected in model outputs!")
        return
    
    if policy.sum() < 0.99 or policy.sum() > 1.01:
        print(f"   ⚠️ Policy doesn't sum to 1.0 (sum={policy.sum():.4f})")
    
    print("   ✅ Model outputs look valid")
    
    # 3. Test MCTS
    print("\n3. Testing MCTS...")
    config = MCTSConfig(num_simulations=100, temperature=0)
    mcts = MCTS(model, config)
    
    state = game.get_state()
    mcts_policy = mcts.search(state, add_noise=False)
    
    print(f"   MCTS policy: {mcts_policy}")
    print(f"   Best move: pit {np.argmax(mcts_policy) + 1}")
    print("   ✅ MCTS works")
    
    # 4. Play a sample game (AlphaZero vs Random)
    print("\n4. Playing AlphaZero vs Random (10 games)...")
    az_wins = 0
    random_wins = 0
    draws = 0
    
    config = MCTSConfig(num_simulations=200, temperature=0)
    mcts = MCTS(model, config)
    
    for game_num in range(10):
        game = TogyzQumalaq()
        move_count = 0
        max_moves = 300
        
        while not game.is_terminal() and move_count < max_moves:
            state = game.get_state()
            current_player = state.current_player
            
            if current_player == Player.WHITE:
                # AlphaZero plays
                policy = mcts.search(state, add_noise=False)
                move = int(np.argmax(policy))
            else:
                # Random plays
                valid_moves = game.get_valid_moves_list()
                move = np.random.choice(valid_moves) if valid_moves else 0
            
            success, winner = game.make_move(move)
            if not success:
                break
            move_count += 1
        
        winner = game.get_winner()
        if winner == Player.WHITE:
            az_wins += 1
        elif winner == Player.BLACK:
            random_wins += 1
        else:
            draws += 1
    
    print(f"   AlphaZero wins: {az_wins}/10")
    print(f"   Random wins: {random_wins}/10")
    print(f"   Draws: {draws}/10")
    
    if az_wins < 5:
        print("   ⚠️ AlphaZero isn't winning most games against Random!")
    else:
        print("   ✅ AlphaZero beats Random")
    
    # 5. Check value predictions make sense
    print("\n5. Testing value predictions...")
    
    # Initial position should be close to 0
    game = TogyzQumalaq()
    _, value_initial = model.predict(game.encode_state())
    print(f"   Initial position value: {value_initial:.4f}")
    
    # Create winning position for white
    game = TogyzQumalaq()
    game.state.kazan[Player.WHITE] = 80  # Almost winning
    game.state.kazan[Player.BLACK] = 0
    _, value_winning = model.predict(game.encode_state())
    print(f"   Winning position value (kazan 80-0): {value_winning:.4f}")
    
    # Create losing position
    game = TogyzQumalaq()
    game.state.kazan[Player.WHITE] = 0
    game.state.kazan[Player.BLACK] = 80  # Almost losing
    _, value_losing = model.predict(game.encode_state())
    print(f"   Losing position value (kazan 0-80): {value_losing:.4f}")
    
    if value_winning > value_losing:
        print("   ✅ Value predictions have correct relative order")
    else:
        print("   ❌ Value predictions are inverted or incorrect!")
    
    # 6. Check policy on obvious positions
    print("\n6. Testing policy on specific positions...")
    
    # Position where only one move is valid
    game = TogyzQumalaq()
    for i in range(8):  # Empty all but pit 9
        game.state.pits[Player.WHITE, i] = 0
    game.state.pits[Player.WHITE, 8] = 20  # Only pit 9 has stones
    
    policy, _ = model.predict(game.encode_state())
    print(f"   Position with only pit 9 valid:")
    print(f"   Policy: {policy}")
    print(f"   Highest policy for pit {np.argmax(policy) + 1} ({policy[np.argmax(policy)]:.3f})")
    
    valid = game.get_valid_moves()
    print(f"   Valid moves mask: {valid}")
    
    # 7. Check training data consistency
    print("\n7. Summary...")
    print("=" * 70)
    
    issues = []
    
    if az_wins < 5:
        issues.append("AlphaZero loses to Random too often")
    
    if value_winning <= value_losing:
        issues.append("Value predictions don't reflect board state")
    
    if abs(value_initial) > 0.5:
        issues.append("Initial position value is too biased")
    
    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  ❌ {issue}")
        print("\nPossible causes:")
        print("  - Not enough training iterations")
        print("  - Bug in training loop or loss calculation")
        print("  - Bug in game logic (Python vs JavaScript mismatch)")
        print("  - Bug in state encoding")
    else:
        print("✅ Model passes basic diagnostics")
        print("\nIf AlphaZero still loses to Minimax, possible reasons:")
        print("  - Need more training (more iterations)")
        print("  - Need more MCTS simulations (800+)")
        print("  - Minimax with good eval is still stronger than 50 iteration AlphaZero")
    
    print("=" * 70)
    
    return model


def compare_game_logic():
    """Compare Python game logic with expected behavior"""
    print("\n" + "=" * 70)
    print("Game Logic Verification")
    print("=" * 70)
    
    # Test 1: Basic move
    print("\nTest 1: Basic move from pit 3 (index 2)")
    game = TogyzQumalaq()
    print(f"Before: White pits = {game.state.pits[0]}")
    
    game.make_move(2)  # Play pit 3
    print(f"After: White pits = {game.state.pits[0]}")
    print(f"After: Black pits = {game.state.pits[1]}")
    print(f"White kazan: {game.state.kazan[0]}")
    
    # Test 2: Check capture
    print("\nTest 2: Capture (even number)")
    game = TogyzQumalaq()
    # Setup a position where capture can happen
    game.state.pits[Player.WHITE] = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
    game.state.pits[Player.BLACK] = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
    print(f"Before: Black pit 2 = {game.state.pits[1, 1]}")
    
    game.make_move(0)  # White plays pit 1
    print(f"After: Black pit 2 = {game.state.pits[1, 1]}")
    print(f"White kazan: {game.state.kazan[0]} (should be 2 if captured)")
    
    # Test 3: Tuzdyk creation
    print("\nTest 3: Tuzdyk creation (exactly 3 stones)")
    game = TogyzQumalaq()
    game.state.pits[Player.WHITE] = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
    game.state.pits[Player.BLACK] = np.array([0, 2, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
    print(f"Before: Black pit 2 = {game.state.pits[1, 1]}, White tuzdyk = {game.state.tuzdyk[0]}")
    
    game.make_move(0)  # White plays pit 1, lands on Black pit 2 making it 3
    print(f"After: White tuzdyk = {game.state.tuzdyk[0]} (should be 1 for pit 2)")
    print(f"After: White kazan = {game.state.kazan[0]} (should be 3)")
    
    # Test 4: Win detection
    print("\nTest 4: Win detection")
    game = TogyzQumalaq()
    game.state.kazan[Player.WHITE] = 82
    print(f"White kazan = 82, is_terminal = {game.is_terminal()}")
    print(f"Winner: {game.get_winner()} (0 = White)")


if __name__ == "__main__":
    compare_game_logic()
    print()
    diagnose_model()

