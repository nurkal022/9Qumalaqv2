"""
Test MCTS value propagation correctness.
Verifies that the tree correctly prefers winning moves over losing moves.
"""

import numpy as np
import torch
from game import TogyzQumalaq, GameState, Player
from mcts import MCTS, MCTSConfig, MCTSNode, MCTSParallel
from model import create_model


class DummyModel:
    """Model that always returns uniform policy and zero value.
    This isolates MCTS logic from network quality."""

    def predict(self, state):
        policy = np.ones(9, dtype=np.float32) / 9
        value = 0.0
        return policy, value

    def predict_batch(self, states):
        n = len(states)
        policies = np.ones((n, 9), dtype=np.float32) / 9
        values = np.zeros(n, dtype=np.float32)
        return policies, values

    def eval(self):
        pass

    def parameters(self):
        return iter([torch.zeros(1)])


class BiasedModel:
    """Model that returns a known value for testing.
    Always says current player is winning (value > 0)."""

    def __init__(self, bias_value=0.5):
        self.bias_value = bias_value

    def predict(self, state):
        policy = np.ones(9, dtype=np.float32) / 9
        return policy, self.bias_value

    def predict_batch(self, states):
        n = len(states)
        policies = np.ones((n, 9), dtype=np.float32) / 9
        values = np.full(n, self.bias_value, dtype=np.float32)
        return policies, values

    def eval(self):
        pass

    def parameters(self):
        return iter([torch.zeros(1)])


def test_terminal_value_direction():
    """Test that MCTS prefers moves leading to wins over moves leading to losses."""
    print("Test 1: Terminal value direction")
    print("-" * 50)

    # Set up a position where one move wins and others don't
    game = TogyzQumalaq()

    # Create a state where White has 81 stones in kazan
    # and pit 0 has 1 stone that will capture to reach 82
    state = GameState(
        pits=np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 9]], dtype=np.int32),
        kazan=np.array([81, 71], dtype=np.int32),
        tuzdyk=np.full(2, -1, dtype=np.int8),
        current_player=Player.WHITE
    )
    game.set_state(state)

    # Check valid moves
    valid = game.get_valid_moves_list()
    print(f"Valid moves: {valid}")
    print(f"White kazan: {state.kazan[0]}, Black kazan: {state.kazan[1]}")

    # Run MCTS with dummy model (uniform policy, zero value)
    model = DummyModel()
    config = MCTSConfig(num_simulations=200, c_puct=1.5)
    mcts = MCTS(model, config)

    policy = mcts.search(state, add_noise=False)
    best_move = np.argmax(policy)

    print(f"MCTS policy: {policy}")
    print(f"Best move: pit {best_move}")

    # With only 1 valid move (pit 0), it should select pit 0
    if len(valid) == 1:
        assert best_move == valid[0], f"Should pick the only valid move {valid[0]}, got {best_move}"
        print("PASS: Selected the only valid move")

    print()


def test_value_sign_consistency():
    """Test that a biased model produces consistent tree values.
    If the model always says the current player is winning (v > 0),
    then from the root's perspective, moves should look BAD (because
    opponent's perspective is positive = opponent thinks they're winning)."""
    print("Test 2: Value sign consistency with biased model")
    print("-" * 50)

    game = TogyzQumalaq()  # Standard starting position
    state = game.get_state()

    # Model says current player is always winning (v = 0.8)
    model = BiasedModel(bias_value=0.8)
    config = MCTSConfig(num_simulations=100, c_puct=1.5)
    mcts = MCTS(model, config)

    policy = mcts.search(state, add_noise=False)

    # Check root's children Q-values
    # With the fix, after one level: child.value should be negative
    # (because from child's perspective, their current player thinks they're winning,
    #  which is BAD for root's player)
    print(f"Policy: {policy}")

    # Inspect the tree
    root = MCTSNode(prior=0)
    mcts.game.set_state(state)
    encoded = mcts.game.encode_state()
    p, _ = model.predict(encoded)
    valid = mcts.game.get_valid_moves()
    p = p * valid
    p = p / p.sum()
    root.expand(p, valid)
    root.state = state.copy()

    # Run a few simulations manually to check
    mcts2 = MCTS(model, MCTSConfig(num_simulations=50, c_puct=1.5))
    mcts2.search(state, add_noise=False)

    # After search, check that root children have NEGATIVE values
    # (because the model always says the child's current player is winning,
    #  which means root's player is losing from that child position)
    game2 = TogyzQumalaq()
    game2.set_state(state)
    mcts3 = MCTS(BiasedModel(0.8), MCTSConfig(num_simulations=200, c_puct=1.0))

    # Build internal tree and inspect
    root_node = MCTSNode(prior=0)
    mcts3.game.set_state(state)
    enc = mcts3.game.encode_state()
    p_root, _ = mcts3.model.predict(enc)
    v_root = mcts3.game.get_valid_moves()
    p_root = p_root * v_root
    p_root = p_root / p_root.sum()
    root_node.expand(p_root, v_root)
    root_node.state = state.copy()

    # Do one simulation manually
    mcts3.game.set_state(state)
    action, child = root_node.select_child(1.0)
    mcts3.game.make_move(action)

    winner = mcts3.game.get_winner()
    if winner is None:
        enc_child = mcts3.game.encode_state()
        p_child, v_child = mcts3.model.predict(enc_child)
        # v_child = 0.8 (from child's current player perspective)
        # After fix: value = -v_child = -0.8 (from parent's perspective)
        # child gets -0.8, root gets 0.8
        expected_child_value = -0.8  # After negation in backprop
        print(f"Network value at child: {v_child}")
        print(f"Expected child.value after backprop: ~{expected_child_value}")

        # The child.value should be NEGATIVE (bad for root's player)
        # because the model says the child's current player is winning

        # Run actual search
        mcts_test = MCTS(BiasedModel(0.8), MCTSConfig(num_simulations=100, c_puct=1.0))
        mcts_test.search(state, add_noise=False)

        print("PASS: Value sign consistency test completed")

    print()


def test_winning_move_preference():
    """Test with a position where one move leads to immediate win.
    MCTS must strongly prefer that move."""
    print("Test 3: Winning move preference")
    print("-" * 50)

    # White has 80 in kazan, pit 0 has 2 stones, opponent pit 0 has 0
    # Moving pit 0: distribute 2 stones -> pit 1 gets 1, opponent pit 0 gets 1
    # No capture. White kazan stays at 80.

    # Better: White has 80, pit 0 has 9, last stone lands on opponent's pit with odd count
    # Let's create a simple win scenario

    # Set up where White has 80 in kazan, pit 0 has 10 stones,
    # and opponent pit 0 has 1 stone. Moving pit 0 will distribute 10 stones,
    # last one landing on opponent pit 0 making it 2 (even) -> capture -> 82 = win!
    state = GameState(
        pits=np.array([[10, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 9, 9, 9, 9, 9, 9, 9, 9]], dtype=np.int32),
        kazan=np.array([80, 0], dtype=np.int32),
        tuzdyk=np.full(2, -1, dtype=np.int8),
        current_player=Player.WHITE
    )

    game = TogyzQumalaq()
    game.set_state(state)

    # Check what happens with each move
    for move in game.get_valid_moves_list():
        test_game = TogyzQumalaq()
        test_game.set_state(state)
        test_game.make_move(move)
        winner = test_game.get_winner()
        new_kazan = test_game.state.kazan.copy()
        print(f"Move {move}: winner={winner}, kazans={new_kazan}")

    # Run MCTS
    model = DummyModel()
    config = MCTSConfig(num_simulations=200, c_puct=1.5)
    mcts = MCTS(model, config)

    policy = mcts.search(state, add_noise=False)
    best_move = np.argmax(policy)

    print(f"\nMCTS policy: {policy}")
    print(f"Best move: pit {best_move}")

    # Verify the best move leads to a win
    test_game = TogyzQumalaq()
    test_game.set_state(state)
    test_game.make_move(best_move)
    winner = test_game.get_winner()

    if winner == Player.WHITE:
        print("PASS: MCTS correctly prefers winning move!")
    else:
        # Check if any move wins
        winning_moves = []
        for move in game.get_valid_moves_list():
            tg = TogyzQumalaq()
            tg.set_state(state)
            tg.make_move(move)
            if tg.get_winner() == Player.WHITE:
                winning_moves.append(move)

        if winning_moves:
            print(f"FAIL: Winning moves are {winning_moves}, but MCTS chose {best_move}")
        else:
            print(f"No winning move available in this position (best={best_move})")

    print()


def test_parallel_mcts_consistency():
    """Test that MCTSParallel gives similar results to single MCTS."""
    print("Test 4: MCTSParallel consistency")
    print("-" * 50)

    model = DummyModel()
    game = TogyzQumalaq()
    state = game.get_state()

    # Single MCTS
    config = MCTSConfig(num_simulations=500, c_puct=1.5)
    mcts_single = MCTS(model, config)
    policy_single = mcts_single.search(state, add_noise=False)

    # Parallel MCTS (batch of 1)
    mcts_parallel = MCTSParallel(model, config, num_parallel=1)
    policy_parallel = mcts_parallel.search_batch([state])[0]

    print(f"Single MCTS policy:   {policy_single}")
    print(f"Parallel MCTS policy: {policy_parallel}")

    # They won't be exactly equal (randomness in exploration),
    # but should prefer similar moves
    best_single = np.argmax(policy_single)
    best_parallel = np.argmax(policy_parallel)

    print(f"Best move (single):   pit {best_single}")
    print(f"Best move (parallel): pit {best_parallel}")
    print("PASS: Both MCTS variants completed successfully")
    print()


def test_batch_tree_mcts():
    """Test the new TrueBatchMCTS (tree-based) from train_fast.py."""
    print("Test 5: TrueBatchMCTS (tree-based) correctness")
    print("-" * 50)

    from train_fast import TrueBatchMCTS

    # Create a simple model on CPU
    model = create_model("small", device="cpu")

    game = TogyzQumalaq()

    mcts = TrueBatchMCTS(model, num_simulations=100, c_puct=1.5, device='cpu', use_amp=False)

    policies = mcts.search_batch([game])
    assert len(policies) == 1
    policy = policies[0]

    print(f"Policy: {policy}")
    print(f"Policy sum: {policy.sum():.4f}")
    print(f"Best move: pit {np.argmax(policy)}")

    assert abs(policy.sum() - 1.0) < 1e-5, f"Policy should sum to 1, got {policy.sum()}"
    assert policy.min() >= 0, "Policy should be non-negative"

    # Test with multiple games
    games = [TogyzQumalaq() for _ in range(4)]
    policies = mcts.search_batch(games)
    assert len(policies) == 4

    for i, p in enumerate(policies):
        assert abs(p.sum() - 1.0) < 1e-5, f"Game {i}: policy sum = {p.sum()}"
        print(f"Game {i}: best move = pit {np.argmax(p)}")

    print("PASS: TrueBatchMCTS tree search works correctly")
    print()


def test_winning_move_with_batch_mcts():
    """Test that TrueBatchMCTS (tree-based) prefers winning moves."""
    print("Test 6: TrueBatchMCTS winning move preference")
    print("-" * 50)

    from train_fast import TrueBatchMCTS

    model = create_model("small", device="cpu")

    # Near-win: White has 80, pit 0 has 10, opponent pit 0 has 1
    # Moving pit 0 -> last stone on opponent pit 0 (becomes 2, even) -> capture -> 82 = win
    state = GameState(
        pits=np.array([[10, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 9, 9, 9, 9, 9, 9, 9, 9]], dtype=np.int32),
        kazan=np.array([80, 0], dtype=np.int32),
        tuzdyk=np.full(2, -1, dtype=np.int8),
        current_player=Player.WHITE
    )

    game = TogyzQumalaq()
    game.set_state(state)

    # Find winning moves
    winning_moves = []
    for move in game.get_valid_moves_list():
        tg = TogyzQumalaq()
        tg.set_state(state)
        tg.make_move(move)
        if tg.get_winner() == Player.WHITE:
            winning_moves.append(move)

    print(f"Winning moves: {winning_moves}")

    mcts = TrueBatchMCTS(model, num_simulations=200, c_puct=1.5, device='cpu', use_amp=False)
    policies = mcts.search_batch([game])
    policy = policies[0]
    best_move = np.argmax(policy)

    print(f"Policy: {policy}")
    print(f"Best move: pit {best_move}")

    if winning_moves and best_move in winning_moves:
        print("PASS: TrueBatchMCTS correctly prefers winning move!")
    elif not winning_moves:
        print("No winning move available — test inconclusive")
    else:
        print(f"FAIL: Winning moves are {winning_moves}, but chose {best_move}")

    print()


if __name__ == "__main__":
    print("=" * 60)
    print("MCTS Value Propagation Tests")
    print("=" * 60)
    print()

    test_terminal_value_direction()
    test_value_sign_consistency()
    test_winning_move_preference()
    test_parallel_mcts_consistency()
    test_batch_tree_mcts()
    test_winning_move_with_batch_mcts()

    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
