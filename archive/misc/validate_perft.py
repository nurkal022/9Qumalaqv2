"""
Cross-validate Rust engine perft counts against Python game engine.
Ensures both implementations agree on move generation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'alphazero-code', 'alphazero'))
from game import TogyzQumalaq


def perft(game: TogyzQumalaq, depth: int) -> int:
    if depth == 0:
        return 1
    if game.is_terminal():
        return 1

    moves = game.get_valid_moves_list()
    nodes = 0
    for move in moves:
        child = TogyzQumalaq()
        child.set_state(game.get_state())
        child.make_move(move)
        nodes += perft(child, depth - 1)
    return nodes


def main():
    # Expected values from Rust engine
    rust_perft = {
        1: 9,
        2: 73,
        3: 613,
        4: 5199,
        5: 43184,
        6: 360035,
    }

    game = TogyzQumalaq()
    max_depth = 6  # depth 6 is slow in Python but manageable

    print("Cross-validating Python vs Rust perft counts:")
    print(f"{'Depth':<8} {'Python':<12} {'Rust':<12} {'Match'}")
    print("-" * 45)

    all_match = True
    for depth in range(1, max_depth + 1):
        py_nodes = perft(game, depth)
        rust_nodes = rust_perft.get(depth, "?")
        match = py_nodes == rust_nodes
        if not match:
            all_match = False
        print(f"{depth:<8} {py_nodes:<12} {rust_nodes:<12} {'OK' if match else 'MISMATCH!'}")

    print()
    if all_match:
        print("All perft counts match! Game logic is consistent.")
    else:
        print("MISMATCH DETECTED! Game logic differs between engines.")

    return 0 if all_match else 1


if __name__ == '__main__':
    sys.exit(main())
