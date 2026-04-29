"""
Export positions from parsed games for Texel tuning.
Each position: board state + game result from side-to-move perspective.
Output: simple CSV that Rust can parse quickly.
"""

import sys
import os
import json
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'alphazero-code', 'alphazero'))
from game import TogyzQumalaq


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    games_path = os.path.join(base_dir, 'parsed_games', 'valid_games.json')

    with open(games_path, 'r') as f:
        games = json.load(f)

    print(f"Loaded {len(games)} games")

    output_path = os.path.join(base_dir, 'engine', 'positions.txt')
    positions = []

    for game in games:
        result_str = game['result']
        if result_str not in ('1-0', '0-1', '1/2-1/2'):
            continue

        # Result from White's perspective: 1.0 = white wins, 0.0 = black wins, 0.5 = draw
        if result_str == '1-0':
            white_result = 1.0
        elif result_str == '0-1':
            white_result = 0.0
        else:
            white_result = 0.5

        moves = [m - 1 for m in game['moves']]  # convert to 0-indexed
        g = TogyzQumalaq()

        for i, pit in enumerate(moves):
            state = g.get_state()

            # Skip very early positions (first 4 moves) - too little info
            if i < 4:
                g.make_move(pit)
                continue

            # Result from side-to-move's perspective
            current_player = int(state.current_player)
            if current_player == 0:  # White
                result = white_result
            else:  # Black
                result = 1.0 - white_result

            # Extract board state
            pits_w = ','.join(str(int(x)) for x in state.pits[0])
            pits_b = ','.join(str(int(x)) for x in state.pits[1])
            kazan_w = int(state.kazan[0])
            kazan_b = int(state.kazan[1])
            tuzdyk_w = int(state.tuzdyk[0])
            tuzdyk_b = int(state.tuzdyk[1])

            # Format: pits_w(9),pits_b(9),kazan_w,kazan_b,tuzdyk_w,tuzdyk_b,side,result
            line = f"{pits_w},{pits_b},{kazan_w},{kazan_b},{tuzdyk_w},{tuzdyk_b},{current_player},{result:.1f}"
            positions.append(line)

            success, winner = g.make_move(pit)
            if winner is not None:
                break

    # Shuffle to avoid any ordering bias
    random.seed(42)
    random.shuffle(positions)

    with open(output_path, 'w') as f:
        for line in positions:
            f.write(line + '\n')

    print(f"Exported {len(positions)} positions to {output_path}")


if __name__ == '__main__':
    main()
