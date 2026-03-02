"""
Convert master game positions to NNUE training data (26-byte binary format).

Replays games from valid_games.json, records positions with material-based eval
and game results. Filters by Elo for quality.

Usage:
    python convert_master_games.py [--min-elo 1800] [--output master_training.bin]
"""

import json
import struct
import argparse
import numpy as np
from pathlib import Path

RECORD_SIZE = 26
NUM_PITS = 9


class Board:
    """Togyz Kumalak board implementation matching the Rust engine."""

    def __init__(self):
        self.pits = [[9] * 9, [9] * 9]
        self.kazan = [0, 0]
        self.tuzdyk = [-1, -1]
        self.side = 0  # 0=white, 1=black
        self.move_count = 0

    def is_valid_move(self, pit):
        if pit < 0 or pit >= 9:
            return False
        me = self.side
        opp = 1 - me
        return self.pits[me][pit] > 0 and self.tuzdyk[opp] != pit

    def make_move(self, pit):
        me = self.side
        opp = 1 - me
        stones = self.pits[me][pit]
        assert stones > 0, f"Empty pit {pit}"

        self.pits[me][pit] = 0
        current_pit = pit
        current_side = me

        if stones == 1:
            current_pit += 1
            if current_pit > 8:
                current_pit = 0
                current_side = opp

            if current_side == opp and self.tuzdyk[me] == current_pit:
                self.kazan[me] += 1
            elif current_side == me and self.tuzdyk[opp] == current_pit:
                self.kazan[opp] += 1
            else:
                self.pits[current_side][current_pit] += 1
        else:
            # First stone back to source
            self.pits[current_side][current_pit] += 1
            remaining = stones - 1

            while remaining > 0:
                current_pit += 1
                if current_pit > 8:
                    current_pit = 0
                    current_side = 1 - current_side

                if current_side == opp and self.tuzdyk[me] == current_pit:
                    self.kazan[me] += 1
                elif current_side == me and self.tuzdyk[opp] == current_pit:
                    self.kazan[opp] += 1
                else:
                    self.pits[current_side][current_pit] += 1

                remaining -= 1

        # Check capture / tuzdyk
        is_tuzdyk_pit = (current_side == opp and self.tuzdyk[me] == current_pit) or \
                        (current_side == me and self.tuzdyk[opp] == current_pit)

        if current_side == opp and not is_tuzdyk_pit:
            count = self.pits[opp][current_pit]
            if count == 3 and self._can_create_tuzdyk(me, current_pit):
                self.tuzdyk[me] = current_pit
                self.kazan[me] += count
                self.pits[opp][current_pit] = 0
            elif count % 2 == 0 and count > 0:
                self.kazan[me] += count
                self.pits[opp][current_pit] = 0

        self.side = 1 - self.side
        self.move_count += 1

    def _can_create_tuzdyk(self, player, pit_index):
        if self.tuzdyk[player] != -1:
            return False
        if pit_index == 8:
            return False
        opponent = 1 - player
        if self.tuzdyk[opponent] == pit_index:
            return False
        return True

    def is_terminal(self):
        if self.kazan[0] >= 82 or self.kazan[1] >= 82:
            return True
        w_empty = all(x == 0 for x in self.pits[0])
        b_empty = all(x == 0 for x in self.pits[1])
        return w_empty or b_empty

    def total_stones(self):
        return sum(self.pits[0]) + sum(self.pits[1]) + self.kazan[0] + self.kazan[1]

    def pack(self, eval_score, result):
        """Pack position into 26-byte binary record."""
        buf = bytearray(26)
        for i in range(9):
            buf[i] = self.pits[0][i]
            buf[9 + i] = self.pits[1][i]
        buf[18] = self.kazan[0]
        buf[19] = self.kazan[1]
        buf[20] = self.tuzdyk[0] & 0xFF
        buf[21] = self.tuzdyk[1] & 0xFF
        buf[22] = self.side
        struct.pack_into('<h', buf, 23, max(-3000, min(3000, eval_score)))
        buf[25] = result
        return bytes(buf)


def material_eval(board):
    """Simple material-based evaluation from side-to-move perspective."""
    me = board.side
    opp = 1 - me
    # Material difference (kazan)
    mat_diff = board.kazan[me] - board.kazan[opp]
    # Pit stones
    my_pit = sum(board.pits[me])
    opp_pit = sum(board.pits[opp])
    # Tuzdyk bonus
    tuz = 0
    if board.tuzdyk[me] >= 0:
        tuz += 500
    if board.tuzdyk[opp] >= 0:
        tuz -= 500
    return mat_diff * 21 + (my_pit - opp_pit) * 3 + tuz


def convert_games(games, min_elo, skip_first_n=4, endgame_weight=1):
    """Convert games to training positions.

    Args:
        games: list of game dicts
        min_elo: minimum Elo for both players
        skip_first_n: skip first N moves (too random)
        endgame_weight: duplicate endgame positions this many times
    """
    positions = []
    skipped = 0
    errors = 0

    for game in games:
        w_elo = game.get('white_elo', 0) or 0
        b_elo = game.get('black_elo', 0) or 0

        if w_elo < min_elo or b_elo < min_elo:
            skipped += 1
            continue

        result_str = game.get('result', '')
        if result_str == '1-0':
            result = 2  # white win
        elif result_str == '0-1':
            result = 0  # black win (white loss)
        elif result_str == '1/2-1/2' or result_str == '0.5-0.5':
            result = 1  # draw
        else:
            skipped += 1
            continue

        moves = game.get('moves', [])
        if len(moves) < 10:
            skipped += 1
            continue

        board = Board()
        valid = True

        for move_idx, move in enumerate(moves):
            pit = move - 1  # 1-indexed to 0-indexed

            if not board.is_valid_move(pit):
                errors += 1
                valid = False
                break

            # Record position (skip first few moves — too early / opening book)
            if move_idx >= skip_first_n:
                ev = material_eval(board)
                packed = board.pack(ev, result)

                # Verify stone conservation
                total = board.total_stones()
                if total == 162:
                    positions.append(packed)
                    # Extra weight for endgame positions
                    board_stones = sum(board.pits[0]) + sum(board.pits[1])
                    if board_stones <= 30 and endgame_weight > 1:
                        for _ in range(endgame_weight - 1):
                            positions.append(packed)

            board.make_move(pit)

            if board.is_terminal():
                break

    return positions, skipped, errors


def main():
    parser = argparse.ArgumentParser(description='Convert master games to NNUE training data')
    parser.add_argument('--input', default='../parsed_games/valid_games.json',
                        help='Path to valid_games.json')
    parser.add_argument('--min-elo', type=int, default=1800,
                        help='Minimum Elo for both players')
    parser.add_argument('--output', default='master_training.bin',
                        help='Output binary file')
    parser.add_argument('--endgame-weight', type=int, default=3,
                        help='Duplicate endgame positions N times')
    parser.add_argument('--all-elos', action='store_true',
                        help='Also generate a file with all Elo levels')
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        # Try alternative path
        alt = Path('../parsed_games/valid_games.json')
        if alt.exists():
            input_path = alt
        else:
            print(f"Error: {input_path} not found")
            return

    with open(input_path) as f:
        games = json.load(f)

    print(f"Loaded {len(games)} games from {input_path}")

    # Convert high-Elo games
    positions, skipped, errors = convert_games(
        games, min_elo=args.min_elo, endgame_weight=args.endgame_weight
    )
    print(f"\nElo >= {args.min_elo}:")
    print(f"  Positions: {len(positions):,}")
    print(f"  Skipped games: {skipped}")
    print(f"  Errors: {errors}")

    if positions:
        # Shuffle
        np.random.seed(42)
        indices = np.random.permutation(len(positions))
        shuffled = [positions[i] for i in indices]

        output_path = Path(args.output)
        with open(output_path, 'wb') as f:
            for p in shuffled:
                f.write(p)
        print(f"  Saved: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")

    # Also generate all-Elo version for broader coverage
    if args.all_elos:
        positions_all, skipped_all, errors_all = convert_games(
            games, min_elo=0, endgame_weight=2
        )
        print(f"\nAll Elo levels:")
        print(f"  Positions: {len(positions_all):,}")

        if positions_all:
            np.random.seed(42)
            indices = np.random.permutation(len(positions_all))
            shuffled = [positions_all[i] for i in indices]

            all_path = Path('master_all_elo_training.bin')
            with open(all_path, 'wb') as f:
                for p in shuffled:
                    f.write(p)
            print(f"  Saved: {all_path} ({all_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == '__main__':
    main()
