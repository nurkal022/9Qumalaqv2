#!/usr/bin/env python3
"""
Convert human game database to NNUE binary training format.

Replays games from valid_games.json, extracts board positions,
and writes them in the 26-byte format expected by train_nnue_v2.py.

Since we don't have engine evaluations for human games,
eval is set to 0. The training script handles this via has_eval
(positions with |eval| <= 1 use 100% game result as target).
"""

import json
import struct
import sys
import os
import numpy as np


NUM_PITS = 9
INITIAL_STONES = 9
WIN_THRESHOLD = 82
RECORD_SIZE = 26


class Board:
    def __init__(self):
        self.pits = [[INITIAL_STONES]*NUM_PITS, [INITIAL_STONES]*NUM_PITS]
        self.kazan = [0, 0]
        self.tuzdyk = [-1, -1]
        self.side_to_move = 0  # 0=White, 1=Black

    def valid_moves(self):
        me = self.side_to_move
        opp = 1 - me
        opp_tuz = self.tuzdyk[opp]
        moves = []
        for i in range(NUM_PITS):
            if self.pits[me][i] > 0 and opp_tuz != i:
                moves.append(i)
        return moves

    def can_create_tuzdyk(self, player, pit_index):
        if self.tuzdyk[player] != -1:
            return False
        if pit_index == 8:  # can't at pit 9
            return False
        opponent = 1 - player
        if self.tuzdyk[opponent] == pit_index:
            return False
        return True

    def make_move(self, pit_index):
        me = self.side_to_move
        opp = 1 - me
        stones = self.pits[me][pit_index]
        assert stones > 0, f"Empty pit {pit_index}"

        self.pits[me][pit_index] = 0
        current_pit = pit_index
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

        # Check capture and tuzdyk
        is_tuzdyk_pit = (current_side == opp and self.tuzdyk[me] == current_pit) or \
                        (current_side == me and self.tuzdyk[opp] == current_pit)

        if current_side == opp and not is_tuzdyk_pit:
            count = self.pits[opp][current_pit]
            if count == 3 and self.can_create_tuzdyk(me, current_pit):
                self.tuzdyk[me] = current_pit
                self.kazan[me] += count
                self.pits[opp][current_pit] = 0
            elif count % 2 == 0 and count > 0:
                self.kazan[me] += count
                self.pits[opp][current_pit] = 0

        self.side_to_move = 1 - self.side_to_move

    def is_terminal(self):
        if self.kazan[0] >= WIN_THRESHOLD or self.kazan[1] >= WIN_THRESHOLD:
            return True
        white_empty = all(p == 0 for p in self.pits[0])
        black_empty = all(p == 0 for p in self.pits[1])
        return white_empty or black_empty

    def pack_position(self, eval_score, result_byte):
        buf = bytearray(RECORD_SIZE)
        for i in range(9):
            buf[i] = self.pits[0][i]
            buf[9 + i] = self.pits[1][i]
        buf[18] = self.kazan[0]
        buf[19] = self.kazan[1]
        buf[20] = self.tuzdyk[0] & 0xFF
        buf[21] = self.tuzdyk[1] & 0xFF
        buf[22] = self.side_to_move
        eval_bytes = struct.pack('<h', eval_score)
        buf[23] = eval_bytes[0]
        buf[24] = eval_bytes[1]
        buf[25] = result_byte
        return bytes(buf)


def process_game(game):
    """Replay a game and return list of packed positions."""
    result_str = game['result']
    if result_str == '1-0':
        result_byte = 2  # white win
    elif result_str == '0-1':
        result_byte = 0  # white loss (black win)
    elif result_str in ('1/2-1/2', '1/2'):
        result_byte = 1  # draw
    else:
        return []

    board = Board()
    positions = []

    for move_num in game['moves']:
        if board.is_terminal():
            break

        pit_idx = move_num - 1  # convert 1-indexed to 0-indexed

        if pit_idx < 0 or pit_idx >= 9:
            break

        valid = board.valid_moves()
        if pit_idx not in valid:
            break  # invalid move, stop

        # Record position before making the move
        positions.append(board.pack_position(0, result_byte))

        board.make_move(pit_idx)

    return positions


def main():
    games_file = os.path.join(os.path.dirname(__file__), '..', 'parsed_games', 'valid_games.json')
    if not os.path.exists(games_file):
        print(f"Error: {games_file} not found")
        sys.exit(1)

    with open(games_file) as f:
        games = json.load(f)

    print(f"Loaded {len(games)} games")

    # Filter by minimum Elo
    min_elo = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'human_games_training.bin'

    if min_elo > 0:
        filtered = [g for g in games if
                    g.get('white_elo', 0) >= min_elo or g.get('black_elo', 0) >= min_elo]
        print(f"Filtered to {len(filtered)} games with Elo >= {min_elo}")
    else:
        filtered = games
        print(f"Using all {len(filtered)} games")

    total_positions = 0
    total_games = 0
    failed = 0

    with open(output_file, 'wb') as f:
        for game in filtered:
            try:
                positions = process_game(game)
                if positions:
                    for pos in positions:
                        f.write(pos)
                    total_positions += len(positions)
                    total_games += 1
            except Exception as e:
                failed += 1

    size_mb = os.path.getsize(output_file) / 1e6
    print(f"\nDone!")
    print(f"Games processed: {total_games}/{len(filtered)} (failed: {failed})")
    print(f"Positions: {total_positions:,}")
    print(f"Output: {output_file} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
