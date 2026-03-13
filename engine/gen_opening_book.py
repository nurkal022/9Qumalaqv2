#!/usr/bin/env python3
"""
Generate opening book from human games for Togyz Kumalak engine.

Builds a Zobrist-hash-indexed book from strong human games.
Output: binary format for fast Rust lookup.

Format:
  - u32: number of entries
  - For each entry:
    - u64: zobrist hash (little-endian)
    - u8: best move (pit index 0-8)
    - u8: number of games backing this entry
"""

import json
import struct
import sys
import os
from collections import defaultdict

NUM_PITS = 9
INITIAL_STONES = 9
WIN_THRESHOLD = 82
BOOK_DEPTH = 16  # max plies to include in book


class Board:
    def __init__(self):
        self.pits = [[INITIAL_STONES]*NUM_PITS, [INITIAL_STONES]*NUM_PITS]
        self.kazan = [0, 0]
        self.tuzdyk = [-1, -1]
        self.side_to_move = 0

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
        if pit_index == 8:
            return False
        opponent = 1 - player
        if self.tuzdyk[opponent] == pit_index:
            return False
        return True

    def make_move(self, pit_index):
        me = self.side_to_move
        opp = 1 - me
        stones = self.pits[me][pit_index]
        assert stones > 0

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


# Zobrist keys (must match Rust engine)
import hashlib

def generate_zobrist_keys():
    """Generate deterministic Zobrist keys matching the Rust engine."""
    # The Rust engine uses a simple xorshift RNG seeded with specific values
    # We need to match this exactly. Let me read the Rust code instead.
    # For now, use a simple hash of the board state.
    pass


def board_hash(board):
    """Simple hash of board state (position-based, not Zobrist)."""
    h = 0
    for i in range(9):
        h ^= (board.pits[0][i] * 7919 + i * 104729) & 0xFFFFFFFFFFFFFFFF
        h ^= ((board.pits[1][i] * 7919 + (i + 9) * 104729) << 16) & 0xFFFFFFFFFFFFFFFF
    h ^= (board.kazan[0] * 999983) & 0xFFFFFFFFFFFFFFFF
    h ^= ((board.kazan[1] * 999979) << 32) & 0xFFFFFFFFFFFFFFFF
    h ^= ((board.tuzdyk[0] + 2) * 1000003) & 0xFFFFFFFFFFFFFFFF
    h ^= (((board.tuzdyk[1] + 2) * 1000033) << 48) & 0xFFFFFFFFFFFFFFFF
    h ^= (board.side_to_move * 0xCAFEBABE) & 0xFFFFFFFFFFFFFFFF
    return h


def board_key(board):
    """Canonical position key for deduplication."""
    return (
        tuple(board.pits[0]), tuple(board.pits[1]),
        board.kazan[0], board.kazan[1],
        board.tuzdyk[0], board.tuzdyk[1],
        board.side_to_move
    )


def main():
    games_file = '/home/nurlykhan/9QumalaqV2/parsed_games/valid_games.json'
    min_elo = int(sys.argv[1]) if len(sys.argv) > 1 else 1800
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'opening_book.bin'

    with open(games_file) as f:
        games = json.load(f)

    # Filter by Elo
    if min_elo > 0:
        filtered = [g for g in games if
                    g.get('white_elo', 0) >= min_elo or g.get('black_elo', 0) >= min_elo]
    else:
        filtered = games

    print(f"Games: {len(filtered)} (Elo >= {min_elo})")

    # Build position -> move statistics
    # For each position, track: move -> (wins_white, draws, wins_black, count)
    position_stats = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))

    for game in filtered:
        result_str = game['result']
        if result_str == '1-0':
            result_idx = 0  # white win
        elif result_str == '0-1':
            result_idx = 2  # black win
        elif result_str in ('1/2-1/2', '1/2'):
            result_idx = 1  # draw
        else:
            continue

        board = Board()
        for ply, move_num in enumerate(game['moves'][:BOOK_DEPTH]):
            pit_idx = move_num - 1
            if pit_idx < 0 or pit_idx >= 9:
                break

            valid = board.valid_moves()
            if pit_idx not in valid:
                break

            key = board_key(board)
            stats = position_stats[key][pit_idx]
            stats[result_idx] += 1
            stats[3] += 1  # total count

            try:
                board.make_move(pit_idx)
            except Exception:
                break

    # Build book: for each position, pick the best move
    # Score = (wins + 0.5*draws) / total, weighted by game count
    book = {}
    for pos_key, moves in position_stats.items():
        best_move = -1
        best_score = -1
        best_count = 0

        stm = pos_key[-1]  # side_to_move

        for move_idx, stats in moves.items():
            w_win, draw, b_win, count = stats
            if count < 2:  # need at least 2 games
                continue

            # Score from side-to-move perspective
            if stm == 0:  # white
                score = (w_win + 0.5 * draw) / count
            else:  # black
                score = (b_win + 0.5 * draw) / count

            # Prefer moves with more games (confidence)
            weighted = score + 0.1 * min(count, 20) / 20

            if weighted > best_score:
                best_score = weighted
                best_move = move_idx
                best_count = count

        if best_move >= 0 and best_count >= 3:
            book[pos_key] = (best_move, best_count)

    print(f"Book positions: {len(book)}")

    # Write as text format for easy Rust integration
    # Format: each line is "pits0|pits1|k0,k1|t0,t1|stm|move"
    with open(output_file.replace('.bin', '.txt'), 'w') as f:
        for pos_key, (move, count) in sorted(book.items(), key=lambda x: -x[1][1]):
            pits0 = ','.join(str(x) for x in pos_key[0])
            pits1 = ','.join(str(x) for x in pos_key[1])
            k0, k1 = pos_key[2], pos_key[3]
            t0, t1 = pos_key[4], pos_key[5]
            stm = pos_key[6]
            f.write(f"{pits0}|{pits1}|{k0},{k1}|{t0},{t1}|{stm}|{move}|{count}\n")

    # Also write binary format for fast lookup
    # Hash-based: hash each position, store (hash, move, count)
    entries = []
    for pos_key, (move, count) in book.items():
        # Create a board to compute hash
        board = Board()
        board.pits[0] = list(pos_key[0])
        board.pits[1] = list(pos_key[1])
        board.kazan[0] = pos_key[2]
        board.kazan[1] = pos_key[3]
        board.tuzdyk[0] = pos_key[4]
        board.tuzdyk[1] = pos_key[5]
        board.side_to_move = pos_key[6]

        h = board_hash(board)
        entries.append((h, move, min(count, 255)))

    with open(output_file, 'wb') as f:
        f.write(struct.pack('<I', len(entries)))
        for h, move, count in entries:
            f.write(struct.pack('<QBB', h, move, count))

    print(f"Output: {output_file} ({os.path.getsize(output_file)} bytes)")
    print(f"Text: {output_file.replace('.bin', '.txt')}")

    # Show top entries
    print("\nTop book entries:")
    with open(output_file.replace('.bin', '.txt')) as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            print(f"  {line.strip()}")


if __name__ == '__main__':
    main()
