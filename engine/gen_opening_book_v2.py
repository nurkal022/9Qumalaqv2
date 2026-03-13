#!/usr/bin/env python3
"""
Generate expanded opening book from 360K+ PlayOK expert games.
Extracts positions up to 40 half-moves from games with ELO >= 1600.
Output: opening_book.txt for Rust engine.
"""

import os
import re
import sys
from collections import defaultdict
from pathlib import Path

NUM_PITS = 9
INITIAL_STONES = 9
BOOK_DEPTH = 40  # 40 half-moves (expanded from 16)
MIN_ELO = 1600
MIN_GAMES = 3    # minimum games to include position


class Board:
    def __init__(self):
        self.pits = [[INITIAL_STONES]*NUM_PITS, [INITIAL_STONES]*NUM_PITS]
        self.kazan = [0, 0]
        self.tuzdyk = [-1, -1]
        self.side_to_move = 0

    def clone(self):
        b = Board.__new__(Board)
        b.pits = [list(self.pits[0]), list(self.pits[1])]
        b.kazan = list(self.kazan)
        b.tuzdyk = list(self.tuzdyk)
        b.side_to_move = self.side_to_move
        return b

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
        if stones <= 0:
            return False

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
        return True


def board_key(board):
    return (
        tuple(board.pits[0]), tuple(board.pits[1]),
        board.kazan[0], board.kazan[1],
        board.tuzdyk[0], board.tuzdyk[1],
        board.side_to_move
    )


def parse_game_moves(text):
    """Extract moves and metadata from a PGN-like game text."""
    headers = {}
    for m in re.finditer(r'\[(\w+)\s+"([^"]*)"\]', text):
        headers[m.group(1)] = m.group(2)

    # Extract move text (everything after headers)
    lines = text.strip().split('\n')
    move_lines = []
    in_moves = False
    for line in lines:
        if in_moves:
            move_lines.append(line.strip())
        elif not line.startswith('[') and line.strip():
            in_moves = True
            move_lines.append(line.strip())

    move_text = ' '.join(move_lines)

    # Parse result
    result_match = re.search(r'\s*(1-0|0-1|1/2-1/2|\*)\s*$', move_text)
    result = result_match.group(1) if result_match else headers.get('Result', '*')

    # Extract pit_from for each move (first digit of 2-digit code)
    pit_moves = []
    for m in re.finditer(r'(\d{2})(X?)(?:\((\d+)\))?', move_text):
        pit_from = int(m.group(1)[0])  # 1-9
        if 1 <= pit_from <= 9:
            pit_moves.append(pit_from - 1)  # convert to 0-indexed

    return headers, pit_moves, result


def main():
    base = '/home/nurlykhan/9QumalaqV2'
    sources = [
        os.path.join(base, 'game-pars/games'),
        os.path.join(base, 'game-pars/linux_data/games'),
        os.path.join(base, 'gameNew2'),
    ]

    # Parse all games and filter by ELO
    print(f"Generating opening book: depth={BOOK_DEPTH}, min_elo={MIN_ELO}", flush=True)

    # position_key -> {move_idx -> [w_win, draw, b_win, count]}
    position_stats = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))
    total_games = 0
    expert_games = 0

    for src_path in sources:
        if not os.path.exists(src_path):
            print(f"  Skipping {src_path} (not found)")
            continue

        files = sorted(Path(src_path).glob('*.txt'))
        print(f"  Parsing {src_path}: {len(files)} files...", flush=True)

        for i, fp in enumerate(files):
            if (i + 1) % 100000 == 0:
                print(f"    {i+1}/{len(files)}...", flush=True)
            try:
                content = fp.read_text(encoding='utf-8', errors='ignore')
                parts = re.split(r'(?=\[Event\s)', content)

                for part in parts:
                    part = part.strip()
                    if not part or '[Event' not in part:
                        continue

                    headers, pit_moves, result = parse_game_moves(part)
                    total_games += 1

                    # ELO filter
                    w_elo = int(headers.get('WhiteElo', '0') or '0')
                    b_elo = int(headers.get('BlackElo', '0') or '0')
                    if w_elo < MIN_ELO and b_elo < MIN_ELO:
                        continue

                    if result not in ('1-0', '0-1', '1/2-1/2'):
                        continue

                    if len(pit_moves) < 4:
                        continue

                    expert_games += 1

                    # Determine result index
                    if result == '1-0':
                        result_idx = 0
                    elif result == '0-1':
                        result_idx = 2
                    else:
                        result_idx = 1

                    # Replay game and record positions
                    board = Board()
                    for ply, pit_idx in enumerate(pit_moves[:BOOK_DEPTH]):
                        valid = board.valid_moves()
                        if pit_idx not in valid:
                            break

                        key = board_key(board)
                        stats = position_stats[key][pit_idx]
                        stats[result_idx] += 1
                        stats[3] += 1

                        if not board.make_move(pit_idx):
                            break

            except Exception:
                continue

    print(f"Total games parsed: {total_games}")
    print(f"Expert games (ELO >= {MIN_ELO}): {expert_games}")
    print(f"Unique positions: {len(position_stats)}")

    # Build book: for each position, pick the best move
    book = {}
    for pos_key, moves in position_stats.items():
        best_move = -1
        best_score = -1
        best_count = 0
        stm = pos_key[-1]

        for move_idx, stats in moves.items():
            w_win, draw, b_win, count = stats
            if count < MIN_GAMES:
                continue

            if stm == 0:  # white to move
                score = (w_win + 0.5 * draw) / count
            else:  # black to move
                score = (b_win + 0.5 * draw) / count

            # Confidence bonus: prefer moves with more games
            weighted = score + 0.1 * min(count, 50) / 50

            if weighted > best_score:
                best_score = weighted
                best_move = move_idx
                best_count = count

        if best_move >= 0 and best_count >= MIN_GAMES:
            book[pos_key] = (best_move, best_count)

    print(f"Book entries: {len(book)}")

    # Write text format
    output_file = os.path.join(os.path.dirname(__file__), 'opening_book.txt')
    with open(output_file, 'w') as f:
        for pos_key, (move, count) in sorted(book.items(), key=lambda x: -x[1][1]):
            pits0 = ','.join(str(x) for x in pos_key[0])
            pits1 = ','.join(str(x) for x in pos_key[1])
            k0, k1 = pos_key[2], pos_key[3]
            t0, t1 = pos_key[4], pos_key[5]
            stm = pos_key[6]
            f.write(f"{pits0}|{pits1}|{k0},{k1}|{t0},{t1}|{stm}|{move}|{count}\n")

    print(f"Written: {output_file} ({len(book)} entries)")

    # Show depth distribution
    depth_counts = defaultdict(int)
    board = Board()
    for pos_key in book:
        total_stones = sum(pos_key[0]) + sum(pos_key[1]) + pos_key[2] + pos_key[3]
        stones_captured = 162 - total_stones  # rough ply estimate
        depth_counts[min(stones_captured // 4, 10)] += 1

    print("\nTop 10 entries:")
    with open(output_file) as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            print(f"  {line.strip()}")


if __name__ == '__main__':
    main()
