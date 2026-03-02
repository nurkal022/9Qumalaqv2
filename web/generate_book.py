"""
Generate opening book from master games for Togyzkumalaq Championship Engine.

Reads parsed_games/valid_games.json, replays each game through a Python board
simulation, and records move frequencies + outcomes at each position.

Output: opening_book.json
"""
import json
import os
import sys
from copy import deepcopy

NUM_PITS = 9
BOOK_DEPTH = 16  # max half-moves to record
MIN_GAMES = 3    # minimum games at a position to include
MIN_ELO = 1600   # minimum Elo for either player


class Board:
    """Togyz Kumalak board simulation matching the Rust engine."""

    def __init__(self):
        self.pits = [[9]*9, [9]*9]
        self.kazan = [0, 0]
        self.tuzdyk = [-1, -1]
        self.side = 0  # 0=white, 1=black

    def pos_string(self):
        wp = ','.join(str(x) for x in self.pits[0])
        bp = ','.join(str(x) for x in self.pits[1])
        k = f"{self.kazan[0]},{self.kazan[1]}"
        t = f"{self.tuzdyk[0]},{self.tuzdyk[1]}"
        return f"{wp}/{bp}/{k}/{t}/{self.side}"

    def is_valid_move(self, pit):
        """Check if pit (0-indexed) is a valid move."""
        me = self.side
        opp = 1 - me
        return self.pits[me][pit] > 0 and self.tuzdyk[opp] != pit

    def make_move(self, pit):
        """Make a move (pit is 0-indexed). Returns True if valid."""
        me = self.side
        opp = 1 - me
        stones = self.pits[me][pit]
        if stones == 0:
            return False

        if stones == 1:
            self.pits[me][pit] = 0
            next_pit = pit + 1
            next_side = me
            if next_pit >= 9:
                next_pit = 0
                next_side = opp
            self.pits[next_side][next_pit] += 1
            land_side = next_side
            land_pit = next_pit
        else:
            self.pits[me][pit] = 1
            stones -= 1
            cur_side = me
            cur_pit = pit
            for _ in range(stones):
                cur_pit += 1
                if cur_pit >= 9:
                    cur_pit = 0
                    cur_side = 1 - cur_side
                self.pits[cur_side][cur_pit] += 1
            land_side = cur_side
            land_pit = cur_pit

        # Capture / tuzdyk
        if land_side == opp:
            if (self.pits[opp][land_pit] == 3 and land_pit != 8 and
                    self.tuzdyk[me] == -1 and self.tuzdyk[opp] != land_pit):
                self.tuzdyk[me] = land_pit
                self.kazan[me] += self.pits[opp][land_pit]
                self.pits[opp][land_pit] = 0
            elif self.pits[opp][land_pit] % 2 == 0 and self.pits[opp][land_pit] > 0:
                self.kazan[me] += self.pits[opp][land_pit]
                self.pits[opp][land_pit] = 0

        # Collect tuzdyk stones
        for side in range(2):
            if self.tuzdyk[side] >= 0:
                tuz_pit = self.tuzdyk[side]
                opp_side = 1 - side
                if self.pits[opp_side][tuz_pit] > 0:
                    self.kazan[side] += self.pits[opp_side][tuz_pit]
                    self.pits[opp_side][tuz_pit] = 0

        self.side = opp
        return True

    def is_terminal(self):
        if self.kazan[0] >= 82 or self.kazan[1] >= 82:
            return True
        if self.kazan[0] == 81 and self.kazan[1] == 81:
            return True
        me = self.side
        opp = 1 - me
        has_move = any(self.pits[me][i] > 0 and self.tuzdyk[opp] != i for i in range(9))
        return not has_move


def main():
    games_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'parsed_games', 'valid_games.json')
    if not os.path.exists(games_path):
        print(f"Games file not found: {games_path}")
        sys.exit(1)

    with open(games_path) as f:
        games = json.load(f)
    print(f"Loaded {len(games)} games")

    # Filter by Elo
    filtered = []
    for g in games:
        w_elo = g.get('white_elo', 0)
        b_elo = g.get('black_elo', 0)
        if max(w_elo, b_elo) >= MIN_ELO and len(g.get('moves', [])) > 4:
            filtered.append(g)
    print(f"Filtered to {len(filtered)} games (max Elo >= {MIN_ELO})")

    # Build position -> move stats
    positions = {}
    errors = 0

    for g in filtered:
        moves = g.get('moves', [])
        result_str = g.get('result', '')

        # Parse result: "1-0" = white win, "0-1" = black win, "1/2-1/2" = draw
        if result_str == '1-0':
            white_result = 'win'
        elif result_str == '0-1':
            white_result = 'loss'
        else:
            white_result = 'draw'

        board = Board()
        for move_idx, move_1indexed in enumerate(moves):
            if move_idx >= BOOK_DEPTH:
                break
            if board.is_terminal():
                break

            pit = move_1indexed - 1  # convert to 0-indexed
            if pit < 0 or pit >= 9:
                errors += 1
                break

            if not board.is_valid_move(pit):
                errors += 1
                break

            pos = board.pos_string()
            side = board.side

            if pos not in positions:
                positions[pos] = {'moves': {}, 'total': 0}

            move_key = str(pit)
            if move_key not in positions[pos]['moves']:
                positions[pos]['moves'][move_key] = {'count': 0, 'wins': 0, 'draws': 0, 'losses': 0}

            positions[pos]['moves'][move_key]['count'] += 1
            positions[pos]['total'] += 1

            # Record outcome from side-to-move perspective
            if side == 0:  # white to move
                if white_result == 'win':
                    positions[pos]['moves'][move_key]['wins'] += 1
                elif white_result == 'draw':
                    positions[pos]['moves'][move_key]['draws'] += 1
                else:
                    positions[pos]['moves'][move_key]['losses'] += 1
            else:  # black to move
                if white_result == 'loss':
                    positions[pos]['moves'][move_key]['wins'] += 1
                elif white_result == 'draw':
                    positions[pos]['moves'][move_key]['draws'] += 1
                else:
                    positions[pos]['moves'][move_key]['losses'] += 1

            board.make_move(pit)

    if errors > 0:
        print(f"  Skipped {errors} invalid moves")

    # Filter: only keep positions with enough games
    filtered_pos = {}
    for pos, data in positions.items():
        if data['total'] >= MIN_GAMES:
            # Also filter individual moves
            good_moves = {}
            for move, stats in data['moves'].items():
                if stats['count'] >= 2:
                    good_moves[move] = stats
            if good_moves:
                filtered_pos[pos] = {'moves': good_moves, 'total': data['total']}

    book = {
        'format': 'togyz_book_v1',
        'min_elo': MIN_ELO,
        'book_depth': BOOK_DEPTH,
        'source_games': len(filtered),
        'positions': filtered_pos,
    }

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'opening_book.json')
    with open(output_path, 'w') as f:
        json.dump(book, f, indent=2)

    print(f"\nOpening book generated:")
    print(f"  Positions: {len(filtered_pos)}")
    print(f"  Saved to: {output_path}")

    # Show initial position stats
    init_pos = '9,9,9,9,9,9,9,9,9/9,9,9,9,9,9,9,9,9/0,0/-1,-1/0'
    if init_pos in filtered_pos:
        print(f"\nInitial position moves:")
        for move, stats in sorted(filtered_pos[init_pos]['moves'].items(),
                                   key=lambda x: -x[1]['count']):
            wr = (stats['wins'] + stats['draws'] * 0.5) / max(stats['count'], 1)
            print(f"  Pit {int(move)+1}: {stats['count']} games, {wr*100:.0f}% win rate")


if __name__ == '__main__':
    main()
