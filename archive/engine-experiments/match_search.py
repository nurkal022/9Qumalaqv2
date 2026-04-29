#!/usr/bin/env python3
"""Match two different engine binaries against each other via analyze command."""
import subprocess
import json
import random
import sys

ENGINE_A = "./target/release/togyzkumalaq-engine-improved3"
ENGINE_B = "./target/release/togyzkumalaq-engine-original2"
NUM_GAMES = 200
TIME_MS = 300

class Board:
    def __init__(self):
        self.pits = [[9]*9, [9]*9]
        self.kazan = [0, 0]
        self.tuzdyk = [-1, -1]
        self.side = 0  # 0=white, 1=black

    def to_str(self):
        w = ",".join(str(x) for x in self.pits[0])
        b = ",".join(str(x) for x in self.pits[1])
        k = f"{self.kazan[0]},{self.kazan[1]}"
        t = f"{self.tuzdyk[0]},{self.tuzdyk[1]}"
        return f"{w}/{b}/{k}/{t}/{self.side}"

    def valid_moves(self):
        moves = []
        me = self.side
        opp = 1 - me
        for i in range(9):
            if self.pits[me][i] > 0 and self.tuzdyk[opp] != i:
                moves.append(i)
        return moves

    def make_move(self, pit):
        me = self.side
        opp = 1 - me
        stones = self.pits[me][pit]
        if stones == 0:
            return

        if stones == 1:
            self.pits[me][pit] = 0
            next_pit = pit + 1
            if next_pit > 8:
                side, pos = opp, 0
            else:
                side, pos = me, next_pit
            self.pits[side][pos] += 1
            last_side, last_pit = side, pos
        else:
            self.pits[me][pit] = 1  # leave one stone
            stones -= 1
            cur_side, cur_pit = me, pit + 1
            if cur_pit > 8:
                cur_side, cur_pit = 1 - cur_side, 0
            for s in range(stones):
                # Skip tuzdyk pits
                while (self.tuzdyk[0] == cur_pit and cur_side == 1) or \
                      (self.tuzdyk[1] == cur_pit and cur_side == 0):
                    cur_pit += 1
                    if cur_pit > 8:
                        cur_side, cur_pit = 1 - cur_side, 0

                self.pits[cur_side][cur_pit] += 1
                if s == stones - 1:
                    last_side, last_pit = cur_side, cur_pit
                cur_pit += 1
                if cur_pit > 8:
                    cur_side, cur_pit = 1 - cur_side, 0

        # Check capture
        if last_side == opp:
            count = self.pits[opp][last_pit]
            if count % 2 == 0:
                self.kazan[me] += count
                self.pits[opp][last_pit] = 0
            elif count == 3 and self.tuzdyk[me] == -1 and last_pit != 8:
                # Check tuzdyk rules
                if not (self.tuzdyk[opp] >= 0 and self.tuzdyk[opp] == last_pit):
                    if self.tuzdyk[opp] != 8 - last_pit:  # symmetric rule
                        self.tuzdyk[me] = last_pit

        # Collect tuzdyk
        for s in range(2):
            if self.tuzdyk[s] >= 0:
                opp_s = 1 - s
                t = self.tuzdyk[s]
                if self.pits[opp_s][t] > 0:
                    self.kazan[s] += self.pits[opp_s][t]
                    self.pits[opp_s][t] = 0

        self.side = 1 - self.side

    def is_terminal(self):
        if self.kazan[0] >= 82 or self.kazan[1] >= 82:
            return True
        if self.kazan[0] == 81 and self.kazan[1] == 81:
            return True
        # Check if current player has moves
        me = self.side
        opp = 1 - me
        has_move = False
        for i in range(9):
            if self.pits[me][i] > 0 and self.tuzdyk[opp] != i:
                has_move = True
                break
        if not has_move:
            return True
        return False

    def result(self):
        """Returns 1 if white wins, -1 if black wins, 0 for draw."""
        # Collect remaining stones
        total = [self.kazan[0], self.kazan[1]]
        for s in range(2):
            total[s] += sum(self.pits[s])
        if total[0] > total[1]:
            return 1
        elif total[1] > total[0]:
            return -1
        return 0

def engine_move(engine_path, board_str, time_ms):
    try:
        r = subprocess.run(
            [engine_path, "analyze", board_str, str(time_ms)],
            capture_output=True, text=True, timeout=time_ms/1000 + 10
        )
        for line in r.stdout.strip().split('\n'):
            line = line.strip()
            if line.startswith('{'):
                data = json.loads(line)
                return data['bestmove']
    except Exception as e:
        pass
    # Fallback: random move
    return None

def main():
    a_wins, b_wins, draws = 0, 0, 0
    rng = random.Random(42)

    for game_num in range(NUM_GAMES):
        a_is_white = game_num % 2 == 0
        board = Board()

        # Random opening (4 plies), same for paired games
        if game_num % 2 == 0:
            opening_moves = []
            for _ in range(4):
                moves = board.valid_moves()
                if not moves or board.is_terminal():
                    break
                m = rng.choice(moves)
                opening_moves.append(m)
                board.make_move(m)
            saved_opening = opening_moves
        else:
            board = Board()
            for m in saved_opening:
                board.make_move(m)

        move_count = 0
        while not board.is_terminal() and move_count < 300:
            is_white = board.side == 0
            use_a = is_white == a_is_white
            engine = ENGINE_A if use_a else ENGINE_B

            mv = engine_move(engine, board.to_str(), TIME_MS)
            if mv is None:
                moves = board.valid_moves()
                if not moves:
                    break
                mv = moves[0]

            board.make_move(mv)
            move_count += 1

        r = board.result()
        if r == 1:  # white wins
            if a_is_white: a_wins += 1
            else: b_wins += 1
        elif r == -1:  # black wins
            if not a_is_white: a_wins += 1
            else: b_wins += 1
        else:
            draws += 1

        total = game_num + 1
        if total % 10 == 0 or total == NUM_GAMES:
            score = (a_wins + draws * 0.5) / total * 100
            print(f"Game {total}/{NUM_GAMES}: A {a_wins}-{draws}-{b_wins} B ({score:.1f}%)", flush=True)

    print(f"\nFinal: A {a_wins} - {draws} - {b_wins} B")
    score = (a_wins + draws * 0.5) / NUM_GAMES
    print(f"A (improved) score: {score*100:.1f}%")
    if score > 0.5:
        import math
        elo = -400 * math.log10(1/score - 1)
        print(f"A Elo advantage: +{elo:.0f}")
    elif score < 0.5:
        import math
        elo = -400 * math.log10(1/(1-score) - 1)
        print(f"B Elo advantage: +{elo:.0f}")

if __name__ == "__main__":
    main()
