#!/usr/bin/env python3
"""Match two engine binaries against each other via serve protocol."""
import subprocess, sys, os, random, time

NUM_PITS = 9
INITIAL_STONES = 9

class Board:
    def __init__(self):
        self.pits = [[INITIAL_STONES]*NUM_PITS, [INITIAL_STONES]*NUM_PITS]
        self.kazan = [0, 0]
        self.tuzdyk = [-1, -1]
        self.side_to_move = 0
        self.move_count = 0

    def to_pos(self):
        wp = ','.join(str(x) for x in self.pits[0])
        bp = ','.join(str(x) for x in self.pits[1])
        k = f"{self.kazan[0]},{self.kazan[1]}"
        t = f"{self.tuzdyk[0]},{self.tuzdyk[1]}"
        return f"{wp}/{bp}/{k}/{t}/{self.side_to_move}"

    def valid_moves(self):
        me = self.side_to_move
        opp = 1 - me
        opp_tuz = self.tuzdyk[opp]
        return [i for i in range(NUM_PITS) if self.pits[me][i] > 0 and opp_tuz != i]

    def can_create_tuzdyk(self, player, pit_index):
        if self.tuzdyk[player] != -1: return False
        if pit_index == 8: return False
        if self.tuzdyk[1 - player] == pit_index: return False
        return True

    def is_terminal(self):
        if self.kazan[0] >= 82 or self.kazan[1] >= 82: return True
        return all(x == 0 for x in self.pits[0]) or all(x == 0 for x in self.pits[1])

    def game_result(self):
        """Returns 0=white wins, 1=black wins, 2=draw, None=not terminal"""
        if self.kazan[0] >= 82: return 0
        if self.kazan[1] >= 82: return 1
        if self.kazan[0] == 81 and self.kazan[1] == 81: return 2
        w_empty = all(x == 0 for x in self.pits[0])
        b_empty = all(x == 0 for x in self.pits[1])
        if w_empty or b_empty:
            # Collect remaining stones
            k0 = self.kazan[0] + sum(self.pits[0])
            k1 = self.kazan[1] + sum(self.pits[1])
            if k0 > k1: return 0
            if k1 > k0: return 1
            return 2
        return None

    def make_move(self, pit_index):
        me = self.side_to_move
        opp = 1 - me
        stones = self.pits[me][pit_index]
        if stones <= 0: return False
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
        is_tuz = (current_side == opp and self.tuzdyk[me] == current_pit) or \
                 (current_side == me and self.tuzdyk[opp] == current_pit)
        if current_side == opp and not is_tuz:
            count = self.pits[opp][current_pit]
            if count == 3 and self.can_create_tuzdyk(me, current_pit):
                self.tuzdyk[me] = current_pit
                self.kazan[me] += count
                self.pits[opp][current_pit] = 0
            elif count % 2 == 0 and count > 0:
                self.kazan[me] += count
                self.pits[opp][current_pit] = 0
        self.side_to_move = 1 - self.side_to_move
        self.move_count += 1
        return True


class Engine:
    def __init__(self, binary, nnue_weights):
        # cwd must be the engine directory where weights/egtb files live
        engine_dir = os.path.dirname(os.path.abspath(nnue_weights))
        self.proc = subprocess.Popen(
            [os.path.abspath(binary), 'serve'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
            cwd=engine_dir,
        )
        line = self.proc.stdout.readline().strip()
        if line != 'ready':
            raise RuntimeError(f"Engine failed to start: {line}")

    def get_move(self, pos, time_ms=500):
        self.proc.stdin.write(f'go time {time_ms} pos {pos}\n')
        self.proc.stdin.flush()
        line = self.proc.stdout.readline().strip()
        if line.startswith('bestmove'):
            parts = line.split()
            for i, p in enumerate(parts):
                if p == 'bestmove' and i+1 < len(parts):
                    return int(parts[i+1])
        if line.startswith('terminal'):
            return -1
        return -1

    def newgame(self):
        self.proc.stdin.write('newgame\n')
        self.proc.stdin.flush()
        self.proc.stdout.readline()

    def close(self):
        try:
            self.proc.stdin.write('quit\n')
            self.proc.stdin.flush()
            self.proc.wait(timeout=3)
        except:
            self.proc.kill()


def play_game(eng_a, eng_b, time_ms=500, random_opening_plies=4):
    board = Board()
    eng_a.newgame()
    eng_b.newgame()
    # Random opening
    for _ in range(random_opening_plies):
        moves = board.valid_moves()
        if not moves or board.is_terminal(): break
        board.make_move(random.choice(moves))
    # Play
    for _ in range(300):
        if board.is_terminal(): break
        r = board.game_result()
        if r is not None: break
        eng = eng_a if board.side_to_move == 0 else eng_b
        mv = eng.get_move(board.to_pos(), time_ms)
        if mv < 0: break
        if mv not in board.valid_moves():
            break
        board.make_move(mv)
    r = board.game_result()
    if r is None:
        # Collect remaining
        k0 = board.kazan[0] + sum(board.pits[0])
        k1 = board.kazan[1] + sum(board.pits[1])
        if k0 > k1: r = 0
        elif k1 > k0: r = 1
        else: r = 2
    return r


def main():
    if len(sys.argv) < 4:
        print("Usage: match_engines.py <binary_a> <binary_b> <nnue_weights> [games=200] [time_ms=500]")
        sys.exit(1)

    bin_a, bin_b, weights = sys.argv[1], sys.argv[2], sys.argv[3]
    games = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    time_ms = int(sys.argv[5]) if len(sys.argv) > 5 else 500

    print(f"Engine A: {os.path.basename(bin_a)}")
    print(f"Engine B: {os.path.basename(bin_b)}")
    print(f"Weights: {os.path.basename(weights)}")
    print(f"Games: {games}, Time: {time_ms}ms")

    eng_a = Engine(bin_a, weights)
    eng_b = Engine(bin_b, weights)

    wins_a, wins_b, draws = 0, 0, 0
    for g in range(games):
        # Alternate colors
        if g % 2 == 0:
            r = play_game(eng_a, eng_b, time_ms)
            if r == 0: wins_a += 1
            elif r == 1: wins_b += 1
            else: draws += 1
        else:
            r = play_game(eng_b, eng_a, time_ms)
            if r == 1: wins_a += 1
            elif r == 0: wins_b += 1
            else: draws += 1

        if (g+1) % 20 == 0:
            total = wins_a + wins_b + draws
            score = (wins_a + draws * 0.5) / total * 100
            print(f"  [{g+1}/{games}] A: {wins_a}W {draws}D {wins_b}L ({score:.1f}%)")

    total = wins_a + wins_b + draws
    score = (wins_a + draws * 0.5) / total
    import math
    if 0 < score < 1:
        elo = -400 * math.log10(1/score - 1)
    else:
        elo = 999 if score >= 1 else -999
    print(f"\nFinal: A {wins_a}W - {draws}D - {wins_b}L ({score*100:.1f}%, Elo {elo:+.0f})")

    eng_a.close()
    eng_b.close()


if __name__ == '__main__':
    main()
