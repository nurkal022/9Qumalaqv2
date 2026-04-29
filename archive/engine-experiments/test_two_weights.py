#!/usr/bin/env python3
"""Test two NNUE weight files against each other using the same engine binary."""
import subprocess, sys, os, random, time, shutil, tempfile

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_BIN = os.path.join(ENGINE_DIR, 'target', 'release', 'togyzkumalaq-engine')

NUM_PITS = 9
INITIAL_STONES = 9


class Board:
    def __init__(self):
        self.pits = [[INITIAL_STONES]*NUM_PITS, [INITIAL_STONES]*NUM_PITS]
        self.kazan = [0, 0]
        self.tuzdyk = [-1, -1]
        self.side_to_move = 0

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

    def can_create_tuzdyk(self, player, pit_idx):
        if self.tuzdyk[player] != -1:
            return False
        opp = 1 - player
        if pit_idx == 8:
            return False
        if self.tuzdyk[opp] == pit_idx:
            return False
        return True

    def make_move(self, pit):
        me = self.side_to_move
        opp = 1 - me
        stones = self.pits[me][pit]
        if stones == 0:
            return False, None
        self.pits[me][pit] = 0
        cur = pit
        side = me
        for _ in range(stones):
            cur += 1
            if cur == NUM_PITS:
                cur = 0
                side = 1 - side
            target_tuz = self.tuzdyk[1 - side]
            if side != me and target_tuz == cur:
                self.pits[me][cur] = 0
                self.kazan[me] += 1
                continue
            self.pits[side][cur] += 1
            if side == opp and self.pits[opp][cur] % 2 == 0 and self.pits[opp][cur] > 0:
                captured = self.pits[opp][cur]
                self.pits[opp][cur] = 0
                self.kazan[me] += captured
            if side == opp and self.can_create_tuzdyk(me, cur) and self.pits[opp][cur] == 3:
                self.tuzdyk[me] = cur
                self.pits[opp][cur] = 0
                self.kazan[me] += 3

        total_a = sum(self.pits[0]) + self.kazan[0]
        total_b = sum(self.pits[1]) + self.kazan[1]
        max_possible = NUM_PITS * INITIAL_STONES * 2
        if self.kazan[0] > max_possible // 2 or self.kazan[1] > max_possible // 2:
            if self.kazan[0] > self.kazan[1]:
                return True, 0
            elif self.kazan[1] > self.kazan[0]:
                return True, 1
            else:
                return True, 2

        if not any(self.pits[1 - me]):
            if self.kazan[0] > self.kazan[1]:
                return True, 0
            elif self.kazan[1] > self.kazan[0]:
                return True, 1
            else:
                return True, 2

        self.side_to_move = 1 - me
        return False, None


def start_engine(weights_path):
    """Start engine with specific weights by using that file's directory as cwd."""
    weights_dir = os.path.dirname(os.path.abspath(weights_path))
    weights_name = os.path.basename(weights_path)
    # Create temp dir with symlinks to needed files
    tmpdir = tempfile.mkdtemp()
    # Link egtb file
    egtb_src = os.path.join(ENGINE_DIR, 'egtb.bin')
    if os.path.exists(egtb_src):
        os.symlink(egtb_src, os.path.join(tmpdir, 'egtb.bin'))
    # Link opening book
    ob_src = os.path.join(ENGINE_DIR, 'opening_book.bin')
    if os.path.exists(ob_src):
        os.symlink(ob_src, os.path.join(tmpdir, 'opening_book.bin'))
    # Copy weights as nnue_weights.bin
    shutil.copy(weights_path, os.path.join(tmpdir, 'nnue_weights.bin'))

    proc = subprocess.Popen(
        [ENGINE_BIN, 'serve'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, cwd=tmpdir,
    )
    line = proc.stdout.readline().strip()
    if line != 'ready':
        raise RuntimeError(f"Engine failed to start: {line!r}")
    proc._tmpdir = tmpdir
    return proc


def get_move(proc, pos, time_ms):
    proc.stdin.write(f'go time {time_ms} pos {pos}\n')
    proc.stdin.flush()
    line = proc.stdout.readline().strip()
    if line.startswith('bestmove'):
        parts = line.split()
        for i, p in enumerate(parts):
            if p == 'bestmove' and i+1 < len(parts):
                return int(parts[i+1])
    return None


def play_game(proc_a, proc_b, time_ms, opening_rng=None):
    """Returns 0 if A wins, 1 if B wins, 2 if draw."""
    board = Board()
    proc_a.stdin.write('newgame\n'); proc_a.stdin.flush(); proc_a.stdout.readline()
    proc_b.stdin.write('newgame\n'); proc_b.stdin.flush(); proc_b.stdout.readline()

    # Random opening (4 plies)
    if opening_rng:
        for _ in range(4):
            moves = board.valid_moves()
            if not moves:
                break
            m = opening_rng.choice(moves)
            done, winner = board.make_move(m)
            if done:
                return winner if winner != 2 else 2

    for move_num in range(300):
        pos = board.to_pos()
        if board.side_to_move == 0:
            m = get_move(proc_a, pos, time_ms)
        else:
            m = get_move(proc_b, pos, time_ms)

        if m is None or m not in board.valid_moves():
            moves = board.valid_moves()
            if not moves:
                break
            m = moves[0]

        done, winner = board.make_move(m)
        if done:
            return winner if winner is not None and winner != 2 else 2

    k0, k1 = board.kazan[0] + sum(board.pits[0]), board.kazan[1] + sum(board.pits[1])
    if k0 > k1:
        return 0
    elif k1 > k0:
        return 1
    else:
        return 2


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('weights_a', help='Path to weights A (e.g., Gen7)')
    parser.add_argument('weights_b', help='Path to weights B (e.g., Gen8_supervised)')
    parser.add_argument('--games', type=int, default=40)
    parser.add_argument('--time', type=int, default=1000, help='Engine think time ms')
    parser.add_argument('--label-a', default='Gen7')
    parser.add_argument('--label-b', default='Gen8_supervised')
    args = parser.parse_args()

    print(f"A: {args.label_a} ({os.path.basename(args.weights_a)})")
    print(f"B: {args.label_b} ({os.path.basename(args.weights_b)})")
    print(f"Games: {args.games}, Time: {args.time}ms")

    proc_a = start_engine(args.weights_a)
    proc_b = start_engine(args.weights_b)

    wins_a = wins_b = draws = 0
    rng = random.Random(42)

    for g in range(args.games):
        opening_rng = random.Random(g * 1000)
        if g % 2 == 0:
            # A=White, B=Black
            result = play_game(proc_a, proc_b, args.time, opening_rng)
            if result == 0: wins_a += 1
            elif result == 1: wins_b += 1
            else: draws += 1
        else:
            # B=White, A=Black
            result = play_game(proc_b, proc_a, args.time, opening_rng)
            if result == 0: wins_b += 1
            elif result == 1: wins_a += 1
            else: draws += 1

        total = wins_a + wins_b + draws
        score_a = (wins_a + 0.5 * draws) / total * 100
        print(f"  [{g+1}/{args.games}] {args.label_a}: {wins_a}W {draws}D {wins_b}L ({score_a:.1f}%)", flush=True)

    proc_a.stdin.write('quit\n'); proc_a.stdin.flush()
    proc_b.stdin.write('quit\n'); proc_b.stdin.flush()
    proc_a.terminate(); proc_b.terminate()

    total = wins_a + wins_b + draws
    score_a = (wins_a + 0.5 * draws) / total * 100
    print(f"\nFinal: {args.label_a} {wins_a}W-{draws}D-{wins_b}L = {score_a:.1f}%")
    if score_a > 55:  # A wins more than 55% = A is better
        print(f"  {args.label_b} is BETTER (+Elo above {args.label_a})")
    elif score_a < 45:
        print(f"  {args.label_a} is better ({args.label_b} FAILED)")
    else:
        print("  Results inconclusive (roughly equal)")


if __name__ == '__main__':
    main()

# Note: labels are A=first arg (Gen7), B=second arg (Gen8). score_a=Gen7 winrate.
# Fixed logic: if score_a < 45%, B is better (not A).
