#!/usr/bin/env python3
"""A/B match of two engine builds via the `serve` protocol.

Referee uses the official game-end rule (atsyz qalu): the game ends when
the side to move has no stones; the stones left on the board go to the
player on whose side they lie.

Methodology (so results mean something):
  * games are played in pairs — the same random opening with colors swapped,
    which cancels most of the opening/first-move bias;
  * several pairs run in parallel, each worker pinned to its own CPU core
      (both engines of a game share that core, they never think at once);
  * reports Elo with a 95% confidence interval and an SPRT log-likelihood
    ratio, and stops early once the SPRT bounds are crossed.

Examples:
  # new build vs baseline, same weights (engine/ dir), 100 ms/move
  python3 match_engines.py --a target/release/togyzkumalaq-engine \\
      --b /path/to/engine_old --games 400 --time 100

  # two different NNUE nets: point each engine at a dir with its own
  # nnue_weights.bin (plus optional opening_book.txt / egtb.bin)
  python3 match_engines.py --a ./engine --dir-a netA/ --b ./engine --dir-b netB/
"""
import argparse
import math
import multiprocessing as mp
import os
import random
import subprocess
import sys

NUM_PITS = 9


class Board:
    def __init__(self):
        self.pits = [[9] * NUM_PITS, [9] * NUM_PITS]
        self.kazan = [0, 0]
        self.tuzdyk = [-1, -1]
        self.side_to_move = 0

    def to_pos(self):
        wp = ','.join(map(str, self.pits[0]))
        bp = ','.join(map(str, self.pits[1]))
        return (f"{wp}/{bp}/{self.kazan[0]},{self.kazan[1]}/"
                f"{self.tuzdyk[0]},{self.tuzdyk[1]}/{self.side_to_move}")

    def valid_moves(self):
        me, opp = self.side_to_move, 1 - self.side_to_move
        return [i for i in range(NUM_PITS)
                if self.pits[me][i] > 0 and self.tuzdyk[opp] != i]

    def game_result(self):
        """0 = white wins, 1 = black wins, 2 = draw, None = game goes on."""
        if self.kazan[0] >= 82:
            return 0
        if self.kazan[1] >= 82:
            return 1
        me, opp = self.side_to_move, 1 - self.side_to_move
        if all(x == 0 for x in self.pits[me]):
            k_me = self.kazan[me]
            k_opp = self.kazan[opp] + sum(self.pits[opp])
            if k_me == k_opp:
                return 2
            return me if k_me > k_opp else opp
        return None

    def make_move(self, pit):
        me, opp = self.side_to_move, 1 - self.side_to_move
        stones = self.pits[me][pit]
        assert stones > 0
        self.pits[me][pit] = 0
        side, cur = me, pit
        if stones == 1:
            to_sow = 1
        else:
            self.pits[me][pit] += 1
            to_sow = stones - 1
        for _ in range(to_sow):
            cur += 1
            if cur > 8:
                cur = 0
                side = 1 - side
            if side == opp and self.tuzdyk[me] == cur:
                self.kazan[me] += 1
            elif side == me and self.tuzdyk[opp] == cur:
                self.kazan[opp] += 1
            else:
                self.pits[side][cur] += 1
        on_tuz = ((side == opp and self.tuzdyk[me] == cur) or
                  (side == me and self.tuzdyk[opp] == cur))
        if side == opp and not on_tuz:
            count = self.pits[opp][cur]
            if (count == 3 and self.tuzdyk[me] == -1 and cur != 8
                    and self.tuzdyk[opp] != cur):
                self.tuzdyk[me] = cur
                self.kazan[me] += count
                self.pits[opp][cur] = 0
            elif count % 2 == 0 and count > 0:
                self.kazan[me] += count
                self.pits[opp][cur] = 0
        self.side_to_move = opp


class Engine:
    def __init__(self, binary, workdir, cpu):
        cmd = [os.path.abspath(binary), 'serve']
        if cpu is not None and sys.platform.startswith('linux'):
            cmd = ['taskset', '-c', str(cpu)] + cmd
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
            cwd=os.path.abspath(workdir))
        line = self.proc.stdout.readline().strip()
        if line != 'ready':
            raise RuntimeError(f"{binary}: engine failed to start: {line!r}")

    def cmd(self, line):
        self.proc.stdin.write(line + '\n')
        self.proc.stdin.flush()
        return self.proc.stdout.readline().strip()

    def newgame(self):
        self.cmd('newgame')

    def position(self, pos):
        self.cmd(f'position {pos}')

    def go(self, pos, time_ms):
        parts = self.cmd(f'go time {time_ms} pos {pos}').split()
        if len(parts) >= 2 and parts[0] == 'bestmove':
            return int(parts[1])
        return -1

    def close(self):
        try:
            self.proc.stdin.write('quit\n')
            self.proc.stdin.flush()
            self.proc.wait(timeout=3)
        except Exception:
            self.proc.kill()


def random_opening(rng, plies):
    board, moves = Board(), []
    for _ in range(plies):
        legal = board.valid_moves()
        if not legal or board.game_result() is not None:
            break
        m = rng.choice(legal)
        board.make_move(m)
        moves.append(m)
    return moves


def play_game(white, black, opening, time_ms, max_plies):
    """Returns 0/1/2 (white win / black win / draw) and the ply count."""
    board = Board()
    engines = (white, black)
    for e in engines:
        e.newgame()
    for m in opening:
        board.make_move(m)
    # Both engines learn the opening positions for repetition detection.
    for e in engines:
        e.position(board.to_pos())
    plies = 0
    while board.game_result() is None and plies < max_plies:
        eng = engines[board.side_to_move]
        mv = eng.go(board.to_pos(), time_ms)
        if mv not in board.valid_moves():
            # Illegal/no move: forfeit.
            return 1 - board.side_to_move, plies
        board.make_move(mv)
        engines[board.side_to_move].position(board.to_pos())
        plies += 1
    r = board.game_result()
    if r is None:  # adjudicate by material incl. stones on own side
        k0 = board.kazan[0] + sum(board.pits[0])
        k1 = board.kazan[1] + sum(board.pits[1])
        r = 2 if k0 == k1 else (0 if k0 > k1 else 1)
    return r, plies


def worker(args, cpu, jobs, results):
    a = Engine(args.a, args.dir_a, cpu)
    b = Engine(args.b, args.dir_b, cpu)
    try:
        while True:
            job = jobs.get()
            if job is None:
                break
            opening = job
            pair = []
            # Game 1: A white. Game 2: B white. Score from A's point of view.
            r, _ = play_game(a, b, opening, args.time, args.max_plies)
            pair.append({0: 1.0, 1: 0.0, 2: 0.5}[r])
            r, _ = play_game(b, a, opening, args.time, args.max_plies)
            pair.append({0: 0.0, 1: 1.0, 2: 0.5}[r])
            results.put(pair)
    finally:
        a.close()
        b.close()


def elo(score):
    score = min(max(score, 1e-6), 1 - 1e-6)
    return -400 * math.log10(1 / score - 1)


def pentanomial_stats(pairs):
    """Mean score and its std. error, treating each game pair as one sample
    (pairs with the same opening are correlated, so this is the honest
    variance estimate)."""
    n = len(pairs)
    xs = [sum(p) / 2 for p in pairs]
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / max(n - 1, 1)
    return mean, math.sqrt(var / n)


def sprt_llr(pairs, elo0, elo1):
    """Normal-approximation GSPRT on pair scores (as in fishtest)."""
    n = len(pairs)
    if n < 2:
        return 0.0
    mean, se = pentanomial_stats(pairs)
    var = (se ** 2) * n
    if var <= 0:
        return 0.0
    s0 = 1 / (1 + 10 ** (-elo0 / 400))
    s1 = 1 / (1 + 10 ** (-elo1 / 400))
    return n * (s1 - s0) * (2 * mean - s0 - s1) / (2 * var)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument('--a', required=True, help='engine A binary (the candidate)')
    p.add_argument('--b', required=True, help='engine B binary (the baseline)')
    p.add_argument('--dir-a', default=here, help='cwd for A (nnue_weights.bin, book, egtb)')
    p.add_argument('--dir-b', default=here, help='cwd for B')
    p.add_argument('--games', type=int, default=400, help='max games (rounded to pairs)')
    p.add_argument('--time', type=int, default=100, help='ms per move')
    p.add_argument('--concurrency', type=int, default=os.cpu_count() or 1)
    p.add_argument('--cpu-offset', type=int, default=0,
                   help='first CPU to pin workers to (to run two matches side by side)')
    p.add_argument('--opening-plies', type=int, default=6)
    p.add_argument('--max-plies', type=int, default=400)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--elo0', type=float, default=0.0)
    p.add_argument('--elo1', type=float, default=10.0)
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--beta', type=float, default=0.05)
    p.add_argument('--no-sprt', action='store_true', help='play all games, no early stop')
    args = p.parse_args()

    n_pairs = max(1, args.games // 2)
    rng = random.Random(args.seed)
    jobs, results = mp.Queue(), mp.Queue()
    for _ in range(n_pairs):
        jobs.put(random_opening(rng, args.opening_plies))
    for _ in range(args.concurrency):
        jobs.put(None)

    lower = math.log(args.beta / (1 - args.alpha))
    upper = math.log((1 - args.beta) / args.alpha)
    print(f"A: {args.a} [{args.dir_a}]\nB: {args.b} [{args.dir_b}]")
    print(f"{n_pairs * 2} games max, {args.time} ms/move, {args.concurrency} workers, "
          f"SPRT elo0={args.elo0} elo1={args.elo1} bounds=({lower:.2f}, {upper:.2f})")

    procs = [mp.Process(target=worker, args=(args, (args.cpu_offset + i) % (os.cpu_count() or 1), jobs, results))
             for i in range(args.concurrency)]
    for pr in procs:
        pr.start()

    pairs, w, d, l = [], 0, 0, 0
    verdict = None
    try:
        for _ in range(n_pairs):
            pair = results.get()
            pairs.append(pair)
            for s in pair:
                w += s == 1.0
                d += s == 0.5
                l += s == 0.0
            mean, se = pentanomial_stats(pairs)
            llr = sprt_llr(pairs, args.elo0, args.elo1)
            if len(pairs) % 10 == 0 or len(pairs) == n_pairs:
                lo, hi = elo(mean - 1.96 * se), elo(mean + 1.96 * se)
                print(f"  {2 * len(pairs):4d} games  A {w}W {d}D {l}L  "
                      f"{mean * 100:5.1f}%  Elo {elo(mean):+6.1f} [{lo:+.0f}, {hi:+.0f}]  "
                      f"LLR {llr:+.2f}", flush=True)
            if not args.no_sprt and len(pairs) >= 20:
                if llr >= upper:
                    verdict = 'H1 accepted: A is stronger'
                    break
                if llr <= lower:
                    verdict = 'H0 accepted: A is not stronger'
                    break
    finally:
        for pr in procs:
            pr.terminate()

    mean, se = pentanomial_stats(pairs)
    lo, hi = elo(mean - 1.96 * se), elo(mean + 1.96 * se)
    print(f"\nFinal: {2 * len(pairs)} games, A {w}W {d}D {l}L ({mean * 100:.1f}%)")
    print(f"Elo A-B: {elo(mean):+.1f}  95% CI [{lo:+.1f}, {hi:+.1f}]")
    print(f"SPRT: {verdict or 'inconclusive'} (LLR {sprt_llr(pairs, args.elo0, args.elo1):+.2f})")


if __name__ == '__main__':
    main()
