#!/usr/bin/env python3
"""Engine-vs-engine A/B match harness for Togyzkumalak.

Pits two engine binaries against each other via the line-based `serve` protocol,
using the validated Python rules class (alphazero-code) as a neutral referee to
advance the board and detect terminal states. Colors are swapped every game for
fairness. Reports W/D/L from engine A's perspective with a simple Elo estimate.
Exits nonzero if A scores below --min (regression gate).

Usage:
    python3 tools/ab_match.py <engineA> <engineB> [games] [time_ms] \
        [--nobook] [--jobs N] [--min PCT] [--tt MB]

Engines run with cwd=engine/ so they find egtb.bin / nnue_weights.bin /
opening_book.txt. Move indices on the wire are 0-8 (raw best_move).
"""
import sys, os, subprocess, math, time, threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAME_DIR = os.path.join(ROOT, "archive/old-impls/alphazero-code/alphazero")
assert os.path.isfile(os.path.join(GAME_DIR, "game.py")), f"rules class not found at {GAME_DIR}/game.py"
sys.path.insert(0, GAME_DIR)
from game import TogyzQumalaq  # noqa: E402

ENGINE_CWD = os.path.join(ROOT, "engine")


def pos_str(g):
    s = g.get_state()
    w = ",".join(str(int(x)) for x in s.pits[0])
    b = ",".join(str(int(x)) for x in s.pits[1])
    return f"{w}/{b}/{int(s.kazan[0])},{int(s.kazan[1])}/{int(s.tuzdyk[0])},{int(s.tuzdyk[1])}/{int(s.current_player)}"


class Engine:
    def __init__(self, path, tt_mb, nobook=False):
        env = dict(os.environ, TT_SIZE_MB=str(tt_mb))
        self.path = path
        self.nobook = nobook
        self.p = subprocess.Popen(
            [os.path.abspath(path), "serve"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            cwd=ENGINE_CWD, text=True, bufsize=1, env=env,
        )
        self._ready()

    def _ready(self):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError(f"{self.path}: engine died during init")
            if line.strip() == "ready":
                return

    def newgame(self):
        self.p.stdin.write("newgame\n"); self.p.stdin.flush()
        self._ready()

    def bestmove(self, g, time_ms):
        cmd = f"go pos {pos_str(g)} time {time_ms}{' nobook' if self.nobook else ''}\n"
        self.p.stdin.write(cmd); self.p.stdin.flush()
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError(f"{self.path}: engine died during search")
            line = line.strip()
            if line.startswith("bestmove"):
                return int(line.split()[1])
            if line.startswith("terminal"):
                return None
            if line.startswith("error"):
                raise RuntimeError(f"{self.path}: {line}")

    def close(self):
        try:
            self.p.stdin.write("quit\n"); self.p.stdin.flush()
            self.p.wait(timeout=2)
        except Exception:
            self.p.kill()


def play_game(eng_white, eng_black, time_ms, max_plies=400):
    g = TogyzQumalaq(); g.reset()
    for _ in range(max_plies):
        if g.is_terminal():
            break
        side = g.get_state().current_player
        eng = eng_white if side == 0 else eng_black
        mv = eng.bestmove(g, time_ms)
        if mv is None:
            break
        if mv not in g.get_valid_moves_list():
            return 1 - side  # illegal -> forfeit
        g.make_move(mv)
    if g.is_terminal():
        w = g.get_winner()
        return w if w is not None else 2
    return 2


def elo(score, n):
    if n == 0:
        return 0.0
    p = min(max(score / n, 1e-4), 1 - 1e-4)
    return -400 * math.log10(1 / p - 1)


def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(2)
    a_path, b_path = sys.argv[1], sys.argv[2]
    pos = [x for x in sys.argv[3:] if not x.startswith("--")]
    games = int(pos[0]) if len(pos) > 0 else 20
    time_ms = int(pos[1]) if len(pos) > 1 else 200

    def opt(name, default):
        if name in sys.argv:
            i = sys.argv.index(name)
            return sys.argv[i + 1] if i + 1 < len(sys.argv) else default
        return default

    nobook = "--nobook" in sys.argv
    a_nobook = nobook or "--a-nobook" in sys.argv
    b_nobook = nobook or "--b-nobook" in sys.argv
    jobs = int(opt("--jobs", "6"))
    min_pct = float(opt("--min", "0"))
    tt_mb = int(opt("--tt", "64"))

    # assign games to workers; game index i -> A plays white iff i even (fairness)
    lock = threading.Lock()
    tally = {"aw": 0, "ad": 0, "al": 0, "done": 0, "err": 0}

    def worker(indices):
        try:
            A = Engine(a_path, tt_mb, a_nobook); B = Engine(b_path, tt_mb, b_nobook)
        except Exception as e:
            with lock:
                tally["err"] += len(indices)
            print(f"  worker init failed: {e}", flush=True)
            return
        try:
            for i in indices:
                A.newgame(); B.newgame()
                a_white = (i % 2 == 0)
                if a_white:
                    res = play_game(A, B, time_ms)
                    a_won, b_won = (res == 0), (res == 1)
                else:
                    res = play_game(B, A, time_ms)
                    a_won, b_won = (res == 1), (res == 0)
                with lock:
                    if a_won: tally["aw"] += 1
                    elif b_won: tally["al"] += 1
                    else: tally["ad"] += 1
                    tally["done"] += 1
                    d = tally["done"]
                    sc = tally["aw"] + 0.5 * tally["ad"]
                    print(f"  {d:>3}/{games}  A:{tally['aw']}W-{tally['ad']}D-{tally['al']}L  "
                          f"({100*sc/d:.1f}%)", flush=True)
        finally:
            A.close(); B.close()

    buckets = [[] for _ in range(jobs)]
    for i in range(games):
        buckets[i % jobs].append(i)
    t0 = time.time()
    threads = [threading.Thread(target=worker, args=(b,)) for b in buckets if b]
    for t in threads: t.start()
    for t in threads: t.join()

    aw, ad, al = tally["aw"], tally["ad"], tally["al"]
    n = aw + ad + al
    score = aw + 0.5 * ad
    dt = time.time() - t0
    pct = 100 * score / n if n else 0
    print("\n=== RESULT ===")
    print(f"A = {a_path}")
    print(f"B = {b_path}")
    print(f"A: {aw}W-{ad}D-{al}L over {n} games  =  {pct:.1f}%   "
          f"(Elo A-B ~ {elo(score, n):+.0f})   [{dt:.0f}s, {time_ms}ms/move, {jobs} jobs, TT={tt_mb}MB]")
    if min_pct > 0 and pct < min_pct:
        print(f"FAIL: {pct:.1f}% < required {min_pct:.1f}%")
        sys.exit(1)


if __name__ == "__main__":
    main()
