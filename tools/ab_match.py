#!/usr/bin/env python3
"""Engine-vs-engine A/B match harness for Togyzkumalak.

Pits two engine binaries against each other via the line-based `serve` protocol,
using the validated Python rules class (alphazero-code) as a neutral referee to
advance the board and detect terminal states. Colors are swapped every game for
fairness. Reports W/D/L from engine A's perspective with a simple Elo estimate.
Exits nonzero if A scores below --min (regression gate).

Usage:
    python3 tools/ab_match.py <engineA> <engineB> [games] [time_ms] \
        [--nobook] [--jobs N] [--min PCT] [--tt MB] [--save DIR]

Engines run with cwd=engine/ so they find egtb.bin / nnue_weights.bin /
opening_book.txt. Move indices on the wire are 0-8 (raw best_move).

--save DIR: write one JSON record per finished game, appended as it finishes (not
held in memory) to a single JSONL file `DIR/ab_match_<unix_ts>.jsonl` -- one file per
run, one line per game, so a crash mid-run only loses games not yet played, never
games already completed. Each record has both engines' paths + weights sha256 (see
tools/engine_provenance.py, shared with tools/9qum/match.py), which engine played
which colour, the full move list (0-8 wire indices, and the 1-9 human labels), the
final position and both kazans, the result from A's perspective, ply count, the time
control, and the git commit -- enough to replay and inspect any single game later
(see tools/9qum/lead_profile.py). Without --save, output/exit-code/scoring are
unchanged.
"""
import sys, os, subprocess, math, time, threading, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAME_DIR = os.path.join(ROOT, "research/alphazero")
assert os.path.isfile(os.path.join(GAME_DIR, "game.py")), f"rules class not found at {GAME_DIR}/game.py"
sys.path.insert(0, GAME_DIR)
from game import TogyzQumalaq  # noqa: E402
from engine_provenance import compute_engine_meta  # noqa: E402  (shared with tools/9qum/match.py)

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


def play_game(eng_white, eng_black, time_ms, max_plies=400, moves_out=None, final_state_out=None):
    """Play one game to a result (0=white, 1=black, 2=draw/unknown).

    `moves_out`, if given (a list), gets each move actually applied to the board
    appended to it in play order, as the wire-protocol 0-8 pit index -- an illegal
    move that ends the game in forfeit is NOT appended, since it was never applied.
    `final_state_out`, if given (a dict), is filled with the game's final pits/kazan/
    tuzdyk/side and its `go pos` string right before returning. Both are no-ops when
    left None (the default), so existing callers are unaffected.
    """
    g = TogyzQumalaq(); g.reset()
    result = None
    for _ in range(max_plies):
        if g.is_terminal():
            break
        side = g.get_state().current_player
        eng = eng_white if side == 0 else eng_black
        mv = eng.bestmove(g, time_ms)
        if mv is None:
            break
        if mv not in g.get_valid_moves_list():
            result = 1 - side  # illegal -> forfeit
            break
        if moves_out is not None:
            moves_out.append(mv)
        g.make_move(mv)
    if result is None:
        if g.is_terminal():
            w = g.get_winner()
            result = w if w is not None else 2
        else:
            result = 2
    if final_state_out is not None:
        s = g.get_state()
        final_state_out["pits"] = [[int(x) for x in s.pits[0]], [int(x) for x in s.pits[1]]]
        final_state_out["kazan"] = [int(s.kazan[0]), int(s.kazan[1])]
        final_state_out["tuzdyk"] = [int(s.tuzdyk[0]), int(s.tuzdyk[1])]
        final_state_out["side"] = int(s.current_player)
        final_state_out["pos"] = pos_str(g)
    return result


def elo(score, n):
    if n == 0:
        return 0.0
    p = min(max(score / n, 1e-4), 1 - 1e-4)
    return -400 * math.log10(1 / p - 1)


def make_game_record(meta_a, meta_b, game_index, a_white, moves, final_state, result_a,
                      time_ms, tt_mb, a_nobook, b_nobook, ts=None):
    """Build one JSON-able record for a finished --save game.

    `moves` is the wire-protocol (0-8) pit index for every move actually applied to
    the board, in play order, starting from the standard start position -- replaying
    them through tools/playok/engine.py's Engine.apply_move must reproduce
    `final_state`'s pits/kazan exactly (see tools/test_ab_match.py, and
    tools/9qum/lead_profile.py which does this for real analysis). `result_a` is
    "W"/"D"/"L" from engine A's perspective, matching the console tally exactly.
    `meta_a`/`meta_b` are tools/engine_provenance.compute_engine_meta(...) dicts.
    """
    return {
        "schema": "ab_match_game_v1",
        "engine_a": {"path": meta_a["engine_path"],
                     "weights_path": meta_a["engine_weights_path"],
                     "weights_size": meta_a["engine_weights_size"],
                     "weights_sha256": meta_a["engine_weights_sha256"]},
        "engine_b": {"path": meta_b["engine_path"],
                     "weights_path": meta_b["engine_weights_path"],
                     "weights_size": meta_b["engine_weights_size"],
                     "weights_sha256": meta_b["engine_weights_sha256"]},
        "white": "A" if a_white else "B",
        "black": "B" if a_white else "A",
        "moves": list(moves),
        "moves_1based": [m + 1 for m in moves],
        "final_position": final_state["pos"],
        "kazan": final_state["kazan"],
        "tuzdyk": final_state["tuzdyk"],
        "result_a": result_a,
        "plies": len(moves),
        "time_control_ms": time_ms,
        "tt_mb": tt_mb,
        "a_nobook": a_nobook,
        "b_nobook": b_nobook,
        # same repo for both engines in this harness, so either meta's git_commit does;
        # A's is used for a single unambiguous top-level field.
        "git_commit": meta_a["git_commit"],
        "game_index": game_index,
        "ts": ts if ts is not None else int(time.time()),
    }


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
    save_dir = opt("--save", None)

    # --save setup: computed once, up front, so a slow sha256 of the weights file
    # never happens mid-game. Nothing here executes when --save is absent.
    save_path = None
    meta_a = meta_b = None
    save_lock = threading.Lock()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"ab_match_{int(time.time())}.jsonl")
        meta_a = compute_engine_meta(a_path)
        meta_b = compute_engine_meta(b_path)
        print(f"saving game records to {save_path}")

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
                moves = [] if save_path else None
                final_state = {} if save_path else None
                if a_white:
                    res = play_game(A, B, time_ms, moves_out=moves, final_state_out=final_state)
                    a_won, b_won = (res == 0), (res == 1)
                else:
                    res = play_game(B, A, time_ms, moves_out=moves, final_state_out=final_state)
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
                if save_path:
                    result_a = "W" if a_won else ("L" if b_won else "D")
                    rec = make_game_record(
                        meta_a, meta_b, i, a_white, moves, final_state, result_a,
                        time_ms, tt_mb, a_nobook, b_nobook,
                    )
                    # write as this game finishes (never batched in memory): a crash
                    # partway through a 30-60min run must not lose completed games.
                    line = json.dumps(rec, ensure_ascii=False) + "\n"
                    with save_lock:
                        with open(save_path, "a", encoding="utf-8") as f:
                            f.write(line)
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
