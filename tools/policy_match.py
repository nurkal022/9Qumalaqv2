#!/usr/bin/env python3
"""Measure whether the NN policy is good enough to help alpha-beta move ordering.

For a sample of positions from champion games, compare the NN policy's ranking
against (a) the strong engine's best move (serve), and (b) the champion's actual
move. Move ordering benefits if the engine's best move is in the NN's top-1/top-3
often (search the likely-best move first -> more cutoffs).

Usage: python3.12 tools/policy_match.py <net.onnx> [num_positions] [engine_ms]
"""
import sys, os, re, subprocess, random
import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "archive/old-impls/alphazero-code/alphazero"))
from game import TogyzQumalaq  # noqa: E402

GAMES_DIR = os.path.join(ROOT, "archive/datasets/game-pars/games")
IDS = os.path.join(ROOT, "archive/datasets/game-pars/mcts_games.txt")
ENGINE = os.path.join(ROOT, "engine/target/release/togyzkumalaq-engine")


def parse_moves(text):
    body = "\n".join(l for l in text.splitlines() if not l.strip().startswith("["))
    body = re.sub(r"\b\d+\.", " ", body)
    body = re.sub(r"(1-0|0-1|1/2-1/2|\*)\s*$", " ", body)
    return [int(m.group(1)) - 1 for m in re.finditer(r"(\d)(\d)(X?)(?:\((\d+)\))?", body)
            if 1 <= int(m.group(1)) <= 9]


def pos_str(g):
    s = g.get_state()
    w = ",".join(str(int(x)) for x in s.pits[0]); b = ",".join(str(int(x)) for x in s.pits[1])
    return f"{w}/{b}/{int(s.kazan[0])},{int(s.kazan[1])}/{int(s.tuzdyk[0])},{int(s.tuzdyk[1])}/{int(s.current_player)}"


class Serve:
    def __init__(self, path):
        self.p = subprocess.Popen([path, "serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, cwd=os.path.join(ROOT, "engine"), text=True, bufsize=1)
        self._ready(); self.p.stdin.write("newgame\n"); self.p.stdin.flush(); self._ready()

    def _ready(self):
        while True:
            l = self.p.stdout.readline()
            if not l or l.strip() == "ready":
                return

    def best(self, g, ms):
        self.p.stdin.write(f"go pos {pos_str(g)} time {ms} nobook\n"); self.p.stdin.flush()
        while True:
            l = self.p.stdout.readline()
            if not l:
                return None
            if l.startswith("bestmove"):
                return int(l.split()[1])
            if l.startswith("terminal"):
                return None


def collect(num):
    ids = [l.strip() for l in open(IDS) if l.strip()]
    random.seed(7); random.shuffle(ids)
    out = []
    for gid in ids:
        try:
            text = open(os.path.join(GAMES_DIR, gid + ".txt"), encoding="utf-8", errors="ignore").read()
        except FileNotFoundError:
            continue
        moves = parse_moves(text)
        if len(moves) < 10:
            continue
        g = TogyzQumalaq(); g.reset()
        for ply, mv in enumerate(moves):
            if mv not in g.get_valid_moves_list():
                break
            # sample positions across the game (skip the very first plies)
            if ply >= 4 and len(g.get_valid_moves_list()) >= 3 and random.random() < 0.15:
                out.append((g.get_state().copy(), mv))
            g.make_move(mv)
        if len(out) >= num:
            break
    return out[:num]


def main():
    net = sys.argv[1]
    num = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    ms = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    print(f"sampling {num} positions from champion games...")
    positions = collect(num)
    print(f"  got {len(positions)} positions")
    sess = ort.InferenceSession(net, ort.SessionOptions(), providers=["CPUExecutionProvider"])
    eng = Serve(ENGINE)

    nn_eq_eng1 = nn_eng_top3 = nn_eq_champ = eng_eq_champ = n = 0
    for st, champ_mv in positions:
        g = TogyzQumalaq(); g.set_state(st)
        valid = g.get_valid_moves_list()
        state = g.encode_state().reshape(1, 7, 9).astype(np.float32)
        logp, _ = sess.run(None, {"state": state})
        pol = np.exp(np.array(logp).flatten())
        mask = np.full(9, -1.0); mask[valid] = pol[valid]
        order = [int(i) for i in np.argsort(-mask) if i in valid]
        nn_top1 = order[0]; nn_top3 = set(order[:3])
        eng_best = eng.best(g, ms)
        if eng_best is None or eng_best not in valid:
            continue
        n += 1
        if nn_top1 == eng_best: nn_eq_eng1 += 1
        if eng_best in nn_top3: nn_eng_top3 += 1
        if nn_top1 == champ_mv: nn_eq_champ += 1
        if eng_best == champ_mv: eng_eq_champ += 1
    eng.p.kill()
    print(f"\n=== policy quality over {n} positions (engine {ms}ms) ===")
    print(f"NN top-1 == engine best move : {100*nn_eq_eng1/max(1,n):.1f}%   (random ~{100/6:.0f}%)")
    print(f"engine best in NN top-3      : {100*nn_eng_top3/max(1,n):.1f}%")
    print(f"NN top-1 == champion move    : {100*nn_eq_champ/max(1,n):.1f}%")
    print(f"engine best == champion move : {100*eng_eq_champ/max(1,n):.1f}%")
    print("\n(NN priors help ordering if 'engine best in NN top-3' is high — search likely-best first.)")


if __name__ == "__main__":
    main()
