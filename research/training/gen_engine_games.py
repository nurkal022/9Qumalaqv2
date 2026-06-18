#!/usr/bin/env python3
"""
Generate ENGINE-vs-ENGINE games and label each position by the GAME OUTCOME
(sweep-correct), not by static eval. This is the "right" AlphaZero value target
-- what a position is worth under STRONG play -- and stronger than human +/-1
(humans play weaker than the deep engine). Pure CPU; scales across cores.

Output: per-worker .npy with states[N,7,9] (current-player perspective) and
values[N] in [-1,1] (sign = did side-to-move win; magnitude from swept margin).

Usage:
  python3.12 gen_engine_games.py --games 2000 --workers 24 --time-ms 120 \
      --out-dir /tmp/eng_games
"""
import sys, os, argparse, random, time
import numpy as np
import multiprocessing as mp

REPO = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, os.path.join(REPO, 'tools', 'playok'))
from engine import Engine, _parse_pos, START_POSITION, NUM_PITS  # noqa

PIT_NORM, KAZAN_NORM = 50.0, 82.0


def encode_pos(pos):
    w, b, kaz, tuz, side = _parse_pos(pos)
    me, opp = (w, b) if side == 0 else (b, w)
    mk, ok = (kaz[0], kaz[1]) if side == 0 else (kaz[1], kaz[0])
    mt, ot = (tuz[0], tuz[1]) if side == 0 else (tuz[1], tuz[0])
    s = np.zeros((7, 9), dtype=np.float32)
    s[0] = np.array(me, dtype=np.float32) / PIT_NORM
    s[1] = np.array(opp, dtype=np.float32) / PIT_NORM
    s[2] = mk / KAZAN_NORM
    s[3] = ok / KAZAN_NORM
    if mt >= 0:
        s[4, mt] = 1.0
    if ot >= 0:
        s[5, ot] = 1.0
    s[6] = 1.0 if side == 0 else 0.0
    return s


def swept_winner(pos):
    """Sweep rule: each side scoops its own remaining board stones into its kazan."""
    w, b, kaz, tuz, side = _parse_pos(pos)
    wt = kaz[0] + sum(x for x in w if x > 0)
    bt = kaz[1] + sum(x for x in b if x > 0)
    return wt, bt  # white_total, black_total


def worker(args):
    wid, n_games, time_ms, rand_open, max_plies, seed = args
    random.seed(seed)
    eng = Engine()
    eng.start()
    S, V = [], []
    for g in range(n_games):
        pos = START_POSITION
        recs = []  # (state, side_to_move)
        ropen = random.randint(rand_open[0], rand_open[1])
        for ply in range(max_plies):
            side = _parse_pos(pos)[4]
            if ply < ropen:
                legal = Engine.legal_moves(pos)
                if not legal:
                    break
                mv = random.choice(legal)
            else:
                r = eng.bestmove(pos, time_ms=time_ms)
                if isinstance(r, tuple):  # terminal
                    break
                mv = r
                recs.append((encode_pos(pos), side))
            try:
                pos = Engine.apply_move(pos, mv)
            except Exception:
                break
            # terminal if mover's opponent has no stones to move next
            if not Engine.legal_moves(pos):
                break
        wt, bt = swept_winner(pos)
        if wt == bt:
            continue  # skip draws (rare)
        winner = 0 if wt > bt else 1
        mag = min(1.0, max(0.3, abs(wt - bt) / 82.0))
        for st, side in recs:
            V.append(mag if side == winner else -mag)
            S.append(st)
    eng.stop()
    if S:
        np.save(f"/tmp/eng_games/w{wid}_s.npy", np.array(S, dtype=np.float32))
        np.save(f"/tmp/eng_games/w{wid}_v.npy", np.array(V, dtype=np.float32))
    return wid, len(S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', type=int, default=2000)
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--time-ms', type=int, default=120)
    ap.add_argument('--rand-open-min', type=int, default=2)
    ap.add_argument('--rand-open-max', type=int, default=8)
    ap.add_argument('--max-plies', type=int, default=220)
    ap.add_argument('--out', default='/tmp/eng_games/engine_outcomes.npz')
    args = ap.parse_args()
    os.makedirs('/tmp/eng_games', exist_ok=True)

    per = args.games // args.workers
    tasks = [(w, per + (1 if w < args.games % args.workers else 0), args.time_ms,
              (args.rand_open_min, args.rand_open_max), args.max_plies, 1000 + w)
             for w in range(args.workers)]
    print(f"Generating {args.games} engine-vs-engine games on {args.workers} workers "
          f"({args.time_ms}ms/move) ...", flush=True)
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        results = pool.map(worker, tasks)
    total = sum(n for _, n in results)
    # merge
    Ss, Vs = [], []
    for w in range(args.workers):
        sp, vp = f"/tmp/eng_games/w{w}_s.npy", f"/tmp/eng_games/w{w}_v.npy"
        if os.path.exists(sp):
            Ss.append(np.load(sp)); Vs.append(np.load(vp))
            os.remove(sp); os.remove(vp)
    S = np.concatenate(Ss); V = np.concatenate(Vs)
    np.savez(args.out, states=S, values=V)
    print(f"Done in {time.time()-t0:.0f}s: {len(S)} positions "
          f"(value mean={V.mean():.3f} std={V.std():.3f}) -> {args.out}", flush=True)


if __name__ == '__main__':
    main()
