#!/usr/bin/env python3
"""Floor-free diagnostic: does the value head predict the TRUE (swept) endgame
outcome? Compares two ONNX nets on near-terminal positions from champion games.

For each game that ends by board-emptying (both kazans <82 — the cases the old
raw-kazan label bug got wrong), we take positions in the last K plies, compute the
true swept value from the side-to-move's perspective, and check each net's value
SIGN against it. A net trained on corrected labels should match the swept truth
more often (and the old raw-kazan winner less often).

Usage: python3.12 tools/value_calibration.py <netA.onnx> <netB.onnx>
"""
import sys, os, re
import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research/alphazero"))
from game import TogyzQumalaq, GameState, Player  # noqa: E402
import numpy as _np

GAMES_DIR = os.path.join(ROOT, "archive/datasets/game-pars/games")
IDS = os.path.join(ROOT, "archive/datasets/game-pars/mcts_games.txt")
LAST_K = 12  # plies before terminal to sample


def parse_moves(text):
    body = "\n".join(l for l in text.splitlines() if not l.strip().startswith("["))
    body = re.sub(r"\b\d+\.", " ", body)
    body = re.sub(r"(1-0|0-1|1/2-1/2|\*)\s*$", " ", body)
    return [int(m.group(1)) - 1 for m in re.finditer(r"(\d)(\d)(X?)(?:\((\d+)\))?", body)
            if 1 <= int(m.group(1)) <= 9]


def collect_positions():
    """Return list of (GameState_copy, true_swept_value_from_stm, raw_value_from_stm,
    is_empty_side_ending)."""
    ids = [l.strip() for l in open(IDS) if l.strip()]
    out = []
    for gid in ids:
        fp = os.path.join(GAMES_DIR, gid + ".txt")
        try:
            text = open(fp, encoding="utf-8", errors="ignore").read()
        except FileNotFoundError:
            continue
        moves = parse_moves(text)
        if len(moves) < 6:
            continue
        g = TogyzQumalaq(); g.reset()
        states = []
        ok = True
        for mv in moves:
            if mv not in g.get_valid_moves_list():
                ok = False; break
            states.append(g.get_state().copy())
            g.make_move(mv)
        if not ok or not g.is_terminal():
            continue
        s = g.get_state()
        kw, kb = int(s.kazan[0]), int(s.kazan[1])
        sw, sb = int(s.pits[0].sum()), int(s.pits[1].sum())
        if kw >= 82 or kb >= 82:
            empty_ending = False
        elif sw == 0 or sb == 0:
            empty_ending = True
        else:
            continue  # not a clean terminal
        # true swept winner / raw winner (0=W,1=B,2=draw)
        swept = 0 if (kw + sw) > (kb + sb) else (1 if (kb + sb) > (kw + sw) else 2)
        raw = 0 if kw > kb else (1 if kb > kw else 2)
        # sample last K plies
        for st in states[-LAST_K:]:
            stm = int(st.current_player)
            def val_from(winner):
                if winner == 2:
                    return 0.0
                return 1.0 if winner == stm else -1.0
            out.append((st, val_from(swept), val_from(raw), empty_ending))
    return out


def net_value(sess, st):
    g = TogyzQumalaq(); g.set_state(st)
    state = g.encode_state().reshape(1, 7, 9).astype(np.float32)
    _, value = sess.run(None, {"state": state})
    return float(np.array(value).flatten()[0])


def evaluate(name, sess, positions):
    n = ne = 0
    sweep_sign_ok = raw_sign_ok = 0
    sweep_sign_ok_e = raw_sign_ok_e = 0
    mae_sweep = 0.0
    for st, vsweep, vraw, empty in positions:
        if vsweep == 0.0:
            continue  # skip true draws for sign accuracy
        pred = net_value(sess, st)
        n += 1
        if (pred > 0) == (vsweep > 0):
            sweep_sign_ok += 1
        if (pred > 0) == (vraw > 0):
            raw_sign_ok += 1
        mae_sweep += abs(pred - vsweep)
        if empty:
            ne += 1
            if (pred > 0) == (vsweep > 0):
                sweep_sign_ok_e += 1
            if (pred > 0) == (vraw > 0):
                raw_sign_ok_e += 1
    print(f"\n[{name}]  n={n} decisive positions ({ne} from empty-side endings)")
    print(f"  value-sign matches TRUE swept outcome : {100*sweep_sign_ok/max(1,n):.1f}%")
    print(f"  value-sign matches OLD raw-kazan       : {100*raw_sign_ok/max(1,n):.1f}%")
    print(f"  mean|pred - swept_target|              : {mae_sweep/max(1,n):.3f}")
    if ne:
        print(f"  -- on EMPTY-SIDE endings (the buggy cases) --")
        print(f"     sign matches swept truth : {100*sweep_sign_ok_e/ne:.1f}%")
        print(f"     sign matches raw-kazan   : {100*raw_sign_ok_e/ne:.1f}%")


def main():
    a, b = sys.argv[1], sys.argv[2]
    print("collecting near-terminal positions from champion games...")
    positions = collect_positions()
    ne = sum(1 for p in positions if p[3])
    print(f"  {len(positions)} positions ({ne} from empty-side endings)")
    so = ort.SessionOptions()
    for name, path in [("A", a), ("B", b)]:
        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        evaluate(f"{name}: {os.path.basename(path)}", sess, positions)


if __name__ == "__main__":
    main()
