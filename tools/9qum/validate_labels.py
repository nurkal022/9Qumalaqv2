#!/usr/bin/env python3
"""Are 9qum's win% labels actually worth training on?

Ground truth is the game result recorded by their referee (which our board code already
reproduces ply-for-ply). For every labelled ply we ask:

  1. PERSPECTIVE   whose win probability is it? (seat0's, or the side to move's)
  2. CALIBRATION   do positions labelled p% actually end that way p% of the time?
  3. SKILL         Brier / log-loss vs three references:
                     - constant 0.5                (no information)
                     - kazan difference -> logistic (material only)
                     - OUR engine's search score   (what we already have)
  4. PHASE         the same numbers split by game phase — the endgame is where we lose
  5. NEWS          correlation with our own eval: highly correlated labels teach nothing

Only games that were played out (`по камням` / `сдача`) count: on a flag-fall or an
abandon the recorded winner says nothing about the position.

Usage:
  python3 tools/9qum/validate_labels.py                      # their eval vs material
  python3 tools/9qum/validate_labels.py --engine-sample 3000 # also run our engine
"""
import argparse
import gzip
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PLAYED_OUT = ("по камням", "сдача")


def jsonl_read(path):
    if not os.path.exists(path):
        return
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except ValueError:
                    continue


def brier(pairs):
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs)


def logloss(pairs, eps=1e-6):
    return -sum(y * math.log(max(p, eps)) + (1 - y) * math.log(max(1 - p, eps))
                for p, y in pairs) / len(pairs)


def accuracy(pairs):
    return sum(1 for p, y in pairs if (p > 0.5) == (y > 0.5)) / len(pairs)


def calibration(pairs, bins=10):
    rows = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(p, y) for p, y in pairs if (lo <= p < hi or (b == bins - 1 and p == 1.0))]
        if sel:
            rows.append((lo, hi, len(sel), sum(p for p, _ in sel) / len(sel),
                         sum(y for _, y in sel) / len(sel)))
    return rows


def clamp_score(sc, cap=2000):
    """Our engine reports mate/EGTB scores up to +-90000; left raw they blow up the
    standardisation in the logistic fit and flatten every normal score to zero."""
    return max(-cap, min(cap, sc))


def fit_logistic(xs, ys, iters=400, lr=0.05):
    """1-D logistic regression by plain gradient descent: feature -> win probability."""
    a, b = 0.1, 0.0
    n = len(xs)
    m = sum(xs) / n
    sd = (sum((x - m) ** 2 for x in xs) / n) ** 0.5 or 1.0
    for _ in range(iters):
        ga = gb = 0.0
        for x, y in zip(xs, ys):
            z = a * (x - m) / sd + b
            p = 1 / (1 + math.exp(-max(-30, min(30, z))))
            ga += (p - y) * (x - m) / sd
            gb += (p - y)
        a -= lr * ga / n
        b -= lr * gb / n
    return lambda x: 1 / (1 + math.exp(-max(-30, min(30, a * (x - m) / sd + b))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/9qum")
    ap.add_argument("--engine-sample", type=int, default=0,
                    help="also evaluate N sampled positions with our engine (0 = skip)")
    ap.add_argument("--engine", default=str(REPO / "models" / "engine" / "baseline"))
    ap.add_argument("--engine-ms", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    random.seed(a.seed)

    curves = {c["game_id"]: c for c in jsonl_read(os.path.join(a.out, "analysis", "curves.jsonl.gz"))}
    print(f"curves loaded: {len(curves)} games")

    # ---- join curves with the games they describe
    samples = []      # (game_id, ply, win_pred, winner_seat0, to_move, kazan_diff_seat0, pits, kazan, tuz)
    games_used = 0
    for g in jsonl_read(os.path.join(a.out, "games", "replays.jsonl.gz")):
        c = curves.get(g["game_id"])
        if not c or g.get("reason") not in PLAYED_OUT:
            continue
        w = g.get("winner")
        if w not in (0, 1):          # draws carry no 0/1 target
            continue
        states = g.get("states") or []
        pts = {p["ply"]: p["win"] for p in c.get("points") or []}
        if not pts or not states:
            continue
        games_used += 1
        for ply, st in enumerate(states):
            if ply not in pts:
                continue
            samples.append({
                "gid": g["game_id"], "ply": ply, "pred": pts[ply] / 100.0,
                "y0": 1.0 if w == 0 else 0.0, "to_move": st["to_move"],
                "kdiff": st["kazan"][0] - st["kazan"][1], "st": st,
                "nply": len(states), "reason": g.get("reason"),
                "finished_at": (g.get("_meta") or {}).get("finished_at") or 0,
            })
    print(f"joined: {games_used} played-out decisive games, {len(samples):,} labelled plies\n")
    if not samples:
        sys.exit("nothing to validate")

    # ---- 1. perspective
    A = [(s["pred"], s["y0"]) for s in samples]                                    # seat0's win prob
    B = [(s["pred"], 1.0 if (s["y0"] == 1.0) == (s["to_move"] == 0) else 0.0) for s in samples]  # mover's
    print("1. PERSPECTIVE (which reading of `win` fits the outcomes?)")
    print(f"   as seat0's win probability : acc {100 * accuracy(A):.1f}%  brier {brier(A):.4f}")
    print(f"   as side-to-move's          : acc {100 * accuracy(B):.1f}%  brier {brier(B):.4f}")
    pairs, label = (A, "seat0") if brier(A) <= brier(B) else (B, "side-to-move")
    print(f"   -> using the {label} reading\n")

    # ---- 2/3. skill vs references
    # split by GAME, not by ply: plies inside one game are heavily autocorrelated, so a
    # per-ply split would leak the outcome into the fitted baseline.
    gids = sorted({s["gid"] for s in samples})
    random.shuffle(gids)
    train_g = set(gids[:len(gids) // 2])
    train = [i for i, s in enumerate(samples) if s["gid"] in train_g]
    test = [i for i, s in enumerate(samples) if s["gid"] not in train_g]
    print(f"   (split by game: {len(train_g)} train / {len(gids) - len(train_g)} test games)\n")
    kaz_fit = fit_logistic([samples[i]["kdiff"] for i in train], [pairs[i][1] for i in train])
    ref_const = [(0.5, pairs[i][1]) for i in test]
    ref_kaz = [(kaz_fit(samples[i]["kdiff"]), pairs[i][1]) for i in test]
    theirs = [pairs[i] for i in test]

    print("2. SKILL on a held-out half (lower brier/logloss = better)")
    hdr = f"   {'predictor':<26}{'brier':>9}{'logloss':>10}{'acc':>8}"
    print(hdr)
    print(f"   {'constant 0.5':<26}{brier(ref_const):>9.4f}{logloss(ref_const):>10.4f}{100 * accuracy(ref_const):>7.1f}%")
    print(f"   {'kazan diff (logistic)':<26}{brier(ref_kaz):>9.4f}{logloss(ref_kaz):>10.4f}{100 * accuracy(ref_kaz):>7.1f}%")
    print(f"   {'9qum net win%':<26}{brier(theirs):>9.4f}{logloss(theirs):>10.4f}{100 * accuracy(theirs):>7.1f}%")

    # ---- our engine on a sample
    eng_pairs = None
    if a.engine_sample:
        sys.path.insert(0, str(REPO / "tools" / "playok"))
        from engine import Engine  # noqa: E402

        class EvalEngine(Engine):
            """Same serve protocol, but we need the search score, not just the move."""

            def score(self, pos, time_ms=100):
                with self._lock:
                    self._cmd(f"go time {time_ms} pos {pos}")
                    while True:
                        line = self._proc.stdout.readline()
                        if not line:
                            return None
                        line = line.strip()
                        if line.startswith("bestmove"):
                            t = line.split()
                            return int(t[t.index("score") + 1]) if "score" in t else None
                        if line.startswith(("terminal", "error")):
                            return None

        e = EvalEngine(Path(a.engine))
        e.start()
        sub = random.sample(test, min(a.engine_sample, len(test)))
        xs, ys, scored = [], [], []
        for k, i in enumerate(sub):
            s = samples[i]
            st = s["st"]
            tz = st.get("tuzdyk") or [None, None]
            pos = (",".join(map(str, st["pits"][0:9])) + "/" + ",".join(map(str, st["pits"][9:18])) +
                   f"/{st['kazan'][0]},{st['kazan'][1]}/"
                   f"{-1 if tz[0] is None else tz[0] - 9},{-1 if tz[1] is None else tz[1]}/{st['to_move']}")
            sc = e.score(pos, time_ms=a.engine_ms)
            if sc is None:
                continue
            # engine score is relative to the side to move (verified: 72% vs 49% sign accuracy)
            sc0 = clamp_score(sc if st["to_move"] == 0 else -sc)
            scored.append((i, sc0))
            if (k + 1) % 500 == 0:
                print(f"     our engine: {k + 1}/{len(sub)} positions", flush=True)
        e.stop()
        # fit on one part of the sample, score the rest, so our engine gets the same treatment
        cut = len(scored) // 2
        xs = [s for _, s in scored[:cut]]
        ys = [pairs[i][1] for i, _ in scored[:cut]]
        our_fit = fit_logistic(xs, ys)
        eng_pairs = [(our_fit(s), pairs[i][1]) for i, s in scored[cut:]]
        theirs_same = [pairs[i] for i, _ in scored[cut:]]
        print(f"   {'our engine score (logistic)':<26}{brier(eng_pairs):>9.4f}{logloss(eng_pairs):>10.4f}"
              f"{100 * accuracy(eng_pairs):>7.1f}%   (on {len(eng_pairs)} shared positions)")
        print(f"   {'9qum net, same positions':<26}{brier(theirs_same):>9.4f}{logloss(theirs_same):>10.4f}"
              f"{100 * accuracy(theirs_same):>7.1f}%")
        # 5. how much new information?
        ours = [p for p, _ in eng_pairs]
        thrs = [p for p, _ in theirs_same]
        mo, mt = sum(ours) / len(ours), sum(thrs) / len(thrs)
        cov = sum((x - mo) * (y - mt) for x, y in zip(ours, thrs))
        so = math.sqrt(sum((x - mo) ** 2 for x in ours)) or 1
        st_ = math.sqrt(sum((y - mt) ** 2 for y in thrs)) or 1
        print(f"\n5. NEW INFORMATION: correlation(our eval, their eval) = {cov / (so * st_):.3f}")

    # ---- 4. calibration + phase split
    print("\n3. CALIBRATION of their win% (held-out half)")
    print(f"   {'bin':<12}{'n':>8}{'predicted':>12}{'actual':>10}")
    for lo, hi, n, pm, ym in calibration(theirs):
        print(f"   {f'{lo:.1f}-{hi:.1f}':<12}{n:>8}{100 * pm:>11.1f}%{100 * ym:>9.1f}%")

    print("\n4. BY GAME PHASE (their win% vs kazan-diff baseline)")
    phases = [("opening  ply<40", lambda s: s["ply"] < 40),
              ("middle   40-80", lambda s: 40 <= s["ply"] < 80),
              ("late     80-120", lambda s: 80 <= s["ply"] < 120),
              ("endgame  120+", lambda s: s["ply"] >= 120),
              ("close endgame (|kazan diff|<=8, ply>=80)",
               lambda s: s["ply"] >= 80 and abs(s["kdiff"]) <= 8)]
    for name, sel in phases:
        ii = [i for i in test if sel(samples[i])]
        if len(ii) < 50:
            continue
        t = [pairs[i] for i in ii]
        k = [(kaz_fit(samples[i]["kdiff"]), pairs[i][1]) for i in ii]
        print(f"   {name:<42} n={len(ii):>6}  their brier {brier(t):.4f} acc {100 * accuracy(t):.1f}%"
              f"   |  kazan brier {brier(k):.4f} acc {100 * accuracy(k):.1f}%")

    # ---- 7. memorisation check: their net trained on ~329k human games, so positions from
    # games it may have seen could look unfairly easy. Their current run started 2026-07-17
    # (unix 1784257133); games finished after that cannot be in the v101 training corpus.
    CUT = 1784257133
    seen = [pairs[i] for i in test if 0 < samples[i]["finished_at"] < CUT]
    unseen = [pairs[i] for i in test if samples[i]["finished_at"] >= CUT]
    print("\n7. MEMORISATION CHECK (could they just remember these games?)")
    for name, sub in (("games older than their training run", seen),
                      ("games played after it started", unseen)):
        if len(sub) > 500:
            print(f"   {name:<40} n={len(sub):>6}  brier {brier(sub):.4f}  acc {100 * accuracy(sub):.1f}%")
        else:
            print(f"   {name:<40} n={len(sub):>6}  (too few to judge)")

    # ---- sanity: does the curve agree with the result at the very end?
    last = defaultdict(lambda: (-1, None))
    for i, s in enumerate(samples):
        if s["ply"] > last[s["gid"]][0]:
            last[s["gid"]] = (s["ply"], i)
    for reason in PLAYED_OUT:
        fin = [pairs[i] for _, (_, i) in last.items() if i is not None and samples[i]["reason"] == reason]
        if fin:
            print(f"\n6. FINAL-PLY SANITY [{reason}]: their eval names the winner in "
                  f"{100 * accuracy(fin):.1f}% of {len(fin)} games")


if __name__ == "__main__":
    main()
