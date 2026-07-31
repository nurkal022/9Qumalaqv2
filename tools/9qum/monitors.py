#!/usr/bin/env python3
"""Cheap offline monitors for an engine's evaluation and move choice.

These are the fast loop: minutes per candidate, no calls to 9qum's API, so iteration is not
bounded by their rate limit. The gate (match.py) is the slow, authoritative loop.

Run: python3.12 tools/9qum/monitors.py --engine models/engine/baseline --ms 100
"""
import argparse
import gzip
import json
import math
import os
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "playok"))
from engine import Engine  # noqa: E402


class Ev(Engine):
    def score_and_move(self, pos, ms):
        with self._lock:
            self._cmd(f"go time {ms} pos {pos}")
            while True:
                line = self._proc.stdout.readline()
                if not line:
                    return None, None
                line = line.strip()
                if line.startswith("bestmove"):
                    t = line.split()
                    sc = int(t[t.index("score") + 1]) if "score" in t else None
                    return sc, int(t[1])
                if line.startswith(("terminal", "error")):
                    return None, None


def pos_of(st):
    tz = st.get("tuzdyk") or [None, None]
    return (",".join(map(str, st["pits"][0:9])) + "/" + ",".join(map(str, st["pits"][9:18])) +
            f"/{st['kazan'][0]},{st['kazan'][1]}/"
            f"{-1 if tz[0] is None else tz[0] - 9},{-1 if tz[1] is None else tz[1]}/{st['to_move']}")


def load_val_positions(corpus, sample_per_bucket, seed=5):
    with open(os.path.join(corpus, "train", "split.json"), encoding="utf-8") as f:
        val_games = set(json.load(f)["val_games"])
    buckets = {"mid": [], "close": [], "clear": [], "policy": []}
    with gzip.open(os.path.join(corpus, "games", "replays.jsonl.gz"), "rt", encoding="utf-8") as f:
        for line in f:
            g = json.loads(line)
            if g["game_id"] not in val_games or g.get("winner") not in (0, 1):
                continue
            meta = g.get("_meta") or {}
            r0 = meta.get("r0_before") or 0
            r1 = meta.get("r1_before") or 0
            moves = (g.get("states") or [{}])[0].get("moves") or []
            gid = g["game_id"]
            for ply, st in enumerate(g["states"]):
                y = 1.0 if g["winner"] == st["to_move"] else 0.0
                dk = st["kazan"][0] - st["kazan"][1]
                if 40 <= ply < 80:
                    buckets["mid"].append((st, y, gid, ply))
                elif ply >= 80 and abs(dk) <= 8:
                    buckets["close"].append((st, y, gid, ply))
                elif ply >= 80 and abs(dk) >= 20:
                    buckets["clear"].append((st, y, gid, ply))
                if ply < len(moves) and min(r0, r1) >= 2000:
                    buckets["policy"].append((st, moves[ply]["hole"]))
    rng = random.Random(seed)
    for k in buckets:
        if len(buckets[k]) > sample_per_bucket:
            buckets[k] = rng.sample(buckets[k], sample_per_bucket)
    return buckets


def load_curves(corpus):
    """game_id -> {ply: win} where win is seat 0's win probability, as a percent (0-100).

    NOT the side-to-move's perspective -- read as seat 0 it agrees with recorded outcomes
    ~82% of the time; read as the mover it collapses to ~49.9% (see task-5 fix-round-1
    finding). Callers must convert with the position's own to_move before comparing.
    """
    curves = {}
    path = os.path.join(corpus, "analysis", "curves.jsonl.gz")
    if not os.path.exists(path):
        return curves
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            g = json.loads(line)
            curves[g["game_id"]] = {p["ply"]: p["win"] for p in (g.get("points") or [])}
    return curves


def acc_brier(pairs):
    """Shared scoring code so our engine and the 9qum reference are graded identically."""
    if not pairs:
        return None, None
    acc = 100 * sum(1 for p, y in pairs if (p > 0.5) == (y > 0.5)) / len(pairs)
    brier = sum((p - y) ** 2 for p, y in pairs) / len(pairs)
    return acc, brier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=str(REPO / "models" / "engine" / "baseline"))
    ap.add_argument("--ms", type=int, default=100)
    ap.add_argument("--corpus", default="data/9qum")
    ap.add_argument("--sample", type=int, default=900)
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    buckets = load_val_positions(a.corpus, a.sample)
    curves = load_curves(a.corpus)
    e = Ev(Path(a.engine))
    e.start()
    out = {"engine": a.engine, "ms": a.ms, "label": a.label}
    for name in ("mid", "close", "clear"):
        pairs = []
        pairs_9qum = []
        sampled = buckets[name]
        for st, y, gid, ply in sampled:
            sc, _ = e.score_and_move(pos_of(st), a.ms)
            if sc is not None:
                sc = max(-2000, min(2000, sc))      # mate/EGTB scores would swamp the scale
                pairs.append((1 / (1 + math.exp(-sc / 350)), y))
            win_seat0 = (curves.get(gid) or {}).get(ply)   # 9qum's label: seat 0's win %, not the mover's
            if win_seat0 is not None:
                p9 = win_seat0 / 100 if st["to_move"] == 0 else 1 - win_seat0 / 100
                pairs_9qum.append((p9, y))
        acc, brier = acc_brier(pairs)
        acc9, brier9 = acc_brier(pairs_9qum)
        coverage = 100 * len(pairs_9qum) / len(sampled) if sampled else 0.0
        out[f"{name}_acc"] = round(acc, 1)
        out[f"{name}_brier"] = round(brier, 4)
        out[f"{name}_acc_9qum"] = round(acc9, 1) if acc9 is not None else None
        out[f"{name}_brier_9qum"] = round(brier9, 4) if brier9 is not None else None
        out[f"{name}_9qum_n"] = len(pairs_9qum)
        out[f"{name}_9qum_coverage"] = round(coverage, 1)
        ref_str = (f"acc {acc9:5.1f}%  brier {brier9:.4f}" if pairs_9qum else "no labels")
        print(f"{name:>6}: n={len(pairs):>5}  acc {acc:5.1f}%  brier {brier:.4f}   "
              f"| 9qum: n={len(pairs_9qum):>5} ({coverage:4.1f}% coverage)  {ref_str}")
    hits = tot = 0
    for st, played in buckets["policy"]:
        _, mv = e.score_and_move(pos_of(st), a.ms)
        if mv is None:
            continue
        tot += 1
        hits += (mv == played)
    e.stop()
    out["policy_match"] = round(100 * hits / max(1, tot), 1)
    print(f"policy match-rate vs >=2000 humans: {out['policy_match']}% of {tot}")

    print("\ncomparison (identical sampled positions, our engine vs the 9qum reference):")
    for name in ("mid", "close", "clear"):
        a_ours = out[f"{name}_acc"]
        a_9 = out[f"{name}_acc_9qum"]
        if a_9 is None:
            print(f"  {name:>6}: ours {a_ours:5.1f}%   9qum   n/a (no labels)")
        else:
            gap = round(a_ours - a_9, 1)
            print(f"  {name:>6}: ours {a_ours:5.1f}%   9qum {a_9:5.1f}%   gap {gap:+.1f}   "
                  f"(coverage {out[f'{name}_9qum_coverage']:.1f}% of {len(buckets[name])})")

    with open(os.path.join(a.corpus, "train", "monitors.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
