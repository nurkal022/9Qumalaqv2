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
            for ply, st in enumerate(g["states"]):
                y = 1.0 if g["winner"] == st["to_move"] else 0.0
                dk = st["kazan"][0] - st["kazan"][1]
                if 40 <= ply < 80:
                    buckets["mid"].append((st, y))
                elif ply >= 80 and abs(dk) <= 8:
                    buckets["close"].append((st, y))
                elif ply >= 80 and abs(dk) >= 20:
                    buckets["clear"].append((st, y))
                if ply < len(moves) and min(r0, r1) >= 2000:
                    buckets["policy"].append((st, moves[ply]["hole"]))
    rng = random.Random(seed)
    for k in buckets:
        if len(buckets[k]) > sample_per_bucket:
            buckets[k] = rng.sample(buckets[k], sample_per_bucket)
    return buckets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=str(REPO / "models" / "engine" / "baseline"))
    ap.add_argument("--ms", type=int, default=100)
    ap.add_argument("--corpus", default="data/9qum")
    ap.add_argument("--sample", type=int, default=900)
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    buckets = load_val_positions(a.corpus, a.sample)
    e = Ev(Path(a.engine))
    e.start()
    out = {"engine": a.engine, "ms": a.ms, "label": a.label}
    for name in ("mid", "close", "clear"):
        pairs = []
        for st, y in buckets[name]:
            sc, _ = e.score_and_move(pos_of(st), a.ms)
            if sc is None:
                continue
            sc = max(-2000, min(2000, sc))          # mate/EGTB scores would swamp the scale
            pairs.append((1 / (1 + math.exp(-sc / 350)), y))
        acc = 100 * sum(1 for p, y in pairs if (p > 0.5) == (y > 0.5)) / len(pairs)
        brier = sum((p - y) ** 2 for p, y in pairs) / len(pairs)
        out[f"{name}_acc"] = round(acc, 1)
        out[f"{name}_brier"] = round(brier, 4)
        print(f"{name:>6}: n={len(pairs):>5}  acc {acc:5.1f}%  brier {brier:.4f}")
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
    with open(os.path.join(a.corpus, "train", "monitors.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
