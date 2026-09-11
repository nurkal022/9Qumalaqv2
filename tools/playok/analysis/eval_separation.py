#!/usr/bin/env python3
"""How well does an engine's eval at the MIDDLE of a game separate eventual wins from
losses? Uses every recorded PlayOK game (wins AND losses), so unlike the June 2026
"blindness" metric it has no survivorship bias. Cheap screen before an external gate.

Usage: python3.12 tools/playok/analysis/eval_separation.py <engine-binary> [--ms 300]
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

GAMES_DIR = Path(__file__).resolve().parent.parent / "games"
LINE = re.compile(r"^(\d+)\. ([WB])\d+ \[[^\]]*\]\s+(\S+)")


def white_won(final_pos: str):
    """Winner under the sweep rule from the last recorded position; None on a draw."""
    w, b, k, _, _ = final_pos.split("/")
    ws = sum(map(int, w.split(","))) + int(k.split(",")[0])
    bs = sum(map(int, b.split(","))) + int(k.split(",")[1])
    if ws == bs:
        return None
    return ws > bs


def midgame_positions(games_dir: Path):
    """(position at the 50% ply, 1 if White won else 0) per decisive game."""
    rows = []
    for f in sorted(games_dir.glob("game_*.txt")):
        plies = []
        for line in f.read_text().splitlines():
            m = LINE.match(line)
            if m:
                plies.append(m.group(3))
        if len(plies) < 2:
            continue
        label = white_won(plies[-1])
        if label is None:
            continue
        rows.append((plies[len(plies) // 2 - 1], 1 if label else 0))
    return rows


def auc(pos_scores, neg_scores):
    """Probability a random positive outscores a random negative (ties count half)."""
    wins = 0.0
    for p in pos_scores:
        for n in neg_scores:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos_scores) * len(neg_scores))


def eval_white_pov(engine: str, pos: str, ms: int) -> float:
    out = subprocess.run([engine, "analyze", pos, str(ms)], capture_output=True, text=True,
                         cwd=str(Path(engine).parent)).stdout
    j = json.loads(out.strip().splitlines()[-1])
    if j.get("terminal"):
        return {"white_win": 1e6, "black_win": -1e6}.get(j.get("result"), 0.0)
    score = float(j["score"])                      # side-to-move POV
    return score if pos.endswith("/0") else -score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine")
    ap.add_argument("--ms", type=int, default=300)
    a = ap.parse_args()
    rows = midgame_positions(GAMES_DIR)
    pos, neg = [], []
    for p, label in rows:
        (pos if label else neg).append(eval_white_pov(a.engine, p, a.ms))
    print(f"games={len(rows)} wins={len(pos)} losses={len(neg)} AUC={auc(pos, neg):.3f}")


if __name__ == "__main__":
    main()
