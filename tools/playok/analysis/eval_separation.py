#!/usr/bin/env python3
"""How well does an engine's eval at the MIDDLE of a game separate eventual wins from
losses? Scores every recorded PlayOK game that has both a halfway ply to evaluate and a
decisive (non-drawn) sweep-rule outcome, using both eventual wins AND eventual losses --
unlike the June 2026 "blindness" metric, which looked at losses only and so could not
tell whether a bad-looking loss rate meant a bad eval or just a bad sample of losses.
Two categories of recorded game are excluded, and reported (not hidden) by the CLI:
games with fewer than 2 recorded plies (no ply exists at the halfway point -- these are
mostly opponent walk-offs recorded as 1-ply wins, so silently dropping them is NOT
symmetric across classes) and drawn games (no win/loss label to score against). Cheap
screen before an external gate.

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


def midgame_positions(games_dir: Path, stats: dict | None = None):
    """(position at the 50% ply, 1 if White won else 0) per decisive game.

    Uses both eventual wins and eventual losses (see module docstring). Two categories
    of recorded game are excluded because neither yields a usable (midgame position,
    outcome) pair: games with fewer than 2 recorded plies (skipped_short -- no ply
    exists at the halfway point) and drawn games (skipped_draw -- no win/loss label).

    If `stats` is given (a dict), it is filled in with the exclusion accounting:
    "total" (games scanned), "kept" (== len(the returned list)), "skipped_short",
    "skipped_draw" -- so a caller can report what was excluded and why, rather than
    silently dropping it. Passing no `stats` (the default) leaves this function's
    return value exactly the list of rows, as before.
    """
    rows = []
    total = skipped_short = skipped_draw = 0
    for f in sorted(games_dir.glob("game_*.txt")):
        total += 1
        plies = []
        for line in f.read_text().splitlines():
            m = LINE.match(line)
            if m:
                plies.append(m.group(3))
        if len(plies) < 2:
            skipped_short += 1
            continue
        label = white_won(plies[-1])
        if label is None:
            skipped_draw += 1
            continue
        rows.append((plies[len(plies) // 2 - 1], 1 if label else 0))
    if stats is not None:
        stats.update(total=total, kept=len(rows), skipped_short=skipped_short,
                      skipped_draw=skipped_draw)
    return rows


def auc(pos_scores, neg_scores):
    """Probability a random positive outscores a random negative (ties count half)."""
    wins = 0.0
    for p in pos_scores:
        for n in neg_scores:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos_scores) * len(neg_scores))


def eval_white_pov(engine: str, pos: str, ms: int) -> float:
    # Resolve to absolute: engine assets load relative to the binary's own directory
    # regardless of the calling process's cwd, so an absolute exec path is both
    # necessary (a relative one plus a changed cwd resolves against the WRONG
    # directory -- see subprocess's chdir-then-exec order) and sufficient (no cwd
    # argument is needed at all once the path is absolute).
    engine_path = str(Path(engine).resolve())
    out = subprocess.run([engine_path, "analyze", pos, str(ms)], capture_output=True,
                         text=True).stdout
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
    stats = {}
    rows = midgame_positions(GAMES_DIR, stats)
    pos, neg = [], []
    for p, label in rows:
        (pos if label else neg).append(eval_white_pov(a.engine, p, a.ms))
    print(f"games={len(rows)} wins={len(pos)} losses={len(neg)} AUC={auc(pos, neg):.3f} "
          f"| scanned={stats['total']} skipped_short={stats['skipped_short']} "
          f"skipped_draw={stats['skipped_draw']}")


if __name__ == "__main__":
    main()
