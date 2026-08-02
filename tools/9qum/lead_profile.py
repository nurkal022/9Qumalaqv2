#!/usr/bin/env python3
"""Material-lead profile: how big a lead does each engine build, and does it hold it?

Reads saved game records from either match harness --

  * tools/ab_match.py's `--save DIR` output (one `ab_match_<ts>.jsonl` per run, one
    JSON record per finished game, schema "ab_match_game_v1"), or
  * tools/9qum/match.py's `data/9qum/matches/*.jsonl` (one record per game against
    the external 9qum opponent, our_seat/record/opening fields)

-- and reports, per engine and split by result (W/D/L), the median material lead
(kazan[engine] - kazan[opponent]) at the 25%, 50%, 75% and final points of the game,
plus the median PEAK lead reached at any point. This is the analysis that showed the
production engine builds a +17 lead and throws it away late while a candidate never
built a lead at all (see docs/MEASUREMENT_PROTOCOL.md rule 5) -- previously that was
only possible for 9qum games, because tools/ab_match.py stored no move list at all.

*** CAVEAT: the lead reported here is the RAW kazan difference at that point in the
*** game. It is NOT the outcome. Togyzkumalak decides the winner only after the
*** end-of-game sweep (each side's remaining board stones go to its OWN kazan) -- a
*** position with a big raw lead can still lose once the sweep is applied. This
*** caveat was missed once already and produced a wrong reading; every run of this
*** script restates it (see the banner it prints, and MEASUREMENT_PROTOCOL.md rule 5).

Positions are reconstructed by replaying each game's recorded move list through
tools/playok/engine.py's `Engine.apply_move` (the same pure-Python board transition
tools/9qum/match.py and the PlayOK bridge already rely on) -- a SECOND, independent
implementation from whichever referee actually produced the record. If a move fails
to replay at all (bad index, corrupt record), that game is skipped with a warning.

NOTE on why a replayed FINAL kazan can legitimately differ from a harness's own
recorded final kazan for a naturally-terminal game: `apply_move` is a pure per-move
transition and deliberately does NOT perform the end-of-game sweep (moving the
still-non-empty side's remaining board stones into its own kazan) -- that sweep is
an end-of-game accounting step, not a move, and different harnesses record it
differently (9qum's server mutates its own kazan for real; our own alphazero-code
referee only computes the swept total transiently to decide the winner, never
writing it back into the state ab_match.py records). So the RAW "final" lead this
script reports is deliberately the pre-sweep board arithmetic, consistent with the
25/50/75% checkpoints -- exactly the number that can mislead you about who won,
which is the whole reason for the caveat below. This is not treated as a
mismatch/error.

Usage:
  python3 tools/9qum/lead_profile.py                        # default: data/9qum/matches
  python3 tools/9qum/lead_profile.py data/9qum/matches
  python3 tools/9qum/lead_profile.py /path/to/ab_match_saves
  python3 tools/9qum/lead_profile.py file1.jsonl file2.jsonl dir/
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "playok"))
from engine import Engine as PlayokEngine, START_POSITION, _parse_pos  # noqa: E402

DEFAULT_PATHS = [str(REPO / "data" / "9qum" / "matches")]
CHECKPOINTS = (0.25, 0.5, 0.75, 1.0)


def _engine_label(path, weights_sha256):
    base = path or "unknown-engine"
    if weights_sha256:
        return f"{base}  [nnue {weights_sha256[:10]}]"
    return base


# --------------------------------------------------------------------------
# Per-schema adapters: normalize a raw JSON record into 0+ "game views" -- one per
# engine that took part, from THAT engine's own perspective (side, result, moves).
# --------------------------------------------------------------------------
def views_from_ab_match_record(rec, source):
    """tools/ab_match.py's --save schema (schema=='ab_match_game_v1'): one game, two
    engines (A and B) -- yields one view per engine, from its own side/result."""
    moves = rec.get("moves")
    white = rec.get("white")
    result_a = rec.get("result_a")
    if moves is None or white not in ("A", "B") or result_a not in ("W", "D", "L"):
        return []
    result_b = {"W": "L", "L": "W", "D": "D"}[result_a]
    engine_a = rec.get("engine_a") or {}
    engine_b = rec.get("engine_b") or {}
    label_a = _engine_label(engine_a.get("path"), engine_a.get("weights_sha256"))
    label_b = _engine_label(engine_b.get("path"), engine_b.get("weights_sha256"))
    side_a = 0 if white == "A" else 1
    side_b = 1 - side_a
    recorded_kazan = rec.get("kazan")
    return [
        {"engine": label_a, "side": side_a, "result": result_a, "moves": moves,
         "source": source, "recorded_kazan": recorded_kazan},
        {"engine": label_b, "side": side_b, "result": result_b, "moves": moves,
         "source": source, "recorded_kazan": recorded_kazan},
    ]


def views_from_9qum_record(rec, source):
    """tools/9qum/match.py's schema: one game, our engine only (the opponent is the
    external 9qum net). VOID games are never counted -- they never had a result.

    `record` holds the post-board-creation moves as 1-9 human labels (space
    separated); when the game started from an N-ply opening line, `opening` holds
    that N-ply prefix in the SAME 1-9, comma-separated encoding (see
    tools/9qum/match.py's opening_suite/play_game) -- concatenating the two gives the
    true full move list from the standard start position.
    """
    result = rec.get("result")
    our_seat = rec.get("our_seat")
    if result not in ("W", "D", "L") or our_seat not in (0, 1):
        return []
    opening = rec.get("opening") or ""
    record_str = rec.get("record") or ""
    pre_moves = [int(x) - 1 for x in opening.split(",")] if opening else []
    post_moves = [int(x) - 1 for x in record_str.split()] if record_str else []
    moves = pre_moves + post_moves
    label = _engine_label(rec.get("engine_path"), rec.get("engine_weights_sha256"))
    return [{"engine": label, "side": our_seat, "result": result, "moves": moves,
             "source": source, "recorded_kazan": rec.get("kazan")}]


def load_views(paths):
    views, skipped_records = [], 0
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            files = sorted(pp.glob("*.jsonl"))
            if not files:
                print(f"WARNING: no .jsonl files in directory {pp}", file=sys.stderr)
        elif pp.is_file():
            files = [pp]
        else:
            print(f"WARNING: path not found, skipping: {p}", file=sys.stderr)
            continue
        for f in files:
            with open(f, encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    source = f"{f}:{lineno}"
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError as exc:
                        print(f"WARNING: bad JSON at {source} ({exc}), skipping", file=sys.stderr)
                        skipped_records += 1
                        continue
                    if rec.get("schema", "").startswith("ab_match_game") or \
                            ("engine_a" in rec and "engine_b" in rec):
                        views.extend(views_from_ab_match_record(rec, source))
                    elif "our_seat" in rec:
                        views.extend(views_from_9qum_record(rec, source))
                    else:
                        print(f"WARNING: unrecognized record schema at {source}, skipping", file=sys.stderr)
                        skipped_records += 1
    return views, skipped_records


# --------------------------------------------------------------------------
# Replay: the pure-Python board transition (tools/playok/engine.py), independent of
# whichever referee produced the record.
# --------------------------------------------------------------------------
def replay(moves, our_side):
    """Replay `moves` (0-8 wire indices) from the standard start position. Returns
    (leads, final_kazan) where leads[k] = kazan[our_side] - kazan[opp_side] after k
    of the moves have been applied (leads[0] == 0, the start position)."""
    pos = START_POSITION
    opp_side = 1 - our_side
    leads = [0]
    for mv in moves:
        pos = PlayokEngine.apply_move(pos, mv)
        _, _, kaz, _, _ = _parse_pos(pos)
        leads.append(kaz[our_side] - kaz[opp_side])
    _, _, final_kazan, _, _ = _parse_pos(pos)
    return leads, final_kazan


def checkpoint_leads(leads):
    """leads -> {p25, p50, p75, final, peak, plies} using the actual ply count (not
    max_plies) as the 100% mark. Returns None for an empty (0-ply) game."""
    n = len(leads) - 1
    if n <= 0:
        return None
    idx = {pct: min(n, max(0, round(pct * n))) for pct in CHECKPOINTS}
    return {
        "p25": leads[idx[0.25]], "p50": leads[idx[0.5]], "p75": leads[idx[0.75]],
        "final": leads[idx[1.0]], "peak": max(leads), "plies": n,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", default=DEFAULT_PATHS,
                    help="jsonl files and/or directories of jsonl files (default: data/9qum/matches)")
    args = ap.parse_args()

    views, skipped = load_views(args.paths)

    stats = defaultdict(lambda: defaultdict(list))
    for v in views:
        try:
            leads, _replayed_final_kazan = replay(v["moves"], v["side"])
        except Exception as exc:
            # a move that doesn't replay at all (bad index, corrupt record) is a real
            # data problem -- unlike the expected raw-vs-swept final-kazan divergence
            # documented above, this is skipped.
            print(f"WARNING: replay failed for {v['source']} ({v['engine']}): {exc} -- skipping",
                  file=sys.stderr)
            skipped += 1
            continue
        cps = checkpoint_leads(leads)
        if cps is None:
            skipped += 1
            continue
        stats[v["engine"]][v["result"]].append(cps)

    print("=" * 78)
    print("MATERIAL-LEAD PROFILE")
    print("=" * 78)
    print("CAVEAT: every number below is the RAW kazan[engine] - kazan[opponent]")
    print("difference at that point in the game -- it is NOT the outcome. Togyzkumalak")
    print("decides the winner only after the end-of-game sweep (each side's remaining")
    print("board stones go to its OWN kazan); a position with a big raw lead can still")
    print("be a loss once the sweep is applied. Read 'lead' as 'how the score looked")
    print("while the game was still in progress', never as a claim about who was ahead.")
    print("(See docs/MEASUREMENT_PROTOCOL.md rule 5 -- this exact caveat was missed once")
    print("and produced a wrong reading.)")
    print()

    if not stats:
        print("No countable game records found.")
        if skipped:
            print(f"({skipped} record(s)/game-view(s) skipped -- see warnings above)")
        return

    header = f"{'result':<6}{'n':>5}   {'lead@25%':>9} {'lead@50%':>9} {'lead@75%':>9} {'lead@final':>11} {'peak lead':>10}"
    for engine in sorted(stats):
        print(f"-- {engine} --")
        print(header)
        for result in ("W", "D", "L"):
            games = stats[engine].get(result, [])
            if not games:
                continue
            n = len(games)
            med = lambda key: statistics.median(g[key] for g in games)  # noqa: E731
            print(f"{result:<6}{n:>5}   {med('p25'):>9.1f} {med('p50'):>9.1f} {med('p75'):>9.1f} "
                  f"{med('final'):>11.1f} {med('peak'):>10.1f}")
        print()

    if skipped:
        print(f"({skipped} record(s)/game-view(s) skipped -- unparseable, VOID/unrecognized, "
              f"0-ply, or a move that failed to replay; see warnings above)")


if __name__ == "__main__":
    main()
