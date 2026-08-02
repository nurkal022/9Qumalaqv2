#!/usr/bin/env python3
"""The Python feature builder must agree with the Rust one feature-for-feature.

Two implementations of one layout is how silent training/inference skew happens: the net
learns on Python features and plays on Rust features. This test compares both on real
positions from the harvested corpus.

Run: python3.12 research/data/test_features_v2.py
"""
import gzip
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
import features_v2 as fv

ENGINE = "target/release/togyzkumalaq-engine"
REPLAYS = "data/9qum/games/replays.jsonl.gz"


def rust_features(pos: str):
    out = subprocess.run([ENGINE, "features", pos], capture_output=True, text=True, check=True)
    parts = out.stdout.strip().split()
    bucket = int(parts[parts.index("bucket") + 1])
    idx = [int(x) for x in parts[: parts.index("bucket")]]
    return sorted(idx), bucket


def sample_states(n):
    states = []
    with gzip.open(REPLAYS, "rt", encoding="utf-8") as f:
        for line in f:
            g = json.loads(line)
            for st in (g.get("states") or [])[::9]:
                states.append(st)
                if len(states) >= n:
                    return states
    return states


def test_start_position():
    idx, bucket = rust_features("9,9,9,9,9,9,9,9,9/9,9,9,9,9,9,9,9,9/0,0/-1,-1/0")
    mine = sorted(fv.build_features([9] * 18, [0, 0], [None, None], 0))
    assert mine == idx, f"start position differs: {mine} vs {idx}"
    assert bucket == fv.phase_bucket([9] * 18) == 0
    assert len(mine) == 23


def test_matches_rust_on_real_positions():
    states = sample_states(200)
    assert len(states) >= 200, "need the harvested corpus in data/9qum"
    for st in states:
        pos = fv.pos_string(st["pits"], st["kazan"], st["tuzdyk"], st["to_move"])
        idx, bucket = rust_features(pos)
        mine = sorted(fv.build_features(st["pits"], st["kazan"], st["tuzdyk"], st["to_move"]))
        assert mine == idx, f"mismatch at {pos}:\n python {mine}\n rust   {idx}"
        assert bucket == fv.phase_bucket(st["pits"]), f"bucket mismatch at {pos}"


def test_validate_position_accepts_legal_and_rejects_invalid():
    """The validator is the fix for a real incident: a hand-typed position summing to
    105 (not 162) was used as evidence for a scale-mismatch diagnosis and had to be
    retracted (see .superpowers/sdd/2026-07-31-beat-9qum-phase-a/progress.md:46). One
    legal position must pass; each of three distinct invariant violations must be
    rejected with a message naming the actual problem."""
    # A real, physically-reachable position: 44 stones still in pits, 118 in the two
    # kazans, one tuzdyk per side -- 44 + 118 == 162.
    legal_pits = [3, 0, 5, 2, 0, 4, 1, 6, 0, 0, 7, 2, 0, 3, 1, 0, 4, 6]
    assert sum(legal_pits) == 44
    legal_kazan = [67, 51]
    legal_tuzdyk = [12, 3]     # white's tuzdyk on black's row (9..17); black's on white's (0..8)
    legal_to_move = 1

    fv.validate_position(legal_pits, legal_kazan, legal_tuzdyk, legal_to_move)  # must not raise

    # 1) wrong stone total: bump one pit by 1 -> sums to 163, not 162.
    bad_total = list(legal_pits)
    bad_total[0] += 1
    try:
        fv.validate_position(bad_total, legal_kazan, legal_tuzdyk, legal_to_move)
    except fv.InvalidPositionError as exc:
        assert "163" in str(exc) and "162" in str(exc), f"message doesn't name the totals: {exc}"
    else:
        raise AssertionError("expected InvalidPositionError for a wrong stone total")

    # 2) negative count.
    bad_negative = list(legal_pits)
    bad_negative[4] = -1
    try:
        fv.validate_position(bad_negative, legal_kazan, legal_tuzdyk, legal_to_move)
    except fv.InvalidPositionError as exc:
        assert "negative" in str(exc) and "pit 4" in str(exc), f"message doesn't name the negative pit: {exc}"
    else:
        raise AssertionError("expected InvalidPositionError for a negative pit count")

    # 3) tuzdyk on the wrong side: 3 is a legal pit for tuzdyk[1] (0..8) but not
    #    tuzdyk[0] (9..17) -- exactly the "physically impossible" shape of mistake.
    bad_tuzdyk = [3, None]
    try:
        fv.validate_position(legal_pits, legal_kazan, bad_tuzdyk, legal_to_move)
    except fv.InvalidPositionError as exc:
        assert "tuzdyk[0]" in str(exc) and "OTHER side" in str(exc), f"message unclear: {exc}"
    else:
        raise AssertionError("expected InvalidPositionError for a tuzdyk on the wrong side")

    print("OK: validate_position accepts a legal position and rejects 3 invalid ones (4/4)")


if __name__ == "__main__":
    test_start_position()
    test_matches_rust_on_real_positions()
    print("OK: Python and Rust feature builders agree (2/2)")
    test_validate_position_accepts_legal_and_rejects_invalid()
