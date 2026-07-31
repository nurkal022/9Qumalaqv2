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


if __name__ == "__main__":
    test_start_position()
    test_matches_rust_on_real_positions()
    print("OK: Python and Rust feature builders agree (2/2)")
