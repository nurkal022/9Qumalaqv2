#!/usr/bin/env python3
"""Guards for the 9qum -> training-bin converter.

Three failure modes this catches, all of which silently poison training:
  * a record that no longer decodes to the position it came from
  * the value stored from the wrong side's perspective (today's class of bug)
  * the same game appearing in both train and val, which leaks the outcome

Run: python3.12 research/data/test_convert_9qum.py
"""
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(__file__))
import convert_9qum as cv

TRAIN = "data/9qum/train/train.bin"
VAL = "data/9qum/train/val.bin"
SPLIT = "data/9qum/train/split.json"


def read_records(path, limit=None):
    out = []
    with open(path, "rb") as f:
        while True:
            buf = f.read(cv.RECORD_SIZE)
            if len(buf) < cv.RECORD_SIZE:
                break
            out.append(cv.decode_record(buf))
            if limit and len(out) >= limit:
                break
    return out


def test_record_size_and_roundtrip():
    assert cv.RECORD_SIZE == 68
    rec = cv.encode_record(
        pits=[9] * 18, kazan=[0, 0], tuzdyk=[None, None], to_move=0,
        move=6, value=0.75, score=0.1, mask=cv.MASK_POLICY | cv.MASK_VALUE_NET | cv.MASK_SCORE)
    d = cv.decode_record(rec)
    assert d["pits"] == [9] * 18 and d["kazan"] == [0, 0]
    assert d["tuzdyk"] == [-1, -1] and d["to_move"] == 0
    assert d["policy"][6] == 1.0 and sum(d["policy"]) == 1.0
    assert abs(d["value"] - 0.75) < 1e-6 and abs(d["score"] - 0.1) < 1e-6
    assert d["mask"] & cv.MASK_VALUE_NET


def test_value_perspective_matches_outcomes():
    """A value stored for the wrong side turns the label set into noise with the sign
    flipped. Their net is 82-84% accurate, so a correct conversion must land near that."""
    recs = read_records(VAL, limit=20000)
    net = [r for r in recs if r["mask"] & cv.MASK_VALUE_NET]
    assert len(net) > 1000, f"expected net-labelled records in val, got {len(net)}"
    # value > 0.5 must mean "the side to move went on to win" more often than not
    agree = sum(1 for r in net if (r["value"] > 0.5) == (r["outcome_stm"] > 0.5))
    rate = agree / len(net)
    assert rate >= 0.80, f"value perspective looks wrong: only {100 * rate:.1f}% agreement"


def test_splits_are_disjoint_by_game():
    with open(SPLIT, encoding="utf-8") as f:
        split = json.load(f)
    val_games = set(split["val_games"])
    assert len(val_games) > 100, "val split is suspiciously small"
    train_games = set(split["train_games"])
    assert not (val_games & train_games), "a game appears in both splits"
    assert os.path.getsize(TRAIN) % cv.RECORD_SIZE == 0
    assert os.path.getsize(VAL) % cv.RECORD_SIZE == 0


if __name__ == "__main__":
    test_record_size_and_roundtrip()
    test_value_perspective_matches_outcomes()
    test_splits_are_disjoint_by_game()
    print("OK: converter records, value perspective and splits (3/3)")
