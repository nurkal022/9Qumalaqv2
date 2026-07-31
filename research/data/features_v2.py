#!/usr/bin/env python3
"""NNUE v2 feature layout — the Python side of the contract in engine/src/nnue.rs.

292 features, 23 active. Keep this file and build_features_v2() in nnue.rs in lockstep;
test_features_v2.py fails if they drift.
"""
NUM_FEATURES = 292
NUM_BUCKETS = 4


def count_bucket(c: int) -> int:
    if c <= 9:
        return c
    if c <= 12:
        return 10
    if c <= 16:
        return 11
    if c <= 24:
        return 12
    return 13


def _tuz_rel(tuzdyk, side):
    """9qum stores tuzdyk[p] as an absolute pit index on the opponent's side; the engine
    wants the index inside that side's row, or 9 for none."""
    t = tuzdyk[side]
    if t is None or t < 0:
        return 9
    return t - 9 if side == 0 else t


def build_features(pits, kazan, tuzdyk, to_move):
    me, opp = to_move, 1 - to_move
    rows = [pits[0:9], pits[9:18]]
    f = []
    for i in range(9):
        f.append(i * 14 + count_bucket(rows[me][i]))
    for i in range(9):
        f.append(126 + i * 14 + count_bucket(rows[opp][i]))
    f.append(252 + min(8, kazan[me] // 10))
    f.append(261 + min(8, kazan[opp] // 10))
    f.append(270 + _tuz_rel(tuzdyk, me))
    f.append(280 + _tuz_rel(tuzdyk, opp))
    f.append(290 + (sum(pits) % 2))
    assert len(f) == 23
    return f


def phase_bucket(pits) -> int:
    total = sum(pits)
    if total >= 121:
        return 0
    if total >= 81:
        return 1
    if total >= 41:
        return 2
    return 3


def pos_string(pits, kazan, tuzdyk, to_move) -> str:
    tw = -1 if tuzdyk[0] is None else tuzdyk[0] - 9
    tb = -1 if tuzdyk[1] is None else tuzdyk[1]
    return (",".join(map(str, pits[0:9])) + "/" + ",".join(map(str, pits[9:18])) +
            f"/{kazan[0]},{kazan[1]}/{tw},{tb}/{to_move}")
