#!/usr/bin/env python3
"""Tests for the pure helpers in tools/playok/engine.py.

Run: python3.12 tools/playok/test_engine.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import START_POSITION, move_budget_ms  # noqa: E402

FORTY = "3,3,3,3,3,3,2,0,0/3,3,3,3,3,3,2,0,0/61,61/-1,-1/0"       # 20 + 20 = 40 stones
FORTY_ONE = "3,3,3,3,3,3,3,0,0/3,3,3,3,3,3,2,0,0/61,60/-1,-1/0"   # 21 + 20 = 41 stones


def test_start_position_uses_base():
    assert move_budget_ms(START_POSITION, 1800, 12000) == 1800


def test_forty_stones_uses_endgame_budget():
    assert move_budget_ms(FORTY, 1800, 12000) == 12000


def test_forty_one_stones_uses_base():
    assert move_budget_ms(FORTY_ONE, 1800, 12000) == 1800


def test_no_clock_means_no_cap():
    assert move_budget_ms(FORTY, 1800, 12000, clock_left_ms=None) == 12000


def test_clock_caps_budget_to_keep_reserve():
    # 130 s left, 120 s reserve -> at most 10 s this move.
    assert move_budget_ms(FORTY, 1800, 12000, clock_left_ms=130_000) == 10_000


def test_clock_nearly_out_never_below_half_clock_or_100ms():
    assert move_budget_ms(FORTY, 1800, 12000, clock_left_ms=1_000) == 500
    assert move_budget_ms(FORTY, 1800, 12000, clock_left_ms=100) == 100


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("OK", name)
