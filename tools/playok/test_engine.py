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


def test_clock_below_floor_never_exceeds_clock():
    # 50 ms left is below the usual 100 ms floor -- the floor must not push the
    # budget past what is actually left on the clock.
    assert move_budget_ms(FORTY, 1800, 12000, clock_left_ms=50) == 50


def test_nonpositive_clock_still_returns_a_positive_budget():
    # A clock that is already exhausted (0) or overrun (negative -- spent_ms can exceed
    # the nominal clock once it tracks real elapsed time) must still yield something the
    # engine subprocess can parse as a positive `go time`, never zero or negative.
    zero = move_budget_ms(FORTY, 1800, 12000, clock_left_ms=0)
    negative = move_budget_ms(FORTY, 1800, 12000, clock_left_ms=-100)
    assert zero > 0
    assert negative > 0
    # And where the clock IS positive, the budget still never exceeds it.
    assert move_budget_ms(FORTY, 1800, 12000, clock_left_ms=1) <= 1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("OK", name)
