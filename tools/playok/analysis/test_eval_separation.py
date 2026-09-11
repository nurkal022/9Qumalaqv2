#!/usr/bin/env python3
"""Run: python3.12 tools/playok/analysis/test_eval_separation.py"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_separation import auc, midgame_positions, white_won  # noqa: E402

GAME = """# PlayOK togyzkumalak  20260101_000000
# White(seat0)=bot  Black(seat1)=opp
1. W4 [43(10)]  9,9,9,1,10,10,10,10,10/10,10,0,9,9,9,9,9,9/10,0/-1,-1/1
2. B9 [98]  10,10,10,2,11,11,11,11,10/10,10,0,9,9,9,9,9,1/10,0/-1,-1/0
3. W9 [99(12)]  10,10,10,2,11,11,11,11,1/11,11,1,10,10,10,10,10,0/12,0/-1,-1/1
4. B8 [88(12)]  0,0,0,0,0,0,0,0,0/11,11,1,10,10,10,10,1,1/60,20/-1,-1/0
"""


def test_auc_perfect_and_random():
    assert auc([3, 2], [1, 0]) == 1.0
    assert auc([1, 1], [1, 1]) == 0.5
    assert auc([0], [1]) == 0.0


def test_white_won_uses_sweep():
    # final pos above: white 60+0=60, black 20+65=85 -> black won
    assert white_won("0,0,0,0,0,0,0,0,0/11,11,1,10,10,10,10,1,1/60,20/-1,-1/0") is False
    assert white_won("5,0,0,0,0,0,0,0,0/0,0,0,0,0,0,0,0,0/80,77/-1,-1/1") is True


def test_midgame_positions_takes_half_ply():
    with tempfile.TemporaryDirectory() as d:
        Path(d, "game_x_vs_opp.txt").write_text(GAME)
        rows = midgame_positions(Path(d))
    assert rows == [("10,10,10,2,11,11,11,11,10/10,10,0,9,9,9,9,9,1/10,0/-1,-1/0", 0)]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("OK", name)
