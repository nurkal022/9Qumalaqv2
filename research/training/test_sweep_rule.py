#!/usr/bin/env python3
"""Regression guard for the end-game SWEEP rule in the training referee.

The Python game logic (research/alphazero/game.py) generates every NN value
label. It used to compare RAW kazans at an empty-side terminal, giving the wrong
winner on ~54% of such terminals (proven against PlayOK results 96/96). The correct
rule sweeps each side's remaining board stones into its own kazan first. This test
fails if that fix ever regresses.

Run: python3.12 research/training/test_sweep_rule.py   (or via pytest)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "alphazero"))
from game import TogyzQumalaq, GameState, Player  # noqa: E402


def _state(white_pits, black_pits, kw, kb, stm=0):
    return GameState(
        pits=np.array([white_pits, black_pits], dtype=np.int32),
        kazan=np.array([kw, kb], dtype=np.int32),
        tuzdyk=np.array([-1, -1], dtype=np.int8),
        current_player=stm,
    )


def test_sweep_breaks_kazan_tie():
    # White empty, Black holds 30 on board, kazans tied 66-66 (total 162).
    # Raw-kazan -> Draw (the bug). Sweep -> Black 96 vs White 66 -> Black wins.
    g = TogyzQumalaq()
    g.set_state(_state([0] * 9, [4, 4, 4, 4, 4, 4, 4, 2, 0], 66, 66))
    assert g.get_winner() == Player.BLACK, "sweep rule regressed: tied kazans must not be a draw"


def test_sweep_matches_playok_example():
    # tg1589049-style: kazan 66-66, White has 30 on board, Black empty -> White wins.
    g = TogyzQumalaq()
    g.set_state(_state([6, 6, 6, 6, 6, 0, 0, 0, 0], [0] * 9, 66, 66))
    assert g.get_winner() == Player.WHITE


def test_sweep_conserves_162():
    g = TogyzQumalaq()
    g.set_state(_state([0] * 9, [3, 0, 5, 0, 2, 0, 1, 0, 0], 75, 76))
    s = g.get_state()
    assert int(s.kazan[0]) + int(s.kazan[1]) + int(s.pits[0].sum()) + int(s.pits[1].sum()) == 162


if __name__ == "__main__":
    test_sweep_breaks_kazan_tie()
    test_sweep_matches_playok_example()
    test_sweep_conserves_162()
    print("OK: training referee uses the correct end-game sweep rule (3/3)")
