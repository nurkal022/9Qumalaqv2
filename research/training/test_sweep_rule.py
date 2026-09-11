#!/usr/bin/env python3
"""Regression guard for the end-game SWEEP rule in the training referee.

The Python game logic (alphazero-code/alphazero/game.py) generates every NN value
label. It used to compare RAW kazans at an empty-side terminal, giving the wrong
winner on ~54% of such terminals (proven against PlayOK results 96/96). The correct
rule sweeps each side's remaining board stones into its own kazan first. This test
fails if that fix ever regresses.

Run: python3.12 research/training/test_sweep_rule.py   (or via pytest)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "alphazero-code", "alphazero"))
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
    # Black is to move with an empty side -- that is what makes this terminal under
    # the real rule (stm must be the empty side, not just a default).
    g.set_state(_state([6, 6, 6, 6, 6, 0, 0, 0, 0], [0] * 9, 66, 66, stm=1))
    assert g.get_winner() == Player.WHITE


def test_sweep_conserves_162():
    g = TogyzQumalaq()
    g.set_state(_state([0] * 9, [3, 0, 5, 0, 2, 0, 1, 0, 0], 75, 76))
    s = g.get_state()
    assert int(s.kazan[0]) + int(s.kazan[1]) + int(s.pits[0].sum()) + int(s.pits[1].sum()) == 162


def test_empty_side_not_terminal_when_opponent_to_move():
    g = TogyzQumalaq()
    g.state = _state([0] * 9, [0, 0, 0, 0, 0, 0, 0, 0, 3], 80, 79, stm=1)
    assert not g.is_terminal(), "White emptied itself; Black must still move"


def test_empty_side_terminal_when_it_is_to_move():
    g = TogyzQumalaq()
    g.state = _state([0] * 9, [0, 0, 0, 0, 0, 0, 0, 0, 3], 80, 79, stm=0)
    assert g.is_terminal()
    assert g.get_winner() == Player.BLACK  # 80 vs 79+3


def test_forced_feed_continues_game():
    g = TogyzQumalaq()
    g.state = _state([0] * 9, [0, 0, 0, 0, 0, 0, 0, 0, 3], 80, 79, stm=1)
    ok, winner = g.make_move(8)  # 1 stays in pit 9, 2 land on White pits 1-2
    assert ok and winner is None
    assert int(g.state.pits[0].sum()) == 2
    assert not g.is_terminal()


if __name__ == "__main__":
    test_sweep_breaks_kazan_tie()
    test_sweep_matches_playok_example()
    test_sweep_conserves_162()
    print("OK: training referee uses the correct end-game sweep rule (3/3)")
    test_empty_side_not_terminal_when_opponent_to_move()
    test_empty_side_terminal_when_it_is_to_move()
    test_forced_feed_continues_game()
    print("OK: terminal rule fires only when the side to move has no stones (3/3)")
