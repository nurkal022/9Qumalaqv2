"""Unit tests for engine/stream.py — no subprocess required."""

import pytest
from app.engine.stream import parse_line, InfoLine, BestMove, TerminalResult


# ---------------------------------------------------------------------------
# bestmove parsing
# ---------------------------------------------------------------------------

def test_parse_bestmove_basic():
    """Minimal bestmove line."""
    out = parse_line("bestmove 4")
    assert isinstance(out, BestMove)
    assert out.move == 4


def test_parse_bestmove_full():
    """Full bestmove line as emitted by the engine."""
    out = parse_line(
        "bestmove 6 score 120 depth 14 nodes 48391 time 312 nps 155100"
    )
    assert isinstance(out, BestMove)
    assert out.move == 6
    assert out.score == 120
    assert out.depth == 14
    assert out.nodes == 48391
    assert out.time_ms == 312
    assert out.nps == 155100


def test_parse_bestmove_zero_pit():
    """Engine can play pit 0 (first pit)."""
    out = parse_line("bestmove 0 score -5 depth 8 nodes 1000 time 50 nps 20000")
    assert isinstance(out, BestMove)
    assert out.move == 0


def test_parse_bestmove_negative_score():
    """Negative score (losing position)."""
    out = parse_line("bestmove 3 score -2400 depth 10 nodes 2000 time 100 nps 20000")
    assert isinstance(out, BestMove)
    assert out.score == -2400


# ---------------------------------------------------------------------------
# terminal result parsing
# ---------------------------------------------------------------------------

def test_parse_terminal_white_win():
    out = parse_line("terminal white_win")
    assert isinstance(out, TerminalResult)
    assert out.result == "white_win"


def test_parse_terminal_black_win():
    out = parse_line("terminal black_win")
    assert isinstance(out, TerminalResult)
    assert out.result == "black_win"


def test_parse_terminal_draw():
    out = parse_line("terminal draw")
    assert isinstance(out, TerminalResult)
    assert out.result == "draw"


# ---------------------------------------------------------------------------
# info parsing (future-proofing; not emitted by engine today)
# ---------------------------------------------------------------------------

def test_parse_info_with_pv():
    """UCI-style info line with pv."""
    out = parse_line("info depth 8 score cp 24 nodes 1234 time 12 pv 1 2 9")
    assert isinstance(out, InfoLine)
    assert out.depth == 8
    assert out.cp == 24
    assert out.nodes == 1234
    assert out.time_ms == 12
    assert out.pv == ["1", "2", "9"]


def test_parse_info_no_pv():
    out = parse_line("info depth 5 score cp -10 nodes 500 time 8")
    assert isinstance(out, InfoLine)
    assert out.depth == 5
    assert out.cp == -10
    assert out.pv == []


# ---------------------------------------------------------------------------
# lines that return None
# ---------------------------------------------------------------------------

def test_parse_ready_returns_none():
    assert parse_line("ready") is None


def test_parse_pong_returns_none():
    assert parse_line("pong") is None


def test_parse_empty_returns_none():
    assert parse_line("") is None


def test_parse_whitespace_returns_none():
    assert parse_line("   ") is None


def test_parse_unknown_returns_none():
    assert parse_line("hello") is None


def test_parse_error_returns_none():
    """Error lines are not search results."""
    assert parse_line("error unknown command: foo") is None


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------

def test_parse_bestmove_strips_whitespace():
    out = parse_line("  bestmove 2 score 0 depth 1 nodes 9 time 1 nps 9000  ")
    assert isinstance(out, BestMove)
    assert out.move == 2
