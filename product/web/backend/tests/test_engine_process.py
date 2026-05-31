"""Tests for engine/process.py.

Integration tests (marked with ``@pytest.mark.skipif``) are skipped when the
engine binary is not present at ``settings.engine_path``.  The binary lives at
``engine/target/release/togyzkumalaq-engine`` relative to the repo root.

Unit tests (Python-side apply_move / _apply_move_to_pos) always run.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from app.config import settings
from app.engine.process import (
    EngineProcess,
    START_POSITION,
    _apply_move_to_pos,
)
from app.engine.stream import BestMove, TerminalResult

_ENGINE_MISSING = not Path(settings.engine_path).exists()
_SKIP_INTEGRATION = pytest.mark.skipif(
    _ENGINE_MISSING, reason="engine binary not built — run: cd engine && cargo build --release"
)


# ---------------------------------------------------------------------------
# Unit tests — Python-side move application (_apply_move_to_pos)
# ---------------------------------------------------------------------------

def test_apply_move_changes_position():
    """After any legal move, the position string must change."""
    new_pos = _apply_move_to_pos(START_POSITION, 0)
    assert new_pos != START_POSITION


def test_apply_move_switches_side():
    """Side to move must flip after a move."""
    # START_POSITION ends with /0 (White to move)
    new_pos = _apply_move_to_pos(START_POSITION, 0)
    assert new_pos.endswith("/1"), f"Expected side=1 after White's move, got: {new_pos}"


def test_apply_move_start_pit_sown_back():
    """For multi-stone moves, the source pit is NOT empty — the first stone sows back.

    Per engine/src/board.rs make_move(): when stones > 1, the first stone is
    deposited at the current_pit (source pit) before sowing the rest forward.
    So pit 0 starts empty (9 stones removed) then gets 1 stone back → 1 stone.
    Only a 1-stone pit would become truly empty after being played.
    """
    new_pos = _apply_move_to_pos(START_POSITION, 0)
    parts = new_pos.split("/")
    white_pits = [int(x) for x in parts[0].split(",")]
    # Source pit has 1 stone (sown back) not 0
    assert white_pits[0] == 1, f"Pit 0 should have 1 stone (sown back); got {white_pits}"


def test_apply_move_stone_count_conserved():
    """Total stone count on the board + kazan must equal 162."""
    new_pos = _apply_move_to_pos(START_POSITION, 3)
    parts = new_pos.split("/")
    white = sum(int(x) for x in parts[0].split(","))
    black = sum(int(x) for x in parts[1].split(","))
    kaz_w, kaz_b = (int(x) for x in parts[2].split(","))
    assert white + black + kaz_w + kaz_b == 162


def test_apply_move_invalid_pit_raises():
    """Playing an out-of-range pit raises ValueError."""
    with pytest.raises(ValueError):
        _apply_move_to_pos(START_POSITION, 9)


def test_apply_move_invalid_position_raises():
    """A malformed position string raises ValueError."""
    with pytest.raises(ValueError):
        _apply_move_to_pos("garbage", 0)


def test_apply_move_multiple_moves():
    """Chain legal moves alternating sides and verify no crash or conservation error."""
    # After White plays pit 0 → Black to move.  Black plays pit 0.
    # After Black plays pit 0 → White to move.  White plays pit 1.
    pos = START_POSITION
    # White plays pit 0
    pos = _apply_move_to_pos(pos, 0)  # now Black to move
    # Black plays pit 0
    pos = _apply_move_to_pos(pos, 0)  # now White to move
    # White plays pit 1
    pos = _apply_move_to_pos(pos, 1)  # now Black to move

    # Verify stone conservation after 3 moves
    parts = pos.split("/")
    white = sum(int(x) for x in parts[0].split(","))
    black = sum(int(x) for x in parts[1].split(","))
    kaz_w, kaz_b = (int(x) for x in parts[2].split(","))
    assert white + black + kaz_w + kaz_b == 162


def test_apply_move_stones_sown_correctly():
    """Pit 0 has 9 stones; first stone goes BACK to pit 0 (Rust rule), rest sow into 1..8.

    From engine/src/board.rs make_move():
      - stones > 1: first stone deposited back to source pit, then remaining 8
        stones go to pits 1–8.
      - All 9 sown pits remain on White's side (8 wraps are needed to reach Black).
    """
    new_pos = _apply_move_to_pos(START_POSITION, 0)
    parts = new_pos.split("/")
    white = [int(x) for x in parts[0].split(",")]
    black = [int(x) for x in parts[1].split(",")]

    # White pit 0: emptied then 1 stone sown back → 1 stone
    assert white[0] == 1, f"white[0]={white[0]}, expected 1 (stone sown back)"
    # White pits 1–8: each gains 1 → 10 stones
    for i in range(1, 9):
        assert white[i] == 10, f"white[{i}]={white[i]}, expected 10"
    # Black pits unchanged (9): last stone went to white[8], no overflow to Black
    for i in range(9):
        assert black[i] == 9, f"black[{i}]={black[i]}, expected 9"


# ---------------------------------------------------------------------------
# Integration tests — require real engine binary
# ---------------------------------------------------------------------------

@_SKIP_INTEGRATION
@pytest.mark.asyncio
async def test_engine_emits_bestmove():
    """Engine must respond with BestMove for the start position."""
    proc = EngineProcess(settings.engine_path)
    await proc.start()
    try:
        out = []
        async for ev in proc.think(position_pos=START_POSITION, time_ms=200):
            out.append(ev)
        assert len(out) >= 1, "Engine emitted no response"
        last = out[-1]
        assert isinstance(last, (BestMove, TerminalResult)), (
            f"Last event should be BestMove or TerminalResult, got {type(last)}"
        )
        if isinstance(last, BestMove):
            assert 0 <= last.move <= 8, f"Move index {last.move} out of 0–8 range"
    finally:
        await proc.stop()


@_SKIP_INTEGRATION
@pytest.mark.asyncio
async def test_engine_alive_after_start():
    proc = EngineProcess(settings.engine_path)
    assert not proc.alive
    await proc.start()
    try:
        assert proc.alive
    finally:
        await proc.stop()
    assert not proc.alive


@_SKIP_INTEGRATION
@pytest.mark.asyncio
async def test_engine_new_game():
    """newgame command should succeed without errors."""
    proc = EngineProcess(settings.engine_path)
    await proc.start()
    try:
        await proc.new_game()  # should not raise
    finally:
        await proc.stop()


@_SKIP_INTEGRATION
@pytest.mark.asyncio
async def test_engine_apply_move_returns_new_pos():
    """apply_move (Python-side) returns a changed, valid position.

    We also verify that the engine can search from the resulting position.
    """
    proc = EngineProcess(settings.engine_path)
    await proc.start()
    try:
        new_pos = await proc.apply_move(position_pos=START_POSITION, move=4)
        assert isinstance(new_pos, str)
        assert new_pos != START_POSITION

        # Engine must be able to search from the new position
        out = []
        async for ev in proc.think(position_pos=new_pos, time_ms=200):
            out.append(ev)
        assert isinstance(out[-1], (BestMove, TerminalResult))
    finally:
        await proc.stop()


@_SKIP_INTEGRATION
@pytest.mark.asyncio
async def test_engine_push_position():
    """push_position should not raise and engine should remain alive."""
    proc = EngineProcess(settings.engine_path)
    await proc.start()
    try:
        pos_after_human = _apply_move_to_pos(START_POSITION, 2)
        await proc.push_position(pos_after_human)  # should not raise
        assert proc.alive
    finally:
        await proc.stop()
