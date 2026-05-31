"""Tests for engine/pool.py.

EngineProcess.think() is an async generator that yields BestMove | TerminalResult | InfoLine.
FakeProc mirrors this interface exactly.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import pytest

from app.engine.pool import EnginePool, EngineError
from app.engine.stream import BestMove, TerminalResult


class FakeProc:
    """Stand-in for EngineProcess.

    Implements .alive, .start(), .stop(), .think() (async generator),
    .push_position(), .new_game(), and .apply_move().
    """

    def __init__(self, binary: Path | None = None):
        self.alive = False
        self._fail_next_start = False
        self._die_next_think = False
        self._terminal_next = False

    async def start(self):
        if self._fail_next_start:
            self._fail_next_start = False
            raise RuntimeError("boom")
        self.alive = True

    async def stop(self):
        self.alive = False

    async def think(self, *, position_pos: str, time_ms: int):
        """Async generator matching EngineProcess.think() signature."""
        if self._die_next_think:
            self._die_next_think = False
            raise RuntimeError("engine died mid-think")
        if self._terminal_next:
            self._terminal_next = False
            yield TerminalResult(result="white_win")
            return
        yield BestMove(move=3, score=12, depth=8, nodes=100, time_ms=time_ms, nps=0)

    async def push_position(self, position_pos: str):
        pass

    async def new_game(self):
        pass

    async def apply_move(self, *, position_pos: str, move: int) -> str:
        return "modified-pos"


async def test_think_returns_bestmove(monkeypatch):
    """Pool.think() returns EngineResult with correct move and eval fields."""
    pool = EnginePool(Path("/dev/null"))
    fake = FakeProc()
    fake.alive = True
    monkeypatch.setattr(pool, "_proc", fake)
    res = await pool.think(position_pos="any", time_ms=100)
    assert res.move == 3
    assert res.final_eval_cp == 12
    assert res.final_depth == 8


async def test_think_terminal_position(monkeypatch):
    """Pool.think() returns EngineResult with move=-1 when engine reports terminal."""
    pool = EnginePool(Path("/dev/null"))
    fake = FakeProc()
    fake.alive = True
    fake._terminal_next = True
    monkeypatch.setattr(pool, "_proc", fake)
    res = await pool.think(position_pos="any", time_ms=100)
    assert res.move == -1
    assert res.terminal == "white_win"


async def test_think_auto_restarts_after_death(monkeypatch):
    """After subprocess dies mid-think, pool restarts and succeeds on next call."""
    pool = EnginePool(Path("/dev/null"))
    fake = FakeProc()
    fake.alive = True
    fake._die_next_think = True
    monkeypatch.setattr(pool, "_proc", fake)

    # First think will die; pool retries once (restart + think), second attempt succeeds
    # because _die_next_think is cleared after first trigger.
    # But after restart alive=True, so second attempt in the loop should pass.
    res = await pool.think(position_pos="x", time_ms=10)
    assert res.move == 3


async def test_think_raises_engine_error_if_both_attempts_fail(monkeypatch):
    """EngineError raised when subprocess dies on both retry attempts."""
    pool = EnginePool(Path("/dev/null"))

    class AlwaysDyingProc(FakeProc):
        async def think(self, *, position_pos: str, time_ms: int):
            # Always raises RuntimeError
            raise RuntimeError("always dead")
            yield  # make this an async generator

    fake = AlwaysDyingProc()
    fake.alive = True
    monkeypatch.setattr(pool, "_proc", fake)
    with pytest.raises(EngineError):
        await pool.think(position_pos="x", time_ms=10)


async def test_think_serializes_concurrent(monkeypatch):
    """Concurrent think() calls are serialised (both complete successfully)."""
    pool = EnginePool(Path("/dev/null"))
    fake = FakeProc()
    fake.alive = True
    monkeypatch.setattr(pool, "_proc", fake)
    results = await asyncio.gather(
        pool.think(position_pos="a", time_ms=10),
        pool.think(position_pos="b", time_ms=10),
    )
    assert all(r.move == 3 for r in results)


async def test_push_position_noop_when_alive(monkeypatch):
    """push_position() completes without error when proc is alive."""
    pool = EnginePool(Path("/dev/null"))
    fake = FakeProc()
    fake.alive = True
    monkeypatch.setattr(pool, "_proc", fake)
    await pool.push_position("some-pos")  # should not raise


async def test_push_position_noop_when_dead(monkeypatch):
    """push_position() is silently skipped when proc is not alive."""
    pool = EnginePool(Path("/dev/null"))
    fake = FakeProc()
    fake.alive = False
    monkeypatch.setattr(pool, "_proc", fake)
    await pool.push_position("some-pos")  # should not raise


async def test_new_game_noop_when_alive(monkeypatch):
    """new_game() completes without error when proc is alive."""
    pool = EnginePool(Path("/dev/null"))
    fake = FakeProc()
    fake.alive = True
    monkeypatch.setattr(pool, "_proc", fake)
    await pool.new_game()  # should not raise


async def test_apply_move_delegates_to_proc(monkeypatch):
    """apply_move() delegates to _proc.apply_move() and returns result."""
    pool = EnginePool(Path("/dev/null"))
    fake = FakeProc()
    fake.alive = True
    monkeypatch.setattr(pool, "_proc", fake)
    result = await pool.apply_move(position_pos="pos", move=3)
    assert result == "modified-pos"
