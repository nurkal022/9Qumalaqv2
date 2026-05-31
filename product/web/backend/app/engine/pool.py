"""Engine pool: serialises think() calls via asyncio.Lock, auto-restarts subprocess once.

EngineProcess.think() is an async generator that yields InfoLine | BestMove | TerminalResult.
This pool iterates it to completion and returns an EngineResult dataclass.

No subscriber/fan-out machinery — the engine does not stream info lines in serve mode.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path

from app.engine.process import EngineProcess
from app.engine.stream import BestMove, TerminalResult


@dataclass
class EngineResult:
    move: int                    # 0-indexed pit (0-8); -1 if terminal (game over)
    final_eval_cp: int | None
    final_depth: int | None
    think_time_ms: int
    terminal: str | None = None  # "white_win" / "black_win" / "draw" when game is over


class EngineError(RuntimeError):
    pass


class EnginePool:
    """Single subprocess engine wrapper.

    Serialises think() calls via asyncio.Lock so concurrent requests from
    different games queue up.  Auto-restarts the subprocess once if it dies.
    """

    def __init__(self, binary: Path) -> None:
        self._binary = binary
        self._proc = EngineProcess(binary)
        self._lock = asyncio.Lock()
        self._build_hash = "unknown"

    @property
    def alive(self) -> bool:
        return self._proc.alive

    @property
    def build_hash(self) -> str:
        return self._build_hash

    async def start(self) -> None:
        await self._proc.start()
        try:
            self._build_hash = hashlib.sha256(self._binary.read_bytes()).hexdigest()[:12]
        except Exception:
            self._build_hash = "unknown"

    async def stop(self) -> None:
        await self._proc.stop()

    async def think(self, *, position_pos: str, time_ms: int) -> EngineResult:
        """Ask the engine for the best move from *position_pos* with *time_ms* budget.

        Serialises calls (one think at a time across the whole pool).
        Auto-restarts subprocess once on death.

        EngineProcess.think() is an async generator; we iterate it to completion
        and return the BestMove (or TerminalResult) wrapped in an EngineResult.
        """
        async with self._lock:
            for attempt in (0, 1):
                if not self._proc.alive:
                    try:
                        await self._proc.start()
                    except Exception as e:
                        if attempt == 1:
                            raise EngineError("engine_start_failed") from e
                        continue
                try:
                    best_move: BestMove | None = None
                    terminal: TerminalResult | None = None
                    async for event in self._proc.think(
                        position_pos=position_pos, time_ms=time_ms
                    ):
                        if isinstance(event, BestMove):
                            best_move = event
                        elif isinstance(event, TerminalResult):
                            terminal = event

                    if terminal is not None:
                        return EngineResult(
                            move=-1,
                            final_eval_cp=None,
                            final_depth=None,
                            think_time_ms=time_ms,
                            terminal=terminal.result,
                        )
                    if best_move is not None:
                        return EngineResult(
                            move=best_move.move,
                            final_eval_cp=best_move.score,
                            final_depth=best_move.depth,
                            think_time_ms=best_move.time_ms or time_ms,
                        )
                    raise RuntimeError("engine returned no bestmove or terminal line")
                except RuntimeError:
                    await self._proc.stop()
                    if attempt == 1:
                        raise EngineError("engine_died")
            raise EngineError("engine_unreachable")

    async def push_position(self, position_pos: str) -> None:
        """Notify engine of a position for repetition detection."""
        async with self._lock:
            if self._proc.alive:
                await self._proc.push_position(position_pos)

    async def new_game(self) -> None:
        """Reset engine TT and game history at start of a new game."""
        async with self._lock:
            if self._proc.alive:
                await self._proc.new_game()

    async def apply_move(self, *, position_pos: str, move: int) -> str:
        """Apply a move using the Python rules port (no engine round-trip needed).

        Provided here so callers can use a single dependency (the pool) for all
        engine ops.
        """
        return await self._proc.apply_move(position_pos=position_pos, move=move)
