"""
Low-level async subprocess wrapper for the Togyzkumalaq engine.

Protocol summary (see protocol_notes.md):
- Start with `serve` argument; engine emits `ready` on stdout.
- `go time <ms> pos <position_string>` → single-line response: `bestmove ...` or `terminal ...`
- `position <position_string>` → `ready`  (push position into game history)
- `newgame` → `ready`  (clear TT + game history)
- `ping` → `pong`
- `quit` → engine exits

There are NO streaming `info` lines in serve mode.

apply_move design note
----------------------
The engine has no command to apply a move and return the new position string.
The existing web/server.py design passes the full board state from the client
with every `go` call.  For programmatic callers (e.g. Task 11) that need to
compute new positions server-side, apply_move() implements a Python-side
board state transition.  See _apply_move_to_pos() for the implementation.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncIterator

from app.engine.stream import BestMove, InfoLine, TerminalResult, parse_line

# The start position string accepted by the engine.
START_POSITION = "9,9,9,9,9,9,9,9,9/9,9,9,9,9,9,9,9,9/0,0/-1,-1/0"

NUM_PITS = 9


class EngineProcess:
    """Manage a single engine subprocess in serve mode.

    Usage::

        proc = EngineProcess(settings.engine_path)
        await proc.start()
        try:
            async for event in proc.think(position_pos=START_POSITION, time_ms=1000):
                if isinstance(event, BestMove):
                    print("engine plays pit", event.move + 1)
        finally:
            await proc.stop()
    """

    def __init__(self, binary: Path) -> None:
        self.binary = binary
        self._proc: asyncio.subprocess.Process | None = None

    @property
    def alive(self) -> bool:
        """True if the subprocess is running."""
        return self._proc is not None and self._proc.returncode is None

    async def start(self) -> None:
        """Launch the engine subprocess and wait for the `ready` handshake."""
        self._proc = await asyncio.create_subprocess_exec(
            str(self.binary),
            "serve",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        # Engine emits exactly one line: "ready"
        assert self._proc.stdout is not None
        banner = await asyncio.wait_for(
            self._proc.stdout.readline(), timeout=10.0
        )
        banner_text = banner.decode(errors="replace").strip()
        if banner_text != "ready":
            raise RuntimeError(
                f"Engine did not emit 'ready' on startup; got: {banner_text!r}"
            )

    async def stop(self) -> None:
        """Gracefully terminate the engine (SIGTERM, then SIGKILL after 2 s)."""
        if self._proc is None or self._proc.returncode is not None:
            return
        try:
            assert self._proc.stdin is not None
            self._proc.stdin.write(b"quit\n")
            await self._proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            self._proc.kill()

    async def send(self, cmd: str) -> None:
        """Write a command line to the engine's stdin."""
        assert self._proc and self._proc.stdin, "engine not started"
        self._proc.stdin.write((cmd + "\n").encode())
        await self._proc.stdin.drain()

    async def think(
        self, *, position_pos: str, time_ms: int
    ) -> AsyncIterator[InfoLine | BestMove | TerminalResult]:
        """Ask the engine to search from *position_pos*.

        Yields parsed response objects.  In practice the engine emits exactly
        one meaningful line (BestMove or TerminalResult) since serve mode is
        silent.  The generator exits after that line.

        Parameters
        ----------
        position_pos:
            Full position string in engine format:
            ``w0,...,w8/b0,...,b8/kw,kb/tw,tb/side``
        time_ms:
            Thinking time budget in milliseconds.
        """
        await self.send(f"go time {time_ms} pos {position_pos}")
        assert self._proc and self._proc.stdout
        while True:
            raw = await self._proc.stdout.readline()
            if not raw:
                raise RuntimeError("engine closed stdout unexpectedly")
            parsed = parse_line(raw.decode(errors="replace"))
            if parsed is None:
                # ready / pong / empty — skip
                continue
            yield parsed
            if isinstance(parsed, (BestMove, TerminalResult)):
                return

    async def push_position(self, position_pos: str) -> None:
        """Push a position into the engine's game-history for repetition detection.

        After the human makes a move, call this so the engine's repetition
        table includes the position the human reached.  The engine responds
        with ``ready`` which is consumed here.

        Parameters
        ----------
        position_pos:
            Full position string after the human's move.
        """
        await self.send(f"position {position_pos}")
        assert self._proc and self._proc.stdout
        response = await asyncio.wait_for(
            self._proc.stdout.readline(), timeout=5.0
        )
        text = response.decode(errors="replace").strip()
        if text != "ready":
            raise RuntimeError(
                f"Unexpected response to 'position' command: {text!r}"
            )

    async def new_game(self) -> None:
        """Reset the engine's TT and game history for a new game."""
        await self.send("newgame")
        assert self._proc and self._proc.stdout
        response = await asyncio.wait_for(
            self._proc.stdout.readline(), timeout=5.0
        )
        text = response.decode(errors="replace").strip()
        if text != "ready":
            raise RuntimeError(
                f"Unexpected response to 'newgame' command: {text!r}"
            )

    async def apply_move(self, *, position_pos: str, move: int) -> str:
        """Apply *move* to *position_pos* and return the resulting position string.

        This is implemented **entirely in Python** because the engine has no
        command to apply a move and return the new board state (see
        protocol_notes.md §8).

        Parameters
        ----------
        position_pos:
            Full position string before the move.
        move:
            0-based pit index (0–8) to play.

        Returns
        -------
        str
            New position string in engine format after the move is applied.

        Raises
        ------
        ValueError
            If the position string is malformed or the move is illegal.
        """
        return _apply_move_to_pos(position_pos, move)


# ---------------------------------------------------------------------------
# Python-side board state transition
# ---------------------------------------------------------------------------

def _parse_pos(pos: str) -> tuple[
    list[int], list[int], list[int], list[int], int
]:
    """Parse position string into mutable lists.

    Returns (white_pits, black_pits, kazan, tuzdyk, side)
    """
    parts = pos.split("/")
    if len(parts) != 5:
        raise ValueError(f"Expected 5 '/'-separated parts, got {len(parts)}: {pos!r}")
    white = [int(x) for x in parts[0].split(",")]
    black = [int(x) for x in parts[1].split(",")]
    kaz = [int(x) for x in parts[2].split(",")]
    tuz = [int(x) for x in parts[3].split(",")]
    side = int(parts[4])
    if len(white) != NUM_PITS or len(black) != NUM_PITS:
        raise ValueError("Expected 9 pits per side")
    if len(kaz) != 2 or len(tuz) != 2:
        raise ValueError("Kazan/tuzdyk arrays must have 2 elements each")
    return white, black, kaz, tuz, side


def _encode_pos(
    white: list[int],
    black: list[int],
    kaz: list[int],
    tuz: list[int],
    side: int,
) -> str:
    wp = ",".join(str(x) for x in white)
    bp = ",".join(str(x) for x in black)
    k = f"{kaz[0]},{kaz[1]}"
    t = f"{tuz[0]},{tuz[1]}"
    return f"{wp}/{bp}/{k}/{t}/{side}"


def _apply_move_to_pos(pos: str, pit: int) -> str:
    """Pure-Python Togyzkumalak move application.

    Faithfully mirrors engine/src/board.rs ``make_move()``.

    Sowing rules (from Rust source):
      - Single stone (stones == 1): place it in (pit+1), wrapping to opponent's
        row after pit 8.
      - Multiple stones: first stone goes BACK to the source pit (pit itself);
        then remaining stones go to pit+1, pit+2, … wrapping through opponent's
        row and back to own row.

    Row order: own pits 0..8, then opponent's pits 0..8, repeating.

    Capture / tuzdyk rules:
      - Checked only on the pit where the LAST stone lands.
      - Only triggers when landing on the OPPONENT's side.
      - Tuzdyk capture: if landing pit == mover's tuzdyk → capture all.
      - New tuzdyk: if count == 3 AND pit_index != 8 AND mover has no tuzdyk
        AND opponent's tuzdyk is not at the same index → create tuzdyk + capture.
      - Normal capture: count is even AND > 0.
    """
    white, black, kaz, tuz, side = _parse_pos(pos)
    pits = [white, black]
    opp = 1 - side

    if pit < 0 or pit >= NUM_PITS:
        raise ValueError(f"Pit index {pit} out of range 0–8")
    if pits[side][pit] == 0:
        raise ValueError(f"Pit {pit} is empty for side {side}")

    stones = pits[side][pit]
    pits[side][pit] = 0

    def _deposit(cur_side: int, cur_pit: int) -> None:
        """Deposit one stone, respecting tuzdyk captures."""
        if cur_side == opp and tuz[side] == cur_pit:
            # Our tuzdyk on opponent's side — collect
            kaz[side] += 1
        elif cur_side == side and tuz[opp] == cur_pit:
            # Opponent's tuzdyk on our side — they collect
            kaz[opp] += 1
        else:
            pits[cur_side][cur_pit] += 1

    current_pit = pit
    current_side = side

    if stones == 1:
        # Single stone: move to next pit
        current_pit += 1
        if current_pit > 8:
            current_pit = 0
            current_side = opp
        _deposit(current_side, current_pit)
    else:
        # First stone goes back to source pit
        _deposit(current_side, current_pit)
        remaining = stones - 1
        while remaining > 0:
            current_pit += 1
            if current_pit > 8:
                current_pit = 0
                current_side = 1 - current_side
            _deposit(current_side, current_pit)
            remaining -= 1

    # Check capture / tuzdyk: only if last stone landed on opponent's side
    # and not in a tuzdyk pit
    is_tuzdyk_pit = (
        (current_side == opp and tuz[side] == current_pit)
        or (current_side == side and tuz[opp] == current_pit)
    )

    if current_side == opp and not is_tuzdyk_pit:
        count = pits[opp][current_pit]

        # Tuzdyk creation check
        def _can_create_tuzdyk() -> bool:
            if tuz[side] != -1:
                return False
            if current_pit == 8:
                return False
            if tuz[opp] == current_pit:
                return False
            return True

        if count == 3 and _can_create_tuzdyk():
            tuz[side] = current_pit
            kaz[side] += count
            pits[opp][current_pit] = 0
        elif count % 2 == 0 and count > 0:
            kaz[side] += count
            pits[opp][current_pit] = 0

    # Switch sides
    new_side = opp
    return _encode_pos(pits[0], pits[1], kaz, tuz, new_side)
