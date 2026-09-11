"""
Synchronous wrapper around the Togyzkumalaq engine (serve mode) + a pure-Python
board-state transition, for the PlayOK bridge.

Engine: models/engine/baseline (NNUE + EGTB + opening book), launched as
`baseline serve`. Protocol:
    go time <ms> pos <pos>  -> "bestmove <pit> score ..."  or  "terminal <result>"
    position <pos>          -> "ready"
    newgame                 -> "ready"
    quit

Position string: w0,...,w8/b0,...,b8/kw,kb/tw,tb/side   (side 0=white,1=black)
Move = 0-based pit index 0..8.

The board transition `apply_move()` is ported verbatim from
product/web/backend/app/engine/process.py (_apply_move_to_pos), which mirrors
engine/src/board.rs make_move(). Kept dependency-free on purpose.
"""
from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
# TK_ENGINE env override lets us A/B a candidate engine live without touching prod.
DEFAULT_ENGINE = Path(os.environ.get("TK_ENGINE", str(REPO_ROOT / "models" / "engine" / "baseline")))
START_POSITION = "9,9,9,9,9,9,9,9,9/9,9,9,9,9,9,9,9,9/0,0/-1,-1/0"
NUM_PITS = 9

# Board-stone count at or below which a move gets the endgame budget. Measured on 244
# PlayOK games: losses end with a median of 30 stones on the board after 40-90 plies of
# lone-stone tempo play; branching there is 2-4, so extra time buys real depth.
ENDGAME_STONES = 40


def move_budget_ms(pos: str, base_ms: int, endgame_ms: int, *, threshold: int = ENDGAME_STONES,
                   clock_left_ms: int | None = None, reserve_ms: int = 120_000) -> int:
    """Thinking budget for `pos`: `endgame_ms` once <= `threshold` stones remain on the
    board, else `base_ms`. When the remaining clock is known, never spend more than what
    keeps `reserve_ms` on the clock, falling back to half the clock once even that is
    gone -- and the result is always clamped to `clock_left_ms` itself, so the 100 ms
    floor below can never push the budget past what is actually left on the clock. With
    no clock given the floor is a plain 100 ms so the engine still returns a move. The
    result is always at least 1 ms -- even with a zero or already-negative clock -- so
    the engine subprocess is always handed a positive value it can parse."""
    white, black, _, _, _ = _parse_pos(pos)
    budget = endgame_ms if sum(white) + sum(black) <= threshold else base_ms
    if clock_left_ms is not None:
        cap = max(clock_left_ms - reserve_ms, min(base_ms, clock_left_ms // 2))
        budget = min(budget, cap, clock_left_ms)
        return max(budget, min(100, clock_left_ms), 1)
    return max(budget, 100)


class Engine:
    """One engine subprocess in serve mode (synchronous)."""

    def __init__(self, binary: Path = DEFAULT_ENGINE):
        self.binary = Path(binary)
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        self._proc = subprocess.Popen(
            [str(self.binary), "serve"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1,
        )
        banner = self._proc.stdout.readline().strip()
        if banner != "ready":
            raise RuntimeError(f"engine did not emit 'ready'; got {banner!r}")

    def _cmd(self, line: str) -> None:
        assert self._proc and self._proc.stdin
        self._proc.stdin.write(line + "\n")
        self._proc.stdin.flush()

    def new_game(self) -> None:
        with self._lock:
            self._cmd("newgame")
            self._proc.stdout.readline()  # ready

    def bestmove(self, pos: str, time_ms: int = 1500):
        """Return (pit:int) to play, or ('terminal', result_str) if game over."""
        with self._lock:
            self._cmd(f"go time {time_ms} pos {pos}")
            while True:
                raw = self._proc.stdout.readline()
                if not raw:
                    raise RuntimeError("engine closed stdout")
                line = raw.strip()
                if line.startswith("bestmove"):
                    return int(line.split()[1])
                if line.startswith("terminal"):
                    return ("terminal", line.split(maxsplit=1)[1] if len(line.split()) > 1 else "")
                if line.startswith("error"):
                    raise ValueError(line)
                # ready / pong / blank -> ignore

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._cmd("quit")
                self._proc.wait(timeout=2)
            except Exception:
                self._proc.kill()

    @staticmethod
    def apply_move(pos: str, pit: int) -> str:
        return _apply_move_to_pos(pos, pit)

    @staticmethod
    def legal_moves(pos: str) -> list[int]:
        white, black, _, _, side = _parse_pos(pos)
        row = white if side == 0 else black
        return [i for i in range(NUM_PITS) if row[i] > 0]


# --------------------------------------------------------------------------
# Pure-Python board transition (ported from process.py / board.rs)
# --------------------------------------------------------------------------
def _parse_pos(pos: str):
    parts = pos.split("/")
    if len(parts) != 5:
        raise ValueError(f"bad position: {pos!r}")
    white = [int(x) for x in parts[0].split(",")]
    black = [int(x) for x in parts[1].split(",")]
    kaz = [int(x) for x in parts[2].split(",")]
    tuz = [int(x) for x in parts[3].split(",")]
    side = int(parts[4])
    if len(white) != NUM_PITS or len(black) != NUM_PITS:
        raise ValueError("expected 9 pits per side")
    return white, black, kaz, tuz, side


def _encode_pos(white, black, kaz, tuz, side) -> str:
    return (",".join(map(str, white)) + "/" + ",".join(map(str, black)) +
            f"/{kaz[0]},{kaz[1]}/{tuz[0]},{tuz[1]}/{side}")


def _apply_move_to_pos(pos: str, pit: int) -> str:
    white, black, kaz, tuz, side = _parse_pos(pos)
    pits = [white, black]
    opp = 1 - side
    if pit < 0 or pit >= NUM_PITS:
        raise ValueError(f"pit {pit} out of range")
    if pits[side][pit] == 0:
        raise ValueError(f"pit {pit} empty for side {side}")
    stones = pits[side][pit]
    pits[side][pit] = 0

    def _deposit(cur_side: int, cur_pit: int) -> None:
        if cur_side == opp and tuz[side] == cur_pit:
            kaz[side] += 1
        elif cur_side == side and tuz[opp] == cur_pit:
            kaz[opp] += 1
        else:
            pits[cur_side][cur_pit] += 1

    current_pit, current_side = pit, side
    if stones == 1:
        current_pit += 1
        if current_pit > 8:
            current_pit, current_side = 0, opp
        _deposit(current_side, current_pit)
    else:
        _deposit(current_side, current_pit)
        remaining = stones - 1
        while remaining > 0:
            current_pit += 1
            if current_pit > 8:
                current_pit, current_side = 0, 1 - current_side
            _deposit(current_side, current_pit)
            remaining -= 1

    is_tuzdyk_pit = (
        (current_side == opp and tuz[side] == current_pit)
        or (current_side == side and tuz[opp] == current_pit)
    )
    if current_side == opp and not is_tuzdyk_pit:
        count = pits[opp][current_pit]

        def _can_create_tuzdyk() -> bool:
            return tuz[side] == -1 and current_pit != 8 and tuz[opp] != current_pit

        if count == 3 and _can_create_tuzdyk():
            tuz[side] = current_pit
            kaz[side] += count
            pits[opp][current_pit] = 0
        elif count % 2 == 0 and count > 0:
            kaz[side] += count
            pits[opp][current_pit] = 0

    return _encode_pos(pits[0], pits[1], kaz, tuz, opp)


if __name__ == "__main__":
    e = Engine()
    e.start()
    pos = START_POSITION
    print("start:", pos)
    for ply in range(6):
        mv = e.bestmove(pos, time_ms=800)
        if isinstance(mv, tuple):
            print("terminal:", mv); break
        print(f"ply {ply}: side-to-move plays pit {mv} (hole #{mv+1})")
        pos = e.apply_move(pos, mv)
        print("  ->", pos)
    e.stop()
