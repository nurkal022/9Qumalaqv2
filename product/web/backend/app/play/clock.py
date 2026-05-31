from datetime import datetime, timezone
from dataclasses import dataclass
from app.db.models import Game


@dataclass
class ClockState:
    initial_ms: int
    increment_ms: int
    white_ms: int
    black_ms: int
    running_side: int | None  # 0|1|None


def project_clock(game: Game, now: datetime | None = None) -> ClockState:
    if game.clock_initial_ms == 0:
        return ClockState(0, 0, 0, 0, None)
    now = now or datetime.now(timezone.utc)
    elapsed_ms = 0
    if game.last_clock_at and game.status == "active":
        last = game.last_clock_at
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        delta = now - last
        elapsed_ms = int(delta.total_seconds() * 1000)
    w, b = game.clock_white_ms, game.clock_black_ms
    if game.status == "active":
        if game.side_to_move == 0:
            w = max(0, w - elapsed_ms)
        else:
            b = max(0, b - elapsed_ms)
    return ClockState(
        initial_ms=game.clock_initial_ms,
        increment_ms=game.clock_increment_ms,
        white_ms=w, black_ms=b,
        running_side=game.side_to_move if game.status == "active" else None,
    )


def apply_move_to_clock(game: Game, now: datetime) -> tuple[int, int]:
    """After a move is committed: subtract elapsed, add increment to the side that just moved.
    Returns (new_white_ms, new_black_ms). Caller updates game.last_clock_at to `now`.
    """
    if game.clock_initial_ms == 0:
        return game.clock_white_ms, game.clock_black_ms
    proj = project_clock(game, now)
    side_just_moved = game.side_to_move  # the one whose clock was running
    w, b = proj.white_ms, proj.black_ms
    if side_just_moved == 0:
        w = w + game.clock_increment_ms
    else:
        b = b + game.clock_increment_ms
    return w, b
