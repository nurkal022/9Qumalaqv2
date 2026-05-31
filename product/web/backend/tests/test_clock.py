from datetime import datetime, timedelta, timezone
from app.db.models import Game
from app.play.clock import project_clock, apply_move_to_clock


def _g(**over):
    base = dict(
        clock_initial_ms=300_000, clock_increment_ms=2000,
        clock_white_ms=300_000, clock_black_ms=300_000,
        last_clock_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        side_to_move=0, status="active",
    )
    base.update(over)
    return Game(**{k: v for k, v in base.items() if k in Game.__table__.columns.keys()})


def test_clock_zero_means_no_clock():
    g = _g(clock_initial_ms=0, clock_white_ms=0, clock_black_ms=0)
    s = project_clock(g)
    assert s.running_side is None and s.white_ms == 0


def test_clock_decrements_running_side_only():
    g = _g()
    now = g.last_clock_at + timedelta(seconds=10)
    s = project_clock(g, now=now)
    assert s.white_ms == 290_000
    assert s.black_ms == 300_000
    assert s.running_side == 0


def test_increment_added_after_move():
    g = _g()
    now = g.last_clock_at + timedelta(seconds=10)
    w, b = apply_move_to_clock(g, now)
    assert w == 290_000 + 2000
    assert b == 300_000


def test_clock_does_not_go_negative():
    g = _g(clock_white_ms=5000)
    now = g.last_clock_at + timedelta(seconds=999)
    s = project_clock(g, now=now)
    assert s.white_ms == 0
