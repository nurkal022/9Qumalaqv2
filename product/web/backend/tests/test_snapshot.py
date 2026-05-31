from datetime import datetime, timezone
from app.db.models import Game
from app.play.snapshot import build_snapshot


def test_snapshot_basic_shape():
    g = Game(
        id=1, mode="solo", side=0, opponent_kind="engine",
        clock_initial_ms=0, clock_increment_ms=0, clock_white_ms=0, clock_black_ms=0,
        start_fen="x", current_fen="x", current_ply=0, side_to_move=0,
        status="active", started_at=datetime(2026,1,1,tzinfo=timezone.utc),
    )
    g.moves = []; g.events = []
    s = build_snapshot(g)
    assert s.id == 1 and s.engineThinking is False
