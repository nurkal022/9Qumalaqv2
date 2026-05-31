import json
from app.db.models import Game
from app.play.clock import project_clock
from app.play.schemas import GameStateOut, ClockOut, MoveOut, EventOut


HINTS_LIMIT = 3


def build_snapshot(game: Game, *, hints_used: int = 0, engine_thinking: bool = False) -> GameStateOut:
    clock = project_clock(game)
    moves = [MoveOut(
        ply=m.ply, side=m.side, actor=m.actor, moveUci=m.move_uci, fenAfter=m.fen_after,
        evalCp=m.eval_cp, evalDepth=m.eval_depth, thinkTimeMs=m.think_time_ms, clockAfterMs=m.clock_after_ms,
    ) for m in game.moves]
    events = [EventOut(
        plyAt=e.ply_at, actor=e.actor, type=e.type,
        payload=json.loads(e.payload_json) if e.payload_json else None,
    ) for e in game.events]
    return GameStateOut(
        id=game.id, mode=game.mode, side=game.side, status=game.status,
        result=game.result, resultReason=game.result_reason, finalScore=game.final_score,
        startFen=game.start_fen, currentFen=game.current_fen,
        currentPly=game.current_ply, sideToMove=game.side_to_move,
        clock=ClockOut(initialMs=clock.initial_ms, incrementMs=clock.increment_ms,
                       whiteMs=clock.white_ms, blackMs=clock.black_ms, runningSide=clock.running_side),
        engineThinking=engine_thinking,
        hintsUsed=hints_used, hintsLimit=HINTS_LIMIT,
        moves=moves, events=events,
        startedAt=game.started_at.isoformat(),
        finishedAt=game.finished_at.isoformat() if game.finished_at else None,
    )
