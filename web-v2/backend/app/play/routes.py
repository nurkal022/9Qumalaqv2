import json
from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from app.deps import get_db, get_session, get_engine
from app.auth.session import CurrentSession
from app.engine.pool import EnginePool, EngineError
from app.errors import AppError
from app.play.schemas import NewGameReq, MoveReq, TakebackReq
from app.play.service import (
    create_game, apply_move, takeback, resign, undo_last_pair, LEVEL_TO_MS,
)
from app.play.snapshot import build_snapshot, HINTS_LIMIT
from app.db.models import Game, GameEvent
from app.main import limiter


router = APIRouter(prefix="/api/play", tags=["play"])

# Literal start position (from app/engine/protocol_notes.md)
START_POS = "9,9,9,9,9,9,9,9,9/9,9,9,9,9,9,9,9,9/0,0/-1,-1/0"

# In-memory hint counter, per game_id (per process). Resets on restart.
HINTS_USED: dict[int, int] = {}


def _move_str_to_int(move_uci: str) -> int:
    try:
        n = int(move_uci)
    except ValueError:
        raise AppError("illegal_move", 400, {"reason": "non-numeric move"})
    if n < 0 or n > 8:
        raise AppError("illegal_move", 400, {"reason": "move out of range"})
    return n


async def _refresh_rels(db: AsyncSession, g: Game) -> Game:
    """Eagerly reload relationship collections so build_snapshot never lazy-loads."""
    await db.refresh(g, ["moves", "events"])
    return g


async def _load_owned(db: AsyncSession, sess: CurrentSession, game_id: int) -> Game:
    g = await db.get(Game, game_id)
    if g is None:
        raise AppError("not_found", 404)
    if (sess.user and g.user_id != sess.user.id) or (sess.anon and g.anon_session_id != sess.anon.id):
        raise AppError("not_owner", 403)
    # Force-load relationships for snapshot building
    await _refresh_rels(db, g)
    return g


@router.post("/new")
@limiter.limit("30/hour")
async def new_game(
    request: Request, req: NewGameReq,
    sess: CurrentSession = Depends(get_session),
    db: AsyncSession = Depends(get_db),
    engine: EnginePool = Depends(get_engine),
):
    owner = {"user_id": sess.user.id} if sess.user else {"anon_session_id": sess.anon.id}
    g = await create_game(
        db, owner=owner, side=req.side, engine_level=req.engineLevel,
        clock=req.clock, use_book=req.useBook,
        start_fen=req.startFen or START_POS, engine_build=engine.build_hash,
    )
    await _refresh_rels(db, g)
    return {"game": build_snapshot(g)}


@router.get("/{game_id}")
async def get_game(
    game_id: int,
    sess: CurrentSession = Depends(get_session),
    db: AsyncSession = Depends(get_db),
):
    g = await _load_owned(db, sess, game_id)
    return {"game": build_snapshot(g, hints_used=HINTS_USED.get(g.id, 0))}


@router.post("/{game_id}/move")
@limiter.limit("60/minute")
async def make_move(
    request: Request, game_id: int, req: MoveReq,
    sess: CurrentSession = Depends(get_session),
    db: AsyncSession = Depends(get_db),
    engine: EnginePool = Depends(get_engine),
):
    g = await _load_owned(db, sess, game_id)
    if g.status != "active":
        raise AppError("game_not_active", 400)
    if g.side_to_move != g.side:
        raise AppError("validation_failed", 400, {"reason": "not your turn"})

    move_int = _move_str_to_int(req.moveUci)

    # 1) Compute the new position via Python rules port.
    try:
        new_pos = await engine.apply_move(position_pos=g.current_fen, move=move_int)
    except (EngineError, NotImplementedError) as e:
        raise AppError("engine_unavailable", 503) from e
    except (ValueError, IndexError) as e:
        # Python rules raised — illegal move (e.g., empty source pit).
        raise AppError("illegal_move", 400, {"reason": str(e)}) from e

    # 2) Round-trip validate: send the computed position to the engine.
    # If the engine rejects (returns "error"), the Python rules diverged from Rust.
    # In that case raise illegal_move so the client retries.
    try:
        await engine.push_position(new_pos)
    except (EngineError, RuntimeError) as e:
        raise AppError("illegal_move", 400, {"reason": "engine rejected position"}) from e

    # 3) Commit human move via the play service.
    await apply_move(db, game=g, move_uci=req.moveUci, actor="human", fen_after=new_pos)
    await _refresh_rels(db, g)

    # 4) If the engine is now to move and the game is still active, run the
    #    engine reply synchronously and commit it. This keeps the move flow
    #    self-contained in a single REST round-trip — simpler than splitting
    #    across REST + WS, and avoids the "second move = not your turn" trap
    #    where the engine never replies.
    if g.status == "active" and g.side_to_move != g.side:
        try:
            level_ms = LEVEL_TO_MS.get(g.engine_level, LEVEL_TO_MS["normal"])
            r = await engine.think(position_pos=g.current_fen, time_ms=level_ms)
        except EngineError:
            # Engine failed — game stays in active/engine-thinking state; user can retry
            # by reloading. Surface a 503 so the toast informs them.
            db.add(GameEvent(game_id=g.id, ply_at=g.current_ply, actor="system", type="engine_error"))
            await db.commit()
            raise AppError("engine_unavailable", 503)

        # Handle terminal results (the engine reports game-over instead of a move)
        if r.move == -1 or getattr(r, "terminal", None):
            from datetime import datetime, timezone
            terminal = getattr(r, "terminal", "unknown")
            g.status = "finished"
            g.result_reason = "rules_end"
            g.finished_at = datetime.now(timezone.utc)
            if terminal == "white_win":
                g.result = "win_white"
            elif terminal == "black_win":
                g.result = "win_black"
            else:
                g.result = "draw"
            await db.commit()
            await _refresh_rels(db, g)
            return {"game": build_snapshot(g, hints_used=HINTS_USED.get(g.id, 0))}

        # Engine returned a normal move — apply it via Python rules and commit
        try:
            engine_pos = await engine.apply_move(position_pos=g.current_fen, move=r.move)
        except (EngineError, NotImplementedError):
            raise AppError("engine_unavailable", 503)
        except (ValueError, IndexError):
            # Engine produced a move our Python rules can't apply. This is a
            # rule-divergence bug; surface as 503 so the user retries.
            raise AppError("engine_unavailable", 503)

        await apply_move(
            db, game=g, move_uci=str(r.move), actor="engine", fen_after=engine_pos,
            eval_cp=r.final_eval_cp, eval_depth=r.final_depth, think_time_ms=r.think_time_ms,
        )
        await _refresh_rels(db, g)

    return {"game": build_snapshot(g, hints_used=HINTS_USED.get(g.id, 0))}


@router.post("/{game_id}/undo")
async def undo(
    game_id: int,
    sess: CurrentSession = Depends(get_session),
    db: AsyncSession = Depends(get_db),
):
    g = await _load_owned(db, sess, game_id)
    await undo_last_pair(db, game=g)
    await _refresh_rels(db, g)
    return {"game": build_snapshot(g, hints_used=HINTS_USED.get(g.id, 0))}


@router.post("/{game_id}/takeback")
async def do_takeback(
    game_id: int, req: TakebackReq,
    sess: CurrentSession = Depends(get_session),
    db: AsyncSession = Depends(get_db),
):
    g = await _load_owned(db, sess, game_id)
    await takeback(db, game=g, to_ply=req.toPly)
    await _refresh_rels(db, g)
    return {"game": build_snapshot(g, hints_used=HINTS_USED.get(g.id, 0))}


@router.post("/{game_id}/resign")
async def do_resign(
    game_id: int,
    sess: CurrentSession = Depends(get_session),
    db: AsyncSession = Depends(get_db),
):
    g = await _load_owned(db, sess, game_id)
    await resign(db, game=g)
    await _refresh_rels(db, g)
    return {"game": build_snapshot(g, hints_used=HINTS_USED.get(g.id, 0))}


@router.post("/{game_id}/draw_offer")
async def draw_offer(
    game_id: int,
    sess: CurrentSession = Depends(get_session),
    db: AsyncSession = Depends(get_db),
    engine: EnginePool = Depends(get_engine),
):
    g = await _load_owned(db, sess, game_id)
    if g.status != "active":
        raise AppError("game_not_active", 400)
    accepted = False
    try:
        result = await engine.think(position_pos=g.current_fen, time_ms=200)
        if result.final_eval_cp is not None and abs(result.final_eval_cp) < 50:
            from datetime import datetime, timezone
            g.status = "finished"
            g.result = "draw"
            g.result_reason = "draw_agreement"
            g.finished_at = datetime.now(timezone.utc)
            db.add(GameEvent(game_id=g.id, ply_at=g.current_ply, actor="engine", type="draw_accept"))
            await db.commit()
            await _refresh_rels(db, g)
            accepted = True
        else:
            db.add(GameEvent(game_id=g.id, ply_at=g.current_ply, actor="engine", type="draw_decline"))
            await db.commit()
            await _refresh_rels(db, g)
    except EngineError as e:
        raise AppError("engine_unavailable", 503) from e
    return {"game": build_snapshot(g), "accepted": accepted}


@router.post("/{game_id}/hint")
async def hint(
    game_id: int,
    sess: CurrentSession = Depends(get_session),
    db: AsyncSession = Depends(get_db),
    engine: EnginePool = Depends(get_engine),
):
    g = await _load_owned(db, sess, game_id)
    used = HINTS_USED.get(g.id, 0)
    if used >= HINTS_LIMIT:
        raise AppError("rate_limited", 429, {"reason": "hint limit"})
    try:
        r = await engine.think(position_pos=g.current_fen, time_ms=2000)
    except EngineError as e:
        raise AppError("engine_unavailable", 503) from e
    HINTS_USED[g.id] = used + 1
    db.add(GameEvent(
        game_id=g.id, ply_at=g.current_ply, actor="system",
        type="hint_used",
        payload_json=json.dumps({"move": r.move, "evalCp": r.final_eval_cp}),
    ))
    await db.commit()
    return {"move": str(r.move), "evalCp": r.final_eval_cp, "depth": r.final_depth, "pv": []}
