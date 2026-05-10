import json
from datetime import datetime, timezone
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import Game, Move, GameEvent
from app.errors import AppError
from app.play.clock import apply_move_to_clock


LEVEL_TO_MS = {"easy": 500, "normal": 2000, "hard": 8000}


async def create_game(s: AsyncSession, *, owner: dict, side: int, engine_level: str,
                      clock: dict | None, use_book: bool, start_fen: str, engine_build: str) -> Game:
    initial = clock["initialMs"] if clock else 0
    increment = clock["incrementMs"] if clock else 0
    g = Game(
        **owner,
        mode="solo", side=side, opponent_kind="engine", opponent_ref=engine_build,
        engine_level=engine_level,
        clock_initial_ms=initial, clock_increment_ms=increment,
        clock_white_ms=initial, clock_black_ms=initial,
        last_clock_at=datetime.now(timezone.utc) if initial else None,
        start_fen=start_fen, current_fen=start_fen,
        current_ply=0, side_to_move=0, status="active",
    )
    s.add(g)
    await s.commit()
    await s.refresh(g)
    return g


async def apply_move(s: AsyncSession, *, game: Game, move_uci: str, actor: str,
                     fen_after: str, eval_cp: int | None = None, eval_depth: int | None = None,
                     pv: list[str] | None = None, think_time_ms: int | None = None) -> Move:
    if game.status != "active":
        raise AppError("game_not_active", 400)
    now = datetime.now(timezone.utc)
    new_w, new_b = apply_move_to_clock(game, now)
    side = game.side_to_move
    clock_after = new_w if side == 0 else new_b
    move = Move(
        game_id=game.id, ply=game.current_ply + 1, side=side, actor=actor,
        move_uci=move_uci, fen_after=fen_after,
        eval_cp=eval_cp, eval_depth=eval_depth,
        pv=json.dumps(pv) if pv else None,
        think_time_ms=think_time_ms, clock_after_ms=clock_after,
    )
    s.add(move)
    game.current_ply += 1
    game.current_fen = fen_after
    game.side_to_move = 1 - side
    game.clock_white_ms, game.clock_black_ms = new_w, new_b
    game.last_clock_at = now
    await s.commit()
    await s.refresh(game)
    return move


async def takeback(s: AsyncSession, *, game: Game, to_ply: int) -> None:
    if to_ply >= game.current_ply or to_ply < 0:
        raise AppError("validation_failed", 400, {"reason": "to_ply out of range"})
    await s.execute(delete(Move).where(Move.game_id == game.id, Move.ply > to_ply))
    if to_ply == 0:
        game.current_fen = game.start_fen
        game.side_to_move = 0
    else:
        last = (await s.execute(select(Move).where(Move.game_id == game.id, Move.ply == to_ply))).scalar_one()
        game.current_fen = last.fen_after
        game.side_to_move = 1 - last.side
    game.current_ply = to_ply
    s.add(GameEvent(
        game_id=game.id, ply_at=game.current_ply, actor="human",
        type="takeback", payload_json=json.dumps({"toPly": to_ply}),
    ))
    await s.commit()
    await s.refresh(game)


async def resign(s: AsyncSession, *, game: Game) -> None:
    if game.status != "active":
        raise AppError("game_not_active", 400)
    game.status = "finished"
    game.result = "win_black" if game.side == 0 else "win_white"
    game.result_reason = "resign"
    game.finished_at = datetime.now(timezone.utc)
    s.add(GameEvent(game_id=game.id, ply_at=game.current_ply, actor="human", type="resign"))
    await s.commit()
    await s.refresh(game)


async def undo_last_pair(s: AsyncSession, *, game: Game) -> None:
    """Roll back the most recent human move and the engine's reply (if any)."""
    if game.current_ply == 0:
        return
    target = game.current_ply
    while target > 0:
        m = (await s.execute(select(Move).where(Move.game_id == game.id, Move.ply == target))).scalar_one()
        target -= 1
        if m.actor == "human":
            break
    await takeback(s, game=game, to_ply=target)
