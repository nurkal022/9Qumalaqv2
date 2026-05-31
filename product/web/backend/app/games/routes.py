from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from app.deps import get_db, get_session
from app.auth.session import CurrentSession
from app.errors import AppError
from app.db.models import Game, Move
from app.play.snapshot import build_snapshot
from app.games.schemas import GameSummary


router = APIRouter(prefix="/api/games", tags=["games"])


@router.get("")
async def list_games(
    page: int = 1, pageSize: int = 20, status: str | None = None,
    sess: CurrentSession = Depends(get_session),
    db: AsyncSession = Depends(get_db),
):
    q = select(Game)
    if sess.user:
        q = q.where(Game.user_id == sess.user.id)
    elif sess.anon:
        q = q.where(Game.anon_session_id == sess.anon.id)
    if status:
        q = q.where(Game.status == status)

    total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar() or 0
    rows = (await db.execute(
        q.order_by(Game.started_at.desc()).limit(pageSize).offset((page - 1) * pageSize)
    )).scalars().all()

    items = []
    for g in rows:
        mc = (await db.execute(
            select(func.count(Move.id)).where(Move.game_id == g.id)
        )).scalar() or 0
        dur = int((g.finished_at - g.started_at).total_seconds() * 1000) if g.finished_at else None
        items.append(GameSummary(
            id=g.id,
            mode=g.mode,
            opponentLabel=f"NNUE {g.opponent_ref or ''}".strip(),
            result=g.result,
            finalScore=g.final_score,
            side=g.side,
            moveCount=mc,
            startedAt=g.started_at.isoformat(),
            finishedAt=g.finished_at.isoformat() if g.finished_at else None,
            durationMs=dur,
        ))
    return {"items": items, "total": total, "page": page, "pageSize": pageSize}


@router.get("/{game_id}")
async def get_game(
    game_id: int,
    sess: CurrentSession = Depends(get_session),
    db: AsyncSession = Depends(get_db),
):
    g = await db.get(Game, game_id)
    if g is None:
        raise AppError("not_found", 404)
    if (sess.user and g.user_id != sess.user.id) or (sess.anon and g.anon_session_id != sess.anon.id):
        raise AppError("not_owner", 403)
    await db.refresh(g, ["moves", "events"])
    return {"game": build_snapshot(g)}


@router.delete("/{game_id}", status_code=204)
async def delete_game(
    game_id: int,
    sess: CurrentSession = Depends(get_session),
    db: AsyncSession = Depends(get_db),
):
    g = await db.get(Game, game_id)
    if g is None:
        raise AppError("not_found", 404)
    if (sess.user and g.user_id != sess.user.id) or (sess.anon and g.anon_session_id != sess.anon.id):
        raise AppError("not_owner", 403)
    if g.status == "active":
        raise AppError("game_not_active", 400, {"reason": "cannot delete active game"})
    await db.delete(g)
    await db.commit()
