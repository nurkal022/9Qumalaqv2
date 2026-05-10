import uuid
from datetime import datetime, timezone
from fastapi import Request
from sqlalchemy import update
from app.config import settings
from app.db.base import SessionLocal
from app.db.models import AnonSession, User
from app.auth.jwt import decode_user
from app.auth.session import CurrentSession


COOKIE_ANON = "anon_session"
COOKIE_JWT = "auth_token"


async def resolve_session(request: Request) -> CurrentSession:
    """Read cookies. Return (user, anon) — exactly one is non-None.
    If neither present, create new anon session and stash it on request.state for the
    response middleware to set the cookie.
    """
    token = request.cookies.get(COOKIE_JWT)
    if token:
        uid = decode_user(token)
        if uid is not None:
            async with SessionLocal() as s:
                u = await s.get(User, uid)
                if u is not None:
                    return CurrentSession(user=u, anon=None)

    anon_id = request.cookies.get(COOKIE_ANON)
    async with SessionLocal() as s:
        anon: AnonSession | None = None
        if anon_id:
            anon = await s.get(AnonSession, anon_id)
        if anon is None:
            anon = AnonSession(id=str(uuid.uuid4()))
            s.add(anon)
            await s.commit()
            await s.refresh(anon)
            request.state.new_anon_id = anon.id
        else:
            await s.execute(
                update(AnonSession).where(AnonSession.id == anon.id).values(last_seen_at=datetime.now(timezone.utc))
            )
            await s.commit()
        return CurrentSession(user=None, anon=anon)


def attach_anon_cookie_if_new(request: Request, response) -> None:
    new_id = getattr(request.state, "new_anon_id", None)
    if new_id:
        response.set_cookie(
            COOKIE_ANON, new_id,
            max_age=settings.anon_cookie_ttl_days * 24 * 3600,
            httponly=True, samesite="lax",
            secure=settings.env == "prod",
        )
