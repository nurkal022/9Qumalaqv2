from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession
from app.auth.schemas import RegisterReq, LoginReq, UserOut, MeOut
from app.auth.session import CurrentSession
from app.auth.jwt import encode_user
from app.auth.anonymous import COOKIE_JWT
from app.auth.service import register as svc_register, login as svc_login
from app.deps import get_db, get_session
from app.config import settings

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _set_jwt(response: Response, user_id: int) -> None:
    response.set_cookie(
        COOKIE_JWT, encode_user(user_id),
        max_age=settings.jwt_ttl_days * 24 * 3600,
        httponly=True, samesite="lax", secure=settings.env == "prod",
    )


@router.post("/register")
async def register(req: RegisterReq, response: Response, sess: CurrentSession = Depends(get_session), db: AsyncSession = Depends(get_db)):
    user, migrated = await svc_register(
        db, username=req.username, password=req.password, locale=req.locale,
        anon_session_id=sess.anon.id if sess.anon else None,
    )
    _set_jwt(response, user.id)
    return {"user": UserOut.model_validate(user, from_attributes=True), "migratedGamesCount": migrated}


@router.post("/login")
async def login(req: LoginReq, response: Response, db: AsyncSession = Depends(get_db)):
    user = await svc_login(db, username=req.username, password=req.password)
    _set_jwt(response, user.id)
    return {"user": UserOut.model_validate(user, from_attributes=True)}


@router.post("/logout", status_code=204)
async def logout(response: Response):
    response.delete_cookie(COOKIE_JWT)


@router.get("/me", response_model=MeOut)
async def me(sess: CurrentSession = Depends(get_session)):
    if sess.user:
        return MeOut(kind="user", user=UserOut.model_validate(sess.user, from_attributes=True))
    return MeOut(kind="anon", anonId=sess.anon.id)
