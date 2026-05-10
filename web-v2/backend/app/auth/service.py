import bcrypt
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import User, Game
from app.errors import AppError


def hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt(rounds=12)).decode()


def verify_password(pw: str, hashed: str) -> bool:
    return bcrypt.checkpw(pw.encode(), hashed.encode())


async def register(s: AsyncSession, *, username: str, password: str, locale: str, anon_session_id: str | None) -> tuple[User, int]:
    existing = (await s.execute(select(User).where(User.username == username))).scalar_one_or_none()
    if existing:
        raise AppError("username_taken", 409)
    user = User(username=username, password_hash=hash_password(password), locale=locale)
    s.add(user)
    await s.flush()
    migrated = 0
    if anon_session_id:
        result = await s.execute(
            update(Game)
            .where(Game.anon_session_id == anon_session_id)
            .values(user_id=user.id, anon_session_id=None)
        )
        migrated = result.rowcount or 0
    await s.commit()
    await s.refresh(user)
    return user, migrated


async def login(s: AsyncSession, *, username: str, password: str) -> User:
    user = (await s.execute(select(User).where(User.username == username))).scalar_one_or_none()
    if user is None or not verify_password(password, user.password_hash):
        raise AppError("invalid_credentials", 401)
    return user
