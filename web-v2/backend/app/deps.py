from typing import AsyncIterator
from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.base import SessionLocal
from app.auth.session import CurrentSession


async def get_db() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session


def get_session(request: Request) -> CurrentSession:
    return request.state.session
