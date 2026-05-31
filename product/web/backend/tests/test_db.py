from sqlalchemy import text
from app.db.base import SessionLocal


async def test_db_connection_works():
    async with SessionLocal() as s:
        result = await s.execute(text("SELECT 1"))
        assert result.scalar() == 1
