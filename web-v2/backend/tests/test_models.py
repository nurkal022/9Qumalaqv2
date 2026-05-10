from app.db.base import SessionLocal
from app.db.models import User, AnonSession


async def test_user_can_persist():
    async with SessionLocal() as s:
        s.add(User(username="alice", password_hash="x", locale="kk"))
        await s.commit()
        users = (await s.execute(__import__("sqlalchemy").select(User).where(User.username == "alice"))).scalars().all()
        assert len(users) == 1


async def test_anon_session_can_persist():
    async with SessionLocal() as s:
        import uuid
        sid = str(uuid.uuid4())
        s.add(AnonSession(id=sid))
        await s.commit()
