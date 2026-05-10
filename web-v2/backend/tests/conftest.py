import os
import pathlib

# MUST happen before any app.* import — sets env so pydantic-settings picks up the override
_TEST_DB = pathlib.Path(__file__).parent / "test_data.db"
_TEST_DB.unlink(missing_ok=True)
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_TEST_DB.resolve()}"

import pytest
from httpx import ASGITransport, AsyncClient
from app.main import create_app
from app.db.base import Base, engine


@pytest.fixture(autouse=True)
async def _reset_schema():
    """Drop+recreate all tables before each test for isolation."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield


@pytest.fixture
async def client():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
