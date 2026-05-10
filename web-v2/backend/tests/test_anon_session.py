import pytest
from httpx import ASGITransport, AsyncClient
from app.main import create_app


@pytest.fixture
async def client():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def test_first_request_sets_anon_cookie(client):
    r = await client.get("/api/health")
    assert r.status_code == 200
    set_cookie = r.headers.get("set-cookie", "")
    assert "anon_session=" in set_cookie


async def test_second_request_reuses_anon_cookie(client):
    r1 = await client.get("/api/health")
    cookies = r1.cookies
    r2 = await client.get("/api/health", cookies=cookies)
    assert "anon_session=" not in r2.headers.get("set-cookie", "")
