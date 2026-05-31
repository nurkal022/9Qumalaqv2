import pytest


async def test_new_game_creates_active_game(client):
    r = await client.post(
        "/api/play/new",
        json={"side": 0, "engineLevel": "normal", "clock": None, "useBook": False},
    )
    assert r.status_code == 200, r.text
    g = r.json()["game"]
    assert g["status"] == "active" and g["currentPly"] == 0


async def test_get_returns_owned_game(client):
    r = await client.post("/api/play/new", json={"side": 0, "engineLevel": "normal", "clock": None, "useBook": False})
    gid = r.json()["game"]["id"]
    r2 = await client.get(f"/api/play/{gid}")
    assert r2.status_code == 200


async def test_make_move_increments_ply(client):
    """Real engine round-trip — requires the engine binary to be built.
    The /move endpoint now also runs the engine reply synchronously, so after
    one human move we expect ply=2 (human ply 1 + engine ply 2)."""
    from pathlib import Path
    from app.config import settings
    if not Path(settings.engine_path).exists():
        pytest.skip("engine binary not built")
    r = await client.post("/api/play/new", json={"side": 0, "engineLevel": "test", "clock": None, "useBook": False})
    gid = r.json()["game"]["id"]
    # Move "0" = pit 1 (first pit). At start every pit has 9 stones, so "0" is a legal first move.
    r2 = await client.post(f"/api/play/{gid}/move", json={"moveUci": "0"})
    assert r2.status_code == 200, r2.text
    g = r2.json()["game"]
    # Human ply 1 + engine reply ply 2 = currentPly == 2 (game is back to player's turn)
    assert g["currentPly"] == 2
    assert g["sideToMove"] == 0  # back to player
    assert g["moves"][0]["actor"] == "human"
    assert g["moves"][1]["actor"] == "engine"


async def test_resign_finalizes(client):
    r = await client.post("/api/play/new", json={"side": 0, "engineLevel": "normal", "clock": None, "useBook": False})
    gid = r.json()["game"]["id"]
    r2 = await client.post(f"/api/play/{gid}/resign")
    g = r2.json()["game"]
    assert g["status"] == "finished" and g["result"] == "win_black"


async def test_foreign_game_forbidden(client):
    """Same client makes the game then we use a fresh AsyncClient to attempt access."""
    from httpx import ASGITransport, AsyncClient
    from app.main import create_app
    r = await client.post("/api/play/new", json={"side": 0, "engineLevel": "normal", "clock": None, "useBook": False})
    gid = r.json()["game"]["id"]
    # Fresh client = fresh anon session = different owner
    async with AsyncClient(transport=ASGITransport(app=create_app()), base_url="http://test") as other:
        r2 = await other.get(f"/api/play/{gid}")
        assert r2.status_code == 403


async def test_illegal_move_returned_for_invalid_pit(client):
    """Empty pit → Python rules raise → illegal_move."""
    from pathlib import Path
    from app.config import settings
    if not Path(settings.engine_path).exists():
        pytest.skip("engine binary not built")
    r = await client.post("/api/play/new", json={"side": 0, "engineLevel": "normal", "clock": None, "useBook": False})
    gid = r.json()["game"]["id"]
    # Move 99 is out of range → illegal
    r2 = await client.post(f"/api/play/{gid}/move", json={"moveUci": "99"})
    assert r2.status_code == 400
    assert r2.json()["error"]["code"] == "illegal_move"
