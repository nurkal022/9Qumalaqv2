from app.db.base import SessionLocal
from app.db.models import Game


async def test_anon_game_migrates_on_register(client):
    # Anon user
    await client.get("/api/auth/me")  # ensures anon cookie set
    cookies = client.cookies
    anon_id = cookies.get("anon_session")
    # Manually create a game for the anon session
    async with SessionLocal() as s:
        s.add(Game(anon_session_id=anon_id, mode="solo", side=0, opponent_kind="engine",
                   clock_initial_ms=0, clock_increment_ms=0, clock_white_ms=0, clock_black_ms=0,
                   start_fen="x", current_fen="x", status="active"))
        await s.commit()

    # Register transfers ownership
    r = await client.post("/api/auth/register", json={"username": "dan", "password": "secret123"})
    assert r.json()["migratedGamesCount"] == 1


async def test_anon_game_migrates_on_login(client, client_factory=None):
    # Need fresh client for login flow (already-registered user logging in from new anon session)
    # Step 1: register a user (uses one anon session)
    r1 = await client.post("/api/auth/register", json={"username": "eve", "password": "secret123"})
    assert r1.status_code == 200

    # Step 2: clear cookies (simulate fresh browser session) and create a game tied to a new anon
    client.cookies.clear()
    await client.get("/api/auth/me")  # acquires fresh anon cookie
    new_anon_id = client.cookies.get("anon_session")

    async with SessionLocal() as s:
        s.add(Game(anon_session_id=new_anon_id, mode="solo", side=0, opponent_kind="engine",
                   clock_initial_ms=0, clock_increment_ms=0, clock_white_ms=0, clock_black_ms=0,
                   start_fen="x", current_fen="x", status="active"))
        await s.commit()

    # Step 3: login as the existing user — should migrate the new anon's game
    r2 = await client.post("/api/auth/login", json={"username": "eve", "password": "secret123"})
    assert r2.status_code == 200
    assert r2.json()["migratedGamesCount"] == 1
