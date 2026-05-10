async def test_register_creates_user_and_sets_jwt(client):
    r = await client.post("/api/auth/register", json={"username": "alice", "password": "secret123"})
    assert r.status_code == 200
    assert r.json()["user"]["username"] == "alice"
    assert "auth_token=" in r.headers["set-cookie"]


async def test_login_invalid_credentials(client):
    await client.post("/api/auth/register", json={"username": "bob", "password": "secret123"})
    r = await client.post("/api/auth/login", json={"username": "bob", "password": "wrong"})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "invalid_credentials"


async def test_me_returns_anon_for_fresh_client(client):
    r = await client.get("/api/auth/me")
    assert r.json()["kind"] == "anon"


async def test_username_taken(client):
    await client.post("/api/auth/register", json={"username": "carol", "password": "secret123"})
    r = await client.post("/api/auth/register", json={"username": "carol", "password": "secret123"})
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "username_taken"
