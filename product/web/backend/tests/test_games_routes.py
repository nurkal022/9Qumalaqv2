async def test_list_games_paginates(client):
    for _ in range(3):
        await client.post("/api/play/new", json={"side": 0, "engineLevel": "normal", "clock": None, "useBook": False})
    r = await client.get("/api/games?page=1&pageSize=2")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == 3 and len(body["items"]) == 2
    assert body["page"] == 1 and body["pageSize"] == 2


async def test_list_games_empty_for_fresh_user(client):
    r = await client.get("/api/games")
    assert r.json()["total"] == 0


async def test_get_game_returns_full_snapshot(client):
    r = await client.post("/api/play/new", json={"side": 0, "engineLevel": "normal", "clock": None, "useBook": False})
    gid = r.json()["game"]["id"]
    r2 = await client.get(f"/api/games/{gid}")
    assert r2.status_code == 200
    g = r2.json()["game"]
    assert g["id"] == gid


async def test_delete_active_game_rejected(client):
    r = await client.post("/api/play/new", json={"side": 0, "engineLevel": "normal", "clock": None, "useBook": False})
    gid = r.json()["game"]["id"]
    r2 = await client.delete(f"/api/games/{gid}")
    assert r2.status_code == 400


async def test_delete_finished_game_succeeds(client):
    r = await client.post("/api/play/new", json={"side": 0, "engineLevel": "normal", "clock": None, "useBook": False})
    gid = r.json()["game"]["id"]
    await client.post(f"/api/play/{gid}/resign")
    r2 = await client.delete(f"/api/games/{gid}")
    assert r2.status_code == 204


async def test_status_filter(client):
    # one active, one finished
    r1 = await client.post("/api/play/new", json={"side": 0, "engineLevel": "normal", "clock": None, "useBook": False})
    r2 = await client.post("/api/play/new", json={"side": 0, "engineLevel": "normal", "clock": None, "useBook": False})
    g2id = r2.json()["game"]["id"]
    await client.post(f"/api/play/{g2id}/resign")
    finished = await client.get("/api/games?status=finished")
    assert finished.json()["total"] == 1
