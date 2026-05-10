"""WebSocket endpoint tests.

Uses starlette.testclient.TestClient (sync WebSocket support) called from
async test functions so that the conftest _reset_schema autouse fixture
(which is async) runs before each test to set up a clean DB.
"""
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from app.main import create_app


def test_ws_sends_snapshot_on_connect(_reset_schema):
    app = create_app()
    with TestClient(app) as c:
        r = c.post(
            "/api/play/new",
            json={"side": 0, "engineLevel": "normal", "clock": None, "useBook": False},
        )
        gid = r.json()["game"]["id"]
        with c.websocket_connect(f"/ws/games/{gid}") as ws:
            ws.send_json({"type": "hello"})
            msg = ws.receive_json()
            assert msg["type"] == "snapshot"
            assert msg["game"]["id"] == gid


def test_ws_foreign_game_closed_4003(_reset_schema):
    app = create_app()
    with TestClient(app) as c:
        r = c.post(
            "/api/play/new",
            json={"side": 0, "engineLevel": "normal", "clock": None, "useBook": False},
        )
        gid = r.json()["game"]["id"]

    # Fresh client = fresh anon = different owner
    app2 = create_app()
    with TestClient(app2) as c2:
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with c2.websocket_connect(f"/ws/games/{gid}") as ws:
                ws.receive_json()
        assert exc_info.value.code == 4003


def test_ws_pong_on_ping(_reset_schema):
    app = create_app()
    with TestClient(app) as c:
        r = c.post(
            "/api/play/new",
            json={"side": 0, "engineLevel": "normal", "clock": None, "useBook": False},
        )
        gid = r.json()["game"]["id"]
        with c.websocket_connect(f"/ws/games/{gid}") as ws:
            ws.send_json({"type": "hello"})
            _snap = ws.receive_json()
            ws.send_json({"type": "ping"})
            msg = ws.receive_json()
            assert msg["type"] == "pong"
