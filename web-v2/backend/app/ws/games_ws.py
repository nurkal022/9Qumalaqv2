"""WebSocket endpoint for per-game state push.

Protocol (server → client):
  snapshot        — full game state (on connect or on request_snapshot)
  move_applied    — after human move committed
  engine_thinking — {started: bool}
  engine_move     — after engine commits its reply
  event           — game event (e.g. engine_error)
  game_finished   — when status transitions to "finished"
  pong            — reply to ping
  error           — WS-level error before close

Protocol (client → server):
  hello           — {type:"hello", lastSeenSeq?:int}  (first frame, 5s timeout)
  ping            — keepalive
  request_snapshot — re-send current snapshot
  move            — {type:"move", moveUci:"3"}
  cancel_thinking — no-op (engine doesn't stream, can't cancel mid-think)

Resume mechanic:
  Per-game ring buffer of last 200 server messages, each stamped with seq.
  On connect, if client sends lastSeenSeq and that seq is still in the buffer,
  replay messages with seq > lastSeenSeq. Otherwise send fresh snapshot.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict, deque
from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.db.base import SessionLocal
from app.db.models import Game, GameEvent
from app.engine.pool import EngineError, EnginePool
from app.play.service import LEVEL_TO_MS, apply_move
from app.play.snapshot import build_snapshot

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ws"])

# ---------------------------------------------------------------------------
# Per-game in-memory state (lost on restart)
# ---------------------------------------------------------------------------
_seq: dict[int, int] = defaultdict(int)
_buf: dict[int, deque[dict]] = defaultdict(lambda: deque(maxlen=200))
_clients: dict[int, set[asyncio.Queue]] = defaultdict(set)


def _publish(game_id: int, msg: dict) -> None:
    """Stamp msg with seq, append to ring buffer, fan out to all client queues."""
    seq = _seq[game_id]
    _seq[game_id] = seq + 1
    msg = {"seq": seq, **msg}
    _buf[game_id].append(msg)
    for q in list(_clients[game_id]):
        if q.full():
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            pass  # already drained one slot above; if still full, drop


def _snapshot_msg(game: Game, hints_used: int = 0, engine_thinking: bool = False) -> dict:
    snap = build_snapshot(game, hints_used=hints_used, engine_thinking=engine_thinking)
    return {"type": "snapshot", "game": snap.model_dump()}


# ---------------------------------------------------------------------------
# Auth helpers (WebSocket-compatible)
# ---------------------------------------------------------------------------

class _WsCookies:
    """Minimal Request-like wrapper so resolve_session works on a WebSocket."""
    def __init__(self, ws: WebSocket) -> None:
        self.cookies = dict(ws.cookies)
        self.state = _State()


class _State:
    pass


async def _resolve_ws_session(ws: WebSocket):
    """Return CurrentSession by adapting the WebSocket into a request-like object."""
    from app.auth.anonymous import resolve_session
    fake_req = _WsCookies(ws)
    return await resolve_session(fake_req)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Move validation / engine interaction
# ---------------------------------------------------------------------------

def _parse_move_uci(move_uci: str) -> int:
    try:
        n = int(move_uci)
    except ValueError:
        raise ValueError("non-numeric move")
    if n < 0 or n > 8:
        raise ValueError("move out of range 0-8")
    return n


async def _load_game_in_session(s, game_id: int) -> Game | None:
    g = await s.get(Game, game_id)
    if g is None:
        return None
    await s.refresh(g, ["moves", "events"])
    return g


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------

@router.websocket("/ws/games/{game_id}")
async def games_ws(ws: WebSocket, game_id: int) -> None:
    # 1. Accept early so we can send close codes.
    await ws.accept()

    # 2. Validate ownership.
    try:
        sess = await _resolve_ws_session(ws)
    except Exception:
        await ws.close(code=4003)
        return

    async with SessionLocal() as s:
        g = await s.get(Game, game_id)
        if g is None:
            await ws.close(code=4004)
            return
        if (sess.user and g.user_id != sess.user.id) or \
           (sess.anon and g.anon_session_id != sess.anon.id):
            await ws.close(code=4003)
            return

    # 3. Read first frame: hello (5s timeout).
    last_seen_seq: int | None = None
    try:
        raw = await asyncio.wait_for(ws.receive_text(), timeout=5.0)
        frame = json.loads(raw)
        if frame.get("type") == "hello":
            last_seen_seq = frame.get("lastSeenSeq")
    except (asyncio.TimeoutError, Exception):
        # Timeout or bad frame — treat as missing hello, proceed with snapshot.
        last_seen_seq = None

    # 4. Determine what to send on connect: replay or fresh snapshot.
    client_q: asyncio.Queue = asyncio.Queue(maxsize=64)
    _clients[game_id].add(client_q)

    try:
        buf = _buf[game_id]
        replayed = False
        if last_seen_seq is not None and buf:
            # Check if the oldest buffered seq is <= last_seen_seq + 1
            # (i.e., the client can resume from the buffer).
            oldest_seq = buf[0]["seq"]
            if oldest_seq <= last_seen_seq + 1:
                # Replay all messages with seq > last_seen_seq
                for msg in buf:
                    if msg["seq"] > last_seen_seq:
                        await ws.send_json(msg)
                replayed = True

        if not replayed:
            # Send fresh snapshot. Wrap the actual send in a guard — if the client
            # disconnected between accept and now, the WS is in DISCONNECTED state
            # and send_json would crash uvicorn's ASGI dispatch.
            from starlette.websockets import WebSocketState
            async with SessionLocal() as s:
                g = await _load_game_in_session(s, game_id)
                if g is None:
                    await ws.close(code=4004)
                    return
                snap_msg = _snapshot_msg(g)
            _publish(game_id, snap_msg)
            try:
                queued = client_q.get_nowait()
            except asyncio.QueueEmpty:
                queued = snap_msg
            try:
                if ws.client_state == WebSocketState.CONNECTED:
                    await ws.send_json(queued)
            except (WebSocketDisconnect, RuntimeError):
                # Client disappeared during the handshake — clean up and exit.
                _clients[game_id].discard(client_q)
                return

        # 5. Get engine pool from app state.
        engine: EnginePool = ws.app.state.engine_pool

        # 6. Concurrent reader / writer loop.
        stop_event = asyncio.Event()

        async def writer() -> None:
            from starlette.websockets import WebSocketState
            while not stop_event.is_set():
                try:
                    msg = await asyncio.wait_for(client_q.get(), timeout=1.0)
                    if ws.client_state != WebSocketState.CONNECTED:
                        # Client gone; stop trying to send.
                        break
                    await ws.send_json(msg)
                except asyncio.TimeoutError:
                    continue
                except WebSocketDisconnect:
                    break
                except Exception:
                    break

        async def reader() -> None:
            while not stop_event.is_set():
                try:
                    raw = await ws.receive_text()
                except WebSocketDisconnect:
                    break
                except Exception:
                    break
                try:
                    frame = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                ftype = frame.get("type")

                if ftype == "ping":
                    pong = {"type": "pong"}
                    _publish(game_id, pong)

                elif ftype == "request_snapshot":
                    async with SessionLocal() as s:
                        g2 = await _load_game_in_session(s, game_id)
                        if g2 is not None:
                            _publish(game_id, _snapshot_msg(g2))

                elif ftype == "move":
                    await _handle_move(game_id, frame.get("moveUci", ""), sess, engine)

                elif ftype == "cancel_thinking":
                    # Engine doesn't stream — nothing to cancel.
                    pass

            stop_event.set()

        reader_task = asyncio.ensure_future(reader())
        writer_task = asyncio.ensure_future(writer())

        await asyncio.gather(reader_task, writer_task, return_exceptions=True)
        for t in (reader_task, writer_task):
            if not t.done():
                t.cancel()

    finally:
        _clients[game_id].discard(client_q)


async def _handle_move(
    game_id: int, move_uci: str, sess, engine: EnginePool
) -> None:
    """Process a human move and then trigger engine reply if applicable."""
    # Parse move.
    try:
        move_int = _parse_move_uci(move_uci)
    except ValueError as e:
        _publish(game_id, {"type": "error", "code": "illegal_move", "reason": str(e)})
        return

    async with SessionLocal() as s:
        g = await _load_game_in_session(s, game_id)
        if g is None:
            _publish(game_id, {"type": "error", "code": "not_found"})
            return
        if g.status != "active":
            _publish(game_id, {"type": "error", "code": "game_not_active"})
            return
        # Ownership re-check (session may have changed).
        if (sess.user and g.user_id != sess.user.id) or \
           (sess.anon and g.anon_session_id != sess.anon.id):
            _publish(game_id, {"type": "error", "code": "not_owner"})
            return
        if g.side_to_move != g.side:
            _publish(game_id, {"type": "error", "code": "not_your_turn"})
            return

        # 1) Compute new position.
        try:
            new_pos = await engine.apply_move(position_pos=g.current_fen, move=move_int)
        except (EngineError, NotImplementedError) as e:
            _publish(game_id, {"type": "error", "code": "engine_unavailable"})
            return
        except (ValueError, IndexError) as e:
            _publish(game_id, {"type": "error", "code": "illegal_move", "reason": str(e)})
            return

        # 2) Round-trip validate.
        try:
            await engine.push_position(new_pos)
        except (EngineError, RuntimeError) as e:
            _publish(game_id, {"type": "error", "code": "illegal_move", "reason": "engine rejected position"})
            return

        # 3) Commit human move.
        await apply_move(s, game=g, move_uci=move_uci, actor="human", fen_after=new_pos)
        await s.refresh(g, ["moves", "events"])

        # 4) Publish move_applied.
        snap = build_snapshot(g)
        _publish(game_id, {"type": "move_applied", "game": snap.model_dump()})

        # 5) Check if game finished after human move.
        if g.status == "finished":
            _publish(game_id, {"type": "game_finished", "game": snap.model_dump()})
            return

        # 6) Engine's turn?
        if g.side_to_move == g.side:
            # Not engine's turn yet (shouldn't happen in solo vs engine, but guard).
            return

        engine_level = g.engine_level
        current_fen = g.current_fen
        g_id = g.id

    # Outside the DB session so engine think() can run freely.
    await _run_engine_turn(game_id=g_id, engine=engine,
                           engine_level=engine_level, current_fen=current_fen)


async def _run_engine_turn(
    *, game_id: int, engine: EnginePool, engine_level: str, current_fen: str
) -> None:
    """Ask engine for a move and commit it."""
    time_ms = LEVEL_TO_MS.get(engine_level, 2000)

    # Publish thinking started.
    _publish(game_id, {"type": "engine_thinking", "started": True})

    try:
        result = await engine.think(position_pos=current_fen, time_ms=time_ms)
    except EngineError as e:
        # Log an engine_error event, publish event, do NOT finalize.
        async with SessionLocal() as s:
            g = await _load_game_in_session(s, game_id)
            if g is not None:
                s.add(GameEvent(
                    game_id=game_id, ply_at=g.current_ply,
                    actor="engine", type="engine_error",
                    payload_json=json.dumps({"error": str(e)}),
                ))
                await s.commit()
        _publish(game_id, {"type": "event", "event": "engine_error"})
        _publish(game_id, {"type": "engine_thinking", "started": False})
        return

    # Terminal result (game over after human move — engine detects it).
    if result.terminal is not None:
        async with SessionLocal() as s:
            g = await _load_game_in_session(s, game_id)
            if g is not None:
                _finalize_game(g, result.terminal)
                s.add(g)
                await s.commit()
                await s.refresh(g, ["moves", "events"])
                snap = build_snapshot(g)
                _publish(game_id, {"type": "game_finished", "game": snap.model_dump()})
        _publish(game_id, {"type": "engine_thinking", "started": False})
        return

    # Normal engine move.
    try:
        new_pos = await engine.apply_move(position_pos=current_fen, move=result.move)
    except Exception as e:
        _publish(game_id, {"type": "event", "event": "engine_error"})
        _publish(game_id, {"type": "engine_thinking", "started": False})
        return

    async with SessionLocal() as s:
        g = await _load_game_in_session(s, game_id)
        if g is None:
            _publish(game_id, {"type": "engine_thinking", "started": False})
            return

        await apply_move(
            s, game=g,
            move_uci=str(result.move), actor="engine", fen_after=new_pos,
            eval_cp=result.final_eval_cp, eval_depth=result.final_depth,
            think_time_ms=result.think_time_ms,
        )
        await s.refresh(g, ["moves", "events"])

        snap = build_snapshot(g)
        _publish(game_id, {"type": "engine_move", "game": snap.model_dump()})

        if g.status == "finished":
            _publish(game_id, {"type": "game_finished", "game": snap.model_dump()})

    _publish(game_id, {"type": "engine_thinking", "started": False})


def _finalize_game(game: Game, terminal: str) -> None:
    """Set game.status/result/result_reason/finished_at from a terminal result string."""
    game.status = "finished"
    game.finished_at = datetime.now(timezone.utc)
    if terminal == "white_win":
        game.result = "win_white"
    elif terminal == "black_win":
        game.result = "win_black"
    else:
        game.result = "draw"
    game.result_reason = "engine_terminal"
