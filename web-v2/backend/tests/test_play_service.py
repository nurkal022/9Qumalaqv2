import pytest
from app.db.base import SessionLocal
from app.db.models import AnonSession
from app.play.service import create_game, apply_move, takeback, resign


async def _new_anon(sid: str = "test-anon") -> str:
    async with SessionLocal() as s:
        a = AnonSession(id=sid)
        s.add(a)
        await s.commit()
    return sid


async def test_create_game_persists():
    anon_id = await _new_anon()
    async with SessionLocal() as s:
        g = await create_game(s, owner={"anon_session_id": anon_id}, side=0,
                              engine_level="normal", clock=None, use_book=False,
                              start_fen="X", engine_build="abc")
        assert g.id is not None and g.status == "active"


async def test_apply_move_increments_ply():
    anon_id = await _new_anon("anon-2")
    async with SessionLocal() as s:
        g = await create_game(s, owner={"anon_session_id": anon_id}, side=0,
                              engine_level="normal", clock=None, use_book=False,
                              start_fen="X", engine_build="abc")
        await apply_move(s, game=g, move_uci="1-3", actor="human", fen_after="Y")
        assert g.current_ply == 1
        assert g.current_fen == "Y"
        assert g.side_to_move == 1


async def test_takeback_to_zero_resets_to_start():
    anon_id = await _new_anon("anon-3")
    async with SessionLocal() as s:
        g = await create_game(s, owner={"anon_session_id": anon_id}, side=0,
                              engine_level="normal", clock=None, use_book=False,
                              start_fen="X", engine_build="abc")
        await apply_move(s, game=g, move_uci="1-3", actor="human", fen_after="Y")
        await apply_move(s, game=g, move_uci="2-1", actor="engine", fen_after="Z")
        await takeback(s, game=g, to_ply=0)
        assert g.current_ply == 0 and g.current_fen == "X"


async def test_resign_finalizes_game():
    anon_id = await _new_anon("anon-4")
    async with SessionLocal() as s:
        g = await create_game(s, owner={"anon_session_id": anon_id}, side=0,
                              engine_level="normal", clock=None, use_book=False,
                              start_fen="X", engine_build="abc")
        await resign(s, game=g)
        assert g.status == "finished"
        assert g.result == "win_black"
        assert g.result_reason == "resign"
