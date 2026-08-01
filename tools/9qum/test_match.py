#!/usr/bin/env python3
"""Board-expiry recovery in tools/9qum/match.py, tested WITHOUT any network access.

The 9qum gate for phase A died mid-run: after 9 completed games, POST game/move
returned HTTP 404 "Партия не найдена" (their analysis board had expired) and the whole
process raised, losing the rest of that measurement. match.py now recovers by
recreating the board from the current position and voids only the unsalvageable game
instead of crashing or miscounting it.

This test never touches the real API: `Api.post` is monkeypatched (per-test, on a real
`match.Api` instance so the request-building/rate-limiter code is untouched) to replay a
scripted list of responses/exceptions, and the opponent engine is a tiny stub. A live
measurement is running against 9qum's rate-limited API right now, so no test here may
open a socket.

Run: python3.12 tools/9qum/test_match.py
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
import match


# --------------------------------------------------------------------------
# Fakes: a scripted Api.post (no sockets) and a scripted engine.
# --------------------------------------------------------------------------
def make_scripted_api(script):
    """A real match.Api instance with `.post` monkeypatched to replay `script`:
    {path: [response_dict_or_exception, ...]}, consumed in call order per path.

    Raising the exact exception instances play_game must react to (GameNotFoundError,
    or a plain Exception to simulate recreation itself failing) exercises the recovery
    logic exactly as a real 404/500 would via the real Api.post -- just without a socket.
    """
    api = match.Api(token="fake-token-not-real", rps=1000)  # high rps: no rate-limit sleeps in tests
    queues = {path: list(responses) for path, responses in script.items()}
    calls = []

    def fake_post(path, payload, timeout=180):
        calls.append((path, payload))
        queue = queues.get(path)
        if not queue:
            raise AssertionError(f"unscripted extra call to {path!r} (payload={payload})")
        entry = queue.pop(0)
        if isinstance(entry, BaseException):
            raise entry
        return entry

    api.post = fake_post
    api.calls = calls
    api.queues = queues
    return api


class FakeEngine:
    """Duck-types tools/playok/engine.py's Engine.bestmove; always plays a fixed pit.
    The scripted API decides the resulting state regardless of which hole was submitted,
    so the move value itself doesn't need to be legal for these tests."""

    def __init__(self, move=0):
        self.move = move

    def bestmove(self, pos, time_ms):
        return self.move


def state(gid, to_move, finished=False, winner=None, **extra):
    base = {
        "game_id": gid, "to_move": to_move, "finished": finished, "winner": winner,
        "pits": [9] * 18, "kazan": [0, 0], "tuzdyk": [None, None],
        "record": "", "tfen": "", "net_version": 400,
    }
    base.update(extra)
    return base


class FakeResponse:
    """Stands in for requests.Response for Api.post's own 404-detection logic."""

    def __init__(self, status_code, data):
        self.status_code = status_code
        self._data = data

    def json(self):
        return self._data


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


TMP_DIR = None


def _log_path(name):
    return os.path.join(TMP_DIR, f"{name}.jsonl")


# --------------------------------------------------------------------------
# Api.post: 404/not-found detection (the real request-handling code, fake HTTP only)
# --------------------------------------------------------------------------
def test_api_post_raises_game_not_found_on_404():
    api = match.Api(token="fake-token-not-real", rps=1000)
    api.plain.post = lambda url, json=None, timeout=None: FakeResponse(
        404, {"detail": "Партия не найдена"})
    try:
        api.post("game/move", {"game_id": "g1", "hole": 0})
    except match.GameNotFoundError:
        pass
    else:
        raise AssertionError("expected GameNotFoundError on HTTP 404")


def test_api_post_detects_not_found_message_even_off_404():
    # their server could plausibly return a different status for an expired board;
    # the message-based fallback must still catch it.
    api = match.Api(token="fake-token-not-real", rps=1000)
    api.plain.post = lambda url, json=None, timeout=None: FakeResponse(
        410, {"detail": "Board not found"})
    try:
        api.post("game/move", {"game_id": "g1", "hole": 0})
    except match.GameNotFoundError:
        pass
    else:
        raise AssertionError("expected GameNotFoundError when detail says not-found, even off HTTP 404")


def test_api_post_other_4xx_is_plain_runtimeerror_not_game_not_found():
    api = match.Api(token="fake-token-not-real", rps=1000)
    api.plain.post = lambda url, json=None, timeout=None: FakeResponse(
        400, {"detail": "invalid move"})
    try:
        api.post("game/move", {"game_id": "g1", "hole": 0})
    except match.GameNotFoundError:
        raise AssertionError("a genuine bad-move error must not be treated as board expiry")
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError on HTTP 400")


def test_api_post_returns_data_on_200():
    api = match.Api(token="fake-token-not-real", rps=1000)
    api.plain.post = lambda url, json=None, timeout=None: FakeResponse(200, {"ok": True})
    assert api.post("game/move", {"game_id": "g1", "hole": 0}) == {"ok": True}


# --------------------------------------------------------------------------
# _position_of: what recovery restores from
# --------------------------------------------------------------------------
def test_position_of_reads_back_the_harness_current_state():
    st = state("gX", to_move=1, kazan=[12, 7], tuzdyk=[3, None])
    assert match._position_of(st) == {
        "pits": st["pits"], "kazan": [12, 7], "tuzdyk": [3, None], "to_move": 1,
    }


# --------------------------------------------------------------------------
# play_game: normal game, one recovery, and an unrecoverable (void) game
# --------------------------------------------------------------------------
def test_normal_game_completes_and_counts():
    log_path = _log_path("normal")
    api = make_scripted_api({
        "analysis/new": [state("g1", to_move=0)],
        "game/move": [state("g1", to_move=1),
                      state("g1", to_move=0, finished=True, winner=0)],
        "ai/think": [{"moves": [{"hole": 3, "N": 10}], "best": 3}],
    })
    outcome = match.play_game(api, FakeEngine(move=0), our_seat=0, level="i", move_ms=100,
                              pick="argmax", their_sims=90, log_path=log_path, idx=0)
    assert outcome == {"result": "W", "recoveries": 0}
    assert all(len(q) == 0 for q in api.queues.values()), "scripted responses left unconsumed"

    rows = read_jsonl(log_path)
    assert len(rows) == 1
    assert rows[0]["result"] == "W" and rows[0]["recoveries"] == 0
    assert "void_reason" not in rows[0]


def test_recovers_from_one_404_mid_game_and_completes():
    log_path = _log_path("recovers")
    api = make_scripted_api({
        # initial board, then the board recreated after the 404
        "analysis/new": [state("g2", to_move=0), state("g2b", to_move=1)],
        "game/move": [
            state("g2", to_move=1),                                    # our move: fine
            match.GameNotFoundError("game/move: HTTP 404 Партия не найдена"),  # board expired
            state("g2b", to_move=0, finished=True, winner=0),          # retried on the new board
        ],
        "ai/think": [{"moves": [{"hole": 5, "N": 7}], "best": 5}],
    })
    outcome = match.play_game(api, FakeEngine(move=0), our_seat=0, level="i", move_ms=100,
                              pick="argmax", their_sims=90, log_path=log_path, idx=1)
    assert outcome == {"result": "W", "recoveries": 1}
    assert all(len(q) == 0 for q in api.queues.values())

    # the retried move must land on the RECREATED board, not the expired one
    move_calls = [payload for (path, payload) in api.calls if path == "game/move"]
    assert move_calls[-1]["game_id"] == "g2b"
    # recreation must restore the CURRENT position (to_move=1, mid-game), not the opening
    recreate_calls = [payload for (path, payload) in api.calls if path == "analysis/new"]
    assert recreate_calls[-1]["position"]["to_move"] == 1

    rows = read_jsonl(log_path)
    assert len(rows) == 1
    assert rows[0]["result"] == "W" and rows[0]["recoveries"] == 1


def test_void_when_board_recreation_fails():
    log_path = _log_path("void")
    api = make_scripted_api({
        "analysis/new": [state("g3", to_move=0),
                          RuntimeError("analysis/new: HTTP 500 their server is down")],
        "game/move": [match.GameNotFoundError("game/move: HTTP 404 Партия не найдена")],
        "ai/think": [],
    })
    outcome = match.play_game(api, FakeEngine(move=0), our_seat=0, level="i", move_ms=100,
                              pick="argmax", their_sims=90, log_path=log_path, idx=2)
    assert outcome["result"] == "VOID"
    assert outcome["recoveries"] == 1
    assert all(len(q) == 0 for q in api.queues.values())

    rows = read_jsonl(log_path)
    assert len(rows) == 1
    assert rows[0]["result"] == "VOID"
    assert rows[0]["recoveries"] == 1
    assert rows[0].get("void_reason"), "a void game must record WHY it was abandoned"


def test_void_after_exhausting_the_recovery_budget():
    log_path = _log_path("exhausted")
    # every single game/move 404s; with max_recoveries=2 the 3rd 404 must void, never loop forever
    api = make_scripted_api({
        "analysis/new": [
            state("g4", to_move=0),
            state("g4a", to_move=0),
            state("g4b", to_move=0),
        ],
        "game/move": [
            match.GameNotFoundError("404 #1"),
            match.GameNotFoundError("404 #2"),
            match.GameNotFoundError("404 #3"),
        ],
        "ai/think": [],
    })
    outcome = match.play_game(api, FakeEngine(move=0), our_seat=0, level="i", move_ms=100,
                              pick="argmax", their_sims=90, log_path=log_path, idx=3,
                              max_recoveries=2)
    assert outcome["result"] == "VOID"
    assert outcome["recoveries"] == 2, "must stop recovering at the bound, not retry forever"
    rows = read_jsonl(log_path)
    assert rows[0]["void_reason"]


# --------------------------------------------------------------------------
# aggregate(): void games must never be folded into the W/D/L score
# --------------------------------------------------------------------------
def test_aggregate_excludes_void_from_totals_and_counts_recoveries():
    results = [
        {"result": "W", "recoveries": 0},
        {"result": "L", "recoveries": 0},
        {"result": "D", "recoveries": 1},
        {"result": "VOID", "recoveries": 1},
        {"result": "VOID", "recoveries": 3},
    ]
    agg = match.aggregate(results)
    assert agg["w"] == 1 and agg["d"] == 1 and agg["l"] == 1
    assert agg["n"] == 3, "void games must be excluded from the counted total"
    assert agg["void"] == 2
    assert agg["recovered"] == 3, "any game (void or completed) that needed >=1 recovery"


def test_aggregate_all_void_gives_zero_counted_not_a_zero_score_loss():
    agg = match.aggregate([{"result": "VOID", "recoveries": 3}])
    assert agg["n"] == 0 and agg["w"] == agg["d"] == agg["l"] == 0
    assert agg["void"] == 1


TESTS = [
    test_api_post_raises_game_not_found_on_404,
    test_api_post_detects_not_found_message_even_off_404,
    test_api_post_other_4xx_is_plain_runtimeerror_not_game_not_found,
    test_api_post_returns_data_on_200,
    test_position_of_reads_back_the_harness_current_state,
    test_normal_game_completes_and_counts,
    test_recovers_from_one_404_mid_game_and_completes,
    test_void_when_board_recreation_fails,
    test_void_after_exhausting_the_recovery_budget,
    test_aggregate_excludes_void_from_totals_and_counts_recoveries,
    test_aggregate_all_void_gives_zero_counted_not_a_zero_score_loss,
]


if __name__ == "__main__":
    TMP_DIR = tempfile.mkdtemp(prefix="match_test_")
    try:
        for t in TESTS:
            t()
        print(f"OK: board-expiry recovery + void accounting ({len(TESTS)}/{len(TESTS)})")
    finally:
        shutil.rmtree(TMP_DIR, ignore_errors=True)
