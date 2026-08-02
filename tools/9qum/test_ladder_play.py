#!/usr/bin/env python3
"""Offline tests for tools/9qum/ladder_play.py -- the one-shot 9qum.com ladder-play
bridge. This drives a real websocket against a live rated game when actually run, so
every test here is REQUIRED to open zero sockets: WsClient and Engine are replaced with
scripted fakes (same pattern as tools/9qum/test_match.py's monkeypatched Api), and pure
helper functions (table selection, state extraction, clock budgeting, move-diffing) are
exercised directly.

Run: python3.12 tools/9qum/test_ladder_play.py
"""
import glob
import io
import json
import os
import sys
import tempfile
import shutil
from threading import Event, Lock

sys.path.insert(0, os.path.dirname(__file__))
import ladder_play as lp


# --------------------------------------------------------------------------
# find_target_table
# --------------------------------------------------------------------------
def _table(**kw):
    base = {"id": "t1", "no": 1, "name": "Ищу соперника", "tc": {"timeMin": 7, "fischer": 2},
            "rated": True, "access": "open", "closed": False, "stake": 0,
            "status": "waiting", "seats": [None, {"name": "ИИ 9qum", "rating": 2175, "rank": "II"}],
            "spectators": 0}
    base.update(kw)
    return base


def test_find_target_table_matches_open_waiting_free_seat():
    lobby = {"tables": [_table()]}
    t, seat = lp.find_target_table(lobby, "ИИ 9qum")
    assert t is not None and t["id"] == "t1"
    assert seat == 0


def test_find_target_table_picks_correct_free_seat_when_bot_is_seat0():
    lobby = {"tables": [_table(seats=[{"name": "ИИ 9qum", "rating": 2175, "rank": "II"}, None])]}
    t, seat = lp.find_target_table(lobby, "ИИ 9qum")
    assert t is not None
    assert seat == 1


def test_find_target_table_rejects_closed_table():
    """Regression test for a real observed 9qum table: status='waiting' with the
    requested opponent's exact name, but access='closed'/closed=True (almost certainly
    a private challenge slot). Must NOT be selected even though the name matches."""
    lobby = {"tables": [_table(access="closed", closed=True)]}
    t, seat = lp.find_target_table(lobby, "ИИ 9qum")
    assert t is None and seat is None


def test_find_target_table_rejects_non_open_access_even_if_not_marked_closed():
    lobby = {"tables": [_table(access="invite", closed=False)]}
    t, seat = lp.find_target_table(lobby, "ИИ 9qum")
    assert t is None


def test_find_target_table_rejects_status_playing():
    lobby = {"tables": [_table(status="playing")]}
    t, seat = lp.find_target_table(lobby, "ИИ 9qum")
    assert t is None


def test_find_target_table_rejects_wrong_opponent_name():
    lobby = {"tables": [_table()]}
    t, seat = lp.find_target_table(lobby, "some other bot")
    assert t is None


def test_find_target_table_rejects_both_seats_full():
    lobby = {"tables": [_table(seats=[{"name": "a", "rating": 1}, {"name": "ИИ 9qum", "rating": 2175}])]}
    t, seat = lp.find_target_table(lobby, "ИИ 9qum")
    assert t is None


def test_find_target_table_returns_none_on_empty_lobby():
    t, seat = lp.find_target_table({"tables": []}, "ИИ 9qum")
    assert t is None and seat is None
    t, seat = lp.find_target_table(None, "ИИ 9qum")
    assert t is None and seat is None


def test_find_target_table_picks_first_of_several_matches():
    lobby = {"tables": [_table(id="a"), _table(id="b")]}
    t, seat = lp.find_target_table(lobby, "ИИ 9qum")
    assert t["id"] == "a"


# --------------------------------------------------------------------------
# summarize_table / summarize_lobby
# --------------------------------------------------------------------------
def test_summarize_table_shape():
    s = lp.summarize_table(_table())
    assert s["id"] == "t1"
    assert s["access"] == "open"
    assert s["seats"] == [None, ["ИИ 9qum", 2175]]


def test_summarize_lobby_lists_opponent_tables_even_if_unjoinable():
    lobby = {"online": 5, "tables": [_table(access="closed", closed=True)]}
    summary = lp.summarize_lobby(lobby, "ИИ 9qum")
    assert summary["online"] == 5
    assert summary["num_tables"] == 1
    assert len(summary["opponent_tables"]) == 1  # visible in the report even though unjoinable


# --------------------------------------------------------------------------
# extract_game_state
# --------------------------------------------------------------------------
GAME = {"pits": [9] * 18, "kazan": [0, 0], "tuzdyk": [None, None], "to_move": 0,
        "finished": False, "winner": None, "legal_moves": list(range(9))}


def test_extract_game_state_from_direct_table_push():
    evt = {"event": "message", "payload": {"id": "t1", "seats": [None, {}], "game": GAME}}
    g = lp.extract_game_state(evt, "t1")
    assert g == GAME


def test_extract_game_state_from_nested_table_key():
    evt = {"event": "message", "payload": {"type": "table.update", "table": {"id": "t1", "game": GAME}}}
    g = lp.extract_game_state(evt, "t1")
    assert g == GAME


def test_extract_game_state_from_tables_list():
    evt = {"event": "message", "payload": {"type": "lobby", "tables": [{"id": "other"}, {"id": "t1", "game": GAME}]}}
    g = lp.extract_game_state(evt, "t1")
    assert g == GAME


def test_extract_game_state_inline_pits_no_wrapper():
    evt = {"event": "message", "payload": {"id": "t1", "pits": GAME["pits"], "kazan": [0, 0],
                                           "tuzdyk": [None, None], "to_move": 1}}
    g = lp.extract_game_state(evt, "t1")
    assert g["to_move"] == 1


def test_extract_game_state_returns_none_for_unrelated_table():
    evt = {"event": "message", "payload": {"id": "other", "game": GAME}}
    assert lp.extract_game_state(evt, "t1") is None


def test_extract_game_state_ignores_non_message_events():
    assert lp.extract_game_state({"event": "open"}, "t1") is None
    assert lp.extract_game_state({"event": "close", "code": 1000}, "t1") is None
    assert lp.extract_game_state(None, "t1") is None


def test_extract_game_state_ignores_message_without_pits():
    evt = {"event": "message", "payload": {"id": "t1", "seats": [None, {}]}}
    assert lp.extract_game_state(evt, "t1") is None


# --------------------------------------------------------------------------
# is_error_event
# --------------------------------------------------------------------------
def test_is_error_event_detects_explicit_error_type():
    evt = {"event": "message", "payload": {"type": "error", "message": "nope"}}
    assert lp.is_error_event(evt) == "nope"


def test_is_error_event_detects_generic_error_key():
    evt = {"event": "message", "payload": {"type": "table", "error": "seat taken"}}
    assert lp.is_error_event(evt) == "seat taken"


def test_is_error_event_ignores_normal_message():
    evt = {"event": "message", "payload": {"type": "lobby", "tables": []}}
    assert lp.is_error_event(evt) is None


def test_is_error_event_ignores_non_message_events():
    assert lp.is_error_event({"event": "open"}) is None
    assert lp.is_error_event({"event": "_bridge_eof"}) is None


def test_is_error_event_detects_bridge_level_error():
    """ws_bridge.js's own `{"event":"error",...}` (e.g. a send attempted before the
    socket opened) must be surfaced too, not just server-side error messages."""
    evt = {"event": "error", "error": "send while not open (state=0)"}
    assert lp.is_error_event(evt) == "send while not open (state=0)"


# --------------------------------------------------------------------------
# read_remaining_ms
# --------------------------------------------------------------------------
def test_read_remaining_ms_list_form():
    assert lp.read_remaining_ms({"clocks": [12345, 6789]}, 0) == 12345
    assert lp.read_remaining_ms({"clocks": [12345, 6789]}, 1) == 6789


def test_read_remaining_ms_dict_str_keys():
    assert lp.read_remaining_ms({"clocks": {"0": 1000, "1": 2000}}, 1) == 2000


def test_read_remaining_ms_dict_int_keys():
    assert lp.read_remaining_ms({"clocks": {0: 1000, 1: 2000}}, 0) == 1000


def test_read_remaining_ms_missing_field_returns_none():
    assert lp.read_remaining_ms({}, 0) is None


def test_read_remaining_ms_short_list_returns_none():
    assert lp.read_remaining_ms({"clocks": [1000]}, 1) is None


def test_read_remaining_ms_bad_type_returns_none():
    assert lp.read_remaining_ms({"clocks": "nonsense"}, 0) is None


# --------------------------------------------------------------------------
# compute_move_ms
# --------------------------------------------------------------------------
def test_compute_move_ms_normal_scales_with_remaining():
    assert lp.compute_move_ms(50000) == 2000  # 50000/25


def test_compute_move_ms_clamps_low():
    assert lp.compute_move_ms(1000) == 500  # 1000/25=40, clamped up to 500


def test_compute_move_ms_clamps_high():
    assert lp.compute_move_ms(500000) == 4000  # 500000/25=20000, clamped down to 4000


def test_compute_move_ms_fallback_when_clock_unreadable():
    assert lp.compute_move_ms(None) == 2000


# --------------------------------------------------------------------------
# diff_move
# --------------------------------------------------------------------------
def test_diff_move_detects_our_move_with_exact_hole():
    prev = {"to_move": 1}
    cur = {"to_move": 0}
    d = lp.diff_move(prev, cur, our_seat=1, last_sent_hole=4)
    assert d == {"seat": 1, "hole": 4, "hole_source": "sent"}


def test_diff_move_detects_opponent_move_hole_unknown():
    prev = {"to_move": 0}
    cur = {"to_move": 1}
    d = lp.diff_move(prev, cur, our_seat=1, last_sent_hole=None)
    assert d == {"seat": 0, "hole": None, "hole_source": "not_observed"}


def test_diff_move_returns_none_when_to_move_unchanged():
    prev = {"to_move": 0}
    cur = {"to_move": 0}
    assert lp.diff_move(prev, cur, our_seat=1, last_sent_hole=None) is None


def test_diff_move_returns_none_when_prev_to_move_missing():
    assert lp.diff_move({}, {"to_move": 0}, our_seat=0, last_sent_hole=None) is None


# --------------------------------------------------------------------------
# result_for_us
# --------------------------------------------------------------------------
def test_result_for_us_win():
    assert lp.result_for_us({"winner": 1}, our_seat=1) == "win"


def test_result_for_us_loss():
    assert lp.result_for_us({"winner": 0}, our_seat=1) == "loss"


def test_result_for_us_draw_on_minus_one():
    assert lp.result_for_us({"winner": -1}, our_seat=1) == "draw"


def test_result_for_us_draw_on_none():
    assert lp.result_for_us({"winner": None}, our_seat=1) == "draw"


# --------------------------------------------------------------------------
# find_table_by_id / find_table_by_creator / find_table_with_both_seated -- the
# --mode challenge/invite table-acquisition helpers, tolerant of the same push-shape
# variety as extract_game_state (a lobby snapshot's `tables` list, a `table` key, or
# the payload already looking like a table).
# --------------------------------------------------------------------------
def test_find_table_by_id_direct_payload():
    payload = {"id": "t9", "seats": [None, {"name": "x"}]}
    assert lp.find_table_by_id(payload, "t9") == payload


def test_find_table_by_id_nested_table_key():
    inner = {"id": "t9", "seats": [None, None]}
    payload = {"type": "table.update", "table": inner}
    assert lp.find_table_by_id(payload, "t9") == inner


def test_find_table_by_id_from_tables_list():
    inner = {"id": "t9", "seats": [None, None]}
    payload = {"type": "lobby", "tables": [{"id": "other"}, inner]}
    assert lp.find_table_by_id(payload, "t9") == inner


def test_find_table_by_id_returns_none_when_absent():
    assert lp.find_table_by_id({"type": "lobby", "tables": []}, "t9") is None
    assert lp.find_table_by_id(None, "t9") is None


def test_find_table_by_creator_matches_even_with_no_seats_filled():
    """Regression test for a real observed 9qum push: table.create's own table.state
    response came back with seats=[None, None] (the creator is NOT auto-seated) but a
    top-level `creator` field naming us -- this must still be found."""
    table = _table(id="new1", creator="Qonaq_5599", seats=[None, None])
    payload = {"type": "table.state", **table}
    assert lp.find_table_by_creator(payload, "Qonaq_5599") == payload


def test_find_table_by_creator_from_tables_list():
    table = _table(id="new1", creator="Qonaq_5599", seats=[None, None])
    payload = {"type": "lobby", "tables": [_table(id="unrelated", creator="someone else"), table]}
    assert lp.find_table_by_creator(payload, "Qonaq_5599")["id"] == "new1"


def test_find_table_by_creator_returns_none_when_absent():
    payload = {"type": "lobby", "tables": [_table(creator="someone else")]}
    assert lp.find_table_by_creator(payload, "Qonaq_5599") is None


def test_find_table_with_both_seated_matches_paired_table():
    table = _table(id="pair1", seats=[{"name": "Qonaq_5599", "rating": 2000},
                                       {"name": "ИИ 9qum", "rating": 2175}])
    payload = {"type": "lobby", "tables": [_table(id="unrelated"), table]}
    t, seat = lp.find_table_with_both_seated(payload, "Qonaq_5599", "ИИ 9qum")
    assert t["id"] == "pair1" and seat == 0


def test_find_table_with_both_seated_none_when_opponent_absent():
    payload = {"type": "lobby", "tables": [_table(seats=[{"name": "Qonaq_5599"}, None])]}
    t, seat = lp.find_table_with_both_seated(payload, "Qonaq_5599", "ИИ 9qum")
    assert t is None and seat is None


# --------------------------------------------------------------------------
# build_settings / verify_settings_recorded
# --------------------------------------------------------------------------
def test_build_settings_shape():
    s = lp.build_settings(False, 7.0, 2)
    assert s == {"rated": False, "tc": {"timeMin": 7.0, "fischer": 2}}


def test_build_settings_coerces_truthy_rated():
    assert lp.build_settings(1, 5, 3)["rated"] is True


def test_verify_settings_recorded_matches_when_equal():
    v = lp.verify_settings_recorded({"rated": False, "access": "open", "tc": {"timeMin": 7}}, False)
    assert v == {"requested_rated": False, "server_rated": False, "access": "open",
                 "tc": {"timeMin": 7}, "matches": True}


def test_verify_settings_recorded_flags_mismatch():
    """The main open question this task exists to answer: whether the server honours
    an unrated request at all. If it silently returns a rated table, this must be
    reported as a mismatch, not treated as success."""
    v = lp.verify_settings_recorded({"rated": True, "access": "open", "tc": {"timeMin": 7}}, False)
    assert v["matches"] is False
    assert v["requested_rated"] is False and v["server_rated"] is True


# --------------------------------------------------------------------------
# WsClient: logging + wire format, WITHOUT spawning any subprocess
# --------------------------------------------------------------------------
def _bare_wsclient(fh, stdin=None, opened=True):
    """A WsClient instance with __init__ bypassed (no subprocess, no threads) --
    just enough state (`log_fh`, `_log_lock`, `_opened`, `proc`) for `_log`/`send` to
    run. `opened=True` (the default) pretends the handshake already completed, since
    most tests care about the logging/wire-format behaviour, not the handshake gate
    itself (see test_wsclient_send_raises_if_socket_never_opens for that)."""
    ws = object.__new__(lp.WsClient)
    ws.log_fh = fh
    ws._log_lock = Lock()
    ws._opened = Event()
    if opened:
        ws._opened.set()

    class _FakeProc:
        def __init__(self):
            self.stdin = stdin if stdin is not None else io.StringIO()
    ws.proc = _FakeProc()
    return ws


def test_wsclient_log_format_records_direction_and_payload():
    fh = io.StringIO()
    ws = _bare_wsclient(fh)
    ws._log("send", {"payload": {"type": "auth", "token": "x"}})
    ws._log("recv", {"event": "message", "payload": {"type": "auth.ok"}})
    lines = [json.loads(l) for l in fh.getvalue().splitlines()]
    assert lines[0]["dir"] == "send" and lines[0]["payload"]["type"] == "auth"
    assert lines[1]["dir"] == "recv" and lines[1]["payload"]["type"] == "auth.ok"
    assert "ts" in lines[0] and "ts" in lines[1]


def test_wsclient_send_writes_wire_format_to_bridge_stdin_and_logs_it():
    fh = io.StringIO()
    stdin = io.StringIO()
    ws = _bare_wsclient(fh, stdin=stdin)
    ws.send({"type": "game.move", "tableId": "t1", "hole": 3})
    wire = json.loads(stdin.getvalue().strip())
    assert wire == {"cmd": "send", "payload": {"type": "game.move", "tableId": "t1", "hole": 3}}
    logged = json.loads(fh.getvalue().strip())
    assert logged["dir"] == "send"
    assert logged["payload"] == {"type": "game.move", "tableId": "t1", "hole": 3}


def test_wsclient_send_raises_if_socket_never_opens():
    """Regression test: a real dry-run against 9qum.com sent `auth` before the bridge's
    websocket finished its TCP+TLS+WS handshake. ws_bridge.js silently dropped it
    (readyState != OPEN) and the run died 15s later on a confusing "no auth.ok"
    timeout. send() must now block on the handshake and raise a clear error instead of
    silently dropping the message."""
    fh = io.StringIO()
    ws = _bare_wsclient(fh, opened=False)
    orig_timeout = lp.OPEN_TIMEOUT_S
    lp.OPEN_TIMEOUT_S = 0.05
    try:
        try:
            ws.send({"type": "auth", "token": "x"})
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "did not open" in str(e)
    finally:
        lp.OPEN_TIMEOUT_S = orig_timeout
    # nothing was written to the (fake) bridge stdin, and nothing was logged as sent
    assert ws.proc.stdin.getvalue() == ""
    assert fh.getvalue() == ""


def test_wsclient_send_proceeds_once_opened():
    fh = io.StringIO()
    ws = _bare_wsclient(fh, opened=False)
    ws._opened.set()  # simulate the reader thread having observed {"event":"open"}
    ws.send({"type": "auth", "token": "x"})
    assert json.loads(ws.proc.stdin.getvalue().strip())["payload"]["type"] == "auth"


# --------------------------------------------------------------------------
# wait_for_type (uses only .next_event, so a minimal fake suffices)
# --------------------------------------------------------------------------
class _FakeWsEvents:
    def __init__(self, events):
        self._events = list(events)

    def next_event(self, timeout=None):
        if self._events:
            return self._events.pop(0)
        return None


def test_wait_for_type_returns_first_matching_event():
    ws = _FakeWsEvents([
        {"event": "message", "payload": {"type": "notify"}},
        {"event": "message", "payload": {"type": "auth.ok", "user": {"name": "x"}}},
    ])
    evt = lp.wait_for_type(ws, "auth.ok", timeout=1.0)
    assert evt["payload"]["user"]["name"] == "x"


def test_wait_for_type_raises_on_error_event():
    ws = _FakeWsEvents([{"event": "message", "payload": {"type": "error", "message": "bad token"}}])
    try:
        lp.wait_for_type(ws, "auth.ok", timeout=1.0)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "bad token" in str(e)


def test_wait_for_type_raises_on_bridge_eof():
    ws = _FakeWsEvents([{"event": "_bridge_eof"}])
    try:
        lp.wait_for_type(ws, "auth.ok", timeout=1.0)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "bridge exited" in str(e)


def test_wait_for_type_times_out_to_none():
    ws = _FakeWsEvents([])
    assert lp.wait_for_type(ws, "auth.ok", timeout=0.05) is None


# --------------------------------------------------------------------------
# main(): full integration with WsClient and Engine both faked (no socket, no
# subprocess) -- mirrors test_match.py's monkeypatched-Api pattern.
# --------------------------------------------------------------------------
AUTH_OK = {"event": "message", "payload": {"type": "auth.ok", "user": {"name": "Qonaq_5599", "rating": 2000}}}


def _lobby_evt(tables):
    return {"event": "message", "payload": {"type": "lobby", "online": 3, "tables": tables}}


class _FakeWsClient:
    """Replaces ladder_play.WsClient for main()-level tests: feeds a scripted list of
    incoming events in order and records every outgoing send(), with no socket and no
    subprocess. Constructed with WsClient's own call signature so monkeypatching the
    class name is a drop-in swap.
    """
    def __init__(self, log_fh, url=None, node_bin=None):
        self.log_fh = log_fh
        self._events = list(_FakeWsClient.SCRIPT)
        self.sent = []
        self.closed = False

    def send(self, payload):
        self.sent.append(payload)

    def next_event(self, timeout=None):
        if self._events:
            return self._events.pop(0)
        return None

    def close(self):
        self.closed = True


def _make_fake_ws_factory(events):
    _FakeWsClient.SCRIPT = events
    return _FakeWsClient


class _FakeEngine:
    """Replaces tools/playok/engine.py's Engine for main()-level tests."""
    instances = []

    def __init__(self, binary):
        self.binary = binary
        self.started = False
        self.stopped = False
        self._moves = list(_FakeEngine.SCRIPT)
        _FakeEngine.instances.append(self)

    def start(self):
        self.started = True

    def bestmove(self, pos, time_ms=1500):
        return self._moves.pop(0)

    def stop(self):
        self.stopped = True


def _make_fake_engine_factory(moves):
    _FakeEngine.SCRIPT = moves
    _FakeEngine.instances = []
    return _FakeEngine


def _run_main(argv, ws_events, engine_moves=None):
    """Run ladder_play.main() with sys.argv=argv and WsClient/Engine monkeypatched to
    scripted fakes; returns (exit_code, record_dict, fake_ws, fake_engine_instances).
    Restores all monkeypatches and sys.argv afterward regardless of outcome.
    """
    orig_argv = sys.argv
    orig_ws = lp.WsClient
    orig_engine = lp.Engine
    lp.WsClient = _make_fake_ws_factory(ws_events)
    lp.Engine = _make_fake_engine_factory(engine_moves or [])
    fake_ws_holder = {}
    orig_init = _FakeWsClient.__init__

    def capturing_init(self, log_fh, url=None, node_bin=None):
        orig_init(self, log_fh, url, node_bin)
        fake_ws_holder["ws"] = self
    _FakeWsClient.__init__ = capturing_init
    try:
        sys.argv = argv
        rc = lp.main()
        engine_instances = list(_FakeEngine.instances)
    finally:
        sys.argv = orig_argv
        lp.WsClient = orig_ws
        lp.Engine = orig_engine
        _FakeWsClient.__init__ = orig_init

    record_dir = None
    for a, nxt in zip(argv, argv[1:]):
        if a == "--log-dir":
            record_dir = nxt
    records = glob.glob(os.path.join(record_dir, "*_record.json"))
    assert len(records) == 1, f"expected exactly one record file, found {records}"
    with open(records[0], encoding="utf-8") as f:
        record = json.load(f)
    return rc, record, fake_ws_holder["ws"], engine_instances


def _session_file(tmpdir):
    p = os.path.join(tmpdir, "session.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"token": "fake-token-not-real"}, f)
    return p


def test_main_dry_run_reports_target_and_never_joins():
    tmpdir = tempfile.mkdtemp(prefix="ladder_play_test_")
    try:
        session = _session_file(tmpdir)
        table = _table()
        events = [AUTH_OK, _lobby_evt([table])]
        argv = ["ladder_play.py", "--dry-run", "--session", session, "--log-dir", tmpdir,
                "--opponent", "ИИ 9qum", "--max-wait", "1"]
        rc, record, ws, _engines = _run_main(argv, events)
        assert rc == 0
        assert record["outcome"] == "dry_run_ok"
        assert record["our_seat"] == 0
        assert record["target_table"]["id"] == "t1"
        # only the auth handshake may be sent -- never table.join/table.sit/game.move
        sent_types = [m.get("type") for m in ws.sent]
        assert sent_types == ["auth"], sent_types
        assert ws.closed is True
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_main_opponent_unavailable_within_max_wait_does_not_join():
    tmpdir = tempfile.mkdtemp(prefix="ladder_play_test_")
    try:
        session = _session_file(tmpdir)
        # A table for the opponent exists but is closed/private -- must be reported as
        # unavailable, exactly like the real table this project observed live.
        events = [AUTH_OK, _lobby_evt([_table(access="closed", closed=True)])]
        argv = ["ladder_play.py", "--session", session, "--log-dir", tmpdir,
                "--opponent", "ИИ 9qum", "--max-wait", "0.3"]
        rc, record, ws, _engines = _run_main(argv, events)
        assert rc == 0
        assert record["outcome"] == "opponent_unavailable"
        assert len(record["lobby_snapshot"]["opponent_tables"]) == 1  # visible, just not joinable
        sent_types = [m.get("type") for m in ws.sent]
        assert sent_types == ["auth"], sent_types
        assert ws.closed is True
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_main_full_game_plays_moves_records_result_and_leaves_cleanly():
    tmpdir = tempfile.mkdtemp(prefix="ladder_play_test_")
    try:
        session = _session_file(tmpdir)
        table = _table()  # our_seat = 0, opponent ("ИИ 9qum") is seat 1
        s0 = {"pits": [9] * 18, "kazan": [0, 0], "tuzdyk": [None, None], "to_move": 1,
              "finished": False, "winner": None, "legal_moves": list(range(9))}
        s1 = dict(s0, to_move=0, kazan=[0, 0])       # opponent (seat1) moved -> our turn
        s2 = dict(s0, to_move=1, kazan=[9, 0], finished=True, winner=0)  # we (seat0) moved, game over, we win
        push0 = {"event": "message", "payload": {"id": "t1", "game": s0}}
        push1 = {"event": "message", "payload": {"id": "t1", "game": s1}}
        push2 = {"event": "message", "payload": {"id": "t1", "game": s2}}
        events = [
            AUTH_OK,
            _lobby_evt([table]),
            push0,   # confirms our seat (a game push for our table implies we're seated)
            push0,   # first play-loop iteration: establishes prev_game, not our turn yet
            push1,   # opponent's move detected (hole unobserved) -> then it's our turn
            push2,   # our move detected (hole=3, exact) -> finished
        ]
        argv = ["ladder_play.py", "--session", session, "--log-dir", tmpdir,
                "--opponent", "ИИ 9qum", "--max-wait", "1"]
        rc, record, ws, engines = _run_main(argv, events, engine_moves=[3])
        assert rc == 0
        assert record["outcome"] == "completed"
        assert record["result"] == "win"
        assert record["final_kazan"] == [9, 0]
        assert record["plies"] == 2
        moves = record["moves"]
        assert moves[0]["seat"] == 1 and moves[0]["hole"] is None  # opponent, best-effort
        assert moves[1]["seat"] == 0 and moves[1]["hole"] == 3     # us, exact (we sent it)
        # exactly: join, sit, one move, then leave -- no resign (game finished normally),
        # no second join/sit, no move sent after the game was already over.
        kinds = [s["type"] for s in ws.sent]
        assert kinds == ["auth", "table.join", "table.sit", "game.move", "table.leave"]
        assert ws.sent[3]["hole"] == 3
        assert ws.closed is True
        assert engines[0].started and engines[0].stopped
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_main_illegal_engine_move_aborts_resigns_and_leaves():
    """If the engine ever proposes a move outside the server's own legal_moves (e.g. a
    schema mismatch bug), main() must not send it -- it must abort, resign the game
    (never just vanish mid-game) and still leave the table, rather than trusting a
    possibly-corrupt move on a live rated board."""
    tmpdir = tempfile.mkdtemp(prefix="ladder_play_test_")
    try:
        session = _session_file(tmpdir)
        table = _table()
        s0 = {"pits": [9] * 18, "kazan": [0, 0], "tuzdyk": [None, None], "to_move": 1,
              "finished": False, "winner": None, "legal_moves": list(range(9))}
        s1 = dict(s0, to_move=0, legal_moves=[0, 1, 2])  # our turn; engine will propose 8 (illegal)
        push0 = {"event": "message", "payload": {"id": "t1", "game": s0}}
        push1 = {"event": "message", "payload": {"id": "t1", "game": s1}}
        events = [AUTH_OK, _lobby_evt([table]), push0, push0, push1]
        argv = ["ladder_play.py", "--session", session, "--log-dir", tmpdir,
                "--opponent", "ИИ 9qum", "--max-wait", "1"]
        rc, record, ws, engines = _run_main(argv, events, engine_moves=[8])
        assert rc == 1
        assert record["outcome"] == "error"
        assert "not in server legal_moves" in record["error"]
        kinds = [s["type"] for s in ws.sent]
        assert kinds == ["auth", "table.join", "table.sit", "game.resign", "table.leave"]
        assert ws.closed is True
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# --------------------------------------------------------------------------
# main(): --mode challenge / --mode invite, same fake-WsClient/fake-Engine harness.
# No test here opens a real socket.
# --------------------------------------------------------------------------
def test_main_dry_run_challenge_reports_settings_and_never_sends_challenge():
    tmpdir = tempfile.mkdtemp(prefix="ladder_play_test_")
    try:
        session = _session_file(tmpdir)
        events = [AUTH_OK, _lobby_evt([])]
        argv = ["ladder_play.py", "--dry-run", "--mode", "challenge", "--session", session,
                "--log-dir", tmpdir, "--opponent", "ИИ 9qum", "--max-wait", "1"]
        rc, record, ws, _engines = _run_main(argv, events)
        assert rc == 0
        assert record["outcome"] == "dry_run_ok"
        assert record["would_send"] == {"type": "game.challenge", "to": "ИИ 9qum",
                                         "rated": False, "tc": {"timeMin": 7.0, "fischer": 2}}
        sent_types = [m.get("type") for m in ws.sent]
        assert sent_types == ["auth"], sent_types  # never actually sent game.challenge
        assert ws.closed is True
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_main_challenge_accepted_plays_and_leaves_no_join_sit_sent():
    """A `game.challenge` acceptance shows up as a table (server-assigned id, unknown
    to us up front) where both we and the opponent are seated -- unlike --mode sit,
    there is no table.join/table.sit to send at all."""
    tmpdir = tempfile.mkdtemp(prefix="ladder_play_test_")
    try:
        session = _session_file(tmpdir)
        paired = _table(id="c1", rated=False,
                         seats=[{"name": "Qonaq_5599", "rating": 2000},
                                {"name": "ИИ 9qum", "rating": 2175}])
        s0 = {"pits": [9] * 18, "kazan": [0, 0], "tuzdyk": [None, None], "to_move": 1,
              "finished": False, "winner": None, "legal_moves": list(range(9))}
        s1 = dict(s0, to_move=0, kazan=[0, 0])
        s2 = dict(s0, to_move=1, kazan=[9, 0], finished=True, winner=0)
        push0 = {"event": "message", "payload": {"id": "c1", "game": s0}}
        push1 = {"event": "message", "payload": {"id": "c1", "game": s1}}
        push2 = {"event": "message", "payload": {"id": "c1", "game": s2}}
        events = [
            AUTH_OK,
            _lobby_evt([]),           # initial lobby snapshot, no tables yet
            _lobby_evt([paired]),     # opponent accepted -> a paired table appears
            push0, push1, push2,
        ]
        argv = ["ladder_play.py", "--mode", "challenge", "--session", session,
                "--log-dir", tmpdir, "--opponent", "ИИ 9qum", "--max-wait", "5"]
        rc, record, ws, engines = _run_main(argv, events, engine_moves=[3])
        assert rc == 0
        assert record["outcome"] == "completed"
        assert record["result"] == "win"
        assert record["our_seat"] == 0
        assert record["target_table"]["id"] == "c1"
        assert record["settings_verification"]["matches"] is True
        kinds = [s["type"] for s in ws.sent]
        assert kinds == ["auth", "game.challenge", "game.move", "table.leave"], kinds
        assert ws.closed is True
        assert engines[0].started and engines[0].stopped
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_main_challenge_ignored_reports_declined_and_sends_only_auth_and_challenge():
    tmpdir = tempfile.mkdtemp(prefix="ladder_play_test_")
    try:
        session = _session_file(tmpdir)
        events = [AUTH_OK, _lobby_evt([])]  # opponent never shows up seated with us
        argv = ["ladder_play.py", "--mode", "challenge", "--session", session,
                "--log-dir", tmpdir, "--opponent", "ИИ 9qum", "--max-wait", "0.3"]
        rc, record, ws, _engines = _run_main(argv, events)
        assert rc == 0
        assert record["outcome"] == "opponent_declined_or_ignored"
        sent_types = [m.get("type") for m in ws.sent]
        # nothing to clean up -- we never held a table (no join/sit, no leave/resign)
        assert sent_types == ["auth", "game.challenge"], sent_types
        assert ws.closed is True
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_main_invite_accepted_unrated_table_matches_request():
    """--unrated is the default; when the server's readback of our created table
    genuinely says rated=False, settings_verification must report a match.

    Event shapes mirror a real observed 9qum exchange: table.create's own readback
    has seats=[None, None] and only a `creator` field (the creator is NOT auto-seated
    -- see find_table_by_creator's docstring), so main() must itself send table.sit
    before table.invite; the opponent then appears in the OTHER seat once accepted."""
    tmpdir = tempfile.mkdtemp(prefix="ladder_play_test_")
    try:
        session = _session_file(tmpdir)
        created = _table(id="inv1", rated=False, access="open", creator="Qonaq_5599",
                          seats=[None, None])
        accepted = dict(created, seats=[{"name": "Qonaq_5599", "rating": 2000},
                                         {"name": "ИИ 9qum", "rating": 2175}])
        s0 = {"pits": [9] * 18, "kazan": [0, 0], "tuzdyk": [None, None], "to_move": 1,
              "finished": False, "winner": None, "legal_moves": list(range(9))}
        s1 = dict(s0, to_move=0, kazan=[0, 0])
        s2 = dict(s0, to_move=1, kazan=[9, 0], finished=True, winner=0)
        push0 = {"event": "message", "payload": {"id": "inv1", "game": s0}}
        push1 = {"event": "message", "payload": {"id": "inv1", "game": s1}}
        push2 = {"event": "message", "payload": {"id": "inv1", "game": s2}}
        events = [
            AUTH_OK,
            _lobby_evt([]),             # initial lobby
            {"event": "message", "payload": {"type": "table.state", **created}},  # readback
            _lobby_evt([accepted]),     # we sat + opponent joined -> accepted
            push0, push1, push2,
        ]
        argv = ["ladder_play.py", "--mode", "invite", "--unrated", "--session", session,
                "--log-dir", tmpdir, "--opponent", "ИИ 9qum", "--max-wait", "5"]
        rc, record, ws, engines = _run_main(argv, events, engine_moves=[3])
        assert rc == 0
        assert record["outcome"] == "completed"
        assert record["result"] == "win"
        assert record["our_seat"] == 0
        v = record["settings_verification"]
        assert v == {"requested_rated": False, "server_rated": False, "access": "open",
                     "tc": {"timeMin": 7, "fischer": 2}, "matches": True}
        kinds = [s["type"] for s in ws.sent]
        assert kinds == ["auth", "table.create", "table.sit", "table.invite",
                          "game.move", "table.leave"], kinds
        assert ws.closed is True
        assert engines[0].started and engines[0].stopped
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_main_invite_server_returns_rated_table_reports_mismatch_and_leaves():
    """The main open question this task exists to answer: if we ask for --unrated and
    the server hands back a table with rated=True anyway, that must be reported as a
    mismatch (not silently trusted), and since nobody ever joins in this script, cleanup
    must leave/cancel the table we created without sending a meaningless game.resign
    (no game -- and so no second player -- ever existed)."""
    tmpdir = tempfile.mkdtemp(prefix="ladder_play_test_")
    try:
        session = _session_file(tmpdir)
        created_rated = _table(id="inv2", rated=True, access="open", creator="Qonaq_5599",
                                seats=[None, None])
        events = [
            AUTH_OK,
            _lobby_evt([]),
            # server recorded rated=True despite --unrated
            {"event": "message", "payload": {"type": "table.state", **created_rated}},
        ]
        argv = ["ladder_play.py", "--mode", "invite", "--unrated", "--session", session,
                "--log-dir", tmpdir, "--opponent", "ИИ 9qum", "--max-wait", "0.3"]
        rc, record, ws, _engines = _run_main(argv, events)
        assert rc == 0
        assert record["outcome"] == "opponent_declined_or_ignored"
        v = record["settings_verification"]
        assert v["matches"] is False
        assert v["requested_rated"] is False
        assert v["server_rated"] is True
        kinds = [s["type"] for s in ws.sent]
        # table.sit (our own seat) + table.leave to cancel the table we created, but NO
        # game.resign -- no game (and no second player) ever existed for this run.
        assert kinds == ["auth", "table.create", "table.sit", "table.invite", "table.leave"], kinds
        assert ws.closed is True
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


TESTS = [
    test_find_target_table_matches_open_waiting_free_seat,
    test_find_target_table_picks_correct_free_seat_when_bot_is_seat0,
    test_find_target_table_rejects_closed_table,
    test_find_target_table_rejects_non_open_access_even_if_not_marked_closed,
    test_find_target_table_rejects_status_playing,
    test_find_target_table_rejects_wrong_opponent_name,
    test_find_target_table_rejects_both_seats_full,
    test_find_target_table_returns_none_on_empty_lobby,
    test_find_target_table_picks_first_of_several_matches,
    test_summarize_table_shape,
    test_summarize_lobby_lists_opponent_tables_even_if_unjoinable,
    test_extract_game_state_from_direct_table_push,
    test_extract_game_state_from_nested_table_key,
    test_extract_game_state_from_tables_list,
    test_extract_game_state_inline_pits_no_wrapper,
    test_extract_game_state_returns_none_for_unrelated_table,
    test_extract_game_state_ignores_non_message_events,
    test_extract_game_state_ignores_message_without_pits,
    test_is_error_event_detects_explicit_error_type,
    test_is_error_event_detects_generic_error_key,
    test_is_error_event_ignores_normal_message,
    test_is_error_event_ignores_non_message_events,
    test_is_error_event_detects_bridge_level_error,
    test_read_remaining_ms_list_form,
    test_read_remaining_ms_dict_str_keys,
    test_read_remaining_ms_dict_int_keys,
    test_read_remaining_ms_missing_field_returns_none,
    test_read_remaining_ms_short_list_returns_none,
    test_read_remaining_ms_bad_type_returns_none,
    test_compute_move_ms_normal_scales_with_remaining,
    test_compute_move_ms_clamps_low,
    test_compute_move_ms_clamps_high,
    test_compute_move_ms_fallback_when_clock_unreadable,
    test_diff_move_detects_our_move_with_exact_hole,
    test_diff_move_detects_opponent_move_hole_unknown,
    test_diff_move_returns_none_when_to_move_unchanged,
    test_diff_move_returns_none_when_prev_to_move_missing,
    test_result_for_us_win,
    test_result_for_us_loss,
    test_result_for_us_draw_on_minus_one,
    test_result_for_us_draw_on_none,
    test_find_table_by_id_direct_payload,
    test_find_table_by_id_nested_table_key,
    test_find_table_by_id_from_tables_list,
    test_find_table_by_id_returns_none_when_absent,
    test_find_table_by_creator_matches_even_with_no_seats_filled,
    test_find_table_by_creator_from_tables_list,
    test_find_table_by_creator_returns_none_when_absent,
    test_find_table_with_both_seated_matches_paired_table,
    test_find_table_with_both_seated_none_when_opponent_absent,
    test_build_settings_shape,
    test_build_settings_coerces_truthy_rated,
    test_verify_settings_recorded_matches_when_equal,
    test_verify_settings_recorded_flags_mismatch,
    test_wsclient_log_format_records_direction_and_payload,
    test_wsclient_send_writes_wire_format_to_bridge_stdin_and_logs_it,
    test_wsclient_send_raises_if_socket_never_opens,
    test_wsclient_send_proceeds_once_opened,
    test_wait_for_type_returns_first_matching_event,
    test_wait_for_type_raises_on_error_event,
    test_wait_for_type_raises_on_bridge_eof,
    test_wait_for_type_times_out_to_none,
    test_main_dry_run_reports_target_and_never_joins,
    test_main_opponent_unavailable_within_max_wait_does_not_join,
    test_main_full_game_plays_moves_records_result_and_leaves_cleanly,
    test_main_illegal_engine_move_aborts_resigns_and_leaves,
    test_main_dry_run_challenge_reports_settings_and_never_sends_challenge,
    test_main_challenge_accepted_plays_and_leaves_no_join_sit_sent,
    test_main_challenge_ignored_reports_declined_and_sends_only_auth_and_challenge,
    test_main_invite_accepted_unrated_table_matches_request,
    test_main_invite_server_returns_rated_table_reports_mismatch_and_leaves,
]


if __name__ == "__main__":
    for t in TESTS:
        t()
    print(f"OK: ladder_play.py table-selection + state-extraction + clock-budget + "
          f"main() integration ({len(TESTS)}/{len(TESTS)})")
