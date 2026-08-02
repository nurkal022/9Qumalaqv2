#!/usr/bin/env python3
"""Play exactly ONE live game against a specific opponent bot on 9qum.com's real
websocket ladder (not the analysis-board API tools/9qum/match.py uses).

This is live rated play on someone else's server, so this tool is deliberately
conservative: one websocket connection for the whole run, one table, one game, a
human-plausible pause before every move, a time budget that cannot lose on the clock,
and a clean table.leave + socket close on every exit path -- success, "opponent not
available", or an unexpected error.

Transport: Python has no websocket library available in this environment (pip is
externally-managed here and no `websockets`/`websocket-client` package is installed),
while Node 18 + the `ws` package resolve globally and are a mature, well-tested framing
implementation -- worth trusting over a hand-rolled one for a real rated game. See
`ws_bridge.js` in this directory: it is a dumb line-JSON relay with no protocol logic
of its own (no auth, no table/game handling); everything below -- auth, table
selection, move timing, logging, cleanup -- lives here in Python where it can be
exercised offline (see test_ladder_play.py).

Usage:
  python3.12 tools/9qum/ladder_play.py --dry-run
  python3.12 tools/9qum/ladder_play.py --opponent "ИИ 9qum" --max-wait 180

Three `--mode`s reach a seat differently, then share the same play loop:
  sit       (default, today's behaviour) join a table where the opponent is already
            waiting with an open, non-closed seat -- see find_target_table.
  challenge send `game.challenge` to --opponent directly from the lobby; the opponent
            may accept (a table appears with both of us seated) or never respond.
  invite    create our own table (`table.create`) with settings we choose (in
            particular --rated/--unrated, default unrated), then `table.invite`
            --opponent to it. Either way, if the opponent never accepts within
            --max-wait, this is reported as `opponent_declined_or_ignored` and cleaned
            up rather than retried or left hanging.

Position/engine plumbing is reused, not reimplemented: `pos_from_state` is imported
from tools/9qum/match.py (the already-tested tuzdyk absolute->relative conversion), and
the engine subprocess wrapper is tools/playok/engine.py's `Engine`.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "playok"))
from engine import Engine  # noqa: E402  (repo-local helper)
sys.path.insert(0, str(REPO / "tools" / "9qum"))
from match import pos_from_state  # noqa: E402  (reuse the tested tuzdyk conversion)

WS_URL = "wss://9qum.com/ws"
DEFAULT_OPPONENT = "ИИ 9qum"
DEFAULT_SESSION = REPO / "data" / "9qum" / "session.json"
DEFAULT_ENGINE = REPO / "models" / "nets" / "nnue_v2" / "eng_v4" / "togyzkumalaq-engine"
DEFAULT_LOG_DIR = REPO / ".superpowers" / "sdd" / "2026-07-31-beat-9qum-phase-a"
BRIDGE_JS = Path(__file__).resolve().with_name("ws_bridge.js")

# Not exposed as flags -- these are safety/liveness constants, not knobs the brief
# asked to tune from the command line.
AUTH_TIMEOUT_S = 15.0
LOBBY_TIMEOUT_S = 15.0
JOIN_CONFIRM_TIMEOUT_S = 15.0
IDLE_POLL_S = 5.0          # how often the play loop wakes up to check the stall clock
STALL_TIMEOUT_S = 90.0     # no message of ANY kind (not just game-relevant) for this long -> abort
OPEN_TIMEOUT_S = 15.0      # TCP+TLS+WS handshake budget before WsClient.send() gives up
MAX_PLIES = 400            # sanity bound; a real game ends far sooner
PRE_MOVE_DELAY_RANGE = (0.3, 0.9)   # human-plausible pause before sending our move


# --------------------------------------------------------------------------
# Websocket transport: a single Node `ws_bridge.js` subprocess, relayed over
# line-delimited JSON on its stdin/stdout. See the module docstring for why.
# --------------------------------------------------------------------------
class WsClient:
    """Owns the ONE websocket connection for this run. Every message sent or received
    is appended to `log_fh` as one JSON line (with a wall-clock timestamp and
    direction), which is the "log every sent and received message" requirement --
    logging happens here, centrally, rather than being scattered through the caller.
    """

    def __init__(self, log_fh, url=WS_URL, bridge_path=BRIDGE_JS, node_bin="node"):
        self.log_fh = log_fh
        self._log_lock = Lock()
        self.queue: Queue = Queue()
        # The bridge needs a real TCP+TLS+WS handshake before it can send anything
        # (ws_bridge.js rejects a `send` command with readyState != OPEN); send() waits
        # on this so no caller has to remember to wait for the "open" event itself --
        # a real dry-run hit exactly this race (auth sent before open, silently
        # dropped, then a spurious "no auth.ok" timeout) before this event existed.
        self._opened = Event()
        self.proc = subprocess.Popen(
            [node_bin, str(bridge_path), url],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        self.closed = False
        self._reader = Thread(target=self._read_stdout, daemon=True)
        self._reader.start()
        self._stderr_reader = Thread(target=self._read_stderr, daemon=True)
        self._stderr_reader.start()

    def _log(self, direction: str, obj: dict) -> None:
        rec = {"ts": time.time(), "dir": direction, **obj}
        line = json.dumps(rec, ensure_ascii=False)
        with self._log_lock:
            self.log_fh.write(line + "\n")
            self.log_fh.flush()

    def _read_stdout(self) -> None:
        try:
            for raw in self.proc.stdout:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    evt = json.loads(raw)
                except json.JSONDecodeError:
                    self._log("recv_unparsed", {"raw": raw})
                    continue
                self._log("recv", evt)
                if evt.get("event") == "open":
                    self._opened.set()
                self.queue.put(evt)
        except Exception as exc:  # bridge pipe died mid-read
            self._log("reader_error", {"error": str(exc)})
        # stdout EOF (bridge exited) -> tell the consumer loop so it doesn't block forever
        self.queue.put({"event": "_bridge_eof"})

    def _read_stderr(self) -> None:
        try:
            for raw in self.proc.stderr:
                raw = raw.rstrip()
                if raw:
                    self._log("bridge_stderr", {"raw": raw})
        except Exception:
            pass

    def send(self, payload: dict) -> None:
        if not self._opened.wait(timeout=OPEN_TIMEOUT_S):
            raise RuntimeError(f"websocket did not open within {OPEN_TIMEOUT_S}s; refusing to send")
        self._log("send", {"payload": payload})
        line = json.dumps({"cmd": "send", "payload": payload}, ensure_ascii=False)
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def next_event(self, timeout=None):
        try:
            return self.queue.get(timeout=timeout)
        except Empty:
            return None

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write(json.dumps({"cmd": "close"}) + "\n")
            self.proc.stdin.flush()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


# --------------------------------------------------------------------------
# Pure logic: table selection, state extraction, clock budgeting, move-diffing.
# Kept free of any I/O so test_ladder_play.py can exercise them without a socket.
# --------------------------------------------------------------------------
def find_target_table(lobby_payload: dict, opponent_name: str):
    """First normal, joinable table in this lobby snapshot: waiting, open access, not
    closed, one seat empty and the OTHER seat occupied by exactly `opponent_name`.

    Deliberately strict (access == "open", closed is falsy) -- a live probe of the real
    lobby found a table with this exact opponent name that was status="waiting" but
    access="closed"/closed=true (almost certainly a private challenge slot, not a public
    seat); joining that one would not be the "normal case" the brief describes, so it
    must not match here even though the name matches.
    """
    for t in (lobby_payload or {}).get("tables", []) or []:
        if t.get("status") != "waiting":
            continue
        if t.get("closed"):
            continue
        if t.get("access") != "open":
            continue
        seats = t.get("seats") or []
        if len(seats) != 2:
            continue
        for seat_idx in (0, 1):
            if seats[seat_idx] is not None:
                continue
            other = seats[1 - seat_idx]
            if other and other.get("name") == opponent_name:
                return t, seat_idx
    return None, None


def summarize_table(t: dict) -> dict:
    seats = t.get("seats") or [None, None]
    return {
        "id": t.get("id"), "no": t.get("no"), "status": t.get("status"),
        "access": t.get("access"), "closed": t.get("closed"), "rated": t.get("rated"),
        "stake": t.get("stake"), "tc": t.get("tc"), "spectators": t.get("spectators"),
        "seats": [([s.get("name"), s.get("rating")] if s else None) for s in seats],
    }


def summarize_lobby(lobby_payload: dict, opponent_name: str) -> dict:
    tables = (lobby_payload or {}).get("tables", []) or []
    return {
        "online": (lobby_payload or {}).get("online"),
        "num_tables": len(tables),
        "tables": [summarize_table(t) for t in tables],
        "opponent_tables": [
            summarize_table(t) for t in tables
            if any(s and s.get("name") == opponent_name for s in (t.get("seats") or []))
        ],
    }


def extract_game_state(evt: dict, table_id: str):
    """Best-effort pull of the live `game` dict (pits/kazan/tuzdyk/to_move/...) for
    `table_id` out of an arbitrary pushed message, without assuming a specific message
    `type` string -- the brief is explicit that field names beyond the documented ones
    should not be assumed. Looks for a dict that either IS a table matching `table_id`
    (checked via id/tableId/table_id) with a nested "game", or a dict that already has
    "pits" inline; scans the top level, a "table" key, and a "tables" list so it works
    whether the push wraps a single table or the whole lobby.
    """
    if not evt or evt.get("event") != "message":
        return None
    payload = evt.get("payload") or {}
    if not isinstance(payload, dict):
        return None

    candidates = []
    if payload.get("id") == table_id or payload.get("tableId") == table_id or payload.get("table_id") == table_id:
        candidates.append(payload)
    table = payload.get("table")
    if isinstance(table, dict) and table.get("id") == table_id:
        candidates.append(table)
    for t in payload.get("tables", []) or []:
        if isinstance(t, dict) and t.get("id") == table_id:
            candidates.append(t)

    for cand in candidates:
        game = cand.get("game")
        if isinstance(game, dict) and "pits" in game:
            return game
        if "pits" in cand:
            return cand
    return None


def is_error_event(evt: dict) -> str | None:
    """Returns a human-readable error string if this event is either a bridge-level
    error (ws_bridge.js's own `{"event":"error",...}`, e.g. a send attempted before the
    socket finished opening) or a server-side error push wrapped in `{"event":
    "message", ...}`, else None. Generic on the server's message shape on purpose (its
    `type` string is not guaranteed)."""
    if not evt:
        return None
    if evt.get("event") == "error":
        return str(evt.get("error"))
    if evt.get("event") != "message":
        return None
    payload = evt.get("payload") or {}
    if not isinstance(payload, dict):
        return None
    t = payload.get("type")
    if t == "error" or "error" in payload:
        return str(payload.get("message") or payload.get("error") or payload)
    return None


def read_remaining_ms(game: dict, our_seat: int):
    """Best-effort read of our seat's remaining clock (ms) out of `game["clocks"]`,
    whose exact shape the brief does not guarantee. Returns None if it cannot be
    confidently parsed, so the caller falls back to the brief's fixed-2000ms rule for
    exactly that case -- never guesses in a way that could overrun the clock (every
    fallback/parse-failure path yields a SHORTER budget than trusting a big number
    blindly would, never a longer one).
    """
    clocks = game.get("clocks")
    if clocks is None:
        return None
    try:
        if isinstance(clocks, (list, tuple)):
            if len(clocks) <= our_seat:
                return None
            v = clocks[our_seat]
        elif isinstance(clocks, dict):
            v = clocks.get(str(our_seat), clocks.get(our_seat))
        else:
            return None
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError, IndexError):
        return None


def compute_move_ms(remaining_ms, *, lo=500, hi=4000, divisor=25.0, fallback=2000) -> int:
    """`min(4000, max(500, remaining_ms/25))`, falling back to a fixed 2000ms when the
    clock could not be read -- exactly the rule the brief specifies."""
    if remaining_ms is None:
        return int(fallback)
    return int(min(hi, max(lo, remaining_ms / divisor)))


def diff_move(prev_game: dict, game: dict, our_seat: int, last_sent_hole):
    """Best-effort description of the single move that turned prev_game into game.

    Seat attribution is exact: togyzkumalaq always passes the turn (no extra-turn
    rule), so a to_move flip is exactly one move by whoever `to_move` was before the
    flip -- the same invariant tools/playok/engine.py's apply_move relies on. The hole
    played is exact for OUR OWN moves (we log what we sent, `last_sent_hole`) and
    best-effort/None for the opponent's, since the brief promises `pits`/`kazan`/
    `tuzdyk`/`to_move`/... but not an explicit "last move" field. The raw send/recv log
    still has full fidelity for audit either way.
    """
    mover = prev_game.get("to_move")
    if mover is None or mover == game.get("to_move"):
        return None
    if mover == our_seat:
        return {"seat": mover, "hole": last_sent_hole, "hole_source": "sent"}
    return {"seat": mover, "hole": None, "hole_source": "not_observed"}


def result_for_us(game: dict, our_seat: int) -> str:
    winner = game.get("winner")
    if winner is None or winner == -1:
        return "draw"
    return "win" if winner == our_seat else "loss"


# --------------------------------------------------------------------------
# Table acquisition for --mode challenge/invite: unlike `sit`, the table involved
# isn't known from a pre-existing lobby entry -- it either doesn't exist yet
# (challenge, until accepted) or was just created by us (invite). These helpers are
# deliberately tolerant of the exact push shape (a full lobby snapshot's `tables`
# list, a single table wrapped in a `table` key, or a message that already looks like
# a table) for the same reason extract_game_state is: the brief documents fields, not
# a specific `type` string for every push.
# --------------------------------------------------------------------------
def _candidate_tables(payload: dict):
    """Every table-shaped dict reachable from an arbitrary pushed payload."""
    if not isinstance(payload, dict):
        return
    if "seats" in payload or "id" in payload:
        yield payload
    table = payload.get("table")
    if isinstance(table, dict):
        yield table
    for t in payload.get("tables", []) or []:
        if isinstance(t, dict):
            yield t


def find_table_by_id(payload: dict, table_id: str):
    """First candidate table in `payload` whose id matches `table_id`, or None. Used
    by --mode invite to re-read our own just-created table (to detect the opponent
    joining it) once its id is already known."""
    for t in _candidate_tables(payload):
        if t.get("id") == table_id:
            return t
    return None


def find_table_by_creator(payload: dict, name: str):
    """First candidate table in `payload` whose `creator` field is `name`, or None.
    Used by --mode invite right after table.create, before the new table's id is known
    to us. NOT seat occupancy: a real live probe of 9qum.com found that table.create
    does NOT auto-seat the creator -- the table.state push it triggers came back with
    `seats: [None, None]` and a separate top-level `creator` field, so a table we just
    created must be identified by that field, not by scanning for our name in a seat
    (which find_table_with_both_seated/find_target_table rightly do for tables OTHER
    parties already occupy)."""
    for t in _candidate_tables(payload):
        if t.get("creator") == name:
            return t
    return None


def find_table_with_both_seated(payload: dict, name_a: str, name_b: str):
    """First candidate table in `payload` where both `name_a` and `name_b` occupy a
    seat. Returns (table, name_a's seat index) or (None, None). Used by --mode
    challenge, whose table (if the opponent accepts) is server-created with an id we
    never chose, so it must be found by who's sitting at it rather than by id."""
    for t in _candidate_tables(payload):
        seats = t.get("seats") or []
        names = [s.get("name") if s else None for s in seats]
        if name_a in names and name_b in names:
            return t, names.index(name_a)
    return None, None


def build_settings(rated: bool, tc_minutes: float, tc_fischer: int) -> dict:
    """Minimal table-settings object for game.challenge / table.create. The brief
    notes the real client spreads its own table-settings UI state into these calls and
    says to try the minimal form first rather than re-deriving that state; `tc`'s
    shape (timeMin/fischer) matches every real table this project has observed in a
    9qum lobby snapshot (including both the closed 'ИИ 9qum' table and the open
    'ИИ 9qumалак' one, both tc={"timeMin": 7, "fischer": 2})."""
    return {"rated": bool(rated), "tc": {"timeMin": tc_minutes, "fischer": tc_fischer}}


def verify_settings_recorded(table: dict, requested_rated: bool) -> dict:
    """What the server actually recorded for a table we created/joined via challenge
    or invite, compared to what we asked for. The brief is explicit that the
    requested settings must never be assumed honoured -- whether unrated play is even
    possible at all is the main thing this task exists to establish, so a mismatch
    must be surfaced loudly, not silently."""
    server_rated = table.get("rated")
    return {
        "requested_rated": requested_rated,
        "server_rated": server_rated,
        "access": table.get("access"),
        "tc": table.get("tc"),
        "matches": server_rated == requested_rated,
    }


# --------------------------------------------------------------------------
# Small blocking helpers built on WsClient.next_event
# --------------------------------------------------------------------------
def wait_for_type(ws: WsClient, type_name: str, timeout: float):
    deadline = time.time() + timeout
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            return None
        evt = ws.next_event(timeout=remaining)
        if evt is None:
            continue
        if evt.get("event") == "_bridge_eof":
            raise RuntimeError("websocket bridge exited unexpectedly while waiting for "
                               f"{type_name!r}")
        err = is_error_event(evt)
        if err:
            raise RuntimeError(f"server error while waiting for {type_name!r}: {err}")
        if evt.get("event") == "message" and evt.get("payload", {}).get("type") == type_name:
            return evt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--opponent", default=DEFAULT_OPPONENT,
                    help="exact display name of the opponent bot to play")
    ap.add_argument("--mode", choices=("sit", "challenge", "invite"), default="sit",
                    help="'sit' (default, unchanged behaviour): join a waiting table "
                         "where the opponent already holds an open, non-closed seat. "
                         "'challenge': send game.challenge to --opponent directly from "
                         "the lobby. 'invite': create our own table (table.create), "
                         "then table.invite --opponent to it.")
    ap.add_argument("--engine", default=str(DEFAULT_ENGINE), help="engine binary path")
    ap.add_argument("--max-wait", type=float, default=180.0,
                    help="seconds to wait for: a joinable table (sit), the opponent to "
                         "accept a challenge (challenge), or the opponent to accept an "
                         "invite (invite)")
    rated_group = ap.add_mutually_exclusive_group()
    rated_group.add_argument("--rated", dest="rated", action="store_true",
                              help="request a rated table (challenge/invite only)")
    rated_group.add_argument("--unrated", dest="rated", action="store_false",
                              help="request an unrated table (challenge/invite only) "
                                   "-- the default, since an unrated game has no side "
                                   "effects on anyone's rating")
    ap.set_defaults(rated=None)
    ap.add_argument("--tc-minutes", type=float, default=7.0,
                    help="base time-control minutes to request for challenge/invite "
                         "settings (default matches every real table this project has "
                         "observed in a 9qum lobby snapshot)")
    ap.add_argument("--tc-fischer", type=int, default=2,
                    help="fischer increment seconds to request for challenge/invite "
                         "settings")
    ap.add_argument("--session", default=str(DEFAULT_SESSION))
    ap.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    ap.add_argument("--ws-url", default=WS_URL)
    ap.add_argument("--node-bin", default="node")
    ap.add_argument("--dry-run", action="store_true",
                    help="sit: observe the lobby and report the table that would be "
                         "joined, never join/sit/move. challenge/invite: report the "
                         "settings and message that would be sent, never actually "
                         "challenge/create/invite.")
    args = ap.parse_args()
    # args.rated defaults to None (neither --rated nor --unrated given) -> unrated,
    # per the brief: an unrated game has no side effects on anyone's rating.
    requested_rated = False if args.rated is None else args.rated

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    tag = "dryrun" if args.dry_run else "game"
    log_path = log_dir / f"ladder_play_{tag}_{run_id}.jsonl"
    record_path = log_dir / f"ladder_play_{tag}_{run_id}_record.json"

    with open(args.session, encoding="utf-8") as f:
        session = json.load(f)
    token = session["token"]

    record = {
        "schema": "ladder_play_game_v1",
        "opponent_requested": args.opponent,
        "mode": args.mode,
        "dry_run": args.dry_run,
        "engine_path": str(args.engine),
        "started_ts": time.time(),
        "outcome": "unknown",
    }

    seated = False
    own_table_created = False
    game_finished = False
    table_id = None
    our_seat = None

    with open(log_path, "a", encoding="utf-8") as log_fh:
        ws = WsClient(log_fh, url=args.ws_url, node_bin=args.node_bin)
        try:
            ws.send({"type": "auth", "token": token})
            auth_evt = wait_for_type(ws, "auth.ok", timeout=AUTH_TIMEOUT_S)
            if auth_evt is None:
                raise RuntimeError(f"no auth.ok within {AUTH_TIMEOUT_S}s; token may be rejected")
            our_name = auth_evt["payload"].get("user", {}).get("name")
            record["our_name"] = our_name
            print(f"authenticated as {our_name!r}")

            lobby_evt = wait_for_type(ws, "lobby", timeout=LOBBY_TIMEOUT_S)
            if lobby_evt is None:
                raise RuntimeError(f"no lobby push within {LOBBY_TIMEOUT_S}s after auth")
            lobby = lobby_evt["payload"]

            # --- 1. acquire a table + seat, per --mode ---
            if args.mode == "sit":
                deadline = time.time() + args.max_wait
                target, seat_idx = find_target_table(lobby, args.opponent)
                while target is None and time.time() < deadline:
                    evt = ws.next_event(timeout=max(0.5, deadline - time.time()))
                    if evt is None:
                        continue
                    if evt.get("event") == "_bridge_eof":
                        raise RuntimeError("websocket bridge exited unexpectedly while waiting for a table")
                    if evt.get("event") == "message" and evt.get("payload", {}).get("type") == "lobby":
                        lobby = evt["payload"]
                        target, seat_idx = find_target_table(lobby, args.opponent)

                record["lobby_snapshot"] = summarize_lobby(lobby, args.opponent)

                if target is None:
                    record["outcome"] = "opponent_unavailable"
                    print(f"'{args.opponent}' was not available at a joinable (waiting/open/"
                          f"not-closed) table within {args.max_wait:.0f}s.")
                    print(json.dumps(record["lobby_snapshot"], ensure_ascii=False, indent=1))
                    return 0

                other_seat = target["seats"][1 - seat_idx]
                print(f"target table: id={target['id']} no={target.get('no')} "
                      f"seat={seat_idx} vs {other_seat.get('name')} (rating {other_seat.get('rating')}) "
                      f"tc={target.get('tc')} rated={target.get('rated')} stake={target.get('stake')}")
                record["target_table"] = target
                record["our_seat"] = seat_idx
                record["time_control"] = target.get("tc")

                if args.dry_run:
                    record["outcome"] = "dry_run_ok"
                    print("DRY RUN: stopping here, no join/sit/move sent.")
                    return 0

                table_id = target["id"]
                our_seat = seat_idx
                ws.send({"type": "table.join", "tableId": table_id})
                time.sleep(0.5)
                ws.send({"type": "table.sit", "tableId": table_id, "seat": our_seat})
                seated = True  # we attempted to occupy a seat; cleanup should try to vacate it either way

                confirmed = False
                confirm_deadline = time.time() + JOIN_CONFIRM_TIMEOUT_S
                while time.time() < confirm_deadline:
                    evt = ws.next_event(timeout=max(0.5, confirm_deadline - time.time()))
                    if evt is None:
                        continue
                    if evt.get("event") == "_bridge_eof":
                        raise RuntimeError("websocket bridge exited unexpectedly while confirming our seat")
                    err = is_error_event(evt)
                    if err:
                        raise RuntimeError(f"server rejected table.join/table.sit: {err}")
                    game = extract_game_state(evt, table_id)
                    payload = evt.get("payload") or {}
                    seats = None
                    if payload.get("id") == table_id:
                        seats = payload.get("seats")
                    elif isinstance(payload.get("table"), dict) and payload["table"].get("id") == table_id:
                        seats = payload["table"].get("seats")
                    if seats and len(seats) > our_seat and seats[our_seat] and seats[our_seat].get("name") == our_name:
                        confirmed = True
                        break
                    if game is not None:
                        # a game push for our table implies the sit succeeded (a table
                        # without both seats filled cannot have a live game object)
                        confirmed = True
                        break
                if not confirmed:
                    raise RuntimeError(f"could not confirm seat {our_seat} at table {table_id} "
                                       f"within {JOIN_CONFIRM_TIMEOUT_S}s")
                print(f"seated: table={table_id} seat={our_seat}")

            elif args.mode == "challenge":
                settings = build_settings(requested_rated, args.tc_minutes, args.tc_fischer)
                record["requested_settings"] = settings
                print(f"mode=challenge opponent={args.opponent!r} settings={settings}")

                if args.dry_run:
                    record["outcome"] = "dry_run_ok"
                    would_send = {"type": "game.challenge", "to": args.opponent, **settings}
                    record["would_send"] = would_send
                    print(f"DRY RUN: would send {would_send}")
                    return 0

                ws.send({"type": "game.challenge", "to": args.opponent, **settings})
                print(f"sent game.challenge to {args.opponent!r}; waiting up to "
                      f"{args.max_wait:.0f}s for acceptance")

                target, seat_idx = None, None
                deadline = time.time() + args.max_wait
                while target is None and time.time() < deadline:
                    evt = ws.next_event(timeout=max(0.5, deadline - time.time()))
                    if evt is None:
                        continue
                    if evt.get("event") == "_bridge_eof":
                        raise RuntimeError("websocket bridge exited unexpectedly while "
                                           "waiting for challenge acceptance")
                    err = is_error_event(evt)
                    if err:
                        raise RuntimeError(f"server error after game.challenge: {err}")
                    if evt.get("event") == "message":
                        payload = evt.get("payload") or {}
                        if payload.get("type") == "lobby":
                            lobby = payload
                        target, seat_idx = find_table_with_both_seated(payload, our_name, args.opponent)

                record["lobby_snapshot"] = summarize_lobby(lobby, args.opponent)

                if target is None:
                    record["outcome"] = "opponent_declined_or_ignored"
                    print(f"'{args.opponent}' did not accept the challenge within "
                          f"{args.max_wait:.0f}s.")
                    return 0

                table_id = target["id"]
                our_seat = seat_idx
                seated = True  # a real two-party table now exists; cleanup should vacate it
                record["target_table"] = target
                record["our_seat"] = our_seat
                record["time_control"] = target.get("tc")
                verification = verify_settings_recorded(target, requested_rated)
                record["settings_verification"] = verification
                print(f"challenge accepted: table={table_id} seat={our_seat} "
                      f"server_rated={verification['server_rated']} "
                      f"requested_rated={verification['requested_rated']} "
                      f"access={verification['access']} tc={verification['tc']}")
                if not verification["matches"]:
                    print(f"WARNING: requested rated={requested_rated} but the server "
                          f"recorded rated={verification['server_rated']} on this table "
                          f"-- our settings were NOT honoured as requested!")

            elif args.mode == "invite":
                settings = build_settings(requested_rated, args.tc_minutes, args.tc_fischer)
                record["requested_settings"] = settings
                print(f"mode=invite opponent={args.opponent!r} settings={settings}")

                if args.dry_run:
                    record["outcome"] = "dry_run_ok"
                    would_send = {"type": "table.create", **settings}
                    record["would_send"] = would_send
                    print(f"DRY RUN: would send {would_send}")
                    return 0

                ws.send({"type": "table.create", **settings})
                our_table = None
                deadline = time.time() + LOBBY_TIMEOUT_S
                while our_table is None and time.time() < deadline:
                    evt = ws.next_event(timeout=max(0.5, deadline - time.time()))
                    if evt is None:
                        continue
                    if evt.get("event") == "_bridge_eof":
                        raise RuntimeError("websocket bridge exited unexpectedly while "
                                           "waiting for table.create to be acknowledged")
                    err = is_error_event(evt)
                    if err:
                        raise RuntimeError(f"server error after table.create: {err}")
                    if evt.get("event") == "message":
                        our_table = find_table_by_creator(evt.get("payload") or {}, our_name)
                if our_table is None:
                    raise RuntimeError(f"table.create sent but no table with "
                                       f"creator={our_name!r} was seen within "
                                       f"{LOBBY_TIMEOUT_S}s")
                table_id = our_table["id"]
                own_table_created = True  # we hold a table now; cleanup must leave/cancel
                                          # it even if the invite is never accepted
                record["created_table"] = our_table
                verification = verify_settings_recorded(our_table, requested_rated)
                record["settings_verification"] = verification
                print(f"table created: id={table_id} server_rated={verification['server_rated']} "
                      f"requested_rated={verification['requested_rated']} "
                      f"access={verification['access']} tc={verification['tc']}")
                if not verification["matches"]:
                    print(f"WARNING: requested rated={requested_rated} but the server "
                          f"recorded rated={verification['server_rated']} on this table "
                          f"-- our settings were NOT honoured as requested!")

                # table.create does NOT auto-seat the creator -- a live probe of
                # 9qum.com came back with seats=[None, None] and only a `creator` field
                # set, so we still have to sit ourselves before anyone can play. Seat 0
                # by convention (the brief does not specify which seat the inviter takes).
                our_seat = 0
                ws.send({"type": "table.sit", "tableId": table_id, "seat": our_seat})
                time.sleep(0.3)

                ws.send({"type": "table.invite", "tableId": table_id, "to": args.opponent})
                print(f"invited {args.opponent!r} to table {table_id}; waiting up to "
                      f"{args.max_wait:.0f}s for acceptance")

                accepted = False
                deadline = time.time() + args.max_wait
                while not accepted and time.time() < deadline:
                    evt = ws.next_event(timeout=max(0.5, deadline - time.time()))
                    if evt is None:
                        continue
                    if evt.get("event") == "_bridge_eof":
                        raise RuntimeError("websocket bridge exited unexpectedly while "
                                           "waiting for invite acceptance")
                    err = is_error_event(evt)
                    if err:
                        raise RuntimeError(f"server error after table.invite: {err}")
                    if evt.get("event") == "message":
                        t = find_table_by_id(evt.get("payload") or {}, table_id)
                        if t is not None:
                            our_table = t
                            seats = t.get("seats") or []
                            names = [s.get("name") if s else None for s in seats]
                            if our_name in names:
                                our_seat = names.index(our_name)
                            if args.opponent in names:
                                accepted = True

                record["target_table"] = our_table
                record["our_seat"] = our_seat
                record["time_control"] = our_table.get("tc")

                if not accepted:
                    record["outcome"] = "opponent_declined_or_ignored"
                    print(f"'{args.opponent}' did not accept the invite within "
                          f"{args.max_wait:.0f}s.")
                    return 0
                seated = True  # opponent joined our table; cleanup should vacate it
                print(f"invite accepted: table={table_id} seat={our_seat}")

            # --- 2. play loop ---
            eng = Engine(Path(args.engine))
            eng.start()
            moves_log = []
            prev_game = None
            last_sent_hole = None
            last_activity = time.time()
            final_game = None
            try:
                while True:
                    evt = ws.next_event(timeout=IDLE_POLL_S)
                    if evt is None:
                        if time.time() - last_activity > STALL_TIMEOUT_S:
                            raise RuntimeError(f"no server activity for {STALL_TIMEOUT_S}s; aborting")
                        continue
                    last_activity = time.time()
                    if evt.get("event") == "_bridge_eof":
                        raise RuntimeError("websocket bridge exited unexpectedly during play")
                    err = is_error_event(evt)
                    if err:
                        raise RuntimeError(f"server error during play: {err}")

                    game = extract_game_state(evt, table_id)
                    if game is None:
                        continue

                    if prev_game is not None:
                        mv_desc = diff_move(prev_game, game, our_seat, last_sent_hole)
                        if mv_desc is not None:
                            mv_desc["ply"] = len(moves_log)
                            mv_desc["resulting_kazan"] = game.get("kazan")
                            mv_desc["resulting_tuzdyk"] = game.get("tuzdyk")
                            moves_log.append(mv_desc)
                            last_sent_hole = None
                            print(f"  ply {mv_desc['ply']}: seat{mv_desc['seat']} "
                                  f"hole={mv_desc['hole']} kazan={mv_desc['resulting_kazan']}")
                    prev_game = game
                    final_game = game

                    if game.get("finished"):
                        game_finished = True
                        break
                    if len(moves_log) >= MAX_PLIES:
                        raise RuntimeError(f"exceeded sanity bound of {MAX_PLIES} plies without a finish")

                    if game.get("to_move") == our_seat:
                        remaining = read_remaining_ms(game, our_seat)
                        move_ms = compute_move_ms(remaining)
                        pos = pos_from_state(game)
                        mv = eng.bestmove(pos, time_ms=move_ms)
                        if isinstance(mv, tuple):
                            raise RuntimeError(f"engine reports terminal ({mv}) but server "
                                               f"game is not finished")
                        legal = game.get("legal_moves")
                        if legal is not None and mv not in legal:
                            raise RuntimeError(f"engine chose move {mv} not in server "
                                               f"legal_moves={legal}")
                        time.sleep(random.uniform(*PRE_MOVE_DELAY_RANGE))
                        ws.send({"type": "game.move", "tableId": table_id, "hole": mv})
                        last_sent_hole = mv
            finally:
                eng.stop()

            record["moves"] = moves_log
            record["plies"] = len(moves_log)
            if final_game is not None:
                record["final_position"] = pos_from_state(final_game)
                record["final_kazan"] = final_game.get("kazan")
                record["final_tuzdyk"] = final_game.get("tuzdyk")
                record["winner_seat"] = final_game.get("winner")
                record["result"] = result_for_us(final_game, our_seat)
            record["outcome"] = "completed"
            print(f"game finished: result={record.get('result')} "
                  f"kazan={record.get('final_kazan')} plies={record['plies']}")
            return 0

        except KeyboardInterrupt:
            record["outcome"] = "interrupted"
            record["error"] = "KeyboardInterrupt"
            print("interrupted; cleaning up", file=sys.stderr)
            return 130
        except Exception as exc:
            # dry_run_ok / opponent_unavailable / completed all `return` from inside the
            # same try block, which runs `finally` but never reaches this `except` --
            # so getting here always means a genuine mid-flow failure.
            record["outcome"] = "error"
            record["error"] = str(exc)
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        finally:
            try:
                if seated and not game_finished and table_id is not None:
                    # We're leaving mid-game (error/interrupt): resign so the game ends
                    # cleanly for the opponent too, instead of just vanishing.
                    ws.send({"type": "game.resign", "tableId": table_id})
                    time.sleep(0.3)
                if (seated or own_table_created) and table_id is not None:
                    # `seated` covers sit/challenge/invite once a real two-party table
                    # exists; `own_table_created` additionally covers an invite table we
                    # created but the opponent never accepted -- that table still needs
                    # to be left/cancelled even though no game (and so no resign) exists.
                    ws.send({"type": "table.leave", "tableId": table_id})
                    time.sleep(0.3)
            except Exception as cleanup_exc:
                print(f"WARNING: cleanup send failed: {cleanup_exc}", file=sys.stderr)
            ws.close()
            record["ended_ts"] = time.time()
            with open(record_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=1)
            print(f"record written: {record_path}")
            print(f"log written: {log_path}")


if __name__ == "__main__":
    sys.exit(main())
