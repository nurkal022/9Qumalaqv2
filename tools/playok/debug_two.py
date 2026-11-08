"""
Two-account end-to-end debug for PlayOK Togyz Kumalak.

Full game flow: private table, configurable settings, auto-rematch, score tracking.

Usage:
  python3 debug_two.py [options]

Options:
  --games N         play N games then stop (default: 3)
  --no-rematch      stop after the first game
  --time-min M      game clock in minutes per side (default: 30; 0 = use server default)
  --increment S     Fischer increment in seconds (default: 0)
  --rated           play rated games (default: non-rated)
  --public          make the table public (default: private)
  --scenario S      full | invite | settings (default: full)

Creds via env or hardcoded defaults:
  A_USER/A_PW  (host, alemgamer)
  B_USER/B_PW  (guest, nurkal022)
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
import threading
from typing import Optional, Callable

from playok import PlayokClient
from engine import Engine, START_POSITION

SKIP_OPS = {25, 51, 27, 31, 30, 32, 33, 28, 23, 22}

# op-93 sub-actions (outgoing)
ACT_REQUEST   = 1   # draw / undo offer (+ type: 1=draw, 2=undo)
ACT_ACCEPT    = 2   # accept the pending request
ACT_REJECT    = 3   # reject the pending request
ACT_RESIGN    = 4   # resign (sent after user confirms)
DRAW_TYPE     = 1
UNDO_TYPE     = 2


def _decode_result(result: int, my_seat: Optional[int] = None) -> str:
    """Interpret op-90 b[4] result code.

    The server sends the SAME raw value to both players.
    From tg.js ye(): positive=white-wins(score=result+1), negative=black-wins(score=-result), 9=draw.
    Each side's client displays it as 'YOU WON/LOST' depending on their seat.
    """
    if result == 9:
        return "DRAW"
    if result < 0:
        winner, score = "BLACK wins", -result
    else:
        winner, score = "WHITE wins", result + 1
    if my_seat is not None:
        white_won = result >= 0
        my_win = (white_won and my_seat == 0) or (not white_won and my_seat == 1)
        perspective = "YOU WIN" if my_win else "YOU LOSE"
        return f"{perspective} — {winner} (score: {score})"
    return f"{winner} (score: {score})"


# ──────────────────────────────────────────────────────────────────────────────
# Per-player state
# ──────────────────────────────────────────────────────────────────────────────

class PlayerState:
    def __init__(self, tag: str, seat: Optional[int] = None):
        self.tag = tag
        self.k: Optional[int] = None          # table number
        self.seat = seat                       # 0=white,1=black (may be set after sit)
        self.invite_k: Optional[int] = None
        self.inviter: Optional[str] = None

        self.seated = False
        self.both_seated = False               # True when op-70 shows both filled
        self.started = False                   # start pressed at least once
        self.game_on = False                   # game currently in progress
        self.game_finished = False             # set when result arrives
        self.result: Optional[int] = None     # raw result from op-90

        self.pos = START_POSITION
        self.pending_request: Optional[int] = None  # type of incoming op-93 request
        self.wins = 0
        self.losses = 0
        self.draws = 0

    # ---- board tracking -----------------------------------------------
    def my_turn(self) -> bool:
        if not self.game_on or self.seat is None:
            return False
        return int(self.pos.split("/")[-1]) == self.seat

    def apply_move(self, pit: int):
        self.pos = Engine.apply_move(self.pos, pit)

    def legal_moves(self) -> list[int]:
        return Engine.legal_moves(self.pos)

    def reset_game(self):
        self.pos = START_POSITION
        self.game_on = False
        self.game_finished = False
        self.result = None
        self.pending_request = None


def legal_mask_from_op90(i: list) -> Optional[int]:
    for k in range(2, len(i) - 1):
        if i[k] == 5 and isinstance(i[k + 1], int) and 0 <= i[k + 1] <= 511:
            return i[k + 1]
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Frame handler factory
# ──────────────────────────────────────────────────────────────────────────────

def make_handler(
    client: PlayokClient,
    st: PlayerState,
    on_play: Optional[Callable] = None,
    on_both_seated: Optional[Callable] = None,
    on_game_end: Optional[Callable] = None,
    on_draw_request: Optional[Callable] = None,
):
    def on(i, s):
        op = i[0] if i else -1
        if op not in SKIP_OPS:
            print(f"[{st.tag}] op={op:<3}  i={i}  s={s}", flush=True)

        # ── track table K ──────────────────────────────────────────────
        if op == 84 and len(i) > 1:
            if st.k is None or not st.started:
                st.k = i[1]
        elif op == 29 and s and str(s[-1]).startswith("#"):
            try:
                k = int(str(s[-1])[1:])
                if st.k is None or not st.started:
                    st.k = k
            except ValueError:
                pass

        # ── incoming invite ────────────────────────────────────────────
        if op == 75 and len(i) > 1:
            st.invite_k = i[1]
            st.inviter = s[1] if len(s) > 1 else "?"
            print(f"[{st.tag}] *** INVITE from {st.inviter} → table #{st.invite_k} ***", flush=True)

        # ── seat state from op-88 [88, K, flags, ?, seat] ─────────────
        if (op == 88 and len(i) >= 5 and st.k and i[1] == st.k
                and i[4] in (0, 1)):
            if not st.seated or st.seat is None:
                st.seat = i[4]
                st.seated = True

        # ── table occupancy (op-70) ────────────────────────────────────
        if (op == 70 and len(i) > 1 and st.k and i[1] == st.k
                and len(s) >= 3):
            both = bool(s[1] and s[2])
            if both and not st.both_seated and not st.started:
                st.both_seated = True
                print(f"[{st.tag}] both seated: {s[1]} vs {s[2]}", flush=True)
                if on_both_seated:
                    on_both_seated(client, st)

        # ── move echoed (op-92) ────────────────────────────────────────
        if op == 92 and len(i) >= 3 and st.game_on:
            cell = i[2]
            hole = cell - 10 if cell >= 10 else cell
            if 0 <= hole <= 8:
                try:
                    st.apply_move(hole)
                    print(f"[{st.tag}] move hole {hole+1} → pos={st.pos}", flush=True)
                except Exception as e:
                    print(f"[{st.tag}] !! apply_move error: {e}", flush=True)
            if on_play and st.my_turn() and not st.game_finished:
                on_play(client, st)

        # ── game state (op-90) ─────────────────────────────────────────
        if op == 90 and len(i) >= 3 and st.k and i[1] == st.k:
            state_tag = i[2]

            # game just started
            if state_tag >= 7 and not st.game_on and st.started:
                st.reset_game()
                st.game_on = True
                print(f"[{st.tag}] *** GAME ON (state_tag={state_tag}) ***", flush=True)

            # game result (when state_count >= 2 and result != 8)
            if len(i) >= 5 and state_tag >= 2 and st.game_on and not st.game_finished:
                result = i[4]
                if result != 8:
                    st.game_finished = True
                    st.game_on = False
                    st.result = result
                    if result == 9:
                        st.draws += 1
                    elif result > 0:
                        st.wins += 1
                    else:
                        st.losses += 1
                    print(f"[{st.tag}] *** GAME OVER: {_decode_result(result, st.seat)} "
                          f"(raw={result}) | W:{st.wins} L:{st.losses} D:{st.draws} ***",
                          flush=True)
                    if on_game_end:
                        on_game_end(client, st)

            # legal mask → it's our turn
            mask = legal_mask_from_op90(i)
            if mask is not None and on_play and st.game_on and not st.game_finished:
                if st.my_turn():
                    on_play(client, st)

        # ── request (op-93): draw / undo offer ────────────────────────
        if op == 93 and len(i) >= 3 and st.k and i[1] == st.k:
            req_type = i[2]
            if req_type == ACT_REQUEST and len(i) >= 4:
                st.pending_request = i[3]
                print(f"[{st.tag}] incoming request type={i[3]}", flush=True)
                if on_draw_request:
                    on_draw_request(client, st, i[3])

    return on


# ──────────────────────────────────────────────────────────────────────────────
# Table setup helpers
# ──────────────────────────────────────────────────────────────────────────────

def configure_table(client: PlayokClient, k: int, *,
                    private: bool = True,
                    rated: bool = False,
                    time_min: int = 30,
                    increment_s: int = 0):
    """Apply standard table settings via op-82."""
    # ttype: 2=private, 0=public
    client.send([82, k, 2 if private else 0], ["ttype"])
    time.sleep(0.3)
    # gtype: 0=non-rated, 1=rated
    client.send([82, k, 0 if not rated else 1], ["gtype"])
    time.sleep(0.3)
    # tg: game time in minutes (0 may mean "unlimited" on PlayOK; use 0 to try)
    if time_min == 0:
        client.send([82, k, 0], ["tg"])
    else:
        client.send([82, k, time_min], ["tg"])
    time.sleep(0.3)
    # tm: Fischer increment in seconds
    client.send([82, k, increment_s], ["tm"])
    time.sleep(0.3)
    label_parts = [
        "PRIVATE" if private else "PUBLIC",
        "non-rated" if not rated else "RATED",
        f"{time_min}m" if time_min else "unlimited",
        f"+{increment_s}s",
    ]
    print(f"[setup] K={k} configured: {' | '.join(label_parts)}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Game actions
# ──────────────────────────────────────────────────────────────────────────────

def press_start(client: PlayokClient, st: PlayerState):
    print(f"[{st.tag}] pressing START at #{st.k}", flush=True)
    st.started = True
    st.both_seated = False   # reset for next game detection
    client.start(st.k)


def play_random_move(client: PlayokClient, st: PlayerState,
                     resign_after: int = 0):
    """Play a random legal move. If resign_after > 0, resign once we've played that many moves."""
    if not hasattr(st, '_moves_played'):
        st._moves_played = 0
    if resign_after > 0 and st._moves_played >= resign_after:
        print(f"[{st.tag}] resign_after={resign_after} reached — resigning", flush=True)
        # op-93: [93, K, 4, deciseconds]
        client.send([93, st.k, 4, 0], None)
        return
    moves = st.legal_moves()
    if not moves:
        print(f"[{st.tag}] no legal moves", flush=True)
        return
    hole = random.choice(moves)
    st._moves_played += 1
    print(f"[{st.tag}] PLAYING hole {hole+1} (pit {hole})  (move #{st._moves_played})", flush=True)
    client.move(st.k, hole, think_ds=5)


# ──────────────────────────────────────────────────────────────────────────────
# Rematch coordinator
# ──────────────────────────────────────────────────────────────────────────────

class RematchCoordinator:
    """Triggers both players to press start after each game ends."""
    def __init__(self, a: PlayokClient, b: PlayokClient,
                 a_st: PlayerState, b_st: PlayerState,
                 max_games: int = 3, delay_s: float = 3.0):
        self.a, self.b = a, b
        self.a_st, self.b_st = a_st, b_st
        self.max_games = max_games
        self.delay_s = delay_s
        self.games_played = 0
        self._lock = threading.Lock()
        self._pending = 0   # count of sides that have signalled game-over this round

    def on_game_end(self, client: PlayokClient, st: PlayerState):
        with self._lock:
            self._pending += 1
            self.games_played = max(
                # crude: use whichever tracker has higher game count
                self.a_st.wins + self.a_st.losses + self.a_st.draws,
                self.b_st.wins + self.b_st.losses + self.b_st.draws,
            )
            if self._pending < 2:
                return   # wait for the other side to also detect game-over
            self._pending = 0

        if self.games_played >= self.max_games:
            print(f"[rematch] {self.games_played}/{self.max_games} games done — stopping.",
                  flush=True)
            return

        print(f"[rematch] game {self.games_played}/{self.max_games} done — "
              f"rematch in {self.delay_s}s ...", flush=True)
        threading.Thread(target=self._do_rematch, daemon=True).start()

    def _do_rematch(self):
        time.sleep(self.delay_s)
        print("[rematch] pressing START on both sides", flush=True)
        # reset state for both
        for st in (self.a_st, self.b_st):
            st.both_seated = False
            st.started = True
        self.a.start(self.a_st.k)
        time.sleep(0.3)
        self.b.start(self.b_st.k)


# ──────────────────────────────────────────────────────────────────────────────
# Scenarios
# ──────────────────────────────────────────────────────────────────────────────

def scenario_full(a: PlayokClient, b: PlayokClient,
                  a_st: PlayerState, b_st: PlayerState,
                  cfg: argparse.Namespace):
    max_games = 1 if cfg.no_rematch else cfg.games
    rematch = RematchCoordinator(a, b, a_st, b_st,
                                 max_games=max_games, delay_s=3.0)
    resign_after = cfg.resign_after

    def on_end(client, st):
        rematch.on_game_end(client, st)

    def on_start(client, st):
        if hasattr(st, '_moves_played'):
            st._moves_played = 0   # reset move counter for rematch
        press_start(client, st)

    def on_play_a(client, st):
        play_random_move(client, st, resign_after=resign_after)

    def on_play_b(client, st):
        play_random_move(client, st, resign_after=resign_after)

    a.on_frame = make_handler(a, a_st,
                              on_play=on_play_a,
                              on_both_seated=on_start,
                              on_game_end=on_end)
    b.on_frame = make_handler(b, b_st,
                              on_play=on_play_b,
                              on_both_seated=on_start,
                              on_game_end=on_end)

    print("\n=== [FULL] A creates table ===", flush=True)
    a.new_table()
    time.sleep(2)
    K = a_st.k
    if not K:
        print("!! A did not get table K", flush=True)
        return

    print(f"=== [FULL] K={K}: configuring table ===", flush=True)
    configure_table(a, K,
                    private=not cfg.public,
                    rated=cfg.rated,
                    time_min=cfg.time_min,
                    increment_s=cfg.increment)
    time.sleep(0.5)

    print(f"=== [FULL] K={K}: A sits seat 0 ===", flush=True)
    a.sit(K, 0)
    time.sleep(1)

    print(f"=== [FULL] K={K}: A invites {b.nick} ===", flush=True)
    a.invite(K, b.nick)

    # wait for invite
    deadline = time.time() + 12
    while not b_st.invite_k and time.time() < deadline:
        time.sleep(0.3)
    if not b_st.invite_k:
        print("!! B never got invite", flush=True)
        return

    ik = b_st.invite_k
    print(f"=== [FULL] B accepts #{ik} (op 72) + sits seat 1 ===", flush=True)
    b.send([72, ik])
    time.sleep(2)
    b.sit(ik, 1)

    # now watch: both press start when op-70 shows both seated (via handlers)
    # RematchCoordinator handles subsequent games
    total_secs = max_games * 300  # up to 5 min per game
    print(f"=== [FULL] running up to {max_games} game(s) ({total_secs}s) ===", flush=True)
    time.sleep(total_secs)


def scenario_settings(a: PlayokClient, b: PlayokClient,
                      a_st: PlayerState, b_st: PlayerState,
                      cfg: argparse.Namespace):
    """Show all table options and their current values via op-89, then cycle through settings."""
    a.on_frame = make_handler(a, a_st)
    b.on_frame = make_handler(b, b_st)

    print("\n=== [SETTINGS] creating table ===", flush=True)
    a.new_table()
    time.sleep(2)
    K = a_st.k
    if not K:
        print("!! no K", flush=True)
        return

    print(f"\n=== [SETTINGS] K={K}: current op-89 settings will appear above ===\n", flush=True)
    print(f"=== Cycling through settings: ===", flush=True)
    tests = [
        ("ttype", 2,  "PRIVATE (ttype=2)"),
        ("gtype", 0,  "non-rated (gtype=0)"),
        ("tg",    30, "clock=30min (tg=30)"),
        ("tm",    0,  "no increment (tm=0)"),
        ("ttype", 0,  "PUBLIC (ttype=0)"),
        ("gtype", 1,  "rated (gtype=1)"),
        ("tg",    0,  "no clock (tg=0) — may be ignored"),
        ("ttype", 2,  "PRIVATE again (ttype=2)"),
    ]
    for key, val, desc in tests:
        print(f"  -> set {key}={val}  ({desc})", flush=True)
        a.send([82, K, val], [key])
        time.sleep(1.5)
    print("\n=== [SETTINGS] done — watch op-89 acks above ===", flush=True)
    time.sleep(5)


def scenario_invite(a: PlayokClient, b: PlayokClient,
                    a_st: PlayerState, b_st: PlayerState,
                    cfg: argparse.Namespace):
    """Create table, invite B, observe frames for 30s."""
    a.on_frame = make_handler(a, a_st)
    b.on_frame = make_handler(b, b_st)

    print("\n=== [INVITE] creating table ===", flush=True)
    a.new_table()
    time.sleep(2)
    K = a_st.k
    if K:
        configure_table(a, K, private=not cfg.public, rated=cfg.rated,
                        time_min=cfg.time_min, increment_s=cfg.increment)
        a.sit(K, 0)
        time.sleep(1)
        a.invite(K, b.nick)
        print(f"=== A invited {b.nick} → watching 30s ===", flush=True)
    time.sleep(30)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="PlayOK two-account debug: full game + rematch + settings")
    ap.add_argument("--scenario", default="full",
                    choices=["full", "invite", "settings"])
    ap.add_argument("--games", type=int, default=3,
                    help="number of games to play (default: 3)")
    ap.add_argument("--no-rematch", action="store_true",
                    help="play only one game, no rematch")
    ap.add_argument("--time-min", type=int, default=30,
                    help="game clock in minutes per side (0 = try unlimited, default: 30)")
    ap.add_argument("--increment", type=int, default=0,
                    help="Fischer increment in seconds (default: 0)")
    ap.add_argument("--rated", action="store_true",
                    help="play rated games (default: non-rated)")
    ap.add_argument("--public", action="store_true",
                    help="public table (default: private)")
    ap.add_argument("--resign-after", type=int, default=0,
                    help="resign after N moves (for fast end-game testing, 0=play to end)")
    cfg = ap.parse_args()

    a_user = os.environ.get("A_USER", "alemgamer")
    a_pw   = os.environ["A_PW"]
    b_user = os.environ.get("B_USER", "nurkal022")
    b_pw   = os.environ["B_PW"]

    a = PlayokClient(a_user, a_pw, verbose=False)
    b = PlayokClient(b_user, b_pw, verbose=False)
    a_st = PlayerState(f"A:{a_user}", seat=0)   # host always takes seat 0
    b_st = PlayerState(f"B:{b_user}", seat=1)   # guest takes seat 1

    print("=== logging in ===", flush=True)
    a.login(); b.login()
    print(f"=== A={a.nick}  B={b.nick} ===", flush=True)
    a.connect(); b.connect()
    time.sleep(2)

    SCENARIOS = {
        "full":     scenario_full,
        "invite":   scenario_invite,
        "settings": scenario_settings,
    }
    SCENARIOS[cfg.scenario](a, b, a_st, b_st, cfg)

    print(f"\n=== FINAL SCORES (A={a.nick}): W:{a_st.wins} L:{a_st.losses} D:{a_st.draws} ===",
          flush=True)
    print(f"=== FINAL SCORES (B={b.nick}): W:{b_st.wins} L:{b_st.losses} D:{b_st.draws} ===",
          flush=True)
    a.close(); b.close()
    print("=== done ===", flush=True)


if __name__ == "__main__":
    main()
