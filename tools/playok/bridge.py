"""
PlayOK <-> engine bridge for Togyz Kumalak.

Pipeline:  PlayOK long-poll frames  ->  GameState  ->  engine (bestmove)  ->  op 92 move

Verified working: transport/login (playok.py), engine+rules (engine.py),
outgoing commands (create=71, ttype private=82, sit=83, invite=95, start=85, move=92).

Game protocol (top-level frames):
  op 90  state; contains [..., 5, <legalMask>, ...] when it is OUR turn (mask = bitset of legal holes)
  op 92  a move was played (by anyone) — server echoes it to both sides
  op 88  seat/turn flags
  op 91  full move history

Board sync model: we apply EVERY op-92 (our own echoed move and the opponent's)
to a board tracked from the start position, in order. We learn the op-92 layout
from the echo of our OWN first move (we know we played engine's pick), then the
same layout decodes the opponent's moves.

NOTE: automated play is against PlayOK ToS — consensual PRIVATE games only.
"""
from __future__ import annotations

import argparse
import datetime
import os
import threading
import time

from engine import Engine, START_POSITION, move_budget_ms
from playok import PlayokClient

GAMES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "games")


def legal_mask_from_state(i: list):
    """op-90 carries [..., 5, <mask>, ...] when it's our turn. Return mask or None."""
    for k in range(2, len(i) - 1):
        if i[k] == 5 and isinstance(i[k + 1], int) and 0 <= i[k + 1] <= 511:
            return i[k + 1]
    return None


class GameState:
    def __init__(self, my_side: int = 0, orient_flip: bool = False):
        self.my_side = my_side
        self.orient_flip = orient_flip
        self.reset()

    def reset(self):
        self.pos = START_POSITION
        self.active = False
        self.last_sent_pos = None

    def playok_to_engine(self, hole: int) -> int:
        return (8 - hole) if self.orient_flip else hole

    def engine_to_playok(self, pit: int) -> int:
        return (8 - pit) if self.orient_flip else pit

    @property
    def side_to_move(self) -> int:
        return int(self.pos.split("/")[-1])

    def my_turn(self) -> bool:
        return self.active and self.side_to_move == self.my_side


class Bridge:
    def __init__(self, user, pw, *, invite=None, dry_run=True, move_time_ms=1800,
                 endgame_move_time_ms=12000,
                 my_side=0, orient_flip=False, join_k=None, room=None,
                 accept=False, accept_from=None,
                 private=True, rated=False, time_min=30, increment_s=0,
                 rematch=False, max_games=0,
                 # random-opponent mode
                 random_mode=False, scout_mode=False, rematch_wait_s=20,
                 min_elo=0, blocked_nicks=None):
        self.room = room
        self.created = False
        self.accept = accept
        self.accept_from = accept_from
        self.client = PlayokClient(user, pw, verbose=True)
        self.engine = Engine()
        self.invite_nick = invite
        self.dry_run = dry_run
        self.move_time_ms = move_time_ms
        self.endgame_move_time_ms = endgame_move_time_ms
        self.spent_ms = 0            # our thinking time used in the current game
        self.rule_mismatches = 0     # engine said terminal but the server wanted a move
        self.join_k = join_k
        self.table_k = join_k
        self.game = GameState(my_side=my_side, orient_flip=orient_flip)
        self.sat = False
        self.invited = False
        self.started = False
        self.side_locked = False
        self.room_switched = False
        self.entered = False
        self.thinking = False
        self._game_state_seen = False  # True once state>=7 seen in active game
        # random-opponent mode always needs a public table
        if random_mode:
            private = False
        # table settings
        self.private = private
        self.rated = rated
        self.time_min = time_min
        self.increment_s = increment_s
        # rematch
        self.rematch = rematch
        self.max_games = max_games
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        # random-opponent mode
        self.random_mode = random_mode    # True → play strangers continuously
        self.scout_mode = scout_mode      # True → join from lobby; False → host own table
        self.rematch_wait_s = rematch_wait_s
        self.min_elo = min_elo            # skip opponents below this rating
        self.blocked_nicks = {n.lower() for n in (blocked_nicks or [])}
        self.opponent_nick = None         # current opponent
        self.opponents_history: list = [] # [(nick, result_str), ...]
        self._losses_vs: dict = {}        # nick -> consecutive loss count
        self._rematch_timer = None
        self._tables: dict = {}           # k -> (seat0, seat1) lobby map (from op-70/73)
        self._game_records: list = []     # per-game dicts for final summary
        self._stop_event = threading.Event()  # set when max_games reached
        self._table_had_opponent = False  # True once an opponent sat at our current table

    def start(self):
        self.engine.start()
        self.client.login()
        # game recorder (set up BEFORE connect so no move is missed)
        os.makedirs(GAMES_DIR, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        opp = self.invite_nick or "opponent"
        self.rec_path = os.path.join(GAMES_DIR, f"game_{ts}_vs_{opp}.txt")
        self.ply = 0
        with open(self.rec_path, "w") as f:
            f.write(f"# PlayOK togyzkumalak  {ts}\n")
            f.write(f"# White(seat0)={self.client.nick}  Black(seat1)={opp}\n")
            f.write("# ply. side hole(1-9) [playok_notation]  position_after\n\n")
        print(f"[bridge] recording -> {self.rec_path}", flush=True)
        self.client.on_frame = self.on_frame
        self.client.connect()

    # ----------------------------------------------------------------
    def on_frame(self, i: list, s: list):
        if not i:
            return
        op = i[0]

        # op-84 = server tells us we entered table K; only trust if it's our nick.
        # In scout/join mode join_k is set explicitly in _try_join_from_lobby / accept handler.
        if op == 84 and len(i) > 1 and (not s or s[0] == self.client.nick or not self.client.nick):
            # In random HOST mode we always create our own table — never join others.
            if not self.join_k and not (self.random_mode and not self.scout_mode):
                k84 = i[1]
                seats = self._tables.get(k84, ('', ''))
                p0, p1 = seats
                me = self.client.nick
                # if another player is already in a seat → we're joining their table
                if (p0 and p0 != me) or (p1 and p1 != me):
                    self.join_k = k84
                    self.entered = True
                self.table_k = k84
            # join mode: join_k already set; just confirm match
        # op-29 = player status broadcast; '#N' suffix = their table (only use if it's US)
        elif op == 29 and s and s[-1].startswith("#") and not self.join_k and not self.scout_mode:
            me = self.client.nick
            if not s[0] or s[0] == me:  # ignore broadcasts about other players
                try:
                    self.table_k = int(s[-1][1:])
                except ValueError:
                    pass

        # track all tables via op-70/73 broadcasts (builds lobby map for scout mode)
        if op in (70, 73) and len(i) > 1 and len(s) >= 3:
            self._tables[i[1]] = (s[1], s[2])
            if self.scout_mode and not self.sat and not self.game.active and self.join_k is None:
                self._try_join_from_lobby()

        # op-71 = server lobby snapshot sent on join: parse all tables into _tables
        # Format: i=[71, ?, ?, K1, ?, ?, ?, K2, ...] s=[tc1, seat0_1, seat1_1, tc2, ...]
        if op == 71 and len(i) > 3 and len(s) >= 3:
            j = 0
            idx = 3
            while idx < len(i) and j * 3 + 2 < len(s):
                k = i[idx]
                if isinstance(k, int) and k > 0:
                    p0 = s[j * 3 + 1] if j * 3 + 1 < len(s) else ""
                    p1 = s[j * 3 + 2] if j * 3 + 2 < len(s) else ""
                    self._tables[k] = (p0, p1)
                idx += 4
                j += 1
            if self.scout_mode and not self.sat and not self.game.active and self.join_k is None:
                self._try_join_from_lobby()

        # accept-invite mode: friend hosts, we accept (op 72 = enter table) + sit
        if self.accept and not self.entered and op == 75 and len(i) > 1:
            k = i[1]
            inviter = s[1] if len(s) > 1 else "?"
            if (not self.accept_from) or inviter.lower() == self.accept_from.lower():
                print(f"[bridge] invite from {inviter} to table #{k} -> accept (op 72)", flush=True)
                self.client.send([72, k])
                self.join_k = k
                self.table_k = k
                self.entered = True
                self.room_switched = True

        # switch room
        target_room = (self.join_k // 100) * 100 if self.join_k else self.room
        if target_room and not self.room_switched and op == 32 and s:
            target = f"#{target_room}"
            for line in str(s[0]).split("\n"):
                if line.strip().startswith(target):
                    token = line.split(" ")[0]
                    print(f"[bridge] switching room -> {token}", flush=True)
                    self.client.chat("/join " + token)
                    self.room_switched = True
                    break
        if (self.room_switched and not self.join_k and self.invite_nick
                and not self.created):
            self.created = True
            print("[bridge] creating table in room (op 71)...", flush=True)
            self.client.new_table()

        if self.join_k:
            self._join_logic(op, i, s)
        elif self.table_k is not None and not self.sat and not self.scout_mode:
            self._setup_table()

        # track seats at our table (op-70)
        if op == 70 and len(i) > 1 and self.table_k and i[1] == self.table_k and len(s) >= 3:
            me = self.client.nick
            p0, p1 = s[1], s[2]
            # learn opponent nick
            if p0 == me and p1:
                self.opponent_nick = p1
                self._table_had_opponent = True
            elif p1 == me and p0:
                self.opponent_nick = p0
                self._table_had_opponent = True
            # random mode: if opponent left after being seated → find new game
            # Guard: only trigger if opponent actually sat (not just our own seat-assignment op-70)
            if (self.random_mode and self.sat and self._table_had_opponent):
                me_seated = (p0 == me or p1 == me)
                opp_seated = (p0 and p0 != me) or (p1 and p1 != me)
                if me_seated and not opp_seated:
                    # If game is "active" but no moves yet → opponent abandoned at start
                    if self.game.active and self.ply == 0:
                        print(f"[random] opponent left before first move — aborting game", flush=True)
                        self.game.active = False
                    if not self.game.active:
                        if self._rematch_timer:
                            self._rematch_timer.cancel()
                            self._rematch_timer = None
                        print(f"[random] opponent left — creating new table in 3s", flush=True)
                        threading.Timer(3.0, self._random_new_table).start()

        # press start once both seats filled at our table (host mode)
        if (not self.started and self.sat and op == 70 and len(i) > 1
                and i[1] == self.table_k and len(s) >= 3 and s[1] and s[2]):
            opp = s[2] if s[1] == self.client.nick else s[1]
            if opp.lower() in self.blocked_nicks:
                print(f"[bridge] {opp} is blocked — leaving table", flush=True)
                threading.Timer(1.0, self._random_new_table).start()
            else:
                print(f"[bridge] both seated ({s[1]} vs {s[2]}) -> START  opponent={opp}", flush=True)
                self.client.start(self.table_k)
                self.started = True

        # log game frames
        if op in (70, 73, 88, 90, 91, 92):
            print(f"[game] op{op:<3} i={i} s={s}", flush=True)

        # --- game state / result (op 90) ------------------------------
        if op == 90 and len(i) >= 3 and self.table_k and i[1] == self.table_k:
            state_tag = i[2]
            mask = legal_mask_from_state(i)

            # game result detection (state_count >= 2, result != 8)
            # NOTE: check game-over BEFORE game-start so the same frame can't both
            # activate and immediately end the game.
            # Guard: require _game_state_seen so a transition state=4 right after GAME ON
            # doesn't immediately end the game before any moves are played.
            if len(i) >= 5 and state_tag >= 2 and self.game.active and self._game_state_seen:
                result = i[4]
                if result != 8:
                    self.game.active = False
                    self.games_played += 1
                    self.spent_ms = 0
                    # The PlayOK result field (i[4]) sign is NOT a reliable
                    # white/black-wins flag — empirically i[4]==1 occurs for BOTH
                    # white and black wins. The trustworthy signal is our own
                    # tracked board + the endgame sweep rule: each side scoops the
                    # stones remaining on its side into its kazan, then compare.
                    try:
                        parts = self.game.pos.split('/')
                        w_pits = sum(map(int, parts[0].split(',')))
                        b_pits = sum(map(int, parts[1].split(',')))
                        wk, bk = map(int, parts[2].split(','))
                        w_total = wk + w_pits   # white kazan after sweep
                        b_total = bk + b_pits   # black kazan after sweep
                    except Exception:
                        w_total, b_total = 0, 0
                    # we always sit seat 0 (white) on PlayOK; my_side tracks it
                    our_total = w_total if self.game.my_side == 0 else b_total
                    opp_total = b_total if self.game.my_side == 0 else w_total
                    if our_total == opp_total or result == 9:
                        self.draws += 1
                        outcome = f"DRAW ({our_total}-{opp_total})"
                    elif our_total > opp_total:
                        self.wins += 1
                        outcome = f"WIN ({our_total} vs {opp_total})"
                    else:
                        self.losses += 1
                        outcome = f"LOSS ({our_total} vs {opp_total})"
                    self._last_outcome = outcome
                    print(f"[bridge] *** GAME {self.games_played} OVER: {outcome} "
                          f"(raw={result}) | W:{self.wins} L:{self.losses} D:{self.draws} ***",
                          flush=True)
                    self._on_game_over()

            # track that game is truly active (needed for _game_state_seen guard above)
            if state_tag >= 7 and self.game.active and len(i) >= 5 and i[4] == 8:
                self._game_state_seen = True

            # game just started — only after game-over check, and only if result == 8
            # (ongoing). This prevents a game-over frame from also triggering a new start.
            result_field = i[4] if len(i) >= 5 else 8
            if (state_tag >= 7 and not self.game.active and self.started
                    and result_field == 8):
                if self._rematch_timer:
                    self._rematch_timer.cancel()
                    self._rematch_timer = None
                self.game.reset()
                self.game.active = True
                self._game_state_seen = False  # reset: wait for first active state frame
                # i[3] tells us who moves first: 0=white, 1=black, -1=not yet known.
                # PlayOK alternates first mover each game. The engine pos always starts
                # with /0 (white) but we need /1 when black goes first so that apply_move
                # sows from the correct side.
                if len(i) > 3 and i[3] == 1:
                    self.game.pos = self.game.pos[:-1] + "1"
                print(f"[bridge] GAME ON. my_side={self.game.my_side} "
                      f"({'white' if self.game.my_side == 0 else 'black'}) "
                      f"first_mover={i[3] if len(i) > 3 else '?'} "
                      f"pos={self.game.pos}", flush=True)

            if self.game.active and mask is not None:
                self.maybe_play(mask)

        # --- a move was played (echoed to both sides) -----------------
        if op == 92 and self.game.active:
            self._on_move(i, s)

    def _on_move(self, i, s):
        """Apply an echoed move (ours or the opponent's).

        Confirmed live (2026-06-09): incoming move frame is
            i = [92, K, cell]      s = ['<notation>']
        where cell 0-8 = white's hole, cell 10-18 = black's hole.
        So hole = cell - 10 if cell >= 10 else cell. Orientation is direct
        (PlayOK hole == engine pit).
        """
        before = self.game.pos
        if len(i) < 3:
            return
        cell = i[2]
        hole = cell - 10 if cell >= 10 else cell
        if not (0 <= hole <= 8):
            print(f"[move] !! unexpected cell {cell} in {i}", flush=True)
            return
        try:
            pit = self.game.playok_to_engine(hole)
            self.game.pos = self.engine.apply_move(before, pit)
        except Exception as e:
            print(f"[move] !! illegal hole {hole} from {before}: {e}", flush=True)
            return
        mover = int(before.split("/")[-1])           # side that just moved
        side_char = "W" if mover == 0 else "B"
        self.ply += 1
        note = s[0] if s else ""
        line = f"{self.ply}. {side_char}{hole + 1} [{note}]  {self.game.pos}"
        try:
            with open(self.rec_path, "a") as f:
                f.write(line + "\n")
        except Exception:
            pass
        print(f"[move] {line}", flush=True)
        # Do NOT call maybe_play() here — op-90 follows op-92 and carries
        # the legal mask; let it trigger play so sweep endgames have a mask.

    def maybe_play(self, mask=None):
        """Kick off move computation in a WORKER thread so the read/keepalive
        loop is never blocked (blocking it 6s+ makes the server time us out)."""
        g = self.game
        if not g.my_turn() or self.thinking:
            return
        if g.last_sent_pos == g.pos:
            return  # already sent / computing for this position
        self.thinking = True
        g.last_sent_pos = g.pos            # mark immediately to prevent re-entry
        threading.Thread(target=self._compute_and_send,
                         args=(g.pos, mask), daemon=True).start()

    def _on_game_over(self):
        opp = self.opponent_nick or "opponent"
        outcome = getattr(self, '_last_outcome', "?")
        if hasattr(self, '_last_outcome'):
            self.opponents_history.append((opp, outcome))
        self._game_records.append({
            "game": self.games_played,
            "opponent": opp,
            "result": outcome,
            "file": os.path.basename(getattr(self, 'rec_path', '')),
        })

        if self.random_mode:
            winrate = round(self.wins / self.games_played * 100) if self.games_played else 0
            print(f"[random] game {self.games_played}/{self.max_games} vs {opp}: {outcome}  "
                  f"W:{self.wins} L:{self.losses} D:{self.draws} ({winrate}%)", flush=True)

            # check if we've reached the game limit
            if self.max_games > 0 and self.games_played >= self.max_games:
                print(f"[random] *** {self.max_games} games complete. Session done! ***", flush=True)
                self._save_summary()
                self.client.running = False
                self._stop_event.set()
                return

            # track total losses vs this opponent (never resets)
            if "LOSS" in outcome:
                self._losses_vs[opp] = self._losses_vs.get(opp, 0) + 1

            # if lost 3+ times total to same opponent → find a new table
            if self._losses_vs.get(opp, 0) >= 3:
                print(f"[random] {self._losses_vs[opp]} total losses to {opp} — leaving for new opponent",
                      flush=True)
                self.game.reset()
                self.game.last_sent_pos = None
                self.ply = 0
                threading.Timer(2.0, self._random_new_table).start()
                return

            # reset for next game recording
            self.game.reset()
            self.game.last_sent_pos = None
            self.ply = 0
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.rec_path = os.path.join(GAMES_DIR, f"game_{ts}_vs_{opp}.txt")
            with open(self.rec_path, "w") as f:
                f.write(f"# PlayOK togyzkumalak  {ts}  (random game {self.games_played + 1})\n")
                f.write(f"# {self.client.nick} vs ?\n\n")
            if self.table_k:
                self.started = True          # press start = offer rematch
                self.client.start(self.table_k)
            self._rematch_timer = threading.Timer(self.rematch_wait_s, self._random_new_table)
            self._rematch_timer.start()
            return

        limit_hit = self.max_games > 0 and self.games_played >= self.max_games
        if not self.rematch or limit_hit:
            print(f"[bridge] session done: {self.games_played} game(s). "
                  f"W:{self.wins} L:{self.losses} D:{self.draws}", flush=True)
            self._save_summary()
            self._stop_event.set()
            return
        max_label = f"/{self.max_games}" if self.max_games else ""
        print(f"[bridge] rematch in 3s... ({self.games_played}{max_label})", flush=True)
        threading.Timer(3.0, self._do_rematch).start()

    def _do_rematch(self):
        if self.table_k is None:
            return
        self.game.reset()
        self.game.my_side = 1 - self.game.my_side  # PlayOK alternates first mover each rematch
        self.started = True
        self.game.last_sent_pos = None
        self.ply = 0
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        opp = self.opponent_nick or self.invite_nick or "opponent"
        self.rec_path = os.path.join(GAMES_DIR, f"game_{ts}_vs_{opp}.txt")
        with open(self.rec_path, "w") as f:
            f.write(f"# PlayOK togyzkumalak  {ts}  (rematch {self.games_played})\n")
            f.write(f"# White(seat0)={self.client.nick}  Black(seat1)={opp}\n")
            f.write("# ply. side hole(1-9) [playok_notation]  position_after\n\n")
        print(f"[bridge] rematch recording -> {self.rec_path}", flush=True)
        self.client.start(self.table_k)

    # ----------------------------------------------------------------
    # Random-mode helpers
    # ----------------------------------------------------------------
    def _try_join_from_lobby(self):
        """Scout mode: find a table with exactly one empty seat and enter it."""
        me = self.client.nick
        for k, (p0, p1) in list(self._tables.items()):
            if bool(p0) == bool(p1):
                continue      # either both seated or both empty
            occ = p0 if p0 else p1
            if occ == me:
                continue      # that's our own lonely table
            print(f"[random] scout: table #{k}  seats=({p0!r},{p1!r}) → enter", flush=True)
            self.join_k = k
            self.table_k = k
            self.client.send([72, k])
            self.entered = True
            return
        print(f"[random] scout: no joinable table ({len(self._tables)} known)", flush=True)

    def _save_summary(self):
        """Save session stats to a JSON summary file and print a report."""
        import json
        total = self.games_played
        winrate = round(self.wins / total * 100, 1) if total else 0
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        summary = {
            "timestamp": ts,
            "account": self.client.nick,
            "total_games": total,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "winrate_pct": winrate,
            "games": self._game_records,
        }
        path = os.path.join(GAMES_DIR, f"summary_{ts}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n{'='*50}", flush=True)
        print(f"SESSION SUMMARY  ({ts})", flush=True)
        print(f"  Account : {self.client.nick}", flush=True)
        print(f"  Games   : {total}", flush=True)
        print(f"  W/L/D   : {self.wins}/{self.losses}/{self.draws}  ({winrate}%)", flush=True)
        print(f"  Saved   : {path}", flush=True)
        print(f"{'='*50}\n", flush=True)

    def _random_new_table(self):
        """After rematch timeout: leave table and seek a new opponent."""
        if self._rematch_timer:
            self._rematch_timer.cancel()
            self._rematch_timer = None
        old_k = self.table_k
        print(f"[random] leaving table #{old_k} → seeking new opponent", flush=True)
        if old_k:
            self.client.send([73, old_k])       # op-73 = leave table
            self._tables.pop(old_k, None)       # remove from lobby map so we don't rejoin
        # full state reset
        self.sat = False
        self.started = False
        self.entered = False
        self.invited = False
        self.created = False
        self.side_locked = False
        self.join_k = None
        self.table_k = None
        self.game.reset()
        self.game.last_sent_pos = None
        self.opponent_nick = None
        self.thinking = False
        self.spent_ms = 0  # fresh clock accounting for the next game
        self._table_had_opponent = False  # reset so new table won't false-trigger
        if self.scout_mode:
            # next op-70/73 lobby update will trigger _try_join_from_lobby()
            print("[random] scout mode: waiting for lobby update (op-70/73)...", flush=True)
        else:
            print("[random] host mode: creating new public table...", flush=True)
            time.sleep(1.5)
            self.client.new_table()

    def _compute_and_send(self, pos, mask):
        start = time.monotonic()
        try:
            clock_left = None if self.time_min == 0 else self.time_min * 60_000 - self.spent_ms
            budget = move_budget_ms(pos, self.move_time_ms, self.endgame_move_time_ms,
                                    clock_left_ms=clock_left)
            pick = self.engine.bestmove(pos, budget)
            if isinstance(pick, tuple):
                # With the real terminal rule (core fix 2026-09-12) the engine and the
                # server must agree. If they ever disagree again this is a RULES BUG, not
                # a sweep quirk: play the lowest legal hole so we don't lose on time, and
                # count it so the mismatch is visible in the log.
                self.rule_mismatches += 1
                print(f"[engine] *** RULE MISMATCH #{self.rule_mismatches}: engine says terminal "
                      f"({pick[1]}) but server expects a move; pos={pos}", flush=True)
                if mask is not None and mask != 0:
                    hole = (mask & -mask).bit_length() - 1  # lowest set bit
                    if not self.dry_run:
                        self.client.move(self.table_k, hole, think_ds=1)
                        print(f"[bridge] SENT fallback hole {hole}", flush=True)
                return
            hole = self.game.engine_to_playok(pick)
            legal_ok = (mask is None) or bool(mask & (1 << hole))
            print(f"[engine] pos={pos} budget={budget}ms -> pit {pick} -> hole {hole} (mask ok={legal_ok})", flush=True)
            if self.dry_run:
                print(f"[DRY-RUN] would send [92,{self.table_k},1,{hole},..]", flush=True)
                return
            self.client.move(self.table_k, hole, think_ds=max(1, budget // 100))
            print(f"[bridge] SENT move hole {hole} (board advances on server echo)", flush=True)
        finally:
            # Actual wall time spent, not the requested budget -- covers the engine
            # round-trip and the client.move() network call, and is recorded on every
            # exit path (including the rule-mismatch return and any exception) so the
            # clock tracker never undercounts what the game clock actually charged us.
            self.spent_ms += int((time.monotonic() - start) * 1000)
            self.thinking = False

    def _abandon_table(self):
        """Scout: leave current join target and look for another table."""
        old_k = self.join_k
        print(f"[random] scout: abandoning #{old_k} — will look for another table", flush=True)
        if old_k:
            self.client.send([73, old_k])
            self._tables.pop(old_k, None)
        self.join_k = None
        self.table_k = None
        self.entered = False
        self.sat = False
        self.side_locked = False
        self.spent_ms = 0  # fresh clock accounting for the next game
        self._try_join_from_lobby()

    def _join_logic(self, op, i, s):
        """Join Mers13-style: sit at the free seat of an existing table self.join_k."""
        # op-85 = "player left table"; retry sit if we haven't seated yet
        if op == 85 and len(i) > 1 and i[1] == self.join_k and not self.sat:
            print(f"[bridge] player left #{self.join_k} while we wait → retry sit", flush=True)
            self.client.sit(self.join_k, 0)
            time.sleep(0.2)
            self.client.sit(self.join_k, 1)
            return

        seats = None
        if op in (70, 73) and len(i) > 1 and i[1] == self.join_k and len(s) >= 3:
            seats = (s[1], s[2])
        if seats is None:
            return
        s0, s1 = seats
        me = self.client.nick
        if me in (s0, s1):                       # already seated -> lock side
            if not self.side_locked:
                self.game.my_side = 0 if s0 == me else 1
                self.side_locked = True
                self.sat = True
                print(f"[bridge] seated at #{self.join_k} as "
                      f"{'white' if self.game.my_side == 0 else 'black'} "
                      f"(side {self.game.my_side})", flush=True)
            # both seats filled → press ready (needed so game-start detection fires)
            if s0 and s1 and not self.started:
                self.started = True
                self.client.start(self.join_k)
                print(f"[bridge] both seated at #{self.join_k} ({s0} vs {s1}) → READY", flush=True)
            return
        # both seats taken by others (race condition) → abandon and find new table
        if s0 and s1 and me not in (s0, s1) and self.entered and not self.sat:
            print(f"[random] scout: #{self.join_k} full ({s0} vs {s1}) without us → abandon",
                  flush=True)
            self._abandon_table()
            return
        if not self.entered:                      # first ENTER the table (op 72)
            self.entered = True
            print(f"[bridge] entering table #{self.join_k} (op 72)...", flush=True)
            self.client.send([72, self.join_k])
            return
        if not self.sat:                          # then take a free seat
            pos = 0 if s0 == "" else (1 if s1 == "" else None)
            if pos is not None:
                print(f"[bridge] #{self.join_k} seats=({s0!r},{s1!r}) -> sit seat {pos}", flush=True)
                self.client.sit(self.join_k, pos)

    # ----------------------------------------------------------------
    def create_table(self):
        time.sleep(1.5)
        print("[bridge] creating table (op 71)...", flush=True)
        self.client.new_table()

    def _setup_table(self):
        self.sat = True
        k = self.table_k
        # apply table settings
        self.client.set_table_type(k, closed=self.private)
        time.sleep(0.3)
        self.client.send([82, k, 0 if not self.rated else 1], ["gtype"])
        time.sleep(0.3)
        if self.time_min == 0:
            self.client.send([82, k, 0], ["tg"])
        else:
            self.client.send([82, k, self.time_min], ["tg"])
        time.sleep(0.3)
        self.client.send([82, k, self.increment_s], ["tm"])
        time.sleep(0.3)
        self.client.sit(k, 0)
        if self.invite_nick and not self.invited:
            self.client.invite(k, self.invite_nick)
            self.invited = True
        priv_label = "PRIVATE" if self.private else "PUBLIC"
        rated_label = "rated" if self.rated else "non-rated"
        time_label = f"{self.time_min}m+{self.increment_s}s" if self.time_min else "unlimited"
        print(f"[bridge] >>> table #{k} READY  [{priv_label} | {rated_label} | {time_label}]  "
              f"Opponent: {self.invite_nick} — accept invite, sit, press START.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default=os.environ.get("PLAYOK_USER"))
    ap.add_argument("--pw", default=os.environ.get("PLAYOK_PW"))
    ap.add_argument("--invite")
    ap.add_argument("--seconds", type=int, default=86400)
    ap.add_argument("--move-time-ms", type=int, default=1800)
    ap.add_argument("--endgame-move-time-ms", type=int, default=12000,
                    help="thinking time once <=40 stones remain on the board (default: 12000)")
    ap.add_argument("--my-side", type=int, default=0, help="engine side we control (seat 0 -> 0)")
    ap.add_argument("--orient-flip", action="store_true")
    ap.add_argument("--join", type=int, help="join an EXISTING table by number (op 72 enter + sit)")
    ap.add_argument("--room", type=int, help="switch to this room (e.g. 400 pavlodar) before hosting")
    ap.add_argument("--accept", action="store_true", help="wait for an invite and auto-accept (friend hosts)")
    ap.add_argument("--accept-from", help="only accept invites from this nick")
    ap.add_argument("--live", action="store_true", help="actually SEND moves")
    # table settings
    ap.add_argument("--public", action="store_true", help="public table (default: private)")
    ap.add_argument("--rated", action="store_true", help="rated game (default: non-rated)")
    ap.add_argument("--time-min", type=int, default=30,
                    help="game clock per side in minutes (0=unlimited, default: 30)")
    ap.add_argument("--increment", type=int, default=0,
                    help="Fischer increment in seconds (default: 0)")
    # rematch
    ap.add_argument("--rematch", action="store_true", help="auto-rematch after each game")
    ap.add_argument("--max-games", type=int, default=0,
                    help="max games to play; 0=unlimited (default: 0)")
    # random-opponent mode
    ap.add_argument("--random", action="store_true",
                    help="random mode: play strangers continuously (host public table)")
    ap.add_argument("--scout", action="store_true",
                    help="scout mode: join existing tables from the lobby instead of hosting")
    ap.add_argument("--random-wait", type=int, default=20,
                    help="seconds to wait for rematch before finding next opponent (default: 20)")
    ap.add_argument("--min-elo", type=int, default=0,
                    help="minimum opponent ELO for scout mode (default: 0 = no filter)")
    ap.add_argument("--block", nargs="*", default=[],
                    help="nicks to never play against (e.g. --block edelveis uide)")
    args = ap.parse_args()
    if not args.user or not args.pw:
        raise SystemExit("set --user/--pw or PLAYOK_USER/PLAYOK_PW")

    b = Bridge(args.user, args.pw, invite=args.invite, dry_run=not args.live,
               move_time_ms=args.move_time_ms, endgame_move_time_ms=args.endgame_move_time_ms,
               my_side=args.my_side,
               orient_flip=args.orient_flip, join_k=args.join, room=args.room,
               accept=args.accept or bool(args.accept_from), accept_from=args.accept_from,
               private=not args.public, rated=args.rated,
               time_min=args.time_min, increment_s=args.increment,
               rematch=args.rematch, max_games=args.max_games,
               random_mode=args.random or args.scout, scout_mode=args.scout,
               rematch_wait_s=args.random_wait, min_elo=args.min_elo,
               blocked_nicks=args.block)
    b.start()
    if args.random or args.scout:
        mode = "SCOUT (join lobby)" if args.scout else "HOST (create public table)"
        print(f"[bridge] RANDOM mode [{mode}]  rematch-wait={args.random_wait}s  "
              f"{'LIVE' if args.live else 'DRY-RUN'}", flush=True)
        if not args.scout:
            # Leave any stale table from old session, then reset state
            if b.table_k:
                b.client.send([73, b.table_k])
                time.sleep(0.5)
            b.join_k = None
            b.table_k = None
            b.sat = False
            b.started = False
            b.entered = False
            b.side_locked = False
            b.game.reset()
            b._table_had_opponent = False
            b.create_table()
        # else: first op-70/73 lobby frame triggers scout scan automatically
    elif args.accept or args.accept_from:
        print(f"[bridge] ACCEPT mode: waiting for invite"
              f"{(' from ' + args.accept_from) if args.accept_from else ''}...", flush=True)
    elif args.join:
        print(f"[bridge] JOIN mode: looking for table #{args.join} to sit at...", flush=True)
    elif args.invite and not args.room:
        b.create_table()
    # invite + room: table is created after the room switch (in on_frame)
    games_label = f"{args.max_games} games" if args.max_games else "unlimited games"
    print(f"[bridge] running {games_label} / up to {args.seconds}s  "
          f"({'LIVE' if args.live else 'DRY-RUN'})", flush=True)
    b._stop_event.wait(args.seconds)
    if not b._stop_event.is_set():
        b._save_summary()
    b.client.close()
    b.engine.stop()
    print("[bridge] done.", flush=True)
