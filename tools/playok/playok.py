"""
PlayOK (playok.com) live Togyz Kumalak client — long-poll transport.

Reverse-engineered 2026-06-08 from /j/tg.js (k2ver=263). See memory
playok-live-protocol.md. This module is transport + protocol only; the
engine bridge lives in bridge.py.

Transport (proven working):
  - Login:   POST https://www.playok.com/ru/togyzkumalak/  (username, pw) -> cookies
  - Page:    GET  same URL with cookie kbeta=tg -> injects window.ge / window.ap
  - Session: POST https://x.playok.com:443/r/0  body "1"  -> session id g
  - Write:   POST .../w/{g}  body = JSON frame(s)
  - Read:    POST .../r/{g}  (long-poll, MUST be POST) -> newline-separated frames

Frame: {"i":[ints], "s":[strings]}  (s omitted when empty).
  ping  incoming i[0]==1  -> reply {"i":[2]}
  keepalive every 30s     -> {"i":[]}

NOTE: automated play is almost certainly against PlayOK ToS (ban risk).
Intended for consensual private games only.
"""
from __future__ import annotations

import json
import re
import threading
import time
from typing import Callable, Optional

import requests

LOGIN_URL = "https://www.playok.com/ru/togyzkumalak/"
BASE = "https://x.playok.com:443"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
HANDSHAKE_OP = 1727
CLIENT_VER = 263

# Outgoing opcodes (reversed from tg.js)
OP_PING_REPLY = 2
OP_CHAT = 20          # {"i":[20],"s":[text]}  (text may be a "/cmd")
OP_WHISPER = 21       # {"i":[21],"s":[to,text]}
OP_NEW_TABLE = 71     # {"i":[71]} -> server creates a table, replies with its K
OP_CONFIG_TABLE = 96  # {"i":[96,K],"s":[opt,...]}
OP_MOVE = 92          # {"i":[92,K,1,hole,think_ds]}
OP_INVITE = 95        # {"i":[95,K,0],"s":[nick]}
OP_RESPOND = 77       # {"i":[77,0|1],"s":[nick]}  (0=accept,1=reject)
OP_SIT = 83           # {"i":[83,K,seatPos]}  take a seat (pos 0/1)
OP_STAND = 84         # {"i":[84,K,seatPos]}  leave a seat
OP_START = 85         # {"i":[85,K]}          press "start"
OP_LEAVE = 73         # {"i":[73,K]}          leave the table

# Incoming opcodes observed live (partial; calibrate the rest on a real game)
INCOMING_NAMES = {
    18: "login_ack", 51: "ui_labels", 25: "lobby_player",
    70: "table_info", 24: "leave?", 72: "?",
}


class PlayokClient:
    def __init__(self, user: str, pw: str, verbose: bool = True):
        self.user = user
        self.pw = pw
        self.verbose = verbose
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA})
        self.g: Optional[str] = None
        self.ks_token: Optional[str] = None
        self.nick: Optional[str] = None
        self.ge: Optional[str] = None
        self.ap: Optional[str] = None
        self.running = False
        self.on_frame: Optional[Callable[[list, list], None]] = None
        self._wlock = threading.Lock()

    # ---- helpers -------------------------------------------------------
    def _hdrs(self) -> dict:
        return {"Origin": "https://www.playok.com", "Referer": LOGIN_URL}

    def _log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    # ---- handshake -----------------------------------------------------
    def login(self) -> None:
        r = self.s.post(LOGIN_URL, data={"username": self.user, "pw": self.pw},
                        allow_redirects=False, timeout=15)
        ks = self.s.cookies.get("ksession")
        if not ks:
            raise RuntimeError(f"login failed (status {r.status_code}, no ksession cookie)")
        parts = ks.split(":")
        self.ks_token = parts[0]
        self.nick = parts[1] if len(parts) > 1 else self.user
        # fetch the logged-in game page to grab fresh ge/ap (regenerated per load)
        self.s.cookies.set("kbexp", "0", domain="playok.com")
        self.s.cookies.set("kbeta", "tg", domain="playok.com")
        page = self.s.get(LOGIN_URL, timeout=15).text
        m_ge = re.search(r"window\.ge\s*=\s*(\d+)", page)
        m_ap = re.search(r"window\.ap\s*=\s*(\d+)", page)
        if not (m_ge and m_ap):
            raise RuntimeError("could not extract ge/ap from logged-in page")
        self.ge, self.ap = m_ge.group(1), m_ap.group(1)
        self._log(f"[login] ok as {self.nick}  ge={self.ge} ap={self.ap}")

    def connect(self) -> None:
        self.g = self.s.post(f"{BASE}/r/0", data="1", headers=self._hdrs(),
                             timeout=15).text.strip()
        if not self.g or not self.g.isdigit():
            raise RuntimeError(f"session open failed: {self.g!r}")
        self._log(f"[connect] session g={self.g}")
        hs = [f"{self.ks_token}|{self.ap}|{self.ge}", "ru", "b", "", UA,
              "/0/1", "w", "1920x1080 1", f"ref:{LOGIN_URL}", f"ver:{CLIENT_VER}"]
        self.send([HANDSHAKE_OP], hs)
        self.running = True
        threading.Thread(target=self._read_loop, daemon=True).start()
        threading.Thread(target=self._keepalive, daemon=True).start()

    # ---- io ------------------------------------------------------------
    def send(self, i: list, s: Optional[list] = None, _retries: int = 3) -> None:
        obj = {"i": i}
        if s:
            obj["s"] = s
        frame = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        last_err = None
        for attempt in range(_retries):
            try:
                with self._wlock:
                    self.s.post(f"{BASE}/w/{self.g}", data=frame.encode("utf-8"),
                                headers=self._hdrs(), timeout=15)
                break
            except requests.RequestException as e:
                last_err = e
                if attempt < _retries - 1:
                    time.sleep(0.5 * (attempt + 1))
        else:
            self._log(f"  !! send failed after {_retries} attempts: {last_err}")
            raise last_err
        if self.verbose and i and i[0] not in (2,):
            self._log(f"  >> SEND {frame}")

    def _read_loop(self) -> None:
        while self.running:
            try:
                r = self.s.post(f"{BASE}/r/{self.g}", data="", headers=self._hdrs(),
                                timeout=45)
                if r.status_code != 200:
                    time.sleep(0.2)
                    continue
                for line in r.text.split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except ValueError:
                        continue
                    i = msg.get("i", [])
                    s = msg.get("s", [])
                    if i and i[0] == 1:        # ping
                        self.send([OP_PING_REPLY])
                        continue
                    name = INCOMING_NAMES.get(i[0] if i else -1, "")
                    self._log(f"  << RECV i={i} s={s}  {('('+name+')') if name else ''}")
                    if self.on_frame:
                        try:
                            self.on_frame(i, s)
                        except Exception as e:  # noqa
                            self._log(f"  !! on_frame error: {e}")
            except requests.RequestException:
                time.sleep(0.3)

    def _keepalive(self) -> None:
        while self.running:
            time.sleep(30)
            try:
                self.send([])
            except requests.RequestException:
                pass

    def close(self) -> None:
        self.running = False

    # ---- high-level actions -------------------------------------------
    def new_table(self) -> None:
        self.send([OP_NEW_TABLE])

    def invite(self, table_k: int, nick: str) -> None:
        self.send([OP_INVITE, table_k, 0], [nick])

    def move(self, table_k: int, hole: int, think_ds: int = 10) -> None:
        self.send([OP_MOVE, table_k, 1, hole, think_ds])

    def sit(self, table_k: int, pos: int = 0) -> None:
        self.send([OP_SIT, table_k, pos])

    def set_table_type(self, table_k: int, closed: bool = True) -> None:
        # op 82 sets a single table option: {"i":[82,K,value],"s":[key]}
        # ttype: 0 = open (открытый), 2 = closed (закрытый)
        self.send([82, table_k, 2 if closed else 0], ["ttype"])

    def set_game_time(self, table_k: int, minutes: int) -> None:
        self.send([82, table_k, int(minutes)], ["tg"])   # game clock, minutes

    def set_increment(self, table_k: int, seconds: int) -> None:
        self.send([82, table_k, int(seconds)], ["tm"])   # Fischer increment, seconds

    def start(self, table_k: int) -> None:
        self.send([OP_START, table_k])

    def leave(self, table_k: int) -> None:
        self.send([OP_LEAVE, table_k])

    def chat(self, text: str) -> None:
        self.send([OP_CHAT], [text])


if __name__ == "__main__":
    import argparse
    import os

    ap = argparse.ArgumentParser(description="PlayOK live client smoke test")
    ap.add_argument("--user", default=os.environ.get("PLAYOK_USER"))
    ap.add_argument("--pw", default=os.environ.get("PLAYOK_PW"))
    ap.add_argument("--seconds", type=int, default=20)
    ap.add_argument("--new-table", action="store_true",
                    help="create a table and log the resulting frames")
    args = ap.parse_args()
    if not args.user or not args.pw:
        raise SystemExit("set --user/--pw or PLAYOK_USER/PLAYOK_PW")

    c = PlayokClient(args.user, args.pw)
    c.login()
    c.connect()
    if args.new_table:
        time.sleep(2)
        print("--- creating table (op 71) ---", flush=True)
        c.new_table()
    time.sleep(args.seconds)
    c.close()
    print("--- done ---", flush=True)
