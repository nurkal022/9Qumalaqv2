#!/usr/bin/env python3
"""Play our engine against 9qum's neural net through their public play-vs-AI API.

This is an out-of-lineage strength test: their net has never seen our eval, so unlike
our own A/B matches it cannot be gamed by shared blind spots.

Their /api/ai/think returns the full visit distribution, so `--pick argmax` plays their
net at full strength; their own `best` field is deliberately weakened at the free levels
(level I carries mix=0.22, drop=0.12 — two calls on one position gave different moves).

Games are created in the single-player AI mode, so nothing here touches their human
ladder or ratings.

Usage:
  python3 tools/9qum/match.py --games 10
  python3 tools/9qum/match.py --games 20 --move-ms 2000 --parallel 3 --level i
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent import futures
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "playok"))
from engine import Engine, move_budget_ms  # noqa: E402  (repo-local helper)
sys.path.insert(0, str(REPO / "research" / "data"))
import features_v2 as fv  # noqa: E402  (repo-local helper: shared position validator)
sys.path.insert(0, str(REPO / "tools"))
from engine_provenance import compute_engine_meta  # noqa: E402  (shared with ab_match.py)

BASE = "https://9qum.com/api"
UA = "9qumalaq-research/1.0 (friendly engine research; contact via 9qum founder)"

# Their analysis boards expire (idle timeout or server-side eviction) well within the
# lifetime of a 200+-ply game at ~2.5s/request; a 404 here is routine, not fatal.
NOT_FOUND_MARKERS = ("не найдена", "not found")
MAX_RECOVERIES_PER_GAME = 3  # bound on board recreations before a game is abandoned (void)


class GameNotFoundError(RuntimeError):
    """The API reports the analysis board no longer exists (expired/evicted).

    Kept distinct from plain RuntimeError so callers can recover-by-recreating instead
    of the whole match dying, while other HTTP failures (bad move, auth, etc.) still
    raise as before.
    """


class GameVoidError(RuntimeError):
    """A game could not be salvaged — board recreation failed, or recovery budget for
    this game was exhausted. The caller must record this game as VOID and move on,
    never fold it into the win/draw/loss counts.
    """


def pos_from_state(st):
    """9qum state -> our engine position string w0..w8/b0..b8/kw,kb/tw,tb/side."""
    pits = st["pits"]
    white, black = pits[0:9], pits[9:18]
    tuz = st.get("tuzdyk") or [None, None]
    # their tuzdyk[p] is an absolute pit index on the opponent's side; ours is the pit
    # index inside the opponent's row (-1 = none)
    tw = -1 if tuz[0] is None else tuz[0] - 9
    tb = -1 if tuz[1] is None else tuz[1]
    k = st["kazan"]
    return (",".join(map(str, white)) + "/" + ",".join(map(str, black)) +
            f"/{k[0]},{k[1]}/{tw},{tb}/{st['to_move']}")


class Api:
    """Creating a game needs the guest token; moves and think must go out WITHOUT it —
    for an authenticated session the server insists on playing the AI side itself
    ("Ход ИИ считает сервер"), which would leave us unable to drive the opponent."""

    def __init__(self, token, rps=0.7):
        self.auth = requests.Session()
        self.auth.headers.update({"User-Agent": UA, "Authorization": f"Bearer {token}",
                                  "Content-Type": "application/json"})
        self.plain = requests.Session()
        self.plain.headers.update({"User-Agent": UA, "Content-Type": "application/json"})
        self.min_gap = 1.0 / rps
        self.last = 0.0
        self.lock = threading.Lock()

    def post(self, path, payload, timeout=180):
        s = self.auth if path == "game/new" else self.plain
        for attempt in range(10):
            with self.lock:
                gap = self.min_gap - (time.time() - self.last)
                if gap > 0:
                    time.sleep(gap)
                self.last = time.time()
            try:
                r = s.post(f"{BASE}/{path}", json=payload, timeout=timeout)
                if r.status_code == 429:      # "слишком часто" — their per-IP limiter
                    time.sleep(4.0 + 2.0 * attempt)
                    continue
                if r.status_code >= 500:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                data = r.json()
                if r.status_code != 200:
                    detail = data.get("detail") if isinstance(data, dict) else None
                    if r.status_code == 404 or any(m in str(detail or "").lower() for m in NOT_FOUND_MARKERS):
                        raise GameNotFoundError(f"{path}: HTTP {r.status_code} {detail}")
                    raise RuntimeError(f"{path}: HTTP {r.status_code} {detail}")
                return data
            except requests.RequestException:
                if attempt == 9:
                    raise
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"{path}: retries exhausted")


START = {"pits": [9] * 18, "kazan": [0, 0], "tuzdyk": [None, None], "to_move": 0}


def build_summary(engine_meta, agg, num_results, args, log_path):
    """The final summary for one match run: aggregate score plus the same engine
    provenance recorded on every game record, so the summary alone (without opening
    the log) says which engine+weights+commit produced it."""
    w, d, l, n = agg["w"], agg["d"], agg["l"], agg["n"]
    score = (w + 0.5 * d) / n if n else 0.0
    return {
        **engine_meta,
        **agg,
        "score": score,
        "elo": elo(score, n),
        "num_results": num_results,
        "level": args.get("level"), "mode": args.get("mode"), "pick": args.get("pick"),
        "move_ms": args.get("move_ms"),
        "log_path": log_path,
        "ts": int(time.time()),
    }


def opening_suite(path, plies, min_count, n):
    """A balanced opening suite from the harvested 9qum opening tree.

    Without this both engines follow their book/argmax and every game repeats the same
    line — a 20-game match then measures one opening, not strength.
    """
    import gzip
    lines = []
    opener = gzip.open if path.endswith(".gz") else open
    if not os.path.exists(path):
        return []
    with opener(path, "rt", encoding="utf-8") as f:
        for row in f:
            try:
                node = json.loads(row)
            except ValueError:
                continue
            if node["depth"] != plies:
                continue
            board = node["data"].get("board") or {}
            games = sum(m.get("count", 0) for m in node["data"].get("moves", []))
            if games >= min_count and board.get("pits"):
                try:
                    fv.validate_position(board["pits"], board["kazan"],
                                         board.get("tuzdyk") or [None, None],
                                         board.get("to_move"))
                except fv.InvalidPositionError as exc:
                    # A physically impossible position from the harvested tree must
                    # never be used as a real opening (see fv.InvalidPositionError's
                    # docstring for the incident this guards against) -- skip just this
                    # line rather than crashing the whole suite load over one bad row.
                    print(f"WARNING: skipping opening line {node.get('line')!r}: {exc}")
                    continue
                lines.append((games, node["line"], board))
    lines.sort(reverse=True, key=lambda x: x[0])
    return [{"line": ln, "games": g,
             "position": {"pits": b["pits"], "kazan": b["kazan"],
                          "tuzdyk": b["tuzdyk"], "to_move": b["to_move"]}}
            for g, ln, b in lines[:n]]


def _position_of(st):
    """The position payload analysis/new expects, read back from a state the server gave
    us. This is what makes board recreation lossless: recovery always restores the
    position the harness currently holds, never the game's original opening."""
    return {"pits": st["pits"], "kazan": st["kazan"], "tuzdyk": st.get("tuzdyk"), "to_move": st["to_move"]}


def play_game(api, eng, our_seat, level, move_ms, pick, their_sims, log_path, idx, mode="analysis",
              opening=None, max_recoveries=MAX_RECOVERIES_PER_GAME, engine_meta=None, endgame_ms=None):
    """One game. In `analysis` mode the board is an analysis board: we may post moves for
    both sides, so their net plays argmax(visits) — full strength. In `ai-game` mode the
    server owns the AI side and only accepts its own (mix/drop-randomised) choice.

    Their analysis boards expire (idle timeout or server-side eviction); a 404 on any
    board-scoped call is recovered by recreating the board at the harness's CURRENT
    position (not the game's opening) and retrying the same request. Recovery is bounded
    per game by `max_recoveries`; if it is exhausted, or recreation itself fails, the
    game is abandoned and returned/logged as VOID instead of raising (which would kill
    the whole match) or being folded into the win/draw/loss counts as a loss.

    `engine_meta` (see compute_engine_meta) is stamped onto every record -- win, loss,
    draw or void -- so which engine build/weights/commit produced a game is never left
    to be reconstructed from game ids in a log file after the fact.
    """
    engine_meta = engine_meta or {}
    recoveries = 0
    gid = None
    st = None
    plies = 0
    void_reason = None

    def recreate_board():
        nonlocal gid, recoveries
        if recoveries >= max_recoveries:
            raise GameVoidError(f"exceeded {max_recoveries} board recoveries in one game")
        recoveries += 1
        try:
            new_st = api.post("analysis/new", {"position": _position_of(st)})
        except Exception as exc:   # recreation itself failed -> nothing left to retry
            raise GameVoidError(f"board recreation failed: {exc}") from exc
        gid = new_st["game_id"]

    def post_board(path, payload_fn):
        """POST a board-scoped call (payload_fn takes the current game id); on a 404
        (board expired) recreate the board and retry, up to max_recoveries per game."""
        while True:
            try:
                return api.post(path, payload_fn(gid))
            except GameNotFoundError:
                recreate_board()

    try:
        if mode == "analysis":
            st = api.post("analysis/new", {"position": (opening or {}).get("position", START)})
        else:
            st = api.post("game/new", {"human_player": our_seat, "ai_level": level})
        gid = st["game_id"]
        lvl = st.get("ai_level") or {}

        while not st.get("finished") and plies < 400:
            if st["to_move"] == our_seat:
                pos = pos_from_state(st)
                budget = move_budget_ms(pos, move_ms, move_ms if endgame_ms is None else endgame_ms)
                mv = eng.bestmove(pos, time_ms=budget)
                if isinstance(mv, tuple):       # engine says terminal
                    break
            else:
                # ai/think works on analysis boards too, is unmetered and returns the full
                # visit distribution; analysis/hint is the paid product and 429s quickly.
                think = post_board("ai/think", lambda g: {"game_id": g, "n_sims": their_sims})
                cand = [m for m in think.get("moves", []) if m.get("N", 0) > 0]
                if pick == "argmax" and cand:
                    mv = max(cand, key=lambda m: m["N"])["hole"]
                else:
                    mv = think.get("best", cand[0]["hole"] if cand else None)
                if mv is None:
                    break
            st = post_board("game/move", lambda g: {"game_id": g, "hole": mv})
            plies += 1
    except GameVoidError as exc:
        void_reason = str(exc)

    if void_reason is not None:
        rec = {"game_id": gid, "mode": mode, "opening": (opening or {}).get("line", ""),
               "our_seat": our_seat, "level": level, "pick": pick, "move_ms": move_ms,
               "endgame_ms": endgame_ms,
               "result": "VOID", "void_reason": void_reason, "recoveries": recoveries,
               "plies": plies, "kazan": (st or {}).get("kazan"), "tuzdyk": (st or {}).get("tuzdyk"),
               "engine_path": engine_meta.get("engine_path"),
               "engine_weights_path": engine_meta.get("engine_weights_path"),
               "engine_weights_size": engine_meta.get("engine_weights_size"),
               "engine_weights_sha256": engine_meta.get("engine_weights_sha256"),
               "git_commit": engine_meta.get("git_commit"),
               "ts": int(time.time())}
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  game {idx + 1}: seat{our_seat} opening=[{(opening or {}).get('line', 'start')}] -> "
              f"VOID ({void_reason}) plies={plies} recoveries={recoveries} ({gid})",
              flush=True)
        return {"result": "VOID", "recoveries": recoveries}

    winner = st.get("winner")
    result = "D" if winner == -1 else ("W" if winner == our_seat else "L")
    rec = {"game_id": gid, "mode": mode, "opening": (opening or {}).get("line", ""), "our_seat": our_seat, "level": level, "level_sims": lvl.get("sims"),
           "level_mix": lvl.get("mix"), "level_drop": lvl.get("drop"), "pick": pick,
           "move_ms": move_ms, "endgame_ms": endgame_ms, "result": result, "winner": winner, "plies": plies,
           "kazan": st.get("kazan"), "tuzdyk": st.get("tuzdyk"), "record": st.get("record"),
           "tfen": st.get("tfen"), "net_version": st.get("net_version"), "recoveries": recoveries,
           "engine_path": engine_meta.get("engine_path"),
           "engine_weights_path": engine_meta.get("engine_weights_path"),
           "engine_weights_size": engine_meta.get("engine_weights_size"),
           "engine_weights_sha256": engine_meta.get("engine_weights_sha256"),
           "git_commit": engine_meta.get("git_commit"),
           "ts": int(time.time())}
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  game {idx + 1}: seat{our_seat} opening=[{(opening or {}).get('line', 'start')}] -> {result}  "
          f"kazan={st.get('kazan')} plies={plies} recoveries={recoveries} ({gid})",
          flush=True)
    return {"result": result, "recoveries": recoveries}


def aggregate(results):
    """Split per-game outcomes (as returned by play_game) into the counted W/D/L/n and
    the void/recovered side-counts, kept separate on purpose: a void game (board never
    recovered) must never be folded into the win/draw/loss totals as a loss, since that
    would silently bias the very measurement this harness exists to produce.
    """
    counted = [r for r in results if r["result"] != "VOID"]
    void = [r for r in results if r["result"] == "VOID"]
    recovered = [r for r in results if r["recoveries"] > 0]
    return {
        "w": sum(1 for r in counted if r["result"] == "W"),
        "d": sum(1 for r in counted if r["result"] == "D"),
        "l": sum(1 for r in counted if r["result"] == "L"),
        "n": len(counted),
        "void": len(void),
        "recovered": len(recovered),
    }


def elo(score, n):
    if n == 0 or score in (0.0, 1.0):
        return None
    import math
    return -400 * math.log10(1 / score - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--level", default="i", help="their AI level id (free: iii, ii, i)")
    ap.add_argument("--their-sims", type=int, default=90, help="requested sims (server may clamp to the level)")
    ap.add_argument("--pick", default="argmax", choices=["argmax", "server"],
                    help="argmax = their net at full strength; server = their handicapped choice")
    ap.add_argument("--move-ms", type=int, default=1000, help="thinking time for our engine")
    ap.add_argument("--endgame-move-ms", type=int, default=None,
                    help="thinking time at <=40 board stones (default: same as --move-ms)")
    ap.add_argument("--engine", default=str(REPO / "models" / "engine" / "baseline"))
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--rps", type=float, default=0.7, help="requests/s to their API (they 429 above ~1/2s)")
    ap.add_argument("--mode", default="analysis", choices=["analysis", "ai-game"],
                    help="analysis = their net at argmax(visits); ai-game = their handicapped product level")
    ap.add_argument("--out", default="data/9qum/matches")
    ap.add_argument("--opening-plies", type=int, default=4,
                    help="start each game from a distinct N-ply line of their opening tree (0 = start position)")
    ap.add_argument("--opening-min-count", type=int, default=50)
    ap.add_argument("--opening-offset", type=int, default=0,
                    help="skip the first N lines of the suite (lets a second run use fresh openings)")
    ap.add_argument("--openings", default="data/9qum/openings.jsonl")
    a = ap.parse_args()

    with open("data/9qum/session.json", encoding="utf-8") as f:
        token = json.load(f)["token"]
    api = Api(token, a.rps)
    os.makedirs(a.out, exist_ok=True)
    log_path = os.path.join(a.out, f"match_{int(time.time())}.jsonl")
    suite = opening_suite(a.openings, a.opening_plies, a.opening_min_count,
                          a.opening_offset + max(1, a.games // 2))[a.opening_offset:] \
        if a.opening_plies else []
    if a.opening_plies and not suite:
        print("WARNING: no opening suite found — every game would repeat one line")

    # Computed once per run and stamped on every game record + the final summary (see
    # compute_engine_meta/build_summary) so which engine build/weights/commit produced
    # this match is never left to be reconstructed from game ids after the fact.
    engine_meta = compute_engine_meta(a.engine)
    print(f"our engine: {a.engine} @ {a.move_ms}ms/move")
    print(f"  weights:  {engine_meta['engine_weights_path']} "
          f"(sha256 {engine_meta['engine_weights_sha256']})")
    print(f"  git commit: {engine_meta['git_commit']}")
    if suite:
        print(f"openings:   {len(suite)} distinct {a.opening_plies}-ply lines, each played from both sides")
    print(f"their AI:   level '{a.level}', mode={a.mode}, pick={a.pick}  ->  {log_path}")

    engines = []

    def worker(idx):
        with threading.Lock():
            pass
        e = Engine(Path(a.engine))
        e.start()
        engines.append(e)
        try:
            op = suite[(idx // 2) % len(suite)] if suite else None
            return play_game(api, e, idx % 2, a.level, a.move_ms, a.pick, a.their_sims, log_path, idx,
                             a.mode, op, engine_meta=engine_meta, endgame_ms=a.endgame_move_ms)
        finally:
            e.stop()

    results = []
    with futures.ThreadPoolExecutor(max_workers=a.parallel) as pool:
        for r in pool.map(worker, range(a.games)):
            results.append(r)

    agg = aggregate(results)
    w, d, l, n = agg["w"], agg["d"], agg["l"], agg["n"]
    score = (w + 0.5 * d) / n if n else 0
    e = elo(score, n)
    print(f"\nOUR ENGINE vs 9qum level '{a.level}' ({a.mode}/{a.pick}): {w}W-{d}D-{l}L over {n} counted games "
          f"= {100 * score:.1f}%" + (f" (Elo {e:+.0f})" if e is not None else "") +
          f"  [{agg['void']} void, {agg['recovered']} recovered]")
    if agg["void"]:
        print(f"  CAVEAT: {agg['void']}/{len(results)} games were VOID (board unrecoverable within "
              f"{MAX_RECOVERIES_PER_GAME} tries) and are excluded from the score above — "
              f"see void_reason per game in {log_path}.")

    summary = build_summary(engine_meta, agg, len(results),
                            {"level": a.level, "mode": a.mode, "pick": a.pick, "move_ms": a.move_ms},
                            log_path)
    summary_path = log_path[:-len(".jsonl")] + "_summary.json" if log_path.endswith(".jsonl") \
        else log_path + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"wrote summary {summary_path}")


if __name__ == "__main__":
    main()
