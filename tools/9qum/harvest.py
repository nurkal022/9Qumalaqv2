#!/usr/bin/env python3
"""Harvest everything 9qum.com exposes publicly.

Phases (all idempotent + resumable, run in this order by default):
  meta         small one-shot endpoints (health, ai levels, leaderboard, altynqor, shop, ...)
  train        append a /api/train/status snapshot (their AlphaZero telemetry)
  openings     walk the opening tree (/api/openings?line=..) with counts + winrates
  players      BFS the player graph via /opponents; profile + last-100 games + repertoire
  games        download full replays for every known game id (move-by-move states)

Game ids come from player game lists, tools/9qum/ws_tournaments.js output and /api/games/recent.

Usage:
  python3 tools/9qum/harvest.py                        # everything
  python3 tools/9qum/harvest.py --phase games --rps 8
  python3 tools/9qum/harvest.py --phase train          # cheap; safe to cron
"""
import argparse
import gzip
from concurrent import futures
import json
import os
import sys
import threading
import time

import requests

from collections import Counter

BASE = "https://9qum.com"
UA = "9qumalaq-research/1.0 (friendly engine research; contact via 9qum founder)"
REAL_PLY = 20  # shorter records are no-shows / instant flag-falls

META_ENDPOINTS = [
    "health", "ai/levels", "lobby", "leaderboard", "leaderboard/activity",
    "games/recent", "altynqor", "fed/ratings", "fed/texts", "fed/protocols",
    "battle/live", "skins", "skins/market", "shop/items", "ads/board", "site/analytics",
]


class Client:
    """Thread-safe HTTP client with a global request-rate cap."""

    def __init__(self, rps: float):
        self.min_gap = 1.0 / rps if rps > 0 else 0.0
        self.last = 0.0
        self.s = requests.Session()
        self.s.headers["User-Agent"] = UA
        self.s.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=16))
        self.n_req = 0
        self.lock = threading.Lock()

    def get(self, path: str, tries: int = 5):
        for attempt in range(tries):
            with self.lock:
                gap = self.min_gap - (time.time() - self.last)
                if gap > 0:
                    time.sleep(gap)
                self.last = time.time()
                self.n_req += 1
            try:
                r = self.s.get(f"{BASE}/api/{path}", timeout=30)
            except requests.RequestException as e:
                if attempt == tries - 1:
                    return None, str(e)
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code == 429 or r.status_code >= 500:
                if attempt == tries - 1:
                    return None, f"HTTP {r.status_code}"
                time.sleep(4.0 + 3.0 * attempt)
                continue
            try:
                return r.json(), None if r.status_code == 200 else f"HTTP {r.status_code}"
            except ValueError:
                return None, f"HTTP {r.status_code} (non-json)"
        return None, "retries exhausted"

    def post(self, path: str, payload: dict, tries: int = 2):
        for attempt in range(tries):
            with self.lock:
                gap = self.min_gap - (time.time() - self.last)
                if gap > 0:
                    time.sleep(gap)
                self.last = time.time()
                self.n_req += 1
            try:
                r = self.s.post(f"{BASE}/api/{path}", json=payload, timeout=120)
                return r.json(), None if r.status_code == 200 else f"HTTP {r.status_code}"
            except (requests.RequestException, ValueError) as e:
                if attempt == tries - 1:
                    return None, str(e)
                time.sleep(2.0)
        return None, "retries exhausted"


def jsonl_append(path, obj):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "at", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def jsonl_read(path):
    if not os.path.exists(path):
        return
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except ValueError:
                    continue


RETRYABLE = ("429", "retries exhausted", "timed out", "Timeout", "Connection")


def transient(err):
    """A 429/timeout must NOT be recorded as done, or the id is dropped for good."""
    return err is not None and any(t in str(err) for t in RETRYABLE)


def load_done(path):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {ln.strip() for ln in f if ln.strip()}


def mark_done(path, key):
    with open(path, "a", encoding="utf-8") as f:
        f.write(key + "\n")


# ---------------------------------------------------------------- phases

def phase_meta(c: Client, out: str):
    d = os.path.join(out, "meta")
    os.makedirs(d, exist_ok=True)
    stamp = int(time.time())
    for ep in META_ENDPOINTS:
        data, err = c.get(ep)
        name = ep.replace("/", "_")
        if data is None:
            print(f"  {ep}: {err}")
            continue
        with open(os.path.join(d, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump({"fetched_at": stamp, "endpoint": ep, "data": data}, f, ensure_ascii=False, indent=1)
        rows = {k: len(v) for k, v in data.items() if isinstance(v, list)} if isinstance(data, dict) else {}
        print(f"  {ep}: ok {rows}" if rows else f"  {ep}: ok")

    # Altyn Qor: the official-tournament archive, one file per tournament
    arch, _ = c.get("altynqor")
    for t in (arch or {}).get("tournaments", []):
        data, err = c.get(f"altynqor/{t['id']}")
        if data is None:
            print(f"  altynqor/{t['id']}: {err}")
            continue
        with open(os.path.join(d, f"altynqor_{t['id']}.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        print(f"  altynqor/{t['id']}: {len(data.get('rounds', []))} rounds, "
              f"{sum(len(r.get('boards', [])) for r in data.get('rounds', []))} boards")


def phase_train(c: Client, out: str):
    path = os.path.join(out, "train_status.jsonl")
    data, err = c.get("train/status")
    if data is None:
        print(f"  train/status: {err}")
        return
    prev = list(jsonl_read(path))
    data["_fetched_at"] = int(time.time())
    if prev and prev[-1].get("reported_at") == data.get("reported_at"):
        print(f"  train/status unchanged (reported_at={data.get('reported_at')}), {len(prev)} snapshots on disk")
        return
    jsonl_append(path, data)
    t = data.get("totals", {})
    hist = data.get("history", [])
    print(f"  train/status: iter={data.get('iteration')} v={data.get('version')} sims={data.get('sims')} "
          f"selfplay_games={t.get('games')} positions={t.get('positions')} "
          f"human={t.get('human_games')}g/{t.get('human_positions')}p level_elo={t.get('level_elo')}")
    for h in hist:
        print(f"    it{h.get('iteration')}: loss={h.get('loss')} p={h.get('policy_loss')} v={h.get('value_loss')} "
              f"score={h.get('score_loss')} gate={h.get('winrate')} accepted={h.get('accepted')} "
              f"match={h.get('match_rate')}/{h.get('match_top3')}")
    print(f"  snapshots on disk: {len(prev) + 1}")


def phase_openings(c: Client, out: str, min_count: int, max_depth: int):
    path = os.path.join(out, "openings.jsonl")
    done_path = os.path.join(out, "openings.done")
    # Cache what we already walked, so a re-run with a lower --min-count can descend
    # THROUGH known nodes instead of stopping at them.
    cache = {row["line"]: row["data"] for row in jsonl_read(path)}
    done = load_done(done_path)
    queue = [""]
    n_new = 0
    while queue:
        line = queue.pop(0)
        depth = 0 if not line else len(line.split(","))
        if line in cache:
            data = cache[line]
        else:
            data, err = c.get("openings" + (f"?line={line}" if line else ""))
            if data is None:
                print(f"  line='{line}': {err}")
                continue
            jsonl_append(path, {"line": line, "depth": depth, "data": data})
            mark_done(done_path, line)
            cache[line] = data
            done.add(line)
            n_new += 1
            if n_new % 50 == 0:
                print(f"  {n_new} new nodes (queue {len(queue)}, depth {depth})", flush=True)
        if depth >= max_depth:
            continue
        for mv in data.get("moves", []):
            if mv.get("count", 0) >= min_count:
                child = f"{line},{mv['pit']}" if line else str(mv["pit"])
                queue.append(child)
    total = sum(1 for _ in jsonl_read(path))
    print(f"  opening tree: +{n_new} nodes this run, {total} total (min_count={min_count}, max_depth={max_depth})")


def seed_players(c: Client, out: str):
    names = set()
    lb, _ = c.get("leaderboard")
    for p in (lb or {}).get("players", []):
        names.add(p["name"])
    act, _ = c.get("leaderboard/activity")
    for key in ("by_games", "by_wins", "by_rating"):
        for p in (act or {}).get(key, []) or []:
            names.add(p["name"])
    lob, _ = c.get("lobby")
    for p in (lob or {}).get("players", []):
        names.add(p["name"])
    rec, _ = c.get("games/recent")
    for g in (rec or {}).get("games", []):
        names.update([g.get("seat0"), g.get("seat1")])
    tp = os.path.join(out, "tournaments.json")
    if os.path.exists(tp):
        with open(tp, encoding="utf-8") as f:
            tj = json.load(f)
        for d in tj.get("detail", {}).values():
            names.update(d.get("players", []))
            for s in d.get("standings", []):
                names.add(s.get("name"))
    return {n for n in names if n}


def phase_players(c: Client, out: str, max_players: int):
    prof_path = os.path.join(out, "players.jsonl")
    games_path = os.path.join(out, "player_games.jsonl")
    done_path = os.path.join(out, "players.done")
    done = load_done(done_path)
    seeds = seed_players(c, out)
    print(f"  seeds: {len(seeds)} players, already harvested: {len(done)}")

    queue = sorted(seeds - done)
    known = set(seeds) | done
    n = 0
    while queue and n < max_players:
        name = queue.pop(0)
        if name in done:
            continue
        prof, err = c.get(f"player/{requests.utils.quote(name, safe='')}")
        if prof is None:
            print(f"  {name}: {err}")
            mark_done(done_path, name)
            done.add(name)
            continue
        games, _ = c.get(f"player/{requests.utils.quote(name, safe='')}/games")
        opps, _ = c.get(f"player/{requests.utils.quote(name, safe='')}/opponents")
        reps, _ = c.get(f"player/{requests.utils.quote(name, safe='')}/openings")
        jsonl_append(prof_path, {
            "name": name, "profile": prof, "openings": reps,
            "opponents": (opps or {}).get("opponents", []), "fetched_at": int(time.time()),
        })
        jsonl_append(games_path, {"name": name, "games": (games or {}).get("games", [])})
        mark_done(done_path, name)
        done.add(name)
        n += 1
        for o in (opps or {}).get("opponents", []):
            nm = o.get("name")
            if nm and nm not in known:
                known.add(nm)
                queue.append(nm)
        if n % 25 == 0:
            print(f"  {n} players harvested, {len(queue)} queued, {len(known)} discovered")
    print(f"  players: +{n} this run, {len(done)} total, {len(queue)} still queued")


def collect_game_ids(out: str):
    """Every game id we know about, with whatever metadata we have."""
    ids = {}
    for row in jsonl_read(os.path.join(out, "player_games.jsonl")):
        for g in row.get("games", []):
            gid = g.get("id")
            if gid and gid not in ids:
                ids[gid] = {"src": "player", "ply": g.get("ply_count"), "public": g.get("public"),
                            "can_open": g.get("can_open"), "meta": g}
    tp = os.path.join(out, "tournaments.json")
    if os.path.exists(tp):
        with open(tp, encoding="utf-8") as f:
            tj = json.load(f)
        for tid, d in tj.get("detail", {}).items():
            for p in d.get("pairings", []):
                gid = p.get("game_id")
                if gid and gid not in ids:
                    ids[gid] = {"src": f"tour:{tid}", "ply": None, "public": d["tournament"].get("is_public"),
                                "can_open": None, "meta": p}
    rp = os.path.join(out, "meta", "games_recent.json")
    if os.path.exists(rp):
        with open(rp, encoding="utf-8") as f:
            for g in json.load(f).get("data", {}).get("games", []):
                gid = g.get("id")
                if gid and gid not in ids:
                    ids[gid] = {"src": "recent", "ply": g.get("ply_count"), "public": 1,
                                "can_open": True, "meta": g}
    return ids


def phase_games(c: Client, out: str, min_ply: int, limit: int, workers: int = 1):
    gdir = os.path.join(out, "games")
    os.makedirs(gdir, exist_ok=True)
    replays = os.path.join(gdir, "replays.jsonl.gz")
    done_path = os.path.join(gdir, "replays.done")
    skipped_path = os.path.join(gdir, "skipped.jsonl")
    done = load_done(done_path)

    ids = collect_game_ids(out)
    todo = []
    n_short = 0
    for gid, info in ids.items():
        if gid in done:
            continue
        if info["ply"] is not None and info["ply"] < min_ply:
            n_short += 1
            mark_done(done_path, gid)  # no-shows / flag-falls: nothing to download
            jsonl_append(skipped_path, {"id": gid, "why": f"ply={info['ply']}", "src": info["src"]})
            continue
        todo.append((gid, info))
    print(f"  known ids: {len(ids)}, already done: {len(done)}, skipped as ply<{min_ply}: {n_short}, to fetch: {len(todo)}")

    todo = todo[:limit] if limit else todo
    counts = {"ok": 0, "err": 0, "n": 0}
    write_lock = threading.Lock()

    def fetch(item):
        gid, info = item
        data, err = c.get(f"games/{gid}/replay")
        ok = data is not None and "states" in data
        if ok:
            data["_src"] = info["src"]
            data["_meta"] = info["meta"]
        with write_lock:
            if ok:
                jsonl_append(replays, data)
                counts["ok"] += 1
            elif transient(err):
                counts["retry"] += 1          # transient: keep it for the next run
            else:
                jsonl_append(skipped_path, {"id": gid, "why": err or (data or {}).get("detail"), "src": info["src"]})
                counts["err"] += 1
            if ok or not transient(err):
                mark_done(done_path, gid)
            counts["n"] += 1
            if counts["n"] % 200 == 0:
                size = os.path.getsize(replays) / 1e6
                print(f"  {counts['n']}/{len(todo)} fetched (ok {counts['ok']}, err {counts['err']}), "
                      f"replays.jsonl.gz = {size:.1f} MB", flush=True)

    if workers > 1:
        with futures.ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(fetch, todo))
    else:
        for item in todo:
            fetch(item)
    print(f"  games: +{counts['ok']} replays, {counts['err']} unavailable")


def phase_analysis(c: Client, out: str, workers: int, reviews: bool, level: str):
    """Their engine's opinion on our harvested games — an evaluator outside our lineage.

    curve  = win% for every ply of a game (instant, free)
    review = the 8 worst moments with best move, q_played/q_best and a 6-move PV
             (costs them ~5s of CPU per game, so it is opt-in)
    """
    adir = os.path.join(out, "analysis")
    os.makedirs(adir, exist_ok=True)
    curves = os.path.join(adir, "curves.jsonl.gz")
    revs = os.path.join(adir, "reviews.jsonl.gz")
    done_c, done_r = load_done(os.path.join(adir, "curves.done")), load_done(os.path.join(adir, "reviews.done"))
    token = None
    spath = os.path.join(out, "session.json")
    if os.path.exists(spath):
        with open(spath, encoding="utf-8") as f:
            token = json.load(f).get("token")
    if token:
        c.s.headers["Authorization"] = f"Bearer {token}"
    else:
        print("  no data/9qum/session.json — run ws_tournaments.js first to mint a guest token")
        return

    ids = [gid for gid, info in collect_game_ids(out).items()
           if (info["ply"] or 0) >= REAL_PLY or info["ply"] is None]
    # only games we actually hold a replay for can be aligned ply-by-ply
    have = load_done(os.path.join(out, "games", "replays.done"))
    ids = [g for g in ids if g in have]
    todo_c = [g for g in ids if g not in done_c]
    print(f"  games with replays: {len(ids)}; curves to fetch: {len(todo_c)}"
          + (f"; reviews to fetch: {len([g for g in ids if g not in done_r])}" if reviews else ""))

    lock = threading.Lock()
    stats = Counter()

    def curve(gid):
        data, err = c.get(f"analysis/curve/{gid}")
        with lock:
            if data and data.get("points"):
                jsonl_append(curves, data)
                stats["curve_ok"] += 1
                stats["points"] += len(data["points"])
                mark_done(os.path.join(adir, "curves.done"), gid)
            elif transient(err):
                stats["curve_retry"] += 1      # leave it undone so a re-run picks it up
            else:
                stats["curve_err"] += 1
                mark_done(os.path.join(adir, "curves.done"), gid)
            if (stats["curve_ok"] + stats["curve_err"]) % 250 == 0:
                print(f"  curves: {stats['curve_ok']} ok / {stats['curve_err']} err, "
                      f"{stats['points']:,} labelled plies", flush=True)

    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(curve, todo_c))
    print(f"  curves: +{stats['curve_ok']} games, {stats['points']:,} evaluated plies "
          f"({stats['curve_err']} permanently unavailable, {stats['curve_retry']} rate-limited -> will retry)")

    if not reviews:
        return
    for gid in [g for g in ids if g not in done_r]:
        data, err = c.post("analysis/review", {"game_id": gid, "ai_level": level})
        for _ in range(12):  # queued server-side: "работаю" -> "готово"
            if data and data.get("state") == "готово":
                break
            time.sleep(2.0)
            data, err = c.post("analysis/review", {"game_id": gid, "ai_level": level})
        if data and data.get("review"):
            jsonl_append(revs, data["review"])
            stats["review_ok"] += 1
        else:
            stats["review_err"] += 1
        mark_done(os.path.join(adir, "reviews.done"), gid)
        if (stats["review_ok"] + stats["review_err"]) % 25 == 0:
            print(f"  reviews: {stats['review_ok']} ok / {stats['review_err']} err", flush=True)
    print(f"  reviews: +{stats['review_ok']} games ({stats['review_err']} failed)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/9qum")
    ap.add_argument("--phase", default="all",
                    choices=["all", "meta", "train", "openings", "players", "games", "analysis"])
    ap.add_argument("--rps", type=float, default=5.0, help="max requests per second")
    ap.add_argument("--max-players", type=int, default=5000)
    ap.add_argument("--min-count", type=int, default=15, help="opening-tree: expand nodes with >= N games")
    ap.add_argument("--max-depth", type=int, default=14, help="opening-tree: max plies")
    ap.add_argument("--min-ply", type=int, default=10, help="games: skip records shorter than this")
    ap.add_argument("--limit", type=int, default=0, help="games: stop after N replays (0 = all)")
    ap.add_argument("--workers", type=int, default=4, help="games/analysis: parallel downloads")
    ap.add_argument("--reviews", action="store_true", help="analysis: also request their engine's game reviews")
    ap.add_argument("--review-level", default="i", help="analysis: ai_level for reviews (i = 90 sims, free)")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    c = Client(a.rps)
    phases = ["meta", "train", "openings", "players", "games", "analysis"] if a.phase == "all" else [a.phase]
    t0 = time.time()
    for p in phases:
        print(f"\n=== {p} ===")
        {"meta": lambda: phase_meta(c, a.out),
         "train": lambda: phase_train(c, a.out),
         "openings": lambda: phase_openings(c, a.out, a.min_count, a.max_depth),
         "players": lambda: phase_players(c, a.out, a.max_players),
         "games": lambda: phase_games(c, a.out, a.min_ply, a.limit, a.workers),
         "analysis": lambda: phase_analysis(c, a.out, a.workers, a.reviews, a.review_level)}[p]()
    print(f"\ndone: {c.n_req} requests in {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    sys.exit(main())
