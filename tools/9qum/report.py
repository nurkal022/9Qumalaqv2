#!/usr/bin/env python3
"""Inventory + quality report over the harvested 9qum corpus.

Also cross-checks our endgame sweep rule against 9qum's own referee: for every game
that ended "по камням" we recompute the winner from the final position with (a) raw
kazan and (b) the sweep rule, and compare both to the winner 9qum recorded. PlayOK
already validated sweep 96/96; this is a second, independent implementation.

Usage: python3 tools/9qum/report.py [--out data/9qum]
"""
import argparse
import gzip
import json
import os
from collections import Counter, defaultdict

REAL_PLY = 20          # below this a "game" is a no-show / instant flag
DECISIVE = "по камням"  # 9qum's reason string for a game played out to the stones


def jsonl_read(path):
    if not os.path.exists(path):
        return
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except ValueError:
                    continue


def sweep_winner(state):
    """Winner under the real rule: each side sweeps its own remaining stones into its
    own kazan, except pits the opponent holds as tuzdyk, which pay the opponent."""
    pits, kazan, tuz = state["pits"], list(state["kazan"]), state.get("tuzdyk") or [None, None]
    for side in (0, 1):
        for i in range(9):
            idx = side * 9 + i
            stones = pits[idx]
            if not stones:
                continue
            owner = side
            for p in (0, 1):                    # tuzdyk[p] sits on the opponent's side
                if tuz[p] is not None and tuz[p] == idx:
                    owner = p
            kazan[owner] += stones
    if kazan[0] == kazan[1]:
        return -1, kazan
    return (0 if kazan[0] > kazan[1] else 1), kazan


def raw_winner(state):
    k = state["kazan"]
    if k[0] == k[1]:
        return -1
    return 0 if k[0] > k[1] else 1


def rating_of(meta, seat):
    if not isinstance(meta, dict):
        return None
    return meta.get(f"r{seat}_before") or meta.get(f"r{seat}_after")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/9qum")
    a = ap.parse_args()
    out = a.out

    # ---------------- players
    profiles = {}
    for row in jsonl_read(os.path.join(out, "players.jsonl")):
        profiles[row["name"]] = row.get("profile") or {}
    print(f"PLAYERS: {len(profiles)} crawled")
    if profiles:
        rated = [(p.get("rating") or 0, n) for n, p in profiles.items() if p.get("games")]
        rated.sort(reverse=True)
        print("  top by rating:", ", ".join(f"{n} {r}" for r, n in rated[:8]))
        buckets = Counter()
        for r, _ in rated:
            buckets[f"{int(r // 100) * 100}"] += 1
        print("  rating histogram:", dict(sorted(buckets.items(), reverse=True)))
        print(f"  total games claimed by crawled players: {sum(p.get('games', 0) for p in profiles.values())}")

    # ---------------- games
    path = os.path.join(out, "games", "replays.jsonl.gz")
    n = n_real = n_pos = 0
    reasons, srcs, bots = Counter(), Counter(), defaultdict(Counter)
    rating_buckets = Counter()
    sweep_ok = raw_ok = checked = 0
    differ = differ_sweep_ok = differ_raw_ok = swept_finals = 0
    mismatch_examples = []
    ply_hist = Counter()

    for g in jsonl_read(path):
        n += 1
        ply = g.get("ply_count") or 0
        reasons[g.get("reason")] += 1
        srcs[(g.get("_src") or "?").split(":")[0]] += 1
        if ply >= REAL_PLY:
            n_real += 1
            n_pos += ply
            ply_hist[min(ply // 20 * 20, 200)] += 1
            r0, r1 = rating_of(g.get("_meta"), 0), rating_of(g.get("_meta"), 1)
            if r0 and r1:
                rating_buckets[f"{int(min(r0, r1) // 100) * 100}+"] += 1
            for seat in (0, 1):
                name = g.get(f"seat{seat}") or ""
                if name.startswith("ИИ"):
                    w = g.get("winner")
                    bots[name]["games"] += 1
                    bots[name]["W" if w == seat else ("D" if w == -1 else "L")] += 1

        # sweep-rule cross-check on decisively finished games
        states = g.get("states") or []
        if g.get("reason") == DECISIVE and ply >= REAL_PLY and states:
            final = states[-1]
            sw, _ = sweep_winner(final)
            rw = raw_winner(final)
            theirs = g.get("winner")
            checked += 1
            swept_finals += (sum(final["kazan"]) == 162)
            sweep_ok += (sw == theirs)
            raw_ok += (rw == theirs)
            if sw != rw:  # only these positions can tell the two rules apart
                differ += 1
                differ_sweep_ok += (sw == theirs)
                differ_raw_ok += (rw == theirs)
            if sw != theirs and len(mismatch_examples) < 5:
                mismatch_examples.append((g["game_id"], theirs, sw, rw, final["kazan"], final.get("tuzdyk")))

    print(f"\nGAMES: {n} replays on disk, {n_real} real (>={REAL_PLY} plies) = {n_pos:,} training positions")
    print("  by source:", dict(srcs))
    print("  by end reason:", dict(reasons.most_common()))
    print("  ply histogram (real games):", dict(sorted(ply_hist.items())))
    if rating_buckets:
        print("  min-rating of the pair (rated games):", dict(sorted(rating_buckets.items(), reverse=True)))
    if bots:
        print("  their bots:")
        for name, c in sorted(bots.items(), key=lambda kv: -kv[1]["games"]):
            g_, w, l, d = c["games"], c["W"], c["L"], c["D"]
            print(f"    {name}: {g_} games  {w}W-{d}D-{l}L  = {100 * (w + 0.5 * d) / max(1, g_):.1f}%")

    if checked:
        print(f"\nSWEEP-RULE CROSS-CHECK on {checked} decisive games (9qum referee = ground truth):")
        print(f"  sweep rule agrees:      {sweep_ok}/{checked} = {100 * sweep_ok / checked:.1f}%")
        print(f"  raw-kazan rule agrees:  {raw_ok}/{checked} = {100 * raw_ok / checked:.1f}%")
        print(f"  positions where the two rules DISAGREE (the only discriminating ones): {differ}"
              + (f" -> sweep {differ_sweep_ok}/{differ}, raw {differ_raw_ok}/{differ}" if differ else ""))
        print(f"  fully-accounted finals (kazan sums to 162, i.e. 9qum stores the position AFTER the sweep): "
              f"{swept_finals}/{checked}")
        if not differ:
            print("  => 9qum's referee sweeps too; because it records post-sweep states this corpus")
            print("     confirms the rule but cannot discriminate sweep vs raw-kazan (PlayOK 96/96 already did).")
        for ex in mismatch_examples:
            print(f"  mismatch {ex[0]}: theirs={ex[1]} sweep={ex[2]} raw={ex[3]} kazan={ex[4]} tuzdyk={ex[5]}")

    # ---------------- openings
    nodes = list(jsonl_read(os.path.join(out, "openings.jsonl")))
    if nodes:
        root = next((x for x in nodes if x["line"] == ""), None)
        print(f"\nOPENINGS: {len(nodes)} nodes, corpus = {root['data'].get('games') if root else '?'} games")
        by_depth = Counter(x["depth"] for x in nodes)
        print("  nodes by depth:", dict(sorted(by_depth.items())))
        if root:
            print("  first move:", ", ".join(
                f"pit{m['pit']} {m['share']}% wr{m['winrate']}" for m in root["data"]["moves"][:5]))
        best = []
        for x in nodes:
            for m in x["data"].get("moves", []):
                if m.get("count", 0) >= 100:
                    best.append((m["winrate"], m["count"], x["line"], m["pit"]))
        best.sort(reverse=True)
        print("  best-scoring continuations (>=100 games):")
        for wr, cnt, line, pit in best[:8]:
            print(f"    after [{line or 'start'}] play {pit}: wr {wr}% over {cnt} games")

    # ---------------- their engine's opinion on those games
    curves = list(jsonl_read(os.path.join(out, "analysis", "curves.jsonl.gz")))
    if curves:
        plies = sum(len(c.get("points") or []) for c in curves)
        flat = [p["win"] for c in curves for p in (c.get("points") or [])]
        decided = sum(1 for w in flat if w > 90 or w < 10)
        print(f"\nTHEIR ENGINE'S EVAL (independent value labels): {len(curves)} games, {plies:,} evaluated plies")
        print(f"  win% spread: min {min(flat):.1f} / median {sorted(flat)[len(flat) // 2]:.1f} / max {max(flat):.1f}; "
              f"{100 * decided / len(flat):.0f}% of plies are already decided (>90% or <10%)")
    reviews = list(jsonl_read(os.path.join(out, "analysis", "reviews.jsonl.gz")))
    if reviews:
        spots = [s for r in reviews for s in (r.get("spots") or [])]
        same = sum(1 for s in spots if s.get("same"))
        print(f"  reviews: {len(reviews)} games, {len(spots)} labelled spots "
              f"(best-move + PV), {same} where the human already played the engine's move")

    # ---------------- their training telemetry
    snaps = list(jsonl_read(os.path.join(out, "train_status.jsonl")))
    if snaps:
        s = snaps[-1]
        t = s.get("totals", {})
        print(f"\nTHEIR TRAINING (snapshots: {len(snaps)}): iter={s.get('iteration')} version={s.get('version')} "
              f"sims={s.get('sims')} replay={s.get('replay')} where={s.get('where')}")
        print(f"  totals: selfplay {t.get('games'):,}g / {t.get('positions'):,}pos, "
              f"human {t.get('human_games'):,}g / {t.get('human_positions'):,}pos, "
              f"self-claimed elo {t.get('level_elo')}")
        acc = [h for h in s.get("history", []) if h.get("accepted")]
        print(f"  gate: {len(acc)}/{len(s.get('history', []))} iterations accepted; "
              f"winrates {[h.get('winrate') for h in s.get('history', [])]}")


if __name__ == "__main__":
    main()
