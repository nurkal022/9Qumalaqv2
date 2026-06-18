#!/usr/bin/env python3
"""
Analyze all PlayOK bot game files → summary CSV + per-game logs.

Usage:
    python analyze.py [--games-dir ../games] [--out-dir .]
"""
import os
import re
import csv
import json
import shutil
import argparse
from pathlib import Path


GAMES_DIR = Path(__file__).parent.parent / "games"
OUT_DIR   = Path(__file__).parent


def parse_game(path: Path) -> dict:
    """Parse one game file. Returns dict with all metadata + move list."""
    lines = path.read_text().splitlines()
    # extract opponent from filename: game_TIMESTAMP_vs_NICK.txt
    m_nick = re.search(r"_vs_(.+)\.txt$", path.name)
    opp_from_file = m_nick.group(1) if m_nick else "?"
    meta = {"file": path.name, "opponent": opp_from_file, "game_num": None,
            "result": None, "our_kazan": None, "opp_kazan": None,
            "total_moves": 0, "our_tuzdyk": -1, "opp_tuzdyk": -1,
            "opening": [], "moves": []}

    # header
    for line in lines[:3]:
        m = re.search(r"random game (\d+)", line)
        if m:
            meta["game_num"] = int(m.group(1))
        m = re.search(r"vs (\S+)", line)
        if m and m.group(1) != "?":
            meta["opponent"] = m.group(1)

    last_pos = None
    for line in lines:
        m = re.match(r"(\d+)\.\s+([WB])(\d+)\s+\[([^\]]+)\]\s+(.+)", line)
        if not m:
            continue
        ply, side, hole_s, note, pos_s = m.groups()
        ply = int(ply)
        hole = int(hole_s)
        parts = pos_s.split("/")
        try:
            w_pits = list(map(int, parts[0].split(",")))
            b_pits = list(map(int, parts[1].split(",")))
            kw, kb = map(int, parts[2].split(","))
            tw, tb = map(int, parts[3].split(","))
            side_to_move = int(parts[4])
        except Exception:
            continue

        last_pos = (w_pits, b_pits, kw, kb, tw, tb)

        # detect tuzdyk (X in notation)
        if "X" in note:
            if side == "W" and meta["our_tuzdyk"] == -1:
                meta["our_tuzdyk"] = ply
            elif side == "B" and meta["opp_tuzdyk"] == -1:
                meta["opp_tuzdyk"] = ply

        meta["moves"].append({
            "ply": ply, "side": side, "hole": hole, "note": note,
            "kw": kw, "kb": kb
        })
        meta["total_moves"] = ply

        if ply <= 10:
            meta["opening"].append(f"{side}{hole}")

    # result from final position
    if last_pos:
        w_pits, b_pits, kw, kb, tw, tb = last_pos
        w_total = kw + sum(w_pits)
        b_total = kb + sum(b_pits)
        meta["our_kazan"]  = w_total   # alemgamer = white = seat 0
        meta["opp_kazan"]  = b_total
        meta["final_w_board"] = sum(w_pits)
        meta["final_b_board"] = sum(b_pits)
        if w_total > b_total:
            meta["result"] = "WIN"
        elif b_total > w_total:
            meta["result"] = "LOSS"
        else:
            meta["result"] = "DRAW"
        meta["score_diff"] = w_total - b_total

    return meta


def turning_point(moves: list) -> int:
    """Return ply where opponent first pulled ahead in kazan."""
    for m in moves:
        if m["kb"] > m["kw"]:
            return m["ply"]
    return -1


def write_detailed_log(meta: dict, out_path: Path):
    """Write a human-readable analysis log for one game."""
    lines = []
    lines.append(f"=== Game {meta['game_num']} vs {meta['opponent']} ===")
    lines.append(f"Result : {meta['result']}  ({meta['our_kazan']} vs {meta['opp_kazan']})")
    lines.append(f"Moves  : {meta['total_moves']}")
    lines.append(f"Opening: {' '.join(meta['opening'])}")
    lines.append(f"Our tuzdyk : move {meta['our_tuzdyk']} (-1 = none)")
    lines.append(f"Opp tuzdyk : move {meta['opp_tuzdyk']} (-1 = none)")

    tp = turning_point(meta["moves"])
    lines.append(f"Opp pulled ahead at move: {tp} (-1 = never)")

    if meta["result"] == "LOSS":
        lines.append("")
        lines.append("--- Loss analysis ---")
        if meta["opp_tuzdyk"] != -1 and meta["our_tuzdyk"] == -1:
            lines.append(f"! Opponent got tuzdyk (move {meta['opp_tuzdyk']}), we did NOT → material disadvantage")
        elif meta["opp_tuzdyk"] < meta["our_tuzdyk"] and meta["opp_tuzdyk"] != -1:
            lines.append(f"! Opponent tuzdyk (move {meta['opp_tuzdyk']}) earlier than ours (move {meta['our_tuzdyk']})")
        sweep_loss = meta.get("final_b_board", 0) - meta.get("final_w_board", 0)
        if sweep_loss > 5:
            lines.append(f"! Sweep endgame cost us {sweep_loss} stones (opponent had more on board at end)")
        diff = abs(meta.get("score_diff", 0))
        if diff <= 4:
            lines.append(f"! Very close game ({diff} stone margin) — single tactical error decided it")
        elif diff > 15:
            lines.append(f"! Dominant loss ({diff} stones) — structural/positional problem from early")

    lines.append("")
    lines.append("Move-by-move kazan:")
    for m in meta["moves"]:
        marker = " <-- OPP AHEAD" if m["kb"] > m["kw"] and (
            meta["moves"].index(m) == 0 or
            meta["moves"][meta["moves"].index(m)-1]["kw"] >= meta["moves"][meta["moves"].index(m)-1]["kb"]
        ) else ""
        lines.append(f"  {m['ply']:3}. {m['side']}{m['hole']}  kw={m['kw']:3} kb={m['kb']:3}{marker}")

    out_path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games-dir", default=str(GAMES_DIR))
    ap.add_argument("--out-dir",   default=str(OUT_DIR))
    args = ap.parse_args()

    games_dir = Path(args.games_dir)
    out_dir   = Path(args.out_dir)

    all_games = []
    files = sorted(games_dir.glob("game_*.txt"))
    print(f"Processing {len(files)} game files...")

    for f in files:
        try:
            meta = parse_game(f)
        except Exception as e:
            print(f"  SKIP {f.name}: {e}")
            continue
        if meta["result"] is None:
            continue
        all_games.append(meta)

        # copy to wins/losses/draws + write detailed log
        bucket = {"WIN": "wins", "LOSS": "losses", "DRAW": "draws"}[meta["result"]]
        dest_game = out_dir / bucket / f.name
        shutil.copy2(f, dest_game)

        log_name = f.stem + "_analysis.txt"
        write_detailed_log(meta, out_dir / bucket / log_name)

    # summary CSV
    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "game_num","opponent","result","our_kazan","opp_kazan",
            "score_diff","total_moves","our_tuzdyk","opp_tuzdyk","opening","file"
        ])
        w.writeheader()
        for g in all_games:
            w.writerow({
                "game_num":   g["game_num"],
                "opponent":   g["opponent"],
                "result":     g["result"],
                "our_kazan":  g["our_kazan"],
                "opp_kazan":  g["opp_kazan"],
                "score_diff": g.get("score_diff", 0),
                "total_moves":g["total_moves"],
                "our_tuzdyk": g["our_tuzdyk"],
                "opp_tuzdyk": g["opp_tuzdyk"],
                "opening":    " ".join(g["opening"]),
                "file":       g["file"],
            })

    # stats
    wins   = [g for g in all_games if g["result"] == "WIN"]
    losses = [g for g in all_games if g["result"] == "LOSS"]
    draws  = [g for g in all_games if g["result"] == "DRAW"]

    print(f"\n=== Session Summary ===")
    print(f"Total  : {len(all_games)}")
    print(f"Wins   : {len(wins)}  ({100*len(wins)//max(1,len(all_games))}%)")
    print(f"Losses : {len(losses)}")
    print(f"Draws  : {len(draws)}")

    if losses:
        print(f"\n--- Loss breakdown ---")
        # by opponent
        from collections import Counter
        by_opp = Counter(g["opponent"] for g in losses)
        for opp, cnt in by_opp.most_common():
            print(f"  {opp}: {cnt} losses")

        # sweep vs tuzdyk losses
        sweep_losses = [g for g in losses
                        if g.get("final_b_board", 0) - g.get("final_w_board", 0) > 5]
        tuzdyk_losses = [g for g in losses
                         if g["opp_tuzdyk"] != -1 and g["our_tuzdyk"] == -1]
        close_losses = [g for g in losses if abs(g.get("score_diff", 0)) <= 4]

        print(f"\n  Sweep endgame losses  : {len(sweep_losses)}")
        print(f"  Tuzdyk disadvantage   : {len(tuzdyk_losses)}")
        print(f"  Close games (≤4 diff) : {len(close_losses)}")

    # save JSON summary
    json_path = out_dir / "summary.json"
    with open(json_path, "w") as fh:
        json.dump({
            "total": len(all_games), "wins": len(wins),
            "losses": len(losses), "draws": len(draws),
            "winrate_pct": 100*len(wins)//max(1,len(all_games)),
            "loss_reasons": {
                "sweep": len([g for g in losses if g.get("final_b_board",0)-g.get("final_w_board",0)>5]),
                "tuzdyk": len([g for g in losses if g["opp_tuzdyk"]!=-1 and g["our_tuzdyk"]==-1]),
                "close": len([g for g in losses if abs(g.get("score_diff",0))<=4]),
            }
        }, fh, indent=2)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Detailed logs: {out_dir}/wins|losses|draws/")


if __name__ == "__main__":
    main()
