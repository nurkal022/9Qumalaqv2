#!/usr/bin/env python3
"""Generate a champion-mimicking opening book from the 943 PlayOK 'mcts'/'mcts2'
games (rival champions, ELO ~2500 peak).

Records ONLY positions where the champion is to move, picks the champion's actual
move weighted by the champion's result, capped at BOOK_DEPTH plies. Output matches
the Rust engine's book format (engine/src/book.rs):
    pits0|pits1|kazan|tuzdyk|stm|best_move|count

Fixes the tokenizer bug in gen_opening_book_v2.py: move numbers >= 10 (e.g. "10.")
were captured by the 2-digit regex as phantom moves, so only ~3.7% of games
replayed. We strip `\\b\\d+\\.` move numbers first and validate every move against
the legal move list; a game that desyncs is dropped after its last legal ply.
"""
import sys, os, re, argparse
from collections import defaultdict
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "engine"))
import gen_opening_book_v2 as gob  # noqa: E402  (provides Board, board_key)

Board, board_key = gob.Board, gob.board_key

CHAMPION_NICKS = {"mcts", "mcts2"}


def parse_headers(text):
    return {m.group(1): m.group(2) for m in re.finditer(r'\[(\w+)\s+"([^"]*)"\]', text)}


def parse_moves_fixed(text):
    """Return list of 0-indexed from-pits. Strips move numbers, then reads XY tokens
    where X=from-pit(1-9), Y=landing-pit. Returns (moves, landing_digits)."""
    # move text = lines that don't start with '['
    body = "\n".join(l for l in text.splitlines() if not l.strip().startswith("["))
    body = re.sub(r"\b\d+\.", " ", body)          # strip "1." "10." move numbers
    body = re.sub(r"(1-0|0-1|1/2-1/2|\*)\s*$", " ", body)
    moves, landings = [], []
    for m in re.finditer(r"(\d)(\d)(X?)(?:\((\d+)\))?", body):
        frm = int(m.group(1))
        land = int(m.group(2))
        if 1 <= frm <= 9:
            moves.append(frm - 1)
            landings.append(land - 1)
    return moves, landings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=24, help="max plies recorded")
    ap.add_argument("--min-games", type=int, default=2, help="min champion games per entry")
    ap.add_argument("--out", default=os.path.join(ROOT, "engine", "opening_book.txt"))
    ap.add_argument("--games-dir", default=os.path.join(ROOT, "archive/datasets/game-pars/games"))
    ap.add_argument("--id-list", default=os.path.join(ROOT, "archive/datasets/game-pars/mcts_games.txt"))
    args = ap.parse_args()

    ids = [l.strip() for l in open(args.id_list) if l.strip()]
    print(f"champion game ids: {len(ids)}", flush=True)

    # position_key -> move -> weight, and -> count
    weight = defaultdict(lambda: defaultdict(float))
    count = defaultdict(lambda: defaultdict(int))
    legal_ok = 0
    legal_bad = 0
    champ_white = champ_black = 0

    for gid in ids:
        fp = os.path.join(args.games_dir, gid + ".txt")
        try:
            text = open(fp, encoding="utf-8", errors="ignore").read()
        except FileNotFoundError:
            continue
        h = parse_headers(text)
        w, b = h.get("White", ""), h.get("Black", "")
        if w in CHAMPION_NICKS:
            champ_side = 0; champ_white += 1
        elif b in CHAMPION_NICKS:
            champ_side = 1; champ_black += 1
        else:
            continue
        res = h.get("Result", "*")
        if res not in ("1-0", "0-1", "1/2-1/2"):
            continue
        # champion result weight
        if res == "1/2-1/2":
            w_res = 1.0
        elif (res == "1-0") == (champ_side == 0):
            w_res = 3.0   # champion won
        else:
            w_res = 0.25  # champion lost

        moves, _ = parse_moves_fixed(text)
        if len(moves) < 4:
            continue

        board = Board()
        good = True
        for ply, mv in enumerate(moves[: args.depth]):
            if mv not in board.valid_moves():
                good = False
                break
            if board.side_to_move == champ_side:
                k = board_key(board)
                weight[k][mv] += w_res
                count[k][mv] += 1
            board.make_move(mv)
        if good:
            legal_ok += 1
        else:
            legal_bad += 1

    print(f"champion as White: {champ_white}, as Black: {champ_black}")
    print(f"legal full-replay: {legal_ok}, truncated/desync: {legal_bad} "
          f"({100*legal_ok/max(1,legal_ok+legal_bad):.1f}% clean)", flush=True)
    print(f"unique champion-to-move positions: {len(weight)}", flush=True)

    # pick best move per position
    book = {}
    for k, mvs in weight.items():
        best_mv, best_w, best_c = -1, -1.0, 0
        tot_c = sum(count[k].values())
        for mv, wgt in mvs.items():
            if wgt > best_w:
                best_w, best_mv, best_c = wgt, mv, count[k][mv]
        if tot_c >= args.min_games:
            book[k] = (best_mv, tot_c)

    print(f"book entries (>= {args.min_games} games): {len(book)}", flush=True)

    with open(args.out, "w") as f:
        for k, (mv, c) in sorted(book.items(), key=lambda x: -x[1][1]):
            pits0 = ",".join(str(x) for x in k[0])
            pits1 = ",".join(str(x) for x in k[1])
            f.write(f"{pits0}|{pits1}|{k[2]},{k[3]}|{k[4]},{k[5]}|{k[6]}|{mv}|{c}\n")
    print(f"written {args.out} ({len(book)} entries)")

    # quick repertoire sanity: champion's first move distribution (White)
    first = defaultdict(int)
    start = board_key(Board())
    for mv, c in count[start].items():
        first[mv] += c
    if first:
        tot = sum(first.values())
        dist = ", ".join(f"pit{mv+1}={100*c/tot:.0f}%" for mv, c in sorted(first.items(), key=lambda x: -x[1]))
        print(f"champion White 1st-move dist: {dist}")


if __name__ == "__main__":
    main()
