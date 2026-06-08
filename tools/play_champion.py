#!/usr/bin/env python3
"""Live-game second for playing the improved engine against a champion (e.g. PlayOK).

You relay moves: the engine plays YOUR side and tells you which pit to play; you type
the opponent's (champion's) replies as they happen. Rules/board use the validated
Python logic with the correct end-game sweep rule, so the winner shown is right.

Usage:
    python3 tools/play_champion.py --side white --time 5000
      --side    white|black  : the colour the ENGINE plays (= the side you control)
      --time    ms per move  : engine think time (raise for stronger play; match the
                               time control you're under)
      --book                 : allow the opening book (default OFF — pure search is
                               proven stronger than the book here)
      --tt      MB           : transposition table size (default 512)

Commands during play: a pit number 1-9, 'undo' (take back one ply), 'board', 'quit'.
"""
import sys, os, subprocess, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "archive/old-impls/alphazero-code/alphazero"))
from game import TogyzQumalaq, Player  # noqa: E402

ENGINE = os.path.join(ROOT, "models/engine/baseline")


def pos_str(g):
    s = g.get_state()
    w = ",".join(str(int(x)) for x in s.pits[0])
    b = ",".join(str(int(x)) for x in s.pits[1])
    return f"{w}/{b}/{int(s.kazan[0])},{int(s.kazan[1])}/{int(s.tuzdyk[0])},{int(s.tuzdyk[1])}/{int(s.current_player)}"


class Engine:
    def __init__(self, tt_mb, nobook):
        env = dict(os.environ, TT_SIZE_MB=str(tt_mb))
        self.nobook = nobook
        self.p = subprocess.Popen([ENGINE, "serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env)
        self._ready()
        self.p.stdin.write("newgame\n"); self.p.stdin.flush(); self._ready()

    def _ready(self):
        while True:
            l = self.p.stdout.readline()
            if not l:
                raise RuntimeError("engine died")
            if l.strip() == "ready":
                return

    def best(self, g, time_ms):
        cmd = f"go pos {pos_str(g)} time {time_ms}{' nobook' if self.nobook else ''}\n"
        self.p.stdin.write(cmd); self.p.stdin.flush()
        while True:
            l = self.p.stdout.readline()
            if not l:
                raise RuntimeError("engine died")
            t = l.split()
            if t and t[0] == "bestmove":
                # line: bestmove <mv> score <s> depth <d> nodes <n> time <ms> nps <x>
                d = {t[i]: t[i + 1] for i in range(2, len(t) - 1, 2)}
                return int(t[1]), int(d.get("score", 0)), int(d.get("depth", 0)), int(d.get("time", 0))
            if t and t[0] == "terminal":
                return None, 0, 0, 0


def cell(g, side, i):
    """Render one pit: stone count, or 'X' if it's a tuzdyk (captured by the owner)."""
    s = g.get_state()
    # white's tuzdyk sits in black's row (tuzdyk[0]); black's tuzdyk in white's row (tuzdyk[1])
    if side == 1 and int(s.tuzdyk[0]) == i:
        return " X"
    if side == 0 and int(s.tuzdyk[1]) == i:
        return " X"
    return f"{int(s.pits[side][i]):>2}"


def render(g, engine_side):
    s = g.get_state()
    you = "WHITE" if engine_side == 0 else "BLACK"
    opp = "BLACK" if engine_side == 0 else "WHITE"
    tw = int(s.tuzdyk[0]); tb = int(s.tuzdyk[1])
    print("=" * 54)
    print(f"  BLACK kazan {int(s.kazan[1]):>3}   tuzdyk: {('opp pit '+str(tb+1)) if tb>=0 else '-'}"
          + ("   <- CHAMPION" if engine_side == 0 else "   <- ENGINE (you)"))
    print("   " + " ".join(cell(g, 1, i) for i in range(9)) + "   (black pits)")
    print("    " + "  ".join(str(i) for i in range(1, 10)))
    print("   " + "-" * 38)
    print("    " + "  ".join(str(i) for i in range(1, 10)))
    print("   " + " ".join(cell(g, 0, i) for i in range(9)) + "   (white pits)")
    print(f"  WHITE kazan {int(s.kazan[0]):>3}   tuzdyk: {('opp pit '+str(tw+1)) if tw>=0 else '-'}"
          + ("   <- ENGINE (you)" if engine_side == 0 else "   <- CHAMPION"))
    print(f"  to move: {'WHITE' if s.current_player==0 else 'BLACK'}   "
          f"(engine plays {you}, champion plays {opp})")
    print("=" * 54)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["white", "black"], default="white",
                    help="colour the ENGINE plays (= your side)")
    ap.add_argument("--time", type=int, default=5000, help="engine think time per move (ms)")
    ap.add_argument("--book", action="store_true", help="allow opening book (default: pure search)")
    ap.add_argument("--tt", type=int, default=512, help="TT size MB")
    args = ap.parse_args()
    engine_side = 0 if args.side == "white" else 1

    eng = Engine(args.tt, nobook=not args.book)
    g = TogyzQumalaq(); g.reset()
    history = []  # list of (state_before, move) for undo

    print("\nEngine ready (NNUE + EGTB + opening book loaded).")
    print(f"Engine plays {args.side.upper()} at {args.time} ms/move, "
          f"book {'ON' if args.book else 'OFF (pure search)'}.")
    print("Relay moves to/from your board. Pit numbers are 1-9 for the side to move.\n")

    while True:
        render(g, engine_side)
        if g.is_terminal():
            w = g.get_winner()
            res = {0: "WHITE wins", 1: "BLACK wins", 2: "DRAW"}.get(w, "over")
            print(f"\n*** GAME OVER: {res} (swept rule) ***")
            break

        side = g.get_state().current_player
        if side == engine_side:
            mv, score, depth, t = eng.best(g, args.time)
            if mv is None:
                print("engine: no move (terminal)"); continue
            print(f"\n  >>> ENGINE: play PIT {mv+1}   (eval {score:+d}, depth {depth}, {t}ms)\n")
            history.append((g.get_state().copy(), mv))
            g.make_move(mv)
        else:
            raw = input(f"  Champion's move (pit 1-9 / undo / board / quit): ").strip().lower()
            if raw in ("quit", "q", "exit"):
                break
            if raw == "board":
                continue
            if raw == "undo":
                # take back the last ply (and the engine ply before it, so it's your turn input again)
                for _ in range(2):
                    if history:
                        st, _m = history.pop()
                        g.set_state(st)
                print("  (undone)")
                continue
            if not (raw.isdigit() and 1 <= int(raw) <= 9):
                print("  ! enter a pit number 1-9"); continue
            mv = int(raw) - 1
            if mv not in g.get_valid_moves_list():
                print(f"  ! pit {raw} is empty or blocked (valid: "
                      f"{[i+1 for i in g.get_valid_moves_list()]})"); continue
            history.append((g.get_state().copy(), mv))
            g.make_move(mv)

    eng.p.kill()


if __name__ == "__main__":
    main()
