#!/usr/bin/env python3
"""
Endgame diagnosis: extract midgame positions from real losses where White
led >=10 in kazan at the 55-65% point, then play them out at full strength
to determine if those positions are objectively winning (eval vs search issue).

Usage:
    python3 tools/endgame_diagnosis.py [--time-ms MS] [--seeds N] [--jobs N]
"""
import sys, os, re, subprocess, threading, time, math, random
from pathlib import Path

ROOT = Path(__file__).parent.parent
GAME_DIR = ROOT / "research/alphazero"
assert (GAME_DIR / "game.py").exists(), f"game.py not found at {GAME_DIR}"
sys.path.insert(0, str(GAME_DIR))
from game import TogyzQumalaq, GameState, Player

ENGINE = os.environ.get("PROBE_ENGINE", str(ROOT / "models/engine/baseline"))
ENGINE_CWD = ROOT / "engine"
GAMES_DIR = ROOT / "tools/playok/games"

# ---- Position parsing ----

def parse_pos_str(pos_str: str) -> TogyzQumalaq:
    """Parse position string: W_pits/B_pits/Wkaz,Bkaz/Wtuz,Btuz/side"""
    parts = pos_str.strip().split("/")
    w_pits = list(map(int, parts[0].split(",")))
    b_pits = list(map(int, parts[1].split(",")))
    kaz = list(map(int, parts[2].split(",")))
    tuz = list(map(int, parts[3].split(",")))
    side = int(parts[4])
    import numpy as np
    state = GameState(
        pits=np.array([w_pits, b_pits], dtype=np.int32),
        kazan=np.array(kaz, dtype=np.int32),
        tuzdyk=np.array(tuz, dtype=np.int8),
        current_player=side,
    )
    g = TogyzQumalaq()
    g.set_state(state)
    return g


def pos_str(g: TogyzQumalaq) -> str:
    s = g.get_state()
    w = ",".join(str(int(x)) for x in s.pits[0])
    b = ",".join(str(int(x)) for x in s.pits[1])
    return f"{w}/{b}/{int(s.kazan[0])},{int(s.kazan[1])}/{int(s.tuzdyk[0])},{int(s.tuzdyk[1])}/{int(s.current_player)}"


def sweep_total(s, player):
    """Total score after sweep rule = kazan + board stones."""
    return int(s.kazan[player]) + int(s.pits[player].sum())


# ---- Game file parsing ----

PLY_RE = re.compile(r"^(\d+)\.\s+[WB]\d+\s+\[.*?\]\s+(\S+)")

def parse_game(path: Path):
    """
    Returns list of (ply, pos_str) for plies in file.
    Also returns (is_loss, final_Wscore, final_Bscore).
    """
    header = ""
    plies = []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("#"):
                header += line + "\n"
                continue
            m = PLY_RE.match(line)
            if m:
                ply_n = int(m.group(1))
                pos = m.group(2)
                plies.append((ply_n, pos))

    if not plies:
        return None

    # Determine outcome from last position using sweep rule
    last_pos = plies[-1][1]
    try:
        g = parse_pos_str(last_pos)
    except Exception:
        return None

    # After last ply, compute final scores (sweep)
    s = g.get_state()
    ws = sweep_total(s, 0)
    bs = sweep_total(s, 1)
    is_loss = bs > ws  # White (bot) lost

    # Extract White(seat0) player name from header
    bot_name = None
    m2 = re.search(r"White\(seat0\)=(\S+)", header)
    if m2:
        bot_name = m2.group(1)

    return {
        "path": path,
        "bot_name": bot_name,
        "plies": plies,
        "total_plies": len(plies),
        "is_loss": is_loss,
        "final_ws": ws,
        "final_bs": bs,
    }


# ---- Extract midgame positions ----

def extract_midgame_positions(games_dir: Path, min_lead: int = 10,
                              min_pct: float = 0.55, max_pct: float = 0.65,
                              exclude_self_play: bool = True):
    """
    From loss games: extract positions at 55-65% point where White
    led >= min_lead in kazan (raw kazan only, not sweep, as that's the eval).
    """
    game_files = sorted(games_dir.glob("*.txt"))
    positions = []
    games_processed = 0
    games_loss = 0

    SELF_PLAY_NAMES = {"nurkal022", "alemgamer"}

    for gf in game_files:
        # Skip self-play files
        fname = gf.name
        if exclude_self_play:
            skip = False
            for name in SELF_PLAY_NAMES:
                if f"_vs_{name}" in fname:
                    skip = True
                    break
            if skip:
                continue

        info = parse_game(gf)
        if info is None or info["total_plies"] < 20:
            continue

        games_processed += 1
        if not info["is_loss"]:
            continue
        games_loss += 1

        plies = info["plies"]
        n = len(plies)
        # Find plies in 55-65% window
        lo = int(n * min_pct)
        hi = int(n * max_pct)
        if lo >= hi:
            # Game too short for window
            lo = max(0, n // 2 - 1)
            hi = min(n, n // 2 + 2)

        for idx in range(lo, hi):
            ply_n, pos = plies[idx]
            try:
                g = parse_pos_str(pos)
            except Exception:
                continue
            s = g.get_state()
            w_kaz = int(s.kazan[0])
            b_kaz = int(s.kazan[1])
            lead = w_kaz - b_kaz
            if lead >= min_lead:
                # Extract tuzdyk structure for diagnostic notes
                w_board = int(s.pits[0].sum())
                b_board = int(s.pits[1].sum())
                w_tuz = int(s.tuzdyk[0])
                b_tuz = int(s.tuzdyk[1])
                positions.append({
                    "pos_str": pos,
                    "game": fname,
                    "ply": ply_n,
                    "total_plies": n,
                    "pct": ply_n / n,
                    "w_kaz": w_kaz,
                    "b_kaz": b_kaz,
                    "lead": lead,
                    "w_board": w_board,
                    "b_board": b_board,
                    "w_tuz": w_tuz,
                    "b_tuz": b_tuz,
                    "final_ws": info["final_ws"],
                    "final_bs": info["final_bs"],
                })

    print(f"Games processed: {games_processed}  Losses: {games_loss}  Candidate positions: {len(positions)}")
    return positions


def deduplicate_positions(positions, max_from_game: int = 2):
    """Keep at most max_from_game positions per game, highest lead."""
    from collections import defaultdict
    by_game = defaultdict(list)
    for p in positions:
        by_game[p["game"]].append(p)

    result = []
    for game, plist in by_game.items():
        plist.sort(key=lambda x: x["lead"], reverse=True)
        result.extend(plist[:max_from_game])

    result.sort(key=lambda x: x["lead"], reverse=True)
    return result


# ---- Engine playout ----

class Engine:
    def __init__(self, path, tt_mb=64, nobook=True):
        env = dict(os.environ, TT_SIZE_MB=str(tt_mb), ENGINE_THREADS="2")
        self.path = str(path)
        self.p = subprocess.Popen(
            [str(path), "serve"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            cwd=str(ENGINE_CWD), text=True, bufsize=1, env=env,
        )
        self._ready()

    def _ready(self):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("Engine died during init")
            if line.strip() == "ready":
                return

    def newgame(self):
        self.p.stdin.write("newgame\n"); self.p.stdin.flush()
        self._ready()

    def bestmove(self, g, time_ms):
        cmd = f"go pos {pos_str(g)} time {time_ms} nobook\n"
        self.p.stdin.write(cmd); self.p.stdin.flush()
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("Engine died during search")
            line = line.strip()
            if line.startswith("bestmove"):
                return int(line.split()[1])
            if line.startswith("terminal"):
                return None
            if line.startswith("error"):
                raise RuntimeError(f"Engine error: {line}")

    def close(self):
        try:
            self.p.stdin.write("quit\n"); self.p.stdin.flush()
            self.p.wait(timeout=2)
        except Exception:
            self.p.kill()


def play_from_position(pos_s: str, time_ms: int, max_plies: int = 500):
    """
    Play engine vs itself from given position.
    Returns (winner, moves_played) where winner=0 White wins, 1 Black wins, 2 draw.
    """
    eng = Engine(ENGINE, tt_mb=64)
    try:
        eng.newgame()
        g = parse_pos_str(pos_s)

        for _ in range(max_plies):
            if g.is_terminal():
                break
            mv = eng.bestmove(g, time_ms)
            if mv is None:
                break
            valid = g.get_valid_moves_list()
            if mv not in valid:
                # Forfeit
                side = g.get_state().current_player
                return (1 - side, 0)
            g.make_move(mv)

        if g.is_terminal():
            w = g.get_winner()
            return (w if w is not None else 2, _)
        # No terminal reached; use sweep to decide
        s = g.get_state()
        ws = sweep_total(s, 0)
        bs = sweep_total(s, 1)
        if ws > bs:
            return (0, max_plies)
        elif bs > ws:
            return (1, max_plies)
        else:
            return (2, max_plies)
    finally:
        eng.close()


def run_playout_test(positions, time_ms: int = 2000, seeds_per_pos: int = 2,
                     jobs: int = 3, verbose: bool = True):
    """
    For each position, play it out seeds_per_pos times (engine vs itself, single
    instance per seed, sequential since engine is single-instance).
    Returns list of result dicts.
    """
    # Expand: one task per (position, seed)
    tasks = []
    for i, p in enumerate(positions):
        for seed in range(seeds_per_pos):
            tasks.append((i, seed, p))

    results_by_pos = {i: [] for i in range(len(positions))}
    lock = threading.Lock()
    task_queue = list(tasks)
    tq_lock = threading.Lock()
    done = [0]

    def worker():
        while True:
            with tq_lock:
                if not task_queue:
                    return
                task = task_queue.pop(0)
            pos_idx, seed, p = task
            try:
                winner, moves = play_from_position(p["pos_str"], time_ms)
            except Exception as e:
                print(f"  ERROR pos {pos_idx} seed {seed}: {e}", flush=True)
                winner, moves = -1, 0
            with lock:
                results_by_pos[pos_idx].append(winner)
                done[0] += 1
                n_done = done[0]
                # Count wins
                w_wins = sum(1 for v in results_by_pos[pos_idx] if v == 0)
                total_tested = len(results_by_pos[pos_idx])
                print(
                    f"  [{n_done:>3}/{len(tasks)}] pos={pos_idx:>2} seed={seed} "
                    f"winner={'W' if winner==0 else 'B' if winner==1 else 'D'} "
                    f"moves={moves}  game={p['game']} ply={p['ply']}/{p['total_plies']} "
                    f"lead=+{p['lead']} (kaz {p['w_kaz']}v{p['b_kaz']})",
                    flush=True,
                )

    threads = [threading.Thread(target=worker) for _ in range(jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return results_by_pos


def main():
    time_ms = 2000
    seeds = 2
    jobs = 4
    max_positions = 20

    for arg in sys.argv[1:]:
        if arg.startswith("--time-ms="):
            time_ms = int(arg.split("=")[1])
        elif arg.startswith("--seeds="):
            seeds = int(arg.split("=")[1])
        elif arg.startswith("--jobs="):
            jobs = int(arg.split("=")[1])
        elif arg.startswith("--max="):
            max_positions = int(arg.split("=")[1])

    print(f"Config: time_ms={time_ms}  seeds_per_pos={seeds}  jobs={jobs}  max_positions={max_positions}")
    print()

    # Step 1: Extract positions
    print("=== Step 1: Extracting midgame positions from loss games ===")
    positions = extract_midgame_positions(GAMES_DIR, min_lead=10)
    positions = deduplicate_positions(positions, max_from_game=2)

    if not positions:
        print("ERROR: No qualifying positions found")
        sys.exit(1)

    print(f"\nTotal qualifying positions (deduped): {len(positions)}")
    print("Top positions by lead:")
    for i, p in enumerate(positions[:10]):
        print(f"  {i:>2}: lead=+{p['lead']:>2}  kaz={p['w_kaz']}v{p['b_kaz']}"
              f"  board={p['w_board']}v{p['b_board']}"
              f"  tuz=W{p['w_tuz']}B{p['b_tuz']}"
              f"  ply={p['ply']}/{p['total_plies']} ({100*p['pct']:.0f}%)"
              f"  {p['game']}")

    # Limit to max_positions, taking a spread (not just top by lead to avoid bias)
    if len(positions) > max_positions:
        # Take every Nth to get a sample across the lead distribution
        step = len(positions) / max_positions
        selected = [positions[int(i * step)] for i in range(max_positions)]
    else:
        selected = positions

    print(f"\nSelected {len(selected)} positions for playout test")
    print()

    # Step 2: Run playout tests
    print(f"=== Step 2: Playing out {len(selected)} positions ({seeds} seeds each, {time_ms}ms/move) ===")
    t0 = time.time()
    results_by_pos = run_playout_test(selected, time_ms=time_ms, seeds_per_pos=seeds, jobs=jobs)
    dt = time.time() - t0
    print(f"\nDone in {dt:.0f}s")
    print()

    # Step 3: Analyze
    print("=== Step 3: Analysis ===")
    white_wins = 0
    draws = 0
    black_wins = 0
    errors = 0
    total_trials = 0

    print(f"{'#':>3} {'Lead':>4} {'W_kaz':>5} {'B_kaz':>5} {'W_tuz':>5} {'B_tuz':>5} "
          f"{'Results':>12} {'Holds':>6} {'Game (ply/total)'}")
    for i, p in enumerate(selected):
        res = results_by_pos[i]
        w = sum(1 for r in res if r == 0)
        d = sum(1 for r in res if r == 2)
        b = sum(1 for r in res if r == 1)
        e = sum(1 for r in res if r == -1)
        n = len(res) - e
        white_wins += w; draws += d; black_wins += b; errors += e
        total_trials += n
        pct = 100 * w / n if n > 0 else 0
        holds_str = f"{pct:.0f}%"
        result_str = f"{w}W/{d}D/{b}B"
        print(f"{i:>3} +{p['lead']:>3}  {p['w_kaz']:>5}  {p['b_kaz']:>5}  "
              f"{p['w_tuz']:>5}  {p['b_tuz']:>5}  "
              f"{result_str:>12}  {holds_str:>6}  {p['game']} ({p['ply']}/{p['total_plies']})")

    print()
    n = total_trials
    holds_pct = 100 * white_wins / n if n > 0 else 0
    print(f"SUMMARY: {white_wins}W / {draws}D / {black_wins}B / {errors}err  over {n} trials")
    print(f"holds_lead_pct = {holds_pct:.1f}%  ({white_wins}/{n})")
    print()

    # Verdict
    if holds_pct >= 70:
        verdict = "search"
        interpretation = (
            "The +10 lead is OBJECTIVELY WINNING at full strength. "
            "Live losses arise from weaker live search (time, early-exit, threading). "
            "Fix = search depth/threading."
        )
    elif holds_pct <= 40:
        verdict = "eval"
        # Check tuzdyk structure: if opponent has tuzdyk in most positions, earlier
        b_has_tuz = sum(1 for p in selected if p["b_tuz"] >= 0)
        if b_has_tuz >= len(selected) * 0.6:
            verdict = "earlier_opening_tuzdyk"
            interpretation = (
                f"Opponent already has tuzdyk in {b_has_tuz}/{len(selected)} positions. "
                "The engine OVER-RATES positions that are actually lost — loss decided "
                "before midgame by tuzdyk + hoarding structure. Fix = eval/opening."
            )
        else:
            interpretation = (
                "The +10 lead position is NOT objectively winning at full strength. "
                "The engine eval OVER-RATES these midgame positions. "
                "Fix = retrain value head on sweep-correct labels."
            )
    else:
        verdict = "mixed"
        interpretation = (
            "Mixed: some positions are won, some lost at full strength. "
            "Both eval quality AND search depth contribute to live losses."
        )

    print(f"VERDICT: {verdict}")
    print(f"INTERPRETATION: {interpretation}")
    print()

    # Print sample positions for evidence
    print("=== Sample positions (top 5 by lead) ===")
    top5 = sorted(selected, key=lambda x: x["lead"], reverse=True)[:5]
    for p in top5:
        print(f"  pos_str: {p['pos_str']}")
        print(f"    lead=+{p['lead']} kaz={p['w_kaz']}v{p['b_kaz']} board={p['w_board']}v{p['b_board']} "
              f"tuz=W{p['w_tuz']}B{p['b_tuz']} ply={p['ply']}/{p['total_plies']}")

    return holds_pct, len(selected), verdict


if __name__ == "__main__":
    main()
