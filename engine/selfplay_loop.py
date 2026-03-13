#!/usr/bin/env python3
"""
Iterative Self-Play Training Loop for Togyz Kumalak NNUE

This script automates the cycle:
1. Generate training data with current best engine (datagen)
2. Train NNUE on the new data
3. Build engine with new weights
4. Test new engine vs previous generation (match)
5. If improved, promote to new best and repeat

Usage:
    python3 selfplay_loop.py --generations 10 --games-per-gen 15000 --depth 10
"""

import subprocess
import sys
import os
import shutil
import time
import argparse
import json


ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_BIN = os.path.join(ENGINE_DIR, "target/release/togyzkumalaq-engine")
WEIGHTS_BIN = os.path.join(ENGINE_DIR, "nnue_weights.bin")
TRAIN_SCRIPT = os.path.join(ENGINE_DIR, "train_nnue.py")


def run_cmd(cmd, desc="", timeout=None):
    """Run command, print output, return (returncode, stdout)"""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    print(f"{'='*60}")
    start = time.time()
    result = subprocess.run(
        cmd, shell=isinstance(cmd, str),
        capture_output=True, text=True, timeout=timeout,
        cwd=ENGINE_DIR
    )
    elapsed = time.time() - start
    if result.stdout:
        print(result.stdout[-2000:])  # last 2000 chars
    if result.stderr:
        # Print stderr selectively (skip NNUE load messages)
        for line in result.stderr.split('\n'):
            if line.strip() and 'NNUE loaded' not in line and 'info depth' not in line:
                print(f"  [stderr] {line}")
    print(f"  (completed in {elapsed:.1f}s, exit code {result.returncode})")
    return result.returncode, result.stdout


def build_engine():
    """Build the Rust engine in release mode"""
    code, _ = run_cmd(
        ["cargo", "build", "--release"],
        desc="Building engine"
    )
    if code != 0:
        print("ERROR: Engine build failed!")
        sys.exit(1)


def generate_data(gen_num, num_games, depth, threads):
    """Generate training data via self-play"""
    prefix = f"gen{gen_num}"
    output_file = f"{prefix}_training_data.bin"

    # Remove old gen files if they exist
    for f in os.listdir(ENGINE_DIR):
        if f.startswith(prefix) and f.endswith('.bin') and 'training_data' not in f:
            os.remove(os.path.join(ENGINE_DIR, f))

    code, stdout = run_cmd(
        [ENGINE_BIN, "datagen", str(num_games), str(depth), str(threads), prefix],
        desc=f"Generating data: Gen{gen_num} ({num_games} games, depth {depth})",
        timeout=7200  # 2 hours max
    )

    output_path = os.path.join(ENGINE_DIR, output_file)
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / 1e6
        num_positions = os.path.getsize(output_path) // 26
        print(f"  Generated: {output_path} ({size_mb:.1f} MB, {num_positions:,} positions)")
        return output_path
    else:
        print(f"ERROR: Output file not found: {output_path}")
        return None


def train_nnue(data_path, gen_num, epochs=100, batch_size=4096, hidden1=256, hidden2=32, lr=0.001):
    """Train NNUE on generated data"""
    code, stdout = run_cmd(
        [sys.executable, TRAIN_SCRIPT,
         "--data", data_path,
         "--epochs", str(epochs),
         "--batch-size", str(batch_size),
         "--hidden1", str(hidden1),
         "--hidden2", str(hidden2),
         "--lr", str(lr),
         "--lam", "0.75"],
        desc=f"Training NNUE Gen{gen_num} (epochs={epochs}, batch={batch_size})",
        timeout=3600  # 1 hour max
    )

    # Check outputs exist
    best_pt = os.path.join(ENGINE_DIR, "nnue_best.pt")
    weights_bin = os.path.join(ENGINE_DIR, "nnue_weights.bin")

    if os.path.exists(weights_bin):
        print(f"  Weights exported: {weights_bin}")
        return True
    else:
        print("ERROR: Training failed - no weights exported")
        return False


def run_match(num_games=100, time_ms=500):
    """Run NNUE vs HCE match, return (nnue_wins, draws, hce_wins, elo_diff)"""
    code, stdout = run_cmd(
        [ENGINE_BIN, "match", str(num_games), str(time_ms)],
        desc=f"Match: NNUE-hybrid vs HCE ({num_games} games, {time_ms}ms/move)",
        timeout=3600
    )

    # Parse results from output
    nnue_wins = 0
    hce_wins = 0
    draws = 0
    score_pct = 50.0
    elo_diff = 0.0

    for line in stdout.split('\n'):
        if 'Final:' in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == 'NNUE' and i + 4 < len(parts):
                    try:
                        nnue_wins = int(parts[i+1])
                        draws = int(parts[i+3])
                        hce_wins = int(parts[i+5])
                    except (ValueError, IndexError):
                        pass
        if 'NNUE score:' in line:
            try:
                score_pct = float(line.split(':')[1].strip().replace('%', ''))
            except:
                pass
        if 'Elo advantage:' in line:
            try:
                elo_diff = float(line.split('+')[1].strip())
            except:
                pass

    return nnue_wins, draws, hce_wins, score_pct, elo_diff


def main():
    parser = argparse.ArgumentParser(description='Self-Play Training Loop')
    parser.add_argument('--generations', type=int, default=5, help='Number of generations')
    parser.add_argument('--games-per-gen', type=int, default=15000, help='Games per generation')
    parser.add_argument('--depth', type=int, default=10, help='Search depth for datagen')
    parser.add_argument('--threads', type=int, default=0, help='Threads (0=auto)')
    parser.add_argument('--match-games', type=int, default=100, help='Games per match test')
    parser.add_argument('--match-time', type=int, default=500, help='Time per move in match (ms)')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=4096, help='Training batch size')
    parser.add_argument('--hidden1', type=int, default=256, help='First hidden layer size')
    parser.add_argument('--hidden2', type=int, default=32, help='Second hidden layer size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--start-gen', type=int, default=0, help='Starting generation number')
    args = parser.parse_args()

    if args.threads == 0:
        args.threads = os.cpu_count() or 4

    print(f"""
╔══════════════════════════════════════════════════════╗
║     Togyz Kumalak Self-Play Training Loop            ║
╠══════════════════════════════════════════════════════╣
║  Generations:     {args.generations:<35}║
║  Games/gen:       {args.games_per_gen:<35}║
║  Depth:           {args.depth:<35}║
║  Threads:         {args.threads:<35}║
║  Match games:     {args.match_games:<35}║
║  Architecture:    40→{args.hidden1}→{args.hidden2}→1{' '*(28-len(str(args.hidden1))-len(str(args.hidden2)))}║
║  Batch size:      {args.batch_size:<35}║
║  Epochs:          {args.epochs:<35}║
╚══════════════════════════════════════════════════════╝
""")

    results_log = []
    total_start = time.time()

    # Save baseline weights
    baseline_weights = os.path.join(ENGINE_DIR, "nnue_weights_gen0_baseline.bin")
    if not os.path.exists(baseline_weights):
        shutil.copy2(WEIGHTS_BIN, baseline_weights)
        print(f"Saved baseline weights: {baseline_weights}")

    # Initial match to establish baseline
    print("\n" + "="*60)
    print("  BASELINE TEST")
    print("="*60)
    build_engine()
    nw, d, hw, score_pct, elo = run_match(args.match_games, args.match_time)
    print(f"\n  Baseline: NNUE {nw}-{d}-{hw} HCE ({score_pct:.1f}%, +{elo:.0f} Elo)")
    results_log.append({
        'gen': 0, 'nnue_wins': nw, 'draws': d, 'hce_wins': hw,
        'score': score_pct, 'elo': elo, 'status': 'baseline'
    })

    best_elo = elo
    best_gen = 0

    for gen in range(args.start_gen + 1, args.start_gen + args.generations + 1):
        gen_start = time.time()
        print(f"\n{'#'*60}")
        print(f"#  GENERATION {gen}")
        print(f"{'#'*60}")

        # 1. Generate data
        data_path = generate_data(gen, args.games_per_gen, args.depth, args.threads)
        if data_path is None:
            print(f"  Skipping Gen{gen} due to datagen failure")
            continue

        # 2. Train NNUE
        success = train_nnue(
            data_path, gen, epochs=args.epochs,
            batch_size=args.batch_size,
            hidden1=args.hidden1, hidden2=args.hidden2,
            lr=args.lr
        )
        if not success:
            print(f"  Skipping Gen{gen} due to training failure")
            continue

        # 3. Build engine with new weights
        build_engine()

        # 4. Test new engine
        nw, d, hw, score_pct, elo = run_match(args.match_games, args.match_time)

        gen_elapsed = time.time() - gen_start
        result = {
            'gen': gen, 'nnue_wins': nw, 'draws': d, 'hce_wins': hw,
            'score': score_pct, 'elo': elo,
            'time_min': gen_elapsed / 60
        }

        if elo > best_elo:
            result['status'] = 'IMPROVED'
            best_elo = elo
            best_gen = gen
            # Save this generation's weights
            gen_weights = os.path.join(ENGINE_DIR, f"nnue_weights_gen{gen}_best.bin")
            shutil.copy2(WEIGHTS_BIN, gen_weights)
            print(f"\n  NEW BEST! Gen{gen}: +{elo:.0f} Elo (was +{results_log[-1]['elo']:.0f})")
        else:
            result['status'] = 'no improvement'
            # Restore best weights
            best_weights = os.path.join(ENGINE_DIR, f"nnue_weights_gen{best_gen}_best.bin")
            if best_gen == 0:
                best_weights = baseline_weights
            if os.path.exists(best_weights):
                shutil.copy2(best_weights, WEIGHTS_BIN)
                print(f"\n  No improvement. Restored Gen{best_gen} weights.")

        results_log.append(result)

        # Print summary table
        print(f"\n{'='*60}")
        print(f"  RESULTS AFTER GEN {gen}")
        print(f"{'='*60}")
        print(f"  {'Gen':>4} {'NNUE':>5} {'Draw':>5} {'HCE':>5} {'Score':>7} {'Elo':>6} {'Status':<12}")
        print(f"  {'-'*50}")
        for r in results_log:
            status = r.get('status', '')
            print(f"  {r['gen']:>4} {r['nnue_wins']:>5} {r['draws']:>5} {r['hce_wins']:>5} "
                  f"{r['score']:>6.1f}% {r['elo']:>+5.0f} {status:<12}")
        print(f"\n  Best: Gen{best_gen} (+{best_elo:.0f} Elo)")

        # Save results log
        with open(os.path.join(ENGINE_DIR, 'selfplay_results.json'), 'w') as f:
            json.dump(results_log, f, indent=2)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  SELF-PLAY LOOP COMPLETE")
    print(f"  Total time: {total_elapsed/3600:.1f} hours")
    print(f"  Best generation: Gen{best_gen} (+{best_elo:.0f} Elo)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
