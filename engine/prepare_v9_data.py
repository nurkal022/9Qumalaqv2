"""
Prepare V9 training data by merging all available sources.

Sources:
1. V7 combined data (v7_combined.bin) — existing HCE + earlier NNUE data
2. V7 improved self-play (v7_improved_training_data.bin) — data with repetition detection
3. Master games Elo 1800+ (master_training.bin)
4. Master all-Elo (master_all_elo_training.bin)
5. Synthetic endgame positions (endgame_synthetic.bin) — targeted endgame training
6. PlayOK data (playok_training_data.bin) — real human games (eval=0, result only)

Strategy:
- Endgame positions get 3x weight (duplicated)
- Synthetic endgame data is already endgame-focused
- Shuffle everything
"""

import os
import numpy as np
import argparse
from pathlib import Path

RECORD_SIZE = 26


def load_and_analyze(path, label=""):
    """Load binary training data and print stats."""
    if not os.path.exists(path):
        print(f"  [{label}] NOT FOUND: {path}")
        return None

    size = os.path.getsize(path)
    n = size // RECORD_SIZE
    data = np.frombuffer(open(path, 'rb').read(n * RECORD_SIZE), dtype=np.uint8).reshape(n, RECORD_SIZE)

    evals = data[:, 23:25].copy().view(np.int16).flatten()
    results = data[:, 25]
    pit_sums = data[:, 0:9].sum(axis=1).astype(np.int32) + data[:, 9:18].sum(axis=1).astype(np.int32)
    endgame_count = np.sum(pit_sums <= 30)

    print(f"  [{label}] {n:,} positions ({size/1e6:.1f} MB)")
    print(f"    Eval: mean={evals.mean():.1f}, std={evals.std():.1f}")
    print(f"    Results: W={np.sum(results==2)}, D={np.sum(results==1)}, L={np.sum(results==0)}")
    print(f"    Endgame (stones<=30): {endgame_count:,} ({100*endgame_count/n:.1f}%)")

    return data


def extract_endgame_positions(data, threshold=30):
    pit_sums = data[:, 0:9].sum(axis=1).astype(np.int32) + data[:, 9:18].sum(axis=1).astype(np.int32)
    return data[pit_sums <= threshold]


def main():
    parser = argparse.ArgumentParser(description='Prepare V9 training data')
    parser.add_argument('--output', default='v9_combined.bin')
    parser.add_argument('--endgame-copies', type=int, default=2,
                        help='Extra copies of endgame positions from self-play data')
    parser.add_argument('--include-playok', action='store_true',
                        help='Include PlayOK data (has no eval)')
    args = parser.parse_args()

    print("=== Preparing V9 Training Data ===\n")

    all_data = []

    # Source 1: V7 combined (HCE + earlier NNUE data)
    d = load_and_analyze('v7_combined.bin', 'V7 combined')
    if d is not None:
        all_data.append(d)
        endgame = extract_endgame_positions(d)
        if len(endgame) > 0:
            for _ in range(args.endgame_copies):
                all_data.append(endgame)
            print(f"    + {len(endgame) * args.endgame_copies:,} endgame duplicates")

    # Source 2: V7 improved self-play
    d = load_and_analyze('v7_improved_training_data.bin', 'V7 improved self-play')
    if d is not None:
        all_data.append(d)
        endgame = extract_endgame_positions(d)
        if len(endgame) > 0:
            for _ in range(args.endgame_copies):
                all_data.append(endgame)
            print(f"    + {len(endgame) * args.endgame_copies:,} endgame duplicates")

    # Source 3: Master games Elo 1800+
    d = load_and_analyze('master_training.bin', 'Master Elo 1800+')
    if d is not None:
        all_data.append(d)

    # Source 4: Master all-Elo
    d = load_and_analyze('master_all_elo_training.bin', 'Master all Elo')
    if d is not None:
        all_data.append(d)

    # Source 5: Synthetic endgame positions (no duplication needed, already endgame-focused)
    d = load_and_analyze('endgame_synthetic.bin', 'Synthetic endgame')
    if d is not None:
        all_data.append(d)

    # Source 6: PlayOK human games (optional, has eval=0)
    if args.include_playok:
        d = load_and_analyze('playok_training_data.bin', 'PlayOK human games')
        if d is not None:
            # Only take a subset to not overwhelm other data
            n_playok = min(len(d), 2_000_000)
            if n_playok < len(d):
                indices = np.random.default_rng(42).permutation(len(d))[:n_playok]
                d = d[indices]
                print(f"    Sampled {n_playok:,} from PlayOK")
            all_data.append(d)

    if not all_data:
        print("\nNo data found!")
        return

    merged = np.concatenate(all_data, axis=0)
    print(f"\n=== Total: {len(merged):,} positions ===")

    # Analyze final composition
    pit_sums = merged[:, 0:9].sum(axis=1).astype(np.int32) + merged[:, 9:18].sum(axis=1).astype(np.int32)
    evals = merged[:, 23:25].copy().view(np.int16).flatten()
    print(f"Endgame (<=30): {np.sum(pit_sums <= 30):,} ({100*np.sum(pit_sums <= 30)/len(merged):.1f}%)")
    print(f"Deep endgame (<=15): {np.sum(pit_sums <= 15):,} ({100*np.sum(pit_sums <= 15)/len(merged):.1f}%)")
    print(f"Positions with eval: {np.sum(np.abs(evals) > 1):,}")

    # Validate stone conservation on sample
    sample = merged[:min(10000, len(merged))]
    valid = 0
    for i in range(len(sample)):
        total = sum(sample[i, 0:9]) + sum(sample[i, 9:18]) + sample[i, 18] + sample[i, 19]
        if total == 162:
            valid += 1
    print(f"Validation: {valid}/{len(sample)} samples pass stone conservation")

    # Shuffle
    print("Shuffling...")
    np.random.seed(2025)
    np.random.shuffle(merged)

    # Write
    output = Path(args.output)
    output.write_bytes(merged.tobytes())
    print(f"Saved: {output} ({output.stat().st_size / 1e6:.1f} MB)")
    print(f"Positions: {len(merged):,}")


if __name__ == '__main__':
    main()
