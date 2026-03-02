"""
Prepare V8 training data by merging multiple sources.

Sources:
1. V7 improved self-play data (v7_improved_training_data.bin) — fresh data with repetition detection
2. Master games (master_training.bin) — Elo 1800+ with endgame emphasis
3. V7 combined data (v7_combined.bin) — existing V1 HCE + V6 NNUE data
4. Master all-Elo (master_all_elo_training.bin) — broader coverage

Strategy:
- Weight endgame positions more heavily (duplicate them)
- Shuffle everything for good training distribution
"""

import os
import struct
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

    # Analyze
    evals = data[:, 23:25].copy().view(np.int16).flatten()
    results = data[:, 25]

    # Count endgame positions (few stones on board)
    pit_sums = data[:, 0:9].sum(axis=1).astype(np.int32) + data[:, 9:18].sum(axis=1).astype(np.int32)
    endgame_count = np.sum(pit_sums <= 30)

    print(f"  [{label}] {n:,} positions ({size/1e6:.1f} MB)")
    print(f"    Eval: mean={evals.mean():.1f}, std={evals.std():.1f}, range=[{evals.min()}, {evals.max()}]")
    print(f"    Results: W={np.sum(results==2)}, D={np.sum(results==1)}, L={np.sum(results==0)}")
    print(f"    Endgame (stones<=30): {endgame_count:,} ({100*endgame_count/n:.1f}%)")

    return data


def extract_endgame_positions(data, threshold=30):
    """Extract endgame positions from data."""
    pit_sums = data[:, 0:9].sum(axis=1).astype(np.int32) + data[:, 9:18].sum(axis=1).astype(np.int32)
    mask = pit_sums <= threshold
    return data[mask]


def main():
    parser = argparse.ArgumentParser(description='Prepare V8 training data')
    parser.add_argument('--output', default='v8_combined.bin', help='Output file')
    parser.add_argument('--endgame-duplication', type=int, default=2,
                        help='How many extra copies of endgame positions')
    args = parser.parse_args()

    print("=== Preparing V8 Training Data ===\n")

    all_data = []

    # Source 1: V7 improved self-play (fresh data with better search)
    d = load_and_analyze('v7_improved_training_data.bin', 'V7 improved self-play')
    if d is not None:
        all_data.append(d)
        # Extra endgame emphasis
        endgame = extract_endgame_positions(d)
        if len(endgame) > 0:
            for _ in range(args.endgame_duplication):
                all_data.append(endgame)
            print(f"    + {len(endgame) * args.endgame_duplication:,} endgame duplicates")

    # Source 2: Master games (Elo 1800+)
    d = load_and_analyze('master_training.bin', 'Master Elo 1800+')
    if d is not None:
        # Master data is already endgame-weighted from convert_master_games.py
        all_data.append(d)

    # Source 3: Master all-Elo
    d = load_and_analyze('master_all_elo_training.bin', 'Master all Elo')
    if d is not None:
        all_data.append(d)

    # Source 4: Existing V7 combined data
    d = load_and_analyze('v7_combined.bin', 'V7 combined (existing)')
    if d is not None:
        all_data.append(d)
        endgame = extract_endgame_positions(d)
        if len(endgame) > 0:
            all_data.append(endgame)  # 1 extra copy
            print(f"    + {len(endgame):,} endgame duplicates")

    if not all_data:
        print("\nNo data found!")
        return

    # Concatenate
    merged = np.concatenate(all_data, axis=0)
    print(f"\n=== Total: {len(merged):,} positions ===")

    # Validate stone conservation on sample
    sample = merged[:min(5000, len(merged))]
    valid = 0
    for i in range(len(sample)):
        total = sum(sample[i, 0:9]) + sum(sample[i, 9:18]) + sample[i, 18] + sample[i, 19]
        if total == 162:
            valid += 1
    print(f"Validation: {valid}/{len(sample)} samples pass stone conservation")

    # Shuffle
    print("Shuffling...")
    np.random.seed(2024)
    np.random.shuffle(merged)

    # Write
    output = Path(args.output)
    output.write_bytes(merged.tobytes())
    print(f"Saved: {output} ({output.stat().st_size / 1e6:.1f} MB)")
    print(f"Positions: {len(merged):,}")


if __name__ == '__main__':
    main()
