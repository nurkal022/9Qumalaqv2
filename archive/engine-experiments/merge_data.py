"""
Merge binary training data files from multiple machines.

Usage:
    python merge_data.py [input_dir] [output_file]

Default: merge all *_training_data.bin in current dir → merged_training_data.bin
"""

import os
import sys
import struct
import numpy as np
from pathlib import Path

RECORD_SIZE = 26


def validate_record(record):
    """Check that total stones = 162 (conservation law)."""
    pits_w = sum(record[0:9])
    pits_b = sum(record[9:18])
    kazan_w = record[18]
    kazan_b = record[19]
    total = pits_w + pits_b + kazan_w + kazan_b
    return total == 162


def merge_files(input_dir, output_file, shuffle=True):
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob("*_training_data.bin"))

    if not files:
        print(f"No *_training_data.bin files found in {input_dir}")
        return

    print(f"Found {len(files)} data files:")
    total_records = 0
    all_data = []

    for f in files:
        size = f.stat().st_size
        n_records = size // RECORD_SIZE
        remainder = size % RECORD_SIZE

        if remainder != 0:
            print(f"  WARNING: {f.name} has {remainder} extra bytes (truncating)")

        data = f.read_bytes()
        if remainder:
            data = data[:n_records * RECORD_SIZE]

        print(f"  {f.name}: {n_records:,} positions ({size / 1e6:.1f} MB)")
        total_records += n_records
        all_data.append(data)

    print(f"\nTotal: {total_records:,} positions ({total_records * RECORD_SIZE / 1e6:.1f} MB)")

    # Concatenate
    merged = b''.join(all_data)

    # Validate a sample
    sample_size = min(1000, total_records)
    arr = np.frombuffer(merged[:sample_size * RECORD_SIZE], dtype=np.uint8).reshape(sample_size, RECORD_SIZE)
    valid = 0
    for i in range(sample_size):
        if validate_record(arr[i]):
            valid += 1
    print(f"Validation: {valid}/{sample_size} sampled records have correct stone count (162)")

    if valid < sample_size * 0.95:
        print("WARNING: More than 5% of records fail validation!")

    # Parse eval and result distributions
    evals = np.frombuffer(merged, dtype=np.uint8).reshape(total_records, RECORD_SIZE)
    eval_bytes = evals[:, 23:25].copy()
    eval_vals = eval_bytes.view(np.int16).flatten()
    results = evals[:, 25]

    print(f"\nEval distribution:")
    print(f"  Mean: {eval_vals.mean():.1f}, Std: {eval_vals.std():.1f}")
    print(f"  Min: {eval_vals.min()}, Max: {eval_vals.max()}")

    w_wins = np.sum(results == 2)
    draws = np.sum(results == 1)
    b_wins = np.sum(results == 0)
    print(f"\nResults: W={w_wins:,} D={draws:,} B={b_wins:,}")

    # Shuffle if requested
    if shuffle:
        print(f"\nShuffling {total_records:,} records...")
        arr_all = np.frombuffer(merged, dtype=np.uint8).reshape(total_records, RECORD_SIZE).copy()
        np.random.seed(42)
        np.random.shuffle(arr_all)
        merged = arr_all.tobytes()
        print("Shuffled.")

    # Write output
    output_path = Path(output_file)
    output_path.write_bytes(merged)
    print(f"\nSaved to: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    output_file = sys.argv[2] if len(sys.argv) > 2 else "merged_training_data.bin"
    merge_files(input_dir, output_file)
