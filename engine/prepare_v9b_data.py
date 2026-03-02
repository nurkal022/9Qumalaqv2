"""
V9b: Clean training data without synthetic endgame.
Only uses search-evaluated and real-game data with heavy endgame duplication.
"""

import os
import numpy as np
from pathlib import Path

RECORD_SIZE = 26


def load(path, label=""):
    if not os.path.exists(path):
        print(f"  [{label}] NOT FOUND")
        return None
    size = os.path.getsize(path)
    n = size // RECORD_SIZE
    data = np.frombuffer(open(path, 'rb').read(n * RECORD_SIZE), dtype=np.uint8).reshape(n, RECORD_SIZE)
    pit_sums = data[:, 0:9].sum(axis=1).astype(np.int32) + data[:, 9:18].sum(axis=1).astype(np.int32)
    eg = np.sum(pit_sums <= 30)
    print(f"  [{label}] {n:,} pos, endgame: {eg:,} ({100*eg/n:.1f}%)")
    return data


def endgame(data, threshold=30):
    pit_sums = data[:, 0:9].sum(axis=1).astype(np.int32) + data[:, 9:18].sum(axis=1).astype(np.int32)
    return data[pit_sums <= threshold]


print("=== V9b: Clean data (no synthetic) ===\n")
all_data = []

d = load('v7_combined.bin', 'V7 combined')
if d is not None:
    all_data.append(d)
    eg = endgame(d)
    for _ in range(3):  # 3x endgame weight
        all_data.append(eg)
    print(f"    + {len(eg)*3:,} endgame copies (3x)")

d = load('v7_improved_training_data.bin', 'V7 improved')
if d is not None:
    all_data.append(d)
    eg = endgame(d)
    for _ in range(3):
        all_data.append(eg)
    print(f"    + {len(eg)*3:,} endgame copies (3x)")

d = load('master_training.bin', 'Master 1800+')
if d is not None:
    all_data.append(d)

d = load('master_all_elo_training.bin', 'Master all')
if d is not None:
    all_data.append(d)

merged = np.concatenate(all_data, axis=0)
print(f"\nTotal: {len(merged):,}")

pit_sums = merged[:, 0:9].sum(axis=1).astype(np.int32) + merged[:, 9:18].sum(axis=1).astype(np.int32)
print(f"Endgame: {np.sum(pit_sums <= 30):,} ({100*np.sum(pit_sums <= 30)/len(merged):.1f}%)")

np.random.seed(2025)
np.random.shuffle(merged)

Path('v9b_combined.bin').write_bytes(merged.tobytes())
print(f"Saved: v9b_combined.bin ({os.path.getsize('v9b_combined.bin')/1e6:.1f} MB)")
