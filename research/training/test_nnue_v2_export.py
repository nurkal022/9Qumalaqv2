#!/usr/bin/env python3
"""A net that trains in torch and plays in Rust must compute the same number.

This test catches layout, ordering and bucket-selection mistakes in the exporter — the
class of bug that shows up as "the net was great in training and weak in play".

Run: python3.12 research/training/test_nnue_v2_export.py
"""
import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
import features_v2 as fv
import train_nnue_v2 as tn

ENGINE = "target/release/togyzkumalaq-engine"
TMP = "/tmp/nnue_v2_equality.bin"

POSITIONS = [
    ([9] * 18, [0, 0], [None, None], 0),
    ([1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 8, 7, 6, 5, 4, 3, 2, 1], [10, 12], [15, 4], 1),
    ([0, 0, 2, 0, 1, 0, 3, 0, 0, 1, 0, 0, 4, 0, 0, 2, 0, 0], [70, 66], [11, None], 0),
    ([0] * 9 + [2, 0, 0, 0, 0, 0, 0, 0, 0], [80, 80], [None, 3], 1),
]


def test_vectorised_features_match_scalar():
    """train_nnue_v2.build_feature_matrix is a third implementation of the layout (after
    Rust and the scalar Python one). Without this check it can drift and the net trains on
    features nothing else produces."""
    import numpy as np
    pits = np.array([p for p, _, _, _ in POSITIONS], dtype=np.int64)
    kazan = np.array([k for _, k, _, _ in POSITIONS], dtype=np.int64)
    tuz = np.array([[-1 if t[0] is None else t[0] - 9, -1 if t[1] is None else t[1]]
                    for _, _, t, _ in POSITIONS], dtype=np.int64)
    stm = np.array([s for _, _, _, s in POSITIONS], dtype=np.int64)
    feats, phase = tn.build_feature_matrix(pits, kazan, tuz, stm)
    for i, (p, k, t, s) in enumerate(POSITIONS):
        want = sorted(fv.build_features(p, k, t, s))
        got = sorted(int(x) for x in feats[i])
        assert want == got, f"row {i}: vectorised {got} != scalar {want}"
        assert int(phase[i]) == fv.phase_bucket(p), f"row {i}: phase bucket differs"
    print(f"OK: vectorised and scalar feature builders agree on {len(POSITIONS)} rows")


def main():
    test_vectorised_features_match_scalar()
    torch.manual_seed(0)
    model = tn.NnueV2()
    model.eval()
    tn.export_nnu2(model, TMP)
    worst = 0.0
    for pits, kazan, tuz, stm in POSITIONS:
        feats = fv.build_features(pits, kazan, tuz, stm)
        bucket = fv.phase_bucket(pits)
        with torch.no_grad():
            want = model.forward_single(feats, bucket).item()
        pos = fv.pos_string(pits, kazan, tuz, stm)
        out = subprocess.run([ENGINE, "evalpos", TMP, pos],
                             capture_output=True, text=True, check=True)
        got = float(out.stdout.split()[1])
        worst = max(worst, abs(want - got))
        assert abs(want - got) < 1e-3, f"{pos}: torch {want:.6f} vs rust {got:.6f}"
    print(f"OK: torch and Rust agree on {len(POSITIONS)} positions (max diff {worst:.2e})")


if __name__ == "__main__":
    main()
