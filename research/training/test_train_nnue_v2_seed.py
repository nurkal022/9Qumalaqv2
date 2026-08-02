#!/usr/bin/env python3
"""Reproducibility of research/training/train_nnue_v2.py, tested WITHOUT training a
real net.

The trainer had no random seed: every run drew a different weight initialisation and
minibatch shuffle order from whatever RNG state torch happened to be in. Two runs of
the exact same recipe therefore landed 0.001-0.002 apart in val loss purely from that
noise -- enough that four separate candidate-vs-candidate comparisons made during one
night's experiments were uninterpretable (nobody could tell a real improvement from
RNG noise). train_nnue_v2.py now takes a --seed flag (fixed default, so this is true
even for a bare invocation) that seeds python random / numpy / torch (incl. CUDA) via
set_seed(), and prints the seed in its summary.

This test runs the real script twice, end to end, as a subprocess (so it exercises
exactly what an operator runs) but kept cheap with --limit (first N records only) and
--epochs 1, forced onto --device cpu (some CUDA ops, e.g. EmbeddingBag's backward, have
no deterministic implementation, which would make a GPU run of this test flaky). No
network access; CPU cost is small (a few thousand records, one epoch).

Run: python3.12 research/training/test_train_nnue_v2_seed.py
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(REPO, "research", "training", "train_nnue_v2.py")
TRAIN_BIN = os.path.join(REPO, "data", "9qum", "train", "train.bin")
VAL_BIN = os.path.join(REPO, "data", "9qum", "train", "val.bin")

EPOCH_RE = re.compile(r"^epoch\s+\d+\s+train\s+[\d.]+\s+val\s+[\d.]+$", re.MULTILINE)


def run_training(seed, out_dir, limit=3000):
    result = subprocess.run(
        [sys.executable, SCRIPT,
         "--train", TRAIN_BIN, "--val", VAL_BIN,
         "--limit", str(limit), "--epochs", "1", "--seed", str(seed),
         "--device", "cpu", "--out", out_dir, "--name", "seedtest"],
        capture_output=True, text=True, cwd=REPO, timeout=120,
    )
    assert result.returncode == 0, (
        f"training subprocess failed (seed={seed}):\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    return result.stdout


def first_epoch_line(stdout):
    m = EPOCH_RE.search(stdout)
    assert m, f"no 'epoch 1 ...' line found in stdout:\n{stdout}"
    return m.group(0)


def test_same_seed_gives_bit_identical_first_epoch_loss():
    tmp = tempfile.mkdtemp(prefix="nnue_seed_same_")
    try:
        out_a = run_training(seed=42, out_dir=os.path.join(tmp, "a"))
        out_b = run_training(seed=42, out_dir=os.path.join(tmp, "b"))
        line_a, line_b = first_epoch_line(out_a), first_epoch_line(out_b)
        assert line_a == line_b, (
            f"same --seed 42 must give a bit-identical first-epoch loss line:\n"
            f"  run A: {line_a!r}\n  run B: {line_b!r}"
        )
        assert "seed 42" in out_a and "seed 42" in out_b, "the seed must be recorded in the printed summary"
        assert f"seed={42}" in out_a.splitlines()[-1], "the final summary line must record the seed"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_different_seeds_give_different_first_epoch_loss():
    tmp = tempfile.mkdtemp(prefix="nnue_seed_diff_")
    try:
        out_a = run_training(seed=1, out_dir=os.path.join(tmp, "a"))
        out_b = run_training(seed=2, out_dir=os.path.join(tmp, "b"))
        line_a, line_b = first_epoch_line(out_a), first_epoch_line(out_b)
        assert line_a != line_b, (
            f"different seeds must NOT coincidentally give the same first-epoch loss line "
            f"(both were {line_a!r}) -- if this ever legitimately happens, pick different "
            f"probe seeds, don't weaken the assertion"
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


TESTS = [
    test_same_seed_gives_bit_identical_first_epoch_loss,
    test_different_seeds_give_different_first_epoch_loss,
]


if __name__ == "__main__":
    for t in TESTS:
        t()
    print(f"OK: train_nnue_v2.py --seed is reproducible and seed-sensitive ({len(TESTS)}/{len(TESTS)})")
