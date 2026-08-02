#!/usr/bin/env python3
"""Shared engine-provenance helper for the match harnesses.

Both `tools/ab_match.py` (engine-vs-engine) and `tools/9qum/match.py` (vs the
external 9qum opponent) need to stamp every saved game record with which engine
binary, which weights (sha256), and which code (git commit) produced it. Without
this, telling two runs apart means reconstructing it after the fact from game ids
scattered across log files -- that already happened once and mis-grouped a real
analysis (see the history of tools/9qum/match.py).

`compute_engine_meta` was originally written directly in tools/9qum/match.py; it now
lives here as the single source so `tools/ab_match.py` reuses it rather than growing
a second, slightly-different implementation.
"""
import hashlib
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def compute_engine_meta(engine_path, repo=REPO):
    """Identify which engine build + weights + code produced a match, once per run.

    Returns a dict with `engine_path` (resolved, always present), and
    `engine_weights_path`/`engine_weights_size`/`engine_weights_sha256` (None if no
    `nnue_weights.bin` sits beside the binary, or it can't be read), and `git_commit`
    (None outside a git checkout, or if git itself is unavailable) -- never raises.
    """
    engine_path = Path(engine_path).resolve()
    weights_path = engine_path.parent / "nnue_weights.bin"
    weights_size = weights_sha256 = None
    if weights_path.is_file():
        try:
            data = weights_path.read_bytes()
            weights_size = len(data)
            weights_sha256 = hashlib.sha256(data).hexdigest()
        except OSError:
            pass  # unreadable: leave size/sha256 as None rather than failing the match

    git_commit = None
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            git_commit = out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass  # not a git checkout, or git unavailable: leave None rather than failing

    return {
        "engine_path": str(engine_path),
        "engine_weights_path": str(weights_path) if weights_path.is_file() else None,
        "engine_weights_size": weights_size,
        "engine_weights_sha256": weights_sha256,
        "git_commit": git_commit,
    }
