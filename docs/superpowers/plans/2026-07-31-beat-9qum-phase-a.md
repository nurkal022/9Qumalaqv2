# Phase A — Rebuild the evaluation (beat 9qum's net) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the engine's evaluation input encoding and head so the eval stops being
material-anchored in the endgame (76.1% accuracy today vs 94.9% for 9qum's net), trained on
9qum's calibrated out-of-lineage win% labels.

**Architecture:** The current NNUE feeds each pit's stone count as one scaled scalar, so exact
counts and parity — what actually decides togyz endgames — are close to unrepresentable. We
replace it with a sparse bucketed binary input (292 features, 23 active), a 1024-wide
accumulator, and 4 phase-specific output heads so endgame weights stop competing with opening
weights. Weights are f32 in a new `NNU2` file format; the legacy loader stays untouched.
Training data comes from a converter over the harvested 9qum corpus.

**Tech Stack:** Rust (engine, `cargo 1.93`), Python 3.12 (`python3.12` — the only interpreter
with torch 2.10+cu128 and numpy; plain `python3` is miniconda base and has neither), torch,
numpy. No pytest anywhere in this environment: tests are plain assert scripts with a
`if __name__ == "__main__"` runner, matching `research/training/test_sweep_rule.py`.

**Spec:** `docs/superpowers/specs/2026-07-31-beat-9qum-design.md`

## Global Constraints

- Run every Python command with `python3.12`. `python3` has no torch/numpy.
- Build the engine with `cargo build --release` from the repo root; the binary lands in
  `target/release/togyzkumalaq-engine`. Engine assets (`nnue_weights.bin`, `egtb.bin`,
  `opening_book.txt`) load **binary-relative**, so a test binary must sit next to them or be
  pointed at them explicitly.
- The production engine `models/engine/baseline` is never overwritten in this plan. New weights
  go to `models/nets/nnue_v2/`.
- Harvested corpus lives in `data/9qum/` and is gitignored. Never commit it.
- Train/val splits are **by game id, never by ply**: plies inside a game are autocorrelated and
  a per-ply split leaks the outcome.
- Value in a training record is the win probability **for the side to move**, in [0,1].
- The exact feature layout is 292 features with 23 active per position. (The spec says "≈313 /
  ~40 active"; those were approximations, this plan is the exact instantiation.)
- Commit after every task. Do not batch commits.

---

### Task 1: Feature layout v2 in Rust + `features` CLI

**Files:**
- Modify: `engine/src/nnue.rs` (add constants, builders and an inline test module)
- Modify: `engine/src/main.rs:156` (add a `features` subcommand next to `"serve"`)

**Interfaces:**
- Consumes: `core::board::Board` fields `pits[side][i]`, `kazan[side]`, `tuzdyk[side]` (`i8`,
  `-1` = none), `side_to_move.index()`, and `NUM_PITS`.
- Produces:
  - `pub const NUM_FEATURES_V2: usize = 292;`
  - `pub const NUM_BUCKETS_V2: usize = 4;`
  - `pub fn build_features_v2(board: &Board) -> Vec<u16>` — sorted-by-construction list of
    active feature indices (always 23 entries)
  - `pub fn phase_bucket(board: &Board) -> usize` — 0..=3
  - CLI: `togyzkumalaq-engine features <pos>` prints the active indices space-separated, where
    `<pos>` is the existing position string `w0,..,w8/b0,..,b8/kw,kb/tw,tb/side`

**Layout (this is the contract Task 2 mirrors):**

```
0..126     me pits:   pit i -> i*14 + count_bucket(count)
126..252   opp pits:  pit i -> 126 + i*14 + count_bucket(count)
252..261   me kazan:  252 + min(8, kazan/10)
261..270   opp kazan: 261 + min(8, kazan/10)
270..280   me tuzdyk:  270 + (pit index 0..8, or 9 when none)
280..290   opp tuzdyk: 280 + (pit index 0..8, or 9 when none)
290..292   parity:    290 + (total stones on board % 2)

count_bucket(c) = c for c <= 9, 10 for 10..12, 11 for 13..16, 12 for 17..24, 13 for >= 25
phase_bucket:  total stones on board  121..162 -> 0, 81..120 -> 1, 41..80 -> 2, 0..40 -> 3
```

- [ ] **Step 1: Write the failing tests**

Append to `engine/src/nnue.rs`:

```rust
#[cfg(test)]
mod tests_v2 {
    use super::*;
    use board::Board;

    #[test]
    fn start_position_features() {
        let b = Board::new();
        let f = build_features_v2(&b);
        assert_eq!(f.len(), 23, "23 active features per position");
        // every pit holds 9 stones -> bucket 9
        for i in 0..9 {
            assert!(f.contains(&((i * 14 + 9) as u16)), "me pit {i} bucket 9");
            assert!(f.contains(&((126 + i * 14 + 9) as u16)), "opp pit {i} bucket 9");
        }
        assert!(f.contains(&252), "me kazan 0 -> bucket 0");
        assert!(f.contains(&261), "opp kazan 0 -> bucket 0");
        assert!(f.contains(&(270 + 9)), "no tuzdyk for me");
        assert!(f.contains(&(280 + 9)), "no tuzdyk for opp");
        assert!(f.contains(&290), "162 stones on board -> even parity");
        assert_eq!(phase_bucket(&b), 0, "full board is phase bucket 0");
    }

    #[test]
    fn count_buckets_are_step_functions() {
        assert_eq!(count_bucket(0), 0);
        assert_eq!(count_bucket(2), 2);      // the tuzdyk-threat count must be its own bucket
        assert_eq!(count_bucket(9), 9);
        assert_eq!(count_bucket(10), 10);
        assert_eq!(count_bucket(12), 10);
        assert_eq!(count_bucket(13), 11);
        assert_eq!(count_bucket(16), 11);
        assert_eq!(count_bucket(17), 12);
        assert_eq!(count_bucket(24), 12);
        assert_eq!(count_bucket(25), 13);
        assert_eq!(count_bucket(90), 13);
    }

    #[test]
    fn tuzdyk_and_phase_are_encoded() {
        let mut b = Board::new();
        for i in 0..9 {
            b.pits[0][i] = 1;
            b.pits[1][i] = 1;
        }
        b.kazan[0] = 72;
        b.kazan[1] = 72;
        b.tuzdyk[0] = 6;
        b.tuzdyk[1] = -1;
        let f = build_features_v2(&b);
        assert!(f.contains(&(270 + 6)), "me tuzdyk on pit 6");
        assert!(f.contains(&(280 + 9)), "opp has no tuzdyk");
        assert!(f.contains(&(252 + 7)), "kazan 72 -> bucket 7");
        assert_eq!(phase_bucket(&b), 3, "18 stones on board is the last phase bucket");
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p togyzkumalaq-engine tests_v2 2>&1 | tail -20`
Expected: FAIL — `cannot find function build_features_v2` / `count_bucket` / `phase_bucket`.

- [ ] **Step 3: Implement the builders**

Add to `engine/src/nnue.rs` (above the `#[cfg(test)]` module):

```rust
pub const NUM_FEATURES_V2: usize = 292;
pub const NUM_BUCKETS_V2: usize = 4;
const ACTIVE_FEATURES_V2: usize = 23;

/// Stone counts enter as one-hot buckets, not as a scalar: endgames turn on exact counts
/// and parity (a pit holding exactly 2 is a tuzdyk threat), and a first layer over a
/// scaled scalar can only rescale it.
fn count_bucket(c: u8) -> usize {
    match c {
        0..=9 => c as usize,
        10..=12 => 10,
        13..=16 => 11,
        17..=24 => 12,
        _ => 13,
    }
}

fn board_stones(board: &Board) -> u32 {
    (0..NUM_PITS)
        .map(|i| board.pits[0][i] as u32 + board.pits[1][i] as u32)
        .sum()
}

pub fn build_features_v2(board: &Board) -> Vec<u16> {
    let me = board.side_to_move.index();
    let opp = 1 - me;
    let mut f = Vec::with_capacity(ACTIVE_FEATURES_V2);
    for i in 0..NUM_PITS {
        f.push((i * 14 + count_bucket(board.pits[me][i])) as u16);
    }
    for i in 0..NUM_PITS {
        f.push((126 + i * 14 + count_bucket(board.pits[opp][i])) as u16);
    }
    f.push((252 + (board.kazan[me] as usize / 10).min(8)) as u16);
    f.push((261 + (board.kazan[opp] as usize / 10).min(8)) as u16);
    let tuz = |t: i8| if t >= 0 { t as usize } else { 9 };
    f.push((270 + tuz(board.tuzdyk[me])) as u16);
    f.push((280 + tuz(board.tuzdyk[opp])) as u16);
    f.push((290 + (board_stones(board) % 2) as usize) as u16);
    debug_assert_eq!(f.len(), ACTIVE_FEATURES_V2);
    f
}

pub fn phase_bucket(board: &Board) -> usize {
    match board_stones(board) {
        121..=162 => 0,
        81..=120 => 1,
        41..=80 => 2,
        _ => 3,
    }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p togyzkumalaq-engine tests_v2 2>&1 | tail -10`
Expected: PASS, 3 tests.

- [ ] **Step 5: Add the `features` CLI subcommand**

In `engine/src/main.rs`, inside the `match args[1].as_str()` block, next to `"serve" => run_serve(),`:

```rust
            "features" => {
                // features <pos>  -> active NNUE v2 feature indices, for the Python cross-check
                let pos = args.get(2).map(|s| s.as_str()).unwrap_or("");
                match board::parse_position(pos) {
                    Ok(b) => {
                        let f = nnue::build_features_v2(&b);
                        let idx: Vec<String> = f.iter().map(|x| x.to_string()).collect();
                        println!("{} bucket {}", idx.join(" "), nnue::phase_bucket(&b));
                    }
                    Err(e) => {
                        eprintln!("error: {e}");
                        std::process::exit(2);
                    }
                }
            }
```

Both paths already resolve: `board::parse_position` is the parser `run_serve` uses
(`engine/src/main.rs:887`) and `mod nnue;` is declared at `engine/src/main.rs:6`. No new
imports, and do not write a second parser.

- [ ] **Step 6: Verify the CLI on the start position**

Run:
```bash
cargo build --release 2>&1 | tail -3
./target/release/togyzkumalaq-engine features "9,9,9,9,9,9,9,9,9/9,9,9,9,9,9,9,9,9/0,0/-1,-1/0"
```
Expected: 23 indices ending with `290 bucket 0`; the first nine are `9 23 37 51 65 79 93 107 121`.

- [ ] **Step 7: Commit**

```bash
git add engine/src/nnue.rs engine/src/main.rs
git commit -m "engine: NNUE v2 sparse bucketed feature layout + features CLI"
```

---

### Task 2: Python mirror of the layout + cross-implementation test

**Files:**
- Create: `research/data/features_v2.py`
- Create: `research/data/test_features_v2.py`

**Interfaces:**
- Consumes: the CLI `togyzkumalaq-engine features <pos>` from Task 1.
- Produces:
  - `features_v2.count_bucket(c: int) -> int`
  - `features_v2.build_features(pits: list[int], kazan: list[int], tuzdyk: list[int|None], to_move: int) -> list[int]`
    where `pits` is the 18-long absolute array (side 0 first), `tuzdyk[p]` is an **absolute**
    pit index 0..17 or None (the 9qum replay convention)
  - `features_v2.phase_bucket(pits) -> int`
  - `features_v2.pos_string(pits, kazan, tuzdyk, to_move) -> str` (engine position string)

- [ ] **Step 1: Write the failing test**

`research/data/test_features_v2.py`:

```python
#!/usr/bin/env python3
"""The Python feature builder must agree with the Rust one feature-for-feature.

Two implementations of one layout is how silent training/inference skew happens: the net
learns on Python features and plays on Rust features. This test compares both on real
positions from the harvested corpus.

Run: python3.12 research/data/test_features_v2.py
"""
import gzip
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
import features_v2 as fv

ENGINE = "target/release/togyzkumalaq-engine"
REPLAYS = "data/9qum/games/replays.jsonl.gz"


def rust_features(pos: str):
    out = subprocess.run([ENGINE, "features", pos], capture_output=True, text=True, check=True)
    parts = out.stdout.strip().split()
    bucket = int(parts[parts.index("bucket") + 1])
    idx = [int(x) for x in parts[: parts.index("bucket")]]
    return sorted(idx), bucket


def sample_states(n):
    states = []
    with gzip.open(REPLAYS, "rt", encoding="utf-8") as f:
        for line in f:
            g = json.loads(line)
            for st in (g.get("states") or [])[::9]:
                states.append(st)
                if len(states) >= n:
                    return states
    return states


def test_start_position():
    idx, bucket = rust_features("9,9,9,9,9,9,9,9,9/9,9,9,9,9,9,9,9,9/0,0/-1,-1/0")
    mine = sorted(fv.build_features([9] * 18, [0, 0], [None, None], 0))
    assert mine == idx, f"start position differs: {mine} vs {idx}"
    assert bucket == fv.phase_bucket([9] * 18) == 0
    assert len(mine) == 23


def test_matches_rust_on_real_positions():
    states = sample_states(200)
    assert len(states) >= 200, "need the harvested corpus in data/9qum"
    for st in states:
        pos = fv.pos_string(st["pits"], st["kazan"], st["tuzdyk"], st["to_move"])
        idx, bucket = rust_features(pos)
        mine = sorted(fv.build_features(st["pits"], st["kazan"], st["tuzdyk"], st["to_move"]))
        assert mine == idx, f"mismatch at {pos}:\n python {mine}\n rust   {idx}"
        assert bucket == fv.phase_bucket(st["pits"]), f"bucket mismatch at {pos}"


if __name__ == "__main__":
    test_start_position()
    test_matches_rust_on_real_positions()
    print("OK: Python and Rust feature builders agree (2/2)")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python3.12 research/data/test_features_v2.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'features_v2'`.

- [ ] **Step 3: Implement the mirror**

`research/data/features_v2.py`:

```python
#!/usr/bin/env python3
"""NNUE v2 feature layout — the Python side of the contract in engine/src/nnue.rs.

292 features, 23 active. Keep this file and build_features_v2() in nnue.rs in lockstep;
test_features_v2.py fails if they drift.
"""
NUM_FEATURES = 292
NUM_BUCKETS = 4


def count_bucket(c: int) -> int:
    if c <= 9:
        return c
    if c <= 12:
        return 10
    if c <= 16:
        return 11
    if c <= 24:
        return 12
    return 13


def _tuz_rel(tuzdyk, side):
    """9qum stores tuzdyk[p] as an absolute pit index on the opponent's side; the engine
    wants the index inside that side's row, or 9 for none."""
    t = tuzdyk[side]
    if t is None or t < 0:
        return 9
    return t - 9 if side == 0 else t


def build_features(pits, kazan, tuzdyk, to_move):
    me, opp = to_move, 1 - to_move
    rows = [pits[0:9], pits[9:18]]
    f = []
    for i in range(9):
        f.append(i * 14 + count_bucket(rows[me][i]))
    for i in range(9):
        f.append(126 + i * 14 + count_bucket(rows[opp][i]))
    f.append(252 + min(8, kazan[me] // 10))
    f.append(261 + min(8, kazan[opp] // 10))
    f.append(270 + _tuz_rel(tuzdyk, me))
    f.append(280 + _tuz_rel(tuzdyk, opp))
    f.append(290 + (sum(pits) % 2))
    assert len(f) == 23
    return f


def phase_bucket(pits) -> int:
    total = sum(pits)
    if total >= 121:
        return 0
    if total >= 81:
        return 1
    if total >= 41:
        return 2
    return 3


def pos_string(pits, kazan, tuzdyk, to_move) -> str:
    tw = -1 if tuzdyk[0] is None else tuzdyk[0] - 9
    tb = -1 if tuzdyk[1] is None else tuzdyk[1]
    return (",".join(map(str, pits[0:9])) + "/" + ",".join(map(str, pits[9:18])) +
            f"/{kazan[0]},{kazan[1]}/{tw},{tb}/{to_move}")
```

- [ ] **Step 4: Run it to verify it passes**

Run: `python3.12 research/data/test_features_v2.py`
Expected: `OK: Python and Rust feature builders agree (2/2)`

If the tuzdyk assertions fail, the likely cause is the relative/absolute convention: 9qum's
`tuzdyk[0]` sits on side 1 (indices 9..17) and `tuzdyk[1]` on side 0 (0..8). Fix
`_tuz_rel`, not the Rust side — Rust already receives engine-relative indices.

- [ ] **Step 5: Commit**

```bash
git add research/data/features_v2.py research/data/test_features_v2.py
git commit -m "train: Python mirror of the NNUE v2 feature layout + Rust cross-check"
```

---

### Task 3: `NNU2` weight format + Rust loader, eval and `evalpos` CLI

**Files:**
- Modify: `engine/src/nnue.rs` (magic detection in `load`, v2 fields, `evaluate` dispatch)
- Modify: `engine/src/main.rs` (add an `evalpos` subcommand)

**Interfaces:**
- Consumes: `build_features_v2`, `phase_bucket` (Task 1).
- Produces:
  - v2 file format (little-endian throughout):
    ```
    u32 magic = 0x324E554E   ("NNU2")
    u16 version = 2
    u16 num_features = 292
    u16 acc_size = 1024
    u16 hidden = 32
    u16 buckets = 4
    u16 pad = 0
    f32 fc1_w[num_features * acc_size]      indexed fc1_w[feature * acc_size + j]
    f32 fc1_b[acc_size]
    per bucket:
        f32 fc2_w[acc_size * hidden]        indexed fc2_w[j * acc_size + i]
        f32 fc2_b[hidden]
        f32 fc3_w[hidden]
        f32 fc3_b[1]
    ```
  - `NnueNetwork::evaluate` returns the same units as before (search divides by 64). The net
    outputs a **logit** of "side to move wins"; the engine converts with `cp = 350 * logit`,
    clamped to ±3000, then ×64.
  - CLI: `togyzkumalaq-engine evalpos <weights.bin> <pos>` prints `logit <x> cp <y>`.

- [ ] **Step 1: Write the failing test**

Append to the `tests_v2` module in `engine/src/nnue.rs`:

```rust
    /// Build a minimal v2 file in memory: one feature contributes 1.0 to accumulator 0,
    /// bucket 0 reads accumulator 0 with weight 1.0, everything else is zero. Then the
    /// logit for the start position is exactly the number of active features that map to
    /// accumulator 0 — a value we can compute by hand.
    fn synthetic_v2(num_features: usize, acc: usize, hidden: usize, buckets: usize) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&0x324E554Eu32.to_le_bytes());
        for v in [2u16, num_features as u16, acc as u16, hidden as u16, buckets as u16, 0u16] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        let mut push_f32 = |out: &mut Vec<u8>, v: f32| out.extend_from_slice(&v.to_le_bytes());
        for f in 0..num_features {
            for j in 0..acc {
                // feature 9 (me pit 0 holding 9 stones) is the only one that fires
                push_f32(&mut out, if f == 9 && j == 0 { 1.0 } else { 0.0 });
            }
        }
        for _ in 0..acc {
            push_f32(&mut out, 0.0);
        }
        for b in 0..buckets {
            for j in 0..hidden {
                for i in 0..acc {
                    push_f32(&mut out, if b == 0 && j == 0 && i == 0 { 1.0 } else { 0.0 });
                }
            }
            for _ in 0..hidden {
                push_f32(&mut out, 0.0);
            }
            for j in 0..hidden {
                push_f32(&mut out, if b == 0 && j == 0 { 1.0 } else { 0.0 });
            }
            push_f32(&mut out, 0.0);
        }
        out
    }

    #[test]
    fn loads_v2_and_evaluates_by_hand() {
        let bytes = synthetic_v2(NUM_FEATURES_V2, 8, 4, NUM_BUCKETS_V2);
        let path = std::env::temp_dir().join("nnue_v2_synthetic.bin");
        std::fs::write(&path, &bytes).unwrap();
        let net = NnueNetwork::load(path.to_str().unwrap()).expect("v2 file loads");
        let b = Board::new();
        // start position fires feature 9 once -> acc0 = 1.0 -> hidden0 = 1.0 -> logit = 1.0
        let cp = net.evaluate(&b) / 64;
        assert_eq!(cp, 350, "logit 1.0 must map to 350 cp, got {cp}");
    }

    #[test]
    fn legacy_weights_still_load() {
        let net = NnueNetwork::load("models/engine/nnue_weights.bin")
            .expect("the shipped legacy weights must keep loading");
        let cp = net.evaluate(&Board::new());
        assert!(cp.abs() < 100_000, "legacy eval returns a sane number, got {cp}");
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p togyzkumalaq-engine tests_v2 2>&1 | tail -20`
Expected: FAIL — the v2 test errors while loading (the legacy loader reads the magic as a
hidden size), `legacy_weights_still_load` passes.

- [ ] **Step 3: Implement the v2 path**

In `engine/src/nnue.rs`:

1. Add fields to `NnueNetwork`:

```rust
    // --- v2 (NNU2) ---
    v2: bool,
    acc_size: usize,
    buckets: usize,
    v2_fc1_w: Vec<f32>,   // [num_features * acc_size]
    v2_fc1_b: Vec<f32>,   // [acc_size]
    v2_fc2_w: Vec<f32>,   // [buckets][hidden * acc_size] flattened
    v2_fc2_b: Vec<f32>,   // [buckets][hidden] flattened
    v2_fc3_w: Vec<f32>,   // [buckets][hidden] flattened
    v2_fc3_b: Vec<f32>,   // [buckets]
```

Initialise them to `false`/`0`/`Vec::new()` everywhere `NnueNetwork` is constructed in the
legacy path.

2. At the top of `load`, before any legacy parsing:

```rust
        let data = std::fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
        if data.len() >= 4 && u32::from_le_bytes([data[0], data[1], data[2], data[3]]) == 0x324E554E {
            return Self::load_v2(&data);
        }
```

Keep the rest of `load` exactly as it is (it already reads the file itself; if it re-reads,
pass `data` down instead of reading twice).

3. Add the v2 reader:

```rust
    fn load_v2(data: &[u8]) -> Result<Self, String> {
        let rd_u16 = |off: usize| u16::from_le_bytes([data[off], data[off + 1]]) as usize;
        let version = rd_u16(4);
        if version != 2 {
            return Err(format!("unsupported NNU2 version {version}"));
        }
        let num_features = rd_u16(6);
        let acc_size = rd_u16(8);
        let hidden = rd_u16(10);
        let buckets = rd_u16(12);
        if num_features != NUM_FEATURES_V2 {
            return Err(format!("expected {NUM_FEATURES_V2} features, file has {num_features}"));
        }
        let mut off = 16;
        let mut take = |n: usize| -> Result<Vec<f32>, String> {
            if off + n * 4 > data.len() {
                return Err(format!("NNU2 truncated: need {} bytes, have {}", off + n * 4, data.len()));
            }
            let v = (0..n)
                .map(|k| {
                    let p = off + k * 4;
                    f32::from_le_bytes([data[p], data[p + 1], data[p + 2], data[p + 3]])
                })
                .collect();
            off += n * 4;
            Ok(v)
        };
        let v2_fc1_w = take(num_features * acc_size)?;
        let v2_fc1_b = take(acc_size)?;
        let mut v2_fc2_w = Vec::new();
        let mut v2_fc2_b = Vec::new();
        let mut v2_fc3_w = Vec::new();
        let mut v2_fc3_b = Vec::new();
        for _ in 0..buckets {
            v2_fc2_w.extend(take(acc_size * hidden)?);
            v2_fc2_b.extend(take(hidden)?);
            v2_fc3_w.extend(take(hidden)?);
            v2_fc3_b.extend(take(1)?);
        }
        Ok(Self {
            input_size: num_features,
            hidden1: acc_size,
            hidden2: hidden,
            hidden3: 0,
            fc1_weight: Vec::new(), fc1_bias: Vec::new(),
            fc2_weight: Vec::new(), fc2_bias: Vec::new(),
            fc3_weight: Vec::new(), fc3_bias: Vec::new(),
            fc4_weight: Vec::new(), fc4_bias: Vec::new(),
            v2: true, acc_size, buckets,
            v2_fc1_w, v2_fc1_b, v2_fc2_w, v2_fc2_b, v2_fc3_w, v2_fc3_b,
        })
    }

    /// Win-probability logit for the side to move.
    pub fn logit_v2(&self, board: &Board) -> f32 {
        let acc_n = self.acc_size;
        let hidden = self.hidden2;
        let mut acc = self.v2_fc1_b.clone();
        for f in build_features_v2(board) {
            let base = f as usize * acc_n;
            for j in 0..acc_n {
                acc[j] += self.v2_fc1_w[base + j];
            }
        }
        for a in acc.iter_mut() {
            *a = a.max(0.0);
        }
        let b = phase_bucket(board).min(self.buckets - 1);
        let w2 = &self.v2_fc2_w[b * hidden * acc_n..(b + 1) * hidden * acc_n];
        let b2 = &self.v2_fc2_b[b * hidden..(b + 1) * hidden];
        let w3 = &self.v2_fc3_w[b * hidden..(b + 1) * hidden];
        let mut out = self.v2_fc3_b[b];
        for j in 0..hidden {
            let row = &w2[j * acc_n..(j + 1) * acc_n];
            let mut h = b2[j];
            for i in 0..acc_n {
                h += row[i] * acc[i];
            }
            if h > 0.0 {
                out += w3[j] * h;
            }
        }
        out
    }
```

4. At the very start of `evaluate`, dispatch:

```rust
        if self.v2 {
            let cp = (350.0 * self.logit_v2(board)).clamp(-3000.0, 3000.0);
            return (cp * 64.0) as i32;
        }
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p togyzkumalaq-engine tests_v2 2>&1 | tail -10`
Expected: PASS, 5 tests.

- [ ] **Step 5: Add the `evalpos` CLI**

In `engine/src/main.rs`, beside the `"features"` arm:

```rust
            "evalpos" => {
                // evalpos <weights.bin> <pos>  -> logit and cp, for the torch/Rust equality test
                let w = args.get(2).map(|s| s.as_str()).unwrap_or("");
                let pos = args.get(3).map(|s| s.as_str()).unwrap_or("");
                let net = match NnueNetwork::load(w) {
                    Ok(n) => n,
                    Err(e) => { eprintln!("error: {e}"); std::process::exit(2); }
                };
                match board::parse_position(pos) {
                    Ok(b) => println!("logit {:.6} cp {}", net.logit_v2(&b), net.evaluate(&b) / 64),
                    Err(e) => { eprintln!("error: {e}"); std::process::exit(2); }
                }
            }
```

- [ ] **Step 6: Verify the CLI runs**

Run:
```bash
cargo build --release 2>&1 | tail -3
./target/release/togyzkumalaq-engine evalpos models/engine/nnue_weights.bin \
  "9,9,9,9,9,9,9,9,9/9,9,9,9,9,9,9,9,9/0,0/-1,-1/0"
```
Expected: it prints `logit 0.000000 cp <legacy value>` — `logit_v2` is meaningless for a
legacy file, which is fine; the command must not crash.

- [ ] **Step 7: Commit**

```bash
git add engine/src/nnue.rs engine/src/main.rs
git commit -m "engine: NNU2 weight format, f32 sparse eval with phase buckets, evalpos CLI"
```

---

### Task 4: Converter — harvested corpus to training bins

**Files:**
- Create: `research/data/convert_9qum.py`
- Create: `research/data/test_convert_9qum.py`

**Interfaces:**
- Consumes: `data/9qum/games/replays.jsonl.gz`, `data/9qum/analysis/curves.jsonl.gz`,
  `features_v2.pos_string` (Task 2).
- Produces: `data/9qum/train/{train,val}.bin` of 68-byte records, plus
  `data/9qum/train/split.json` (`{"val_games": [...], "counts": {...}}`).
  Record layout, little-endian:
  ```
  0..9    pits[0]        u8 x9        (absolute: side 0 row)
  9..18   pits[1]        u8 x9
  18      kazan[0]       u8
  19      kazan[1]       u8
  20      tuzdyk[0]      i8           (engine-relative pit index, -1 = none)
  21      tuzdyk[1]      i8
  22      side_to_move   u8
  23..59  policy         f32 x9       one-hot of the played move, zeros if unknown
  59..63  value          f32          P(side to move wins), 0..1
  63..67  score          f32          final kazan diff for the side to move, / 82
  67      mask           u8           bit0 policy valid, bit1 value from their net,
                                      bit2 value from the game outcome, bit3 score valid
  ```
- Exposed for tests: `decode_record(buf: bytes) -> dict`, `RECORD_SIZE = 68`.

- [ ] **Step 1: Write the failing test**

`research/data/test_convert_9qum.py`:

```python
#!/usr/bin/env python3
"""Guards for the 9qum -> training-bin converter.

Three failure modes this catches, all of which silently poison training:
  * a record that no longer decodes to the position it came from
  * the value stored from the wrong side's perspective (today's class of bug)
  * the same game appearing in both train and val, which leaks the outcome

Run: python3.12 research/data/test_convert_9qum.py
"""
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(__file__))
import convert_9qum as cv

TRAIN = "data/9qum/train/train.bin"
VAL = "data/9qum/train/val.bin"
SPLIT = "data/9qum/train/split.json"


def read_records(path, limit=None):
    out = []
    with open(path, "rb") as f:
        while True:
            buf = f.read(cv.RECORD_SIZE)
            if len(buf) < cv.RECORD_SIZE:
                break
            out.append(cv.decode_record(buf))
            if limit and len(out) >= limit:
                break
    return out


def test_record_size_and_roundtrip():
    assert cv.RECORD_SIZE == 68
    rec = cv.encode_record(
        pits=[9] * 18, kazan=[0, 0], tuzdyk=[None, None], to_move=0,
        move=6, value=0.75, score=0.1, mask=cv.MASK_POLICY | cv.MASK_VALUE_NET | cv.MASK_SCORE)
    d = cv.decode_record(rec)
    assert d["pits"] == [9] * 18 and d["kazan"] == [0, 0]
    assert d["tuzdyk"] == [-1, -1] and d["to_move"] == 0
    assert d["policy"][6] == 1.0 and sum(d["policy"]) == 1.0
    assert abs(d["value"] - 0.75) < 1e-6 and abs(d["score"] - 0.1) < 1e-6
    assert d["mask"] & cv.MASK_VALUE_NET


def test_value_perspective_matches_outcomes():
    """A value stored for the wrong side turns the label set into noise with the sign
    flipped. Their net is 82-84% accurate, so a correct conversion must land near that."""
    recs = read_records(VAL, limit=20000)
    net = [r for r in recs if r["mask"] & cv.MASK_VALUE_NET]
    assert len(net) > 1000, f"expected net-labelled records in val, got {len(net)}"
    # value > 0.5 must mean "the side to move went on to win" more often than not
    agree = sum(1 for r in net if (r["value"] > 0.5) == (r["outcome_stm"] > 0.5))
    rate = agree / len(net)
    assert rate >= 0.80, f"value perspective looks wrong: only {100 * rate:.1f}% agreement"


def test_splits_are_disjoint_by_game():
    with open(SPLIT, encoding="utf-8") as f:
        split = json.load(f)
    val_games = set(split["val_games"])
    assert len(val_games) > 100, "val split is suspiciously small"
    train_games = set(split["train_games"])
    assert not (val_games & train_games), "a game appears in both splits"
    assert os.path.getsize(TRAIN) % cv.RECORD_SIZE == 0
    assert os.path.getsize(VAL) % cv.RECORD_SIZE == 0


if __name__ == "__main__":
    test_record_size_and_roundtrip()
    test_value_perspective_matches_outcomes()
    test_splits_are_disjoint_by_game()
    print("OK: converter records, value perspective and splits (3/3)")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3.12 research/data/test_convert_9qum.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'convert_9qum'`.

- [ ] **Step 3: Implement the converter**

`research/data/convert_9qum.py`:

```python
#!/usr/bin/env python3
"""Turn the harvested 9qum corpus into NNUE v2 training records.

Sources per position:
  * value  — 9qum's own win% for that ply when we have it (calibrated, out-of-lineage,
             Brier 0.111 vs our engine's 0.195), otherwise the game outcome
  * policy — the move a human actually played
  * score  — the final kazan difference, kept for the phase-B score head

Games that ended on a flag-fall or an abandon are dropped: the recorded winner says
nothing about the position. Splits are by game, never by ply.

Run: python3.12 research/data/convert_9qum.py --out data/9qum/train
"""
import argparse
import gzip
import hashlib
import json
import os
import struct

RECORD_SIZE = 68
MASK_POLICY = 1
MASK_VALUE_NET = 2
MASK_VALUE_OUTCOME = 4
MASK_SCORE = 8
PLAYED_OUT = ("по камням", "сдача")


def _tuz_rel(tuzdyk, side):
    t = tuzdyk[side]
    if t is None or t < 0:
        return -1
    return t - 9 if side == 0 else t


def encode_record(pits, kazan, tuzdyk, to_move, move, value, score, mask, outcome_stm=0.0):
    pol = [0.0] * 9
    if move is not None and 0 <= move < 9:
        pol[move] = 1.0
    return (bytes(bytearray(pits)) +
            bytes(bytearray([kazan[0], kazan[1]])) +
            struct.pack("<bb", _tuz_rel(tuzdyk, 0), _tuz_rel(tuzdyk, 1)) +
            struct.pack("<B", to_move) +
            struct.pack("<9f", *pol) +
            struct.pack("<f", value) +
            struct.pack("<f", score) +
            struct.pack("<B", mask))


def decode_record(buf):
    pits = list(buf[0:18])
    kazan = [buf[18], buf[19]]
    tw, tb = struct.unpack("<bb", buf[20:22])
    to_move = buf[22]
    policy = list(struct.unpack("<9f", buf[23:59]))
    value = struct.unpack("<f", buf[59:63])[0]
    score = struct.unpack("<f", buf[63:67])[0]
    mask = buf[67]
    # a record's own value is the training target; outcome_stm is recoverable from score's
    # sign, which is what the perspective test checks
    return {"pits": pits, "kazan": kazan, "tuzdyk": [tw, tb], "to_move": to_move,
            "policy": policy, "value": value, "score": score, "mask": mask,
            "outcome_stm": 1.0 if score > 0 else 0.0}


def jsonl(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def is_val(game_id, val_pct):
    h = int(hashlib.md5(game_id.encode()).hexdigest()[:8], 16)
    return (h % 100) < val_pct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/9qum")
    ap.add_argument("--out", default="data/9qum/train")
    ap.add_argument("--val-pct", type=int, default=10)
    ap.add_argument("--min-ply", type=int, default=20)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    curves = {}
    cpath = os.path.join(a.corpus, "analysis", "curves.jsonl.gz")
    if os.path.exists(cpath):
        for c in jsonl(cpath):
            curves[c["game_id"]] = {p["ply"]: p["win"] for p in c.get("points") or []}
    print(f"curves: {len(curves)} games")

    fh = {"train": open(os.path.join(a.out, "train.bin"), "wb"),
          "val": open(os.path.join(a.out, "val.bin"), "wb")}
    games = {"train": set(), "val": set()}
    counts = {"train": 0, "val": 0, "net": 0, "outcome": 0, "dropped_games": 0}

    for g in jsonl(os.path.join(a.corpus, "games", "replays.jsonl.gz")):
        if g.get("reason") not in PLAYED_OUT or g.get("winner") not in (0, 1):
            counts["dropped_games"] += 1
            continue
        states = g.get("states") or []
        if len(states) < a.min_ply:
            counts["dropped_games"] += 1
            continue
        gid = g["game_id"]
        which = "val" if is_val(gid, a.val_pct) else "train"
        games[which].add(gid)
        winner = g["winner"]
        final = states[-1]
        kd0 = final["kazan"][0] - final["kazan"][1]
        cv = curves.get(gid, {})
        moves = states[0].get("moves") or []
        for ply, st in enumerate(states):
            stm = st["to_move"]
            outcome_stm = 1.0 if winner == stm else 0.0
            if ply in cv:
                win0 = cv[ply] / 100.0
                value = win0 if stm == 0 else 1.0 - win0
                mask = MASK_VALUE_NET
                counts["net"] += 1
            else:
                value = outcome_stm
                mask = MASK_VALUE_OUTCOME
                counts["outcome"] += 1
            move = moves[ply]["hole"] if ply < len(moves) else None
            if move is not None:
                mask |= MASK_POLICY
            score = (kd0 if stm == 0 else -kd0) / 82.0
            mask |= MASK_SCORE
            fh[which].write(encode_record(st["pits"], st["kazan"], st["tuzdyk"], stm,
                                          move, value, score, mask, outcome_stm))
            counts[which] += 1

    for f in fh.values():
        f.close()
    with open(os.path.join(a.out, "split.json"), "w", encoding="utf-8") as f:
        json.dump({"val_games": sorted(games["val"]), "train_games": sorted(games["train"]),
                   "counts": counts}, f)
    print(f"train {counts['train']:,} records / {len(games['train'])} games; "
          f"val {counts['val']:,} / {len(games['val'])} games; "
          f"net-labelled {counts['net']:,}, outcome-only {counts['outcome']:,}, "
          f"games dropped {counts['dropped_games']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the converter, then the test**

Run:
```bash
python3.12 research/data/convert_9qum.py --out data/9qum/train
python3.12 research/data/test_convert_9qum.py
```
Expected: the converter prints record counts (order of 400k train records), then
`OK: converter records, value perspective and splits (3/3)`.

If `test_value_perspective_matches_outcomes` reports ~16-20% agreement, the value was
stored from the opponent's perspective — fix the `stm == 0` branch, do not relax the test.

- [ ] **Step 5: Commit**

```bash
git add research/data/convert_9qum.py research/data/test_convert_9qum.py
git commit -m "train: converter from the 9qum corpus to NNUE v2 records (68-byte, by-game splits)"
```

---

### Task 5: Monitors and the A0 baseline

**Files:**
- Create: `tools/9qum/monitors.py`

**Interfaces:**
- Consumes: `data/9qum/train/val.bin`, `data/9qum/train/split.json`,
  `data/9qum/games/replays.jsonl.gz`, `tools/playok/engine.py`'s `Engine` (serve protocol).
- Produces: `monitors.value_report(engine_path, ms) -> dict` and
  `monitors.policy_report(engine_path, ms, min_rating) -> dict`, plus a CLI that prints both
  and appends a row to `data/9qum/train/monitors.jsonl`.

**Definitions (from the spec, do not change them or the numbers stop comparing):**
- *value accuracy* — over held-out games only, the fraction of positions where the sign of the
  predicted win probability matches the actual result of that game, inside the bucket:
  midgame `40 <= ply < 80`; close endgame `ply >= 80 and |kazan diff| <= 8`; clear endgame
  `ply >= 80 and |kazan diff| >= 20`.
- *policy match-rate* — the fraction of positions from ≥2000-rated players where the engine's
  best move equals the move actually played (plus the top-3 rate for the net in phase B).
- Targets to beat: 86 / 91 / 95% and Brier ≤ 0.11. Today's engine: 78.3 / 77.0 / 76.1%,
  Brier 0.195. 9qum's net: 85.5 / 91.2 / 94.9%, Brier 0.111.

- [ ] **Step 1: Write the monitor script**

`tools/9qum/monitors.py`:

```python
#!/usr/bin/env python3
"""Cheap offline monitors for an engine's evaluation and move choice.

These are the fast loop: minutes per candidate, no calls to 9qum's API, so iteration is not
bounded by their rate limit. The gate (match.py) is the slow, authoritative loop.

Run: python3.12 tools/9qum/monitors.py --engine models/engine/baseline --ms 100
"""
import argparse
import gzip
import json
import math
import os
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "playok"))
from engine import Engine  # noqa: E402


class Ev(Engine):
    def score_and_move(self, pos, ms):
        with self._lock:
            self._cmd(f"go time {ms} pos {pos}")
            while True:
                line = self._proc.stdout.readline()
                if not line:
                    return None, None
                line = line.strip()
                if line.startswith("bestmove"):
                    t = line.split()
                    sc = int(t[t.index("score") + 1]) if "score" in t else None
                    return sc, int(t[1])
                if line.startswith(("terminal", "error")):
                    return None, None


def pos_of(st):
    tz = st.get("tuzdyk") or [None, None]
    return (",".join(map(str, st["pits"][0:9])) + "/" + ",".join(map(str, st["pits"][9:18])) +
            f"/{st['kazan'][0]},{st['kazan'][1]}/"
            f"{-1 if tz[0] is None else tz[0] - 9},{-1 if tz[1] is None else tz[1]}/{st['to_move']}")


def load_val_positions(corpus, sample_per_bucket, seed=5):
    with open(os.path.join(corpus, "train", "split.json"), encoding="utf-8") as f:
        val_games = set(json.load(f)["val_games"])
    buckets = {"mid": [], "close": [], "clear": [], "policy": []}
    with gzip.open(os.path.join(corpus, "games", "replays.jsonl.gz"), "rt", encoding="utf-8") as f:
        for line in f:
            g = json.loads(line)
            if g["game_id"] not in val_games or g.get("winner") not in (0, 1):
                continue
            meta = g.get("_meta") or {}
            r0 = meta.get("r0_before") or 0
            r1 = meta.get("r1_before") or 0
            moves = (g.get("states") or [{}])[0].get("moves") or []
            for ply, st in enumerate(g["states"]):
                y = 1.0 if g["winner"] == st["to_move"] else 0.0
                dk = st["kazan"][0] - st["kazan"][1]
                if 40 <= ply < 80:
                    buckets["mid"].append((st, y))
                elif ply >= 80 and abs(dk) <= 8:
                    buckets["close"].append((st, y))
                elif ply >= 80 and abs(dk) >= 20:
                    buckets["clear"].append((st, y))
                if ply < len(moves) and min(r0, r1) >= 2000:
                    buckets["policy"].append((st, moves[ply]["hole"]))
    rng = random.Random(seed)
    for k in buckets:
        if len(buckets[k]) > sample_per_bucket:
            buckets[k] = rng.sample(buckets[k], sample_per_bucket)
    return buckets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=str(REPO / "models" / "engine" / "baseline"))
    ap.add_argument("--ms", type=int, default=100)
    ap.add_argument("--corpus", default="data/9qum")
    ap.add_argument("--sample", type=int, default=900)
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    buckets = load_val_positions(a.corpus, a.sample)
    e = Ev(Path(a.engine))
    e.start()
    out = {"engine": a.engine, "ms": a.ms, "label": a.label}
    for name in ("mid", "close", "clear"):
        pairs = []
        for st, y in buckets[name]:
            sc, _ = e.score_and_move(pos_of(st), a.ms)
            if sc is None:
                continue
            sc = max(-2000, min(2000, sc))          # mate/EGTB scores would swamp the scale
            pairs.append((1 / (1 + math.exp(-sc / 350)), y))
        acc = 100 * sum(1 for p, y in pairs if (p > 0.5) == (y > 0.5)) / len(pairs)
        brier = sum((p - y) ** 2 for p, y in pairs) / len(pairs)
        out[f"{name}_acc"] = round(acc, 1)
        out[f"{name}_brier"] = round(brier, 4)
        print(f"{name:>6}: n={len(pairs):>5}  acc {acc:5.1f}%  brier {brier:.4f}")
    hits = tot = 0
    for st, played in buckets["policy"]:
        _, mv = e.score_and_move(pos_of(st), a.ms)
        if mv is None:
            continue
        tot += 1
        hits += (mv == played)
    e.stop()
    out["policy_match"] = round(100 * hits / max(1, tot), 1)
    print(f"policy match-rate vs >=2000 humans: {out['policy_match']}% of {tot}")
    with open(os.path.join(a.corpus, "train", "monitors.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Record the A0 baseline for the current engine**

Run:
```bash
python3.12 tools/9qum/monitors.py --engine models/engine/baseline --ms 100 --label baseline
```
Expected: mid/close/clear accuracies within a couple of points of 78.3 / 77.0 / 76.1 and a
policy match-rate printed for the first time. If the accuracies come out wildly different,
stop and reconcile — the measured baseline in the spec is what every later comparison uses.

- [ ] **Step 3: Commit**

```bash
git add tools/9qum/monitors.py
git commit -m "tools: offline value/policy monitors with the A0 baseline"
```

---

### Task 6: Trainer, exporter, and torch/Rust equality

**Files:**
- Create: `research/training/train_nnue_v2.py`
- Create: `research/training/test_nnue_v2_export.py`

**Interfaces:**
- Consumes: `data/9qum/train/{train,val}.bin` (Task 4), `research/data/features_v2.py` (Task 2),
  the `NNU2` format and `evalpos` CLI (Task 3).
- Produces: `models/nets/nnue_v2/<name>.bin` (NNU2) and `<name>.pt`, plus
  `train_nnue_v2.export_nnu2(model, path)`.

- [ ] **Step 1: Write the failing equality test**

`research/training/test_nnue_v2_export.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3.12 research/training/test_nnue_v2_export.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'train_nnue_v2'`.

- [ ] **Step 3: Implement the trainer and exporter**

`research/training/train_nnue_v2.py`:

```python
#!/usr/bin/env python3
"""Train the NNUE v2 evaluation on the 9qum corpus and export it in the NNU2 format.

Input is sparse and binary (23 active features of 292), so the first layer is a sum of
selected columns — an EmbeddingBag with mode="sum". Four output heads are selected by phase
bucket, so endgame weights stop competing with opening weights (June's finding: pushing
endgame conversion cost general strength).

The net outputs a logit of "the side to move wins"; the engine multiplies it by 350 to get
centipawn-like units.

Run: python3.12 research/training/train_nnue_v2.py --epochs 12 --name v2_e12
"""
import argparse
import os
import struct
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
import features_v2 as fv

RECORD_SIZE = 68
MASK_VALUE_NET = 2
MASK_VALUE_OUTCOME = 4
NUM_FEATURES = fv.NUM_FEATURES
ACC = 1024
HIDDEN = 32
BUCKETS = fv.NUM_BUCKETS


class NnueV2(nn.Module):
    def __init__(self, num_features=NUM_FEATURES, acc=ACC, hidden=HIDDEN, buckets=BUCKETS):
        super().__init__()
        self.acc, self.hidden, self.buckets = acc, hidden, buckets
        self.emb = nn.EmbeddingBag(num_features, acc, mode="sum")
        self.acc_bias = nn.Parameter(torch.zeros(acc))
        self.fc2 = nn.ModuleList([nn.Linear(acc, hidden) for _ in range(buckets)])
        self.fc3 = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(buckets)])

    def forward(self, feats, offsets, bucket):
        a = torch.relu(self.emb(feats, offsets) + self.acc_bias)
        out = torch.zeros(a.shape[0], device=a.device)
        for b in range(self.buckets):
            m = bucket == b
            if m.any():
                out[m] = self.fc3[b](torch.relu(self.fc2[b](a[m]))).squeeze(-1)
        return out

    def forward_single(self, feature_indices, bucket):
        feats = torch.tensor(feature_indices, dtype=torch.long)
        offsets = torch.tensor([0], dtype=torch.long)
        return self.forward(feats, offsets, torch.tensor([bucket]))[0]


def export_nnu2(model, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<I", 0x324E554E))
        f.write(struct.pack("<6H", 2, NUM_FEATURES, model.acc, model.hidden, model.buckets, 0))
        w = model.emb.weight.detach().cpu().numpy().astype(np.float32)   # [features, acc]
        f.write(w.tobytes())                                              # feature-major
        f.write(model.acc_bias.detach().cpu().numpy().astype(np.float32).tobytes())
        for b in range(model.buckets):
            f.write(model.fc2[b].weight.detach().cpu().numpy().astype(np.float32).tobytes())
            f.write(model.fc2[b].bias.detach().cpu().numpy().astype(np.float32).tobytes())
            f.write(model.fc3[b].weight.detach().cpu().numpy().astype(np.float32).ravel().tobytes())
            f.write(model.fc3[b].bias.detach().cpu().numpy().astype(np.float32).tobytes())


def load_bin(path):
    raw = np.fromfile(path, dtype=np.uint8)
    n = len(raw) // RECORD_SIZE
    r = raw[: n * RECORD_SIZE].reshape(n, RECORD_SIZE)
    pits = r[:, 0:18].astype(np.int64)
    kazan = r[:, 18:20].astype(np.int64)
    tuz = r[:, 20:22].view(np.int8).astype(np.int64)
    stm = r[:, 22].astype(np.int64)
    value = r[:, 59:63].copy().view(np.float32).ravel()
    mask = r[:, 67]
    return pits, kazan, tuz, stm, value, mask


def build_feature_matrix(pits, kazan, tuz, stm):
    """Vectorised mirror of features_v2.build_features; test_features_v2.py owns correctness
    of the layout, this only has to agree with it."""
    n = pits.shape[0]
    feats = np.zeros((n, 23), dtype=np.int64)
    rows = np.stack([pits[:, 0:9], pits[:, 9:18]], axis=1)          # [n, 2, 9]
    me = stm
    opp = 1 - stm
    ar = np.arange(n)
    bucket_lut = np.array([min(c, 9) if c <= 9 else (10 if c <= 12 else (11 if c <= 16 else (12 if c <= 24 else 13)))
                           for c in range(163)], dtype=np.int64)
    for i in range(9):
        feats[:, i] = i * 14 + bucket_lut[rows[ar, me, i]]
        feats[:, 9 + i] = 126 + i * 14 + bucket_lut[rows[ar, opp, i]]
    feats[:, 18] = 252 + np.minimum(8, kazan[ar, me] // 10)
    feats[:, 19] = 261 + np.minimum(8, kazan[ar, opp] // 10)
    tz = np.where(tuz < 0, 9, tuz)
    feats[:, 20] = 270 + tz[ar, me]
    feats[:, 21] = 280 + tz[ar, opp]
    total = pits.sum(axis=1)
    feats[:, 22] = 290 + (total % 2)
    phase = np.where(total >= 121, 0, np.where(total >= 81, 1, np.where(total >= 41, 2, 3)))
    return feats, phase


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/9qum/train/train.bin")
    ap.add_argument("--val", default="data/9qum/train/val.bin")
    ap.add_argument("--out", default="models/nets/nnue_v2")
    ap.add_argument("--name", default="v2")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--w-net", type=float, default=1.0, help="weight of 9qum-labelled records")
    ap.add_argument("--w-outcome", type=float, default=0.3, help="weight of outcome-only records")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = NnueV2().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    lossf = nn.BCEWithLogitsLoss(reduction="none")

    def prep(path):
        pits, kazan, tuz, stm, value, mask = load_bin(path)
        feats, phase = build_feature_matrix(pits, kazan, tuz, stm)
        w = np.where(mask & MASK_VALUE_NET, a.w_net, a.w_outcome).astype(np.float32)
        return (torch.tensor(feats), torch.tensor(phase), torch.tensor(value),
                torch.tensor(w))

    tr = prep(a.train)
    va = prep(a.val)
    print(f"train {tr[0].shape[0]:,} records, val {va[0].shape[0]:,}")

    def run_epoch(data, train):
        feats, phase, value, w = data
        n = feats.shape[0]
        order = torch.randperm(n) if train else torch.arange(n)
        tot = cnt = 0.0
        model.train(train)
        for s in range(0, n, a.batch):
            idx = order[s: s + a.batch]
            fb = feats[idx].to(dev)
            offs = torch.arange(0, fb.shape[0] * 23, 23, device=dev)
            out = model(fb.reshape(-1), offs, phase[idx].to(dev))
            l = lossf(out, value[idx].to(dev)) * w[idx].to(dev)
            l = l.mean()
            if train:
                opt.zero_grad()
                l.backward()
                opt.step()
            tot += l.item() * idx.numel()
            cnt += idx.numel()
        return tot / cnt

    os.makedirs(a.out, exist_ok=True)
    best = float("inf")
    for ep in range(1, a.epochs + 1):
        trl = run_epoch(tr, True)
        with torch.no_grad():
            val = run_epoch(va, False)
        print(f"epoch {ep:>3}  train {trl:.4f}  val {val:.4f}")
        if val < best:
            best = val
            torch.save(model.state_dict(), os.path.join(a.out, f"{a.name}.pt"))
            export_nnu2(model, os.path.join(a.out, f"{a.name}.bin"))
            print(f"  saved {a.name}.bin (val {val:.4f})")
    print(f"best val loss {best:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the equality test**

Run:
```bash
python3.12 research/training/test_nnue_v2_export.py
```
Expected: `OK: torch and Rust agree on 4 positions (max diff ...e-0x)`.

If the diff is large only on the third and fourth positions, the bucket selection differs —
check `phase_bucket` boundaries on both sides. If every position differs, the fc1 matrix is
transposed: the format wants feature-major (`fc1_w[feature * acc + j]`), which is exactly
`emb.weight` in row-major order.

- [ ] **Step 5: Train a first net**

Run:
```bash
python3.12 research/training/train_nnue_v2.py --epochs 12 --name v2_e12 2>&1 | tail -20
```
Expected: val loss decreasing; `models/nets/nnue_v2/v2_e12.bin` written.

- [ ] **Step 6: Commit**

```bash
git add research/training/train_nnue_v2.py research/training/test_nnue_v2_export.py
git commit -m "train: NNUE v2 trainer (sparse first layer, phase heads) + NNU2 exporter"
```

---

### Task 7: A1 acceptance — monitors, NPS, and the gates

**Files:**
- Create: `models/nets/nnue_v2/RESULTS.md`
- No source changes; this task measures and records.

**Interfaces:**
- Consumes: `models/nets/nnue_v2/<name>.bin` (Task 6), `tools/9qum/monitors.py` (Task 5),
  `tools/ab_match.py`, `tools/9qum/match.py`.
- Produces: `models/nets/nnue_v2/RESULTS.md` with a row per candidate, and a decision:
  promote, iterate (Task 8), or record a negative result.

- [ ] **Step 1: Build a test engine that loads the v2 weights**

The engine loads `nnue_weights.bin` from beside the binary, so give the candidate its own
directory rather than touching `models/engine/`:

```bash
mkdir -p /tmp/eng_v2
cp target/release/togyzkumalaq-engine /tmp/eng_v2/
cp models/engine/egtb.bin models/engine/opening_book.txt /tmp/eng_v2/
cp models/nets/nnue_v2/v2_e12.bin /tmp/eng_v2/nnue_weights.bin
printf 'go time 200 pos 9,9,9,9,9,9,9,9,9/9,9,9,9,9,9,9,9,9/0,0/-1,-1/0\nquit\n' \
  | /tmp/eng_v2/togyzkumalaq-engine serve
```
Expected: the banner reports the NNUE loaded from `/tmp/eng_v2/nnue_weights.bin`, then a
`bestmove` line. If it reports the legacy dims (40 → 256), the magic check in Task 3 is not
firing.

- [ ] **Step 2: Run the monitors and compare against the targets**

```bash
python3.12 tools/9qum/monitors.py --engine /tmp/eng_v2/togyzkumalaq-engine --ms 100 --label v2_e12
```

Task 5 replaced the spec's whole-corpus figures with a like-for-like measurement: the monitor
now scores our engine and 9qum's stored labels over the **same** sampled val-split positions
in one pass, so these are the numbers that govern. Baseline and reference, 900 positions per
bucket, 100% label coverage:

| bucket | our baseline | 9qum reference | gap | A1 screen (half the gap) |
|---|---|---|---|---|
| midgame 40≤ply<80 | 81.5% | 86.7% | −5.2 | ≥84.1% |
| close endgame | 79.3% | 92.9% | −13.6 | ≥86.1% |
| clear endgame | 84.7% | 96.0% | −11.3 | ≥90.4% |
| Brier (mid/close/clear) | 0.184/0.173/0.136 | 0.095/0.054/0.026 | — | ≤0.140/0.114/0.081 |

Acceptance for A1 is the **A1 screen** column — at least half of each per-bucket gap closed.
Parity with the reference is the phase target, not the screen. Below the screen, go to Task 8
rather than to the gate: the gate costs hours, the monitors cost minutes. The screen is a
filter, never the verdict — only `ab_match` at equal time and the 9qum match gate decide
whether the engine actually got stronger.

- [ ] **Step 3: Measure the NPS cost honestly**

```bash
for e in models/engine/baseline /tmp/eng_v2/togyzkumalaq-engine; do
  echo "== $e"
  printf 'go time 3000 pos 9,9,9,9,9,9,9,9,9/9,9,9,9,9,9,9,9,9/0,0/-1,-1/0\nquit\n' \
    | $e serve | grep bestmove
done
```
Read `nps` from both lines. Acceptance: the v2 engine's NPS is no worse than half the
baseline's. A larger drop is allowed only if Step 4 still passes.

- [ ] **Step 4: Control gate at equal time**

```bash
python3.12 tools/ab_match.py /tmp/eng_v2/togyzkumalaq-engine models/engine/baseline 100 1000 --jobs 8
```
Acceptance: ≥55% for the v2 engine. This is the check that a better evaluator has not been
paid for with lost depth — equal time per move, not equal depth.

- [ ] **Step 5: Working gate against 9qum's net**

```bash
python3.12 tools/9qum/match.py --games 48 --parallel 1 --move-ms 1000 --rps 0.7 \
  --opening-plies 4 --engine /tmp/eng_v2/togyzkumalaq-engine
```
Baseline to beat: 31.2%. Promote at ≥55% over 96 games (run a second 48 with
`--opening-offset 12` and pool them).

- [ ] **Step 6: Record the results**

Write `models/nets/nnue_v2/RESULTS.md` with one row per candidate: name, training flags,
mid/close/clear accuracy, Brier, policy match-rate, NPS ratio, `ab_match` score, and the
9qum gate score with the number of games. State plainly which of the acceptance thresholds
were met and which were not — a candidate that improves the monitors but fails `ab_match` at
equal time is a negative result and must be recorded as one.

- [ ] **Step 7: Commit**

```bash
git add models/nets/nnue_v2/RESULTS.md
git commit -m "train: NNUE v2 first-candidate results (monitors, NPS, ab_match, 9qum gate)"
```

---

### Task 8: A2 — data-weighting iterations with keep-best

**Files:**
- Create: `research/training/sweep_nnue_v2.sh`
- Modify: `models/nets/nnue_v2/RESULTS.md` (append a row per candidate)

**Interfaces:**
- Consumes: everything from Tasks 4-7.
- Produces: the best candidate under `models/nets/nnue_v2/best.bin` plus the decision to move
  to phase B or to stop.

- [ ] **Step 1: Write the sweep script**

`research/training/sweep_nnue_v2.sh`:

```bash
#!/usr/bin/env bash
# A2: iterate on how the two label sources are weighted and on capacity, judging each
# candidate by the offline monitors only. Only the best one goes to the expensive gates.
set -euo pipefail
cd "$(dirname "$0")/../.."

for wo in 0.1 0.3 1.0; do
  for ep in 12 25; do
    name="v2_wo${wo}_e${ep}"
    echo "=== $name"
    python3.12 research/training/train_nnue_v2.py \
      --epochs "$ep" --w-net 1.0 --w-outcome "$wo" --name "$name" 2>&1 | tail -3
    mkdir -p /tmp/eng_v2
    cp target/release/togyzkumalaq-engine /tmp/eng_v2/
    cp models/engine/egtb.bin models/engine/opening_book.txt /tmp/eng_v2/
    cp "models/nets/nnue_v2/${name}.bin" /tmp/eng_v2/nnue_weights.bin
    python3.12 tools/9qum/monitors.py \
      --engine /tmp/eng_v2/togyzkumalaq-engine --ms 100 --label "$name"
  done
done
echo "monitor rows are appended to data/9qum/train/monitors.jsonl"
```

- [ ] **Step 2: Run the sweep**

```bash
chmod +x research/training/sweep_nnue_v2.sh && ./research/training/sweep_nnue_v2.sh
```
Expected: six candidates, each with a monitor row.

- [ ] **Step 3: Pick the best candidate by the monitors**

```bash
python3.12 - <<'PY'
import json
rows = [json.loads(l) for l in open("data/9qum/train/monitors.jsonl", encoding="utf-8")]
rows.sort(key=lambda r: (r.get("close_brier", 9) + r.get("clear_brier", 9)))
for r in rows[:8]:
    print(f"{r['label']:<20} mid {r.get('mid_acc')}%  close {r.get('close_acc')}%  "
          f"clear {r.get('clear_acc')}%  brier {r.get('close_brier')}/{r.get('clear_brier')}  "
          f"policy {r.get('policy_match')}%")
PY
```

- [ ] **Step 4: Gate the winner over 96 games**

Pick the winner programmatically so the choice is reproducible, then gate it:

```bash
python3.12 -c "
import json, shutil
rows = [json.loads(l) for l in open('data/9qum/train/monitors.jsonl', encoding='utf-8')]
rows = [r for r in rows if r.get('label', '').startswith('v2_')]
best = min(rows, key=lambda r: r.get('close_brier', 9) + r.get('clear_brier', 9))
shutil.copy(f\"models/nets/nnue_v2/{best['label']}.bin\", 'models/nets/nnue_v2/best.bin')
print('best candidate:', best['label'])
"
cp models/nets/nnue_v2/best.bin /tmp/eng_v2/nnue_weights.bin
python3.12 tools/ab_match.py /tmp/eng_v2/togyzkumalaq-engine models/engine/baseline 100 1000 --jobs 8
python3.12 tools/9qum/match.py --games 48 --move-ms 1000 --rps 0.7 --opening-plies 4 \
  --engine /tmp/eng_v2/togyzkumalaq-engine
python3.12 tools/9qum/match.py --games 48 --move-ms 1000 --rps 0.7 --opening-plies 4 \
  --opening-offset 12 --engine /tmp/eng_v2/togyzkumalaq-engine
```

- [ ] **Step 5: Apply the stopping rule and record the outcome**

Append to `models/nets/nnue_v2/RESULTS.md`: every candidate's monitors, the winner's gate
score over 96 games, and one of three verdicts:
- **≥55%** — phase A met its target; promote (a separate decision by the user, since
  `models/engine/baseline` is production) and start phase B for the ceiling.
- **moved by ≥8% but below 55%** — continue A with the next lever (capacity: `--acc 2048`,
  or more phase buckets).
- **moved by <8%** — record A as a negative result with the numbers, and move to phase B.

- [ ] **Step 6: Commit**

```bash
git add research/training/sweep_nnue_v2.sh models/nets/nnue_v2/RESULTS.md
git commit -m "train: A2 label-weight sweep and phase-A verdict"
```

---

## Self-review notes

- **Spec coverage.** Gates and monitors → Tasks 5, 7, 8. Sparse bucketed input → Tasks 1, 2.
  1024 accumulator and 4 phase heads → Tasks 3, 6. Training on their labels plus human
  outcomes with by-game splits → Task 4. NPS requirement verified at equal time → Task 7
  Steps 3-4. Stopping rule → Task 8 Step 5. Score head is stored by the converter (Task 4)
  but deliberately unused until phase B, which has its own plan.
- **Not covered here, by design:** the incremental accumulator. The v2 eval recomputes the
  accumulator from 23 active features per call; if Task 7 Step 3 shows the NPS drop is worse
  than 2×, the fix is an incremental accumulator threaded through make/unmake, and that gets
  planned then rather than speculatively now.
- **Type consistency.** `build_features_v2`/`build_features` return feature indices only,
  never counts; `logit_v2` returns the raw logit and `evaluate` returns cp×64; `RECORD_SIZE`
  is 68 in the converter, the trainer and the tests.

---

## Redirect after the A1 measurement (2026-07-31)

Task 7 measured a NEGATIVE result for `v2_e12` and, importantly, named the causes. Two
independent 100-game runs at equal time gave 5.0% and 11.5% (Elo −512 and −354), and the NPS
ratio replicated at 0.23× / 0.243× against a ≥0.5× requirement. The close-endgame value did
improve as designed (79.3% → 85.2%, Brier 0.173 → 0.116), so the encoding hypothesis holds;
what failed is the engineering around it.

Task 8's label-weight sweep is therefore **deferred**: while the engine is 4× slower and its
eval is on a different numeric scale, the monitors and the gate measure different things and
any weighting would be fitted to an artefact. It returns once speed and scale reach parity.

### Task 9: restore eval speed (i16 quantisation + incremental accumulator)

**Files:** Modify `engine/src/nnue.rs`, `research/training/train_nnue_v2.py` (exporter), and
add tests to the existing `tests_v2` module.

**Interfaces:** `NNU2` gains a version 3 that stores i16 weights with documented scale factors;
`load_v2` keeps reading version 2 f32 files so `v2_e12.bin` stays loadable for comparison.

Acceptance: NPS ≥ 0.5× the baseline on the non-book position
`1,12,12,12,12,3,1,13,12/12,0,11,11,11,1,9,1,2/22,4/-1,-1/1` at `go time 3000`, AND the
quantised net's logit stays within 0.02 of the f32 net's on 200 sampled positions (a test), so
the speedup does not silently change what the net says.

### Task 10: calibrate the output scale to the search's expectations

**Files:** Modify `engine/src/nnue.rs` (the logit → cp mapping only).

The v2 mapping `cp = 350 × logit` saturates toward ±3000 while the legacy eval's normal range
is in the tens: on `0,1,1,1,2,3,3,1,4/1,2,0,0,5,5,3,1,2/40,30/-1,-1/0` the baseline scores +94
and v2 scores −589. The search's static-eval pruning margins were tuned for the legacy scale.

Fit an affine map from the v2 logit to the legacy eval's units on a sample of positions
(regress the baseline engine's `score` against `logit` over a few thousand val-split positions,
excluding EGTB/mate-range scores), replace the hardcoded 350 with the fitted coefficients, and
record them in the report. Acceptance: on the same sample, the v2 engine's score distribution
has a mean and standard deviation within 25% of the baseline's, and `ab_match` at equal time
improves materially over the 5–11.5% baseline measured for `v2_e12`.

### Task 11: re-measure A1

Re-run the full acceptance set (monitors at the full 900-position sample, NPS, `ab_match` 100
games at 1000 ms) and record a new row in `RESULTS.md`. Only if `ab_match` clears 55% does the
9qum live gate become worth its ~1.5 hours.

**Long measurements run from the controller, not from a subagent:** a subagent's Bash
backgrounds anything past its tool timeout, and its stdout dies with the turn — this already
cost one full 50-minute `ab_match` run. Subagents implement and test; the controller runs the
long measurements and hands back the numbers in a file.
