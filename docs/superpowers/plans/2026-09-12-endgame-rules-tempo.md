# Endgame: terminal rule, clock use, tempo-aware eval — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the engine play the real end-of-game rule, spend its clock where games are
decided, and evaluate the tempo endgame (locked hoards + waiting moves) — gated by paired
external matches, then promoted to production.

**Architecture:** Three independent levers in dependency order. (1) `game_result()` in the
Rust core and `_check_winner()` in the Python rules class become "terminal iff the side to
move has no stones"; EGTB is regenerated under the fixed rule; the bridge's sweep fallback
becomes an alarm. (2) A pure `move_budget_ms()` helper in `tools/playok/engine.py` gives the
PlayOK bridge and the 9qum match harness a bigger budget at ≤40 board stones, capped by the
clock. (3) Two pure functions in `engine/src/eval.rs` (`tempo_reserve`, `locked_stones`)
feed a side-to-move correction added to the live NNUE eval path in `search.rs`. Every
strength claim comes from `tools/9qum/match.py` runs inside one window, per
`docs/MEASUREMENT_PROTOCOL.md`.

**Tech Stack:** Rust (`cargo 1.93`, workspace at repo root), Python 3.12 (`python3.12` only —
plain `python3` has no numpy/torch), numpy. No pytest: tests are assert scripts with an
`if __name__ == "__main__"` runner (pattern: `research/training/test_sweep_rule.py`).

**Spec:** `docs/superpowers/specs/2026-09-12-endgame-rules-tempo-design.md`

## Global Constraints

- Work on branch `endgame-rules`, created from the current `nnue-v2` HEAD (Task 0).
- Run every Python command with `python3.12` from the repo root.
- Build with `cargo build --release` from the repo root; binary at
  `target/release/togyzkumalaq-engine`. Engine assets (`nnue_weights.bin`, `egtb.bin`,
  `opening_book.txt`) are resolved **next to the binary**; a test engine is assembled with
  `tools/9qum/make_test_engine.sh <weights> <dest>` (never under `/tmp`, never into
  `models/engine/`).
- `models/engine/baseline` (production binary) is overwritten only in Task 10, after the
  gate AND explicit user approval. `models/engine/egtb.bin` is gitignored and is
  regenerated in Task 3 deliberately (the old table is provably wrong and fires in ~0% of
  games).
- Measurement rules (`docs/MEASUREMENT_PROTOCOL.md`): compare only arms run back to back in
  the same window with `tools/9qum/run_measurements.py`; whole opening suite; 24 games is
  "a direction", 96 is a promotion; record the opponent's `level_mix`/`level_drop`. In-lineage
  `tools/ab_match.py` results are regression guards only, never evidence of improvement.
- Commit after every task with the attribution line
  `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`. Do not batch commits.

---

### Task 0: Branch

**Files:** none

- [ ] **Step 1: Create the branch from the current HEAD**

```bash
git checkout -b endgame-rules
git log --oneline -1
```
Expected: `eda9d7a docs: phase-A verdict — ...` (or later) as the branch base.

---

### Task 1: Terminal rule in the Rust core

**Files:**
- Modify: `core/src/board.rs:237-266` (`game_result`) and its doc comment
- Test: `core/src/board.rs` `mod tests`

**Interfaces:**
- Produces: `Board::game_result(&self) -> Option<GameResult>` — unchanged signature, new
  semantics: `Some(..)` iff a kazan ≥ 82 **or the side to move has no stones**.
  `is_terminal()`, `stones_on_side()`, `total_board_stones()` unchanged.

- [ ] **Step 1: Write the failing tests** (append inside `mod tests` in `core/src/board.rs`)

```rust
    #[test]
    fn test_empty_side_not_terminal_when_opponent_to_move() {
        // White emptied itself; Black still must move. NOT over yet.
        let mut b = Board::new();
        b.pits[0] = [0; 9];
        b.pits[1] = [0, 0, 0, 0, 0, 0, 0, 0, 3];
        b.kazan = [80, 79];
        b.side_to_move = Side::Black;
        assert_eq!(b.game_result(), None);
    }

    #[test]
    fn test_empty_side_terminal_when_it_is_to_move() {
        // Same board, but it is White's turn and White has nothing: game over, sweep.
        let mut b = Board::new();
        b.pits[0] = [0; 9];
        b.pits[1] = [0, 0, 0, 0, 0, 0, 0, 0, 3];
        b.kazan = [80, 79];
        b.side_to_move = Side::White;
        assert_eq!(b.game_result(), Some(GameResult::Win(Side::Black))); // 80 vs 79+3
    }

    #[test]
    fn test_forced_feed_continues_game() {
        // Black's only move (pit 9, 3 stones) drops 2 stones onto White's side:
        // White is back in the game.
        let mut b = Board::new();
        b.pits[0] = [0; 9];
        b.pits[1] = [0, 0, 0, 0, 0, 0, 0, 0, 3];
        b.kazan = [80, 79];
        b.side_to_move = Side::Black;
        b.make_move(8);
        assert_eq!(b.pits[0], [1, 1, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(b.side_to_move, Side::White);
        assert_eq!(b.game_result(), None);
        let mut moves = [0usize; NUM_PITS];
        assert_eq!(b.valid_moves_array(&mut moves), 2);
    }

    #[test]
    fn test_lone_stone_from_pit9_then_safe_reply_ends_game() {
        // White plays its last stone from pit 9 -> White empty, Black to move (not over).
        // Black answers with a lone stone that stays on its side -> White to move with
        // nothing -> over; Black sweeps its 6 board stones.
        let mut b = Board::new();
        b.pits[0] = [0, 0, 0, 0, 0, 0, 0, 0, 1];
        b.pits[1] = [0, 0, 0, 0, 0, 0, 0, 0, 5];
        b.kazan = [78, 78];
        b.side_to_move = Side::White;
        b.make_move(8); // lone stone -> Black pit 1
        assert_eq!(b.pits[1], [1, 0, 0, 0, 0, 0, 0, 0, 5]);
        assert_eq!(b.game_result(), None, "self-emptying must not end the game");
        b.make_move(0); // Black: lone stone pit 1 -> pit 2, White still empty
        assert_eq!(b.game_result(), Some(GameResult::Win(Side::Black))); // 78 vs 78+6
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p togyzkumalaq-core empty_side -- --nocapture && cargo test -p togyzkumalaq-core forced_feed && cargo test -p togyzkumalaq-core lone_stone`
Expected: `test_empty_side_not_terminal_when_opponent_to_move`, `test_forced_feed_continues_game`
and `test_lone_stone_from_pit9_then_safe_reply_ends_game` FAIL (they get `Some(..)` where
`None` is expected). `test_empty_side_terminal_when_it_is_to_move` passes already.

- [ ] **Step 3: Implement the rule** (replace the body of `game_result` at `core/src/board.rs:237-266`)

```rust
    /// Check game result. Returns Some(winner) / Some(Draw), or None if the game goes on.
    ///
    /// Terminal when (a) a kazan reaches WIN_THRESHOLD, or (b) the SIDE TO MOVE has no
    /// stones. Rule (b) is deliberately one-sided: a player who empties its own side (a
    /// lone stone from pit 9, or a lone stone into the opponent's tuzdyk) has NOT ended the
    /// game — the opponent must move, and if that move drops stones back, play continues.
    /// Verified 2026-09-12 against PlayOK (110 self-empties, play always continued) and
    /// 9qum replays (1120 such states, 0 contradictions). At a terminal each side sweeps
    /// its own remaining board stones into its kazan before comparing.
    pub fn game_result(&self) -> Option<GameResult> {
        if self.kazan[0] >= WIN_THRESHOLD {
            return Some(GameResult::Win(Side::White));
        }
        if self.kazan[1] >= WIN_THRESHOLD {
            return Some(GameResult::Win(Side::Black));
        }

        let stm = self.side_to_move.index();
        if self.pits[stm].iter().all(|&x| x == 0) {
            let kw = self.kazan[0] as u16 + self.stones_on_side(Side::White);
            let kb = self.kazan[1] as u16 + self.stones_on_side(Side::Black);
            return Some(match kw.cmp(&kb) {
                std::cmp::Ordering::Greater => GameResult::Win(Side::White),
                std::cmp::Ordering::Less => GameResult::Win(Side::Black),
                std::cmp::Ordering::Equal => GameResult::Draw,
            });
        }

        None
    }
```

Also change the header comment at `core/src/board.rs:10` from
`/// Win condition: first to capture 82+ stones in kazan.` to
`/// Win: 82+ in kazan, or the side to move has no stones (then each side sweeps its own board stones).`

- [ ] **Step 4: Run the whole workspace test suite**

Run: `cargo test -p togyzkumalaq-core && cargo test -p togyzkumalaq-engine && cargo build --release`
Expected: all core tests pass (the two existing sweep tests still pass: they set White to
move with an empty White row). Engine tests pass. Release build succeeds (mcts links ORT
at runtime only; `cargo build --release` must still compile it).

- [ ] **Step 5: Commit**

```bash
git add core/src/board.rs
git commit -m "core: game ends only when the side to move has no stones (real rule)

PlayOK (110 self-empties) and 9qum (1120 states, 0 contradictions) both continue
play after a player empties its own side; the opponent must move and may be forced
to feed. Search, EGTB, MCTS labels and datagen all inherit the fix via game_result().

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Same rule in the Python rules class

**Files:**
- Modify: `archive/old-impls/alphazero-code/alphazero/game.py:197-213` (`_check_winner`)
- Test: `research/training/test_sweep_rule.py`

**Interfaces:**
- Consumes: `TogyzQumalaq`, `GameState`, `Player` from `game.py`; the `_state(white, black, kw, kb, stm)` helper already in the test file.
- Produces: `TogyzQumalaq._check_winner()` returns `None` when a side is empty but the
  other side is to move. Consumers: `tools/ab_match.py`, `research/training/train_loop.py`,
  `train_hybrid*.py`, `train_alphazero.py`, `tools/policy_match.py`, `tools/endgame_diagnosis.py`.

- [ ] **Step 1: Write the failing tests** (append to `research/training/test_sweep_rule.py` before the `if __name__ == "__main__":` block, and register them in that runner the same way the three existing tests are)

```python
def test_empty_side_not_terminal_when_opponent_to_move():
    g = TogyzQumalaq()
    g.state = _state([0] * 9, [0, 0, 0, 0, 0, 0, 0, 0, 3], 80, 79, stm=1)
    assert not g.is_terminal(), "White emptied itself; Black must still move"


def test_empty_side_terminal_when_it_is_to_move():
    g = TogyzQumalaq()
    g.state = _state([0] * 9, [0, 0, 0, 0, 0, 0, 0, 0, 3], 80, 79, stm=0)
    assert g.is_terminal()
    assert g.get_winner() == Player.BLACK  # 80 vs 79+3


def test_forced_feed_continues_game():
    g = TogyzQumalaq()
    g.state = _state([0] * 9, [0, 0, 0, 0, 0, 0, 0, 0, 3], 80, 79, stm=1)
    ok, winner = g.make_move(8)  # 1 stays in pit 9, 2 land on White pits 1-2
    assert ok and winner is None
    assert int(g.state.pits[0].sum()) == 2
    assert not g.is_terminal()
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3.12 research/training/test_sweep_rule.py`
Expected: the first and third new tests raise `AssertionError`; the second passes.

- [ ] **Step 3: Implement** (replace lines 197-213 of `game.py`, the `white_empty or black_empty` block)

```python
        # Only the side TO MOVE having no stones ends the game (verified 2026-09-12 on
        # PlayOK and 9qum). A player who empties itself has not ended the game: the
        # opponent must move and may be forced to feed stones back. At the terminal each
        # side sweeps its own remaining board stones into its kazan before comparing.
        stm = self.state.current_player
        if np.all(self.state.pits[stm] == 0):
            kw = int(self.state.kazan[Player.WHITE]) + int(self.state.pits[Player.WHITE].sum())
            kb = int(self.state.kazan[Player.BLACK]) + int(self.state.pits[Player.BLACK].sum())
            if kw > kb:
                return Player.WHITE
            elif kb > kw:
                return Player.BLACK
            else:
                return 2  # Draw
```

- [ ] **Step 4: Run to verify they pass, plus the existing three**

Run: `python3.12 research/training/test_sweep_rule.py`
Expected: all six tests print OK / no assertion.

- [ ] **Step 5: Commit**

```bash
git add archive/old-impls/alphazero-code/alphazero/game.py research/training/test_sweep_rule.py
git commit -m "rules(py): terminal only when the side to move has no stones

Same fix as the Rust core; this class referees ab_match and every trainer.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Regenerate the EGTB under the fixed rule

**Files:**
- Regenerate: `models/engine/egtb.bin` (gitignored)
- No code changes: `engine/src/egtb.rs` calls `game_result()`.

**Interfaces:**
- Consumes: `togyzkumalaq-engine egtb-gen <max_stones> <output>` and `egtb-verify <n>`
  (`engine/src/main.rs:197-209`). `egtb-verify` loads `egtb.bin` from next to the binary.

- [ ] **Step 1: Record the old table's identity**

```bash
cp models/engine/egtb.bin runs/egtb_old_rule.bin 2>/dev/null || (mkdir -p runs && cp models/engine/egtb.bin runs/egtb_old_rule.bin)
xxd -l 16 runs/egtb_old_rule.bin
```
Expected: `TKEGTB01`, max_stones `04`, then num_entries `38b7 3c00` (= 3,979,064).

- [ ] **Step 2: Generate**

```bash
cargo build --release
./target/release/togyzkumalaq-engine egtb-gen 4 models/engine/egtb.bin 2>&1 | tail -5
xxd -l 16 models/engine/egtb.bin
```
Expected: "Done! File size: ..." and a header with max_stones `04`. The entry count will
differ from 3,979,064: positions with an empty row and the *other* side to move are now
non-terminal and get solved entries instead of being skipped.

- [ ] **Step 3: Verify consistency**

```bash
cp models/engine/egtb.bin target/release/egtb.bin
./target/release/togyzkumalaq-engine egtb-verify 20000 2>&1 | tail -3
rm target/release/egtb.bin
```
Expected: the verifier reports 0 errors.

- [ ] **Step 4: Spot-check one rule-sensitive position with the engine**

```bash
./target/release/togyzkumalaq-engine analyze "0,0,0,0,0,0,0,0,0/0,0,0,0,0,0,0,0,3/80,79/-1,-1/1" 200
```
Expected: JSON with `"terminal":false` absent, i.e. a `bestmove` line (Black must move), not
`{"terminal":true,...}`.

- [ ] **Step 5: Commit** (nothing tracked changed; record the regeneration in the log via an empty commit)

```bash
git commit --allow-empty -m "egtb: regenerated 4-stone table under the fixed terminal rule

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Time budget helper + bridge + match harness

**Files:**
- Modify: `tools/playok/engine.py` (add `ENDGAME_STONES`, `move_budget_ms`)
- Create: `tools/playok/test_engine.py`
- Modify: `tools/playok/bridge.py:70-86` (ctor), `:289` (game over), `:568-592` (`_compute_and_send`), `:694` (argparse), `:723-732` (Bridge call)
- Modify: `tools/9qum/match.py:191` (`play_game` signature), `:245`, `:266`, `:286`, `:336`, `:384`

**Interfaces:**
- Produces: `move_budget_ms(pos: str, base_ms: int, endgame_ms: int, *, threshold: int = ENDGAME_STONES, clock_left_ms: int | None = None, reserve_ms: int = 120_000) -> int`
  in `tools/playok/engine.py`. `ENDGAME_STONES = 40`.
- Bridge flag `--endgame-move-time-ms` (default 12000). Match flag `--endgame-move-ms`
  (default `None` = same as `--move-ms`, so existing measurements are unchanged).

- [ ] **Step 1: Write the failing tests** — create `tools/playok/test_engine.py`

```python
#!/usr/bin/env python3
"""Tests for the pure helpers in tools/playok/engine.py.

Run: python3.12 tools/playok/test_engine.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import START_POSITION, move_budget_ms  # noqa: E402

FORTY = "3,3,3,3,3,3,2,0,0/3,3,3,3,3,3,2,0,0/61,61/-1,-1/0"       # 20 + 20 = 40 stones
FORTY_ONE = "3,3,3,3,3,3,3,0,0/3,3,3,3,3,3,2,0,0/61,60/-1,-1/0"   # 21 + 20 = 41 stones


def test_start_position_uses_base():
    assert move_budget_ms(START_POSITION, 1800, 12000) == 1800


def test_forty_stones_uses_endgame_budget():
    assert move_budget_ms(FORTY, 1800, 12000) == 12000


def test_forty_one_stones_uses_base():
    assert move_budget_ms(FORTY_ONE, 1800, 12000) == 1800


def test_no_clock_means_no_cap():
    assert move_budget_ms(FORTY, 1800, 12000, clock_left_ms=None) == 12000


def test_clock_caps_budget_to_keep_reserve():
    # 130 s left, 120 s reserve -> at most 10 s this move.
    assert move_budget_ms(FORTY, 1800, 12000, clock_left_ms=130_000) == 10_000


def test_clock_nearly_out_never_below_half_clock_or_100ms():
    assert move_budget_ms(FORTY, 1800, 12000, clock_left_ms=1_000) == 500
    assert move_budget_ms(FORTY, 1800, 12000, clock_left_ms=100) == 100


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("OK", name)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3.12 tools/playok/test_engine.py`
Expected: `ImportError: cannot import name 'move_budget_ms'`.

- [ ] **Step 3: Implement the helper** (add to `tools/playok/engine.py` right after `NUM_PITS = 9`)

```python
# Board-stone count at or below which a move gets the endgame budget. Measured on 244
# PlayOK games: losses end with a median of 30 stones on the board after 40-90 plies of
# lone-stone tempo play; branching there is 2-4, so extra time buys real depth.
ENDGAME_STONES = 40


def move_budget_ms(pos: str, base_ms: int, endgame_ms: int, *, threshold: int = ENDGAME_STONES,
                   clock_left_ms: int | None = None, reserve_ms: int = 120_000) -> int:
    """Thinking budget for `pos`: `endgame_ms` once <= `threshold` stones remain on the
    board, else `base_ms`. When the remaining clock is known, never spend more than what
    keeps `reserve_ms` on the clock, and when even that is gone fall back to half the
    clock (never below 100 ms so the engine still returns a move)."""
    white, black, _, _, _ = _parse_pos(pos)
    budget = endgame_ms if sum(white) + sum(black) <= threshold else base_ms
    if clock_left_ms is not None:
        cap = max(clock_left_ms - reserve_ms, min(base_ms, clock_left_ms // 2))
        budget = min(budget, cap)
    return max(budget, 100)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3.12 tools/playok/test_engine.py`
Expected: six `OK test_...` lines.

- [ ] **Step 5: Wire the bridge**

In `tools/playok/bridge.py`:

(a) import: after `from engine import Engine` (find with `grep -n "^from engine import" tools/playok/bridge.py`) make it
`from engine import Engine, move_budget_ms`.

(b) ctor signature (line 70): add `endgame_move_time_ms=12000,` right after `move_time_ms=1800,`.
After `self.move_time_ms = move_time_ms` (line 86) add:

```python
        self.endgame_move_time_ms = endgame_move_time_ms
        self.spent_ms = 0            # our thinking time used in the current game
        self.rule_mismatches = 0     # engine said terminal but the server wanted a move
```

(c) game over (line 289, right after `self.games_played += 1`): add `self.spent_ms = 0`.

(d) `_compute_and_send` (lines 568-592): replace the body of the `try:` block with

```python
            clock_left = None if self.time_min == 0 else self.time_min * 60_000 - self.spent_ms
            budget = move_budget_ms(pos, self.move_time_ms, self.endgame_move_time_ms,
                                    clock_left_ms=clock_left)
            pick = self.engine.bestmove(pos, budget)
            self.spent_ms += budget
            if isinstance(pick, tuple):
                # With the real terminal rule (core fix 2026-09-12) the engine and the
                # server must agree. If they ever disagree again this is a RULES BUG, not
                # a sweep quirk: play the lowest legal hole so we don't lose on time, and
                # count it so the mismatch is visible in the log.
                self.rule_mismatches += 1
                print(f"[engine] *** RULE MISMATCH #{self.rule_mismatches}: engine says terminal "
                      f"({pick[1]}) but server expects a move; pos={pos}", flush=True)
                if mask is not None and mask != 0:
                    hole = (mask & -mask).bit_length() - 1  # lowest set bit
                    if not self.dry_run:
                        self.client.move(self.table_k, hole, think_ds=1)
                        print(f"[bridge] SENT fallback hole {hole}", flush=True)
                return
            hole = self.game.engine_to_playok(pick)
            legal_ok = (mask is None) or bool(mask & (1 << hole))
            print(f"[engine] pos={pos} budget={budget}ms -> pit {pick} -> hole {hole} (mask ok={legal_ok})", flush=True)
            if self.dry_run:
                print(f"[DRY-RUN] would send [92,{self.table_k},1,{hole},..]", flush=True)
                return
            self.client.move(self.table_k, hole, think_ds=max(1, budget // 100))
            print(f"[bridge] SENT move hole {hole} (board advances on server echo)", flush=True)
```

(e) argparse (after line 694 `--move-time-ms`): add

```python
    ap.add_argument("--endgame-move-time-ms", type=int, default=12000,
                    help="thinking time once <=40 stones remain on the board (default: 12000)")
```

(f) `Bridge(...)` call (line 723): add `endgame_move_time_ms=args.endgame_move_time_ms,` after `move_time_ms=args.move_time_ms,`.

- [ ] **Step 6: Wire the match harness**

In `tools/9qum/match.py`:

(a) line 31: `from engine import Engine, move_budget_ms  # noqa: E402`.

(b) `play_game` signature (line 191): append parameter `endgame_ms=None`.

(c) line 245: replace `mv = eng.bestmove(pos_from_state(st), time_ms=move_ms)` with

```python
                pos = pos_from_state(st)
                budget = move_budget_ms(pos, move_ms, move_ms if endgame_ms is None else endgame_ms)
                mv = eng.bestmove(pos, time_ms=budget)
```

(d) lines 266 and 286 (the two dicts carrying `"move_ms": move_ms`): add `"endgame_ms": endgame_ms,` next to it.

(e) argparse after line 336: `ap.add_argument("--endgame-move-ms", type=int, default=None, help="thinking time at <=40 board stones (default: same as --move-ms)")`.

(f) line 384: pass `endgame_ms=a.endgame_move_ms` to `play_game`.

- [ ] **Step 7: Run the harness tests and a dry-run of the bridge**

```bash
python3.12 tools/9qum/test_match.py
python3.12 tools/playok/test_engine.py
python3.12 tools/playok/bridge.py --help | grep -A1 endgame
```
Expected: existing match tests pass; helper tests pass; the new flag is listed.

- [ ] **Step 8: Commit**

```bash
git add tools/playok/engine.py tools/playok/test_engine.py tools/playok/bridge.py tools/9qum/match.py
git commit -m "tools: endgame time budget (12 s at <=40 board stones) for the PlayOK bridge and 9qum matches

The bridge used 1.8 s per move of a 30-minute clock. Budget is capped by the clock and
the sweep fallback is now a rule-mismatch alarm.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Paired external gate A — rule fix + clock vs production

**Files:**
- Create: `runs/2026-09-12-gateA/jobs.json`
- Uses: `tools/9qum/make_test_engine.sh`, `tools/9qum/run_measurements.py`, `tools/9qum/lead_profile.py`

- [ ] **Step 1: Assemble the candidate engine from the new binary with production weights**

```bash
cargo build --release
tools/9qum/make_test_engine.sh models/engine/nnue_weights.bin models/nets/eng_rulefix --force
```
Expected: "verified ... loads weights from models/nets/eng_rulefix", sha256 printed.

- [ ] **Step 2: Write the job list** — `runs/2026-09-12-gateA/jobs.json`

```json
[
  {"name": "prod_1800",
   "cmd": ["python3.12", "tools/9qum/match.py", "--games", "48", "--level", "i",
           "--engine", "models/engine/baseline", "--move-ms", "1800",
           "--out", "runs/2026-09-12-gateA/prod_1800"]},
  {"name": "rulefix_1800_eg12000",
   "cmd": ["python3.12", "tools/9qum/match.py", "--games", "48", "--level", "i",
           "--engine", "models/nets/eng_rulefix/togyzkumalaq-engine", "--move-ms", "1800",
           "--endgame-move-ms", "12000",
           "--out", "runs/2026-09-12-gateA/rulefix_1800_eg12000"]}
]
```

- [ ] **Step 3: Run sequentially in one process**

```bash
python3.12 tools/9qum/run_measurements.py --jobs runs/2026-09-12-gateA/jobs.json --run-dir runs/2026-09-12-gateA
```
Expected: a manifest with two exit-code-0 jobs. Runtime: several hours (the second arm
thinks up to 12 s per endgame move).

- [ ] **Step 4: Read the numbers and the loss structure**

```bash
python3.12 tools/9qum/lead_profile.py runs/2026-09-12-gateA/prod_1800 runs/2026-09-12-gateA/rulefix_1800_eg12000
grep -h '"level_mix"' runs/2026-09-12-gateA/*/*.jsonl | sort | uniq -c
```
Record in `runs/2026-09-12-gateA/RESULT.md`: score % per arm, the opponent config
(`level_mix`/`level_drop` must match across arms or the comparison is void), and the median
lead at 25/50/75/final. Decision: a difference ≥ 12 points at 48 games is a direction; the
rule fix and clock stay regardless (they are correctness/clock, not tuning).

- [ ] **Step 5: Commit the result note**

```bash
git add -f runs/2026-09-12-gateA/RESULT.md runs/2026-09-12-gateA/jobs.json
git commit -m "measure: gate A (rule fix + endgame clock) vs production, paired on 9qum

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```
(`runs/` is gitignored, hence `-f` for these two small text files.)

---

### Task 6: Tempo and locked-stone functions

**Files:**
- Modify: `engine/src/eval.rs` (add three `pub fn` + constants before `#[cfg(test)]`)
- Test: `engine/src/eval.rs` `mod tests`

**Interfaces:**
- Produces:
  - `pub fn tempo_reserve(row: &[u8; NUM_PITS]) -> i32`
  - `pub fn locked_stones(row: &[u8; NUM_PITS]) -> i32`
  - `pub fn endgame_tempo_correction(board: &Board) -> i32` (side-to-move POV, 0 when
    `total_board_stones() > TEMPO_PHASE_STONES`)
  - `pub const TEMPO_LOCK_WEIGHT: i32 = 12; pub const TEMPO_WEIGHT: i32 = 4; pub const TEMPO_DIFF_CLAMP: i32 = 30; pub const TEMPO_PHASE_STONES: u16 = 60;`

- [ ] **Step 1: Write the failing tests** (append inside `mod tests` in `engine/src/eval.rs`)

```rust
    #[test]
    fn tempo_lone_stone_counts_steps_to_pit9() {
        let mut row = [0u8; NUM_PITS];
        row[0] = 1;
        assert_eq!(tempo_reserve(&row), 8);
        let mut row = [0u8; NUM_PITS];
        row[8] = 1;
        assert_eq!(tempo_reserve(&row), 0, "a lone stone in pit 9 must cross next move");
    }

    #[test]
    fn tempo_locked_pile_gives_no_tempo_but_is_locked_material() {
        let mut row = [0u8; NUM_PITS];
        row[3] = 20; // 3 + 19 > 8: cannot move without crossing
        assert_eq!(tempo_reserve(&row), 0);
        assert_eq!(locked_stones(&row), 20);
    }

    #[test]
    fn tempo_small_pile_spreads_into_lone_stones() {
        let mut row = [0u8; NUM_PITS];
        row[6] = 3; // 6 + 2 = 8: movable; leaves lone stones at 6,7,8 -> 2 + 1 + 0
        assert_eq!(tempo_reserve(&row), 3);
        assert_eq!(locked_stones(&row), 0);
    }

    #[test]
    fn tempo_correction_zero_in_midgame() {
        assert_eq!(endgame_tempo_correction(&Board::new()), 0);
    }

    #[test]
    fn tempo_correction_penalises_side_out_of_tempo_facing_a_hoard() {
        // The measured loss shape: side to move leads in kazan, has only lone stones,
        // opponent holds locked hoards and more tempo.
        let mut b = Board::new();
        b.kazan = [55, 40];
        b.pits[0] = [1, 1, 0, 0, 0, 1, 1, 0, 1]; // tempo 8+7+3+2+0 = 20, locked 0
        b.pits[1] = [0, 2, 5, 0, 8, 3, 4, 0, 8]; // tempo 13+20+6 = 39, locked 8+4+8 = 20
        let c = endgame_tempo_correction(&b); // White to move
        assert!(c < -100, "got {}", c);
    }

    #[test]
    fn tempo_correction_is_antisymmetric_in_side_to_move() {
        let mut b = Board::new();
        b.kazan = [55, 40];
        b.pits[0] = [1, 1, 0, 0, 0, 1, 1, 0, 1];
        b.pits[1] = [0, 2, 5, 0, 8, 3, 4, 0, 8];
        let white = endgame_tempo_correction(&b);
        b.side_to_move = Side::Black;
        let black = endgame_tempo_correction(&b);
        assert_eq!(white, -black);
    }
```
Add `use crate::board::Side;` to the test module's imports if it is not already there.

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p togyzkumalaq-engine tempo`
Expected: compile error `cannot find function tempo_reserve`.

- [ ] **Step 3: Implement** (insert before `#[cfg(test)]` in `engine/src/eval.rs`)

```rust
// ---------------------------------------------------------------------------
// Tempo endgame (2026-09-12). Measured on 244 PlayOK games: losses end with ~30 stones
// on the board after 40-90 plies of lone-stone play, the bot's row empty and the
// opponent's row holding a hoard that sweeps into its kazan. Two quantities describe
// that phase: how many waiting moves a row has before a stone must cross, and how many
// stones sit in pits that cannot move without crossing (a hoard).
// ---------------------------------------------------------------------------

/// Waiting moves a row can make before any stone crosses to the opponent, ignoring the
/// opponent's replies. A pit at index `i` with `k` stones moves without crossing iff
/// `i + k - 1 <= 8`; that spreads it into lone stones at i..i+k-1, each with `8 - j`
/// further moves. Locked pits (would cross) contribute nothing. A lone stone that would
/// land in the opponent's tuzdyk is treated as staying (rare; ignored).
#[inline]
pub fn tempo_reserve(row: &[u8; NUM_PITS]) -> i32 {
    let mut t = 0i32;
    for i in 0..NUM_PITS {
        let k = row[i] as usize;
        if k == 0 || i + k - 1 > 8 {
            continue;
        }
        let (ki, ii) = (k as i32, i as i32);
        t += ki * (8 - ii) - ki * (ki - 1) / 2;
    }
    t
}

/// Stones in pits that cannot move without crossing over. While their owner still has
/// tempo they never have to move, so under the sweep rule they are that side's material.
#[inline]
pub fn locked_stones(row: &[u8; NUM_PITS]) -> i32 {
    let mut s = 0i32;
    for i in 0..NUM_PITS {
        let k = row[i] as usize;
        if k > 0 && i + k - 1 > 8 {
            s += k as i32;
        }
    }
    s
}

/// Weight of one locked stone (about half a kazan stone, MATERIAL_WEIGHT = 21): it is
/// sweep material only while its owner keeps tempo. Tuned in Task 8/9 by external gate.
pub const TEMPO_LOCK_WEIGHT: i32 = 12;
/// Weight of one waiting move of tempo advantage.
pub const TEMPO_WEIGHT: i32 = 4;
/// Tempo advantage beyond this many moves is not worth more.
pub const TEMPO_DIFF_CLAMP: i32 = 30;
/// Board stones at or below which the tempo correction applies.
pub const TEMPO_PHASE_STONES: u16 = 60;

/// Side-to-move correction for the tempo endgame. Zero above TEMPO_PHASE_STONES.
#[inline]
pub fn endgame_tempo_correction(board: &Board) -> i32 {
    if board.total_board_stones() > TEMPO_PHASE_STONES {
        return 0;
    }
    let me = board.side_to_move.index();
    let opp = 1 - me;
    let lock = locked_stones(&board.pits[me]) - locked_stones(&board.pits[opp]);
    let tempo = (tempo_reserve(&board.pits[me]) - tempo_reserve(&board.pits[opp]))
        .clamp(-TEMPO_DIFF_CLAMP, TEMPO_DIFF_CLAMP);
    lock * TEMPO_LOCK_WEIGHT + tempo * TEMPO_WEIGHT
}
```

- [ ] **Step 4: Run to verify they pass**

Run: `cargo test -p togyzkumalaq-engine tempo`
Expected: 6 passed. (Check the hoard test by hand: lock = 0 − 20 = −20 → −240; tempo =
20 − 39 = −19 → −76; total −316.)

- [ ] **Step 5: Commit**

```bash
git add engine/src/eval.rs
git commit -m "eval: tempo_reserve / locked_stones / endgame_tempo_correction (not wired yet)

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Wire the correction into the live NNUE eval path

**Files:**
- Modify: `engine/src/search.rs:180-222` (`Searcher::eval`, the `if total <= 60 { ... }` block)
- Test: `engine/src/search.rs` (new `mod tests` at the end of the file)

**Interfaces:**
- Consumes: `crate::eval::endgame_tempo_correction(&Board) -> i32` (Task 6).
- Produces: `pub(crate) fn nnue_endgame_terms(board: &Board) -> i32` in `search.rs` — the
  existing mobility/starvation/finish/kazan corrections, moved out of `eval` unchanged so
  the wiring can be asserted. `Searcher::eval` in the endgame branch returns
  `base + nnue_endgame_terms(board) + endgame_tempo_correction(board)`.
- Does not touch `eval::evaluate` (the handcrafted eval stays the independent reference
  opponent for in-lineage regression guards).

- [ ] **Step 1: Write the failing tests** (append at the end of `engine/src/search.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Side;

    fn searcher_with_production_net() -> Searcher {
        let mut s = Searcher::new(1);
        // cargo test runs with cwd = engine/, so the production weights are one level up.
        let net = crate::nnue::NnueNetwork::load("../models/engine/nnue_weights.bin")
            .expect("production weights present");
        s.set_nnue(net);
        s
    }

    fn hoard_position() -> Board {
        let mut b = Board::new();
        b.kazan = [55, 40];
        b.pits[0] = [1, 1, 0, 0, 0, 1, 1, 0, 1];
        b.pits[1] = [0, 2, 5, 0, 8, 3, 4, 0, 8];
        b.side_to_move = Side::White;
        b
    }

    #[test]
    fn live_eval_is_base_plus_endgame_terms_plus_tempo() {
        let s = searcher_with_production_net();
        let b = hoard_position();
        let base = s.nnue.as_ref().unwrap().evaluate(&b) / 64;
        let expected = base + nnue_endgame_terms(&b) + crate::eval::endgame_tempo_correction(&b);
        assert_eq!(s.eval(&b), expected);
        assert!(crate::eval::endgame_tempo_correction(&b) < -100, "precondition: the term is active here");
    }

    #[test]
    fn live_eval_has_no_endgame_terms_in_the_opening() {
        let s = searcher_with_production_net();
        let b = Board::new(); // 162 stones on the board
        let base = s.nnue.as_ref().unwrap().evaluate(&b) / 64;
        assert_eq!(s.eval(&b), base);
        assert_eq!(nnue_endgame_terms(&b), 0);
    }
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p togyzkumalaq-engine live_eval`
Expected: compile error `cannot find function nnue_endgame_terms`.

- [ ] **Step 3: Implement** — replace the whole `if total <= 60 { ... } else { base }` block
inside `Searcher::eval` (search.rs ~lines 187-222) with

```rust
            if total <= 60 {
                base + nnue_endgame_terms(board) + crate::eval::endgame_tempo_correction(board)
            } else {
                base
            }
```
and add this free function directly above `impl Searcher` (the body is the code that was
inside the block, moved verbatim; only `total`, `me`, `opp`, `my_stones`, `opp_stones` are
recomputed locally):

```rust
/// Handcrafted endgame corrections added on top of the NNUE output (side-to-move POV):
/// mobility, starvation pressure, finishing bonus, kazan proximity. Zero above 60 board
/// stones. Moved out of `Searcher::eval` unchanged (2026-09-12) so the wiring is testable.
pub(crate) fn nnue_endgame_terms(board: &Board) -> i32 {
    let me = board.side_to_move.index();
    let opp = 1 - me;
    let my_stones: u16 = board.pits[me].iter().map(|&x| x as u16).sum();
    let opp_stones: u16 = board.pits[opp].iter().map(|&x| x as u16).sum();
    let total = my_stones + opp_stones;
    if total > 60 {
        return 0;
    }
    let my_active = board.pits[me].iter().filter(|&&x| x > 0).count() as i32;
    let opp_active = board.pits[opp].iter().filter(|&&x| x > 0).count() as i32;

    // Scale corrections smoothly: total=60→1, total=30→2, total=15→3, total=5→4
    let scale = ((65 - total as i32).max(1)) / 15;
    let scale = scale.clamp(1, 4);

    // Mobility: critical endgame factor (PlayOK: mobility weight = 124 in HCE)
    let mobility_bonus = (my_active - opp_active) * 3 * scale;

    // Starvation: quadratic pressure when opponent running low
    let starvation = if opp_stones <= 20 {
        let pressure = 21 - opp_stones as i32;
        (pressure * pressure * scale) / 8
    } else { 0 };

    // Finishing: huge bonus to close out won games
    let my_kazan = board.kazan[me] as i32;
    let opp_kazan = board.kazan[opp] as i32;
    let finish_bonus = if my_kazan > opp_kazan + 5 && opp_stones <= 8 {
        (9 - opp_stones as i32) * 8 * scale
    } else { 0 };

    // Kazan proximity: accelerate when close to 82
    let kazan_bonus = if my_kazan >= 65 {
        (my_kazan - 65) * 2 * scale
    } else { 0 };

    mobility_bonus + starvation + finish_bonus + kazan_bonus
}
```
Delete the now-unused locals (`my_active`, `opp_active`, `scale`, `mobility_bonus`,
`starvation`, `finish_bonus`, `kazan_bonus`) from `Searcher::eval`; keep `total`.

- [ ] **Step 4: Run tests and a bench**

```bash
cargo test -p togyzkumalaq-engine
cargo build --release && ./target/release/togyzkumalaq-engine bench 2>&1 | tail -2
```
Expected: all tests pass; bench NPS within ~5% of the pre-change figure (≈9.3M nps on the
opening position; the correction only runs at ≤60 stones).

- [ ] **Step 5: Commit**

```bash
git add engine/src/search.rs
git commit -m "search: add tempo/locked-hoard correction to the live NNUE endgame eval

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Offline separation screen (both classes, no survivorship bias)

**Files:**
- Create: `tools/playok/analysis/eval_separation.py`
- Create: `tools/playok/analysis/test_eval_separation.py`

**Interfaces:**
- Produces: `auc(pos_scores: list[float], neg_scores: list[float]) -> float` and
  `midgame_positions(games_dir: Path) -> list[tuple[str, int]]` (position string at the
  50% ply, label 1 if White won else 0; draws skipped). CLI: `python3.12
  tools/playok/analysis/eval_separation.py <engine-binary> [--ms 300]` prints `AUC=…`.

- [ ] **Step 1: Write the failing tests** — `tools/playok/analysis/test_eval_separation.py`

```python
#!/usr/bin/env python3
"""Run: python3.12 tools/playok/analysis/test_eval_separation.py"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_separation import auc, midgame_positions, white_won  # noqa: E402

GAME = """# PlayOK togyzkumalak  20260101_000000
# White(seat0)=bot  Black(seat1)=opp
1. W4 [43(10)]  9,9,9,1,10,10,10,10,10/10,10,0,9,9,9,9,9,9/10,0/-1,-1/1
2. B9 [98]  10,10,10,2,11,11,11,11,10/10,10,0,9,9,9,9,9,1/10,0/-1,-1/0
3. W9 [99(12)]  10,10,10,2,11,11,11,11,1/11,11,1,10,10,10,10,10,0/12,0/-1,-1/1
4. B8 [88(12)]  0,0,0,0,0,0,0,0,0/11,11,1,10,10,10,10,1,1/60,20/-1,-1/0
"""


def test_auc_perfect_and_random():
    assert auc([3, 2], [1, 0]) == 1.0
    assert auc([1, 1], [1, 1]) == 0.5
    assert auc([0], [1]) == 0.0


def test_white_won_uses_sweep():
    # final pos above: white 60+0=60, black 20+65=85 -> black won
    assert white_won("0,0,0,0,0,0,0,0,0/11,11,1,10,10,10,10,1,1/60,20/-1,-1/0") is False
    assert white_won("5,0,0,0,0,0,0,0,0/0,0,0,0,0,0,0,0,0/80,77/-1,-1/1") is True


def test_midgame_positions_takes_half_ply():
    with tempfile.TemporaryDirectory() as d:
        Path(d, "game_x_vs_opp.txt").write_text(GAME)
        rows = midgame_positions(Path(d))
    assert rows == [("10,10,10,2,11,11,11,11,10/10,10,0,9,9,9,9,9,1/10,0/-1,-1/0", 0)]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("OK", name)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3.12 tools/playok/analysis/test_eval_separation.py`
Expected: `ModuleNotFoundError: No module named 'eval_separation'`.

- [ ] **Step 3: Implement** — `tools/playok/analysis/eval_separation.py`

```python
#!/usr/bin/env python3
"""How well does an engine's eval at the MIDDLE of a game separate eventual wins from
losses? Uses every recorded PlayOK game (wins AND losses), so unlike the June 2026
"blindness" metric it has no survivorship bias. Cheap screen before an external gate.

Usage: python3.12 tools/playok/analysis/eval_separation.py <engine-binary> [--ms 300]
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

GAMES_DIR = Path(__file__).resolve().parent.parent / "games"
LINE = re.compile(r"^(\d+)\. ([WB])\d+ \[[^\]]*\]\s+(\S+)")


def white_won(final_pos: str):
    """Winner under the sweep rule from the last recorded position; None on a draw."""
    w, b, k, _, _ = final_pos.split("/")
    ws = sum(map(int, w.split(","))) + int(k.split(",")[0])
    bs = sum(map(int, b.split(","))) + int(k.split(",")[1])
    if ws == bs:
        return None
    return ws > bs


def midgame_positions(games_dir: Path):
    """(position at the 50% ply, 1 if White won else 0) per decisive game."""
    rows = []
    for f in sorted(games_dir.glob("game_*.txt")):
        plies = []
        for line in f.read_text().splitlines():
            m = LINE.match(line)
            if m:
                plies.append(m.group(3))
        if len(plies) < 2:
            continue
        label = white_won(plies[-1])
        if label is None:
            continue
        rows.append((plies[len(plies) // 2 - 1], 1 if label else 0))
    return rows


def auc(pos_scores, neg_scores):
    """Probability a random positive outscores a random negative (ties count half)."""
    wins = 0.0
    for p in pos_scores:
        for n in neg_scores:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos_scores) * len(neg_scores))


def eval_white_pov(engine: str, pos: str, ms: int) -> float:
    out = subprocess.run([engine, "analyze", pos, str(ms)], capture_output=True, text=True,
                         cwd=str(Path(engine).parent)).stdout
    j = json.loads(out.strip().splitlines()[-1])
    if j.get("terminal"):
        return {"white_win": 1e6, "black_win": -1e6}.get(j.get("result"), 0.0)
    score = float(j["score"])                      # side-to-move POV
    return score if pos.endswith("/0") else -score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine")
    ap.add_argument("--ms", type=int, default=300)
    a = ap.parse_args()
    rows = midgame_positions(GAMES_DIR)
    pos, neg = [], []
    for p, label in rows:
        (pos if label else neg).append(eval_white_pov(a.engine, p, a.ms))
    print(f"games={len(rows)} wins={len(pos)} losses={len(neg)} AUC={auc(pos, neg):.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests, then the screen on both engines**

```bash
python3.12 tools/playok/analysis/test_eval_separation.py
python3.12 tools/playok/analysis/eval_separation.py models/engine/baseline
python3.12 tools/playok/analysis/eval_separation.py models/nets/eng_rulefix/togyzkumalaq-engine
```
(Rebuild `models/nets/eng_rulefix` first with `tools/9qum/make_test_engine.sh
models/engine/nnue_weights.bin models/nets/eng_rulefix --force` so it carries Task 7's
binary.) Expected: three OK lines; then two AUC lines. Screen passes if the candidate AUC
is higher than the baseline's. If it is lower, stop and revisit the weights in Task 6
before spending hours on Task 9.

- [ ] **Step 5: Commit**

```bash
git add tools/playok/analysis/eval_separation.py tools/playok/analysis/test_eval_separation.py
git commit -m "analysis: midgame eval separation (AUC over all recorded wins+losses)

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Paired external gate B — tempo eval (same clock both arms)

**Files:**
- Create: `runs/2026-09-12-gateB/jobs.json`, `runs/2026-09-12-gateB/RESULT.md`

- [ ] **Step 1: Two arms, identical budgets, only the binary differs**

Both arms use the rule-fixed code and 12 s endgame budget so the only difference is Task 7.
Build the "no-tempo" arm from the Task 5 commit:

```bash
git stash list >/dev/null  # make sure the tree is clean first: git status --short
git worktree add /home/nurlykhan/9QumalaqV2-notempo $(git log --format=%H --grep="tools: endgame time budget" -1)
(cd /home/nurlykhan/9QumalaqV2-notempo && cargo build --release)
mkdir -p models/nets/eng_notempo && cp /home/nurlykhan/9QumalaqV2-notempo/target/release/togyzkumalaq-engine models/nets/eng_notempo/
cp models/engine/nnue_weights.bin models/engine/egtb.bin models/engine/opening_book.txt models/nets/eng_notempo/
tools/9qum/make_test_engine.sh models/engine/nnue_weights.bin models/nets/eng_tempo --force
```

- [ ] **Step 2: Job list** — `runs/2026-09-12-gateB/jobs.json`

```json
[
  {"name": "notempo",
   "cmd": ["python3.12", "tools/9qum/match.py", "--games", "48", "--level", "i",
           "--engine", "models/nets/eng_notempo/togyzkumalaq-engine",
           "--move-ms", "1800", "--endgame-move-ms", "12000",
           "--out", "runs/2026-09-12-gateB/notempo"]},
  {"name": "tempo",
   "cmd": ["python3.12", "tools/9qum/match.py", "--games", "48", "--level", "i",
           "--engine", "models/nets/eng_tempo/togyzkumalaq-engine",
           "--move-ms", "1800", "--endgame-move-ms", "12000",
           "--out", "runs/2026-09-12-gateB/tempo"]}
]
```

- [ ] **Step 3: Run, then read**

```bash
python3.12 tools/9qum/run_measurements.py --jobs runs/2026-09-12-gateB/jobs.json --run-dir runs/2026-09-12-gateB
python3.12 tools/9qum/lead_profile.py runs/2026-09-12-gateB/notempo runs/2026-09-12-gateB/tempo
```
Write `RESULT.md` with both scores, opponent config, lead profile, and the AUCs from Task 8.

Decision rule:
- tempo − notempo ≥ +12 points → run the same two jobs again with `--games 96` into
  `runs/2026-09-12-gateB96/` (promotion evidence), then Task 10.
- within ±12 → one retune only: halve `TEMPO_LOCK_WEIGHT` to 6 and `TEMPO_WEIGHT` to 2
  (or double them) based on the lead profile (if the candidate still empties first, raise;
  if it stops building any lead, lower), rerun Task 8's screen and this gate once. Two
  failed retunes → stop, do not promote the term; keep Tasks 1-5.
- ≤ −12 → revert Task 7 (`git revert` that commit), keep Task 6's functions for Phase 4.

- [ ] **Step 4: Commit the result note**

```bash
git add -f runs/2026-09-12-gateB/RESULT.md runs/2026-09-12-gateB/jobs.json
git commit -m "measure: gate B (tempo correction vs none, same clock), paired on 9qum

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
git worktree remove /home/nurlykhan/9QumalaqV2-notempo
```

---

### Task 10: Promote to production (requires the gate AND user approval)

**Files:**
- Overwrite: `models/engine/baseline` (binary)
- Modify: `docs/MEASUREMENT_PROTOCOL.md` (append rule 10)

- [ ] **Step 1: Ask the user** — show `runs/2026-09-12-gateA/RESULT.md` and
  `runs/2026-09-12-gateB*/RESULT.md` and get an explicit "promote". Do not continue without it.

- [ ] **Step 2: Promote and verify the served engine**

```bash
cp models/engine/baseline runs/baseline_before_2026-09-12
cargo build --release && cp target/release/togyzkumalaq-engine models/engine/baseline
printf 'go time 200 pos 0,0,0,0,0,0,0,0,0/0,0,0,0,0,0,0,0,3/80,79/-1,-1/1\nquit\n' | models/engine/baseline serve
```
Expected: `ready` then a `bestmove 8 ...` line (Black must move; the old binary printed
`terminal black_win`).

- [ ] **Step 3: Append to `docs/MEASUREMENT_PROTOCOL.md`**

```markdown
## 10. The referee must implement the real terminal rule

The game ends only when the side to move has no stones. Until 2026-09-12 the Rust core,
the Python rules class and the EGTB all ended it as soon as either side was empty; ~110
PlayOK games and 1120 9qum states show play continuing after a self-empty (forced feeds in
~3%). Any harness that referees a game must use `Board::game_result()` /
`TogyzQumalaq._check_winner()` from after that fix; `research/training/gen_engine_games.py`
had it right all along. Engine-generated data from before that date carries the wrong
endgame model in its play (and in ~3% of labels).
```

- [ ] **Step 4: Commit**

```bash
git add models/engine/baseline docs/MEASUREMENT_PROTOCOL.md
git commit -m "product: promote rule-fixed engine with endgame clock (+tempo eval if gate B passed)

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: Live validation on PlayOK against the real target

**Files:** none (logs under `tools/playok/games/`)

- [ ] **Step 1: Play 20 scouted games against ≥1800 opponents with the promoted engine**

```bash
python3.12 tools/playok/bridge.py --live --scout --min-elo 1800 --max-games 20 \
    --move-time-ms 1800 --endgame-move-time-ms 12000 --rated
```
Expected: the bridge prints `budget=12000ms` lines once ≤40 stones remain, and zero
`RULE MISMATCH` lines over the run. Any mismatch line is a bug: stop and report it with
the position.

- [ ] **Step 2: Re-run the loss-structure analysis**

```bash
python3.12 tools/playok/analysis/analyze.py
python3.12 - <<'EOF'
import csv, re
rows = list(csv.DictReader(open('tools/playok/analysis/summary.csv')))
new = [r for r in rows if r['file'] >= 'game_20260912']
for r in new:
    last = [l for l in open('tools/playok/games/' + r['file']).read().splitlines() if re.match(r'\d+\. ', l)][-1].split()[-1]
    w, b = last.split('/')[:2]
    print(r['result'], 'bot_board=', sum(map(int, w.split(','))), 'opp_board=', sum(map(int, b.split(','))), 'opp_max_pit=', max(map(int, b.split(','))), r['opponent'])
EOF
```
Expected: in the new games the bot no longer ends losses with 0 stones on its side while
the opponent holds a 15+ hoard (the pre-fix signature: 30 of 63 losses). Record the table
in memory and in `runs/2026-09-12-gateB/RESULT.md` under "PlayOK live".

- [ ] **Step 3: Commit the recorded games**

```bash
git add tools/playok/games/game_202609*.txt tools/playok/analysis/summary.csv tools/playok/analysis/summary.json
git commit -m "playok: 20 live games after the endgame fixes

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Out of scope (next plan)

Phase 4 (retraining with a margin target and exact-count inputs) is written after Task 9's
numbers exist; see the spec's "Phase 4 (deferred)" section for the direction.
