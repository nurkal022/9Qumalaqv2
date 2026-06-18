# Sweep-aware Endgame Evaluation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the NNUE-path evaluator sweep-aware in the endgame so it stops over-rating a greedy kazan lead and counts both sides' board stones as future kazan (the measured cause of the +15→−10 endgame collapse).

**Architecture:** Add one tested pure function `endgame_sweep_correction(board)` (plus helper `sweep_weight`) to `engine/src/eval.rs`. Wire it into the NNUE branch of `Searcher::eval()` in `engine/src/search.rs`, replacing the current correction block (which rewards the inverted "starve the opponent" dynamic). The handcrafted fallback `evaluate()` already has an equivalent `pit_asymmetry` term, so it is left unchanged. Validate with `tools/ab_match.py` (same NNUE on both sides) and a re-run of the 62-loss eval-blindness probe.

**Tech Stack:** Rust (cargo workspace: `core`, `engine`, `mcts`), Python validation harness (`tools/ab_match.py`), engine `serve` protocol.

**Commit policy:** This repo commits only on the user's go-ahead. Commit steps are included per TDD discipline; confirm with the user before the first `git commit`. Branch is `rust-mcts` (feature branch, not main) — no new branch needed.

**Production safety:** `models/engine/baseline` (live web + PlayOK engine) is NOT modified until Task 4, gated on the ab_match result AND explicit user approval.

---

## File Structure

- `engine/src/eval.rs` — ADD `sweep_weight()` + `endgame_sweep_correction()` (pub) and unit tests. `evaluate()` itself unchanged.
- `engine/src/search.rs` — MODIFY `Searcher::eval()` (lines ~174-222): replace the inline endgame-correction block with a call to `crate::eval::endgame_sweep_correction(board)`.
- `/tmp/tk_evaltest_new.py` — validation script (adapted copy of `/tmp/tk_evaltest.py`) pointed at the new binary.

---

## Task 1: Sweep-aware correction function in eval.rs (TDD)

**Files:**
- Modify: `engine/src/eval.rs` (add functions after `evaluate()`'s helpers, before `#[cfg(test)]` at line 312; add tests inside the existing `mod tests`)
- Test: `engine/src/eval.rs` (`mod tests`)

- [ ] **Step 1: Write the failing tests** (add inside `mod tests` at `engine/src/eval.rs`)

```rust
    #[test]
    fn test_sweep_weight_monotonic_and_bounds() {
        // Above the endgame threshold: no correction.
        assert_eq!(sweep_weight(61), 0);
        assert_eq!(sweep_weight(120), 0);
        // Inside the endgame: weight rises as the board empties, toward kazan value.
        assert_eq!(sweep_weight(60), 4);
        assert_eq!(sweep_weight(10), 18);
        // Monotonic non-increasing in board_total.
        let mut prev = i32::MAX;
        for t in 0..=70u16 {
            let w = sweep_weight(t);
            assert!(w <= prev, "sweep_weight not monotonic at {}: {} > {}", t, w, prev);
            prev = w;
        }
    }

    #[test]
    fn test_sweep_correction_zero_in_midgame() {
        let b = Board::new(); // 162 stones on board -> midgame
        assert_eq!(endgame_sweep_correction(&b), 0,
            "midgame (board>60) must get no sweep correction");
    }

    #[test]
    fn test_sweep_correction_penalizes_emptying_side() {
        // The measured loss shape: side-to-move (White) leads in kazan but its
        // board is nearly empty while the opponent hoards stones that will sweep.
        let mut b = Board::new();
        b.kazan = [55, 40];
        b.pits[0] = [1, 1, 0, 0, 0, 1, 1, 0, 1]; // white board = 5
        b.pits[1] = [0, 2, 5, 0, 8, 3, 4, 0, 8]; // black board = 30  (total 35 -> w=7)
        let c = endgame_sweep_correction(&b); // white to move (default)
        assert!(c < -100,
            "side whose board is emptied while opp hoards must be penalized, got {}", c);
    }

    #[test]
    fn test_sweep_correction_rewards_hoarding_side() {
        // Mirror: side-to-move holds the hoard, opponent's board is empty.
        let mut b = Board::new();
        b.kazan = [40, 55];
        b.pits[0] = [0, 2, 5, 0, 8, 3, 4, 0, 8]; // white board = 30
        b.pits[1] = [1, 1, 0, 0, 0, 1, 1, 0, 1]; // black board = 5
        let c = endgame_sweep_correction(&b); // white to move
        assert!(c > 100, "side holding the sweepable hoard must be rewarded, got {}", c);
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p togyzkumalaq-engine sweep`
Expected: FAIL — `cannot find function sweep_weight` / `endgame_sweep_correction` in this scope.

- [ ] **Step 3: Implement the functions** (add in `engine/src/eval.rs` immediately before `#[cfg(test)]` at line 312)

```rust
/// Endgame sweep weight: under the end-of-game sweep rule each side scoops its
/// remaining board stones into its own kazan, so an own-side board stone
/// converges to kazan value (MATERIAL_WEIGHT = 21) as the board empties.
/// Returns 0 above the endgame threshold (midgame: the base eval handles it).
/// Knots are an initial estimate; tuned via ab_match in Task 3.
#[inline]
pub fn sweep_weight(board_total: u16) -> i32 {
    match board_total {
        0..=10 => 18,
        11..=20 => 14,
        21..=30 => 10,
        31..=45 => 7,
        46..=60 => 4,
        _ => 0,
    }
}

/// Sweep-aware endgame correction (side-to-move POV) to ADD to a base eval that
/// under-counts on-board stones. Captures the measured loss dynamic: the
/// opponent hoards board stones that sweep into its kazan while our side empties.
/// Zero in the midgame (board_total > 60).
#[inline]
pub fn endgame_sweep_correction(board: &Board) -> i32 {
    let me = board.side_to_move.index();
    let opp = 1 - me;
    let my_board: i32 = board.pits[me].iter().map(|&x| x as i32).sum();
    let opp_board: i32 = board.pits[opp].iter().map(|&x| x as i32).sum();
    let total = (my_board + opp_board) as u16;
    let w = sweep_weight(total);
    if w == 0 {
        return 0;
    }
    // Primary: own-side stones are future kazan (the sweep term).
    let mut c = (my_board - opp_board) * w;

    // Retained light mobility/tempo term (kept from the old correction).
    let my_active = board.pits[me].iter().filter(|&&x| x > 0).count() as i32;
    let opp_active = board.pits[opp].iter().filter(|&&x| x > 0).count() as i32;
    let scale = ((65 - total as i32).max(1) / 15).clamp(1, 4);
    c += (my_active - opp_active) * 3 * scale;

    // Finishing: close out a genuinely won game when the opponent is nearly empty.
    if board.kazan[me] as i32 > board.kazan[opp] as i32 + 5 && opp_board <= 8 {
        c += (9 - opp_board) * 8 * scale;
    }
    c
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p togyzkumalaq-engine sweep`
Expected: PASS (4 tests). Also run `cargo test -p togyzkumalaq-engine` to confirm no existing eval test regressed.

- [ ] **Step 5: Commit** (after user go-ahead)

```bash
git add engine/src/eval.rs
git commit -m "engine: add sweep-aware endgame_sweep_correction + tests"
```

---

## Task 2: Wire correction into the NNUE path in search.rs

**Files:**
- Modify: `engine/src/search.rs:174-222` (`Searcher::eval()`)

- [ ] **Step 1: Replace the NNUE-branch correction block**

Replace the body of `fn eval(&self, board: &Board) -> i32` (lines ~174-222) with:

```rust
    fn eval(&self, board: &Board) -> i32 {
        if let Some(ref nnue) = self.nnue {
            let base = nnue.evaluate(board) / 64;
            // Sweep-aware endgame correction (0 in midgame). Replaces the prior
            // mobility/starvation/finish block, which rewarded the inverted
            // dynamic (starving the opponent) while real losses come from the
            // opponent hoarding board stones that sweep into its kazan.
            base + crate::eval::endgame_sweep_correction(board)
        } else {
            evaluate(board)
        }
    }
```

- [ ] **Step 2: Build to verify it compiles**

Run: `cargo build --release -p togyzkumalaq-engine`
Expected: builds clean (no warnings about unused `predict_landing`/`move_creates_tuzdyk` — those are used elsewhere in search.rs; if a now-unused helper warning appears, leave the helper, it is used by move ordering).

- [ ] **Step 3: Run the full engine test suite**

Run: `cargo test -p togyzkumalaq-engine`
Expected: PASS (all existing + 4 new sweep tests).

- [ ] **Step 4: Commit** (after user go-ahead)

```bash
git add engine/src/search.rs
git commit -m "engine: NNUE eval uses sweep-aware endgame correction"
```

---

## Task 3: Validate — A/B gate (same NNUE) + blindness re-test

**Files:**
- Create: `/tmp/tk_evaltest_new.py` (copy of `/tmp/tk_evaltest.py`, BIN changed)

- [ ] **Step 1: Sync engine/ assets to production so both A/B sides use the SAME (Jun-8 production) NNUE**

```bash
cd /home/nurlykhan/9QumalaqV2
cp models/engine/nnue_weights.bin models/engine/egtb.bin models/engine/opening_book.txt engine/
```
(These engine/*.bin are gitignored; this only affects local A/B fidelity, reversible.)

- [ ] **Step 2: Build the BASELINE binary from current source BEFORE editing is irrelevant now — instead build baseline from git HEAD into a temp path**

Build the pre-change baseline by stashing the change, building, then restoring:
```bash
git stash push -- engine/src/eval.rs engine/src/search.rs
cargo build --release -p togyzkumalaq-engine
cp target/release/togyzkumalaq-engine /tmp/eng_base
git stash pop
cargo build --release -p togyzkumalaq-engine   # rebuild WITH the change
```
Expected: `/tmp/eng_base` = current-HEAD behavior; `target/release/togyzkumalaq-engine` = new behavior. Both load `engine/`'s (now production) assets when run with cwd=engine/.

- [ ] **Step 3: Run the A/B regression gate (100 games, 1500 ms — bot's live time)**

Run:
```bash
python3 tools/ab_match.py target/release/togyzkumalaq-engine /tmp/eng_base 100 1500 --jobs 8 --min 50
```
Expected: prints W/D/L from the new engine's perspective + Elo. **Gate: new engine ≥ 50% (no regression); target ≥ 55% with positive Elo.** Exit code 0 if ≥ --min.

- [ ] **Step 4: Blindness re-test (confirms we fixed the CAUSE, not just Elo)**

Create `/tmp/tk_evaltest_new.py` as a copy of `/tmp/tk_evaltest.py` with the binary path changed:
```bash
sed 's#/models/engine/baseline#/target/release/togyzkumalaq-engine#' /tmp/tk_evaltest.py > /tmp/tk_evaltest_new.py
python3 /tmp/tk_evaltest_new.py
```
Expected (per spec's board-diff trajectory): at the **75%** mark the engine should flip most lost games from "winning" to losing/contested (was: median +28, 58% rated ≥0); at **90%** near-universally losing; at **50%** a clear downward shift from +128 (partial, board-diff only −5 there). If 75%/90% do NOT improve, the weight ramp is too weak — raise `sweep_weight` knots and repeat Task 3.

- [ ] **Step 5: Record the result** (no commit; results go in the response + memory)

Summarize: A/B score% and Elo; before/after blindness numbers at 50/75/90%.

---

## Task 4: Promote to production (GATED — user approval required)

**Files:**
- Modify: `models/engine/baseline` (binary copy)

- [ ] **Step 1: Only if Task 3 gate passed AND user approves**, promote:

```bash
cp target/release/togyzkumalaq-engine models/engine/baseline
```
(Reversible: previous baseline is recoverable from git history of the promote commit / prior binary.)

- [ ] **Step 2: Commit** (after user go-ahead)

```bash
git add -A
git commit -m "product: promote sweep-aware endgame eval (ab_match <score>%, Elo +<n>)"
```

- [ ] **Step 3: Sanity-check the promoted engine still serves**

Run: `printf 'go time 1000 pos 9,9,9,9,9,9,9,9,9/9,9,9,9,9,9,9,9,9/0,0/-1,-1/0\nquit\n' | ./models/engine/baseline serve`
Expected: `ready` then `bestmove 3 ...` (hole 4 opening).

---

## Self-Review

- **Spec coverage:** sweep term (Task 1/2) ✓; remove inverted starve (Task 2) ✓; eval.rs alignment — HCE already has `pit_asymmetry`, documented as no-change ✓; ab_match gate (Task 3) ✓; blindness re-test (Task 3) ✓; unit tests incl. crafted sweep position (Task 1) ✓; production untouched until gate+approval (Task 4) ✓; same-NNUE A/B correctness (Task 3 Step 1-2) ✓.
- **Placeholder scan:** none — all code/commands concrete; ramp knots are explicit values with a tuning note.
- **Type consistency:** `sweep_weight(u16) -> i32` and `endgame_sweep_correction(&Board) -> i32` used identically in eval.rs tests and the search.rs call site (`crate::eval::endgame_sweep_correction`). Board fields `pits`/`kazan` are `pub`; `side_to_move.index()` matches existing search.rs usage.
