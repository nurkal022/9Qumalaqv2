# Monorepo Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the monorepo into clean zones (`core/` + `engine/` + `mcts/` Rust workspace, `product/web`, `research/`, `models/`) with the game rules deduplicated into a single shared `core` crate, in one verified big-bang cutover.

**Architecture:** Cargo workspace at the root with a new `togyzkumalaq-core` crate holding the single `board.rs`; `engine` and `mcts` depend on it via a one-line module alias (`pub(crate) use togyzkumalaq_core as board;`) so existing `crate::board::*` references resolve unchanged. Folders move with `git mv` (history preserved). The product reads its engine only from `models/`. Everything lands on one branch, merged only when every build/test passes.

**Tech Stack:** Rust (Cargo workspace, 2021 edition), Python/FastAPI (backend, pytest, .venv), React/TS (Vite, vitest), git.

**Spec:** `docs/superpowers/specs/2026-05-31-monorepo-restructure-design.md`

**Environment note:** Building/testing the `mcts` crate links ONNX Runtime (`ort` with `load-dynamic`) and needs `LD_LIBRARY_PATH` to include the nvidia pip package libs (per project notes). If `cargo build -p mcts` fails to link `onnxruntime`, set that env first.

**Pre-flight (execution time):** This plan should be executed on an isolated branch/worktree created via `superpowers:using-git-worktrees`. All `git` commands below assume that branch. Baseline before starting (must already be green): `cargo test --manifest-path engine/Cargo.toml` (15 tests), frontend `npm test` (18 tests), backend `.venv/bin/pytest` (75 tests).

---

### Task 1: Branch baseline + root `.gitignore`

**Files:**
- Create/Modify: `.gitignore`

- [ ] **Step 1: Confirm the green baseline**

Run:
```bash
cargo test --manifest-path engine/Cargo.toml 2>&1 | tail -2
```
Expected: `test result: ok. 15 passed`.

- [ ] **Step 2: Append the restructure ignores to `.gitignore`**

Append this block to `.gitignore` (keep existing entries):
```gitignore
# --- restructure: build & artifact ignores ---
target/
**/__pycache__/
**/node_modules/
**/dist/
*.db
datasets/**
research/runs/**
!research/runs/**/config.yaml
!research/runs/**/summary.md
```

- [ ] **Step 3: Verify negations work**

Run:
```bash
git check-ignore -v research/runs/foo/checkpoints/x.pt research/runs/foo/summary.md; echo "exit=$?"
```
Expected: the `.pt` path prints a matching rule; `summary.md` prints nothing (un-ignored). The command exit code is non-zero because the last path is not ignored — that is correct.

- [ ] **Step 4: Commit**

```bash
git add .gitignore
git commit -m "chore: restructure-ready .gitignore (target, runs/, datasets)"
```

---

### Task 2: Cargo workspace + `togyzkumalaq-core` crate; engine uses it

**Files:**
- Create: `Cargo.toml` (workspace root)
- Create: `core/Cargo.toml`
- Create: `core/src/lib.rs`
- Move: `engine/src/board.rs` → `core/src/board.rs`
- Modify: `engine/Cargo.toml` (add dep)
- Modify: `engine/src/main.rs:1` (`mod board;` → alias)

- [ ] **Step 1: Move the board module into the new crate**

```bash
mkdir -p core/src
git mv engine/src/board.rs core/src/board.rs
```

- [ ] **Step 2: Create `core/src/lib.rs`**

```rust
//! togyzkumalaq-core — shared Togyzkumalak game rules (single source of truth).
mod board;
pub use board::*;
```

- [ ] **Step 3: Create `core/Cargo.toml`**

```toml
[package]
name = "togyzkumalaq-core"
version = "0.1.0"
edition = "2021"

[lib]
name = "togyzkumalaq_core"
path = "src/lib.rs"
```

- [ ] **Step 4: Create the workspace root `Cargo.toml`**

```toml
[workspace]
resolver = "2"
members = ["core", "engine", "rust-mcts"]
```

- [ ] **Step 5: Add the core dependency to `engine/Cargo.toml`**

Change the `[dependencies]` section to:
```toml
[dependencies]
togyzkumalaq-core = { path = "../core" }
```

- [ ] **Step 6: Point engine at the shared module via alias**

In `engine/src/main.rs`, replace line 1 (`mod board;`) with:
```rust
pub(crate) use togyzkumalaq_core as board;
```
(Every other file keeps using `crate::board::...` / `use crate::board::...` / `board::...` unchanged — they now resolve to the core crate through this alias.)

- [ ] **Step 7: Build + test engine through the workspace**

Run:
```bash
cargo test -p togyzkumalaq-engine 2>&1 | tail -3
```
Expected: `test result: ok` — engine's own non-board tests pass (the 10 board tests have moved to `togyzkumalaq-core`, verified in Step 8). So engine now reports ~5 tests, not 15; that drop is expected.

- [ ] **Step 8: Test the core crate directly**

Run:
```bash
cargo test -p togyzkumalaq-core 2>&1 | tail -3
```
Expected: `test result: ok. 10 passed` (the board.rs unit tests, incl. the unmake/conservation tests added earlier).

- [ ] **Step 9: Commit**

```bash
git add Cargo.toml core engine/Cargo.toml engine/src/main.rs
git commit -m "refactor(core): extract togyzkumalaq-core crate; engine depends on it"
```

---

### Task 3: `mcts` (rust-mcts) uses core — remove the duplicate board.rs

**Files:**
- Delete: `rust-mcts/src/board.rs`
- Modify: `rust-mcts/Cargo.toml` (add dep)
- Modify: `rust-mcts/src/main.rs:1` (`mod board;` → alias)

- [ ] **Step 1: Confirm the duplicate is byte-identical before deleting**

Run:
```bash
diff -q rust-mcts/src/board.rs core/src/board.rs && echo IDENTICAL
```
Expected: `IDENTICAL` (so nothing mcts-specific is lost).

- [ ] **Step 2: Remove the duplicate**

```bash
git rm rust-mcts/src/board.rs
```

- [ ] **Step 3: Add the core dependency to `rust-mcts/Cargo.toml`**

Add to the `[dependencies]` table (keep the existing ort/ndarray/etc. lines):
```toml
togyzkumalaq-core = { path = "../core" }
```

- [ ] **Step 4: Point mcts at the shared module via alias**

In `rust-mcts/src/main.rs`, replace line 1 (`mod board;`) with:
```rust
pub(crate) use togyzkumalaq_core as board;
```

- [ ] **Step 5: Build mcts**

Run (set `LD_LIBRARY_PATH` first if linking fails — see Environment note):
```bash
cargo build -p rust-mcts 2>&1 | tail -3
```
Expected: `Finished` with no errors. (Board tests now live in `togyzkumalaq-core`, not in this crate — run `cargo test -p togyzkumalaq-core` if you want to re-confirm the 10 board tests.)

- [ ] **Step 6: Build the whole workspace**

Run:
```bash
cargo build 2>&1 | tail -3
```
Expected: `Finished` with no errors (core + engine + rust-mcts).

- [ ] **Step 7: Commit**

```bash
git add rust-mcts/Cargo.toml rust-mcts/src/main.rs
git commit -m "refactor(mcts): use togyzkumalaq-core; drop duplicate board.rs (single source)"
```

---

### Task 4: Move `parse_position` into core (dedup the two copies)

**Files:**
- Modify: `core/src/board.rs` (add `parse_position` + a unit test)
- Modify: `core/src/lib.rs` (already `pub use board::*` — no change needed)
- Modify: `engine/src/main.rs` (delete local `parse_position`, call `board::parse_position`)
- Modify: `rust-mcts/src/main.rs` (delete local `parse_position`, call `board::parse_position`)

- [ ] **Step 1: Read both copies and pick the canonical one**

Run:
```bash
sed -n '707,735p' engine/src/main.rs; echo '--- mcts ---'; sed -n '321,346p' rust-mcts/src/main.rs
```
Use the engine copy as canonical (it returns `Result<Board, String>`); the mcts copy differs only in how it assigns `side_to_move`/tuzdyks — fold any genuine difference into the single core version.

- [ ] **Step 2: Add `parse_position` to `core/src/board.rs`**

Append inside `core/src/board.rs` (outside `impl Board`, as a free `pub fn`), using the canonical body from Step 1:
```rust
/// Parse position string: "w0,..,w8/b0,..,b8/kw,kb/tw,tb/side" into a Board.
pub fn parse_position(pos: &str) -> Result<Board, String> {
    // <canonical body from engine/src/main.rs:707-735>
}
```

- [ ] **Step 3: Add a round-trip unit test in `core/src/board.rs` tests module**

```rust
#[test]
fn test_parse_position_initial() {
    let b = parse_position("9,9,9,9,9,9,9,9,9/9,9,9,9,9,9,9,9,9/0,0/-1,-1/0").unwrap();
    assert_eq!(b, Board::new());
}

#[test]
fn test_parse_position_rejects_malformed() {
    assert!(parse_position("bad").is_err());
}
```

- [ ] **Step 4: Run the core tests (expect the new ones to pass)**

Run:
```bash
cargo test -p togyzkumalaq-core parse_position 2>&1 | tail -4
```
Expected: 2 passed.

- [ ] **Step 5: Delete the engine local copy and call core's**

In `engine/src/main.rs`: delete the local `fn parse_position` (lines ~707-735) and replace every call `parse_position(` with `board::parse_position(`.

- [ ] **Step 6: Delete the mcts local copy and call core's**

In `rust-mcts/src/main.rs`: delete the local `fn parse_position` (lines ~321-346) and replace every call `parse_position(` with `board::parse_position(`.

- [ ] **Step 7: Build + test the workspace**

Run:
```bash
cargo build 2>&1 | tail -2 && cargo test -p togyzkumalaq-engine 2>&1 | tail -2
```
Expected: `Finished`; engine tests pass.

- [ ] **Step 8: Smoke-test engine serve (parse_position is on the hot path)**

Run:
```bash
printf 'go time 100 pos 9,9,9,9,9,9,9,9,9/9,9,9,9,9,9,9,9,9/0,0/-1,-1/0\nquit\n' | cargo run -q -p togyzkumalaq-engine -- serve 2>/dev/null | head -2
```
Expected: a `ready` line then a `bestmove ...` line.

- [ ] **Step 9: Commit**

```bash
git add core engine/src/main.rs rust-mcts/src/main.rs
git commit -m "refactor(core): single parse_position in core; drop both copies"
```

---

### Task 5: Rename `rust-mcts/` → `mcts/` (folder + package + binary refs)

**Files:**
- Move: `rust-mcts/` → `mcts/`
- Modify: `Cargo.toml` (workspace member)
- Modify: `mcts/Cargo.toml` (package name)
- Modify: any script/doc referencing the `rust-mcts` binary

- [ ] **Step 1: Move the folder**

```bash
git mv rust-mcts mcts
```

- [ ] **Step 2: Rename the package**

In `mcts/Cargo.toml`, change `name = "rust-mcts"` to `name = "mcts"`.

- [ ] **Step 3: Update the workspace member**

In root `Cargo.toml`, change `members = ["core", "engine", "rust-mcts"]` to `members = ["core", "engine", "mcts"]`.

- [ ] **Step 4: Find references to the old binary/path**

Run:
```bash
grep -rIn "rust-mcts" --include=*.sh --include=*.py --include=*.md . | grep -v "^./mcts/" | grep -v node_modules
```
For each hit that names the binary (`target/release/rust-mcts`) or the old folder path, update it to `mcts` / `mcts/...`.

- [ ] **Step 5: Build + test from the new location**

Run:
```bash
cargo build 2>&1 | tail -2 && cargo test -p mcts board:: 2>&1 | tail -3
```
Expected: `Finished`; board tests reachable.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: rename rust-mcts crate/folder to mcts; update references"
```

---

### Task 6: Move `web-v2/` → `product/web/` + fix backend `REPO_ROOT`

**Files:**
- Move: `web-v2/` → `product/web/`
- Modify: `product/web/backend/app/config.py` (`REPO_ROOT` depth + engine_path comment)

- [ ] **Step 1: Move the product**

```bash
mkdir -p product
git mv web-v2 product/web
```

- [ ] **Step 2: Fix the repo-root depth**

In `product/web/backend/app/config.py`, the file moved one level deeper. Change:
```python
REPO_ROOT = Path(__file__).resolve().parents[3]
```
to:
```python
REPO_ROOT = Path(__file__).resolve().parents[4]
```

- [ ] **Step 3: Verify REPO_ROOT resolves correctly**

Run:
```bash
cd product/web/backend && .venv/bin/python -c "from app.config import REPO_ROOT; print(REPO_ROOT)"; cd -
```
Expected: prints the repo root absolute path (e.g. `/home/nurlykhan/9QumalaqV2`), NOT a subdirectory.

- [ ] **Step 4: Backend test suite**

Run:
```bash
cd product/web/backend && timeout 240 .venv/bin/pytest -q 2>&1 | tail -3; cd -
```
Expected: `75 passed`.

- [ ] **Step 5: Frontend typecheck + tests + build**

Run:
```bash
npm --prefix product/web/frontend run typecheck && \
npm --prefix product/web/frontend test && \
npm --prefix product/web/frontend run build 2>&1 | tail -3
```
Expected: typecheck clean, `18 passed`, `built in ...`.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: move web-v2 -> product/web; fix backend REPO_ROOT depth"
```

---

### Task 7: Create `research/` zone (move training/data/eval scripts + legacy artifacts)

**Files:**
- Create: `research/{training,data,eval,configs,runs}/`
- Move: `mcts/scripts/*` and root `*.sh`/eval files into the right subfolders
- Modify: moved scripts that hardcode old paths

- [ ] **Step 1: Create the zone**

```bash
mkdir -p research/training research/data research/eval research/configs research/runs/_legacy
```

- [ ] **Step 2: Move training scripts**

```bash
git mv mcts/scripts/train_loop.py mcts/scripts/train_alphazero.py mcts/scripts/train_distillation.py mcts/scripts/train_hybrid.py mcts/scripts/train_master.py research/training/
```

- [ ] **Step 3: Move data scripts**

```bash
git mv mcts/scripts/collect_master_games.py mcts/scripts/export_onnx.py research/data/
```

- [ ] **Step 4: Move eval scripts/harness**

```bash
git mv mcts/scripts/eval_configb_style.py mcts/scripts/eval_vs_engine.py mcts/scripts/summarize_training.py research/eval/
git mv mcts/run_eval_all.sh mcts/run_eval_quick.sh mcts/run_eval_1ply.sh mcts/run_final_eval.sh mcts/run_champ_test.sh research/eval/ 2>/dev/null || true
git mv play_champion.sh play_mcts.sh research/eval/ 2>/dev/null || true
```

- [ ] **Step 5: Move legacy artifacts out of the way (then gitignored)**

```bash
mv mcts/checkpoints_v3 mcts/checkpoints_night research/runs/_legacy/ 2>/dev/null || true
mv eval_results*.txt final_eval.txt nigtht_report.md research/runs/_legacy/ 2>/dev/null || true
```
(These are mostly untracked, so `mv` not `git mv`. `research/runs/**` is gitignored, so they leave the index cleanly.)

- [ ] **Step 6: Fix hardcoded paths in moved scripts**

Run:
```bash
grep -rIn "rust-mcts\|/scripts/\|checkpoints_v3\|9QumalaqV2/" research/ | grep -v Binary
```
For each Python/sh hit, update the path to the new layout (or make it a CLI/env arg). At minimum the scripts must `ast.parse` clean:
```bash
for f in research/training/*.py research/data/*.py research/eval/*.py; do python3 -c "import ast,sys;ast.parse(open(sys.argv[1]).read())" "$f" || echo "PARSE FAIL: $f"; done
```
Expected: no `PARSE FAIL` lines.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: research/ zone (training/data/eval); legacy artifacts to runs/_legacy"
```

---

### Task 8: `models/` zone + product reads engine from it

**Files:**
- Create: `models/{engine,nets}/`
- Move: baseline engine binary → `models/engine/baseline`; best net → `models/nets/`
- Modify: `product/web/backend/app/config.py` (`engine_path` → models)

- [ ] **Step 1: Create the zone and place the champion binary**

```bash
mkdir -p models/engine models/nets
cp engine/target/release/togyzkumalaq-engine-baseline models/engine/baseline
chmod +x models/engine/baseline
```

- [ ] **Step 2: Place the best net (if present)**

```bash
[ -f research/runs/_legacy/checkpoints_v3/iter_2645.pt ] && cp research/runs/_legacy/checkpoints_v3/iter_2645.pt models/nets/iter_2645.pt || echo "net not found — skip, note in summary"
```

- [ ] **Step 3: Point the product at the model**

In `product/web/backend/app/config.py`, change the `engine_path` default to:
```python
    engine_path: Path = REPO_ROOT / "models" / "engine" / "baseline"
```
and update its comment to reference `models/engine/`.

- [ ] **Step 4: Verify the product starts the model engine (backend tests spawn it)**

Run:
```bash
cd product/web/backend && timeout 240 .venv/bin/pytest -q tests/test_engine_pool.py 2>&1 | tail -3; cd -
```
Expected: the engine-pool tests pass (the `models/engine/baseline` binary answers serve mode).

- [ ] **Step 5: Smoke duel sanity (baseline serve protocol alive)**

Run:
```bash
printf 'go time 100 pos 9,9,9,9,9,9,9,9,9/9,9,9,9,9,9,9,9,9/0,0/-1,-1/0\nquit\n' | models/engine/baseline serve 2>/dev/null | head -2
```
Expected: `ready` then `bestmove ...`.

- [ ] **Step 6: Commit**

```bash
git add models product/web/backend/app/config.py
git commit -m "feat(product): serve engine from models/engine/baseline (decoupled from build output)"
```

---

### Task 9: Archive old web, move tools/docs, handle the APK + symlinks

**Files:**
- Move: `web/` → `archive/web-old/`; `deploy_lan.py` → `tools/`; `CHAMPION_SETUP.md` → `docs/`
- Decide: `6666.apk`
- Fix: symlinks `alphazero-code`, `game-pars`

- [ ] **Step 1: Move old web + tools + docs**

```bash
mkdir -p tools
git mv web archive/web-old
git mv deploy_lan.py tools/deploy_lan.py
git mv CHAMPION_SETUP.md docs/ 2>/dev/null || true
```

- [ ] **Step 2: Remove the committed APK from the repo**

```bash
git rm --cached 6666.apk 2>/dev/null; rm -f 6666.apk; echo "6666.apk" >> .gitignore
```
(A 39 MB built artifact does not belong in git; rebuild from source when needed.)

- [ ] **Step 3: Re-point the symlinks to the new layout**

Run:
```bash
ls -l alphazero-code game-pars
```
If they point at moved paths, recreate them (e.g. `ln -sfn archive/old-impls/alphazero-code alphazero-code`). `game-pars -> ~/game-pars` is outside the repo and unaffected.

- [ ] **Step 4: Verify no references to old top-level paths remain**

Run:
```bash
grep -rIn "web-v2/\|rust-mcts/\|\./web/" --include=*.py --include=*.ts --include=*.tsx --include=*.sh --include=*.toml --include=*.md . | grep -v node_modules | grep -v "archive/" | grep -v docs/superpowers
```
Expected: no hits (or only intentional historical mentions in docs).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: archive old web, move deploy to tools/, drop APK from repo"
```

---

### Task 10: Final green gate + README

**Files:**
- Modify: `README.md` (new layout)

- [ ] **Step 1: Full Rust workspace build + test**

Run:
```bash
cargo build 2>&1 | tail -2 && cargo test -p togyzkumalaq-core -p togyzkumalaq-engine 2>&1 | grep "test result"
```
Expected: `Finished`; all `test result: ok`.

- [ ] **Step 2: Frontend gate**

Run:
```bash
npm --prefix product/web/frontend run typecheck && npm --prefix product/web/frontend test && npm --prefix product/web/frontend run build 2>&1 | tail -2
```
Expected: clean typecheck, `18 passed`, build OK.

- [ ] **Step 3: Backend gate**

Run:
```bash
cd product/web/backend && timeout 240 .venv/bin/pytest -q 2>&1 | tail -2; cd -
```
Expected: `75 passed`.

- [ ] **Step 4: Update `README.md` with the new top-level layout**

Replace the directory-structure section of `README.md` with the `core/ engine/ mcts/ product/ research/ models/ tools/ docs/ archive/` layout and a one-line description of each zone.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: README reflects new monorepo layout"
```

- [ ] **Step 6: Hand off for merge**

The branch is ready. Open a PR (or merge) — this is the single big-bang cutover. Do NOT merge unless Steps 1-3 above are all green.

---

## Self-Review

**Spec coverage:**
- §3 target structure → Tasks 2,5,6,7,8,9 (all zones created). ✓
- §4 move map → Tasks 5–9 cover every row (web-v2, rust-mcts, scripts, checkpoints, baseline, net, web, deploy, reports, apk, symlinks). ✓
- §5 workspace+core, parse_position dedup → Tasks 2,3,4. ✓
- §6 gitignore + runs convention → Task 1 (gitignore); runs/ dir created Task 7. Launcher is explicitly out-of-scope (§11). ✓
- §7 product decoupling → Task 8. ✓
- §8 path-fix checklist → REPO_ROOT (T6), hardcoded paths (T7), symlinks (T9), deploy/cargo (T5,T9), gitignore-first (T1). ✓
- §9 execution + green gate → Task 10. ✓
- §11 out-of-scope (script dedup, god-file split, texel/eval, launcher, LFS) → not tasked, correctly. ✓
- §12 acceptance → Task 10 gate + single board.rs (T3) + product from models (T8). ✓

**Placeholder scan:** Task 4 Steps 2/5/6 reference "canonical body from Step 1" — acceptable because the body is read live in Step 1 and the two copies are near-identical; the test in Step 3 pins correctness. No TBD/TODO/"add error handling" placeholders elsewhere.

**Type/name consistency:** crate `togyzkumalaq-core` / lib `togyzkumalaq_core`; alias `board`; package rename `rust-mcts`→`mcts`; `engine_path` → `models/engine/baseline`; `REPO_ROOT parents[4]`. Consistent across tasks.
