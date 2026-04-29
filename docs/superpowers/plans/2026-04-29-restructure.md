# Project Restructuring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the 9QumalaqV2 working tree per [the design](../specs/2026-04-29-restructure-design.md) — slim `engine/` and `rust-mcts/` to production assets, move all historical/experimental material into `archive/`, leave `web/` untouched.

**Architecture:** Pure file movement, no source edits. Use `git mv` for tracked files (preserves history) and plain `mv` for gitignored items (atomic rename on same ext4 filesystem). Verify after every batch by re-running `cargo check` on both crates and confirming production assets are still in place.

**Tech Stack:** Bash, `git mv`, `mv`, `cargo check`, Python `ast.parse` for `web/server.py` syntax check.

---

## Baseline (recorded 2026-04-29 before any changes)

- `cd engine && cargo check` → succeeds, 17 warnings, 0 errors
- `cd rust-mcts && cargo check` → succeeds, 10 warnings, 0 errors
- `web/server.py` references only `../engine/` (untouched), `web/games_log/` (untouched), `web/opening_book.json` (untouched) — no path into project root
- Filesystem: single ext4 device (`/dev/nvme0n1p6`), so `mv` rename is atomic and instant

After any task that modifies files, re-running the two `cargo check` commands MUST still succeed with 0 errors. If errors appear, stop and investigate before continuing.

---

## Task 1: Pre-flight — update `.gitignore` and create `archive/` skeleton

**Files:**
- Modify: `/home/nurlykhan/9QumalaqV2/.gitignore`
- Create: `/home/nurlykhan/9QumalaqV2/archive/` and subdirectories
- Create: `/home/nurlykhan/9QumalaqV2/archive/README.md`

- [ ] **Step 1: Confirm baseline build passes**

Run:
```bash
cd /home/nurlykhan/9QumalaqV2/engine && cargo check 2>&1 | tail -2
cd /home/nurlykhan/9QumalaqV2/rust-mcts && cargo check 2>&1 | tail -2
```
Expected: both end with `Finished \`dev\` profile ... target(s) in <time>` and 0 errors.

- [ ] **Step 2: Add `archive/` to `.gitignore`**

Append the following block to `/home/nurlykhan/9QumalaqV2/.gitignore` (use Edit tool, find the last `# Claude` block and add after it):

```
# Archive of historical/experimental material (out-of-tree, not committed)
archive/
```

- [ ] **Step 3: Create archive skeleton**

Run:
```bash
cd /home/nurlykhan/9QumalaqV2 && \
  mkdir -p archive/reports archive/misc archive/old-impls \
           archive/datasets archive/research \
           archive/engine-experiments archive/mcts-experiments
```

- [ ] **Step 4: Write `archive/README.md`**

Create `/home/nurlykhan/9QumalaqV2/archive/README.md` with content:

```markdown
# archive/

Historical and experimental material moved out of the active project tree on 2026-04-29 as part of the pre-rewrite restructuring (see [docs/superpowers/specs/2026-04-29-restructure-design.md](../docs/superpowers/specs/2026-04-29-restructure-design.md)).

This directory is gitignored. Nothing here is built or tested in CI; nothing here is imported by production code.

## Layout

- `reports/` — historical Markdown reports and analysis JSON from project root.
- `misc/` — one-off utility/analysis Python scripts, empty test-output directories, stray logs.
- `old-impls/alphazero-code/` — Python AlphaZero implementation, replaced by `rust-mcts/`.
- `datasets/` — large parsed/derived game datasets. Source PGNs can be re-fetched via `fetch_games.py` if needed.
- `research/` — research notes and the `research.zip` archive.
- `engine-experiments/` — NNUE weight variants, training scripts, source backups (`.bak/.improved/.new`), training logs.
- `mcts-experiments/` — old MCTS checkpoint dirs, superseded model files, training logs.

If you need anything here, copy it back into the active tree — don't symlink.
```

- [ ] **Step 5: Drop unused tracked binary `engine/opening_book.bin`**

This file is already deleted from the working tree but still tracked in git. The engine binary uses `opening_book.txt` (see `engine/src/main.rs:20`), not the `.bin` version. Stage the deletion:

```bash
cd /home/nurlykhan/9QumalaqV2 && \
  git rm engine/opening_book.bin
```
Expected: `rm 'engine/opening_book.bin'`. If the command fails saying the file doesn't exist, run `git rm --cached engine/opening_book.bin` instead.

- [ ] **Step 6: Stage `.gitignore` and commit**

Run:
```bash
cd /home/nurlykhan/9QumalaqV2 && \
  git add .gitignore && \
  git status
```
Expected: `.gitignore` modified, `engine/opening_book.bin` deleted, `archive/` NOT shown (gitignored).

```bash
cd /home/nurlykhan/9QumalaqV2 && \
  git commit -m "$(cat <<'EOF'
chore: prep restructure — gitignore archive/, drop unused opening_book.bin

Adds archive/ to .gitignore so historical material can be moved out of
the active tree without entering git. Drops engine/opening_book.bin
which is unused (engine reads opening_book.txt per src/main.rs:20).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Move root-level reports and analysis JSON to `archive/reports/`

**Files moved (all tracked):**
- `ALPHAZERO_INTEGRATION.md`, `ANALYSIS.md`, `FINAL_REPORT.md`, `FULL_REPORT.md`, `GAME_ANALYSIS.md`, `MCTS_EXPERIMENTS_REPORT.md`, `MCTS_PROJECT_PRESENTATION.md`, `NNUE_EXPERIMENTS.md`, `REPORT.md`
- `late_phase_analysis.json` (untracked), `engine_mistakes.json` (untracked)

- [ ] **Step 1: Verify files exist where the plan says**

Run:
```bash
cd /home/nurlykhan/9QumalaqV2 && \
  ls ALPHAZERO_INTEGRATION.md ANALYSIS.md FINAL_REPORT.md FULL_REPORT.md \
     GAME_ANALYSIS.md MCTS_EXPERIMENTS_REPORT.md MCTS_PROJECT_PRESENTATION.md \
     NNUE_EXPERIMENTS.md REPORT.md late_phase_analysis.json engine_mistakes.json
```
Expected: all 11 files listed without "No such file" errors.

- [ ] **Step 2: `git mv` the tracked .md files**

Run:
```bash
cd /home/nurlykhan/9QumalaqV2 && \
  git mv ALPHAZERO_INTEGRATION.md ANALYSIS.md FINAL_REPORT.md FULL_REPORT.md \
         GAME_ANALYSIS.md MCTS_EXPERIMENTS_REPORT.md MCTS_PROJECT_PRESENTATION.md \
         NNUE_EXPERIMENTS.md REPORT.md \
         archive/reports/
```

- [ ] **Step 3: `mv` the untracked JSONs**

Run:
```bash
cd /home/nurlykhan/9QumalaqV2 && \
  mv late_phase_analysis.json engine_mistakes.json archive/reports/
```

- [ ] **Step 4: Verify the moves**

Run:
```bash
cd /home/nurlykhan/9QumalaqV2 && \
  ls archive/reports/ | wc -l && \
  ls *.md 2>/dev/null | grep -E '^(ALPHAZERO|ANALYSIS|FINAL_REPORT|FULL_REPORT|GAME_ANALYSIS|MCTS_|NNUE_|REPORT)' | wc -l
```
Expected: first count `>= 11`, second count `0`.

- [ ] **Step 5: Re-verify both engines still build**

Run:
```bash
cd /home/nurlykhan/9QumalaqV2/engine && cargo check 2>&1 | tail -1
cd /home/nurlykhan/9QumalaqV2/rust-mcts && cargo check 2>&1 | tail -1
```
Expected: both end with `Finished` line, 0 errors. (No source change, but cheap insurance.)

---

## Task 3: Move root-level utility scripts and stray files to `archive/misc/`

**Files moved (all tracked unless noted):**
- `analyze_deep.py`, `analyze_games.py`, `analyze_late_phase.py` (untracked), `parse_games.py`, `find_engine_mistakes.py` (untracked), `export_positions.py`, `validate_perft.py`, `deploy.py`, `deploy_web_only.py` (untracked), `setup_tunnel.py` (untracked), `fetch_games.py` (untracked)
- `match_52ft_s123.log` (untracked), `match_endgame_v3_200g.log` (untracked) — small log files
- `combo_asp35_lmr30/` (empty, untracked), `lmr18_test/` (empty, untracked), `report/` (untracked — gitignored)
- `.DS_Store` (untracked — gitignored)

- [ ] **Step 1: Verify items present**

Run:
```bash
cd /home/nurlykhan/9QumalaqV2 && \
  ls analyze_deep.py analyze_games.py analyze_late_phase.py parse_games.py \
     find_engine_mistakes.py export_positions.py validate_perft.py deploy.py \
     deploy_web_only.py setup_tunnel.py fetch_games.py \
     match_52ft_s123.log match_endgame_v3_200g.log .DS_Store && \
  ls -d combo_asp35_lmr30 lmr18_test report
```
Expected: every item listed without errors.

- [ ] **Step 2: `git mv` the tracked items**

Run:
```bash
cd /home/nurlykhan/9QumalaqV2 && \
  git mv analyze_deep.py analyze_games.py parse_games.py export_positions.py \
         validate_perft.py deploy.py \
         archive/misc/
```

- [ ] **Step 3: `mv` the untracked items**

Run:
```bash
cd /home/nurlykhan/9QumalaqV2 && \
  mv analyze_late_phase.py find_engine_mistakes.py deploy_web_only.py \
     setup_tunnel.py fetch_games.py \
     match_52ft_s123.log match_endgame_v3_200g.log .DS_Store \
     archive/misc/ && \
  mv combo_asp35_lmr30 lmr18_test report archive/misc/
```

- [ ] **Step 4: Verify moves**

Run:
```bash
cd /home/nurlykhan/9QumalaqV2 && \
  ls archive/misc/ | head -30 && \
  ls *.py 2>/dev/null | grep -vE '^(deploy_lan)\.py$' && \
  ls -d combo_asp35_lmr30 lmr18_test report 2>&1 | grep -c "No such"
```
Expected: third count `3` (all three dirs gone from root). Second command should print nothing (only `deploy_lan.py` survives in root).

- [ ] **Step 5: Re-verify build**

Run:
```bash
cd /home/nurlykhan/9QumalaqV2/engine && cargo check 2>&1 | tail -1
cd /home/nurlykhan/9QumalaqV2/rust-mcts && cargo check 2>&1 | tail -1
```
Expected: both `Finished`, 0 errors.

---

## Task 4: Move large root-level directories to `archive/`

**Items moved (all untracked / gitignored):**
- `alphazero-code/` (617 MB) → `archive/old-impls/alphazero-code/`
- `game-pars/` (2.7 GB) → `archive/datasets/game-pars/`
- `mergeData/` (737 MB) → `archive/datasets/mergeData/`
- `parsed_games/` (11 MB) → `archive/datasets/parsed_games/`
- `gameNew2/` (3.5 MB) → `archive/datasets/gameNew2/`
- `expertsRV/` (816 KB) → `archive/datasets/expertsRV/`
- `datagen_pack/` (168 KB) → `archive/datasets/datagen_pack/`
- `server_games_log/` (404 KB) → `archive/datasets/server_games_log/`
- `research/` + `research.zip` → `archive/research/`

- [ ] **Step 1: Verify items present and on same filesystem**

Run:
```bash
cd /home/nurlykhan/9QumalaqV2 && \
  ls -d alphazero-code game-pars mergeData parsed_games gameNew2 \
        expertsRV datagen_pack server_games_log research && \
  ls research.zip && \
  df -T . archive/ | tail -2
```
Expected: all 9 dirs + 1 file listed; `df` shows the same filesystem on both rows (so `mv` is rename, not copy).

- [ ] **Step 2: Move large datasets**

Run:
```bash
cd /home/nurlykhan/9QumalaqV2 && \
  mv game-pars mergeData parsed_games gameNew2 expertsRV datagen_pack \
     server_games_log \
     archive/datasets/
```

- [ ] **Step 3: Move alphazero-code**

Run:
```bash
cd /home/nurlykhan/9QumalaqV2 && \
  mv alphazero-code archive/old-impls/
```

- [ ] **Step 4: Move research**

Run:
```bash
cd /home/nurlykhan/9QumalaqV2 && \
  mv research research.zip archive/research/
```

- [ ] **Step 5: Verify moves**

Run:
```bash
cd /home/nurlykhan/9QumalaqV2 && \
  ls -d alphazero-code game-pars mergeData parsed_games gameNew2 \
        expertsRV datagen_pack server_games_log research 2>&1 | grep -c "No such" && \
  du -sh archive/datasets archive/old-impls archive/research
```
Expected: first count `9` (all gone from root). `du` shows the data landed in the right place (e.g., `~3.5 GB` total in datasets).

- [ ] **Step 6: Re-verify build**

Run:
```bash
cd /home/nurlykhan/9QumalaqV2/engine && cargo check 2>&1 | tail -1
cd /home/nurlykhan/9QumalaqV2/rust-mcts && cargo check 2>&1 | tail -1
```
Expected: both `Finished`, 0 errors.

---

## Task 5: Slim `engine/` — move source backups and weight variants

**Items moved to `archive/engine-experiments/`:**
- `engine/src/search.rs.bak`, `engine/src/search.rs.improved`, `engine/src/search.rs.improved2`, `engine/src/search.rs.new` (all four UNTRACKED — confirmed via `git ls-files`)
- All `engine/nnue_weights_*.{pt,bin}` and other weight variants (untracked, gitignored) — but **NOT** `engine/nnue_weights.bin`
- `engine/nnue_weights.json` (TRACKED)

- [ ] **Step 1: Verify the production weight file is NOT included in any glob**

Run:
```bash
cd /home/nurlykhan/9QumalaqV2/engine && \
  ls nnue_weights_*.bin 2>&1 | grep -c '^nnue_weights\.bin$' && \
  ls nnue_weights.bin
```
Expected: first count `0` (the bare `nnue_weights.bin` does NOT match `nnue_weights_*.bin`); second command prints `nnue_weights.bin` confirming production file is intact.

- [ ] **Step 2: Move source backup files (all untracked)**

Run:
```bash
cd /home/nurlykhan/9QumalaqV2 && \
  ls engine/src/search.rs.bak engine/src/search.rs.improved \
     engine/src/search.rs.improved2 engine/src/search.rs.new
```
Expected: all four listed.

```bash
cd /home/nurlykhan/9QumalaqV2 && \
  mv engine/src/search.rs.bak engine/src/search.rs.improved \
     engine/src/search.rs.improved2 engine/src/search.rs.new \
     archive/engine-experiments/
```

- [ ] **Step 3: Move all NNUE weight variants (untracked)**

Run:
```bash
cd /home/nurlykhan/9QumalaqV2/engine && \
  ls nnue_weights_*.pt nnue_weights_*.bin nnue_*_best.pt nnue_v*.pt \
     nnue_combined_*.pt nnue_dropout_best.pt nnue_final.pt nnue_hybrid_*.pt \
     nnue_hce_*.pt nnue_human_*.pt nnue_k*.pt nnue_lam*.pt nnue_gen*.pt 2>/dev/null \
  | wc -l
```
Expected: a count > 100 (these are all the experimental weights).

Now move them:

```bash
cd /home/nurlykhan/9QumalaqV2/engine && \
  for pat in 'nnue_weights_*.pt' 'nnue_weights_*.bin' 'nnue_*_best.pt' \
             'nnue_v*.pt' 'nnue_combined_*.pt' 'nnue_dropout_best.pt' \
             'nnue_final.pt' 'nnue_hybrid_*.pt' 'nnue_hce_*.pt' \
             'nnue_human_*.pt' 'nnue_k*.pt' 'nnue_lam*.pt' 'nnue_gen*.pt'; do
    for f in $pat; do
      [ -f "$f" ] && mv "$f" ../archive/engine-experiments/
    done
  done && \
  ls nnue_weights.bin
```
Expected: `nnue_weights.bin` still listed (production weights NOT moved by these globs).

- [ ] **Step 4: Move tracked `nnue_weights.json`**

Run:
```bash
cd /home/nurlykhan/9QumalaqV2 && \
  git mv engine/nnue_weights.json archive/engine-experiments/
```

- [ ] **Step 5: Verify `nnue_weights.bin` (production) is still in place**

Run:
```bash
ls -la /home/nurlykhan/9QumalaqV2/engine/nnue_weights.bin
```
Expected: file present, ~37 KB.

- [ ] **Step 6: Build verification**

Run:
```bash
cd /home/nurlykhan/9QumalaqV2/engine && cargo check 2>&1 | tail -1
```
Expected: `Finished` line, 0 errors.

---

## Task 6: Slim `engine/` — move training scripts, logs, datagen artefacts

**Items moved to `archive/engine-experiments/`:**
- Tracked Python: `convert_human_games.py`, `convert_master_games.py`, `experiment_52_finetune.py`, `experiment_pured14.py`, `extract_expert_positions.py`, `finetune_nnue.py`, `finetune_transfer.py`, `gen_endgame_data.py`, `match_engines.py`, `match_search.py`, `merge_data.py`, `prepare_v8_data.py`, `prepare_v9_data.py`, `prepare_v9b_data.py`, `selfplay_loop.py`, `test_two_weights.py`, `train_custom_k.py`, `train_dropout.py`, `train_endgame.py`, `train_gen8.py`, `train_gen8_lr.py`, `train_multiseed.py`, `train_nnue.py`, `train_nnue_v2.py`, `transfer_58feat.py`, `pipeline.py`, `pipeline_results.json`, `run_datagen.bat`, `run_training.sh`
- Tracked logs: keep — but logs are gitignored per `.gitignore`, so they're untracked
- Untracked: all `*.log`, `local_thread_*.bin` (24 zero-byte files), `gen6_newsearch_training_data.bin`, `gen7_training_data.bin`, `positions.txt`, `__pycache__/`, `cd/`, `cp/`, `.DS_Store`

**Keep in `engine/`:** `gen_opening_book.py`, `gen_opening_book_v2.py` (still needed to regenerate the opening book if required).

- [ ] **Step 1: Move tracked Python and shell scripts**

Run:
```bash
cd /home/nurlykhan/9QumalaqV2 && \
  git mv engine/convert_human_games.py engine/convert_master_games.py \
         engine/experiment_52_finetune.py engine/experiment_pured14.py \
         engine/extract_expert_positions.py engine/finetune_nnue.py \
         engine/finetune_transfer.py engine/gen_endgame_data.py \
         engine/match_engines.py engine/match_search.py engine/merge_data.py \
         engine/prepare_v8_data.py engine/prepare_v9_data.py \
         engine/prepare_v9b_data.py engine/selfplay_loop.py \
         engine/test_two_weights.py engine/train_custom_k.py \
         engine/train_dropout.py engine/train_endgame.py engine/train_gen8.py \
         engine/train_gen8_lr.py engine/train_multiseed.py \
         engine/train_nnue.py engine/train_nnue_v2.py \
         engine/transfer_58feat.py engine/pipeline.py \
         engine/pipeline_results.json engine/run_datagen.bat \
         engine/run_training.sh \
         archive/engine-experiments/
```

If any of these has been moved by a previous step or doesn't exist, drop it from the command and re-run. Do NOT `git mv engine/gen_opening_book*.py` — those stay.

- [ ] **Step 2: Move untracked logs and datagen artefacts**

Run:
```bash
cd /home/nurlykhan/9QumalaqV2/engine && \
  for f in *.log; do
    [ -f "$f" ] && mv "$f" ../archive/engine-experiments/
  done && \
  for f in local_thread_*.bin; do
    [ -f "$f" ] && mv "$f" ../archive/engine-experiments/
  done && \
  for f in gen6_newsearch_training_data.bin gen7_training_data.bin positions.txt .DS_Store; do
    [ -f "$f" ] && mv "$f" ../archive/engine-experiments/
  done && \
  for d in __pycache__ cd cp; do
    [ -d "$d" ] && mv "$d" ../archive/engine-experiments/
  done
```

- [ ] **Step 3: Verify the surviving `engine/` contents match the spec**

Run:
```bash
cd /home/nurlykhan/9QumalaqV2/engine && \
  ls --color=never | sort
```
Expected output (exactly this set, possibly plus `target/`):
```
.cargo
Cargo.lock
Cargo.toml
egtb.bin
gen_opening_book.py
gen_opening_book_v2.py
nnue_weights.bin
opening_book.txt
src
target
```

If any unexpected file appears, decide per the design (move to `archive/engine-experiments/` if experimental, keep if production-relevant) and document the decision in the commit message later.

- [ ] **Step 4: Verify `src/` is clean (no `.bak`/`.improved`/`.new`)**

Run:
```bash
ls /home/nurlykhan/9QumalaqV2/engine/src/ | sort
```
Expected:
```
board.rs
book.rs
datagen.rs
egtb.rs
eval.rs
main.rs
nnue.rs
search.rs
texel.rs
tt.rs
zobrist.rs
```

- [ ] **Step 5: Build verification**

Run:
```bash
cd /home/nurlykhan/9QumalaqV2/engine && cargo check 2>&1 | tail -1
```
Expected: `Finished` line, 0 errors. (Same baseline 17 warnings expected.)

---

## Task 7: Slim `rust-mcts/` — move old checkpoints, models, logs

**Items moved to `archive/mcts-experiments/`:**
- Old checkpoint dirs (untracked): `checkpoints/`, `checkpoints_2m/`, `checkpoints_2m_v2/`, `checkpoints_distill/`, `checkpoints_max/`
- Old models (untracked): `model.pt`, `model.onnx`, `model.onnx.data`, `model.trt`, `model_v2.pt`, `model_v2.onnx`, `model_v2.onnx.data`
- Logs (untracked): `training.log`, `training_stdout.log`, `game_collection.log`
- Empty placeholders (untracked): `master_games.bin`, `replay_buffer.bin`
- Datasets (untracked): `master_games/`, `distill_data/`

**Keep in `rust-mcts/`:** `Cargo.toml`, `Cargo.lock`, `src/`, `scripts/`, `checkpoints_v3/`, `model_2m.{onnx,onnx.data,trt}`, `target/`.

- [ ] **Step 1: Verify items present**

Run:
```bash
cd /home/nurlykhan/9QumalaqV2/rust-mcts && \
  ls -d checkpoints checkpoints_2m checkpoints_2m_v2 checkpoints_distill \
        checkpoints_max master_games distill_data 2>&1 | head && \
  ls model.pt model.onnx model.onnx.data model.trt \
     model_v2.pt model_v2.onnx model_v2.onnx.data \
     training.log training_stdout.log game_collection.log \
     master_games.bin replay_buffer.bin
```
Expected: every item listed.

- [ ] **Step 2: Move old checkpoint directories**

Run:
```bash
cd /home/nurlykhan/9QumalaqV2/rust-mcts && \
  for d in checkpoints checkpoints_2m checkpoints_2m_v2 checkpoints_distill checkpoints_max; do
    [ -d "$d" ] && mv "$d" ../archive/mcts-experiments/
  done
```

- [ ] **Step 3: Move old model files**

Run:
```bash
cd /home/nurlykhan/9QumalaqV2/rust-mcts && \
  for f in model.pt model.onnx model.onnx.data model.trt \
           model_v2.pt model_v2.onnx model_v2.onnx.data; do
    [ -f "$f" ] && mv "$f" ../archive/mcts-experiments/
  done
```

- [ ] **Step 4: Move logs and empty placeholders and datasets**

Run:
```bash
cd /home/nurlykhan/9QumalaqV2/rust-mcts && \
  for f in training.log training_stdout.log game_collection.log \
           master_games.bin replay_buffer.bin; do
    [ -f "$f" ] && mv "$f" ../archive/mcts-experiments/
  done && \
  for d in master_games distill_data; do
    [ -d "$d" ] && mv "$d" ../archive/mcts-experiments/
  done
```

- [ ] **Step 5: Confirm production model file `model_2m.trt` is still in place**

Run:
```bash
ls -la /home/nurlykhan/9QumalaqV2/rust-mcts/model_2m.trt \
       /home/nurlykhan/9QumalaqV2/rust-mcts/model_2m.onnx \
       /home/nurlykhan/9QumalaqV2/rust-mcts/model_2m.onnx.data && \
  ls -d /home/nurlykhan/9QumalaqV2/rust-mcts/checkpoints_v3
```
Expected: all four exist; `checkpoints_v3/` is non-empty.

- [ ] **Step 6: Verify `rust-mcts/` content**

Run:
```bash
ls /home/nurlykhan/9QumalaqV2/rust-mcts/ | sort
```
Expected (allowing `target/`):
```
Cargo.lock
Cargo.toml
checkpoints_v3
model_2m.onnx
model_2m.onnx.data
model_2m.trt
scripts
src
target
```

- [ ] **Step 7: Build verification**

Run:
```bash
cd /home/nurlykhan/9QumalaqV2/rust-mcts && cargo check 2>&1 | tail -1
```
Expected: `Finished`, 0 errors.

---

## Task 8: Audit `web/server.py` and confirm web stack still works

**Goal:** Make sure none of the moves broke `web/server.py` (it should be self-contained per pre-flight audit, but verify).

- [ ] **Step 1: Syntax-check `web/server.py`**

Run:
```bash
python3 -c "import ast; ast.parse(open('/home/nurlykhan/9QumalaqV2/web/server.py').read()); print('OK')"
```
Expected: `OK`.

- [ ] **Step 2: Confirm `web/` references resolve to existing paths**

Run:
```bash
ls /home/nurlykhan/9QumalaqV2/engine/target/release/togyzkumalaq-engine 2>&1 || echo "binary not built (OK if release not yet compiled)"
ls /home/nurlykhan/9QumalaqV2/web/opening_book.json
ls -d /home/nurlykhan/9QumalaqV2/web/games_log
```
Expected: `opening_book.json` and `games_log/` listed; `togyzkumalaq-engine` may or may not exist (release build is separate; OK either way for syntax/path correctness).

- [ ] **Step 3: Confirm `generate_book.py` references the engine correctly**

Run:
```bash
grep -nE 'open\(|path|\.\./' /home/nurlykhan/9QumalaqV2/web/generate_book.py | head -15
```
Inspect output — confirm no path points to a directory we moved (none should, per pre-flight audit). If any moved path appears, stop and re-evaluate before continuing.

---

## Task 9: Create root `README.md` and final verification

**Files:**
- Create: `/home/nurlykhan/9QumalaqV2/README.md`

- [ ] **Step 1: Write `README.md`**

Create `/home/nurlykhan/9QumalaqV2/README.md` with content:

```markdown
# 9QumalaqV2 — Togyzkumalak engine & training infrastructure

This repository hosts two independent production engines for the Togyzkumalak board game and a web interface that serves them.

## Layout

| Path | Purpose |
|---|---|
| [`engine/`](engine/) | NNUE search engine (Rust). Production binary deployed to LAN server. |
| [`rust-mcts/`](rust-mcts/) | MCTS training and evaluation infrastructure (Rust + Python). |
| [`web/`](web/) | Flask web interface and opening-book server. |
| [`docs/`](docs/) | Design specs and implementation plans (under `docs/superpowers/`). |
| [`archive/`](archive/) | Historical and experimental material — gitignored. See `archive/README.md` for layout. |
| `deploy_lan.py` | Deployment script for the LAN server. |

## Production assets

- **NNUE weights:** [`engine/nnue_weights.bin`](engine/nnue_weights.bin)
- **Opening book:** [`engine/opening_book.txt`](engine/opening_book.txt), [`web/opening_book.json`](web/opening_book.json)
- **Endgame tablebase:** [`engine/egtb.bin`](engine/egtb.bin)
- **MCTS production model:** [`rust-mcts/model_2m.trt`](rust-mcts/model_2m.trt) (+ `.onnx` source)
- **Best MCTS checkpoint:** [`rust-mcts/checkpoints_v3/`](rust-mcts/checkpoints_v3/)

## Build

```bash
cd engine    && cargo build --release
cd rust-mcts && cargo build --release
```

## History

Pre-2026-04-29 reports, NNUE weight experiments, and old MCTS checkpoints have been moved to `archive/`. The full project history is preserved in git.
```

- [ ] **Step 2: Stage the new README**

Run:
```bash
cd /home/nurlykhan/9QumalaqV2 && \
  git add README.md
```

- [ ] **Step 3: Final verification — production assets**

Run:
```bash
cd /home/nurlykhan/9QumalaqV2 && \
  for f in engine/nnue_weights.bin engine/opening_book.txt \
           engine/egtb.bin rust-mcts/model_2m.trt rust-mcts/model_2m.onnx \
           rust-mcts/model_2m.onnx.data web/index.html web/server.py \
           web/opening_book.json; do
    if [ -e "$f" ]; then echo "OK $f"; else echo "MISSING $f"; fi
  done && \
  [ -d engine/src ] && echo "OK engine/src" && \
  [ -d rust-mcts/src ] && echo "OK rust-mcts/src" && \
  [ -d rust-mcts/checkpoints_v3 ] && echo "OK rust-mcts/checkpoints_v3"
```
Expected: every line starts with `OK`, no `MISSING`.

- [ ] **Step 4: Final verification — top-level layout matches spec**

Run:
```bash
cd /home/nurlykhan/9QumalaqV2 && ls --color=never | sort
```
Expected (besides hidden dirs `.git`, `.gitignore`, `.claude`):
```
README.md
archive
deploy_lan.py
docs
engine
rust-mcts
web
```

If any other entry appears, decide whether it belongs in active tree or archive, and move accordingly before committing.

- [ ] **Step 5: Final build verification**

Run:
```bash
cd /home/nurlykhan/9QumalaqV2/engine && cargo check 2>&1 | tail -2
cd /home/nurlykhan/9QumalaqV2/rust-mcts && cargo check 2>&1 | tail -2
```
Expected: both `Finished`, 0 errors. Warning counts should match baseline (17 + 10) since no source was edited.

- [ ] **Step 6: `git status` review**

Run:
```bash
cd /home/nurlykhan/9QumalaqV2 && \
  git status && \
  echo "---" && \
  git diff --cached --stat | tail -20
```
Expected: only renames (`R `) and the new `README.md` and modified `.gitignore` (already committed in Task 1, so not in staged diff). Untracked: nothing in project root that shouldn't be there. `archive/` should NOT appear in `git status` (it's gitignored).

If anything unexpected appears (untracked files in root other than what was intended; unstaged changes to source files), pause and investigate before committing.

---

## Task 10: Commit

- [ ] **Step 1: Stage all renames**

The `git mv` calls already staged renames. Confirm:

```bash
cd /home/nurlykhan/9QumalaqV2 && \
  git diff --cached --name-status | head -40 && \
  echo "..." && \
  git diff --cached --name-status | wc -l
```
Expected: many `R` (rename) entries plus one `A README.md`. No `D` (delete) entries — every move should be a rename.

- [ ] **Step 2: Commit**

Run:
```bash
cd /home/nurlykhan/9QumalaqV2 && \
  git commit -m "$(cat <<'EOF'
chore: restructure tree — slim engine/ and rust-mcts/, archive history

Move all historical reports, experimental NNUE weights, old MCTS
checkpoints, large parsed datasets, and the abandoned alphazero-code
Python implementation into a new archive/ tree (gitignored). Slim the
production engine/ and rust-mcts/ directories down to only the files
needed to build and run the deployed binaries.

No source code is edited. Both crates still build (cargo check, 0
errors, baseline warning counts unchanged). web/ is untouched and is
the subject of a separate upcoming rewrite.

Spec: docs/superpowers/specs/2026-04-29-restructure-design.md
Plan: docs/superpowers/plans/2026-04-29-restructure.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)" && \
  git status
```
Expected: commit succeeds, working tree clean (apart from `archive/` which is gitignored).

- [ ] **Step 3: Sanity check — git log**

Run:
```bash
cd /home/nurlykhan/9QumalaqV2 && git log --oneline -3
```
Expected: top entry is the restructure commit; second is the gitignore commit from Task 1.

---

## Done

After Task 10, the active tree contains only production-relevant files. `engine/`, `rust-mcts/`, and `web/` are ready for the upcoming web rewrite. `archive/` holds everything else for reference; it is not committed to git but lives on disk.

The web rewrite is **out of scope** of this plan — it's a separate spec/plan.
