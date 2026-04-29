# Project Restructuring Design

**Date:** 2026-04-29
**Author:** brainstorming session (Claude + nurkal022)
**Status:** Approved, awaiting spec review

## Goal

Prepare the 9QumalaqV2 project for a major upcoming web-interface rewrite by reorganizing the working tree: keep production assets cleanly accessible, move historical/experimental material into a single `archive/` tree, and remove no files permanently.

## Non-goals

- No edits to source code (`*.rs`, `*.py`) — strictly file movement.
- No changes to `web/` content — it stays in production until the rewrite begins.
- No changes to deployment infrastructure or the running server.
- No git history rewrites — moves preserve history via `git mv`.

## Constraints

1. **Production must keep working** after the restructure:
   - `engine/` (NNUE) compiles and runs — currently deployed to LAN server (10.0.34.22).
   - `rust-mcts/` compiles and runs — used for training/eval.
   - `web/` is untouched (served by `web/server.py`).
2. **Nothing is deleted permanently** — everything that leaves the active tree goes into `archive/`. Per user decision, even large datasets (4 GB+) are archived rather than dropped.
3. **Both engines stay** — the user explicitly requested keeping NNUE *and* MCTS as production entities, separate from each other.

## Final Top-Level Layout

```
9QumalaqV2/
├── engine/        # PROD: NNUE engine (slim — only files needed to build/run)
├── rust-mcts/     # PROD: MCTS training/eval (slim — only current best + scripts)
├── web/           # PROD: current UI — UNCHANGED in this pass
├── archive/       # NEW: all historical/experimental material
├── docs/          # NEW: project docs (this design lives here)
├── deploy_lan.py  # current deploy script
├── README.md      # NEW: single navigation README
├── .git/, .gitignore, .claude/
```

## File Disposition

### Keep in `engine/`

- `Cargo.toml`, `Cargo.lock`, `.cargo/`
- `src/{board,book,datagen,egtb,eval,main,nnue,search,texel,tt,zobrist}.rs`
- `nnue_weights.bin` — production weights
- `opening_book.txt` — production book (engine reads .txt per `src/main.rs:20`; the tracked but unused `opening_book.bin` is dropped during restructure)
- `egtb.bin` — endgame tablebase (~65 MB) needed by binary
- `gen_opening_book.py`, `gen_opening_book_v2.py` — kept (regenerates book)
- `target/` — gitignored Rust build artefact

### Move from `engine/` to `archive/engine-experiments/`

- Source backups: `src/search.rs.bak`, `src/search.rs.improved`, `src/search.rs.improved2`, `src/search.rs.new`
- All NNUE weight variants (~150 files): `nnue_weights_*.pt`, `nnue_weights_*.bin` (except production `nnue_weights.bin`!), `nnue_*_best.pt`, `nnue_v*.pt`, `nnue_combined_*.pt`, `nnue_dropout_best.pt`, `nnue_final.pt`, `nnue_hybrid_*.pt`, `nnue_human_*.pt`, `nnue_hce_*.pt`, `nnue_k*.pt`, `nnue_lam*.pt`, `nnue_gen*.pt`
- `nnue_weights.json`
- Training/experiment scripts: `train_*.py`, `prepare_v*.py`, `experiment_*.py`, `finetune_*.py`, `transfer_58feat.py`, `selfplay_loop.py`, `pipeline*.py`, `pipeline*.json`, `merge_data.py`, `convert_*.py`, `extract_expert_positions.py`, `gen_endgame_data.py`, `match_engines.py`, `match_search.py`, `test_two_weights.py`
- Logs: all `*.log`
- Empty `local_thread_*.bin` (24 zero-byte files)
- Training data: `gen6_newsearch_training_data.bin`, `gen7_training_data.bin`, `positions.txt`
- Misc: `__pycache__/`, `cd/`, `cp/`, `run_datagen.bat`, `run_training.sh`, `.DS_Store`

### Keep in `rust-mcts/`

- `Cargo.toml`, `Cargo.lock`
- `src/` (all `.rs` files)
- `scripts/` (all `.py` files — `train_*.py`, `eval_*.py`, `export_onnx.py`, `pipeline_max.sh`, `run_*.sh`, `collect_master_games.py`)
- `checkpoints_v3/` — best model (iter500, p_loss=1.09) per project memory
- `model_2m.onnx`, `model_2m.onnx.data`, `model_2m.trt` — production model
- `target/` — gitignored

### Move from `rust-mcts/` to `archive/mcts-experiments/`

- Old checkpoint dirs: `checkpoints/`, `checkpoints_2m/`, `checkpoints_2m_v2/`, `checkpoints_distill/`, `checkpoints_max/`
- Old models: `model.pt`, `model.onnx`, `model.onnx.data`, `model.trt`, `model_v2.pt`, `model_v2.onnx`, `model_v2.onnx.data`
- Logs: `training.log`, `training_stdout.log`, `game_collection.log`
- Empty placeholders: `master_games.bin`, `replay_buffer.bin`
- Datasets: `master_games/`, `distill_data/`

### Move from project root

- **To `archive/reports/`:** `ALPHAZERO_INTEGRATION.md`, `ANALYSIS.md`, `FINAL_REPORT.md`, `FULL_REPORT.md`, `GAME_ANALYSIS.md`, `MCTS_EXPERIMENTS_REPORT.md`, `MCTS_PROJECT_PRESENTATION.md`, `NNUE_EXPERIMENTS.md`, `REPORT.md`, `late_phase_analysis.json`, `engine_mistakes.json`
- **To `archive/misc/`:** `analyze_deep.py`, `analyze_games.py`, `analyze_late_phase.py`, `parse_games.py`, `find_engine_mistakes.py`, `export_positions.py`, `validate_perft.py`, `deploy.py`, `deploy_web_only.py`, `setup_tunnel.py`, `fetch_games.py`, `combo_asp35_lmr30/`, `lmr18_test/`, `report/`, `match_52ft_s123.log`, `match_endgame_v3_200g.log`, `.DS_Store`
- **To `archive/old-impls/`:** `alphazero-code/` (617 MB)
- **To `archive/datasets/`:** `game-pars/` (2.7 GB), `mergeData/` (737 MB), `parsed_games/`, `gameNew2/`, `expertsRV/`, `datagen_pack/`, `server_games_log/`
- **To `archive/research/`:** `research/`, `research.zip`

### Stay in project root

- `engine/`, `rust-mcts/`, `web/`, `.git/`, `.gitignore`, `.claude/`, `docs/`
- `deploy_lan.py` — current deployment script
- `README.md` — to be created (single navigation document)

## Execution Strategy

1. **Tooling:**
   - Tracked files: `git mv` (preserves history).
   - Untracked files (everything in `.gitignore` — checkpoints, weights, large datasets): plain `mv`.
2. **Order:** create destination dirs first → move files in batches grouped by source dir → verify build → commit once.
3. **Atomicity:** single commit at the end. If anything breaks mid-way (e.g., a build fails), `git reset --hard HEAD` reverts staged moves; manually restore unstaged moves from `archive/` back to original locations.
4. **`.gitignore` update:** add `archive/` so the multi-GB tree never enters git.
5. **No source edits:** if any reference to a moved file is found in source code (e.g., a hardcoded path), pause and reconsider — don't silently fix.

## Verification

After the moves and before commit:

1. `cd engine && cargo check` — must succeed.
2. `cd rust-mcts && cargo check` — must succeed.
3. `python3 -c "import ast; ast.parse(open('web/server.py').read())"` — syntax check on the web server.
4. Production assets present at expected paths:
   - `engine/nnue_weights.bin`
   - `engine/opening_book.txt`
   - `engine/egtb.bin`
   - `rust-mcts/model_2m.trt`, `rust-mcts/model_2m.onnx`
   - `rust-mcts/checkpoints_v3/` non-empty
   - `web/index.html`, `web/server.py`
5. `git status` — only renames; no untracked source files left in old locations.
6. `archive/README.md` exists and lists the archive layout.

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Hardcoded path inside engine binary references moved file | Run `cargo check` immediately after moves. If breaks, inspect — may need to keep a file in place. |
| `web/server.py` references moved file (e.g., opening book at old path) | Audit `web/server.py` for paths to root-level files before moving. |
| Large `mv` of `game-pars/` (2.7 GB) takes time / fails mid-way | Use `mv` (rename within same filesystem is atomic, near-instant). Confirm `archive/` is on same FS as source. |
| Missing items not enumerated above | After moves, `ls -la` on root and `engine/`, `rust-mcts/` to confirm only intended items remain. |
| Accidentally moving `nnue_weights.bin` (production) instead of variants | Use explicit globs that exclude it (e.g., `nnue_weights_*.bin` matches variants only); verify with `ls` before `mv`. |

## Out of Scope (future work)

- Web rewrite — separate spec/plan after this restructuring lands.
- Merging `engine/` and `rust-mcts/` under a unified workspace — user explicitly chose to keep them separate.
- Compressing `archive/` to a tarball — can be done later if disk pressure arises.
- Documentation refresh (consolidating the 9 archived reports into a single technical history) — separate task.
