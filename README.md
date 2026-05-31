# 9QumalaqV2 — Togyzkumalak engine, training & product

A monorepo organized into clean zones: shared game rules, two Rust engines, the
player-facing product, the research/experiment pipeline, and blessed champion
artifacts.

## Layout

| Path | Purpose |
|---|---|
| [`core/`](core/) | `togyzkumalaq-core` crate — the single source of truth for game rules (board, moves, position parsing). Both engines depend on it. |
| [`engine/`](engine/) | Classical NNUE/alpha-beta search engine (Rust). |
| [`mcts/`](mcts/) | AlphaZero-style MCTS engine (Rust). |
| [`product/web/`](product/web/) | The product: FastAPI backend + React/Vite frontend served to players. |
| [`research/`](research/) | Experiment **code**: `training/`, `data/`, `eval/`, `configs/`. Each run lands in `research/runs/<date-name>/` (gitignored; only `config.yaml` + `summary.md` tracked). |
| [`models/`](models/) | Blessed champions the product depends on — `engine/baseline` (tracked) and promoted `nets/`. See [`models/README.md`](models/README.md). |
| [`tools/`](tools/) | Deploy / ops scripts. |
| [`docs/`](docs/) | Design specs & implementation plans (`docs/superpowers/`). |
| [`archive/`](archive/) | Frozen historical material — gitignored. |

The three Rust crates form one Cargo workspace (root `Cargo.toml`); the release
profile (LTO) is set at the workspace root.

## Build & test

```bash
# Rust workspace (core + engine + mcts)
cargo build --release          # build all crates
cargo test  -p togyzkumalaq-core   # rules tests (board, make/unmake, parse)
cargo test  -p togyzkumalaq-engine # engine tests
# mcts links ONNX Runtime at runtime; set LD_LIBRARY_PATH to the nvidia pip libs to run it.

# Product — backend
cd product/web/backend && .venv/bin/python -m pytest -q

# Product — frontend
npm --prefix product/web/frontend run typecheck
npm --prefix product/web/frontend test
npm --prefix product/web/frontend run build
```

## Product engine

The backend serves the engine at `models/engine/baseline` (override via the
`ENGINE_PATH` env var). Promote a stronger engine by verifying it in a serve-mode
duel and copying it over `models/engine/baseline`.

## History

The full project history is preserved in git. Old/superseded material lives under
`archive/`; large artifacts (checkpoints, datasets, build output, APKs) are
gitignored. See `docs/superpowers/specs/` and `docs/superpowers/plans/` for the
restructure design and plan.
