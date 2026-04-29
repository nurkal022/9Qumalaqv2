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
