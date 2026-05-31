# Champion-Ready Setup — Togyz Kumalak

The strongest playing entities available on this machine, ready to face a champion.

## Quick start

| What you want | Run |
|---|---|
| Play interactively against the strongest engine | `./play_champion.sh` |
| Use the neural model programmatically (stdin/stdout) | `./play_mcts.sh` |
| Train more (NOT recommended — see "Why not just train more") | `bash rust-mcts/scripts/run_night_training.sh` |

## What's strongest, and why

The **strongest single playing entity is `togyzkumalaq-engine-baseline`** (Mar 18 build) — a hand-crafted alpha-beta engine augmented with NNUE eval (40-256-32-1), a 3.97 M-position endgame tablebase, and a 21 K-position opening book. Empirical match-ups (1-ply Rust eval, 40-game samples):

| Player | vs `engine-baseline` | vs `engine` (current, Apr 29) |
|---|---|---|
| `night_best` (our NN, iter 2650) | 0.0% | 15.0% |
| `start_iter_2645` (pre-night NN) | 1.2% | 11.2% |
| `night_latest` (post-night NN) | — | 7.5% |

Two takeaways:

1. **Gen7-baseline beats both Gen7-current and the NN model.** "Baseline" here is a *misnomer* — it is in fact the strongest engine variant on disk. The "improved/improved2/improved3" naming refers to refactoring/speed cleanups, not playing strength.
2. **The neural model alone is not yet at champion level.** ~15% vs the current alpha-beta engine, ~0–1% vs the strongest variant. Champion-vs-champion has to lean on the alpha-beta + NNUE engine.

## What does each script run?

### `play_champion.sh`

Launches `togyzkumalaq-engine-baseline play` — interactive REPL with a printed board:

```
==================================================
  Black Kazan: 0  |  Tuzdyk: -
    9  9  9  9  9  9  9  9  9
    1  2  3  4  5  6  7  8  9
--------------------------------------------------
    1  2  3  4  5  6  7  8  9
    9  9  9  9  9  9  9  9  9
  White Kazan: 0  |  Tuzdyk: -
==================================================
Your move (pit 1-9):
```

Type `1`–`9` to move; `undo` takes back; `quit` exits. The engine plays Black by default.

### `play_mcts.sh`

Launches `rust-mcts --serve` with `eval_onnx_final/night_best.onnx` (the strongest neural-net checkpoint from the night training). Speaks a simple line-based protocol on stdin/stdout for embedding in another tool (e.g. PlayOK proxy):

```
> newgame
ready
> go pos 9,9,9,9,9,9,9,9,9/9,9,9,9,9,9,9,9,9/0,0/-1,-1/0 time 1000
bestmove 8 score 0 depth 1 nodes 10 time 784 nps 100
> quit
```

Move selection is **Gumbel 1-ply** (raw policy, no deep search). Full `--eval-sims=200` MCTS search exists but currently *regresses* (the value head is unreliable, so more sims pull the policy toward worse moves). 1-ply matches the strength measurement we trust.

## Why not just train more?

We did. The night session ran 158 selfplay+train iterations on top of `iter_2645`, the previous best. Loss dropped 54 % (`p_loss` 1.13 → 0.52), but **playing strength stayed inside its previous noise band** (~11–15 % vs Gen7-current). See [nigtht_report.md](nigtht_report.md) for the full data.

The existing pipeline has saturated this dataset distribution. Real gains now require *different* data:

1. **Distillation from `togyzkumalaq-engine-baseline` depth-12 datagen** (not selfplay). Generates ~10 K games of strong-engine moves, used as supervised target. Run: `bash rust-mcts/scripts/run_max.sh` (will need ~3–4 h GPU).
2. **Distillation from the 965 archived `mcts` PlayOK games** (rival at ELO ~2520). Located in `archive/datasets/game-pars/games/`; pre-extracted as `archive/datasets/game-pars/mcts_training.bin`.
3. **Larger model.** RTX 5080 has the headroom; 4 M–8 M params may break the value-head ceiling that 2 M sits at.

## File map

```
9QumalaqV2/
├── play_champion.sh                # strongest interactive play (Gen7-baseline)
├── play_mcts.sh                    # NN serve mode (night_best ONNX)
├── CHAMPION_SETUP.md               # this file
├── nigtht_report.md                # full night-run findings (training, evals, verdict)
├── engine/
│   ├── target/release/
│   │   ├── togyzkumalaq-engine-baseline   # ★ strongest binary (use this)
│   │   ├── togyzkumalaq-engine            # current default, ~10% weaker
│   │   └── ...                            # improved/improved2/etc — not stronger
│   ├── nnue_weights.bin            # NNUE weights (40-256-32-1, 18753 params)
│   ├── egtb.bin                    # endgame tablebase (~3.97 M positions)
│   └── opening_book.txt            # opening book (~21 K positions)
└── rust-mcts/
    ├── target/release/rust-mcts    # AlphaZero-style MCTS binary
    ├── checkpoints_night/best.pt   # ★ strongest NN checkpoint (iter 2650)
    ├── checkpoints_night/latest.pt # final NN checkpoint (iter 2803)
    ├── checkpoints_v3/iter_2645.pt # pre-night strongest (still tied with best)
    └── eval_onnx_final/
        ├── night_best.onnx         # used by play_mcts.sh
        ├── night_latest.onnx
        └── start_iter_2645.onnx
```

## If you want PlayOK champion-match support

The serve protocol is already plug-in compatible with the production server in `web/server.py`. To deploy `night_best` to the LAN play server (`10.0.34.22`, (credentials in DEPLOY_PASSWORD env)):

```bash
# from project root
python3 deploy_lan.py
```

The deploy script uploads engine + ONNX + web frontend and restarts the service. Note: this replaces the *currently running* model — it is reversible by re-deploying the previous `model_2m.onnx`. Confirm before running.

## Honest summary

For an actual champion match **today, on this machine**, the safest bet is `play_champion.sh` (Gen7-baseline alpha-beta + NNUE + EGTB + book). It is the strongest single thing in this repo. The neural network is interesting research but is not yet at championship strength on this hardware/dataset.

To make the neural network competitive, the next move is **dataset replacement**, not more iterations of the same pipeline. The infrastructure is ready — only the data recipe is the bottleneck.
