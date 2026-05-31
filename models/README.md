# models/ — blessed champions

Promoted, production-blessed artifacts the **product** depends on. Decoupled from
research build output: the web backend reads its engine **only** from here
(`product/web/backend/app/config.py` → `engine_path = models/engine/baseline`).

## engine/
- **`baseline`** — the champion classical engine served to players. Rebuilt from
  commit `bb1ced9` (Mar "baseline" era); ~88–93% vs the current build in serve-mode
  head-to-head (2026-05-31). Tracked in git (~660 KB), so it ships with the repo.

Promote a new engine: build it, verify it beats `baseline` in a serve-mode duel,
then `cp` it over `models/engine/baseline` (or point `ENGINE_PATH` at it).

## nets/
Champion neural nets (AlphaZero/MCTS). Heavy `.pt`/`.onnx` are gitignored — track
via git-LFS if/when needed. The current best net (not used by the classical-engine
product) is `research/runs/_legacy/checkpoints_v3/iter_2645.pt` (~27 MB).
