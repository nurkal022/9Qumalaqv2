# Night Report — 6h Autonomous Run

**Started:** 2026-05-06 02:03 (Asia/Almaty)
**Ended:** 2026-05-06 06:30 (final eval), report finished ~06:35
**Budget:** 6 hours, current hardware (RTX 5080 Laptop 16GB, 24 cores, 30GB RAM)
**Goal:** Maximize playing strength on current hardware in 6 hours.

---

## TL;DR

**Honest result: the model didn't get measurably stronger tonight.** 158 selfplay+train iterations on top of iter_2645 reduced training loss by 54% but **playing strength vs Gen7 stayed flat at ~11-15%** (within noise). The training pipeline that built v3 has hit its ceiling — more of the same will not produce a champion.

**3 useful side-discoveries:**
1. Memory was wrong: actual best v3 checkpoint is **iter_2645**, not iter_500 (verified 15% vs 7.5% in 1-ply eval).
2. **965 games of rival `mcts` (ELO 2520)** are already extracted in `archive/datasets/game-pars/games/` — gold dataset for distillation.
3. **Gen7-baseline (Mar 18 binary) is much stronger than Gen7-current (Apr 29).** Models score 1% vs baseline and 11% vs current. Suggests engine regression worth investigating.

**What to do next** (concrete, in order of expected value):
1. Pure distillation training from Gen7-baseline depth-12 datagen + the 965 mcts games + PlayOK 2000+ — fresh model, not selfplay continuation.
2. Investigate Gen7-current vs Gen7-baseline regression. If real, swap deployed engine.
3. Try 4M-8M params (we have GPU headroom) — current 2M may be capacity-limited like NNUE was.

**Nothing was deployed.** Current production server unchanged.

**Champion-ready setup delivered:** see [CHAMPION_SETUP.md](CHAMPION_SETUP.md), `play_champion.sh`, `play_mcts.sh`.

---

## Recon Findings (00:00–00:15)

### Pipeline state
- **Latest checkpoint:** `iter_2645.pt` (Apr 14, 3 weeks idle)
- **Memory says best:** `iter_500.pt` (p_loss 1.09) — but never re-verified vs later iters
- **Gap:** v3 trained 500→2645 (≈2145 iters more), strength unknown
- **Pipeline scripts:** `run_long.sh` (continuous selfplay+train), `run_max.sh` (Gen7 datagen + distillation), `pipeline_max.sh` (multi-model train)
- **Built binaries:** `rust-mcts`, `togyzkumalaq-engine` (release) — both ready

### Data sources discovered
- **PlayOK corpus:** 182,670 games at `/home/nurlykhan/game-pars/games/`
- **Rival "mcts" found:** 7 games, ELO **2519–2523** (vs opponents 1197–1949)
  - Result: **6 wins, 1 draw, 0 losses** (winrate 92.8%)
  - Opponents: wmw3166g(1197), madridr(1947–1949), camry75(1828–1861)
  - All games dated 2025-09-12 (single batch)
  - **Note:** 7 games is too few for distillation. Useful only as diagnostic targets.
- **Other "mcts*" nicks:** mcts4 (1226–1384), mcts7 (1429) — much weaker, not the rival.

### Sample mcts (ELO 2520) games
```
[White "mcts"] vs [Black "wmw3166g" 1197] → 1-0
[White "madridr" 1947] vs [Black "mcts" 2523] → 1/2-1/2
[White "madridr" 1949] vs [Black "mcts" 2521] → 0-1 (mcts won)
[White "mcts" 2518] vs [Black "camry75" 1861] → 1-0
[White "mcts" 2522] vs [Black "madridr" 1948] → 1-0
[White "mcts" 2520] vs [Black "camry75" 1828] → 1-0
[White "camry75" 1829] vs [Black "mcts" 2519] → 0-1
```

### BLOCKER (resolved): GPU driver
- `nvidia-smi` failed: kernel module not loaded for kernel 6.17.0-23
- Fix applied: `sudo apt install linux-modules-nvidia-580-open-6.17.0-23-generic` + `modprobe nvidia`
- GPU ready: RTX 5080 Laptop 16303 MiB, driver 580.142, CUDA 13.0

### Bonus discovery: archive has 965 mcts player games
- `/home/nurlykhan/9QumalaqV2/archive/datasets/game-pars/games/` — 965 games of `mcts` player (ELO 2520)
- `mcts_training.bin` (2.1MB) — already extracted training data
- Lots of supervised PlayOK data: `playok_all_elo1400.bin` (172M), `playok_elo1500.bin` (51M), `playok_elo1600.bin` (27M), `merged_training_data.bin` (351M)
- Could be used for distillation if main path fails

### Engine binaries
- Default `togyzkumalaq-engine` (Apr 29) — newer than memory's "Gen7"
- `togyzkumalaq-engine-baseline` (Mar 18) — older, might be original Gen7
- This may explain why memory's "50% vs Gen7" measurements differ from current eval

---

## Diagnostic Eval Results (1-ply, 10 pairs / 20 games, engine=100ms)

| Checkpoint | W-D-L | Winrate | Pair score |
|---|---|---|---|
| iter_500   | 1W-1D-18L | 7.5%   | 0W-1D-9L |
| iter_1500  | 0W-0D-20L | **0.0%** | 0W-0D-10L |
| **iter_2645** | **2W-2D-16L** | **15.0%** | **0W-2D-8L** |

**Surprise finding:** iter_2645 (latest) is ~2x stronger than iter_500 (memory's "best"). Training from 500→2645 was non-monotonic (iter_1500 dipped to 0%) but did improve overall.

**Decision:** Use iter_2645 as starting point for night training.

**Note:** Even iter_2645 is well below 50% vs current Gen7. Memory's "50% vs Gen7" was likely vs an older binary. Today's engine is stronger.

---

## Plan

### Phase 1 — Diagnostic (30 min after GPU)
Run mini-tournament between v3 checkpoints to find true best:
- Candidates: iter_500, iter_1000, iter_1500, iter_2000, iter_2645
- Format: round-robin, 20 games per pair (color-paired), 1-ply (raw policy)
- Goal: identify "current best" baseline

### Phase 2 — Vs Gen7 baseline (30 min)
Best checkpoint vs Gen7 engine, 50 game pairs, MCTS 200 sims.
Goal: absolute strength number.

### Phase 3 — Improvement (3.5h)
**Path A (default):** Continue MCTS selfplay+train from current_best for 3.5h
- Resume `run_long.sh`-style training
- Save new checkpoints to `checkpoints_v3/`
- Expected: ~50–80 new iterations on top of current best
- ~100K new selfplay games

**Path B (alternative):** Distillation from Gen7 engine (depth 12) for 3h, then 30 min eval
- If checkpoint tournament shows v3 has plateaued, switch to distillation
- Use existing pipeline `run_max.sh` (already configured)

### Phase 4 — Final eval (30 min)
- New best vs old best (50 game pairs)
- New best vs Gen7 (50 game pairs)
- Decision: deploy or not (server is at 10.0.34.22 LAN per memory)

---

## Live Log

| Time  | Event |
|-------|-------|
| 02:03 | Recon complete. GPU driver missing for current kernel. |
| 02:58 | GPU driver fixed (apt install + modprobe). RTX 5080 ready. |
| 03:01 | Smoke test: 200-sim eval shows iter_500 = 0% vs Gen7 (value head issue at high sims). |
| 03:08 | Switched to 1-ply eval (engine_time=100ms, 10 pairs/checkpoint). |
| 03:12 | Diagnostic eval done: iter_2645 = 15%, iter_500 = 7.5%, iter_1500 = 0%. |
| 03:15 | Starting long training from iter_2645 (3h budget, run_long-style). |
| 03:14 | First training restart: forgot to symlink game-pars. Restarted with expert data (19206 positions, PlayOK 2000+, 20% mixing). |
| 03:15 | Iter 2646: League 51s, 9088 positions, 1.9 g/s. Loss=1.249 (p=1.13, v=0.12). Iter time 53s. |
| 03:15 | Projection: ~200 iters in 3h → final at iter ~2845. Evals at iter 2650, 2675, 2700, ... 2845 (every 25 iters). |
| 03:18 | After 5 iters: loss 1.25 → 0.94 → **0.83 best p_loss** (memory's iter_500 was 1.09 — training is converging quickly on new data). |
| 03:20 | Iter 2650 eval: 0W-3D-7L = 15.0% (5 pairs). Same as pre-training baseline. |
| 03:40 | Iter 2675 eval: 0W-1D-9L = **5.0%** (5 pairs). Drop, but 5-pair std ≈ 15% so within noise. |
| 04:06 | Iter 2700 eval: 0W-1D-9L = **5.0%** (5 pairs). Two consecutive low. Loss 1.25 → **0.66** (p=0.60). |
| 04:08 | **Stat note:** combined 20-game eval (15% pre, 5%×2 post) overlap in 95% CI. Cannot conclude strength change yet. Final eval (40 games each) will be authoritative. |
| 04:39 | Iter 2725 eval: 1W-1D-8L = **15.0%** (5 pairs). Recovered. Pattern 15→5→5→15 = noise within ±10%. |
| 05:10 | Iter 2750 eval: 1W-0D-9L = **10.0%** (5 pairs). 4 post-training evals avg ~10%. Loss plateaued at p=0.55 (from 1.13 init). |
| 05:38 | Iter 2775 eval: 0W-1D-9L = **5.0%** (5 pairs). Aggregate 6 post evals (60 games) = **9.2%**, vs pre 15% (20 games). |
| 06:10 | Iter 2800 eval: 1W-0D-9L = **10.0%** (5 pairs). |
| 06:14 | Training timeout reached (3h). 158 iters done (2646→2803). Final loss p=0.52, v=0.060 (from p=1.13/v=0.12 init = -54% / -50%). |
| 06:14 | Launching final 20-pair eval (3-way: start/best/latest, vs current and baseline engine). |

---

## Training Summary

| Metric | Init (iter 2645) | Final (iter 2803) | Change |
|---|---|---|---|
| Iterations done | — | 158 | — |
| Total selfplay positions | — | ~1.3M | — |
| Loss (total) | 1.249 | 0.585 | **−53%** |
| p_loss | 1.126 | 0.525 | **−53%** |
| v_loss | 0.123 | 0.060 | **−51%** |
| 1-ply eval winrate | 15% (20 games) | 9% (60 post-train games) | within noise |
| Best monitoring eval | — | 15% @ iter 2650 | tied init |

**Loss decreased dramatically (−50%+) while playing strength stayed flat.** Classic overfitting-to-data pattern: the model fits selfplay+expert targets well, but the value+policy heads aren't translating to wins vs Gen7. This matches memory's note: "Value head unreliable — policy head is the primary signal."

The 158 iterations on iter_2645 are essentially diminishing returns — the same training pipeline that took 2645 iters to reach this strength can't push much further with current data sources.

---

## Final Eval Results (40 games each, 1-ply, engine=100ms)

### vs Gen7-current (Apr 29 binary)

| Checkpoint | W-D-L | Winrate | Pair score |
|---|---|---|---|
| start_iter_2645 | 3W-3D-34L | **11.2%** | 0W-3D-17L |
| **night_best (iter 2650)** | 6W-0D-34L | **15.0%** | **1W-4D-15L** |
| night_latest (iter 2803) | 2W-2D-36L | 7.5% | 0W-2D-18L |

**Best: `night_best` at 15.0%** — 3.8 pp above start, 7.5 pp above latest. With 40-game std ≈ 5.5%, gap vs start is borderline-significant; gap vs latest is significant.

### vs Gen7-baseline (Mar 18 binary, supposedly older)

| Checkpoint | W-D-L | Winrate |
|---|---|---|
| start_iter_2645 | 0W-1D-39L | **1.2%** |
| night_latest | 0W-0D-40L | **0.0%** |

**Surprise:** Baseline engine is *much* stronger than current. The naming "improved/improved2/improved3" likely refers to code restructuring or speed, not strength — the Mar 18 binary plays better. Worth investigating later (could be a regression in the current engine).

---

## Verdict

| Question | Answer |
|---|---|
| Did 158 iters improve the model? | **No, marginally hurt or flat.** |
| Best checkpoint after night | `checkpoints_night/best.pt` (iter 2650) |
| Strength vs Gen7-current | ~15% (no real change from start) |
| Strength vs Gen7-baseline | ~1% — Gen7-baseline dominates |
| ELO vs rival "mcts" (2520) | unmeasured this session — would need PlayOK eval |

**Tonight's training was a controlled negative result.** The pipeline is functional, the GPU works, the data flows, but **continuous-selfplay-with-expert-mixing has reached its ceiling on iter_2645's data distribution.** More iterations of the same recipe will not produce a champion-level engine.

---

## Recommendations for Next Steps

1. **Switch to distillation, not selfplay** — train a fresh model on Gen7-baseline (the strong one) generated data via `togyzkumalaq-engine datagen 10000 12 16` (~3-4h on this hardware). Memory says NNUE hit ceiling but transformer-based 2M model with engine teacher could do better.

2. **Investigate Gen7-current vs Gen7-baseline regression.** If baseline really is stronger, switch deployed engine. If it's a binary/build artifact, fix and re-test.

3. **Try larger model.** 4M-8M params with same training data may break the "value head unreliable" ceiling. RTX 5080 has the headroom.

4. **Adversarial selfplay (league mode).** Force the model to play vs Gen7-baseline (not just self) — increases diversity of training signal.

5. **Direct distillation from `mcts` PlayOK player (2520 ELO).** 965 games available in archive (`/home/nurlykhan/9QumalaqV2/archive/datasets/game-pars/games/`). Combine with Gen7-baseline distillation.

6. **Don't deploy `night_latest`.** If anything, swap deployed model with `night_best` (iter 2650) — but the gain is within noise, so probably not worth the deploy risk. Current production model is fine.

---

## Artifacts Produced This Night

| Path | Description |
|---|---|
| `rust-mcts/checkpoints_night/best.pt` | iter 2650, 15.0% vs Gen7-current (best from training) |
| `rust-mcts/checkpoints_night/latest.pt` | iter 2803, 7.5% vs Gen7-current (final) |
| `rust-mcts/checkpoints_night/iter_2650...2800.pt` | 31 intermediate checkpoints |
| `rust-mcts/eval_onnx_final/*.onnx` | ONNX exports for eval (start, best, latest) |
| `rust-mcts/eval_results_quick.txt` | Initial diagnostic eval (3 ckpts × 10 pairs) |
| `rust-mcts/final_eval.txt` | Final eval (5 runs × 20 pairs) |
| `rust-mcts/scripts/run_night_training.sh` | Night training launcher |
| `rust-mcts/scripts/summarize_training.py` | Log parser / summary tool |
| `rust-mcts/run_eval_quick.sh`, `run_final_eval.sh` | Eval scripts |
| `nigtht_report.md` | This report |

Symlinks created (training requires them to find data/code):
- `/home/nurlykhan/alphazero-code` → archive (model code)
- `/home/nurlykhan/9QumalaqV2/alphazero-code` → archive
- `/home/nurlykhan/9QumalaqV2/game-pars` → real PlayOK dataset

System changes (require sudo):
- Installed `linux-modules-nvidia-580-open-6.17.0-23-generic` to fix GPU on current kernel.

