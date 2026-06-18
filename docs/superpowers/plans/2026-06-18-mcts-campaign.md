# MCTS / AlphaZero campaign to a strong (2500+) Togyz Kumalak engine

Date: 2026-06-18
Status: plan (the blocker bug is fixed; this is the path to strength)

## Context (what the overnight session established)
- The deployed alpha-beta + NNUE engine is near a low ceiling (~37% on PlayOK; barely beats HCE; the endgame collapse is structural). NNUE retrains / handcrafted eval terms do not break it.
- The MCTS/AlphaZero line (the method competitors used for 2500+) was **abandoned due to a bug**, not a fundamental limit: `mcts/src/mcts.rs:262` backed up terminal values with inverted sign → deep search optimized toward LOSING → "more sims = worse" → the 200-sim eval showed a misleading 0%. **FIXED** (committed 8a37ccc). Post-fix, search HELPS (iter_2645: 1-ply ~5% → 50-400 sims 7.5-12.5%).
- Bootstrap pieces proven: supervised expert imitation gives a strong POLICY (≥1800 games → ~22% @1-ply, 10x the old self-play net). A clean+SPREAD value (engine-eval distillation, sigmoid-normalized) makes search help (D2: 10%→13.3%). Human-outcome value is too noisy; compressed engine value is uninformative.
- **Why naive self-play DEGRADED both bootstraps to 0%** (the key lesson this plan must fix): see Root Causes below.

## Root causes of the self-play degradation (must all be addressed)
1. **No gating.** `train_loop.py` is literally "Training Loop v3 (No Gating)". A bad training step lowers the net, then self-play continues *from the worse net* → downward spiral. AlphaZero needs **keep-best gating**: only promote a new net if it beats the current in an eval match.
2. **Self-play used Gumbel 1-ply, not full MCTS.** The old code avoided full PUCT tree (it was the buggy/inverted path). Gumbel-1-ply policy targets are barely better than the raw policy → no improvement signal. **Now that the tree search is fixed, use full-MCTS self-play** (sims ≥ 400): the visit-count policy target is genuinely *better* than the raw policy → the AlphaZero virtuous cycle (improve → better data → improve).
3. **Expert anchor too weak (0.3) + lr too high for a weak net.** With a ~13% net, low expert weight let self-play drift to the net's own weak distribution. Need a **high expert anchor early**, decaying only as the net surpasses experts.
4. **Eval was hardcoded to 1-ply** (`train_loop.py:546` ignores `--eval-sims`). The real strength signal is 200-sim search. Monitoring/gating must use 200 sims.
5. **Value calibration.** Value targets must be clean (consistent play) AND spread (win-prob-like). Self-play outcomes are clean; bootstrap value from engine-eval (sigmoid-normalized) until self-play takes over.

## Engine/code prep (Phase 0 — ~half a day, do first)
These are the prerequisites; without them the campaign will degrade again.
**Status 2026-06-18: items 1, 3, 4 DONE + the league-hang bug fixed; smoke-test validated end-to-end. Item 2 (full-MCTS self-play) is the remaining engine change.**

1. ✅ **DONE — keep-best gating in `train_loop.py`.** After each iter, the candidate net is evaluated vs the fixed alpha-beta baseline at `--eval-sims`; promoted (becomes the selfplay net `current.onnx` + saved `best.pt`) only if `wr >= best_wr + --gate-margin`. Self-play always uses the current BEST net, never a regressed one. Gate is seeded with the initial bootstrap net's strength before the loop. This is the single most important change (fixes the overnight downward spiral). Validated: PROMOTED/REJECTED logic fires correctly.
2. ⬜ **TODO — full-MCTS self-play** (not Gumbel-1-ply), sims 400-800. Confirmed `self_play.rs` uses `gumbel::gumbel_search` (root + all children in ONE batch = genuine 1-ply; `self_play.rs:1`). The fixed PUCT tree `mcts::search` (`mcts.rs:124`) already returns `(visit_count_policy, root_val)` + Dirichlet noise + configurable sims — exactly the needed primitive. Wiring needed: plumb a `--selfplay-sims` (and a use-full-MCTS flag) through `main.rs::run_league` → `self_play::worker_loop` → `play_one_game`, build an `EvalContext` alongside the `GumbelContext`, and call `mcts::search` for the "full search" fraction (keep `fast_move` for the rest). Rebuild + retest. NOTE: gating (item 1) already prevents *degradation* with gumbel self-play; full-MCTS is what enables *improvement* (deeper policy targets → the AlphaZero virtuous cycle), so do it before the long campaign.
3. ✅ **DONE — gate eval at full sims** (`--eval-sims`, default 800; pass e.g. 200). The old hardcoded `eval_sims=1` is gone; the gate is the eval vs the fixed alpha-beta baseline.
4. ✅ **DONE — expert-anchor schedule.** `--expert-ratio` (start/max), `--expert-decay` (per-iter multiplier, 1.0=off), `--expert-min` (floor). `eff_expert = max(expert_min, expert_ratio * expert_decay**iter)`, logged each iter. Opt-in (defaults preserve old behavior).
5. ✅ Value targets stay spread (sigmoid win-prob in the bootstrap convert; self-play uses swept score-proportional values, clamped 0.3–1.0 in `self_play.rs`).

### Phase-0 fixes discovered while validating
- **DISK SAFETY:** removed the accumulating `iter_N.pt` checkpoints (every 5 iters → ~1.1G over a night). Now only `latest.pt` (resume) + `best.pt` (gate winner) + transient `candidate.onnx` (removed after each gate). Smoke-test footprint stayed at ~44M. Critical given only ~2.2G free.
- **LEAGUE-HANG BUG (latent, now fixed):** `rust_league` built `env` (ORT_DYLIB_PATH + CUDA `LD_LIBRARY_PATH`) but never passed `env=env` to `subprocess.run` — unlike the `--eval` call. When the parent shell lacked the nvidia lib paths, the league child's ORT CUDA provider deadlocked during init (all 18 worker threads stuck on a futex, never reaching the GPU). It only "worked overnight" because `tools/train_corrected.sh` exports those paths in the parent shell. Fixed by passing `env=env`. Now the league runs ~7s regardless of how the parent is launched.

## Phase 1 — Strong bootstrap (~1 day)
Goal: a starting net where 200-sim search **clearly beats the alpha-beta baseline** (target ≥ 50%), so self-play has a strong base that won't regress.
1. **Best policy:** `supervised_pretrain.py --min-elo 2000 --model-size large2m --epochs ~40` on the full PlayOK corpus (380K games) — imitate the strongest humans. (≥2000 gives ~135K positions; consider ≥1900 for more data.)
2. **Good value:** combine with engine-eval distillation. Convert datagen (engine eval, sweep-correct) → 63-byte MCTS replay with **sigmoid-spread** value (the `convert_spread.py` from this session), train value jointly (the D2 recipe), or value-finetune.
3. Iterate the bootstrap (policy data size, value normalization K, epochs) until **200-sim winrate vs the alpha-beta baseline ≥ 50%**. D2 only reached 13% — the bootstrap needs more elite policy data + better value calibration before self-play is worth starting.
4. Export ONNX (dynamo=False), set as the initial BEST net.

## Phase 2 — Self-play campaign (multi-day, the core)
Run the gated AlphaZero loop with the Phase-0 safeguards:
- Self-play: full MCTS, sims 400-800, Dirichlet α≈1.1, temperature decay, ~100-200 games/iter, replay buffer ~500K-1M.
- Train: lr 1e-4 (cosine), 1-2 epochs/iter, expert-anchor schedule (Phase 0.4).
- **Gate every iter at 200 sims; promote only if ≥55% vs current best.** Reject regressions (this is what prevents the collapse we saw).
- Monitor: 200-sim winrate vs the fixed alpha-beta baseline should climb past 50% → 70% → toward saturation.
- Throughput: the old loop ran ~36 iters/hr (gumbel). Full-MCTS self-play is slower but the gating + better targets make each iter count. Budget days-to-weeks of GPU.

## Phase 3 — Push to 2500+ and validate (ongoing)
1. Once the net consistently beats the alpha-beta baseline, keep running self-play (AlphaZero scaling: more iters → stronger).
2. If it plateaus, scale the net (more res blocks/channels — the engine supports larger) and/or more sims.
3. **Validate out-of-lineage** (the hard lesson from the NNUE work): vs HCE, vs the alpha-beta baseline, and LIVE on PlayOK vs strong humans (non-rated first). Cross-check results against the official PlayOK profile, not just bridge logs.
4. Deploy via the mcts `--serve` mode (drop-in for the NNUE engine in the bridge) once it beats the deployed engine live.

## Compute / milestones / risks
- **Compute:** single RTX 5080. Phase 1 ~hours. Phase 2-3 = days-to-weeks (AlphaZero is compute-hungry; competitors likely spent more). This is the honest cost.
- **Milestones (gates to continue):** (M1) bootstrap ≥50% vs alpha-beta @200 sims; (M2) self-play net ≥70% vs alpha-beta; (M3) beats strong humans live; (M4) approaches 2500+ on PlayOK.
- **Risks:** (a) self-play still degrades → the gating + full-MCTS + expert anchor are the mitigations; if it still regresses, lower lr / raise sims / raise expert anchor. (b) net capacity ceiling → scale the net. (c) compute insufficient for 2500+ → accept a strong-but-sub-2500 result, or get more GPU.
- **What NOT to do:** repeat the abandoned conclusions (more sims hurts — that was the bug; value head useless — it was the inverted eval). Don't run the loop without gating (that caused the overnight collapse).

## Artifacts / pointers
- Bug fix: `mcts/src/mcts.rs` (committed). Engine: `target/release/mcts`. Net: TogyzNet large2m (2.24M), input [7,9], policy[9]+value[1], replay 63B/record.
- Bootstrap nets saved: `models/nets/mcts_2026-06-18/` (d2.pt/onnx = best foundation; sup_e1800.pt = strong policy).
- Tools: `supervised_pretrain.py`, `train_alphazero.py`, `train_loop.py` (needs the Phase-0 gating + eval-sims fixes). ONNX export dynamo=False. CUDA/ORT env per `tools/train_corrected.sh`. mcts `--eval` needs ABSOLUTE engine path.
- Memory: [[mcts-value-sign-bug-2026-06-18]].
