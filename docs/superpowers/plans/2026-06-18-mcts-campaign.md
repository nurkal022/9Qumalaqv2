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
1. **Add keep-best gating to the training loop** (new script or fix `train_loop.py`): after training iter N, play candidate vs current-best at 200 sims (e.g. 30-40 pairs); promote only if winrate ≥ 55%. Self-play always uses the current BEST net, never a regressed one. This single change is the most important.
2. **Use full-MCTS self-play** (not Gumbel-1-ply) in the self-play data generation, sims 400-800. Verify the `mcts` binary's self-play path uses the (now-fixed) PUCT tree, or add a flag. Confirm Dirichlet noise + temperature are on for exploration.
3. **Fix monitoring eval to 200 sims** (`train_loop.py:546` → pass `args.eval_sims`); add eval vs a FIXED reference (alpha-beta baseline) AND vs the previous best, both at 200 sims.
4. **Expert-anchor schedule:** expert-ratio starts ~0.6, decays (e.g. ×0.97/iter) toward ~0.1 as the net's eval surpasses the alpha-beta baseline.
5. Keep the value-target convention spread (sigmoid win-prob), and clamp magnitudes like the self-play code already does.

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
