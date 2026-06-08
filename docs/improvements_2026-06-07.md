# Engine Improvement Session — 2026-06-07

Goal: find concrete, measured methods to make the engine strong enough to beat the
PlayOK champions `mcts` / `mcts2` (ELO ~2520). Approach: 15-agent codebase audit →
prioritized roadmap → implement + **measure every change** with real games.

## TL;DR — measured results

| Change | Result | Status |
|---|---|---|
| **Revert eval-correction regression** (search.rs) | HEAD was **0%** vs baseline → **55.5%** (100g, Elo +38) | ✅ shipped to working tree |
| **End-game sweep rule fix** (board.rs + game.py + EGTB) | **+122 Elo** vs unfixed engine under correct rules; sweep proven 96/96 vs PlayOK | ✅ shipped to working tree |
| **Cumulative: improved engine vs Mar-18 baseline** | **58.0%** (57W-2D-41L, Elo +56, 100g, correct rules) | ✅ |
| Champion opening book | **−76 Elo** (hurts) | ⛔ reverted (negative result) |

The maintained codebase now **beats the previous champion baseline** and the
single worst long-standing bug (the root of "value head unreliable") is fixed.

All matches: `tools/ab_match.py`, color-swapped, 150 ms/move, `--nobook`, neutral
Python referee. 100-game std ≈ 5%.

---

## Finding 1 — Self-inflicted engine regression (recovered + exceeded)

The Apr-29 build lost **0–4, then 0%** to `models/engine/baseline` (Mar-18) *with the
same nnue/egtb/book loaded* → a pure **code** regression, not data.

**Root cause:** the NNUE-path `eval()` correction block in
[engine/src/search.rs](../engine/src/search.rs) (was ~lines 182–247) stacked untuned
heuristics — mobility `×8×(up to 10)=×80`, pit-asymmetry on raw stone-diff `×12`,
`±600` zugzwang, `finish ×100`, quadratic starvation — that **dwarfed the trained
NNUE base** (~centipawn scale), making eval wild. (Note: matches always run with
NNUE — `run_match` does `NnueNetwork::load(...).unwrap()` — so the HCE `eval.rs` is
only the no-NNUE fallback. The load-bearing eval is this search.rs block.)

**Fix:** reverted to baseline `bb1ced9`'s smooth `scale=((65-total)/15).clamp(1,4)`
version (mobility `×3×scale`, etc.), `ASP_DELTA` 35→20.
**Measured: 55.5% vs baseline / 100 games (Elo +38).** The maintained build kept the
*good* search additions (countermove, continuation-history, qsearch TT) and dropped
only the bad eval → now stronger than baseline.

## Finding 2 — ★ The end-game stone-sweep rule was wrong everywhere

Real Togyzkumalak: at a terminal where a side is empty (neither kazan ≥82), each side
sweeps its **own** remaining board stones into its **own** kazan before comparing.
The codebase compared **raw kazans** — wrong winner on **54%** of empty-side
terminals.

**Proven against reality** (`/tmp/check_rule.py`): replayed the 943 champion games;
106 ended by board-emptying with both kazans <82. The sweep rule matches the recorded
PlayOK `Result` **96/96** discriminating cases; raw-kazan matches **0/96**.
Example: `kazan 66-66, board W30/B0, PlayOK=1-0` → raw says *Draw*, sweep says
*White*, PlayOK says *White*.

**This is the concrete root cause of "value head unreliable."** ~59% of decisive
endgames (106 of 179) are decided by the empty-side rule, and the NN training pipeline
labeled them all with the **wrong winner**. The value head learned from corrupted
targets — it was never inherently broken.

**Fixed in 3 places (all required to be consistent):**
- [core/src/board.rs](../core/src/board.rs) `game_result()` — sweep; 2 tests added (TDD), 14/14 green.
- `archive/old-impls/alphazero-code/alphazero/game.py` `_check_winner()` — the referee
  used by `ab_match.py` **and the rules that generate every NN training label**.
  ⚠️ This is in gitignored `archive/` — **the fix is not version-controlled; move the
  rules into the tracked core (or have training call core) so it can't regress.**
- EGTB regenerated (`engine egtb-gen 4`): new `egtb.bin` is **95.9%** self-consistent
  vs the old **93.8%** — corrected terminals improved it.

**Measured:** under the *correct* referee the sweep-engine beats the raw-engine
**66.9% (Elo +122)**. Under the OLD (wrong) referee it scored 35% — a pure
referee/engine-mismatch artifact. *Lesson: measure a rule change with a referee that
uses the same rule.*

## Finding 3 — Champion opening book hurts (negative result)

Built a champion-mimicking book from the 943 `mcts` games (`tools/gen_champion_book.py`).
Fixed a tokenizer bug in `gen_opening_book_v2.py` (the 2-digit regex captured move
numbers ≥10 as phantom moves) → legal replay **3.7% → 100%**. Reproduced the
champion repertoire exactly: **White 1st move pit7=35%, pit6=24%**, Black reply to
pit7 → pit9 86%, tuzdyk on opponent pit 6.

A/B (book ON vs OFF, same engine): **39.2% / Elo −76** — the book *hurts*. Our
recovered search outplays a frequency-book derived from games against weaker
opposition. Reverted the book + the book-gate change. **The path to champions is the
strong search, not copying opening moves.** (Tool + tokenizer fix kept for future use.)

---

## Recommended next steps (prioritized, not done this session)

1. **★ Retrain the NN value head with corrected labels** (GPU). Now that `game.py`
   sweeps correctly, regenerate training targets — this is the real fix for "value
   head unreliable" and may finally let MCTS search >1-ply help instead of hurt.
   First move the corrected rules into tracked code.
2. **Promote the improved engine to product** after a ≥100-game serve duel:
   `cp target/release/togyzkumalaq-engine models/engine/baseline` (reversible; tracked
   in git). The product currently serves the *weaker* state. (Hold for confirmation.)
3. **CI regression gate** (the audit's recommendation that would have caught both
   shipped regressions): `tools/ab_match.py <new> models/engine/baseline 100 120
   --nobook --min 45` fails the build if score <45%. Commit `ab_match.py` (untracked).
4. **Singular extensions** (search.rs): currently disabled on a rationale the audit
   refuted — branching ≤9 and *shrinks* to 2–4 in the decisive endgame, and the game
   is full of forced/only-good moves, so SE verification is cheap. Needs an
   `excluded_move` param threaded through `alpha_beta` + a ≥200-game SPRT. Medium ELO.
5. **Unify texel tuner with the live eval** (texel.rs tunes a stale 11-weight copy;
   the per-column `TUZDYK_VALUE[9]` it prints doesn't exist in eval.rs). Then re-tune
   the tuzdyk/parity terms on champion + 2000-ELO data. (Note: only affects HCE
   fallback unless the engine is switched to HCE.)
6. **Test qsearch-TT removal** (search.rs ~977–991 probe, 1077/1087 stores): the probe
   returns Exact ignoring the α/β window and depth-0 stores can clobber deeper
   entries. Quick A/B — keep only if neutral-or-positive.

---

# Session 2 (2026-06-08) — value-head retrain + singular extensions

## Finding 4 — Value head retrain with corrected labels: VALIDATED (but NN still floored)

Fixed the training pipeline (`train_loop.py` RUST_BINARY path; engine-asset symlink
`engine/target/release`; swept value **magnitude** in `self_play.rs` + `league.rs` —
not just the sign), then trained 60 league iters from iter_2645 with corrected labels
(`tools/train_corrected.sh`). Floor-free `tools/value_calibration.py` on champion
endgames:

| value-sign accuracy on empty-side endings | iter_2645 | retrained |
|---|---|---|
| matches TRUE swept outcome | 59.4% | **73.0%** (+13.6) |
| matches OLD raw-kazan rule | 49.5% | **39.3%** (de-learned) |

So corrected labels measurably **heal the value head** — the direct proof. **But** vs
the now-strong classical engine the 2 M net is floored ~0% (sims=1: 15%→0%), and its
policy is only **34% top-1** vs the engine (`tools/policy_match.py`) — too weak to
help alpha-beta move ordering. The NN path needs a bigger model + long training, or a
policy distilled *from the engine's own moves* (datagen records best_move at byte 26).

## Finding 5 — Singular extensions: +55 Elo

Re-enabled SE in `search.rs` (the audit refuted the "node explosion" disable rationale
— branching ≤9 and shrinks to 2–4 in the endgame). Implemented with a field-based
`excluded_move` (read-and-clear at node entry, no param threading), a verification
search at `(depth-1)/2` with a window around `tt_score − 2·depth`, gated `depth≥6 &
tt_depth≥depth−3 & LowerBound/Exact`, skipping TT cutoff+store during verification.

**A/B SE-on vs SE-off** (same binary, toggled by the depth gate, 120 games):
**57.9% / Elo +55.** The margin (2·depth) is untuned — a 3·depth variant is built
(`target/eng_se3`) for comparison.

## Working-tree changes (both sessions)

Tracked: `core/src/board.rs` (sweep fix + tests), `engine/src/search.rs` (eval revert
+ singular extensions), `mcts/src/{self_play,league}.rs` (swept value magnitude),
`research/training/train_loop.py` (path fix), `tools/{ab_match,gen_champion_book,
train_corrected,value_calibration,policy_match}.py` (new). Gitignored-but-changed:
`engine/egtb.bin` (regenerated on corrected terminals), `archive/.../game.py` (referee
sweep fix — ⚠️ move into tracked core), `engine/target/release/togyzkumalaq-engine`
(symlink for asset resolution). Nothing committed; nothing deployed to production.
