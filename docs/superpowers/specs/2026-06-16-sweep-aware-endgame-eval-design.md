# Sweep-aware endgame evaluation — design (Phase 1)

Date: 2026-06-16
Status: approved (design), pending implementation plan

## Context & evidence

Re-analysis of 226 real PlayOK games (`tools/playok/games/`) showed the engine
(`models/engine/baseline`: alpha-beta + NNUE + EGTB + book) is strong overall
(155-63-8 = 68.6%) but loses to strong opponents via a single repeatable failure:

- In **61/63 losses the bot led ≥10 kazan, then collapsed in the endgame.**
  White kazan-lead curve at 25/50/75/end of game = **+15 / +15 / +16 / −10**
  (wins: +11/+16/+20/+27). The midgame is identical to wins; the entire
  difference is the endgame.
- Re-running the engine on every loss: at 50% of the game it still rates White
  ≥0 in **90%** of these (eventually lost) games (median eval **+128**); it only
  recognizes the loss at ~90% of the game (forced-loss in 49/62). **The
  evaluator is blind to the coming sweep collapse until it is too late.**
- Endgame sweep rule (each side scoops its own remaining board stones into its
  kazan at game end) means: a stone safely on your own side is worth nearly as
  much as a kazan stone. In losses the opponent ends with avg **31.3** board
  stones (a hoard that sweeps into their kazan) vs the bot's **5.3**.

### Root cause in code
`engine/src/search.rs` `eval()` (lines ~174-222) is the live evaluator: NNUE
output `/64` plus a handcrafted **endgame correction** applied when total board
stones ≤ 60. The correction models the **wrong dynamic**:
- It rewards *starving the opponent* (`opp_stones <= 20` → starvation bonus) and
  *finishing them off* — but in losses the opponent is NOT starved, it HOARDS.
- There is **no term** for "my own-side stones are future kazan" or "the
  opponent's board hoard is future opponent kazan."

`engine/src/eval.rs` (handcrafted fallback, used only when NNUE absent) has the
same blind spot: `MATERIAL_WEIGHT = 21` (kazan stone) vs
`PIT_STONES_WEIGHT = 3` (board stone) — board stones flat-weighted at 1/7 of
kazan regardless of game phase, though under the sweep rule own-side stones
approach kazan value as the board empties.

### Why not EGTB
Measured: the decisive endgame lives at ~44-58 board stones; the sweep-ending
terminates games with a median of **32 stones still on the board**. The current
4-stone EGTB fires in **1/265 games (~0%)**; even a ≤20-stone table (infeasible)
would fire in only 22% and only for the last ply or two. EGTB cannot reach the
phase where the result is decided. The only viable lever is the evaluator.

## Goal (Phase 1)
Make the evaluator sweep-aware in the endgame so it stops over-rating a greedy
kazan lead and accounts for both sides' board stones as future kazan. Concretely:
1. The +15→−10 collapse signature should shrink or disappear.
2. On the 62 recorded losses, the evaluator should no longer rate those midgame
   positions as comfortably winning (the +128-at-50% blindness should drop).
3. No regression vs the current baseline overall (ab_match ≥ 50%, target ≥ 55%
   with measurable Elo gain).

## Non-goals (Phase 1)
- NNUE retraining (Phase 2).
- Tuzdyk-race / opening for second player (Phase 3).
- Search tuning / early-exit (Phase 4).
- Touching `models/engine/baseline` (production) before the gate passes AND the
  user approves promotion.

## Design

### Approach (chosen: A)
Rewrite the endgame-correction block in `search.rs` `eval()` to be sweep-aware,
keeping the NNUE base eval unchanged. Align the `eval.rs` fallback for
consistency. (Alternatives rejected: B = NNUE-off HCE-only, weaker midgame,
regression risk; C = A+B, unnecessary risk.)

### Key quantitative basis (measured in losses)
The opponent's board-hoard advantage builds steadily and is visible well before
the end, while the bot's kazan lead stays flat:

| stage | my board | opp board | board diff (my−opp) | kazan lead |
|---|---|---|---|---|
| 25% | 35.6 | 36.1 | −0.5 | +14.6 |
| 50% | 23.3 | 28.3 | **−4.9** | +15.2 |
| 75% | 13.5 | 27.8 | **−14.4** | +16.0 |
| 90% | 7.7 | 29.5 | **−21.8** | +16.4 |
| end | 4.7 | 31.0 | **−26.3** | +16.1 |

Decisive identity: **final margin (≈ −10) ≈ kazan_lead (+16) + board_diff (−26)**
— exactly the sweep rule (board stones get scooped into each side's kazan). So a
term `(my_board − opp_board) × w`, with `w` ramped toward `MATERIAL_WEIGHT` as
the board empties, makes the endgame material estimate converge to the true
terminal evaluation, and it corrects progressively (already −5 at 50%, −14 at
75%, −22 at 90%). This is the principled fix, not a tuned patch.

### Core idea: endgame material = projected final kazan
Under the sweep rule, as the board empties, an own-side stone converges to the
value of a kazan stone. Replace the "starve/finish" bonuses with a single
dominant **sweep term** plus retained light mobility:

```
let board_total = my_stones + opp_stones;            // both sides' pits
if board_total <= ENDGAME_THRESHOLD (≈60) {
    // weight ramps from ~ (near PIT_STONES baseline) up toward MATERIAL_WEIGHT
    // as the board empties — own-side stones become future kazan.
    let sweep_w = ramp(board_total);                  // e.g. 3 .. ~18, monotonic
    let sweep_term = (my_stones - opp_stones) * sweep_w;

    // keep a small mobility term (tempo still matters in sparse endgames)
    let mobility = (my_active - opp_active) * MOBILITY_SMALL * scale(board_total);

    base + sweep_term + mobility
} else { base }
```

- `ramp(board_total)`: monotonic decreasing in board_total. Tunable; start from
  the intuition "at ~10 stones left, own-side stone ≈ kazan stone (~18-21); at
  ~60 stones left, ≈ current baseline (~3)." Exact knots to be set during
  implementation and A/B tuning.
- Remove the inverted `starvation` bonus and `finish_bonus` keyed on
  `opp_stones <= N` (they reward the opposite of the losing dynamic). Keep a
  kazan-proximity nudge only if A/B shows it helps; default: drop it to isolate
  the sweep term's effect.
- Sign/perspective: all terms from side-to-move POV, consistent with existing
  `eval()`.

### Fallback (`eval.rs`) alignment
Make `PIT_STONES_WEIGHT` phase-dependent with the same ramp intuition (or add an
endgame sweep term mirroring the above), so the handcrafted path is consistent.
Lower priority than the search.rs path (NNUE is active in production) but done
for correctness.

### Units / boundaries
- Single, self-contained change inside `Searcher::eval()` and `eval.rs::evaluate()`.
- The ramp is a small pure helper (easy to unit-test: monotonic, bounded,
  endpoints correct).
- No change to search, move-gen, NNUE, EGTB, or board rules.

## Validation
1. **Build** a new binary (e.g. `engine/target/release/togyzkumalaq-engine`),
   leave `models/engine/baseline` untouched.
2. **ab_match gate**: `tools/ab_match.py <new> models/engine/baseline 100 1500`
   — require new ≥ 50% (no regression), target ≥ 55% with positive Elo.
3. **Blindness re-test** (the direct check that we fixed the *cause*): re-run the
   62-loss eval probe (`/tmp/tk_evaltest.py`, adapted to the new binary).
   Expected, given the board-diff trajectory: **at 75% the eval should flip from
   "winning" to losing/contested in most lost games** (board diff −14 there is
   the strongest, cleanest signal), at 90% near-universally losing, and at 50% a
   clear downward shift from the +128 median (board diff only −5 there, so a
   partial — not full — correction is expected and acceptable; the residual 50%
   over-optimism is an NNUE-judgment issue for Phase 2).
4. **Unit tests**: ramp monotonic/bounded; a crafted "I am +15 kazan but
   opponent hoards 30 on board, my side near-empty" position should now evaluate
   as bad/contested for the side that is structurally lost.

## Rollback / safety
- Production binary `models/engine/baseline` is not modified in Phase 1.
- The change is a localized eval edit; reverting is a single git revert.
- Promotion to `models/engine/baseline` happens only after the ab_match gate
  passes and the user explicitly approves (it is the live web + PlayOK engine).

## Out of scope (future phases, tracked in memory)
- Phase 2: regenerate sweep-labeled self-play data (`datagen.rs`) + retrain NNUE.
- Phase 3: tuzdyk-race incentives + second-player opening prep.
- Phase 4: search tuning (early-exit budget, endgame depth).
