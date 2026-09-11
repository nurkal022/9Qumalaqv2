# Endgame: terminal rule, clock use, tempo-aware eval — design

Date: 2026-09-12
Status: approved (design), plan in `docs/superpowers/plans/2026-09-12-endgame-rules-tempo.md`

## Context & evidence (audit of 2026-09-12)

### 1. The core terminal rule is wrong

`core/src/board.rs::game_result()` ends the game as soon as **either** side has no stones.
The real rule (PlayOK, 9qum, official): the game ends only when **the side to move** has no
stones. A player who empties itself (lone stone from pit 9, or a lone stone into the
opponent's tuzdyk on its own side) does not end the game — the opponent must move once, and
if that move feeds stones back, play continues.

Evidence:

| source | observation |
|---|---|
| PlayOK bot games, ~110 self-empties | play continued 1 ply in nearly all; 10, 12, 17 and 33 more plies in five games |
| 9qum replays, 6000 games | 1120 unfinished states with one side empty and the other to move; **0** states with the empty side to move; 31 forced-feed continuations |
| our own code | `research/training/gen_engine_games.py:79` already uses the correct rule; the Rust core and the Python rules class (`archive/old-impls/alphazero-code/alphazero/game.py::_check_winner`) do not |

Consumers of the wrong rule: alpha-beta search (terminal + repetition), EGTB generation,
MCTS self-play / league / eval-vs-engine labels, Rust datagen, `tools/ab_match.py` (via the
Python rules class), every trainer that imports `game.py`. The bridge's "sweep fallback"
(`tools/playok/bridge.py`) exists only to paper over this: the engine says `terminal` while
the server still expects a move.

Human-game labels (PlayOK, 9qum) are unaffected — a real referee produced them.

### 2. What the losses actually look like

Final positions of the 63 recorded PlayOK losses vs 174 wins (bot = white):

| | losses | wins |
|---|---|---|
| opponent's largest pit (median) | 13 (≥15 in 30/63) | 6 |
| opponent board stones | 27 | 18 |
| bot board stones | 0 | 19 |
| total board stones at the end | 30 | 37 |

The loss is a **tempo endgame**: the bot leads in kazan, the opponent parks a hoard in one
pit and plays lone stones; the bot runs out of waiting moves first, empties, and the hoard
sweeps into the opponent's kazan. Only 3/63 losses end with ≤12 stones on the board, so an
EGTB indexed by stone count cannot reach this phase (the June 2026 spec measured the same).

### 3. What the evaluator is missing

- **Tempo.** Nothing counts waiting moves. A move from pit index `i` with `k` stones keeps
  every stone on our side iff `i + k - 1 <= 8`. A lone stone at index 0 is 8 tempi; at
  index 8 it is none; a hoard of 20 at index 3 is none (it is *locked*: it cannot move
  without crossing). The handcrafted eval counts non-empty pits ("mobility"), which treats
  a lone stone in pit 9 and a lone stone in pit 1 the same.
- **Locked material.** Stones in a locked pit stay on our side as long as we have tempo,
  so they are sweep material. The June 2026 attempt weighted *all* own-side stones as
  future kazan (`(my_board - opp_board) * w`) and regressed: movable stones are loans that
  cross over, not material. Only locked stones are.
- **Counting precision (v2 nets only).** NNUE v2 buckets kazan by 10 and pits ≥25 into one
  bucket; in a phase decided by a two-stone margin it cannot count. The production net
  (`models/engine/nnue_weights.bin`, legacy 40-input dense) has exact kazan inputs.

### 4. The clock is unused

`tools/playok/bridge.py` thinks 1.8 s per move on a 30-minute clock: ~2 minutes used per
150-ply game. Thinking already runs off the poll thread, so longer budgets are safe. In the
tempo endgame branching is 2–4 and the engine does ~9M nodes/s, so 10–20 s reaches depth
30+, frequently the end of the game.

## Goals

1. Rust core and the Python rules class implement the real terminal rule; EGTB regenerated
   under it; the bridge no longer needs the sweep fallback (kept as an alarm).
2. The bridge and the measurement harness spend more time when ≤40 stones are on the board,
   bounded so a 30-minute game cannot be lost on time.
3. A tempo/locked-material correction in the live evaluator, gated **externally** (paired
   9qum matches in one window per `docs/MEASUREMENT_PROTOCOL.md`), with a cheap offline
   screen (eval-at-50% separation of wins vs losses over all 244 recorded games, both
   classes, so no survivorship bias).
4. Production (`models/engine/baseline`) promoted only after the gate and explicit approval.

## Non-goals (this plan)

- Retraining any net (Phase 4 below).
- Opening book changes, tuzdyk race, midgame eval.
- Repetition-as-draw semantics in search (no evidence it matters; separate note).
- EGTB beyond 4 stones.

## Phase 4 (deferred — its own plan after Phase 3 is measured)

Retrain on the same human corpus with (a) target = expected final swept margin
(`kazan + own-side stones at the end`, sign-corrected for side to move) with win/loss as an
auxiliary head, and (b) exact scalar inputs for both kazans and per-pit counts alongside the
sparse buckets, plus the two tempo scalars from Phase 3 as inputs. Design decisions
(architecture width, loss weights, which corpus split) depend on what Phase 3 measures and
are not fixed here.
