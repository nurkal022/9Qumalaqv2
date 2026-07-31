# Design: an engine that beats 9qum's net

Date: 2026-07-31
Status: approved design (implementation plan to follow)
Goal: beat 9qum.com's live bot **ИИ 9qum (rating 2209)** and their strong ladder players.

## Where we actually stand (measured this session, not assumed)

Everything below was measured against 9qum's own referee and their own net, i.e. outside our
own lineage — the failure mode that produced fake "+140 Elo" results in June.

**Head-to-head.** Our production engine (`models/engine/baseline`, alpha-beta + NNUE + EGTB,
1000 ms/move) vs their net v400 at 90 sims, 48 games over 22 balanced opening lines played
from both sides: **14W-2D-32L = 31.2%, Elo −137**. We score 43.8% moving first and 18.8%
moving second; from the *same* opening the first player scores 43.8% for us and 81.2% for
them. **25 of the 32 losses had our kazan ≥70** — close games lost late.

**Evaluation quality**, on 407k plies of 4.1k played-out games (`tools/9qum/validate_labels.py`),
split by game so outcomes cannot leak:

| predictor | Brier | logloss | accuracy |
|---|---|---|---|
| constant 0.5 | 0.2500 | 0.6931 | — |
| kazan difference | 0.2131 | 0.6117 | 65.3% |
| our engine (100 ms search) | 0.1951 | 0.5735 | 75.4% |
| **their net** | **0.1109** | **0.3435** | **83.7%** |

Correlation between our eval and theirs is 0.653, so ~43% of their signal is information we
do not already have. Their labels are calibrated (≤6 pp deviation across all ten bins),
unambiguously the first player's win probability (82.0% vs 49.9% under the other reading),
and *more* accurate on games played after their training run started (83.4% vs 74.2%) — so
they generalise rather than memorise.

**The diagnosis, quantified.** A kazan lead of ≥20 after ply 80 still loses **27.3% of the
time** (8298/30369 positions). Accuracy by phase:

| positions | kazan sign | our engine | their net |
|---|---|---|---|
| close endgame (ply≥80, \|Δk\|≤8) | 55.0% | 77.0% | 91.2% |
| clear endgame (ply≥80, \|Δk\|≥20) | 72.7% | 76.1% | **94.9%** |
| midgame (40–80) | 70.4% | 78.3% | 85.5% |

Our accuracy is **flat at 76–78% across every phase** and barely above raw material in the
endgame; theirs sharpens as the position resolves (85 → 91 → 95%). We believe a material
lead; their net knows when that lead is a trap. This is the same mechanism as the PlayOK
collapse (led ≥10, lost the endgame) and as the 25/32 match losses.

**About them.** Their AI ladder is one net plus a sims count (III 40 … ЗМС 1280); free tiers
are deliberately weakened (`mix 0.22, drop 0.12` at level I) and the server clamps requested
sims to the level even on analysis boards. Their public `/api/train_status` shows their
AlphaZero loop at iteration 8 with **0 of 7 candidates passing the gate** (46.2/45.4/44.6/
47.2/43.5/45.7/43.8%) on 32 cores 24/7 — out-computing them on self-play is not the winning
axis (source: their public `/api/train/status`). Their product is capped at 1280 sims; our
inference is not capped.

## Assets available

- **754k positions** with full move records from 8098 real games (`data/9qum/games/`)
- **their calibrated win% per ply**: 6325 games (~600k plies) on disk, backfilling towards all
  8337 games we hold replays for (`data/9qum/analysis/curves.jsonl.gz`). The numbers in this
  document were measured on the first 4088 games.
- **teacher on demand**: `ai/think` returns visit counts + Q + priors for *any* position we
  construct — ~34k targets/day at their rate limit, i.e. policy and value together
- their opening tree with winrates (1518 nodes over 16.8k games)
- a free out-of-lineage benchmark (`tools/9qum/match.py`) and the 2209 bot as the final target
- our board code, validated against their referee ply-for-ply on 22,414 plies

## Approach: A then B

**A — fix the evaluation inside our existing alpha-beta engine.** Fast, and it attacks the
measured gap directly.
**B — the policy+value+score MCTS net.** Raises the ceiling and exploits sims scaling, which
their product cannot follow.

Transition from A to B when either A reaches ≥55% on the working gate, or A's gate stops
moving by +8% per phase. Data pipeline, labels, monitors and gates are shared, so nothing is
thrown away at the transition.

## Gates and metrics

Architecture-independent; these decide every promotion.

- **Working gate:** `tools/9qum/match.py` vs their net at 90 sims, balanced opening suite,
  48 games for a quick read (±7%), 96 for a decision (±5%). Baseline 31.2%, promote at **≥55%**.
- **Control gates:** `tools/ab_match.py` vs `models/engine/baseline` and vs HCE. In-lineage,
  but they catch regressions — in June they exposed that "+140 Elo" was self-lineage overfit.
- **Reserved openings** never used in gating, kept for final validation only.
- **Cheap monitors** (no API calls, minutes per iteration) — the table to beat. "Value accuracy"
  means: how often the sign of the predicted win probability matches the actual result of that
  game, over held-out games, restricted to the stated bucket. "Policy match-rate" means: how
  often the top move equals the move a ≥2000-rated human played, with top-3 in brackets; ours
  is measured in A0 on the same corpus so the comparison is like-for-like.

| monitor | their net | ours today | target |
|---|---|---|---|
| value accuracy, midgame 40–80 | 85.5% | 78.3% | ≥86% |
| value accuracy, close endgame (ply≥80, \|Δk\|≤8) | 91.2% | 77.0% | ≥91% |
| value accuracy, clear endgame (ply≥80, \|Δk\|≥20) | 94.9% | 76.1% | ≥95% |
| Brier on held-out games | 0.111 | 0.195 | ≤0.11 |
| policy match-rate vs strong humans | 51.6% / 84.2% top-3 | measured in A0 | ≥52% |

Held-out sets are split **by game, never by ply** — plies inside a game are autocorrelated
and a per-ply split leaks the outcome.

**Stopping rule per phase:** if the working gate has not moved by at least +8% over 96 games,
the phase is recorded as a negative result and is not extended "just a bit further" (the June
lesson: 158 iterations of one pipeline with no gain).

## Phases and acceptance

Estimates assume the local RTX 5080 working continuously. Every phase ends in a decision point.

| phase | work | acceptance |
|---|---|---|
| **A0** ~½ day | Record format (board + policy + value + score + per-source masks) and the converter `replays + curves → bin`. Measure our policy match-rate. | Converter asserts green; baseline monitors recorded (78.3 / 77.0 / 76.1%) |
| **A1** 1–2 days | Sparse bucketed input, 1024 accumulator, phase output buckets in Rust; trainer for the new scheme. | Monitors ≥86 / 91 / 95%; NPS no worse than 2×; `ab_match` vs baseline ≥55% **at equal time**; 48-game gate |
| **A2** 1–2 days | Data iterations: source weights (their labels vs human outcomes), bucket count, LR/epochs. | Best candidate through a 96-game gate; transition at ≥55% or when the gate stalls |
| **B0** 1–2 days | Score head + value on their labels for the MCTS net (`sup1500` as the policy start). | Same monitors plus the gate; policy match-rate ≥52% |
| **B1** parallel | Teacher on demand: our own positions, endgames first, → `ai/think` → policy+value targets, ~34k/day, cached. | Monitor gains specifically in the endgame buckets |
| **B2** onwards | MCTS at 800–1600 sims with keep-best gating; self-play only after ≥55%. | 96-game gate, monotone improvement |
| **C** final | Ladder bridge: live games vs ИИ 9qum (2209) and their strong humans. | On an account agreed with Eldar and marked as a bot |

The implementation plan covers **A0–A2**. Phase B gets its own plan written at the transition,
when we know what A did and did not fix — planning it now would be guesswork.

## Phase A design

**Root cause of the eval ceiling is the input encoding, not the width.** `build_input_40` in
`engine/src/nnue.rs` feeds each pit's stone count as a single scaled scalar
(`pits[i] * SCALE / 50`). Togyzkumalak endgames turn on exact counts and parity: a pit holding
exactly 2 stones is a tuzdyk threat, parity of a sow decides captures, a race is decided by
tempo. A first layer over a scalar can only rescale it, so those step functions are close to
unrepresentable — which is exactly what the flat 76–78% accuracy shows. The 58-input variant
patched symptoms with 18 handcrafted counters (`opp pits with exactly 2 stones`, `even
stones > 0`). June's conclusion "NNUE 256→32→1 is at its ceiling" was right about the fact and
wrong about the cause: the ceiling is **40 dense inputs**, and no width fixes that.

Changes:

1. **Sparse bucketed input.** Per pit (18), one-hot over count buckets
   `{0,1,2,3,4,5,6,7,8,9,10-12,13-16,17-24,25+}` = 252 features; kazan in buckets (~40);
   tuzdyk one-hot (20); side to move (1). ≈313 binary features, ~40 active. The net can finally
   see "exactly 2", "even", "empty" as distinct entities.
2. **Accumulator 1024** instead of 256, still incrementally updated (the first layer is a sum
   over active features, so a move toggles ~20 features rather than forcing a full recompute).
   Modest by chess-NNUE standards (768→2×1024 at millions of NPS).
3. **Phase output buckets** — 4 buckets by total stones on the board. This is the direct answer
   to June's finding that improving the endgame cost general strength: the endgame gets its own
   output weights instead of competing with the opening.
4. **Targets:** their 400k calibrated win% (converted to side-to-move perspective) plus 754k
   human outcomes under the sweep-correct referee, split by game.
5. **Hard speed requirement:** no worse than ~2× NPS loss, verified by `ab_match.py` at **equal
   time per move**, not equal depth. A net that evaluates better but loses strength through lost
   depth fails the phase.

The NNUE binary header already carries `input_size`, so this becomes a new format version;
existing weight files keep loading.

## Phase B design

1. **B0** — start from the existing `sup1500` policy net (43.8% @1-ply), add a value head
   trained on their out-of-lineage labels and a **score head** predicting the final kazan
   difference (their `score_loss`; a dense signal precisely where value saturates). The score
   head is an auxiliary training loss only — it does not enter the search.
2. **B1** — teacher on demand: sample positions from our own play, preferring endgames and
   positions where our value is uncertain, query `ai/think`, store visit distribution as the
   policy target and Q/value as the value target. Cached, with a hard daily request cap.
3. **B2** — MCTS at 800–1600 sims with keep-best gating. Self-play is enabled **only** once the
   gate is ≥55%: it is proven here that self-play cannot bootstrap from a weak net (17% net
   produced no improvement signal), while gating reliably prevents degradation.

## Phase C — final validation

A bridge to their ladder (reusing `match.py`'s position mapping plus the websocket table
protocol) to play ИИ 9qum (2209) and their strong humans. Their ladder is rated, so our bot
moves real players' ratings: this runs on an account agreed with Eldar and marked as a bot.

## Risks, each with its check

| risk | check |
|---|---|
| overfitting to their net, since it is our gate | `ab_match` controls vs baseline and HCE; reserved openings excluded from gating |
| their labels' ceiling (~2000-level practical win%) | watch for monitors flattening exactly at their 85/91/95 — that means we hit the teacher, and only the score head and self-play go further |
| a bigger net costing NPS and therefore depth | `ab_match` at **equal time**, not equal depth |
| perspective error in the converter (today's class of bug) | assert on a sample: value sign matches the outcome ≥80%, else the converter fails |
| their net changes (v400 → newer) | `net_version` stored with every label; on change, re-measure the gate, otherwise we compare against different opponents |
| self-play degradation | keep-best gating, already built and shown not to degrade |
| an inverted-sign class of bug in search | regression test "more depth is not worse than less" on a tactical suite, in CI |
| load on their API | daily request cap, cache, single connection |

## Testing

- **Converter** — round-trip (a decoded record equals the replay state), value perspective,
  and no leakage: train/val split by game.
- **Rust NNUE** — incremental accumulator after a sequence of moves equals a full recompute;
  old weight format still loads; the existing 14 tests stay green.
- **Search** — "more depth is not worse" regression test.
- **CI gate** — `ab_match.py --min 45` as the regression guard (it would have caught both past
  regressions).

## Tooling already in place

`tools/9qum/`: `harvest.py` (corpus, labels, opening tree, their telemetry), `match.py` (the
out-of-lineage gate), `validate_labels.py` (label validity), `report.py` (inventory),
`ws_tournaments.js`, `README.md` (endpoint and format map). Corpus in `data/9qum/` (gitignored).
