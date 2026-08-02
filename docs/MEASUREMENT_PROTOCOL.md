# Measurement protocol

Rules for measuring engine strength in this project. Every one of them was learned by getting
a wrong answer first; the cost of each mistake is noted so nobody relaxes a rule for
convenience. Read this before trusting or quoting a number.

## 1. An in-lineage match proves nothing about strength

Beating our own previous engine measures how well a candidate exploits *that* engine, not how
strong it is. Measured 2026-08-02: `v2_e12_acc256` beat `models/engine/baseline` **78% (Elo
+220)** over 100 games at equal time, and against an independent opponent in the same window it
scored **12.5%** — exactly what the baseline scored. The gain was entirely non-transitive.

This project has now produced three such phantom gains (June's "+140/+398 Elo" NNUE self-play
retrain, and this one). Use `tools/ab_match.py` only as a **regression guard**, never as
evidence of improvement.

## 2. Only paired measurements inside one window count

The external opponent is a live service and is not a constant. On 2026-08-01/02 their level-I
configuration changed from `mix 0.22, drop 0.12` to `mix 0.5, drop 0.25` mid-experiment, and
our production engine's score on the identical suite fell from 31.2% to 12.5%. Any comparison
against a number from an earlier session is void.

So: measure every engine you want to compare **back to back**, same code, same suite, same day,
and record the opponent's reported configuration with every game (`match.py` stores
`level_mix`, `level_drop`, `net_version` per record — check they match across the runs you
compare).

## 3. The opening suite is balanced for colour, not for difficulty

`match.py --opening-plies 4` draws lines from the harvested opening tree **ordered by
popularity**, and their net is best trained on the popular ones. A truncated run therefore
samples the hardest openings. Measured: 9 games gave 0%, 24 games of the same engine gave
12.5%, and the production engine gave 8.3% on the first three openings against its own 31.2%
average over twelve.

Run the **whole** suite, or an explicitly stated subset, and never compare a short run to a
long one. Each line must be played from both colours.

## 4. Sample sizes

At n=6 the standard error is about 19 percentage points; 0/6 versus 2/6 is noise. Minimums:

| purpose | games |
|---|---|
| smoke test that the harness works | 2–6 |
| a direction worth investigating | 24 |
| a promotion decision | 96 |

A 24-game difference of less than ~12 points is not a result.

## 5. Offline evaluation metrics do not measure strength

The monitors score how often the sign of a predicted win probability matches the game's
outcome. That is *classification of the eventual winner*, and a net can be excellent at it while
being useless to a search. Measured: `acc256` beat the opponent's own reference in the midgame
(88.2% vs 86.7%) and gained zero Elo externally, because a saturating win-probability eval
cannot distinguish winning by 2 from winning by 20 — and in this game the stone margin *is* the
win condition. Replaying the games showed the mechanism: the old engine builds a +17 kazan lead
and loses it late; the new one never built a lead at all.

Treat monitors as a **fast screen** that can reject a candidate cheaply. Only games decide.

## 6. Compare evaluators only on identical positions

Two evaluators measured on different position populations cannot be compared. Measured: the
same engine scored 76.1% and 84.7% on "clear endgame" depending on whether the sample came from
the whole corpus or the validation split. `tools/9qum/monitors.py` therefore samples once and
scores both our engine and the opponent's stored labels over that one sample, reporting the gap
and the label coverage.

Held-out sets are split **by game, never by ply** — plies inside a game are autocorrelated and a
per-ply split leaks the outcome.

## 7. Long measurements belong to the controller, in files

A subagent's shell backgrounds anything past its tool timeout and its stdout dies with its turn:
that lost a finished 50-minute 100-game match. Waiting on a process by text pattern
(`pgrep -f`) failed three times — the waiter matched itself, then a zombie waiter from an earlier
attempt matched the pattern and stalled a queued run for eight hours.

Use `tools/9qum/run_measurements.py`: it runs jobs sequentially in one process, writes each log
to a file, holds a lock so two runners cannot overlap, and records a manifest with exit codes so
an abandoned run is distinguishable from a finished one. Never build a test engine in `/tmp`
(one was wiped by a restart mid-run) — use `tools/9qum/make_test_engine.sh`, which installs into
a gitignored directory beside the weights and verifies the engine loads them.

## 8. Validate positions before using them as evidence

A scale-mismatch diagnosis was built on a hand-written position whose stones summed to 105
instead of 162 — physically unreachable, so the net's opinion of it meant nothing. Any hand-written position used as *evidence about strength or evaluation quality* must pass
`validate_position()` in `research/data/features_v2.py`: the 18 pits plus both kazans sum to
exactly 162, no negative counts, and each tuzdyk is a legal pit on the correct side.

The exception, deliberately: tests that check two implementations compute the same arithmetic
(the torch↔Rust equality test) may use unreachable positions, because what they assert is
numerical agreement, not the net's opinion. Legality matters when a number is being read as a
statement about the game.

## 9. Reproducibility

`research/training/train_nnue_v2.py` takes `--seed` and seeds Python, numpy and torch. Without
it, run-to-run val-loss noise was 0.001–0.002 — larger than the differences between four
candidates we compared, which made those comparisons uninterpretable. Always pass a seed, and
report it with the number.
