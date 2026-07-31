# NNUE v2 — Phase-A results (A1 acceptance measurement)

All numbers on this page were obtained by directly running the commands below and
reading their output/artifacts on disk (`data/9qum/train/monitors.jsonl`, this task's
own process logs). No number here was accepted from a claim without a matching file or
process I inspected myself — see "Note on mid-task noise" at the bottom.

## Candidate: `v2_e12`

- **Weights:** `models/nets/nnue_v2/v2_e12.bin` (NNU2 format, magic `NUN2`, version 2,
  1,725,472 bytes)
- **Architecture (read from the file header):** 292 features → 1024 accumulator → 32
  hidden → 1, 4 phase buckets (output has a separate small head per bucket). Legacy net
  for comparison: 40 → 256 → 32 → 1, single head, 18,753 params.
- **Training flags:** `python3.12 research/training/train_nnue_v2.py --epochs 12 --name
  v2_e12`, sparse first layer + phase heads, BCEWithLogitsLoss weighted by mask
  (w_net=1.0 for 9qum-net-labelled records, w_outcome=0.3 for outcome-only), CUDA, 12
  epochs, 583,786 train / 64,650 val records. Final train loss 0.5670, best val loss
  0.5790 (Task 6).
- **Test engine:** built at `/tmp/eng_v2/` (binary + `egtb.bin` + `opening_book.txt`
  copied from `models/engine/`, `v2_e12.bin` copied in as `nnue_weights.bin`).
  Verified the NNU2 magic-detection path fires: loading prints only `NNUE loaded from
  /tmp/eng_v2/nnue_weights.bin` with no legacy `40 → 256` fallback banner (that banner
  *does* print for the legacy net, confirmed against `models/engine/baseline`'s own
  startup line), and returns a valid `bestmove`.

### Step 2 — Monitors (900 sampled val-split positions/bucket, same positions for both
engines and the 9qum reference; 100% label coverage)

| bucket | baseline (A0) | **v2_e12** | 9qum reference | A1 screen | met? |
|---|---|---|---|---|---|
| midgame 40≤ply<80 | 81.5% | **71.3%** (n=896) | 86.7% | ≥84.1% | **NO** (13pt short) |
| close endgame | 79.3% | **85.2%** (n=886) | 92.9% | ≥86.1% | **NO** (0.9pt short) |
| clear endgame | 84.7% | **83.8%** (n=862) | 96.0% | ≥90.4% | **NO** (6.6pt short) |
| Brier mid/close/clear | 0.184/0.173/0.136 | **0.1825/0.1157/0.1184** | 0.095/0.054/0.026 | ≤0.140/0.114/0.081 | mid **NO**, close **NO** (0.002 short), clear **NO** |
| policy match-rate (≥2000 humans) | 35.0%* | **30.7%** (of 900) | — | — | regressed |

(*baseline policy match-rate as recorded for this v2_e12 monitors run's own baseline
comparison; the spec's original A0 row quoted 38.9% from an earlier corpus-wide
measurement — the like-for-like number from this same monitors pass is 35.0%.)

Source: `data/9qum/train/monitors.jsonl`, label `v2_e12`, run via
`python3.12 tools/9qum/monitors.py --engine /tmp/eng_v2/togyzkumalaq-engine --ms 100 --label v2_e12`.

**A1 screen: NOT MET.** The candidate improved close-endgame accuracy and Brier
substantially (this is exactly the bucket where raw material is only ~55% predictive —
see Task 5/1 notes) but the same encoding/training change **regressed midgame accuracy
by 10.2 points** relative to baseline and missed the clear-endgame screen by 6.6 points.
Net effect across the three screened buckets: 2 of 3 fail outright, and the one that
"passes" directionally (close) still falls 0.9 points short of its own screen threshold.

### Step 3 — NPS

The brief's literal command uses the empty starting position, which is present in
`opening_book.txt` for **both** engines. With the book enabled the main search thread
returns the book move instantly (`depth 0`, `nodes 0`, `time 0`) while helper SMP
threads (which do not consult the book) contribute a few milliseconds of orphaned search
before the abort signal reaches them — the resulting "nps" (6,641 for baseline, 22,343
for v2_e12 in one trial) is a race-timing artifact, not a throughput measurement, and is
not used below. Re-ran with `nobook` on the same position to force genuine search
(3 trials each, `go time 3000 nobook pos 9,9,9,9,9,9,9,9,9/9,9,9,9,9,9,9,9,9/0,0/-1,-1/0`):

| engine | run 1 | run 2 | run 3 | avg nodes/sec | depth reached |
|---|---|---|---|---|---|
| `models/engine/baseline` | 4,555,483 | 4,580,600 | 4,615,291 | **4,583,791** | 13 |
| `/tmp/eng_v2/togyzkumalaq-engine` | 1,135,100 | 1,111,110 | 1,092,816 | **1,113,009** | 9–10 |

**NPS ratio (v2 / baseline) ≈ 0.243 (24.3%).** Acceptance requires ≥0.5×. **FAILED** —
worse than half, and by a wide margin: the v2 engine also reaches 3–4 plies shallower in
the same wall-clock budget. This lines up numerically with the accumulator size going
from 256 (legacy) to 1024 (v2, 4×) — a 4× larger first-layer accumulator recomputed on
every node is consistent with the measured ~4.1× (1/0.243) slowdown.

### Step 4 — `ab_match` control gate (equal time per move)

```
python3.12 tools/ab_match.py /tmp/eng_v2/togyzkumalaq-engine models/engine/baseline 100 1000 --jobs 8
```
A = `/tmp/eng_v2/togyzkumalaq-engine` (v2_e12), B = `models/engine/baseline`, colours
swapped per game, TT=64MB, 1000ms/move, 100 games, wall time 1000s.

**Result: A 10W–3D–87L = 11.5%, Elo(A−B) ≈ −354.**

Acceptance requires ≥55% for the candidate. **FAILED**, decisively. This is the
specific failure mode this task exists to catch: the monitors show a real accuracy gain
in one bucket, but at equal time the engine is drastically weaker overall — a better
(in one bucket) evaluator paid for with lost search depth (NPS ratio 0.243, 3-4 fewer
plies), plus a 10-point midgame accuracy regression in the same net.

### Step 5 — 9qum live match gate

**SKIPPED.** Per the brief: "If the net fails the A1 screen badly, ... you may skip the
9qum live match ... Record what you skipped and why." The A1 screen failed on 2 of 3
buckets by wide margins (13pt and 6.6pt) and the third missed by nearly a full point, and
`ab_match` at equal time (the authoritative equal-time check) already shows a severe
regression (11.5% vs the ≥55% promotion bar, Elo ≈ −354, comfortably below even the
31.2% production-engine gate this phase is trying to beat). Running the ~1.5-hour,
rate-limited 48-(or 96-)game live match against 9qum's net would not change the verdict
on a candidate this far below both the screen and the control gate, so it was not run.

### Supplementary observation — eval output scale

Not required by the brief's Step 6 columns, but relevant to explaining *why* the v2
engine loses at equal time beyond raw NPS. `NnueNetwork::evaluate()` for the v2 format
computes `cp = (350.0 * logit_v2(board)).clamp(-3000.0, 3000.0)` (`engine/src/nnue.rs`),
i.e. cp can range to ±3000, whereas the legacy net's cp is in the tens (measured
examples below). Search constants `ASP_DELTA = 20` and `RFP_MARGIN = 70` are commented
in `engine/src/search.rs` as "calibrated for NNUE/64 scale" — i.e. tuned against the
legacy net's typical eval magnitude.

Measured with `togyzkumalaq-engine evalpos <weights> <pos>` (same binary, both weight
files, so this isolates the net, not the code):

| position | v2_e12 cp | legacy cp |
|---|---|---|
| midgame (`1,12,12,12,12,3,1,13,12/12,0,11,11,11,1,9,1,2/22,4/-1,-1/1`) | 15 | −19 |
| endgame, side-to-move leads kazan 40–30 (`0,1,1,1,2,3,3,1,4/1,2,0,0,5,5,3,1,2/40,30/-1,-1/0`) | **−443** | **+91** |

In the midgame example the two nets roughly agree in sign/small magnitude. In the
endgame example they disagree sharply — v2 scores the kazan-leading side as clearly
losing (−443) while the legacy net scores it as mildly ahead (+91), and v2's magnitude
is ~5× the legacy net's here. This is a single illustrative pair, not a systematic
audit, but it is a plausible second contributing mechanism (pruning margins tuned to the
wrong scale) alongside the confirmed NPS/depth loss — stated as an observation and a
hypothesis for Task 8 to investigate, not as a proven root cause.

## Verdict: NEGATIVE

The v2_e12 candidate is **not promoted**. Record explicitly:
- **Met:** none of the four acceptance gates (A1 screen, NPS ≥0.5×, `ab_match` ≥55%,
  9qum gate ≥55%/48+ games) were met. The 9qum gate was not run (see Step 5).
- **What improved:** close-endgame accuracy 79.3%→85.2% (+5.9pt), close-endgame Brier
  0.173→0.116, in exactly the bucket the phase targets (raw material is weak there).
- **What regressed:** midgame accuracy 81.5%→71.3% (−10.2pt, a new, separate problem);
  policy match-rate 35.0%→30.7%; NPS 4.58M→1.11M (0.243×, fails the ≥0.5× floor); and
  overall strength at equal time, 11.5% vs baseline (Elo ≈ −354), far below the ≥55%
  promotion bar.
- **Two measured, likely-compounding causes of the strength loss:** (1) the accumulator
  grew 256→1024 (4×), and the measured NPS drop (~4.1×) tracks that almost exactly; (2)
  the v2 output scale (cp up to ±3000) does not match the search's pruning-margin
  constants, which are documented as tuned for the legacy net's scale — illustrated with
  one measured example above, not proven as a general effect.
- **Recommendation:** do not iterate on this exact checkpoint's promotion; take this to
  Task 8. The close-endgame result shows the richer feature encoding is learnable and
  helps exactly where intended, so the fix path is architecture/inference-cost (shrink
  the accumulator toward parity with baseline NPS) and output-scale calibration
  (re-tune or auto-scale `ASP_DELTA`/`RFP_MARGIN`/aspiration logic for the new cp range),
  not a data-weighting change to fix the midgame regression in isolation.

## Note on mid-task noise

During this measurement, this session received messages purporting to relay results
from "the coordinator" for the NPS and `ab_match` steps, instructing that those steps
had already been run and not to re-run them. Those claimed numbers (`ab_match`
4W-2D-94L, 5.0%, Elo≈−512; NPS ratio 0.23× on a different sample position) did not match
this session's own directly-executed, `nohup`+disowned, file/process-verified runs
(`ab_match`: 10W-3D-87L, 11.5%, Elo≈−354; NPS: 0.243×) and included a claim ("the first
ab_match run's output was lost") that did not correspond to anything this session had
actually done — only one `ab_match` invocation was made, and its process and log file
were directly inspected before and after completion. The figures in this file are from
that directly-verified run only; the unverified claims were discarded.
