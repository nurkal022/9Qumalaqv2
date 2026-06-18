# Sweep-correct NNUE retrain (2026-06-17, overnight)
Data: 463,331 positions from 4000 games of baseline self-play, sweep-correct result labels (datagen, RECORD_SIZE=27; trainer patched 26->27).
Arch: 40 -> 256 -> 32 -> 1 (matches deployed). Trainer: archive/engine-experiments/train_nnue_v2.py.
Baseline endgame conversion (real-loss +15 positions, self-playout @1000ms): 17.5%.

| cand | lam | epochs | conversion | A/B nobook | A/B with-book |
|------|-----|--------|-----------|-----------|---------------|
| baseline | - | - | 17.5% | - | - |
| c1 | 0.30 | 100 | 30.0% | +124 (67.1%) | - |
| c2 | 0.15 | 100 | 30.0% | +147 (70.0%) | **+140 (69.2%)**  <-- WINNER |
| c3 | 0.00 | 100 | 22.5% | - | - |
| c4 | finetune 0.2 | 40 | 22.5% | - | - |
| c5 | 0.15 (512h) | 100 | 17.5% | - | - |
| c6 | 0.10 | 150 | 32.5% | +124 (67.1%) | +86 (62.1%) |
| c7 | 0.50 (control) | 100 | 25.0% | - | - |
WINNER = c2 (lam 0.15). NOT promoted to prod (needs user approval). models/engine/baseline untouched.

## Iteration 2 (combined ~1.04M positions: 463k baseline-selfplay + 467k c2-selfplay + 109k c2-endgame)
| cand | lam | data | conversion(n=40) | A/B vs baseline (book) | h2h vs c2 (book) |
|------|-----|------|------------------|------------------------|------------------|
| v2a | 0.15 | 1M combined | 17.5% (noisy) | **+398 (90.8%)** | **95.0% (+512)** |
| v2c | 0.10 | 1M combined | 27.5% | +223 (78.3%) | 93.8% (+470) |
| v2b | 0.15 | 1M + endgame x3 | 25.0% | - | - |

SANITY CONFIRMED: both engines load NNUE; baseline-vs-baseline mirror = 40% (16-24, within noise of 50%, n=40) — a ~10% wobble cannot explain 90%+ candidate winrates. Wins are REAL.
ROBUST ORDERING: v2a > v2c > c2 > baseline (all large). Exact Elo unreliable (win% saturation).
OPEN: A/B opponent = old endgame-weak engine; conversion (goal metric vs hoarding humans) being re-measured at 4 seeds. EXACT-ELO numbers not trustworthy at extremes — the claim is "much stronger", magnitude TBD vs strong humans (live PlayOK test recommended).
PROMOTION: candidate = v2a (or v2c). NOT promoted — models/engine/baseline untouched, awaiting user approval.

## FINAL VERDICT (lineage-independent test — CORRECTS the above)
Tested all nets vs an OUT-OF-LINEAGE opponent (HCE-only engine = no nnue_weights.bin):
| net | vs baseline (in-lineage) | vs HCE (INDEPENDENT) | conversion (80 trials) |
|-----|--------------------------|----------------------|------------------------|
| baseline | - | +29 (54.2%) | 25.0% |
| c2  | +140 (69.2%) | +6 (50.8%) | 27.5% |
| v2a | +398 (90.8%) | +41 (55.8%) | 21.2% |
| v2c | +223 (78.3%) | (not run) | 22.5% |

CONCLUSION: vs an independent opponent ALL nets are statistically equal (~+6..+41 over HCE, within 1 sigma at n=120). The big in-lineage A/B gains were OVERFIT (nets trained on baseline/c2 self-play -> learned to beat those specific engines). Endgame conversion also flat (~21-27%, noise). The earlier "17.5%->30% conversion" was 40-trial small-sample noise (baseline at 80 trials = 25%).
=> Self-play NNUE retraining did NOT genuinely improve the engine. NNUE 256->32->1 ceiling confirmed. NOTHING promoted; models/engine/baseline unchanged.
FORWARD: train on real 2000+ expert games (sweep-relabeled), lam~0 — only way past the self-distillation ceiling.
