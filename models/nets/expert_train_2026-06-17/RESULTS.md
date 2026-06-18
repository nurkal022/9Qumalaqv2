# Track 2 — Expert-data NNUE training (2026-06-17)
Goal: fix the endgame ceiling using real strong-human games (sweep-correct labels = PlayOK results).
Data: playok_elo1600.bin (1.06M pos, >=1600) ; playok_elo2000.bin (135k pos, both >=2000, extracted via convert_playok.py min_elo=2000). eval=0 -> trained lam=0 (pure outcome).

## TRUSTWORTHY metrics only (vs HCE = independent ref; conversion@80 = endgame on real-loss positions)
| net | training | vs HCE | conversion@80 |
|-----|----------|--------|---------------|
| baseline (deployed) | - | +29 (54.2%) | 25% |
| E1 | scratch, >=1600, lam0 | +24 (53.5%) | 27.5% |
| E2 | scratch, >=2000, lam0 | +28 (54.0%) | **40.0%** |
| E6 | finetune baseline on >=2000, LR3e-4/40ep | -9 (48.7%) | 27.5% |
| E7 | finetune baseline on >=2000, LR5e-4/80ep | +2 (50.3%) | 36.2% |

## Findings
1. ELITE >=2000 data genuinely TEACHES the endgame: conversion 25% -> 40% (E2), 36% (E7). Reproducible, on independent positions. The endgame ceiling IS breakable.
2. NO candidate is clearly STRONGER OVERALL vs the independent HCE (all +/- noise around baseline's +29). The small NNUE (256->32->1) TRADES general<->endgame: pushing endgame conversion tends to cost general strength. Capacity-limited.
3. E2 is the best single candidate: highest conversion (40%) AND baseline-level general (HCE +28).
4. head-to-head vs baseline is a BROKEN metric (overfit/non-transitive: gave fake +140..+512 across experiments that never replicate vs HCE). IGNORE it.

## Verdict
- Real progress: endgame is learnable from elite data.
- Real limit: small NNUE can't be best-general AND best-endgame at once -> need a BIGGER eval net to hold both (the genuine next frontier).
- E2 = candidate "baseline-general + better-endgame"; only a LIVE test vs strong humans can confirm if the endgame gain beats the players that beat baseline.
- NOTHING promoted; models/engine/baseline UNCHANGED.
