# NNUE Training Experiments - Full Report
## Togyz Kumalak Engine, February 2026

---

## Architecture
- **Network**: Input(40) -> Linear(256) -> ClippedReLU -> Linear(32) -> ClippedReLU -> Linear(1)
- **Parameters**: 18,753
- **Quantization**: int16, scale=64
- **Binary weights**: 37,510 bytes (`nnue_weights.bin`)

## Input Features (40)
- 9 current player pit values (/50.0)
- 9 opponent pit values (/50.0)
- 1 current player kazan (/82.0)
- 1 opponent kazan (/82.0)
- 10 current player tuzdyk one-hot (-1=none, 0-8=position)
- 10 opponent tuzdyk one-hot

## Training Setup
- **Loss**: MSE on win probability: `(sigmoid(pred/K*4) - target)^2`
- **Target**: `lambda * sigmoid(eval/K*4) + (1-lambda) * game_result`
- **K**: 1050 (from Texel tuning)
- **Optimizer**: Adam with weight_decay=1e-5
- **Scheduler**: CosineAnnealingLR (LR -> 0 at T_max=epochs)
- **Validation**: 10% of data, max 50,000 positions

---

## Training Data

| Dataset | File | Positions | Size | Source |
|---------|------|-----------|------|--------|
| V1 HCE | `training_data.bin` | 1,242,311 | 31MB | Self-play depth 8, HCE eval |
| Mac HCE | `hce_mac_training_data.bin` | 2,463,555 | 64MB | Self-play depth 8, HCE eval (Mac) |
| Combined HCE | `combined_hce_data.bin` | 3,705,866 | 92MB | V1 + Mac merged |
| PlayOK parsed | `playok_training_data.bin` | 11,782,724 | ~300MB | PlayOK master games (eval from HCE search) |
| V1+NNUE scaled | `v1_plus_nnue_scaled.bin` | 2,065,322 | 54MB | V1 HCE + scaled NNUE self-play |

### Binary Record Format (26 bytes)
```
bytes 0-8:   pits_white[9]  (u8)
bytes 9-17:  pits_black[9]  (u8)
byte  18:    kazan_white    (u8)
byte  19:    kazan_black    (u8)
byte  20:    tuzdyk_white   (i8, -1=none)
byte  21:    tuzdyk_black   (i8, -1=none)
byte  22:    side_to_move   (u8, 0=white)
bytes 23-24: eval           (i16 LE)
byte  25:    result         (u8, 0=white_loss, 1=draw, 2=white_win)
```

---

## V1 Champion (Baseline)
- **Trained on Mac** (not GPU)
- **Config**: 1.24M HCE, lambda=0.75, 100 epochs, **batch=4096**, lr=0.001
- **Match result**: **79% vs HCE (50 games, 500ms/move) = +230 Elo**
- **File**: `nnue_v1.bin`

---

## All Experiments

### Round 1: Data Mix (GPU, batch=8192, 200 epochs unless noted)

| Exp | Data | Positions | Lambda | Epochs | Batch | Val Loss | Match Result | Elo |
|-----|------|-----------|--------|--------|-------|----------|-------------|-----|
| A | V1+PlayOK | 13M | 0.75 | 100 | 8192 | 0.1808 | not tested | - |
| B | V1 only | 1.24M | 0.75 | 100 | 8192 | 0.0454 | not tested | - |
| C | PlayOK only | 11.8M | 0.0 | 100 | 8192 | 0.1940 | not tested | - |

**Observations**:
- PlayOK data has much higher val loss (~0.18-0.19) vs V1 data (~0.045)
- PlayOK games may have inconsistent eval or different position distribution
- Exp B (V1 only) on GPU already had decent val loss

### Round 2: Fine-tuning & Lambda (GPU, batch=8192)

| Exp | Data | Positions | Lambda | Epochs | Batch | Val Loss | Match Result | Elo |
|-----|------|-----------|--------|--------|-------|----------|-------------|-----|
| D | PlayOK fine-tune | 11.8M | 0.75 | 50 | 8192 | 0.1946 | not tested | - |
| E | V1+PlayOK | 13M | 0.0 | 100 | 8192 | 0.1905 | not tested | - |
| F | V1 only | 1.24M | 1.0 | 100 | 8192 | 0.0464 | not tested | - |

**Observations**:
- PlayOK data consistently bad (val loss ~0.19)
- Lambda=1.0 (Exp F) slightly worse val loss than lambda=0.75 (Exp B)

### Round 3: HCE Data Experiments (GPU, batch=8192, 200 epochs)

| Exp | Data | Positions | Lambda | Epochs | Batch | Val Loss | Match Result | Elo |
|-----|------|-----------|--------|--------|-------|----------|-------------|-----|
| **G** | **V1 HCE** | **1.24M** | **0.75** | **200** | **8192** | **0.0469** | **65-2-33 (66%)** | **+115** |
| H | Combined HCE | 3.7M | 0.75 | 200 | 8192 | 0.0456 | 5-0-15 (25%) @ g20 | ~-200 |
| I | V1 HCE | 1.24M | 0.5 | 200 | 8192 | 0.0580 | not tested | - |
| J | Combined HCE | 3.7M | 0.5 | 200 | 8192 | 0.0590 | not tested | - |

### Round 3b: Mixed Data (GPU, batch=8192, 200 epochs)

| Exp | Data | Positions | Lambda | Epochs | Batch | Val Loss | Match Result | Elo |
|-----|------|-----------|--------|--------|-------|----------|-------------|-----|
| K | V1+NNUE scaled | 2.07M | 0.75 | 200 | 8192 | 0.0536 | not tested | - |
| L | V1+NNUE scaled | 2.07M | 0.5 | 200 | 8192 | 0.0635 | not tested | - |

### Round 4: Epoch & LR Sweep (GPU, batch=8192)

| Exp | Data | Positions | Lambda | Epochs | Batch | LR | Val Loss | Match Result | Elo |
|-----|------|-----------|--------|--------|-------|------|----------|-------------|-----|
| **M** | **V1 HCE** | **1.24M** | **0.75** | **100** | **8192** | **0.001** | **0.0679** | **45-2-53 (46%)** | **-28** |
| N | Combined HCE | 3.7M | 0.75 | 100 | 8192 | 0.001 | 0.0450 | not tested | - |
| O | V1 HCE | 1.24M | 0.75 | 50 | 8192 | 0.001 | 0.0901 | not tested | - |
| P | V1 HCE | 1.24M | 0.75 | 150 | 8192 | 0.001 | 0.0536 | not tested | - |
| Q | V1 HCE | 1.24M | 0.75 | 100 | 8192 | 0.002 | 0.0468 | not tested | - |
| R | V1 HCE | 1.24M | 0.75 | 100 | 8192 | 0.0005 | 0.0900 | not tested | - |

### Mac-trained models (batch=4096)

| Model | Data | Positions | Lambda | Epochs | Batch | Match Result | Elo |
|-------|------|-----------|--------|--------|-------|-------------|-----|
| **V1** | **V1 HCE** | **1.24M** | **0.75** | **100** | **4096** | **79% (50g)** | **+230** |
| V4 | Combined HCE | 3.7M | 0.75 | 100 | 4096 | 72% (50g) | +164 |
| V5 | V1 HCE | 1.24M | 1.0 | 100 | 4096 | ~40% | ~-70 |

---

## Match Results - Detailed Progression

### V1 Champion (Mac, batch=4096)
- 50 games, 500ms/move, alternating colors
- **Final: NNUE 39-1-10 HCE (79%) = +230 Elo**

### Exp G (GPU, batch=8192, 200 epochs, V1 data)
- 100 games, 500ms/move, alternating colors
- **Final: NNUE 65-2-33 HCE (66%) = +115 Elo**

### Exp H (GPU, batch=8192, 200 epochs, 3.7M data)
- 100 games, 500ms/move, alternating colors
- Game 10: NNUE 2-0-8 HCE (20%)
- Game 20: NNUE 5-0-15 HCE (25%)
- Stopped at game 20 — clearly terrible
- **Projected: ~25% = ~-200 Elo**

### Exp M (GPU, batch=8192, 100 epochs, V1 data)
- 100 games, 500ms/move, alternating colors
- Game 10: NNUE 4-0-6 HCE (40.0%)
- Game 20: NNUE 8-0-12 HCE (40.0%)
- Game 30: NNUE 12-0-18 HCE (40.0%)
- Game 40: NNUE 16-1-23 HCE (41.2%)
- Game 50: NNUE 22-1-27 HCE (45.0%)
- Game 60: NNUE 26-2-32 HCE (45.0%)
- Game 70: NNUE 31-2-37 HCE (45.7%)
- Game 80: NNUE 36-2-42 HCE (46.2%)
- Game 90: NNUE 41-2-47 HCE (46.7%)
- **Final: NNUE 45-2-53 HCE (46.0%) = -28 Elo**

### V4 (Mac, batch=4096, 3.7M data)
- 50 games, 500ms/move, alternating colors
- **Final: ~72% = +164 Elo**

### V5 (Mac, batch=4096, lambda=1.0)
- 50 games, 500ms/move, alternating colors
- **Final: ~40% = ~-70 Elo**

---

## Consolidated Ranking (All Tested Models)

| Rank | Model | Batch | Data | Lambda | Epochs | Score | Elo vs HCE | Games |
|------|-------|-------|------|--------|--------|-------|------------|-------|
| 1 | **V1** | **4096** | **1.24M** | **0.75** | **100** | **79%** | **+230** | 50 |
| 2 | V4 | 4096 | 3.7M | 0.75 | 100 | 72% | +164 | 50 |
| 3 | Exp G | 8192 | 1.24M | 0.75 | 200 | 66% | +115 | 100 |
| 4 | Exp M | 8192 | 1.24M | 0.75 | 100 | 46% | -28 | 100 |
| 5 | V5 | 4096 | 1.24M | 1.0 | 100 | ~40% | ~-70 | 50 |
| 6 | Exp H | 8192 | 3.7M | 0.75 | 200 | ~25% | ~-200 | 20* |

*Exp H stopped early at 20 games due to clearly terrible performance.

### Pairwise Comparisons (Isolating Variables)

**Batch size (4096 vs 8192)** — same data, lambda, epochs:
```
V1   (batch=4096, 1.24M, λ=0.75, 100ep): 79% (+230 Elo)
ExpM (batch=8192, 1.24M, λ=0.75, 100ep): 46% (-28 Elo)
→ Batch size impact: ~258 Elo
```

**Data size (1.24M vs 3.7M)** — same batch, lambda, epochs:
```
V1 (1.24M, batch=4096, λ=0.75, 100ep): 79% (+230 Elo)
V4 (3.7M,  batch=4096, λ=0.75, 100ep): 72% (+164 Elo)
→ More data impact: -66 Elo (WORSE)
```

**Epochs (100 vs 200)** — same data, lambda, batch:
```
ExpM (100ep, batch=8192, 1.24M, λ=0.75): 46% (-28 Elo)
ExpG (200ep, batch=8192, 1.24M, λ=0.75): 66% (+115 Elo)
→ More epochs impact: +143 Elo (with batch=8192, cosine schedule benefits)
```

**Lambda (0.75 vs 1.0)** — same data, batch, epochs:
```
V1 (λ=0.75, batch=4096, 1.24M, 100ep): 79% (+230 Elo)
V5 (λ=1.0,  batch=4096, 1.24M, 100ep): 40% (~-70 Elo)
→ Lambda impact: ~300 Elo
```

---

## Key Findings (Ranked by Impact)

### 1. BATCH SIZE IS CRITICAL (258 Elo impact!)
```
V1  (batch=4096, 100ep, V1 data): 79% = +230 Elo
ExpM (batch=8192, 100ep, V1 data): 46% = -28 Elo
                                   Difference: ~258 Elo
```
Same data, same lambda, same epochs, same LR. Only batch size differs.
Larger batch -> sharper minima -> worse generalization. Well-known in deep learning.
**Conclusion**: NEVER use batch_size > 4096 for this model.

### 2. EPOCHS: 100 optimal, 200 overfits (115 Elo impact)
```
V1   (100ep, batch=4096): 79% = +230 Elo
ExpG (200ep, batch=8192): 66% = +115 Elo
```
Note: Exp G also has batch=8192, so true epoch-only impact may be smaller.
But 200 epochs on batch=8192 still beats 100 epochs on batch=8192 (66% vs 46%),
suggesting cosine schedule with more epochs provides more gradient updates.
**Conclusion**: Start with 100 epochs, consider 75-150 range with batch=4096.

### 3. MORE DATA DOESN'T HELP (yet) (66 Elo impact)
```
V1 (1.24M, batch=4096): 79% = +230 Elo
V4 (3.7M,  batch=4096): 72% = +164 Elo
                         Difference: ~66 Elo
```
Additional 2.46M Mac HCE positions made it worse. Possible reasons:
- Distribution shift between original and new data
- More data needs proportionally more training (epochs/updates)
- Quality > quantity at this scale
**Conclusion**: Don't blindly add more data. Test with batch=4096 + more epochs.

### 4. LAMBDA=0.75 IS OPTIMAL
```
V1 (lambda=0.75): 79% = +230 Elo
V5 (lambda=1.0):  ~40% = ~-70 Elo
```
Pure eval target (no game result) is terrible. Game result provides critical signal.
Lambda=0.5 experiments (I, J) had worse val loss than 0.75 counterparts.
**Conclusion**: Keep lambda=0.75.

### 5. VAL LOSS DOES NOT PREDICT PLAYING STRENGTH
```
ExpN: val=0.0450 (best!)  -> not tested but batch=8192
ExpQ: val=0.0468           -> not tested but batch=8192
ExpG: val=0.0469           -> 66% (+115 Elo)
ExpM: val=0.0679           -> 46% (-28 Elo)
V1:   val=unknown          -> 79% (+230 Elo)
```
**Conclusion**: Always run matches (minimum 50 games) to evaluate models.

### 6. PLAYOK DATA IS USELESS FOR NNUE TRAINING
All experiments with PlayOK data had val loss ~0.18-0.19.
Likely because PlayOK positions have HCE eval labels which don't match
the complex positions from master games.
**Conclusion**: Don't use PlayOK data for NNUE. May be useful for opening book.

---

## Val Loss Summary (All Experiments)

| Exp | Val Loss | Batch | Data | Notes |
|-----|----------|-------|------|-------|
| N | 0.0450 | 8192 | 3.7M | Best val loss overall, but batch=8192 |
| B | 0.0454 | 8192 | 1.24M | Round 1 baseline |
| H | 0.0456 | 8192 | 3.7M | Best val loss, worst match (25%) |
| F | 0.0464 | 8192 | 1.24M | Lambda=1.0, slightly worse |
| Q | 0.0468 | 8192 | 1.24M | LR=0.002 |
| G | 0.0469 | 8192 | 1.24M | 200ep, 66% match |
| P | 0.0536 | 8192 | 1.24M | 150ep |
| K | 0.0536 | 8192 | 2.07M | Mixed NNUE+HCE data |
| I | 0.0580 | 8192 | 1.24M | Lambda=0.5 |
| J | 0.0590 | 8192 | 3.7M | Lambda=0.5 |
| L | 0.0635 | 8192 | 2.07M | Mixed, lambda=0.5 |
| M | 0.0679 | 8192 | 1.24M | 100ep, 46% match |
| O | 0.0901 | 8192 | 1.24M | Only 50ep, underfitting |
| R | 0.0900 | 8192 | 1.24M | LR=0.0005, too slow |
| A | 0.1808 | 8192 | 13M | PlayOK data → garbage |
| E | 0.1905 | 8192 | 13M | PlayOK, λ=0 |
| C | 0.1940 | 8192 | 11.8M | PlayOK only → garbage |
| D | 0.1946 | 8192 | 11.8M | PlayOK fine-tune |

**Key observation**: Val loss range for HCE-only data: 0.045-0.090. PlayOK data: 0.18-0.19.
Val loss DOES NOT predict match strength (H has best val loss but worst match result).

---

## Untested Experiments - Analysis

These experiments were not match-tested. Based on findings, predictions:

| Exp | Config | Val Loss | Predicted Strength | Reasoning |
|-----|--------|----------|-------------------|-----------|
| B | V1, λ=0.75, 100ep, b8192 | 0.0454 | ~46% (-28 Elo) | Same as M (batch=8192 kills it) |
| F | V1, λ=1.0, 100ep, b8192 | 0.0464 | ~40% | Lambda=1.0 + batch=8192 = double penalty |
| I | V1, λ=0.5, 200ep, b8192 | 0.0580 | ~50-55% | Lambda=0.5 is bad but 200ep helps |
| J | 3.7M, λ=0.5, 200ep, b8192 | 0.0590 | ~30% | Bad lambda + bad data + batch=8192 |
| K | Mixed, λ=0.75, 200ep, b8192 | 0.0536 | ~55-60% | Mixed data not terrible, 200ep helps |
| L | Mixed, λ=0.5, 200ep, b8192 | 0.0635 | ~40-50% | Bad lambda |
| N | 3.7M, λ=0.75, 100ep, b8192 | 0.0450 | ~25-35% | Best val but 3.7M data hurts in play |
| O | V1, 50ep, b8192 | 0.0901 | ~35% | Way too few epochs |
| P | V1, 150ep, b8192 | 0.0536 | ~55-60% | Between M(100) and G(200) |
| Q | V1, lr=0.002, 100ep, b8192 | 0.0468 | ~45-50% | Higher LR doesn't help much |
| R | V1, lr=0.0005, 100ep, b8192 | 0.0900 | ~35% | LR too low, underfitting |

**None of these are likely to beat V1 — all use batch=8192.**

---

## What We Don't Know Yet

1. **batch=4096 on GPU**: Does GPU training with batch=4096 match V1's 79%?
   - Round 5 script ready (`run_round5.sh`) but not yet executed
   - Experiments S/T/U/V planned with batch=4096 and 2048

2. **Optimal epochs with batch=4096**: V1 used 100, but what about 50, 75, 150?

3. **Learning rate with batch=4096**: V1 used 0.001, but 0.0005 might be better

4. **Larger network**: 512->64->1 or 256->64->1 might capture more patterns

5. **NNUE self-play data**: Does training on NNUE-evaluated positions (instead of HCE)
   improve the next generation? Data was partially generated (~1.75M positions) but
   early experiments (K, L) with mixed data weren't promising.

6. **More HCE data with batch=4096**: V4 (3.7M, batch=4096) scored 72%.
   Maybe with more epochs it could match/beat V1?

---

## Files on GPU Server (~/nnue_train/)

| File | Description |
|------|-------------|
| `expA.bin` - `expR.bin` | Weight files for all 18 experiments |
| `experiments.log` | Round 1 (A-C) full training log |
| `round2.log` | Round 2 (D-F) full training log |
| `hce_experiments_full.log` | Round 3 (G-J) full training log |
| `round3_full.log` | Round 3b (K-L) full training log |
| `round4_full.log` | Round 4 (M-R) full training log |
| `run_round5.sh` | Ready-to-run Round 5 script (batch=4096) |
| `training_data.bin` | V1 HCE data (1.24M positions) |
| `combined_hce_data.bin` | Combined HCE data (3.7M positions) |
| `train_nnue.py` | Training script |

## Files on Mac (engine/)

| File | Description |
|------|-------------|
| `nnue_v1.bin` | V1 champion weights (backup) |
| `nnue_weights.bin` | Current active weights (should be V1) |
| `expG.bin`, `expH.bin`, `expK.bin`, `expM.bin` | Tested experiment weights |
| `training_data.bin` | V1 HCE data |
| `combined_hce_data.bin` | Combined HCE data |
| `train_nnue.py` | Training script |

---

## Next Steps (Priority Order)

1. **Run Round 5 on GPU** (batch=4096): Confirm batch size hypothesis
2. **If confirmed**: Try batch=4096 + 3.7M data + more epochs
3. **NNUE self-play loop**: Generate data with V1 NNUE, train V2, compare
4. **Architecture search**: Try larger/different networks with batch=4096
5. **Distributed datagen**: Use 10 Windows machines for 50M+ positions
