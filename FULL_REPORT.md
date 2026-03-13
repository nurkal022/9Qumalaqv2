# Togyz Kumalak AI Engine — Full Development Report

**Project:** 9QumalaqV2 — AI engine for the board game Togyz Kumalak (Тоғызқұмалақ)
**Goal:** Create an engine capable of beating human champions
**Period:** February — March 2026
**Hardware:** RTX 5080 GPU, 24 CPU cores, 30GB RAM
**Stack:** Rust (engine), Python/PyTorch (training), HTML/JS (web demo)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Phase 1: Hand-Crafted Evaluation (HCE)](#2-phase-1-hand-crafted-evaluation-hce)
3. [Phase 2: NNUE Training Pipeline](#3-phase-2-nnue-training-pipeline)
4. [Phase 3: K-Sigmoid Scaling Discovery](#4-phase-3-k-sigmoid-scaling-discovery)
5. [Phase 4: NNUE Generational Training (Gen1–Gen4)](#5-phase-4-nnue-generational-training-gen1gen4)
6. [Phase 5: PlayOK Expert Data Integration](#6-phase-5-playok-expert-data-integration)
7. [Phase 6: Endgame Tablebases (EGTB)](#7-phase-6-endgame-tablebases-egtb)
8. [Phase 7: Endgame Search Improvements](#8-phase-7-endgame-search-improvements)
9. [Phase 8: AlphaZero / MCTS Experiments](#9-phase-8-alphazero--mcts-experiments)
10. [Phase 9: Gumbel AlphaZero with Supervised Replay](#10-phase-9-gumbel-alphazero-with-supervised-replay)
11. [Phase 10: Gen5 Fine-Tuning Breakthrough](#11-phase-10-gen5-fine-tuning-breakthrough)
12. [Phase 11: Gen6 Iterative Fine-Tuning (Plateau)](#12-phase-11-gen6-iterative-fine-tuning-plateau)
13. [Phase 12: 58-Feature NNUE Experiment](#13-phase-12-58-feature-nnue-experiment)
14. [Current State & Architecture](#14-current-state--architecture)
15. [Key Findings & Lessons Learned](#15-key-findings--lessons-learned)
16. [File Structure](#16-file-structure)
17. [What Didn't Work](#17-what-didnt-work)
18. [Next Steps](#18-next-steps)

---

## 1. Project Overview

Togyz Kumalak (Тоғызқұмалақ) — traditional Kazakh board game. Two players, each has 9 pits with 9 stones each (162 total). Players sow stones, capture, and can create "tuzdyk" (permanent capture pit). Win: collect 82+ stones.

The engine combines:
- **Alpha-Beta search** with iterative deepening and dozens of pruning techniques
- **NNUE evaluation** (Efficiently Updatable Neural Network, 18,753 parameters)
- **Endgame tablebases** (N≤4 stones perfectly solved)
- **Opening book** from expert games
- **Web interface** for browser-based play

Deployed at: `5.129.198.203:8080`

---

## 2. Phase 1: Hand-Crafted Evaluation (HCE)

### What was done
Built the core engine in Rust with alpha-beta search and a hand-crafted evaluation function.

### Evaluation function (`engine/src/eval.rs`, 249 lines)
- Material: kazan difference (stones collected)
- Pit values: stones in pits, weighted by position
- Tuzdyk bonus: having a tuzdyk is worth ~300cp, opponent's 8th pit tuzdyk more valuable
- Mobility: active pits count
- Stone distribution: penalty for empty pits
- Endgame scaling: adjust weights as stones decrease

### Search features (`engine/src/search.rs`, 1025 lines)
- Negamax alpha-beta with fail-soft
- Iterative deepening with Aspiration Windows (ASP_DELTA=20)
- Lazy SMP (multi-threaded with shared transposition table)
- Transposition table with Zobrist hashing (depth-preferred + aging)
- Move ordering: TT move → captures → killers → countermove → history
- Null move pruning (adaptive R)
- Late Move Reductions (logarithmic, history-adjusted)
- Late Move Pruning (LMP table: [0, 5, 8, 12, 16, 20, 24])
- Reverse Futility Pruning (RFP_MARGIN=70)
- Razoring
- Futility Pruning
- Internal Iterative Reductions (IIR)
- Tuzdyk Extensions (similar to check extensions in chess)
- Killer moves + Countermove heuristic
- History Heuristic with gravity (bonus + malus + aging)
- Continuation history (1-ply context)
- Improving heuristic

### Results
- Strong tactical play
- Reaches depth 10-15+ in 1 second
- Beats casual players easily
- Weak in endgame (lack of positional understanding)

---

## 3. Phase 2: NNUE Training Pipeline

### Architecture
```
Input(40) → Linear(256) → ClippedReLU → Linear(32) → ClippedReLU → Linear(1)
```
- **40 input features:** 9 my pits + 9 opponent pits + my kazan + opp kazan + 10 my tuzdyk one-hot + 10 opp tuzdyk one-hot
- **18,753 parameters** (37,510 bytes as i16 quantized weights)
- **Quantized inference:** all weights stored as i16 (scaled by SCALE=64), arithmetic in i32
- **Output:** `nnue.evaluate(board) / 64` gives centipawn-scale evaluation

### Training data format (26 bytes per position)
```
[0-8]   white pits (u8)
[9-17]  black pits (u8)
[18]    white kazan (u8)
[19]    black kazan (u8)
[20]    white tuzdyk (i8 as u8)
[21]    black tuzdyk (i8 as u8)
[22]    side to move (u8)
[23-24] eval (i16 LE) - search evaluation
[25]    result (u8): 0=black win, 1=draw, 2=white win
```

### Data generation (`engine/src/datagen.rs`, 395 lines)
- CLI: `datagen <num_games> <depth> <threads> <prefix> [--endgame]`
- Each thread plays independent games using the engine's own search
- Records every position with search eval and game result
- Supports `--endgame` flag for random endgame starting positions (3-50 stones)
- Auto-loads NNUE weights if available (uses NNUE eval during search)
- Typical output: ~100 positions per game, ~100K positions per hour per thread

### Training scripts
- `train_nnue_v2.py` — original trainer (hardcoded K=1050)
- `train_custom_k.py` — configurable K, lambda, multi-file input
- `train_endgame.py` — multi-file with progressive endgame weights (2x/4x/8x/12x)

### Training loss function
```python
loss = lambda * MSE(sigmoid(pred/K), sigmoid(eval/K)) + (1-lambda) * MSE(sigmoid(pred/K), result)
```
Where:
- `pred` = model output
- `eval` = search evaluation from datagen
- `result` = game outcome (0, 0.5, 1)
- `K` = sigmoid scaling factor
- `lambda` = weight between eval and result (0.5 = equal mix)

### Alternative architectures tested
| Architecture | Params | Bytes | Result |
|-------------|--------|-------|--------|
| 256→32→1 | 18,753 | 37,510 | **Best (current)** |
| 512→64→1 | ~50K | ~100K | Lower val_loss but WEAKER play |
| 256→64→1 | ~25K | ~50K | Marginally better val_loss, not stronger |

**Key finding:** Larger NNUE architectures had lower validation loss but DID NOT translate to stronger play. 256→32→1 remained optimal.

---

## 4. Phase 3: K-Sigmoid Scaling Discovery

### Problem
Initial training used K=1050 (from Stockfish convention). Testing revealed this was far from optimal.

### K-value sweep (200-game definitive matches, NNUE vs HCE)
| K value | Win rate vs HCE | Notes |
|---------|----------------|-------|
| 200 | 92.2% | Too aggressive sigmoid |
| 300 | 95.5% | Good |
| 350 | 94.8% | Good |
| **400** | **96.8%** | **Best** |
| 450 | ~94% | Declining |
| 500 | ~92% | Declining |
| 700 | ~90% | Poor |
| 800 | ~88% | Poor |
| 900 | 88.0% | Poor |
| 1050 | ~85% | Original, worst |

### Result
**K=400 is optimal: +256 Elo improvement** over K=1050.

### Seed variance discovery
Same K value with different random seeds showed massive variance:
- K=400, seed 1: 96.8%
- K=400, seed 2: 72.5%
- K=400, seed 3: 95.2%

This means a single training run is not reliable — need to either average runs or use large datasets.

---

## 5. Phase 4: NNUE Generational Training (Gen1–Gen4)

### Principle
Iterative self-play: train NNUE → generate data with NNUE engine → train next gen → repeat.

### Gen1: Initial NNUE
- Data: HCE engine self-play, depth 8
- ~500K positions
- First NNUE, significant improvement over HCE

### Gen2: Second generation
- Data: Gen1 NNUE engine self-play
- Stronger evaluations → better training data

### Gen3: Third generation
- Data: Gen2 NNUE self-play + expert data
- **Gen3 vs Gen2: +104 Elo** (200 games, random openings, 64.5%, 127-4-69)
- Weights: `nnue_weights_gen3_champion_backup.bin` (md5: 55a11781)

### Gen4: Fourth generation
- Data: 11.09M positions from 8 sources:
  - Multiple generations of self-play data
  - PlayOK expert games (filtered ELO ≥ 1500)
  - Endgame-focused data
- Training: K=400, Lambda=0.5, endgame weighting, 100 epochs
- **Gen4 vs Gen3: +38 Elo** (200 games, random openings, 55.5%, 110-2-88)
- **Cumulative: +142 Elo from Gen2 to Gen4**
- Weights: `nnue_weights_gen4_256.bin` → `nnue_weights.bin` (md5: fbf208c6)

### Generational Elo gains
```
Gen3 vs Gen2: +104 Elo (retrain from scratch)
Gen4 vs Gen3:  +38 Elo (retrain from scratch, diminishing)
Gen5 vs Gen4:  +42 Elo (fine-tuning breakthrough)
Gen6 vs Gen5:   +0 Elo (depth-10 plateau)
```

### Gen5 distillation attempt (FAILED)
- Tried training with distilled AlphaZero NN data (139K positions)
- Gen5 WEAKER than Gen4 (37.5% in 20 games)
- Gen4 weights restored as champion
- Later succeeded with fine-tuning approach (see Phase 10)

### Important testing methodology fixes
- **Old match code was BROKEN**: deterministic moves → 100-0-100 pattern → only 2 unique games played
- **Fix**: Added 4 random opening plies to each game
- **Old results inflated**: what was measured as +295 Elo was actually +104 Elo
- **100-game matches unreliable**: need 200-game minimum for meaningful results
- **99% vs HCE is saturated**: must use NNUE vs NNUE for real comparison

---

## 6. Phase 5: PlayOK Expert Data Integration

### Source
- 184,444 game files from PlayOK platform
- Player ratings: 563–2523
- Format: custom PGN-like text files

### Filtering
- Both players ELO ≥ 1500: 19.6K games → 2.04M positions
- Both players ELO ≥ 1600: 10.3K games → 1.06M positions (also trained)

### Key insight
Expert data has `eval=0` (no engine evaluation available). Training uses `has_eval` mechanism:
- If eval exists: `loss = lambda * eval_loss + (1-lambda) * result_loss`
- If eval=0: `loss = result_loss` only (pure game outcome supervision, effectively lambda=0)

### Impact
Real human endgame patterns break the circular self-play learning problem. Human games contain:
- Diverse opening choices
- Natural endgame play patterns
- Tuzdyk strategies from experienced players
- Stone preservation techniques

This data was crucial for Gen3 and Gen4 improvements.

---

## 7. Phase 6: Endgame Tablebases (EGTB)

### Motivation
Champions tested the engine and found it plays well for 25-30 moves but then makes bad moves: doesn't conserve stones, gives opponent initiative, plays poorly with few stones.

### Implementation (`engine/src/egtb.rs`, 785 lines)

**Retrograde analysis algorithm (bottom-up):**
1. Enumerate all valid positions with N stones on board
2. Mark terminal positions as WIN/LOSS/DRAW with DTM=0
3. For non-terminal positions, try all moves → look up successor:
   - Any successor is LOSS for opponent → this is WIN, DTM = 1 + min(succ DTM)
   - All successors are WIN for opponent → this is LOSS, DTM = 1 + max(succ DTM)
4. Iterate until convergence
5. Remaining unresolved = DRAW

**Key property:** Total board stones never increase after a move (captures go to kazan). So N=0 solves first, then N=1 uses N≤0 results, etc.

**Storage:** WDL+DTM packed in 1 byte (bits 7:6 = result, bits 5:0 = DTM 0-63)

**Position encoding:** Pack board into u128 key (18 pits x 4 bits + 2 kazans x 8 bits + 2 tuzdyks x 4 bits + 1 side bit = 97 bits)

### Results
| Scope | Positions | File size | Notes |
|-------|-----------|-----------|-------|
| N≤3 | ~800K | ~3 MB | Quick test, verified correct |
| **N≤4** | **170M** | **67.6 MB** | **Production (egtb.bin)** |
| N≤5 | Too large | >30 GB RAM | Not feasible on current hardware |

### Integration
- Probed in `alpha_beta()` after terminal check, before depth check
- Probed in `quiescence()` after terminal check
- Binary search on sorted keys: ~25 comparisons for 170M entries ≈ 125ns per probe
- Returns exact score with distance-to-mate

---

## 8. Phase 7: Endgame Search Improvements

### Three changes made to `engine/src/search.rs`

#### 1. Endgame Evaluation Correction (line ~163)
```rust
if total <= 30 {
    let my_active = board.pits[me].iter().filter(|&&x| x > 0).count() as i32;
    let opp_active = board.pits[opp].iter().filter(|&&x| x > 0).count() as i32;
    let mobility_bonus = (my_active - opp_active) * 2;  // +2cp per active pit
    let stone_bonus = (my_stones as i32 - opp_stones as i32) / 3;
    base + mobility_bonus + stone_bonus
}
```

#### 2. Endgame Depth Extensions (line ~668)
```rust
let endgame_ext: i32 = if is_deep_endgame && ply <= 6 { 1 }  // ≤15 stones
    else if is_endgame && ply <= 2 { 1 }                        // ≤30 stones
    else { 0 };
```
Extends search depth by 1 ply in endgame positions to see further.

#### 3. Deep Endgame Quiescence Search (line ~812+)
```rust
let qsearch_all_moves = total_board_stones <= 15 && qs_depth < 4;
```
When ≤15 stones on board: search ALL moves (not just captures/tuzdyks) in first 4 plies of quiescence. Prevents horizon effect where engine misses quiet but critical endgame moves.

Split `quiescence()` into `quiescence()` + `quiescence_inner(qs_depth)` to track depth and prevent explosion.

### Impact
**NNUE vs HCE: 84.0% → 94.5% (+206 Elo from search changes alone)**

This was the single biggest improvement from a code change, demonstrating that search improvements can be more impactful than evaluation improvements.

---

## 9. Phase 8: AlphaZero / MCTS Experiments

### Motivation
Explore whether MCTS + large neural network (AlphaZero-style) could surpass alpha-beta + NNUE.

### Architecture: TogyzNet (`alphazero-code/alphazero/model.py`)
Three sizes tested:

| Size | Blocks | Channels | Parameters |
|------|--------|----------|------------|
| Small | MLP | 256→256→128 | ~30K |
| **Medium** | **10 ResBlocks** | **128** | **1,003,542** |
| Large | 20 ResBlocks | 256 | ~5M |

Input: 7 channels x 9 positions (Conv1d architecture)
- Channel 0-1: my pits, opponent pits (normalized)
- Channel 2-3: my kazan, opponent kazan
- Channel 4-5: my tuzdyk, opponent tuzdyk (one-hot collapsed)
- Channel 6: side to move

Output: policy (9 moves) + value (-1 to 1)

### Step 1: Supervised Pretraining (`supervised_pretrain.py`)
- Trained medium model on 3.35M expert positions from PlayOK
- **Result: 67.6% move accuracy** (top-1 match with expert moves)
- Checkpoint: `checkpoints/supervised_pretrained.pt`

### Step 2: MCTS Implementation (`train_fast.py`)
- `TrueBatchMCTS`: batch MCTS with virtual losses for GPU efficiency
- UCB with PUCT: `Q(a) + c_puct * P(a) * sqrt(N_parent) / (1 + N_child)`
- Dirichlet noise at root for exploration
- `ParallelSelfPlay`: plays multiple games in parallel

### Step 3: Testing MCTS vs NNUE Engine (`test_vs_nnue.py`)
```
MCTS (200 sims, pretrained) vs NNUE engine (1s/move):
Result: MCTS 0 - 20 Engine (0% win rate)
```

The pretrained MCTS model with 200 simulations lost every single game to the NNUE alpha-beta engine.

### Step 4: Self-Play Refinement Attempt
Ran 5 iterations of standard AlphaZero self-play training.

**Catastrophic forgetting observed:**
- Policy loss INCREASED from 1.25 → 1.42 over 5 iterations
- The self-play data destroyed the supervised expert knowledge
- Model became weaker, not stronger

### Step 5: NN Distillation to NNUE (`generate_nnue_data.py`)
Attempt to distill the large NN's knowledge into NNUE training data:
- Generated 139,363 positions from 1000 MCTS self-play games
- Converted NN value [-1,1] to centipawns via sigmoid inverse
- Combined with main training data (2.9M positions)
- **Result: Gen5 NNUE WEAKER than Gen4** (37.5% in 20 games)
- The 139K distilled positions were too few and too noisy to help

### Why MCTS Loses
1. **Search depth**: Alpha-beta reaches depth 15+ in 1s; MCTS with 200 sims reaches depth 3-4
2. **Efficiency**: NNUE (18K params, integer math) evaluates millions of positions/sec; NN (1M params, float GPU) evaluates thousands
3. **Game characteristics**: Togyz Kumalak rewards deep tactical calculation more than pattern recognition
4. **Branching factor**: 9 moves max — narrow enough for alpha-beta to be highly efficient

---

## 10. Phase 9: Gumbel AlphaZero with Supervised Replay

### Motivation
Fix two problems with standard MCTS approach:
1. Standard MCTS needs 800+ sims → Gumbel MCTS works with 32 sims
2. Self-play destroys supervised knowledge → Supervised replay buffer prevents forgetting

### Gumbel MCTS (`gumbel_az.py`, 722 lines)

**Algorithm (from "Policy Improvement by Planning with Gumbel", Danihelka et al., 2022):**

1. Sample Gumbel noise g(a) for each action
2. Select top-k actions by: `g(a) + log π(a)` (Gumbel-Top-k)
3. Sequential Halving: divide simulation budget into log2(k) phases
   - Each phase: allocate sims equally among remaining actions
   - After each phase: discard bottom half by `g(a) + logits(a) + σ(q̂(a))`
4. Improved policy: `π_improved(a) ∝ exp(logits(a) + σ(q̂(a)))`

Where `σ(q) = c_visit * q` is a monotonically increasing transform.

**Key advantage:** Guarantees policy improvement with as few as 16-32 simulations.

### Supervised Replay Buffer
- Load 500K expert positions from PlayOK (ELO ≥ 1400)
- Each training batch: 70% self-play data + **30% expert data**
- Expert data never evicted from buffer
- Prevents catastrophic forgetting of supervised knowledge

### Training Configuration
```
Model: medium (1,003,542 params)
Sims: 32 (Gumbel)
Games per iteration: 100
Expert ratio: 30%
Batch size: 512
Learning rate: 0.001
Iterations: 50
Expert buffer: 500,099 positions
```

### Training Results (50 iterations, 5000 games, 124 minutes)

| Iteration | Policy Loss | Value Loss | Win vs Random |
|-----------|------------|-----------|---------------|
| 1 | 0.891 | 0.237 | — |
| 5 | 0.892 | 0.253 | 100.0% |
| 10 | 0.852 | 0.239 | 93.3% |
| 15 | 0.818 | 0.208 | 93.3% |
| 20 | 0.809 | 0.195 | 96.7% |
| **25** | **0.786** | **0.186** | **100.0%** |
| 30 | 0.781 | 0.184 | 96.7% |
| 35 | 0.788 | 0.184 | 96.7% |
| 40 | 0.798 | 0.189 | 93.3% |
| 45 | 0.806 | 0.191 | 93.3% |
| 50 | 0.818 | 0.188 | 91.7% |

**Key observations:**
- Policy loss improved from 0.89 → 0.78 (11% reduction) — best at iter 25-31
- **NO catastrophic forgetting** — supervised replay works as designed
- Win rate vs random stays 93-100% throughout
- After iter 30, slight overfitting — loss begins to rise
- Self-play generates ~9,500 positions per iteration (100 games)

### Testing Best Checkpoint vs NNUE Engine
```
Gumbel AZ (iter 25, 200 sims) vs NNUE engine (1s/move):
Result: MCTS 0 - 20 Engine (0% win rate)
```

**Still 100% loss to the NNUE alpha-beta engine.** The Gumbel improvements and supervised replay worked for training stability, but the fundamental MCTS depth disadvantage remains.

### Saved Checkpoints
```
checkpoints/supervised_pretrained.pt   — Base supervised model
checkpoints/gumbel_iter5.pt            — Iteration 5
checkpoints/gumbel_iter10.pt           — Iteration 10
checkpoints/gumbel_iter15.pt           — Iteration 15
checkpoints/gumbel_iter20.pt           — Iteration 20
checkpoints/gumbel_iter25.pt           — Best (lowest policy loss)
checkpoints/gumbel_iter30.pt           — Iteration 30
checkpoints/gumbel_iter35.pt           — Iteration 35
checkpoints/gumbel_iter40.pt           — Iteration 40
checkpoints/gumbel_iter45.pt           — Iteration 45
checkpoints/gumbel_iter50.pt           — Final
```

---

## 11. Phase 10: Gen5 Fine-Tuning Breakthrough

### Problem
Gen4 was the reigning champion, but iterative self-play with retraining from scratch showed diminishing returns (Gen3 +104 → Gen4 +38). The question: how to extract further Elo from the same pipeline?

### Failed attempts

#### Gen5v1: AlphaZero distillation (37.5%)
- 139K positions from MCTS NN self-play distilled into NNUE data
- Mixed with main training set (2.9M)
- Result: 37.5% vs Gen4 — WEAKER (see Phase 8)

#### Gen5v2: Heavy PlayOK mix (19%)
- 90% PlayOK expert data + 10% self-play
- 2.18M depth-10 selfplay + massive PlayOK
- Result: 19% vs Gen4 — MUCH WEAKER
- PlayOK data (eval=0) massively dilutes the eval signal

#### Gen5v3: Balanced PlayOK mix (32%)
- 60% self-play + 40% PlayOK (8.3M positions total)
- Better balance but still too much eval-less data
- Result: 32% vs Gen4 (-131 Elo) — FAILED

#### Gen5v4: Pure self-play retrain (48.5%)
- 6.29M positions, ALL self-play with evals, NO PlayOK
- Combined from 5 self-play sources across generations
- Trained from scratch with `train_custom_k.py`
- Result: 48.5% vs Gen4 (-10 Elo) — essentially tied
- Insight: retraining from scratch on self-play data just reproduces Gen4

### Winning approach: Fine-tuning

**Key insight:** Instead of retraining from scratch, fine-tune the existing Gen4 weights on new, higher-quality data (depth-10 self-play) with a low learning rate.

#### Data generation
- Gen4 engine self-play, depth 10, 16 threads, 20K games
- Output: `gen5v2_d10_training_data.bin` — 2.18M positions (56.8 MB)
- Depth 10 (vs depth 8 in earlier gens) = more accurate eval labels

#### Fine-tuning setup (`finetune_nnue.py`)
```
Base weights: nnue_weights_gen4_256.bin (Gen4 champion)
Data: 2.18M depth-10 self-play positions
Learning rate: 0.0001 (10x lower than from-scratch training)
Epochs: 30
Lambda: 0.5
K: 400 (fixed from 1050 — critical bug fix in finetune script)
Optimizer: Adam, weight_decay=1e-5
Scheduler: CosineAnnealing
```

#### Critical fix
`finetune_nnue.py` originally had `K = 1050.0` (line 11), causing sigmoid mismatch with the engine's K=400. Changed to `K = 400.0` to match.

#### Result
```
Gen5-FT vs Gen4: 55-2-43 (56.0%) → +42 Elo
```

**Gen5-FT becomes new champion.** Weights: `nnue_weights_gen5_ft.bin` (md5: d52407e5)

### Why fine-tuning works
1. **Preserves existing knowledge:** Gen4 weights encode 4 generations of learning. Retraining from scratch discards this.
2. **Low LR nudges, doesn't overwrite:** lr=0.0001 makes small adjustments guided by better depth-10 eval labels.
3. **Higher quality labels:** Depth-10 search gives more accurate evaluations than depth-8, so the model learns small corrections.
4. **No eval-less data:** Using only self-play data with evals (no PlayOK) keeps the training signal clean.

### Gen5 Training Lessons
| Approach | Win% vs Gen4 | Elo | Verdict |
|----------|-------------|-----|---------|
| AZ distillation | 37.5% | -90 | Failed |
| 90% PlayOK | 19% | -250 | Failed badly |
| 60/40 PlayOK/selfplay | 32% | -131 | Failed |
| Pure selfplay retrain | 48.5% | -10 | Tied |
| **Fine-tune depth-10** | **56.0%** | **+42** | **Winner** |

---

## 12. Phase 11: Gen6 Iterative Fine-Tuning (Plateau)

### Hypothesis
If fine-tuning Gen4→Gen5 gave +42 Elo, repeating the same loop (generate depth-10 data with Gen5, fine-tune Gen5→Gen6) should give another boost.

### Data generation
- Gen5 engine self-play, depth 10, 16 threads, 20K games
- Duration: ~4h 52m (64.5 games/min)
- Output: `gen6_d10_training_data.bin` — 2.04M positions (53.1 MB)

### Experiment 1: Direct fine-tune Gen5→Gen6
```
Base: nnue_weights.bin (Gen5-FT)
Data: 2.04M Gen6 depth-10 positions
LR: 0.0001, Epochs: 30, Lambda: 0.5
Best val loss: 0.029194 (epoch 1)
```

**Result: Gen6-FT vs Gen5: 48-2-50 (49.0%) → -7 Elo. TIE.**

The best checkpoint was epoch 1 (minimal change), meaning Gen5 weights are already well-calibrated for this data distribution.

### Experiment 2: Combined data, fine-tune from Gen4
```
Base: nnue_weights_gen4_backup.bin (Gen4)
Data: gen5 + gen6 combined = 4.23M positions (104.8 MB)
LR: 0.0003, Epochs: 50, Lambda: 0.5
Best val loss: 0.027929 (epoch 1)
```

**Result: Gen6-Combined vs Gen5: 49-2-49 (50.0%) → 0 Elo. EXACT TIE.**

Again, epoch 1 was best — the model immediately moves away from the good region.

### Conclusion: Depth-10 plateau
Iterative fine-tuning on depth-10 self-play data has reached a plateau. Gen5 already extracted all available information from depth-10 evaluations. Possible paths forward:

1. **Deeper search data (depth 12+):** More accurate eval labels (~4x slower generation)
2. **Search improvements:** Better move ordering, pruning, extensions
3. **New input features:** Mobility, connectivity, piece-square tables as NNUE inputs
4. **Curriculum learning:** Progressive difficulty from endgame to full game

---

## 13. Phase 12: 58-Feature NNUE Experiment

### Motivation
Despite reaching ~926 Elo above baseline, champion-level players reported the engine plays well for 25-30 moves but then makes strategic mistakes: doesn't conserve stones, gives initiative, lacks positional understanding in transitions and endgames. The hypothesis was that the 40-input feature set (raw pit values + kazans + tuzdyk one-hot) lacks explicit strategic information that could help the NNUE understand higher-level concepts.

### 18 New Strategic Features (indices 40-57)

Extended the NNUE input from 40 to 58 features (`engine/src/nnue.rs`, `build_input_58()`). All features normalized to [0, SCALE] range:

| Index | Feature | Formula | Rationale |
|-------|---------|---------|-----------|
| 40-41 | Total pit stones (my/opp) | `sum(pits) / 81 * SCALE` | Stone conservation awareness |
| 42-43 | Active pits (my/opp) | `count(pit > 0) / 9 * SCALE` | Mobility/flexibility |
| 44-45 | Heavy pits (my/opp) | `count(pit >= 12) / 9 * SCALE` | Long-range sowing potential |
| 46-47 | Weak pits (my/opp) | `count(1 <= pit <= 2) / 9 * SCALE` | Vulnerability awareness |
| 48-49 | Right-zone stones (my/opp) | `sum(pits[6..9]) / 81 * SCALE` | Board zone control |
| 50 | Game phase | `total_board_stones / 162 * SCALE` | Phase-dependent strategy |
| 51 | Kazan difference | `(my_kazan - opp_kazan) / 82 * SCALE` | Explicit advantage signal |
| 52-53 | Tuzdyk threats (my/opp) | `count(opp_pits == 2 && valid) / 8 * SCALE` | Tuzdyk tactical awareness |
| 54-55 | Starvation pressure (opp/my) | `max(0, 20 - stones)² / 400 * SCALE` | Quadratic starvation danger |
| 56-57 | Capture targets (my/opp) | `count(even_stones > 0) / 9 * SCALE` | Capture opportunity count |

The same features were implemented in both Rust (`nnue.rs:build_input_58()`) and Python (`train_nnue_v2.py:compute_features_58()`), ensuring training/inference parity.

### Infrastructure Changes

**Rust engine (`nnue.rs`):**
- Added `build_input_58()` method computing 18 strategic features
- Auto-detects input size from binary weight file header
- Supports 40, 52, and 58-input models simultaneously
- Zero performance overhead for 40-input models (feature computation skipped)

**Python training (`train_nnue_v2.py`):**
- Added `compute_features_58()` mirroring Rust implementation
- Added `--input-size 58` argument for 58-feature models
- Added `--init-weights` argument for fine-tuning from pretrained weights
- Added `load_binary_weights()` supporting both old (40-input) and new formats

**New scripts:**
- `transfer_58feat.py` — Transfer Gen5 weights to 58-input model
- `finetune_transfer.py` — Differential learning rate fine-tuning

**Binary format:** New header `[input_size: u16, h1: u16, h2: u16, h3: u16]` (8 bytes) for non-40-input models. Backward compatible: old format `[h1: u16, h2: u16]` (4 bytes) auto-detected when first u16 >= 128.

### Approach 1: Training From Scratch

Trained two models from scratch on large selfplay datasets with 58-input architecture.

#### Model v1: 58→256→64→1 (from scratch, 6.7M positions)
```
Architecture: 58→256→64→1 (31,617 params, 63,242 bytes)
Data: gen7d14_training_data.bin (1.49M) + gen6big_training_data.bin (1.02M)
      + gen8_58feat_training_data.bin (1.01M) + additional selfplay files
      Total: ~6.7M positions
Training: K=400, Lambda=0.5, lr=0.001, 100 epochs, batch=4096
Best val_loss: 0.029154
```
**Result: v1 vs Gen5 champion = 57-49-94 (40.8%) → -65 Elo**

#### Model v2: 58→256→64→1 (from scratch, retrained on fresh data)
```
Architecture: 58→256→64→1 (31,617 params, 63,242 bytes)
Data: Different data combination, ~2.5M positions
Training: K=400, Lambda=0.5, lr=0.001, 100 epochs
Best val_loss: 0.030412
```
**Result: v2 vs Gen5 champion = 64-44-92 (43.0%) → -49 Elo**

Both from-scratch models significantly worse than Gen5 champion (40-input, 256→32→1). The larger architecture (256→64→1 vs 256→32→1) and extra features did not compensate for loss of generational knowledge.

### Approach 2: Transfer Learning from Gen5

Instead of training from scratch, copy Gen5's learned weights into a 58-input model.

#### Transfer method (`transfer_58feat.py`)
```
Source: nnue_weights_gen5_champion.bin (40→256→32→1, 37,510 bytes)
Target: nnue_weights_58feat_transfer.bin (58→256→32→1, 46,730 bytes)

fc1.weight[:, 0:40]  ← exact copy from Gen5 (preserves all learned knowledge)
fc1.weight[:, 40:58] ← small random init (N(0, 0.01)), breaks symmetry
fc1.bias              ← exact copy from Gen5
fc2.*                 ← exact copy from Gen5
fc3.*                 ← exact copy from Gen5
```
This model starts at Gen5 strength (for positions where new features are ~0) and has capacity to learn to use the 18 new features through fine-tuning.

#### Model ft: Transfer + simple fine-tuning (BEST RESULT)
```
Architecture: 58→256→32→1 (23,361 params, 46,730 bytes)
Base weights: nnue_weights_58feat_transfer.bin (transferred from Gen5)
Data: gen7d14_training_data.bin (1.49M, depth-14) + gen6big_training_data.bin (1.02M, depth-10)
      Total: 2.51M positions
Training: K=400, Lambda=0.5, lr=0.0001, 30 epochs, batch=4096
          Adam optimizer, weight_decay=1e-5, CosineAnnealing scheduler
Best val_loss: 0.030981
```
**Result: ft vs Gen5 champion = 88-30-82 (51.5%) → +10 Elo**

Best of all 58-feature models, but +10 Elo is within statistical noise for 200-game match (±25 Elo confidence).

#### Model ft2: Transfer + differential learning rates (FAILED)
```
Architecture: 58→256→32→1 (23,361 params, 46,730 bytes)
Base weights: nnue_weights_58feat_transfer.bin
Data: same as ft (2.51M positions)
Phase 1 (20 epochs): Only train fc1.weight[:, 40:58] (new feature connections)
  - All other params frozen
  - Gradient hook zeros out fc1.weight[:, :40] gradients
  - lr=0.001, CosineAnnealing
Phase 2 (40 epochs): All params trainable with differential LR
  - Old params (fc2, fc3, biases): lr=0.00003
  - New params (fc1.weight): lr=0.0001
Best val_loss: 0.031586
```
**Result: ft2 vs Gen5 champion = 61-53-86 (43.8%) → -44 Elo**

Differential LR approach much worse than simple fine-tuning. Phase 1 (training only new columns) damaged the model — high loss from restricted training propagated into Phase 2 weights.

### Approach 3: Selfplay Data Generation + Fine-Tuning

Tested whether selfplay data from 58-feat models could improve beyond the ft baseline.

#### Selfplay datagen with v1 model (8000 games, depth 10)
```
Engine: v1 (58→256→64→1, from scratch)
Games: 8000, depth 10, 22 threads
Output: gen8_58feat_training_data.bin (1,012,245 positions, 26.3 MB)
Duration: ~2 hours
```

#### Model Gen2: ft + fine-tuned on v1 selfplay
```
Base: ft model (best 58-feat)
Data: gen8_58feat_training_data.bin (1.01M positions from v1 selfplay)
Training: K=400, Lambda=0.5, lr=0.0001, 30 epochs
```
**Result: Gen2 vs Gen5 champion = 69-55-76 (48.2%) → -12 Elo**

Worse than ft alone. Selfplay data from the weaker v1 model degrades quality.

#### Model Gen2b: ft + fine-tuned on mixed v1 selfplay + depth-14 data
```
Base: ft model
Data: gen8_58feat_training_data.bin (1.01M, v1 selfplay d10)
    + gen7d14_training_data.bin (1.49M, Gen5 d14)
    Total: 2.51M positions
Training: K=400, Lambda=0.5, lr=0.0001, 30 epochs
```
**Result: Gen2b vs Gen5 champion = 72-53-75 (49.2%) → -6 Elo**

Still slightly below baseline. High-quality d14 data partially compensates for weak v1 selfplay.

#### Selfplay datagen with ft model (8000 games, depth 10)
```
Engine: ft (58→256→32→1, transfer + fine-tuned, best model)
Games: 8000, depth 10, 22 threads
Output: gen9_58feat_ft_training_data.bin (1,009,716 positions, 26.3 MB)
Duration: ~2 hours
```

#### Model Gen3: ft + fine-tuned on ft selfplay
```
Base: ft model
Data: gen9_58feat_ft_training_data.bin (1.01M positions from ft selfplay)
Training: K=400, Lambda=0.5, lr=0.0001, 30 epochs
```
**Result: Gen3 vs Gen5 champion = 35-13-42 @90 games (~46%) → ~-28 Elo** (match interrupted at 90/200)

Selfplay on the same depth (d10) does not provide improvement signal, confirming the Gen6 plateau finding: the model cannot learn from its own d10 evaluations because it already knows everything depth-10 can teach.

### Complete Results Summary

All 200-game matches at 500ms/move with 4 random opening plies. Model A = 58-feat, Model B = Gen5 champion (40→256→32→1).

| Model | Architecture | Method | W-D-L | Score | Elo vs Gen5 |
|-------|-------------|--------|-------|-------|-------------|
| **ft** | **58→256→32→1** | **Transfer Gen5 + FT d14+d10** | **88-30-82** | **51.5%** | **+10** |
| Gen2b | 58→256→32→1 | ft + FT selfplay(v1)+d14 | 72-53-75 | 49.2% | -6 |
| Gen2 | 58→256→32→1 | ft + FT selfplay(v1) | 69-55-76 | 48.2% | -12 |
| Gen3 | 58→256→32→1 | ft + FT selfplay(ft) d10 | 35-13-42 @90g | ~46% | ~-28 |
| v2 | 58→256→64→1 | From scratch (2.5M pos) | 64-44-92 | 43.0% | -49 |
| ft2 | 58→256→32→1 | Transfer + Differential LR | 61-53-86 | 43.8% | -44 |
| v1 | 58→256→64→1 | From scratch (6.7M pos) | 57-49-94 | 40.8% | -65 |

### Key Findings from 58-Feature Experiment

#### 1. Extra features provide no meaningful improvement (+10 Elo = noise)
The best 58-feature model (ft) achieved 51.5% in 200 games — within statistical error. The 18 strategic features do not improve play quality. This is surprising because the features capture genuinely useful concepts (tuzdyk threats, starvation, mobility).

**Why features don't help:**
- The 40-input model already **implicitly encodes** these patterns through the first layer. For example, `tuzdyk threats = count(opp_pits == 2)` can be computed from raw pit values by the existing fc1 layer.
- The features are **highly correlated** with existing inputs. `total_stones` is the sum of the 9 pit features. `active_pits` is a non-linear function of the same 9 inputs.
- The 256→32→1 architecture is **too small** (23K params) to effectively use 18 additional dimensions. The model must prioritize — and the original 40 features already contain the essential signal.

#### 2. Transfer learning is essential for extended architectures
From-scratch training (v1: 40.8%, v2: 43.0%) is catastrophically worse than transfer learning (ft: 51.5%). The Gen5 champion's weights encode 5 generations of accumulated knowledge. Discarding these weights and starting fresh from random initialization loses all of this — more features and data cannot compensate.

#### 3. Same-depth selfplay is a dead end (confirmed 3×)
Three independent experiments confirmed that selfplay data generated at the same search depth as training provides no improvement:
- Gen2 (selfplay from v1, d10): 48.2% — worse than ft
- Gen2b (selfplay from v1 + d14 mix): 49.2% — worse than ft
- Gen3 (selfplay from ft, d10): ~46% — worse than ft

This mirrors the Gen6 plateau: the model cannot improve from data it could already produce. Depth increase (d10→d14) is the only known way to create a useful selfplay improvement signal.

#### 4. Differential learning rates hurt more than they help
The ft2 model (43.8%) performed much worse than simple uniform fine-tuning ft (51.5%). Phase 1 (freezing all params except new feature columns) produced high training loss because the model couldn't adjust its existing weights to accommodate new input patterns. By the time Phase 2 unfroze everything, the fc1 new-feature columns were already trained to wrong values.

#### 5. Depth-14 datagen is impractical for 58-feat
Depth-14 games with 22 threads produced 0 complete games in 2 minutes (d14 is ~16× slower than d10). For the 40-input Gen5 model, d14 datagen was feasible from previous sessions. But the 58-input model's larger first layer slightly slows inference, making d14 even more expensive.

### Weight Files Produced

```
nnue_weights_58feat.bin           — v1: 58→256→64→1 from scratch (63,242 bytes)
nnue_weights_58feat_v2.bin        — v2: 58→256→64→1 from scratch (63,242 bytes)
nnue_weights_58feat_transfer.bin  — Transferred Gen5 weights, pre-FT (46,730 bytes)
nnue_weights_58feat_ft.bin        — ft: Transfer + FT, best model (46,730 bytes)
nnue_weights_58feat_ft2.bin       — ft2: Transfer + Differential LR (46,730 bytes)
nnue_weights_58feat_gen2.bin      — Gen2: ft + v1 selfplay FT (46,730 bytes)
nnue_weights_58feat_gen2b.bin     — Gen2b: ft + mixed data FT (46,730 bytes)
nnue_weights_58feat_gen3.bin      — Gen3: ft + ft selfplay FT (46,730 bytes)
```

### Conclusion

**The 58-feature experiment is negative.** Adding 18 handcrafted strategic features to the NNUE input does not improve playing strength. The Gen5 champion (40→256→32→1) remains the best model. The experiment consumed ~12 hours of GPU+CPU time and tested 7 model variants, providing strong evidence that the current 40-input representation is sufficient for the 256→32→1 architecture.

**Implication for future work:** Input feature engineering is not the bottleneck. Improvements should focus on search quality (deeper datagen, better pruning/extensions) or architecture scaling (if at all — previous experiments showed 512→64→1 was also worse despite lower val_loss).

---

## 14. Current State & Architecture

### Active Engine Configuration
- **Weights:** `nnue_weights.bin` (Gen5-FT, md5: d52407e5, 37,510 bytes)
- **EGTB:** `egtb.bin` (N≤4, 67.6 MB, 170M positions)
- **Opening book:** `opening_book.bin` (9.3 KB)
- **Search:** All features listed in Phase 1 + endgame improvements from Phase 7

### Engine Codebase (Rust, 4,704 lines total)
```
engine/src/board.rs    — 467 lines — Board representation, make_move(), game rules
engine/src/search.rs   — 1025 lines — Alpha-beta search, all pruning/extensions
engine/src/egtb.rs     — 785 lines — Endgame tablebases
engine/src/main.rs     — 845 lines — CLI, serve protocol, SMP
engine/src/datagen.rs  — 395 lines — Training data generation
engine/src/eval.rs     — 249 lines — Hand-crafted evaluation
engine/src/nnue.rs     — 167 lines — NNUE inference
engine/src/tt.rs       — 143 lines — Transposition table
engine/src/zobrist.rs  — 110 lines — Zobrist hashing
engine/src/book.rs     — 95 lines  — Opening book
engine/src/texel.rs    — 423 lines — Texel tuning
```

### AlphaZero Codebase (Python, 6,947 lines total)
```
alphazero-code/alphazero/
  gumbel_az.py          — 722 lines — Gumbel AZ with supervised replay
  train_fast.py         — 763 lines — Standard AlphaZero training
  train.py              — 402 lines — Original training script
  model.py              — 339 lines — NN architectures (Small/Medium/Large)
  game.py               — 348 lines — Game logic in Python
  test_vs_nnue.py       — 176 lines — Testing MCTS vs NNUE engine
  generate_nnue_data.py — 275 lines — NN→NNUE distillation
  supervised_pretrain.py — 335 lines — Supervised pretraining on expert data
  ...and test/diagnostic scripts
```

### Training Scripts (Python)
```
engine/train_custom_k.py  — Main NNUE trainer (configurable K, lambda)
engine/train_nnue_v2.py   — Original trainer (K=1050, deprecated)
engine/train_endgame.py   — Endgame-weighted trainer
engine/train_dropout.py   — Dropout experiment trainer
engine/finetune_nnue.py   — Fine-tuning script
```

### Deployment
- Server: 5.129.198.203:8080 (1 vCPU, 961MB RAM)
- `deploy.py`: Paramiko SSH deployment
- Uploads: NNUE weights, EGTB, opening book, web files
- Builds engine on server, creates systemd service

---

## 15. Key Findings & Lessons Learned

### Training & Evaluation
1. **Val_loss is NOT predictive of playing strength.** Gen3 512x64 had lower val_loss but played weaker than 256x32. Always test by playing games.
2. **Seed variance is MASSIVE.** K=400 seed 1: 96.8%, seed 2: 72.5%. Single training runs are unreliable.
3. **100-game matches are unreliable.** Need 200-game minimum for meaningful Elo comparisons.
4. **Random opening plies are REQUIRED.** Without them, deterministic engines play the same 2 games over and over.
5. **99% vs HCE is saturated.** When both NNUE models beat HCE 95%+, must compare NNUE vs NNUE directly.

### NNUE Specifics
6. **K=400 sigmoid scaling is optimal.** K=1050 (Stockfish convention) was terrible for this game (+256 Elo difference).
7. **256→32→1 architecture is optimal.** Bigger is not better — 512→64→1 performed worse.
8. **Weight averaging DESTROYS performance.** Averaging checkpoints from different seeds was tried and failed.
9. **Iterative self-play yields consistent gains** but with diminishing returns (Gen3 +104, Gen4 +38).
10. **Expert data breaks circular learning.** PlayOK games inject real human patterns that self-play can't discover.

### MCTS / AlphaZero
11. **MCTS+NN (1M params, 200 sims) loses 100% to NNUE alpha-beta (18K params).**
12. **Standard self-play causes catastrophic forgetting** of supervised knowledge (policy loss 1.25→1.42).
13. **Gumbel MCTS + supervised replay fixes forgetting** (policy loss stable/improving), but MCTS still can't match alpha-beta depth.
14. **Togyz Kumalak favors depth over breadth.** With branching factor 9 and tactical tuzdyk moves, alpha-beta is far more efficient than MCTS.
15. **NN distillation to NNUE doesn't help.** 139K positions from NN self-play were too few/noisy to improve 2.9M dataset.

### Search Improvements
16. **Endgame search changes gave +206 Elo** — single biggest code-level improvement.
17. **Deep qsearch in endgame is critical.** Searching all moves (not just captures) when ≤15 stones prevents horizon effect.
18. **ASP_DELTA=20 and RFP_MARGIN=70** are optimal for NNUE/64 eval scale. Tested alternatives, all worse.

### Infrastructure
19. **PYTHONUNBUFFERED=1 is essential** for real-time output monitoring.
20. **torch.compile('reduce-overhead') works** on RTX 5080, gives ~15% speedup.

### Fine-Tuning & Generational Training
21. **Fine-tuning >> retraining from scratch.** Fine-tune (lr=0.0001) on depth-10 data gave +42 Elo; retraining from scratch on 6.3M self-play positions tied.
22. **PlayOK data (eval=0) always hurts when mixed with self-play.** 90% PlayOK: -250 Elo, 40% PlayOK: -131 Elo, 0% PlayOK: best results.
23. **Depth-10 fine-tuning plateaus after one generation.** Gen5 (+42) but Gen6 (+0). Same-depth data provides no new information.
24. **Best epoch = epoch 1 is a warning sign.** Means the base model is already optimal for this data distribution — no improvement possible.

### Input Feature Engineering (Phase 12)
25. **18 handcrafted strategic features provide zero improvement.** Best 58-feat model scored 51.5% (+10 Elo = noise) vs 40-feat Gen5 champion.
26. **NNUE implicitly learns strategic patterns** from raw features. Explicit features (mobility, tuzdyk threats, starvation) are redundant — the network already computes them.
27. **Transfer learning is essential** when extending input features. From-scratch 58-feat (40-43%) vs transfer-based (51.5%). 5 generations of accumulated knowledge cannot be replaced by more features.
28. **Differential learning rates hurt.** Freezing old weights while training new feature columns (ft2: 43.8%) is much worse than uniform fine-tuning (ft: 51.5%).
29. **Same-depth selfplay is a dead end (confirmed 5 times total).** Gen6 d10 (+0), 58-feat Gen2 d10 (-12), Gen2b d10 (-6), Gen3 d10 (-28). Only depth increase provides improvement signal.

---

## 16. File Structure

```
9QumalaqV2/
├── engine/                           # Rust engine
│   ├── src/                          # Source code (4,704 lines)
│   ├── nnue_weights.bin              # Current best (Gen4, 37 KB)
│   ├── egtb.bin                      # Endgame tablebases (67 MB)
│   ├── opening_book.bin              # Opening book (9 KB)
│   ├── nnue_weights_*_backup.bin     # Historical weight files
│   ├── train_custom_k.py             # Main NNUE trainer
│   ├── train_endgame.py              # Endgame-weighted trainer
│   ├── gen*_thread_*.bin             # Self-play data files
│   └── Cargo.toml                    # Rust dependencies
│
├── alphazero-code/alphazero/         # Python MCTS/AlphaZero
│   ├── game.py                       # Game logic
│   ├── model.py                      # NN architectures
│   ├── gumbel_az.py                  # Gumbel AZ (latest)
│   ├── train_fast.py                 # Standard AZ training
│   ├── supervised_pretrain.py        # Expert supervised training
│   ├── test_vs_nnue.py               # MCTS vs engine testing
│   ├── generate_nnue_data.py         # NN→NNUE distillation
│   └── checkpoints/                  # Saved models
│
├── game-pars/games/                  # 184K PlayOK expert games
├── web/index.html                    # Web interface
├── deploy.py                         # Server deployment
└── FULL_REPORT.md                    # This file
```

---

## 17. What Didn't Work

| Approach | What happened | Why |
|----------|--------------|-----|
| NNUE 512→64→1 | Lower val_loss but weaker play | Overfitting to eval noise |
| K=1050 sigmoid | 85% vs HCE | Wrong scale for this game's eval range |
| Weight averaging | Destroyed performance | Models trained from different seeds have incompatible weight spaces |
| Standard AZ self-play | Catastrophic forgetting (policy 1.25→1.42) | Self-play data overwhelms supervised knowledge |
| NN distillation to NNUE | Gen5 weaker than Gen4 | 139K noisy positions too few vs 2.9M clean data |
| MCTS 200 sims vs alpha-beta | 0-20 loss | MCTS too shallow (3-4 ply) vs alpha-beta (15+ ply) |
| Gumbel AZ 32 sims + supervised replay | Training stable but 0-20 vs engine | Fundamental depth disadvantage remains |
| N≤5 EGTB | Out of memory | >30GB RAM needed |
| Pure MCTS (no NN) | Not tested, known to be weaker | Random rollouts give very noisy eval |
| Gen5 90% PlayOK data | 19% vs Gen4 (-250 Elo) | eval=0 data massively dilutes training signal |
| Gen5 60/40 PlayOK/selfplay | 32% vs Gen4 (-131 Elo) | Even 40% eval-less data hurts |
| Gen5 pure selfplay retrain | 48.5% vs Gen4 (tie) | Retraining from scratch reproduces, doesn't improve |
| Gen6 fine-tune (depth 10) | 49% vs Gen5 (tie) | Depth-10 plateau — no new information |
| Gen6 combined 4.2M data | 50% vs Gen5 (tie) | More data at same depth doesn't help |
| 58-feat from scratch (v1) | 40.8% vs Gen5 (-65 Elo) | Larger input + architecture can't compensate for loss of generational knowledge |
| 58-feat from scratch (v2) | 43.0% vs Gen5 (-49 Elo) | Same — random init loses 5 generations of accumulated learning |
| 58-feat transfer + FT (ft) | 51.5% vs Gen5 (+10 Elo, noise) | Best 58-feat model — but 18 strategic features provide no real gain |
| 58-feat differential LR (ft2) | 43.8% vs Gen5 (-44 Elo) | Freezing old params during Phase 1 damages training |
| 58-feat selfplay Gen2 | 48.2% vs Gen5 (-12 Elo) | Same-depth selfplay from weak model degrades quality |
| 58-feat selfplay Gen3 | ~46% vs Gen5 (~-28 Elo) | Same-depth selfplay from best model still no improvement |

---

## 18. Next Steps

### Most Promising: Deeper Search Data
Depth-10 fine-tuning has plateaued (Gen5 +42, Gen6 +0). 58-feature experiment confirmed that input engineering is not the answer. The breakthrough requires higher quality eval labels from deeper search.

**Plan:** Fine-tune Gen5 on depth-12 or depth-14 data:
1. Datagen at depth 12 with Gen5 engine (~17 games/min, ~10h for 10K games)
2. Fine-tune Gen5 on depth-12 data (lr=0.0001, 30 epochs)
3. Test vs Gen5 (200 games)
4. If successful, continue to depth 14

**Why this should work:** Depth 10→12 is a significant quality jump. Deeper search resolves more tactical positions correctly, giving the NNUE better labels to learn from. This is the same principle that made Gen5 work (depth 8→10).

### Search Improvements (No Training Needed)
Search improvements gave the biggest single boost (+206 Elo in Phase 7). Potential areas:
- **Singular extensions:** Extend search when one move is clearly best
- **Multi-cut pruning:** Prune when multiple moves cause beta cutoffs
- **Better time management:** Allocate more time in critical positions
- **Improved endgame play:** Tune endgame extensions and qsearch

### Ruled Out (by experiment)
- **New NNUE input features:** 58-feat experiment showed no improvement (+10 Elo = noise). The 40-input representation is sufficient.
- **Larger architectures:** 256→64→1 and 512→64→1 both tested, both worse despite lower val_loss.
- **Same-depth selfplay loops:** Confirmed 5 times as dead end.

### Other Potential Improvements
- **Larger EGTB:** N≤5 on a machine with 64GB+ RAM
- **Opening book expansion:** Generate from expert games + engine analysis
- **Texel tuning:** Re-tune pruning margins for Gen5 eval characteristics
- **Match against champions:** Get feedback on specific weaknesses to target

---

## Appendix: Elo Progression

```
Baseline (HCE only):                    0 Elo
+ NNUE Gen1 (K=1050):                 +~200 Elo
+ K=400 sigmoid fix:                  +256 Elo  (cumulative ~456)
+ Gen2 self-play:                     +~50 Elo  (cumulative ~506)
+ Gen3 (+ expert data):              +104 Elo  (cumulative ~610)
+ Gen4 (11M positions):               +38 Elo  (cumulative ~648)
+ Endgame search improvements:       +206 Elo  (cumulative ~854)
+ EGTB N≤4:                          +~30 Elo  (cumulative ~884)
+ Gen5 fine-tuning (depth 10):        +42 Elo  (cumulative ~926)
+ Gen6 (depth-10 plateau):            +0 Elo  (cumulative ~926)
+ 58-feat experiment:                  +0 Elo  (cumulative ~926, best +10 = noise)
= Current estimated strength:         ~926 Elo above baseline HCE
```

*Note: Elo values are approximate and measured in different conditions. The relative ordering is accurate but absolute values should not be summed directly.*
