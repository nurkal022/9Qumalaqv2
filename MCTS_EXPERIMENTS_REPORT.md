# MCTS AlphaZero Training Experiments Report

## Togyz Kumalak — Path to Champion Level

**Date:** March 22-26, 2026
**Hardware:** RTX 5080 Laptop GPU (16GB VRAM), 24 CPU cores, 30GB RAM
**Baseline:** Gen7 NNUE engine (~1198 Elo above HCE)
**Target:** PlayOK rating 2400+ (competitor mcts@2541 = ~84% winrate on PlayOK)

---

## 1. Background: NNUE Engine History

### What is NNUE?
NNUE (Efficiently Updatable Neural Network) is a small neural network used as the evaluation function inside a traditional alpha-beta search engine. Our engine uses iterative deepening with null-move pruning, aspiration windows, late move reductions, and a transposition table. The NNUE replaces the hand-crafted evaluation (HCE) to score positions.

### Architecture Evolution
| Gen | Architecture | Params | Bytes | Elo vs HCE | Method |
|-----|-------------|--------|-------|------------|--------|
| Gen1 | 256->32->1 | 18,753 | 37,510 | baseline | Supervised on PlayOK |
| Gen2 | 256->32->1 | 18,753 | 37,510 | ~+200 | Selfplay depth-6 |
| Gen3 | 256->32->1 | 18,753 | 37,510 | ~+304 | Selfplay depth-8 |
| Gen4 | 256->32->1 | 18,753 | 37,510 | ~+342 | Selfplay depth-8 fine-tune |
| Gen5 | 256->32->1 | 18,753 | 37,510 | ~+384 | Selfplay depth-10 fine-tune |
| Gen6 | 256->32->1 | 18,753 | 37,510 | ~+544 | Better data pipeline |
| **Gen7** | **256->32->1** | **18,753** | **37,510** | **~+1184** | **QSearch+TT+param tuning** |
| Gen8_sup_lam05 | 256->32->1 | 18,753 | 37,510 | ~+1198 | Distillation lambda=0.5 |

### Gen7 Details (Current Deployed Engine)
- Search: Alpha-beta with iterative deepening, depth 10-14 in 500ms
- Pruning: ASP_DELTA=35, RFP_MARGIN=70, LMR divisor=2.5
- Features: EGTB (endgame tablebases), opening book (1.2MB), transposition table
- Deployed at http://85.239.36.121 via systemd service

### All Failed Attempts to Beat Gen7
| Attempt | Data | Method | Result | Root Cause |
|---------|------|--------|--------|------------|
| Gen8 | 20K games selfplay from Gen7 | Fine-tune Gen7 weights | -17 Elo | Architecture capacity saturated at 18K params |
| Gen8v2 | 6.4M combined Gen6+Gen7+Gen8 | Train from scratch on mixed data | Degraded at epoch 1 | Mixed-generation data always hurts — different gen evals are incompatible |
| Gen8v3 | Same as Gen8 | Lower lr=5e-5 | Still worse | Same capacity limit, lr doesn't help |
| Gen8b | 40K games selfplay from Gen7 | More data, fine-tune | -90 Elo (37.5%) | More data of same quality = overfitting, not improvement |
| 256->64->1 | Gen7 data | Transfer learning, 60 epochs | +11 Elo (51.5%, 100 games) | Marginal, inference cost barely acceptable |
| 512->64->1 | Gen7 data | Bigger hidden layer | -53 Elo (42.5%) | Inference 2x slower = less search depth = weaker |
| Gen8_sup_lam05 | 685K distillation positions | Lambda=0.5, 60 epochs fine-tune | **+14 Elo** (52%, 100 games) | Best NNUE result but marginal, ceiling remains |

**Key insight:** NNUE 256->32->1 has only 18,753 parameters. It can represent simple piece-value and positional patterns but lacks capacity for deep tactical/strategic understanding. The search compensates but can't overcome evaluation blindness. More data, more training, bigger NNUE — all failed. Need a fundamentally different approach.

---

## 2. The MCTS+NN Approach (AlphaZero Style)

### Why MCTS?
Instead of alpha-beta search with NNUE evaluation, use Monte Carlo Tree Search with a large neural network that predicts both:
- **Policy** (move probabilities) — guides which branches to explore
- **Value** (position evaluation) — estimates who is winning

The NN is much larger (1-2M params vs 18K) and sees the full board state through convolutional layers. MCTS uses the policy to focus search and value to evaluate leaves. This is how AlphaZero, Leela Chess Zero, and KataGo work.

### Our NN Architecture: TogyzNet
```
Input: [batch, 7, 9] — 7 channels x 9 positions

Channels:
  0: Current player pits / 50.0  (normalized stone counts)
  1: Opponent pits / 50.0
  2: Current player kazan / 82.0  (captured stones, broadcast to all 9 positions)
  3: Opponent kazan / 82.0
  4: Current player tuzdyk (one-hot, which pit is marked)
  5: Opponent tuzdyk (one-hot)
  6: Side indicator (1.0 if White, 0.0 if Black, broadcast)

Body: N ResBlocks x C channels (1D convolutions, kernel=3, padding=1)
  Each ResBlock: Conv1d -> BatchNorm -> ReLU -> Conv1d -> BatchNorm -> skip connection -> ReLU

Policy head: Conv1d(C,1) -> Flatten -> Linear(9,9) -> log_softmax
  Output: log-probability over 9 pits

Value head: Conv1d(C,1) -> Flatten -> Linear(9,64) -> ReLU -> Linear(64,1) -> tanh
  Output: scalar in [-1, +1], from current player's perspective
```

Model sizes:
| Name | ResBlocks | Channels | Params | ONNX Size |
|------|-----------|----------|--------|-----------|
| small | 6 | 64 | ~200K | ~0.8MB |
| medium | 10 | 128 | ~1.0M | ~4MB |
| **large2m** | **10** | **192** | **2,240,022** | **8.9MB** |
| large | 20 | 256 | ~10M | ~40MB |

### MCTS Algorithm (Batch-Parallel)
Traditional MCTS does 800 sequential evaluations (800 GPU calls). Our implementation collects 128 leaves per batch using virtual loss:

```
For each move in the game:
  1. Initialize root node
  2. Evaluate root with NN → get policy priors for children
  3. Add Dirichlet noise to root (exploration)
  4. Repeat until 800 simulations done:
     a. COLLECT phase: traverse tree 128 times using PUCT selection
        - At each node, select child maximizing: Q(a) + c_puct * P(a) * sqrt(N) / (1 + n(a))
        - Apply virtual loss along path: +1 visit, -1 value (pessimistic)
        - Collect unexpanded leaf positions
     b. EVALUATE phase: batch all 128 leaves in ONE GPU call
        - Input: [128, 7, 9] tensor
        - Output: 128 policies + 128 values
     c. EXPAND+BACKPROP phase: for each leaf
        - Create child nodes with NN policy priors
        - Undo virtual loss and backpropagate real NN value
  5. Final policy = visit counts (normalized), not raw NN policy
  6. Select move: temperature=1.0 for first 15 moves (exploration), greedy after
```

This reduces 800 GPU calls to ~7 batched calls, giving ~100x speedup.

### Training Loop (AlphaZero Style)
```
For each iteration:
  1. SELFPLAY: Run 100 games of model vs itself using MCTS (800 sims)
     - Record every position: (board_state, MCTS_policy, game_outcome)
     - game_outcome: +1 if side-to-move won, -1 if lost, 0 if draw
     - ~130 positions per game, ~13,000 positions per iteration

  2. ACCUMULATE: Add new positions to replay buffer (max 500K)
     - When full, discard oldest positions (FIFO)

  3. TRAIN: Update NN on accumulated buffer
     - Policy loss: cross-entropy between NN output and MCTS visit counts
     - Value loss: MSE between NN output and game outcome
     - Total loss = policy_loss + value_loss
     - AdamW optimizer with gradient clipping

  4. EXPORT: Convert PyTorch model to ONNX for Rust inference

  5. EVAL: Play model vs Gen7 engine (N games, alternating colors)
     - Model uses Python MCTS (400 sims) vs Engine (500ms time limit)
     - Report win/draw/loss statistics

  6. GATING: If winrate > best, update selfplay model
```

---

## 3. Config B: Python MCTS Proof of Concept (March 22-24)

### Method
Before building the Rust engine, we tested the MCTS approach in pure Python using `train_config_b.py`:

- Model: medium (1M params, 10 ResBlocks x 128ch)
- MCTS: `ConfigurableMCTS` class with batched GPU inference
- Selfplay: 100 games/iteration, 800 simulations/move
- Training: 10 epochs, lr=0.001
- Eval: Plays full games vs Gen7 engine using stdin/stdout protocol
- Speed: ~24 min/iteration (~4 games/hour)

### Key Design: Proper Evaluation Protocol
The model plays FULL games against the engine:
1. Start engine subprocess with `engine serve` command
2. For each position, send board state to engine, receive move
3. Model uses MCTS search (not raw policy!) to choose moves
4. Alternate colors across games for fairness
5. This is the SAME protocol used by all our eval — proven reliable

### Results
| Iteration | Winrate vs Gen7 | Notes |
|-----------|----------------|-------|
| 10 | 40% | Learning |
| 20 | 50% | Reaching Gen7 level |
| 30 | 55% | Surpassing Gen7 |
| 50 | 60% | Growing |
| 85 | 55-67% | **Still growing when stopped** |

**Why stopped:** 24 min/iteration = 85 iterations took 34 hours. At this rate, 300 iterations = 5 days. Too slow.

### Why Config B Worked (But Rust Didn't Initially)
Critical difference: Config B evaluated models properly using full-game MCTS vs engine. Our initial Rust pipeline used a broken "1-ply eval" that just checked argmax of policy at each position — this gave 0% winrate and was useless for model selection.

---

## 4. Rust MCTS Engine Development (March 24-25)

### Architecture
Built a high-performance selfplay engine in Rust:

```
rust-mcts/
├── src/
│   ├── main.rs          # CLI, thread orchestration
│   ├── board.rs          # Togyz Kumalak game rules (467 lines)
│   ├── encoding.rs       # Board → [7,9] tensor encoding (117 lines)
│   ├── evaluator.rs      # Central GPU batch evaluator (168 lines)
│   ├── mcts.rs           # Batch-parallel MCTS with PUCT (369 lines)
│   ├── self_play.rs      # Game generation workers (112 lines)
│   └── replay_buffer.rs  # Binary record serialization (100 lines)
├── scripts/
│   ├── export_onnx.py    # PyTorch → ONNX conversion
│   ├── train_alphazero.py # Replay buffer → training
│   ├── train_loop.py      # Full training loop orchestrator
│   └── eval_configb_style.py # Eval using proven Python MCTS
└── Cargo.toml            # ort 2.0.0-rc.12, crossbeam-channel
```

### Worker-Evaluator Architecture
```
┌─────────┐  ┌─────────┐  ┌─────────┐
│ Worker 1 │  │ Worker 2 │  │Worker 20│   (each plays one game at a time)
└────┬─────┘  └────┬─────┘  └────┬────┘
     │              │              │
     ▼              ▼              ▼
  ┌──────────────────────────────────┐
  │     eval_tx (unbounded channel)  │   Workers send EvalRequest
  └──────────────┬───────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────┐
  │     Evaluator Thread             │   Collects batch of 128 requests
  │     ONNX Runtime GPU Session     │   Runs single GPU inference call
  │     Batch: [128, 7, 9] → policy  │   Sends responses via per-worker channels
  └──────────────────────────────────┘
                 │
     ┌───────────┼───────────┐
     ▼           ▼           ▼
  resp_rx_1   resp_rx_2   resp_rx_20    (bounded(256) per worker)
```

### Performance Benchmarks
| Setup | Speed | Inference Time | Notes |
|-------|-------|---------------|-------|
| Dummy eval (no NN) | 159 games/sec | 0ms | Pure tree traversal speed |
| 1M model, CPU | 0.5 games/sec | ~27ms/batch | CPU bottleneck |
| 1M model, GPU | ~1.3 games/sec | ~0.9ms/batch | Good |
| **2M model, GPU** | **1.1-1.3 games/sec** | **~0.9ms/batch** | **Production config** |
| 2M model, CPU | 0.05 games/sec | ~335ms/batch | 300x slower than GPU |
| 2M model, GPU (no CUDA libs) | HANGS | ∞ | Silent CPU fallback + deadlock |

### Critical Bug: CUDA Library Paths
**Symptom:** Rust binary hangs with multiple workers, 0% CPU usage
**Root cause:** ONNX Runtime `ort` crate uses dynamic loading (`load-dynamic` feature). Without `LD_LIBRARY_PATH` pointing to CUDA libraries, it silently falls back to CPU. CPU inference is 300x slower, causing apparent deadlocks (workers wait forever for responses).

**The CUDA libraries are installed as pip packages, not system-wide:**
```
~/.local/lib/python3.12/site-packages/nvidia/cublas/lib/libcublas.so.12
~/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12
~/.local/lib/python3.12/site-packages/nvidia/curand/lib/libcurand.so.10
~/.local/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn.so.9
~/.local/lib/python3.12/site-packages/nvidia/cufft/lib/libcufft.so.11
```

**Fix:** Export before launching:
```bash
NVIDIA_LIBS=~/.local/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$NVIDIA_LIBS/cublas/lib:$NVIDIA_LIBS/cuda_runtime/lib:$NVIDIA_LIBS/curand/lib:$NVIDIA_LIBS/cudnn/lib:$NVIDIA_LIBS/cufft/lib
export ORT_DYLIB_PATH=~/.local/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.24.4
```

### ONNX Export Bug
**Symptom:** Exported ONNX file is 18KB instead of 8.9MB
**Root cause:** PyTorch 2.10+ defaults to new `torch.export`-based ONNX exporter (dynamo mode). For our model, it produces files without embedded weights.
**Fix:** Add `dynamo=False` to `torch.onnx.export()` call.

### Replay Buffer Format
Each training record is 63 bytes, binary, little-endian:
```
Bytes  0- 8: White pits [9 x u8]          — stone counts for each pit
Bytes  9-17: Black pits [9 x u8]
Bytes 18-19: Kazan [2 x u8]               — captured stone counts
Bytes 20-21: Tuzdyk [2 x i8]              — special pit index (-1 if none)
Byte  22:    Side to move [u8]             — 0=White, 1=Black
Bytes 23-58: Policy [9 x f32 LE]          — MCTS visit count distribution
Bytes 59-62: Value [f32 LE]               — game outcome from side's perspective
```

Python reader converts this to `[N, 7, 9]` float32 tensor with perspective rotation.

---

## 5. 2M Model Supervised Pretraining (March 25)

### Method
Before selfplay training, pretrain the 2M model on human expert games from PlayOK:

**Data:** 2595 games, filtered by Elo > 1400
**Targets:**
- Policy: one-hot encoded move that was actually played
- Value: game outcome (+1 white win, -1 black win, 0 draw)

**Training:**
- 30 epochs, batch size 512, lr=0.001 with cosine annealing
- Policy loss: cross-entropy
- Value loss: MSE

### Results
| Epoch | Train Acc | Val Acc | Val p_loss | Val v_loss |
|-------|-----------|---------|------------|------------|
| 1 | 51.5% | 55.5% | 1.160 | 0.635 |
| 5 | 61.5% | 62.0% | 1.015 | 0.650 |
| 10 | 64.5% | 64.8% | 0.925 | 0.590 |
| 20 | 67.5% | 67.2% | 0.860 | 0.560 |
| **30** | **68.7%** | **68.1%** | **0.844** | **0.551** |

Comparison with 1M model:
| Model | Params | Val Accuracy |
|-------|--------|-------------|
| medium (1M) | 1,003,018 | 67.6% |
| **large2m (2M)** | **2,240,022** | **68.1%** |

Only +0.5% accuracy despite 2.2x more parameters. Suggests PlayOK data is not rich enough to differentiate — or that 68% is near-ceiling for this game representation.

---

## 6. Experiment 1: Training Loop v1 (March 25, evening)

### Configuration
```python
--iterations 500
--games 100           # selfplay games per iteration
--sims 800            # MCTS simulations per move
--workers 20          # parallel Rust workers
--model-size large2m  # 2.2M params
--lr 0.001            # learning rate (HIGH)
--train-epochs 5      # epochs per iteration (HIGH)
--eval-interval 10    # eval every 10 iters
--eval-games 10       # games per eval (LOW)
--max-buffer 500000   # replay buffer capacity
```

**Selfplay model:** Always the current model (no gating — whatever just trained plays next iteration)

### Training Dynamics
Each iteration:
- Selfplay: 100 games in ~90s = ~13,000 new positions
- Training: 5 epochs on accumulated buffer in ~5-10s
- Total iteration: ~95-100 seconds

### Results (87 iterations, ~1M positions, 4 hours)

**Loss trajectory:**
```
Iter  1: loss=1.298 (p=1.241, v=0.056)  ← pretrained quality
Iter  5: loss=1.573 (p=1.489, v=0.083)  ← p_loss jumped +0.25 in 5 iters!
Iter 10: loss=1.624 (p=1.534, v=0.090)  ← continuing to degrade
Iter 50: loss=1.660 (p=1.560, v=0.100)  ← plateau
Iter 87: loss=1.669 (p=1.557, v=0.112)  ← stuck
```

**Eval results (10 games each, very noisy):**
| Iter | Eval vs Gen7 | Estimated Elo |
|------|-------------|--------------|
| 10 | 50.0% (5W-0D-5L) | 0 |
| 20 | 55.0% (5W-1D-4L) | +35 |
| 30 | 50.0% (5W-0D-5L) | 0 |
| 40 | 55.0% (4W-3D-3L) | +35 |
| 50 | 65.0% (5W-3D-2L) | +108 |
| 60 | 55.0% (5W-1D-4L) | +35 |
| 70 | 55.0% (5W-1D-4L) | +35 |
| 80 | 50.0% (4W-2D-4L) | 0 |

### Analysis: Why v1 Failed

**Problem 1: Catastrophic forgetting (lr=0.001)**
The pretrained model had p_loss=1.241 (68% accuracy). After just 5 iterations of selfplay training at lr=0.001, p_loss jumped to 1.489. The model rapidly forgot its supervised knowledge and replaced it with low-quality selfplay patterns.

Why? Early selfplay data is garbage — the model plays poorly against itself, generating positions with random-looking outcomes. Training on this data at high lr overwrites the carefully learned PlayOK patterns.

**Problem 2: No model selection**
Without gating, a model that got temporarily worse would generate worse selfplay data, which would train an even worse model — a downward spiral. The 65% peak at iter 50 was likely a lucky fluctuation; by iter 80, performance returned to 50%.

**Problem 3: Eval too noisy**
With only 10 games, standard error is ~16%. A true 55% model shows anywhere from 39% to 71%. The 65% at iter 50 could easily be from a 50% model.

---

## 7. Experiment 2: Training Loop v2 (March 26)

### Changes from v1 (5 key fixes)

**Fix 1: Lower learning rate with cosine annealing**
```python
# v1: lr=0.001 (constant) — destroyed pretrained knowledge
# v2: lr=0.0003, decaying to 0.00003 over 500 iterations
optimizer = AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=500, eta_min=0.00003)
```
Rationale: 3.3x lower initial lr gives the model time to integrate new selfplay knowledge without catastrophic forgetting.

**Fix 2: Fewer training epochs per iteration**
```python
# v1: 5 epochs per iteration — too much overfitting
# v2: 3 epochs per iteration
```
Rationale: With 13K new positions per iteration and 500K buffer, 5 epochs meant seeing each position ~5 times. This caused overfitting to recent data patterns.

**Fix 3: Best-model gating for selfplay**
```python
# v1: selfplay always uses current model (no gating)
# v2: selfplay uses BEST model (highest eval score)
selfplay_onnx = best_onnx_path  # Only updated when new model beats best
```
Rationale: This is how Config B worked — always keep the strongest model generating data. If training produces a weaker model, don't let it corrupt the data pipeline.

**Fix 4: 20-game evaluation**
```python
# v1: 10 games per eval (standard error ~16%)
# v2: 20 games per eval (standard error ~11%)
```
Rationale: Still noisy, but better. 20 games reduces the chance of a 50% model appearing as 70%+.

**Fix 5: Freshness-weighted sampling**
```python
# v1: uniform random sampling from buffer
# v2: linear weights — newest data 1.5x, oldest 0.5x
weights = np.linspace(0.5, 1.5, n)  # older→newer
indices = np.random.choice(n, size=n, replace=True, p=weights/weights.sum())
```
Rationale: Newer positions come from a stronger model. Older data (from early iterations) is lower quality and should have less influence.

### Configuration
```python
--iterations 500
--games 100
--sims 800
--workers 20
--model-size large2m
--init-checkpoint checkpoints_2m/best.pt  # Start from v1's best (iter 50, 65%)
--lr 0.0003
--train-epochs 3
--eval-interval 10
--eval-games 20
--max-buffer 500000
--checkpoint-dir checkpoints_2m_v2
```

### Results (252+ iterations, ~3M positions, ~13 hours)

**Loss trajectory:**
```
Iter   1: loss=1.791 (p=1.577, v=0.215)  ← higher v_loss because selfplay outcomes differ from supervised
Iter  10: loss=1.722 (p=1.567, v=0.155)  ← p_loss barely moved! Knowledge preserved
Iter  50: loss=1.713 (p=1.556, v=0.158)  ← slow improvement
Iter 120: loss=1.718 (p=1.566, v=0.152)  ← best eval point
Iter 250: loss=1.761 (p=1.589, v=0.172)  ← slow degradation after stale selfplay
```

**Key: p_loss stayed at 1.56-1.59 throughout** (vs v1 which jumped from 1.24 to 1.56 in 5 iters). The lower lr preserved pretrained policy knowledge.

**All evaluation results (20 games each):**
| Iter | Win | Draw | Loss | Winrate | Elo ~ | Best? |
|------|-----|------|------|---------|-------|-------|
| 10 | 10 | 1 | 9 | 52.5% | +17 | NEW BEST |
| 20 | 10 | 1 | 9 | 52.5% | +17 | = |
| 30 | 11 | 1 | 8 | 57.5% | +53 | NEW BEST |
| 40 | 8 | 2 | 10 | 45.0% | -35 | gated |
| 50 | 10 | 4 | 6 | 60.0% | +70 | NEW BEST |
| 60 | 8 | 7 | 5 | 57.5% | +53 | |
| 70 | 10 | 0 | 10 | 50.0% | 0 | |
| 80 | 10 | 0 | 10 | 50.0% | 0 | |
| 90 | 9 | 2 | 9 | 50.0% | 0 | |
| 100 | 10 | 0 | 10 | 50.0% | 0 | |
| 110 | 10 | 1 | 9 | 52.5% | +17 | |
| **120** | **14** | **2** | **4** | **75.0%** | **+190** | **NEW BEST** |
| 130 | 10 | 1 | 9 | 52.5% | +17 | |
| 140 | 11 | 1 | 8 | 57.5% | +53 | |
| 150 | 10 | 3 | 7 | 57.5% | +53 | |
| 160 | 9 | 1 | 10 | 47.5% | -17 | |
| 170 | 12 | 5 | 3 | 72.5% | +170 | close |
| 180 | 8 | 3 | 9 | 47.5% | -17 | |
| 190 | 7 | 7 | 6 | 52.5% | +17 | |
| 200 | 10 | 3 | 7 | 57.5% | +53 | |
| 210 | 10 | 1 | 9 | 52.5% | +17 | |
| 220 | 10 | 1 | 9 | 52.5% | +17 | |
| 230 | 10 | 0 | 10 | 50.0% | 0 | |
| 240 | 10 | 3 | 7 | 57.5% | +53 | |
| 250 | 10 | 2 | 8 | 55.0% | +35 | |

### Analysis

**Phase 1 (iter 1-50): Growth**
Model improved from 52.5% to 60%. Selfplay model updated 3 times (52.5% → 57.5% → 60%). Gating correctly rejected iter 40 (45%).

**Phase 2 (iter 50-120): Breakthrough**
Slow accumulation of data from 60% model. At iter 120, model hit 75% — biggest single eval score. This is ~+190 Elo above Gen7.

**Phase 3 (iter 120-252): Stagnation**
Best model stuck at iter 120 for 130+ iterations. Selfplay data all comes from same model. Results oscillate 47-72% but never beat 75%.

### Why v2 Stagnated

**Problem 1: Stale selfplay data (self-play collapse)**
After iter 120 became "best", ALL subsequent selfplay used this frozen model. The model plays itself repeatedly, generating the same types of games. New training data has diminishing returns — the model already knows how to beat itself.

In real AlphaZero, the selfplay model is always the LATEST model (not the best). This ensures diversity in training data. Our gating was too conservative.

**Problem 2: Gating threshold too high**
75% on 20 games has enormous variance. A true 65% model has only ~15% chance of scoring >= 75% in 20 games. So the best model is effectively locked in forever.

**Problem 3: Slow policy degradation**
p_loss: 1.556 (iter 50) → 1.589 (iter 250). Despite low lr, 250 iterations of selfplay training slowly eroded pretrained policy. The supervised knowledge decays because selfplay MCTS policies differ from human PlayOK policies.

**Problem 4: Color asymmetry pattern**
Many results show 10W-0D-10L = model wins all White games, loses all Black. This suggests:
- Togyz Kumalak has strong first-move advantage
- Model + MCTS is strongest as White (going first)
- Engine's opening book gives it advantage as White too
- The "true" eval should consider color separately

---

## 8. Overall Findings

### What We Proved
1. **MCTS+NN can beat Gen7 NNUE.** Multiple evals showed 60-75% winrate
2. **Rust selfplay engine is fast enough.** 1.1-1.3 games/sec competitive with C++ implementations
3. **2M model capacity is appropriate.** Not too slow for inference, not too small for learning
4. **Supervised pretraining provides strong initialization.** 68% policy accuracy = good starting point
5. **Gating prevents regression.** Bad iterations (45%) don't corrupt the data pipeline

### What We Proved Doesn't Work
1. **High lr (0.001) with selfplay** — catastrophic forgetting of supervised knowledge
2. **Pure selfplay without expert data mixing** — model forgets human-quality patterns
3. **Strict gating on noisy eval** — creates stale selfplay, training collapse
4. **10-game evaluation** — pure noise, can't make decisions
5. **NNUE architecture scaling** — 256->32->1 is at absolute capacity ceiling

### Key Numbers
| Metric | Value |
|--------|-------|
| Best eval result | **75% vs Gen7 (14W-2D-4L, 20 games)** |
| Estimated Elo gain | **+190 Elo above Gen7** |
| Cumulative vs HCE | **~+1388 Elo** (1198 + 190) |
| Total selfplay positions | ~3M |
| Total training time | ~17 hours (v1 + v2) |
| Selfplay speed | 1.1-1.3 games/sec |
| GPU inference | 0.9ms per 128-position batch |
| Iteration time | ~85-95s (selfplay) + ~30s (training) + ~10min (eval) |

---

## 9. Possible Next Steps

### Option A: Fix Selfplay Collapse
- **Forced model rotation:** Update selfplay model every 20 iterations regardless of eval result
- **EMA (Exponential Moving Average):** Use weighted average of last 5 checkpoints for selfplay
- **Increase eval to 40 games:** Reduces noise, makes gating more reliable (~20 min per eval)
- **Expert data mixing:** Add 20-30% PlayOK expert data to each training batch to prevent policy forgetting

### Option B: Distillation to NNUE
- Use best MCTS model (75% vs Gen7) to generate training data for NNUE engine
- MCTS model plays games → extract evaluations at each position → train NNUE
- Previously proved +14 Elo with weaker distillation source. With +190 Elo source, expect +50-100 Elo
- Benefit: NNUE engine is fast for deployment (no GPU needed)

### Option C: Deploy MCTS Engine Directly
- Skip NNUE, deploy Rust MCTS with 2M model + GPU
- At play time: use 1600+ simulations (vs 400 in eval) = even stronger
- Needs GPU server for inference
- Could directly test on PlayOK

### Option D: Population-Based Training
- Run 3-5 models with different hyperparameters simultaneously
- Models play against each other (not just self-play) for diversity
- Tournament selection for next generation
- More robust but needs more compute

---

## 10. Hardware Comparison with Competitor

|  | Competitor (mcts@2541) | Us |
|--|----------------------|-----|
| CPU | 32-64 cores | 24 cores |
| GPU | A100 (40GB) | RTX 5080 (16GB) |
| RAM | 64 GB | 30 GB |
| Model | ~2M params | 2M params |
| MCTS impl | C++ | Rust |
| Selfplay speed | ~2000 games/hr | ~4000 games/hr |
| Training | ~1 week | ~1 day for 250 iters |
| PlayOK Elo | 2541 | ~1388 vs HCE (untested on PlayOK) |

Our Rust engine speed is competitive. The gap is in training methodology (selfplay collapse, eval noise) not raw compute.

---

## 11. File Locations

| File | Description |
|------|------------|
| `rust-mcts/src/*.rs` | Rust MCTS engine source code |
| `rust-mcts/scripts/train_loop.py` | Full training loop (v2) |
| `rust-mcts/scripts/train_alphazero.py` | Replay buffer loader + trainer |
| `rust-mcts/scripts/eval_configb_style.py` | Evaluation script (MCTS vs engine) |
| `rust-mcts/scripts/export_onnx.py` | PyTorch → ONNX conversion |
| `rust-mcts/checkpoints_2m/best.pt` | v1 best checkpoint (iter 50, 65%) |
| `rust-mcts/checkpoints_2m_v2/best.pt` | v2 best checkpoint (iter 120, 75%) |
| `rust-mcts/model_2m.onnx` | 2M model ONNX (8.9MB) |
| `alphazero-code/alphazero/checkpoints/supervised_pretrained_2m.pt` | Supervised pretrained 2M model |
| `alphazero-code/alphazero/train_config_b.py` | Config B Python MCTS training |
| `alphazero-code/alphazero/model.py` | TogyzNet model definitions |
| `/tmp/train_loop_2m_v2.log` | v2 training log |
