# Why MCTS Training Isn't Working — Root Cause Analysis

## The Core Problem

We trained 5 versions of MCTS models (v1-v5) over 2000+ total iterations. Loss dropped dramatically (p_loss 1.68 → 0.72, -57%). But **playing strength never improved beyond 50% vs Gen7 engine.**

The model learns to predict its own moves perfectly. But it doesn't learn to play well against an opponent.

## What We Tried

| Version | Selfplay | Sims | Speed | p_loss | Eval vs Gen7 |
|---------|----------|------|-------|--------|-------------|
| v1 | Deep MCTS (PUCT) | 800 | 1.1 g/s | 1.56 | 50-55% |
| v2 | Deep MCTS + gating | 800 | 1.1 g/s | 1.59 | 50-75% (noisy) |
| v3 | Deep MCTS + playout cap | 200 | 8.7 g/s | 1.09 | 50-62% |
| v4 | Deep MCTS + score values | 400 | 5.9 g/s | 1.17 | 50% |
| v5 | Gumbel MCTS | 1-ply | 17 g/s | 0.72 | 50% |

Every version converges to **50% = wins as White, loses as Black** (first-move advantage).

## The Three Interlocking Problems

### Problem 1: Self-Play Echo Chamber

AlphaZero selfplay works because each iteration makes the model slightly stronger, generating better training data, creating a virtuous cycle. But this requires the model to **actually improve at the game**, not just at predicting its own moves.

Our model plays against itself. It learns patterns like "when I play move X, I tend to win." But these patterns are specific to its own play style. Against an engine with completely different strategy (alpha-beta search, opening book, NNUE eval), these patterns are useless.

**Evidence:** p_loss dropped from 1.68 to 0.72 (model predicts its own moves with 72%+ accuracy), but eval vs engine stayed at 50%.

### Problem 2: First-Move Advantage Dominates

In Togyz Kumalak, White (first player) has a massive advantage. Our model-vs-model test showed **10W-0D-10L** — whoever moves first wins, regardless of which model plays. Even our raw policy vs engine: 5W-0D-5L = perfect White wins, perfect Black losses.

This means:
- Model playing White vs Engine playing Black → Model wins (not because model is good, but because White advantage)
- Model playing Black vs Engine playing White → Engine wins (same reason)
- Net: 50% always, regardless of model strength

The model would need to be **significantly** stronger than the engine to overcome first-move advantage and win as Black. We're not there yet.

### Problem 3: Evaluation Method Mismatch

Python Gumbel MCTS eval showed 65-70% but was misleading:
- Gumbel noise randomly helps sometimes (lucky move selection)
- Sequential Halving with deterministic 1-ply = same as single eval
- The 65-70% was variance from Gumbel noise, not real strength

Rust 1-ply eval (deterministic, no noise) consistently shows 50%. This is the accurate measurement.

## What We Built That Works

1. **Rust MCTS Engine**: 17 games/sec selfplay, GPU batched inference (0.9ms/batch)
2. **2M Parameter Model**: 68% supervised accuracy, p_loss trained down to 0.72
3. **Gumbel MCTS in Rust**: Matching Python implementation, faster
4. **Full Training Pipeline**: Selfplay → Train → Export → Eval, all automated
5. **Eval vs Engine in Rust**: Color-paired, deterministic, 1 min per 20 games
6. **Score-proportional values**: Better value training signal
7. **PlayOK 2000+ data**: 128K expert positions for training

## What Actually Needs to Change

### The Fundamental Issue

Self-play AlphaZero works for games where:
- The model starts from scratch (random) and gradually improves
- OR the model is already strong enough that self-play produces diverse, challenging games

Our model starts from supervised pretraining (68% accuracy) which is decent but not strong enough for productive self-play. It plays stereotyped games against itself, and training on those games doesn't make it stronger against different opponents.

### Option 1: Train Against the Engine (Most Direct)

Instead of self-play (model vs model), play **model vs engine** for training data. The engine plays differently (opening book, deep alpha-beta search, NNUE eval). This forces the model to learn strategies that work against strong, different opponents.

- Each game: model plays one side, engine plays the other
- Policy target: model's actual moves (or improved policy from search)
- Value target: game outcome from model's perspective
- This is closer to "reinforcement learning against a fixed opponent"

**Pros:** Direct signal for improving against the actual opponent we care about
**Cons:** May overfit to engine's specific weaknesses; slower (engine takes 200-500ms/move)

### Option 2: Mixed Opponents (Self-Play + Engine + Historical)

Play against a mix:
- 40% vs engine (learn to beat the target)
- 30% vs self (maintain general strength)
- 30% vs historical checkpoints (diversity)

This prevents overfitting to one opponent while still learning useful strategies.

### Option 3: Supervised Learning on Strong Data Only

Abandon self-play entirely. Train purely on:
- PlayOK 2000+ games (128K positions)
- Master games from server (when available)
- Possibly download more high-Elo games

The model was 68% accuracy after supervised pretraining. With better data (2000+ only) and longer training, it could reach 75%+. This doesn't need self-play at all.

**Pros:** Simple, proven to work (supervised pretraining was our best starting point)
**Cons:** Ceiling is limited by data quality; can't discover novel strategies

### Option 4: Engine-Guided Self-Play (Hybrid)

Use the engine as a "teacher":
1. Model makes a move candidate
2. Engine evaluates the resulting position (gives score)
3. Use engine's score as the value target (not game outcome)
4. This gives per-move feedback, not just end-of-game

**Pros:** Rich signal (every move gets scored), combines engine strength with model learning
**Cons:** Engine eval may not be accurate enough; adds complexity

## Recommended Path

**Start with Option 1** (train against engine) because:
- It's the most direct path to beating the engine
- We already have all the infrastructure (Rust selfplay, engine serve protocol)
- Model plays as both colors, alternating, so it learns to win as Black too
- Training data is always fresh and challenging (engine plays well)

If that plateaus, switch to **Option 2** (mixed opponents).

If no self-play approach works, fall back to **Option 3** (pure supervised on 2000+ data).

## Time Estimates

| Approach | Implementation | Training | Total |
|----------|---------------|----------|-------|
| Option 1: vs Engine | 2-3 hours | 6-12 hours | 1 day |
| Option 2: Mixed | 4-6 hours | 12-24 hours | 2 days |
| Option 3: Supervised | 1 hour | 2-4 hours | 4 hours |
| Option 4: Engine-guided | 6-8 hours | 12-24 hours | 2-3 days |
