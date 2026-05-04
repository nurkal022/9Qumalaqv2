# Catch-Up to MCTS Rival — Design Spec

**Date:** 2026-05-04
**Status:** Draft → Pending user review
**Goal:** Bring our MCTS engine to within striking distance of (or surpass) the rival `mcts` bot on PlayOK that beats human champions at 2200-2500 ELO.

---

## 1. Context

### 1.1 Rival profile

The competitor runs a Togyz Kumalak MCTS bot under the nickname `mcts` on PlayOK. Their published / known setup:

- 1× A100 / L40S / RTX 4090 (16+ GB VRAM)
- 32-64 CPU cores
- 64 GB RAM
- Network: ~2M params, FP16 inference
- 600 simulations per move during selfplay
- 16-32 parallel selfplay workers (C++ MCTS)
- Reaches ~2500 ELO on PlayOK

### 1.2 Asymmetric advantage we hold

We have **1421 of their games** in our PlayOK scrape (`archive/datasets/parsed_games/all_games.json`):

- ELO trajectory: 1360 → 2541 across March-September 2025
- Mean ELO 2261, peak 2541
- Win rate 84.8% overall; 81% vs 2200+ opponents (158-5-32 of 195 games)
- Of these, **~926 games where mcts ELO ≥ 2300** are usable as gold-standard training data

The rival does not have access to our games. This data lets us shortcut their selfplay trajectory.

We also already have **360,054 raw `.txt` PGN files** in `archive/datasets/game-pars/games/` covering many strong human players (T2 candidates), and prior parsed datasets `training_elo2300.npz` (~32K positions) and `training_elo2000.npz` (~50K positions).

### 1.3 Our current state

- Hardware: RTX 5080 Laptop (16 GB VRAM), 24 CPU cores, 30 GB RAM
- Best checkpoint: v3 iter500 (2M params, p_loss 1.09)
- Selfplay throughput: 8-9 games/sec @ 200 sims
- Raw policy ~50% vs Gen7; Gumbel MCTS search 65-70% vs Gen7
- Rust MCTS infrastructure operational (selfplay, GPU inference, Gumbel root, playout cap randomization, expert data mixing)

### 1.4 Hardware budget

Local machine for now. Cloud rental (A100 ~$1.5/hr) is on the table if we hit a clear plateau. No fixed time horizon — result-driven.

---

## 2. Architecture

```
[1421 mcts games + new scrape]      [v3 iter500 checkpoint]
         │                                    │
         ▼                                    │
[tier-filtered expert dataset]                │
   T1: mcts ELO≥2300                          │
   T2: humans ELO≥2000 (no mcts)              │
         │                                    │
         ▼                                    ▼
[expert .bin files] ────────► [training loop] ◄──── [selfplay workers]
                                    │                (Rust MCTS:
                                    │                 Gumbel root +
                                    │                 forced playouts +
                                    │                 policy target pruning +
                                    │                 playout cap 25/75)
                                    ▼
                             [iter_N checkpoint]
                                    │
                                    ├──► [ONNX export] ──► [GPU inference batch server]
                                    │
                                    ▼
                             [tournament evaluator]
                                    │
                                    ▼
                       (every 10 iter: vs baseline pool;
                        every 50 iter: test set value/top-3;
                        on-demand: vs mcts on PlayOK)
```

### 2.1 What changes from v3

| Component | v3 | This iteration |
|---|---|---|
| Starting checkpoint | random init | v3 iter500 |
| Expert data | PlayOK ELO≥1400 (noisy, 20% mix) | T1 mcts 2300+ (15%) + T2 humans 2000+ (12%) |
| MCTS | Gumbel + playout cap | Gumbel + playout cap + forced playouts + policy target pruning |
| Progress metric | win rate vs Gen7 (uninformative ~50%) | tournament vs baseline pool (objective ELO delta) |

### 2.2 What stays the same

- Network architecture: large2m (~2M params)
- Playout cap randomization: 25% full (200 sims) / 75% fast (40 sims) during selfplay
- Dirichlet noise α=1.1 at root
- Batch GPU inference path (ONNX runtime, FP16)
- 200 games per training iteration

---

## 3. Components

### 3.1 Data pipeline

**Pre-training:**

1. **Incremental scrape** (running now, background): re-run `scrape_playok.py` with `mcts` as seed. Resumes from existing 360K downloads, BFS captures any new strong opponents. Expected to complete in 30-60 minutes.
2. **Tier extraction:** extend `load_expert_data` in `train_alphazero.py` to support tier filtering:
   - T1 = games where `mcts` is a player AND mcts ELO ≥ 2300
   - T2 = games where both players' ELO ≥ 2000 AND neither player is `mcts`
   - Output: two separate `(states, policies, values)` tensor sets
3. **Expected scale after extraction:** T1 ~50K positions (926 mcts 2300+ games × ~60 ply post-opening), T2 ~50-80K positions (depends on scrape completion).
4. **Blunder filter:** deferred. Initial run uses unfiltered T1 + T2. If T2 quality looks like a problem (e.g. value head learns wrong calls on test positions), revisit by scoring positions with our current network's value head and dropping those with > 0.4 swing.

**During training (every batch):**

```
batch composition:
  73% selfplay (from replay buffer, sliding window)
  15% T1 (uniform sample from mcts 2300+ positions)
  12% T2 (uniform sample from humans 2000+ positions)
```

### 3.2 MCTS algorithmic additions

**Forced playouts** (in `mcts.rs`, UCB selection path):

For every child action `a`, compute the minimum forced visit count:

```
n_forced(a) = ceil(sqrt(k_force * P(a) * sum_visits))
```

with `k_force = 2`. If a child has `visits(a) < n_forced(a)`, override the standard UCB selection and pick that child. Tie-break by policy prior.

**Policy target pruning** (in `self_play.rs`, when building policy target from final visit counts):

For each move at the root, subtract its forced visits from its visit count before normalizing into the policy target. Floor at zero. Then normalize.

```
visits_pruned(a) = max(0, visits(a) - n_forced(a))
policy_target(a) = visits_pruned(a) / sum(visits_pruned)
```

**Rationale:** forced visits exist to inject exploration diversity; treating them as the network's own confidence would train it to copy artificial exploration noise.

### 3.3 Tournament evaluator (new)

A new Rust binary or extended `eval_vs_engine` that runs round-robin head-to-head matches between a candidate checkpoint and a fixed pool of baselines.

**Pool composition:**

- v3 iter500 (anchor — the previous best)
- Previous accepted checkpoint from this run (most recent gate-passer)
- Gen7 engine (sanity check against the legacy NNUE engine)

**Match settings:**

- 100 games per pair (50 candidate-as-white + 50 candidate-as-black)
- 1000 sims/move (search-on, not raw policy)
- Both sides use Gumbel MCTS
- First 4 ply random (uniform over legal moves) to ensure opening diversity across the 100 games

**Output:** ELO delta of candidate vs each baseline, computed via simple logistic from win rate. A candidate "passes the gate" if ELO delta vs the most recent accepted checkpoint ≥ +20.

### 3.4 Test-set evaluator (new, secondary)

Runs every 50 iterations on a frozen test set of ~100 positions sampled from games where `mcts` won at ELO ≥ 2300.

**Metrics:**

- **Value head accuracy:** fraction of positions where `sign(value_pred) == sign(true_outcome_for_side_to_move)`
- **Top-3 policy hit rate:** fraction of positions where mcts's actual move appears in our top-3 policy probabilities

These are tracked as trend indicators only — not gates.

### 3.5 PlayOK anchor evaluator (manual, infrequent)

When tournament-eval shows the candidate stably > v3 iter500 by +200 ELO, manually run the candidate against `mcts` on PlayOK for ~30 games. This is the ground-truth check.

**Targets:**

- Minimum success: ≥40% win rate vs `mcts`
- Stretch: ≥50% (parity or better)

---

## 4. Training loop

### 4.1 Per-iteration flow (unchanged from v3 except batch composition)

```
for iter in 501..∞:
    1. selfplay 200 games using current ONNX model
       (200 sims, playout cap 25/75, forced playouts ON)
    2. add games to replay buffer (sliding window)
    3. train: mix replay + T1 + T2 per batch composition
       (forced visits subtracted from policy targets)
    4. save iter_N.pt + export iter_N.onnx
    5. log p_loss, v_loss, gradient norms
    6. if iter % 10 == 0: tournament eval
    7. if iter % 50 == 0: test-set eval
```

### 4.2 Decision gates

| Trigger | Condition | Action |
|---|---|---|
| Healthy progress | every 10 iter, candidate beats prev by ≥+20 ELO | continue, accept candidate as new baseline |
| Single plateau | one 10-iter window with no progress | continue (one bad window is normal noise) |
| Sustained plateau | 3 consecutive 10-iter windows with no progress (= 30 iter) | raise selfplay sims 200 → 400; revisit T1/T2 mix ratios |
| Hard plateau | 5 consecutive (= 50 iter) | escalate to cloud (A100 + 32-64 vCPU) |
| Regression | candidate loses by ≥30 ELO | rollback, investigate (LR schedule, expert mix, recent code change) |

### 4.3 Cloud escalation (contingent)

Triggered only by hard plateau. Setup:

- A100 (or L40S) instance, 32-64 vCPU, 64 GB RAM
- Transfer: latest checkpoint + replay buffer + expert .bin via scp
- Run for 2-4 weeks continuous
- Estimated cost: $500-2000

---

## 5. Implementation order (informs the plan, not yet the plan)

1. Tournament evaluator (we need objective measurement before changing anything else, otherwise we cannot tell if changes help)
2. Tier extraction in `load_expert_data` (data prep — independent of MCTS changes)
3. Forced playouts + policy target pruning in Rust MCTS
4. Smoke test: 1 iteration end-to-end with all changes
5. Long run: continuation from v3 iter500
6. (Conditional) cloud escalation

---

## 6. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Forced playouts hurt search depth | Medium | k_force = 2 is conservative; can tune down; gate via tournament eval after 10 iter |
| T1 too small (~50K), overfits to mcts style | Medium | Mix with T2 + heavy selfplay (73%); monitor test-set top-3 — if it saturates while tournament ELO stalls, that's the symptom |
| v3 iter500 starting point is locally trapped | Low-Medium | Fallback: cold restart from random + same data pipeline; expensive but available |
| Tournament eval too noisy at 100 games | Low | If observed, raise to 200 games or use Bayesian ELO with priors |
| `mcts` on PlayOK changes / disappears | Low | We have 1421 games archived; training is decoupled from their availability |
| Cloud escalation fails to break plateau | Medium | Re-examine architecture (we've assumed 2M params is enough — may need to revisit) |

---

## 7. Out of scope (deferred)

- Auxiliary heads (territory, owner) — large network refactor with unclear benefit for Togyz Kumalak
- MCTS subtree reuse between moves — risk/benefit poor
- LCB selection at root for final move pick — not critical for selfplay
- Architecture changes (transformer, deeper residual) — only revisit if hard plateau persists post-cloud
- NNUE — abandoned per prior decision (capacity ceiling)

---

## 8. Success criteria

- **Primary:** tournament ELO of final checkpoint ≥ v3 iter500 + 200
- **Anchor:** ≥40% win rate vs `mcts` on PlayOK over 30 games
- **Stretch:** ≥50% win rate vs `mcts` (parity or surpassing)
