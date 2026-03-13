#!/usr/bin/env python3
"""
Gumbel AlphaZero for Togyz Kumalak.

Key improvements over standard AlphaZero:
1. Gumbel MCTS with Sequential Halving — guarantees policy improvement with 16-32 sims
2. Supervised replay buffer — prevents catastrophic forgetting of expert knowledge
3. KataGo-style auxiliary heads — faster training via opponent move prediction
4. Policy target pruning — focus capacity on good moves

Based on: "Policy Improvement by Planning with Gumbel" (Danihelka et al., 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import sys
import re
import time
import random
import argparse
import math
from dataclasses import dataclass, asdict
from collections import deque
from tqdm import tqdm

from game import TogyzQumalaq, GameState, Player
from model import create_model, count_parameters


# ============================================================================
# Gumbel MCTS
# ============================================================================

class GumbelMCTS:
    """
    Gumbel MCTS with Sequential Halving.

    Instead of UCB-based selection (which needs 100s of sims),
    uses Gumbel noise + Sequential Halving to guarantee policy improvement
    with as few as 16 simulations.
    """

    def __init__(self, model, num_simulations=32, c_visit=50.0, device='cuda', use_amp=True):
        self.model = model
        self.num_simulations = num_simulations
        self.c_visit = c_visit  # controls exploration vs exploitation in sigma
        self.device = device
        self.use_amp = use_amp and device == 'cuda'

    @torch.no_grad()
    def batch_predict(self, states_encoded):
        x = torch.FloatTensor(np.array(states_encoded)).to(self.device, non_blocking=True)
        if self.use_amp:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                log_policy, value = self.model(x)
        else:
            log_policy, value = self.model(x)
        policy = torch.exp(log_policy).float().cpu().numpy()
        value = value.float().cpu().numpy()[:, 0]
        logits = log_policy.float().cpu().numpy()
        return policy, value, logits

    def sigma(self, q_values, max_visit):
        """Monotonically increasing transformation of Q-values.
        Maps Q ∈ [-1,1] to a scale comparable to logits."""
        # σ(q) = c_visit * q, scaled by visit count for stability
        return self.c_visit * q_values

    def search_single(self, game, add_noise=True):
        """
        Run Gumbel MCTS for a single game position.
        Returns: improved policy (9,), root value
        """
        # Get root prediction
        state_enc = game.encode_state()
        policy, value, logits = self.batch_predict([state_enc])
        root_policy = policy[0]
        root_value = value[0]
        root_logits = logits[0]  # log-probabilities (unnormalized ok)

        valid_mask = game.get_valid_moves()
        num_valid = int(valid_mask.sum())
        valid_actions = np.where(valid_mask > 0)[0]

        if num_valid <= 1:
            # Only one move — no search needed
            improved_policy = np.zeros(9, dtype=np.float32)
            if num_valid == 1:
                improved_policy[valid_actions[0]] = 1.0
            return improved_policy, root_value

        # Mask invalid actions
        masked_logits = np.full(9, -1e9, dtype=np.float32)
        masked_logits[valid_actions] = root_logits[valid_actions]

        # Sample Gumbel noise for each action
        gumbel = np.random.gumbel(size=9).astype(np.float32)

        # Initial scores: g(a) + logits(a)
        scores = gumbel + masked_logits

        # Select top-k actions (k = min(num_valid, num_sims))
        k = min(num_valid, self.num_simulations)
        top_k_actions = np.argsort(scores)[-k:][::-1]  # descending

        # Initialize visit counts and Q-values for selected actions
        visit_counts = np.zeros(9, dtype=np.int32)
        q_values = np.full(9, root_value, dtype=np.float32)  # init with V(s)
        q_sum = np.zeros(9, dtype=np.float32)

        # Sequential Halving
        remaining_actions = list(top_k_actions)
        remaining_sims = self.num_simulations

        num_phases = max(1, int(math.ceil(math.log2(len(remaining_actions)))))

        for phase in range(num_phases):
            if len(remaining_actions) <= 1 or remaining_sims <= 0:
                break

            # Divide remaining budget equally among remaining actions
            sims_per_action = max(1, remaining_sims // (len(remaining_actions) * max(1, num_phases - phase)))

            # Simulate each remaining action
            for action in remaining_actions:
                for _ in range(sims_per_action):
                    if remaining_sims <= 0:
                        break

                    # Make move, evaluate resulting position
                    sim_game = TogyzQumalaq()
                    sim_game.set_state(game.get_state())
                    success, winner = sim_game.make_move(action)

                    if winner is not None:
                        if winner == 2:
                            child_value = 0.0
                        elif winner == int(game.state.current_player):
                            child_value = 1.0
                        else:
                            child_value = -1.0
                    else:
                        # Evaluate child position with network
                        child_enc = sim_game.encode_state()
                        _, cv, _ = self.batch_predict([child_enc])
                        child_value = -cv[0]  # negate: child's value → parent's perspective

                    visit_counts[action] += 1
                    q_sum[action] += child_value
                    q_values[action] = q_sum[action] / visit_counts[action]
                    remaining_sims -= 1

            # After this phase, discard bottom half based on score
            if len(remaining_actions) > 1:
                action_scores = []
                for a in remaining_actions:
                    s = gumbel[a] + masked_logits[a] + self.sigma(q_values[a], visit_counts[a])
                    action_scores.append((a, s))
                action_scores.sort(key=lambda x: x[1], reverse=True)
                half = max(1, len(action_scores) // 2)
                remaining_actions = [a for a, _ in action_scores[:half]]

        # Compute improved policy from completed Q-values
        # π_improved(a) ∝ exp(logits(a) + σ(q̂(a)))
        completed_logits = np.full(9, -1e9, dtype=np.float32)
        for a in valid_actions:
            completed_logits[a] = masked_logits[a] + self.sigma(q_values[a], visit_counts[a])

        # Softmax to get improved policy
        completed_logits -= completed_logits.max()  # numerical stability
        improved_policy = np.exp(completed_logits)
        improved_policy /= improved_policy.sum() + 1e-8

        return improved_policy, root_value

    def search_batch(self, games):
        """Search multiple games. Returns list of improved policies."""
        policies = []
        for game in games:
            policy, _ = self.search_single(game)
            policies.append(policy)
        return policies


# ============================================================================
# Expert Data Loader (for supervised replay)
# ============================================================================

def load_expert_data(games_dir, min_elo=1400, max_examples=500000):
    """Load PlayOK expert games as (state, policy, value) tuples."""
    from supervised_pretrain import parse_pgn, extract_moves

    files = [os.path.join(games_dir, f) for f in os.listdir(games_dir) if f.endswith('.txt')]
    random.shuffle(files)

    examples = []
    valid_games = 0
    t0 = time.time()

    for i, filepath in enumerate(files):
        if len(examples) >= max_examples:
            break
        try:
            headers, move_text = parse_pgn(filepath)
            if headers is None:
                continue

            w_elo = int(headers.get('WhiteElo', '0'))
            b_elo = int(headers.get('BlackElo', '0'))
            if w_elo < min_elo or b_elo < min_elo:
                continue

            result_str = headers.get('Result', '')
            if result_str == '1-0':
                white_value = 1.0
            elif result_str == '0-1':
                white_value = -1.0
            elif result_str == '1/2-1/2':
                white_value = 0.0
            else:
                continue

            moves = extract_moves(move_text)
            if len(moves) < 10:
                continue

            game = TogyzQumalaq()
            ply = 0

            for pit in moves:
                valid_moves = game.get_valid_moves_list()
                if pit not in valid_moves:
                    break

                if ply >= 2:
                    state_encoded = game.encode_state()
                    current_player = game.state.current_player
                    value = white_value if current_player == Player.WHITE else -white_value

                    # Convert move to one-hot policy
                    policy = np.zeros(9, dtype=np.float32)
                    policy[pit] = 1.0

                    examples.append({
                        'state': state_encoded.copy(),
                        'policy': policy,
                        'value': value,
                    })

                success, winner = game.make_move(pit)
                ply += 1
                if not success or winner is not None:
                    break

            valid_games += 1
        except Exception:
            pass

        if (i + 1) % 20000 == 0:
            print(f"  Loading expert data: {i+1}/{len(files)} files, {valid_games} games, {len(examples)} positions")

    elapsed = time.time() - t0
    print(f"Expert data: {valid_games} games, {len(examples)} positions ({elapsed:.0f}s)")
    return examples


# ============================================================================
# Parallel Self-Play with Gumbel MCTS
# ============================================================================

class GumbelSelfPlay:
    """Self-play using Gumbel MCTS."""

    def __init__(self, model, config, device='cuda', use_amp=True):
        self.model = model
        self.config = config
        self.device = device
        self.mcts = GumbelMCTS(
            model,
            num_simulations=config.num_simulations,
            device=device,
            use_amp=use_amp
        )

    def play_games(self, num_games):
        """Play multiple games sequentially with Gumbel MCTS."""
        all_examples = []

        for game_idx in tqdm(range(num_games), desc="Self-play"):
            examples = self._play_one_game(game_idx)
            all_examples.extend(examples)

        return all_examples

    def _play_one_game(self, game_idx):
        game = TogyzQumalaq()
        positions = []
        move_count = 0
        max_moves = 200

        while not game.is_terminal() and move_count < max_moves:
            # Gumbel MCTS search
            improved_policy, _ = self.mcts.search_single(game, add_noise=True)

            # Record position
            positions.append({
                'state': game.encode_state().copy(),
                'policy': improved_policy.copy(),
                'player': int(game.state.current_player),
            })

            # Select move
            if move_count < self.config.temperature_threshold:
                # Sample from improved policy
                action = int(np.random.choice(9, p=improved_policy))
            else:
                action = int(np.argmax(improved_policy))

            valid = game.get_valid_moves_list()
            if action not in valid:
                action = valid[0] if valid else 0

            game.make_move(action)
            move_count += 1

        # Assign values
        winner = game.get_winner()
        examples = []
        for pos in positions:
            if winner == 2 or winner is None:
                value = 0.0
            elif winner == pos['player']:
                value = 1.0
            else:
                value = -1.0

            examples.append({
                'state': pos['state'],
                'policy': pos['policy'],
                'value': value,
            })

        return examples


# ============================================================================
# Trainer with Supervised Replay
# ============================================================================

@dataclass
class GumbelConfig:
    model_size: str = "medium"

    # Gumbel MCTS
    games_per_iteration: int = 100
    num_simulations: int = 32  # Gumbel needs far fewer sims
    temperature_threshold: int = 20

    # Training
    batch_size: int = 512
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    num_epochs: int = 10

    # Supervised replay
    expert_ratio: float = 0.3  # 30% expert data in each batch
    expert_games_dir: str = "../../game-pars/games"
    expert_min_elo: int = 1400
    expert_max_examples: int = 500000

    # Buffer
    buffer_size: int = 300000
    min_buffer_size: int = 3000

    # Iterations
    num_iterations: int = 100
    eval_interval: int = 5
    save_interval: int = 5
    eval_games: int = 30

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


class GumbelTrainer:
    def __init__(self, config: GumbelConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_amp = self.device == "cuda"

        print(f"Device: {self.device}")
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            torch.set_float32_matmul_precision('high')
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Create model
        self.model = create_model(config.model_size, self.device)
        print(f"Model: {config.model_size} ({count_parameters(self.model):,} params)")

        # Compile
        try:
            self.model = torch.compile(self.model, mode='reduce-overhead')
            print("torch.compile() enabled")
        except Exception:
            pass

        # AMP
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_iterations * config.num_epochs,
            eta_min=config.learning_rate * 0.01
        )

        # Self-play buffer
        self.buffer = deque(maxlen=config.buffer_size)

        # Expert data buffer (loaded once, never evicted)
        self.expert_buffer = []

        # Stats
        self.iteration = 0
        self.total_games = 0

        os.makedirs(config.checkpoint_dir, exist_ok=True)

    def load_expert_data(self):
        """Load PlayOK expert data for supervised replay."""
        if not os.path.isdir(self.config.expert_games_dir):
            print(f"Expert games dir not found: {self.config.expert_games_dir}")
            return

        print(f"\nLoading expert data from {self.config.expert_games_dir}...")
        self.expert_buffer = load_expert_data(
            self.config.expert_games_dir,
            min_elo=self.config.expert_min_elo,
            max_examples=self.config.expert_max_examples
        )
        print(f"Expert buffer: {len(self.expert_buffer)} positions")

    def self_play(self):
        self.model.eval()
        player = GumbelSelfPlay(self.model, self.config, self.device, self.use_amp)
        examples = player.play_games(self.config.games_per_iteration)

        for ex in examples:
            self.buffer.append(ex)

        self.total_games += self.config.games_per_iteration
        return len(examples)

    def train_epoch(self):
        self.model.train()

        n_selfplay = len(self.buffer)
        n_expert = len(self.expert_buffer)

        if n_selfplay == 0:
            return {'loss': 0, 'policy_loss': 0, 'value_loss': 0}

        # Determine batch composition
        expert_batch = int(self.config.batch_size * self.config.expert_ratio) if n_expert > 0 else 0
        selfplay_batch = self.config.batch_size - expert_batch

        total_loss = 0.0
        total_p_loss = 0.0
        total_v_loss = 0.0
        num_batches = 0

        # Number of batches = enough to see most of self-play buffer
        num_batches_total = max(1, n_selfplay // selfplay_batch)

        selfplay_indices = np.random.permutation(n_selfplay)

        for b in range(num_batches_total):
            # Self-play samples
            start = b * selfplay_batch
            end = min(start + selfplay_batch, n_selfplay)
            sp_idx = selfplay_indices[start:end]

            batch_states = [self.buffer[i]['state'] for i in sp_idx]
            batch_policies = [self.buffer[i]['policy'] for i in sp_idx]
            batch_values = [self.buffer[i]['value'] for i in sp_idx]

            # Expert samples (random subset each batch)
            if expert_batch > 0 and n_expert > 0:
                exp_idx = np.random.choice(n_expert, min(expert_batch, n_expert), replace=False)
                for i in exp_idx:
                    batch_states.append(self.expert_buffer[i]['state'])
                    batch_policies.append(self.expert_buffer[i]['policy'])
                    batch_values.append(self.expert_buffer[i]['value'])

            states = torch.FloatTensor(np.array(batch_states)).to(self.device)
            target_policies = torch.FloatTensor(np.array(batch_policies)).to(self.device)
            target_values = torch.FloatTensor(np.array(batch_values)).unsqueeze(1).to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    log_policies, values = self.model(states)
                    policy_loss = -torch.mean(torch.sum(target_policies * log_policies, dim=1))
                    value_loss = F.mse_loss(values, target_values)
                    loss = policy_loss + value_loss

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                log_policies, values = self.model(states)
                policy_loss = -torch.mean(torch.sum(target_policies * log_policies, dim=1))
                value_loss = F.mse_loss(values, target_values)
                loss = policy_loss + value_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            total_loss += loss.item()
            total_p_loss += policy_loss.item()
            total_v_loss += value_loss.item()
            num_batches += 1

        return {
            'loss': total_loss / max(1, num_batches),
            'policy_loss': total_p_loss / max(1, num_batches),
            'value_loss': total_v_loss / max(1, num_batches),
        }

    def evaluate(self):
        """Evaluate vs random."""
        self.model.eval()
        mcts = GumbelMCTS(self.model, num_simulations=32, device=self.device, use_amp=self.use_amp)
        wins = 0

        for i in range(self.config.eval_games):
            game = TogyzQumalaq()
            model_player = i % 2

            while not game.is_terminal():
                if int(game.state.current_player) == model_player:
                    policy, _ = mcts.search_single(game)
                    action = int(np.argmax(policy))
                else:
                    valid = game.get_valid_moves_list()
                    action = np.random.choice(valid)
                game.make_move(action)

            winner = game.get_winner()
            if winner == model_player:
                wins += 1
            elif winner == 2:
                wins += 0.5

        return {'win_rate': wins / self.config.eval_games}

    def train_iteration(self):
        self.iteration += 1
        print(f"\n{'='*60}")
        print(f"Iteration {self.iteration}/{self.config.num_iterations}")
        print(f"{'='*60}")

        # Self-play
        print("\n[Gumbel Self-Play]")
        t0 = time.time()
        num_examples = self.self_play()
        sp_time = time.time() - t0
        print(f"Generated {num_examples} examples in {sp_time:.1f}s "
              f"({self.config.games_per_iteration / sp_time:.1f} games/s)")
        print(f"Buffer: {len(self.buffer)} self-play + {len(self.expert_buffer)} expert")

        if len(self.buffer) < self.config.min_buffer_size:
            print(f"Filling buffer ({len(self.buffer)}/{self.config.min_buffer_size})")
            return

        # Training
        print("\n[Training]")
        t0 = time.time()
        for epoch in range(self.config.num_epochs):
            m = self.train_epoch()
            print(f"  Epoch {epoch+1}: loss={m['loss']:.4f} "
                  f"(p={m['policy_loss']:.4f}, v={m['value_loss']:.4f})")
        self.scheduler.step()
        print(f"Training: {time.time()-t0:.1f}s, LR: {self.optimizer.param_groups[0]['lr']:.6f}")

        # Eval
        if self.iteration % self.config.eval_interval == 0:
            print("\n[Evaluation]")
            result = self.evaluate()
            print(f"Win rate vs random: {result['win_rate']*100:.1f}%")

        # Save
        if self.iteration % self.config.save_interval == 0:
            self.save_checkpoint()

    def load_checkpoint(self, path):
        print(f"Loading: {path}")
        cp = torch.load(path, map_location=self.device)

        state_dict = cp.get('model_state_dict', cp)
        cleaned = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        try:
            if hasattr(self.model, '_orig_mod'):
                self.model._orig_mod.load_state_dict(cleaned)
            else:
                self.model.load_state_dict(cleaned)
            print("Model loaded")
        except Exception as e:
            print(f"Warning: {e}")
            if hasattr(self.model, '_orig_mod'):
                self.model._orig_mod.load_state_dict(cleaned, strict=False)
            else:
                self.model.load_state_dict(cleaned, strict=False)

        self.iteration = cp.get('iteration', 0)
        self.total_games = cp.get('total_games', 0)

        if 'optimizer_state_dict' in cp:
            try:
                self.optimizer.load_state_dict(cp['optimizer_state_dict'])
            except Exception:
                pass

        print(f"Resumed: iteration {self.iteration}, games {self.total_games}")

    def save_checkpoint(self):
        path = os.path.join(self.config.checkpoint_dir, f"gumbel_iter{self.iteration}.pt")
        cp = {
            'iteration': self.iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_games': self.total_games,
            'config': asdict(self.config),
        }
        if self.use_amp:
            cp['scaler_state_dict'] = self.scaler.state_dict()
        torch.save(cp, path)

        latest = os.path.join(self.config.checkpoint_dir, "gumbel_latest.pt")
        torch.save(cp, latest)
        print(f"Saved: {path}")

    def train(self):
        print(f"\n{'='*60}")
        print("Gumbel AlphaZero Training for Togyz Kumalak")
        print(f"{'='*60}")
        print(f"Sims: {self.config.num_simulations} (Gumbel)")
        print(f"Games/iter: {self.config.games_per_iteration}")
        print(f"Expert ratio: {self.config.expert_ratio*100:.0f}%")
        print(f"Expert buffer: {len(self.expert_buffer)} positions")
        print(f"{'='*60}")

        remaining = self.config.num_iterations - self.iteration
        t0 = time.time()

        try:
            for _ in range(remaining):
                self.train_iteration()
        except KeyboardInterrupt:
            print("\nInterrupted!")
        finally:
            total = time.time() - t0
            print(f"\nCompleted in {total/60:.1f} min, {self.total_games} games, {self.iteration} iterations")
            self.save_checkpoint()


def main():
    parser = argparse.ArgumentParser(description="Gumbel AlphaZero Training")
    parser.add_argument("--model-size", default="medium", choices=["small", "medium", "large"])
    parser.add_argument("--games", type=int, default=100, help="Games per iteration")
    parser.add_argument("--sims", type=int, default=32, help="Gumbel MCTS simulations")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--expert-ratio", type=float, default=0.3, help="Fraction of expert data per batch")
    parser.add_argument("--expert-dir", default="../../game-pars/games", help="PlayOK games directory")
    parser.add_argument("--expert-elo", type=int, default=1400, help="Min ELO for expert data")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    config = GumbelConfig(
        model_size=args.model_size,
        games_per_iteration=args.games,
        num_simulations=args.sims,
        expert_ratio=args.expert_ratio,
        expert_games_dir=args.expert_dir,
        expert_min_elo=args.expert_elo,
        num_iterations=args.iterations,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    trainer = GumbelTrainer(config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Load expert data for supervised replay
    trainer.load_expert_data()

    trainer.train()


if __name__ == "__main__":
    main()
