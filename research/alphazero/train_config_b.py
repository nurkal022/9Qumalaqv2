#!/usr/bin/env python3
"""
Config B: Supervised Pretrain + TrueBatchMCTS 800 sims
=======================================================
Takes the best of both worlds:
- TrueBatchMCTS from train_fast.py (proper multi-ply tree search)
- Expert replay buffer from gumbel_az.py (prevents forgetting)
- supervised_pretrained.pt as starting point (67.6% move accuracy)

Key parameters (Config B from expert analysis):
- Simulations: 800 (proper tree MCTS, not 1-ply Gumbel)
- c_puct: 2.5 (higher exploration since supervised policy already good)
- Dirichlet: α=0.5, ε=0.25
- Temperature: τ=1.0 first 15 moves, then τ→0
- Expert replay: 30% of each training batch
- Eval: every 10 iterations vs Gen7 alpha-beta engine (1s/move)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import sys
import time
import random
import argparse
import subprocess
import tempfile
import shutil
from dataclasses import dataclass, asdict
from collections import deque
from tqdm import tqdm

from game import TogyzQumalaq, GameState, Player
from model import create_model, count_parameters
from train_fast import TrueBatchMCTS, MCTSNodeFast, ParallelSelfPlay, FastConfig
from gumbel_az import load_expert_data


# ============================================================================
# Config B parameters
# ============================================================================

@dataclass
class ConfigB:
    model_size: str = "medium"

    # Self-play with TrueBatchMCTS
    games_per_iteration: int = 100
    num_simulations: int = 800
    c_puct: float = 2.5          # Higher for supervised pretrained policy
    dirichlet_alpha: float = 0.5  # More noise (supervised policy is strong)
    dirichlet_eps: float = 0.25
    temperature_threshold: int = 15
    batch_size_games: int = 8     # Games in parallel

    # Training
    batch_size: int = 512
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    num_epochs: int = 10

    # Expert replay
    expert_ratio: float = 0.3
    expert_games_dir: str = "../../game-pars/games"
    expert_min_elo: int = 1400
    expert_max_examples: int = 500000

    # Buffer
    buffer_size: int = 500000
    min_buffer_size: int = 3000

    # Iterations
    num_iterations: int = 200
    eval_interval: int = 10
    save_interval: int = 5
    eval_games_vs_engine: int = 20  # Games vs Gen7 alpha-beta
    eval_engine_time_ms: int = 1000
    early_stop_iter: int = 30      # Check after this many iters
    early_stop_min_wr: float = 0.05  # Min winrate to continue

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    engine_path: str = "../../engine/target/release/togyzkumalaq-engine"
    engine_weights: str = "../../engine/nnue_weights_gen7.bin"


# ============================================================================
# Extended TrueBatchMCTS with configurable parameters
# ============================================================================

class ConfigurableMCTS(TrueBatchMCTS):
    """TrueBatchMCTS with configurable c_puct and Dirichlet noise."""

    def __init__(self, model, num_simulations=800, c_puct=2.5,
                 dirichlet_alpha=0.5, dirichlet_eps=0.25,
                 device='cuda', use_amp=True):
        super().__init__(model, num_simulations, c_puct, device, use_amp)
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps


# ============================================================================
# Self-play with configurable MCTS
# ============================================================================

class ConfigBSelfPlay:
    """Self-play using TrueBatchMCTS with Config B parameters."""

    def __init__(self, model, config: ConfigB, device='cuda', use_amp=True):
        self.model = model
        self.config = config
        self.device = device
        self.mcts = ConfigurableMCTS(
            model,
            num_simulations=config.num_simulations,
            c_puct=config.c_puct,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_eps=config.dirichlet_eps,
            device=device,
            use_amp=use_amp,
        )

    def play_games(self, num_games: int) -> list:
        all_examples = []
        batch_size = self.config.batch_size_games

        num_batches = (num_games + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches), desc="Self-play"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_games)
            batch_games = end_idx - start_idx
            batch_examples = self._play_batch(batch_games)
            all_examples.extend(batch_examples)

        return all_examples

    def _play_batch(self, num_games: int) -> list:
        games = [TogyzQumalaq() for _ in range(num_games)]
        all_examples = [[] for _ in range(num_games)]
        active_indices = list(range(num_games))
        move_counts = [0] * num_games
        max_moves = 200

        while active_indices:
            active_games = [games[i] for i in active_indices]
            policies = self.mcts.search_batch(active_games)

            new_active = []
            for idx, policy in zip(active_indices, policies):
                game = games[idx]
                move_count = move_counts[idx]

                if game.is_terminal() or move_count >= max_moves:
                    continue

                all_examples[idx].append({
                    'state': game.encode_state().copy(),
                    'policy': policy.copy(),
                    'player': int(game.state.current_player),
                })

                if move_count < self.config.temperature_threshold:
                    action = int(np.random.choice(9, p=policy))
                else:
                    action = int(np.argmax(policy))

                valid = game.get_valid_moves_list()
                if action not in valid:
                    action = valid[0] if valid else 0

                game.make_move(action)
                move_counts[idx] += 1

                if not game.is_terminal() and move_counts[idx] < max_moves:
                    new_active.append(idx)

            active_indices = new_active

        # Assign values based on game outcomes
        training_examples = []
        for idx, examples in enumerate(all_examples):
            winner = games[idx].get_winner()
            for ex in examples:
                if winner == 2 or winner is None:
                    value = 0.0
                elif winner == ex['player']:
                    value = 1.0
                else:
                    value = -1.0
                training_examples.append({
                    'state': ex['state'],
                    'policy': ex['policy'],
                    'value': value,
                })

        return training_examples


# ============================================================================
# Evaluation vs Gen7 Alpha-Beta Engine
# ============================================================================

def eval_vs_engine(model, config: ConfigB, device='cuda', use_amp=True):
    """
    Play games against the Gen7 alpha-beta NNUE engine.
    MCTS model plays with TrueBatchMCTS (800 sims) vs engine (1s/move).
    """
    engine_path = os.path.abspath(config.engine_path)
    weights_path = os.path.abspath(config.engine_weights)

    if not os.path.exists(engine_path):
        print(f"  Engine not found: {engine_path}")
        return {'win_rate': -1, 'wins': 0, 'draws': 0, 'losses': 0}
    if not os.path.exists(weights_path):
        print(f"  Weights not found: {weights_path}")
        return {'win_rate': -1, 'wins': 0, 'draws': 0, 'losses': 0}

    # Create temp dir with engine weights + egtb + opening book
    tmpdir = tempfile.mkdtemp(prefix="mcts_eval_")
    try:
        # Symlink engine dependencies
        engine_dir = os.path.dirname(weights_path)
        shutil.copy2(weights_path, os.path.join(tmpdir, "nnue_weights.bin"))
        egtb = os.path.join(engine_dir, "egtb.bin")
        book = os.path.join(engine_dir, "opening_book.txt")
        if os.path.exists(egtb):
            os.symlink(egtb, os.path.join(tmpdir, "egtb.bin"))
        if os.path.exists(book):
            os.symlink(book, os.path.join(tmpdir, "opening_book.txt"))

        # Start engine process
        proc = subprocess.Popen(
            [engine_path, "serve"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=tmpdir, text=True, bufsize=1,
        )

        # Wait for engine to be ready
        while True:
            line = proc.stdout.readline().strip()
            if line == "ready":
                break
            if not line:
                break

        # MCTS searcher for model
        mcts = ConfigurableMCTS(
            model, num_simulations=config.num_simulations, c_puct=config.c_puct,
            dirichlet_alpha=0.0, dirichlet_eps=0.0,  # No noise during eval
            device=device, use_amp=use_amp,
        )

        wins, draws, losses = 0, 0, 0

        for game_idx in range(config.eval_games_vs_engine):
            mcts_is_white = (game_idx % 2 == 0)

            game = TogyzQumalaq()
            move_count = 0

            while not game.is_terminal() and move_count < 200:
                current_player = int(game.state.current_player)
                mcts_turn = (current_player == 0 and mcts_is_white) or \
                            (current_player == 1 and not mcts_is_white)

                if mcts_turn:
                    # MCTS move
                    policy = mcts.search_batch([game])[0]
                    action = int(np.argmax(policy))
                else:
                    # Engine move
                    action = get_engine_move(proc, game, config.eval_engine_time_ms)
                    if action < 0:
                        break

                valid = game.get_valid_moves_list()
                if action not in valid:
                    action = valid[0] if valid else 0

                game.make_move(action)
                move_count += 1

            winner = game.get_winner()
            mcts_player = 0 if mcts_is_white else 1

            if winner == 2 or winner is None:
                draws += 1
            elif winner == mcts_player:
                wins += 1
            else:
                losses += 1

            total = game_idx + 1
            wr = (wins + 0.5 * draws) / total
            print(f"  [{total}/{config.eval_games_vs_engine}] "
                  f"MCTS: {wins}W-{draws}D-{losses}L ({wr*100:.1f}%)")

        proc.stdin.write("quit\n")
        proc.stdin.flush()
        proc.terminate()

        total = wins + draws + losses
        win_rate = (wins + 0.5 * draws) / max(1, total)
        return {'win_rate': win_rate, 'wins': wins, 'draws': draws, 'losses': losses}

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def get_engine_move(proc, game, time_ms):
    """Ask the engine process for a move via the serve protocol."""
    # Encode position for engine
    state = game.get_state()
    w_pits = ",".join(str(state.pits[0][i]) for i in range(9))
    b_pits = ",".join(str(state.pits[1][i]) for i in range(9))
    kw, kb = int(state.kazan[0]), int(state.kazan[1])
    tw = int(state.tuzdyk[0]) if state.tuzdyk[0] >= 0 else -1
    tb = int(state.tuzdyk[1]) if state.tuzdyk[1] >= 0 else -1
    side = "w" if state.current_player == 0 else "b"
    pos_str = f"{w_pits}/{b_pits}/{kw},{kb}/{tw},{tb}/{side}"

    cmd = f"go pos {pos_str} time {time_ms}\n"
    try:
        proc.stdin.write(cmd)
        proc.stdin.flush()

        response = proc.stdout.readline().strip()
        if response.startswith("bestmove"):
            parts = response.split()
            move = int(parts[1])
            return move
        elif response.startswith("terminal"):
            return -1
        return -1
    except Exception as e:
        print(f"  Engine error: {e}")
        return -1


# ============================================================================
# Trainer (Config B)
# ============================================================================

class ConfigBTrainer:
    def __init__(self, config: ConfigB):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_amp = self.device == "cuda"

        print(f"Device: {self.device}")
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            torch.set_float32_matmul_precision('high')
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Model
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
            weight_decay=config.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_iterations * config.num_epochs,
            eta_min=config.learning_rate * 0.01,
        )

        # Self-play buffer
        self.buffer = deque(maxlen=config.buffer_size)

        # Expert data
        self.expert_buffer = []

        # Stats
        self.iteration = 0
        self.total_games = 0
        self.best_winrate = 0.0
        self.history = []

        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

    def load_pretrained(self, path):
        """Load supervised_pretrained.pt as starting weights."""
        print(f"\nLoading pretrained: {path}")
        cp = torch.load(path, map_location=self.device)

        state_dict = cp.get('model_state_dict', cp)
        cleaned = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        try:
            if hasattr(self.model, '_orig_mod'):
                self.model._orig_mod.load_state_dict(cleaned)
            else:
                self.model.load_state_dict(cleaned)
            print("Pretrained weights loaded successfully")
        except Exception as e:
            print(f"Warning loading pretrained: {e}")
            if hasattr(self.model, '_orig_mod'):
                self.model._orig_mod.load_state_dict(cleaned, strict=False)
            else:
                self.model.load_state_dict(cleaned, strict=False)

    def load_checkpoint(self, path):
        """Resume from checkpoint."""
        print(f"\nResuming from: {path}")
        cp = torch.load(path, map_location=self.device)

        state_dict = cp.get('model_state_dict', cp)
        cleaned = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        try:
            if hasattr(self.model, '_orig_mod'):
                self.model._orig_mod.load_state_dict(cleaned)
            else:
                self.model.load_state_dict(cleaned)
        except Exception as e:
            print(f"Warning: {e}")
            if hasattr(self.model, '_orig_mod'):
                self.model._orig_mod.load_state_dict(cleaned, strict=False)
            else:
                self.model.load_state_dict(cleaned, strict=False)

        self.iteration = cp.get('iteration', 0)
        self.total_games = cp.get('total_games', 0)
        self.best_winrate = cp.get('best_winrate', 0.0)

        if 'optimizer_state_dict' in cp:
            try:
                self.optimizer.load_state_dict(cp['optimizer_state_dict'])
            except Exception:
                pass

        print(f"Resumed: iteration {self.iteration}, games {self.total_games}, best WR {self.best_winrate*100:.1f}%")

    def load_expert_data(self):
        if not os.path.isdir(self.config.expert_games_dir):
            print(f"Expert dir not found: {self.config.expert_games_dir}")
            return
        print(f"\nLoading expert data from {self.config.expert_games_dir}...")
        self.expert_buffer = load_expert_data(
            self.config.expert_games_dir,
            min_elo=self.config.expert_min_elo,
            max_examples=self.config.expert_max_examples,
        )
        print(f"Expert buffer: {len(self.expert_buffer)} positions")

    def self_play(self):
        self.model.eval()
        player = ConfigBSelfPlay(self.model, self.config, self.device, self.use_amp)
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

        expert_batch = int(self.config.batch_size * self.config.expert_ratio) if n_expert > 0 else 0
        selfplay_batch = self.config.batch_size - expert_batch

        total_loss, total_p_loss, total_v_loss = 0.0, 0.0, 0.0
        num_batches = 0

        num_batches_total = max(1, n_selfplay // selfplay_batch)
        selfplay_indices = np.random.permutation(n_selfplay)

        for b in range(num_batches_total):
            start = b * selfplay_batch
            end = min(start + selfplay_batch, n_selfplay)
            sp_idx = selfplay_indices[start:end]

            batch_states = [self.buffer[i]['state'] for i in sp_idx]
            batch_policies = [self.buffer[i]['policy'] for i in sp_idx]
            batch_values = [self.buffer[i]['value'] for i in sp_idx]

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

    def evaluate_vs_engine(self):
        """Evaluate model vs Gen7 alpha-beta NNUE engine."""
        self.model.eval()
        print("\n[Evaluation vs Gen7 Alpha-Beta Engine]")
        result = eval_vs_engine(self.model, self.config, self.device, self.use_amp)

        if result['win_rate'] >= 0:
            wr = result['win_rate']
            print(f"  Final: {result['wins']}W-{result['draws']}D-{result['losses']}L "
                  f"= {wr*100:.1f}% vs Gen7")
            if wr > self.best_winrate:
                self.best_winrate = wr
                self.save_checkpoint(suffix="_best")
                print(f"  NEW BEST! Winrate {wr*100:.1f}%")
        return result

    def train_iteration(self):
        self.iteration += 1
        print(f"\n{'='*60}")
        print(f"Iteration {self.iteration}/{self.config.num_iterations}  "
              f"(buffer: {len(self.buffer)}, expert: {len(self.expert_buffer)}, "
              f"games: {self.total_games})")
        print(f"{'='*60}")

        # Self-play
        print("\n[Self-Play: TrueBatchMCTS, {0} sims]".format(self.config.num_simulations))
        t0 = time.time()
        num_examples = self.self_play()
        sp_time = time.time() - t0
        print(f"Generated {num_examples} examples in {sp_time:.1f}s "
              f"({self.config.games_per_iteration / sp_time:.1f} games/s)")

        if len(self.buffer) < self.config.min_buffer_size:
            print(f"Filling buffer ({len(self.buffer)}/{self.config.min_buffer_size})")
            return

        # Training
        print("\n[Training]")
        t0 = time.time()
        for epoch in range(self.config.num_epochs):
            m = self.train_epoch()
            self.scheduler.step()
        print(f"  loss={m['loss']:.4f} (p={m['policy_loss']:.4f}, v={m['value_loss']:.4f})")
        print(f"  {time.time()-t0:.1f}s, LR: {self.optimizer.param_groups[0]['lr']:.6f}")

        # Eval vs Engine
        if self.iteration % self.config.eval_interval == 0:
            result = self.evaluate_vs_engine()

            # Early stopping check
            if self.iteration >= self.config.early_stop_iter and result['win_rate'] >= 0:
                if result['win_rate'] < self.config.early_stop_min_wr:
                    print(f"\n  EARLY STOP: winrate {result['win_rate']*100:.1f}% < "
                          f"{self.config.early_stop_min_wr*100:.0f}% after {self.iteration} iterations")
                    return False  # Signal to stop

            self.history.append({
                'iteration': self.iteration,
                'win_rate': result['win_rate'],
                'wins': result['wins'],
                'draws': result['draws'],
                'losses': result['losses'],
            })

        # Save
        if self.iteration % self.config.save_interval == 0:
            self.save_checkpoint()

        return True  # Continue training

    def save_checkpoint(self, suffix=""):
        name = f"configb_iter{self.iteration}{suffix}.pt"
        path = os.path.join(self.config.checkpoint_dir, name)
        cp = {
            'iteration': self.iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_games': self.total_games,
            'best_winrate': self.best_winrate,
            'config': asdict(self.config),
            'history': self.history,
        }
        if self.use_amp:
            cp['scaler_state_dict'] = self.scaler.state_dict()
        torch.save(cp, path)

        latest = os.path.join(self.config.checkpoint_dir, f"configb_latest{suffix}.pt")
        torch.save(cp, latest)
        print(f"  Saved: {path}")

    def train(self):
        print(f"\n{'='*60}")
        print("Config B: Supervised Pretrain + TrueBatchMCTS 800 sims")
        print(f"{'='*60}")
        print(f"Sims: {self.config.num_simulations} (TrueBatchMCTS, tree depth ~6-10)")
        print(f"c_puct: {self.config.c_puct}")
        print(f"Games/iter: {self.config.games_per_iteration}")
        print(f"Expert ratio: {self.config.expert_ratio*100:.0f}%")
        print(f"Expert buffer: {len(self.expert_buffer)} positions")
        print(f"Eval: every {self.config.eval_interval} iter vs Gen7 ({self.config.eval_games_vs_engine} games)")
        print(f"Early stop: winrate < {self.config.early_stop_min_wr*100:.0f}% after iter {self.config.early_stop_iter}")
        print(f"{'='*60}")

        remaining = self.config.num_iterations - self.iteration
        t0 = time.time()

        try:
            for _ in range(remaining):
                should_continue = self.train_iteration()
                if should_continue is False:
                    break
        except KeyboardInterrupt:
            print("\nInterrupted!")
        finally:
            total = time.time() - t0
            print(f"\nCompleted in {total/60:.1f} min, {self.total_games} games, "
                  f"{self.iteration} iterations, best WR: {self.best_winrate*100:.1f}%")
            self.save_checkpoint()


def main():
    parser = argparse.ArgumentParser(description="Config B: Supervised + TrueBatchMCTS 800")
    parser.add_argument("--model-size", default="medium")
    parser.add_argument("--games", type=int, default=100, help="Games per iteration")
    parser.add_argument("--sims", type=int, default=800, help="MCTS simulations per move")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--pretrained", type=str, default="checkpoints/supervised_pretrained.pt",
                        help="Supervised pretrained checkpoint to start from")
    parser.add_argument("--expert-dir", default="../../game-pars/games")
    parser.add_argument("--expert-ratio", type=float, default=0.3)
    parser.add_argument("--batch-games", type=int, default=8, help="Parallel self-play games")
    parser.add_argument("--c-puct", type=float, default=2.5)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-games", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=5)
    args = parser.parse_args()

    config = ConfigB(
        model_size=args.model_size,
        games_per_iteration=args.games,
        num_simulations=args.sims,
        num_iterations=args.iterations,
        expert_games_dir=args.expert_dir,
        expert_ratio=args.expert_ratio,
        batch_size_games=args.batch_games,
        c_puct=args.c_puct,
        eval_interval=args.eval_interval,
        eval_games_vs_engine=args.eval_games,
        save_interval=args.save_interval,
    )

    trainer = ConfigBTrainer(config)

    # Load weights
    if args.resume:
        trainer.load_checkpoint(args.resume)
    elif args.pretrained and os.path.exists(args.pretrained):
        trainer.load_pretrained(args.pretrained)
    else:
        print("WARNING: No pretrained weights! Starting from random init.")

    # Load expert data
    trainer.load_expert_data()

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
