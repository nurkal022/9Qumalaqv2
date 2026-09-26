#!/usr/bin/env python3
"""Test Gumbel AZ checkpoint vs NNUE Gen7 engine."""
import os
import sys
import time
import argparse
import subprocess
import numpy as np
import torch

from game import TogyzQumalaq
from model import create_model
from gumbel_az import GumbelMCTS

ENGINE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'engine')
ENGINE_PATH = os.path.join(ENGINE_DIR, 'target', 'release', 'togyzkumalaq-engine')


def start_engine():
    proc = subprocess.Popen(
        [ENGINE_PATH, 'serve'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, cwd=ENGINE_DIR,
    )
    line = proc.stdout.readline().strip()
    if line != 'ready':
        raise RuntimeError(f"Engine didn't start: {line}")
    return proc


def engine_move(proc, game, time_ms=3000):
    s = game.state
    wp = ','.join(str(int(x)) for x in s.pits[0])
    bp = ','.join(str(int(x)) for x in s.pits[1])
    k = f"{int(s.kazan[0])},{int(s.kazan[1])}"
    t = f"{int(s.tuzdyk[0])},{int(s.tuzdyk[1])}"
    side = str(int(s.current_player))
    pos = f"{wp}/{bp}/{k}/{t}/{side}"

    proc.stdin.write(f'go time {time_ms} pos {pos}\n')
    proc.stdin.flush()
    line = proc.stdout.readline().strip()

    if line.startswith('bestmove'):
        parts = line.split()
        for i, p in enumerate(parts):
            if p == 'bestmove' and i + 1 < len(parts):
                return int(parts[i + 1])
    return None


def gumbel_move(mcts, game):
    policy, _ = mcts.search_single(game, add_noise=False)
    return int(np.argmax(policy))


def play_match(model, num_games=20, mcts_sims=800, engine_time_ms=3000):
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    mcts = GumbelMCTS(model, num_simulations=mcts_sims, device=device)

    proc = start_engine()
    proc.stdin.write('newgame\n')
    proc.stdin.flush()
    proc.stdout.readline()

    mcts_wins = 0
    engine_wins = 0
    draws = 0

    for game_num in range(num_games):
        game = TogyzQumalaq()
        mcts_player = game_num % 2  # alternate sides

        # Random opening (4 plies)
        rng = np.random.RandomState(game_num * 1000)
        for _ in range(4):
            valid = game.get_valid_moves_list()
            if not valid:
                break
            action = rng.choice(valid)
            success, winner = game.make_move(action)
            if not success or winner is not None:
                break

        if game.is_terminal():
            continue

        move_count = 0
        max_moves = 200

        while not game.is_terminal() and move_count < max_moves:
            cur_player = int(game.state.current_player)
            if cur_player == mcts_player:
                action = gumbel_move(mcts, game)
            else:
                action = engine_move(proc, game, engine_time_ms)
                if action is None:
                    valid = game.get_valid_moves_list()
                    action = valid[0] if valid else 0

            success, winner = game.make_move(action)
            move_count += 1
            if not success or winner is not None:
                break

        winner = game.get_winner()
        if winner == 2:
            draws += 1
            result = 'draw'
        elif winner == mcts_player:
            mcts_wins += 1
            result = 'mcts_win'
        else:
            engine_wins += 1
            result = 'engine_win'

        side = 'White' if mcts_player == 0 else 'Black'
        print(f"  Game {game_num+1:2d} (Gumbel={side}): {result}  [{mcts_wins}W-{draws}D-{engine_wins}L]", flush=True)

    proc.stdin.write('quit\n')
    proc.stdin.flush()
    proc.terminate()

    total = mcts_wins + engine_wins + draws
    winrate = (mcts_wins + 0.5 * draws) / max(1, total)
    print(f"\nResult: {mcts_wins}W-{draws}D-{engine_wins}L = {winrate*100:.1f}% winrate")
    print(f"Games: {total}, Sims: {mcts_sims}, Engine time: {engine_time_ms}ms")
    return winrate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help='Path to checkpoint (.pt)')
    parser.add_argument('--games', type=int, default=20)
    parser.add_argument('--sims', type=int, default=800)
    parser.add_argument('--time', type=int, default=3000, help='Engine think time ms')
    parser.add_argument('--model-size', default='medium', choices=['small', 'medium', 'large'])
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Loading: {args.checkpoint}")

    model = create_model(args.model_size).to(device)
    cp = torch.load(args.checkpoint, map_location=device)
    state_dict = cp.get('model_state_dict', cp)
    cleaned = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)
    model.eval()

    iteration = cp.get('iteration', '?')
    print(f"Checkpoint iteration: {iteration}")

    play_match(model, num_games=args.games, mcts_sims=args.sims, engine_time_ms=args.time)


if __name__ == '__main__':
    main()
