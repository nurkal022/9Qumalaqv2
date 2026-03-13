#!/usr/bin/env python3
"""
Test MCTS+NN model against the Rust NNUE alpha-beta engine.
Communicates with the engine via its 'serve' protocol.
"""
import os
import sys
import time
import argparse
import subprocess
import numpy as np
import torch

from game import TogyzQumalaq, Player
from model import create_model
from train_fast import TrueBatchMCTS

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
    """Get engine's best move for current position."""
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


def mcts_move(mcts, game):
    """Get MCTS best move."""
    policies = mcts.search_batch([game])
    return int(np.argmax(policies[0]))


def play_match(model, num_games=100, mcts_sims=400, engine_time_ms=3000):
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    mcts = TrueBatchMCTS(model, num_simulations=mcts_sims, device=device, use_amp=(device == 'cuda'))

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

        # Random opening (4 plies) for diversity
        rng = np.random.RandomState(game_num * 1000)
        for _ in range(4):
            if game.is_terminal():
                break
            moves = game.get_valid_moves_list()
            if not moves:
                break
            game.make_move(rng.choice(moves))

        move_count = 0
        while not game.is_terminal() and move_count < 300:
            if int(game.state.current_player) == mcts_player:
                action = mcts_move(mcts, game)
            else:
                action = engine_move(proc, game, time_ms=engine_time_ms)
                if action is None:
                    break

            valid = game.get_valid_moves_list()
            if action not in valid:
                if valid:
                    action = valid[0]
                else:
                    break

            game.make_move(action)
            move_count += 1

        winner = game.get_winner()
        if winner is None or winner == 2:
            draws += 1
            result = "Draw"
        elif int(winner) == mcts_player:
            mcts_wins += 1
            result = "MCTS wins"
        else:
            engine_wins += 1
            result = "Engine wins"

        total = game_num + 1
        print(f"Game {total}: {result} (moves={move_count}, mcts_side={'W' if mcts_player == 0 else 'B'}) "
              f"| MCTS {mcts_wins}-{draws}-{engine_wins} Engine "
              f"({mcts_wins/total*100:.0f}%-{engine_wins/total*100:.0f}%)")

        # Reset engine state between games
        proc.stdin.write('newgame\n')
        proc.stdin.flush()
        proc.stdout.readline()

    proc.stdin.write('quit\n')
    proc.stdin.flush()
    try:
        proc.wait(timeout=5)
    except:
        proc.kill()

    total = num_games
    print(f"\n{'='*50}")
    print(f"Final: MCTS {mcts_wins}-{draws}-{engine_wins} Engine")
    print(f"MCTS win rate: {(mcts_wins + draws * 0.5) / total * 100:.1f}%")
    print(f"{'='*50}")

    return mcts_wins, draws, engine_wins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint path')
    parser.add_argument('--model-size', default='medium')
    parser.add_argument('--games', type=int, default=50)
    parser.add_argument('--sims', type=int, default=400, help='MCTS simulations per move')
    parser.add_argument('--engine-time', type=int, default=3000, help='Engine time per move (ms)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_model(args.model_size, device)

    # Load checkpoint
    cp = torch.load(args.checkpoint, map_location=device)
    state_dict = cp.get('model_state_dict', cp)
    # Strip _orig_mod prefix if present
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.replace('_orig_mod.', '')] = v
    model.load_state_dict(cleaned)
    model.eval()

    print(f"Model: {args.checkpoint}")
    print(f"MCTS sims: {args.sims}, Engine time: {args.engine_time}ms")
    print(f"Games: {args.games}")
    print()

    play_match(model, num_games=args.games, mcts_sims=args.sims, engine_time_ms=args.engine_time)


if __name__ == '__main__':
    main()
