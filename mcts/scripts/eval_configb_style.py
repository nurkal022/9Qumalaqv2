#!/usr/bin/env python3
"""
Evaluate PyTorch model vs Gen7 engine using ConfigurableMCTS (proper tree search).
Same approach as train_config_b.py eval_vs_engine — proven to work (55-67% vs Gen7).
"""
import sys, os, argparse, subprocess, tempfile, shutil
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../alphazero-code/alphazero'))
from game import TogyzQumalaq, Player
from model import create_model
from train_config_b import ConfigurableMCTS, get_engine_move


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-size", default="medium")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--sims", type=int, default=800)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--time", type=int, default=1000)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    model = create_model(args.model_size, device)
    cp = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = cp.get('model_state_dict', cp)
    cleaned = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    model.load_state_dict(cleaned, strict=False)
    model.eval()

    # MCTS (no noise for eval)
    mcts = ConfigurableMCTS(
        model, num_simulations=args.sims, c_puct=2.5,
        dirichlet_alpha=0.0, dirichlet_eps=0.0,
        device=device, use_amp=(device == 'cuda'),
    )

    # Setup engine
    engine_dir = os.path.dirname(args.engine)
    tmpdir = tempfile.mkdtemp()
    for f in ['nnue_weights.bin', 'egtb.bin', 'opening_book.txt']:
        src = os.path.join(engine_dir, f)
        if os.path.exists(src):
            os.symlink(src, os.path.join(tmpdir, f))

    proc = subprocess.Popen(
        [args.engine, "serve"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=tmpdir, text=True, bufsize=1,
    )
    while True:
        line = proc.stdout.readline().strip()
        if line == "ready" or not line:
            break

    wins, draws, losses = 0, 0, 0
    for g in range(args.games):
        mcts_is_white = (g % 2 == 0)
        game = TogyzQumalaq()
        moves = 0

        while not game.is_terminal() and moves < 200:
            cp_val = int(game.state.current_player)
            mcts_turn = (cp_val == 0 and mcts_is_white) or (cp_val == 1 and not mcts_is_white)

            if mcts_turn:
                policy = mcts.search_batch([game])[0]
                action = int(np.argmax(policy))
            else:
                action = get_engine_move(proc, game, args.time)
                if action < 0:
                    break

            valid = game.get_valid_moves_list()
            if action not in valid:
                action = valid[0] if valid else 0
            game.make_move(action)
            moves += 1

        winner = game.get_winner()
        mcts_player = 0 if mcts_is_white else 1
        if winner == 2 or winner is None:
            draws += 1
        elif winner == mcts_player:
            wins += 1
        else:
            losses += 1

        total = g + 1
        wr = (wins + 0.5 * draws) / total * 100
        print(f"  [{total}/{args.games}] MCTS: {wins}W-{draws}D-{losses}L ({wr:.1f}%)", flush=True)

    proc.terminate()
    shutil.rmtree(tmpdir, ignore_errors=True)

    total = wins + draws + losses
    wr = (wins + 0.5 * draws) / total * 100
    print(f"Final: {wins}W-{draws}D-{losses}L = {wr:.1f}% vs Gen7")


if __name__ == "__main__":
    main()
