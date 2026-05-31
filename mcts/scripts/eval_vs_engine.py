#!/usr/bin/env python3
"""Evaluate ONNX model (via MCTS) against the Rust alpha-beta engine."""

import sys
import os
import argparse
import subprocess
import tempfile
import shutil
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../alphazero-code/alphazero'))
from game import TogyzQumalaq, Player


class OnnxMCTS:
    """Simple MCTS using ONNX model for evaluation."""

    def __init__(self, session, num_sims=800):
        self.session = session
        self.num_sims = num_sims
        self.c_puct = 1.5

    def predict(self, game):
        state = game.encode_state().reshape(1, 7, 9).astype(np.float32)
        log_policy, value = self.session.run(None, {'state': state})
        policy = np.exp(log_policy[0])
        policy /= policy.sum() + 1e-8
        return policy, float(value[0][0])

    def search(self, game):
        """Run MCTS and return best move."""
        valid_list = game.get_valid_moves_list()
        if len(valid_list) <= 1:
            return valid_list[0] if valid_list else 0

        # Simple MCTS: expand root, run N sims (1-ply lookahead with NN value)
        root_policy, root_value = self.predict(game)
        valid = game.get_valid_moves()

        # Run simulations: for each valid move, evaluate child with NN
        visit_counts = np.zeros(9, dtype=np.float32)
        value_sums = np.zeros(9, dtype=np.float64)

        # Initialize with prior
        masked_policy = root_policy * valid
        masked_policy /= masked_policy.sum() + 1e-8

        for sim in range(self.num_sims):
            # PUCT selection
            total_visits = visit_counts.sum() + 1
            sqrt_total = np.sqrt(total_visits)

            best_action = -1
            best_score = -1e9
            for a in valid_list:
                q = value_sums[a] / (visit_counts[a] + 1e-8) if visit_counts[a] > 0 else 0
                u = self.c_puct * masked_policy[a] * sqrt_total / (1 + visit_counts[a])
                score = q + u
                if score > best_score:
                    best_score = score
                    best_action = a

            # Simulate: make move, evaluate with NN
            sim_game = TogyzQumalaq()
            sim_game.set_state(game.get_state())
            success, winner = sim_game.make_move(best_action)

            if winner is not None:
                child_value = 1.0 if winner == int(game.state.current_player) else (-1.0 if winner != 2 else 0.0)
            else:
                _, cv = self.predict(sim_game)
                child_value = -cv  # negate: child perspective → parent perspective

            visit_counts[best_action] += 1
            value_sums[best_action] += child_value

        # Pick most visited
        best = valid_list[0]
        best_visits = 0
        for a in valid_list:
            if visit_counts[a] > best_visits:
                best_visits = visit_counts[a]
                best = a
        return best


def start_engine(engine_path, weights_dir):
    """Start engine serve process."""
    proc = subprocess.Popen(
        [engine_path, "serve"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=weights_dir, text=True, bufsize=1,
    )
    # Read "ready" line
    line = proc.stdout.readline().strip()
    return proc


def get_engine_move(proc, game, time_ms):
    state = game.get_state()
    me = state.current_player
    opp = 1 - me

    w_pits = ",".join(str(int(state.pits[0][i])) for i in range(9))
    b_pits = ",".join(str(int(state.pits[1][i])) for i in range(9))
    kw, kb = int(state.kazan[0]), int(state.kazan[1])
    tw = int(state.tuzdyk[0]) if state.tuzdyk[0] >= 0 else -1
    tb = int(state.tuzdyk[1]) if state.tuzdyk[1] >= 0 else -1
    side = "w" if state.current_player == 0 else "b"
    pos = f"{w_pits}/{b_pits}/{kw},{kb}/{tw},{tb}/{side}"

    proc.stdin.write(f"go pos {pos} time {time_ms}\n")
    proc.stdin.flush()

    resp = proc.stdout.readline().strip()
    if resp.startswith("bestmove"):
        return int(resp.split()[1])
    return -1


def play_game(mcts, engine_proc, time_ms, mcts_is_white):
    game = TogyzQumalaq()
    moves = 0

    while not game.is_terminal() and moves < 300:
        current = game.state.current_player
        is_mcts_turn = (current == 0 and mcts_is_white) or (current == 1 and not mcts_is_white)

        if is_mcts_turn:
            action = mcts.search(game)
        else:
            action = get_engine_move(engine_proc, game, time_ms)
            if action < 0:
                break

        valid = game.get_valid_moves_list()
        if action not in valid:
            action = valid[0] if valid else 0

        game.make_move(action)
        moves += 1

    winner = game.get_winner()
    if winner is None or winner == 2:
        return 0.5  # draw
    mcts_side = 0 if mcts_is_white else 1
    return 1.0 if winner == mcts_side else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="ONNX model path")
    parser.add_argument("--model-size", default="medium")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--sims", type=int, default=800)
    parser.add_argument("--engine", required=True, help="Engine binary path")
    parser.add_argument("--time", type=int, default=1000, help="Engine time per move (ms)")
    args = parser.parse_args()

    # Load ONNX model
    sess = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    mcts = OnnxMCTS(sess, args.sims)

    # Setup engine working directory with weights
    engine_dir = os.path.dirname(args.engine)
    tmpdir = tempfile.mkdtemp()
    # Symlink needed files
    for f in ['nnue_weights.bin', 'egtb.bin', 'opening_book.txt']:
        src = os.path.join(engine_dir, f)
        if os.path.exists(src):
            os.symlink(src, os.path.join(tmpdir, f))

    proc = start_engine(args.engine, tmpdir)

    wins, draws, losses = 0, 0, 0
    for g in range(args.games):
        mcts_is_white = (g % 2 == 0)
        result = play_game(mcts, proc, args.time, mcts_is_white)
        if result == 1.0:
            wins += 1
        elif result == 0.5:
            draws += 1
        else:
            losses += 1
        total = wins + draws + losses
        wr = (wins + 0.5 * draws) / total * 100
        print(f"  [{total}/{args.games}] MCTS: {wins}W-{draws}D-{losses}L ({wr:.1f}%)", flush=True)

    proc.terminate()
    shutil.rmtree(tmpdir, ignore_errors=True)

    total = wins + draws + losses
    wr = (wins + 0.5 * draws) / total * 100
    print(f"Final: {wins}W-{draws}D-{losses}L = {wr:.1f}% vs Gen7")


if __name__ == "__main__":
    main()
