#!/usr/bin/env python3
"""
Generate NNUE training data using the expert-trained NN + MCTS.
Plays self-play games with MCTS, saves positions in NNUE binary format.
This distills knowledge from the 1M-param NN into data the 18K-param NNUE can learn from.

Output format: 26 bytes per record (same as engine datagen):
  [0-8]   white pits (u8)
  [9-17]  black pits (u8)
  [18]    white kazan (u8)
  [19]    black kazan (u8)
  [20]    white tuzdyk (i8 as u8)
  [21]    black tuzdyk (i8 as u8)
  [22]    side to move (u8)
  [23-24] eval (i16 LE) - MCTS value scaled to centipawns
  [25]    result (u8): 0=black win, 1=draw, 2=white win
"""
import os
import sys
import struct
import time
import argparse
import numpy as np
import torch

from game import TogyzQumalaq, Player
from model import create_model
from train_fast import TrueBatchMCTS


def value_to_cp(value, k=400):
    """Convert NN value [-1,1] to centipawns using sigmoid inverse."""
    value = max(-0.999, min(0.999, value))
    # sigmoid(cp/k) = (value+1)/2  =>  cp = k * ln((1+v)/(1-v)) / 2
    cp = k * np.log((1 + value) / (1 - value)) / 2
    return int(np.clip(cp, -32000, 32000))


def board_to_record(game, eval_cp, result_byte):
    """Convert game state to 26-byte NNUE record."""
    state = game.state
    record = bytearray(26)

    for i in range(9):
        record[i] = min(255, int(state.pits[0][i]))
    for i in range(9):
        record[9 + i] = min(255, int(state.pits[1][i]))

    record[18] = min(255, int(state.kazan[0]))
    record[19] = min(255, int(state.kazan[1]))

    t0 = int(state.tuzdyk[0])
    t1 = int(state.tuzdyk[1])
    record[20] = t0 & 0xFF
    record[21] = t1 & 0xFF

    record[22] = int(state.current_player)
    struct.pack_into('<h', record, 23, eval_cp)
    record[25] = result_byte

    return bytes(record)


def play_game_mcts(mcts, game_idx, sims_per_move=200, temp_threshold=15):
    """Play one game using MCTS, return list of (game_state_snapshot, mcts_value)."""
    game = TogyzQumalaq()
    positions = []
    move_count = 0

    while not game.is_terminal() and move_count < 300:
        # Get MCTS policy and value
        policies = mcts.search_batch([game])
        policy = policies[0]

        # Get value estimate from root node (approximation: use model directly)
        state_enc = game.encode_state()
        state_tensor = torch.FloatTensor(state_enc).unsqueeze(0)
        if next(mcts.model.parameters()).is_cuda:
            state_tensor = state_tensor.cuda()
        with torch.no_grad():
            _, value = mcts.model(state_tensor)
        value = value.item()

        # Record position (skip first 4 plies)
        if move_count >= 4:
            positions.append({
                'game': TogyzQumalaq(),  # will copy state below
                'state': game.get_state().copy(),
                'value': value,
                'player': int(game.state.current_player),
            })

        # Select move
        if move_count < temp_threshold:
            # Temperature-based selection
            action = int(np.random.choice(9, p=policy))
        else:
            action = int(np.argmax(policy))

        valid = game.get_valid_moves_list()
        if action not in valid:
            if valid:
                action = valid[0]
            else:
                break

        game.make_move(action)
        move_count += 1

    # Get game result
    winner = game.get_winner()
    if winner is None or winner == 2:
        result_byte = 1  # draw
    elif int(winner) == 0:
        result_byte = 2  # white win
    else:
        result_byte = 0  # black win

    # Convert to records
    records = []
    for pos in positions:
        # Value is from current player's perspective
        # Convert to side-to-move perspective eval
        v = pos['value']
        eval_cp = value_to_cp(v)

        # Create temporary game to get the board state
        tmp = TogyzQumalaq()
        tmp.set_state(pos['state'])
        records.append(board_to_record(tmp, eval_cp, result_byte))

    return records, move_count, winner


def play_batch_mcts(mcts, batch_size=16, sims_per_move=200, temp_threshold=15):
    """Play a batch of games in parallel using batch MCTS."""
    games = [TogyzQumalaq() for _ in range(batch_size)]
    all_positions = [[] for _ in range(batch_size)]
    active = list(range(batch_size))
    move_counts = [0] * batch_size
    max_moves = 300

    while active:
        active_games = [games[i] for i in active]

        # Batch MCTS
        policies = mcts.search_batch(active_games)

        # Get batch values
        states_enc = np.array([games[i].encode_state() for i in active])
        states_tensor = torch.FloatTensor(states_enc)
        if next(mcts.model.parameters()).is_cuda:
            states_tensor = states_tensor.cuda()
        with torch.no_grad():
            _, values = mcts.model(states_tensor)
        values = values.cpu().numpy()[:, 0]

        new_active = []
        for j, (idx, policy) in enumerate(zip(active, policies)):
            game = games[idx]
            mc = move_counts[idx]

            if game.is_terminal() or mc >= max_moves:
                continue

            # Record position (skip first 4 plies)
            if mc >= 4:
                all_positions[idx].append({
                    'state': game.get_state().copy(),
                    'value': float(values[j]),
                    'player': int(game.state.current_player),
                })

            # Select move
            if mc < temp_threshold:
                action = int(np.random.choice(9, p=policy))
            else:
                action = int(np.argmax(policy))

            valid = game.get_valid_moves_list()
            if action not in valid:
                if valid:
                    action = valid[0]
                else:
                    continue

            game.make_move(action)
            move_counts[idx] += 1

            if not game.is_terminal() and move_counts[idx] < max_moves:
                new_active.append(idx)

        active = new_active

    # Convert to records
    all_records = []
    for idx in range(batch_size):
        game = games[idx]
        winner = game.get_winner()
        if winner is None or winner == 2:
            result_byte = 1
        elif int(winner) == 0:
            result_byte = 2
        else:
            result_byte = 0

        for pos in all_positions[idx]:
            v = pos['value']
            eval_cp = value_to_cp(v)
            tmp = TogyzQumalaq()
            tmp.set_state(pos['state'])
            all_records.append(board_to_record(tmp, eval_cp, result_byte))

    return all_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--model-size', default='medium')
    parser.add_argument('--output', required=True, help='Output .bin file')
    parser.add_argument('--games', type=int, default=2000)
    parser.add_argument('--sims', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_model(args.model_size, device)

    cp = torch.load(args.checkpoint, map_location=device)
    state_dict = cp.get('model_state_dict', cp)
    cleaned = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned)
    model.eval()

    print(f"Model: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Games: {args.games}, Sims: {args.sims}, Batch: {args.batch_size}")

    mcts = TrueBatchMCTS(model, num_simulations=args.sims, device=device,
                         use_amp=(device == 'cuda'))

    all_records = []
    num_batches = (args.games + args.batch_size - 1) // args.batch_size
    t0 = time.time()
    save_interval = max(1, 500 // args.batch_size)  # save every ~500 games

    # Open output file for incremental writing
    out_f = open(args.output, 'wb')

    for batch_idx in range(num_batches):
        batch_games = min(args.batch_size, args.games - batch_idx * args.batch_size)
        records = play_batch_mcts(mcts, batch_size=batch_games, sims_per_move=args.sims)
        all_records.extend(records)

        elapsed = time.time() - t0
        games_done = (batch_idx + 1) * args.batch_size
        games_done = min(games_done, args.games)
        rate = games_done / elapsed if elapsed > 0 else 0
        print(f"  Batch {batch_idx+1}/{num_batches}: {len(all_records)} positions, "
              f"{games_done} games ({rate:.1f} games/s, {elapsed:.0f}s)")

        # Incremental save every save_interval batches
        if (batch_idx + 1) % save_interval == 0 or batch_idx == num_batches - 1:
            import random
            import io
            tmp = list(all_records)
            random.shuffle(tmp)
            out_f.seek(0)
            out_f.truncate()
            for rec in tmp:
                out_f.write(rec)
            out_f.flush()

    out_f.close()

    elapsed = time.time() - t0
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"\nSaved: {args.output} ({size_mb:.1f} MB, {len(all_records)} positions)")
    print(f"Total time: {elapsed:.0f}s ({args.games / elapsed:.1f} games/s)")


if __name__ == '__main__':
    main()
