#!/usr/bin/env python3
"""
AlphaZero training loop for Rust MCTS self-play data.

Reads binary replay buffer from Rust, trains PyTorch model, exports ONNX.

Usage:
  python train_alphazero.py --replay replay_buffer.bin --checkpoint model.pt --output model.onnx
"""

import sys
import os
import argparse
import struct
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../alphazero-code/alphazero'))
from model import create_model
from game import TogyzQumalaq, Player

# ── Replay buffer reading ────────────────────────────────

RECORD_SIZE = 63  # 23 bytes board + 36 bytes policy + 4 bytes value

def load_replay_buffer(path: str):
    """Read binary replay buffer produced by Rust MCTS."""
    data = np.fromfile(path, dtype=np.uint8)
    n = len(data) // RECORD_SIZE
    if n == 0:
        return np.zeros((0, 7, 9), dtype=np.float32), np.zeros((0, 9), dtype=np.float32), np.zeros(0, dtype=np.float32)

    records = data[:n * RECORD_SIZE].reshape(n, RECORD_SIZE)

    # Parse board states (first 23 bytes)
    boards = records[:, :23]
    # Parse policy (bytes 23-58, 9 x f32 LE)
    policy_bytes = records[:, 23:59].copy()  # ensure contiguous
    policies = np.frombuffer(policy_bytes.tobytes(), dtype='<f4').reshape(n, 9)
    # Parse value (bytes 59-62, f32 LE)
    value_bytes = records[:, 59:63].copy()
    values = np.frombuffer(value_bytes.tobytes(), dtype='<f4').reshape(n)

    # Encode board states to neural network input format [n, 7, 9]
    states = np.zeros((n, 7, 9), dtype=np.float32)
    PIT_NORM = 50.0
    KAZAN_NORM = 82.0

    for i in range(n):
        b = boards[i]
        pits0 = b[0:9].astype(np.float32)   # White pits
        pits1 = b[9:18].astype(np.float32)   # Black pits
        kazan0 = float(b[18])
        kazan1 = float(b[19])
        tuzdyk0 = np.int8(b[20])  # can be -1 (0xFF as u8 → -1 as i8)
        tuzdyk1 = np.int8(b[21])
        side = int(b[22])  # 0=White, 1=Black

        if side == 0:  # White to move
            me_pits, opp_pits = pits0, pits1
            me_kazan, opp_kazan = kazan0, kazan1
            me_tuzdyk, opp_tuzdyk = tuzdyk0, tuzdyk1
        else:  # Black to move
            me_pits, opp_pits = pits1, pits0
            me_kazan, opp_kazan = kazan1, kazan0
            me_tuzdyk, opp_tuzdyk = tuzdyk1, tuzdyk0

        states[i, 0] = me_pits / PIT_NORM
        states[i, 1] = opp_pits / PIT_NORM
        states[i, 2] = me_kazan / KAZAN_NORM
        states[i, 3] = opp_kazan / KAZAN_NORM
        if me_tuzdyk >= 0:
            states[i, 4, me_tuzdyk] = 1.0
        if opp_tuzdyk >= 0:
            states[i, 5, opp_tuzdyk] = 1.0
        states[i, 6] = 1.0 if side == 0 else 0.0

    return states, policies, values


def load_expert_data(games_dir, min_elo=1400, max_examples=200000):
    """Load PlayOK expert data for supervised replay."""
    from supervised_pretrain import parse_pgn, extract_moves

    files = [os.path.join(games_dir, f) for f in os.listdir(games_dir) if f.endswith('.txt')]
    np.random.shuffle(files)

    examples_s, examples_p, examples_v = [], [], []
    valid_games = 0

    for filepath in files:
        if len(examples_s) >= max_examples:
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
            if result_str == '1-0': white_value = 1.0
            elif result_str == '0-1': white_value = -1.0
            elif result_str == '1/2-1/2': white_value = 0.0
            else: continue

            moves = extract_moves(move_text)
            if len(moves) < 10:
                continue

            game = TogyzQumalaq()
            for ply, pit in enumerate(moves):
                valid_moves = game.get_valid_moves_list()
                if pit not in valid_moves:
                    break
                if ply >= 2:
                    state = game.encode_state()
                    cp = game.state.current_player
                    value = white_value if cp == Player.WHITE else -white_value
                    policy = np.zeros(9, dtype=np.float32)
                    policy[pit] = 1.0

                    examples_s.append(state)
                    examples_p.append(policy)
                    examples_v.append(value)

                success, winner = game.make_move(pit)
                if not success or winner is not None:
                    break
            valid_games += 1
        except Exception:
            pass

    if examples_s:
        return np.array(examples_s), np.array(examples_p), np.array(examples_v)
    return None, None, None


# ── Training ─────────────────────────────────────────────

def train(model, optimizer, states, policies, values,
          expert_data=None, expert_ratio=0.3,
          epochs=10, batch_size=512, device='cuda'):
    """Train model on replay buffer data + optional expert data."""
    model.train()
    n = len(states)

    for epoch in range(epochs):
        indices = np.random.permutation(n)
        total_loss, total_p, total_v, num_batches = 0, 0, 0, 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]
            actual_batch = end - start

            s = torch.FloatTensor(states[idx]).to(device)
            p_target = torch.FloatTensor(policies[idx]).to(device)
            v_target = torch.FloatTensor(values[idx]).unsqueeze(1).to(device)

            # Mix in expert data
            if expert_data is not None:
                ex_s, ex_p, ex_v = expert_data
                n_expert = int(actual_batch * expert_ratio)
                if n_expert > 0 and len(ex_s) > 0:
                    ex_idx = np.random.choice(len(ex_s), min(n_expert, len(ex_s)), replace=False)
                    s = torch.cat([s, torch.FloatTensor(ex_s[ex_idx]).to(device)])
                    p_target = torch.cat([p_target, torch.FloatTensor(ex_p[ex_idx]).to(device)])
                    v_target = torch.cat([v_target, torch.FloatTensor(ex_v[ex_idx]).unsqueeze(1).to(device)])

            log_p, v = model(s)
            p_loss = -torch.mean(torch.sum(p_target * log_p, dim=1))
            v_loss = F.mse_loss(v, v_target)
            loss = p_loss + v_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_p += p_loss.item()
            total_v += v_loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        avg_p = total_p / max(1, num_batches)
        avg_v = total_v / max(1, num_batches)
        print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} (p={avg_p:.4f}, v={avg_v:.4f})")

    return avg_loss


# ── ONNX Export ──────────────────────────────────────────

def export_onnx(model, output_path):
    """Export model to ONNX."""
    model.eval()
    model_cpu = model.cpu()
    dummy = torch.randn(1, 7, 9)

    torch.onnx.export(
        model_cpu, dummy, output_path,
        export_params=True,
        opset_version=17,
        input_names=['state'],
        output_names=['log_policy', 'value'],
        dynamic_axes={
            'state': {0: 'batch'},
            'log_policy': {0: 'batch'},
            'value': {0: 'batch'},
        },
    )
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Exported ONNX: {output_path}")


# ── Main ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", required=True, help="Replay buffer .bin from Rust")
    parser.add_argument("--checkpoint", default=None, help="Resume from checkpoint")
    parser.add_argument("--output-onnx", default="model.onnx", help="Output ONNX model")
    parser.add_argument("--output-pt", default="model.pt", help="Output PyTorch checkpoint")
    parser.add_argument("--model-size", default="medium")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--expert-dir", default="../../game-pars/games")
    parser.add_argument("--expert-ratio", type=float, default=0.3)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load replay buffer
    print(f"Loading replay buffer: {args.replay}")
    states, policies, values = load_replay_buffer(args.replay)
    print(f"  {len(states)} positions loaded")

    # Load expert data
    expert_data = None
    if os.path.isdir(args.expert_dir):
        print(f"Loading expert data from {args.expert_dir}...")
        expert_data = load_expert_data(args.expert_dir)
        if expert_data[0] is not None:
            print(f"  {len(expert_data[0])} expert positions")

    # Create/load model
    model = create_model(args.model_size, device=device)
    if args.checkpoint and os.path.exists(args.checkpoint):
        cp = torch.load(args.checkpoint, map_location=device, weights_only=False)
        sd = cp.get('model_state_dict', cp)
        cleaned = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        model.load_state_dict(cleaned, strict=False)
        print(f"Loaded checkpoint: {args.checkpoint}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Train
    print(f"\nTraining ({args.epochs} epochs, batch {args.batch_size})...")
    t0 = time.time()
    train(model, optimizer, states, policies, values,
          expert_data=expert_data, expert_ratio=args.expert_ratio,
          epochs=args.epochs, batch_size=args.batch_size, device=device)
    print(f"Training done in {time.time()-t0:.1f}s")

    # Save checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, args.output_pt)
    print(f"Saved: {args.output_pt}")

    # Export ONNX
    export_onnx(model, args.output_onnx)


if __name__ == "__main__":
    main()
