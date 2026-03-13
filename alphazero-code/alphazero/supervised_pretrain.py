#!/usr/bin/env python3
"""
Supervised pre-training for AlphaZero policy+value network.
Uses PlayOK expert games (184K files) to bootstrap the network.

This gives a MASSIVE head start over training from scratch via self-play.
Original AlphaGo used this approach before self-play refinement.

Usage:
    python supervised_pretrain.py [--games-dir PATH] [--min-elo 1500] [--epochs 50] [--model-size medium]
"""
import os
import sys
import re
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from game import TogyzQumalaq, Player
from model import create_model


class PlayOKDataset(Dataset):
    """Dataset of (encoded_state, move, result) from PlayOK expert games."""

    def __init__(self, examples):
        self.states = torch.FloatTensor(np.array([e[0] for e in examples]))
        self.moves = torch.LongTensor(np.array([e[1] for e in examples]))
        self.values = torch.FloatTensor(np.array([e[2] for e in examples]))

    def __len__(self):
        return len(self.moves)

    def __getitem__(self, idx):
        return self.states[idx], self.moves[idx], self.values[idx]


def parse_pgn(filepath):
    """Parse PlayOK PGN file."""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    headers = {}
    for m in re.finditer(r'\[(\w+)\s+"(.*?)"\]', text):
        headers[m.group(1)] = m.group(2)
    last_bracket = text.rfind(']')
    if last_bracket < 0:
        return None, None
    move_text = text[last_bracket + 1:].strip()
    return headers, move_text


def extract_moves(move_text):
    """Extract 0-indexed pit moves from PlayOK move notation."""
    move_text = re.sub(r'(1-0|0-1|1/2-1/2)\s*(\{.*?\})?\s*$', '', move_text)
    move_text = re.sub(r'\d+\.', '', move_text)
    tokens = move_text.split()
    moves = []
    for token in tokens:
        m = re.match(r'^(\d)(\d)(?:\(\d+\))?X?$', token)
        if m:
            pit = int(m.group(1))
            if 1 <= pit <= 9:
                moves.append(pit - 1)
    return moves


def process_games(games_dir, min_elo=1500, max_games=None):
    """Process all PlayOK game files, extract training examples."""
    files = [os.path.join(games_dir, f) for f in os.listdir(games_dir) if f.endswith('.txt')]
    random.shuffle(files)
    if max_games:
        files = files[:max_games * 2]  # oversample since some games filtered

    examples = []
    valid_games = 0
    skipped = 0
    t0 = time.time()

    for i, filepath in enumerate(files):
        try:
            headers, move_text = parse_pgn(filepath)
            if headers is None:
                continue

            # ELO filter
            w_elo = int(headers.get('WhiteElo', '0'))
            b_elo = int(headers.get('BlackElo', '0'))
            if w_elo < min_elo or b_elo < min_elo:
                continue

            # Result
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

            # Replay game, extract examples
            game = TogyzQumalaq()
            ply = 0

            for pit in moves:
                valid_moves = game.get_valid_moves_list()
                if pit not in valid_moves:
                    break

                # Record: (encoded_state, move_played, value_from_current_player)
                if ply >= 2:  # skip first 2 plies (too deterministic)
                    state_encoded = game.encode_state()  # (7, 9) from current player's perspective
                    current_player = game.state.current_player
                    # Value from current player's perspective
                    if current_player == Player.WHITE:
                        value = white_value
                    else:
                        value = -white_value

                    examples.append((state_encoded, pit, value))

                success, winner = game.make_move(pit)
                ply += 1
                if not success or winner is not None:
                    break

            valid_games += 1

        except Exception:
            skipped += 1

        if (i + 1) % 10000 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(files)} files | {valid_games} games | {len(examples)} positions | {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"\nTotal: {valid_games} games, {len(examples)} positions ({elapsed:.0f}s)")
    print(f"Skipped: {skipped} files with errors")

    return examples


def train_supervised(model, train_examples, val_examples, device, epochs=50, batch_size=2048, lr=0.001):
    """Train policy+value network with supervised learning."""

    train_dataset = PlayOKDataset(train_examples)
    val_dataset = PlayOKDataset(val_examples)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None

    print(f"\nTraining: {len(train_examples)} train, {len(val_examples)} val")
    print(f"Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")
    print(f"Device: {device}\n")

    for epoch in range(epochs):
        # Train
        model.train()
        train_policy_loss = 0
        train_value_loss = 0
        train_correct = 0
        train_total = 0

        for states, moves, values in train_loader:
            states = states.to(device, non_blocking=True)
            moves = moves.to(device, non_blocking=True)
            values = values.to(device, non_blocking=True).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)

            log_policy, value_pred = model(states)
            p_loss = policy_criterion(log_policy, moves)
            v_loss = value_criterion(value_pred, values)
            loss = p_loss + v_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_policy_loss += p_loss.item() * states.size(0)
            train_value_loss += v_loss.item() * states.size(0)
            preds = log_policy.argmax(dim=1)
            train_correct += (preds == moves).sum().item()
            train_total += states.size(0)

        scheduler.step()

        # Validate
        model.eval()
        val_policy_loss = 0
        val_value_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for states, moves, values in val_loader:
                states = states.to(device, non_blocking=True)
                moves = moves.to(device, non_blocking=True)
                values = values.to(device, non_blocking=True).unsqueeze(1)

                log_policy, value_pred = model(states)
                p_loss = policy_criterion(log_policy, moves)
                v_loss = value_criterion(value_pred, values)

                val_policy_loss += p_loss.item() * states.size(0)
                val_value_loss += v_loss.item() * states.size(0)
                preds = log_policy.argmax(dim=1)
                val_correct += (preds == moves).sum().item()
                val_total += states.size(0)

        train_p = train_policy_loss / train_total
        train_v = train_value_loss / train_total
        train_acc = train_correct / train_total * 100
        val_p = val_policy_loss / val_total
        val_v = val_value_loss / val_total
        val_acc = val_correct / val_total * 100
        val_total_loss = val_p + val_v

        improved = val_total_loss < best_val_loss
        if improved:
            best_val_loss = val_total_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        lr_now = scheduler.get_last_lr()[0]
        marker = " *" if improved else ""
        print(f"Epoch {epoch+1:3d}/{epochs}  "
              f"train: p={train_p:.4f} v={train_v:.4f} acc={train_acc:.1f}%  "
              f"val: p={val_p:.4f} v={val_v:.4f} acc={val_acc:.1f}%  "
              f"lr={lr_now:.6f}{marker}")

    # Load best model
    if best_state:
        model.load_state_dict(best_state)

    return model, best_val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--games-dir', default='../../game-pars/games', help='PlayOK game files directory')
    parser.add_argument('--min-elo', type=int, default=1400, help='Minimum ELO for both players')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=2048, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.002, help='Learning rate')
    parser.add_argument('--model-size', default='medium', choices=['small', 'medium', 'large'])
    parser.add_argument('--output', default='checkpoints/supervised_pretrained.pt')
    parser.add_argument('--val-split', type=float, default=0.05, help='Validation split ratio')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Load games
    print(f"\nLoading PlayOK games from {args.games_dir}")
    print(f"Min ELO: {args.min_elo}")

    examples = process_games(args.games_dir, min_elo=args.min_elo)

    if len(examples) < 1000:
        print("Not enough examples! Check games directory and ELO threshold.")
        sys.exit(1)

    # Shuffle and split
    random.shuffle(examples)
    val_size = int(len(examples) * args.val_split)
    val_examples = examples[:val_size]
    train_examples = examples[val_size:]

    print(f"\nTrain: {len(train_examples)}, Val: {len(val_examples)}")

    # Create model
    model = create_model(args.model_size, str(device))
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model_size} ({total_params:,} params)")

    # Train
    model, best_val_loss = train_supervised(
        model, train_examples, val_examples, device,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr
    )

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_size': args.model_size,
        'best_val_loss': best_val_loss,
        'num_train_examples': len(train_examples),
        'min_elo': args.min_elo,
        'training_type': 'supervised_pretrain',
    }
    torch.save(checkpoint, args.output)
    print(f"\nSaved: {args.output} (val_loss={best_val_loss:.4f})")

    # Quick policy accuracy test
    model.eval()
    game = TogyzQumalaq()
    state = torch.FloatTensor(game.encode_state()).unsqueeze(0).to(device)
    with torch.no_grad():
        log_policy, value = model(state)
    policy = torch.exp(log_policy).cpu().numpy()[0]
    value = value.item()
    print(f"\nInitial position test:")
    print(f"  Policy: {['%.1f%%' % (p*100) for p in policy]}")
    print(f"  Value: {value:.4f}")
    print(f"  Best move: pit {policy.argmax()+1}")


if __name__ == '__main__':
    main()
