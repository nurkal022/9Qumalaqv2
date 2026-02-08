"""
NNUE Training for Togyz Kumalak

Reads binary training data from datagen, trains a small neural network,
and exports weights for Rust inference.

Architecture: Input(40) → Linear(256) → ClippedReLU → Linear(32) → ClippedReLU → Linear(1)

Input features (40):
  - 9 current player pit values (/ 50.0)
  - 9 opponent pit values (/ 50.0)
  - 1 current player kazan (/ 82.0)
  - 1 opponent kazan (/ 82.0)
  - 10 current player tuzdyk one-hot (-1=none, 0-8=position)
  - 10 opponent tuzdyk one-hot
"""

import struct
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import json


RECORD_SIZE = 26
INPUT_SIZE = 40
K = 1050.0  # Sigmoid scaling from Texel tuning


class PositionDataset(Dataset):
    def __init__(self, path, max_positions=None):
        file_size = os.path.getsize(path)
        num_records = file_size // RECORD_SIZE
        if max_positions:
            num_records = min(num_records, max_positions)

        print(f"Loading {num_records:,} positions from {path}...")

        with open(path, 'rb') as f:
            raw = f.read(num_records * RECORD_SIZE)

        data = np.frombuffer(raw, dtype=np.uint8).reshape(num_records, RECORD_SIZE)

        # Parse fields
        pits_w = data[:, 0:9].astype(np.float32)
        pits_b = data[:, 9:18].astype(np.float32)
        kazan_w = data[:, 18:19].astype(np.float32)
        kazan_b = data[:, 19:20].astype(np.float32)
        tuzdyk_w = data[:, 20].astype(np.int8)
        tuzdyk_b = data[:, 21].astype(np.int8)
        side = data[:, 22]

        # Eval: i16 little-endian
        eval_bytes = data[:, 23:25].copy()
        evals = eval_bytes.view(np.int16).astype(np.float32).flatten()

        # Result: 0=white_loss, 1=draw, 2=white_win
        results = data[:, 25].astype(np.float32) / 2.0  # -> 0.0, 0.5, 1.0

        # Build features from side-to-move perspective
        is_white = (side == 0)
        is_black = ~is_white

        # Pits: current player first, opponent second
        my_pits = np.where(is_white[:, None], pits_w, pits_b) / 50.0
        opp_pits = np.where(is_white[:, None], pits_b, pits_w) / 50.0

        # Kazan
        my_kazan = np.where(is_white[:, None], kazan_w, kazan_b) / 82.0
        opp_kazan = np.where(is_white[:, None], kazan_b, kazan_w) / 82.0

        # Tuzdyk one-hot (10 dims each: index -1 maps to position 9, 0-8 map to 0-8)
        my_tuz_idx = np.where(is_white, tuzdyk_w, tuzdyk_b).astype(np.int64)
        opp_tuz_idx = np.where(is_white, tuzdyk_b, tuzdyk_w).astype(np.int64)

        my_tuz = np.zeros((num_records, 10), dtype=np.float32)
        opp_tuz = np.zeros((num_records, 10), dtype=np.float32)

        for i in range(num_records):
            idx = my_tuz_idx[i]
            my_tuz[i, idx if idx >= 0 else 9] = 1.0
            idx = opp_tuz_idx[i]
            opp_tuz[i, idx if idx >= 0 else 9] = 1.0

        # Concatenate all features
        self.features = np.concatenate([
            my_pits, opp_pits, my_kazan, opp_kazan, my_tuz, opp_tuz
        ], axis=1)

        # Target: eval from side-to-move perspective (already correct from datagen)
        # Convert to win probability via sigmoid for training
        self.eval_targets = evals

        # Result from side-to-move perspective
        result_stm = np.where(is_white, results, 1.0 - results)
        self.result_targets = result_stm

        # Combined target: lambda * sigmoid(eval/K) + (1-lambda) * result
        # We'll compute this in training loop with configurable lambda

        print(f"  Features shape: {self.features.shape}")
        print(f"  Eval range: [{evals.min():.0f}, {evals.max():.0f}]")
        print(f"  Results: W={np.sum(results > 0.6)}, D={np.sum(np.abs(results - 0.5) < 0.1)}, L={np.sum(results < 0.4)}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx]),
            torch.tensor(self.eval_targets[idx]),
            torch.tensor(self.result_targets[idx]),
        )


class NNUE(nn.Module):
    """Small NNUE network for Togyz Kumalak"""

    def __init__(self, input_size=INPUT_SIZE, hidden1=256, hidden2=32):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)

    def forward(self, x):
        x = torch.clamp(self.fc1(x), 0.0, 1.0)  # ClippedReLU
        x = torch.clamp(self.fc2(x), 0.0, 1.0)  # ClippedReLU
        x = self.fc3(x)
        return x.squeeze(-1)


def sigmoid_eval(x, k=K):
    """Convert eval to win probability"""
    return torch.sigmoid(x / (k / 4.0))  # scale factor for torch sigmoid


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Load data
    dataset = PositionDataset(args.data, max_positions=args.max_positions)

    # Split train/val
    n = len(dataset)
    n_val = min(n // 10, 50000)
    n_train = n - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size * 2,
                            num_workers=2, pin_memory=True)

    # Model
    model = NNUE(hidden1=args.hidden1, hidden2=args.hidden2).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {INPUT_SIZE} → {args.hidden1} → {args.hidden2} → 1")
    print(f"Parameters: {params:,}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    lam = args.lam  # blend between eval and result targets

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for features, evals, results in train_loader:
            features = features.to(device)
            evals = evals.to(device)
            results = results.to(device)

            pred = model(features)

            # Prediction as win probability
            pred_wp = sigmoid_eval(pred)

            # Target: blend of search eval and game result
            eval_wp = sigmoid_eval(evals)
            target = lam * eval_wp + (1 - lam) * results

            loss = torch.mean((pred_wp - target) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for features, evals, results in val_loader:
                features = features.to(device)
                evals = evals.to(device)
                results = results.to(device)

                pred = model(features)
                pred_wp = sigmoid_eval(pred)
                eval_wp = sigmoid_eval(evals)
                target = lam * eval_wp + (1 - lam) * results

                val_loss += torch.mean((pred_wp - target) ** 2).item()
                n_val_batches += 1

        avg_train = total_loss / n_batches
        avg_val = val_loss / n_val_batches

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), 'nnue_best.pt')

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs}  train={avg_train:.6f}  val={avg_val:.6f}  lr={scheduler.get_last_lr()[0]:.6f}")

    # Save final model
    torch.save(model.state_dict(), 'nnue_final.pt')

    # Load best and export for Rust
    model.load_state_dict(torch.load('nnue_best.pt', weights_only=True))
    export_weights(model, 'nnue_weights.json', args)

    print(f"\nTraining complete! Best val loss: {best_val_loss:.6f}")
    print(f"Exported weights to nnue_weights.json")


def export_weights(model, path, args):
    """Export weights as JSON for Rust to read"""
    state = model.state_dict()

    weights = {
        'input_size': INPUT_SIZE,
        'hidden1': args.hidden1,
        'hidden2': args.hidden2,
        'fc1_weight': state['fc1.weight'].cpu().numpy().tolist(),
        'fc1_bias': state['fc1.bias'].cpu().numpy().tolist(),
        'fc2_weight': state['fc2.weight'].cpu().numpy().tolist(),
        'fc2_bias': state['fc2.bias'].cpu().numpy().tolist(),
        'fc3_weight': state['fc3.weight'].cpu().numpy().tolist(),
        'fc3_bias': state['fc3.bias'].cpu().numpy().tolist(),
    }

    with open(path, 'w') as f:
        json.dump(weights, f)

    # Also export as binary for fast Rust loading
    export_binary(model, 'nnue_weights.bin', args)


def export_binary(model, path, args):
    """Export quantized weights as binary for Rust.

    Format:
      header: hidden1(u16), hidden2(u16)
      fc1_weight: hidden1 * input_size i16 (row-major)
      fc1_bias: hidden1 i16
      fc2_weight: hidden2 * hidden1 i16
      fc2_bias: hidden2 i16
      fc3_weight: 1 * hidden2 i16
      fc3_bias: 1 i16

    Quantization: multiply floats by 64 and round to i16
    """
    SCALE = 64  # quantization scale

    state = model.state_dict()

    with open(path, 'wb') as f:
        f.write(struct.pack('<HH', args.hidden1, args.hidden2))

        for name in ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias',
                      'fc3.weight', 'fc3.bias']:
            tensor = state[name].cpu().float().numpy().flatten()
            quantized = np.clip(tensor * SCALE, -32000, 32000).astype(np.int16)
            f.write(quantized.tobytes())

    total_bytes = os.path.getsize(path)
    print(f"Binary weights: {path} ({total_bytes:,} bytes, scale={SCALE})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train NNUE for Togyz Kumalak')
    parser.add_argument('--data', default='training_data.bin', help='Binary training data')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden1', type=int, default=256)
    parser.add_argument('--hidden2', type=int, default=32)
    parser.add_argument('--lam', type=float, default=0.75,
                        help='Blend: 0=pure result, 1=pure eval')
    parser.add_argument('--max-positions', type=int, default=None)
    args = parser.parse_args()

    train(args)
