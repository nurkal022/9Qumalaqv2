"""
NNUE Training v2 for Togyz Kumalak
- Adaptive lambda: uses eval when available, result-only for PlayOK data (eval=0)
- Supports multiple data files
- Better logging
"""
import struct
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import json
import time

RECORD_SIZE = 26
INPUT_SIZE = 40
K = 1050.0


class PositionDataset(Dataset):
    def __init__(self, paths, max_positions=None):
        """Load from one or more binary files."""
        if isinstance(paths, str):
            paths = [paths]

        all_raw = []
        for path in paths:
            file_size = os.path.getsize(path)
            n = file_size // RECORD_SIZE
            print(f"  Loading {path}: {n:,} positions")
            with open(path, 'rb') as f:
                all_raw.append(f.read(n * RECORD_SIZE))

        raw = b''.join(all_raw)
        num_records = len(raw) // RECORD_SIZE
        if max_positions:
            num_records = min(num_records, max_positions)
            raw = raw[:num_records * RECORD_SIZE]

        print(f"  Total: {num_records:,} positions")

        data = np.frombuffer(raw, dtype=np.uint8).reshape(num_records, RECORD_SIZE)

        # Parse fields
        pits_w = data[:, 0:9].astype(np.float32)
        pits_b = data[:, 9:18].astype(np.float32)
        kazan_w = data[:, 18:19].astype(np.float32)
        kazan_b = data[:, 19:20].astype(np.float32)
        tuzdyk_w = data[:, 20].astype(np.int8)
        tuzdyk_b = data[:, 21].astype(np.int8)
        side = data[:, 22]

        eval_bytes = data[:, 23:25].copy()
        evals = eval_bytes.view(np.int16).astype(np.float32).flatten()

        # Rescale evals: if mean abs eval is too large, normalize to HCE-like scale
        # HCE evals: mean_abs ~519, K=1050. Sigmoid works well in [-800, 800] range
        nonzero_mask = evals != 0
        if nonzero_mask.sum() > 0:
            mean_abs = np.abs(evals[nonzero_mask]).mean()
            TARGET_MEAN_ABS = 500.0  # HCE-like scale
            if mean_abs > TARGET_MEAN_ABS * 2:
                scale = TARGET_MEAN_ABS / mean_abs
                print(f"  Rescaling evals: mean_abs={mean_abs:.0f} -> {TARGET_MEAN_ABS:.0f} (scale={scale:.4f})")
                evals[nonzero_mask] *= scale
            # Final clip at ±1500 for safety
            clipped = np.sum(np.abs(evals) > 1500)
            if clipped > 0:
                print(f"  Clipping {clipped:,} outlier evals beyond ±1500")
                evals = np.clip(evals, -1500, 1500)

        results = data[:, 25].astype(np.float32) / 2.0

        # Side-to-move perspective
        is_white = (side == 0)

        my_pits = np.where(is_white[:, None], pits_w, pits_b) / 50.0
        opp_pits = np.where(is_white[:, None], pits_b, pits_w) / 50.0
        my_kazan = np.where(is_white[:, None], kazan_w, kazan_b) / 82.0
        opp_kazan = np.where(is_white[:, None], kazan_b, kazan_w) / 82.0

        my_tuz_idx = np.where(is_white, tuzdyk_w, tuzdyk_b).astype(np.int64)
        opp_tuz_idx = np.where(is_white, tuzdyk_b, tuzdyk_w).astype(np.int64)

        my_tuz = np.zeros((num_records, 10), dtype=np.float32)
        opp_tuz = np.zeros((num_records, 10), dtype=np.float32)

        for i in range(num_records):
            idx = my_tuz_idx[i]
            my_tuz[i, idx if idx >= 0 else 9] = 1.0
            idx = opp_tuz_idx[i]
            opp_tuz[i, idx if idx >= 0 else 9] = 1.0

        self.features = np.concatenate([
            my_pits, opp_pits, my_kazan, opp_kazan, my_tuz, opp_tuz
        ], axis=1)

        self.eval_targets = evals
        result_stm = np.where(is_white, results, 1.0 - results)
        self.result_targets = result_stm

        # Per-record lambda: 0.0 if eval==0 (PlayOK), lam_eval otherwise
        self.has_eval = (evals != 0).astype(np.float32)

        n_with_eval = int(self.has_eval.sum())
        n_no_eval = num_records - n_with_eval
        print(f"  With eval: {n_with_eval:,} ({n_with_eval*100//num_records}%)")
        print(f"  No eval (result only): {n_no_eval:,} ({n_no_eval*100//num_records}%)")

        abs_evals = np.abs(evals[evals != 0]) if n_with_eval > 0 else np.array([0])
        print(f"  Eval stats (non-zero): mean={abs_evals.mean():.0f}, median={np.median(abs_evals):.0f}, p90={np.percentile(abs_evals, 90):.0f}")
        print(f"  Results: W={np.sum(results > 0.6):,}, D={np.sum(np.abs(results - 0.5) < 0.1):,}, L={np.sum(results < 0.4):,}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx]),
            torch.tensor(self.eval_targets[idx]),
            torch.tensor(self.result_targets[idx]),
            torch.tensor(self.has_eval[idx]),
        )


class NNUE(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden1=256, hidden2=32):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)

    def forward(self, x):
        x = torch.clamp(self.fc1(x), 0.0, 1.0)
        x = torch.clamp(self.fc2(x), 0.0, 1.0)
        x = self.fc3(x)
        return x.squeeze(-1)


def sigmoid_eval(x, k=K):
    return torch.sigmoid(x / (k / 4.0))


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Lambda for eval records: {args.lam}")
    print()

    # Load data
    data_files = args.data.split(',')
    dataset = PositionDataset(data_files, max_positions=args.max_positions)

    # Split train/val
    n = len(dataset)
    n_val = min(n // 10, 100000)
    n_train = n - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size * 2,
                            num_workers=2, pin_memory=True)

    model = NNUE(hidden1=args.hidden1, hidden2=args.hidden2).to(device)
    if args.resume:
        print(f"Resuming from: {args.resume}")
        model.load_state_dict(torch.load(args.resume, weights_only=True, map_location=device))
    params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {INPUT_SIZE} → {args.hidden1} → {args.hidden2} → 1")
    print(f"Parameters: {params:,}")
    print(f"Train: {n_train:,}, Val: {n_val:,}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for features, evals, results, has_eval in train_loader:
            features = features.to(device)
            evals = evals.to(device)
            results = results.to(device)
            has_eval = has_eval.to(device)

            pred = model(features)
            pred_wp = sigmoid_eval(pred)

            # Adaptive target per record:
            # If has_eval: target = lam * sigmoid(eval) + (1-lam) * result
            # If no eval:  target = result
            eval_wp = sigmoid_eval(evals)
            per_lam = has_eval * args.lam  # 0 for PlayOK, lam for self-play
            target = per_lam * eval_wp + (1 - per_lam) * results

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
            for features, evals, results, has_eval in val_loader:
                features = features.to(device)
                evals = evals.to(device)
                results = results.to(device)
                has_eval = has_eval.to(device)

                pred = model(features)
                pred_wp = sigmoid_eval(pred)
                eval_wp = sigmoid_eval(evals)
                per_lam = has_eval * args.lam
                target = per_lam * eval_wp + (1 - per_lam) * results

                val_loss += torch.mean((pred_wp - target) ** 2).item()
                n_val_batches += 1

        avg_train = total_loss / n_batches
        avg_val = val_loss / n_val_batches

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), args.output + '_best.pt')

        elapsed = time.time() - start_time
        eta = elapsed / (epoch + 1) * (args.epochs - epoch - 1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs}  train={avg_train:.6f}  val={avg_val:.6f}  lr={scheduler.get_last_lr()[0]:.6f}  ETA:{eta/60:.0f}m", flush=True)

    # Save final
    torch.save(model.state_dict(), args.output + '_final.pt')

    # Export best
    model.load_state_dict(torch.load(args.output + '_best.pt', weights_only=True))
    export_binary(model, args.output + '.bin', args)

    print(f"\nDone! Best val loss: {best_val_loss:.6f}")
    print(f"Total time: {(time.time()-start_time)/60:.1f} min")


def export_binary(model, path, args):
    SCALE = 64
    state = model.state_dict()
    with open(path, 'wb') as f:
        f.write(struct.pack('<HH', args.hidden1, args.hidden2))
        for name in ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias',
                      'fc3.weight', 'fc3.bias']:
            tensor = state[name].cpu().float().numpy().flatten()
            quantized = np.clip(tensor * SCALE, -32000, 32000).astype(np.int16)
            f.write(quantized.tobytes())
    print(f"Binary weights: {path} ({os.path.getsize(path):,} bytes)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Binary data file(s), comma-separated')
    parser.add_argument('--output', default='nnue_exp', help='Output prefix')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8192)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden1', type=int, default=256)
    parser.add_argument('--hidden2', type=int, default=32)
    parser.add_argument('--lam', type=float, default=0.75)
    parser.add_argument('--max-positions', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None, help='Resume from .pt checkpoint')
    args = parser.parse_args()
    train(args)
