#!/usr/bin/env python3
"""
Multi-seed NNUE training with round-robin tournament to select best model.
Trains N models with different seeds, runs matches, promotes the winner.

Usage (fine-tune Gen5 on depth-14 data, 3 seeds):
  python train_multiseed.py \
    --data gen7_d14_training_data.bin \
    --base nnue_weights.bin \
    --seeds 42 123 777 \
    --lr 0.0001 --epochs 30 --K 400 --lam 0.5 \
    --arch 40-256-32-1 \
    --output nnue_weights_gen7.bin

Usage (train from scratch, new 52-input arch):
  python train_multiseed.py \
    --data gen7_d14_training_data.bin \
    --seeds 42 123 777 \
    --lr 0.001 --epochs 100 --K 400 --lam 0.5 \
    --arch 52-256-64-32-1 \
    --output nnue_weights_gen7_52.bin
"""
import struct
import subprocess
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

RECORD_SIZE = 26


class PositionDataset(Dataset):
    def __init__(self, paths, input_size=40):
        if isinstance(paths, str):
            paths = [paths]
        all_data = []
        for path in paths:
            if not os.path.exists(path):
                print(f"Warning: {path} not found, skipping")
                continue
            num_records = os.path.getsize(path) // RECORD_SIZE
            print(f"  Loading {num_records:,} positions from {path}...")
            with open(path, 'rb') as f:
                raw = f.read(num_records * RECORD_SIZE)
            data = np.frombuffer(raw, dtype=np.uint8).reshape(num_records, RECORD_SIZE)
            all_data.append(data)

        data = np.concatenate(all_data, axis=0)
        n = len(data)
        print(f"  Total: {n:,} positions")

        pits_w = data[:, 0:9].astype(np.float32)
        pits_b = data[:, 9:18].astype(np.float32)
        kaz_w = data[:, 18:19].astype(np.float32)
        kaz_b = data[:, 19:20].astype(np.float32)
        tuz_w = data[:, 20].astype(np.int8)
        tuz_b = data[:, 21].astype(np.int8)
        side = data[:, 22]
        evals = data[:, 23:25].copy().view(np.int16).astype(np.float32).flatten()
        results = data[:, 25].astype(np.float32) / 2.0

        is_w = (side == 0)
        my_p = np.where(is_w[:, None], pits_w, pits_b) / 50.0
        op_p = np.where(is_w[:, None], pits_b, pits_w) / 50.0
        my_k = np.where(is_w[:, None], kaz_w, kaz_b) / 82.0
        op_k = np.where(is_w[:, None], kaz_b, kaz_w) / 82.0

        my_ti = np.where(is_w, tuz_w, tuz_b).astype(np.int64)
        op_ti = np.where(is_w, tuz_b, tuz_w).astype(np.int64)
        my_t = np.zeros((n, 10), dtype=np.float32)
        op_t = np.zeros((n, 10), dtype=np.float32)
        for i in range(n):
            my_t[i, my_ti[i] if my_ti[i] >= 0 else 9] = 1.0
            op_t[i, op_ti[i] if op_ti[i] >= 0 else 9] = 1.0

        base = np.concatenate([my_p, op_p, my_k, op_k, my_t, op_t], axis=1)  # 40 features

        if input_size == 52:
            my_stones = my_p.sum(axis=1, keepdims=True)
            op_stones = op_p.sum(axis=1, keepdims=True)
            my_act = (my_p * 50 > 0).sum(axis=1, keepdims=True).astype(np.float32) / 9.0
            op_act = (op_p * 50 > 0).sum(axis=1, keepdims=True).astype(np.float32) / 9.0
            my_hvy = (my_p * 50 >= 12).sum(axis=1, keepdims=True).astype(np.float32) / 9.0
            op_hvy = (op_p * 50 >= 12).sum(axis=1, keepdims=True).astype(np.float32) / 9.0
            my_wk = ((my_p * 50 >= 1) & (my_p * 50 <= 2)).sum(axis=1, keepdims=True).astype(np.float32) / 9.0
            op_wk = ((op_p * 50 >= 1) & (op_p * 50 <= 2)).sum(axis=1, keepdims=True).astype(np.float32) / 9.0
            my_r = my_p[:, 6:9].sum(axis=1, keepdims=True)
            op_r = op_p[:, 6:9].sum(axis=1, keepdims=True)
            phase = ((pits_w.sum(axis=1) + pits_b.sum(axis=1)) / 162.0).reshape(-1, 1)
            kd = (my_k - op_k)
            ext = np.concatenate([my_stones, op_stones, my_act, op_act,
                                   my_hvy, op_hvy, my_wk, op_wk,
                                   my_r, op_r, phase, kd], axis=1)
            self.features = np.concatenate([base, ext], axis=1)
        else:
            self.features = base

        self.eval_targets = evals
        self.result_targets = np.where(is_w, results, 1.0 - results)
        stones = (pits_w.sum(axis=1) + pits_b.sum(axis=1))
        self.weights = np.ones(n, dtype=np.float32)
        self.weights[stones <= 30] = 2.0
        self.weights[stones <= 15] = 3.0
        self.has_eval = (np.abs(evals) > 1).astype(np.float32)

    def __len__(self): return len(self.features)
    def __getitem__(self, i):
        return (torch.tensor(self.features[i]), torch.tensor(self.eval_targets[i]),
                torch.tensor(self.result_targets[i]), torch.tensor(self.weights[i]),
                torch.tensor(self.has_eval[i]))


class NNUE(nn.Module):
    def __init__(self, input_size, h1, h2, h3=0):
        super().__init__()
        self.h3 = h3
        self.fc1 = nn.Linear(input_size, h1)
        self.fc2 = nn.Linear(h1, h2)
        if h3 > 0:
            self.fc3 = nn.Linear(h2, h3)
            self.fc4 = nn.Linear(h3, 1)
        else:
            self.fc3 = nn.Linear(h2, 1)
            self.fc4 = None

    def forward(self, x):
        x = torch.clamp(self.fc1(x), 0.0, 1.0)
        x = torch.clamp(self.fc2(x), 0.0, 1.0)
        if self.h3 > 0:
            x = torch.clamp(self.fc3(x), 0.0, 1.0)
            x = self.fc4(x)
        else:
            x = self.fc3(x)
        return x.squeeze(-1)


def load_base_weights(model, path):
    """Load Gen5 40-input weights into model (only fc1-fc3 layers)."""
    with open(path, 'rb') as f:
        data = f.read()
    first_val = struct.unpack_from('<H', data, 0)[0]
    if first_val >= 128:
        # Old format
        h1 = first_val
        h2 = struct.unpack_from('<H', data, 2)[0]
        offset = 4
        in_sz = 40
        h3 = 0
    else:
        in_sz = first_val
        h1 = struct.unpack_from('<H', data, 2)[0]
        h2 = struct.unpack_from('<H', data, 4)[0]
        h3 = struct.unpack_from('<H', data, 6)[0]
        offset = 8

    SCALE = 64.0
    def read_weights(count):
        nonlocal offset
        raw = data[offset:offset + count * 2]
        offset += count * 2
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / SCALE

    fc1_w = read_weights(h1 * in_sz).reshape(h1, in_sz)
    fc1_b = read_weights(h1)
    fc2_w = read_weights(h2 * h1).reshape(h2, h1)
    fc2_b = read_weights(h2)

    sd = model.state_dict()
    if model.fc1.in_features == in_sz:
        # Same input size — direct load
        sd['fc1.weight'] = torch.tensor(fc1_w)
        sd['fc1.bias'] = torch.tensor(fc1_b)
        sd['fc2.weight'] = torch.tensor(fc2_w)
        sd['fc2.bias'] = torch.tensor(fc2_b)
        fc3_w = read_weights(h2)  # output layer weights
        fc3_b = read_weights(1)
        if model.h3 > 0:
            # Can't load fc3/fc4 from 3-layer to 4-layer — skip
            pass
        else:
            sd['fc3.weight'] = torch.tensor(fc3_w.reshape(1, h2))
            sd['fc3.bias'] = torch.tensor(fc3_b)
    else:
        # Different input size — copy shared fc2+ layers only
        print(f"  Input size mismatch ({model.fc1.in_features} vs {in_sz}): loading fc2+ only")
        sd['fc2.weight'] = torch.tensor(fc2_w)
        sd['fc2.bias'] = torch.tensor(fc2_b)

    model.load_state_dict(sd)
    print(f"  Loaded base weights from {path}")


def export_binary(model, path, input_size, h1, h2, h3):
    SCALE = 64
    state = model.state_dict()
    with open(path, 'wb') as f:
        if input_size < 128:
            f.write(struct.pack('<HHHH', input_size, h1, h2, h3))
            names = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias',
                     'fc3.weight', 'fc3.bias']
            if h3 > 0:
                names += ['fc4.weight', 'fc4.bias']
        else:
            f.write(struct.pack('<HH', h1, h2))
            names = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias',
                     'fc3.weight', 'fc3.bias']
        for name in names:
            t = state[name].cpu().float().numpy().flatten()
            q = np.clip(t * SCALE, -32000, 32000).astype(np.int16)
            f.write(q.tobytes())
    print(f"  Exported: {path} ({os.path.getsize(path):,} bytes)")


def train_model(dataset, input_size, h1, h2, h3, lr, epochs, K, lam, seed, base_weights=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n = len(dataset)
    n_val = min(n // 10, 50000)
    n_train = n - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )
    train_loader = DataLoader(train_set, batch_size=4096, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=8192, num_workers=4, pin_memory=True)

    model = NNUE(input_size, h1, h2, h3).to(device)
    if base_weights:
        load_base_weights(model, base_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = float('inf')
    best_state = None

    def sig(x): return torch.sigmoid(x / (K / 4.0))

    for epoch in range(epochs):
        model.train()
        for feat, ev, res, wt, he in train_loader:
            feat, ev, res, wt, he = feat.to(device), ev.to(device), res.to(device), wt.to(device), he.to(device)
            pred = model(feat)
            eff_lam = lam * he
            tgt = eff_lam * sig(ev) + (1 - eff_lam) * res
            loss = torch.mean((sig(pred) - tgt) ** 2 * wt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        nb = 0
        with torch.no_grad():
            for feat, ev, res, wt, he in val_loader:
                feat, ev, res, wt, he = feat.to(device), ev.to(device), res.to(device), wt.to(device), he.to(device)
                pred = model(feat)
                eff_lam = lam * he
                tgt = eff_lam * sig(ev) + (1 - eff_lam) * res
                val_loss += torch.mean((sig(pred) - tgt) ** 2 * wt).item()
                nb += 1
        avg_val = val_loss / nb
        if avg_val < best_val:
            best_val = avg_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            marker = " *"
        else:
            marker = ""
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  val={avg_val:.6f}{marker}")

    model.load_state_dict(best_state)
    return model, best_val


def run_match(engine, weights_a, weights_b, games=100, time_ms=500):
    """Run NNUE vs NNUE match, return score of A."""
    result = subprocess.run(
        [engine, 'match-nnue', weights_a, weights_b, str(games), str(time_ms)],
        capture_output=True, text=True, cwd=os.path.dirname(engine)
    )
    output = result.stdout + result.stderr
    # Parse "A score: XX.X%"
    for line in output.split('\n'):
        if 'A score:' in line or 'score:' in line.lower():
            try:
                pct = float(line.split('%')[0].split()[-1])
                return pct / 100.0
            except: pass
    return 0.5  # fallback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Comma-separated data files')
    parser.add_argument('--base', default=None, help='Base weights for fine-tuning')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 777])
    parser.add_argument('--arch', default='40-256-32-1',
                        help='Architecture: input-h1-h2[-h3]-1 (e.g. 52-256-64-32-1)')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--K', type=float, default=400.0)
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--output', default='nnue_weights_gen7.bin')
    parser.add_argument('--match-games', type=int, default=100)
    parser.add_argument('--engine', default='./target/release/togyzkumalaq-engine')
    args = parser.parse_args()

    # Parse architecture
    arch_parts = [int(x) for x in args.arch.split('-')]
    assert arch_parts[-1] == 1, "Architecture must end with 1 (output)"
    input_size = arch_parts[0]
    if len(arch_parts) == 4:  # input-h1-h2-1
        h1, h2, h3 = arch_parts[1], arch_parts[2], 0
    elif len(arch_parts) == 5:  # input-h1-h2-h3-1
        h1, h2, h3 = arch_parts[1], arch_parts[2], arch_parts[3]
    else:
        raise ValueError(f"Unsupported arch: {args.arch}")

    print(f"Architecture: {args.arch}")
    print(f"Seeds: {args.seeds}")
    print(f"LR={args.lr}, Epochs={args.epochs}, K={args.K}, Lambda={args.lam}")
    if args.base:
        print(f"Fine-tuning from: {args.base}")
    print()

    # Load dataset once
    data_files = args.data.split(',')
    dataset = PositionDataset(data_files, input_size=input_size)

    # Train each seed
    models = {}
    for seed in args.seeds:
        print(f"\n{'='*50}")
        print(f"Training seed {seed}...")
        model, best_val = train_model(
            dataset, input_size, h1, h2, h3,
            args.lr, args.epochs, args.K, args.lam,
            seed, base_weights=args.base
        )
        out_path = args.output.replace('.bin', f'_s{seed}.bin')
        export_binary(model, out_path, input_size, h1, h2, h3)
        models[seed] = (model, best_val, out_path)
        print(f"  Best val loss: {best_val:.6f} → {out_path}")

    if len(args.seeds) == 1:
        # Single seed — just copy output
        s = args.seeds[0]
        os.rename(models[s][2], args.output)
        print(f"\nOutput: {args.output}")
        return

    # Round-robin match to find best model
    print(f"\n{'='*50}")
    print(f"Round-robin match ({args.match_games} games each)...")
    engine_path = os.path.abspath(args.engine)
    scores = {s: 0.0 for s in args.seeds}

    for i, sa in enumerate(args.seeds):
        for sb in args.seeds[i+1:]:
            path_a = models[sa][2]
            path_b = models[sb][2]
            score = run_match(engine_path, path_a, path_b, args.match_games)
            print(f"  seed {sa} vs seed {sb}: {score*100:.1f}%")
            scores[sa] += score
            scores[sb] += (1.0 - score)

    best_seed = max(scores, key=lambda s: scores[s])
    print(f"\nBest seed: {best_seed} (total score {scores[best_seed]:.2f})")

    import shutil
    shutil.copy(models[best_seed][2], args.output)
    print(f"Output: {args.output}")

    # Show summary
    print("\nAll seeds:")
    for s in args.seeds:
        val = models[s][1]
        sc = scores[s]
        best_mark = " ← BEST" if s == best_seed else ""
        print(f"  Seed {s}: val_loss={val:.6f}, match_score={sc:.2f}{best_mark}")


if __name__ == '__main__':
    main()
