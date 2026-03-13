"""
NNUE Training V2 for Togyz Kumalak - Scaled Architecture

Supports configurable architecture for the self-play loop.
Key improvement: support for larger networks (512→32→1).

Architecture options:
  - Standard: Input(40) → 256 → 32 → 1  (18K params)
  - Scaled:   Input(40) → 512 → 32 → 1  (33K params)
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
K = 1050.0  # Sigmoid scaling from Texel tuning (overridden by --K arg)


class PositionDataset(Dataset):
    def __init__(self, paths, max_positions=None):
        """Load from one or more binary files"""
        if isinstance(paths, str):
            paths = [paths]

        all_data = []
        for path in paths:
            if not os.path.exists(path):
                print(f"Warning: {path} not found, skipping")
                continue
            file_size = os.path.getsize(path)
            num_records = file_size // RECORD_SIZE
            print(f"  Loading {num_records:,} positions from {path}...")
            with open(path, 'rb') as f:
                raw = f.read(num_records * RECORD_SIZE)
            data = np.frombuffer(raw, dtype=np.uint8).reshape(num_records, RECORD_SIZE)
            all_data.append(data)

        data = np.concatenate(all_data, axis=0)
        num_records = len(data)

        if max_positions and num_records > max_positions:
            idx = np.random.choice(num_records, max_positions, replace=False)
            data = data[idx]
            num_records = max_positions

        print(f"  Total: {num_records:,} positions")

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
        results = data[:, 25].astype(np.float32) / 2.0

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

        base_features = np.concatenate([
            my_pits, opp_pits, my_kazan, opp_kazan, my_tuz, opp_tuz
        ], axis=1)  # shape: (N, 40)

        # Extended features (12 strategic features for 52-input architecture)
        my_stones = my_pits.sum(axis=1, keepdims=True)        # total pit stones / 50 (already scaled)
        opp_stones = opp_pits.sum(axis=1, keepdims=True)
        my_active = (my_pits * 50 > 0).sum(axis=1, keepdims=True).astype(np.float32) / 9.0
        opp_active = (opp_pits * 50 > 0).sum(axis=1, keepdims=True).astype(np.float32) / 9.0
        my_heavy = (my_pits * 50 >= 12).sum(axis=1, keepdims=True).astype(np.float32) / 9.0
        opp_heavy = (opp_pits * 50 >= 12).sum(axis=1, keepdims=True).astype(np.float32) / 9.0
        my_weak = ((my_pits * 50 >= 1) & (my_pits * 50 <= 2)).sum(axis=1, keepdims=True).astype(np.float32) / 9.0
        opp_weak = ((opp_pits * 50 >= 1) & (opp_pits * 50 <= 2)).sum(axis=1, keepdims=True).astype(np.float32) / 9.0
        my_right = my_pits[:, 6:9].sum(axis=1, keepdims=True)   # already /50 scaled
        opp_right = opp_pits[:, 6:9].sum(axis=1, keepdims=True)
        # Game phase: total board stones / 162 (normalize: pits are /50 scaled, 9 pits × 2 sides)
        total_stones_raw = (pits_w.sum(axis=1) + pits_b.sum(axis=1)).reshape(-1, 1) / 162.0
        # Kazan difference: (my_kazan - opp_kazan) / 82
        kaz_diff = (my_kazan - opp_kazan)  # already /82 scaled

        # === NEW: 6 strategic features (52-57) ===
        # Use raw pit values (not /50 scaled) for exact comparisons
        my_pits_raw = np.where(is_white[:, None], pits_w, pits_b)
        opp_pits_raw = np.where(is_white[:, None], pits_b, pits_w)
        my_tuz_idx_arr = my_tuz_idx  # shape: (N,)
        opp_tuz_idx_arr = opp_tuz_idx

        # [52-53] Tuzdyk threats: opp pits with exactly 2 stones (candidates for tuzdyk)
        # Only count if side doesn't already have a tuzdyk
        # Exclude pit 9 (index 8) — tuzdyk can only be on pits 1-8 (indices 0-7)
        opp_is_2 = (opp_pits_raw[:, :8] == 2)  # shape: (N, 8)
        # Exclude opponent's existing tuzdyk position
        for pit_idx in range(8):
            opp_is_2[:, pit_idx] &= (opp_tuz_idx_arr != pit_idx)
        my_has_no_tuz = (my_tuz_idx_arr < 0)  # shape: (N,)
        my_threats = opp_is_2.sum(axis=1) * my_has_no_tuz  # zero if already have tuzdyk
        my_threats = my_threats.reshape(-1, 1).astype(np.float32) / 8.0

        my_is_2 = (my_pits_raw[:, :8] == 2)
        for pit_idx in range(8):
            my_is_2[:, pit_idx] &= (my_tuz_idx_arr != pit_idx)
        opp_has_no_tuz = (opp_tuz_idx_arr < 0)
        opp_threats = my_is_2.sum(axis=1) * opp_has_no_tuz
        opp_threats = opp_threats.reshape(-1, 1).astype(np.float32) / 8.0

        # [54-55] Starvation pressure: max(0, 20 - stones)^2 / 400
        my_stones_raw = my_pits_raw.sum(axis=1).astype(np.float32)
        opp_stones_raw = opp_pits_raw.sum(axis=1).astype(np.float32)
        opp_starv = np.maximum(0, 20.0 - opp_stones_raw) ** 2 / 400.0
        my_starv = np.maximum(0, 20.0 - my_stones_raw) ** 2 / 400.0
        opp_starv = opp_starv.reshape(-1, 1)
        my_starv = my_starv.reshape(-1, 1)

        # [56-57] Capture targets: opponent pits with even stones > 0
        opp_even = ((opp_pits_raw > 0) & (opp_pits_raw % 2 == 0)).sum(axis=1).reshape(-1, 1).astype(np.float32) / 9.0
        my_even = ((my_pits_raw > 0) & (my_pits_raw % 2 == 0)).sum(axis=1).reshape(-1, 1).astype(np.float32) / 9.0

        ext_features = np.concatenate([
            my_stones, opp_stones,   # [40, 41]: total stones (using /50 scale, consistent with pits)
            my_active, opp_active,   # [42, 43]
            my_heavy, opp_heavy,     # [44, 45]
            my_weak, opp_weak,       # [46, 47]
            my_right, opp_right,     # [48, 49]
            total_stones_raw,        # [50]: game phase
            kaz_diff,                # [51]: kazan diff
            my_threats, opp_threats,  # [52, 53]: tuzdyk threats
            opp_starv, my_starv,     # [54, 55]: starvation pressure
            opp_even, my_even,       # [56, 57]: capture targets
        ], axis=1)  # shape: (N, 18)

        self.features_40 = base_features
        self.features = np.concatenate([base_features, ext_features], axis=1)  # shape: (N, 58)

        self.eval_targets = evals
        result_stm = np.where(is_white, results, 1.0 - results)
        self.result_targets = result_stm

        pit_stones = pits_w.sum(axis=1) + pits_b.sum(axis=1)
        self.sample_weights = np.ones(num_records, dtype=np.float32)
        self.sample_weights[pit_stones <= 30] = 2.0
        self.sample_weights[pit_stones <= 15] = 3.0
        self.has_eval = (np.abs(evals) > 1).astype(np.float32)

        endgame_count = np.sum(pit_stones <= 30)
        print(f"  Features: {self.features.shape}")
        print(f"  Eval range: [{evals.min():.0f}, {evals.max():.0f}]")
        print(f"  Results: W={np.sum(results > 0.6)}, D={np.sum(np.abs(results - 0.5) < 0.1)}, L={np.sum(results < 0.4)}")
        print(f"  Endgame: {endgame_count:,} ({100*endgame_count/num_records:.1f}%)")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx]),
            torch.tensor(self.eval_targets[idx]),
            torch.tensor(self.result_targets[idx]),
            torch.tensor(self.sample_weights[idx]),
            torch.tensor(self.has_eval[idx]),
        )


class NNUE(nn.Module):
    """NNUE with configurable architecture — supports 3 or 4 layers"""

    def __init__(self, input_size=INPUT_SIZE, hidden1=256, hidden2=32, hidden3=0):
        super().__init__()
        self.hidden3 = hidden3
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        if hidden3 > 0:
            self.fc3 = nn.Linear(hidden2, hidden3)
            self.fc4 = nn.Linear(hidden3, 1)
        else:
            self.fc3 = nn.Linear(hidden2, 1)
            self.fc4 = None

    def forward(self, x):
        x = torch.clamp(self.fc1(x), 0.0, 1.0)
        x = torch.clamp(self.fc2(x), 0.0, 1.0)
        if self.hidden3 > 0:
            x = torch.clamp(self.fc3(x), 0.0, 1.0)
            x = self.fc4(x)
        else:
            x = self.fc3(x)
        return x.squeeze(-1)


def sigmoid_eval(x, k=400.0):
    return torch.sigmoid(x / (k / 4.0))


def load_binary_weights(model, bin_path, input_size):
    """Load quantized binary weights into PyTorch model (supports old and new formats)"""
    SCALE_Q = 64
    with open(bin_path, 'rb') as f:
        data = f.read()

    first_u16 = struct.unpack('<H', data[0:2])[0]
    if first_u16 < 128:
        # New format: [input_size, hidden1, hidden2, hidden3]
        inp, h1, h2, h3 = struct.unpack('<HHHH', data[0:8])
        offset = 8
        print(f"  New format: {inp}→{h1}→{h2}→{h3}→1")
    else:
        # Old format: [hidden1, hidden2]
        h1, h2 = struct.unpack('<HH', data[0:4])
        h3 = 0
        inp = 40
        offset = 4
        print(f"  Old format: {inp}→{h1}→{h2}→1")

    assert inp == input_size, f"Weight input_size {inp} != requested {input_size}"

    def read_i16_array(n):
        nonlocal offset
        arr = np.frombuffer(data[offset:offset+n*2], dtype=np.int16).astype(np.float32) / SCALE_Q
        offset += n * 2
        return arr

    state = model.state_dict()
    state['fc1.weight'] = torch.tensor(read_i16_array(h1 * inp).reshape(h1, inp))
    state['fc1.bias'] = torch.tensor(read_i16_array(h1))
    state['fc2.weight'] = torch.tensor(read_i16_array(h2 * h1).reshape(h2, h1))
    state['fc2.bias'] = torch.tensor(read_i16_array(h2))
    if h3 > 0:
        state['fc3.weight'] = torch.tensor(read_i16_array(h3 * h2).reshape(h3, h2))
        state['fc3.bias'] = torch.tensor(read_i16_array(h3))
        state['fc4.weight'] = torch.tensor(read_i16_array(h3).reshape(1, h3))
        state['fc4.bias'] = torch.tensor(read_i16_array(1))
    else:
        state['fc3.weight'] = torch.tensor(read_i16_array(h2).reshape(1, h2))
        state['fc3.bias'] = torch.tensor(read_i16_array(1))

    model.load_state_dict(state)
    print(f"  Loaded binary weights from {bin_path}")
    return model


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    data_files = args.data.split(',')
    dataset = PositionDataset(data_files, max_positions=args.max_positions)

    n = len(dataset)
    n_val = min(n // 10, 50000)
    n_train = n - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size * 2,
                            num_workers=4, pin_memory=True)

    input_size = args.input_size
    # Select feature set: 58 (full), 52 (trim last 6), or 40 (base only)
    if input_size == 58:
        pass  # already 58-wide
    elif input_size == 52:
        dataset.features = dataset.features[:, :52]
    elif input_size == 40:
        dataset.features = dataset.features_40

    model = NNUE(input_size=input_size, hidden1=args.hidden1, hidden2=args.hidden2, hidden3=args.hidden3)

    # Load pretrained weights if specified (for fine-tuning)
    if args.init_weights:
        print(f"Loading initial weights for fine-tuning:")
        load_binary_weights(model, args.init_weights, input_size)

    model = model.to(device)
    params = sum(p.numel() for p in model.parameters())
    if args.hidden3 > 0:
        arch_str = f"{input_size} → {args.hidden1} → {args.hidden2} → {args.hidden3} → 1"
    else:
        arch_str = f"{input_size} → {args.hidden1} → {args.hidden2} → 1"
    print(f"Architecture: {arch_str}")
    print(f"Parameters: {params:,}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    lam = args.lam
    K_val = args.K
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for features, evals, results, weights, has_eval in train_loader:
            features = features.to(device)
            evals = evals.to(device)
            results = results.to(device)
            weights = weights.to(device)
            has_eval = has_eval.to(device)

            pred = model(features)
            pred_wp = sigmoid_eval(pred, K_val)
            effective_lam = lam * has_eval
            eval_wp = sigmoid_eval(evals, K_val)
            target = effective_lam * eval_wp + (1 - effective_lam) * results
            per_sample_loss = (pred_wp - target) ** 2
            loss = torch.mean(per_sample_loss * weights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for features, evals, results, weights, has_eval in val_loader:
                features = features.to(device)
                evals = evals.to(device)
                results = results.to(device)
                weights = weights.to(device)
                has_eval = has_eval.to(device)

                pred = model(features)
                pred_wp = sigmoid_eval(pred, K_val)
                effective_lam = lam * has_eval
                eval_wp = sigmoid_eval(evals, K_val)
                target = effective_lam * eval_wp + (1 - effective_lam) * results
                val_loss += torch.mean((pred_wp - target) ** 2 * weights).item()
                n_val_batches += 1

        avg_train = total_loss / n_batches
        avg_val = val_loss / n_val_batches

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), args.output_pt)
            marker = " *"
        else:
            marker = ""

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs}  train={avg_train:.6f}  val={avg_val:.6f}  lr={scheduler.get_last_lr()[0]:.6f}{marker}")

    model.load_state_dict(torch.load(args.output_pt, weights_only=True))
    export_binary(model, args.output_bin, args)

    print(f"\nTraining complete! Best val loss: {best_val_loss:.6f}")
    print(f"Weights: {args.output_bin}")


def export_binary(model, path, args):
    """Export quantized weights as binary for Rust.

    New format (input_size < 128): [input_size, hidden1, hidden2, hidden3] + layers
    Old format (hidden1 >= 128):   [hidden1, hidden2] + layers (backward compat)
    """
    SCALE = 64
    state = model.state_dict()

    with open(path, 'wb') as f:
        if args.hidden3 > 0 or args.input_size != 40:
            # New format: explicit input_size header (required for non-40 inputs or 4-layer)
            f.write(struct.pack('<HHHH', args.input_size, args.hidden1, args.hidden2, args.hidden3))
            if args.hidden3 > 0:
                layer_names = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias',
                               'fc3.weight', 'fc3.bias', 'fc4.weight', 'fc4.bias']
            else:
                layer_names = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias',
                               'fc3.weight', 'fc3.bias']
        else:
            # Old format: 3-layer 40-input network (h1 >= 128 serves as format marker)
            f.write(struct.pack('<HH', args.hidden1, args.hidden2))
            layer_names = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias',
                           'fc3.weight', 'fc3.bias']

        for name in layer_names:
            tensor = state[name].cpu().float().numpy().flatten()
            quantized = np.clip(tensor * SCALE, -32000, 32000).astype(np.int16)
            f.write(quantized.tobytes())

    total_bytes = os.path.getsize(path)
    print(f"Binary weights: {path} ({total_bytes:,} bytes)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train NNUE V2 for Togyz Kumalak')
    parser.add_argument('--data', required=True, help='Comma-separated training data files')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--input-size', type=int, default=40, choices=[40, 52, 58],
                        help='Input features: 40=legacy, 52=extended, 58=strategic')
    parser.add_argument('--hidden1', type=int, default=256)
    parser.add_argument('--hidden2', type=int, default=32)
    parser.add_argument('--hidden3', type=int, default=0,
                        help='Third hidden layer size (0=disabled, e.g. 64 for 4-layer net)')
    parser.add_argument('--K', type=float, default=400.0, help='Sigmoid scaling K parameter')
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--max-positions', type=int, default=None)
    parser.add_argument('--init-weights', default=None, help='Binary weights to fine-tune from')
    parser.add_argument('--output-pt', default='nnue_best.pt', help='PyTorch output')
    parser.add_argument('--output-bin', default='nnue_weights.bin', help='Binary output')
    args = parser.parse_args()

    train(args)
