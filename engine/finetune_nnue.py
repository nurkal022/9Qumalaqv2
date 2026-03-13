"""Fine-tune NNUE from existing weights with lower learning rate"""
import struct
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os

RECORD_SIZE = 26
INPUT_SIZE = 40
K = 400.0

class PositionDataset(Dataset):
    def __init__(self, paths, max_positions=None):
        if isinstance(paths, str):
            paths = [paths]
        all_data = []
        for path in paths:
            if not os.path.exists(path):
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
        self.features = np.concatenate([my_pits, opp_pits, my_kazan, opp_kazan, my_tuz, opp_tuz], axis=1)
        self.eval_targets = evals
        result_stm = np.where(is_white, results, 1.0 - results)
        self.result_targets = result_stm
        pit_stones = pits_w.sum(axis=1) + pits_b.sum(axis=1)
        self.sample_weights = np.ones(num_records, dtype=np.float32)
        self.sample_weights[pit_stones <= 30] = 2.0
        self.sample_weights[pit_stones <= 15] = 3.0
        self.has_eval = (np.abs(evals) > 1).astype(np.float32)

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

def load_binary_weights(model, bin_path):
    """Load quantized binary weights into PyTorch model"""
    SCALE = 64
    with open(bin_path, 'rb') as f:
        data = f.read()
    
    hidden1 = struct.unpack('<H', data[0:2])[0]
    hidden2 = struct.unpack('<H', data[2:4])[0]
    offset = 4
    
    def read_i16_array(n):
        nonlocal offset
        arr = np.frombuffer(data[offset:offset+n*2], dtype=np.int16).astype(np.float32) / SCALE
        offset += n * 2
        return arr
    
    state = model.state_dict()
    fc1_w = read_i16_array(hidden1 * INPUT_SIZE).reshape(hidden1, INPUT_SIZE)
    state['fc1.weight'] = torch.tensor(fc1_w)
    fc1_b = read_i16_array(hidden1)
    state['fc1.bias'] = torch.tensor(fc1_b)
    fc2_w = read_i16_array(hidden2 * hidden1).reshape(hidden2, hidden1)
    state['fc2.weight'] = torch.tensor(fc2_w)
    fc2_b = read_i16_array(hidden2)
    state['fc2.bias'] = torch.tensor(fc2_b)
    fc3_w = read_i16_array(hidden2).reshape(1, hidden2)
    state['fc3.weight'] = torch.tensor(fc3_w)
    fc3_b = read_i16_array(1)
    state['fc3.bias'] = torch.tensor(fc3_b)
    
    model.load_state_dict(state)
    print(f"Loaded binary weights from {bin_path}: {hidden1}→{hidden2}→1")
    return model

def export_binary(model, path, hidden1, hidden2):
    SCALE = 64
    state = model.state_dict()
    with open(path, 'wb') as f:
        f.write(struct.pack('<HH', hidden1, hidden2))
        for name in ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias']:
            tensor = state[name].cpu().float().numpy().flatten()
            quantized = np.clip(tensor * SCALE, -32000, 32000).astype(np.int16)
            f.write(quantized.tobytes())
    print(f"Binary weights: {path} ({os.path.getsize(path):,} bytes)")

def finetune():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--init-weights', required=True, help='Binary weights to start from')
    parser.add_argument('--data', required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--hidden1', type=int, default=256)
    parser.add_argument('--hidden2', type=int, default=32)
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--output-bin', default='nnue_finetuned.bin')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Fine-tuning from: {args.init_weights}")
    print(f"Lambda: {args.lam}, LR: {args.lr}, Epochs: {args.epochs}\n")

    data_files = args.data.split(',')
    dataset = PositionDataset(data_files)
    n = len(dataset)
    n_val = min(n // 10, 50000)
    n_train = n - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size * 2, num_workers=4, pin_memory=True)

    model = NNUE(hidden1=args.hidden1, hidden2=args.hidden2).to('cpu')
    load_binary_weights(model, args.init_weights)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    lam = args.lam
    best_val_loss = float('inf')
    best_pt = args.output_bin.replace('.bin', '.pt')

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
            pred_wp = sigmoid_eval(pred)
            effective_lam = lam * has_eval
            eval_wp = sigmoid_eval(evals)
            target = effective_lam * eval_wp + (1 - effective_lam) * results
            loss = torch.mean((pred_wp - target) ** 2 * weights)
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
                pred_wp = sigmoid_eval(pred)
                effective_lam = lam * has_eval
                eval_wp = sigmoid_eval(evals)
                target = effective_lam * eval_wp + (1 - effective_lam) * results
                val_loss += torch.mean((pred_wp - target) ** 2 * weights).item()
                n_val_batches += 1

        avg_train = total_loss / n_batches
        avg_val = val_loss / n_val_batches
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), best_pt)
            marker = " *"
        else:
            marker = ""
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs}  train={avg_train:.6f}  val={avg_val:.6f}  lr={scheduler.get_last_lr()[0]:.6f}{marker}")

    model.load_state_dict(torch.load(best_pt, weights_only=True))
    export_binary(model, args.output_bin, args.hidden1, args.hidden2)
    print(f"\nFine-tuning complete! Best val loss: {best_val_loss:.6f}")

if __name__ == '__main__':
    finetune()
