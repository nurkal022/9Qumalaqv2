"""Train NNUE with enhanced endgame weighting.

Loads multiple data files (regular + endgame) and applies higher weights
to endgame positions to improve endgame play quality.

Usage: python train_endgame.py <K> <lambda> <output.bin> <data1.bin> [data2.bin ...]
"""
import struct, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os, sys

RECORD_SIZE = 26
INPUT_SIZE = 40

class EndgameDataset(Dataset):
    def __init__(self, paths, max_positions=None):
        if isinstance(paths, str):
            paths = [paths]
        all_data = []
        for path in paths:
            if not os.path.exists(path):
                print(f"  Warning: {path} not found, skipping")
                continue
            file_size = os.path.getsize(path)
            num_records = file_size // RECORD_SIZE
            with open(path, 'rb') as f:
                raw = f.read(num_records * RECORD_SIZE)
            data = np.frombuffer(raw, dtype=np.uint8).reshape(num_records, RECORD_SIZE)
            all_data.append(data)
            print(f"  Loaded {path}: {num_records:,} positions")
        data = np.concatenate(all_data, axis=0)
        num_records = len(data)
        if max_positions and num_records > max_positions:
            idx = np.random.choice(num_records, max_positions, replace=False)
            data = data[idx]
            num_records = max_positions
        print(f"  Total: {num_records:,}")

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

        # Enhanced endgame weighting
        pit_stones = pits_w.sum(axis=1) + pits_b.sum(axis=1)
        self.sample_weights = np.ones(num_records, dtype=np.float32)
        self.sample_weights[pit_stones <= 50] = 2.0    # light endgame
        self.sample_weights[pit_stones <= 30] = 4.0    # mid endgame
        self.sample_weights[pit_stones <= 15] = 8.0    # deep endgame
        self.sample_weights[pit_stones <= 5] = 12.0    # near-tablebase endgame

        # Stats
        n_total = num_records
        n_50 = int((pit_stones <= 50).sum())
        n_30 = int((pit_stones <= 30).sum())
        n_15 = int((pit_stones <= 15).sum())
        n_5 = int((pit_stones <= 5).sum())
        print(f"  Stone distribution: <=50: {n_50:,} ({100*n_50/n_total:.1f}%), "
              f"<=30: {n_30:,} ({100*n_30/n_total:.1f}%), "
              f"<=15: {n_15:,} ({100*n_15/n_total:.1f}%), "
              f"<=5: {n_5:,} ({100*n_5/n_total:.1f}%)")

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

if len(sys.argv) < 5:
    print("Usage: python train_endgame.py <K> <lambda> <output.bin> <data1.bin> [data2.bin ...] [--h1 N] [--h2 N] [--epochs N]")
    sys.exit(1)

# Parse optional flags from args
all_args = sys.argv[1:]
h1, h2, epochs = 256, 32, 100
data_files = []
i = 0
K = float(all_args[0]); lam = float(all_args[1]); output_bin = all_args[2]
i = 3
while i < len(all_args):
    if all_args[i] == '--h1': h1 = int(all_args[i+1]); i += 2
    elif all_args[i] == '--h2': h2 = int(all_args[i+1]); i += 2
    elif all_args[i] == '--epochs': epochs = int(all_args[i+1]); i += 2
    else: data_files.append(all_args[i]); i += 1
print(f"K={K}, lambda={lam}, output={output_bin}, arch={h1}→{h2}→1, epochs={epochs}")
print(f"Data files: {data_files}")

def sigmoid_eval(x, k=K):
    return torch.sigmoid(x / (k / 4.0))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

dataset = EndgameDataset(data_files)
n = len(dataset)
n_val = min(n // 10, 50000)
train_set, val_set = torch.utils.data.random_split(dataset, [n - n_val, n_val])
train_loader = DataLoader(train_set, batch_size=4096, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=8192, num_workers=4, pin_memory=True)

model = NNUE(hidden1=h1, hidden2=h2).to(device)
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
best_val = float('inf')
best_pt = output_bin.replace('.bin', '.pt')

for epoch in range(epochs):
    model.train()
    tl = 0; nb = 0
    for features, evals, results, weights, has_eval in train_loader:
        features, evals, results, weights, has_eval = [x.to(device) for x in [features, evals, results, weights, has_eval]]
        pred = model(features)
        pred_wp = sigmoid_eval(pred)
        eff_lam = lam * has_eval
        eval_wp = sigmoid_eval(evals)
        target = eff_lam * eval_wp + (1 - eff_lam) * results
        loss = torch.mean((pred_wp - target) ** 2 * weights)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        tl += loss.item(); nb += 1
    scheduler.step()
    model.eval()
    vl = 0; nvb = 0
    with torch.no_grad():
        for features, evals, results, weights, has_eval in val_loader:
            features, evals, results, weights, has_eval = [x.to(device) for x in [features, evals, results, weights, has_eval]]
            pred = model(features)
            pred_wp = sigmoid_eval(pred)
            eff_lam = lam * has_eval
            eval_wp = sigmoid_eval(evals)
            target = eff_lam * eval_wp + (1 - eff_lam) * results
            vl += torch.mean((pred_wp - target) ** 2 * weights).item(); nvb += 1
    avg_val = vl / nvb
    if avg_val < best_val:
        best_val = avg_val
        torch.save(model.state_dict(), best_pt)
        m = " *"
    else:
        m = ""
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}/{epochs}  val={avg_val:.6f}{m}")

model.load_state_dict(torch.load(best_pt, weights_only=True))
SCALE = 64
state = model.state_dict()
with open(output_bin, 'wb') as f:
    f.write(struct.pack('<HH', h1, h2))
    for name in ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias']:
        tensor = state[name].cpu().float().numpy().flatten()
        quantized = np.clip(tensor * SCALE, -32000, 32000).astype(np.int16)
        f.write(quantized.tobytes())
print(f"\n{output_bin}: {os.path.getsize(output_bin):,} bytes, val_loss={best_val:.6f}")
