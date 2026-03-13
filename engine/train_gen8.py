#!/usr/bin/env python3
"""Gen8: Fine-tune Gen5 champion on depth-16 selfplay data (1.8M positions).
Also try combined d10+d14+d16 data.
3 seeds, 30 epochs, lr=0.0001, K=400, lam=0.5
"""
import sys, os, struct, logging, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import DS, sig

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.info

BASE = os.path.dirname(os.path.abspath(__file__))

class NNUE(nn.Module):
    def __init__(self, h1=256, h2=32):
        super().__init__()
        self.fc1 = nn.Linear(40, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 1)
    def forward(self, x):
        x = torch.clamp(self.fc1(x), 0, 1)
        x = torch.clamp(self.fc2(x), 0, 1)
        return self.fc3(x).squeeze(-1)

def load_weights(model, path):
    with open(path, 'rb') as f:
        data = f.read()
    first = struct.unpack_from('<H', data, 0)[0]
    if first >= 128:
        h1, h2 = first, struct.unpack_from('<H', data, 2)[0]
        offset = 4
    else:
        raise ValueError("Expected old format")
    SCALE = 64.0
    def rd(cnt):
        nonlocal offset
        v = np.frombuffer(data[offset:offset+cnt*2], dtype=np.int16).astype(np.float32) / SCALE
        offset += cnt * 2
        return v
    sd = model.state_dict()
    sd['fc1.weight'] = torch.tensor(rd(h1*40).reshape(h1,40))
    sd['fc1.bias'] = torch.tensor(rd(h1))
    sd['fc2.weight'] = torch.tensor(rd(h2*h1).reshape(h2,h1))
    sd['fc2.bias'] = torch.tensor(rd(h2))
    sd['fc3.weight'] = torch.tensor(rd(h2).reshape(1,h2))
    sd['fc3.bias'] = torch.tensor(rd(1))
    model.load_state_dict(sd)
    log(f"  Loaded weights from {os.path.basename(path)}")

def export_bin(model, path):
    SCALE = 64
    sd = model.state_dict()
    with open(path, 'wb') as f:
        f.write(struct.pack('<HH', 256, 32))
        for name in ['fc1.weight','fc1.bias','fc2.weight','fc2.bias','fc3.weight','fc3.bias']:
            t = sd[name].cpu().float().numpy().flatten()
            f.write(np.clip(t * SCALE, -32000, 32000).astype(np.int16).tobytes())
    log(f"  Exported: {os.path.basename(path)} ({os.path.getsize(path):,} bytes)")

def train_variant(name, data_files, champion, seeds, epochs=30, lr=0.0001, K=400, lam=0.5):
    log(f"\n=== {name} ===")
    dataset = DS(data_files, input_size=40)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = NNUE().to(dev)
        load_weights(model, champion)

        n = len(dataset)
        nv = min(n // 10, 50000)
        tr, vl = torch.utils.data.random_split(dataset, [n-nv, nv],
            generator=torch.Generator().manual_seed(seed))
        tl = DataLoader(tr, 4096, shuffle=True, num_workers=4, pin_memory=True)
        vl_loader = DataLoader(vl, 8192, num_workers=4, pin_memory=True)

        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        best_v = float('inf')
        best_sd = None
        for ep in range(epochs):
            model.train()
            for feat, ev, res, wt, he in tl:
                feat, ev, res, wt, he = feat.to(dev), ev.to(dev), res.to(dev), wt.to(dev), he.to(dev)
                lam_eff = lam * he
                tgt = lam_eff * sig(ev, K) + (1 - lam_eff) * res
                loss = torch.mean((sig(model(feat), K) - tgt) ** 2 * wt)
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()

            model.eval()
            vl_ = 0.0; nb = 0
            with torch.no_grad():
                for feat, ev, res, wt, he in vl_loader:
                    feat, ev, res, wt, he = feat.to(dev), ev.to(dev), res.to(dev), wt.to(dev), he.to(dev)
                    lam_eff = lam * he
                    tgt = lam_eff * sig(ev, K) + (1 - lam_eff) * res
                    vl_ += torch.mean((sig(model(feat), K) - tgt) ** 2 * wt).item()
                    nb += 1
            av = vl_ / nb
            if av < best_v:
                best_v = av
                best_sd = {k: v.clone() for k, v in model.state_dict().items()}
                m = '*'
            else:
                m = ''
            if (ep + 1) % 5 == 0 or ep == 0:
                log(f"  s{seed} ep {ep+1:3d}/{epochs}  val={av:.6f} {m}")

        model.load_state_dict(best_sd)
        out_bin = os.path.join(BASE, f'nnue_weights_{name}_s{seed}.bin')
        export_bin(model, out_bin)
        results[seed] = (out_bin, best_v)
        log(f"  Seed {seed}: best_val={best_v:.6f}")

    return results

def run_matches(results, champion, name):
    log(f"\n=== Matching {name} vs Gen5 ===")
    import subprocess
    ENGINE = os.path.join(BASE, 'target', 'release', 'togyzkumalaq-engine')
    for seed, (out_bin, val) in results.items():
        r = subprocess.run([ENGINE, 'match-nnue', out_bin, champion, '200', '500'],
            capture_output=True, text=True, cwd=BASE, timeout=7200)
        out = r.stdout + r.stderr
        for line in out.split('\n'):
            if 'A score' in line or 'Final' in line or 'Elo' in line or 'Game' in line:
                if 'Final' in line or 'Elo' in line or 'score' in line:
                    log(f"  s{seed}: {line.strip()}")

def main():
    champion = os.path.join(BASE, 'nnue_weights.bin')
    d16_file = os.path.join(BASE, 'gen8_d16_training_data.bin')
    d14_file = os.path.join(BASE, 'gen7_d14_training_data.bin')
    d10_file = os.path.join(BASE, 'gen_combined_d10.bin')

    seeds = [42, 123, 777]

    # Variant 1: Pure D16 fine-tune (best data quality)
    res_d16 = train_variant('gen8_d16', [d16_file], champion, seeds)
    run_matches(res_d16, champion, 'gen8_d16')

    # Variant 2: Combined D10+D16 (more data, mixed quality)
    combined = [d16_file]
    if os.path.exists(d10_file):
        combined.append(d10_file)
    res_combined = train_variant('gen8_combined', combined, champion, seeds[:2])
    run_matches(res_combined, champion, 'gen8_combined')

    # Variant 3: Combined D14+D16 (both high depth)
    deep_combined = [d16_file]
    if os.path.exists(d14_file):
        deep_combined.append(d14_file)
    res_deep = train_variant('gen8_deep', deep_combined, champion, seeds[:2])
    run_matches(res_deep, champion, 'gen8_deep')

    log("\n=== ALL DONE ===")

if __name__ == '__main__':
    main()
