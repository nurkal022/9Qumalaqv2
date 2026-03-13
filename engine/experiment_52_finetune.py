#!/usr/bin/env python3
"""Fine-tune 52-input NNUE from Gen5 weights (transfer learning).
Architecture: 52→256→32→1 (same hidden layers as Gen5, just 12 more inputs).
Gen5 fc1 weights (40 cols) are copied, 12 new input columns initialized near zero.
"""
import sys, os, struct, logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import DS, sig

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.info

BASE = os.path.dirname(os.path.abspath(__file__))

class NNUE52(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(52, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.clamp(self.fc1(x), 0, 1)
        x = torch.clamp(self.fc2(x), 0, 1)
        return self.fc3(x).squeeze(-1)


def load_gen5_into_52(model, gen5_path):
    """Load Gen5 weights (40→256→32→1) into 52-input model."""
    with open(gen5_path, 'rb') as f:
        data = f.read()
    first = struct.unpack_from('<H', data, 0)[0]
    if first >= 128:
        h1, h2 = first, struct.unpack_from('<H', data, 2)[0]
        offset = 4
    else:
        raise ValueError("Expected old format")
    in_sz = 40
    SCALE = 64.0
    def rd(cnt):
        nonlocal offset
        v = np.frombuffer(data[offset:offset+cnt*2], dtype=np.int16).astype(np.float32) / SCALE
        offset += cnt * 2
        return v

    fc1_w = rd(h1 * in_sz).reshape(h1, in_sz)  # [256, 40]
    fc1_b = rd(h1)
    fc2_w = rd(h2 * h1).reshape(h2, h1)  # [32, 256]
    fc2_b = rd(h2)
    fc3_w = rd(h2).reshape(1, h2)  # [1, 32]
    fc3_b = rd(1)

    sd = model.state_dict()
    # fc1: copy first 40 columns, initialize 12 new columns near zero
    new_fc1_w = torch.zeros(256, 52)
    new_fc1_w[:, :40] = torch.tensor(fc1_w)
    new_fc1_w[:, 40:] = torch.randn(256, 12) * 0.01  # tiny random init
    sd['fc1.weight'] = new_fc1_w
    sd['fc1.bias'] = torch.tensor(fc1_b)
    # fc2, fc3: copy exactly
    sd['fc2.weight'] = torch.tensor(fc2_w)
    sd['fc2.bias'] = torch.tensor(fc2_b)
    sd['fc3.weight'] = torch.tensor(fc3_w)
    sd['fc3.bias'] = torch.tensor(fc3_b)
    model.load_state_dict(sd)
    log(f"Loaded Gen5 weights into 52-input model (40 cols copied, 12 new)")


def export_52_bin(model, path):
    """Export 52→256→32→1 as new format [52, 256, 32, 0]."""
    SCALE = 64
    sd = model.state_dict()
    with open(path, 'wb') as f:
        f.write(struct.pack('<HHHH', 52, 256, 32, 0))
        for name in ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias']:
            t = sd[name].cpu().float().numpy().flatten()
            f.write(np.clip(t * SCALE, -32000, 32000).astype(np.int16).tobytes())
    log(f"Exported: {os.path.basename(path)} ({os.path.getsize(path):,} bytes)")


def main():
    champion = os.path.join(BASE, 'nnue_weights.bin')
    d14_file = os.path.join(BASE, 'gen7_d14_training_data.bin')
    d10_file = os.path.join(BASE, 'gen_combined_d10.bin')

    data_files = [d14_file]
    if os.path.exists(d10_file):
        data_files.append(d10_file)

    log("=== 52-input Fine-tune from Gen5 ===")
    dataset = DS(data_files, input_size=52)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    K = 400.0
    lam = 0.5

    seeds = [42, 123]
    results = {}
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = NNUE52().to(dev)
        load_gen5_into_52(model, champion)

        n = len(dataset)
        nv = min(n // 10, 50000)
        tr, vl = torch.utils.data.random_split(dataset, [n-nv, nv],
            generator=torch.Generator().manual_seed(seed))
        tl = DataLoader(tr, 4096, shuffle=True, num_workers=4, pin_memory=True)
        vl_loader = DataLoader(vl, 8192, num_workers=4, pin_memory=True)

        # Two-phase LR: higher for new features (fc1 cols 40-51), lower for rest
        # Actually simpler: just use low lr since most weights are already good
        opt = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)

        best_v = float('inf')
        best_sd = None

        for ep in range(40):
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
            vl_ = 0.0
            nb = 0
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
                log(f"  s{seed} ep {ep+1:3d}/40  val={av:.6f} {m}")

        model.load_state_dict(best_sd)
        out_bin = os.path.join(BASE, f'nnue_weights_52ft_s{seed}.bin')
        export_52_bin(model, out_bin)
        results[seed] = (out_bin, best_v)
        log(f"  Seed {seed}: best_val={best_v:.6f}")

    # Match best vs champion
    log("\n=== Matching vs Gen5 ===")
    import subprocess
    ENGINE = os.path.join(BASE, 'target', 'release', 'togyzkumalaq-engine')
    for seed in seeds:
        out_bin = results[seed][0]
        r = subprocess.run([ENGINE, 'match-nnue', out_bin, champion, '200', '500'],
            capture_output=True, text=True, cwd=BASE)
        out = r.stdout + r.stderr
        for line in out.split('\n'):
            if 'A score:' in line or 'Final:' in line or 'Elo' in line:
                log(f"  s{seed}: {line.strip()}")

    log("Done!")


if __name__ == '__main__':
    main()
