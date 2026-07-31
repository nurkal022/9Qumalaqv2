#!/usr/bin/env python3
"""Train the NNUE v2 evaluation on the 9qum corpus and export it in the NNU2 format.

Input is sparse and binary (23 active features of 292), so the first layer is a sum of
selected columns — an EmbeddingBag with mode="sum". Four output heads are selected by phase
bucket, so endgame weights stop competing with opening weights (June's finding: pushing
endgame conversion cost general strength).

The net outputs a logit of "the side to move wins"; the engine multiplies it by 350 to get
centipawn-like units.

Run: python3.12 research/training/train_nnue_v2.py --epochs 12 --name v2_e12
"""
import argparse
import os
import struct
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
import features_v2 as fv

RECORD_SIZE = 68
MASK_VALUE_NET = 2
MASK_VALUE_OUTCOME = 4
NUM_FEATURES = fv.NUM_FEATURES
ACC = 1024
HIDDEN = 32
BUCKETS = fv.NUM_BUCKETS


class NnueV2(nn.Module):
    def __init__(self, num_features=NUM_FEATURES, acc=ACC, hidden=HIDDEN, buckets=BUCKETS):
        super().__init__()
        self.acc, self.hidden, self.buckets = acc, hidden, buckets
        self.emb = nn.EmbeddingBag(num_features, acc, mode="sum")
        self.acc_bias = nn.Parameter(torch.zeros(acc))
        self.fc2 = nn.ModuleList([nn.Linear(acc, hidden) for _ in range(buckets)])
        self.fc3 = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(buckets)])

    def forward(self, feats, offsets, bucket):
        a = torch.relu(self.emb(feats, offsets) + self.acc_bias)
        out = torch.zeros(a.shape[0], device=a.device)
        for b in range(self.buckets):
            m = bucket == b
            if m.any():
                out[m] = self.fc3[b](torch.relu(self.fc2[b](a[m]))).squeeze(-1)
        return out

    def forward_single(self, feature_indices, bucket):
        feats = torch.tensor(feature_indices, dtype=torch.long)
        offsets = torch.tensor([0], dtype=torch.long)
        return self.forward(feats, offsets, torch.tensor([bucket]))[0]


def export_nnu2(model, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<I", 0x324E554E))
        f.write(struct.pack("<6H", 2, NUM_FEATURES, model.acc, model.hidden, model.buckets, 0))
        w = model.emb.weight.detach().cpu().numpy().astype(np.float32)   # [features, acc]
        f.write(w.tobytes())                                              # feature-major
        f.write(model.acc_bias.detach().cpu().numpy().astype(np.float32).tobytes())
        for b in range(model.buckets):
            f.write(model.fc2[b].weight.detach().cpu().numpy().astype(np.float32).tobytes())
            f.write(model.fc2[b].bias.detach().cpu().numpy().astype(np.float32).tobytes())
            f.write(model.fc3[b].weight.detach().cpu().numpy().astype(np.float32).ravel().tobytes())
            f.write(model.fc3[b].bias.detach().cpu().numpy().astype(np.float32).tobytes())


def load_bin(path):
    raw = np.fromfile(path, dtype=np.uint8)
    n = len(raw) // RECORD_SIZE
    r = raw[: n * RECORD_SIZE].reshape(n, RECORD_SIZE)
    pits = r[:, 0:18].astype(np.int64)
    kazan = r[:, 18:20].astype(np.int64)
    tuz = r[:, 20:22].view(np.int8).astype(np.int64)
    stm = r[:, 22].astype(np.int64)
    value = r[:, 59:63].copy().view(np.float32).ravel()
    mask = r[:, 67]
    return pits, kazan, tuz, stm, value, mask


def build_feature_matrix(pits, kazan, tuz, stm):
    """Vectorised mirror of features_v2.build_features; test_features_v2.py owns correctness
    of the layout, this only has to agree with it."""
    n = pits.shape[0]
    feats = np.zeros((n, 23), dtype=np.int64)
    rows = np.stack([pits[:, 0:9], pits[:, 9:18]], axis=1)          # [n, 2, 9]
    me = stm
    opp = 1 - stm
    ar = np.arange(n)
    bucket_lut = np.array([min(c, 9) if c <= 9 else (10 if c <= 12 else (11 if c <= 16 else (12 if c <= 24 else 13)))
                           for c in range(163)], dtype=np.int64)
    for i in range(9):
        feats[:, i] = i * 14 + bucket_lut[rows[ar, me, i]]
        feats[:, 9 + i] = 126 + i * 14 + bucket_lut[rows[ar, opp, i]]
    feats[:, 18] = 252 + np.minimum(8, kazan[ar, me] // 10)
    feats[:, 19] = 261 + np.minimum(8, kazan[ar, opp] // 10)
    tz = np.where(tuz < 0, 9, tuz)
    feats[:, 20] = 270 + tz[ar, me]
    feats[:, 21] = 280 + tz[ar, opp]
    total = pits.sum(axis=1)
    feats[:, 22] = 290 + (total % 2)
    phase = np.where(total >= 121, 0, np.where(total >= 81, 1, np.where(total >= 41, 2, 3)))
    return feats, phase


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/9qum/train/train.bin")
    ap.add_argument("--val", default="data/9qum/train/val.bin")
    ap.add_argument("--out", default="models/nets/nnue_v2")
    ap.add_argument("--name", default="v2")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--w-net", type=float, default=1.0, help="weight of 9qum-labelled records")
    ap.add_argument("--w-outcome", type=float, default=0.3, help="weight of outcome-only records")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = NnueV2().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    lossf = nn.BCEWithLogitsLoss(reduction="none")

    def prep(path):
        pits, kazan, tuz, stm, value, mask = load_bin(path)
        feats, phase = build_feature_matrix(pits, kazan, tuz, stm)
        w = np.where(mask & MASK_VALUE_NET, a.w_net, a.w_outcome).astype(np.float32)
        return (torch.tensor(feats), torch.tensor(phase), torch.tensor(value),
                torch.tensor(w))

    tr = prep(a.train)
    va = prep(a.val)
    print(f"train {tr[0].shape[0]:,} records, val {va[0].shape[0]:,}")

    def run_epoch(data, train):
        feats, phase, value, w = data
        n = feats.shape[0]
        order = torch.randperm(n) if train else torch.arange(n)
        tot = cnt = 0.0
        model.train(train)
        for s in range(0, n, a.batch):
            idx = order[s: s + a.batch]
            fb = feats[idx].to(dev)
            offs = torch.arange(0, fb.shape[0] * 23, 23, device=dev)
            out = model(fb.reshape(-1), offs, phase[idx].to(dev))
            l = lossf(out, value[idx].to(dev)) * w[idx].to(dev)
            l = l.mean()
            if train:
                opt.zero_grad()
                l.backward()
                opt.step()
            tot += l.item() * idx.numel()
            cnt += idx.numel()
        return tot / cnt

    os.makedirs(a.out, exist_ok=True)
    best = float("inf")
    for ep in range(1, a.epochs + 1):
        trl = run_epoch(tr, True)
        with torch.no_grad():
            val = run_epoch(va, False)
        print(f"epoch {ep:>3}  train {trl:.4f}  val {val:.4f}")
        if val < best:
            best = val
            torch.save(model.state_dict(), os.path.join(a.out, f"{a.name}.pt"))
            export_nnu2(model, os.path.join(a.out, f"{a.name}.bin"))
            print(f"  saved {a.name}.bin (val {val:.4f})")
    print(f"best val loss {best:.4f}")


if __name__ == "__main__":
    main()
