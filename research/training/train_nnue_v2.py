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
MASK_SCORE = 8  # bit 3: the record's `score` field (final kazan diff / 82) is trustworthy
NUM_FEATURES = fv.NUM_FEATURES
ACC = 1024
HIDDEN = 32
BUCKETS = fv.NUM_BUCKETS


class NnueV2(nn.Module):
    def __init__(self, num_features=NUM_FEATURES, acc=ACC, hidden=HIDDEN, buckets=BUCKETS,
                 score_head=False):
        super().__init__()
        self.acc, self.hidden, self.buckets = acc, hidden, buckets
        self.score_head = score_head
        self.emb = nn.EmbeddingBag(num_features, acc, mode="sum")
        self.acc_bias = nn.Parameter(torch.zeros(acc))
        self.fc2 = nn.ModuleList([nn.Linear(acc, hidden) for _ in range(buckets)])
        self.fc3 = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(buckets)])
        if score_head:
            # Auxiliary only (task 12, lever 1): predicts the record's normalised final
            # kazan-difference `score`, trained jointly with the value head so the shared
            # emb/fc2 representation also has to explain the exact margin, not just win/loss.
            # NEVER read by export_nnu2/export_nnu2_v3 (they only ever serialise
            # emb/acc_bias/fc2/fc3) and therefore never reaches the engine's NNU2 loader.
            self.fc_score = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(buckets)])

    def forward(self, feats, offsets, bucket, return_score=False):
        a = torch.relu(self.emb(feats, offsets) + self.acc_bias)
        out = torch.zeros(a.shape[0], device=a.device)
        score_out = torch.zeros(a.shape[0], device=a.device) if return_score else None
        for b in range(self.buckets):
            m = bucket == b
            if m.any():
                h = torch.relu(self.fc2[b](a[m]))
                out[m] = self.fc3[b](h).squeeze(-1)
                if return_score:
                    score_out[m] = self.fc_score[b](h).squeeze(-1)
        if return_score:
            return out, score_out
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


def _pick_scale(*arrays):
    """floor(32767 / absmax * 0.999): the largest per-tensor integer scale that keeps
    every quantised value inside int16 range with a small margin (so a rounded value
    can never land exactly on the +-32768 boundary). Falls back to 1.0 for an
    all-zero tensor. Mirrors NnueNetwork::pick_scale in engine/src/nnue.rs — the two
    must stay identical, see task-9-report.md for why this scale was chosen (per-tensor
    absmax: real weight magnitudes differ ~7x bucket to bucket, so each bucket's
    fc2/fc3 get their own scale rather than sharing one sized for the largest)."""
    absmax = max(float(np.abs(a).max()) for a in arrays)
    if absmax <= 0:
        return 1.0
    return float(np.floor(32767.0 / absmax * 0.999))


def _quantise(arr, scale):
    q = np.round(arr.astype(np.float64) * scale)
    if np.abs(q).max() > 32767:
        raise ValueError(f"quantisation overflow at scale {scale}: max |q|={np.abs(q).max()}")
    return q.astype(np.int16)


def export_nnu2_v3(model, path):
    """Version-3 NNU2 writer: same header as export_nnu2 (magic, num_features, acc,
    hidden, buckets, pad) but version=3, followed by a scale-factor section, then i16
    weights instead of f32 — for the integer forward pass in
    engine/src/nnue.rs::NnueNetwork::logit_v3 (NnueNetwork::load_nnu2_i16 reads this
    exact layout; keep the two in lockstep).

    Scale factors (see _pick_scale): fc1_w and acc_bias share one scale (`scale_l1`)
    because they are summed directly into the same accumulator; each bucket's
    fc2 weight+bias share a scale, and each bucket's fc3 weight+bias share a
    (different) scale, because this net's weight magnitudes vary a lot bucket to
    bucket (measured fc2 absmax ~0.048-0.34 across the 4 buckets) — sharing one
    scale across buckets would waste most of int16's range on the smaller ones.

    Byte layout:
      u32 magic, u16 version(=3), u16 num_features, u16 acc, u16 hidden, u16 buckets, u16 pad
      f32 scale_l1
      per bucket: f32 scale_fc2[b], f32 scale_fc3[b]
      i16 fc1_w[num_features*acc]  (feature-major, same order as export_nnu2)
      i16 fc1_b[acc]
      per bucket: i16 fc2_w[acc*hidden], i16 fc2_b[hidden], i16 fc3_w[hidden], i16 fc3_b[1]
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    emb = model.emb.weight.detach().cpu().numpy().astype(np.float32)
    bias = model.acc_bias.detach().cpu().numpy().astype(np.float32)
    scale_l1 = _pick_scale(emb, bias)

    fc2w = [model.fc2[b].weight.detach().cpu().numpy().astype(np.float32) for b in range(model.buckets)]
    fc2b = [model.fc2[b].bias.detach().cpu().numpy().astype(np.float32) for b in range(model.buckets)]
    fc3w = [model.fc3[b].weight.detach().cpu().numpy().astype(np.float32).ravel() for b in range(model.buckets)]
    fc3b = [model.fc3[b].bias.detach().cpu().numpy().astype(np.float32) for b in range(model.buckets)]
    scale_fc2 = [_pick_scale(fc2w[b], fc2b[b]) for b in range(model.buckets)]
    scale_fc3 = [_pick_scale(fc3w[b], fc3b[b]) for b in range(model.buckets)]

    with open(path, "wb") as f:
        f.write(struct.pack("<I", 0x324E554E))
        f.write(struct.pack("<6H", 3, NUM_FEATURES, model.acc, model.hidden, model.buckets, 0))
        f.write(struct.pack("<f", scale_l1))
        for b in range(model.buckets):
            f.write(struct.pack("<ff", scale_fc2[b], scale_fc3[b]))
        f.write(_quantise(emb, scale_l1).tobytes())
        f.write(_quantise(bias, scale_l1).tobytes())
        for b in range(model.buckets):
            f.write(_quantise(fc2w[b], scale_fc2[b]).tobytes())
            f.write(_quantise(fc2b[b], scale_fc2[b]).tobytes())
            f.write(_quantise(fc3w[b], scale_fc3[b]).tobytes())
            f.write(_quantise(fc3b[b], scale_fc3[b]).tobytes())
    return {"scale_l1": scale_l1, "scale_fc2": scale_fc2, "scale_fc3": scale_fc3}


def export_nnu2_v4(model, path):
    """Version-4 NNU2 writer (task 13): identical layout to export_nnu2_v3 (i16,
    per-tensor f32 scales) through the value head, plus an auxiliary score head
    appended to the scale section and to each bucket's weight block, so the engine can
    blend a win-probability term with a predicted-final-margin term
    (engine/src/nnue.rs::NnueNetwork::evaluate, version-4 branch). Requires
    `model.score_head` (the `fc_score` ModuleList built by `NnueV2(..., score_head=True)`
    / trained with `--w-score > 0`).

    Byte layout (must mirror engine/src/nnue.rs::NnueNetwork::load_nnu2_v4 field-for-field):
      u32 magic, u16 version(=4), u16 num_features, u16 acc, u16 hidden, u16 buckets, u16 pad
      f32 scale_l1
      per bucket: f32 scale_fc2[b], f32 scale_fc3[b], f32 scale_score[b]
      i16 fc1_w[num_features*acc]  (feature-major, same order as export_nnu2/_v3)
      i16 fc1_b[acc]
      per bucket: i16 fc2_w[acc*hidden], i16 fc2_b[hidden], i16 fc3_w[hidden], i16 fc3_b[1],
                  i16 score_w[hidden], i16 score_b[1]
    """
    if not getattr(model, "score_head", False):
        raise ValueError(
            "export_nnu2_v4 requires a model built with score_head=True and trained "
            "with --w-score > 0 (no fc_score submodule found)"
        )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    emb = model.emb.weight.detach().cpu().numpy().astype(np.float32)
    bias = model.acc_bias.detach().cpu().numpy().astype(np.float32)
    scale_l1 = _pick_scale(emb, bias)

    fc2w = [model.fc2[b].weight.detach().cpu().numpy().astype(np.float32) for b in range(model.buckets)]
    fc2b = [model.fc2[b].bias.detach().cpu().numpy().astype(np.float32) for b in range(model.buckets)]
    fc3w = [model.fc3[b].weight.detach().cpu().numpy().astype(np.float32).ravel() for b in range(model.buckets)]
    fc3b = [model.fc3[b].bias.detach().cpu().numpy().astype(np.float32) for b in range(model.buckets)]
    scorew = [model.fc_score[b].weight.detach().cpu().numpy().astype(np.float32).ravel() for b in range(model.buckets)]
    scoreb = [model.fc_score[b].bias.detach().cpu().numpy().astype(np.float32) for b in range(model.buckets)]

    scale_fc2 = [_pick_scale(fc2w[b], fc2b[b]) for b in range(model.buckets)]
    scale_fc3 = [_pick_scale(fc3w[b], fc3b[b]) for b in range(model.buckets)]
    scale_score = [_pick_scale(scorew[b], scoreb[b]) for b in range(model.buckets)]

    with open(path, "wb") as f:
        f.write(struct.pack("<I", 0x324E554E))
        f.write(struct.pack("<6H", 4, NUM_FEATURES, model.acc, model.hidden, model.buckets, 0))
        f.write(struct.pack("<f", scale_l1))
        for b in range(model.buckets):
            f.write(struct.pack("<fff", scale_fc2[b], scale_fc3[b], scale_score[b]))
        f.write(_quantise(emb, scale_l1).tobytes())
        f.write(_quantise(bias, scale_l1).tobytes())
        for b in range(model.buckets):
            f.write(_quantise(fc2w[b], scale_fc2[b]).tobytes())
            f.write(_quantise(fc2b[b], scale_fc2[b]).tobytes())
            f.write(_quantise(fc3w[b], scale_fc3[b]).tobytes())
            f.write(_quantise(fc3b[b], scale_fc3[b]).tobytes())
            f.write(_quantise(scorew[b], scale_score[b]).tobytes())
            f.write(_quantise(scoreb[b], scale_score[b]).tobytes())
    return {
        "scale_l1": scale_l1, "scale_fc2": scale_fc2, "scale_fc3": scale_fc3,
        "scale_score": scale_score,
    }


def load_bin(path):
    raw = np.fromfile(path, dtype=np.uint8)
    n = len(raw) // RECORD_SIZE
    r = raw[: n * RECORD_SIZE].reshape(n, RECORD_SIZE)
    pits = r[:, 0:18].astype(np.int64)
    kazan = r[:, 18:20].astype(np.int64)
    tuz = r[:, 20:22].view(np.int8).astype(np.int64)
    stm = r[:, 22].astype(np.int64)
    value = r[:, 59:63].copy().view(np.float32).ravel()
    score = r[:, 63:67].copy().view(np.float32).ravel()
    mask = r[:, 67]
    return pits, kazan, tuz, stm, value, score, mask


def phase_from_total(total, buckets=BUCKETS):
    """Descending phase-bucket assignment from the total stones still on the board
    (0..162: 18 pits x 9 stones at the opening, both kazans start at 0).

    buckets == BUCKETS (4) is the canonical, EXPORTED, playable mapping -- thresholds
    121/81/41, bit-identical to features_v2.phase_bucket() and to
    engine/src/nnue.rs::phase_bucket(); test_nnue_v2_export.py checks this against the
    scalar Python implementation.

    Any other value is a TRAINING-ONLY refinement (task 12, lever 2): each of the 4
    canonical ranges is split into buckets/BUCKETS equal-width sub-ranges by total stone
    count (more stones -> lower sub-index), so e.g. buckets=8 doubles resolution inside
    each existing range without moving its outer edges. This mapping must NEVER be
    exported as an NNU2 file: engine/src/nnue.rs::phase_bucket() only ever returns 0..3
    (`.min(self.buckets - 1)` at the call site does not change this), so a >4-bucket
    file would have most of its extra heads unreachable at play time, and the reachable
    ones would disagree with the boundaries used here. See the --buckets help text.
    """
    if buckets == BUCKETS:
        return np.where(total >= 121, 0, np.where(total >= 81, 1, np.where(total >= 41, 2, 3)))
    if buckets < BUCKETS or buckets % BUCKETS != 0:
        raise ValueError(f"--buckets {buckets} must be a positive multiple of {BUCKETS} "
                          f"(each canonical range is split into buckets/{BUCKETS} sub-ranges)")
    sub = buckets // BUCKETS
    parent = np.where(total >= 121, 0, np.where(total >= 81, 1, np.where(total >= 41, 2, 3)))
    lo = np.select([parent == 0, parent == 1, parent == 2, parent == 3], [121, 81, 41, 0])
    hi = np.select([parent == 0, parent == 1, parent == 2, parent == 3], [163, 121, 81, 41])
    width = np.maximum((hi - lo) / sub, 1e-9)
    sidx = np.clip(np.floor((hi - 1 - total) / width).astype(np.int64), 0, sub - 1)
    return parent * sub + sidx


def build_feature_matrix(pits, kazan, tuz, stm, buckets=BUCKETS):
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
    phase = phase_from_total(total, buckets)
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
    ap.add_argument("--acc", type=int, default=ACC,
                     help="accumulator width (first-layer/EmbeddingBag output size); "
                          "the fc2 dot product costs acc*hidden MACs, so this is the main "
                          "eval-speed knob (task 9b: narrower acc, same recipe, to see "
                          "whether the close-endgame accuracy gain survives shrinking "
                          "the accumulator back toward legacy speed)")
    ap.add_argument("--w-score", type=float, default=0.0,
                     help="task 12 lever 1: weight of an AUXILIARY score-prediction loss. "
                          "When > 0, adds a second head (hidden -> 1) per bucket that predicts "
                          "the record's normalised final-kazan-difference `score` field "
                          "(bytes 63..67), trained with mask-bit-3-gated MSE (unmasked "
                          "records contribute exactly zero score gradient) and added to the "
                          "value BCE loss as total = value_bce + w_score * score_mse. The "
                          "score head is NEVER exported: export_nnu2/export_nnu2_v3 only "
                          "ever serialise emb/acc_bias/fc2/fc3, so the exported NNU2 file and "
                          "the engine's forward pass are byte-for-byte unchanged; this only "
                          "shapes the shared representation during training. The printed "
                          "epoch loss and the best-checkpoint criterion are always the "
                          "value-only val loss, so runs with different --w-score stay "
                          "comparable on the same quantity.")
    ap.add_argument("--buckets", type=int, default=BUCKETS,
                     help="task 12 lever 2: number of phase buckets. The default (4, "
                          "BUCKETS) is the canonical, PLAYABLE mapping the engine's own "
                          "phase_bucket() implements. Any other value (must be a positive "
                          "multiple of 4, e.g. 8) is a TRAINING-ONLY refinement -- see "
                          "phase_from_total()'s docstring. Such a run's automatic .bin export "
                          "is SKIPPED (only the .pt checkpoint is written): "
                          "engine/src/nnue.rs::phase_bucket() only ever returns 0..3, so a "
                          ">4-bucket NNU2 file would be silently mis-read at play time. Use "
                          "this only to compare val loss; never deploy its .bin.")
    ap.add_argument("--quantize-only", metavar="PT_PATH", default=None,
                     help="skip training: load a .pt checkpoint (e.g. an already-trained "
                          "v2_e12.pt) and export it as version-3 (i16 quantised) NNU2 to "
                          "PT_PATH with '.pt' replaced by '_v3.bin', then exit. Pass the "
                          "same --acc used to train that checkpoint (default 1024) so the "
                          "freshly constructed model's shape matches the saved state_dict.")
    ap.add_argument("--export-v4", metavar="PT_PATH", default=None,
                     help="skip training: load a .pt checkpoint trained with --w-score > 0 "
                          "(so it has an fc_score head) and export it as version-4 (i16 "
                          "quantised value head + auxiliary score head, task 13) NNU2 to "
                          "PT_PATH with '.pt' replaced by '_v4.bin', then exit. Pass the "
                          "same --acc used to train that checkpoint, as with --quantize-only.")
    a = ap.parse_args()

    if a.quantize_only:
        model = NnueV2(acc=a.acc)
        model.load_state_dict(torch.load(a.quantize_only, map_location="cpu"))
        model.eval()
        out_path = os.path.splitext(a.quantize_only)[0] + "_v3.bin"
        scales = export_nnu2_v3(model, out_path)
        print(f"wrote {out_path}")
        print(f"  scale_l1={scales['scale_l1']}")
        print(f"  scale_fc2={scales['scale_fc2']}")
        print(f"  scale_fc3={scales['scale_fc3']}")
        return

    if a.export_v4:
        model = NnueV2(acc=a.acc, score_head=True)
        state = torch.load(a.export_v4, map_location="cpu")
        if "fc_score.0.weight" not in state:
            raise SystemExit(
                f"{a.export_v4} has no score head (fc_score.*) in its state dict -- "
                "retrain with --w-score > 0 before exporting a version-4 net"
            )
        model.load_state_dict(state)
        model.eval()
        out_path = os.path.splitext(a.export_v4)[0] + "_v4.bin"
        scales = export_nnu2_v4(model, out_path)
        print(f"wrote {out_path}")
        print(f"  scale_l1={scales['scale_l1']}")
        print(f"  scale_fc2={scales['scale_fc2']}")
        print(f"  scale_fc3={scales['scale_fc3']}")
        print(f"  scale_score={scales['scale_score']}")
        return

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = NnueV2(acc=a.acc, buckets=a.buckets, score_head=(a.w_score > 0)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    lossf = nn.BCEWithLogitsLoss(reduction="none")

    def prep(path):
        pits, kazan, tuz, stm, value, score, mask = load_bin(path)
        feats, phase = build_feature_matrix(pits, kazan, tuz, stm, buckets=a.buckets)
        w = np.where(mask & MASK_VALUE_NET, a.w_net, a.w_outcome).astype(np.float32)
        score_mask = ((mask & MASK_SCORE) != 0).astype(np.float32)
        return (torch.tensor(feats), torch.tensor(phase), torch.tensor(value),
                torch.tensor(w), torch.tensor(score), torch.tensor(score_mask))

    tr = prep(a.train)
    va = prep(a.val)
    print(f"train {tr[0].shape[0]:,} records, val {va[0].shape[0]:,}")

    def run_epoch(data, train):
        feats, phase, value, w, score, score_mask = data
        n = feats.shape[0]
        order = torch.randperm(n) if train else torch.arange(n)
        tot = cnt = 0.0
        model.train(train)
        for s in range(0, n, a.batch):
            idx = order[s: s + a.batch]
            fb = feats[idx].to(dev)
            offs = torch.arange(0, fb.shape[0] * 23, 23, device=dev)
            bkt = phase[idx].to(dev)
            if a.w_score > 0:
                out, score_pred = model(fb.reshape(-1), offs, bkt, return_score=True)
            else:
                out = model(fb.reshape(-1), offs, bkt)
            value_loss = (lossf(out, value[idx].to(dev)) * w[idx].to(dev)).mean()
            if a.w_score > 0:
                sm = score_mask[idx].to(dev)
                se = (score_pred - score[idx].to(dev)).pow(2) * sm
                denom = sm.sum().clamp(min=1.0)
                score_loss = se.sum() / denom
                loss = value_loss + a.w_score * score_loss
            else:
                loss = value_loss
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            # Always accumulate the VALUE-only loss: the score term is an auxiliary
            # training signal, not part of the reported/compared quantity (task 12).
            tot += value_loss.item() * idx.numel()
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
            if model.buckets == BUCKETS:
                export_nnu2(model, os.path.join(a.out, f"{a.name}.bin"))
                print(f"  saved {a.name}.bin (val {val:.4f})")
            else:
                print(f"  saved {a.name}.pt only (val {val:.4f}) -- buckets={model.buckets} "
                      f"!= {BUCKETS}: NOT exporting an NNU2 .bin, it would be silently "
                      f"mis-read at play time (see --buckets help)")
    print(f"best val loss {best:.4f}")


if __name__ == "__main__":
    main()
