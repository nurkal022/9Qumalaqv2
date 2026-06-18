#!/usr/bin/env python3
"""
Engine-distillation trainer: imitate the depth-12 alpha-beta engine directly
(policy = engine best move, value = tanh(engine_eval / SCALE)), instead of weak
humans. Source: archive/mcts-experiments/distill_data/big_d12_training_data.bin
(27-byte records: 23 board + i16 eval@23 + flag@25 + move@26), 1.78M positions,
eval verified side-to-move perspective (matches the [7,9] encode convention).

Usage:
  python3.12 train_distill.py --data <bin> --model-size large2m --epochs 20 \
      --scale 300 --output <out.pt>
"""
import sys, os, argparse, time
import numpy as np
import torch, torch.nn as nn
sys.path.insert(0, '.')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'archive', 'old-impls', 'alphazero-code', 'alphazero'))
from model import create_model

REC = 27


def load_distill(path, scale):
    raw = np.fromfile(path, dtype=np.uint8)
    n = raw.size // REC
    rec = raw[:n * REC].reshape(n, REC)
    boards = rec[:, 0:23]
    pits0 = boards[:, 0:9].astype(np.float32)
    pits1 = boards[:, 9:18].astype(np.float32)
    kaz0 = boards[:, 18].astype(np.float32)
    kaz1 = boards[:, 19].astype(np.float32)
    tuz0 = boards[:, 20].astype(np.int8).astype(np.int32)
    tuz1 = boards[:, 21].astype(np.int8).astype(np.int32)
    side = boards[:, 22].astype(np.int32)
    ev = np.frombuffer(rec[:, 23:25].copy().tobytes(), dtype='<i2').astype(np.float32)
    move = rec[:, 26].astype(np.int64)

    white = (side == 0)
    me_pits = np.where(white[:, None], pits0, pits1)
    opp_pits = np.where(white[:, None], pits1, pits0)
    me_kaz = np.where(white, kaz0, kaz1)
    opp_kaz = np.where(white, kaz1, kaz0)
    me_tuz = np.where(white, tuz0, tuz1)
    opp_tuz = np.where(white, tuz1, tuz0)

    states = np.zeros((n, 7, 9), dtype=np.float32)
    states[:, 0] = me_pits / 50.0
    states[:, 1] = opp_pits / 50.0
    states[:, 2] = (me_kaz / 82.0)[:, None]
    states[:, 3] = (opp_kaz / 82.0)[:, None]
    rows = np.arange(n)
    m4 = me_tuz >= 0
    states[rows[m4], 4, me_tuz[m4]] = 1.0
    m5 = opp_tuz >= 0
    states[rows[m5], 5, opp_tuz[m5]] = 1.0
    states[:, 6] = white[:, None].astype(np.float32)

    values = np.tanh(ev / scale).astype(np.float32)
    # keep only legal-move rows (engine move must be a non-empty current pit)
    legal = me_pits[rows, move] > 0
    return states[legal], move[legal], values[legal]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--model-size', default='large2m')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--scale', type=float, default=300.0, help='value = tanh(eval/scale)')
    ap.add_argument('--value-weight', type=float, default=1.0)
    ap.add_argument('--val-split', type=float, default=0.03)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"Device: {dev}")

    print(f"Loading {args.data} (scale={args.scale}) ...")
    S, M, V = load_distill(args.data, args.scale)
    print(f"  {len(S)} legal records | value mean={V.mean():.3f} std={V.std():.3f} | "
          f"move-entropy={-(np.bincount(M, minlength=9)/len(M) * np.log(np.bincount(M, minlength=9)/len(M)+1e-9)).sum():.3f}")

    n = len(S)
    perm = np.random.permutation(n)
    S, M, V = S[perm], M[perm], V[perm]
    nval = int(n * args.val_split)
    Sv, Mv, Vv = S[:nval], M[:nval], V[:nval]
    St, Mt, Vt = S[nval:], M[nval:], V[nval:]
    print(f"  train {len(St)} / val {len(Sv)}")

    model = create_model(args.model_size, dev)
    print(f"Model: {args.model_size} ({sum(p.numel() for p in model.parameters()):,} params)")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.05)
    pcrit = nn.CrossEntropyLoss()
    vcrit = nn.MSELoss()

    Sv_t = torch.from_numpy(Sv).to(dev)
    Mv_t = torch.from_numpy(Mv).to(dev)
    Vv_t = torch.from_numpy(Vv).unsqueeze(1).to(dev)

    best_val = float('inf'); best_state = None
    bs = args.batch_size
    for ep in range(args.epochs):
        model.train()
        idx = np.random.permutation(len(St))
        t0 = time.time(); tp = tv = ta = nb = 0
        for st in range(0, len(St), bs):
            bi = idx[st:st + bs]
            s = torch.from_numpy(St[bi]).to(dev)
            m = torch.from_numpy(Mt[bi]).to(dev)
            v = torch.from_numpy(Vt[bi]).unsqueeze(1).to(dev)
            logp, vp = model(s)
            pl = pcrit(logp, m)
            vl = vcrit(vp, v)
            loss = pl + args.value_weight * vl
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tp += pl.item(); tv += vl.item(); nb += 1
            ta += (logp.argmax(1) == m).float().mean().item()
        sched.step()
        # val
        model.eval()
        with torch.no_grad():
            vacc = vp_loss = vv_loss = 0.0; vnb = 0
            for st in range(0, len(Sv), 4096):
                logp, vp = model(Sv_t[st:st + 4096])
                mm = Mv_t[st:st + 4096]
                vp_loss += pcrit(logp, mm).item()
                vv_loss += vcrit(vp, Vv_t[st:st + 4096]).item()
                vacc += (logp.argmax(1) == mm).float().sum().item(); vnb += 1
            vp_loss /= vnb; vv_loss /= vnb; vacc /= len(Sv)
        vtot = vp_loss + vv_loss
        mark = ''
        if vtot < best_val:
            best_val = vtot; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; mark = ' *'
        print(f"Epoch {ep+1:3d}/{args.epochs}  train p={tp/nb:.4f} v={tv/nb:.4f} acc={ta/nb*100:.1f}%  "
              f"val p={vp_loss:.4f} v={vv_loss:.4f} acc={vacc*100:.1f}%  ({time.time()-t0:.0f}s){mark}", flush=True)

    if best_state:
        model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(), 'model_size': args.model_size,
                'best_val_loss': best_val, 'scale': args.scale, 'training_type': 'engine_distill_d12'}, args.output)
    print(f"Saved: {args.output} (val_loss={best_val:.4f})")


if __name__ == '__main__':
    main()
