#!/usr/bin/env python3
"""
Fine-tune ONLY the value head of a strong-policy net (sup1500) on ENGINE-GAME
OUTCOME values (gen_engine_games.py .npz), with the trunk + policy FROZEN. This
keeps the strong policy exactly (43.8% @1-ply) while giving the value an
OUTCOME-of-strong-play target (not the engine's static eval, which hurt search).
Decisive test: does 200-sim search now HELP?
"""
import sys, os, argparse, time
import numpy as np
import torch, torch.nn as nn
sys.path.insert(0, '.')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'archive', 'old-impls', 'alphazero-code', 'alphazero'))
from model import create_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init', required=True)
    ap.add_argument('--npz', required=True)
    ap.add_argument('--model-size', default='large2m')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=2048)
    ap.add_argument('--lr', type=float, default=0.002)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True

    model = create_model(args.model_size, dev)
    cp = torch.load(args.init, map_location=dev, weights_only=False)
    model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in cp['model_state_dict'].items()}, strict=False)
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith('value_')
    print("Trainable:", [n for n, p in model.named_parameters() if p.requires_grad])

    d = np.load(args.npz)
    S, V = d['states'].astype(np.float32), d['values'].astype(np.float32)
    print(f"Loaded {len(S)} positions | value mean={V.mean():.3f} std={V.std():.3f}", flush=True)
    n = len(S); perm = np.random.permutation(n); S, V = S[perm], V[perm]
    nval = int(n * 0.03)
    Sv = torch.from_numpy(S[:nval]).to(dev); Vv = torch.from_numpy(V[:nval]).unsqueeze(1).to(dev)
    St, Vt = S[nval:], V[nval:]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.05)
    crit = nn.MSELoss(); bs = args.batch_size; best = float('inf'); best_state = None
    for ep in range(args.epochs):
        model.eval()  # freeze all BN running stats (trunk identical to sup1500)
        idx = np.random.permutation(len(St)); t0 = time.time(); tv = nb = 0
        for s0 in range(0, len(St), bs):
            bi = idx[s0:s0 + bs]
            s = torch.from_numpy(St[bi]).to(dev); v = torch.from_numpy(Vt[bi]).unsqueeze(1).to(dev)
            _lp, vp = model(s); loss = crit(vp, v)
            opt.zero_grad(); loss.backward(); opt.step(); tv += loss.item(); nb += 1
        sched.step()
        with torch.no_grad():
            vv = vnb = 0
            for s0 in range(0, len(Sv), 8192):
                _l, vp = model(Sv[s0:s0 + 8192]); vv += crit(vp, Vv[s0:s0 + 8192]).item(); vnb += 1
            vv /= max(1, vnb)
        mark = ''
        if vv < best:
            best = vv; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; mark = ' *'
        print(f"Epoch {ep+1:3d}/{args.epochs}  train v={tv/nb:.5f}  val v={vv:.5f}  ({time.time()-t0:.0f}s){mark}", flush=True)
    if best_state:
        model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(), 'model_size': args.model_size,
                'best_val_loss': best, 'training_type': 'value_head_engine_outcome'}, args.output)
    print(f"Saved: {args.output} (val_v={best:.5f})", flush=True)


if __name__ == '__main__':
    main()
