#!/usr/bin/env python3
"""
Masked hybrid bootstrap: HUMAN-move policy (easy to imitate; sup1500 reached
43.8% @1-ply) + ENGINE-eval value (clean; makes 200-sim search HELP). One net,
union of two datasets, MASKED losses so each source only teaches its strength:
  - human >=ELO positions   -> policy loss ONLY (value masked: +/-1 too noisy)
  - big_d12 engine positions -> value loss ONLY (policy masked: engine moves are
    hard to imitate -> they would weaken the policy that actually plays well)
The shared trunk learns both -> keep sup1500's strong policy AND gain a clean
value so search helps (the d2 recipe, with stronger data on both sides).
"""
import sys, os, argparse, time
import numpy as np
import torch, torch.nn as nn
sys.path.insert(0, '.')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'archive', 'old-impls', 'alphazero-code', 'alphazero'))
from model import create_model
from train_distill import load_distill


def load_human(games_dir, min_elo, cap):
    from supervised_pretrain import parse_pgn, extract_moves
    from game import TogyzQumalaq
    files = [os.path.join(games_dir, f) for f in os.listdir(games_dir) if f.endswith('.txt')]
    np.random.shuffle(files)
    S, M = [], []
    for fp in files:
        if len(S) >= cap:
            break
        try:
            h, mt = parse_pgn(fp)
            if h is None:
                continue
            if int(h.get('WhiteElo', '0')) < min_elo or int(h.get('BlackElo', '0')) < min_elo:
                continue
            moves = extract_moves(mt)
            if len(moves) < 10:
                continue
            g = TogyzQumalaq()
            for ply, pit in enumerate(moves):
                if pit not in g.get_valid_moves_list():
                    break
                if ply >= 2:
                    S.append(g.encode_state()); M.append(pit)
                ok, win = g.make_move(pit)
                if not ok or win is not None:
                    break
        except Exception:
            pass
    return np.array(S, dtype=np.float32), np.array(M, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games-dir', default='/home/nurlykhan/game-pars/games')
    ap.add_argument('--min-elo', type=int, default=1500)
    ap.add_argument('--engine-data', required=True)
    ap.add_argument('--model-size', default='large2m')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--scale', type=float, default=300.0)
    ap.add_argument('--cap-human', type=int, default=900000)
    ap.add_argument('--cap-engine', type=int, default=900000)
    ap.add_argument('--value-weight', type=float, default=1.0)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True

    print(f"Loading human policy data (>={args.min_elo}) ...", flush=True)
    Sh, Mh = load_human(args.games_dir, args.min_elo, args.cap_human)
    print(f"  human: {len(Sh)} positions", flush=True)
    print("Loading engine value data ...", flush=True)
    Se, _Me, Ve = load_distill(args.engine_data, args.scale)
    if len(Se) > args.cap_engine:
        idx = np.random.choice(len(Se), args.cap_engine, replace=False); Se, Ve = Se[idx], Ve[idx]
    print(f"  engine: {len(Se)} positions | value std={Ve.std():.3f}", flush=True)

    Sh = Sh.reshape(-1, 7, 9)
    n = len(Sh) + len(Se)
    S = np.concatenate([Sh, Se]).astype(np.float32)
    Mt = np.concatenate([Mh, np.zeros(len(Se), dtype=np.int64)])
    Vt = np.concatenate([np.zeros(len(Sh), dtype=np.float32), Ve]).astype(np.float32)
    pmask = np.concatenate([np.ones(len(Sh), np.float32), np.zeros(len(Se), np.float32)])
    vmask = np.concatenate([np.zeros(len(Sh), np.float32), np.ones(len(Se), np.float32)])
    print(f"Total {n} ({len(Sh)} policy / {len(Se)} value)", flush=True)

    perm = np.random.permutation(n)
    S, Mt, Vt, pmask, vmask = S[perm], Mt[perm], Vt[perm], pmask[perm], vmask[perm]
    nval = int(n * 0.03)
    model = create_model(args.model_size, dev)
    print(f"Model: {args.model_size} ({sum(p.numel() for p in model.parameters()):,} params)", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.05)

    def losses(logp, vp, m, v, pm, vm):
        pl = (-logp.gather(1, m.unsqueeze(1)).squeeze(1) * pm).sum() / pm.sum().clamp(min=1)
        vl = (((vp.squeeze(1) - v) ** 2) * vm).sum() / vm.sum().clamp(min=1)
        return pl, vl

    Sv = torch.from_numpy(S[:nval]).to(dev); Mv = torch.from_numpy(Mt[:nval]).to(dev)
    Vv = torch.from_numpy(Vt[:nval]).to(dev); PMv = torch.from_numpy(pmask[:nval]).to(dev); VMv = torch.from_numpy(vmask[:nval]).to(dev)
    St, Mtt, Vtt, PMt, VMt = S[nval:], Mt[nval:], Vt[nval:], pmask[nval:], vmask[nval:]
    best = float('inf'); best_state = None; bs = args.batch_size
    for ep in range(args.epochs):
        model.train(); idx = np.random.permutation(len(St)); t0 = time.time(); tp = tv = ta = nb = ps = 0
        for s0 in range(0, len(St), bs):
            bi = idx[s0:s0 + bs]
            s = torch.from_numpy(St[bi]).to(dev); m = torch.from_numpy(Mtt[bi]).to(dev)
            v = torch.from_numpy(Vtt[bi]).to(dev); pm = torch.from_numpy(PMt[bi]).to(dev); vm = torch.from_numpy(VMt[bi]).to(dev)
            logp, vp = model(s)
            pl, vl = losses(logp, vp, m, v, pm, vm)
            loss = pl + args.value_weight * vl
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tp += pl.item(); tv += vl.item(); nb += 1
            if pm.sum() > 0:
                ta += (((logp.argmax(1) == m).float() * pm).sum() / pm.sum()).item(); ps += 1
        sched.step()
        model.eval()
        with torch.no_grad():
            vpl = vvl = vacc = 0.0; vnb = pn = 0
            for s0 in range(0, len(Sv), 4096):
                logp, vp = model(Sv[s0:s0 + 4096])
                pl, vl = losses(logp, vp, Mv[s0:s0 + 4096], Vv[s0:s0 + 4096], PMv[s0:s0 + 4096], VMv[s0:s0 + 4096])
                vpl += pl.item(); vvl += vl.item(); vnb += 1
                pmb = PMv[s0:s0 + 4096]
                if pmb.sum() > 0:
                    vacc += (((logp.argmax(1) == Mv[s0:s0 + 4096]).float() * pmb).sum() / pmb.sum()).item(); pn += 1
            vpl /= vnb; vvl /= vnb; vacc /= max(1, pn)
        vtot = vpl + vvl; mark = ''
        if vtot < best:
            best = vtot; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; mark = ' *'
        print(f"Epoch {ep+1:3d}/{args.epochs}  train p={tp/nb:.4f} v={tv/nb:.4f} pacc={ta/max(1,ps)*100:.1f}%  "
              f"val p={vpl:.4f} v={vvl:.4f} pacc={vacc*100:.1f}%  ({time.time()-t0:.0f}s){mark}", flush=True)
    if best_state:
        model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(), 'model_size': args.model_size,
                'best_val_loss': best, 'training_type': 'hybrid_masked_humanpolicy_enginevalue'}, args.output)
    print(f"Saved: {args.output} (val_loss={best:.4f})", flush=True)


if __name__ == '__main__':
    main()
