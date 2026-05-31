#!/usr/bin/env python3
"""
Master training pipeline — combines best ideas from all previous approaches.

Features:
- Hybrid data: engine distillation + PlayOK supervised + optional selfplay
- Weighted sampling by data source quality
- AMP (bfloat16) training on GPU
- GPU data loading (no DataLoader overhead)
- Cosine LR schedule with warmup
- Label smoothing option for policy
- Early stopping on val accuracy
- Proper train/val split with shuffling
- Checkpoint best model
"""

import sys, os, glob, argparse, time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../alphazero-code/alphazero'))
from model import create_model
from game import TogyzQumalaq, Player

RECORD_SIZE = 27


def load_engine_data(path_pattern):
    """Load engine distillation data (27-byte format with best_move + eval)."""
    files = sorted(glob.glob(path_pattern))
    if not files:
        return None, None, None

    all_r = []
    for f in files:
        data = np.fromfile(f, dtype=np.uint8)
        n = len(data) // RECORD_SIZE
        if n > 0:
            all_r.append(data[:n*RECORD_SIZE].reshape(n, RECORD_SIZE))

    if not all_r:
        return None, None, None

    records = np.concatenate(all_r)
    print(f'  Engine data: {len(records):,} records from {len(files)} files')

    # Parse
    n = len(records)
    boards = records[:, :23]
    evals = np.frombuffer(records[:, 23:25].copy().tobytes(), dtype='<i2').reshape(n)
    results = records[:, 25]
    best_moves = records[:, 26]

    # Filter: valid best_move, non-zero eval (skip polluted positions)
    mask = (best_moves < 9) & (np.abs(evals) > 0)
    if mask.sum() < n * 0.7:
        # Don't filter too aggressively - maybe eval=0 are legitimate draws
        mask = best_moves < 9
    records = records[mask]
    boards = boards[mask]
    evals = evals[mask]
    results = results[mask]
    best_moves = best_moves[mask]
    n = len(records)

    # Encode states
    states = _encode_boards(boards)

    # Policy: one-hot on engine best move
    policies = np.zeros((n, 9), dtype=np.float32)
    policies[np.arange(n), best_moves] = 1.0

    # Value: blend eval + result (lambda=0.7 weights eval more)
    eval_norm = np.tanh(evals / 300.0).astype(np.float32)
    sides = records[:, 22]
    result_white = (results.astype(np.float32) - 1.0)  # -1, 0, +1
    result_stm = np.where(sides == 0, result_white, -result_white)
    LAMBDA = 0.7
    values = LAMBDA * eval_norm + (1.0 - LAMBDA) * result_stm

    return states, policies, values


def _encode_boards(boards):
    """Encode raw board bytes to [N, 7, 9] tensor."""
    n = len(boards)
    states = np.zeros((n, 7, 9), dtype=np.float32)
    PIT = 50.0
    KAZ = 82.0
    for i in range(n):
        b = boards[i]
        p0 = b[0:9].astype(np.float32)
        p1 = b[9:18].astype(np.float32)
        k0, k1 = float(b[18]), float(b[19])
        t0 = np.int8(b[20]); t1 = np.int8(b[21])
        side = int(b[22])
        if side == 0:
            me_p, opp_p = p0, p1; me_k, opp_k = k0, k1; me_t, opp_t = t0, t1
        else:
            me_p, opp_p = p1, p0; me_k, opp_k = k1, k0; me_t, opp_t = t1, t0
        states[i, 0] = me_p / PIT
        states[i, 1] = opp_p / PIT
        states[i, 2] = me_k / KAZ
        states[i, 3] = opp_k / KAZ
        if me_t >= 0: states[i, 4, me_t] = 1.0
        if opp_t >= 0: states[i, 5, opp_t] = 1.0
        states[i, 6] = 1.0 if side == 0 else 0.0
    return states


def load_playok_data(games_dir, min_elo=1500, max_examples=500000):
    """Load PlayOK games."""
    from supervised_pretrain import parse_pgn, extract_moves

    files = [os.path.join(games_dir, f) for f in os.listdir(games_dir) if f.endswith('.txt')]
    np.random.shuffle(files)

    s_list, p_list, v_list = [], [], []
    count = 0
    for fp in files:
        if len(s_list) >= max_examples: break
        try:
            headers, move_text = parse_pgn(fp)
            if headers is None: continue
            w_elo = int(headers.get('WhiteElo', '0'))
            b_elo = int(headers.get('BlackElo', '0'))
            if w_elo < min_elo or b_elo < min_elo: continue
            result = headers.get('Result', '')
            if result == '1-0': wv = 1.0
            elif result == '0-1': wv = -1.0
            elif result == '1/2-1/2': wv = 0.0
            else: continue
            moves = extract_moves(move_text)
            if len(moves) < 10: continue

            game = TogyzQumalaq()
            for ply, pit in enumerate(moves):
                valid = game.get_valid_moves_list()
                if pit not in valid: break
                if ply >= 2:
                    st = game.encode_state()
                    cp = game.state.current_player
                    val = wv if cp == Player.WHITE else -wv
                    pol = np.zeros(9, dtype=np.float32); pol[pit] = 1.0
                    s_list.append(st); p_list.append(pol); v_list.append(val)
                ok, w = game.make_move(pit)
                if not ok or w is not None: break
            count += 1
        except: pass

    print(f'  PlayOK data: {len(s_list):,} positions from {count:,} games (min_elo={min_elo})')
    return (np.array(s_list, dtype=np.float32),
            np.array(p_list, dtype=np.float32),
            np.array(v_list, dtype=np.float32))


def train(args):
    device = 'cuda'
    print(f'Device: {device}, AMP: bfloat16')

    # Load all data sources
    all_s, all_p, all_v, all_w = [], [], [], []

    if args.engine_data:
        print('\n[1/2] Loading engine distillation data...')
        es, ep, ev = load_engine_data(args.engine_data)
        if es is not None:
            all_s.append(es); all_p.append(ep); all_v.append(ev)
            all_w.append(np.full(len(es), args.engine_weight, dtype=np.float32))
            print(f'  Weight: {args.engine_weight}')

    if args.playok_dir:
        print('\n[2/2] Loading PlayOK data...')
        ps, pp, pv = load_playok_data(args.playok_dir, args.playok_min_elo, args.playok_max)
        if len(ps) > 0:
            all_s.append(ps); all_p.append(pp); all_v.append(pv)
            all_w.append(np.full(len(ps), args.playok_weight, dtype=np.float32))
            print(f'  Weight: {args.playok_weight}')

    states = np.concatenate(all_s)
    policies = np.concatenate(all_p)
    values = np.concatenate(all_v)
    weights = np.concatenate(all_w)
    n = len(states)
    print(f'\nTotal: {n:,} positions')

    # Move to GPU
    states_t = torch.from_numpy(states).cuda()
    policies_t = torch.from_numpy(policies).cuda()
    values_t = torch.from_numpy(values).cuda()
    weights_t = torch.from_numpy(weights).cuda()

    # Model
    model = create_model(args.model_size, device=device)
    params = sum(p.numel() for p in model.parameters())
    print(f'\nModel: {args.model_size} ({params:,} params)')

    if args.init_checkpoint and os.path.exists(args.init_checkpoint):
        cp = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        sd = cp.get('model_state_dict', cp)
        cleaned = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        model.load_state_dict(cleaned, strict=False)
        print(f'Loaded init: {args.init_checkpoint}')

    # Split
    perm = torch.randperm(n).cuda()
    val_n = min(n // 10, 30000)
    val_idx = perm[:val_n]
    train_idx = perm[val_n:]
    print(f'Train: {len(train_idx):,}, Val: {len(val_idx):,}')

    # Optimizer with warmup + cosine
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_steps = 3
    total_steps = args.epochs
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress)) * 0.9 + 0.1  # min 10% of peak
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_acc = 0
    no_improve = 0
    patience = 8

    print(f'\nTraining {args.epochs} epochs, batch={args.batch_size}, lr={args.lr}, wd={args.weight_decay}...')
    for epoch in range(args.epochs):
        model.train()
        perm_ep = torch.randperm(len(train_idx), device='cuda')
        ids = train_idx[perm_ep]
        tp, tv, nb = 0, 0, 0
        t0 = time.time()

        for start in range(0, len(ids), args.batch_size):
            bi = ids[start:start + args.batch_size]
            s = states_t[bi]; pt = policies_t[bi]; vt = values_t[bi].unsqueeze(1); w = weights_t[bi]

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                log_p, v = model(s)
                # Label smoothing
                if args.label_smooth > 0:
                    smooth = args.label_smooth / 9
                    pt_smooth = pt * (1 - args.label_smooth) + smooth
                    p_loss_per = -torch.sum(pt_smooth * log_p, dim=1)
                else:
                    p_loss_per = -torch.sum(pt * log_p, dim=1)
                p_loss = (p_loss_per * w).sum() / w.sum()
                v_loss_per = F.mse_loss(v.float(), vt, reduction='none').squeeze(1)
                v_loss = (v_loss_per * w).sum() / w.sum()
                loss = p_loss + v_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tp += p_loss.item(); tv += v_loss.item(); nb += 1

        scheduler.step()
        t_epoch = time.time() - t0

        # Validation
        model.eval()
        with torch.no_grad():
            val_p, val_v, val_acc, val_n_total = 0, 0, 0, 0
            for start in range(0, len(val_idx), args.batch_size):
                bi = val_idx[start:start + args.batch_size]
                s = states_t[bi]; pt = policies_t[bi]; vt = values_t[bi].unsqueeze(1)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    log_p, v = model(s)
                log_p = log_p.float(); v = v.float()
                val_p += (-torch.sum(pt * log_p, dim=1).sum()).item()
                val_v += F.mse_loss(v, vt, reduction='sum').item()
                val_acc += (torch.argmax(log_p, dim=1) == torch.argmax(pt, dim=1)).sum().item()
                val_n_total += len(bi)
            val_p /= val_n_total
            val_v /= val_n_total
            val_acc = val_acc / val_n_total * 100

        lr_now = optimizer.param_groups[0]['lr']
        saved = ''
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save({'model_state_dict': model.state_dict(),
                        'epoch': epoch, 'val_acc': val_acc}, args.output)
            saved = ' *'
        else:
            no_improve += 1

        print(f'Epoch {epoch+1:3d}/{args.epochs} ({t_epoch:.1f}s lr={lr_now:.5f}): '
              f'train p={tp/nb:.3f} v={tv/nb:.3f} | '
              f'val p={val_p:.3f} v={val_v:.3f} acc={val_acc:.1f}% (best={best_val_acc:.1f}%){saved}',
              flush=True)

        if no_improve >= patience:
            print(f'Early stopping (no improvement for {patience} epochs)')
            break

    print(f'\nBest val accuracy: {best_val_acc:.1f}%')
    print(f'Saved: {args.output}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine-data', default=None, help='Glob for engine distillation .bin')
    parser.add_argument('--engine-weight', type=float, default=1.0)
    parser.add_argument('--playok-dir', default=None)
    parser.add_argument('--playok-min-elo', type=int, default=1500)
    parser.add_argument('--playok-max', type=int, default=500000)
    parser.add_argument('--playok-weight', type=float, default=1.0)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model-size', default='large2m')
    parser.add_argument('--init-checkpoint', default=None)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--label-smooth', type=float, default=0.0)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
