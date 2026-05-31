#!/usr/bin/env python3
"""
Hybrid distillation training: combines multiple data sources.

Data sources:
1. Engine distillation (policy=engine_best_move, value=blended eval+result)
2. PlayOK 2000+ (policy=human_move, value=game_result)
3. Master games from server (policy=human_move, value=game_result)

Target: model that's stronger than Gen7 by combining:
- Engine's search strength (tactics)
- Human strategic understanding (strategy)
"""

import sys, os, glob, argparse, time, struct
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../alphazero-code/alphazero'))
from model import create_model
from game import TogyzQumalaq, Player

# Import distillation decoder
from train_distillation import load_distillation_data, decode_records


def load_playok_data(games_dir, min_elo=2000, max_examples=300000):
    """Load PlayOK games as (states, policies, values)."""
    from supervised_pretrain import parse_pgn, extract_moves

    files = [os.path.join(games_dir, f) for f in os.listdir(games_dir) if f.endswith('.txt')]
    np.random.shuffle(files)

    states, policies, values = [], [], []
    for filepath in files:
        if len(states) >= max_examples:
            break
        try:
            headers, move_text = parse_pgn(filepath)
            if headers is None: continue
            w_elo = int(headers.get('WhiteElo', '0'))
            b_elo = int(headers.get('BlackElo', '0'))
            if w_elo < min_elo or b_elo < min_elo: continue

            result_str = headers.get('Result', '')
            if result_str == '1-0': wv = 1.0
            elif result_str == '0-1': wv = -1.0
            elif result_str == '1/2-1/2': wv = 0.0
            else: continue

            moves = extract_moves(move_text)
            if len(moves) < 10: continue

            game = TogyzQumalaq()
            for ply, pit in enumerate(moves):
                valid = game.get_valid_moves_list()
                if pit not in valid: break
                if ply >= 2:
                    state = game.encode_state()
                    cp = game.state.current_player
                    value = wv if cp == Player.WHITE else -wv
                    policy = np.zeros(9, dtype=np.float32)
                    policy[pit] = 1.0
                    states.append(state); policies.append(policy); values.append(value)
                success, winner = game.make_move(pit)
                if not success or winner is not None: break
        except: pass

    return (np.array(states, dtype=np.float32),
            np.array(policies, dtype=np.float32),
            np.array(values, dtype=np.float32))


def train(args):
    device = 'cuda'
    use_amp = True
    print(f'Device: {device}, AMP: {use_amp}')

    # Load all data sources
    all_s, all_p, all_v, all_w = [], [], [], []

    if args.engine_data:
        print(f'\n[Source 1] Engine distillation...')
        records = load_distillation_data(args.engine_data)
        s, p, v = decode_records(records)
        all_s.append(s); all_p.append(p); all_v.append(v)
        all_w.append(np.full(len(s), args.engine_weight, dtype=np.float32))
        print(f'  {len(s):,} positions (weight={args.engine_weight})')

    if args.playok_dir:
        print(f'\n[Source 2] PlayOK {args.playok_min_elo}+...')
        s, p, v = load_playok_data(args.playok_dir, min_elo=args.playok_min_elo, max_examples=args.playok_max)
        all_s.append(s); all_p.append(p); all_v.append(v)
        all_w.append(np.full(len(s), args.playok_weight, dtype=np.float32))
        print(f'  {len(s):,} positions (weight={args.playok_weight})')

    # Combine
    states = np.concatenate(all_s)
    policies = np.concatenate(all_p)
    values = np.concatenate(all_v)
    weights = np.concatenate(all_w)
    n = len(states)
    print(f'\nTotal: {n:,} positions')

    # Move to GPU if fits
    states_t = torch.from_numpy(states).cuda()
    policies_t = torch.from_numpy(policies).cuda()
    values_t = torch.from_numpy(values).cuda()
    weights_t = torch.from_numpy(weights).cuda()

    # Model
    model = create_model(args.model_size, device=device)
    params = sum(p.numel() for p in model.parameters())
    print(f'Model: {args.model_size} ({params:,} params)')

    if args.init_checkpoint and os.path.exists(args.init_checkpoint):
        cp = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        sd = cp.get('model_state_dict', cp)
        cleaned = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        model.load_state_dict(cleaned, strict=False)
        print(f'Loaded init: {args.init_checkpoint}')

    # Train/val split
    perm = torch.randperm(n).cuda()
    val_n = min(n // 10, 20000)
    val_idx = perm[:val_n]
    train_idx = perm[val_n:]
    print(f'Train: {len(train_idx):,}, Val: {len(val_idx):,}')

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_acc, no_improve = 0, 0
    patience = 5

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

        saved = ''
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'val_acc': val_acc}, args.output)
            saved = ' *'
        else:
            no_improve += 1

        print(f'Epoch {epoch+1:2d}/{args.epochs} ({t_epoch:.1f}s): '
              f'train p={tp/nb:.3f} v={tv/nb:.3f} | val p={val_p:.3f} v={val_v:.3f} '
              f'acc={val_acc:.1f}% (best={best_val_acc:.1f}%){saved}', flush=True)

        if no_improve >= patience:
            print(f'Early stopping')
            break

    print(f'\nBest val accuracy: {best_val_acc:.1f}%')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine-data', default=None, help='Glob for engine distillation .bin files')
    parser.add_argument('--engine-weight', type=float, default=2.0, help='Loss weight for engine samples')
    parser.add_argument('--playok-dir', default=None, help='PlayOK games directory')
    parser.add_argument('--playok-min-elo', type=int, default=2000)
    parser.add_argument('--playok-max', type=int, default=300000)
    parser.add_argument('--playok-weight', type=float, default=1.0)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model-size', default='large2m')
    parser.add_argument('--init-checkpoint', default=None)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.0005)
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
