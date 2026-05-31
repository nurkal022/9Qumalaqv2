#!/usr/bin/env python3
"""
Distillation training: learn from Gen7 engine's (best_move, eval) labels.

Binary format (27 bytes per record):
  0-8:   white pits (9 u8)
  9-17:  black pits (9 u8)
  18:    kazan[0] (u8)
  19:    kazan[1] (u8)
  20:    tuzdyk[0] (i8)
  21:    tuzdyk[1] (i8)
  22:    side_to_move (u8, 0=white, 1=black)
  23-24: eval (i16 LE, from side-to-move perspective)
  25:    result (0=white_loss, 1=draw, 2=white_win)
  26:    best_move (0-8, or 255 if unknown)
"""

import sys, os, glob, argparse, time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../alphazero-code/alphazero'))
from model import create_model

RECORD_SIZE = 27


def load_distillation_data(path_pattern):
    """Load binary distillation data from one or more files."""
    files = sorted(glob.glob(path_pattern))
    if not files:
        raise FileNotFoundError(f'No files match: {path_pattern}')

    all_records = []
    for f in files:
        data = np.fromfile(f, dtype=np.uint8)
        n = len(data) // RECORD_SIZE
        if n > 0:
            all_records.append(data[:n*RECORD_SIZE].reshape(n, RECORD_SIZE))

    records = np.concatenate(all_records) if all_records else np.zeros((0, RECORD_SIZE), dtype=np.uint8)
    print(f'Loaded {len(records):,} records from {len(files)} files')
    return records


def decode_records(records):
    """Decode binary records into (states, policies, values) for training."""
    n = len(records)

    # Parse fields
    boards = records[:, :23]
    evals = np.frombuffer(records[:, 23:25].copy().tobytes(), dtype='<i2').reshape(n)
    results = records[:, 25]
    best_moves = records[:, 26]

    # Filter: only records with valid best_move (0-8)
    mask = best_moves < 9
    records = records[mask]
    boards = boards[mask]
    evals = evals[mask]
    results = results[mask]
    best_moves = best_moves[mask]
    n = len(records)
    print(f'After filter: {n:,} valid records')

    # Encode states
    states = np.zeros((n, 7, 9), dtype=np.float32)
    PIT_NORM = 50.0
    KAZAN_NORM = 82.0

    for i in range(n):
        b = boards[i]
        pits0 = b[0:9].astype(np.float32)
        pits1 = b[9:18].astype(np.float32)
        kazan0 = float(b[18])
        kazan1 = float(b[19])
        tuzdyk0 = np.int8(b[20])
        tuzdyk1 = np.int8(b[21])
        side = int(b[22])

        if side == 0:
            me_pits, opp_pits = pits0, pits1
            me_kazan, opp_kazan = kazan0, kazan1
            me_tuzdyk, opp_tuzdyk = tuzdyk0, tuzdyk1
        else:
            me_pits, opp_pits = pits1, pits0
            me_kazan, opp_kazan = kazan1, kazan0
            me_tuzdyk, opp_tuzdyk = tuzdyk1, tuzdyk0

        states[i, 0] = me_pits / PIT_NORM
        states[i, 1] = opp_pits / PIT_NORM
        states[i, 2] = me_kazan / KAZAN_NORM
        states[i, 3] = opp_kazan / KAZAN_NORM
        if me_tuzdyk >= 0:
            states[i, 4, me_tuzdyk] = 1.0
        if opp_tuzdyk >= 0:
            states[i, 5, opp_tuzdyk] = 1.0
        states[i, 6] = 1.0 if side == 0 else 0.0

    # Policy: one-hot on engine's best move
    policies = np.zeros((n, 9), dtype=np.float32)
    policies[np.arange(n), best_moves] = 1.0

    # Value: blend of eval and game result (lambda=0.5 like NNUE distillation)
    # eval is in centipawn scale, scale to [-1, 1] via tanh(eval / 400)
    eval_value = np.tanh(evals / 400.0).astype(np.float32)

    # Result from side_to_move perspective
    sides = records[:, 22]
    # result byte: 0=white_loss, 1=draw, 2=white_win
    result_white = (results.astype(np.float32) - 1.0)  # -1, 0, +1
    result_stm = np.where(sides == 0, result_white, -result_white)

    # Lambda=0.5: mix eval and game result
    LAMBDA = 0.5
    values = LAMBDA * eval_value + (1.0 - LAMBDA) * result_stm

    return states, policies, values


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp = device == 'cuda'
    print(f'Device: {device}, AMP: {use_amp}')

    # Load data
    print(f'\nLoading data from {args.data}...')
    records = load_distillation_data(args.data)
    states, policies, values = decode_records(records)

    # Move data to GPU once (if fits)
    states_t = torch.from_numpy(states)
    policies_t = torch.from_numpy(policies)
    values_t = torch.from_numpy(values)

    gpu_mem_gb = states_t.element_size() * states_t.numel() / 1e9 * 3  # rough estimate
    if use_amp and gpu_mem_gb < 8:
        print(f'Moving data to GPU (~{gpu_mem_gb:.1f} GB)')
        states_t = states_t.cuda()
        policies_t = policies_t.cuda()
        values_t = values_t.cuda()
        data_on_gpu = True
    else:
        data_on_gpu = False

    # Optional: load supervised pretrained model to start from
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
    n = len(states)
    perm = np.random.permutation(n)
    val_n = min(n // 10, 20000)
    val_idx = torch.from_numpy(perm[:val_n]).long()
    train_idx = torch.from_numpy(perm[val_n:]).long()
    if data_on_gpu:
        val_idx = val_idx.cuda()
        train_idx = train_idx.cuda()
    print(f'Train: {len(train_idx):,}, Val: {len(val_idx):,}')

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_acc = 0
    best_val_p = float('inf')
    early_stop_patience = 5
    no_improvement = 0

    print(f'\nTraining {args.epochs} epochs, batch={args.batch_size}, lr={args.lr}...')
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(train_idx), device=train_idx.device)
        idx_shuffled = train_idx[perm]
        total_p, total_v, num_b = 0, 0, 0
        t0 = time.time()

        for start in range(0, len(train_idx), args.batch_size):
            end = min(start + args.batch_size, len(train_idx))
            bi = idx_shuffled[start:end]

            if data_on_gpu:
                s = states_t[bi]
                pt = policies_t[bi]
                vt = values_t[bi].unsqueeze(1)
            else:
                s = states_t[bi].to(device, non_blocking=True)
                pt = policies_t[bi].to(device, non_blocking=True)
                vt = values_t[bi].unsqueeze(1).to(device, non_blocking=True)

            if use_amp:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    log_p, v = model(s)
                    p_loss = -torch.mean(torch.sum(pt * log_p, dim=1))
                    v_loss = F.mse_loss(v.float(), vt)
                    loss = p_loss + v_loss
            else:
                log_p, v = model(s)
                p_loss = -torch.mean(torch.sum(pt * log_p, dim=1))
                v_loss = F.mse_loss(v, vt)
                loss = p_loss + v_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_p += p_loss.item()
            total_v += v_loss.item()
            num_b += 1

        scheduler.step()
        train_time = time.time() - t0

        # Validation
        model.eval()
        with torch.no_grad():
            val_p_total, val_v_total, val_acc_sum, val_n_total = 0, 0, 0, 0
            for start in range(0, len(val_idx), args.batch_size):
                end = min(start + args.batch_size, len(val_idx))
                bi = val_idx[start:end]
                if data_on_gpu:
                    s = states_t[bi]; pt = policies_t[bi]; vt = values_t[bi].unsqueeze(1)
                else:
                    s = states_t[bi].to(device); pt = policies_t[bi].to(device); vt = values_t[bi].unsqueeze(1).to(device)
                if use_amp:
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        log_p, v = model(s)
                else:
                    log_p, v = model(s)
                log_p = log_p.float(); v = v.float()
                p_l = -torch.mean(torch.sum(pt * log_p, dim=1)).item()
                v_l = F.mse_loss(v, vt).item()
                pred = torch.argmax(log_p, dim=1)
                target = torch.argmax(pt, dim=1)
                acc = (pred == target).float().sum().item()
                val_p_total += p_l * len(bi)
                val_v_total += v_l * len(bi)
                val_acc_sum += acc
                val_n_total += len(bi)

            val_p = val_p_total / val_n_total
            val_v = val_v_total / val_n_total
            val_acc = val_acc_sum / val_n_total * 100

        saved = ''
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improvement = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'val_p': val_p,
            }, args.output)
            saved = ' *'
        else:
            no_improvement += 1

        print(f'Epoch {epoch+1:2d}/{args.epochs} ({train_time:.1f}s): '
              f'train p={total_p/num_b:.3f} v={total_v/num_b:.3f} | '
              f'val p={val_p:.3f} v={val_v:.3f} acc={val_acc:.1f}% (best={best_val_acc:.1f}%){saved}', flush=True)

        if no_improvement >= early_stop_patience:
            print(f'Early stopping: no improvement for {early_stop_patience} epochs')
            break

    print(f'\nDone! Best val accuracy: {best_val_acc:.1f}%')
    print(f'Saved: {args.output}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Glob pattern for .bin files')
    parser.add_argument('--output', required=True, help='Output .pt file')
    parser.add_argument('--model-size', default='large2m')
    parser.add_argument('--init-checkpoint', default=None, help='Start from existing checkpoint')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
