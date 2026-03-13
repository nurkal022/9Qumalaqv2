"""Fine-tune transferred 58-input model with differential learning rates.

Phase 1 (20 epochs): Only train fc1.weight[:, 40:58] (new feature connections)
Phase 2 (40 epochs): Train all parameters with low LR
"""
import struct
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import sys
sys.path.insert(0, '.')
from train_nnue_v2 import PositionDataset, NNUE, sigmoid_eval, load_binary_weights, export_binary
import argparse

def train_differential(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    data_files = args.data.split(',')
    dataset = PositionDataset(data_files, max_positions=args.max_positions)

    # Use 58 features
    assert args.input_size == 58

    n = len(dataset)
    n_val = min(n // 10, 50000)
    n_train = n - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=4096, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=8192, num_workers=4, pin_memory=True)

    model = NNUE(input_size=58, hidden1=args.hidden1, hidden2=args.hidden2, hidden3=args.hidden3)
    load_binary_weights(model, args.init_weights, 58)
    model = model.to(device)

    lam = args.lam
    K_val = 400.0
    best_val_loss = float('inf')

    def eval_val():
        model.eval()
        val_loss = 0.0
        n_b = 0
        with torch.no_grad():
            for features, evals, results, weights, has_eval in val_loader:
                features, evals, results = features.to(device), evals.to(device), results.to(device)
                weights, has_eval = weights.to(device), has_eval.to(device)
                pred = model(features)
                pred_wp = sigmoid_eval(pred, K_val)
                effective_lam = lam * has_eval
                eval_wp = sigmoid_eval(evals, K_val)
                target = effective_lam * eval_wp + (1 - effective_lam) * results
                val_loss += torch.mean((pred_wp - target) ** 2 * weights).item()
                n_b += 1
        return val_loss / n_b

    def train_epoch():
        model.train()
        total_loss = 0.0
        n_b = 0
        for features, evals, results, weights, has_eval in train_loader:
            features, evals, results = features.to(device), evals.to(device), results.to(device)
            weights, has_eval = weights.to(device), has_eval.to(device)
            pred = model(features)
            pred_wp = sigmoid_eval(pred, K_val)
            effective_lam = lam * has_eval
            eval_wp = sigmoid_eval(evals, K_val)
            target = effective_lam * eval_wp + (1 - effective_lam) * results
            loss = torch.mean((pred_wp - target) ** 2 * weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_b += 1
        return total_loss / n_b

    # Phase 1: Only train new feature connections (fc1.weight[:, 40:58])
    print("=== Phase 1: Training only new feature connections (20 epochs) ===")

    # Create a mask for new features
    # We can't directly mask parameters, so we use parameter groups
    # Freeze all params except fc1.weight, and we'll manually mask gradients
    for name, param in model.named_parameters():
        if name != 'fc1.weight':
            param.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    # Hook to zero out gradients for old features (columns 0-39)
    def mask_old_grads(grad):
        grad_masked = grad.clone()
        grad_masked[:, :40] = 0
        return grad_masked

    handle = model.fc1.weight.register_hook(mask_old_grads)

    for epoch in range(20):
        avg_train = train_epoch()
        scheduler.step()
        avg_val = eval_val()
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), 'nnue_58feat_ft2_best.pt')
            marker = " *"
        else:
            marker = ""
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/20  train={avg_train:.6f}  val={avg_val:.6f}  lr={scheduler.get_last_lr()[0]:.6f}{marker}")

    handle.remove()

    # Phase 2: Fine-tune all parameters with low LR
    print("\n=== Phase 2: Fine-tuning all parameters (40 epochs) ===")
    for param in model.parameters():
        param.requires_grad = True

    # Differential LR: lower for old weights, higher for new feature connections
    old_params = []
    new_params = []
    for name, param in model.named_parameters():
        if name == 'fc1.weight':
            # This param has both old and new, but we can't split it easily
            # Just use medium LR for all of fc1
            new_params.append(param)
        else:
            old_params.append(param)

    optimizer = torch.optim.Adam([
        {'params': old_params, 'lr': 0.00003},
        {'params': new_params, 'lr': 0.0001},
    ], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

    for epoch in range(40):
        avg_train = train_epoch()
        scheduler.step()
        avg_val = eval_val()
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), 'nnue_58feat_ft2_best.pt')
            marker = " *"
        else:
            marker = ""
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/40  train={avg_train:.6f}  val={avg_val:.6f}  lr={scheduler.get_last_lr()[0]:.5f}/{scheduler.get_last_lr()[0]*3.33:.5f}{marker}")

    # Export best model
    model.load_state_dict(torch.load('nnue_58feat_ft2_best.pt', weights_only=True))

    # Manual export since args structure differs
    SCALE = 64
    state = model.state_dict()
    out_path = args.output_bin
    with open(out_path, 'wb') as f:
        f.write(struct.pack('<HHHH', 58, args.hidden1, args.hidden2, args.hidden3))
        layers = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias']
        if args.hidden3 > 0:
            layers.extend(['fc4.weight', 'fc4.bias'])
        for name in layers:
            tensor = state[name].cpu().float().numpy().flatten()
            quantized = np.clip(tensor * SCALE, -32000, 32000).astype(np.int16)
            f.write(quantized.tobytes())
    print(f"\nExported: {out_path} ({os.path.getsize(out_path):,} bytes)")
    print(f"Best val loss: {best_val_loss:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--init-weights', required=True)
    parser.add_argument('--input-size', type=int, default=58)
    parser.add_argument('--hidden1', type=int, default=256)
    parser.add_argument('--hidden2', type=int, default=32)
    parser.add_argument('--hidden3', type=int, default=0)
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--max-positions', type=int, default=None)
    parser.add_argument('--output-bin', default='nnue_weights_58feat_ft2.bin')
    args = parser.parse_args()

    train_differential(args)
