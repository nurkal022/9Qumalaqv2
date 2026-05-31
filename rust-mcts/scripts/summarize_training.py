#!/usr/bin/env python3
"""Parse night_train.log and print summary tables.

Usage: python3 scripts/summarize_training.py [path/to/night_train.log]
"""
import sys
import re
import os

LOG = sys.argv[1] if len(sys.argv) > 1 else "checkpoints_night/night_train.log"

if not os.path.exists(LOG):
    print(f"Log not found: {LOG}")
    sys.exit(1)

iters = []      # list of dicts
evals = []      # list of dicts

current_iter = None
with open(LOG) as f:
    for line in f:
        line = line.rstrip()

        m = re.search(r"--- Iteration (\d+)", line)
        if m:
            current_iter = int(m.group(1))
            continue

        m = re.search(r"Loss: ([\d.]+) \(p=([\d.]+), v=([\d.]+)\)", line)
        if m and current_iter is not None:
            iters.append({
                "iter": current_iter,
                "loss": float(m.group(1)),
                "p_loss": float(m.group(2)),
                "v_loss": float(m.group(3)),
            })
            continue

        m = re.search(r"Eval: (\d+)W-(\d+)D-(\d+)L = ([\d.]+)%", line)
        if m and current_iter is not None:
            w, d, l = int(m.group(1)), int(m.group(2)), int(m.group(3))
            wr = float(m.group(4))
            evals.append({
                "iter": current_iter,
                "wins": w, "draws": d, "losses": l, "winrate": wr,
                "total": w + d + l,
            })

# Summary
print(f"=== Training summary: {LOG} ===")
print(f"Iters logged: {len(iters)}")

if iters:
    first = iters[0]
    last = iters[-1]
    losses = [it["loss"] for it in iters]
    p_losses = [it["p_loss"] for it in iters]
    v_losses = [it["v_loss"] for it in iters]
    print(f"\nFirst iter: {first['iter']} | loss={first['loss']:.4f} (p={first['p_loss']:.4f}, v={first['v_loss']:.4f})")
    print(f"Last  iter: {last['iter']} | loss={last['loss']:.4f} (p={last['p_loss']:.4f}, v={last['v_loss']:.4f})")
    print(f"Min   loss: {min(losses):.4f} | min p={min(p_losses):.4f} | min v={min(v_losses):.4f}")

print(f"\nEvals logged: {len(evals)}")
if evals:
    print(f"\n  iter | W-D-L     | wr%  | running_best%")
    print(f"  -----+-----------+------+--------------")
    best_wr = -1
    for ev in evals:
        if ev["winrate"] > best_wr:
            best_wr = ev["winrate"]
        marker = " *" if ev["winrate"] == best_wr else "  "
        print(f"  {ev['iter']:4d} | {ev['wins']:2d}W-{ev['draws']:2d}D-{ev['losses']:2d}L | {ev['winrate']:4.1f}{marker} | {best_wr:4.1f}")
