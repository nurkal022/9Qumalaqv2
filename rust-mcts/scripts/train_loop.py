#!/usr/bin/env python3
"""
Full AlphaZero training loop v2: Rust selfplay -> Python train -> ONNX export -> eval -> repeat.

Key improvements over v1:
- Best model selection: selfplay always uses best checkpoint (not current)
- Lower lr (0.0003) with cosine annealing to prevent forgetting pretrained knowledge
- 20-game eval for more reliable signal
- Freshness weighting: recent positions sampled 2x more often
- Gating: new model must beat current best to become selfplay model

Usage:
  python scripts/train_loop.py --iterations 500 --games 100 --sims 800 --model-size large2m \
      --init-checkpoint ../alphazero-code/alphazero/checkpoints/supervised_pretrained_2m.pt
"""

import sys, os, argparse, subprocess, time, shutil, tempfile, math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../alphazero-code/alphazero'))
from model import create_model
from game import TogyzQumalaq, Player

from train_alphazero import load_replay_buffer, export_onnx

# ── Config ──────────────────────────────────────────────────

RUST_BINARY = os.path.join(os.path.dirname(__file__), '..', 'target', 'release', 'rust-mcts')
ENGINE_BINARY = os.path.join(os.path.dirname(__file__), '..', '..', 'engine', 'target', 'release', 'togyzkumalaq-engine')

NVIDIA_LIBS = os.path.expanduser('~/.local/lib/python3.12/site-packages/nvidia')
CUDA_LD_PATH = ':'.join([
    f'{NVIDIA_LIBS}/cublas/lib',
    f'{NVIDIA_LIBS}/cuda_runtime/lib',
    f'{NVIDIA_LIBS}/curand/lib',
    f'{NVIDIA_LIBS}/cudnn/lib',
    f'{NVIDIA_LIBS}/cufft/lib',
])
ORT_DYLIB = os.path.expanduser('~/.local/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.24.4')


def rust_selfplay(model_onnx, output_bin, games=100, sims=800, workers=20, batch_size=128):
    """Run Rust MCTS self-play and produce replay buffer."""
    env = os.environ.copy()
    env['ORT_DYLIB_PATH'] = ORT_DYLIB
    env['LD_LIBRARY_PATH'] = CUDA_LD_PATH + ':' + env.get('LD_LIBRARY_PATH', '')

    cmd = [
        os.path.abspath(RUST_BINARY),
        '--model', os.path.abspath(model_onnx),
        '--games', str(games),
        '--sims', str(sims),
        '--workers', str(workers),
        '--batch-size', str(batch_size),
        '--output', os.path.abspath(output_bin),
    ]

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True,
                           cwd=os.path.dirname(os.path.abspath(RUST_BINARY)),
                           timeout=600)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  Rust selfplay FAILED (exit {result.returncode})")
        if result.stderr:
            print(result.stderr[-500:])
        return False, 0, elapsed

    positions = 0
    for line in result.stderr.splitlines():
        if 'Positions:' in line:
            try:
                positions = int(line.split('Positions:')[1].strip())
            except:
                pass

    return True, positions, elapsed


def train_on_buffer(model, optimizer, buffer_path,
                    max_buffer=500000, epochs=3, batch_size=512, device='cuda',
                    freshness_weight=True):
    """Train model on replay buffer with freshness weighting."""
    states, policies, values = load_replay_buffer(buffer_path)
    n = len(states)
    if n == 0:
        return 0, 0, 0, 0

    # Trim to max
    if n > max_buffer:
        states = states[-max_buffer:]
        policies = policies[-max_buffer:]
        values = values[-max_buffer:]
        n = max_buffer

    # Freshness weights: newer data sampled more often
    if freshness_weight and n > 10000:
        # Linear weight: oldest=0.5, newest=1.5
        weights = np.linspace(0.5, 1.5, n)
        weights /= weights.sum()
    else:
        weights = None

    model.train()

    for epoch in range(epochs):
        if weights is not None:
            # Weighted sampling (with replacement)
            indices = np.random.choice(n, size=n, replace=True, p=weights)
        else:
            indices = np.random.permutation(n)

        total_loss, total_p, total_v, num_batches = 0, 0, 0, 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]

            s = torch.FloatTensor(states[idx]).to(device)
            p_target = torch.FloatTensor(policies[idx]).to(device)
            v_target = torch.FloatTensor(values[idx]).unsqueeze(1).to(device)

            log_p, v = model(s)
            p_loss = -torch.mean(torch.sum(p_target * log_p, dim=1))
            v_loss = F.mse_loss(v, v_target)
            loss = p_loss + v_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_p += p_loss.item()
            total_v += v_loss.item()
            num_batches += 1

    avg_loss = total_loss / max(1, num_batches)
    avg_p = total_p / max(1, num_batches)
    avg_v = total_v / max(1, num_batches)
    return avg_loss, avg_p, avg_v, n


def eval_vs_engine(model, model_size, device, num_games=20, sims=400, time_ms=500):
    """Eval: MCTS model vs engine. Returns win rate."""
    from train_config_b import ConfigurableMCTS, get_engine_move

    mcts = ConfigurableMCTS(
        model, num_simulations=sims, c_puct=2.5,
        dirichlet_alpha=0.0, dirichlet_eps=0.0,
        device=device, use_amp=(device == 'cuda'),
    )

    engine_path = os.path.abspath(ENGINE_BINARY)
    engine_dir = os.path.dirname(engine_path)
    tmpdir = tempfile.mkdtemp()
    for f in ['nnue_weights.bin', 'egtb.bin', 'opening_book.txt']:
        src = os.path.join(engine_dir, f)
        if os.path.exists(src):
            try:
                os.symlink(src, os.path.join(tmpdir, f))
            except:
                shutil.copy2(src, os.path.join(tmpdir, f))

    proc = subprocess.Popen(
        [engine_path, "serve"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=tmpdir, text=True, bufsize=1,
    )
    while True:
        line = proc.stdout.readline().strip()
        if line == "ready" or not line:
            break

    wins, draws, losses = 0, 0, 0
    for g in range(num_games):
        mcts_is_white = (g % 2 == 0)
        game = TogyzQumalaq()
        moves = 0

        while not game.is_terminal() and moves < 200:
            cp_val = int(game.state.current_player)
            mcts_turn = (cp_val == 0 and mcts_is_white) or (cp_val == 1 and not mcts_is_white)

            if mcts_turn:
                policy = mcts.search_batch([game])[0]
                action = int(np.argmax(policy))
            else:
                action = get_engine_move(proc, game, time_ms)
                if action < 0:
                    break

            valid = game.get_valid_moves_list()
            if action not in valid:
                action = valid[0] if valid else 0
            game.make_move(action)
            moves += 1

        winner = game.get_winner()
        mcts_player = 0 if mcts_is_white else 1
        if winner == 2 or winner is None:
            draws += 1
        elif winner == mcts_player:
            wins += 1
        else:
            losses += 1

    proc.terminate()
    shutil.rmtree(tmpdir, ignore_errors=True)

    total = wins + draws + losses
    wr = (wins + 0.5 * draws) / max(1, total) * 100
    return wins, draws, losses, wr


def main():
    parser = argparse.ArgumentParser(description="AlphaZero training loop v2")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--games", type=int, default=100, help="Games per selfplay iteration")
    parser.add_argument("--sims", type=int, default=800, help="MCTS simulations per move")
    parser.add_argument("--workers", type=int, default=20, help="Rust worker threads")
    parser.add_argument("--model-size", default="large2m", help="Model architecture")
    parser.add_argument("--init-checkpoint", default=None, help="Initial PyTorch checkpoint")
    parser.add_argument("--resume", default=None, help="Resume from training checkpoint")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--train-epochs", type=int, default=3, help="Training epochs per iteration")
    parser.add_argument("--eval-interval", type=int, default=10, help="Eval every N iterations")
    parser.add_argument("--eval-games", type=int, default=20, help="Games per eval")
    parser.add_argument("--max-buffer", type=int, default=500000, help="Max replay buffer size")
    parser.add_argument("--checkpoint-dir", default="checkpoints_2m_v2", help="Checkpoint directory")
    parser.add_argument("--log", default="/tmp/train_loop_2m_v2.log", help="Log file")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    log_file = open(args.log, 'a')
    def log(msg):
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        log_file.write(line + '\n')
        log_file.flush()

    log(f"=== Training Loop v2 Start ===")
    log(f"Config: {vars(args)}")

    # Create model
    model = create_model(args.model_size, device=device)
    params = sum(p.numel() for p in model.parameters())
    log(f"Model: {args.model_size} ({params:,} params)")

    start_iter = 0
    best_wr = 0.0

    # Load checkpoint
    if args.resume and os.path.exists(args.resume):
        cp = torch.load(args.resume, map_location=device, weights_only=False)
        sd = cp.get('model_state_dict', cp)
        cleaned = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        model.load_state_dict(cleaned, strict=False)
        start_iter = cp.get('iteration', 0)
        best_wr = cp.get('best_wr', 0.0)
        log(f"Resumed from iter {start_iter}, best_wr={best_wr:.1f}%")
    elif args.init_checkpoint and os.path.exists(args.init_checkpoint):
        cp = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        sd = cp.get('model_state_dict', cp)
        cleaned = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        model.load_state_dict(cleaned, strict=False)
        log(f"Loaded init checkpoint: {args.init_checkpoint}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.iterations, eta_min=args.lr * 0.1
    )

    # Working files
    best_onnx_path = os.path.join(args.checkpoint_dir, 'best.onnx')
    current_onnx_path = os.path.join(args.checkpoint_dir, 'current.onnx')
    buffer_path = os.path.join(args.checkpoint_dir, 'replay_buffer.bin')
    accum_buffer_path = os.path.join(args.checkpoint_dir, 'accumulated_buffer.bin')

    # Initial ONNX export — this is our best model for selfplay
    export_onnx(model, best_onnx_path)
    export_onnx(model, current_onnx_path)
    log(f"Initial ONNX exported")

    total_positions = 0
    no_improvement_count = 0

    for iteration in range(start_iter, args.iterations):
        iter_start = time.time()
        log(f"\n--- Iteration {iteration+1}/{args.iterations} ---")

        # Step 1: Rust selfplay — ALWAYS use best model
        selfplay_onnx = best_onnx_path if os.path.exists(best_onnx_path) else current_onnx_path
        log(f"  Selfplay: {args.games} games, {args.sims} sims (using {'best' if selfplay_onnx == best_onnx_path else 'current'})")
        ok, positions, sp_time = rust_selfplay(
            selfplay_onnx, buffer_path,
            games=args.games, sims=args.sims, workers=args.workers,
        )
        if not ok:
            log(f"  Selfplay failed, skipping iteration")
            continue

        total_positions += positions
        gps = args.games / max(0.1, sp_time)
        log(f"  Selfplay: {positions} pos in {sp_time:.0f}s ({gps:.1f} g/s)")

        # Accumulate buffer
        if os.path.exists(accum_buffer_path):
            with open(accum_buffer_path, 'ab') as f:
                with open(buffer_path, 'rb') as src:
                    f.write(src.read())
        else:
            shutil.copy2(buffer_path, accum_buffer_path)

        # Trim accumulated buffer to max
        RECORD_SIZE = 63
        accum_size = os.path.getsize(accum_buffer_path)
        accum_records = accum_size // RECORD_SIZE
        if accum_records > args.max_buffer:
            with open(accum_buffer_path, 'rb') as f:
                f.seek((accum_records - args.max_buffer) * RECORD_SIZE)
                data = f.read()
            with open(accum_buffer_path, 'wb') as f:
                f.write(data)
            accum_records = args.max_buffer

        # Step 2: Train (on accumulated buffer with freshness weighting)
        log(f"  Training on {accum_records} positions ({args.train_epochs} epochs, lr={optimizer.param_groups[0]['lr']:.6f})...")
        loss, p_loss, v_loss, n_train = train_on_buffer(
            model, optimizer, accum_buffer_path,
            epochs=args.train_epochs, batch_size=512, device=device,
            max_buffer=args.max_buffer, freshness_weight=True,
        )
        scheduler.step()
        log(f"  Loss: {loss:.4f} (p={p_loss:.4f}, v={v_loss:.4f})")

        # Step 3: Export current ONNX (for eval comparison)
        export_onnx(model, current_onnx_path)

        # Step 4: Save checkpoint
        if (iteration + 1) % 5 == 0:
            cp_path = os.path.join(args.checkpoint_dir, f'iter_{iteration+1}.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iteration': iteration + 1,
                'best_wr': best_wr,
                'total_positions': total_positions,
            }, cp_path)
            log(f"  Checkpoint: {cp_path}")

        # Always save latest
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iteration': iteration + 1,
            'best_wr': best_wr,
            'total_positions': total_positions,
        }, os.path.join(args.checkpoint_dir, 'latest.pt'))

        # Step 5: Eval with gating
        if (iteration + 1) % args.eval_interval == 0:
            log(f"  Evaluating vs Gen7 ({args.eval_games} games, 400 sims)...")
            model.eval()
            try:
                w, d, l, wr = eval_vs_engine(
                    model, args.model_size, device,
                    num_games=args.eval_games, sims=400,
                )
                log(f"  Eval: {w}W-{d}D-{l}L = {wr:.1f}% vs Gen7")

                if wr > best_wr:
                    best_wr = wr
                    no_improvement_count = 0
                    # Update best model for selfplay
                    best_pt_path = os.path.join(args.checkpoint_dir, 'best.pt')
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'iteration': iteration + 1,
                        'best_wr': best_wr,
                    }, best_pt_path)
                    export_onnx(model, best_onnx_path)
                    log(f"  NEW BEST: {wr:.1f}% — selfplay model updated!")
                else:
                    no_improvement_count += 1
                    log(f"  No improvement (best={best_wr:.1f}%, streak={no_improvement_count})")

            except Exception as e:
                log(f"  Eval failed: {e}")

        iter_time = time.time() - iter_start
        log(f"  Iter time: {iter_time:.0f}s | Total positions: {total_positions:,}")

    log(f"\n=== Training Complete ===")
    log(f"Total positions generated: {total_positions:,}")
    log(f"Best winrate vs Gen7: {best_wr:.1f}%")
    log_file.close()


if __name__ == "__main__":
    main()
