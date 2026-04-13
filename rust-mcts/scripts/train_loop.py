#!/usr/bin/env python3
"""
AlphaZero training loop v3 — fixes for self-play collapse.

Key changes from v2:
1. NO GATING — always use latest model for selfplay (like AlphaZero)
2. 200 sims + playout cap randomization (4x throughput)
3. Dirichlet alpha=1.1 (correct for branching factor ~9)
4. Expert data mixing (20%) to prevent policy degradation
5. Color-paired evaluation (40 games = 20 pairs)
6. Eval is monitoring only, never gates
7. Temperature: τ=1.0 for 25 moves, linear decay to 0.3

Usage:
  python scripts/train_loop.py --iterations 500 --games 200 --sims 200 --model-size large2m \
      --init-checkpoint checkpoints_2m_v2/best.pt
"""

import sys, os, argparse, subprocess, time, shutil, tempfile
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
EXPERT_DIR = os.path.join(os.path.dirname(__file__), '../../game-pars/games')


def rust_league(model_onnx, sp_output, eng_output,
                selfplay_games=120, engine_games=40, sims=200,
                workers=20, engine_workers=4, engine_time=200, batch_size=128):
    """Run league mode: selfplay + engine games."""
    env = os.environ.copy()
    env['ORT_DYLIB_PATH'] = ORT_DYLIB
    env['LD_LIBRARY_PATH'] = CUDA_LD_PATH + ':' + env.get('LD_LIBRARY_PATH', '')

    cmd = [
        os.path.abspath(RUST_BINARY),
        '--league',
        '--model', os.path.abspath(model_onnx),
        '--games', str(selfplay_games),
        '--sims', str(sims),
        '--workers', str(workers),
        '--engine-games', str(engine_games),
        '--engine-workers', str(engine_workers),
        '--engine', os.path.abspath(ENGINE_BINARY),
        '--engine-time', str(engine_time),
        '--output', os.path.abspath(sp_output),
        '--engine-output', os.path.abspath(eng_output),
        '--temp-threshold', '25',
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

    # Count positions from both output files
    positions = 0
    RECORD_SIZE = 63
    for f in [sp_output, eng_output]:
        af = os.path.abspath(f)
        if os.path.exists(af):
            positions += os.path.getsize(af) // RECORD_SIZE

    # Also parse from stderr for backward compat
    for line in result.stderr.splitlines():
        if 'positions' in line.lower() and 'Positions:' not in line:
            try:
                positions = int(line.split('Positions:')[1].strip())
            except:
                pass

    return True, positions, elapsed


def load_expert_positions(max_examples=100000):
    """Load PlayOK expert data as (states, policies, values)."""
    if not os.path.isdir(EXPERT_DIR):
        return None

    from supervised_pretrain import parse_pgn, extract_moves

    files = [os.path.join(EXPERT_DIR, f) for f in os.listdir(EXPERT_DIR) if f.endswith('.txt')]
    np.random.shuffle(files)

    examples_s, examples_p, examples_v = [], [], []

    for filepath in files:
        if len(examples_s) >= max_examples:
            break
        try:
            headers, move_text = parse_pgn(filepath)
            if headers is None:
                continue
            w_elo = int(headers.get('WhiteElo', '0'))
            b_elo = int(headers.get('BlackElo', '0'))
            if w_elo < 2000 or b_elo < 2000:
                continue

            result_str = headers.get('Result', '')
            if result_str == '1-0': white_value = 1.0
            elif result_str == '0-1': white_value = -1.0
            elif result_str == '1/2-1/2': white_value = 0.0
            else: continue

            moves = extract_moves(move_text)
            if len(moves) < 10:
                continue

            game = TogyzQumalaq()
            for ply, pit in enumerate(moves):
                valid_moves = game.get_valid_moves_list()
                if pit not in valid_moves:
                    break
                if ply >= 2:
                    state = game.encode_state()
                    cp = game.state.current_player
                    value = white_value if cp == Player.WHITE else -white_value
                    policy = np.zeros(9, dtype=np.float32)
                    policy[pit] = 1.0
                    examples_s.append(state)
                    examples_p.append(policy)
                    examples_v.append(value)
                success, winner = game.make_move(pit)
                if not success or winner is not None:
                    break
        except Exception:
            pass

    if examples_s:
        return (np.array(examples_s, dtype=np.float32),
                np.array(examples_p, dtype=np.float32),
                np.array(examples_v, dtype=np.float32))
    return None


def train_on_buffer(model, optimizer, buffer_path,
                    engine_buffer_path=None, engine_weight=3.0,
                    expert_data=None, expert_ratio=0.2,
                    max_buffer=500000, epochs=2, batch_size=512, device='cuda'):
    """Train model on replay buffer with engine data oversampling and expert mixing.

    Positions with all-zero policy (from fast playout cap) train value only.
    Engine game positions are oversampled by engine_weight factor.
    """
    states, policies, values = load_replay_buffer(buffer_path)

    # Add engine game data with oversampling
    if engine_buffer_path and os.path.exists(engine_buffer_path) and os.path.getsize(engine_buffer_path) > 0:
        eng_s, eng_p, eng_v = load_replay_buffer(engine_buffer_path)
        if len(eng_s) > 0:
            # Oversample engine data
            repeats = max(1, int(engine_weight))
            for _ in range(repeats):
                states = np.concatenate([states, eng_s])
                policies = np.concatenate([policies, eng_p])
                values = np.concatenate([values, eng_v])

    n = len(states)
    if n == 0:
        return 0, 0, 0, 0

    if n > max_buffer:
        states = states[-max_buffer:]
        policies = policies[-max_buffer:]
        values = values[-max_buffer:]
        n = max_buffer

    model.train()

    for epoch in range(epochs):
        indices = np.random.permutation(n)
        total_loss, total_p, total_v, num_batches = 0, 0, 0, 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]
            actual_batch = end - start

            s = torch.FloatTensor(states[idx]).to(device)
            p_target = torch.FloatTensor(policies[idx]).to(device)
            v_target = torch.FloatTensor(values[idx]).unsqueeze(1).to(device)

            # Mix in expert data
            if expert_data is not None:
                ex_s, ex_p, ex_v = expert_data
                n_expert = int(actual_batch * expert_ratio)
                if n_expert > 0 and len(ex_s) > 0:
                    ex_idx = np.random.choice(len(ex_s), min(n_expert, len(ex_s)), replace=False)
                    s = torch.cat([s, torch.FloatTensor(ex_s[ex_idx]).to(device)])
                    p_target = torch.cat([p_target, torch.FloatTensor(ex_p[ex_idx]).to(device)])
                    v_target = torch.cat([v_target, torch.FloatTensor(ex_v[ex_idx]).unsqueeze(1).to(device)])

            log_p, v = model(s)

            # Policy loss: only on positions with non-zero policy
            # (playout cap: fast-search positions have all-zero policy)
            policy_mask = (p_target.sum(dim=1) > 0.5).float()
            if policy_mask.sum() > 0:
                p_loss_per = -torch.sum(p_target * log_p, dim=1)
                p_loss = (p_loss_per * policy_mask).sum() / policy_mask.sum()
            else:
                p_loss = torch.tensor(0.0, device=device)

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


def eval_vs_engine_python(model, model_size, device, num_pairs=10, sims=400):
    """Python MCTS eval with color pairs."""
    from train_config_b import ConfigurableMCTS, get_engine_move

    mcts_eval = ConfigurableMCTS(
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
            try: os.symlink(src, os.path.join(tmpdir, f))
            except: shutil.copy2(src, os.path.join(tmpdir, f))

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
    pair_scores = []

    for pair_id in range(num_pairs):
        pair_score = 0
        for color in [0, 1]:
            mcts_is_white = (color == 0)
            game = TogyzQumalaq()
            moves = 0
            while not game.is_terminal() and moves < 200:
                cp_val = int(game.state.current_player)
                mcts_turn = (cp_val == 0 and mcts_is_white) or (cp_val == 1 and not mcts_is_white)
                if mcts_turn:
                    policy = mcts_eval.search_batch([game])[0]
                    action = int(np.argmax(policy))
                else:
                    action = get_engine_move(proc, game, 500)
                    if action < 0: break
                valid = game.get_valid_moves_list()
                if action not in valid:
                    action = valid[0] if valid else 0
                game.make_move(action)
                moves += 1

            winner = game.get_winner()
            mcts_player = 0 if mcts_is_white else 1
            if winner == 2 or winner is None:
                draws += 1; pair_score += 0.5
            elif winner == mcts_player:
                wins += 1; pair_score += 1.0
            else:
                losses += 1
        pair_scores.append(pair_score)

    proc.terminate()
    shutil.rmtree(tmpdir, ignore_errors=True)

    total = wins + draws + losses
    wr = (wins + 0.5 * draws) / max(1, total) * 100
    pw = sum(1 for s in pair_scores if s > 1.0)
    pd = sum(1 for s in pair_scores if s == 1.0)
    pl = sum(1 for s in pair_scores if s < 1.0)
    return wins, draws, losses, wr, pw, pd, pl


def eval_vs_engine_rust(onnx_path, num_pairs=20, eval_sims=400, engine_time=200):
    """Color-paired evaluation using Rust MCTS binary (much faster than Python).
    Returns (wins, draws, losses, winrate, pair_wins, pair_draws, pair_losses).
    """
    import json

    env = os.environ.copy()
    env['ORT_DYLIB_PATH'] = ORT_DYLIB
    env['LD_LIBRARY_PATH'] = CUDA_LD_PATH + ':' + env.get('LD_LIBRARY_PATH', '')

    cmd = [
        os.path.abspath(RUST_BINARY),
        '--eval',
        '--model', os.path.abspath(onnx_path),
        '--games', str(num_pairs),
        '--eval-sims', str(eval_sims),
        '--engine', os.path.abspath(ENGINE_BINARY),
        '--engine-time', str(engine_time),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True,
                           cwd=os.path.dirname(os.path.abspath(RUST_BINARY)),
                           env=env, timeout=600)

    if result.returncode != 0:
        raise RuntimeError(f"Rust eval failed: {result.stderr[-300:]}")

    # Parse JSON from stdout
    data = json.loads(result.stdout.strip())
    return (data['wins'], data['draws'], data['losses'], data['winrate'],
            data['pair_wins'], data['pair_draws'], data['pair_losses'])


def main():
    parser = argparse.ArgumentParser(description="AlphaZero training loop v3")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--games", type=int, default=200, help="Games per selfplay iteration")
    parser.add_argument("--sims", type=int, default=200, help="MCTS sims (full search)")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--model-size", default="large2m")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--train-epochs", type=int, default=2)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-pairs", type=int, default=20, help="Color-paired eval games (total=2x)")
    parser.add_argument("--eval-sims", type=int, default=800, help="Sims for eval (higher than selfplay)")
    parser.add_argument("--max-buffer", type=int, default=500000)
    parser.add_argument("--expert-ratio", type=float, default=0.2, help="Expert data mixing ratio")
    parser.add_argument("--checkpoint-dir", default="checkpoints_v3")
    parser.add_argument("--log", default="/tmp/train_loop_v3.log")
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

    log(f"=== Training Loop v3 (No Gating) ===")
    log(f"Config: {vars(args)}")

    model = create_model(args.model_size, device=device)
    params = sum(p.numel() for p in model.parameters())
    log(f"Model: {args.model_size} ({params:,} params)")

    start_iter = 0

    if args.resume and os.path.exists(args.resume):
        cp = torch.load(args.resume, map_location=device, weights_only=False)
        sd = cp.get('model_state_dict', cp)
        cleaned = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        model.load_state_dict(cleaned, strict=False)
        start_iter = cp.get('iteration', 0)
        log(f"Resumed from iter {start_iter}")
    elif args.init_checkpoint and os.path.exists(args.init_checkpoint):
        cp = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        sd = cp.get('model_state_dict', cp)
        cleaned = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        model.load_state_dict(cleaned, strict=False)
        log(f"Loaded init checkpoint: {args.init_checkpoint}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.iterations, eta_min=args.lr * 0.1
    )

    # Load expert data once
    log("Loading expert data...")
    expert_data = load_expert_positions(max_examples=100000)
    if expert_data is not None:
        log(f"  Expert data: {len(expert_data[0])} positions (mixing ratio={args.expert_ratio})")
    else:
        log("  No expert data found")

    # Working files
    onnx_path = os.path.join(args.checkpoint_dir, 'current.onnx')
    sp_buffer_path = os.path.join(args.checkpoint_dir, 'selfplay_buffer.bin')
    eng_buffer_path = os.path.join(args.checkpoint_dir, 'engine_buffer.bin')
    accum_buffer_path = os.path.join(args.checkpoint_dir, 'accumulated_buffer.bin')
    accum_eng_path = os.path.join(args.checkpoint_dir, 'accumulated_engine.bin')

    export_onnx(model, onnx_path)
    log(f"Initial ONNX exported")

    total_positions = 0
    best_wr = 0.0

    for iteration in range(start_iter, args.iterations):
        iter_start = time.time()
        log(f"\n--- Iteration {iteration+1}/{args.iterations} ---")

        # Step 1: League play — selfplay + engine games
        selfplay_games = int(args.games * 0.6)
        engine_games = int(args.games * 0.4)
        log(f"  League: {selfplay_games} selfplay + {engine_games} engine games ({args.sims} sims)")

        ok, positions, sp_time = rust_league(
            onnx_path, sp_buffer_path, eng_buffer_path,
            selfplay_games=selfplay_games, engine_games=engine_games,
            sims=args.sims, workers=args.workers, engine_workers=8, engine_time=100,
        )
        if not ok:
            log(f"  League failed, skipping")
            continue

        total_positions += positions
        gps = args.games / max(0.1, sp_time)
        log(f"  League: {positions} pos in {sp_time:.0f}s ({gps:.1f} g/s)")

        # Accumulate selfplay buffer
        if os.path.exists(accum_buffer_path):
            with open(accum_buffer_path, 'ab') as f:
                with open(sp_buffer_path, 'rb') as src:
                    f.write(src.read())
        else:
            shutil.copy2(sp_buffer_path, accum_buffer_path)

        # Accumulate engine buffer
        if os.path.exists(eng_buffer_path) and os.path.getsize(eng_buffer_path) > 0:
            if os.path.exists(accum_eng_path):
                with open(accum_eng_path, 'ab') as f:
                    with open(eng_buffer_path, 'rb') as src:
                        f.write(src.read())
            else:
                shutil.copy2(eng_buffer_path, accum_eng_path)

        # Trim selfplay buffer
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

        # Trim engine buffer (keep last 100K)
        if os.path.exists(accum_eng_path):
            eng_size = os.path.getsize(accum_eng_path)
            eng_records = eng_size // RECORD_SIZE
            if eng_records > 100000:
                with open(accum_eng_path, 'rb') as f:
                    f.seek((eng_records - 100000) * RECORD_SIZE)
                    data = f.read()
                with open(accum_eng_path, 'wb') as f:
                    f.write(data)

        # Step 2: Train with engine data oversampling + expert mixing
        lr_now = optimizer.param_groups[0]['lr']
        eng_recs = os.path.getsize(accum_eng_path) // RECORD_SIZE if os.path.exists(accum_eng_path) else 0
        log(f"  Training on {accum_records}+{eng_recs}eng positions ({args.train_epochs} epochs, lr={lr_now:.6f})...")
        loss, p_loss, v_loss, n_train = train_on_buffer(
            model, optimizer, accum_buffer_path,
            engine_buffer_path=accum_eng_path, engine_weight=3.0,
            expert_data=expert_data, expert_ratio=args.expert_ratio,
            epochs=args.train_epochs, batch_size=512, device=device,
            max_buffer=args.max_buffer,
        )
        scheduler.step()
        log(f"  Loss: {loss:.4f} (p={p_loss:.4f}, v={v_loss:.4f})")

        # Step 3: Export LATEST model for next selfplay (no gating!)
        export_onnx(model, onnx_path)

        # Step 4: Save checkpoint
        if (iteration + 1) % 5 == 0:
            cp_path = os.path.join(args.checkpoint_dir, f'iter_{iteration+1}.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iteration': iteration + 1,
                'total_positions': total_positions,
            }, cp_path)
            log(f"  Checkpoint: {cp_path}")

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iteration': iteration + 1,
            'total_positions': total_positions,
        }, os.path.join(args.checkpoint_dir, 'latest.pt'))

        # Step 5: Color-paired eval via Rust 1-ply (monitoring only, NOT gating)
        if (iteration + 1) % args.eval_interval == 0:
            log(f"  Eval vs Gen7 ({args.eval_pairs} pairs, Rust 1-ply)...")
            model.eval()
            try:
                w, d, l, wr, pw, pd, pl = eval_vs_engine_rust(
                    onnx_path,
                    num_pairs=args.eval_pairs, eval_sims=1,
                    engine_time=200,
                )
                log(f"  Eval: {w}W-{d}D-{l}L = {wr:.1f}% | Pairs: {pw}W-{pd}D-{pl}L")

                if wr > best_wr:
                    best_wr = wr
                    best_path = os.path.join(args.checkpoint_dir, 'best.pt')
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'iteration': iteration + 1,
                        'best_wr': best_wr,
                    }, best_path)
                    log(f"  New monitoring best: {wr:.1f}%")
            except Exception as e:
                log(f"  Eval failed: {e}")

        iter_time = time.time() - iter_start
        log(f"  Iter time: {iter_time:.0f}s | Total positions: {total_positions:,}")

    log(f"\n=== Training Complete ===")
    log(f"Total positions: {total_positions:,} | Best monitoring wr: {best_wr:.1f}%")
    log_file.close()


if __name__ == "__main__":
    main()
