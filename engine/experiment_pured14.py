import subprocess
import sys, struct
sys.path.insert(0, '.')
from pipeline import DS, NNUE, load_base_weights, train_one, run_match
import os, logging, numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.info

def export_old_fmt(model, path):
    """Export with old 4-byte header (h1, h2) matching champion format."""
    SCALE = 64
    st = model.state_dict()
    with open(path, 'wb') as f:
        f.write(struct.pack('<HH', 256, 32))
        for nm in ['fc1.weight','fc1.bias','fc2.weight','fc2.bias','fc3.weight','fc3.bias']:
            t = st[nm].cpu().float().numpy().flatten()
            f.write(np.clip(t*SCALE, -32000, 32000).astype(np.int16).tobytes())
    log(f"  Exported: {os.path.basename(path)} ({os.path.getsize(path):,} bytes)")

BASE = '/home/nurlykhan/9QumalaqV2/engine'
d14_file = os.path.join(BASE, 'gen7_d14_training_data.bin')
champion = os.path.join(BASE, 'nnue_weights.bin')

log("=== Pure D14 Fine-tune Experiment ===")
dataset = DS([d14_file], input_size=40)

seeds = [42, 123, 777]
results = {}
for seed in seeds:
    log(f"Seed {seed}...")
    out_bin = os.path.join(BASE, f'nnue_weights_pured14_s{seed}.bin')
    model, best_val = train_one(
        dataset, 40, 256, 32, 0,
        lr=0.0001, epochs=30, K=400.0, lam=0.5,
        seed=seed, base=champion
    )
    export_old_fmt(model, out_bin)
    results[seed] = (out_bin, best_val)
    log(f"Seed {seed}: val={best_val:.6f}")

# Pick best by val_loss
best_seed = min(results, key=lambda s: results[s][1])
best_bin = results[best_seed][0]
log(f"Best seed: {best_seed}")

# Match vs champion
score = run_match(best_bin, champion, games=200)
elo = -400*np.log10(1/max(score,1e-6)-1) if score > 0.5 else 400*np.log10(1/max(1-score,1e-6)-1)
log(f"Pure D14 vs Gen5: {score*100:.1f}% (Elo {'+' if score>0.5 else ''}{elo:.0f})")

# Also try higher lambda (0.7) - more weight on eval
log("\n=== Lambda 0.7 experiment ===")
model_l7, val_l7 = train_one(
    dataset, 40, 256, 32, 0,
    lr=0.0001, epochs=30, K=400.0, lam=0.7,
    seed=42, base=champion
)
out_l7 = os.path.join(BASE, 'nnue_weights_pured14_lam70.bin')
export_old_fmt(model_l7, out_l7)
score_l7 = run_match(out_l7, champion, games=200)
elo_l7 = -400*np.log10(1/max(score_l7,1e-6)-1) if score_l7 > 0.5 else 400*np.log10(1/max(1-score_l7,1e-6)-1)
log(f"Lambda 0.7 vs Gen5: {score_l7*100:.1f}% (Elo {'+' if score_l7>0.5 else ''}{elo_l7:.0f})")

print("\n=== SUMMARY ===")
print(f"Pure D14 best (s{best_seed}): {score*100:.1f}% vs Gen5 (Elo {elo:+.0f})")
print(f"Lambda 0.7: {score_l7*100:.1f}% vs Gen5 (Elo {elo_l7:+.0f})")
