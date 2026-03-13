#!/usr/bin/env python3
"""
Autonomous Gen7 training pipeline.
Monitors datagen → fine-tune 3 seeds → match → promote → deploy → train 52-arch

Runs fully autonomously, logs everything to pipeline.log
"""
import os, sys, time, subprocess, struct, shutil, logging, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
ENGINE = os.path.join(BASE, 'target', 'release', 'togyzkumalaq-engine')
LOG = os.path.join(BASE, 'pipeline.log')
RECORD_SIZE = 26

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    handlers=[logging.FileHandler(LOG), logging.StreamHandler()]
)
log = logging.info

# ─── dataset ────────────────────────────────────────────────────────────────

class DS(Dataset):
    def __init__(self, paths, input_size=40):
        all_data = []
        for p in (paths if isinstance(paths, list) else [paths]):
            if not os.path.exists(p): continue
            n = os.path.getsize(p) // RECORD_SIZE
            with open(p, 'rb') as f:
                raw = f.read(n * RECORD_SIZE)
            all_data.append(np.frombuffer(raw, dtype=np.uint8).reshape(n, RECORD_SIZE))
        data = np.concatenate(all_data)
        n = len(data)
        log(f"  Dataset: {n:,} positions")

        pw = data[:, 0:9].astype(np.float32)
        pb = data[:, 9:18].astype(np.float32)
        kw = data[:, 18:19].astype(np.float32)
        kb = data[:, 19:20].astype(np.float32)
        tw = data[:, 20].astype(np.int8)
        tb = data[:, 21].astype(np.int8)
        side = data[:, 22]
        evals = data[:, 23:25].copy().view(np.int16).astype(np.float32).flatten()
        results = data[:, 25].astype(np.float32) / 2.0
        is_w = (side == 0)

        mp = np.where(is_w[:,None], pw, pb) / 50.0
        op = np.where(is_w[:,None], pb, pw) / 50.0
        mk = np.where(is_w[:,None], kw, kb) / 82.0
        ok = np.where(is_w[:,None], kb, kw) / 82.0
        mti = np.where(is_w, tw, tb).astype(np.int64)
        oti = np.where(is_w, tb, tw).astype(np.int64)
        mt = np.zeros((n,10), dtype=np.float32)
        ot = np.zeros((n,10), dtype=np.float32)
        for i in range(n):
            mt[i, mti[i] if mti[i]>=0 else 9] = 1.0
            ot[i, oti[i] if oti[i]>=0 else 9] = 1.0

        base = np.concatenate([mp,op,mk,ok,mt,ot], axis=1)

        if input_size == 52:
            my_s = mp.sum(1,keepdims=True)
            op_s = op.sum(1,keepdims=True)
            ma = (mp*50>0).sum(1,keepdims=True).astype(np.float32)/9
            oa = (op*50>0).sum(1,keepdims=True).astype(np.float32)/9
            mh = (mp*50>=12).sum(1,keepdims=True).astype(np.float32)/9
            oh = (op*50>=12).sum(1,keepdims=True).astype(np.float32)/9
            mw = ((mp*50>=1)&(mp*50<=2)).sum(1,keepdims=True).astype(np.float32)/9
            ow = ((op*50>=1)&(op*50<=2)).sum(1,keepdims=True).astype(np.float32)/9
            mr = mp[:,6:9].sum(1,keepdims=True)
            or_ = op[:,6:9].sum(1,keepdims=True)
            phase = ((pw.sum(1)+pb.sum(1))/162).reshape(-1,1)
            kd = (mk - ok)
            ext = np.concatenate([my_s,op_s,ma,oa,mh,oh,mw,ow,mr,or_,phase,kd],axis=1)
            self.features = np.concatenate([base, ext], axis=1)
        else:
            self.features = base

        self.ev = evals
        self.res = np.where(is_w, results, 1-results)
        stones = (pw.sum(1)+pb.sum(1))
        self.wt = np.ones(n, dtype=np.float32)
        self.wt[stones<=30] = 2.0; self.wt[stones<=15] = 3.0
        self.he = (np.abs(evals)>1).astype(np.float32)

    def __len__(self): return len(self.features)
    def __getitem__(self, i):
        return (torch.tensor(self.features[i]), torch.tensor(self.ev[i]),
                torch.tensor(self.res[i]), torch.tensor(self.wt[i]),
                torch.tensor(self.he[i]))


# ─── model ──────────────────────────────────────────────────────────────────

class NNUE(nn.Module):
    def __init__(self, in_sz, h1, h2, h3=0):
        super().__init__()
        self.h3 = h3
        self.fc1 = nn.Linear(in_sz, h1)
        self.fc2 = nn.Linear(h1, h2)
        if h3:
            self.fc3 = nn.Linear(h2, h3)
            self.fc4 = nn.Linear(h3, 1)
        else:
            self.fc3 = nn.Linear(h2, 1)
            self.fc4 = None

    def forward(self, x):
        x = torch.clamp(self.fc1(x), 0, 1)
        x = torch.clamp(self.fc2(x), 0, 1)
        if self.h3:
            x = torch.clamp(self.fc3(x), 0, 1)
            x = self.fc4(x)
        else:
            x = self.fc3(x)
        return x.squeeze(-1)


def load_base_weights(model, path):
    with open(path,'rb') as f: data=f.read()
    first = struct.unpack_from('<H', data, 0)[0]
    if first >= 128:
        h1,h2 = first, struct.unpack_from('<H',data,2)[0]
        offset,in_sz = 4,40
    else:
        in_sz,h1,h2 = first, struct.unpack_from('<H',data,2)[0], struct.unpack_from('<H',data,4)[0]
        offset = 8
    SCALE=64.0
    def rd(cnt):
        nonlocal offset
        v=np.frombuffer(data[offset:offset+cnt*2],dtype=np.int16).astype(np.float32)/SCALE
        offset+=cnt*2; return v
    fw=rd(h1*in_sz).reshape(h1,in_sz); fb=rd(h1)
    sw=rd(h2*h1).reshape(h2,h1); sb=rd(h2)
    sd=model.state_dict()
    if model.fc1.in_features==in_sz:
        sd['fc1.weight']=torch.tensor(fw); sd['fc1.bias']=torch.tensor(fb)
        sd['fc2.weight']=torch.tensor(sw); sd['fc2.bias']=torch.tensor(sb)
        ow=rd(h2); ob=rd(1)
        if not model.h3:
            sd['fc3.weight']=torch.tensor(ow.reshape(1,h2)); sd['fc3.bias']=torch.tensor(ob)
    model.load_state_dict(sd)
    log(f"  Loaded base weights ({in_sz}→{h1}→{h2}) from {os.path.basename(path)}")


def export_bin(model, path, in_sz, h1, h2, h3):
    SCALE=64
    st=model.state_dict()
    with open(path,'wb') as f:
        if h3 > 0:
            # New format: 4-layer network (52-input or similar)
            f.write(struct.pack('<HHHH', in_sz, h1, h2, h3))
            names=['fc1.weight','fc1.bias','fc2.weight','fc2.bias','fc3.weight','fc3.bias',
                   'fc4.weight','fc4.bias']
        else:
            # Old format: 3-layer network (compatible with any input size)
            f.write(struct.pack('<HH', h1, h2))
            names=['fc1.weight','fc1.bias','fc2.weight','fc2.bias','fc3.weight','fc3.bias']
        for nm in names:
            t=st[nm].cpu().float().numpy().flatten()
            f.write(np.clip(t*SCALE,-32000,32000).astype(np.int16).tobytes())
    log(f"  Exported: {os.path.basename(path)} ({os.path.getsize(path):,} bytes)")


def sig(x, K=400.0): return torch.sigmoid(x / (K/4.0))


def train_one(dataset, in_sz, h1, h2, h3, lr, epochs, K, lam, seed, base=None):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n=len(dataset); nv=min(n//10,50000)
    tr,vl=torch.utils.data.random_split(dataset,[n-nv,nv],
        generator=torch.Generator().manual_seed(seed))
    tl=DataLoader(tr,4096,shuffle=True,num_workers=4,pin_memory=True)
    vl=DataLoader(vl,8192,num_workers=4,pin_memory=True)
    model=NNUE(in_sz,h1,h2,h3).to(dev)
    if base: load_base_weights(model, base)
    opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-5)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    best_v=float('inf'); best_sd=None
    for ep in range(epochs):
        model.train()
        for feat,ev,res,wt,he in tl:
            feat,ev,res,wt,he=feat.to(dev),ev.to(dev),res.to(dev),wt.to(dev),he.to(dev)
            lam_eff=lam*he
            tgt=lam_eff*sig(ev,K)+(1-lam_eff)*res
            loss=torch.mean((sig(model(feat),K)-tgt)**2*wt)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval(); vl_=0.0; nb=0
        with torch.no_grad():
            for feat,ev,res,wt,he in vl:
                feat,ev,res,wt,he=feat.to(dev),ev.to(dev),res.to(dev),wt.to(dev),he.to(dev)
                lam_eff=lam*he
                tgt=lam_eff*sig(ev,K)+(1-lam_eff)*res
                vl_+=torch.mean((sig(model(feat),K)-tgt)**2*wt).item(); nb+=1
        av=vl_/nb
        if av<best_v: best_v=av; best_sd={k:v.clone() for k,v in model.state_dict().items()}; m='*'
        else: m=''
        if (ep+1)%5==0 or ep==0:
            log(f"    ep {ep+1:3d}/{epochs}  val={av:.6f} {m}")
    model.load_state_dict(best_sd)
    return model, best_v


def run_match(wa, wb, games=200):
    """Returns score of wa vs wb (0.0-1.0)"""
    r=subprocess.run([ENGINE,'match-nnue',wa,wb,str(games),'500'],
        capture_output=True, text=True, cwd=BASE)
    out=r.stdout+r.stderr
    for line in out.split('\n'):
        if 'A score:' in line:
            try: return float(line.split('%')[0].split()[-1])/100.0
            except: pass
    # fallback: parse "Final: A W - D - L B"
    for line in out.split('\n'):
        if 'Final:' in line and '-' in line:
            try:
                parts=line.split()
                w=int(parts[2]); d=int(parts[4]); l=int(parts[6])
                total=w+d+l
                return (w+0.5*d)/total if total>0 else 0.5
            except: pass
    log(f"  WARNING: could not parse match output:\n{out[-500:]}")
    return 0.5


def wait_for_datagen(prefix, check_interval=120):
    """Wait for {prefix}_training_data.bin to appear"""
    out_file=os.path.join(BASE,f'{prefix}_training_data.bin')
    log(f"Waiting for datagen output: {out_file}")
    while not os.path.exists(out_file):
        # Check progress from thread files
        thread_files=sorted([f for f in os.listdir(BASE) if f.startswith(prefix+'_thread_')])
        total=sum(os.path.getsize(os.path.join(BASE,f)) for f in thread_files)
        pos=total//RECORD_SIZE
        log(f"  Datagen in progress: {pos:,} positions so far...")
        time.sleep(check_interval)
    sz=os.path.getsize(out_file)
    pos=sz//RECORD_SIZE
    log(f"Datagen complete: {pos:,} positions ({sz/1e6:.1f} MB)")
    return out_file


def deploy():
    log("Deploying to server...")
    r=subprocess.run(['python3', os.path.join(ROOT,'deploy.py')],
        capture_output=True, text=True, cwd=ROOT)
    for line in (r.stdout+r.stderr).split('\n'):
        if any(k in line for k in ['Deployed','Error','Finished','Building','Active','Opening']):
            log(f"  {line.strip()}")
    return r.returncode == 0


# ════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def main():
    log("="*60)
    log("Gen7 Autonomous Training Pipeline")
    log("="*60)

    # ── Phase 1: Wait for depth-14 datagen ───────────────────────
    d14_file = wait_for_datagen('gen7_d14', check_interval=120)

    # ── Phase 2: Fine-tune Gen5 → Gen7 (3 seeds, 40-input arch) ──
    log("\n" + "="*60)
    log("Phase 2: Fine-tune Gen7 (40-256-32-1, 3 seeds)")
    log("="*60)

    champion = os.path.join(BASE, 'nnue_weights.bin')
    gen7_base = champion  # fine-tune from Gen5

    # Combine d14 data with existing gen5 data for diversity
    d10_file = os.path.join(BASE, 'gen_combined_d10.bin')
    data_files = [d14_file]
    if os.path.exists(d10_file):
        # Mix: 60% d14 (new quality) + 40% previous
        log(f"  Using: {d14_file} + {d10_file}")
        data_files.append(d10_file)

    dataset_40 = DS(data_files, input_size=40)

    seeds = [42, 123, 777]
    models_40 = {}
    for seed in seeds:
        log(f"\n  Seed {seed}...")
        out_pt = os.path.join(BASE, f'nnue_gen7_s{seed}.pt')
        out_bin = os.path.join(BASE, f'nnue_weights_gen7_s{seed}.bin')
        model, best_val = train_one(
            dataset_40, 40, 256, 32, 0,
            lr=0.0001, epochs=30, K=400.0, lam=0.5,
            seed=seed, base=gen7_base
        )
        export_bin(model, out_bin, 40, 256, 32, 0)
        models_40[seed] = (out_bin, best_val)
        log(f"  Seed {seed}: best_val={best_val:.6f}")

    # ── Phase 3: Round-robin match → pick best Gen7 ──────────────
    log("\n" + "="*60)
    log("Phase 3: Round-robin match (200 games each)")
    log("="*60)

    scores = {s: 0.0 for s in seeds}
    for i, sa in enumerate(seeds):
        for sb in seeds[i+1:]:
            wa, wb = models_40[sa][0], models_40[sb][0]
            sc = run_match(wa, wb, games=200)
            log(f"  s{sa} vs s{sb}: {sc*100:.1f}%")
            scores[sa] += sc
            scores[sb] += 1-sc

    best_seed = max(scores, key=lambda s: scores[s])
    best_bin = models_40[best_seed][0]
    log(f"\n  Best seed: {best_seed} (score {scores[best_seed]:.2f})")
    for s in seeds:
        log(f"    s{s}: {scores[s]:.2f} (val={models_40[s][1]:.6f})")

    # ── Phase 4: Test Gen7 vs Gen5 ────────────────────────────────
    log("\n" + "="*60)
    log("Phase 4: Gen7 vs Gen5 (200 games)")
    log("="*60)

    score_vs_gen5 = run_match(best_bin, champion, games=200)
    elo = -400*np.log10(1/max(score_vs_gen5,1e-6) - 1) if score_vs_gen5 > 0.5 else \
           400*np.log10(1/max(1-score_vs_gen5,1e-6) - 1)
    log(f"  Gen7 score vs Gen5: {score_vs_gen5*100:.1f}% (Elo {'+'if score_vs_gen5>0.5 else ''}{elo:.0f})")

    # ── Phase 5: Promote if better ───────────────────────────────
    if score_vs_gen5 > 0.52:  # need >52% to be confident
        log(f"\n  ✓ Gen7 IS STRONGER — promoting!")
        shutil.copy(champion, os.path.join(BASE, 'nnue_weights_gen5_backup.bin'))
        shutil.copy(best_bin, champion)
        log(f"  Backed up Gen5 → nnue_weights_gen5_backup.bin")
        log(f"  Promoted Gen7 seed {best_seed} → nnue_weights.bin")
        promoted = True
    else:
        log(f"\n  ✗ Gen7 not stronger ({score_vs_gen5*100:.1f}% ≤ 52%). Keeping Gen5.")
        promoted = False

    # Save results
    results = {
        'phase': 'gen7',
        'score_vs_gen5': score_vs_gen5,
        'elo_vs_gen5': float(elo),
        'best_seed': best_seed,
        'promoted': promoted,
        'seed_scores': {str(s): scores[s] for s in seeds},
        'seed_val_losses': {str(s): float(models_40[s][1]) for s in seeds},
    }
    with open(os.path.join(BASE, 'pipeline_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # ── Phase 6: Train 52-input architecture from scratch ────────
    log("\n" + "="*60)
    log("Phase 6: Train Gen7-52 (52→256→64→32→1, 3 seeds)")
    log("="*60)

    dataset_52 = DS(data_files, input_size=52)
    models_52 = {}
    for seed in seeds:
        log(f"\n  Seed {seed} (52-input)...")
        out_bin = os.path.join(BASE, f'nnue_weights_gen7_52_s{seed}.bin')
        model, best_val = train_one(
            dataset_52, 52, 256, 64, 32,
            lr=0.001, epochs=100, K=400.0, lam=0.5,
            seed=seed, base=None  # from scratch
        )
        export_bin(model, out_bin, 52, 256, 64, 32)
        models_52[seed] = (out_bin, best_val)
        log(f"  Seed {seed} (52): best_val={best_val:.6f}")

    # Round-robin for 52-input models
    log("\n  Round-robin (52-input):")
    scores_52 = {s: 0.0 for s in seeds}
    for i, sa in enumerate(seeds):
        for sb in seeds[i+1:]:
            sc = run_match(models_52[sa][0], models_52[sb][0], games=100)
            log(f"  s{sa} vs s{sb}: {sc*100:.1f}%")
            scores_52[sa] += sc; scores_52[sb] += 1-sc

    best_52_seed = max(scores_52, key=lambda s: scores_52[s])
    best_52_bin = models_52[best_52_seed][0]

    # Test 52-input vs current champion
    current_champ = champion  # might be Gen7 or Gen5
    score_52 = run_match(best_52_bin, current_champ, games=200)
    elo_52 = -400*np.log10(1/max(score_52,1e-6)-1) if score_52>0.5 else \
              400*np.log10(1/max(1-score_52,1e-6)-1)
    log(f"\n  Gen7-52 vs champion: {score_52*100:.1f}% (Elo {'+'if score_52>0.5 else ''}{elo_52:.0f})")

    if score_52 > 0.52:
        log(f"  ✓ 52-input IS STRONGER — promoting!")
        shutil.copy(current_champ, os.path.join(BASE, 'nnue_weights_pre52_backup.bin'))
        shutil.copy(best_52_bin, champion)
        log(f"  Promoted Gen7-52 seed {best_52_seed} → nnue_weights.bin")
        promoted = True
    else:
        log(f"  ✗ 52-input not stronger. Keeping current champion.")

    # ── Phase 7: Deploy ──────────────────────────────────────────
    log("\n" + "="*60)
    log("Phase 7: Deploying to server")
    log("="*60)
    ok = deploy()
    log(f"  Deploy {'SUCCESS' if ok else 'FAILED'}")

    log("\n" + "="*60)
    log("Pipeline complete!")
    log(f"  Gen7 (40-input): {score_vs_gen5*100:.1f}% vs Gen5 (Elo {elo:+.0f})")
    log(f"  Gen7-52 (52-input): {score_52*100:.1f}% vs champion (Elo {elo_52:+.0f})")
    log("="*60)


if __name__ == '__main__':
    main()
