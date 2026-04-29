#!/usr/bin/env python3
"""Gen8 LR sweep: test lr=0.0003 and lr=0.00005 on d16 data."""
import sys, os, struct, logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import DS, sig

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.info
BASE = os.path.dirname(os.path.abspath(__file__))

class NNUE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(40, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 1)
    def forward(self, x):
        x = torch.clamp(self.fc1(x), 0, 1)
        x = torch.clamp(self.fc2(x), 0, 1)
        return self.fc3(x).squeeze(-1)

def load_weights(model, path):
    with open(path, 'rb') as f: data = f.read()
    first = struct.unpack_from('<H', data, 0)[0]
    h1, h2 = first, struct.unpack_from('<H', data, 2)[0]
    offset = 4; SCALE = 64.0
    def rd(cnt):
        nonlocal offset
        v = np.frombuffer(data[offset:offset+cnt*2], dtype=np.int16).astype(np.float32)/SCALE
        offset += cnt*2; return v
    sd = model.state_dict()
    sd['fc1.weight']=torch.tensor(rd(h1*40).reshape(h1,40))
    sd['fc1.bias']=torch.tensor(rd(h1))
    sd['fc2.weight']=torch.tensor(rd(h2*h1).reshape(h2,h1))
    sd['fc2.bias']=torch.tensor(rd(h2))
    sd['fc3.weight']=torch.tensor(rd(h2).reshape(1,h2))
    sd['fc3.bias']=torch.tensor(rd(1))
    model.load_state_dict(sd)

def export_bin(model, path):
    SCALE=64; sd=model.state_dict()
    with open(path,'wb') as f:
        f.write(struct.pack('<HH',256,32))
        for n in ['fc1.weight','fc1.bias','fc2.weight','fc2.bias','fc3.weight','fc3.bias']:
            t=sd[n].cpu().float().numpy().flatten()
            f.write(np.clip(t*SCALE,-32000,32000).astype(np.int16).tobytes())

def train_one(name, lr, epochs, data_files, champion, seed=42):
    log(f"\n=== {name} (lr={lr}, ep={epochs}, seed={seed}) ===")
    dataset = DS(data_files, input_size=40)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed); np.random.seed(seed)
    model = NNUE().to(dev)
    load_weights(model, champion)
    n=len(dataset); nv=min(n//10,50000)
    tr,vl=torch.utils.data.random_split(dataset,[n-nv,nv],generator=torch.Generator().manual_seed(seed))
    tl=DataLoader(tr,4096,shuffle=True,num_workers=4,pin_memory=True)
    vl_l=DataLoader(vl,8192,num_workers=4,pin_memory=True)
    opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-5)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    K=400.0; lam=0.5; best_v=float('inf'); best_sd=None
    for ep in range(epochs):
        model.train()
        for feat,ev,res,wt,he in tl:
            feat,ev,res,wt,he=feat.to(dev),ev.to(dev),res.to(dev),wt.to(dev),he.to(dev)
            tgt=lam*he*sig(ev,K)+(1-lam*he)*res
            loss=torch.mean((sig(model(feat),K)-tgt)**2*wt)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval(); vl_=0; nb=0
        with torch.no_grad():
            for feat,ev,res,wt,he in vl_l:
                feat,ev,res,wt,he=feat.to(dev),ev.to(dev),res.to(dev),wt.to(dev),he.to(dev)
                tgt=lam*he*sig(ev,K)+(1-lam*he)*res
                vl_+=torch.mean((sig(model(feat),K)-tgt)**2*wt).item(); nb+=1
        av=vl_/nb
        if av<best_v: best_v=av; best_sd={k:v.clone() for k,v in model.state_dict().items()}; m='*'
        else: m=''
        if (ep+1)%10==0 or ep==0: log(f"  ep {ep+1:3d}/{epochs} val={av:.6f} {m}")
    model.load_state_dict(best_sd)
    out=os.path.join(BASE,f'nnue_weights_{name}.bin')
    export_bin(model,out)
    log(f"  Best val={best_v:.6f}, exported {os.path.basename(out)}")
    # Match
    import subprocess
    ENGINE=os.path.join(BASE,'target','release','togyzkumalaq-engine')
    r=subprocess.run([ENGINE,'match-nnue',out,champion,'200','500'],capture_output=True,text=True,cwd=BASE,timeout=7200)
    for line in (r.stdout+r.stderr).split('\n'):
        if 'Final' in line or 'Elo' in line or 'score' in line:
            log(f"  {line.strip()}")
    return best_v

def main():
    champion=os.path.join(BASE,'nnue_weights.bin')
    d16=os.path.join(BASE,'gen8_d16_training_data.bin')
    d10=os.path.join(BASE,'gen_combined_d10.bin')
    
    # Higher LR
    train_one('gen8_lr3', 0.0003, 30, [d16], champion, seed=42)
    # Lower LR, more epochs
    train_one('gen8_lr05_60ep', 0.00005, 60, [d16], champion, seed=42)
    # Combined d10+d16, higher LR
    files = [d16]
    if os.path.exists(d10): files.append(d10)
    train_one('gen8_comb_lr3', 0.0003, 30, files, champion, seed=42)
    
    log("\n=== LR sweep done ===")

if __name__=='__main__':
    main()
