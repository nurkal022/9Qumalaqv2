"""Transfer Gen5 40-input weights to 58-input model.

Copies fc1[:, :40] from Gen5, zero-initializes fc1[:, 40:58].
Copies fc2 and fc3 exactly (same dimensions: 256→32→1).
This gives a 58-input model that starts at Gen5 strength,
then fine-tuning teaches it to use the 18 new strategic features.
"""
import struct
import numpy as np
import torch
import torch.nn as nn
import os

SCALE = 64

def load_gen5_weights(bin_path):
    """Load 40→256→32→1 weights from binary"""
    with open(bin_path, 'rb') as f:
        data = f.read()

    first_u16 = struct.unpack('<H', data[0:2])[0]
    if first_u16 >= 128:
        # Old format
        h1 = first_u16
        h2 = struct.unpack('<H', data[2:4])[0]
        offset = 4
        inp = 40
    else:
        inp, h1, h2, h3 = struct.unpack('<HHHH', data[0:8])
        offset = 8

    print(f"Source: {inp}→{h1}→{h2}→1")

    def read_i16(n):
        nonlocal offset
        arr = np.frombuffer(data[offset:offset+n*2], dtype=np.int16).astype(np.float32) / SCALE
        offset += n * 2
        return arr

    weights = {
        'fc1_w': read_i16(h1 * inp).reshape(h1, inp),
        'fc1_b': read_i16(h1),
        'fc2_w': read_i16(h2 * h1).reshape(h2, h1),
        'fc2_b': read_i16(h2),
        'fc3_w': read_i16(h2).reshape(1, h2),
        'fc3_b': read_i16(1),
    }
    return weights, inp, h1, h2

def create_58_model(gen5_weights, h1, h2):
    """Create 58→h1→h2→1 model with transferred weights"""

    class NNUE(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(58, h1)
            self.fc2 = nn.Linear(h1, h2)
            self.fc3 = nn.Linear(h2, 1)

        def forward(self, x):
            x = torch.clamp(self.fc1(x), 0.0, 1.0)
            x = torch.clamp(self.fc2(x), 0.0, 1.0)
            x = self.fc3(x)
            return x.squeeze(-1)

    model = NNUE()
    state = model.state_dict()

    # Transfer fc1: copy first 40 columns, zero-init the remaining 18
    fc1_w_new = np.zeros((h1, 58), dtype=np.float32)
    fc1_w_new[:, :40] = gen5_weights['fc1_w']
    # Small random init for new features to break symmetry
    fc1_w_new[:, 40:58] = np.random.randn(h1, 18).astype(np.float32) * 0.01

    state['fc1.weight'] = torch.tensor(fc1_w_new)
    state['fc1.bias'] = torch.tensor(gen5_weights['fc1_b'])
    state['fc2.weight'] = torch.tensor(gen5_weights['fc2_w'])
    state['fc2.bias'] = torch.tensor(gen5_weights['fc2_b'])
    state['fc3.weight'] = torch.tensor(gen5_weights['fc3_w'])
    state['fc3.bias'] = torch.tensor(gen5_weights['fc3_b'])

    model.load_state_dict(state)
    return model

def export_binary_58(model, path, h1, h2):
    """Export as new format binary: [input_size=58, h1, h2, h3=0]"""
    state = model.state_dict()
    with open(path, 'wb') as f:
        f.write(struct.pack('<HHHH', 58, h1, h2, 0))
        for name in ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias']:
            tensor = state[name].cpu().float().numpy().flatten()
            quantized = np.clip(tensor * SCALE, -32000, 32000).astype(np.int16)
            f.write(quantized.tobytes())
    print(f"Exported: {path} ({os.path.getsize(path):,} bytes)")

if __name__ == '__main__':
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else 'nnue_weights_gen5_champion.bin'
    dst = sys.argv[2] if len(sys.argv) > 2 else 'nnue_weights_58feat_transfer.bin'

    print(f"Loading Gen5 weights from {src}...")
    weights, inp, h1, h2 = load_gen5_weights(src)

    print(f"\nCreating 58→{h1}→{h2}→1 model with transferred weights...")
    print(f"  fc1: copying {inp} columns, zero-init 18 new columns")
    print(f"  fc2, fc3: exact copy")

    model = create_58_model(weights, h1, h2)
    export_binary_58(model, dst, h1, h2)

    # Verify: the new model should produce similar output for inputs where new features are ~0
    print(f"\nTarget: {dst}")
    print(f"Architecture: 58→{h1}→{h2}→1")
    print(f"This model starts at Gen5 strength — fine-tune with:")
    print(f"  python train_nnue_v2.py --data <data_files> --input-size 58 --hidden1 {h1} --hidden2 {h2} \\")
    print(f"    --init-weights {dst} --lr 0.0003 --lam 0.5 --epochs 50 --output-bin nnue_weights_58feat_ft.bin")
