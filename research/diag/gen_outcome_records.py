"""NNUE v2 training records (68-byte format of research/data/convert_9qum.py) with OUTCOME
labels from the clean corpus (corpus.py, real rules). Split by game (hash of id) so the
value head cannot memorise games; at most K positions per game (AlphaGo lesson), sampled
with extra weight on the middlegame/endgame (<= 90 board stones) where we lose.
Usage: gen_outcome_records.py <corpus.jsonl.gz> <out_dir> [min_elo=1700] [K=12] [val_pct=5]"""
import gzip, hashlib, json, os, random, struct, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tk2 import Board

MASK_VALUE_OUTCOME, MASK_POLICY, MASK_SCORE, MASK_OUTCOME_WIN = 4, 1, 8, 16


def rec(b, move, value, score, mask):
    pol = [0.0] * 9
    if move is not None:
        pol[move] = 1.0
        mask |= MASK_POLICY
    if value > 0.5:
        mask |= MASK_OUTCOME_WIN
    return (bytes(bytearray(b.p[0] + b.p[1])) + bytes(bytearray(b.k)) +
            struct.pack('<bb', b.t[0], b.t[1]) + struct.pack('<B', b.s) + struct.pack('<9f', *pol) +
            struct.pack('<f', value) + struct.pack('<f', score) + struct.pack('<B', mask))


def main():
    src, out = sys.argv[1], sys.argv[2]
    min_elo = int(sys.argv[3]) if len(sys.argv) > 3 else 1700
    K = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    val_pct = int(sys.argv[5]) if len(sys.argv) > 5 else 5
    only_src = sys.argv[6] if len(sys.argv) > 6 else None   # e.g. 'playok' (9qum is in data/9qum/train)
    os.makedirs(out, exist_ok=True)
    fh = {w: open(os.path.join(out, f'{w}.bin'), 'wb') for w in ('train', 'val')}
    rnd = random.Random(7)
    cnt = {'games': 0, 'train': 0, 'val': 0, 'score_valid': 0}
    for line in gzip.open(src, 'rt'):
        g = json.loads(line)
        if only_src and g['src'] != only_src:
            continue
        if min(g['we'] or 0, g['be'] or 0) < min_elo and 'mcts' not in (g['w'], g['b']):
            continue
        which = 'val' if int(hashlib.md5(g['id'].encode()).hexdigest(), 16) % 100 < val_pct else 'train'
        b = Board()
        states = []
        for m in g['moves']:
            states.append((b.copy(), m))
            b.move(m)
        term = b.result() is not None
        res = g['res']
        margin_w = b.final_margin() if term else None
        # sample <= K positions, weight 3x at <= 90 board stones, skip the first 4 plies
        cand = [(i, st) for i, st in enumerate(states) if i >= 4]
        if not cand:
            continue
        wts = [3.0 if st[0].stones() <= 90 else 1.0 for _, st in cand]
        pick = set()
        while len(pick) < min(K, len(cand)):
            pick.add(rnd.choices(range(len(cand)), weights=wts)[0])
        for j in pick:
            _, (pb, mv) = cand[j]
            value = 0.5 if res == 2 else (1.0 if res == pb.s else 0.0)
            mask = MASK_VALUE_OUTCOME
            score = 0.0
            if margin_w is not None:
                score = (margin_w if pb.s == 0 else -margin_w) / 82.0
                mask |= MASK_SCORE
                cnt['score_valid'] += 1
            fh[which].write(rec(pb, mv, value, score, mask))
            cnt[which] += 1
        cnt['games'] += 1
    for f in fh.values():
        f.close()
    print(cnt)


if __name__ == '__main__':
    main()
