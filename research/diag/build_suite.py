"""Positional/endgame test suite from the clean corpus (corpus.py): for strong games, take
the first position with MAXST-12..MAXST stones on the board (the phase where our engine
collapses), keep open ones (|kazan diff| <= KD), dedupe, sample N.
Usage: build_suite.py <corpus.jsonl.gz> <out.json> [N=240] [MAXST=50] [KD=10] [min_elo=1900]"""
import gzip, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tk2 import Board

src, out = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 240
MAXST = int(sys.argv[4]) if len(sys.argv) > 4 else 50
KD = int(sys.argv[5]) if len(sys.argv) > 5 else 10
MIN_ELO = int(sys.argv[6]) if len(sys.argv) > 6 else 1900
cands, seen = [], set()
for line in gzip.open(src, 'rt'):
    g = json.loads(line)
    strong = min(g['we'] or 0, g['be'] or 0) >= MIN_ELO or 'mcts' in (g['w'], g['b'])
    if not strong:
        continue
    b = Board()
    for ply, m in enumerate(g['moves']):
        b.move(m)
        if b.result() is not None:
            break
        st = b.stones()
        if st <= MAXST:
            if st >= MAXST - 12 and abs(b.k[0] - b.k[1]) <= KD and b.key() not in seen:
                seen.add(b.key())
                cands.append({'pos': b.to_str(), 'src': g['src'], 'game': g['id'], 'ply': ply + 1,
                              'stones': st, 'kd': b.k[0] - b.k[1]})
            break
random.seed(2027)
random.shuffle(cands)
suite = cands[:N]
json.dump(suite, open(out, 'w'))
by = {}
for c in suite:
    by[c['src']] = by.get(c['src'], 0) + 1
print(f'candidates {len(cands)} -> suite {len(suite)} {by}')
