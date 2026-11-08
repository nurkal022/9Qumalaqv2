"""Line A data: engine self-play OUTCOME labels from positions of strong human games.

For each sample: take a random human game from the clean real-rules corpus
(corpus.py output), stop at a random ply, optionally play one random legal move
(diversity), then let the engine play BOTH sides to the end under the real rules
(tk2 referee). Every position from the start onwards gets the playout's result
(side-to-move POV) and the final swept margin as labels, in the 68-byte NNUE v2
record format of gen_outcome_records.py. At most K positions per playout, weighted
x3 at <= 90 board stones (the phase we lose). Split train/val by the SOURCE human
game id, so positions of one human game never sit on both sides of the split.

Usage:
  selfplay_outcomes.py --corpus corpus_1500.jsonl.gz --engine-spec spec.json \
      --out DIR --games N [--workers 8] [--limit time200] [--k 12] [--val-pct 5] [--seed 1]
Output (appended, resumable by re-running with a new --seed):
  DIR/train.bin, DIR/val.bin   68-byte records
  DIR/games.jsonl               one line per playout (source, start ply, moves, result)
"""
import argparse, gzip, hashlib, json, os, random, subprocess, sys, threading, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tk2 import Board                                   # noqa: E402  real-rules referee
from gen_outcome_records import rec, MASK_VALUE_OUTCOME, MASK_SCORE   # noqa: E402

MAX_PLIES = 400


class Eng:
    def __init__(self, spec):
        env = dict(os.environ, ENGINE_THREADS='1', TT_SIZE_MB='64')
        env.update(spec.get('env', {}))
        self.p = subprocess.Popen([spec['bin'], 'serve'], cwd=spec['res'], env=env, text=True, bufsize=1,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        line = self.p.stdout.readline().strip()
        assert line == 'ready', f'engine did not start: {line!r}'

    def cmd(self, c):
        self.p.stdin.write(c + '\n')
        self.p.stdin.flush()
        return self.p.stdout.readline().strip()

    def close(self):
        try:
            self.p.kill()
        except Exception:
            pass


def lim_str(limit):
    return f'time {limit[4:]}' if limit.startswith('time') else f'depth {limit[5:]}'


def load_corpus(path, min_len=20, exclude=frozenset()):
    games = []
    for line in gzip.open(path, 'rt'):
        g = json.loads(line)
        if len(g['moves']) >= min_len and g['id'] not in exclude:
            games.append({'id': g['id'], 'moves': g['moves']})
    return games


def playout(eng, lim, games, rnd):
    g = rnd.choice(games)
    hi = min(len(g['moves']) - 4, 140)
    start = rnd.randint(6, max(6, hi))
    b = Board()
    for m in g['moves'][:start]:
        b.move(m)
    if b.result() is not None:
        return None
    rand_move = None
    if rnd.random() < 0.5:
        rand_move = rnd.choice(b.legal())
        b.move(rand_move)
        if b.result() is not None:
            return None
    eng.cmd('newgame')
    states, moves = [], []
    for _ in range(MAX_PLIES):
        r = b.result()
        if r is not None:
            return {'g': g, 'start': start, 'rand': rand_move, 'states': states, 'moves': moves,
                    'res': r, 'margin_w': b.final_margin()}
        out = eng.cmd(f'go pos {b.to_str()} {lim} nobook').split()
        legal = b.legal()
        if not out or out[0] != 'bestmove' or int(out[1]) not in legal:
            return None          # engine disagreed with the referee: drop the playout
        m = int(out[1])
        states.append((b.copy(), m))
        moves.append(m)
        b.move(m)
    return None                  # no result within MAX_PLIES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', required=True)
    ap.add_argument('--engine-spec', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--games', type=int, required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', default='time200')
    ap.add_argument('--k', type=int, default=12)
    ap.add_argument('--val-pct', type=int, default=5)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--exclude-split', default=None,
                    help='split.json with val_games to keep out of the start positions (no leakage '
                         'into a validation set built from the same games)')
    a = ap.parse_args()

    spec = json.load(open(a.engine_spec))
    exclude = frozenset()
    if a.exclude_split:
        exclude = frozenset(json.load(open(a.exclude_split))['val_games'])
    games = load_corpus(a.corpus, exclude=exclude)
    os.makedirs(a.out, exist_ok=True)
    fh = {w: open(os.path.join(a.out, f'{w}.bin'), 'ab') for w in ('train', 'val')}
    gl = open(os.path.join(a.out, 'games.jsonl'), 'a')
    lock = threading.Lock()
    cnt = {'playouts': 0, 'dropped': 0, 'train': 0, 'val': 0, 'w': 0, 'b': 0, 'd': 0}
    todo = [a.games]
    t0 = time.time()
    lim = lim_str(a.limit)
    print(f'excluded {len(exclude)} val games; corpus {len(games)} games, engine {spec["bin"]} ({spec["res"]}), limit {a.limit}', flush=True)

    def worker(wid):
        rnd = random.Random(a.seed * 1000 + wid)
        eng = Eng(spec)
        while True:
            with lock:
                if todo[0] <= 0:
                    break
                todo[0] -= 1
            po = playout(eng, lim, games, rnd)
            with lock:
                if po is None:
                    cnt['dropped'] += 1
                    continue
                which = 'val' if int(hashlib.md5(po['g']['id'].encode()).hexdigest(), 16) % 100 < a.val_pct else 'train'
                res, mw = po['res'], po['margin_w']
                cnt['wbd'[res] if res in (0, 1) else 'd'] += 1
                st = po['states']
                wts = [3.0 if s[0].stones() <= 90 else 1.0 for s in st]
                pick = set()
                while len(pick) < min(a.k, len(st)):
                    pick.add(rnd.choices(range(len(st)), weights=wts)[0])
                for j in sorted(pick):
                    pb, mv = st[j]
                    value = 0.5 if res == 2 else (1.0 if res == pb.s else 0.0)
                    score = (mw if pb.s == 0 else -mw) / 82.0
                    fh[which].write(rec(pb, mv, value, score, MASK_VALUE_OUTCOME | MASK_SCORE))
                    cnt[which] += 1
                gl.write(json.dumps({'src': po['g']['id'], 'start': po['start'], 'rand': po['rand'],
                                     'moves': po['moves'], 'res': res, 'margin_w': mw, 'split': which}) + '\n')
                cnt['playouts'] += 1
                if cnt['playouts'] % 200 == 0:
                    el = time.time() - t0
                    print(f'  {cnt}  {cnt["playouts"]/el*3600:.0f} playouts/h', flush=True)
        eng.close()

    th = [threading.Thread(target=worker, args=(i,)) for i in range(a.workers)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    for f in fh.values():
        f.close()
    gl.close()
    print(f'done {cnt} in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
