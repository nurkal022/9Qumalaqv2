"""Paired engine-vs-engine match from a suite of start positions (real-rules referee, tk2).
Each position is played twice with colours swapped. Engines run single-threaded with a
small hash table (TT_SIZE_MB=32: the serve default of 1 GB x 40 processes froze the laptop).
Usage: ee_match.py <suite.json> <A.json> <B.json> <limit: time300 | depth10> <out.json>
engine spec json: {"bin": path, "res": dir with nnue_weights.bin/egtb.bin, "env": {...},
                   "limit": optional per-engine override}"""
import json, math, os, subprocess, sys, threading, time, queue
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tk2 import Board

MAX_PLIES = 600


class Eng:
    def __init__(self, spec):
        env = dict(os.environ, ENGINE_THREADS='1', TT_SIZE_MB='32')
        env.update(spec.get('env', {}))
        self.p = subprocess.Popen([spec['bin'], 'serve'], cwd=spec['res'], env=env, text=True, bufsize=1,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        assert self.p.stdout.readline().strip() == 'ready'

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


def play(eA, eB, pos, a_side, limA, limB):
    b = Board.parse(pos)
    eA.cmd('newgame')
    eB.cmd('newgame')
    la, lb = lim_str(limA), lim_str(limB)
    bad = 0
    for _ in range(MAX_PLIES):
        r = b.result()
        if r is not None:
            return (0.5 if r == 2 else (1.0 if r == a_side else 0.0)), bad
        e, lim = (eA, la) if b.s == a_side else (eB, lb)
        out = e.cmd(f'go pos {b.to_str()} {lim} nobook').split()
        if out[0] != 'bestmove' or int(out[1]) not in b.legal():
            bad += 1
            m = b.legal()[0]
        else:
            m = int(out[1])
        b.move(m)
    return 0.5, bad


def main():
    suite = json.load(open(sys.argv[1]))
    A, B = json.load(open(sys.argv[2])), json.load(open(sys.argv[3]))
    limit, out_path = sys.argv[4], sys.argv[5]
    limA, limB = A.get('limit', limit), B.get('limit', limit)
    nw = int(os.environ.get('WORKERS', '10'))     # 2 engines per worker -> 20 processes
    jobs = queue.Queue()
    for i, s in enumerate(suite):
        stm = Board.parse(s['pos']).s
        jobs.put((i, 0, s['pos'], stm))
        jobs.put((i, 1, s['pos'], 1 - stm))
    res, lock, t0 = {}, threading.Lock(), time.time()

    def worker():
        eA, eB = Eng(A), Eng(B)
        while True:
            try:
                i, k, pos, a_side = jobs.get_nowait()
            except queue.Empty:
                break
            sc, bad = play(eA, eB, pos, a_side, limA, limB)
            with lock:
                res[(i, k)] = (sc, bad)
        eA.close()
        eB.close()

    th = [threading.Thread(target=worker) for _ in range(nw)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    n = len(suite)
    pair = [(res[(i, 0)][0] + res[(i, 1)][0]) / 2 for i in range(n)]
    sc = [res[(i, k)][0] for i in range(n) for k in (0, 1)]
    bad = sum(v[1] for v in res.values())
    m = sum(pair) / n
    se = math.sqrt(sum((p - m) ** 2 for p in pair) / (n - 1) / n)
    elo = lambda p: -400 * math.log10(1 / min(max(p, 1e-3), 1 - 1e-3) - 1)
    print(f'A={os.path.basename(sys.argv[2])}({limA}) vs B={os.path.basename(sys.argv[3])}({limB}): {2*n} games, '
          f'A score {m*100:.1f}% +-{1.96*se*100:.1f}  Elo {elo(m):+.0f} [{elo(m-1.96*se):+.0f},{elo(m+1.96*se):+.0f}]  '
          f'bad-replies {bad}  {time.time()-t0:.0f}s', flush=True)
    json.dump({'A': A, 'B': B, 'limA': limA, 'limB': limB, 'scores': sc}, open(out_path, 'w'))


if __name__ == '__main__':
    main()
