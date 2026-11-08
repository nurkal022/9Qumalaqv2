"""Build one clean, deduplicated game corpus (real rules) from:
  - PlayOK PGN tg*.txt (~/game-pars/games + archive/datasets/game-pars/games); only real
    PlayOK files (tg*), so the fake-Elo yernar_apk over-the-board files are excluded
  - 9qum replays (data/9qum/games/replays.jsonl.gz)
Each game is replayed with tk2 (real rules); games with an illegal move are dropped.
Output: one JSON per line {src,id,w,b,we,be,res,moves,end,date}
  res: 0 white won, 1 black won, 2 draw (from the header / server), moves: 0-based pits,
  end: 'terminal' if the move list reaches a rules-terminal position, else 'resign/flag'.
Usage: corpus.py <out.jsonl.gz> [min_elo=1500]"""
import gzip, json, os, re, sys
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tk2 import Board

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
PGN_DIRS = [os.path.expanduser('~/game-pars/games'), os.path.join(ROOT, 'archive/datasets/game-pars/games')]
TOK = re.compile(r'^([1-9])([1-9])(\(\d+\))?X?$')
MIN_ELO = int(sys.argv[2]) if len(sys.argv) > 2 else 1500


def replay(moves):
    b = Board()
    for m in moves:
        if b.result() is not None or m not in b.legal():
            return None
        b.move(m)
    return 'terminal' if b.result() is not None else 'open'


def parse_pgn(path):
    try:
        txt = open(path, encoding='utf-8', errors='replace').read()
    except OSError:
        return None
    h = dict(re.findall(r'\[(\w+) "([^"]*)"\]', txt))
    try:
        we, be = int(h.get('WhiteElo', 0)), int(h.get('BlackElo', 0))
    except ValueError:
        return None
    special = h.get('White') == 'mcts' or h.get('Black') == 'mcts'
    if min(we, be) < MIN_ELO and not special:
        return None
    r = {'1-0': 0, '0-1': 1, '1/2-1/2': 2}.get(h.get('Result'))
    if r is None:
        return None
    body = txt[txt.rfind(']') + 1:]
    body = re.sub(r'\{[^}]*\}', ' ', body)
    moves = []
    for tok in body.split():
        if re.match(r'^\d+\.$', tok) or tok in ('1-0', '0-1', '1/2-1/2', '*'):
            continue
        m = TOK.match(tok)
        if not m:
            return None           # unknown token: don't guess
        moves.append(int(m.group(1)) - 1)
    end = replay(moves)
    if end is None or len(moves) < 10:
        return None
    return {'src': 'playok', 'id': os.path.basename(path)[:-4], 'w': h.get('White'), 'b': h.get('Black'),
            'we': we, 'be': be, 'res': r, 'moves': moves, 'end': 'terminal' if end == 'terminal' else 'resign/flag',
            'date': h.get('Date', '')}


def from_9qum(line):
    g = json.loads(line)
    meta = g.get('_meta', {})
    last = g['states'][-1]
    moves = [mv['hole'] for mv in last.get('moves', [])]
    w = g.get('winner')
    r = 2 if w == -1 else (w if w in (0, 1) else None)
    if r is None or len(moves) < 10:
        return None
    end = replay(moves)
    if end is None:
        return None
    return {'src': '9qum', 'id': g['game_id'], 'w': g.get('seat0'), 'b': g.get('seat1'),
            'we': meta.get('r0_before') or 0, 'be': meta.get('r1_before') or 0, 'res': r, 'moves': moves,
            'end': 'terminal' if end == 'terminal' else 'resign/flag', 'reason': g.get('reason'),
            'date': meta.get('finished_at', '')}


if __name__ == '__main__':
    out = sys.argv[1]
    files, seen = [], set()
    for d in PGN_DIRS:
        for f in os.listdir(d):
            if f.startswith('tg') and f.endswith('.txt') and f not in seen:
                seen.add(f)
                files.append(os.path.join(d, f))
    n = {'playok': 0, '9qum': 0}
    with gzip.open(out, 'wt') as fo, Pool(8) as pool:
        for g in pool.imap_unordered(parse_pgn, files, chunksize=500):
            if g:
                fo.write(json.dumps(g) + '\n'); n['playok'] += 1
        q = os.path.join(ROOT, 'data/9qum/games/replays.jsonl.gz')
        ids = set()
        for line in gzip.open(q, 'rt'):
            try:
                g = from_9qum(line)
            except (KeyError, ValueError, IndexError):
                continue
            if g and g['id'] not in ids:
                ids.add(g['id']); fo.write(json.dumps(g) + '\n'); n['9qum'] += 1
    print(f'files {len(files)} -> kept {n}')
