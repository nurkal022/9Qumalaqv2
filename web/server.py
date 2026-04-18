"""
Web server for Togyzkumalaq Championship Engine v1.0.
Uses persistent engine process for maximum strength (TT + game history preserved).
"""
import json
import os
import random
import subprocess
import threading
import atexit
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__)

ENGINE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'engine')
ENGINE_PATH = os.path.join(ENGINE_DIR, 'target', 'release', 'togyzkumalaq-engine')

# Game logging
GAMES_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'games_log')
os.makedirs(GAMES_LOG_DIR, exist_ok=True)

# Per-session game state tracking
game_sessions = {}  # session_id -> {moves: [], positions: [], start_time, ...}
game_sessions_lock = threading.Lock()

# Persistent engine process
engine_proc = None
engine_lock = threading.Lock()

# Opening book
BOOK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'opening_book.json')
opening_book = None


def start_engine():
    """Start the persistent engine subprocess."""
    global engine_proc
    engine_proc = subprocess.Popen(
        [ENGINE_PATH, 'serve'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        cwd=ENGINE_DIR,
    )
    line = engine_proc.stdout.readline().strip()
    if line != 'ready':
        raise RuntimeError(f"Engine did not start properly, got: {line}")
    print(f"Engine started (PID {engine_proc.pid})")


def stop_engine():
    """Stop the engine process cleanly."""
    global engine_proc
    if engine_proc and engine_proc.poll() is None:
        try:
            engine_proc.stdin.write('quit\n')
            engine_proc.stdin.flush()
            engine_proc.wait(timeout=5)
        except Exception:
            engine_proc.kill()
        print("Engine stopped")


def send_command(cmd):
    """Send a command to the engine and read one line of response."""
    global engine_proc
    if engine_proc is None or engine_proc.poll() is not None:
        print("Engine not running, restarting...")
        start_engine()
    engine_proc.stdin.write(cmd + '\n')
    engine_proc.stdin.flush()
    line = engine_proc.stdout.readline().strip()
    return line


def board_to_pos(state):
    """Convert board state dict to engine position string."""
    wp = ','.join(str(x) for x in state['pits'][0])
    bp = ','.join(str(x) for x in state['pits'][1])
    k = f"{state['kazan'][0]},{state['kazan'][1]}"
    t = f"{state['tuzdyk'][0]},{state['tuzdyk'][1]}"
    s = str(state['side_to_move'])
    return f"{wp}/{bp}/{k}/{t}/{s}"


def load_opening_book():
    """Load opening book from JSON file."""
    global opening_book
    if os.path.exists(BOOK_PATH):
        with open(BOOK_PATH) as f:
            opening_book = json.load(f)
        print(f"Opening book loaded: {len(opening_book.get('positions', {}))} positions")
    else:
        print("No opening book found (run generate_book.py to create one)")


def lookup_book_move(pos_string):
    """Look up a book move. Returns (move, book_info) or (None, None).

    Only for OPENING positions (< 20 stones played).
    Beyond that, book moves are statistical (not optimal) and cause
    the engine to lose positions it can't recover from.
    """
    if not opening_book:
        return None, None

    # Parse position to check game phase
    try:
        parts = pos_string.split('/')
        if len(parts) >= 3:
            kazan = parts[2].split(',')
            stones_played = int(kazan[0]) + int(kazan[1])
            if stones_played >= 20:
                return None, None
    except Exception:
        pass

    positions = opening_book.get('positions', {})
    entry = positions.get(pos_string)
    if not entry or entry.get('total', 0) < 3:
        return None, None

    moves = entry.get('moves', {})
    if not moves:
        return None, None

    # Weighted selection by frequency * win rate
    candidates = []
    for move_str, stats in moves.items():
        count = stats.get('count', 0)
        if count < 2:
            continue
        wins = stats.get('wins', 0)
        draws = stats.get('draws', 0)
        win_rate = (wins + draws * 0.5) / max(count, 1)
        weight = count * (0.3 + win_rate)
        candidates.append((int(move_str), weight, stats, win_rate))

    if not candidates:
        return None, None

    total_weight = sum(w for _, w, _, _ in candidates)
    r = random.random() * total_weight
    cumulative = 0
    for move, weight, stats, win_rate in candidates:
        cumulative += weight
        if cumulative >= r:
            return move, {
                'source': 'book',
                'games': stats.get('count', 0),
                'win_rate': round(win_rate, 3),
            }

    return candidates[0][0], {'source': 'book', 'games': candidates[0][2].get('count', 0)}


def get_session_id():
    """Get or create a session ID from request."""
    return request.headers.get('X-Session-Id', request.remote_addr or 'unknown')


def log_game(session_id, result=None):
    """Save completed game to log file."""
    with game_sessions_lock:
        session = game_sessions.pop(session_id, None)
    if not session or not session.get('moves'):
        return
    game_data = {
        'session_id': session_id,
        'start_time': session['start_time'],
        'end_time': datetime.utcnow().isoformat(),
        'result': result,
        'moves': session['moves'],
        'positions': session['positions'],
        'human_color': session.get('human_color'),
        'num_moves': len(session['moves']),
    }
    # Append to daily log file
    date_str = datetime.utcnow().strftime('%Y-%m-%d')
    log_file = os.path.join(GAMES_LOG_DIR, f'games_{date_str}.jsonl')
    with open(log_file, 'a') as f:
        f.write(json.dumps(game_data) + '\n')
    print(f"Game logged: {session_id}, {len(session['moves'])} moves, result={result}")


def record_move(session_id, pos, move_info, who):
    """Record a move in the current session."""
    with game_sessions_lock:
        if session_id not in game_sessions:
            game_sessions[session_id] = {
                'start_time': datetime.utcnow().isoformat(),
                'moves': [],
                'positions': [],
                'human_color': None,
            }
        session = game_sessions[session_id]
        session['positions'].append(pos)
        session['moves'].append({
            'who': who,
            'position': pos,
            **move_info,
        })


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/api/move', methods=['POST'])
def get_engine_move():
    data = request.json
    pos = board_to_pos(data['board'])
    time_ms = data.get('time_ms', 3000)
    use_book = data.get('use_book', True)
    session_id = get_session_id()

    # Try opening book first
    if use_book:
        book_move, book_info = lookup_book_move(pos)
        if book_move is not None:
            move_info = {'bestmove': book_move, 'source': 'book', 'score': 0, 'depth': 0}
            record_move(session_id, pos, move_info, 'engine')
            return jsonify({
                'bestmove': book_move,
                'score': 0,
                'depth': 0,
                'nodes': 0,
                'time_ms': 0,
                'nps': 0,
                'book': book_info,
            })

    # Engine search with persistent process
    with engine_lock:
        try:
            response = send_command(f'go time {time_ms} pos {pos}')

            if response.startswith('terminal'):
                parts = response.split()
                result_str = parts[1] if len(parts) > 1 else 'unknown'
                log_game(session_id, result=result_str)
                return jsonify({'terminal': True, 'result': result_str})

            if response.startswith('bestmove'):
                parts = response.split()
                result = {}
                i = 0
                while i < len(parts) - 1:
                    key = parts[i]
                    val = parts[i + 1]
                    if key == 'bestmove':
                        result['bestmove'] = int(val)
                    elif key == 'score':
                        result['score'] = int(val)
                    elif key == 'depth':
                        result['depth'] = int(val)
                    elif key == 'nodes':
                        result['nodes'] = int(val)
                    elif key == 'time':
                        result['time_ms'] = int(val)
                    elif key == 'nps':
                        result['nps'] = int(val)
                    i += 1
                record_move(session_id, pos, result, 'engine')
                return jsonify(result)

            if response.startswith('error'):
                return jsonify({'error': response}), 500

            return jsonify({'error': f'Unexpected: {response}'}), 500
        except Exception as e:
            return jsonify({'error': str(e)}), 500


@app.route('/api/newgame', methods=['POST'])
def new_game():
    """Reset engine state for a new game."""
    session_id = get_session_id()
    # Save previous game if any
    log_game(session_id, result='abandoned')
    with engine_lock:
        try:
            response = send_command('newgame')
            return jsonify({'status': 'ok'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500


@app.route('/api/position', methods=['POST'])
def push_position():
    """Push a position to engine's game history (after human moves)."""
    data = request.json
    pos = board_to_pos(data['board'])
    session_id = get_session_id()
    move_pit = data.get('move_pit')
    record_move(session_id, pos, {'source': 'human', 'bestmove': move_pit}, 'human')
    with engine_lock:
        try:
            response = send_command(f'position {pos}')
            return jsonify({'status': 'ok'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500


@app.route('/api/games', methods=['GET'])
def list_games():
    """API endpoint to view logged games (for analysis)."""
    games = []
    if os.path.exists(GAMES_LOG_DIR):
        for fname in sorted(os.listdir(GAMES_LOG_DIR), reverse=True):
            if fname.endswith('.jsonl'):
                fpath = os.path.join(GAMES_LOG_DIR, fname)
                with open(fpath) as f:
                    for line in f:
                        if line.strip():
                            games.append(json.loads(line))
    return jsonify({'total': len(games), 'games': games[-100:]})


atexit.register(stop_engine)

if __name__ == '__main__':
    print(f"Engine: {ENGINE_PATH}")
    load_opening_book()
    start_engine()
    print(f"Starting server at http://localhost:8080")
    app.run(host='0.0.0.0', port=8080, debug=False)
