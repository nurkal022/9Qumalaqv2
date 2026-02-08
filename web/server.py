"""
Web server for Togyz Kumalak AI Demo.
Serves the game UI and proxies moves to the Rust engine.
"""
import json
import os
import subprocess
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__)

ENGINE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'engine')
ENGINE_PATH = os.path.join(ENGINE_DIR, 'target', 'release', 'togyzkumalaq-engine')


def board_to_pos(state):
    """Convert board state dict to engine position string."""
    wp = ','.join(str(x) for x in state['pits'][0])
    bp = ','.join(str(x) for x in state['pits'][1])
    k = f"{state['kazan'][0]},{state['kazan'][1]}"
    t = f"{state['tuzdyk'][0]},{state['tuzdyk'][1]}"
    s = str(state['side_to_move'])
    return f"{wp}/{bp}/{k}/{t}/{s}"


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/api/move', methods=['POST'])
def get_engine_move():
    data = request.json
    pos = board_to_pos(data['board'])
    time_ms = data.get('time_ms', 3000)

    try:
        result = subprocess.run(
            [ENGINE_PATH, 'analyze', pos, str(time_ms)],
            capture_output=True, text=True, timeout=30,
            cwd=ENGINE_DIR,
        )
        output = result.stdout.strip()
        if not output:
            return jsonify({'error': 'Engine produced no output', 'stderr': result.stderr}), 500
        return jsonify(json.loads(output))
    except subprocess.TimeoutExpired:
        return jsonify({'error': 'Engine timed out'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print(f"Engine: {ENGINE_PATH}")
    print(f"Starting server at http://localhost:8080")
    app.run(host='0.0.0.0', port=8080, debug=False)
