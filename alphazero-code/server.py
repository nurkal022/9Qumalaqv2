#!/usr/bin/env python3
"""
Тоғызқұмалақ - Game Data Logger Server
Simple Flask server for logging game data
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
from datetime import datetime
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data directory
DATA_DIR = 'game_logs'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def get_log_file_path():
    """Get path to today's log file"""
    today = datetime.now().strftime('%Y-%m-%d')
    return os.path.join(DATA_DIR, f'games_{today}.json')

def load_today_games():
    """Load today's games from file"""
    file_path = get_log_file_path()
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading games: {e}")
            return []
    return []

def save_games(games):
    """Save games to today's file"""
    file_path = get_log_file_path()
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(games, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(games)} games to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving games: {e}")
        return False

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/games', methods=['POST'])
def save_game():
    """Save a complete game"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['id', 'timestamp', 'mode', 'moves', 'result']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Load today's games
        games = load_today_games()
        
        # Check if game already exists
        existing_index = next((i for i, g in enumerate(games) if g.get('id') == data['id']), None)
        
        if existing_index is not None:
            # Update existing game
            games[existing_index] = data
            logger.info(f"Updated game {data['id']}")
        else:
            # Add new game
            games.append(data)
            logger.info(f"Added new game {data['id']}")
        
        # Save to file
        if save_games(games):
            return jsonify({
                'success': True,
                'gameId': data['id'],
                'totalGames': len(games)
            }), 200
        else:
            return jsonify({'error': 'Failed to save game'}), 500
            
    except Exception as e:
        logger.error(f"Error saving game: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/games', methods=['GET'])
def get_games():
    """Get all games from today"""
    try:
        games = load_today_games()
        return jsonify({
            'games': games,
            'count': len(games),
            'date': datetime.now().strftime('%Y-%m-%d')
        }), 200
    except Exception as e:
        logger.error(f"Error getting games: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/games/stats', methods=['GET'])
def get_stats():
    """Get statistics from all games"""
    try:
        games = load_today_games()
        
        stats = {
            'totalGames': len(games),
            'whiteWins': 0,
            'blackWins': 0,
            'draws': 0,
            'totalMoves': 0,
            'totalDuration': 0,
            'modes': {},
            'aiLevels': {}
        }
        
        for game in games:
            if 'result' in game and game['result']:
                result = game['result']
                winner = result.get('winner', '')
                
                if winner == 'white':
                    stats['whiteWins'] += 1
                elif winner == 'black':
                    stats['blackWins'] += 1
                else:
                    stats['draws'] += 1
                
                stats['totalMoves'] += result.get('totalMoves', 0)
                stats['totalDuration'] += result.get('duration', 0)
            
            # Count by mode
            mode = game.get('mode', 'unknown')
            stats['modes'][mode] = stats['modes'].get(mode, 0) + 1
            
            # Count by AI level
            ai_level = game.get('aiLevel')
            if ai_level:
                stats['aiLevels'][ai_level] = stats['aiLevels'].get(ai_level, 0) + 1
        
        if stats['totalGames'] > 0:
            stats['avgMoves'] = round(stats['totalMoves'] / stats['totalGames'])
            stats['avgDuration'] = round(stats['totalDuration'] / stats['totalGames'] / 1000)
        
        return jsonify(stats), 200
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/games/export', methods=['GET'])
def export_games():
    """Export all games as JSON"""
    try:
        games = load_today_games()
        return jsonify({
            'games': games,
            'exportDate': datetime.now().isoformat(),
            'count': len(games)
        }), 200
    except Exception as e:
        logger.error(f"Error exporting games: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import sys
    
    # Production mode if not debug
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print("=" * 50)
    print("Тоғызқұмалақ Game Logger Server")
    print("=" * 50)
    print(f"Server starting on http://0.0.0.0:5000")
    print(f"Data directory: {os.path.abspath(DATA_DIR)}")
    print(f"Debug mode: {debug_mode}")
    print("=" * 50)
    
    app.run(host='127.0.0.1', port=5000, debug=debug_mode)

