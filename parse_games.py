"""
PlayOK Togyz Kumalak Game Record Parser

Parses PGN-like game records from PlayOK, validates moves through
the game engine, and outputs clean training data.

Move notation: 2 digits where:
  - First digit (1-9) = pit played from
  - Second digit (1-9) = landing position (informational)
  - (number) = cumulative kazan total after capture
  - X = tuzdyk was created
  - {zero} = game ended by stone exhaustion
"""

import re
import sys
import os
import json
import hashlib
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'alphazero-code', 'alphazero'))
from game import TogyzQumalaq, Player


@dataclass
class ParsedGame:
    white: str
    black: str
    result: str  # "1-0", "0-1", "1/2-1/2"
    white_elo: int
    black_elo: int
    date: str
    time_control: int
    moves: List[int]  # 0-indexed pit indices
    num_moves: int
    source_file: str
    valid: bool
    truncated: bool = False  # True if trailing moves were removed
    error: Optional[str] = None


def parse_pgn_file(filepath: str) -> List[dict]:
    """Parse a PlayOK PGN file into list of raw game dicts."""
    games = []
    current_headers = {}
    move_lines = []
    in_moves = False

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        if not line:
            if in_moves and move_lines:
                current_headers['moves_text'] = ' '.join(move_lines)
                games.append(current_headers)
                current_headers = {}
                move_lines = []
                in_moves = False
            elif in_moves:
                if current_headers:
                    games.append(current_headers)
                current_headers = {}
                move_lines = []
                in_moves = False
            continue

        header_match = re.match(r'^\[(\w+)\s+"(.*)"\]$', line)
        if header_match:
            if in_moves and move_lines:
                current_headers['moves_text'] = ' '.join(move_lines)
                games.append(current_headers)
                current_headers = {}
                move_lines = []
                in_moves = False
            key, value = header_match.groups()
            current_headers[key] = value
        else:
            in_moves = True
            move_lines.append(line)

    if move_lines:
        current_headers['moves_text'] = ' '.join(move_lines)
        games.append(current_headers)
    elif current_headers:
        games.append(current_headers)

    return games


def extract_moves(move_text: str) -> Tuple[List[int], str]:
    """
    Extract pit indices from PlayOK move text.

    Returns:
        (moves, result) where moves is list of 0-indexed pit indices
    """
    result = None
    result_match = re.search(r'(1-0|0-1|1/2-1/2)\s*(\{.*?\})?\s*$', move_text)
    if result_match:
        result = result_match.group(1)
        move_text = move_text[:result_match.start()]

    move_text = re.sub(r'\d+\.', '', move_text)
    tokens = move_text.split()

    moves = []
    for token in tokens:
        match = re.match(r'^(\d)(\d)(?:\(\d+\))?X?$', token)
        if match:
            pit = int(match.group(1))
            if 1 <= pit <= 9:
                moves.append(pit - 1)

    return moves, result


def validate_game(moves: List[int], expected_result: str,
                  max_trailing: int = 5) -> Tuple[bool, bool, List[int], Optional[str]]:
    """
    Replay game through engine to validate moves.
    If game ends early with <= max_trailing remaining moves, truncate and accept.

    Returns:
        (valid, truncated, validated_moves, error_message)
    """
    game = TogyzQumalaq()

    for i, pit_index in enumerate(moves):
        valid_moves_list = game.get_valid_moves_list()

        if pit_index not in valid_moves_list:
            return False, False, moves[:i], \
                f"Move {i+1}: pit {pit_index+1} invalid, valid={[m+1 for m in valid_moves_list]}"

        success, winner = game.make_move(pit_index)
        if not success:
            return False, False, moves[:i], f"Move {i+1}: pit {pit_index+1} failed"

        if winner is not None and i < len(moves) - 1:
            remaining = len(moves) - i - 1
            if remaining <= max_trailing:
                # Trailing moves after game ended — truncate and accept
                return True, True, moves[:i+1], None
            else:
                return False, False, moves[:i+1], \
                    f"Move {i+1}: game ended early (winner={winner}), {remaining} moves remaining"

    return True, False, moves, None


def game_fingerprint(moves: List[int], white: str, black: str) -> str:
    """Create a fingerprint to detect duplicate games across files."""
    key = f"{white}|{black}|{''.join(str(m) for m in moves[:20])}"
    return hashlib.md5(key.encode()).hexdigest()


def process_all_files(file_paths: List[str], min_elo: int = 0, min_moves: int = 5) -> dict:
    """
    Process all game record files with deduplication.

    Returns:
        Summary statistics and clean game data
    """
    all_games = []
    seen_fingerprints = set()
    stats = defaultdict(int)
    elo_distribution = defaultdict(int)

    for filepath in file_paths:
        filename = os.path.basename(filepath)
        print(f"\nProcessing: {filename}")

        raw_games = parse_pgn_file(filepath)
        stats[f'raw_{filename}'] = len(raw_games)
        print(f"  Raw games parsed: {len(raw_games)}")

        file_valid = 0
        file_invalid = 0
        file_dupes = 0
        file_truncated = 0

        for raw in raw_games:
            stats['total_raw'] += 1

            white = raw.get('White', '?')
            black = raw.get('Black', '?')
            result = raw.get('Result', '?')
            date = raw.get('Date', '?')

            try:
                white_elo = int(raw.get('WhiteElo', 0))
            except ValueError:
                white_elo = 0
            try:
                black_elo = int(raw.get('BlackElo', 0))
            except ValueError:
                black_elo = 0
            try:
                time_control = int(raw.get('TimeControl', 0))
            except ValueError:
                time_control = 0

            move_text = raw.get('moves_text', '')
            if not move_text:
                stats['no_moves'] += 1
                continue

            moves, parsed_result = extract_moves(move_text)
            if parsed_result:
                result = parsed_result

            if len(moves) < min_moves:
                stats['too_short'] += 1
                continue

            # Deduplication
            fp = game_fingerprint(moves, white, black)
            if fp in seen_fingerprints:
                stats['duplicates'] += 1
                file_dupes += 1
                continue
            seen_fingerprints.add(fp)

            # ELO filter
            max_elo = max(white_elo, black_elo)
            if max_elo > 0 and max_elo < min_elo:
                stats['below_elo'] += 1
                continue

            # Validate through game engine (with trailing move truncation)
            valid, truncated, validated_moves, error = validate_game(moves, result)

            game = ParsedGame(
                white=white,
                black=black,
                result=result,
                white_elo=white_elo,
                black_elo=black_elo,
                date=date,
                time_control=time_control,
                moves=validated_moves,
                num_moves=len(validated_moves),
                source_file=filename,
                valid=valid,
                truncated=truncated,
                error=error
            )

            if valid:
                file_valid += 1
                stats['valid'] += 1
                if truncated:
                    file_truncated += 1
                    stats['truncated'] += 1
                elo_bucket = (max_elo // 100) * 100
                elo_distribution[elo_bucket] += 1
            else:
                file_invalid += 1
                stats['invalid'] += 1

            all_games.append(game)

        print(f"  Valid: {file_valid} (truncated: {file_truncated}), Invalid: {file_invalid}, Dupes: {file_dupes}")

    return {
        'games': all_games,
        'stats': dict(stats),
        'elo_distribution': dict(sorted(elo_distribution.items())),
    }


def assign_values(game: ParsedGame) -> List[dict]:
    """
    Create training examples from a validated game.
    Each position gets a value based on game outcome from current player's perspective.
    """
    if not game.valid or game.result not in ('1-0', '0-1', '1/2-1/2'):
        return []

    examples = []
    g = TogyzQumalaq()

    for i, pit_index in enumerate(game.moves):
        state = g.get_state()
        encoded = g.encode_state()
        current_player = int(state.current_player)

        # Create policy (one-hot for the played move)
        policy = np.zeros(9, dtype=np.float32)
        policy[pit_index] = 1.0

        # Value from current player's perspective
        if game.result == '1/2-1/2':
            value = 0.0
        elif game.result == '1-0':
            value = 1.0 if current_player == 0 else -1.0
        else:  # 0-1
            value = 1.0 if current_player == 1 else -1.0

        examples.append({
            'state': encoded,
            'policy': policy,
            'value': value,
        })

        g.make_move(pit_index)

    return examples


def save_training_data(games: List[ParsedGame], output_path: str, min_elo: int = 0):
    """Save validated games as numpy training data."""
    all_states = []
    all_policies = []
    all_values = []

    for game in games:
        if not game.valid:
            continue
        max_elo = max(game.white_elo, game.black_elo)
        if max_elo > 0 and max_elo < min_elo:
            continue

        examples = assign_values(game)
        for ex in examples:
            all_states.append(ex['state'])
            all_policies.append(ex['policy'])
            all_values.append(ex['value'])

    if not all_states:
        print("No training examples to save!")
        return 0

    states = np.array(all_states)
    policies = np.array(all_policies)
    values = np.array(all_values)

    np.savez_compressed(output_path,
                        states=states,
                        policies=policies,
                        values=values)

    print(f"\nSaved training data to {output_path}")
    print(f"  States shape: {states.shape}")
    print(f"  Policies shape: {policies.shape}")
    print(f"  Values shape: {values.shape}")
    print(f"  Value distribution: wins={np.sum(values > 0)}, losses={np.sum(values < 0)}, draws={np.sum(values == 0)}")

    return len(all_states)


def save_games_json(games: List[ParsedGame], output_path: str):
    """Save parsed games as JSON for inspection."""
    data = []
    for g in games:
        d = {
            'white': g.white,
            'black': g.black,
            'result': g.result,
            'white_elo': g.white_elo,
            'black_elo': g.black_elo,
            'date': g.date,
            'time_control': g.time_control,
            'moves': [m + 1 for m in g.moves],  # 1-indexed for readability
            'num_moves': g.num_moves,
            'source_file': g.source_file,
            'valid': g.valid,
            'truncated': g.truncated,
            'error': g.error,
        }
        data.append(d)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(data)} games to {output_path}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    game_files = [
        os.path.join(base_dir, 'mcts.txt'),
        os.path.join(base_dir, 'mcts 2.txt'),
        os.path.join(base_dir, 'mcts 3.txt'),
        os.path.join(base_dir, 'mcts 4.txt'),
        os.path.join(base_dir, 'temirtau.txt'),
    ]

    existing_files = [f for f in game_files if os.path.exists(f)]
    missing_files = [f for f in game_files if not os.path.exists(f)]

    if missing_files:
        print(f"Warning: Missing files: {[os.path.basename(f) for f in missing_files]}")

    print(f"Processing {len(existing_files)} game record files...")
    print("=" * 60)

    result = process_all_files(existing_files, min_elo=0, min_moves=5)

    games = result['games']
    stats = result['stats']

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total raw games:     {stats.get('total_raw', 0)}")
    print(f"Duplicates removed:  {stats.get('duplicates', 0)}")
    print(f"Too short (<5):      {stats.get('too_short', 0)}")
    print(f"Valid games:         {stats.get('valid', 0)}")
    print(f"  (truncated):       {stats.get('truncated', 0)}")
    print(f"Invalid games:       {stats.get('invalid', 0)}")

    valid_games = [g for g in games if g.valid]

    if valid_games:
        avg_moves = np.mean([g.num_moves for g in valid_games])
        elos = [max(g.white_elo, g.black_elo) for g in valid_games if max(g.white_elo, g.black_elo) > 0]
        avg_elo = np.mean(elos) if elos else 0
        results_count = defaultdict(int)
        for g in valid_games:
            results_count[g.result] += 1

        print(f"\nValid game statistics:")
        print(f"  Average moves:    {avg_moves:.1f}")
        print(f"  Average max ELO:  {avg_elo:.0f}")
        print(f"  Results: W={results_count.get('1-0', 0)}, B={results_count.get('0-1', 0)}, D={results_count.get('1/2-1/2', 0)}")

        # Unique players
        players = set()
        for g in valid_games:
            players.add(g.white)
            players.add(g.black)
        print(f"  Unique players:   {len(players)}")

        print(f"\nELO distribution:")
        for elo, count in sorted(result['elo_distribution'].items()):
            bar = '#' * min(count, 50)
            print(f"  {elo:4d}-{elo+99:4d}: {count:4d} {bar}")

    # Show invalid game examples
    invalid_games = [g for g in games if not g.valid]
    if invalid_games:
        print(f"\nRemaining invalid games ({len(invalid_games)}):")
        for g in invalid_games[:5]:
            print(f"  {g.white} vs {g.black} ({g.white_elo}/{g.black_elo}): {g.error}")
        if len(invalid_games) > 5:
            print(f"  ... and {len(invalid_games) - 5} more")

    # Save outputs
    output_dir = os.path.join(base_dir, 'parsed_games')
    os.makedirs(output_dir, exist_ok=True)

    save_games_json(valid_games, os.path.join(output_dir, 'valid_games.json'))

    # Training data - all valid games
    total = save_training_data(valid_games, os.path.join(output_dir, 'training_all.npz'))

    # Training data - high ELO (2000+)
    high_elo = [g for g in valid_games if max(g.white_elo, g.black_elo) >= 2000]
    if high_elo:
        n = save_training_data(high_elo, os.path.join(output_dir, 'training_elo2000.npz'))
        print(f"\nHigh ELO (2000+): {len(high_elo)} games, {n} positions")

    # Training data - top ELO (2300+)
    top_elo = [g for g in valid_games if max(g.white_elo, g.black_elo) >= 2300]
    if top_elo:
        n = save_training_data(top_elo, os.path.join(output_dir, 'training_elo2300.npz'))
        print(f"Top ELO (2300+): {len(top_elo)} games, {n} positions")

    print(f"\nAll outputs saved to: {output_dir}/")


if __name__ == '__main__':
    main()
