#!/usr/bin/env python3
"""
Download master games from server and convert to training data.

Games are logged as JSONL on the server. Each game has:
- moves: list of {who, position, bestmove, source}
- result: "white_win", "black_win", "draw", "abandoned"
- human_color: which side the human played

Usage:
  python scripts/collect_master_games.py --output master_games.bin
"""

import sys, os, json, argparse, struct
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../alphazero-code/alphazero'))

def download_games(host, password, output_dir='master_games'):
    """Download game logs from server."""
    import paramiko
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, username='root', password=password, timeout=10)
    sftp = ssh.open_sftp()

    os.makedirs(output_dir, exist_ok=True)

    remote_dir = '/opt/togyzkumalaq/web/games_log/'
    for f in sftp.listdir(remote_dir):
        if f.endswith('.jsonl'):
            remote = remote_dir + f
            local = os.path.join(output_dir, f)
            print(f'  Downloading {f}...')
            sftp.get(remote, local)

    sftp.close()
    ssh.close()
    return output_dir


def parse_position_string(pos_str):
    """Parse "w0,w1,...,w8/b0,...,b8/kw,kb/tw,tb/side" into board arrays."""
    parts = pos_str.split('/')
    if len(parts) != 5:
        return None

    white_pits = [int(x) for x in parts[0].split(',')]
    black_pits = [int(x) for x in parts[1].split(',')]
    kazans = [int(x) for x in parts[2].split(',')]
    tuzdyks = [int(x) for x in parts[3].split(',')]
    side = int(parts[4])

    if len(white_pits) != 9 or len(black_pits) != 9:
        return None

    return {
        'white_pits': white_pits,
        'black_pits': black_pits,
        'kazan': kazans,
        'tuzdyk': tuzdyks,
        'side': side,
    }


def encode_position(pos, side):
    """Encode position to [7, 9] tensor (perspective-relative)."""
    state = np.zeros((7, 9), dtype=np.float32)

    if side == 0:  # White to move
        me_pits = np.array(pos['white_pits'], dtype=np.float32)
        opp_pits = np.array(pos['black_pits'], dtype=np.float32)
        me_kazan, opp_kazan = pos['kazan'][0], pos['kazan'][1]
        me_tuzdyk, opp_tuzdyk = pos['tuzdyk'][0], pos['tuzdyk'][1]
    else:  # Black to move
        me_pits = np.array(pos['black_pits'], dtype=np.float32)
        opp_pits = np.array(pos['white_pits'], dtype=np.float32)
        me_kazan, opp_kazan = pos['kazan'][1], pos['kazan'][0]
        me_tuzdyk, opp_tuzdyk = pos['tuzdyk'][1], pos['tuzdyk'][0]

    state[0] = me_pits / 50.0
    state[1] = opp_pits / 50.0
    state[2] = me_kazan / 82.0
    state[3] = opp_kazan / 82.0
    if me_tuzdyk >= 0:
        state[4, me_tuzdyk] = 1.0
    if opp_tuzdyk >= 0:
        state[5, opp_tuzdyk] = 1.0
    state[6] = 1.0 if side == 0 else 0.0

    return state


def pack_record(pos, move, value):
    """Pack into 63-byte binary record matching Rust replay buffer format."""
    record = bytearray(63)

    # Bytes 0-8: White pits
    for i in range(9):
        record[i] = pos['white_pits'][i]
    # Bytes 9-17: Black pits
    for i in range(9):
        record[9 + i] = pos['black_pits'][i]
    # Bytes 18-19: Kazan
    record[18] = pos['kazan'][0]
    record[19] = pos['kazan'][1]
    # Bytes 20-21: Tuzdyk (as i8)
    record[20] = pos['tuzdyk'][0] & 0xFF
    record[21] = pos['tuzdyk'][1] & 0xFF
    # Byte 22: Side
    record[22] = pos['side']
    # Bytes 23-58: Policy (one-hot for master move)
    policy = [0.0] * 9
    if 0 <= move < 9:
        policy[move] = 1.0
    for i in range(9):
        struct.pack_into('<f', record, 23 + i * 4, policy[i])
    # Bytes 59-62: Value
    struct.pack_into('<f', record, 59, value)

    return bytes(record)


def process_games(games_dir, output_path, min_moves=10):
    """Process JSONL game logs into binary training data."""
    all_records = []
    stats = {'total': 0, 'completed': 0, 'abandoned': 0, 'positions': 0}

    for fname in sorted(os.listdir(games_dir)):
        if not fname.endswith('.jsonl'):
            continue

        with open(os.path.join(games_dir, fname)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    game = json.loads(line)
                except json.JSONDecodeError:
                    continue

                stats['total'] += 1
                result = game.get('result', 'abandoned')

                if result == 'abandoned':
                    stats['abandoned'] += 1
                    continue
                if len(game.get('moves', [])) < min_moves:
                    continue

                stats['completed'] += 1

                # Determine game outcome
                if result == 'white_win':
                    white_value = 1.0
                elif result == 'black_win':
                    white_value = -1.0
                elif result == 'draw':
                    white_value = 0.0
                else:
                    continue

                # Extract human moves as training targets
                for move_data in game['moves']:
                    if move_data.get('who') != 'human':
                        continue

                    pos_str = move_data.get('position', '')
                    pos = parse_position_string(pos_str)
                    if pos is None:
                        continue

                    bestmove = move_data.get('bestmove', -1)
                    if bestmove < 0 or bestmove > 8:
                        continue

                    # Value from current player's perspective
                    if pos['side'] == 0:
                        value = white_value
                    else:
                        value = -white_value

                    record = pack_record(pos, bestmove, value)
                    all_records.append(record)
                    stats['positions'] += 1

    # Write binary file
    with open(output_path, 'wb') as f:
        for r in all_records:
            f.write(r)

    print(f"\nStats:")
    print(f"  Total games: {stats['total']}")
    print(f"  Completed: {stats['completed']}")
    print(f"  Abandoned: {stats['abandoned']}")
    print(f"  Master positions: {stats['positions']}")
    print(f"  Output: {output_path} ({os.path.getsize(output_path)} bytes)")

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='85.239.36.121')
    parser.add_argument('--password', default='eWNvr3_r,D4-r3')
    parser.add_argument('--output', default='master_games.bin')
    parser.add_argument('--games-dir', default='master_games')
    parser.add_argument('--skip-download', action='store_true')
    args = parser.parse_args()

    if not args.skip_download:
        print("Downloading games from server...")
        download_games(args.host, args.password, args.games_dir)

    print("\nProcessing games...")
    process_games(args.games_dir, args.output)


if __name__ == '__main__':
    main()
