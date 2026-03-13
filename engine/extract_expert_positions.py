#!/usr/bin/env python3
"""
Extract expert midgame positions (ply 34-40) from PlayOK games.
Output: binary file of board states for datagen starting positions.

Format per position (23 bytes):
  pits[0][0..9]: 9 bytes
  pits[1][0..9]: 9 bytes
  kazan[0]: 1 byte
  kazan[1]: 1 byte
  tuzdyk[0]: 1 byte (i8 as u8)
  tuzdyk[1]: 1 byte (i8 as u8)
  side_to_move: 1 byte
"""

import os
import re
import struct
from pathlib import Path

NUM_PITS = 9
INITIAL_STONES = 9
MIN_ELO = 1600
PLY_MIN = 20
PLY_MAX = 60


class Board:
    def __init__(self):
        self.pits = [[INITIAL_STONES]*NUM_PITS, [INITIAL_STONES]*NUM_PITS]
        self.kazan = [0, 0]
        self.tuzdyk = [-1, -1]
        self.side_to_move = 0

    def valid_moves(self):
        me = self.side_to_move
        opp = 1 - me
        opp_tuz = self.tuzdyk[opp]
        return [i for i in range(NUM_PITS) if self.pits[me][i] > 0 and opp_tuz != i]

    def can_create_tuzdyk(self, player, pit_index):
        if self.tuzdyk[player] != -1:
            return False
        if pit_index == 8:
            return False
        if self.tuzdyk[1 - player] == pit_index:
            return False
        return True

    def is_terminal(self):
        if self.kazan[0] >= 82 or self.kazan[1] >= 82:
            return True
        w_empty = all(x == 0 for x in self.pits[0])
        b_empty = all(x == 0 for x in self.pits[1])
        return w_empty or b_empty

    def make_move(self, pit_index):
        me = self.side_to_move
        opp = 1 - me
        stones = self.pits[me][pit_index]
        if stones <= 0:
            return False

        self.pits[me][pit_index] = 0
        current_pit = pit_index
        current_side = me

        if stones == 1:
            current_pit += 1
            if current_pit > 8:
                current_pit = 0
                current_side = opp
            if current_side == opp and self.tuzdyk[me] == current_pit:
                self.kazan[me] += 1
            elif current_side == me and self.tuzdyk[opp] == current_pit:
                self.kazan[opp] += 1
            else:
                self.pits[current_side][current_pit] += 1
        else:
            self.pits[current_side][current_pit] += 1
            remaining = stones - 1
            while remaining > 0:
                current_pit += 1
                if current_pit > 8:
                    current_pit = 0
                    current_side = 1 - current_side
                if current_side == opp and self.tuzdyk[me] == current_pit:
                    self.kazan[me] += 1
                elif current_side == me and self.tuzdyk[opp] == current_pit:
                    self.kazan[opp] += 1
                else:
                    self.pits[current_side][current_pit] += 1
                remaining -= 1

        is_tuzdyk_pit = (current_side == opp and self.tuzdyk[me] == current_pit) or \
                        (current_side == me and self.tuzdyk[opp] == current_pit)

        if current_side == opp and not is_tuzdyk_pit:
            count = self.pits[opp][current_pit]
            if count == 3 and self.can_create_tuzdyk(me, current_pit):
                self.tuzdyk[me] = current_pit
                self.kazan[me] += count
                self.pits[opp][current_pit] = 0
            elif count % 2 == 0 and count > 0:
                self.kazan[me] += count
                self.pits[opp][current_pit] = 0

        self.side_to_move = 1 - self.side_to_move
        return True

    def pack(self):
        """Pack board into 23 bytes."""
        buf = bytearray(23)
        for i in range(9):
            buf[i] = self.pits[0][i]
            buf[9 + i] = self.pits[1][i]
        buf[18] = self.kazan[0]
        buf[19] = self.kazan[1]
        buf[20] = self.tuzdyk[0] & 0xFF  # i8 as u8
        buf[21] = self.tuzdyk[1] & 0xFF
        buf[22] = self.side_to_move
        return bytes(buf)


def main():
    base = '/home/nurlykhan/9QumalaqV2'
    sources = [
        os.path.join(base, 'game-pars/games'),
        os.path.join(base, 'game-pars/linux_data/games'),
        os.path.join(base, 'gameNew2'),
    ]

    print(f"Extracting expert midgame positions (ply {PLY_MIN}-{PLY_MAX}, ELO >= {MIN_ELO})")

    positions = []
    seen = set()
    total_games = 0
    expert_games = 0

    for src_path in sources:
        if not os.path.exists(src_path):
            continue

        files = sorted(Path(src_path).glob('*.txt'))
        print(f"  Parsing {src_path}: {len(files)} files...")

        for i, fp in enumerate(files):
            if (i + 1) % 100000 == 0:
                print(f"    {i+1}/{len(files)}...")
            try:
                content = fp.read_text(encoding='utf-8', errors='ignore')
                parts = re.split(r'(?=\[Event\s)', content)

                for part in parts:
                    part = part.strip()
                    if not part or '[Event' not in part:
                        continue

                    # Parse headers
                    headers = {}
                    for m in re.finditer(r'\[(\w+)\s+"([^"]*)"\]', part):
                        headers[m.group(1)] = m.group(2)

                    total_games += 1

                    w_elo = int(headers.get('WhiteElo', '0') or '0')
                    b_elo = int(headers.get('BlackElo', '0') or '0')
                    if w_elo < MIN_ELO and b_elo < MIN_ELO:
                        continue

                    # Parse moves
                    lines = part.strip().split('\n')
                    move_lines = []
                    in_moves = False
                    for line in lines:
                        if in_moves:
                            move_lines.append(line.strip())
                        elif not line.startswith('[') and line.strip():
                            in_moves = True
                            move_lines.append(line.strip())

                    move_text = ' '.join(move_lines)
                    pit_moves = []
                    for m in re.finditer(r'(\d{2})(X?)(?:\((\d+)\))?', move_text):
                        pit_from = int(m.group(1)[0])
                        if 1 <= pit_from <= 9:
                            pit_moves.append(pit_from - 1)

                    if len(pit_moves) < PLY_MIN + 10:
                        continue

                    expert_games += 1

                    # Replay to extract position at ply PLY_MIN..PLY_MAX
                    board = Board()
                    valid = True
                    for ply, pit_idx in enumerate(pit_moves):
                        if pit_idx not in board.valid_moves():
                            valid = False
                            break

                        if PLY_MIN <= ply <= PLY_MAX and not board.is_terminal():
                            key = board.pack()
                            if key not in seen:
                                seen.add(key)
                                positions.append(key)

                        if not board.make_move(pit_idx):
                            valid = False
                            break

                        if board.is_terminal():
                            break

            except Exception:
                continue

    print(f"\nTotal games: {total_games}")
    print(f"Expert games (ELO >= {MIN_ELO}, >= {PLY_MAX} moves): {expert_games}")
    print(f"Unique midgame positions: {len(positions)}")

    # Write binary
    output = os.path.join(base, 'engine', 'expert_starts.bin')
    with open(output, 'wb') as f:
        f.write(struct.pack('<I', len(positions)))
        for pos in positions:
            f.write(pos)

    size_mb = os.path.getsize(output) / 1e6
    print(f"Output: {output} ({size_mb:.1f} MB, {len(positions)} positions)")


if __name__ == '__main__':
    main()
