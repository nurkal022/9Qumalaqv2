"""
Generate dedicated endgame training data.

Creates random endgame positions (few stones on board, high kazan values)
and records them with material-based evaluation. This provides the NNUE
with endgame-specific positions it normally wouldn't see enough of.

Output format: same 26-byte binary format as datagen.
"""

import struct
import numpy as np
import argparse
from pathlib import Path

RECORD_SIZE = 26
NUM_PITS = 9


def material_eval(pits_w, pits_b, kazan_w, kazan_b, tuzdyk_w, tuzdyk_b, side):
    """Simple material-based evaluation from side-to-move perspective."""
    me = side
    opp = 1 - side

    my_pits = pits_w if me == 0 else pits_b
    opp_pits = pits_b if me == 0 else pits_w
    my_kazan = kazan_w if me == 0 else kazan_b
    opp_kazan = kazan_b if me == 0 else kazan_w
    my_tuzdyk = tuzdyk_w if me == 0 else tuzdyk_b
    opp_tuzdyk = tuzdyk_b if me == 0 else tuzdyk_w

    # Material (kazan difference) - most important
    mat_diff = my_kazan - opp_kazan
    score = mat_diff * 21

    # Endgame material boost
    total_kazan = kazan_w + kazan_b
    if total_kazan > 100:
        score += mat_diff * (-6)

    # Proximity to win bonus
    if my_kazan >= 70:
        score += (my_kazan - 70) * 30
    if opp_kazan >= 70:
        score -= (opp_kazan - 70) * 30

    # Tuzdyk evaluation
    if my_tuzdyk >= 0:
        score += 504
        center_bonus = {3: 3, 4: 3, 5: 3, 2: 2, 6: 2, 1: 1, 7: 1}.get(my_tuzdyk, 0)
        score += center_bonus * 63
    if opp_tuzdyk >= 0:
        score -= 504
        center_bonus = {3: 3, 4: 3, 5: 3, 2: 2, 6: 2, 1: 1, 7: 1}.get(opp_tuzdyk, 0)
        score -= center_bonus * 63

    # Pit stones
    my_pit_total = sum(my_pits)
    opp_pit_total = sum(opp_pits)
    score += (my_pit_total - opp_pit_total) * 3

    # Mobility
    my_moves = sum(1 for i in range(9) if my_pits[i] > 0 and opp_tuzdyk != i)
    opp_moves = sum(1 for i in range(9) if opp_pits[i] > 0 and my_tuzdyk != i)
    score += (my_moves - opp_moves) * 124

    return max(-3000, min(3000, score))


def pack_position(pits_w, pits_b, kazan_w, kazan_b, tuzdyk_w, tuzdyk_b, side, eval_score, result):
    """Pack position into 26-byte record."""
    buf = bytearray(26)
    for i in range(9):
        buf[i] = pits_w[i]
        buf[9 + i] = pits_b[i]
    buf[18] = kazan_w
    buf[19] = kazan_b
    buf[20] = tuzdyk_w & 0xFF
    buf[21] = tuzdyk_b & 0xFF
    buf[22] = side
    struct.pack_into('<h', buf, 23, max(-3000, min(3000, eval_score)))
    buf[25] = result
    return bytes(buf)


def generate_endgame_positions(n_positions, rng):
    """Generate random endgame positions with valid stone counts."""
    positions = []

    for _ in range(n_positions):
        # Total stones in game is always 162
        # In endgame, most stones are in kazans

        # Random kazan values (both sides have captured a lot)
        total_kazan = rng.integers(100, 160)  # Endgame: lots of stones captured

        # Split between players with some variance
        split = rng.integers(max(0, total_kazan - 82), min(82, total_kazan) + 1)
        kazan_w = min(split, 81)  # Don't exceed 81 (82 = game over)
        kazan_b = min(total_kazan - kazan_w, 81)

        remaining = 162 - kazan_w - kazan_b
        if remaining < 2 or remaining > 50:
            continue

        # Distribute remaining stones across pits
        pits_w = [0] * 9
        pits_b = [0] * 9

        for s in range(remaining):
            side = rng.integers(0, 2)
            pit = rng.integers(0, 9)
            if side == 0:
                pits_w[pit] += 1
            else:
                pits_b[pit] += 1

        # Random tuzdyk (most endgame positions have tuzdyks established)
        tuzdyk_w = -1
        tuzdyk_b = -1

        if rng.random() < 0.6:  # 60% chance of tuzdyk
            candidates_w = [i for i in range(8) if pits_b[i] > 0]  # tuzdyk on opponent's side
            if candidates_w:
                tuzdyk_w = rng.choice(candidates_w)

        if rng.random() < 0.6:
            candidates_b = [i for i in range(8) if pits_w[i] > 0 and i != tuzdyk_w]
            if candidates_b:
                tuzdyk_b = rng.choice(candidates_b)

        # Validate: at least one side must have valid moves
        side = rng.integers(0, 2)

        # Check validity
        total = sum(pits_w) + sum(pits_b) + kazan_w + kazan_b
        if total != 162:
            continue

        # Must have at least one valid move
        me_pits = pits_w if side == 0 else pits_b
        opp_tuzdyk = tuzdyk_b if side == 0 else tuzdyk_w
        has_move = any(me_pits[i] > 0 and opp_tuzdyk != i for i in range(9))
        if not has_move:
            continue

        # Evaluate
        ev = material_eval(pits_w, pits_b, kazan_w, kazan_b, tuzdyk_w, tuzdyk_b, side)

        # Determine likely result based on material
        diff = kazan_w - kazan_b
        if diff > 10:
            result = 2  # white wins
        elif diff < -10:
            result = 0  # black wins
        elif abs(diff) <= 3 and remaining < 10:
            result = 1  # draw
        else:
            # Use material advantage to guess result
            if diff > 0:
                result = 2
            elif diff < 0:
                result = 0
            else:
                result = 1

        packed = pack_position(pits_w, pits_b, kazan_w, kazan_b,
                               tuzdyk_w, tuzdyk_b, side, ev, result)
        positions.append(packed)

        # Also add the mirrored position (swap sides)
        ev_opp = material_eval(pits_w, pits_b, kazan_w, kazan_b,
                                tuzdyk_w, tuzdyk_b, 1 - side)
        packed_mirror = pack_position(pits_w, pits_b, kazan_w, kazan_b,
                                       tuzdyk_w, tuzdyk_b, 1 - side, ev_opp, result)
        positions.append(packed_mirror)

    return positions


def generate_near_winning_positions(n_positions, rng):
    """Generate positions where one side is close to winning (kazan near 82)."""
    positions = []

    for _ in range(n_positions):
        # One side is very close to winning
        winning_side = rng.integers(0, 2)
        winning_kazan = rng.integers(75, 82)  # 75-81

        # Opponent has less
        remaining_after_winner = 162 - winning_kazan
        losing_kazan = rng.integers(
            max(0, remaining_after_winner - 50),
            min(81, remaining_after_winner - 2) + 1
        )

        if winning_side == 0:
            kazan_w, kazan_b = winning_kazan, losing_kazan
        else:
            kazan_w, kazan_b = losing_kazan, winning_kazan

        remaining = 162 - kazan_w - kazan_b
        if remaining < 1 or remaining > 50:
            continue

        # Distribute stones
        pits_w = [0] * 9
        pits_b = [0] * 9

        for s in range(remaining):
            side = rng.integers(0, 2)
            pit = rng.integers(0, 9)
            if side == 0:
                pits_w[pit] += 1
            else:
                pits_b[pit] += 1

        # Tuzdyk
        tuzdyk_w = -1
        tuzdyk_b = -1
        if rng.random() < 0.5:
            candidates = [i for i in range(8) if pits_b[i] > 0]
            if candidates:
                tuzdyk_w = rng.choice(candidates)
        if rng.random() < 0.5:
            candidates = [i for i in range(8) if pits_w[i] > 0 and i != tuzdyk_w]
            if candidates:
                tuzdyk_b = rng.choice(candidates)

        total = sum(pits_w) + sum(pits_b) + kazan_w + kazan_b
        if total != 162:
            continue

        # Result: the side close to winning usually wins
        result = 2 if winning_side == 0 else 0

        # Generate for both sides to move
        for side in [0, 1]:
            me_pits = pits_w if side == 0 else pits_b
            opp_tuzdyk = tuzdyk_b if side == 0 else tuzdyk_w
            has_move = any(me_pits[i] > 0 and opp_tuzdyk != i for i in range(9))
            if not has_move:
                continue

            ev = material_eval(pits_w, pits_b, kazan_w, kazan_b,
                               tuzdyk_w, tuzdyk_b, side)
            packed = pack_position(pits_w, pits_b, kazan_w, kazan_b,
                                    tuzdyk_w, tuzdyk_b, side, ev, result)
            positions.append(packed)

    return positions


def main():
    parser = argparse.ArgumentParser(description='Generate endgame training data')
    parser.add_argument('--output', default='endgame_synthetic.bin')
    parser.add_argument('--n-endgame', type=int, default=200000,
                        help='Number of random endgame positions')
    parser.add_argument('--n-near-win', type=int, default=200000,
                        help='Number of near-winning positions')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print("=== Generating Endgame Training Data ===\n")

    # General endgame positions
    print(f"Generating {args.n_endgame:,} random endgame positions...")
    endgame_pos = generate_endgame_positions(args.n_endgame, rng)
    print(f"  Generated: {len(endgame_pos):,}")

    # Near-winning positions
    print(f"Generating {args.n_near_win:,} near-winning positions...")
    near_win_pos = generate_near_winning_positions(args.n_near_win, rng)
    print(f"  Generated: {len(near_win_pos):,}")

    # Combine and shuffle
    all_positions = endgame_pos + near_win_pos
    rng.shuffle(all_positions)

    # Write
    output = Path(args.output)
    with open(output, 'wb') as f:
        for p in all_positions:
            f.write(p)

    print(f"\nTotal positions: {len(all_positions):,}")
    print(f"Output: {output} ({output.stat().st_size / 1e6:.1f} MB)")

    # Validate
    import os
    data = np.frombuffer(open(output, 'rb').read(), dtype=np.uint8).reshape(-1, RECORD_SIZE)
    evals = data[:, 23:25].copy().view(np.int16).flatten()
    pit_sums = data[:, 0:9].sum(axis=1).astype(np.int32) + data[:, 9:18].sum(axis=1).astype(np.int32)
    print(f"Eval range: [{evals.min()}, {evals.max()}]")
    print(f"Avg board stones: {pit_sums.mean():.1f}")
    print(f"Positions with <=15 stones: {np.sum(pit_sums <= 15):,}")
    print(f"Positions with <=30 stones: {np.sum(pit_sums <= 30):,}")


if __name__ == '__main__':
    main()
