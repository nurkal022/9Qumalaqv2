#!/usr/bin/env python3
"""Comprehensive analysis of all parsed Togyz Kumalak games."""

import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
import json

def parse_game(text):
    """Parse a single game from PGN-like text."""
    game = {}
    # Parse headers
    for m in re.finditer(r'\[(\w+)\s+"([^"]*)"\]', text):
        game[m.group(1)] = m.group(2)

    # Parse moves - everything after the last header line
    header_end = 0
    for m in re.finditer(r'\][^\[]*$', text):
        pass
    lines = text.strip().split('\n')
    move_lines = []
    in_moves = False
    for line in lines:
        if in_moves:
            move_lines.append(line.strip())
        elif not line.startswith('[') and line.strip():
            in_moves = True
            move_lines.append(line.strip())

    move_text = ' '.join(move_lines)
    # Remove result from end
    move_text = re.sub(r'\s*(1-0|0-1|1/2-1/2|\*)\s*$', '', move_text)

    # Extract individual moves
    # Format: "1. 43(10) 98 2. 78(20) 55(12)"
    # Each number is a 2-digit move: first digit = pit (1-9), second digit = pit (1-9)
    # (xx) suffix = score after move, X = tuzdyk
    moves = []
    tuzdyk_count = 0
    for m in re.finditer(r'(\d{2})(X?)(?:\((\d+)\))?', move_text):
        move_str = m.group(1)
        is_tuzdyk = m.group(2) == 'X'
        score = m.group(3)
        moves.append({
            'move': move_str,
            'pit': int(move_str[0]) if move_str[0].isdigit() else None,
            'tuzdyk': is_tuzdyk,
            'score': int(score) if score else None
        })
        if is_tuzdyk:
            tuzdyk_count += 1

    game['moves'] = moves
    game['num_moves'] = len(moves)
    game['tuzdyk_count'] = tuzdyk_count

    return game

def parse_file(filepath):
    """Parse all games from a file (may contain multiple games)."""
    games = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except:
        return games

    # Split by [Event
    parts = re.split(r'(?=\[Event\s)', content)
    for part in parts:
        part = part.strip()
        if not part or not part.startswith('[Event'):
            continue
        try:
            game = parse_game(part)
            if game.get('White') and game.get('num_moves', 0) > 0:
                games.append(game)
        except:
            continue
    return games

def analyze_all():
    base = '/home/nurlykhan/9QumalaqV2'

    # Collect all game sources
    sources = {
        'games (mac scrape)': os.path.join(base, 'game-pars/games'),
        'linux_data': os.path.join(base, 'game-pars/linux_data/games'),
        'gameNew2': os.path.join(base, 'gameNew2'),
    }

    all_games = []
    source_counts = {}

    for src_name, src_path in sources.items():
        if not os.path.exists(src_path):
            continue
        count = 0
        files = sorted(Path(src_path).glob('*.txt'))
        total_files = len(files)
        print(f"\nParsing {src_name}: {total_files} files...", flush=True)

        for i, fp in enumerate(files):
            if (i+1) % 50000 == 0:
                print(f"  {i+1}/{total_files}...", flush=True)
            games = parse_file(fp)
            for g in games:
                g['_source'] = src_name
            all_games.extend(games)
            count += len(games)

        source_counts[src_name] = count
        print(f"  → {count} games parsed", flush=True)

    print(f"\n{'='*70}")
    print(f"TOTAL GAMES: {len(all_games):,}")
    print(f"{'='*70}")

    for src, cnt in source_counts.items():
        print(f"  {src}: {cnt:,}")

    # ====== DATE ANALYSIS ======
    print(f"\n{'='*70}")
    print("DATE ANALYSIS")
    print(f"{'='*70}")

    dates = []
    year_month = Counter()
    years = Counter()
    for g in all_games:
        d = g.get('Date', '')
        if d and d != '?':
            dates.append(d)
            parts = d.split('.')
            if len(parts) >= 2:
                year_month[f"{parts[0]}.{parts[1]}"] += 1
                years[parts[0]] += 1

    print(f"\nGames with dates: {len(dates):,}")
    print(f"\nBy year:")
    for y in sorted(years.keys()):
        print(f"  {y}: {years[y]:,} games")

    print(f"\nBy month (top 20):")
    for ym, cnt in year_month.most_common(20):
        print(f"  {ym}: {cnt:,}")

    if dates:
        dates_sorted = sorted(dates)
        print(f"\nDate range: {dates_sorted[0]} — {dates_sorted[-1]}")

    # ====== ELO ANALYSIS ======
    print(f"\n{'='*70}")
    print("ELO ANALYSIS")
    print(f"{'='*70}")

    white_elos = []
    black_elos = []
    all_elos = []
    player_elos = defaultdict(list)  # player -> list of elos

    for g in all_games:
        we = g.get('WhiteElo', '')
        be = g.get('BlackElo', '')
        w_name = g.get('White', '')
        b_name = g.get('Black', '')

        if we and we.isdigit():
            we_int = int(we)
            white_elos.append(we_int)
            all_elos.append(we_int)
            if w_name:
                player_elos[w_name].append(we_int)
        if be and be.isdigit():
            be_int = int(be)
            black_elos.append(be_int)
            all_elos.append(be_int)
            if b_name:
                player_elos[b_name].append(be_int)

    if all_elos:
        all_elos_sorted = sorted(all_elos)
        print(f"\nTotal ELO observations: {len(all_elos):,}")
        print(f"Min: {all_elos_sorted[0]}")
        print(f"Max: {all_elos_sorted[-1]}")
        print(f"Mean: {sum(all_elos)/len(all_elos):.0f}")
        print(f"Median: {all_elos_sorted[len(all_elos)//2]}")

        # ELO distribution buckets
        print(f"\nELO distribution:")
        buckets = [(0, 800), (800, 1000), (1000, 1200), (1200, 1400),
                   (1400, 1600), (1600, 1800), (1800, 2000), (2000, 2200), (2200, 2600)]
        for lo, hi in buckets:
            cnt = sum(1 for e in all_elos if lo <= e < hi)
            pct = cnt / len(all_elos) * 100
            bar = '#' * int(pct / 2)
            print(f"  {lo:4d}-{hi:4d}: {cnt:7,} ({pct:5.1f}%) {bar}")

    # ====== PLAYER ANALYSIS ======
    print(f"\n{'='*70}")
    print("PLAYER ANALYSIS")
    print(f"{'='*70}")

    player_games = Counter()
    player_wins = Counter()
    player_losses = Counter()
    player_draws = Counter()

    for g in all_games:
        w = g.get('White', '')
        b = g.get('Black', '')
        r = g.get('Result', '')

        if w:
            player_games[w] += 1
        if b:
            player_games[b] += 1

        if r == '1-0':
            player_wins[w] += 1
            player_losses[b] += 1
        elif r == '0-1':
            player_wins[b] += 1
            player_losses[w] += 1
        elif r == '1/2-1/2':
            player_draws[w] += 1
            player_draws[b] += 1

    print(f"\nUnique players: {len(player_games):,}")

    print(f"\nTop 30 most active players:")
    print(f"  {'Player':<20} {'Games':>6} {'W':>5} {'L':>5} {'D':>4} {'Win%':>6} {'Avg ELO':>8}")
    print(f"  {'-'*60}")
    for player, cnt in player_games.most_common(30):
        w = player_wins.get(player, 0)
        l = player_losses.get(player, 0)
        d = player_draws.get(player, 0)
        winpct = w / cnt * 100 if cnt > 0 else 0
        avg_elo = sum(player_elos[player]) / len(player_elos[player]) if player_elos[player] else 0
        print(f"  {player:<20} {cnt:>6} {w:>5} {l:>5} {d:>4} {winpct:>5.1f}% {avg_elo:>7.0f}")

    # Top players by ELO
    print(f"\nTop 30 players by peak ELO (min 10 games):")
    print(f"  {'Player':<20} {'Peak':>5} {'Avg':>5} {'Games':>6}")
    print(f"  {'-'*42}")
    player_peak = {}
    for p, elos in player_elos.items():
        if player_games.get(p, 0) >= 10:
            player_peak[p] = max(elos)
    for p, peak in sorted(player_peak.items(), key=lambda x: -x[1])[:30]:
        avg = sum(player_elos[p]) / len(player_elos[p])
        print(f"  {p:<20} {peak:>5} {avg:>5.0f} {player_games[p]:>6}")

    # ====== GAME LENGTH ANALYSIS ======
    print(f"\n{'='*70}")
    print("GAME LENGTH ANALYSIS")
    print(f"{'='*70}")

    lengths = [g['num_moves'] for g in all_games]
    if lengths:
        lengths_sorted = sorted(lengths)
        print(f"\nTotal moves analyzed: {sum(lengths):,}")
        print(f"Min game length: {lengths_sorted[0]} moves")
        print(f"Max game length: {lengths_sorted[-1]} moves")
        print(f"Mean: {sum(lengths)/len(lengths):.1f} moves")
        print(f"Median: {lengths_sorted[len(lengths)//2]} moves")

        print(f"\nGame length distribution:")
        len_buckets = [(1,20), (20,40), (40,60), (60,80), (80,100),
                       (100,120), (120,150), (150,200), (200,500)]
        for lo, hi in len_buckets:
            cnt = sum(1 for l in lengths if lo <= l < hi)
            pct = cnt / len(lengths) * 100
            bar = '#' * int(pct / 2)
            print(f"  {lo:3d}-{hi:3d}: {cnt:7,} ({pct:5.1f}%) {bar}")

    # ====== RESULT ANALYSIS ======
    print(f"\n{'='*70}")
    print("RESULT ANALYSIS")
    print(f"{'='*70}")

    results = Counter()
    for g in all_games:
        results[g.get('Result', '?')] += 1

    total_results = sum(results.values())
    for r, cnt in results.most_common():
        pct = cnt / total_results * 100
        print(f"  {r:<10}: {cnt:>7,} ({pct:5.1f}%)")

    # White vs Black advantage
    w_wins = results.get('1-0', 0)
    b_wins = results.get('0-1', 0)
    draws = results.get('1/2-1/2', 0)
    decided = w_wins + b_wins
    if decided > 0:
        print(f"\nFirst-mover advantage:")
        print(f"  White wins: {w_wins:,} ({w_wins/decided*100:.1f}% of decided)")
        print(f"  Black wins: {b_wins:,} ({b_wins/decided*100:.1f}% of decided)")
        print(f"  Draws: {draws:,} ({draws/total_results*100:.1f}% of all)")

    # ====== TIME CONTROL ANALYSIS ======
    print(f"\n{'='*70}")
    print("TIME CONTROL ANALYSIS")
    print(f"{'='*70}")

    tc = Counter()
    for g in all_games:
        tc[g.get('TimeControl', '?')] += 1

    print(f"\nTime controls:")
    for t, cnt in tc.most_common(15):
        pct = cnt / len(all_games) * 100
        label = f"{int(t)//60}min" if t.isdigit() else t
        print(f"  {t:>6}s ({label:>6}): {cnt:>7,} ({pct:5.1f}%)")

    # ====== OPENING ANALYSIS ======
    print(f"\n{'='*70}")
    print("OPENING ANALYSIS (First moves)")
    print(f"{'='*70}")

    # First move by White
    first_white = Counter()
    first_black = Counter()
    opening_2 = Counter()  # first 2 moves (White+Black)
    opening_4 = Counter()  # first 4 half-moves

    for g in all_games:
        moves = g['moves']
        if len(moves) >= 1:
            first_white[moves[0]['move']] += 1
        if len(moves) >= 2:
            first_black[moves[1]['move']] += 1
            opening_2[f"{moves[0]['move']} {moves[1]['move']}"] += 1
        if len(moves) >= 4:
            opening_4[f"{moves[0]['move']} {moves[1]['move']} {moves[2]['move']} {moves[3]['move']}"] += 1

    # Map pit numbers to human-readable
    pit_names = {str(i): f"pit {i}" for i in range(1, 10)}

    print(f"\nWhite's first move (pit chosen):")
    total_w = sum(first_white.values())
    for mv, cnt in first_white.most_common():
        pit = mv[0] if mv[0].isdigit() else '?'
        pct = cnt / total_w * 100
        bar = '#' * int(pct / 2)
        print(f"  Pit {pit}: {cnt:>7,} ({pct:5.1f}%) {bar}")

    print(f"\nBlack's first response:")
    total_b = sum(first_black.values())
    for mv, cnt in first_black.most_common(9):
        pit = mv[0] if mv[0].isdigit() else '?'
        pct = cnt / total_b * 100
        bar = '#' * int(pct / 2)
        print(f"  Pit {pit}: {cnt:>7,} ({pct:5.1f}%) {bar}")

    print(f"\nTop 20 opening pairs (W B):")
    for op, cnt in opening_2.most_common(20):
        pct = cnt / len(all_games) * 100
        print(f"  {op}: {cnt:>6,} ({pct:.1f}%)")

    print(f"\nTop 20 four-move openings:")
    for op, cnt in opening_4.most_common(20):
        pct = cnt / len(all_games) * 100
        print(f"  {op}: {cnt:>5,} ({pct:.1f}%)")

    # ====== TUZDYK ANALYSIS ======
    print(f"\n{'='*70}")
    print("TUZDYK ANALYSIS")
    print(f"{'='*70}")

    tuzdyk_games = sum(1 for g in all_games if g['tuzdyk_count'] > 0)
    tuzdyk_counts = Counter(g['tuzdyk_count'] for g in all_games)

    # When does tuzdyk happen (which move number)?
    tuzdyk_timing = []
    tuzdyk_pit = Counter()
    for g in all_games:
        for i, m in enumerate(g['moves']):
            if m['tuzdyk']:
                tuzdyk_timing.append(i + 1)
                tuzdyk_pit[m['move'][1]] += 1  # second digit = target pit

    print(f"\nGames with tuzdyk: {tuzdyk_games:,} ({tuzdyk_games/len(all_games)*100:.1f}%)")
    print(f"Total tuzdyks: {len(tuzdyk_timing):,}")

    print(f"\nTuzdyk count per game:")
    for tc_val in sorted(tuzdyk_counts.keys()):
        cnt = tuzdyk_counts[tc_val]
        pct = cnt / len(all_games) * 100
        print(f"  {tc_val} tuzdyks: {cnt:>7,} ({pct:5.1f}%)")

    if tuzdyk_timing:
        tt_sorted = sorted(tuzdyk_timing)
        print(f"\nTuzdyk timing:")
        print(f"  Earliest: move {tt_sorted[0]}")
        print(f"  Latest: move {tt_sorted[-1]}")
        print(f"  Mean: move {sum(tuzdyk_timing)/len(tuzdyk_timing):.1f}")
        print(f"  Median: move {tt_sorted[len(tt_sorted)//2]}")

        print(f"\nTuzdyk timing distribution:")
        timing_buckets = [(1,5), (5,10), (10,15), (15,20), (20,30), (30,50), (50,100)]
        for lo, hi in timing_buckets:
            cnt = sum(1 for t in tuzdyk_timing if lo <= t < hi)
            pct = cnt / len(tuzdyk_timing) * 100
            bar = '#' * int(pct / 2)
            print(f"  Move {lo:3d}-{hi:3d}: {cnt:>6,} ({pct:5.1f}%) {bar}")

    if tuzdyk_pit:
        print(f"\nTuzdyk target pit (opponent's pit captured):")
        for pit, cnt in tuzdyk_pit.most_common():
            pct = cnt / len(tuzdyk_timing) * 100
            bar = '#' * int(pct / 2)
            print(f"  Pit {pit}: {cnt:>6,} ({pct:5.1f}%) {bar}")

    # ====== ELO vs WIN RATE ANALYSIS ======
    print(f"\n{'='*70}")
    print("ELO DIFFERENCE vs WIN RATE")
    print(f"{'='*70}")

    elo_diff_buckets = defaultdict(lambda: {'w': 0, 'b': 0, 'd': 0})
    for g in all_games:
        we = g.get('WhiteElo', '')
        be = g.get('BlackElo', '')
        r = g.get('Result', '')
        if we.isdigit() and be.isdigit() and r in ('1-0', '0-1', '1/2-1/2'):
            diff = int(we) - int(be)  # positive = white stronger
            bucket = (diff // 50) * 50
            bucket = max(-400, min(400, bucket))
            if r == '1-0':
                elo_diff_buckets[bucket]['w'] += 1
            elif r == '0-1':
                elo_diff_buckets[bucket]['b'] += 1
            else:
                elo_diff_buckets[bucket]['d'] += 1

    print(f"\n{'Elo diff (W-B)':>15} {'Games':>7} {'W win%':>7} {'B win%':>7} {'Draw%':>6}")
    print(f"  {'-'*45}")
    for bucket in sorted(elo_diff_buckets.keys()):
        d = elo_diff_buckets[bucket]
        total = d['w'] + d['b'] + d['d']
        if total >= 50:
            wpct = d['w'] / total * 100
            bpct = d['b'] / total * 100
            dpct = d['d'] / total * 100
            print(f"  {bucket:>+4d} to {bucket+49:>+4d}: {total:>6} {wpct:>6.1f}% {bpct:>6.1f}% {dpct:>5.1f}%")

    # ====== HIGH ELO PATTERNS ======
    print(f"\n{'='*70}")
    print("HIGH ELO (1800+) vs LOW ELO (<1200) PATTERNS")
    print(f"{'='*70}")

    high_elo_games = [g for g in all_games
                      if g.get('WhiteElo','').isdigit() and g.get('BlackElo','').isdigit()
                      and int(g['WhiteElo']) >= 1800 and int(g['BlackElo']) >= 1800]
    low_elo_games = [g for g in all_games
                     if g.get('WhiteElo','').isdigit() and g.get('BlackElo','').isdigit()
                     and int(g['WhiteElo']) < 1200 and int(g['BlackElo']) < 1200]

    for label, subset in [("HIGH (1800+)", high_elo_games), ("LOW (<1200)", low_elo_games)]:
        if not subset:
            print(f"\n{label}: No games")
            continue
        print(f"\n{label}: {len(subset):,} games")

        # Average game length
        avg_len = sum(g['num_moves'] for g in subset) / len(subset)
        print(f"  Avg game length: {avg_len:.1f} moves")

        # Tuzdyk rate
        tz_rate = sum(1 for g in subset if g['tuzdyk_count'] > 0) / len(subset) * 100
        print(f"  Games with tuzdyk: {tz_rate:.1f}%")

        # Draw rate
        draws = sum(1 for g in subset if g.get('Result') == '1/2-1/2')
        print(f"  Draw rate: {draws/len(subset)*100:.1f}%")

        # First move distribution
        fm = Counter()
        for g in subset:
            if g['moves']:
                fm[g['moves'][0]['move'][0]] += 1
        total_fm = sum(fm.values())
        print(f"  First move preferences:")
        for pit, cnt in fm.most_common():
            print(f"    Pit {pit}: {cnt/total_fm*100:.1f}%")

        # White advantage
        ww = sum(1 for g in subset if g.get('Result') == '1-0')
        bw = sum(1 for g in subset if g.get('Result') == '0-1')
        decided = ww + bw
        if decided > 0:
            print(f"  White win rate (of decided): {ww/decided*100:.1f}%")

    # ====== ENDGAME PATTERNS ======
    print(f"\n{'='*70}")
    print("GAME DURATION BY ELO BUCKET")
    print(f"{'='*70}")

    elo_length = defaultdict(list)
    for g in all_games:
        we = g.get('WhiteElo', '')
        be = g.get('BlackElo', '')
        if we.isdigit() and be.isdigit():
            avg_elo = (int(we) + int(be)) // 2
            bucket = (avg_elo // 200) * 200
            elo_length[bucket].append(g['num_moves'])

    print(f"\n{'Avg ELO':>10} {'Games':>7} {'Avg Moves':>10} {'Med Moves':>10}")
    print(f"  {'-'*42}")
    for bucket in sorted(elo_length.keys()):
        lengths = elo_length[bucket]
        if len(lengths) >= 20:
            avg = sum(lengths) / len(lengths)
            med = sorted(lengths)[len(lengths)//2]
            print(f"  {bucket:>4}-{bucket+199:>4}: {len(lengths):>6} {avg:>9.1f} {med:>9}")

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")

if __name__ == '__main__':
    analyze_all()
