#!/usr/bin/env python3
"""Deep analysis: game completeness, move efficiency, expert patterns."""

import os
import re
from collections import Counter, defaultdict
from pathlib import Path
import math

def parse_game(text):
    """Parse a single game from PGN-like text."""
    game = {}
    for m in re.finditer(r'\[(\w+)\s+"([^"]*)"\]', text):
        game[m.group(1)] = m.group(2)

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
    result_match = re.search(r'\s*(1-0|0-1|1/2-1/2|\*)\s*$', move_text)
    result_in_moves = result_match.group(1) if result_match else None
    move_text = re.sub(r'\s*(1-0|0-1|1/2-1/2|\*)\s*$', '', move_text)

    moves = []
    scores = []
    tuzdyk_moves = []
    for i, m in enumerate(re.finditer(r'(\d{2})(X?)(?:\((\d+)\))?', move_text)):
        move_str = m.group(1)
        is_tuzdyk = m.group(2) == 'X'
        score = int(m.group(3)) if m.group(3) else None
        side = 'white' if i % 2 == 0 else 'black'
        moves.append({
            'move': move_str,
            'pit_from': int(move_str[0]),
            'pit_to': int(move_str[1]),
            'tuzdyk': is_tuzdyk,
            'score': score,
            'side': side,
            'half_move': i
        })
        if score is not None:
            scores.append((i, score, side))
        if is_tuzdyk:
            tuzdyk_moves.append(i)

    game['moves'] = moves
    game['scores'] = scores
    game['tuzdyk_moves'] = tuzdyk_moves
    game['num_moves'] = len(moves)
    game['result_in_moves'] = result_in_moves
    return game

def parse_all_games():
    base = '/home/nurlykhan/9QumalaqV2'
    sources = {
        'games': os.path.join(base, 'game-pars/games'),
        'linux_data': os.path.join(base, 'game-pars/linux_data/games'),
        'gameNew2': os.path.join(base, 'gameNew2'),
    }
    all_games = []
    for src_name, src_path in sources.items():
        if not os.path.exists(src_path):
            continue
        files = sorted(Path(src_path).glob('*.txt'))
        print(f"Parsing {src_name}: {len(files)} files...", flush=True)
        for i, fp in enumerate(files):
            if (i+1) % 100000 == 0:
                print(f"  {i+1}/{len(files)}...", flush=True)
            try:
                content = fp.read_text(encoding='utf-8', errors='ignore')
                parts = re.split(r'(?=\[Event\s)', content)
                for part in parts:
                    part = part.strip()
                    if not part.startswith('[Event'):
                        continue
                    g = parse_game(part)
                    if g.get('White') and g.get('num_moves', 0) > 0:
                        g['_source'] = src_name
                        all_games.append(g)
            except:
                continue
    return all_games

def analyze(all_games):
    print(f"\n{'='*80}")
    print(f"ГЛУБОКИЙ АНАЛИЗ: {len(all_games):,} ИГР")
    print(f"{'='*80}")

    # ================================================================
    # 1. GAME COMPLETION ANALYSIS
    # ================================================================
    print(f"\n{'='*80}")
    print("1. АНАЛИЗ ЗАВЕРШЁННОСТИ ИГР")
    print(f"{'='*80}")

    # Categories:
    # - Normal finish (82+ stones, result present)
    # - Very short (<10 moves) = likely abandoned/disconnected
    # - Short (10-30 moves) = early resignation
    # - Medium (30-80 moves) = mid-game resignation
    # - Normal length (80-300 moves) = full game
    # - Very long (300+) = marathon/grinding

    categories = {
        'abandoned': [],      # <10 moves
        'early_resign': [],   # 10-30 moves
        'mid_resign': [],     # 30-60 moves
        'normal_short': [],   # 60-120 moves
        'normal': [],         # 120-200 moves
        'long': [],           # 200-300 moves
        'marathon': [],       # 300+ moves
    }

    for g in all_games:
        n = g['num_moves']
        if n < 10:
            categories['abandoned'].append(g)
        elif n < 30:
            categories['early_resign'].append(g)
        elif n < 60:
            categories['mid_resign'].append(g)
        elif n < 120:
            categories['normal_short'].append(g)
        elif n < 200:
            categories['normal'].append(g)
        elif n < 300:
            categories['long'].append(g)
        else:
            categories['marathon'].append(g)

    labels = {
        'abandoned': 'Брошенные (<10 ходов)',
        'early_resign': 'Ранняя сдача (10-30)',
        'mid_resign': 'Сдача в мидгейме (30-60)',
        'normal_short': 'Короткая игра (60-120)',
        'normal': 'Нормальная (120-200)',
        'long': 'Длинная (200-300)',
        'marathon': 'Марафон (300+)',
    }

    print(f"\n{'Категория':<35} {'Игр':>8} {'Доля':>7}")
    print(f"{'-'*55}")
    for key in ['abandoned', 'early_resign', 'mid_resign', 'normal_short', 'normal', 'long', 'marathon']:
        cnt = len(categories[key])
        pct = cnt / len(all_games) * 100
        print(f"  {labels[key]:<33} {cnt:>8,} {pct:>6.1f}%")

    # Abandoned games analysis
    print(f"\n--- Брошенные игры (<10 ходов) ---")
    abandoned = categories['abandoned']
    if abandoned:
        ab_results = Counter(g.get('Result', '?') for g in abandoned)
        for r, cnt in ab_results.most_common():
            print(f"  Результат {r}: {cnt:,} ({cnt/len(abandoned)*100:.1f}%)")

        ab_lengths = Counter(g['num_moves'] for g in abandoned)
        print(f"  По длине:")
        for l in sorted(ab_lengths.keys()):
            print(f"    {l} ходов: {ab_lengths[l]:,}")

        # ELO of abandoned
        ab_elos = []
        for g in abandoned:
            for field in ['WhiteElo', 'BlackElo']:
                if g.get(field, '').isdigit():
                    ab_elos.append(int(g[field]))
        if ab_elos:
            print(f"  Средний ELO в брошенных: {sum(ab_elos)/len(ab_elos):.0f}")

    # Score progression in games (did the game end naturally at 82?)
    print(f"\n--- Анализ финальных счетов ---")
    games_with_final_score = 0
    final_scores = []
    natural_finish = 0  # score reaches 82+
    resign_likely = 0   # last score < 82 but game ended

    for g in all_games:
        scores = g['scores']
        if scores:
            last_score = scores[-1][1]
            final_scores.append(last_score)
            games_with_final_score += 1
            if last_score >= 82:
                natural_finish += 1

    if final_scores:
        print(f"  Игр с финальным счётом: {games_with_final_score:,}")
        print(f"  Естественное завершение (82+): {natural_finish:,} ({natural_finish/games_with_final_score*100:.1f}%)")
        print(f"  Сдача до 82: {games_with_final_score - natural_finish:,} ({(games_with_final_score-natural_finish)/games_with_final_score*100:.1f}%)")

        # Final score distribution
        print(f"\n  Распределение финальных счетов:")
        score_buckets = [(0,40), (40,60), (60,70), (70,75), (75,80), (80,82), (82,85), (85,90), (90,100), (100,165)]
        for lo, hi in score_buckets:
            cnt = sum(1 for s in final_scores if lo <= s < hi)
            pct = cnt / len(final_scores) * 100
            bar = '#' * int(pct)
            print(f"    {lo:>3}-{hi:>3}: {cnt:>7,} ({pct:>5.1f}%) {bar}")

    # ================================================================
    # 2. TIME-BASED COMPLETION ANALYSIS
    # ================================================================
    print(f"\n{'='*80}")
    print("2. АНАЛИЗ ТАЙМ-КОНТРОЛЯ И ЗАВЕРШЕНИЯ")
    print(f"{'='*80}")

    tc_stats = defaultdict(lambda: {'total': 0, 'abandoned': 0, 'short': 0, 'normal': 0, 'long': 0})
    for g in all_games:
        tc = g.get('TimeControl', '?')
        n = g['num_moves']
        tc_stats[tc]['total'] += 1
        if n < 10:
            tc_stats[tc]['abandoned'] += 1
        elif n < 60:
            tc_stats[tc]['short'] += 1
        elif n < 200:
            tc_stats[tc]['normal'] += 1
        else:
            tc_stats[tc]['long'] += 1

    print(f"\n{'TC (сек)':<10} {'Всего':>8} {'Брош%':>7} {'Коротк%':>8} {'Норм%':>7} {'Длин%':>7}")
    print(f"{'-'*50}")
    for tc in ['60', '120', '180', '300', '420', '600', '900', '1200']:
        s = tc_stats.get(tc, {'total':0})
        if s['total'] > 100:
            t = s['total']
            print(f"  {tc:>6}s  {t:>8,} {s['abandoned']/t*100:>6.1f}% {s['short']/t*100:>7.1f}% {s['normal']/t*100:>6.1f}% {s['long']/t*100:>6.1f}%")

    # ================================================================
    # 3. EXPERT GAME ANALYSIS (1600+)
    # ================================================================
    print(f"\n{'='*80}")
    print("3. АНАЛИЗ ЭКСПЕРТНЫХ ИГР (ELO 1600+)")
    print(f"{'='*80}")

    expert_games = [g for g in all_games
                    if g.get('WhiteElo','').isdigit() and g.get('BlackElo','').isdigit()
                    and int(g['WhiteElo']) >= 1600 and int(g['BlackElo']) >= 1600]
    all_rated = [g for g in all_games
                 if g.get('WhiteElo','').isdigit() and g.get('BlackElo','').isdigit()]

    print(f"\n  Экспертных игр (оба 1600+): {len(expert_games):,}")
    print(f"  Всего с рейтингом: {len(all_rated):,}")

    # Expert opening efficiency
    print(f"\n--- Эффективность первого хода (ELO 1600+) ---")
    first_move_stats = defaultdict(lambda: {'w': 0, 'b': 0, 'd': 0, 'total': 0})
    for g in expert_games:
        if g['moves']:
            pit = g['moves'][0]['pit_from']
            r = g.get('Result', '')
            first_move_stats[pit]['total'] += 1
            if r == '1-0':
                first_move_stats[pit]['w'] += 1
            elif r == '0-1':
                first_move_stats[pit]['b'] += 1
            elif r == '1/2-1/2':
                first_move_stats[pit]['d'] += 1

    print(f"  {'Лунка':<8} {'Игр':>6} {'White%':>8} {'Black%':>8} {'Draw%':>7} {'WinRate':>8}")
    print(f"  {'-'*50}")
    for pit in sorted(first_move_stats.keys()):
        s = first_move_stats[pit]
        t = s['total']
        if t >= 20:
            decided = s['w'] + s['b']
            wr = s['w'] / decided * 100 if decided > 0 else 0
            print(f"  Pit {pit}   {t:>6} {s['w']/t*100:>7.1f}% {s['b']/t*100:>7.1f}% {s['d']/t*100:>6.1f}% {wr:>7.1f}%")

    # Expert opening pairs efficiency
    print(f"\n--- Эффективность дебютных пар (ELO 1600+) ---")
    pair_stats = defaultdict(lambda: {'w': 0, 'b': 0, 'd': 0, 'total': 0})
    for g in expert_games:
        if len(g['moves']) >= 2:
            pair = f"{g['moves'][0]['pit_from']}-{g['moves'][1]['pit_from']}"
            r = g.get('Result', '')
            pair_stats[pair]['total'] += 1
            if r == '1-0':
                pair_stats[pair]['w'] += 1
            elif r == '0-1':
                pair_stats[pair]['b'] += 1
            elif r == '1/2-1/2':
                pair_stats[pair]['d'] += 1

    print(f"  {'Дебют (W-B)':<14} {'Игр':>6} {'W win%':>8} {'B win%':>8} {'Draw%':>7}")
    print(f"  {'-'*48}")
    sorted_pairs = sorted(pair_stats.items(), key=lambda x: -x[1]['total'])
    for pair, s in sorted_pairs[:25]:
        t = s['total']
        if t >= 30:
            print(f"  {pair:<14} {t:>6} {s['w']/t*100:>7.1f}% {s['b']/t*100:>7.1f}% {s['d']/t*100:>6.1f}%")

    # ================================================================
    # 4. MOVE EFFICIENCY BY POSITION
    # ================================================================
    print(f"\n{'='*80}")
    print("4. ЭФФЕКТИВНОСТЬ ХОДОВ ПО ПОЗИЦИЯМ")
    print(f"{'='*80}")

    # Which pit_from leads to most score gains?
    # Score changes after each move
    print(f"\n--- Анализ набора очков по лункам (все игры) ---")
    pit_score_gains = defaultdict(list)  # pit -> list of score_deltas
    for g in all_games:
        scores = g['scores']
        for i in range(1, len(scores)):
            prev_idx, prev_score, prev_side = scores[i-1]
            curr_idx, curr_score, curr_side = scores[i]
            if curr_idx < len(g['moves']):
                move = g['moves'][curr_idx]
                delta = curr_score - prev_score
                # Score is from one player's perspective, so we track the side
                pit_score_gains[move['pit_from']].append(delta)

    print(f"  {'Лунка':<8} {'Ходов':>10} {'Ср. +очков':>12} {'Медиана':>10}")
    print(f"  {'-'*44}")
    for pit in sorted(pit_score_gains.keys()):
        gains = pit_score_gains[pit]
        if len(gains) > 1000:
            avg = sum(gains) / len(gains)
            sorted_g = sorted(gains)
            med = sorted_g[len(sorted_g)//2]
            print(f"  Pit {pit}   {len(gains):>10,} {avg:>11.2f} {med:>9}")

    # ================================================================
    # 5. TUZDYK EFFECTIVENESS
    # ================================================================
    print(f"\n{'='*80}")
    print("5. ЭФФЕКТИВНОСТЬ ТҰЗДЫҚ")
    print(f"{'='*80}")

    # Win rate when you set tuzdyk vs opponent doesn't
    tz_white_only = {'w': 0, 'b': 0, 'd': 0}
    tz_black_only = {'w': 0, 'b': 0, 'd': 0}
    tz_both = {'w': 0, 'b': 0, 'd': 0}
    tz_none = {'w': 0, 'b': 0, 'd': 0}
    tz_first_wins = 0
    tz_first_total = 0

    for g in all_games:
        r = g.get('Result', '')
        if r not in ('1-0', '0-1', '1/2-1/2'):
            continue

        white_tz = False
        black_tz = False
        first_tz_side = None
        for i, mv in enumerate(g['moves']):
            if mv['tuzdyk']:
                side = 'white' if i % 2 == 0 else 'black'
                if side == 'white':
                    white_tz = True
                else:
                    black_tz = True
                if first_tz_side is None:
                    first_tz_side = side

        key = 'w' if r == '1-0' else ('b' if r == '0-1' else 'd')

        if white_tz and not black_tz:
            tz_white_only[key] += 1
        elif black_tz and not white_tz:
            tz_black_only[key] += 1
        elif white_tz and black_tz:
            tz_both[key] += 1
        else:
            tz_none[key] += 1

        if first_tz_side:
            tz_first_total += 1
            if (first_tz_side == 'white' and r == '1-0') or (first_tz_side == 'black' and r == '0-1'):
                tz_first_wins += 1

    print(f"\n--- Тұздық и результат ---")
    for label, stats in [
        ("Только White ставит", tz_white_only),
        ("Только Black ставит", tz_black_only),
        ("Оба ставят", tz_both),
        ("Никто не ставит", tz_none),
    ]:
        t = stats['w'] + stats['b'] + stats['d']
        if t > 0:
            decided = stats['w'] + stats['b']
            w_pct = stats['w'] / t * 100
            b_pct = stats['b'] / t * 100
            d_pct = stats['d'] / t * 100
            print(f"  {label:<25}: {t:>7,} игр | W {w_pct:.1f}% B {b_pct:.1f}% D {d_pct:.1f}%")

    if tz_first_total > 0:
        print(f"\n  Кто первый ставит тұздық — побеждает: {tz_first_wins:,}/{tz_first_total:,} ({tz_first_wins/tz_first_total*100:.1f}%)")

    # Tuzdyk target effectiveness
    print(f"\n--- Win rate по целевой лунке тұздық (только White) ---")
    tz_target_stats = defaultdict(lambda: {'w': 0, 'b': 0, 'd': 0})
    for g in all_games:
        r = g.get('Result', '')
        if r not in ('1-0', '0-1', '1/2-1/2'):
            continue
        for i, mv in enumerate(g['moves']):
            if mv['tuzdyk'] and i % 2 == 0:  # white's tuzdyk
                target = mv['pit_to']
                key = 'w' if r == '1-0' else ('b' if r == '0-1' else 'd')
                tz_target_stats[target][key] += 1
                break  # only first tuzdyk per game

    print(f"  {'Цель':<8} {'Игр':>7} {'W win%':>8} {'B win%':>8}")
    print(f"  {'-'*35}")
    for pit in sorted(tz_target_stats.keys()):
        s = tz_target_stats[pit]
        t = s['w'] + s['b'] + s['d']
        if t >= 100:
            print(f"  Pit {pit}   {t:>7,} {s['w']/t*100:>7.1f}% {s['b']/t*100:>7.1f}%")

    # ================================================================
    # 6. GAME PHASE ANALYSIS
    # ================================================================
    print(f"\n{'='*80}")
    print("6. АНАЛИЗ ФАЗ ИГРЫ (эксперты 1600+)")
    print(f"{'='*80}")

    # Score progression: how fast do experts build advantage?
    print(f"\n--- Средний счёт по ходам (эксперты vs новички) ---")
    expert_scores_by_move = defaultdict(list)
    novice_scores_by_move = defaultdict(list)

    novice_games = [g for g in all_games
                    if g.get('WhiteElo','').isdigit() and g.get('BlackElo','').isdigit()
                    and int(g['WhiteElo']) < 1200 and int(g['BlackElo']) < 1200]

    for subset, store in [(expert_games, expert_scores_by_move), (novice_games, novice_scores_by_move)]:
        for g in subset:
            for idx, score, side in g['scores']:
                move_num = idx // 2 + 1  # full move number
                if move_num <= 80:
                    store[move_num].append(score)

    print(f"  {'Ход':<6} {'Экспертный ср.':>15} {'Новичок ср.':>15} {'Разница':>10}")
    print(f"  {'-'*50}")
    for move_num in [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80]:
        e_scores = expert_scores_by_move.get(move_num, [])
        n_scores = novice_scores_by_move.get(move_num, [])
        e_avg = sum(e_scores) / len(e_scores) if e_scores else 0
        n_avg = sum(n_scores) / len(n_scores) if n_scores else 0
        print(f"  {move_num:>4}   {e_avg:>14.1f} {n_avg:>14.1f} {e_avg-n_avg:>+9.1f}")

    # ================================================================
    # 7. RESIGNATION ANALYSIS
    # ================================================================
    print(f"\n{'='*80}")
    print("7. АНАЛИЗ СДАЧ И НЕПОЛНЫХ ИГР")
    print(f"{'='*80}")

    # Games where the last recorded score is significantly different from 82
    resign_scores = []
    timeout_likely = 0
    clean_finish = 0
    no_score_games = 0

    for g in all_games:
        scores = g['scores']
        r = g.get('Result', '')
        if r not in ('1-0', '0-1'):
            continue
        if not scores:
            no_score_games += 1
            continue

        last_score = scores[-1][1]
        if last_score >= 82:
            clean_finish += 1
        else:
            resign_scores.append(last_score)
            # Very low scores with many moves = likely timeout
            if g['num_moves'] > 100 and last_score < 50:
                timeout_likely += 1

    decided = sum(1 for g in all_games if g.get('Result') in ('1-0', '0-1'))
    print(f"\n  Решённых игр: {decided:,}")
    print(f"  Чистое завершение (82+): {clean_finish:,} ({clean_finish/decided*100:.1f}%)")
    print(f"  Сдача/таймаут: {len(resign_scores):,} ({len(resign_scores)/decided*100:.1f}%)")
    print(f"  Без счёта: {no_score_games:,}")
    print(f"  Вероятный таймаут (100+ ходов, <50 очков): {timeout_likely:,}")

    if resign_scores:
        rs = sorted(resign_scores)
        print(f"\n  Счёт при сдаче:")
        print(f"    Минимум: {rs[0]}")
        print(f"    Максимум: {rs[-1]}")
        print(f"    Среднее: {sum(rs)/len(rs):.1f}")
        print(f"    Медиана: {rs[len(rs)//2]}")

        print(f"\n  Распределение счёта при сдаче:")
        for lo, hi in [(0,30), (30,50), (50,60), (60,70), (70,75), (75,80), (80,82)]:
            cnt = sum(1 for s in rs if lo <= s < hi)
            pct = cnt / len(rs) * 100
            bar = '#' * int(pct / 2)
            print(f"    {lo:>3}-{hi:>3}: {cnt:>7,} ({pct:>5.1f}%) {bar}")

    # ================================================================
    # 8. EXPERT SPECIFIC PATTERNS
    # ================================================================
    print(f"\n{'='*80}")
    print("8. ПАТТЕРНЫ ТОПОВЫХ ИГРОКОВ (ELO 1800+)")
    print(f"{'='*80}")

    top_players = {}  # player -> {games, wins, losses, first_moves, tz_timing}
    for g in all_games:
        for side_field, elo_field in [('White', 'WhiteElo'), ('Black', 'BlackElo')]:
            name = g.get(side_field, '')
            elo = g.get(elo_field, '')
            if not name or not elo.isdigit() or int(elo) < 1800:
                continue

            if name not in top_players:
                top_players[name] = {
                    'games': 0, 'wins': 0, 'losses': 0, 'draws': 0,
                    'first_moves_white': Counter(),
                    'tz_timings': [],
                    'game_lengths': [],
                    'elos': [],
                    'as_white': 0, 'as_black': 0,
                }

            p = top_players[name]
            p['games'] += 1
            p['elos'].append(int(elo))
            p['game_lengths'].append(g['num_moves'])

            r = g.get('Result', '')
            is_white = side_field == 'White'
            if is_white:
                p['as_white'] += 1
            else:
                p['as_black'] += 1

            if (is_white and r == '1-0') or (not is_white and r == '0-1'):
                p['wins'] += 1
            elif (is_white and r == '0-1') or (not is_white and r == '1-0'):
                p['losses'] += 1
            elif r == '1/2-1/2':
                p['draws'] += 1

            if is_white and g['moves']:
                p['first_moves_white'][g['moves'][0]['pit_from']] += 1

            for tm in g['tuzdyk_moves']:
                move_side = 'white' if tm % 2 == 0 else 'black'
                if (is_white and move_side == 'white') or (not is_white and move_side == 'black'):
                    p['tz_timings'].append(tm + 1)

    # Top 20 players with most games at 1800+
    sorted_top = sorted(top_players.items(), key=lambda x: -x[1]['games'])

    print(f"\n  Игроков на 1800+: {len(top_players):,}")
    print(f"\n  {'Игрок':<18} {'Игр':>5} {'W%':>6} {'Пик':>5} {'Ср.Длн':>7} {'Тұз.ход':>8} {'Дебют':>20}")
    print(f"  {'-'*75}")
    for name, p in sorted_top[:30]:
        if p['games'] < 20:
            continue
        winpct = p['wins'] / p['games'] * 100
        peak_elo = max(p['elos'])
        avg_len = sum(p['game_lengths']) / len(p['game_lengths'])
        avg_tz = sum(p['tz_timings']) / len(p['tz_timings']) if p['tz_timings'] else 0

        # Most common first move
        if p['first_moves_white']:
            top_move = p['first_moves_white'].most_common(1)[0]
            fm_total = sum(p['first_moves_white'].values())
            fm_str = f"Pit{top_move[0]}:{top_move[1]*100//fm_total}%"
        else:
            fm_str = "-"

        print(f"  {name:<18} {p['games']:>5} {winpct:>5.1f}% {peak_elo:>5} {avg_len:>6.0f} {avg_tz:>7.1f} {fm_str:>20}")

    # ================================================================
    # 9. OPENING BOOK QUALITY ANALYSIS
    # ================================================================
    print(f"\n{'='*80}")
    print("9. АНАЛИЗ КАЧЕСТВА ДЕБЮТОВ (первые 6 полуходов)")
    print(f"{'='*80}")

    # Analyze 3-move deep openings with win rates by ELO
    opening_3_stats = defaultdict(lambda: {
        'total': 0, 'w': 0, 'b': 0, 'd': 0,
        'expert_total': 0, 'expert_w': 0, 'expert_b': 0,
    })

    for g in all_games:
        if len(g['moves']) < 6:
            continue
        opening = '-'.join(str(g['moves'][i]['pit_from']) for i in range(6))
        r = g.get('Result', '')
        s = opening_3_stats[opening]
        s['total'] += 1
        if r == '1-0':
            s['w'] += 1
        elif r == '0-1':
            s['b'] += 1
        elif r == '1/2-1/2':
            s['d'] += 1

        we = g.get('WhiteElo', '')
        be = g.get('BlackElo', '')
        if we.isdigit() and be.isdigit() and int(we) >= 1600 and int(be) >= 1600:
            s['expert_total'] += 1
            if r == '1-0':
                s['expert_w'] += 1
            elif r == '0-1':
                s['expert_b'] += 1

    # Top openings by frequency
    sorted_openings = sorted(opening_3_stats.items(), key=lambda x: -x[1]['total'])

    print(f"\n  Топ-30 дебютов (6 полуходов, мин. 500 игр):")
    print(f"  {'Дебют':<22} {'Игр':>7} {'W%':>6} {'B%':>6} {'D%':>5} {'Эксп.':>6} {'ExpW%':>6}")
    print(f"  {'-'*65}")
    shown = 0
    for opening, s in sorted_openings:
        if s['total'] < 500:
            continue
        t = s['total']
        wpct = s['w'] / t * 100
        bpct = s['b'] / t * 100
        dpct = s['d'] / t * 100
        et = s['expert_total']
        ewpct = s['expert_w'] / et * 100 if et > 0 else 0
        print(f"  {opening:<22} {t:>7,} {wpct:>5.1f}% {bpct:>5.1f}% {dpct:>4.1f}% {et:>5} {ewpct:>5.1f}%")
        shown += 1
        if shown >= 30:
            break

    # ================================================================
    # 10. ENDGAME PATTERNS
    # ================================================================
    print(f"\n{'='*80}")
    print("10. АНАЛИЗ ЭНДШПИЛЬНЫХ ПАТТЕРНОВ")
    print(f"{'='*80}")

    # Moves per game in games won by White vs Black
    w_win_lengths = [g['num_moves'] for g in all_games if g.get('Result') == '1-0']
    b_win_lengths = [g['num_moves'] for g in all_games if g.get('Result') == '0-1']
    draw_lengths = [g['num_moves'] for g in all_games if g.get('Result') == '1/2-1/2']

    print(f"\n  Средняя длина по результату:")
    if w_win_lengths:
        print(f"    White wins: {sum(w_win_lengths)/len(w_win_lengths):.1f} ходов (медиана: {sorted(w_win_lengths)[len(w_win_lengths)//2]})")
    if b_win_lengths:
        print(f"    Black wins: {sum(b_win_lengths)/len(b_win_lengths):.1f} ходов (медиана: {sorted(b_win_lengths)[len(b_win_lengths)//2]})")
    if draw_lengths:
        print(f"    Draws:      {sum(draw_lengths)/len(draw_lengths):.1f} ходов (медиана: {sorted(draw_lengths)[len(draw_lengths)//2]})")

    # Late game moves (move 60+): which pits used most?
    print(f"\n  Использование лунок в эндшпиле (ход 100+, эксперты 1600+):")
    endgame_pits = Counter()
    midgame_pits = Counter()
    for g in expert_games:
        for i, mv in enumerate(g['moves']):
            if i >= 200:  # move 100+ (half-moves 200+)
                endgame_pits[mv['pit_from']] += 1
            elif 40 <= i < 120:  # moves 20-60
                midgame_pits[mv['pit_from']] += 1

    if endgame_pits:
        total_eg = sum(endgame_pits.values())
        total_mg = sum(midgame_pits.values())
        print(f"\n  {'Лунка':<8} {'Миттельшпиль':>15} {'Эндшпиль':>15}")
        print(f"  {'-'*42}")
        for pit in sorted(set(list(endgame_pits.keys()) + list(midgame_pits.keys()))):
            mg_pct = midgame_pits[pit] / total_mg * 100 if total_mg > 0 else 0
            eg_pct = endgame_pits[pit] / total_eg * 100 if total_eg > 0 else 0
            print(f"  Pit {pit}   {mg_pct:>14.1f}% {eg_pct:>14.1f}%")

    print(f"\n{'='*80}")
    print("АНАЛИЗ ЗАВЕРШЁН")
    print(f"{'='*80}")

if __name__ == '__main__':
    games = parse_all_games()
    analyze(games)
