# 9qum.com harvester

Pulls everything 9qum.com (friendly competitor platform) exposes publicly: full game
records, the player graph, their opening statistics and their AlphaZero training
telemetry. Output lands in `data/9qum/` (gitignored).

```bash
node   tools/9qum/ws_tournaments.js data/9qum/tournaments.json   # tournaments (websocket only)
python3 tools/9qum/harvest.py                                    # meta + train + openings + players + games
python3 tools/9qum/report.py                                     # inventory / quality report
```

Everything is resumable: each phase keeps a `*.done` list and skips what it already has,
so re-running is how you pick up newly played games. `--rps` caps the request rate
(default 5/s) and the `games` phase takes `--workers` (default 4) because it is
latency-bound. Guest tokens are rate-limited per IP, so the websocket session is cached
in `data/9qum/session.json` and reused.

## What each endpoint gives us

| endpoint | contents |
|---|---|
| `GET /api/train/status` | their whole training loop: iteration, sims, replay size, per-iteration loss (policy / value / **score**), gate winrate + accept flag, `match_rate`/`match_top3`, self-play and human corpus totals, hardware |
| `GET /api/openings?line=7,4,…` | opening tree over their game corpus: per-pit `count` / `share` / `winrate`, plus the resulting board + TFEN. `line` is comma-separated pit labels (1-9) |
| `GET /api/games/recent` | last 30 finished games (id, seats, winner, reason, ply count) |
| `GET /api/games/{id}/replay` | **full record**: one state per ply (pits, kazan, tuzdyk, legal moves, TFEN, clock) plus the move list with notation and captures. No auth. Closed-tournament games return 403 |
| `GET /api/player/{name}` `/games` `/opponents` `/openings` | profile, **last 100 games with rating before/after**, the full opponent graph (crawlable), and the player's first-move repertoire |
| `GET /api/leaderboard`, `/leaderboard/activity` | top 100 by rating; top 10 by games/wins/winrate/hours/week/climb |
| `GET /api/ai/levels` | their AI ladder — pure sims counts (III 40 … ЗМС 1280) mapped to claimed 2000-3000 ratings |
| `GET /api/altynqor[/{id}]` | archive of official KZ tournaments (from t.me/altynqortogyz): rounds, boards, standings — results only, **no moves** |
| WS `{type:"tournament.get",tid}` | tournament detail incl. every `game_id` (public tournaments only) |
| `GET /api/analysis/curve/{id}` | **their net's win% for every ply of any game** — instant, free, needs only a guest Bearer token. Value labels from an evaluator outside our lineage |
| `POST /api/analysis/review` `{game_id, ai_level}` | queued review (poll until `state == "готово"`): the 8 worst moments with `played` / `best` / `q_played` / `q_best` / `cost` / `win_before` / `win_after` and a 6-move `pv`. Level `i` (90 sims) is free; МС/МСМК/ЗМС cost coins |
| `GET /api/lobby`, `/battle/live`, `/skins`, `/shop/items`, `/fed/*` | live tables, battle history, product/economy metadata |

## Data layout

```
data/9qum/
  meta/*.json            one file per small endpoint (+ altynqor_<id>.json)
  train_status.jsonl     one snapshot per run — re-run to build their training history
  openings.jsonl         one line per opening node {line, depth, data}
  players.jsonl          profile + repertoire + opponent list per player
  player_games.jsonl     per player: last 100 games with r0/r1 before+after
  tournaments.json       list + full detail (pairings, standings) for every tournament
  games/replays.jsonl.gz full move-by-move replays (one JSON per line, `_src`/`_meta` added)
  games/skipped.jsonl    ids with nothing to fetch (no-shows, 403 closed tournaments)
  analysis/curves.jsonl.gz   their net's win% per ply, one line per game
  analysis/reviews.jsonl.gz  their engine's worst-moment reviews (only with --reviews)
```

`--reviews` makes their server think for ~5s per game, so it stays opt-in; curves are
free and instant, so the default `analysis` phase takes those for every game we hold.

## Notation

TFEN (their FEN analogue, worth supporting for interop):

```
1,12,12,12,12,3,1,13,12/12,0,11,11,11,1,9,1,2 22 4 - - 1
side0 pits / side1 pits  kazan0 kazan1  tuzdyk0 tuzdyk1  side_to_move
```

`record` is the space-separated list of played pit labels (`"7 9 6 8 1"`); per-move
`notation` is `<from><to>` plus captures in parentheses, e.g. `76(10)`.

In replay states `tuzdyk[p]` is an **absolute** pit index 0-17 (the pit player `p` owns on
the opponent's side); its label is `idx % 9 + 1`. Their referee applies the endgame sweep
and stores the *post-sweep* final position (kazan sums to 162), which matches our rule in
`core/src/board.rs`.
