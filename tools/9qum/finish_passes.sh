#!/usr/bin/env bash
# Second sweep: once the player crawl stops discovering, re-run games (to pick up ids
# found late in the crawl) and analysis (curves for the newly downloaded replays).
# Repeats until a pass adds nothing new.
set -u
cd "$(dirname "$0")/../.."

wait_for() { while pgrep -f "^python3 .*harvest.py --phase $1" >/dev/null; do sleep 20; done; }

wait_for players
wait_for games
wait_for analysis

for pass in 2 3 4; do
  before=$(wc -l < data/9qum/games/replays.done)
  echo "[pass $pass] starting; replays.done=$before"
  python3 -u tools/9qum/harvest.py --phase games    --rps 8 --workers 6 >> data/9qum/log_games.txt 2>&1
  python3 -u tools/9qum/harvest.py --phase analysis --rps 6 --workers 4 >> data/9qum/log_analysis.txt 2>&1
  after=$(wc -l < data/9qum/games/replays.done)
  echo "[pass $pass] done; replays.done=$after (+$((after - before)))"
  [ "$after" -eq "$before" ] && break
done
echo "ALL PASSES COMPLETE"
