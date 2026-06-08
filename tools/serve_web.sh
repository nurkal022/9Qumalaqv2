#!/bin/bash
# Launch the web product locally to play the IMPROVED engine in the browser.
# Backend (FastAPI :8001) serves models/engine/baseline (improved engine + NNUE/EGTB/book).
# Frontend (Vite :5173) proxies /api and /ws to the backend.
#
# Usage:  bash tools/serve_web.sh   then open http://localhost:5173
# Stop:   bash tools/serve_web.sh stop
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BE="$ROOT/product/web/backend"
FE="$ROOT/product/web/frontend"

if [ "${1:-}" = "stop" ]; then
  pkill -f 'uvicorn app.main:app --port 8001' 2>/dev/null && echo "backend stopped"
  pkill -f 'vite --host 0.0.0.0 --port 5173' 2>/dev/null && echo "frontend stopped"
  exit 0
fi

# Backend: console scripts have a stale shebang, so call via python -m.
cd "$BE"
"$BE/.venv/bin/python" -m alembic upgrade head >/dev/null 2>&1
nohup "$BE/.venv/bin/python" -m uvicorn app.main:app --port 8001 --host 0.0.0.0 \
  > /tmp/web_backend.log 2>&1 &
echo "backend  -> http://localhost:8001  (pid $!, log /tmp/web_backend.log)"

cd "$FE"
nohup npm run dev -- --host 0.0.0.0 --port 5173 > /tmp/web_frontend.log 2>&1 &
echo "frontend -> http://localhost:5173  (pid $!, log /tmp/web_frontend.log)"

sleep 6
echo ""
echo "Open  http://localhost:5173  (or http://$(hostname -I 2>/dev/null | awk '{print $1}'):5173 from another device)"
echo "New game -> pick your side -> level 'hard' (6s/move) for strongest play. Book off."
