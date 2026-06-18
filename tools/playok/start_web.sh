#!/bin/bash
# PlayOK bot web UI launcher
# Usage: ./start_web.sh [--port 5050]
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$DIR/.venv"

if [ ! -f "$VENV/bin/python" ]; then
  echo "[start_web] creating venv..."
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install flask -q
fi

cd "$DIR"
export PLAYOK_USER="${PLAYOK_USER:-alemgamer}"
export PLAYOK_PW="${PLAYOK_PW:-REDACTED}"

echo "[start_web] http://localhost:5050"
"$VENV/bin/python" web.py "$@"
