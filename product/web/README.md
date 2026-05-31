# web-v2

Sub-project A of the new Togyzkumalak platform. See:
- Spec: [`docs/superpowers/specs/2026-05-10-web-v2-foundation-design.md`](../docs/superpowers/specs/2026-05-10-web-v2-foundation-design.md)
- Plan: [`docs/superpowers/plans/2026-05-10-web-v2-foundation.md`](../docs/superpowers/plans/2026-05-10-web-v2-foundation.md)

## Architecture

- **Backend** (`backend/`): FastAPI + SQLAlchemy 2 async + SQLite (WAL) + Alembic. JWT cookie + anonymous session middleware. Engine pool wraps the Rust NNUE binary in `serve` mode.
- **Frontend** (`frontend/`): React 19 + Vite + TypeScript + TailwindCSS 4 + shadcn-style primitives, TanStack Query for server state, Zustand for UI state, hash routing, KK + RU localization.
- **Engine integration**: Togyzkumalak rules are ported into Python (`engine/process.py:_apply_move_to_pos`) because the Rust engine has no apply-move-and-return-position command. The play move endpoint round-trip-validates the resulting position with the engine's `position` command to detect divergence.

## Run dev

```bash
# 1. Build engine once
cd ../engine && cargo build --release && cd ..

# 2. Backend (port 8001)
cd web-v2/backend
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
alembic upgrade head
uvicorn app.main:app --port 8001 --reload

# 3. Frontend (port 5173, separate shell)
cd web-v2/frontend
npm install
npm run dev
```

Open <http://localhost:5173>.

## Tests

```bash
# Backend
cd backend && . .venv/bin/activate && pytest -q

# Frontend
cd frontend && npm run typecheck && npm run test
```

## Notable design decisions

- Single source of truth for game state is the server. Client (TanStack Query cache) is repopulated on tab focus and WS reconnect.
- WebSocket protocol uses a per-game seq ring buffer (200 msgs) for reconnect replay; client sends `hello{lastSeenSeq}` on connect.
- Engine subprocess is single-instance with `asyncio.Lock` serializing `think()`. Auto-restart once on death.
- Hint limited to 3/game (in-memory counter, resets on server restart). Resign and draw_offer immediately finalize the game.
- Locale is KK by default; toggleable to RU; persisted to localStorage.
