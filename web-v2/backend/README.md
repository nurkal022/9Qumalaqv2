# Backend (web-v2)

FastAPI + SQLAlchemy async + SQLite. See `docs/superpowers/specs/2026-05-10-web-v2-foundation-design.md`.

## Dev setup
    python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"

## Run
    uvicorn app.main:app --port 8001 --reload

## Test
    pytest -q
