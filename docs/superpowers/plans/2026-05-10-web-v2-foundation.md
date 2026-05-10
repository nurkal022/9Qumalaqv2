# web-v2 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build sub-project A of the new Togyzkumalak platform per [the design](../specs/2026-05-10-web-v2-foundation-design.md): a polished mobile-first React+FastAPI app at `web-v2/` for solo play vs the NNUE engine, with anonymous sessions + optional accounts, persistent state across reloads, live engine eval over WebSocket, and KK+RU localization. Replaces the fragile current `web/`.

**Architecture:** Two parallel apps under `web-v2/{backend,frontend}`. Backend is FastAPI + SQLAlchemy 2 async + SQLite (WAL); single subprocess engine pool with `asyncio.Lock` and per-game subscribers; WebSocket endpoint streams eval and supports resume via per-game seq ring buffer. Frontend is React 19 + Vite + TypeScript + Tailwind 4 + shadcn/ui, TanStack Query for server state, Zustand for UI state, custom WebSocket hook with reconnect.

**Tech Stack:** FastAPI 0.115+, SQLAlchemy 2.x async, aiosqlite, Alembic, bcrypt, PyJWT, slowapi, pytest+httpx; React 19, Vite 5, TypeScript 5, TailwindCSS 4, shadcn/ui (Radix), TanStack Query 5, Zustand, React Router 7, react-hook-form + zod, react-i18next, vitest + @testing-library/react.

---

## Milestones

1. **Backend skeleton + DB** (Tasks 1-3): FastAPI boots, `/health` reachable, SQLite + Alembic baseline, models migrated.
2. **Auth foundation** (Tasks 4-6): anon session middleware, register/login/logout/me, anon→user migration.
3. **Engine pool** (Tasks 7-8): subprocess wrapper + pool with lock, subscribers, auto-restart.
4. **Play service** (Tasks 9-11): snapshot builder + clock, REST play endpoints, hint rate limit + draw offer.
5. **WebSocket** (Task 12): snapshot, eval stream, resume, backpressure, heartbeat.
6. **Games history endpoints** (Task 13): list / get / delete.
7. **Frontend skeleton** (Tasks 14-16): Vite + React + Tailwind + shadcn + i18n + router + TanStack Query + Zustand.
8. **Auth UI** (Task 17): Login / Register / Profile + useAuth hook.
9. **Lobby** (Task 18): NewGameForm with validation.
10. **Board + Game** (Tasks 19-22): Board / Hole / Pebbles; useGameSocket; Game route assembled; end-to-end smoke.
11. **History + Replay** (Tasks 23-24).
12. **Polish** (Tasks 25-27): error envelope wiring, observability, ornaments + README.

After every task: `pytest -q` (BE) + `npm run typecheck && npm run test` (FE) all green.

---

## Conventions used throughout this plan

- **All paths relative to repo root** `/home/nurlykhan/9QumalaqV2/`.
- **Backend cwd:** `web-v2/backend/`. **Frontend cwd:** `web-v2/frontend/`.
- **Python tooling:** `python -m venv .venv` + `pip install -e ".[dev]"`. (`uv` is fine if available — substitute commands at install time.)
- **Node tooling:** `npm`. If `pnpm` is preferred, substitute commands at install time.
- **TDD:** for every Python module with logic and every TypeScript hook/util, write the failing test first, watch it fail, write code, watch it pass. Pure presentational components (no behavior beyond rendering tokens) get a smoke render test.
- **Commits:** every task ends with one commit. Branch is whatever the user is on; do not switch branches.
- **Engine binary:** assumed built at `engine/target/release/togyzkumalaq-engine`. If missing, `cd engine && cargo build --release` once.
- **No emojis in code or commits.**
- **Server URLs in dev:** backend `http://localhost:8001`, frontend `http://localhost:5173`. Vite proxies `/api` and `/ws` to backend.

---

## File Structure (locked from spec)

Backend `web-v2/backend/app/`: `main.py`, `config.py`, `deps.py`, `errors.py`, `db/{base,models}.py`, `auth/{routes,schemas,service,jwt,anonymous}.py`, `play/{routes,schemas,service,clock,snapshot}.py`, `games/{routes,schemas,service}.py`, `engine/{pool,stream}.py`, `ws/games_ws.py`. Tests mirror under `tests/`.

Frontend `web-v2/frontend/src/`: `main.tsx`, `App.tsx`, `api/{client,auth,play,games,ws}.ts`, `stores/{ui,auth}.ts`, `hooks/{useGameSocket,useGameQuery,useAuth}.ts`, `domain/{board,notation,ornaments}.ts`, `i18n/{index,kk,ru}/`, `design/{tokens,globals.css}`, `routes/{Lobby,Game,History,Replay,Login,Register,Profile}.tsx`, `components/{board,play,lobby,history,layout,ui}/`.

---

## Task 1: Backend skeleton — FastAPI app, pytest, /health

**Files:**
- Create: `web-v2/backend/pyproject.toml`
- Create: `web-v2/backend/app/__init__.py` (empty)
- Create: `web-v2/backend/app/main.py`
- Create: `web-v2/backend/app/config.py`
- Create: `web-v2/backend/tests/__init__.py` (empty)
- Create: `web-v2/backend/tests/conftest.py`
- Create: `web-v2/backend/tests/test_health.py`
- Create: `web-v2/backend/.gitignore`
- Create: `web-v2/backend/README.md`
- Modify: `.gitignore` (root) — add web-v2 dev artifacts

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "togyzkumalaq-backend"
version = "0.1.0"
description = "FastAPI backend for web-v2"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "sqlalchemy[asyncio]>=2.0",
    "aiosqlite>=0.20",
    "alembic>=1.13",
    "pydantic>=2.9",
    "pydantic-settings>=2.6",
    "bcrypt>=4.2",
    "pyjwt>=2.10",
    "slowapi>=0.1.9",
    "python-multipart>=0.0.20",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "httpx>=0.28",
    "anyio>=4.0",
    "ruff>=0.7",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["app"]
```

- [ ] **Step 2: Create `app/config.py`**

```python
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    env: str = "dev"
    database_url: str = "sqlite+aiosqlite:///./web_v2.db"
    jwt_secret: str = "dev-only-replace-me"
    jwt_algorithm: str = "HS256"
    jwt_ttl_days: int = 7
    anon_cookie_ttl_days: int = 365
    engine_path: Path = REPO_ROOT / "engine" / "target" / "release" / "togyzkumalaq-engine"
    cors_origins: list[str] = ["http://localhost:5173"]


settings = Settings()
```

- [ ] **Step 3: Failing health test**

`tests/conftest.py`:
```python
import pytest
from httpx import ASGITransport, AsyncClient
from app.main import create_app

@pytest.fixture
async def client():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
```

`tests/test_health.py`:
```python
async def test_health_returns_ok(client):
    r = await client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
```

- [ ] **Step 4: Run, expect ImportError**

```bash
cd web-v2/backend && python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]" && python -m pytest -q
```
Expected: `ImportError: cannot import name 'create_app'`.

- [ ] **Step 5: Implement `app/main.py`**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings


def create_app() -> FastAPI:
    app = FastAPI(title="Togyzkumalak web-v2 API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
```

- [ ] **Step 6: Run, expect pass**

`python -m pytest -q` → `1 passed`.

- [ ] **Step 7: Smoke run**

```bash
uvicorn app.main:app --port 8001 --reload &
sleep 1 && curl -s http://localhost:8001/api/health && kill %1
```
Expected: `{"status":"ok"}`.

- [ ] **Step 8: `.gitignore` files**

`web-v2/backend/.gitignore`:
```
.venv/
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
.env
*.db
*.db-journal
*.db-wal
*.db-shm
```

Append to root `.gitignore` (only if not already there):
```
# web-v2 dev artifacts
web-v2/backend/.venv/
web-v2/backend/*.db*
web-v2/frontend/node_modules/
web-v2/frontend/dist/
```

- [ ] **Step 9: README + commit**

`web-v2/backend/README.md`:
```markdown
# Backend (web-v2)

FastAPI + SQLAlchemy async + SQLite. See `docs/superpowers/specs/2026-05-10-web-v2-foundation-design.md`.

## Dev setup
    python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"

## Run
    uvicorn app.main:app --port 8001 --reload

## Test
    pytest -q
```

```bash
git add web-v2/backend pyproject.toml .gitignore
git commit -m "$(cat <<'EOF'
feat(web-v2): backend skeleton with FastAPI + pytest

Bootstraps web-v2/backend with FastAPI app factory, pydantic-settings
config, /api/health endpoint, and pytest+httpx test harness. CORS
configured for the Vite dev server origin.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: DB base + Alembic + initial migration

**Files:**
- Create: `web-v2/backend/app/db/__init__.py` (empty)
- Create: `web-v2/backend/app/db/base.py`
- Create: `web-v2/backend/app/deps.py`
- Create: `web-v2/backend/alembic.ini`
- Create: `web-v2/backend/alembic/env.py`
- Create: `web-v2/backend/alembic/script.py.mako`
- Create: `web-v2/backend/alembic/versions/.gitkeep`

- [ ] **Step 1: `app/db/base.py`**

```python
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from app.config import settings


class Base(DeclarativeBase):
    pass


engine = create_async_engine(settings.database_url, echo=False, future=True)
SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
```

- [ ] **Step 2: `app/deps.py`**

```python
from typing import AsyncIterator
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.base import SessionLocal


async def get_db() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session
```

- [ ] **Step 3: Set up Alembic**

```bash
cd web-v2/backend && . .venv/bin/activate && alembic init alembic
```

Edit `alembic.ini`: change `sqlalchemy.url = ` to empty (we set in env.py).

Edit `alembic/env.py` — replace body with:
```python
import asyncio
from logging.config import fileConfig
from alembic import context
from sqlalchemy.ext.asyncio import async_engine_from_config
from sqlalchemy import pool
from app.config import settings
from app.db.base import Base
import app.db.models  # noqa: F401  (registers models — created next task)

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", settings.database_url.replace("+aiosqlite", ""))
target_metadata = Base.metadata


def run_migrations_offline():
    context.configure(url=config.get_main_option("sqlalchemy.url"), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online():
    cfg = config.get_section(config.config_ini_section) or {}
    cfg["sqlalchemy.url"] = settings.database_url
    connectable = async_engine_from_config(cfg, prefix="sqlalchemy.", poolclass=pool.NullPool)
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
```

- [ ] **Step 4: Test DB connectivity**

`tests/test_db.py`:
```python
from sqlalchemy import text
from app.db.base import SessionLocal


async def test_db_connection_works():
    async with SessionLocal() as s:
        result = await s.execute(text("SELECT 1"))
        assert result.scalar() == 1
```

Run `pytest tests/test_db.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add web-v2/backend/app/db web-v2/backend/app/deps.py web-v2/backend/alembic* web-v2/backend/tests/test_db.py
git commit -m "feat(web-v2): SQLAlchemy async base + Alembic scaffolding"
```

---

## Task 3: SQLAlchemy models + initial migration

**Files:**
- Create: `web-v2/backend/app/db/models.py`
- Create: `web-v2/backend/alembic/versions/0001_initial.py` (autogenerated)
- Create: `web-v2/backend/tests/test_models.py`

- [ ] **Step 1: Define models**

`app/db/models.py` — model exactly the schema from the spec:
```python
from datetime import datetime
from sqlalchemy import (
    Boolean, CheckConstraint, DateTime, ForeignKey, Index, Integer,
    String, Text, UniqueConstraint, func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(128))
    password_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    locale: Mapped[str] = mapped_column(String(8), nullable=False, default="kk")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp(), nullable=False)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime)


class AnonSession(Base):
    __tablename__ = "anon_sessions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp(), nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp(), nullable=False)
    locale: Mapped[str] = mapped_column(String(8), nullable=False, default="kk")


class Game(Base):
    __tablename__ = "games"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    anon_session_id: Mapped[str | None] = mapped_column(ForeignKey("anon_sessions.id", ondelete="SET NULL"))
    mode: Mapped[str] = mapped_column(String(16), nullable=False)
    side: Mapped[int] = mapped_column(Integer, nullable=False)
    opponent_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    opponent_ref: Mapped[str | None] = mapped_column(String(128))
    engine_level: Mapped[str] = mapped_column(String(16), nullable=False, default="normal")
    clock_initial_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    clock_increment_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    clock_white_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    clock_black_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    last_clock_at: Mapped[datetime | None] = mapped_column(DateTime)
    start_fen: Mapped[str] = mapped_column(Text, nullable=False)
    current_fen: Mapped[str] = mapped_column(Text, nullable=False)
    current_ply: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    side_to_move: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    result: Mapped[str | None] = mapped_column(String(16))
    result_reason: Mapped[str | None] = mapped_column(String(32))
    final_score: Mapped[str | None] = mapped_column(String(16))
    started_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp(), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)

    moves: Mapped[list["Move"]] = relationship(back_populates="game", cascade="all, delete-orphan", order_by="Move.ply")
    events: Mapped[list["GameEvent"]] = relationship(back_populates="game", cascade="all, delete-orphan", order_by="GameEvent.created_at")

    __table_args__ = (
        CheckConstraint(
            "(user_id IS NOT NULL AND anon_session_id IS NULL) OR "
            "(user_id IS NULL AND anon_session_id IS NOT NULL)",
            name="games_owner_xor",
        ),
        Index("idx_games_user", "user_id", "started_at"),
        Index("idx_games_anon", "anon_session_id", "started_at"),
        Index("idx_games_active", "status"),
    )


class Move(Base):
    __tablename__ = "moves"
    id: Mapped[int] = mapped_column(primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id", ondelete="CASCADE"), nullable=False)
    ply: Mapped[int] = mapped_column(Integer, nullable=False)
    side: Mapped[int] = mapped_column(Integer, nullable=False)
    actor: Mapped[str] = mapped_column(String(16), nullable=False)
    move_uci: Mapped[str] = mapped_column(String(8), nullable=False)
    fen_after: Mapped[str] = mapped_column(Text, nullable=False)
    eval_cp: Mapped[int | None] = mapped_column(Integer)
    eval_depth: Mapped[int | None] = mapped_column(Integer)
    pv: Mapped[str | None] = mapped_column(Text)
    think_time_ms: Mapped[int | None] = mapped_column(Integer)
    clock_after_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp(), nullable=False)

    game: Mapped[Game] = relationship(back_populates="moves")
    __table_args__ = (UniqueConstraint("game_id", "ply"), Index("idx_moves_game", "game_id", "ply"))


class GameEvent(Base):
    __tablename__ = "game_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id", ondelete="CASCADE"), nullable=False)
    ply_at: Mapped[int] = mapped_column(Integer, nullable=False)
    actor: Mapped[str] = mapped_column(String(16), nullable=False)
    type: Mapped[str] = mapped_column(String(32), nullable=False)
    payload_json: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp(), nullable=False)

    game: Mapped[Game] = relationship(back_populates="events")
    __table_args__ = (Index("idx_events_game", "game_id", "created_at"),)


class Engine(Base):
    __tablename__ = "engines"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    binary_path: Mapped[str] = mapped_column(Text, nullable=False)
    weights_path: Mapped[str | None] = mapped_column(Text)
    build_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp(), nullable=False)
```

- [ ] **Step 2: Generate migration**

```bash
cd web-v2/backend && . .venv/bin/activate
alembic revision --autogenerate -m "initial schema"
alembic upgrade head
```

Inspect generated `alembic/versions/0001_*_initial_schema.py` — confirm tables match. Hand-edit if Alembic missed the username `LOWER` index; add `op.create_index("idx_users_username_lower", "users", [sa.text("LOWER(username)")], unique=True)`.

- [ ] **Step 3: Test models persist**

`tests/test_models.py`:
```python
from app.db.base import SessionLocal
from app.db.models import User, AnonSession


async def test_user_can_persist():
    async with SessionLocal() as s:
        s.add(User(username="alice", password_hash="x", locale="kk"))
        await s.commit()
        users = (await s.execute(__import__("sqlalchemy").select(User).where(User.username == "alice"))).scalars().all()
        assert len(users) == 1


async def test_anon_session_can_persist():
    async with SessionLocal() as s:
        import uuid
        sid = str(uuid.uuid4())
        s.add(AnonSession(id=sid))
        await s.commit()
```

Run `pytest -q` → all green.

- [ ] **Step 4: Commit**

```bash
git add web-v2/backend/app/db/models.py web-v2/backend/alembic/versions/ web-v2/backend/tests/test_models.py
git commit -m "feat(web-v2): SQLAlchemy models + initial Alembic migration"
```

---

## Task 4: Anon session middleware + cookie management

**Files:**
- Create: `web-v2/backend/app/auth/__init__.py` (empty)
- Create: `web-v2/backend/app/auth/anonymous.py`
- Create: `web-v2/backend/app/auth/jwt.py`
- Create: `web-v2/backend/app/auth/session.py`
- Modify: `web-v2/backend/app/main.py` (register middleware)
- Create: `web-v2/backend/tests/test_anon_session.py`

- [ ] **Step 1: `app/auth/jwt.py`**

```python
from datetime import datetime, timedelta, timezone
import jwt
from app.config import settings


def encode_user(user_id: int) -> str:
    now = datetime.now(timezone.utc)
    payload = {"sub": str(user_id), "iat": now, "exp": now + timedelta(days=settings.jwt_ttl_days)}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_user(token: str) -> int | None:
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        return int(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        return None
```

- [ ] **Step 2: `app/auth/session.py` — current-session resolver**

```python
from dataclasses import dataclass
from app.db.models import User, AnonSession


@dataclass
class CurrentSession:
    user: User | None
    anon: AnonSession | None

    @property
    def kind(self) -> str:
        return "user" if self.user else "anon"

    @property
    def owner_filter(self) -> dict:
        return {"user_id": self.user.id} if self.user else {"anon_session_id": self.anon.id}
```

- [ ] **Step 3: `app/auth/anonymous.py` — middleware**

```python
import uuid
from datetime import datetime, timezone
from fastapi import Request
from sqlalchemy import select, update
from app.config import settings
from app.db.base import SessionLocal
from app.db.models import AnonSession, User
from app.auth.jwt import decode_user
from app.auth.session import CurrentSession


COOKIE_ANON = "anon_session"
COOKIE_JWT = "auth_token"


async def resolve_session(request: Request) -> CurrentSession:
    """Read cookies. Return (user, anon) — exactly one is non-None.
    If neither present, create new anon session and stash it on request.state for the
    response middleware to set the cookie.
    """
    token = request.cookies.get(COOKIE_JWT)
    if token:
        uid = decode_user(token)
        if uid is not None:
            async with SessionLocal() as s:
                u = await s.get(User, uid)
                if u is not None:
                    return CurrentSession(user=u, anon=None)

    anon_id = request.cookies.get(COOKIE_ANON)
    async with SessionLocal() as s:
        anon: AnonSession | None = None
        if anon_id:
            anon = await s.get(AnonSession, anon_id)
        if anon is None:
            anon = AnonSession(id=str(uuid.uuid4()))
            s.add(anon)
            await s.commit()
            await s.refresh(anon)
            request.state.new_anon_id = anon.id
        else:
            await s.execute(
                update(AnonSession).where(AnonSession.id == anon.id).values(last_seen_at=datetime.now(timezone.utc))
            )
            await s.commit()
        return CurrentSession(user=None, anon=anon)


def attach_anon_cookie_if_new(request: Request, response) -> None:
    new_id = getattr(request.state, "new_anon_id", None)
    if new_id:
        response.set_cookie(
            COOKIE_ANON, new_id,
            max_age=settings.anon_cookie_ttl_days * 24 * 3600,
            httponly=True, samesite="lax",
            secure=settings.env == "prod",
        )
```

- [ ] **Step 4: Wire middleware in `app/main.py`**

Add to `create_app()`:
```python
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from app.auth.anonymous import resolve_session, attach_anon_cookie_if_new

class SessionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request.state.session = await resolve_session(request)
        response = await call_next(request)
        attach_anon_cookie_if_new(request, response)
        return response

app.add_middleware(SessionMiddleware)
```

Also export `get_session` dep in `app/deps.py`:
```python
from fastapi import Request
from app.auth.session import CurrentSession

def get_session(request: Request) -> CurrentSession:
    return request.state.session
```

- [ ] **Step 5: Tests**

`tests/test_anon_session.py`:
```python
async def test_first_request_sets_anon_cookie(client):
    r = await client.get("/api/health")
    assert r.status_code == 200
    set_cookie = r.headers.get("set-cookie", "")
    assert "anon_session=" in set_cookie

async def test_second_request_reuses_anon_cookie(client):
    r1 = await client.get("/api/health")
    cookies = r1.cookies
    r2 = await client.get("/api/health", cookies=cookies)
    assert "anon_session=" not in r2.headers.get("set-cookie", "")
```

Run `pytest tests/test_anon_session.py -v` → PASS.

- [ ] **Step 6: Commit**

```bash
git add web-v2/backend/app/auth web-v2/backend/app/main.py web-v2/backend/app/deps.py web-v2/backend/tests/test_anon_session.py
git commit -m "feat(web-v2): anonymous session middleware + JWT helpers"
```

---

## Task 5: Auth endpoints — register / login / logout / me

**Files:**
- Create: `web-v2/backend/app/auth/schemas.py`
- Create: `web-v2/backend/app/auth/service.py`
- Create: `web-v2/backend/app/auth/routes.py`
- Create: `web-v2/backend/app/errors.py`
- Modify: `web-v2/backend/app/main.py` (mount router)
- Create: `web-v2/backend/tests/test_auth.py`

- [ ] **Step 1: `app/errors.py` — unified error envelope**

```python
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

I18N = {
    "validation_failed": ("Деректер дұрыс емес", "Невалидные данные"),
    "auth_required": ("Кіру қажет", "Требуется вход"),
    "not_owner": ("Бұл партия сізге тиесілі емес", "Эта партия не принадлежит вам"),
    "not_found": ("Табылмады", "Не найдено"),
    "username_taken": ("Бұл логин бос емес", "Логин занят"),
    "invalid_credentials": ("Логин не пароль қате", "Неверный логин или пароль"),
    "rate_limited": ("Тым жиі. Кейінірек көріңіз.", "Слишком часто. Попробуйте позже."),
    "engine_unavailable": ("Қозғалтқыш қол жетімсіз", "Движок недоступен"),
    "illegal_move": ("Заңсыз жүріс", "Недопустимый ход"),
    "game_not_active": ("Партия аяқталған", "Партия завершена"),
    "internal": ("Серверде қате", "Ошибка сервера"),
}


class AppError(HTTPException):
    def __init__(self, code: str, http: int, details: dict | None = None):
        super().__init__(status_code=http, detail={"code": code, "details": details or {}})
        self.code = code


def _envelope(code: str, details: dict | None = None) -> dict:
    kk, ru = I18N.get(code, ("Қате", "Ошибка"))
    return {"error": {"code": code, "messageKk": kk, "messageRu": ru, "details": details or {}}}


def install_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error(_, exc: AppError):
        return JSONResponse(status_code=exc.status_code, content=_envelope(exc.code, exc.detail.get("details")))

    @app.exception_handler(ValidationError)
    async def validation(_, exc: ValidationError):
        return JSONResponse(status_code=400, content=_envelope("validation_failed", {"errors": exc.errors()}))

    @app.exception_handler(Exception)
    async def fallback(_, exc: Exception):
        return JSONResponse(status_code=500, content=_envelope("internal", {"type": exc.__class__.__name__}))
```

Call `install_handlers(app)` from `create_app()`.

- [ ] **Step 2: `app/auth/schemas.py`**

```python
from pydantic import BaseModel, Field, field_validator


class RegisterReq(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=6, max_length=128)
    locale: str = Field(default="kk", pattern="^(kk|ru)$")

    @field_validator("username")
    @classmethod
    def lowercase_alnum(cls, v: str) -> str:
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("invalid characters")
        return v


class LoginReq(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    id: int
    username: str
    displayName: str | None = None
    locale: str

    model_config = {"from_attributes": True}


class MeOut(BaseModel):
    kind: str  # 'user' | 'anon'
    user: UserOut | None = None
    anonId: str | None = None
```

- [ ] **Step 3: `app/auth/service.py`**

```python
import bcrypt
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import User, Game
from app.errors import AppError


def hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt(rounds=12)).decode()


def verify_password(pw: str, hashed: str) -> bool:
    return bcrypt.checkpw(pw.encode(), hashed.encode())


async def register(s: AsyncSession, *, username: str, password: str, locale: str, anon_session_id: str | None) -> tuple[User, int]:
    existing = (await s.execute(select(User).where(User.username == username))).scalar_one_or_none()
    if existing:
        raise AppError("username_taken", 409)
    user = User(username=username, password_hash=hash_password(password), locale=locale)
    s.add(user)
    await s.flush()
    migrated = 0
    if anon_session_id:
        result = await s.execute(
            update(Game)
            .where(Game.anon_session_id == anon_session_id)
            .values(user_id=user.id, anon_session_id=None)
        )
        migrated = result.rowcount or 0
    await s.commit()
    await s.refresh(user)
    return user, migrated


async def login(s: AsyncSession, *, username: str, password: str) -> User:
    user = (await s.execute(select(User).where(User.username == username))).scalar_one_or_none()
    if user is None or not verify_password(password, user.password_hash):
        raise AppError("invalid_credentials", 401)
    return user
```

- [ ] **Step 4: `app/auth/routes.py`**

```python
from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession
from app.auth.schemas import RegisterReq, LoginReq, UserOut, MeOut
from app.auth.session import CurrentSession
from app.auth.jwt import encode_user
from app.auth.anonymous import COOKIE_JWT
from app.auth.service import register as svc_register, login as svc_login
from app.deps import get_db, get_session
from app.config import settings

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _set_jwt(response: Response, user_id: int) -> None:
    response.set_cookie(
        COOKIE_JWT, encode_user(user_id),
        max_age=settings.jwt_ttl_days * 24 * 3600,
        httponly=True, samesite="lax", secure=settings.env == "prod",
    )


@router.post("/register")
async def register(req: RegisterReq, response: Response, sess: CurrentSession = Depends(get_session), db: AsyncSession = Depends(get_db)):
    user, migrated = await svc_register(
        db, username=req.username, password=req.password, locale=req.locale,
        anon_session_id=sess.anon.id if sess.anon else None,
    )
    _set_jwt(response, user.id)
    return {"user": UserOut.model_validate(user, from_attributes=True), "migratedGamesCount": migrated}


@router.post("/login")
async def login(req: LoginReq, response: Response, db: AsyncSession = Depends(get_db)):
    user = await svc_login(db, username=req.username, password=req.password)
    _set_jwt(response, user.id)
    return {"user": UserOut.model_validate(user, from_attributes=True)}


@router.post("/logout", status_code=204)
async def logout(response: Response):
    response.delete_cookie(COOKIE_JWT)


@router.get("/me", response_model=MeOut)
async def me(sess: CurrentSession = Depends(get_session)):
    if sess.user:
        return MeOut(kind="user", user=UserOut.model_validate(sess.user, from_attributes=True))
    return MeOut(kind="anon", anonId=sess.anon.id)
```

Mount in `create_app()`: `app.include_router(auth.routes.router)`.

- [ ] **Step 5: Tests**

`tests/test_auth.py`:
```python
async def test_register_creates_user_and_sets_jwt(client):
    r = await client.post("/api/auth/register", json={"username": "alice", "password": "secret123"})
    assert r.status_code == 200
    assert r.json()["user"]["username"] == "alice"
    assert "auth_token=" in r.headers["set-cookie"]


async def test_login_invalid_credentials(client):
    await client.post("/api/auth/register", json={"username": "bob", "password": "secret123"})
    r = await client.post("/api/auth/login", json={"username": "bob", "password": "wrong"})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "invalid_credentials"


async def test_me_returns_anon_for_fresh_client(client):
    r = await client.get("/api/auth/me")
    assert r.json()["kind"] == "anon"


async def test_username_taken(client):
    await client.post("/api/auth/register", json={"username": "carol", "password": "secret123"})
    r = await client.post("/api/auth/register", json={"username": "carol", "password": "secret123"})
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "username_taken"
```

Use a fresh test DB per test — update `conftest.py` to set `settings.database_url` to a tmp file and `alembic upgrade head` per session:
```python
import os, pytest
from pathlib import Path

@pytest.fixture(autouse=True, scope="session")
def _test_db(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("db") / "test.db"
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{db_path}"
    # Reload settings & run migrations
    from importlib import reload
    from app import config as cfg_mod
    reload(cfg_mod)
    import subprocess
    subprocess.run(["alembic", "upgrade", "head"], check=True, cwd=str(Path(__file__).parent.parent))
    yield
```
(If reload-based config swap is fiddly, the simpler approach is per-test `engine = create_async_engine(...)` plus `Base.metadata.create_all` — replace fixture above.)

Run `pytest tests/test_auth.py -v` → all PASS.

- [ ] **Step 6: Commit**

```bash
git add web-v2/backend/app/auth web-v2/backend/app/errors.py web-v2/backend/app/main.py web-v2/backend/tests/test_auth.py web-v2/backend/tests/conftest.py
git commit -m "feat(web-v2): auth endpoints (register/login/logout/me) with JWT cookie"
```

---

## Task 6: Anon-to-user migration on register (already in service.py); login-time migration + tests

**Files:**
- Modify: `web-v2/backend/app/auth/routes.py` (login also migrates)
- Modify: `web-v2/backend/app/auth/service.py` (extract migration helper)
- Create: `web-v2/backend/tests/test_anon_migration.py`

- [ ] **Step 1: Extract migration helper in `service.py`**

```python
async def migrate_anon_games(s: AsyncSession, *, user_id: int, anon_session_id: str) -> int:
    result = await s.execute(
        update(Game).where(Game.anon_session_id == anon_session_id).values(user_id=user_id, anon_session_id=None)
    )
    return result.rowcount or 0
```

Use it from both `register` and a new `login_with_migration` flow. In `register`, call it instead of inline UPDATE.

- [ ] **Step 2: Extend login route**

```python
@router.post("/login")
async def login(req: LoginReq, response: Response, sess: CurrentSession = Depends(get_session), db: AsyncSession = Depends(get_db)):
    user = await svc_login(db, username=req.username, password=req.password)
    migrated = 0
    if sess.anon:
        migrated = await migrate_anon_games(db, user_id=user.id, anon_session_id=sess.anon.id)
        await db.commit()
    _set_jwt(response, user.id)
    return {"user": UserOut.model_validate(user, from_attributes=True), "migratedGamesCount": migrated}
```

- [ ] **Step 3: Test**

`tests/test_anon_migration.py`:
```python
from app.db.base import SessionLocal
from app.db.models import Game


async def test_anon_game_migrates_on_login(client):
    # Anon user
    await client.get("/api/auth/me")  # ensures anon cookie set
    cookies = client.cookies
    # Manually create a game for the anon session (via DB; play endpoint not yet implemented)
    anon_id = cookies.get("anon_session")
    async with SessionLocal() as s:
        s.add(Game(anon_session_id=anon_id, mode="solo", side=0, opponent_kind="engine",
                   clock_initial_ms=0, clock_increment_ms=0, clock_white_ms=0, clock_black_ms=0,
                   start_fen="x", current_fen="x", status="active"))
        await s.commit()

    # Register transfers ownership
    r = await client.post("/api/auth/register", json={"username": "dan", "password": "secret123"})
    assert r.json()["migratedGamesCount"] == 1
```

Run `pytest tests/test_anon_migration.py -v`.

- [ ] **Step 4: Commit**

```bash
git commit -am "feat(web-v2): migrate anonymous games on register + login"
```

---

## Task 7: Engine subprocess wrapper

**Files:**
- Create: `web-v2/backend/app/engine/__init__.py` (empty)
- Create: `web-v2/backend/app/engine/protocol_notes.md` — documented engine wire protocol
- Create: `web-v2/backend/app/engine/stream.py` — parse engine `info`/`bestmove` lines
- Create: `web-v2/backend/app/engine/process.py` — low-level subprocess wrapper
- Create: `web-v2/backend/tests/test_engine_stream.py`
- Create: `web-v2/backend/tests/test_engine_process.py`

- [ ] **Step 0: Document the engine protocol from existing code**

This is a hard prerequisite. The engine's wire protocol is not specified anywhere — `web/server.py` is the only source of truth. Before writing any code, read it end to end and capture answers to:

```bash
cat web/server.py
```

Write `app/engine/protocol_notes.md` with:
1. Exact `serve` mode startup sequence (any handshake? any banner line to expect?).
2. `position` command syntax — does it accept `position fen <FEN>` or `position startpos`?
3. How to apply a single move and get the new FEN back. (Either: send `position fen X moves 1-3` and the engine echoes the resulting FEN; or send `apply 1-3` after `position`; or the engine never echoes the FEN and the backend must implement Togyzkumalak rules locally — in which case STOP and escalate, because that is a much bigger task.)
4. Format of `info` lines — what tokens exist (`depth`, `score cp`, `pv`, `nodes`, `time`, `nps`)?
5. Format of `bestmove` line — `bestmove 1-3`? `bestmove 1-3 ponder ...`?
6. The exact start FEN — copy the literal string the engine emits or accepts.
7. Whether `info` lines stream during `go movetime` or only at the end.

If the engine does NOT have a way to return a FEN after a move, this entire plan must pause and you escalate to discuss either: (a) extending the Rust engine to support a FEN-after-move command, or (b) implementing rule application in the backend. Do not paper over this with a placeholder.

Once protocol_notes.md is written, the rest of Task 7 implements exactly what's documented there.

- [ ] **Step 1: `engine/stream.py` — pure parser**

The token list below assumes the standard tokens documented in `protocol_notes.md` from Step 0. If the engine's tokens differ, adjust this parser to match the documented format.

```python
from dataclasses import dataclass


@dataclass
class InfoLine:
    depth: int | None
    cp: int | None
    pv: list[str]
    nodes: int | None
    time_ms: int | None


@dataclass
class BestMove:
    move: str


def parse_line(line: str) -> InfoLine | BestMove | None:
    line = line.strip()
    if line.startswith("bestmove "):
        return BestMove(move=line.split()[1])
    if not line.startswith("info "):
        return None
    tokens = line.split()
    out = InfoLine(depth=None, cp=None, pv=[], nodes=None, time_ms=None)
    i = 1
    while i < len(tokens):
        tok = tokens[i]
        if tok == "depth": out.depth = int(tokens[i+1]); i += 2
        elif tok == "score" and tokens[i+1] == "cp": out.cp = int(tokens[i+2]); i += 3
        elif tok == "nodes": out.nodes = int(tokens[i+1]); i += 2
        elif tok == "time": out.time_ms = int(tokens[i+1]); i += 2
        elif tok == "pv": out.pv = tokens[i+1:]; break
        else: i += 1
    return out
```

`tests/test_engine_stream.py`:
```python
from app.engine.stream import parse_line, InfoLine, BestMove


def test_parse_info_with_pv():
    out = parse_line("info depth 8 score cp 24 nodes 1234 time 12 pv 1-3 2-1 9-2")
    assert out == InfoLine(depth=8, cp=24, pv=["1-3","2-1","9-2"], nodes=1234, time_ms=12)


def test_parse_bestmove():
    assert parse_line("bestmove 1-3") == BestMove(move="1-3")


def test_parse_unknown_returns_none():
    assert parse_line("hello") is None
```

Run → PASS.

- [ ] **Step 2: `engine/process.py` — subprocess wrapper**

```python
import asyncio
from pathlib import Path
from app.engine.stream import parse_line, InfoLine, BestMove


class EngineProcess:
    def __init__(self, binary: Path):
        self.binary = binary
        self._proc: asyncio.subprocess.Process | None = None

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def start(self) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            str(self.binary), "serve",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def stop(self) -> None:
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=2)
            except asyncio.TimeoutError:
                self._proc.kill()

    async def send(self, cmd: str) -> None:
        assert self._proc and self._proc.stdin
        self._proc.stdin.write((cmd + "\n").encode())
        await self._proc.stdin.drain()

    async def think(self, *, position_fen: str, time_ms: int):
        """Yields InfoLine objects until the engine emits BestMove, which is yielded last."""
        await self.send(f"position fen {position_fen}")
        await self.send(f"go movetime {time_ms}")
        assert self._proc and self._proc.stdout
        while True:
            raw = await self._proc.stdout.readline()
            if not raw:
                raise RuntimeError("engine closed pipe")
            parsed = parse_line(raw.decode(errors="replace"))
            if parsed is None:
                continue
            yield parsed
            if isinstance(parsed, BestMove):
                return

    async def apply_move(self, *, position_fen: str, move: str) -> str:
        """Send a move to the engine and read the FEN it returns. Implementation depends
        on the protocol documented in protocol_notes.md from Step 0. Replace the body with
        the exact command sequence the engine actually accepts; the docstring is the only
        thing about this method that is universal."""
        raise NotImplementedError("implement using protocol_notes.md")
```

(Implementing `apply_move` correctly is mandatory before Task 11. If the engine has no such command, this is the escalation point flagged in Step 0.)

- [ ] **Step 3: Integration test (skips if binary missing)**

`tests/test_engine_process.py`:
```python
import pytest
from pathlib import Path
from app.config import settings
from app.engine.process import EngineProcess
from app.engine.stream import BestMove


@pytest.mark.skipif(not Path(settings.engine_path).exists(), reason="engine binary not built")
async def test_engine_emits_bestmove():
    proc = EngineProcess(settings.engine_path)
    await proc.start()
    try:
        # Standard start FEN — replace with the actual one if the engine is strict
        out = []
        async for ev in proc.think(position_fen="9/9 0 0 w 1", time_ms=100):
            out.append(ev)
        assert isinstance(out[-1], BestMove)
    finally:
        await proc.stop()
```

(Substitute the real start FEN format from existing `web/server.py` — read it once, copy verbatim.)

Run → PASS or SKIP if engine isn't built. If running, build engine first: `cd engine && cargo build --release`.

- [ ] **Step 4: Commit**

```bash
git add web-v2/backend/app/engine web-v2/backend/tests/test_engine_*.py
git commit -m "feat(web-v2): engine subprocess wrapper + info/bestmove parser"
```

---

## Task 8: Engine pool — lock + subscribers + auto-restart

**Files:**
- Create: `web-v2/backend/app/engine/pool.py`
- Modify: `web-v2/backend/app/main.py` — start/stop pool in lifespan
- Modify: `web-v2/backend/app/deps.py` — add `get_engine`
- Create: `web-v2/backend/tests/test_engine_pool.py`

- [ ] **Step 1: `engine/pool.py`**

```python
import asyncio
from dataclasses import dataclass
from pathlib import Path
from app.engine.process import EngineProcess
from app.engine.stream import InfoLine, BestMove


@dataclass
class EngineResult:
    move: str
    final_eval_cp: int | None
    final_depth: int | None
    think_time_ms: int


class EngineError(RuntimeError):
    pass


class EnginePool:
    def __init__(self, binary: Path):
        self._binary = binary
        self._proc = EngineProcess(binary)
        self._lock = asyncio.Lock()
        self._subs: dict[int, list[asyncio.Queue]] = {}
        self._build_hash = "unknown"

    @property
    def alive(self) -> bool:
        return self._proc.alive

    @property
    def build_hash(self) -> str:
        return self._build_hash

    async def start(self) -> None:
        await self._proc.start()
        try:
            import hashlib
            self._build_hash = hashlib.sha256(self._binary.read_bytes()).hexdigest()[:12]
        except Exception:
            self._build_hash = "unknown"

    async def stop(self) -> None:
        await self._proc.stop()

    def subscribe(self, game_id: int, q: asyncio.Queue) -> None:
        self._subs.setdefault(game_id, []).append(q)

    def unsubscribe(self, game_id: int, q: asyncio.Queue) -> None:
        lst = self._subs.get(game_id, [])
        if q in lst:
            lst.remove(q)
        if not lst:
            self._subs.pop(game_id, None)

    async def think(self, *, position_fen: str, time_ms: int, game_id: int) -> EngineResult:
        async with self._lock:
            for attempt in (0, 1):
                if not self._proc.alive:
                    try:
                        await self._proc.start()
                    except Exception as e:
                        if attempt == 1:
                            raise EngineError("engine_start_failed") from e
                        continue
                try:
                    last_info: InfoLine | None = None
                    async for ev in self._proc.think(position_fen=position_fen, time_ms=time_ms):
                        if isinstance(ev, InfoLine):
                            last_info = ev
                            for q in list(self._subs.get(game_id, [])):
                                if q.full():
                                    try: q.get_nowait()
                                    except asyncio.QueueEmpty: pass
                                q.put_nowait(("eval", ev))
                        elif isinstance(ev, BestMove):
                            return EngineResult(
                                move=ev.move,
                                final_eval_cp=last_info.cp if last_info else None,
                                final_depth=last_info.depth if last_info else None,
                                think_time_ms=last_info.time_ms if last_info else time_ms,
                            )
                    raise EngineError("engine_no_bestmove")
                except RuntimeError:
                    await self._proc.stop()
                    if attempt == 1:
                        raise EngineError("engine_died")
            raise EngineError("engine_unreachable")
```

- [ ] **Step 2: Lifespan in `main.py`**

```python
from contextlib import asynccontextmanager
from app.engine.pool import EnginePool
from app.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = EnginePool(settings.engine_path)
    try:
        await pool.start()
    except Exception:
        # Engine binary missing — pool stays dead; play endpoints return 503
        pass
    app.state.engine_pool = pool
    yield
    await pool.stop()

# In create_app:
app = FastAPI(..., lifespan=lifespan)
```

`app/deps.py`:
```python
from fastapi import Request
from app.engine.pool import EnginePool

def get_engine(request: Request) -> EnginePool:
    return request.app.state.engine_pool
```

- [ ] **Step 3: Tests with a fake EngineProcess**

`tests/test_engine_pool.py`:
```python
import asyncio, pytest
from app.engine.pool import EnginePool, EngineError
from app.engine.stream import InfoLine, BestMove
from pathlib import Path


class FakeProc:
    def __init__(self, binary): self.alive = False; self._fail_next = False
    async def start(self):
        if self._fail_next: self._fail_next = False; raise RuntimeError("boom")
        self.alive = True
    async def stop(self): self.alive = False
    async def think(self, *, position_fen, time_ms):
        yield InfoLine(depth=8, cp=12, pv=["1-3"], nodes=10, time_ms=time_ms)
        yield BestMove(move="1-3")


async def test_think_returns_bestmove(monkeypatch):
    pool = EnginePool(Path("/dev/null"))
    monkeypatch.setattr(pool, "_proc", FakeProc(None))
    res = await pool.think(position_fen="x", time_ms=100, game_id=1)
    assert res.move == "1-3"
    assert res.final_eval_cp == 12


async def test_think_fans_out_to_subscribers(monkeypatch):
    pool = EnginePool(Path("/dev/null"))
    monkeypatch.setattr(pool, "_proc", FakeProc(None))
    q = asyncio.Queue()
    pool.subscribe(7, q)
    await pool.think(position_fen="x", time_ms=100, game_id=7)
    kind, ev = await asyncio.wait_for(q.get(), timeout=0.5)
    assert kind == "eval"
    assert ev.cp == 12


async def test_think_serializes_concurrent(monkeypatch):
    pool = EnginePool(Path("/dev/null"))
    proc = FakeProc(None)
    monkeypatch.setattr(pool, "_proc", proc)
    res = await asyncio.gather(
        pool.think(position_fen="x", time_ms=10, game_id=1),
        pool.think(position_fen="y", time_ms=10, game_id=2),
    )
    assert all(r.move == "1-3" for r in res)
```

Run → PASS.

- [ ] **Step 4: Commit**

```bash
git add web-v2/backend/app/engine/pool.py web-v2/backend/app/main.py web-v2/backend/app/deps.py web-v2/backend/tests/test_engine_pool.py
git commit -m "feat(web-v2): engine pool with asyncio lock + per-game subscribers"
```

---

## Task 9: Clock arithmetic + GameState snapshot builder

**Files:**
- Create: `web-v2/backend/app/play/__init__.py` (empty)
- Create: `web-v2/backend/app/play/clock.py`
- Create: `web-v2/backend/app/play/snapshot.py`
- Create: `web-v2/backend/app/play/schemas.py`
- Create: `web-v2/backend/tests/test_clock.py`
- Create: `web-v2/backend/tests/test_snapshot.py`

- [ ] **Step 1: `play/clock.py`**

```python
from datetime import datetime, timezone
from dataclasses import dataclass
from app.db.models import Game


@dataclass
class ClockState:
    initial_ms: int
    increment_ms: int
    white_ms: int
    black_ms: int
    running_side: int | None  # 0|1|None


def project_clock(game: Game, now: datetime | None = None) -> ClockState:
    if game.clock_initial_ms == 0:
        return ClockState(0, 0, 0, 0, None)
    now = now or datetime.now(timezone.utc)
    elapsed_ms = 0
    if game.last_clock_at and game.status == "active":
        delta = now - game.last_clock_at.replace(tzinfo=timezone.utc) if game.last_clock_at.tzinfo is None else now - game.last_clock_at
        elapsed_ms = int(delta.total_seconds() * 1000)
    w, b = game.clock_white_ms, game.clock_black_ms
    if game.status == "active":
        if game.side_to_move == 0:
            w = max(0, w - elapsed_ms)
        else:
            b = max(0, b - elapsed_ms)
    return ClockState(
        initial_ms=game.clock_initial_ms,
        increment_ms=game.clock_increment_ms,
        white_ms=w, black_ms=b,
        running_side=game.side_to_move if game.status == "active" else None,
    )


def apply_move_to_clock(game: Game, now: datetime) -> tuple[int, int]:
    """After a move is committed: subtract elapsed, add increment to the side that just moved.
    Returns (new_white_ms, new_black_ms). Caller updates game.last_clock_at to `now`.
    """
    if game.clock_initial_ms == 0:
        return game.clock_white_ms, game.clock_black_ms
    proj = project_clock(game, now)
    side_just_moved = game.side_to_move  # the one whose clock was running
    w, b = proj.white_ms, proj.black_ms
    if side_just_moved == 0:
        w = w + game.clock_increment_ms
    else:
        b = b + game.clock_increment_ms
    return w, b
```

- [ ] **Step 2: Tests**

`tests/test_clock.py`:
```python
from datetime import datetime, timedelta, timezone
from app.db.models import Game
from app.play.clock import project_clock, apply_move_to_clock


def _g(**over):
    base = dict(
        clock_initial_ms=300_000, clock_increment_ms=2000,
        clock_white_ms=300_000, clock_black_ms=300_000,
        last_clock_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        side_to_move=0, status="active",
    )
    base.update(over)
    return Game(**{k: v for k, v in base.items() if k in Game.__table__.columns.keys()})


def test_clock_zero_means_no_clock():
    g = _g(clock_initial_ms=0, clock_white_ms=0, clock_black_ms=0)
    s = project_clock(g)
    assert s.running_side is None and s.white_ms == 0


def test_clock_decrements_running_side_only():
    g = _g()
    now = g.last_clock_at + timedelta(seconds=10)
    s = project_clock(g, now=now)
    assert s.white_ms == 290_000
    assert s.black_ms == 300_000
    assert s.running_side == 0


def test_increment_added_after_move():
    g = _g()
    now = g.last_clock_at + timedelta(seconds=10)
    w, b = apply_move_to_clock(g, now)
    assert w == 290_000 + 2000
    assert b == 300_000


def test_clock_does_not_go_negative():
    g = _g(clock_white_ms=5000)
    now = g.last_clock_at + timedelta(seconds=999)
    s = project_clock(g, now=now)
    assert s.white_ms == 0
```

- [ ] **Step 3: `play/schemas.py` — Pydantic shapes for API**

```python
from pydantic import BaseModel, Field
from typing import Literal


class ClockOut(BaseModel):
    initialMs: int
    incrementMs: int
    whiteMs: int
    blackMs: int
    runningSide: Literal[0, 1] | None


class MoveOut(BaseModel):
    ply: int
    side: int
    actor: str
    moveUci: str
    fenAfter: str
    evalCp: int | None = None
    evalDepth: int | None = None
    thinkTimeMs: int | None = None
    clockAfterMs: int | None = None


class EventOut(BaseModel):
    plyAt: int
    actor: str
    type: str
    payload: dict | None = None


class GameStateOut(BaseModel):
    id: int
    mode: str
    side: int
    status: str
    result: str | None = None
    resultReason: str | None = None
    finalScore: str | None = None
    startFen: str
    currentFen: str
    currentPly: int
    sideToMove: int
    clock: ClockOut
    engineThinking: bool
    hintsUsed: int
    hintsLimit: int
    moves: list[MoveOut] = []
    events: list[EventOut] = []
    startedAt: str
    finishedAt: str | None = None


class NewGameReq(BaseModel):
    side: int = Field(ge=0, le=1)
    engineLevel: Literal["easy", "normal", "hard"] = "normal"
    clock: dict | None = None  # {"initialMs": int, "incrementMs": int}
    useBook: bool = False
    startFen: str | None = None


class MoveReq(BaseModel):
    moveUci: str = Field(min_length=2, max_length=8)


class TakebackReq(BaseModel):
    toPly: int = Field(ge=0)
```

- [ ] **Step 4: `play/snapshot.py`**

```python
import json
from app.db.models import Game
from app.play.clock import project_clock
from app.play.schemas import GameStateOut, ClockOut, MoveOut, EventOut


HINTS_LIMIT = 3


def build_snapshot(game: Game, *, hints_used: int = 0, engine_thinking: bool = False) -> GameStateOut:
    clock = project_clock(game)
    moves = [MoveOut(
        ply=m.ply, side=m.side, actor=m.actor, moveUci=m.move_uci, fenAfter=m.fen_after,
        evalCp=m.eval_cp, evalDepth=m.eval_depth, thinkTimeMs=m.think_time_ms, clockAfterMs=m.clock_after_ms,
    ) for m in game.moves]
    events = [EventOut(
        plyAt=e.ply_at, actor=e.actor, type=e.type,
        payload=json.loads(e.payload_json) if e.payload_json else None,
    ) for e in game.events]
    return GameStateOut(
        id=game.id, mode=game.mode, side=game.side, status=game.status,
        result=game.result, resultReason=game.result_reason, finalScore=game.final_score,
        startFen=game.start_fen, currentFen=game.current_fen,
        currentPly=game.current_ply, sideToMove=game.side_to_move,
        clock=ClockOut(initialMs=clock.initial_ms, incrementMs=clock.increment_ms,
                       whiteMs=clock.white_ms, blackMs=clock.black_ms, runningSide=clock.running_side),
        engineThinking=engine_thinking,
        hintsUsed=hints_used, hintsLimit=HINTS_LIMIT,
        moves=moves, events=events,
        startedAt=game.started_at.isoformat(),
        finishedAt=game.finished_at.isoformat() if game.finished_at else None,
    )
```

`tests/test_snapshot.py` — minimal shape check:
```python
from datetime import datetime, timezone
from app.db.models import Game
from app.play.snapshot import build_snapshot


def test_snapshot_basic_shape():
    g = Game(
        id=1, mode="solo", side=0, opponent_kind="engine",
        clock_initial_ms=0, clock_increment_ms=0, clock_white_ms=0, clock_black_ms=0,
        start_fen="x", current_fen="x", current_ply=0, side_to_move=0,
        status="active", started_at=datetime(2026,1,1,tzinfo=timezone.utc),
    )
    g.moves = []; g.events = []
    s = build_snapshot(g)
    assert s.id == 1 and s.engineThinking is False
```

- [ ] **Step 5: Commit**

```bash
git add web-v2/backend/app/play/{__init__,clock,snapshot,schemas}.py web-v2/backend/tests/test_{clock,snapshot}.py
git commit -m "feat(web-v2): clock arithmetic + GameState snapshot builder"
```

---

## Task 10: Play service — apply_move, takeback, resign, undo

**Files:**
- Create: `web-v2/backend/app/play/service.py`
- Create: `web-v2/backend/tests/test_play_service.py`

> The service is the only place that writes to `games`/`moves`/`game_events`. Routes wrap it.

- [ ] **Step 1: Service skeleton**

```python
import json
from datetime import datetime, timezone
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import Game, Move, GameEvent
from app.engine.pool import EnginePool, EngineError
from app.errors import AppError
from app.play.clock import apply_move_to_clock, project_clock


LEVEL_TO_MS = {"easy": 500, "normal": 2000, "hard": 8000}


async def create_game(s: AsyncSession, *, owner: dict, side: int, engine_level: str,
                     clock: dict | None, use_book: bool, start_fen: str, engine_build: str) -> Game:
    initial = clock["initialMs"] if clock else 0
    increment = clock["incrementMs"] if clock else 0
    g = Game(
        **owner,
        mode="solo", side=side, opponent_kind="engine", opponent_ref=engine_build,
        engine_level=engine_level,
        clock_initial_ms=initial, clock_increment_ms=increment,
        clock_white_ms=initial, clock_black_ms=initial,
        last_clock_at=datetime.now(timezone.utc) if initial else None,
        start_fen=start_fen, current_fen=start_fen,
        current_ply=0, side_to_move=0, status="active",
    )
    s.add(g)
    await s.commit()
    await s.refresh(g)
    return g


async def apply_move(s: AsyncSession, *, game: Game, move_uci: str, actor: str,
                     fen_after: str, eval_cp: int | None = None, eval_depth: int | None = None,
                     pv: list[str] | None = None, think_time_ms: int | None = None) -> Move:
    if game.status != "active":
        raise AppError("game_not_active", 400)
    now = datetime.now(timezone.utc)
    new_w, new_b = apply_move_to_clock(game, now)
    side = game.side_to_move
    clock_after = new_w if side == 0 else new_b
    move = Move(
        game_id=game.id, ply=game.current_ply + 1, side=side, actor=actor,
        move_uci=move_uci, fen_after=fen_after,
        eval_cp=eval_cp, eval_depth=eval_depth,
        pv=json.dumps(pv) if pv else None,
        think_time_ms=think_time_ms, clock_after_ms=clock_after,
    )
    s.add(move)
    game.current_ply += 1
    game.current_fen = fen_after
    game.side_to_move = 1 - side
    game.clock_white_ms, game.clock_black_ms = new_w, new_b
    game.last_clock_at = now
    await s.commit()
    await s.refresh(game)
    return move


async def takeback(s: AsyncSession, *, game: Game, to_ply: int) -> None:
    if to_ply >= game.current_ply or to_ply < 0:
        raise AppError("validation_failed", 400, {"reason": "to_ply out of range"})
    await s.execute(delete(Move).where(Move.game_id == game.id, Move.ply > to_ply))
    if to_ply == 0:
        game.current_fen = game.start_fen
        game.side_to_move = 0
    else:
        last = (await s.execute(select(Move).where(Move.game_id == game.id, Move.ply == to_ply))).scalar_one()
        game.current_fen = last.fen_after
        game.side_to_move = 1 - last.side
    game.current_ply = to_ply
    s.add(GameEvent(
        game_id=game.id, ply_at=game.current_ply, actor="human",
        type="takeback", payload_json=json.dumps({"toPly": to_ply}),
    ))
    await s.commit()
    await s.refresh(game)


async def resign(s: AsyncSession, *, game: Game) -> None:
    if game.status != "active":
        raise AppError("game_not_active", 400)
    game.status = "finished"
    game.result = "win_black" if game.side == 0 else "win_white"
    game.result_reason = "resign"
    game.finished_at = datetime.now(timezone.utc)
    s.add(GameEvent(game_id=game.id, ply_at=game.current_ply, actor="human", type="resign"))
    await s.commit()
    await s.refresh(game)


async def undo_last_pair(s: AsyncSession, *, game: Game) -> None:
    """Roll back the most recent human move and the engine's reply (if any)."""
    if game.current_ply == 0:
        return
    target = game.current_ply
    while target > 0:
        m = (await s.execute(select(Move).where(Move.game_id == game.id, Move.ply == target))).scalar_one()
        target -= 1
        if m.actor == "human":
            break
    await takeback(s, game=game, to_ply=target)
```

- [ ] **Step 2: Tests for service (use a real DB session via fixture)**

`tests/test_play_service.py`:
```python
from app.db.base import SessionLocal
from app.db.models import AnonSession
from app.play.service import create_game, apply_move, takeback, resign


async def _new_anon():
    async with SessionLocal() as s:
        a = AnonSession(id="test-anon")
        s.add(a)
        await s.commit()
    return "test-anon"


async def test_create_game_persists():
    anon_id = await _new_anon()
    async with SessionLocal() as s:
        g = await create_game(s, owner={"anon_session_id": anon_id}, side=0,
                              engine_level="normal", clock=None, use_book=False,
                              start_fen="X", engine_build="abc")
        assert g.id is not None and g.status == "active"


async def test_apply_move_increments_ply():
    anon_id = await _new_anon()
    async with SessionLocal() as s:
        g = await create_game(s, owner={"anon_session_id": anon_id}, side=0,
                              engine_level="normal", clock=None, use_book=False,
                              start_fen="X", engine_build="abc")
        await apply_move(s, game=g, move_uci="1-3", actor="human", fen_after="Y")
        assert g.current_ply == 1
        assert g.current_fen == "Y"
        assert g.side_to_move == 1


async def test_takeback_to_zero_resets_to_start():
    anon_id = await _new_anon()
    async with SessionLocal() as s:
        g = await create_game(s, owner={"anon_session_id": anon_id}, side=0,
                              engine_level="normal", clock=None, use_book=False,
                              start_fen="X", engine_build="abc")
        await apply_move(s, game=g, move_uci="1-3", actor="human", fen_after="Y")
        await apply_move(s, game=g, move_uci="2-1", actor="engine", fen_after="Z")
        await takeback(s, game=g, to_ply=0)
        assert g.current_ply == 0 and g.current_fen == "X"
```

(Adjust fixtures so each test gets a fresh in-memory or temp DB to avoid leaking state.)

- [ ] **Step 3: Commit**

```bash
git add web-v2/backend/app/play/service.py web-v2/backend/tests/test_play_service.py
git commit -m "feat(web-v2): play service — create / move / takeback / resign / undo"
```

---

## Task 11: Play routes — REST endpoints + hint rate limit + draw_offer

**Files:**
- Create: `web-v2/backend/app/play/routes.py`
- Modify: `web-v2/backend/app/main.py` (mount router, install slowapi)
- Create: `web-v2/backend/tests/test_play_routes.py`

> The default start FEN must come from `protocol_notes.md` written in Task 7 Step 0. Hardcode it as `START_FEN` constant in `play/routes.py`. If the documented FEN format ever changes, the constant and the parser are the two places to update.

- [ ] **Step 1: Slowapi setup in `main.py`**

```python
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse

limiter = Limiter(key_func=get_remote_address)

# in create_app:
app.state.limiter = limiter
@app.exception_handler(RateLimitExceeded)
async def _rate(_, __):
    return JSONResponse(status_code=429, content=_envelope("rate_limited"))
```

- [ ] **Step 2: `play/routes.py`**

```python
import json
from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.deps import get_db, get_session, get_engine
from app.auth.session import CurrentSession
from app.engine.pool import EnginePool, EngineError
from app.errors import AppError
from app.play.schemas import NewGameReq, MoveReq, TakebackReq, GameStateOut
from app.play.service import (
    create_game, apply_move, takeback, resign, undo_last_pair, LEVEL_TO_MS,
)
from app.play.snapshot import build_snapshot, HINTS_LIMIT
from app.db.models import Game, GameEvent
from app.main import limiter


router = APIRouter(prefix="/api/play", tags=["play"])

START_FEN = "9/9 0 0 w 1"  # set this to the literal start FEN documented in app/engine/protocol_notes.md
HINTS_USED: dict[int, int] = {}  # in-memory per-process counter


async def _load_owned(db: AsyncSession, sess: CurrentSession, game_id: int) -> Game:
    g = await db.get(Game, game_id)
    if g is None:
        raise AppError("not_found", 404)
    own_uid = g.user_id
    own_aid = g.anon_session_id
    if (sess.user and own_uid != sess.user.id) or (sess.anon and own_aid != sess.anon.id):
        raise AppError("not_owner", 403)
    return g


@router.post("/new", response_model=dict)
@limiter.limit("30/hour")
async def new_game(request: Request, req: NewGameReq, sess: CurrentSession = Depends(get_session),
                   db: AsyncSession = Depends(get_db), engine: EnginePool = Depends(get_engine)):
    owner = {"user_id": sess.user.id} if sess.user else {"anon_session_id": sess.anon.id}
    g = await create_game(
        db, owner=owner, side=req.side, engine_level=req.engineLevel,
        clock=req.clock, use_book=req.useBook,
        start_fen=req.startFen or START_FEN, engine_build=engine.build_hash,
    )
    return {"game": build_snapshot(g)}


@router.get("/{game_id}", response_model=dict)
async def get_game(game_id: int, sess: CurrentSession = Depends(get_session), db: AsyncSession = Depends(get_db)):
    g = await _load_owned(db, sess, game_id)
    return {"game": build_snapshot(g, hints_used=HINTS_USED.get(g.id, 0))}


@router.post("/{game_id}/move", response_model=dict)
@limiter.limit("60/minute")
async def make_move(request: Request, game_id: int, req: MoveReq,
                    sess: CurrentSession = Depends(get_session),
                    db: AsyncSession = Depends(get_db),
                    engine: EnginePool = Depends(get_engine)):
    g = await _load_owned(db, sess, game_id)
    if g.side_to_move != g.side:
        raise AppError("validation_failed", 400, {"reason": "not your turn"})
    # Engine validates and returns the resulting FEN. If the move is illegal, apply_move
    # raises and the request fails with 400 illegal_move.
    try:
        fen_after = await engine._proc.apply_move(position_fen=g.current_fen, move=req.moveUci)
    except (EngineError, NotImplementedError):
        raise AppError("engine_unavailable", 503)
    except ValueError:
        raise AppError("illegal_move", 400)
    await apply_move(db, game=g, move_uci=req.moveUci, actor="human", fen_after=fen_after)
    # Engine reply is triggered by the WebSocket layer (Task 12); the REST move endpoint
    # only commits the human move and returns. Clients should be connected to /ws/games/{id}
    # to receive the engine's reply via streamed events.
    return {"game": build_snapshot(g, hints_used=HINTS_USED.get(g.id, 0), engine_thinking=True)}


@router.post("/{game_id}/undo", response_model=dict)
async def undo(game_id: int, sess: CurrentSession = Depends(get_session), db: AsyncSession = Depends(get_db)):
    g = await _load_owned(db, sess, game_id)
    await undo_last_pair(db, game=g)
    return {"game": build_snapshot(g, hints_used=HINTS_USED.get(g.id, 0))}


@router.post("/{game_id}/takeback", response_model=dict)
async def do_takeback(game_id: int, req: TakebackReq,
                      sess: CurrentSession = Depends(get_session), db: AsyncSession = Depends(get_db)):
    g = await _load_owned(db, sess, game_id)
    await takeback(db, game=g, to_ply=req.toPly)
    return {"game": build_snapshot(g, hints_used=HINTS_USED.get(g.id, 0))}


@router.post("/{game_id}/resign", response_model=dict)
async def do_resign(game_id: int, sess: CurrentSession = Depends(get_session), db: AsyncSession = Depends(get_db)):
    g = await _load_owned(db, sess, game_id)
    await resign(db, game=g)
    return {"game": build_snapshot(g, hints_used=HINTS_USED.get(g.id, 0))}


@router.post("/{game_id}/draw_offer", response_model=dict)
async def draw_offer(game_id: int, sess: CurrentSession = Depends(get_session),
                     db: AsyncSession = Depends(get_db), engine: EnginePool = Depends(get_engine)):
    g = await _load_owned(db, sess, game_id)
    if g.status != "active":
        raise AppError("game_not_active", 400)
    accepted = False
    try:
        result = await engine.think(position_fen=g.current_fen, time_ms=200, game_id=g.id)
        if result.final_eval_cp is not None and abs(result.final_eval_cp) < 50:
            g.status = "finished"; g.result = "draw"; g.result_reason = "draw_agreement"
            from datetime import datetime, timezone
            g.finished_at = datetime.now(timezone.utc)
            db.add(GameEvent(game_id=g.id, ply_at=g.current_ply, actor="engine", type="draw_accept"))
            await db.commit(); await db.refresh(g)
            accepted = True
        else:
            db.add(GameEvent(game_id=g.id, ply_at=g.current_ply, actor="engine", type="draw_decline"))
            await db.commit()
    except EngineError:
        raise AppError("engine_unavailable", 503)
    return {"game": build_snapshot(g), "accepted": accepted}


@router.post("/{game_id}/hint", response_model=dict)
async def hint(game_id: int, sess: CurrentSession = Depends(get_session),
               db: AsyncSession = Depends(get_db), engine: EnginePool = Depends(get_engine)):
    g = await _load_owned(db, sess, game_id)
    used = HINTS_USED.get(g.id, 0)
    if used >= HINTS_LIMIT:
        raise AppError("rate_limited", 429, {"reason": "hint limit"})
    try:
        r = await engine.think(position_fen=g.current_fen, time_ms=2000, game_id=g.id)
    except EngineError:
        raise AppError("engine_unavailable", 503)
    HINTS_USED[g.id] = used + 1
    db.add(GameEvent(game_id=g.id, ply_at=g.current_ply, actor="system",
                     type="hint_used", payload_json=json.dumps({"move": r.move, "evalCp": r.final_eval_cp})))
    await db.commit()
    return {"move": r.move, "evalCp": r.final_eval_cp, "depth": r.final_depth, "pv": []}
```

Mount in `create_app()`: `app.include_router(play.routes.router)`.

- [ ] **Step 3: Tests**

`tests/test_play_routes.py` — at minimum:
```python
async def test_new_game_creates_active_game(client):
    r = await client.post("/api/play/new", json={"side": 0, "engineLevel": "normal", "clock": None, "useBook": False})
    assert r.status_code == 200
    g = r.json()["game"]
    assert g["status"] == "active" and g["currentPly"] == 0


async def test_make_move_increments_ply(client):
    r = await client.post("/api/play/new", json={"side": 0, "engineLevel": "normal", "clock": None, "useBook": False})
    gid = r.json()["game"]["id"]
    r2 = await client.post(f"/api/play/{gid}/move", json={"moveUci": "1-3"})
    assert r2.json()["game"]["currentPly"] == 1


async def test_resign_finalizes(client):
    r = await client.post("/api/play/new", json={"side": 0, "engineLevel": "normal", "clock": None, "useBook": False})
    gid = r.json()["game"]["id"]
    r2 = await client.post(f"/api/play/{gid}/resign")
    g = r2.json()["game"]
    assert g["status"] == "finished" and g["result"] == "win_black"


async def test_foreign_game_forbidden(client_factory):
    a = client_factory(); b = client_factory()
    r = await a.post("/api/play/new", json={"side": 0, "engineLevel": "normal", "clock": None, "useBook": False})
    gid = r.json()["game"]["id"]
    r2 = await b.get(f"/api/play/{gid}")
    assert r2.status_code == 403
```

(Add `client_factory` fixture in `conftest.py` that mints a fresh AsyncClient with isolated cookies.)

- [ ] **Step 4: Commit**

```bash
git add web-v2/backend/app/play/routes.py web-v2/backend/app/main.py web-v2/backend/tests/test_play_routes.py
git commit -m "feat(web-v2): play REST routes + hint rate limit + draw offer"
```

---

## Task 12: WebSocket endpoint with eval stream + resume

**Files:**
- Create: `web-v2/backend/app/ws/__init__.py` (empty)
- Create: `web-v2/backend/app/ws/games_ws.py`
- Modify: `web-v2/backend/app/main.py` (register WS route)
- Create: `web-v2/backend/tests/test_ws.py`

- [ ] **Step 1: Per-game ring buffer + dispatcher**

```python
# app/ws/games_ws.py
import asyncio, json
from collections import defaultdict, deque
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from app.deps import get_engine
from app.engine.pool import EnginePool, EngineError
from app.auth.anonymous import resolve_session
from app.db.base import SessionLocal
from app.db.models import Game, GameEvent
from app.play.snapshot import build_snapshot
from app.play.service import apply_move, LEVEL_TO_MS


router = APIRouter()

# State per game (lives in the process; lost on restart, recovered via snapshot on reconnect)
_seq: dict[int, int] = defaultdict(int)
_buf: dict[int, deque] = defaultdict(lambda: deque(maxlen=200))
_clients: dict[int, set[asyncio.Queue]] = defaultdict(set)


def _publish(game_id: int, msg: dict) -> None:
    _seq[game_id] += 1
    msg = {"seq": _seq[game_id], **msg}
    _buf[game_id].append(msg)
    for q in list(_clients[game_id]):
        if q.full():
            try: q.get_nowait()
            except asyncio.QueueEmpty: pass
        q.put_nowait(msg)


async def _send_snapshot(ws: WebSocket, game_id: int) -> None:
    async with SessionLocal() as s:
        g = await s.get(Game, game_id)
        if g is None: return
        await s.refresh(g, ["moves", "events"])
        await ws.send_json({"seq": _seq[game_id], "type": "snapshot", "game": build_snapshot(g).model_dump()})


@router.websocket("/ws/games/{game_id}")
async def games_ws(ws: WebSocket, game_id: int, engine: EnginePool = Depends(get_engine)):
    await ws.accept()
    sess = await resolve_session(ws)  # cookies on handshake
    async with SessionLocal() as s:
        g = await s.get(Game, game_id)
        if g is None:
            await ws.close(code=4004); return
        if (sess.user and g.user_id != sess.user.id) or (sess.anon and g.anon_session_id != sess.anon.id):
            await ws.close(code=4003); return

    q: asyncio.Queue = asyncio.Queue(maxsize=64)
    _clients[game_id].add(q)
    try:
        # Wait for hello — replay or snapshot
        try:
            hello_raw = await asyncio.wait_for(ws.receive_json(), timeout=5)
        except asyncio.TimeoutError:
            hello_raw = {"type": "hello"}
        last_seen = (hello_raw or {}).get("lastSeenSeq")
        replayed = False
        if last_seen is not None and _buf[game_id] and _buf[game_id][0]["seq"] <= last_seen + 1:
            for msg in list(_buf[game_id]):
                if msg["seq"] > last_seen:
                    await ws.send_json(msg)
            replayed = True
        if not replayed:
            await _send_snapshot(ws, game_id)

        # Read+write loop
        async def reader():
            while True:
                msg = await ws.receive_json()
                if msg.get("type") == "ping":
                    await ws.send_json({"type": "pong", "seq": _seq[game_id]})
                elif msg.get("type") == "request_snapshot":
                    await _send_snapshot(ws, game_id)
                elif msg.get("type") == "move":
                    # delegate to REST is fine, but for WS-only flow, call the service directly
                    await _handle_move(game_id, msg["moveUci"], engine)
        async def writer():
            while True:
                msg = await q.get()
                await ws.send_json(msg)

        done, pending = await asyncio.wait(
            [asyncio.create_task(reader()), asyncio.create_task(writer())],
            return_when=asyncio.FIRST_EXCEPTION,
        )
        for t in pending: t.cancel()
    except WebSocketDisconnect:
        pass
    finally:
        _clients[game_id].discard(q)


async def _handle_move(game_id: int, move_uci: str, engine: EnginePool) -> None:
    """Apply a human move, then ask engine for the reply, broadcasting all events.
    The REST endpoint already committed the human move; this WS handler only triggers
    the engine reply and broadcasts. If you arrive here without the move already applied
    (because the client used the WS-only path), call apply_move first."""
    async with SessionLocal() as s:
        g = await s.get(Game, game_id)
        await s.refresh(g, ["moves", "events"])
        if g.status != "active": return
        level_ms = LEVEL_TO_MS.get(g.engine_level, LEVEL_TO_MS["normal"])
        _publish(game_id, {"type": "move_applied", "ply": g.current_ply, "side": 1 - g.side_to_move,
                           "moveUci": move_uci, "fenAfter": fen_after,
                           "clock": {"whiteMs": g.clock_white_ms, "blackMs": g.clock_black_ms,
                                     "runningSide": g.side_to_move}})
        if g.status == "active" and g.side_to_move != g.side:
            _publish(game_id, {"type": "engine_thinking", "started": True})
            # subscribe to engine events for this game
            sub_q: asyncio.Queue = asyncio.Queue(maxsize=64)
            engine.subscribe(g.id, sub_q)
            level_ms = LEVEL_TO_MS.get(g.engine_level, LEVEL_TO_MS["normal"])
            try:
                # Run think and forward eval events from sub_q until bestmove
                think_task = asyncio.create_task(engine.think(position_fen=g.current_fen, time_ms=level_ms, game_id=g.id))
                while not think_task.done():
                    try:
                        kind, ev = await asyncio.wait_for(sub_q.get(), timeout=0.1)
                        _publish(game_id, {"type": "eval", "depth": ev.depth, "cp": ev.cp,
                                           "pv": ev.pv, "thinkMs": ev.time_ms or 0})
                    except asyncio.TimeoutError:
                        pass
                result = await think_task
                eng_fen_after = g.current_fen
                await apply_move(s, game=g, move_uci=result.move, actor="engine",
                                 fen_after=eng_fen_after, eval_cp=result.final_eval_cp,
                                 eval_depth=result.final_depth, think_time_ms=result.think_time_ms)
                _publish(game_id, {"type": "engine_move", "ply": g.current_ply,
                                   "side": 1 - g.side_to_move, "moveUci": result.move,
                                   "fenAfter": eng_fen_after, "evalCp": result.final_eval_cp,
                                   "evalDepth": result.final_depth, "thinkMs": result.think_time_ms,
                                   "clock": {"whiteMs": g.clock_white_ms, "blackMs": g.clock_black_ms,
                                             "runningSide": g.side_to_move}})
            except EngineError:
                s.add(GameEvent(game_id=g.id, ply_at=g.current_ply, actor="system", type="engine_error"))
                await s.commit()
                _publish(game_id, {"type": "event", "event": {"type": "engine_error", "actor": "system"}})
            finally:
                engine.unsubscribe(g.id, sub_q)
                _publish(game_id, {"type": "engine_thinking", "started": False})
```

Mount: `app.include_router(games_ws.router)` in `create_app()`.

- [ ] **Step 2: Tests using `httpx_ws` or starlette TestClient WS**

`tests/test_ws.py`:
```python
from fastapi.testclient import TestClient
from app.main import create_app


def test_ws_sends_snapshot_on_connect():
    app = create_app()
    with TestClient(app) as c:
        # Create game via REST first
        r = c.post("/api/play/new", json={"side": 0, "engineLevel": "normal", "clock": None, "useBook": False})
        gid = r.json()["game"]["id"]
        with c.websocket_connect(f"/ws/games/{gid}") as ws:
            ws.send_json({"type": "hello"})
            msg = ws.receive_json()
            assert msg["type"] == "snapshot"
            assert msg["game"]["id"] == gid


def test_ws_resume_replays_after_lastSeen():
    # 1. Connect, capture snapshot seq.
    # 2. Disconnect.
    # 3. Reconnect with lastSeenSeq → expect either replay or fresh snapshot, not both.
    pass  # implement when buffer is exercised
```

(WS testing is finicky in pytest-asyncio; using sync `TestClient` works because Starlette wraps the loop.)

- [ ] **Step 3: Commit**

```bash
git add web-v2/backend/app/ws web-v2/backend/app/main.py web-v2/backend/tests/test_ws.py
git commit -m "feat(web-v2): WebSocket endpoint with snapshot + eval stream + resume"
```

---

## Task 13: Games history endpoints

**Files:**
- Create: `web-v2/backend/app/games/__init__.py` (empty)
- Create: `web-v2/backend/app/games/schemas.py`
- Create: `web-v2/backend/app/games/routes.py`
- Modify: `web-v2/backend/app/main.py` (mount router)
- Create: `web-v2/backend/tests/test_games_routes.py`

- [ ] **Step 1: Schemas**

```python
# games/schemas.py
from pydantic import BaseModel


class GameSummary(BaseModel):
    id: int
    mode: str
    opponentLabel: str
    result: str | None
    finalScore: str | None
    side: int
    moveCount: int
    startedAt: str
    finishedAt: str | None
    durationMs: int | None
```

- [ ] **Step 2: Routes**

```python
# games/routes.py
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from app.deps import get_db, get_session
from app.auth.session import CurrentSession
from app.errors import AppError
from app.db.models import Game, Move
from app.play.snapshot import build_snapshot
from app.games.schemas import GameSummary


router = APIRouter(prefix="/api/games", tags=["games"])


@router.get("")
async def list_games(page: int = 1, pageSize: int = 20, status: str | None = None,
                     sess: CurrentSession = Depends(get_session), db: AsyncSession = Depends(get_db)):
    q = select(Game)
    if sess.user:
        q = q.where(Game.user_id == sess.user.id)
    elif sess.anon:
        q = q.where(Game.anon_session_id == sess.anon.id)
    if status:
        q = q.where(Game.status == status)
    total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar()
    rows = (await db.execute(q.order_by(Game.started_at.desc()).limit(pageSize).offset((page - 1) * pageSize))).scalars().all()
    items = []
    for g in rows:
        mc = (await db.execute(select(func.count()).select_from(select(Move).where(Move.game_id == g.id).subquery()))).scalar()
        dur = int((g.finished_at - g.started_at).total_seconds() * 1000) if g.finished_at else None
        items.append(GameSummary(
            id=g.id, mode=g.mode,
            opponentLabel=f"NNUE {g.opponent_ref or ''}".strip(),
            result=g.result, finalScore=g.final_score,
            side=g.side, moveCount=mc or 0,
            startedAt=g.started_at.isoformat(),
            finishedAt=g.finished_at.isoformat() if g.finished_at else None,
            durationMs=dur,
        ))
    return {"items": items, "total": total or 0, "page": page, "pageSize": pageSize}


@router.get("/{game_id}")
async def get_game(game_id: int, sess: CurrentSession = Depends(get_session), db: AsyncSession = Depends(get_db)):
    g = await db.get(Game, game_id)
    if g is None: raise AppError("not_found", 404)
    if (sess.user and g.user_id != sess.user.id) or (sess.anon and g.anon_session_id != sess.anon.id):
        raise AppError("not_owner", 403)
    await db.refresh(g, ["moves", "events"])
    return {"game": build_snapshot(g)}


@router.delete("/{game_id}", status_code=204)
async def delete_game(game_id: int, sess: CurrentSession = Depends(get_session), db: AsyncSession = Depends(get_db)):
    g = await db.get(Game, game_id)
    if g is None: raise AppError("not_found", 404)
    if (sess.user and g.user_id != sess.user.id) or (sess.anon and g.anon_session_id != sess.anon.id):
        raise AppError("not_owner", 403)
    if g.status == "active":
        raise AppError("game_not_active", 400, {"reason": "cannot delete active game"})
    await db.delete(g); await db.commit()
```

- [ ] **Step 3: Tests**

`tests/test_games_routes.py`:
```python
async def test_list_games_paginates(client):
    for _ in range(3):
        await client.post("/api/play/new", json={"side": 0, "engineLevel": "normal", "clock": None, "useBook": False})
    r = await client.get("/api/games?page=1&pageSize=2")
    body = r.json()
    assert body["total"] == 3 and len(body["items"]) == 2
```

- [ ] **Step 4: Commit**

```bash
git add web-v2/backend/app/games web-v2/backend/app/main.py web-v2/backend/tests/test_games_routes.py
git commit -m "feat(web-v2): games history endpoints (list/get/delete)"
```

---

## Task 14: Frontend skeleton — Vite + React + TS + Tailwind 4 + i18n

**Files (frontend):**
- Create: `web-v2/frontend/package.json`
- Create: `web-v2/frontend/vite.config.ts`
- Create: `web-v2/frontend/tsconfig.json`
- Create: `web-v2/frontend/tailwind.config.ts`
- Create: `web-v2/frontend/postcss.config.js`
- Create: `web-v2/frontend/index.html`
- Create: `web-v2/frontend/src/{main.tsx,App.tsx,design/globals.css,design/tokens.ts}`
- Create: `web-v2/frontend/src/i18n/{index.ts,kk/common.json,ru/common.json}`
- Create: `web-v2/frontend/.gitignore`

- [ ] **Step 1: Init**

```bash
cd web-v2 && npm create vite@latest frontend -- --template react-ts
cd frontend && npm install
npm install -D tailwindcss@next @tailwindcss/postcss postcss autoprefixer
npm install react-router@7 @tanstack/react-query zustand i18next react-i18next zod react-hook-form @hookform/resolvers lucide-react clsx tailwind-merge
npm install -D vitest @testing-library/react @testing-library/jest-dom jsdom
```

- [ ] **Step 2: Tailwind 4 setup**

`postcss.config.js`:
```js
export default { plugins: { '@tailwindcss/postcss': {} } }
```

`src/design/globals.css`:
```css
@import "tailwindcss";

@theme {
  --color-bg-base: #0d1419;
  --color-bg-raised: #141c22;
  --color-bg-inset: #080d11;
  --color-bg-border: #1f2a32;
  --color-fg-primary: #ebe4d6;
  --color-fg-secondary: #9ba3a8;
  --color-fg-muted: #5d666c;
  --color-accent-teal: #2ba99c;
  --color-accent-gold: #d4a548;
  --color-state-win: #3fc9b7;
  --color-state-loss: #c04848;
  --color-state-draw: #9ba3a8;
  --color-board-wood: #3a2a1c;
  --color-board-hole: #1a1108;
  --color-board-pebble: #d4c2a0;
  --font-sans: "Manrope", "Inter", system-ui, sans-serif;
  --font-mono: "JetBrains Mono", monospace;
}

html, body, #root { height: 100%; }
body { background: var(--color-bg-base); color: var(--color-fg-primary); }
```

Import in `src/main.tsx`: `import "./design/globals.css"`.

- [ ] **Step 3: i18n**

`src/i18n/kk/common.json`:
```json
{
  "app.title": "Тоғызқұмалақ",
  "lobby.newGame": "Жаңа партия",
  "lobby.engineLevel": "Қозғалтқыш деңгейі",
  "common.easy": "Жеңіл",
  "common.normal": "Орташа",
  "common.hard": "Қиын",
  "errors.engine_unavailable": "Қозғалтқыш қол жетімсіз",
  "engine.thinking": "Ойланып жатыр…",
  "game.resign": "Бас тарту",
  "game.draw": "Тең",
  "game.hint": "Ишара"
}
```
`src/i18n/ru/common.json`: parallel keys in Russian.

`src/i18n/index.ts`:
```ts
import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import kk from "./kk/common.json";
import ru from "./ru/common.json";

i18n.use(initReactI18next).init({
  resources: { kk: { common: kk }, ru: { common: ru } },
  lng: "kk", fallbackLng: "kk",
  defaultNS: "common",
  interpolation: { escapeValue: false },
});

export default i18n;
```

- [ ] **Step 4: Vite config — proxy for /api and /ws**

`vite.config.ts`:
```ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8001',
      '/ws':  { target: 'ws://localhost:8001', ws: true },
    },
  },
  test: { environment: 'jsdom', setupFiles: ['./src/test-setup.ts'] },
})
```

`src/test-setup.ts`:
```ts
import '@testing-library/jest-dom';
```

- [ ] **Step 5: App skeleton**

`src/App.tsx`:
```tsx
import { useTranslation } from "react-i18next";

export default function App() {
  const { t } = useTranslation();
  return (
    <main className="p-8">
      <h1 className="text-3xl font-bold">{t("app.title")}</h1>
      <p className="mt-2 text-fg-secondary">{t("engine.thinking")}</p>
    </main>
  );
}
```

`src/main.tsx`:
```tsx
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./i18n";
import "./design/globals.css";

createRoot(document.getElementById("root")!).render(<StrictMode><App /></StrictMode>);
```

- [ ] **Step 6: Smoke test**

`src/App.test.tsx`:
```tsx
import { render, screen } from "@testing-library/react";
import "./i18n";
import App from "./App";

test("renders title", () => {
  render(<App />);
  expect(screen.getByText("Тоғызқұмалақ")).toBeInTheDocument();
});
```

`package.json` scripts:
```json
"scripts": {
  "dev": "vite",
  "build": "tsc -b && vite build",
  "preview": "vite preview",
  "typecheck": "tsc --noEmit",
  "test": "vitest run"
}
```

Run: `npm run typecheck && npm run test && npm run dev` — open `http://localhost:5173`, see the Kazakh title.

- [ ] **Step 7: Commit**

```bash
git add web-v2/frontend
git commit -m "feat(web-v2): frontend skeleton — Vite + React + TS + Tailwind 4 + i18n"
```

---

## Task 15: API client + TanStack Query + Zustand stores

**Files:**
- Create: `web-v2/frontend/src/api/client.ts`
- Create: `web-v2/frontend/src/api/auth.ts`
- Create: `web-v2/frontend/src/api/play.ts`
- Create: `web-v2/frontend/src/api/games.ts`
- Create: `web-v2/frontend/src/stores/{ui,auth}.ts`
- Modify: `web-v2/frontend/src/main.tsx` — wrap with QueryClientProvider

- [ ] **Step 1: `api/client.ts`**

```ts
export class ApiError extends Error {
  constructor(public code: string, public status: number, public messageKk: string, public messageRu: string, public details?: any) {
    super(code);
  }
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(path, {
    ...init,
    credentials: "include",
    headers: { "Content-Type": "application/json", ...(init.headers || {}) },
  });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const e = body?.error;
    throw new ApiError(e?.code ?? "unknown", res.status, e?.messageKk ?? "Қате", e?.messageRu ?? "Ошибка", e?.details);
  }
  if (res.status === 204) return undefined as T;
  return res.json();
}
```

- [ ] **Step 2: API modules**

```ts
// api/auth.ts
import { api } from "./client";

export type UserOut = { id: number; username: string; displayName?: string | null; locale: string };
export type Me = { kind: "user"; user: UserOut } | { kind: "anon"; anonId: string };

export const authApi = {
  me: () => api<Me>("/api/auth/me"),
  register: (body: { username: string; password: string; locale?: string }) =>
    api<{ user: UserOut; migratedGamesCount: number }>("/api/auth/register", { method: "POST", body: JSON.stringify(body) }),
  login: (body: { username: string; password: string }) =>
    api<{ user: UserOut; migratedGamesCount: number }>("/api/auth/login", { method: "POST", body: JSON.stringify(body) }),
  logout: () => api<void>("/api/auth/logout", { method: "POST" }),
};
```

```ts
// api/play.ts
import { api } from "./client";
export type GameState = { /* mirror backend shape */ id: number; status: string; currentFen: string; sideToMove: 0|1; engineThinking: boolean; /* ...etc */ };

export const playApi = {
  new: (body: { side: 0|1; engineLevel: 'easy'|'normal'|'hard'; clock: { initialMs: number; incrementMs: number } | null; useBook: boolean }) =>
    api<{ game: GameState }>("/api/play/new", { method: "POST", body: JSON.stringify(body) }),
  get: (id: number) => api<{ game: GameState }>(`/api/play/${id}`),
  move: (id: number, moveUci: string) => api<{ game: GameState }>(`/api/play/${id}/move`, { method: "POST", body: JSON.stringify({ moveUci }) }),
  undo: (id: number) => api<{ game: GameState }>(`/api/play/${id}/undo`, { method: "POST" }),
  takeback: (id: number, toPly: number) => api<{ game: GameState }>(`/api/play/${id}/takeback`, { method: "POST", body: JSON.stringify({ toPly }) }),
  resign: (id: number) => api<{ game: GameState }>(`/api/play/${id}/resign`, { method: "POST" }),
  drawOffer: (id: number) => api<{ game: GameState; accepted: boolean }>(`/api/play/${id}/draw_offer`, { method: "POST" }),
  hint: (id: number) => api<{ move: string; evalCp: number | null; depth: number | null; pv: string[] }>(`/api/play/${id}/hint`, { method: "POST" }),
};
```

```ts
// api/games.ts
import { api } from "./client";

export type GameSummary = { id: number; mode: string; opponentLabel: string; result: string | null; finalScore: string | null; side: number; moveCount: number; startedAt: string; finishedAt: string | null; durationMs: number | null };

export const gamesApi = {
  list: (params: { page?: number; pageSize?: number; status?: string } = {}) => {
    const qs = new URLSearchParams(Object.entries(params).filter(([,v]) => v !== undefined) as any).toString();
    return api<{ items: GameSummary[]; total: number; page: number; pageSize: number }>(`/api/games${qs ? `?${qs}` : ""}`);
  },
  get: (id: number) => api<{ game: any }>(`/api/games/${id}`),
  delete: (id: number) => api<void>(`/api/games/${id}`, { method: "DELETE" }),
};
```

- [ ] **Step 3: Zustand stores**

`stores/ui.ts`:
```ts
import { create } from "zustand";

type UI = {
  drawerOpen: boolean;
  toggleDrawer: () => void;
  locale: 'kk' | 'ru';
  setLocale: (l: 'kk' | 'ru') => void;
};

export const useUI = create<UI>((set) => ({
  drawerOpen: false,
  toggleDrawer: () => set((s) => ({ drawerOpen: !s.drawerOpen })),
  locale: (localStorage.getItem("locale") as any) || "kk",
  setLocale: (l) => { localStorage.setItem("locale", l); set({ locale: l }); },
}));
```

`stores/auth.ts`:
```ts
import { create } from "zustand";
import { authApi, Me } from "../api/auth";

type S = { me: Me | null; loading: boolean; refresh: () => Promise<void>; logout: () => Promise<void> };

export const useAuth = create<S>((set) => ({
  me: null, loading: true,
  refresh: async () => { try { const me = await authApi.me(); set({ me, loading: false }); } catch { set({ loading: false }); } },
  logout: async () => { await authApi.logout(); set({ me: null }); },
}));
```

- [ ] **Step 4: Wrap with QueryClientProvider**

`main.tsx`:
```tsx
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import App from "./App";
import "./i18n";
import "./design/globals.css";

const qc = new QueryClient({
  defaultOptions: { queries: { staleTime: 5_000, refetchOnWindowFocus: true } },
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={qc}>
      <App />
    </QueryClientProvider>
  </StrictMode>,
);
```

- [ ] **Step 5: Smoke test for `api/client.ts` using MSW**

```bash
cd web-v2/frontend && npm install -D msw
```

`src/api/client.test.ts`:
```ts
import { describe, it, expect } from "vitest";
import { setupServer } from "msw/node";
import { http, HttpResponse } from "msw";
import { api, ApiError } from "./client";

const server = setupServer(
  http.get("/api/health", () => HttpResponse.json({ status: "ok" })),
  http.get("/api/fail", () => HttpResponse.json({ error: { code: "boom", messageKk: "Қате", messageRu: "Ошибка" } }, { status: 500 })),
);
beforeAll(() => server.listen());
afterAll(() => server.close());

it("returns parsed body on success", async () => {
  expect(await api("/api/health")).toEqual({ status: "ok" });
});

it("throws ApiError with code on error envelope", async () => {
  await expect(api("/api/fail")).rejects.toThrow(ApiError);
});
```

`npm run test` → PASS.

- [ ] **Step 6: Commit**

```bash
git add web-v2/frontend
git commit -m "feat(web-v2): API client + TanStack Query + Zustand stores"
```

---

## Task 16: Router + layout (Header, Drawer, OrnamentBorder)

**Files:**
- Create: `web-v2/frontend/src/components/layout/{Header,Drawer,OrnamentBorder}.tsx`
- Create: `web-v2/frontend/src/components/ui/{Button,Toast}.tsx` (shadcn copies, themed) — or skip and use plain Tailwind for A
- Create: `web-v2/frontend/src/routes/{Lobby,Game,History,Replay,Login,Register,Profile}.tsx` (placeholders)
- Modify: `web-v2/frontend/src/App.tsx` — `createHashRouter`

- [ ] **Step 1: Routes set up**

```tsx
// App.tsx
import { createHashRouter, RouterProvider, Outlet } from "react-router";
import Header from "./components/layout/Header";
import Lobby from "./routes/Lobby";
import Game from "./routes/Game";
import History from "./routes/History";
import Replay from "./routes/Replay";
import Login from "./routes/Login";
import Register from "./routes/Register";
import Profile from "./routes/Profile";

function Layout() {
  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <main className="flex-1"><Outlet /></main>
    </div>
  );
}

const router = createHashRouter([
  { element: <Layout />, children: [
    { path: "/", element: <Lobby /> },
    { path: "/lobby", element: <Lobby /> },
    { path: "/play/:id", element: <Game /> },
    { path: "/history", element: <History /> },
    { path: "/replay/:id", element: <Replay /> },
    { path: "/login", element: <Login /> },
    { path: "/register", element: <Register /> },
    { path: "/profile", element: <Profile /> },
  ]},
]);

export default function App() { return <RouterProvider router={router} />; }
```

- [ ] **Step 2: Header**

```tsx
// components/layout/Header.tsx
import { Link } from "react-router";
import { useTranslation } from "react-i18next";
import { useUI } from "../../stores/ui";
import { useAuth } from "../../stores/auth";

export default function Header() {
  const { t, i18n } = useTranslation();
  const { locale, setLocale } = useUI();
  const me = useAuth((s) => s.me);

  return (
    <header className="sticky top-0 z-10 bg-bg-raised border-b border-bg-border h-14 flex items-center px-4 gap-4">
      <Link to="/" className="font-semibold text-lg">{t("app.title")}</Link>
      <nav className="flex gap-3 text-sm text-fg-secondary">
        <Link to="/lobby">{t("lobby.newGame")}</Link>
        <Link to="/history">{t("history.title")}</Link>
      </nav>
      <div className="ml-auto flex items-center gap-3 text-sm">
        <button onClick={() => { const next = locale === "kk" ? "ru" : "kk"; setLocale(next); i18n.changeLanguage(next); }}>
          {locale.toUpperCase()}
        </button>
        {me?.kind === "user" ? <Link to="/profile">{me.user.username}</Link> : <Link to="/login">{t("auth.login")}</Link>}
      </div>
    </header>
  );
}
```

(Add corresponding strings to `kk/common.json` and `ru/common.json`: `history.title`, `auth.login`, `auth.register`, etc.)

- [ ] **Step 3: Stub routes (replaced by later tasks)**

Each route file is a one-line stub returning the route name. Tasks 17-24 replace these with real implementations.

```tsx
// routes/Lobby.tsx
export default function Lobby() {
  return <div className="p-4">Lobby</div>;
}
```

- [ ] **Step 4: Smoke test**

`src/App.test.tsx` — replace with router-aware test:
```tsx
import { render, screen } from "@testing-library/react";
import App from "./App";
import "./i18n";

test("renders Lobby on /", () => {
  render(<App />);
  expect(screen.getByText(/Lobby/i)).toBeInTheDocument();
});
```

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(web-v2): router + Header + placeholder routes"
```

---

## Task 17: Auth UI — Login / Register / Profile

**Files:**
- Modify: `web-v2/frontend/src/routes/{Login,Register,Profile}.tsx`
- Create: `web-v2/frontend/src/hooks/useAuth.ts`
- Create: `web-v2/frontend/src/routes/Login.test.tsx`

- [ ] **Step 1: `hooks/useAuth.ts`**

```ts
import { useEffect } from "react";
import { useAuth as useAuthStore } from "../stores/auth";

export function useAuthBootstrap() {
  const { me, loading, refresh } = useAuthStore();
  useEffect(() => { if (!me && loading) refresh(); }, [me, loading, refresh]);
  return { me, loading };
}
```

Call `useAuthBootstrap()` once in `App.tsx`.

- [ ] **Step 2: Login form (with react-hook-form + zod)**

```tsx
// routes/Login.tsx
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { useNavigate } from "react-router";
import { authApi } from "../api/auth";
import { useAuth } from "../stores/auth";
import { useTranslation } from "react-i18next";

const schema = z.object({ username: z.string().min(3).max(32), password: z.string().min(6).max(128) });

export default function Login() {
  const { t } = useTranslation();
  const nav = useNavigate();
  const refresh = useAuth((s) => s.refresh);
  const { register, handleSubmit, formState: { errors, isSubmitting }, setError } = useForm({ resolver: zodResolver(schema) });

  return (
    <form
      className="max-w-sm mx-auto p-6 space-y-3"
      onSubmit={handleSubmit(async (v) => {
        try { await authApi.login(v); await refresh(); nav("/lobby"); }
        catch (e: any) { setError("password", { message: e.messageKk ?? "Қате" }); }
      })}
    >
      <h2 className="text-2xl">{t("auth.login")}</h2>
      <input className="w-full rounded bg-bg-inset p-2" placeholder={t("auth.username")} {...register("username")} />
      {errors.username && <p className="text-state-loss text-sm">{errors.username.message}</p>}
      <input type="password" className="w-full rounded bg-bg-inset p-2" placeholder={t("auth.password")} {...register("password")} />
      {errors.password && <p className="text-state-loss text-sm">{errors.password.message}</p>}
      <button type="submit" className="w-full bg-accent-teal text-bg-base rounded p-2" disabled={isSubmitting}>
        {t("auth.login")}
      </button>
    </form>
  );
}
```

`Register.tsx` — same pattern with `authApi.register`. `Profile.tsx` — show username, locale toggle, logout button.

- [ ] **Step 3: Test**

```tsx
// routes/Login.test.tsx
import { render, screen, fireEvent } from "@testing-library/react";
import { setupServer } from "msw/node";
import { http, HttpResponse } from "msw";
import { MemoryRouter } from "react-router";
import Login from "./Login";
import "../i18n";

const server = setupServer(
  http.post("/api/auth/login", () => HttpResponse.json({ user: { id: 1, username: "a", locale: "kk" } })),
);
beforeAll(() => server.listen()); afterAll(() => server.close());

test("submits login and clears error on success", async () => {
  render(<MemoryRouter><Login /></MemoryRouter>);
  const username = screen.getByPlaceholderText(/username|логин/i);
  const password = screen.getByPlaceholderText(/password|пароль/i);
  const submit = screen.getByRole("button", { name: /login|кіру|войти/i });
  fireEvent.change(username, { target: { value: "alice" } });
  fireEvent.change(password, { target: { value: "secret123" } });
  fireEvent.click(submit);
  await screen.findByText(/.+/);
  expect(screen.queryByText(/Қате|Ошибка/i)).toBeNull();
});
```

- [ ] **Step 4: Commit**

```bash
git commit -am "feat(web-v2): Login/Register/Profile screens with react-hook-form + zod"
```

---

## Task 18: Lobby with NewGameForm

**Files:**
- Modify: `web-v2/frontend/src/routes/Lobby.tsx`
- Create: `web-v2/frontend/src/components/lobby/NewGameForm.tsx`

- [ ] **Step 1: `NewGameForm`**

```tsx
import { useForm } from "react-hook-form";
import { useNavigate } from "react-router";
import { playApi } from "../../api/play";
import { useTranslation } from "react-i18next";

type FormVals = { side: 0 | 1; engineLevel: 'easy'|'normal'|'hard'; clockMode: 'none'|'5+0'|'10+5'; useBook: boolean };

const CLOCKS = { 'none': null, '5+0': { initialMs: 5*60_000, incrementMs: 0 }, '10+5': { initialMs: 10*60_000, incrementMs: 5_000 } };

export default function NewGameForm() {
  const { t } = useTranslation();
  const nav = useNavigate();
  const { register, handleSubmit } = useForm<FormVals>({ defaultValues: { side: 0, engineLevel: "normal", clockMode: "none", useBook: false } });

  return (
    <form
      className="bg-bg-raised rounded-xl p-6 space-y-4 max-w-md"
      onSubmit={handleSubmit(async (v) => {
        const r = await playApi.new({ side: v.side, engineLevel: v.engineLevel, clock: CLOCKS[v.clockMode], useBook: v.useBook });
        nav(`/play/${r.game.id}`);
      })}
    >
      <h2 className="text-xl">{t("lobby.newGame")}</h2>

      <label className="block">{t("lobby.engineLevel")}
        <select className="w-full mt-1 bg-bg-inset rounded p-2" {...register("engineLevel")}>
          <option value="easy">{t("common.easy")}</option>
          <option value="normal">{t("common.normal")}</option>
          <option value="hard">{t("common.hard")}</option>
        </select>
      </label>

      <label className="block">{t("lobby.clock")}
        <select className="w-full mt-1 bg-bg-inset rounded p-2" {...register("clockMode")}>
          <option value="none">{t("lobby.noClock")}</option>
          <option value="5+0">5+0</option>
          <option value="10+5">10+5</option>
        </select>
      </label>

      <label className="flex items-center gap-2"><input type="checkbox" {...register("useBook")} /> {t("lobby.useBook")}</label>

      <button type="submit" className="w-full bg-accent-teal text-bg-base rounded p-2">{t("lobby.start")}</button>
    </form>
  );
}
```

- [ ] **Step 2: Lobby route**

```tsx
import NewGameForm from "../components/lobby/NewGameForm";
export default function Lobby() {
  return <div className="p-4 max-w-screen-md mx-auto"><NewGameForm /></div>;
}
```

- [ ] **Step 3: Add i18n strings**

In both `kk/common.json` and `ru/common.json`: `lobby.clock`, `lobby.noClock`, `lobby.useBook`, `lobby.start`.

- [ ] **Step 4: Commit**

```bash
git commit -am "feat(web-v2): Lobby with NewGameForm"
```

---

## Task 19: Board, Hole, Pebbles components

**Files:**
- Create: `web-v2/frontend/src/components/board/{Board,Hole,Pebbles,MoveOverlay}.tsx`
- Create: `web-v2/frontend/src/domain/board.ts`
- Create: `web-v2/frontend/src/components/board/Board.test.tsx`

> Togyzkumalak board: 18 holes (2×9) + 2 touz (one per side). Holes 1-9 at bottom (white), 10-18 at top (black, mirrored). Each hole holds N pebbles, touz holds captured set.

- [ ] **Step 1: `domain/board.ts`**

```ts
export type BoardLayout = {
  bottomRow: Array<{ index: number; pebbles: number; isTouz: boolean }>; // indexes 1..9
  topRow:    Array<{ index: number; pebbles: number; isTouz: boolean }>; // indexes 10..18, displayed reversed
  whiteTouz: number;  // captured pebbles
  blackTouz: number;
};

export function parseFen(fen: string): BoardLayout {
  // FEN format from engine: e.g. "9/9 0 0 w 1" — replace with the actual format the engine emits.
  // Until the protocol is confirmed, assume: "<top:9 ints space-separated>/<bottom>" + " <whiteTouz> <blackTouz> <stm> <ply>"
  const [boardPart, rest] = fen.split(" ", 2);
  const [topStr, botStr] = boardPart.split("/");
  const top = topStr.split(/\s+|/g).filter(Boolean).map(Number);
  const bot = botStr.split(/\s+|/g).filter(Boolean).map(Number);
  const [w, b] = (rest || "0 0").split(" ").map(Number);
  return {
    bottomRow: bot.map((p, i) => ({ index: i + 1, pebbles: p, isTouz: false })),
    topRow:    top.map((p, i) => ({ index: i + 10, pebbles: p, isTouz: false })),
    whiteTouz: w || 0,
    blackTouz: b || 0,
  };
}
```

FEN tokenization here mirrors the format documented in `app/engine/protocol_notes.md` (Task 7 Step 0). Add a unit test that asserts `parseFen` correctly handles the literal start FEN copied from those notes — if the engine emits a different shape, fix `parseFen` in this task before moving on.

- [ ] **Step 2: `Board.tsx`**

```tsx
import { BoardLayout } from "../../domain/board";
import Hole from "./Hole";

type Props = {
  layout: BoardLayout;
  side: 0 | 1;                      // viewer's side
  sideToMove: 0 | 1;
  onMove: (holeIndex: number) => void;
  disabled?: boolean;
  highlightLastMoveTo?: number;
};

export default function Board({ layout, side, sideToMove, onMove, disabled, highlightLastMoveTo }: Props) {
  const myRow = side === 0 ? layout.bottomRow : layout.topRow;
  const oppRow = side === 0 ? layout.topRow : layout.bottomRow;
  const myTouz = side === 0 ? layout.whiteTouz : layout.blackTouz;
  const oppTouz = side === 0 ? layout.blackTouz : layout.whiteTouz;

  return (
    <div className="bg-board-wood rounded-2xl p-4 shadow-xl select-none">
      <div className="grid grid-cols-9 gap-2 rotate-180" data-testid="opp-row">
        {oppRow.map((h) => (
          <div key={h.index} className="rotate-180">
            <Hole pebbles={h.pebbles} interactive={false} highlight={h.index === highlightLastMoveTo} />
          </div>
        ))}
      </div>
      <div className="flex justify-between items-center my-2 text-fg-secondary text-sm">
        <div>♔ {oppTouz}</div>
        <div>{myTouz} ♕</div>
      </div>
      <div className="grid grid-cols-9 gap-2" data-testid="my-row">
        {myRow.map((h) => (
          <Hole
            key={h.index}
            pebbles={h.pebbles}
            interactive={!disabled && sideToMove === side && h.pebbles > 0}
            highlight={h.index === highlightLastMoveTo}
            onClick={() => onMove(h.index)}
          />
        ))}
      </div>
    </div>
  );
}
```

`Hole.tsx`:
```tsx
type Props = { pebbles: number; interactive: boolean; highlight?: boolean; onClick?: () => void };

export default function Hole({ pebbles, interactive, highlight, onClick }: Props) {
  return (
    <button
      type="button"
      disabled={!interactive}
      onClick={onClick}
      className={[
        "aspect-square rounded-full bg-board-hole flex items-center justify-center min-h-11 min-w-11",
        interactive ? "ring-2 ring-accent-teal/40 hover:ring-accent-teal cursor-pointer" : "",
        highlight ? "ring-2 ring-accent-gold" : "",
      ].join(" ")}
    >
      <span className="font-mono text-board-pebble text-lg">{pebbles}</span>
    </button>
  );
}
```

- [ ] **Step 3: Test**

```tsx
import { render, screen } from "@testing-library/react";
import Board from "./Board";

const layout = {
  bottomRow: Array.from({ length: 9 }, (_, i) => ({ index: i+1, pebbles: 9, isTouz: false })),
  topRow:    Array.from({ length: 9 }, (_, i) => ({ index: i+10, pebbles: 9, isTouz: false })),
  whiteTouz: 0, blackTouz: 0,
};

test("renders 18 holes", () => {
  render(<Board layout={layout} side={0} sideToMove={0} onMove={() => {}} />);
  expect(screen.getByTestId("my-row").children).toHaveLength(9);
  expect(screen.getByTestId("opp-row").children).toHaveLength(9);
});

test("hole is interactive only on side to move with pebbles > 0", () => {
  render(<Board layout={{ ...layout, bottomRow: layout.bottomRow.map((h, i) => ({ ...h, pebbles: i === 0 ? 0 : 9 })) }}
                side={0} sideToMove={0} onMove={() => {}} />);
  const holes = screen.getAllByRole("button");
  expect(holes[0]).toBeDisabled();
  expect(holes[1]).not.toBeDisabled();
});
```

- [ ] **Step 4: Commit**

```bash
git commit -am "feat(web-v2): Board / Hole components + FEN parser"
```

---

## Task 20: useGameSocket hook with reconnect + lastSeenSeq

**Files:**
- Create: `web-v2/frontend/src/api/ws.ts`
- Create: `web-v2/frontend/src/hooks/useGameSocket.ts`
- Create: `web-v2/frontend/src/hooks/useGameSocket.test.ts`

- [ ] **Step 1: `api/ws.ts` — typed message classes**

```ts
export type ServerMsg = { seq: number } & (
  | { type: "snapshot"; game: any }
  | { type: "move_applied"; ply: number; side: 0|1; moveUci: string; fenAfter: string; clock: any }
  | { type: "engine_thinking"; started: boolean; sinceMs?: number }
  | { type: "eval"; depth: number; cp: number; pv: string[]; thinkMs: number }
  | { type: "engine_move"; ply: number; side: 0|1; moveUci: string; fenAfter: string; evalCp: number|null; evalDepth: number|null; thinkMs: number; clock: any }
  | { type: "event"; event: { type: string; actor: string; payload?: any } }
  | { type: "clock_tick"; whiteMs: number; blackMs: number; runningSide: 0|1|null }
  | { type: "game_finished"; result: string; resultReason: string; finalScore: string }
  | { type: "pong" }
  | { type: "error"; code: string; messageKk: string; messageRu: string; fatal: boolean }
);
```

- [ ] **Step 2: Hook**

```ts
import { useEffect, useRef, useState } from "react";
import { ServerMsg } from "../api/ws";

type Status = "connecting" | "connected" | "reconnecting" | "down";

export function useGameSocket(gameId: number, onMsg: (m: ServerMsg) => void) {
  const [status, setStatus] = useState<Status>("connecting");
  const wsRef = useRef<WebSocket | null>(null);
  const lastSeqRef = useRef<number | undefined>(undefined);
  const retryRef = useRef(0);

  useEffect(() => {
    let cancelled = false;
    function connect() {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(`${proto}://${location.host}/ws/games/${gameId}`);
      wsRef.current = ws;
      ws.onopen = () => {
        retryRef.current = 0;
        setStatus("connected");
        ws.send(JSON.stringify({ type: "hello", lastSeenSeq: lastSeqRef.current }));
      };
      ws.onmessage = (ev) => {
        const m: ServerMsg = JSON.parse(ev.data);
        if (typeof m.seq === "number") lastSeqRef.current = m.seq;
        onMsg(m);
      };
      ws.onclose = () => {
        if (cancelled) return;
        if (retryRef.current >= 3) { setStatus("down"); return; }
        setStatus("reconnecting");
        const delay = 250 * (2 ** retryRef.current);
        retryRef.current += 1;
        setTimeout(connect, delay);
      };
      ws.onerror = () => ws.close();
    }
    connect();
    const heartbeat = setInterval(() => wsRef.current?.readyState === 1 && wsRef.current.send(JSON.stringify({ type: "ping" })), 30_000);
    return () => { cancelled = true; clearInterval(heartbeat); wsRef.current?.close(); };
  }, [gameId, onMsg]);

  function send(msg: object) { wsRef.current?.send(JSON.stringify(msg)); }
  return { status, send };
}
```

- [ ] **Step 3: Tests using `mock-socket`**

```bash
npm install -D mock-socket
```

```ts
// useGameSocket.test.ts
import { renderHook, act } from "@testing-library/react";
import { Server } from "mock-socket";
import { useGameSocket } from "./useGameSocket";

test("sends hello with lastSeenSeq on connect", async () => {
  const url = "ws://localhost/ws/games/1";
  const server = new Server(url);
  const helloFrames: any[] = [];
  server.on("connection", (s) => s.on("message", (m: any) => helloFrames.push(JSON.parse(m))));
  // monkey-patch location for the hook
  Object.defineProperty(window, "location", { value: { protocol: "ws:", host: "localhost" }, writable: true });
  renderHook(() => useGameSocket(1, () => {}));
  await new Promise((r) => setTimeout(r, 50));
  expect(helloFrames[0].type).toBe("hello");
  server.close();
});
```

- [ ] **Step 4: Commit**

```bash
git commit -am "feat(web-v2): useGameSocket hook with reconnect + lastSeenSeq"
```

---

## Task 21: Game route assembly — Board + EvalBar + Clock + EngineStatus + GameControls + MoveList

**Files:**
- Create: `web-v2/frontend/src/components/play/{EvalBar,Clock,EngineStatus,MoveList,GameControls}.tsx`
- Modify: `web-v2/frontend/src/routes/Game.tsx`
- Create: `web-v2/frontend/src/hooks/useGameQuery.ts`

- [ ] **Step 1: `useGameQuery`**

```ts
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { playApi } from "../api/play";

export function useGameQuery(id: number) {
  const qc = useQueryClient();
  const q = useQuery({
    queryKey: ["game", id],
    queryFn: () => playApi.get(id).then((r) => r.game),
    refetchOnWindowFocus: true,
  });
  function applyServerSnapshot(game: any) { qc.setQueryData(["game", id], game); }
  return { ...q, applyServerSnapshot };
}
```

- [ ] **Step 2: Components (kept lean)**

`EvalBar.tsx`:
```tsx
type Props = { cp: number | null | undefined; perspectiveSide: 0 | 1 };
export default function EvalBar({ cp, perspectiveSide }: Props) {
  const v = cp ?? 0;
  const adj = perspectiveSide === 0 ? v : -v;
  const pct = Math.max(0, Math.min(100, 50 + adj * 0.05));  // tune
  return (
    <div className="h-2 w-full bg-bg-inset rounded">
      <div className="h-full bg-accent-teal rounded-l" style={{ width: `${pct}%` }} />
    </div>
  );
}
```

`Clock.tsx`:
```tsx
export default function Clock({ ms, running }: { ms: number; running: boolean }) {
  const m = Math.floor(ms / 60_000);
  const s = Math.floor((ms % 60_000) / 1000);
  const cls = ms < 30_000 ? "text-state-loss" : "text-fg-primary";
  return <span className={`font-mono text-2xl ${cls} ${running ? "" : "opacity-60"}`}>{m}:{String(s).padStart(2, "0")}</span>;
}
```

`EngineStatus.tsx`:
```tsx
export default function EngineStatus({ thinking, depth, cp }: { thinking: boolean; depth?: number; cp?: number }) {
  if (!thinking) return null;
  return <p className="text-sm text-fg-secondary">d={depth ?? "?"} cp={cp ?? "?"}</p>;
}
```

`MoveList.tsx`:
```tsx
type Move = { ply: number; side: 0|1; moveUci: string; evalCp?: number|null };
export default function MoveList({ moves }: { moves: Move[] }) {
  return (
    <ol className="font-mono text-sm space-y-0.5">
      {moves.map((m) => (
        <li key={m.ply} className="flex gap-3">
          <span className="text-fg-muted w-6">{m.ply}.</span>
          <span>{m.moveUci}</span>
          <span className="text-fg-muted">{m.evalCp != null ? (m.evalCp > 0 ? "+" : "") + (m.evalCp / 100).toFixed(2) : ""}</span>
        </li>
      ))}
    </ol>
  );
}
```

`GameControls.tsx`:
```tsx
type Props = { onResign(): void; onDraw(): void; onUndo(): void; onHint(): void; hintsUsed: number; hintsLimit: number; disabled?: boolean };
export default function GameControls(p: Props) {
  return (
    <div className="flex gap-2">
      <button onClick={p.onResign} disabled={p.disabled} className="bg-state-loss/20 text-state-loss px-3 py-2 rounded">Resign</button>
      <button onClick={p.onDraw}   disabled={p.disabled} className="bg-bg-raised px-3 py-2 rounded">Draw</button>
      <button onClick={p.onUndo}   disabled={p.disabled} className="bg-bg-raised px-3 py-2 rounded">Undo</button>
      <button onClick={p.onHint}   disabled={p.disabled || p.hintsUsed >= p.hintsLimit} className="bg-accent-gold/20 text-accent-gold px-3 py-2 rounded">
        Hint ({p.hintsUsed}/{p.hintsLimit})
      </button>
    </div>
  );
}
```

- [ ] **Step 3: `routes/Game.tsx`**

```tsx
import { useParams } from "react-router";
import { useState } from "react";
import { useGameQuery } from "../hooks/useGameQuery";
import { useGameSocket } from "../hooks/useGameSocket";
import { playApi } from "../api/play";
import Board from "../components/board/Board";
import EvalBar from "../components/play/EvalBar";
import Clock from "../components/play/Clock";
import EngineStatus from "../components/play/EngineStatus";
import MoveList from "../components/play/MoveList";
import GameControls from "../components/play/GameControls";
import { parseFen } from "../domain/board";

export default function Game() {
  const { id } = useParams<{ id: string }>();
  const gid = Number(id);
  const { data: g, isLoading, applyServerSnapshot } = useGameQuery(gid);
  const [liveCp, setLiveCp] = useState<number | undefined>(undefined);
  const [liveDepth, setLiveDepth] = useState<number | undefined>(undefined);

  const { status, send } = useGameSocket(gid, (m) => {
    if (m.type === "snapshot") applyServerSnapshot(m.game);
    if (m.type === "engine_move" || m.type === "move_applied" || m.type === "game_finished") applyServerSnapshot({ ...g, ...{} }); // refetch
    if (m.type === "eval") { setLiveCp(m.cp); setLiveDepth(m.depth); }
  });

  if (isLoading || !g) return <p className="p-4">Loading…</p>;

  const layout = parseFen(g.currentFen);
  const myClockMs = g.side === 0 ? g.clock.whiteMs : g.clock.blackMs;
  const oppClockMs = g.side === 0 ? g.clock.blackMs : g.clock.whiteMs;

  async function onMove(holeIdx: number) {
    const moveUci = String(holeIdx);  // engine UCI format — confirm and adjust
    await playApi.move(gid, moveUci);
    send({ type: "move", moveUci });
  }

  return (
    <div className="p-3 max-w-screen-md mx-auto space-y-3">
      <div className="flex justify-between items-center">
        <EngineStatus thinking={g.engineThinking} depth={liveDepth} cp={liveCp} />
        <Clock ms={oppClockMs} running={g.clock.runningSide !== g.side && g.status === "active"} />
      </div>
      <EvalBar cp={liveCp ?? null} perspectiveSide={g.side} />
      <Board
        layout={layout}
        side={g.side}
        sideToMove={g.sideToMove}
        onMove={onMove}
        disabled={g.status !== "active" || g.engineThinking}
      />
      <div className="flex justify-between items-center">
        <Clock ms={myClockMs} running={g.clock.runningSide === g.side && g.status === "active"} />
        <GameControls
          hintsUsed={g.hintsUsed} hintsLimit={g.hintsLimit}
          disabled={g.status !== "active"}
          onResign={() => playApi.resign(gid).then((r) => applyServerSnapshot(r.game))}
          onDraw={() => playApi.drawOffer(gid).then((r) => applyServerSnapshot(r.game))}
          onUndo={() => playApi.undo(gid).then((r) => applyServerSnapshot(r.game))}
          onHint={() => playApi.hint(gid)}
        />
      </div>
      <details>
        <summary className="text-fg-secondary">Moves</summary>
        <MoveList moves={g.moves} />
      </details>
      {status !== "connected" && <p className="text-sm text-state-warn">WS: {status}</p>}
    </div>
  );
}
```

- [ ] **Step 4: Commit**

```bash
git commit -am "feat(web-v2): Game route — Board + EvalBar + Clock + EngineStatus + Controls"
```

---

## Task 22: End-to-end smoke — start engine, run dev servers, play 3 moves

- [ ] **Step 1: Build engine**

```bash
cd engine && cargo build --release && cd ..
```

- [ ] **Step 2: Run backend**

```bash
cd web-v2/backend && . .venv/bin/activate && uvicorn app.main:app --port 8001 --reload
```

- [ ] **Step 3: Run frontend**

```bash
cd web-v2/frontend && npm run dev
```

- [ ] **Step 4: Manual checklist (record results)**

- [ ] Open `http://localhost:5173`. Lobby renders.
- [ ] Start a new game with engine_level=normal, no clock. Redirects to `/play/:id`.
- [ ] Board shows 18 holes, 9 pebbles each.
- [ ] Click own hole → human move applied, engine reply within ~3 s.
- [ ] EvalBar updates while engine thinks (shows `eval` events).
- [ ] EngineStatus shows depth/cp during think.
- [ ] Reload tab → game restored exactly as left.
- [ ] Click Resign → game becomes finished, controls disabled.
- [ ] Visit `/history` → game appears.

If anything fails, file the issue but DO NOT fix here — open a follow-up task. The point of this milestone is to confirm the loop works end-to-end before polish.

- [ ] **Step 5: Commit (only if any fixes were necessary)**

```bash
# only if tweaks were made
git commit -am "fix(web-v2): smoke-test fixes for end-to-end play"
```

---

## Task 23: History list

**Files:**
- Modify: `web-v2/frontend/src/routes/History.tsx`
- Create: `web-v2/frontend/src/components/history/GameRow.tsx`

- [ ] **Step 1: `GameRow.tsx`**

```tsx
import { Link } from "react-router";
import { GameSummary } from "../../api/games";

const RESULT_CLASS: Record<string, string> = {
  win_white: "bg-state-win/10 text-state-win",
  win_black: "bg-state-win/10 text-state-win",
  draw: "bg-state-draw/10 text-state-draw",
};

export default function GameRow({ g, mySide }: { g: GameSummary; mySide: number }) {
  const tone = g.result ? (g.result.endsWith(mySide === 0 ? "white" : "black") ? "win" : g.result === "draw" ? "draw" : "loss") : "active";
  return (
    <Link to={`/replay/${g.id}`} className={`flex items-center justify-between p-3 rounded ${RESULT_CLASS[g.result ?? "draw"] ?? ""}`}>
      <span>{g.opponentLabel}</span>
      <span className="text-sm">{g.finalScore ?? "—"}</span>
      <span className="text-sm text-fg-muted">{new Date(g.startedAt).toLocaleString()}</span>
    </Link>
  );
}
```

- [ ] **Step 2: `History.tsx`**

```tsx
import { useQuery } from "@tanstack/react-query";
import { gamesApi } from "../api/games";
import GameRow from "../components/history/GameRow";

export default function History() {
  const { data } = useQuery({ queryKey: ["games", { page: 1 }], queryFn: () => gamesApi.list({ page: 1, pageSize: 20 }) });
  if (!data) return <p className="p-4">Loading…</p>;
  return (
    <div className="p-4 max-w-screen-md mx-auto space-y-2">
      {data.items.map((g) => <GameRow key={g.id} g={g} mySide={g.side} />)}
      {data.items.length === 0 && <p className="text-fg-secondary">Әлі ойналмаған</p>}
    </div>
  );
}
```

- [ ] **Step 3: Commit**

```bash
git commit -am "feat(web-v2): History list"
```

---

## Task 24: Replay view (prev/next ply through saved moves)

**Files:**
- Modify: `web-v2/frontend/src/routes/Replay.tsx`

- [ ] **Step 1: Replay**

```tsx
import { useParams } from "react-router";
import { useQuery } from "@tanstack/react-query";
import { gamesApi } from "../api/games";
import { useState } from "react";
import Board from "../components/board/Board";
import { parseFen } from "../domain/board";

export default function Replay() {
  const { id } = useParams<{ id: string }>();
  const { data } = useQuery({ queryKey: ["replay", id], queryFn: () => gamesApi.get(Number(id)).then((r) => r.game) });
  const [ply, setPly] = useState(0);
  if (!data) return <p className="p-4">Loading…</p>;

  const fen = ply === 0 ? data.startFen : data.moves[ply - 1].fenAfter;
  const layout = parseFen(fen);

  return (
    <div className="p-3 max-w-screen-md mx-auto space-y-3">
      <Board layout={layout} side={data.side} sideToMove={ply % 2} onMove={() => {}} disabled />
      <div className="flex justify-between items-center">
        <button onClick={() => setPly((p) => Math.max(0, p - 1))}>‹ Prev</button>
        <span className="font-mono">{ply}/{data.moves.length}</span>
        <button onClick={() => setPly((p) => Math.min(data.moves.length, p + 1))}>Next ›</button>
      </div>
      <p className="text-sm text-fg-secondary">eval: {data.moves[ply - 1]?.evalCp ?? "—"}</p>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git commit -am "feat(web-v2): Replay view with prev/next ply"
```

---

## Task 25: Error envelope wiring + Toaster

**Files:**
- Create: `web-v2/frontend/src/components/ui/Toaster.tsx` + simple toast store (or install `sonner`)
- Modify: `web-v2/frontend/src/api/client.ts` — toast on uncaught
- Modify: `web-v2/frontend/src/App.tsx` — render `<Toaster />`

- [ ] **Step 1: Install sonner**

```bash
cd web-v2/frontend && npm install sonner
```

- [ ] **Step 2: Wire**

In `App.tsx`, change the default export to render `Toaster` next to the router:

```tsx
import { RouterProvider } from "react-router";
import { Toaster } from "sonner";

export default function App() {
  return (
    <>
      <RouterProvider router={router} />
      <Toaster richColors theme="dark" position="top-center" />
    </>
  );
}
```

`api/client.ts` — wrap in a top-level error reporter (optional) or rely on per-mutation `onError`:
```ts
import { toast } from "sonner";
// in api(), on ApiError:
const e = new ApiError(...);
if (e.code !== "auth_required" && e.code !== "validation_failed") {
  toast.error(localStorage.getItem("locale") === "ru" ? e.messageRu : e.messageKk);
}
throw e;
```

- [ ] **Step 3: Commit**

```bash
git commit -am "feat(web-v2): error toaster wired into API client"
```

---

## Task 26: Observability — request_id, JSON logs

**Files:**
- Create: `web-v2/backend/app/middleware.py`
- Modify: `web-v2/backend/app/main.py` — install logging middleware
- Modify: `web-v2/backend/app/errors.py` — include request_id in envelope on 500

- [ ] **Step 1: Logging middleware**

```python
# app/middleware.py
import json, time, uuid, logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

log = logging.getLogger("web_v2")
log.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s"))
log.addHandler(handler)


class RequestLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = rid
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception(json.dumps({"rid": rid, "path": request.url.path, "method": request.method, "status": 500}))
            raise
        latency = int((time.perf_counter() - t0) * 1000)
        sess = getattr(request.state, "session", None)
        owner = (f"u{sess.user.id}" if sess and sess.user else f"a{sess.anon.id[:6]}" if sess and sess.anon else "?")
        log.info(json.dumps({"rid": rid, "path": request.url.path, "method": request.method, "status": response.status_code, "latency_ms": latency, "owner": owner}))
        response.headers["X-Request-ID"] = rid
        return response
```

Add `app.add_middleware(RequestLogMiddleware)` BEFORE `SessionMiddleware`.

- [ ] **Step 2: Commit**

```bash
git commit -am "feat(web-v2): request_id + JSON access logs"
```

---

## Task 27: Ornaments + final design pass + README

**Files:**
- Create: `web-v2/frontend/public/ornaments/{koshkar-muiyz,tumarsha,iretker,bota-koz,triangles}.svg`
- Modify: `web-v2/frontend/src/components/layout/OrnamentBorder.tsx`
- Use ornaments selectively in Header underline, Modal frame, loading spinner.
- Create: `web-v2/README.md` (top-level)

- [ ] **Step 1: SVGs (geometric primitives)**

Create five small SVGs (≤ 1 KB each). Examples:

`koshkar-muiyz.svg` (ram-horn corner accent — two interlocking spirals from straight lines):
```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="2">
  <path d="M2 16 Q8 2 16 16 Q24 30 30 16"/>
</svg>
```

`tumarsha.svg` (eight-pointed star strip):
```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 16" fill="currentColor">
  <polygon points="8,2 10,6 14,8 10,10 8,14 6,10 2,8 6,6"/>
  <polygon points="32,2 34,6 38,8 34,10 32,14 30,10 26,8 30,6"/>
  <polygon points="56,2 58,6 62,8 58,10 56,14 54,10 50,8 54,6"/>
</svg>
```

(Other three follow the same approach: pure geometric shapes, single-color, scaled for token usage.)

- [ ] **Step 2: `OrnamentBorder.tsx`**

```tsx
export default function OrnamentBorder({ className = "" }: { className?: string }) {
  return <div className={`text-accent-gold/40 ${className}`}>
    <img src="/ornaments/tumarsha.svg" alt="" className="w-full h-3 opacity-60" />
  </div>;
}
```

Insert under Header in `Layout`.

- [ ] **Step 3: Top-level README**

`web-v2/README.md`:
```markdown
# web-v2

Sub-project A of the Togyzkumalak platform (see [the spec](../docs/superpowers/specs/2026-05-10-web-v2-foundation-design.md)).

## Run dev

    # 1. Build engine once
    cd ../engine && cargo build --release && cd -

    # 2. Backend
    cd backend && python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"
    alembic upgrade head
    uvicorn app.main:app --port 8001 --reload

    # 3. Frontend (separate shell)
    cd frontend && npm install && npm run dev

Open http://localhost:5173.
```

- [ ] **Step 4: Final commit**

```bash
git commit -am "feat(web-v2): Kazakh ornaments + README — A milestone complete"
```

---

## After Task 27

You now have a working web-v2/ that fulfills sub-project A: anonymous + accounts, solo play vs the engine with persistent state, live eval over WebSocket, history + replay, KK + RU.

Next sub-projects (separate spec + plan each):

- **B (PvP):** matchmaking lobby, real-time PvP, Elo, spectator mode. Postgres migration is a precondition.
- **C (Bot Arena):** engine registry, tournaments, GPU worker pool.
- **D (Hosting & Ops):** domain, HTTPS via Caddy, Docker Compose, backup, monitoring.

Do not start B/C/D until A is in production-shape and at least one user (you) has played 50+ games on it without a bug surfacing.
