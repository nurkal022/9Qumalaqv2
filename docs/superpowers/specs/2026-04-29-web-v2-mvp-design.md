# web-v2 MVP Design

**Date:** 2026-04-29
**Author:** brainstorming session (Claude + nurkal022)
**Status:** Approved, awaiting spec review
**Sub-project:** 1 of 3 (MVP). Sub-projects 2 (analytics) and 3 (extensions) get their own specs later.

## Goal

Build a working end-to-end web application for playing Togyzkumalak vs the NNUE engine, with anonymous + optional account auth, persistent game history, real-time engine evaluation streaming over WebSocket, and a polished mobile-first UI in a dark Kazakh-themed design. Lives at `web-v2/` next to the existing `web/`, which keeps running unchanged until the new app is ready.

## Non-goals (deferred to Sub-project 2 or 3)

- Detailed analytical replay (eval graph over full game, best alternatives per position)
- Aggregated user stats (W/L by engine level, accuracy, openings)
- MCTS engine option in lobby
- Move animation (pebbles physically traveling)
- Multiplayer / matchmaking
- Rating system (Elo/Glicko)
- PWA / offline
- i18n runtime switcher (Russian/English) — Kazakh only in MVP
- Light theme — dark only in MVP
- Deployment (LAN/public/HTTPS)
- Email verification, password recovery, OAuth, 2FA, refresh tokens

## Constraints

1. The new app must not interfere with the existing `web/` — both can coexist locally during development.
2. Reuse the existing NNUE engine binary (`engine/target/release/togyzkumalaq-engine` in `serve` mode); do not modify the engine.
3. Local-only execution for MVP (`uvicorn` dev server + Vite dev server). Deploy is out of scope.
4. All game logic (legal moves, captures, end conditions) is delegated to the engine — the frontend never validates rules.

## Stack

| Layer | Choice | Rationale |
|---|---|---|
| Frontend framework | Svelte 5 + Vite | Smallest bundle, no virtual DOM overhead, simplest mental model |
| Frontend styling | TailwindCSS 4 | Design tokens via theme extension, no CSS-in-JS overhead |
| Frontend routing | svelte-spa-router | Hash-based, no SSR concerns |
| Frontend lang | TypeScript | Type-safe API contracts |
| Backend framework | FastAPI | Async-native, OpenAPI docs free, Pydantic validation |
| Backend ASGI server | uvicorn | Standard FastAPI runner |
| ORM | SQLAlchemy 2.x async | Postgres migration path later |
| Database | SQLite (file-based) | Single-process, single-file backup, sufficient for MVP |
| Migrations | Alembic | Standard SQLAlchemy companion |
| Auth | JWT (HS256) in HTTP-only cookie | No localStorage XSS risk |
| Password hash | bcrypt 12 rounds | Standard |
| Rate limiting | slowapi | Lightweight FastAPI integration |
| WebSocket | FastAPI native (`@app.websocket`) | No extra dependency |
| Engine integration | `asyncio.create_subprocess_exec` | Same approach as current `web/server.py` |

## High-level architecture

```
Browser (Svelte SPA)
  ├─ REST → FastAPI: auth, game CRUD, move submission
  └─ WebSocket → FastAPI: live engine eval stream during think

FastAPI app
  ├─ Routers: /api/auth, /api/play, /api/games
  ├─ WebSocket: /ws/engine/{game_id}
  ├─ Engine pool (singleton subprocess + asyncio.Lock)
  ├─ JWT/anon-cookie middleware
  └─ SQLAlchemy async → SQLite

NNUE engine (Rust, persistent subprocess in `serve` mode)
```

Single engine subprocess serves all users; `asyncio.Lock` serializes simultaneous `think` requests. Queue depth > 3 → 503. Acceptable for MVP audience (1–3 concurrent users).

## Directory layout

```
web-v2/
├── README.md
├── backend/
│   ├── pyproject.toml
│   ├── alembic.ini
│   ├── alembic/versions/
│   ├── app/
│   │   ├── main.py            # app factory, lifespan (engine start/stop)
│   │   ├── config.py          # pydantic-settings, env vars
│   │   ├── deps.py            # DI: get_db, get_current_user, get_engine
│   │   ├── db/
│   │   │   ├── base.py        # async engine + session factory
│   │   │   └── models.py      # User, Game, Move
│   │   ├── auth/
│   │   │   ├── routes.py      # /api/auth/{register,login,logout,me}
│   │   │   ├── schemas.py
│   │   │   ├── service.py
│   │   │   ├── jwt.py
│   │   │   └── anonymous.py   # session cookie + games migration
│   │   ├── play/
│   │   │   ├── routes.py      # /api/play/{new,move,undo,takeback,resign,draw_offer,hint}
│   │   │   ├── schemas.py
│   │   │   └── session.py     # in-memory game-state cache (per game_id)
│   │   ├── games/
│   │   │   ├── routes.py      # /api/games (list, get)
│   │   │   ├── schemas.py
│   │   │   └── service.py
│   │   ├── engine/
│   │   │   ├── pool.py        # singleton subprocess + lock + subscribers
│   │   │   └── stream.py      # parse engine info-lines
│   │   └── ws/
│   │       └── engine_ws.py   # /ws/engine/{game_id}
│   └── tests/
│       ├── test_auth.py
│       ├── test_play.py
│       ├── test_engine.py
│       ├── test_ws.py
│       └── test_games.py
└── frontend/
    ├── package.json
    ├── vite.config.ts
    ├── tailwind.config.js
    ├── postcss.config.js
    ├── svelte.config.js
    ├── tsconfig.json
    ├── index.html
    ├── public/
    │   ├── favicon.svg
    │   └── ornaments/         # SVG patterns
    └── src/
        ├── main.ts
        ├── App.svelte         # routing root
        ├── lib/
        │   ├── api/
        │   │   ├── client.ts        # fetch wrapper
        │   │   ├── auth.ts
        │   │   ├── play.ts
        │   │   ├── games.ts
        │   │   └── engine-ws.ts     # WS client + reconnect
        │   ├── stores/
        │   │   ├── user.ts
        │   │   ├── game.ts
        │   │   └── ui.ts
        │   ├── domain/
        │   │   ├── board.ts
        │   │   ├── notation.ts
        │   │   └── ornaments.ts
        │   ├── i18n/
        │   │   └── kk.ts            # Kazakh strings
        │   └── design/tokens.ts
        ├── routes/
        │   ├── Lobby.svelte
        │   ├── Game.svelte
        │   ├── History.svelte
        │   ├── GameView.svelte
        │   ├── Login.svelte
        │   ├── Register.svelte
        │   └── Profile.svelte
        └── components/
            ├── Board/
            │   ├── Board.svelte
            │   ├── Hole.svelte
            │   ├── Pebbles.svelte
            │   └── MoveOverlay.svelte
            ├── EvalBar.svelte
            ├── MoveList.svelte
            ├── EngineStatus.svelte
            ├── LobbyControls.svelte
            ├── GameControls.svelte
            ├── Drawer.svelte
            ├── OrnamentBorder.svelte
            ├── Button.svelte
            ├── Toast.svelte
            └── Modal.svelte
```

## REST API

| Method | Path | Auth | Purpose |
|---|---|---|---|
| POST | `/api/auth/register` | — | Register (username, password). Migrates anonymous games to new user. |
| POST | `/api/auth/login` | — | Sets JWT cookie. |
| POST | `/api/auth/logout` | session | Clears cookie. |
| GET | `/api/auth/me` | optional | Returns user object or `{isAnon: true, anonId}`. |
| POST | `/api/play/new` | session/auth | Body: `{side, engineTimeMs, useBook}` → `{gameId}`. |
| POST | `/api/play/{game_id}/move` | owner | Body: `{move?: string}` (null = ask engine first). Returns updated game state. |
| POST | `/api/play/{game_id}/undo` | owner | Reverts last human move + engine reply. |
| POST | `/api/play/{game_id}/takeback` | owner | Body: `{toPly: int}` revert to ply N. |
| POST | `/api/play/{game_id}/resign` | owner | Finalizes game with `result=loss`. |
| POST | `/api/play/{game_id}/draw_offer` | owner | Engine accepts if `\|eval\| < threshold (e.g. 50 cp)`. |
| POST | `/api/play/{game_id}/hint` | owner | Quick 2 s engine think; returns suggested move + eval. Limited to 3/game. |
| GET | `/api/games?page=N` | auth | Paginated list (anonymous games only via `/api/auth/me` link). |
| GET | `/api/games/{id}` | owner | Game detail with all moves. |

WebSocket: `/ws/engine/{game_id}` (session/auth, owner only). Server pushes during engine think:

```ts
type EvalUpdate = { type: "eval", depth: number, eval: number, pv: string[] }
type EvalDone   = { type: "done", move: string, finalEval: number, finalDepth: number }
type EvalError  = { type: "error", reason: string }
```

## Database schema (SQLite)

```sql
CREATE TABLE users (
    id            INTEGER PRIMARY KEY,
    username      TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_login_at DATETIME
);

CREATE TABLE games (
    id              INTEGER PRIMARY KEY,
    user_id         INTEGER REFERENCES users(id) ON DELETE SET NULL,
    anon_session    TEXT,
    side            INTEGER NOT NULL,
    engine_time_ms  INTEGER NOT NULL,
    use_book        BOOLEAN NOT NULL,
    engine_build    TEXT NOT NULL,
    started_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at     DATETIME,
    result          TEXT,
    final_score     TEXT,
    move_count      INTEGER DEFAULT 0,
    CHECK ((user_id IS NOT NULL) OR (anon_session IS NOT NULL))
);
CREATE INDEX idx_games_user ON games(user_id, started_at DESC);
CREATE INDEX idx_games_anon ON games(anon_session, started_at DESC);

CREATE TABLE moves (
    id           INTEGER PRIMARY KEY,
    game_id      INTEGER NOT NULL REFERENCES games(id) ON DELETE CASCADE,
    ply          INTEGER NOT NULL,
    side         INTEGER NOT NULL,
    move_uci     TEXT NOT NULL,
    actor        TEXT NOT NULL,                    -- 'human' | 'engine' | 'book'
    position_fen TEXT NOT NULL,
    eval_cp      INTEGER,
    depth        INTEGER,
    time_ms      INTEGER,
    created_at   DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (game_id, ply)
);
CREATE INDEX idx_moves_game ON moves(game_id, ply);
```

## Engine integration

A single `EnginePool` instance owns the persistent `togyzkumalaq-engine serve` subprocess. Public API:

```python
class EnginePool:
    async def start(self) -> None
    async def stop(self) -> None
    async def think(self, *, position_fen: str, time_ms: int, game_id: int) -> EngineResult
    def subscribe(self, game_id: int, queue: asyncio.Queue) -> None
    def unsubscribe(self, game_id: int, queue: asyncio.Queue) -> None
```

Internals:

- `asyncio.Lock` serializes concurrent `think` calls.
- `_stream_subscribers: dict[int, list[asyncio.Queue]]` — WS handlers register per game_id; when the engine emits an `info` line during a `think`, the pool fans it out to subscribers of that `game_id`.
- Lifecycle hooks live in FastAPI `lifespan_context`: `start()` on app startup, `stop()` on shutdown.
- If the subprocess dies, `think()` raises `EngineError`; one auto-restart attempt is made before propagating.

## Auth flow

**Anonymous (default):** First request gets a `Set-Cookie: anon_session=UUID; HttpOnly; SameSite=Lax`. All play endpoints accept either JWT or anon session. Games are inserted with `(user_id=NULL, anon_session=UUID)`.

**Register:** `POST /api/auth/register` creates user, issues JWT cookie, runs `UPDATE games SET user_id=:uid, anon_session=NULL WHERE anon_session=:cookie_uuid`, returns `{user, migratedGamesCount}`.

**Login:** Verifies password, issues JWT cookie. Optionally migrates current anon-session games into the user (if any played as anon during the session).

**JWT:** HS256 with `JWT_SECRET` env var. 7-day expiry, no refresh; relogin required after expiry. Stored in `auth_token` cookie (`HttpOnly`, `SameSite=Lax`, `Secure` only if `ENV=prod`).

**CSRF protection:** SameSite=Lax cookies + `Origin` header verification on state-changing requests. No separate CSRF token in MVP.

## Frontend behavior

- **Routing:** hash-based (`svelte-spa-router`). `/` redirects to `/lobby` if no active game, else `/game`.
- **Stores:**
  - `user`: `{id, username, isAnon} | null` — populated from `/api/auth/me` on app load.
  - `game`: `{gameId, position, moveHistory, evalHistory, currentEval, status, side, engineThinking}`.
  - `ui`: `{toasts, modals, drawerOpen, loading}`.
- **WebSocket lifecycle:** opened when `/game` route mounts, closed on unmount or game finalization. Reconnects 3× with exponential backoff (250 ms, 500 ms, 1 s); after that shows a banner.
- **Engine status:** `EngineStatus.svelte` subscribes to `game.evalHistory[last]` to render `думает... d18 +1.4`.

## Design tokens

Colors:

```
bg.base     #0d1419   page background
bg.raised   #141c22   cards, modals
bg.inset    #080d11   inputs, hole interiors
bg.border   #1f2a32   subtle separators

fg.primary    #ebe4d6   body text
fg.secondary  #9ba3a8   captions
fg.muted      #5d666c   placeholders
fg.inverse    #0d1419   text on accents

accent.teal     #2ba99c   primary actions
accent.teal2    #3fc9b7   hover
accent.tealDim  #1d6e66   disabled / borders
accent.gold     #d4a548   secondary, last-move highlight
accent.goldDim  #8d6f30   borders

state.win   #3fc9b7
state.loss  #c04848
state.draw  #9ba3a8
state.warn  #d4a548
```

Fonts: `Manrope` (display + body), `Inter` fallback, `JetBrains Mono` for move list. All chosen for full Kazakh-Cyrillic glyph coverage (`ә ө ұ ү ң қ ғ һ і`).

Ornaments (SVG, `public/ornaments/`):
- `koshkar-muiyz.svg` — modal frame edges
- `tumarsha.svg` — divider underlines
- `iretker.svg` — animated loading spinner
- `bota-koz.svg` — small accent dots
- `triangles.svg` — subtle background tile (opacity 0.04)

Used as accents only — never as wallpaper.

## Mobile-first layout

```
≤ 640 px (mobile primary):
  Header 56 px sticky
  Eval bar — horizontal 6 px above board
  Board — 95 % width, square aspect
  EngineStatus — single line
  GameControls — row of 3 icon buttons
  Move history — bottom-sheet drawer

≥ 1024 px (desktop):
  Centered column ~960 px wide
  Eval bar — vertical 12 px left of board
  Board — 450 px square center
  Move list — right panel
  Controls — bar below board
```

## Error handling

| Class | HTTP status | Frontend response |
|---|---|---|
| Validation | 400 | Toast with localized message |
| Auth missing | 401 | Redirect to `/login` |
| Forbidden (foreign game) | 403 | Toast "Бұл партия сізге тиесілі емес" |
| Not found (game) | 404 | Redirect to `/lobby` |
| Conflict (duplicate username) | 409 | Inline form error |
| Engine error | 503 | Toast + game marked `result='engine_error'` (engine-side stops, game un-finalized) |
| Rate limited | 429 | Toast with retry-after |
| Server error | 500 | Toast "Серверде қате орын алды" |
| WS disconnect | — | 3 retries → "Қозғалтқышпен байланыс үзілді — қайта жүктеңіз" banner |

Global FastAPI exception handlers map `EngineError`, `IntegrityError`, validation errors, and a fallback for unexpected `Exception` (logged + 500).

## Rate limiting

- Login: 5 attempts / 15 min per IP
- Register: 10 / hour per IP
- Hint: 3 / game (not per user — per game-id session state)

## Testing

**Backend (pytest + httpx async):**
- Auth: register/login/logout/me, password verification, anonymous session, games migration on register.
- Play: full game scenario (new → moves → engine reply → resign), undo, takeback to mid-game, draw offer accept/reject (with mocked engine eval), hint rate-limit.
- Engine pool: concurrent `think()` serializes correctly; subprocess restart after crash; timeout handling.
- WebSocket: eval stream sequence (info × N → done); auth (foreign game → 4003 close).
- Games: list pagination, get game with moves, foreign-game forbidden.
- Coverage target: ≥ 70% on `app/`.

**Frontend (vitest + @testing-library/svelte):**
- `Board` renders 18 holes + 2 touz; pebble layout per count; hover-highlight only own side.
- `EvalBar` height proportional to eval, animated transitions.
- `stores/game` reducers (applyMove, applyEvalUpdate, takeback) produce correct state.
- `engine-ws` mock socket: reconnect on disconnect, message parsing.

E2E (Playwright, optional): one smoke test — anon plays 3 moves and observes eval bar updates.

## Open questions / risks

| Risk | Mitigation |
|---|---|
| Sourcing licensable Kazakh ornament SVGs | Start with simple geometric approximations (triangles, rhombuses); replace with proper assets when found. Geometric primitives aren't copyrightable. |
| Engine eval format may change | Pin engine binary to current build hash in DB (`engine_build`); add minimal regression test parsing one canned `info` line. |
| `serve` mode protocol assumptions | Mirror `web/server.py` parsing exactly; reuse `engine.send` / `read until bestmove` patterns. |
| Single-process engine bottleneck under load | Acceptable for MVP audience; mark for revisit in Sub-project 2 (multi-process pool). |
| Mobile board touch targets too small | Set minimum 44 × 44 px tap area for holes; verify on real device early. |

## Out of scope (explicit)

See "Non-goals" at top of document.
