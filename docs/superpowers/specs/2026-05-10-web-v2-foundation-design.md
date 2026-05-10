# web-v2 Foundation Design (Sub-project A)

**Date:** 2026-05-10
**Author:** brainstorming session (Claude + nurkal022)
**Status:** Draft, awaiting user review
**Replaces:** [`2026-04-29-web-v2-mvp-design.md`](2026-04-29-web-v2-mvp-design.md) (Svelte stack and SQLite-only assumptions are obsolete; the apr-29 file is kept for history but superseded by this document for implementation)

## Position in the larger plan

This is **sub-project A of 4** in the new Togyzkumalak platform:

| # | Sub-project | Status |
|---|---|---|
| **A** | **Foundation + Solo UI** (this doc) | brainstormed, drafting plan next |
| B | PvP Multiplayer (lobby, matchmaking, real-time, Elo) | future spec |
| C | Bot Arena (engine registry, tournaments, dashboard) | future spec |
| D | Hosting & Ops (domain, HTTPS, deploy, backup) | future spec |

A is foundation for all three. Schema and stack choices below are made with B/C/D in mind even where A does not use them yet.

## Goal

Replace the current `web/` (single-file FastAPI + static HTML, fragile, ugly) with a polished mobile-first web app at `web-v2/` for playing Togyzkumalak vs the NNUE engine. The app must:

1. Look like chess.com plus understated Kazakh ornaments — not Material, not Lichess minimalism.
2. Survive page reloads, mid-game device switches, and engine subprocess crashes without losing the active game.
3. Have explicit UI states (loading / engine-thinking / error / restored / finished) so the user always knows what is happening.
4. Support Kazakh + Russian out of the box.
5. Carry a database schema and engine-pool design that B (PvP) and C (Arena) can extend without rewriting the foundation.

## Non-goals

The following are deliberately out of scope for A and belong to B/C/D:

- PvP / matchmaking / real-time human-vs-human (B)
- Elo / glicko / rating (B)
- Spectator mode, chat, friends, presence (B)
- Multi-engine registry, tournaments, brackets, GPU worker pool (C)
- Public domain, HTTPS, auto-deploy, monitoring, backups, rate-limiting at edge, anti-abuse (D)
- PWA / offline mode
- Light theme (dark only in A)
- OAuth, email verification, password recovery, 2FA, refresh tokens

## Constraints

1. The current `web/` keeps running until `web-v2/` is ready. Both coexist locally.
2. The Rust NNUE engine binary at `engine/target/release/togyzkumalaq-engine` is reused unchanged in `serve` mode.
3. All game-rule enforcement (legal moves, captures, end conditions) is delegated to the engine; the frontend never validates rules.
4. SQLite for A. Migration to Postgres is a precondition of B and is a separate task that only changes the SQLAlchemy URL and rebuilds Alembic baseline.

## Stack

| Layer | Choice | Why |
|---|---|---|
| Frontend framework | **React 19** + Vite + TypeScript | user choice; mainstream, plenty of tooling |
| Styling | TailwindCSS 4 | design tokens via theme; no CSS-in-JS overhead |
| Component primitives | **shadcn/ui** (copied into repo, built on Radix) | accessible primitives we can theme heavily for Kazakh accents; no vendor lock |
| Server state | **TanStack Query (React Query) v5** | cache, retry, optimistic updates — solves "state lost on reload" out of the box |
| Client state | **Zustand** | tiny store for UI prefs (locale, drawer); Redux is overkill |
| Routing | **React Router 7** | standard |
| Forms / validation | **react-hook-form + zod** | typed schemas; keep client/server contract aligned by hand for A, codegen optional later |
| WebSocket client | thin custom hook on native `WebSocket` | reconnect + replay logic is small and bespoke |
| i18n | **i18next + react-i18next** | KK + RU at launch, lazy namespaces |
| Backend framework | **FastAPI** + uvicorn | async-native, OpenAPI for free, integrates well with subprocess engine |
| ORM | **SQLAlchemy 2.x async** + Alembic | painless SQLite → Postgres migration |
| DB driver | aiosqlite (A), asyncpg (B onward) | swap via DSN |
| Auth | JWT (HS256) in HTTP-only cookie + anon-session cookie | no localStorage XSS; gust-friendly |
| Password hash | bcrypt 12 rounds | standard |
| Rate limiting | slowapi | lightweight, FastAPI-native |
| Engine integration | persistent subprocess in `serve` mode + `asyncio.Lock` | mirrors current `web/server.py` |
| Tests (BE) | pytest + pytest-asyncio + httpx | async-friendly |
| Tests (FE) | vitest + React Testing Library + MSW | DOM tests + mocked HTTP |
| E2E (optional) | Playwright | one smoke flow |

## High-level architecture

```
Browser (React SPA)
  ├── HTTPS REST → FastAPI:  /api/auth, /api/play, /api/games
  └── WSS → FastAPI:         /ws/games/{id}   (eval stream + state push)

FastAPI app
  ├── routers: auth, play, games, meta
  ├── ws: games_ws.py
  ├── EnginePool (singleton subprocess + asyncio.Lock + per-game subscribers)
  ├── auth middleware: ensure-session (anon cookie) + JWT decoder
  └── SQLAlchemy async → SQLite (WAL)

NNUE engine (Rust, persistent subprocess: `togyzkumalaq-engine serve`)
```

A single engine subprocess serves all sessions. `asyncio.Lock` serializes simultaneous `think` calls. Engine death triggers one auto-restart attempt; if that fails, `/api/health` reports `engine: down` and play endpoints return `503 engine_unavailable` until a manual restart.

## Directory layout

```
web-v2/
├── README.md
├── backend/
│   ├── pyproject.toml
│   ├── alembic.ini
│   ├── alembic/versions/
│   ├── app/
│   │   ├── main.py                # app factory + lifespan
│   │   ├── config.py
│   │   ├── deps.py                # get_db, get_session, get_engine
│   │   ├── db/
│   │   │   ├── base.py
│   │   │   └── models.py
│   │   ├── auth/
│   │   │   ├── routes.py
│   │   │   ├── schemas.py
│   │   │   ├── service.py
│   │   │   ├── jwt.py
│   │   │   └── anonymous.py
│   │   ├── play/
│   │   │   ├── routes.py
│   │   │   ├── schemas.py
│   │   │   ├── service.py         # apply_move, takeback, resign, ...
│   │   │   ├── clock.py           # clock arithmetic
│   │   │   └── snapshot.py        # GameState builder
│   │   ├── games/
│   │   │   ├── routes.py
│   │   │   ├── schemas.py
│   │   │   └── service.py
│   │   ├── engine/
│   │   │   ├── pool.py
│   │   │   └── stream.py          # parse engine info-lines
│   │   ├── ws/
│   │   │   └── games_ws.py
│   │   └── errors.py              # unified error envelope
│   └── tests/
│       ├── conftest.py
│       ├── test_auth.py
│       ├── test_play.py
│       ├── test_engine_pool.py
│       ├── test_ws.py
│       ├── test_games.py
│       └── test_clock.py
└── frontend/
    ├── package.json
    ├── vite.config.ts
    ├── tailwind.config.ts
    ├── postcss.config.js
    ├── tsconfig.json
    ├── index.html
    ├── public/
    │   ├── favicon.svg
    │   └── ornaments/             # SVG accents
    └── src/
        ├── main.tsx
        ├── App.tsx                # router root
        ├── api/
        │   ├── client.ts          # fetch wrapper with credentials:'include'
        │   ├── auth.ts
        │   ├── play.ts
        │   ├── games.ts
        │   └── ws.ts              # GameSocket class
        ├── stores/
        │   ├── ui.ts              # Zustand: drawerOpen, locale
        │   └── auth.ts            # Zustand: current user (hydrated from /me)
        ├── hooks/
        │   ├── useGameSocket.ts
        │   ├── useGameQuery.ts
        │   └── useAuth.ts
        ├── domain/
        │   ├── board.ts           # FEN parse, hole counts
        │   ├── notation.ts
        │   └── ornaments.ts
        ├── i18n/
        │   ├── index.ts
        │   ├── kk/
        │   └── ru/
        ├── design/
        │   ├── tokens.ts
        │   └── globals.css
        ├── routes/
        │   ├── Lobby.tsx
        │   ├── Game.tsx
        │   ├── History.tsx
        │   ├── Replay.tsx
        │   ├── Login.tsx
        │   ├── Register.tsx
        │   └── Profile.tsx
        └── components/
            ├── board/
            │   ├── Board.tsx
            │   ├── Hole.tsx
            │   ├── Pebbles.tsx
            │   └── MoveOverlay.tsx
            ├── play/
            │   ├── EvalBar.tsx
            │   ├── Clock.tsx
            │   ├── EngineStatus.tsx
            │   ├── MoveList.tsx
            │   └── GameControls.tsx
            ├── lobby/
            │   └── NewGameForm.tsx
            ├── history/
            │   └── GameRow.tsx
            ├── ui/                 # shadcn copies, themed
            └── layout/
                ├── Header.tsx
                ├── Drawer.tsx
                └── OrnamentBorder.tsx
```

## Data model (SQLite, WAL mode)

```sql
-- users (optional; anon works without insertion)
CREATE TABLE users (
    id              INTEGER PRIMARY KEY,
    username        TEXT NOT NULL UNIQUE,
    display_name    TEXT,
    password_hash   TEXT NOT NULL,            -- bcrypt 12
    locale          TEXT NOT NULL DEFAULT 'kk',  -- 'kk' | 'ru'
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_login_at   TIMESTAMP
);
CREATE UNIQUE INDEX idx_users_username_lower ON users(LOWER(username));

CREATE TABLE anon_sessions (
    id              TEXT PRIMARY KEY,         -- UUID v4
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_seen_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    locale          TEXT NOT NULL DEFAULT 'kk'
);
CREATE INDEX idx_anon_last_seen ON anon_sessions(last_seen_at);

CREATE TABLE games (
    id                  INTEGER PRIMARY KEY,
    user_id             INTEGER REFERENCES users(id) ON DELETE SET NULL,
    anon_session_id     TEXT REFERENCES anon_sessions(id) ON DELETE SET NULL,

    mode                TEXT NOT NULL,          -- 'solo' (in A); 'pvp'/'arena' added in B/C
    side                INTEGER NOT NULL,       -- 0|1, owner's side
    opponent_kind       TEXT NOT NULL,          -- 'engine'|'human'|'bot'
    opponent_ref        TEXT,                   -- engine_build hash for solo

    -- clocks (modeled even for solo so PvP reuses the same shape)
    clock_initial_ms    INTEGER NOT NULL,       -- 0 = no clock
    clock_increment_ms  INTEGER NOT NULL DEFAULT 0,
    clock_white_ms      INTEGER NOT NULL,
    clock_black_ms      INTEGER NOT NULL,
    last_clock_at       TIMESTAMP,

    -- position
    start_fen           TEXT NOT NULL,
    current_fen         TEXT NOT NULL,
    current_ply         INTEGER NOT NULL DEFAULT 0,
    side_to_move        INTEGER NOT NULL DEFAULT 0,

    status              TEXT NOT NULL,          -- 'active'|'finished'|'aborted'
    result              TEXT,                   -- 'win_white'|'win_black'|'draw'|NULL
    result_reason       TEXT,                   -- 'resign'|'timeout'|'draw_agreement'|'rules_end'|'engine_error'
    final_score         TEXT,                   -- '83-78' for display

    started_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at         TIMESTAMP,

    CHECK (
        (user_id IS NOT NULL AND anon_session_id IS NULL) OR
        (user_id IS NULL  AND anon_session_id IS NOT NULL)
    )
);
CREATE INDEX idx_games_user   ON games(user_id, started_at DESC);
CREATE INDEX idx_games_anon   ON games(anon_session_id, started_at DESC);
CREATE INDEX idx_games_active ON games(status) WHERE status = 'active';

CREATE TABLE moves (
    id              INTEGER PRIMARY KEY,
    game_id         INTEGER NOT NULL REFERENCES games(id) ON DELETE CASCADE,
    ply             INTEGER NOT NULL,           -- 1-based
    side            INTEGER NOT NULL,
    actor           TEXT NOT NULL,              -- 'human'|'engine'|'book'
    move_uci        TEXT NOT NULL,              -- '1-3'
    fen_after       TEXT NOT NULL,
    eval_cp         INTEGER,
    eval_depth      INTEGER,
    pv              TEXT,                       -- JSON array
    think_time_ms   INTEGER,
    clock_after_ms  INTEGER,
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (game_id, ply)
);
CREATE INDEX idx_moves_game ON moves(game_id, ply);

CREATE TABLE game_events (
    id              INTEGER PRIMARY KEY,
    game_id         INTEGER NOT NULL REFERENCES games(id) ON DELETE CASCADE,
    ply_at          INTEGER NOT NULL,
    actor           TEXT NOT NULL,              -- 'human'|'engine'|'system'
    type            TEXT NOT NULL,              -- 'resign'|'draw_offer'|'draw_accept'|'draw_decline'|'takeback'|'hint_used'|'engine_error'|'timeout'
    payload_json    TEXT,
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_events_game ON game_events(game_id, created_at);

-- Engine registry (used trivially in A, real in C)
CREATE TABLE engines (
    id              INTEGER PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    binary_path     TEXT NOT NULL,
    weights_path    TEXT,
    build_hash      TEXT NOT NULL,
    is_active       BOOLEAN NOT NULL DEFAULT 1,
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

### Invariants

1. Exactly one of `user_id` / `anon_session_id` is set (CHECK).
2. `current_fen` is a cache. `start_fen` + `moves` is source of truth. A `verify_game(game_id)` helper replays moves and asserts equality to `current_fen`. Called in tests and exposed as a debug endpoint.
3. `moves` is append-only. `takeback` removes rows and writes a `takeback` event; never UPDATE.
4. Clock remaining at instant `now` is computed: if `side_to_move = 0` then `white_remaining = max(0, clock_white_ms - (now - last_clock_at))`. No timer task ticks the DB; the value is computed when read.
5. Anonymous → user migration on register/login: `UPDATE games SET user_id=:uid, anon_session_id=NULL WHERE anon_session_id=:sid`.

## REST API

All state-changing endpoints return the **full updated `GameState` snapshot**. The client treats the response as source of truth and re-applies it via TanStack Query cache.

### Auth

| Method | Path | Auth | Body | Response |
|---|---|---|---|---|
| POST | `/api/auth/register` | session | `{username, password, locale?}` | `{user, migratedGamesCount}` + JWT cookie |
| POST | `/api/auth/login` | — | `{username, password}` | `{user}` + JWT cookie |
| POST | `/api/auth/logout` | session | — | `204` |
| GET | `/api/auth/me` | optional | — | `{kind:'user', user}` or `{kind:'anon', anonId}` |
| PATCH | `/api/auth/me` | session | `{displayName?, locale?}` | `{user}` |

### Play

| Method | Path | Auth | Body | Response |
|---|---|---|---|---|
| POST | `/api/play/new` | session | `NewGameReq` | `{game}` |
| GET | `/api/play/{id}` | owner | — | `{game}` |
| POST | `/api/play/{id}/move` | owner | `{moveUci}` | `{game}` |
| POST | `/api/play/{id}/undo` | owner | — | `{game}` (rolls back human + engine reply) |
| POST | `/api/play/{id}/takeback` | owner | `{toPly}` | `{game}` |
| POST | `/api/play/{id}/resign` | owner | — | `{game}` |
| POST | `/api/play/{id}/draw_offer` | owner | — | `{game, accepted}` |
| POST | `/api/play/{id}/hint` | owner | `{thinkMs?}` | `{move, evalCp, depth, pv}` (limit 3/game) |

```ts
type NewGameReq = {
  side: 0 | 1
  engineLevel: 'easy' | 'normal' | 'hard'
  clock: { initialMs: number; incrementMs: number } | null
  useBook: boolean
  startFen?: string
}

type GameState = {
  id: number
  mode: 'solo'
  side: 0 | 1
  status: 'active' | 'finished' | 'aborted'
  result?: 'win_white' | 'win_black' | 'draw'
  resultReason?: string
  finalScore?: string
  startFen: string
  currentFen: string
  currentPly: number
  sideToMove: 0 | 1
  clock: {
    initialMs: number
    incrementMs: number
    whiteMs: number
    blackMs: number
    runningSide: 0 | 1 | null
  }
  engineThinking: boolean
  hintsUsed: number
  hintsLimit: number
  moves: Array<{
    ply: number; side: 0 | 1; actor: 'human' | 'engine' | 'book'
    moveUci: string; fenAfter: string
    evalCp?: number; evalDepth?: number; thinkTimeMs?: number; clockAfterMs?: number
  }>
  events: Array<{ plyAt: number; actor: string; type: string; payload?: unknown }>
  startedAt: string; finishedAt?: string
}
```

### Games

| Method | Path | Auth | Query / Body | Response |
|---|---|---|---|---|
| GET | `/api/games` | session | `?page&pageSize&status&result` | `{items: GameSummary[], total, page, pageSize}` |
| GET | `/api/games/{id}` | owner | — | `{game: GameState}` |
| DELETE | `/api/games/{id}` | owner | — | `204` (only finished/aborted) |

### Meta / Health

| Method | Path | Auth | Response |
|---|---|---|---|
| GET | `/api/health` | — | `{status, engine:'up'\|'down', dbLatencyMs}` |
| GET | `/api/meta/engines` | — | `[{name, buildHash, levels:[{key, thinkMs}]}]` |

### Rate limits (slowapi)

- Login 5 / 15min / IP
- Register 10 / hour / IP
- New game 30 / hour / session
- Hint 3 / game (in-memory counter)
- Move 60 / min / session

### Error envelope

```json
{ "error": { "code": "illegal_move", "messageKk": "Заңсыз жүріс", "messageRu": "Недопустимый ход", "details": {} } }
```

| HTTP | code |
|---|---|
| 400 | `validation_failed`, `illegal_move`, `game_not_active` |
| 401 | `auth_required` (rare; middleware auto-creates anon) |
| 403 | `not_owner` |
| 404 | `not_found` |
| 409 | `username_taken`, `concurrent_modification` |
| 429 | `rate_limited` |
| 503 | `engine_unavailable` |
| 500 | `internal` |

## WebSocket protocol

Endpoint: `GET /ws/games/{id}` (Upgrade). Auth via cookies. Owner-only; close codes `4003 not_owner`, `4004 not_found`, `1011 idle`.

### Server → Client

Each message has monotonic `seq: number` (per game):

```ts
type ServerMsg = { seq: number } & (
  | { type: 'snapshot', game: GameState }
  | { type: 'move_applied', ply, side, moveUci, fenAfter, clock }
  | { type: 'engine_thinking', started: boolean, sinceMs?: number }
  | { type: 'eval', depth: number, cp: number, pv: string[], thinkMs: number }
  | { type: 'engine_move', ply, side, moveUci, fenAfter, evalCp, evalDepth, thinkMs, clock }
  | { type: 'event', event: { type, actor, payload? } }
  | { type: 'clock_tick', whiteMs, blackMs, runningSide }
  | { type: 'game_finished', result, resultReason, finalScore }
  | { type: 'pong' }
  | { type: 'error', code, messageKk, messageRu, fatal: boolean }
)
```

### Client → Server

```ts
type ClientMsg =
  | { type: 'hello', lastSeenSeq?: number }
  | { type: 'move', moveUci: string }
  | { type: 'request_snapshot' }
  | { type: 'cancel_thinking' }
  | { type: 'ping' }
```

### Resume mechanic

- Server keeps a per-game in-memory ring buffer of the last 200 server messages with their `seq`.
- Client opens WS, sends `hello{lastSeenSeq?}`. If `lastSeenSeq` is in the buffer the server replays everything after; otherwise it sends a fresh `snapshot`.
- Client never trusts local cache on reconnect — waits for replay or snapshot.

### Backpressure

`asyncio.Queue(maxsize=64)` per subscriber. When the queue is full, intermediate `eval` events are dropped (idempotent — newer wins). State-changing messages (`engine_move`, `event`, `game_finished`) are never dropped; if they would overflow, the connection is closed with `1013 try_again_later`.

### Heartbeat

Client `ping` every 30 s. Server closes with `1011 idle` if no frame for 60 s. Engine `think` continues regardless of WS connectivity; `eval` writes still land in DB on the engine_move.

## Engine pool

```python
class EnginePool:
    def __init__(self, binary: Path, weights: Path | None) -> None: ...
    async def start(self) -> None
    async def stop(self) -> None
    async def think(
        self, *, position_fen: str, time_ms: int, game_id: int
    ) -> EngineResult
    def subscribe(self, game_id: int, q: asyncio.Queue) -> None
    def unsubscribe(self, game_id: int, q: asyncio.Queue) -> None
    @property
    def alive(self) -> bool
    @property
    def build_hash(self) -> str
```

Internals:
- One persistent subprocess in `serve` mode launched on FastAPI startup (lifespan).
- `asyncio.Lock` serializes `think()` so concurrent requests from different games queue.
- Stdout reader task parses `info ...` and `bestmove ...` lines (logic ported from `web/server.py`).
- Per-game subscribers (`dict[int, list[asyncio.Queue]]`) get fan-out of parsed `info` events while the engine thinks for that `game_id`.
- On subprocess death: detected by exit code; one auto-restart attempt with 2 s backoff. Active `think()` raises `EngineError`. Caller (play service) marks the move attempt as failed but does NOT finalize the game; sends `event{type:'engine_error'}` and shows a banner. The user can retry the move.
- `build_hash` is read once at start and stored on each `games.opponent_ref` for replay determinism logging.

## State persistence (the "why this never loses your game" section)

Three layers, all reconstruct fully from the database:

1. **Server DB** — every move, event, and clock checkpoint persists immediately. No in-memory game state is authoritative.
2. **Server WS ring buffer** — last 200 messages per game, in-memory. Survives WS disconnects, lost on process restart.
3. **Client TanStack Query cache** — populated from REST + WS. Treated as cache only; on mount, the Game route always issues `GET /api/play/{id}` and opens WS to seed truth.

Restoration scenarios:

| Scenario | Handling |
|---|---|
| Tab reload | TanStack refetches `/api/play/{id}` and opens WS; gets snapshot. |
| Tab background → foreground | `visibilitychange` triggers `request_snapshot` over WS. |
| Network blip (< 60 s) | WS reconnects, sends `hello{lastSeenSeq}`; missing events replayed. |
| Server restart | Buffer lost; client reconnects, server sends fresh `snapshot`. |
| Switch device | Login (or restore anon cookie if same browser) → game appears in History → click → restored. |
| Engine crash mid-think | Engine error event; game stays `active`; user retries the move; clocks paused while engine restarts. |

## Frontend component tree

Routes: `/`, `/lobby`, `/play/:id`, `/history`, `/replay/:id`, `/login`, `/register`, `/profile`. Hash routing (`react-router-dom` with `createHashRouter`) so deploy is just static files.

```
<App>
  <Header>           # sticky 56px, locale switch, account menu
  <Routes>
    <Lobby>          # NewGameForm + last-3 active-games tiles
    <Game>           # the play screen, see below
    <History>        # paginated list of GameSummary
    <Replay>         # GameView with prev/next ply controls + EvalGraph
    <Login> <Register> <Profile>
  </Routes>
  <Drawer>           # slide-in for history on mobile
  <Toaster>          # shadcn sonner
```

`<Game>` (mobile-first):
```
+--------------------------------+
| EngineStatus  Clock(opp)        |
| EvalBar (horizontal, 6 px)      |
| Board (95% width, square)       |
| Clock(self)  Hints/Resign/Draw  |
| MoveList (collapsed → drawer)   |
+--------------------------------+
```

`<Game>` (≥ 1024 px):
```
+--------------+   +---------+
| EvalBar (V)  |   | MoveList|
| Board 460×460|   |  ply ply|
|              |   |   eval  |
+--------------+   |  ...    |
| GameControls |   +---------+
+--------------+
```

## Design tokens

```ts
// src/design/tokens.ts
export const colors = {
  bg:         { base: '#0d1419', raised: '#141c22', inset: '#080d11', border: '#1f2a32' },
  fg:         { primary: '#ebe4d6', secondary: '#9ba3a8', muted: '#5d666c', inverse: '#0d1419' },
  accent:     { teal: '#2ba99c', tealHover: '#3fc9b7', tealDim: '#1d6e66',
                gold: '#d4a548', goldHover: '#e6bc63', goldDim: '#8d6f30' },
  state:      { win: '#3fc9b7', loss: '#c04848', draw: '#9ba3a8', warn: '#d4a548' },
  board:      { wood: '#3a2a1c', woodLight: '#5a3f28', hole: '#1a1108', holeRim: '#7a5a3a',
                pebble: '#d4c2a0', pebbleShadow: '#000a' },
}
```

Type ramp: Manrope (UI), Inter (fallback), JetBrains Mono (move list, eval). All cover Kazakh-Cyrillic glyphs (`ә ө ұ ү ң қ ғ һ і`).

Ornaments (`public/ornaments/*.svg`) used as accents only — never wallpaper:
- `koshkar-muiyz.svg` — modal corners
- `tumarsha.svg` — divider underline
- `iretker.svg` — animated loading spinner
- `bota-koz.svg` — small dots
- `triangles.svg` — opacity-0.04 background tile

## UI states (canonical)

For each major component an explicit state machine. Components must render distinguishable visuals for every state in their list — never collapse states into "loading".

| Component | States |
|---|---|
| Board | `idle` / `dragging` / `awaiting-server-ack` / `engine-thinking` / `disabled-finished` / `restoring` |
| EvalBar | `unknown` / `streaming` (animated tick) / `final` (number) / `engine-down` (greyed) |
| Clock | `paused` / `running` / `low-time` (< 30 s, color shift) / `flagged` (red) |
| EngineStatus | `idle` / `thinking d=N cp=M` / `restarting` / `down` |
| Hint button | `available (n/3)` / `pending` / `delivered` / `exhausted` |
| GameRow (history) | `loading-skel` / `won` / `lost` / `drawn` / `aborted` |
| WS connection | `connected` / `reconnecting` / `down` (banner with retry) |

## Error handling & observability

- Single FastAPI exception handler maps `EngineError`, validation errors, `IntegrityError`, and the catch-all into the error envelope.
- Frontend `apiClient` interceptor: on `error.code === 'auth_required'` redirects to `/login`; on `engine_unavailable` shows persistent banner; on any other → toast.
- Server logs structured JSON to stdout (uvicorn picks up). Fields: `ts, level, request_id, user_id|anon_id, path, status, latency_ms, error_code?`. No external Sentry in A; stdout → file is sufficient.
- `request_id` is generated in middleware, returned in `X-Request-ID` header, logged in any error toast for support.

## Testing

**Backend (pytest async):**
- `test_auth.py` — register/login/logout, password rounds, anonymous session lifecycle, anon→user migration.
- `test_play.py` — full game scenario; undo and takeback to mid-game; resign; draw_offer accept/reject (mocked engine eval); hint rate limit; engine error → game stays active; clock arithmetic across moves.
- `test_engine_pool.py` — concurrent `think()` serialize; subprocess crash → auto-restart; subscriber fan-out; subscriber leak on early disconnect.
- `test_ws.py` — eval stream sequence; resume after disconnect; backpressure drops eval but not engine_move; foreign-game close 4003.
- `test_games.py` — pagination, foreign-game forbidden, soft-delete only finished.
- `test_clock.py` — pure unit tests for clock formula at boundaries.
- Coverage target ≥ 70 % on `app/`.

**Frontend (vitest + RTL + MSW):**
- `Board` renders 18 holes + 2 touz; pebble layout per count; only own side is interactive while `sideToMove === side`.
- `EvalBar` updates as eval messages arrive (mocked WS).
- `useGameSocket` reconnect logic: 3× backoff, then "down" state; `lastSeenSeq` is sent on reconnect.
- `useGameQuery` cache invalidation after move REST returns.
- `Lobby` `NewGameForm` zod validation messages in Kazakh and Russian.

**E2E (Playwright, optional after the rest is green):**
- One smoke flow: anonymous user opens app, starts a game, plays 3 moves, sees eval bar update, reloads tab, game is restored, finishes by resigning.

## Out of scope (explicit, repeating non-goals)

PvP, ratings, chat, friends, OAuth, password recovery, email verification, refresh tokens, multi-engine registry, tournaments, public deploy, HTTPS, monitoring, backups, light theme, PWA, English locale.

## Open risks & mitigations

| Risk | Mitigation |
|---|---|
| Sourcing licensable Kazakh ornament SVGs | Start with simple geometric primitives (triangles, rhombuses, eight-pointed star); replace with proper assets later. Geometric primitives aren't copyrightable. |
| Engine `serve` protocol drift | Pin `build_hash` per game; one regression test parses a captured `info` line. |
| SQLite under WAL still single-writer | Acceptable for A (1–3 concurrent sessions). Migration to Postgres is a precondition of B and is a one-task migration. |
| Engine bottleneck (single subprocess) | Acceptable for A. Multi-process pool is a C-level concern and ships there. |
| Mobile board touch targets | Holes 44 × 44 px minimum, padding tested on a real phone before leaving Task 13. |
| Time drift between server clock and client | Server is the only authority for clock arithmetic; client never decrements its own state — only renders what the latest `clock_tick` / snapshot says. |
