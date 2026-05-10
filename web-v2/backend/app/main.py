from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from app.config import settings
from app.auth.anonymous import resolve_session, attach_anon_cookie_if_new
from app.auth import routes as auth_routes
from app.engine.pool import EnginePool
from app.errors import install_handlers
from app.middleware import RequestLogMiddleware


limiter = Limiter(key_func=get_remote_address)


class SessionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request.state.session = await resolve_session(request)
        response = await call_next(request)
        attach_anon_cookie_if_new(request, response)
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = EnginePool(settings.engine_path)
    try:
        await pool.start()
    except Exception:
        # Engine binary missing or failed to start — pool stays dead;
        # play endpoints should return 503 when engine is not alive.
        pass
    app.state.engine_pool = pool
    yield
    await pool.stop()


def create_app() -> FastAPI:
    app = FastAPI(title="Togyzkumalak web-v2 API", version="0.1.0", lifespan=lifespan)
    install_handlers(app)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(SessionMiddleware)
    app.add_middleware(RequestLogMiddleware)

    app.state.limiter = limiter

    @app.exception_handler(RateLimitExceeded)
    async def _rate(_, __):
        from app.errors import _envelope
        return JSONResponse(status_code=429, content=_envelope("rate_limited"))

    app.include_router(auth_routes.router)

    from app.play import routes as play_routes
    app.include_router(play_routes.router)

    from app.ws import games_ws
    app.include_router(games_ws.router)

    from app.games import routes as games_routes
    app.include_router(games_routes.router)

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
