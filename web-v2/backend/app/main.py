from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from app.config import settings
from app.auth.anonymous import resolve_session, attach_anon_cookie_if_new
from app.auth import routes as auth_routes
from app.errors import install_handlers


class SessionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request.state.session = await resolve_session(request)
        response = await call_next(request)
        attach_anon_cookie_if_new(request, response)
        return response


def create_app() -> FastAPI:
    app = FastAPI(title="Togyzkumalak web-v2 API", version="0.1.0")
    install_handlers(app)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(SessionMiddleware)

    app.include_router(auth_routes.router)

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
