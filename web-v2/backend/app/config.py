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
    # Engine binary served to players. The Mar-18 "baseline" build is markedly
    # stronger than the current build (head-to-head 28-2 = 93% over 30 games in
    # serve mode @250ms/move, 2026-05-31), so we serve it by default. Override with the
    # ENGINE_PATH env var. NOTE: the chosen binary must exist on the deploy host
    # (ship togyzkumalaq-engine-baseline, or set ENGINE_PATH).
    engine_path: Path = REPO_ROOT / "engine" / "target" / "release" / "togyzkumalaq-engine-baseline"
    cors_origins: list[str] = ["http://localhost:5173"]


settings = Settings()
