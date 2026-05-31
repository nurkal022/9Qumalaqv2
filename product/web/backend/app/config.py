from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


REPO_ROOT = Path(__file__).resolve().parents[4]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    env: str = "dev"
    database_url: str = "sqlite+aiosqlite:///./web_v2.db"
    jwt_secret: str = "dev-only-replace-me"
    jwt_algorithm: str = "HS256"
    jwt_ttl_days: int = 7
    anon_cookie_ttl_days: int = 365
    # Engine binary served to players — the blessed champion tracked in models/.
    # It is the Mar "baseline"-era build (rebuilt from commit bb1ced9), markedly
    # stronger than the current build (head-to-head ~88-93% in serve mode @250ms,
    # 2026-05-31). models/engine/baseline is tracked in git, so it ships with the
    # repo. Override with the ENGINE_PATH env var.
    engine_path: Path = REPO_ROOT / "models" / "engine" / "baseline"
    cors_origins: list[str] = ["http://localhost:5173"]


settings = Settings()
