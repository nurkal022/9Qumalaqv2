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
