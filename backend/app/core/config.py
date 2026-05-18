"""
core/config.py — Centralised, environment-driven configuration.

All tuneable values live here. Override any field via a .env file or
real environment variables — never hard-code secrets in source.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Application ───────────────────────────────────────────────────────────
    PROJECT_NAME: str = "ScamShield"
    VERSION: str = "0.1.0"
    DEBUG: bool = True

    # ── Server ────────────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000"]

    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite+aiosqlite:///./scamshield.db"

    # ── Security ──────────────────────────────────────────────────────────────
    SECRET_KEY: str = "change-me-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # ── AI Model ──────────────────────────────────────────────────────────────
    # Path where trained model artefacts are stored/loaded from.
    MODEL_DIR: str = "app/models/ml"
    MODEL_CONFIDENCE_THRESHOLD: float = 0.75


# Singleton — import `settings` everywhere instead of re-instantiating.
settings = Settings()
