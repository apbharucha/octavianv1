"""
Octavian Backend — Centralized Configuration

All settings are loaded from environment variables with sensible defaults.
For local development, copy `.env.example` to `.env` and fill in your keys.
"""

from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- Application ---
    APP_NAME: str = "Octavian Intelligence Platform"
    APP_VERSION: str = "5.0.0"
    DEBUG: bool = Field(default=False)
    SECRET_KEY: str = Field(default="change-me-in-production-octavian-2026")
    ALLOWED_ORIGINS: list[str] = Field(default=["http://localhost:3000", "http://localhost:8501"])

    # --- Database ---
    DATABASE_URL: str = Field(
        default="sqlite:///./octavian.db",
        description="PostgreSQL in production: postgresql+asyncpg://user:pass@host/db",
    )

    # --- JWT Auth ---
    JWT_SECRET: str = Field(default="octavian-jwt-secret-change-me")
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 30

    # --- Redis (optional — gracefully degrades to in-memory) ---
    REDIS_URL: str | None = Field(default=None)

    # --- Vendor API Keys ---
    POLYGON_API_KEY: str = Field(default="")
    ALPHA_VANTAGE_KEY: str = Field(default="")
    OANDA_API_KEY: str = Field(default="")
    OANDA_BASE_URL: str = Field(default="https://api-fxpractice.oanda.com")

    # --- Stripe ---
    STRIPE_SECRET_KEY: str = Field(default="")
    STRIPE_WEBHOOK_SECRET: str = Field(default="")
    STRIPE_PRICE_ID_FREE: str = Field(default="")
    STRIPE_PRICE_ID_PRO: str = Field(default="")
    STRIPE_PRICE_ID_INSTITUTIONAL: str = Field(default="")

    # --- LLM ---
    LM_STUDIO_URL: str = Field(default="http://localhost:1234/v1")

    # --- Rate Limits (per user per minute) ---
    RATE_LIMIT_FREE: int = 10
    RATE_LIMIT_PRO: int = 60
    RATE_LIMIT_INSTITUTIONAL: int = 200
    RATE_LIMIT_ENTERPRISE: int = 1000

    # --- Paths ---
    PROJECT_ROOT: Path = Field(default_factory=lambda: Path(__file__).parent.parent)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Cached singleton settings instance."""
    return Settings()
