"""
Centralized runtime configuration.

For publishable builds, all secrets are read from environment variables
or Streamlit secrets (if available) and never hardcoded in source.
"""

from __future__ import annotations

import os


def _get_secret(name: str, default: str = "") -> str:
    """Read from environment first, then Streamlit secrets."""
    val = os.getenv(name, "").strip()
    if val:
        return val
    try:
        import streamlit as st  # lazy optional import

        secret_val = st.secrets.get(name, default)
        if secret_val is None:
            return default
        return str(secret_val).strip()
    except Exception:
        return default


# Vendor API keys
POLYGON_API_KEY = _get_secret("POLYGON_API_KEY")
ALPHA_VANTAGE_KEY = _get_secret("ALPHA_VANTAGE_KEY")
OANDA_API_KEY = _get_secret("OANDA_API_KEY")
MASSIVE_API_KEY = _get_secret("MASSIVE_API_KEY")
EODHD_API_KEY = _get_secret("EODHD_API_KEY")

# Flask backend session signing key. MUST be overridden in production via
# the SECRET_KEY environment variable; an ephemeral key is only generated so
# that local development sessions keep working.
import secrets as _secrets
FLASK_SECRET_KEY = os.getenv("SECRET_KEY", "").strip() or _secrets.token_hex(32)

# Restrict CORS origins for the Flask backend. Defaults to the local Streamlit origin.
CORS_ORIGINS = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501").split(",")
    if o.strip()
]

# Optional shared bearer token for the Flask API backend's protected endpoints.
API_BACKEND_KEY = os.getenv("API_BACKEND_KEY", "").strip()

# Non-secret defaults
OANDA_BASE_URL = os.getenv("OANDA_BASE_URL", "https://api-fxpractice.oanda.com")
