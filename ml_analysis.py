"""
ml_analysis.py — Compatibility shim for Octavian

This module provides the MLMarketAnalyzer class and get_analyzer() factory
that several modules import. The heavy ML work is delegated to
advanced_ml_engine.py and quant_ensemble_model.py; this shim keeps
the import surface stable.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy-dependency flags (checked lazily)
# ---------------------------------------------------------------------------
SKLEARN_AVAILABLE = False
XGBOOST_AVAILABLE = False
_ml_checked = False


def ensure_ml_libraries() -> None:
    """Attempt to import optional ML libraries and set availability flags."""
    global SKLEARN_AVAILABLE, XGBOOST_AVAILABLE, _ml_checked
    if _ml_checked:
        return
    try:
        import sklearn  # noqa: F401
        SKLEARN_AVAILABLE = True
    except ImportError:
        logger.warning("scikit-learn not available — ML features degraded.")

    try:
        import xgboost  # noqa: F401
        XGBOOST_AVAILABLE = True
    except ImportError:
        logger.warning("XGBoost not available — ensemble features degraded.")

    _ml_checked = True


# ---------------------------------------------------------------------------
# MLMarketAnalyzer
# ---------------------------------------------------------------------------

class MLMarketAnalyzer:
    """
    Lightweight ML market analyzer.

    Wraps the quant ensemble model when available and falls back to a
    simple heuristic (RSI + SMA crossover) when heavy dependencies are
    missing.
    """

    def __init__(self) -> None:
        self._ensemble = None
        self._trained = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_on_data(self, df: pd.DataFrame) -> None:
        """Train / warm-up the underlying model on price data."""
        try:
            from quant_ensemble_model import get_quant_ensemble
            self._ensemble = get_quant_ensemble()
            self._trained = True
        except Exception as exc:
            logger.debug("Ensemble model unavailable, using heuristic: %s", exc)
            self._trained = False

    def predict(
        self,
        df: pd.DataFrame,
        symbol: str = "",
    ) -> Dict[str, Any]:
        """
        Return a prediction dict compatible with the rest of the codebase.

        Keys: signal, confidence, bullish_prob, signal_factors
        """
        if df is None or df.empty:
            return self._neutral_prediction()

        # Try ensemble first
        if self._ensemble is not None:
            try:
                result = self._ensemble.predict(symbol, df)
                direction = getattr(result, "direction", "NEUTRAL")
                prob = float(getattr(result, "probability", 0.5))
                conf = float(getattr(result, "confidence", 0.5))
                return {
                    "signal": direction,
                    "confidence": conf,
                    "bullish_prob": prob,
                    "signal_factors": [],
                }
            except Exception as exc:
                logger.debug("Ensemble predict failed, falling back: %s", exc)

        # Heuristic fallback
        return self._heuristic_predict(df)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _heuristic_predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Simple RSI + SMA crossover heuristic."""
        try:
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.dropna().astype(float)

            if len(close) < 20:
                return self._neutral_prediction()

            sma20 = close.rolling(20).mean().iloc[-1]
            sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else sma20
            price = close.iloc[-1]

            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
            rs = gain / (loss + 1e-10)
            rsi = float((100 - 100 / (1 + rs)).iloc[-1])

            # Signal
            if price > sma20 and sma20 > sma50 and rsi < 70:
                signal, prob = "BULLISH", min(0.5 + abs(price - sma20) / sma20 * 5, 0.90)
            elif price < sma20 and sma20 < sma50 and rsi > 30:
                signal, prob = "BEARISH", max(0.5 - abs(price - sma20) / sma20 * 5, 0.10)
            else:
                signal, prob = "NEUTRAL", 0.50

            conf = min(abs(prob - 0.5) * 2 + 0.4, 0.85)
            factors = [
                (f"RSI({rsi:.1f})", "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral", ""),
                (f"SMA20 vs SMA50", "Bullish" if sma20 > sma50 else "Bearish", ""),
            ]
            return {
                "signal": signal,
                "confidence": round(conf, 3),
                "bullish_prob": round(prob, 3),
                "signal_factors": factors,
            }
        except Exception as exc:
            logger.debug("Heuristic predict failed: %s", exc)
            return self._neutral_prediction()

    @staticmethod
    def _neutral_prediction() -> Dict[str, Any]:
        return {
            "signal": "NEUTRAL",
            "confidence": 0.5,
            "bullish_prob": 0.5,
            "signal_factors": [],
        }


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_analyzer_instance: Optional[MLMarketAnalyzer] = None


def get_analyzer() -> MLMarketAnalyzer:
    """Return the singleton MLMarketAnalyzer instance."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = MLMarketAnalyzer()
    return _analyzer_instance
