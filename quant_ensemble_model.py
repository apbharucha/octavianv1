"""
Quant Ensemble Model
Unified signal model for quant terminal/sim/lab with options-aware augmentation.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from advanced_ml_engine import get_ensemble_engine
import options_engine


@dataclass
class QuantSignal:
    direction: str
    confidence: float
    expected_return: float
    optimal_position_size: float
    driving_factor: str
    factor_scores: Dict[str, float]
    options_overlay: Dict[str, Any]
    # Backward-compatible dynamic fields used across terminal/portal/lab
    probability: float = 0.5
    sharpe_estimate: float = 0.0
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.05
    sub_model_signals: Dict[str, Any] = None
    risk_metrics: Dict[str, Any] = None


class QuantEnsembleModel:
    def __init__(self):
        self.ml = get_ensemble_engine()
        self.opt = options_engine.get_options_engine()

    def _to_df(self, prices) -> pd.DataFrame:
        """Convert various input types to a DataFrame with at least a 'Close' column."""
        _FIELD_HINTS = ("close", "open", "high", "low", "volume", "adj")
        _FIELD_EXACT = {"close", "open", "high", "low", "volume", "adj close", "adj_close"}

        def _pick_field_level(cols) -> list:
            """For a MultiIndex, return the level that looks like OHLCV field names."""
            for lvl in range(cols.nlevels):
                vals = [str(v).lower() for v in cols.get_level_values(lvl)]
                # Prefer an exact field-name match (a ticker like "LOW" or "OPEN"
                # should not win the substring scan)
                if any(v in _FIELD_EXACT for v in vals):
                    return [cols.get_level_values(lvl)[i] for i in range(len(cols))]
            for lvl in range(cols.nlevels):
                vals = [str(v).lower() for v in cols.get_level_values(lvl)]
                if any(any(h in v for h in _FIELD_HINTS) for v in vals):
                    return [cols.get_level_values(lvl)[i] for i in range(len(cols))]
            # Fallback: last level (yfinance orientation is (Ticker, Field))
            return list(cols.get_level_values(cols.nlevels - 1))

        if isinstance(prices, pd.DataFrame):
            # Already a DataFrame — ensure 'Close' column exists
            df = prices.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = _pick_field_level(df.columns)
                if df.columns.duplicated().any():
                    df = df.loc[:, ~df.columns.duplicated(keep='first')]
            if 'Close' in df.columns:
                return df
            elif len(df.columns) == 0:
                # Empty/column-less frame — return empty so predict()'s empty check handles it
                return pd.DataFrame({"Close": pd.Series(dtype=float)})
            else:
                # DataFrame without Close — pick the column whose name looks like Close,
                # otherwise treat the first column as Close
                close_col = next((c for c in df.columns if str(c).lower() in ("close", "adj close", "adj_close")), df.columns[0])
                s = df[close_col].astype(float).dropna()
                return pd.DataFrame({"Close": s})
        elif isinstance(prices, pd.Series):
            return pd.DataFrame({"Close": prices.astype(float).dropna()})
        else:
            s = pd.Series(prices).astype(float).dropna()
            return pd.DataFrame({"Close": s})

    def predict(
        self,
        prices,
        volumes=None,  # compatibility argument
        symbol: str = "UNKNOWN",
        asset_type: str = "STOCK",
        options_context: Optional[Dict[str, Any]] = None
    ) -> QuantSignal:
        try:
            df = self._to_df(prices)
            if df.empty or len(df) < 30:
                return self._neutral_signal()
            
            out = self.ml.analyze_symbol_ensemble(
                data=df,
                symbol=symbol,
                asset_type=asset_type,
                options_context=options_context
            )

            direction = "NEUTRAL"
            if out["decision"] == "BULLISH":
                direction = "BULLISH"
            elif out["decision"] == "BEARISH":
                direction = "BEARISH"

            conf = float(out["confidence"])
            exp = float(out["predicted_return"])
            vol = max(0.05, float(out.get("volatility", 0.25)))

            # dynamic sizing from model confidence + volatility
            size = np.clip((conf * 0.9) / (1.0 + 2.0 * vol), 0.02, 0.30)

            # dynamic probability from alpha
            alpha = float(out.get("alpha_score", 50.0))
            probability = float(np.clip(0.5 + (alpha - 50.0) / 100.0, 0.01, 0.99))

            # dynamic risk controls from volatility and expected return
            stop_loss = float(np.clip(0.01 + vol * 0.35, 0.01, 0.12))
            take_profit = float(np.clip(max(0.02, abs(exp) * 2.2), 0.02, 0.25))
            sharpe_est = float(exp / max(vol, 1e-6))

            factors = out.get("factors", {})
            weights = out.get("weights", {})
            sub_model = {
                k: {
                    "probability": float(np.clip(0.5 + (v - 50.0) / 100.0, 0.01, 0.99)),
                    "weight": float(weights.get(k, 0.0)),
                    "weighted_contribution": float((v / 100.0) * weights.get(k, 0.0)),
                }
                for k, v in factors.items()
            }

            risk_metrics = {
                "volatility": float(vol),
                "var_95": float(1.65 * vol / np.sqrt(252)),
                "max_drawdown": float(min(0.0, -2.5 * vol / np.sqrt(252))),
                "tail_ratio": float(np.clip((take_profit / max(stop_loss, 1e-6)), 0.1, 5.0)),
            }

            return QuantSignal(
                direction=direction,
                confidence=conf,
                expected_return=exp,
                optimal_position_size=float(size),
                driving_factor=str(out.get("driving_factor", "FUNDAMENTAL")),
                factor_scores=factors,
                options_overlay=out.get("options_edge") or {},
                probability=probability,
                sharpe_estimate=sharpe_est,
                stop_loss_pct=stop_loss,
                take_profit_pct=take_profit,
                sub_model_signals=sub_model,
                risk_metrics=risk_metrics,
            )
        except Exception:
            return self._neutral_signal()

    def _neutral_signal(self) -> QuantSignal:
        """Return a safe neutral signal when prediction fails."""
        return QuantSignal(
            direction="NEUTRAL",
            confidence=0.0,
            expected_return=0.0,
            optimal_position_size=0.0,
            driving_factor="FUNDAMENTAL",
            factor_scores={"fundamental": 50.0, "technical": 50.0, "sentiment": 50.0, "options": 50.0},
            options_overlay={},
            probability=0.5,
            sharpe_estimate=0.0,
            stop_loss_pct=0.02,
            take_profit_pct=0.05,
            sub_model_signals={},
            risk_metrics={"volatility": 0.25, "var_95": 0.0, "max_drawdown": 0.0, "tail_ratio": 1.0},
        )


_quant_ensemble = None

def get_quant_ensemble() -> QuantEnsembleModel:
    global _quant_ensemble
    if _quant_ensemble is None:
        _quant_ensemble = QuantEnsembleModel()
    return _quant_ensemble
