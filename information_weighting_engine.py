"""
Information Weighting Engine
============================
Adaptively weights information sources for bull/bear/neutral signal generation.
Tracks which information types have historically been most predictive and adjusts
weights dynamically based on market regime and asset class.

Architecture:
  - Base weights per category (technical, quant, news, options, macro, volume, momentum)
  - Regime multipliers (trending vs ranging vs high-vol)
  - Asset-class multipliers (stock vs crypto vs futures vs forex)
  - Accuracy tracking: if a signal was right → boost its weight; if wrong → reduce it
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
import json
import os
import time
import logging

logger = logging.getLogger(__name__)

# ── Default base weights (sum to 1.0) ────────────────────────────────────────
_BASE_WEIGHTS: Dict[str, float] = {
    "technical_signals":    0.22,  # SMA, RSI, MACD, Bollinger, breakout patterns
    "quant_model":          0.25,  # Quant ensemble ML model output
    "price_momentum":       0.18,  # Short-term price momentum (5d, 20d returns)
    "news_sentiment":       0.15,  # News headline + article sentiment
    "volume_confirmation":  0.10,  # Volume anomaly / direction confirmation
    "options_flow":         0.06,  # Options implied vol, skew, put/call ratio
    "macro_regime":         0.04,  # VIX, market regime, risk-on/off
}

# ── Regime multipliers — applied to base weights depending on market state ────
# Keys: 'trending_bull', 'trending_bear', 'ranging', 'high_vol', 'low_vol'
_REGIME_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    "trending_bull": {
        "technical_signals":    1.2,
        "quant_model":          1.1,
        "price_momentum":       1.3,
        "news_sentiment":       0.9,
        "volume_confirmation":  1.1,
        "options_flow":         0.8,
        "macro_regime":         0.7,
    },
    "trending_bear": {
        "technical_signals":    1.2,
        "quant_model":          1.1,
        "price_momentum":       1.2,
        "news_sentiment":       1.1,
        "volume_confirmation":  1.1,
        "options_flow":         1.2,
        "macro_regime":         1.0,
    },
    "ranging": {
        "technical_signals":    0.9,
        "quant_model":          1.0,
        "price_momentum":       0.7,
        "news_sentiment":       1.0,
        "volume_confirmation":  1.2,
        "options_flow":         1.0,
        "macro_regime":         1.1,
    },
    "high_vol": {
        "technical_signals":    0.8,
        "quant_model":          0.9,
        "price_momentum":       0.7,
        "news_sentiment":       1.1,
        "volume_confirmation":  1.0,
        "options_flow":         1.4,
        "macro_regime":         1.4,
    },
    "low_vol": {
        "technical_signals":    1.1,
        "quant_model":          1.1,
        "price_momentum":       1.2,
        "news_sentiment":       0.9,
        "volume_confirmation":  0.9,
        "options_flow":         0.9,
        "macro_regime":         0.8,
    },
}

# ── Asset-class multipliers ───────────────────────────────────────────────────
_ASSET_CLASS_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    "stock": {
        "technical_signals":    1.0,
        "quant_model":          1.0,
        "price_momentum":       1.0,
        "news_sentiment":       1.1,
        "volume_confirmation":  1.0,
        "options_flow":         1.1,
        "macro_regime":         0.9,
    },
    "etf": {
        "technical_signals":    1.0,
        "quant_model":          1.0,
        "price_momentum":       1.0,
        "news_sentiment":       0.8,
        "volume_confirmation":  1.1,
        "options_flow":         1.0,
        "macro_regime":         1.2,
    },
    "crypto": {
        "technical_signals":    1.0,
        "quant_model":          1.0,
        "price_momentum":       1.3,
        "news_sentiment":       1.3,
        "volume_confirmation":  1.2,
        "options_flow":         0.6,
        "macro_regime":         0.9,
    },
    "futures": {
        "technical_signals":    1.0,
        "quant_model":          0.9,
        "price_momentum":       1.1,
        "news_sentiment":       0.9,
        "volume_confirmation":  1.1,
        "options_flow":         0.9,
        "macro_regime":         1.5,
    },
    "forex": {
        "technical_signals":    1.1,
        "quant_model":          0.9,
        "price_momentum":       1.0,
        "news_sentiment":       1.0,
        "volume_confirmation":  0.8,
        "options_flow":         0.7,
        "macro_regime":         1.5,
    },
}

# ── Labels to display in UI ───────────────────────────────────────────────────
_WEIGHT_LABELS: Dict[str, str] = {
    "technical_signals":    "Technical Analysis",
    "quant_model":          "Quant ML Model",
    "price_momentum":       "Price Momentum",
    "news_sentiment":       "News & Sentiment",
    "volume_confirmation":  "Volume Analysis",
    "options_flow":         "Options Flow",
    "macro_regime":         "Macro / Market Regime",
}


@dataclass
class SignalAccuracyRecord:
    """Tracks historical accuracy for adaptive weight adjustment."""
    category: str
    correct_calls: int = 0
    total_calls: int = 0
    accuracy_multiplier: float = 1.0  # Starts neutral, adjusts over time

    @property
    def accuracy(self) -> float:
        if self.total_calls == 0:
            return 0.5  # No observations — assume 50%
        # Laplace-style smoothing: prevents extreme 0/1 readings on tiny samples
        # while still being responsive from the very first observation.
        return (self.correct_calls + 0.5) / (self.total_calls + 1.0)

    def record_outcome(self, was_correct: bool):
        """
        Record whether this signal was correct.

        Active from day 1 — the accuracy multiplier adapts immediately with
        every recorded outcome. Early adjustments are dampened (influence grows
        with sample size) so a single lucky call can't skew the weights, but
        there is no cold-start period.
        """
        self.total_calls += 1
        if was_correct:
            self.correct_calls += 1

        # Map accuracy to a target multiplier: 70%+ → ~1.2x, 40%-70% → ~1.0x, <40% → ~0.8x
        acc = self.accuracy
        if acc >= 0.70:
            target = min(1.35, self.accuracy_multiplier * 1.05)
        elif acc >= 0.55:
            target = min(1.20, self.accuracy_multiplier * 1.02)
        elif acc >= 0.40:
            target = max(0.85, self.accuracy_multiplier * 0.99)
        else:
            target = max(0.70, self.accuracy_multiplier * 0.95)

        # Move a fraction of the way toward the target; full speed after ~10 calls
        influence = min(1.0, self.total_calls / 10.0)
        self.accuracy_multiplier = self.accuracy_multiplier + (target - self.accuracy_multiplier) * influence


class InformationWeightingEngine:
    """
    Adaptive information weighting engine.
    
    Dynamically weights different information categories (technical, quant,
    news, options, momentum, volume, macro) based on:
      1. Asset class (stocks behave differently from futures/crypto)
      2. Market regime (trending vs ranging, high vol vs low vol)
      3. Historical accuracy tracking (which signals were actually right)
    
    Usage:
        engine = get_weighting_engine()
        weights = engine.get_weights(asset_class='stock', market_regime='trending_bull')
        # weights: {'technical_signals': 0.24, 'quant_model': 0.26, ...}
    """

    def __init__(self, persistence_path: Optional[str] = None):
        self._accuracy_records: Dict[str, SignalAccuracyRecord] = {
            cat: SignalAccuracyRecord(category=cat)
            for cat in _BASE_WEIGHTS
        }
        self._persistence_path = persistence_path or os.path.join(
            os.path.dirname(__file__), ".weight_accuracy.json"
        )
        self._load_accuracy()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_weights(
        self,
        asset_class: str = "stock",
        market_regime: str = "ranging",
    ) -> Dict[str, float]:
        """
        Return normalized weights dict for the given asset class and market regime.
        
        Args:
            asset_class: 'stock' | 'etf' | 'crypto' | 'futures' | 'forex'
            market_regime: 'trending_bull' | 'trending_bear' | 'ranging' | 'high_vol' | 'low_vol'
        
        Returns:
            Dict mapping category → weight (values sum to 1.0)
        """
        asset_class = asset_class.lower() if asset_class else "stock"
        market_regime = market_regime.lower() if market_regime else "ranging"

        # Normalize unknown asset class
        if asset_class not in _ASSET_CLASS_MULTIPLIERS:
            asset_class = "stock"
        if market_regime not in _REGIME_MULTIPLIERS:
            market_regime = "ranging"

        regime_mults = _REGIME_MULTIPLIERS[market_regime]
        asset_mults = _ASSET_CLASS_MULTIPLIERS[asset_class]

        raw_weights = {}
        for cat, base_w in _BASE_WEIGHTS.items():
            r_mult = regime_mults.get(cat, 1.0)
            a_mult = asset_mults.get(cat, 1.0)
            acc_mult = self._accuracy_records[cat].accuracy_multiplier
            raw_weights[cat] = base_w * r_mult * a_mult * acc_mult

        # Normalize to sum to 1.0
        total = sum(raw_weights.values())
        if total <= 0:
            total = 1.0
        return {cat: round(w / total, 4) for cat, w in raw_weights.items()}

    def get_weights_labeled(
        self,
        asset_class: str = "stock",
        market_regime: str = "ranging",
    ) -> List[Tuple[str, str, float]]:
        """Return list of (category, human_label, weight) tuples, sorted by weight desc."""
        weights = self.get_weights(asset_class, market_regime)
        result = [
            (cat, _WEIGHT_LABELS.get(cat, cat), w)
            for cat, w in weights.items()
        ]
        return sorted(result, key=lambda x: x[2], reverse=True)

    def infer_regime_from_market(
        self,
        vix: Optional[float] = None,
        market_bias: Optional[str] = None,
        price_momentum_20d: Optional[float] = None,
    ) -> str:
        """
        Infer the current market regime from available market data.
        
        Args:
            vix: Current VIX value
            market_bias: 'BULLISH' | 'BEARISH' | 'NEUTRAL' from master strategy engine
            price_momentum_20d: 20-day price change % of the relevant index
        
        Returns:
            regime string compatible with get_weights()
        """
        try:
            if vix is not None and vix > 25:
                return "high_vol"
            if vix is not None and vix < 14:
                return "low_vol"
            if market_bias == "BULLISH" and price_momentum_20d and price_momentum_20d > 3:
                return "trending_bull"
            if market_bias == "BEARISH" and price_momentum_20d and price_momentum_20d < -3:
                return "trending_bear"
        except Exception:
            pass
        return "ranging"

    def record_signal_outcome(
        self,
        category: str,
        signal_direction: str,
        actual_direction: str,
    ):
        """
        Record the accuracy of a signal after the outcome is known.
        
        Args:
            category: One of the weight categories (e.g. 'technical_signals')
            signal_direction: 'BULLISH' | 'BEARISH' | 'NEUTRAL'
            actual_direction: The actual market direction that occurred
        """
        if category not in self._accuracy_records:
            return
        was_correct = signal_direction.upper() == actual_direction.upper()
        self._accuracy_records[category].record_outcome(was_correct)
        self._save_accuracy()

    def get_accuracy_summary(self) -> Dict[str, Dict]:
        """Return accuracy stats for all categories."""
        return {
            cat: {
                "label": _WEIGHT_LABELS.get(cat, cat),
                "accuracy": round(rec.accuracy * 100, 1),
                "total_calls": rec.total_calls,
                "multiplier": round(rec.accuracy_multiplier, 3),
            }
            for cat, rec in self._accuracy_records.items()
        }

    def infer_asset_class(self, symbol: str) -> str:
        """Infer asset class from symbol format."""
        if not symbol:
            return "stock"
        s = symbol.upper()
        if s.endswith("=F") or s in ("ES", "NQ", "CL", "GC", "SI"):
            return "futures"
        if s.endswith("-USD") or s in ("BTC", "ETH", "SOL", "ADA"):
            return "crypto"
        if "=X" in s or "/" in s:
            return "forex"
        if s in ("SPY", "QQQ", "IWM", "VTI", "GLD", "TLT", "XLF", "XLK", "XLE",
                  "ARKK", "IAU", "USO", "UNG", "SLV", "PDBC"):
            return "etf"
        return "stock"

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_accuracy(self):
        """Persist accuracy records to disk."""
        try:
            data = {
                cat: {
                    "correct_calls": rec.correct_calls,
                    "total_calls": rec.total_calls,
                    "accuracy_multiplier": rec.accuracy_multiplier,
                }
                for cat, rec in self._accuracy_records.items()
            }
            with open(self._persistence_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.debug(f"Could not save weight accuracy data: {e}")

    def _load_accuracy(self):
        """Load persisted accuracy records from disk."""
        try:
            if os.path.exists(self._persistence_path):
                with open(self._persistence_path) as f:
                    data = json.load(f)
                for cat, vals in data.items():
                    if cat in self._accuracy_records:
                        self._accuracy_records[cat].correct_calls = vals.get("correct_calls", 0)
                        self._accuracy_records[cat].total_calls = vals.get("total_calls", 0)
                        self._accuracy_records[cat].accuracy_multiplier = vals.get("accuracy_multiplier", 1.0)
        except Exception as e:
            logger.debug(f"Could not load weight accuracy data: {e}")


# ── Singleton ─────────────────────────────────────────────────────────────────
_weighting_engine: Optional[InformationWeightingEngine] = None


def get_weighting_engine() -> InformationWeightingEngine:
    """Return the global singleton InformationWeightingEngine instance."""
    global _weighting_engine
    if _weighting_engine is None:
        _weighting_engine = InformationWeightingEngine()
    return _weighting_engine
