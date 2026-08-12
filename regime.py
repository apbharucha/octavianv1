"""
regime.py — Market Regime Detection Module
==========================================
Provides lightweight regime classification utilities used across the Octavian platform.
Replaces the missing `regime` module that custom_dashboard.py, ai_chatbot.py, and
document_analyzer.py depend on.
"""

from __future__ import annotations

import logging
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core Regime Functions
# ---------------------------------------------------------------------------

def volatility_regime(vix: Optional[float] = None) -> str:
    """
    Classify the current volatility regime based on VIX or live data.

    Returns:
        str: 'LOW', 'NORMAL', 'ELEVATED', or 'EXTREME'
    """
    if vix is None:
        try:
            import yfinance as yf
            tk = yf.Ticker("^VIX")
            fast = tk.fast_info
            vix = float(getattr(fast, "last_price", None) or 0)
        except Exception:
            vix = 0.0

    if vix <= 0:
        return "NORMAL"

    if vix < 14:
        return "LOW"
    elif vix < 20:
        return "NORMAL"
    elif vix < 30:
        return "ELEVATED"
    else:
        return "EXTREME"


def risk_on_off(spy_change: Optional[float] = None, vix: Optional[float] = None) -> str:
    """
    Determine current risk sentiment (Risk-On vs Risk-Off).

    Returns:
        str: 'Risk-On', 'Risk-Off', or 'Neutral'
    """
    try:
        if spy_change is None or vix is None:
            import yfinance as yf

            spy_info = yf.Ticker("SPY").fast_info
            spy_price = float(getattr(spy_info, "last_price", 0) or 0)
            spy_prev  = float(getattr(spy_info, "previous_close", spy_price) or spy_price)
            spy_change = (spy_price / spy_prev - 1) * 100 if spy_prev > 0 else 0.0

            vix_info  = yf.Ticker("^VIX").fast_info
            vix = float(getattr(vix_info, "last_price", 20) or 20)

        # Simple rule: SPY up + VIX low → Risk-On; SPY down + VIX high → Risk-Off
        if spy_change > 0.3 and vix < 20:
            return "Risk-On"
        elif spy_change < -0.3 or vix > 25:
            return "Risk-Off"
        else:
            return "Neutral"
    except Exception:
        return "Neutral"


def classify_regime(returns: pd.Series) -> str:
    """
    Classify market regime from a returns series.

    Returns:
        str: 'Trending Bull', 'Trending Bear', 'High Volatility', or 'Consolidation'
    """
    try:
        if returns is None or len(returns) < 20:
            return "Consolidation"

        rets = returns.dropna()
        ann_return = float(rets.mean() * 252)
        ann_vol    = float(rets.std() * np.sqrt(252))

        if ann_return > 0.10 and ann_vol < 0.25:
            return "Trending Bull"
        elif ann_return < -0.10:
            return "Trending Bear"
        elif ann_vol > 0.30:
            return "High Volatility"
        else:
            return "Consolidation"
    except Exception:
        return "Consolidation"


def get_regime_context() -> dict:
    """
    Return a full regime context dict for use in analysis engines.
    """
    vol_regime = volatility_regime()
    risk_mode  = risk_on_off()
    return {
        "volatility_regime": vol_regime,
        "risk_mode": risk_mode,
        "regime_score": {
            "LOW": 0.8,
            "NORMAL": 0.5,
            "ELEVATED": -0.2,
            "EXTREME": -0.8,
        }.get(vol_regime, 0.0),
    }
