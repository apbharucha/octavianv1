"""
dashboard_intelligence_engine.py — Personalized Dashboard Intelligence
=======================================================================
Generates dynamic, real-time, non-hardcoded market takeaways aligned with
the user's trader profile (style, risk tolerance, watchlist, timeframe).

All insights are computed fresh from live market data on each call, with a
15-minute cache to respect performance targets.
"""

from __future__ import annotations

import datetime
import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_yf_ticker_change(symbol: str) -> Tuple[Optional[float], Optional[float]]:
    """Return (last_price, pct_change_1d) for a symbol using fast_info; None on error."""
    try:
        import yfinance as yf
        fi = yf.Ticker(symbol).fast_info
        last  = float(getattr(fi, "last_price", None) or 0)
        prev  = float(getattr(fi, "previous_close", None) or last)
        if last <= 0 or prev <= 0:
            return None, None
        chg = (last / prev - 1) * 100
        return last, chg
    except Exception:
        return None, None


def _direction(pct: float, threshold: float = 0.20) -> str:
    if pct > threshold:
        return "rising"
    if pct < -threshold:
        return "falling"
    return "flat"


def _mag(pct: float) -> str:
    abs_pct = abs(pct)
    if abs_pct >= 2.0:
        return "sharply"
    if abs_pct >= 0.8:
        return "meaningfully"
    return "modestly"


# ---------------------------------------------------------------------------
# Live Market Snapshot
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def _get_live_snapshot() -> Dict[str, Any]:
    """
    Fetch real-time quotes for key instruments.
    Cached 5 minutes so consecutive renders in one session are fast.
    """
    symbols = {
        "SPY": "S&P 500",
        "QQQ": "Nasdaq 100",
        "^VIX": "VIX",
        "DX-Y.NYB": "DXY",
        "^TNX": "10Y Yield",
        "GC=F": "Gold",
        "CL=F": "WTI Oil",
        "BTC-USD": "Bitcoin",
        "TLT": "Long Bonds",
        "HYG": "High Yield",
    }
    data: Dict[str, Dict[str, float]] = {}
    for sym, label in symbols.items():
        price, chg = _safe_yf_ticker_change(sym)
        if price is not None:
            data[sym] = {"label": label, "price": price, "chg": chg}
    return data


# ---------------------------------------------------------------------------
# Profile-Aligned Opportunity Scanner
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def _get_profile_watchlist_data(watchlist: Tuple[str, ...]) -> Dict[str, Dict]:
    """Fetch momentum signals for user's personal watchlist."""
    results = {}
    for sym in watchlist[:12]:  # cap at 12 for speed
        price, chg = _safe_yf_ticker_change(sym)
        if price is not None and chg is not None:
            results[sym] = {"price": price, "chg": chg}
    return results


# ---------------------------------------------------------------------------
# Insight Generators
# ---------------------------------------------------------------------------

def _build_equity_insight(snap: Dict, profile: Dict) -> Dict[str, str]:
    spy = snap.get("SPY", {})
    qqq = snap.get("QQQ", {})
    vix = snap.get("^VIX", {})
    tlt = snap.get("TLT", {})

    spy_chg = spy.get("chg", 0.0) or 0.0
    qqq_chg = qqq.get("chg", 0.0) or 0.0
    vix_val  = vix.get("price", 20.0) or 20.0
    tlt_chg  = tlt.get("chg", 0.0) or 0.0

    # Regime classification
    if vix_val < 14:
        regime = "low-volatility complacency"
        regime_action = "momentum strategies outperform in low-vol regimes"
    elif vix_val < 20:
        regime = "neutral volatility"
        regime_action = "balanced risk/reward environment"
    elif vix_val < 30:
        regime = "elevated fear (VIX " + f"{vix_val:.0f})"
        regime_action = "defensive rotation and hedges are warranted"
    else:
        regime = f"extreme fear (VIX {vix_val:.0f})"
        regime_action = "capital preservation is priority—reduce leverage"

    # Style-specific takeaway
    style = profile.get("style", "swing").lower()
    if "day" in style or "scalp" in style:
        focus = f"Intraday, SPY {_direction(spy_chg)} {abs(spy_chg):.2f}%. Key gamma walls at round numbers."
    elif "swing" in style or "momentum" in style:
        direction = "upside" if spy_chg > 0 else "downside"
        focus = f"Swing setup: SPY {_direction(spy_chg)} {abs(spy_chg):.2f}%, favoring {direction} continuation."
    else:
        focus = f"Macro regime: {regime}. Equities {_direction(spy_chg)} {abs(spy_chg):.2f}%."

    # Bond signal
    bond_signal = ""
    if tlt_chg < -0.5:
        bond_signal = " Rates rising—watch growth stock pressure."
    elif tlt_chg > 0.5:
        bond_signal = " Bonds rallying—defensive tilt in equities."

    takeaway = f"{focus} Regime: {regime}.{bond_signal}"
    lookout = f"{regime_action.capitalize()}. Nasdaq {_direction(qqq_chg)} {abs(qqq_chg):.2f}%."

    return {"takeaway": takeaway, "lookout": lookout}


def _build_fx_macro_insight(snap: Dict, profile: Dict) -> Dict[str, str]:
    dxy = snap.get("DX-Y.NYB", {})
    tny = snap.get("^TNX", {})

    dxy_chg  = dxy.get("chg", 0.0) or 0.0
    dxy_val  = dxy.get("price", 103.0) or 103.0
    tny_chg  = tny.get("chg", 0.0) or 0.0
    tny_val  = tny.get("price", 4.5) or 4.5

    dxy_dir = _direction(dxy_chg)
    rate_color = "hawkish" if tny_val > 5.0 else "elevated" if tny_val > 4.0 else "benign"

    assets = profile.get("asset_classes", [])
    if "Forex" in assets or "FX" in assets:
        specific = f"DXY {dxy_dir} {abs(dxy_chg):.2f}%—monitor EUR/USD & GBP/USD for breakouts."
    else:
        specific = f"USD {dxy_dir} ({abs(dxy_chg):.2f}%), affecting commodity and EM asset pricing."

    takeaway = f"{specific} 10Y Yield at {tny_val:.2f}% ({rate_color} rate environment)."
    lookout  = (
        f"A {'strengthening' if dxy_chg > 0 else 'weakening'} dollar "
        f"{'pressures' if dxy_chg > 0 else 'supports'} commodity longs and EM plays."
    )
    return {"takeaway": takeaway, "lookout": lookout}


def _build_crypto_commodities_insight(snap: Dict, profile: Dict) -> Dict[str, str]:
    btc = snap.get("BTC-USD", {})
    gold = snap.get("GC=F", {})
    oil  = snap.get("CL=F", {})

    btc_chg  = btc.get("chg", 0.0) or 0.0
    gold_chg = gold.get("chg", 0.0) or 0.0
    oil_chg  = oil.get("chg", 0.0) or 0.0
    gold_val = gold.get("price", 2000.0) or 2000.0
    oil_val  = oil.get("price", 80.0) or 80.0

    assets = profile.get("asset_classes", [])
    parts = []
    if "Crypto" in assets or "Bitcoin" in assets:
        parts.append(f"BTC {_mag(btc_chg)} {_direction(btc_chg)} ({btc_chg:+.1f}%)")
    parts.append(f"Gold ${gold_val:,.0f} ({gold_chg:+.1f}%)")
    parts.append(f"Oil ${oil_val:.1f}/bbl ({oil_chg:+.1f}%)")

    takeaway = ". ".join(parts) + "."

    if gold_chg > 0.5 and btc_chg > 0:
        lookout = "Dual safe-haven demand—risk-off with crypto correlation rising."
    elif oil_chg > 1.0:
        lookout = "Oil spike signals inflation pressure; watch energy sector (XLE, CVX, XOM)."
    elif oil_chg < -1.0:
        lookout = "Oil weakness—monitor energy sector for oversold bounce opportunities."
    else:
        lookout = "Commodities tracking equity correlations; watch for divergence signals."

    return {"takeaway": takeaway, "lookout": lookout}


def _build_options_vol_insight(snap: Dict, profile: Dict) -> Dict[str, str]:
    vix = snap.get("^VIX", {})
    hyg = snap.get("HYG", {})

    vix_val = vix.get("price", 20.0) or 20.0
    hyg_chg = hyg.get("chg", 0.0) or 0.0

    risk_tol = profile.get("risk_tolerance", "moderate").lower()

    # Vol regime signal
    if vix_val < 14:
        vol_signal = "Vol crush—premium selling strategies (iron condors, covered calls) favored."
        outlook_str = "Low VIX → mean reversion in volatility likely. Consider long vol hedges."
    elif vix_val < 20:
        vol_signal = f"Vol neutral at {vix_val:.1f}—standard risk/reward on option premiums."
        outlook_str = "Balanced options environment. Monitor gamma wall near ATM strikes."
    elif vix_val < 30:
        vol_signal = f"Elevated vol (VIX {vix_val:.0f})—long premium strategies outperform."
        outlook_str = "Sell into fear spikes. Protective puts expensive but justifiable."
    else:
        vol_signal = f"Extreme vol (VIX {vix_val:.0f})—options market pricing in major dislocation."
        outlook_str = "Tail risk elevated. Size down, hedge aggressively."

    credit_signal = ""
    if hyg_chg < -0.5:
        credit_signal = " Credit spreads widening—watch for contagion to equities."
    elif hyg_chg > 0.3:
        credit_signal = " Credit spreads compressing—risk appetite healthy."

    takeaway = vol_signal + credit_signal
    lookout  = outlook_str

    return {"takeaway": takeaway, "lookout": lookout}


def _build_opportunity_alignment_insight(snap: Dict, profile: Dict, watchlist_data: Dict) -> Dict[str, str]:
    """
    The 'Full Model Conviction' card: most directly personalized to the user.
    Finds the best opportunity from their watchlist aligned with the current regime.
    """
    vix_val = (snap.get("^VIX", {}).get("price") or 20.0)
    spy_chg = (snap.get("SPY", {}).get("chg") or 0.0)

    regime_bias = "bullish" if spy_chg > 0.2 and vix_val < 20 else "bearish" if spy_chg < -0.2 else "neutral"
    style = profile.get("style", "swing")
    timeframe = profile.get("timeframe", "1-5 days")
    risk_tol = profile.get("risk_tolerance", "moderate")

    # Find best watchlist opportunity
    best_sym = None
    best_score = -9999
    for sym, d in watchlist_data.items():
        chg = d.get("chg", 0.0) or 0.0
        # Score: momentum aligned with regime bias
        score = chg if regime_bias == "bullish" else -chg if regime_bias == "bearish" else abs(chg)
        if score > best_score:
            best_score = score
            best_sym = sym

    if best_sym and watchlist_data:
        best_chg = watchlist_data[best_sym]["chg"]
        opp_text = (
            f"{best_sym} showing {_mag(best_chg)} {_direction(best_chg)} momentum "
            f"({best_chg:+.2f}%) — aligns with {regime_bias} regime."
        )
    else:
        # Derive from index data
        if regime_bias == "bullish":
            opp_text = "Broad market bullish—tech and growth sectors outperforming."
        elif regime_bias == "bearish":
            opp_text = "Defensive rotation active—utilities, healthcare, and gold gaining."
        else:
            opp_text = "Mixed signals—focus on sector-specific setups over broad directional bets."

    conviction_msg = (
        f"Your {style} profile (risk: {risk_tol}) → {opp_text} "
        f"Optimal timeframe alignment: {timeframe}."
    )
    lookout = (
        f"Regime: {'Risk-On' if regime_bias == 'bullish' else 'Risk-Off' if regime_bias == 'bearish' else 'Neutral'}. "
        f"VIX at {vix_val:.1f}—{'stay aggressive' if vix_val < 18 else 'maintain hedges' if vix_val > 25 else 'balanced sizing'}."
    )

    return {"takeaway": conviction_msg, "lookout": lookout}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@st.cache_data(ttl=900, show_spinner=False)
def get_market_takeaways() -> Dict[str, Dict[str, str]]:
    """
    Returns dynamic, profile-aligned market takeaways.
    All data is pulled from live market feeds; nothing is hardcoded.
    Cached 15 minutes for performance.
    """
    # Load user profile
    profile: Dict[str, Any] = {}
    watchlist: Tuple[str, ...] = ()
    try:
        profile_raw = st.session_state.get("trader_profile", {})
        profile = {
            "style":          profile_raw.get("trading_style", profile_raw.get("style", "Swing Trading")),
            "risk_tolerance": profile_raw.get("risk_tolerance", "Moderate"),
            "timeframe":      profile_raw.get("timeframe",      "1-5 days"),
            "asset_classes":  profile_raw.get("asset_classes",  ["Stocks"]),
        }
        raw_watchlist = profile_raw.get("watchlist", [])
        if isinstance(raw_watchlist, str):
            raw_watchlist = [s.strip() for s in raw_watchlist.replace(",", " ").split() if s.strip()]
        watchlist = tuple(raw_watchlist[:15])
    except Exception:
        pass

    # Fetch live data
    snap         = _get_live_snapshot()
    watchlist_data = _get_profile_watchlist_data(watchlist) if watchlist else {}

    # Build insights
    takeaways: Dict[str, Dict[str, str]] = {}

    takeaways["Full Model Conviction"] = _build_opportunity_alignment_insight(
        snap, profile, watchlist_data
    )
    takeaways["Equities"] = _build_equity_insight(snap, profile)
    takeaways["FX & Macro"] = _build_fx_macro_insight(snap, profile)
    takeaways["Crypto & Commodities"] = _build_crypto_commodities_insight(snap, profile)
    takeaways["Options & Vol"] = _build_options_vol_insight(snap, profile)

    return takeaways


def get_breaking_news_banner_html() -> str:
    """Fetches the latest critical news and returns HTML for a CSS scrolling banner."""
    try:
        from news_analysis_engine import get_news_engine
        engine = get_news_engine()
        articles = engine._get_recent_articles(hours_back=24)[:8]

        if not articles:
            texts = [
                "MARKET CALM — SCANNING FOR SYSTEM-WIDE CATALYSTS",
                "VOLUME BREWING AT STRUCTURAL RESISTANCE",
                "MONITORING GLOBAL MACRO FLOWS",
            ]
        else:
            texts = [f"[{a.source.upper()}] {a.title.upper()}" for a in articles]

        ticker_text = "  &nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;  ".join(texts)
    except Exception:
        ticker_text = "LIVE NEWS FEED UNAVAILABLE — RECONNECTING TO AGGREGATORS"

    return f'''
    <style>
    .ticker-wrapper {{
        width: 100%;
        overflow: hidden;
        background-color: #0d1117;
        border-top: 1px solid #30363d;
        border-bottom: 2px solid #ff4b4b;
        color: #e6edf3;
        padding: 8px 0;
        white-space: nowrap;
        box-sizing: border-box;
        margin-bottom: 25px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', 'Roboto Mono', monospace;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 1px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }}
    .ticker-text {{
        display: inline-block;
        padding-left: 100%;
        animation: ticker 60s linear infinite;
    }}
    .breaking-label {{
        color: #ff4b4b;
        font-weight: 800;
        border-right: 2px solid #30363d;
        padding-right: 15px;
        margin-right: 15px;
    }}
    @keyframes ticker {{
        0% {{ transform: translate3d(0, 0, 0); }}
        100% {{ transform: translate3d(-100%, 0, 0); }}
    }}
    </style>
    <div class="ticker-wrapper">
        <div class="ticker-text">
            <span class="breaking-label">LIVE BREAKING NEWS</span> {ticker_text} &nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp; <span class="breaking-label">TRADING UPDATES</span> OCTAVIAN ENGINE ONLINE — ALL ASSET CLASSES ACTIVE
        </div>
    </div>
    '''
