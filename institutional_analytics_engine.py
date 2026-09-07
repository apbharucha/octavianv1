"""
Institutional Analytics Engine
===============================
Research-grade quantitative analytics for the Octavian Simulation Hub.
Provides:
  - Bayesian Network modelling with probability propagation
  - Macro market analysis (interest rates, inflation, GDP, liquidity)
  - Micro market analysis (sector rotation, momentum, breadth)
  - Multi-timeframe regime detection
  - AI-driven market scenario generation
  - Automatic Bayesian network construction from data

All functionality is ADDITIVE — nothing in existing modules is modified.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

try:
    from data_sources import get_stock, get_futures_proxy, get_fx
    HAS_DATA_SOURCES = True
except ImportError:
    HAS_DATA_SOURCES = False


# ────────────────────────────────────────────────────────────────────
# BAYESIAN NETWORK
# ────────────────────────────────────────────────────────────────────

@dataclass
class BayesianNode:
    """A single node in a Bayesian network."""
    name: str
    layer: str                          # 'macro', 'market', 'asset'
    state: float = 0.5                  # probability in [0, 1]
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    cpt: Dict[str, float] = field(default_factory=dict)  # parent→weight
    description: str = ""


class BayesianNetwork:
    """
    Directed acyclic graph of probabilistic nodes with forward
    propagation.  Nodes are grouped into macro / market / asset layers.
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, BayesianNode] = {}
        self._build_default_network()

    # ── Construction ──────────────────────────────────────────────

    def add_node(self, node: BayesianNode) -> None:
        self.nodes[node.name] = node

    def add_edge(self, parent: str, child: str, weight: float = 0.5) -> None:
        if parent in self.nodes and child in self.nodes:
            if parent not in self.nodes[child].parents:
                self.nodes[child].parents.append(parent)
            if child not in self.nodes[parent].children:
                self.nodes[parent].children.append(child)
            self.nodes[child].cpt[parent] = weight

    def _build_default_network(self) -> None:
        """Construct the three-layer institutional network."""
        # ── Macro layer ──
        macro_nodes = [
            ("Interest Rates",   "Central bank policy rate level"),
            ("Inflation",        "Consumer & producer price trends"),
            ("GDP Growth",       "Real economic growth trajectory"),
            ("Liquidity",        "Central bank balance sheets / M2"),
            ("Monetary Policy",  "Hawkish/dovish policy stance"),
        ]
        for name, desc in macro_nodes:
            self.add_node(BayesianNode(name=name, layer="macro", state=0.5, description=desc))

        # ── Market layer ──
        market_nodes = [
            ("Volatility",       "Implied / realised vol regime"),
            ("Risk Sentiment",   "Risk-on / risk-off market mode"),
            ("Credit Spreads",   "IG/HY credit spread levels"),
            ("Market Liquidity", "Bid-ask depth & volume"),
        ]
        for name, desc in market_nodes:
            self.add_node(BayesianNode(name=name, layer="market", state=0.5, description=desc))

        # ── Asset layer ──
        asset_nodes = [
            ("Equity Indices",  "S&P 500, NASDAQ, global indices"),
            ("Bonds",           "UST 10Y, IG, HY"),
            ("Commodities",     "Oil, Gold, Copper, Ag"),
            ("Crypto",          "BTC, ETH, crypto market"),
            ("Sector Indices",  "XLK, XLF, XLE, …"),
        ]
        for name, desc in asset_nodes:
            self.add_node(BayesianNode(name=name, layer="asset", state=0.5, description=desc))

        # ── Edges (macro → market) ──
        self.add_edge("Interest Rates",  "Volatility",       0.6)
        self.add_edge("Interest Rates",  "Credit Spreads",   0.7)
        self.add_edge("Interest Rates",  "Market Liquidity", 0.5)
        self.add_edge("Inflation",       "Interest Rates",   0.8)
        self.add_edge("Inflation",       "Risk Sentiment",   0.4)
        self.add_edge("GDP Growth",      "Risk Sentiment",   0.6)
        self.add_edge("GDP Growth",      "Market Liquidity", 0.4)
        self.add_edge("Liquidity",       "Market Liquidity", 0.8)
        self.add_edge("Liquidity",       "Volatility",       0.5)
        self.add_edge("Monetary Policy", "Interest Rates",   0.9)
        self.add_edge("Monetary Policy", "Liquidity",        0.7)

        # ── Edges (market → asset) ──
        self.add_edge("Volatility",      "Equity Indices",  0.7)
        self.add_edge("Volatility",      "Crypto",          0.6)
        self.add_edge("Risk Sentiment",  "Equity Indices",  0.8)
        self.add_edge("Risk Sentiment",  "Bonds",           0.5)
        self.add_edge("Risk Sentiment",  "Commodities",     0.4)
        self.add_edge("Risk Sentiment",  "Crypto",          0.7)
        self.add_edge("Credit Spreads",  "Bonds",           0.8)
        self.add_edge("Credit Spreads",  "Equity Indices",  0.5)
        self.add_edge("Market Liquidity","Equity Indices",  0.6)
        self.add_edge("Market Liquidity","Crypto",          0.7)
        self.add_edge("Market Liquidity","Sector Indices",  0.5)

        # ── Edges (market → market cross-links) ──
        self.add_edge("Volatility",      "Risk Sentiment",  0.6)
        self.add_edge("Credit Spreads",  "Risk Sentiment",  0.5)

    # ── Propagation ───────────────────────────────────────────────

    def propagate(self, shock_node: str, shock_value: float,
                  iterations: int = 3) -> Dict[str, float]:
        """
        Apply a shock to *shock_node* and propagate forward through the
        DAG.  Returns a dict of {node_name: updated_probability}.
        """
        if shock_node not in self.nodes:
            return {n: self.nodes[n].state for n in self.nodes}

        self.nodes[shock_node].state = max(0.0, min(1.0, shock_value))

        for _ in range(iterations):
            for name, node in self.nodes.items():
                if name == shock_node:
                    continue
                if not node.parents:
                    continue
                # weighted average of parent states
                weighted = 0.0
                total_w = 0.0
                for p in node.parents:
                    w = node.cpt.get(p, 0.5)
                    weighted += self.nodes[p].state * w
                    total_w += w
                if total_w > 0:
                    new_state = weighted / total_w
                    # dampen towards prior (avoid wild swings)
                    node.state = 0.3 * node.state + 0.7 * new_state

        return {n: round(self.nodes[n].state, 4) for n in self.nodes}

    def get_layer_nodes(self, layer: str) -> List[BayesianNode]:
        return [n for n in self.nodes.values() if n.layer == layer]

    def get_edges(self) -> List[Tuple[str, str, float]]:
        edges = []
        for name, node in self.nodes.items():
            for child in node.children:
                weight = self.nodes[child].cpt.get(name, 0.5)
                edges.append((name, child, weight))
        return edges

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict for Streamlit session state."""
        return {
            "nodes": {
                n: {"layer": nd.layer, "state": nd.state,
                    "parents": nd.parents, "children": nd.children,
                    "description": nd.description}
                for n, nd in self.nodes.items()
            },
            "edges": self.get_edges(),
        }


# ────────────────────────────────────────────────────────────────────
# AUTOMATIC BAYESIAN NETWORK FROM DATA
# ────────────────────────────────────────────────────────────────────

def build_bayesian_network_from_data(
    market_data: Any,
    threshold: float = 0.35,
) -> BayesianNetwork:
    """
    Build a Bayesian network from market data by detecting statistical
    dependencies via rolling correlation.

    Parameters
    ----------
    market_data : dict  mapping symbol → DataFrame with 'Close' column
    threshold   : minimum |correlation| to create an edge

    Returns a BayesianNetwork with nodes for the given symbols.
    """
    # The viewer accepts a symbol list for a quick network, while the analytics
    # engine accepts the richer symbol -> OHLC mapping. Normalize both at the
    # boundary so callers cannot fail with ``list.keys()``.
    if isinstance(market_data, (list, tuple, set)):
        symbols = [str(symbol).strip().upper() for symbol in market_data if str(symbol).strip()]
        market_data = _fetch_network_market_data(symbols)
    elif isinstance(market_data, pd.DataFrame):
        market_data = {"ASSET": market_data}
    elif not isinstance(market_data, dict):
        raise TypeError("market_data must be a symbol list, mapping, or DataFrame")

    net = BayesianNetwork()
    # Clear default nodes — we're building from data
    net.nodes.clear()

    symbols = list(market_data.keys())
    # Create nodes
    for sym in symbols:
        layer = _infer_layer(sym)
        net.add_node(BayesianNode(name=sym, layer=layer, state=0.5,
                                  description=f"Auto-detected from data"))

    # Build returns matrix
    returns: Dict[str, pd.Series] = {}
    for sym, df in market_data.items():
        if df is None or df.empty:
            continue
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        returns[sym] = close.pct_change().dropna()

    # Compute pairwise correlations → edges
    sym_list = list(returns.keys())
    for i, s1 in enumerate(sym_list):
        for s2 in sym_list[i + 1:]:
            common_idx = returns[s1].index.intersection(returns[s2].index)
            if len(common_idx) < 30:
                continue
            corr = float(returns[s1].loc[common_idx].corr(returns[s2].loc[common_idx]))
            if abs(corr) >= threshold:
                parent, child = (s1, s2) if corr > 0 else (s2, s1)
                net.add_edge(parent, child, abs(corr))

    return net


def _fetch_network_market_data(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """Fetch enough normalized close history for a quick Bayesian network."""
    result: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            if HAS_DATA_SOURCES:
                if "/" in symbol:
                    frame = get_fx(symbol)
                elif "=F" in symbol:
                    frame = get_futures_proxy(symbol, period="1y")
                else:
                    frame = get_stock(symbol, period="1y")
            elif HAS_YF:
                frame = yf.Ticker(symbol).history(period="1y")
            else:
                frame = None
            if isinstance(frame, pd.DataFrame) and not frame.empty and "Close" in frame:
                result[symbol] = frame
        except Exception:
            continue
    # Preserve requested nodes even when a provider is temporarily unavailable.
    return result or {symbol: pd.DataFrame({"Close": []}) for symbol in symbols}


def _infer_layer(symbol: str) -> str:
    """Guess the Bayesian-network layer from a ticker symbol."""
    macro_keywords = {"^TNX", "^TYX", "^IRX", "DX-Y.NYB", "TLT", "IEF", "SHY"}
    if symbol.upper() in macro_keywords or "=F" in symbol:
        return "macro"
    market_keywords = {"^VIX", "SPY", "QQQ", "DIA", "IWM", "VTI"}
    if symbol.upper() in market_keywords:
        return "market"
    return "asset"


# ────────────────────────────────────────────────────────────────────
# MACRO MARKET ANALYZER
# ────────────────────────────────────────────────────────────────────

@dataclass
class MacroRegimeResult:
    regime: str                     # e.g. "Risk-On (Expansionary)"
    trend_direction: str            # "Bullish", "Bearish", "Neutral"
    volatility_regime: str          # "Low", "Normal", "High", "Extreme"
    liquidity_conditions: str       # "Ample", "Normal", "Tight"
    risk_environment: str           # "Risk-On", "Risk-Off"
    factor_correlations: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""


def run_macro_analysis(
    start_date: str = "",
    end_date: str = "",
    timeframes: Optional[List[str]] = None,
) -> MacroRegimeResult:
    """
    Institutional macro analysis engine.
    Analyses: interest-rate trends, inflation regimes, liquidity cycles,
    GDP growth, currency strength, commodity cycles.

    Returns a structured MacroRegimeResult.
    """
    if timeframes is None:
        timeframes = ["daily", "weekly", "monthly"]

    now = datetime.now()
    if not start_date:
        start_date = (now - timedelta(days=365)).strftime("%Y-%m-%d")
    if not end_date:
        end_date = now.strftime("%Y-%m-%d")

    # Proxy symbols for macro factors
    proxies = {
        "interest_rates":   "^TNX",      # 10Y yield
        "inflation":        "TIP",       # TIPS ETF
        "gdp_growth":       "SPY",       # equity as growth proxy
        "liquidity":        "TLT",       # long treasuries
        "usd_strength":     "UUP",       # USD bull ETF
        "commodities":      "DJP",       # commodity index ETF
        "volatility":       "^VIX",      # VIX
    }

    data: Dict[str, Optional[pd.DataFrame]] = {}
    for factor, sym in proxies.items():
        try:
            if HAS_DATA_SOURCES:
                df = get_stock(sym, period="1y")
            elif HAS_YF:
                df = yf.download(sym, start=start_date, end=end_date,
                                 progress=False)
            else:
                df = None
            data[factor] = df if df is not None and not df.empty else None
        except Exception:
            data[factor] = None

    # ── Derive regime signals ──
    def _safe_return(df: Optional[pd.DataFrame], lookback: int = 63) -> float:
        if df is None or df.empty or len(df) < lookback + 1:
            return 0.0
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        try:
            return float((close.iloc[-1] / close.iloc[-lookback] - 1))
        except Exception:
            return 0.0

    rate_chg   = _safe_return(data.get("interest_rates"), 63)
    infl_chg   = _safe_return(data.get("inflation"), 63)
    growth_chg = _safe_return(data.get("gdp_growth"), 63)
    liq_chg    = _safe_return(data.get("liquidity"), 63)
    usd_chg    = _safe_return(data.get("usd_strength"), 63)
    vol_level  = 0.0
    vix_df = data.get("volatility")
    if vix_df is not None and not vix_df.empty:
        close = vix_df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        vol_level = float(close.iloc[-1]) if len(close) > 0 else 20.0

    # Regime classification
    if vol_level > 30:
        vol_regime = "Extreme"
    elif vol_level > 22:
        vol_regime = "High"
    elif vol_level > 15:
        vol_regime = "Normal"
    else:
        vol_regime = "Low"

    risk_env = "Risk-On" if growth_chg > 0 and vol_level < 22 else "Risk-Off"

    if liq_chg > 0.02:
        liq_cond = "Ample"
    elif liq_chg < -0.02:
        liq_cond = "Tight"
    else:
        liq_cond = "Normal"

    if growth_chg > 0.03:
        trend_dir = "Bullish"
    elif growth_chg < -0.03:
        trend_dir = "Bearish"
    else:
        trend_dir = "Neutral"

    # Composite regime label
    regime_label = f"{risk_env} ({'Expansionary' if growth_chg > 0 else 'Contractionary'})"

    correlations = {
        "Interest Rates vs Equities": round(float(np.corrcoef([rate_chg], [growth_chg])[0, 1]) if rate_chg != 0 else 0, 3),
        "Inflation vs Bonds": round(-infl_chg / max(abs(liq_chg), 0.001), 3),
        "USD vs Commodities": round(-usd_chg / max(abs(rate_chg), 0.001), 3),
    }

    details = {
        "interest_rate_change_63d": round(rate_chg * 100, 2),
        "inflation_proxy_change_63d": round(infl_chg * 100, 2),
        "growth_proxy_change_63d": round(growth_chg * 100, 2),
        "liquidity_proxy_change_63d": round(liq_chg * 100, 2),
        "usd_change_63d": round(usd_chg * 100, 2),
        "vix_level": round(vol_level, 2),
        "timeframes_analysed": timeframes,
    }

    return MacroRegimeResult(
        regime=regime_label,
        trend_direction=trend_dir,
        volatility_regime=vol_regime,
        liquidity_conditions=liq_cond,
        risk_environment=risk_env,
        factor_correlations=correlations,
        details=details,
        timestamp=now.isoformat(),
    )


# ────────────────────────────────────────────────────────────────────
# MICRO MARKET ANALYZER
# ────────────────────────────────────────────────────────────────────

@dataclass
class MicroRegimeResult:
    sector_rankings: List[Dict[str, Any]]
    momentum_breakdown: Dict[str, float]
    breadth_metrics: Dict[str, float]
    volatility_clusters: List[Dict[str, Any]]
    correlation_regime: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""


def run_micro_analysis(
    start_date: str = "",
    end_date: str = "",
    timeframes: Optional[List[str]] = None,
) -> MicroRegimeResult:
    """
    Institutional micro-structure analysis.
    Analyses: sector rotations, momentum factors, market breadth,
    volatility clusters, asset correlations, liquidity proxies.
    """
    if timeframes is None:
        timeframes = ["daily", "weekly"]

    now = datetime.now()
    if not start_date:
        start_date = (now - timedelta(days=180)).strftime("%Y-%m-%d")
    if not end_date:
        end_date = now.strftime("%Y-%m-%d")

    sector_etfs = {
        "Technology": "XLK",
        "Financials": "XLF",
        "Healthcare": "XLV",
        "Energy": "XLE",
        "Industrials": "XLI",
        "Consumer Disc.": "XLY",
        "Consumer Staples": "XLP",
        "Utilities": "XLU",
        "Materials": "XLB",
        "Real Estate": "XLRE",
        "Communication": "XLC",
    }

    # ── Sector rankings ──
    sector_data: Dict[str, pd.DataFrame] = {}
    sector_rankings: List[Dict[str, Any]] = []
    for sector_name, etf in sector_etfs.items():
        try:
            if HAS_DATA_SOURCES:
                df = get_stock(etf, period="6mo")
            elif HAS_YF:
                df = yf.download(etf, start=start_date, end=end_date, progress=False)
            else:
                df = None
            if df is not None and not df.empty:
                sector_data[sector_name] = df
                close = df["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                ret_21d = float(close.iloc[-1] / close.iloc[-min(21, len(close))] - 1) if len(close) > 21 else 0.0
                ret_63d = float(close.iloc[-1] / close.iloc[-min(63, len(close))] - 1) if len(close) > 63 else 0.0
                vol_21d = float(close.pct_change().tail(21).std() * np.sqrt(252)) if len(close) > 22 else 0.0

                sector_rankings.append({
                    "sector": sector_name,
                    "etf": etf,
                    "return_21d": round(ret_21d * 100, 2),
                    "return_63d": round(ret_63d * 100, 2),
                    "volatility_21d": round(vol_21d * 100, 1),
                    "momentum_score": round((ret_21d * 0.6 + ret_63d * 0.4) * 100, 2),
                })
        except Exception:
            continue

    sector_rankings.sort(key=lambda x: x["momentum_score"], reverse=True)

    # ── Momentum breakdown ──
    momentum_breakdown = {}
    for sr in sector_rankings:
        momentum_breakdown[sr["sector"]] = sr["momentum_score"]

    # ── Breadth metrics (simple proxy) ──
    above_50d = 0
    below_50d = 0
    for _, df in sector_data.items():
        try:
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            sma50 = close.rolling(50).mean()
            if len(sma50.dropna()) > 0 and close.iloc[-1] > sma50.iloc[-1]:
                above_50d += 1
            else:
                below_50d += 1
        except Exception:
            continue

    total_sectors = above_50d + below_50d
    breadth_metrics = {
        "pct_above_50d_sma": round(above_50d / max(total_sectors, 1) * 100, 1),
        "sectors_above_50d": above_50d,
        "sectors_below_50d": below_50d,
        "breadth_signal": "Healthy" if above_50d > below_50d else "Weak",
    }

    # ── Volatility clusters ──
    vol_clusters: List[Dict[str, Any]] = []
    for sr in sector_rankings:
        cluster = "Low" if sr["volatility_21d"] < 12 else ("High" if sr["volatility_21d"] > 25 else "Normal")
        vol_clusters.append({"sector": sr["sector"], "vol": sr["volatility_21d"], "cluster": cluster})

    # ── Correlation regime ──
    if len(sector_rankings) >= 5:
        returns_list = [sr["return_21d"] for sr in sector_rankings]
        dispersion = float(np.std(returns_list))
        corr_regime = "High Correlation" if dispersion < 2 else ("Dispersion" if dispersion > 5 else "Normal")
    else:
        corr_regime = "Insufficient Data"

    return MicroRegimeResult(
        sector_rankings=sector_rankings,
        momentum_breakdown=momentum_breakdown,
        breadth_metrics=breadth_metrics,
        volatility_clusters=vol_clusters,
        correlation_regime=corr_regime,
        details={"timeframes": timeframes, "sectors_analysed": len(sector_rankings)},
        timestamp=now.isoformat(),
    )


# ────────────────────────────────────────────────────────────────────
# MULTI-TIMEFRAME REGIME DETECTION
# ────────────────────────────────────────────────────────────────────

@dataclass
class RegimeAnalysis:
    short_term: Dict[str, str]    # 1h–1d
    medium_term: Dict[str, str]   # 1w–1m
    long_term: Dict[str, str]     # 3m–5y
    alignment: str                # "Aligned", "Divergent", "Transitioning"
    conflicts: List[str] = field(default_factory=list)


def detect_regime(
    symbol: str = "SPY",
    lookback_days: int = 252 * 5,
) -> RegimeAnalysis:
    """
    Multi-timeframe regime analysis.
    Compares short-term (≤1d), medium-term (1w-1m), and long-term (3m+).
    """
    try:
        if HAS_DATA_SOURCES:
            df = get_stock(symbol, period="5y")
        elif HAS_YF:
            df = yf.download(symbol, period="5y", progress=False)
        else:
            df = None
    except Exception:
        df = None

    def _trend_label(ret: float) -> str:
        if ret > 0.03:
            return "Bullish"
        elif ret < -0.03:
            return "Bearish"
        return "Neutral"

    def _vol_label(vol: float) -> str:
        if vol > 0.25:
            return "High"
        elif vol < 0.12:
            return "Low"
        return "Normal"

    if df is None or df.empty or len(df) < 30:
        fallback = {"trend": "Unknown", "volatility": "Unknown", "regime": "Unknown"}
        return RegimeAnalysis(
            short_term=fallback, medium_term=fallback, long_term=fallback,
            alignment="Unknown", conflicts=["Insufficient data"],
        )

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    # Short-term (last 5 days)
    st_ret = float(close.iloc[-1] / close.iloc[-min(5, len(close))] - 1)
    st_vol = float(close.pct_change().tail(5).std() * np.sqrt(252))
    short_term = {"trend": _trend_label(st_ret), "volatility": _vol_label(st_vol),
                  "return": f"{st_ret*100:.1f}%", "regime": f"{_trend_label(st_ret)} / {_vol_label(st_vol)} Vol"}

    # Medium-term (last 21 days)
    mt_ret = float(close.iloc[-1] / close.iloc[-min(21, len(close))] - 1)
    mt_vol = float(close.pct_change().tail(21).std() * np.sqrt(252))
    medium_term = {"trend": _trend_label(mt_ret), "volatility": _vol_label(mt_vol),
                   "return": f"{mt_ret*100:.1f}%", "regime": f"{_trend_label(mt_ret)} / {_vol_label(mt_vol)} Vol"}

    # Long-term (last 252 days)
    lt_lookback = min(252, len(close) - 1)
    lt_ret = float(close.iloc[-1] / close.iloc[-lt_lookback] - 1)
    lt_vol = float(close.pct_change().tail(lt_lookback).std() * np.sqrt(252))
    long_term = {"trend": _trend_label(lt_ret), "volatility": _vol_label(lt_vol),
                 "return": f"{lt_ret*100:.1f}%", "regime": f"{_trend_label(lt_ret)} / {_vol_label(lt_vol)} Vol"}

    # Alignment
    trends = [short_term["trend"], medium_term["trend"], long_term["trend"]]
    if len(set(trends)) == 1:
        alignment = "Aligned"
    elif trends.count("Neutral") >= 2:
        alignment = "Transitioning"
    else:
        alignment = "Divergent"

    conflicts = []
    if short_term["trend"] != long_term["trend"] and "Neutral" not in (short_term["trend"], long_term["trend"]):
        conflicts.append(f"Short-term {short_term['trend']} vs Long-term {long_term['trend']}")
    if medium_term["trend"] != long_term["trend"] and "Neutral" not in (medium_term["trend"], long_term["trend"]):
        conflicts.append(f"Medium-term {medium_term['trend']} vs Long-term {long_term['trend']}")

    return RegimeAnalysis(
        short_term=short_term,
        medium_term=medium_term,
        long_term=long_term,
        alignment=alignment,
        conflicts=conflicts,
    )


# ────────────────────────────────────────────────────────────────────
# AI-GENERATED MARKET SCENARIOS
# ────────────────────────────────────────────────────────────────────

@dataclass
class MarketScenario:
    name: str
    probability: float            # 0-1
    macro_impact: str
    market_impact: str
    affected_assets: List[str]
    causal_chain: List[str]       # propagation pathway
    expected_equity_move: float   # percent
    expected_vol_change: float    # percent
    severity: str                 # "Low", "Medium", "High", "Critical"


def generate_market_scenarios(
    market_data: Optional[Dict[str, Any]] = None,
    macro_result: Optional[MacroRegimeResult] = None,
) -> List[MarketScenario]:
    """
    Generate realistic macroeconomic and market-shock scenarios with
    probability estimates and causal propagation pathways.
    """
    vol_regime = "Normal"
    risk_env = "Risk-On"
    if macro_result:
        vol_regime = macro_result.volatility_regime
        risk_env = macro_result.risk_environment

    # Base scenario templates
    scenarios = [
        MarketScenario(
            name="Inflation Spike",
            probability=0.15 if vol_regime in ("High", "Extreme") else 0.10,
            macro_impact="CPI prints 50bp above consensus; core PCE re-accelerates",
            market_impact="Rates reprice higher, equity multiples compress, gold rallies",
            affected_assets=["Equity Indices", "Bonds", "Commodities", "Sector Indices"],
            causal_chain=["Inflation shock", "Central bank tightening expectations",
                          "Higher discount rates", "Equity PE compression",
                          "Risk-off rotation to quality"],
            expected_equity_move=-4.5,
            expected_vol_change=35.0,
            severity="High",
        ),
        MarketScenario(
            name="Central Bank Tightening",
            probability=0.12,
            macro_impact="Surprise 50bp hike or hawkish pivot; forward guidance shifts",
            market_impact="Yield curve flattens, bank stocks rally, tech reprices lower",
            affected_assets=["Bonds", "Equity Indices", "Sector Indices"],
            causal_chain=["Hawkish policy surprise", "Front-end rates surge",
                          "Curve flattening", "Duration sell-off",
                          "Growth-to-value rotation"],
            expected_equity_move=-3.0,
            expected_vol_change=25.0,
            severity="High",
        ),
        MarketScenario(
            name="Liquidity Shock",
            probability=0.08,
            macro_impact="Repo rate spikes, credit markets freeze temporarily",
            market_impact="Risk assets sold, flight to UST, crypto drops sharply",
            affected_assets=["Equity Indices", "Crypto", "Bonds", "Commodities"],
            causal_chain=["Liquidity withdrawal", "Margin calls", "Forced selling",
                          "Volatility spike", "Contagion to credit", "Fed intervention"],
            expected_equity_move=-8.0,
            expected_vol_change=80.0,
            severity="Critical",
        ),
        MarketScenario(
            name="Commodity Supply Disruption",
            probability=0.10,
            macro_impact="Oil supply cut (OPEC+ / geopolitical), 20%+ crude surge",
            market_impact="Energy sector outperforms, transport costs spike, inflation fears",
            affected_assets=["Commodities", "Equity Indices", "Sector Indices"],
            causal_chain=["Supply disruption", "Crude oil +20%",
                          "Energy cost pass-through", "Margin pressure on industrials",
                          "Inflation expectations rise", "Rate repricing"],
            expected_equity_move=-2.5,
            expected_vol_change=20.0,
            severity="Medium",
        ),
        MarketScenario(
            name="Volatility Spike (VIX >35)",
            probability=0.12 if vol_regime == "Low" else 0.18,
            macro_impact="Rapid de-risking event; systematic strategy unwind",
            market_impact="All correlations → 1, diversification fails, cash is king",
            affected_assets=["Equity Indices", "Crypto", "Commodities", "Sector Indices"],
            causal_chain=["Tail event trigger", "Vol-targeting funds deleverage",
                          "CTA trend-reversal", "Market-maker delta hedging",
                          "Liquidity vacuum", "Feedback loop"],
            expected_equity_move=-6.0,
            expected_vol_change=100.0,
            severity="Critical",
        ),
        MarketScenario(
            name="Credit Spread Widening",
            probability=0.10,
            macro_impact="HY spreads widen 100bp+; IG under pressure",
            market_impact="Financials underperform, BBB downgrade cycle fears",
            affected_assets=["Bonds", "Equity Indices", "Sector Indices"],
            causal_chain=["Credit deterioration", "Spread widening",
                          "Funding costs rise", "Reduced buybacks",
                          "Earnings revision cycle"],
            expected_equity_move=-3.5,
            expected_vol_change=30.0,
            severity="High",
        ),
        MarketScenario(
            name="Goldilocks (Soft Landing)",
            probability=0.20 if risk_env == "Risk-On" else 0.10,
            macro_impact="Disinflation continues, employment holds, rate cuts begin",
            market_impact="Equity multiple expansion, credit tightens, cyclicals rally",
            affected_assets=["Equity Indices", "Bonds", "Sector Indices", "Crypto"],
            causal_chain=["Inflation moderates", "Fed cuts 25bp", "Risk appetite rises",
                          "Multiple expansion", "Broadening participation"],
            expected_equity_move=5.0,
            expected_vol_change=-15.0,
            severity="Low",
        ),
        MarketScenario(
            name="Geopolitical Escalation",
            probability=0.08,
            macro_impact="Major conflict escalation; energy/trade disruption",
            market_impact="Safe-haven bid (UST, Gold, CHF), risk assets sold",
            affected_assets=["Equity Indices", "Commodities", "Bonds", "Crypto"],
            causal_chain=["Geopolitical event", "Oil supply risk premium",
                          "Flight to safety", "Defence stocks rally",
                          "Consumer confidence drops"],
            expected_equity_move=-5.5,
            expected_vol_change=50.0,
            severity="High",
        ),
    ]

    # Normalise probabilities to sum to 1
    total_prob = sum(s.probability for s in scenarios)
    if total_prob > 0:
        for s in scenarios:
            s.probability = round(s.probability / total_prob, 3)

    return scenarios


# ────────────────────────────────────────────────────────────────────
# CONVENIENCE / INTEGRATION
# ────────────────────────────────────────────────────────────────────

def get_institutional_summary() -> Dict[str, Any]:
    """One-call function returning all institutional analytics."""
    macro = run_macro_analysis()
    micro = run_micro_analysis()
    regime = detect_regime()
    scenarios = generate_market_scenarios(macro_result=macro)
    network = BayesianNetwork()

    return {
        "macro": macro,
        "micro": micro,
        "regime": regime,
        "scenarios": scenarios,
        "bayesian_network": network,
    }
