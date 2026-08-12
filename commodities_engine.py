"""
Commodities Engine - Octavian Terminal
Institutional-grade commodities analysis, term structure modeling,
spread trading, and seasonal pattern detection.
"""

import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

logger = logging.getLogger("OctavianCommodities")

# ---------------------------------------------------------------------------
# COMMODITY UNIVERSE
# ---------------------------------------------------------------------------

COMMODITY_UNIVERSE = {
    # Energy
    "CL=F": {"name": "WTI Crude Oil", "sector": "Energy", "unit": "$/bbl", "contract_size": 1000, "tick": 0.01, "margin": 6500},
    "BZ=F": {"name": "Brent Crude Oil", "sector": "Energy", "unit": "$/bbl", "contract_size": 1000, "tick": 0.01, "margin": 7000},
    "NG=F": {"name": "Natural Gas", "sector": "Energy", "unit": "$/MMBtu", "contract_size": 10000, "tick": 0.001, "margin": 3500},
    "RB=F": {"name": "RBOB Gasoline", "sector": "Energy", "unit": "$/gal", "contract_size": 42000, "tick": 0.0001, "margin": 7500},
    "HO=F": {"name": "Heating Oil", "sector": "Energy", "unit": "$/gal", "contract_size": 42000, "tick": 0.0001, "margin": 6000},
    # Precious Metals
    "GC=F": {"name": "Gold", "sector": "Precious Metals", "unit": "$/oz", "contract_size": 100, "tick": 0.10, "margin": 9000},
    "SI=F": {"name": "Silver", "sector": "Precious Metals", "unit": "$/oz", "contract_size": 5000, "tick": 0.005, "margin": 8500},
    "PL=F": {"name": "Platinum", "sector": "Precious Metals", "unit": "$/oz", "contract_size": 50, "tick": 0.10, "margin": 4500},
    "PA=F": {"name": "Palladium", "sector": "Precious Metals", "unit": "$/oz", "contract_size": 100, "tick": 0.05, "margin": 15000},
    # Base Metals
    "HG=F": {"name": "Copper", "sector": "Base Metals", "unit": "$/lb", "contract_size": 25000, "tick": 0.0005, "margin": 5500},
    # Agriculture
    "ZC=F": {"name": "Corn", "sector": "Agriculture", "unit": "cents/bu", "contract_size": 5000, "tick": 0.25, "margin": 1500},
    "ZS=F": {"name": "Soybeans", "sector": "Agriculture", "unit": "cents/bu", "contract_size": 5000, "tick": 0.25, "margin": 2500},
    "ZW=F": {"name": "Wheat", "sector": "Agriculture", "unit": "cents/bu", "contract_size": 5000, "tick": 0.25, "margin": 2000},
    "KC=F": {"name": "Coffee", "sector": "Softs", "unit": "cents/lb", "contract_size": 37500, "tick": 0.05, "margin": 4500},
    "SB=F": {"name": "Sugar", "sector": "Softs", "unit": "cents/lb", "contract_size": 112000, "tick": 0.01, "margin": 1200},
    "CC=F": {"name": "Cocoa", "sector": "Softs", "unit": "$/mt", "contract_size": 10, "tick": 1.0, "margin": 3000},
    "CT=F": {"name": "Cotton", "sector": "Softs", "unit": "cents/lb", "contract_size": 50000, "tick": 0.01, "margin": 2500},
    # Livestock
    "LE=F": {"name": "Live Cattle", "sector": "Livestock", "unit": "cents/lb", "contract_size": 40000, "tick": 0.025, "margin": 2000},
    "HE=F": {"name": "Lean Hogs", "sector": "Livestock", "unit": "cents/lb", "contract_size": 40000, "tick": 0.025, "margin": 1500},
}

SPREAD_TEMPLATES = {
    "Crack Spread (3:2:1)": {"legs": [("CL=F", 3), ("RB=F", -2), ("HO=F", -1)], "sector": "Energy"},
    "Gold/Silver Ratio": {"legs": [("GC=F", 1), ("SI=F", -1)], "sector": "Precious Metals"},
    "Soybean Crush": {"legs": [("ZS=F", 1), ("ZC=F", -1)], "sector": "Agriculture"},
    "Crude Brent-WTI": {"legs": [("BZ=F", 1), ("CL=F", -1)], "sector": "Energy"},
    "Copper/Gold Ratio": {"legs": [("HG=F", 1), ("GC=F", -1)], "sector": "Macro"},
}

SEASONAL_PATTERNS = {
    "CL=F": {1: -0.02, 2: 0.01, 3: 0.03, 4: 0.04, 5: 0.02, 6: 0.01, 7: -0.01, 8: -0.02, 9: -0.03, 10: 0.01, 11: -0.01, 12: 0.02},
    "NG=F": {1: 0.05, 2: -0.03, 3: -0.05, 4: -0.04, 5: -0.02, 6: 0.02, 7: 0.03, 8: 0.02, 9: 0.01, 10: 0.04, 11: 0.06, 12: 0.05},
    "GC=F": {1: 0.03, 2: 0.02, 3: -0.01, 4: 0.01, 5: -0.02, 6: 0.01, 7: 0.02, 8: 0.03, 9: 0.01, 10: -0.01, 11: 0.02, 12: 0.03},
    "ZC=F": {1: -0.01, 2: 0.01, 3: 0.02, 4: 0.03, 5: 0.04, 6: 0.05, 7: 0.03, 8: -0.02, 9: -0.04, 10: -0.03, 11: -0.02, 12: 0.01},
    "ZS=F": {1: 0.02, 2: 0.01, 3: 0.03, 4: 0.04, 5: 0.05, 6: 0.03, 7: -0.01, 8: -0.03, 9: -0.04, 10: -0.02, 11: 0.01, 12: 0.02},
    "ZW=F": {1: -0.01, 2: 0.02, 3: 0.03, 4: 0.04, 5: 0.05, 6: 0.02, 7: -0.03, 8: -0.04, 9: -0.03, 10: -0.01, 11: 0.01, 12: 0.02},
    "HG=F": {1: 0.03, 2: 0.02, 3: 0.01, 4: 0.02, 5: -0.01, 6: -0.02, 7: 0.01, 8: -0.01, 9: -0.02, 10: 0.02, 11: 0.01, 12: 0.03},
}


@dataclass
class CommodityAnalysis:
    symbol: str
    name: str
    sector: str
    current_price: float
    change_1d_pct: float
    change_5d_pct: float
    change_1m_pct: float
    volatility_annual: float
    rsi: float
    momentum_score: float
    trend_direction: str
    seasonal_bias: str
    seasonal_return: float
    support_level: float
    resistance_level: float
    signal: str  # BULLISH / BEARISH / NEUTRAL
    confidence: float
    risk_factors: List[str] = field(default_factory=list)
    catalysts: List[str] = field(default_factory=list)


@dataclass
class SpreadAnalysis:
    spread_name: str
    current_value: float
    historical_mean: float
    z_score: float
    direction: str
    confidence: float
    description: str


class CommoditiesEngine:
    """
    Institutional-grade Commodities Analysis Engine.
    Provides term structure, seasonal, spread, and technical analysis.
    """

    def __init__(self):
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._cache_ttl = 300

    def _cache_get(self, key: str):
        if key in self._cache:
            ts, val = self._cache[key]
            import time
            if time.time() - ts < self._cache_ttl:
                return val
        return None

    def _cache_set(self, key: str, val: Any):
        import time
        self._cache[key] = (time.time(), val)

    def _fetch_data(self, symbol: str, period: str = "6mo") -> Optional[pd.DataFrame]:
        cached = self._cache_get(f"data:{symbol}:{period}")
        if cached is not None:
            return cached
        if not HAS_YF:
            return None
        try:
            df = yf.download(symbol, period=period, interval="1d", progress=False)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                self._cache_set(f"data:{symbol}:{period}", df)
                return df
        except Exception as e:
            logger.warning(f"Failed to fetch {symbol}: {e}")
        return None

    def get_universe(self) -> Dict[str, Dict]:
        return COMMODITY_UNIVERSE

    def get_sectors(self) -> List[str]:
        return list(set(v["sector"] for v in COMMODITY_UNIVERSE.values()))

    def get_symbols_by_sector(self, sector: str) -> List[str]:
        return [k for k, v in COMMODITY_UNIVERSE.items() if v["sector"] == sector]

    # -------------------------------------------------------------------
    # CORE ANALYSIS
    # -------------------------------------------------------------------

    def analyze_commodity(self, symbol: str) -> Optional[CommodityAnalysis]:
        meta = COMMODITY_UNIVERSE.get(symbol)
        if not meta:
            return None

        df = self._fetch_data(symbol)
        if df is None or df.empty or len(df) < 20:
            return None

        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna().astype(float)
        prices = close.values
        current = float(prices[-1])

        # Returns
        c1d = ((prices[-1] / prices[-2]) - 1) * 100 if len(prices) >= 2 else 0
        c5d = ((prices[-1] / prices[-6]) - 1) * 100 if len(prices) >= 6 else 0
        c1m = ((prices[-1] / prices[-22]) - 1) * 100 if len(prices) >= 22 else 0

        # Volatility
        rets = np.diff(prices) / prices[:-1]
        vol = float(np.std(rets[-60:]) * np.sqrt(252)) if len(rets) >= 60 else float(np.std(rets) * np.sqrt(252))

        # RSI
        rsi = self._calc_rsi(prices)

        # Momentum
        mom = float(np.mean([c1d, c5d / 5, c1m / 22]) * 10)

        # Trend
        sma20 = np.mean(prices[-20:]) if len(prices) >= 20 else current
        sma50 = np.mean(prices[-50:]) if len(prices) >= 50 else current
        trend = "BULLISH" if current > sma20 > sma50 else "BEARISH" if current < sma20 < sma50 else "NEUTRAL"

        # Seasonal
        month = datetime.now().month
        seasonal = SEASONAL_PATTERNS.get(symbol, {})
        s_ret = seasonal.get(month, 0)
        s_bias = "BULLISH" if s_ret > 0.01 else "BEARISH" if s_ret < -0.01 else "NEUTRAL"

        # Support / Resistance
        support = float(np.min(prices[-20:]))
        resistance = float(np.max(prices[-20:]))

        # Signal
        score = 0
        if trend == "BULLISH": score += 30
        elif trend == "BEARISH": score -= 30
        if rsi < 30: score += 20
        elif rsi > 70: score -= 20
        if s_bias == "BULLISH": score += 10
        elif s_bias == "BEARISH": score -= 10
        score += np.clip(mom, -20, 20)

        signal = "BULLISH" if score > 15 else "BEARISH" if score < -15 else "NEUTRAL"
        confidence = min(abs(score) / 80 * 100, 95)

        # Risk factors
        risks = []
        if vol > 0.35: risks.append(f"High volatility ({vol:.0%} annualized)")
        if rsi > 75: risks.append("Overbought RSI - pullback risk")
        if rsi < 25: risks.append("Oversold RSI - reversal potential")

        catalysts = []
        if meta["sector"] == "Energy":
            catalysts.append("OPEC+ supply decisions")
            catalysts.append("US inventory data (EIA)")
        elif meta["sector"] == "Precious Metals":
            catalysts.append("Federal Reserve rate policy")
            catalysts.append("USD strength/weakness")
        elif meta["sector"] == "Agriculture":
            catalysts.append("USDA crop reports")
            catalysts.append("Weather patterns in growing regions")

        return CommodityAnalysis(
            symbol=symbol, name=meta["name"], sector=meta["sector"],
            current_price=current, change_1d_pct=c1d, change_5d_pct=c5d,
            change_1m_pct=c1m, volatility_annual=vol, rsi=rsi,
            momentum_score=mom, trend_direction=trend,
            seasonal_bias=s_bias, seasonal_return=s_ret,
            support_level=support, resistance_level=resistance,
            signal=signal, confidence=confidence,
            risk_factors=risks, catalysts=catalysts,
        )

    def analyze_all(self, sector: Optional[str] = None) -> List[CommodityAnalysis]:
        symbols = self.get_symbols_by_sector(sector) if sector else list(COMMODITY_UNIVERSE.keys())
        results = []
        for s in symbols:
            try:
                a = self.analyze_commodity(s)
                if a:
                    results.append(a)
            except Exception as e:
                logger.warning(f"Analysis failed for {s}: {e}")
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results

    # -------------------------------------------------------------------
    # SPREAD ANALYSIS
    # -------------------------------------------------------------------

    def analyze_spread(self, spread_name: str) -> Optional[SpreadAnalysis]:
        template = SPREAD_TEMPLATES.get(spread_name)
        if not template:
            return None

        values = []
        for symbol, weight in template["legs"]:
            df = self._fetch_data(symbol, period="1y")
            if df is None or df.empty:
                return None
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            values.append(close.dropna().astype(float) * weight)

        # Align on common dates
        combined = pd.concat(values, axis=1).dropna()
        if combined.empty or len(combined) < 30:
            return None

        spread_series = combined.sum(axis=1)
        current = float(spread_series.iloc[-1])
        mean = float(spread_series.mean())
        std = float(spread_series.std())
        z = (current - mean) / std if std > 0 else 0

        direction = "SELL" if z > 1.5 else "BUY" if z < -1.5 else "NEUTRAL"
        confidence = min(abs(z) / 3 * 100, 90)

        return SpreadAnalysis(
            spread_name=spread_name, current_value=current,
            historical_mean=mean, z_score=z,
            direction=direction, confidence=confidence,
            description=f"Z-score: {z:+.2f} | Mean: {mean:.2f} | Current: {current:.2f}",
        )

    # -------------------------------------------------------------------
    # SEASONAL ANALYSIS
    # -------------------------------------------------------------------

    def get_seasonal_calendar(self, symbol: str) -> Dict[int, float]:
        return SEASONAL_PATTERNS.get(symbol, {})

    # -------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------

    def _calc_rsi(self, prices, n=14) -> float:
        if len(prices) < n + 1:
            return 50.0
        deltas = np.diff(prices[-(n + 1):])
        up = np.mean(np.where(deltas > 0, deltas, 0))
        down = np.mean(np.where(deltas < 0, -deltas, 0))
        if down == 0:
            return 100.0
        rs = up / down
        return float(100 - (100 / (1 + rs)))

    def is_commodity(self, symbol: str) -> bool:
        return symbol in COMMODITY_UNIVERSE


# ---------------------------------------------------------------------------
# SINGLETON
# ---------------------------------------------------------------------------

_engine: Optional[CommoditiesEngine] = None

def get_commodities_engine() -> CommoditiesEngine:
    global _engine
    if _engine is None:
        _engine = CommoditiesEngine()
    return _engine
