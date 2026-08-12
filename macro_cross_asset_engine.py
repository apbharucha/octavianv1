"""
Macro Cross-Asset Engine
=========================
Tracks and quantifies relationships across asset classes to detect regime
shifts, lead-lag signals, and contagion risk.

Core Relationships Modeled:
  1. Bond Yields ↔ Equity Valuations (rate sensitivity, duration risk)
  2. Oil → Inflation → Interest Rates (commodity-macro chain)
  3. USD Strength ↔ Commodities (dollar dominance effects)
  4. Liquidity → Risk Assets → Crypto (liquidity transmission)
  5. Credit Spreads → Equity Risk Premium
  6. Yield Curve → Economic Regime → Sector Rotation
  7. VIX → Cross-Asset Volatility Transmission
  8. Global Growth → EM vs DM Rotation

Each relationship produces:
  - current_reading, historical_percentile, signal_direction,
    strength, lead_lag_days, trade_implications
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

try:
    import yfinance as yf

    HAS_YF = True
except ImportError:
    HAS_YF = False

try:
    from scipy import stats as scipy_stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# 
# Data Structures
# 


@dataclass
class CrossAssetSignal:
    relationship: str
    chain: str  # e.g. "Oil → Inflation → Rates"
    assets: list[str]
    direction: str  # BULLISH / BEARISH / NEUTRAL / WATCH
    strength: float  # 0-100
    confidence: float  # 0-100
    current_reading: float
    historical_percentile: float
    lead_lag_days: int  # positive = asset_a leads asset_b by N days
    description: str
    trade_implications: list[str]
    regime: str  # current macro regime label
    metadata: dict = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class MacroRegime:
    name: str  # e.g. "Risk-On Growth", "Stagflation", "Deflationary Bust"
    probability: float  # 0-100
    description: str
    favoured_assets: list[str]
    avoided_assets: list[str]
    historical_analogue: str
    key_indicators: list[str]
    typical_duration_months: int


@dataclass
class CrossAssetDashboard:
    timestamp: str
    signals: list[CrossAssetSignal]
    macro_regime: MacroRegime
    contagion_risk: float  # 0-100
    correlation_breakdown_risk: float
    liquidity_score: float  # 0-100 (100 = abundant liquidity)
    yield_curve_regime: str
    dollar_regime: str
    volatility_regime: str
    sector_rotation_signal: str
    summary: str


# 
# Asset Universe
# 

# Primary macro proxies
MACRO_PROXIES = {
    # Rates
    "10Y_YIELD": "^TNX",
    "2Y_YIELD": "^IRX",
    "30Y_YIELD": "^TYX",
    "TLT": "TLT",  # Long bond ETF
    "SHY": "SHY",  # Short bond ETF
    # Equities
    "SPY": "SPY",
    "QQQ": "QQQ",
    "IWM": "IWM",  # Small cap
    "EEM": "EEM",  # Emerging markets
    "EFA": "EFA",  # Developed ex-US
    # Commodities
    "OIL": "USO",  # Oil
    "GOLD": "GLD",  # Gold
    "SILVER": "SLV",
    "COPPER": "CPER",  # Copper
    "COMMODITIES": "GSG",  # Broad commodities
    # Dollar
    "USD": "UUP",  # Dollar index ETF
    # Credit
    "HY_CREDIT": "HYG",  # High yield
    "IG_CREDIT": "LQD",  # Investment grade
    # Volatility
    "VIX": "^VIX",
    # Crypto
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    # Sectors
    "XLE": "XLE",  # Energy
    "XLF": "XLF",  # Financials
    "XLU": "XLU",  # Utilities
    "XLI": "XLI",  # Industrials
    "XLK": "XLK",  # Tech
    "XLRE": "XLRE",  # Real estate
}

# Sector sensitivity to macro factors
SECTOR_RATE_SENSITIVITY = {
    "XLU": -1.5,  # Utilities: very rate-sensitive (negative)
    "XLRE": -1.3,  # Real estate: rate-sensitive
    "XLK": -0.8,  # Tech: duration-sensitive (high P/E)
    "XLF": +1.2,  # Financials: benefit from higher rates
    "XLE": +0.3,  # Energy: mild positive
    "XLV": -0.3,  # Healthcare: mild negative
    "XLI": +0.1,  # Industrials: neutral
    "XLY": -0.5,  # Consumer Discretionary: negative
    "XLP": -0.6,  # Consumer Staples: defensive but rate-sensitive
}

SECTOR_DOLLAR_SENSITIVITY = {
    "XLE": -0.8,  # Energy: commodities priced in USD
    "XLK": -0.5,  # Tech: large international revenue
    "XLF": +0.3,  # Financials: mild positive
    "XLI": -0.4,  # Industrials: exports hurt by strong USD
    "XLU": +0.2,  # Utilities: domestic, USD positive
    "XLP": -0.2,  # Staples: international exposure
}

# Historical macro regime definitions
MACRO_REGIMES = {
    "Risk-On / Goldilocks": MacroRegime(
        name="Risk-On / Goldilocks",
        probability=0.0,
        description=(
            "Growth above trend, inflation moderate, central banks accommodative or neutral. "
            "Equities outperform, credit spreads tight, volatility low."
        ),
        favoured_assets=["SPY", "QQQ", "HYG", "EEM", "XLK", "XLY"],
        avoided_assets=["TLT", "GLD", "UUP", "XLU"],
        historical_analogue="2003-2007, 2010-2019, 2023",
        key_indicators=[
            "Yield curve positively sloped",
            "ISM Manufacturing > 50",
            "Credit spreads < 400bps",
            "VIX < 18",
        ],
        typical_duration_months=18,
    ),
    "Stagflation": MacroRegime(
        name="Stagflation",
        probability=0.0,
        description=(
            "High inflation + slowing growth. Central banks forced to hike into weakness. "
            "Hardest regime for traditional 60/40 portfolios — both stocks and bonds fall."
        ),
        favoured_assets=["GLD", "USO", "CPER", "XLE", "SLV", "TIPS"],
        avoided_assets=["TLT", "QQQ", "XLU", "XLRE", "HYG"],
        historical_analogue="1973-1974, 1979-1980, partial 2022",
        key_indicators=[
            "CPI > 5% with falling PMI",
            "Yield curve flat or inverted",
            "Real rates negative but rising",
            "Energy prices surging",
        ],
        typical_duration_months=12,
    ),
    "Deflationary Bust": MacroRegime(
        name="Deflationary Bust",
        probability=0.0,
        description=(
            "Growth collapsing, inflation falling, credit stress. "
            "Flight to quality. Long bonds outperform. Cash is king short-term."
        ),
        favoured_assets=["TLT", "GLD", "SHY", "UUP", "XLV", "XLP"],
        avoided_assets=["HYG", "XLE", "EEM", "IWM", "XLK"],
        historical_analogue="2008-2009, Q1 2020, 2000-2002",
        key_indicators=[
            "Yield curve inverted then steepening via rally",
            "Credit spreads > 600bps",
            "VIX > 30",
            "Unemployment rising sharply",
        ],
        typical_duration_months=8,
    ),
    "Inflationary Boom": MacroRegime(
        name="Inflationary Boom",
        probability=0.0,
        description=(
            "Strong growth, rising inflation, central banks beginning to hike. "
            "Cyclicals and real assets outperform. Tech underperforms on multiple compression."
        ),
        favoured_assets=["XLE", "XLI", "XLF", "GLD", "CPER", "USO", "IWM"],
        avoided_assets=["TLT", "QQQ", "XLRE", "XLU"],
        historical_analogue="1999-2000, 2021, early 2022",
        key_indicators=[
            "PMI > 55, CPI > 4%",
            "Yield curve steepening",
            "Commodity prices surging",
            "Banks outperforming tech",
        ],
        typical_duration_months=10,
    ),
    "Risk-Off / Slow Growth": MacroRegime(
        name="Risk-Off / Slow Growth",
        probability=0.0,
        description=(
            "Growth below trend, inflation moderate, policy uncertain. "
            "Defensive positioning. Quality and low-vol outperform."
        ),
        favoured_assets=["TLT", "GLD", "XLV", "XLP", "XLU", "UUP"],
        avoided_assets=["HYG", "EEM", "IWM", "XLE", "XLY"],
        historical_analogue="2015-2016, 2018 Q4, late 2023",
        key_indicators=[
            "ISM below 50, VIX 20-30",
            "Yield curve flattening",
            "Credit spreads widening slowly",
            "Dollar strengthening",
        ],
        typical_duration_months=6,
    ),
}


# 
# Data Access Layer
# 


class MacroDataLayer:
    """Fetches and caches macro asset price data."""

    def __init__(self, cache_ttl: int = 600):
        self._cache: dict[str, tuple[float, Any]] = {}
        self._ttl = cache_ttl

    def _cache_get(self, key: str) -> Any | None:
        entry = self._cache.get(key)
        if entry and time.time() - entry[0] < self._ttl:
            return entry[1]
        return None

    def _cache_set(self, key: str, val: Any) -> None:
        self._cache[key] = (time.time(), val)

    def get_close(self, ticker: str, period: str = "2y") -> pd.Series | None:
        key = f"close:{ticker}:{period}"
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        if not HAS_YF:
            return None
        try:
            # Small delay between API calls to prevent rate limiting
            time.sleep(0.15)
            raw = yf.download(
                ticker,
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
                timeout=12,
            )
            if raw is None or raw.empty:
                return None
            close = raw["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.dropna()
            if len(close) < 20:
                return None
            self._cache_set(key, close)
            return close
        except Exception:
            return None

    def get_returns(self, ticker: str, period: str = "2y") -> pd.Series | None:
        close = self.get_close(ticker, period)
        if close is None or len(close) < 2:
            return None
        return close.pct_change().dropna()

    def get_momentum(
        self, ticker: str, days: int = 63, period: str = "2y"
    ) -> float | None:
        close = self.get_close(ticker, period)
        if close is None or len(close) < days + 1:
            return None
        return float(close.iloc[-1] / close.iloc[-days - 1] - 1)

    def get_rolling_corr(
        self, ticker_a: str, ticker_b: str, window: int = 60, period: str = "2y"
    ) -> pd.Series | None:
        ret_a = self.get_returns(ticker_a, period)
        ret_b = self.get_returns(ticker_b, period)
        if ret_a is None or ret_b is None:
            return None
        aligned = pd.concat([ret_a, ret_b], axis=1).dropna()
        if len(aligned) < window + 10:
            return None
        aligned.columns = [ticker_a, ticker_b]
        return aligned[ticker_a].rolling(window).corr(aligned[ticker_b]).dropna()

    def get_zscore(
        self, ticker: str, window: int = 252, period: str = "2y"
    ) -> float | None:
        close = self.get_close(ticker, period)
        if close is None or len(close) < window:
            return None
        tail = close.tail(window)
        return float((close.iloc[-1] - tail.mean()) / max(tail.std(), 1e-10))

    def get_vol(
        self, ticker: str, window: int = 20, period: str = "1y"
    ) -> float | None:
        rets = self.get_returns(ticker, period)
        if rets is None or len(rets) < window:
            return None
        return float(rets.tail(window).std() * math.sqrt(252) * 100)

    def compute_lead_lag(
        self, ticker_a: str, ticker_b: str, max_lag: int = 20, period: str = "2y"
    ) -> dict:
        """
        Find the lag at which ticker_a best predicts ticker_b.
        Returns: {'best_lag': int, 'best_corr': float}
        Positive best_lag = ticker_a leads ticker_b by N days.
        """
        ret_a = self.get_returns(ticker_a, period)
        ret_b = self.get_returns(ticker_b, period)
        if ret_a is None or ret_b is None:
            return {"best_lag": 0, "best_corr": 0.0}
        aligned = pd.concat([ret_a, ret_b], axis=1).dropna()
        aligned.columns = ["A", "B"]
        if len(aligned) < max_lag * 3:
            return {"best_lag": 0, "best_corr": 0.0}

        best_lag = 0
        best_corr = float(aligned["A"].corr(aligned["B"]))
        for lag in range(1, max_lag + 1):
            try:
                c = float(aligned["A"].shift(lag).corr(aligned["B"]))
                if abs(c) > abs(best_corr):
                    best_corr = c
                    best_lag = lag
            except Exception:
                pass
        return {"best_lag": best_lag, "best_corr": round(best_corr, 4)}


# 
# Cross-Asset Analysis Engine
# 


class MacroCrossAssetEngine:
    """
    Analyzes and quantifies macro cross-asset relationships.
    Generates regime-aware trade signals based on inter-market dynamics.
    """

    def __init__(self):
        self.data = MacroDataLayer()
        self._rng_base = int(datetime.utcnow().strftime("%Y%m%d%H"))

    def _rng(self, salt: str = "") -> np.random.Generator:
        seed = int(hashlib.md5(f"{self._rng_base}:{salt}".encode()).hexdigest(), 16) % (
            2**32
        )
        return np.random.default_rng(seed)

    def _pct(self, val: float | None, lo: float = -0.5, hi: float = 0.5) -> float:
        """Convert a return to a 0-100 percentile assuming normal distribution."""
        if val is None:
            return 50.0
        if HAS_SCIPY:
            return float(scipy_stats.norm.cdf(val, loc=0, scale=(hi - lo) / 4) * 100)
        return float(np.clip(50 + val * 200, 0, 100))

    # 
    # 1. BOND YIELDS ↔ EQUITY VALUATIONS
    # 

    def analyze_bond_equity_relationship(self) -> CrossAssetSignal:
        """
        The Fed Model: earnings yield vs 10Y yield.
        When yields rise, equity multiples compress (inverse relationship).
        Detects: rate sensitivity regime, equity premium attractiveness.
        """
        tlt_mom_3m = self.data.get_momentum("TLT", 63)
        spy_mom_3m = self.data.get_momentum("SPY", 63)
        tlt_mom_1m = self.data.get_momentum("TLT", 21)
        spy_mom_1m = self.data.get_momentum("SPY", 21)

        # Rolling correlation (60-day)
        corr_series = self.data.get_rolling_corr("TLT", "SPY", window=60)
        recent_corr = (
            float(corr_series.iloc[-1])
            if corr_series is not None and len(corr_series) > 0
            else -0.3
        )
        hist_corr = (
            float(corr_series.mean())
            if corr_series is not None and len(corr_series) > 20
            else -0.3
        )

        tlt_z = self.data.get_zscore("TLT", 252) or 0.0
        spy_z = self.data.get_zscore("SPY", 252) or 0.0

        # Bond-equity regime classification
        tlt_up = (tlt_mom_3m or 0) > 0.02
        spy_up = (spy_mom_3m or 0) > 0.02

        if tlt_up and spy_up:
            regime = "Both rising: growth optimism + rate cut expectations"
            direction = "BULLISH"
            implications = [
                "Bonds and equities both rising: market pricing simultaneous growth + rate cuts",
                "Historically short-lived; watch for one leg reversing",
                "Long equities, hedge with gold as insurance",
            ]
        elif not tlt_up and spy_up:
            regime = "Equities up, bonds down: reflation / growth without cuts"
            direction = "BULLISH"
            implications = [
                "Classic risk-on: equities price growth, bonds price higher rates",
                "Favour cyclicals (XLF, XLE, XLI) over defensives",
                "Underweight duration (TLT, XLRE, XLU)",
                "Rate-sensitive sectors vulnerable if yields continue rising",
            ]
        elif tlt_up and not spy_up:
            regime = "Bonds up, equities down: flight to safety / recession fears"
            direction = "BEARISH"
            implications = [
                "Flight-to-quality: bond market pricing economic slowdown",
                "Historically equities follow bonds with 1-3 month lag",
                "Increase defensive positioning (XLV, XLP, TLT, GLD)",
                "Reduce cyclical exposure",
            ]
        else:
            regime = "Both falling: stagflationary bust or liquidity crisis"
            direction = "BEARISH"
            implications = [
                "Worst regime for 60/40: stocks AND bonds falling simultaneously",
                "Stagflation signal: favour commodities, gold, energy",
                "Avoid duration; avoid growth equities",
                "TIPS, commodity ETFs (GSG, GLD, USO) as alternatives",
            ]

        # Correlation regime commentary
        corr_shift = recent_corr - hist_corr
        if corr_shift > 0.3:
            implications.append(
                f"WARNING: Bond-equity correlation turning positive ({recent_corr:.2f}) — "
                "traditional 60/40 diversification is breaking down (stagflation regime signal)"
            )

        strength = min(95, 50 + abs((tlt_mom_3m or 0) - (spy_mom_3m or 0)) * 300)
        reading = float((tlt_z - spy_z))

        return CrossAssetSignal(
            relationship="Bond Yields ↔ Equity Valuations",
            chain="10Y Yield → P/E Multiple → Equity Valuations",
            assets=["TLT", "SPY", "^TNX"],
            direction=direction,
            strength=round(strength, 1),
            confidence=68.0,
            current_reading=round(reading, 3),
            historical_percentile=round(self._pct(reading, -3, 3), 1),
            lead_lag_days=30,
            description=(
                f"Bond-equity regime: '{regime}'. "
                f"TLT 3M: {(tlt_mom_3m or 0):+.1%} | SPY 3M: {(spy_mom_3m or 0):+.1%}. "
                f"60-day correlation: {recent_corr:.2f} (hist avg {hist_corr:.2f}). "
                f"TLT Z-score: {tlt_z:+.2f} | SPY Z-score: {spy_z:+.2f}."
            ),
            trade_implications=implications,
            regime=regime,
            metadata={
                "tlt_mom_3m": tlt_mom_3m,
                "spy_mom_3m": spy_mom_3m,
                "bond_equity_corr_60d": recent_corr,
                "corr_shift": corr_shift,
                "tlt_z": tlt_z,
                "spy_z": spy_z,
            },
        )

    # 
    # 2. OIL → INFLATION → INTEREST RATES CHAIN
    # 

    def analyze_oil_inflation_rates_chain(self) -> CrossAssetSignal:
        """
        Oil prices are the primary driver of headline CPI.
        Rising oil → inflation expectations rise → bond yields rise → equity multiples compress.
        Lead time: oil moves ~6-8 weeks ahead of CPI prints.
        """
        uso_mom_1m = self.data.get_momentum("USO", 21)
        uso_mom_3m = self.data.get_momentum("USO", 63)
        tlt_mom_1m = self.data.get_momentum("TLT", 21)
        gld_mom_1m = self.data.get_momentum("GLD", 21)
        xle_mom_1m = self.data.get_momentum("XLE", 21)

        uso_z = self.data.get_zscore("USO", 252) or 0.0

        # Lead-lag: oil leads CPI, CPI leads rates
        # We use TLT inverse as rates proxy
        ll_result = self.data.compute_lead_lag("USO", "TLT", max_lag=30)
        best_lag = ll_result.get("best_lag", 15)
        best_corr = ll_result.get("best_corr", -0.3)

        oil_surging = (uso_mom_1m or 0) > 0.08
        oil_collapsing = (uso_mom_1m or 0) < -0.08
        oil_rising = (uso_mom_1m or 0) > 0.02
        oil_falling = (uso_mom_1m or 0) < -0.02

        if oil_surging:
            direction = "BEARISH"
            regime = "Oil Shock — Inflation Risk Rising"
            implications = [
                f"Oil +{(uso_mom_1m or 0):.1%} in 1M — CPI upside risk in 6-8 weeks",
                "Bonds (TLT) likely to face selling pressure as inflation expectations rise",
                "Rate-sensitive sectors (XLRE, XLU) at risk of derating",
                "Beneficiaries: XLE (energy stocks), inflation-linked assets, GLD",
                f"Historical: oil spikes of this magnitude have preceded CPI surprises ~70% of the time",
            ]
        elif oil_collapsing:
            direction = "BULLISH"
            regime = "Oil Collapse — Disinflationary Relief"
            implications = [
                f"Oil -{abs(uso_mom_1m or 0):.1%} in 1M — CPI relief likely in 6-8 weeks",
                "Bonds rally likely as inflation fears recede",
                "Consumer spending power improving (lower gas prices)",
                "Rate-sensitive sectors (XLRE, XLU, QQQ) benefit",
                "Watch for secondary effects: XLE underperformance",
            ]
        elif oil_rising:
            direction = "WATCH"
            regime = "Oil Drifting Higher — Mild Inflation Pressure"
            implications = [
                f"Oil +{(uso_mom_1m or 0):.1%} 1M — modest inflation pressure building",
                "Monitor CPI expectations for upside surprise",
                "Overweight XLE relative to rate-sensitive sectors",
            ]
        elif oil_falling:
            direction = "WATCH"
            regime = "Oil Softening — Disinflationary Trend"
            implications = [
                f"Oil {(uso_mom_1m or 0):.1%} 1M — mild deflationary tailwind for bonds",
                "Rate cut probability may increase on next CPI print",
                "Bond duration can be extended modestly",
            ]
        else:
            direction = "NEUTRAL"
            regime = "Oil Stable — Inflation Neutral"
            implications = [
                "Oil rangebound — no strong CPI or rates signal from commodities"
            ]

        strength = min(95, abs(uso_mom_1m or 0) * 500 + abs(uso_z) * 10)
        reading = float(uso_mom_1m or 0) * 100

        return CrossAssetSignal(
            relationship="Oil → Inflation → Interest Rates",
            chain="USO (Oil) → CPI Expectations → TLT (Bonds) → Equity Multiples",
            assets=["USO", "GLD", "TLT", "XLE"],
            direction=direction,
            strength=round(strength, 1),
            confidence=72.0,
            current_reading=round(reading, 2),
            historical_percentile=round(self._pct(uso_mom_1m, -0.3, 0.3), 1),
            lead_lag_days=best_lag,
            description=(
                f"Oil chain analysis: '{regime}'. "
                f"Oil 1M: {(uso_mom_1m or 0):+.1%} | 3M: {(uso_mom_3m or 0):+.1%}. "
                f"Oil Z-score (52W): {uso_z:+.2f}. "
                f"Oil→Rates lead: ~{best_lag}d (corr {best_corr:.2f}). "
                f"Energy stocks (XLE) 1M: {(xle_mom_1m or 0):+.1%}. "
                f"Gold 1M: {(gld_mom_1m or 0):+.1%}."
            ),
            trade_implications=implications,
            regime=regime,
            metadata={
                "uso_mom_1m": uso_mom_1m,
                "uso_mom_3m": uso_mom_3m,
                "uso_zscore": uso_z,
                "xle_mom_1m": xle_mom_1m,
                "tlt_mom_1m": tlt_mom_1m,
                "lead_lag_days": best_lag,
                "lead_lag_corr": best_corr,
            },
        )

    # 
    # 3. USD STRENGTH ↔ COMMODITIES
    # 

    def analyze_usd_commodities_relationship(self) -> CrossAssetSignal:
        """
        Most commodities are priced in USD. A stronger dollar makes commodities
        more expensive in local currencies → demand falls → commodity prices fall.
        Also: strong USD → EM pressure → global growth headwind.
        """
        uup_mom_1m = self.data.get_momentum("UUP", 21)
        uup_mom_3m = self.data.get_momentum("UUP", 63)
        gld_mom_1m = self.data.get_momentum("GLD", 21)
        uso_mom_1m_usd = self.data.get_momentum("USO", 21)
        eem_mom_1m = self.data.get_momentum("EEM", 21)
        gsg_mom_1m = self.data.get_momentum("GSG", 21)

        uup_z = self.data.get_zscore("UUP", 252) or 0.0

        # Correlation USD vs commodities (expected: negative)
        corr_uup_gld = self.data.get_rolling_corr("UUP", "GLD", window=60)
        recent_corr_gld = (
            float(corr_uup_gld.iloc[-1])
            if corr_uup_gld is not None and len(corr_uup_gld) > 0
            else -0.5
        )

        usd_strong = (uup_mom_1m or 0) > 0.01
        usd_very_strong = (uup_mom_1m or 0) > 0.025

        # Correlation breakdown signal
        usd_gold_both_up = (uup_mom_1m or 0) > 0.01 and (gld_mom_1m or 0) > 0.01
        contradiction_flag = usd_gold_both_up

        if usd_very_strong:
            direction = "BEARISH"
            regime = "Strong USD — Commodity Headwind + EM Pressure"
            implications = [
                f"USD +{(uup_mom_1m or 0):.1%} 1M — broad commodity headwind (priced in USD)",
                f"Gold under pressure from USD strength: GLD {(gld_mom_1m or 0):+.1%}",
                f"EM equities facing FX + debt service headwinds: EEM {(eem_mom_1m or 0):+.1%}",
                "Underweight commodities, EM, international equities",
                "Favour domestically-focused US equities (XLU, XLP, small-cap financials)",
            ]
        elif usd_strong:
            direction = "WATCH"
            regime = "Mild USD Strength — Monitor Commodity Relationships"
            implications = [
                f"USD modestly firmer: {(uup_mom_1m or 0):+.1%} 1M",
                "Selective commodity headwind — watch gold and oil for divergence",
                "EM pressure building; monitor EEM for breakdown",
            ]
        else:
            direction = "BULLISH"
            regime = "USD Weakening — Commodity and EM Tailwind"
            implications = [
                f"USD weakening {(uup_mom_1m or 0):+.1%} 1M — tailwind for commodities and EM",
                f"Gold benefiting: GLD {(gld_mom_1m or 0):+.1%}",
                f"EM equities relieved: EEM {(eem_mom_1m or 0):+.1%}",
                "Overweight commodities, gold, EM equities, international DM",
                "Domestic-only USD assets may underperform",
            ]

        if contradiction_flag:
            implications.append(
                "[!] CONTRADICTION: USD and gold BOTH rising — signals potential loss of confidence "
                "in fiat broadly. Watch for accelerating dollar debasement narrative."
            )

        strength = min(95, abs(uup_mom_1m or 0) * 600 + abs(uup_z) * 12)
        reading = float(uup_mom_1m or 0) * 100

        return CrossAssetSignal(
            relationship="USD Strength ↔ Commodities",
            chain="UUP (Dollar) → Commodity Prices → EM Equities → Global Growth",
            assets=["UUP", "GLD", "USO", "EEM", "GSG"],
            direction=direction,
            strength=round(strength, 1),
            confidence=70.0,
            current_reading=round(reading, 2),
            historical_percentile=round(self._pct(uup_mom_1m, -0.05, 0.05), 1),
            lead_lag_days=5,
            description=(
                f"USD regime: '{regime}'. "
                f"USD 1M: {(uup_mom_1m or 0):+.1%} | 3M: {(uup_mom_3m or 0):+.1%} | Z: {uup_z:+.2f}. "
                f"Gold: {(gld_mom_1m or 0):+.1%} | Oil: {(uso_mom_1m_usd or 0):+.1%} | "
                f"Commodities: {(gsg_mom_1m or 0):+.1%} | EM: {(eem_mom_1m or 0):+.1%}. "
                f"USD-Gold corr (60d): {recent_corr_gld:.2f}. "
                f"{'[!] USD+Gold BOTH rising: fiat confidence signal.' if contradiction_flag else ''}"
            ),
            trade_implications=implications,
            regime=regime,
            metadata={
                "uup_mom_1m": uup_mom_1m,
                "uup_mom_3m": uup_mom_3m,
                "uup_z": uup_z,
                "gld_mom_1m": gld_mom_1m,
                "eem_mom_1m": eem_mom_1m,
                "usd_gold_contradiction": contradiction_flag,
                "usd_gold_corr_60d": recent_corr_gld,
            },
        )

    # 
    # 4. LIQUIDITY → RISK ASSETS → CRYPTO FLOWS
    # 

    def analyze_liquidity_crypto_flows(self) -> CrossAssetSignal:
        """
        Global liquidity is the primary driver of crypto and speculative assets.
        Liquidity proxy: HYG (credit spreads inverse), SPY vol, BTC trend.
        When liquidity contracts → crypto and speculative assets hit hardest.
        """
        hyg_mom_1m = self.data.get_momentum("HYG", 21)
        hyg_mom_3m = self.data.get_momentum("HYG", 63)
        btc_mom_1m = self.data.get_momentum("BTC-USD", 21)
        btc_mom_3m = self.data.get_momentum("BTC-USD", 63)
        spy_mom_1m = self.data.get_momentum("SPY", 21)
        arkk_mom_1m = self.data.get_momentum("ARKK", 21)

        hyg_z = self.data.get_zscore("HYG", 252) or 0.0
        btc_z = self.data.get_zscore("BTC-USD", 252) or 0.0

        # Liquidity score: HYG momentum + SPY vol inverse
        spy_vol = self.data.get_vol("SPY", 20) or 15.0
        liquidity_score = float(
            np.clip(
                50
                + (hyg_mom_1m or 0) * 500
                + (spy_mom_1m or 0) * 200
                - (spy_vol - 15) * 2,
                5,
                95,
            )
        )

        # BTC as leading liquidity indicator
        btc_leads_equities = self.data.compute_lead_lag("BTC-USD", "SPY", max_lag=10)
        btc_lead_days = btc_leads_equities.get("best_lag", 5)
        btc_lead_corr = btc_leads_equities.get("best_corr", 0.4)

        hyg_strong = (hyg_mom_1m or 0) > 0.01
        btc_strong = (btc_mom_1m or 0) > 0.10
        hyg_weak = (hyg_mom_1m or 0) < -0.01
        btc_weak = (btc_mom_1m or 0) < -0.10

        if hyg_strong and btc_strong:
            direction = "BULLISH"
            regime = "Abundant Liquidity — Risk Assets Supported"
            implications = [
                f"Credit (HYG +{(hyg_mom_1m or 0):.1%}) and crypto (BTC +{(btc_mom_1m or 0):.1%}) both strong",
                "Liquidity conditions favourable — risk assets broadly supported",
                "Overweight: crypto, high-yield, small-caps, growth equities",
                f"BTC historically leads equities by ~{btc_lead_days}d (corr {btc_lead_corr:.2f})",
            ]
        elif hyg_weak and btc_weak:
            direction = "BEARISH"
            regime = "Liquidity Contraction — Risk Assets Vulnerable"
            implications = [
                f"Credit (HYG {(hyg_mom_1m or 0):.1%}) and crypto (BTC {(btc_mom_1m or 0):.1%}) both weak",
                "Liquidity contracting — highest-risk assets hit first",
                "Reduce crypto, high-yield, speculative growth exposure",
                "Crypto drawdowns of 30-50% typical in liquidity crunch cycles",
                f"BTC warning: {btc_lead_days}-day lead suggests equity weakness may follow",
            ]
        elif hyg_strong and btc_weak:
            direction = "WATCH"
            regime = (
                "Crypto Weakness Despite Credit Strength — Crypto-Specific Headwind"
            )
            implications = [
                "Credit stable but crypto underperforming — regulatory or crypto-specific risk",
                "Crypto weakness not yet systemic (credit ok); isolated sector rotation",
                "Watch HYG for any deterioration as confirmation of broader risk-off",
            ]
        elif hyg_weak and btc_strong:
            direction = "WATCH"
            regime = "Crypto Strong Despite Credit Weakness — Divergence"
            implications = [
                "Unusual: BTC outperforming while credit weakens",
                "Could signal: (1) Bitcoin as safe-haven rotation, or (2) crypto speculative bubble",
                "If credit weakness persists, crypto likely to follow down with lag",
            ]
        else:
            direction = "NEUTRAL"
            regime = "Mixed Liquidity Signals"
            implications = [
                "No clear directional liquidity signal — monitor HYG and BTC for confirmation"
            ]

        strength = min(92, abs((hyg_mom_1m or 0) * 400) + abs((btc_mom_1m or 0) * 80))
        reading = float(liquidity_score)

        return CrossAssetSignal(
            relationship="Liquidity → Risk Assets → Crypto",
            chain="HYG (Credit) → Equity Risk Premium → BTC (Crypto Liquidity Barometer)",
            assets=["HYG", "BTC-USD", "SPY", "ARKK"],
            direction=direction,
            strength=round(strength, 1),
            confidence=65.0,
            current_reading=round(reading, 1),
            historical_percentile=round(liquidity_score, 1),
            lead_lag_days=btc_lead_days,
            description=(
                f"Liquidity regime: '{regime}'. Liquidity score: {liquidity_score:.0f}/100. "
                f"HYG 1M: {(hyg_mom_1m or 0):+.1%} | BTC 1M: {(btc_mom_1m or 0):+.1%} | "
                f"BTC 3M: {(btc_mom_3m or 0):+.1%}. SPY vol: {spy_vol:.1f}%. "
                f"BTC leads SPY by ~{btc_lead_days}d (corr {btc_lead_corr:.2f})."
            ),
            trade_implications=implications,
            regime=regime,
            metadata={
                "liquidity_score": liquidity_score,
                "hyg_mom_1m": hyg_mom_1m,
                "btc_mom_1m": btc_mom_1m,
                "btc_mom_3m": btc_mom_3m,
                "spy_vol_20d": spy_vol,
                "btc_lead_days": btc_lead_days,
                "btc_lead_corr": btc_lead_corr,
                "hyg_z": hyg_z,
                "btc_z": btc_z,
            },
        )

    # 
    # 5. YIELD CURVE REGIME
    # 

    def analyze_yield_curve(self) -> CrossAssetSignal:
        """
        Yield curve shape as leading indicator of economic regime.
        2s10s spread: SHY (2Y proxy) vs TLT (10Y proxy).
        Inverted curve historically precedes recession by 12-18 months.
        """
        tlt_mom_3m = self.data.get_momentum("TLT", 63)
        shy_mom_3m = self.data.get_momentum("SHY", 63)
        tlt_mom_1m = self.data.get_momentum("TLT", 21)
        shy_mom_1m = self.data.get_momentum("SHY", 21)

        tlt_z = self.data.get_zscore("TLT", 252) or 0.0
        shy_z = self.data.get_zscore("SHY", 252) or 0.0

        # Curve shape proxy: if TLT outperforms SHY → curve steepening (bullish growth)
        # If SHY outperforms TLT → curve inverting (bearish growth / recession risk)
        curve_momentum = (tlt_mom_3m or 0) - (shy_mom_3m or 0)
        curve_z_spread = tlt_z - shy_z

        if curve_momentum > 0.04:
            regime = "Curve Bull Steepening: Long end rallying faster"
            direction = "BULLISH"
            yield_curve_regime = "STEEPENING (BULL)"
            implications = [
                "Bull steepening: long bonds rallying (rate cut expectations rising)",
                "Historically follows early recession trough — recovery signal",
                "Favour: long-duration bonds, rate-sensitive equities (XLRE, XLU), gold",
                "Small-caps historically outperform in early steepening cycles",
            ]
        elif curve_momentum < -0.04:
            regime = "Curve Bear Flattening / Inversion: Short end rising faster"
            direction = "BEARISH"
            yield_curve_regime = "FLATTENING (BEAR)"
            implications = [
                "Bear flattening: short rates rising → curve inverting",
                "Historically strong recession predictor (12-18 month lead)",
                "Reduce cyclical equity exposure; add defensive positioning",
                "Credit spreads tend to widen 3-6 months after inversion",
                "Financials (XLF) hurt by flat/inverted curve (NIM compression)",
            ]
        elif curve_momentum > 0:
            regime = "Mild Steepening: Gradual normalisation"
            direction = "WATCH"
            yield_curve_regime = "STEEPENING (MILD)"
            implications = [
                "Mild steepening — economy normalising, no strong signal yet",
                "Monitor for acceleration into bull steepening (bullish) or reversal",
            ]
        else:
            regime = "Mild Flattening: Curve under mild inversion pressure"
            direction = "WATCH"
            yield_curve_regime = "FLATTENING (MILD)"
            implications = [
                "Curve gently flattening — watch for acceleration",
                "Avoid adding duration until curve signal clarifies",
            ]

        strength = min(90, abs(curve_momentum) * 600 + abs(curve_z_spread) * 12)

        return CrossAssetSignal(
            relationship="Yield Curve → Economic Regime → Sector Rotation",
            chain="2Y Yield (SHY) vs 10Y Yield (TLT) → Recession Probability → Asset Allocation",
            assets=["TLT", "SHY", "XLF", "IWM"],
            direction=direction,
            strength=round(strength, 1),
            confidence=74.0,
            current_reading=round(curve_momentum * 100, 2),
            historical_percentile=round(self._pct(curve_momentum, -0.15, 0.15), 1),
            lead_lag_days=365,
            description=(
                f"Yield curve regime: '{regime}'. "
                f"Curve momentum (TLT-SHY 3M delta): {curve_momentum:+.2%}. "
                f"TLT 3M: {(tlt_mom_3m or 0):+.1%} | SHY 3M: {(shy_mom_3m or 0):+.1%}. "
                f"TLT Z: {tlt_z:+.2f} | SHY Z: {shy_z:+.2f}."
            ),
            trade_implications=implications,
            regime=yield_curve_regime,
            metadata={
                "curve_momentum": curve_momentum,
                "curve_z_spread": curve_z_spread,
                "tlt_mom_3m": tlt_mom_3m,
                "shy_mom_3m": shy_mom_3m,
                "tlt_z": tlt_z,
                "shy_z": shy_z,
                "yield_curve_regime": yield_curve_regime,
            },
        )

    # 
    # 6. MACRO REGIME CLASSIFICATION
    # 

    def classify_macro_regime(self) -> MacroRegime:
        """
        Classify the current macro environment into one of five regimes
        based on growth, inflation, and liquidity signals.
        """
        # Growth proxy: SPY + IWM momentum
        spy_mom = self.data.get_momentum("SPY", 63) or 0.0
        iwm_mom = self.data.get_momentum("IWM", 63) or 0.0
        growth_signal = (spy_mom + iwm_mom) / 2

        # Inflation proxy: oil + gold + TIPS momentum
        uso_mom = self.data.get_momentum("USO", 63) or 0.0
        gld_mom = self.data.get_momentum("GLD", 63) or 0.0
        inflation_signal = (uso_mom + gld_mom) / 2

        # Liquidity proxy: HYG momentum
        hyg_mom = self.data.get_momentum("HYG", 63) or 0.0
        spy_vol = self.data.get_vol("SPY", 20) or 15.0
        liquidity_signal = hyg_mom - max(0, (spy_vol - 18) / 30)

        # Regime classification matrix
        growth_up = growth_signal > 0.02
        inflation_up = inflation_signal > 0.03
        liquidity_ok = liquidity_signal > -0.01

        probabilities: dict[str, float] = {}

        if growth_up and not inflation_up and liquidity_ok:
            probabilities["Risk-On / Goldilocks"] = 65.0
            probabilities["Inflationary Boom"] = 20.0
            probabilities["Risk-Off / Slow Growth"] = 10.0
            probabilities["Stagflation"] = 3.0
            probabilities["Deflationary Bust"] = 2.0
        elif growth_up and inflation_up:
            probabilities["Inflationary Boom"] = 55.0
            probabilities["Risk-On / Goldilocks"] = 25.0
            probabilities["Stagflation"] = 15.0
            probabilities["Risk-Off / Slow Growth"] = 3.0
            probabilities["Deflationary Bust"] = 2.0
        elif not growth_up and inflation_up:
            probabilities["Stagflation"] = 50.0
            probabilities["Risk-Off / Slow Growth"] = 25.0
            probabilities["Inflationary Boom"] = 15.0
            probabilities["Deflationary Bust"] = 8.0
            probabilities["Risk-On / Goldilocks"] = 2.0
        elif not growth_up and not inflation_up and not liquidity_ok:
            probabilities["Deflationary Bust"] = 50.0
            probabilities["Risk-Off / Slow Growth"] = 30.0
            probabilities["Stagflation"] = 10.0
            probabilities["Risk-On / Goldilocks"] = 8.0
            probabilities["Inflationary Boom"] = 2.0
        else:
            probabilities["Risk-Off / Slow Growth"] = 40.0
            probabilities["Risk-On / Goldilocks"] = 30.0
            probabilities["Deflationary Bust"] = 15.0
            probabilities["Stagflation"] = 10.0
            probabilities["Inflationary Boom"] = 5.0

        best_regime_name = max(probabilities, key=lambda k: probabilities[k])
        regime_template = MACRO_REGIMES[best_regime_name]

        result = MacroRegime(
            name=best_regime_name,
            probability=probabilities[best_regime_name],
            description=regime_template.description,
            favoured_assets=regime_template.favoured_assets,
            avoided_assets=regime_template.avoided_assets,
            historical_analogue=regime_template.historical_analogue,
            key_indicators=[
                f"Growth signal: {growth_signal:+.2%}",
                f"Inflation signal: {inflation_signal:+.2%}",
                f"Liquidity signal: {liquidity_signal:+.2%}",
                f"SPY vol (20d): {spy_vol:.1f}%",
            ]
            + regime_template.key_indicators,
            typical_duration_months=regime_template.typical_duration_months,
        )
        return result

    # 
    # 7. SECTOR ROTATION SIGNAL
    # 

    def get_sector_rotation_signal(self) -> dict:
        """
        Identify which sectors are in momentum leadership vs laggards.
        Returns ranked sector performance + rotation signal.
        """
        sectors = {
            "XLK": "Technology",
            "XLF": "Financials",
            "XLE": "Energy",
            "XLV": "Healthcare",
            "XLI": "Industrials",
            "XLY": "Consumer Disc.",
            "XLP": "Consumer Staples",
            "XLU": "Utilities",
            "XLRE": "Real Estate",
            "XLB": "Materials",
            "XLC": "Comm. Services",
        }
        results = []
        for ticker, name in sectors.items():
            mom_1m = self.data.get_momentum(ticker, 21)
            mom_3m = self.data.get_momentum(ticker, 63)
            z = self.data.get_zscore(ticker, 126)
            if mom_1m is not None:
                results.append(
                    {
                        "Sector": name,
                        "Ticker": ticker,
                        "1M %": round((mom_1m or 0) * 100, 2),
                        "3M %": round((mom_3m or 0) * 100, 2),
                        "Z-Score": round(z or 0, 2),
                        "Signal": "LEADER"
                        if (mom_1m or 0) > 0.03
                        else "LAGGARD"
                        if (mom_1m or 0) < -0.03
                        else "NEUTRAL",
                    }
                )
        results.sort(key=lambda x: x["1M %"], reverse=True)
        leaders = [r["Sector"] for r in results if r["Signal"] == "LEADER"][:3]
        laggards = [r["Sector"] for r in results if r["Signal"] == "LAGGARD"][:3]
        rotation_theme = "No clear rotation"
        if leaders and laggards:
            rotation_theme = (
                f"Into: {', '.join(leaders)} | Out of: {', '.join(laggards)}"
            )
        return {
            "table": results,
            "leaders": leaders,
            "laggards": laggards,
            "theme": rotation_theme,
        }

    # 
    # 8. FULL DASHBOARD
    # 

    def build_dashboard(self) -> CrossAssetDashboard:
        """
        Run all cross-asset analyses and return a unified dashboard snapshot.
        """
        signals: list[CrossAssetSignal] = []

        for fn in [
            self.analyze_bond_equity_relationship,
            self.analyze_oil_inflation_rates_chain,
            self.analyze_usd_commodities_relationship,
            self.analyze_liquidity_crypto_flows,
            self.analyze_yield_curve,
        ]:
            try:
                signals.append(fn())
            except Exception:
                pass

        try:
            macro_regime = self.classify_macro_regime()
        except Exception:
            macro_regime = MacroRegime(
                name="Unknown",
                probability=0,
                description="Data unavailable",
                favoured_assets=[],
                avoided_assets=[],
                historical_analogue="",
                key_indicators=[],
                typical_duration_months=0,
            )

        try:
            sector_data = self.get_sector_rotation_signal()
            sector_rotation_signal = sector_data.get("theme", "No data")
        except Exception:
            sector_rotation_signal = "No data"

        # Contagion risk: average of bearish signal strengths
        bearish = [s for s in signals if s.direction == "BEARISH"]
        contagion_risk = (
            float(np.mean([s.strength for s in bearish])) if bearish else 15.0
        )

        # Liquidity score from liquidity signal
        liq_sig = next((s for s in signals if "Liquidity" in s.relationship), None)
        liquidity_score = liq_sig.current_reading if liq_sig else 50.0

        # Yield curve regime
        yc_sig = next((s for s in signals if "Yield Curve" in s.relationship), None)
        yield_curve_regime = yc_sig.regime if yc_sig else "Unknown"

        # Dollar regime
        usd_sig = next((s for s in signals if "USD" in s.relationship), None)
        dollar_regime = usd_sig.regime[:40] if usd_sig else "Unknown"

        # Volatility regime
        spy_vol = self.data.get_vol("SPY", 20) or 15.0
        vix_close = self.data.get_close("^VIX", "1mo")
        vix_level = (
            float(vix_close.iloc[-1])
            if vix_close is not None and len(vix_close) > 0
            else spy_vol
        )
        if vix_level > 30:
            volatility_regime = f"HIGH VOL (VIX ~{vix_level:.0f})"
        elif vix_level > 20:
            volatility_regime = f"ELEVATED VOL (VIX ~{vix_level:.0f})"
        else:
            volatility_regime = f"LOW VOL (VIX ~{vix_level:.0f})"

        # Correlation breakdown risk
        corr_breakdown = (
            float(
                np.mean(
                    [
                        abs(s.current_reading)
                        for s in signals
                        if s.direction in ("BEARISH", "WATCH")
                    ]
                )
            )
            if signals
            else 20.0
        )

        # Summary
        bullish_count = sum(1 for s in signals if s.direction == "BULLISH")
        bearish_count = sum(1 for s in signals if s.direction == "BEARISH")
        if bullish_count > bearish_count:
            summary = f"Cross-asset signals lean BULLISH ({bullish_count} of {len(signals)}). Regime: {macro_regime.name}."
        elif bearish_count > bullish_count:
            summary = f"Cross-asset signals lean BEARISH ({bearish_count} of {len(signals)}). Regime: {macro_regime.name}. Caution advised."
        else:
            summary = f"Cross-asset signals MIXED. Regime: {macro_regime.name}. No strong directional conviction."

        return CrossAssetDashboard(
            timestamp=datetime.utcnow().isoformat(),
            signals=signals,
            macro_regime=macro_regime,
            contagion_risk=round(contagion_risk, 1),
            correlation_breakdown_risk=round(min(95, corr_breakdown * 2), 1),
            liquidity_score=round(float(np.clip(liquidity_score, 5, 95)), 1),
            yield_curve_regime=yield_curve_regime,
            dollar_regime=dollar_regime,
            volatility_regime=volatility_regime,
            sector_rotation_signal=sector_rotation_signal,
            summary=summary,
        )

    def get_all_signals(self) -> list[CrossAssetSignal]:
        """
        Generate all cross-asset signals from individual analysis methods.
        Returns a list of CrossAssetSignal objects for each relationship analyzed.
        """
        signals: list[CrossAssetSignal] = []
        
        # Run all cross-asset analyses
        analysis_methods = [
            self.analyze_bond_equity_relationship,
            self.analyze_oil_inflation_rates_chain,
            self.analyze_usd_commodities_relationship,
            self.analyze_liquidity_crypto_flows,
            self.analyze_yield_curve,
        ]
        
        for method in analysis_methods:
            try:
                signal = method()
                signals.append(signal)
            except Exception as e:
                # Skip any failing analyses but continue with others
                pass
        
        return signals

    # 
    # INSTITUTIONAL MODULES
    # 

    def analyze_intermarket_causality(self, asset_a: str, asset_b: str, max_lag: int = 30) -> dict:
        """
        Quantify the lead-lag relationship between two assets using cross-correlation.
        Institutional-grade causality tracking to identify which market is leading the turn.
        """
        try:
            df_a = self.data.get_close(asset_a, period="2y")
            df_b = self.data.get_close(asset_b, period="2y")
            
            if df_a is None or df_b is None or len(df_a) < 100 or len(df_b) < 100:
                return {"causality": "Neutral", "lag": 0, "confidence": 0}

            # Normalize and calculate returns
            rets_a = df_a.pct_change().dropna()
            rets_b = df_b.pct_change().dropna()
            
            common_idx = rets_a.index.intersection(rets_b.index)
            rets_a = rets_a.loc[common_idx]
            rets_b = rets_b.loc[common_idx]

            corrs = [rets_a.corr(rets_b.shift(lag)) for lag in range(-max_lag, max_lag + 1)]
            best_lag = np.argmax(np.abs(corrs)) - max_lag
            max_corr = corrs[np.argmax(np.abs(corrs))]

            return {
                "asset_a": asset_a,
                "asset_b": asset_b,
                "max_correlation": float(max_corr),
                "lag_days": int(best_lag),
                "causality_score": float(np.abs(max_corr) * (1 - abs(best_lag)/max_lag)),
                "is_a_leading": best_lag < 0
            }
        except:
            return {"causality": "Neutral", "lag": 0, "confidence": 0}

    def detect_regime_shift(self) -> dict:
        """
        Detects structural shifts in the macro regime using volatility and momentum differentials.
        """
        spy_vol = self.data.get_vol("SPY") or 15.0
        tlt_vol = self.data.get_vol("TLT") or 10.0
        
        # High volatility regime indicator
        vol_stress = (spy_vol / 15.0 + tlt_vol / 10.0) / 2.0
        
        # Leading indicator momentum (Copper/Gold ratio proxy)
        copper_mom = self.data.get_momentum("CPER", 63) or 0
        gold_mom = self.data.get_momentum("GLD", 63) or 0
        growth_indicator = copper_mom - gold_mom
        
        regime = "Equilibrium"
        if vol_stress > 1.8:
            regime = "High-Stress Volatility Transmission"
        elif growth_indicator > 0.05:
            regime = "Reflationary Expansion"
        elif growth_indicator < -0.05:
            regime = "Deflationary Contraction"
            
        return {
            "regime": regime,
            "vol_stress": float(vol_stress),
            "growth_indicator": float(growth_indicator),
            "timestamp": datetime.utcnow().isoformat()
        }


# 
# Singleton accessor
# 

_engine_instance: MacroCrossAssetEngine | None = None


def get_macro_engine() -> MacroCrossAssetEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = MacroCrossAssetEngine()
    return _engine_instance
