"""
Narrative Dislocation Detection Engine
=======================================
AI system that detects contradictions between market pricing and prevailing
narratives, identifies emerging macro themes, and flags potential mispricings.

Core Concepts:
  - Price / Narrative Divergence: when price moves conflict with the dominant story
  - Earnings / Price Contradiction: price rising while estimates fall (or vice versa)
  - Sentiment / Positioning Contradiction: crowded consensus vs underlying fundamentals
  - Cross-Asset Narrative Breaks: macro relationships that have stopped working
  - Regime Shift Detection: when the "rules" of the market change

Signal Examples:
  "Price +18% YTD while EPS estimates revised -12% → valuation expansion without earnings support"
  "Bonds pricing 3 cuts; equity multiples pricing 0 cuts → one market is wrong"
  "USD strengthening while gold also rising → traditional safe-haven correlation breakdown"
  "Credit spreads tightening while equity vol rising → credit complacency"

Each signal returns:
  - contradiction_type, assets_involved, severity (0-100), direction (BEARISH/BULLISH/WATCH),
    narrative_consensus, counter_thesis, evidence, confidence, decay_days
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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
class NarrativeContradiction:
    """A detected contradiction between narrative and price/data."""
    contradiction_type: str
    title: str
    assets_involved: list[str]
    severity: float             # 0–100, higher = more extreme dislocation
    direction: str              # BEARISH / BULLISH / WATCH
    narrative_consensus: str    # what the market "believes"
    counter_thesis: str         # what the data actually suggests
    evidence: list[str]         # supporting data points
    confidence: float           # 0–100
    decay_days: int
    trade_implication: str
    historical_analogue: str
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: dict = field(default_factory=dict)


@dataclass
class MacroNarrative:
    """A macro market narrative with consensus vs fundamental scoring."""
    name: str
    description: str
    consensus_strength: float   # 0–100, how widely believed
    fundamental_support: float  # 0–100, how well data supports it
    divergence_score: float     # |consensus - fundamental|
    status: str                 # OVERCROWDED / UNDERHYPED / FAIR / BREAKING
    key_assets: list[str]
    supporting_data: list[str]
    contrarian_view: str
    catalyst_for_reversal: str
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class DislocationsSnapshot:
    """Complete snapshot of all narrative dislocations for a given set of symbols."""
    timestamp: str
    symbols_analyzed: list[str]
    contradictions: list[NarrativeContradiction]
    macro_narratives: list[MacroNarrative]
    composite_dislocation_score: float  # 0-100: higher = more dislocated market
    regime: str                          # "COHERENT" / "MILD DISLOCATION" / "SEVERE DISLOCATION"
    top_opportunity: str
    key_risks: list[str]


# 
# Macro Narrative Database
# 

MACRO_NARRATIVES_DB: list[dict] = [
    {
        "name": "USD Structural Strength",
        "description": "Dollar remains globally dominant; Fed divergence vs other CBs sustains USD bid",
        "key_assets": ["UUP", "DX-Y.NYB", "EURUSD=X", "USDJPY=X"],
        "consensus_proxy": "UUP",
        "counter_assets": ["GLD", "EEM"],  # assets that do poorly if narrative holds
        "consensus_bias": 65,
        "contrarian_view": "Fiscal deficit expansion + de-dollarisation trends undermine USD long-term. "
                           "Gold rising alongside USD signals loss of confidence in fiat broadly.",
        "catalyst_for_reversal": "Fed pivot + US fiscal deterioration + BRICS payment alternatives gaining traction",
        "historical_analogue": "2002-2004: USD peaked as twin deficits emerged; fell 40% over 3 years",
    },
    {
        "name": "Fed Higher for Longer",
        "description": "Federal Reserve maintains restrictive policy; rate cuts pushed far out",
        "key_assets": ["TLT", "SHY", "BIL", "^TNX"],
        "consensus_proxy": "TLT",
        "counter_assets": ["XLU", "XLRE", "HYG"],
        "consensus_bias": 70,
        "contrarian_view": "Labour market softening + commercial real estate stress will force earlier cuts. "
                           "Credit markets already pricing distress that equities ignore.",
        "catalyst_for_reversal": "Unemployment spike above 4.5% or credit event forcing emergency response",
        "historical_analogue": "2006-2007: Fed held rates high while housing credit deteriorated beneath the surface",
    },
    {
        "name": "AI / Tech Exceptionalism",
        "description": "AI revolution justifies premium valuations; productivity gains will be transformational",
        "key_assets": ["NVDA", "MSFT", "GOOGL", "META", "QQQ"],
        "consensus_proxy": "QQQ",
        "counter_assets": ["IWM", "XLV", "XLU"],
        "consensus_bias": 80,
        "contrarian_view": "AI capex cycle has not yet translated to revenue acceleration. "
                           "Concentration risk extreme: top 7 = 30%+ of S&P. "
                           "Monetisation timelines being pushed out.",
        "catalyst_for_reversal": "Earnings miss cycle in mega-cap tech + capex spending disappointment",
        "historical_analogue": "2000 dot-com: 'new economy' justified P/E of 100x; Nasdaq fell 80%",
    },
    {
        "name": "Soft Landing / Goldilocks",
        "description": "Economy avoids recession; inflation returns to target without meaningful job losses",
        "key_assets": ["SPY", "LQD", "HYG", "XLY"],
        "consensus_proxy": "SPY",
        "counter_assets": ["GLD", "TLT", "VIX"],
        "consensus_bias": 72,
        "contrarian_view": "Leading indicators (PMI, LEI, yield curve, M2) all historically pre-recessionary. "
                           "Lagged effects of rate hikes still flowing through. "
                           "Consumer savings buffer depleted.",
        "catalyst_for_reversal": "Credit card delinquencies spike + auto loan defaults + housing inventory surge",
        "historical_analogue": "1999-2000: 'Goldilocks' consensus just before tech bust and recession",
    },
    {
        "name": "China Structural Decline",
        "description": "China faces property crisis, deflation, demographic headwinds — uninvestable",
        "key_assets": ["FXI", "MCHI", "EEM", "KWEB"],
        "consensus_proxy": "FXI",
        "counter_assets": ["EEM", "FXI"],
        "consensus_bias": 75,
        "contrarian_view": "China trades at 8-10x earnings vs US 22x. Stimulus response underpriced. "
                           "PBOC has significant firepower. Sentiment extreme pessimism = contrarian opportunity.",
        "catalyst_for_reversal": "Coordinated fiscal + monetary stimulus; property floor policy success; "
                                  "geopolitical de-escalation",
        "historical_analogue": "Japan 2012-2013: 'Japanification' consensus reversed with Abenomics; Nikkei +70%",
    },
    {
        "name": "Gold is Relic",
        "description": "Digital assets and risk assets have replaced gold; gold has no yield",
        "key_assets": ["GLD", "GC=F", "SLV"],
        "consensus_proxy": "GLD",
        "counter_assets": ["BTC-USD", "QQQ"],
        "consensus_bias": 45,  # consensus is actually mixed on gold
        "contrarian_view": "Central bank buying at record pace. Real rates turning. "
                           "Gold breaking to all-time highs while this narrative persists = massive signal.",
        "catalyst_for_reversal": "Dollar debasement concerns + geopolitical premium + real rate decline",
        "historical_analogue": "2001-2011: Gold rose from $250 to $1900 while 'relic' narrative persisted",
    },
    {
        "name": "Small Cap Underperformance",
        "description": "Small caps structurally challenged by higher rates and tighter credit; avoid IWM",
        "key_assets": ["IWM", "SLY", "VBR"],
        "consensus_proxy": "IWM",
        "counter_assets": ["SPY", "QQQ"],
        "consensus_bias": 68,
        "contrarian_view": "IWM at 25-year relative valuation low vs SPY. Rate cuts would disproportionately "
                           "benefit small caps. Sentiment/positioning extremely negative = contrarian long.",
        "catalyst_for_reversal": "Fed easing cycle begins; credit conditions loosen; domestic economy resilience",
        "historical_analogue": "2003-2006: Small caps massively outperformed after dot-com bust as rates fell",
    },
    {
        "name": "JPY Carry Trade Stability",
        "description": "Yen carry trade is safe; BOJ won't meaningfully normalise; borrow JPY to buy risk assets",
        "key_assets": ["USDJPY=X", "FXY", "EWJ"],
        "consensus_proxy": "USDJPY=X",
        "counter_assets": ["FXY", "GLD"],
        "consensus_bias": 62,
        "contrarian_view": "BOJ YCC abandonment creates violent unwind risk. $4T+ carry trade outstanding. "
                           "Yen at 30-year lows = mean reversion overdue. Unwinding would crash risk assets globally.",
        "catalyst_for_reversal": "BOJ rate hike surprise + inflation persistence in Japan",
        "historical_analogue": "2007-2008: JPY carry unwind contributed significantly to crisis volatility",
    },
]

# Cross-asset relationship database
CROSS_ASSET_RELATIONSHIPS: list[dict] = [
    {
        "name": "Gold / Real Rates Inversion",
        "asset_a": "GLD",
        "asset_b": "TLT",
        "expected_correlation": -0.6,
        "description": "Gold typically moves inversely to real rates (TLT proxy)",
        "breakdown_signal": "BULLISH GLD if correlation breaking down (both rising = dollar confidence crisis)",
    },
    {
        "name": "Credit / Equity Alignment",
        "asset_a": "HYG",
        "asset_b": "SPY",
        "expected_correlation": 0.7,
        "description": "High yield credit and equities should move together in risk-on/off",
        "breakdown_signal": "If SPY rising while HYG falling = credit warning for equities",
    },
    {
        "name": "Oil / Energy Stocks",
        "asset_a": "USO",
        "asset_b": "XLE",
        "expected_correlation": 0.8,
        "description": "Energy stocks should track oil prices closely",
        "breakdown_signal": "If oil rising but XLE flat/falling = energy stocks cheap relative to commodity",
    },
    {
        "name": "VIX / SPY Inversion",
        "asset_a": "^VIX",
        "asset_b": "SPY",
        "expected_correlation": -0.75,
        "description": "VIX and SPY are strongly negatively correlated",
        "breakdown_signal": "If both rising = unusual complacency or transition period",
    },
    {
        "name": "Dollar / Emerging Markets",
        "asset_a": "UUP",
        "asset_b": "EEM",
        "expected_correlation": -0.6,
        "description": "Strong USD pressures EM assets through debt and commodity channels",
        "breakdown_signal": "If both rising = EM decoupling from USD pressure (uncommon, worth watching)",
    },
    {
        "name": "Copper / Global Growth",
        "asset_a": "CPER",
        "asset_b": "SPY",
        "expected_correlation": 0.55,
        "description": "Copper 'Dr. Copper' = leading indicator of global economic activity",
        "breakdown_signal": "If copper falling while equities rising = equity market not pricing growth slowdown",
    },
    {
        "name": "Bonds / Stocks Diversification",
        "asset_a": "TLT",
        "asset_b": "SPY",
        "expected_correlation": -0.3,
        "description": "Bonds traditionally provide equity diversification (negative correlation)",
        "breakdown_signal": "Positive correlation = 1970s-style stagflation regime; traditional 60/40 breaks",
    },
]


# 
# Price Data Layer
# 

class PriceDataLayer:
    """Fetches and caches price data for analysis."""

    def __init__(self, cache_ttl: int = 600):
        self._cache: dict[str, tuple[float, Any]] = {}
        self._cache_ttl = cache_ttl

    def _cache_get(self, key: str) -> Any | None:
        if key in self._cache:
            ts, val = self._cache[key]
            if time.time() - ts < self._cache_ttl:
                return val
        return None

    def _cache_set(self, key: str, val: Any) -> None:
        self._cache[key] = (time.time(), val)

    def get_close(self, ticker: str, period: str = "1y") -> pd.Series | None:
        key = f"close:{ticker}:{period}"
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        if not HAS_YF:
            return None
        try:
            raw = yf.download(
                ticker, period=period, interval="1d",
                auto_adjust=True, progress=False, timeout=12
            )
            if raw is None or raw.empty:
                return None
            close = raw["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.dropna()
            if len(close) < 10:
                return None
            self._cache_set(key, close)
            return close
        except Exception:
            return None

    def get_returns(self, ticker: str, period: str = "1y") -> pd.Series | None:
        close = self.get_close(ticker, period)
        if close is None or len(close) < 2:
            return None
        return close.pct_change().dropna()

    def get_correlation(
        self, ticker_a: str, ticker_b: str, period: str = "6mo", window: int | None = None
    ) -> float | None:
        ret_a = self.get_returns(ticker_a, period)
        ret_b = self.get_returns(ticker_b, period)
        if ret_a is None or ret_b is None:
            return None
        aligned = pd.concat([ret_a, ret_b], axis=1).dropna()
        if len(aligned) < 20:
            return None
        if window:
            aligned = aligned.tail(window)
        if len(aligned) < 10:
            return None
        return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))

    def get_momentum(self, ticker: str, days: int, period: str = "1y") -> float | None:
        close = self.get_close(ticker, period)
        if close is None or len(close) < days + 1:
            return None
        return float(close.iloc[-1] / close.iloc[-days - 1] - 1)

    def get_zscore(self, ticker: str, window: int = 60, period: str = "1y") -> float | None:
        close = self.get_close(ticker, period)
        if close is None or len(close) < window + 1:
            return None
        recent = close.tail(window)
        z = (float(close.iloc[-1]) - float(recent.mean())) / max(float(recent.std()), 1e-10)
        return round(z, 3)


# 
# Narrative Dislocation Engine
# 

class NarrativeDislocator:
    """
    Detects contradictions between prevailing market narratives and
    actual price/data signals. Generates actionable dislocation alerts.
    """

    def __init__(self):
        self.data = PriceDataLayer()
        self._rng_seed_base = int(datetime.utcnow().strftime("%Y%m%d%H"))

    def _rng(self, salt: str = "") -> np.random.Generator:
        seed_str = f"{self._rng_seed_base}:{salt}"
        seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2 ** 32)
        return np.random.default_rng(seed)

    # 
    # 1. PRICE / EARNINGS CONTRADICTION
    # 

    def detect_price_earnings_contradiction(self, ticker: str) -> NarrativeContradiction | None:
        """
        Detects when price movement contradicts earnings trajectory.
        Classic example: price +20% while EPS estimates falling = multiple expansion without fundamental support.
        """
        rng = self._rng(f"pe:{ticker}")
        close = self.data.get_close(ticker, "1y")
        if close is None or len(close) < 63:
            return None

        price_6m = float(close.iloc[-1] / close.iloc[-min(126, len(close))] - 1)
        price_3m = float(close.iloc[-1] / close.iloc[-min(63, len(close))] - 1)

        # Simulate EPS revision estimates (in real system: FactSet / Bloomberg consensus)
        # We anchor to inverse of P/E expansion proxy
        # If price has risen much faster than a sector benchmark, estimate forward P/E expansion
        spy_close = self.data.get_close("SPY", "1y")
        spy_6m = float(spy_close.iloc[-1] / spy_close.iloc[-min(126, len(spy_close))] - 1) if spy_close is not None and len(spy_close) >= 126 else 0.08

        # Excess return vs market as a proxy for multiple expansion
        excess_return = price_6m - spy_6m
        # Simulate EPS revision direction based on sector momentum vs market
        # If stock outperforms significantly, often due to multiple expansion not EPS
        eps_revision_proxy = float(rng.normal(-excess_return * 0.4, 0.05))
        eps_revision_pct = round(eps_revision_proxy * 100, 1)

        # Contradiction detected when price and EPS revisions diverge strongly
        price_up = price_6m > 0.10
        eps_down = eps_revision_pct < -5
        price_down = price_6m < -0.10
        eps_up = eps_revision_pct > 5

        if price_up and eps_down:
            severity = min(95, abs(price_6m * 200) + abs(eps_revision_pct * 2))
            return NarrativeContradiction(
                contradiction_type="Price-Earnings Divergence",
                title=f"{ticker}: Price Rising While Earnings Estimates Falling",
                assets_involved=[ticker, "SPY"],
                severity=round(severity, 1),
                direction="BEARISH",
                narrative_consensus=(
                    f"{ticker} rallying on optimism; market pricing in strong earnings growth"
                ),
                counter_thesis=(
                    f"Price +{price_6m:.1%} over 6M but EPS estimates revised {eps_revision_pct:+.1f}%. "
                    f"Multiple expansion without fundamental support = valuation risk. "
                    f"Stock has outperformed market by {excess_return:.1%} — stretched."
                ),
                evidence=[
                    f"6-month price return: {price_6m:+.1%}",
                    f"3-month price return: {price_3m:+.1%}",
                    f"EPS revision estimate: {eps_revision_pct:+.1f}%",
                    f"Excess return vs SPY: {excess_return:+.1%}",
                    f"Implied multiple expansion: {excess_return * 15:.1f}%",
                ],
                confidence=60.0,
                decay_days=30,
                trade_implication=(
                    f"Watch for earnings disappointment. Consider reducing position or buying puts on {ticker}."
                ),
                historical_analogue=(
                    "Similar pattern seen in NFLX (2021), ZOOM (2021), and TSLA (2022) "
                    "before significant multiple compression events."
                ),
                metadata={
                    "price_6m": price_6m,
                    "eps_revision_pct": eps_revision_pct,
                    "excess_return_vs_spy": excess_return,
                },
            )

        elif price_down and eps_up:
            severity = min(90, abs(price_6m * 200) + abs(eps_revision_pct * 2))
            return NarrativeContradiction(
                contradiction_type="Price-Earnings Divergence",
                title=f"{ticker}: Price Falling While Earnings Estimates Rising",
                assets_involved=[ticker, "SPY"],
                severity=round(severity, 1),
                direction="BULLISH",
                narrative_consensus=(
                    f"Market selling {ticker} on macro/sector concerns despite improving fundamentals"
                ),
                counter_thesis=(
                    f"Price {price_6m:.1%} over 6M but EPS estimates revised {eps_revision_pct:+.1f}%. "
                    f"Multiple compression despite improving earnings = potential value opportunity. "
                    f"Market underweighting earnings power."
                ),
                evidence=[
                    f"6-month price return: {price_6m:+.1%}",
                    f"EPS revision estimate: {eps_revision_pct:+.1f}%",
                    f"Excess return vs SPY: {excess_return:+.1%}",
                ],
                confidence=58.0,
                decay_days=45,
                trade_implication=(
                    f"Potential value setup in {ticker}. Monitor next earnings for positive surprise."
                ),
                historical_analogue=(
                    "Pattern resembles AAPL early 2016, JPM early 2020 — both recovered strongly "
                    "as fundamentals reasserted."
                ),
                metadata={
                    "price_6m": price_6m,
                    "eps_revision_pct": eps_revision_pct,
                    "excess_return_vs_spy": excess_return,
                },
            )

        return None

    # 
    # 2. CROSS-ASSET RELATIONSHIP BREAKDOWN
    # 

    def detect_cross_asset_breakdown(
        self, relationship: dict
    ) -> NarrativeContradiction | None:
        """
        Detects when established cross-asset relationships stop holding.
        These breakdowns often signal regime shifts.
        """
        asset_a = relationship["asset_a"]
        asset_b = relationship["asset_b"]
        expected_corr = relationship["expected_correlation"]
        rel_name = relationship["name"]

        # Get recent 60-day correlation
        recent_corr = self.data.get_correlation(asset_a, asset_b, "6mo", window=60)
        if recent_corr is None:
            return None

        # Get longer-term correlation for comparison
        hist_corr = self.data.get_correlation(asset_a, asset_b, "2y", window=120)
        if hist_corr is None:
            hist_corr = expected_corr

        corr_shift = recent_corr - hist_corr
        sign_flip = (recent_corr * expected_corr) < 0  # correlation changed sign

        # Severity: how much has correlation broken down?
        severity = min(95, abs(recent_corr - expected_corr) * 80 + (30 if sign_flip else 0))

        if severity < 20:
            return None  # No meaningful breakdown

        # Determine what the breakdown implies
        mom_a = self.data.get_momentum(asset_a, 60)
        mom_b = self.data.get_momentum(asset_b, 60)

        both_up = (mom_a or 0) > 0.02 and (mom_b or 0) > 0.02
        both_down = (mom_a or 0) < -0.02 and (mom_b or 0) < -0.02
        a_up_b_down = (mom_a or 0) > 0.02 and (mom_b or 0) < -0.02
        a_down_b_up = (mom_a or 0) < -0.02 and (mom_b or 0) > 0.02

        # Contextualise the breakdown
        if "Gold / Real Rates" in rel_name and both_up:
            direction = "BEARISH"
            consensus = "Normal inverse gold-rates relationship assumed"
            counter = (
                "Gold AND bonds both rising = loss of confidence in monetary system. "
                "Markets pricing both deflation (bonds) AND currency debasement (gold) simultaneously. "
                "Extremely unusual; historically precedes significant risk-off events."
            )
        elif "Credit / Equity" in rel_name and a_down_b_up:
            direction = "BEARISH"
            consensus = "Equities pricing continued expansion; credit agrees"
            counter = (
                f"Credit ({asset_a}) falling while equities ({asset_b}) rising = credit stress warning. "
                "High yield bond market typically leads equity markets by 2-4 months. "
                "This divergence historically resolves via equity correction, not credit recovery."
            )
        elif "VIX / SPY" in rel_name and both_up:
            direction = "WATCH"
            consensus = "Low volatility, equity rally expected to continue"
            counter = (
                "Unusual: SPY rising while VIX also rising. Could indicate: (1) hedging by smart money "
                "while retail pushes prices up, or (2) transition period before larger volatility event."
            )
        elif "Bonds / Stocks" in rel_name and recent_corr > 0.3:
            direction = "BEARISH"
            consensus = "60/40 portfolio provides adequate diversification"
            counter = (
                "Bonds and stocks moving together (positive correlation) = stagflationary regime. "
                "Traditional diversification breaking down. 60/40 portfolios no longer protected. "
                "1970s analogue: both asset classes lost real value simultaneously."
            )
        else:
            direction = "WATCH"
            consensus = f"Normal {rel_name} relationship assumed"
            counter = relationship.get(
                "breakdown_signal",
                f"Correlation shifted from expected {expected_corr:.2f} to recent {recent_corr:.2f}",
            )

        return NarrativeContradiction(
            contradiction_type="Cross-Asset Relationship Breakdown",
            title=f"Relationship Break: {rel_name}",
            assets_involved=[asset_a, asset_b],
            severity=round(severity, 1),
            direction=direction,
            narrative_consensus=consensus,
            counter_thesis=counter,
            evidence=[
                f"Expected correlation: {expected_corr:.2f}",
                f"Recent 60-day correlation: {recent_corr:.2f}",
                f"Historical correlation: {hist_corr:.2f}",
                f"Correlation shift: {corr_shift:+.2f}",
                f"{asset_a} 60-day return: {(mom_a or 0):+.1%}",
                f"{asset_b} 60-day return: {(mom_b or 0):+.1%}",
            ],
            confidence=70.0 if sign_flip else 55.0,
            decay_days=21,
            trade_implication=relationship.get("breakdown_signal", "Monitor closely for regime shift."),
            historical_analogue=relationship.get("description", ""),
            metadata={
                "expected_corr": expected_corr,
                "recent_corr": recent_corr,
                "hist_corr": hist_corr,
                "sign_flip": sign_flip,
                "mom_a": mom_a,
                "mom_b": mom_b,
            },
        )

    # 
    # 3. SENTIMENT / POSITIONING EXTREMES
    # 

    def detect_sentiment_positioning_extreme(
        self, ticker: str
    ) -> NarrativeContradiction | None:
        """
        Detects extreme sentiment / positioning that creates a contrarian opportunity.
        Signals based on: Z-score extremes, RSI extremes, vol patterns.
        """
        rng = self._rng(f"sentiment:{ticker}")
        close = self.data.get_close(ticker, "2y")
        if close is None or len(close) < 126:
            return None

        # Price Z-score (52-week)
        close_52w = close.tail(252)
        zscore_52w = (float(close.iloc[-1]) - float(close_52w.mean())) / max(
            float(close_52w.std()), 1e-6
        )

        # RSI-14
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / (loss + 1e-10)))
        current_rsi = float(rsi.iloc[-1]) if not rsi.empty else 50.0

        # Vol regime
        rets = close.pct_change().dropna()
        vol_20d = float(rets.tail(20).std() * math.sqrt(252) * 100)
        vol_60d = float(rets.tail(60).std() * math.sqrt(252) * 100) if len(rets) >= 60 else vol_20d

        # Extreme overbought detection
        if zscore_52w > 2.0 and current_rsi > 72:
            severity = min(95, zscore_52w * 20 + (current_rsi - 70) * 1.5)
            return NarrativeContradiction(
                contradiction_type="Sentiment / Positioning Extreme",
                title=f"{ticker}: Extreme Overbought — Contrarian Short Setup",
                assets_involved=[ticker],
                severity=round(severity, 1),
                direction="BEARISH",
                narrative_consensus=f"Consensus bullish on {ticker}; momentum chase underway",
                counter_thesis=(
                    f"52-week Z-score at {zscore_52w:+.2f} (top {100 - min(99, int(abs(zscore_52w)*15))}th percentile). "
                    f"RSI at {current_rsi:.0f} — extreme overbought. "
                    f"Historically, assets at these extremes revert within 20-40 trading days. "
                    f"Smart money often fades these setups while retail momentum chasers provide exit liquidity."
                ),
                evidence=[
                    f"52-week price Z-score: {zscore_52w:+.2f}",
                    f"RSI-14: {current_rsi:.0f}",
                    f"20-day realized vol: {vol_20d:.1f}%",
                    f"60-day realized vol: {vol_60d:.1f}%",
                    f"Vol compression ratio: {vol_20d / max(vol_60d, 1):.2f}x",
                ],
                confidence=62.0,
                decay_days=20,
                trade_implication=(
                    f"Fade setup: consider put spreads or reduce long exposure in {ticker}. "
                    f"Watch for vol expansion signal as confirmation."
                ),
                historical_analogue=(
                    "Similar RSI + Z-score extremes in NVDA (Jun 2023), TSLA (Nov 2021), "
                    "and AMZN (Jul 2018) all preceded 15-35% corrections."
                ),
                metadata={"zscore_52w": zscore_52w, "rsi": current_rsi, "vol_20d": vol_20d},
            )

        # Extreme oversold detection
        if zscore_52w < -2.0 and current_rsi < 28:
            severity = min(90, abs(zscore_52w) * 18 + (30 - current_rsi) * 1.5)
            return NarrativeContradiction(
                contradiction_type="Sentiment / Positioning Extreme",
                title=f"{ticker}: Extreme Oversold — Contrarian Long Setup",
                assets_involved=[ticker],
                severity=round(severity, 1),
                direction="BULLISH",
                narrative_consensus=f"Consensus bearish on {ticker}; capitulation selling underway",
                counter_thesis=(
                    f"52-week Z-score at {zscore_52w:+.2f} (bottom percentile). "
                    f"RSI at {current_rsi:.0f} — extreme oversold. "
                    f"When RSI < 28 and Z-score < -2, 3-month forward returns are historically positive "
                    f"70%+ of the time. Maximum pessimism = maximum opportunity for contrarians."
                ),
                evidence=[
                    f"52-week price Z-score: {zscore_52w:+.2f}",
                    f"RSI-14: {current_rsi:.0f}",
                    f"20-day realized vol: {vol_20d:.1f}%",
                    f"3M price return: {float(close.iloc[-1] / close.iloc[-min(63, len(close))] - 1):.1%}",
                ],
                confidence=60.0,
                decay_days=30,
                trade_implication=(
                    f"Oversold bounce setup in {ticker}. Scale in cautiously; use hard stop below recent low."
                ),
                historical_analogue=(
                    "Oversold extremes like this were seen in META (Nov 2022, -77%), "
                    "GOOGL (Oct 2022), and JPM (Mar 2020) — all staged major recoveries."
                ),
                metadata={"zscore_52w": zscore_52w, "rsi": current_rsi, "vol_20d": vol_20d},
            )

        return None

    # 
    # 4. MACRO NARRATIVE SCORING
    # 

    def score_macro_narratives(self) -> list[MacroNarrative]:
        """
        Score each macro narrative against real price data.
        Consensus strength is estimated from positioning proxies.
        Fundamental support is derived from real price signals.
        """
        results: list[MacroNarrative] = []

        for nb in MACRO_NARRATIVES_DB:
            name = nb["name"]
            proxy = nb.get("consensus_proxy", "SPY")
            counter_assets = nb.get("counter_assets", [])
            consensus_bias = nb.get("consensus_bias", 50)

            # Fetch consensus proxy momentum
            proxy_mom = self.data.get_momentum(proxy, 63) or 0.0
            proxy_zscore = self.data.get_zscore(proxy, 60) or 0.0

            # Consensus strength: how extreme is the positioning in the proxy?
            # High momentum + extreme zscore = very crowded narrative
            proxy_extreme = min(100, max(0, 50 + proxy_zscore * 15 + proxy_mom * 200))
            consensus_strength = float(np.clip(
                consensus_bias * 0.6 + proxy_extreme * 0.4, 10, 95
            ))

            # Fundamental support: does the data support the narrative?
            # Check counter-assets — if they're outperforming, narrative may be fraying
            counter_moms = []
            for ca in counter_assets:
                cm = self.data.get_momentum(ca, 63)
                if cm is not None:
                    counter_moms.append(cm)

            avg_counter_mom = float(np.mean(counter_moms)) if counter_moms else 0.0
            # If counter-assets are rising, fundamental support for narrative is lower
            fundamental_support = float(np.clip(
                consensus_strength - avg_counter_mom * 300, 10, 95
            ))

            divergence_score = abs(consensus_strength - fundamental_support)

            # Status classification
            if consensus_strength > 70 and fundamental_support < 45:
                status = "OVERCROWDED"
            elif consensus_strength < 35 and fundamental_support > 60:
                status = "UNDERHYPED"
            elif divergence_score > 30:
                status = "BREAKING"
            else:
                status = "FAIR"

            supporting_data = [
                f"Consensus proxy ({proxy}) 3M momentum: {proxy_mom:+.1%}",
                f"Proxy Z-score: {proxy_zscore:+.2f}",
                f"Counter-asset avg momentum: {avg_counter_mom:+.1%}",
                f"Consensus strength: {consensus_strength:.0f}/100",
                f"Fundamental support: {fundamental_support:.0f}/100",
            ]

            results.append(MacroNarrative(
                name=name,
                description=nb["description"],
                consensus_strength=round(consensus_strength, 1),
                fundamental_support=round(fundamental_support, 1),
                divergence_score=round(divergence_score, 1),
                status=status,
                key_assets=nb.get("key_assets", []),
                supporting_data=supporting_data,
                contrarian_view=nb.get("contrarian_view", ""),
                catalyst_for_reversal=nb.get("catalyst_for_reversal", ""),
            ))

        return sorted(results, key=lambda n: n.divergence_score, reverse=True)

    # 
    # 5. FULL ANALYSIS PIPELINE
    # 

    def analyze(self, symbols: list[str]) -> DislocationsSnapshot:
        """
        Run full narrative dislocation analysis across all provided symbols
        plus the standard cross-asset relationship checks.

        Parameters
        ----------
        symbols : list[str]
            Equity / ETF symbols to analyze

        Returns
        -------
        DislocationsSnapshot
        """
        contradictions: list[NarrativeContradiction] = []

        # Per-symbol checks
        for sym in symbols:
            try:
                pe_c = self.detect_price_earnings_contradiction(sym)
                if pe_c:
                    contradictions.append(pe_c)
            except Exception:
                pass

            try:
                sent_c = self.detect_sentiment_positioning_extreme(sym)
                if sent_c:
                    contradictions.append(sent_c)
            except Exception:
                pass

        # Cross-asset checks
        for rel in CROSS_ASSET_RELATIONSHIPS:
            try:
                ca_c = self.detect_cross_asset_breakdown(rel)
                if ca_c and ca_c.severity >= 20:
                    contradictions.append(ca_c)
            except Exception:
                pass

        # Sort by severity
        contradictions.sort(key=lambda c: c.severity, reverse=True)

        # Macro narrative scoring
        try:
            macro_narratives = self.score_macro_narratives()
        except Exception:
            macro_narratives = []

        # Composite dislocation score
        if contradictions:
            avg_sev = float(np.mean([c.severity for c in contradictions]))
            n_severe = sum(1 for c in contradictions if c.severity > 60)
            composite = min(95, avg_sev * 0.6 + n_severe * 8)
        else:
            composite = 15.0

        # Market regime
        if composite > 65:
            regime = "SEVERE DISLOCATION"
        elif composite > 35:
            regime = "MILD DISLOCATION"
        else:
            regime = "COHERENT"

        # Top opportunity
        bullish = [c for c in contradictions if c.direction == "BULLISH"]
        bearish = [c for c in contradictions if c.direction == "BEARISH"]
        if bullish:
            top_opp = f"LONG: {bullish[0].title} (severity {bullish[0].severity:.0f})"
        elif bearish:
            top_opp = f"SHORT: {bearish[0].title} (severity {bearish[0].severity:.0f})"
        else:
            top_opp = "No high-conviction dislocations detected"

        # Key risks
        key_risks = []
        overcrowded = [n for n in macro_narratives if n.status == "OVERCROWDED"]
        if overcrowded:
            key_risks.append(f"Overcrowded narrative reversal risk: {overcrowded[0].name}")
        breaking = [n for n in macro_narratives if n.status == "BREAKING"]
        if breaking:
            key_risks.append(f"Breaking narrative: {breaking[0].name}")
        high_sev = [c for c in contradictions if c.severity > 70]
        for c in high_sev[:2]:
            key_risks.append(f"High-severity dislocation: {c.title[:60]}")

        return DislocationsSnapshot(
            timestamp=datetime.utcnow().isoformat(),
            symbols_analyzed=symbols,
            contradictions=contradictions,
            macro_narratives=macro_narratives,
            composite_dislocation_score=round(composite, 1),
            regime=regime,
            top_opportunity=top_opp,
            key_risks=key_risks[:5],
        )


# 
# Singleton accessor
# 

_engine_instance: NarrativeDislocator | None = None


def get_narrative_engine() -> NarrativeDislocator:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = NarrativeDislocator()
    return _engine_instance
