"""
Factor Crowding Engine
=======================
Detects crowded trades, factor concentration risk, and hedge fund holdings
overlap. Used by quant funds to avoid entering positions already saturated
with institutional capital — where the exit risk is highest.

Core Capabilities:
  1. Factor Crowding Score — how saturated each factor is with institutional money
  2. Crowded Trade Detection — stocks / ETFs with dangerously high overlap
  3. Factor Decay Analysis — measuring how quickly alpha is eroding
  4. Hedge Fund Holdings Overlap — simulated 13-F overlap scoring
  5. Factor Momentum vs Crowding Tradeoff — signal strength adjusted for crowd risk
  6. Unwind Risk Assessment — probability and severity of crowding unwind

Based on academic research:
  - Khandani & Lo (2007): "What Happened to the Quants in August 2007?"
  - McLean & Pontiff (2016): "Does Publishing Research Destroy Stock Return Predictability?"
  - Haddad, Kozak & Santosh (2020): "Factor Timing"
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import time
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
class FactorCrowdingScore:
    factor_name: str
    crowding_score: float  # 0-100: higher = more crowded
    capacity_remaining: float  # 0-100: % of capacity still available
    alpha_decay_rate: float  # % per year alpha is eroding
    estimated_unwind_impact: float  # bps of price impact if unwound
    signal_haircut: float  # % to reduce raw factor signal by (crowding adj)
    status: str  # OVERCROWDED / CROWDED / NORMAL / UNDERCROWDED
    description: str
    evidence: list[str]
    risk_adjusted_signal: float  # -1 to +1 net signal after crowding adjustment


@dataclass
class CrowdedTrade:
    ticker: str
    factor_exposures: dict[str, float]  # factor -> exposure score
    hf_overlap_score: float  # 0-100: % of top 100 HFs holding
    crowding_percentile: float  # 0-100
    unwind_risk: str  # LOW / MEDIUM / HIGH / EXTREME
    expected_drawdown_on_unwind: float  # % loss if crowding reverses
    momentum_score: float  # raw momentum before crowding adj
    crowding_adjusted_signal: float  # signal after crowding penalty
    description: str
    avoid: bool  # True if too crowded to enter


@dataclass
class FactorDecayAnalysis:
    factor_name: str
    factor_returns_1y: float  # IS return in last 12 months
    factor_returns_3y: float  # IS return in last 36 months
    half_life_years: float  # estimated years until alpha halves
    publication_decay: bool  # True if alpha decreased after published
    crowding_decay: bool  # True if alpha decreasing due to crowding
    current_alpha_estimate: float  # estimated current annual alpha (%)
    original_alpha_estimate: float  # alpha at discovery / publication (%)
    decay_explanation: str


@dataclass
class HedgeFundOverlapReport:
    ticker: str
    estimated_hf_holders: int
    estimated_hf_ownership_pct: float  # % of float held by hedge funds
    overlap_clusters: list[str]
    overlap_score: float
    note: str


@dataclass
class CrowdingDashboard:
    generated_at: str
    factor_scores: list[FactorCrowdingScore]
    crowded_trades: list[CrowdedTrade]
    decay_analysis: list[FactorDecayAnalysis]
    overlap_reports: list[HedgeFundOverlapReport]
    top_risks: list[str]


# 
# Factor Universe Definition
# 

FACTOR_DEFINITIONS: dict[str, dict] = {
    "Momentum": {
        "description": "Price momentum: buy winners, sell losers (12-1 month lookback)",
        "proxy_long": ["QQQ", "NVDA", "MSFT", "AAPL", "META"],
        "proxy_short": ["XLP", "XLU", "TLT"],
        "academic_alpha_pct": 12.0,  # alpha at discovery (~1993 Jegadeesh & Titman)
        "publication_year": 1993,
        "known_crowding_events": ["Aug 2007 quant crisis", "Mar 2020 COVID crash"],
        "typical_crowding_unwind_bps": 800,
        "capacity_usd_bn": 200,  # estimated total factor capacity
    },
    "Value (P/B, P/E)": {
        "description": "Value: buy cheap stocks (low P/B, P/E), sell expensive",
        "proxy_long": ["XLF", "XLE", "IWM", "VTV"],
        "proxy_short": ["QQQ", "IWM"],
        "academic_alpha_pct": 8.5,
        "publication_year": 1992,
        "known_crowding_events": ["2017-2020 value drought", "2022 value comeback"],
        "typical_crowding_unwind_bps": 400,
        "capacity_usd_bn": 500,
    },
    "Quality (ROE, Low Debt)": {
        "description": "Quality: buy high-ROE, low-leverage businesses",
        "proxy_long": ["MSFT", "AAPL", "JNJ", "V", "MA"],
        "proxy_short": ["HYG", "XLE"],
        "academic_alpha_pct": 6.0,
        "publication_year": 2008,
        "known_crowding_events": ["2022 quality selloff"],
        "typical_crowding_unwind_bps": 350,
        "capacity_usd_bn": 300,
    },
    "Low Volatility": {
        "description": "Low vol anomaly: low-risk stocks outperform on risk-adjusted basis",
        "proxy_long": ["XLU", "XLP", "USMV", "SPLV"],
        "proxy_short": ["IWM", "XLE", "ARKK"],
        "academic_alpha_pct": 5.5,
        "publication_year": 2006,
        "known_crowding_events": ["2020-2022 low vol underperformance"],
        "typical_crowding_unwind_bps": 300,
        "capacity_usd_bn": 400,
    },
    "Size (Small Cap)": {
        "description": "Size premium: small caps outperform large caps (SMB factor)",
        "proxy_long": ["IWM", "SLY", "VBR"],
        "proxy_short": ["SPY", "QQQ"],
        "academic_alpha_pct": 3.5,
        "publication_year": 1992,
        "known_crowding_events": ["2018-2023 small cap drought"],
        "typical_crowding_unwind_bps": 250,
        "capacity_usd_bn": 150,
    },
    "Carry (FX / Rates)": {
        "description": "Carry: borrow in low-rate currencies, invest in high-rate ones",
        "proxy_long": ["FXA", "FXY"],
        "proxy_short": ["UUP", "FXY"],
        "academic_alpha_pct": 9.0,
        "publication_year": 1980,
        "known_crowding_events": ["2008 JPY carry unwind", "2022 JPY volatility"],
        "typical_crowding_unwind_bps": 1200,
        "capacity_usd_bn": 600,
    },
    "Trend Following (CTA)": {
        "description": "Systematic trend following across futures markets",
        "proxy_long": ["DBMF", "KMLM"],
        "proxy_short": [],
        "academic_alpha_pct": 10.0,
        "publication_year": 1983,
        "known_crowding_events": ["Aug 2007", "2013 taper tantrum", "Mar 2020"],
        "typical_crowding_unwind_bps": 600,
        "capacity_usd_bn": 250,
    },
    "AI / Tech (2023-Present)": {
        "description": "AI thematic: long AI-exposed equities (NVDA, MSFT, GOOGL)",
        "proxy_long": ["NVDA", "MSFT", "GOOGL", "META", "SMCI"],
        "proxy_short": ["IWM", "XLE"],
        "academic_alpha_pct": 0.0,  # thematic, not academic
        "publication_year": 2023,
        "known_crowding_events": ["Jul 2024 rotation"],
        "typical_crowding_unwind_bps": 1500,
        "capacity_usd_bn": 800,
    },
}

# HF Clustering — stocks commonly held together by hedge funds
HF_CLUSTER_MAP: dict[str, list[str]] = {
    "AI Cluster": ["NVDA", "MSFT", "GOOGL", "META", "AMZN", "ORCL", "AMD"],
    "Momentum Large Cap": ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA"],
    "Quality Compounder": ["MSFT", "V", "MA", "UNH", "LLY", "AAPL", "JNJ"],
    "Value Cyclical": ["JPM", "BAC", "XOM", "CVX", "GS", "WFC", "C"],
    "Short Basket": ["COIN", "MSTR", "AMC", "RIOT", "TLRY"],
    "Macro / Gold": ["GLD", "SLV", "GDX", "TLT", "UUP"],
    "Small Cap Value": ["IWM", "VBR", "SLY", "AVUV"],
}

# Factor decay history (real-world observations)
FACTOR_DECAY_DATA: dict[str, dict] = {
    "Momentum": {
        "alpha_1y_recent": 8.5,
        "alpha_3y_avg": 6.2,
        "half_life_years": 12.0,
        "publication_decay": True,
        "crowding_decay": True,
        "decay_explanation": (
            "Momentum alpha has declined from ~12% (pre-publication) to ~6% as AUM tracking "
            "the factor grew. August 2007 showed momentum can reverse violently when "
            "crowded funds de-lever simultaneously."
        ),
    },
    "Value (P/B, P/E)": {
        "alpha_1y_recent": 2.1,
        "alpha_3y_avg": 3.8,
        "half_life_years": 20.0,
        "publication_decay": True,
        "crowding_decay": False,
        "decay_explanation": (
            "Value alpha has declined significantly since publication. The 2017-2020 "
            "decade-long underperformance led to significant outflows; now undercrowded "
            "but alpha itself has structurally compressed."
        ),
    },
    "Quality (ROE, Low Debt)": {
        "alpha_1y_recent": 4.5,
        "alpha_3y_avg": 5.2,
        "half_life_years": 15.0,
        "publication_decay": False,
        "crowding_decay": True,
        "decay_explanation": (
            "Quality remains relatively intact but is increasingly crowded as passive "
            "quality ETFs have grown. Concentration in mega-cap tech (which scores high "
            "on quality metrics) creates concentration risk."
        ),
    },
    "Low Volatility": {
        "alpha_1y_recent": 2.8,
        "alpha_3y_avg": 1.5,
        "half_life_years": 10.0,
        "publication_decay": True,
        "crowding_decay": True,
        "decay_explanation": (
            "Low vol anomaly has largely been arbitraged away by smart beta ETFs. "
            "Severe underperformance in 2022 as rate sensitivity was exposed."
        ),
    },
    "Size (Small Cap)": {
        "alpha_1y_recent": 1.2,
        "alpha_3y_avg": 0.5,
        "half_life_years": 8.0,
        "publication_decay": True,
        "crowding_decay": False,
        "decay_explanation": (
            "Size premium has essentially disappeared in US markets post-publication. "
            "International markets still show modest premium. Now undercrowded."
        ),
    },
    "Carry (FX / Rates)": {
        "alpha_1y_recent": 6.5,
        "alpha_3y_avg": 7.0,
        "half_life_years": 25.0,
        "publication_decay": False,
        "crowding_decay": True,
        "decay_explanation": (
            "Carry remains one of the most persistent risk premia. However, crowding "
            "creates tail risk: JPY carry unwind events can cause 10%+ drawdowns in days."
        ),
    },
    "Trend Following (CTA)": {
        "alpha_1y_recent": 7.2,
        "alpha_3y_avg": 6.8,
        "half_life_years": 30.0,
        "publication_decay": False,
        "crowding_decay": True,
        "decay_explanation": (
            "Trend following has shown remarkable persistence. Alpha has not decayed "
            "significantly post-publication — the strategy requires patience that most "
            "investors lack, limiting overcrowding."
        ),
    },
    "AI / Tech (2023-Present)": {
        "alpha_1y_recent": 45.0,
        "alpha_3y_avg": 25.0,
        "half_life_years": 3.0,
        "publication_decay": False,
        "crowding_decay": True,
        "decay_explanation": (
            "AI thematic is extremely crowded — most hedge funds have large NVDA/MSFT "
            "positions. When concentration unwinds (like Jul 2024), losses are severe. "
            "Half-life estimated at 3 years as AI monetization timelines extend."
        ),
    },
}


# 
# Price Data Layer
# 


class CrowdingDataLayer:
    def __init__(self, cache_ttl: int = 600):
        self.cache_ttl = cache_ttl
        self._cache: dict[str, tuple[float, Any]] = {}

    def _get(self, key: str) -> Any | None:
        hit = self._cache.get(key)
        if not hit:
            return None
        ts, val = hit
        if (time.time() - ts) > self.cache_ttl:
            return None
        return val

    def _set(self, key: str, val: Any) -> None:
        self._cache[key] = (time.time(), val)

    def get_close(self, ticker: str, period: str = "2y") -> pd.Series | None:
        key = f"close::{ticker}::{period}"
        c = self._get(key)
        if c is not None:
            return c
        if not HAS_YF:
            return None
        try:
            df = yf.Ticker(ticker).history(period=period)
            if df is None or df.empty or "Close" not in df.columns:
                return None
            s = df["Close"]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            s = s.dropna().astype(float)
            self._set(key, s)
            return s
        except Exception:
            return None

    def get_returns(self, ticker: str, period: str = "2y") -> pd.Series | None:
        c = self.get_close(ticker, period)
        if c is None or c.empty:
            return None
        r = c.pct_change().dropna()
        return r if not r.empty else None

    def get_momentum(self, ticker: str, days: int = 63) -> float | None:
        c = self.get_close(ticker, "1y")
        if c is None or len(c) < days + 1:
            return None
        return float(c.iloc[-1] / c.iloc[-days - 1] - 1.0)

    def get_vol(self, ticker: str, window: int = 20) -> float | None:
        r = self.get_returns(ticker, "1y")
        if r is None or len(r) < window:
            return None
        return float(r.rolling(window).std().iloc[-1] * np.sqrt(252))

    def get_zscore(self, ticker: str, window: int = 252) -> float | None:
        c = self.get_close(ticker, "2y")
        if c is None or len(c) < min(window, 30):
            return None
        w = c.tail(window) if len(c) >= window else c
        mu, sd = float(w.mean()), float(w.std())
        if sd <= 1e-12:
            return 0.0
        return float((w.iloc[-1] - mu) / sd)

    def get_pair_corr(self, tickers: list[str], period: str = "1y") -> pd.DataFrame | None:
        if not tickers:
            return None
        rets = {}
        for t in tickers:
            r = self.get_returns(t, period)
            if r is not None and not r.empty:
                rets[t] = r
        if len(rets) < 2:
            return None
        df = pd.DataFrame(rets).dropna(how="any")
        if df.empty:
            return None
        return df.corr()


# 
# Factor Crowding Engine
# 


class FactorCrowdingEngine:
    def __init__(self):
        self.data = CrowdingDataLayer()

    def _rng(self, salt: str = "") -> np.random.Generator:
        seed = abs(hash(f"crowding::{salt}")) % (2**32 - 1)
        return np.random.default_rng(seed)

    def score_factor_crowding(self, factor_name: str) -> FactorCrowdingScore:
        meta = FACTOR_DEFINITIONS.get(factor_name, {})
        decay = FACTOR_DECAY_DATA.get(factor_name, {})
        rng = self._rng(factor_name)

        # Proxy momentum/dispersion from factor proxies
        proxies = meta.get("proxy_long", [])[:5]
        vals = []
        for p in proxies:
            m = self.data.get_momentum(p, 63)
            if m is not None:
                vals.append(m)
        proxy_heat = float(np.clip(np.mean(vals) if vals else rng.normal(0.05, 0.03), -0.2, 0.4))

        crowding_score = float(np.clip(50 + proxy_heat * 180 + rng.normal(0, 8), 0, 100))
        capacity_remaining = float(np.clip(100 - crowding_score, 0, 100))

        alpha_now = float(decay.get("alpha_1y_recent", 4.0))
        alpha_orig = float(meta.get("academic_alpha_pct", max(alpha_now, 1.0)))
        alpha_decay_rate = float(max(0.0, (alpha_orig - alpha_now) / max(alpha_orig, 1e-6) * 100))

        unwind = float(meta.get("typical_crowding_unwind_bps", 400) * (crowding_score / 100))
        signal_haircut = float(np.clip((crowding_score - 40) * 0.9, 0, 60))

        if crowding_score >= 80:
            status = "OVERCROWDED"
        elif crowding_score >= 65:
            status = "CROWDED"
        elif crowding_score >= 35:
            status = "NORMAL"
        else:
            status = "UNDERCROWDED"

        raw_signal = np.tanh(proxy_heat * 4)
        risk_adjusted_signal = float(np.clip(raw_signal * (1 - signal_haircut / 100), -1, 1))

        evidence = [
            f"Proxy heat: {proxy_heat:+.2%}",
            f"Alpha decay: {alpha_decay_rate:.1f}%",
            f"Capacity remaining: {capacity_remaining:.1f}%",
        ]

        return FactorCrowdingScore(
            factor_name=factor_name,
            crowding_score=crowding_score,
            capacity_remaining=capacity_remaining,
            alpha_decay_rate=alpha_decay_rate,
            estimated_unwind_impact=unwind,
            signal_haircut=signal_haircut,
            status=status,
            description=meta.get("description", "No description"),
            evidence=evidence,
            risk_adjusted_signal=risk_adjusted_signal,
        )

    def detect_crowded_trades(self, symbols: list[str]) -> list[CrowdedTrade]:
        out: list[CrowdedTrade] = []
        for s in symbols:
            ov = self.simulate_hf_overlap(s)
            mom = self.data.get_momentum(s, 63)
            vol = self.data.get_vol(s, 20)
            mom = float(mom if mom is not None else 0.0)
            vol = float(vol if vol is not None else 0.25)

            exposures = {}
            for fn, meta in FACTOR_DEFINITIONS.items():
                score = 0.0
                if s in meta.get("proxy_long", []):
                    score += 0.8
                if s in meta.get("proxy_short", []):
                    score -= 0.6
                exposures[fn] = float(score)

            crowding_percentile = float(np.clip(ov.overlap_score * 0.7 + max(mom, 0) * 120, 0, 100))
            if crowding_percentile >= 85:
                unwind = "EXTREME"
            elif crowding_percentile >= 70:
                unwind = "HIGH"
            elif crowding_percentile >= 50:
                unwind = "MEDIUM"
            else:
                unwind = "LOW"

            expected_dd = float(np.clip(crowding_percentile / 6.0, 2, 30))
            adj_signal = float(np.clip(np.tanh(mom * 4) * (1 - crowding_percentile / 120), -1, 1))

            out.append(CrowdedTrade(
                ticker=s,
                factor_exposures=exposures,
                hf_overlap_score=ov.overlap_score,
                crowding_percentile=crowding_percentile,
                unwind_risk=unwind,
                expected_drawdown_on_unwind=expected_dd,
                momentum_score=float(mom),
                crowding_adjusted_signal=adj_signal,
                description=f"{s}: overlap={ov.overlap_score:.1f}, unwind risk={unwind}",
                avoid=crowding_percentile >= 80,
            ))
        return out

    def analyze_factor_decay(self, factor_name: str) -> FactorDecayAnalysis:
        d = FACTOR_DECAY_DATA.get(factor_name, {})
        orig = FACTOR_DEFINITIONS.get(factor_name, {}).get("academic_alpha_pct", d.get("alpha_3y_avg", 0.0))
        return FactorDecayAnalysis(
            factor_name=factor_name,
            factor_returns_1y=float(d.get("alpha_1y_recent", 0.0)),
            factor_returns_3y=float(d.get("alpha_3y_avg", 0.0)),
            half_life_years=float(d.get("half_life_years", 10.0)),
            publication_decay=bool(d.get("publication_decay", False)),
            crowding_decay=bool(d.get("crowding_decay", False)),
            current_alpha_estimate=float(d.get("alpha_1y_recent", 0.0)),
            original_alpha_estimate=float(orig),
            decay_explanation=str(d.get("decay_explanation", "No explanation available.")),
        )

    def simulate_hf_overlap(self, ticker: str) -> HedgeFundOverlapReport:
        clusters = [name for name, names in HF_CLUSTER_MAP.items() if ticker in names]
        rng = self._rng(f"hf::{ticker}")
        base = 20 + 18 * len(clusters)
        overlap = float(np.clip(base + rng.normal(0, 8), 5, 95))
        holders = int(np.clip(overlap * 1.6, 5, 140))
        own_pct = float(np.clip(overlap * 0.28, 1.0, 45.0))
        note = "High overlap across known hedge-fund clusters." if overlap > 65 else "Moderate overlap."
        return HedgeFundOverlapReport(
            ticker=ticker,
            estimated_hf_holders=holders,
            estimated_hf_ownership_pct=own_pct,
            overlap_clusters=clusters,
            overlap_score=overlap,
            note=note,
        )

    def build_dashboard(self, symbols: list[str]) -> CrowdingDashboard:
        factor_scores = [self.score_factor_crowding(f) for f in FACTOR_DEFINITIONS.keys()]
        crowded = self.detect_crowded_trades(symbols)
        decay = [self.analyze_factor_decay(f) for f in FACTOR_DEFINITIONS.keys()]
        overlap = [self.simulate_hf_overlap(s) for s in symbols]

        top_risks = []
        for c in sorted(crowded, key=lambda x: x.crowding_percentile, reverse=True)[:5]:
            top_risks.append(f"{c.ticker}: {c.unwind_risk} unwind risk ({c.crowding_percentile:.1f} pct)")

        return CrowdingDashboard(
            generated_at=datetime.utcnow().isoformat(),
            factor_scores=factor_scores,
            crowded_trades=crowded,
            decay_analysis=decay,
            overlap_reports=overlap,
            top_risks=top_risks,
        )


# 
# Singleton accessor
# 

_engine_instance: FactorCrowdingEngine | None = None


def get_crowding_engine() -> FactorCrowdingEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = FactorCrowdingEngine()
    return _engine_instance
