"""
Comparative Analysis Engine — Octavian Terminal
================================================
Institutional-grade, cross-asset comparison system.

Supports: Equities, Crypto, Commodities, Forex, Options, Futures
- Unlimited asset comparisons (parallel async)
- Custom model tailoring (Quant/Fundamentals/Macro/Technical/Full/Custom)
- AI-written strategy theses via LM Studio
- Correlation matrix, relative scoring, cross-asset strategy suggestions
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("ComparativeAnalysis")

# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AssetScore:
    symbol: str
    asset_type: str          # stock, crypto, commodity, forex, futures, options
    current_price: float
    change_1d: float
    change_1w: float
    change_1m: float
    volatility_annual: float

    # Model scores (0–100 scale)
    quant_score: float = 50.0
    fundamental_score: float = 50.0
    macro_score: float = 50.0
    technical_score: float = 50.0
    overall_score: float = 50.0

    # Signal
    decision: str = "NEUTRAL"       # STRONG BUY / BUY / NEUTRAL / SELL / STRONG SELL
    conviction: float = 0.0         # 0.0–1.0
    expected_return_30d: float = 0.0
    regime_fit: str = "Neutral"

    # Technical levels
    support: float = 0.0
    resistance: float = 0.0
    rsi: float = 50.0
    macd_signal: str = "NEUTRAL"

    # Options Greeks (if applicable)
    iv: float = 0.0
    delta: float = 0.0
    theta: float = 0.0

    # Narrative
    individual_conclusion: str = ""
    model_breakdown: Dict[str, Any] = field(default_factory=dict)

    error: Optional[str] = None


@dataclass
class CrossAssetStrategy:
    name: str
    type: str                       # PAIRS_TRADE, BASKET_LONG, BASKET_SHORT, SPREAD, ROTATION
    assets_long: List[str]
    assets_short: List[str]
    rationale: str                  # Rule-based reason
    ai_thesis: str = ""             # LLM-written narrative
    expected_return: float = 0.0
    risk_level: str = "MEDIUM"      # LOW / MEDIUM / HIGH
    time_horizon: str = "2–4 weeks"
    adjustable_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparativeResult:
    assets: Dict[str, AssetScore]
    correlation_matrix: pd.DataFrame
    ranked_by_score: List[str]
    ranked_by_momentum: List[str]
    strategies: List[CrossAssetStrategy]
    comparative_conclusions: Dict[str, str]   # e.g., "AAPL vs BTC": "..."
    model_config: Dict[str, bool]
    run_time_ms: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# ASSET TYPE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_asset_type(symbol: str) -> str:
    """Auto-detect asset type from symbol format."""
    s = symbol.upper()
    # Crypto
    if s.endswith("-USD") or s.endswith("-USDT") or s.endswith("-BTC"):
        return "crypto"
    # Futures / Commodities
    if s.endswith("=F"):
        return "futures"
    # Forex
    if s.endswith("=X") or ("/" in s and len(s) == 7):
        return "forex"
    # Options (format: AAPL240119C00150000)
    if len(s) > 10 and any(c in s for c in ["C0", "P0"]):
        return "options"
    # Known crypto without -USD
    _CRYPTO_BASES = {"BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "DOGE", "AVAX", "DOT", "MATIC", "LTC", "ATOM"}
    if s in _CRYPTO_BASES:
        return "crypto"
    return "stock"


# ─────────────────────────────────────────────────────────────────────────────
# TECHNICAL INDICATOR HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _calc_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty else 50.0


def _calc_macd_signal(series: pd.Series) -> Tuple[str, float]:
    ema12 = series.ewm(span=12).mean()
    ema26 = series.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    histogram = macd - signal
    last_hist = float(histogram.iloc[-1]) if not histogram.empty else 0.0
    if last_hist > 0.001:
        return "BULLISH", last_hist
    elif last_hist < -0.001:
        return "BEARISH", last_hist
    return "NEUTRAL", last_hist


def _calc_support_resistance(series: pd.Series, lookback: int = 20) -> Tuple[float, float]:
    recent = series.tail(lookback)
    return float(recent.min()), float(recent.max())


def _calc_technical_score(close: pd.Series, volume: pd.Series = None) -> float:
    """Score 0-100 based on trend, RSI, MACD, and moving averages."""
    score = 50.0
    try:
        rsi = _calc_rsi(close)
        macd_sig, macd_hist = _calc_macd_signal(close)

        sma20 = float(close.rolling(20).mean().iloc[-1])
        sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else sma20
        price = float(close.iloc[-1])

        # RSI contribution (±15 pts)
        if rsi < 30:
            score += 15  # oversold → bullish potential
        elif rsi < 45:
            score += 5
        elif rsi > 70:
            score -= 15  # overbought → bearish
        elif rsi > 55:
            score -= 5

        # MACD contribution (±15 pts)
        if macd_sig == "BULLISH":
            score += 15
        elif macd_sig == "BEARISH":
            score -= 15

        # MA trend (±20 pts)
        if price > sma20 and sma20 > sma50:
            score += 20  # strong uptrend
        elif price > sma20:
            score += 10
        elif price < sma20 and sma20 < sma50:
            score -= 20  # strong downtrend
        elif price < sma20:
            score -= 10

    except Exception as e:
        logger.debug(f"Technical score calc error: {e}")

    return float(np.clip(score, 0, 100))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class ComparativeAnalysisEngine:
    """
    Institutional cross-asset comparison engine.
    Supports unlimited assets with configurable model tailoring.
    """

    def __init__(self):
        self._ml_engine = None
        self._macro_engine = None
        self._fund_engine = None

    # ── Lazy-loaded engines ────────────────────────────────────────────────

    @property
    def ml_engine(self):
        if self._ml_engine is None:
            from advanced_ml_engine import get_ensemble_engine
            self._ml_engine = get_ensemble_engine()
        return self._ml_engine

    @property
    def macro_engine(self):
        if self._macro_engine is None:
            try:
                from macro_cross_asset_engine import get_macro_engine
                self._macro_engine = get_macro_engine()
            except Exception:
                self._macro_engine = None
        return self._macro_engine

    @property
    def fund_engine(self):
        if self._fund_engine is None:
            try:
                from fundamental_analyzer import FundamentalAnalyzer
                self._fund_engine = FundamentalAnalyzer()
            except Exception:
                self._fund_engine = None
        return self._fund_engine

    # ── Data Fetching ─────────────────────────────────────────────────────

    def _fetch_price_data(self, symbol: str, asset_type: str, period: str = "1y", retries: int = 2) -> Optional[pd.DataFrame]:
        """Fetch price data with retry logic and symbol normalization.

        Attempts up to ``retries``+1 fetches with short backoff, normalizing
        the symbol for the target asset class (futures via the futures-aware
        fetcher, forex via the FX fetcher, etc.). Returns None only after all
        attempts fail, and records a per-symbol error message for the UI.
        """
        from data_sources import get_stock, get_fx, get_futures_proxy

        symbol = (symbol or "").strip().upper()
        attempts = max(1, int(retries) + 1)
        last_err: Optional[str] = None
        for attempt in range(attempts):
            try:
                if asset_type in ("stock", "crypto", "options"):
                    df = get_stock(symbol, period=period)
                elif asset_type == "forex":
                    df = get_fx(symbol, period=period)
                elif asset_type in ("futures", "commodity"):
                    df = get_futures_proxy(symbol, period=period)
                else:
                    df = get_stock(symbol, period=period)

                if df is not None and not df.empty and "Close" in df.columns:
                    # Ensure Close is a clean 1D series
                    close = df["Close"]
                    if isinstance(close, pd.DataFrame):
                        close = close.iloc[:, 0]
                    df["Close"] = close.astype(float)
                    return df
                last_err = f"empty data or no Close column for {symbol}"
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                logger.error(f"Data fetch attempt {attempt + 1}/{attempts} failed for {symbol}: {e}")
            # Small backoff between retries (rate-limit friendly)
            if attempt < attempts - 1:
                time.sleep(0.5 * (attempt + 1))
        if last_err:
            logger.error(f"Data fetch error {symbol} after {attempts} attempts: {last_err}")
        return None

    def _safe_pct_change(self, close: pd.Series, lookback: int) -> float:
        try:
            if len(close) < lookback + 1:
                return 0.0
            return float((close.iloc[-1] / close.iloc[-lookback] - 1) * 100)
        except Exception:
            return 0.0

    # ── Single-Asset Analysis ────────────────────────────────────────────

    def _analyze_single_asset(
        self,
        symbol: str,
        asset_type: str,
        model_config: Dict[str, bool],
    ) -> AssetScore:
        """Full single-asset analysis with all enabled models."""

        result = AssetScore(symbol=symbol, asset_type=asset_type,
                            current_price=0, change_1d=0, change_1w=0,
                            change_1m=0, volatility_annual=0)

        df = self._fetch_price_data(symbol, asset_type)
        if df is None or df.empty:
            result.error = f"No market data available for {symbol}"
            return result

        close = df["Close"].astype(float).dropna()
        volume = df["Volume"].astype(float).dropna() if "Volume" in df.columns else pd.Series(dtype=float)

        if len(close) < 5:
            result.error = f"Insufficient history for {symbol}"
            return result

        # ── Base price metrics ───────────────────────────────────────────
        result.current_price = float(close.iloc[-1])
        result.change_1d = self._safe_pct_change(close, 1)
        result.change_1w = self._safe_pct_change(close, 5)
        result.change_1m = self._safe_pct_change(close, 21)

        returns = close.pct_change().dropna()
        result.volatility_annual = float(returns.std() * np.sqrt(252)) if len(returns) > 2 else 0.0

        # Technical levels
        result.support, result.resistance = _calc_support_resistance(close)
        result.rsi = _calc_rsi(close)
        macd_sig, _ = _calc_macd_signal(close)
        result.macd_signal = macd_sig

        # ── Model scoring ────────────────────────────────────────────────
        enabled_scores = []

        if model_config.get("use_technical", True):
            result.technical_score = _calc_technical_score(close, volume)
            enabled_scores.append(result.technical_score)
            result.model_breakdown["technical"] = result.technical_score

        if model_config.get("use_quant", True):
            try:
                ml_out = self.ml_engine.analyze_symbol_ensemble(
                    df, symbol=symbol, asset_type=asset_type.upper(), fast_mode=True
                )
                result.quant_score = float(ml_out.get("alpha_score", 50.0))
                result.expected_return_30d = float(ml_out.get("predicted_return", 0.0)) * 100
                result.model_breakdown["quant"] = result.quant_score
                result.model_breakdown["ml_detail"] = ml_out.get("model_breakdown", {})
                enabled_scores.append(result.quant_score)
            except Exception as e:
                logger.warning(f"Quant engine error for {symbol}: {e}")

        if model_config.get("use_macro", False) and self.macro_engine:
            try:
                dash = self.macro_engine.build_dashboard()
                regime = dash.macro_regime.name
                result.regime_fit = regime
                # Macro score: 1 if regime is growth-positive, else penalty
                macro_bullish = any(k in regime.lower() for k in ["reflation", "goldilocks", "risk-on"])
                macro_bear = any(k in regime.lower() for k in ["stagflation", "contraction", "risk-off"])
                result.macro_score = 65.0 if macro_bullish else (35.0 if macro_bear else 50.0)
                result.model_breakdown["macro"] = result.macro_score
                enabled_scores.append(result.macro_score)
            except Exception as e:
                logger.warning(f"Macro engine error for {symbol}: {e}")

        if model_config.get("use_fundamentals", False) and self.fund_engine and asset_type == "stock":
            try:
                fund = self.fund_engine.fetch_fundamentals(symbol)
                if fund:
                    fund_raw = getattr(fund, "score", 0.0)
                    result.fundamental_score = float(np.clip((fund_raw + 1.0) / 2.0 * 100, 0, 100))
                    result.model_breakdown["fundamentals"] = result.fundamental_score
                    enabled_scores.append(result.fundamental_score)
            except Exception as e:
                logger.warning(f"Fundamentals error for {symbol}: {e}")

        # ── Composite overall score ──────────────────────────────────────
        if enabled_scores:
            result.overall_score = float(np.mean(enabled_scores))
        else:
            # Fall back to momentum-based heuristic
            momentum = (result.change_1w * 0.5) + (result.change_1m * 0.5)
            result.overall_score = float(np.clip(50 + momentum * 2, 0, 100))

        # ── Decision signal ──────────────────────────────────────────────
        s = result.overall_score
        if s >= 75:
            result.decision = "STRONG BUY"
            result.conviction = min((s - 75) / 25 * 0.8 + 0.6, 0.95)
        elif s >= 60:
            result.decision = "BUY"
            result.conviction = (s - 60) / 15 * 0.4 + 0.3
        elif s <= 25:
            result.decision = "STRONG SELL"
            result.conviction = min((25 - s) / 25 * 0.8 + 0.6, 0.95)
        elif s <= 40:
            result.decision = "SELL"
            result.conviction = (40 - s) / 15 * 0.4 + 0.3
        else:
            result.decision = "NEUTRAL"
            result.conviction = 0.2

        # Options IV if applicable
        if asset_type == "options":
            try:
                if "impliedVolatility" in df.columns:
                    result.iv = float(df["impliedVolatility"].iloc[-1])
            except Exception:
                pass

        # ── Individual conclusion (heuristic, LLM will enhance) ──────────
        result.individual_conclusion = self._build_individual_conclusion(result)

        return result

    def _build_individual_conclusion(self, r: AssetScore) -> str:
        """Deterministic conclusion for a single asset."""
        parts = []
        # Signal header
        parts.append(f"**Signal:** {r.decision} ({r.conviction*100:.0f}% conviction)")

        # Price action
        trend = "uptrend" if r.change_1m > 2 else "downtrend" if r.change_1m < -2 else "sideways consolidation"
        parts.append(f"**Trend:** {trend} | 1D: {r.change_1d:+.2f}% | 1W: {r.change_1w:+.2f}% | 1M: {r.change_1m:+.2f}%")

        # Technical
        rsi_comment = "oversold" if r.rsi < 30 else "overbought" if r.rsi > 70 else "neutral"
        parts.append(f"**RSI ({r.rsi:.1f}):** {rsi_comment} | MACD: {r.macd_signal}")

        # Support/Resistance
        if r.support and r.resistance:
            parts.append(f"**Key Levels:** Support ${r.support:.2f} / Resistance ${r.resistance:.2f}")

        # Volatility
        parts.append(f"**Annual Volatility:** {r.volatility_annual*100:.1f}%")

        if r.expected_return_30d != 0:
            parts.append(f"**30D Expected Return:** {r.expected_return_30d:+.2f}%")

        return "\n".join(parts)

    # ── Correlation Matrix ────────────────────────────────────────────────

    def _build_correlation_matrix(self, price_data: Dict[str, pd.Series]) -> pd.DataFrame:
        """Build pairwise Pearson correlation matrix from daily returns."""
        returns_dict = {}
        for sym, prices in price_data.items():
            if prices is not None and len(prices) > 5:
                ret = prices.pct_change().dropna()
                returns_dict[sym] = ret

        if len(returns_dict) < 2:
            return pd.DataFrame()

        df = pd.DataFrame(returns_dict).dropna()
        if df.empty:
            return pd.DataFrame()

        corr = df.corr()
        return corr

    # ── Cross-Asset Strategies (Rule-Based) ───────────────────────────────

    def _generate_strategies(
        self,
        scores: Dict[str, AssetScore],
        corr: pd.DataFrame,
    ) -> List[CrossAssetStrategy]:
        """Generate deterministic cross-asset strategies from scores + correlation."""
        strategies: List[CrossAssetStrategy] = []
        sorted_by_score = sorted(scores.values(), key=lambda x: x.overall_score, reverse=True)
        top = [s for s in sorted_by_score if s.error is None]

        if len(top) < 2:
            return strategies

        best = top[0]
        worst = top[-1]

        # 1. Pairs Trade (Long best / Short worst)
        if best.overall_score > 60 and worst.overall_score < 40:
            strategies.append(CrossAssetStrategy(
                name=f"Pairs Trade: Long {best.symbol} / Short {worst.symbol}",
                type="PAIRS_TRADE",
                assets_long=[best.symbol],
                assets_short=[worst.symbol],
                rationale=(
                    f"{best.symbol} scores {best.overall_score:.1f}/100 ({best.decision}) "
                    f"vs {worst.symbol} at {worst.overall_score:.1f}/100 ({worst.decision}). "
                    f"Score spread of {best.overall_score - worst.overall_score:.1f} pts justifies the relative value trade. "
                    f"Expected 30D return spread: {(best.expected_return_30d - worst.expected_return_30d):+.2f}%."
                ),
                expected_return=(best.expected_return_30d - worst.expected_return_30d) * 0.5,
                risk_level="MEDIUM",
                adjustable_params={
                    "long_weight": 0.5, "short_weight": 0.5,
                    "stop_loss_pct": 5.0, "target_pct": 10.0
                }
            ))

        # 2. Long Basket (Top 3 by score)
        top3 = [s for s in top[:3] if s.overall_score >= 60]
        if len(top3) >= 2:
            symbols = [s.symbol for s in top3]
            weights = {s.symbol: round(1 / len(top3), 2) for s in top3}
            avg_return = np.mean([s.expected_return_30d for s in top3])
            strategies.append(CrossAssetStrategy(
                name=f"Momentum Basket: {' + '.join(symbols)}",
                type="BASKET_LONG",
                assets_long=symbols,
                assets_short=[],
                rationale=(
                    f"All three assets score above 60/100 with {', '.join([f'{s.symbol}: {s.decision}' for s in top3])}. "
                    f"Equal-weight basket expected 30D return: {avg_return:+.2f}%. "
                    f"Cross-asset diversification reduces single-name concentration risk."
                ),
                expected_return=avg_return,
                risk_level="MEDIUM",
                adjustable_params={"weights": weights, "rebalance_frequency": "weekly"}
            ))

        # 3. Low-Correlation Diversification Play
        if not corr.empty and len(corr) >= 2:
            min_corr_pair = None
            min_corr_val = 1.0
            for i, sym_a in enumerate(corr.columns):
                for sym_b in corr.columns[i+1:]:
                    val = corr.loc[sym_a, sym_b]
                    if abs(val) < abs(min_corr_val) and sym_a in scores and sym_b in scores:
                        sc_a = scores[sym_a]
                        sc_b = scores[sym_b]
                        if sc_a.error is None and sc_b.error is None:
                            min_corr_val = val
                            min_corr_pair = (sym_a, sym_b, sc_a, sc_b)

            if min_corr_pair and abs(min_corr_val) < 0.35:
                sa, sb, score_a, score_b = min_corr_pair
                strategies.append(CrossAssetStrategy(
                    name=f"Diversification Play: {sa} + {sb} (ρ = {min_corr_val:.2f})",
                    type="ROTATION",
                    assets_long=[sa, sb],
                    assets_short=[],
                    rationale=(
                        f"{sa} and {sb} show near-zero correlation (ρ = {min_corr_val:.2f}), "
                        f"making them ideal portfolio diversifiers. "
                        f"{sa} volatility: {score_a.volatility_annual*100:.1f}%, "
                        f"{sb} volatility: {score_b.volatility_annual*100:.1f}%. "
                        f"Combined holding reduces portfolio-level drawdown without sacrificing return."
                    ),
                    expected_return=np.mean([score_a.expected_return_30d, score_b.expected_return_30d]),
                    risk_level="LOW",
                    adjustable_params={"allocation_a": 0.5, "allocation_b": 0.5}
                ))

        # 4. Cross-Asset Spread (highest volatility short as hedge)
        high_vol = sorted([s for s in top if s.error is None], key=lambda x: x.volatility_annual, reverse=True)
        low_vol = sorted([s for s in top if s.error is None], key=lambda x: x.volatility_annual)
        if high_vol and low_vol and high_vol[0].symbol != low_vol[0].symbol:
            strategies.append(CrossAssetStrategy(
                name=f"Volatility Spread: Long {low_vol[0].symbol} / Hedge {high_vol[0].symbol}",
                type="SPREAD",
                assets_long=[low_vol[0].symbol],
                assets_short=[high_vol[0].symbol],
                rationale=(
                    f"{low_vol[0].symbol} ({low_vol[0].volatility_annual*100:.1f}% annual vol) provides stable directional exposure. "
                    f"Paired with a partial short in {high_vol[0].symbol} ({high_vol[0].volatility_annual*100:.1f}% vol) "
                    f"creates a volatility-adjusted spread. Net position benefits from relative vol compression."
                ),
                expected_return=low_vol[0].expected_return_30d * 0.6,
                risk_level="MEDIUM",
                adjustable_params={"long_size": 1.0, "hedge_size": 0.3}
            ))

        return strategies

    # ── Cross-Asset Comparative Conclusions ──────────────────────────────

    def _build_comparative_conclusions(self, scores: Dict[str, AssetScore]) -> Dict[str, str]:
        """Head-to-head written comparison for each asset pair."""
        conclusions = {}
        symbols = [sym for sym, s in scores.items() if s.error is None]

        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                a_sym, b_sym = symbols[i], symbols[j]
                a, b = scores[a_sym], scores[b_sym]

                winner = a_sym if a.overall_score >= b.overall_score else b_sym
                loser = b_sym if winner == a_sym else a_sym
                w = scores[winner]
                l = scores[loser]

                score_diff = abs(a.overall_score - b.overall_score)
                conviction_text = "decisively" if score_diff > 20 else "marginally" if score_diff < 8 else "clearly"

                conclusion = (
                    f"**{winner} vs {loser}:** {winner} {conviction_text} outperforms on the composite score "
                    f"({w.overall_score:.1f} vs {l.overall_score:.1f}). "
                )

                if w.change_1m > l.change_1m:
                    conclusion += f"{winner} also leads on 30-day price momentum ({w.change_1m:+.2f}% vs {l.change_1m:+.2f}%). "

                if w.volatility_annual < l.volatility_annual:
                    conclusion += f"It achieves this with lower volatility ({w.volatility_annual*100:.1f}% vs {l.volatility_annual*100:.1f}%), indicating superior risk-adjusted characteristics. "
                else:
                    conclusion += f"However, {loser} carries lower volatility ({l.volatility_annual*100:.1f}% vs {w.volatility_annual*100:.1f}%), which may suit more risk-averse allocators. "

                if a.asset_type != b.asset_type:
                    conclusion += f"Note: {a_sym} ({a.asset_type}) and {b_sym} ({b.asset_type}) belong to different asset classes — cross-class comparison should be interpreted in a diversification context rather than a direct substitution context."

                conclusions[f"{a_sym} vs {b_sym}"] = conclusion

        return conclusions

    # ── AI Thesis Generation (LM Studio) ─────────────────────────────────

    def _enrich_strategies_with_ai(
        self,
        strategies: List[CrossAssetStrategy],
        scores: Dict[str, AssetScore],
        model_config: Dict[str, bool],
    ) -> List[CrossAssetStrategy]:
        """Call LM Studio to generate AI-written theses for each strategy."""
        try:
            from financial_llm_engine import _call_llm, check_llm_connectivity
            if not check_llm_connectivity():
                logger.info("LM Studio not available — using rule-based theses only")
                return strategies
        except ImportError:
            return strategies

        system_prompt = """You are OCTAVIAN — an institutional-grade portfolio strategist.
You generate concise, signal-dense investment theses for cross-asset strategies.
Rules:
- 3–5 sentences max
- Reference specific assets, their scores, and expected returns
- Include a risk factor and invalidation condition
- NO filler text, NO disclaimers
- Output ONLY the thesis paragraph"""

        enriched = []
        for strat in strategies:
            try:
                asset_context = ""
                for sym in strat.assets_long + strat.assets_short:
                    if sym in scores and scores[sym].error is None:
                        s = scores[sym]
                        asset_context += (
                            f"{sym}: Score={s.overall_score:.1f}, Signal={s.decision}, "
                            f"1M={s.change_1m:+.2f}%, Vol={s.volatility_annual*100:.1f}%, "
                            f"Exp.Return={s.expected_return_30d:+.2f}%. "
                        )

                prompt = f"""Strategy: {strat.name}
Type: {strat.type}
Rule-Based Rationale: {strat.rationale}
Asset Data: {asset_context}
Active Models: {', '.join(k for k, v in model_config.items() if v)}

Write a 3–5 sentence institutional thesis for this strategy."""

                ai_text = _call_llm(prompt=prompt, system=system_prompt)
                strat.ai_thesis = ai_text if ai_text else ""
            except Exception as e:
                logger.warning(f"AI thesis generation failed for {strat.name}: {e}")
            enriched.append(strat)

        return enriched

    # ── Main Entry Point ──────────────────────────────────────────────────

    def run_analysis(
        self,
        assets: List[Dict[str, str]],
        model_config: Dict[str, bool],
    ) -> ComparativeResult:
        """
        Run full comparative analysis.

        assets = [{"symbol": "AAPL", "asset_type": "stock"}, ...]
        model_config = {
            "use_quant": True, "use_fundamentals": False,
            "use_macro": True, "use_technical": True
        }
        """
        t_start = time.time()

        def normalize_symbol(symbol: str) -> str:
            mapping = {
                "CRUDE OIL": "CL=F", "OIL": "CL=F", "WTI": "CL=F",
                "GOLD": "GC=F", "SILVER": "SI=F", "NATURAL GAS": "NG=F",
                "COPPER": "HG=F", "SP500": "^GSPC", "S&P500": "^GSPC",
                "S&P 500": "^GSPC", "NASDAQ": "^IXIC", "DOW JONES": "^DJI",
                "DOW": "^DJI", "BITCOIN": "BTC-USD", "BTC": "BTC-USD",
                "ETHEREUM": "ETH-USD", "ETH": "ETH-USD", "VIX": "^VIX"
            }
            return mapping.get(symbol.upper().strip(), symbol.strip())

        # 1. Resolve asset types
        resolved = []
        for a in assets:
            sym = normalize_symbol(a["symbol"])
            atype = a.get("asset_type") or detect_asset_type(sym)
            resolved.append({"symbol": sym, "asset_type": atype})

        # 2. Parallel single-asset analysis
        scores: Dict[str, AssetScore] = {}
        price_series: Dict[str, Optional[pd.Series]] = {}

        def _analyze_and_cache(asset_info):
            sym = asset_info["symbol"]
            atype = asset_info["asset_type"]
            result = self._analyze_single_asset(sym, atype, model_config)
            df = self._fetch_price_data(sym, atype, period="6mo")
            close = None
            if df is not None and not df.empty and "Close" in df.columns:
                close = df["Close"].astype(float)
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
            return sym, result, close

        with ThreadPoolExecutor(max_workers=min(len(resolved), 8)) as pool:
            futures = [pool.submit(_analyze_and_cache, a) for a in resolved]
            for f in futures:
                try:
                    sym, result, close = f.result(timeout=90)
                    scores[sym] = result
                    price_series[sym] = close
                except Exception as e:
                    logger.error(f"Asset analysis thread error: {e}")

        # 3. Correlation matrix
        valid_prices = {k: v for k, v in price_series.items() if v is not None and len(v) > 5}
        corr_matrix = self._build_correlation_matrix(valid_prices)

        # 4. Rankings
        valid_scores = {k: v for k, v in scores.items() if v.error is None}
        ranked_by_score = [
            s.symbol for s in sorted(valid_scores.values(), key=lambda x: x.overall_score, reverse=True)
        ]
        ranked_by_momentum = [
            s.symbol for s in sorted(valid_scores.values(), key=lambda x: x.change_1m, reverse=True)
        ]

        # 5. Strategies (rule-based)
        strategies = self._generate_strategies(valid_scores, corr_matrix)

        # 6. AI thesis enrichment
        strategies = self._enrich_strategies_with_ai(strategies, valid_scores, model_config)

        # 7. Cross-asset comparative conclusions
        comparative_conclusions = self._build_comparative_conclusions(valid_scores)

        run_time_ms = int((time.time() - t_start) * 1000)

        return ComparativeResult(
            assets=scores,
            correlation_matrix=corr_matrix,
            ranked_by_score=ranked_by_score,
            ranked_by_momentum=ranked_by_momentum,
            strategies=strategies,
            comparative_conclusions=comparative_conclusions,
            model_config=model_config,
            run_time_ms=run_time_ms,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

_engine: Optional[ComparativeAnalysisEngine] = None


def get_comparative_engine() -> ComparativeAnalysisEngine:
    global _engine
    if _engine is None:
        _engine = ComparativeAnalysisEngine()
    return _engine
