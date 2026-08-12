import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import streamlit as st

from data_sources import get_stock, get_realtime_price
from quant_ensemble_model import get_quant_ensemble
import options_engine

try:
    from financial_llm_engine import get_llm_engine
except ImportError:
    get_llm_engine = None

try:
    from news_analysis_engine import get_news_engine
except ImportError:
    get_news_engine = None

logger = logging.getLogger("TargetProbability")


def _first_hitting_prob(current: float, target: float, mu_annual: float,
                        sigma_annual: float, t_years: float) -> float:
    """First-hitting-time (touch) probability of reaching a target.

    This answers the user's actual question — "what is the probability that the
    price REACHES the target within the timeframe" — as opposed to the endpoint
    probability P(S_T >= K) of merely finishing above it at expiry, which is
    structurally capped and understates realistic outcomes.

    For geometric Brownian motion S_t = S0 * exp((mu - sigma^2/2)t + sigma W_t)
    the probability that the price touches level K at ANY point in [0, T] is
    (b = |ln(K/S0)|, mu_t = mu - sigma^2/2):

      Upside  (K > S0): P = Phi(( mu_t*T - b) / (sigma*sqrt(T)))
                            + (K/S0)^(2*mu_t/sigma^2) * Phi((-mu_t*T - b) / (sigma*sqrt(T)))
      Downside (K < S0): P = Phi((-mu_t*T - b) / (sigma*sqrt(T)))
                            + (K/S0)^(2*mu_t/sigma^2) * Phi(( mu_t*T - b) / (sigma*sqrt(T)))

    Under zero log-drift this reduces to the reflection-principle result
    P = 2 * Phi(-b / (sigma*sqrt(T))) — exactly twice the endpoint probability —
    so realistic targets can legitimately exceed 50%.

    Distance is measured in units of sigma*sqrt(T) (standard deviations over the
    horizon), so an impossible target (e.g. +263,000% in 5 days) yields ~0%
    regardless of how bullish the signals are: signals only shift the drift,
    they cannot manufacture probability mass.
    """
    if t_years <= 0 or sigma_annual <= 0 or current <= 0 or target <= 0:
        return 50.0
    from scipy.stats import norm
    mu_t = mu_annual - 0.5 * sigma_annual ** 2          # annual log drift
    sigma_T = sigma_annual * np.sqrt(t_years)
    b = abs(np.log(target / current))                    # barrier distance in log space
    if target >= current:
        d1 = (mu_t * t_years - b) / sigma_T
        d2 = (-mu_t * t_years - b) / sigma_T
    else:
        d1 = (-mu_t * t_years - b) / sigma_T
        d2 = (mu_t * t_years - b) / sigma_T
    # (K/S0)^(2*mu_t/sigma^2) = exp(2*mu_t*ln(K/S0)/sigma^2). Clip the exponent
    # so the coefficient is always finite (<= e^50) even for extreme targets;
    # a finite coefficient times a ~0 cdf is exactly 0.0, never inf*0 -> nan.
    coef = float(np.exp(np.clip(2.0 * mu_t * np.log(target / current) / (sigma_annual ** 2),
                                -50.0, 50.0)))
    term2 = coef * float(norm.cdf(d2))
    return float(np.clip((norm.cdf(d1) + term2) * 100.0, 0.0, 100.0))


@dataclass
class TargetAnalysisResult:
    symbol: str
    current_price: float
    target_price: float
    target_pct: float
    timeframe_days: int
    
    # Probabilities (0-100)
    statistical_probability: float
    options_implied_probability: float
    ml_probability: float
    final_probability: float
    
    # Model's base prediction
    model_predicted_price: float
    model_predicted_pct: float
    
    # Qualitative Reasoning
    market_status: str
    what_it_takes_to_happen: str
    what_it_takes_to_fail: str
    final_conclusion: str
    
    # Technical & Sentiment Context
    volatility: float
    sentiment_score: float
    news_summary: str

    # Distance-aware per-factor probabilities (0-100)
    technical_probability: float = 50.0
    quant_probability: float = 50.0
    news_probability: float = 50.0

    # Extended multi-factor analysis (WS1 sprint)
    bull_case_reasoning: List[str] = field(default_factory=list)
    bear_case_reasoning: List[str] = field(default_factory=list)
    what_it_takes: str = ""          # narrative of conditions needed for outcome
    what_prevents_it: str = ""       # narrative of what could derail it
    news_sentiment_score: float = 0.0  # weighted news sentiment (-1 to +1)
    technical_signal: str = "NEUTRAL"  # BULLISH / BEARISH / NEUTRAL
    technical_subscores: Dict[str, float] = field(default_factory=dict)
    information_weights: Dict[str, float] = field(default_factory=dict)

    # Volatility provenance & calibration support (post-recalibration)
    signal_drift: float = 0.0          # blended annualized drift used by the model
    dist_sigmas: float = 0.0           # target distance in std devs over the horizon
    volatility_source: str = "historical"  # 'options-implied' or 'historical'
    historical_volatility: float = 0.0     # realized annual vol (fallback / reference)
    implied_volatility: float = 0.0        # ATM implied vol from live chain when available
    implied_vol_expiry: str = ""           # chain expiry used for the implied vol

class TargetProbabilityEngine:
    def __init__(self):
        self.quant = get_quant_ensemble()
        try:
            self.options = options_engine.get_options_engine()
        except Exception as e:
            logger.warning(f"Options engine unavailable for target analysis: {e}")
            self.options = None
        self.llm = get_llm_engine() if get_llm_engine else None
        self.news = get_news_engine() if get_news_engine else None

    # Per-symbol ATM implied-vol cache: symbol -> (raw_median_iv, expiry, fetched_at)
    # NOTE: only the RAW median IV is cached (the expensive part). Winsorization
    # and shrinkage vs the historical estimate are re-applied on every call so
    # the accept/reject decision always reflects the CURRENT analysis's
    # historical vol, never a stale value baked into the cache.
    _IV_CACHE: Dict[str, Tuple[float, Optional[str], float]] = {}
    _IV_LOCK = threading.Lock()
    _IV_TTL_SECONDS = 6 * 3600

    def _get_implied_vol(self, symbol: str, fallback_iv: float) -> Tuple[float, Optional[str], str]:
        """Best-effort ATM implied volatility from the live options chain.

        Robust extraction:
          - Prefers a MID-DATED expiry (30-180 days to expiration, closest to
            60 DTE) rather than the nearest listed expiry, whose IVs are
            unreliable (pinning, illiquidity, wide spreads near the money) —
            a same-day expiry can read 200%+ for a blue chip.
          - Uses the MEDIAN of valid ATM IVs (calls + puts within +/-10% of
            spot), which resists outlier strikes/quotes better than the mean.
          - Winsorizes against the historical estimate: an implied reading
            outside [0.4x, 3.0x] of the historical vol is treated as a bad
            quote and rejected (historical fallback). Accepted readings are
            shrunk 70/30 toward the historical vol and clamped to that band,
            so a single garbage chain can never poison the model while
            legitimately elevated regimes (e.g. a biotech ahead of a catalyst)
            are still honored.
        Only the raw median IV is cached per symbol for 6 hours; the
        winsorization decision is re-evaluated against each call's historical
        estimate.
        """
        try:
            cached = self._IV_CACHE.get(symbol)
            if cached and (time.time() - cached[2]) < self._IV_TTL_SECONDS:
                raw_iv, expiry, _ = cached
                if raw_iv is None:
                    return fallback_iv, None, "historical"
                return self._apply_vol_sanity(symbol, raw_iv, expiry, fallback_iv)
            from data_sources import get_options_chain
            chain = get_options_chain(symbol)
            if chain.get('error'):
                raise RuntimeError(chain['error'])
            expiry = chain.get('expiration')
            # Prefer a mid-dated expiry: near-expiry IVs are unreliable.
            chosen = self._pick_chain_expiry(chain.get('expirations') or [], expiry)
            if chosen and chosen != expiry:
                chain2 = get_options_chain(symbol, expiration=chosen)
                if not chain2.get('error') and chain2.get('expiration'):
                    chain = chain2
                    expiry = chain2.get('expiration')
            spot = chain.get('underlying_price')
            ivs: List[float] = []
            for side in ('calls', 'puts'):
                df = chain.get(side)
                if df is None or df.empty or not spot:
                    continue
                strikes = pd.to_numeric(df['strike'], errors='coerce')
                near = df[(strikes.notna()) & ((strikes - spot).abs() / spot <= 0.10)]
                if near.empty or 'impliedVolatility' not in near.columns:
                    continue
                ivs.extend(pd.to_numeric(near['impliedVolatility'], errors='coerce')
                           .dropna().astype(float).tolist())
            # Plausible-band filter, then a median across all ATM quotes
            ivs = [v for v in ivs if 0.02 < v < 1.5]
            raw_iv = float(np.median(ivs)) if ivs else None
            with self._IV_LOCK:
                self._IV_CACHE[symbol] = (raw_iv, expiry, time.time())
            if raw_iv is None:
                return fallback_iv, None, "historical"
            return self._apply_vol_sanity(symbol, raw_iv, expiry, fallback_iv)
        except Exception as e:
            logger.warning(f"[TargetProb] implied vol unavailable for {symbol}: {e}")
        return fallback_iv, None, "historical"

    @staticmethod
    def _apply_vol_sanity(symbol: str, raw_iv: float, expiry: Optional[str],
                          fallback_iv: float) -> Tuple[float, Optional[str], str]:
        """Winsorize + shrink the raw implied vol against the historical estimate.

        Band [0.4x, 3.0x] of the historical vol: outside -> reject (use the
        historical estimate, clearly labeled). Inside -> shrink 70/30 toward
        historical and clamp to the band so extreme-but-plausible readings are
        tempered without being discarded.
        """
        lo, hi = 0.4 * fallback_iv, 3.0 * fallback_iv
        if not (lo <= raw_iv <= hi):
            logger.warning(
                f"[TargetProb] {symbol} implied vol {raw_iv*100:.1f}% outside "
                f"[{lo*100:.1f}%, {hi*100:.1f}%] vs historical {fallback_iv*100:.1f}% - rejected"
            )
            return fallback_iv, None, "historical"
        vol = float(np.clip(0.7 * raw_iv + 0.3 * fallback_iv, lo, hi))
        return vol, expiry, "options-implied"

    @staticmethod
    def _pick_chain_expiry(expirations: List[str], current_expiry: Optional[str]) -> Optional[str]:
        """Choose the listed expiry closest to ~60 DTE within [30, 180] days.

        Near-expiry IVs (0-2 weeks) are unreliable for a forward-looking
        volatility estimate (pinning, wide spreads, tiny time value), so a
        mid-dated contract is preferred. Falls back to the current (nearest)
        expiry if no qualifying date exists.
        """
        try:
            if not expirations:
                return current_expiry
            today = datetime.now(timezone.utc).date()

            def dte(exp_str) -> Optional[int]:
                try:
                    d = datetime.strptime(str(exp_str)[:10], "%Y-%m-%d").date()
                    return (d - today).days
                except Exception:
                    return None

            scored = [(dte(e), e) for e in expirations if dte(e) is not None]
            if not scored:
                return current_expiry
            mid = [s for s in scored if 30 <= s[0] <= 180]
            pool = mid or [s for s in scored if s[0] >= 14]
            if not pool:
                return current_expiry
            pool.sort(key=lambda s: abs(s[0] - 60))
            return pool[0][1]
        except Exception:
            return current_expiry

    def analyze_target(self, symbol: str, target_val: float, is_pct: bool, timeframe_days: int) -> TargetAnalysisResult:
        """Run comprehensive target probability analysis."""
        sym = symbol.upper().strip()
        
        # 1. Fetch current data
        df = get_stock(sym, period="max")
        if df is None or df.empty:
            raise ValueError(f"Could not retrieve historical data for {sym}")
        
        # Normalize MultiIndex columns (yfinance can return ticker-indexed MultiIndex)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        # Deduplicate columns
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated(keep='first')]
        
        # Extract a clean float Close series
        close_raw = df['Close']
        if isinstance(close_raw, pd.DataFrame):
            close_raw = close_raw.iloc[:, 0]
        close_series = close_raw.dropna().astype(float)
        
        if len(close_series) < 30:
            raise ValueError(f"Insufficient historical data for {sym} (only {len(close_series)} rows)")
        
        current_price = float(close_series.iloc[-1])
        logger.info(f"[TargetProb] {sym} current price: {current_price:.4f}, history: {len(close_series)} rows")
        
        if is_pct:
            target_pct = target_val
            target_price = current_price * (1 + target_pct / 100.0)
        else:
            target_price = target_val
            target_pct = ((target_price / current_price) - 1) * 100.0
            
        # 2. Historical & Statistical Analysis
        returns = close_series.pct_change().dropna()
        daily_vol = float(returns.std())
        ann_vol = max(daily_vol * np.sqrt(252), 0.05)  # floor to avoid degenerate blow-ups
        T = timeframe_days / 365.0

        # Market-volatility provenance: prefer the live ATM options-implied vol
        # (the market's forward-looking volatility) over the historical estimate.
        model_vol, iv_expiry, vol_source = self._get_implied_vol(sym, ann_vol)

        # ── ML / Quant Analysis — use the already-clean close_series ──────
        # NOTE: the 30-day ML forecast is used ONLY to derive the annualized
        # drift mu_ml below. The "Model Forecast" shown to the user is computed
        # later from the SAME blended drift that drives the probabilities, so
        # the displayed forecast and the probability are always consistent.
        try:
            from advanced_ml_engine import get_ensemble_engine
            _ml_engine = get_ensemble_engine()
            _clean_df = pd.DataFrame({'Close': close_series})
            _ml_out = _ml_engine.analyze_symbol_ensemble(
                data=_clean_df, symbol=sym, asset_type="STOCK", options_context=None
            )
            _model_30d_price = float(_ml_out.get("final_predicted_price") or 0)
            if _model_30d_price <= 0 or abs(_model_30d_price - current_price) < 0.001:
                raise ValueError(f"ML engine returned invalid/unchanged predicted price: {_model_30d_price:.4f} vs current {current_price:.4f}")
            _30d_return_factor = _model_30d_price / current_price
            logger.info(f"[TargetProb] ML 30d price: {_model_30d_price:.2f}, factor: {_30d_return_factor:.4f}")
        except Exception as _e:
            logger.warning(f"[TargetProb] ML engine direct call failed ({_e}), using quant signal fallback")
            signal = self.quant.predict(close_series, symbol=sym)
            _30d_return_factor = 1.0 + signal.expected_return

        signal = self.quant.predict(close_series, symbol=sym)

        # ── Directional signal stack (technical) ──────────────────────────
        tech_subscores = self._build_technical_signal_stack(close_series, current_price)
        technical_score = float(np.clip(tech_subscores.get('overall', 0.0), -1.0, 1.0))
        technical_signal = "BULLISH" if technical_score > 0.15 else "BEARISH" if technical_score < -0.15 else "NEUTRAL"

        # ── Quant ensemble signal ─────────────────────────────────────────
        q_dir = getattr(signal, 'direction', 'NEUTRAL')
        q_prob = float(getattr(signal, 'probability', 0.5) or 0.5)
        q_conf = float(getattr(signal, 'confidence', 0.0) or 0.0)
        quant_score = (1.0 if q_dir == 'BULLISH' else -1.0 if q_dir == 'BEARISH' else 0.0) * q_prob

        # ── News sentiment — full fetch and scoring ───────────────────────
        try:
            from advanced_news_processor import AdvancedNewsProcessor
            _news_result = AdvancedNewsProcessor().analyze_symbol_sentiment(sym, timeout=8)
            if _news_result and isinstance(_news_result.get('score'), (int, float)):
                news_sentiment_score = float(np.clip(_news_result['score'], -1.0, 1.0))
                sentiment_score = 50.0 + news_sentiment_score * 50.0
                _headlines = _news_result.get('top_headlines') or []
                news_summary = " | ".join(str(h) for h in _headlines[:3]) if _headlines else "No recent headlines found for this asset."
            else:
                raise ValueError("No sentiment score returned")
        except Exception:
            news_sentiment_score = 0.0
            sentiment_score = 50.0
            news_summary = "News analysis unavailable (no matching headlines)."

        # ── 3. First-hitting-time (touch) probabilities per factor ─────────
        # Each factor contributes an annualized drift to the lognormal dynamics;
        # the probability of the price TOUCHING the target at any point within
        # the timeframe is then read off the distribution via the closed-form
        # barrier formula (see _first_hitting_prob). This answers the user's
        # question — "probability of REACHING the target within T days" — and
        # makes every factor distance-aware: an impossible target (e.g.
        # +263,000% in 5 days) gets ~0% from EVERY factor no matter how bullish
        # the signals are, while realistic targets can legitimately exceed 50%.
        t_years = max(T, 1e-9)

        # Distance of the target from spot in standard deviations over the
        # horizon — the core contextual quantity the model must respect.
        dist_sigmas = abs(np.log(target_price / current_price)) / max(model_vol * np.sqrt(t_years), 1e-9)

        # Drift from ML forecast (30-day factor, annualized, capped)
        mu_ml = float(np.clip(np.log(max(_30d_return_factor, 1e-6)) / (30.0 / 365.0), -0.8, 0.8))
        # Drift from quant expected return (assumed ~30d horizon, annualized, capped)
        mu_quant = float(np.clip(np.log(max(1.0 + signal.expected_return, 1e-6)) / (30.0 / 365.0), -0.8, 0.8))
        # Drift from technical stack — score in [-1,1] mapped to an annual drift
        mu_tech = float(np.clip(technical_score * 0.35, -0.7, 0.7))
        # Drift from news sentiment — score in [-1,1] mapped to a small annual drift
        mu_news = float(np.clip(news_sentiment_score * 0.15, -0.3, 0.3))

        # Monte Carlo Simulation (drift-adjusted, seeded for reproducibility).
        # Bounded so absurdly long horizons (e.g. a 30-year "months" input)
        # cannot balloon the path matrix into gigabytes: the effective horizon is
        # capped at 10 years and the path count scales down so total simulated
        # cells stay under ~20M floats (~160 MB). For horizons beyond the cap the
        # drift term dominates the outcome anyway, so truncation is immaterial.
        _MC_MAX_CELLS = 20_000_000
        _MC_MAX_DAYS = 2520  # 10 years
        eff_days = max(1, min(int(timeframe_days), _MC_MAX_DAYS))
        sim_runs = min(10000, max(500, int(_MC_MAX_CELLS / (eff_days * 4))))
        _rng = np.random.default_rng(42)
        # Blend the four signal drifts into the MC drift (50% weight on signals,
        # 50% on historical mean so the simulation stays grounded)
        _blend_mu = 0.5 * returns.mean() * 252.0 + 0.5 * (0.40 * mu_ml + 0.30 * mu_quant + 0.20 * mu_tech + 0.10 * mu_news)

        # ── Model forecast — internally consistent with the probability math ──
        # Expected price under the SAME blended annualized drift that drives the
        # Monte Carlo paths and the closed-form factors: current * exp(mu*T).
        # This guarantees the displayed forecast and implied return always agree
        # with the probability being shown (no more "+0.00%" degenerate reads).
        model_pred_price = current_price * float(np.exp(_blend_mu * t_years))
        ml_return_pct = (model_pred_price / current_price - 1) * 100.0
        logger.info(f"[TargetProb] blended drift {_blend_mu*100:+.1f}%/yr -> forecast ${model_pred_price:,.2f} ({ml_return_pct:+.2f}%) over {timeframe_days}d")
        # Intraday sub-steps so the simulated touch frequency approximates
        # continuous monitoring. Daily-close sampling would understate touches
        # that occur mid-session and diverge from the closed-form barrier
        # factors (the classic discrete-vs-continuous barrier bias).
        substeps = 4
        sim_dt = 1.0 / (252.0 * substeps)          # fraction of a year per sub-step
        model_daily_vol = model_vol / np.sqrt(252.0)
        sim_daily_vol = model_daily_vol / np.sqrt(substeps)
        # Log drift per sub-step: annual drift scaled by dt, minus the
        # per-sub-step convexity term (variance now scaled by 1/substeps).
        sim_step_drift = _blend_mu * sim_dt - 0.5 * sim_daily_vol ** 2
        sim_returns = _rng.normal(sim_step_drift, sim_daily_vol,
                                  (eff_days * substeps, sim_runs))
        sim_paths = current_price * np.exp(np.cumsum(sim_returns, axis=0))

        # Monte Carlo touch frequency: fraction of paths that REACH the target at
        # any point during the horizon (endpoint hits understate the chance of
        # touching the level intra-horizon).
        if target_price > current_price:
            stat_prob = float(np.mean(np.max(sim_paths, axis=0) >= target_price) * 100.0)
        else:
            stat_prob = float(np.mean(np.min(sim_paths, axis=0) <= target_price) * 100.0)

        # Closed-form first-hitting-time probabilities per factor
        opt_prob = _first_hitting_prob(current_price, target_price, 0.0, model_vol, t_years)
        ml_prob = _first_hitting_prob(current_price, target_price, mu_ml, model_vol, t_years)
        tech_prob = _first_hitting_prob(current_price, target_price, mu_tech, model_vol, t_years)
        quant_prob = _first_hitting_prob(current_price, target_price, mu_quant, model_vol, t_years)
        news_prob = _first_hitting_prob(current_price, target_price, mu_news, model_vol, t_years)

        # 6. Final Probability — weighted multi-factor blend (sums to 100%)
        final_prob = (stat_prob * 0.25) + (opt_prob * 0.15) + (tech_prob * 0.20) + (quant_prob * 0.25) + (news_prob * 0.15)
        final_prob = float(np.clip(final_prob, 0.0, 100.0))
        information_weights = {
            'statistical_monte_carlo': 0.25,
            'options_implied': 0.15,
            'technical_signal_stack': 0.20,
            'quant_ensemble': 0.25,
            'news_sentiment': 0.15,
        }
        
        # 7. Bull / Bear Case Reasoning (rule-based, data-backed)
        bull_case_reasoning, bear_case_reasoning = self._build_case_reasoning(
            sym, current_price, target_price, target_pct,
            tech_subscores, technical_signal, quant_score, q_dir, q_prob, q_conf,
            news_sentiment_score, model_vol, model_pred_price, final_prob
        )

        # 8. Generative AI Narrative (what it takes / what prevents it)
        direction = "up" if target_pct > 0 else "down"
        
        llm_prompt = f"""
        You are a quantitative AI strategist.
        Asset: {sym}
        Current Price: ${current_price:.2f}
        Target: ${target_price:.2f} ({target_pct:+.2f}%)
        Timeframe: {timeframe_days} days
        
        Calculated Probability: {final_prob:.1f}%
        Volatility: {model_vol*100:.1f}% ({vol_source})
        Sentiment: {sentiment_score}/100
        Model predicts price will hit: ${model_pred_price:.2f} ({ml_return_pct:+.2f}%)
        Target distance from current price: {dist_sigmas:.2f} standard deviations over the {timeframe_days}-day horizon
        
        Provide:
        1. Market Status: Quick 1-sentence current state of the asset.
        2. what_it_takes: A 2-3 sentence narrative of the specific catalysts, news, or technical moves required for the target to be reached. If the target is extremely far from the current price relative to volatility, state explicitly that it is statistically implausible.
        3. what_prevents_it: A 2-3 sentence narrative of the key risks, events, or conditions that could derail the outcome.
        4. What it takes to happen: 1 sentence summary.
        5. What it takes to fail: 1 sentence summary.
        6. Final Conclusion: Institutional verdict on this user's expectation, calibrated to the calculated probability.
        
        Format as JSON with keys: 'market_status', 'what_it_takes', 'what_prevents_it', 'what_it_takes_to_happen', 'what_it_takes_to_fail', 'final_conclusion'.
        """
        
        try:
            if self.llm:
                ai_resp = self.llm.analyze_json(llm_prompt)
            else:
                raise RuntimeError("LLM engine not available")
        except Exception:
            # Fallback if LLM fails — data-driven narrative built from the actual
            # computed quantities (distance in sigmas, direction, drift, vol).
            if dist_sigmas >= 4.0:
                reachable = f"{sym} would need a move of {dist_sigmas:.1f} standard deviations, which the model treats as effectively impossible at {model_vol*100:.1f}% annualized volatility over {timeframe_days} days."
                block = "Ordinary volatility and mean reversion make this target statistically unreachable within the timeframe."
                verdict = f"At {final_prob:.1f}% this target is not realistic — no combination of signals makes a {dist_sigmas:.1f}σ move plausible in {timeframe_days} days."
            elif target_pct < 0:
                reachable = (f"For {sym} to fall to ${target_price:,.2f} ({target_pct:+.1f}%, a {dist_sigmas:.1f}σ move at "
                             f"{model_vol*100:.1f}% vol), it needs sustained selling pressure: earnings/guidance misses, "
                             f"deteriorating fundamentals, macro risk-off, or a sector de-rating — not just one bad day.")
                block = (f"A {_blend_mu*100:+.1f}% annualized drift + any supportive catalysts (upgrades, buybacks, "
                         f"resilient results) would keep price elevated and prevent the fall.")
                verdict = f"At {final_prob:.1f}% the decline is {('likely' if final_prob > 55 else 'possible but far from assured')} — the drift is working against it."
            else:
                reachable = (f"For {sym} to rise to ${target_price:,.2f} ({target_pct:+.1f}%, a {dist_sigmas:.1f}σ move at "
                             f"{model_vol*100:.1f}% vol), it needs a sustained catalyst: earnings beats, guidance raises, "
                             f"fundamental improvement, or sector momentum — not just a single green candle.")
                block = (f"A {_blend_mu*100:+.1f}% annualized drift or negative catalysts (misses, downgrades, macro "
                         f"shocks) would keep price below the target.")
                verdict = f"At {final_prob:.1f}% the advance is {('likely' if final_prob > 55 else 'possible but not assured')} over {timeframe_days} days."
            ai_resp = {
                "market_status": (f"{sym} trades at ${current_price:.2f} with {model_vol*100:.1f}% annualized volatility "
                                  f"({vol_source}); the target is {dist_sigmas:.1f}σ away over {timeframe_days} days."),
                "what_it_takes_to_happen": reachable,
                "what_it_takes_to_fail": block,
                "what_it_takes": reachable,
                "what_prevents_it": block,
                "final_conclusion": verdict,
            }

        what_it_takes = ai_resp.get("what_it_takes") or ai_resp.get("what_it_takes_to_happen", "")
        what_prevents_it = ai_resp.get("what_prevents_it") or ai_resp.get("what_it_takes_to_fail", "")
            
        return TargetAnalysisResult(
            symbol=sym,
            current_price=current_price,
            target_price=target_price,
            target_pct=target_pct,
            timeframe_days=timeframe_days,
            statistical_probability=stat_prob,
            options_implied_probability=opt_prob,
            ml_probability=ml_prob,
            technical_probability=tech_prob,
            quant_probability=quant_prob,
            news_probability=news_prob,
            final_probability=final_prob,
            model_predicted_price=model_pred_price,
            model_predicted_pct=ml_return_pct,
            market_status=ai_resp.get("market_status", ""),
            what_it_takes_to_happen=ai_resp.get("what_it_takes_to_happen", ""),
            what_it_takes_to_fail=ai_resp.get("what_it_takes_to_fail", ""),
            final_conclusion=ai_resp.get("final_conclusion", ""),
            volatility=model_vol,
            dist_sigmas=dist_sigmas,
            historical_volatility=ann_vol,
            implied_volatility=(model_vol if vol_source == "options-implied" else 0.0),
            implied_vol_expiry=iv_expiry or "",
            volatility_source=vol_source,
            signal_drift=_blend_mu,
            sentiment_score=sentiment_score,
            news_summary=news_summary,
            bull_case_reasoning=bull_case_reasoning,
            bear_case_reasoning=bear_case_reasoning,
            what_it_takes=what_it_takes,
            what_prevents_it=what_prevents_it,
            news_sentiment_score=news_sentiment_score,
            technical_signal=technical_signal,
            technical_subscores=tech_subscores,
            information_weights=information_weights,
        )

    # ── Directional technical signal stack (SMA, RSI, MACD, momentum, Bollinger) ──
    def _build_technical_signal_stack(self, close_series: pd.Series, current_price: float) -> Dict[str, float]:
        """Compute directional sub-scores plus an 'overall' composite in [-1, 1]."""
        sub: Dict[str, float] = {}
        try:
            s = close_series.astype(float)
            # SMA trend (20 vs 50)
            sma20 = float(s.rolling(20).mean().iloc[-1])
            sma50 = float(s.rolling(50).mean().iloc[-1]) if len(s) >= 50 else sma20
            sub['sma_trend'] = float(np.clip(np.sign(current_price - sma50) * min(abs(current_price / sma50 - 1) * 10.0, 1.0), -1.0, 1.0))

            # RSI
            rsi = self._rsi(s, 14)
            if rsi >= 70:
                sub['rsi'] = -0.5  # overbought — short-term extension risk
            elif rsi <= 30:
                sub['rsi'] = 0.5   # oversold — bounce potential
            else:
                sub['rsi'] = float(np.clip((rsi - 50) / 50, -1.0, 1.0))

            # MACD (12, 26, 9)
            ema12 = float(s.ewm(span=12).mean().iloc[-1])
            ema26 = float(s.ewm(span=26).mean().iloc[-1])
            macd = ema12 - ema26
            # NOTE: signal line approximated as EMA9-of-price − EMA26 — a lightweight
            # proxy for the true EMA9-of-MACD signal line (directional proxy only).
            macd_signal = float(s.ewm(span=9).mean().iloc[-1] - ema26)
            denom = max(abs(current_price * 0.02), 1e-9)
            sub['macd'] = float(np.clip(np.sign(macd - macd_signal) * min(abs(macd) / denom, 1.0), -1.0, 1.0))

            # 20-day momentum
            mom20 = (current_price / float(s.iloc[-21]) - 1) if len(s) >= 21 else 0.0
            sub['momentum'] = float(np.clip(np.tanh(mom20 * 8), -1.0, 1.0))

            # Bollinger position (-1 lower band, +1 upper band)
            ma20 = float(s.rolling(20).mean().iloc[-1])
            std20 = float(s.rolling(20).std().iloc[-1])
            if std20 > 0:
                lower, upper = ma20 - 2 * std20, ma20 + 2 * std20
                sub['bollinger'] = float(np.clip((current_price - lower) / (upper - lower) * 2 - 1, -1.0, 1.0))
            else:
                sub['bollinger'] = 0.0

            overall = float(np.clip(
                0.30 * sub['sma_trend'] + 0.20 * sub['rsi'] + 0.20 * sub['macd'] +
                0.20 * sub['momentum'] + 0.10 * sub['bollinger'], -1.0, 1.0))
            sub['overall'] = overall
        except Exception:
            sub = {'sma_trend': 0.0, 'rsi': 0.0, 'macd': 0.0, 'momentum': 0.0, 'bollinger': 0.0, 'overall': 0.0}
        return sub

    def _rsi(self, s: pd.Series, period: int = 14) -> float:
        """Wilder-style RSI; 50.0 on failure."""
        try:
            delta = s.diff()
            gain = delta.where(delta > 0, 0.0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
            rs = gain / (loss + 1e-9)
            rsi = 100.0 - 100.0 / (1.0 + rs)
            v = float(rsi.iloc[-1])
            return 50.0 if (np.isnan(v) or np.isinf(v)) else v
        except Exception:
            return 50.0

    def _build_case_reasoning(self, sym: str, current_price: float, target_price: float, target_pct: float,
                              tech_sub: Dict[str, float], tech_signal: str, quant_score: float,
                              q_dir: str, q_prob: float, q_conf: float, news_sent: float, ann_vol: float,
                              model_price: float, final_prob: float) -> Tuple[List[str], List[str]]:
        """Specific, data-backed reasons the target IS / ISN'T achievable."""
        bull: List[str] = []
        bear: List[str] = []
        up = target_pct >= 0

        # Model forecast vs target distance (direction-aware wording)
        if up:
            if model_price >= target_price:
                bull.append(f"The model forecasts ${model_price:,.2f} — already at/above your ${target_price:,.2f} target (implied {((model_price / current_price) - 1) * 100:+.1f}%)")
            else:
                bear.append(f"The model forecasts only ${model_price:,.2f} — ${target_price - model_price:,.2f} short of your ${target_price:,.2f} target")
        else:
            if model_price <= target_price:
                bull.append(f"The model forecasts ${model_price:,.2f} — already at/below your ${target_price:,.2f} downside target (implied {((model_price / current_price) - 1) * 100:+.1f}%)")
            else:
                bear.append(f"The model forecasts ${model_price:,.2f} — still ${model_price - target_price:,.2f} above your ${target_price:,.2f} downside target, so it does not expect the fall")

        # Technical stack
        if tech_signal == "BULLISH":
            bull.append("Technical stack (SMA, RSI, MACD, momentum, Bollinger) is aligned bullish")
        elif tech_signal == "BEARISH":
            bear.append("Technical stack is bearish — trend and momentum favor the downside")
        if tech_sub.get('momentum', 0) > 0.3:
            bull.append(f"20-day momentum strongly positive ({tech_sub['momentum']:+.2f} normalized)")
        if tech_sub.get('rsi', 0) >= 0.3:
            bull.append(f"RSI momentum supportive ({tech_sub['rsi']:+.2f})")
        elif tech_sub.get('rsi', 0) <= -0.3:
            bear.append(f"RSI momentum negative ({tech_sub['rsi']:+.2f})")

        # Quant ensemble
        if q_dir == "BULLISH":
            bull.append(f"Quant ensemble BULLISH ({q_prob*100:.0f}% probability, {q_conf*100:.0f}% confidence)")
        elif q_dir == "BEARISH":
            bear.append(f"Quant ensemble BEARISH ({q_prob*100:.0f}% probability)")

        # News sentiment
        if abs(news_sent) > 0.1:
            if news_sent > 0:
                bull.append(f"News sentiment positive ({news_sent:+.2f}) — headline tailwind")
            else:
                bear.append(f"News sentiment negative ({news_sent:+.2f}) — headline headwind")

        # Volatility
        if ann_vol > 0.5 and up:
            bear.append(f"Elevated annual volatility ({ann_vol*100:.0f}%) widens the uncertainty band around the target")
        elif ann_vol < 0.2:
            bull.append(f"Low volatility ({ann_vol*100:.0f}%) keeps the drift path to target comparatively stable")

        # Probability context
        if final_prob >= 55:
            bull.append(f"Statistical model puts {final_prob:.0f}% odds on this outcome — above coin-flip territory")
        elif final_prob <= 35:
            bear.append(f"Only {final_prob:.0f}% implied odds — the market is pricing against this outcome")

        if not bull:
            bull.append("No clear supportive catalysts — the target requires a sustained favorable shift")
        if not bear:
            bear.append("No dominant obstacle identified, but headline/macro shocks can always intervene")
        return bull[:5], bear[:5]

# Singleton Pattern
_engine = None
def get_target_probability_engine():
    global _engine
    if _engine is None:
        _engine = TargetProbabilityEngine()
    return _engine

def reset_target_probability_engine():
    """Force re-instantiation (call this after code changes)."""
    global _engine
    _engine = None


# ---------------------------------------------------------------------------
# CALIBRATION LOG  (persistence + reliability audit)
# ---------------------------------------------------------------------------
_LOG_LOCK = threading.Lock()


def _log_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data_cache", "target_probability_log.json")


def record_target_analysis(result: TargetAnalysisResult) -> None:
    """Append one completed analysis to the calibration log (atomic, capped)."""
    try:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "symbol": result.symbol,
            "current_price": round(result.current_price, 4),
            "target_price": round(result.target_price, 4),
            "target_pct": round(result.target_pct, 4),
            "timeframe_days": int(result.timeframe_days),
            "probability": round(result.final_probability, 2),
            "direction": "up" if result.target_price >= result.current_price else "down",
        }
        path = _log_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with _LOG_LOCK:
            entries = load_target_analyses()
            entries.append(entry)
            entries = entries[-2000:]
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(entries, fh, indent=1)
            os.replace(tmp, path)
    except Exception as e:
        logger.warning(f"[TargetProb] could not record analysis: {e}")


def load_target_analyses() -> List[dict]:
    """Load recorded analyses from disk (empty list on any failure)."""
    try:
        with open(_log_path(), encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _realized_touch(entry: dict) -> Optional[bool]:
    """Did the target get touched (any close in the window) after the call?"""
    try:
        from data_sources import get_stock
        # 2y window covers typical horizons; calls older than that are dropped
        # from the audit (documented limitation)
        df = get_stock(entry["symbol"], period="2y")
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        close = df['Close']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        idx = pd.to_datetime(df.index)
        start = pd.Timestamp(entry["ts"])
        if getattr(idx, 'tz', None) is not None:
            mask = idx >= start
        else:
            start_naive = start.tz_localize(None) if start.tzinfo is not None else start
            mask = idx >= start_naive
        window = close[mask].dropna().astype(float)
        if window.empty:
            return None
        if entry["direction"] == "up":
            return bool((window >= float(entry["target_price"])).any())
        return bool((window <= float(entry["target_price"])).any())
    except Exception as e:
        logger.warning(f"[TargetProb] realized-touch check failed: {e}")
        return None


def evaluate_calibration(entries: Optional[List[dict]] = None, max_checks: int = 30) -> dict:
    """Reliability audit: buckets of stated probabilities vs realized touch rate.

    Only entries whose timeframe has elapsed are evaluated; the touch outcome is
    checked against live price history from the call date onward (so intra-horizon
    touches count, matching the model's first-hitting-time semantics). Returns
    {'sample', 'bins': [{bucket, n, stated_mean, realized_rate}], 'brier'}.
    """
    if entries is None:
        entries = load_target_analyses()
    now = datetime.now(timezone.utc)
    expired = []
    for e in entries:
        try:
            # Handles both legacy 'Z' suffixes and modern '+00:00' offsets
            ts = datetime.fromisoformat(str(e.get("ts", "")).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        tf = int(e.get("timeframe_days", 0) or 0)
        if tf > 0 and (now - ts).days >= tf:
            expired.append(e)
    expired = expired[-max_checks:]

    results = []
    for e in expired:
        realized = _realized_touch(e)
        if realized is not None:
            results.append({"prob": float(e.get("probability", 0.0) or 0.0),
                            "realized": realized})

    if not results:
        return {"sample": 0, "bins": [], "brier": None}

    bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 101)]
    bin_rows = []
    for lo, hi in bins:
        rows = [r for r in results if lo <= r["prob"] < hi]
        if not rows:
            continue
        hit = sum(1 for r in rows if r["realized"])
        bin_rows.append({
            "bucket": f"{lo}-{hi}%",
            "n": len(rows),
            "stated_mean": round(sum(r["prob"] for r in rows) / len(rows), 1),
            "realized_rate": round(hit / len(rows) * 100.0, 1),
        })
    brier = sum((r["prob"] / 100.0 - (1.0 if r["realized"] else 0.0)) ** 2
                for r in results) / len(results)
    return {"sample": len(results), "bins": bin_rows, "brier": round(brier, 4)}


# ---------------------------------------------------------------------------
# TARGET PROBABILITY UI  (price plausibility analysis)
# ---------------------------------------------------------------------------


def _prob_cell_color(p: float) -> str:
    """Heat color for a probability cell (green high, red low, amber mid)."""
    p = float(np.clip(p, 0.0, 100.0))
    if p >= 70:
        return "rgba(76,175,80,0.55)"
    if p >= 45:
        return "rgba(201,168,76,0.45)"
    if p >= 25:
        return "rgba(255,152,0,0.40)"
    return "rgba(239,83,80,0.45)"


def _render_sensitivity(result: TargetAnalysisResult) -> None:
    """Volatility x timeframe sensitivity grid of touch probabilities."""
    vol_mults = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    tf_mults = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    drift = result.signal_drift
    current, target = result.current_price, result.target_price
    base_vol = max(result.volatility, 1e-6)
    base_tf = max(result.timeframe_days, 1)

    header = ("<tr><th style='text-align:left;padding:6px 10px;'>Volatility \\ Timeframe</th>" +
              "".join(f"<th style='padding:6px 10px;'>{m:.2f}x</th>" for m in tf_mults) + "</tr>")
    rows_html = []
    for vm in vol_mults:
        cells = []
        for tm in tf_mults:
            p = _first_hitting_prob(current, target, drift,
                                    base_vol * vm,
                                    max(base_tf * tm / 365.0, 1e-9))
            cells.append(
                f"<td style='padding:6px 10px;text-align:center;background:{_prob_cell_color(p)};"
                f"border-radius:4px;font-weight:600;'>{p:.0f}%</td>"
            )
        rows_html.append(
            f"<tr><td style='padding:6px 10px;font-weight:600;'>{vm:.2f}x</td>" +
            "".join(cells) + "</tr>"
        )
    st.markdown(
        f"<div style='overflow-x:auto;'><table style='width:100%;border-collapse:separate;"
        f"border-spacing:4px;font-size:0.85rem;'>" + header + "".join(rows_html) + "</table></div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Base case: volatility {result.volatility*100:.1f}% · timeframe {result.timeframe_days} days · "
        f"blended drift {drift*100:+.1f}% annualized. Rows scale volatility, columns scale the timeframe."
    )

def show_target_probability():
    """
    Streamlit UI for the price-plausibility engine.

    User inputs an asset, a target price (or target return %), and a timeframe.
    The engine returns the probability of reaching that target, powered by:
      - Historical Monte Carlo simulation (25%)
      - Options-implied probability (15%)
      - Technical signal stack (20%)
      - Quant ensemble ML signal (25%)
      - News sentiment (15%)
    Plus the model's own forecast and institutional-grade bull/bear reasoning.
    """
    import plotly.graph_objs as go

    st.title("Target Probability Engine")
    st.caption(
        "Price plausibility analysis: enter an asset, a target price, and a timeframe. "
        "The model computes the probability of reaching that target using Monte Carlo, "
        "options-implied distributions, technical signals, quant ML, and news sentiment."
    )

    # ---- Inputs ----
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        symbol = st.text_input(
            "Asset Symbol", value=st.session_state.get("tp_symbol", "AAPL"),
            placeholder="e.g. AAPL, NVDA, EURUSD=X, BTC-USD",
            help="Any ticker supported by the live data layer.",
        ).strip().upper()
    with c2:
        mode = st.radio("Target Input", ["Price ($)", "Return (%)"], horizontal=False, key="tp_mode")
    with c3:
        if mode == "Price ($)":
            target_val = st.number_input("Target Price ($)", 0.01, 1e7, 250.0, step=1.0, key="tp_price")
        else:
            target_val = st.number_input("Target Return (%)", -90.0, 1000.0, 10.0, step=1.0, key="tp_pct")
    with c4:
        timeframe_days = st.number_input("Timeframe (Days)", 5, 3650, 90, step=5, key="tp_days")

    run = st.button("Compute Target Probability", type="primary", width='stretch')

    if not run:
        st.info(
            "Configure the asset, target, and timeframe above, then click "
            "**Compute Target Probability**."
        )
        st.markdown(
            "**How it works** — the engine blends five independent, distance-aware "
            "probability estimates. Each factor computes the first-hitting-time "
            "(**touch**) probability that the price **reaches** the target at any "
            "point within the timeframe, driven by that factor's own drift:\n\n"
            "1. **Monte Carlo** (25%) — 10,000 drift-adjusted paths; % of paths that touch the target\n"
            "2. **Options-Implied** (15%) — risk-neutral (zero-drift) barrier probability\n"
            "3. **Technical Stack** (20%) — SMA / RSI / MACD / momentum / Bollinger drift\n"
            "4. **Quant Ensemble** (25%) — ML forecast drift from the ensemble model\n"
            "5. **News Sentiment** (15%) — live headline sentiment drift\n\n"
            "Every factor measures the target's distance in standard deviations over "
            "the horizon, so an unrealistic target (e.g. +1,000% in a week) correctly "
            "shows a near-zero probability even when signals are bullish — while "
            "realistic targets can legitimately clear 50% when the drift supports "
            "them. The result is a single institutional-grade probability with full "
            "factor transparency and data-backed bull/bear reasoning."
        )
        return

    if not symbol:
        st.error("Enter an asset symbol.")
        return

    # ---- Run engine ----
    with st.spinner(f"Analyzing {symbol} at {mode.replace(' ($)', '').replace(' (%)', '')} target "
                    f"{target_val:,.2f} over {timeframe_days:,.0f} days..."):
        try:
            engine = get_target_probability_engine()
            is_pct = mode == "Return (%)"
            result = engine.analyze_target(symbol, float(target_val), is_pct, int(timeframe_days))
        except ValueError as e:
            st.error(str(e))
            return
        except Exception as e:
            st.error(f"Target probability analysis failed: {e}")
            return

    st.session_state["tp_result"] = result
    st.session_state["tp_symbol"] = symbol

    # Persist this analysis for the calibration audit
    record_target_analysis(result)

    # ---- Verdict banner ----
    prob = result.final_probability
    if prob >= 70:
        verdict, vcolor = "HIGH PROBABILITY", "#4caf50"
    elif prob >= 45:
        verdict, vcolor = "MODERATE PROBABILITY", "#c9a84c"
    elif prob >= 25:
        verdict, vcolor = "LOW PROBABILITY", "#ff9800"
    else:
        verdict, vcolor = "VERY LOW PROBABILITY", "#ef5350"

    st.markdown(
        f'<div style="background:linear-gradient(135deg,#132240,#1a2d4a);border:1px solid {vcolor};'
        f'border-radius:10px;padding:22px 26px;margin:12px 0;text-align:center;">'
        f'<div style="color:#a0a8b8;font-size:0.72rem;letter-spacing:0.14em;text-transform:uppercase;">'
        f'Probability {symbol} reaches target</div>'
        f'<div style="font-size:3.2rem;font-weight:800;color:{vcolor};font-family:JetBrains Mono,monospace;">'
        f'{prob:.1f}%</div>'
        f'<div style="color:{vcolor};font-weight:700;letter-spacing:0.1em;font-size:0.85rem;">{verdict}</div>'
        f'<div style="color:#a0a8b8;font-size:0.8rem;margin-top:8px;">'
        f'Target: <b>${result.target_price:,.2f}</b> ({result.target_pct:+.2f}%) · '
        f'Current: <b>${result.current_price:,.2f}</b> · {result.timeframe_days} days</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ---- Factor probability breakdown ----
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    col_m1.metric("Monte Carlo", f"{result.statistical_probability:.1f}%")
    col_m2.metric("Options-Implied", f"{result.options_implied_probability:.1f}%")
    col_m3.metric("Technical", f"{result.technical_probability:.1f}%")
    col_m4.metric("Quant ML", f"{result.quant_probability:.1f}%")
    col_m5.metric("News Sentiment", f"{result.news_probability:.1f}%")

    # ---- Model forecast ----
    col_f1, col_f2, col_f3, col_f4, col_f5 = st.columns(5)
    col_f1.metric("Model Forecast", f"${result.model_predicted_price:,.2f}")
    col_f2.metric("Implied Return", f"{result.model_predicted_pct:+.2f}%")
    col_f3.metric("Annual Volatility", f"{result.volatility*100:.1f}%")
    col_f4.metric("Target Distance", f"{result.dist_sigmas:.2f}σ",
                  help="Distance of the target from spot in standard deviations over the timeframe "
                       "(the core plausibility quantity). <1σ = within ordinary daily/weekly noise; "
                       ">3σ = extreme; >5σ = statistically near-impossible.")
    col_f5.metric("News Sentiment", f"{result.sentiment_score:.0f}/100")
    if result.volatility_source == "options-implied":
        _src = ("options-implied (chain expiry " + str(result.implied_vol_expiry) + ")"
                if result.implied_vol_expiry else "options-implied")
    else:
        _src = "historical realized (options-implied unavailable or rejected as implausible)"
    st.caption(f"Model volatility: {result.volatility*100:.1f}% ({_src}) · "
               f"historical realized: {result.historical_volatility*100:.1f}% · "
               f"target {result.target_pct:+.1f}% = {result.dist_sigmas:.2f}σ over {result.timeframe_days} days")

    # ---- Narrative ----
    st.markdown("### Market Context")
    if result.market_status:
        st.info(result.market_status)
    if result.what_it_takes:
        st.markdown("**What it takes to get there:**")
        st.markdown(result.what_it_takes)
    if result.what_prevents_it:
        st.markdown("**What could derail it:**")
        st.markdown(result.what_prevents_it)
    if result.final_conclusion:
        st.success(result.final_conclusion)

    # ---- Bull / Bear reasoning ----
    bc1, bc2 = st.columns(2)
    with bc1:
        st.markdown("### Bull Case")
        for reason in result.bull_case_reasoning:
            st.markdown(f"- {reason}")
    with bc2:
        st.markdown("### Bear Case")
        for reason in result.bear_case_reasoning:
            st.markdown(f"- {reason}")

    # ---- Probability gauge chart ----
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        number=dict(suffix="%", font=dict(color=vcolor)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#a0a8b8"),
            bar=dict(color=vcolor),
            steps=[
                {"range": [0, 25], "color": "rgba(239,83,80,0.25)"},
                {"range": [25, 45], "color": "rgba(255,152,0,0.22)"},
                {"range": [45, 70], "color": "rgba(201,168,76,0.25)"},
                {"range": [70, 100], "color": "rgba(76,175,80,0.25)"},
            ],
        ),
    ))
    fig.update_layout(
        height=260, template="plotly_dark",
        paper_bgcolor="#0a1628", font=dict(color="#e8eaf0"),
        margin=dict(l=20, r=20, t=20, b=10),
        title=f"Likelihood of ${result.target_price:,.2f} within {result.timeframe_days} days",
    )
    st.plotly_chart(fig, width='stretch')

    # ---- Probability sensitivity grid (volatility x timeframe) ----
    st.markdown("### Probability Sensitivity")
    st.caption(
        "How the probability responds to volatility and timeframe assumptions, "
        "holding the model's blended drift constant. The base case is the center "
        "of the grid (1.0x / 1.0x)."
    )
    _render_sensitivity(result)

    # ---- Technical subscores + weights ----
    if result.technical_subscores:
        st.markdown("### Technical Signal Stack")
        tc = st.columns(len(result.technical_subscores))
        for i, (k, v) in enumerate(result.technical_subscores.items()):
            if k == "overall":
                continue
            with tc[i % len(tc)]:
                st.metric(k.replace("_", " ").title(), f"{v:+.2f}")

    if result.information_weights:
        with st.expander("Model Weighting & Methodology", expanded=False):
            st.json(result.information_weights)
            st.caption(
                "Final probability = weighted blend of five distance-aware factors. "
                "Each factor is a first-hitting-time (touch) probability: the chance "
                "the price REACHES the target at any point within the timeframe, "
                "computed from a lognormal distribution with that factor's own drift "
                "and measured against the target's distance in standard deviations "
                "over the horizon. Monte Carlo paths use a 50/50 blend of historical "
                "mean drift and signal drift (ML 40% / quant 30% / technical 20% / "
                "news 10%) and count paths that touch the target. Weights are fixed "
                "by design so probabilities are reproducible; all data is fetched "
                "live at analysis time."
            )

    if result.news_summary:
        st.markdown("### News Summary")
        st.caption(result.news_summary)

    # ---- Calibration audit (how accurate have past calls been?) ----
    with st.expander("Probability Calibration & Backtest", expanded=False):
        st.caption(
            "Every completed analysis is logged locally. This audit compares the "
            "probabilities the model issued against whether the targets were actually "
            "touched within their timeframes (checked against live price history from "
            "the call date, matching the model's touch semantics)."
        )
        if st.button("Run Calibration Audit", key="tp_cal_btn"):
            cal = evaluate_calibration()
            if cal["sample"] == 0:
                st.info(
                    "No expired probability calls to evaluate yet — run analyses and "
                    "return after their timeframes elapse."
                )
            else:
                cc1, cc2 = st.columns(2)
                cc1.metric("Expired calls evaluated", cal["sample"])
                cc2.metric(
                    "Brier score", f"{cal['brier']:.4f}",
                    help="Mean squared error between stated probability and the 0/1 outcome "
                         "(0 = perfect calibration, 1 = worst possible).",
                )
                if cal["bins"]:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Bar(
                        x=[b["bucket"] for b in cal["bins"]],
                        y=[b["realized_rate"] for b in cal["bins"]],
                        name="Realized touch rate",
                        marker_color="#4caf50",
                    ))
                    fig2.add_trace(go.Scatter(
                        x=[b["bucket"] for b in cal["bins"]],
                        y=[b["stated_mean"] for b in cal["bins"]],
                        name="Stated probability (mean)",
                        mode="lines+markers",
                        line=dict(color="#c9a84c", dash="dash"),
                    ))
                    fig2.update_layout(
                        title="Stated probability vs realized touch rate by bucket",
                        yaxis=dict(range=[0, 100], title="Percent"),
                        template="plotly_dark", paper_bgcolor="#0a1628",
                        font=dict(color="#e8eaf0"), height=320,
                        margin=dict(l=40, r=20, t=40, b=10),
                    )
                    st.plotly_chart(fig2, width='stretch')
                    st.dataframe(pd.DataFrame(cal["bins"]), width='stretch')
                    st.caption(
                        "A well-calibrated model shows realized rates close to the stated "
                        "probabilities in each bucket (the dashed line). Small samples "
                        "(n < 10) are indicative only."
                    )


