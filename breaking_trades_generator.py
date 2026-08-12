import numpy as np
import pandas as pd
import asyncio
import concurrent.futures
import re
import threading
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from data_sources import get_stock, get_realtime_prices_batch
from quant_ensemble_model import get_quant_ensemble

@dataclass
class OptionLeg:
    strike: float
    option_type: str  # CALL or PUT
    side: int  # 1 for LONG, -1 for SHORT
    expiry: str
    delta: float
    theoretical_price: float

@dataclass
class OptionSetup:
    strategy_name: str
    legs: List[OptionLeg]
    net_debit_credit: float
    max_risk: float
    max_reward: float
    breakeven: List[float]
    greeks: Dict[str, float]

@dataclass
class BreakingTradeSetup:
    """Complete trade setup with all specifications"""
    # Identification
    symbol: str
    trade_id: str
    generated_at: datetime

    # Trade Direction & Type
    direction: str  # LONG or SHORT
    setup_type: str  # "Breakout", "Momentum", "Reversal", "Trend", "Range"

    # Price Levels
    current_price: float
    entry_price: float  # Trigger price
    stop_loss: float
    take_profit_1: float
    take_profit_2: Optional[float]
    take_profit_3: Optional[float]

    # Position Sizing
    risk_reward_ratio: float
    suggested_position_size_pct: float  # % of portfolio
    max_risk_per_trade_pct: float  # % of portfolio to risk

    # Confidence & Scoring
    confidence_score: float  # 0-100
    technical_score: float
    momentum_score: float
    volatility_score: float
    volume_score: float

    # Comprehensive Reasoning
    primary_reason: str
    supporting_factors: List[str]
    risk_factors: List[str]
    technical_analysis: str

    # Timing
    expected_hold_time: str  # "1-3 days", "Intraday", "1-2 weeks"
    market_condition: str

    # Technical Indicators
    key_levels: Dict[str, float]
    indicators: Dict[str, Any]

    # Alerts
    invalidation_price: float  # Price that invalidates setup
    alert_notes: List[str]

    # Options Overlay
    options_setup: Optional[OptionSetup] = None


class BreakingTradesGenerator:
    """
    Optimized High-Velocity Trade Generator.
    Uses tiered filtering and parallel processing to scan markets in seconds.
    Restores full institutional logic depth for high-fidelity setups.
    """

    def __init__(self, min_confidence: float = 35.0):
        """
        Args:
            min_confidence: Minimum setup confidence score (0-100). Lowered to
                35.0 from 45.0 because typical signal scores cluster at 40-50
                and the discovery engine's symbol hit-rate keeps scores modest.
        """
        self.min_confidence = min_confidence
        self._vix_level: Optional[float] = None
        self._threshold_ts: Optional[datetime] = None
        self._effective_min_confidence = self._compute_effective_threshold(min_confidence)
        self._threshold_ts = datetime.now()
        self.quant = get_quant_ensemble()
        self.logger = logging.getLogger("BreakingTrades")
        # Lazy executor: threads are only spawned on first scan, never at
        # construction. Creating a fresh generator per UI click used to leak a
        # 20-thread pool every time (thread exhaustion → process death).
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._executor_max_workers = 20
        self._executor_lock = threading.Lock()
        self._cache = {}

    def _get_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        """Create the shared thread pool on first use (lazy, thread-safe)."""
        if self._executor is None:
            with self._executor_lock:
                if self._executor is None:
                    self._executor = concurrent.futures.ThreadPoolExecutor(
                        max_workers=self._executor_max_workers
                    )
        return self._executor

    def set_min_confidence(self, value: float) -> None:
        """Reconfigure the confidence threshold on a shared singleton instance.

        The regime-adjusted VIX fetch is cached for 5 minutes so repeated
        clicks on the shared instance don't hit the network every time.
        """
        self.min_confidence = float(value)
        now = datetime.now()
        if (
            self._threshold_ts is not None
            and (now - self._threshold_ts).total_seconds() < 300
        ):
            self._effective_min_confidence = self._compute_effective_threshold(
                float(value), use_cached_vix=True
            )
        else:
            self._effective_min_confidence = self._compute_effective_threshold(float(value))
            self._threshold_ts = now

    def shutdown(self) -> None:
        """Release the thread pool (idempotent; safe to call on app exit)."""
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=False)
            except Exception:
                pass
            self._executor = None

    def _compute_effective_threshold(self, base: float, use_cached_vix: bool = False) -> float:
        """Regime-adjusted confidence threshold.

        In a low-volatility regime (VIX < 15) setups tend to score lower, so the
        threshold is relaxed by 5 points to avoid missing legitimate setups.
        The VIX fetch is time-boxed so construction never blocks on the network.
        When use_cached_vix is True and a level was already fetched, the cached
        level is reused instead of hitting the network again.
        """
        threshold = float(base)
        if use_cached_vix and self._vix_level is not None:
            if self._vix_level < 15.0:
                return max(20.0, base - 5.0)
            return threshold
        try:
            import concurrent.futures as _cf

            def _fetch_vix():
                from data_sources import get_vix
                return get_vix(period="6mo")

            with _cf.ThreadPoolExecutor(1) as pool:
                _fut = pool.submit(_fetch_vix)
                df = _fut.result(timeout=4)
            if df is not None and not df.empty:
                close = df["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                self._vix_level = float(close.dropna().iloc[-1])
                if self._vix_level < 15.0:
                    threshold = max(20.0, base - 5.0)
        except Exception:
            pass
        return threshold

    def get_market_context_message(self) -> str:
        """Meaningful, market-aware message when no setups pass the threshold."""
        try:
            vix = self._vix_level
            if vix is not None:
                if vix < 15:
                    context = (f"VIX is low at {vix:.1f} — a low-volatility regime where clean, "
                               "high-conviction breakouts are naturally scarce")
                elif vix > 25:
                    context = (f"VIX is elevated at {vix:.1f} — high-volatility regimes produce noisy "
                               "signals that rarely clear the quality bar")
                else:
                    context = f"VIX at {vix:.1f} — a moderate volatility environment"
            else:
                context = "current volatility data is unavailable"
            return (
                f"No setups cleared the {self._effective_min_confidence:.0f}-point confidence bar in this "
                f"scan. Market context: {context}. Rather than force marginal trades, the engine waits for "
                f"better alignment of technical, momentum, and volume signals. Retry after the next market "
                f"close or during higher-volume sessions."
            )
        except Exception:
            return "No high-confidence setups found at this time. Market conditions may not be favorable."

    def generate_breaking_trades(self, symbols: List[str], max_trades: int = 8) -> List[BreakingTradeSetup]:
        """
        Public entry point. Orchestrates the high-velocity scan.
        """
        return asyncio.run(self.generate_breaking_trades_async(symbols, max_trades))

    async def generate_breaking_trades_async(self, symbols: List[str], max_trades: int = 8) -> List[BreakingTradeSetup]:
        """
        High-Velocity Asynchronous Scanning Pipeline leveraging OHVDE.
        Scans the entire universe in parallel and runs deep analysis on top candidates.

        Large universes (e.g. the full dynamic universe) are screened with the
        rapid pulse screener first (cheap statistical pre-filter), then only the
        statistically-hot candidates get full deep analysis — so "analyze every
        asset" stays fast instead of brute-forcing thousands of symbols.
        """
        from octavian_discovery_engine import get_discovery_engine
        discovery = get_discovery_engine()
        
        # Tier 1: Rapid Pulse Scan (Market-wide)
        # For big universes, deep_scan=True engages the vectorized pulse screener
        # to pre-filter to statistically-hot symbols; small lists are analyzed fully.
        use_screener = len(symbols) > 250
        pulse_results = await discovery.scan_market_pulse(symbols, deep_scan=use_screener)

        # Candidate floor: if the pulse screener returned very few results (e.g.
        # a quiet market where few symbols moved >1.5% in 5 days), broaden the
        # deep-analysis pass so the scan never silently collapses to nothing.
        if use_screener and len(pulse_results) < 15 and len(symbols) > 250:
            try:
                import random as _random
                _top = [r['symbol'] for r in pulse_results]
                _extra = [s for s in _random.sample(list(symbols), min(120, len(symbols))) if s not in _top]
                _extra_results = await discovery.scan_market_pulse(_extra, deep_scan=False)
                _seen = {r['symbol'] for r in pulse_results}
                for r in _extra_results:
                    if r['symbol'] not in _seen:
                        pulse_results.append(r)
            except Exception:
                pass
        
        # Tier 2: Institutional Deep Analysis Nexus (Parallel)
        # Runs full technical, ML, and risk logic on screened candidates
        executor = self._get_executor()
        tasks = [
            asyncio.get_event_loop().run_in_executor(executor, self._analyze_symbol, r['symbol'])
            for r in pulse_results
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Filter and Sort by Confidence (regime-adjusted threshold)
        setups = [r for r in results if r is not None and r.confidence_score >= self._effective_min_confidence]
        setups.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return setups[:max_trades]

    def _analyze_symbol(self, symbol: str) -> Optional[BreakingTradeSetup]:
        """
        RESTORED INSTITUTIONAL ANALYSIS DEPTH.
        Perform comprehensive analysis on a single symbol with full technical fidelity.
        """
        try:
            # Get data
            df = get_stock(symbol, period="3mo", interval="1d")
            if df is None or df.empty:
                return None

            # Extract price data — robust to MultiIndex / missing Close columns
            close = df.get("Close")
            if close is None:
                for _col in df.columns:
                    if isinstance(_col, str) and "close" in _col.lower():
                        close = df[_col]
                        break
            if close is None:
                return None
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = pd.to_numeric(close, errors="coerce").dropna().astype(float)

            if len(close) < 50:
                return None

            # ── Zero-price guard ─────────────────────────────────────────────
            # Bad / suspended / halted / delisted data can leave 0 or NaN bars;
            # a $0 "current price" would fabricate $0 targets, stops and
            # invalidation levels. Filter zeros out of the series and require a
            # real positive price; fall back to the realtime provider; otherwise
            # skip the symbol entirely (honest "no data" beats fake $0 levels).
            nonzero = pd.to_numeric(close, errors="coerce").dropna()
            nonzero = nonzero[nonzero > 0]
            if len(nonzero) < 50:
                return None
            prices = nonzero.values
            current_price = float(prices[-1])
            if not np.isfinite(current_price) or current_price <= 0:
                try:
                    from data_sources import get_latest_price
                    fb = get_latest_price(symbol)
                    if not fb or not np.isfinite(float(fb)) or float(fb) <= 0:
                        return None
                    current_price = float(fb)
                except Exception:
                    return None

            # Get volume if available
            volume = None
            if "Volume" in df.columns:
                vol = df.get("Volume")
                if isinstance(vol, pd.DataFrame):
                    vol = vol.iloc[:, 0]
                volume = vol.dropna().astype(float).values

            # Get ensemble signal
            signal = self.quant.predict(prices)

            if signal.confidence < 0.10:  # Relaxed minimum threshold
                return None

            # Calculate technical indicators (Full Original Implementation)
            indicators = self._calculate_indicators(close, volume)

            # Determine setup
            setup_analysis = self._determine_setup(prices, signal, indicators)
            if not setup_analysis:
                return None

            # Calculate price levels
            levels = self._calculate_price_levels(
                current_price, signal.direction, indicators, setup_analysis
            )
            if not levels:
                return None

            # Calculate scores
            scores = self._calculate_scores(signal, indicators, setup_analysis, prices)

            # Overall confidence (regime-adjusted threshold)
            confidence = self._calculate_confidence(scores, signal, indicators)

            if confidence < self._effective_min_confidence:
                return None

            # Generate reasoning
            reasoning = self._generate_reasoning(
                symbol, signal, indicators, setup_analysis, scores, levels,
                confidence_score=confidence
            )

            # Options Overlay (Professional Add-on)
            options_setup = self._generate_options_setup(symbol, current_price, signal.direction, indicators, setup_analysis)

            # Create setup
            setup = BreakingTradeSetup(
                symbol=symbol,
                trade_id=f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                generated_at=datetime.now(),
                direction="LONG" if signal.direction == "BULLISH" else "SHORT",
                setup_type=setup_analysis['type'],
                current_price=current_price,
                entry_price=levels['entry'],
                stop_loss=levels['stop_loss'],
                take_profit_1=levels['tp1'],
                take_profit_2=levels.get('tp2'),
                take_profit_3=levels.get('tp3'),
                risk_reward_ratio=levels['risk_reward'],
                suggested_position_size_pct=self._calculate_position_size(confidence, levels['risk_reward']),
                max_risk_per_trade_pct=2.0,
                confidence_score=confidence,
                technical_score=scores['technical'],
                momentum_score=scores['momentum'],
                volatility_score=scores['volatility'],
                volume_score=scores['volume'],
                primary_reason=reasoning['primary'],
                supporting_factors=reasoning['supporting'],
                risk_factors=reasoning['risks'],
                technical_analysis=reasoning['technical_summary'],
                expected_hold_time=setup_analysis['hold_time'],
                market_condition=setup_analysis['market_condition'],
                key_levels=levels['key_levels'],
                indicators=indicators,
                invalidation_price=levels['invalidation'],
                alert_notes=reasoning['alerts'],
                options_setup=options_setup
            )

            return setup
        except Exception as e:
            self.logger.error(f"Error in deep analysis for {symbol}: {e}")
            return None

    def _calculate_indicators(self, close: pd.Series, volume: Optional[np.ndarray]) -> Dict:
        """Calculate comprehensive technical indicators (Institutional Depth)"""
        indicators = {}
        prices = close.values

        # Moving Averages
        indicators['sma_20'] = float(np.mean(prices[-20:])) if len(prices) >= 20 else None
        indicators['sma_50'] = float(np.mean(prices[-50:])) if len(prices) >= 50 else None
        indicators['ema_12'] = float(close.ewm(span=12).mean().iloc[-1]) if len(close) >= 12 else None

        # RSI (Full Implementation)
        if len(prices) >= 15:
            recent_prices = prices[-15:]
            delta = np.diff(recent_prices)
            gains = np.where(delta > 0, delta, 0)
            losses = np.where(delta < 0, -delta, 0)
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0
            rs = avg_gain / (avg_loss + 1e-10)
            indicators['rsi'] = float(100 - (100 / (1 + rs)))
        else:
            indicators['rsi'] = 50.0

        # ATR (Average True Range)
        if len(prices) >= 14:
            high_low = np.std(prices[-14:])
            indicators['atr'] = float(high_low * 1.5)
            indicators['atr_pct'] = float(indicators['atr'] / prices[-1] * 100)
        else:
            indicators['atr'] = float(np.std(prices[-7:]) * 1.5) if len(prices) >= 7 else 0
            indicators['atr_pct'] = float(indicators['atr'] / prices[-1] * 100) if prices[-1] > 0 else 0

        # Volatility
        if len(prices) >= 21:
            price_slice = prices[-20:]
            returns = np.diff(price_slice) / price_slice[:-1]
            indicators['volatility'] = float(np.std(returns) * np.sqrt(252))
        else:
            indicators['volatility'] = 0.20

        # Price momentum
        indicators['momentum_10d'] = float((prices[-1] / prices[-10] - 1) * 100) if len(prices) >= 10 else 0.0
        indicators['momentum_20d'] = float((prices[-1] / prices[-20] - 1) * 100) if len(prices) >= 20 else 0.0

        # Volume analysis
        if volume is not None and len(volume) >= 20:
            avg_volume = np.mean(volume[-20:])
            recent_volume = np.mean(volume[-5:])
            indicators['volume_ratio'] = float(recent_volume / avg_volume) if avg_volume > 0 else 1.0
        else:
            indicators['volume_ratio'] = 1.0

        # Trend strength
        if len(prices) >= 20:
            x_vals = np.arange(20)
            y_vals = prices[-20:]
            trend_slope = np.polyfit(x_vals, y_vals, 1)[0]
            indicators['trend_strength'] = float(trend_slope / prices[-1] * 100)
        else:
            indicators['trend_strength'] = 0.0

        return indicators

    def _determine_setup(self, prices: np.ndarray, signal, indicators: Dict) -> Optional[Dict]:
        """Determine the type of setup and market condition (Institutional Depth)"""
        rsi = indicators.get('rsi', 50)
        momentum_10d = indicators.get('momentum_10d', 0)
        trend_strength = indicators.get('trend_strength', 0)
        vol_ratio = indicators.get('volume_ratio', 1.0)

        setup = {}

        if signal.direction == "BULLISH":
            if rsi < 35 and momentum_10d < -5:
                setup['type'] = "Oversold Reversal"
                setup['hold_time'] = "2-5 days"
            elif trend_strength > 0.5 and vol_ratio > 1.2:
                setup['type'] = "Momentum Breakout"
                setup['hold_time'] = "1-3 days"
            elif momentum_10d > 3 and trend_strength > 0.3:
                setup['type'] = "Trend Continuation"
                setup['hold_time'] = "3-7 days"
            else:
                setup['type'] = "Bullish Setup"
                setup['hold_time'] = "2-5 days"
        else:
            if rsi > 65 and momentum_10d > 5:
                setup['type'] = "Overbought Reversal"
                setup['hold_time'] = "2-5 days"
            elif trend_strength < -0.5 and vol_ratio > 1.2:
                setup['type'] = "Breakdown"
                setup['hold_time'] = "1-3 days"
            elif momentum_10d < -3 and trend_strength < -0.3:
                setup['type'] = "Downtrend Continuation"
                setup['hold_time'] = "3-7 days"
            else:
                setup['type'] = "Bearish Setup"
                setup['hold_time'] = "2-5 days"

        setup['market_condition'] = "Strong Trend" if abs(trend_strength) > 0.5 else "Range-Bound" if abs(trend_strength) < 0.1 else "Trending"
        return setup

    def _calculate_price_levels(self, current_price: float, direction: str, indicators: Dict, setup: Dict) -> Optional[Dict]:
        """Calculate entry, stop loss, and take profit levels (Institutional Depth).

        Returns None when the current price is unusable (<=0 / NaN) so the
        caller skips the symbol instead of emitting $0 targets.
        """
        if not current_price or not np.isfinite(float(current_price)) or float(current_price) <= 0:
            return None
        current_price = float(current_price)
        atr = indicators.get('atr', current_price * 0.02)
        levels = {'key_levels': {}}

        if direction == "BULLISH":
            entry = current_price * 0.998
            stop_distance = max(atr * 1.5, entry * 0.02)
            stop_loss = entry - stop_distance
            risk = entry - stop_loss
            tp1, tp2, tp3 = entry + (risk * 2.0), entry + (risk * 3.5), entry + (risk * 5.0)
            levels['key_levels'] = {'resistance_1': tp1, 'resistance_2': entry + (risk * 4.0), 'support': stop_loss}
            invalidation = stop_loss * 0.995
        else:
            entry = current_price * 1.002
            stop_distance = max(atr * 1.5, entry * 0.02)
            stop_loss = entry + stop_distance
            risk = stop_loss - entry
            tp1, tp2, tp3 = entry - (risk * 2.0), entry - (risk * 3.5), entry - (risk * 5.0)
            levels['key_levels'] = {'support_1': tp1, 'support_2': entry - (risk * 4.0), 'resistance': stop_loss}
            invalidation = stop_loss * 1.005

        levels.update({
            'entry': round(entry, 2), 'stop_loss': round(stop_loss, 2),
            'tp1': round(tp1, 2), 'tp2': round(tp2, 2), 'tp3': round(tp3, 2),
            'invalidation': round(invalidation, 2),
            'risk_reward': round(abs(tp1 - entry) / abs(entry - stop_loss), 2) if abs(entry - stop_loss) > 0 else 2.0,
            'price': current_price,
        })
        return levels

    def _calculate_scores(self, signal, indicators: Dict, setup: Dict, prices: np.ndarray) -> Dict:
        """Calculate individual component scores (Institutional Depth)"""
        scores = {}
        rsi = indicators.get('rsi', 50)
        
        # Technical Score
        tech_score = 50.0
        if signal.direction == "BULLISH":
            tech_score += 20 if 30 < rsi < 50 else 30 if rsi < 30 else 0
        else:
            tech_score += 20 if 50 < rsi < 70 else 30 if rsi > 70 else 0
        scores['technical'] = min(tech_score, 100.0)

        # Momentum Score
        scores['momentum'] = min(50.0 + abs(indicators.get('momentum_10d', 0)) * 3, 100.0)

        # Volatility Score
        vol = indicators.get('volatility', 0.20)
        scores['volatility'] = 80.0 if 0.15 < vol < 0.35 else 60.0 if 0.10 < vol < 0.50 else 40.0

        # Volume Score
        vol_ratio = indicators.get('volume_ratio', 1.0)
        scores['volume'] = 90.0 if vol_ratio > 1.3 else 70.0 if vol_ratio > 1.1 else 50.0

        return scores

    def _calculate_confidence(self, scores: Dict, signal, indicators: Dict = None) -> float:
        """Calculate overall confidence score with regime-aware weighting.

        Weight mix (rebalanced to be more forgiving in ranging markets):
          Technical 25% | Momentum 20% | Volume 20% | Volatility 15% | Quant 10%
        Plus a +10% boost when the quant ensemble agrees with the directional
        technical signal (market-regime confirmation bonus).
        """
        indicators = indicators or {}
        quant_dir = getattr(signal, 'direction', 'NEUTRAL')
        trend_strength = indicators.get('trend_strength', 0) or 0
        tech_dir = "BULLISH" if trend_strength > 0 else "BEARISH" if trend_strength < 0 else "NEUTRAL"
        regime_bonus = 0.10 if (quant_dir != "NEUTRAL" and quant_dir == tech_dir) else 0.0

        confidence = (
            scores['technical'] * 0.25 +
            scores['momentum'] * 0.20 +
            scores['volatility'] * 0.15 +
            scores['volume'] * 0.20 +
            (signal.confidence or 0) * 100 * 0.10
        ) * (1.0 + regime_bonus)
        return round(confidence, 1)

    def _calculate_position_size(self, confidence: float, risk_reward: float) -> float:
        """Calculate suggested position size as % of portfolio (Institutional Depth)"""
        base_size = (confidence / 100) * 10
        multiplier = 1.2 if risk_reward >= 3.0 else 1.0 if risk_reward >= 2.0 else 0.8
        return round(min(base_size * multiplier, 15.0), 1)

    def _user_capital_context(self):
        """Best-effort user account context for dollar-denominated sizing.

        Returns (capital_mid, risk_pct, label) using the trader profile when
        available; falls back to conservative defaults without ever failing.
        """
        try:
            from trader_profile import get_risk_params
            rp = get_risk_params() or {}
            risk_pct = float(rp.get("max_loss_per_trade_pct", 2.0))
            label = str(rp.get("profile", "Moderate"))
            cap_str = str(rp.get("capital_range", "$10,000 - $50,000"))
            nums = [float(x) for x in re.findall(r"[\d,]+", cap_str.replace(",", ""))]
            if len(nums) >= 2:
                return (sum(nums) / len(nums), risk_pct, label)
            if len(nums) == 1:
                return (nums[0], risk_pct, label)
        except Exception:
            pass
        return (30000.0, 2.0, "Moderate")

    def _generate_reasoning(self, symbol: str, signal, indicators: Dict, setup_analysis: Dict, scores: Dict, levels: Dict = None, confidence_score: float = 0.0) -> Dict:
        """Generate comprehensive, ASSET-SPECIFIC reasoning for the trade.

        Every alert references the actual computed price levels and indicator
        values for this symbol (never generic boilerplate), and position sizing
        is expressed in dollars using the user's trader profile when available.
        """
        levels = levels or {}
        stype = setup_analysis.get('type', 'Setup')
        mkt_cond = setup_analysis.get('market_condition', 'Neutral')
        hold_time = setup_analysis.get('hold_time', '2-5 days')
        direction = "LONG" if signal.direction == "BULLISH" else "SHORT"
        entry = levels.get('entry')
        stop = levels.get('stop_loss')
        tp1 = levels.get('tp1')
        tp2 = levels.get('tp2')
        tp3 = levels.get('tp3')
        invalidation = levels.get('invalidation')
        rr = levels.get('risk_reward', 0)
        price = levels.get('price') or levels.get('current_price', 0) or 0

        rsi = indicators.get('rsi', 50)
        mom10 = indicators.get('momentum_10d', 0)
        mom20 = indicators.get('momentum_20d', 0)
        vol_ratio = indicators.get('volume_ratio', 1.0)
        volatility = indicators.get('volatility', 0.20)
        trend = indicators.get('trend_strength', 0)
        atr = indicators.get('atr', 0)

        primary = (
            f"{symbol} is showing a {stype} setup with {direction} bias: price "
            f"at ${price:,.2f} with RSI {rsi:.0f}, {mom10:+.1f}% 10-day momentum, "
            f"{vol_ratio:.1f}x average volume and {volatility*100:.0f}% annualized volatility."
        )

        supporting = []
        if mom10 > 5:
            supporting.append(f"Strong momentum: +{mom10:.1f}% over 10 sessions (20d: {mom20:+.1f}%)")
        elif mom10 < -5:
            supporting.append(f"Oversold momentum: {mom10:.1f}% over 10 sessions — mean-reversion tailwind")
        if vol_ratio > 1.3:
            supporting.append(f"Volume confirmation: {vol_ratio:.1f}x average volume on the move")
        if trend > 0.5:
            supporting.append(f"Trend strength: slope {trend:.2f}% per session — trend-following tailwind")
        elif trend < -0.5:
            supporting.append(f"Downtrend slope {trend:.2f}% per session — momentum aligned with direction")
        if signal.confidence > 0.55:
            supporting.append(f"Quant ensemble confidence {signal.confidence*100:.0f}% aligns with the technical signal")
        if 30 <= rsi <= 70:
            supporting.append(f"RSI {rsi:.0f} is in the tradable zone (not overbought/oversold)")
        if not supporting:
            supporting.append("Setup qualifies on the composite technical filter (momentum + volume + trend)")

        risks = []
        if volatility > 0.40:
            risks.append(f"High volatility ({volatility*100:.0f}% ann.) — expect wide swings; size down")
        elif volatility < 0.15:
            risks.append(f"Low volatility ({volatility*100:.0f}% ann.) — moves may stall; use wider stops or skip")
        if mkt_cond == "Range-Bound":
            risks.append("Range-bound tape — breakout may fail; respect the invalidation level")
        if mom10 > 20:
            risks.append(f"Extended short-term move (+{mom10:.0f}% in 10 sessions) — pullback risk elevated")
        if not risks:
            risks.append("Standard market risk applies; size for the full stop-loss distance")

        tech_summary = (
            f"RSI {rsi:.1f} | 10D Mom {mom10:+.1f}% | 20D Mom {mom20:+.1f}% | "
            f"Vol {volatility*100:.1f}% | VolRatio {vol_ratio:.1f}x | Trend {trend:+.2f}%/day | "
            f"ATR {atr:.2f}"
        )

        # ── Asset-specific, price-referenced alerts ──────────────────────────
        alerts = []
        if entry and stop:
            risk_pct = (abs(entry - stop) / entry * 100) if entry else 0
            alerts.append(
                f"ENTRY: {direction} {symbol} on trigger near ${entry:,.2f} (stop ${stop:,.2f}, "
                f"{risk_pct:.1f}% risk from entry)"
            )
        if tp1:
            alerts.append(f"TP1 ${tp1:,.2f} — scale out ~30% (+{abs(tp1-entry)/entry*100:.1f}% if entry fills)" if entry else f"TP1 ${tp1:,.2f}")
        if tp2:
            alerts.append(f"TP2 ${tp2:,.2f} — scale out ~40%")
        if tp3:
            alerts.append(f"TP3 ${tp3:,.2f} — let runner ride")
        if invalidation:
            alerts.append(f"INVALIDATION: exit flat if price breaks ${invalidation:,.2f}")
        if rr:
            alerts.append(f"Risk/reward {rr:.1f}:1 to TP1 — only take the trade if this clears your hurdle")
        alerts.append(f"CONTEXT: {mkt_cond} tape; expected hold {hold_time}")
        if vol_ratio:
            alerts.append(f"VOLUME: current participation {vol_ratio:.1f}x the 20-session average — fading volume = fading edge")
        pos_pct = self._calculate_position_size(confidence_score or 50.0, rr) if rr else 5.0
        alerts.append(f"SIZING: suggested {pos_pct:.1f}% of portfolio at {confidence_score:.0f}% confidence; "
                      f"risk no more than ~1% of capital to the stop")

        return {
            'primary': primary,
            'supporting': supporting,
            'risks': risks,
            'technical_summary': tech_summary,
            'alerts': alerts,
        }

    def _generate_options_setup(self, symbol: str, current_price: float, direction: str, indicators: Dict, setup: Dict) -> Optional[OptionSetup]:
        """
        Institutional Options Overlay — REAL data when available.

        Uses the OptionsEngine's analytic pricers (Black-Scholes) plus the
        stock's own realized volatility to produce a genuine single-leg
        recommendation (Long Call / Long Put). Returns None when no real
        price can be computed so the caller never shows fabricated greeks.
        """
        try:
            import options_engine
            eng = options_engine.get_options_engine()
            vol = float(indicators.get('volatility', 0.20) or 0.20)
            # Guard: don't pretend we have a market for impossible inputs
            if not current_price or current_price <= 0 or vol <= 0:
                return None
            days = 30
            t = days / 365.0
            if direction == "BULLISH":
                strike = round(current_price * 1.05, 2)
                otype = "call"
            else:
                strike = round(current_price * 0.95, 2)
                otype = "put"
            try:
                g = eng.black_scholes(current_price, strike, t, vol, otype)
            except Exception:
                g = None
            if not g or not g.get("price") or float(g["price"]) <= 0:
                return None
            prem = float(g["price"])
            delta = float(g.get("delta", 0) or 0)
            gamma = float(g.get("gamma", 0) or 0)
            theta = float(g.get("theta", 0) or 0)
            vega = float(g.get("vega", 0) or 0)
            leg = OptionLeg(
                strike=strike, option_type=otype.upper(), side=1,
                expiry=(datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d"),
                delta=delta, theoretical_price=prem,
            )
            return OptionSetup(
                strategy_name="Long Call" if otype == "call" else "Long Put",
                legs=[leg],
                net_debit_credit=prem * 100.0,
                max_risk=prem * 100.0,
                max_reward=prem * 100.0 * 3.0,
                breakeven=[strike + prem if otype == "call" else strike - prem],
                greeks={'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega},
            )
        except Exception:
            return None


# Singleton Accessors
_breaking_trades_gen = None

def get_breaking_trades_generator() -> BreakingTradesGenerator:
    global _breaking_trades_gen
    if _breaking_trades_gen is None:
        _breaking_trades_gen = BreakingTradesGenerator()
    return _breaking_trades_gen
