"""
Octavian Unbiased Market Analyzer
Pure profit-maximizing analysis without bias towards well-known tickers

This module provides completely unbiased market analysis focused solely on
maximizing trader profits through pure model conclusions and data-driven insights.

Author: APB - Octavian Team
"""

import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import requests
from dataclasses import dataclass, field
import asyncio
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

from data_sources import get_stock, get_fx, get_futures_proxy, get_options_chain

# Dynamic ticker universe
try:
    from ticker_universe import get_ticker_universe
    HAS_UNIVERSE = True
except ImportError:
    HAS_UNIVERSE = False

# Quant Ensemble integration
try:
    from quant_ensemble_model import get_quant_ensemble
    HAS_QUANT_ENSEMBLE = True
except ImportError:
    HAS_QUANT_ENSEMBLE = False

# Dynamic information weighting integration
try:
    from information_weighting_engine import get_weighting_engine
    HAS_WEIGHTING = True
except ImportError:
    HAS_WEIGHTING = False

# Safe comprehensive universe check — never crashes
def _has_comprehensive_universe() -> bool:
    try:
        from comprehensive_ticker_universe import get_comprehensive_universe
        return get_comprehensive_universe() is not None
    except Exception:
        return False

def _get_comprehensive_universe_safe():
    try:
        from comprehensive_ticker_universe import get_comprehensive_universe
        return get_comprehensive_universe()
    except Exception:
        return None

@dataclass
class UnbiasedAnalysis:
    """Pure unbiased analysis result focused on profit maximization."""
    symbol: str
    profit_probability: float
    expected_return: float
    risk_adjusted_return: float
    confidence_score: float
    entry_signals: List[str]
    exit_signals: List[str]
    risk_factors: List[str]
    profit_catalysts: List[str]
    model_reasoning: List[str]
    raw_data_insights: Dict[str, Any]
    timestamp: datetime

    # ── Bull / Bear / Neutral case reasoning (always generated) ─────────
    bull_reasoning: List[str] = field(default_factory=list)
    bear_reasoning: List[str] = field(default_factory=list)
    neutral_reasoning: str = ""

class UnbiasedMarketAnalyzer:
    """Completely unbiased market analyzer focused on pure profit maximization."""
    
    def __init__(self):
        self.bias_filters = []
        self.profit_focus_weights = {
            'momentum': 0.20,
            'volatility_opportunity': 0.15,
            'volume_anomalies': 0.10,
            'price_inefficiencies': 0.10,
            'technical_breakouts': 0.10,
            'statistical_edges': 0.10,
            'quant_ensemble': 0.25,
        }  # Legacy attribute kept for reference; probability blending now uses InformationWeightingEngine
        
        # Universe expansion — safe pattern, never crashes
        self._comprehensive_universe = _get_comprehensive_universe_safe()
        if self._comprehensive_universe is not None:
            self.asset_universe = self._build_comprehensive_universe_v2()
        else:
            self._universe = get_ticker_universe() if HAS_UNIVERSE else None
            self.asset_universe = self._build_comprehensive_universe()
        
        # Initialize quant ensemble
        self._quant_ensemble = None
        if HAS_QUANT_ENSEMBLE:
            try:
                self._quant_ensemble = get_quant_ensemble()
            except Exception:
                pass
                
        # Dynamic information weighting engine (regime/asset-class-aware)
        self._weighting_engine = None
        if HAS_WEIGHTING:
            try:
                from information_weighting_engine import get_weighting_engine
                self._weighting_engine = get_weighting_engine()
            except Exception as e:
                print(f"Failed to load information weighting engine: {e}")

        # Cached news processor (heavy to construct — reuse across calls)
        self._news_processor = None

        # Runtime state
        self._vix_cache: Optional[float] = None
        self._vix_ts: float = 0.0
        self._last_data_is_proxy: bool = False
        self._last_proxy_for: Optional[str] = None
        self._last_weights: Optional[Dict[str, float]] = None
        self._last_asset_class: str = "stock"
        self._last_regime: str = "ranging"

    def _build_comprehensive_universe_v2(self) -> Dict[str, List[str]]:
        """Build comprehensive universe using 10,000+ ticker source."""
        u = self._comprehensive_universe
        return {
            'stocks': u.get_stocks()[:8000],
            'etfs': u.get_etfs(),
            'options': self._get_options_universe(),
            'forex': u.get_forex(),
            'futures': u.get_futures(),
            'crypto': u.get_crypto(),
            'commodities': self._get_commodities_universe(),
            'bonds': self._get_bonds_universe(),
            'international': u.get_category('international_adrs') or self._get_international_universe(),
        }

    def _build_comprehensive_universe(self) -> Dict[str, List[str]]:
        """Build comprehensive universe of ALL tradeable assets."""
        if self._universe:
            return {
                'stocks': self._universe.get_all_stocks(),
                'etfs': self._universe.get_etfs(),
                'options': self._get_options_universe(),
                'forex': self._universe.get_forex(),
                'futures': self._universe.get_futures(),
                'crypto': self._universe.get_crypto(),
                'commodities': self._get_commodities_universe(),
                'bonds': self._get_bonds_universe(),
                'international': self._universe.get_sector_tickers("international") or self._get_international_universe(),
            }
        return {
            'stocks': self._get_all_stocks(),
            'etfs': self._get_all_etfs(),
            'options': self._get_options_universe(),
            'forex': self._get_forex_universe(),
            'futures': self._get_futures_universe(),
            'crypto': self._get_crypto_universe(),
            'commodities': self._get_commodities_universe(),
            'bonds': self._get_bonds_universe(),
            'international': self._get_international_universe(),
        }
    
    def _get_all_stocks(self) -> List[str]:
        """Delegate to centralized ticker universe."""
        if self._universe:
            return self._universe.get_all_stocks()
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'JPM', 'V',
            'UNH', 'LLY', 'JNJ', 'XOM', 'MA', 'PG', 'HD', 'MRK', 'ABBV', 'CVX',
            'CRM', 'AMD', 'INTC', 'NFLX', 'ADBE', 'PEP', 'KO', 'COST', 'TMO', 'ABT',
            'NKE', 'MCD', 'DIS', 'CMCSA', 'WMT', 'CAT', 'DE', 'BA', 'GE', 'RTX',
            'SPY', 'QQQ', 'IWM', 'GLD', 'TLT',
        ]

    def _get_all_etfs(self) -> List[str]:
        if self._universe:
            return self._universe.get_etfs()
        return ['SPY', 'QQQ', 'IWM', 'VTI', 'GLD', 'SLV', 'TLT']

    def _get_options_universe(self) -> List[str]:
        """Get options-enabled symbols."""
        return ['SPY', 'QQQ', 'IWM', 'AAPL', 'TSLA', 'NVDA', 'AMD', 'AMZN']
    
    def _get_forex_universe(self) -> List[str]:
        if self._universe:
            return self._universe.get_forex()
        return ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD']

    def _get_futures_universe(self) -> List[str]:
        if self._universe:
            return self._universe.get_futures()
        return ['ES=F', 'NQ=F', 'CL=F', 'GC=F', 'SI=F']

    def _get_crypto_universe(self) -> List[str]:
        if self._universe:
            return self._universe.get_crypto()
        return ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD']

    def _get_commodities_universe(self) -> List[str]:
        """Get commodity-related instruments."""
        return ['GLD', 'SLV', 'USO', 'UNG', 'DBA', 'DBC', 'PDBC', 'GSG']
    
    def _get_bonds_universe(self) -> List[str]:
        """Get bond ETFs and instruments."""
        return ['TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'JNK', 'EMB', 'TIP']
    
    def _get_international_universe(self) -> List[str]:
        """Get international opportunities."""
        return ['FXI', 'EWJ', 'EWZ', 'EWY', 'INDA', 'EEM', 'VEA', 'VWO']
    
    async def analyze_unbiased(self, symbol: str, timeframe: str = '1y') -> UnbiasedAnalysis:
        """Perform completely unbiased analysis — now includes quant ensemble."""
        try:
            self.current_symbol = symbol
            data = await self._fetch_raw_data(symbol, timeframe)
            
            # Save for downstream modules (like ai_chatbot chart generation)
            self._last_fetched_data = data

            if data is None or data.empty:
                return self._create_null_analysis(symbol)

            if 'Returns' not in data.columns:
                data['Returns'] = data['Close'].pct_change()
            
            profit_signals = self._calculate_profit_signals(data, fast_mode=False)
            
            #  QUANT ENSEMBLE INTEGRATION 
            quant_signal = None
            if self._quant_ensemble is not None:
                try:
                    close = data['Close']
                    if isinstance(close, pd.DataFrame):
                        close = close.iloc[:, 0]
                    prices = close.dropna().astype(float).values
                    
                    volumes = None
                    if 'Volume' in data.columns:
                        v = data['Volume']
                        if isinstance(v, pd.DataFrame):
                            v = v.iloc[:, 0]
                        volumes = v.dropna().astype(float).values
                    
                    if len(prices) >= 40:
                        quant_signal = self._quant_ensemble.predict(prices, volumes)
                        profit_signals['quant_probability'] = quant_signal.probability
                        profit_signals['quant_confidence'] = quant_signal.confidence
                        profit_signals['quant_expected_return'] = quant_signal.expected_return
                except Exception as e:
                    pass

            risk_metrics = self._calculate_pure_risk_metrics(data)
            momentum_analysis = self._analyze_momentum_unbiased(data)
            volatility_opportunities = self._identify_volatility_opportunities(data)
            options_metrics = self._calculate_options_metrics(symbol, data) # NEW
            statistical_edges = self._find_statistical_edges(data)
            
            # ── Multi-factor signal context (news, RSI, dynamic weights) ──
            rsi_value = self._calculate_rsi(data['Close'])
            news_sentiment = self._fetch_news_sentiment(symbol)  # None if unavailable

            weights, asset_class, regime = self._get_dynamic_weights(momentum_analysis)
            self._last_weights = weights
            self._last_asset_class = asset_class
            self._last_regime = regime

            # Engine-weighted probability (quant + news + options all inside the blend)
            profit_probability = self._calculate_profit_probability(
                profit_signals, momentum_analysis, volatility_opportunities, statistical_edges,
                options_metrics, quant_signal, news_sentiment, weights
            )

            # ── Always generate all three cases regardless of direction ──
            bull_reasoning, bear_reasoning, neutral_reasoning = self._generate_bull_bear_neutral(
                symbol, profit_signals, momentum_analysis, volatility_opportunities,
                options_metrics, quant_signal, news_sentiment, rsi_value, data
            )

            expected_return = self._calculate_expected_return(data, profit_signals)
            risk_adjusted_return = expected_return / max(risk_metrics['volatility'], 0.01)
            confidence_score = self._calculate_model_confidence(data, profit_signals)
            
            # Boost confidence if quant model agrees
            if quant_signal is not None and quant_signal.confidence > 0.4:
                base_dir = "BULLISH" if profit_probability > 0.55 else "BEARISH" if profit_probability < 0.45 else "NEUTRAL"
                if base_dir == quant_signal.direction:
                    confidence_score = min(0.95, confidence_score * 1.15)

            entry_signals = self._generate_entry_signals(profit_signals, momentum_analysis)
            exit_signals = self._generate_exit_signals(profit_signals, risk_metrics)
            risk_factors = self._identify_risk_factors(data, risk_metrics)
            profit_catalysts = self._identify_profit_catalysts(data, profit_signals)
            model_reasoning = self._generate_model_reasoning(
                profit_signals, momentum_analysis, volatility_opportunities
            )
            
            # Add quant reasoning
            if quant_signal is not None:
                model_reasoning.append(f"* QUANT ENSEMBLE: {quant_signal.direction} "
                                      f"(prob={quant_signal.probability:.1%}, conf={quant_signal.confidence:.1%})")
                model_reasoning.append(f"  Driving Factor: {quant_signal.driving_factor}")
            
            raw_data_insights = {
                'price_momentum': momentum_analysis,
                'volatility_metrics': risk_metrics,
                'volume_analysis': self._analyze_volume_patterns(data),
                'statistical_properties': statistical_edges,
                'market_microstructure': self._analyze_microstructure(data),
                'options_intelligence': options_metrics, # NEW
                'rsi': rsi_value,
                'news_sentiment': (
                    {'score': news_sentiment, 'note': 'Best-effort headline sentiment [-1, 1]'}
                    if news_sentiment is not None
                    else {'score': 0.0, 'note': 'News sentiment unavailable (no matching headlines)'}
                ),
                'information_weights': {
                    'asset_class': asset_class,
                    'market_regime': regime,
                    'weights': weights,
                },
                'data_provenance': {
                    'is_proxy': bool(getattr(self, '_last_data_is_proxy', False)),
                    'proxy_for': getattr(self, '_last_proxy_for', None),
                    'note': (
                        f"Data sourced from ETF proxy ({self._last_proxy_for}) — treated as proxy, "
                        "not direct futures data."
                        if getattr(self, '_last_data_is_proxy', False)
                        else "Direct instrument data"
                    ),
                },
            }
            if quant_signal:
                raw_data_insights['quant_ensemble'] = {
                    'direction': quant_signal.direction,
                    'probability': quant_signal.probability,
                    'confidence': quant_signal.confidence,
                    'sub_models': quant_signal.sub_model_signals,
                    'risk_metrics': quant_signal.risk_metrics,
                }
            
            analysis = UnbiasedAnalysis(
                symbol=symbol,
                profit_probability=profit_probability,
                expected_return=expected_return,
                risk_adjusted_return=risk_adjusted_return,
                confidence_score=confidence_score,
                entry_signals=entry_signals,
                exit_signals=exit_signals,
                risk_factors=risk_factors,
                profit_catalysts=profit_catalysts,
                model_reasoning=model_reasoning,
                raw_data_insights=raw_data_insights,
                timestamp=datetime.now(),
                bull_reasoning=bull_reasoning,
                bear_reasoning=bear_reasoning,
                neutral_reasoning=neutral_reasoning,
            )
            
            # Add signal_direction for chatbot compatibility
            if profit_probability > 0.55:
                analysis.signal_direction = 'BULLISH'
            elif profit_probability < 0.45:
                analysis.signal_direction = 'BEARISH'
            else:
                analysis.signal_direction = 'NEUTRAL'
            
            return analysis
            
        except Exception as e:
            print(f"Unbiased analysis error for {symbol}: {e}")
            return self._create_null_analysis(symbol)
    
    async def _fetch_raw_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Fetch raw market data without any preprocessing bias."""
        try:
            # Forex: EUR/USD, USD/JPY etc.
            if '/' in symbol:
                fx_symbol = symbol.replace('/', '_')
                df = get_fx(fx_symbol)
                if df is not None and not df.empty:
                    return df
                # Fallback: try Yahoo format EURUSD=X
                yf_sym = symbol.replace('/', '') + '=X'
                return get_stock(yf_sym, period=timeframe)
            if symbol.endswith('=X'):
                # e.g. USDJPY=X → USD_JPY
                base = symbol.replace('=X', '')
                if len(base) == 6:
                    fx_key = base[:3] + '_' + base[3:]
                    df = get_fx(fx_key)
                    if df is not None and not df.empty:
                        return df
                return get_stock(symbol, period=timeframe)
            if '=F' in symbol:
                return self._fetch_futures_data(symbol, timeframe)
            return get_stock(symbol, period=timeframe)
            
        except Exception as e:
            print(f"Data fetch error for {symbol}: {e}")
            return None

    # ── Futures-aware data fetching ──────────────────────────────────────
    def _fetch_futures_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Fetch futures OHLCV data.

        Order of preference:
          1. Direct yfinance futures contract (e.g. CL=F, period='1y')
          2. get_futures_proxy (direct first, then ETF proxy fallback)

        When ETF-proxy data is used, the proxy flag is set so downstream
        consumers (chatbot, UI) can label it accurately instead of treating
        it as direct futures data.
        """
        self._last_data_is_proxy = False
        self._last_proxy_for = None
        try:
            df = get_stock(symbol, period='1y')
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
        try:
            df = get_futures_proxy(symbol, period=timeframe)
            if df is not None and not df.empty:
                self._last_data_is_proxy = True
                self._last_proxy_for = symbol
            return df
        except Exception as e:
            print(f"Futures data fetch error for {symbol}: {e}")
            return None

    def _get_vix_level(self) -> Optional[float]:
        """Best-effort current VIX level, cached for 15 minutes."""
        now = time.time()
        if self._vix_cache is not None and now - self._vix_ts < 900:
            return self._vix_cache
        try:
            from data_sources import get_vix
            df = get_vix(period="6mo")
            if df is not None and not df.empty:
                close = df['Close']
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                self._vix_cache = float(close.dropna().iloc[-1])
                self._vix_ts = now
                return self._vix_cache
        except Exception:
            pass
        return None

    def _get_market_bias(self) -> Optional[str]:
        """Live market bias (BULLISH/BEARISH/NEUTRAL) from the adaptive engine."""
        try:
            from adaptive_reasoning_engine import get_adaptive_engine
            regime = getattr(get_adaptive_engine(), 'market_regime', 'NEUTRAL')
            if regime == 'BULL_TREND':
                return 'BULLISH'
            if regime == 'BEAR_TREND':
                return 'BEARISH'
            return 'NEUTRAL'
        except Exception:
            return None

    def _get_dynamic_weights(self, momentum_analysis: Optional[Dict] = None) -> Tuple[Dict[str, float], str, str]:
        """Return (weights, asset_class, regime) from the InformationWeightingEngine."""
        fallback = {'price_momentum': 0.25, 'technical_signals': 0.25, 'quant_model': 0.25,
                    'volume_confirmation': 0.10, 'options_flow': 0.05, 'macro_regime': 0.05,
                    'news_sentiment': 0.05}
        if self._weighting_engine is None:
            return fallback, 'stock', 'ranging'
        try:
            asset_class = self._weighting_engine.infer_asset_class(getattr(self, 'current_symbol', '') or '')
            mom20 = None
            if momentum_analysis:
                v = momentum_analysis.get('20d')
                if isinstance(v, (int, float)):
                    mom20 = float(v) * 100.0
            regime = self._weighting_engine.infer_regime_from_market(
                vix=self._get_vix_level(),
                market_bias=self._get_market_bias(),
                price_momentum_20d=mom20,
            )
            return self._weighting_engine.get_weights(asset_class, regime), asset_class, regime
        except Exception:
            return fallback, 'stock', 'ranging'

    def _calculate_rsi(self, close) -> float:
        """Wilder-style RSI (14) from a close series; 50.0 on any failure."""
        try:
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.dropna().astype(float)
            if len(close) < 16:
                return 50.0
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
            rs = gain / (loss + 1e-9)
            rsi = 100.0 - (100.0 / (1.0 + rs))
            val = float(rsi.iloc[-1])
            return 50.0 if (np.isnan(val) or np.isinf(val)) else val
        except Exception:
            return 50.0

    def _fetch_news_sentiment(self, symbol: str) -> Optional[float]:
        """Best-effort weighted news sentiment in [-1, 1]. Returns None if unavailable.

        Uses the AdvancedNewsProcessor's per-symbol sentiment analyzer, which
        returns a credibility-weighted score within a hard timeout. The
        processor instance is cached to avoid repeated heavy construction.
        """
        try:
            if self._news_processor is None:
                from advanced_news_processor import AdvancedNewsProcessor
                self._news_processor = AdvancedNewsProcessor()
            result = self._news_processor.analyze_symbol_sentiment(symbol, timeout=5)
            if result and 'score' in result:
                score = result.get('score')
                if isinstance(score, (int, float)):
                    return float(np.clip(score, -1.0, 1.0))
        except Exception:
            pass
        return None

    def _generate_bull_bear_neutral(self, symbol: str, profit_signals: Dict[str, float],
                                    momentum_analysis: Dict[str, float],
                                    volatility_opportunities: Dict[str, float],
                                    options_metrics: Dict[str, float],
                                    quant_signal: Any, news_sentiment: Optional[float],
                                    rsi: float, data: pd.DataFrame) -> Tuple[List[str], List[str], str]:
        """Always produce bull, bear, and neutral cases with concrete data points.

        Sources: momentum direction, RSI zone, volume trend, quant signal
        direction, news sentiment sign, and options skew.
        """
        bull: List[str] = []
        bear: List[str] = []

        mom = momentum_analysis or {}
        mom_values = [v for v in mom.values() if isinstance(v, (int, float))]
        mom_score = float(np.mean(mom_values)) if mom_values else 0.0
        mom20 = mom.get('20d', 0) or 0
        mom5 = mom.get('5d', 0) or 0

        # Momentum direction
        if isinstance(mom20, (int, float)) and mom20 > 0.02:
            bull.append(f"20-day momentum is positive (+{mom20*100:.1f}%) — trend tailwind")
        elif isinstance(mom20, (int, float)) and mom20 < -0.02:
            bear.append(f"20-day momentum is negative ({mom20*100:.1f}%) — trend headwind")
        if isinstance(mom5, (int, float)) and mom5 > 0.01:
            bull.append(f"5-day momentum positive (+{mom5*100:.1f}%) — near-term buying pressure")
        elif isinstance(mom5, (int, float)) and mom5 < -0.01:
            bear.append(f"5-day momentum negative ({mom5*100:.1f}%) — near-term selling pressure")

        # RSI zone
        if rsi >= 70:
            bear.append(f"RSI at {rsi:.0f} — overbought, extension / mean-reversion pullback risk")
        elif rsi <= 30:
            bull.append(f"RSI at {rsi:.0f} — oversold, bounce setup with favorable entry")
        elif rsi >= 55:
            bull.append(f"RSI at {rsi:.0f} — bullish momentum zone (not yet overbought)")
        elif rsi <= 45:
            bear.append(f"RSI at {rsi:.0f} — bearish momentum zone")

        # Volume trend
        vol_analysis = self._analyze_volume_patterns(data)
        if vol_analysis.get('available'):
            vol_ratio = vol_analysis.get('current_vs_average', 1.0) or 1.0
            vol_trend = vol_analysis.get('trend')
            if vol_ratio > 1.2 and mom_score > 0:
                bull.append(f"Volume at {vol_ratio:.1f}x average — participation confirms the move")
            elif vol_ratio > 1.2 and mom_score < 0:
                bear.append(f"Volume at {vol_ratio:.1f}x average — heavy distribution during the decline")
            elif vol_trend == 'decreasing' and mom_score > 0:
                bear.append("Volume contracting — rally lacks conviction")

        # Quant ensemble signal
        if quant_signal is not None:
            q_dir = getattr(quant_signal, 'direction', 'NEUTRAL')
            q_prob = getattr(quant_signal, 'probability', 0.5) or 0.5
            q_conf = getattr(quant_signal, 'confidence', 0.5) or 0.5
            if q_dir == 'BULLISH':
                bull.append(f"Quant ensemble BULLISH ({q_prob:.0%} prob, {q_conf:.0%} conf)")
            elif q_dir == 'BEARISH':
                bear.append(f"Quant ensemble BEARISH ({q_prob:.0%} prob, {q_conf:.0%} conf)")

        # News sentiment sign
        if news_sentiment is not None and abs(news_sentiment) > 0.15:
            if news_sentiment > 0:
                bull.append(f"News sentiment positive ({news_sentiment:+.2f}) — headlines supportive")
            else:
                bear.append(f"News sentiment negative ({news_sentiment:+.2f}) — headlines pressuring")

        # Options skew
        if options_metrics:
            cpr = options_metrics.get('net_call_put_ratio', 1.0) or 1.0
            if cpr > 1.3:
                bull.append(f"Options flow call-heavy ({cpr:.2f} call/put) — leaning bullish")
            elif cpr < 0.7:
                bear.append(f"Options flow put-heavy ({cpr:.2f} call/put) — hedging demand elevated")

        # Volatility context
        vol_vals = [v for v in volatility_opportunities.values() if isinstance(v, (int, float))]
        vol_score = float(np.mean(vol_vals)) if vol_vals else 0.0
        if vol_score > 1.2:
            bear.append(f"Volatility expansion ({vol_score:.2f}x) — larger adverse moves possible")
        elif 0 < vol_score < 0.8:
            bull.append(f"Compressed volatility ({vol_score:.2f}x) — positioning for expansion")

        if not bull:
            bull.append("No dominant bullish catalysts identified — upside needs fresh momentum")
        if not bear:
            bear.append("No dominant bearish catalysts identified — downside limited absent a shock")

        # Neutral explanation
        neutral_bits = []
        if abs(mom_score) < 0.05:
            neutral_bits.append("momentum is broadly flat")
        if 45 <= rsi <= 55:
            neutral_bits.append("RSI sits in the neutral zone")
        if not neutral_bits:
            neutral_bits.append("signals are mixed across timeframes")
        neutral_reasoning = (
            f"The setup for {symbol} is balanced: " + ", ".join(neutral_bits)
            + ", so direction is not yet resolved. Position sizing should stay defensive."
        )

        return bull[:5], bear[:5], neutral_reasoning
    
    def _calculate_profit_signals(self, data: pd.DataFrame, fast_mode: bool = False) -> Dict[str, float]:
        """Calculate pure profit signals based on mathematical patterns."""
        signals = {}
        
        try:
            # Momentum signals
            signals['price_momentum'] = self._calculate_momentum_score(data)
            signals['volume_momentum'] = self._calculate_volume_momentum(data)
            signals['volatility_momentum'] = self._calculate_volatility_momentum(data)
            
            # Mean reversion signals
            signals['mean_reversion'] = self._calculate_mean_reversion_score(data)
            signals['bollinger_position'] = self._calculate_bollinger_position(data)
            
            # Breakout signals
            signals['breakout_probability'] = self._calculate_breakout_probability(data)
            signals['volume_breakout'] = self._calculate_volume_breakout_score(data)
            
            # Statistical arbitrage signals
            signals['statistical_edge'] = self._calculate_statistical_edge(data)
            signals['autocorrelation'] = self._calculate_autocorrelation_score(data)
            
            # Skip heavy ensemble engine to avoid hangs — only use if explicitly not fast_mode
            if not fast_mode:
                try:
                    import concurrent.futures
                    def _try_ensemble():
                        from advanced_ml_engine import get_ensemble_engine
                        engine = get_ensemble_engine()
                        return engine.analyze_symbol_ensemble(
                            data,
                            self.current_symbol if hasattr(self, 'current_symbol') else 'UNKNOWN',
                            fast_mode=True
                        )
                    with concurrent.futures.ThreadPoolExecutor(1) as pool:
                        future = pool.submit(_try_ensemble)
                        ensemble_results = future.result(timeout=5)
                    signals['ensemble_decision'] = 1.0 if ensemble_results['decision'] == 'BUY' else -1.0 if ensemble_results['decision'] == 'SELL' else 0.0
                    signals['ensemble_confidence'] = ensemble_results['confidence']
                    signals['ensemble_predicted_return'] = ensemble_results['predicted_return']
                    self.last_ensemble_results = ensemble_results
                except Exception:
                    signals['ensemble_decision'] = 0.0
                    signals['ensemble_confidence'] = 0.0
                    self.last_ensemble_results = {}
            else:
                signals['ensemble_decision'] = 0.0
                signals['ensemble_confidence'] = 0.0
                self.last_ensemble_results = {}

            # Microstructure signals
            signals['bid_ask_pressure'] = self._estimate_bid_ask_pressure(data)
            signals['order_flow_imbalance'] = self._estimate_order_flow_imbalance(data)
            
        except Exception as e:
            print(f"Profit signals calculation error: {e}")
        
        return signals
    
    def _calculate_momentum_score(self, data: pd.DataFrame) -> float:
        """Calculate pure momentum score."""
        try:
            # Multiple timeframe momentum
            mom_1d = data['Close'].iloc[-1] / data['Close'].iloc[-2] - 1
            mom_5d = data['Close'].iloc[-1] / data['Close'].iloc[-6] - 1 if len(data) > 5 else 0
            mom_20d = data['Close'].iloc[-1] / data['Close'].iloc[-21] - 1 if len(data) > 20 else 0
            
            # Weighted momentum score
            momentum_score = (mom_1d * 0.5 + mom_5d * 0.3 + mom_20d * 0.2)
            
            # Normalize to [-1, 1]
            return np.tanh(momentum_score * 10)
            
        except Exception as e:
            return 0.0
    
    def _calculate_volume_momentum(self, data: pd.DataFrame) -> float:
        """Calculate volume-based momentum."""
        try:
            if 'Volume' not in data.columns:
                return 0.0
            
            vol = data['Volume']
            if isinstance(vol, pd.DataFrame):
                vol = vol.iloc[:, 0]
            vol = vol.dropna().astype(float)
            
            # Check for zero/meaningless volume (common in FX, some indices)
            if len(vol) < 20 or vol.sum() == 0 or vol.max() < 1:
                return 0.0
            
            recent_volume = vol.tail(5).mean()
            avg_volume = vol.tail(20).mean()
            
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
            
            return np.tanh((volume_ratio - 1) * 2)
            
        except Exception:
            return 0.0
    
    def _calculate_volatility_momentum(self, data: pd.DataFrame) -> float:
        """Calculate volatility momentum for opportunity identification."""
        try:
            if len(data) < 20:
                return 0.0
            
            recent_vol = data['Returns'].tail(5).std() * np.sqrt(252)
            avg_vol = data['Returns'].tail(20).std() * np.sqrt(252)
            
            vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1
            
            # Higher volatility = higher opportunity (but also risk)
            return np.tanh((vol_ratio - 1) * 1.5)
            
        except Exception as e:
            return 0.0
    
    def _calculate_mean_reversion_score(self, data: pd.DataFrame) -> float:
        """Calculate mean reversion opportunity score."""
        try:
            if len(data) < 20:
                return 0.0
            
            # Calculate z-score relative to moving average
            ma_20 = data['Close'].rolling(20).mean()
            std_20 = data['Close'].rolling(20).std()
            
            current_price = data['Close'].iloc[-1]
            current_ma = ma_20.iloc[-1]
            current_std = std_20.iloc[-1]
            
            if current_std == 0:
                return 0.0
            
            z_score = (current_price - current_ma) / current_std
            
            # Mean reversion opportunity increases with extreme z-scores
            return -np.tanh(z_score)  # Negative because we want to fade extremes
            
        except Exception as e:
            return 0.0
    
    def _calculate_breakout_probability(self, data: pd.DataFrame) -> float:
        """Calculate probability of price breakout."""
        try:
            if len(data) < 20:
                return 0.0
            
            # Calculate recent trading range
            high_20 = data['High'].tail(20).max()
            low_20 = data['Low'].tail(20).min()
            current_price = data['Close'].iloc[-1]
            
            # Position within range
            range_position = (current_price - low_20) / (high_20 - low_20) if high_20 > low_20 else 0.5
            
            # Volume confirmation
            if 'Volume' in data.columns:
                recent_volume = data['Volume'].tail(5).mean()
                avg_volume = data['Volume'].tail(20).mean()
                volume_confirmation = recent_volume / avg_volume if avg_volume > 0 else 1
            else:
                volume_confirmation = 1.0
            
            # Breakout probability higher near range extremes with volume
            if range_position > 0.8:  # Near highs
                breakout_prob = (range_position - 0.8) * 5 * min(volume_confirmation, 2)
            elif range_position < 0.2:  # Near lows
                breakout_prob = (0.2 - range_position) * 5 * min(volume_confirmation, 2)
            else:
                breakout_prob = 0
            
            return min(breakout_prob, 1.0)
            
        except Exception as e:
            return 0.0
    
    def _calculate_statistical_edge(self, data: pd.DataFrame) -> float:
        """Calculate statistical arbitrage edge."""
        try:
            if len(data) < 50:
                return 0.0
            
            returns = data['Returns'].dropna()
            
            # Test for statistical properties that create edges
            # 1. Autocorrelation
            autocorr = returns.autocorr(lag=1)
            
            # 2. Skewness (asymmetry in returns)
            skewness = returns.skew()
            
            # 3. Kurtosis (tail risk/opportunity)
            kurtosis = returns.kurtosis()
            
            # 4. Volatility clustering (GARCH effects)
            vol_clustering = returns.abs().autocorr(lag=1)
            
            # Combine into statistical edge score
            edge_score = (
                abs(autocorr) * 0.3 +
                abs(skewness) * 0.2 +
                min(abs(kurtosis), 5) / 5 * 0.2 +
                abs(vol_clustering) * 0.3
            )
            
            return min(edge_score, 1.0)
            
        except Exception as e:
            return 0.0
    
    def _calculate_profit_probability(self, profit_signals: Dict[str, float],
                                    momentum_analysis: Dict[str, float],
                                    volatility_opportunities: Dict[str, float],
                                    statistical_edges: Dict[str, float],
                                    options_metrics: Dict[str, float] = None,
                                    quant_signal: Any = None,
                                    news_sentiment: Optional[float] = None,
                                    weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate overall profit probability using the dynamic information-weighting engine.

        Each signal component (momentum, technicals, quant, volume, options,
        macro/vol, news) is scored in [-1, 1] and blended with regime- and
        asset-class-aware weights from InformationWeightingEngine (falling back
        to fixed defaults if the engine is unavailable).
        """
        try:
            # Dynamic weights (engine-backed, with safe fallback)
            if not weights:
                weights, _, _ = self._get_dynamic_weights(momentum_analysis)
            else:
                weights = dict(weights)

            momentum_score = float(np.mean(list(momentum_analysis.values()))) if momentum_analysis else 0.0
            volatility_score = float(np.mean(list(volatility_opportunities.values()))) if volatility_opportunities else 0.0
            statistical_score = float(np.mean(list(statistical_edges.values()))) if statistical_edges else 0.0
            mom_sign = 1.0 if momentum_score >= 0 else -1.0

            # Signed technical score (Bollinger position, mean reversion, breakout, statistical edge)
            breakout = profit_signals.get('breakout_probability', 0.0) or 0.0
            technical_directional = float(np.clip(
                (profit_signals.get('bollinger_position', 0.0) or 0.0) * 0.35 +
                (profit_signals.get('mean_reversion', 0.0) or 0.0) * 0.30 +
                (breakout * mom_sign) * 0.20 +
                statistical_score * 0.15,
                -1.0, 1.0))

            # Signed volume score
            vol_break = (profit_signals.get('volume_breakout', 0.0) or 0.0) * mom_sign
            volume_score = float(np.clip(
                (profit_signals.get('volume_momentum', 0.0) or 0.0) * 0.6 + vol_break * 0.4,
                -1.0, 1.0))

            # Quant ensemble score (direction × probability)
            quant_score = 0.0
            if quant_signal is not None:
                q_dir = getattr(quant_signal, 'direction', 'NEUTRAL')
                q_prob = float(getattr(quant_signal, 'probability', 0.5) or 0.5)
                quant_score = (1.0 if q_dir == 'BULLISH' else -1.0 if q_dir == 'BEARISH' else 0.0) * q_prob
            else:
                ensemble_impact = (profit_signals.get('ensemble_decision', 0.0) or 0.0) * \
                                  (profit_signals.get('ensemble_confidence', 0.5) or 0.5)
                quant_score = float(np.clip(ensemble_impact, -1.0, 1.0))

            # Options intelligence impact
            options_impact = 0.0
            if options_metrics:
                sentiment_bias = np.tanh((options_metrics.get('net_call_put_ratio', 1.0) - 1) * 0.5)
                iv_rank = options_metrics.get('iv_rank', 1.0) or 1.0
                iv_impact = 0.0
                if iv_rank > 1.8 and abs(momentum_score) > 0.6:
                    iv_impact = -np.sign(momentum_score) * 0.15
                elif iv_rank < 0.6:
                    iv_impact = np.sign(momentum_score or 0.1) * 0.05
                options_impact = sentiment_bias * 0.1 + iv_impact

            # Macro/vol component (vol expansion/contraction — signed by momentum)
            macro_component = float(np.clip(np.sign(momentum_score or 0.1) * abs(volatility_score - 1.0), -1.0, 1.0))

            # News sentiment component ([-1, 1] or neutral 0)
            news_component = float(np.clip(news_sentiment or 0.0, -1.0, 1.0))

            # Engine-weighted blend (weights sum to 1.0)
            blended = (
                momentum_score * weights.get('price_momentum', 0.25) +
                technical_directional * weights.get('technical_signals', 0.25) +
                quant_score * weights.get('quant_model', 0.25) +
                volume_score * weights.get('volume_confirmation', 0.10) +
                options_impact * weights.get('options_flow', 0.05) +
                macro_component * weights.get('macro_regime', 0.05) +
                news_component * weights.get('news_sentiment', 0.05)
            )

            # Normalize to [0, 1]
            return max(0.0, min(1.0, (blended + 1.0) / 2.0))

        except Exception as e:
            print(f"Profit probability calculation error: {e}")
            return 0.5
    
    def _calculate_expected_return(self, data: pd.DataFrame, profit_signals: Dict[str, float]) -> float:
        """Calculate expected return based on historical patterns."""
        try:
            if len(data) < 20:
                return 0.0
            
            returns = data['Returns'].dropna()
            
            # Historical return patterns
            avg_return = returns.mean() * 252  # Annualized
            
            # Adjust based on current signals
            signal_strength = np.mean(list(profit_signals.values()))
            
            # Expected return = base return + signal adjustment
            expected_return = avg_return * (1 + signal_strength)
            
            return expected_return
            
        except Exception as e:
            return 0.0
    
    def _calculate_pure_risk_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate pure risk metrics without bias."""
        risk_metrics = {}
        
        try:
            returns = data['Returns'].dropna()
            
            if len(returns) < 20:
                return {'volatility': 0.5, 'max_drawdown': 0.1, 'var_95': 0.05}
            
            # Volatility
            risk_metrics['volatility'] = returns.std() * np.sqrt(252)
            
            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            risk_metrics['max_drawdown'] = abs(drawdown.min())
            
            # Value at Risk (95%)
            risk_metrics['var_95'] = abs(returns.quantile(0.05))
            
            # Downside deviation
            downside_returns = returns[returns < 0]
            risk_metrics['downside_deviation'] = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            
            # Tail risk
            risk_metrics['tail_risk'] = abs(returns.quantile(0.01))
            
        except Exception as e:
            print(f"Risk metrics calculation error: {e}")
            risk_metrics = {'volatility': 0.5, 'max_drawdown': 0.1, 'var_95': 0.05}
        
        return risk_metrics
    
    def _analyze_momentum_unbiased(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze momentum without any bias."""
        momentum = {}
        
        try:
            # Price momentum across timeframes
            momentum['1d'] = self._calculate_momentum_score(data)
            momentum['5d'] = data['Close'].iloc[-1] / data['Close'].iloc[-6] - 1 if len(data) > 5 else 0
            momentum['20d'] = data['Close'].iloc[-1] / data['Close'].iloc[-21] - 1 if len(data) > 20 else 0
            
            # Volume momentum
            momentum['volume'] = self._calculate_volume_momentum(data)
            
            # Volatility momentum
            momentum['volatility'] = self._calculate_volatility_momentum(data)
            
        except Exception as e:
            print(f"Momentum analysis error: {e}")
        
        return momentum
    
    def _identify_volatility_opportunities(self, data: pd.DataFrame) -> Dict[str, float]:
        """Identify volatility-based opportunities."""
        opportunities = {}
        
        try:
            if len(data) < 20:
                return opportunities
            
            returns = data['Returns'].dropna()
            
            # Current vs historical volatility
            current_vol = returns.tail(5).std() * np.sqrt(252)
            historical_vol = returns.std() * np.sqrt(252)
            
            opportunities['vol_expansion'] = current_vol / historical_vol if historical_vol > 0 else 1
            
            # Volatility clustering
            opportunities['vol_clustering'] = returns.abs().autocorr(lag=1)
            
            # GARCH effects
            squared_returns = returns ** 2
            opportunities['garch_effect'] = squared_returns.autocorr(lag=1)
            
        except Exception as e:
            print(f"Volatility opportunities error: {e}")
        
        return opportunities
    
    def _calculate_options_metrics(self, symbol: str, data: pd.DataFrame) -> Dict[str, float]:
        """Deep dive into options intelligence: IV Rank, Gamma, Net Premium Flow."""
        metrics = {
            'iv_rank': 0.5,
            'iv_pct': 0.5,
            'gamma_risk_level': 0.0,
            'net_call_put_ratio': 1.0,
            'est_option_volume': 0.0
        }
        
        try:
            # We fetch chain to get real-time IV and positioning
            chain = get_options_chain(symbol)
            if not chain or not chain.get('calls') or len(chain['calls']) == 0:
                # If no chain, use HV as a proxy for the IV component
                hv = data['Returns'].std() * np.sqrt(252)
                metrics['iv_rank'] = hv
                return metrics
            
            # ATM IV Extraction
            calls = chain['calls']
            current_price = data['Close'].iloc[-1]
            atm_call = min(calls, key=lambda x: abs(x['strike'] - current_price))
            current_iv = atm_call.get('impliedVolatility', 0)
            
            # IV vs HV comparison (The 'Edge' component)
            hv = data['Returns'].std() * np.sqrt(252)
            metrics['iv_rank'] = current_iv / hv if hv > 0 else 1.0
            metrics['iv_pct'] = current_iv
            
            # Directional sentiment from volume
            c_vol = sum(c.get('volume', 0) or 0 for c in chain['calls'])
            p_vol = sum(p.get('volume', 0) or 0 for p in chain['puts'])
            metrics['net_call_put_ratio'] = c_vol / (p_vol + 1e-8)
            
            # Gamma approximation from ATM volume concentration
            metrics['gamma_risk_level'] = (atm_call.get('volume', 0) or 0) / (c_vol + 1e-8)
            
        except Exception:
            pass
            
        return metrics

    def _find_statistical_edges(self, data: pd.DataFrame) -> Dict[str, float]:
        """Find statistical arbitrage edges."""
        edges = {}
        
        try:
            if len(data) < 50:
                return edges
            
            returns = data['Returns'].dropna()
            
            # Autocorrelation edges
            edges['autocorr_1'] = returns.autocorr(lag=1)
            edges['autocorr_5'] = returns.autocorr(lag=5)
            
            # Distribution edges
            edges['skewness'] = returns.skew()
            edges['kurtosis'] = min(returns.kurtosis(), 10) / 10  # Normalize
            
            # Volatility edges
            edges['vol_clustering'] = returns.abs().autocorr(lag=1)
            
        except Exception as e:
            print(f"Statistical edges error: {e}")
        
        return edges
    
    def _create_null_analysis(self, symbol: str) -> UnbiasedAnalysis:
        """Create null analysis for failed cases."""
        return UnbiasedAnalysis(
            symbol=symbol,
            profit_probability=0.5,
            expected_return=0.0,
            risk_adjusted_return=0.0,
            confidence_score=0.0,
            entry_signals=['Insufficient data'],
            exit_signals=['Insufficient data'],
            risk_factors=['Data unavailable'],
            profit_catalysts=['Analysis failed'],
            model_reasoning=['Unable to analyze'],
            raw_data_insights={},
            timestamp=datetime.now()
        )
    
    # Additional helper methods would continue here...
    def _calculate_model_confidence(self, data: pd.DataFrame, profit_signals: Dict[str, float]) -> float:
        """Calculate model confidence based on data quality and signal strength."""
        try:
            # Data quality factors
            data_quality = min(len(data) / 252, 1.0)  # More data = higher confidence
            
            # Signal consistency
            signal_values = list(profit_signals.values())
            signal_consistency = 1 - np.std(signal_values) if signal_values else 0
            
            # Volume consistency (if available)
            volume_consistency = 1.0
            if 'Volume' in data.columns:
                volume_cv = data['Volume'].std() / data['Volume'].mean() if data['Volume'].mean() > 0 else 1
                volume_consistency = max(0, 1 - volume_cv)
            
            # Combined confidence
            confidence = (data_quality * 0.4 + signal_consistency * 0.4 + volume_consistency * 0.2)
            
            return max(0.1, min(0.95, confidence))
            
        except Exception as e:
            return 0.5
    
    def _generate_entry_signals(self, profit_signals: Dict[str, float], 
                              momentum_analysis: Dict[str, float]) -> List[str]:
        """Generate pure entry signals based on profit potential."""
        signals = []
        
        try:
            # Momentum-based entries
            if profit_signals.get('price_momentum', 0) > 0.3:
                signals.append(f"Strong price momentum detected ({profit_signals['price_momentum']:.2f})")
            
            if profit_signals.get('volume_momentum', 0) > 0.3:
                signals.append(f"Volume surge confirms move ({profit_signals['volume_momentum']:.2f})")
            
            # Breakout entries
            if profit_signals.get('breakout_probability', 0) > 0.6:
                signals.append(f"High breakout probability ({profit_signals['breakout_probability']:.2f})")
            
            # Statistical entries
            if profit_signals.get('statistical_edge', 0) > 0.4:
                signals.append(f"Statistical edge identified ({profit_signals['statistical_edge']:.2f})")
            
            # Mean reversion entries
            if profit_signals.get('mean_reversion', 0) < -0.5:
                signals.append(f"Extreme oversold - mean reversion opportunity")
            elif profit_signals.get('mean_reversion', 0) > 0.5:
                signals.append(f"Extreme overbought - short opportunity")
            
            if not signals:
                signals.append("No clear entry signals - wait for better setup")
                
        except Exception as e:
            signals.append(f"Entry signal generation error: {str(e)}")
        
        return signals
    
    def _generate_exit_signals(self, profit_signals: Dict[str, float], 
                             risk_metrics: Dict[str, float]) -> List[str]:
        """Generate exit signals based on risk management."""
        signals = []
        
        try:
            # Risk-based exits
            if risk_metrics.get('volatility', 0) > 0.5:
                signals.append(f"High volatility - consider position sizing reduction")
            
            if risk_metrics.get('max_drawdown', 0) > 0.2:
                signals.append(f"Significant drawdown risk - tight stops recommended")
            
            # Profit-taking signals
            if profit_signals.get('mean_reversion', 0) > 0.7:
                signals.append(f"Extreme overbought - profit-taking opportunity")
            elif profit_signals.get('mean_reversion', 0) < -0.7:
                signals.append(f"Extreme oversold - cover shorts")
            
            # Momentum exhaustion
            if profit_signals.get('price_momentum', 0) < -0.5:
                signals.append("Momentum turning negative - consider exit")
            elif profit_signals.get('price_momentum', 0) > 0.8:
                signals.append("Momentum extended - trailing stop recommended")
            
            # Volume divergence
            if profit_signals.get('volume_momentum', 0) < -0.3 and profit_signals.get('price_momentum', 0) > 0.3:
                signals.append("Price-volume divergence - weakening conviction")
            
            if not signals:
                signals.append("No exit signals - hold current position")
                
        except Exception as e:
            signals.append(f"Exit signal generation error: {str(e)}")
        
        return signals

    def _identify_risk_factors(self, data: pd.DataFrame, risk_metrics: Dict[str, float]) -> List[str]:
        """Identify pure risk factors from data."""
        risks = []
        
        try:
            # Volatility risks
            if risk_metrics.get('volatility', 0) > 0.4:
                risks.append(f"High volatility ({risk_metrics['volatility']:.1%}) increases position risk")
            
            # Drawdown risks
            if risk_metrics.get('max_drawdown', 0) > 0.15:
                risks.append(f"Historical max drawdown of {risk_metrics['max_drawdown']:.1%}")
            
            # Tail risks
            if risk_metrics.get('var_95', 0) > 0.05:
                risks.append(f"High tail risk - 5% chance of {risk_metrics['var_95']:.1%} daily loss")
            
            # Liquidity risks (volume-based)
            if 'Volume' in data.columns:
                recent_volume = data['Volume'].tail(5).mean()
                if recent_volume < data['Volume'].quantile(0.2):
                    risks.append("Low recent volume may impact liquidity")
            
            if not risks:
                risks.append("Low risk profile based on historical data")
                
        except Exception as e:
            risks.append(f"Risk identification error: {str(e)}")
        
        return risks
    
    def _identify_profit_catalysts(self, data: pd.DataFrame, profit_signals: Dict[str, float]) -> List[str]:
        """Identify potential profit catalysts."""
        catalysts = []
        
        try:
            # Technical catalysts
            if profit_signals.get('breakout_probability', 0) > 0.6:
                catalysts.append("Technical breakout setup - range expansion likely")
            
            # Volume catalysts
            if profit_signals.get('volume_momentum', 0) > 0.5:
                catalysts.append("Unusual volume activity - institutional interest")
            
            # Volatility catalysts
            if profit_signals.get('volatility_momentum', 0) > 0.4:
                catalysts.append("Volatility expansion - larger moves expected")
            
            # Statistical catalysts
            if profit_signals.get('statistical_edge', 0) > 0.5:
                catalysts.append("Strong statistical edge - probability favors direction")
            
            # Mean reversion catalysts
            mean_rev = profit_signals.get('mean_reversion', 0)
            if abs(mean_rev) > 0.6:
                direction = "upward" if mean_rev < 0 else "downward"
                catalysts.append(f"Extreme deviation suggests {direction} reversion")
            
            if not catalysts:
                catalysts.append("No clear catalysts identified - monitor for developments")
                
        except Exception as e:
            catalysts.append(f"Catalyst identification error: {str(e)}")
        
        return catalysts
    
    def _generate_model_reasoning(self, profit_signals: Dict[str, float],
                                momentum_analysis: Dict[str, float],
                                volatility_opportunities: Dict[str, float]) -> List[str]:
        """Generate transparent model reasoning."""
        reasoning = []
        
        try:
            # Explain the model's logic
            reasoning.append("Model Analysis Logic:")
            
            # Momentum reasoning
            momentum_score = np.mean(list(momentum_analysis.values())) if momentum_analysis else 0
            if momentum_score > 0.2:
                reasoning.append(f"• Positive momentum detected across timeframes ({momentum_score:.2f})")
            elif momentum_score < -0.2:
                reasoning.append(f"• Negative momentum detected ({momentum_score:.2f})")
            else:
                reasoning.append(f"• Neutral momentum environment ({momentum_score:.2f})")
            
            # Signal strength reasoning
            signal_strength = np.mean(list(profit_signals.values()))
            reasoning.append(f"• Overall signal strength: {signal_strength:.2f}")
            
            # Volatility reasoning
            vol_score = np.mean(list(volatility_opportunities.values())) if volatility_opportunities else 0
            if vol_score > 0.3:
                reasoning.append(f"• High volatility environment creates opportunities ({vol_score:.2f})")
            else:
                reasoning.append(f"• Low volatility environment ({vol_score:.2f})")
            
            # Risk-reward reasoning
            reasoning.append("• Model prioritizes mathematical edges over market sentiment")
            
            # Massive Ensemble Reasoning
            if 'ensemble_decision' in profit_signals:
                decision_val = profit_signals['ensemble_decision']
                confidence = profit_signals.get('ensemble_confidence', 0.0) * 100
                pred_return = profit_signals.get('ensemble_predicted_return', 0.0) * 100
                
                direction = "BUY" if decision_val > 0 else "SELL" if decision_val < 0 else "HOLD"
                
                reasoning.append(f"* MASSIVE ENSEMBLE CONSENSUS: {direction} (Confidence: {confidence:.1f}%)")
                reasoning.append(f"* Ensemble predicts {pred_return:+.2f}% short-term move")
                reasoning.append("* Models: LSTM, Transformer, Deep MLP, Random Forest, GBM")
            
            reasoning.append("• Analysis is purely data-driven without bias toward popular stocks")
            
        except Exception as e:
            reasoning.append(f"Reasoning generation error: {str(e)}")
        
        return reasoning
    
    def _analyze_volume_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume patterns for insights."""
        if 'Volume' not in data.columns:
            return {'available': False, 'note': 'Volume data not available (normal for FX/OTC markets)'}
        
        try:
            vol = data['Volume']
            if isinstance(vol, pd.DataFrame):
                vol = vol.iloc[:, 0]
            vol = vol.dropna().astype(float)
            
            # Check for zero/meaningless volume
            if len(vol) < 20 or vol.sum() == 0 or vol.max() < 1:
                return {'available': False, 'note': 'Volume data is zero/unavailable for this asset'}
            
            volume_analysis = {'available': True}
            
            volume_ma_5 = vol.rolling(5).mean()
            volume_ma_20 = vol.rolling(20).mean()
            
            volume_analysis['trend'] = 'increasing' if volume_ma_5.iloc[-1] > volume_ma_20.iloc[-1] else 'decreasing'
            volume_analysis['current_vs_average'] = vol.iloc[-1] / volume_ma_20.iloc[-1] if volume_ma_20.iloc[-1] > 0 else 1
            
            volume_threshold = volume_ma_20.iloc[-1] * 2
            recent_spikes = (vol.tail(5) > volume_threshold).sum()
            volume_analysis['recent_spikes'] = int(recent_spikes)
            
            return volume_analysis
            
        except Exception as e:
            return {'available': False, 'error': str(e)}
    
    def _analyze_microstructure(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market microstructure patterns."""
        try:
            microstructure = {}
            
            # Price-volume relationship
            price_change = data['Close'].pct_change()
            if 'Volume' in data.columns:
                volume_change = data['Volume'].pct_change()
                correlation = price_change.corr(volume_change)
                microstructure['price_volume_correlation'] = correlation if not pd.isna(correlation) else 0
            else:
                microstructure['price_volume_correlation'] = 0.0
            
            # Intraday patterns (using OHLC)
            daily_range = (data['High'] - data['Low']) / data['Close']
            microstructure['average_daily_range'] = daily_range.mean()
            
            # Gap analysis
            gaps = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
            microstructure['gap_frequency'] = (abs(gaps) > 0.02).mean()  # Gaps > 2%
            
            return microstructure
            
        except Exception as e:
            return {'error': str(e)}
    
    # Additional calculation methods for completeness
    def _calculate_bollinger_position(self, data: pd.DataFrame) -> float:
        """Calculate position within Bollinger Bands."""
        try:
            if len(data) < 20:
                return 0.0
            
            ma_20 = data['Close'].rolling(20).mean()
            std_20 = data['Close'].rolling(20).std()
            
            upper_band = ma_20 + (2 * std_20)
            lower_band = ma_20 - (2 * std_20)
            
            current_price = data['Close'].iloc[-1]
            current_upper = upper_band.iloc[-1]
            current_lower = lower_band.iloc[-1]
            
            if current_upper == current_lower:
                return 0.0
            
            # Position within bands (-1 = lower band, +1 = upper band)
            position = (current_price - current_lower) / (current_upper - current_lower) * 2 - 1
            
            return max(-1, min(1, position))
            
        except Exception as e:
            return 0.0
    
    def _calculate_volume_breakout_score(self, data: pd.DataFrame) -> float:
        """Calculate volume breakout score."""
        try:
            if 'Volume' not in data.columns or len(data) < 20:
                return 0.0
            
            vol = data['Volume']
            if isinstance(vol, pd.DataFrame):
                vol = vol.iloc[:, 0]
            vol = vol.dropna().astype(float)
            
            # No meaningful volume → no volume breakout signal
            if len(vol) < 20 or vol.sum() == 0 or vol.max() < 1:
                return 0.0
            
            current_volume = vol.iloc[-1]
            avg_volume = vol.rolling(20).mean().iloc[-1]
            
            if avg_volume == 0:
                return 0.0
            
            volume_ratio = current_volume / avg_volume
            return min(1.0, (volume_ratio - 1) / 2)
            
        except Exception:
            return 0.0
    
    def _calculate_autocorrelation_score(self, data: pd.DataFrame) -> float:
        """Calculate autocorrelation score for mean reversion/momentum."""
        try:
            if len(data) < 30:
                return 0.0
            
            returns = data['Returns'].dropna()
            
            # 1-day autocorrelation
            autocorr = returns.autocorr(lag=1)
            
            return autocorr if not pd.isna(autocorr) else 0.0
            
        except Exception as e:
            return 0.0
    
    def _estimate_bid_ask_pressure(self, data: pd.DataFrame) -> float:
        """Estimate bid-ask pressure from OHLC data."""
        try:
            # Use close position within daily range as proxy
            daily_position = (data['Close'] - data['Low']) / (data['High'] - data['Low'])
            daily_position = daily_position.fillna(0.5)
            
            # Recent average position
            recent_position = daily_position.tail(5).mean()
            
            # Convert to pressure score (-1 = sell pressure, +1 = buy pressure)
            return (recent_position - 0.5) * 2
            
        except Exception as e:
            return 0.0
    
    def _estimate_order_flow_imbalance(self, data: pd.DataFrame) -> float:
        """Estimate order flow imbalance."""
        try:
            if 'Volume' not in data.columns:
                return 0.0
            
            vol = data['Volume']
            if isinstance(vol, pd.DataFrame):
                vol = vol.iloc[:, 0]
            vol = vol.dropna().astype(float)
            
            # No meaningful volume → no order flow signal
            if len(vol) < 20 or vol.sum() == 0 or vol.max() < 1:
                return 0.0
                
            price_change = data['Close'].pct_change()
            signed_volume = np.sign(price_change) * vol
            
            cvd = signed_volume.rolling(20).sum()
            avg_vol = vol.rolling(20).mean() * 20
            
            if avg_vol.iloc[-1] == 0:
                return 0.0
                
            imbalance = cvd.iloc[-1] / avg_vol.iloc[-1]
            return max(-1.0, min(1.0, imbalance))
            
        except Exception:
            return 0.0

    async def scan_entire_market(self, max_symbols: int = 200, min_confidence: float = 0.15) -> List[UnbiasedAnalysis]:
        """
        Perform a completely unbiased scan of the entire market universe.
        Prioritizes pure profit potential regardless of asset name/fame.
        """
        all_symbols = []
        for category, symbols in self.asset_universe.items():
            all_symbols.extend(symbols)
            
        # Randomize scan order to avoid alphabetical bias if limited
        import random
        random.shuffle(all_symbols)
        
        scan_targets = all_symbols[:max_symbols]
        results = []
        
        # Parallel processing for speed
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {
                executor.submit(self.analyze_symbol_pure_profit, symbol): symbol 
                for symbol in scan_targets
            }
            
            for future in future_to_symbol:
                symbol = future_to_symbol[future]
                try:
                    analysis = future.result()
                    if analysis and analysis.confidence_score > min_confidence: # Filter low confidence noise
                        results.append(analysis)
                except Exception as e:
                    # Log silently to avoid clutter
                    continue
                    
        return results

    def analyze_symbol_pure_profit(self, symbol: str, fast_mode: bool = True) -> Optional['UnbiasedAnalysis']:
        """Analyze a single symbol for pure profit potential."""
        try:
            # Determine asset type and fetch correctly
            if '/' in symbol:
                # Forex: EUR/USD → EUR_USD
                fx_key = symbol.replace('/', '_')
                df = get_fx(fx_key)
                if (df is None or df.empty):
                    # Fallback Yahoo
                    yf_sym = symbol.replace('/', '') + '=X'
                    df = get_stock(yf_sym, period="6mo")
            elif '=F' in symbol:
                df = get_futures_proxy(symbol, period="6mo")
            elif '=' in symbol:
                df = get_stock(symbol, period="6mo")
            elif '-' in symbol:
                df = get_stock(symbol, period="6mo")
            else:
                df = get_stock(symbol, period="6mo")
                
            if df is None or df.empty or len(df) < 30:
                return None
            
            # Calculate returns for analysis
            df['Returns'] = df['Close'].pct_change()
            
            # 2. Calculate Profit Signals
            profit_signals = self._calculate_profit_signals(df, fast_mode=fast_mode)
            
            # 3. Derive Probabilities
            profit_prob = self._derive_profit_probability(profit_signals)
            expected_return = self._calculate_expected_return(df, profit_signals)
            
            # 4. Risk Assessment
            risk_metrics = self._calculate_pure_risk_metrics(df)
            
            # 5. Generate Unbiased Conclusion
            
            # Determine primary signal direction based on signals
            momentum = profit_signals.get('price_momentum', 0)
            mean_rev = profit_signals.get('mean_reversion', 0)
            
            # Simple ensemble: weighted average of signals
            signal_strength = (momentum * 0.6 + mean_rev * 0.4)
            
            # Construct analysis object
            analysis = UnbiasedAnalysis(
                symbol=symbol,
                profit_probability=profit_prob,
                expected_return=expected_return,
                risk_adjusted_return=expected_return / (risk_metrics.get('volatility', 1) or 1),
                confidence_score=min(1.0, abs(signal_strength) * 2), # Scale to 0-1
                entry_signals=[k for k, v in profit_signals.items() if v > 0.5],
                exit_signals=[k for k, v in profit_signals.items() if v < -0.5],
                risk_factors=[k for k, v in risk_metrics.items() if v > 0.5], # High risk metrics
                profit_catalysts=[],
                model_reasoning=[f"Strong {k}" for k, v in profit_signals.items() if abs(v) > 0.7],
                raw_data_insights=profit_signals,
                timestamp=datetime.now()
            )
            
            # Add dynamic attribute for direction needed by chatbot
            # This is a bit hacky but ensures compatibility with the chatbot's expectation
            if signal_strength > 0.1:
                analysis.signal_direction = 'BULLISH'
            elif signal_strength < -0.1:
                analysis.signal_direction = 'BEARISH'
            else:
                analysis.signal_direction = 'NEUTRAL'
                
            analysis.primary_driver = max(profit_signals, key=profit_signals.get) if profit_signals else "None"
            analysis.current_price = float(df['Close'].iloc[-1])
            analysis.risk_reward_ratio = abs(expected_return / (risk_metrics.get('max_drawdown', 0.1) + 1e-6))

            return analysis
            
        except Exception as e:
            # print(f"Analysis error for {symbol}: {e}")
            return None
    
    def _derive_profit_probability(self, profit_signals: Dict[str, float]) -> float:
        """Derive profit probability from signals (used by scan path)."""
        try:
            signal_values = list(profit_signals.values())
            if not signal_values:
                return 0.5
            avg_signal = np.mean(signal_values)
            return max(0.05, min(0.95, (avg_signal + 1) / 2))
        except Exception:
            return 0.5