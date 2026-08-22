"""
Octavian Enhanced AI Market Chatbot - Completely Unbiased & Profit-Maximizing
Advanced AI-powered chatbot with source credibility weighting, timeframe analysis, and trader profiling
NOW WITH: Complete market coverage, zero bias, pure profit focus, and daily simulation learning

This module provides a sophisticated AI chatbot that integrates:
1. Unbiased market analysis covering ALL assets equally
2. Pure profit-maximizing insights without bias toward popular stocks
3. Source credibility weighting system for news analysis
4. Multi-timeframe analysis engine for context-aware insights
5. Enhanced trader profiling with conversation learning
6. Daily market simulation learning integration
7. Comprehensive market analysis with anticipation factors

Author: APB - Octavian Team
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import re
import json
import time
import uuid
from typing import Dict, List, Optional, Tuple, Any
import threading
from concurrent.futures import ThreadPoolExecutor
import asyncio

from data_sources import get_stock, get_fx, get_futures_proxy


# Queries that would NOT benefit from a price chart — knowledge / definitional
# asks where a candlestick adds nothing and costs latency. Any query that
# explicitly requests a visual always warrants charts. Patterns are precise
# (regex-anchored) so "what does COST do?" is excluded while "what does the
# price of oil affect..." / "what is the probability X reaches Y" still chart.
_NO_CHART_PATTERNS = (
    re.compile(r"\btell me about\b"),
    re.compile(r"\bwho are\b"),
    re.compile(r"\bwho is\b"),
    re.compile(r"\bwhat sector is\b"),
    re.compile(r"\bbusiness model\b"),
    re.compile(r"\bmain competitors\b"),
    re.compile(r"\bcompetitors\b"),
    re.compile(r"\bpay a dividend\b"),
    re.compile(r"\bdividend yield\b"),
    re.compile(r"\bincome stock\b"),
    re.compile(r"\bgrowing its dividend\b"),
    re.compile(r"\bwhat happened to\b"),
    re.compile(r"\bwhy did\b"),
    re.compile(r"\bdoes it pay\b"),
    re.compile(r"\bwhat does [a-z0-9.^-]{1,8} do\b"),
    re.compile(r"\bwhat is [a-z0-9.^-]{1,8} the company\b"),
    re.compile(r"\bwhat is 's\b"),
)


def _query_warrants_charts(query: str) -> bool:
    """True when a query should receive price-chart visuals."""
    q = (query or "").lower()
    if any(w in q for w in ("chart", "graph", "plot", "show me", "visual",
                            "candlestick", "picture")):
        return True
    return not any(p.search(q) for p in _NO_CHART_PATTERNS)
from sector_scanner import scan_sectors
from ml_analysis import get_analyzer, MLMarketAnalyzer
from indicators import add_indicators
from regime import volatility_regime, risk_on_off
from database_manager import get_database_manager
from news_analysis_engine import get_news_engine
from multi_asset_analyzer import MultiAssetAnalyzer
from advanced_news_processor import AdvancedNewsProcessor
from cross_sector_analyzer import CrossSectorAnalyzer
from narrative_generator import NarrativeGenerator

# Import new enhanced systems
from source_credibility_engine import SourceCredibilityEngine
from timeframe_analysis_engine import TimeframeAnalysisEngine, TimeframeScope
from trader_profile import (
    learn_from_user_interaction, 
    get_timeframe_context_for_analysis, get_recommendation_style
)

# Import unbiased analyzer and simulation engine
from unbiased_market_analyzer import UnbiasedMarketAnalyzer
from market_simulation_engine import MarketSimulationEngine
from response_formatter import get_response_formatter, FormattedResponse

# Import dynamic intent detection engine
from intent_detection_engine import (
    get_intent_detection_engine, 
    IntentAnalysis, 
    IntentCategory, 
    ResponseFormat, 
    DetailLevel,
    Urgency  # Add missing import
)

# Dynamic ticker universe — single source of truth for all tickers
try:
    from ticker_universe import get_ticker_universe
    _HAS_TICKER_UNIVERSE = True
except ImportError:
    _HAS_TICKER_UNIVERSE = False

class OctavianEnhancedChatbot:
    """Enhanced AI chatbot with complete market coverage, zero bias, and pure profit focus."""
    
    def __init__(self):
        # Lazy initialization for heavy components
        self._analyzer = None
        self._llm_agent = None
        self._db_manager = None
        self._news_engine = None
        self._multi_asset_analyzer = None
        self._advanced_news_processor = None
        self._cross_sector_analyzer = None
        self._narrative_generator = None
        self._source_credibility_engine = None
        self._timeframe_engine = None
        self._unbiased_analyzer = None
        self._simulation_engine = None
        self._simulation_thread = None
        self._response_formatter = None
        
        self.session_id = str(uuid.uuid4())
        self.conversation_context = {}
        self.user_preferences = {}

        # Local-LLM (LM Studio) structured-intent interpreter state. The flag is
        # lazily probed on first use (see _call_lm_studio_interpreter); callers may
        # also set it explicitly to force enable/disable.
        self.lm_studio_online = None  # None = unknown, probed on first call
        self._lm_studio_narrative = ""
        
        # REMOVED ALL BIAS - Enhanced intent patterns for complete market coverage
        self.intent_patterns = {
            'price_query': [
                r'what.*(price|cost|value|worth|trading at)',
                r'how much.*(is|are|does|cost)',
                r'current.*(price|level|value)',
                r'(show|display|get|tell me).*price',
                r'(quote|quotes) for',
            ],
            'prediction': [
                r'(predict|forecast|will|should|outlook|future)',
                r'(bullish|bearish|buy|sell|long|short)',
                r'what.*(think|expect|believe)',
                r'(recommendation|suggest|advice)',
                r'where.*(heading|going)',
                r'(target|price target)',
            ],
            'analysis': [
                r'(analyze|analysis|technical|fundamental)',
                r'(trend|momentum|volatility|strength)',
                r'(support|resistance|levels)',
                r'(rsi|macd|ema|sma|bollinger|stochastic)',
                r'(overbought|oversold|neutral)',
                r'(breakout|breakdown|reversal)',
            ],
            'risk': [
                r'(risk|risks|risky|risk.analysis)',
                r'(volatility|vol|variance|std.dev)',
                r'(downside|drawdown|max.loss)',
                r'(sharpe|sortino|risk.adjusted)',
                r'(beta|correlation|covariance)',
                r'(var|value.at.risk|cvar)',
                r'(risk.management|risk.metrics)',
            ],
            'unbiased_scan': [
                r'(scan|screen|find|discover|search)',
                r'(opportunities|best|top|profitable)',
                r'(entire.market|all.stocks|complete.analysis)',
                r'(unbiased|objective|pure.profit)',
            ],
            'profit_maximization': [
                r'(profit|money|returns|gains)',
                r'(maximize|optimize|best.return)',
                r'(edge|advantage|opportunity)',
                r'(statistical|mathematical|quantitative)',
            ],
            'timeframe_specific': [
                r'(scalp|scalping|minutes|seconds)',
                r'(intraday|day.trade|same.day)',
                r'(swing|days|week|short.term)',
                r'(position|weeks|month|medium.term)',
                r'(invest|long.term|months|years)',
            ],
            'news_sentiment': [
                r'(news|sentiment|market mood|feeling)',
                r'(what.*happening|whats going on)',
                r'(catalyst|events|earnings)',
                r'(headlines|articles|reports)',
            ],
            'comprehensive': [
                r'(comprehensive|full|complete|detailed)',
                r'(everything|all.analysis|deep.dive)',
                r'(octavian|full.power|maximum.analysis)',
            ],
            'sector': [
                r'(sector|sectors|industry|industries|sector rotation)',
                r'(which sectors|best sectors|worst sectors)',
            ]
        }
        
    def _call_lm_studio_interpreter(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Query the local LM Studio instance and parse its structured JSON reply.

        Returns a dict with ``intent``, ``symbols``, ``timeframe`` and
        ``analysis_type`` keys, or ``None`` when the local LLM is offline,
        unreachable, returns nothing, or emits no parseable structured intent.
        The raw narrative is always stored on ``self._lm_studio_narrative``.
        """
        # Auto-probe connectivity on first use when not explicitly set.
        if self.lm_studio_online is None:
            try:
                from financial_llm_engine import check_llm_connectivity
                self.lm_studio_online = bool(check_llm_connectivity())
            except Exception:
                self.lm_studio_online = False

        if not self.lm_studio_online:
            return None

        raw = None
        try:
            from financial_llm_engine import _call_llm
            raw = _call_llm(prompt)
        except Exception:
            return None

        if raw is None:
            return None
        raw_str = str(raw)
        self._lm_studio_narrative = raw_str
        if not raw_str.strip():
            return None

        parsed = None
        # 1) Direct parse
        try:
            parsed = json.loads(raw_str)
        except Exception:
            parsed = None

        # 2) Regex fallback: pull the first balanced JSON object (handles
        #    markdown code fences, leading prose, trailing commentary).
        if not isinstance(parsed, dict):
            try:
                match = re.search(r'\{.*\}', raw_str, re.DOTALL)
                if match:
                    candidate = match.group(0)
                    parsed = json.loads(candidate)
            except Exception:
                parsed = None

        if not isinstance(parsed, dict):
            return None

        intent = parsed.get('intent')
        if not intent:
            return None

        symbols = parsed.get('symbols', [])
        if not isinstance(symbols, list):
            symbols = []

        return {
            'intent': str(intent),
            'symbols': [str(s) for s in symbols if s],
            'timeframe': str(parsed.get('timeframe', '') or ''),
            'analysis_type': str(parsed.get('analysis_type', '') or ''),
        }

    @property
    def analyzer(self):
        if self._analyzer is None:
            self._analyzer = get_analyzer()
        return self._analyzer

    @property
    def db_manager(self):
        if self._db_manager is None:
            self._db_manager = get_database_manager()
        return self._db_manager

    @property
    def news_engine(self):
        if self._news_engine is None:
            self._news_engine = get_news_engine()
        return self._news_engine

    @property
    def multi_asset_analyzer(self):
        if self._multi_asset_analyzer is None:
            self._multi_asset_analyzer = MultiAssetAnalyzer()
        return self._multi_asset_analyzer

    @property
    def advanced_news_processor(self):
        if self._advanced_news_processor is None:
            self._advanced_news_processor = AdvancedNewsProcessor()
        return self._advanced_news_processor

    @property
    def cross_sector_analyzer(self):
        if self._cross_sector_analyzer is None:
            self._cross_sector_analyzer = CrossSectorAnalyzer()
        return self._cross_sector_analyzer

    @property
    def narrative_generator(self):
        if self._narrative_generator is None:
            self._narrative_generator = NarrativeGenerator()
        return self._narrative_generator

    @property
    def source_credibility_engine(self):
        if self._source_credibility_engine is None:
            self._source_credibility_engine = SourceCredibilityEngine()
        return self._source_credibility_engine

    @property
    def timeframe_engine(self):
        if self._timeframe_engine is None:
            self._timeframe_engine = TimeframeAnalysisEngine()
        return self._timeframe_engine
    
    @property
    def intent_engine(self):
        """Get dynamic intent detection engine."""
        return get_intent_detection_engine()
        if self._timeframe_engine is None:
            self._timeframe_engine = TimeframeAnalysisEngine()
        return self._timeframe_engine

    @property
    def unbiased_analyzer(self):
        if self._unbiased_analyzer is None:
            self._unbiased_analyzer = UnbiasedMarketAnalyzer()
        return self._unbiased_analyzer

    @property
    def simulation_engine(self):
        if self._simulation_engine is None:
            self._simulation_engine = MarketSimulationEngine()
            # Start thread only when engine is first accessed/needed
            self._simulation_thread = self._simulation_engine.start_simulation_engine()
        return self._simulation_engine

    @property
    def response_formatter(self):
        if self._response_formatter is None:
            self._response_formatter = get_response_formatter()
        return self._response_formatter

    def _get_asset_examples(self, per_asset: int = 4) -> Dict[str, List[str]]:
        """Dynamically source example tickers from the live multi-asset universe.

        Never hardcodes symbols — the examples always mirror the currently
        supported universe (equities, ETFs, crypto, forex, futures).
        """
        examples: Dict[str, List[str]] = {
            'Stocks': [], 'ETFs': [], 'Crypto': [], 'Forex': [], 'Futures': [],
        }
        try:
            from ticker_universe import get_ticker_universe
            u = get_ticker_universe()
            stocks = u.get_all_stocks() or []
            etfs = u.get_etfs() or []
            crypto = u.get_crypto() or []
            fx = u.get_forex() or []
            fut = u.get_futures() or []
            examples['Stocks'] = [s for s in stocks if len(s) <= 6][:per_asset]
            examples['ETFs'] = etfs[:per_asset]
            examples['Crypto'] = crypto[:per_asset]
            examples['Forex'] = [f.replace('-', '/') for f in fx[:per_asset]]
            examples['Futures'] = fut[:per_asset]
        except Exception:
            pass
        # Guarantee every category is non-empty so the UI guidance is always useful.
        fallback = {
            'Stocks': ['AAPL', 'MSFT', 'NVDA'],
            'ETFs': ['SPY', 'QQQ', 'IWM'],
            'Crypto': ['BTC-USD', 'ETH-USD', 'SOL-USD'],
            'Forex': ['EUR/USD', 'GBP/USD', 'USD/JPY'],
            'Futures': ['ES=F', 'NQ=F', 'CL=F'],
        }
        for k, v in examples.items():
            if not v:
                examples[k] = fallback.get(k, [])[:per_asset]
        return examples

    def _fallback_analysis_summary(self, symbol_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback when _create_enhanced_analysis_summary is not available."""
        summary = {
            'total_symbols': len(symbol_analyses),
            'bullish_count': 0,
            'bearish_count': 0,
            'neutral_count': 0,
            'avg_confidence': 0,
            'high_confidence_signals': [],
            'credibility_insights': {},
            'timeframe_insights': {}
        }
        confidences = []
        for symbol, analysis in symbol_analyses.items():
            prediction = analysis.get('prediction', {})
            signal = prediction.get('signal', 'NEUTRAL')
            confidence = prediction.get('confidence', 0.5)
            confidences.append(confidence)
            if signal == 'BULLISH':
                summary['bullish_count'] += 1
            elif signal == 'BEARISH':
                summary['bearish_count'] += 1
            else:
                summary['neutral_count'] += 1
            if confidence > 0.7:
                summary['high_confidence_signals'].append({'symbol': symbol, 'signal': signal, 'confidence': confidence})
        if confidences:
            summary['avg_confidence'] = float(np.mean(confidences))
        return summary

    def _fallback_suggestions(self, symbol_analyses: Dict[str, Any]) -> List[str]:
        """Fallback when _generate_enhanced_suggestions is not available."""
        suggestions = []
        for symbol in list(symbol_analyses.keys())[:5]:
            suggestions.append(f"Show risk analysis for {symbol}")
            suggestions.append(f"Compare {symbol} with sector")
        suggestions.extend([
            "Overall market sentiment",
            "Cross-sector correlation",
            "Volatility regime",
        ])
        return suggestions[:8]
    
    def _detect_intent_and_timeframe(self, query: str) -> Tuple[List[str], Optional[TimeframeScope]]:
        """Detect user intent and preferred timeframe from query."""
        query_lower = query.lower()
        detected_intents = []
        detected_timeframe = None
        
        # Detect intents
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    detected_intents.append(intent)
                    break
        
        # Detect specific timeframe mentions
        if any(word in query_lower for word in ['scalp', 'seconds', 'minutes', 'quick']):
            detected_timeframe = TimeframeScope.SCALPING
        elif any(word in query_lower for word in ['intraday', 'day trade', 'same day', 'hours']):
            detected_timeframe = TimeframeScope.INTRADAY
        elif any(word in query_lower for word in ['swing', 'days', 'week', 'short term']):
            detected_timeframe = TimeframeScope.SWING
        elif any(word in query_lower for word in ['position', 'weeks', 'month', 'medium term']):
            detected_timeframe = TimeframeScope.POSITION
        elif any(word in query_lower for word in ['invest', 'long term', 'months', 'years']):
            detected_timeframe = TimeframeScope.INVESTMENT
        
        # Default to analysis if no clear intent
        if not detected_intents:
            detected_intents = ['analysis']
        
        return list(set(detected_intents)), detected_timeframe
    
    def _extract_symbols(self, query: str) -> List[str]:
        """COMPLETELY UNBIASED symbol extraction - treats ALL symbols equally."""
        query_upper = query.upper()
        symbols = []
        
        # Basic English stopwords to avoid treating normal words as tickers
        english_stopwords = {
            "THE", "AND", "OR", "BUT", "IF", "THEN", "ELSE", "WHEN", "WHAT", "WHY", "HOW",
            "IS", "ARE", "AM", "WAS", "WERE", "BE", "BEEN", "BEING", "DO", "DID", "DOES",
            "I", "YOU", "HE", "SHE", "IT", "WE", "THEY", "ME", "HIM", "HER", "US", "THEM",
            "THIS", "THAT", "THESE", "THOSE",
            "IN", "ON", "AT", "BY", "FOR", "WITH", "ABOUT", "AGAINST", "BETWEEN", "INTO",
            "THROUGH", "DURING", "BEFORE", "AFTER", "ABOVE", "BELOW", "FROM", "UP", "DOWN",
            "OUT", "OVER", "UNDER", "AGAIN", "FURTHER", "ONCE", "TO", "A", "AN", "AS",
            "SECTOR", "SECTORS", "SHORT", "LONG", "LOOK", "RIGHT", "NOW", "BEAR", "BEARISH",
            "BULL", "BULLISH", "EXTREME", "EXTREMELY", "POTENTIAL", "CANDIDATE", "CANDIDATES",
            "TICKER", "TICKERS", "THROUGHOUT", "SEEM", "SEEMS", "VERY", "MOST", "WORST", "BEST",
            "FOR", "POSITION", "PRICES", "PRICE", "TREND", "TRENDS", "MARKET", "MARKETS"
        }
        
        # Word to ticker mapping for common commodities and indices
        word_to_ticker = {
            "OIL": "CL=F",
            "GOLD": "GC=F",
            "SILVER": "SI=F",
            "COPPER": "HG=F",
            "BITCOIN": "BTC-USD",
            "ETHEREUM": "ETH-USD",
            "SPX": "SPY",
            "NASDAQ": "QQQ",
            "DOW": "DIA",
            "VIX": "^VIX",
        }
        
        # NO BIAS - Extract ALL potential symbols equally, but avoid obvious English words
        potential_symbols = re.findall(r'\b[A-Z]{1,6}\b', query_upper)
        
        for symbol in potential_symbols:
            if symbol in english_stopwords:
                continue
            if symbol in word_to_ticker:
                symbols.append(word_to_ticker[symbol])
                continue
            # Allow typical ticker patterns: at least 2 chars or includes a digit
            if len(symbol) >= 2 or any(ch.isdigit() for ch in symbol):
                symbols.append(symbol)
        
        # Enhanced FX pair detection - restrict to real currency codes to avoid false positives
        currency_codes = {
            "EUR", "USD", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD", "SEK", "NOK", "DKK",
            "PLN", "CZK", "HUF", "TRY", "ZAR", "MXN", "BRL", "CNY", "INR", "KRW", "SGD",
            "HKD", "THB", "MYR", "IDR", "PHP",
        }

        fx_matches = re.findall(r'\b([A-Z]{3})[\/_-]([A-Z]{3})\b', query_upper)
        fx_matches += re.findall(r'\b([A-Z]{3})([A-Z]{3})\b', query_upper)

        for c1, c2 in fx_matches:
            if c1 in currency_codes and c2 in currency_codes and c1 != c2:
                symbols.append(f"{c1}/{c2}")
        
        # Enhanced crypto detection - ALL cryptos treated equally
        crypto_patterns = [
            r'(BTC|BITCOIN|ETH|ETHEREUM|BNB|BINANCE|XRP|RIPPLE|ADA|CARDANO|SOL|SOLANA|DOGE|DOGECOIN|DOT|POLKADOT|AVAX|AVALANCHE|SHIB|SHIBAINU|MATIC|POLYGON|LTC|LITECOIN|ALGO|ALGORAND|ATOM|COSMOS|LINK|CHAINLINK|XLM|STELLAR|VET|VECHAIN|ICP|INTERNET|NEAR|PROTOCOL|FTM|FANTOM|SAND|SANDBOX|MANA|DECENTRALAND|CRV|CURVE|UNI|UNISWAP|SUSHI|SUSHISWAP|COMP|COMPOUND|AAVE|MKR|MAKER|SNX|SYNTHETIX|YFI|YEARN|BAL|BALANCER|REN|REPUBLIC|ZRX|ZEROX|BAT|BASIC|ATTENTION|TOKEN)[-]?USD',
        ]
        
        for pattern in crypto_patterns:
            matches = re.findall(pattern, query_upper)
            for match in matches:
                if isinstance(match, str):
                    symbols.append(f"{match}-USD")
                elif isinstance(match, tuple):
                    symbols.append(f"{match[0]}-USD")
        
        # Futures detection - ALL futures treated equally
        futures_symbols = {
            'ES': 'ES=F', 'NQ': 'NQ=F', 'YM': 'YM=F', 'RTY': 'RTY=F',
            'CL': 'CL=F', 'GC': 'GC=F', 'SI': 'SI=F', 'ZN': 'ZN=F',
            'ZB': 'ZB=F', 'ZF': 'ZF=F', 'ZC': 'ZC=F', 'ZS': 'ZS=F',
            'NG': 'NG=F', 'HG': 'HG=F', 'PA': 'PA=F', 'PL': 'PL=F',
            'KC': 'KC=F', 'SB': 'SB=F', 'CC': 'CC=F', 'CT': 'CT=F'
        }
        
        for symbol in symbols.copy():
            if symbol in futures_symbols:
                symbols.append(futures_symbols[symbol])
        
        # Remove duplicates while preserving order
        symbols = list(dict.fromkeys(symbols))
        
        # NO BIAS - Return ALL symbols found, don't prioritize "well-known" ones
        return symbols[:10]  # Reasonable limit for performance
    
    def _generate_opportunity_charts(self, opportunities: list) -> list:
        """Generate summary charts from scan opportunities.

        Since UnbiasedAnalysis structs carry no raw price data, we generate
        a simple Plotly bar chart showing profit probability, expected return,
        and confidence for the top opportunities.
        """
        if not opportunities:
            return []
        charts = []
        try:
            top = opportunities[:15]
            labels = [o.symbol for o in top]
            probs = [o.profit_probability * 100 for o in top]
            rets = [o.expected_return * 100 for o in top]
            confs = [o.confidence_score * 100 for o in top]

            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.06,
                subplot_titles=("Profit Probability (%)", "Expected Return (%)", "Confidence Score (%)"),
                row_heights=[0.34, 0.33, 0.33],
            )
            fig.add_trace(go.Bar(x=labels, y=probs, marker_color="green", name="Profit Prob"), row=1, col=1)
            fig.add_trace(go.Bar(x=labels, y=rets, marker_color="royalblue", name="Exp Return"), row=2, col=1)
            fig.add_trace(go.Bar(x=labels, y=confs, marker_color="orange", name="Confidence"), row=3, col=1)
            fig.update_layout(
                height=500,
                title_text="Top Scan Opportunities",
                template="plotly_dark",
                showlegend=False,
            )
            charts.append({
                "type": "opportunity_summary",
                "title": "Top Scan Opportunities",
                "figure": fig,
            })
        except Exception as e:
            print(f"Error generating opportunity charts: {e}")
        return charts

    async def _perform_unbiased_market_scan(self, query: str, intents: List[str], 
                                          analysis_timeframe: TimeframeScope,
                                          trader_context: Dict[str, Any], 
                                          start_time: float) -> Dict[str, Any]:
        """Perform complete unbiased market scan for profit opportunities."""
        
        print("[SCAN] Performing unbiased market scan across ALL assets...")
        
        # Scan entire market without bias
        market_opportunities = await self.unbiased_analyzer.scan_entire_market(max_symbols=200)
        
        # Sort by pure profit potential
        market_opportunities.sort(
            key=lambda x: x.profit_probability * x.confidence_score * (1 + abs(x.expected_return)), 
            reverse=True
        )
        
        # Get top opportunities
        top_opportunities = market_opportunities[:20]
        
        # Use response formatter for clean user text
        formatted = self.response_formatter.format_scan_response(
            opportunities=top_opportunities,
            timeframe=analysis_timeframe.value
        )
        
        response = {
            'text': formatted.user_text,
            'raw_analysis': formatted.raw_analysis,
            'show_reasoning_available': formatted.show_reasoning_available,
            'response_metadata': formatted.response_metadata,
            'charts': self._generate_opportunity_charts(top_opportunities),
            'intents': intents,
            'symbols': [opp.symbol for opp in top_opportunities[:10]],
            'timeframe_context': analysis_timeframe.value,
            'analysis_summary': self._create_opportunity_summary(top_opportunities),
            'unbiased_insights': self._extract_scan_insights(top_opportunities),
            'profit_opportunities': self._format_profit_opportunities(top_opportunities),
            'simulation_learnings': await self._get_simulation_learnings([]),
            'suggestions': self._generate_scan_suggestions(top_opportunities, analysis_timeframe),
            'response_time_ms': int((time.time() - start_time) * 1000)
        }
        
        return response
    
    @property
    def llm_agent(self):
        return None

    async def process_enhanced_query(self, query: str, user_id: str = None) -> dict:
        import time
        from financial_llm_engine import generate_financial_analysis, expand_query_intents
        start_time = time.time()

        # Extract intents, tickers, and sectors for metadata
        intents, tickers, sectors = expand_query_intents(query)

        # Optionally pull heartbeat context for richer grounding
        heartbeat_ctx = None
        try:
            from market_heartbeat_system import fetch_heartbeat_data, generate_market_heartbeat
            g, u = fetch_heartbeat_data()
            if g and u:
                hb = generate_market_heartbeat(g, u)
                heartbeat_ctx = f"Market State: {hb.get('env_class', ('Unknown',''))[0]} | VIX regime: {hb.get('vol', ('Unknown',''))[0]}"
        except Exception:
            pass

        # Generate the deep financial analysis with live data grounding
        response_text = generate_financial_analysis(query, context_data=heartbeat_ctx)

        # Build detected intent labels
        active_intents = [k for k, v in intents.items() if v]
        if sectors:
            active_intents.extend([s.replace("_", " ").title() for s in sectors])

        # Generate charts for the analyzed symbols — but ONLY when the query
        # actually warrants visuals. Knowledge/definitional queries ("tell me
        # about COST", "who are X's competitors", "does X pay a dividend") get
        # no charts: a price chart adds nothing there and costs latency.
        #
        # REVIEW FIX 1 (the 4-chart problem): deep-dive institutional theses
        # NEVER get generic price charts. They get exactly FOUR distinct
        # analytical charts built from the memo's own numbers — expectation
        # gap, scenario distribution, valuation sensitivity, risk/reward —
        # each with a unique chart ID, analytical purpose, source dataset,
        # timestamp, provenance, calculation, run identifier and a dataset
        # hash so identical datasets can never silently reuse a prior figure.
        charts = []
        _is_dd = False
        try:
            from financial_llm_engine import _is_deep_dive_query
            _is_dd = _is_deep_dive_query(query, tickers)
        except Exception:
            _is_dd = False
        if _is_dd:
            try:
                from deep_dive_charts import build_deep_dive_charts
                from financial_llm_engine import (_dd_fetch_fundamentals,
                                                  _fetch_live_data_for_tickers)
                _fund = {}
                try:
                    _fund = _dd_fetch_fundamentals(tickers[0]) if tickers else {}
                except Exception:
                    _fund = {}
                _live = _fetch_live_data_for_tickers(tickers[:3]) if tickers else {}
                charts = build_deep_dive_charts(
                    query, tickers, sectors, _live, _fund or None)
            except Exception as e:
                print(f"Deep-dive chart generation error: {e}")
                charts = []
        elif _query_warrants_charts(query):
            try:
                from data_sources import get_stock

                def _fetch_and_chart(sym):
                    try:
                        df = get_stock(sym, period='6mo', interval='1d')
                        if df is not None and not df.empty and 'Close' in df.columns:
                            fig = self._create_advanced_price_chart(df, sym, True, 'standard')
                            if fig is not None:
                                return {
                                    'type': 'price_analysis',
                                    'symbol': sym,
                                    'figure': fig,
                                    'title': f'{sym} — Price Analysis (6M)',
                                }
                    except Exception as e:
                        print(f"Chart error for {sym}: {e}")
                    return None

                with ThreadPoolExecutor(max_workers=min(6, len(tickers))) as pool:
                    results = list(pool.map(_fetch_and_chart, tickers[:6]))
                charts = [r for r in results if r is not None]
            except Exception as e:
                print(f"Chart generation error: {e}")

        return {
            'text': response_text,
            'charts': charts,
            'intents': active_intents if active_intents else ['general_analysis'],
            'symbols': tickers[:10],
            'timeframe_context': 'swing',
            'trader_profile': 'neutral',
            'suggestions': self._generate_followup_suggestions(query, tickers, sectors),
            'response_time_ms': int((time.time() - start_time) * 1000),
        }

    def _generate_followup_suggestions(self, query, tickers, sectors):
        """Generate contextual follow-up suggestions based on the query."""
        suggestions = []
        if tickers:
            suggestions.append(f"Deep dive on {tickers[0]}")
            suggestions.append(f"Risk analysis for {tickers[0]}")
        if sectors:
            s = sectors[0].replace("_", " ").title()
            suggestions.append(f"Bearish outlook for {s}")
            suggestions.append(f"Top dividend stocks in {s}")
        if not tickers and not sectors:
            suggestions.extend(["Current macro regime analysis", "Best sectors this week", "VIX and volatility outlook"])
        return suggestions[:4]

    async def process_unbiased_query(self, query: str, user_id: str = None) -> dict:
        """Second process_unbiased_query that delegates to enhanced for LLM-rich path
        BUT also dispatches to the first process_unbiased_query when scan intents detected."""
        from financial_llm_engine import expand_query_intents

        # Portfolio-aware routing: questions about the user's own holdings are
        # answered from their actual paper/manual portfolio data.
        try:
            from portfolio_chatbot_context import (
                detect_portfolio_intent, handle_portfolio_query,
                load_all_portfolios, _get_all_holdings,
            )
            pf_intent = detect_portfolio_intent(query)
            if pf_intent:
                ctx = load_all_portfolios(user_id)
                holdings = _get_all_holdings(ctx) if ctx.get('has_data') else []
                res = handle_portfolio_query(query, pf_intent, user_id)
                if asyncio.iscoroutine(res):
                    res = await res
                return {
                    'text': res if isinstance(res, str) else str(res),
                    'charts': [],
                    'intents': [pf_intent],
                    'symbols': [h.get('symbol') for h in holdings if h.get('symbol')],
                    'suggestions': [],
                    'response_time_ms': 0,
                }
        except Exception:
            pass

        _, tickers, sectors = expand_query_intents(query)
        if not tickers and 'scan' in query.lower():
            return await self._perform_unbiased_market_scan(
                query, ['unbiased_scan'], TimeframeScope.SWING, {}, time.time()
            )
        return await self.process_enhanced_query(query, user_id)

    def _generate_market_wide_brief(self, analysis_timeframe: TimeframeScope) -> str:
        try:
            assets = {
                "Equities": ["SPY", "QQQ"],
                "Rates": ["TLT", "IEF"],
                "Vol": ["^VIX"],
                "Futures": ["ES=F", "NQ=F", "CL=F", "GC=F"],
                "FX": ["EUR_USD", "USD_JPY"],
            }

            def _ret_5d(sym: str) -> float | None:
                try:
                    # Get chart timeframe from settings
                    chart_settings = st.session_state.get('chart_settings', {})
                    chart_period = chart_settings.get('chart_timeframe', '6 Months')
                    
                    # Convert display period to yfinance period
                    period_map = {
                        '1 Month': '1mo',
                        '3 Months': '3mo', 
                        '6 Months': '6mo',
                        '1 Year': '1y',
                        '2 Years': '2y'
                    }
                    period = period_map.get(chart_period, '6mo')
                    
                    if sym in ("EUR_USD", "USD_JPY"):
                        df = get_fx(sym)
                    elif "=F" in sym or sym.endswith("=f"):
                        df = get_futures_proxy(sym, period=period)
                    else:
                        df = get_stock(sym, period=period)
                    if df is None or df.empty or "Close" not in df.columns or len(df) < 6:
                        return None
                    close = df["Close"]
                    if isinstance(close, pd.DataFrame):
                        close = close.iloc[:, 0]
                    prev = float(close.iloc[-6])
                    curr = float(close.iloc[-1])
                    if prev == 0:
                        return None
                    return ((curr / prev) - 1) * 100
                except Exception:
                    return None

            lines = []
            lines.append("#  Market-Wide Brief")
            lines.append("")
            lines.append(f"**Timeframe context:** {analysis_timeframe.value.replace('_', ' ').title()}")
            lines.append("")

            try:
                vol_regime = volatility_regime()
                risk_mode = risk_on_off()
                lines.append(f"**Regime:** {vol_regime} volatility | {risk_mode}")
                lines.append("")
            except Exception:
                pass

            lines.append("## [PIN] Cross-Asset Tape (5D move)")
            for group, syms in assets.items():
                parts = []
                for s in syms:
                    r5 = _ret_5d(s)
                    if r5 is None:
                        parts.append(f"`{s}`: n/a")
                    else:
                        parts.append(f"`{s}`: {r5:+.2f}%")
                lines.append(f"- **{group}:** " + " | ".join(parts))

            lines.append("")

            try:
                sector_df = scan_sectors()
                if sector_df is not None and not sector_df.empty:
                    bearish = sector_df.nsmallest(min(3, len(sector_df)), 'TrendScore')
                    bullish = sector_df.nlargest(min(3, len(sector_df)), 'TrendScore')
                    lines.append("##  Sector Rotation Snapshot")
                    lines.append("- **Weakest:** " + ", ".join([str(x) for x in bearish['Sector'].tolist()]))
                    lines.append("- **Strongest:** " + ", ".join([str(x) for x in bullish['Sector'].tolist()]))
                    lines.append("")
            except Exception:
                pass

            try:
                fetched = self.news_engine.fetch_and_process_news()
                if fetched:
                    fetched = sorted(fetched, key=lambda a: (a.relevance_score or 0), reverse=True)[:15]
                    avg_sent = float(np.mean([float(a.sentiment_score or 0) for a in fetched])) if fetched else 0.0
                    lines.append("##  Market News Pulse")
                    lines.append(f"- **Credibility-blind avg sentiment (top headlines):** {avg_sent:+.2f}")
                    lines.append("- **Top headlines:**")
                    for a in fetched[:10]:
                        src = a.source
                        title = a.title
                        lines.append(f"  - {title} ({src})")
                    lines.append("")
            except Exception:
                pass

            lines.append("##  Trade Framing")
            lines.append("- If equities are weak while vol is rising, favor defensive posture / short rallies.")
            lines.append("- If USD is strengthening, watch pressure on commodities and EM risk.")
            lines.append("- Pair trades: short weakest sector vs long strongest sector to reduce beta.")

            return "\n".join(lines)
        except Exception:
            return "Unable to generate a market-wide brief right now. Please try again."
    
    def _analyze_symbol_unbiased(self, symbol: str, intents: List[str], 
                               timeframe_scope: TimeframeScope, 
                               trader_context: Dict[str, Any]) -> Dict[str, Any]:
        """COMPLETELY UNBIASED symbol analysis focused on PURE PROFIT MAXIMIZATION."""
        try:
            # Use unbiased analyzer - NO BIAS toward popular stocks
            unbiased_analysis = asyncio.run(self.unbiased_analyzer.analyze_unbiased(symbol))
            
            if not unbiased_analysis or unbiased_analysis.confidence_score < 0.1:
                return None
            
            analysis = {
                'symbol': symbol,
                'unbiased_analysis': unbiased_analysis,
                'data': getattr(self.unbiased_analyzer, '_last_fetched_data', None),
                'profit_probability': unbiased_analysis.profit_probability,
                'expected_return': unbiased_analysis.expected_return,
                'risk_adjusted_return': unbiased_analysis.risk_adjusted_return,
                'confidence_score': unbiased_analysis.confidence_score,
                'timeframe_scope': timeframe_scope.value,
                'timestamp': datetime.now().isoformat()
            }
            
            # Get additional context if high confidence
            if unbiased_analysis.confidence_score > 0.6:
                # Get timeframe-specific analysis
                try:
                    timeframe_analysis = asyncio.run(
                        self.timeframe_engine.analyze_timeframe_context(
                            symbol, timeframe_scope
                        )
                    )
                    analysis['timeframe_analysis'] = timeframe_analysis
                except Exception as e:
                    print(f"Timeframe analysis error for {symbol}: {e}")
                
                # Get credibility-weighted news if requested
                if 'news_sentiment' in intents or 'comprehensive' in intents:
                    try:
                        news_data = self.news_engine.get_news_for_symbol(symbol, limit=30)
                        if news_data:
                            weighted_news = self.source_credibility_engine.get_weighted_news_summary(
                                news_data, timeframe_scope.value
                            )
                            analysis['credibility_weighted_news'] = weighted_news
                    except Exception as e:
                        print(f"News analysis error for {symbol}: {e}")
            
            return analysis
            
        except Exception as e:
            print(f"Unbiased analysis error for {symbol}: {e}")
            return None

    def _analyze_symbol_enhanced(self, symbol: str, intents: List[str], 
                               timeframe_scope: TimeframeScope, 
                               trader_context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced symbol analysis used by process_enhanced_query.

        Delegates to the unbiased analysis pipeline AND maps the result into
        the data structure that _generate_enhanced_symbol_analysis expects,
        including 'prediction', 'metrics', 'technical', and 'asset_type'.
        """
        # Get the raw unbiased analysis
        raw_result = self._analyze_symbol_unbiased(symbol, intents, timeframe_scope, trader_context)
        if raw_result is None:
            return None

        # Pull out the UnbiasedAnalysis object
        ua = raw_result.get('unbiased_analysis')
        if ua is None:
            return None

        # --- Fetch live price data for real metrics ---
        # Try using the DataFrame already fetched by the unbiased analyzer
        # to avoid a second expensive network call
        current_price = 0.0
        daily_change = 0.0
        asset_type = 'Stock'
        price_df = raw_result.get('data')  # DataFrame from unbiased analysis
        try:
            if price_df is None or (hasattr(price_df, 'empty') and price_df.empty):
                # Fallback: fetch only if unbiased analyzer didn't provide data
                price_df, asset_type = self._get_real_time_data(symbol, period='3mo')
            else:
                # Determine asset type from symbol
                if '/' in symbol or '=X' in symbol:
                    asset_type = 'Forex'
                elif '=F' in symbol:
                    asset_type = 'Futures'
                elif symbol.endswith('-USD'):
                    asset_type = 'Crypto'
                else:
                    # Dynamically check against the ticker universe ETF list
                    _is_etf = False
                    if _HAS_TICKER_UNIVERSE:
                        try:
                            _is_etf = symbol in set(get_ticker_universe().get_etfs())
                        except Exception:
                            pass
                    if _is_etf:
                        asset_type = 'ETF'
                    else:
                        asset_type = 'Stock'
            
            if price_df is not None and not price_df.empty:
                close = price_df['Close']
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                close = close.dropna().astype(float)
                if len(close) >= 2:
                    current_price = float(close.iloc[-1])
                    prev_price = float(close.iloc[-2])
                    daily_change = ((current_price / prev_price) - 1) * 100 if prev_price > 0 else 0.0
                elif len(close) == 1:
                    current_price = float(close.iloc[-1])
        except Exception as e:
            print(f"Price fetch for enhanced analysis of {symbol}: {e}")

        # --- Determine signal direction ---
        signal_direction = getattr(ua, 'signal_direction', 'NEUTRAL')
        profit_prob = ua.profit_probability
        confidence = ua.confidence_score

        # --- Build 'prediction' dict ---
        prediction = {
            'signal': signal_direction,
            'confidence': confidence,
            'bullish_prob': profit_prob,
            'expected_return': ua.expected_return,
            'risk_adjusted_return': ua.risk_adjusted_return,
        }

        # --- Build 'metrics' dict ---
        metrics = {
            'current_price': current_price,
            'daily_change': daily_change,
        }

        # --- Build 'technical' dict from entry/exit signals and reasoning ---
        trend = signal_direction  # Use overall direction
        primary_signals = ua.entry_signals[:3] if ua.entry_signals else []
        secondary_signals = ua.exit_signals[:2] if ua.exit_signals else []

        technical = {
            'trend': trend,
            'primary_signals': primary_signals,
            'secondary_signals': secondary_signals,
        }

        # --- Assemble the full result expected by the response generators ---
        result = {
            'symbol': symbol,
            'asset_type': asset_type,
            'prediction': prediction,
            'metrics': metrics,
            'technical': technical,
            'data': price_df,  # DataFrame for chart generation
            'entry_signals': ua.entry_signals,
            'exit_signals': ua.exit_signals,
            'risk_factors': ua.risk_factors,
            'profit_catalysts': ua.profit_catalysts,
            'model_reasoning': ua.model_reasoning,
            'raw_data_insights': ua.raw_data_insights,
            'unbiased_analysis': ua,
            'profit_probability': profit_prob,
            'expected_return': ua.expected_return,
            'risk_adjusted_return': ua.risk_adjusted_return,
            'confidence_score': confidence,
            'timeframe_scope': timeframe_scope.value,
            'timestamp': datetime.now().isoformat(),
        }

        # Copy optional enrichments from _analyze_symbol_unbiased
        for enrichment_key in ('timeframe_analysis', 'credibility_weighted_news'):
            if enrichment_key in raw_result:
                result[enrichment_key] = raw_result[enrichment_key]

        return result
    
    async def _generate_unbiased_profit_response(self, query: str, symbol_analyses: Dict[str, Any], 
                                               intents: List[str], timeframe_scope: TimeframeScope,
                                               market_context: Dict[str, Any], 
                                               trader_context: Dict[str, Any]) -> str:
        """Generate UNBIASED response focused on PURE PROFIT MAXIMIZATION."""
        
        # Get trader profile info
        recommendation_style = get_recommendation_style()
        should_specify_timeframe = recommendation_style.get('should_specify_timeframe', True) if hasattr(recommendation_style, 'get') else True
        
        # Start with Octavian branding - UNBIASED PROFIT FOCUS
        response = f"""#  **OCTAVIAN** by APB - Unbiased Profit Intelligence
*Pure Mathematical Analysis • Zero Bias • Maximum Profit Focus*

"""
        
        # Add INTENT-SPECIFIC header to make responses unique
        response += self._generate_intent_specific_header(query, intents, timeframe_scope)
        
        # Add timeframe context if needed
        if should_specify_timeframe:
            response += f"""##  Analysis Timeframe: **{timeframe_scope.value.replace('_', ' ').title()}**
*Unbiased analysis optimized for {timeframe_scope.value.replace('_', ' ')} profit maximization*

"""
        
        # Add market context
        if market_context:
            vol_regime = market_context.get('volatility_regime', 'Unknown')
            risk_mode = market_context.get('risk_mode', 'Unknown')
            
            response += f"""##  Current Market Environment
- **Volatility Regime:** {vol_regime}
- **Risk Sentiment:** {risk_mode}
- **Assets Analyzed:** {len(symbol_analyses)} (selected without bias)
- **Analysis Focus:** {self._get_focus_from_intents(intents)}

---

"""
        
        # Analyze each symbol with UNBIASED insights
        for symbol, analysis in symbol_analyses.items():
            response += await self._generate_unbiased_symbol_analysis(
                symbol, analysis, timeframe_scope, trader_context, intents
            )
            response += "\n---\n\n"
        
        # Add profit maximization insights
        response += self._generate_profit_maximization_section(symbol_analyses)
        
        # Add unbiased recommendations
        response += self._generate_unbiased_recommendations(
            symbol_analyses, timeframe_scope, trader_context
        )
        
        # Add Octavian unbiased signature
        response += self._generate_octavian_unbiased_signature(symbol_analyses, market_context)
        
        return response
    
    def _generate_intent_specific_header(self, query: str, intents: List[str], timeframe_scope: TimeframeScope) -> str:
        """Generate unique header based on detected intents."""
        if 'risk' in intents:
            return f"""##  Risk Analysis Request
*Query: "{query}"*
*Focus: Comprehensive risk metrics, volatility analysis, and downside protection*

"""
        elif 'prediction' in intents:
            return f"""##  Prediction & Forecast Request
*Query: "{query}"*
*Focus: Forward-looking analysis, price targets, and directional bias*

"""
        elif 'price_query' in intents:
            return f"""##  Price Information Request
*Query: "{query}"*
*Focus: Current pricing, valuation levels, and market positioning*

"""
        elif 'news_sentiment' in intents:
            return f"""##  News & Sentiment Analysis
*Query: "{query}"*
*Focus: Market sentiment, news impact, and catalyst analysis*

"""
        elif 'sector' in intents:
            return f"""##  Sector Analysis Request
*Query: "{query}"*
*Focus: Sector performance, rotation signals, and relative strength*

"""
        elif 'comprehensive' in intents:
            return f"""##  Comprehensive Analysis Request
*Query: "{query}"*
*Focus: Full-spectrum analysis across all dimensions*

"""
        else:
            return f"""##  Analysis Request
*Query: "{query}"*
*Focus: Technical and fundamental analysis*

"""
    
    def _get_focus_from_intents(self, intents: List[str]) -> str:
        """Get analysis focus description from intents."""
        if 'risk' in intents:
            return "Risk metrics and volatility analysis"
        elif 'prediction' in intents:
            return "Predictive modeling and forecasting"
        elif 'price_query' in intents:
            return "Current pricing and valuation"
        elif 'news_sentiment' in intents:
            return "News sentiment and catalyst tracking"
        elif 'sector' in intents:
            return "Sector analysis and rotation"
        else:
            return "Pure profit maximization"
    
    def _generate_risk_analysis_section(self, symbol: str, analysis: Dict[str, Any]) -> str:
        """Generate detailed risk analysis section when risk intent is detected."""
        text = "###  RISK ANALYSIS (Requested)\n"
        
        # Get risk metrics from analysis
        risk_data = analysis.get('risk', {})
        technical_data = analysis.get('technical', {})
        unbiased_analysis = analysis.get('unbiased_analysis')
        
        if risk_data:
            # Volatility metrics
            volatility = risk_data.get('volatility', 0)
            beta = risk_data.get('beta', 0)
            sharpe = risk_data.get('sharpe_ratio', 0)
            max_drawdown = risk_data.get('max_drawdown', 0)
            
            text += f"- **Volatility (Annualized):** {volatility:.2%}\n"
            text += f"- **Beta (Market Correlation):** {beta:.2f}\n"
            text += f"- **Sharpe Ratio:** {sharpe:.2f}\n"
            text += f"- **Maximum Drawdown:** {max_drawdown:.2%}\n"
            
            # Risk classification
            if volatility > 0.40:
                risk_level = "HIGH RISK - Extreme volatility"
            elif volatility > 0.25:
                risk_level = "MODERATE-HIGH RISK - Above average volatility"
            elif volatility > 0.15:
                risk_level = "MODERATE RISK - Average volatility"
            else:
                risk_level = "LOW-MODERATE RISK - Below average volatility"
            
            text += f"- **Risk Classification:** {risk_level}\n"
        
        # Add risk-adjusted return if available
        if unbiased_analysis:
            risk_adjusted_return = unbiased_analysis.risk_adjusted_return
            text += f"- **Risk-Adjusted Return Score:** {risk_adjusted_return:.2f}\n"
        
        # Technical risk indicators
        if technical_data:
            rsi = technical_data.get('rsi', 50)
            if rsi > 70:
                text += f"- **Overbought Risk:** RSI at {rsi:.1f} (potential pullback)\n"
            elif rsi < 30:
                text += f"- **Oversold Opportunity:** RSI at {rsi:.1f} (potential bounce)\n"
        
        text += "\n"
        return text
    
    async def _generate_unbiased_symbol_analysis(self, symbol: str, analysis: Dict[str, Any],
                                               timeframe_scope: TimeframeScope,
                                               trader_context: Dict[str, Any],
                                               intents: List[str] = None) -> str:
        """Generate UNBIASED symbol analysis focused on PROFIT POTENTIAL with intent-specific emphasis."""
        
        if intents is None:
            intents = []
        
        unbiased_analysis = analysis.get('unbiased_analysis')
        if not unbiased_analysis or not isinstance(unbiased_analysis, dict):
            return f"##  {symbol} - Analysis Failed\nInsufficient data for unbiased analysis.\n"
        
        profit_probability = unbiased_analysis.profit_probability
        expected_return = unbiased_analysis.expected_return
        risk_adjusted_return = unbiased_analysis.risk_adjusted_return
        confidence_score = unbiased_analysis.confidence_score
        
        # Header with PROFIT FOCUS
        text = f"""##  {symbol} - Pure Profit Analysis (No Bias)

###  Profit Metrics
- **Profit Probability:** {profit_probability:.1%}
- **Expected Return:** {expected_return:+.2%}
- **Risk-Adjusted Return:** {risk_adjusted_return:.2f}
- **Model Confidence:** {confidence_score:.1%}

"""
        
        # INTENT-SPECIFIC SECTIONS - Show risk analysis prominently if requested
        if 'risk' in intents:
            text += self._generate_risk_analysis_section(symbol, analysis)
        
        # Entry/Exit Signals (Pure Mathematical)
        if unbiased_analysis.entry_signals:
            text += "###  Entry Signals (Mathematical)\n"
            for signal in unbiased_analysis.entry_signals[:3]:
                text += f"- {signal}\n"
            text += "\n"
        
        if unbiased_analysis.exit_signals:
            text += "###  Exit Signals (Risk Management)\n"
            for signal in unbiased_analysis.exit_signals[:3]:
                text += f"- {signal}\n"
            text += "\n"
        
        # Profit Catalysts
        if unbiased_analysis.profit_catalysts:
            text += "###  Profit Catalysts\n"
            for catalyst in unbiased_analysis.profit_catalysts[:3]:
                text += f"- {catalyst}\n"
            text += "\n"
        
        # Risk Factors (Unbiased)
        if unbiased_analysis.risk_factors:
            text += "###  Risk Factors\n"
            for risk in unbiased_analysis.risk_factors[:3]:
                text += f"- {risk}\n"
            text += "\n"
        
        # Model Reasoning (Transparent)
        if unbiased_analysis.model_reasoning:
            text += "###  Model Logic\n"
            for reasoning in unbiased_analysis.model_reasoning[:3]:
                text += f"- {reasoning}\n"
            text += "\n"
        
        # Timeframe-specific insights
        timeframe_analysis = analysis.get('timeframe_analysis')
        if timeframe_analysis and isinstance(timeframe_analysis, dict) and not timeframe_analysis.current_context.get('error'):
            text += f"###  {timeframe_scope.value.replace('_', ' ').title()} Specific Insights\n"
            
            actionable_insights = timeframe_analysis.actionable_insights
            if actionable_insights:
                for insight in actionable_insights[:2]:
                    text += f"- {insight}\n"
            
            # Forward projections
            forward_projections = timeframe_analysis.forward_projections
            if forward_projections:
                expected_return_tf = forward_projections.get('probability_weighted_outcome', {}).get('expected_return', 0)
                text += f"- **Timeframe Projection:** {expected_return_tf:+.1%} expected return\n"
            
            text += "\n"
        
        # Source-weighted news (if available)
        credibility_news = analysis.get('credibility_weighted_news', {})
        if credibility_news and isinstance(credibility_news, dict) and not credibility_news.get('error'):
            text += "###  Credibility-Weighted News Impact\n"
            
            weighted_sentiment = credibility_news.get('weighted_sentiment', 0)
            total_weight = credibility_news.get('total_weight', 0)
            item_count = credibility_news.get('item_count', 0)
            
            sentiment_label = self._sentiment_to_label(weighted_sentiment)
            
            text += f"- **News Sentiment:** {sentiment_label} ({weighted_sentiment:+.2f})\n"
            text += f"- **Source Quality:** {total_weight:.1f} credibility weight from {item_count} articles\n"
            
            # Source breakdown
            source_breakdown = credibility_news.get('source_breakdown', {})
            if source_breakdown:
                tier1_count = source_breakdown.get('tier_1_premium', 0)
                if tier1_count > 0:
                    text += f"- **Premium Sources:** {tier1_count} articles from Bloomberg/Reuters/WSJ\n"
            
            text += "\n"
        
        return text
    
    def _generate_profit_maximization_section(self, symbol_analyses: Dict[str, Any]) -> str:
        """Generate profit maximization insights section."""
        text = "##  Profit Maximization Intelligence\n\n"
        
        # Rank symbols by profit potential
        profit_rankings = []
        for symbol, analysis in symbol_analyses.items():
            unbiased_analysis = analysis.get('unbiased_analysis')
            if unbiased_analysis:
                profit_score = (
                    unbiased_analysis.profit_probability * 
                    unbiased_analysis.confidence_score * 
                    (1 + abs(unbiased_analysis.expected_return))
                )
                profit_rankings.append({
                    'symbol': symbol,
                    'profit_score': profit_score,
                    'profit_probability': unbiased_analysis.profit_probability,
                    'expected_return': unbiased_analysis.expected_return,
                    'confidence': unbiased_analysis.confidence_score
                })
        
        # Sort by profit score
        profit_rankings.sort(key=lambda x: x['profit_score'], reverse=True)
        
        if profit_rankings:
            text += "###  Profit Potential Rankings (Unbiased)\n"
            for i, ranking in enumerate(profit_rankings[:5], 1):
                text += f"{i}. **{ranking['symbol']}** - "
                text += f"Profit Score: {ranking['profit_score']:.3f} "
                text += f"({ranking['profit_probability']:.1%} probability, "
                text += f"{ranking['expected_return']:+.1%} expected return)\n"
            
            text += "\n"
            
            # Best opportunity
            best_opportunity = profit_rankings[0]
            text += f"###  Top Opportunity: {best_opportunity['symbol']}\n"
            text += f"- **Mathematical Edge:** {best_opportunity['profit_score']:.3f}\n"
            text += f"- **Profit Probability:** {best_opportunity['profit_probability']:.1%}\n"
            text += f"- **Expected Return:** {best_opportunity['expected_return']:+.2%}\n"
            text += f"- **Model Confidence:** {best_opportunity['confidence']:.1%}\n"
            text += "\n"
        
        # Portfolio diversification insights
        if len(profit_rankings) > 1:
            text += "###  Portfolio Optimization Insights\n"
            
            total_expected_return = sum(r['expected_return'] * r['confidence'] for r in profit_rankings)
            avg_confidence = sum(r['confidence'] for r in profit_rankings) / len(profit_rankings)
            
            text += f"- **Portfolio Expected Return:** {total_expected_return:+.2%}\n"
            text += f"- **Average Model Confidence:** {avg_confidence:.1%}\n"
            
            # Risk diversification
            positive_returns = [r for r in profit_rankings if r['expected_return'] > 0]
            negative_returns = [r for r in profit_rankings if r['expected_return'] < 0]
            
            if positive_returns and negative_returns:
                text += "- **Diversification:** Mix of long and short opportunities identified\n"
            elif positive_returns:
                text += f"- **Bias:** {len(positive_returns)} bullish opportunities identified\n"
            elif negative_returns:
                text += f"- **Bias:** {len(negative_returns)} bearish opportunities identified\n"
            
            text += "\n"
        
        return text
    
    def _generate_unbiased_recommendations(self, symbol_analyses: Dict[str, Any],
                                         timeframe_scope: TimeframeScope,
                                         trader_context: Dict[str, Any]) -> str:
        """Generate unbiased trading recommendations focused on profit."""
        text = f"##  Unbiased {timeframe_scope.value.replace('_', ' ').title()} Recommendations\n\n"
        
        # Get trader profile recommendations
        recommendation_style = get_recommendation_style()
        
        for symbol, analysis in symbol_analyses.items():
            if not isinstance(analysis, dict):
                continue  # Skip invalid analysis entries
            unbiased_analysis = analysis.get('unbiased_analysis')
            if not unbiased_analysis:
                continue
            
            profit_probability = unbiased_analysis.profit_probability
            expected_return = unbiased_analysis.expected_return
            confidence = unbiased_analysis.confidence_score
            
            text += f"### {symbol} - Pure Mathematical Recommendation\n"
            
            # Generate recommendation based on pure math
            if profit_probability > 0.65 and expected_return > 0.03 and confidence > 0.6:
                text += "** STRONG BUY** - High probability profit opportunity\n"
                text += f"- **Action:** Initiate long position\n"
                text += f"- **Rationale:** {profit_probability:.1%} profit probability with {expected_return:+.1%} expected return\n"
                
            elif profit_probability < 0.35 and expected_return < -0.03 and confidence > 0.6:
                text += "** STRONG SELL** - High probability decline expected\n"
                text += f"- **Action:** Initiate short position or avoid\n"
                text += f"- **Rationale:** {(1-profit_probability):.1%} decline probability with {expected_return:+.1%} expected return\n"
                
            elif profit_probability > 0.55 and confidence > 0.5:
                text += "**[NOTE] MODERATE BUY** - Positive mathematical edge\n"
                text += f"- **Action:** Small long position\n"
                text += f"- **Rationale:** {profit_probability:.1%} profit probability\n"
                
            elif profit_probability < 0.45 and confidence > 0.5:
                text += "**[SELL] MODERATE SELL** - Negative mathematical edge\n"
                text += f"- **Action:** Reduce exposure or small short\n"
                text += f"- **Rationale:** {(1-profit_probability):.1%} decline probability\n"
                
            else:
                text += "** NEUTRAL** - No clear mathematical edge\n"
                text += f"- **Action:** Hold or wait for better setup\n"
                text += f"- **Rationale:** Insufficient edge (confidence: {confidence:.1%})\n"
            
            # Position sizing based on Kelly criterion
            if confidence > 0.5:
                kelly_fraction = self._calculate_kelly_fraction_simple(
                    profit_probability, expected_return, confidence
                )
                
                if kelly_fraction > 0.05:
                    text += f"- **Position Size:** {kelly_fraction:.1%} of portfolio (Kelly criterion)\n"
                else:
                    text += f"- **Position Size:** Minimal (<5% of portfolio)\n"
            
            # Risk management
            text += f"- **Stop Loss:** Mathematical risk management required\n"
            text += f"- **Time Horizon:** {timeframe_scope.value.replace('_', ' ').title()}\n"
            
            text += "\n"
        
        return text
    
    def _generate_octavian_unbiased_signature(self, symbol_analyses: Dict[str, Any],
                                            market_context: Dict[str, Any]) -> str:
        """Generate Octavian's unbiased signature insights."""
        text = "##  Octavian Unbiased Intelligence Summary\n\n"
        
        # Overall market assessment
        total_symbols = len(symbol_analyses)
        profitable_opportunities = sum(
            1 for analysis in symbol_analyses.values()
            if analysis.get('unbiased_analysis', {}).profit_probability > 0.6
        )
        
        text += f"###  Market Opportunity Assessment\n"
        text += f"- **Assets Analyzed:** {total_symbols} (selected without bias)\n"
        text += f"- **Profitable Opportunities:** {profitable_opportunities}/{total_symbols}\n"
        text += f"- **Opportunity Rate:** {profitable_opportunities/total_symbols:.1%}\n"
        
        # Calculate aggregate metrics
        all_analyses = [a.get('unbiased_analysis') for a in symbol_analyses.values() if a.get('unbiased_analysis')]
        
        if all_analyses:
            avg_profit_prob = np.mean([a.profit_probability for a in all_analyses])
            avg_expected_return = np.mean([a.expected_return for a in all_analyses])
            avg_confidence = np.mean([a.confidence_score for a in all_analyses])
            
            text += f"- **Average Profit Probability:** {avg_profit_prob:.1%}\n"
            text += f"- **Average Expected Return:** {avg_expected_return:+.2%}\n"
            text += f"- **Average Model Confidence:** {avg_confidence:.1%}\n"
        
        text += "\n"
        
        # Market regime insights
        vol_regime = market_context.get('volatility_regime', 'Unknown')
        risk_mode = market_context.get('risk_mode', 'Unknown')
        
        text += "###  Regime-Aware Insights\n"
        text += f"- **Current Regime:** {vol_regime} volatility, {risk_mode} sentiment\n"
        
        # Regime-specific recommendations
        if 'high' in vol_regime.lower():
            text += "- **Regime Strategy:** High volatility creates more profit opportunities but requires tighter risk management\n"
        elif 'low' in vol_regime.lower():
            text += "- **Regime Strategy:** Low volatility environment - focus on higher-conviction opportunities\n"
        
        if 'risk_on' in risk_mode.lower():
            text += "- **Sentiment Strategy:** Risk-on environment favors momentum and growth opportunities\n"
        elif 'risk_off' in risk_mode.lower():
            text += "- **Sentiment Strategy:** Risk-off environment - focus on defensive plays and short opportunities\n"
        
        text += "\n"
        
        # Octavian's final unbiased assessment
        text += "###  Octavian's Mathematical Conclusion\n"
        text += "Based on pure mathematical analysis without bias toward popular stocks:\n\n"
        
        if profitable_opportunities > total_symbols * 0.6:
            text += "**BULLISH MARKET ENVIRONMENT:** Multiple mathematical edges identified across diverse assets. "
            text += "Consider increasing overall exposure with appropriate risk management.\n"
        elif profitable_opportunities < total_symbols * 0.3:
            text += "**BEARISH MARKET ENVIRONMENT:** Limited profitable opportunities identified. "
            text += "Focus on capital preservation and selective short opportunities.\n"
        else:
            text += "**NEUTRAL MARKET ENVIRONMENT:** Mixed mathematical signals. "
            text += "Maintain selective approach and focus on highest-conviction opportunities.\n"
        
        text += "\n*Analysis based on pure mathematical models without bias toward company size, popularity, or market sentiment.*\n"
        
        return text
    
    def _calculate_enhanced_metrics(self, df: pd.DataFrame, timeframe_scope: TimeframeScope) -> Dict[str, Any]:
        """Calculate metrics with timeframe-specific focus."""
        if df.empty:
            return {}
        
        metrics = {}
        current_price = float(df['Close'].iloc[-1])
        
        try:
            # Timeframe-specific return calculations
            if timeframe_scope == TimeframeScope.SCALPING:
                # Focus on very short-term metrics
                if len(df) > 1:
                    metrics['1min_change'] = (df['Close'].iloc[-1] / df['Close'].iloc[-2] - 1) * 100
                if len(df) > 5:
                    metrics['5min_change'] = (df['Close'].iloc[-1] / df['Close'].iloc[-6] - 1) * 100
                    
            elif timeframe_scope == TimeframeScope.INTRADAY:
                # Focus on intraday metrics
                if len(df) > 1:
                    metrics['hourly_change'] = (df['Close'].iloc[-1] / df['Close'].iloc[-2] - 1) * 100
                if len(df) > 6:
                    metrics['session_change'] = (df['Close'].iloc[-1] / df['Close'].iloc[-7] - 1) * 100
                    
            elif timeframe_scope == TimeframeScope.SWING:
                # Focus on swing trading metrics
                if len(df) > 5:
                    metrics['weekly_change'] = (df['Close'].iloc[-1] / df['Close'].iloc[-6] - 1) * 100
                if len(df) > 10:
                    metrics['10d_change'] = (df['Close'].iloc[-1] / df['Close'].iloc[-11] - 1) * 100
                    
            elif timeframe_scope in [TimeframeScope.POSITION, TimeframeScope.INVESTMENT]:
                # Focus on longer-term metrics
                if len(df) > 20:
                    metrics['monthly_change'] = (df['Close'].iloc[-1] / df['Close'].iloc[-21] - 1) * 100
                if len(df) > 60:
                    metrics['quarterly_change'] = (df['Close'].iloc[-1] / df['Close'].iloc[-61] - 1) * 100
            
            # Universal metrics
            if len(df) > 1:
                metrics['daily_change'] = (df['Close'].iloc[-1] / df['Close'].iloc[-2] - 1) * 100
            
            # Volatility with timeframe adjustment
            returns = df['Close'].pct_change().dropna()
            if len(returns) > 10:
                lookback_periods = {
                    TimeframeScope.SCALPING: 10,
                    TimeframeScope.INTRADAY: 20,
                    TimeframeScope.SWING: 30,
                    TimeframeScope.POSITION: 60,
                    TimeframeScope.INVESTMENT: 120
                }
                
                lookback = lookback_periods.get(timeframe_scope, 20)
                vol_period = min(lookback, len(returns))
                
                metrics['volatility'] = returns.tail(vol_period).std() * np.sqrt(252) * 100
            
            metrics['current_price'] = current_price
            
        except Exception as e:
            print(f"Error calculating enhanced metrics: {e}")
        
        return metrics
    
    def _extract_timeframe_technical_signals(self, df: pd.DataFrame, 
                                           timeframe_scope: TimeframeScope) -> Dict[str, Any]:
        """Extract technical signals with timeframe-specific weighting."""
        if df.empty:
            return {}
        
        latest = df.iloc[-1]
        signals = {
            'timeframe_scope': timeframe_scope.value,
            'trend': 'NEUTRAL',
            'momentum': 'NEUTRAL',
            'primary_signals': [],
            'secondary_signals': []
        }
        
        try:
            # Timeframe-specific indicator focus
            if timeframe_scope == TimeframeScope.SCALPING:
                # Focus on momentum and volume
                if 'Volume' in df.columns and len(df) > 5:
                    recent_vol = df['Volume'].tail(5).mean()
                    avg_vol = df['Volume'].tail(20).mean() if len(df) > 20 else recent_vol
                    
                    if recent_vol > avg_vol * 1.5:
                        signals['primary_signals'].append('High volume momentum')
                
                # Price action signals
                if len(df) > 3:
                    recent_range = df['High'].tail(3).max() - df['Low'].tail(3).min()
                    if recent_range > 0:
                        current_position = (latest['Close'] - df['Low'].tail(3).min()) / recent_range
                        if current_position > 0.8:
                            signals['primary_signals'].append('Near recent highs')
                        elif current_position < 0.2:
                            signals['primary_signals'].append('Near recent lows')
            
            elif timeframe_scope == TimeframeScope.INTRADAY:
                # Focus on intraday patterns and RSI
                if 'rsi' in latest and pd.notna(latest['rsi']):
                    rsi = latest['rsi']
                    if rsi > 70:
                        signals['primary_signals'].append(f'RSI overbought ({rsi:.1f})')
                    elif rsi < 30:
                        signals['primary_signals'].append(f'RSI oversold ({rsi:.1f})')
                
                # EMA crossovers
                if 'ema20' in latest and 'ema50' in latest:
                    if pd.notna(latest['ema20']) and pd.notna(latest['ema50']):
                        if latest['ema20'] > latest['ema50']:
                            signals['trend'] = 'BULLISH'
                            signals['primary_signals'].append('EMA 20 > EMA 50 (bullish)')
                        else:
                            signals['trend'] = 'BEARISH'
                            signals['primary_signals'].append('EMA 20 < EMA 50 (bearish)')
            
            elif timeframe_scope == TimeframeScope.SWING:
                # Focus on swing patterns and trend strength
                if len(df) > 50:
                    # Calculate trend strength
                    price_50d_ago = df['Close'].iloc[-51]
                    trend_strength = (latest['Close'] / price_50d_ago - 1) * 100
                    
                    if trend_strength > 10:
                        signals['trend'] = 'STRONG_BULLISH'
                        signals['primary_signals'].append(f'Strong uptrend ({trend_strength:.1f}%)')
                    elif trend_strength < -10:
                        signals['trend'] = 'STRONG_BEARISH'
                        signals['primary_signals'].append(f'Strong downtrend ({trend_strength:.1f}%)')
                
                # Support/resistance levels
                if len(df) >= 50:
                    high_50 = df['High'].tail(50).max()
                    low_50 = df['Low'].tail(50).min()
                    current_position = (latest['Close'] - low_50) / (high_50 - low_50) if high_50 > low_50 else 0.5
                    
                    if current_position > 0.9:
                        signals['secondary_signals'].append('Near resistance level')
                    elif current_position < 0.1:
                        signals['secondary_signals'].append('Near support level')
            
            elif timeframe_scope in [TimeframeScope.POSITION, TimeframeScope.INVESTMENT]:
                # Focus on longer-term fundamentals and trends
                if len(df) > 100:
                    # Long-term trend analysis
                    price_100d_ago = df['Close'].iloc[-101]
                    long_term_trend = (latest['Close'] / price_100d_ago - 1) * 100
                    
                    if long_term_trend > 20:
                        signals['trend'] = 'STRONG_BULLISH'
                        signals['primary_signals'].append(f'Strong long-term uptrend ({long_term_trend:.1f}%)')
                    elif long_term_trend < -20:
                        signals['trend'] = 'STRONG_BEARISH'
                        signals['primary_signals'].append(f'Long-term downtrend ({long_term_trend:.1f}%)')
                
                # Volatility regime
                returns = df['Close'].pct_change().dropna()
                if len(returns) > 60:
                    recent_vol = returns.tail(60).std() * np.sqrt(252) * 100
                    if recent_vol > 30:
                        signals['secondary_signals'].append('High volatility regime')
                    elif recent_vol < 15:
                        signals['secondary_signals'].append('Low volatility regime')
        
        except Exception as e:
            print(f"Error extracting timeframe technical signals: {e}")
        
        return signals
    
    async def _generate_octavian_response(self, query: str, symbol_analyses: Dict[str, Any], 
                                        intents: List[str], timeframe_scope: TimeframeScope,
                                        market_context: Dict[str, Any], 
                                        trader_context: Dict[str, Any]) -> str:
        """Generate comprehensive Octavian response with enhanced insights."""
        
        # Get trader profile info
        recommendation_style = get_recommendation_style()
        should_specify_timeframe = recommendation_style.get('should_specify_timeframe', True) if hasattr(recommendation_style, 'get') else True
        
        # Start with Octavian branding
        response = f"""#  **OCTAVIAN** by APB - Advanced Market Intelligence
*Powered by AI • Enhanced with Source Credibility • Timeframe-Aware Analysis*

"""
        
        # Add timeframe context if needed
        if should_specify_timeframe or len(symbol_analyses) > 1:
            response += f"""##  Analysis Timeframe: **{timeframe_scope.value.replace('_', ' ').title()}**
*This analysis is specifically tailored for {timeframe_scope.value.replace('_', ' ')} trading/investing strategies*

"""
        
        # Add market context
        if market_context:
            vol_regime = market_context.get('volatility_regime', 'Unknown')
            risk_mode = market_context.get('risk_mode', 'Unknown')
            
            response += f"""##  Current Market Environment
- **Volatility Regime:** {vol_regime}
- **Risk Sentiment:** {risk_mode}
- **Analysis Scope:** {len(symbol_analyses)} asset{'s' if len(symbol_analyses) > 1 else ''}
- **Trader Profile:** {recommendation_style.get('timeframe_focus', 'General').replace('_', ' ').title() if hasattr(recommendation_style, 'get') else str(recommendation_style).title()}

---

"""
        
        # Analyze each symbol with enhanced insights
        for symbol, analysis in symbol_analyses.items():
            response += await self._generate_enhanced_symbol_analysis(
                symbol, analysis, timeframe_scope, trader_context
            )
            response += "\n---\n\n"
        
        # Add source credibility insights
        response += self._generate_credibility_insights_section(symbol_analyses)
        
        # Add timeframe-specific recommendations
        response += self._generate_timeframe_recommendations(
            symbol_analyses, timeframe_scope, trader_context
        )
        
        # Add Octavian signature insights
        response += self._generate_octavian_signature_insights(symbol_analyses, market_context)
        
        return response
    
    async def _generate_enhanced_symbol_analysis(self, symbol: str, analysis: Dict[str, Any],
                                               timeframe_scope: TimeframeScope,
                                               trader_context: Dict[str, Any]) -> str:
        """Generate enhanced symbol analysis with all new features."""
        
        prediction = analysis.get('prediction', {})
        metrics = analysis.get('metrics', {})
        technical = analysis.get('technical', {})
        timeframe_analysis = analysis.get('timeframe_analysis')
        credibility_weighted_news = analysis.get('credibility_weighted_news', {})
        narrative_comparison = analysis.get('narrative_comparison')
        
        current_price = metrics.get('current_price', 0)
        daily_change = metrics.get('daily_change', 0)
        
        signal = prediction.get('signal', 'NEUTRAL')
        confidence = prediction.get('confidence', 0.5) * 100
        bullish_prob = prediction.get('bullish_prob', 0.5) * 100
        
        # Header with enhanced status
        text = f"""##  {symbol} ({analysis.get('asset_type', 'Asset')}) - Octavian Enhanced Analysis

###  Current Market Status
- **Price:** ${current_price:.2f} ({daily_change:+.2f}%)
- **AI Signal:** **{signal}** ({confidence:.1f}% confidence)
- **Bullish Probability:** {bullish_prob:.1f}%
- **Timeframe Focus:** {timeframe_scope.value.replace('_', ' ').title()}

"""
        
        # Timeframe-specific analysis
        if timeframe_analysis and not timeframe_analysis.current_context.get('error'):
            text += "###  Timeframe-Specific Analysis\n"
            
            # Current context insights
            current_context = timeframe_analysis.current_context
            primary_factors = current_context.get('primary_factors_status', {})
            
            if primary_factors:
                text += "**Primary Factors for this Timeframe:**\n"
                for factor, status in list(primary_factors.items())[:3]:
                    factor_status = status.get('status', 'neutral')
                    factor_strength = status.get('strength', 0.5)
                    text += f"- **{factor.replace('_', ' ').title()}:** {factor_status.title()} (Strength: {factor_strength:.1%})\n"
            
            # Forward projections
            forward_projections = timeframe_analysis.forward_projections
            if forward_projections:
                expected_return = forward_projections.get('probability_weighted_outcome', {}).get('expected_return', 0)
                text += f"\n**Timeframe Projection:** {expected_return:+.1%} expected return\n"
            
            # Actionable insights
            actionable_insights = timeframe_analysis.actionable_insights
            if actionable_insights:
                text += f"\n**Key Insights for {timeframe_scope.value.replace('_', ' ').title()} Traders:**\n"
                for insight in actionable_insights[:3]:
                    text += f"- {insight}\n"
            
            text += "\n"
        
        # Source credibility weighted news analysis
        if credibility_weighted_news and not credibility_weighted_news.get('error'):
            text += "###  Source-Weighted News Analysis\n"
            
            weighted_sentiment = credibility_weighted_news.get('weighted_sentiment', 0)
            total_weight = credibility_weighted_news.get('total_weight', 0)
            item_count = credibility_weighted_news.get('item_count', 0)
            
            sentiment_label = self._sentiment_to_label(weighted_sentiment)
            
            text += f"- **Weighted News Sentiment:** {sentiment_label} ({weighted_sentiment:+.2f})\n"
            text += f"- **Articles Analyzed:** {item_count} (Total Credibility Weight: {total_weight:.1f})\n"
            
            # Source breakdown
            source_breakdown = credibility_weighted_news.get('source_breakdown', {})
            if source_breakdown:
                tier1_count = source_breakdown.get('tier_1_premium', 0)
                tier2_count = source_breakdown.get('tier_2_major', 0)
                
                if tier1_count > 0:
                    text += f"- **Premium Sources:** {tier1_count} articles from Bloomberg, Reuters, WSJ\n"
                if tier2_count > 0:
                    text += f"- **Major Sources:** {tier2_count} articles from CNBC, MarketWatch, etc.\n"
            
            text += "\n"
        
        # Technical analysis with timeframe context
        if technical:
            text += "###  Technical Analysis\n"
            
            trend = technical.get('trend', 'NEUTRAL')
            primary_signals = technical.get('primary_signals', [])
            secondary_signals = technical.get('secondary_signals', [])
            
            text += f"- **Trend Direction:** {trend}\n"
            
            if primary_signals:
                text += "- **Primary Signals:**\n"
                for signal in primary_signals[:3]:
                    text += f"  • {signal}\n"
            
            if secondary_signals:
                text += "- **Secondary Signals:**\n"
                for signal in secondary_signals[:2]:
                    text += f"  • {signal}\n"
            
            text += "\n"
        
        # AI vs Market Narrative (if available)
        if narrative_comparison and hasattr(narrative_comparison, 'narrative_text'):
            text += "###  AI vs Market Sentiment Analysis\n"
            
            confidence_divergence = getattr(narrative_comparison, 'confidence_divergence', 0)
            decision_implications = getattr(narrative_comparison, 'decision_implications', [])
            
            text += f"- **AI-Market Divergence:** {confidence_divergence:.1%}\n"
            
            if confidence_divergence > 0.3:
                text += "- **High Divergence:** Significant disagreement between AI models and market sentiment\n"
            elif confidence_divergence > 0.15:
                text += "- **Moderate Divergence:** Some disagreement worth monitoring\n"
            else:
                text += "- **Low Divergence:** AI and market sentiment generally aligned\n"
            
            if decision_implications:
                text += "- **Key Decision Implications:**\n"
                for implication in decision_implications[:2]:
                    text += f"  • {implication}\n"
            
            text += "\n"
        
        # --- Profit Catalysts & Expected Return ---
        profit_catalysts = analysis.get('profit_catalysts', [])
        expected_ret = analysis.get('expected_return', 0)
        risk_adj_ret = analysis.get('risk_adjusted_return', 0)
        
        if profit_catalysts or expected_ret != 0:
            text += "###  Profit Catalysts & Return Outlook\n"
            if expected_ret != 0:
                text += f"- **Expected Annual Return:** {expected_ret:+.2%}\n"
                text += f"- **Risk-Adjusted Return:** {risk_adj_ret:+.2f}\n"
            if profit_catalysts:
                text += "- **Key Catalysts:**\n"
                for catalyst in profit_catalysts[:5]:
                    text += f"  • {catalyst}\n"
            text += "\n"
        
        # --- Risk Factors ---
        risk_factors = analysis.get('risk_factors', [])
        if risk_factors:
            text += "###  Risk Factors\n"
            for risk in risk_factors[:5]:
                text += f"- {risk}\n"
            text += "\n"
        
        # --- Model Reasoning (How the AI reached its conclusion) ---
        model_reasoning = analysis.get('model_reasoning', [])
        if model_reasoning:
            text += "###  Model Reasoning\n"
            for reason in model_reasoning[:6]:
                text += f"- {reason}\n"
            text += "\n"
        
        return text
    
    def _sentiment_to_label(self, sentiment: float) -> str:
        """Convert sentiment score to readable label."""
        if sentiment > 0.6:
            return "VERY BULLISH"
        elif sentiment > 0.2:
            return "BULLISH"
        elif sentiment > -0.2:
            return "NEUTRAL"
        elif sentiment > -0.6:
            return "BEARISH"
        else:
            return "VERY BEARISH"
    
    def _generate_credibility_insights_section(self, symbol_analyses: Dict[str, Any]) -> str:
        """Generate section highlighting source credibility insights."""
        text = "##  Source Credibility Insights\n\n"
        
        # Aggregate credibility data
        total_articles = 0
        premium_articles = 0
        weighted_sentiments = []
        
        for symbol, analysis in symbol_analyses.items():
            credibility_news = analysis.get('credibility_weighted_news', {})
            if credibility_news and not credibility_news.get('error'):
                item_count = credibility_news.get('item_count', 0)
                total_articles += item_count
                
                source_breakdown = credibility_news.get('source_breakdown', {})
                premium_articles += source_breakdown.get('tier_1_premium', 0)
                
                weighted_sentiment = credibility_news.get('weighted_sentiment', 0)
                if weighted_sentiment != 0:
                    weighted_sentiments.append(weighted_sentiment)
        
        if total_articles > 0:
            premium_ratio = premium_articles / total_articles
            avg_weighted_sentiment = np.mean(weighted_sentiments) if weighted_sentiments else 0
            
            text += f"- **Total Articles Analyzed:** {total_articles}\n"
            text += f"- **Premium Source Coverage:** {premium_ratio:.1%} ({premium_articles} premium articles)\n"
            text += f"- **Credibility-Weighted Sentiment:** {self._sentiment_to_label(avg_weighted_sentiment)}\n"
            
            if premium_ratio > 0.3:
                text += "- **High Credibility:** Strong coverage from premium financial sources\n"
            elif premium_ratio > 0.1:
                text += "- **Moderate Credibility:** Decent coverage from established sources\n"
            else:
                text += "- **Lower Credibility:** Limited premium source coverage - exercise caution\n"
        else:
            text += "- **Limited News Coverage:** Insufficient news data for credibility analysis\n"
        
        text += "\n"
        return text
    
    def _generate_timeframe_recommendations(self, symbol_analyses: Dict[str, Any],
                                          timeframe_scope: TimeframeScope,
                                          trader_context: Dict[str, Any]) -> str:
        """Generate timeframe-specific trading recommendations."""
        text = f"##  {timeframe_scope.value.replace('_', ' ').title()} Trading Recommendations\n\n"
        
        # Get trader profile recommendations
        recommendation_style = get_recommendation_style()
        
        for symbol, analysis in symbol_analyses.items():
            prediction = analysis.get('prediction', {})
            timeframe_analysis = analysis.get('timeframe_analysis')
            
            signal = prediction.get('signal', 'NEUTRAL')
            confidence = prediction.get('confidence', 0.5)
            
            text += f"### {symbol} - {timeframe_scope.value.replace('_', ' ').title()} Strategy\n"
            
            # Timeframe-specific recommendations
            if timeframe_scope == TimeframeScope.SCALPING:
                text += "**Scalping Strategy:**\n"
                if signal == 'BULLISH' and confidence > 0.7:
                    text += "- Look for quick long entries on pullbacks\n"
                    text += "- Target 0.1-0.3% moves with tight stops\n"
                    text += "- Monitor level 2 data for entry timing\n"
                elif signal == 'BEARISH' and confidence > 0.7:
                    text += "- Consider quick short entries on bounces\n"
                    text += "- Target small moves with very tight risk management\n"
                else:
                    text += "- Avoid trading - wait for clearer signals\n"
                    text += "- Focus on high-volume breakouts only\n"
            
            elif timeframe_scope == TimeframeScope.INTRADAY:
                text += "**Intraday Strategy:**\n"
                if signal == 'BULLISH' and confidence > 0.6:
                    text += "- Consider long positions on morning dips\n"
                    text += "- Use support levels for entries\n"
                    text += "- Target resistance levels for exits\n"
                elif signal == 'BEARISH' and confidence > 0.6:
                    text += "- Look for short opportunities on rallies\n"
                    text += "- Use resistance levels for entries\n"
                else:
                    text += "- Range-bound trading approach\n"
                    text += "- Buy support, sell resistance\n"
            
            elif timeframe_scope == TimeframeScope.SWING:
                text += "**Swing Trading Strategy:**\n"
                if signal == 'BULLISH' and confidence > 0.6:
                    text += "- Build long positions on weekly dips\n"
                    text += "- Hold for 1-3 week moves\n"
                    text += "- Use 20-day EMA as dynamic support\n"
                elif signal == 'BEARISH' and confidence > 0.6:
                    text += "- Consider short positions on rallies\n"
                    text += "- Target multi-week downward moves\n"
                else:
                    text += "- Wait for clearer trend development\n"
                    text += "- Focus on risk management\n"
            
            elif timeframe_scope in [TimeframeScope.POSITION, TimeframeScope.INVESTMENT]:
                text += "**Position/Investment Strategy:**\n"
                if signal == 'BULLISH' and confidence > 0.6:
                    text += "- Accumulate on significant dips\n"
                    text += "- Focus on fundamental strength\n"
                    text += "- Hold for multi-month moves\n"
                elif signal == 'BEARISH' and confidence > 0.6:
                    text += "- Reduce exposure or hedge positions\n"
                    text += "- Wait for better entry opportunities\n"
                else:
                    text += "- Maintain current allocation\n"
                    text += "- Focus on diversification\n"
            
            # Add risk management based on trader profile
            risk_level = recommendation_style.get('risk_level', 'medium') if hasattr(recommendation_style, 'get') else 'medium'
            if risk_level == 'high':
                text += "- **Risk Management:** Aggressive position sizing acceptable\n"
            elif risk_level == 'low':
                text += "- **Risk Management:** Conservative position sizing recommended\n"
            else:
                text += "- **Risk Management:** Standard position sizing with appropriate stops\n"
            
            text += "\n"
        
        return text
    
    def _generate_octavian_signature_insights(self, symbol_analyses: Dict[str, Any],
                                            market_context: Dict[str, Any]) -> str:
        """Generate Octavian's signature advanced insights."""
        text = "##  Octavian Advanced Intelligence\n\n"
        
        # Cross-asset correlation insights
        if len(symbol_analyses) > 1:
            text += "###  Cross-Asset Intelligence\n"
            
            signals = []
            confidences = []
            
            for symbol, analysis in symbol_analyses.items():
                prediction = analysis.get('prediction', {})
                signals.append(prediction.get('signal', 'NEUTRAL'))
                confidences.append(prediction.get('confidence', 0.5))
            
            # Analyze signal alignment
            bullish_count = signals.count('BULLISH')
            bearish_count = signals.count('BEARISH')
            avg_confidence = np.mean(confidences)
            
            if bullish_count > bearish_count:
                text += f"- **Portfolio Bias:** Bullish alignment across {bullish_count}/{len(signals)} assets\n"
            elif bearish_count > bullish_count:
                text += f"- **Portfolio Bias:** Bearish alignment across {bearish_count}/{len(signals)} assets\n"
            else:
                text += "- **Portfolio Bias:** Mixed signals - diversified approach recommended\n"
            
            text += f"- **Average Model Confidence:** {avg_confidence:.1%}\n"
            
            if avg_confidence > 0.7:
                text += "- **Signal Strength:** High confidence across assets - strong directional bias\n"
            elif avg_confidence < 0.5:
                text += "- **Signal Strength:** Low confidence - uncertain market conditions\n"
            
            text += "\n"
        
        # Market regime insights
        vol_regime = market_context.get('volatility_regime', 'Unknown')
        risk_mode = market_context.get('risk_mode', 'Unknown')
        
        text += "###  Market Regime Analysis\n"
        text += f"- **Current Regime:** {vol_regime} volatility, {risk_mode} sentiment\n"
        
        # Regime-specific insights
        if 'high' in vol_regime.lower():
            text += "- **Regime Implication:** High volatility favors shorter timeframes and tighter risk management\n"
        elif 'low' in vol_regime.lower():
            text += "- **Regime Implication:** Low volatility environment - consider longer-term positions\n"
        
        if 'risk_on' in risk_mode.lower():
            text += "- **Sentiment Implication:** Risk-on environment favors growth assets and momentum strategies\n"
        elif 'risk_off' in risk_mode.lower():
            text += "- **Sentiment Implication:** Risk-off environment - focus on defensive assets and hedging\n"
        
        text += "\n"
        
        # Octavian's final assessment
        text += "###  Octavian's Final Assessment\n"
        text += "Based on our comprehensive analysis integrating AI predictions, source-weighted news, "
        text += "timeframe-specific factors, and market regime analysis:\n\n"
        
        # Generate overall market view
        total_symbols = len(symbol_analyses)
        bullish_signals = sum(1 for analysis in symbol_analyses.values() 
                             if analysis.get('prediction', {}).get('signal') == 'BULLISH')
        
        if bullish_signals > total_symbols * 0.6:
            text += "**BULLISH OUTLOOK:** Multiple assets showing positive signals with strong AI confidence. "
            text += "Consider increasing exposure with appropriate risk management.\n"
        elif bullish_signals < total_symbols * 0.4:
            text += "**BEARISH OUTLOOK:** Defensive positioning recommended. "
            text += "Focus on capital preservation and hedging strategies.\n"
        else:
            text += "**NEUTRAL OUTLOOK:** Mixed signals suggest a balanced approach. "
            text += "Maintain diversification and wait for clearer directional signals.\n"
        
        text += "\n*Remember: This analysis is for informational purposes only. "
        text += "Always conduct your own research and consider your risk tolerance.*\n"
        
        return text
    
    def _detect_intent(self, query: str) -> List[str]:
        """Detect user intent from query."""
        query_lower = query.lower()
        detected_intents = []
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    detected_intents.append(intent)
                    break
        
        # Default to analysis if no clear intent
        if not detected_intents:
            detected_intents = ['analysis']
        
        return list(set(detected_intents))
    
    def _extract_symbols_legacy(self, query: str) -> List[str]:
        """Legacy symbol extraction (kept for backward compatibility; do not use)."""
        query_upper = query.upper()
        symbols = []
        
        # Enhanced symbol patterns
        # Stock tickers (1-5 uppercase letters)
        potential_symbols = re.findall(r'\b[A-Z]{1,5}\b', query_upper)
        
        common_words = {
            'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HAD',
            'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'HAS', 'WHAT', 'WHEN', 'WILL', 'HOW',
            'THIS', 'THAT', 'WITH', 'SHOW', 'GIVE', 'TELL', 'MAKE', 'JUST', 'BUY',
            'SELL', 'GET', 'RSI', 'EMA', 'SMA', 'MACD', 'ATR', 'ADX', 'FROM', 'THEY',
            'HAVE', 'BEEN', 'THEIR', 'SAID', 'EACH', 'WHICH', 'WOULD', 'THERE',
            'COULD', 'OTHER', 'AFTER', 'FIRST', 'WELL', 'ALSO', 'NEW', 'WANT',
            'BECAUSE', 'ANY', 'THESE', 'DAY', 'MOST', 'US', 'IS', 'AM',
            'CHART', 'GRAPH', 'PLOT', 'ANALYSIS', 'PREDICT', 'FORECAST',
            'LOOK', 'RIGHT', 'NOW', 'SECTOR', 'SECTORS', 'BEAR', 'BEARISH', 'BULL', 'BULLISH',
            'EXTREME', 'EXTREMELY', 'CANDIDATE', 'CANDIDATES', 'SHORT', 'LONG',
        }
        
        # Filter symbols
        for symbol in potential_symbols:
            if symbol not in common_words and len(symbol) >= 2:
                symbols.append(symbol)
        
        currency_codes = {
            "EUR", "USD", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD", "SEK", "NOK", "DKK",
            "PLN", "CZK", "HUF", "TRY", "ZAR", "MXN", "BRL", "CNY", "INR", "KRW", "SGD",
            "HKD", "THB", "MYR", "IDR", "PHP",
        }
        fx_matches = re.findall(r'\b([A-Z]{3})[\/_-]([A-Z]{3})\b', query_upper)
        fx_matches += re.findall(r'\b([A-Z]{3})([A-Z]{3})\b', query_upper)
        for c1, c2 in fx_matches:
            if c1 in currency_codes and c2 in currency_codes and c1 != c2:
                symbols.append(f"{c1}/{c2}")
        
        # Enhanced crypto detection
        crypto_patterns = [
            r'(BTC|BITCOIN)[-]?USD',
            r'(ETH|ETHEREUM)[-]?USD', 
            r'(SOL|SOLANA)[-]?USD',
            r'(DOGE|DOGECOIN)[-]?USD',
            r'(ADA|CARDANO)[-]?USD',
            r'(MATIC|POLYGON)[-]?USD',
            r'(AVAX|AVALANCHE)[-]?USD'
        ]
        
        for pattern in crypto_patterns:
            matches = re.findall(pattern, query_upper)
            for match in matches:
                if isinstance(match, str):
                    symbols.append(f"{match}-USD")
                elif isinstance(match, tuple):
                    symbols.append(f"{match[0]}-USD")
        
        # Futures detection
        futures_symbols = {
            'ES': 'ES=F', 'NQ': 'NQ=F', 'YM': 'YM=F', 'RTY': 'RTY=F',
            'CL': 'CL=F', 'GC': 'GC=F', 'SI': 'SI=F', 'ZN': 'ZN=F',
            'ZB': 'ZB=F', 'ZF': 'ZF=F', 'ZC': 'ZC=F', 'ZS': 'ZS=F'
        }
        
        for symbol in symbols.copy():
            if symbol in futures_symbols:
                symbols.append(futures_symbols[symbol])
        
        # Remove duplicates and limit
        symbols = list(dict.fromkeys(symbols))  # Preserve order while removing duplicates
        
        return symbols[:6]
    
    def _create_comparison_chart(self, symbols: List[str], period: str = '6mo') -> go.Figure:
        """Create a comparison chart for multiple symbols."""
        # Get chart timeframe from settings
        chart_settings = st.session_state.get('chart_settings', {})
        chart_period = chart_settings.get('chart_timeframe', '6 Months')
        
        # Convert display period to yfinance period
        period_map = {
            '1 Month': '1mo',
            '3 Months': '3mo', 
            '6 Months': '6mo',
            '1 Year': '1y',
            '2 Years': '2y'
        }
        period = period_map.get(chart_period, '6mo')
        fig = go.Figure()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, symbol in enumerate(symbols[:5]):
            df, _ = self._get_real_time_data(symbol, period)
            if df is not None and not df.empty:
                # Normalize to percentage change from first value
                first_val = float(df['Close'].iloc[0])
                normalized = ((df['Close'] - first_val) / first_val * 100)
                
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=normalized,
                    mode='lines',
                    name=symbol,
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
        
        fig.update_layout(
            height=500,
            template='plotly_dark',
            title='Performance Comparison (Normalized %)',
            xaxis_title='Date',
            yaxis_title='Change (%)',
            hovermode='x unified'
        )
        
        return fig
    
    def _create_volatility_chart(self, df: pd.DataFrame, symbol: str) -> go.Figure:
        """Create volatility analysis chart."""
        if df.empty or len(df) < 30:
            return None
        
        returns = df['Close'].pct_change().dropna()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Rolling Volatility (20d)',
                'Returns Distribution',
                'Volatility Cone',
                'Risk Metrics'
            ),
            specs=[[{"type": "scatter"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "table"}]]
        )
        
        # Rolling volatility
        vol_20 = returns.rolling(20).std() * np.sqrt(252) * 100
        fig.add_trace(
            go.Scatter(
                x=vol_20.index,
                y=vol_20,
                mode='lines',
                name='20d Volatility',
                line=dict(color='#ff7043', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 112, 67, 0.2)'
            ),
            row=1, col=1
        )
        
        # Returns distribution
        fig.add_trace(
            go.Histogram(
                x=returns * 100,
                nbinsx=50,
                name='Daily Returns',
                marker_color='#42a5f5',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        # Volatility cone (historical percentiles)
        vol_windows = [5, 10, 20, 40, 60]
        vol_percentiles = []
        for window in vol_windows:
            if len(returns) >= window:
                vol = returns.rolling(window).std() * np.sqrt(252) * 100
                vol_percentiles.append({
                    'window': window,
                    'current': vol.iloc[-1] if pd.notna(vol.iloc[-1]) else 0,
                    'avg': vol.mean() if pd.notna(vol.mean()) else 0
                })
        
        if vol_percentiles:
            windows = [v['window'] for v in vol_percentiles]
            current_vols = [v['current'] for v in vol_percentiles]
            avg_vols = [v['avg'] for v in vol_percentiles]
            
            fig.add_trace(
                go.Scatter(
                    x=windows,
                    y=current_vols,
                    mode='lines+markers',
                    name='Current Vol',
                    line=dict(color='#ef5350', width=2)
                ),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=windows,
                    y=avg_vols,
                    mode='lines+markers',
                    name='Average Vol',
                    line=dict(color='#26a69a', width=2)
                ),
                row=2, col=1
            )
        
        # Risk metrics table
        current_vol = vol_20.iloc[-1] if len(vol_20) > 0 and pd.notna(vol_20.iloc[-1]) else 0
        max_drawdown = ((df['Close'].cummax() - df['Close']) / df['Close'].cummax()).max() * 100
        var_95 = returns.quantile(0.05) * 100
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value']),
                cells=dict(values=[
                    ['Current Volatility', 'Max Drawdown', 'VaR (95%)', 'Sharpe Ratio'],
                    [f'{current_vol:.2f}%', 
                     f'{max_drawdown:.2f}%', 
                     f'{var_95:.2f}%',
                     f'{(returns.mean() / returns.std() * np.sqrt(252)):.2f}' if returns.std() > 0 else 'N/A']
                ])
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=700,
            template='plotly_dark',
            showlegend=True,
            title=f'{symbol} Volatility Analysis'
        )
        
        return fig
    
    def _generate_analysis_text(self, df: pd.DataFrame, symbol: str, 
                                 prediction: Dict, asset_type: str) -> str:
        """Generate detailed analysis text explanation."""
        if df.empty:
            return f"Unable to fetch data for {symbol}."
        
        latest = df.iloc[-1]
        current_price = float(latest['Close'])
        
        # Calculate key metrics
        returns_1d = df['Close'].pct_change().iloc[-1] * 100 if len(df) > 1 else 0
        returns_5d = (df['Close'].iloc[-1] / df['Close'].iloc[-6] - 1) * 100 if len(df) > 5 else 0
        returns_20d = (df['Close'].iloc[-1] / df['Close'].iloc[-21] - 1) * 100 if len(df) > 20 else 0
        
        # Volatility
        vol_20d = df['Close'].pct_change().tail(20).std() * np.sqrt(252) * 100 if len(df) > 20 else 0
        
        # Build analysis text
        signal = prediction.get('signal', 'NEUTRAL')
        confidence = prediction.get('confidence', 0.5) * 100
        bullish_prob = prediction.get('bullish_prob', 0.5) * 100
        
        analysis = f"""
##  {symbol} Analysis Summary

### Current Market Status
- **Current Price:** ${current_price:.2f}
- **1-Day Change:** {returns_1d:+.2f}%
- **5-Day Change:** {returns_5d:+.2f}%
- **20-Day Change:** {returns_20d:+.2f}%
- **20-Day Volatility:** {vol_20d:.2f}% (annualized)

### ML Model Prediction
- **Signal:** {signal}
- **Confidence:** {confidence:.1f}%
- **Bullish Probability:** {bullish_prob:.1f}%

### Technical Analysis
"""
        
        # Add technical indicator analysis
        df_with_ind = add_indicators(df.copy())
        if not df_with_ind.empty:
            latest_ind = df_with_ind.iloc[-1]
            
            if 'rsi' in latest_ind and pd.notna(latest_ind['rsi']):
                rsi = latest_ind['rsi']
                rsi_status = 'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral'
                analysis += f"- **RSI (14):** {rsi:.1f} - {rsi_status}\n"
            
            if 'ema20' in latest_ind and 'ema50' in latest_ind:
                if pd.notna(latest_ind['ema20']) and pd.notna(latest_ind['ema50']):
                    ema_trend = 'Bullish' if latest_ind['ema20'] > latest_ind['ema50'] else 'Bearish'
                    analysis += f"- **EMA Trend:** {ema_trend} (20-day: ${latest_ind['ema20']:.2f}, 50-day: ${latest_ind['ema50']:.2f})\n"
        
        # Add signal factors if available
        signal_factors = prediction.get('signal_factors', [])
        if signal_factors:
            analysis += "\n### Key Signal Factors\n"
            for factor in signal_factors[:5]:
                analysis += f"- {factor[0]}: {factor[1]} ({factor[2]})\n"
        
        # Add recommendation
        analysis += f"""
### Recommendation
Based on the multi-layer ensemble analysis, {symbol} shows a **{signal}** signal with {confidence:.1f}% confidence.

"""
        
        if signal == 'BULLISH':
            analysis += "The technical indicators and ML model suggest favorable conditions for long positions. "
            analysis += f"Consider entry points near support levels with appropriate stop-loss management."
        elif signal == 'BEARISH':
            analysis += "The analysis indicates caution. Consider reducing exposure or hedging existing positions. "
            analysis += "Short-term traders may look for shorting opportunities at resistance levels."
        else:
            analysis += "The market shows mixed signals. Consider maintaining current positions with tight risk management. "
            analysis += "Wait for clearer directional signals before initiating new positions."
        
        return analysis
    
    def process_query(self, query: str, user_id: str = None) -> Dict[str, Any]:
        """Process user query with advanced AI analysis, real-time data, and comprehensive response generation."""
        start_time = time.time()

        # Portfolio-aware routing: questions about the user's own holdings are
        # answered from their actual paper/manual portfolio data.
        try:
            from portfolio_chatbot_context import (
                detect_portfolio_intent, handle_portfolio_query,
                load_all_portfolios, _get_all_holdings,
            )
            pf_intent = detect_portfolio_intent(query)
            if pf_intent:
                ctx = load_all_portfolios(user_id)
                holdings = _get_all_holdings(ctx) if ctx.get('has_data') else []
                res = handle_portfolio_query(query, pf_intent, user_id)
                if asyncio.iscoroutine(res):
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # Cannot block inside a running loop; run in a fresh loop.
                            res = asyncio.run(res) if asyncio.iscoroutine(res) else res
                        else:
                            res = loop.run_until_complete(res)
                    except Exception:
                        # Final fallback: run in a brand-new event loop.
                        try:
                            res = asyncio.run(res) if asyncio.iscoroutine(res) else res
                        except Exception:
                            pass
                return {
                    'text': res if isinstance(res, str) else str(res),
                    'charts': [],
                    'intents': [pf_intent],
                    'symbols': [h.get('symbol') for h in holdings if h.get('symbol')],
                    'suggestions': [],
                    'response_time_ms': int((time.time() - start_time) * 1000),
                }
        except Exception:
            pass

        # Extract symbols first (needed for intent detection)
        symbols = self._extract_symbols(query)

        # Explicit tool asks (DCF, Bayesian network, Markov model, dark pool,
        # 13F, correlation, greeks, crowding, regime) are outsourced to
        # Octavian's own engines and their computed output is repackaged as
        # the answer — the LLM never improvises a DCF or a regime call.
        try:
            from tool_router import route_tool_query
            _tool_text = route_tool_query(query, symbols, [], {})
        except Exception:
            _tool_text = None
        if _tool_text:
            return {
                'text': _tool_text,
                'charts': [],
                'intents': ['tool'],
                'symbols': symbols,
                'suggestions': [],
                'response_time_ms': int((time.time() - start_time) * 1000),
            }

        # USE ADVANCED DYNAMIC INTENT DETECTION
        intent_analysis = self.intent_engine.analyze_intent(query, symbols)
        
        # Convert to legacy format for compatibility (will be fully migrated later)
        intents = [intent_analysis.primary_intent.value] + \
                  [intent.value for intent in intent_analysis.secondary_intents]
        
        # Store full intent analysis for advanced response generation
        full_intent_analysis = intent_analysis
        
        # Get conversation context
        conversation_history = self.db_manager.get_conversation_history(self.session_id, limit=10)
        
        # If no symbols found, try to infer from context or provide guidance
        if not symbols:
            # Check conversation history for recent symbols
            for conv in conversation_history[-3:]:  # Check last 3 conversations
                if conv.get('symbols'):
                    symbols.extend(conv['symbols'][:2])  # Add up to 2 recent symbols
                    break
            
            # If still no symbols, check for market-wide queries — use full universe
            market_keywords = ['market', 'spy', 's&p', 'nasdaq', 'dow', 'index', 'stocks', 'overall']
            if any(keyword in query.lower() for keyword in market_keywords):
                symbols = ['SPY', 'QQQ', '^VIX']
            elif any(keyword in query.lower() for keyword in ['crypto', 'bitcoin', 'btc']):
                # Pull crypto tickers dynamically from universe
                if _HAS_TICKER_UNIVERSE:
                    try:
                        crypto_list = get_ticker_universe().get_crypto()
                        symbols = crypto_list[:5] if crypto_list else ['BTC-USD', 'ETH-USD']
                    except Exception:
                        symbols = ['BTC-USD', 'ETH-USD']
                else:
                    symbols = ['BTC-USD', 'ETH-USD']
            elif any(keyword in query.lower() for keyword in ['forex', 'fx', 'currency', 'dollar']):
                # Pull forex tickers dynamically from universe
                if _HAS_TICKER_UNIVERSE:
                    try:
                        fx_list = get_ticker_universe().get_forex()
                        symbols = fx_list[:5] if fx_list else ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X']
                    except Exception:
                        symbols = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X']
                else:
                    symbols = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X']
            else:
                return {
                    'text': self._generate_helpful_response(query, intents),
                    'charts': [],
                    'intents': intents,
                    'intent_analysis': intent_analysis,  # Include full analysis
                    'symbols': [],
                    'suggestions': self._get_query_suggestions(),
                    'response_time_ms': int((time.time() - start_time) * 1000)
                }
        
        # Remove duplicates while preserving order
        symbols = list(dict.fromkeys(symbols))
        
        response = {
            'text': '',
            'charts': [],
            'intents': intents,
            'intent_analysis': full_intent_analysis,  # Full intent analysis
            'symbols': symbols,
            'analysis_summary': {},
            'market_context': {},
            'suggestions': [],
            'response_time_ms': 0
        }
        
        # Get market regime context
        try:
            vol_regime = volatility_regime()
            risk_mode = risk_on_off()
            response['market_context'] = {
                'volatility_regime': vol_regime,
                'risk_mode': risk_mode,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error getting market context: {e}")
        
        # Process each symbol with parallel execution for better performance
        symbol_analyses = {}
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_symbol = {}
            
            for symbol in symbols[:3]:  # Limit to 3 symbols for performance
                future = executor.submit(self._analyze_symbol_comprehensive, symbol, intents)
                future_to_symbol[future] = symbol
            
            for future in future_to_symbol:
                symbol = future_to_symbol[future]
                try:
                    analysis = future.result(timeout=30)  # 30 second timeout
                    if analysis:
                        symbol_analyses[symbol] = analysis
                except Exception as e:
                    print(f"Error analyzing {symbol}: {e}")
                    response['text'] += f"\n Unable to analyze {symbol}: {str(e)}\n"
        
        # Generate comprehensive response USING INTENT ANALYSIS
        if symbol_analyses:
            response['text'] = self._generate_intent_aware_response(
                query, symbol_analyses, full_intent_analysis, response['market_context']
            )
            response['analysis_summary'] = self._create_analysis_summary(symbol_analyses)
            
            # Generate charts based on intent analysis
            response['charts'] = self._generate_intent_aware_charts(
                symbol_analyses, full_intent_analysis
            )
            
            # Add suggestions for follow-up questions
            response['suggestions'] = self._generate_contextual_suggestions(symbol_analyses, intents)
        
        # Handle comparison intent
        if intent_analysis.is_comparison and len(symbol_analyses) > 1:
            comparison_chart = self._create_comparison_chart(list(symbol_analyses.keys()))
            if comparison_chart:
                response['charts'].append({
                    'type': 'comparison',
                    'symbols': list(symbol_analyses.keys()),
                    'figure': comparison_chart,
                    'title': 'Performance Comparison'
                })
        
        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)
        response['response_time_ms'] = response_time_ms
        
        # Store conversation in database
        try:
            conversation_id = self.db_manager.store_conversation(
                self.session_id, query, response, user_id, response_time_ms
            )
            
            # Store ML predictions for each symbol
            for symbol, analysis in symbol_analyses.items():
                if 'prediction' in analysis:
                    self.db_manager.store_ml_prediction(symbol, analysis['prediction'])
            
            # Log performance metrics
            self.db_manager.log_system_metric('response_time_ms', response_time_ms)
            self.db_manager.log_system_metric('symbols_analyzed', len(symbol_analyses))
            self.db_manager.log_system_metric('charts_generated', len(response['charts']))
            
        except Exception as e:
            print(f"Error storing conversation: {e}")
        
        return response
    
    def _analyze_symbol_comprehensive(self, symbol: str, intents: List[str]) -> Dict[str, Any]:
        """Comprehensive analysis of a single symbol with all requested insights including multi-asset, cross-sector, and anticipation factors."""
        try:
            # Get real-time data
            df, asset_type = self._get_real_time_data(symbol)
            
            if df is None or df.empty:
                return None
            
            analysis = {
                'symbol': symbol,
                'asset_type': asset_type,
                'data': df,
                'current_price': float(df['Close'].iloc[-1]),
                'timestamp': datetime.now().isoformat()
            }
            
            # Get ML prediction
            try:
                self.analyzer.train_on_data(df)
                prediction = self.analyzer.predict(df)
                analysis['prediction'] = prediction
            except Exception as e:
                print(f"ML prediction error for {symbol}: {e}")
                analysis['prediction'] = {'signal': 'NEUTRAL', 'confidence': 0.5, 'bullish_prob': 0.5}
            
            # Calculate key metrics
            try:
                analysis['metrics'] = self._calculate_key_metrics(df)
            except Exception as e:
                print(f"Metrics error for {symbol}: {e}")
                analysis['metrics'] = {}
            
            # Technical analysis
            try:
                df_with_indicators = add_indicators(df.copy())
                if not df_with_indicators.empty:
                    analysis['technical'] = self._extract_technical_signals(df_with_indicators)
            except Exception as e:
                print(f"Technical analysis error for {symbol}: {e}")
                analysis['technical'] = {}
            
            # Risk analysis
            if 'risk' in intents:
                try:
                    analysis['risk'] = self._calculate_risk_metrics(df)
                except Exception as e:
                    print(f"Risk analysis error for {symbol}: {e}")
                    analysis['risk'] = {}
            
            # Price levels and targets
            try:
                analysis['levels'] = self._calculate_price_levels(df)
            except Exception as e:
                print(f"Price levels error for {symbol}: {e}")
                analysis['levels'] = {}
            
            # Multi-asset specific analysis
            if 'options' in intents and asset_type.lower() == 'stock':
                try:
                    options_analysis = self.multi_asset_analyzer.analyze_options(symbol)
                    analysis['options_analysis'] = options_analysis
                except Exception as e:
                    print(f"Options analysis error for {symbol}: {e}")
            
            if 'fx' in intents or asset_type.lower() == 'fx':
                try:
                    fx_analysis = self.multi_asset_analyzer.analyze_fx_pair(symbol)
                    analysis['fx_analysis'] = fx_analysis
                except Exception as e:
                    print(f"FX analysis error for {symbol}: {e}")
            
            if 'futures' in intents or asset_type.lower() == 'futures':
                try:
                    futures_analysis = self.multi_asset_analyzer.analyze_futures(symbol)
                    analysis['futures_analysis'] = futures_analysis
                except Exception as e:
                    print(f"Futures analysis error for {symbol}: {e}")
            
            # Advanced news and sentiment analysis with hundreds of articles
            if 'news_sentiment' in intents or 'comprehensive' in intents or any(keyword in intents for keyword in ['news', 'sentiment']):
                try:
                    # Process hundreds of news articles
                    processed_articles = asyncio.run(
                        self.advanced_news_processor.process_news_comprehensive(max_articles=300)
                    )
                    
                    # Filter articles relevant to symbol
                    relevant_articles = [
                        article for article in processed_articles 
                        if symbol in article.symbols_mentioned or 
                        any(sector in article.sectors_mentioned for sector in self._get_symbol_sectors(symbol))
                    ]
                    
                    # Aggregate news analysis
                    news_summary = self._aggregate_news_analysis(relevant_articles, symbol)
                    analysis['advanced_news_analysis'] = news_summary
                    
                except Exception as e:
                    print(f"Advanced news analysis error for {symbol}: {e}")
                    analysis['advanced_news_analysis'] = {
                        'error': 'Advanced news analysis temporarily unavailable',
                        'fallback_sentiment': self.news_engine.get_sentiment_for_symbol(symbol)
                    }
            
            # Cross-sector correlation analysis
            if 'cross_sector' in intents or 'comprehensive' in intents:
                try:
                    cross_sector_analysis = asyncio.run(
                        self.cross_sector_analyzer.analyze_symbol_cross_sector_impact(symbol)
                    )
                    analysis['cross_sector_analysis'] = cross_sector_analysis
                except Exception as e:
                    print(f"Cross-sector analysis error for {symbol}: {e}")
            
            # Generate AI vs Market narrative
            if 'comprehensive' in intents or len(intents) > 3:
                try:
                    narrative_comparison = asyncio.run(
                        self.narrative_generator.generate_comprehensive_narrative(symbol, include_cross_sector=True)
                    )
                    analysis['narrative_comparison'] = narrative_comparison
                except Exception as e:
                    print(f"Narrative generation error for {symbol}: {e}")
            
            # Enhanced anticipation factor analysis
            try:
                analysis['anticipation_factors'] = self._calculate_enhanced_anticipation_factors(
                    analysis, intents
                )
            except Exception as e:
                print(f"Anticipation factors error for {symbol}: {e}")
                analysis['anticipation_factors'] = {}
            
            return analysis
            
        except Exception as e:
            print(f"Comprehensive analysis error for {symbol}: {e}")
            # Instead of returning None, try to return whatever we've gathered
            if 'analysis' in locals():
                return locals()['analysis']
            return None
    
    def _get_symbol_sectors(self, symbol: str) -> List[str]:
        """Get sectors associated with a symbol."""
        # Simplified sector mapping - in real implementation would use comprehensive database
        sector_mappings = {
            'AAPL': ['technology', 'consumer_discretionary'],
            'MSFT': ['technology'],
            'GOOGL': ['technology', 'communication'],
            'AMZN': ['technology', 'consumer_discretionary'],
            'TSLA': ['technology', 'consumer_discretionary'],
            'JPM': ['finance'],
            'XOM': ['energy'],
            'JNJ': ['healthcare'],
            'SPY': ['broad_market'],
            'QQQ': ['technology'],
            'BTC-USD': ['cryptocurrency'],
            'EUR/USD': ['forex'],
            'CL=F': ['energy', 'commodities']
        }
        
        return sector_mappings.get(symbol, ['general'])
    
    def _aggregate_news_analysis(self, articles: List, symbol: str) -> Dict[str, Any]:
        """Aggregate analysis from hundreds of processed articles."""
        if not articles:
            return {'error': 'No relevant articles found'}
        
        try:
            # Aggregate sentiment scores
            sentiment_scores = [article.sentiment_score for article in articles if hasattr(article, 'sentiment_score')]
            overall_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
            
            # Aggregate market impact scores
            impact_scores = [article.market_impact_score for article in articles if hasattr(article, 'market_impact_score')]
            overall_impact = np.mean(impact_scores) if impact_scores else 0
            
            # Extract key themes
            all_themes = []
            for article in articles:
                if hasattr(article, 'narrative_elements'):
                    all_themes.extend(article.narrative_elements)
            
            # Count theme frequency
            theme_counts = {}
            for theme in all_themes:
                theme_counts[theme] = theme_counts.get(theme, 0) + 1
            
            top_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Aggregate cross-sector implications
            cross_sector_implications = {}
            for article in articles:
                if hasattr(article, 'cross_sector_implications'):
                    for sector, impact in article.cross_sector_implications.items():
                        if sector not in cross_sector_implications:
                            cross_sector_implications[sector] = []
                        cross_sector_implications[sector].append(impact)
            
            # Average cross-sector impacts
            for sector in cross_sector_implications:
                cross_sector_implications[sector] = np.mean(cross_sector_implications[sector])
            
            # Aggregate anticipation factors
            anticipation_factors = {}
            for article in articles:
                if hasattr(article, 'anticipation_factors'):
                    for factor, value in article.anticipation_factors.items():
                        if factor not in anticipation_factors:
                            anticipation_factors[factor] = []
                        anticipation_factors[factor].append(value)
            
            # Average anticipation factors
            for factor in anticipation_factors:
                anticipation_factors[factor] = np.mean(anticipation_factors[factor])
            
            # Extract decision implications
            all_decision_implications = []
            for article in articles:
                if hasattr(article, 'decision_implications'):
                    all_decision_implications.extend(article.decision_implications)
            
            return {
                'articles_analyzed': len(articles),
                'overall_sentiment': overall_sentiment,
                'sentiment_category': self._score_to_sentiment_category(overall_sentiment),
                'overall_market_impact': overall_impact,
                'top_themes': [theme[0] for theme in top_themes],
                'theme_frequencies': dict(top_themes),
                'cross_sector_implications': cross_sector_implications,
                'anticipation_factors': anticipation_factors,
                'decision_implications': list(set(all_decision_implications))[:10],  # Top 10 unique implications
                'sentiment_distribution': {
                    'very_positive': len([s for s in sentiment_scores if s > 0.6]),
                    'positive': len([s for s in sentiment_scores if 0.2 < s <= 0.6]),
                    'neutral': len([s for s in sentiment_scores if -0.2 <= s <= 0.2]),
                    'negative': len([s for s in sentiment_scores if -0.6 <= s < -0.2]),
                    'very_negative': len([s for s in sentiment_scores if s < -0.6])
                },
                'processing_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': f'News aggregation failed: {str(e)}'}
    
    def _score_to_sentiment_category(self, score: float) -> str:
        """Convert sentiment score to category."""
        if score > 0.6:
            return 'VERY_BULLISH'
        elif score > 0.2:
            return 'BULLISH'
        elif score > -0.2:
            return 'NEUTRAL'
        elif score > -0.6:
            return 'BEARISH'
        else:
            return 'VERY_BEARISH'
    
    def _calculate_enhanced_anticipation_factors(self, analysis: Dict[str, Any], intents: List[str]) -> Dict[str, float]:
        """Calculate enhanced anticipation factors across all model logic."""
        try:
            anticipation_factors = {}
            
            # Technical momentum anticipation
            prediction = analysis.get('prediction', {})
            technical = analysis.get('technical', {})
            
            if prediction.get('confidence', 0) > 0.7:
                anticipation_factors['high_confidence_signal'] = prediction['confidence']
            
            if technical.get('trend') in ['STRONG_BULLISH', 'STRONG_BEARISH']:
                anticipation_factors['strong_trend_continuation'] = 0.8
            
            # News sentiment anticipation
            news_analysis = analysis.get('advanced_news_analysis', {})
            if news_analysis and not news_analysis.get('error'):
                news_anticipation = news_analysis.get('anticipation_factors', {})
                for factor, value in news_anticipation.items():
                    anticipation_factors[f'news_{factor}'] = value
            
            # Cross-sector anticipation
            cross_sector = analysis.get('cross_sector_analysis', {})
            if cross_sector and not cross_sector.get('error'):
                if cross_sector.get('sector_rotation_signal'):
                    anticipation_factors['sector_rotation'] = 0.7
                
                if cross_sector.get('cross_asset_divergence', 0) > 0.3:
                    anticipation_factors['cross_asset_divergence'] = cross_sector['cross_asset_divergence']
            
            # Volatility regime anticipation
            metrics = analysis.get('metrics', {})
            if metrics.get('volatility_20d', 0) > 30:
                anticipation_factors['high_volatility_regime'] = 0.6
            elif metrics.get('volatility_20d', 0) < 10:
                anticipation_factors['low_volatility_regime'] = 0.5
            
            # Multi-asset specific anticipation factors
            if 'options_analysis' in analysis:
                options = analysis['options_analysis']
                if not options.get('error'):
                    if options.get('iv_percentile', 50) > 80:
                        anticipation_factors['high_iv_mean_reversion'] = 0.7
                    elif options.get('iv_percentile', 50) < 20:
                        anticipation_factors['low_iv_expansion'] = 0.6
            
            if 'fx_analysis' in analysis:
                fx = analysis['fx_analysis']
                if not fx.get('error'):
                    carry_analysis = fx.get('carry_trade_analysis', {})
                    if carry_analysis.get('recommendation') == 'FAVORABLE_CARRY_LONG':
                        anticipation_factors['favorable_carry_trade'] = 0.8
            
            if 'futures_analysis' in analysis:
                futures = analysis['futures_analysis']
                if not futures.get('error'):
                    curve_analysis = futures.get('curve_analysis', {})
                    if curve_analysis.get('structure') == 'BACKWARDATION':
                        anticipation_factors['supply_tightness'] = 0.7
            
            # Narrative comparison anticipation
            narrative = analysis.get('narrative_comparison')
            if narrative and not getattr(narrative, 'narrative_text', '').startswith('Error'):
                if narrative.confidence_divergence > 0.4:
                    anticipation_factors['ai_market_divergence'] = narrative.confidence_divergence
                
                # Add anticipation factors from narrative
                narrative_anticipation = getattr(narrative, 'anticipation_factors', {})
                for factor, value in narrative_anticipation.items():
                    anticipation_factors[f'narrative_{factor}'] = value
            
            return anticipation_factors
            
        except Exception as e:
            print(f"Enhanced anticipation factors calculation error: {e}")
            return {}
    
    def _calculate_key_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate key financial metrics."""
        if df.empty:
            return {}
        
        current_price = float(df['Close'].iloc[-1])
        
        metrics = {
            'current_price': current_price,
            'daily_change': 0,
            'daily_change_pct': 0,
            'weekly_change_pct': 0,
            'monthly_change_pct': 0,
            'ytd_change_pct': 0,
            'volatility_20d': 0,
            'volume_ratio': 1.0
        }
        
        try:
            # Daily change
            if len(df) > 1:
                prev_close = float(df['Close'].iloc[-2])
                metrics['daily_change'] = current_price - prev_close
                metrics['daily_change_pct'] = (metrics['daily_change'] / prev_close) * 100
            
            # Weekly change
            if len(df) > 5:
                week_ago_price = float(df['Close'].iloc[-6])
                metrics['weekly_change_pct'] = ((current_price / week_ago_price) - 1) * 100
            
            # Monthly change
            if len(df) > 20:
                month_ago_price = float(df['Close'].iloc[-21])
                metrics['monthly_change_pct'] = ((current_price / month_ago_price) - 1) * 100
            
            # YTD change (approximate)
            if len(df) > 60:
                ytd_start_price = float(df['Close'].iloc[-252] if len(df) > 252 else df['Close'].iloc[0])
                metrics['ytd_change_pct'] = ((current_price / ytd_start_price) - 1) * 100
            
            # Volatility
            if len(df) > 20:
                returns = df['Close'].pct_change().dropna()
                metrics['volatility_20d'] = returns.tail(20).std() * np.sqrt(252) * 100
            
            # Volume ratio
            if 'Volume' in df.columns and len(df) > 20:
                current_volume = float(df['Volume'].iloc[-1])
                avg_volume = df['Volume'].tail(20).mean()
                if avg_volume > 0:
                    metrics['volume_ratio'] = current_volume / avg_volume
        
        except Exception as e:
            print(f"Error calculating metrics: {e}")
        
        return metrics
    
    def _extract_technical_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract technical analysis signals."""
        if df.empty:
            return {}
        
        latest = df.iloc[-1]
        signals = {
            'trend': 'NEUTRAL',
            'momentum': 'NEUTRAL',
            'rsi_signal': 'NEUTRAL',
            'ema_signal': 'NEUTRAL',
            'support_resistance': {}
        }
        
        try:
            # RSI analysis
            if 'rsi' in latest and pd.notna(latest['rsi']):
                rsi = latest['rsi']
                if rsi > 70:
                    signals['rsi_signal'] = 'OVERBOUGHT'
                elif rsi < 30:
                    signals['rsi_signal'] = 'OVERSOLD'
                else:
                    signals['rsi_signal'] = 'NEUTRAL'
            
            # EMA trend
            if 'ema20' in latest and 'ema50' in latest:
                if pd.notna(latest['ema20']) and pd.notna(latest['ema50']):
                    if latest['ema20'] > latest['ema50']:
                        signals['ema_signal'] = 'BULLISH'
                        signals['trend'] = 'UPTREND'
                    else:
                        signals['ema_signal'] = 'BEARISH'
                        signals['trend'] = 'DOWNTREND'
            
            # Support and resistance
            if len(df) >= 50:
                high_50 = df['High'].tail(50).max()
                low_50 = df['Low'].tail(50).min()
                signals['support_resistance'] = {
                    'resistance': float(high_50),
                    'support': float(low_50),
                    'current_position': (float(latest['Close']) - low_50) / (high_50 - low_50) if high_50 > low_50 else 0.5
                }
        
        except Exception as e:
            print(f"Error extracting technical signals: {e}")
        
        return signals
    
    def _calculate_risk_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive risk metrics."""
        if df.empty or len(df) < 20:
            return {}
        
        returns = df['Close'].pct_change().dropna()
        
        risk_metrics = {}
        
        try:
            # Volatility
            risk_metrics['volatility_annual'] = returns.std() * np.sqrt(252) * 100
            
            # Value at Risk (95%)
            risk_metrics['var_95'] = returns.quantile(0.05) * 100
            
            # Maximum Drawdown
            cum_returns = (1 + returns).cumprod()
            running_max = cum_returns.expanding().max()
            drawdown = (cum_returns - running_max) / running_max
            risk_metrics['max_drawdown'] = drawdown.min() * 100
            
            # Sharpe Ratio (assuming 2% risk-free rate)
            excess_returns = returns - (0.02 / 252)  # Daily risk-free rate
            risk_metrics['sharpe_ratio'] = (excess_returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
            
            # Sortino Ratio
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_std = downside_returns.std()
                risk_metrics['sortino_ratio'] = (excess_returns.mean() / downside_std * np.sqrt(252)) if downside_std > 0 else 0
            else:
                risk_metrics['sortino_ratio'] = float('inf')
            
            # Beta (if we have market data)
            # This would require SPY data for comparison - simplified for now
            risk_metrics['beta'] = 1.0  # Placeholder
            
        except Exception as e:
            print(f"Error calculating risk metrics: {e}")
        
        return risk_metrics
    
    def _calculate_price_levels(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate key price levels and targets."""
        if df.empty:
            return {}
        
        current_price = float(df['Close'].iloc[-1])
        levels = {'current': current_price}
        
        try:
            # Fibonacci retracements (using recent high/low)
            if len(df) >= 50:
                recent_high = df['High'].tail(50).max()
                recent_low = df['Low'].tail(50).min()
                
                fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
                levels['fibonacci'] = {}
                
                for fib in fib_levels:
                    levels['fibonacci'][f'fib_{fib}'] = recent_low + (recent_high - recent_low) * fib
            
            # Pivot points (daily)
            if len(df) >= 3:
                yesterday = df.iloc[-2]
                pivot = (yesterday['High'] + yesterday['Low'] + yesterday['Close']) / 3
                levels['pivot'] = float(pivot)
                levels['r1'] = float(2 * pivot - yesterday['Low'])
                levels['s1'] = float(2 * pivot - yesterday['High'])
                levels['r2'] = float(pivot + (yesterday['High'] - yesterday['Low']))
                levels['s2'] = float(pivot - (yesterday['High'] - yesterday['Low']))
        
        except Exception as e:
            print(f"Error calculating price levels: {e}")
        
        return levels


    def _get_real_time_data(self, symbol: str, period: str = '1y') -> Tuple[Optional[pd.DataFrame], str]:
        """Fetch real-time market data with caching and error handling."""
        start_time = time.time()
        
        try:
            # Check cache first
            cached_data = self.db_manager.get_market_data(symbol, days=365)
            
            # Determine if we need fresh data (older than 1 hour)
            need_fresh_data = True
            if not cached_data.empty:
                latest_timestamp = cached_data.index[-1]
                time_diff = datetime.now() - latest_timestamp.to_pydatetime()
                if time_diff.total_seconds() < 3600:  # Less than 1 hour old
                    need_fresh_data = False
            
            df = None
            asset_type = 'Unknown'
            
            if need_fresh_data:
                # Determine asset type and fetch data
                if '/' in symbol or '_' in symbol or any(curr in symbol for curr in ['EUR', 'GBP', 'JPY', 'CHF']):
                    # FX pair
                    fx_symbol = symbol.replace('/', '_').replace('-', '_')
                    df = get_fx(fx_symbol)
                    asset_type = 'FX'
                elif '=F' in symbol or symbol in ['ES', 'NQ', 'CL', 'GC', 'SI']:
                    # Futures
                    df = get_futures_proxy(symbol, period=period)
                    asset_type = 'Futures'
                else:
                    # Stock or crypto
                    df = get_stock(symbol, period=period)
                    asset_type = 'Crypto' if 'USD' in symbol and '-' in symbol else 'Stock'
                
                # Store in database if successful
                if df is not None and not df.empty:
                    self.db_manager.store_market_data(symbol, df, asset_type.lower())
                    
                    # Store technical indicators
                    df_with_indicators = add_indicators(df.copy())
                    if not df_with_indicators.empty:
                        self.db_manager.store_technical_indicators(symbol, df_with_indicators)
            else:
                # Use cached data
                df = cached_data
                # Determine asset type from symbol
                if '/' in symbol or '_' in symbol:
                    asset_type = 'FX'
                elif '=F' in symbol:
                    asset_type = 'Futures'
                elif '-USD' in symbol:
                    asset_type = 'Crypto'
                else:
                    asset_type = 'Stock'
            
            # Log performance metrics
            response_time = (time.time() - start_time) * 1000
            self.db_manager.log_query_analytics(
                'data_fetch', symbol, int(response_time), 
                df is not None and not df.empty
            )
            
            return df, asset_type
            
        except Exception as e:
            error_msg = f"Error fetching data for {symbol}: {e}"
            print(error_msg)
            
            # Log error
            response_time = (time.time() - start_time) * 1000
            self.db_manager.log_query_analytics(
                'data_fetch', symbol, int(response_time), False, str(e)
            )
            
            return None, 'Unknown'
    
    def _create_advanced_price_chart(self, df: pd.DataFrame, symbol: str, 
                                     show_indicators: bool = True, 
                                     chart_style: str = 'comprehensive') -> go.Figure:
        """Create advanced interactive price chart with multiple analysis layers."""
        if df.empty:
            return None
        
        # Use more data for better analysis
        df_display = df.tail(200).copy()
        df_with_ind = add_indicators(df_display.copy())
        
        if chart_style == 'comprehensive':
            # 4-panel comprehensive chart
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(
                    f'{symbol} - Octavian Advanced Analysis', 
                    'Momentum Indicators (RSI & MACD)', 
                    'Volume Profile & Analysis',
                    'Volatility & Market Structure'
                ),
                row_heights=[0.4, 0.25, 0.2, 0.15]
            )
        else:
            # Standard 3-panel chart
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=(f'{symbol} Price Action', 'RSI', 'Volume'),
                row_heights=[0.5, 0.25, 0.25]
            )
        
        # Enhanced Candlestick with Octavian colors (gold theme)
        fig.add_trace(
            go.Candlestick(
                x=df_display.index,
                open=df_display['Open'],
                high=df_display['High'],
                low=df_display['Low'],
                close=df_display['Close'],
                name='Price',
                increasing_line_color='#FFD700',  # Gold
                decreasing_line_color='#B8860B',  # Dark gold
                increasing_fillcolor='#FFD700',
                decreasing_fillcolor='#B8860B'
            ),
            row=1, col=1
        )
        
        if show_indicators and not df_with_ind.empty:
            # Multiple EMAs with Octavian theme
            ema_configs = [
                ('ema20', '#DAA520', 'EMA 20', 2),  # Goldenrod
                ('ema50', '#CD853F', 'EMA 50', 2),  # Peru
            ]
            
            # Add 200 EMA if we have enough data
            if len(df_display) >= 200:
                df_with_ind['ema200'] = df_display['Close'].ewm(span=200, adjust=False).mean()
                ema_configs.append(('ema200', '#8B4513', 'EMA 200', 3))  # Saddle brown
            
            for ema_col, color, name, width in ema_configs:
                if ema_col in df_with_ind.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df_with_ind.index,
                            y=df_with_ind[ema_col],
                            mode='lines',
                            name=name,
                            line=dict(color=color, width=width),
                            opacity=0.8
                        ),
                        row=1, col=1
                    )
            
            # Enhanced RSI with Octavian styling
            if 'rsi' in df_with_ind.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_with_ind.index,
                        y=df_with_ind['rsi'],
                        mode='lines',
                        name='RSI',
                        line=dict(color='#DAA520', width=2),  # Goldenrod
                        fill='tozeroy',
                        fillcolor='rgba(218, 165, 32, 0.2)'
                    ),
                    row=2, col=1
                )
                
                # RSI levels with Octavian theme
                rsi_levels = [
                    (80, "#FF6B6B", "Extreme Overbought"),
                    (70, "#FFA500", "Overbought"),
                    (50, "#DAA520", "Neutral"),
                    (30, "#32CD32", "Oversold"),
                    (20, "#228B22", "Extreme Oversold")
                ]
                
                for level, color, label in rsi_levels:
                    fig.add_hline(
                        y=level, 
                        line_dash="dash" if level in [70, 30] else "dot", 
                        line_color=color, 
                        opacity=0.7 if level in [70, 30] else 0.4,
                        annotation_text=label if level in [80, 20] else "",
                        row=2, col=1
                    )
        
        # Enhanced Volume Analysis with Octavian colors
        if 'Volume' in df_display.columns:
            colors = []
            for i in range(len(df_display)):
                if df_display['Close'].iloc[i] > df_display['Open'].iloc[i]:
                    colors.append('#FFD700')  # Gold for up days
                else:
                    colors.append('#B8860B')  # Dark gold for down days
            
            volume_row = 3 if chart_style != 'comprehensive' else 3
            
            fig.add_trace(
                go.Bar(
                    x=df_display.index,
                    y=df_display['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=volume_row, col=1
            )
        
        # Enhanced layout with Octavian branding
        fig.update_layout(
            height=900 if chart_style == 'comprehensive' else 700,
            template='plotly_dark',
            showlegend=True,
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=1.02, 
                xanchor="right", 
                x=1,
                bgcolor="rgba(0,0,0,0.5)"
            ),
            xaxis_rangeslider_visible=False,
            title={
                'text': f' {symbol} - Octavian Advanced Analysis',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#DAA520'}
            },
            hovermode='x unified',
            paper_bgcolor='rgba(0,0,0,0.9)',
            plot_bgcolor='rgba(0,0,0,0.9)'
        )
        
        # Update axes with Octavian styling
        fig.update_xaxes(title_text="Date", row=-1, col=1, color='#DAA520')
        fig.update_yaxes(title_text="Price ($)", row=1, col=1, color='#DAA520')
        fig.update_yaxes(title_text="RSI" if chart_style != 'comprehensive' else "Momentum", row=2, col=1, color='#DAA520')
        fig.update_yaxes(title_text="Volume", row=3, col=1, color='#DAA520')
        
        if chart_style == 'comprehensive':
            fig.update_yaxes(title_text="Volatility (%)", row=4, col=1, color='#DAA520')
        
        return fig
    
    def _create_prediction_chart(self, df: pd.DataFrame, symbol: str, 
                                  prediction: Dict) -> go.Figure:
        """Create a chart showing prediction visualization with Octavian styling."""
        if df.empty:
            return None
        
        df_recent = df.tail(60).copy()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'{symbol} Recent Price Action',
                'Signal Strength',
                'Model Confidence',
                'Price Position'
            ),
            specs=[[{"type": "scatter"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]]
        )
        
        # Price chart with trend line (Octavian colors)
        fig.add_trace(
            go.Scatter(
                x=df_recent.index,
                y=df_recent['Close'],
                mode='lines',
                name='Price',
                line=dict(color='#DAA520', width=2)  # Goldenrod
            ),
            row=1, col=1
        )
        
        # Add trend line
        x_numeric = np.arange(len(df_recent))
        z = np.polyfit(x_numeric, df_recent['Close'].values, 1)
        p = np.poly1d(z)
        trend_color = '#FFD700' if z[0] > 0 else '#B8860B'  # Gold theme
        
        fig.add_trace(
            go.Scatter(
                x=df_recent.index,
                y=p(x_numeric),
                mode='lines',
                name='Trend',
                line=dict(color=trend_color, width=2, dash='dash')
            ),
            row=1, col=1
        )
        
        # Signal strength gauge with Octavian colors
        signal = prediction.get('signal', 'NEUTRAL')
        bullish_prob = prediction.get('bullish_prob', 0.5) * 100
        
        gauge_color = '#FFD700' if signal == 'BULLISH' else '#B8860B' if signal == 'BEARISH' else '#DAA520'
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=bullish_prob,
                title={'text': f"Signal: {signal}"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': gauge_color},
                    'steps': [
                        {'range': [0, 40], 'color': '#2F1B14'},
                        {'range': [40, 60], 'color': '#8B4513'},
                        {'range': [60, 100], 'color': '#DAA520'}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': bullish_prob
                    }
                }
            ),
            row=1, col=2
        )
        
        # Confidence indicator
        confidence = prediction.get('confidence', 0.5) * 100
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=confidence,
                title={'text': "Confidence %"},
                delta={'reference': 50, 'relative': False},
                number={'suffix': '%', 'font': {'color': '#DAA520'}}
            ),
            row=2, col=1
        )
        
        # Price position indicator
        high_20 = df_recent['High'].max()
        low_20 = df_recent['Low'].min()
        current = df_recent['Close'].iloc[-1]
        position = (current - low_20) / (high_20 - low_20) * 100 if high_20 > low_20 else 50
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=position,
                title={'text': "Price Position (20d Range)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': '#DAA520'},
                    'steps': [
                        {'range': [0, 20], 'color': '#2F1B14'},
                        {'range': [20, 80], 'color': '#8B4513'},
                        {'range': [80, 100], 'color': '#CD853F'}
                    ]
                }
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            template='plotly_dark',
            showlegend=True,
            title=f' {symbol} Octavian ML Prediction',
            paper_bgcolor='rgba(0,0,0,0.9)',
            plot_bgcolor='rgba(0,0,0,0.9)'
        )
        
        return fig




def show_advanced_chatbot():
    """Advanced Streamlit interface for the AI chatbot with enhanced features."""
    st.header(" Advanced AI Market Assistant")
    
    # Enhanced description with capabilities
    st.markdown("""
    ###  **Ne xt-Generation Market Intelligence**
    
    I'm your advanced AI market analyst powered by real-time data, machine learning, and comprehensive technical analysis. Here's what I can do:
    
    **[SCAN] Real-Time Analysis:**
    - Live market data with 100% accuracy
    - Advanced technical indicators and patterns
    - Multi-timeframe analysis (intraday to long-term)
    
    ** AI-Powered Insights:**
    - Machine learning predictions with confidence levels
    - Multi-model ensemble analysis
    - Sentiment and momentum detection
    
    ** Visual Intelligence:**
    - Interactive charts with 20+ technical indicators
    - Risk and volatility visualizations
    - Comparative performance analysis
    
    ** Smart Features:**
    - Conversation memory and context
    - Personalized recommendations
    - Follow-up suggestions
    - Performance analytics
    
    ---
    
    ** Example Queries:**
    """)
    
    # Example queries in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ** Analysis & Predictions**
        - "Analyze AAPL with full technical analysis"
        - "What's your prediction for TSLA?"
        - "Bitcoin price forecast for next week"
        - "Show me risk analysis for SPY"
        """)
    
    with col2:
        st.markdown("""
        ** Comparisons**
        - "Compare MSFT vs GOOGL performance"
        - "NVDA versus AMD analysis"
        - "SPY vs QQQ which is better?"
        - "Bitcoin vs Ethereum comparison"
        """)
    
    with col3:
        st.markdown("""
        ** Specific Insights**
        - "Support and resistance for AMZN"
        - "Is META overbought right now?"
        - "Volatility analysis for VIX"
        - "Forex EUR/USD technical analysis"
        """)
    
    st.markdown("---")
    
    # Initialize chatbot in session state
    if 'advanced_chatbot' not in st.session_state:
        st.session_state.advanced_chatbot = get_chatbot()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'user_id' not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    
    # Sidebar with analytics and settings
    with st.sidebar:
        st.markdown("###  Session Analytics")
        
        if st.session_state.chat_history:
            total_queries = len(st.session_state.chat_history)
            total_charts = sum(len(chat.get('response', {}).get('charts', [])) for chat in st.session_state.chat_history)
            avg_response_time = np.mean([chat.get('response', {}).get('response_time_ms', 0) for chat in st.session_state.chat_history])
            
            st.metric("Total Queries", total_queries)
            st.metric("Charts Generated", total_charts)
            st.metric("Avg Response Time", f"{avg_response_time:.0f}ms")
        
        st.markdown("###  Settings")
        
        # Chart preferences
        chart_style = st.selectbox(
            "Chart Style",
            ["Comprehensive", "Standard", "Minimal"],
            index=0,
            help="Choose the level of detail in charts"
        )
        
        # Analysis depth
        analysis_depth = st.selectbox(
            "Analysis Depth",
            ["Deep", "Standard", "Quick"],
            index=1,
            help="Choose how detailed the analysis should be"
        )
        
        # Auto-suggestions
        show_suggestions = st.checkbox(
            "Show Follow-up Suggestions",
            value=True,
            help="Display contextual suggestions after each response"
        )
        
        st.markdown("###  Actions")
        
        if st.button(" View Analytics Dashboard"):
            st.session_state.show_analytics = True
        
        if st.button(" Export Chat History"):
            if st.session_state.chat_history:
                chat_df = pd.DataFrame([
                    {
                        'timestamp': chat.get('timestamp', ''),
                        'query': chat.get('query', ''),
                        'symbols': ', '.join(chat.get('response', {}).get('symbols', [])),
                        'intents': ', '.join(chat.get('response', {}).get('intents', [])),
                        'response_time_ms': chat.get('response', {}).get('response_time_ms', 0)
                    }
                    for chat in st.session_state.chat_history
                ])
                
                csv = chat_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        if st.button(" Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Display analytics dashboard if requested
    if st.session_state.get('show_analytics', False):
        st.markdown("###  Analytics Dashboard")
        
        try:
            db_manager = get_database_manager()
            analytics = db_manager.get_analytics_dashboard()
            
            if analytics:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    conv_stats = analytics.get('conversation_stats', {})
                    st.metric("Total Conversations", conv_stats.get('total_conversations', 0))
                
                with col2:
                    st.metric("Unique Sessions", conv_stats.get('unique_sessions', 0))
                
                with col3:
                    avg_time = conv_stats.get('avg_response_time', 0)
                    st.metric("Avg Response Time", f"{avg_time:.0f}ms")
                
                with col4:
                    avg_charts = conv_stats.get('avg_charts_per_query', 0)
                    st.metric("Avg Charts/Query", f"{avg_charts:.1f}")
                
                # Popular symbols
                if analytics.get('popular_symbols'):
                    st.markdown("####  Most Analyzed Symbols")
                    symbols_df = pd.DataFrame(analytics['popular_symbols'])
                    st.bar_chart(symbols_df.set_index('symbol')['requests'])
                
                # Query types
                if analytics.get('query_volume'):
                    st.markdown("####  Query Types")
                    query_df = pd.DataFrame(analytics['query_volume'])
                    st.bar_chart(query_df.set_index('query_type')['count'])
        
        except Exception as e:
            st.error(f"Error loading analytics: {e}")
        
        if st.button(" Close Analytics"):
            st.session_state.show_analytics = False
            st.rerun()
    
    # Chat interface
    st.markdown("###  Chat Interface")
    
    # Display chat history with enhanced formatting
    for i, chat in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(f"**Query #{i+1}:** {chat['query']}")
            
            # Show metadata
            response = chat.get('response', {})
            if response.get('symbols') or response.get('intents'):
                metadata_parts = []
                if response.get('symbols'):
                    metadata_parts.append(f" Symbols: {', '.join(response['symbols'])}")
                if response.get('intents'):
                    metadata_parts.append(f" Intent: {', '.join(response['intents'])}")
                if response.get('response_time_ms'):
                    metadata_parts.append(f" {response['response_time_ms']}ms")
                
                st.caption(" | ".join(metadata_parts))
        
        with st.chat_message("assistant"):
            # Display response text
            st.markdown(response.get('text', 'No response available'))
            
            # Display charts
            charts = response.get('charts', [])
            if charts:
                st.markdown(f"** Generated {len(charts)} chart{'s' if len(charts) > 1 else ''}:**")
                
                for chart_data in charts:
                    if 'figure' in chart_data and chart_data['figure'] is not None:
                        chart_title = chart_data.get('title', f"{chart_data.get('type', 'Chart')} - {chart_data.get('symbol', 'Unknown')}")
                        st.plotly_chart(chart_data['figure'], width='stretch', key=f"chart_{i}_{chart_data.get('type', 'unknown')}")
            
            # Show analysis summary if available
            if response.get('analysis_summary'):
                summary = response['analysis_summary']
                
                with st.expander(" Analysis Summary", expanded=False):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Symbols Analyzed", summary.get('total_symbols', 0))
                    with col2:
                        st.metric("Bullish Signals", summary.get('bullish_count', 0))
                    with col3:
                        st.metric("Bearish Signals", summary.get('bearish_count', 0))
                    with col4:
                        st.metric("Avg Confidence", f"{summary.get('avg_confidence', 0)*100:.0f}%")
                    
                    # High confidence signals
                    if summary.get('high_confidence_signals'):
                        st.markdown("** High Confidence Signals:**")
                        for signal in summary['high_confidence_signals']:
                            st.write(f"- {signal['symbol']}: {signal['signal']} ({signal['confidence']*100:.0f}% confidence)")
            
            # Show suggestions if enabled
            if show_suggestions and response.get('suggestions'):
                st.markdown("** Follow-up Suggestions:**")
                suggestions = response['suggestions'][:4]  # Limit to 4
                
                cols = st.columns(min(len(suggestions), 2))
                for idx, suggestion in enumerate(suggestions):
                    with cols[idx % 2]:
                        if st.button(f" {suggestion}", key=f"suggestion_{i}_{idx}"):
                            st.session_state.suggested_query = suggestion
    
    # Chat input
    user_query = st.chat_input("Ask me anything about the markets...")
    
    # Handle suggested query
    if st.session_state.get('suggested_query'):
        user_query = st.session_state.suggested_query
        st.session_state.suggested_query = None
    
    # Process new query
    if user_query:
        with st.chat_message("user"):
            query_display = f"**Query #{len(st.session_state.chat_history)+1}:** {user_query}"
            st.write(query_display)
        
        with st.chat_message("assistant"):
            with st.spinner(" Analyzing markets with AI..."):
                start_time = time.time()
                
                try:
                    response = st.session_state.advanced_chatbot.process_query(
                        user_query, 
                        user_id=st.session_state.user_id
                    )
                    
                    # Display response
                    st.markdown(response.get('text', 'No response generated'))
                    
                    # Display charts
                    charts = response.get('charts', [])
                    if charts:
                        st.markdown(f"** Generated {len(charts)} chart{'s' if len(charts) > 1 else ''}:**")
                        
                        for chart_data in charts:
                            if 'figure' in chart_data and chart_data['figure'] is not None:
                                chart_title = chart_data.get('title', f"{chart_data.get('type', 'Chart')} - {chart_data.get('symbol', 'Unknown')}")
                                st.plotly_chart(chart_data['figure'], width='stretch')
                    
                    # Show analysis summary
                    if response.get('analysis_summary'):
                        summary = response['analysis_summary']
                        
                        with st.expander(" Analysis Summary", expanded=False):
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Symbols Analyzed", summary.get('total_symbols', 0))
                            with col2:
                                st.metric("Bullish Signals", summary.get('bullish_count', 0))
                            with col3:
                                st.metric("Bearish Signals", summary.get('bearish_count', 0))
                            with col4:
                                st.metric("Avg Confidence", f"{summary.get('avg_confidence', 0)*100:.0f}%")
                    
                    # Show suggestions
                    if show_suggestions and response.get('suggestions'):
                        st.markdown("** Follow-up Suggestions:**")
                        suggestions = response['suggestions'][:4]
                        
                        cols = st.columns(min(len(suggestions), 2))
                        for idx, suggestion in enumerate(suggestions):
                            with cols[idx % 2]:
                                if st.button(f" {suggestion}", key=f"new_suggestion_{idx}"):
                                    st.session_state.suggested_query = suggestion
                                    st.rerun()
                    
                    # Store in history
                    st.session_state.chat_history.append({
                        'query': user_query,
                        'response': response,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Show success metrics
                    response_time = response.get('response_time_ms', 0)
                    symbols_count = len(response.get('symbols', []))
                    charts_count = len(response.get('charts', []))
                    
                    st.success(f" Analysis complete! Processed {symbols_count} symbol{'s' if symbols_count != 1 else ''}, generated {charts_count} chart{'s' if charts_count != 1 else ''} in {response_time}ms")
                
                except Exception as e:
                    st.error(f" Error processing query: {str(e)}")
                    st.error("Please try rephrasing your question or contact support if the issue persists.")
    
    # Quick action buttons at the bottom
    st.markdown("###  Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button(" Market Overview", key="quick_market"):
            st.session_state.suggested_query = "Give me a comprehensive market overview with SPY, QQQ, and VIX analysis"
            st.rerun()
    
    with col2:
        if st.button(" Top Movers", key="quick_movers"):
            st.session_state.suggested_query = "Show me today's top performing stocks and analyze the best ones"
            st.rerun()
    
    with col3:
        if st.button(" Crypto Analysis", key="quick_crypto"):
            st.session_state.suggested_query = "Analyze Bitcoin and Ethereum with predictions and technical analysis"
            st.rerun()
    
    with col4:
        if st.button(" Forex Update", key="quick_forex"):
            st.session_state.suggested_query = "EUR/USD and GBP/USD technical analysis with predictions"
            st.rerun()
    
    # Footer with tips
    st.markdown("---")
    st.markdown("""
    ** Pro Tips:**
    - Be specific with symbol names (e.g., "AAPL" not "Apple")
    - Ask for multiple types of analysis: "Analyze TSLA with risk assessment and predictions"
    - Compare assets: "MSFT vs GOOGL performance comparison"
    - Request specific timeframes: "NVDA analysis for next week"
    - Use follow-up questions to dive deeper into specific aspects
    """)



    def _generate_helpful_response(self, query: str, intents: List[str]) -> str:
        """Generate helpful response when no symbols are found."""
        base_response = "I'd be happy to help with your market analysis! "
        
        if 'prediction' in intents:
            return base_response + """To provide predictions, please specify which assets you'd like me to analyze. For example:
            
**Stock Examples:**
- "What's your prediction for AAPL?"
- "Should I buy TSLA stock?"
- "Analyze NVDA for next week"

**ETF Examples:**
- "Predict SPY movement"
- "QQQ outlook for this month"

**Crypto Examples:**
- "Bitcoin price prediction"
- "Analyze BTC-USD"
- "ETH-USD forecast"

**FX Examples:**
- "EUR/USD analysis"
- "GBP/USD prediction"
"""
        
        elif 'analysis' in intents:
            ex = self._get_asset_examples()
            return (base_response +
                    "I can provide comprehensive technical analysis for any asset. Please specify which one you'd like me to analyze:\n\n"
                    f"**Stocks:** {', '.join(ex['Stocks'])}\n"
                    f"**ETFs:** {', '.join(ex['ETFs'])}\n"
                    f"**Cryptocurrencies:** {', '.join(ex['Crypto'])}\n"
                    f"**Forex Pairs:** {', '.join(ex['Forex'])}\n"
                    f"**Futures:** {', '.join(ex['Futures'])}\n\n"
                    "Just ask: \"Analyze [SYMBOL]\" or \"Technical analysis of [SYMBOL]\"\n")
        
        elif 'comparison' in intents:
            return base_response + """I can compare multiple assets for you! Please specify which assets you'd like me to compare:

**Examples:**
- "Compare AAPL vs MSFT"
- "TSLA versus NIO performance"
- "SPY vs QQQ analysis"
- "Bitcoin vs Ethereum"
- "EUR/USD vs GBP/USD"

I'll show you relative performance, correlation analysis, and detailed comparisons.
"""
        
        else:
            return base_response + """Here are some things I can help you with:

[SCAN] **Analysis & Predictions**
- Technical analysis of any stock, ETF, crypto, or forex pair
- ML-powered price predictions with confidence levels
- Risk assessment and volatility analysis

 **Visual Analytics**
- Interactive price charts with technical indicators
- Comparison charts for multiple assets
- Volatility and risk visualization

 **Market Insights**
- Real-time market data and trends
- Support/resistance levels
- Trading recommendations

**Just ask me about any symbol!** For example: "Analyze AAPL", "Bitcoin prediction", or "Compare SPY vs QQQ"
"""
    
    def _get_query_suggestions(self) -> List[str]:
        """Get contextual query suggestions."""
        return [
            "Analyze SPY with technical indicators",
            "What's your prediction for AAPL?",
            "Compare TSLA vs NIO performance",
            "Bitcoin price analysis and forecast",
            "Show me risk analysis for QQQ",
            "EUR/USD technical analysis",
            "Market volatility analysis today"
        ]
    
    
    def _generate_intent_aware_response(self, query: str, symbol_analyses: Dict[str, Any],
                                       intent_analysis: IntentAnalysis, market_context: Dict[str, Any]) -> str:
        """
        Generate response tailored to detected intent analysis.
        Uses advanced intent detection to format response appropriately.
        """
        
        # Build response based on response format preference
        if intent_analysis.response_format == ResponseFormat.CONCISE:
            return self._generate_concise_response(query, symbol_analyses, intent_analysis, market_context)
        elif intent_analysis.response_format == ResponseFormat.DETAILED:
            return self._generate_detailed_response(query, symbol_analyses, intent_analysis, market_context)
        elif intent_analysis.response_format == ResponseFormat.NARRATIVE:
            return self._generate_narrative_response(query, symbol_analyses, intent_analysis, market_context)
        elif intent_analysis.response_format == ResponseFormat.DATA_FOCUSED:
            return self._generate_data_focused_response(query, symbol_analyses, intent_analysis, market_context)
        elif intent_analysis.response_format == ResponseFormat.ACTIONABLE:
            return self._generate_actionable_response(query, symbol_analyses, intent_analysis, market_context)
        elif intent_analysis.response_format == ResponseFormat.EDUCATIONAL:
            return self._generate_educational_response(query, symbol_analyses, intent_analysis, market_context)
        else:
            # Standard format
            return self._generate_standard_intent_response(query, symbol_analyses, intent_analysis, market_context)
    
    def _generate_standard_intent_response(self, query: str, symbol_analyses: Dict[str, Any],
                                          intent_analysis: IntentAnalysis, market_context: Dict[str, Any]) -> str:
        """Generate standard response with intent-specific emphasis."""
        
        # Header with intent context
        response = f"""#  OCTAVIAN Market Intelligence
*Query: "{query}"*
*Intent: {intent_analysis.primary_intent.value.replace('_', ' ').title()}*
*Confidence: {intent_analysis.confidence_score:.0%}*
*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

"""
        
        # Add urgency indicator if immediate
        if intent_analysis.urgency == Urgency.IMMEDIATE:
            response += "**URGENT ANALYSIS REQUESTED**\n\n"
        
        # Lead with what user wants to see first
        lead_with = intent_analysis.lead_with
        
        if lead_with == 'risk_metrics':
            response += "##  RISK ANALYSIS (Primary Focus)\n\n"
            for symbol, analysis in symbol_analyses.items():
                response += self._generate_risk_analysis_section(symbol, analysis)
                response += "\n"
        
        elif lead_with == 'forecast':
            response += "##  PREDICTION & FORECAST (Primary Focus)\n\n"
            for symbol, analysis in symbol_analyses.items():
                response += self._generate_prediction_section(symbol, analysis)
                response += "\n"
        
        elif lead_with == 'current_price':
            response += "##  CURRENT PRICING (Primary Focus)\n\n"
            for symbol, analysis in symbol_analyses.items():
                response += self._generate_price_section(symbol, analysis)
                response += "\n"
        
        # Add emphasized sections
        for emphasis in intent_analysis.emphasize[:3]:  # Top 3 emphasis areas
            if emphasis == 'comparison' and intent_analysis.is_comparison:
                response += self._generate_comparison_section(symbol_analyses)
            elif emphasis == 'volatility':
                response += "##  Volatility Analysis\n\n"
                for symbol, analysis in symbol_analyses.items():
                    response += self._generate_volatility_section(symbol, analysis)
        
        # Add standard analysis for remaining symbols
        response += "\n##  Complete Analysis\n\n"
        for symbol, analysis in symbol_analyses.items():
            response += f"### {symbol}\n"
            response += self._generate_symbol_summary(symbol, analysis, intent_analysis)
            response += "\n"
        
        # Add warnings if needed
        if intent_analysis.needs_risk_warning:
            response += "\n**RISK WARNING:** Trading involves substantial risk. "
            response += "This analysis is for informational purposes only.\n"
        
        return response
    
    def _generate_concise_response(self, query: str, symbol_analyses: Dict[str, Any],
                                   intent_analysis: IntentAnalysis, market_context: Dict[str, Any]) -> str:
        """Generate brief, bullet-point response."""
        response = f"**Quick Answer:** {query}\n\n"
        
        for symbol, analysis in symbol_analyses.items():
            prediction = analysis.get('prediction', {})
            metrics = analysis.get('metrics', {})
            
            response += f"**{symbol}:**\n"
            response += f"- Price: ${metrics.get('current_price', 0):.2f} ({metrics.get('daily_change_pct', 0):+.2f}%)\n"
            response += f"- Signal: {prediction.get('signal', 'NEUTRAL')} ({prediction.get('confidence', 0.5):.0%} confidence)\n"
            
            if intent_analysis.primary_intent == IntentCategory.RISK_ANALYSIS:
                risk = analysis.get('risk', {})
                response += f"- Volatility: {risk.get('volatility', 0):.1%}\n"
            
            response += "\n"
        
        return response
    
    def _generate_prediction_section(self, symbol: str, analysis: Dict[str, Any]) -> str:
        """Generate prediction-focused section."""
        prediction = analysis.get('prediction', {})
        
        text = f"**{symbol} Forecast:**\n"
        text += f"- Direction: {prediction.get('signal', 'NEUTRAL')}\n"
        text += f"- Confidence: {prediction.get('confidence', 0.5):.0%}\n"
        text += f"- Bullish Probability: {prediction.get('bullish_prob', 0.5):.0%}\n"
        
        if 'target_price' in prediction:
            text += f"- Price Target: ${prediction['target_price']:.2f}\n"
        
        return text
    
    def _generate_price_section(self, symbol: str, analysis: Dict[str, Any]) -> str:
        """Generate price-focused section."""
        metrics = analysis.get('metrics', {})
        
        text = f"**{symbol} Current Pricing:**\n"
        text += f"- Current: ${metrics.get('current_price', 0):.2f}\n"
        text += f"- Change: {metrics.get('daily_change_pct', 0):+.2f}%\n"
        text += f"- Volume: {metrics.get('volume', 0):,.0f}\n"
        
        return text
    
    def _generate_symbol_summary(self, symbol: str, analysis: Dict[str, Any],
                                 intent_analysis: IntentAnalysis) -> str:
        """Generate symbol summary based on intent."""
        prediction = analysis.get('prediction', {})
        metrics = analysis.get('metrics', {})
        
        summary = f"Price: ${metrics.get('current_price', 0):.2f} | "
        summary += f"Signal: {prediction.get('signal', 'NEUTRAL')} | "
        summary += f"Confidence: {prediction.get('confidence', 0.5):.0%}\n"
        
        return summary
    
    def _generate_intent_aware_charts(self, symbol_analyses: Dict[str, Any],
                                     intent_analysis: IntentAnalysis) -> List[Any]:
        """Generate charts based on intent analysis."""
        charts = []
        
        # Always include price chart if visualization requested
        if intent_analysis.requires_charts or intent_analysis.response_format == ResponseFormat.VISUAL:
            for symbol in list(symbol_analyses.keys())[:3]:
                chart = self._create_price_chart(symbol, symbol_analyses[symbol])
                if chart:
                    charts.append(chart)
        
        # Add risk chart if risk analysis requested
        if intent_analysis.primary_intent == IntentCategory.RISK_ANALYSIS:
            for symbol in list(symbol_analyses.keys())[:2]:
                vol_chart = self._create_volatility_chart(
                    symbol_analyses[symbol].get('df'),
                    symbol
                )
                if vol_chart:
                    charts.append(vol_chart)
        
        # Add comparison chart if comparison requested
        if intent_analysis.is_comparison and len(symbol_analyses) > 1:
            comp_chart = self._create_comparison_chart(list(symbol_analyses.keys()))
            if comp_chart:
                charts.append(comp_chart)
        
        return charts
    
    def _generate_comprehensive_response(self, query: str, symbol_analyses: Dict[str, Any], 
                                       intents: List[str], market_context: Dict[str, Any]) -> str:
        """Generate comprehensive AI response with market context."""
        
        # Start with market context if relevant
        response = f"""#  AI Market Analysis Report
*Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} EST*

"""
        
        # Add market regime context
        if market_context:
            vol_regime = market_context.get('volatility_regime', 'Unknown')
            risk_mode = market_context.get('risk_mode', 'Unknown')
            
            response += f"""##  Current Market Environment
- **Volatility Regime:** {vol_regime}
- **Risk Sentiment:** {risk_mode}
- **Analysis Scope:** {len(symbol_analyses)} asset{'s' if len(symbol_analyses) > 1 else ''}

---

"""
        
        # Analyze each symbol
        for symbol, analysis in symbol_analyses.items():
            response += self._generate_symbol_analysis_text(symbol, analysis, intents)
            response += "\n---\n\n"
        
        # Add comparative insights if multiple symbols
        if len(symbol_analyses) > 1:
            response += self._generate_comparative_insights(symbol_analyses)
        
        # Add trading recommendations
        response += self._generate_trading_recommendations(symbol_analyses, intents, market_context)
        
        return response
    
    def _generate_symbol_analysis_text(self, symbol: str, analysis: Dict[str, Any], intents: List[str]) -> str:
        """Generate detailed analysis text for a single symbol with enhanced multi-asset and cross-sector insights."""
        
        prediction = analysis.get('prediction', {})
        metrics = analysis.get('metrics', {})
        technical = analysis.get('technical', {})
        news_sentiment = analysis.get('news_sentiment', {})
        advanced_news = analysis.get('advanced_news_analysis', {})
        cross_sector = analysis.get('cross_sector_analysis', {})
        narrative_comparison = analysis.get('narrative_comparison')
        anticipation_factors = analysis.get('anticipation_factors', {})
        
        current_price = metrics.get('current_price', 0)
        daily_change = metrics.get('daily_change', 0)
        daily_change_pct = metrics.get('daily_change_pct', 0)
        
        signal = prediction.get('signal', 'NEUTRAL')
        confidence = prediction.get('confidence', 0.5) * 100
        bullish_prob = prediction.get('bullish_prob', 0.5) * 100
        
        # Header with current status
        text = f"""##  {symbol} ({analysis.get('asset_type', 'Asset')}) - Comprehensive AI Analysis

###  Current Market Status
- **Price:** ${current_price:.2f} ({daily_change:+.2f} | {daily_change_pct:+.2f}%)
- **AI Signal:** **{signal}** ({confidence:.1f}% confidence)
- **Bullish Probability:** {bullish_prob:.1f}%

"""
        
        # Multi-Asset Specific Analysis
        if 'options_analysis' in analysis:
            options = analysis['options_analysis']
            if not options.get('error'):
                text += "###  Options Analysis\n"
                text += f"- **Implied Volatility:** {options.get('implied_volatility', 0):.1f}% ({options.get('iv_percentile', 50):.0f}th percentile)\n"
                text += f"- **Historical Volatility:** {options.get('historical_volatility', 0):.1f}%\n"
                
                greeks = options.get('greeks', {})
                if greeks:
                    text += f"- **Delta:** {greeks.get('delta', 0):.3f} | **Gamma:** {greeks.get('gamma', 0):.3f}\n"
                    text += f"- **Theta:** {greeks.get('theta', 0):.3f} | **Vega:** {greeks.get('vega', 0):.3f}\n"
                
                strategies = options.get('strategies', [])
                if strategies:
                    text += f"- **Recommended Strategy:** {strategies[0].get('strategy', 'N/A')}\n"
                    text += f"- **Rationale:** {strategies[0].get('rationale', 'N/A')}\n"
                
                text += "\n"
        
        if 'fx_analysis' in analysis:
            fx = analysis['fx_analysis']
            if not fx.get('error'):
                text += "###  FX Analysis\n"
                carry_analysis = fx.get('carry_trade_analysis', {})
                if carry_analysis:
                    text += f"- **Carry Differential:** {carry_analysis.get('carry_differential', 0):.2f}%\n"
                    text += f"- **Carry Attractiveness:** {carry_analysis.get('carry_attractiveness', 'N/A')}\n"
                    text += f"- **Recommendation:** {carry_analysis.get('recommendation', 'N/A')}\n"
                
                cb_analysis = fx.get('central_bank_analysis', {})
                if cb_analysis:
                    text += f"- **Policy Divergence:** {cb_analysis.get('policy_divergence', 'N/A')}\n"
                
                text += "\n"
        
        if 'futures_analysis' in analysis:
            futures = analysis['futures_analysis']
            if not futures.get('error'):
                text += "###  Futures Analysis\n"
                curve_analysis = futures.get('curve_analysis', {})
                if curve_analysis:
                    text += f"- **Curve Structure:** {curve_analysis.get('structure', 'N/A')}\n"
                    text += f"- **Curve Slope:** {curve_analysis.get('curve_slope_percent', 0):.2f}%\n"
                    text += f"- **Interpretation:** {curve_analysis.get('interpretation', 'N/A')}\n"
                
                seasonality = futures.get('seasonality', {})
                if seasonality and not seasonality.get('error'):
                    current_month = seasonality.get('current_month', 0)
                    seasonal_bias = seasonality.get('seasonal_bias', 'NEUTRAL')
                    text += f"- **Seasonal Bias (Month {current_month}):** {seasonal_bias}\n"
                
                text += "\n"
        
        # Advanced News Analysis (Hundreds of Articles)
        if advanced_news and not advanced_news.get('error'):
            text += "###  Advanced News Analysis (Comprehensive)\n"
            articles_count = advanced_news.get('articles_analyzed', 0)
            overall_sentiment = advanced_news.get('overall_sentiment', 0)
            sentiment_category = advanced_news.get('sentiment_category', 'NEUTRAL')
            
            # Sentiment emoji mapping
            sentiment_emoji = {
                'VERY_BULLISH': '',
                'BULLISH': '',
                'NEUTRAL': '',
                'BEARISH': '',
                'VERY_BEARISH': ''
            }
            
            text += f"- **Articles Analyzed:** {articles_count} from multiple sources\n"
            text += f"- **Overall Sentiment:** {sentiment_emoji.get(sentiment_category, '')} {sentiment_category} ({overall_sentiment:+.2f})\n"
            text += f"- **Market Impact Score:** {advanced_news.get('overall_market_impact', 0):.2f}/1.0\n"
            
            # Top themes from hundreds of articles
            top_themes = advanced_news.get('top_themes', [])
            if top_themes:
                text += f"- **Key Themes:** {', '.join(top_themes[:5])}\n"
            
            # Sentiment distribution
            sentiment_dist = advanced_news.get('sentiment_distribution', {})
            if sentiment_dist:
                total_articles = sum(sentiment_dist.values())
                if total_articles > 0:
                    bullish_pct = (sentiment_dist.get('very_positive', 0) + sentiment_dist.get('positive', 0)) / total_articles * 100
                    bearish_pct = (sentiment_dist.get('very_negative', 0) + sentiment_dist.get('negative', 0)) / total_articles * 100
                    text += f"- **Sentiment Distribution:** {bullish_pct:.0f}% Bullish, {bearish_pct:.0f}% Bearish\n"
            
            # Decision implications from news
            decision_implications = advanced_news.get('decision_implications', [])
            if decision_implications:
                text += f"- **Key Decision Implications:** {decision_implications[0]}\n"
            
            text += "\n"
        
        # Cross-Sector Correlation Analysis
        if cross_sector and not cross_sector.get('error'):
            text += "###  Cross-Sector Analysis\n"
            
            sector_correlations = cross_sector.get('sector_correlations', {})
            if sector_correlations:
                text += "- **Sector Correlations:** "
                high_corr_sectors = [sector for sector, corr in sector_correlations.items() if abs(corr) > 0.7]
                if high_corr_sectors:
                    text += f"Highly correlated with {', '.join(high_corr_sectors[:3])}\n"
                else:
                    text += "Low cross-sector correlations\n"
            
            rotation_impact = cross_sector.get('sector_rotation_impact', {})
            if rotation_impact:
                rotation_signal = rotation_impact.get('rotation_signal', 'NEUTRAL')
                text += f"- **Sector Rotation Signal:** {rotation_signal}\n"
            
            geopolitical_sensitivity = cross_sector.get('geopolitical_sensitivity', {})
            if geopolitical_sensitivity:
                sensitivity_level = geopolitical_sensitivity.get('sensitivity_level', 'MEDIUM')
                text += f"- **Geopolitical Sensitivity:** {sensitivity_level}\n"
            
            text += "\n"
        
        # AI vs Market Sentiment Narrative
        if narrative_comparison and not getattr(narrative_comparison, 'narrative_text', '').startswith('Error'):
            text += "###  AI vs Market Sentiment Narrative\n"
            
            confidence_divergence = getattr(narrative_comparison, 'confidence_divergence', 0)
            text += f"- **AI-Market Divergence:** {confidence_divergence:.2f} (0=aligned, 1=opposite)\n"
            
            # Extract key narrative points
            narrative_text = getattr(narrative_comparison, 'narrative_text', '')
            if narrative_text:
                # Extract first few sentences of narrative
                sentences = narrative_text.split('. ')[:3]
                key_narrative = '. '.join(sentences) + '.'
                text += f"- **Key Narrative:** {key_narrative}\n"
            
            # Decision implications from narrative
            decision_implications = getattr(narrative_comparison, 'decision_implications', [])
            if decision_implications:
                text += f"- **Decision Implication:** {decision_implications[0]}\n"
            
            text += "\n"
        
        # Enhanced Anticipation Factors
        if anticipation_factors:
            text += "###  Anticipation Factors Analysis\n"
            
            # Sort anticipation factors by value
            sorted_factors = sorted(anticipation_factors.items(), key=lambda x: x[1], reverse=True)
            
            for factor, value in sorted_factors[:5]:  # Top 5 factors
                factor_name = factor.replace('_', ' ').title()
                confidence_level = 'High' if value > 0.7 else 'Medium' if value > 0.5 else 'Low'
                text += f"- **{factor_name}:** {confidence_level} ({value:.2f})\n"
            
            # Overall anticipation assessment
            avg_anticipation = np.mean(list(anticipation_factors.values())) if anticipation_factors else 0
            if avg_anticipation > 0.6:
                anticipation_outlook = "Strong anticipation signals suggest significant moves ahead"
            elif avg_anticipation > 0.4:
                anticipation_outlook = "Moderate anticipation factors indicate potential opportunities"
            else:
                anticipation_outlook = "Low anticipation signals suggest stable conditions"
            
            text += f"- **Overall Anticipation Outlook:** {anticipation_outlook}\n\n"
        
        # Performance metrics
        if metrics:
            text += "###  Performance Metrics\n"
            
            if metrics.get('weekly_change_pct'):
                text += f"- **Weekly:** {metrics['weekly_change_pct']:+.2f}%\n"
            if metrics.get('monthly_change_pct'):
                text += f"- **Monthly:** {metrics['monthly_change_pct']:+.2f}%\n"
            if metrics.get('volatility_20d'):
                text += f"- **Volatility (20d):** {metrics['volatility_20d']:.2f}%\n"
            if metrics.get('volume_ratio'):
                vol_status = "High" if metrics['volume_ratio'] > 1.5 else "Normal" if metrics['volume_ratio'] > 0.7 else "Low"
                text += f"- **Volume:** {vol_status} ({metrics['volume_ratio']:.2f}x average)\n"
            
            text += "\n"
        
        # Technical analysis
        if technical:
            text += "### [SCAN] Technical Analysis\n"
            
            trend = technical.get('trend', 'NEUTRAL')
            rsi_signal = technical.get('rsi_signal', 'NEUTRAL')
            ema_signal = technical.get('ema_signal', 'NEUTRAL')
            
            text += f"- **Trend:** {trend}\n"
            text += f"- **RSI Signal:** {rsi_signal}\n"
            text += f"- **EMA Signal:** {ema_signal}\n"
            
            # Support/Resistance
            sr = technical.get('support_resistance', {})
            if sr:
                text += f"- **Support:** ${sr.get('support', 0):.2f}\n"
                text += f"- **Resistance:** ${sr.get('resistance', 0):.2f}\n"
                
                position = sr.get('current_position', 0.5)
                if position > 0.8:
                    text += "- **Position:** Near resistance (potential reversal zone)\n"
                elif position < 0.2:
                    text += "- **Position:** Near support (potential bounce zone)\n"
                else:
                    text += f"- **Position:** {position*100:.0f}% of range\n"
            
            text += "\n"
        
        # Risk analysis (if requested)
        if 'risk' in intents and 'risk' in analysis:
            risk = analysis['risk']
            text += "###  Risk Assessment\n"
            
            if risk.get('volatility_annual'):
                vol = risk['volatility_annual']
                vol_rating = "Very High" if vol > 40 else "High" if vol > 30 else "Moderate" if vol > 20 else "Low"
                text += f"- **Volatility:** {vol:.1f}% ({vol_rating})\n"
            
            if risk.get('max_drawdown'):
                text += f"- **Max Drawdown:** {risk['max_drawdown']:.2f}%\n"
            
            if risk.get('sharpe_ratio'):
                sharpe = risk['sharpe_ratio']
                sharpe_rating = "Excellent" if sharpe > 2 else "Good" if sharpe > 1 else "Fair" if sharpe > 0.5 else "Poor"
                text += f"- **Sharpe Ratio:** {sharpe:.2f} ({sharpe_rating})\n"
            
            text += "\n"
        
        # ML Model insights
        signal_factors = prediction.get('signal_factors', [])
        if signal_factors:
            text += "###  AI Model Insights\n"
            for factor in signal_factors[:5]:  # Top 5 factors
                if isinstance(factor, (list, tuple)) and len(factor) >= 3:
                    text += f"- **{factor[0]}:** {factor[1]} → {factor[2]}\n"
            text += "\n"
        
        # Enhanced Trading Implications with Anticipation Factors
        text += "###  Enhanced Trading Implications\n"
        
        # Factor in all analysis components
        total_bullish_signals = 0
        total_bearish_signals = 0
        total_weight = 0
        
        # ML signal weight
        ml_weight = 0.3
        if signal == 'BULLISH':
            total_bullish_signals += confidence/100 * ml_weight
        elif signal == 'BEARISH':
            total_bearish_signals += confidence/100 * ml_weight
        total_weight += ml_weight
        
        # News sentiment weight
        if advanced_news and not advanced_news.get('error'):
            news_weight = 0.25
            news_sentiment_score = advanced_news.get('overall_sentiment', 0)
            if news_sentiment_score > 0.2:
                total_bullish_signals += abs(news_sentiment_score) * news_weight
            elif news_sentiment_score < -0.2:
                total_bearish_signals += abs(news_sentiment_score) * news_weight
            total_weight += news_weight
        
        # Cross-sector weight
        if cross_sector and not cross_sector.get('error'):
            cross_sector_weight = 0.2
            rotation_impact = cross_sector.get('sector_rotation_impact', {})
            if rotation_impact.get('rotation_signal') == 'BULLISH':
                total_bullish_signals += 0.7 * cross_sector_weight
            elif rotation_impact.get('rotation_signal') == 'BEARISH':
                total_bearish_signals += 0.7 * cross_sector_weight
            total_weight += cross_sector_weight
        
        # Anticipation factors weight
        if anticipation_factors:
            anticipation_weight = 0.25
            avg_anticipation = np.mean(list(anticipation_factors.values()))
            
            # Determine if anticipation factors are bullish or bearish
            bullish_factors = ['high_confidence_signal', 'strong_trend_continuation', 'favorable_carry_trade', 'low_iv_expansion']
            bearish_factors = ['high_iv_mean_reversion', 'supply_tightness', 'high_volatility_regime']
            
            anticipation_bullish = sum(anticipation_factors.get(factor, 0) for factor in bullish_factors)
            anticipation_bearish = sum(anticipation_factors.get(factor, 0) for factor in bearish_factors)
            
            if anticipation_bullish > anticipation_bearish:
                total_bullish_signals += anticipation_bullish * anticipation_weight
            else:
                total_bearish_signals += anticipation_bearish * anticipation_weight
            total_weight += anticipation_weight
        
        # Calculate combined signal
        if total_weight > 0:
            combined_bullish_prob = total_bullish_signals / total_weight
            combined_bearish_prob = total_bearish_signals / total_weight
            
            if combined_bullish_prob > combined_bearish_prob and combined_bullish_prob > 0.6:
                text += f"- **Enhanced Signal:** STRONG BULLISH (Combined confidence: {combined_bullish_prob*100:.0f}%)\n"
                text += "- **Strategy:** Aggressive long positioning with multi-timeframe confirmation\n"
                text += "- **Risk Management:** Standard stops with anticipation factor monitoring\n"
                text += "- **Catalyst Watch:** Monitor news flow and cross-sector developments\n"
            elif combined_bearish_prob > combined_bullish_prob and combined_bearish_prob > 0.6:
                text += f"- **Enhanced Signal:** STRONG BEARISH (Combined confidence: {combined_bearish_prob*100:.0f}%)\n"
                text += "- **Strategy:** Defensive positioning or short opportunities\n"
                text += "- **Risk Management:** Tight stops with volatility adjustments\n"
                text += "- **Catalyst Watch:** Monitor for sentiment reversal signals\n"
            else:
                text += "- **Enhanced Signal:** MIXED - Multiple analysis layers show conflicting signals\n"
                text += "- **Strategy:** Range-bound trading or wait for clearer alignment\n"
                text += "- **Risk Management:** Very tight stops due to uncertainty\n"
                text += "- **Catalyst Watch:** Look for breaking news or technical breakouts\n"
        else:
            # Fall back to ML-only recommendations
            if signal == 'BULLISH':
                text += f"- **Long Position:** Favorable with {confidence:.0f}% confidence\n"
                text += "- **Strategy:** Consider buying on dips to support levels\n"
                text += "- **Risk Management:** Set stop-loss below recent support\n"
            elif signal == 'BEARISH':
                text += f"- **Short Position:** Favorable with {confidence:.0f}% confidence\n"
                text += "- **Strategy:** Consider shorting on rallies to resistance\n"
                text += "- **Risk Management:** Set stop-loss above recent resistance\n"
            else:
                text += "- **Position:** Neutral - wait for clearer signals\n"
                text += "- **Strategy:** Range-bound trading or await breakout\n"
                text += "- **Risk Management:** Tight stops due to uncertainty\n"
        
        return text
    
    def _generate_comparative_insights(self, symbol_analyses: Dict[str, Any]) -> str:
        """Generate comparative insights across multiple symbols."""
        
        text = "##  Comparative Analysis\n\n"
        
        # Compare signals
        bullish_symbols = []
        bearish_symbols = []
        neutral_symbols = []
        
        for symbol, analysis in symbol_analyses.items():
            signal = analysis.get('prediction', {}).get('signal', 'NEUTRAL')
            if signal == 'BULLISH':
                bullish_symbols.append(symbol)
            elif signal == 'BEARISH':
                bearish_symbols.append(symbol)
            else:
                neutral_symbols.append(symbol)
        
        if bullish_symbols:
            text += f"** Bullish Signals:** {', '.join(bullish_symbols)}\n"
        if bearish_symbols:
            text += f"** Bearish Signals:** {', '.join(bearish_symbols)}\n"
        if neutral_symbols:
            text += f"**[NOTE] Neutral Signals:** {', '.join(neutral_symbols)}\n"
        
        text += "\n"
        
        # Performance comparison
        text += "###  Performance Comparison\n"
        
        performance_data = []
        for symbol, analysis in symbol_analyses.items():
            metrics = analysis.get('metrics', {})
            performance_data.append({
                'symbol': symbol,
                'daily': metrics.get('daily_change_pct', 0),
                'weekly': metrics.get('weekly_change_pct', 0),
                'monthly': metrics.get('monthly_change_pct', 0)
            })
        
        # Sort by daily performance
        performance_data.sort(key=lambda x: x['daily'], reverse=True)
        
        text += "**Daily Performance Ranking:**\n"
        for i, data in enumerate(performance_data, 1):
            text += f"{i}. {data['symbol']}: {data['daily']:+.2f}%\n"
        
        text += "\n"
        
        return text
    
    def _generate_trading_recommendations(self, symbol_analyses: Dict[str, Any], 
                                        intents: List[str], market_context: Dict[str, Any]) -> str:
        """Generate overall trading recommendations."""
        
        text = "##  Overall Trading Recommendations\n\n"
        
        # Market environment consideration
        vol_regime = market_context.get('volatility_regime', 'Unknown')
        risk_mode = market_context.get('risk_mode', 'Unknown')
        
        text += f"###  Market Environment Considerations\n"
        text += f"Given the current **{vol_regime}** volatility regime and **{risk_mode}** risk sentiment:\n\n"
        
        if 'high' in vol_regime.lower():
            text += "- **Position Sizing:** Reduce position sizes due to high volatility\n"
            text += "- **Stop Losses:** Use wider stops to avoid whipsaws\n"
            text += "- **Time Horizon:** Consider shorter holding periods\n"
        elif 'low' in vol_regime.lower():
            text += "- **Position Sizing:** Normal to slightly larger positions acceptable\n"
            text += "- **Stop Losses:** Tighter stops can be used effectively\n"
            text += "- **Time Horizon:** Longer holding periods may be suitable\n"
        
        if 'risk off' in risk_mode.lower():
            text += "- **Asset Selection:** Favor defensive assets and quality names\n"
            text += "- **Diversification:** Increase diversification across asset classes\n"
        elif 'risk on' in risk_mode.lower():
            text += "- **Asset Selection:** Growth and cyclical assets may outperform\n"
            text += "- **Concentration:** Moderate concentration in high-conviction ideas acceptable\n"
        
        text += "\n"
        
        # Symbol-specific recommendations
        text += "###  Symbol-Specific Actions\n"
        
        for symbol, analysis in symbol_analyses.items():
            prediction = analysis.get('prediction', {})
            signal = prediction.get('signal', 'NEUTRAL')
            confidence = prediction.get('confidence', 0.5) * 100
            
            if signal == 'BULLISH' and confidence > 60:
                text += f"**{symbol}:** Strong buy candidate - consider accumulating on dips\n"
            elif signal == 'BULLISH':
                text += f"**{symbol}:** Moderate buy - small position with tight risk management\n"
            elif signal == 'BEARISH' and confidence > 60:
                text += f"**{symbol}:** Strong sell candidate - consider reducing exposure\n"
            elif signal == 'BEARISH':
                text += f"**{symbol}:** Moderate sell - consider profit-taking or hedging\n"
            else:
                text += f"**{symbol}:** Hold/Watch - await clearer directional signals\n"
        
        text += "\n"
        
        # Risk management
        text += "###  Risk Management Guidelines\n"
        text += "- **Portfolio Allocation:** Don't risk more than 2-3% per position\n"
        text += "- **Correlation:** Monitor correlation between positions\n"
        text += "- **Time Diversification:** Stagger entries over time\n"
        text += "- **Review Frequency:** Reassess positions daily in volatile markets\n\n"
        
        # Disclaimer
        text += "---\n"
        text += "* This analysis is for informational purposes only and should not be considered as financial advice. Always conduct your own research and consider your risk tolerance before making investment decisions.*"
        
        return text
    
    def _create_analysis_summary(self, symbol_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Create a structured summary of the analysis."""
        
        summary = {
            'total_symbols': len(symbol_analyses),
            'bullish_count': 0,
            'bearish_count': 0,
            'neutral_count': 0,
            'avg_confidence': 0,
            'high_confidence_signals': [],
            'risk_levels': {},
            'top_performers': [],
            'bottom_performers': []
        }
        
        confidences = []
        performance_data = []
        
        for symbol, analysis in symbol_analyses.items():
            prediction = analysis.get('prediction', {})
            metrics = analysis.get('metrics', {})
            
            signal = prediction.get('signal', 'NEUTRAL')
            confidence = prediction.get('confidence', 0.5)
            
            # Count signals
            if signal == 'BULLISH':
                summary['bullish_count'] += 1
            elif signal == 'BEARISH':
                summary['bearish_count'] += 1
            else:
                summary['neutral_count'] += 1
            
            confidences.append(confidence)
            
            # High confidence signals
            if confidence > 0.7:
                summary['high_confidence_signals'].append({
                    'symbol': symbol,
                    'signal': signal,
                    'confidence': confidence
                })
            
            # Performance tracking
            daily_change = metrics.get('daily_change_pct', 0)
            performance_data.append({
                'symbol': symbol,
                'daily_change': daily_change
            })
            
            # Risk levels
            risk = analysis.get('risk', {})
            if risk.get('volatility_annual'):
                vol = risk['volatility_annual']
                if vol > 40:
                    risk_level = 'Very High'
                elif vol > 30:
                    risk_level = 'High'
                elif vol > 20:
                    risk_level = 'Moderate'
                else:
                    risk_level = 'Low'
                
                summary['risk_levels'][symbol] = risk_level
        
        # Calculate averages
        if confidences:
            summary['avg_confidence'] = np.mean(confidences)
        
        # Sort performance
        performance_data.sort(key=lambda x: x['daily_change'], reverse=True)
        summary['top_performers'] = performance_data[:3]
        summary['bottom_performers'] = performance_data[-3:]
        
        return summary
    
    def _generate_charts(self, symbol_analyses: Dict[str, Any], intents: List[str]) -> List[Dict[str, Any]]:
        """Generate appropriate charts based on analysis and intents."""
        
        charts = []
        
        for symbol, analysis in symbol_analyses.items():
            df = analysis.get('data')
            if df is None or df.empty:
                continue
            
            # Always create price chart for analysis
            if 'chart' in intents or 'analysis' in intents or 'prediction' in intents:
                chart_style = 'comprehensive' if len(symbol_analyses) == 1 else 'standard'
                price_chart = self._create_advanced_price_chart(df, symbol, True, chart_style)
                if price_chart:
                    charts.append({
                        'type': 'price_analysis',
                        'symbol': symbol,
                        'figure': price_chart,
                        'title': f'{symbol} Advanced Technical Analysis'
                    })
            
            # Prediction visualization
            if 'prediction' in intents:
                prediction = analysis.get('prediction', {})
                pred_chart = self._create_prediction_chart(df, symbol, prediction)
                if pred_chart:
                    charts.append({
                        'type': 'prediction',
                        'symbol': symbol,
                        'figure': pred_chart,
                        'title': f'{symbol} ML Prediction Analysis'
                    })
            
            # Risk analysis charts
            if 'risk' in intents:
                vol_chart = self._create_volatility_chart(df, symbol)
                if vol_chart:
                    charts.append({
                        'type': 'risk_analysis',
                        'symbol': symbol,
                        'figure': vol_chart,
                        'title': f'{symbol} Risk & Volatility Analysis'
                    })
        
        return charts
    
    def _generate_contextual_suggestions(self, symbol_analyses: Dict[str, Any], intents: List[str]) -> List[str]:
        """Generate contextual follow-up suggestions."""
        
        suggestions = []
        symbols = list(symbol_analyses.keys())
        
        if len(symbols) == 1:
            symbol = symbols[0]
            suggestions.extend([
                f"Show me risk analysis for {symbol}",
                f"Compare {symbol} with similar assets",
                f"What are the key support and resistance levels for {symbol}?",
                f"Give me a detailed prediction for {symbol}",
            ])
        elif len(symbols) > 1:
            suggestions.extend([
                f"Compare performance of {' vs '.join(symbols[:2])}",
                f"Which is the better investment: {symbols[0]} or {symbols[1]}?",
                "Show me correlation analysis between these assets",
            ])
        
        # Add general suggestions based on intents
        if 'prediction' not in intents:
            suggestions.append("What's your prediction for the next week?")
        
        if 'risk' not in intents:
            suggestions.append("Show me risk analysis and volatility")
        
        if 'sector' not in intents:
            suggestions.append("How is the overall sector performing?")
        
        # Add market-wide suggestions
        suggestions.extend([
            "What's the current market sentiment?",
            "Analyze SPY for overall market direction",
            "Show me today's top movers",
            "What are the key economic events this week?"
        ])
        
        return suggestions[:6]  # Limit to 6 suggestions



    
    def _generate_octavian_guidance(self, query: str, intents: List[str], 
                                  timeframe_scope: TimeframeScope) -> str:
        """Generate helpful Octavian guidance when no symbols are found.

        Example assets are sourced dynamically from the live ticker universe
        (never a hardcoded list) so the guidance always reflects the current
        multi-asset coverage.
        """
        tf = timeframe_scope.value.replace('_', ' ')
        base_response = f"""#  **OCTAVIAN** - I'm here to help!

I'd be happy to provide advanced market analysis tailored to your **{tf} trading/investing** approach.

"""
        ex = self._get_asset_examples()
        stocks = ', '.join(ex['Stocks'])
        etfs = ', '.join(ex['ETFs'])
        crypto = ', '.join(ex['Crypto'])
        forex = ', '.join(ex['Forex'])
        futures = ', '.join(ex['Futures'])

        if 'prediction' in intents:
            return base_response + f"""##  AI Predictions Available For:

**Stocks:** {stocks}
**ETFs:** {etfs}
**Crypto:** {crypto}
**Forex:** {forex}
**Futures:** {futures}

### Example Queries:
- "Octavian, predict AAPL for {tf} trading"
- "What's your {tf} outlook for Bitcoin?"
- "Should I buy TSLA for {tf} holding?"
"""
        
        elif 'analysis' in intents:
            return base_response + f"""##  Comprehensive Analysis Available:

I provide **source-weighted news analysis**, **timeframe-specific insights**, and **AI vs market sentiment** comparisons.

### What I Analyze:
- **Technical Analysis** with {tf}-specific indicators
- **Source-Credibility Weighted News** (Bloomberg, Reuters, WSJ prioritized)
- **AI vs Market Sentiment** divergence analysis
- **Cross-Sector Correlations** and anticipation factors
- **Risk Assessment** tailored to your timeframe

### Example Queries:
- "Analyze AAPL with full Octavian intelligence"
- "Comprehensive {tf} analysis of NVDA"
- "Show me credibility-weighted news analysis for SPY"
"""
        
        else:
            return base_response + f"""##  What Makes Octavian Different:

###  **Source Credibility Weighting**
- Bloomberg, Reuters, WSJ articles get higher weight
- Social media and unverified sources get lower weight
- Time-decay factors for news relevance

###  **Timeframe-Aware Analysis**
- Currently optimized for: **{tf} strategies**
- Different indicators and factors for each timeframe
- Personalized based on your trading profile

###  **AI vs Market Sentiment**
- Compare AI model predictions with market sentiment
- Identify divergences for contrarian opportunities
- Confidence-weighted decision making

###  **Example Queries:**
- "Octavian, analyze AAPL with full intelligence"
- "Compare TSLA vs NIO with source-weighted analysis"
- "Bitcoin comprehensive analysis for {tf} trading"
- "Show me AI vs market sentiment for SPY"

**Just mention any stock symbol and I'll provide comprehensive Octavian analysis!**
"""
    
    def _get_contextual_suggestions(self, timeframe_scope: TimeframeScope) -> List[str]:
        """Get contextual suggestions based on timeframe."""
        base_suggestions = [
            f"Analyze SPY for {timeframe_scope.value.replace('_', ' ')} trading",
            f"Octavian comprehensive analysis of AAPL",
            f"Compare TSLA vs NIO with source weighting",
            f"Bitcoin analysis with AI vs market sentiment"
        ]
        
        if timeframe_scope == TimeframeScope.SCALPING:
            base_suggestions.extend([
                "Show me high-frequency trading opportunities",
                "Scalping analysis for NVDA with volume data"
            ])
        elif timeframe_scope == TimeframeScope.INTRADAY:
            base_suggestions.extend([
                "Intraday momentum analysis for QQQ",
                "Day trading setup for AMZN"
            ])
        elif timeframe_scope == TimeframeScope.SWING:
            base_suggestions.extend([
                "Swing trading opportunities in tech sector",
                "Weekly analysis of market leaders"
            ])
        elif timeframe_scope in [TimeframeScope.POSITION, TimeframeScope.INVESTMENT]:
            base_suggestions.extend([
                "Long-term investment analysis of growth stocks",
                "Position sizing recommendations for portfolio"
            ])
        
        return base_suggestions
    
    def _generate_enhanced_charts(self, symbol_analyses: Dict[str, Any], 
                                intents: List[str], timeframe_scope: TimeframeScope) -> List[Dict[str, Any]]:
        """Generate enhanced charts with timeframe context."""
        charts = []
        
        for symbol, analysis in symbol_analyses.items():
            df = analysis.get('data')
            if df is not None and not df.empty:
                # Main price chart with timeframe-specific indicators
                try:
                    chart_style = 'comprehensive' if 'comprehensive' in intents else 'standard'
                    price_chart = self._create_advanced_price_chart(df, symbol, True, chart_style)
                    
                    if price_chart:
                        charts.append({
                            'type': 'price_analysis',
                            'symbol': symbol,
                            'figure': price_chart,
                            'title': f'{symbol} - {timeframe_scope.value.replace("_", " ").title()} Analysis'
                        })
                except Exception as e:
                    print(f"Error creating price chart for {symbol}: {e}")
                
                # Prediction visualization
                if 'prediction' in intents:
                    try:
                        prediction = analysis.get('prediction', {})
                        pred_chart = self._create_prediction_chart(df, symbol, prediction)
                        
                        if pred_chart:
                            charts.append({
                                'type': 'prediction',
                                'symbol': symbol,
                                'figure': pred_chart,
                                'title': f'{symbol} - AI Prediction Visualization'
                            })
                    except Exception as e:
                        print(f"Error creating prediction chart for {symbol}: {e}")
        
        return charts
    
    def _generate_enhanced_suggestions(self, symbol_analyses: Dict[str, Any], 
                                     intents: List[str], timeframe_scope: TimeframeScope,
                                     trader_context: Dict[str, Any]) -> List[str]:
        """Generate enhanced contextual suggestions."""
        suggestions = []
        
        # Symbol-specific suggestions
        for symbol in symbol_analyses.keys():
            suggestions.extend([
                f"Show me risk analysis for {symbol}",
                f"Compare {symbol} with sector peers",
                f"{symbol} options analysis and strategies"
            ])
        
        # Timeframe-specific suggestions
        if timeframe_scope == TimeframeScope.SCALPING:
            suggestions.extend([
                "Show me high-volume breakout opportunities",
                "Level 2 data analysis for scalping",
                "Micro-structure analysis for entries"
            ])
        elif timeframe_scope == TimeframeScope.SWING:
            suggestions.extend([
                "Weekly sector rotation analysis",
                "Earnings calendar impact analysis",
                "Technical breakout patterns this week"
            ])
        elif timeframe_scope in [TimeframeScope.POSITION, TimeframeScope.INVESTMENT]:
            suggestions.extend([
                "Long-term fundamental analysis",
                "Dividend growth stock analysis",
                "Portfolio diversification recommendations"
            ])
        
        # Market-wide suggestions
        suggestions.extend([
            "Overall market sentiment analysis",
            "Cross-sector correlation heatmap",
            "Volatility regime analysis",
            "AI vs market divergence opportunities"
        ])
        
        return suggestions[:8]  # Limit to 8 suggestions
    
    def _create_enhanced_analysis_summary(self, symbol_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Create enhanced analysis summary with credibility insights."""
        summary = {
            'total_symbols': len(symbol_analyses),
            'bullish_count': 0,
            'bearish_count': 0,
            'neutral_count': 0,
            'avg_confidence': 0,
            'high_confidence_signals': [],
            'credibility_insights': {},
            'timeframe_insights': {}
        }
        
        confidences = []
        
        for symbol, analysis in symbol_analyses.items():
            prediction = analysis.get('prediction', {})
            signal = prediction.get('signal', 'NEUTRAL')
            confidence = prediction.get('confidence', 0.5)
            
            confidences.append(confidence)
            
            if signal == 'BULLISH':
                summary['bullish_count'] += 1
            elif signal == 'BEARISH':
                summary['bearish_count'] += 1
            else:
                summary['neutral_count'] += 1
            
            if confidence > 0.7:
                summary['high_confidence_signals'].append({
                    'symbol': symbol,
                    'signal': signal,
                    'confidence': confidence
                })
            
            # Credibility insights
            credibility_news = analysis.get('credibility_weighted_news', {})
            if credibility_news and not credibility_news.get('error'):
                summary['credibility_insights'][symbol] = {
                    'weighted_sentiment': credibility_news.get('weighted_sentiment', 0),
                    'total_weight': credibility_news.get('total_weight', 0),
                    'article_count': credibility_news.get('item_count', 0)
                }
        
        if confidences:
            summary['avg_confidence'] = np.mean(confidences)
        
        return summary
    
    def _extract_credibility_insights(self, symbol_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key credibility-weighted insights."""
        insights = {
            'total_premium_articles': 0,
            'total_articles': 0,
            'premium_ratio': 0,
            'weighted_sentiment_average': 0,
            'high_credibility_symbols': [],
            'low_credibility_symbols': []
        }
        
        weighted_sentiments = []
        
        for symbol, analysis in symbol_analyses.items():
            credibility_news = analysis.get('credibility_weighted_news', {})
            if credibility_news and not credibility_news.get('error'):
                article_count = credibility_news.get('item_count', 0)
                insights['total_articles'] += article_count
                
                source_breakdown = credibility_news.get('source_breakdown', {})
                premium_count = source_breakdown.get('tier_1_premium', 0)
                insights['total_premium_articles'] += premium_count
                
                weighted_sentiment = credibility_news.get('weighted_sentiment', 0)
                weighted_sentiments.append(weighted_sentiment)
                
                # Categorize symbols by credibility
                if article_count > 0:
                    premium_ratio = premium_count / article_count
                    if premium_ratio > 0.3:
                        insights['high_credibility_symbols'].append(symbol)
                    elif premium_ratio < 0.1:
                        insights['low_credibility_symbols'].append(symbol)
        
        if insights['total_articles'] > 0:
            insights['premium_ratio'] = insights['total_premium_articles'] / insights['total_articles']
        
        if weighted_sentiments:
            insights['weighted_sentiment_average'] = np.mean(weighted_sentiments)
        
        return insights
    
    # Keep existing methods from original chatbot
    def _get_real_time_data(self, symbol: str, period: str = '1y') -> Tuple[Optional[pd.DataFrame], str]:
        """Fetch real-time market data with caching and error handling."""
        start_time = time.time()
        
        try:
            # Check cache first
            cached_data = self.db_manager.get_market_data(symbol, days=365)
            
            # Determine if we need fresh data (older than 1 hour)
            need_fresh_data = True
            if not cached_data.empty:
                latest_timestamp = cached_data.index[-1]
                time_diff = datetime.now() - latest_timestamp.to_pydatetime()
                if time_diff.total_seconds() < 3600:  # Less than 1 hour old
                    need_fresh_data = False
            
            df = None
            asset_type = 'Unknown'
            
            if need_fresh_data:
                # Determine asset type and fetch data
                if '/' in symbol or '_' in symbol or any(curr in symbol for curr in ['EUR', 'GBP', 'JPY', 'CHF']):
                    # FX pair
                    fx_symbol = symbol.replace('/', '_').replace('-', '_')
                    df = get_fx(fx_symbol)
                    asset_type = 'FX'
                elif '=F' in symbol or symbol in ['ES', 'NQ', 'CL', 'GC', 'SI']:
                    # Futures
                    df = get_futures_proxy(symbol, period=period)
                    asset_type = 'Futures'
                else:
                    # Stock or crypto
                    df = get_stock(symbol, period=period)
                    asset_type = 'Crypto' if 'USD' in symbol and '-' in symbol else 'Stock'
                
                # Store in database if successful
                if df is not None and not df.empty:
                    self.db_manager.store_market_data(symbol, df, asset_type.lower())
                    
                    # Store technical indicators
                    df_with_indicators = add_indicators(df.copy())
                    if not df_with_indicators.empty:
                        self.db_manager.store_technical_indicators(symbol, df_with_indicators)
            else:
                # Use cached data
                df = cached_data
                # Determine asset type from symbol
                if '/' in symbol or '_' in symbol:
                    asset_type = 'FX'
                elif '=F' in symbol:
                    asset_type = 'Futures'
                elif '-USD' in symbol:
                    asset_type = 'Crypto'
                else:
                    asset_type = 'Stock'
            
            # Log performance metrics
            response_time = (time.time() - start_time) * 1000
            self.db_manager.log_query_analytics(
                'data_fetch', symbol, int(response_time), 
                df is not None and not df.empty
            )
            
            return df, asset_type
            
        except Exception as e:
            error_msg = f"Error fetching data for {symbol}: {e}"
            print(error_msg)
            
            # Log error
            response_time = (time.time() - start_time) * 1000
            self.db_manager.log_query_analytics(
                'data_fetch', symbol, int(response_time), False, str(e)
            )
            
            return None, 'Unknown'
    
    def _create_advanced_price_chart(self, df: pd.DataFrame, symbol: str, 
                                     show_indicators: bool = True, 
                                     chart_style: str = 'comprehensive') -> go.Figure:
        """Create advanced interactive price chart with multiple analysis layers."""
        if df.empty:
            return None
        
        # Use more data for better analysis
        df_display = df.tail(200).copy()
        df_with_ind = add_indicators(df_display.copy())
        
        if chart_style == 'comprehensive':
            # 4-panel comprehensive chart
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(
                    f'{symbol} - Octavian Advanced Analysis', 
                    'Momentum Indicators (RSI & MACD)', 
                    'Volume Profile & Analysis',
                    'Volatility & Market Structure'
                ),
                row_heights=[0.4, 0.25, 0.2, 0.15]
            )
        else:
            # Standard 3-panel chart
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=(f'{symbol} Price Action', 'RSI', 'Volume'),
                row_heights=[0.5, 0.25, 0.25]
            )
        
        # Enhanced Candlestick with Octavian colors (gold theme)
        fig.add_trace(
            go.Candlestick(
                x=df_display.index,
                open=df_display['Open'],
                high=df_display['High'],
                low=df_display['Low'],
                close=df_display['Close'],
                name='Price',
                increasing_line_color='#FFD700',  # Gold
                decreasing_line_color='#B8860B',  # Dark gold
                increasing_fillcolor='#FFD700',
                decreasing_fillcolor='#B8860B'
            ),
            row=1, col=1
        )
        
        if show_indicators and not df_with_ind.empty:
            # Multiple EMAs with Octavian theme
            ema_configs = [
                ('ema20', '#DAA520', 'EMA 20', 2),  # Goldenrod
                ('ema50', '#CD853F', 'EMA 50', 2),  # Peru
            ]
            
            # Add 200 EMA if we have enough data
            if len(df_display) >= 200:
                df_with_ind['ema200'] = df_display['Close'].ewm(span=200, adjust=False).mean()
                ema_configs.append(('ema200', '#8B4513', 'EMA 200', 3))  # Saddle brown
            
            for ema_col, color, name, width in ema_configs:
                if ema_col in df_with_ind.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df_with_ind.index,
                            y=df_with_ind[ema_col],
                            mode='lines',
                            name=name,
                            line=dict(color=color, width=width),
                            opacity=0.8
                        ),
                        row=1, col=1
                    )
            
            # Enhanced RSI with Octavian styling
            if 'rsi' in df_with_ind.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_with_ind.index,
                        y=df_with_ind['rsi'],
                        mode='lines',
                        name='RSI',
                        line=dict(color='#DAA520', width=2),  # Goldenrod
                        fill='tozeroy',
                        fillcolor='rgba(218, 165, 32, 0.2)'
                    ),
                    row=2, col=1
                )
                
                # RSI levels with Octavian theme
                rsi_levels = [
                    (80, "#FF6B6B", "Extreme Overbought"),
                    (70, "#FFA500", "Overbought"),
                    (50, "#DAA520", "Neutral"),
                    (30, "#32CD32", "Oversold"),
                    (20, "#228B22", "Extreme Oversold")
                ]
                
                for level, color, label in rsi_levels:
                    fig.add_hline(
                        y=level, 
                        line_dash="dash" if level in [70, 30] else "dot", 
                        line_color=color, 
                        opacity=0.7 if level in [70, 30] else 0.4,
                        annotation_text=label if level in [80, 20] else "",
                        row=2, col=1
                    )
        
        # Enhanced Volume Analysis with Octavian colors
        if 'Volume' in df_display.columns:
            colors = []
            for i in range(len(df_display)):
                if df_display['Close'].iloc[i] > df_display['Open'].iloc[i]:
                    colors.append('#FFD700')  # Gold for up days
                else:
                    colors.append('#B8860B')  # Dark gold for down days
            
            volume_row = 3 if chart_style != 'comprehensive' else 3
            
            fig.add_trace(
                go.Bar(
                    x=df_display.index,
                    y=df_display['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=volume_row, col=1
            )
        
        # Enhanced layout with Octavian branding
        fig.update_layout(
            height=900 if chart_style == 'comprehensive' else 700,
            template='plotly_dark',
            showlegend=True,
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=1.02, 
                xanchor="right", 
                x=1,
                bgcolor="rgba(0,0,0,0.5)"
            ),
            xaxis_rangeslider_visible=False,
            title={
                'text': f' {symbol} - Octavian Advanced Analysis',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#DAA520'}
            },
            hovermode='x unified',
            paper_bgcolor='rgba(0,0,0,0.9)',
            plot_bgcolor='rgba(0,0,0,0.9)'
        )
        
        # Update axes with Octavian styling
        fig.update_xaxes(title_text="Date", row=-1, col=1, color='#DAA520')
        fig.update_yaxes(title_text="Price ($)", row=1, col=1, color='#DAA520')
        fig.update_yaxes(title_text="RSI" if chart_style != 'comprehensive' else "Momentum", row=2, col=1, color='#DAA520')
        fig.update_yaxes(title_text="Volume", row=3, col=1, color='#DAA520')
        
        if chart_style == 'comprehensive':
            fig.update_yaxes(title_text="Volatility (%)", row=4, col=1, color='#DAA520')
        
        return fig
    
    def _create_prediction_chart(self, df: pd.DataFrame, symbol: str, 
                                  prediction: Dict) -> go.Figure:
        """Create a chart showing prediction visualization with Octavian styling."""
        if df.empty:
            return None
        
        df_recent = df.tail(60).copy()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'{symbol} Recent Price Action',
                'Signal Strength',
                'Model Confidence',
                'Price Position'
            ),
            specs=[[{"type": "scatter"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]]
        )
        
        # Price chart with trend line (Octavian colors)
        fig.add_trace(
            go.Scatter(
                x=df_recent.index,
                y=df_recent['Close'],
                mode='lines',
                name='Price',
                line=dict(color='#DAA520', width=2)  # Goldenrod
            ),
            row=1, col=1
        )
        
        # Add trend line
        x_numeric = np.arange(len(df_recent))
        z = np.polyfit(x_numeric, df_recent['Close'].values, 1)
        p = np.poly1d(z)
        trend_color = '#FFD700' if z[0] > 0 else '#B8860B'  # Gold theme
        
        fig.add_trace(
            go.Scatter(
                x=df_recent.index,
                y=p(x_numeric),
                mode='lines',
                name='Trend',
                line=dict(color=trend_color, width=2, dash='dash')
            ),
            row=1, col=1
        )
        
        # Signal strength gauge with Octavian colors
        signal = prediction.get('signal', 'NEUTRAL')
        bullish_prob = prediction.get('bullish_prob', 0.5) * 100
        
        gauge_color = '#FFD700' if signal == 'BULLISH' else '#B8860B' if signal == 'BEARISH' else '#DAA520'
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=bullish_prob,
                title={'text': f"Signal: {signal}"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': gauge_color},
                    'steps': [
                        {'range': [0, 40], 'color': '#2F1B14'},
                        {'range': [40, 60], 'color': '#8B4513'},
                        {'range': [60, 100], 'color': '#DAA520'}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': bullish_prob
                    }
                }
            ),
            row=1, col=2
        )
        
        # Confidence indicator
        confidence = prediction.get('confidence', 0.5) * 100
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=confidence,
                title={'text': "Confidence %"},
                delta={'reference': 50, 'relative': False},
                number={'suffix': '%', 'font': {'color': '#DAA520'}}
            ),
            row=2, col=1
        )
        
        # Price position indicator
        high_20 = df_recent['High'].max()
        low_20 = df_recent['Low'].min()
        current = df_recent['Close'].iloc[-1]
        position = (current - low_20) / (high_20 - low_20) * 100 if high_20 > low_20 else 50
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=position,
                title={'text': "Price Position (20d Range)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': '#DAA520'},
                    'steps': [
                        {'range': [0, 20], 'color': '#2F1B14'},
                        {'range': [20, 80], 'color': '#8B4513'},
                        {'range': [80, 100], 'color': '#CD853F'}
                    ]
                }
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            template='plotly_dark',
            showlegend=True,
            title=f' {symbol} Octavian ML Prediction',
            paper_bgcolor='rgba(0,0,0,0.9)',
            plot_bgcolor='rgba(0,0,0,0.9)'
        )
        
        return fig


def show_octavian_chatbot():
    """Enhanced Streamlit interface for Octavian AI chatbot."""
    # Octavian header with branding
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%); border-radius: 10px; margin-bottom: 20px;">
        <h1 style="color: #DAA520; font-size: 3em; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
             OCTAVIAN
        </h1>
        <h3 style="color: #CD853F; margin: 10px 0; font-style: italic;">
            by APB - Advanced Market Intelligence
        </h3>
        <p style="color: #B8860B; margin: 0; font-size: 1.1em;">
            Powered by AI • Enhanced with Source Credibility • Timeframe-Aware Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show trader profile selection
    from trader_profile import show_trader_selection
    show_trader_selection()
    
    # Enhanced description
    st.markdown("""
    ###  **Next-Generation Market Intelligence**
    
    Octavian represents the pinnacle of AI-powered market analysis, integrating:
    
    ** Source Credibility Weighting:**
    - Bloomberg, Reuters, WSJ articles receive premium weighting
    - Social media and unverified sources appropriately discounted
    - Time-decay factors for news relevance
    
    ** Timeframe-Aware Analysis:**
    - Scalping (seconds to minutes)
    - Intraday (minutes to hours) 
    - Swing (days to weeks)
    - Position (weeks to months)
    - Investment (months to years)
    
    ** AI vs Market Sentiment:**
    - Compare AI predictions with market sentiment
    - Identify divergences for contrarian opportunities
    - Confidence-weighted decision making
    
    ** Cross-Asset Intelligence:**
    - Multi-asset correlation analysis
    - Sector rotation insights
    - Anticipation factor modeling
    
    ---
    """)
    
    # Initialize enhanced chatbot
    if 'octavian_chatbot' not in st.session_state:
        st.session_state.octavian_chatbot = OctavianEnhancedChatbot()
    
    if 'octavian_chat_history' not in st.session_state:
        st.session_state.octavian_chat_history = []
    
    if 'user_id' not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    
    # Enhanced sidebar
    with st.sidebar:
        st.markdown("###  Octavian Analytics")
        
        if st.session_state.octavian_chat_history:
            total_queries = len(st.session_state.octavian_chat_history)
            total_charts = sum(len(chat.get('response', {}).get('charts', [])) for chat in st.session_state.octavian_chat_history)
            avg_response_time = np.mean([chat.get('response', {}).get('response_time_ms', 0) for chat in st.session_state.octavian_chat_history])
            
            st.metric("Total Queries", total_queries)
            st.metric("Charts Generated", total_charts)
            st.metric("Avg Response Time", f"{avg_response_time:.0f}ms")
        
        st.markdown("###  Octavian Settings")
        
        # Analysis preferences
        analysis_depth = st.selectbox(
            "Analysis Depth",
            ["Comprehensive", "Standard", "Quick"],
            index=0,
            help="Choose the level of Octavian analysis detail"
        )
        
        # Credibility weighting
        credibility_weighting = st.checkbox(
            "Enhanced Source Credibility",
            value=True,
            help="Apply advanced source credibility weighting to news analysis"
        )
        
        # Timeframe override
        timeframe_override = st.selectbox(
            "Timeframe Override",
            ["Auto (Profile-Based)", "Scalping", "Intraday", "Swing", "Position", "Investment"],
            index=0,
            help="Override automatic timeframe detection"
        )
        
        # Chart display preferences
        st.markdown("####  Chart Display")
        
        # Price units
        price_unit = st.selectbox(
            "Price Display",
            ["Auto", "Dollars ($)", "Percent (%)", "Points"],
            index=0,
            help="Choose how to display price values on charts"
        )
        
        # Volume display
        volume_display = st.checkbox(
            "Show Volume",
            value=True,
            help="Display volume bars on price charts"
        )
        
        # Chart timeframe
        chart_timeframe = st.selectbox(
            "Chart Period",
            ["1 Month", "3 Months", "6 Months", "1 Year", "2 Years"],
            index=2,
            help="Default chart time period"
        )
        
        # Store chart settings in session state for use in chart generation
        st.session_state['chart_settings'] = {
            'price_unit': price_unit,
            'volume_display': volume_display,
            'chart_timeframe': chart_timeframe,
            'analysis_depth': analysis_depth
        }
        
        if st.button(" Clear Octavian History"):
            st.session_state.octavian_chat_history = []
            st.rerun()
    
    # Display chat history with enhanced formatting
    st.markdown("###  Octavian Intelligence Interface")
    
    for i, chat in enumerate(st.session_state.octavian_chat_history):
        with st.chat_message("user"):
            st.write(f"**Query #{i+1}:** {chat['query']}")
            
            # Enhanced metadata
            response = chat.get('response', {})
            metadata_parts = []
            
            if response.get('symbols'):
                metadata_parts.append(f" {', '.join(response['symbols'])}")
            if response.get('timeframe_context'):
                metadata_parts.append(f" {response['timeframe_context'].replace('_', ' ').title()}")
            if response.get('response_time_ms'):
                metadata_parts.append(f" {response['response_time_ms']}ms")
            
            if metadata_parts:
                st.caption(" | ".join(metadata_parts))
        
        with st.chat_message("assistant"):
            # Display Octavian response
            st.markdown(response.get('text', 'No response available'))
            
            # Display enhanced charts
            charts = response.get('charts', [])
            if charts:
                st.markdown(f"** Octavian Generated {len(charts)} Chart{'s' if len(charts) > 1 else ''}:**")
                
                for chart_data in charts:
                    if 'figure' in chart_data and chart_data['figure'] is not None:
                        st.plotly_chart(chart_data['figure'], width='stretch', 
                                      key=f"octavian_chart_{i}_{chart_data.get('type', 'unknown')}")
            
            # Show enhanced analysis summary
            if response.get('analysis_summary'):
                summary = response['analysis_summary']
                
                with st.expander(" Octavian Analysis Summary", expanded=False):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Symbols Analyzed", summary.get('total_symbols', 0))
                    with col2:
                        st.metric("Bullish Signals", summary.get('bullish_count', 0))
                    with col3:
                        st.metric("Bearish Signals", summary.get('bearish_count', 0))
                    with col4:
                        st.metric("Avg Confidence", f"{summary.get('avg_confidence', 0)*100:.0f}%")
                    
                    # Credibility insights
                    credibility_insights = response.get('credibility_weighted_insights', {})
                    if credibility_insights:
                        st.markdown("** Source Credibility Insights:**")
                        premium_ratio = credibility_insights.get('premium_ratio', 0)
                        st.write(f"- Premium Source Coverage: {premium_ratio:.1%}")
                        st.write(f"- Total Articles: {credibility_insights.get('total_articles', 0)}")
            
            # Show enhanced suggestions
            suggestions = response.get('suggestions', [])
            if suggestions:
                st.markdown("** Octavian Suggestions:**")
                
                cols = st.columns(min(len(suggestions[:4]), 2))
                for idx, suggestion in enumerate(suggestions[:4]):
                    with cols[idx % 2]:
                        if st.button(f" {suggestion}", key=f"octavian_suggestion_{i}_{idx}"):
                            st.session_state.suggested_query = suggestion
    
    # Enhanced chat input
    user_query = st.chat_input("Ask Octavian anything about the markets...")
    
    # Handle suggested query
    if st.session_state.get('suggested_query'):
        user_query = st.session_state.suggested_query
        st.session_state.suggested_query = None
    
    # Process new query with enhanced features
    if user_query:
        with st.chat_message("user"):
            st.write(f"**Query #{len(st.session_state.octavian_chat_history)+1}:** {user_query}")
        
        with st.chat_message("assistant"):
            with st.spinner(" Octavian is analyzing markets with advanced AI..."):
                try:
                    # Process with enhanced chatbot
                    response = asyncio.run(
                        st.session_state.octavian_chatbot.process_enhanced_query(
                            user_query, 
                            user_id=st.session_state.user_id
                        )
                    )
                    
                    # Safely display response with error handling
                    if not isinstance(response, dict):
                        st.error("Unable to generate analysis. Please try a different query.")
                        response = {'text': 'Analysis unavailable. Please try rephrasing your question.', 'charts': [], 'analysis_summary': {}, 'suggestions': []}
                    else:
                        # Ensure all expected keys exist
                        response.setdefault('text', 'Analysis completed but no detailed response available.')
                        response.setdefault('charts', [])
                        response.setdefault('analysis_summary', {})
                        response.setdefault('suggestions', [])
                        
                except Exception as e:
                    st.error(f"Octavian encountered an error: {str(e)[:100]}")
                    response = {'text': f'Analysis failed: {str(e)[:200]}. Please try rephrasing your question.', 'charts': [], 'analysis_summary': {}, 'suggestions': []}
                
                # Display enhanced response (runs for BOTH success and error cases)
                st.markdown(response.get('text', 'No response generated'))
                
                # Show AI Reasoning expandable (NEW FEATURE)
                if response.get('show_reasoning_available', False) and response.get('raw_analysis'):
                    with st.expander(" Show AI Reasoning", expanded=False):
                        st.markdown("### Detailed Analysis Breakdown")
                        
                        raw_analysis = response.get('raw_analysis', {})
                        
                        # Symbol analyses
                        if 'symbol_analyses' in raw_analysis:
                            st.markdown("#### Symbol-by-Symbol Analysis")
                            for symbol, analysis in raw_analysis['symbol_analyses'].items():
                                with st.expander(f" {symbol} - Detailed Breakdown"):
                                    
                                    # Prediction details
                                    if 'prediction' in analysis:
                                        st.markdown("**Prediction:**")
                                        pred = analysis['prediction']
                                        col_pred1, col_pred2, col_pred3 = st.columns(3)
                                        with col_pred1:
                                            st.metric("Signal", pred.get('signal', 'N/A'))
                                        with col_pred2:
                                            st.metric("Confidence", f"{pred.get('confidence', 0):.1%}")
                                        with col_pred3:
                                            st.metric("Bullish Prob", f"{pred.get('bullish_prob', 0):.1%}")
                                    
                                    # Technical indicators
                                    if 'technical_indicators' in analysis:
                                        st.markdown("**Technical Indicators:**")
                                        st.json(analysis['technical_indicators'])
                                    
                                    # ML scores
                                    if 'ml_scores' in analysis:
                                        st.markdown("**ML Model Scores:**")
                                        st.json(analysis['ml_scores'])
                                    
                                    # Sentiment data
                                    if 'sentiment_data' in analysis:
                                        st.markdown("**Sentiment Analysis:**")
                                        st.json(analysis['sentiment_data'])
                                    
                                    # Signal factors
                                    if 'signal_factors' in analysis:
                                        st.markdown("**Key Signal Factors:**")
                                        for factor in analysis['signal_factors']:
                                            if isinstance(factor, dict):
                                                st.write(f"- {factor.get('factor', factor)}")
                                            else:
                                                st.write(f"- {factor}")
                        
                        # Market context
                        if 'market_context' in raw_analysis:
                            st.markdown("#### Market Context")
                            st.json(raw_analysis['market_context'])
                        
                        # Timeframe context
                        if 'timeframe_context' in raw_analysis:
                            st.markdown("#### Timeframe Analysis")
                            st.write(f"**Timeframe:** {raw_analysis['timeframe_context']}")
                        
                        # Detailed breakdown
                        if 'detailed_breakdown' in raw_analysis:
                            st.markdown("#### Complete Technical Breakdown")
                            st.json(raw_analysis['detailed_breakdown'])
                
                # Display charts with error handling
                try:
                    charts = response.get('charts', [])
                    if charts:
                        st.markdown(f"** Octavian Generated {len(charts)} Chart{'s' if len(charts) > 1 else ''}:**")
                        
                        for chart_data in charts:
                            if 'figure' in chart_data and chart_data['figure'] is not None:
                                st.plotly_chart(chart_data['figure'], width='stretch')
                except Exception:
                    st.warning("Some charts could not be displayed.")
                
                # Show enhanced analysis summary
                if response.get('analysis_summary'):
                    summary = response['analysis_summary']
                    
                    with st.expander(" Octavian Analysis Summary", expanded=False):
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Symbols Analyzed", summary.get('total_symbols', 0))
                        with col2:
                            st.metric("Bullish Signals", summary.get('bullish_count', 0))
                        with col3:
                            st.metric("Bearish Signals", summary.get('bearish_count', 0))
                        with col4:
                            st.metric("Avg Confidence", f"{summary.get('avg_confidence', 0)*100:.0f}%")
                
                # Show enhanced suggestions
                suggestions = response.get('suggestions', [])
                if suggestions:
                    st.markdown("** Octavian Suggestions:**")
                    
                    suggestion_cols = st.columns(min(len(suggestions[:4]), 2))
                    for idx, suggestion in enumerate(suggestions[:4]):
                        with suggestion_cols[idx % 2]:
                            if st.button(f" {suggestion}", key=f"new_octavian_suggestion_{idx}"):
                                st.session_state.suggested_query = suggestion
                                st.rerun()
                
                # Store in history
                st.session_state.octavian_chat_history.append({
                    'query': user_query,
                    'response': response,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Show enhanced success metrics
                response_time = response.get('response_time_ms', 0)
                symbols_count = len(response.get('symbols', []))
                charts_count = len(response.get('charts', []))
                timeframe = response.get('timeframe_context', 'Unknown')
                
                st.success(f" Octavian analysis complete! Processed {symbols_count} symbol{'s' if symbols_count != 1 else ''} for {timeframe.replace('_', ' ')} timeframe, generated {charts_count} chart{'s' if charts_count != 1 else ''} in {response_time}ms")
    
    # Enhanced quick actions
    st.markdown("###  Octavian Quick Intelligence")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button(" Market Overview", key="octavian_market"):
            st.session_state.suggested_query = "Octavian, give me comprehensive market intelligence with SPY, QQQ, and VIX"
            st.rerun()
    
    with col2:
        if st.button(" AI Divergence", key="octavian_divergence"):
            st.session_state.suggested_query = "Show me AI vs market sentiment divergence opportunities"
            st.rerun()
    
    with col3:
        if st.button(" Crypto Intelligence", key="octavian_crypto"):
            st.session_state.suggested_query = "Octavian comprehensive crypto analysis with source weighting"
            st.rerun()
    
    with col4:
        if st.button(" Cross-Asset", key="octavian_cross"):
            st.session_state.suggested_query = "Cross-asset correlation analysis with anticipation factors"
            st.rerun()
    
    # Enhanced footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 15px; background: rgba(218, 165, 32, 0.1); border-radius: 8px;">
        <h4 style="color: #DAA520; margin: 0;"> Octavian by APB</h4>
        <p style="color: #CD853F; margin: 5px 0; font-style: italic;">
            "Where Ancient Wisdom Meets Modern AI"
        </p>
        <p style="color: #B8860B; margin: 0; font-size: 0.9em;">
            Advanced Market Intelligence • Source Credibility Weighting • Timeframe-Aware Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)


# Global instance
_octavian_chatbot = None

def get_chatbot():
    """Get or create Octavian enhanced chatbot instance."""
    global _octavian_chatbot
    if _octavian_chatbot is None:
        _octavian_chatbot = OctavianEnhancedChatbot()
    return _octavian_chatbot