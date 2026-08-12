"""
Octavian Dynamic Intent Detection Engine
Advanced NLP-based intent classification and response strategy formulation.

This engine analyzes user queries to determine:
1. Primary intent (what they're asking)
2. Secondary intents (additional context)
3. Response format preference (how they want the answer)
4. Detail level (depth of analysis required)
5. Urgency (time sensitivity)
6. Comparison needs (single vs multiple assets)
7. Visualization preferences (charts, tables, text)

Author: APB - Octavian Team
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class IntentCategory(Enum):
    """Primary intent categories."""
    PRICE_QUERY = "price_query"
    PREDICTION = "prediction"
    RISK_ANALYSIS = "risk_analysis"
    TECHNICAL_ANALYSIS = "technical_analysis"
    FUNDAMENTAL_ANALYSIS = "fundamental_analysis"
    NEWS_SENTIMENT = "news_sentiment"
    COMPARISON = "comparison"
    SECTOR_ANALYSIS = "sector_analysis"
    PORTFOLIO_ADVICE = "portfolio_advice"
    EDUCATION = "education"
    MARKET_OVERVIEW = "market_overview"
    STRATEGY = "strategy"
    TIMING = "timing"
    VALUATION = "valuation"


class ResponseFormat(Enum):
    """Preferred response format."""
    STANDARD = "standard"  # Default format
    CONCISE = "concise"  # Brief, bullet points
    DETAILED = "detailed"  # Comprehensive analysis
    NARRATIVE = "narrative"  # Story-like explanation
    DATA_FOCUSED = "data_focused"  # Numbers and metrics
    ACTIONABLE = "actionable"  # Clear next steps
    EDUCATIONAL = "educational"  # Explain concepts
    VISUAL = "visual"  # Charts and graphs emphasis


class DetailLevel(Enum):
    """Level of detail required."""
    QUICK = "quick"  # 1-2 sentences
    STANDARD = "standard"  # Normal analysis
    DEEP = "deep"  # Comprehensive deep dive
    EXPERT = "expert"  # Professional-level detail


class Urgency(Enum):
    """Time sensitivity of the query."""
    IMMEDIATE = "immediate"  # Right now, urgent
    TODAY = "today"  # Within trading day
    SHORT_TERM = "short_term"  # Days to weeks
    MEDIUM_TERM = "medium_term"  # Weeks to months
    LONG_TERM = "long_term"  # Months to years
    GENERAL = "general"  # No specific timeframe


@dataclass
class IntentAnalysis:
    """Complete intent analysis result."""
    primary_intent: IntentCategory
    secondary_intents: List[IntentCategory] = field(default_factory=list)
    response_format: ResponseFormat = ResponseFormat.STANDARD
    detail_level: DetailLevel = DetailLevel.STANDARD
    urgency: Urgency = Urgency.GENERAL
    
    # Query characteristics
    is_comparison: bool = False
    comparison_count: int = 1
    requires_charts: bool = False
    requires_tables: bool = False
    requires_calculations: bool = False
    
    # Emotional tone
    is_anxious: bool = False
    is_confident: bool = False
    is_curious: bool = False
    
    # Specific needs
    needs_explanation: bool = False
    needs_validation: bool = False
    needs_alternatives: bool = False
    needs_risk_warning: bool = False
    
    # Response strategy
    lead_with: str = "analysis"  # What to show first
    emphasize: List[str] = field(default_factory=list)  # What to highlight
    de_emphasize: List[str] = field(default_factory=list)  # What to minimize
    
    # Confidence scores
    confidence_score: float = 0.0
    ambiguity_score: float = 0.0
    
    # Raw query info
    original_query: str = ""
    detected_symbols: List[str] = field(default_factory=list)
    detected_keywords: List[str] = field(default_factory=list)


class DynamicIntentDetectionEngine:
    """
    Advanced intent detection engine using multi-layered analysis.
    
    Analyzes queries through multiple lenses:
    1. Keyword matching (basic)
    2. Phrase pattern recognition (intermediate)
    3. Contextual analysis (advanced)
    4. Sentiment and tone detection (expert)
    5. Response strategy formulation (strategic)
    """
    
    def __init__(self):
        self._initialize_patterns()
        self._initialize_response_strategies()
    
    def _initialize_patterns(self):
        """Initialize comprehensive pattern matching rules."""
        
        # Primary intent patterns with confidence weights
        self.intent_patterns = {
            IntentCategory.PRICE_QUERY: {
                'patterns': [
                    (r'\b(what|whats|what\'s)\s+(is|are)\s+the\s+(price|cost|value)', 0.95),
                    (r'\b(current|latest|today\'s)\s+(price|level|quote)', 0.90),
                    (r'\b(how\s+much|trading\s+at|priced\s+at)', 0.85),
                    (r'\b(quote|quotes)\s+(for|on)', 0.80),
                ],
                'keywords': ['price', 'cost', 'value', 'quote', 'level', 'trading at'],
            },
            IntentCategory.PREDICTION: {
                'patterns': [
                    (r'\b(will|should|going\s+to|expect|forecast|predict)', 0.90),
                    (r'\b(outlook|future|next|upcoming|ahead)', 0.85),
                    (r'\b(target|price\s+target|where\s+.*\s+heading)', 0.85),
                    (r'\b(bullish|bearish|buy|sell|long|short)\s+(on|signal)', 0.80),
                ],
                'keywords': ['predict', 'forecast', 'outlook', 'future', 'target', 'will', 'expect'],
            },
            IntentCategory.RISK_ANALYSIS: {
                'patterns': [
                    (r'\b(risk|risks|risky|risk\s+analysis)', 0.95),
                    (r'\b(volatility|volatile|vol|variance)', 0.90),
                    (r'\b(downside|drawdown|max\s+loss|worst\s+case)', 0.90),
                    (r'\b(safe|safety|protect|hedge|insurance)', 0.85),
                    (r'\b(sharpe|sortino|beta|correlation)', 0.95),
                ],
                'keywords': ['risk', 'volatility', 'downside', 'drawdown', 'safe', 'dangerous'],
            },
            IntentCategory.TECHNICAL_ANALYSIS: {
                'patterns': [
                    (r'\b(technical|chart|pattern|indicator)', 0.90),
                    (r'\b(rsi|macd|ema|sma|bollinger|stochastic)', 0.95),
                    (r'\b(support|resistance|breakout|breakdown)', 0.90),
                    (r'\b(trend|momentum|overbought|oversold)', 0.85),
                ],
                'keywords': ['technical', 'chart', 'indicator', 'rsi', 'macd', 'support', 'resistance'],
            },
            IntentCategory.FUNDAMENTAL_ANALYSIS: {
                'patterns': [
                    (r'\b(fundamental|fundamentals|valuation)', 0.90),
                    (r'\b(earnings|revenue|profit|eps|pe\s+ratio)', 0.90),
                    (r'\b(balance\s+sheet|cash\s+flow|debt)', 0.85),
                    (r'\b(growth|margin|return\s+on)', 0.80),
                ],
                'keywords': ['fundamental', 'earnings', 'revenue', 'valuation', 'pe', 'eps'],
            },
            IntentCategory.NEWS_SENTIMENT: {
                'patterns': [
                    (r'\b(news|headlines|articles|reports)', 0.90),
                    (r'\b(sentiment|mood|feeling|vibe)', 0.85),
                    (r'\b(catalyst|event|announcement|release)', 0.85),
                    (r'\b(what.*happening|going\s+on|latest)', 0.80),
                ],
                'keywords': ['news', 'sentiment', 'catalyst', 'event', 'headlines'],
            },
            IntentCategory.COMPARISON: {
                'patterns': [
                    (r'\b(compare|comparison|versus|vs|vs\.|better|worse)', 0.95),
                    (r'\b(which\s+(is|are)\s+(better|best|worse|worst))', 0.90),
                    (r'\b(difference|differences|between)', 0.85),
                    (r'\b(or)\b.*\b(or)\b', 0.70),  # "X or Y or Z"
                ],
                'keywords': ['compare', 'versus', 'vs', 'better', 'difference', 'between'],
            },
            IntentCategory.SECTOR_ANALYSIS: {
                'patterns': [
                    (r'\b(sector|sectors|industry|industries)', 0.95),
                    (r'\b(rotation|sector\s+rotation)', 0.90),
                    (r'\b(which\s+sector|best\s+sector|worst\s+sector)', 0.90),
                ],
                'keywords': ['sector', 'industry', 'rotation'],
            },
            IntentCategory.PORTFOLIO_ADVICE: {
                'patterns': [
                    (r'\b(portfolio|allocation|diversif)', 0.90),
                    (r'\b(should\s+i\s+(buy|sell|hold|add))', 0.85),
                    (r'\b(position\s+size|how\s+much|percentage)', 0.85),
                ],
                'keywords': ['portfolio', 'allocation', 'diversify', 'position'],
            },
            IntentCategory.EDUCATION: {
                'patterns': [
                    (r'\b(what\s+(is|are|does)|explain|how\s+does)', 0.90),
                    (r'\b(mean|meaning|definition|understand)', 0.85),
                    (r'\b(learn|teach|educate|beginner)', 0.85),
                ],
                'keywords': ['explain', 'what is', 'how does', 'mean', 'learn'],
            },
            IntentCategory.MARKET_OVERVIEW: {
                'patterns': [
                    (r'\b(market|markets|overall|general)', 0.80),
                    (r'\b(today|this\s+week|this\s+month)', 0.75),
                    (r'\b(summary|overview|snapshot|update)', 0.85),
                ],
                'keywords': ['market', 'overall', 'summary', 'overview'],
            },
            IntentCategory.STRATEGY: {
                'patterns': [
                    (r'\b(strategy|strategies|approach|plan)', 0.90),
                    (r'\b(trade|trading|invest|investing)', 0.75),
                    (r'\b(entry|exit|stop\s+loss|take\s+profit)', 0.85),
                ],
                'keywords': ['strategy', 'approach', 'plan', 'entry', 'exit'],
            },
            IntentCategory.TIMING: {
                'patterns': [
                    (r'\b(when|timing|time\s+to)', 0.85),
                    (r'\b(now|right\s+now|immediately|today)', 0.80),
                    (r'\b(wait|hold\s+off|patience)', 0.75),
                ],
                'keywords': ['when', 'timing', 'now', 'wait'],
            },
            IntentCategory.VALUATION: {
                'patterns': [
                    (r'\b(overvalued|undervalued|fair\s+value)', 0.95),
                    (r'\b(cheap|expensive|worth|valued)', 0.80),
                    (r'\b(intrinsic|dcf|discounted)', 0.90),
                ],
                'keywords': ['overvalued', 'undervalued', 'cheap', 'expensive', 'worth'],
            },
        }
        
        # Response format indicators
        self.format_indicators = {
            ResponseFormat.CONCISE: [
                'quick', 'brief', 'short', 'summary', 'tldr', 'simple', 'fast'
            ],
            ResponseFormat.DETAILED: [
                'detailed', 'comprehensive', 'full', 'complete', 'thorough', 'in-depth', 'deep'
            ],
            ResponseFormat.NARRATIVE: [
                'explain', 'story', 'walk me through', 'tell me', 'describe'
            ],
            ResponseFormat.DATA_FOCUSED: [
                'numbers', 'data', 'metrics', 'statistics', 'stats', 'figures'
            ],
            ResponseFormat.VISUAL: [
                'chart', 'graph', 'plot', 'visual', 'show me', 'display'
            ],
            ResponseFormat.ACTIONABLE: [
                'what should', 'recommend', 'advice', 'action', 'do', 'next steps'
            ],
            ResponseFormat.EDUCATIONAL: [
                'explain', 'teach', 'learn', 'understand', 'how does', 'what is'
            ],
        }
        
        # Detail level indicators
        self.detail_indicators = {
            DetailLevel.QUICK: [
                'quick', 'fast', 'brief', 'short', 'tldr', 'just', 'simply'
            ],
            DetailLevel.STANDARD: [
                'normal', 'regular', 'standard', 'typical'
            ],
            DetailLevel.DEEP: [
                'deep', 'detailed', 'comprehensive', 'thorough', 'full', 'complete'
            ],
            DetailLevel.EXPERT: [
                'expert', 'professional', 'advanced', 'technical', 'institutional'
            ],
        }
        
        # Urgency indicators
        self.urgency_indicators = {
            Urgency.IMMEDIATE: [
                'now', 'right now', 'immediately', 'urgent', 'asap', 'quick'
            ],
            Urgency.TODAY: [
                'today', 'this morning', 'this afternoon', 'end of day', 'eod'
            ],
            Urgency.SHORT_TERM: [
                'this week', 'next week', 'few days', 'short term', 'near term'
            ],
            Urgency.MEDIUM_TERM: [
                'this month', 'next month', 'few weeks', 'medium term', 'swing'
            ],
            Urgency.LONG_TERM: [
                'long term', 'months', 'years', 'invest', 'hold', 'buy and hold'
            ],
        }
        
        # Emotional tone indicators
        self.emotion_indicators = {
            'anxious': ['worried', 'concerned', 'nervous', 'scared', 'afraid', 'anxious'],
            'confident': ['confident', 'sure', 'certain', 'convinced', 'bullish'],
            'curious': ['curious', 'wondering', 'interested', 'exploring', 'learning'],
        }

    
    def _initialize_response_strategies(self):
        """Initialize response strategy templates for each intent."""
        
        self.response_strategies = {
            IntentCategory.PRICE_QUERY: {
                'lead_with': 'current_price',
                'emphasize': ['price', 'change', 'volume'],
                'de_emphasize': ['long_term_outlook', 'fundamentals'],
                'include_chart': False,
                'include_comparison': False,
            },
            IntentCategory.PREDICTION: {
                'lead_with': 'forecast',
                'emphasize': ['direction', 'targets', 'probability', 'timeframe'],
                'de_emphasize': ['historical_data'],
                'include_chart': True,
                'include_comparison': False,
            },
            IntentCategory.RISK_ANALYSIS: {
                'lead_with': 'risk_metrics',
                'emphasize': ['volatility', 'downside', 'risk_factors', 'protection'],
                'de_emphasize': ['upside_potential'],
                'include_chart': True,
                'include_comparison': False,
            },
            IntentCategory.TECHNICAL_ANALYSIS: {
                'lead_with': 'technical_indicators',
                'emphasize': ['indicators', 'patterns', 'levels', 'momentum'],
                'de_emphasize': ['fundamentals', 'news'],
                'include_chart': True,
                'include_comparison': False,
            },
            IntentCategory.FUNDAMENTAL_ANALYSIS: {
                'lead_with': 'fundamentals',
                'emphasize': ['earnings', 'valuation', 'growth', 'financials'],
                'de_emphasize': ['technical_indicators'],
                'include_chart': False,
                'include_comparison': True,
            },
            IntentCategory.NEWS_SENTIMENT: {
                'lead_with': 'news_summary',
                'emphasize': ['sentiment', 'catalysts', 'recent_news', 'impact'],
                'de_emphasize': ['technical_details'],
                'include_chart': False,
                'include_comparison': False,
            },
            IntentCategory.COMPARISON: {
                'lead_with': 'comparison_table',
                'emphasize': ['differences', 'relative_strength', 'pros_cons'],
                'de_emphasize': ['individual_deep_dive'],
                'include_chart': True,
                'include_comparison': True,
            },
            IntentCategory.SECTOR_ANALYSIS: {
                'lead_with': 'sector_performance',
                'emphasize': ['rotation', 'relative_performance', 'leaders_laggards'],
                'de_emphasize': ['individual_stocks'],
                'include_chart': True,
                'include_comparison': True,
            },
            IntentCategory.PORTFOLIO_ADVICE: {
                'lead_with': 'recommendation',
                'emphasize': ['allocation', 'position_size', 'risk_management', 'action_steps'],
                'de_emphasize': ['technical_jargon'],
                'include_chart': False,
                'include_comparison': False,
            },
            IntentCategory.EDUCATION: {
                'lead_with': 'explanation',
                'emphasize': ['concepts', 'examples', 'step_by_step', 'context'],
                'de_emphasize': ['complex_math', 'jargon'],
                'include_chart': True,
                'include_comparison': False,
            },
            IntentCategory.MARKET_OVERVIEW: {
                'lead_with': 'market_summary',
                'emphasize': ['overall_trend', 'key_movers', 'sentiment', 'context'],
                'de_emphasize': ['individual_analysis'],
                'include_chart': True,
                'include_comparison': False,
            },
            IntentCategory.STRATEGY: {
                'lead_with': 'strategy_outline',
                'emphasize': ['approach', 'entry_exit', 'risk_management', 'execution'],
                'de_emphasize': ['theory'],
                'include_chart': True,
                'include_comparison': False,
            },
            IntentCategory.TIMING: {
                'lead_with': 'timing_analysis',
                'emphasize': ['current_conditions', 'triggers', 'patience_factors'],
                'de_emphasize': ['long_term_fundamentals'],
                'include_chart': True,
                'include_comparison': False,
            },
            IntentCategory.VALUATION: {
                'lead_with': 'valuation_assessment',
                'emphasize': ['fair_value', 'multiples', 'comparison_to_peers'],
                'de_emphasize': ['short_term_technicals'],
                'include_chart': False,
                'include_comparison': True,
            },
        }
    
    def analyze_intent(self, query: str, detected_symbols: List[str] = None) -> IntentAnalysis:
        """
        Perform comprehensive intent analysis on user query.
        
        Args:
            query: User's query string
            detected_symbols: List of symbols detected in query
        
        Returns:
            IntentAnalysis object with complete analysis
        """
        query_lower = query.lower()
        
        # Initialize result
        result = IntentAnalysis(
            primary_intent=IntentCategory.TECHNICAL_ANALYSIS,  # Default
            original_query=query,
            detected_symbols=detected_symbols or []
        )
        
        # Step 1: Detect all matching intents with confidence scores
        intent_scores = self._score_all_intents(query_lower)
        
        # Step 2: Determine primary and secondary intents
        result.primary_intent, result.secondary_intents, result.confidence_score = \
            self._determine_intent_hierarchy(intent_scores)
        
        # Step 3: Detect response format preference
        result.response_format = self._detect_response_format(query_lower)
        
        # Step 4: Detect detail level
        result.detail_level = self._detect_detail_level(query_lower)
        
        # Step 5: Detect urgency
        result.urgency = self._detect_urgency(query_lower)
        
        # Step 6: Detect comparison needs
        result.is_comparison, result.comparison_count = self._detect_comparison(
            query_lower, detected_symbols or []
        )
        
        # Step 7: Detect visualization needs
        result.requires_charts = self._requires_charts(query_lower)
        result.requires_tables = self._requires_tables(query_lower)
        result.requires_calculations = self._requires_calculations(query_lower)
        
        # Step 8: Detect emotional tone
        result.is_anxious = self._detect_emotion(query_lower, 'anxious')
        result.is_confident = self._detect_emotion(query_lower, 'confident')
        result.is_curious = self._detect_emotion(query_lower, 'curious')
        
        # Step 9: Detect specific needs
        result.needs_explanation = self._needs_explanation(query_lower)
        result.needs_validation = self._needs_validation(query_lower)
        result.needs_alternatives = self._needs_alternatives(query_lower)
        result.needs_risk_warning = self._needs_risk_warning(query_lower, result.primary_intent)
        
        # Step 10: Formulate response strategy
        strategy = self._formulate_response_strategy(result)
        result.lead_with = strategy['lead_with']
        result.emphasize = strategy['emphasize']
        result.de_emphasize = strategy['de_emphasize']
        
        # Step 11: Calculate ambiguity score
        result.ambiguity_score = self._calculate_ambiguity(intent_scores)
        
        # Step 12: Extract keywords
        result.detected_keywords = self._extract_keywords(query_lower)
        
        return result
    
    def _score_all_intents(self, query_lower: str) -> Dict[IntentCategory, float]:
        """Score all intents against the query."""
        scores = {}
        
        for intent, config in self.intent_patterns.items():
            score = 0.0
            matches = 0
            
            # Pattern matching
            for pattern, weight in config['patterns']:
                if re.search(pattern, query_lower):
                    score += weight
                    matches += 1
            
            # Keyword matching
            for keyword in config['keywords']:
                if keyword in query_lower:
                    score += 0.3
                    matches += 1
            
            # Normalize by number of matches
            if matches > 0:
                scores[intent] = score / max(matches, 1)
            else:
                scores[intent] = 0.0
        
        return scores
    
    def _determine_intent_hierarchy(self, scores: Dict[IntentCategory, float]) -> Tuple[IntentCategory, List[IntentCategory], float]:
        """Determine primary and secondary intents from scores."""
        sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_intents or sorted_intents[0][1] == 0:
            # No clear intent, default to technical analysis
            return IntentCategory.TECHNICAL_ANALYSIS, [], 0.5
        
        primary = sorted_intents[0][0]
        primary_score = sorted_intents[0][1]
        
        # Secondary intents are those with score > 0.5 and within 70% of primary
        threshold = primary_score * 0.7
        secondary = [
            intent for intent, score in sorted_intents[1:]
            if score > 0.5 and score >= threshold
        ]
        
        # Confidence is based on how much primary dominates
        if len(sorted_intents) > 1:
            gap = primary_score - sorted_intents[1][1]
            confidence = min(0.95, 0.5 + (gap * 0.5))
        else:
            confidence = 0.8
        
        return primary, secondary[:3], confidence  # Max 3 secondary intents
    
    def _detect_response_format(self, query_lower: str) -> ResponseFormat:
        """Detect preferred response format."""
        format_scores = {}
        
        for format_type, indicators in self.format_indicators.items():
            score = sum(1 for indicator in indicators if indicator in query_lower)
            if score > 0:
                format_scores[format_type] = score
        
        if not format_scores:
            return ResponseFormat.STANDARD
        
        return max(format_scores.items(), key=lambda x: x[1])[0]
    
    def _detect_detail_level(self, query_lower: str) -> DetailLevel:
        """Detect required detail level."""
        level_scores = {}
        
        for level, indicators in self.detail_indicators.items():
            score = sum(1 for indicator in indicators if indicator in query_lower)
            if score > 0:
                level_scores[level] = score
        
        if not level_scores:
            return DetailLevel.STANDARD
        
        return max(level_scores.items(), key=lambda x: x[1])[0]
    
    def _detect_urgency(self, query_lower: str) -> Urgency:
        """Detect time urgency."""
        urgency_scores = {}
        
        for urgency, indicators in self.urgency_indicators.items():
            score = sum(1 for indicator in indicators if indicator in query_lower)
            if score > 0:
                urgency_scores[urgency] = score
        
        if not urgency_scores:
            return Urgency.GENERAL
        
        return max(urgency_scores.items(), key=lambda x: x[1])[0]
    
    def _detect_comparison(self, query_lower: str, symbols: List[str]) -> Tuple[bool, int]:
        """Detect if comparison is needed."""
        comparison_keywords = ['compare', 'versus', 'vs', 'vs.', 'better', 'worse', 'difference']
        has_comparison_keyword = any(kw in query_lower for kw in comparison_keywords)
        
        # Check for "or" pattern (X or Y)
        or_count = query_lower.count(' or ')
        
        # Check symbol count
        symbol_count = len(symbols) if symbols else 0
        
        is_comparison = has_comparison_keyword or or_count > 0 or symbol_count > 1
        comparison_count = max(symbol_count, or_count + 1, 2 if has_comparison_keyword else 1)
        
        return is_comparison, comparison_count
    
    def _requires_charts(self, query_lower: str) -> bool:
        """Check if charts are requested."""
        chart_keywords = ['chart', 'graph', 'plot', 'visual', 'show me', 'display']
        return any(kw in query_lower for kw in chart_keywords)
    
    def _requires_tables(self, query_lower: str) -> bool:
        """Check if tables are requested."""
        table_keywords = ['table', 'comparison table', 'data', 'metrics', 'breakdown']
        return any(kw in query_lower for kw in table_keywords)
    
    def _requires_calculations(self, query_lower: str) -> bool:
        """Check if calculations are needed."""
        calc_keywords = ['calculate', 'compute', 'math', 'formula', 'equation']
        return any(kw in query_lower for kw in calc_keywords)
    
    def _detect_emotion(self, query_lower: str, emotion: str) -> bool:
        """Detect emotional tone."""
        indicators = self.emotion_indicators.get(emotion, [])
        return any(indicator in query_lower for indicator in indicators)
    
    def _needs_explanation(self, query_lower: str) -> bool:
        """Check if explanation is needed."""
        explain_keywords = ['explain', 'what is', 'what are', 'how does', 'why', 'mean']
        return any(kw in query_lower for kw in explain_keywords)
    
    def _needs_validation(self, query_lower: str) -> bool:
        """Check if user seeks validation."""
        validation_keywords = ['right', 'correct', 'good idea', 'should i', 'am i', 'agree']
        return any(kw in query_lower for kw in validation_keywords)
    
    def _needs_alternatives(self, query_lower: str) -> bool:
        """Check if alternatives are requested."""
        alt_keywords = ['alternative', 'instead', 'other', 'else', 'different']
        return any(kw in query_lower for kw in alt_keywords)
    
    def _needs_risk_warning(self, query_lower: str, primary_intent: IntentCategory) -> bool:
        """Determine if risk warning should be included."""
        # Always warn for portfolio advice
        if primary_intent == IntentCategory.PORTFOLIO_ADVICE:
            return True
        
        # Warn if asking about risky actions
        risky_keywords = ['yolo', 'all in', 'leverage', 'margin', 'options', 'short']
        return any(kw in query_lower for kw in risky_keywords)
    
    def _formulate_response_strategy(self, analysis: IntentAnalysis) -> Dict[str, Any]:
        """Formulate complete response strategy."""
        base_strategy = self.response_strategies.get(
            analysis.primary_intent,
            self.response_strategies[IntentCategory.TECHNICAL_ANALYSIS]
        ).copy()
        
        # Adjust based on response format
        if analysis.response_format == ResponseFormat.CONCISE:
            base_strategy['emphasize'] = base_strategy['emphasize'][:2]  # Limit emphasis
        elif analysis.response_format == ResponseFormat.DETAILED:
            base_strategy['de_emphasize'] = []  # Show everything
        
        # Adjust based on detail level
        if analysis.detail_level == DetailLevel.QUICK:
            base_strategy['lead_with'] = 'summary'
        elif analysis.detail_level == DetailLevel.EXPERT:
            base_strategy['emphasize'].extend(['methodology', 'assumptions', 'limitations'])
        
        # Adjust for comparison
        if analysis.is_comparison:
            base_strategy['include_comparison'] = True
            if 'comparison' not in base_strategy['emphasize']:
                base_strategy['emphasize'].insert(0, 'comparison')
        
        # Adjust for urgency
        if analysis.urgency == Urgency.IMMEDIATE:
            base_strategy['lead_with'] = 'action_now'
            base_strategy['emphasize'].insert(0, 'immediate_action')
        
        return base_strategy
    
    def _calculate_ambiguity(self, scores: Dict[IntentCategory, float]) -> float:
        """Calculate how ambiguous the query is."""
        if not scores:
            return 1.0
        
        sorted_scores = sorted(scores.values(), reverse=True)
        
        if len(sorted_scores) < 2:
            return 0.0
        
        # Ambiguity is high when top scores are similar
        top_score = sorted_scores[0]
        second_score = sorted_scores[1]
        
        if top_score == 0:
            return 1.0
        
        # Ratio of second to first (closer to 1 = more ambiguous)
        ratio = second_score / top_score
        return min(1.0, ratio)
    
    def _extract_keywords(self, query_lower: str) -> List[str]:
        """Extract important keywords from query."""
        # Remove common words
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been'}
        
        words = re.findall(r'\b\w+\b', query_lower)
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        
        return keywords[:10]  # Top 10 keywords


# Singleton instance
_intent_engine = None

def get_intent_detection_engine() -> DynamicIntentDetectionEngine:
    """Get singleton intent detection engine."""
    global _intent_engine
    if _intent_engine is None:
        _intent_engine = DynamicIntentDetectionEngine()
    return _intent_engine
