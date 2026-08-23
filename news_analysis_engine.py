"""
Comprehensive News and Market Analysis Engine

This module provides advanced news analysis and market sentiment capabilities:
1. Real-time news aggregation from multiple sources
2. AI-powered sentiment analysis and market impact assessment
3. Market whispers and social sentiment tracking
4. Event-driven analysis and correlation detection
5. Integration with ML models and chatbot responses

Author: AI Market Team
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import re
import json
from typing import Dict, List, Optional, Any, Tuple
import threading
import time
import random
from concurrent.futures import ThreadPoolExecutor
import logging
from dataclasses import dataclass
from enum import Enum
import hashlib
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime


def _naive_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Normalize any datetime to naive UTC for safe comparison.

    RSS feed dates parsed with ``parsedate_to_datetime`` are timezone-aware,
    while ``datetime.now()`` and ``datetime.min`` are naive — mixing them in
    sort keys or cutoff comparisons raises "can't compare offset-naive and
    offset-aware datetimes".  Converting everything to naive UTC keeps every
    comparison consistent (None passes through unchanged).
    """
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


# NLP and sentiment analysis
NLTK_AVAILABLE = False
try:
    from textblob import TextBlob
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
except ImportError:
    print("NLTK/TextBlob not available - using basic sentiment analysis")

def _ensure_nltk_data():
    """Ensure NLTK data is available without blocking imports."""
    if not NLTK_AVAILABLE:
        return
        
    try:
        import os
        project_root = os.path.dirname(os.path.abspath(__file__))
        nltk_data_path = os.path.join(project_root, 'nltk_data')
        if nltk_data_path not in nltk.data.path:
            nltk.data.path.append(nltk_data_path)
            
        # Check if already downloaded to avoid re-downloading
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon', download_dir=nltk_data_path, quiet=True)
            
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
            nltk.download('punkt_tab', download_dir=nltk_data_path, quiet=True)
            
    except Exception as e:
        print(f"Warning: NLTK data check/download failed: {e}")

# NLTK data is initialized on first use (not at import time) to speed up app startup
_nltk_data_ready = False

def _lazy_ensure_nltk():
    global _nltk_data_ready
    if not _nltk_data_ready and NLTK_AVAILABLE:
        _ensure_nltk_data()
        _nltk_data_ready = True

# Web scraping
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    print("BeautifulSoup not available - limited web scraping")

from database_manager import get_database_manager
from config import ALPHA_VANTAGE_KEY
from api_rate_limiter import get_api_handler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentScore(Enum):
    """Sentiment score categories."""
    VERY_BEARISH = -2
    BEARISH = -1
    NEUTRAL = 0
    BULLISH = 1
    VERY_BULLISH = 2

@dataclass
class NewsArticle:
    """News article data structure."""
    title: str
    summary: str
    url: str
    source: str
    published_at: datetime
    symbols_mentioned: List[str]
    sentiment_score: float
    sentiment_category: SentimentScore
    market_impact_score: float
    relevance_score: float
    article_id: str
    full_text: Optional[str] = None
    author: Optional[str] = None
    tags: List[str] = None

@dataclass
class MarketWhisper:
    """Market whisper/rumor data structure."""
    content: str
    source: str
    confidence_level: float
    symbols_mentioned: List[str]
    whisper_type: str  # 'earnings', 'merger', 'product', 'regulatory', 'other'
    timestamp: datetime
    social_mentions: int
    verification_status: str  # 'unverified', 'partially_verified', 'verified', 'debunked'

@dataclass
class SentimentAnalysis:
    """Comprehensive sentiment analysis result."""
    overall_sentiment: float
    sentiment_category: SentimentScore
    confidence: float
    key_phrases: List[str]
    emotional_indicators: Dict[str, float]
    market_relevance: float

class NewsAnalysisEngine:
    """Comprehensive news and sentiment analysis engine."""
    
    def __init__(self):
        _lazy_ensure_nltk()
        self.db_manager = get_database_manager()
        self.api_handler = get_api_handler()

        # Initialize sentiment analyzer
        if NLTK_AVAILABLE:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        else:
            self.sentiment_analyzer = None
        
        # News sources configuration
        self.news_sources = {
            'alpha_vantage': {
                'url': 'https://www.alphavantage.co/query',
                'api_key': ALPHA_VANTAGE_KEY,
                'enabled': True
            },
            'financial_modeling_prep': {
                'url': 'https://financialmodelingprep.com/api/v3/stock_news',
                'enabled': False  # Requires API key
            },
            'reddit_finance': {
                'url': 'https://www.reddit.com/r/investing.json',
                'enabled': True
            },
            'yahoo_finance_rss': {
                'url': 'https://feeds.finance.yahoo.com/rss/2.0/headline',
                'enabled': True
            },
            'rss_feeds': {
                'enabled': True,
                'feeds': {
                    'CNBC Markets': 'https://www.cnbc.com/id/10000664/device/rss/rss.html',
                    'CNBC Economy': 'https://www.cnbc.com/id/20910258/device/rss/rss.html',
                    'CNBC Finance': 'https://www.cnbc.com/id/10000664/device/rss/rss.html',
                    'MarketWatch': 'http://feeds.marketwatch.com/marketwatch/marketpulse/',
                    'Seeking Alpha': 'https://seekingalpha.com/market_currents.xml',
                    'WSJ Markets': 'https://feeds.a.dj.com/rss/RSSMarketsMain.xml',
                    'WSJ Tech': 'https://feeds.a.dj.com/rss/RSSWSJD.xml',
                    'Investing.com': 'https://www.investing.com/rss/news.rss',
                    'CoinDesk': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
                    'CoinTelegraph': 'https://cointelegraph.com/rss',
                    'Yahoo Finance Top': 'https://finance.yahoo.com/news/rssindex',
                    'TechCrunch': 'https://techcrunch.com/feed/',
                    'OilPrice': 'https://oilprice.com/rss/main',
                    'Financial Times': 'https://www.ft.com/?format=rss',
                    'ForexLive': 'https://www.forexlive.com/feed',
                    'BBC Business': 'http://feeds.bbci.co.uk/news/business/rss.xml',
                    'CNN Business': 'http://rss.cnn.com/rss/money_latest.rss',
                }
            }
        }
        
        # Market keywords for relevance scoring
        self.market_keywords = {
            'high_impact': [
                'earnings', 'revenue', 'profit', 'loss', 'guidance', 'outlook',
                'merger', 'acquisition', 'ipo', 'bankruptcy', 'dividend',
                'fed', 'interest rate', 'inflation', 'gdp', 'unemployment'
            ],
            'medium_impact': [
                'partnership', 'contract', 'product launch', 'expansion',
                'regulatory', 'approval', 'investigation', 'lawsuit',
                'analyst', 'upgrade', 'downgrade', 'target price'
            ],
            'low_impact': [
                'conference', 'presentation', 'interview', 'statement',
                'comment', 'opinion', 'forecast', 'estimate'
            ]
        }
        
        # Symbol extraction patterns
        self.symbol_patterns = [
            r'\b[A-Z]{1,5}\b',  # Basic ticker pattern
            r'\$[A-Z]{1,5}\b',  # Twitter-style ticker
            r'\b[A-Z]{1,5}:[A-Z]{2,3}\b',  # Exchange:Ticker format
            r'\b\^[A-Z]{1,6}\b',  # Index symbols like ^GSPC
            r'\b[A-Z]{1,3}=F\b',  # Futures like ES=F
            r'\b[A-Z]{3}[\/_-][A-Z]{3}\b',  # FX like EUR/USD
        ]
        
        # Cache for processed articles
        self.article_cache = {}
        self.cache_expiry = 3600  # 1 hour

        # Short-term in-memory cache for fetch_and_process_news()
        self._last_fetch_ts: float = 0.0
        self._last_processed_articles: List[NewsArticle] = []
        
        # Background processing
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.running = False
        
        # Performance metrics
        self.metrics = {
            'articles_processed': 0,
            'sentiment_analyses': 0,
            'whispers_detected': 0,
            'api_calls': 0,
            'cache_hits': 0,
            'last_update': None
        }
    
    def start_background_processing(self):
        """Start background news processing."""
        self.running = True
        
        def process_news_continuously():
            while self.running:
                try:
                    self.fetch_and_process_news()
                    time.sleep(300)  # Process every 5 minutes
                except Exception as e:
                    logger.error(f"Error in background news processing: {e}")
                    time.sleep(60)  # Wait 1 minute on error
        
        thread = threading.Thread(target=process_news_continuously, daemon=True)
        thread.start()
        logger.info("Background news processing started")
    
    def fetch_and_process_news(self) -> List[NewsArticle]:
        """Fetch and process news from all enabled sources."""
        # Avoid refetching on every dashboard render
        now_ts = time.time()
        if self._last_processed_articles and (now_ts - self._last_fetch_ts) < 300:
            return list(self._last_processed_articles)

        all_articles = []
        
        # Fetch from each enabled source
        for source_name, config in self.news_sources.items():
            if not config.get('enabled', False):
                continue
            
            try:
                if source_name == 'alpha_vantage':
                    articles = self._fetch_alpha_vantage_news()
                elif source_name == 'reddit_finance':
                    articles = self._fetch_reddit_finance()
                elif source_name == 'yahoo_finance_rss':
                    articles = self._fetch_yahoo_rss()
                elif source_name == 'rss_feeds':
                    articles = self._fetch_generic_rss()
                else:
                    continue
                
                all_articles.extend(articles)
                self.metrics['api_calls'] += 1
                
                # Sleep briefly between sources to avoid rate limits
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error fetching from {source_name}: {e}")
        
        # Process and deduplicate articles
        processed_articles = self._process_articles(all_articles)
        
        # Store in database
        self._store_articles(processed_articles)
        
        self.metrics['articles_processed'] += len(processed_articles)
        self.metrics['last_update'] = datetime.now()

        self._last_fetch_ts = now_ts
        self._last_processed_articles = list(processed_articles)
        
        return processed_articles

    def _fetch_generic_rss(self) -> List[NewsArticle]:
        """Fetch news from a set of generic RSS feeds with rate limiting and caching."""
        out: List[NewsArticle] = []
        try:
            cfg = self.news_sources.get('rss_feeds', {})
            feeds = cfg.get('feeds', {}) if isinstance(cfg, dict) else {}
            for label, url in feeds.items():
                try:
                    response = self.api_handler.fetch_with_retry(
                        url=url,
                        service='rss',
                        max_retries=2,
                        timeout=12
                    )
                    
                    if not response:
                        logger.warning(f"Failed to fetch RSS for {label}")
                        continue
                    
                    out.extend(self._parse_rss_xml(response.text, label))
                    time.sleep(0.1)  # Additional rate limiting
                except Exception as e:
                    logger.error(f"RSS fetch error for {label}: {e}")
        except Exception as e:
            logger.error(f"Generic RSS error: {e}")
        return out

    def _parse_rss_xml(self, xml_text: str, source_label: str) -> List[NewsArticle]:
        """Parse RSS/Atom XML into NewsArticle list."""
        articles: List[NewsArticle] = []
        try:
            root = ET.fromstring(xml_text)

            # RSS
            items = root.findall('.//item')
            if items:
                # No artificial limit, fetch all available
                for item in items:
                    title = (item.findtext('title') or '').strip()
                    link = (item.findtext('link') or '').strip()
                    desc = (item.findtext('description') or item.findtext('summary') or '').strip()
                    pub = (item.findtext('pubDate') or '').strip()
                    try:
                        published_at = parsedate_to_datetime(pub) if pub else datetime.now()
                    except Exception:
                        published_at = datetime.now()

                    text_content = f"{title} {desc}".strip()
                    if not text_content:
                        continue
                        
                    # Basic cleaning
                    text_content = re.sub(r'<[^>]+>', '', text_content)

                    symbols = self._extract_symbols(text_content)
                    sentiment = self.analyze_sentiment(text_content)
                    articles.append(NewsArticle(
                        title=title or f"{source_label} Update",
                        summary=desc[:200] + "..." if len(desc) > 200 else desc,
                        url=link,
                        source=source_label,
                        published_at=published_at,
                        symbols_mentioned=symbols,
                        sentiment_score=sentiment.overall_sentiment,
                        sentiment_category=sentiment.sentiment_category,
                        market_impact_score=self._calculate_market_impact(text_content),
                        relevance_score=self._calculate_relevance(text_content),
                        article_id=hashlib.md5((link or title or desc).encode()).hexdigest(),
                        author=None,
                    ))
                return articles

            # Atom (best-effort)
            entries = root.findall('.//{http://www.w3.org/2005/Atom}entry')
            for entry in entries:
                title = (entry.findtext('{http://www.w3.org/2005/Atom}title') or '').strip()
                link_el = entry.find('{http://www.w3.org/2005/Atom}link')
                link = (link_el.attrib.get('href') if link_el is not None else '') or ''
                summary = (entry.findtext('{http://www.w3.org/2005/Atom}summary') or '').strip()
                updated = (entry.findtext('{http://www.w3.org/2005/Atom}updated') or '').strip()

                try:
                    published_at = datetime.fromisoformat(updated.replace('Z', '+00:00')) if updated else datetime.now()
                except Exception:
                    published_at = datetime.now()

                text_content = f"{title} {summary}".strip()
                if not text_content:
                    continue

                # Basic cleaning
                text_content = re.sub(r'<[^>]+>', '', text_content)

                symbols = self._extract_symbols(text_content)
                sentiment = self.analyze_sentiment(text_content)
                articles.append(NewsArticle(
                    title=title or f"{source_label} Update",
                    summary=summary[:200] + "..." if len(summary) > 200 else summary,
                    url=link,
                    source=source_label,
                    published_at=published_at,
                    symbols_mentioned=symbols,
                    sentiment_score=sentiment.overall_sentiment,
                    sentiment_category=sentiment.sentiment_category,
                    market_impact_score=self._calculate_market_impact(text_content),
                    relevance_score=self._calculate_relevance(text_content),
                    article_id=hashlib.md5((link or title or summary).encode()).hexdigest(),
                    author=None,
                ))
        except Exception as e:
            logger.error(f"RSS parse error ({source_label}): {e}")

        return articles
    
    def _fetch_alpha_vantage_news(self) -> List[NewsArticle]:
        """Fetch news from Alpha Vantage API."""
        if not ALPHA_VANTAGE_KEY:
            return []
        
        try:
            url = self.news_sources['alpha_vantage']['url']
            params = {
                'function': 'NEWS_SENTIMENT',
                'apikey': ALPHA_VANTAGE_KEY,
                'limit': 50
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            articles = []
            
            if 'feed' in data:
                for item in data['feed']:
                    # Extract symbols mentioned
                    symbols = []
                    if 'ticker_sentiment' in item:
                        symbols = [ts['ticker'] for ts in item['ticker_sentiment']]
                    
                    # Calculate sentiment score
                    sentiment_score = 0.0
                    if 'overall_sentiment_score' in item:
                        sentiment_score = float(item['overall_sentiment_score'])
                    
                    article = NewsArticle(
                        title=item.get('title', ''),
                        summary=item.get('summary', ''),
                        url=item.get('url', ''),
                        source=item.get('source', 'Alpha Vantage'),
                        published_at=datetime.strptime(item.get('time_published', ''), '%Y%m%dT%H%M%S'),
                        symbols_mentioned=symbols,
                        sentiment_score=sentiment_score,
                        sentiment_category=self._score_to_category(sentiment_score),
                        market_impact_score=self._calculate_market_impact(item.get('title', '') + ' ' + item.get('summary', '')),
                        relevance_score=self._calculate_relevance(item.get('title', '') + ' ' + item.get('summary', '')),
                        article_id=hashlib.md5(item.get('url', '').encode()).hexdigest(),
                        author=item.get('authors', [None])[0] if item.get('authors') else None
                    )
                    
                    articles.append(article)
            
            return articles
        
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage news: {e}")
            return []
    
    def _fetch_reddit_finance(self) -> List[NewsArticle]:
        """Fetch financial discussions from Reddit with proper rate limiting and retry logic."""
        try:
            response = self.api_handler.fetch_with_retry(
                url='https://www.reddit.com/r/investing.json',
                service='reddit',
                max_retries=3,
                timeout=10
            )
            
            if not response:
                logger.warning("Failed to fetch Reddit data after retries")
                return []
            
            data = response.json()
            articles = []
            
            if 'data' in data and 'children' in data['data']:
                for post in data['data']['children']:
                    post_data = post['data']
                    
                    # Extract symbols from title and text
                    text_content = post_data.get('title', '') + ' ' + post_data.get('selftext', '')
                    symbols = self._extract_symbols(text_content)
                    
                    # Analyze sentiment
                    sentiment_analysis = self.analyze_sentiment(text_content)
                    
                    article = NewsArticle(
                        title=post_data.get('title', ''),
                        summary=post_data.get('selftext', '')[:500] + '...' if len(post_data.get('selftext', '')) > 500 else post_data.get('selftext', ''),
                        url=f"https://reddit.com{post_data.get('permalink', '')}",
                        source='Reddit r/investing',
                        published_at=datetime.fromtimestamp(post_data.get('created_utc', 0)),
                        symbols_mentioned=symbols,
                        sentiment_score=sentiment_analysis.overall_sentiment,
                        sentiment_category=sentiment_analysis.sentiment_category,
                        market_impact_score=self._calculate_market_impact(text_content),
                        relevance_score=self._calculate_relevance(text_content),
                        article_id=post_data.get('id', ''),
                        author=post_data.get('author', '')
                    )
                    
                    articles.append(article)
            
            return articles
        
        except Exception as e:
            logger.error(f"Error fetching Reddit finance: {e}")
            return []
    
    def _fetch_yahoo_rss(self) -> List[NewsArticle]:
        """Fetch news from Yahoo Finance RSS with rate limiting and caching."""
        try:
            # Use a small diverse set of "market-wide" anchors
            anchors = ['SPY', 'QQQ', 'TLT', '^VIX', 'CL=F', 'GC=F']
            articles: List[NewsArticle] = []
            base_url = self.news_sources['yahoo_finance_rss']['url']

            for a in anchors:
                params = {
                    's': a,
                    'region': 'US',
                    'lang': 'en-US'
                }
                
                response = self.api_handler.fetch_with_retry(
                    url=base_url,
                    service='yahoo',
                    params=params,
                    max_retries=3,
                    timeout=10
                )
                
                if not response:
                    logger.warning(f"Failed to fetch Yahoo RSS for {a}")
                    continue

                root = ET.fromstring(response.text)
                for item in root.findall('.//item')[:25]:
                    title = (item.findtext('title') or '').strip()
                    link = (item.findtext('link') or '').strip()
                    desc = (item.findtext('description') or '').strip()
                    pub = (item.findtext('pubDate') or '').strip()

                    if not title and not desc:
                        continue

                    try:
                        published_at = parsedate_to_datetime(pub) if pub else datetime.now()
                    except Exception:
                        published_at = datetime.now()

                    text_content = f"{title} {desc}".strip()
                    symbols = self._extract_symbols(text_content)
                    sentiment = self.analyze_sentiment(text_content)

                    article = NewsArticle(
                        title=title or f"Yahoo Finance Update ({a})",
                        summary=desc,
                        url=link,
                        source='Yahoo Finance RSS',
                        published_at=published_at,
                        symbols_mentioned=symbols,
                        sentiment_score=sentiment.overall_sentiment,
                        sentiment_category=sentiment.sentiment_category,
                        market_impact_score=self._calculate_market_impact(text_content),
                        relevance_score=self._calculate_relevance(text_content),
                        article_id=hashlib.md5((link or title).encode()).hexdigest(),
                        author=None,
                    )
                    articles.append(article)

                # Rate limiting handled by api_handler
                time.sleep(0.2)

            return articles
        except Exception as e:
            logger.error(f"Error fetching Yahoo RSS: {e}")
            return []
    
    def _process_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Process and deduplicate articles."""
        processed = []
        seen_urls = set()
        
        for article in articles:
            # Skip duplicates
            if article.url in seen_urls:
                continue
            seen_urls.add(article.url)
            
            # Enhanced sentiment analysis if not already done
            if article.sentiment_score == 0.0:
                text_content = article.title + ' ' + article.summary
                sentiment_analysis = self.analyze_sentiment(text_content)
                article.sentiment_score = sentiment_analysis.overall_sentiment
                article.sentiment_category = sentiment_analysis.sentiment_category
            
            # Extract symbols if not already done
            if not article.symbols_mentioned:
                text_content = article.title + ' ' + article.summary
                article.symbols_mentioned = self._extract_symbols(text_content)
            
            # Calculate scores if not already done
            if article.market_impact_score == 0.0:
                text_content = article.title + ' ' + article.summary
                article.market_impact_score = self._calculate_market_impact(text_content)
                article.relevance_score = self._calculate_relevance(text_content)
            
            processed.append(article)
        
        return processed
    
    def analyze_sentiment(self, text: str) -> SentimentAnalysis:
        """Comprehensive sentiment analysis of text."""
        if not text.strip():
            return SentimentAnalysis(
                overall_sentiment=0.0,
                sentiment_category=SentimentScore.NEUTRAL,
                confidence=0.0,
                key_phrases=[],
                emotional_indicators={},
                market_relevance=0.0
            )
        
        # NLTK VADER sentiment analysis
        sentiment_score = 0.0
        confidence = 0.0
        emotional_indicators = {}
        
        if NLTK_AVAILABLE and self.sentiment_analyzer:
            scores = self.sentiment_analyzer.polarity_scores(text)
            sentiment_score = scores['compound']
            confidence = max(abs(scores['pos']), abs(scores['neg']), abs(scores['neu']))
            emotional_indicators = {
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu']
            }
        else:
            # Fallback: simple keyword-based sentiment
            sentiment_score = self._simple_sentiment_analysis(text)
            confidence = 0.5
        
        # Extract key phrases
        key_phrases = self._extract_key_phrases(text)
        
        # Calculate market relevance
        market_relevance = self._calculate_relevance(text)
        
        return SentimentAnalysis(
            overall_sentiment=sentiment_score,
            sentiment_category=self._score_to_category(sentiment_score),
            confidence=confidence,
            key_phrases=key_phrases,
            emotional_indicators=emotional_indicators,
            market_relevance=market_relevance
        )
    
    def _simple_sentiment_analysis(self, text: str) -> float:
        """Simple keyword-based sentiment analysis fallback."""
        positive_words = [
            'bullish', 'buy', 'strong', 'growth', 'profit', 'gain', 'rise', 'up',
            'positive', 'good', 'excellent', 'beat', 'exceed', 'outperform'
        ]
        
        negative_words = [
            'bearish', 'sell', 'weak', 'decline', 'loss', 'fall', 'down',
            'negative', 'bad', 'poor', 'miss', 'underperform', 'crash'
        ]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        # Normalize to [-1, 1] range
        sentiment = (positive_count - negative_count) / max(total_words / 10, 1)
        return max(-1.0, min(1.0, sentiment))
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text with improved filtering."""
        phrases = []
        text_lower = text.lower()
        
        # Financial terms - explicit whitelist
        financial_terms = [
            'earnings', 'revenue', 'profit', 'loss', 'guidance', 'outlook',
            'merger', 'acquisition', 'dividend', 'buyback', 'split',
            'inflation', 'recession', 'rates', 'fed', 'fomc', 'yields',
            'volatility', 'correction', 'rally', 'breakout', 'support', 'resistance',
            'supply chain', 'shortage', 'inventory', 'margin', 'valuation',
            'upgrade', 'downgrade', 'analyst', 'target price'
        ]
        
        for term in financial_terms:
            if term in text_lower:
                phrases.append(term.title())
        
        # Extract capitalized phrases but filter out common noise
        # This regex matches consecutive capitalized words
        capitalized_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
        
        # Blocklist for common forum/social noise
        blocklist = {
            'Daily', 'General', 'Discussion', 'Thread', 'Advice', 'Question', 'Help',
            'February', 'January', 'March', 'April', 'May', 'June', 'July',
            'August', 'September', 'October', 'November', 'December',
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
            'Morning', 'Evening', 'Night', 'Weekend', 'Weekly', 'Monthly',
            'Moves', 'Tomorrow', 'Today', 'Yesterday', 'What', 'How', 'Why',
            'Remember', 'There', 'Here', 'This', 'That', 'The', 'A', 'An',
            'Great', 'Good', 'Bad', 'Best', 'Worst', 'Top', 'Bottom',
            'Market', 'Stock', 'Stocks', 'Share', 'Shares', 'Price', 'Value',
            'Key', 'Theme', 'Themes', 'News', 'Update', 'Alert', 'Report',
            'Want', 'Have', 'Like', 'Just', 'Going', 'Get', 'Got', 'Make', 'Made',
            'Know', 'Think', 'Thought', 'Need', 'See', 'Saw', 'Look', 'Looking',
            'Bear', 'Bull', 'Mover', 'Movers', 'Prediction', 'Predictions'
        }
        
        # Phrases to explicitly ignore (case-insensitive check)
        ignored_phrases = {
            'daily discussion', 'weekend discussion', 'daily general discussion',
            'advice thread', 'moronic monday', 'moves tomorrow', 'what are your moves',
            'the great bear', 'moves there', 'remember the', 'market moves'
        }
        
        # Filter phrases
        valid_cap_phrases = []
        for phrase in capitalized_phrases:
            phrase_clean = phrase.strip()
            if not phrase_clean:
                continue
                
            # Check explicit ignored phrases
            if phrase_clean.lower() in ignored_phrases:
                continue
            
            words = phrase_clean.split()
            # Skip if any word is in blocklist
            if any(w in blocklist for w in words):
                continue
            # Skip if phrase is just common words (e.g. "The Market")
            if len(words) < 2:
                continue
            valid_cap_phrases.append(phrase_clean)
            
        phrases.extend(valid_cap_phrases[:5])
        
        # If we have NLTK, try to extract noun phrases (better quality)
        if NLTK_AVAILABLE:
            try:
                tokens = nltk.word_tokenize(text)
                tagged = nltk.pos_tag(tokens)
                # Extract NNP (Proper Noun) sequences
                chunk_gram = r"""Chunk: {<NNP>+}"""
                chunk_parser = nltk.RegexpParser(chunk_gram)
                chunked = chunk_parser.parse(tagged)
                
                for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Chunk'):
                    phrase = ' '.join([leaf[0] for leaf in subtree.leaves()])
                    if len(phrase.split()) > 1 and not any(w in blocklist for w in phrase.split()):
                        phrases.append(phrase)
            except:
                pass

        return list(dict.fromkeys(phrases))[:10]  # Deduplicate and limit
    
    def _extract_symbols(self, text: str) -> List[str]:
        """Extract stock symbols from text."""
        symbols = set()
        
        for pattern in self.symbol_patterns:
            matches = re.findall(pattern, text)
            symbols.update(matches)
        
        # Filter out common words
        common_words = {
            'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN',
            'HAS', 'HAD', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET',
            'USE', 'MAN', 'NEW', 'NOW', 'WAY', 'MAY', 'SAY', 'EACH', 'WHICH'
        }

        currency_codes = {
            "EUR", "USD", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD", "SEK", "NOK", "DKK",
            "PLN", "CZK", "HUF", "TRY", "ZAR", "MXN", "BRL", "CNY", "INR", "KRW", "SGD",
            "HKD", "THB", "MYR", "IDR", "PHP",
        }

        cleaned = []
        for s in symbols:
            token = s.upper().replace('$', '')
            if token in common_words:
                continue
            if '/' in token or '-' in token or '_' in token:
                parts = re.split(r'[\/_-]', token)
                if len(parts) == 2 and parts[0] in currency_codes and parts[1] in currency_codes:
                    cleaned.append(f"{parts[0]}/{parts[1]}")
                continue
            cleaned.append(token)

        # Deduplicate while preserving order
        return list(dict.fromkeys(cleaned))[:10]  # Limit to 10 symbols
    
    def _calculate_market_impact(self, text: str) -> float:
        """Calculate potential market impact score."""
        text_lower = text.lower()
        impact_score = 0.0
        
        # High impact keywords
        for keyword in self.market_keywords['high_impact']:
            if keyword in text_lower:
                impact_score += 0.3
        
        # Medium impact keywords
        for keyword in self.market_keywords['medium_impact']:
            if keyword in text_lower:
                impact_score += 0.2
        
        # Low impact keywords
        for keyword in self.market_keywords['low_impact']:
            if keyword in text_lower:
                impact_score += 0.1
        
        # Normalize to [0, 1] range
        return min(1.0, impact_score)
    
    def _calculate_relevance(self, text: str) -> float:
        """Calculate market relevance score."""
        text_lower = text.lower()
        relevance_score = 0.0
        
        # Financial terms boost relevance
        financial_terms = [
            'stock', 'market', 'trading', 'investment', 'portfolio',
            'analyst', 'forecast', 'price', 'valuation', 'financial'
        ]
        
        for term in financial_terms:
            if term in text_lower:
                relevance_score += 0.1
        
        # Symbol mentions boost relevance
        symbols = self._extract_symbols(text)
        relevance_score += len(symbols) * 0.05
        
        # Normalize to [0, 1] range
        return min(1.0, relevance_score)
    
    def _score_to_category(self, score: float) -> SentimentScore:
        """Convert numerical sentiment score to category."""
        if score <= -0.6:
            return SentimentScore.VERY_BEARISH
        elif score <= -0.2:
            return SentimentScore.BEARISH
        elif score >= 0.6:
            return SentimentScore.VERY_BULLISH
        elif score >= 0.2:
            return SentimentScore.BULLISH
        else:
            return SentimentScore.NEUTRAL
    
    def _store_articles(self, articles: List[NewsArticle]):
        """Store articles in database."""
        try:
            # This would integrate with the database manager
            # For now, we'll store in a simple format
            for article in articles:
                # Store article data
                article_data = {
                    'article_id': article.article_id,
                    'title': article.title,
                    'summary': article.summary,
                    'url': article.url,
                    'source': article.source,
                    'published_at': article.published_at.isoformat(),
                    'symbols_mentioned': json.dumps(article.symbols_mentioned),
                    'sentiment_score': article.sentiment_score,
                    'sentiment_category': article.sentiment_category.value,
                    'market_impact_score': article.market_impact_score,
                    'relevance_score': article.relevance_score,
                    'author': article.author,
                    'created_at': datetime.now().isoformat()
                }
                
                # In a real implementation, this would use the database manager
                # self.db_manager.store_news_article(article_data)
        
        except Exception as e:
            logger.error(f"Error storing articles: {e}")

    def _get_recent_articles(self, hours_back: int = 24) -> List[NewsArticle]:
        try:
            # Force refresh to ensure we have latest data
            articles = self.fetch_and_process_news() or []
            
            # If we don't have enough articles (less than 50), try to fetch more or generate safe fallbacks for analysis
            if len(articles) < 50:
                # In a real scenario, this would trigger a deeper historical fetch
                pass
                
            cutoff = _naive_utc(datetime.now(timezone.utc)) - timedelta(hours=hours_back)
            out = []
            for a in articles:
                try:
                    pub = _naive_utc(a.published_at)
                    if pub and pub >= cutoff:
                        out.append(a)
                except Exception:
                    continue
            return out
        except Exception:
            return []

    def get_market_sentiment(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get aggregated sentiment across the whole market from recent articles."""
        try:
            articles = self._get_recent_articles(hours_back=hours_back)
            if not articles:
                return {}

            sentiments = [float(a.sentiment_score or 0) for a in articles]
            overall = float(np.mean(sentiments)) if sentiments else 0.0
            sources = sorted(list({a.source for a in articles if a.source}))
            avg_impact = float(np.mean([float(a.market_impact_score or 0) for a in articles]))

            themes = []
            for a in articles[:100]:
                themes.extend(self._extract_key_phrases((a.title or '') + ' ' + (a.summary or '')))
            themes = list(dict.fromkeys([t for t in themes if t]))[:10]

            return {
                'symbol': 'MARKET',
                'overall_sentiment': overall,
                'sentiment_category': self._score_to_category(overall).name,
                'confidence': min(0.95, max(0.35, len(articles) / 60)),
                'article_count': len(articles),
                'sources': sources,
                'key_themes': themes,
                'sentiment_trend': {
                    'last_24h': overall,
                    'last_7d': float(np.mean([float(a.sentiment_score or 0) for a in self._get_recent_articles(hours_back=24*7)] or [0.0])),
                    'last_30d': float(np.mean([float(a.sentiment_score or 0) for a in self._get_recent_articles(hours_back=24*30)] or [0.0])),
                },
                'market_impact_score': avg_impact,
                'timestamp': datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error getting market sentiment: {e}")
            return {}
    
    def get_sentiment_for_symbol(self, symbol: str, hours_back: int = 24) -> Dict[str, Any]:
        """Get aggregated sentiment analysis for a specific symbol."""
        try:
            sym = (symbol or '').upper().strip()
            if sym in ('MARKET', 'ALL'):
                return self.get_market_sentiment(hours_back=hours_back)

            articles = self._get_recent_articles(hours_back=hours_back)
            relevant = [a for a in articles if sym in (a.symbols_mentioned or [])]
            if not relevant:
                return {}

            sentiments = [float(a.sentiment_score or 0) for a in relevant]
            overall = float(np.mean(sentiments)) if sentiments else 0.0
            sources = sorted(list({a.source for a in relevant if a.source}))
            avg_impact = float(np.mean([float(a.market_impact_score or 0) for a in relevant]))

            themes = []
            for a in relevant[:80]:
                themes.extend(self._extract_key_phrases((a.title or '') + ' ' + (a.summary or '')))
            themes = list(dict.fromkeys([t for t in themes if t]))[:10]

            return {
                'symbol': sym,
                'overall_sentiment': overall,
                'sentiment_category': self._score_to_category(overall).name,
                'confidence': min(0.95, max(0.35, len(relevant) / 30)),
                'article_count': len(relevant),
                'sources': sources,
                'key_themes': themes,
                'sentiment_trend': {
                    'last_24h': overall,
                    'last_7d': float(np.mean([float(a.sentiment_score or 0) for a in self._get_recent_articles(hours_back=24*7) if sym in (a.symbols_mentioned or [])] or [0.0])),
                    'last_30d': float(np.mean([float(a.sentiment_score or 0) for a in self._get_recent_articles(hours_back=24*30) if sym in (a.symbols_mentioned or [])] or [0.0])),
                },
                'market_impact_score': avg_impact,
                'timestamp': datetime.now().isoformat(),
            }
        
        except Exception as e:
            logger.error(f"Error getting sentiment for {symbol}: {e}")
            return {}
    
    def get_market_whispers(self, symbol: Optional[str] = None) -> List[MarketWhisper]:
        """Market whispers derived from REAL fetched news coverage.

        Every returned whisper is grounded in an actual fetched article:
        - content = the real headline/summary
        - source  = the real publishing outlet
        - timestamp = the real publication time
        - confidence = sentiment-strength × relevance (both computed from the
          article itself, never invented)
        - verification = 'verified' when >= 3 distinct outlets cover the same
          symbol (corroboration), 'partially_verified' at 2, else 'unverified'

        No fabricated rumors, fake sources, or invented mention counts are ever
        produced.  When there is no real coverage the result is an empty list,
        and the UI already renders the honest empty state.
        """
        whispers: List[MarketWhisper] = []
        try:
            articles = self.fetch_and_process_news() or []
        except Exception as e:
            logger.error(f"get_market_whispers: could not fetch news: {e}")
            return []
        if not articles:
            return []

        sym = (symbol or "").upper().strip()
        # Symbols we focus on: the requested one (if any) plus the most-covered
        # names in the real fetched corpus — not a hardcoded universe.
        coverage_counts: Dict[str, int] = {}
        for a in articles:
            for s in (a.symbols_mentioned or []):
                s = str(s).upper().strip()
                if s:
                    coverage_counts[s] = coverage_counts.get(s, 0) + 1
        top_covered = sorted(coverage_counts, key=coverage_counts.get, reverse=True)[:8]
        if sym:
            target_symbols = [sym] + [s for s in top_covered if s != sym][:3]
        else:
            target_symbols = top_covered or []
        if not target_symbols:
            return []

        relevant = [
            a for a in articles
            if a.symbols_mentioned and any(
                str(s).upper().strip() in target_symbols for s in a.symbols_mentioned
            )
        ]
        # Most recent first, cap for performance
        relevant = sorted(
            relevant,
            key=lambda a: _naive_utc(a.published_at) or datetime.min,
            reverse=True,
        )[:40]

        for a in relevant:
            mentioned = [
                str(s).upper().strip() for s in (a.symbols_mentioned or [])
                if str(s).upper().strip() in target_symbols
            ]
            # Corroboration: how many distinct outlets cover this symbol
            distinct_outlets = set()
            for other in articles:
                if not (other.symbols_mentioned or []):
                    continue
                if any(str(s).upper().strip() in mentioned for s in other.symbols_mentioned):
                    if other.source:
                        distinct_outlets.add(other.source)
            n_outlets = len(distinct_outlets)
            if n_outlets >= 3:
                verification = "verified"
            elif n_outlets == 2:
                verification = "partially_verified"
            else:
                verification = "unverified"

            sentiment = float(a.sentiment_score or 0.0)
            relevance = float(a.relevance_score or 0.5)
            # Confidence: stronger sentiment + higher relevance => higher
            confidence = min(0.95, max(0.2, abs(sentiment) * 0.5 + relevance * 0.5))

            title = (a.title or "").strip()
            summary = (a.summary or "").strip()
            content = f"{title}. {summary}".strip()
            if not content:
                continue

            # Whisper type inferred from tags/title keywords, else generic
            wtype = self._infer_whisper_type(a)

            # Real social_mentions are not available; we never fabricate a
            # number.  coverage proxy = number of real articles on this symbol.
            real_coverage = sum(
                1 for other in articles
                if (other.symbols_mentioned or []) and any(
                    str(s).upper().strip() in mentioned for s in other.symbols_mentioned
                )
            )

            whispers.append(MarketWhisper(
                content=content,
                source=a.source or "Unknown",
                confidence_level=confidence,
                symbols_mentioned=mentioned,
                whisper_type=wtype,
                timestamp=a.published_at or datetime.now(),
                social_mentions=real_coverage,
                verification_status=verification,
            ))

        # De-duplicate near-identical headlines
        seen_titles: set = set()
        unique: List[MarketWhisper] = []
        for w in whispers:
            key = w.content[:80].lower()
            if key in seen_titles:
                continue
            seen_titles.add(key)
            unique.append(w)

        unique.sort(key=lambda w: _naive_utc(w.timestamp) or datetime.min, reverse=True)
        return unique[:15]

    def _infer_whisper_type(self, article: "NewsArticle") -> str:
        """Infer a whisper category from article tags/keywords (real content)."""
        hay = " ".join([
            article.title or "", article.summary or "",
            " ".join(article.tags or []),
        ]).lower()
        for kw, wtype in [
            (("earnings", "eps", "guidance", "revenue", "quarter"), "earnings"),
            (("merger", "acquisition", "takeover", "bid", "acquir"), "merger"),
            (("sue", "lawsuit", "regulat", "probe", "compliance", "antitrust"), "regulatory"),
            (("product", "launch", "patent", "approval", "device"), "product"),
            (("inflation", "fed", "rate", "treasury", "yield", "recession"), "macro"),
            (("option", "hedge", "flow", "volatility", "gamma"), "market_structure"),
            (("sector", "rotation", "industry", "outperform"), "sector_news"),
        ]:
            if any(k in hay for k in kw):
                return wtype
        return "other"
    
    def get_news_summary_for_symbol(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive news summary for a symbol."""
        try:
            sym = (symbol or '').upper().strip()
            sentiment_data = self.get_sentiment_for_symbol(sym)
            whispers = self.get_market_whispers(sym)

            fetched = self.fetch_and_process_news() or []
            relevant = [a for a in fetched if sym in (a.symbols_mentioned or [])]
            relevant = sorted(
                relevant,
                key=lambda a: _naive_utc(a.published_at) or datetime.min,
                reverse=True,
            )[:20]
            recent_articles = [
                {
                    'title': a.title,
                    'summary': a.summary,
                    'sentiment_score': float(a.sentiment_score or 0),
                    'published_at': a.published_at.isoformat() if a.published_at else datetime.now().isoformat(),
                    'source': a.source,
                    'url': a.url,
                }
                for a in relevant
            ]
            
            summary = {
                'symbol': sym,
                'sentiment_analysis': sentiment_data,
                'recent_articles': recent_articles,
                'market_whispers': [
                    {
                        'content': w.content,
                        'confidence': w.confidence_level,
                        'type': w.whisper_type,
                        'mentions': w.social_mentions
                    } for w in whispers
                ],
                'key_insights': [
                    f'Overall sentiment for {symbol} is {sentiment_data.get("sentiment_category", "NEUTRAL").lower()}',
                    f'Market impact score: {sentiment_data.get("market_impact_score", 0.5):.2f}/1.0',
                    f'Based on {sentiment_data.get("article_count", 0)} recent articles'
                ],
                'timestamp': datetime.now().isoformat()
            }
            
            return summary
        
        except Exception as e:
            logger.error(f"Error generating news summary for {symbol}: {e}")
            return {}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get news analysis engine metrics."""
        return {
            'metrics': self.metrics.copy(),
            'sources_enabled': sum(1 for config in self.news_sources.values() if config.get('enabled', False)),
            'cache_size': len(self.article_cache),
            'timestamp': datetime.now().isoformat()
        }
    
    def stop(self):
        """Stop background processing."""
        self.running = False
        logger.info("News analysis engine stopped")

# Global instance
_news_engine = None

def get_news_engine() -> NewsAnalysisEngine:
    """Get or create news analysis engine instance."""
    global _news_engine
    if _news_engine is None:
        _news_engine = NewsAnalysisEngine()
    return _news_engine