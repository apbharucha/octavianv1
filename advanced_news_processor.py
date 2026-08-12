"""
Advanced News Processing Engine

This module processes hundreds of news articles and market rumors to provide:
1. Comprehensive news aggregation from multiple sources
2. Advanced sentiment analysis and market impact assessment
3. Cross-sector correlation analysis
4. Anticipation factor implementation
5. Narrative generation comparing AI analysis vs market sentiment
6. Decision-making implications analysis

Author: AI Market Team
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import json
from typing import Dict, List, Optional, Any, Tuple
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict
import hashlib

from news_analysis_engine import SentimentScore # NEW IMPORT

# NLP and sentiment analysis
NLTK_AVAILABLE = False
try:
    from textblob import TextBlob
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer
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
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
            
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', download_dir=nltk_data_path, quiet=True)
            
    except Exception as e:
        print(f"Warning: NLTK data check/download failed: {e}")

# NLTK data is initialized on first use (not at import time) to speed up app startup
_nltk_data_ready = False

def _lazy_ensure_nltk():
    global _nltk_data_ready
    if not _nltk_data_ready and NLTK_AVAILABLE:
        _ensure_nltk_data()
        _nltk_data_ready = True

# Web scraping and RSS
try:
    from bs4 import BeautifulSoup
    import feedparser
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    print("BeautifulSoup/feedparser not available - limited web scraping")

from database_manager import get_database_manager
from config import ALPHA_VANTAGE_KEY
from api_rate_limiter import get_api_handler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessedArticle:
    """Enhanced article data structure with cross-sector analysis."""
    article_id: str
    title: str
    content: str
    url: str
    source: str
    published_at: datetime
    symbols_mentioned: List[str]
    sectors_mentioned: List[str]
    countries_mentioned: List[str]
    sentiment_score: float
    sentiment_category: SentimentScore
    market_impact_score: float
    relevance_score: float
    cross_sector_implications: Dict[str, float]
    anticipation_factors: Dict[str, float]
    narrative_elements: List[str]
    decision_implications: List[str]
    processed_at: datetime

@dataclass
class MarketNarrative:
    """Market narrative comparing AI analysis vs market sentiment."""
    symbol: str
    ai_analysis: Dict[str, Any]
    market_sentiment: Dict[str, Any]
    narrative_comparison: str
    confidence_divergence: float
    decision_implications: List[str]
    anticipation_factors: Dict[str, float]
    timestamp: datetime

class AdvancedNewsProcessor:
    """Advanced news processing engine with cross-sector analysis and anticipation factors."""
    
    def __init__(self):
        _lazy_ensure_nltk()
        self.db_manager = get_database_manager()
        self.api_handler = get_api_handler()
        
        # Enhanced news sources with RSS feeds and APIs
        self.news_sources = {
            'financial_apis': {
                'alpha_vantage': {
                    'url': 'https://www.alphavantage.co/query',
                    'api_key': ALPHA_VANTAGE_KEY,
                    'enabled': bool(ALPHA_VANTAGE_KEY),
                    'rate_limit': 5  # calls per minute
                },
                'newsapi': {
                    'url': 'https://newsapi.org/v2/everything',
                    'api_key': None,  # Would need API key
                    'enabled': False,
                    'rate_limit': 100
                }
            },
            'rss_feeds': {
                'reuters_business': 'http://feeds.reuters.com/reuters/businessNews',
                'bloomberg_markets': 'https://feeds.bloomberg.com/markets/news.rss',
                'cnbc_markets': 'https://www.cnbc.com/id/10000664/device/rss/rss.html',
                'marketwatch': 'http://feeds.marketwatch.com/marketwatch/marketpulse/',
                'yahoo_finance': 'https://feeds.finance.yahoo.com/rss/2.0/headline',
                'seeking_alpha': 'https://seekingalpha.com/market_currents.xml',
                'zerohedge': 'http://feeds.feedburner.com/zerohedge/feed'
            },
            'social_sources': {
                'reddit_investing': 'https://www.reddit.com/r/investing.json',
                'reddit_stocks': 'https://www.reddit.com/r/stocks.json',
                'reddit_wallstreetbets': 'https://www.reddit.com/r/wallstreetbets.json',
                'stocktwits_trending': 'https://api.stocktwits.com/api/2/trending/symbols.json'
            }
        }
        
        # Cross-sector mapping
        self.sector_mappings = {
            'technology': ['XLK', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA'],
            'finance': ['XLF', 'KBE', 'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C'],
            'healthcare': ['XLV', 'IBB', 'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO'],
            'energy': ['XLE', 'USO', 'XOM', 'CVX', 'COP', 'EOG', 'SLB'],
            'consumer': ['XLY', 'XLP', 'PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD'],
            'industrials': ['XLI', 'IYT', 'BA', 'CAT', 'GE', 'HON', 'UPS', 'LMT'],
            'materials': ['XLB', 'GLD', 'SLV', 'LIN', 'FCX', 'NEM', 'DOW'],
            'utilities': ['XLU', 'NEE', 'DUK', 'SO', 'AEP', 'EXC'],
            'real_estate': ['XLRE', 'VNQ', 'AMT', 'PLD', 'CCI', 'EQIX', 'O'],
            'volatility': ['^VIX', 'VXX', 'UVXY', 'SVXY'],
            'rates': ['TLT', 'IEF', 'SHY', 'LQD', 'HYG'],
            'index_futures': ['ES=F', 'NQ=F', 'YM=F', 'RTY=F'],
            'commodities': ['CL=F', 'GC=F', 'SI=F']
        }
        
        # Country/region mappings for geopolitical analysis
        self.country_mappings = {
            'united_states': ['USD', 'SPY', 'QQQ', 'DIA', 'ES=F'],
            'europe': ['EUR/USD', 'GBP/USD', 'EWU', 'EWG', 'EWI'],
            'asia': ['USD/JPY', 'USD/CNY', 'EWJ', 'FXI', 'EWY'],
            'emerging_markets': ['EEM', 'VWO', 'EWZ', 'INDA', 'USD/MXN', 'USD/ZAR'],
            'commodities': ['GLD', 'SLV', 'USO', 'UNG', 'DBA', 'CL=F', 'GC=F']
        }
        
        # Enhanced keyword dictionaries
        self.market_keywords = {
            'high_impact': [
                'earnings', 'revenue', 'profit', 'loss', 'guidance', 'outlook',
                'merger', 'acquisition', 'ipo', 'bankruptcy', 'dividend', 'buyback',
                'fed', 'federal reserve', 'interest rate', 'inflation', 'gdp',
                'unemployment', 'jobs report', 'cpi', 'ppi', 'fomc'
            ],
            'geopolitical': [
                'trade war', 'tariff', 'sanctions', 'brexit', 'election',
                'war', 'conflict', 'treaty', 'agreement', 'policy',
                'regulation', 'investigation', 'lawsuit', 'settlement'
            ],
            'sector_rotation': [
                'rotation', 'outperform', 'underperform', 'leadership',
                'defensive', 'cyclical', 'growth', 'value', 'momentum'
            ],
            'anticipation_signals': [
                'expects', 'anticipates', 'forecasts', 'predicts', 'outlook',
                'guidance', 'estimates', 'projections', 'targets', 'goals'
            ]
        }
        
        # Initialize NLP components
        if NLTK_AVAILABLE:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        
        # Processing metrics
        self.processing_metrics = {
            'articles_processed': 0,
            'sources_active': 0,
            'processing_rate': 0,
            'error_rate': 0,
            'last_update': None
        }
        
        # Article cache and deduplication
        self.article_cache = {}
        self.processed_urls = set()
        self.cache_expiry = 3600  # 1 hour
    
    async def process_news_comprehensive(self, max_articles: int = 500) -> List[ProcessedArticle]:
        """Process hundreds of news articles comprehensively with cross-sector analysis."""
        start_time = time.time()
        all_articles = []
        
        try:
            # Fetch from all sources concurrently
            tasks = []
            
            # RSS feeds
            for source_name, rss_url in self.news_sources['rss_feeds'].items():
                tasks.append(self._fetch_rss_articles(source_name, rss_url))
            
            # API sources
            for source_name, config in self.news_sources['financial_apis'].items():
                if config.get('enabled', False):
                    tasks.append(self._fetch_api_articles(source_name, config))
            
            # Social sources
            for source_name, url in self.news_sources['social_sources'].items():
                tasks.append(self._fetch_social_articles(source_name, url))
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect articles from all sources
            for result in results:
                if isinstance(result, list):
                    all_articles.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"Error fetching articles: {result}")
            
            # Remove duplicates and limit to max_articles
            unique_articles = self._deduplicate_articles(all_articles)
            limited_articles = unique_articles[:max_articles]
            
            # Process articles with advanced analysis
            processed_articles = []
            
            # Process in batches for better performance
            batch_size = 50
            for i in range(0, len(limited_articles), batch_size):
                batch = limited_articles[i:i + batch_size]
                batch_processed = await self._process_article_batch(batch)
                processed_articles.extend(batch_processed)
            
            # Update metrics
            processing_time = time.time() - start_time
            self.processing_metrics.update({
                'articles_processed': len(processed_articles),
                'sources_active': len([r for r in results if isinstance(r, list) and r]),
                'processing_rate': len(processed_articles) / processing_time if processing_time > 0 else 0,
                'last_update': datetime.now()
            })
            
            logger.info(f"Processed {len(processed_articles)} articles in {processing_time:.2f}s")
            return processed_articles
            
        except Exception as e:
            logger.error(f"Comprehensive news processing error: {e}")
            return []
    
    async def _fetch_rss_articles(self, source_name: str, rss_url: str) -> List[Dict[str, Any]]:
        """Fetch articles from RSS feeds."""
        if not BS4_AVAILABLE:
            return []
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(rss_url) as response:
                    if response.status == 200:
                        content = await response.text()
                        feed = feedparser.parse(content)
                        
                        articles = []
                        for entry in feed.entries[:50]:  # Limit per source
                            article = {
                                'title': entry.get('title', ''),
                                'content': entry.get('summary', entry.get('description', '')),
                                'url': entry.get('link', ''),
                                'source': source_name,
                                'published_at': self._parse_date(entry.get('published', '')),
                                'raw_entry': entry
                            }
                            articles.append(article)
                        
                        return articles
            
        except Exception as e:
            logger.error(f"RSS fetch error for {source_name}: {e}")
            return []
    
    async def _fetch_api_articles(self, source_name: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch articles from API sources."""
        try:
            if source_name == 'alpha_vantage' and config.get('api_key'):
                return await self._fetch_alpha_vantage_news(config)
            elif source_name == 'newsapi' and config.get('api_key'):
                return await self._fetch_newsapi_articles(config)
            else:
                return []
        
        except Exception as e:
            logger.error(f"API fetch error for {source_name}: {e}")
            return []
    
    async def _fetch_alpha_vantage_news(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch news from Alpha Vantage API."""
        try:
            url = config['url']
            params = {
                'function': 'NEWS_SENTIMENT',
                'apikey': config['api_key'],
                'limit': 200,  # Increased limit
                'sort': 'LATEST'
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        articles = []
                        if 'feed' in data:
                            for item in data['feed']:
                                article = {
                                    'title': item.get('title', ''),
                                    'content': item.get('summary', ''),
                                    'url': item.get('url', ''),
                                    'source': f"Alpha Vantage - {item.get('source', 'Unknown')}",
                                    'published_at': self._parse_alpha_vantage_date(item.get('time_published', '')),
                                    'symbols_mentioned': [ts['ticker'] for ts in item.get('ticker_sentiment', [])],
                                    'sentiment_score': float(item.get('overall_sentiment_score', 0)),
                                    'raw_item': item
                                }
                                articles.append(article)
                        
                        return articles
            
        except Exception as e:
            logger.error(f"Alpha Vantage news fetch error: {e}")
            return []
    
    async def _fetch_social_articles(self, source_name: str, url: str) -> List[Dict[str, Any]]:
        """Fetch articles from social sources."""
        try:
            headers = {'User-Agent': 'MarketAI/1.0'}
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        articles = []
                        
                        if 'reddit' in source_name and 'data' in data:
                            for post in data['data']['children'][:25]:  # Limit Reddit posts
                                post_data = post['data']
                                article = {
                                    'title': post_data.get('title', ''),
                                    'content': post_data.get('selftext', '')[:1000],  # Limit content
                                    'url': f"https://reddit.com{post_data.get('permalink', '')}",
                                    'source': f"Reddit - {source_name.split('_')[-1]}",
                                    'published_at': datetime.fromtimestamp(post_data.get('created_utc', 0)),
                                    'upvotes': post_data.get('ups', 0),
                                    'comments': post_data.get('num_comments', 0),
                                    'raw_post': post_data
                                }
                                articles.append(article)
                        
                        return articles
            
        except Exception as e:
            logger.error(f"Social fetch error for {source_name}: {e}")
            return []
    
    def _deduplicate_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate articles based on URL and title similarity."""
        seen_urls = set()
        seen_titles = set()
        unique_articles = []
        
        for article in articles:
            url = article.get('url', '')
            title = article.get('title', '').lower().strip()
            
            # Skip if URL already seen
            if url and url in seen_urls:
                continue
            
            # Skip if very similar title already seen
            title_hash = hashlib.md5(title.encode()).hexdigest()
            if title_hash in seen_titles:
                continue
            
            # Skip if URL already processed recently
            if url in self.processed_urls:
                continue
            
            seen_urls.add(url)
            seen_titles.add(title_hash)
            unique_articles.append(article)
        
        return unique_articles
    
    async def _process_article_batch(self, articles: List[Dict[str, Any]]) -> List[ProcessedArticle]:
        """Process a batch of articles with advanced analysis."""
        processed_articles = []
        
        for article in articles:
            try:
                processed = await self._process_single_article(article)
                if processed:
                    processed_articles.append(processed)
            except Exception as e:
                logger.error(f"Error processing article: {e}")
        
        return processed_articles
    
    async def _process_single_article(self, article: Dict[str, Any]) -> Optional[ProcessedArticle]:
        """Process a single article with comprehensive analysis."""
        try:
            title = article.get('title', '')
            content = article.get('content', '')
            full_text = f"{title} {content}"
            
            if not full_text.strip():
                return None
            
            # Generate article ID
            article_id = hashlib.md5(f"{title}{article.get('url', '')}".encode()).hexdigest()
            
            # Extract symbols, sectors, and countries
            symbols = self._extract_symbols_advanced(full_text)
            sectors = self._extract_sectors(full_text, symbols)
            countries = self._extract_countries(full_text)
            
            # Advanced sentiment analysis
            sentiment_analysis = self._analyze_sentiment_advanced(full_text)
            
            # Calculate market impact and relevance
            market_impact = self._calculate_market_impact_advanced(full_text, symbols, sectors)
            relevance = self._calculate_relevance_advanced(full_text, symbols, sectors)
            
            # Cross-sector implications analysis
            cross_sector_implications = self._analyze_cross_sector_implications(
                full_text, symbols, sectors, sentiment_analysis
            )
            
            # Anticipation factors
            anticipation_factors = self._extract_anticipation_factors(full_text, symbols)
            
            # Narrative elements
            narrative_elements = self._extract_narrative_elements(full_text)
            
            # Decision implications
            decision_implications = self._extract_decision_implications(
                full_text, symbols, sentiment_analysis, anticipation_factors
            )
            
            processed_article = ProcessedArticle(
                article_id=article_id,
                title=title,
                content=content,
                url=article.get('url', ''),
                source=article.get('source', 'Unknown'),
                published_at=article.get('published_at', datetime.now()),
                symbols_mentioned=symbols,
                sectors_mentioned=sectors,
                countries_mentioned=countries,
                sentiment_score=sentiment_analysis['overall_sentiment'],
                sentiment_category=sentiment_analysis['sentiment_category'],
                market_impact_score=market_impact,
                relevance_score=relevance,
                cross_sector_implications=cross_sector_implications,
                anticipation_factors=anticipation_factors,
                narrative_elements=narrative_elements,
                decision_implications=decision_implications,
                processed_at=datetime.now()
            )
            
            # Add to processed URLs
            if article.get('url'):
                self.processed_urls.add(article['url'])
            
            return processed_article
            
        except Exception as e:
            logger.error(f"Single article processing error: {e}")
            return None
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string from various formats."""
        if not date_str:
            return datetime.now()
        
        try:
            # Try common date formats
            formats = [
                '%a, %d %b %Y %H:%M:%S %Z',
                '%a, %d %b %Y %H:%M:%S %z',
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            
            # If all formats fail, return current time
            return datetime.now()
            
        except Exception:
            return datetime.now()
    
    def _parse_alpha_vantage_date(self, date_str: str) -> datetime:
        """Parse Alpha Vantage date format."""
        try:
            if len(date_str) == 15:  # Format: YYYYMMDDTHHMMSS
                return datetime.strptime(date_str, '%Y%m%dT%H%M%S')
            else:
                return self._parse_date(date_str)
        except Exception:
            return datetime.now()
    
    def _analyze_sentiment_advanced(self, text: str) -> Dict[str, Any]:
        """Advanced sentiment analysis with multiple techniques."""
        if not text.strip():
            return {
                'overall_sentiment': 0.0,
                'sentiment_category': 'NEUTRAL',
                'confidence': 0.0,
                'emotional_indicators': {}
            }
        
        try:
            sentiment_scores = []
            
            # NLTK VADER sentiment
            if NLTK_AVAILABLE and self.sentiment_analyzer:
                vader_scores = self.sentiment_analyzer.polarity_scores(text)
                sentiment_scores.append(vader_scores['compound'])
            
            # TextBlob sentiment
            if NLTK_AVAILABLE:
                try:
                    blob = TextBlob(text)
                    sentiment_scores.append(blob.sentiment.polarity)
                except:
                    pass
            
            # Custom financial sentiment
            financial_sentiment = self._financial_sentiment_analysis(text)
            sentiment_scores.append(financial_sentiment)
            
            # Average sentiment scores
            overall_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
            
            # Determine sentiment category
            if overall_sentiment > 0.6:
                category = 'VERY_BULLISH'
            elif overall_sentiment > 0.2:
                category = 'BULLISH'
            elif overall_sentiment > -0.2:
                category = 'NEUTRAL'
            elif overall_sentiment > -0.6:
                category = 'BEARISH'
            else:
                category = 'VERY_BEARISH'
            
            # Calculate confidence
            confidence = min(abs(overall_sentiment) * 1.5, 1.0)
            
            return {
                'overall_sentiment': overall_sentiment,
                'sentiment_category': category,
                'confidence': confidence,
                'emotional_indicators': {
                    'positive': max(0, overall_sentiment),
                    'negative': max(0, -overall_sentiment),
                    'neutral': 1 - abs(overall_sentiment)
                }
            }
            
        except Exception as e:
            logger.error(f"Advanced sentiment analysis error: {e}")
            return {
                'overall_sentiment': 0.0,
                'sentiment_category': 'NEUTRAL',
                'confidence': 0.0,
                'emotional_indicators': {}
            }
    
    def _financial_sentiment_analysis(self, text: str) -> float:
        """Custom financial sentiment analysis."""
        try:
            text_lower = text.lower()
            
            # Financial positive words
            positive_words = [
                'bullish', 'buy', 'strong', 'growth', 'profit', 'gain', 'rise', 'up',
                'positive', 'good', 'excellent', 'beat', 'exceed', 'outperform',
                'upgrade', 'target', 'momentum', 'breakout', 'rally', 'surge'
            ]
            
            # Financial negative words
            negative_words = [
                'bearish', 'sell', 'weak', 'decline', 'loss', 'fall', 'down',
                'negative', 'bad', 'poor', 'miss', 'underperform', 'crash',
                'downgrade', 'correction', 'pullback', 'resistance', 'breakdown'
            ]
            
            # Count occurrences
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            # Calculate sentiment
            total_words = len(text.split())
            if total_words == 0:
                return 0.0
            
            # Normalize sentiment
            sentiment = (positive_count - negative_count) / max(total_words / 20, 1)
            return max(-1.0, min(1.0, sentiment))
            
        except Exception:
            return 0.0
    
    def _extract_symbols_advanced(self, text: str) -> List[str]:
        """Advanced symbol extraction with context awareness."""
        symbols = set()
        text_upper = text.upper()
        
        try:
            # Stock ticker patterns
            stock_patterns = [
                r'\b[A-Z]{1,5}\b',  # Basic ticker
                r'\$[A-Z]{1,5}\b',  # Twitter-style
                r'\b[A-Z]{1,5}:[A-Z]{2,3}\b'  # Exchange:Ticker
            ]
            
            for pattern in stock_patterns:
                matches = re.findall(pattern, text_upper)
                symbols.update(matches)
            
            # FX pairs
            fx_patterns = [
                r'([A-Z]{3})[/_-]([A-Z]{3})',
                r'(EUR|USD|GBP|JPY|CHF|CAD|AUD|NZD)[/_-]?(EUR|USD|GBP|JPY|CHF|CAD|AUD|NZD)'
            ]
            
            for pattern in fx_patterns:
                matches = re.findall(pattern, text_upper)
                for match in matches:
                    if isinstance(match, tuple) and len(match) == 2 and match[0] != match[1]:
                        symbols.add(f"{match[0]}/{match[1]}")
            
            # Crypto patterns
            crypto_patterns = [
                r'(BTC|BITCOIN)[-]?USD',
                r'(ETH|ETHEREUM)[-]?USD',
                r'(SOL|SOLANA)[-]?USD'
            ]
            
            for pattern in crypto_patterns:
                matches = re.findall(pattern, text_upper)
                for match in matches:
                    if isinstance(match, str):
                        symbols.add(f"{match}-USD")
                    elif isinstance(match, tuple):
                        symbols.add(f"{match[0]}-USD")
            
            # Filter out common words
            common_words = {
                'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN',
                'HAS', 'HAD', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET',
                'USE', 'MAN', 'NEW', 'NOW', 'WAY', 'MAY', 'SAY', 'EACH', 'WHICH',
                'FROM', 'THEY', 'KNOW', 'WANT', 'BEEN', 'GOOD', 'MUCH', 'SOME',
                'TIME', 'VERY', 'WHEN', 'COME', 'HERE', 'HOW', 'JUST', 'LIKE',
                'LONG', 'MAKE', 'MANY', 'OVER', 'SUCH', 'TAKE', 'THAN', 'THEM',
                'WELL', 'WERE', 'WILL', 'WITH', 'WOULD', 'THERE', 'COULD', 'OTHER'
            }
            
            filtered_symbols = []
            for symbol in symbols:
                clean_symbol = symbol.replace('$', '').replace(':', '')
                if clean_symbol not in common_words and len(clean_symbol) >= 2:
                    filtered_symbols.append(clean_symbol)
            
            return filtered_symbols[:10]  # Limit to 10 symbols
            
        except Exception as e:
            logger.error(f"Advanced symbol extraction error: {e}")
            return []
    
    def _extract_sectors(self, text: str, symbols: List[str]) -> List[str]:
        """Extract sectors mentioned in text."""
        try:
            text_lower = text.lower()
            sectors = []
            
            # Sector keywords
            sector_keywords = {
                'technology': ['tech', 'technology', 'software', 'ai', 'artificial intelligence', 'cloud', 'semiconductor'],
                'finance': ['bank', 'banking', 'financial', 'finance', 'insurance', 'credit'],
                'healthcare': ['health', 'healthcare', 'pharma', 'pharmaceutical', 'biotech', 'medical'],
                'energy': ['energy', 'oil', 'gas', 'renewable', 'solar', 'wind', 'petroleum'],
                'consumer': ['consumer', 'retail', 'shopping', 'e-commerce', 'discretionary', 'staples'],
                'industrials': ['industrial', 'manufacturing', 'aerospace', 'defense', 'construction'],
                'materials': ['materials', 'mining', 'metals', 'chemicals', 'steel', 'aluminum'],
                'utilities': ['utility', 'utilities', 'electric', 'power', 'water', 'gas utility'],
                'real_estate': ['real estate', 'reit', 'property', 'housing', 'commercial real estate']
            }
            
            for sector, keywords in sector_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    sectors.append(sector)
            
            # Also map symbols to sectors
            symbol_sector_map = {
                'AAPL': 'technology', 'MSFT': 'technology', 'GOOGL': 'technology',
                'JPM': 'finance', 'BAC': 'finance', 'WFC': 'finance',
                'JNJ': 'healthcare', 'PFE': 'healthcare', 'UNH': 'healthcare',
                'XOM': 'energy', 'CVX': 'energy', 'COP': 'energy'
            }
            
            for symbol in symbols:
                if symbol in symbol_sector_map:
                    sector = symbol_sector_map[symbol]
                    if sector not in sectors:
                        sectors.append(sector)
            
            return sectors
            
        except Exception as e:
            logger.error(f"Sector extraction error: {e}")
            return []
    
    def _extract_countries(self, text: str) -> List[str]:
        """Extract countries/regions mentioned in text."""
        try:
            text_lower = text.lower()
            countries = []
            
            country_keywords = {
                'united_states': ['usa', 'us', 'america', 'american', 'united states'],
                'china': ['china', 'chinese', 'beijing', 'shanghai'],
                'europe': ['europe', 'european', 'eu', 'eurozone'],
                'japan': ['japan', 'japanese', 'tokyo', 'yen'],
                'united_kingdom': ['uk', 'britain', 'british', 'london', 'england'],
                'germany': ['germany', 'german', 'berlin', 'deutschland'],
                'france': ['france', 'french', 'paris'],
                'canada': ['canada', 'canadian', 'toronto'],
                'australia': ['australia', 'australian', 'sydney']
            }
            
            for country, keywords in country_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    countries.append(country)
            
            return countries
            
        except Exception as e:
            logger.error(f"Country extraction error: {e}")
            return []
    
    def _calculate_market_impact_advanced(self, text: str, symbols: List[str], sectors: List[str]) -> float:
        """Calculate advanced market impact score."""
        try:
            impact_score = 0.0
            text_lower = text.lower()
            
            # High impact keywords
            high_impact_keywords = [
                'earnings', 'revenue', 'profit', 'merger', 'acquisition', 'ipo',
                'fed', 'interest rate', 'inflation', 'gdp', 'unemployment',
                'bankruptcy', 'lawsuit', 'investigation', 'regulation'
            ]
            
            for keyword in high_impact_keywords:
                if keyword in text_lower:
                    impact_score += 0.3
            
            # Medium impact keywords
            medium_impact_keywords = [
                'guidance', 'outlook', 'forecast', 'analyst', 'upgrade', 'downgrade',
                'partnership', 'contract', 'expansion', 'product launch'
            ]
            
            for keyword in medium_impact_keywords:
                if keyword in text_lower:
                    impact_score += 0.2
            
            # Symbol and sector multipliers
            if symbols:
                impact_score += len(symbols) * 0.1
            
            if sectors:
                impact_score += len(sectors) * 0.05
            
            # Normalize to [0, 1]
            return min(1.0, impact_score)
            
        except Exception as e:
            logger.error(f"Market impact calculation error: {e}")
            return 0.0
    
    def _calculate_relevance_advanced(self, text: str, symbols: List[str], sectors: List[str]) -> float:
        """Calculate advanced relevance score."""
        try:
            relevance_score = 0.0
            text_lower = text.lower()
            
            # Financial terms
            financial_terms = [
                'stock', 'market', 'trading', 'investment', 'portfolio',
                'price', 'valuation', 'financial', 'economic', 'business'
            ]
            
            for term in financial_terms:
                if term in text_lower:
                    relevance_score += 0.1
            
            # Symbol and sector relevance
            relevance_score += len(symbols) * 0.15
            relevance_score += len(sectors) * 0.1
            
            # Length penalty for very short texts
            word_count = len(text.split())
            if word_count < 10:
                relevance_score *= 0.5
            
            return min(1.0, relevance_score)
            
        except Exception as e:
            logger.error(f"Relevance calculation error: {e}")
            return 0.0
    
    def _analyze_cross_sector_implications(self, text: str, symbols: List[str], 
                                         sectors: List[str], sentiment_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Analyze cross-sector implications."""
        try:
            implications = {}
            text_lower = text.lower()
            sentiment_score = sentiment_analysis.get('overall_sentiment', 0)
            
            # Cross-sector impact patterns
            cross_sector_patterns = {
                'technology': {
                    'finance': 0.3,  # Fintech impact
                    'healthcare': 0.2,  # Health tech
                    'consumer': 0.4   # Consumer tech
                },
                'energy': {
                    'materials': 0.5,  # Commodity correlation
                    'industrials': 0.4,  # Industrial demand
                    'utilities': 0.3   # Energy utilities
                },
                'finance': {
                    'real_estate': 0.6,  # Mortgage/lending
                    'consumer': 0.3,     # Consumer lending
                    'industrials': 0.2   # Commercial lending
                }
            }
            
            # Calculate implications based on mentioned sectors
            for sector in sectors:
                if sector in cross_sector_patterns:
                    for related_sector, correlation in cross_sector_patterns[sector].items():
                        implications[related_sector] = sentiment_score * correlation
            
            # Add geopolitical implications
            if any(country in text_lower for country in ['china', 'trade war', 'tariff']):
                implications['technology'] = implications.get('technology', 0) + sentiment_score * 0.4
                implications['industrials'] = implications.get('industrials', 0) + sentiment_score * 0.3
            
            return implications
            
        except Exception as e:
            logger.error(f"Cross-sector implications error: {e}")
            return {}
    
    def _extract_anticipation_factors(self, text: str, symbols: List[str]) -> Dict[str, float]:
        """Extract anticipation factors from text."""
        try:
            factors = {}
            text_lower = text.lower()
            
            # Anticipation keywords and their weights
            anticipation_patterns = {
                'earnings_anticipation': ['earnings', 'quarterly', 'results', 'report'],
                'guidance_anticipation': ['guidance', 'outlook', 'forecast', 'expects'],
                'event_anticipation': ['meeting', 'announcement', 'decision', 'release'],
                'regulatory_anticipation': ['approval', 'regulation', 'policy', 'ruling'],
                'market_anticipation': ['volatility', 'uncertainty', 'risk', 'opportunity']
            }
            
            for factor, keywords in anticipation_patterns.items():
                keyword_count = sum(1 for keyword in keywords if keyword in text_lower)
                if keyword_count > 0:
                    factors[factor] = min(keyword_count * 0.2, 1.0)
            
            # Time-based anticipation
            time_indicators = ['next week', 'next month', 'upcoming', 'soon', 'expected']
            if any(indicator in text_lower for indicator in time_indicators):
                factors['time_sensitive'] = 0.7
            
            return factors
            
        except Exception as e:
            logger.error(f"Anticipation factors extraction error: {e}")
            return {}
    
    def _extract_narrative_elements(self, text: str) -> List[str]:
        """Extract narrative elements from text."""
        try:
            elements = []
            text_lower = text.lower()
            
            # Narrative patterns
            narrative_patterns = {
                'growth_story': ['growth', 'expansion', 'scaling', 'increasing'],
                'turnaround_story': ['turnaround', 'recovery', 'improvement', 'restructuring'],
                'disruption_story': ['disruption', 'innovation', 'transformation', 'digital'],
                'value_story': ['undervalued', 'cheap', 'discount', 'bargain'],
                'momentum_story': ['momentum', 'trend', 'acceleration', 'surge'],
                'risk_story': ['risk', 'concern', 'challenge', 'headwind']
            }
            
            for narrative, keywords in narrative_patterns.items():
                if any(keyword in text_lower for keyword in keywords):
                    elements.append(narrative)
            
            return elements
            
        except Exception as e:
            logger.error(f"Narrative elements extraction error: {e}")
            return []
    
    def _extract_decision_implications(self, text: str, symbols: List[str], 
                                     sentiment_analysis: Dict[str, Any], 
                                     anticipation_factors: Dict[str, float]) -> List[str]:
        """Extract decision-making implications."""
        try:
            implications = []
            text_lower = text.lower()
            sentiment_score = sentiment_analysis.get('overall_sentiment', 0)
            
            # Decision keywords
            if any(word in text_lower for word in ['buy', 'purchase', 'acquire']):
                if sentiment_score > 0.3:
                    implications.append('Consider long positions')
                else:
                    implications.append('Exercise caution on long positions')
            
            if any(word in text_lower for word in ['sell', 'exit', 'reduce']):
                if sentiment_score < -0.3:
                    implications.append('Consider reducing exposure')
                else:
                    implications.append('Monitor for exit opportunities')
            
            # Risk implications
            if any(word in text_lower for word in ['risk', 'volatile', 'uncertain']):
                implications.append('Increase risk management measures')
            
            # Opportunity implications
            if any(word in text_lower for word in ['opportunity', 'potential', 'upside']):
                implications.append('Monitor for entry opportunities')
            
            # Time-sensitive implications
            if anticipation_factors.get('time_sensitive', 0) > 0.5:
                implications.append('Time-sensitive decision required')
            
            return implications[:5]  # Limit to top 5
            
        except Exception as e:
            logger.error(f"Decision implications extraction error: {e}")
            return []
    
    async def get_symbol_news_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive news analysis for a specific symbol."""
        try:
            # Process recent news
            processed_articles = await self.process_news_comprehensive(max_articles=200)
            
            # Filter articles relevant to symbol
            relevant_articles = [
                article for article in processed_articles 
                if symbol in article.symbols_mentioned
            ]
            
            if not relevant_articles:
                return {'error': f'No recent news found for {symbol}'}
            
            # Aggregate analysis
            sentiment_scores = [article.sentiment_score for article in relevant_articles]
            overall_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
            
            # Extract key themes
            all_themes = []
            for article in relevant_articles:
                all_themes.extend(article.narrative_elements)
            
            theme_counts = {}
            for theme in all_themes:
                theme_counts[theme] = theme_counts.get(theme, 0) + 1
            
            top_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Recent headlines
            recent_headlines = [
                {
                    'title': article.title,
                    'source': article.source,
                    'sentiment_score': article.sentiment_score,
                    'published_at': article.published_at.isoformat()
                }
                for article in sorted(relevant_articles, key=lambda x: x.published_at, reverse=True)[:10]
            ]
            
            # Market moving news (high impact)
            market_moving = [
                article for article in relevant_articles 
                if article.market_impact_score > 0.7
            ]
            
            # Anticipation signals
            anticipation_signals = []
            for article in relevant_articles:
                for factor, value in article.anticipation_factors.items():
                    if value > 0.6:
                        anticipation_signals.append({
                            'factor': factor,
                            'strength': value,
                            'source': article.source
                        })
            
            return {
                'symbol': symbol,
                'article_count': len(relevant_articles),
                'overall_sentiment': overall_sentiment,
                'sentiment_category': self._score_to_sentiment_category(overall_sentiment),
                'key_themes': [theme[0] for theme in top_themes],
                'recent_headlines': recent_headlines,
                'market_moving_news': [
                    {
                        'title': article.title,
                        'impact_score': article.market_impact_score,
                        'sentiment': article.sentiment_score
                    }
                    for article in market_moving[:5]
                ],
                'anticipation_signals': anticipation_signals[:10],
                'cross_sector_mentions': self._aggregate_cross_sector_mentions(relevant_articles),
                'geopolitical_factors': self._extract_geopolitical_factors(relevant_articles),
                'processing_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Symbol news analysis error: {e}")
            return {'error': f'News analysis failed for {symbol}: {str(e)}'}
    
    def _aggregate_cross_sector_mentions(self, articles: List) -> Dict[str, int]:
        """Aggregate cross-sector mentions from articles."""
        try:
            sector_mentions = {}
            
            for article in articles:
                for sector in article.sectors_mentioned:
                    sector_mentions[sector] = sector_mentions.get(sector, 0) + 1
            
            return sector_mentions
            
        except Exception as e:
            logger.error(f"Cross-sector aggregation error: {e}")
            return {}
    
    def _extract_geopolitical_factors(self, articles: List) -> List[str]:
        """Extract geopolitical factors from articles."""
        try:
            geopolitical_factors = []
            
            geopolitical_keywords = [
                'trade war', 'tariff', 'sanctions', 'brexit', 'election',
                'policy', 'regulation', 'treaty', 'agreement', 'conflict'
            ]
            
            for article in articles:
                text_lower = (article.title + ' ' + article.content).lower()
                for keyword in geopolitical_keywords:
                    if keyword in text_lower and keyword not in geopolitical_factors:
                        geopolitical_factors.append(keyword)
            
            return geopolitical_factors[:10]
            
        except Exception as e:
            logger.error(f"Geopolitical factors extraction error: {e}")
            return []
    
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

    def analyze_symbol_sentiment(self, symbol: str, timeout: int = 10) -> Optional[Dict]:
        """
        Perform per-asset sentiment analysis for a specific symbol.

        Fetches news articles, filters to those explicitly mentioning the symbol,
        and computes a weighted aggregate sentiment score using SourceCredibilityEngine
        credibility scores.

        Args:
            symbol: The ticker symbol to analyze (e.g. "AAPL", "BTC-USD").
            timeout: Maximum seconds to wait before returning None (default 10).

        Returns:
            Dict with keys:
                score (float): Weighted aggregate sentiment in [-1.0, 1.0]
                label (str): VERY_BEARISH / BEARISH / NEUTRAL / BULLISH / VERY_BULLISH
                article_count (int): Number of articles that mentioned the symbol
                top_headlines (List[str]): Up to 3 most impactful headlines
            Returns None if the timeout is exceeded.

        Requirements: 2.1, 2.2, 2.5, 2.6
        """
        from source_credibility_engine import SourceCredibilityEngine

        result_container: Dict = {}
        exception_container: Dict = {}

        def _run():
            try:
                # Run the async fetch in a fresh event loop inside this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    articles = loop.run_until_complete(
                        self.process_news_comprehensive(max_articles=300)
                    )
                finally:
                    loop.close()

                # Filter to articles that explicitly mention the symbol (Requirement 2.1)
                symbol_upper = symbol.upper()
                # Also build a clean base ticker for matching (strip suffixes like =F, =X, -USD)
                base_ticker = re.sub(r'[=\-][A-Z0-9]+$', '', symbol_upper)

                relevant: List[ProcessedArticle] = []
                for article in articles:
                    text_upper = (article.title + ' ' + article.content).upper()
                    # Match exact ticker or base ticker as a whole word
                    if (re.search(r'\b' + re.escape(symbol_upper) + r'\b', text_upper) or
                            (base_ticker and re.search(r'\b' + re.escape(base_ticker) + r'\b', text_upper)) or
                            symbol_upper in article.symbols_mentioned):
                        relevant.append(article)

                if not relevant:
                    result_container['result'] = {
                        'score': 0.0,
                        'label': SentimentScore.NEUTRAL,
                        'article_count': 0,
                        'top_headlines': []
                    }
                    return

                # Compute weighted aggregate sentiment (Requirement 2.6)
                credibility_engine = SourceCredibilityEngine()
                weighted_sum = 0.0
                total_weight = 0.0

                for article in relevant:
                    source_cred = credibility_engine.get_source_credibility(article.source)
                    weight = source_cred.final_weight
                    weighted_sum += article.sentiment_score * weight
                    total_weight += weight

                if total_weight > 0:
                    raw_score = weighted_sum / total_weight
                else:
                    raw_score = float(np.mean([a.sentiment_score for a in relevant]))

                # Clamp to [-1.0, 1.0] (Requirement 2.2)
                score = float(max(-1.0, min(1.0, raw_score)))

                # Determine label (Requirement 2.2)
                label = self._score_to_sentiment_category(score)

                # Top 3 headlines by market impact score (Requirement 2.3)
                sorted_articles = sorted(relevant, key=lambda a: a.market_impact_score, reverse=True)
                top_headlines = [a.title for a in sorted_articles[:3] if a.title]

                result_container['result'] = {
                    'score': score,
                    'label': label,
                    'article_count': len(relevant),
                    'top_headlines': top_headlines
                }

            except Exception as exc:
                exception_container['error'] = exc
                logger.error(f"Error in analyze_symbol_sentiment thread: {exc}")

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            # Timeout exceeded (Requirement 2.5)
            logger.warning(f"analyze_symbol_sentiment timed out after {timeout}s for {symbol}")
            return None

        if 'error' in exception_container:
            logger.error(f"analyze_symbol_sentiment error for {symbol}: {exception_container['error']}")
            return None

        return result_container.get('result')

# Global instance
_advanced_news_processor = None

def get_advanced_news_processor() -> AdvancedNewsProcessor:
    """Get or create advanced news processor instance."""
    global _advanced_news_processor
    if _advanced_news_processor is None:
        _advanced_news_processor = AdvancedNewsProcessor()
    return _advanced_news_processor