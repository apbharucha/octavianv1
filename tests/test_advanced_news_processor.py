import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, create_autospec
import asyncio
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st
import contextlib
import logging
import threading
from dataclasses import dataclass
import re

import advanced_news_processor
import uuid

from advanced_news_processor import AdvancedNewsProcessor, ProcessedArticle
from news_analysis_engine import SentimentScore

@dataclass
class MockSourceCredibility:
    final_weight: float = 0.5

def article_strategy():
    """Hypothesis strategy to generate ProcessedArticle instances."""
    return st.builds(
        ProcessedArticle,
        article_id=st.uuids().map(str),
        source=st.text(min_size=1, max_size=20),
        title=st.text(min_size=1, max_size=50),
        content=st.text(min_size=10, max_size=200),
        url=st.just("http://example.com/article"),
        published_at=st.datetimes(min_value=datetime(2023, 1, 1), max_value=datetime(2023, 12, 31)),
        symbols_mentioned=st.lists(st.text(min_size=1, max_size=10), max_size=3),
        sectors_mentioned=st.just([]),
        countries_mentioned=st.just([]),
        sentiment_score=st.floats(min_value=-1.0, max_value=1.0),
    sentiment_category=st.just(SentimentScore.NEUTRAL), # Placeholder, will be overwritten if needed
        market_impact_score=st.floats(min_value=0.0, max_value=1.0),
        relevance_score=st.floats(min_value=0.0, max_value=1.0),
        cross_sector_implications=st.just({}),
        anticipation_factors=st.just({}),
        narrative_elements=st.just([]),
        decision_implications=st.just([]),
        processed_at=st.just(datetime.now())
    )

@pytest.fixture(scope="function")
def advanced_news_processor_factory():
    """
    Fixture to create AdvancedNewsProcessor instances with mocked dependencies for clamping tests.
    Returns a context manager that yields the processor and the mock_cred_instance.
    """
    @contextlib.contextmanager
    def _factory():
        mock_cred_instance = MagicMock()
        mock_cred_instance.get_source_credibility.return_value = MockSourceCredibility(final_weight=0.5)

        with patch('advanced_news_processor.get_database_manager'), \
             patch('advanced_news_processor.get_api_handler'), \
             patch('source_credibility_engine.SourceCredibilityEngine', return_value=mock_cred_instance), \
             patch('advanced_news_processor.logger'):
            processor = AdvancedNewsProcessor()
            yield processor, mock_cred_instance
    return _factory

class TestClamping:
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    @given(articles=st.lists(article_strategy(), min_size=1, max_size=10))
    @pytest.mark.asyncio
    async def test_sentiment_score_clamped_to_range(self, advanced_news_processor_factory, articles):
        """
        Tests that the sentiment score is correctly clamped to [-1.0, 1.0]
        within the article processing flow.
        """
        with advanced_news_processor_factory() as (processor, mock_cred_instance):
            # Mock _analyze_sentiment_advanced to return the raw sentiment score from hypothesis-generated articles
            # and ensure the category is valid for ProcessedArticle construction.
            original_analyze_sentiment_advanced = processor._analyze_sentiment_advanced
            
            def mock_analyze_sentiment_advanced(text_content):
                # Find the corresponding article based on title/content if possible, or just use a generic one
                # For this test, we care about the overall_sentiment value being passed through.
                # The actual implementation of _process_single_article will call this mock,
                # then construct ProcessedArticle, where clamping happens.
                # We need to ensure that the 'overall_sentiment' we return here is the 'raw_score'
                # that _process_single_article then clamps when creating ProcessedArticle.

                # This is a bit of a hack: we'll use the first article's sentiment_score as the raw score
                # This works because hypothesis generates unique articles, but _process_single_article
                # will be called for each one sequentially.
                # A more robust solution would be to pass the 'raw_score' via a different mock.
                # For now, let's assume the sentiment_score of the hypothesis-generated article is the 'raw' score.
                # The _process_single_article will call _analyze_sentiment_advanced, get this raw score,
                # then clamp it during ProcessedArticle construction.
                # So we just need to provide a valid sentiment_category for the ProcessedArticle constructor.
                return {
                    'overall_sentiment': articles[0].sentiment_score, # This will be the raw score
                    'sentiment_category': SentimentScore.NEUTRAL # Placeholder
                }
            
            processor._analyze_sentiment_advanced = MagicMock(side_effect=mock_analyze_sentiment_advanced)

            # Mock the internal fetch methods of process_news_comprehensive to return our raw article data.
            # process_news_comprehensive expects raw dicts, not ProcessedArticle objects directly from fetchers.
            raw_article_dicts = []
            for article in articles:
                raw_article_dicts.append({
                    "title": article.title,
                    "content": article.content,
                    "url": article.url,
                    "source": article.source,
                    "published_at": article.published_at,
                    # Add raw sentiment score to mimic original data that _process_single_article would get
                    "sentiment_score_raw": article.sentiment_score 
                })

            processor._fetch_rss_articles = AsyncMock(return_value=raw_article_dicts)
            processor._fetch_api_articles = AsyncMock(return_value=[])
            processor._fetch_social_articles = AsyncMock(return_value=[])

            # Call the main processing method
            processed_results = await processor.process_news_comprehensive(max_articles=len(articles))
            
            # Assert that the sentiment scores in the processed_results are clamped
            for i, result_article in enumerate(processed_results):
                raw_score_from_hypothesis = articles[i].sentiment_score
                expected_clamped_score = float(max(-1.0, min(1.0, raw_score_from_hypothesis)))
                
                assert pytest.approx(result_article.sentiment_score) == expected_clamped_score
                assert -1.0 <= result_article.sentiment_score <= 1.0

# New fixture for TestAnalyzeSymbolSentiment
@pytest.fixture
def sentiment_processor_factory():
    """Fixture for AdvancedNewsProcessor with mocked dependencies for sentiment analysis tests."""
    @contextlib.contextmanager
    def _factory(
        mock_process_news_comprehensive_return_value=None,
        mock_credibility_weights=None,
        mock_score_to_sentiment_category_return_value=SentimentScore.NEUTRAL
    ):
        mock_cred_instance = MagicMock()
        if mock_credibility_weights:
            mock_cred_instance.get_source_credibility.side_effect = \
                lambda source: MockSourceCredibility(final_weight=mock_credibility_weights.get(source, 0.5))
        else:
            mock_cred_instance.get_source_credibility.return_value = MockSourceCredibility(final_weight=0.5)

        with patch('advanced_news_processor.get_database_manager'), \
             patch('advanced_news_processor.get_api_handler'), \
             patch('source_credibility_engine.SourceCredibilityEngine', return_value=mock_cred_instance), \
             patch('advanced_news_processor.logger') as mock_logger, \
             patch('advanced_news_processor.threading.Thread') as mock_thread_class, \
             patch('asyncio.new_event_loop') as mock_new_event_loop, \
             patch('asyncio.set_event_loop') as mock_set_event_loop:

            mock_loop = create_autospec(asyncio.AbstractEventLoop) # Use create_autospec
            mock_loop.is_running.return_value = False # Add this, might be checked
            mock_new_event_loop.return_value = mock_loop
            mock_loop.run_until_complete.return_value = mock_process_news_comprehensive_return_value # Set return value here
            mock_loop.close.return_value = None # Ensure close is callable

            mock_logger.error.side_effect = lambda *args, **kwargs: print(f"MOCK_LOGGER_ERROR: {args}, {kwargs}")
            mock_logger.exception.side_effect = lambda *args, **kwargs: print(f"MOCK_LOGGER_EXCEPTION: {args}, {kwargs}")

            def thread_mock_constructor(target, args=(), kwargs={}, daemon=None):
                print(f"Thread mock constructor called for target: {target.__name__}")
                mock_thread_instance = MagicMock()
                mock_thread_instance._actual_target = target
                mock_thread_instance._actual_args = args
                mock_thread_instance._actual_kwargs = kwargs
                mock_thread_instance.daemon = daemon
                mock_thread_instance.is_alive.return_value = False # Assume thread completes

                def start_mock():
                    print(f"Executing _run directly for test: {mock_thread_instance._actual_target.__name__}")
                    mock_thread_instance._actual_target(*mock_thread_instance._actual_args, **mock_thread_instance._actual_kwargs)
                
                mock_thread_instance.start = MagicMock(side_effect=start_mock)
                mock_thread_instance.join = MagicMock()
                return mock_thread_instance

            mock_thread_class.side_effect = thread_mock_constructor

            processor = AdvancedNewsProcessor()
            processor.processing_metrics = MagicMock()
            processor.logger = mock_logger # Assign the mocked logger

            # NEW: Mock process_news_comprehensive
            processor.process_news_comprehensive = AsyncMock(return_value=mock_process_news_comprehensive_return_value)

            with patch.object(processor, '_score_to_sentiment_category', return_value=mock_score_to_sentiment_category_return_value):
                yield processor

    return _factory


class TestAnalyzeSymbolSentiment:

    def test_articles_mentioning_symbol(self, sentiment_processor_factory):
        symbol = "AAPL"
        
        # Generate some articles, some mentioning AAPL, some not
        # Using article_strategy to create base articles then modifying
        # This approach ensures articles have all required fields.
        base_articles = [article_strategy().example() for _ in range(4)]
        
        # Manually adjust content for better control
        base_articles[0].title = "AAPL rises on strong earnings"
        base_articles[0].content = "Apple Inc. (AAPL) stock performed well."
        base_articles[0].sentiment_score = 0.8
        base_articles[0].source = "Reliable_News"
        base_articles[0].symbols_mentioned = ["AAPL"]
        base_articles[0].market_impact_score = 0.9

        base_articles[1].title = "Another AAPL surge expected"
        base_articles[1].content = "Analysts predict a significant increase for AAPL."
        base_articles[1].sentiment_score = 0.7
        base_articles[1].source = "Semi_Reliable_Blog"
        base_articles[1].symbols_mentioned = ["AAPL", "NASDAQ"]
        base_articles[1].market_impact_score = 0.8

        base_articles[2].title = "Google (GOOGL) announces new AI"
        base_articles[2].content = "Alphabet's GOOGL stock gained."
        base_articles[2].sentiment_score = 0.6
        base_articles[2].source = "Reliable_News"
        base_articles[2].symbols_mentioned = ["GOOGL"]
        base_articles[2].market_impact_score = 0.7

        base_articles[3].title = "Tesla (TSLA) stock drops"
        base_articles[3].content = "Elon Musk's TSLA had a day."
        base_articles[3].sentiment_score = -0.5
        base_articles[3].source = "Unreliable_Scoop"
        base_articles[3].symbols_mentioned = ["TSLA"]
        base_articles[3].market_impact_score = 0.6

        all_articles = [base_articles[0], base_articles[2], base_articles[1], base_articles[3]]

        credibility_weights = {
            "Reliable_News": 0.9,
            "Semi_Reliable_Blog": 0.7,
            "Unreliable_Scoop": 0.3
        }

        # Expected score calculation for AAPL:
        # Article 0: 0.8 * 0.9 = 0.72
        # Article 1: 0.7 * 0.7 = 0.49
        # Weighted sum = 0.72 + 0.49 = 1.21
        # Total weight = 0.9 + 0.7 = 1.6
        # Expected score = 1.21 / 1.6 = 0.75625

        expected_score = 0.75625

        with sentiment_processor_factory(
            mock_process_news_comprehensive_return_value=all_articles,
            mock_credibility_weights=credibility_weights,
            mock_score_to_sentiment_category_return_value=SentimentScore.BULLISH
        ) as processor:
            # No await here because analyze_symbol_sentiment is not async itself
            result = processor.analyze_symbol_sentiment(symbol)
            
            assert result is not None
            assert result['article_count'] == 2
            assert pytest.approx(result['score'], abs=1e-6) == expected_score
            assert result['label'] == SentimentScore.BULLISH
            assert "AAPL rises on strong earnings" in result['top_headlines']
            assert "Another AAPL surge expected" in result['top_headlines']
            assert len(result['top_headlines']) == 2 # Only 2 AAPL articles, so only 2 headlines

    def test_no_matching_articles(self, sentiment_processor_factory):
        symbol = "XYZ"
        
        # Manually create ProcessedArticle instances that are guaranteed
        # not to mention the symbol "XYZ" in title, content, or symbols_mentioned.
        non_matching_articles = [
            ProcessedArticle(
                article_id=str(uuid.uuid4()),
                title=f"Generic News Title {i}",
                content=f"This is some generic news content for test article {i}. It mentions other companies like Alpha and Beta.",
                url=f"http://example.com/article_{i}",
                source="TestSource",
                published_at=datetime.now() - timedelta(days=i),
                symbols_mentioned=["ALPHA", "BETA"], # Ensure no "XYZ" here
                sectors_mentioned=[],
                countries_mentioned=[],
                sentiment_score=0.1,
                sentiment_category=SentimentScore.NEUTRAL,
                market_impact_score=0.1,
                relevance_score=0.1,
                cross_sector_implications={},
                anticipation_factors={},
                narrative_elements=[],
                decision_implications=[],
                processed_at=datetime.now()
            ) for i in range(3)
        ]

        with sentiment_processor_factory(
            mock_process_news_comprehensive_return_value=non_matching_articles,
            mock_credibility_weights={},
            mock_score_to_sentiment_category_return_value=SentimentScore.NEUTRAL
        ) as processor:
            result = processor.analyze_symbol_sentiment(symbol)

            assert result is not None
            assert result['article_count'] == 0
            assert result['score'] == 0.0
            assert result['label'] == SentimentScore.NEUTRAL
            assert result['top_headlines'] == []

    def test_timeout_exceeded(self): # Do not use sentiment_processor_factory here
        symbol = "MSFT"
        # Create a dummy processor instance
        processor = AdvancedNewsProcessor()
        processor.processing_metrics = MagicMock()
        processor.process_news_comprehensive = AsyncMock(return_value=[article_strategy().example()])

        with patch('threading.Thread') as mock_thread_class, \
             patch.object(processor, '_score_to_sentiment_category', return_value=SentimentScore.NEUTRAL), \
             patch('advanced_news_processor.logger') as mock_logger: # Patch logger locally

            processor.logger = mock_logger # Assign the mocked logger
            
            mock_thread_instance = MagicMock()
            mock_thread_class.return_value = mock_thread_instance
            mock_thread_instance.start.return_value = None
            mock_thread_instance.join.return_value = None 
            mock_thread_instance.is_alive.return_value = True # Simulate thread still alive after join(timeout)

            result = processor.analyze_symbol_sentiment(symbol, timeout=0.1) # Set a very short timeout

            assert result is None
            mock_thread_instance.join.assert_called_once_with(timeout=0.1)
            assert processor.logger.warning.called
            processor.logger.warning.assert_called_with(f"analyze_symbol_sentiment timed out after {0.1}s for {symbol}")


    def test_process_news_comprehensive_raises_exception(self, monkeypatch):
        symbol = "IBM"
        processor = AdvancedNewsProcessor()
        processor.processing_metrics = MagicMock()

        mock_logger = MagicMock()
        monkeypatch.setattr(advanced_news_processor, 'logger', mock_logger)
        
        # We need to mock the asyncio loop and thread creation within the analyze_symbol_sentiment method
        # directly in this test, without relying on the factory for this specific test case.

        mock_loop = MagicMock()
        mock_loop.run_until_complete.side_effect = Exception("API Error")
        mock_loop.close.return_value = None

        mock_new_event_loop = MagicMock(return_value=mock_loop)
        monkeypatch.setattr(asyncio, 'new_event_loop', mock_new_event_loop)
        monkeypatch.setattr(asyncio, 'set_event_loop', MagicMock()) # Mock set_event_loop as well

        # Mock threading.Thread
        def mock_thread_constructor(target, args=(), kwargs={}, daemon=None):
            mock_thread_instance = MagicMock()
            mock_thread_instance.start.side_effect = lambda: target(*args, **kwargs)
            mock_thread_instance.join.return_value = None
            mock_thread_instance.is_alive.return_value = False
            return mock_thread_instance
        monkeypatch.setattr(threading, 'Thread', mock_thread_constructor)


        result = processor.analyze_symbol_sentiment(symbol)

        assert result is None
        assert mock_logger.error.call_count == 2
        mock_logger.error.assert_any_call(f"Error in analyze_symbol_sentiment thread: API Error")
        mock_logger.error.assert_any_call(f"analyze_symbol_sentiment error for {symbol}: API Error")


    def test_all_articles_filtered_out(self, sentiment_processor_factory):
        symbol = "FOO"
        # Articles that explicitly DO NOT mention FOO
        articles = [article_strategy().example() for _ in range(3)]
        for article in articles:
            article.symbols_mentioned = ["BAR", "BAZ"]
            article.title = "Irrelevant news"
            article.content = "Some content about BAR and BAZ"

        with sentiment_processor_factory(
            mock_process_news_comprehensive_return_value=articles,
            mock_credibility_weights={},
            mock_score_to_sentiment_category_return_value=SentimentScore.NEUTRAL
        ) as processor:
            result = processor.analyze_symbol_sentiment(symbol)

            assert result is not None
            assert result['article_count'] == 0
            assert result['score'] == 0.0
            assert result['label'] == SentimentScore.NEUTRAL
            assert result['top_headlines'] == []

    def test_empty_process_news_comprehensive_return(self, sentiment_processor_factory):
        symbol = "EMPTY"

        with sentiment_processor_factory(
            mock_process_news_comprehensive_return_value=[], # Empty list of articles
            mock_credibility_weights={},
            mock_score_to_sentiment_category_return_value=SentimentScore.NEUTRAL
        ) as processor:
            result = processor.analyze_symbol_sentiment(symbol)

            assert result is not None
            assert result['article_count'] == 0
            assert result['score'] == 0.0
            assert result['label'] == SentimentScore.NEUTRAL
            assert result['top_headlines'] == []

    def test_mixed_case_symbol_matching(self, sentiment_processor_factory):
        symbol = "aapl" # Lowercase symbol
        articles = [article_strategy().example()]
        articles[0].title = "Apple Inc. (AAPL) results"
        articles[0].content = "Strong performance for aapl."
        articles[0].sentiment_score = 0.8
        articles[0].source = "CredibleSource"
        articles[0].symbols_mentioned = ["AAPL"]
        articles[0].market_impact_score = 0.9

        credibility_weights = {"CredibleSource": 1.0}
        expected_score = 0.8

        with sentiment_processor_factory(
            mock_process_news_comprehensive_return_value=articles,
            mock_credibility_weights=credibility_weights,
            mock_score_to_sentiment_category_return_value=SentimentScore.BULLISH
        ) as processor:
            result = processor.analyze_symbol_sentiment(symbol)

            assert result is not None
            assert result['article_count'] == 1
            assert pytest.approx(result['score'], abs=1e-6) == expected_score
            assert result['label'] == SentimentScore.BULLISH
            assert "Apple Inc. (AAPL) results" in result['top_headlines']
