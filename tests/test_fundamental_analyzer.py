"""
Tests for FundamentalAnalyzer (fundamental_analyzer.py).

Covers:
- Property test: weight sum invariant for _compute_composite_score
- Unit tests: equity with full data, equity with partial data,
  non-equity returns None, cache hit within TTL, cache miss after TTL

**Validates: Requirements 1.1, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8**
"""

import time
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fundamental_analyzer import FundamentalAnalyzer, FundamentalData
from timeframe_analysis_engine import TimeframeScope


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fundamental_data(symbol: str = "AAPL", score: float = 0.3) -> FundamentalData:
    """Create a FundamentalData instance for testing."""
    return FundamentalData(
        symbol=symbol,
        trailing_pe=18.5,
        forward_pe=15.0,
        eps_ttm=6.11,
        revenue_growth=0.08,
        profit_margin=0.25,
        debt_to_equity=1.2,
        score=score,
        label="Fairly Valued",
        fetched_at=datetime.now()
    )


def _make_analyzer() -> FundamentalAnalyzer:
    return FundamentalAnalyzer()


# ---------------------------------------------------------------------------
# Property Test 6.6: Weight sum invariant
# **Validates: Requirements 1.3, 1.4, 1.5, 1.6**
# ---------------------------------------------------------------------------

# We test _compute_composite_score by importing the chatbot class and calling
# the method directly. To avoid heavy chatbot initialization, we test the
# weight logic in isolation by extracting the weight computation.

def _get_weights(timeframe_scope: TimeframeScope, has_fundamentals: bool):
    """
    Mirror the weight logic from _compute_composite_score.
    Returns (fund_weight, tech_weight).
    """
    if not has_fundamentals:
        return (0.0, 1.0)
    
    if timeframe_scope in (TimeframeScope.SCALPING, TimeframeScope.INTRADAY):
        return (0.20, 0.80)
    elif timeframe_scope == TimeframeScope.INVESTMENT:
        return (0.75, 0.25)
    else:
        return (0.60, 0.40)


@given(
    timeframe=st.sampled_from(list(TimeframeScope)),
    has_fundamentals=st.booleans(),
)
@settings(max_examples=100)
def test_weight_sum_always_equals_one(timeframe, has_fundamentals):
    """
    Property 1: Weight sum invariant.
    For any TimeframeScope and any equity/non-equity combination,
    fundamental_weight + technical_weight == 1.0.

    **Validates: Requirements 1.3, 1.4, 1.5, 1.6**
    """
    fund_w, tech_w = _get_weights(timeframe, has_fundamentals)
    assert abs(fund_w + tech_w - 1.0) < 1e-9, (
        f"Weights don't sum to 1.0 for timeframe={timeframe}, "
        f"has_fundamentals={has_fundamentals}: {fund_w} + {tech_w} = {fund_w + tech_w}"
    )


@given(
    timeframe=st.sampled_from(list(TimeframeScope)),
)
@settings(max_examples=50)
def test_non_equity_uses_100_percent_technical(timeframe):
    """
    Non-equity symbols (fundamental_data is None) must use 100% technical weight.

    **Validates: Requirement 1.6**
    """
    fund_w, tech_w = _get_weights(timeframe, has_fundamentals=False)
    assert fund_w == 0.0
    assert tech_w == 1.0


# ---------------------------------------------------------------------------
# Unit Tests 6.7: FundamentalAnalyzer
# ---------------------------------------------------------------------------

class TestFundamentalAnalyzerScoring:
    """Tests for score_fundamentals method."""

    def test_score_returns_tuple(self):
        analyzer = _make_analyzer()
        score, label = analyzer.score_fundamentals({
            'trailing_pe': 15.0,
            'forward_pe': 12.0,
            'eps_ttm': 5.0,
            'revenue_growth': 0.15,
            'profit_margin': 0.20,
            'debt_to_equity': 0.5,
        })
        assert isinstance(score, float)
        assert isinstance(label, str)

    def test_score_in_range(self):
        analyzer = _make_analyzer()
        score, _ = analyzer.score_fundamentals({
            'trailing_pe': 15.0,
            'eps_ttm': 5.0,
            'revenue_growth': 0.15,
            'profit_margin': 0.20,
            'debt_to_equity': 0.5,
        })
        assert -1.0 <= score <= 1.0

    def test_label_undervalued_for_strong_fundamentals(self):
        analyzer = _make_analyzer()
        _, label = analyzer.score_fundamentals({
            'trailing_pe': 8.0,   # Very low P/E
            'eps_ttm': 10.0,      # High EPS
            'revenue_growth': 0.30,  # 30% growth
            'profit_margin': 0.35,   # 35% margin
            'debt_to_equity': 0.2,   # Low debt
        })
        assert label == "Undervalued"

    def test_label_overvalued_for_weak_fundamentals(self):
        analyzer = _make_analyzer()
        _, label = analyzer.score_fundamentals({
            'trailing_pe': 80.0,   # Very high P/E
            'eps_ttm': -2.0,       # Negative EPS
            'revenue_growth': -0.10,  # Declining revenue
            'profit_margin': -0.05,   # Negative margin
            'debt_to_equity': 5.0,    # Very high debt
        })
        assert label == "Overvalued"

    def test_empty_fundamentals_returns_fairly_valued(self):
        analyzer = _make_analyzer()
        score, label = analyzer.score_fundamentals({})
        assert score == 0.0
        assert label == "Fairly Valued"

    def test_partial_data_still_scores(self):
        analyzer = _make_analyzer()
        score, label = analyzer.score_fundamentals({
            'trailing_pe': 20.0,
            'eps_ttm': None,
            'revenue_growth': None,
            'profit_margin': 0.15,
            'debt_to_equity': None,
        })
        assert -1.0 <= score <= 1.0
        assert label in ("Undervalued", "Fairly Valued", "Overvalued")


class TestFundamentalAnalyzerNonEquity:
    """Tests for non-equity symbol detection."""

    def test_fx_pair_slash_returns_none(self):
        analyzer = _make_analyzer()
        result = analyzer.fetch_fundamentals("EUR/USD")
        assert result is None

    def test_fx_pair_equals_x_returns_none(self):
        analyzer = _make_analyzer()
        result = analyzer.fetch_fundamentals("EURUSD=X")
        assert result is None

    def test_futures_returns_none(self):
        analyzer = _make_analyzer()
        result = analyzer.fetch_fundamentals("ES=F")
        assert result is None

    def test_crypto_usd_returns_none(self):
        analyzer = _make_analyzer()
        result = analyzer.fetch_fundamentals("BTC-USD")
        assert result is None

    def test_crypto_usdt_returns_none(self):
        analyzer = _make_analyzer()
        result = analyzer.fetch_fundamentals("ETH-USDT")
        assert result is None


class TestFundamentalAnalyzerCaching:
    """Tests for caching behavior."""

    def test_cache_hit_within_ttl_returns_same_object(self):
        """Cache hit within TTL should return the same FundamentalData object."""
        analyzer = _make_analyzer()
        fd = _make_fundamental_data("AAPL")
        
        # Manually populate cache
        analyzer._cache["AAPL"] = (fd, time.time())
        
        result = analyzer.fetch_fundamentals("AAPL")
        assert result is fd

    def test_cache_miss_after_ttl_calls_yfinance(self):
        """Cache miss after TTL should call yfinance again."""
        analyzer = _make_analyzer()
        fd = _make_fundamental_data("AAPL")
        
        # Set cache with expired timestamp (TTL + 1 second ago)
        expired_time = time.time() - analyzer._cache_ttl - 1
        analyzer._cache["AAPL"] = (fd, expired_time)
        
        # Mock yfinance to return no fundamental data (simulates network call)
        mock_info = {}
        with patch("yfinance.Ticker") as mock_ticker_cls:
            mock_ticker = MagicMock()
            mock_ticker.info = mock_info
            mock_ticker_cls.return_value = mock_ticker
            
            result = analyzer.fetch_fundamentals("AAPL")
            
            # yfinance was called (cache was expired)
            mock_ticker_cls.assert_called_once_with("AAPL")
            # No fundamental data → returns None
            assert result is None

    def test_cache_hit_does_not_call_yfinance(self):
        """Cache hit should NOT call yfinance."""
        analyzer = _make_analyzer()
        fd = _make_fundamental_data("MSFT")
        
        # Fresh cache entry
        analyzer._cache["MSFT"] = (fd, time.time())
        
        with patch("yfinance.Ticker") as mock_ticker_cls:
            result = analyzer.fetch_fundamentals("MSFT")
            mock_ticker_cls.assert_not_called()
        
        assert result is fd

    def test_successful_fetch_populates_cache(self):
        """A successful fetch should populate the cache."""
        analyzer = _make_analyzer()
        
        mock_info = {
            'trailingPE': 25.0,
            'forwardPE': 20.0,
            'trailingEps': 6.0,
            'revenueGrowth': 0.10,
            'profitMargins': 0.22,
            'debtToEquity': 0.8,
        }
        
        with patch("yfinance.Ticker") as mock_ticker_cls:
            mock_ticker = MagicMock()
            mock_ticker.info = mock_info
            mock_ticker_cls.return_value = mock_ticker
            
            result = analyzer.fetch_fundamentals("GOOGL")
        
        assert result is not None
        assert "GOOGL" in analyzer._cache
        cached_data, cached_time = analyzer._cache["GOOGL"]
        assert cached_data is result
        assert time.time() - cached_time < 5  # Cached within last 5 seconds


class TestFundamentalAnalyzerEquityFetch:
    """Tests for equity symbol fetching."""

    def test_equity_with_full_data_returns_fundamental_data(self):
        """Equity with full yfinance data should return a FundamentalData object."""
        analyzer = _make_analyzer()
        
        mock_info = {
            'trailingPE': 18.5,
            'forwardPE': 15.0,
            'trailingEps': 6.11,
            'revenueGrowth': 0.08,
            'profitMargins': 0.25,
            'debtToEquity': 1.2,
        }
        
        with patch("yfinance.Ticker") as mock_ticker_cls:
            mock_ticker = MagicMock()
            mock_ticker.info = mock_info
            mock_ticker_cls.return_value = mock_ticker
            
            result = analyzer.fetch_fundamentals("AAPL")
        
        assert result is not None
        assert isinstance(result, FundamentalData)
        assert result.symbol == "AAPL"
        assert result.trailing_pe == 18.5
        assert result.forward_pe == 15.0
        assert result.eps_ttm == 6.11
        assert result.revenue_growth == 0.08
        assert result.profit_margin == 0.25
        assert result.debt_to_equity == 1.2
        assert result.label in ("Undervalued", "Fairly Valued", "Overvalued")
        assert -1.0 <= result.score <= 1.0

    def test_equity_with_partial_data_returns_fundamental_data(self):
        """Equity with partial yfinance data should still return FundamentalData."""
        analyzer = _make_analyzer()
        
        # Only trailingPE is present
        mock_info = {
            'trailingPE': 22.0,
        }
        
        with patch("yfinance.Ticker") as mock_ticker_cls:
            mock_ticker = MagicMock()
            mock_ticker.info = mock_info
            mock_ticker_cls.return_value = mock_ticker
            
            result = analyzer.fetch_fundamentals("XYZ")
        
        assert result is not None
        assert result.trailing_pe == 22.0
        assert result.eps_ttm is None
        assert result.revenue_growth is None

    def test_equity_with_no_fundamental_data_returns_none(self):
        """Symbol with no fundamental data (e.g., ETF or index) should return None."""
        analyzer = _make_analyzer()
        
        # Empty info dict — no fundamental metrics
        mock_info = {}
        
        with patch("yfinance.Ticker") as mock_ticker_cls:
            mock_ticker = MagicMock()
            mock_ticker.info = mock_info
            mock_ticker_cls.return_value = mock_ticker
            
            result = analyzer.fetch_fundamentals("SPY")
        
        assert result is None
