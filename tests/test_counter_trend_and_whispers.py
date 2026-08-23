"""
Regression tests for:
1. Market whispers datetime fix — RSS dates parsed by `parsedate_to_datetime`
   are timezone-aware while `datetime.min` / `datetime.now()` are naive;
   sorting / recency-cutoff comparisons must never raise
   "can't compare offset-naive and offset-aware datetimes".
2. Counter-trend analyzer improvements — live-price momentum confirmation
   (falling-knife filter), data-driven narrative score refresh, and the new
   momentum fields on CounterTrendSignal.
"""
import math

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# 1. MARKET WHISPERS — NAIVE/AWARE DATETIME SAFETY
# ─────────────────────────────────────────────────────────────────────────────
def test_naive_utc_normalizes_aware_datetimes():
    from datetime import datetime, timezone
    from email.utils import parsedate_to_datetime

    from news_analysis_engine import _naive_utc

    aware = parsedate_to_datetime("Mon, 21 Aug 2026 14:30:00 GMT")
    naive = datetime.now()

    # Aware datetimes become naive UTC (no tzinfo) so they compare with datetime.min
    out = _naive_utc(aware)
    assert out.tzinfo is None
    assert _naive_utc(naive) == naive
    assert _naive_utc(None) is None


def test_market_whispers_sort_with_mixed_naive_and_aware_timestamps():
    """A corpus mixing aware (RSS) and naive timestamps must not crash."""
    from datetime import datetime, timezone
    from email.utils import parsedate_to_datetime

    from news_analysis_engine import NewsAnalysisEngine

    eng = NewsAnalysisEngine.__new__(NewsAnalysisEngine)
    aware = parsedate_to_datetime("Mon, 21 Aug 2026 14:30:00 GMT")
    naive = datetime.now()

    class FakeArticle:
        def __init__(self, title, dt, source="Reuters", syms=("NVDA",)):
            self.title = title
            self.summary = "summary"
            self.source = source
            self.published_at = dt
            self.symbols_mentioned = list(syms)
            self.sentiment_score = 0.5
            self.relevance_score = 0.6
            self.tags = []
            self.url = ""
            self.content = ""

    articles = [
        FakeArticle("NVDA earnings beat", aware),
        FakeArticle("NVDA chip news", naive),
        FakeArticle("NVDA another", naive, source="Bloomberg"),
        FakeArticle("no symbol", naive, syms=()),
    ]
    eng.fetch_and_process_news = lambda: articles
    eng._infer_whisper_type = lambda a: "earnings"

    whispers = eng.get_market_whispers("NVDA")
    assert len(whispers) == 3  # the no-symbol article is excluded
    # Timestamps flow through unharmed (either naive or aware, never crashed)
    assert all(w.timestamp is not None for w in whispers)


def test_get_recent_articles_mixed_timestamps():
    """_get_recent_articles recency cutoff must handle aware RSS timestamps."""
    from datetime import datetime, timedelta, timezone
    from email.utils import parsedate_to_datetime

    from news_analysis_engine import NewsAnalysisEngine

    eng = NewsAnalysisEngine.__new__(NewsAnalysisEngine)
    fresh_aware = parsedate_to_datetime("Mon, 21 Aug 2026 14:30:00 GMT")
    stale_aware = parsedate_to_datetime("Mon, 21 Jul 2026 14:30:00 GMT")
    fresh_naive = datetime.now()

    class FakeArticle:
        def __init__(self, dt):
            self.title = "t"
            self.summary = "s"
            self.source = "Reuters"
            self.published_at = dt
            self.symbols_mentioned = ["AAPL"]
            self.sentiment_score = 0.0
            self.relevance_score = 0.5
            self.tags = []
            self.url = ""
            self.content = ""

    articles = [
        FakeArticle(fresh_aware),
        FakeArticle(stale_aware),
        FakeArticle(fresh_naive),
    ]
    eng.fetch_and_process_news = lambda: articles
    recent = eng._get_recent_articles(hours_back=24 * 7)
    assert len(recent) == 2  # the stale aware article is excluded


def test_news_summary_mixed_timestamps():
    """get_news_summary_for_symbol must not crash on mixed timestamps."""
    from datetime import datetime
    from email.utils import parsedate_to_datetime

    from news_analysis_engine import NewsAnalysisEngine

    eng = NewsAnalysisEngine.__new__(NewsAnalysisEngine)
    aware = parsedate_to_datetime("Mon, 21 Aug 2026 14:30:00 GMT")
    naive = datetime.now()

    class FakeArticle:
        def __init__(self, dt):
            self.title = "t"
            self.summary = "s"
            self.source = "Reuters"
            self.published_at = dt
            self.symbols_mentioned = ["NVDA"]
            self.sentiment_score = 0.0
            self.relevance_score = 0.5
            self.tags = []
            self.url = ""
            self.content = ""

    eng.fetch_and_process_news = lambda: [FakeArticle(aware), FakeArticle(naive)]
    eng.get_sentiment_for_symbol = lambda sym: {}
    eng.get_market_whispers = lambda sym: []
    summary = eng.get_news_summary_for_symbol("NVDA")
    assert summary  # must return a dict, not raise


# ─────────────────────────────────────────────────────────────────────────────
# 2. COUNTER-TREND ANALYZER — MOMENTUM CONFIRMATION
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def ct():
    from counter_trend_analyzer import CounterTrendAnalyzer

    return CounterTrendAnalyzer()


def _synth(start, n=40, drift=0.0, noise=0.01):
    out = []
    v = start
    for i in range(n):
        v = v * (1 + drift + noise * math.sin(i))
        out.append(v)
    return out


def test_momentum_filter_reversing_boosts_strength(ct):
    sig = ct.generate_counter_signals(divergence_threshold=15.0, min_strength=45.0)
    gld = [s for s in sig if s.instrument == "GLD"]
    assert gld, "GLD LONG fade signal expected"
    base = gld[0].signal_strength

    # GLD rising gently = price moving in the LONG fade direction
    out = ct.apply_momentum_filter(sig, {"GLD": _synth(210, drift=0.006)})
    gld2 = [s for s in out if s.instrument == "GLD"]
    assert gld2
    assert gld2[0].momentum_state == "REVERSING"
    assert gld2[0].momentum_score > 0
    assert gld2[0].signal_strength > base
    assert gld2[0].live_price > 0
    assert gld2[0].entry_condition


def test_momentum_filter_falling_knife_dropped(ct):
    sig = ct.generate_counter_signals(divergence_threshold=15.0, min_strength=45.0)
    # GLD crashing -4%/day: classic falling knife for a LONG fade -> dropped
    out = ct.apply_momentum_filter(sig, {"GLD": _synth(210, drift=-0.04)})
    survivors = [s for s in out if s.instrument == "GLD"]
    assert survivors == []


def test_momentum_filter_mild_decline_downweighted_not_dropped(ct):
    sig = ct.generate_counter_signals(divergence_threshold=15.0, min_strength=45.0)
    gld = [s for s in sig if s.instrument == "GLD"][0]
    base = gld.signal_strength

    out = ct.apply_momentum_filter(sig, {"GLD": _synth(210, drift=-0.015)})
    survivors = [s for s in out if s.instrument == "GLD"]
    assert survivors, "mild decline should be retained, not dropped"
    assert survivors[0].momentum_state == "TREND_ACCELERATING"
    assert survivors[0].signal_strength < base
    assert "stall" in survivors[0].entry_condition.lower()


def test_momentum_filter_no_data_leaves_signals_untouched(ct):
    sig = ct.generate_counter_signals(divergence_threshold=15.0, min_strength=45.0)
    before = [(s.instrument, s.signal_strength, s.confidence) for s in sig]
    out = ct.apply_momentum_filter(sig, {})
    assert len(out) == len(sig)
    assert all(s.momentum_state == "NO_DATA" for s in out)
    assert [(s.instrument, s.signal_strength, s.confidence) for s in out] == before


def test_momentum_filter_missing_instrument_degrades_gracefully(ct):
    sig = ct.generate_counter_signals(divergence_threshold=15.0, min_strength=45.0)
    # price data only for an instrument not in the signal set + one with
    # insufficient history
    out = ct.apply_momentum_filter(
        sig, {"GLD": [1.0, 2.0], "QQQ": _synth(500, drift=0.005)}
    )
    qqq = [s for s in out if s.instrument == "QQQ"]
    assert qqq and qqq[0].momentum_state != "NO_DATA"
    assert any(s.momentum_state == "NO_DATA" for s in out)


def test_signal_has_momentum_fields_by_default(ct):
    sig = ct.generate_counter_signals(divergence_threshold=15.0, min_strength=45.0)
    s = sig[0]
    assert s.momentum_state == "NO_DATA"
    assert s.momentum_score == 0.0
    assert s.price_confirmation == ""
    assert s.live_price == 0.0
    assert s.entry_condition == ""


# ─────────────────────────────────────────────────────────────────────────────
# 3. COUNTER-TREND ANALYZER — DATA-DRIVEN SCORE REFRESH
# ─────────────────────────────────────────────────────────────────────────────
def test_refresh_from_market_data_updates_expected_themes(ct):
    updated = ct.refresh_from_market_data(
        {
            "vix": 28.0,
            "us_10y_yield": 3.8,
            "gold_price": 2700.0,
            "usd_index": 96.0,
            "usdjpy": 143.0,
        }
    )
    assert "Fed Hawkishness Permanence" in updated
    assert "Gold is a Relic" in updated
    assert "JPY Carry Trade is Safe" in updated
    # Scores actually changed from the static seeds
    narr = {n.theme: n for n in ct.get_all_narratives()}
    assert narr["Fed Hawkishness Permanence"].consensus_score < 65.0  # high VIX, low yield
    assert narr["Gold is a Relic"].fundamental_score > 20.0  # gold at 2700


def test_refresh_from_market_data_empty_is_noop(ct):
    updated = ct.refresh_from_market_data({})
    assert updated == []
    narr = {n.theme: n for n in ct.get_all_narratives()}
    assert narr["Gold is a Relic"].fundamental_score == 20.0  # untouched


def test_refresh_clamps_scores_to_range(ct):
    ct.refresh_from_market_data({"usd_index": 300.0, "gold_price": 100000.0})
    for n in ct.get_all_narratives():
        assert 5.0 <= n.consensus_score <= 95.0
        assert 5.0 <= n.fundamental_score <= 95.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. SINGLETON + LEGACY API STABILITY
# ─────────────────────────────────────────────────────────────────────────────
def test_get_counter_trend_analyzer_is_singleton():
    from counter_trend_analyzer import get_counter_trend_analyzer

    assert get_counter_trend_analyzer() is get_counter_trend_analyzer()


def test_legacy_generate_counter_signals_still_works(ct):
    """The pre-existing API must keep working with no price data supplied."""
    sig = ct.generate_counter_signals(divergence_threshold=18.0, min_strength=50.0)
    assert sig
    s = sig[0]
    # All pre-existing fields present
    for attr in (
        "theme", "direction", "asset_class", "instrument", "signal_strength",
        "confidence", "divergence_score", "entry_rationale", "macro_narrative",
        "counter_thesis", "key_risks", "catalyst_needed", "time_horizon",
        "position_size_pct",
    ):
        assert hasattr(s, attr)
    # New fields default cleanly
    assert s.momentum_state == "NO_DATA"
    assert s.entry_condition == ""
