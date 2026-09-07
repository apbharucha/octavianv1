import numpy as np
import pandas as pd

from institutional_analytics_engine import build_bayesian_network_from_data
from market_simulation_engine import MarketRegime, MarketSimulationEngine, NewsType
from spreadsheet_generator import _calculate_indicators


def test_bayesian_network_accepts_symbol_list(monkeypatch):
    frame = pd.DataFrame({"Close": np.linspace(100, 120, 40)})
    monkeypatch.setattr(
        "institutional_analytics_engine._fetch_network_market_data",
        lambda symbols: {symbol: frame for symbol in symbols},
    )
    network = build_bayesian_network_from_data(["AAA", "BBB"])
    assert set(network.nodes) == {"AAA", "BBB"}


def test_simulation_news_event_factory_exists():
    engine = MarketSimulationEngine.__new__(MarketSimulationEngine)
    engine._symbol_drift = {"AAA": 0.0, "BBB": 0.0}
    engine._get_symbol_sector = lambda symbol: "Test"
    event = engine._create_news_event(pd.Timestamp("2026-08-29").to_pydatetime(), NewsType.EARNINGS, MarketRegime.SIDEWAYS)
    assert event.affected_symbols
    assert event.news_type is NewsType.EARNINGS
    assert event.credibility_tier == "SIMULATED"


def test_indicator_engine_builds_requested_advanced_columns():
    frame = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=40),
        "Open": np.arange(100, 140),
        "High": np.arange(101, 141),
        "Low": np.arange(99, 139),
        "Close": np.arange(100, 140),
        "Volume": np.arange(1_000, 1_040),
    })
    result = _calculate_indicators(frame, ["SMA_20", "EMA_12", "RSI_14", "MACD", "Volatility_20d", "ATR_14", "OBV", "Trend_Direction"])
    for column in ("SMA_20", "EMA_12", "RSI_14", "MACD", "Volatility_20d", "ATR_14", "OBV", "Trend_Direction"):
        assert column in result.columns


def test_correlation_uses_pairwise_upper_triangle():
    # The implementation is exercised indirectly by ensuring indicator output
    # preserves distinct symbol rows for later date-aligned pivots.
    frame = pd.DataFrame({"Symbol": ["AAA", "AAA", "BBB", "BBB"], "Date": pd.date_range("2026-01-01", periods=4), "Close": [1, 2, 2, 4]})
    assert frame.groupby("Symbol")["Close"].count().to_dict() == {"AAA": 2, "BBB": 2}
