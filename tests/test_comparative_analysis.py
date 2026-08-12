import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock
from comparative_analysis_engine import (
    get_comparative_engine,
    detect_asset_type,
    AssetScore,
    CrossAssetStrategy,
    ComparativeResult,
    ComparativeAnalysisEngine,
)


# ─────────────────────────────────────────────────────────────────────────────
# detect_asset_type tests
# ─────────────────────────────────────────────────────────────────────────────

def test_detect_asset_type():
    assert detect_asset_type("AAPL") == "stock"
    assert detect_asset_type("BTC-USD") == "crypto"
    # EUR/USD — 7-char with slash → forex
    assert detect_asset_type("EUR/USD") == "forex"
    # Futures (=F suffix)
    assert detect_asset_type("ES=F") == "futures"
    # CL=F (Crude Oil) IS a futures contract — engine correctly returns "futures"
    assert detect_asset_type("CL=F") == "futures"
    # Options long-form ticker
    assert detect_asset_type("AAPL260619C00150000") == "options"
    # Unknown defaults to stock
    assert detect_asset_type("UNKNOWN") == "stock"
    # Crypto base symbols
    assert detect_asset_type("BTC") == "crypto"
    assert detect_asset_type("ETH") == "crypto"
    # Forex with =X
    assert detect_asset_type("EURUSD=X") == "forex"


# ─────────────────────────────────────────────────────────────────────────────
# ComparativeAnalysisEngine.run_analysis integration test
# ─────────────────────────────────────────────────────────────────────────────

@patch("data_sources.get_stock")
def test_comparative_engine_run(mock_get_stock):
    """
    Full integration test for run_analysis with mocked data sources and engines.
    Validates result structure, asset presence, score types, and correlation matrix.
    """
    dates = pd.date_range(end="2026-06-08", periods=100, freq="D")
    mock_df = pd.DataFrame({
        "Close": np.linspace(100, 110, 100),
        "Volume": np.linspace(1_000_000, 2_000_000, 100),
    }, index=dates)
    mock_get_stock.return_value = mock_df

    # Use a fresh engine instance so previous singleton state doesn't interfere
    engine = ComparativeAnalysisEngine()

    # ── Mock the ML engine ──────────────────────────────────────────────────
    # The engine calls: self.ml_engine.analyze_symbol_ensemble(df, symbol, asset_type, fast_mode)
    # and reads: out.get("alpha_score"), out.get("predicted_return")
    mock_ml = MagicMock()
    mock_ml.analyze_symbol_ensemble.return_value = {
        "alpha_score": 72.0,
        "predicted_return": 0.065,   # 6.5% → stored as expected_return_30d = 6.5
        "model_breakdown": {"rf": 0.7, "xgb": 0.75},
    }
    engine._ml_engine = mock_ml

    # ── Mock the macro engine ───────────────────────────────────────────────
    # The engine calls: self.macro_engine.build_dashboard()
    # and reads: dash.macro_regime.name
    mock_macro_dash = MagicMock()
    mock_macro_dash.macro_regime.name = "Goldilocks"   # triggers macro_bullish path → score = 65.0
    mock_macro = MagicMock()
    mock_macro.build_dashboard.return_value = mock_macro_dash
    engine._macro_engine = mock_macro

    # ── Mock the fundamentals engine ────────────────────────────────────────
    # The engine calls: self.fund_engine.fetch_fundamentals(symbol)
    # and reads: getattr(fund, "score", 0.0)
    # fundamental_score = clip((fund_raw + 1.0) / 2.0 * 100, 0, 100)
    # For score = 0.64: (0.64 + 1.0) / 2.0 * 100 = 82.0
    mock_fund_result = MagicMock()
    mock_fund_result.score = 0.64
    mock_fund = MagicMock()
    mock_fund.fetch_fundamentals.return_value = mock_fund_result
    engine._fund_engine = mock_fund

    # ── Disable AI thesis (no LM Studio in test env) ───────────────────────
    engine._enrich_strategies_with_ai = lambda strategies, scores, cfg: strategies

    assets = [
        {"symbol": "AAPL", "asset_type": "stock"},
        {"symbol": "MSFT", "asset_type": "stock"},
    ]
    model_config = {
        "use_quant": True,
        "use_fundamentals": True,
        "use_macro": True,
        "use_technical": True,
    }

    result = engine.run_analysis(assets, model_config)

    # ── Result structure ────────────────────────────────────────────────────
    assert isinstance(result, ComparativeResult)
    assert "AAPL" in result.assets
    assert "MSFT" in result.assets

    # ── Score sanity ────────────────────────────────────────────────────────
    aapl = result.assets["AAPL"]
    assert aapl.error is None, f"AAPL analysis failed: {aapl.error}"

    # Quant score = alpha_score from mock ML
    assert aapl.quant_score == pytest.approx(72.0, abs=0.1)

    # Expected return from ML: predicted_return * 100 = 6.5
    assert aapl.expected_return_30d == pytest.approx(6.5, abs=0.1)

    # Fundamental score: (0.64 + 1.0) / 2.0 * 100 = 82.0
    assert aapl.fundamental_score == pytest.approx(82.0, abs=0.1)

    # Macro: Goldilocks regime → 65.0
    assert aapl.macro_score == pytest.approx(65.0, abs=0.1)

    # Overall = mean of [technical, quant, macro, fundamental]
    # technical is computed from the mock price data (deterministic), so just validate it's in range
    assert 0 <= aapl.overall_score <= 100
    assert aapl.decision in ("STRONG BUY", "BUY", "NEUTRAL", "SELL", "STRONG SELL")
    assert 0.0 <= aapl.conviction <= 1.0

    # ── Rankings ────────────────────────────────────────────────────────────
    assert set(result.ranked_by_score) == {"AAPL", "MSFT"}
    assert set(result.ranked_by_momentum) == {"AAPL", "MSFT"}

    # ── Correlation matrix ──────────────────────────────────────────────────
    assert isinstance(result.correlation_matrix, pd.DataFrame)
    assert "AAPL" in result.correlation_matrix.columns
    assert "MSFT" in result.correlation_matrix.index
    # Self-correlation must be 1.0
    assert result.correlation_matrix.loc["AAPL", "AAPL"] == pytest.approx(1.0, abs=1e-6)

    # ── Comparative conclusions ──────────────────────────────────────────────
    assert isinstance(result.comparative_conclusions, dict)
    assert len(result.comparative_conclusions) >= 1  # 1 pair: AAPL vs MSFT

    # ── Run time tracked ────────────────────────────────────────────────────
    assert result.run_time_ms >= 0


# ─────────────────────────────────────────────────────────────────────────────
# Strategy generation tests
# ─────────────────────────────────────────────────────────────────────────────

def test_strategy_generation_pairs_trade():
    """Verify pairs trade strategy is generated when score spread is large enough."""
    engine = ComparativeAnalysisEngine()

    scores = {
        "AAPL": AssetScore(
            symbol="AAPL", asset_type="stock",
            current_price=180, change_1d=0.5, change_1w=1.2, change_1m=3.5,
            volatility_annual=0.22,
            overall_score=78.0, decision="STRONG BUY", conviction=0.8,
            expected_return_30d=5.0,
        ),
        "META": AssetScore(
            symbol="META", asset_type="stock",
            current_price=90, change_1d=-0.3, change_1w=-1.5, change_1m=-4.0,
            volatility_annual=0.35,
            overall_score=32.0, decision="SELL", conviction=0.6,
            expected_return_30d=-3.0,
        ),
    }
    corr = pd.DataFrame({"AAPL": [1.0, 0.65], "META": [0.65, 1.0]},
                        index=["AAPL", "META"])

    strategies = engine._generate_strategies(scores, corr)
    assert len(strategies) >= 1

    types = [s.type for s in strategies]
    assert "PAIRS_TRADE" in types

    pairs = [s for s in strategies if s.type == "PAIRS_TRADE"][0]
    assert "AAPL" in pairs.assets_long
    assert "META" in pairs.assets_short


def test_strategy_generation_low_correlation_play():
    """Verify diversification play is generated for near-zero correlation pairs."""
    engine = ComparativeAnalysisEngine()

    scores = {
        "AAPL": AssetScore(
            symbol="AAPL", asset_type="stock",
            current_price=180, change_1d=0.5, change_1w=1.2, change_1m=3.5,
            volatility_annual=0.22,
            overall_score=65.0, decision="BUY", conviction=0.6,
            expected_return_30d=4.0,
        ),
        "GC=F": AssetScore(
            symbol="GC=F", asset_type="futures",
            current_price=2300, change_1d=0.1, change_1w=0.8, change_1m=1.1,
            volatility_annual=0.14,
            overall_score=60.0, decision="BUY", conviction=0.5,
            expected_return_30d=2.0,
        ),
    }
    # Near-zero correlation → should trigger diversification play
    corr = pd.DataFrame({"AAPL": [1.0, 0.05], "GC=F": [0.05, 1.0]},
                        index=["AAPL", "GC=F"])

    strategies = engine._generate_strategies(scores, corr)
    types = [s.type for s in strategies]
    # Either ROTATION (diversification) or at least SPREAD should appear
    assert len(strategies) >= 1
    assert any(t in types for t in ("ROTATION", "SPREAD", "BASKET_LONG"))
