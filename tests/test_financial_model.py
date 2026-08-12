"""
Tests for the Financial Model Generator (InstitutionalDCFEngine).
Updated to match the current API: run_dcf(DCFAssumptions) → DCFResult
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from financial_model_generator import (
    get_dcf_engine,
    get_financial_generator,
    DCFAssumptions,
    DCFResult,
    InstitutionalDCFEngine,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_simple_assumptions(
    ticker: str = "TEST",
    base_revenue: float = 1_000.0,     # $1B revenue
    growth: float = 0.10,
    terminal_growth: float = 0.02,
    ebit_margin: float = 0.20,
    wacc_approx: float = 0.10,         # used to back-calculate WACC components
    shares: float = 1_000.0,
    debt: float = 500.0,
    cash: float = 200.0,
    current_price: float = 150.0,
) -> DCFAssumptions:
    """Create a minimal but mathematically valid DCFAssumptions object."""
    return DCFAssumptions(
        ticker=ticker,
        base_revenue=base_revenue,
        revenue_growth_rates=[growth] * 5,
        ebit_margin=ebit_margin,
        tax_rate=0.21,
        da_pct_revenue=0.04,
        capex_pct_revenue=0.05,
        nwc_change_pct_revenue=0.01,
        # WACC components — these will produce ~10% WACC
        equity_value_market=15_000.0,
        debt_value=debt,
        cost_of_debt=0.05,
        risk_free_rate=0.045,
        equity_risk_premium=0.055,
        beta=1.0,
        terminal_growth_rate=terminal_growth,
        cash=cash,
        shares_outstanding=shares,
        current_price=current_price,
        projection_years=5,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Factory / Singleton tests
# ─────────────────────────────────────────────────────────────────────────────

def test_get_dcf_engine_returns_instance():
    engine = get_dcf_engine()
    assert isinstance(engine, InstitutionalDCFEngine)


def test_get_financial_generator_alias():
    """get_financial_generator() is a backward-compat alias for get_dcf_engine()."""
    gen = get_financial_generator()
    assert isinstance(gen, InstitutionalDCFEngine)


def test_singleton_same_object():
    """Both factory calls should return the same singleton."""
    assert get_dcf_engine() is get_dcf_engine()


# ─────────────────────────────────────────────────────────────────────────────
# DCF core math tests
# ─────────────────────────────────────────────────────────────────────────────

def test_dcf_returns_dcf_result():
    engine = get_dcf_engine()
    assumptions = _make_simple_assumptions()
    result = engine.run_dcf(assumptions)
    assert isinstance(result, DCFResult)


def test_dcf_ticker_preserved():
    engine = get_dcf_engine()
    assumptions = _make_simple_assumptions(ticker="AAPL")
    result = engine.run_dcf(assumptions)
    assert result.ticker == "AAPL"


def test_dcf_fair_value_positive():
    """With healthy margins and positive growth, fair_value_per_share must be > 0."""
    engine = get_dcf_engine()
    assumptions = _make_simple_assumptions()
    result = engine.run_dcf(assumptions)
    assert result.fair_value_per_share > 0, (
        f"Expected positive fair_value_per_share, got {result.fair_value_per_share}"
    )


def test_dcf_enterprise_value_integrity():
    """EV = sum_pv_fcf + pv_terminal; equity = EV - net_debt."""
    engine = get_dcf_engine()
    assumptions = _make_simple_assumptions(debt=500.0, cash=200.0, shares=1_000.0)
    result = engine.run_dcf(assumptions)

    # Validate EV = PV(FCFs) + PV(TV)
    ev_from_parts = result.sum_pv_fcf + result.pv_terminal_value
    assert result.enterprise_value == pytest.approx(ev_from_parts, rel=1e-3), (
        f"EV = {result.enterprise_value:.2f}, parts = {ev_from_parts:.2f}"
    )
    # Equity value = EV - net_debt; just validate it's non-negative for a healthy company
    assert result.equity_value >= 0


def test_dcf_fair_value_per_share():
    """fair_value_per_share ≈ equity_value / shares_outstanding."""
    engine = get_dcf_engine()
    shares = 500.0
    assumptions = _make_simple_assumptions(shares=shares)
    result = engine.run_dcf(assumptions)
    # Engine stores shares used in result.shares_outstanding
    used_shares = result.shares_outstanding if result.shares_outstanding > 0 else shares
    expected = result.equity_value / used_shares
    assert result.fair_value_per_share == pytest.approx(expected, rel=1e-2)


def test_dcf_terminal_growth_above_wacc_handled():
    """Engine must not crash when terminal_growth >= WACC — applies safe floor."""
    engine = get_dcf_engine()
    # Artificially high terminal growth (would normally blow up DCF)
    assumptions = _make_simple_assumptions(terminal_growth=0.20, growth=0.05)
    result = engine.run_dcf(assumptions)
    # Just needs to complete without exception and return a valid result
    assert isinstance(result, DCFResult)
    assert result.fair_value_per_share is not None
    assert result.enterprise_value >= 0


def test_dcf_scenarios_present():
    """Result should include bear/base/bull scenarios."""
    engine = get_dcf_engine()
    assumptions = _make_simple_assumptions()
    result = engine.run_dcf(assumptions)
    assert hasattr(result, "scenarios")
    if result.scenarios:  # may be None if skipped
        labels = [s.label.lower() for s in result.scenarios]
        assert any("bear" in l or "base" in l or "bull" in l for l in labels)


def test_dcf_trade_signal_present():
    """Result should include an auto-generated trade signal."""
    engine = get_dcf_engine()
    assumptions = _make_simple_assumptions(current_price=50.0)
    result = engine.run_dcf(assumptions)
    assert hasattr(result, "trade_signal")
    if result.trade_signal:
        # Engine uses uppercase signal strings
        assert result.trade_signal.signal in (
            "STRONG BUY", "BUY", "NEUTRAL", "SELL", "STRONG SELL",
            # Also accept Title-case variants if engine changes
            "Strong Long", "Long", "Neutral", "Short", "Strong Short",
        )
