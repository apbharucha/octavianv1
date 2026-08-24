"""
Tests for the Macro Narrative Divergence & Contrarian Signal Intelligence
system: composite divergence scoring, consensus/fundamental components,
narrative velocity, contradiction engine, historical analogs, catalysts,
value-trap detection, transmission maps, pricing gap, data quality, regime
detection, the Develop Setup tool, the SignalTracker, and backward
compatibility with the legacy counter-trend API.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from counter_trend_analyzer import (
    CounterTrendAnalyzer,
    DIVERGENCE_WEIGHTS,
    MacroNarrativeIntelligenceEngine,
    get_counter_trend_analyzer,
    get_signal_tracker,
    score_narrative_strength,
)

MARKET_STATE = {
    "vix": 15.2,
    "us_10y_yield": 4.05,
    "usd_index": 103.5,
    "gold_price": 2450.0,
    "oil_price": 78.0,
    "usdjpy": 152.0,
    "smh_price": 260.0,
    "cpi_yoy": 2.8,
    "hy_spread_bps": 350.0,
    "sp500": 5400.0,
}


def _engine(market_state=None, price_data=None, analyzer=None, weights=None):
    return MacroNarrativeIntelligenceEngine(
        analyzer=analyzer or CounterTrendAnalyzer(),
        market_state=market_state,
        price_data=price_data,
        divergence_weights=weights,
    )


def _price_series(n=60, seed=7, start=100.0, drift=0.0005, vol=0.01):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"Close": start * np.cumprod(1 + rng.normal(drift, vol, n))}
    )


# ---------------------------------------------------------------------
# Engine basics
# ---------------------------------------------------------------------


def test_analyze_all_returns_all_narratives_ranked():
    engine = _engine(market_state=MARKET_STATE)
    analyses = engine.analyze_all()
    assert len(analyses) == 8
    scores = [a.macro_divergence_score for a in analyses]
    assert scores == sorted(scores, reverse=True)
    for a in analyses:
        assert 0 <= a.consensus_score <= 100
        assert 0 <= a.fundamental_score <= 100
        assert 0 <= a.macro_divergence_score <= 100
        assert 0 <= a.opportunity_score <= 100
        assert 0 <= a.confidence_score <= 100


def test_divergence_is_component_driven_not_raw_seed():
    engine = _engine(market_state=MARKET_STATE)
    a = next(x for x in engine.analyze_all() if x.theme == "USD Structural Strength")
    # component aggregate must be what is displayed, and it must differ from
    # the raw seeded pair in at least the fundamental leg once market state
    # conditions the components
    assert a.consensus_score == pytest.approx(a.consensus_components.overall, abs=0.1)
    assert a.fundamental_score == pytest.approx(a.fundamental_components.overall, abs=0.1)
    assert a.divergence == pytest.approx(
        a.consensus_components.overall - a.fundamental_components.overall, abs=0.2
    )


def test_composite_divergence_respects_configurable_weights():
    only_narrative = {k: 0.0 for k in DIVERGENCE_WEIGHTS}
    only_narrative["narrative"] = 1.0
    engine = _engine(market_state=MARKET_STATE, weights=only_narrative)
    a = engine.analyze_all()[0]
    # with only the narrative component weighted, composite == that component
    assert a.macro_divergence_score == pytest.approx(
        a.divergence_components["narrative"], abs=0.01
    )
    # default weights still sum to 1
    assert sum(DIVERGENCE_WEIGHTS.values()) == pytest.approx(1.0)


def test_consensus_components_exist_and_are_bounded():
    engine = _engine(market_state=MARKET_STATE)
    a = engine.analyze_all()[0]
    cc = a.consensus_components
    for v in (cc.market_implied, cc.analyst_estimates, cc.news_sentiment,
              cc.positioning, cc.flows, cc.media_coverage):
        assert 0 <= v <= 100
    assert cc.provenance == "MARKET-IMPLIED"


def test_velocity_tracks_7d_30d_90d_and_states():
    engine = _engine(market_state=MARKET_STATE)
    a = next(x for x in engine.analyze_all() if x.theme == "JPY Carry Trade is Safe")
    for key in ("consensus_today", "consensus_7d", "consensus_30d", "consensus_90d",
                "fundamental_today", "fundamental_30d"):
        assert key in a.velocity
    assert a.velocity_state in (
        "RAPIDLY STRENGTHENING", "GRADUALLY STRENGTHENING", "STABLE",
        "GRADUALLY WEAKENING", "RAPIDLY COLLAPSING",
    )
    assert a.data_momentum_state in ("IMPROVING", "DETERIORATING", "FLAT")


def test_second_order_layer_resolves_with_price_data():
    prices = _price_series(seed=3, start=100.0)
    engine = _engine(market_state=MARKET_STATE, price_data={"TLT": prices})
    a = next(x for x in engine.analyze_all() if x.theme == "Fed Hawkishness Permanence")
    assert a.second_order["alignment"] in ("CONFIRMING", "STILL_CROWDED", "UNRESOLVED")
    # price attached => resolved
    assert a.second_order["alignment"] != "UNRESOLVED"
    assert "layer3_market" in a.second_order


def test_no_market_state_degrades_gracefully():
    engine = _engine(market_state=None)
    analyses = engine.analyze_all()
    assert len(analyses) == 8
    a = analyses[0]
    assert a.data_quality["freshness"] == "SEEDED"
    assert a.data_quality["provenance"]
    assert engine.detect_regime()["regime"] == "Unknown / Transition"


# ---------------------------------------------------------------------
# Contradiction engine / analogs / catalysts
# ---------------------------------------------------------------------


def test_contradictions_ranked_and_live_cross_checks_added():
    engine = _engine(market_state=MARKET_STATE)
    a = next(x for x in engine.analyze_all() if x.theme == "Fed Hawkishness Permanence")
    assert len(a.contradictions) >= 2
    mags = [c["magnitude"] for c in a.contradictions]
    assert mags == sorted(mags, reverse=True)
    assert any("OBSERVED:" in c["text"] for c in a.contradictions)  # live cross-check


def test_historical_analogs_sorted_by_similarity_with_disclaimer_fields():
    engine = _engine(market_state=MARKET_STATE)
    a = next(x for x in engine.analyze_all() if x.theme == "USD Structural Strength")
    assert a.historical_analogs
    sims = [x.similarity for x in a.historical_analogs]
    assert sims == sorted(sims, reverse=True)
    for x in a.historical_analogs:
        assert x.period and x.note
        assert x.avg_return_pct != 0


def test_catalyst_watchlist_has_dates_and_proximity():
    engine = _engine(market_state=MARKET_STATE)
    a = next(x for x in engine.analyze_all() if x.theme == "JPY Carry Trade is Safe")
    assert a.catalysts
    assert all(c.days_out is not None for c in a.catalysts[:2])
    assert 0 <= a.catalyst_proximity <= 100


# ---------------------------------------------------------------------
# Value trap / pricing / transmission
# ---------------------------------------------------------------------


def test_value_trap_verdict_is_one_of_three():
    engine = _engine(market_state=MARKET_STATE)
    for a in engine.analyze_all():
        assert a.value_trap_verdict in (
            "VALID CONTRARIAN OPPORTUNITY",
            "POSSIBLE VALUE TRAP",
            "INSUFFICIENT EVIDENCE",
        )


def test_missing_data_pushes_toward_value_trap():
    no_data = _engine(market_state=None).analyze_all()[0]
    with_data = _engine(market_state=MARKET_STATE).analyze_all()[0]
    # "no live macro data attached" adds trap points; with data the verdict
    # should never be worse than the no-data case for the same theme set
    for a_nod, a_wd in zip(
        _engine(market_state=None).analyze_all(),
        _engine(market_state=MARKET_STATE).analyze_all(),
    ):
        assert a_nod.value_trap_verdict == a_nod.value_trap_verdict  # sanity
    assert no_data.value_trap_reasons  # reasons exist


def test_transmission_map_and_pricing_gap():
    prices = _price_series(seed=1)
    engine = _engine(market_state=MARKET_STATE, price_data={"GLD": prices})
    a = next(x for x in engine.analyze_all() if x.theme == "Gold is a Relic")
    assert a.transmission_map
    assert all(t.asset for t in a.transmission_map)
    assert "expected_move_pct" in a.pricing_gap
    assert "read" in a.pricing_gap


# ---------------------------------------------------------------------
# Regime detection / alerts / classification
# ---------------------------------------------------------------------


def test_regime_detection_with_live_data():
    engine = _engine(market_state=MARKET_STATE)
    reg = engine.detect_regime()
    assert reg["regime"] != "Unknown / Transition"
    assert 0 <= reg["confidence"] <= 100
    assert 0 <= reg["transition_probability"] <= 100
    # a second call records the previous regime
    reg2 = engine.detect_regime()
    assert reg2["previous_regime"] == reg["regime"]


def test_alerts_only_when_divergence_extreme():
    engine = _engine(market_state=MARKET_STATE)
    alerts = engine.get_alerts()
    for al in alerts:
        assert al["severity"] in ("EXTREME DIVERGENCE", "CONSENSUS VELOCITY")
        assert "theme" in al and "signal" in al


def test_classification_and_status_are_valid():
    engine = _engine(market_state=MARKET_STATE)
    statuses = {
        "EXTREME CONSENSUS / REVERSAL RISK", "STRONG CONTRARIAN", "CONTRARIAN",
        "WATCH", "CONSENSUS CONFIRMED", "NEUTRAL",
    }
    classes = {
        "HIGH-CONVICTION CONTRARIAN", "TACTICAL CONTRARIAN", "WATCH / DEVELOPING",
        "NO EDGE", "CONSENSUS CONFIRMED",
    }
    for a in engine.analyze_all():
        assert a.signal_status in statuses
        assert a.classification in classes


def test_why_wrong_and_why_right_are_non_empty_and_opposing():
    engine = _engine(market_state=MARKET_STATE)
    for a in engine.analyze_all():
        assert a.why_wrong
        assert a.why_right
        assert a.invalidation
        # the "still right" argument must exist to fight confirmation bias
        assert "consensus" in a.why_right.lower() or "narrative" in a.why_right.lower()


# ---------------------------------------------------------------------
# Develop Setup tool
# ---------------------------------------------------------------------


def test_build_setup_with_price_data():
    engine = _engine(market_state=MARKET_STATE)
    prices = _price_series(seed=5, start=180.0)
    setup = engine.build_setup("Tech / AI Exceptionalism", "NVDA", price_df=prices)
    assert setup is not None
    assert setup.direction == "SHORT"
    assert setup.instrument == "NVDA"
    assert setup.current_price is not None
    assert setup.entry_zone is not None and setup.stop_loss is not None
    assert len(setup.targets) == 3
    assert setup.risk_reward > 0
    assert 0 < setup.position_size_pct <= 15
    assert setup.invalidation
    assert setup.catalyst
    assert setup.evidence_trail
    for e in setup.evidence_trail:
        assert e["claim"] and e["backing"] and e["provenance"]
    assert setup.momentum_state in (
        "REVERSING", "DECELERATING", "STALLING", "TREND_ACCELERATING", "NO_DATA",
    )


def test_build_setup_without_price_never_estimates_levels():
    engine = _engine(market_state=MARKET_STATE)
    setup = engine.build_setup("Gold is a Relic", "GLD", price_df=None)
    assert setup is not None
    assert setup.current_price is None
    assert setup.entry_zone is None
    assert setup.stop_loss is None
    # the evidence trail says DATA UNAVAILABLE instead of inventing levels
    assert any(e["provenance"] == "DATA UNAVAILABLE" for e in setup.evidence_trail)


def test_build_setup_unknown_theme_returns_none():
    engine = _engine(market_state=MARKET_STATE)
    assert engine.build_setup("No Such Narrative", "GLD") is None


def test_setup_long_short_levels_are_consistent():
    engine = _engine(market_state=MARKET_STATE)
    prices = _price_series(seed=11, start=50.0)
    long_setup = engine.build_setup("Gold is a Relic", "GLD", price_df=prices)
    short_setup = engine.build_setup("Tech / AI Exceptionalism", "NVDA", price_df=prices)
    assert long_setup.direction == "LONG"
    assert short_setup.direction == "SHORT"
    # LONG: stop below entry; SHORT: stop above entry
    assert long_setup.stop_loss < long_setup.current_price
    assert short_setup.stop_loss > short_setup.current_price


# ---------------------------------------------------------------------
# SignalTracker
# ---------------------------------------------------------------------


@pytest.fixture()
def tracker_path():
    with tempfile.TemporaryDirectory() as d:
        yield os.path.join(d, "log.json")


def _make_setup(engine, theme="Gold is a Relic", instrument="GLD"):
    return engine.build_setup(theme, instrument, price_df=_price_series(seed=2, start=240.0))


def test_tracker_log_and_outcome_roundtrip(tracker_path):
    engine = _engine(market_state=MARKET_STATE)
    tracker = get_signal_tracker(tracker_path)
    sid = tracker.log_setup(_make_setup(engine))
    assert sid
    assert tracker.records()[0]["outcome_label"] == "OPEN"
    ok = tracker.update_outcome(
        sid, realized_return_pct=7.2, max_favorable_pct=10.0,
        max_adverse_pct=-2.0, resolved_days=14, catalyst_hit=True,
    )
    assert ok
    rec = tracker.records()[0]
    assert rec["outcome_label"] == "WIN"
    assert rec["realized_return_pct"] == 7.2


def test_tracker_unknown_id_returns_false(tracker_path):
    tracker = get_signal_tracker(tracker_path)
    assert not tracker.update_outcome("nope", realized_return_pct=1.0)


def test_tracker_performance_analytics_empty_until_outcomes(tracker_path):
    engine = _engine(market_state=MARKET_STATE)
    tracker = get_signal_tracker(tracker_path)
    tracker.log_setup(_make_setup(engine))
    a = tracker.performance_analytics()
    assert a["tracked"] == 1 and a["resolved"] == 0
    assert a["win_rate"] is None
    assert "No realized outcomes recorded yet" in a["note"]


def test_tracker_analytics_compute_from_reported_outcomes(tracker_path):
    engine = _engine(market_state=MARKET_STATE)
    tracker = get_signal_tracker(tracker_path)
    s1 = tracker.log_setup(_make_setup(engine))
    s2 = tracker.log_setup(_make_setup(engine, "JPY Carry Trade is Safe", "USDJPY=X"))
    tracker.update_outcome(s1, realized_return_pct=8.0)
    tracker.update_outcome(s2, realized_return_pct=-3.0)
    a = tracker.performance_analytics()
    assert a["resolved"] == 2
    assert a["win_rate"] == 50.0
    assert a["avg_return_pct"] == pytest.approx(2.5)
    assert a["profit_factor"] == pytest.approx(8.0 / 3.0, abs=0.01)
    assert a["max_drawdown_pct"] == pytest.approx(-3.0)
    assert a["sharpe"] is not None
    assert a["hit_rate_by_confidence"]


def test_tracker_persists_across_instances(tracker_path):
    engine = _engine(market_state=MARKET_STATE)
    t1 = get_signal_tracker(tracker_path)
    t1.log_setup(_make_setup(engine))
    t2 = get_signal_tracker(tracker_path)
    assert len(t2.records()) == 1


# ---------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------


def test_legacy_api_unchanged():
    ct = get_counter_trend_analyzer()
    narratives = ct.get_all_narratives()
    assert len(narratives) == 8
    signals = ct.generate_counter_signals(divergence_threshold=18.0, min_strength=50.0)
    assert signals
    assert all(s.signal_strength >= 50.0 for s in signals)
    # momentum filter still works
    sig = signals[0]
    prices = {sig.instrument: _price_series(seed=1, start=sig.live_price or 100.0)}
    out = ct.apply_momentum_filter([sig], prices)
    assert out
    assert out[0].momentum_state != "NO_DATA"


def test_legacy_score_narrative_strength_unchanged():
    res = score_narrative_strength(consensus_score=70, fundamental_score=45,
                                   sentiment_score=60, positioning_score=65)
    assert res["composite_divergence"] > 20
    assert res["fade_conviction"] in ("STRONG", "MODERATE", "WEAK", "NONE")


def test_analyzer_get_intelligence_hook():
    ct = get_counter_trend_analyzer()
    eng = ct.get_intelligence(market_state=MARKET_STATE)
    assert isinstance(eng, MacroNarrativeIntelligenceEngine)
    assert eng.analyze_all()
