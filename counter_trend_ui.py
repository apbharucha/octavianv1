"""
Counter-Trend Macro Signal Intelligence dashboard (Streamlit UI)
================================================================
Professional institutional-research terminal for macro narrative
divergence and contrarian signal intelligence.

Renders the full output of ``MacroNarrativeIntelligenceEngine``:

  WHAT THE MARKET BELIEVES  -> consensus score + component breakdown
  WHAT THE DATA SAYS        -> fundamental score + component breakdown
  WHERE THEY DIVERGE        -> composite Macro Divergence Score
  HOW EXTREME               -> crowding + historical extremity
  WHAT IS PRICED            -> second-order three-layer model
  WHAT COULD FORCE CONVERGENCE -> catalyst watchlist
  WHAT ASSET IS EXPOSED     -> transmission map
  WHAT WOULD INVALIDATE     -> value-trap verdict + invalidation conditions

Every detected trend carries a "Develop Setup" button that builds a full
trade setup around that trend (entry zone, stop, targets, sizing,
catalyst, monitoring plan and an evidence trail that backs each claim
with its provenance).

No emojis are used anywhere in this UI - plain institutional text and
color badges only.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import streamlit as st

from counter_trend_analyzer import (
    PROVENANCE_LABELS,
    MacroNarrativeAnalysis,
    get_counter_trend_analyzer,
    get_signal_tracker,
)

logger = logging.getLogger("CounterTrendUI")

_CLASS_COLORS = {
    "HIGH-CONVICTION CONTRARIAN": "#00e08a",
    "TACTICAL CONTRARIAN": "#ffd166",
    "WATCH / DEVELOPING": "#ff9f43",
    "NO EDGE": "#8d99ae",
    "CONSENSUS CONFIRMED": "#ef476f",
}

_STATUS_COLORS = {
    "EXTREME CONSENSUS / REVERSAL RISK": "#ef476f",
    "STRONG CONTRARIAN": "#00e08a",
    "CONTRARIAN": "#ffd166",
    "WATCH": "#ff9f43",
    "CONSENSUS CONFIRMED": "#8d99ae",
    "NEUTRAL": "#8d99ae",
}


def _badge(text: str, color: str) -> str:
    return (
        f"<span style='background:{color}22;color:{color};border:1px solid {color}66;"
        f"border-radius:4px;padding:1px 8px;font-size:0.75rem;font-weight:600;"
        f"letter-spacing:0.4px;'>{text}</span>"
    )


def _render_regime_banner(regime: Dict, macro_state: Dict[str, float]) -> None:
    st.markdown("### Current Macro Regime")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Regime", regime.get("regime", "Unknown / Transition"))
    c2.metric("Regime Confidence", f"{regime.get('confidence', 0.0):.0f}/100")
    c3.metric("Previous Regime", regime.get("previous_regime") or "n/a")
    c4.metric("Transition Probability", f"{regime.get('transition_probability', 0.0):.0f}%")
    st.caption(
        regime.get("basis", "")
        + " Regime is a MODEL-DERIVED classification from live observations; "
        "it does not change the per-narrative analysis below."
    )

    if macro_state:
        strip = {
            "VIX": macro_state.get("vix"),
            "10Y Yield": macro_state.get("us_10y_yield"),
            "DXY": macro_state.get("usd_index"),
            "Gold": macro_state.get("gold_price"),
            "Oil": macro_state.get("oil_price"),
            "S&P 500": macro_state.get("sp500"),
        }
        cols = st.columns(len(strip))
        for col, (label, value) in zip(cols, strip.items()):
            if value is not None:
                fmt = f"{value:,.2f}" if label in ("DXY", "Gold", "Oil", "S&P 500") else f"{value:.2f}"
                col.metric(label, fmt)
        st.caption("Market state strip: OBSERVED live closes (cached 5 min).")


def _render_alerts(alerts: List[Dict]) -> None:
    if not alerts:
        return
    st.markdown("### Signal Alerts")
    for a in alerts:
        sev = a["severity"]
        color = "#ef476f" if sev == "EXTREME DIVERGENCE" else "#ffd166"
        st.markdown(
            f"<div style='background:#1a1f2e;border-left:4px solid {color};"
            f"border-radius:6px;padding:12px 14px;margin:6px 0;'>"
            f"<div style='color:{color};font-weight:700;font-size:0.85rem;'>"
            f"{sev} - {a['theme']}</div>"
            f"<div style='color:#ccc;font-size:0.85rem;margin-top:4px;'>{a['trigger']}</div>"
            f"<div style='color:#888;font-size:0.78rem;margin-top:4px;'>"
            f"Consensus {a['consensus']:.0f} | Fundamentals {a['fundamentals']:.0f} | "
            f"Divergence {a['divergence']:.0f} | Positioning {a['positioning']} | "
            f"Catalyst proximity {a['catalyst_proximity']:.0f}/100 | Signal {a['signal']}"
            f"</div></div>",
            unsafe_allow_html=True,
        )


def _render_narrative_table(analyses: List[MacroNarrativeAnalysis]) -> None:
    st.markdown("### Macro Narrative Divergence Tracker")
    st.caption(
        "Consensus Score = how strongly the market believes the narrative (0-100, "
        "aggregated from market-implied / analyst / news / positioning / flows / media "
        "components). Fundamental Score = how well the data supports it (0-100, from "
        "inflation / employment / growth / credit / rates / liquidity components). "
        "MDS = composite Macro Divergence Score. Divergence alone is never a trade."
    )
    rows = [
        {
            "Theme": a.theme,
            "Region": a.region,
            "Consensus": round(a.consensus_score, 1),
            "Fundamentals": round(a.fundamental_score, 1),
            "Divergence": round(a.divergence, 1),
            "MDS": round(a.macro_divergence_score, 1),
            "Crowding": round(a.crowding_score, 1),
            "Velocity": a.velocity_state,
            "Status": a.signal_status,
            "Classification": a.classification,
            "Opportunity": round(a.opportunity_score, 1),
            "Confidence": round(a.confidence_score, 1),
        }
        for a in analyses
    ]
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _render_narrative_expander(
    analysis: MacroNarrativeAnalysis,
    index: int,
    engine: Any,
    use_momentum: bool,
    fetch_price_data: Callable,
) -> None:
    theme = analysis.theme
    status_color = _STATUS_COLORS.get(analysis.signal_status, "#8d99ae")
    class_color = _CLASS_COLORS.get(analysis.classification, "#8d99ae")
    badges = _badge(analysis.signal_status, status_color) + " " + _badge(
        analysis.classification, class_color
    )
    with st.expander(
        f"{analysis.primary_direction} {analysis.primary_instrument or 'n/a'}  |  "
        f"{theme}  |  MDS {analysis.macro_divergence_score:.0f}/100  |  "
        f"Divergence {analysis.divergence:+.0f}",
        expanded=False,
    ):
        st.markdown(badges, unsafe_allow_html=True)
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Consensus", f"{analysis.consensus_score:.0f}/100")
        c2.metric("Fundamentals", f"{analysis.fundamental_score:.0f}/100")
        c3.metric("Divergence", f"{analysis.divergence:+.0f} pts")
        c4.metric("Opportunity", f"{analysis.opportunity_score:.0f}/100")
        c5.metric("Confidence", f"{analysis.confidence_score:.0f}/100")
        c6.metric("Crowding", f"{analysis.crowding_score:.0f}/100")

        # Develop Setup tool - one button per detected trend
        developed = st.session_state.setdefault("ct_developed_setups", {})
        if not developed.get(theme):
            if st.button("Develop Setup", key=f"ct_dev_{index}"):
                developed[theme] = True
                st.rerun()
        else:
            _render_developed_setup(theme, engine, fetch_price_data)

        _render_deep_dive(analysis)


def _render_deep_dive(analysis: MacroNarrativeAnalysis) -> None:
    st.markdown("---")
    st.markdown("**Three-Layer Model (Narrative -> Fundamentals -> Market)**")
    so = analysis.second_order
    st.markdown(f"1. **Narrative (belief):** {so.get('layer1_narrative', '')}")
    st.markdown(f"2. **Fundamentals (reality):** {so.get('layer2_fundamentals', '')}")
    st.markdown(f"3. **Market (price):** {so.get('layer3_market', '')}")
    st.caption(f"Alignment: {so.get('alignment', 'UNRESOLVED')}")

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("**Why Consensus Is High (Consensus Components)**")
        cc = analysis.consensus_components
        st.dataframe(
            pd.DataFrame(
                [
                    {"Component": "Market-implied probability", "Score": cc.market_implied},
                    {"Component": "Analyst estimates", "Score": cc.analyst_estimates},
                    {"Component": "News / media sentiment", "Score": cc.news_sentiment},
                    {"Component": "Positioning", "Score": cc.positioning},
                    {"Component": "ETF / fund flows", "Score": cc.flows},
                    {"Component": "Financial media coverage", "Score": cc.media_coverage},
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(f"Aggregate provenance: {cc.provenance}.")
    with c_right:
        st.markdown("**What the Data Says (Fundamental Components)**")
        fc = analysis.fundamental_components
        st.dataframe(
            pd.DataFrame(
                [
                    {"Component": "Inflation", "Score": fc.inflation},
                    {"Component": "Employment", "Score": fc.employment},
                    {"Component": "Growth", "Score": fc.growth},
                    {"Component": "Credit conditions", "Score": fc.credit},
                    {"Component": "Rates", "Score": fc.rates},
                    {"Component": "Liquidity", "Score": fc.liquidity},
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(f"Aggregate provenance: {fc.provenance}.")

    st.markdown("**Narrative Velocity (Consensus Today vs 7D / 30D / 90D)**")
    vel = analysis.velocity
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Series": "Consensus Score",
                    "Today": vel.get("consensus_today"),
                    "7D": vel.get("consensus_7d"),
                    "30D": vel.get("consensus_30d"),
                    "90D": vel.get("consensus_90d"),
                },
                {
                    "Series": "Fundamental Score",
                    "Today": vel.get("fundamental_today"),
                    "7D": vel.get("fundamental_7d"),
                    "30D": vel.get("fundamental_30d"),
                    "90D": vel.get("fundamental_90d"),
                },
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        f"Consensus velocity: {analysis.velocity_state}. "
        f"Fundamental momentum: {analysis.data_momentum_state}. "
        "Distinguishes 'consensus is wrong' from 'consensus is wrong and becoming "
        "increasingly wrong'."
    )

    if analysis.contradictions:
        st.markdown("**Macro Contradictions (ranked)**")
        for c in analysis.contradictions:
            st.markdown(
                f"- {c['text']}  _[magnitude {c['magnitude']:.0f}/100, "
                f"persistence {c['persistence']}, {c['independent_sources']} independent sources]_"
            )

    if analysis.historical_analogs:
        st.markdown("**Historical Regime Matching**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Period": a.period,
                        "Similarity": round(a.similarity, 1),
                        "Avg Subseq. Return": f"{a.avg_return_pct:+.0f}%",
                        "Median": f"{a.median_return_pct:+.0f}%",
                        "Worst": f"{a.worst_pct:+.0f}%",
                        "Best": f"{a.best_pct:+.0f}%",
                        "Resolution (mo)": a.resolution_months,
                        "Note": a.note,
                    }
                    for a in analysis.historical_analogs
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption("Historical analogs describe what happened before - they NEVER guarantee future results.")

    if analysis.catalysts:
        st.markdown("**Catalyst Watchlist**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Catalyst": c.name,
                        "Timing": c.timing,
                        "Days Out": c.days_out if c.days_out is not None else "event-driven",
                        "Impact": round(c.expected_impact, 0),
                        "Probability": f"{c.probability:.0f}%",
                        "Relevance": c.directional_relevance,
                        "Could Invalidate": c.invalidates_narrative,
                    }
                    for c in analysis.catalysts
                ]
            ),
            width="stretch",
            hide_index=True,
        )

    if analysis.transmission_map:
        st.markdown("**Asset Transmission Map**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Macro Variable": t.macro_variable,
                        "Transmission Mechanism": t.mechanism,
                        "Asset": t.asset,
                        "Expected Direction": t.expected_direction,
                        "Already Moved": (
                            "YES" if t.already_moved is True else "NO" if t.already_moved is False else "n/a"
                        ),
                        "20D Move": f"{t.move_pct:+.1f}%",
                    }
                    for t in analysis.transmission_map
                ]
            ),
            width="stretch",
            hide_index=True,
        )

    pg = analysis.pricing_gap
    st.markdown("**How Much Is Already Priced?**")
    st.markdown(pg.get("read", ""))
    if pg.get("expected_move_pct") is not None:
        st.caption(f"Remaining convergence opportunity estimate: {pg['expected_move_pct']}% (MODEL-DERIVED).")

    vt = analysis.value_trap_verdict
    vt_color = (
        "#00e08a" if vt == "VALID CONTRARIAN OPPORTUNITY"
        else "#ef476f" if vt == "POSSIBLE VALUE TRAP"
        else "#ffd166"
    )
    st.markdown(_badge(f"VALUE-TRAP VERDICT: {vt}", vt_color), unsafe_allow_html=True)
    for r in analysis.value_trap_reasons:
        st.markdown(f"- {r}")

    st.markdown("**Why the Market May Be Wrong**")
    st.info(analysis.why_wrong)
    st.markdown("**Why the Market May Still Be Right (strongest opposing argument)**")
    st.warning(analysis.why_right)
    st.markdown("**Invalidation Conditions**")
    st.markdown(analysis.invalidation)

    dq = analysis.data_quality
    st.markdown("**Data Quality**")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Data Confidence", f"{dq.get('data_confidence_score', 0):.0f}/100")
    c2.metric("Freshness", dq.get("freshness", "SEEDED"))
    c3.metric("Independent Sources", dq.get("independent_sources", 0))
    c4.metric("Missing Data", f"{dq.get('missing_data_pct', 0):.0f}%")
    c5.metric("Reliability", dq.get("source_reliability", "MEDIUM"))
    st.caption(
        "Provenance labels: "
        + " | ".join(str(p) for p in (dq.get("provenance") or PROVENANCE_LABELS))
        + ". Never mix OBSERVED, ESTIMATED, MODEL-DERIVED, MARKET-IMPLIED or AI-INFERRED "
        "numbers without labeling them."
    )


def _render_developed_setup(
    theme: str, engine: Any, fetch_price_data: Callable
) -> None:
    """Render the Develop Setup tool output for one trend (built around it)."""
    st.markdown("---")
    st.markdown("### Develop Setup - Trade Construction")
    analysis = next(
        (a for a in engine.analyze_all() if a.theme == theme), None
    )
    if analysis is None:
        st.error(f"Setup unavailable for {theme}.")
        return
    instrument = analysis.primary_instrument or "GLD"
    price_data = fetch_price_data((instrument,))
    price_df = price_data.get(instrument)
    setup = engine.build_setup(theme, instrument, price_df=price_df)
    if setup is None:
        st.error(f"Setup could not be built for {theme}.")
        return

    class_color = _CLASS_COLORS.get(setup.classification, "#8d99ae")
    st.markdown(
        f"**{setup.direction} {setup.instrument}** - fade of the '{setup.narrative}' narrative "
        + _badge(setup.classification, class_color),
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reference Price", f"{setup.current_price:.4f}" if setup.current_price else "n/a")
    if setup.entry_zone:
        c2.metric("Entry Zone", f"{setup.entry_zone[0]:.4f} - {setup.entry_zone[1]:.4f}")
    else:
        c2.metric("Entry Zone", "PRICE DATA UNAVAILABLE")
    c3.metric("Stop Loss", f"{setup.stop_loss:.4f}" if setup.stop_loss else "n/a")
    c4.metric("Risk / Reward", f"{setup.risk_reward:.2f}R")

    t1, t2, t3 = st.columns(3)
    for col, tgt in zip((t1, t2, t3), setup.targets):
        col.metric("Target", f"{tgt:.4f}" if tgt else "n/a")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Position Size", f"{setup.position_size_pct:.1f}%")
    s2.metric("Risk / Trade", f"{setup.risk_per_trade_pct:.1f}%")
    s3.metric("Confidence", f"{setup.confidence_score:.0f}/100")
    s4.metric("Opportunity", f"{setup.opportunity_score:.0f}/100")

    st.markdown(
        f"**Catalyst:** {setup.catalyst}"
        + (f" (~{setup.catalyst_days_out} days out)" if setup.catalyst_days_out else "")
        + f"  |  **Horizon:** {setup.time_horizon}  |  **Half-life:** {setup.half_life}"
    )
    st.markdown(f"**Momentum state:** {setup.momentum_state}")
    st.markdown(f"**Entry condition:** {setup.entry_condition}")
    st.markdown(f"**Invalidation:** {setup.invalidation}")

    st.markdown("**Monitoring Plan**")
    for m in setup.monitoring_plan:
        st.markdown(f"- {m}")

    st.markdown("**Evidence Trail (every claim backed up)**")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Claim": e["claim"],
                    "Backing": e["backing"],
                    "Provenance": e["provenance"],
                }
                for e in setup.evidence_trail
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Setup levels are computed from the instrument's real price series when "
        "available (OBSERVED); otherwise they are left as DATA UNAVAILABLE - never "
        "estimated. Sizing uses a fixed per-trade risk input; review before trading."
    )

    b1, b2 = st.columns(2)
    if b1.button("Log Setup to Signal Tracker", key=f"ct_log_{theme.replace(' ', '_')}"):
        tracker = get_signal_tracker()
        sid = tracker.log_setup(setup)
        st.success(f"Setup logged - signal ID {sid}. Track its outcome in the Signal Tracker section.")
    if b2.button("Remove Setup", key=f"ct_rm_{theme.replace(' ', '_')}"):
        st.session_state["ct_developed_setups"].pop(theme, None)
        st.rerun()


def _render_signal_tracker() -> None:
    st.markdown("---")
    st.markdown("### Signal Tracker & Performance Analytics")
    tracker = get_signal_tracker()
    analytics = tracker.performance_analytics()
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Tracked", analytics.get("tracked", 0))
    c2.metric("Resolved", analytics.get("resolved", 0))
    c3.metric("Win Rate", f"{analytics.get('win_rate', 0):.0f}%" if analytics.get("win_rate") is not None else "n/a")
    c4.metric("Avg Return", f"{analytics.get('avg_return_pct', 0):+.2f}%" if analytics.get("avg_return_pct") is not None else "n/a")
    c5.metric("Profit Factor", f"{analytics.get('profit_factor', 0):.2f}" if analytics.get("profit_factor") is not None else "n/a")
    c6.metric("Max Drawdown", f"{analytics.get('max_drawdown_pct', 0):.1f}%" if analytics.get("max_drawdown_pct") is not None else "n/a")
    st.caption(analytics.get("note", ""))

    records = tracker.records()
    if records:
        with st.expander(f"Signal Log ({len(records)} records)", expanded=False):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "ID": r.get("signal_id"),
                            "Date": r.get("created_at", "")[:10],
                            "Theme": r.get("theme"),
                            "Instrument": r.get("instrument"),
                            "Direction": r.get("direction"),
                            "Confidence": r.get("confidence_score"),
                            "Classification": r.get("classification"),
                            "Entry Ref": r.get("entry_reference"),
                            "Realized %": r.get("realized_return_pct"),
                            "Outcome": r.get("outcome_label", "OPEN"),
                        }
                        for r in records
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "Outcomes are recorded only when reported by the user - performance "
                "metrics are never fabricated. Win rate alone is not the objective; "
                "Sharpe, Sortino, profit factor, drawdown and calibration by confidence "
                "bucket all matter."
            )
            open_records = [r for r in records if r.get("realized_return_pct") is None]
            if open_records:
                st.markdown("**Update Realized Outcome**")
                for r in open_records[:5]:
                    with st.expander(f"{r.get('signal_id')} - {r.get('instrument')}", expanded=False):
                        val = st.number_input(
                            "Realized return %",
                            value=0.0,
                            step=0.5,
                            key=f"ct_out_{r.get('signal_id')}",
                        )
                        mfe = st.number_input(
                            "Max favorable excursion %", value=0.0, step=0.5,
                            key=f"ct_mfe_{r.get('signal_id')}",
                        )
                        mae = st.number_input(
                            "Max adverse excursion %", value=0.0, step=0.5,
                            key=f"ct_mae_{r.get('signal_id')}",
                        )
                        days = st.number_input(
                            "Days to resolution", value=0, step=1,
                            key=f"ct_days_{r.get('signal_id')}",
                        )
                        catalyst = st.checkbox("Catalyst occurred", key=f"ct_cat_{r.get('signal_id')}")
                        if st.button("Save Outcome", key=f"ct_save_{r.get('signal_id')}"):
                            ok = tracker.update_outcome(
                                r.get("signal_id"),
                                realized_return_pct=float(val),
                                max_favorable_pct=float(mfe),
                                max_adverse_pct=float(mae),
                                resolved_days=int(days) if days else None,
                                catalyst_hit=catalyst,
                            )
                            if ok:
                                st.success("Outcome saved.")
                                st.rerun()
                            else:
                                st.error("Unknown signal id.")


def show_counter_trend_dashboard(
    fetch_price_data: Callable,
    fetch_macro_state: Callable,
) -> None:
    """Render the full Counter-Trend Macro Signal Intelligence dashboard."""
    st.subheader("Macro Narrative Divergence & Contrarian Signal Intelligence")
    st.caption(
        "Institutional terminal that answers, in order: WHAT THE MARKET BELIEVES, "
        "WHAT THE DATA SAYS, WHERE THEY DIVERGE, HOW EXTREME THE DIVERGENCE IS, "
        "WHAT IS ALREADY PRICED, WHAT COULD FORCE CONVERGENCE, WHICH ASSET IS MOST "
        "EXPOSED, AND WHAT WOULD INVALIDATE THE THESIS. Divergence alone is never "
        "an automatic trade - opportunity and confidence are scored separately and "
        "a value-trap detector gates every signal."
    )
    st.caption(
        "Provenance legend - every number is labeled: "
        + " | ".join(str(p) for p in PROVENANCE_LABELS)
        + ". DATA UNAVAILABLE is never replaced by an estimate."
    )

    try:
        ct = get_counter_trend_analyzer()
        macro_state = fetch_macro_state()
        engine = ct.get_intelligence(market_state=macro_state)
        regime = engine.detect_regime()
        _render_regime_banner(regime, macro_state)
        analyses = engine.analyze_all()

        _render_alerts(engine.get_alerts())

        regions = sorted({a.region for a in analyses})
        region = st.selectbox("Region / Country", ["All"] + regions)
        filtered = [a for a in analyses if region == "All" or a.region == region]

        _render_narrative_table(filtered)

        st.markdown("---")
        st.markdown("### Narrative Deep-Dive & Setup Development")
        st.caption(
            "Each detected trend has a 'Develop Setup' button that builds a trade "
            "setup around that trend - entry zone, stop, targets, sizing, catalyst, "
            "monitoring plan and a backed-up evidence trail."
        )
        for i, a in enumerate(filtered):
            _render_narrative_expander(a, i, engine, use_momentum=True, fetch_price_data=fetch_price_data)

        _render_signal_tracker()

        with st.expander("Full Narrative Report (Text)", expanded=False):
            st.code(ct.get_narrative_report(), language=None)

    except Exception as e:
        logger.exception("Counter-trend dashboard failed")
        st.error(f"Counter-trend analyzer unavailable: {e}")
        st.info("Ensure counter_trend_analyzer.py is present in the project root.")
