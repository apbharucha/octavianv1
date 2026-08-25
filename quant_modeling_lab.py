"""
Octavian Quantitative Modeling Laboratory
==========================================
Advanced quantitative modeling environment used by quant funds.

Covers:
  - Market Regime Detection (HMM, Bayesian switching)
  - Factor Crowding Detection
  - Machine Learning Framework (RF, GBM, LSTM, RL)
  - Liquidity & Market Impact Modeling (Almgren-Chriss)
  - Risk Management (VaR, CVaR, factor exposure)
  - Narrative Dislocation Detection
  - Cross-Asset Analysis Engine

No emojis — CSS microanimations throughout.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import streamlit as st

#  Theme 
try:
    from octavian_theme import COLORS
except ImportError:
    COLORS = {
        "navy": "#0a1628", "navy_light": "#132240", "gold": "#c9a84c",
        "lavender": "#9b8ec4", "white_soft": "#e0e4ec", "text_primary": "#e8eaf0",
        "text_secondary": "#a0a8b8", "border": "#1e3050",
        "success": "#4caf50", "danger": "#ef5350", "neutral": "#78909c",
    }

#  Lazy imports 
try:
    from factor_crowding_engine import get_crowding_engine
    HAS_CROWDING = True
except ImportError:
    HAS_CROWDING = False

try:
    from narrative_dislocation_engine import NarrativeDislocator
    HAS_NARRATIVE = True
except ImportError:
    HAS_NARRATIVE = False

#  CSS 
_QML_CSS = """
<style>
@keyframes qml-fade-in {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes qml-pulse-ring {
    0%   { box-shadow: 0 0 0 0 rgba(155,142,196,0.4); }
    70%  { box-shadow: 0 0 0 10px rgba(155,142,196,0); }
    100% { box-shadow: 0 0 0 0 rgba(155,142,196,0); }
}
@keyframes qml-bar-grow {
    from { transform: scaleX(0); }
    to   { transform: scaleX(1); }
}
@keyframes qml-regime-glow {
    0%, 100% { opacity: 0.8; }
    50%       { opacity: 1.0; }
}
.qml-card {
    background: linear-gradient(135deg, #132240 0%, #1a2d4a 100%);
    border: 1px solid rgba(155,142,196,0.12);
    border-radius: 10px;
    padding: 16px 20px;
    margin: 8px 0;
    animation: qml-fade-in 0.35s ease-out;
    transition: border-color 0.2s ease;
}
.qml-card:hover { border-color: rgba(155,142,196,0.3); }
.qml-section {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #9b8ec4;
    border-bottom: 1px solid rgba(155,142,196,0.18);
    padding-bottom: 5px;
    margin: 16px 0 10px 0;
    animation: qml-fade-in 0.3s ease-out;
}
.qml-metric {
    background: rgba(10,22,40,0.65);
    border-radius: 7px;
    padding: 10px 14px;
    text-align: center;
}
.qml-metric-label {
    font-size: 0.7rem;
    color: #a0a8b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.qml-metric-value {
    font-size: 1.4rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: #e8eaf0;
}
.qml-regime-badge {
    display: inline-block;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    animation: qml-regime-glow 2s ease-in-out infinite;
}
.qml-prob-bar {
    height: 8px;
    background: rgba(255,255,255,0.07);
    border-radius: 4px;
    overflow: hidden;
    margin: 3px 0;
}
.qml-prob-fill {
    height: 100%;
    border-radius: 4px;
    transform-origin: left;
    animation: qml-bar-grow 0.6s ease-out forwards;
}
</style>
"""


def _section(title: str):
    st.markdown(f'<div class="qml-section">{title}</div>', unsafe_allow_html=True)


def _metric(label: str, value: str, color: str = "#e8eaf0"):
    st.markdown(
        f'<div class="qml-metric">'
        f'<div class="qml-metric-label">{label}</div>'
        f'<div class="qml-metric-value" style="color:{color};">{value}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _dark_layout(**kwargs) -> dict:
    base = dict(
        template="plotly_dark",
        paper_bgcolor=COLORS["navy"],
        plot_bgcolor=COLORS["navy_light"],
        font=dict(color=COLORS["text_primary"], family="Inter, sans-serif"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    base.update(kwargs)
    return base


# 
# TAB 1 — Market Regime Detection
# 

def _render_liquidity_tab():
    _section("Liquidity & Market Impact Modeling")
    st.caption(
        "Simulate realistic trading conditions using the Almgren-Chriss optimal execution model. "
        "Compute expected slippage, fill probability, and optimal order size."
    )

    with st.expander("Order Parameters", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            symbol = st.text_input("Symbol", value="AAPL", key="liq_sym").strip().upper()
            order_size = st.number_input("Order Size (shares)", value=10000, min_value=1, key="liq_size")
            adv_pct = st.slider("% of ADV", 1, 50, 10, key="liq_adv_pct") / 100
        with c2:
            price = st.number_input("Current Price ($)", value=175.0, min_value=0.01, key="liq_price")
            adv = st.number_input("Avg Daily Volume (shares)", value=50_000_000, step=1_000_000, key="liq_adv")
            bid_ask_spread = st.number_input("Bid-Ask Spread ($)", value=0.02, min_value=0.001, step=0.001, key="liq_spread")
        with c3:
            volatility = st.slider("Daily Volatility (%)", 0.5, 10.0, 1.5, step=0.1, key="liq_vol") / 100
            execution_horizon = st.slider("Execution Horizon (days)", 1, 20, 5, key="liq_horizon")
            risk_aversion = st.slider("Risk Aversion (lambda)", 0.01, 1.0, 0.1, step=0.01, key="liq_lambda")

    if st.button("Compute Market Impact", type="primary", key="liq_run"):
        with st.spinner("Computing Almgren-Chriss optimal execution..."):
            try:
                # Almgren-Chriss model parameters
                X = order_size  # total shares to trade
                T = execution_horizon  # trading horizon in days
                sigma = volatility  # daily vol
                eta = bid_ask_spread / (2 * price)  # temporary impact coefficient
                gamma = 0.1 / adv  # permanent impact coefficient
                lam = risk_aversion  # risk aversion

                # Optimal execution trajectory
                kappa = math.sqrt(lam * sigma**2 / eta) if eta > 0 else 1.0
                tau = T / max(T, 1)

                # Optimal trading schedule
                n_intervals = T
                t_vals = np.linspace(0, T, n_intervals + 1)
                x_vals = X * np.sinh(kappa * (T - t_vals)) / np.sinh(kappa * T) if kappa * T > 0.001 else X * (1 - t_vals / T)

                # Trading rate
                n_vals = np.diff(x_vals)  # shares traded per interval

                # Costs
                temp_impact = eta * np.sum(n_vals**2)
                perm_impact = gamma * X**2 / 2
                spread_cost = bid_ask_spread / 2 * X
                total_cost = temp_impact + perm_impact + spread_cost

                # Slippage in bps
                slippage_bps = total_cost / (X * price) * 10000

                # Fill probability (simplified)
                fill_prob = min(0.99, adv / max(X, 1) * 0.5)

                # Optimal order size (1% of ADV)
                optimal_size = int(adv * adv_pct)

                st.session_state["liq_result"] = {
                    "t_vals": t_vals,
                    "x_vals": x_vals,
                    "n_vals": n_vals,
                    "temp_impact": temp_impact,
                    "perm_impact": perm_impact,
                    "spread_cost": spread_cost,
                    "total_cost": total_cost,
                    "slippage_bps": slippage_bps,
                    "fill_prob": fill_prob,
                    "optimal_size": optimal_size,
                    "symbol": symbol,
                    "order_size": order_size,
                    "price": price,
                }
                st.success("Market impact analysis complete.")
            except Exception as e:
                st.error(f"Error: {e}")

    result = st.session_state.get("liq_result")
    if result is None:
        st.info("Configure order parameters and click 'Compute Market Impact'.")
        return

    # Summary metrics
    _section("Market Impact Summary")
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1: _metric("Total Cost ($)", f"${result['total_cost']:,.2f}", COLORS["danger"])
    with mc2: _metric("Slippage (bps)", f"{result['slippage_bps']:.1f}", COLORS["danger"])
    with mc3: _metric("Fill Probability", f"{result['fill_prob']:.1%}", COLORS["success"])
    with mc4: _metric("Optimal Size", f"{result['optimal_size']:,}", COLORS["gold"])

    # Cost breakdown
    _section("Cost Breakdown")
    costs = {
        "Temporary Impact": result["temp_impact"],
        "Permanent Impact": result["perm_impact"],
        "Spread Cost": result["spread_cost"],
    }
    total = sum(costs.values())
    for cost_name, cost_val in costs.items():
        pct = cost_val / max(total, 1e-10)
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;margin:5px 0;">'
            f'<span style="color:{COLORS["text_primary"]};">{cost_name}</span>'
            f'<span style="color:{COLORS["danger"]};">${cost_val:,.2f} ({pct:.0%})</span>'
            f'</div>'
            f'<div class="qml-prob-bar"><div class="qml-prob-fill" style="width:{pct*100:.0f}%;background:{COLORS["danger"]};"></div></div>',
            unsafe_allow_html=True,
        )

    # Execution trajectory
    _section("Optimal Execution Trajectory (Almgren-Chriss)")
    fig_traj = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              row_heights=[0.6, 0.4], vertical_spacing=0.04)

    fig_traj.add_trace(go.Scatter(
        x=result["t_vals"], y=result["x_vals"], mode="lines+markers",
        name="Remaining Inventory",
        line=dict(color=COLORS["gold"], width=2),
        marker=dict(size=6),
    ), row=1, col=1)

    fig_traj.add_trace(go.Bar(
        x=list(range(len(result["n_vals"]))),
        y=np.abs(result["n_vals"]),
        name="Shares Traded per Interval",
        marker_color=COLORS["lavender"], opacity=0.7,
    ), row=2, col=1)

    fig_traj.update_layout(**_dark_layout(
        height=450,
        title=f"Almgren-Chriss Optimal Execution: {result['symbol']} ({result['order_size']:,} shares)",
    ))
    fig_traj.update_yaxes(title_text="Remaining Shares", row=1, col=1)
    fig_traj.update_yaxes(title_text="Shares/Interval", row=2, col=1)
    st.plotly_chart(fig_traj, width='stretch')

    # Liquidity stress test
    _section("Liquidity Stress Test")
    order_sizes = [1000, 5000, 10000, 50000, 100000, 500000]
    stress_results = []
    for sz in order_sizes:
        temp = eta * sz**2 if 'eta' in dir() else result["temp_impact"] * (sz / result["order_size"])**2
        perm = gamma * sz**2 / 2 if 'gamma' in dir() else result["perm_impact"] * (sz / result["order_size"])**2
        sp = bid_ask_spread / 2 * sz if 'bid_ask_spread' in dir() else result["spread_cost"] * sz / result["order_size"]
        total = temp + perm + sp
        slip = total / (sz * result["price"]) * 10000 if sz > 0 else 0
        stress_results.append({"Order Size": f"{sz:,}", "Total Cost ($)": f"${total:,.0f}", "Slippage (bps)": f"{slip:.1f}"})

    st.dataframe(pd.DataFrame(stress_results), width='stretch', hide_index=True)


# 
# TAB 3 — Risk Management
# 

def _render_crowding_tab():
    _section("Factor Crowding Detection")
    st.caption(
        "Institutional crowding analytics: which factors are over-traded, how much "
        "capacity remains, alpha decay rates, and unwind impact. Crowded factors "
        "have reduced expected returns and elevated unwind risk."
    )

    if not HAS_CROWDING:
        st.error("Factor crowding engine is not available.")
        return

    c1, c2 = st.columns(2)
    with c1:
        universe = st.selectbox(
            "Scan Universe", ["Auto (live universe)", "Custom symbols"],
            index=0, key="crowd_universe",
        )
    with c2:
        custom = st.text_input(
            "Symbols (comma-separated)", "", key="crowd_custom",
            placeholder="e.g. AAPL, NVDA, MSFT, TSLA",
        )

    if st.button("Analyze Factor Crowding", type="primary", key="crowd_run"):
        with st.spinner("Analyzing factor crowdedness across the universe..."):
            try:
                from factor_crowding_engine import get_crowding_engine
                engine = get_crowding_engine()
                if universe == "Auto (live universe)" or not custom.strip():
                    from ticker_universe import get_ticker_universe
                    symbols = get_ticker_universe().get_full_universe_sample(40)
                else:
                    symbols = [s.strip().upper() for s in custom.split(",") if s.strip()][:40]
                dashboard = engine.build_dashboard(symbols)
                st.session_state["crowd_dashboard"] = dashboard
                st.success(
                    f"Crowding analysis complete: {len(dashboard.factor_scores)} factors, "
                    f"{len(dashboard.crowded_trades)} crowded trades."
                )
            except Exception as e:
                st.error(f"Crowding analysis failed: {e}")

    dash = st.session_state.get("crowd_dashboard")
    if dash is None:
        st.info("Configure the universe and click **Analyze Factor Crowding**.")
        return

    # Factor crowding bar chart
    _section("Factor Crowding Levels")
    factors = [s.factor_name for s in dash.factor_scores]
    crowding = [s.crowding_score for s in dash.factor_scores]
    fig_crowd = go.Figure(go.Bar(
        x=factors, y=crowding,
        marker_color=[
            "#ef5350" if c >= 75 else "#c9a84c" if c >= 50 else "#4caf50"
            for c in crowding
        ],
        text=[f"{c:.0f}" for c in crowding], textposition="outside",
    ))
    fig_crowd.update_layout(**_dark_layout(
        height=320, title="Crowding Score by Factor (0-100)",
        yaxis_title="Crowding Score",
    ))
    st.plotly_chart(fig_crowd, width='stretch')

    # Factor detail cards
    _section("Factor Detail")
    for s in dash.factor_scores:
        status_color = (
            "#ef5350" if s.status == "OVERCROWDED"
            else "#c9a84c" if s.status == "CROWDED"
            else "#4caf50" if s.status == "UNDERCROWDED"
            else COLORS["neutral"]
        )
        st.markdown(
            f'<div class="qml-card">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<span style="font-weight:700;color:{COLORS["gold"]};">{s.factor_name}</span>'
            f'<span style="color:{status_color};font-weight:700;">{s.status}</span>'
            f'</div>'
            f'<div style="font-size:0.8rem;color:{COLORS["text_secondary"]};margin:6px 0;">{s.description}</div>'
            f'<div style="display:flex;gap:20px;font-size:0.85rem;flex-wrap:wrap;">'
            f'<span>Capacity left: <b style="color:{COLORS["success"]};">{s.capacity_remaining:.0f}%</b></span>'
            f'<span>Alpha decay: <b style="color:{COLORS["danger"]};">{s.alpha_decay_rate:.1f}%/yr</b></span>'
            f'<span>Unwind impact: <b style="color:{COLORS["danger"]};">{s.estimated_unwind_impact:.0f}bps</b></span>'
            f'<span>Signal haircut: <b>{s.signal_haircut:.0%}</b></span>'
            f'</div>'
            f'<div style="font-size:0.82rem;color:{COLORS["text_primary"]};margin-top:8px;">{s.evidence[0] if s.evidence else ""}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Crowded trades
    if dash.crowded_trades:
        _section("Crowded Trades")
        rows = [{
            "Ticker": ct.ticker,
            "Crowding %ile": f"{ct.crowding_percentile:.1f}%",
            "Unwind Risk": ct.unwind_risk,
            "Crowd Signal": f"{ct.crowding_adjusted_signal:+.2f}",
            "Description": getattr(ct, "description", ""),
        } for ct in dash.crowded_trades[:12]]
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    # Top risks
    if dash.top_risks:
        _section("Top Crowding Risks")
        for risk in dash.top_risks[:6]:
            st.markdown(f"- {risk}")


# 
# TAB 6 — Narrative Dislocation Detection
# 

def _render_narrative_tab():
    _section("Narrative Dislocation Detection")
    st.caption(
        "Detects contradictions between prevailing market narratives and actual "
        "price/data signals — price/earnings gaps, broken cross-asset relationships, "
        "and extreme sentiment positioning."
    )

    if not HAS_NARRATIVE:
        st.error("Narrative dislocation engine is not available.")
        return

    c1, c2 = st.columns([2, 1])
    with c1:
        ticker = st.text_input(
            "Ticker", value="SPY", key="narr_ticker").strip().upper()
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_all = st.button("Run Dislocation Scan", type="primary", key="narr_run")

    if run_all:
        with st.spinner("Scanning narratives for dislocations..."):
            try:
                from narrative_dislocation_engine import (
                    get_narrative_engine, CROSS_ASSET_RELATIONSHIPS,
                )
                engine = get_narrative_engine()
                found = []
                # 1. Price/earnings contradiction
                pe = engine.detect_price_earnings_contradiction(ticker)
                if pe:
                    found.append(pe)
                # 2. Sentiment/positioning extreme
                se = engine.detect_sentiment_positioning_extreme(ticker)
                if se:
                    found.append(se)
                # 3. Cross-asset relationship breakdowns
                for rel in CROSS_ASSET_RELATIONSHIPS:
                    try:
                        ca = engine.detect_cross_asset_breakdown(rel)
                        if ca:
                            found.append(ca)
                    except Exception:
                        continue
                st.session_state["narr_found"] = found
                st.success(f"Scan complete — {len(found)} dislocation(s) detected.")
            except Exception as e:
                st.error(f"Dislocation scan failed: {e}")

    found = st.session_state.get("narr_found")
    if not found:
        st.info(
            "Click **Run Dislocation Scan** to detect narrative-vs-data contradictions "
            "for the selected ticker and the core cross-asset relationships."
        )
        return

    for c in found:
        sev_color = (
            "#ef5350" if c.severity >= 60
            else "#c9a84c" if c.severity >= 30
            else COLORS["neutral"]
        )
        dir_color = (
            "#4caf50" if c.direction == "BULLISH"
            else "#ef5350" if c.direction == "BEARISH"
            else COLORS["gold"]
        )
        st.markdown(
            f'<div class="qml-card" style="border-left:4px solid {sev_color};">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<span style="font-weight:700;color:white;">{c.title}</span>'
            f'<span style="color:{dir_color};font-weight:700;">{c.direction} · {c.severity:.0f}/100</span>'
            f'</div>'
            f'<div style="font-size:0.8rem;color:{COLORS["text_secondary"]};margin:6px 0;">'
            f'{", ".join(c.assets_involved)} · confidence {c.confidence:.0f}% · decay {c.decay_days}d</div>'
            f'<div style="font-size:0.82rem;color:{COLORS["text_secondary"]};">'
            f'<b>Narrative:</b> {c.narrative_consensus}</div>'
            f'<div style="font-size:0.86rem;color:{COLORS["text_primary"]};margin-top:6px;">'
            f'<b style="color:{dir_color};">Counter-thesis:</b> {c.counter_thesis}</div>'
            f'<div style="font-size:0.82rem;color:{COLORS["text_secondary"]};margin-top:6px;">'
            f'<b>Trade implication:</b> {c.trade_implication}</div>'
            f'<div style="font-size:0.82rem;color:{COLORS["text_secondary"]};">'
            f'<b>Historical analogue:</b> {c.historical_analogue}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# 
# MAIN ENTRY POINT
# 

def render_quant_modeling_lab():
    """Main entry point for the Quantitative Modeling Lab.

    Contains only the modeling tools that are NOT duplicated elsewhere:
    factor crowding, narrative dislocation, and liquidity/market-impact.
    Regime detection, risk management, ML frameworks, and cross-asset
    analysis live in the Quant Portal to avoid feature duplication.
    """
    st.markdown(_QML_CSS, unsafe_allow_html=True)
    st.title("Quantitative Modeling Lab")
    st.caption(
        "Advanced quantitative modeling environment — factor crowding, narrative "
        "dislocation, and liquidity/market-impact modeling."
    )

    tabs = st.tabs([
        "Factor Crowding",
        "Narrative Dislocation",
        "Liquidity & Impact",
    ])

    with tabs[0]:
        _render_crowding_tab()

    with tabs[1]:
        _render_narrative_tab()

    with tabs[2]:
        _render_liquidity_tab()
