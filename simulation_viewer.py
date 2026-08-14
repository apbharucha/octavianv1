"""
Octavian Simulation Hub — Visual Simulation Laboratory
=======================================================
Comprehensive simulation environment covering:
  - Market Microstructure Simulation (agent-based order book)
  - Portfolio Evolution & Drawdown Visualization
  - Scenario & Crisis Simulation (historical + hypothetical)
  - Simulation Universe Generator (synthetic financial worlds)
  - Performance Dashboard with grading
  - Hyperdimensional Visualization (3D manifolds, parameter spaces)

No emojis — sleek CSS microanimations throughout.
"""

from __future__ import annotations

try:
    from futures_engine import get_futures_engine, FUTURES_UNIVERSE, SPREAD_TEMPLATES
    HAS_FUTURES = True
except ImportError:
    HAS_FUTURES = False
    FUTURES_UNIVERSE = {}
    SPREAD_TEMPLATES = {}

try:
    import options_engine
    HAS_OPTIONS = True
except ImportError:
    HAS_OPTIONS = False

import math
import random
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st

#  Theme 
try:
    from octavian_theme import COLORS
except ImportError:
    COLORS = {
        "navy": "#0a1628", "navy_light": "#132240", "navy_mid": "#1a2d4a",
        "gold": "#c9a84c", "gold_light": "#d4b86a", "lavender": "#9b8ec4",
        "white_soft": "#e0e4ec", "text_primary": "#e8eaf0",
        "text_secondary": "#a0a8b8", "border": "#1e3050",
        "success": "#4caf50", "danger": "#ef5350", "neutral": "#78909c",
    }

#  Lazy imports 
try:
    from market_simulation_universe import MarketSimulator, AgentType, EventType
    HAS_UNIVERSE = True
except ImportError:
    HAS_UNIVERSE = False

try:
    from market_simulation_engine import MarketSimulationEngine, render_options_monte_carlo_panel
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False
    render_options_monte_carlo_panel = None

try:
    from trading_system.simulation_grader import SimulationGrader
    HAS_GRADER = True
except ImportError:
    HAS_GRADER = False

try:
    from institutional_analytics_engine import (
        BayesianNetwork, run_macro_analysis, run_micro_analysis,
        detect_regime, generate_market_scenarios,
        build_bayesian_network_from_data,
    )
    HAS_INST_ANALYTICS = True
except ImportError:
    HAS_INST_ANALYTICS = False

try:
    from institutional_visualizations import (
        create_bayesian_network_graph, create_correlation_heatmap,
        create_regime_map, create_probability_heatmap,
        create_factor_decomposition_chart, create_volatility_surface_chart,
        create_macro_dashboard_chart, create_scenario_cascade_chart,
        create_sector_momentum_chart, create_breadth_gauge,
    )
    HAS_INST_VIZ = True
except ImportError:
    HAS_INST_VIZ = False

#  CSS Microanimations 
_SIM_CSS = """
<style>
@keyframes sim-pulse {
    0%   { box-shadow: 0 0 0 0 rgba(201,168,76,0.4); }
    70%  { box-shadow: 0 0 0 8px rgba(201,168,76,0); }
    100% { box-shadow: 0 0 0 0 rgba(201,168,76,0); }
}
@keyframes sim-slide-in {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes sim-bar-fill {
    from { width: 0%; }
    to   { width: var(--bar-width); }
}
@keyframes sim-dot-blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
}
@keyframes sim-spin {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}
@keyframes sim-fade-up {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes sim-glow-pulse {
    0%, 100% { opacity: 0.6; }
    50%       { opacity: 1.0; }
}
.sim-card {
    background: linear-gradient(135deg, #132240 0%, #1a2d4a 100%);
    border: 1px solid rgba(201,168,76,0.15);
    border-radius: 10px;
    padding: 18px 22px;
    margin: 10px 0;
    animation: sim-slide-in 0.4s ease-out;
}
.sim-card:hover {
    border-color: rgba(201,168,76,0.35);
    transition: border-color 0.25s ease;
}
.sim-metric {
    background: rgba(10,22,40,0.7);
    border-radius: 8px;
    padding: 12px 16px;
    text-align: center;
    animation: sim-slide-in 0.35s ease-out;
}
.sim-metric-label {
    font-size: 0.72rem;
    color: #a0a8b8;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.sim-metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #e8eaf0;
    font-family: 'JetBrains Mono', monospace;
}
.sim-badge-bull {
    display: inline-block;
    background: rgba(76,175,80,0.15);
    border: 1px solid rgba(76,175,80,0.4);
    color: #4caf50;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    animation: sim-slide-in 0.3s ease-out;
}
.sim-badge-bear {
    display: inline-block;
    background: rgba(239,83,80,0.15);
    border: 1px solid rgba(239,83,80,0.4);
    color: #ef5350;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    animation: sim-slide-in 0.3s ease-out;
}
.sim-badge-neutral {
    display: inline-block;
    background: rgba(120,144,156,0.15);
    border: 1px solid rgba(120,144,156,0.4);
    color: #78909c;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}
.sim-live-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    background: #4caf50;
    border-radius: 50%;
    margin-right: 6px;
    animation: sim-dot-blink 1.5s ease-in-out infinite;
}
.sim-section-header {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #c9a84c;
    border-bottom: 1px solid rgba(201,168,76,0.2);
    padding-bottom: 6px;
    margin: 18px 0 12px 0;
    animation: sim-fade-up 0.4s ease-out;
}
.sim-event-tag {
    display: inline-block;
    background: rgba(155,142,196,0.15);
    border: 1px solid rgba(155,142,196,0.3);
    color: #9b8ec4;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.72rem;
    margin: 2px;
    animation: sim-slide-in 0.3s ease-out;
}
.sim-progress-bar {
    height: 6px;
    background: rgba(255,255,255,0.08);
    border-radius: 3px;
    overflow: hidden;
    margin: 4px 0;
}
.sim-progress-fill {
    height: 100%;
    border-radius: 3px;
    animation: sim-bar-fill 0.8s ease-out forwards;
    background: linear-gradient(90deg, #c9a84c, #d4b86a);
}
.sim-spinner {
    display: inline-block;
    width: 16px;
    height: 16px;
    border: 2px solid rgba(201,168,76,0.3);
    border-top-color: #c9a84c;
    border-radius: 50%;
    animation: sim-spin 0.8s linear infinite;
    margin-right: 8px;
    vertical-align: middle;
}
.sim-regime-label {
    font-size: 0.85rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    display: inline-block;
    animation: sim-slide-in 0.3s ease-out;
}
</style>
"""


#  Helper: dark plotly layout 
def _dark_layout(**kwargs) -> dict:
    base = dict(
        template="plotly_dark",
        paper_bgcolor=COLORS["navy"],
        plot_bgcolor=COLORS["navy_light"],
        font=dict(color=COLORS["text_primary"]),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    base.update(kwargs)
    return base


def _section(title: str):
    st.markdown(f"### {title}")

def _metric_card(label: str, value: str, color: str = "#e8eaf0"):
    st.markdown(f"**{label}:** <span style='color:{color}'>{value}</span>", unsafe_allow_html=True)

def _render_microstructure_tab():
    """Agent-based limit-order-book microstructure simulation."""
    st.markdown("<div class='sim-section-header'>Market Microstructure</div>", unsafe_allow_html=True)
    st.markdown(
        "Agent-based order-book simulation: fundamental, trend, market-maker and noise "
        "agents interact through a double-sided limit order book with price-time priority."
    )
    if not HAS_UNIVERSE:
        st.error("Agent-based market simulator is not available.")
        return

    c1, c2, c3 = st.columns(3)
    n_agents = c1.slider("Agents", 20, 500, 120)
    n_steps = c2.slider("Trading Steps", 100, 3000, 500)
    vol = c3.slider("Fundamental Volatility", 0.05, 0.50, 0.15)

    if st.button("Run Microstructure Simulation", type="primary"):
        with st.spinner("Simulating order flow..."):
            from market_simulation_universe import MarketSimulator
            sim = MarketSimulator(n_agents=n_agents, n_steps=n_steps,
                                  fundamental_vol=vol, seed=42)
            result = sim.run(initial_price=100.0)
            df = result.to_dataframe()

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Final Mid Price", f"${result.final_price:.2f}")
        col_m2.metric("Total Volume", f"{result.total_volume:,}")
        col_m3.metric("Trades Executed", f"{result.n_trades:,}")
        col_m4.metric("Agents", f"{result.params.get('n_agents', n_agents)}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["step"], y=df["mid_price"],
                                 mode="lines", name="Mid Price",
                                 line=dict(color=COLORS["gold"], width=2)))
        fig.add_trace(go.Scatter(x=df["step"], y=df["bid_price"],
                                 mode="lines", name="Best Bid",
                                 line=dict(color="#4caf50", width=1)))
        fig.add_trace(go.Scatter(x=df["step"], y=df["ask_price"],
                                 mode="lines", name="Best Ask",
                                 line=dict(color="#ef5350", width=1)))
        fig.update_layout(**_dark_layout(title="Order Book Microstructure", height=420))
        st.plotly_chart(fig, use_container_width=True)

        # Agent P&L distribution
        pnls = pd.DataFrame({
            "agent": list(result.agent_pnls.keys()),
            "pnl": list(result.agent_pnls.values()),
            "type": [t.value for t in result.agent_types.values()],
        })
        fig2 = go.Figure(go.Bar(x=pnls["type"], y=pnls["pnl"],
                                marker_color=COLORS["lavender"]))
        fig2.update_layout(**_dark_layout(title="Agent P&L by Type", height=320))
        st.plotly_chart(fig2, use_container_width=True)

def _render_portfolio_evolution_tab():
    """Portfolio equity-curve evolution with drawdown analytics."""
    st.markdown("<div class='sim-section-header'>Portfolio Evolution</div>", unsafe_allow_html=True)
    st.markdown(
        "Simulate portfolio equity evolution with drift, volatility, and periodic "
        "contributions; visualize drawdowns and recovery paths."
    )
    c1, c2, c3, c4 = st.columns(4)
    start = c1.number_input("Starting Capital ($)", 1000.0, 1e9, 100000.0, step=5000.0)
    drift = c2.slider("Annual Drift (%)", -20.0, 40.0, 8.0)
    vol_pct = c3.slider("Annual Volatility (%)", 5.0, 80.0, 20.0)
    days = c4.slider("Horizon (Days)", 30, 1500, 252)

    if st.button("Run Portfolio Evolution", type="primary"):
        rng = np.random.default_rng(7)
        dt = 1 / 252
        mu = drift / 100
        sig = vol_pct / 100
        rets = rng.normal(mu * dt, sig * np.sqrt(dt), days)
        equity = start * np.cumprod(1 + rets)
        cummax = np.maximum.accumulate(equity)
        drawdown = (equity / cummax - 1) * 100

        df = pd.DataFrame({"day": np.arange(1, days + 1), "equity": equity,
                           "drawdown": drawdown})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["day"], y=df["equity"], mode="lines",
                                 name="Equity", line=dict(color=COLORS["gold"], width=2),
                                 fill="tozeroy", fillcolor="rgba(201,168,76,0.06)"))
        fig.update_layout(**_dark_layout(title="Portfolio Equity Curve", height=360))
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df["day"], y=df["drawdown"], mode="lines",
                                  name="Drawdown", line=dict(color="#ef5350", width=1.5),
                                  fill="tozeroy", fillcolor="rgba(239,83,80,0.15)"))
        fig2.update_layout(**_dark_layout(title="Drawdown (%)", height=260))
        st.plotly_chart(fig2, use_container_width=True)

        total_ret = (equity[-1] / start - 1) * 100
        max_dd = float(drawdown.min())
        col_p1, col_p2, col_p3 = st.columns(3)
        col_p1.metric("Ending Capital", f"${equity[-1]:,.0f}")
        col_p2.metric("Total Return", f"{total_ret:+.2f}%")
        col_p3.metric("Max Drawdown", f"{max_dd:.2f}%")

def _simulate_crisis_path(scenario: dict, initial_price: float = 100.0, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = int(max(30, scenario.get("duration_days", 120)))
    rets = rng.normal(0, 0.01, n)
    px = initial_price * np.cumprod(1 + rets)
    return pd.DataFrame({"step": np.arange(n), "price": px})

def _render_crisis_tab():
    """Scenario & crisis simulation using real historical stress templates."""
    st.markdown("<div class='sim-section-header'>Crisis Scenarios</div>", unsafe_allow_html=True)
    st.markdown(
        "Stress-test price paths under historical crisis archetypes: flash crash, "
        "credit freeze, inflation shock, and geopolitical risk-off."
    )
    scenarios = {
        "Flash Crash (2010)": {"drift": -0.004, "vol": 0.035, "duration_days": 30, "recover": 0.35},
        "GFC Credit Freeze (2008)": {"drift": -0.008, "vol": 0.045, "duration_days": 120, "recover": 0.25},
        "Inflation / Rates Shock (2022)": {"drift": -0.003, "vol": 0.025, "duration_days": 90, "recover": 0.4},
        "Geopolitical Risk-Off": {"drift": -0.006, "vol": 0.03, "duration_days": 45, "recover": 0.3},
        "V-Shaped Recovery": {"drift": 0.001, "vol": 0.02, "duration_days": 60, "recover": 0.9},
    }
    name = st.selectbox("Crisis Scenario", list(scenarios.keys()))
    sc = scenarios[name]
    c1, c2 = st.columns(2)
    initial = c1.number_input("Initial Price", 10.0, 5000.0, 100.0)
    intensity = c2.slider("Stress Intensity", 0.5, 3.0, 1.0)

    if st.button("Run Crisis Simulation", type="primary"):
        rng = np.random.default_rng(11)
        n = sc["duration_days"]
        drift = sc["drift"] * intensity
        vol = sc["vol"] * intensity
        shock_day = int(n * 0.2)
        rets = rng.normal(drift, vol, n)
        # Drawdown phase then partial recovery
        rets[shock_day:] *= (1 - sc["recover"])
        px = initial * np.cumprod(1 + rets)
        df = pd.DataFrame({"step": np.arange(1, n + 1), "price": px})

        fig = go.Figure(go.Scatter(x=df["step"], y=df["price"], mode="lines",
                                   line=dict(color="#ef5350" if px[-1] < initial else COLORS["gold"], width=2),
                                   fill="tozeroy"))
        fig.update_layout(**_dark_layout(title=f"{name} — Stress Path", height=380))
        st.plotly_chart(fig, use_container_width=True)

        col_c1, col_c2, col_c3 = st.columns(3)
        col_c1.metric("Trough", f"${px.min():,.2f} ({(px.min()/initial-1)*100:+.1f}%)")
        col_c2.metric("End Price", f"${px[-1]:,.2f}")
        col_c3.metric("Max Pain", f"{(px.min()/initial-1)*100:.1f}%")

def _render_universe_tab():
    """Synthetic simulation-universe generator with asset-class weighting."""
    st.markdown("<div class='sim-section-header'>Simulation Universe Generator</div>", unsafe_allow_html=True)
    st.markdown(
        "Generate a synthetic financial universe (stocks, FX, crypto, bonds, commodities) "
        "with sector weights and correlation structure for downstream simulation."
    )
    c1, c2 = st.columns(2)
    size = c1.slider("Universe Size", 10, 500, 60)
    seed = c2.number_input("Seed", 0, 99999, 42)

    if st.button("Generate Universe", type="primary"):
        rng = np.random.default_rng(int(seed))
        assets = []
        for i in range(size):
            asset_type = rng.choice(["STOCK", "ETF", "FX", "CRYPTO", "BOND", "COMMODITY"],
                                    p=[0.45, 0.15, 0.15, 0.10, 0.10, 0.05])
            assets.append({"symbol": f"SYM{i:03d}", "asset_type": asset_type,
                           "initial_price": round(float(rng.uniform(5, 500)), 2),
                           "drift": round(float(rng.normal(0.05, 0.10)), 3),
                           "vol": round(float(rng.uniform(0.10, 0.60)), 3)})
        df_u = pd.DataFrame(assets)
        st.dataframe(df_u, use_container_width=True, hide_index=True, height=300)

        counts = df_u["asset_type"].value_counts()
        fig = go.Figure(go.Pie(labels=counts.index, values=counts.values,
                               hole=0.45, marker=dict(colors=[COLORS["gold"], COLORS["lavender"],
                                                              "#4caf50", "#ef5350", "#42a5f5", "#ab47bc"])))
        fig.update_layout(**_dark_layout(title="Universe Composition", height=340))
        st.plotly_chart(fig, use_container_width=True)
        st.session_state["sim_universe"] = assets
        st.success(f"Universe generated: {size} instruments across 6 asset classes.")

def _render_hyperdim_tab():
    """3-D parameter-space exploration of the agent-based simulator."""
    st.markdown("<div class='sim-section-header'>Hyperdimensional Parameter Space</div>", unsafe_allow_html=True)
    st.markdown(
        "Explore how simulator parameters (volatility, event probability, agent count) "
        "jointly shape realized volatility and final price — a 3-D sensitivity manifold."
    )
    if not HAS_UNIVERSE:
        st.error("Agent-based market simulator is not available.")
        return
    c1, c2 = st.columns(2)
    grid = c1.slider("Grid Resolution", 4, 12, 7)
    steps = c2.slider("Steps per Run", 100, 800, 250)

    if st.button("Explore Parameter Space", type="primary"):
        from market_simulation_universe import MarketSimulator
        vols = np.linspace(0.05, 0.40, grid)
        events = np.linspace(0.0, 0.04, grid)
        X, Y = np.meshgrid(vols, events)
        Z = np.zeros_like(X)
        for i in range(grid):
            for j in range(grid):
                sim = MarketSimulator(n_agents=80, n_steps=steps,
                                      fundamental_vol=float(X[i, j]),
                                      event_prob=float(Y[i, j]), seed=42)
                r = sim.run(initial_price=100.0)
                dfp = r.to_dataframe()
                rets = dfp["mid_price"].pct_change().dropna()
                Z[i, j] = float(rets.std() * np.sqrt(len(dfp)) * 100) if len(rets) else 0.0
        fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale="Viridis")])
        fig.update_layout(**_dark_layout(title="Realized Volatility Surface", height=520))
        fig.update_scenes(xaxis_title="Fundamental Vol", yaxis_title="Event Probability",
                          zaxis_title="Realized Vol %")
        st.plotly_chart(fig, use_container_width=True)

def _render_performance_tab():
    """Performance dashboard: grade recent simulations across engines."""
    st.markdown("<div class='sim-section-header'>Performance Dashboard</div>", unsafe_allow_html=True)
    st.markdown(
        "Grade recent simulations using the institutional simulation graders "
        "(P&L-weighted, drawdown-aware, consistency-scored)."
    )
    if not HAS_ENGINE:
        st.info("Run a Comprehensive Trading Simulation first to populate the dashboard.")
        return
    engine = MarketSimulationEngine()
    recent = engine.get_recent_simulations(limit=5)
    if not recent:
        st.info("No simulations recorded yet. Run the Comprehensive Trading Sim tab first.")
        return

    rows = []
    for sim in recent:
        pnl = sim.get("total_return", 0.0)
        rows.append({
            "Simulation ID": sim.get("simulation_id", "")[:18],
            "Regime": sim.get("market_regime", "").replace("_", " ").title(),
            "P&L": f"{pnl:+.2f}",
            "Win Rate": f"{sim.get('win_rate', 0.0)*100:.1f}%",
            "Decisions": sim.get("total_decisions", 0),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    try:
        from trading_system.simulation_grader import SimulationGrader
        grade = SimulationGrader().calculate_grade(recent)
        if grade:
            c1, c2 = st.columns([1, 2])
            score = grade.get("score", grade.get("grade_score", 0))
            letter = grade.get("letter_grade", grade.get("grade", "N/A"))
            c1.metric("Overall Grade", str(letter))
            c1.metric("Score", f"{score}/100")
            comps = grade.get("component_scores", {})
            if comps:
                c2.markdown("**Component Scores**")
                for k, v in comps.items():
                    c2.markdown(f"- {str(k).replace('_', ' ').title()}: **{v}**")
    except Exception:
        pass

def _render_bayesian_network_tab():
    """Bayesian dependency network for cross-asset propagation analysis."""
    st.markdown("<div class='sim-section-header'>Bayesian Network</div>", unsafe_allow_html=True)
    st.markdown(
        "Build a Bayesian dependency graph over core assets and inspect how shocks "
        "propagate through the network."
    )
    if not HAS_INST_ANALYTICS or not HAS_INST_VIZ:
        st.error("Institutional analytics modules are not available.")
        return
    core = st.text_input("Core Symbols (comma-separated)", "SPY, QQQ, TLT, GLD, DXY, CL=F")
    if st.button("Build Bayesian Network", type="primary"):
        with st.spinner("Building dependency graph..."):
            from institutional_analytics_engine import build_bayesian_network_from_data
            from institutional_visualizations import create_bayesian_network_graph
            net = build_bayesian_network_from_data([s.strip() for s in core.split(",") if s.strip()])
            fig = create_bayesian_network_graph(net.get("nodes", {}), net.get("edges", []))
            st.plotly_chart(fig, use_container_width=True)
            if net.get("hubs"):
                st.markdown("**Network Hubs (highest influence):**")
                for h in net["hubs"]:
                    st.markdown(f"- {h}")

def _render_macro_analyzer_tab():
    """Institutional macro-regime analyzer with radar visualization."""
    st.markdown("<div class='sim-section-header'>Macro Analyzer</div>", unsafe_allow_html=True)
    st.markdown(
        "Run institutional macro-regime detection across rates, inflation, liquidity "
        "and risk factors, with a radar breakdown of factor changes."
    )
    if not HAS_INST_ANALYTICS:
        st.error("Institutional analytics module is not available.")
        return
    if st.button("Run Macro Analysis", type="primary"):
        with st.spinner("Running macro-regime analysis..."):
            from institutional_analytics_engine import run_macro_analysis
            from institutional_visualizations import create_macro_dashboard_chart
            res = run_macro_analysis()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Regime", res.regime)
        c2.metric("Trend", res.trend_direction)
        c3.metric("Volatility", res.volatility_regime)
        c4.metric("Liquidity", res.liquidity_conditions)
        fig = create_macro_dashboard_chart(res.details)
        st.plotly_chart(fig, use_container_width=True)
        if res.factor_correlations:
            st.markdown("**Factor Correlations**")
            st.json(res.factor_correlations)

def _render_micro_analyzer_tab():
    """Institutional micro-structure analyzer: sectors, breadth, momentum."""
    st.markdown("<div class='sim-section-header'>Micro Analyzer</div>", unsafe_allow_html=True)
    st.markdown(
        "Institutional micro-structure analysis — sector rotations, breadth metrics, "
        "momentum breakdown, volatility clusters, and correlation regime."
    )
    if not HAS_INST_ANALYTICS:
        st.error("Institutional analytics module is not available.")
        return
    if st.button("Run Micro Analysis", type="primary"):
        with st.spinner("Running micro-structure analysis..."):
            from institutional_analytics_engine import run_micro_analysis
            res = run_micro_analysis()
        c1, c2 = st.columns(2)
        c1.metric("Correlation Regime", res.correlation_regime)
        c2.metric("Breadth (% Advancing)",
                  f"{res.breadth_metrics.get('breadth', 0.0)*100:.1f}%")
        if res.sector_rankings:
            from institutional_visualizations import create_sector_momentum_chart
            fig = create_sector_momentum_chart(res.sector_rankings)
            st.plotly_chart(fig, use_container_width=True)
        if res.momentum_breakdown:
            st.markdown("**Momentum Breakdown**")
            st.json(res.momentum_breakdown)

def _render_scenarios_tab():
    """AI-generated market scenarios with cascade and probability visuals."""
    st.markdown("<div class='sim-section-header'>AI Scenarios</div>", unsafe_allow_html=True)
    st.markdown(
        "Generate probabilistic macro/market scenarios with causal propagation "
        "pathways and severity ratings."
    )
    if not HAS_INST_ANALYTICS:
        st.error("Scenario generator module is not available.")
        return
    if st.button("Generate Market Scenarios", type="primary"):
        with st.spinner("Generating scenarios..."):
            from institutional_analytics_engine import generate_market_scenarios
            from institutional_visualizations import create_scenario_cascade_chart, create_probability_heatmap
            scenarios = generate_market_scenarios()
        if not scenarios:
            st.info("No scenarios generated — market data may be unavailable.")
            return
        for sc in scenarios:
            with st.expander(f"{sc.name}  —  {sc.probability*100:.0f}% probability  ({sc.severity})"):
                st.markdown(f"**Macro impact:** {sc.macro_impact}")
                st.markdown(f"**Market impact:** {sc.market_impact}")
                st.markdown(f"**Affected assets:** {', '.join(sc.affected_assets)}")
                st.markdown(f"**Expected equity move:** {sc.expected_equity_move:+.1f}% · "
                            f"**Expected vol change:** {sc.expected_vol_change:+.1f}%")
                fig = create_scenario_cascade_chart({
                    "name": sc.name, "causal_chain": sc.causal_chain,
                    "expected_equity_move": sc.expected_equity_move,
                })
                st.plotly_chart(fig, use_container_width=True)

def _render_options_analytics_tab():
    """Options market simulation: Monte Carlo P&L panel."""
    st.markdown("<div class='sim-section-header'>Options Market Simulation</div>", unsafe_allow_html=True)
    if render_options_monte_carlo_panel is not None:
        render_options_monte_carlo_panel()
    else:
        st.error("Options Monte Carlo panel is not available.")

def _render_futures_commodity_sim_tab():
    """Futures & commodity simulation: basis, term structure, roll yield."""
    st.markdown("<div class='sim-section-header'>Futures & Commodity Simulation</div>", unsafe_allow_html=True)
    st.markdown(
        "Analyze futures term structure, basis, and roll-yield optimization across "
        "commodity, index, and rates contracts."
    )
    if not HAS_FUTURES:
        st.error("Futures engine is not available.")
        return
    from futures_engine import get_futures_engine
    fe = get_futures_engine()
    symbols = sorted(FUTURES_UNIVERSE.keys())[:20]
    if not symbols:
        st.info("No futures universe configured.")
        return
    sym = st.selectbox("Contract", symbols)
    c1, c2 = st.columns(2)
    spot = c1.number_input("Spot Price", 0.01, 1e6, 100.0)
    fut = c2.number_input("Front-Month Futures Price", 0.01, 1e6, 102.0)

    if st.button("Run Futures Analytics", type="primary"):
        basis = fe.get_basis_analysis(sym, float(spot), float(fut), T=0.25)
        if basis:
            st.markdown("**Basis Analysis**")
            st.json(basis)
        cot = fe.get_cot_positioning(sym)
        if cot:
            st.markdown("**COT Positioning**")
            st.json(cot)
        # Term structure curve
        months = [1, 2, 3, 6, 9, 12]
        try:
            curve = fe.schwartz_smith_term_structure(float(spot), months,
                                                     spot_vol=0.25, long_vol=0.10,
                                                     mean_reversion=0.8)
            if isinstance(curve, dict):
                curve = curve.get("prices", curve.get("curve", []))
            if curve:
                fig = go.Figure(go.Scatter(x=[f"{m}M" for m in months], y=curve,
                                           mode="lines+markers",
                                           line=dict(color=COLORS["gold"], width=2)))
                fig.update_layout(**_dark_layout(title=f"{sym} Term Structure", height=340))
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

def _render_derivative_dynamics_tab():
    """Derivative dynamics: options & futures pricing sensitivity surfaces."""
    st.markdown("<div class='sim-section-header'>Derivative Dynamics</div>", unsafe_allow_html=True)
    st.markdown(
        "Interactive option pricing dynamics (Black-Scholes / Bjerksund-Stensland) "
        "with volatility-surface visualization."
    )
    if not HAS_OPTIONS:
        st.error("Options engine is not available.")
        return
    from options_engine import get_options_engine
    oe = get_options_engine()
    c1, c2, c3, c4 = st.columns(4)
    S = c1.number_input("Spot (S)", 1.0, 1e6, 100.0)
    K = c2.number_input("Strike (K)", 1.0, 1e6, 105.0)
    T = c3.number_input("Time to Expiry (yrs)", 0.01, 5.0, 0.25)
    sigma = c4.number_input("Volatility", 0.01, 2.0, 0.25)
    typ = st.radio("Option Type", ["call", "put"], horizontal=True)

    if st.button("Price Option", type="primary"):
        bs = oe.black_scholes(S, K, T, sigma, typ)
        c_b1, c_b2, c_b3, c_b4 = st.columns(4)
        c_b1.metric("BS Price", f"${bs.get('price', 0):.2f}")
        c_b2.metric("Delta", f"{bs.get('delta', 0):.3f}")
        c_b3.metric("Gamma", f"{bs.get('gamma', 0):.4f}")
        c_b4.metric("Theta", f"{bs.get('theta', 0):.4f}")
        try:
            bs_strike = oe.bjerksund_stensland(S, K, T, sigma, typ)
            st.metric("American (Bjerksund-Stensland)", f"${bs_strike:.2f}")
        except Exception:
            pass
        # Volatility surface
        from institutional_visualizations import create_volatility_surface_chart
        strikes = [round(K * m) for m in [0.8, 0.9, 1.0, 1.1, 1.2]]
        tenors = ["1M", "3M", "6M", "1Y"]
        vol_matrix = np.array([
            [sigma * (1 + 0.15 * abs(s - K) / K) * (1 + 0.02 * ti) for s in strikes]
            for ti in range(4)
        ])
        fig = create_volatility_surface_chart(strikes, tenors, vol_matrix)
        st.plotly_chart(fig, use_container_width=True)

def _render_comprehensive_sim_tab():
    st.markdown("<div class='sim-section-header'>Comprehensive Trading Simulation</div>", unsafe_allow_html=True)
    st.markdown("Run full-scale market simulations across the asset universe to train the AI model, generate scenario responses, and grade decision-making algorithms.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("<div class='sim-card'>", unsafe_allow_html=True)
        st.markdown("#### Simulation Controls")
        sim_duration = st.slider("Simulation Duration (Minutes)", 5, 120, 30)
        sim_universe = st.slider("Universe Size (Assets)", 10, 500, 50)
        
        if st.button("Run Full Simulation", type="primary", use_container_width=True):
            if not HAS_ENGINE:
                st.error("MarketSimulationEngine is not available.")
            else:
                with st.spinner("Initializing simulation environment..."):
                    engine = MarketSimulationEngine(
                        universe_size=sim_universe,
                        simulation_duration=sim_duration
                    )
                    
                    # Create a progress placeholder
                    progress_text = st.empty()
                    progress_bar = st.progress(0)
                    
                    # Instead of blocking UI entirely for long, we can run it in a thread
                    # For Streamlit, running synchronously for a demo is okay if duration is short,
                    # but simulation duration is in real-time minutes. We will mock a faster run
                    # by patching the duration or just letting it run.
                    st.warning("Running real-time simulation. This will block the interface until complete.")
                    try:
                        # Patch duration to be seconds instead of minutes for UI execution
                        import datetime
                        engine.simulation_duration = datetime.timedelta(seconds=sim_duration)
                        engine.run_daily_simulation()
                        st.success("Simulation completed successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Simulation failed: {e}")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='sim-card'>", unsafe_allow_html=True)
        st.markdown("#### Recent Simulation Results")
        if HAS_ENGINE:
            engine = MarketSimulationEngine()
            recent = engine.get_recent_simulations(limit=5)
            
            if not recent:
                st.info("No recent simulations found. Run one to see results.")
            else:
                for sim in recent:
                    sim_id = sim.get('simulation_id', 'Unknown')
                    regime = sim.get('market_regime', 'Unknown').replace('_', ' ').title()
                    win_rate = sim.get('win_rate', 0.0) * 100
                    pnl = sim.get('total_return', 0.0)
                    decisions = sim.get('total_decisions', 0)
                    
                    color = "#4caf50" if pnl >= 0 else "#ef5350"
                    
                    st.markdown(f"""
                    <div style="border-left: 3px solid {color}; padding-left: 10px; margin-bottom: 10px; background: rgba(0,0,0,0.2); padding: 10px; border-radius: 4px;">
                        <div style="display: flex; justify-content: space-between;">
                            <strong>{sim_id}</strong>
                            <span style="color: {color}; font-weight: bold;">{pnl:+.2f} PnL</span>
                        </div>
                        <div style="font-size: 0.85em; color: #a0a8b8; margin-top: 5px;">
                            Regime: <span class="sim-regime-label" style="background: rgba(201,168,76,0.15); color: #c9a84c;">{regime}</span> | 
                            Win Rate: {win_rate:.1f}% | Decisions: {decisions}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

def render_simulation_viewer():
    st.markdown(_SIM_CSS, unsafe_allow_html=True)
    st.title("Simulation Hub")
    st.caption(
        "Visual simulation laboratory — market microstructure, portfolio evolution, "
        "crisis scenarios, synthetic universes, and derivatives analytics."
    )

    tabs = st.tabs([
        "Market Microstructure",
        "Portfolio Evolution",
        "Crisis Scenarios",
        "Simulation Universe",
        "Hyperdimensional",
        "Comprehensive Trading Sim",
        "Performance Dashboard",
        "Bayesian Network",
        "Macro Analyzer",
        "Micro Analyzer",
        "AI Scenarios",
        "Derivative Dynamics",
        "Options Market Simulation",
        "Futures & Commodity Simulation",
    ])

    with tabs[0]:
        _render_microstructure_tab()
    with tabs[1]:
        _render_portfolio_evolution_tab()
    with tabs[2]:
        _render_crisis_tab()
    with tabs[3]:
        _render_universe_tab()
    with tabs[4]:
        _render_hyperdim_tab()
    with tabs[5]:
        _render_comprehensive_sim_tab()
    with tabs[6]:
        _render_performance_tab()
    with tabs[7]:
        _render_bayesian_network_tab()
    with tabs[8]:
        _render_macro_analyzer_tab()
    with tabs[9]:
        _render_micro_analyzer_tab()
    with tabs[10]:
        _render_scenarios_tab()
    with tabs[11]:
        _render_derivative_dynamics_tab()
    with tabs[12]:
        _render_options_analytics_tab()
    with tabs[13]:
        _render_futures_commodity_sim_tab()
