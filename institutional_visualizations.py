"""
Institutional Visualizations
==============================
Research-grade Plotly charting helpers for the Octavian Simulation Hub.
All charts use the dark Octavian theme (plotly_dark + gold/amber palette).

Functions:
  - create_bayesian_network_graph
  - create_correlation_heatmap
  - create_regime_map
  - create_probability_heatmap
  - create_factor_decomposition_chart
  - create_volatility_surface_chart
  - create_macro_dashboard_chart
  - create_scenario_cascade_chart

ADDITIVE module — nothing in existing code is modified.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Octavian palette ──────────────────────────────────────────────
_GOLD = "#c9a84c"
_GOLD_LIGHT = "#e0c97f"
_GOLD_DARK = "#8B6914"
_BG = "#0a1628"
_CARD = "#132240"
_TEXT = "#e8eaf0"
_TEXT_SEC = "#8a9ab5"
_GREEN = "#4caf50"
_RED = "#ef5350"
_BLUE = "#42a5f5"
_PURPLE = "#ab47bc"
_ORANGE = "#ffa726"


def _dark_layout(**overrides: Any) -> dict:
    """Common dark layout kwargs."""
    layout = dict(
        template="plotly_dark",
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(color=_TEXT, family="Inter, sans-serif"),
        margin=dict(l=50, r=30, t=50, b=40),
    )
    layout.update(overrides)
    return layout


# ────────────────────────────────────────────────────────────────────
# BAYESIAN NETWORK GRAPH
# ────────────────────────────────────────────────────────────────────

def create_bayesian_network_graph(
    nodes: Dict[str, Dict],
    edges: List[Tuple[str, str, float]],
) -> go.Figure:
    """
    Interactive network graph for the Bayesian probability model.

    nodes: {name: {"layer": ..., "state": float, "description": ...}}
    edges: [(parent, child, weight), ...]
    """
    # Assign positions by layer
    layer_order = {"macro": 0, "market": 1, "asset": 2}
    layers: Dict[str, list] = {"macro": [], "market": [], "asset": []}
    for name, info in nodes.items():
        layers.setdefault(info["layer"], []).append(name)

    positions: Dict[str, Tuple[float, float]] = {}
    for layer_name, members in layers.items():
        y = 1.0 - layer_order.get(layer_name, 1) * 0.45
        for i, member in enumerate(members):
            x = (i + 1) / (len(members) + 1)
            positions[member] = (x, y)

    fig = go.Figure()

    # ── Draw edges ──
    for parent, child, weight in edges:
        if parent not in positions or child not in positions:
            continue
        x0, y0 = positions[parent]
        x1, y1 = positions[child]

        # Edge colour based on strength
        opacity = max(0.25, min(1.0, weight))
        edge_width = max(1, weight * 5)

        fig.add_trace(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(color=f"rgba(201,168,76,{opacity})", width=edge_width),
            hoverinfo="text",
            text=f"{parent} → {child}<br>Weight: {weight:.2f}",
            showlegend=False,
        ))

    # ── Draw nodes ──
    for name, info in nodes.items():
        if name not in positions:
            continue
        x, y = positions[name]
        state = info.get("state", 0.5)

        # Colour gradient: red (low) → gold → green (high)
        if state < 0.4:
            colour = _RED
        elif state > 0.6:
            colour = _GREEN
        else:
            colour = _GOLD

        size = 28 + state * 20  # 28-48 range

        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            marker=dict(size=size, color=colour, line=dict(width=2, color=_GOLD_LIGHT),
                        opacity=0.9),
            text=f"{name}<br>{state:.0%}",
            textposition="bottom center",
            textfont=dict(size=10, color=_TEXT),
            hovertext=(f"<b>{name}</b><br>"
                       f"Layer: {info.get('layer', '')}<br>"
                       f"State: {state:.1%}<br>"
                       f"{info.get('description', '')}"),
            hoverinfo="text",
            showlegend=False,
        ))

    # Layer labels
    for layer_name, y_pos in [("MACRO", 1.0), ("MARKET", 0.55), ("ASSET", 0.10)]:
        fig.add_annotation(
            x=-0.02, y=y_pos, text=f"<b>{layer_name}</b>",
            showarrow=False, font=dict(size=12, color=_GOLD_LIGHT),
            xanchor="right",
        )

    fig.update_layout(
        **_dark_layout(
            title=dict(text="Bayesian Probability Network", font=dict(size=16, color=_GOLD)),
            height=550,
            xaxis=dict(visible=False, range=[-0.05, 1.05]),
            yaxis=dict(visible=False, range=[-0.05, 1.15]),
            hovermode="closest",
        )
    )
    return fig


# ────────────────────────────────────────────────────────────────────
# CORRELATION HEATMAP
# ────────────────────────────────────────────────────────────────────

def create_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Cross-Asset Correlation Matrix",
) -> go.Figure:
    """Professional heatmap with hierarchical colouring."""
    labels = list(corr_matrix.columns)
    values = corr_matrix.values

    # Custom colourscale: blue (neg) → black (0) → gold (pos)
    colourscale = [
        [0.0, "#1565c0"],
        [0.25, "#42a5f5"],
        [0.5, "#1a1a2e"],
        [0.75, "#c9a84c"],
        [1.0, "#ffd700"],
    ]

    fig = go.Figure(data=go.Heatmap(
        z=values,
        x=labels, y=labels,
        colorscale=colourscale,
        zmin=-1, zmax=1,
        text=np.round(values, 2),
        texttemplate="%{text}",
        textfont=dict(size=10, color=_TEXT),
        hovertemplate="<b>%{x} vs %{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
        colorbar=dict(title="ρ", tickfont=dict(color=_TEXT)),
    ))

    fig.update_layout(**_dark_layout(
        title=dict(text=title, font=dict(color=_GOLD)),
        height=500,
        xaxis=dict(tickangle=45),
    ))
    return fig


# ────────────────────────────────────────────────────────────────────
# REGIME MAP
# ────────────────────────────────────────────────────────────────────

def create_regime_map(regime_data: Dict[str, Dict]) -> go.Figure:
    """
    Multi-timeframe regime overlay chart.

    regime_data: {"short_term": {"trend": ..., "return": ...}, ...}
    """
    timeframes = ["Short-Term\n(1h–1d)", "Medium-Term\n(1w–1m)", "Long-Term\n(3m–5y)"]
    keys = ["short_term", "medium_term", "long_term"]
    colours_map = {"Bullish": _GREEN, "Bearish": _RED, "Neutral": _GOLD, "Unknown": _TEXT_SEC}

    trends = []
    rets = []
    cols = []
    for k in keys:
        rd = regime_data.get(k, {})
        t = rd.get("trend", "Unknown")
        r = rd.get("return", "0%")
        trends.append(t)
        rets.append(r)
        cols.append(colours_map.get(t, _TEXT_SEC))

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=timeframes,
        y=[1, 1, 1],
        marker=dict(color=cols, opacity=0.7, line=dict(width=2, color=_GOLD_LIGHT)),
        text=[f"{t}<br>{r}" for t, r in zip(trends, rets)],
        textposition="inside",
        textfont=dict(size=14, color="white"),
        hoverinfo="text",
        hovertext=[f"Trend: {t} | Return: {r}" for t, r in zip(trends, rets)],
        showlegend=False,
    ))

    fig.update_layout(**_dark_layout(
        title=dict(text="Multi-Timeframe Regime Map", font=dict(color=_GOLD)),
        height=300,
        yaxis=dict(visible=False),
        xaxis=dict(tickfont=dict(size=12, color=_TEXT)),
        bargap=0.15,
    ))
    return fig


# ────────────────────────────────────────────────────────────────────
# PROBABILITY HEATMAP  (scenario × asset)
# ────────────────────────────────────────────────────────────────────

def create_probability_heatmap(scenarios: List[Dict]) -> go.Figure:
    """
    Grid heatmap: rows = scenarios, columns = affected assets.
    Cell value = expected move.
    """
    if not scenarios:
        fig = go.Figure()
        fig.update_layout(**_dark_layout(title="No Scenarios Available", height=200))
        return fig

    scenario_names = [s.get("name", f"S{i}") for i, s in enumerate(scenarios)]
    all_assets = sorted({a for s in scenarios for a in s.get("affected_assets", [])})

    z_vals = []
    for s in scenarios:
        row = []
        equity_move = s.get("expected_equity_move", 0)
        for asset in all_assets:
            if asset in s.get("affected_assets", []):
                # Scale proportionally; bonds move inversely
                if "Bond" in asset:
                    row.append(-equity_move * 0.5)
                elif "Commodit" in asset:
                    row.append(equity_move * 0.6)
                else:
                    row.append(equity_move)
            else:
                row.append(0)
        z_vals.append(row)

    colourscale = [
        [0.0, "#b71c1c"],
        [0.35, "#ef5350"],
        [0.5, "#1a1a2e"],
        [0.65, "#4caf50"],
        [1.0, "#00e676"],
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z_vals,
        x=all_assets,
        y=scenario_names,
        colorscale=colourscale,
        zmid=0,
        text=np.round(z_vals, 1),
        texttemplate="%{text}%",
        textfont=dict(size=10),
        hovertemplate="<b>%{y}</b><br>Asset: %{x}<br>Expected Move: %{z:.1f}%<extra></extra>",
        colorbar=dict(title="Move %", tickfont=dict(color=_TEXT)),
    ))

    fig.update_layout(**_dark_layout(
        title=dict(text="Scenario Impact Heatmap", font=dict(color=_GOLD)),
        height=max(300, len(scenarios) * 50 + 100),
        xaxis=dict(tickangle=30),
    ))
    return fig


# ────────────────────────────────────────────────────────────────────
# FACTOR DECOMPOSITION
# ────────────────────────────────────────────────────────────────────

def create_factor_decomposition_chart(
    factors: Dict[str, float],
    title: str = "Factor Attribution",
) -> go.Figure:
    """Horizontal stacked bar showing factor contributions."""
    sorted_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)
    names = [f[0] for f in sorted_factors]
    values = [f[1] for f in sorted_factors]

    colours = [_GREEN if v > 0 else _RED for v in values]

    fig = go.Figure(go.Bar(
        y=names,
        x=values,
        orientation="h",
        marker=dict(color=colours, opacity=0.85, line=dict(width=1, color=_GOLD_LIGHT)),
        text=[f"{v:+.2f}" for v in values],
        textposition="auto",
        textfont=dict(color=_TEXT, size=11),
    ))

    fig.update_layout(**_dark_layout(
        title=dict(text=title, font=dict(color=_GOLD)),
        height=max(250, len(factors) * 35 + 100),
        xaxis=dict(title="Contribution", zeroline=True, zerolinecolor=_GOLD_DARK),
        yaxis=dict(autorange="reversed"),
    ))
    return fig


# ────────────────────────────────────────────────────────────────────
# VOLATILITY SURFACE (3D)
# ────────────────────────────────────────────────────────────────────

def create_volatility_surface_chart(
    strikes: Optional[List[float]] = None,
    tenors: Optional[List[str]] = None,
    vol_matrix: Optional[np.ndarray] = None,
) -> go.Figure:
    """
    3D implied-volatility surface.
    If no data provided, generates a synthetic ATM surface.
    """
    if strikes is None:
        strikes = list(np.arange(0.80, 1.21, 0.05))
    if tenors is None:
        tenors = ["1W", "2W", "1M", "2M", "3M", "6M", "1Y"]
    if vol_matrix is None:
        # Synthetic smile + term structure
        vol_matrix = np.zeros((len(tenors), len(strikes)))
        for i, _ in enumerate(tenors):
            for j, k in enumerate(strikes):
                atm = 18 + i * 0.8  # term structure slope
                skew = 15 * (1 - k) ** 2  # smile
                vol_matrix[i, j] = atm + skew + np.random.normal(0, 0.5)

    fig = go.Figure(data=[go.Surface(
        z=vol_matrix,
        x=strikes,
        y=list(range(len(tenors))),
        colorscale=[
            [0, "#1565c0"],
            [0.3, "#42a5f5"],
            [0.5, _GOLD],
            [0.7, _ORANGE],
            [1.0, _RED],
        ],
        hovertemplate="Strike: %{x:.2f}<br>Tenor: %{y}<br>IV: %{z:.1f}%<extra></extra>",
        colorbar=dict(title="IV %", tickfont=dict(color=_TEXT)),
    )])

    fig.update_layout(**_dark_layout(
        title=dict(text="Implied Volatility Surface", font=dict(color=_GOLD)),
        height=550,
        scene=dict(
            xaxis=dict(title="Strike (moneyness)", color=_TEXT),
            yaxis=dict(title="Tenor", tickvals=list(range(len(tenors))),
                       ticktext=tenors, color=_TEXT),
            zaxis=dict(title="IV (%)", color=_TEXT),
            bgcolor=_BG,
        ),
    ))
    return fig


# ────────────────────────────────────────────────────────────────────
# MACRO DASHBOARD CHART
# ────────────────────────────────────────────────────────────────────

def create_macro_dashboard_chart(macro_details: Dict[str, Any]) -> go.Figure:
    """Radar chart of macro factor changes."""
    categories = [
        "Interest Rates",
        "Inflation",
        "Growth",
        "Liquidity",
        "USD Strength",
    ]
    values = [
        macro_details.get("interest_rate_change_63d", 0),
        macro_details.get("inflation_proxy_change_63d", 0),
        macro_details.get("growth_proxy_change_63d", 0),
        macro_details.get("liquidity_proxy_change_63d", 0),
        macro_details.get("usd_change_63d", 0),
    ]
    # Close the polygon
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor=f"rgba(201,168,76,0.2)",
        line=dict(color=_GOLD, width=2),
        name="63d Change (%)",
    ))

    fig.update_layout(**_dark_layout(
        title=dict(text="Macro Factor Radar (63-day change %)", font=dict(color=_GOLD)),
        height=450,
        polar=dict(
            bgcolor=_BG,
            radialaxis=dict(visible=True, color=_TEXT_SEC, gridcolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(color=_TEXT),
        ),
        showlegend=False,
    ))
    return fig


# ────────────────────────────────────────────────────────────────────
# SCENARIO CASCADE CHART
# ────────────────────────────────────────────────────────────────────

def create_scenario_cascade_chart(scenario: Dict) -> go.Figure:
    """
    Waterfall/cascade chart showing a single scenario's causal chain
    and cumulative impact.
    """
    chain = scenario.get("causal_chain", [])
    if not chain:
        fig = go.Figure()
        fig.update_layout(**_dark_layout(title="No Causal Chain", height=200))
        return fig

    total_move = scenario.get("expected_equity_move", 0)
    n = len(chain)
    # Distribute move across steps with decaying magnitude
    step_impacts = []
    remaining = total_move
    for i in range(n):
        impact = remaining * 0.45
        step_impacts.append(round(impact, 2))
        remaining -= impact

    fig = go.Figure(go.Waterfall(
        name="Causal Impact",
        orientation="v",
        x=chain,
        y=step_impacts,
        connector=dict(line=dict(color=_GOLD_DARK, width=1)),
        increasing=dict(marker=dict(color=_GREEN)),
        decreasing=dict(marker=dict(color=_RED)),
        totals=dict(marker=dict(color=_GOLD)),
        text=[f"{v:+.2f}%" for v in step_impacts],
        textposition="outside",
        textfont=dict(color=_TEXT, size=10),
    ))

    fig.update_layout(**_dark_layout(
        title=dict(text=f"Causal Cascade: {scenario.get('name', '')}", font=dict(color=_GOLD)),
        height=400,
        xaxis=dict(tickangle=25),
        yaxis=dict(title="Equity Impact (%)", zeroline=True, zerolinecolor=_GOLD_DARK),
    ))
    return fig


# ────────────────────────────────────────────────────────────────────
# SECTOR MOMENTUM BAR CHART
# ────────────────────────────────────────────────────────────────────

def create_sector_momentum_chart(sector_rankings: List[Dict]) -> go.Figure:
    """Horizontal bar chart of sector momentum scores."""
    if not sector_rankings:
        fig = go.Figure()
        fig.update_layout(**_dark_layout(title="No Sector Data", height=200))
        return fig

    names = [s["sector"] for s in sector_rankings]
    scores = [s["momentum_score"] for s in sector_rankings]
    colours = [_GREEN if s > 0 else _RED for s in scores]

    fig = go.Figure(go.Bar(
        y=names,
        x=scores,
        orientation="h",
        marker=dict(color=colours, opacity=0.85, line=dict(width=1, color=_GOLD_LIGHT)),
        text=[f"{s:+.1f}" for s in scores],
        textposition="auto",
        textfont=dict(size=11, color=_TEXT),
    ))

    fig.update_layout(**_dark_layout(
        title=dict(text="Sector Momentum Rankings", font=dict(color=_GOLD)),
        height=max(300, len(names) * 32 + 80),
        xaxis=dict(title="Momentum Score", zeroline=True, zerolinecolor=_GOLD_DARK),
        yaxis=dict(autorange="reversed"),
    ))
    return fig


# ────────────────────────────────────────────────────────────────────
# MARKET BREADTH GAUGE
# ────────────────────────────────────────────────────────────────────

def create_breadth_gauge(pct_above_50d: float) -> go.Figure:
    """Gauge showing % of sectors above their 50-day SMA."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct_above_50d,
        number=dict(suffix="%", font=dict(color=_GOLD, size=36)),
        title=dict(text="Sectors Above 50d SMA", font=dict(color=_TEXT, size=14)),
        gauge=dict(
            axis=dict(range=[0, 100], tickfont=dict(color=_TEXT_SEC)),
            bar=dict(color=_GOLD),
            bgcolor=_BG,
            bordercolor=_GOLD_DARK,
            steps=[
                dict(range=[0, 30], color="rgba(239,83,80,0.3)"),
                dict(range=[30, 60], color="rgba(255,167,38,0.2)"),
                dict(range=[60, 100], color="rgba(76,175,80,0.3)"),
            ],
            threshold=dict(line=dict(color=_GOLD_LIGHT, width=3), thickness=0.75,
                           value=pct_above_50d),
        ),
    ))

    fig.update_layout(**_dark_layout(height=280))
    return fig
