"""
Comparative Analysis Hub — Octavian Terminal
=============================================
Streamlit UI for institutional cross-asset comparisons.

Features:
- Unlimited assets with auto-detect + manual type override per symbol
- Custom model mode selection (Quant / Fundamentals / Macro / Technical / Full / Custom)
- Side-by-side metric cards
- Price overlay chart (normalized to 100)
- Correlation heatmap
- Radar/spider chart (per-asset dimension scoring)
- Rolling performance comparison (30/60/90D)
- Asset ranking bar chart
- Options Greeks table (when applicable)
- Cross-asset conclusions (head-to-head write-ups)
- Adjustable strategy suggestions with AI theses
"""

import streamlit as st
import asyncio
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional

from comparative_analysis_engine import (
    get_comparative_engine,
    ComparativeResult,
    AssetScore,
    CrossAssetStrategy,
    detect_asset_type,
)

# ─────────────────────────────────────────────────────────────────────────────
# UI CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

SIGNAL_COLORS = {
    "STRONG BUY": "#00e676",
    "BUY": "#69f0ae",
    "NEUTRAL": "#ffd740",
    "SELL": "#ff5252",
    "STRONG SELL": "#d50000",
}

ASSET_TYPE_COLORS = {
    "stock": "#4fc3f7",
    "crypto": "#ce93d8",
    "forex": "#a5d6a7",
    "futures": "#ffcc80",
    "commodity": "#ffcc80",
    "options": "#f48fb1",
}

ASSET_ICONS = {
    "stock": "",
    "crypto": "",
    "forex": "",
    "futures": "",
    "commodity": "",
    "options": "",
}

MODEL_PRESETS = {
    "Full Institutional": {"use_quant": True, "use_fundamentals": True, "use_macro": True, "use_technical": True},
    "Quant Only": {"use_quant": True, "use_fundamentals": False, "use_macro": False, "use_technical": False},
    "Fundamentals Only": {"use_quant": False, "use_fundamentals": True, "use_macro": False, "use_technical": False},
    "Macro Only": {"use_quant": False, "use_fundamentals": False, "use_macro": True, "use_technical": False},
    "Technical Only": {"use_quant": False, "use_fundamentals": False, "use_macro": False, "use_technical": True},
    "Quant + Technical": {"use_quant": True, "use_fundamentals": False, "use_macro": False, "use_technical": True},
    "Custom Mix": None,
}

# ─────────────────────────────────────────────────────────────────────────────
# REUSABLE UI COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

def _styled_metric(label: str, value: str, sub: str = "", color: str = "#e8eaf0"):
    """Render a styled metric card."""
    sub_html = f"<div style='font-size:0.7rem; color:#6b7280; margin-top:2px;'>{sub}</div>" if sub else ""
    st.markdown(
        f"""
        <div style='
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 8px;
            padding: 12px 14px;
            margin-bottom: 8px;
        '>
            <div style='font-size:0.72rem; color:#8892b0; text-transform:uppercase;
                        letter-spacing:0.06em; margin-bottom:3px;'>{label}</div>
            <div style='font-size:1.25rem; font-weight:700; color:{color};'>{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _signal_badge(signal: str):
    """Render a colored signal badge."""
    color = SIGNAL_COLORS.get(signal, "#ffd740")
    st.markdown(
        f"""<div style='
            display:inline-block;
            background:{color}22;
            border:1px solid {color};
            border-radius:20px;
            padding:4px 14px;
            color:{color};
            font-weight:700;
            font-size:0.85rem;
            margin:4px 0;
        '>{signal}</div>""",
        unsafe_allow_html=True,
    )


def _section_header(text: str, icon: str = ""):
    st.markdown(
        f"<h3 style='color:#e8eaf0; margin-top:1.8rem; margin-bottom:0.6rem;'>{icon} {text}</h3>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CHART GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def _chart_price_overlay(result: ComparativeResult, assets_input: List[Dict]) -> Optional[go.Figure]:
    """Normalized price overlay chart (all assets rebased to 100)."""
    engine = get_comparative_engine()

    fig = go.Figure()
    colors = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel

    for i, ainfo in enumerate(assets_input):
        sym = ainfo["symbol"]
        atype = ainfo.get("asset_type") or detect_asset_type(sym)
        try:
            df = engine._fetch_price_data(sym, atype, period="6mo")
            if df is None or df.empty:
                continue
            close = df["Close"].astype(float)
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.dropna()
            if len(close) < 2:
                continue
            normalized = (close / close.iloc[0]) * 100
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=normalized.index, y=normalized,
                name=sym,
                line=dict(color=color, width=2),
                mode="lines",
                hovertemplate=f"<b>{sym}</b><br>Normalized: %{{y:.2f}}<extra></extra>"
            ))
        except Exception:
            continue

    if not fig.data:
        return None

    fig.update_layout(
        title="Price Performance — Normalized to 100",
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        font=dict(color="#e8eaf0"),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.15),
        hovermode="x unified",
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        margin=dict(t=50, b=60, l=50, r=20),
    )
    return fig


def _chart_correlation_heatmap(result: ComparativeResult) -> Optional[go.Figure]:
    """Correlation heatmap."""
    if result.correlation_matrix.empty or result.correlation_matrix.shape[0] < 2:
        return None

    corr = result.correlation_matrix
    symbols = list(corr.columns)

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=symbols, y=symbols,
        colorscale=[
            [0.0, "#d50000"], [0.35, "#ff5252"], [0.5, "#1a1f2e"],
            [0.65, "#69f0ae"], [1.0, "#00e676"]
        ],
        zmin=-1, zmax=1,
        text=[[f"{corr.iloc[i, j]:.2f}" for j in range(len(symbols))] for i in range(len(symbols))],
        texttemplate="%{text}",
        textfont=dict(size=12, color="white"),
        hoverongaps=False,
    ))
    fig.update_layout(
        title="Pairwise Correlation Matrix (Daily Returns)",
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        font=dict(color="#e8eaf0"),
        margin=dict(t=50, b=50, l=80, r=20),
    )
    return fig


def _chart_radar(scores: Dict[str, AssetScore]) -> Optional[go.Figure]:
    """Radar/spider chart for multi-dimensional score comparison."""
    valid = {k: v for k, v in scores.items() if v.error is None}
    if not valid:
        return None

    categories = ["Quant", "Fundamentals", "Macro", "Technical", "Momentum", "Overall"]
    colors = px.colors.qualitative.Set2

    # Use a fixed hex palette so rgba conversion is always safe
    HEX_PALETTE = [
        "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
        "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
    ]

    fig = go.Figure()
    for i, (sym, s) in enumerate(valid.items()):
        # Derive momentum score from price action
        momentum_score = float(np.clip(50 + (s.change_1m * 3), 0, 100))
        values = [
            s.quant_score, s.fundamental_score, s.macro_score,
            s.technical_score, momentum_score, s.overall_score
        ]
        hex_color = HEX_PALETTE[i % len(HEX_PALETTE)]
        h = hex_color.lstrip("#")
        r_val, g_val, b_val = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            name=sym,
            fill="toself",
            fillcolor=f"rgba({r_val},{g_val},{b_val},0.15)",
            line=dict(color=hex_color, width=2),
        ))

    fig.update_layout(
        polar=dict(
            bgcolor="#0f1117",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.1)",
                            tickfont=dict(color="#8892b0", size=9)),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)", tickfont=dict(color="#e8eaf0")),
        ),
        title="Multi-Dimensional Score Radar",
        paper_bgcolor="#0f1117",
        font=dict(color="#e8eaf0"),
        showlegend=True,
        legend=dict(orientation="h", y=-0.15, bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=60, b=60),
    )
    return fig


def _chart_ranking_bar(result: ComparativeResult) -> Optional[go.Figure]:
    """Horizontal bar chart ranking all assets by overall score."""
    valid = {k: v for k, v in result.assets.items() if v.error is None}
    if not valid:
        return None

    ranked = sorted(valid.values(), key=lambda x: x.overall_score)
    colors = [SIGNAL_COLORS.get(s.decision, "#ffd740") for s in ranked]

    fig = go.Figure(go.Bar(
        y=[s.symbol for s in ranked],
        x=[s.overall_score for s in ranked],
        orientation="h",
        marker=dict(color=colors, opacity=0.85),
        text=[f"{s.overall_score:.1f}" for s in ranked],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}<extra></extra>",
    ))
    fig.add_vline(x=50, line_dash="dash", line_color="rgba(255,255,255,0.2)")
    fig.update_layout(
        title="Overall Composite Score Ranking",
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        font=dict(color="#e8eaf0"),
        xaxis=dict(range=[0, 110], gridcolor="rgba(255,255,255,0.05)", title="Score / 100"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.0)"),
        margin=dict(t=50, b=40, l=80, r=40),
    )
    return fig


def _chart_rolling_performance(result: ComparativeResult, assets_input: List[Dict]) -> Optional[go.Figure]:
    """30/60/90D rolling return comparison."""
    engine = get_comparative_engine()
    data_rows = []

    for ainfo in assets_input:
        sym = ainfo["symbol"]
        atype = ainfo.get("asset_type") or detect_asset_type(sym)
        try:
            df = engine._fetch_price_data(sym, atype, period="1y")
            if df is None or df.empty:
                continue
            close = df["Close"].astype(float)
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.dropna()

            def safe_ret(lb):
                if len(close) >= lb + 1:
                    return round((close.iloc[-1] / close.iloc[-lb] - 1) * 100, 2)
                return None

            data_rows.append({
                "Symbol": sym,
                "30D (%)": safe_ret(21),
                "60D (%)": safe_ret(42),
                "90D (%)": safe_ret(63),
            })
        except Exception:
            pass

    if not data_rows:
        return None

    df_plot = pd.DataFrame(data_rows).set_index("Symbol")
    fig = go.Figure()
    periods = ["30D (%)", "60D (%)", "90D (%)"]
    colors_set = ["#4fc3f7", "#ce93d8", "#a5d6a7"]

    for col, color in zip(periods, colors_set):
        values = df_plot[col].tolist()
        fig.add_trace(go.Bar(
            name=col.replace(" (%)", ""),
            x=df_plot.index.tolist(),
            y=values,
            marker=dict(color=color, opacity=0.8),
            text=[f"{v:+.1f}%" if v is not None else "N/A" for v in values],
            textposition="outside",
        ))

    fig.update_layout(
        barmode="group",
        title="Rolling Return Comparison (30D / 60D / 90D)",
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        font=dict(color="#e8eaf0"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=True,
                   zerolinecolor="rgba(255,255,255,0.2)", title="Return (%)"),
        legend=dict(orientation="h", y=-0.15, bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=50, b=60, l=60, r=20),
    )
    return fig


def _chart_options_greeks(scores: Dict[str, AssetScore]) -> Optional[go.Figure]:
    """Options Greeks comparison table for options assets."""
    options_assets = {k: v for k, v in scores.items() if v.asset_type == "options" and v.error is None}
    if not options_assets:
        return None

    headers = ["Symbol", "Price", "IV (%)", "Delta", "Theta"]
    rows = []
    for sym, s in options_assets.items():
        rows.append([
            sym,
            f"${s.current_price:.2f}",
            f"{s.iv * 100:.1f}%" if s.iv else "N/A",
            f"{s.delta:.3f}" if s.delta else "N/A",
            f"{s.theta:.4f}" if s.theta else "N/A",
        ])

    fig = go.Figure(data=go.Table(
        header=dict(values=headers, fill_color="#1a1f2e",
                    font=dict(color="#e8eaf0", size=12), align="left",
                    line_color="rgba(255,255,255,0.1)"),
        cells=dict(values=list(zip(*rows)) if rows else [[] for _ in headers],
                   fill_color="#0f1117",
                   font=dict(color="#e8eaf0", size=11), align="left",
                   line_color="rgba(255,255,255,0.05)"),
    ))
    fig.update_layout(
        title="Options Greeks Comparison",
        paper_bgcolor="#0f1117", margin=dict(t=50, b=20),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDE-BY-SIDE ASSET CARDS
# ─────────────────────────────────────────────────────────────────────────────

def _render_asset_cards(result: ComparativeResult):
    """Render side-by-side asset metric cards."""
    valid = {k: v for k, v in result.assets.items() if v.error is None}
    errors = {k: v for k, v in result.assets.items() if v.error}

    if errors:
        for sym, s in errors.items():
            st.warning(f" **{sym}**: {s.error}")

    if not valid:
        return

    # Max 4 columns, then wrap
    chunk_size = 4
    asset_list = list(valid.items())
    chunks = [asset_list[i:i+chunk_size] for i in range(0, len(asset_list), chunk_size)]

    for chunk in chunks:
        cols = st.columns(len(chunk))
        for i, (sym, s) in enumerate(chunk):
            with cols[i]:
                icon = ASSET_ICONS.get(s.asset_type, "")
                atype_color = ASSET_TYPE_COLORS.get(s.asset_type, "#e8eaf0")

                # Header
                st.markdown(
                    f"""<div style='text-align:center; padding:10px 0 4px;'>
                        <span style='font-size:1.8rem;'>{icon}</span><br>
                        <span style='font-size:1.1rem; font-weight:700; color:#e8eaf0;'>{sym}</span><br>
                        <span style='font-size:0.75rem; color:{atype_color}; text-transform:uppercase;
                                     letter-spacing:0.08em;'>{s.asset_type}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )
                st.markdown("<hr style='border-color:rgba(255,255,255,0.08); margin:8px 0;'>", unsafe_allow_html=True)

                # Signal badge
                sig_color = SIGNAL_COLORS.get(s.decision, "#ffd740")
                st.markdown(
                    f"""<div style='text-align:center; margin-bottom:10px;'>
                        <span style='background:{sig_color}22; border:1.5px solid {sig_color};
                                     border-radius:20px; padding:4px 16px; color:{sig_color};
                                     font-weight:700; font-size:0.8rem;'>{s.decision}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

                # Metrics
                price_fmt = f"${s.current_price:,.4f}" if s.current_price < 1 else f"${s.current_price:,.2f}"
                _styled_metric("Price", price_fmt)

                score_color = "#00e676" if s.overall_score >= 65 else "#d50000" if s.overall_score <= 35 else "#ffd740"
                _styled_metric("Overall Score", f"{s.overall_score:.1f} / 100", color=score_color)

                pct_color = "#00e676" if s.change_1m >= 0 else "#ff5252"
                _styled_metric("30D Return", f"{s.change_1m:+.2f}%", color=pct_color)

                _styled_metric("Annual Vol", f"{s.volatility_annual*100:.1f}%")

                _styled_metric("RSI", f"{s.rsi:.1f}",
                               sub="Oversold" if s.rsi < 30 else "Overbought" if s.rsi > 70 else "Neutral")

                macd_color = "#00e676" if s.macd_signal == "BULLISH" else "#ff5252" if s.macd_signal == "BEARISH" else "#ffd740"
                _styled_metric("MACD", s.macd_signal, color=macd_color)

                if s.overall_score != 0:
                    conviction_pct = int(s.conviction * 100)
                    st.markdown(
                        f"""<div style='margin-bottom:8px;'>
                            <div style='font-size:0.72rem; color:#8892b0; text-transform:uppercase;
                                        letter-spacing:0.06em; margin-bottom:4px;'>Conviction</div>
                            <div style='background:rgba(255,255,255,0.05); border-radius:4px; height:6px;'>
                                <div style='width:{conviction_pct}%; background:{sig_color};
                                            border-radius:4px; height:6px;'></div>
                            </div>
                            <div style='font-size:0.72rem; color:#8892b0; margin-top:2px;'>{conviction_pct}%</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY CARDS
# ─────────────────────────────────────────────────────────────────────────────

def _render_strategy_card(strat: CrossAssetStrategy, idx: int):
    """Render an adjustable strategy suggestion card."""
    exp_color = "#00e676" if strat.expected_return >= 0 else "#ff5252"
    risk_color = {"LOW": "#69f0ae", "MEDIUM": "#ffd740", "HIGH": "#ff5252"}.get(strat.risk_level, "#ffd740")

    with st.expander(strat.name, expanded=idx == 0):
        col_left, col_right = st.columns([3, 1])
        with col_left:
            st.markdown(f"**Strategy Type:** {strat.type.replace('_', ' ').title()}")
            st.markdown(f"**Time Horizon:** {strat.time_horizon}")

            if strat.assets_long:
                st.markdown(f" **Long:** {', '.join(f'`{s}`' for s in strat.assets_long)}")
            if strat.assets_short:
                st.markdown(f" **Short:** {', '.join(f'`{s}`' for s in strat.assets_short)}")

        with col_right:
            st.markdown(
                f"""<div style='text-align:right;'>
                    <div style='color:{exp_color}; font-size:1.4rem; font-weight:700;'>{strat.expected_return:+.2f}%</div>
                    <div style='font-size:0.72rem; color:#8892b0;'>Expected 30D</div>
                    <div style='color:{risk_color}; margin-top:8px; font-weight:600;'>{strat.risk_level} RISK</div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("**Rule-Based Rationale:**")
        st.markdown(strat.rationale)

        if strat.ai_thesis:
            st.markdown("**AI Thesis (LM Studio):**")
            st.info(strat.ai_thesis)

        # Adjustable Parameters
        if strat.adjustable_params:
            st.markdown("**Adjustable Parameters:**")
            params_cols = st.columns(min(len(strat.adjustable_params), 3))
            for j, (param, val) in enumerate(strat.adjustable_params.items()):
                with params_cols[j % len(params_cols)]:
                    if isinstance(val, float) and 0 <= val <= 1:
                        new_val = st.slider(
                            param.replace("_", " ").title(),
                            0.0, 1.0, val, 0.05,
                            key=f"strat_{idx}_{param}"
                        )
                    elif isinstance(val, float):
                        new_val = st.number_input(
                            param.replace("_", " ").title(),
                            value=val, step=0.5,
                            key=f"strat_{idx}_{param}"
                        )
                    elif isinstance(val, dict):
                        st.json(val)
                    else:
                        st.text_input(param.replace("_", " ").title(),
                                      value=str(val), key=f"strat_{idx}_{param}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RENDER FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def render_comparative_analysis():
    st.markdown(
        """
        <div style='padding:20px 0 10px;'>
            <h1 style='color:#e8eaf0; font-size:2rem; font-weight:800; margin:0;'>
                 Comparative Analysis Hub
            </h1>
            <p style='color:#8892b0; font-size:0.95rem; margin-top:6px;'>
                Institutional cross-asset intelligence — equities, crypto, commodities, forex, options. Unlimited assets.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── SECTION 1: Asset Entry ────────────────────────────────────────────
    st.markdown("---")
    _section_header("1. Select Assets", "")

    col_symbols, col_add = st.columns([5, 1])
    with col_symbols:
        symbols_raw = st.text_input(
            "Enter symbols (comma-separated)",
            value="AAPL, MSFT, BTC-USD, GC=F, EURUSD=X",
            placeholder="e.g. AAPL, BTC-USD, GC=F, TSLA, ETH-USD, CL=F",
            help="Supports stocks, crypto (-USD), futures (=F), forex (=X), and options.",
        )

    raw_symbols = [s.strip().upper() for s in symbols_raw.split(",") if s.strip()]

    # Per-symbol type override
    if raw_symbols:
        st.markdown("**Override asset type (optional):**")
        type_cols = st.columns(min(len(raw_symbols), 5))
        asset_configs = []
        for i, sym in enumerate(raw_symbols):
            auto = detect_asset_type(sym)
            with type_cols[i % len(type_cols)]:
                override = st.selectbox(
                    sym,
                    options=["stock", "crypto", "commodity", "forex", "futures", "options"],
                    index=["stock", "crypto", "commodity", "forex", "futures", "options"].index(auto)
                    if auto in ["stock", "crypto", "commodity", "forex", "futures", "options"] else 0,
                    key=f"atype_{sym}_{i}",
                )
                asset_configs.append({"symbol": sym, "asset_type": override})

    # ── SECTION 2: Model Tailoring ────────────────────────────────────────
    _section_header("2. Tailor Analysis Models", "")

    preset = st.selectbox(
        "Analysis Mode Preset",
        list(MODEL_PRESETS.keys()),
        index=0,
        help="Choose a preset or 'Custom Mix' to hand-pick engines.",
    )

    if preset == "Custom Mix":
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            use_quant = st.checkbox(" Quant Ensemble (ML)", value=True)
        with c2:
            use_fund = st.checkbox(" Fundamentals (DCF/PE)", value=False,
                                   help="Only applicable to stocks. Ignored for crypto/forex.")
        with c3:
            use_macro = st.checkbox(" Macro Regime", value=False,
                                    help="Applies macro environment weighting (slower).")
        with c4:
            use_tech = st.checkbox(" Technical Analysis", value=True)
        model_config = {
            "use_quant": use_quant,
            "use_fundamentals": use_fund,
            "use_macro": use_macro,
            "use_technical": use_tech,
        }
    else:
        model_config = MODEL_PRESETS[preset]
        active = [k.replace("use_", "").title() for k, v in model_config.items() if v]
        st.info(f"Active engines: **{', '.join(active)}**")

    # ── SECTION 3: Analysis Period ────────────────────────────────────────
    _section_header("3. Options", "")
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        show_ai_thesis = st.checkbox(" Generate AI Strategy Theses (requires LM Studio)", value=True)
    with col_opt2:
        include_correlations = st.checkbox(" Include Correlation Analysis", value=True)

    # ── RUN BUTTON ────────────────────────────────────────────────────────
    st.markdown("---")
    run_btn = st.button(" Run Comparative Analysis", type="primary", use_container_width=True)

    if run_btn:
        if not raw_symbols:
            st.warning("Please enter at least one symbol.")
            return

        if len(raw_symbols) < 2:
            st.info(" For cross-asset comparisons, add at least 2 symbols. Running single-asset analysis...")

        # Force AI thesis flag into model config
        active_config = dict(model_config)
        if not show_ai_thesis:
            # We'll skip AI thesis — pass a flag via a hack
            active_config["_skip_ai_thesis"] = True

        with st.spinner(f"Running analysis across {len(raw_symbols)} asset{'s' if len(raw_symbols) > 1 else ''}..."):
            engine = get_comparative_engine()
            result = engine.run_analysis(asset_configs, active_config)

        # ── RESULTS ───────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown(
            f"""<div style='text-align:right; color:#6b7280; font-size:0.8rem;'>
                Analysis completed in {result.run_time_ms:,} ms
            </div>""",
            unsafe_allow_html=True,
        )

        valid_count = len([v for v in result.assets.values() if v.error is None])
        if valid_count == 0:
            # Per-asset breakdown instead of a single catch-all error
            st.error("Could not retrieve data for any of the selected assets.")
            for sym, v in result.assets.items():
                err = v.error if v.error else "No market data available"
                st.warning(f"**{sym}:** {err} — check the symbol format (e.g. CL=F for futures, "
                           f"BTC-USD for crypto, EURUSD=X for forex) and try again.")
            st.caption("Data providers (e.g. yfinance) can be rate-limited — wait a few seconds and re-run.")
            return
        elif valid_count < len(result.assets):
            # Partial success: flag the failures individually, keep the wins
            for sym, v in result.assets.items():
                if v.error:
                    st.warning(f"**{sym}** could not be analyzed: {v.error}")
            st.caption("Showing results for the assets that loaded successfully.")

        # ── RANKING SUMMARY ───────────────────────────────────────────────
        _section_header("Overall Rankings", "")
        col_rank1, col_rank2 = st.columns(2)
        with col_rank1:
            st.markdown("**By Composite Score:**")
            for rank, sym in enumerate(result.ranked_by_score, 1):
                s = result.assets[sym]
                sig_color = SIGNAL_COLORS.get(s.decision, "#ffd740")
                st.markdown(
                    f"<span style='color:#8892b0;'>#{rank}</span> "
                    f"<b style='color:#e8eaf0;'>{sym}</b> — "
                    f"<span style='color:{sig_color};'>{s.overall_score:.1f}/100 · {s.decision}</span>",
                    unsafe_allow_html=True,
                )
        with col_rank2:
            st.markdown("**By 30-Day Momentum:**")
            for rank, sym in enumerate(result.ranked_by_momentum, 1):
                s = result.assets[sym]
                chg_color = "#00e676" if s.change_1m >= 0 else "#ff5252"
                st.markdown(
                    f"<span style='color:#8892b0;'>#{rank}</span> "
                    f"<b style='color:#e8eaf0;'>{sym}</b> — "
                    f"<span style='color:{chg_color};'>{s.change_1m:+.2f}%</span>",
                    unsafe_allow_html=True,
                )

        # ── SIDE-BY-SIDE CARDS ────────────────────────────────────────────
        _section_header("Side-by-Side Comparison", "")
        _render_asset_cards(result)

        # ── CHARTS ────────────────────────────────────────────────────────
        tab_labels = [" Price Overlay", " Radar Chart", " Score Ranking",
                      " Rolling Returns", " Correlation"]
        if any(v.asset_type == "options" and v.error is None for v in result.assets.values()):
            tab_labels.append(" Options Greeks")

        tabs = st.tabs(tab_labels)

        with tabs[0]:
            fig_overlay = _chart_price_overlay(result, asset_configs)
            if fig_overlay:
                st.plotly_chart(fig_overlay, use_container_width=True)
            else:
                st.info("Price data not available for overlay chart.")

        with tabs[1]:
            fig_radar = _chart_radar(result.assets)
            if fig_radar:
                st.plotly_chart(fig_radar, use_container_width=True)

        with tabs[2]:
            fig_bar = _chart_ranking_bar(result)
            if fig_bar:
                st.plotly_chart(fig_bar, use_container_width=True)

        with tabs[3]:
            fig_rolling = _chart_rolling_performance(result, asset_configs)
            if fig_rolling:
                st.plotly_chart(fig_rolling, use_container_width=True)

        with tabs[4]:
            if include_correlations and not result.correlation_matrix.empty:
                fig_heatmap = _chart_correlation_heatmap(result)
                if fig_heatmap:
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                    # Correlation interpretation
                    corr = result.correlation_matrix
                    symbols = list(corr.columns)
                    st.markdown("**Interpretation:**")
                    for i in range(len(symbols)):
                        for j in range(i+1, len(symbols)):
                            val = corr.iloc[i, j]
                            strength = "highly correlated" if abs(val) > 0.7 else "moderately correlated" if abs(val) > 0.4 else "weakly correlated"
                            direction = "positively" if val > 0 else "negatively"
                            st.markdown(f"- `{symbols[i]}` ↔ `{symbols[j]}`: **{val:.2f}** — {direction} {strength}")
            else:
                st.info("Correlation analysis requires at least 2 assets with price data.")

        if len(tab_labels) > 5:
            with tabs[5]:
                fig_greeks = _chart_options_greeks(result.assets)
                if fig_greeks:
                    st.plotly_chart(fig_greeks, use_container_width=True)

        # ── INDIVIDUAL DETAILED CONCLUSIONS ──────────────────────────────
        _section_header("Individual Asset Deep Dives", "")
        for sym in result.ranked_by_score:
            s = result.assets[sym]
            if s.error:
                continue
            sig_color = SIGNAL_COLORS.get(s.decision, "#ffd740")
            with st.expander(f"{ASSET_ICONS.get(s.asset_type, '')} {sym} — {s.decision} ({s.overall_score:.1f}/100)", expanded=False):
                st.markdown(s.individual_conclusion)
                if s.model_breakdown:
                    st.markdown("**Model Breakdown:**")
                    mb = {k: v for k, v in s.model_breakdown.items() if k not in ("ml_detail",) and isinstance(v, (int, float))}
                    if mb:
                        fig_mb = go.Figure(go.Bar(
                            x=list(mb.keys()), y=list(mb.values()),
                            marker=dict(color=[sig_color] * len(mb), opacity=0.75),
                            text=[f"{v:.1f}" for v in mb.values()],
                            textposition="outside",
                        ))
                        fig_mb.update_layout(
                            paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                            font=dict(color="#e8eaf0", size=11),
                            yaxis=dict(range=[0, 110], gridcolor="rgba(255,255,255,0.05)"),
                            margin=dict(t=10, b=10, l=10, r=10),
                            height=200,
                        )
                        st.plotly_chart(fig_mb, use_container_width=True, key=f"model_breakdown_{sym}")

        # ── HEAD-TO-HEAD COMPARISONS ──────────────────────────────────────
        if result.comparative_conclusions:
            _section_header("Head-to-Head Comparative Analysis", "")
            for pair_label, conclusion in result.comparative_conclusions.items():
                st.markdown(f"**{pair_label}**")
                st.markdown(conclusion)
                st.markdown("---")

        # ── STRATEGY SUGGESTIONS ──────────────────────────────────────────
        _section_header("Cross-Asset Strategy Suggestions", "")
        if result.strategies:
            st.markdown(
                f"*{len(result.strategies)} strategies generated based on composite scores, "
                f"correlation data, and momentum analysis. All parameters are adjustable.*"
            )
            for i, strat in enumerate(result.strategies):
                _render_strategy_card(strat, i)
        else:
            st.info("No cross-asset strategies generated. Add more varied assets (e.g., mix stocks + crypto) for pair trade and basket suggestions.")

        # ── EXPORT ────────────────────────────────────────────────────────
        _section_header("Export", "")
        export_col1, export_col2 = st.columns(2)
        with export_col1:
            # JSON export
            export_data = {
                "generated_at": pd.Timestamp.now().isoformat(),
                "model_config": result.model_config,
                "assets": {
                    sym: {
                        "price": s.current_price,
                        "decision": s.decision,
                        "overall_score": s.overall_score,
                        "change_1d": s.change_1d,
                        "change_1w": s.change_1w,
                        "change_1m": s.change_1m,
                        "volatility": s.volatility_annual,
                        "conviction": s.conviction,
                        "expected_return_30d": s.expected_return_30d,
                    }
                    for sym, s in result.assets.items() if s.error is None
                },
                "ranked_by_score": result.ranked_by_score,
                "strategies": [
                    {"name": st_obj.name, "type": st_obj.type,
                     "assets_long": st_obj.assets_long, "assets_short": st_obj.assets_short,
                     "expected_return": st_obj.expected_return, "risk_level": st_obj.risk_level,
                     "rationale": st_obj.rationale, "ai_thesis": st_obj.ai_thesis}
                    for st_obj in result.strategies
                ],
                "comparative_conclusions": result.comparative_conclusions,
            }
            st.download_button(
                "Export JSON Report",
                data=json.dumps(export_data, indent=2),
                file_name=f"octavian_comparative_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                use_container_width=True,
            )

        with export_col2:
            # CSV export
            csv_rows = []
            for sym, s in result.assets.items():
                if s.error is None:
                    csv_rows.append({
                        "Symbol": sym, "Asset Type": s.asset_type,
                        "Price": s.current_price, "Signal": s.decision,
                        "Overall Score": s.overall_score, "Conviction": f"{s.conviction*100:.0f}%",
                        "1D Chg (%)": s.change_1d, "1W Chg (%)": s.change_1w,
                        "1M Chg (%)": s.change_1m, "Annual Vol (%)": f"{s.volatility_annual*100:.1f}",
                        "Exp. 30D Return (%)": s.expected_return_30d,
                        "RSI": s.rsi, "MACD": s.macd_signal,
                    })
            if csv_rows:
                csv_str = pd.DataFrame(csv_rows).to_csv(index=False)
                st.download_button(
                    "Export CSV Summary",
                    data=csv_str,
                    file_name=f"octavian_comparative_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
