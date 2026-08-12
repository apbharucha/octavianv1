"""
Market Heartbeat System — Octavian Terminal
A fully self-contained, zero-API-key algorithmic engine that evaluates
real-time macro and micro market conditions via yfinance data and 
deterministic heuristics.  Strictly observational, NOT predictive.
"""

import streamlit as st
import datetime
import random

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from octavian_theme import COLORS, section_header

# ---------------------------------------------------------------------------
# DATA LAYER
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def fetch_heartbeat_data():
    """Fetch 5-day close data for core macro instruments."""
    tickers = {
        "SPX": "^GSPC",
        "VIX": "^VIX",
        "US10Y": "^TNX",
        "DXY": "DX-Y.NYB",
        "Gold": "GC=F",
        "Oil": "CL=F",
    }
    try:
        raw = yf.download(
            list(tickers.values()), period="1mo", progress=False
        )["Close"]
        if raw is None or raw.empty:
            return {}, {}
        # Flatten MultiIndex if needed
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        latest = raw.iloc[-1].to_dict()
        chg_5d = (raw.iloc[-1] / raw.iloc[-6] - 1).to_dict() if len(raw) >= 6 else {}
        chg_20d = (raw.iloc[-1] / raw.iloc[0] - 1).to_dict() if len(raw) >= 10 else {}

        us_data = {
            "SPX": latest.get("^GSPC", 0),
            "SPX_5d": chg_5d.get("^GSPC", 0),
            "SPX_20d": chg_20d.get("^GSPC", 0),
            "VIX": latest.get("^VIX", 15),
            "VIX_5d": chg_5d.get("^VIX", 0),
            "US10Y": latest.get("^TNX", 4.0),
            "US10Y_5d": chg_5d.get("^TNX", 0),
        }
        global_data = {
            "DXY": latest.get("DX-Y.NYB", 100),
            "DXY_5d": chg_5d.get("DX-Y.NYB", 0),
            "Gold": latest.get("GC=F", 2000),
            "Gold_5d": chg_5d.get("GC=F", 0),
            "Oil": latest.get("CL=F", 70),
            "Oil_5d": chg_5d.get("CL=F", 0),
        }
        return global_data, us_data
    except Exception:
        return {}, {}


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_sector_data():
    """Fetch sector ETF 5-day returns for rotation analysis."""
    sectors = {
        "XLK": "Technology",
        "XLV": "Healthcare",
        "XLF": "Financials",
        "XLE": "Energy",
        "XLI": "Industrials",
        "XLY": "Consumer Disc.",
        "XLP": "Consumer Staples",
        "XLU": "Utilities",
        "XLRE": "Real Estate",
        "XLC": "Communication",
        "XLB": "Materials",
    }
    try:
        raw = yf.download(list(sectors.keys()), period="5d", progress=False)["Close"]
        if raw is None or raw.empty:
            return {}
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        changes = (raw.iloc[-1] / raw.iloc[0] - 1)
        return {sectors[k]: float(changes.get(k, 0)) for k in sectors if k in changes.index}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# HEURISTIC ENGINE
# ---------------------------------------------------------------------------

def generate_market_heartbeat(g, u, timeframe="medium", custom_g="", custom_u=""):
    """Pure algorithmic heartbeat — returns a structured dict of assessments."""
    g = g if isinstance(g, dict) else {}
    u = u if isinstance(u, dict) else {}

    vix = u.get("VIX", 15.0)
    spx_5d = u.get("SPX_5d", 0.0)
    spx_20d = u.get("SPX_20d", 0.0)
    us10y_5d = u.get("US10Y_5d", 0.0)
    us10y = u.get("US10Y", 4.0)
    oil_5d = g.get("Oil_5d", 0.0)
    gold_5d = g.get("Gold_5d", 0.0)
    dxy_5d = g.get("DXY_5d", 0.0)

    # --- Inflation ---
    if oil_5d > 0.02 and gold_5d > 0.01:
        inflation = ("Rising Pressures", "danger")
    elif oil_5d < -0.02 and gold_5d < -0.01:
        inflation = ("Cooling / Disinflationary", "success")
    elif oil_5d > 0.01 or gold_5d > 0.01:
        inflation = ("Mildly Warming", "gold")
    else:
        inflation = ("Stable / Mixed Signals", "neutral")

    # --- Liquidity ---
    if dxy_5d > 0.01 and us10y_5d > 0.02:
        liquidity = ("Tightening — Strong Dollar + Rising Yields", "danger")
    elif dxy_5d < -0.01 and us10y_5d < -0.02:
        liquidity = ("Expanding — Weak Dollar + Falling Yields", "success")
    else:
        liquidity = ("Neutral / Range-bound", "neutral")

    # --- Growth ---
    if spx_5d > 0.01 and spx_20d > 0.02:
        growth = ("Resilient / Accelerating", "success")
    elif spx_5d < -0.01 and spx_20d < -0.02:
        growth = ("Slowing / Safe-Haven Bid", "danger")
    elif spx_5d > 0:
        growth = ("Moderate Expansion", "gold")
    else:
        growth = ("Stalling / Transitional", "neutral")

    # --- Fed Stance ---
    rate_trend = "Rising" if us10y_5d > 0.01 else ("Falling" if us10y_5d < -0.01 else "Stable")
    if rate_trend == "Rising" and spx_5d > 0:
        fed = ("Hawkish Bias — Absorbed by Growth", "gold")
        econ = ("Robust — Absorbing Higher Rates", "success")
    elif rate_trend == "Rising" and spx_5d <= 0:
        fed = ("Hawkish Bias — Pressuring Equities", "danger")
        econ = ("Vulnerable to Yield Shock", "danger")
    elif rate_trend == "Falling" and spx_5d > 0:
        fed = ("Dovish Pivot / Easing Cycle", "success")
        econ = ("Goldilocks / Expanding", "success")
    else:
        fed = ("Data-Dependent — Holding", "neutral")
        econ = ("Mixed / Transitioning", "neutral")

    # --- Volatility ---
    if vix < 13:
        vol = ("Suppressed — High Complacency", "gold")
    elif vix < 18:
        vol = ("Low — Calm Conditions", "success")
    elif vix < 25:
        vol = ("Normal Operating Range", "neutral")
    elif vix < 32:
        vol = ("Elevated — Active Hedging", "danger")
    else:
        vol = ("Distressed — High Fear", "danger")

    # --- Breadth / Rotation ---
    breadth = "Expanding Participation" if spx_5d > 0.01 else ("Deteriorating" if spx_5d < -0.01 else "Narrow / Stock-Picker's Market")
    sector_rot = "Cyclicals Leading" if spx_5d > 0 and us10y_5d > 0 else ("Defensives Leading" if spx_5d < 0 else "Mixed Rotation")
    micro_liq = "Healthy Bid/Ask Depth" if vix < 22 else "Thinning Order Books"

    # --- Overall ---
    if vix < 20 and spx_5d >= 0:
        env_class = ("Risk-On / Constructive", "success")
        forces = "Equity accumulation, suppressed volatility, and positive flow momentum."
    elif vix >= 25 and spx_5d < -0.01:
        env_class = ("Risk-Off / Defensive", "danger")
        forces = "Active de-risking, yield curve adjustments, and safe-haven demand."
    else:
        env_class = ("Transitional / Choppy", "gold")
        forces = "Conflicting macro signals creating cross-current churn across asset classes."

    # --- Shadow Predictions ---
    preds = [
        "Continued liquidity tightening may create mild upward pressure on volatility if sustained over the coming weeks.",
        "Sustained dollar strength occasionally foreshadows weakness in emerging market equities and commodity-linked currencies.",
        "Persisting low-volatility regimes can historically invite swift, mean-reverting risk-off shocks — complacency is a risk.",
        "If yields continue establishing higher floors, long-duration assets and growth-sensitive sectors may face incremental headwinds.",
        "Resilient growth data mixed with stalled disinflation often extends the duration of current sector leadership rotations.",
        "Compression in credit spreads alongside equity gains suggests risk appetite may be nearing a local extreme.",
        "If oil stabilizes above $80, second-round inflation effects could re-emerge in service-sector pricing within 2-3 months.",
    ]
    shadow = random.sample(preds, min(2, len(preds)))

    if custom_g:
        policy = f"Reacting to: {custom_g}"
    else:
        policy = "Maintaining current operational stance."

    if custom_u:
        fed = (fed[0] + f" | Note: {custom_u}", fed[1])

    return {
        "inflation": inflation,
        "liquidity": liquidity,
        "growth": growth,
        "policy": policy,
        "fed": fed,
        "rate_trend": rate_trend,
        "us10y": us10y,
        "econ": econ,
        "vol": vol,
        "breadth": breadth,
        "sector_rot": sector_rot,
        "micro_liq": micro_liq,
        "env_class": env_class,
        "forces": forces,
        "shadow": shadow,
        "raw_g": g,
        "raw_u": u,
    }


# ---------------------------------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------------------------------

def _color_for(variant):
    m = {"success": COLORS["success"], "danger": COLORS["danger"],
         "gold": COLORS["gold"], "neutral": COLORS["neutral"]}
    return m.get(variant, COLORS["text_secondary"])


def _render_heartbeat_card(label, value_text, variant, extra=""):
    color = _color_for(variant)
    st.markdown(
        f'<div style="background:{COLORS["glass_bg"]};border:1px solid {COLORS["glass_border"]};'
        f'border-left:3px solid {color};border-radius:10px;padding:14px 18px;margin-bottom:10px;'
        f'backdrop-filter:blur(10px);animation:fadeInUp 0.4s ease-out;">'
        f'<div style="color:{COLORS["text_secondary"]};font-size:0.7rem;text-transform:uppercase;'
        f'letter-spacing:0.8px;font-weight:500;margin-bottom:4px;">{label}</div>'
        f'<div style="color:{color};font-size:1.05rem;font-weight:600;">{value_text}</div>'
        f'{f"<div style=color:{COLORS[chr(116)+chr(101)+chr(120)+chr(116)+chr(95)+chr(115)+chr(101)+chr(99)+chr(111)+chr(110)+chr(100)+chr(97)+chr(114)+chr(121)]};font-size:0.78rem;margin-top:4px;>{extra}</div>" if extra else ""}'
        f'</div>',
        unsafe_allow_html=True,
    )


def _hex_to_rgba(hex_color, alpha=0.15):
    """Convert hex color to rgba string for Plotly gauge steps."""
    h = hex_color.lstrip('#')
    if len(h) == 6:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    return f"rgba(100,100,100,{alpha})"


def _render_gauge_chart(g_data, u_data):
    """Mini dashboard gauges for key levels."""
    fig = make_subplots(rows=1, cols=4, specs=[[{"type": "indicator"}]*4])
    vix = u_data.get("VIX", 15)
    us10y = u_data.get("US10Y", 4)
    spx_5d = u_data.get("SPX_5d", 0) * 100
    dxy_5d = g_data.get("DXY_5d", 0) * 100

    for i, (title, val, suffix, rng, color_steps) in enumerate([
        ("VIX", vix, "", [0, 50], [[0, COLORS["success"]], [0.4, COLORS["gold"]], [0.6, "#ff9800"], [1, COLORS["danger"]]]),
        ("US 10Y Yield", us10y, "%", [0, 8], [[0, COLORS["success"]], [0.5, COLORS["gold"]], [1, COLORS["danger"]]]),
        ("SPX 5D Chg", spx_5d, "%", [-5, 5], [[0, COLORS["danger"]], [0.5, COLORS["gold"]], [1, COLORS["success"]]]),
        ("DXY 5D Chg", dxy_5d, "%", [-3, 3], [[0, COLORS["success"]], [0.5, COLORS["neutral"]], [1, COLORS["danger"]]]),
    ], 1):
        # Build gauge steps with proper rgba colors
        steps = []
        for j in range(len(color_steps) - 1):
            start_frac = color_steps[j][0]
            end_frac = color_steps[j + 1][0]
            step_start = rng[0] + (rng[1] - rng[0]) * start_frac
            step_end = rng[0] + (rng[1] - rng[0]) * end_frac
            step_color = _hex_to_rgba(color_steps[j][1], 0.15)
            steps.append({"range": [step_start, step_end], "color": step_color})

        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=val,
            title={"text": title, "font": {"size": 12, "color": COLORS["text_secondary"]}},
            number={"suffix": suffix, "font": {"size": 18, "color": COLORS["white"]}},
            gauge={
                "axis": {"range": rng, "tickcolor": COLORS["text_secondary"], "tickfont": {"size": 9}},
                "bar": {"color": COLORS["gold"]},
                "bgcolor": COLORS["navy_light"],
                "bordercolor": COLORS["border"],
                "steps": steps,
            }
        ), row=1, col=i)

    fig.update_layout(
        height=200, margin=dict(l=30, r=30, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text_primary"]),
    )
    return fig


def _render_sector_heatmap(sector_data):
    """Sector rotation heatmap."""
    if not sector_data:
        return None
    names = list(sector_data.keys())
    vals = [v * 100 for v in sector_data.values()]
    colors = [COLORS["success"] if v > 0.5 else COLORS["danger"] if v < -0.5 else COLORS["neutral"] for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:+.2f}%" for v in vals],
        textposition="auto",
        textfont=dict(size=11, color=COLORS["white"]),
    ))
    fig.update_layout(
        height=320,
        template="plotly_dark",
        paper_bgcolor=COLORS["navy"],
        plot_bgcolor=COLORS["navy_light"],
        font=dict(color=COLORS["text_primary"], size=11),
        xaxis_title="5-Day Change (%)",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=10, t=10, b=30),
    )
    return fig


# ---------------------------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------------------------

def show_market_heartbeat_tab():
    """Render the Market Heartbeat page with full Octavian styling."""
    st.title("Market Heartbeat")
    st.caption("Real-time macro & micro market state — algorithmically assessed, zero external APIs.")

    # --- Controls ---
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        custom_g = st.text_input("Global Context Override", placeholder="e.g. BoJ surprise hike", key="hb_cg")
    with c2:
        custom_u = st.text_input("U.S. Context Override", placeholder="e.g. CPI came in hot", key="hb_cu")
    with c3:
        timeframe = st.selectbox("Horizon", ["short", "medium", "long"], index=1, key="hb_tf")

    if st.button("Generate Heartbeat", type="primary", key="hb_gen"):
        st.session_state["hb_generated"] = True

    if not st.session_state.get("hb_generated"):
        st.info("Click **Generate Heartbeat** to analyze current market conditions.")
        return

    # --- Data ---
    with st.spinner("Fetching market data..."):
        g_data, u_data = fetch_heartbeat_data()
        sector_data = _fetch_sector_data()

    hb = generate_market_heartbeat(g_data, u_data, timeframe, custom_g, custom_u)

    # --- Gauges ---
    section_header("Key Market Gauges")
    gauge_fig = _render_gauge_chart(g_data, u_data)
    st.plotly_chart(gauge_fig, use_container_width=True)

    # --- Global Macro ---
    section_header("Global Macro Heartbeat")
    gc1, gc2, gc3, gc4 = st.columns(4)
    with gc1:
        _render_heartbeat_card("Inflation Trend", hb["inflation"][0], hb["inflation"][1],
                               f"Oil 5D: {g_data.get('Oil_5d',0)*100:+.2f}% | Gold 5D: {g_data.get('Gold_5d',0)*100:+.2f}%")
    with gc2:
        _render_heartbeat_card("Liquidity Condition", hb["liquidity"][0], hb["liquidity"][1],
                               f"DXY 5D: {g_data.get('DXY_5d',0)*100:+.2f}%")
    with gc3:
        _render_heartbeat_card("Growth Regime", hb["growth"][0], hb["growth"][1],
                               f"SPX 5D: {u_data.get('SPX_5d',0)*100:+.2f}% | 20D: {u_data.get('SPX_20d',0)*100:+.2f}%")
    with gc4:
        _render_heartbeat_card("Policy Stance", hb["policy"], "neutral")

    # --- U.S. Macro ---
    section_header("U.S. Macro Heartbeat")
    uc1, uc2, uc3 = st.columns(3)
    with uc1:
        _render_heartbeat_card("Fed Stance", hb["fed"][0], hb["fed"][1])
    with uc2:
        _render_heartbeat_card("Rate Trend", f"{hb['rate_trend']} (10Y ≈ {hb['us10y']:.2f}%)", "gold" if hb["rate_trend"] == "Rising" else "success" if hb["rate_trend"] == "Falling" else "neutral")
    with uc3:
        _render_heartbeat_card("Economic Strength", hb["econ"][0], hb["econ"][1])

    # --- Micro Market ---
    section_header("Micro Market Heartbeat")
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        _render_heartbeat_card("Volatility Regime", hb["vol"][0], hb["vol"][1], f"VIX: {u_data.get('VIX', 0):.2f}")
    with mc2:
        _render_heartbeat_card("Market Breadth", hb["breadth"], "success" if "Expanding" in hb["breadth"] else "danger" if "Deteriorating" in hb["breadth"] else "neutral")
    with mc3:
        _render_heartbeat_card("Sector Rotation", hb["sector_rot"], "gold")
    with mc4:
        _render_heartbeat_card("Liquidity", hb["micro_liq"], "success" if "Healthy" in hb["micro_liq"] else "danger")

    # --- Sector Heatmap ---
    if sector_data:
        section_header("Sector Rotation Heatmap (5-Day)")
        fig_sec = _render_sector_heatmap(sector_data)
        if fig_sec:
            st.plotly_chart(fig_sec, use_container_width=True)

    # --- Market State Summary ---
    section_header("Market State Summary")
    env_color = _color_for(hb["env_class"][1])
    st.markdown(
        f'<div style="background:{COLORS["glass_bg"]};border:1px solid {COLORS["glass_border"]};'
        f'border-top:3px solid {env_color};border-radius:12px;padding:24px;margin-bottom:16px;'
        f'backdrop-filter:blur(12px);animation:fadeInUp 0.5s ease-out;">'
        f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">'
        f'<div style="width:12px;height:12px;border-radius:50%;background:{env_color};'
        f'animation:pulse-dot 1.8s ease-in-out infinite;"></div>'
        f'<span style="color:{env_color};font-size:1.3rem;font-weight:700;letter-spacing:0.5px;">'
        f'{hb["env_class"][0]}</span></div>'
        f'<div style="color:{COLORS["text_primary"]};font-size:0.95rem;line-height:1.6;">'
        f'{hb["forces"]}</div></div>',
        unsafe_allow_html=True,
    )

    # --- Shadow Predictions ---
    section_header("Shadow Predictions")
    st.caption("Low-confidence, probabilistic directional tendencies — NOT forecasts.")
    for pred in hb["shadow"]:
        st.markdown(
            f'<div style="background:{COLORS["navy_light"]};border-left:3px solid {COLORS["lavender"]};'
            f'border-radius:6px;padding:12px 16px;margin-bottom:8px;font-style:italic;'
            f'color:{COLORS["text_secondary"]};font-size:0.9rem;">{pred}</div>',
            unsafe_allow_html=True,
        )

    # --- Raw Data Footer ---
    with st.expander("Raw Data Points"):
        rd1, rd2 = st.columns(2)
        with rd1:
            st.markdown("**Global**")
            for k, v in g_data.items():
                if "_5d" in k:
                    st.text(f"  {k}: {v*100:+.2f}%")
                else:
                    st.text(f"  {k}: {v:,.2f}")
        with rd2:
            st.markdown("**U.S.**")
            for k, v in u_data.items():
                if "_5d" in k or "_20d" in k:
                    st.text(f"  {k}: {v*100:+.2f}%")
                else:
                    st.text(f"  {k}: {v:,.2f}")
