"""
Octavian Dark Pool Intelligence Dashboard
==========================================

Institutional-grade off-exchange / dark-pool analytics terminal.

DESIGN PRINCIPLES
-----------------
* Data integrity over visual polish: every number is traceable to a source,
  carries an OBSERVED / MODELED / INFERENCE label and a confidence level.
* When the authoritative source (FINRA OTC Transparency) is unavailable, the
  dashboard shows a MODELED DATA banner and never presents estimates as facts.
* Three viewing modes (Basic / Advanced / Institutional) power the same data.

Author: Octavian Team
"""

from __future__ import annotations

import time as _time
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
from plotly.subplots import make_subplots

from octavian_theme import COLORS, apply_glass_card, section_header, status_badge

try:
    from dark_pool_engine import (
        get_dark_pool_engine,
        OBSERVED,
        MODELED,
        SCAN_DEFAULT_LIMIT,
        SCAN_MAX_LIMIT,
    )

    _HAS_ENGINE = True
except Exception as _e:  # pragma: no cover
    _HAS_ENGINE = False
    _ENGINE_ERR = str(_e)

DARK_GREEN = "#00ff88"
DARK_RED = "#ff4444"
AMBER = "#ffa500"
GOLD = COLORS["gold"]
LAV = COLORS["lavender"]

CACHE_TTL = 900  # 15 min for analytics payloads

# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #


def _mode_selector() -> str:
    mode = st.radio(
        "Viewing Mode",
        ["Basic", "Advanced", "Institutional"],
        horizontal=True,
        key="dp_mode",
        help="Basic: plain-language summary. Advanced: full statistics. "
             "Institutional: raw data, methodology, lineage and model outputs.",
    )
    return mode


def _provenance_line(mode: str, provenance: Optional[dict]) -> None:
    if not provenance:
        return
    cat = provenance.get("category", MODELED if not provenance.get("observed") else OBSERVED)
    src = provenance.get("source", "?")
    conf = provenance.get("confidence", 0)
    color = DARK_GREEN if cat == OBSERVED else GOLD if cat == MODELED else LAV
    st.caption(
        f"<span style='color:{color};font-weight:600;'>{cat}</span> · source: {src} · "
        f"confidence {conf*100:.0f}% · {provenance.get('method', '')}",
        unsafe_allow_html=True,
    )


def _finra_weekly_note(engine, r: dict) -> None:
    """Clearly flag that the observed FINRA figure is WEEKLY, with the exact week.

    FINRA OTC Transparency publishes weekly summaries with a 1-4 week reporting
    lag. The user must never mistake the observed weekly total for a same-day
    figure — the daily series around it are modeled and labeled as such.
    """
    if not r.get("finra_observed"):
        return
    meta = {}
    try:
        meta = engine.finra_metadata() or {}
    except Exception:
        pass
    vol = r.get("finra_offexchange_vol")
    week = meta.get("week") or "the latest published reporting week"
    tiers = ", ".join(meta.get("tiers") or [])
    tier_txt = f" · tiers: {tiers}" if tiers else ""
    st.info(
        f"**FINRA OTC — OBSERVED · WEEKLY.** {vol:,.0f} shares traded off-exchange "
        f"in the latest published reporting week (week of **{week}**{tier_txt}). "
        f"FINRA publishes **weekly** summaries with a 1–4 week reporting lag — this "
        f"is a weekly total, **not today's figure**. The daily series on this page "
        f"are modeled around this observed anchor and are labeled MODELED.")


def _finra_scan_note(engine) -> None:
    """One-line weekly-granularity flag used by market-wide views."""
    meta = {}
    try:
        meta = engine.finra_metadata() or {}
    except Exception:
        pass
    if meta.get("observed"):
        st.caption(
            f"Off-exchange anchors: **FINRA OTC Transparency — weekly summaries** "
            f"(week of {meta.get('week')}; {meta.get('symbols', 0):,} symbols; tiers "
            f"{', '.join(meta.get('tiers') or [])}). Weekly granularity with a 1–4 week "
            f"reporting lag; daily figures are modeled around the observed anchor.")
    else:
        st.caption(
            "Off-exchange figures are **MODELED** (no FINRA anchor currently available). "
            "FINRA OTC Transparency publishes weekly per-symbol summaries when the API "
            "is reachable; the dashboard will switch to observed anchors automatically.")


def _label_badge(observed: bool) -> str:
    if observed:
        return status_badge("OBSERVED", "success")
    return status_badge("MODELED ESTIMATE", "gold")


def _imbalance_color(v: float) -> str:
    if v is None:
        return "#888"
    if v > 0.1:
        return DARK_GREEN
    if v < -0.1:
        return DARK_RED
    return "#ccc"


def _pct_color(v) -> str:
    try:
        v = float(v)
    except Exception:
        return "#888"
    return DARK_GREEN if v >= 0 else DARK_RED


def _fmt_big(v) -> str:
    try:
        v = float(v)
    except Exception:
        return "—"
    if v >= 1e9:
        return f"${v/1e9:.2f}B"
    if v >= 1e6:
        return f"${v/1e6:.1f}M"
    if v >= 1e3:
        return f"${v/1e3:.1f}K"
    return f"${v:,.0f}"


def _engine():
    if not _HAS_ENGINE:
        st.error("Dark Pool engine unavailable. Check that dark_pool_engine.py imports cleanly.")
        st.stop()
    return get_dark_pool_engine()


def _layout(fig: go.Figure, height: int = 360, **kw):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["navy"],
        plot_bgcolor=COLORS["navy_light"],
        font_color=COLORS["text_primary"],
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        **kw,
    )
    return fig


# --------------------------------------------------------------------------- #
#  Dashboard
# --------------------------------------------------------------------------- #


def show_dark_pool_dashboard() -> None:
    if not _HAS_ENGINE:
        st.error(f"Dark Pool Intelligence unavailable: {_ENGINE_ERR}")
        st.info("Ensure dark_pool_engine.py is present and its dependencies install cleanly.")
        return

    st.title("Dark Pool Intelligence")
    st.caption(
        "Off-exchange flow analytics, institutional positioning inference, "
        "and market-impact research — with full data provenance and confidence labeling."
    )

    mode = _mode_selector()
    engine = _engine()
    regime = engine.detect_regime()

    # Data-integrity banner (honest labeling, always visible)
    dq = engine.data_quality_report()
    if dq.get("banner"):
        st.warning(dq["banner"])

    col_r1, col_r2, col_r3, col_r4 = st.columns([2, 2, 2, 3])
    with col_r1:
        st.markdown(
            f"<span style='color:{regime['color']};font-weight:700;font-size:1.05rem;'>"
            f"REGIME: {regime['label']}</span>", unsafe_allow_html=True)
        st.caption(regime.get("description", ""))
    with col_r2:
        if regime.get("vix") is not None:
            st.metric("VIX", f"{regime['vix']:.1f}",
                      f"{regime.get('vix_1w_chg', 0):+.2f}% 1w")
        else:
            st.metric("VIX", "—")
    with col_r3:
        if regime.get("spy_5d_pct") is not None:
            st.metric("SPY 5D", f"{regime['spy_5d_pct']:+.2f}%")
        else:
            st.metric("SPY 5D", "—")
    with col_r4:
        st.markdown(
            f"<div style='text-align:right;color:#a0a8b8;font-size:0.75rem;'>"
            f"Data Quality Score<br>"
            f"<span style='font-size:1.4rem;font-weight:700;color:{GOLD};'>"
            f"{dq.get('overall_score', 0):.0f}/100</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    tabs = st.tabs([
        "Market Overview", "Dark Pool Scanner", "Ticker Intelligence",
        "Largest Prints", "Imbalance Monitor", "Institutional Activity",
        "Sector Analysis", "Historical Analytics", "Signal Engine",
        "Backtesting", "AI Analyst", "Alerts & Watchlists",
        "Data Quality Center", "Methodology", "Settings & API",
    ])

    with tabs[0]:
        _tab_market_overview(engine, mode)
    with tabs[1]:
        _tab_scanner(engine, mode)
    with tabs[2]:
        _tab_ticker(engine, mode)
    with tabs[3]:
        _tab_prints(engine, mode)
    with tabs[4]:
        _tab_imbalance(engine, mode)
    with tabs[5]:
        _tab_institutional(engine, mode)
    with tabs[6]:
        _tab_sectors(engine, mode)
    with tabs[7]:
        _tab_historical(engine, mode)
    with tabs[8]:
        _tab_signals(engine, mode)
    with tabs[9]:
        _tab_backtest(engine, mode)
    with tabs[10]:
        _tab_ai(engine, mode)
    with tabs[11]:
        _tab_alerts(engine, mode)
    with tabs[12]:
        _tab_data_quality(engine, mode)
    with tabs[13]:
        _tab_methodology(engine, mode)
    with tabs[14]:
        _tab_settings(engine, mode)


# --------------------------------------------------------------------------- #
#  1. Market Overview
# --------------------------------------------------------------------------- #


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def _overview_scan(limit: int = SCAN_DEFAULT_LIMIT) -> pd.DataFrame:
    df = get_dark_pool_engine().scan_market(limit=limit)
    if df is None or df.empty:
        # Never cache a transient failure — the next render retries the scan.
        try:
            _overview_scan.clear()
        except Exception:
            pass
    return df


def _tab_market_overview(engine, mode: str) -> None:
    section_header("Market Overview — Off-Exchange Activity")

    with st.spinner("Scanning the dynamic universe (background-safe, cached 15 min)…"):
        df = _overview_scan()

    if df.empty:
        st.warning("No scan results yet — the market-data layer may be unavailable.")
        return

    df = df.copy()
    # metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Names Scanned", f"{len(df):,}")
    m2.metric(
        "Avg Off-Exchange %",
        f"{df['OffEx%'].mean():.1f}%",
        help="Modeled share of consolidated volume executed off-exchange (20d).",
    )
    m3.metric("Net Imbalance", f"{df['Imbalance'].sum():+.0f}",
              help="Sum of per-name estimated imbalances across the scan.")
    m4.metric("Avg Pressure Score", f"{df['PressureScore'].mean():.0f}/100")
    m5.metric("Top Signal Strength", f"{df['SignalStrength'].max():.0f}")

    _finra_scan_note(engine)

    st.markdown("---")

    top_pressure = df.nlargest(8, "PressureScore")[["Symbol", "Sector", "OffEx%",
                                                    "Imbalance%", "PressureScore",
                                                    "Institutional", "TopSignal"]]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Highest Dark-Pool Pressure**")
        st.dataframe(top_pressure.reset_index(drop=True), width="stretch", hide_index=True)
    with c2:
        # imbalance distribution chart
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df["Imbalance"], nbinsx=24,
            marker_color=GOLD, opacity=0.85,
            name="Imbalance distribution"))
        fig.add_vline(x=0, line_dash="dash", line_color="#888")
        fig.update_layout(title="Estimated Imbalance Distribution (scan)",
                          xaxis_title="5-day buy/sell imbalance", yaxis_title="Names")
        st.plotly_chart(_layout(fig, 320), width="stretch")

    st.markdown("---")
    if mode in ("Advanced", "Institutional"):
        section_header("Full Scan Table")
        st.dataframe(df, width="stretch", hide_index=True)
    if mode == "Institutional":
        _provenance_line(mode, {
            "category": "MIXED", "source": "scan_market()", "method": (
                "consolidated OHLCV observed; off-exchange share + direction modeled; "
                "percentiles from per-name history"),
            "confidence": 0.55,
        })


# --------------------------------------------------------------------------- #
#  2. Scanner
# --------------------------------------------------------------------------- #


def _tab_scanner(engine, mode: str) -> None:
    section_header("Dark Pool Market Scanner")

    c_lim, c_sort, c_min = st.columns([2, 2, 2])
    with c_lim:
        limit = st.slider("Names to scan", 20, SCAN_MAX_LIMIT, SCAN_DEFAULT_LIMIT,
                          step=10, key="dp_scan_limit",
                          help="Scans the dynamic universe — never a preset list.")
    with c_sort:
        sort_by = st.selectbox(
            "Rank by",
            ["PressureScore", "Imbalance", "OffEx%", "RelVolume", "SignalStrength",
             "InstConfidence", "Change1D%"],
            key="dp_scan_sort")
    with c_min:
        min_pressure = st.slider("Min pressure score", 0, 100, 0, key="dp_scan_minp")

    with st.spinner(f"Scanning {limit} names (cached)…"):
        df = _overview_scan(limit=limit)

    if df.empty:
        st.warning("Scan returned no results.")
        return

    df = df[df["PressureScore"] >= min_pressure]
    if df.empty:
        st.info("No names meet the filter.")
        return

    _finra_scan_note(engine)

    # Ranked tabs: Most Active / Most Unusual / Imbalance / Largest Blocks / Signals
    st.markdown("### Rankings")
    r1, r2, r3, r4, r5 = st.tabs([
        "Most Active", "Most Unusual", "Strongest Buy Imbalance",
        "Strongest Sell Imbalance", "Top Signals",
    ])

    cols = ["Symbol", "Sector", "Price", "OffEx%", "Imbalance%", "RelVolume",
            "PressureScore", "Institutional", "InstConfidence", "TopSignal"]

    with r1:
        st.dataframe(df.nlargest(15, "RelVolume")[cols].reset_index(drop=True),
                     width="stretch", hide_index=True)
    with r2:
        un = df.assign(Unusualness=df["OffEx%"].rank(pct=True) * 0.5
                       + df["SignalStrength"].rank(pct=True) * 0.5)
        st.dataframe(un.nlargest(15, "Unusualness")[cols + ["Unusualness"]]
                     .round(2).reset_index(drop=True), width="stretch", hide_index=True)
    with r3:
        st.dataframe(df[df["Imbalance"] > 0].nlargest(15, "Imbalance")[cols]
                     .reset_index(drop=True), width="stretch", hide_index=True)
    with r4:
        st.dataframe(df[df["Imbalance"] < 0].nsmallest(15, "Imbalance")[cols]
                     .reset_index(drop=True), width="stretch", hide_index=True)
    with r5:
        st.dataframe(df.nlargest(15, "SignalStrength")[cols].reset_index(drop=True),
                     width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown("### Sector / Pressure Matrix")
    fig = go.Figure(go.Scatter(
        x=df["OffEx%"], y=df["Imbalance"],
        mode="markers+text",
        text=df["Symbol"],
        textposition="top center",
        marker=dict(
            size=df["PressureScore"].clip(8, 30),
            color=df["PressureScore"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Pressure"),
            line=dict(width=1, color="#0a1628"),
        ),
        hovertemplate="%{text}<br>OffEx %{x:.1f}%<br>Imb %{y:+.2f}<br>Pressure %{marker.color:.0f}<extra></extra>",
    ))
    fig.update_layout(title="Off-Exchange % vs Estimated Imbalance (bubble = pressure)",
                      xaxis_title="Off-exchange % (20d, modeled)", yaxis_title="Imbalance (5d)")
    st.plotly_chart(_layout(fig, 460), width="stretch")

    if mode == "Institutional":
        _provenance_line(mode, df.attrs.get("provenance") or {
            "category": "MIXED", "source": "scan_market()", "method":
                "modeled off-exchange share + heuristic direction on observed OHLCV",
            "confidence": 0.55})


# --------------------------------------------------------------------------- #
#  3. Ticker Intelligence
# --------------------------------------------------------------------------- #


_TICKER_REPORT_CACHE: Dict[str, Tuple[float, dict]] = {}
_TICKER_REPORT_TTL = 300.0


def _ticker_report(symbol: str) -> dict:
    """Ticker analysis with failure-safe caching.

    Only SUCCESSFUL results are cached (5-min TTL). A failed analysis — e.g.
    a transient data-layer hiccup that returned 'insufficient history' — is
    NEVER cached, so the very next render retries the provider instead of
    replaying the error for the whole TTL window.
    """
    now = _time.monotonic()
    # Opportunistic eviction: bound cache growth across a long session.
    if len(_TICKER_REPORT_CACHE) > 256:
        stale = [s for s, (ts, _) in _TICKER_REPORT_CACHE.items()
                 if now - ts >= _TICKER_REPORT_TTL]
        for s in stale:
            _TICKER_REPORT_CACHE.pop(s, None)
        if len(_TICKER_REPORT_CACHE) > 512:  # hard cap fallback
            _TICKER_REPORT_CACHE.clear()
    cached = _TICKER_REPORT_CACHE.get(symbol)
    if cached is not None and now - cached[0] < _TICKER_REPORT_TTL:
        return cached[1]
    r = get_dark_pool_engine().analyze_ticker(symbol)
    if r.get("ok"):
        _TICKER_REPORT_CACHE[symbol] = (_time.monotonic(), r)
    return r


def _tab_ticker(engine, mode: str) -> None:
    section_header("Ticker-Level Dark Pool Intelligence")

    symbol = st.text_input("Enter any supported ticker", value="NVDA",
                           key="dp_ticker", help="Search any ticker in the dynamic universe.")
    symbol = (symbol or "").strip().upper()

    if st.button("Analyze", key="dp_ticker_go", type="primary"):
        st.session_state["dp_analyzed"] = symbol
    symbol = st.session_state.get("dp_analyzed", symbol or "NVDA")

    if not symbol:
        st.info("Enter a ticker to begin.")
        return

    with st.spinner(f"Analyzing {symbol}…"):
        r = _ticker_report(symbol)

    if not r.get("ok"):
        st.error(f"{symbol}: {r.get('error', 'analysis failed')}")
        if mode == "Institutional":
            _provenance_line(mode, r.get("provenance"))
        return

    live = r.get("live_price")
    price_disp = live.get("price") if live else r.get("price")

    # ---- Overview metrics ----
    o1, o2, o3, o4, o5, o6 = st.columns(6)
    o1.metric("Price", f"${price_disp:,.2f}",
              f"{live.get('change_pct'):+.2f}%" if live and live.get("change_pct") is not None
              else f"{r.get('change_1d_pct', 0):+.2f}%")
    o2.metric("Market Cap", _fmt_big(r.get("market_cap")) if r.get("market_cap") else "—")
    o3.metric("Off-Ex Vol (20d)", f"{r.get('offexchange_vol_20d_avg', 0):,.0f}",
              help="Modeled average DAILY off-exchange volume (20d). Daily series is "
                   "modeled around the observed FINRA weekly anchor where available.")
    o4.metric("Off-Ex %", f"{r.get('offexchange_pct_20d', 0):.1f}%",
              delta=f"{r.get('offexchange_pct_today', 0):.1f}% today",
              delta_color="off",
              help="Modeled share of consolidated volume executed off-exchange (20d).")
    o5.metric("Imbalance (5d)", f"{r.get('imbalance_pct', 0):+.1f}%",
              help="Estimated buy/sell imbalance — heuristic, not prints.")
    o6.metric("Pressure Score", f"{r.get('pressure_score', 0):.0f}/100")

    # Weekly-granularity flag (FINRA is a weekly source — never implied daily)
    _finra_weekly_note(engine, r)

    st.markdown("---")

    # ---- Activity chart ----
    series: pd.DataFrame = r.get("series")
    if series is not None and not series.empty:
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.5, 0.25, 0.25],
            vertical_spacing=0.04,
            subplot_titles=("Price", "Off-Exchange Volume (modeled)", "Rolling Imbalance (est.)"))
        fig.add_trace(go.Scatter(x=series.index, y=series["Close"], name="Close",
                                 line=dict(color=GOLD, width=1.8)), row=1, col=1)
        fig.add_trace(go.Bar(x=series.index, y=series["OffExchangeVol"],
                             name="Off-Ex Vol", marker_color="rgba(155,142,196,0.6)"),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=series.index, y=series["RollImbalance"] * 100,
                                 name="Imbalance %", line=dict(color=DARK_GREEN, width=1.5)),
                      row=3, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="#666", row=3, col=1)
        st.plotly_chart(_layout(fig, 520), width="stretch")

    # ---- Key stats + historical context ----
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Key Metrics**")
        rows = [
            ("VWAP (20d)", f"${r.get('vwap_20d', 0):,.2f}"),
            ("Rel Volume (today/20d)", f"{r.get('rel_volume', 0):.2f}x"),
            ("Avg Volume (20d)", f"{r.get('avg_volume_20d', 0):,.0f}"),
            ("Block Vol Today (modeled)", f"{r.get('block_vol_today', 0):,.0f}"),
            ("Buy Vol (5d, est.)", f"{r.get('buy_vol_5d', 0):,.0f}"),
            ("Sell Vol (5d, est.)", f"{r.get('sell_vol_5d', 0):,.0f}"),
            ("Imbalance Acceleration", f"{r.get('imbalance_accel', 0):+.3f}"),
        ]
        if r.get("finra_observed") and r.get("finra_offexchange_vol"):
            rows.append((
                "FINRA off-ex vol — WEEKLY total",
                f"{r.get('finra_offexchange_vol'):,.0f}"))
        for k, v in rows:
            st.markdown(f"**{k}:** {v}")
        if r.get("finra_observed"):
            st.caption(
                "The FINRA figure is the total off-exchange volume for the latest "
                "published **week** — FINRA publishes weekly summaries with a 1–4 week "
                "reporting lag, so it is not a single-day number.")
    with c2:
        st.markdown("**Historical Context (percentiles vs own history)**")
        hist = r.get("historical", {})
        hrows = [
            ("Off-Ex vol — 5d pctile", hist.get("offex_pctile_5d")),
            ("Off-Ex vol — 20d pctile", hist.get("offex_pctile_20d")),
            ("Off-Ex vol — 60d pctile", hist.get("offex_pctile_60d")),
            ("Off-Ex share — 1y pctile", hist.get("share_pctile_1y")),
            ("Off-Ex vol z — 5d", hist.get("offex_z_5d")),
            ("Off-Ex vol z — 20d", hist.get("offex_z_20d")),
            ("Volume — 20d pctile", hist.get("volume_pctile_20d")),
        ]
        for k, v in hrows:
            if v is None:
                st.markdown(f"**{k}:** —")
            elif isinstance(v, float) and abs(v) > 100:
                st.markdown(f"**{k}:** {v:+,.0f}")
            else:
                st.markdown(f"**{k}:** {v}")

    st.markdown("---")

    # ---- Institutional inference ----
    inst = r.get("institutional", {})
    pattern = inst.get("pattern", "Indeterminate")
    pcol = DARK_GREEN if pattern in ("Potential accumulation", "Absorption") else (
        DARK_RED if pattern == "Potential distribution" else "#ccc")
    st.markdown(
        f"<div style='background:#161b22;border-left:4px solid {pcol};border-radius:6px;"
        f"padding:12px 16px;'>"
        f"<span style='color:#aaa;font-size:0.72rem;letter-spacing:1px;'>INSTITUTIONAL INFERENCE "
        f"(confidence {inst.get('confidence', 0)*100:.0f}%)</span><br>"
        f"<span style='color:{pcol};font-weight:700;font-size:1.05rem;'>{pattern}</span><br>"
        f"<span style='color:#ccc;font-size:0.85rem;'>{inst.get('description', '')}</span>"
        f"</div>", unsafe_allow_html=True)

    if mode in ("Advanced", "Institutional"):
        st.markdown("**Evidence for inference**")
        st.json(inst.get("evidence", {}))

    st.markdown("---")

    # ---- Price relationship ----
    rel = r.get("price_relationship", {})
    if rel.get("ok") and rel.get("rows"):
        st.markdown("**Dark-Pool Activity vs Forward Returns (out-of-sample)**")
        st.caption("Historical conditional statistics — not causal. Sample sizes shown.")
        rows = []
        for row in rel["rows"]:
            if row.get("enough"):
                rows.append({
                    "Condition": row["label"],
                    "N": row["n"],
                    "Avg Return %": row["avg_return_pct"],
                    "Baseline %": row["baseline_return_pct"],
                    "Excess %": row["excess_return_pct"],
                    "Win Rate %": row["win_rate_pct"],
                    "t-stat": row["t_stat"],
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    if mode == "Institutional":
        st.markdown("**Provenance**")
        for k, p in r.get("provenance", {}).items():
            with st.expander(f"{k} — {p.get('category', '')}", expanded=False):
                st.json(p)


# --------------------------------------------------------------------------- #
#  4. Largest Prints
# --------------------------------------------------------------------------- #


def _tab_prints(engine, mode: str) -> None:
    section_header("Largest Dark-Pool Prints (Modeled)")

    st.caption(
        "Individual dark-pool prints are not observable in consolidated daily data. "
        "These are **modeled allocations** of estimated block volume for relative "
        "comparison — always labeled MODELED ESTIMATE, never exchange-reported prints.")

    symbol = st.text_input("Ticker", value="NVDA", key="dp_prints_ticker")
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return

    with st.spinner(f"Analyzing {symbol}…"):
        r = _ticker_report(symbol)
    if not r.get("ok"):
        st.warning(r.get("error", "analysis failed"))
        return

    prints = r.get("prints", [])
    if not prints:
        st.info("No meaningful block volume modeled for this name today.")
        return

    pf = pd.DataFrame(prints)
    pf["Notional"] = pf["notional"].apply(lambda v: f"${v:,.0f}")
    pf["Shares"] = pf["shares"].apply(lambda v: f"{v:,.0f}")
    pf["% Daily Vol"] = pf["pct_daily_volume"].apply(lambda v: f"{v:.2f}%")
    pf["Significance"] = pf["significance"]

    st.dataframe(pf[["rank", "symbol", "price", "Shares", "Notional",
                     "% Daily Vol", "Significance", "label"]]
                 .rename(columns={"rank": "#", "symbol": "Ticker", "price": "Price",
                                  "label": "Classification"}),
                 width="stretch", hide_index=True)

    # highlight the top print
    top = prints[0]
    st.markdown(
        f"<div style='background:#161b22;border:1px solid {GOLD};border-radius:8px;"
        f"padding:14px 18px;margin-top:8px;'>"
        f"<span style='color:#aaa;font-size:0.7rem;'>LARGEST MODELED PRINT — {top['symbol']}</span><br>"
        f"<span style='font-size:1.4rem;font-weight:700;color:{GOLD};'>"
        f"${top['notional']:,.0f}</span> "
        f"<span style='color:#ccc;'>({top['shares']:,.0f} sh @ ${top['price']:,.2f})</span><br>"
        f"<span style='color:#aaa;font-size:0.8rem;'>{top['pct_daily_volume']:.2f}% of "
        f"daily consolidated volume · significance {top['significance']:.0f}/100</span>"
        f"</div>", unsafe_allow_html=True)

    if mode == "Institutional":
        _provenance_line(mode, {
            "category": "MODELED", "source": "dark_pool_engine._model_largest_prints",
            "method": "block-volume allocation model on modeled off-exchange share",
            "confidence": 0.4,
            "note": "Sizes and counts are plausible allocations, not observed transactions.",
        })


# --------------------------------------------------------------------------- #
#  5. Imbalance Monitor
# --------------------------------------------------------------------------- #


def _tab_imbalance(engine, mode: str) -> None:
    section_header("Dark-Pool Imbalance Monitor")

    st.caption(
        "Imbalance is an **estimate** derived from daily return + VWAP-position "
        "heuristics applied to modeled off-exchange volume. It is not trade-print data.")

    symbol = st.text_input("Ticker", value="NVDA", key="dp_imb_ticker")
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return

    with st.spinner(f"Analyzing {symbol}…"):
        r = _ticker_report(symbol)
    if not r.get("ok"):
        st.warning(r.get("error", "analysis failed"))
        return

    i1, i2, i3, i4, i5 = st.columns(5)
    imb = r.get("imbalance_5d", 0)
    i1.metric("Buy Vol (5d, est.)", f"{r.get('buy_vol_5d', 0):,.0f}")
    i2.metric("Sell Vol (5d, est.)", f"{r.get('sell_vol_5d', 0):,.0f}")
    i3.metric("Net Imbalance", f"{r.get('buy_vol_5d', 0) - r.get('sell_vol_5d', 0):+,.0f}")
    i4.metric("Imbalance %", f"{r.get('imbalance_pct', 0):+.1f}%")
    i5.metric("Acceleration", f"{r.get('imbalance_accel', 0):+.3f}",
              help="Change in rolling imbalance over the last 5 sessions.")

    # gauge-style indicator
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=imb * 100,
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [-60, 60]},
            "bar": {"color": _imbalance_color(imb)},
            "steps": [
                {"range": [-60, -20], "color": "rgba(255,68,68,0.25)"},
                {"range": [-20, 20], "color": "rgba(204,204,204,0.15)"},
                {"range": [20, 60], "color": "rgba(0,255,136,0.25)"},
            ],
            "threshold": {"line": {"color": "#fff", "width": 2},
                          "thickness": 0.9, "value": 0},
        },
        title={"text": "5-Day Estimated Imbalance"},
    ))
    st.plotly_chart(_layout(fig, 300), width="stretch")

    series = r.get("series")
    if series is not None and not series.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=series.index, y=series["RollImbalance"] * 100,
            fill="tozeroy", line=dict(color=DARK_GREEN if imb > 0 else DARK_RED, width=1.5),
            name="Rolling imbalance %"))
        fig.add_hline(y=0, line_dash="dash", line_color="#888")
        fig.update_layout(title="Rolling Imbalance History (%)", yaxis_title="%")
        st.plotly_chart(_layout(fig, 320), width="stretch")

    if mode in ("Advanced", "Institutional"):
        st.markdown("**Imbalance vs own history**")
        hist = r.get("historical", {})
        st.markdown(
            f"Off-exchange volume percentile (5d): **{hist.get('offex_pctile_5d', '—')}%** — "
            f"a large imbalance is most meaningful when volume is also unusual.")
        if mode == "Institutional":
            _provenance_line(mode, {
                "category": "MODELED", "source": "dark_pool_engine",
                "method": "return/VWAP heuristic on modeled off-exchange volume",
                "confidence": 0.45,
                "note": "Directional classification is estimated, not observed.",
            })


# --------------------------------------------------------------------------- #
#  6. Institutional Activity
# --------------------------------------------------------------------------- #


def _tab_institutional(engine, mode: str) -> None:
    section_header("Institutional Activity Detection")

    st.caption(
        "Patterns below are **inferences** from the combination of off-exchange "
        "share, estimated imbalance and price action. The engine never claims to "
        "identify a specific institution, and labels every conclusion with confidence.")

    symbol = st.text_input("Ticker", value="NVDA", key="dp_inst_ticker")
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return

    with st.spinner(f"Analyzing {symbol}…"):
        r = _ticker_report(symbol)
    if not r.get("ok"):
        st.warning(r.get("error", "analysis failed"))
        return

    inst = r.get("institutional", {})
    pattern = inst.get("pattern", "Indeterminate")
    pcol = DARK_GREEN if pattern in ("Potential accumulation", "Absorption") else (
        DARK_RED if pattern == "Potential distribution" else "#ccc")
    conf = inst.get("confidence", 0) * 100

    st.markdown(
        f"<div style='background:#161b22;border:1px solid {pcol};border-radius:10px;"
        f"padding:18px 22px;'>"
        f"<div style='color:#aaa;font-size:0.72rem;letter-spacing:1.5px;'>CLASSIFICATION "
        f"· INFERENCE · CONFIDENCE {conf:.0f}%</div>"
        f"<div style='color:{pcol};font-size:1.6rem;font-weight:700;margin:4px 0;'>"
        f"{pattern}</div>"
        f"<div style='color:#ccc;font-size:0.92rem;'>{inst.get('description', '')}</div>"
        f"</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Evidence**")
    ev = inst.get("evidence", {})
    e1, e2, e3, e4, e5 = st.columns(5)
    e1.metric("Off-Ex pctile (20d)", f"{ev.get('offex_pctile_20d', '—')}%")
    e2.metric("Off-Ex z (20d)", f"{ev.get('offex_z_20d', 0):+.2f}")
    e3.metric("Imbalance (5d)", f"{ev.get('imbalance_5d', 0):+.2f}")
    e4.metric("10d Return", f"{ev.get('ret_10d_pct', 0):+.2f}%")
    e5.metric("Vol ratio (10/40)", f"{ev.get('vol_ratio_10_40', 0):.2f}x")

    # Pattern glossary
    with st.expander("What each pattern means"):
        st.markdown(
            "**Potential accumulation** — elevated off-exchange volume + positive imbalance "
            "while price is flat; consistent with institutional buying absorbing supply.\n\n"
            "**Potential distribution** — elevated off-exchange volume + negative imbalance "
            "while price is flat; consistent with institutional selling into strength.\n\n"
            "**Absorption** — elevated volume, balanced flow, flat price; two-sided "
            "institutional participation.\n\n"
            "**Liquidity transfer** — rising volume with moderately elevated off-exchange "
            "share; market-making / two-sided provision.\n\n"
            "**Indeterminate** — activity too normal or mixed to infer intent. This is the "
            "honest default when evidence is insufficient.")

    if mode == "Institutional":
        _provenance_line(mode, {
            "category": "INFERENCE", "source": "dark_pool_engine._institutional_inference",
            "method": "off-exchange share + imbalance + price-action pattern rules",
            "confidence": conf / 100,
            "note": "No specific institution is ever identified; this is pattern inference.",
        })


# --------------------------------------------------------------------------- #
#  7. Sector Analysis
# --------------------------------------------------------------------------- #


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def _sector_table(limit: int) -> pd.DataFrame:
    df = get_dark_pool_engine().sector_analysis(limit=limit)
    if df is None or df.empty:
        # Never cache a transient failure — the next render retries.
        try:
            _sector_table.clear()
        except Exception:
            pass
    return df


def _tab_sectors(engine, mode: str) -> None:
    section_header("Sector-Level Dark Pool Analytics")

    limit = st.slider("Names per scan", 20, SCAN_MAX_LIMIT, SCAN_DEFAULT_LIMIT,
                      step=10, key="dp_sector_limit")
    with st.spinner("Aggregating sector analytics (cached)…"):
        df = _sector_table(limit)

    if df.empty:
        st.warning("No sector data yet.")
        return

    s1, s2, s3 = st.columns(3)
    s1.metric("Sectors", f"{len(df)}")
    s2.metric("Highest Pressure Sector",
              df.iloc[0]["Sector"] if not df.empty else "—")
    s3.metric("Net Sector Imbalance", f"{df['AvgImbalance'].sum():+.2f}")

    st.markdown("---")
    st.dataframe(df, width="stretch", hide_index=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Sector"], y=df["AvgPressure"],
        marker_color=[DARK_GREEN if v >= 0 else DARK_RED for v in df["AvgImbalance"]],
        text=[f"{v:.0f}" for v in df["AvgPressure"]], textposition="outside",
        name="Avg Pressure"))
    fig.update_layout(title="Average Dark-Pool Pressure by Sector",
                      xaxis_tickangle=-30, yaxis_title="Pressure /100")
    st.plotly_chart(_layout(fig, 380), width="stretch")

    if mode in ("Advanced", "Institutional"):
        st.markdown("**Top names by sector**")
        for _, row in df.iterrows():
            with st.expander(f"{row['Sector']} — {row['Tickers']} names", expanded=False):
                st.markdown(f"Representative names: {row['Names']}")
                st.markdown(
                    f"Avg off-exchange %: {row['AvgOffExPct']:.1f}% · "
                    f"avg imbalance: {row['AvgImbalance']:+.2f} · "
                    f"avg pressure: {row['AvgPressure']:.0f} · "
                    f"max signal: {row['MaxSignal']:.0f}")


# --------------------------------------------------------------------------- #
#  8. Historical Analytics
# --------------------------------------------------------------------------- #


def _tab_historical(engine, mode: str) -> None:
    section_header("Historical Analytics & Statistical Context")

    st.caption(
        "Percentiles and z-scores compare a name's CURRENT activity against ITS OWN "
        "trailing distributions — a big imbalance is only meaningful relative to "
        "that name's normal behaviour.")

    symbol = st.text_input("Ticker", value="NVDA", key="dp_hist_ticker")
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return

    with st.spinner(f"Analyzing {symbol}…"):
        r = _ticker_report(symbol)
    if not r.get("ok"):
        st.warning(r.get("error", "analysis failed"))
        return

    hist = r.get("historical", {})

    # percentile bars
    items = [
        ("Off-Ex vol · 5d", hist.get("offex_pctile_5d")),
        ("Off-Ex vol · 20d", hist.get("offex_pctile_20d")),
        ("Off-Ex vol · 60d", hist.get("offex_pctile_60d")),
        ("Off-Ex share · 1y", hist.get("share_pctile_1y")),
        ("Volume · 20d", hist.get("volume_pctile_20d")),
    ]
    labels = [k for k, _ in items if _[1] is not None]
    values = [v for _, v in items if v is not None]
    if labels:
        fig = go.Figure(go.Bar(
            x=values, y=labels, orientation="h",
            marker_color=[DARK_GREEN if v >= 70 else DARK_RED if v <= 30 else GOLD
                          for v in values],
            text=[f"{v:.0f}%" for v in values], textposition="outside"))
        fig.add_vline(x=50, line_dash="dash", line_color="#666")
        fig.update_layout(title=f"{symbol} — Activity Percentiles vs Own History",
                          xaxis_title="Percentile (%)", xaxis_range=[0, 100])
        st.plotly_chart(_layout(fig, 300), width="stretch")
    else:
        st.info("Not enough history for percentile context.")

    # z-score table
    zrows = [
        ("Off-Ex vol z · 5d", hist.get("offex_z_5d")),
        ("Off-Ex vol z · 20d", hist.get("offex_z_20d")),
        ("Off-Ex vol z · 60d", hist.get("offex_z_60d")),
        ("Off-Ex share z · 1y", hist.get("share_z_1y")),
    ]
    st.markdown("**Z-scores (std dev from own mean)**")
    zdf = pd.DataFrame([
        {"Metric": k, "Z": v} for k, v in zrows if v is not None
    ])
    if not zdf.empty:
        zdf["Signal"] = zdf["Z"].apply(
            lambda z: "UNUSUAL" if abs(z) >= 2 else "elevated" if abs(z) >= 1 else "normal")
        st.dataframe(zdf, width="stretch", hide_index=True)
        st.caption("|z| ≥ 2 is statistically unusual; |z| ≥ 1 is elevated. "
                   "Based on each name's own trailing window.")

    # 3D-ish scatter of share vs return vs volume (advanced)
    series = r.get("series")
    if mode in ("Advanced", "Institutional") and series is not None and len(series) > 30:
        st.markdown("**Off-exchange share vs daily return (last 60 sessions)**")
        tail = series.tail(60).copy()
        tail["share"] = tail["OffExchangeVol"] / tail["Volume"].replace(0, np.nan)
        fig = go.Figure(go.Scatter(
            x=tail["share"] * 100, y=tail["ReturnPct"],
            mode="markers",
            marker=dict(color=tail["RollImbalance"] * 100,
                        colorscale="RdYlGn", size=7,
                        colorbar=dict(title="Imb %"),
                        line=dict(width=0.5, color="#0a1628")),
            text=tail.index.strftime("%Y-%m-%d"),
            hovertemplate="%{text}<br>share %{x:.1f}%<br>ret %{y:+.2f}%<extra></extra>"))
        fig.update_layout(title="Off-Exchange Share vs Return (color = imbalance)",
                          xaxis_title="Off-ex share %", yaxis_title="Daily return %")
        st.plotly_chart(_layout(fig, 360), width="stretch")

    if mode == "Institutional":
        _provenance_line(mode, {
            "category": "MIXED", "source": "dark_pool_engine._historical_context",
            "method": "trailing-window percentiles and z-scores on modeled series",
            "confidence": 0.55,
        })


# --------------------------------------------------------------------------- #
#  9. Signal Engine
# --------------------------------------------------------------------------- #


def _tab_signals(engine, mode: str) -> None:
    section_header("Dark Pool Signal Engine")

    st.caption("Each signal shows: Signal → Evidence → Strength → Confidence → "
               "Interpretation. All signals derive from modeled flow and carry "
               "confidence — none is presented as a trade recommendation.")

    symbol = st.text_input("Ticker", value="NVDA", key="dp_sig_ticker")
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return

    with st.spinner(f"Analyzing {symbol}…"):
        r = _ticker_report(symbol)
    if not r.get("ok"):
        st.warning(r.get("error", "analysis failed"))
        return

    signals = r.get("signals", [])
    if not signals:
        st.info("No signals.")
        return

    for sig in signals:
        strength = sig.get("strength", 0)
        scolor = DARK_GREEN if strength >= 70 else AMBER if strength >= 45 else "#ccc"
        kind = sig.get("kind", "neutral")
        kcolor = {"volume": GOLD, "print": LAV, "direction": DARK_GREEN,
                  "liquidity": AMBER, "neutral": "#ccc"}.get(kind, "#ccc")
        with st.container():
            st.markdown(
                f"<div style='background:#161b22;border-left:4px solid {kcolor};"
                f"border-radius:6px;padding:12px 16px;margin:6px 0;'>"
                f"<div style='display:flex;justify-content:space-between;'>"
                f"<span style='color:{kcolor};font-weight:700;'>{sig.get('name', '')}</span>"
                f"<span style='color:{scolor};font-weight:700;'>strength {strength:.0f} · "
                f"confidence {sig.get('confidence', 0):.0f}%</span></div>"
                f"<div style='color:#ccc;font-size:0.85rem;margin-top:4px;'>"
                f"{sig.get('evidence', '')}</div>"
                f"<div style='color:#a0a8b8;font-size:0.82rem;margin-top:4px;'>"
                f"<b>Interpretation:</b> {sig.get('interpretation', '')}</div>"
                f"</div>", unsafe_allow_html=True)
            if mode == "Institutional":
                st.caption(f"Classification: MODELED ESTIMATE · {sig.get('observed', False)}")

    if mode == "Institutional":
        _provenance_line(mode, {
            "category": "MODELED", "source": "dark_pool_engine._signal_engine",
            "method": "threshold rules over modeled volume/share/imbalance series",
            "confidence": 0.5,
            "note": "Signals are descriptive analytics, not investment advice.",
        })


# --------------------------------------------------------------------------- #
#  10. Backtesting
# --------------------------------------------------------------------------- #


def _tab_backtest(engine, mode: str) -> None:
    section_header("Dark Pool Signal Backtesting")

    st.caption(
        "Test what historically happened after a dark-pool condition fired. Uses a "
        "60/40 chronological train/test split, reports sample sizes, t-stats and "
        "confidence intervals. Historical relationships ≠ causation.")

    symbol = st.text_input("Ticker", value="NVDA", key="dp_bt_ticker")
    condition = st.selectbox(
        "Condition", [
            "high_share", "rising_imbalance", "low_share", "high_volume"],
        format_func=lambda c: {
            "high_share": "Off-exchange share > 80th pctile",
            "rising_imbalance": "Imbalance jump > 80th pctile",
            "low_share": "Off-exchange share < 20th pctile",
            "high_volume": "Consolidated volume > 80th pctile",
        }[c], key="dp_bt_cond")
    holding = st.slider("Holding period (trading days)", 1, 30, 5, key="dp_bt_hold")

    symbol = (symbol or "").strip().upper()
    if not symbol:
        return

    if st.button("Run Backtest", key="dp_bt_run", type="primary"):
        with st.spinner(f"Backtesting {symbol}…"):
            res = engine.backtest_signal(symbol, condition=condition, holding=holding)

        if not res.get("ok"):
            st.warning(res.get("error", "backtest failed"))
            return

        b1, b2, b3, b4 = st.columns(4)
        oos = res.get("out_of_sample", {})
        ins = res.get("in_sample", {})
        b1.metric("Out-of-sample N", oos.get("n", "—"))
        b2.metric("OOS Avg Return", f"{oos.get('avg_return_pct', 0):+.2f}%")
        b3.metric("OOS Win Rate", f"{oos.get('win_rate_pct', 0):.0f}%")
        b4.metric("OOS t-stat", f"{oos.get('t_stat', 0):+.2f}")

        st.markdown("---")
        bt_df = pd.DataFrame([
            {
                "Sample": "In-sample",
                "N": ins.get("n"),
                "Avg Ret %": ins.get("avg_return_pct"),
                "Median %": ins.get("median_return_pct"),
                "Baseline %": ins.get("baseline_return_pct"),
                "Excess %": ins.get("excess_return_pct"),
                "Win Rate %": ins.get("win_rate_pct"),
                "Max Gain %": ins.get("max_gain_pct"),
                "Max Loss %": ins.get("max_loss_pct"),
                "Vol %": ins.get("vol_pct"),
                "Sharpe-like": ins.get("sharpe_like"),
                "t-stat": ins.get("t_stat"),
                "95% CI": str(ins.get("ci95")),
            },
            {
                "Sample": "Out-of-sample",
                "N": oos.get("n"),
                "Avg Ret %": oos.get("avg_return_pct"),
                "Median %": oos.get("median_return_pct"),
                "Baseline %": oos.get("baseline_return_pct"),
                "Excess %": oos.get("excess_return_pct"),
                "Win Rate %": oos.get("win_rate_pct"),
                "Max Gain %": oos.get("max_gain_pct"),
                "Max Loss %": oos.get("max_loss_pct"),
                "Vol %": oos.get("vol_pct"),
                "Sharpe-like": oos.get("sharpe_like"),
                "t-stat": oos.get("t_stat"),
                "95% CI": str(oos.get("ci95")),
            },
        ])
        st.dataframe(bt_df, width="stretch", hide_index=True)

        st.info(res.get("note", ""))
        st.caption(f"Train rows: {res.get('train_rows')} · Test rows: {res.get('test_rows')} "
                   f"· Condition: {condition} · Holding: {holding}d")

        if mode == "Institutional":
            _provenance_line(mode, res.get("provenance"))


# --------------------------------------------------------------------------- #
#  11. AI Analyst
# --------------------------------------------------------------------------- #


def _tab_ai(engine, mode: str) -> None:
    section_header("AI Intelligence Layer")

    st.caption(
        "Every AI insight follows: What Happened → Why It Matters → Historical Context "
        "→ Evidence → Confidence → Alternative Explanation → Bottom Line. The AI is "
        "**grounded** — it constructs text only from computed metrics and never "
        "invents transactions, institutions, or statistics.")

    symbol = st.text_input("Ticker", value="NVDA", key="dp_ai_ticker")
    symbol = (symbol or "").strip().upper()

    if st.button("Generate Insight", key="dp_ai_go", type="primary"):
        with st.spinner(f"Generating grounded insight for {symbol}…"):
            r = _ticker_report(symbol)
            insight = engine.ai_insight(r)
            st.session_state["dp_ai_result"] = insight
            st.session_state["dp_ai_symbol"] = symbol

    if "dp_ai_result" in st.session_state and st.session_state["dp_ai_symbol"] == symbol:
        insight = st.session_state["dp_ai_result"]
        if not insight.get("ok"):
            st.warning(insight.get("insight", "Insufficient data to determine this reliably."))
            return

        st.markdown(
            f"<div style='background:#161b22;border:1px solid {GOLD};border-radius:10px;"
            f"padding:16px 20px;'>"
            f"<div style='color:#aaa;font-size:0.72rem;letter-spacing:1.5px;'>"
            f"AI ANALYST — {insight.get('symbol', '')} · {insight.get('label', 'MODELED INFERENCE')}"
            f" · CONFIDENCE {insight.get('confidence', 0)*100:.0f}%</div>"
            f"<div style='color:#eee;font-size:0.95rem;margin-top:8px;line-height:1.55;'>"
            f"{insight.get('insight', '')}</div>"
            f"</div>", unsafe_allow_html=True)

        st.markdown("---")
        for section, text in insight.get("sections", {}).items():
            with st.expander(section, expanded=(section == "Bottom Line")):
                st.markdown(text)

        if mode == "Institutional":
            _provenance_line(mode, insight.get("provenance"))
    else:
        st.info("Enter a ticker and click **Generate Insight**.")

    st.markdown("---")
    with st.expander("AI constraints (anti-hallucination policy)"):
        st.markdown(
            "- The AI never invents dark-pool transactions, venues, prices, or institutions.\n"
            "- Inferred direction is labeled an estimate; institutional patterns are "
            "labeled inferences with confidence.\n"
            "- If data is insufficient, the AI explicitly says: **\"Insufficient data to "
            "determine this reliably.\"**\n"
            "- Correlation is never presented as causation.\n"
            "- No trade recommendation is ever made from a single dark-pool signal.")


# --------------------------------------------------------------------------- #
#  12. Alerts & Watchlists
# --------------------------------------------------------------------------- #


def _tab_alerts(engine, mode: str) -> None:
    section_header("Alerts & Watchlists")

    a1, a2 = st.columns(2)
    with a1:
        st.markdown("**Watchlists**")
        wl = engine.get_watchlists()
        if not wl:
            st.info("No watchlists yet.")
        for name, syms in wl.items():
            with st.expander(f"{name} ({len(syms)})", expanded=True):
                for s in list(syms):
                    c1, c2 = st.columns([4, 1])
                    c1.markdown(f"**{s}**")
                    if c2.button("Remove", key=f"dp_wl_rm_{name}_{s}"):
                        engine.remove_from_watchlist(name, s)
                        st.rerun()
        with st.form("dp_wl_form", clear_on_submit=True):
            wname = st.text_input("New watchlist name", key="dp_wl_new_name")
            wsym = st.text_input("Symbol to add", key="dp_wl_new_sym")
            submit = st.form_submit_button("Create / Add")
        if submit and wname:
            engine.create_watchlist(wname)
            if wsym:
                engine.add_to_watchlist(wname, wsym.strip().upper())
            st.success(f"Watchlist '{wname}' ready.")
            st.rerun()

    with a2:
        st.markdown("**Alerts**")
        alerts = engine.get_alerts()
        if alerts:
            for i, al in enumerate(alerts):
                st.markdown(
                    f"**{al.get('symbol', '?')}** — {al.get('kind', '?')} @ "
                    f"{al.get('threshold', '?')} ({'on' if al.get('enabled', True) else 'off'})")
                if st.button("Delete", key=f"dp_alert_del_{i}"):
                    engine.remove_alert(i)
                    st.rerun()
        else:
            st.info("No alerts configured.")

        with st.form("dp_alert_form", clear_on_submit=True):
            asym = st.text_input("Symbol", key="dp_alert_sym")
            akind = st.selectbox("Trigger", [
                "offex_pctile_90", "large_print", "imbalance_threshold",
                "activity_acceleration", "ai_unusual"], key="dp_alert_kind",
                help="offex_pctile_90: 5d off-exchange volume >= threshold x 100th "
                     "percentile · large_print: modeled block volume >= threshold x "
                     "20d avg volume · imbalance_threshold: |5d imbalance| >= threshold "
                     "· activity_acceleration: |imbalance acceleration| >= threshold · "
                     "ai_unusual: top signal strength >= threshold x 100")
            athresh = st.number_input("Threshold", 0.0, 1.0, 0.9, 0.05,
                                      key="dp_alert_thresh")
            achan = st.multiselect("Channels", ["In-app", "Email", "Webhook"],
                                   default=["In-app"], key="dp_alert_chan")
            asub = st.form_submit_button("Create Alert")
        if asub and asym:
            engine.add_alert({
                "symbol": asym.strip().upper(),
                "kind": akind,
                "threshold": float(athresh),
                "enabled": True,
                "channels": achan,
                "created": datetime.utcnow().isoformat(),
            })
            st.success("Alert created. Use the check below to evaluate it.")
            st.rerun()

        st.markdown("---")
        st.markdown("**Alert evaluation**")
        st.caption(
            "Alerts are checked on demand against each symbol's current report "
            "(reusing the 5-minute ticker cache where possible). A fired alert "
            f"re-arms after a **{engine.ALERT_COOLDOWN_HOURS:.0f}-hour cooldown** so a "
            "persistent condition notifies once per window, not on every check.")
        if st.button("Check alerts now", key="dp_alert_check", type="primary"):
            with st.spinner("Evaluating alerts against current reports…"):
                res = engine.evaluate_alerts()
            fired = res.get("fired", [])
            if fired:
                for f in fired:
                    st.success(
                        f"**{f['symbol']}** · {f['kind']} — {f['message']}")
            else:
                st.info(
                    f"No alerts fired ({res.get('checked', 0)} enabled alert(s) "
                    "checked, within cooldown where applicable).")
            if res.get("errors"):
                st.warning(
                    f"{len(res['errors'])} symbol(s) could not be analyzed this check:")
                st.json(res["errors"])

    st.markdown("---")
    st.caption("Alerts and watchlists persist locally in dark_pool_state.json. "
               "Alerts are descriptive watchdogs — they never place trades.")


# --------------------------------------------------------------------------- #
#  13. Data Quality Center
# --------------------------------------------------------------------------- #


def _tab_data_quality(engine, mode: str) -> None:
    section_header("Data Quality Center")

    dq = engine.data_quality_report()

    q1, q2 = st.columns([1, 3])
    with q1:
        st.markdown(
            f"<div style='text-align:center;background:#161b22;border:1px solid {GOLD};"
            f"border-radius:12px;padding:24px;'>"
            f"<div style='color:#aaa;font-size:0.72rem;letter-spacing:1.5px;'>DATA QUALITY SCORE</div>"
            f"<div style='font-size:2.6rem;font-weight:700;color:{GOLD};'>"
            f"{dq.get('overall_score', 0):.0f}</div>"
            f"<div style='color:#a0a8b8;font-size:0.75rem;'>/ 100</div>"
            f"</div>", unsafe_allow_html=True)
    with q2:
        st.markdown("**Source status**")
        for s in dq.get("sources", []):
            ok = s.get("observed")
            color = DARK_GREEN if ok else AMBER
            st.markdown(
                f"<div style='background:#161b22;border-left:4px solid {color};"
                f"border-radius:6px;padding:10px 14px;margin:4px 0;'>"
                f"<b style='color:{color};'>{s.get('name')}</b> — {s.get('status')}<br>"
                f"<span style='color:#a0a8b8;font-size:0.78rem;'>{s.get('notes', '')}</span>"
                f"</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Component confidence scores**")
    comp = dq.get("component_scores", [])
    if comp:
        cdf = pd.DataFrame(comp)
        fig = go.Figure(go.Bar(
            x=cdf["component"], y=cdf["score"],
            marker_color=[DARK_GREEN if v >= 80 else GOLD if v >= 60 else AMBER
                          for v in cdf["score"]],
            text=[f"{v:.0f}" for v in cdf["score"]], textposition="outside"))
        fig.update_layout(yaxis_range=[0, 100], yaxis_title="Confidence /100",
                          title="Metric Confidence by Component")
        st.plotly_chart(_layout(fig, 320), width="stretch")

    st.markdown("---")
    st.markdown("**Integrity principles enforced by this dashboard**")
    for line in [
        "Observed data (consolidated OHLCV) is never mixed silently with modeled values.",
        "Modeled off-exchange figures are always labeled and carry confidence.",
        "Inferred direction / institutional patterns are labeled INFERENCE.",
        "FINRA OTC Transparency is attempted first; unavailability is displayed, not hidden.",
        "Raw observations are never overwritten — the engine records what it used.",
        "Statistics report sample sizes and separate train/test results.",
    ]:
        st.markdown(f"- {line}")

    if mode == "Institutional":
        st.markdown("---")
        st.json(dq)


# --------------------------------------------------------------------------- #
#  14. Methodology
# --------------------------------------------------------------------------- #


def _tab_methodology(engine, mode: str) -> None:
    section_header("Methodology")
    st.code(engine.methodology(), language=None)

    st.markdown("---")
    st.markdown("**Definitions**")
    defs = [
        ("Off-exchange volume", "Volume executed away from lit exchanges (ATS + "
         "internalisation). Modeled from consolidated volume using sector baselines."),
        ("Off-exchange %", "Off-exchange volume ÷ consolidated volume (20d)."),
        ("Imbalance", "Estimated (buy − sell) ÷ (buy + sell) of modeled off-exchange "
         "flow over 5 sessions. Heuristic, not prints."),
        ("Pressure score", "0–100 composite of imbalance, off-exchange share vs baseline "
         "and relative volume."),
        ("Percentile", "Share of trailing observations below the current value — "
         "computed on each name's own history."),
        ("Z-score", "Standard deviations from the name's own trailing mean."),
        ("Block volume", "Modeled share of off-exchange volume allocated to "
         "institutional-sized prints (≥10k shares)."),
    ]
    for k, v in defs:
        st.markdown(f"**{k}** — {v}")


# --------------------------------------------------------------------------- #
#  15. Settings & API
# --------------------------------------------------------------------------- #


def _tab_settings(engine, mode: str) -> None:
    section_header("Settings & API")

    st.caption(
        "Configure the authoritative data path and inspect provider state. "
        "The FINRA OTC API key is stored locally (dark_pool_state.json) or read "
        "from the FINRA_API_KEY environment variable — it is never displayed "
        "back in full.")

    s1, s2 = st.columns(2)
    with s1:
        st.markdown("**FINRA OTC Transparency API**")
        settings = engine.get_settings()
        has_key = settings.get("finra_api_key_set", False)
        masked = settings.get("finra_api_key_masked", "")
        st.markdown(
            f"Key configured: {'YES' if has_key else 'NO'}"
            f"{' (from environment)' if settings.get('finra_api_key_env') else ''}"
            + (f" — {masked}" if masked else ""))
        key_input = st.text_input(
            "FINRA OTC API key",
            type="password",
            key="dp_set_finra_key",
            help="Free registration: https://developer.finra.org. Leave blank to keep the "
                 "existing key. The key is stored locally in dark_pool_state.json "
                 "(gitignored) and is never displayed back in full.",
        )
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Save key", key="dp_set_finra_save", type="primary"):
                if key_input and key_input.strip():
                    engine.set_finra_api_key(key_input.strip())
                    st.success("FINRA API key saved.")
                    st.rerun()
        with b2:
            if st.button("Clear key", key="dp_set_finra_clear"):
                engine.set_finra_api_key("")
                st.info("FINRA API key cleared. Falling back to modeled estimates.")
                st.rerun()
        if settings.get("finra_cache_ttl_hours"):
            st.caption(f"FINRA cache TTL: {settings['finra_cache_ttl_hours']:.0f} hours "
                       f"(refetched at most once per TTL window regardless of scan size).")

    with s2:
        st.markdown("**Provider status**")
        dq = engine.data_quality_report()
        for s in dq.get("sources", [])[:2]:
            color = DARK_GREEN if s.get("observed") else AMBER
            st.markdown(
                f"<div style='background:#161b22;border-left:4px solid {color};"
                f"border-radius:6px;padding:10px 14px;margin:4px 0;'>"
                f"<b style='color:{color};'>{s.get('name')}</b> — {s.get('status')}"
                f"</div>", unsafe_allow_html=True)
        st.markdown("**Cache management**")
        if st.button("Clear dark-pool caches", key="dp_set_clear_cache"):
            engine.clear_caches()
            _TICKER_REPORT_CACHE.clear()
            # Scoped: only clear this feature's cached functions, never the
            # whole app's cache (other dashboards keep their warm data).
            for fn in (globals().get("_overview_scan"), globals().get("_sector_table")):
                try:
                    if fn is not None:
                        fn.clear()
                except Exception:
                    pass
            st.success("Dark-pool engine + dashboard caches cleared.")
            st.rerun()

    st.markdown("---")
    st.markdown("**Data source hierarchy**")
    st.markdown(
        "1. **Primary**: FINRA OTC Transparency (authoritative off-exchange volume) — API "
        "key or public CDN files, best-effort.\n"
        "2. **Secondary**: Consolidated OHLCV (Yahoo Finance primary, with Stooq / "
        "Polygon / Alpha Vantage fallbacks in data_sources.py).\n"
        "3. **Validation / fallback**: modeled off-exchange share from sector baselines + "
        "volume-trend, clearly labeled MODELED with confidence.\n"
        "When providers disagree, the engine records both, selects the canonical value "
        "and flags unresolved discrepancies — raw observations are never overwritten.")

    if mode == "Institutional":
        st.markdown("---")
        st.markdown("**Settings (API key masked)**")
        st.json(settings)


# --------------------------------------------------------------------------- #
#  Public entry point
# --------------------------------------------------------------------------- #


def render_dark_pool_dashboard() -> None:
    show_dark_pool_dashboard()
