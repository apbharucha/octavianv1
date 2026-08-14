"""
Algorithm Builder — Streamlit UI.

Two build modes:
  * Auto     — describe what you want in plain language (or nothing at all) and
               the engine decides the strategy families, parameters, and (if
               requested) an ensemble composition.
  * Guided   — pick the strategy families, universe, risk profile, direction,
               and advanced execution settings explicitly.

Every generated algorithm is backtested with a train/test split, honest
costs/fills, and a provenance label showing which research source inspired it.
Educational use only — nothing here is financial advice.
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd
import plotly.graph_objs as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algorithm_builder_engine import (  # noqa: E402
    ALL_ARCHETYPE_NAMES,
    ARCHETYPES,
    RISK_PROFILES,
    AlgorithmResult,
    build_algorithms,
    build_request_signature,
)

GOLD = "#d4af37"
BG = "#0d1117"
CARD = "#161b22"
GREEN = "#3fb950"
RED = "#ff5555"
BLUE = "#58a6ff"
MUTED = "#8b949e"


def _card(html: str) -> None:
    st.markdown(
        f"<div style='background:{CARD};border:1px solid #30363d;border-radius:8px;"
        f"padding:14px 16px;margin:6px 0;'>{html}</div>",
        unsafe_allow_html=True,
    )


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _equity_fig(result: AlgorithmResult) -> go.Figure:
    eq = result.equity
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Strategy",
                             line=dict(color=GOLD, width=1.8)))
    # buy & hold baseline
    if result.returns.index is not None and len(result.returns) > 1:
        import numpy as np
        bh = 100_000.0 * np.cumprod(1 + result.returns.fillna(0.0).to_numpy())
        fig.add_trace(go.Scatter(x=eq.index, y=bh, name="Buy & Hold",
                                 line=dict(color=BLUE, width=1.2, dash="dot")))
    fig.update_layout(
        height=300, margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text="Equity Curve (100k start)", font=dict(size=13)),
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color="#c9d1d9"), legend=dict(orientation="h", y=1.1),
        xaxis=dict(gridcolor="#21262d", title=""), yaxis=dict(gridcolor="#21262d", title="Equity"),
    )
    return fig


def _metric_row(m: dict) -> None:
    c = st.columns(6)
    c[0].metric("Return", _pct(m.get("total_return", 0.0)))
    c[1].metric("Sharpe", f"{m.get('sharpe', 0.0):.2f}")
    c[2].metric("Max DD", _pct(m.get("max_drawdown", 0.0)))
    c[3].metric("Win Rate", _pct(m.get("win_rate", 0.0)))
    c[4].metric("Trades", f"{m.get('trades', 0)}")
    c[5].metric("Exposure", _pct(m.get("exposure", 0.0)))


def _render_result(r: AlgorithmResult, idx: int, expanded: bool = False) -> None:
    is_error = any(n.startswith("ERROR:") for n in r.build_notes)
    title = f"{idx}. {r.name}" + (" ⚠️" if is_error else "")
    with st.expander(title, expanded=expanded):
        if is_error:
            for n in r.build_notes:
                st.warning(n)
            return
        _metric_row(r.metrics)
        if r.train_metrics and r.test_metrics:
            tm, om = r.train_metrics, r.test_metrics
            st.caption(
                f"Train Sharpe {tm.get('sharpe', 0):.2f} (DD {tm.get('max_drawdown', 0) * 100:.1f}%) → "
                f"**Test (OOS) Sharpe {om.get('sharpe', 0):.2f}** (DD {om.get('max_drawdown', 0) * 100:.1f}%, "
                f"{om.get('trades', 0)} trades) — parameters were chosen on the train window only."
            )
        st.plotly_chart(_equity_fig(r), use_container_width=True)
        if r.ensemble_weights:
            _card("<b>Ensemble weights (Fast Universalization)</b><br>" +
                  "<br>".join(f"• {k}: {v * 100:.1f}%" for k, v in r.ensemble_weights.items()))
        if r.family == "portfolio" and r.params:
            _card(f"<b>Final portfolio weights</b><br>" +
                  "<br>".join(f"• {k}: {v * 100:.1f}%" for k, v in r.params.items()
                              if isinstance(v, (int, float))))
        if r.provenance:
            st.markdown(f"<div style='font-size:0.8rem;color:{MUTED};'>📚 <b>Provenance:</b> {r.provenance}</div>",
                        unsafe_allow_html=True)
        if r.build_notes:
            for n in r.build_notes:
                if not n.startswith("ERROR:"):
                    st.caption(n)
        with st.expander("Parameters"):
            st.json({"strategy": r.params, "execution": r.backtest_params,
                     "universe": r.universe})
        with st.expander("Generated Python code"):
            st.code(r.code(), language="python")
        st.markdown("**Export**")
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in r.name)
        ec1, ec2, ec3, ec4 = st.columns(4)
        ec1.download_button("⬇ Python", r.code(), f"{safe}.py",
                            mime="text/x-python", key=f"py_{r.id}")
        ec2.download_button("⬇ JSON spec", r.to_json(), f"{safe}.json",
                            mime="application/json", key=f"json_{r.id}")
        ec3.download_button("⬇ Equity CSV", r.to_csv(), f"{safe}_equity.csv",
                            mime="text/csv", key=f"csv_{r.id}")
        ec4.download_button("⬇ Research note", r.to_markdown(), f"{safe}_note.md",
                            mime="text/markdown", key=f"md_{r.id}")


def _render_methodology() -> None:
    with st.expander("📚 Research & methodology (what powers this builder)"):
        st.markdown("""
**Strategy families are grounded in published research and live-tested community algorithms:**

* **Online portfolio selection** — the Glucksman Fellowship paper (NYU Stern, *Online
  Quantitative Trading Strategies*): FTRL (Sharpe ≈ 1.04), CWMR (≈ 1.75), PAMR (≈ 1.63),
  OLMAR, RMR, Anticor, and **Fast-Universalization ensembles that beat every single
  strategy** when CWMR + FTRL + PAMR are combined. Implemented here exactly in that
  online-learning formulation (weights rebalanced before each bar's return is observed).
* **Mean reversion** — RSI(2) + low-volatility filter ("buy low RSI, low volatility"
  from the QuantConnect community, e.g. *The Alpha Formula*), Bollinger-band fading,
  and z-score statistical arbitrage (the cross-sectional Citadel-style playbook,
  approximated on a single series).
* **Trend / momentum** — dual moving-average crossover with hysteresis, Donchian/Turtle
  breakouts (QuantConnect strategy library), and dual momentum with an absolute-momentum
  gate (Antonacci, popularized on the QuantConnect forum).
* **Liquidity** — inventory-skewed two-sided market making (*fair value + inventory
  skew + spread tuned by risk*, the Optiver / Jane Street playbook). Daily OHLCV can
  only approximate spread capture — the label says so and production use would need
  L1/L2 order-book data.
* **Volatility** — volatility targeting / risk scaling, the risk engine of risk-parity
  mandates.

**How results stay honest:**

* Parameters are chosen by random search on a **train window** (first 75%) and reported
  on a **held-out test window** (last 25%) — the test numbers are out-of-sample.
* Backtests charge commission + slippage, fill stops at the *next open* when a gap
  blows through them, and cap leverage / position size to the risk profile you choose.
* Every algorithm carries a **provenance label**; ensembles reweight members by
  cumulative wealth each bar (no look-ahead).
* Everything is seeded and reproducible: the same inputs produce the same algorithms.

**Limits:** past performance ≠ future results; synthetic fills; no live order routing;
not financial advice.
""", unsafe_allow_html=True)


def _build(build_args: dict, progress) -> list:
    """Run a build with a progress bar (fetch -> search -> ensemble)."""
    n = max(build_args["count"], 1)
    progress.progress(0.05, text="Fetching market data…")
    results = build_algorithms(**build_args)
    progress.progress(0.6, text=f"Searching parameter space ({n} families × trials)…")
    progress.progress(0.9, text="Running honest out-of-sample validation…")
    progress.progress(1.0, text="Done.")
    return results


def render_algorithm_builder() -> None:
    st.markdown(
        f"<h2 style='margin-bottom:2px;'>🤖 Algorithm Builder</h2>"
        f"<div style='color:{MUTED};font-size:0.9rem;'>Research-grounded strategy "
        f"generator — build, backtest (train/test split), and export trading algorithms "
        f"from a vague idea or explicit constraints.</div>",
        unsafe_allow_html=True,
    )
    st.caption("Educational research tool. Backtested results are not financial advice "
               "and do not guarantee future performance.")

    mode = st.radio("Build mode", ["Auto — let the engine decide", "Guided — I pick the pieces"],
                    horizontal=True, label_visibility="collapsed")
    auto = mode.startswith("Auto")

    st.markdown("---")
    if auto:
        st.markdown("#### 🧠 Describe what you want")
        request = st.text_area(
            "Describe your strategy in plain language — the more you say, the more it uses.",
            placeholder="e.g. 'mean reversion on low-volatility stocks with tight stops' or "
                        "'turtle-style breakout momentum with a volatility filter' — or leave "
                        "blank and let the engine choose a diverse mix for your risk profile.",
            height=90)
        if not request.strip():
            st.caption("Hint: try *'market making with inventory control'*, *'trend following "
                       "with a regime filter'*, *'online portfolio selection from the Glucksman "
                       "paper'*, or *'an ensemble of everything'*.")
        c1, c2, c3, c4 = st.columns(4)
        count = c1.slider("Number of algorithms", 1, 6, 3)
        ensemble = c2.checkbox("Build an ensemble of them too", value=True)
        risk = c3.selectbox("Risk profile", RISK_PROFILES,
                            index=1,
                            help="Conservative = small size, long-only, tight stops. "
                                 "Aggressive = leverage up to 2x, long/short, wider stops.")
        direction = c4.selectbox("Direction", ["long_only", "long_short"])
        uni_text = st.text_input("Universe (comma-separated tickers — needs ≥2 for portfolio-selection algorithms)",
                                 value="SPY, QQQ, IWM", help="First symbol is the primary instrument for "
                                                             "single-asset strategies. Add ≥2 for the "
                                                             "online-portfolio-selection family.")
        universe = [s.strip() for s in uni_text.split(",") if s.strip()][:6]
        archetypes = None
    else:
        st.markdown("#### 🛠️ Guided configuration")
        g1, g2 = st.columns(2)
        with g1:
            fam_options = [ARCHETYPES[n]["label"] for n in ALL_ARCHETYPE_NAMES if n != "online_ops"]
            fam_map = {ARCHETYPES[n]["label"]: n for n in ALL_ARCHETYPE_NAMES if n != "online_ops"}
            fam_map["Online Portfolio Selection (PAMR/CWMR/FTRL/OLMAR/Anticor)"] = "online_ops"
            fam_options.append("Online Portfolio Selection (PAMR/CWMR/FTRL/OLMAR/Anticor)")
            chosen = st.multiselect("Strategy families", fam_options,
                                    default=[fam_options[0], fam_options[3], fam_options[8]],
                                    help="Each family is grounded in published research / "
                                         "QuantConnect community algorithms — see methodology below.")
            archetypes = [fam_map[c] for c in chosen]
            direction = st.selectbox("Direction", ["long_only", "long_short"])
        with g2:
            risk = st.selectbox("Risk profile", RISK_PROFILES, index=1)
            count = st.slider("Number of algorithms", 1, 6, 3)
            ensemble = st.checkbox("Build an ensemble of them too", value=True)
            uni_text = st.text_input("Universe", value="SPY, QQQ, IWM")
            universe = [s.strip() for s in uni_text.split(",") if s.strip()][:6]
        with st.expander("Advanced execution settings"):
            a1, a2, a3 = st.columns(3)
            stop = a1.number_input("Stop loss %", 0.0, 30.0, 6.0, 0.5,
                                   help="0 = no stop. Fills at next open if a gap blows through.")
            trail = a2.number_input("Trailing stop %", 0.0, 40.0, 8.0, 0.5, help="0 = off")
            max_hold = a3.number_input("Max hold (bars)", 0, 500, 0, step=5, help="0 = no limit")
            a4, a5, a6 = st.columns(3)
            comm = a4.number_input("Commission (bps)", 0.0, 50.0, 3.0, 0.5)
            slippage = a5.number_input("Slippage (bps)", 0.0, 50.0, 5.0, 0.5)
            trials = a6.number_input("Search trials per family", 5, 100, 25, 5,
                                     help="More trials = better fit, slower build.")
            request = ""

    st.markdown("---")
    seed = st.number_input("Reproducibility seed", 0, 999999, 42, 1,
                           help="Same inputs + same seed = identical algorithms.", key="ab_seed")
    locked = {"stop_loss_pct": None}  # execution settings pass through backtest_params, not strategy params

    if st.button("🚀 Build algorithms", type="primary", use_container_width=True):
        build_args = dict(
            request=request or "", mode="auto" if auto else "guided",
            count=int(count), ensemble=bool(ensemble), risk=str(risk),
            direction=str(direction), universe=universe,
            archetypes=archetypes, locked_params={},
            n_trials=int(trials) if not auto else 25,
            seed=int(seed),
        )
        # advanced execution overrides (guided only)
        if not auto:
            bp_override = {}
            if stop > 0:
                bp_override["stop_loss_pct"] = float(stop)
            if trail > 0:
                bp_override["trailing_pct"] = float(trail)
            if max_hold > 0:
                bp_override["max_hold_bars"] = int(max_hold)
            if comm > 0:
                bp_override["commission_bps"] = float(comm)
            if slippage > 0:
                bp_override["slippage_bps"] = float(slippage)
            bp_override["n_trials"] = int(trials)
            build_args["backtest_params_override"] = bp_override

        sig = build_request_signature(**{k: build_args[k] for k in
                                         ("request", "mode", "count", "ensemble", "risk",
                                          "direction", "universe", "archetypes", "locked_params",
                                          "n_trials", "seed")})
        if st.session_state.get("ab_sig") != sig:
            progress = st.progress(0.0, text="Starting…")
            try:
                results = _build(build_args, progress)
                st.session_state["ab_results"] = results
                st.session_state["ab_sig"] = sig
                st.session_state["ab_args"] = build_args
            except Exception as e:  # noqa: BLE001
                st.error(f"Build failed: {e}")
                return

    if st.session_state.get("ab_results"):
        results: list = st.session_state["ab_results"]
        n_ok = sum(1 for r in results if not any(x.startswith("ERROR:") for x in r.build_notes))
        st.markdown(
            f"<div style='color:{MUTED};font-size:0.85rem;margin:8px 0 4px;'>"
            f"Built {len(results)} algorithm(s) ({n_ok} healthy) — "
            f"seed {st.session_state['ab_args'].get('seed', 42)}, "
            f"universe {', '.join(st.session_state['ab_args'].get('universe', []))}</div>",
            unsafe_allow_html=True)
        for i, r in enumerate(results, 1):
            _render_result(r, i, expanded=(i == 1 and not any(x.startswith("ERROR:") for x in r.build_notes)))
        # combined research note for the whole build
        st.markdown("---")
        st.markdown("#### 📦 Export everything")
        combined = ["# Algorithm Builder — Research Bundle", ""]
        for i, r in enumerate(results, 1):
            combined.append(f"\n---\n\n{r.to_markdown(include_code=False)}")
        combined.append("\n---\n*Generated by Octavian Algorithm Builder. Educational use only — not financial advice.*")
        bundle = "\n".join(combined)
        st.download_button("⬇ Full research bundle (Markdown)", bundle,
                           "algorithm_builder_bundle.md", mime="text/markdown")
        summary_rows = []
        for r in results:
            m = r.metrics
            summary_rows.append({
                "Algorithm": r.name, "Family": r.family,
                "Return": _pct(m.get("total_return", 0)), "Sharpe": round(m.get("sharpe", 0), 2),
                "Max DD": _pct(m.get("max_drawdown", 0)), "Trades": m.get("trades", 0),
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Configure your build above, then hit **Build algorithms**. "
                "Auto mode can work from a blank description.")

    st.markdown("---")
    _render_methodology()
