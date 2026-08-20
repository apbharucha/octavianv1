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


def _swot(r: AlgorithmResult) -> tuple:
    """Strengths / weaknesses / unknowns narrative from the backtest metrics."""
    m = r.metrics
    strengths, weaknesses, unknowns = [], [], []
    if m.get("sharpe", 0) > 1.0:
        strengths.append(f"strong risk-adjusted return (Sharpe {m['sharpe']:.2f})")
    if m.get("win_rate", 0) >= 0.5:
        strengths.append(f"win rate {m['win_rate'] * 100:.0f}%")
    if m.get("max_drawdown", 0) > -0.15:
        strengths.append(f"shallow drawdowns (max DD {m['max_drawdown'] * 100:.1f}%)")
    if (r.test_metrics or {}).get("sharpe", 0) > 0.5:
        strengths.append(f"positive out-of-sample Sharpe "
                         f"{(r.test_metrics or {}).get('sharpe', 0):.2f}")
    if not strengths:
        strengths.append("no standout positive metric — worth testing live at small size")
    if m.get("max_drawdown", 0) < -0.25:
        weaknesses.append(f"deep drawdowns (max DD {m['max_drawdown'] * 100:.1f}%)")
    if m.get("win_rate", 0) < 0.4 and m.get("trades", 0) >= 5:
        weaknesses.append(f"low win rate ({m['win_rate'] * 100:.0f}%) — relies on large winners")
    if m.get("trades", 0) < 8:
        weaknesses.append(f"few trades ({m['trades']}) — limited evidence")
    if (r.test_metrics or {}).get("sharpe", 0) < (r.train_metrics or {}).get("sharpe", 0) - 0.5:
        weaknesses.append("out-of-sample Sharpe well below train — possible overfit")
    if m.get("trades", 0) < 20:
        unknowns.append("small trade sample — metrics may not persist out of sample")
    if not r.window_returns or all(v is None for v in r.window_returns.values()):
        unknowns.append("history too short to judge multi-year horizon stability")
    elif r.window_returns.get("5y") is None:
        unknowns.append("no 5-year window yet — longer-term behavior unknown")
    if m.get("trades", 0) == 0:
        unknowns.append("no discrete trades to reason about (portfolio/rebalance style)")
    if not unknowns:
        unknowns.append("regime changes outside the tested window are always unknown")
    return strengths, weaknesses, unknowns


def _render_window_breakdown(r: AlgorithmResult) -> None:
    """Per-window table (5y/3y/2y/1y/6m/3m/1m): return, Sharpe, max DD, trades."""
    rows = []
    for label in ("5y", "3y", "2y", "1y", "6m", "3m", "1m"):
        w = r.window_metrics.get(label)
        if not isinstance(w, dict):
            rows.append({"Window": label, "Return": "—", "Sharpe": "—",
                         "Max DD": "—", "Trades": "—", "Bars": "—"})
            continue
        rows.append({
            "Window": label,
            "Return": f"{w.get('total_return', 0) * 100:+.1f}%",
            "Sharpe": f"{w.get('sharpe', 0):.2f}",
            "Max DD": f"{w.get('max_drawdown', 0) * 100:.1f}%",
            "Trades": f"{w.get('trades', 0)}",
            "Bars": f"{w.get('bars', 0)}",
        })
    st.markdown("**Return / risk by time window**")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("Every window uses the same formula as the headline metrics: "
               "mean/std of the window's daily returns \u00d7 \u221a252, max DD on the "
               "trailing equity. Short windows (1m/3m/6m) hold very few trades, so "
               "their Sharpe is a small-sample estimate and can sit above the "
               "full-period Sharpe \u2014 the headline period also includes the "
               "pre-first-trade cash stretch and any early chop, which drags its "
               "denominator. Bars = trading days in the window.")


def _render_trade_table(r: AlgorithmResult) -> None:
    """Trade-by-trade table with best / worst / riskiest highlights. Shows the
    symbol + asset type of every trade (and source strategy + blend weight for
    the ensemble's aggregated constituent trades)."""
    closed = [t for t in r.trades if t.get("exit_pnl_pct") is not None]
    if not closed:
        return
    df = pd.DataFrame(closed)
    df = df.rename(columns={
        "entry_date": "Entry", "exit_date": "Exit", "direction": "Side",
        "entry_price": "Entry Px", "exit_pnl_pct": "P&L %",
        "hold_bars": "Hold (bars)", "exit_reason": "Exit reason",
    })
    df = df.rename(columns={"symbol": "Symbol", "asset_type": "Asset type",
                            "source": "Source strategy", "weight": "Weight"})
    has_sym = "Symbol" in df.columns and bool(df["Symbol"].notna().any())
    has_src = "Source strategy" in df.columns and bool(df["Source strategy"].notna().any())
    pre = [c for c in ("Symbol", "Asset type") if has_sym and c in df.columns]
    post = [c for c in ("Source strategy", "Weight") if has_src and c in df.columns]
    cols = pre + [c for c in ("Entry", "Exit", "Side", "Entry Px", "P&L %",
                              "Hold (bars)", "Exit reason") if c in df.columns] + post
    df = df[cols]
    sym_col = df["Symbol"] if "Symbol" in df.columns else None

    def _who(i):
        return f"{sym_col.iloc[i]} " if sym_col is not None else ""

    pnl_num = df["P&L %"].astype(float)
    hold_num = df["Hold (bars)"].astype(float).clip(lower=1.0)
    df["P&L %"] = pnl_num.map(lambda v: f"{v:+.2f}%")
    best_idx = int(pnl_num.idxmax())
    worst_idx = int(pnl_num.idxmin())
    risk_eff = pnl_num / hold_num
    risky_idx = int(risk_eff.idxmin())
    with st.expander(f"Trades ({len(closed)} closed)", expanded=False):
        if len(closed) >= 3:
            best, worst = df.loc[best_idx], df.loc[worst_idx]
            risky = df.loc[risky_idx]
            st.markdown(
                f"<span style='color:{GREEN};'>▲ Best: {_who(best_idx)}{best['Entry']} → "
                f"{best['Exit']} ({best['Side']}) {best['P&L %']} "
                f"via {best['Exit reason']}</span> · "
                f"<span style='color:{RED};'>▼ Worst: {_who(worst_idx)}{worst['Entry']} → "
                f"{worst['Exit']} ({worst['Side']}) "
                f"{worst['P&L %']} via {worst['Exit reason']}</span>",
                unsafe_allow_html=True)
            st.markdown(
                f"<span style='color:{MUTED};'>⚠ Riskiest: {_who(risky_idx)}{risky['Entry']} → "
                f"{risky['Exit']} ({risky['Side']}) {risky['P&L %']} over {risky['Hold (bars)']} "
                f"bars ({risk_eff.loc[risky_idx]:+.2f}%/bar)</span>",
                unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, hide_index=True)
        if has_src:
            st.caption("Ensemble trades are the aggregated constituent trades (each "
                       "labeled with its source strategy and the strategy's blend weight); "
                       "returns come from the blended curve, win rate/count from the "
                       "constituent union.")


def _render_factor_ranking(universe: list) -> None:
    """Cross-sectional factor ranking of the build universe — momentum, value
    proxy, low-vol, trend, volume momentum (WorldQuant smart-beta methodology)."""
    from algorithm_builder_engine import rank_factors
    from data_sources import get_stock

    uni = [u for u in (universe or []) if u][:10]
    if len(uni) < 2:
        return
    with st.expander("Factor ranking of the universe (which symbols are rich in which factor)",
                     expanded=False):
        try:
            dfs = {}
            for sym in uni:
                try:
                    df = get_stock(sym, period="1y")
                    if df is not None and len(df) >= 63:
                        dfs[sym] = df
                except Exception:
                    continue
            if len(dfs) < 2:
                st.info("Need at least 2 symbols with 1y of history to rank factors.")
                return
            fr = rank_factors(dfs)
            cols = {
                "symbol": "Symbol", "mom_12_2": "Momentum (12-2m)",
                "value_proxy": "Value proxy", "low_vol": "Low-vol (neg vol)",
                "trend": "Trend", "volume_mom": "Volume mom",
                "composite": "Composite", "rank": "Rank",
            }
            fr = fr.rename(columns=cols)[list(cols.values())]
            st.dataframe(fr, use_container_width=True, hide_index=True)
            st.caption("Factors are z-scored cross-sectionally and equal-weight averaged into "
                       "a composite (WorldQuant retail smart-beta methodology). Momentum is "
                       "the Asness MOM2-12; value is a price proxy (depth below the trailing "
                       "1y average) since the engine has no fundamentals.")
        except Exception as e:  # noqa: BLE001
            st.info(f"Could not fetch universe data for factor ranking: {e}")


def _render_result(r: AlgorithmResult, idx: int, expanded: bool = False) -> None:
    is_error = any(n.startswith("ERROR:") for n in r.build_notes)
    title = f"{idx}. {r.name}" + (" (error)" if is_error else "")
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
        if r.window_metrics:
            _render_window_breakdown(r)
        elif r.window_returns:
            wr = [f"<b>{k}</b>: {v * 100:+.1f}%" if v is not None else f"<b>{k}</b>: —"
                  for k, v in r.window_returns.items()]
            st.markdown(
                f"<div style='color:{MUTED};font-size:0.85rem;'>Return by window — "
                + " · ".join(wr) + "</div>", unsafe_allow_html=True)
        st.plotly_chart(_equity_fig(r), use_container_width=True, key=f"eq_{r.id}")
        if r.family != "ensemble":
            if st.button("Test in paper trading", key=f"pt_deploy_{r.id}",
                         help="Deploy this algorithm to a paper trading account "
                              "(new or existing) and run it with live signals."):
                st.session_state["ab_deploy"] = r.id
                st.session_state["ab_deploy_spec"] = r.strategy_spec()
        if r.ensemble_weights:
            _card("<b>Dynamic ensemble weights (quality-tilted)</b><br>" +
                  "<br>".join(f"• {k}: {v * 100:.1f}%" for k, v in r.ensemble_weights.items()))
        if r.family == "portfolio" and r.params:
            _card(f"<b>Final portfolio weights</b><br>" +
                  "<br>".join(f"• {k}: {v * 100:.1f}%" for k, v in r.params.items()
                              if isinstance(v, (int, float))))
        if r.provenance:
            st.markdown(f"<div style='font-size:0.8rem;color:{MUTED};'><b>Provenance:</b> {r.provenance}</div>",
                        unsafe_allow_html=True)
        if r.build_notes:
            for n in r.build_notes:
                if not n.startswith("ERROR:") and not n.startswith("SUGGESTION:"):
                    st.caption(n)
        _render_trade_table(r)
        if r.trade_narratives:
            with st.expander(f"Trade-by-trade reasoning ({len(r.trade_narratives)} trades)"):
                for n in r.trade_narratives:
                    st.markdown(f"- {n}")
        s, w, u = _swot(r)
        st.markdown("**Strengths / weaknesses / unknowns**")
        st.markdown(" - " + "; ".join(s))
        st.markdown(" - " + "; ".join(w) if w else " - no material weaknesses detected")
        st.markdown(" - " + "; ".join(u))
        with st.expander("Parameters"):
            st.json({"strategy": r.params, "execution": r.backtest_params,
                     "universe": r.universe})
        with st.expander("Generated Python code"):
            st.code(r.code(), language="python")
        st.markdown("**Export**")
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in r.name)
        ec1, ec2, ec3, ec4 = st.columns(4)
        ec1.download_button("Download Python", r.code(), f"{safe}.py",
                            mime="text/x-python", key=f"py_{r.id}")
        ec2.download_button("Download JSON spec", r.to_json(), f"{safe}.json",
                            mime="application/json", key=f"json_{r.id}")
        ec3.download_button("Download Equity CSV", r.to_csv(), f"{safe}_equity.csv",
                            mime="text/csv", key=f"csv_{r.id}")
        ec4.download_button("Download Research note", r.to_markdown(), f"{safe}_note.md",
                            mime="text/markdown", key=f"md_{r.id}")


def _render_methodology() -> None:
    with st.expander("Research & methodology (what powers this builder)"):
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
    runs = int(build_args.get("runs_per_family", 3))
    trials = int(build_args.get("n_trials", 25))
    progress.progress(0.05, text="Fetching 5y market data…")
    results = build_algorithms(**build_args)
    progress.progress(0.6, text=f"Searching parameter space ({n} families × {trials} trials × {runs} runs)…")
    wf = int(build_args.get("wf_folds", 0) or 0)
    wf_txt = f" + walk-forward CV ({wf} folds)" if wf else ""
    progress.progress(0.9, text=f"Running multi-window backtests + per-trade reasoning{wf_txt}…")
    progress.progress(1.0, text="Done.")
    return results


def render_algorithm_builder() -> None:
    st.markdown(
        f"<h2 style='margin-bottom:2px;'>Algorithm Builder</h2>"
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
    wf = 0  # walk-forward CV folds; set by the controls below when enabled

    st.markdown("---")
    if auto:
        st.markdown("#### Describe what you want")
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
        runs = st.slider("Backtests per algorithm (multiple runs = robustness evidence)",
                         1, 5, 3, help="Each family is searched several times with different "
                                       "seeds, so every algorithm gets many backtests.")
        wf = st.slider("Walk-forward CV folds (0 = off)", 0, 6, 4,
                       help="Multi-fold out-of-sample evaluation (Bergmeir & Hyndman 2018): "
                            "each algorithm is re-tested on several trailing windows, not "
                            "just one hold-out, so overfit parameter sets are caught.")
        archetypes = None
    else:
        st.markdown("#### Guided configuration")
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
            a7, a8, a9 = st.columns(3)
            runs = a7.number_input("Backtest runs per family", 1, 5, 3, 1,
                                   help="Each family is searched several times with different "
                                        "seeds — many backtests per algorithm.")
            wf = a8.number_input("Walk-forward CV folds", 0, 6, 4, 1,
                                 help="Multi-fold out-of-sample evaluation (Bergmeir & "
                                      "Hyndman 2018) — re-tests each algorithm on several "
                                      "trailing windows to catch overfitting. 0 = off.")
            request = ""

    st.markdown("---")
    seed = st.number_input("Reproducibility seed", 0, 999999, 42, 1,
                           help="Same inputs + same seed = identical algorithms.", key="ab_seed")
    locked = {"stop_loss_pct": None}  # execution settings pass through backtest_params, not strategy params

    if st.button("Build algorithms", type="primary", use_container_width=True):
        build_args = dict(
            request=request or "", mode="auto" if auto else "guided",
            count=int(count), ensemble=bool(ensemble), risk=str(risk),
            direction=str(direction), universe=universe,
            archetypes=archetypes, locked_params={},
            n_trials=int(trials) if not auto else 25,
            runs_per_family=int(runs),
            seed=int(seed),
            wf_folds=int(wf),
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

        # build_request_signature names the execution-lock dict `locked` while
        # build_algorithms calls it `locked_params` — pass it under the right name.
        sig = build_request_signature(**{k: v for k, v in build_args.items()
                                         if k in ("request", "mode", "count", "ensemble",
                                                  "risk", "direction", "universe",
                                                  "archetypes", "n_trials", "seed",
                                                  "runs_per_family", "wf_folds")},
                                       locked=build_args["locked_params"])
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

        # ---- no-viable-strategy alert: why it failed + what to tweak + alternatives ----
        errs = [r for r in results if any(n.startswith("ERROR:") for n in r.build_notes)]
        healthy = [r for r in results if not any(n.startswith("ERROR:") for n in r.build_notes)]
        if errs:
            with st.expander(
                    f"{len(errs)} strategy famil(y/ies) could not be built — why, and what to tweak",
                    expanded=True):
                for r in errs:
                    st.markdown(f"**{r.name}**")
                    for n in r.build_notes:
                        st.warning(n)
                alts = [r for r in healthy if r.family != "ensemble"]
                if alts:
                    st.markdown("**Alternative strategies that DID work on this data** "
                                "(with their backtests):")
                    for r in alts[:5]:
                        m = r.metrics
                        oos = (r.test_metrics or {}).get("sharpe")
                        st.markdown(
                            f"- **{r.name}**: return {_pct(m.get('total_return', 0))}, "
                            f"Sharpe {m.get('sharpe', 0):.2f}" +
                            (f", OOS Sharpe {oos:.2f}" if oos is not None else "") +
                            f", {m.get('trades', 0)} trades")
                st.caption("Tip: adjust the universe, risk profile, direction, or search trials "
                           "above and rebuild — the engine re-fits everything to your new wants.")

        # ---- which algorithm is best: comparative read + strengths/weaknesses/unknowns ----
        singles = [r for r in healthy if r.family != "ensemble"]
        if singles:
            st.markdown("#### Which algorithm is best?")
            key = lambda r: (r.test_metrics or {}).get("sharpe", r.metrics.get("sharpe", 0.0))
            best = max(singles, key=key)
            worst = min(singles, key=key)
            bm, wm = best.metrics, worst.metrics
            bs, bw, bu = _swot(best)
            st.markdown(
                f"**Best on out-of-sample data: {best.name}** — OOS Sharpe "
                f"{(best.test_metrics or {}).get('sharpe', 0):.2f}, "
                f"return {_pct(bm.get('total_return', 0))}, "
                f"max DD {_pct(bm.get('max_drawdown', 0))}, {bm.get('trades', 0)} trades. "
                f"Strengths: {'; '.join(bs)}. Weaknesses: {'; '.join(bw)}. "
                f"Unknowns: {'; '.join(bu)}.")
            ws_, ww_, wu_ = _swot(worst)
            st.markdown(
                f"**Weakest: {worst.name}** — OOS Sharpe "
                f"{(worst.test_metrics or {}).get('sharpe', 0):.2f}, "
                f"return {_pct(wm.get('total_return', 0))}, "
                f"max DD {_pct(wm.get('max_drawdown', 0))}, {wm.get('trades', 0)} trades. "
                f"Strengths: {'; '.join(ws_)}. Weaknesses: {'; '.join(ww_)}. "
                f"Unknowns: {'; '.join(wu_)}.")
            if len(singles) > 2:
                st.caption("Ranking note: sorted by out-of-sample (test) Sharpe — the number "
                           "that matters most, because it is the only one the search never "
                           "looked at while fitting.")

        # ---- factor ranking of the universe (WorldQuant retail smart-beta
        # methodology: z-score each factor, weighted composite) ----
        _render_factor_ranking(st.session_state["ab_args"].get("universe", []))

        # combined research note for the whole build
        st.markdown("---")
        st.markdown("#### Export everything")
        combined = ["# Algorithm Builder — Research Bundle", ""]
        for i, r in enumerate(results, 1):
            combined.append(f"\n---\n\n{r.to_markdown(include_code=False)}")
        combined.append("\n---\n*Generated by Octavian Algorithm Builder. Educational use only — not financial advice.*")
        bundle = "\n".join(combined)
        st.download_button("Download full research bundle (Markdown)", bundle,
                           "algorithm_builder_bundle.md", mime="text/markdown")
        summary_rows = []
        for r in results:
            m = r.metrics
            oos = (r.test_metrics or {}).get("sharpe")
            wr = r.window_returns or {}
            summary_rows.append({
                "Algorithm": r.name, "Family": r.family,
                "Return": _pct(m.get("total_return", 0)),
                "OOS Sharpe": round(oos, 2) if oos is not None else None,
                "Sharpe": round(m.get("sharpe", 0), 2),
                "Max DD": _pct(m.get("max_drawdown", 0)), "Trades": m.get("trades", 0),
                "5y": _pct(wr["5y"]) if wr.get("5y") is not None else "—",
                "1y": _pct(wr["1y"]) if wr.get("1y") is not None else "—",
                "6m": _pct(wr["6m"]) if wr.get("6m") is not None else "—",
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Configure your build above, then hit **Build algorithms**. "
                "Auto mode can work from a blank description.")

    st.markdown("---")
    _render_deploy_panel(st.session_state.get("ab_results") or [])
    _render_methodology()


def _render_deploy_panel(results: list) -> None:
    """Deploy a built algorithm to paper trading: new account or existing
    portfolio, with an explicit override confirmation when the target account
    already has a strategy in place."""
    deploy_id = st.session_state.get("ab_deploy")
    if not deploy_id:
        return
    r = next((x for x in results if x.id == deploy_id), None)
    if r is None:
        return
    from paper_trading_system import get_paper_trading_system

    spec = r.strategy_spec()
    st.markdown("---")
    st.markdown("#### Test in paper trading")
    m = r.metrics
    _card(f"<b>{r.name}</b> — {r.family} · Sharpe {m.get('sharpe', 0):.2f} · "
          f"return {m.get('total_return', 0) * 100:.1f}% · {m.get('trades', 0)} trades. "
          f"<span style='color:{MUTED};'>Signals are recomputed from this algorithm's "
          f"fitted parameters on live daily data; positions are sized with the same "
          f"risk budget the backtest used.</span>")

    user_id = st.session_state.get("user_id", "default_user")
    pt = get_paper_trading_system()
    accounts = pt.list_accounts(user_id)
    target = st.radio("Deploy to", ["New paper trading account", "Existing portfolio"],
                      horizontal=True, key="ab_deploy_target")

    if target == "New paper trading account":
        name = st.text_input("Account name", value=r.name[:40], key="ab_deploy_name")
        balance = st.number_input("Initial balance ($)", 1000.0, 10_000_000.0,
                                  100_000.0, step=10_000.0, key="ab_deploy_balance")
        if st.button("Create account & deploy", type="primary", key="ab_deploy_create"):
            acc = pt.create_account(user_id, name.strip() or "Algorithm Strategy", float(balance))
            if acc and pt.deploy_strategy(acc.account_id, spec):
                st.success(f"Deployed '{r.name}' to new account '{name.strip() or 'Algorithm Strategy'}'. "
                           "Open the **Paper Trading** tab and enable Automation to start "
                           "running this strategy live.")
                st.session_state["ab_deploy"] = None
            else:
                st.error("Failed to create the account / deploy the strategy.")
    else:
        if not accounts:
            st.info("You don't have any paper trading accounts yet — choose "
                    "'New paper trading account' above to create one.")
            return
        acc_map = {f"{a.account_name} (${a.current_balance:,.2f})": a for a in accounts}
        sel = st.selectbox("Target account", list(acc_map.keys()), key="ab_deploy_account")
        acc = acc_map[sel]
        existing = pt.get_deployed_strategy(acc.account_id)
        override_ok = True
        if existing:
            st.warning(
                f"'{existing.get('name', 'an algorithm')}' is currently deployed on this "
                f"account (Sharpe {existing.get('metrics', {}).get('sharpe', '?')}). "
                "Deploying will replace it.")
            override_ok = st.checkbox("Yes — replace the strategy currently in place",
                                      key="ab_deploy_override")
        if st.button("Deploy to this account", type="primary",
                     key="ab_deploy_existing", disabled=not override_ok):
            if pt.deploy_strategy(acc.account_id, spec):
                st.success(f"Deployed '{r.name}' to account '{acc.account_name}'. "
                           "Open the **Paper Trading** tab and enable Automation to start "
                           "running this strategy live.")
                st.session_state["ab_deploy"] = None
            else:
                st.error("Failed to deploy the strategy to this account.")
