import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Optional, Any
from octavian_theme import COLORS


def _init_widget_defaults(defaults: Dict[str, object]):
    """Seed widget session_state keys only when not already set."""
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _queue_widget_autofill(pending_key: str, widget_values: Dict[str, object], message: str):
    """Store widget updates to apply on the next rerun (before widgets render)."""
    st.session_state[pending_key] = {"widgets": widget_values, "message": message}


def _apply_pending_widget_autofill(pending_key: str) -> Optional[str]:
    """Apply queued widget values before any widgets with those keys are created."""
    payload = st.session_state.pop(pending_key, None)
    if not payload:
        return None
    for key, val in payload.get("widgets", {}).items():
        st.session_state[key] = val
    return payload.get("message")


# ─────────────────────────────────────────────────────────────────────────────
# Shared QA / provenance renderers
# ─────────────────────────────────────────────────────────────────────────────

def _render_qa(qa, module_label: str):
    """Render a Model Integrity / QA dashboard strip."""
    if qa is None:
        return
    color = {"PASS": "#00c853", "WARN": "#e0c97f", "FAIL": "#d32f2f"}.get(qa.status, "#aaa")
    st.markdown(
        f"<div style='background:#161b22;border-left:4px solid {color};border-radius:6px;"
        f"padding:10px 14px;margin:8px 0;'>"
        f"<span style='color:{color};font-weight:700;'>{qa.status} — {module_label} Integrity</span>"
        f"<div style='color:#ccc;font-size:0.8rem;margin-top:4px;'>{qa.summary}</div></div>",
        unsafe_allow_html=True,
    )
    if qa.checks:
        with st.expander("QA check log"):
            for c in qa.checks:
                cc = {"PASS": "#00c853", "WARN": "#e0c97f", "FAIL": "#d32f2f"}.get(c.status, "#aaa")
                st.markdown(
                    f"<span style='color:{cc};font-weight:700;'>{c.status}</span> "
                    f"<b>{c.check}</b> — {c.detail}", unsafe_allow_html=True)


def _render_consensus_dislocation(consensus_layer: Dict[str, Any]):
    """Render the market-consensus / dislocation layer (from DCF results)."""
    if not consensus_layer or "error" in consensus_layer:
        st.info("Consensus layer unavailable for this run.")
        return
    implied = consensus_layer.get("implied_expectations", {})
    disloc = consensus_layer.get("dislocation", {})
    st.markdown("### Market-Implied Expectations (Reverse DCF)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Implied Revenue Growth", f"{implied.get('implied_revenue_growth', 0):.1%}")
    c2.metric("Implied EBIT Margin", f"{implied.get('implied_ebit_margin', 0):.1%}")
    c3.metric("Implied FCF Growth", f"{implied.get('implied_fcf_growth', 0):.1%}")
    c4.metric("Upside vs Market", f"{implied.get('upside_pct', 0):+.1%}")
    st.caption(implied.get("narrative", ""))
    st.markdown("### Consensus Dislocation")
    color = disloc.get("overall_color", "#aaa")
    st.markdown(
        f"<div style='background:#161b22;border-left:4px solid {color};border-radius:6px;"
        f"padding:10px 14px;'><span style='color:{color};font-weight:700;'>"
        f"{disloc.get('overall_reading', '')}</span></div>", unsafe_allow_html=True)
    for f in disloc.get("findings", []):
        fc = {"High": "#ff7043", "Medium": "#e0c97f", "Low": "#aaaaaa"}.get(f.get("severity"), "#aaa")
        st.markdown(
            f"<div style='background:#1a1f2e;border-left:3px solid {fc};border-radius:4px;"
            f"padding:8px 12px;margin:6px 0;'>"
            f"<b style='color:{fc};'>{f.get('severity')}</b> "
            f"<b>{f.get('title')}</b><div style='color:#ccc;font-size:0.82rem;'>{f.get('detail')}</div>"
            f"</div>", unsafe_allow_html=True)


def _render_memo_button(ctx: Any, key: str):
    """Download button for the written investment memo."""
    try:
        from investment_memo import get_memo_generator
        memo_text = get_memo_generator().generate(ctx)
        st.download_button(
            "Download Investment Memo (.md)",
            data=memo_text,
            file_name=f"{ctx.title.replace(' ', '_')}_Investment_Memo.md",
            mime="text/markdown",
            key=key,
        )
    except Exception as e:
        st.warning(f"Memo generator unavailable ({e})")


def _render_pptx_button(label: str, bytes_data: bytes, fname: str, key: str):
    st.download_button(
        label,
        data=bytes_data,
        file_name=fname,
        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        key=key,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def show_financial_generator():
    """Institutional Financial Modeling Engine UI.

    A unified institutional transaction-analysis platform: DCF, LBO, M&A,
    Trading Comps (CCA), Precedent Transactions, IPO Analysis and the
    Cross-Model Valuation Bridge — all interconnected, with market
    consensus/sentiment, model QA and written-memo layers throughout.
    """
    # Defensive import — clears stale .pyc cache on first ImportError then retries
    try:
        from financial_model_generator import (
            DCFAssumptions,
            get_dcf_engine,
            get_financial_generator,
            fetch_ticker_fundamentals,
        )
        _fmg_import_ok = True
    except ImportError:
        import importlib
        import sys
        for _mod in list(sys.modules.keys()):
            if "financial_model_generator" in _mod:
                del sys.modules[_mod]
        try:
            from financial_model_generator import (
                DCFAssumptions,
                get_dcf_engine,
                get_financial_generator,
                fetch_ticker_fundamentals,
            )
            _fmg_import_ok = True
        except Exception as _fmg_err:
            _fmg_import_ok = False
            st.error(
                f"Could not load Financial Model Generator: `{_fmg_err}`\n\n"
                "**Fix:** Stop Streamlit, run `find . -name '*.pyc' -delete`"
                "in the project root, then restart.")
    except Exception as _fmg_err:
        _fmg_import_ok = False
        st.error(f"Financial Model Generator failed to load: `{_fmg_err}`")

    if not _fmg_import_ok:
        st.stop()

    st.title("Institutional Financial Modeling Engine")
    st.caption(
        "Unified transaction-analysis platform: DCF, LBO, M&A, Trading Comps, "
        "Precedent Transactions, IPO — with market consensus, model QA and "
        "cross-model valuation bridge."
    )

    fmg_tabs = st.tabs([
        "DCF Valuation",
        "LBO Model",
        "M&A Accretion / Dilution",
        "Trading Comps (CCA)",
        "Precedent Transactions",
        "IPO Analysis",
        "Cross-Model Valuation Bridge",
    ])

    # ── Tab 0: DCF ───────────────────────────────────────────────────────────
    with fmg_tabs[0]:
        _render_dcf_tab(DCFAssumptions, get_dcf_engine, get_financial_generator,
                        fetch_ticker_fundamentals)

    # ── Tab 1: LBO ───────────────────────────────────────────────────────────
    with fmg_tabs[1]:
        _render_lbo_tab(fetch_ticker_fundamentals)

    # ── Tab 2: M&A ───────────────────────────────────────────────────────────
    with fmg_tabs[2]:
        _render_mna_tab(fetch_ticker_fundamentals)

    # ── Tab 3: Trading Comps ─────────────────────────────────────────────────
    with fmg_tabs[3]:
        _render_comps_tab()

    # ── Tab 4: Precedent Transactions ────────────────────────────────────────
    with fmg_tabs[4]:
        _render_precedents_tab()

    # ── Tab 5: IPO ───────────────────────────────────────────────────────────
    with fmg_tabs[5]:
        _render_ipo_tab()

    # ── Tab 6: Cross-Model Valuation Bridge ──────────────────────────────────
    with fmg_tabs[6]:
        _render_bridge_tab()


# ═════════════════════════════════════════════════════════════════════════════
# DCF TAB
# ═════════════════════════════════════════════════════════════════════════════

def _render_dcf_tab(DCFAssumptions, get_dcf_engine, get_financial_generator,
                    fetch_ticker_fundamentals):
    # Ticker input + auto-fill
    fmg_col_ticker, fmg_col_fetch = st.columns([2, 1])
    with fmg_col_ticker:
        ticker = st.text_input(
            "Ticker Symbol",
            value=st.session_state.get("fmg_ticker", ""),
            placeholder="e.g. MSFT, NVDA, TSLA...",
            help="Enter any ticker. Use 'Auto-Fill from Ticker' to populate assumptions from live data.",
        )
    with fmg_col_fetch:
        st.markdown("<br>", unsafe_allow_html=True)
        do_autofill = st.button(
            "Auto-Fill from Ticker",
            help="Fetches live financials from yfinance and populates all fields automatically.",
        )

    if do_autofill and ticker:
        with st.spinner(f"Fetching financials for {ticker.upper()}…"):
            try:
                data = fetch_ticker_fundamentals(ticker)
                st.session_state["fmg_ticker"] = data["ticker"]
                st.session_state["fmg_price"] = data["price"]
                st.session_state["fmg_rev"] = data["revenue_m"]
                st.session_state["fmg_rev_growth"] = data["revenue_growth"]
                st.session_state["fmg_ebit"] = data["ebit_margin_pct"]
                st.session_state["fmg_tax"] = data["tax_rate_pct"]
                st.session_state["fmg_beta"] = data["beta"]
                st.session_state["fmg_mktcap"] = data["market_cap_m"]
                st.session_state["fmg_debt"] = data["debt_m"]
                st.session_state["fmg_cash"] = data["cash_m"]
                st.session_state["fmg_shares"] = data["shares_m"]
                st.session_state.pop("dcf_result", None)
                st.success(f"Auto-filled assumptions for {data['ticker']} — review and click Run DCF.")
                st.rerun()
            except Exception as _af_err:
                st.warning(f"Auto-fill partial ({_af_err}). Enter assumptions manually.")

    col_input, col_main = st.columns([1, 2])

    with col_input:
        st.subheader("Revenue & Margins")
        current_price = st.number_input(
            "Current Market Price ($)",
            value=float(st.session_state.get("fmg_price", 100.0)),
            min_value=0.01,
        )
        base_revenue = st.number_input(
            "Base Revenue ($M)",
            value=float(st.session_state.get("fmg_rev", 5_000.0)),
            step=100.0,
        )
        rev_growth = st.slider(
            "Revenue Growth Rate (%)", 0.0, 50.0,
            float(st.session_state.get("fmg_rev_growth", 8.0))) / 100
        ebit_margin = st.slider(
            "EBIT Margin (%)", 0.0, 60.0,
            float(st.session_state.get("fmg_ebit", 20.0))) / 100
        tax_rate = st.slider(
            "Tax Rate (%)", 0.0, 40.0,
            float(st.session_state.get("fmg_tax", 21.0))) / 100

        st.subheader("Working Capital & CapEx")
        da_pct = st.slider("D&A (% Revenue)", 0.0, 15.0, 3.0) / 100
        capex_pct = st.slider("CapEx (% Revenue)", 0.0, 20.0, 5.0) / 100
        nwc_pct = st.slider("ΔNWC (% Revenue)", -5.0, 10.0, 1.0) / 100

        st.subheader("WACC / CAPM")
        risk_free = st.slider("Risk-Free Rate (%)", 2.0, 8.0, 4.25) / 100
        erp = st.slider("Equity Risk Premium (%)", 3.0, 8.0, 5.5) / 100
        beta = st.slider("Beta", 0.3, 3.0, float(st.session_state.get("fmg_beta", 1.0)), step=0.05)
        cost_of_debt = st.slider("Cost of Debt (%)", 2.0, 10.0, 4.5) / 100
        equity_val_market = st.number_input(
            "Market Cap ($M)",
            value=float(st.session_state.get("fmg_mktcap", 10_000.0)),
            step=100.0)
        debt_total = st.number_input(
            "Total Debt ($M)",
            value=float(st.session_state.get("fmg_debt", 1_000.0)),
            step=100.0)
        cash = st.number_input(
            "Cash ($M)",
            value=float(st.session_state.get("fmg_cash", 500.0)),
            step=100.0)

        st.subheader("Terminal Value & Shares")
        terminal_growth = st.slider("Terminal Growth Rate (%)", 0.5, 5.0, 2.5) / 100
        shares = st.number_input(
            "Shares Outstanding (M)",
            value=float(st.session_state.get("fmg_shares", 1_000.0)),
            step=10.0)
        proj_years = st.selectbox("Projection Years", [3, 5, 7, 10], index=1)

        st.subheader("Peer Multiples (Relative Valuation)")
        peer_pe = st.number_input("Peer P/E", value=25.0, step=1.0)
        peer_ev_ebitda = st.number_input("Peer EV/EBITDA", value=15.0, step=0.5)
        peer_ev_fcf = st.number_input("Peer EV/FCF", value=20.0, step=0.5)
        peg = st.number_input("PEG Ratio", value=2.0, step=0.1)

        run_model = st.button("Run Institutional DCF", type="primary")

    if run_model:
        assumptions = DCFAssumptions(
            ticker=ticker,
            base_revenue=base_revenue,
            revenue_growth_rates=[rev_growth] * proj_years,
            ebit_margin=ebit_margin,
            tax_rate=tax_rate,
            da_pct_revenue=da_pct,
            capex_pct_revenue=capex_pct,
            nwc_change_pct_revenue=nwc_pct,
            equity_value_market=equity_val_market,
            debt_value=debt_total,
            cost_of_debt=cost_of_debt,
            risk_free_rate=risk_free,
            equity_risk_premium=erp,
            beta=beta,
            terminal_growth_rate=terminal_growth,
            cash=cash,
            shares_outstanding=shares,
            current_price=current_price,
            peer_pe=peer_pe,
            peer_ev_ebitda=peer_ev_ebitda,
            peer_ev_fcf=peer_ev_fcf,
            peg_ratio=peg,
            projection_years=proj_years,
        )
        engine = get_dcf_engine()
        with st.spinner("Running institutional DCF (Monte Carlo 10,000 paths, scenarios, consensus layer)..."):
            result = engine.run_dcf(assumptions)
        st.session_state["dcf_result"] = result
        st.session_state["bridge_context"] = {"dcf": result, "price": current_price,
                                              "ticker": ticker or result.ticker}

    with col_main:
        if "dcf_result" not in st.session_state:
            st.info("Configure assumptions on the left and click **Run Institutional DCF**.")
        else:
            result = st.session_state["dcf_result"]
            sig = result.trade_signal

            # QA strip
            try:
                from model_audit import run_model_qa
                _render_qa(run_model_qa(result, "dcf"), "DCF")
            except Exception:
                pass

            sig_colors = {
                "Strong Long": "#00ff88", "Long": "#7fff7f", "Neutral": "#aaaaaa",
                "Short": "#ff9944", "Strong Short": "#ff4444",
            }
            sig_color = sig_colors.get(sig.signal, "#ffffff")
            st.markdown(
                f"""<div style="background:#161b22;border:1px solid {sig_color};border-radius:8px;
                            padding:12px 16px;margin-bottom:12px;">
                  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
                    <div>
                      <div style="font-size:1.1rem;font-weight:700;color:{sig_color};">{sig.signal.upper()}</div>
                      <div style="color:#aaa;font-size:0.75rem;">{result.ticker} — {sig.risk_level} Risk</div>
                    </div>
                    <div style="text-align:center;">
                      <div style="font-size:1.3rem;font-weight:700;color:white;">${sig.fair_value:.2f}</div>
                      <div style="color:#aaa;font-size:0.7rem;">Fair Value / Share</div>
                    </div>
                    <div style="text-align:center;">
                      <div style="font-size:1.1rem;font-weight:700;color:{sig_color};">{sig.upside_pct:+.1f}%</div>
                      <div style="color:#aaa;font-size:0.7rem;">vs ${sig.market_price:.2f} Mkt</div>
                    </div>
                    <div style="text-align:center;">
                      <div style="font-size:1.1rem;font-weight:700;color:white;">{sig.confidence_pct:.0f}%</div>
                      <div style="color:#aaa;font-size:0.7rem;">Confidence</div>
                    </div>
                  </div>
                </div>""",
                unsafe_allow_html=True,
            )

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Enterprise Value", f"${result.enterprise_value / 1000:.1f}B")
            k2.metric("Equity Value", f"${result.equity_value / 1000:.1f}B")
            k3.metric("WACC", f"{result.wacc:.2%}")
            k4.metric("Cost of Equity", f"{result.cost_of_equity:.2%}")
            k5.metric("Position Size", f"{sig.position_size_pct:.1f}%")

            st.markdown("### Expected Investor Return")
            exp_ret_col1, exp_ret_col2, exp_ret_col3, exp_ret_col4 = st.columns(4)
            with exp_ret_col1:
                bear_irr = result.expected_returns.get('Bear', 0)
                st.markdown(
                    f'<div style="background:#161b22;border:1px solid #ff6666;border-radius:6px;padding:10px;text-align:center;">'
                    f'<div style="color:#aaa;font-size:0.7rem;">BEAR CASE IRR</div>'
                    f'<div style="color:#ff6666;font-size:1.3rem;font-weight:700;">{bear_irr:.1%}</div></div>',
                    unsafe_allow_html=True)
            with exp_ret_col2:
                base_irr = result.expected_returns.get('Base', 0)
                st.markdown(
                    f'<div style="background:#161b22;border:1px solid #e0c97f;border-radius:6px;padding:10px;text-align:center;">'
                    f'<div style="color:#aaa;font-size:0.7rem;">BASE CASE IRR</div>'
                    f'<div style="color:#e0c97f;font-size:1.3rem;font-weight:700;">{base_irr:.1%}</div></div>',
                    unsafe_allow_html=True)
            with exp_ret_col3:
                bull_irr = result.expected_returns.get('Bull', 0)
                st.markdown(
                    f'<div style="background:#161b22;border:1px solid #66ff99;border-radius:6px;padding:10px;text-align:center;">'
                    f'<div style="color:#aaa;font-size:0.7rem;">BULL CASE IRR</div>'
                    f'<div style="color:#66ff99;font-size:1.3rem;font-weight:700;">{bull_irr:.1%}</div></div>',
                    unsafe_allow_html=True)
            with exp_ret_col4:
                weighted_irr = result.probability_weighted_irr
                rec_color = result.recommendation_color
                st.markdown(
                    f'<div style="background:#161b22;border:2px solid {rec_color};border-radius:6px;padding:10px;text-align:center;">'
                    f'<div style="color:#aaa;font-size:0.7rem;">EXPECTED IRR</div>'
                    f'<div style="color:{rec_color};font-size:1.3rem;font-weight:800;">{weighted_irr:.1%}</div>'
                    f'<div style="color:{rec_color};font-size:0.65rem;font-weight:600;margin-top:4px;">{result.investment_recommendation}</div></div>',
                    unsafe_allow_html=True)

            st.markdown("---")

            dcf_tabs = st.tabs([
                "20-Line DCF", "Historical Anchoring", "Reverse DCF", "Scenarios",
                "Monte Carlo", "Relative Valuation", "Sensitivity", "Catalysts",
                "WACC Detail", "Key Drivers", "Consensus & Dislocation",
            ])

            with dcf_tabs[0]:
                st.subheader("Investment Banking 20-Line DCF Template")
                st.caption("All values in $M. Discount Factor = 1/(1+WACC)^t — decreases each year (correct PV factor).")
                fmt_cols = {c: "{:,.1f}" for c in result.line_items.columns if c != "Discount Factor"}
                fmt_cols["Discount Factor"] = "{:.4f}"
                try:
                    st.dataframe(result.line_items.style.format(fmt_cols), width='stretch')
                except Exception:
                    st.dataframe(result.line_items, width='stretch')

                v1, v2, v3 = st.columns(3)
                v1.metric("Sum PV(FCF)", f"${result.sum_pv_fcf:,.0f}M")
                v2.metric("PV(Terminal Value)", f"${result.pv_terminal_value:,.0f}M")
                v3.metric("TV as % of EV", f"{result.pv_terminal_value / result.enterprise_value * 100:.0f}%")

                wf_labels = ["PV FCFs", "PV Terminal Value", "Enterprise Value", "- Net Debt", "Equity Value"]
                wf_values = [result.sum_pv_fcf / 1000, result.pv_terminal_value / 1000, 0,
                             -result.net_debt / 1000, 0]
                wf_measure = ["relative", "relative", "total", "relative", "total"]
                fig_wf = go.Figure(go.Waterfall(
                    name="Valuation Bridge", orientation="v", measure=wf_measure,
                    x=wf_labels, y=wf_values,
                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                    decreasing={"marker": {"color": COLORS.get("danger", "#ff4444")}},
                    increasing={"marker": {"color": COLORS.get("gold", "#e0c97f")}},
                    totals={"marker": {"color": COLORS.get("lavender", "#b39ddb")}},
                ))
                fig_wf.update_layout(title="Valuation Bridge ($B)", template="plotly_dark",
                                     height=350, yaxis_title="Value ($B)")
                st.plotly_chart(fig_wf, width='stretch')

            with dcf_tabs[1]:
                st.subheader("Historical Financial Performance (10 Years)")
                st.caption("Anchoring forecast assumptions to historical company performance")
                hist_data = result.historical_data
                hist_stats = result.historical_stats
                if not hist_data.empty:
                    st.dataframe(hist_data, use_container_width=True, hide_index=True)
                    st.markdown("### 10-Year Averages")
                    stat_col1, stat_col2, stat_col3 = st.columns(3)
                    stat_col1.metric("Avg Revenue Growth", f"{hist_stats['avg_revenue_growth']:.1f}%")
                    stat_col2.metric("Avg EBIT Margin", f"{hist_stats['avg_ebit_margin']:.1f}%")
                    stat_col3.metric("Avg FCF Margin", f"{hist_stats['avg_fcf_margin']:.1f}%")
                    chart_col1, chart_col2 = st.columns(2)
                    with chart_col1:
                        if 'Revenue Growth %' in hist_data.columns:
                            fig_growth = go.Figure()
                            fig_growth.add_trace(go.Scatter(
                                x=hist_data['Year'], y=hist_data['Revenue Growth %'],
                                mode='lines+markers', name='Revenue Growth',
                                line=dict(color='#e0c97f', width=2), marker=dict(size=6)))
                            fig_growth.add_hline(y=hist_stats['avg_revenue_growth'], line_dash="dash",
                                                 line_color="#00ff88",
                                                 annotation_text=f"Avg: {hist_stats['avg_revenue_growth']:.1f}%")
                            fig_growth.update_layout(title="Revenue Growth Trend", template="plotly_dark",
                                                     height=300, yaxis_title="Growth %")
                            st.plotly_chart(fig_growth, use_container_width=True)
                    with chart_col2:
                        fig_margin = go.Figure()
                        fig_margin.add_trace(go.Scatter(
                            x=hist_data['Year'], y=hist_data['EBIT Margin %'],
                            mode='lines+markers', name='EBIT Margin',
                            line=dict(color='#b39ddb', width=2), marker=dict(size=6)))
                        fig_margin.add_hline(y=hist_stats['avg_ebit_margin'], line_dash="dash",
                                             line_color="#00ff88",
                                             annotation_text=f"Avg: {hist_stats['avg_ebit_margin']:.1f}%")
                        fig_margin.update_layout(title="EBIT Margin Trend", template="plotly_dark",
                                                 height=300, yaxis_title="Margin %")
                        st.plotly_chart(fig_margin, use_container_width=True)
                    st.markdown("### Forecast vs Historical Comparison")
                    comparison = result.forecast_vs_historical
                    comp_rows = []
                    for metric_name, metric_data in comparison.items():
                        deviation_pct = metric_data['deviation_pct']
                        highlight = abs(deviation_pct) > 5
                        comp_rows.append({
                            'Metric': metric_name.replace('_', '').title(),
                            'Forecast': f"{metric_data['forecast']:.1f}%",
                            'Historical Avg': f"{metric_data['historical']:.1f}%",
                            'Deviation': f"{metric_data['deviation']:+.1f}%",
                            'Deviation %': f"{deviation_pct:+.1f}%",
                            'Alert': 'Large Deviation' if highlight else 'Within Range'})
                    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)
                else:
                    st.warning("Historical financial data not available for this ticker.")

            with dcf_tabs[2]:
                st.subheader("Market-Implied Expectations (Reverse DCF)")
                st.caption("What assumptions does the current market price imply?")
                exp_color = result.market_expectation_color
                st.markdown(
                    f'<div style="background:#161b22;border:2px solid {exp_color};border-radius:8px;'
                    f'padding:14px;text-align:center;margin:10px 0;">'
                    f'<div style="color:{exp_color};font-size:1.2rem;font-weight:700;">{result.market_expectation_label}</div></div>',
                    unsafe_allow_html=True)
                rev_col1, rev_col2, rev_col3 = st.columns(3)
                with rev_col1:
                    st.markdown(
                        f'<div style="background:#1a1f2e;border:1px solid {exp_color};border-radius:6px;padding:12px;">'
                        f'<div style="color:#aaa;font-size:0.75rem;margin-bottom:6px;">MARKET IMPLIED REVENUE GROWTH</div>'
                        f'<div style="color:white;font-size:1.6rem;font-weight:800;">{result.market_implied_growth:.1%}</div></div>',
                        unsafe_allow_html=True)
                with rev_col2:
                    st.markdown(
                        f'<div style="background:#1a1f2e;border:1px solid {exp_color};border-radius:6px;padding:12px;">'
                        f'<div style="color:#aaa;font-size:0.75rem;margin-bottom:6px;">MARKET IMPLIED EBIT MARGIN</div>'
                        f'<div style="color:white;font-size:1.6rem;font-weight:800;">{result.market_implied_ebit_margin:.1%}</div></div>',
                        unsafe_allow_html=True)
                with rev_col3:
                    st.markdown(
                        f'<div style="background:#1a1f2e;border:1px solid {exp_color};border-radius:6px;padding:12px;">'
                        f'<div style="color:#aaa;font-size:0.75rem;margin-bottom:6px;">MARKET IMPLIED FCF GROWTH</div>'
                        f'<div style="color:white;font-size:1.6rem;font-weight:800;">{result.market_implied_fcf_growth:.1%}</div></div>',
                        unsafe_allow_html=True)
                st.markdown("---")
                st.markdown("### Interpretation")
                st.markdown(
                    f"The current market price of **${current_price:.2f}** implies:\n\n"
                    f"- Revenue growth of **{result.market_implied_growth:.1%}** annually\n"
                    f"- EBIT margin of **{result.market_implied_ebit_margin:.1%}**\n"
                    f"- FCF growth of **{result.market_implied_fcf_growth:.1%}**\n\n"
                    f"Compare these to your forecast assumptions and historical performance to assess if the market is"
                    f"pricing in overly optimistic or pessimistic scenarios.")

            with dcf_tabs[3]:
                st.subheader("Scenario-Weighted Valuation")
                scen_cols = st.columns(3)
                colors_scen = {"Bear": "#ff4444", "Base": "#e0c97f", "Bull": "#00ff88"}
                for i, s in enumerate(result.scenarios):
                    c = colors_scen.get(s.label, "#ffffff")
                    with scen_cols[i % 3]:
                        st.markdown(
                            f"<div style='background:#161b22;border:1px solid {c};"
                            f"border-radius:6px;padding:10px;text-align:center;'>"
                            f"<div style='color:{c};font-size:0.9rem;font-weight:700;'>{s.label}</div>"
                            f"<div style='color:#aaa;font-size:0.72rem;'>P = {s.probability:.0%}</div>"
                            f"<div style='color:white;font-size:1.3rem;font-weight:800;'>${s.fair_value:.2f}</div>"
                            f"<div style='color:{c};font-size:0.82rem;'>{s.upside:+.1%} upside</div>"
                            f"<hr style='border-color:#333;margin:4px 0;'/>"
                            f"<div style='color:#aaa;font-size:0.7rem;'>Growth: {s.revenue_growth_avg:.1%} | EBIT: {s.ebit_margin:.1%}</div>"
                            f"<div style='color:#aaa;font-size:0.7rem;'>WACC: {s.wacc:.2%}</div></div>",
                            unsafe_allow_html=True)
                st.markdown(
                    f"<div style='background:#1a1f2e;border:1px solid #e0c97f;"
                    f"border-radius:6px;padding:10px;text-align:center;margin-top:10px;'>"
                    f"<div style='color:#aaa;font-size:0.8rem;'>Probability-Weighted Fair Value</div>"
                    f"<div style='color:#e0c97f;font-size:1.6rem;font-weight:800;'>${result.scenario_weighted_value:.2f}</div>"
                    f"<div style='color:#aaa;font-size:0.72rem;'>{getattr(result, 'dynamic_scenario_notes', '') or ''}</div></div>",
                    unsafe_allow_html=True)

            with dcf_tabs[4]:
                st.subheader("Monte Carlo Valuation (10,000 Simulations)")
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Median Fair Value", f"${result.mc_median:.2f}")
                mc2.metric("Mean Fair Value", f"${result.mc_mean:.2f}")
                mc3.metric("Upside Probability", f"{result.mc_upside_prob:.1%}")
                mc4.metric("Downside Probability", f"{result.mc_downside_prob:.1%}")
                p = result.mc_percentiles
                pc1, pc2, pc3, pc4, pc5 = st.columns(5)
                pc1.metric("P5", f"${p['p5']:.2f}")
                pc2.metric("P25", f"${p['p25']:.2f}")
                pc3.metric("P50", f"${p['p50']:.2f}")
                pc4.metric("P75", f"${p['p75']:.2f}")
                pc5.metric("P95", f"${p['p95']:.2f}")
                if result.mc_distribution:
                    fig_mc = px.histogram(
                        x=result.mc_distribution, nbins=80,
                        title="Distribution of Fair Values (Monte Carlo)",
                        labels={"x": "Fair Value Per Share ($)"},
                        color_discrete_sequence=["#b39ddb"])
                    fig_mc.add_vline(x=current_price, line_dash="dash", line_color="#ff4444",
                                     annotation_text=f"Market ${current_price:.2f}")
                    fig_mc.add_vline(x=result.mc_median, line_dash="dash", line_color="#00ff88",
                                     annotation_text=f"Median ${result.mc_median:.2f}")
                    fig_mc.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig_mc, width='stretch')

            with dcf_tabs[5]:
                st.subheader("Peer Comparison — Relative Valuation")
                rv = result.relative_valuation
                rv_methods = ["P/E Implied", "EV/EBITDA Implied", "EV/FCF Implied",
                              "PEG Implied", "DCF (This Model)"]
                rv_values = [rv["pe_implied_fv"], rv["ev_ebitda_implied_fv"],
                             rv["ev_fcf_implied_fv"], rv["peg_implied_fv"],
                             result.fair_value_per_share]
                rv_multiples = [f"{rv['peer_pe']:.1f}x", f"{rv['peer_ev_ebitda']:.1f}x",
                                f"{rv['peer_ev_fcf']:.1f}x", f"{rv['peg_ratio']:.2f}", "—"]
                rv_df = pd.DataFrame({
                    "Valuation Method": rv_methods,
                    "Fair Value ($)": [f"${v:.2f}" for v in rv_values],
                    "Peer Multiple": rv_multiples,
                    "vs Market": [f"{(v - current_price) / current_price:+.1%}"
                                  if current_price > 0 else "N/A" for v in rv_values]})
                st.dataframe(rv_df, width='stretch', hide_index=True)
                fig_rv = go.Figure()
                bar_colors = ["#00ff88" if v > current_price else "#ff4444" for v in rv_values]
                fig_rv.add_trace(go.Bar(x=rv_methods, y=rv_values, marker_color=bar_colors,
                                        text=[f"${v:.2f}" for v in rv_values], textposition="outside"))
                fig_rv.add_hline(y=current_price, line_dash="dash", line_color="#e0c97f",
                                 annotation_text=f"Market ${current_price:.2f}")
                fig_rv.update_layout(title="Implied Fair Values by Method", template="plotly_dark",
                                     height=400, yaxis_title="Fair Value ($)")
                st.plotly_chart(fig_rv, width='stretch')
                r1, r2, r3 = st.columns(3)
                r1.metric("EBITDA ($M)", f"${rv['ebitda_M']:,.0f}M")
                r2.metric("FCF ($M)", f"${rv['last_fcf_M']:,.0f}M")
                r3.metric("EPS Proxy", f"${rv['eps_proxy']:.2f}")

            with dcf_tabs[6]:
                st.subheader("Sensitivity Analysis — Fair Value Per Share")
                st.caption("Rows: Terminal Growth Rate | Columns: WACC")
                try:
                    st.dataframe(result.sensitivity.style.format("${:.2f}"), width='stretch')
                except Exception:
                    st.dataframe(result.sensitivity, width='stretch')

            with dcf_tabs[7]:
                st.subheader("Catalyst Tracking Dashboard")
                impact_colors = {"High": "#ff4444", "Medium": "#e0c97f", "Low": "#aaaaaa"}
                dir_colors = {"Positive": "#00ff88", "Negative": "#ff4444", "Neutral": "#aaaaaa"}
                if hasattr(result, 'catalysts') and result.catalysts:
                    for cat in result.catalysts:
                        ic = impact_colors.get(cat.impact, "#aaaaaa")
                        dc = dir_colors.get(cat.direction, "#aaaaaa")
                        st.markdown(
                            f"<div style='background:#161b22;border-left:4px solid {ic};"
                            f"border-radius:6px;padding:14px;margin:8px 0;'>"
                            f"<div style='display:flex;justify-content:space-between;'>"
                            f"<span style='font-weight:700;color:white;font-size:1.05rem;'>{cat.name}</span>"
                            f"<span style='color:{ic};font-weight:600;'>{cat.impact} Impact</span></div>"
                            f"<div style='color:#888;font-size:0.8rem;margin:4px 0;'>Date: {cat.date} |"
                            f"<span style='color:{dc};'>{cat.direction}</span></div>"
                            f"<div style='color:#ccc;font-size:0.9rem;'>{cat.description}</div></div>",
                            unsafe_allow_html=True)
                else:
                    st.info("No catalysts identified. Run the model with a real ticker for catalyst tracking.")

            with dcf_tabs[8]:
                st.subheader("WACC Decomposition")
                wb = result.wacc_breakdown
                wc1, wc2, wc3 = st.columns(3)
                wc1.metric("WACC", f"{wb['wacc']:.2%}")
                wc2.metric("Cost of Equity (CAPM)", f"{wb['cost_of_equity']:.2%}")
                wc3.metric("After-Tax Cost of Debt", f"{wb['cost_of_debt_aftertax']:.2%}")
                wc4, wc5, wc6 = st.columns(3)
                wc4.metric("Weight Equity", f"{wb['weight_equity']:.1%}")
                wc5.metric("Weight Debt", f"{wb['weight_debt']:.1%}")
                wc6.metric("Beta", f"{wb['beta']:.2f}")
                st.markdown("""
                **WACC Formula:**
                `WACC = (E/(D+E)) × Re + (D/(D+E)) × Rd × (1 − T)`
                **Cost of Equity (CAPM):**
                `Re = Rf + β × (Rm − Rf)`""")
                wacc_data = {"Component": ["Risk-Free Rate (Rf)", "Equity Risk Premium (ERP)",
                                           "Beta (β)", "Cost of Equity (Re)", "Pre-Tax Cost of Debt (Rd)",
                                           "After-Tax Cost of Debt", "Weight Equity", "Weight Debt", "WACC"],
                             "Value": [f"{wb['risk_free_rate']:.2%}", f"{wb['equity_risk_premium']:.2%}",
                                       f"{wb['beta']:.2f}", f"{wb['cost_of_equity']:.2%}",
                                       f"{wb['cost_of_debt_pretax']:.2%}", f"{wb['cost_of_debt_aftertax']:.2%}",
                                       f"{wb['weight_equity']:.1%}", f"{wb['weight_debt']:.1%}",
                                       f"{wb['wacc']:.2%}"]}
                st.dataframe(pd.DataFrame(wacc_data), width='stretch', hide_index=True)

            with dcf_tabs[9]:
                st.subheader("Key Assumption Attribution")
                st.caption("Which assumptions drive the majority of valuation — rank by absolute impact.")
                attr = getattr(result, "key_assumption_attribution", {}) or {}
                if attr.get("drivers"):
                    rows = []
                    for d in attr["drivers"]:
                        rows.append({"Assumption": d["assumption"], "Value": d["value"],
                                     "+ Shock Impact": f"{d['+impact']:+.1f}%",
                                     "− Shock Impact": f"{d['-impact']:+.1f}%",
                                     "Share of Impact": f"{d.get('share_of_total_impact', 0):.0%}"})
                    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
                    st.caption(attr.get("note", ""))
                else:
                    st.info("Attribution requires a completed DCF run.")

            with dcf_tabs[10]:
                _render_consensus_dislocation(getattr(result, "consensus_layer", {}) or {})
                td = getattr(result, "terminal_diagnostics", {}) or {}
                if td:
                    st.markdown("### Terminal Value Diagnostics")
                    t1, t2, t3 = st.columns(3)
                    t1.metric("PV(TV) as % of EV", f"{td.get('tv_as_pct_of_ev', 0):.0%}")
                    t2.metric("Implied Terminal FCF Multiple",
                              f"{td.get('implied_terminal_fcf_multiple', 0):.1f}x" if td.get('implied_terminal_fcf_multiple') else "n/a")
                    t3.metric("WACC − g", f"{td.get('wacc_minus_g', 0):.2%}")
                    for flag in td.get("flags", []):
                        st.warning(flag)

            st.markdown("---")
            gen = get_financial_generator()
            try:
                from ib_excel_engine import build_dcf_workbook
                xls_bytes = build_dcf_workbook(result)
                _dcf_label = "Download Editable DCF Model (Excel, live formulas)"
            except Exception:
                model_data_for_export = {"full_result": result}
                xls_bytes = gen.generate_excel(model_data_for_export)
                _dcf_label = "Download Full Institutional Model (Excel)"
            st.download_button(
                _dcf_label,
                data=xls_bytes,
                file_name=f"{result.ticker}_Institutional_DCF.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dcf_xlsx_btn")
            st.caption("Yellow cells = analyst inputs; black cells = live formulas. Change WACC, growth, or margins and the valuation bridge (EV → net debt → equity → fair value/share) recalculates.")
            try:
                from presentation_generator import get_presentation_generator
                dcf_pptx = get_presentation_generator().generate_dcf_pitchbook(
                    result.ticker or ticker or "TICKER", result)
                _render_pptx_button("Download DCF Pitchbook (.pptx)", dcf_pptx,
                                    f"{result.ticker}_DCF_Pitchbook.pptx", "dcf_pptx_btn")
            except Exception:
                pass
            try:
                from investment_memo import MemoContext, get_memo_generator
                cl = getattr(result, "consensus_layer", {}) or {}
                dis = cl.get("dislocation", {}) if isinstance(cl, dict) else {}
                ctx = MemoContext(
                    title=f"{result.ticker} — Equity Valuation",
                    ticker=result.ticker, subject_type="Equity",
                    fair_value=result.fair_value_per_share,
                    current_price=current_price,
                    valuation_narrative=getattr(result, "dynamic_scenario_notes", ""),
                    dislocation_summary=dis.get("summary", ""),
                    risks=[c.description for c in (getattr(result, "catalysts", []) or []) if c.direction == "Negative"],
                    catalysts=[f"{c.name} ({c.date})" for c in (getattr(result, "catalysts", []) or [])[:6]],
                )
                _render_memo_button(ctx, "dcf_memo_btn")
            except Exception:
                pass


# ═════════════════════════════════════════════════════════════════════════════
# LBO TAB
# ═════════════════════════════════════════════════════════════════════════════

def _render_lbo_tab(fetch_ticker_fundamentals):
    st.subheader("Leveraged Buyout (LBO) Model")
    try:
        from lbo_model_engine import LBOAssumptions, get_lbo_engine

        lbo_fill_msg = _apply_pending_widget_autofill("_pending_lbo_fill")
        if lbo_fill_msg:
            st.success(lbo_fill_msg)

        _init_widget_defaults({
            "lbo_ticker_input": "", "lbo_yr_in": 2026, "lbo_yr_out": 2031,
            "lbo_ebitda_in": 500.0, "lbo_entry_mult_in": 10.0, "lbo_exit_mult_in": 10.0,
            "lbo_lev_in": 5.0, "lbo_int_in": 8.0, "lbo_tax_in": 25.0,
            "lbo_rev_growth_in": 5.0, "lbo_ebitda_margin_in": 20.0,
        })

        lbo_col_ticker, lbo_col_fetch = st.columns([2, 1])
        with lbo_col_ticker:
            l_ticker = st.text_input(
                "Target Ticker",
                placeholder="e.g. DELL, HCA, BX...",
                help="Enter any ticker. Use 'Auto-Fill from Ticker' to populate LBO inputs from live data.",
                key="lbo_ticker_input")
        with lbo_col_fetch:
            st.markdown("<br>", unsafe_allow_html=True)
            do_lbo_autofill = st.button(
                "Auto-Fill from Ticker", key="lbo_autofill_btn",
                help="Fetches live financials from yfinance and populates LBO inputs.")

        if do_lbo_autofill and l_ticker:
            with st.spinner(f"Fetching financials for {l_ticker.upper()}..."):
                try:
                    data = fetch_ticker_fundamentals(l_ticker)
                    _queue_widget_autofill(
                        "_pending_lbo_fill",
                        {"lbo_ebitda_in": float(data["ebitda_m"]),
                         "lbo_entry_mult_in": float(data["entry_multiple"]),
                         "lbo_exit_mult_in": float(data["exit_multiple"]),
                         "lbo_lev_in": float(data["leverage_multiple"]),
                         "lbo_int_in": float(data["interest_rate_pct"]),
                         "lbo_tax_in": float(data["tax_rate_pct"]),
                         "lbo_rev_growth_in": float(data["revenue_growth"]),
                         "lbo_ebitda_margin_in": round(float(data["ebitda_margin"]) * 100, 1)},
                        f"Auto-filled LBO inputs for {data['ticker']} "
                        f"(EBITDA ${data['ebitda_m']:,.0f}M @ {data['entry_multiple']:.1f}x) — review and click Run.")
                    st.rerun()
                except Exception as _lbo_err:
                    st.warning(f"Auto-fill partial ({_lbo_err}). Enter assumptions manually.")

        l_col1, l_col2 = st.columns([1, 2])
        with l_col1:
            st.markdown("#### Deal Parameters")
            l_entry_year = st.number_input("Entry Year", step=1, key="lbo_yr_in")
            l_exit_year = st.number_input("Exit Year", step=1, key="lbo_yr_out")

            st.markdown("#### Valuation & Leverage")
            l_ltm_ebitda = st.number_input("LTM EBITDA ($M)", step=25.0, key="lbo_ebitda_in")
            l_entry_mult = st.slider("Entry Multiple (x)", 5.0, 25.0, step=0.5, key="lbo_entry_mult_in")
            l_exit_mult = st.slider("Exit Multiple (x)", 5.0, 25.0, step=0.5, key="lbo_exit_mult_in")
            l_leverage = st.slider("Leverage Multiple (Debt/EBITDA)", 1.0, 8.0, step=0.5, key="lbo_lev_in")
            l_interest = st.slider("Cost of Debt (%)", 4.0, 15.0, step=0.25, key="lbo_int_in") / 100.0
            l_tax = st.slider("Tax Rate (%)", 10.0, 40.0, step=1.0, key="lbo_tax_in") / 100.0
            l_rev_growth = st.slider("Revenue Growth (%)", -10.0, 60.0, step=0.5, key="lbo_rev_growth_in") / 100.0
            l_ebitda_margin = st.slider("EBITDA Margin (%)", 5.0, 80.0, step=0.5, key="lbo_ebitda_margin_in") / 100.0

            st.markdown("#### Advanced Structure")
            l_pik = st.slider("PIK Interest (% of TLB interest)", 0.0, 100.0, 0.0, step=5.0) / 100.0
            l_2l = st.slider("Second Lien (x EBITDA)", 0.0, 3.0, 0.0, step=0.5)
            l_2l_rate = st.slider("Second Lien Rate (%)", 8.0, 18.0, 12.0, step=0.5) / 100.0

            l_run = st.button("Run LBO Analysis", type="primary", key="lbo_run_btn")

        with l_col2:
            if l_run:
                _lbo_ticker_val = st.session_state.get("lbo_ticker_input", l_ticker or "TARGET")
                _hold_years = max(l_exit_year - l_entry_year, 1)
                assumptions = LBOAssumptions(
                    ticker=_lbo_ticker_val,
                    target_name=_lbo_ticker_val,
                    entry_year=l_entry_year,
                    exit_year=l_exit_year,
                    ltm_ebitda=l_ltm_ebitda,
                    entry_multiple=l_entry_mult,
                    exit_multiple=l_exit_mult,
                    leverage_multiple=l_leverage,
                    interest_rate=l_interest,
                    tax_rate=l_tax,
                    revenue_growth_rates=[l_rev_growth] * _hold_years,
                    ebitda_margins=[l_ebitda_margin] * _hold_years,
                    pik_interest_pct=l_pik,
                    second_lien_multiple=l_2l,
                    second_lien_rate=l_2l_rate,
                )
                lbo_engine = get_lbo_engine()
                res = lbo_engine.run_lbo(assumptions)
                st.session_state["lbo_result"] = res
                st.session_state["bridge_context"] = dict(
                    st.session_state.get("bridge_context", {}), lbo=res, ticker=_lbo_ticker_val)

                try:
                    from model_audit import run_model_qa
                    _render_qa(run_model_qa(res, "lbo"), "LBO")
                except Exception:
                    pass

                st.markdown("### LBO Returns Summary")
                rcol1, rcol2, rcol3 = st.columns(3)
                rcol1.metric("Purchase Price", f"${res.purchase_price:,.1f}M")
                rcol2.metric("Equity Check", f"${res.equity_amount:,.1f}M")
                rcol3.metric("Initial Debt", f"${res.debt_amount:,.1f}M")

                rcol4, rcol5, rcol6 = st.columns(3)
                rcol4.metric("Exit Enterprise Value", f"${res.exit_enterprise_value:,.1f}M")
                moic_color = "#00cc66" if res.moic >= 2.0 else "#ffaa00" if res.moic >= 1.5 else "#cc3333"
                rcol5.markdown(
                    f"**MOIC:** <span style='color:{moic_color};font-size:1.5em;'>{res.moic:.2f}x</span>",
                    unsafe_allow_html=True)
                irr_color = "#00cc66" if res.irr >= 0.20 else "#ffaa00" if res.irr >= 0.15 else "#cc3333"
                rcol6.markdown(
                    f"**IRR:** <span style='color:{irr_color};font-size:1.5em;'>{res.irr:.1%}</span>",
                    unsafe_allow_html=True)

                st.markdown("### Pro Forma Cash Flows")
                st.dataframe(
                    res.cash_flows.style.format({
                        "Revenue": "${:,.1f}", "EBITDA": "${:,.1f}",
                        "EBIT": "${:,.1f}", "Net Income": "${:,.1f}",
                        "Cash Flow Available for Debt Service (CFADS)": "${:,.1f}",
                    }),
                    use_container_width=True)
                st.markdown("### Debt Paydown Schedule")
                st.dataframe(
                    res.debt_schedule.style.format({
                        "Ending TLB Balance": "${:,.1f}",
                        "Ending Second Lien Balance": "${:,.1f}",
                        "Ending Revolver Balance": "${:,.1f}",
                        "Total Debt": "${:,.1f}",
                    }),
                    use_container_width=True)

                # Returns attribution (institutional upgrade)
                ra = res.returns_attribution or {}
                if ra.get("components"):
                    st.markdown("### Returns Attribution Bridge")
                    st.caption("Where sponsor returns originate — operating, deleveraging, multiple, cash.")
                    att_rows = []
                    for k, v in ra.get("components", {}).items():
                        att_rows.append({"Driver": k, "$M": f"${v:,.1f}",
                                         "% of Uplift": f"{ra['pct_of_uplift'].get(k, 0):+.1f}%"})
                    st.dataframe(pd.DataFrame(att_rows), use_container_width=True, hide_index=True)

                # Scenario cases (institutional upgrade)
                if res.scenario_cases is not None and not res.scenario_cases.empty:
                    st.markdown("### Scenario Cases")
                    sc_cols = st.columns(3)
                    for i, (_, row) in enumerate(res.scenario_cases.iterrows()):
                        c = ["#ff7043", "#e0c97f", "#00c853"][i % 3]
                        with sc_cols[i]:
                            st.markdown(
                                f"<div style='background:#161b22;border:1px solid {c};border-radius:6px;"
                                f"padding:10px;text-align:center;'>"
                                f"<div style='color:{c};font-weight:700;'>{row['Scenario']}</div>"
                                f"<div style='color:#aaa;font-size:0.75rem;'>P = {row['Probability']:.0%}</div>"
                                f"<div style='color:white;font-size:1.3rem;font-weight:800;'>{row['IRR']:.1%}</div>"
                                f"<div style='color:{c};font-size:0.85rem;'>{row['MOIC']:.2f}x MOIC</div>"
                                f"<div style='color:#aaa;font-size:0.7rem;'>Exit {row['Exit Multiple (x)']:.1f}x | "
                                f"Equity ${row['Exit Equity ($M)']:,.0f}M</div></div>",
                                unsafe_allow_html=True)

                st.markdown("### Strategic Conclusion & Implications")
                conclusion_text = res.conclusion if getattr(res, "conclusion", None) else (
                    "A full strategic verdict was not generated. Review the headline metrics above "
                    "(IRR/MOIC) against your hurdle rate before deciding to proceed.")
                with st.expander("Strategic Conclusion — click to expand", expanded=True):
                    st.markdown(
                        f"<div style='background:#0a1628; border-left: 3px solid #c9a84c;"
                        f"border-radius: 0 8px 8px 0; padding: 16px 20px; margin: 8px 0;'>"
                        f"{conclusion_text.replace(chr(10), '<br>')}</div>",
                        unsafe_allow_html=True)

                try:
                    from ib_excel_engine import build_lbo_workbook
                    wb_bytes = build_lbo_workbook(res)
                    st.download_button(
                        "Download Editable LBO Model (Excel, live formulas)",
                        data=wb_bytes,
                        file_name=f"LBO_{_lbo_ticker_val}_model.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="lbo_xlsx_btn")
                    st.caption("Yellow cells = analyst inputs; black cells = live formulas. Includes full debt schedule (TLB + revolver + min cash), IRR/MOIC on the Returns tab, and entry×exit multiple sensitivity.")
                except Exception as _lbo_xlsx_err:
                    st.warning(f"Excel export unavailable ({_lbo_xlsx_err})")

                try:
                    from presentation_generator import get_presentation_generator
                    lbo_pptx = get_presentation_generator().generate_lbo_pitchbook(_lbo_ticker_val, res)
                    _render_pptx_button("Download LBO Pitchbook (.pptx)", lbo_pptx,
                                        f"LBO_{_lbo_ticker_val}_Pitchbook.pptx", "lbo_pptx_btn")
                except Exception:
                    pass

                try:
                    from investment_memo import MemoContext
                    ctx = MemoContext(
                        title=f"LBO — {_lbo_ticker_val}",
                        ticker=_lbo_ticker_val, subject_type="LBO",
                        transaction_summary=(
                            f"LBO of {_lbo_ticker_val} at {l_entry_mult:.1f}x LTM EBITDA "
                            f"({l_leverage:.1f}x leverage), {_hold_years}-year hold."),
                        transaction_outcome=f"IRR {res.irr:.1%}, MOIC {res.moic:.2f}x.",
                        fair_value=res.equity_amount, current_price=None,
                        risks=[l for l in conclusion_text.split("\n") if "Risk" in l or "risk" in l][:5],
                    )
                    _render_memo_button(ctx, "lbo_memo_btn")
                except Exception:
                    pass

    except Exception as e:
        st.error(f"LBO Engine Error: {e}")
        import traceback
        st.code(traceback.format_exc())


# ═════════════════════════════════════════════════════════════════════════════
# M&A TAB
# ═════════════════════════════════════════════════════════════════════════════

def _render_mna_tab(fetch_ticker_fundamentals):
    st.subheader("M&A Accretion / Dilution Model")
    try:
        from mna_model_engine import MnAAssumptions, get_mna_engine

        def _mna_widget_values(data: dict) -> dict:
            return {"price_in": float(data["price"]), "eps_in": float(data["eps"]),
                    "shrs_in": float(data["shares_m"])}

        def _mna_autofill_message(data: dict, label: str) -> str:
            return (f"{label} auto-filled: {data['ticker']} @ ${data['price']:.2f},"
                    f"EPS ${data['eps']:.2f}, {data['shares_m']:.0f}M shares")

        acq_fill_msg = _apply_pending_widget_autofill("_pending_mna_acq_fill")
        if acq_fill_msg:
            st.success(acq_fill_msg)
        tgt_fill_msg = _apply_pending_widget_autofill("_pending_mna_tgt_fill")
        if tgt_fill_msg:
            st.success(tgt_fill_msg)

        _init_widget_defaults({
            "mna_acq_ticker_in": "", "mna_tgt_ticker_in": "",
            "mna_acq_price_in": 100.0, "mna_acq_eps_in": 5.0, "mna_acq_shrs_in": 1000.0,
            "mna_tgt_price_in": 50.0, "mna_tgt_eps_in": 2.0, "mna_tgt_shrs_in": 200.0,
            "mna_prem": 30.0, "mna_stock": 50.0, "mna_syn": 100.0,
            "mna_debt_cost": 6.0, "mna_tax": 21.0,
        })

        st.markdown("#### Acquirer")
        acq_col1, acq_col2 = st.columns([2, 1])
        with acq_col1:
            m_acq = st.text_input("Acquirer Ticker", placeholder="e.g. MSFT, JPM...",
                                  key="mna_acq_ticker_in")
        with acq_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            do_acq_fill = st.button("Auto-Fill from Ticker", key="mna_acq_fill_btn")

        if do_acq_fill and m_acq:
            with st.spinner(f"Fetching {m_acq.upper()} financials..."):
                try:
                    data = fetch_ticker_fundamentals(m_acq)
                    vals = _mna_widget_values(data)
                    _queue_widget_autofill("_pending_mna_acq_fill",
                                           {f"mna_acq_{k}": v for k, v in vals.items()},
                                           _mna_autofill_message(data, "Acquirer"))
                    st.rerun()
                except Exception as _e:
                    st.warning(f"Auto-fill failed ({_e}). Enter manually.")

        st.markdown("#### Target")
        tgt_col1, tgt_col2 = st.columns([2, 1])
        with tgt_col1:
            m_tgt = st.text_input("Target Ticker", placeholder="e.g. ATVI, TGT...",
                                  key="mna_tgt_ticker_in")
        with tgt_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            do_tgt_fill = st.button("Auto-Fill from Ticker", key="mna_tgt_fill_btn")

        if do_tgt_fill and m_tgt:
            with st.spinner(f"Fetching {m_tgt.upper()} financials..."):
                try:
                    data = fetch_ticker_fundamentals(m_tgt)
                    vals = _mna_widget_values(data)
                    _queue_widget_autofill("_pending_mna_tgt_fill",
                                           {f"mna_tgt_{k}": v for k, v in vals.items()},
                                           _mna_autofill_message(data, "Target"))
                    st.rerun()
                except Exception as _e2:
                    st.warning(f"Auto-fill failed ({_e2}). Enter manually.")

        m_col1, m_col2 = st.columns([1, 2])
        with m_col1:
            st.markdown("#### Acquirer Financials")
            m_acq_price = st.number_input("Acquirer Share Price ($)", step=1.0, key="mna_acq_price_in")
            m_acq_eps = st.number_input("Acquirer EPS ($)", step=0.1, key="mna_acq_eps_in")
            m_acq_shrs = st.number_input("Acquirer Shares Outstanding (M)", step=10.0, key="mna_acq_shrs_in")

            st.markdown("#### Target Financials")
            m_tgt_price = st.number_input("Target Share Price ($)", step=1.0, key="mna_tgt_price_in")
            m_tgt_eps = st.number_input("Target EPS ($)", step=0.1, key="mna_tgt_eps_in")
            m_tgt_shrs = st.number_input("Target Shares Outstanding (M)", step=5.0, key="mna_tgt_shrs_in")

            st.markdown("#### Deal Structure")
            m_prem = st.slider("Offer Premium (%)", 0.0, 100.0, step=1.0, key="mna_prem") / 100.0
            m_stock_pct = st.slider("Stock Consideration (%)", 0.0, 100.0, step=5.0, key="mna_stock") / 100.0
            m_cash_pct = 1.0 - m_stock_pct

            st.markdown("#### Synergies & Financing")
            m_syn = st.number_input("Pre-Tax Synergies ($M)", step=25.0, key="mna_syn")
            m_debt_cost = st.slider("Cost of New Debt (%)", 2.0, 15.0, step=0.25, key="mna_debt_cost") / 100.0
            m_tax = st.slider("Tax Rate (%)", 10.0, 40.0, step=1.0, key="mna_tax") / 100.0

            st.markdown("#### Institutional Extras")
            m_earnout = st.number_input("Earnout (max, $M)", value=0.0, step=25.0)
            m_earnout_prob = st.slider("Earnout Probability (%)", 0.0, 100.0, 100.0, step=5.0) / 100.0
            m_syn_prob = st.slider("Synergy Realization Probability (%)", 0.0, 100.0, 100.0, step=5.0) / 100.0
            m_hold = st.slider("Buyer Hold Period (yrs)", 1, 10, 3)

            m_run = st.button("Run M&A Analysis", type="primary", key="mna_run_btn")

        with m_col2:
            if m_run:
                _acq_t = st.session_state.get("mna_acq_ticker_in", m_acq or "ACQ")
                _tgt_t = st.session_state.get("mna_tgt_ticker_in", m_tgt or "TGT")
                assumptions = MnAAssumptions(
                    acquirer_ticker=_acq_t, target_ticker=_tgt_t,
                    acquirer_price=m_acq_price, acquirer_eps=m_acq_eps,
                    acquirer_shares=m_acq_shrs,
                    target_price=m_tgt_price, target_eps=m_tgt_eps,
                    target_shares=m_tgt_shrs,
                    offer_premium=m_prem, percent_stock=m_stock_pct,
                    percent_cash=m_cash_pct,
                    cost_of_debt=m_debt_cost, tax_rate=m_tax,
                    pre_tax_synergies=m_syn,
                    earnout_value=m_earnout, earnout_probability=m_earnout_prob,
                    synergy_probability=m_syn_prob, buyer_hold_years=m_hold,
                )
                mna_engine = get_mna_engine()
                res = mna_engine.run_mna(assumptions)
                st.session_state["mna_result"] = res
                st.session_state["bridge_context"] = dict(
                    st.session_state.get("bridge_context", {}), mna=res, ticker=_acq_t)

                try:
                    from model_audit import run_model_qa
                    _render_qa(run_model_qa(res, "mna"), "M&A")
                except Exception:
                    pass

                st.markdown("### Deal Output & Value Creation")
                dcol1, dcol2, dcol3 = st.columns(3)
                dcol1.metric("Offer Price / Share", f"${res.offer_price:,.2f}")
                dcol2.metric("Total Deal Value", f"${res.total_deal_value:,.1f}M")
                dcol3.metric("New Shares Issued", f"{res.new_shares_issued:,.1f}M")

                st.markdown("---")
                st.markdown("### Accretion / Dilution Analysis")
                acol1, acol2, acol3 = st.columns(3)
                acol1.metric("Standalone Acquirer EPS", f"${res.assumptions.acquirer_eps:,.2f}")
                acol2.metric("Pro Forma EPS", f"${res.pro_forma_eps:,.2f}")
                ad_color = "#00cc66" if res.is_accretive else "#cc3333"
                ad_text = "Accretive" if res.is_accretive else "Dilutive"
                acol3.markdown(
                    f"**Impact:** <span style='color:{ad_color};font-size:1.5em;'>"
                    f"{ad_text} ({res.accretion_dilution_pct:+.1%})</span>",
                    unsafe_allow_html=True)
                st.info(
                    f"The transaction is structurally **{ad_text.lower()}** by"
                    f"${abs(res.accretion_dilution_dollar):.2f} per share in Year 1,"
                    f"assuming ${m_syn:,.1f}M in pre-tax synergies.")

                # Contribution analysis
                st.markdown("### Contribution Analysis (EPS Bridge)")
                st.dataframe(res.contribution_analysis, use_container_width=True, hide_index=True)

                # Institutional extras
                inst_tabs = st.tabs(["Synergy Schedule", "Earnout", "Buyer / Seller",
                                     "Max Price & Value Bridge", "Transaction Scenarios"])
                with inst_tabs[0]:
                    if not res.synergy_schedule.empty:
                        st.dataframe(res.synergy_schedule.style.format({
                            "Realization %": "{:.0%}", "Pre-tax run-rate synergies ($M)": "${:,.1f}",
                            "Probability-adjusted ($M)": "${:,.1f}", "Implementation costs ($M)": "${:,.1f}"}),
                            use_container_width=True, hide_index=True)
                with inst_tabs[1]:
                    em = res.earnout_metrics
                    e1, e2, e3 = st.columns(3)
                    e1.metric("Max Earnout", f"${em.get('value', 0):,.0f}M")
                    e2.metric("Expected Value", f"${em.get('expected_value', 0):,.0f}M")
                    e3.metric("EPS Impact / Yr", f"${em.get('annual_after_tax_eps_impact', 0):.3f}")
                    st.caption(f"Earnout runs {em.get('years', 0)} year(s); probability {em.get('probability', 0):.0%}.")
                with inst_tabs[2]:
                    br = res.buyer_returns
                    b1, b2 = st.columns(2)
                    b1.metric("Acquirer-Sh. IRR (est.)", f"{br.get('irr_pct', 0):.1%}")
                    b2.metric("MOIC", f"{br.get('moic', 1):.2f}x")
                    st.caption(br.get("note", ""))
                    sp = res.seller_proceeds
                    s1, s2, s3 = st.columns(3)
                    s1.metric("Seller Cash", f"${sp.get('cash_proceeds', 0):,.0f}M")
                    s2.metric("Seller Stock", f"${sp.get('stock_proceeds', 0):,.0f}M")
                    s3.metric("Total Expected", f"${sp.get('total_expected_proceeds', 0):,.0f}M")
                with inst_tabs[3]:
                    p1, p2 = st.columns(2)
                    p1.metric("Max (EPS-neutral) Price / Share", f"${res.max_purchase_price:,.2f}")
                    vb = res.value_bridge
                    p2.metric("Net Value Creation", f"${vb.get('net_value_creation', 0):+,.0f}M")
                    vb_rows = [{"Driver": k.replace('_', ' ').title(), "$M": f"${v:,.1f}"}
                               for k, v in vb.items()]
                    st.dataframe(pd.DataFrame(vb_rows), use_container_width=True, hide_index=True)
                    fa = res.fcf_accretion
                    f1, f2 = st.columns(2)
                    f1.metric("Pro Forma FCF", f"${fa.get('pro_forma_fcf', 0):,.0f}M")
                    f2.metric("FCF Accretion", f"{fa.get('fcf_accretion_pct', 0):+.1%}")
                with inst_tabs[4]:
                    if not res.transaction_scenarios.empty:
                        st.dataframe(res.transaction_scenarios.style.format({
                            "Probability": "{:.0%}", "Accretion/(Dilution) %": "{:+.2%}",
                            "Cost of Debt": "{:.2%}"}),
                            use_container_width=True, hide_index=True)

                st.markdown("### Strategic Conclusion & Implications")
                conclusion_text = res.conclusion if getattr(res, "conclusion", None) else (
                    "A full strategic verdict was not generated. Review the accretion/dilution metrics "
                    "above against your required return before deciding to proceed.")
                with st.expander("Strategic Conclusion — click to expand", expanded=True):
                    st.markdown(
                        f"<div style='background:#0a1628; border-left: 3px solid #c9a84c;"
                        f"border-radius: 0 8px 8px 0; padding: 16px 20px; margin: 8px 0;'>"
                        f"{conclusion_text.replace(chr(10), '<br>')}</div>",
                        unsafe_allow_html=True)

                try:
                    from ib_excel_engine import build_mna_workbook
                    wb_bytes = build_mna_workbook(res)
                    st.download_button(
                        "Download Editable M&A Model (Excel, live formulas)",
                        data=wb_bytes,
                        file_name=f"M&A_{_acq_t}_{_tgt_t}_accretion_model.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="mna_xlsx_btn")
                    st.caption("Yellow cells = analyst inputs; black cells = live formulas. Includes sources=uses check, EPS bridge, and premium×stock / premium×synergy sensitivity tables.")
                except Exception as _mna_xlsx_err:
                    st.warning(f"Excel export unavailable ({_mna_xlsx_err})")

                try:
                    from presentation_generator import get_presentation_generator
                    mna_pptx = get_presentation_generator().generate_mna_pitchbook(_acq_t, _tgt_t, res)
                    _render_pptx_button("Download M&A Pitchbook (.pptx)", mna_pptx,
                                        f"{_acq_t}_{_tgt_t}_MnA_Pitchbook.pptx", "mna_pptx_btn")
                except Exception:
                    pass

                try:
                    from investment_memo import MemoContext
                    ctx = MemoContext(
                        title=f"M&A — {_acq_t} / {_tgt_t}", ticker=_tgt_t, subject_type="M&A",
                        transaction_summary=(
                            f"{_acq_t} proposes to acquire {_tgt_t} at {m_prem:.0%} premium "
                            f"({m_stock_pct:.0%} stock / {m_cash_pct:.0%} cash) with ${m_syn:,.0f}M synergies."),
                        transaction_outcome=(
                            f"Pro forma EPS ${res.pro_forma_eps:.2f} ({res.accretion_dilution_pct:+.1%} "
                            f"{'accretive' if res.is_accretive else 'dilutive'})."),
                        fair_value=res.offer_price, current_price=m_tgt_price,
                        risks=[l.split(" - ")[-1] for l in conclusion_text.split("\n")
                               if l.strip().startswith("-")][:6],
                    )
                    _render_memo_button(ctx, "mna_memo_btn")
                except Exception:
                    pass

    except Exception as e:
        st.error(f"M&A Engine Error: {e}")
        import traceback
        st.code(traceback.format_exc())


# ═════════════════════════════════════════════════════════════════════════════
# TRADING COMPS (CCA) TAB
# ═════════════════════════════════════════════════════════════════════════════

def _render_comps_tab():
    st.subheader("Comparable Company Analysis (Trading Comps)")
    st.caption("Multiples table, relevance scoring, distributional stats, implied valuation, "
               "growth/margin adjustments, regressions and outlier detection.")
    try:
        from comps_engine import CompsCompany, CompsInputs, get_comps_engine

        c1, c2, c3 = st.columns(3)
        subject_ticker = c1.text_input("Subject Ticker",
                                       value=st.session_state.get("comps_subject", "SUBJ"),
                                       key="comps_subj_ticker")
        subject_name = c2.text_input("Subject Company Name", value="Subject Company",
                                     key="comps_subj_name")
        subject_sector = c3.text_input("Subject Sector", value="Technology",
                                       key="comps_subj_sector")

        sc1, sc2, sc3, sc4, sc5 = st.columns(5)
        s_rev = sc1.number_input("Revenue ($M)", value=1000.0, step=50.0,
                                 key="comps_subj_rev")
        s_growth = sc2.number_input("Growth (%)", value=20.0, step=1.0,
                                    key="comps_subj_growth")
        s_margin = sc3.number_input("EBITDA Margin (%)", value=25.0, step=1.0,
                                    key="comps_subj_margin")
        s_nd = sc4.number_input("Net Debt ($M)", value=0.0, step=10.0,
                                key="comps_subj_nd")
        s_shares = sc5.number_input("Shares (M)", value=50.0, step=1.0,
                                    key="comps_subj_shares")
        sc6, sc7 = st.columns(2)
        s_price = sc6.number_input("Current Price ($)", value=40.0, step=1.0,
                                   key="comps_subj_price")
        s_ebitda = sc7.number_input("EBITDA ($M)", value=250.0, step=10.0,
                                    key="comps_subj_ebitda")

        n_peers = st.slider("Number of Peers", 2, 8, 5, key="comps_n_peers")
        peer_data = []
        for i in range(n_peers):
            with st.expander(f"Peer {i + 1}", expanded=(i == 0)):
                p1, p2, p3, p4 = st.columns(4)
                tkr = p1.text_input(f"Ticker {i + 1}", value=f"PEER{i + 1}", key=f"comps_t{i}_ticker")
                sector = p2.text_input(f"Sector {i + 1}", value="Technology", key=f"comps_t{i}_sector")
                model = p3.text_input(f"Model {i + 1}", value="SaaS", key=f"comps_t{i}_model")
                geo = p4.text_input(f"Geo {i + 1}", value="US", key=f"comps_t{i}_geo")
                q1, q2, q3, q4 = st.columns(4)
                rev = q1.number_input(f"Revenue {i + 1} ($M)", value=800.0 + i * 100, step=25.0,
                                      key=f"comps_t{i}_rev")
                growth = q2.number_input(f"Growth {i + 1} (%)", value=15.0 + i * 2, step=1.0,
                                         key=f"comps_t{i}_growth")
                margin = q3.number_input(f"EBITDA Margin {i + 1} (%)", value=22.0 + i, step=1.0,
                                         key=f"comps_t{i}_margin")
                mktcap = q4.number_input(f"Market Cap {i + 1} ($M)", value=2000.0 + i * 200, step=100.0,
                                         key=f"comps_t{i}_mktcap")
                r1, r2, r3, r4 = st.columns(4)
                ebitda = r1.number_input(f"EBITDA {i + 1} ($M)", value=180.0 + i * 15, step=5.0,
                                         key=f"comps_t{i}_ebitda")
                nd = r2.number_input(f"Net Debt {i + 1} ($M)", value=0.0, step=10.0, key=f"comps_t{i}_nd")
                shares = r3.number_input(f"Shares {i + 1} (M)", value=40.0, step=1.0,
                                         key=f"comps_t{i}_shares")
                price = r4.number_input(f"Price {i + 1} ($)", value=50.0, step=1.0,
                                        key=f"comps_t{i}_price")
                peer_data.append({
                    "ticker": tkr, "sector": sector, "business_model": model, "geography": geo,
                    "revenue": rev, "revenue_growth_pct": growth, "ebitda_margin_pct": margin,
                    "ebitda": ebitda, "net_debt": nd, "market_cap": mktcap,
                    "shares_outstanding": shares, "price": price,
                })

        run_comps = st.button("Run Trading Comps", type="primary")

        if run_comps:
            subject = CompsCompany(
                ticker=subject_ticker, name=subject_name, sector=subject_sector,
                revenue=s_rev, revenue_growth_pct=s_growth, ebitda_margin_pct=s_margin,
                ebitda=s_ebitda, net_debt=s_nd, shares_outstanding=s_shares, price=s_price,
                business_model="Subject")
            peers = [CompsCompany(**{k: v for k, v in p.items() if k != "geography"},
                                   geography=p.get("geography", "")) for p in peer_data]
            result = get_comps_engine().run_comps(CompsInputs(subject=subject, peers=peers))
            st.session_state["comps_result"] = result
            st.session_state["bridge_context"] = dict(
                st.session_state.get("bridge_context", {}), comps=result, ticker=subject_ticker)

            try:
                from model_audit import run_model_qa
                _render_qa(run_model_qa(result, "comps"), "Comps")
            except Exception:
                pass

            iv = result.implied_valuation
            blended = iv.get("Blended (weighted)", 0)
            lo, hi = result.implied_range
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Comp-Blended Value", f"${blended:.2f}")
            k2.metric("Comp Range", f"${lo:.2f} – ${hi:.2f}")
            k3.metric("Current Price", f"${s_price:.2f}")
            pd_ = result.premium_discount or {}
            prem = pd_.get("premium_to_blended", 0) or 0
            k4.metric("Premium / (Discount)", f"{prem:+.1%}")

            st.markdown("### Trading Multiples")
            st.dataframe(result.multiples_df, width='stretch', hide_index=True)

            st.markdown("### Multiple Statistics")
            stat_rows = []
            for mult, s in result.stats.items():
                if s.get("count", 0) == 0:
                    continue
                stat_rows.append({"Multiple": mult, "Mean": f"{s['mean']:.1f}x",
                                  "Median": f"{s['median']:.1f}x", "Q1": f"{s['q1']:.1f}x",
                                  "Q3": f"{s['q3']:.1f}x", "Low": f"{s['low']:.1f}x",
                                  "High": f"{s['high']:.1f}x"})
            st.dataframe(pd.DataFrame(stat_rows), width='stretch', hide_index=True)

            st.markdown("### Implied Valuation by Method")
            imp_rows = [{"Method": k, "Value ($)": f"${v:,.2f}"} for k, v in iv.items()]
            st.dataframe(pd.DataFrame(imp_rows), width='stretch', hide_index=True)
            fig_iv = go.Figure()
            labels = list(iv.keys())
            vals = [iv[k] for k in labels]
            fig_iv.add_trace(go.Bar(x=labels, y=vals,
                                    marker_color=["#00c853" if v >= s_price else "#ff7043" for v in vals],
                                    text=[f"${v:.2f}" for v in vals], textposition="outside"))
            fig_iv.add_hline(y=s_price, line_dash="dash", line_color="#e0c97f",
                             annotation_text=f"Market ${s_price:.2f}")
            fig_iv.update_layout(title="Implied Value by Method", template="plotly_dark",
                                 height=380, xaxis_tickangle=-25)
            st.plotly_chart(fig_iv, width='stretch')

            st.markdown("### Peer Relevance")
            rel_rows = [{"Ticker": t, "Score": f"{r['score']:.0%}",
                         "Why": "; ".join(r["reasons"][:3])}
                        for t, r in result.relevance.items()]
            st.dataframe(pd.DataFrame(rel_rows), width='stretch', hide_index=True)

            st.markdown("### Regressions & Outliers")
            reg = result.regressions or {}
            g = reg.get("ev_ebitda_vs_growth", {}) or {}
            m = reg.get("ev_ebitda_vs_margin", {}) or {}
            rc1, rc2 = st.columns(2)
            if g.get("n"):
                rc1.metric("EV/EBITDA vs Growth", f"slope {g['slope']:.2f} | R² {g['r2']:.2f}")
            if m.get("n"):
                rc2.metric("EV/EBITDA vs Margin", f"slope {m['slope']:.2f} | R² {m['r2']:.2f}")
            if result.outliers:
                for o in result.outliers:
                    st.warning(f"{o['ticker']} — {o['multiple']} at {o['value']}x (outside {o['band']})")
            else:
                st.success("No statistical outliers detected in the core peer set.")

            st.markdown("### Conclusion")
            st.info(result.conclusion)

            # Downloads
            try:
                from ib_excel_engine import build_comps_workbook
                comps_xlsx = build_comps_workbook(result.multiples_df,
                                                 title="Comparable Company Analysis")
                st.download_button("Download Comps Workbook (Excel)",
                                   data=comps_xlsx,
                                   file_name=f"{subject_ticker}_Trading_Comps.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   key="comps_xlsx_btn")
            except Exception:
                pass
            try:
                from presentation_generator import get_presentation_generator
                comps_pptx = get_presentation_generator().generate_comps_pitchbook(result)
                _render_pptx_button("Download Comps Pitchbook (.pptx)", comps_pptx,
                                    f"{subject_ticker}_Trading_Comps.pptx", "comps_pptx_btn")
            except Exception:
                pass
            try:
                from investment_memo import MemoContext
                ctx = MemoContext(
                    title=f"{subject_ticker} — Trading Comps", ticker=subject_ticker,
                    subject_type="Comps", fair_value=blended, current_price=s_price,
                    valuation_narrative=result.conclusion,
                    valuation_methods={k: {"low": lo, "high": hi, "mid": blended,
                                           "weight": 1.0 / max(len(iv), 1)} for k in iv},
                )
                _render_memo_button(ctx, "comps_memo_btn")
            except Exception:
                pass

    except Exception as e:
        st.error(f"Comps Engine Error: {e}")
        import traceback
        st.code(traceback.format_exc())


# ═════════════════════════════════════════════════════════════════════════════
# PRECEDENT TRANSACTIONS TAB
# ═════════════════════════════════════════════════════════════════════════════

def _render_precedents_tab():
    st.subheader("Precedent Transactions Analysis")
    st.caption("Deal universe with relevance scoring, transaction multiples, control-premium analysis "
               "and an adjusted valuation range for the subject.")
    try:
        from precedent_transactions_engine import (
            PrecedentTransaction, PrecedentInputs, get_precedent_engine)

        c1, c2, c3 = st.columns(3)
        subj_name = c1.text_input("Subject Name", value="Subject Company",
                                 key="prec_subj_name")
        subj_sector = c2.text_input("Subject Sector", value="Technology",
                                    key="prec_subj_sector")
        regime = c3.selectbox("Current Market Regime", ["bull", "neutral", "bear"],
                              index=1, key="prec_subj_regime")

        s1, s2, s3, s4 = st.columns(4)
        s_rev = s1.number_input("Subject Revenue ($M)", value=1000.0, step=50.0,
                                key="prec_subj_rev")
        s_growth = s2.number_input("Subject Growth (%)", value=20.0, step=1.0,
                                   key="prec_subj_growth")
        s_ebitda = s3.number_input("Subject EBITDA ($M)", value=250.0, step=10.0,
                                   key="prec_subj_ebitda")
        s_margin = s4.number_input("Subject EBITDA Margin (%)", value=25.0, step=1.0,
                                   key="prec_subj_margin")
        s5, s6, s7 = st.columns(3)
        s_nd = s5.number_input("Subject Net Debt ($M)", value=100.0, step=10.0,
                               key="prec_subj_nd")
        s_shares = s6.number_input("Subject Shares (M)", value=50.0, step=1.0,
                                   key="prec_subj_shares")
        s_fcf = s7.number_input("Subject FCF ($M)", value=180.0, step=10.0,
                                key="prec_subj_fcf")

        n_deals = st.slider("Number of Transactions", 2, 8, 4, key="prec_n_deals")
        deals = []
        for i in range(n_deals):
            with st.expander(f"Transaction {i + 1}", expanded=(i == 0)):
                d1, d2, d3 = st.columns(3)
                dname = d1.text_input(f"Deal Name {i + 1}", value=f"Deal {i + 1}", key=f"prec_t{i}_name")
                buyer = d2.text_input(f"Buyer {i + 1}", value="Acquirer", key=f"prec_t{i}_buyer")
                date_str = d3.text_input(f"Announced {i + 1}", value="2024-06-01", key=f"prec_t{i}_date")
                m1, m2, m3, m4 = st.columns(4)
                ev = m1.number_input(f"Enterprise Value {i + 1} ($M)", value=1500.0 + i * 150,
                                     step=50.0, key=f"prec_t{i}_ev")
                rev = m2.number_input(f"Target Revenue {i + 1} ($M)", value=500.0 + i * 50,
                                      step=25.0, key=f"prec_t{i}_rev")
                ebitda = m3.number_input(f"Target EBITDA {i + 1} ($M)", value=120.0 + i * 10,
                                         step=5.0, key=f"prec_t{i}_ebitda")
                prem = m4.number_input(f"Premium {i + 1} (%)", value=28.0 + i * 2, step=1.0,
                                       key=f"prec_t{i}_prem")
                x1, x2, x3 = st.columns(3)
                buyer_type = x1.selectbox(f"Buyer Type {i + 1}", ["strategic", "financial", "mixed"],
                                          key=f"prec_t{i}_type")
                structure = x2.selectbox(f"Structure {i + 1}", ["cash", "stock", "mixed", "earnout"],
                                         key=f"prec_t{i}_struct")
                d_regime = x3.selectbox(f"Regime {i + 1}", ["bull", "neutral", "bear"],
                                        index=1, key=f"prec_t{i}_regime")
                deals.append({
                    "deal_name": dname, "buyer": buyer, "announcement_date": date_str,
                    "enterprise_value": ev, "seller_revenue": rev, "seller_ebitda": ebitda,
                    "control_premium_pct": prem, "buyer_type": buyer_type,
                    "structure": structure, "market_regime": d_regime,
                    "strategic_rationale": subj_sector,
                })

        run_prec = st.button("Run Precedent Analysis", type="primary")

        if run_prec:
            txns = [PrecedentTransaction(**d) for d in deals]
            inputs = PrecedentInputs(
                subject_name=subj_name, subject_sector=subj_sector,
                subject_revenue=s_rev, subject_revenue_growth_pct=s_growth,
                subject_ebitda=s_ebitda, subject_ebitda_margin_pct=s_margin,
                subject_fcf=s_fcf, subject_net_debt=s_nd, subject_shares=s_shares,
                current_market_regime=regime, transactions=txns)
            result = get_precedent_engine().run_precedents(inputs)
            st.session_state["precedents_result"] = result
            st.session_state["bridge_context"] = dict(
                st.session_state.get("bridge_context", {}), precedents=result, ticker=subj_name)

            iv = result.implied_valuation
            blended = iv.get("Blended (weighted)", 0)
            lo, hi = result.adjusted_range
            k1, k2, k3 = st.columns(3)
            k1.metric("Precedent-Derived Control Value", f"${blended:.2f}")
            k2.metric("Adjusted Range", f"${lo:.2f} – ${hi:.2f}")
            k3.metric("Core Deals", f"{len(result.core_deals)}")

            st.markdown("### Transaction Multiples")
            st.dataframe(result.multiples_df, width='stretch', hide_index=True)

            st.markdown("### Multiple Statistics (Core Deal Set)")
            stat_rows = []
            for mult, s in result.stats.items():
                if s.get("count", 0) == 0:
                    continue
                stat_rows.append({"Multiple": mult, "Mean": f"{s['mean']:.1f}x",
                                  "Median": f"{s['median']:.1f}x", "Q1": f"{s['q1']:.1f}x",
                                  "Q3": f"{s['q3']:.1f}x", "High": f"{s['high']:.1f}x"})
            st.dataframe(pd.DataFrame(stat_rows), width='stretch', hide_index=True)

            st.markdown("### Implied Valuation & Adjusted Range")
            imp_rows = [{"Method": k, "Value ($)": f"${v:,.2f}"} for k, v in iv.items()]
            st.dataframe(pd.DataFrame(imp_rows), width='stretch', hide_index=True)

            st.markdown("### Control Premium Analysis")
            cp = result.control_premium_analysis
            c1, c2, c3 = st.columns(3)
            c1.metric("Median Premium", f"{cp.get('median_premium_pct') or 0:.1f}%")
            c2.metric("Mean Premium", f"{cp.get('mean_premium_pct') or 0:.1f}%")
            c3.metric("Premium Range", f"{cp.get('low_premium_pct') or 0:.1f}% – {cp.get('high_premium_pct') or 0:.1f}%")
            st.caption(cp.get("context", ""))

            st.markdown("### Context Adjustments")
            for adj in result.context_adjustments:
                st.info(f"**{adj.get('factor', '')}** — {adj.get('note', '')}")

            st.markdown("### Conclusion")
            st.info(result.conclusion)

            try:
                from ib_excel_engine import build_precedents_workbook
                prec_xlsx = build_precedents_workbook(result)
                st.download_button("Download Precedents Workbook (Excel)",
                                   data=prec_xlsx,
                                   file_name=f"{subj_name.replace(' ', '_')}_Precedents.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   key="prec_xlsx_btn")
            except Exception:
                pass
            try:
                from presentation_generator import get_presentation_generator
                prec_pptx = get_presentation_generator().generate_precedents_pitchbook(result, subj_name)
                _render_pptx_button("Download Precedents Pitchbook (.pptx)", prec_pptx,
                                    f"{subj_name.replace(' ', '_')}_Precedents.pptx", "prec_pptx_btn")
            except Exception:
                pass
            try:
                from investment_memo import MemoContext
                ctx = MemoContext(
                    title=f"{subj_name} — Precedent Transactions", ticker=subj_name,
                    subject_type="M&A", fair_value=blended, current_price=None,
                    valuation_narrative=result.conclusion,
                )
                _render_memo_button(ctx, "prec_memo_btn")
            except Exception:
                pass

    except Exception as e:
        st.error(f"Precedents Engine Error: {e}")
        import traceback
        st.code(traceback.format_exc())


# ═════════════════════════════════════════════════════════════════════════════
# IPO TAB
# ═════════════════════════════════════════════════════════════════════════════

def _render_ipo_tab():
    st.subheader("IPO Analysis & Underwriting")
    st.caption("Market pulse, share-count bridge (TSM), multi-method pricing, bookbuilding demand "
               "framework and dynamic scenario case studies.")
    try:
        from ipo_engine import IPOAssumptions, get_ipo_engine
        from market_consensus_engine import default_market_pulse_inputs

        c1, c2, c3 = st.columns(3)
        ipo_ticker = c1.text_input("Ticker", value="SUBJ", key="ipo_ticker_in")
        ipo_name = c2.text_input("Company Name", value="Subject Company", key="ipo_name_in")
        ipo_sector = c3.text_input("Sector", value="Technology", key="ipo_sector_in")

        f1, f2, f3, f4 = st.columns(4)
        ipo_rev = f1.number_input("Revenue ($M)", value=1000.0, step=50.0, key="ipo_rev_in")
        ipo_growth = f2.number_input("Revenue Growth (%)", value=25.0, step=1.0,
                                     key="ipo_growth_in")
        ipo_ebitda = f3.number_input("EBITDA ($M)", value=250.0, step=10.0, key="ipo_ebitda_in")
        ipo_em = f4.number_input("EBITDA Margin (%)", value=25.0, step=1.0, key="ipo_em_in")
        f5, f6, f7, f8 = st.columns(4)
        ipo_ebit = f5.number_input("EBIT ($M)", value=200.0, step=10.0, key="ipo_ebit_in")
        ipo_fcf = f6.number_input("FCF ($M)", value=180.0, step=10.0, key="ipo_fcf_in")
        ipo_ni = f7.number_input("Net Income ($M)", value=150.0, step=10.0, key="ipo_ni_in")
        ipo_nd = f8.number_input("Net Debt ($M)", value=100.0, step=10.0, key="ipo_nd_in")
        ipo_shares = st.number_input("Shares Outstanding Pre-IPO (M)", value=50.0, step=1.0,
                                     key="ipo_shares_in")

        with st.expander("Dilutive Securities & Offering Structure"):
            d1, d2, d3, d4 = st.columns(4)
            ipo_opt = d1.number_input("Options (M)", value=5.0, step=0.5, key="ipo_opt_in")
            ipo_opt_k = d2.number_input("Options Strike ($)", value=20.0, step=1.0,
                                        key="ipo_opt_k_in")
            ipo_warr = d3.number_input("Warrants (M)", value=2.0, step=0.5, key="ipo_warr_in")
            ipo_warr_k = d4.number_input("Warrants Strike ($)", value=30.0, step=1.0,
                                         key="ipo_warr_k_in")
            e1, e2, e3, e4 = st.columns(4)
            ipo_rsu = e1.number_input("RSUs (M)", value=3.0, step=0.5, key="ipo_rsu_in")
            ipo_conv = e2.number_input("Convertible ($M)", value=0.0, step=25.0,
                                       key="ipo_conv_in")
            ipo_conv_p = e3.number_input("Convertible Premium (%)", value=20.0, step=5.0,
                                         key="ipo_conv_p_in")
            ipo_pref = e4.number_input("Preferred (M)", value=0.0, step=1.0, key="ipo_pref_in")
            o1, o2, o3 = st.columns(3)
            ipo_prim = o1.number_input("Primary Shares (M)", value=12.0, step=1.0,
                                       key="ipo_prim_in")
            ipo_sec = o2.number_input("Secondary Shares (M)", value=5.0, step=1.0,
                                      key="ipo_sec_in")
            ipo_spread = o3.number_input("Gross Spread (%)", value=7.0, step=0.5,
                                         key="ipo_spread_in")

        with st.expander("Valuation Multiples (Comp Context)"):
            v1, v2, v3, v4 = st.columns(4)
            peer_evr = v1.number_input("Peer EV/Revenue", value=6.0, step=0.5,
                                       key="ipo_peer_evr")
            peer_eve = v2.number_input("Peer EV/EBITDA", value=18.0, step=0.5,
                                       key="ipo_peer_eve")
            peer_evi = v3.number_input("Peer EV/EBIT", value=24.0, step=0.5,
                                       key="ipo_peer_evi")
            peer_pe = v4.number_input("Peer P/E", value=30.0, step=1.0, key="ipo_peer_pe")
            w1, w2, w3 = st.columns(3)
            ipo_evr = w1.number_input("IPO Comp EV/Revenue", value=8.0, step=0.5,
                                      key="ipo_evr_in")
            ipo_eve = w2.number_input("IPO Comp EV/EBITDA", value=22.0, step=0.5,
                                      key="ipo_eve_in")
            prev_eve = w3.number_input("Precedent EV/EBITDA", value=25.0, step=0.5,
                                       key="ipo_prev_eve")

        with st.expander("Market Pulse & Demand (estimates — replace with observed data where possible)"):
            mp = default_market_pulse_inputs()
            p1, p2, p3, p4 = st.columns(4)
            vix = p1.number_input("VIX", value=float(mp.vix_level or 18), step=0.5)
            ten_y = p2.number_input("10Y Yield (%)", value=float(mp.ten_year_yield_pct or 4.2), step=0.05)
            credit = p3.number_input("Credit Spread (bps)", value=float(mp.credit_spread_bps or 140), step=5.0)
            risk_app = p4.number_input("Risk Appetite (0-100)", value=float(mp.risk_appetite or 60), step=1.0)
            q1, q2, q3, q4 = st.columns(4)
            fd_ret = q1.number_input("Recent IPO 1st-Day Ret (%)", value=float(mp.recent_ipo_avg_first_day_return_pct or 18), step=1.0)
            pop = q2.number_input("Recent IPO Pop (%)", value=float(mp.recent_ipo_pop_pct or 65), step=1.0)
            withdrawals = q3.number_input("Withdrawals (12m)", value=float(mp.ipo_withdrawals_12m or 4), step=1)
            oversub = q4.number_input("Expected Oversubscription (x)", value=5.0, step=0.5)
            regime_sel = st.selectbox("Equity Regime", ["bull", "neutral", "bear"],
                                      index=1, key="ipo_regime_sel")
            sec_perf = st.number_input("Sector 3M Perf (%)", value=6.0, step=0.5,
                                       key="ipo_sec_perf")

        run_ipo = st.button("Run IPO Analysis", type="primary")

        if run_ipo:
            pulse_inputs = default_market_pulse_inputs()
            pulse_inputs.vix_level = vix
            pulse_inputs.ten_year_yield_pct = ten_y
            pulse_inputs.credit_spread_bps = credit
            pulse_inputs.risk_appetite = risk_app
            pulse_inputs.recent_ipo_avg_first_day_return_pct = fd_ret
            pulse_inputs.recent_ipo_pop_pct = pop
            pulse_inputs.ipo_withdrawals_12m = int(withdrawals)
            pulse_inputs.ipo_postponements_12m = max(0, int(withdrawals) - 2)
            pulse_inputs.equity_regime = regime_sel
            pulse_inputs.sector_performance_3m_pct = sec_perf

            assumptions = IPOAssumptions(
                ticker=ipo_ticker, company_name=ipo_name, sector=ipo_sector,
                revenue=ipo_rev, revenue_growth_pct=ipo_growth,
                ebitda=ipo_ebitda, ebitda_margin_pct=ipo_em,
                ebit=ipo_ebit, fcf=ipo_fcf, net_income=ipo_ni, net_debt=ipo_nd,
                shares_outstanding_m=ipo_shares,
                options_m=ipo_opt, options_strike=ipo_opt_k,
                warrants_m=ipo_warr, warrants_strike=ipo_warr_k,
                rsus_m=ipo_rsu, convertible_m=ipo_conv,
                convertible_premium_pct=ipo_conv_p, preferred_m=ipo_pref,
                primary_shares_m=ipo_prim, secondary_shares_m=ipo_sec,
                gross_spread_pct=ipo_spread,
                peer_ev_revenue=peer_evr, peer_ev_ebitda=peer_eve,
                peer_ev_ebit=peer_evi, peer_pe=peer_pe,
                ipo_comp_ev_revenue=ipo_evr, ipo_comp_ev_ebitda=ipo_eve,
                precedent_ev_ebitda=prev_eve,
                base_revenue=ipo_rev, dcf_growth_rates=[ipo_growth / 100.0] * 5,
                dcf_ebit_margin=max(ipo_em / 100.0 - 0.05, 0.05),
                market_pulse=pulse_inputs,
                expected_oversubscription_x=oversub,
            )
            result = get_ipo_engine().run_ipo(assumptions)
            st.session_state["ipo_result"] = result
            st.session_state["bridge_context"] = dict(
                st.session_state.get("bridge_context", {}), ipo=result, ticker=ipo_ticker)

            try:
                from model_audit import run_model_qa
                _render_qa(run_model_qa(result, "ipo"), "IPO")
            except Exception:
                pass

            lo, hi = result.price_range
            st.markdown("### Indicative IPO Pricing")
            p1, p2, p3, p4, p5 = st.columns(5)
            p1.metric("Range Low", f"${lo:.2f}")
            p2.metric("Range High", f"${hi:.2f}")
            p3.metric("Prob-Weighted Price", f"${result.implied_ipo_price:.2f}")
            p4.metric("Gross Proceeds", f"${result.gross_proceeds:,.0f}M")
            p5.metric("Net Proceeds", f"${result.net_proceeds:,.0f}M")
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Market Cap @ Mid", f"${result.market_cap_at_mid:,.0f}M")
            q2.metric("Primary Dilution", f"{result.dilution_pct:.1f}%")
            q3.metric("Expected 1st-Day Return (est.)", f"{result.expected_first_day_return_pct:+.1f}%")
            q4.metric("Market Environment", result.market_pulse.get("label", ""))

            st.markdown("### Market Pulse — Why This Reading")
            pulse = result.market_pulse
            st.info(pulse.get("explanation", ""))
            ev_rows = [{"Signal": e["signal"], "Score": f"{e['score']:+.2f}",
                        "Detail": e["detail"]} for e in pulse.get("evidence", [])]
            if ev_rows:
                with st.expander("Signal-level evidence"):
                    st.dataframe(pd.DataFrame(ev_rows), width='stretch', hide_index=True)

            st.markdown("### Multi-Method Valuation")
            mrows = [{"Method": k, "Implied EV ($M)": f"${v['ev']:,.0f}",
                      "Multiple": f"{v['multiple']:.1f}x" if v.get("multiple") else "—",
                      "Weight": f"{v['weight']:.1f}"} for k, v in result.valuation_methods.items()]
            st.dataframe(pd.DataFrame(mrows), width='stretch', hide_index=True)
            st.caption(result.pricing.get("methodology_note", ""))
            for w in result.pricing.get("weights", []):
                st.markdown(f"- **{w['method']}** ({w['weight']:.0%} weight): {w['why']}")

            st.markdown("### Share-Count Bridge (Treasury Stock Method)")
            bridge = result.share_count_bridge
            br_rows = [{"Component": "Existing shares", "M": bridge.get("existing_shares", 0)},
                       {"Component": "Options (net, TSM)", "M": bridge.get("options_net_tsm", 0)},
                       {"Component": "Warrants (net, TSM)", "M": bridge.get("warrants_net_tsm", 0)},
                       {"Component": "RSUs", "M": bridge.get("rsus", 0)},
                       {"Component": "Convertible shares", "M": bridge.get("convertible_shares", 0)},
                       {"Component": "Preferred shares", "M": bridge.get("preferred_shares", 0)},
                       {"Component": "Primary new shares", "M": bridge.get("primary_new_shares", 0)},
                       {"Component": "Total post-IPO", "M": bridge.get("total_shares_post_ipo", 0)}]
            st.dataframe(pd.DataFrame(br_rows), width='stretch', hide_index=True)
            st.caption(bridge.get("method", ""))

            st.markdown("### Bookbuilding / Demand Framework")
            bk = result.bookbuilding
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Oversubscription", f"{bk.get('expected_oversubscription_x', 0):.1f}x")
            b2.metric("Free Float", f"{bk.get('free_float_pct', 0):.1f}%")
            b3.metric("Price Pressure", bk.get("price_pressure", "").split("—")[0])
            b4.metric("Aftermarket", bk.get("aftermarket_outlook", "")[:28])
            st.warning(bk.get("disclosure", ""))

            st.markdown("### IPO Scenario Cases")
            sc_cols = st.columns(len(result.scenarios))
            for i, s in enumerate(result.scenarios):
                c = ["#ff7043", "#e0c97f", "#00c853", "#b39ddb"][i % 4]
                with sc_cols[i]:
                    st.markdown(
                        f"<div style='background:#161b22;border:1px solid {c};border-radius:6px;"
                        f"padding:10px;text-align:center;'>"
                        f"<div style='color:{c};font-weight:700;font-size:0.85rem;'>{s.label}</div>"
                        f"<div style='color:#aaa;font-size:0.72rem;'>P = {s.probability:.0%}</div>"
                        f"<div style='color:white;font-size:1.3rem;font-weight:800;'>${s.price:.2f}</div>"
                        f"<div style='color:#aaa;font-size:0.68rem;'>EV ${s.valuation:,.0f}M</div></div>",
                        unsafe_allow_html=True)
            with st.expander("Scenario detail — triggers, assumptions, risks"):
                for s in result.scenarios:
                    st.markdown(f"**{s.label}** (P = {s.probability:.0%}) — {s.rationale}")
                    st.markdown(f"- Triggers: {'; '.join(s.triggers)}")
                    st.markdown(f"- Key assumptions: {'; '.join(s.key_assumptions)}")
                    st.markdown(f"- Risks: {'; '.join(s.risks)}")
                    st.markdown("---")

            st.markdown("### Conclusion")
            st.info(result.conclusion)

            # Downloads
            try:
                from ib_excel_engine import build_ipo_workbook
                ipo_xlsx = build_ipo_workbook(result)
                st.download_button("Download IPO Model (Excel)",
                                   data=ipo_xlsx,
                                   file_name=f"{ipo_ticker}_IPO_Model.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   key="ipo_xlsx_btn")
                st.caption("Yellow cells = analyst inputs; black cells = live formulas. Includes share-count bridge, multi-method valuation, pricing/proceeds, scenarios and checks.")
            except Exception as e:
                st.warning(f"Excel export unavailable ({e})")
            try:
                from presentation_generator import get_presentation_generator
                ipo_pptx = get_presentation_generator().generate_ipo_pitchbook(result, ipo_name)
                _render_pptx_button("Download IPO Pitchbook (.pptx)", ipo_pptx,
                                    f"{ipo_ticker}_IPO_Pitchbook.pptx", "ipo_pptx_btn")
            except Exception:
                pass
            try:
                from investment_memo import MemoContext
                ctx = MemoContext(
                    title=f"{ipo_name} — IPO Analysis", ticker=ipo_ticker,
                    subject_type="IPO", fair_value=result.implied_ipo_price,
                    current_price=None,
                    market_pulse_label=result.market_pulse.get("label", ""),
                    market_pulse_explanation=result.market_pulse.get("explanation", ""),
                    valuation_narrative=result.pricing.get("methodology_note", ""),
                    scenarios=[{"label": s.label, "probability": s.probability,
                                "price": s.price, "rationale": s.rationale}
                               for s in result.scenarios],
                    risks=[t for s in result.scenarios for t in s.risks][:8],
                    catalysts=[t for s in result.scenarios for t in s.catalysts][:8],
                )
                _render_memo_button(ctx, "ipo_memo_btn")
            except Exception:
                pass

    except Exception as e:
        st.error(f"IPO Engine Error: {e}")
        import traceback
        st.code(traceback.format_exc())


# ═════════════════════════════════════════════════════════════════════════════
# CROSS-MODEL VALUATION BRIDGE TAB
# ═════════════════════════════════════════════════════════════════════════════

def _render_bridge_tab():
    st.subheader("Cross-Model Valuation Bridge")
    st.caption("Unify DCF, Comps, Precedents, IPO, LBO and M&A into one valuation dashboard. "
               "Run the individual models in the other tabs first — their outputs flow in automatically.")
    try:
        from valuation_bridge import get_valuation_bridge

        ctx_bridge = st.session_state.get("bridge_context", {}) or {}
        price = ctx_bridge.get("price") or st.number_input(
            "Current Market Price ($)", value=40.0, step=1.0, key="bridge_price_in")
        ticker = ctx_bridge.get("ticker", "")

        bridge = get_valuation_bridge()
        methods = []
        available = []

        if ctx_bridge.get("dcf") is not None:
            m = bridge.add_dcf(ctx_bridge["dcf"])
            if m:
                methods.append(m)
                available.append("DCF")
        if ctx_bridge.get("comps") is not None:
            m = bridge.add_comps(ctx_bridge["comps"])
            if m:
                methods.append(m)
                available.append("Trading Comps")
        if ctx_bridge.get("precedents") is not None:
            m = bridge.add_precedents(ctx_bridge["precedents"])
            if m:
                methods.append(m)
                available.append("Precedents")
        if ctx_bridge.get("ipo") is not None:
            m = bridge.add_ipo(ctx_bridge["ipo"])
            if m:
                methods.append(m)
                available.append("IPO")
        if ctx_bridge.get("lbo") is not None:
            m = bridge.add_lbo(ctx_bridge["lbo"], shares=1.0)
            if m:
                methods.append(m)
                available.append("LBO")
        if ctx_bridge.get("mna") is not None:
            m = bridge.add_mna(ctx_bridge["mna"], shares=1.0)
            if m:
                methods.append(m)
                available.append("M&A")

        if ctx_bridge.get("dcf") is not None:
            methods.append(bridge.add_market(float(price or 0)))
            available.append("Market")

        if not methods:
            st.info(
                "No model outputs yet. Run at least one model (e.g. **Run Institutional DCF**) in "
                "the tabs above — the bridge will pull every available valuation in automatically.")
            return

        st.success(f"Valuation methods available: {', '.join(available)}")

        if st.button("Build Cross-Model Valuation Bridge", type="primary") or "bridge_built" in st.session_state:
            st.session_state["bridge_built"] = True
            result = bridge.run(methods, current_price=float(price or 0))
            st.session_state["bridge_result"] = result

            try:
                from model_audit import run_model_qa
                _render_qa(run_model_qa(result, "bridge"), "Valuation Bridge")
            except Exception:
                pass

            st.markdown("### Valuation Statement")
            st.markdown(
                f"<div style='background:#161b22;border-left:4px solid #c9a84c;border-radius:6px;"
                f"padding:14px;font-size:1.05rem;color:white;'>{result.valuation_statement}</div>",
                unsafe_allow_html=True)

            st.markdown("### Methodologies & Weights")
            rows = [{"Method": m.label, "Low ($)": f"${m.low:.2f}",
                     "High ($)": f"${m.high:.2f}", "Mid ($)": f"${m.mid:.2f}",
                     "Weight": f"{result.weights.get(m.key, 0):.0%}",
                     "Source": m.source, "Note": m.note} for m in result.methods]
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

            st.markdown("### Weighted Range & Convergence")
            lo, hi = result.weighted_range
            c1, c2, c3 = st.columns(3)
            c1.metric("Weighted Low", f"${lo:.2f}")
            c2.metric("Weighted Mid", f"${result.weighted_mid:.2f}")
            c3.metric("Weighted High", f"${hi:.2f}")
            conv_color = "#00c853" if "High convergence" in result.convergence else (
                "#e0c97f" if "Moderate" in result.convergence else "#ff7043")
            st.markdown(
                f"<div style='background:#161b22;border-left:4px solid {conv_color};border-radius:6px;"
                f"padding:10px 14px;'><span style='color:{conv_color};font-weight:700;'>"
                f"{result.convergence}</span></div>", unsafe_allow_html=True)

            st.markdown("### Drivers")
            for d in result.drivers:
                st.markdown(f"- {d}")

            if result.outliers:
                st.markdown("### Outliers")
                for o in result.outliers:
                    st.warning(o)

            # Football-field chart
            fig = go.Figure()
            for i, m in enumerate(result.methods):
                fig.add_trace(go.Bar(
                    name=m.label, x=[m.mid], y=[m.label], orientation="h",
                    marker=dict(color="#b39ddb"),
                    error_x=dict(type="data", symmetric=False,
                                 array=[m.high - m.mid], arrayminus=[m.mid - m.low]),
                    text=[f"${m.mid:.2f}"], textposition="outside"))
            fig.add_vline(x=float(price or 0), line_dash="dash", line_color="#ff7043",
                          annotation_text=f"Market ${price:.2f}")
            fig.update_layout(title="Valuation Football Field", template="plotly_dark",
                              height=360, xaxis_title="$ / share", barmode="overlay",
                              showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, width='stretch')

            # Narrative
            st.markdown("### Narrative")
            st.info(result.narrative)

            # Downloads
            try:
                from ib_excel_engine import build_bridge_workbook
                bridge_xlsx = build_bridge_workbook(result)
                st.download_button("Download Valuation Bridge (Excel)",
                                   data=bridge_xlsx,
                                   file_name=f"{ticker or 'Company'}_Valuation_Bridge.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   key="bridge_xlsx_btn")
            except Exception:
                pass
            try:
                from investment_memo import MemoContext
                ctx = MemoContext(
                    title=f"{ticker or 'Company'} — Cross-Model Valuation",
                    ticker=ticker, subject_type="Equity",
                    fair_value=result.weighted_mid, current_price=float(price or 0),
                    valuation_narrative=result.narrative,
                    valuation_methods={m.label: {"low": m.low, "high": m.high, "mid": m.mid,
                                                 "weight": result.weights.get(m.key, 0)}
                                       for m in result.methods},
                    risks=result.outliers[:6] or ["No structured risks supplied."],
                )
                _render_memo_button(ctx, "bridge_memo_btn")
            except Exception:
                pass

    except Exception as e:
        st.error(f"Valuation Bridge Error: {e}")
        import traceback
        st.code(traceback.format_exc())
