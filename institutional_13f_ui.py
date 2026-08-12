"""
Institutional 13F Analysis UI — Comprehensive Multi-Asset View
Covers equities, options, bonds, commodities, and all asset classes.
Author: Octavian Research
"""

from typing import Dict, List, Any

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sec_13f_engine import get_sec_13f_engine, InstitutionalFiling
from octavian_theme import section_header, COLORS, PLOTLY_TEMPLATE

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

_DARK_TEMPLATE = "plotly_dark"
_PAPER_BG = COLORS["navy"]
_PLOT_BG = COLORS["navy_light"]
_FONT_COLOR = COLORS["text_primary"]
_GOLD = COLORS["gold"]
_LAVENDER = COLORS["lavender"]


def _chart_layout(fig, title: str = "", height: int = 400) -> go.Figure:
    """Apply consistent dark Octavian styling to any plotly figure."""
    fig.update_layout(
        template=_DARK_TEMPLATE,
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_PLOT_BG,
        font=dict(color=_FONT_COLOR, family="Inter, sans-serif"),
        title=dict(text=title, font=dict(color=_GOLD, size=14)) if title else None,
        height=height,
        margin=dict(t=40 if title else 20, l=20, r=20, b=20),
        legend=dict(
            bgcolor="rgba(10,22,40,0.8)",
            bordercolor=COLORS["border"],
            borderwidth=1,
        ),
    )
    return fig


def _fmt_b(val: float) -> str:
    """Format a dollar value in billions."""
    if abs(val) >= 1e9:
        return f"${val/1e9:.2f}B"
    if abs(val) >= 1e6:
        return f"${val/1e6:.0f}M"
    return f"${val:,.0f}"


def _action_color(action: str) -> str:
    mapping = {
        "NEW": "#4caf50", "ADD": "#81c784",
        "REDUCE": "#ef9a9a", "EXIT": "#ef5350", "HOLD": "#78909c",
    }
    return mapping.get(action, "#78909c")



# ---------------------------------------------------------------------------
# TAB 1 — GLOBAL SMART MONEY FLOW
# ---------------------------------------------------------------------------

def _render_tab_global_flow(engine):
    section_header("Net Institutional Flow")

    flows = engine.get_global_smart_money_flow()
    asset_flows = engine.get_asset_class_flows()
    options_summary = engine.get_options_sentiment_summary()

    # --- Top-level metrics ---
    col1, col2, col3, col4 = st.columns(4)
    total_inflow = sum(v for _, v in flows["top_inflows"])
    total_outflow = sum(v for _, v in flows["top_outflows"])
    with col1:
        st.metric("Total Tracked Inflow", _fmt_b(total_inflow))
    with col2:
        st.metric("Total Tracked Outflow", _fmt_b(abs(total_outflow)))
    with col3:
        st.metric(
            "Aggregate Put/Call Ratio",
            f"{options_summary['avg_put_call_ratio']:.2f}",
            delta=options_summary["aggregate_sentiment"],
        )
    with col4:
        filings = engine.fetch_latest_filings()
        avg_duration = sum(f.bond_duration for f in filings) / len(filings) if filings else 0
        st.metric("Avg Bond Duration", f"{avg_duration:.1f} yrs")

    st.divider()

    # --- Flow Treemap ---
    section_header("Equity Flow Heatmap")
    flow_df = pd.DataFrame(
        list(flows["net_flow_map"].items()), columns=["Symbol", "Net Flow"]
    )
    flow_df["Flow Type"] = flow_df["Net Flow"].apply(
        lambda x: "Inflow" if x > 0 else "Outflow"
    )
    flow_df["Absolute Flow"] = flow_df["Net Flow"].abs()
    flow_df = flow_df[flow_df["Absolute Flow"] > 0]

    if not flow_df.empty:
        fig_tree = px.treemap(
            flow_df,
            path=[px.Constant("All Positions"), "Flow Type", "Symbol"],
            values="Absolute Flow",
            color="Net Flow",
            color_continuous_scale=["#ef5350", "#78909c", "#4caf50"],
            color_continuous_midpoint=0,
        )
        fig_tree = _chart_layout(fig_tree, height=380)
        fig_tree.update_traces(textinfo="label+value")
        st.plotly_chart(fig_tree, use_container_width=True)

    st.divider()

    # --- Asset Class Flow Bar Chart ---
    section_header("Asset Class Net Flow")
    ac_flow = asset_flows["asset_class_net_flow"]
    if ac_flow:
        ac_df = pd.DataFrame(
            [(k, v) for k, v in ac_flow.items()], columns=["Asset Class", "Net Flow"]
        )
        ac_df["Color"] = ac_df["Net Flow"].apply(
            lambda x: COLORS["success"] if x > 0 else COLORS["danger"]
        )
        fig_ac = go.Figure(
            go.Bar(
                x=ac_df["Asset Class"],
                y=ac_df["Net Flow"] / 1e9,
                marker_color=ac_df["Color"].tolist(),
                text=[f"${v/1e9:.1f}B" for v in ac_df["Net Flow"]],
                textposition="outside",
            )
        )
        fig_ac = _chart_layout(fig_ac, "Net Flow by Asset Class (USD Billions)", height=320)
        fig_ac.update_yaxes(title_text="Net Flow ($B)")
        st.plotly_chart(fig_ac, use_container_width=True)

    st.divider()

    # --- Options Sentiment Panel ---
    section_header("Options Sentiment Panel")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric(
            "Aggregate Put/Call Ratio",
            f"{options_summary['avg_put_call_ratio']:.2f}",
        )
        sentiment = options_summary["aggregate_sentiment"]
        color = (
            COLORS["danger"] if "Hedging" in sentiment
            else COLORS["success"] if "Long" in sentiment
            else COLORS["neutral"]
        )
        st.markdown(
            f'<div style="color:{color};font-weight:600;font-size:1.1rem;">'
            f'{sentiment}</div>',
            unsafe_allow_html=True,
        )
    with col_b:
        st.metric(
            "Total Call Notional",
            _fmt_b(options_summary["total_call_notional"]),
        )
    with col_c:
        st.metric(
            "Total Put Notional",
            _fmt_b(options_summary["total_put_notional"]),
        )

    # Sentiment distribution bar
    sent_dist = options_summary["sentiment_distribution"]
    if sent_dist:
        sent_df = pd.DataFrame(
            list(sent_dist.items()), columns=["Sentiment", "Fund Count"]
        )
        fig_sent = px.bar(
            sent_df, x="Sentiment", y="Fund Count",
            color="Sentiment",
            color_discrete_map={
                "Hedging": COLORS["danger"],
                "Speculative Long": COLORS["success"],
                "Speculative Short": "#ef5350",
                "Neutral": COLORS["neutral"],
            },
        )
        fig_sent = _chart_layout(fig_sent, "Options Sentiment Distribution Across Funds", height=280)
        st.plotly_chart(fig_sent, use_container_width=True)

    st.divider()

    # --- Bond Market Positioning Summary ---
    section_header("Bond Market Positioning")
    bond_data = engine.get_bond_market_positioning()
    col_x, col_y, col_z = st.columns(3)
    with col_x:
        st.metric(
            "Total Bond AUM",
            _fmt_b(bond_data["total_market_value"]),
        )
    with col_y:
        st.metric(
            "Weighted Avg Duration",
            f"{bond_data['weighted_avg_duration']:.1f} yrs",
        )
    with col_z:
        treasury_val = bond_data["bond_type_breakdown"].get("Treasury", 0)
        corp_val = sum(
            v for k, v in bond_data["bond_type_breakdown"].items()
            if "Corporate" in k
        )
        ratio = treasury_val / corp_val if corp_val > 0 else float("inf")
        st.metric("Treasury / Corporate Ratio", f"{ratio:.1f}x")

    # Duration buckets
    dur_buckets = bond_data["duration_buckets"]
    if dur_buckets:
        dur_df = pd.DataFrame(
            list(dur_buckets.items()), columns=["Duration Bucket", "AUM"]
        )
        fig_dur = px.bar(
            dur_df, x="Duration Bucket", y="AUM",
            color="Duration Bucket",
            color_discrete_sequence=[COLORS["gold"], COLORS["lavender"], COLORS["success"]],
        )
        fig_dur = _chart_layout(fig_dur, "Bond Duration Positioning ($)", height=280)
        fig_dur.update_yaxes(title_text="Market Value ($)")
        st.plotly_chart(fig_dur, use_container_width=True)


# ---------------------------------------------------------------------------
# TAB 2 — OPTIONS INTELLIGENCE
# ---------------------------------------------------------------------------

def _render_tab_options(engine):
    section_header("Cross-Fund Options Flow")

    options_flow = engine.get_cross_fund_options_flow()
    options_sentiment = engine.get_options_sentiment_summary()

    # Aggregate metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Aggregate Put/Call Ratio",
            f"{options_flow['aggregate_pc_ratio']:.2f}",
        )
    with col2:
        st.metric(
            "Total Call Notional",
            _fmt_b(options_flow["total_call_notional"]),
        )
    with col3:
        st.metric(
            "Total Put Notional",
            _fmt_b(options_flow["total_put_notional"]),
        )
    with col4:
        st.metric(
            "Market Sentiment",
            options_flow["aggregate_sentiment"],
        )

    st.divider()

    # Cross-fund options flow table
    section_header("Options Flow by Underlying")
    by_underlying = options_flow["by_underlying"]
    if by_underlying:
        opt_df = pd.DataFrame(by_underlying)
        opt_df["call_notional"] = opt_df["call_notional"].apply(_fmt_b)
        opt_df["put_notional"] = opt_df["put_notional"].apply(_fmt_b)
        opt_df["total_notional"] = opt_df["total_notional"].apply(_fmt_b)
        opt_df["pc_ratio"] = opt_df["pc_ratio"].apply(
            lambda x: f"{x:.2f}" if x != float("inf") else "Puts Only"
        )
        opt_df = opt_df.rename(columns={
            "symbol": "Underlying",
            "call_notional": "Call Notional",
            "put_notional": "Put Notional",
            "total_notional": "Total Notional",
            "pc_ratio": "P/C Ratio",
            "sentiment": "Sentiment",
            "fund_count": "Fund Count",
        })
        st.dataframe(
            opt_df[["Underlying", "Call Notional", "Put Notional",
                    "Total Notional", "P/C Ratio", "Sentiment", "Fund Count"]],
            use_container_width=True,
        )

    st.divider()

    # Top calls vs top puts bar chart
    section_header("Call vs Put Notional by Underlying")
    raw = options_flow["by_underlying"][:12]
    if raw:
        symbols = [r["symbol"] for r in raw]
        calls = [r["call_notional"] / 1e9 for r in raw]
        puts = [r["put_notional"] / 1e9 for r in raw]

        fig_cp = go.Figure()
        fig_cp.add_trace(go.Bar(
            name="Call Notional",
            x=symbols, y=calls,
            marker_color=COLORS["success"],
        ))
        fig_cp.add_trace(go.Bar(
            name="Put Notional",
            x=symbols, y=puts,
            marker_color=COLORS["danger"],
        ))
        fig_cp.update_layout(barmode="group")
        fig_cp = _chart_layout(fig_cp, "Call vs Put Notional by Underlying ($B)", height=380)
        fig_cp.update_yaxes(title_text="Notional ($B)")
        st.plotly_chart(fig_cp, use_container_width=True)

    st.divider()

    # All individual option positions across all funds
    filings = engine.fetch_latest_filings()
    all_calls = []
    all_puts = []
    for f in filings:
        for opt in f.option_positions:
            row = {
                "Fund": f.fund_name.split(" ")[0],
                "Underlying": opt.symbol,
                "Strike": f"${opt.strike:.0f}",
                "Expiry": opt.expiry,
                "Contracts": f"{opt.contracts:,}",
                "Notional": _fmt_b(opt.notional_value),
                "Action": opt.action,
                "% Portfolio": f"{opt.pct_portfolio:.1f}%",
            }
            if opt.contract_type == "CALL":
                all_calls.append(row)
            else:
                all_puts.append(row)

    col_calls, col_puts = st.columns(2)
    with col_calls:
        section_header("Top Call Positions (Bullish Bets)")
        if all_calls:
            calls_df = pd.DataFrame(all_calls)
            st.dataframe(calls_df, use_container_width=True)
        else:
            st.info("No call positions found.")

    with col_puts:
        section_header("Top Put Positions (Hedges / Bearish)")
        if all_puts:
            puts_df = pd.DataFrame(all_puts)
            st.dataframe(puts_df, use_container_width=True)
        else:
            st.info("No put positions found.")

    st.divider()

    # P/C ratio by fund gauge-style bar
    section_header("Put/Call Ratio by Fund")
    pc_by_fund = options_sentiment["pc_ratio_by_fund"]
    if pc_by_fund:
        pc_df = pd.DataFrame(
            list(pc_by_fund.items()), columns=["Fund", "P/C Ratio"]
        )
        pc_df["Fund"] = pc_df["Fund"].apply(lambda x: x.split(" ")[0])
        pc_df["Color"] = pc_df["P/C Ratio"].apply(
            lambda x: COLORS["danger"] if x > 1.2
            else COLORS["success"] if x < 0.7
            else COLORS["neutral"]
        )
        fig_pc = go.Figure(go.Bar(
            x=pc_df["Fund"],
            y=pc_df["P/C Ratio"],
            marker_color=pc_df["Color"].tolist(),
            text=[f"{v:.2f}" for v in pc_df["P/C Ratio"]],
            textposition="outside",
        ))
        fig_pc.add_hline(y=1.0, line_dash="dash", line_color=COLORS["gold"],
                         annotation_text="Neutral (1.0)")
        fig_pc = _chart_layout(fig_pc, "Put/Call Ratio by Fund", height=340)
        fig_pc.update_yaxes(title_text="P/C Ratio")
        st.plotly_chart(fig_pc, use_container_width=True)


# ---------------------------------------------------------------------------
# TAB 3 — FIXED INCOME & BONDS
# ---------------------------------------------------------------------------

def _render_tab_bonds(engine):
    section_header("Fixed Income & Bond Positioning")

    bond_data = engine.get_bond_market_positioning()

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Bond AUM", _fmt_b(bond_data["total_market_value"]))
    with col2:
        st.metric(
            "Weighted Avg Duration",
            f"{bond_data['weighted_avg_duration']:.1f} yrs",
        )
    with col3:
        treasury = bond_data["bond_type_breakdown"].get("Treasury", 0)
        st.metric("Treasury Allocation", _fmt_b(treasury))
    with col4:
        corp_ig = bond_data["bond_type_breakdown"].get("Corporate IG", 0)
        corp_hy = bond_data["bond_type_breakdown"].get("Corporate HY", 0)
        st.metric("Corporate (IG + HY)", _fmt_b(corp_ig + corp_hy))

    st.divider()

    col_left, col_right = st.columns(2)

    # Bond type pie chart
    with col_left:
        section_header("Bond Type Breakdown")
        bt = bond_data["bond_type_breakdown"]
        if bt:
            bt_df = pd.DataFrame(list(bt.items()), columns=["Type", "Market Value"])
            fig_pie = px.pie(
                bt_df, values="Market Value", names="Type",
                hole=0.45,
                color_discrete_sequence=[
                    COLORS["gold"], COLORS["lavender"], COLORS["success"],
                    COLORS["danger"], COLORS["neutral"],
                ],
            )
            fig_pie = _chart_layout(fig_pie, height=340)
            fig_pie.update_traces(textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)

    # Duration buckets bar chart
    with col_right:
        section_header("Duration Positioning")
        dur_buckets = bond_data["duration_buckets"]
        if dur_buckets:
            dur_df = pd.DataFrame(
                list(dur_buckets.items()), columns=["Duration Bucket", "AUM"]
            )
            fig_dur = px.bar(
                dur_df, x="Duration Bucket", y="AUM",
                color="Duration Bucket",
                color_discrete_sequence=[
                    COLORS["gold"], COLORS["lavender"], COLORS["success"]
                ],
            )
            fig_dur = _chart_layout(fig_dur, height=340)
            fig_dur.update_yaxes(title_text="Market Value ($)")
            st.plotly_chart(fig_dur, use_container_width=True)

    st.divider()

    # Top issuers table
    section_header("Top Bond Issuers Across All Funds")
    issuers = bond_data["top_issuers"]
    if issuers:
        issuer_df = pd.DataFrame(issuers)
        issuer_df["total_market_value"] = issuer_df["total_market_value"].apply(_fmt_b)
        issuer_df["avg_coupon"] = issuer_df["avg_coupon"].apply(lambda x: f"{x:.2f}%")
        issuer_df = issuer_df.rename(columns={
            "issuer": "Issuer",
            "bond_type": "Type",
            "total_market_value": "Total Market Value",
            "avg_coupon": "Avg Coupon",
        })
        st.dataframe(
            issuer_df[["Issuer", "Type", "Total Market Value", "Avg Coupon"]],
            use_container_width=True,
        )

    st.divider()

    # Bond market thesis
    section_header("Bond Market Thesis")
    avg_dur = bond_data["weighted_avg_duration"]
    treasury_pct = (
        bond_data["bond_type_breakdown"].get("Treasury", 0)
        / bond_data["total_market_value"] * 100
        if bond_data["total_market_value"] > 0 else 0
    )

    if avg_dur < 3:
        duration_thesis = (
            "Institutions are positioned SHORT duration — implying expectations of "
            "continued elevated rates or further Fed tightening. Short-duration Treasuries "
            "and T-bills dominate, reflecting a preference for capital preservation over "
            "yield maximization."
        )
    elif avg_dur < 6:
        duration_thesis = (
            "Institutions hold MEDIUM duration — a balanced posture suggesting uncertainty "
            "about the rate path. This positioning benefits from modest rate declines while "
            "limiting mark-to-market losses if rates stay elevated."
        )
    else:
        duration_thesis = (
            "Institutions are positioned LONG duration — implying expectations of rate cuts "
            "or a flight-to-safety bid. Long-duration Treasuries (TLT, IEF) dominate, "
            "suggesting a macro view that the Fed's next move is a cut."
        )

    if treasury_pct > 60:
        credit_thesis = (
            f"Treasury concentration is high ({treasury_pct:.0f}% of bond AUM), "
            "indicating a risk-off posture and preference for sovereign safety over "
            "credit spread compression."
        )
    else:
        credit_thesis = (
            f"Corporate bonds represent a meaningful allocation ({100-treasury_pct:.0f}% of bond AUM), "
            "suggesting institutions are reaching for yield and comfortable with credit risk "
            "at current spread levels."
        )

    st.info(f"{duration_thesis}\n\n{credit_thesis}")


# ---------------------------------------------------------------------------
# TAB 4 — COMMODITY EXPOSURE
# ---------------------------------------------------------------------------

def _render_tab_commodities(engine):
    section_header("Commodity Exposure Across Institutions")

    comm_data = engine.get_commodity_exposure()
    filings = engine.fetch_latest_filings()

    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Commodity Notional", _fmt_b(comm_data["total_notional"]))
    with col2:
        top_comm = comm_data["by_commodity"][0] if comm_data["by_commodity"] else None
        st.metric(
            "Largest Commodity Holding",
            top_comm["commodity"] if top_comm else "N/A",
            delta=_fmt_b(top_comm["total_notional"]) if top_comm else "",
        )
    with col3:
        funds_with_comm = sum(
            1 for f in filings if f.commodity_positions
        )
        st.metric("Funds with Commodity Exposure", f"{funds_with_comm}/{len(filings)}")

    st.divider()

    col_left, col_right = st.columns(2)

    # Commodity allocation pie
    with col_left:
        section_header("Commodity Allocation")
        alloc = comm_data["commodity_allocation"]
        if alloc:
            alloc_df = pd.DataFrame(
                list(alloc.items()), columns=["Commodity", "Allocation %"]
            )
            fig_pie = px.pie(
                alloc_df, values="Allocation %", names="Commodity",
                hole=0.4,
                color_discrete_sequence=[
                    COLORS["gold"], COLORS["lavender"], COLORS["success"],
                    COLORS["danger"], COLORS["neutral"], "#64b5f6",
                ],
            )
            fig_pie = _chart_layout(fig_pie, height=340)
            fig_pie.update_traces(textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)

    # Commodity notional bar
    with col_right:
        section_header("Notional by Commodity")
        by_comm = comm_data["by_commodity"]
        if by_comm:
            comm_df = pd.DataFrame(by_comm)
            fig_bar = px.bar(
                comm_df, x="commodity", y="total_notional",
                color="commodity",
                color_discrete_sequence=[
                    COLORS["gold"], COLORS["lavender"], COLORS["success"],
                    COLORS["danger"], COLORS["neutral"], "#64b5f6",
                ],
                text=[_fmt_b(v) for v in comm_df["total_notional"]],
            )
            fig_bar = _chart_layout(fig_bar, height=340)
            fig_bar.update_yaxes(title_text="Notional ($)")
            fig_bar.update_traces(textposition="outside")
            st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # Aggregate commodity holdings table
    section_header("Aggregate Commodity Holdings")
    by_comm = comm_data["by_commodity"]
    if by_comm:
        rows = []
        for c in by_comm:
            rows.append({
                "Commodity": c["commodity"],
                "Total Notional": _fmt_b(c["total_notional"]),
                "Fund Count": c["fund_count"],
                "Instruments": ", ".join(c["instruments"]),
                "Tickers": ", ".join(c["tickers"]),
                "Funds": ", ".join([f.split(" ")[0] for f in c["funds"]]),
            })
        comm_table_df = pd.DataFrame(rows)
        st.dataframe(comm_table_df, use_container_width=True)

    st.divider()

    # Cross-fund commodity comparison heatmap
    section_header("Cross-Fund Commodity Comparison")
    all_commodities = list({
        c.commodity for f in filings for c in f.commodity_positions
    })
    fund_names = [f.fund_name.split(" ")[0] for f in filings]

    if all_commodities and fund_names:
        matrix = []
        for f in filings:
            fund_comm = {c.commodity: c.notional_value for c in f.commodity_positions}
            row = [fund_comm.get(comm, 0) for comm in all_commodities]
            matrix.append(row)

        fig_heat = go.Figure(go.Heatmap(
            z=matrix,
            x=all_commodities,
            y=fund_names,
            colorscale=[[0, _PAPER_BG], [0.5, COLORS["gold_dark"]], [1, COLORS["gold"]]],
            text=[[_fmt_b(v) for v in row] for row in matrix],
            texttemplate="%{text}",
            showscale=True,
        ))
        fig_heat = _chart_layout(fig_heat, "Commodity Exposure by Fund ($)", height=380)
        st.plotly_chart(fig_heat, use_container_width=True)


# ---------------------------------------------------------------------------
# TAB 5 — FUND DEEP DIVE
# ---------------------------------------------------------------------------

def _render_tab_fund_deep_dive(engine):
    filings = engine.fetch_latest_filings()
    fund_names = [f.fund_name for f in filings]

    selected_fund = st.selectbox("Select Institution", fund_names, key="deep_dive_fund")
    filing: InstitutionalFiling = next(
        (f for f in filings if f.fund_name == selected_fund), None
    )

    if not filing:
        st.error("Filing not found.")
        return

    # Provenance badge
    st.markdown(_provenance_badge(filing), unsafe_allow_html=True)
    if not filing.data_available:
        st.warning(
            "This fund's 13F data is currently unavailable from SEC EDGAR. "
            "No simulated holdings are displayed in place of real data."
        )

    # Header metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total AUM", _fmt_b(filing.total_aum))
    with col2:
        st.metric("Filing Date", filing.filing_date)
    with col3:
        st.metric("Report Period", filing.report_period)
    with col4:
        st.metric("Put/Call Ratio", f"{filing.put_call_ratio:.2f}")
    with col5:
        st.metric("Bond Duration", f"{filing.bond_duration:.1f} yrs")

    st.divider()

    # Asset class allocation donut + AI insight side by side
    col_donut, col_insight = st.columns([1, 2])

    with col_donut:
        section_header("Asset Class Allocation")
        if filing.asset_class_allocation:
            ac_df = pd.DataFrame(
                list(filing.asset_class_allocation.items()),
                columns=["Asset Class", "Allocation %"],
            )
            fig_donut = px.pie(
                ac_df, values="Allocation %", names="Asset Class",
                hole=0.5,
                color_discrete_sequence=[
                    COLORS["gold"], COLORS["lavender"], COLORS["success"],
                    COLORS["danger"], COLORS["neutral"],
                ],
            )
            fig_donut = _chart_layout(fig_donut, height=320)
            fig_donut.update_traces(textinfo="percent+label")
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.info("No asset-class breakdown available.")

        # P/C ratio interpretation
        pc = filing.put_call_ratio
        if filing.option_positions:
            if pc > 1.5:
                pc_interp = "Heavily hedged — significant downside protection in place."
            elif pc > 1.0:
                pc_interp = "Net hedged — more puts than calls, cautious posture."
            elif pc > 0.7:
                pc_interp = "Balanced — roughly equal call and put exposure."
            elif pc > 0.3:
                pc_interp = "Net long via calls — constructive/bullish options positioning."
            else:
                pc_interp = "Minimal hedging — very low put activity, high conviction long."
            st.caption(f"P/C {pc:.2f}: {pc_interp}")
        else:
            st.caption("No option positions disclosed in this filing.")

    with col_insight:
        section_header("AI Strategy Insight")
        with st.expander("Generate LLM Insight (Macro Context)", expanded=True):
            with st.spinner("Analyzing filing with macro context..."):
                insight = engine.generate_ai_insights(filing)
            st.info(insight)

    st.divider()

    # Equity positions
    section_header("Equity Positions")
    all_positions = filing.top_buys + filing.top_sells
    if all_positions:
        eq_rows = []
        for p in all_positions:
            eq_rows.append({
                "Symbol": p.symbol,
                "Asset Class": p.asset_class,
                "Action": p.action,
                "Shares Changed": f"{p.shares_changed:,}",
                "Value": _fmt_b(p.value_changed),
                "% Portfolio": f"{p.pct_portfolio:.1f}%",
            })
        eq_df = pd.DataFrame(eq_rows)
        st.dataframe(eq_df, use_container_width=True)

    st.divider()

    # Options positions
    section_header("Options Positions")
    if filing.option_positions:
        opt_rows = []
        for o in filing.option_positions:
            opt_rows.append({
                "Underlying": o.symbol,
                "Type": o.contract_type,
                "Strike": f"${o.strike:.0f}",
                "Expiry": o.expiry,
                "Contracts": f"{o.contracts:,}",
                "Notional": _fmt_b(o.notional_value),
                "Action": o.action,
                "% Portfolio": f"{o.pct_portfolio:.1f}%",
            })
        opt_df = pd.DataFrame(opt_rows)
        st.dataframe(opt_df, use_container_width=True)
    else:
        st.info("No options positions reported.")

    st.divider()

    # Bond positions
    section_header("Bond Positions")
    if filing.bond_positions:
        bond_rows = []
        for b in filing.bond_positions:
            bond_rows.append({
                "Issuer": b.issuer,
                "Type": b.bond_type,
                "Maturity": b.maturity,
                "Coupon": f"{b.coupon:.2f}%",
                "Face Value": _fmt_b(b.face_value),
                "Market Value": _fmt_b(b.market_value),
                "Action": b.action,
                "% Portfolio": f"{b.pct_portfolio:.1f}%",
            })
        bond_df = pd.DataFrame(bond_rows)
        st.dataframe(bond_df, use_container_width=True)
    else:
        st.info(
            "Bond positions are not disclosed in 13F filings. This section "
            "appears only when a fund reports them (e.g., via supplemental data)."
        )

    st.divider()

    # Commodity positions
    section_header("Commodity Positions")
    if filing.commodity_positions:
        comm_rows = []
        for c in filing.commodity_positions:
            comm_rows.append({
                "Commodity": c.commodity,
                "Instrument": c.instrument,
                "Ticker": c.ticker,
                "Notional": _fmt_b(c.notional_value),
                "Action": c.action,
                "% Portfolio": f"{c.pct_portfolio:.1f}%",
            })
        comm_df = pd.DataFrame(comm_rows)
        st.dataframe(comm_df, use_container_width=True)
    else:
        st.info(
            "Commodity positions are not disclosed in 13F filings; this "
            "section appears only when a fund reports them."
        )

    st.divider()

    # Sector allocation
    section_header("Sector Allocation")
    if filing.sector_allocation:
        sec_df = pd.DataFrame(
            list(filing.sector_allocation.items()), columns=["Sector", "Allocation %"]
        )
        fig_sec = px.pie(
            sec_df, values="Allocation %", names="Sector",
            hole=0.4,
            color_discrete_sequence=[
                COLORS["gold"], COLORS["lavender"], COLORS["success"],
                COLORS["danger"], COLORS["neutral"], "#64b5f6", "#ffb74d",
            ],
        )
        fig_sec = _chart_layout(fig_sec, height=320)
        fig_sec.update_traces(textinfo="percent+label")
        st.plotly_chart(fig_sec, use_container_width=True)
    else:
        st.info("No sector allocation available.")


# ---------------------------------------------------------------------------
# TAB 6 — CROSS-FUND COMPARISON
# ---------------------------------------------------------------------------

def _render_tab_cross_fund(engine):
    section_header("Cross-Fund Comparison")

    filings = engine.fetch_latest_filings()
    fund_short_names = [f.fund_name.split(" ")[0] for f in filings]

    # --- Asset class allocation grouped bar chart ---
    section_header("Asset Class Allocation Comparison")
    asset_classes = ["Equities", "Options", "Fixed Income", "Commodities", "Other"]
    colors_ac = [
        COLORS["gold"], COLORS["lavender"], COLORS["success"],
        COLORS["danger"], COLORS["neutral"],
    ]

    fig_grouped = go.Figure()
    for i, ac in enumerate(asset_classes):
        values = [
            f.asset_class_allocation.get(ac, 0) for f in filings
        ]
        fig_grouped.add_trace(go.Bar(
            name=ac,
            x=fund_short_names,
            y=values,
            marker_color=colors_ac[i],
            text=[f"{v:.0f}%" for v in values],
            textposition="inside",
        ))
    fig_grouped.update_layout(barmode="stack")
    fig_grouped = _chart_layout(
        fig_grouped, "Asset Class Allocation by Fund (%)", height=420
    )
    fig_grouped.update_yaxes(title_text="Allocation (%)")
    st.plotly_chart(fig_grouped, use_container_width=True)

    st.divider()

    # --- Shared positions heatmap ---
    section_header("Shared Equity Positions Heatmap")
    all_symbols = list({
        p.symbol
        for f in filings
        for p in (f.top_buys + f.top_sells)
        if p.value_changed > 0
    })

    if all_symbols and filings:
        matrix = []
        for f in filings:
            held = {p.symbol for p in f.top_buys if p.value_changed > 0}
            row = [1 if sym in held else 0 for sym in all_symbols]
            matrix.append(row)

        fig_heat = go.Figure(go.Heatmap(
            z=matrix,
            x=all_symbols,
            y=fund_short_names,
            colorscale=[[0, _PAPER_BG], [1, COLORS["gold"]]],
            showscale=False,
            text=[["Held" if v else "" for v in row] for row in matrix],
            texttemplate="%{text}",
        ))
        fig_heat = _chart_layout(
            fig_heat, "Which Funds Hold Which Positions", height=380
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()

    # --- Conviction overlap: positions held by 3+ funds ---
    section_header("High-Conviction Overlap (3+ Funds)")
    symbol_fund_count: Dict[str, int] = {}
    symbol_fund_names: Dict[str, list] = {}
    for f in filings:
        for p in f.top_buys:
            if p.value_changed > 0:
                symbol_fund_count[p.symbol] = symbol_fund_count.get(p.symbol, 0) + 1
                if p.symbol not in symbol_fund_names:
                    symbol_fund_names[p.symbol] = []
                symbol_fund_names[p.symbol].append(f.fund_name.split(" ")[0])

    conviction_symbols = [
        (sym, cnt, symbol_fund_names[sym])
        for sym, cnt in symbol_fund_count.items()
        if cnt >= 3
    ]
    conviction_symbols.sort(key=lambda x: x[1], reverse=True)

    if conviction_symbols:
        conv_df = pd.DataFrame(
            conviction_symbols, columns=["Symbol", "Fund Count", "Funds"]
        )
        conv_df["Funds"] = conv_df["Funds"].apply(lambda x: ", ".join(x))
        st.dataframe(conv_df, use_container_width=True)

        # Bar chart of conviction
        fig_conv = px.bar(
            conv_df, x="Symbol", y="Fund Count",
            color="Fund Count",
            color_continuous_scale=[[0, COLORS["gold_dark"]], [1, COLORS["gold"]]],
            text="Fund Count",
        )
        fig_conv = _chart_layout(fig_conv, "Conviction Overlap — Funds Holding Same Name", height=300)
        fig_conv.update_traces(textposition="outside")
        st.plotly_chart(fig_conv, use_container_width=True)
    else:
        st.info("No positions held by 3 or more funds simultaneously.")

    st.divider()

    # --- Divergence analysis ---
    section_header("Divergence Analysis (Funds on Opposite Sides)")
    buy_funds: Dict[str, list] = {}
    sell_funds: Dict[str, list] = {}

    for f in filings:
        for p in f.top_buys:
            if p.value_changed > 0:
                if p.symbol not in buy_funds:
                    buy_funds[p.symbol] = []
                buy_funds[p.symbol].append(f.fund_name.split(" ")[0])
        for p in f.top_sells:
            if p.value_changed < 0:
                if p.symbol not in sell_funds:
                    sell_funds[p.symbol] = []
                sell_funds[p.symbol].append(f.fund_name.split(" ")[0])

    divergence_rows = []
    all_traded = set(buy_funds.keys()) | set(sell_funds.keys())
    for sym in all_traded:
        buyers = buy_funds.get(sym, [])
        sellers = sell_funds.get(sym, [])
        if buyers and sellers:
            divergence_rows.append({
                "Symbol": sym,
                "Buyers": ", ".join(buyers),
                "Sellers": ", ".join(sellers),
                "Buyer Count": len(buyers),
                "Seller Count": len(sellers),
            })

    if divergence_rows:
        div_df = pd.DataFrame(divergence_rows)
        div_df = div_df.sort_values("Buyer Count", ascending=False)
        st.dataframe(div_df, use_container_width=True)
    else:
        st.info("No divergent positions detected across funds.")

    st.divider()

    # --- Put/Call ratio comparison ---
    section_header("Put/Call Ratio Comparison")
    pc_data = [(f.fund_name.split(" ")[0], f.put_call_ratio, f.options_sentiment)
               for f in filings]
    pc_df = pd.DataFrame(pc_data, columns=["Fund", "P/C Ratio", "Sentiment"])
    pc_df["Color"] = pc_df["P/C Ratio"].apply(
        lambda x: COLORS["danger"] if x > 1.2
        else COLORS["success"] if x < 0.7
        else COLORS["neutral"]
    )
    fig_pc = go.Figure(go.Bar(
        x=pc_df["Fund"],
        y=pc_df["P/C Ratio"],
        marker_color=pc_df["Color"].tolist(),
        text=[f"{v:.2f}" for v in pc_df["P/C Ratio"]],
        textposition="outside",
    ))
    fig_pc.add_hline(y=1.0, line_dash="dash", line_color=COLORS["gold"],
                     annotation_text="Neutral (1.0)")
    fig_pc = _chart_layout(fig_pc, "Put/Call Ratio by Fund", height=320)
    fig_pc.update_yaxes(title_text="P/C Ratio")
    st.plotly_chart(fig_pc, use_container_width=True)


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------

def _provenance_badge(filing) -> str:
    """HTML badge showing the filing's data provenance."""
    if filing.is_simulated:
        color, label = "#ffb74d", "SIMULATED ESTIMATE"
        detail = "Demo data — not actual SEC filing data."
    elif not filing.data_available:
        color, label = "#ef5350", "DATA UNAVAILABLE"
        detail = "SEC EDGAR unreachable — no fabricated figures shown."
    else:
        color, label = "#4caf50", "LIVE SEC EDGAR DATA"
        detail = "Holdings parsed from the fund's actual 13F-HR filing."
    return (
        f'<div style="display:inline-block;border:1px solid {color};color:{color};'
        f'border-radius:4px;padding:2px 10px;font-size:0.72rem;'
        f'text-transform:uppercase;letter-spacing:1px;margin-right:8px;">{label}</div>'
        f'<span style="color:#888;font-size:0.75rem;">{detail}</span>'
    )


def render_13f_analysis_tab():
    """Main entry point — renders the full 13F analysis UI with 6 tabs."""
    st.subheader("Institutional SEC 13F Analysis")
    st.markdown(
        "Tracks institutional holdings reported in SEC 13F filings. Data is "
        "fetched live from SEC EDGAR. Note: 13F filings disclose equity and "
        "option positions only — bond and commodity allocations are not part "
        "of the 13F disclosure."
    )

    engine = get_sec_13f_engine()

    # Data-source status banner
    try:
        sample = engine.fetch_latest_filings(limit=1)
        if sample and sample[0].data_available:
            st.success(
                "Connected to SEC EDGAR — displaying live 13F filing data. "
                "(EDGAR responses are cached for 24h.)"
            )
        else:
            st.warning(
                "SEC EDGAR is currently unreachable. To avoid presenting "
                "fabricated institutional data, positions are shown as "
                "unavailable rather than estimated. Retry when EDGAR is "
                "reachable."
            )
    except Exception:
        pass

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Global Smart Money Flow",
        "Options Intelligence",
        "Fixed Income & Bonds",
        "Commodity Exposure",
        "Fund Deep Dive",
        "Cross-Fund Comparison",
    ])

    with tab1:
        _render_tab_global_flow(engine)

    with tab2:
        _render_tab_options(engine)

    with tab3:
        _render_tab_bonds(engine)

    with tab4:
        _render_tab_commodities(engine)

    with tab5:
        _render_tab_fund_deep_dive(engine)

    with tab6:
        _render_tab_cross_fund(engine)
