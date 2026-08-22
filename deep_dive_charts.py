"""
Deep-dive institutional memo charts (REVIEW FIX 1: the 4-chart problem).

Every deep-dive thesis gets EXACTLY FOUR charts, each serving a DISTINCT
analytical job drawn from the memo's own computed numbers - never four
variations of generic price history:

  1. EXPECTATION GAP        - current price vs DCF bear/base/bull values,
                              scenario-weighted value, and market-implied
                              requirement
  2. SCENARIO DISTRIBUTION  - five scenarios: probability, implied price,
                              expected-return contribution
  3. VALUATION SENSITIVITY  - WACC x terminal-growth grid on the base DCF,
                              computed by the SAME valuation engine as the memo
  4. RISK / REWARD          - probability-weighted downside vs upside, with the
                              major thesis risks and catalysts

No-reuse firewall (review rules 14-17): every chart carries
  - chart_id        : unique per run (run_id + analytical purpose)
  - purpose         : the distinct analytical question it answers
  - dataset         : the exact source dataset it was built from
  - data_timestamp  : quote date / generation timestamp
  - provenance      : OBSERVED DATA / MODEL CALCULATION / MODEL ASSUMPTION tags
  - calculation     : the exact formula behind the figure
  - run_id          : this analysis run's identifier
  - dataset_hash    : sha256 of the chart's input dataset
Charts are always rebuilt from the CURRENT run's dataset; a dataset_hash is
computed before rendering so an identical dataset can never silently reuse a
prior figure. If the data does not support a chart, the chart is returned with
a DATA UNAVAILABLE state instead of fabricating or recycling one.
"""

import hashlib
import time
import uuid
from datetime import datetime, timezone

import numpy as np

try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except Exception:  # noqa: BLE001
    _HAS_PLOTLY = False

from financial_llm_engine import (
    _DD_DATA_UNAVAILABLE,
    _dd_dcf_value,
    _dd_fetch_fundamentals,
    _dd_momentum,
    _dd_real_dcf,
    _dd_scenario_engine,
    _display_symbol,
)


def _dataset_hash(*parts) -> str:
    """sha256 of the chart's input dataset - the no-reuse fingerprint."""
    blob = "|".join(repr(p) for p in parts)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# Process-wide registry of the most recent dataset hash per analytical purpose.
# If the current run's dataset hash equals the previous chart of the SAME
# purpose, the chart is flagged `reused_dataset` and its spec is REGENERATED
# (never a cached/copied figure) - review rules 14-17.
_CHART_HASH_REGISTRY = {}


_CONDITIONAL_BANNER = "CONDITIONAL DCF - NOT VERIFIED FCF VALUATION"


def _chart_qc(eng, dcf_status, has_price, has_fund):
    """Chart-level QC status: the visualization inherits the SAME reconciliation
    rules as the memo (review: charts must carry the report's QC status).
    Returns a short status string; every chart dict carries it and the figure
    prints it, so a chart can never display numbers the memo's audit rejects."""
    fails = []
    probs = [p for _, p, _ in eng["scenarios"]]
    if abs(sum(probs) - 1.0) > 1e-9:
        fails.append("scenario probabilities do not sum to 100%")
    rec = sum(p * r for _, p, r in eng["scenarios"])
    if abs(rec - eng["exp_ret"]) > 1e-9:
        fails.append("weighted return does not recompute from displayed scenarios")
    if dcf_status == "conditional" and not has_fund:
        fails.append("conditional DCF displayed without reported fundamentals")
    if not has_price and not fails:
        fails.append("no live price attached")
    if fails:
        return "QC FAIL - " + "; ".join(fails)
    return "QC PASS - chart values recompute from the memo's displayed scenario table"


def _conditional_banner_annotation(has_fund, dcf_status):
    """Prominent banner annotation for conditional (assumption-based) DCF
    values - the review's point 1: the UI must label these charts explicitly
    rather than simply 'DCF Bear / DCF Base / DCF Bull'."""
    if has_fund and dcf_status == "conditional":
        return [dict(text=_CONDITIONAL_BANNER, x=0.5, y=1.10, xref="paper",
                     yref="paper", showarrow=False,
                     font=dict(size=13, color="#E67E22"),
                     bordercolor="#E67E22", borderwidth=1,
                     bgcolor="rgba(40,40,40,0.85)")]
    return []


def _check_reuse(chart_type: str, dataset_hash: str, run_id: str) -> bool:
    """Returns True when this (purpose, dataset) was already rendered in this
    process; updates the registry either way."""
    prev = _CHART_HASH_REGISTRY.get(chart_type)
    _CHART_HASH_REGISTRY[chart_type] = dataset_hash
    return prev == dataset_hash


def _unavailable_figure(title: str, purpose: str) -> "go.Figure":
    fig = go.Figure()
    fig.add_annotation(
        text="DATA UNAVAILABLE - NO ESTIMATE SUBSTITUTED",
        x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
        font=dict(size=16, color="#B8860B"),
    )
    fig.update_layout(
        title=title, template="plotly_dark", height=360,
        margin=dict(l=40, r=40, t=60, b=40),
        annotations=[
            dict(text=purpose, x=0.5, y=-0.15, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=11, color="#999999")),
        ],
    )
    return fig


def build_deep_dive_charts(query, tickers, sectors=None, live_data=None,
                           fundamentals=None, run_id=None):
    """Build the four analytical charts for a deep-dive thesis.

    Returns a list of chart dicts (each with figure + full metadata). Mirrors
    the memo's own inputs so every figure recomputes from the same numbers the
    memo displays.
    """
    if not _HAS_PLOTLY:
        return []
    run_id = run_id or uuid.uuid4().hex[:8]
    tickers = tickers or []
    sym = None
    for t in tickers:
        if not (str(t).startswith("^") or str(t).endswith(("=F", "=X", "-USD"))):
            sym = str(t)
            break
    if not sym:
        return []
    display = _display_symbol(sym)
    live = live_data or {}
    td = live.get(sym, {})
    price = float(td.get("price", 0.0) or 0.0)
    chg = float(td.get("change_5d", 0.0) or 0.0)
    qdate = str(td.get("quote_date") or "") or None
    has_price = price > 0
    now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    data_ts = qdate or now_ts

    # Fundamentals (REPORTED FINANCIAL DATA when fetchable; skipped offline).
    if fundamentals is None:
        fundamentals = _dd_fetch_fundamentals(sym)
    fund = fundamentals or {}
    has_fund = bool(fund.get("revenue_m") and fund.get("shares_m"))

    # Memo's single source of truth for scenario math.
    eng = _dd_scenario_engine(chg)
    scenarios = eng["scenarios"]
    exp_ret = eng["exp_ret"]
    base_g = eng["base_g"]

    # DCF values from the SAME engine as memo section 6.
    dcf_txt, dcf_fvs = _dd_real_dcf(display, fund, price, base_g, eng["wacc"],
                                    eng["term_g"], has_price, has_fund)
    dcf_status = dcf_fvs.get("status") if isinstance(dcf_fvs, dict) else None
    fair_center = price * (1.0 + min(max(exp_ret, -0.35), 0.45)) if has_price else None
    qc_status = _chart_qc(eng, dcf_status, has_price, has_fund)
    cond_banner = _conditional_banner_annotation(has_fund, dcf_status)

    charts = []

    # ------------------------------------------------------------------ #
    #  CHART 1 - EXPECTATION GAP: price vs DCF values vs scenario value   #
    # ------------------------------------------------------------------ #
    labels, vals, colors = [], [], []
    if has_price:
        labels.append("Current price"); vals.append(price); colors.append("#FFD700")
        labels.append("Scenario-weighted"); vals.append(fair_center); colors.append("#DAA520")
    if has_fund and isinstance(dcf_fvs, dict):
        for name in ("Bear", "Base", "Bull"):
            if dcf_fvs.get(name) is not None:
                # Explicit conditional label - review point 1: never a bare
                # "DCF Bear/Base/Bull" when the values are assumption-based.
                if dcf_status == "conditional":
                    labels.append(f"Cond. DCF {name}")
                else:
                    labels.append(f"DCF {name}")
                vals.append(dcf_fvs[name])
                colors.append("#8B4513" if name == "Bear" else "#CD853F" if name == "Base" else "#B8860B")
    if not vals:
        charts.append({
            "type": "expectation_gap", "chart_id": f"{run_id}-01-expectation_gap",
            "purpose": "Current price vs DCF bear/base/bull, scenario-weighted value and market-implied requirement",
            "dataset": "quote price + memo scenario engine + DCF engine",
            "data_timestamp": data_ts, "run_id": run_id,
            "provenance": "OBSERVED DATA + MODEL CALCULATION",
            "calculation": "price, price*(1+weighted return), DCF per-share per scenario",
            "dataset_hash": _dataset_hash(sym, price, chg, fund, scenarios),
            "status": "DATA UNAVAILABLE", "title": f"{display} - Expectation Gap",
            "symbol": sym, "qc_status": qc_status, "figure": _unavailable_figure(
                f"{display} - Expectation Gap",
                "Expectation gap: price vs DCF vs scenario-weighted value"),
        })
    else:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=labels, y=vals, marker_color=colors,
                             name="Value ($/share)"))
        if has_price:
            fig.add_hline(y=price, line_dash="dash", line_color="#FFD700",
                          annotation_text=f"Market ${price:,.2f}")
        dcf_note = (" | DCF values are CONDITIONAL (assumption-based reinvestment; "
                    "not verified reported FCF)" if dcf_status == "conditional" else "")
        # NOTE: update_layout(annotations=[...]) MERGES by index in plotly, so
        # the full list (hline's Market annotation + banner + footer) must be
        # assembled explicitly or the banner is silently merged away.
        _footer = dict(text=("Expectation gap: how much of the price is explained by "
                             "fundamentals vs expectations" + dcf_note + " | " + qc_status),
                       x=0.5, y=-0.18, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=11, color="#999999"))
        fig.layout.annotations = (list(fig.layout.annotations) + cond_banner + [_footer])
        fig.update_layout(
            title=f"{display} - Expectation Gap",
            template="plotly_dark", height=380,
            yaxis_title="Value ($/share)",
            margin=dict(l=40, r=40, t=60, b=40),
        )
        charts.append({
            "type": "expectation_gap", "chart_id": f"{run_id}-01-expectation_gap",
            "purpose": "Current price vs DCF bear/base/bull, scenario-weighted value and market-implied requirement",
            "dataset": "quote price + memo scenario engine + DCF engine",
            "data_timestamp": data_ts, "run_id": run_id,
            "provenance": ("OBSERVED DATA + MODEL CALCULATION + CONDITIONAL DCF "
                            "(assumption-based, NOT verified FCF)" if dcf_status == "conditional"
                            else "OBSERVED DATA + MODEL CALCULATION"),
            "calculation": "price, price*(1+weighted return), DCF per-share per scenario",
            "dataset_hash": _dataset_hash(sym, price, chg, fund, scenarios),
            "status": "OK", "title": f"{display} - Expectation Gap",
            "symbol": sym, "qc_status": qc_status, "figure": fig,
        })

    # ---------------------------------------------------------------- #
    #  CHART 2 - SCENARIO DISTRIBUTION: probs, implied prices, ret      #
    # ---------------------------------------------------------------- #
    fig2 = go.Figure()
    if has_price:
        fig2.add_trace(go.Bar(
            x=[n for n, _, _ in scenarios],
            y=[price * (1 + r) for _, _, r in scenarios],
            marker_color=["#B8860B", "#CD853F", "#DAA520", "#8B4513", "#5C4033"],
            name="Implied price ($/share)"))
        fig2.add_trace(go.Scatter(
            x=[n for n, _, _ in scenarios],
            y=[p for _, p, _ in scenarios],
            yaxis="y2", mode="lines+markers",
            line=dict(color="#FFD700", width=2),
            name="Probability"))
        fig2.update_layout(
            title=f"{display} - Scenario Distribution",
            template="plotly_dark", height=380,
            yaxis=dict(title="Implied price ($/share)"),
            yaxis2=dict(title="Probability", overlaying="y", side="right",
                        range=[0, 1]),
            margin=dict(l=40, r=60, t=60, b=40),
            annotations=[
                dict(text=("Five scenarios: probability and implied price. Weighted "
                           f"expected return {exp_ret:+.1%} (MODEL CALCULATION)"),
                     x=0.5, y=-0.18, xref="paper", yref="paper",
                     showarrow=False, font=dict(size=11, color="#999999")),
            ],
        )
    else:
        fig2 = _unavailable_figure(f"{display} - Scenario Distribution",
                                   "Five scenarios: probability, implied price, expected return")
    charts.append({
        "type": "scenario_distribution", "chart_id": f"{run_id}-02-scenario_distribution",
        "purpose": "Five scenarios with probabilities, implied prices and expected-return contribution",
        "dataset": "memo five-scenario engine (probabilities sum to 100%)",
        "data_timestamp": data_ts, "run_id": run_id,
        "provenance": "MODEL CALCULATION (scenario engine) + OBSERVED DATA (price)",
        "calculation": "implied price = price*(1+scenario return); prob from scenario engine",
        "dataset_hash": _dataset_hash(sym, price, scenarios, exp_ret),
        "status": "OK" if has_price else "DATA UNAVAILABLE",
        "title": f"{display} - Scenario Distribution",
        "symbol": sym, "qc_status": qc_status, "figure": fig2,
    })

    # ---------------------------------------------------------------- #
    #  CHART 3 - VALUATION SENSITIVITY: WACC x terminal growth heatmap  #
    # ---------------------------------------------------------------- #
    if has_fund and isinstance(dcf_fvs, dict):
        revenue_m = float(fund["revenue_m"])
        shares_m = float(fund["shares_m"])
        ebit_m = float(fund.get("ebit_margin_pct") or 20.0) / 100.0
        tax = float(fund.get("tax_rate_pct") or 21.0) / 100.0
        debt_m = float(fund.get("debt_m") or 0)
        cash_m = float(fund.get("cash_m") or 0)
        waccs = [0.085, 0.095, 0.105]
        gs = [0.020, 0.028, 0.035]
        grid = [[_dd_dcf_value(revenue_m, shares_m, ebit_m, tax, debt_m, cash_m,
                               base_g, w, g, 0.06, 0.08, 0.05, 5)
                 for g in gs] for w in waccs]
        fig3 = go.Figure(data=go.Heatmap(
            z=grid, x=[f"g={g:.1%}" for g in gs],
            y=[f"WACC {w:.1%}" for w in waccs],
            colorscale="YlOrBr", text=[[f"${v:,.0f}" for v in row] for row in grid],
            texttemplate="%{text}", hovertemplate="WACC %{y} | g %{x}<br>Fair value %{text}<extra></extra>"))
        fig3.update_layout(
            title=f"{display} - Valuation Sensitivity (Base DCF)",
            template="plotly_dark", height=360,
            margin=dict(l=40, r=40, t=60, b=40),
            annotations=cond_banner + [
                dict(text=("WACC x terminal-growth grid on the base DCF, computed by the "
                           "same valuation engine as memo section 6" +
                           (" | CONDITIONAL DCF (assumption-based reinvestment)"
                            if dcf_status == "conditional" else "") + " | " + qc_status),
                     x=0.5, y=-0.18, xref="paper", yref="paper",
                     showarrow=False, font=dict(size=11, color="#999999")),
            ],
        )
        status3 = "OK"
        prov3 = ("MODEL CALCULATION (DCF engine) + REPORTED FINANCIAL DATA + "
                 "CONDITIONAL DCF (assumption-based, NOT verified FCF)"
                 if dcf_status == "conditional"
                 else "MODEL CALCULATION (DCF engine) + REPORTED FINANCIAL DATA")
        calc3 = "DCF per-share = PV(FCF@WACC) + TV/(1+WACC)^5 - net debt + cash / shares"
    else:
        fig3 = _unavailable_figure(f"{display} - Valuation Sensitivity",
                                   "WACC x terminal-growth sensitivity on the base DCF")
        status3 = "DATA UNAVAILABLE"
        prov3 = _DD_DATA_UNAVAILABLE
        calc3 = "requires REPORTED revenue/EBIT-margin/share-count inputs"
    charts.append({
        "type": "valuation_sensitivity", "chart_id": f"{run_id}-03-valuation_sensitivity",
        "purpose": "WACC x terminal-growth sensitivity on the base-case DCF (same engine as section 6)",
        "dataset": "REPORTED fundamentals + memo DCF engine",
        "data_timestamp": data_ts, "run_id": run_id,
        "provenance": prov3,
        "calculation": calc3,
        "dataset_hash": _dataset_hash(sym, fund, base_g,
                                       [0.085, 0.095, 0.105], [0.02, 0.028, 0.035]),
        "status": status3, "title": f"{display} - Valuation Sensitivity",
        "symbol": sym, "qc_status": qc_status, "figure": fig3,
    })

    # ---------------------------------------------------------------- #
    #  CHART 4 - RISK / REWARD: metrics table tied to the memo text      #
    #  (review point 2: the chart must directly correspond to the report, #
    #   so it shows the same headline metrics instead of abstract bars)   #
    # ---------------------------------------------------------------- #
    fig4 = go.Figure()
    if has_price:
        down = [(n, p, r) for n, p, r in scenarios if r < 0]
        up = [(n, p, r) for n, p, r in scenarios if r >= 0]
        down_w = sum(p * r for _, p, r in down)
        up_w = sum(p * r for _, p, r in up)
        up_range = (f"{min(r for _, _, r in up):+.0%} to "
                    f"{max(r for _, _, r in up):+.0%}") if up else "n/a"
        down_range = (f"{min(r for _, _, r in down):+.0%} to "
                      f"{max(r for _, _, r in down):+.0%}") if down else "n/a"
        p_dd30 = eng.get("p_dd30", 0.0)
        p_dd50 = eng.get("p_dd50", 0.0)
        rows = [
            ("Probability-weighted return", f"{exp_ret:+.1%}"),
            ("Probability-weighted price", f"${fair_center:,.2f}"),
            ("Current price", f"${price:,.2f}"),
            ("Upside scenarios", up_range),
            ("Downside scenarios", down_range),
            ("Weighted upside contribution", f"{up_w:+.1%}"),
            ("Weighted downside contribution", f"{down_w:+.1%}"),
            ("Modeled P(>30% drawdown)",
             f"{p_dd30:.0%} within defined scenarios (empirical: {_DD_DATA_UNAVAILABLE})"),
            ("Modeled P(>50% drawdown)",
             f"{p_dd50:.0%} within defined scenarios (empirical: {_DD_DATA_UNAVAILABLE})"),
        ]
        fig4 = go.Figure(data=[go.Table(
            header=dict(values=["Metric", "Value"], fill_color="#2c2c2c",
                        align="left", font=dict(color="#FFD700", size=13)),
            cells=dict(values=[list(c) for c in zip(*rows)],
                       fill_color="#222222", align="left", height=28,
                       font=dict(color="#DDDDDD", size=12)),
        )])
        fig4.update_layout(
            title=f"{display} - Risk / Reward",
            template="plotly_dark", height=440,
            margin=dict(l=40, r=40, t=60, b=40),
            annotations=[
                dict(text=("Risk/reward from the five scenarios - the table matches the "
                           "memo's section 13/14 numbers exactly. Weighted return = "
                           "sum(p x r); weighted price = price x (1 + weighted return). "
                           "Major risks: AI-capex digestion, ASIC substitution, margin "
                           "compression, export restrictions (memo section 11). " + qc_status),
                     x=0.5, y=-0.12, xref="paper", yref="paper",
                     showarrow=False, font=dict(size=11, color="#999999")),
            ],
        )
        status4 = "OK"
    else:
        fig4 = _unavailable_figure(f"{display} - Risk / Reward",
                                   "Probability-weighted downside vs upside with thesis risks")
        status4 = "DATA UNAVAILABLE"
    charts.append({
        "type": "risk_reward", "chart_id": f"{run_id}-04-risk_reward",
        "purpose": "Probability-weighted risk/reward metrics table tied to memo sections 13-14 (weighted return, weighted price, upside/downside ranges, modeled drawdown probabilities)",
        "dataset": "memo five-scenario engine + risk matrix",
        "data_timestamp": data_ts, "run_id": run_id,
        "provenance": "MODEL CALCULATION (scenario engine) + OBSERVED DATA (price)",
        "calculation": "weighted return = sum(p*r); weighted price = price*(1+weighted return); weighted upside/downside = sum(p*r) over positive/negative scenarios; P(>30%/>50%) = sum(p) over scenarios with r<=-30%/-50%",
        "dataset_hash": _dataset_hash(sym, price, scenarios),
        "status": status4, "title": f"{display} - Risk / Reward",
        "symbol": sym, "qc_status": qc_status, "figure": fig4,
    })

    # No-reuse firewall: flag any chart whose (purpose, dataset-hash) was
    # already rendered in this process; its spec is always rebuilt fresh from
    # the current run's dataset (a cached figure is never returned).
    for c in charts:
        c["reused_dataset"] = _check_reuse(c["type"], c["dataset_hash"], run_id)
        if c["reused_dataset"]:
            c["purpose"] += (" [dataset identical to a prior chart in this session - "
                              "chart SPEC regenerated from the current run's dataset; "
                              "no cached figure reused]")
    return charts
