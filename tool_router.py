"""
tool_router.py — route explicit tool asks to Octavian's own engines.

When a user asks the chatbot to "run a DCF", "use the Bayesian network",
"fit a Markov model", "check dark pool flow", "look up 13F positioning",
"correlation between X and Y", "COT data", "options greeks", "factor
crowding", etc., this router detects the explicit tool request, invokes
the REAL engine that powers the corresponding site feature, and repackages
the engine's computed output as the chatbot's answer — instead of the LLM
generating a plausible-sounding but unverified response from memory.

Design rules
------------
* The router only fires on EXPLICIT tool vocabulary. Generic asks
  ("how does NVDA look?", "hedge my XOM position", "options strategy for
  LOW") keep their existing specialist builders.
* Every runner is fully defensive: it returns None on any missing data or
  engine failure, so the normal pipeline is never degraded.
* Output is marked with the engine that produced it and its data basis.
"""

import re
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _pick_symbol(tickers: Optional[List[str]]) -> Optional[str]:
    """Pick the primary symbol, preferring plain equity-like tickers."""
    for t in tickers or []:
        s = str(t)
        if s.startswith("^") or s.endswith("=F") or s.endswith("=X") \
                or "-USD" in s:
            continue
        return s
    return (tickers or [None])[0]


def _equity_symbols(tickers: Optional[List[str]]) -> List[str]:
    """Plain-equity-like symbols (excludes indices, futures, FX, crypto)."""
    out = []
    for t in tickers or []:
        s = str(t)
        if s.startswith("^") or s.endswith("=F") or s.endswith("=X") \
                or "-USD" in s:
            continue
        if s not in out:
            out.append(s)
    return out


def _md_table(header: List[str], rows: List[List[str]]) -> str:
    """Render a markdown table without external deps."""
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join(["---"] * len(header)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# individual tool runners — each returns markdown or None
# ---------------------------------------------------------------------------

def _run_dcf(query: str, symbol: str, tickers: List[str],
             live_data: Dict[str, Any]) -> Optional[str]:
    """Run the site's own InstitutionalDCFEngine on the symbol's reported
    fundamentals and repackage its output (WACC, 20-line FCF projection,
    bear/base/bull fair values, terminal value)."""
    if not symbol:
        return None
    try:
        from financial_model_generator import (fetch_ticker_fundamentals,
                                               get_dcf_engine, DCFAssumptions)
        fund = fetch_ticker_fundamentals(symbol)
        if not fund or not fund.get("revenue_m"):
            return None
        price = fund.get("price") or 0.0
        shares_m = float(fund.get("shares_m") or 0.0)
        if shares_m <= 0:
            return None
        engine = get_dcf_engine()
        g = (fund.get("revenue_growth") or 8.0) / 100.0
        m = (fund.get("ebit_margin_pct") or 15.0) / 100.0
        tax = (fund.get("tax_rate_pct") or 21.0) / 100.0
        debt_m = float(fund.get("debt_m") or 0.0)
        cash_m = float(fund.get("cash_m") or 0.0)
        cap_m = float(fund.get("market_cap_m") or 0.0) or (price * shares_m)
        da_rate, capex_rate, nwc_rate = 0.06, 0.08, 0.05
        term_g = 0.028
        years = 5

        assumptions = DCFAssumptions(
            ticker=symbol, base_revenue=float(fund["revenue_m"]),
            revenue_growth_rates=[g] * years, ebit_margin=m, tax_rate=tax,
            da_pct_revenue=da_rate, capex_pct_revenue=capex_rate,
            nwc_change_pct_revenue=nwc_rate,
            equity_value_market=cap_m, debt_value=debt_m, cost_of_debt=0.045,
            risk_free_rate=float(fund.get("risk_free_rate") or 0.042),
            equity_risk_premium=0.05, beta=float(fund.get("beta") or 1.0),
            terminal_growth_rate=term_g, cash=cash_m,
            shares_outstanding=shares_m, current_price=price or 0.0,
        )
        wacc = float(engine.compute_wacc(assumptions)[0]) or 0.095
        if isinstance(wacc, (tuple, list)):
            wacc = float(wacc[0])
        wacc = 0.095 if wacc <= 0 else wacc

        scen = [("Bear", max(-0.05, g - 0.13), max(m - 0.06, 0.05)),
                ("Base", g, m),
                ("Bull", g + 0.08, min(m + 0.04, 0.80))]
        rows, fvs = [], {}
        for name, g0, m0 in scen:
            a2 = DCFAssumptions(**{**assumptions.__dict__,
                                   "revenue_growth_rates": [g0] * years,
                                   "ebit_margin": m0})
            df = engine.project_fcf(a2, wacc)
            pv = float(df["PV of FCF ($M)"].sum())
            last_fcf = float(df["Free Cash Flow ($M)"].iloc[-1])
            tv, pv_tv = engine.compute_terminal_value(last_fcf, wacc, term_g, years)
            ev = pv + pv_tv
            equity = ev - debt_m + cash_m
            fv = equity / shares_m
            fvs[name] = fv
            rows.append([name, f"{g0:.1%}", f"{m0:.1%}",
                         f"${fv:,.2f}",
                         f"{((fv / price) - 1) * 100:+.1f}%" if price else "n/a"])
        base_fv = fvs.get("Base")
        lines = [
            f"### DCF Valuation: {symbol} (site InstitutionalDCFEngine)",
            "",
            f"**Current price:** ${price:,.2f} · **Base fair value:** "
            f"${base_fv:,.2f}" if base_fv else
            f"**Current price:** ${price:,.2f}",
        ]
        if base_fv and price:
            lines.append(f"**Implied upside (base):** "
                         f"{((base_fv / price) - 1) * 100:+.1f}%")
        lines += [
            "",
            f"**WACC:** {wacc:.1%} (CAPM from the tool, REPORTED beta "
            f"{assumptions.beta:.2f}, risk-free {assumptions.risk_free_rate:.1%}, "
            f"ERP 5.0%) · **Terminal growth:** {term_g:.1%} · "
            f"**Reinvestment assumptions:** D&A {da_rate:.0%}, CapEx {capex_rate:.0%}, "
            f"ΔNWC {nwc_rate:.0%} of revenue (MODEL ASSUMPTIONS — cash-flow "
            f"statement is not part of the fundamentals feed).",
            "",
            _md_table(["Scenario", "Rev growth", "EBIT margin",
                       "Fair value", "Upside"], rows),
            "",
            "*Computed by the site's DCF tool from reported income-statement "
            "inputs; FCF mechanics are the tool's own 20-line projection "
            "(NOPAT + D&A - CapEx - ΔNWC, discounted at WACC, Gordon terminal). "
            "Assumption-based valuation, not a verified reported-FCF figure.*",
        ]
        return "\n".join(lines)
    except Exception:
        return None


def _run_reverse_dcf(query: str, symbol: str, tickers: List[str],
                     live_data: Dict[str, Any]) -> Optional[str]:
    """Route explicit reverse-DCF asks to the market-consensus implied-
    expectations solver."""
    if not symbol:
        return None
    try:
        from financial_model_generator import fetch_ticker_fundamentals
        from market_consensus_engine import reverse_dcf_expectations
        fund = fetch_ticker_fundamentals(symbol)
        price = fund.get("price") or (live_data or {}).get(symbol, {}).get("price")
        if not price or price <= 0:
            return None
        shares = (fund.get("shares_m") or 0.0) * 1e6
        net_debt = ((fund.get("debt_m") or 0.0) - (fund.get("cash_m") or 0.0)) * 1e6
        rev_m = fund.get("revenue_m") or 0.0
        if not shares or not rev_m:
            return None
        imp = reverse_dcf_expectations(
            model_price=float(price), current_price=float(price),
            shares=shares, net_debt=net_debt, base_revenue=rev_m * 1e6,
            model_revenue_growth=(fund.get("revenue_growth") or 8.0) / 100.0,
            model_ebit_margin=(fund.get("ebit_margin_pct") or 15.0) / 100.0,
            wacc=0.095, terminal_growth=0.028)
        d = imp.to_dict()
        return "\n".join([
            f"### Reverse DCF: {symbol} (market-consensus implied expectations)",
            "",
            f"**Current price:** ${price:,.2f}",
            f"- Implied revenue growth: **{d.get('implied_revenue_growth', 0):.1%}**",
            f"- Implied EBIT margin: **{d.get('implied_ebit_margin', 0):.1%}**",
            f"- Implied FCF growth: **{d.get('implied_fcf_growth', 0):.1%}**",
            f"- {d.get('gap_label', '')}",
            "",
            d.get("narrative", ""),
            "",
            "*MARKET-IMPLIED VALUE — reverse-engineered from the observed price "
            "via a simplified DCF identity (EV = FCF1/(WACC-g)), not a reported "
            "figure.*",
        ])
    except Exception:
        return None


def _run_bayesian_network(query: str, symbol: Optional[str], tickers: List[str],
                          live_data: Dict[str, Any]) -> Optional[str]:
    """Run the institutional Bayesian network: apply a shock to a node the
    user names (or a default shock) and report the propagated states."""
    try:
        from institutional_analytics_engine import BayesianNetwork
        bn = BayesianNetwork()
        q = (query or "").lower()

        node_aliases = {
            "interest rate": "Interest Rates", "rates": "Interest Rates",
            "inflation": "Inflation", "gdp": "GDP Growth",
            "growth": "GDP Growth", "liquidity": "Liquidity",
            "monetary policy": "Monetary Policy", "fed": "Monetary Policy",
            "volatility": "Volatility", "risk sentiment": "Risk Sentiment",
            "risk appetite": "Risk Sentiment", "credit": "Credit Spreads",
            "spread": "Credit Spreads", "market liquidity": "Market Liquidity",
            "equity": "Equity Indices", "bonds": "Bonds",
            "commodities": "Commodities", "crypto": "Crypto",
        }
        shock_node, shock_val = None, None
        for alias, node in node_aliases.items():
            if alias in q and node in bn.nodes:
                shock_node = node
                break
        m_val = re.search(r"(?:to|at)\s*(\d{1,2})(?:%|\.\d)", q)
        if m_val:
            raw = float(m_val.group(1))
            shock_val = raw / 100.0 if raw > 1 else raw
            shock_val = max(0.0, min(1.0, shock_val))
        if shock_node is None:
            shock_node = "Inflation"
            shock_val = 0.8
        if shock_val is None:
            shock_val = 0.8

        before = {n: nd.state for n, nd in bn.nodes.items()}
        after = bn.propagate(shock_node, shock_val)

        rows = []
        for layer in ("macro", "market", "asset"):
            for n in bn.nodes.values():
                if n.layer != layer:
                    continue
                b = before.get(n.name, 0.0)
                a = after.get(n.name, 0.0)
                delta = a - b
                rows.append([n.name, f"{b:.0%}", f"{a:.0%}",
                             f"{delta:+.0%}"])
        return "\n".join([
            f"### Bayesian Network: {shock_node} → {shock_val:.0%} "
            "(institutional 3-layer network)",
            "",
            "The shock propagates through the DAG's weighted edges "
            "(macro → market → asset layers), dampened toward priors:",
            "",
            _md_table(["Node", "Prior", "Posterior", "Δ"], rows),
            "",
            "*MODEL CALCULATION — weighted-mean propagation over the network's "
            "declared CPT weights; subjective prior structure, not calibrated "
            "statistics.*",
        ])
    except Exception:
        return None


def _run_markov(query: str, symbol: Optional[str], tickers: List[str],
                live_data: Dict[str, Any]) -> Optional[str]:
    """Fit the site's HMM/GMM regime detector to the symbol's price history
    and report the current regime with confidence."""
    try:
        from data_sources import get_stock
        from hmm_engine import get_regime_detector
        sym = symbol or "SPY"
        df = get_stock(sym, period="2y", interval="1d")
        if df is None or df.empty or "Close" not in df.columns:
            return None
        det = get_regime_detector()
        res = det.predict_regime(df)
        regime = res.get("regime", "Unknown")
        conf = res.get("confidence", 0.0)
        desc = res.get("desc", "")
        probs = res.get("probs") or []
        prob_str = ", ".join(f"{p:.0%}" for p in probs) if probs else "n/a"
        return "\n".join([
            f"### Markov / HMM Regime Detection: {sym}",
            "",
            f"**Current regime: {regime}** ({conf:.0%} confidence)",
            f"- {desc}" if desc else "",
            f"- Per-cluster probabilities: {prob_str}",
            "",
            "*MODEL CALCULATION — unsupervised Gaussian-mixture / hidden-Markov "
            "regime detector (returns, 20d volatility, trend strength), fitted "
            "on the symbol's own price history.*",
        ])
    except Exception:
        return None


def _run_dark_pool(query: str, symbol: str, tickers: List[str],
                   live_data: Dict[str, Any]) -> Optional[str]:
    """Run the dark pool intelligence engine on a symbol and repackage its
    report + AI insight."""
    if not symbol:
        return None
    try:
        from dark_pool_engine import get_dark_pool_engine
        eng = get_dark_pool_engine()
        report = eng.analyze_ticker(symbol)
        if not report or not report.get("ok"):
            return None
        price = report.get("price")
        rows = [
            ["Off-exchange share (20d)", f"{report.get('offexchange_pct_20d', 0):.1f}%"],
            ["Off-exchange share (today)", f"{report.get('offexchange_pct_today', 0):.1f}%"],
            ["Imbalance (5d)", f"{report.get('imbalance_5d', 0):+.3f}"],
            ["Pressure score", f"{report.get('pressure_score', 0):+.1f}"],
            ["Block volume today", f"{report.get('block_vol_today', 0):,.0f}"],
            ["Buy vol (5d)", f"{report.get('buy_vol_5d', 0):,.0f}"],
            ["Sell vol (5d)", f"{report.get('sell_vol_5d', 0):,.0f}"],
        ]
        lines = [
            f"### Dark Pool Intelligence: {symbol}",
            "",
            f"**Price:** ${price:,.2f}" if price else f"**Symbol:** {symbol}",
            "",
            _md_table(["Metric", "Value"], rows),
        ]
        try:
            insight = eng.ai_insight(report)
            if insight and insight.get("ok") and insight.get("sections"):
                lines.append("")
                lines.append("**AI Insight:**")
                for k, v in insight["sections"].items():
                    lines.append(f"- **{k}:** {v}")
        except Exception:
            pass
        lines.append("")
        lines.append("*OBSERVED DATA for OHLCV; off-exchange volume is a "
                     "modeled estimate (sector baseline + volume-trend model) "
                     "unless FINRA OTC figures are available.*")
        return "\n".join(lines)
    except Exception:
        return None


def _run_sec_13f(query: str, symbol: Optional[str], tickers: List[str],
                 live_data: Dict[str, Any]) -> Optional[str]:
    """Pull the 13F smart-money flow engine: per-symbol net flow + top
    institutional inflows/outflows."""
    try:
        from sec_13f_engine import SEC13FEngine
        eng = SEC13FEngine()
        flow = eng.get_global_smart_money_flow()
        net_map = flow.get("net_flow_map", {}) or {}
        top_in = (flow.get("top_inflows") or [])[:6]
        top_out = (flow.get("top_outflows") or [])[:6]

        def _fmt_flow(pair):
            return f"{pair[0]}: ${pair[1]:+,.0f}M" if pair else ""

        lines = ["### Institutional 13F Positioning (smart-money flow)",
                 ""]
        if symbol:
            net = net_map.get(symbol)
            if net is not None:
                lines.append(f"**{symbol} net institutional flow: "
                             f"${net:+,.0f}M**")
            else:
                lines.append(f"**{symbol}:** no 13F holding change captured in "
                             "the latest filings feed (DATA UNAVAILABLE — no "
                             "estimate substituted).")
            lines.append("")
        if top_in:
            lines += ["**Top institutional inflows:**",
                      "- " + "\n- ".join(_fmt_flow(p) for p in top_in), ""]
        if top_out:
            lines += ["**Top institutional outflows:**",
                      "- " + "\n- ".join(_fmt_flow(p) for p in top_out), ""]
        lines.append("*REPORTED 13F data from SEC filings (EDGAR) where "
                     "available; the feed never fabricates fund positions.*")
        return "\n".join(lines)
    except Exception:
        return None


def _run_correlation(query: str, symbol: Optional[str], tickers: List[str],
                     live_data: Dict[str, Any]) -> Optional[str]:
    """Compute the correlation matrix for the named symbols."""
    syms = _equity_symbols(tickers)
    if len(syms) < 2:
        return None
    try:
        from risk_engine import correlation_matrix
        corr = correlation_matrix(syms[:6])
        if corr is None or corr.empty:
            return None
        labels = [str(c) for c in corr.columns]
        rows = []
        for i, r in enumerate(labels):
            rows.append([r] + [f"{corr.iloc[i, j]:.2f}"
                               for j in range(len(labels))])
        # strongest pair (off-diagonal)
        best = None
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                v = corr.iloc[i, j]
                if best is None or abs(v) > abs(best[2]):
                    best = (labels[i], labels[j], v)
        lines = ["### Correlation Matrix",
                 "",
                 _md_table([" "] + labels, rows)]
        if best:
            lines += ["",
                      f"**Strongest pair:** {best[0]} ↔ {best[1]} = "
                      f"{best[2]:+.2f}"]
        lines.append("")
        lines.append("*MODEL CALCULATION — Pearson correlation of daily "
                     "returns over the merged price history (1y).*")
        return "\n".join(lines)
    except Exception:
        return None


def _run_options_greeks(query: str, symbol: str, tickers: List[str],
                        live_data: Dict[str, Any]) -> Optional[str]:
    """Compute Black-Scholes option price + Greeks for a symbol."""
    if not symbol:
        return None
    try:
        from financial_model_generator import fetch_ticker_fundamentals
        from options_engine import get_options_engine
        fund = fetch_ticker_fundamentals(symbol)
        S = fund.get("price") or (live_data or {}).get(symbol, {}).get("price")
        if not S or S <= 0:
            return None
        q = (query or "").lower()
        m_k = re.search(r"\$\s?([\d,]+(?:\.\d+)?)", query)
        K = float(m_k.group(1).replace(",", "")) if m_k else float(S)
        m_t = re.search(r"(\d+)\s*(day|week|month|year)s?", q)
        if m_t:
            n, unit = int(m_t.group(1)), m_t.group(2)
            days = n * {"day": 1, "week": 7, "month": 30, "year": 365}[unit]
        else:
            days = 30
        T = max(days / 365.0, 0.02)
        m_sig = re.search(r"(\d{1,2})\s*%", q)
        sigma = (float(m_sig.group(1)) / 100.0) if m_sig else 0.30
        otype = "put" if "put" in q else "call"
        eng = get_options_engine()
        g = eng.black_scholes(S, K, T, sigma, otype)
        keys = [("Price", "price"), ("Delta", "delta"), ("Gamma", "gamma"),
                ("Theta", "theta"), ("Vega", "vega"), ("Rho", "rho"),
                ("Vanna", "vanna"), ("Charm", "charm"), ("Volga", "volga")]
        rows = [[k, f"{g.get(fk, 0):.4f}"] for k, fk in keys]
        return "\n".join([
            f"### Option Pricing & Greeks: {symbol}",
            "",
            f"**Spot ${S:,.2f} · Strike ${K:,.2f} · {days}d to expiry · "
            f"IV {sigma:.0%} · {otype}**",
            "",
            _md_table(["Greek", "Value"], rows),
            "",
            f"*MODEL CALCULATION — {g.get('engine_mode', 'analytic')} "
            "Black-Scholes with the site's options engine. Strike/IV "
            "defaulted to spot / 30% when not stated in the query.*",
        ])
    except Exception:
        return None


def _run_factor_crowding(query: str, symbol: Optional[str], tickers: List[str],
                         live_data: Dict[str, Any]) -> Optional[str]:
    """Run the factor-crowding engine on the named symbol(s)."""
    syms = _equity_symbols(tickers) or ([symbol] if symbol else [])
    if not syms:
        return None
    try:
        from factor_crowding_engine import get_crowding_engine
        eng = get_crowding_engine()
        crowded = eng.detect_crowded_trades(syms[:5])
        rows = []
        for c in crowded:
            rows.append([c.ticker, f"{c.crowding_percentile:.1f}",
                         c.unwind_risk])
        lines = ["### Factor Crowding",
                 "",
                 _md_table(["Ticker", "Crowding pct", "Unwind risk"], rows),
                 ""]
        for s in syms[:2]:
            try:
                ov = eng.simulate_hf_overlap(s)
                lines.append(f"**{s} hedge-fund overlap:** "
                             f"{getattr(ov, 'overlap_pct', 0):.0%} overlap · "
                             f"{getattr(ov, 'summary', '')}")
            except Exception:
                pass
        lines.append("")
        lines.append("*MODEL CALCULATION — crowding percentiles from the "
                     "factor-crowding engine (momentum, vol, correlation, "
                     "flows proxies); unwind risk is a modeled estimate.*")
        return "\n".join(lines)
    except Exception:
        return None


def _run_market_regime(query: str, symbol: Optional[str], tickers: List[str],
                       live_data: Dict[str, Any]) -> Optional[str]:
    """Lightweight risk-on/off + volatility regime context."""
    try:
        from regime import get_regime_context
        ctx = get_regime_context()
        return "\n".join([
            "### Market Regime Context",
            "",
            f"- **Volatility regime:** {ctx.get('volatility_regime', 'n/a')}",
            f"- **Risk mode:** {ctx.get('risk_mode', 'n/a')}",
            f"- **Regime score:** {ctx.get('regime_score', 0):+.1f}",
            "",
            "*MODEL layer — VIX level and index tape; not a forecast.*",
        ])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

# (name, trigger patterns, runner)
_TOOLS: List[Dict[str, Any]] = [
    {"name": "DCF", "patterns": [
        r"(?<!reverse )\bdcf\b", r"(?<!reverse )discounted cash flow",
        r"cash flow model", r"dcf model", r"dcf valuation"],
     "needs_symbol": True, "run": _run_dcf},
    {"name": "Reverse DCF", "patterns": [
        r"reverse dcf", r"reverse discounted cash flow",
        r"what growth does the price imply", r"implied growth"],
     "needs_symbol": True, "run": _run_reverse_dcf},
    {"name": "Bayesian network", "patterns": [
        r"bayesian network", r"bayes net", r"bayesian analysis",
        r"bayesian propagation", r"bayesian model"],
     "needs_symbol": False, "run": _run_bayesian_network},
    {"name": "Markov / HMM regime", "patterns": [
        r"\bmarkov\b", r"hidden markov", r"\bhmm\s+(?:model|regime)\b",
        r"gaussian mixture", r"regime detection", r"regime model"],
     "needs_symbol": False, "run": _run_markov},
    {"name": "Dark pool", "patterns": [
        r"dark pool", r"off-exchange", r"off exchange", r"offexchange"],
     "needs_symbol": True, "run": _run_dark_pool},
    {"name": "Institutional 13F", "patterns": [
        r"\b13f\b", r"13-f", r"sec filings", r"institutional holdings",
        r"smart money"],
     "needs_symbol": False, "run": _run_sec_13f},
    {"name": "Correlation", "patterns": [
        r"correlation", r"correlation matrix"],
     "needs_symbol": False, "run": _run_correlation},
    {"name": "Options greeks", "patterns": [
        r"greeks", r"black-?scholes", r"option pricing",
        r"delta,? gamma", r"option greeks"],
     "needs_symbol": True, "run": _run_options_greeks},
    {"name": "Factor crowding", "patterns": [
        r"crowding", r"crowded trades?", r"hedge fund overlap"],
     "needs_symbol": False, "run": _run_factor_crowding},
    {"name": "Market regime", "patterns": [
        r"risk-on", r"risk off", r"risk on", r"market regime",
        r"volatility regime"],
     "needs_symbol": False, "run": _run_market_regime},
]

# Explicit "probability of reaching $X" style asks are already handled by the
# target-probability engine via the probability intent; COT needs a live CFTC
# download that can hang a chat turn, so it is intentionally not routed here.


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def detect_tool_requests(query: str) -> List[str]:
    """Return the names of every tool the query explicitly asks for."""
    if not query:
        return []
    q = query.lower()
    hits = []
    for spec in _TOOLS:
        for pat in spec["patterns"]:
            if re.search(pat, q):
                hits.append(spec["name"])
                break
    return hits


def route_tool_query(query: str, tickers: Optional[List[str]] = None,
                     sectors: Optional[List[str]] = None,
                     live_data: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """If the query explicitly names one of Octavian's built-in tools, run it
    and repackage the engine's output as the chatbot's answer. Returns None
    when no tool is requested or every matched tool failed (normal pipeline
    then takes over)."""
    matched = detect_tool_requests(query)
    if not matched:
        return None
    live_data = live_data or {}
    symbol = _pick_symbol(tickers)

    sections = []
    for spec in _TOOLS:
        if spec["name"] not in matched:
            continue
        try:
            if spec["needs_symbol"] and not symbol:
                continue
            out = spec["run"](query, symbol, tickers or [], live_data)
        except Exception:
            out = None
        if out:
            sections.append(out)

    if not sections:
        return None
    if len(sections) == 1:
        return sections[0]
    header = ("### Tool-Powered Analysis\n\nYour request maps to "
              + ", ".join(matched)
              + " — each was run with Octavian's own engine:\n")
    return header + "\n\n---\n\n".join(sections)
