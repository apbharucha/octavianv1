"""Portfolio Chatbot Context — Octavian Terminal
=============================================
Provides portfolio awareness to the AI chatbot.

Features:
- Load portfolio context from both paper trading + manual portfolios
- Detect portfolio-specific intents in user queries
- Answer 6 types of portfolio queries:
  1. Grade / health check
  2. Buy/sell fit (does ticker fit current portfolio?)
  3. Biggest risks right now
  4. Rebalance suggestions
  5. Drag / worst performers
  6. Benchmark comparison vs SPY
- Generate AI-written portfolio summaries via LM Studio
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("PortfolioChatbotContext")

# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO INTENT PATTERNS
# ─────────────────────────────────────────────────────────────────────────────

PORTFOLIO_PATTERNS = [
    # Grade / health check
    (r"how is my portfolio\b", "portfolio_grade"),
    (r"portfolio.*(doing|performing|health|grade|score|rating)", "portfolio_grade"),
    (r"how.*(am I doing|are my holdings|is my account)", "portfolio_grade"),
    (r"grade my portfolio", "portfolio_grade"),

    # Buy/sell fit check
    (r"should I (buy|add|purchase).*(to my|from my|in my|for my)", "portfolio_fit"),
    (r"should I (sell|remove|exit|trim).*(from my|in my|out of my)", "portfolio_fit"),
    (r"should I (buy|add|purchase|sell|exit)\s+[A-Z]{1,6}", "portfolio_fit"),
    (r"(does|would).*(fit|suit) my portfolio", "portfolio_fit"),
    (r"(buy|sell|add).*(good|bad|right|appropriate) for my portfolio", "portfolio_fit"),
    (r"(is|would)\s+[A-Z]{1,6}.*(good|bad|fit|suit|work) (for|in|with) my portfolio", "portfolio_fit"),

    # Risk analysis
    (r"(biggest|main|top|current) risk", "portfolio_risk"),
    (r"what.*(risk|exposure|vulnerable|drawdown)", "portfolio_risk"),
    (r"(stress test|worst case|if.*(crash|drop|fall))", "portfolio_risk"),

    # Rebalance
    (r"(rebalance|reallocate|restructure|optimize) my portfolio", "portfolio_rebalance"),
    (r"how (should|can) I improve my portfolio", "portfolio_rebalance"),
    (r"(position sizing|adjust|change) my (allocation|weights)", "portfolio_rebalance"),

    # Drag / worst performers
    (r"(dragging|pulling|holding) (me|my portfolio) (down|back)", "portfolio_drag"),
    (r"(worst|weakest|poorest).*(position|holding|stock|asset)", "portfolio_drag"),
    (r"what.*(hurting|damaging|costing) my portfolio", "portfolio_drag"),

    # Benchmark comparison
    (r"compare.*(my portfolio|my holdings).*(to|vs|with|against)", "portfolio_benchmark"),
    (r"(how|am I|is my).*(beat|outperform|underperform|vs) (spy|s&p|market|benchmark)", "portfolio_benchmark"),
    (r"(my) (alpha|beta|relative performance)", "portfolio_benchmark"),
]

import re


def detect_portfolio_intent(query: str) -> Optional[str]:
    """Detect if query is asking about a portfolio. Returns intent name or None."""
    q = query.lower()
    for pattern, intent in PORTFOLIO_PATTERNS:
        if re.search(pattern, q):
            return intent
    return None


def extract_ticker_from_fit_query(query: str) -> Optional[str]:
    """Extract ticker from a 'should I buy/sell X'type query."""
    # Common patterns: "should I buy AAPL"or "should I add BTC-USD to my portfolio"
    patterns = [
        r"(?:buy|sell|add|purchase|remove|exit)\s+([A-Z]{1,6}(?:-USD)?)",
        r"([A-Z]{1,6}(?:-USD)?)\s+(?:fit|suit|good|bad|work)",
]
    for pat in patterns:
        match = re.search(pat, query.upper())
        if match:
            return match.group(1)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_all_portfolios(user_id: str = None) -> Dict[str, Any]:
    """
    Load both paper trading accounts and manual portfolios.
    Returns a merged context dict.
    """
    context = {
        "paper_accounts": [],
        "manual_portfolios": [],
        "has_data": False,
}

    # 1. Paper Trading Accounts
    try:
        from paper_trading_system import get_paper_trading_system
        pts = get_paper_trading_system()
        accounts = pts.get_all_accounts() if hasattr(pts, "get_all_accounts") else []
        if not accounts:
            # Try session state fallback
            import streamlit as st
            accounts_data = st.session_state.get("paper_trading_accounts", [])
            for acct in accounts_data:
                acct_id = acct.get("id") or acct.get("account_id", "default")
                account = pts.get_account(acct_id)
                if account:
                    positions = account.get("positions", account.get("holdings", []))
                    context["paper_accounts"].append({
                        "id": acct_id,
                        "name": acct.get("name", acct_id),
                        "positions": positions,
                        "cash": account.get("cash_balance", account.get("cash", 0)),
                        "total_value": account.get("total_value", account.get("portfolio_value", 0)),
                        "total_pnl": account.get("total_pnl", 0),
                    })
        else:
            for acct in accounts:
                context["paper_accounts"].append({
                    "id": acct.get("id", ""),
                    "name": acct.get("name", acct.get("id", "Account")),
                    "positions": acct.get("positions", []),
                    "cash": acct.get("cash_balance", 0),
                    "total_value": acct.get("total_value", 0),
                    "total_pnl": acct.get("total_pnl", 0),
                })
    except Exception as e:
        logger.debug(f"Paper trading load error: {e}")

    # 2. Manual Portfolios
    try:
        from manual_portfolio_system import ManualPortfolioSystem
        mps = ManualPortfolioSystem()
        portfolios = mps.list_portfolios() if hasattr(mps, "list_portfolios") else []
        for pf in portfolios:
            pf_id = pf.get("id", pf.get("name", "manual"))
            pf_data = mps.get_portfolio(pf_id) if hasattr(mps, "get_portfolio") else pf
            if pf_data:
                context["manual_portfolios"].append({
                    "id": pf_id,
                    "name": pf_data.get("name", pf_id),
                    "goal": pf_data.get("goal", ""),
                    "risk_level": pf_data.get("risk_level", "Moderate"),
                    "positions": pf_data.get("positions", []),
                    "history": pf_data.get("history", []),
                })
    except Exception as e:
        logger.debug(f"Manual portfolio load error: {e}")

    context["has_data"] = bool(context["paper_accounts"] or context["manual_portfolios"])
    return context


def _get_all_holdings(context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten all positions from all accounts/portfolios into a list."""
    holdings = []
    for acct in context.get("paper_accounts", []):
        for pos in acct.get("positions", []):
            pos["_source"] = f"Paper: {acct['name']}"
            holdings.append(pos)
    for pf in context.get("manual_portfolios", []):
        for pos in pf.get("positions", []):
            pos["_source"] = f"Manual: {pf['name']}"
            holdings.append(pos)
    return holdings


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO CONTEXT BUILDER (for LLM System Prompt)
# ─────────────────────────────────────────────────────────────────────────────

def build_portfolio_context_string(context: Dict[str, Any], max_length: int = 3000) -> str:
    """Build a concise portfolio summary for injection into the LLM system prompt."""
    if not context["has_data"]:
        return ""
    lines = ["[PORTFOLIO CONTEXT]"]

    for acct in context.get("paper_accounts", []):
        lines.append(f"\n--- Paper Trading: {acct['name']} ---")
        lines.append(f"Total Value: ${acct.get('total_value', 0):,.2f} | Cash: ${acct.get('cash', 0):,.2f} | P&L: ${acct.get('total_pnl', 0):,.2f}")
        positions = acct.get("positions", [])
        if positions:
            lines.append("Positions:")
            for pos in positions[:15]:
                sym = pos.get("symbol", pos.get("ticker", "?"))
                qty = pos.get("quantity", pos.get("shares", 0))
                entry = pos.get("entry_price", pos.get("avg_cost", 0))
                curr = pos.get("current_price", 0)
                pnl_pct = ((curr / entry - 1) * 100) if entry and curr else 0
                lines.append(f"- {sym}: {qty} units @ ${entry:.2f} entry | Current: ${curr:.2f} | P&L: {pnl_pct:+.1f}%")

    for pf in context.get("manual_portfolios", []):
        lines.append(f"\n--- Manual Portfolio: {pf['name']} ---")
        lines.append(f"Goal: {pf.get('goal', 'N/A')} | Risk: {pf.get('risk_level', 'N/A')}")
        positions = pf.get("positions", [])
        if positions:
            lines.append("Positions:")
            for pos in positions[:15]:
                sym = pos.get("symbol", "?")
                qty = pos.get("quantity", 0)
                entry = pos.get("entry_price", 0)
                lines.append(f"- {sym}: {qty} units @ ${entry:.2f}")

    result = "\n".join(lines)
    return result[:max_length] # Respect token budget


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

async def handle_portfolio_grade(context: Dict[str, Any]) -> str:
    """Pull live 25-factor grade summary for all portfolios."""
    if not context["has_data"]:
        return "No portfolio data found. Create a paper trading account or add positions to a manual portfolio first."
    response = "## Portfolio Health Report\n\n"
    # Try to get 25-factor analysis from the engine
    try:
        from portfolio_analyzer_engine import PortfolioAnalyzerEngine
        engine = PortfolioAnalyzerEngine()

        for acct in context.get("paper_accounts", []):
            acct_id = acct["id"]
            response += f"### {acct['name']}\n"
            response += f"- **Total Value:** ${acct.get('total_value', 0):,.2f}\n"
            response += f"- **Cash Balance:** ${acct.get('cash', 0):,.2f}\n"
            response += f"- **Total P&L:** ${acct.get('total_pnl', 0):+,.2f}\n"
            try:
                import asyncio
                analytics = await engine.analyze_portfolio(acct_id)
                if analytics:
                    grade_color = {"A": "", "B": "", "C": "", "D": "", "F": ""}.get(
                        analytics.health_grade[0] if analytics.health_grade else "C", "")
                    response += f"- **Health Grade:** {grade_color} **{analytics.health_grade}** ({analytics.health_score:.1f}/100)\n"
                    response += f"- **Sharpe Ratio:** {analytics.sharpe_ratio:.2f}\n"
                    response += f"- **Annual Volatility:** {analytics.volatility_annual*100:.1f}%\n"
                    response += f"- **Max Drawdown:** {analytics.max_drawdown*100:.1f}%\n"
                    response += f"- **Beta to SPY:** {analytics.beta_to_spy:.2f}\n"
                    response += f"- **30D Expected Return:** {analytics.expected_return_30d*100:+.2f}%\n"
                    response += f"- **Regime Alignment:** {analytics.regime_alignment}\n"
                    if analytics.improvement_ideas:
                        response += "\n** Top Improvement Ideas:**\n"
                        for idea in analytics.improvement_ideas[:3]:
                            response += f"- {idea.get('title', '')}: {idea.get('description', '')}\n"
            except Exception as e:
                logger.warning(f"Portfolio analysis error for {acct_id}: {e}")
                response += "_Detailed 25-factor analysis unavailable — check that positions have market data._\n"
            response += "\n"
        for pf in context.get("manual_portfolios", []):
            response += f"### {pf['name']} (Manual)\n"
            response += f"- **Goal:** {pf.get('goal', 'Not set')}\n"
            response += f"- **Risk Level:** {pf.get('risk_level', 'N/A')}\n"
            positions = pf.get("positions", [])
            if positions:
                response += f"- **Positions:** {len(positions)} holdings\n"
                symbols = [p.get("symbol", "?") for p in positions[:8]]
                response += f"- **Holdings:** {', '.join(f'`{s}`'for s in symbols)}\n"
            response += "\n"
    except ImportError:
        response += "_Portfolio analyzer engine not available._\n"
    return response


async def handle_portfolio_fit(query: str, context: Dict[str, Any]) -> str:
    """Check if a specific ticker fits the current portfolio composition."""
    ticker = extract_ticker_from_fit_query(query)
    if not ticker:
        return "I detected a portfolio fit question but couldn't identify the specific ticker. Try: *'Should I add AAPL to my portfolio?'*"
    holdings = _get_all_holdings(context)
    existing_symbols = [h.get("symbol", h.get("ticker", "")).upper() for h in holdings]

    response = f"## Portfolio Fit Check: `{ticker}`\n\n"
    # Check if already held
    if ticker in existing_symbols:
        acct = next((h.get("_source", "") for h in holdings if h.get("symbol", "").upper() == ticker), "")
        response += f"**Already Held:** `{ticker}`is currently in your portfolio ({acct}).\n\n"
        response += f"**Action Options:**\n"
        response += f"- Consider **adding** more if conviction is high and it's undersized\n"
        response += f"- Consider **trimming** if it's your largest position and score has deteriorated\n\n"
    else:
        response += f"**Not Currently Held:** `{ticker}`is **not** in your portfolio.\n\n"
    # Get current analysis of the ticker
    try:
        from comparative_analysis_engine import get_comparative_engine, detect_asset_type
        engine = get_comparative_engine()
        atype = detect_asset_type(ticker)
        result = engine._analyze_single_asset(
            ticker, atype, {"use_quant": True, "use_technical": True, "use_fundamentals": False, "use_macro": False}
)

        if not result.error:
            sig_emoji = {"STRONG BUY": "", "BUY": "", "NEUTRAL": "", "SELL": "", "STRONG SELL": ""}.get(result.decision, "")
            response += f"### Current Signal for `{ticker}`\n"
            response += f"- **Signal:** {sig_emoji} {result.decision} ({result.conviction*100:.0f}% conviction)\n"
            response += f"- **Score:** {result.overall_score:.1f}/100\n"
            response += f"- **30D Expected Return:** {result.expected_return_30d:+.2f}%\n"
            response += f"- **Annual Volatility:** {result.volatility_annual*100:.1f}%\n\n"
            # Portfolio fit assessment
            current_vol = np.mean([
                0.20 # Default fallback if we can't calculate portfolio vol
            ])

            response += "### Fit Assessment\n"
            if result.decision in ("STRONG BUY", "BUY"):
                response += f"**Signal supports addition** — `{ticker}`is showing positive momentum.\n"
            elif result.decision in ("SELL", "STRONG SELL"):
                response += f"**Signal suggests caution** — `{ticker}`is showing weakness. Consider waiting for a better entry.\n"
            else:
                response += f"**Neutral signal** — No strong catalyst detected. Add only if it fills a specific gap in your portfolio.\n"
            if result.volatility_annual > 0.50:
                response += f"\n **High Volatility Warning:** {ticker} has {result.volatility_annual*100:.1f}% annual volatility — size accordingly (smaller position).\n"
            # Check concentration
            if len(existing_symbols) > 0:
                response += f"\n**Concentration Check:** Your portfolio has {len(existing_symbols)} holdings."
                if len(existing_symbols) < 5:
                    response += "Adding another position would increase diversification."
                elif ticker.endswith("-USD"):
                    crypto_count = sum(1 for s in existing_symbols if s.endswith("-USD") or s in ["BTC", "ETH"])
                    response += f"You currently have {crypto_count} crypto positions."
        else:
            response += f"Could not retrieve data for `{ticker}`: {result.error}\n"
    except Exception as e:
        logger.error(f"Portfolio fit analysis error: {e}")
        response += f"_Detailed analysis for `{ticker}`unavailable._\n"
    # LLM enhancement
    try:
        from financial_llm_engine import _call_llm, check_llm_connectivity
        if check_llm_connectivity():
            holdings_str = ", ".join(existing_symbols[:10]) or "None"
            prompt = (
                f"Portfolio has: {holdings_str}."
                f"User asks if they should {'add more of'if ticker in existing_symbols else 'buy'} {ticker}."
                f"Current signal: {result.decision if not result.error else 'unknown'}."
                f"In 2-3 sentences, give a direct recommendation focused on portfolio construction fit.")
            system = "You are an elite portfolio strategist. Be direct, specific, and actionable. No disclaimers."
            ai_resp = _call_llm(prompt=prompt, system=system)
            if ai_resp:
                response += f"\n### AI Recommendation\n{ai_resp}\n"
    except Exception:
        pass

    return response


async def handle_portfolio_risk(context: Dict[str, Any]) -> str:
    """Run stress tests and identify biggest risks."""
    if not context["has_data"]:
        return "No portfolio data found."
    response = "## Portfolio Risk Assessment\n\n"
    holdings = _get_all_holdings(context)
    symbols = [h.get("symbol", h.get("ticker", "")) for h in holdings if h.get("symbol") or h.get("ticker")]

    if not symbols:
        return response + "_No positions found in your portfolios._\n"
    response += f"**Holdings Analyzed:** {', '.join(f'`{s}`'for s in symbols[:10])}\n\n"
    # Stress test scenarios
    stress_scenarios = {
        "2008 Financial Crisis": {"equity_shock": -0.55, "crypto_shock": -0.70, "gold_shock": 0.25, "bond_shock": 0.15},
        "2022 Rate Hike Cycle": {"equity_shock": -0.20, "crypto_shock": -0.65, "gold_shock": -0.02, "bond_shock": -0.18},
        "2000 Dot-com Bust": {"equity_shock": -0.49, "crypto_shock": -0.60, "gold_shock": 0.10, "bond_shock": 0.10},
        "AI Bubble Burst (Hypothetical)": {"equity_shock": -0.35, "crypto_shock": -0.55, "gold_shock": 0.15, "bond_shock": 0.05},
}

    response += "### Stress Test Scenarios\n"
    response += "_Estimated portfolio impact under historical and hypothetical crash scenarios:_\n\n"
    for scenario_name, shocks in stress_scenarios.items():
        total_positions = len(holdings)
        if total_positions == 0:
            continue

        # Estimate impact based on asset types
        weighted_shock = 0.0
        for pos in holdings:
            sym = (pos.get("symbol") or pos.get("ticker", "")).upper()
            value = pos.get("market_value", pos.get("quantity", 1) * pos.get("current_price", pos.get("entry_price", 100)))

            if sym.endswith("-USD") or sym in {"BTC", "ETH", "SOL", "DOGE"}:
                shock = shocks["crypto_shock"]
            elif sym in {"GC=F", "GOLD", "SI=F", "GLD", "IAU"}:
                shock = shocks["gold_shock"]
            elif sym in {"TLT", "IEF", "BND", "AGG", "LQD"}:
                shock = shocks["bond_shock"]
            else:
                shock = shocks["equity_shock"]

            weighted_shock += shock / total_positions

        response += f"**{scenario_name}:** Estimated portfolio impact: **{weighted_shock*100:+.1f}%**\n"
    response += "\n### Key Risk Factors\n"
    # Concentration risk
    if len(symbols) < 5:
        response += f"**High Concentration:** Only {len(symbols)} holdings — single-name risk is elevated.\n"
    elif len(symbols) > 20:
        response += f"**Well Diversified:** {len(symbols)} holdings reduces concentration risk.\n"
    # Asset class concentration
    crypto_count = sum(1 for s in symbols if s.endswith("-USD") or s in {"BTC", "ETH", "SOL", "DOGE", "AVAX"})
    if crypto_count > len(symbols) * 0.4:
        response += f"**Crypto-Heavy:** {crypto_count}/{len(symbols)} positions are crypto — extreme volatility exposure.\n"
    # Try LLM risk narrative
    try:
        from financial_llm_engine import _call_llm, check_llm_connectivity
        if check_llm_connectivity():
            prompt = (
                f"Portfolio holdings: {', '.join(symbols[:12])}."
                f"Total {len(symbols)} positions."
                f"In 3-4 sentences, identify the 2-3 biggest structural risks in this portfolio and how to hedge them.")
            system = "You are an elite risk manager at a hedge fund. Be specific, direct, and quantitative. No disclaimers."
            ai_resp = _call_llm(prompt=prompt, system=system)
            if ai_resp:
                response += f"\n### AI Risk Narrative\n{ai_resp}\n"
    except Exception:
        pass

    return response


async def handle_portfolio_rebalance(context: Dict[str, Any]) -> str:
    """Generate rebalance/position sizing recommendations."""
    if not context["has_data"]:
        return "No portfolio data found."
    response = "## Rebalance & Position Sizing Recommendations\n\n"
    holdings = _get_all_holdings(context)
    symbols = [h.get("symbol", h.get("ticker", "")) for h in holdings if h.get("symbol") or h.get("ticker")]

    if not symbols:
        return response + "_No positions found._\n"
    response += f"**Current Holdings ({len(symbols)}):** {', '.join(f'`{s}`'for s in symbols[:10])}\n\n"
    # Score all holdings with comparative engine
    try:
        from comparative_analysis_engine import get_comparative_engine, detect_asset_type
        engine = get_comparative_engine()

        scores = {}
        for sym in symbols[:10]:
            atype = detect_asset_type(sym)
            result = engine._analyze_single_asset(
                sym, atype,
                {"use_quant": True, "use_technical": True, "use_fundamentals": False, "use_macro": False}
)
            if not result.error:
                scores[sym] = result

        if scores:
            response += "### Current Holding Scores\n"
            sorted_scores = sorted(scores.items(), key=lambda x: x[1].overall_score, reverse=True)

            response += "| Symbol | Score | Signal | 30D Return | Recommendation |\n"
            response += "|--------|-------|--------|------------|----------------|\n"
            for sym, s in sorted_scores:
                rec = "**Increase**"if s.overall_score > 65 else "**Reduce**"if s.overall_score < 35 else "Hold"
                response += f"| `{sym}`| {s.overall_score:.1f}/100 | {s.decision} | {s.change_1m:+.2f}% | {rec} |\n"
            # Suggest equal-weight target vs score-weighted
            total_holdings = len(scores)
            equal_weight = round(100 / total_holdings, 1)
            response += f"\n**Equal Weight Target:** {equal_weight}% per position\n"
            # Score-weighted allocation
            total_score = sum(s.overall_score for s in scores.values())
            response += "\n**Score-Weighted Target Allocation:**\n"
            for sym, s in sorted_scores[:6]:
                target_pct = round((s.overall_score / total_score) * 100, 1)
                response += f"- `{sym}`: {target_pct}%\n"
    except Exception as e:
        logger.warning(f"Rebalance scoring error: {e}")
        response += "_Could not score holdings for rebalance recommendations._\n"
    # LLM rebalance advice
    try:
        from financial_llm_engine import _call_llm, check_llm_connectivity
        if check_llm_connectivity():
            portfolio_context = build_portfolio_context_string(context, max_length=1500)
            prompt = (
                f"{portfolio_context}\n\n"
                f"Give 3-5 specific, actionable rebalancing recommendations."
                f"Include specific position size changes, what to add, what to reduce, and why.")
            system = "You are an elite portfolio manager. Give specific percentage targets. Be direct and actionable. No disclaimers."
            ai_resp = _call_llm(prompt=prompt, system=system)
            if ai_resp:
                response += f"\n### AI Rebalancing Plan\n{ai_resp}\n"
    except Exception:
        pass

    return response


async def handle_portfolio_drag(context: Dict[str, Any]) -> str:
    """Identify worst-performing positions."""
    if not context["has_data"]:
        return "No portfolio data found."
    response = "## Portfolio Drag Analysis — Weakest Positions\n\n"
    holdings = _get_all_holdings(context)

    if not holdings:
        return response + "_No positions found._\n"
    # Score all holdings
    scored = []
    try:
        from comparative_analysis_engine import get_comparative_engine, detect_asset_type
        engine = get_comparative_engine()

        for pos in holdings[:15]:
            sym = (pos.get("symbol") or pos.get("ticker", "")).upper()
            if not sym:
                continue
            atype = detect_asset_type(sym)
            result = engine._analyze_single_asset(
                sym, atype,
                {"use_quant": True, "use_technical": True, "use_fundamentals": False, "use_macro": False}
)
            if not result.error:
                scored.append((sym, result, pos.get("_source", "")))

        scored.sort(key=lambda x: x[1].overall_score)

        if scored:
            response += "### Bottom Performers\n"
            for sym, s, source in scored[:3]:
                response += f"\n**`{sym}`** ({source})\n"
                response += f"- Score: {s.overall_score:.1f}/100 | Signal: {s.decision}\n"
                response += f"- 30D Performance: {s.change_1m:+.2f}% | RSI: {s.rsi:.1f}\n"
                if s.overall_score < 30:
                    response += f"- **Strong consideration for exit** — Score below 30 indicates significant weakness.\n"
                elif s.overall_score < 45:
                    response += f"- **Watch closely** — Underperforming; consider reducing size or tightening stop.\n"
            response += "\n### Top Performers (for context)\n"
            for sym, s, source in scored[-2:]:
                response += f"- **`{sym}`**: {s.overall_score:.1f}/100 | {s.decision} | 30D: {s.change_1m:+.2f}%\n"
    except Exception as e:
        logger.warning(f"Drag analysis error: {e}")
        response += "_Could not analyze holdings._\n"
    return response


async def handle_portfolio_benchmark(context: Dict[str, Any]) -> str:
    """Compare portfolio performance vs SPY benchmark."""
    if not context["has_data"]:
        return "No portfolio data found."
    response = "## Portfolio vs. Benchmark (SPY)\n\n"
    # Get SPY performance
    try:
        from data_sources import get_stock
        spy_df = get_stock("SPY", period="1y")
        if spy_df is not None and not spy_df.empty:
            spy_close = spy_df["Close"].astype(float)
            if isinstance(spy_close, pd.DataFrame):
                spy_close = spy_close.iloc[:, 0]
            spy_1m = float((spy_close.iloc[-1] / spy_close.iloc[-21] - 1) * 100) if len(spy_close) > 21 else 0
            spy_3m = float((spy_close.iloc[-1] / spy_close.iloc[-63] - 1) * 100) if len(spy_close) > 63 else 0
            spy_1y = float((spy_close.iloc[-1] / spy_close.iloc[0] - 1) * 100) if len(spy_close) > 1 else 0

            response += f"| Period | SPY (Benchmark) | Your Portfolio |\n"
            response += f"|--------|-----------------|----------------|\n"
            # Portfolio returns — use P&L data from accounts
            total_pnl = sum(acct.get("total_pnl", 0) for acct in context.get("paper_accounts", []))
            total_value = sum(acct.get("total_value", 0) for acct in context.get("paper_accounts", []))
            portfolio_1m_pct = (total_pnl / max(total_value - total_pnl, 1) * 100) if total_value > 0 else None

            response += f"| 1 Month | {spy_1m:+.2f}% | {f'{portfolio_1m_pct:+.2f}%'if portfolio_1m_pct is not None else 'N/A'} |\n"
            response += f"| 3 Month | {spy_3m:+.2f}% | N/A |\n"
            response += f"| 1 Year | {spy_1y:+.2f}% | N/A |\n\n"
            if portfolio_1m_pct is not None:
                alpha = portfolio_1m_pct - spy_1m
                response += f"**1M Alpha (vs SPY):** {alpha:+.2f}%\n"
                if alpha > 0:
                    response += "You are **outperforming** the S&P 500 benchmark.\n"
                else:
                    response += "You are **underperforming** the S&P 500 benchmark.\n"
    except Exception as e:
        logger.warning(f"Benchmark comparison error: {e}")
        response += "_Could not load SPY benchmark data._\n"
    return response


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DISPATCHER
# ─────────────────────────────────────────────────────────────────────────────

async def handle_portfolio_query(query: str, intent: str, user_id: str = None) -> str:
    """Route portfolio intent to the correct handler."""
    context = load_all_portfolios(user_id)

    if not context["has_data"]:
        return (
            "I couldn't find any portfolio data to analyze."
            "Make sure you have at least one paper trading account with positions,"
            "or have added positions to a manual portfolio.\n\n"
            "Tip: Set up a paper trading account in the **Paper Trading** section,"
            "or add your real holdings in **Portfolio Analyzer → Manual Portfolio**.")

    if intent == "portfolio_grade":
        return await handle_portfolio_grade(context)
    elif intent == "portfolio_fit":
        return await handle_portfolio_fit(query, context)
    elif intent == "portfolio_risk":
        return await handle_portfolio_risk(context)
    elif intent == "portfolio_rebalance":
        return await handle_portfolio_rebalance(context)
    elif intent == "portfolio_drag":
        return await handle_portfolio_drag(context)
    elif intent == "portfolio_benchmark":
        return await handle_portfolio_benchmark(context)
    else:
        return await handle_portfolio_grade(context) # Default fallback
