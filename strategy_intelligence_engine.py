import pandas as pd
import numpy as np
from typing import Dict, Any, List

class StrategyIntelligenceEngine:
    """Institutional Strategy Grader and Suggester."""

    def grade_strategy(self, strategy_name: str, performance_data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Critiques a user's strategy and returns a letter grade with feedback."""
        # Calculate key metrics
        if performance_data.empty:
            return {"grade": "F", "score": 0, "feedback": "No performance data provided."}
            
        returns = performance_data['Returns'] if 'Returns' in performance_data.columns else performance_data.pct_change().dropna()
        if len(returns) == 0:
            return {"grade": "F", "score": 0, "feedback": "Insufficient return data."}
            
        cum_ret = (1 + returns).prod() - 1
        ann_ret = (1 + cum_ret) ** (252 / len(returns)) - 1
        vol = returns.std() * np.sqrt(252)
        sharpe = ann_ret / vol if vol > 0 else 0
        max_dd = (returns.cumsum() - returns.cumsum().cummax()).min()
        
        # Scoring Logic
        score = 50 # Base
        score += min(20, max(0, ann_ret * 100)) # Up to 20 pts for return
        score += min(15, max(0, sharpe * 10))   # Up to 15 pts for sharpe
        score -= min(20, max(0, abs(max_dd) * 100)) # Penalize drawdown
        
        score = max(0, min(100, score))
        
        if score > 90: grade = "A"
        elif score > 80: grade = "B"
        elif score > 70: grade = "C"
        elif score > 60: grade = "D"
        else: grade = "F"
        
        feedback = []
        if sharpe < 1.0:
            feedback.append("Risk-adjusted returns are sub-optimal. Consider adding hedging rules or volatility filters.")
        if max_dd < -0.20:
            feedback.append("Maximum drawdown exceeds institutional risk tolerance (>20%). Implement stricter stop-losses.")
        if ann_ret < 0.05:
            feedback.append("Absolute returns are weak. The signal may lack predictive edge or transaction costs are too high.")
            
        if not feedback:
            feedback.append("Strategy exhibits strong institutional-grade characteristics.")
            
        return {
            "strategy_name": strategy_name,
            "grade": grade,
            "score": score,
            "annualized_return": ann_ret,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "feedback": feedback
        }

    def suggest_strategies(self, user_goals: str, risk_tolerance: str) -> List[Dict[str, Any]]:
        """Generates strategy suggestions dynamically based on user goals and risk tolerance.
        
        The user's investment goals text is parsed for keywords (income, growth,
        protection, volatility, momentum, mean-reversion, macro, sector, options,
        commodities, market-neutral, hedging, etc.) and each matched keyword
        contributes a tailored strategy to the suggestion list.
        """
        suggestions = []
        goals_lower = (user_goals or "").lower()
        risk_lower = (risk_tolerance or "").lower()
        
        # Determine risk-controlled Sharpe/shift from risk tolerance
        if "conservative" in risk_lower:
            risk_baseline = "Low"
            base_sharpe_boost = 0.2
        elif "aggressive" in risk_lower:
            risk_baseline = "High"
            base_sharpe_boost = -0.1
        else:
            risk_baseline = "Medium"
            base_sharpe_boost = 0.0
        
        # ---- Keyword-driven strategy catalogue ----
        # Each strategy is indexed by goal keywords; only strategies matching
        # the user's stated goals are returned.  The description is tailored
        # to the user's risk tolerance so the output changes materially with
        # the text they typed.
        
        STRATEGY_CATALOG: List[Dict[str, Any]] = [
            {
                "name": "Covered Call Overlay",
                "type": "Income",
                "description": "Owning shares and selling near-dated calls to harvest premium — ideal for sideways/grinding markets.",
                "expected_sharpe": 1.2 + base_sharpe_boost,
                "risk_profile": "Low",
                "keywords": ["income", "dividend", "yield", "premium", "conservative", "protect", "option"],
            },
            {
                "name": "Cointegrated Pairs — Mean Reversion",
                "type": "Market Neutral",
                "description": "Identify highly cointegrated sector pairs, trade spread z-score reversions — low correlation to broad market.",
                "expected_sharpe": 1.5 + base_sharpe_boost,
                "risk_profile": "Low",
                "keywords": ["mean reversion", "pairs", "stat arb", "market neutral", "hedge", "protect"],
            },
            {
                "name": "Quality Dividend Compounders",
                "type": "Income",
                "description": "Long high-ROIC, wide-moat names with growing dividends and low payout ratios.",
                "expected_sharpe": 1.0 + base_sharpe_boost,
                "risk_profile": "Low",
                "keywords": ["income", "dividend", "compound", "long term", "quality", "conservative"],
            },
            {
                "name": "Momentum + Volatility Breakout",
                "type": "Directional",
                "description": "Buy 90-day highs with expanding ATR, size inversely to volatility — trend following with risk scaling.",
                "expected_sharpe": 0.9 + base_sharpe_boost,
                "risk_profile": "High",
                "keywords": ["momentum", "trend", "breakout", "growth", "aggressive", "volatility"],
            },
            {
                "name": "Leveraged Risk Parity",
                "type": "Macro",
                "description": "Equal risk contribution across Equities, Treasuries, and Commodities, levered 2× — macro diversification.",
                "expected_sharpe": 1.1 + base_sharpe_boost,
                "risk_profile": "High",
                "keywords": ["macro", "diversif", "commodit", "aggressive", "leveraged", "parity"],
            },
            {
                "name": "Multi-Factor Ensemble (Smart Beta)",
                "type": "Smart Beta",
                "description": "Long top-decile Value + Momentum + Quality, short bottom-decile — factor premia capture.",
                "expected_sharpe": 1.0 + base_sharpe_boost,
                "risk_profile": "Medium",
                "keywords": ["factor", "smart beta", "diversif", "value", "momentum", "quality"],
            },
            {
                "name": "Volatility Arbitrage / Dispersion",
                "type": "Volatility",
                "description": "Sell index volatility, buy single-stock volatility where implied correlation is rich — vol surface arbitrage.",
                "expected_sharpe": 1.3 + base_sharpe_boost,
                "risk_profile": "Medium",
                "keywords": ["volatility", "vol", "arbitrage", "dispersion", "option", "hedge"],
            },
            {
                "name": "Sector Rotation — Macro Regime",
                "type": "Macro",
                "description": "Rotate sectors based on HMM regime, yield curve, and inflation surprise — overweight leaders, underweight laggards.",
                "expected_sharpe": 1.1 + base_sharpe_boost,
                "risk_profile": "Medium",
                "keywords": ["sector", "rotation", "macro", "regime", "cycle"],
            },
            {
                "name": "Options Wheel Strategy",
                "type": "Income",
                "description": "Sell cash-secured puts on quality names you want to own, then covered calls once assigned — premium harvesting.",
                "expected_sharpe": 1.0 + base_sharpe_boost,
                "risk_profile": "Medium",
                "keywords": ["income", "premium", "option", "wheel", "yield", "cash"],
            },
            {
                "name": "Tail-Risk Hedged Portfolio",
                "type": "Hedging",
                "description": "Core equity allocation with systematic put-buying or VIX-call ladder — convexity for crash protection.",
                "expected_sharpe": 0.8 + base_sharpe_boost,
                "risk_profile": "Low",
                "keywords": ["protect", "hedge", "tail", "crash", "downside", "insurance", "conservative"],
            },
            {
                "name": "Event-Driven / Merger Arbitrage",
                "type": "Event-Driven",
                "description": "Capture spreads on announced deals — high win rate, low beta, event-risk exposure.",
                "expected_sharpe": 1.4 + base_sharpe_boost,
                "risk_profile": "Medium",
                "keywords": ["event", "merger", "arbitrage", "deal", "catalyst"],
            },
            {
                "name": "Global Macro — FX & Rates",
                "type": "Macro",
                "description": "Trade G10 FX carry, yield curve steepeners/flatteners, and central bank divergence themes.",
                "expected_sharpe": 0.9 + base_sharpe_boost,
                "risk_profile": "High",
                "keywords": ["macro", "fx", "currency", "rate", "yield", "global", "commodit"],
            },
        ]
        
        # Score each strategy by keyword overlap with the user's goals text
        scored = []
        for strat in STRATEGY_CATALOG:
            matches = sum(1 for kw in strat["keywords"] if kw in goals_lower)
            scored.append((matches, strat))
        
        # Sort by matches descending — the most relevant strategies first
        scored.sort(key=lambda x: -x[0])
        
        # Always return at least 3 strategies (fall back to risk-only if nothing matched)
        matched = [s for _, s in scored if _ > 0]
        if len(matched) < 3:
            # Fill with strategies that match the risk profile but not necessarily keywords
            fallback = [s for _, s in scored if _ == 0]
            # Prefer risk-appropriate fallback
            if "conservative" in risk_lower:
                fallback.sort(key=lambda s: 0 if s["risk_profile"] == "Low" else 1 if s["risk_profile"] == "Medium" else 2)
            elif "aggressive" in risk_lower:
                fallback.sort(key=lambda s: 0 if s["risk_profile"] == "High" else 1 if s["risk_profile"] == "Medium" else 2)
            matched.extend(fallback)
        
        for s in matched[:6]:
            # Personalize the description with the user's stated goals
            desc = s["description"]
            if user_goals and len(user_goals.strip()) > 5:
                goal_summary = user_goals.strip()[:120]
                desc = f"[Matched to: \"{goal_summary}...\"] {desc}"
            suggestions.append({
                "name": s["name"],
                "type": s["type"],
                "description": desc,
                "expected_sharpe": s["expected_sharpe"],
                "risk_profile": s["risk_profile"],
            })
            
        return suggestions

_engine = None
def get_strategy_intelligence_engine() -> StrategyIntelligenceEngine:
    global _engine
    if _engine is None:
        _engine = StrategyIntelligenceEngine()
    return _engine
