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
        """Generates strategy suggestions based on user goals."""
        suggestions = []
        
        if "conservative" in risk_tolerance.lower():
            suggestions.append({
                "name": "Quality Yield + Covered Call",
                "type": "Income",
                "description": "Hold top-quintile ROIC mega-caps and sell 30-delta calls against positions.",
                "expected_sharpe": 1.2,
                "risk_profile": "Low"
            })
            suggestions.append({
                "name": "Statistical Arbitrage (Cointegrated Pairs)",
                "type": "Market Neutral",
                "description": "Trade mean-reversion on highly cointegrated sector pairs (e.g., KO vs PEP).",
                "expected_sharpe": 1.5,
                "risk_profile": "Low"
            })
        elif "aggressive" in risk_tolerance.lower():
            suggestions.append({
                "name": "Momentum + Volatility Breakout",
                "type": "Directional",
                "description": "Buy 90-day highs with expanding ATR. Size inversely to volatility.",
                "expected_sharpe": 0.8,
                "risk_profile": "High"
            })
            suggestions.append({
                "name": "Leveraged Risk Parity",
                "type": "Macro",
                "description": "Equal risk contribution across Equities, Treasuries, and Commodities, levered 2x.",
                "expected_sharpe": 1.1,
                "risk_profile": "High"
            })
        else:
            suggestions.append({
                "name": "Multi-Factor Ensemble",
                "type": "Smart Beta",
                "description": "Long top decile Value/Momentum/Quality, Short bottom decile.",
                "expected_sharpe": 1.0,
                "risk_profile": "Medium"
            })
            
        return suggestions

_engine = None
def get_strategy_intelligence_engine() -> StrategyIntelligenceEngine:
    global _engine
    if _engine is None:
        _engine = StrategyIntelligenceEngine()
    return _engine
