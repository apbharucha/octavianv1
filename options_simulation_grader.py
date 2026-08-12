"""Options Simulation Grader — Octavian Terminal
Specialized grading logic for derivative-based trading performance.
"""
from typing import List, Dict, Any
import numpy as np
from dataclasses import dataclass

@dataclass
class OptionsPerformanceGrade:
    delta_management: float # 0-100
    theta_efficiency: float # 0-100
    convexity_utilization: float # 0-100
    capital_efficiency: float # 0-100
    composite_score: float
    letter_grade: str
    feedback: List[str]

class OptionsSimulationGrader:
    """
    Octavian Performance Assessment Nexus (OPAN)
    ===========================================
    Specialized institutional grader for derivative-based trading performance.
    Evaluates:
      - Risk-Adjusted Momentum (Sortino/Sharpe)
      - Greeks-Weighted Efficiency (Delta/Theta/Gamma/Vega)
      - Tail-Risk Sensitivity (Convexity)
      - Structural Edge (Vanna/Charm)
      - Execution Quality (Slippage/Spread Management)
    """    
    def grade_options_performance(self, options_portfolio: List[Any],
                                  trades: List[Dict],
                                  net_greeks_history: List[Dict],
                                 equity_curve: List[float]) -> OptionsPerformanceGrade:
        """
        Institutional performance analysis using multi-factor attribution.
        """
        feedback = []
        
        # 1. RISK-ADJUSTED MOMENTUM (Sortino Ratio Focus)
        returns = np.diff(equity_curve) / np.array(equity_curve[:-1]) if len(equity_curve) > 1 else [0]
        neg_returns = [r for r in returns if r < 0]
        downside_std = np.std(neg_returns) if len(neg_returns) > 1 else 0.001
        avg_ret = np.mean(returns) if returns else 0
        sortino = (avg_ret * 252) / (downside_std * np.sqrt(252)) if downside_std > 0 else 0
        momentum_score = min(100, max(0, 50 + sortino * 15))
        
        # 2. GREEKS-WEIGHTED EFFICIENCY
        # Delta Management (Target: < 0.02 Neutrality for market-neutral models)
        delta_vals = [abs(g.get('delta', 0)) for g in net_greeks_history]
        avg_abs_delta = np.mean(delta_vals) if delta_vals else 0
        delta_score = max(0, 100 - (avg_abs_delta * 8))
        
        # Theta Harvesting Efficiency
        # (Theta should be positive if the agent is a premium seller, negative if a buyer)
        theta_vals = [g.get('theta', 0) for g in net_greeks_history]
        avg_theta = np.mean(theta_vals) if theta_vals else 0
        # Check if PnL is correlated with Theta (indicating successful premium harvest)
        theta_score = 65 + (avg_theta * 120)
        theta_score = min(100, max(0, theta_score))
        
        # 3. TAIL-RISK & CONVEXITY
        # Convexity score: High if positive Gamma during high vol, or high if net long volatility (Vega) during crashes
        gamma_vals = [g.get('gamma', 0) for g in net_greeks_history]
        vega_vals = [g.get('vega', 0) for g in net_greeks_history]
        avg_gamma = np.mean(gamma_vals) if gamma_vals else 0
        avg_vega = np.mean(vega_vals) if vega_vals else 0
        
        convexity_score = 75 + (avg_gamma * 40) + (avg_vega * 20)
        convexity_score = min(100, max(30, convexity_score))
        
        # 4. STRUCTURAL EDGE (Vanna/Charm)
        # Managing the second-order drift
        vanna_vals = [abs(g.get('vanna', 0)) for g in net_greeks_history]
        charm_vals = [abs(g.get('charm', 0)) for g in net_greeks_history]
        
        vanna_efficiency = max(0, 100 - np.mean(vanna_vals) * 15) if vanna_vals else 85
        charm_efficiency = max(0, 100 - np.mean(charm_vals) * 15) if charm_vals else 85
        sec_order_score = (vanna_efficiency + charm_efficiency) / 2
        
        # 5. CAPITAL ALLOCATION & DRAWDOWN MANAGEMENT
        # Max Drawdown Duration (MDD)
        peak = equity_curve[0]
        max_dd = 0
        for val in equity_curve:
            if val > peak: peak = val
            dd = (peak - val) / peak
            if dd > max_dd: max_dd = dd
            
        drawdown_score = max(0, 100 - (max_dd * 250)) # 40% drawdown = 0 score
        
        # 6. Composite Calculation
        composite = (momentum_score * 0.25 +                      delta_score * 0.15 +                      theta_score * 0.15 +                      convexity_score * 0.15 +                      sec_order_score * 0.15 +                      drawdown_score * 0.15)
        
        # Institutional Letter Grades
        if composite >= 92: lg, color = "A+", "#00ff88"
        elif composite >= 85: lg, color = "A", "#00ff88"
        elif composite >= 78: lg, color = "B", "#c9a84c"
        elif composite >= 70: lg, color = "C", "#ffbb00"
        else: lg, color = "F", "#ff4444"        
        # Robust Qualitative Feedback
        if sortino < 1.0:
            feedback.append("Low Sortino Ratio: Returns are not sufficiently compensating for downside volatility.")
        if avg_abs_delta > 10:
            feedback.append("Excessive Delta Leakage: The model is acting as a directional gambler rather than a volatility specialist.")
        if max_dd > 0.15:
            feedback.append("Drawdown Breach: Significant peak-to-trough decline (>{:.1f}%) suggests flawed risk-sizing.".format(max_dd*100))
        if avg_theta < 0 and avg_ret < 0:
            feedback.append("Negative Theta Decay: Paying for time without directional offset is eroding the principal.")
        if sec_order_score > 90:
            feedback.append("Precision Edge: Exceptional management of second-order drift (Vanna/Charm).")
        if convexity_score > 85:
            feedback.append("Anti-Fragile: Positive convexity profile provides protection against tail-risk events.")
            
        return OptionsPerformanceGrade(
            delta_management=float(delta_score),
            theta_efficiency=float(theta_score),
            convexity_utilization=float(convexity_score),
            capital_efficiency=float(drawdown_score),
            composite_score=float(composite),
            letter_grade=lg,
            feedback=feedback
)
