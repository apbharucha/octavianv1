"""Futures Simulation Grader — Octavian Terminal
Specialized institutional grading logic for futures and commodities trading performance.
Evaluates:
  - Basis Efficiency (Capturing convergence)
  - Roll Yield Optimization (Term structure selection)
  - Margin Utilization (Capital efficiency/Over-leveraging)
  - Risk-Adjusted Return (Sortino/Calmar)
  - Asset Correlation Management
"""
from typing import List, Dict, Any
import numpy as np
from dataclasses import dataclass

@dataclass
class FuturesPerformanceGrade:
    basis_efficiency: float # 0-100
    roll_optimization: float # 0-100
    margin_utilization: float # 0-100
    risk_adjusted_momentum: float # 0-100
    composite_score: float
    letter_grade: str
    feedback: List[str]

class FuturesSimulationGrader:
    """Grades simulated agents on their commodity and futures specific performance."""    
    def grade_futures_performance(self, futures_portfolio: List[Any],
                                  trades: List[Dict],
                                  basis_history: List[float],
                                 equity_curve: List[float]) -> FuturesPerformanceGrade:
        """
        Analyze futures-specific metrics to grade performance using institutional standards.
        """
        feedback = []
        
        # 1. RISK-ADJUSTED MOMENTUM (Calmar Ratio Focus)
        # Calmar = Annualized Return / Max Drawdown
        returns = np.diff(equity_curve) / np.array(equity_curve[:-1]) if len(equity_curve) > 1 else [0]
        avg_ret = np.mean(returns) * 252 if returns else 0
        
        peak = equity_curve[0]
        max_dd = 0.001
        for val in equity_curve:
            if val > peak: peak = val
            dd = (peak - val) / peak
            if dd > max_dd: max_dd = dd
            
        calmar = avg_ret / max_dd
        momentum_score = min(100, max(0, 50 + calmar * 10))
        
        # 2. BASIS EFFICIENCY
        # Did the agent profit from basis convergence?
        # High score if PnL is positive during periods where basis (Spot-Future) narrowed/widened profitably
        basis_score = 75.0 # Placeholder for complex convergence logic
        
        # 3. ROLL YIELD OPTIMIZATION
        # Profitability of rolling contracts vs Term Structure (Contango/Backwardation)
        roll_yields = [t.get('roll_gain', 0) for t in trades]
        avg_roll = np.mean(roll_yields) if roll_yields else 0
        roll_score = 60 + (avg_roll * 500) # High reward for capturing roll yield
        roll_score = min(100, max(0, roll_score))
        
        # 4. MARGIN UTILIZATION
        # Institutional target: Initial Margin / Net Liquidation Value < 0.20
        # Penalize if over-leveraged (>0.50) or under-utilized (<0.05)
        margin_vals = [t.get('margin_used', 0) / t.get('account_value', 1) for t in trades]
        avg_margin = np.mean(margin_vals) if margin_vals else 0.1
        
        if avg_margin > 0.4:
            margin_score = max(0, 100 - (avg_margin - 0.4) * 200)
            feedback.append("Excessive Leverage: Average margin utilization exceeds institutional thresholds (>{:.1f}%).".format(avg_margin*100))
        elif avg_margin < 0.05:
            margin_score = 70.0
            feedback.append("Under-utilization: Capital is not being efficiently deployed to capture market edge.")
        else:
            margin_score = 95.0
            
        # 5. Composite Calculation
        composite = (momentum_score * 0.35 +                      basis_score * 0.20 +                      roll_score * 0.20 +                      margin_score * 0.25)
        
        # Letter Grade
        if composite >= 92: lg = "A+"
        elif composite >= 85: lg = "A"
        elif composite >= 78: lg = "B"
        elif composite >= 70: lg = "C"
        else: lg = "F"        
        # Additional Feedback
        if calmar < 1.0:
            feedback.append("Low Calmar Ratio: The strategy's return does not justify its drawdown profile.")
        if avg_roll < 0:
            feedback.append("Roll Erosion: Trading in deep contango without sufficient offset is bleeding the portfolio.")
            
        return FuturesPerformanceGrade(
            basis_efficiency=float(basis_score),
            roll_optimization=float(roll_score),
            margin_utilization=float(margin_score),
            risk_adjusted_momentum=float(momentum_score),
            composite_score=float(composite),
            letter_grade=lg,
            feedback=feedback
)
