import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List

from trader_profile import get_trader_profile
from data_sources import get_stock

class AdaptiveReasoningEngine:
    """
    Makes Octavian's models adapt dynamically to current market regimes
    and personalizes output to the user's trader profile.
    """
    def __init__(self):
        self.market_regime = "NEUTRAL"
        self.volatility_state = "NORMAL"
        self.last_update = None
        self._update_market_regime()

    def _update_market_regime(self):
        """Analyze SPY to determine the current macro market regime."""
        try:
            spy = get_stock("SPY", period="6mo")
            if spy is not None and not spy.empty:
                returns = spy['Close'].pct_change().dropna()
                recent_return = (spy['Close'].iloc[-1] / spy['Close'].iloc[-20]) - 1
                vol = returns.tail(20).std() * np.sqrt(252)

                # Determine regime
                if recent_return > 0.05:
                    self.market_regime = "BULL_TREND"
                elif recent_return < -0.05:
                    self.market_regime = "BEAR_TREND"
                else:
                    self.market_regime = "CHOPPY"

                # Determine volatility
                if vol > 0.25:
                    self.volatility_state = "HIGH_VOLATILITY"
                elif vol < 0.12:
                    self.volatility_state = "LOW_VOLATILITY"
                else:
                    self.volatility_state = "NORMAL"

            self.last_update = datetime.now()
        except Exception as e:
            print(f"Failed to update market regime: {e}")

    def get_adaptive_weights(self) -> Dict[str, float]:
        """
        Adjust model weights based on market regime and trader profile.
        Used by UnbiasedMarketAnalyzer.
        """
        # Base weights
        weights = {
            'momentum': 0.20,
            'volatility_opportunity': 0.15,
            'volume_anomalies': 0.10,
            'price_inefficiencies': 0.10,
            'technical_breakouts': 0.10,
            'statistical_edges': 0.10,
            'quant_ensemble': 0.25,
        }

        # 1. Market Regime Adjustments
        if self.market_regime == "BULL_TREND":
            weights['momentum'] += 0.10
            weights['technical_breakouts'] += 0.05
            weights['statistical_edges'] -= 0.15
        elif self.market_regime == "BEAR_TREND":
            weights['volatility_opportunity'] += 0.10
            weights['statistical_edges'] += 0.05
            weights['momentum'] -= 0.15
        elif self.market_regime == "CHOPPY":
            weights['statistical_edges'] += 0.15
            weights['price_inefficiencies'] += 0.10
            weights['momentum'] -= 0.15
            weights['technical_breakouts'] -= 0.10

        if self.volatility_state == "HIGH_VOLATILITY":
            weights['volatility_opportunity'] += 0.15
            weights['quant_ensemble'] -= 0.05
            weights['technical_breakouts'] -= 0.10

        # 2. Trader Profile Adjustments
        profile = get_trader_profile()
        style = profile.get("trading_style", "Swing Trader")
        risk = profile.get("risk_profile", "Moderate")

        if style == "Scalper" or style == "Day Trader":
            weights['volume_anomalies'] += 0.10
            weights['price_inefficiencies'] += 0.10
            weights['quant_ensemble'] -= 0.10
        elif style == "Position Trader" or style == "Long-Term Investor":
            weights['quant_ensemble'] += 0.15
            weights['momentum'] += 0.05
            weights['volatility_opportunity'] -= 0.10
            weights['volume_anomalies'] -= 0.10

        if risk == "Conservative":
            weights['statistical_edges'] += 0.10
            weights['quant_ensemble'] += 0.05
            weights['volatility_opportunity'] -= 0.15
        elif risk == "Aggressive" or risk == "Very Aggressive":
            weights['volatility_opportunity'] += 0.15
            weights['technical_breakouts'] += 0.05
            weights['statistical_edges'] -= 0.10

        # Normalize weights to sum to 1.0
        # Ensure no negative weights
        for k in weights:
            if weights[k] < 0.05:
                weights[k] = 0.05
                
        total_weight = sum(weights.values())
        return {k: v / total_weight for k, v in weights.items()}

    def get_dynamic_insights(self) -> List[str]:
        """Generate personalized, adaptive insights based on profile + regime."""
        profile = get_trader_profile()
        style = profile.get("trading_style", "Swing Trader")
        risk = profile.get("risk_profile", "Moderate")
        exp = profile.get("experience_level", "Intermediate")
        
        insights = []

        # Regime-specific tips
        if self.market_regime == "BULL_TREND":
            insights.append("[TREND] **Market Regime (Bullish):** Trend-following strategies and breakout buying have high mathematical expectancy currently.")
            if risk == "Conservative":
                insights.append("[RISK] **Protective Stance:** Despite the bull trend, conservative portfolios should ensure stops are trailed tightly.")
        elif self.market_regime == "BEAR_TREND":
            insights.append("[TREND] **Market Regime (Bearish):** Momentum is negative. Focus on short setups, relative strength pairs, or cash preservation.")
            if style in ["Scalper", "Day Trader"]:
                insights.append("[VOLATILITY] Intraday bounces will be aggressive. Fade the rips.")
        elif self.market_regime == "CHOPPY":
            insights.append("[TREND] **Market Regime (Choppy/Rangebound):** Breakouts are likely to fail. Switch to mean-reversion and statistical arbitrage strategies.")
            
        if self.volatility_state == "HIGH_VOLATILITY":
            insights.append("[VOLATILITY] **VIX Elevated:** Position sizes must be reduced mathematically to maintain constant portfolio risk.")
        elif self.volatility_state == "LOW_VOLATILITY":
            insights.append("[VOLATILITY] **VIX Suppressed:** Options are cheap. Consider long straddles/strangles ahead of macro catalysts.")

        # Experience-specific adaptations
        if exp == "Beginner" and self.volatility_state == "HIGH_VOLATILITY":
            insights.append("[WARNING] Extreme volatility is dangerous for beginners. Consider paper trading until VIX compresses.")
        elif exp in ["Professional", "Institutional"] and self.market_regime == "CHOPPY":
            insights.append("[PRO] Edge lies in market-neutral dispersion trading and relative value volatility trades right now.")

        return insights

_adaptive_engine = None
def get_adaptive_engine() -> AdaptiveReasoningEngine:
    global _adaptive_engine
    if _adaptive_engine is None:
        _adaptive_engine = AdaptiveReasoningEngine()
    return _adaptive_engine
