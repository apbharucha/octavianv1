import pandas as pd
import numpy as np
import scipy.stats as si
from datetime import datetime, timedelta

class OptionsAnalyzer:
    """
    Analyze options strategies based on market conditions, Volatility, and Greeks.
    """
    def __init__(self):
        pass
    
    def calculate_implied_volatility_estimate(self, df, lookback=20):
        """
        Estimate IV using historical volatility * risk premium buffer.
        Real implementation would use Option Chain data if available.
        """
        if df.empty or len(df) < lookback:
            return 0.0
            
        try:
            # Calculate annualized volatility
            log_returns = np.log(df['Close'] / df['Close'].shift(1))
            hist_vol = log_returns.tail(lookback).std() * np.sqrt(252)
            
            # Add a "Risk Premium" buffer (IV is usually > HV)
            # In high stress, this multiplier increases.
            iv_estimate = hist_vol * 1.25 
            return iv_estimate
        except Exception:
            return 0.0

    def calculate_probability_itm(self, current_price, strike_price, iv, days_to_expiration):
        """
        Approximation of Probability ITM using standard normal distribution (Delta approximation).
        """
        if days_to_expiration <= 0 or current_price <= 0 or strike_price <= 0 or iv <= 0:
            return 0.0
            
        t = days_to_expiration / 365.0
        r = 0.04 # Risk free rate assumption
        
        # d1 is used for Delta
        d1 = (np.log(current_price / strike_price) + (r + 0.5 * iv ** 2) * t) / (iv * np.sqrt(t))
        
        # Calls: N(d1) is approx delta/prob ITM
        if strike_price >= current_price:
             return si.norm.cdf(d1)
        
        # Puts: N(-d1)
        else:
             return si.norm.cdf(-d1)

    def calculate_expected_move(self, current_price, iv, days_to_expiration):
        """
        Expected move = Price * IV * sqrt(Days/365)
        """
        if days_to_expiration <= 0: return 0.0
        return current_price * iv * np.sqrt(days_to_expiration / 365.0)

    def analyze_options_opportunity(self, symbol, df, analysis, current_price, trader_type="Swing Trader"):
        """
        Analyze and recommend options strategies based on market conditions
        """
        iv = self.calculate_implied_volatility_estimate(df)
        
        # Default forecast horizon (Swing = ~14 days)
        dte = 14
        expected_move = self.calculate_expected_move(current_price, iv, dte)
        
        strategy = "None"
        confidence = 0.0
        details = "No clear options setup."
        
        signal = analysis.get("signal", "NEUTRAL")
        # Normalize signal string
        signal = signal.upper()
        
        # Bullish Strategies
        if "BUY" in signal or "BULLISH" in signal:
            target = current_price + expected_move
            prob_itm = self.calculate_probability_itm(current_price, target, iv, dte)
            
            if iv < 0.30:
                strategy = "Long Call"
                details = f"Low IV ({iv:.1%}). Buy ATM/OTM Call. Target: {target:.2f}"
                confidence = 80
            else:
                strategy = "Bull Call Spread"
                details = f"High IV ({iv:.1%}). Buy ATM Call / Sell OTM Call to reduce cost."
                confidence = 75
                
        # Bearish Strategies
        elif "SELL" in signal or "BEARISH" in signal:
            target = current_price - expected_move
            prob_itm = self.calculate_probability_itm(current_price, target, iv, dte)
            
            if iv < 0.30:
                strategy = "Long Put"
                details = f"Low IV ({iv:.1%}). Buy ATM/OTM Put. Target: {target:.2f}"
                confidence = 80
            else:
                strategy = "Bear Put Spread"
                details = f"High IV ({iv:.1%}). Buy ATM Put / Sell OTM Put."
                confidence = 75
                
        # Neutral Strategies
        else:
            if iv > 0.40:
                strategy = "Iron Condor"
                details = f"High IV ({iv:.1%}). Range-bound. Sell OTM Wings."
                confidence = 65
            elif iv < 0.20:
                # Low IV wait
                strategy = "Wait / Cash"
                details = "Low volatility and neutral signal."
                confidence = 50
        
        return {
            "symbol": symbol,
            "iv_annual": round(iv, 3),
            "expected_move_14d": round(expected_move, 2),
            "recommended_strategy": strategy,
            "details": details,
            "confidence": confidence,
            "prob_itm_target": round(self.calculate_probability_itm(current_price, current_price + expected_move, iv, dte) * 100, 1)
        }

def get_options_analyzer():
    return OptionsAnalyzer()
