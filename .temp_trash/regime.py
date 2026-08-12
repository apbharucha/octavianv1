from data_sources import get_vix, get_stock
from indicators import add_indicators
import pandas as pd

def volatility_regime(symbol=None):
    try:
        vix = get_vix()
        if vix.empty or "Close" not in vix.columns:
            # Fallback to simple VIX approximation if API fails
            # This ensures we don't return N/A for UI components
            return "MID VOL (Fallback)"
        
        close_col = vix["Close"]
        if isinstance(close_col, pd.DataFrame):
            close_col = close_col.iloc[:, 0]
        
        lvl = close_col.iloc[-1]
        if pd.isna(lvl):
            return "MID VOL (Fallback)"
        
        lvl = float(lvl)
        if lvl < 15:
            return "LOW VOL (Trend-friendly)"
        elif lvl < 25:
            return "MID VOL"
        else:
            return "HIGH VOL (Risk-Off)"
    except Exception as e:
        print(f"Error in volatility_regime: {e}")
        return "MID VOL (Fallback)"

def risk_on_off():
    try:
        spx = get_stock("^GSPC")
        if spx.empty:
            # Fallback to SPY if GSPC fails (common with some free APIs)
            spx = get_stock("SPY")
            
        if spx.empty:
            return "NEUTRAL (No Data)"
        
        spx = add_indicators(spx)
        if spx.empty or "ema20" not in spx.columns or "ema50" not in spx.columns:
            return "NEUTRAL (Calc Error)"
        
        ema20_val = spx["ema20"].iloc[-1]
        ema50_val = spx["ema50"].iloc[-1]
        
        if pd.isna(ema20_val) or pd.isna(ema50_val) or ema50_val == 0:
            return "NEUTRAL"
        
        trend = (float(ema20_val) - float(ema50_val)) / float(ema50_val)
        
        vix = get_vix()
        vix_val = 20.0 # Default fallback
        
        if not vix.empty and "Close" in vix.columns:
            close_col = vix["Close"]
            if isinstance(close_col, pd.DataFrame):
                close_col = close_col.iloc[:, 0]
            val = close_col.iloc[-1]
            if not pd.isna(val):
                vix_val = float(val)

        if trend > 0 and vix_val < 20:
            return "RISK-ON"
        elif vix_val > 25:
            return "RISK-OFF"
        return "NEUTRAL"
    except Exception as e:
        print(f"Error in risk_on_off: {e}")
        return "NEUTRAL"
