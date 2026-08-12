def generate_thesis(name, score, rsi, regime, risk_mode):
    bias = "Bullish" if score > 0 else "Bearish"
    return f"""
{name} Trade Thesis
-------------------
Bias: {bias}
Trend Score: {round(score,3)}
RSI: {round(rsi,1)}
Volatility Regime: {regime}
Market Mode: {risk_mode}

Reasoning:
Trend alignment + momentum confirmation within current macro regime.
"""
