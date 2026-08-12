import pandas as pd
import numpy as np
from indicators import add_indicators

def analyze_price_chart(df, indicators_df=None):
    """AI analysis of price chart patterns and indicators"""
    if df.empty:
        return "Insufficient data for chart analysis."
    
    if indicators_df is None:
        indicators_df = add_indicators(df.copy())
    
    if indicators_df.empty:
        indicators_df = df.copy()
    
    latest = indicators_df.iloc[-1] if not indicators_df.empty else df.iloc[-1]
    
    analysis = []
    
    # Price action analysis
    if len(df) >= 20:
        recent_high = df['High'].tail(20).max()
        recent_low = df['Low'].tail(20).min()
        current_price = df['Close'].iloc[-1]
        
        price_position = (current_price - recent_low) / (recent_high - recent_low) if recent_high > recent_low else 0.5
        
        if price_position > 0.8:
            analysis.append("[UP] **Price Position:** Trading near recent highs - potential resistance zone. Watch for reversal signals.")
        elif price_position < 0.2:
            analysis.append("[DOWN] **Price Position:** Trading near recent lows - potential support zone. Watch for bounce signals.")
        else:
            analysis.append(" **Price Position:** Trading in middle range - neutral positioning, waiting for breakout.")
    
    # Trend analysis
    if 'ema20' in indicators_df.columns and 'ema50' in indicators_df.columns:
        ema20 = indicators_df['ema20'].iloc[-1]
        ema50 = indicators_df['ema50'].iloc[-1]
        price = df['Close'].iloc[-1]
        
        if pd.notna(ema20) and pd.notna(ema50):
            if ema20 > ema50:
                spread = ((ema20 - ema50) / ema50) * 100
                if spread > 5:
                    analysis.append(f"[GOOD] **Trend:** Strong bullish trend - EMA 20 (${ema20:.2f}) well above EMA 50 (${ema50:.2f}). Uptrend momentum is strong.")
                else:
                    analysis.append(f"[GOOD] **Trend:** Moderate bullish trend - EMA 20 above EMA 50. Uptrend intact but weakening.")
            else:
                spread = ((ema50 - ema20) / ema50) * 100
                if spread > 5:
                    analysis.append(f"[ALERT] **Trend:** Strong bearish trend - EMA 20 (${ema20:.2f}) well below EMA 50 (${ema50:.2f}). Downtrend momentum is strong.")
                else:
                    analysis.append(f"[ALERT] **Trend:** Moderate bearish trend - EMA 20 below EMA 50. Downtrend intact but weakening.")
            
            # Price vs EMAs
            if price > ema20 > ema50:
                analysis.append("[OK] **Price Action:** Price above both EMAs - bullish alignment. Strong uptrend structure.")
            elif price < ema20 < ema50:
                analysis.append("[X] **Price Action:** Price below both EMAs - bearish alignment. Strong downtrend structure.")
            elif ema20 > price > ema50:
                analysis.append("[WARN] **Price Action:** Price between EMAs - potential trend change. Watch for confirmation.")
            elif price > ema50 > ema20:
                analysis.append(" **Price Action:** Mixed signals - price above 50 EMA but below 20 EMA. Possible pullback in uptrend.")
    
    # RSI analysis
    if 'rsi' in indicators_df.columns:
        rsi = indicators_df['rsi'].iloc[-1]
        if pd.notna(rsi):
            if rsi > 70:
                analysis.append(f"[ALERT] **RSI:** {rsi:.1f} - Overbought territory. High probability of pullback or reversal. Consider taking profits on longs or entering shorts.")
            elif rsi < 30:
                analysis.append(f"[GOOD] **RSI:** {rsi:.1f} - Oversold territory. High probability of bounce or reversal. Potential buying opportunity.")
            elif rsi > 60:
                analysis.append(f"[NOTE] **RSI:** {rsi:.1f} - Approaching overbought. Monitor for reversal signals.")
            elif rsi < 40:
                analysis.append(f"[NOTE] **RSI:** {rsi:.1f} - Approaching oversold. Monitor for bounce signals.")
            else:
                analysis.append(f" **RSI:** {rsi:.1f} - Neutral zone. No extreme momentum signals.")
    
    # Volume analysis
    if 'Volume' in df.columns:
        if len(df) >= 20:
            avg_volume = df['Volume'].tail(20).mean()
            current_volume = df['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            if volume_ratio > 1.5:
                price_change = (df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100 if len(df) > 1 else 0
                if price_change > 0:
                    analysis.append(f"[DATA] **Volume:** {volume_ratio:.1f}x average - High volume on up move confirms bullish momentum. Strong buying interest.")
                else:
                    analysis.append(f"[DATA] **Volume:** {volume_ratio:.1f}x average - High volume on down move confirms bearish momentum. Strong selling pressure.")
            elif volume_ratio < 0.7:
                analysis.append(f"[DATA] **Volume:** {volume_ratio:.1f}x average - Low volume suggests lack of conviction. Wait for volume confirmation.")
            else:
                analysis.append(f"[DATA] **Volume:** {volume_ratio:.1f}x average - Normal volume levels. Standard market participation.")
    
    # Candlestick patterns
    if len(df) >= 3:
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Bullish patterns
        if current['Close'] > current['Open'] and prev['Close'] < prev['Open']:
            if current['Close'] > prev['High']:
                analysis.append(" **Pattern:** Bullish engulfing pattern - Strong reversal signal. Potential trend change to upside.")
        
        # Bearish patterns
        if current['Close'] < current['Open'] and prev['Close'] > prev['Open']:
            if current['Close'] < prev['Low']:
                analysis.append(" **Pattern:** Bearish engulfing pattern - Strong reversal signal. Potential trend change to downside.")
        
        # Doji (indecision)
        body_size = abs(current['Close'] - current['Open'])
        range_size = current['High'] - current['Low']
        if range_size > 0 and body_size / range_size < 0.1:
            analysis.append(" **Pattern:** Doji candle - Market indecision. Potential reversal or consolidation ahead.")
    
    # Momentum analysis
    if len(df) >= 10:
        returns_5d = (df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1) * 100 if len(df) >= 5 else 0
        returns_10d = (df['Close'].iloc[-1] / df['Close'].iloc[-10] - 1) * 100
        
        if returns_5d > 5 and returns_10d > 5:
            analysis.append(f"[START] **Momentum:** Strong positive momentum - {returns_5d:.1f}% in 5 days, {returns_10d:.1f}% in 10 days. Uptrend accelerating.")
        elif returns_5d < -5 and returns_10d < -5:
            analysis.append(f"[DOWN] **Momentum:** Strong negative momentum - {returns_5d:.1f}% in 5 days, {returns_10d:.1f}% in 10 days. Downtrend accelerating.")
        elif returns_5d > returns_10d:
            analysis.append(f"[UP] **Momentum:** Improving momentum - Recent gains outpacing longer-term. Potential trend strengthening.")
        elif returns_5d < returns_10d:
            analysis.append(f"[DOWN] **Momentum:** Weakening momentum - Recent losses worse than longer-term. Potential trend weakening.")
    
    # Support/Resistance
    if len(df) >= 50:
        recent_high = df['High'].tail(50).max()
        recent_low = df['Low'].tail(50).min()
        current = df['Close'].iloc[-1]
        
        if abs(current - recent_high) / recent_high < 0.02:
            analysis.append(f"[STOP] **Resistance:** Price near 50-day high (${recent_high:.2f}). Watch for rejection or breakout.")
        elif abs(current - recent_low) / recent_low < 0.02:
            analysis.append(f"[STOP] **Support:** Price near 50-day low (${recent_low:.2f}). Watch for bounce or breakdown.")
    
    # Global Integration: Full Model Outlook
    try:
        from quant_ensemble_model import get_quant_ensemble
        qe = get_quant_ensemble()
        # Note: In this context, we don't always have a symbol string passed in.
        # But if we have DF, we can try to guess or just use the data for prediction if the model allows.
        # For now, we'll only append if we can get a prediction.
        q_signal = qe.predict("CHART", df)
        
        from market_movers import _generate_ai_insights, _calculate_technicals
        ai_tech = _calculate_technicals(df)
        ai_data = _generate_ai_insights("CHART", ai_tech, df)
        
        f_score = ai_data.get("bullish_prob", 0.5) * 100
        q_prob = q_signal.probability * 100
        u_score = (f_score + q_prob) / 2
        
        if u_score >= 80: label = "STRONG OVERWEIGHT"
        elif u_score >= 60: label = "ACCUMULATE"
        elif u_score >= 40: label = "NEUTRAL"
        elif u_score >= 20: label = "REDUCE"
        else: label = "STRONG UNDERWEIGHT"
        
        outlook_str = f"\n\n**FULL MODEL OUTLOOK: {label} ({u_score:.1f} Conviction)**\n"
        outlook_str += f"Technical bias ({f_score:.0f}/100) aligned with Quant signal ({q_signal.direction} {q_prob:.0f}%)."
        analysis.append(outlook_str)
    except Exception:
        pass

    return "\n\n".join(analysis)

def analyze_rsi_chart(rsi_series):
    """Analyze RSI chart specifically"""
    if rsi_series.empty or len(rsi_series) < 10:
        return "Insufficient RSI data for analysis."
    
    latest = rsi_series.iloc[-1]
    analysis = []
    
    # Current level
    if latest > 70:
        analysis.append(f"[ALERT] **Current RSI:** {latest:.1f} - Severely overbought. High probability of reversal.")
    elif latest < 30:
        analysis.append(f"[GOOD] **Current RSI:** {latest:.1f} - Severely oversold. High probability of bounce.")
    elif latest > 60:
        analysis.append(f"[NOTE] **Current RSI:** {latest:.1f} - Approaching overbought. Monitor closely.")
    elif latest < 40:
        analysis.append(f"[NOTE] **Current RSI:** {latest:.1f} - Approaching oversold. Watch for entry.")
    else:
        analysis.append(f" **Current RSI:** {latest:.1f} - Neutral zone. No extreme signals.")
    
    # Divergence detection
    if len(rsi_series) >= 20:
        rsi_trend = rsi_series.tail(10).mean() - rsi_series.tail(20).head(10).mean()
        if rsi_trend > 5:
            analysis.append("[UP] **RSI Trend:** Rising RSI indicates strengthening momentum.")
        elif rsi_trend < -5:
            analysis.append("[DOWN] **RSI Trend:** Falling RSI indicates weakening momentum.")
    
    return "\n\n".join(analysis) if analysis else "RSI analysis: Neutral momentum indicators."

def analyze_volume_chart(volume_series, price_series):
    """Analyze volume chart"""
    if volume_series.empty or price_series.empty:
        return "Insufficient volume data for analysis."
    
    if len(volume_series) < 20:
        return "Insufficient data for volume analysis."
    
    latest_vol = volume_series.iloc[-1]
    avg_vol = volume_series.tail(20).mean()
    vol_ratio = latest_vol / avg_vol if avg_vol > 0 else 1
    
    analysis = []
    
    if vol_ratio > 2:
        analysis.append(f"[DATA] **Volume Spike:** {vol_ratio:.1f}x average volume - Significant interest. {'Bullish' if price_series.iloc[-1] > price_series.iloc[-2] else 'Bearish'} price action confirms direction.")
    elif vol_ratio > 1.5:
        analysis.append(f"[DATA] **Elevated Volume:** {vol_ratio:.1f}x average - Above-normal participation confirms current move.")
    elif vol_ratio < 0.5:
        analysis.append(f"[DATA] **Low Volume:** {vol_ratio:.1f}x average - Lack of conviction. Current move may not be sustainable.")
    else:
        analysis.append(f"[DATA] **Normal Volume:** {vol_ratio:.1f}x average - Standard market participation.")
    
    # Volume trend
    if len(volume_series) >= 10:
        vol_trend = volume_series.tail(5).mean() / volume_series.tail(10).head(5).mean()
        if vol_trend > 1.2:
            analysis.append("[UP] **Volume Trend:** Increasing volume suggests growing interest.")
        elif vol_trend < 0.8:
            analysis.append("[DOWN] **Volume Trend:** Decreasing volume suggests waning interest.")
    
    return "\n\n".join(analysis) if analysis else "Volume analysis: Standard participation levels."

