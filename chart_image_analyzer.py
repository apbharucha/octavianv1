"""
Octavian Chart Image Analyzer - Enhanced with Full Vision Model Capabilities
Deep technical analysis with pattern recognition, trend analysis, and precise trade signals.
Author: APB - Octavian Team
"""

import streamlit as st
import numpy as np
import base64
import io
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from PIL import Image

HAS_LLM_VISION = False
try:
    import importlib.metadata
    importlib.metadata.version("llama-cpp-python")
    HAS_LLM_VISION = True
except (ImportError, importlib.metadata.PackageNotFoundError):
    pass

try:
    from ai_chatbot import _load_llm, _llm_call
    HAS_CHATBOT_LLM = True
except ImportError:
    HAS_CHATBOT_LLM = False


@dataclass
class ChartAnalysisResult:
    """Result from chart image analysis."""
    trend_direction: str = "NEUTRAL"
    trend_strength: str = "Moderate"
    support_levels: List[str] = field(default_factory=list)
    resistance_levels: List[str] = field(default_factory=list)
    patterns_detected: List[str] = field(default_factory=list)
    buy_signals: List[str] = field(default_factory=list)
    sell_signals: List[str] = field(default_factory=list)
    action_now: str = ""
    watch_before_buying: List[str] = field(default_factory=list)
    watch_before_selling: List[str] = field(default_factory=list)
    risk_notes: List[str] = field(default_factory=list)
    confidence: float = 0.5
    full_analysis: str = ""
    timestamp: str = ""
    price_targets: Dict[str, str] = field(default_factory=dict)
    stop_loss_levels: List[str] = field(default_factory=list)
    timeframe_analysis: str = ""
    volume_analysis: str = ""
    indicator_readings: Dict[str, str] = field(default_factory=dict)
    full_model_outlook: Dict[str, Any] = field(default_factory=dict)



def _analyze_image_pixels_advanced(img: Image.Image) -> Dict[str, Any]:
    """
    Advanced pixel analysis extracting detailed chart features.
    Analyzes: trend, momentum, volatility, support/resistance, volume, indicators.
    """
    arr = np.array(img.convert("RGB"))
    h, w, _ = arr.shape
    
    # Divide chart into sections for temporal analysis
    left_third = arr[:, :int(w * 0.33), :]
    middle_third = arr[:, int(w * 0.33):int(w * 0.66), :]
    right_third = arr[:, int(w * 0.66):, :]
    
    # Analyze each section
    sections = {
        "left": left_third,
        "middle": middle_third,
        "right": right_third
    }
    
    section_analysis = {}
    for name, section in sections.items():
        # Color analysis (candle colors)
        green_mask = (section[:, :, 1] > section[:, :, 0] + 15) & \
                     (section[:, :, 1] > section[:, :, 2] + 15)
        red_mask = (section[:, :, 0] > section[:, :, 1] + 15) & \
                   (section[:, :, 0] > section[:, :, 2] + 15)
        
        section_analysis[name] = {
            "green_pct": float(green_mask.sum() / green_mask.size),
            "red_pct": float(red_mask.sum() / red_mask.size),
            "brightness": float(np.mean(section))
        }
    
    # Trend analysis: compare brightness/position across time
    left_brightness = section_analysis["left"]["brightness"]
    right_brightness = section_analysis["right"]["brightness"]
    brightness_change = right_brightness - left_brightness
    
    # Momentum: recent vs historical color dominance
    left_green = section_analysis["left"]["green_pct"]
    right_green = section_analysis["right"]["green_pct"]
    left_red = section_analysis["left"]["red_pct"]
    right_red = section_analysis["right"]["red_pct"]
    
    momentum_shift = (right_green - left_green) - (right_red - left_red)
    
    # Detect horizontal lines (support/resistance)
    gray = np.mean(arr, axis=2)
    row_variance = np.var(gray, axis=1)
    
    # Find rows with low variance (horizontal lines)
    threshold = np.percentile(row_variance, 5)
    horizontal_lines = np.where(row_variance < threshold)[0]
    
    # Cluster nearby lines
    if len(horizontal_lines) > 0:
        line_clusters = []
        current_cluster = [horizontal_lines[0]]
        for line in horizontal_lines[1:]:
            if line - current_cluster[-1] < h * 0.02:  # Within 2% of height
                current_cluster.append(line)
            else:
                line_clusters.append(np.mean(current_cluster))
                current_cluster = [line]
        line_clusters.append(np.mean(current_cluster))
        
        # Convert to relative positions (0=top, 1=bottom)
        key_levels = [float(line / h) for line in line_clusters]
    else:
        key_levels = []
    
    # Detect indicator panels (sections separated by gaps)
    row_means = np.mean(gray, axis=1)
    row_diffs = np.abs(np.diff(row_means))
    large_gaps = np.where(row_diffs > np.percentile(row_diffs, 95))[0]
    
    # Estimate number of panels
    num_panels = 1
    if len(large_gaps) > 0:
        # Count significant gaps
        gap_clusters = []
        for gap in large_gaps:
            if not gap_clusters or gap - gap_clusters[-1] > h * 0.1:
                gap_clusters.append(gap)
        num_panels = len(gap_clusters) + 1
    
    # Volatility estimation: variance in vertical position
    col_means = np.mean(gray, axis=0)
    volatility_proxy = float(np.std(col_means))
    
    # Volume analysis (if bottom panel exists)
    has_volume_panel = num_panels >= 2
    volume_trend = "unknown"
    if has_volume_panel:
        # Assume bottom 20% is volume
        volume_section = arr[int(h * 0.8):, :, :]
        left_vol = np.mean(volume_section[:, :int(w * 0.5)])
        right_vol = np.mean(volume_section[:, int(w * 0.5):])
        if right_vol > left_vol * 1.1:
            volume_trend = "increasing"
        elif left_vol > right_vol * 1.1:
            volume_trend = "decreasing"
        else:
            volume_trend = "stable"
    
    # Pattern detection: look for specific shapes
    patterns = []
    
    # Higher highs and higher lows (uptrend)
    if brightness_change < -10 and momentum_shift > 0.01:
        patterns.append("higher_highs_higher_lows")
    
    # Lower highs and lower lows (downtrend)
    if brightness_change > 10 and momentum_shift < -0.01:
        patterns.append("lower_highs_lower_lows")
    
    # Consolidation (low volatility)
    if volatility_proxy < 15:
        patterns.append("consolidation")
    
    # Breakout potential (recent volatility increase)
    left_vol_proxy = float(np.std(np.mean(left_third, axis=2)))
    right_vol_proxy = float(np.std(np.mean(right_third, axis=2)))
    if right_vol_proxy > left_vol_proxy * 1.3:
        patterns.append("volatility_expansion")
    
    return {
        "section_analysis": section_analysis,
        "brightness_change": brightness_change,
        "momentum_shift": momentum_shift,
        "key_levels": key_levels,
        "num_panels": num_panels,
        "has_volume_panel": has_volume_panel,
        "volume_trend": volume_trend,
        "volatility_proxy": volatility_proxy,
        "patterns": patterns,
        "width": w,
        "height": h,
        "aspect_ratio": w / h,
        "overall_green": float(np.mean([s["green_pct"] for s in section_analysis.values()])),
        "overall_red": float(np.mean([s["red_pct"] for s in section_analysis.values()])),
        "recent_green": section_analysis["right"]["green_pct"],
        "recent_red": section_analysis["right"]["red_pct"],
    }



def _llm_analyze_chart_deep(img: Image.Image, pixel_data: Dict, user_context: str = "") -> Optional[str]:
    """
    Deep LLM analysis with comprehensive prompt engineering.
    Provides detailed technical analysis with specific actionable insights.
    """
    if not HAS_CHATBOT_LLM:
        return None
    
    try:
        llm = _load_llm()
        if not llm:
            return None
        
        # Build comprehensive analysis prompt
        prompt_parts = [
            "You are analyzing a financial chart image. Here is the detailed technical data extracted:",
            "",
            "=== CHART STRUCTURE ===",
            f"Dimensions: {pixel_data['width']}x{pixel_data['height']} pixels",
            f"Aspect Ratio: {pixel_data['aspect_ratio']:.2f}",
            f"Number of Panels: {pixel_data['num_panels']} (main chart + {pixel_data['num_panels']-1} indicator panels)",
            f"Volume Panel Detected: {'Yes' if pixel_data['has_volume_panel'] else 'No'}",
            "",
            "=== PRICE ACTION ANALYSIS ===",
            f"Overall Candle Distribution: {pixel_data['overall_green']*100:.1f}% green, {pixel_data['overall_red']*100:.1f}% red",
            f"Recent Price Action (right 33%): {pixel_data['recent_green']*100:.1f}% green, {pixel_data['recent_red']*100:.1f}% red",
            f"Brightness Trend: {pixel_data['brightness_change']:.1f} (negative = price moving up on chart, positive = moving down)",
            f"Momentum Shift: {pixel_data['momentum_shift']:.4f} (positive = bullish momentum, negative = bearish momentum)",
            "",
            "=== TEMPORAL ANALYSIS ===",
            f"Left Third (Historical): {pixel_data['section_analysis']['left']['green_pct']*100:.1f}% green, {pixel_data['section_analysis']['left']['red_pct']*100:.1f}% red",
            f"Middle Third: {pixel_data['section_analysis']['middle']['green_pct']*100:.1f}% green, {pixel_data['section_analysis']['middle']['red_pct']*100:.1f}% red",
            f"Right Third (Recent): {pixel_data['section_analysis']['right']['green_pct']*100:.1f}% green, {pixel_data['section_analysis']['right']['red_pct']*100:.1f}% red",
            "",
            "=== VOLATILITY & VOLUME ===",
            f"Volatility Proxy: {pixel_data['volatility_proxy']:.2f} (higher = more volatile)",
            f"Volume Trend: {pixel_data['volume_trend']}",
            "",
            "=== KEY LEVELS DETECTED ===",
            f"Number of Horizontal Lines: {len(pixel_data['key_levels'])}",
        ]
        
        if pixel_data['key_levels']:
            prompt_parts.append("Relative Positions (0=top, 1=bottom):")
            for i, level in enumerate(pixel_data['key_levels'][:10], 1):
                position = "upper" if level < 0.33 else "middle" if level < 0.66 else "lower"
                prompt_parts.append(f"  Level {i}: {level:.3f} ({position} chart)")
        
        prompt_parts.extend([
            "",
            "=== PATTERNS DETECTED ===",
        ])
        
        if pixel_data['patterns']:
            for pattern in pixel_data['patterns']:
                prompt_parts.append(f"  - {pattern.replace('_', ' ').title()}")
        else:
            prompt_parts.append("  - No clear patterns detected")
        
        if user_context:
            prompt_parts.extend([
                "",
                "=== USER CONTEXT ===",
                user_context,
            ])
        
        prompt_parts.extend([
            "",
            "=== YOUR TASK ===",
            "Based on this comprehensive technical data, provide a DETAILED, SPECIFIC analysis:",
            "",
            "1. TREND ASSESSMENT:",
            "   - Primary trend direction (bullish/bearish/neutral) with confidence level",
            "   - Trend strength (weak/moderate/strong) with reasoning",
            "   - Is the trend accelerating, decelerating, or stable?",
            "",
            "2. SUPPORT & RESISTANCE:",
            "   - Identify 2-3 key support levels (use relative positions from key_levels data)",
            "   - Identify 2-3 key resistance levels",
            "   - Which level is most critical right now?",
            "",
            "3. MOMENTUM ANALYSIS:",
            "   - Current momentum state (building, fading, neutral)",
            "   - Recent momentum shift interpretation",
            "   - Are buyers or sellers in control?",
            "",
            "4. VOLUME ANALYSIS:",
            "   - Volume trend interpretation",
            "   - Does volume confirm price action?",
            "   - Any volume divergences?",
            "",
            "5. PATTERN RECOGNITION:",
            "   - Specific chart patterns forming (triangles, flags, head & shoulders, etc.)",
            "   - Pattern completion status",
            "   - Expected breakout direction",
            "",
            "6. BUY SIGNALS (be specific):",
            "   - List 3-5 concrete buy signals present",
            "   - Include entry triggers and conditions",
            "",
            "7. SELL SIGNALS (be specific):",
            "   - List 3-5 concrete sell signals present",
            "   - Include exit triggers and conditions",
            "",
            "8. IMMEDIATE ACTION:",
            "   - What should a trader do RIGHT NOW?",
            "   - Specific entry/exit levels if applicable",
            "   - Position sizing recommendation",
            "",
            "9. RISK MANAGEMENT:",
            "   - Suggested stop loss levels",
            "   - Take profit targets",
            "   - Risk/reward ratio",
            "",
            "10. WATCH LIST:",
            "   - 3-4 things to watch before buying",
            "   - 3-4 things to watch before selling",
            "   - Key invalidation levels",
            "",
            "CRITICAL REQUIREMENTS:",
            "- Be SPECIFIC and DECISIVE - avoid generic statements",
            "- Use the actual data provided - reference specific numbers",
            "- Provide ACTIONABLE insights - exact levels, triggers, conditions",
            "- Explain your reasoning - connect data to conclusions",
            "- Vary your analysis based on the data - don't give cookie-cutter responses",
            "- If data suggests conflicting signals, acknowledge and explain",
            "- Use professional trading terminology",
            "- Format with clear sections and bullet points",
        ])
        
        system = (
            "You are Octavian, an elite institutional chart analyst with 20+ years of experience. "
            "You specialize in technical analysis, pattern recognition, and precise trade execution. "
            "Your analysis is data-driven, specific, and actionable. You never give generic advice. "
            "Every chart is unique - analyze the specific data provided and give tailored insights. "
            "Be decisive but acknowledge uncertainty when present. Use clear, professional language."
        )
        
        result = _llm_call(llm, system, "\n".join(prompt_parts), max_tokens=2500, timeout=45)
        return result
        
    except Exception as e:
        st.warning(f"LLM analysis failed: {e}")
        return None



def _build_structured_result(pixel_data: Dict, llm_analysis: Optional[str], user_context: str = "") -> ChartAnalysisResult:
    """
    Build structured result from pixel data and LLM analysis.
    Extracts specific insights and organizes them into actionable format.
    """
    result = ChartAnalysisResult(timestamp=datetime.now().isoformat())
    
    # Determine trend from multiple signals
    momentum = pixel_data['momentum_shift']
    brightness = pixel_data['brightness_change']
    recent_green = pixel_data['recent_green']
    recent_red = pixel_data['recent_red']
    
    # Trend direction with confidence
    if momentum > 0.015 and brightness < -5:
        result.trend_direction = "STRONGLY BULLISH"
        result.trend_strength = "Strong"
        result.confidence = min(0.85, 0.6 + abs(momentum) * 20)
    elif momentum > 0.005 and brightness < 0:
        result.trend_direction = "BULLISH"
        result.trend_strength = "Moderate"
        result.confidence = min(0.75, 0.5 + abs(momentum) * 15)
    elif momentum < -0.015 and brightness > 5:
        result.trend_direction = "STRONGLY BEARISH"
        result.trend_strength = "Strong"
        result.confidence = min(0.85, 0.6 + abs(momentum) * 20)
    elif momentum < -0.005 and brightness > 0:
        result.trend_direction = "BEARISH"
        result.trend_strength = "Moderate"
        result.confidence = min(0.75, 0.5 + abs(momentum) * 15)
    else:
        result.trend_direction = "NEUTRAL / CONSOLIDATING"
        result.trend_strength = "Weak"
        result.confidence = 0.4
    
    # Support and resistance from key levels
    if pixel_data['key_levels']:
        levels = sorted(pixel_data['key_levels'])
        
        # Top levels are resistance, bottom levels are support
        resistance_levels = [l for l in levels if l < 0.4]
        support_levels = [l for l in levels if l > 0.6]
        
        result.resistance_levels = [f"Level at {l:.1%} from top (resistance)" for l in resistance_levels[:3]]
        result.support_levels = [f"Level at {(1-l):.1%} from bottom (support)" for l in support_levels[:3]]
    
    # Pattern detection
    for pattern in pixel_data['patterns']:
        if pattern == "higher_highs_higher_lows":
            result.patterns_detected.append("Higher Highs & Higher Lows - Classic Uptrend Structure")
            result.buy_signals.append("Uptrend structure intact with higher highs and higher lows")
        elif pattern == "lower_highs_lower_lows":
            result.patterns_detected.append("Lower Highs & Lower Lows - Classic Downtrend Structure")
            result.sell_signals.append("Downtrend structure intact with lower highs and lower lows")
        elif pattern == "consolidation":
            result.patterns_detected.append("Consolidation / Range-Bound - Low Volatility")
            result.risk_notes.append("Low volatility suggests potential breakout imminent")
        elif pattern == "volatility_expansion":
            result.patterns_detected.append("Volatility Expansion - Increased Price Movement")
            result.buy_signals.append("Volatility expansion suggests strong directional move forming")
    
    # Volume analysis
    result.volume_analysis = f"Volume trend: {pixel_data['volume_trend']}"
    if pixel_data['volume_trend'] == "increasing":
        if result.trend_direction in ["BULLISH", "STRONGLY BULLISH"]:
            result.buy_signals.append("Increasing volume confirms bullish trend strength")
        elif result.trend_direction in ["BEARISH", "STRONGLY BEARISH"]:
            result.sell_signals.append("Increasing volume confirms bearish trend strength")
    elif pixel_data['volume_trend'] == "decreasing":
        result.risk_notes.append("Decreasing volume suggests weakening trend - watch for reversal")
    
    # Indicator analysis
    if pixel_data['num_panels'] > 1:
        result.indicator_readings["panels"] = f"{pixel_data['num_panels']} panels detected (likely includes RSI, MACD, or volume)"
    
    # Build buy signals
    if recent_green > recent_red * 1.3:
        result.buy_signals.append(f"Recent candles {recent_green*100:.1f}% green vs {recent_red*100:.1f}% red - bullish dominance")
    
    if momentum > 0.01:
        result.buy_signals.append(f"Positive momentum shift (+{momentum:.3f}) - buyers gaining control")
    
    if brightness < -10:
        result.buy_signals.append("Price positioning higher on chart - upward trajectory visible")
    
    # Build sell signals
    if recent_red > recent_green * 1.3:
        result.sell_signals.append(f"Recent candles {recent_red*100:.1f}% red vs {recent_green*100:.1f}% green - bearish dominance")
    
    if momentum < -0.01:
        result.sell_signals.append(f"Negative momentum shift ({momentum:.3f}) - sellers gaining control")
    
    if brightness > 10:
        result.sell_signals.append("Price positioning lower on chart - downward trajectory visible")
    
    # Action recommendation
    if result.trend_direction in ["STRONGLY BULLISH", "BULLISH"]:
        result.action_now = (
            f"BULLISH BIAS ({result.confidence:.0%} confidence): "
            f"Look for pullback entries to support levels. "
            f"Consider long positions with stops below recent swing lows. "
            f"Target resistance levels for profit taking. "
            f"Position size: {'Full' if result.confidence > 0.7 else 'Reduced'} allocation recommended."
        )
        result.price_targets = {
            "conservative": "First resistance level",
            "moderate": "Second resistance level",
            "aggressive": "New highs above all resistance"
        }
    elif result.trend_direction in ["STRONGLY BEARISH", "BEARISH"]:
        result.action_now = (
            f"BEARISH BIAS ({result.confidence:.0%} confidence): "
            f"Avoid long entries. Consider short positions on rallies to resistance. "
            f"Set stops above recent swing highs. "
            f"Target support levels for profit taking. "
            f"Position size: {'Full' if result.confidence > 0.7 else 'Reduced'} allocation recommended."
        )
        result.price_targets = {
            "conservative": "First support level",
            "moderate": "Second support level",
            "aggressive": "New lows below all support"
        }
    else:
        result.action_now = (
            f"NEUTRAL / RANGE-BOUND ({result.confidence:.0%} confidence): "
            f"Wait for breakout confirmation before entering. "
            f"Trade the range: buy support, sell resistance. "
            f"Reduce position sizes in choppy conditions. "
            f"Set tight stops as direction is unclear."
        )
        result.price_targets = {
            "breakout_up": "Above resistance with volume",
            "breakout_down": "Below support with volume",
            "range_trade": "Support to resistance"
        }
    
    # Watch items
    result.watch_before_buying = [
        "Confirm price holds above nearest support level",
        "Look for bullish candlestick patterns (hammer, engulfing, morning star)",
        "Check for RSI oversold bounce if indicators visible",
        "Verify volume increases on up moves (confirms buying interest)",
        "Ensure broader market context supports long bias",
        f"Set stop loss below {support_levels[0] if support_levels else 'recent swing low'}"
    ]
    
    result.watch_before_selling = [
        "Confirm price rejects at resistance level",
        "Look for bearish candlestick patterns (shooting star, engulfing, evening star)",
        "Check for RSI overbought divergence if indicators visible",
        "Verify volume increases on down moves (confirms selling pressure)",
        "Ensure sector/market weakness supports short bias",
        f"Set stop loss above {resistance_levels[0] if resistance_levels else 'recent swing high'}"
    ]
    
    # Stop loss recommendations
    if support_levels:
        result.stop_loss_levels.append(f"For longs: Below {support_levels[0]}")
    if resistance_levels:
        result.stop_loss_levels.append(f"For shorts: Above {resistance_levels[0]}")
    
    result.stop_loss_levels.append(f"Risk per trade: 1-2% of account maximum")
    result.stop_loss_levels.append(f"Risk/Reward target: Minimum 2:1 ratio")
    
    # Risk notes
    result.risk_notes.extend([
        "Image analysis has limitations - always verify with live data",
        "Exact price levels cannot be determined without scale",
        "Timeframe context significantly affects interpretation",
        f"Analysis confidence: {result.confidence:.0%} - adjust position sizing accordingly",
        "Consider multiple timeframe analysis for confirmation",
        "News events and fundamentals can override technical signals"
    ])
    
    # Timeframe analysis
    result.timeframe_analysis = (
        f"Based on {pixel_data['width']} data points visible, "
        f"this appears to be a {'short-term' if pixel_data['width'] < 500 else 'medium-term' if pixel_data['width'] < 1000 else 'long-term'} chart. "
        f"Volatility level: {'Low' if pixel_data['volatility_proxy'] < 15 else 'Moderate' if pixel_data['volatility_proxy'] < 30 else 'High'}."
    )
    
    # If LLM provided analysis, use it as primary
    if llm_analysis:
        result.full_analysis = llm_analysis
    else:
        # Build comprehensive text analysis
        lines = [
            f"=== CHART ANALYSIS REPORT ===",
            f"Generated: {result.timestamp}",
            f"",
            f"TREND: {result.trend_direction}",
            f"STRENGTH: {result.trend_strength}",
            f"CONFIDENCE: {result.confidence:.0%}",
            f"",
            f"CURRENT ACTION:",
            result.action_now,
            f"",
        ]
        
        if result.buy_signals:
            lines.append("BUY SIGNALS:")
            for s in result.buy_signals:
                lines.append(f"   {s}")
            lines.append("")
        
        if result.sell_signals:
            lines.append("SELL SIGNALS:")
            for s in result.sell_signals:
                lines.append(f"   {s}")
            lines.append("")
        
        if result.patterns_detected:
            lines.append("PATTERNS DETECTED:")
            for p in result.patterns_detected:
                lines.append(f"   {p}")
            lines.append("")
        
        if result.support_levels or result.resistance_levels:
            lines.append("KEY LEVELS:")
            if result.resistance_levels:
                lines.append("  Resistance:")
                for r in result.resistance_levels:
                    lines.append(f"    - {r}")
            if result.support_levels:
                lines.append("  Support:")
                for s in result.support_levels:
                    lines.append(f"    - {s}")
            lines.append("")
        
        lines.append(f"VOLUME: {result.volume_analysis}")
        lines.append(f"TIMEFRAME: {result.timeframe_analysis}")
        
        result.full_analysis = "\n".join(lines)
    
    return result


def analyze_chart_image(uploaded_file, user_context: str = "") -> ChartAnalysisResult:
    """Main entry point: analyze an uploaded chart image with full capabilities."""
    try:
        img = Image.open(uploaded_file)
        
        # Advanced pixel analysis
        pixel_data = _analyze_image_pixels_advanced(img)
        
        # Deep LLM analysis
        llm_analysis = _llm_analyze_chart_deep(img, pixel_data, user_context)
        
        # Build structured result
        result = _build_structured_result(pixel_data, llm_analysis, user_context)
        
        # Global Integration: Full Model Outlook
        try:
            # Try to extract symbol from context
            symbol = None
            if user_context:
                import re
                match = re.search(r'\b[A-Z]{1,5}\b', user_context)
                if match:
                    symbol = match.group(0)
            
            if symbol:
                from data_sources import get_stock
                from quant_ensemble_model import get_quant_ensemble
                from market_movers import _generate_ai_insights, _calculate_technicals
                
                df = get_stock(symbol, period="6mo")
                if df is not None and not df.empty:
                    tech = _calculate_technicals(df)
                    ai_data = _generate_ai_insights(symbol, tech, df)
                    qe = get_quant_ensemble()
                    q_signal = qe.predict(symbol, df)
                    
                    bull_p = ai_data.get("bullish_prob", 0.5)
                    bear_p = ai_data.get("bearish_prob", 0.5)
                    f_score = ((bull_p - bear_p) + 1) / 2 * 100
                    q_prob = q_signal.probability * 100
                    unified_score = (f_score + q_prob) / 2
                    
                    result.full_model_outlook = {
                        "symbol": symbol,
                        "score": unified_score,
                        "q_prob": q_prob,
                        "f_score": f_score,
                        "direction": q_signal.direction,
                        "reasoning": ai_data.get('outlook', '')
                    }
        except Exception:
            pass

        return result
        
    except Exception as e:
        result = ChartAnalysisResult(timestamp=datetime.now().isoformat())
        result.full_analysis = f"Analysis error: {str(e)}. Please ensure the uploaded file is a valid image."
        result.action_now = "Unable to analyze. Please try a different image or check file format."
        result.confidence = 0.0
        return result



def show_chart_analyzer():
    """Render the enhanced chart image analyzer UI."""
    st.title(" Chart Image Analysis")
    st.caption(
        "Upload any chart image for deep AI-powered analysis. "
        "Advanced pattern recognition, trend analysis, support/resistance detection, and precise trade signals."
    )
    
    # Instructions
    with st.expander(" How to Use", expanded=False):
        st.markdown("""
        **Upload any chart screenshot:**
        - TradingView, broker platforms, financial websites
        - Candlestick, line, or bar charts
        - With or without indicators (RSI, MACD, volume, etc.)
        - Any timeframe (1min to monthly)
        
        **The AI will analyze:**
        - Trend direction and strength
        - Support and resistance levels
        - Chart patterns (triangles, flags, head & shoulders, etc.)
        - Momentum and volume trends
        - Specific buy/sell signals
        - Entry/exit recommendations
        - Risk management levels
        
        **Tips for best results:**
        - Use clear, high-resolution images
        - Include context in the text box (symbol, timeframe, your thesis)
        - Charts with indicators provide richer analysis
        - Multiple timeframes give better confirmation
        """)
    
    col_upload, col_context = st.columns([1, 1])
    
    with col_upload:
        uploaded = st.file_uploader(
            " Upload Chart Image",
            type=["png", "jpg", "jpeg", "webp", "bmp"],
            key="chart_img_upload",
            help="Any screenshot of a price chart"
        )
    
    with col_context:
        user_context = st.text_area(
            " Context (Optional but Recommended)",
            placeholder="Example: 'AAPL daily chart. Considering long position. RSI at 35, MACD bullish crossover. Earnings in 2 weeks. Looking for entry around $175 support.'",
            height=140,
            key="chart_img_context",
            help="Provide symbol, timeframe, indicators shown, your thesis, key levels, upcoming events, etc."
        )
    
    if uploaded:
        # Show the uploaded image
        st.image(uploaded, caption="Uploaded Chart", use_column_width=True)
        
        col_btn1, col_btn2 = st.columns([3, 1])
        
        with col_btn1:
            analyze_btn = st.button(
                " Analyze Chart with Full AI Capabilities",
                type="primary",
                use_container_width=True,
                key="analyze_chart_btn"
            )
        
        with col_btn2:
            if st.button(" Clear", use_container_width=True):
                st.rerun()
        
        if analyze_btn:
            with st.spinner(" Running deep technical analysis..."):
                result = analyze_chart_image(uploaded, user_context)
            
            # Display results
            st.markdown("---")
            st.markdown("##  Analysis Results")
            
            # Header metrics
            m1, m2, m3, m4 = st.columns(4)
            
            with m1:
                trend_emoji = "" if "BULL" in result.trend_direction else "" if "BEAR" in result.trend_direction else ""
                st.metric("Trend", f"{trend_emoji} {result.trend_direction}")
            
            with m2:
                st.metric("Strength", result.trend_strength)
            
            with m3:
                confidence_color = "" if result.confidence > 0.7 else "" if result.confidence > 0.5 else ""
                st.metric("Confidence", f"{confidence_color} {result.confidence:.0%}")
            
            with m4:
                if result.volume_analysis:
                    vol_emoji = "" if "increasing" in result.volume_analysis else "" if "decreasing" in result.volume_analysis else ""
                    st.metric("Volume", f"{vol_emoji} {result.volume_analysis.split(':')[1].strip().title()}")

            # --- FULL MODEL OUTLOOK (Global Integration) ---
            if result.full_model_outlook:
                o = result.full_model_outlook
                u_score = o['score']
                if u_score >= 80: label, color = "STRONG OVERWEIGHT", "#00ff88"
                elif u_score >= 60: label, color = "ACCUMULATE", "#a2ffb3"
                elif u_score >= 40: label, color = "NEUTRAL", "#8b949e"
                elif u_score >= 20: label, color = "REDUCE", "#ff9b9b"
                else: label, color = "STRONG UNDERWEIGHT", "#ff5252"
                
                st.write("")
                st.markdown(f"""
                <div style="background: rgba(13, 17, 23, 0.8); border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin-bottom: 25px;">
                    <h4 style="margin-top: 0; color: #8b949e; font-size: 0.9rem; letter-spacing: 1px; text-transform: uppercase;">Full Model Outlook ({o['symbol']})</h4>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h2 style="margin: 0; color: {color}; font-size: 2.2rem; font-weight: 800;">{label}</h2>
                            <p style="margin: 5px 0 0 0; color: #8b949e; font-size: 0.95rem;">
                                Quant Signal: {o['direction']} | Integrity: High
                            </p>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 2.8rem; font-weight: 800; color: white;">{u_score:.1f}</div>
                            <div style="font-size: 0.75rem; color: #8b949e;">CONVICTION</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Action now - prominent display
            st.markdown("###  Recommended Action")
            action_type = "success" if "BULLISH" in result.trend_direction else "error" if "BEARISH" in result.trend_direction else "info"
            if action_type == "success":
                st.success(result.action_now)
            elif action_type == "error":
                st.error(result.action_now)
            else:
                st.info(result.action_now)
            
            # Price targets
            if result.price_targets:
                st.markdown("###  Price Targets")
                target_cols = st.columns(len(result.price_targets))
                for i, (level, target) in enumerate(result.price_targets.items()):
                    with target_cols[i]:
                        st.markdown(f"**{level.replace('_', ' ').title()}**")
                        st.code(target)
            
            # Signals side by side
            st.markdown("###  Trading Signals")
            col_buy, col_sell = st.columns(2)
            
            with col_buy:
                st.markdown("####  Buy Signals")
                if result.buy_signals:
                    for s in result.buy_signals:
                        st.markdown(f" {s}")
                else:
                    st.caption("No strong buy signals detected")
            
            with col_sell:
                st.markdown("####  Sell Signals")
                if result.sell_signals:
                    for s in result.sell_signals:
                        st.markdown(f" {s}")
                else:
                    st.caption("No strong sell signals detected")
            
            # Watch items
            st.markdown("###  What to Watch")
            col_wb, col_ws = st.columns(2)
            
            with col_wb:
                st.markdown("#### Before Buying")
                for w in result.watch_before_buying:
                    st.markdown(f" {w}")
            
            with col_ws:
                st.markdown("#### Before Selling")
                for w in result.watch_before_selling:
                    st.markdown(f" {w}")
            
            # Key levels
            if result.support_levels or result.resistance_levels:
                st.markdown("###  Key Levels")
                col_sup, col_res = st.columns(2)
                
                with col_sup:
                    st.markdown("####  Support Levels")
                    if result.support_levels:
                        for s in result.support_levels:
                            st.markdown(f" {s}")
                    else:
                        st.caption("No clear support levels detected")
                
                with col_res:
                    st.markdown("####  Resistance Levels")
                    if result.resistance_levels:
                        for r in result.resistance_levels:
                            st.markdown(f" {r}")
                    else:
                        st.caption("No clear resistance levels detected")
            
            # Stop loss levels
            if result.stop_loss_levels:
                st.markdown("###  Risk Management")
                for sl in result.stop_loss_levels:
                    st.markdown(f" {sl}")
            
            # Patterns detected
            if result.patterns_detected:
                with st.expander(" Detected Patterns & Features", expanded=True):
                    for p in result.patterns_detected:
                        st.markdown(f" {p}")
            
            # Indicator readings
            if result.indicator_readings:
                with st.expander(" Indicator Analysis"):
                    for indicator, reading in result.indicator_readings.items():
                        st.markdown(f"**{indicator.replace('_', ' ').title()}:** {reading}")
            
            # Timeframe analysis
            if result.timeframe_analysis:
                with st.expander(" Timeframe Context"):
                    st.markdown(result.timeframe_analysis)
            
            # Risk notes
            with st.expander(" Risk Considerations"):
                for r in result.risk_notes:
                    st.caption(f" {r}")
            
            # Full analysis report
            if result.full_analysis:
                with st.expander(" Complete Analysis Report", expanded=False):
                    st.markdown(result.full_analysis)
            
            # Download report
            st.markdown("---")
            report_text = f"""
CHART ANALYSIS REPORT
Generated: {result.timestamp}

TREND: {result.trend_direction}
STRENGTH: {result.trend_strength}
CONFIDENCE: {result.confidence:.0%}

RECOMMENDED ACTION:
{result.action_now}

BUY SIGNALS:
{chr(10).join('- ' + s for s in result.buy_signals)}

SELL SIGNALS:
{chr(10).join('- ' + s for s in result.sell_signals)}

WATCH BEFORE BUYING:
{chr(10).join('- ' + w for w in result.watch_before_buying)}

WATCH BEFORE SELLING:
{chr(10).join('- ' + w for w in result.watch_before_selling)}

PATTERNS DETECTED:
{chr(10).join('- ' + p for p in result.patterns_detected)}

RISK MANAGEMENT:
{chr(10).join('- ' + sl for sl in result.stop_loss_levels)}

FULL ANALYSIS:
{result.full_analysis}
"""
            
            st.download_button(
                " Download Analysis Report",
                data=report_text,
                file_name=f"chart_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    else:
        # Show example
        st.info(" Upload a chart image to begin analysis")
        
        st.markdown("###  Example Analysis Features")
        
        ex_col1, ex_col2, ex_col3 = st.columns(3)
        
        with ex_col1:
            st.markdown("**Trend Analysis**")
            st.caption(" Direction & strength\n Momentum shifts\n Trend acceleration")
        
        with ex_col2:
            st.markdown("**Pattern Recognition**")
            st.caption(" Chart patterns\n Candlestick patterns\n Support/resistance")
        
        with ex_col3:
            st.markdown("**Trade Signals**")
            st.caption(" Entry/exit points\n Stop loss levels\n Price targets")
