# AI Response Formatting Guide

## Overview

The AI response formatting system separates user-facing text from detailed technical analysis, providing a clean user experience while maintaining full transparency through an expandable "Show AI Reasoning"feature.

## Architecture

### Response Structure

Every AI response now contains:

```python
{    'text': str, # Clean, user-facing text
    'raw_analysis': dict, # Hidden detailed analysis
    'show_reasoning_available': bool, # Whether to show reasoning button
    'response_metadata': dict, # Additional metadata
    'charts': list, # Chart data
    'intents': list, # Detected intents
    'symbols': list, # Analyzed symbols
    # ... other fields
}
```
### User-Facing Text Format

The `text`field contains:
- Clear, complete sentences
- Direct answers to user questions
- Natural language confidence levels
- Key factors in plain English
- Actionable insights

Example:
```Based on my analysis, AAPL is showing bullish characteristics with upward momentum in the swing trading timeframe. I have high confidence in this assessment (85%).

Key factors driving this view: strong momentum indicators, positive earnings sentiment, technical breakout above resistance.

Current price is $175.50. Key support at $170.00, resistance at $180.00.
```
### Raw Analysis Structure

The `raw_analysis`field contains:

```python
{    'symbol_analyses': {        'AAPL': {            'prediction': {...},
            'technical_indicators': {...},
            'ml_scores': {...},
            'sentiment_data': {...},
            'risk_metrics': {...},
            'signal_factors': [...]
}
    },
    'market_context': {...},
    'intents_detected': [...],
    'timeframe_context': 'swing_trading',
    'analysis_timestamp': '2026-03-04T...',
    'detailed_breakdown': {...}
}
```
## UI Integration

### Display User Text

```python
# In your UI component
response = chatbot.process_enhanced_query(user_query)

# Display clean text
st.markdown(response['text'])
```
### Add "Show AI Reasoning"Button

```python
# Check if reasoning is available
if response.get('show_reasoning_available', False):
    with st.expander("Show AI Reasoning"):
        # Display raw analysis
        raw = response.get('raw_analysis', {})
        
        # Format for display
        st.json(raw) # Or create custom formatting
        
        # Show detailed breakdown
        if 'detailed_breakdown'in raw:
            for symbol, details in raw['detailed_breakdown'].items():
                st.subheader(f"{symbol} Analysis")
                
                # Technical indicators
                if 'technical_indicators'in details:
                    st.write("Technical Indicators:")
                    st.json(details['technical_indicators'])
                
                # ML scores
                if 'ml_scores'in details:
                    st.write("ML Model Scores:")
                    st.json(details['ml_scores'])
                
                # Signal factors
                if 'signal_factors'in details:
                    st.write("Signal Factors:")
                    for factor in details['signal_factors']:
                        st.write(f"- {factor}")
```
### Example UI Layout

```┌─────────────────────────────────────────┐
│ User Query: "Is AAPL bullish?"│
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Based on my analysis, AAPL is showing │
│ bullish characteristics with upward │
│ momentum in the swing trading timeframe.│
│ I have high confidence (85%). │
│ │
│ Key factors: strong momentum, positive │
│ earnings sentiment, technical breakout. │
│ │
│ Current price: $175.50 │
│ Support: $170.00 | Resistance: $180.00 │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Show AI Reasoning ▼ │
│ │
│ Symbol Analyses: │
│ AAPL: │
│ Prediction: │
│ signal: BULLISH │
│ confidence: 0.85 │
│ bullish_prob: 0.78 │
│ Technical Indicators: │
│ rsi: 62.5 │
│ macd: 1.25 │
│ ema20: 173.50 │
│ ML Scores: │
│ random_forest: 0.82 │
│ gradient_boost: 0.88 │
│ Signal Factors: │
│ - Momentum above threshold │
│ - Volume surge detected │
│ - Breakout confirmed │
└─────────────────────────────────────────┘
```
## Response Types

### Single Symbol Analysis

```python
formatted = formatter.format_analysis_response(    query="Is AAPL bullish?",
    symbol_analyses={'AAPL': {...}},
    intents=['prediction', 'analysis'],
    market_context={...},
    timeframe='swing_trading')
```
Output:
- Direct answer about the symbol
- Confidence level
- Key factors (top 3)
- Price levels if available

### Multiple Symbol Analysis

```python
formatted = formatter.format_analysis_response(    query="Compare AAPL, MSFT, GOOGL",
    symbol_analyses={'AAPL': {...}, 'MSFT': {...}, 'GOOGL': {...}},
    intents=['analysis', 'comparison'],
    market_context={...},
    timeframe='swing_trading')
```
Output:
- Summary of all symbols analyzed
- Bullish opportunities (sorted by confidence)
- Bearish signals (sorted by confidence)
- Neutral/mixed signals

### Market Scan

```python
formatted = formatter.format_scan_response(    opportunities=[...], # List of opportunities
    timeframe='swing_trading')
```
Output:
- Number of opportunities found
- Top 5 ranked by profit potential
- Direction (Long/Short)
- Probability, confidence, expected return
- Note about unbiased ranking

### Error Response

```python
formatted = formatter.format_error_response(    error_message="Timeout fetching data")
```
Output:
- User-friendly error message
- Suggestions for resolution
- No technical details exposed

## Customization

### Adjust Confidence Thresholds

In `response_formatter.py`:

```python
# High confidence threshold
if confidence > 0.75: # Change to 0.80 for stricter
    text += f"I have high confidence..."
# Moderate confidence threshold
elif confidence > 0.6: # Change to 0.65 for stricter
    text += f"I have moderate confidence..."```
### Customize Text Templates

```python
# In ResponseFormatter class
def _format_single_symbol_response(self, symbol, analysis, timeframe):
    # Customize opening statement
    text = f"My analysis of {symbol} shows..."    
    # Customize confidence expression
    if confidence > 0.75:
        text += f"with strong conviction ({confidence:.0%})"    
    # Add your custom sections
    text += self._add_custom_section(analysis)
    
    return text
```
### Add Custom Sections

```python
def _add_risk_warning(self, analysis):
    """Add custom risk warning section."""    risk_level = analysis.get('risk_level', 'medium')
    
    if risk_level == 'high':
        return "\n\n High risk: Consider smaller position sizes."    elif risk_level == 'medium':
        return "\n\nModerate risk: Standard position sizing recommended."    else:
        return "\n\nLower risk: Suitable for larger positions."```
## Testing

### Test Single Symbol

```python
chatbot = OctavianEnhancedChatbot()
response = await chatbot.process_enhanced_query("Is AAPL bullish?")

assert 'text'in response
assert 'raw_analysis'in response
assert response['show_reasoning_available'] == True
assert len(response['text']) > 0
print(response['text'])
```
### Test Multiple Symbols

```python
response = await chatbot.process_enhanced_query("Compare AAPL MSFT GOOGL")

assert len(response['symbols']) == 3
assert 'Bullish opportunities'in response['text'] or 'Bearish signals'in response['text']
```
### Test Market Scan

```python
response = await chatbot.process_unbiased_query("Find best profit opportunities")

assert 'opportunities'in response['raw_analysis']
assert 'profit probability'in response['text'].lower()
```
## Best Practices

1. **Keep user text concise** - Aim for 3-5 sentences for single symbol, 10-15 for multiple
2. **Use natural language** - Avoid technical jargon in user text
3. **Preserve all data** - Store complete analysis in raw_analysis
4. **Be consistent** - Use same format across all response types
5. **Handle errors gracefully** - Always provide user-friendly error messages
6. **Test thoroughly** - Verify both user text and raw analysis are correct

## Next Steps

To complete the UI integration:

1. Update `main.py`or dashboard to display `response['text']`2. Add expandable section for `response['raw_analysis']`3. Style the "Show AI Reasoning"button
4. Add copy/export functionality for raw analysis
5. Test with various query types
6. Gather user feedback on clarity

## Files Modified

- `ai_chatbot.py`- Integrated response formatter
- `response_formatter.py`- Core formatting logic (new file)
- `IMPLEMENTATION_STATUS.md`- Updated progress

## Related Features

- Paper Trading System - Uses same response structure for trade reasoning
- Automated Trading - Will use formatter for trade explanations
- Trade Transparency UI - Will display formatted reasoning

## Runtime Compatibility Fixes (Critical)

If you see these errors:

- `Error: 'AltDataSignal'object has no attribute 'signal_type'`- `Error: AdvancedBacktester.init() got an unexpected keyword argument 'symbol'`
apply the following compatibility layer.

### 1) AltDataSignal compatibility

Some components use `signal_type`, others use `type`/`signal`/`category`.

```python
# Add once during startup/import
def patch_alt_data_signal(AltDataSignal):
    if hasattr(AltDataSignal, "signal_type"):
        return

    def _get_signal_type(self):
        for attr in ("type", "signal", "category", "event_type", "label"):
            if hasattr(self, attr):
                val = getattr(self, attr)
                if val is not None:
                    return val
        return "unknown"
    def _set_signal_type(self, value):
        for attr in ("type", "signal", "category", "event_type", "label"):
            if hasattr(self, attr):
                setattr(self, attr, value)
                return
        setattr(self, "_signal_type_compat", value)

    AltDataSignal.signal_type = property(_get_signal_type, _set_signal_type)
```
### 2) AdvancedBacktester.init compatibility

Some callers pass `symbol=...`while engine expects `ticker`/`asset`/`instrument`.

```python
import inspect
from functools import wraps

def patch_advanced_backtester_init(AdvancedBacktester):
    init_fn = getattr(AdvancedBacktester, "init", None)
    if init_fn is None or getattr(init_fn, "_symbol_compat", False):
        return

    params = inspect.signature(init_fn).parameters

    @wraps(init_fn)
    def _wrapped(self, *args, **kwargs):
        if "symbol"in kwargs:
            sym = kwargs.pop("symbol")
            if "ticker"in params and "ticker"not in kwargs:
                kwargs["ticker"] = sym
            elif "asset"in params and "asset"not in kwargs:
                kwargs["asset"] = sym
            elif "instrument"in params and "instrument"not in kwargs:
                kwargs["instrument"] = sym
        return init_fn(self, *args, **kwargs)

    _wrapped._symbol_compat = True
    AdvancedBacktester.init = _wrapped
```
### 3) Apply patches at startup

```python
def apply_runtime_compatibility():
    # import your actual modules/classes here
    from advanced_backtester import AdvancedBacktester
    from alternative_data_engine import AltDataSignal

    patch_alt_data_signal(AltDataSignal)
    patch_advanced_backtester_init(AdvancedBacktester)
```
### 4) Validation checks

Run these after startup:

```python
assert hasattr(AltDataSignal, "signal_type")
# should not raise now:
# backtester.init(symbol="AAPL", ...)
```
### Notes

- These are **non-destructive** compatibility shims.
- Keep until all callers use a single canonical field/argument naming standard.
- Safe for institutional workflows where multiple subsystems evolve independently.

## No-Response Bug Fix (Processed but Empty Output)

If logs show:

- `Octavian analysis complete! Processed 1 symbol ...`- but UI displays no actual answer text,

the likely issue is an empty or missing `response["text"]`after analysis/formatting.

### Required Response Contract (Always Enforce)

Every chatbot path should return:

```python
response["text"] = non_empty_string
response["raw_analysis"] = dict
response["show_reasoning_available"] = True/False
```
Add a final defensive fallback before returning:

```python
def ensure_response_text(response: dict) -> dict:
    text = (response or {}).get("text", "")
    if isinstance(text, str) and text.strip():
        return response

    raw = (response or {}).get("raw_analysis", {}) or {}
    symbols = (response or {}).get("symbols", []) or list((raw.get("symbol_analyses") or {}).keys())

    if symbols:
        sym = symbols[0]
        response["text"] = (            f"I analyzed {len(symbols)} symbol(s)."            f"{sym} has a computed signal in the selected timeframe."            f"Open 'Show AI Reasoning'for full factor breakdown.")
    else:
        response["text"] = (            "I completed analysis, but no display-ready summary was produced."            "Please retry or open 'Show AI Reasoning'for raw diagnostics.")
    return response
```
Call this at the end of **all** query handlers.

---

## Runtime Compatibility Layer (Auto-Apply at Import)

Use a single idempotent compatibility module and import it early in app startup.

### Compatibility Module Pattern

```python
# compatibility_runtime.py
import inspect
from functools import wraps

def patch_alt_data_signal(AltDataSignal):
    if hasattr(AltDataSignal, "signal_type"):
        return

    def _get_signal_type(self):
        for attr in ("type", "signal", "category", "event_type", "label"):
            if hasattr(self, attr):
                v = getattr(self, attr)
                if v is not None:
                    return v
        return "unknown"
    def _set_signal_type(self, value):
        for attr in ("type", "signal", "category", "event_type", "label"):
            if hasattr(self, attr):
                setattr(self, attr, value)
                return
        setattr(self, "_signal_type_compat", value)

    AltDataSignal.signal_type = property(_get_signal_type, _set_signal_type)


def patch_advanced_backtester_init(AdvancedBacktester):
    init_fn = getattr(AdvancedBacktester, "init", None)
    if init_fn is None or getattr(init_fn, "_symbol_compat", False):
        return

    params = inspect.signature(init_fn).parameters

    @wraps(init_fn)
    def _wrapped(self, *args, **kwargs):
        if "symbol"in kwargs:
            sym = kwargs.pop("symbol")
            if "ticker"in params and "ticker"not in kwargs:
                kwargs["ticker"] = sym
            elif "asset"in params and "asset"not in kwargs:
                kwargs["asset"] = sym
            elif "instrument"in params and "instrument"not in kwargs:
                kwargs["instrument"] = sym
        return init_fn(self, *args, **kwargs)

    _wrapped._symbol_compat = True
    AdvancedBacktester.init = _wrapped


def apply_runtime_compatibility():
    from advanced_backtester import AdvancedBacktester
    from alternative_data_engine import AltDataSignal
    patch_alt_data_signal(AltDataSignal)
    patch_advanced_backtester_init(AdvancedBacktester)
```
### Auto-Apply at Import Time

In app bootstrap (`main.py`/ chatbot entrypoint):

```python
from compatibility_runtime import apply_runtime_compatibility
apply_runtime_compatibility() # run once at startup
```
---

## Quick Validation Checklist

```python
assert hasattr(AltDataSignal, "signal_type")
# Should not raise TypeError anymore:
# backtester.init(symbol="AAPL", ...)
# Ensure user-facing text is never empty:
# assert isinstance(response.get("text"), str) and response["text"].strip()
```