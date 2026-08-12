"""
Octavian Adaptive Personalization Engine
========================================
Tracks user behavior and learns preferences to adapt the terminal experience.
Features:
  - Action logging (Page views, searches, tool usage)
  - Interest detection (Symbol clusters, asset class bias)
  - Adaptive UI Context (Filtering and prioritization)

Author: APB - Octavian Team
"""

from __future__ import annotations
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd

from database_manager import get_database_manager
from trader_profile import get_trader_profile, save_profile

class AdaptivePersonaEngine:
    """Model that learns from user interactions and adapts outcomes."""
    
    def __init__(self):
        self.db = get_database_manager()
    
    def log_action(self, user_id: str, action_type: str, detail: Any = None):
        """Record a user action for future learning."""
        self.db.log_user_activity(user_id, action_type, detail)
        
    def analyze_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Analyze recent logs to derive learned preferences.
        Returns a dict of {sectors, asset_classes, symbols, style_bias}.
        """
        logs = self.db.get_user_activity(user_id, limit=500)
        if not logs:
            return {}
            
        df = pd.DataFrame(logs)
        
        # 1. Detect Symbol Interests
        search_logs = df[df['action_type'] == 'symbol_search']
        symbols = []
        for d in search_logs['detail']:
            if isinstance(d, dict) and 'symbol' in d:
                symbols.append(d['symbol'])
        
        top_symbols = pd.Series(symbols).value_counts().head(10).index.tolist()
        
        # 2. Detect Page/Tool Usage
        page_views = df[df['action_type'] == 'page_view']
        top_tools = page_views['detail'].apply(lambda x: x.get('page') if isinstance(x, dict) else None).value_counts()
        
        # 3. Derive Sector Bias (Heuristic)
        # Note: In a real scenario, we'd lookup symbol → sector
        # For now, we use a simple mapping for top symbols
        sectors = []
        tech_syms = {'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'AMD'}
        energy_syms = {'XOM', 'CVX', 'XLE', 'OXY'}
        
        for s in symbols:
            if s in tech_syms: sectors.append("Technology")
            if s in energy_syms: sectors.append("Energy")
            
        top_sectors = pd.Series(sectors).value_counts().head(3).index.tolist()
        
        # 4. Adaptive Mode: Learning specific AI Tone
        # If user frequently uses "Quant Portal" -> prefers "Institutional" style
        style_bias = "balanced"
        if "Quant Portal" in top_tools or "Strategy Research Lab" in top_tools:
            style_bias = "institutional"
        elif "Market Scanner" in top_tools:
            style_bias = "aggressive"
            
        return {
            "learned_top_symbols": top_symbols,
            "learned_sectors": top_sectors,
            "learned_style_bias": style_bias,
            "last_analyzed": datetime.now().isoformat()
        }

    def sync_learned_to_profile(self, user_id: str):
        """Update the shadow 'learned_profile' in session/db."""
        prefs = self.analyze_preferences(user_id)
        if not prefs:
            return
            
        # We store this in a special 'learned_persona' key in the profile
        # to separate it from user-explicit choices.
        profile = get_trader_profile()
        profile["learned_persona"] = prefs
        save_profile(profile, user_id)

    def get_adaptive_ui_layout(self, user_id: str) -> Dict[str, Any]:
        """
        Phase 6: Adaptive User Personas
        Returns a layout configuration dict that determines which UI elements 
        are prioritized (e.g. Volatility first for scalpers, Macro first for investors).
        """
        profile = get_trader_profile()
        learned = profile.get("learned_persona", {})
        style_bias = learned.get("learned_style_bias", "balanced")
        
        layout_config = {
            "show_volatility_first": False,
            "show_macro_first": False,
            "show_greeks_summary": False,
            "default_chart_timeframe": "1D"
        }
        
        if style_bias == "aggressive":
            layout_config["show_volatility_first"] = True
            layout_config["default_chart_timeframe"] = "5Min"
            layout_config["show_greeks_summary"] = True
        elif style_bias == "institutional":
            layout_config["show_macro_first"] = True
            layout_config["show_greeks_summary"] = True
            layout_config["default_chart_timeframe"] = "1W"
            
        return layout_config

def get_adaptive_engine() -> AdaptivePersonaEngine:
    return AdaptivePersonaEngine()
