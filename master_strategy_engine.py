"""
Octavian Master Strategy Engine (Institutional Core)
====================================================
The centralized intelligence nexus for the Octavian Terminal. 
Integrates Macro, ML, Fundamental, and Options engines into a unified Bayesian consensus.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
import threading
import logging

# Type Definitions
from dataclasses import dataclass, field

@dataclass
class OctavianOutlook:
    bias: str                     # "BULLISH", "BEARISH", "NEUTRAL"
    conviction: float             # 0.0 to 1.0
    regime: str                   # "Stagflation", "Reflation", etc.
    primary_reason: str
    key_factors: List[str]
    bayesian_refinement: Dict[str, float]
    inter_market_divergence: List[str]
    volatility_forecast: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class MasterStrategyEngine:
    """
    Institutional Intelligence Nexus.
    Orchestrates specialized engines and provides a unified 'Dominant Perspective'.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MasterStrategyEngine, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized: return
        self.logger = logging.getLogger("OctavianMaster")
        self._initialized = True
        
        # Core Specialized Engines
        self._ml_engine = None
        self._macro_engine = None
        self._risk_engine = None
        self._fund_engine = None

    @property
    def ml_engine(self):
        if self._ml_engine is None:
            from advanced_ml_engine import get_ensemble_engine
            self._ml_engine = get_ensemble_engine()
        return self._ml_engine

    @property
    def macro_engine(self):
        if self._macro_engine is None:
            from macro_cross_asset_engine import get_macro_engine
            self._macro_engine = get_macro_engine()
        return self._macro_engine

    @property
    def risk_engine(self):
        if self._risk_engine is None:
            from risk_engine import InstitutionalRiskEngine
            self._risk_engine = InstitutionalRiskEngine()
        return self._risk_engine

    @property
    def fund_engine(self):
        if self._fund_engine is None:
            from fundamental_analyzer import FundamentalAnalyzer
            self._fund_engine = FundamentalAnalyzer()
        return self._fund_engine

    def get_dominant_outlook(self, refresh: bool = False) -> OctavianOutlook:
        """
        Synthesize the platform-wide dominant market perspective.
        Uses Bayesian posterior refinement to ensure all engines are weighted optimally.
        """
        if not refresh and hasattr(self, "_current_outlook") and self._current_outlook:
            # Check if current outlook is from the same hour
            if self._current_outlook.timestamp.startswith(datetime.now().strftime('%Y-%m-%dT%H')):
                return self._current_outlook

        # 1. Macro Foundation (The 'Prior')
        macro_dash = self.macro_engine.build_dashboard()
        regime = macro_dash.macro_regime.name
        
        # Normalize macro score to [-1, 1]
        raw_macro = np.mean([s.current_reading for s in macro_dash.signals]) if macro_dash.signals else 0.0
        macro_score = np.clip(raw_macro, -1.0, 1.0)
        
        # 2. Evidence Gathering
        # ML Alpha (Kalman-smoothed)
        # Fetch benchmark data if not available to ensure live alpha is calculated
        from data_sources import get_stock
        spy_df = get_stock("SPY", period="3mo")
        ml_res = self.ml_engine.analyze_symbol_ensemble(spy_df, "SPY") if not spy_df.empty else {}
        
        raw_ml = ml_res.get("predicted_return", 0.0) * 50.0 # Scaling target for alpha
        ml_score = np.clip(np.tanh(raw_ml), -1.0, 1.0)
        vol_forecast = ml_res.get("volatility_forecast", 0.20)
        
        # Fundamental Pulse (DCF & Relative Value)
        fund_res = self.fund_engine.fetch_fundamentals("SPY")
        fund_score = np.clip(fund_res.score if fund_res else 0.0, -1.0, 1.0)
        
        # 3. Bayesian Synthesis
        bayesian = self._calculate_bayesian_posterior(macro_score, [ml_score, fund_score])
        
        # 4. Dynamic Weighting (Regime-Aware)
        weights = self._derive_regime_weights(vol_forecast, regime)
        
        final_score = (macro_score * weights['macro']) + \
                      (ml_score * weights['ml']) + \
                      (fund_score * weights['fund']) + \
                      (bayesian['posterior_score'] * 0.15) # Bayesian bias
                      
        bias = "BULLISH" if final_score > 0.05 else "BEARISH" if final_score < -0.05 else "NEUTRAL"
        conviction = np.clip(abs(final_score) * 2.0, 0.1, 0.95)
        
        # 5. Divergence Detection
        divergences = self._scan_intermarket_anomalies(macro_dash)
        
        outlook = OctavianOutlook(
            bias=bias,
            conviction=float(conviction),
            regime=regime,
            primary_reason=self._construct_thesis(bias, regime, bayesian, divergences),
            key_factors=[
                f"Bayesian Core: {bayesian['posterior_score']:+.2f}",
                f"Systemic Volatility: {vol_forecast*100:.1f}%",
                f"ML Factor Alpha: {ml_score:+.2f}",
                f"Fundamental Quality: {fund_score:+.2f}"
            ],
            bayesian_refinement=bayesian,
            inter_market_divergence=divergences,
            volatility_forecast=float(vol_forecast)
        )
        
        self._current_outlook = outlook
        return outlook

    def _derive_regime_weights(self, vol: float, regime: str) -> Dict[str, float]:
        """Adjust engine weights based on volatility and structural state."""
        w = {'macro': 0.35, 'ml': 0.35, 'fund': 0.30}
        if vol > 0.30: w = {'macro': 0.50, 'ml': 0.20, 'fund': 0.30}
        elif vol < 0.15: w = {'macro': 0.20, 'ml': 0.50, 'fund': 0.30}
        return w

    def _calculate_bayesian_posterior(self, prior_score: float, evidence_scores: List[float]) -> Dict[str, float]:
        """Update probability baseline using Bayesian inference."""
        # Ensure input range [0, 1]
        prior = np.clip((prior_score + 1.0) / 2.0, 0.01, 0.99)
        evidence = np.clip(np.mean([(s + 1.0) / 2.0 for s in evidence_scores]), 0.01, 0.99)
        
        # Confidence factor in evidence (likelihood)
        likelihood = 0.75 if abs(prior - evidence) < 0.2 else 0.45
        
        denominator = (likelihood * prior + (1.0 - likelihood) * (1.0 - prior))
        posterior = (likelihood * prior) / denominator if denominator > 0 else prior
        posterior_score = np.clip(posterior * 2.0 - 1.0, -1.0, 1.0)
        
        return {
            "prior_score": float(prior_score),
            "evidence_mean": float(evidence * 2.0 - 1.0),
            "posterior_score": float(posterior_score),
            "refinement_delta": float(posterior_score - prior_score)
        }

    def _scan_intermarket_anomalies(self, dash: Any) -> List[str]:
        """Identifies divergences in correlations."""
        divs = []
        try:
            for s in dash.signals:
                if s.direction == "WATCH" or s.strength > 80:
                    divs.append(f"{s.relationship} Anomaly")
        except: pass
        return divs

    def _construct_thesis(self, bias, regime, bayesian, divs) -> str:
        """Professional synthesis of engine logic."""
        sentiment = "constructive" if bias == "BULLISH" else "cautious" if bias == "BEARISH" else "neutral"
        text = f"Platform intelligence is {sentiment} given the prevailing {regime} regime. "
        text += f"Bayesian consensus models indicate a steady conviction profile, with multi-factor evidence supporting the current structural bias. "
        if divs: text += f"Note: {divs[0]} remains a secondary risk factor."
        return text

def get_master_engine() -> MasterStrategyEngine:
    return MasterStrategyEngine()
