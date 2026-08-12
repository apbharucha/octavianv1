"""
Octavian Backend — Subscription Tier Definitions

Defines the feature gates, rate limits, and capabilities for each tier.
Used by middleware to enforce access control and by the frontend to display
upgrade prompts.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional


class SubscriptionTier(str, Enum):
    FREE = "free"
    PRO = "pro"
    INSTITUTIONAL = "institutional"
    ENTERPRISE = "enterprise"


@dataclass(frozen=True)
class TierLimits:
    """Rate limits and quotas for a subscription tier."""
    api_calls_per_minute: int
    symbol_analyses_per_day: int
    chat_queries_per_day: int
    scanner_runs_per_day: int
    paper_trading_portfolios: int
    paper_trading_capital: int           # Virtual capital in USD
    ml_predictions_enabled: bool
    ml_prediction_models: list[str]      # Which models the user gets
    backtesting_lookback_years: int
    breaking_trades_realtime: bool
    breaking_trades_max_setups: int
    options_full_greeks: bool
    simulation_stress_testing: bool
    document_analyzer_filings_per_month: int
    sec_13f_all_funds: bool
    daily_briefing_email: bool
    daily_briefing_pdf: bool
    api_access: bool
    priority_support: bool


# --- Tier Definitions ---

TIER_LIMITS: Dict[SubscriptionTier, TierLimits] = {
    SubscriptionTier.FREE: TierLimits(
        api_calls_per_minute=10,
        symbol_analyses_per_day=3,
        chat_queries_per_day=10,
        scanner_runs_per_day=5,
        paper_trading_portfolios=1,
        paper_trading_capital=10_000,
        ml_predictions_enabled=False,
        ml_prediction_models=[],
        backtesting_lookback_years=0,
        breaking_trades_realtime=False,
        breaking_trades_max_setups=2,
        options_full_greeks=False,
        simulation_stress_testing=False,
        document_analyzer_filings_per_month=0,
        sec_13f_all_funds=False,
        daily_briefing_email=False,
        daily_briefing_pdf=False,
        api_access=False,
        priority_support=False,
    ),
    SubscriptionTier.PRO: TierLimits(
        api_calls_per_minute=60,
        symbol_analyses_per_day=-1,  # Unlimited
        chat_queries_per_day=100,
        scanner_runs_per_day=-1,
        paper_trading_portfolios=3,
        paper_trading_capital=100_000,
        ml_predictions_enabled=True,
        ml_prediction_models=["LSTM_Neural_Net"],
        backtesting_lookback_years=1,
        breaking_trades_realtime=True,
        breaking_trades_max_setups=5,
        options_full_greeks=True,
        simulation_stress_testing=False,
        document_analyzer_filings_per_month=5,
        sec_13f_all_funds=False,
        daily_briefing_email=True,
        daily_briefing_pdf=False,
        api_access=False,
        priority_support=False,
    ),
    SubscriptionTier.INSTITUTIONAL: TierLimits(
        api_calls_per_minute=200,
        symbol_analyses_per_day=-1,
        chat_queries_per_day=-1,
        scanner_runs_per_day=-1,
        paper_trading_portfolios=-1,
        paper_trading_capital=1_000_000,
        ml_predictions_enabled=True,
        ml_prediction_models=[
            "LSTM_Neural_Net",
            "Transformer_Attention",
            "Deep_MLP_Network",
            "Random_Forest_Ensemble",
            "Gradient_Boosting_Machine",
        ],
        backtesting_lookback_years=5,
        breaking_trades_realtime=True,
        breaking_trades_max_setups=-1,
        options_full_greeks=True,
        simulation_stress_testing=True,
        document_analyzer_filings_per_month=-1,
        sec_13f_all_funds=True,
        daily_briefing_email=True,
        daily_briefing_pdf=True,
        api_access=True,
        priority_support=True,
    ),
    SubscriptionTier.ENTERPRISE: TierLimits(
        api_calls_per_minute=1000,
        symbol_analyses_per_day=-1,
        chat_queries_per_day=-1,
        scanner_runs_per_day=-1,
        paper_trading_portfolios=-1,
        paper_trading_capital=10_000_000,
        ml_predictions_enabled=True,
        ml_prediction_models=[
            "LSTM_Neural_Net",
            "Transformer_Attention",
            "Deep_MLP_Network",
            "Random_Forest_Ensemble",
            "Gradient_Boosting_Machine",
        ],
        backtesting_lookback_years=-1,
        breaking_trades_realtime=True,
        breaking_trades_max_setups=-1,
        options_full_greeks=True,
        simulation_stress_testing=True,
        document_analyzer_filings_per_month=-1,
        sec_13f_all_funds=True,
        daily_briefing_email=True,
        daily_briefing_pdf=True,
        api_access=True,
        priority_support=True,
    ),
}


def get_tier_limits(tier: SubscriptionTier) -> TierLimits:
    """Get limits for a subscription tier."""
    return TIER_LIMITS[tier]


def check_feature_access(tier: SubscriptionTier, feature: str) -> bool:
    """Check if a tier has access to a specific feature.
    
    Returns True if access is granted, False if the user needs to upgrade.
    """
    limits = get_tier_limits(tier)
    
    feature_checks = {
        "ml_predictions": limits.ml_predictions_enabled,
        "backtesting": limits.backtesting_lookback_years > 0,
        "breaking_trades_realtime": limits.breaking_trades_realtime,
        "options_greeks": limits.options_full_greeks,
        "stress_testing": limits.simulation_stress_testing,
        "document_analyzer": limits.document_analyzer_filings_per_month != 0,
        "sec_13f_all": limits.sec_13f_all_funds,
        "daily_briefing_email": limits.daily_briefing_email,
        "daily_briefing_pdf": limits.daily_briefing_pdf,
        "api_access": limits.api_access,
    }
    
    return feature_checks.get(feature, False)
