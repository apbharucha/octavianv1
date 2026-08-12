"""
Patch for MarketSimulationEngine: adds missing _calculate_total_portfolio_value method.
Import this module BEFORE running simulations:
    import simulation_fix
"""

# This file is no longer needed - _calculate_total_portfolio_value
# is now defined directly in MarketSimulationEngine.
# Kept as empty module to avoid ImportError if referenced elsewhere.
