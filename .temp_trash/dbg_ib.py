import sys
sys.path.insert(0, "/Users/aavibharucha/Documents/octavian")
import numpy as np
from mna_model_engine import MnAAssumptions, get_mna_engine
from lbo_model_engine import LBOAssumptions, get_lbo_engine

# --- M&A contribution analysis sum ---
m = get_mna_engine().run_mna(MnAAssumptions(
    acquirer_ticker='ACQ', target_ticker='TGT',
    acquirer_price=100, acquirer_eps=5, acquirer_shares=1000,
    target_price=50, target_eps=2, target_shares=200,
    offer_premium=0.30, percent_stock=0.5, percent_cash=0.5,
    cost_of_debt=0.06, tax_rate=0.21, pre_tax_synergies=100,
))
print("pf EPS:", m.pro_forma_eps)
print("acq eps + sum impacts:", m.assumptions.acquirer_eps + m.contribution_analysis["EPS Impact ($)"].sum())
print(m.contribution_analysis)

# --- LBO IRR/MOIC ---
l = get_lbo_engine().run_lbo(LBOAssumptions(
    ticker='TGT', target_name='Target', entry_year=2026, exit_year=2031,
    ltm_ebitda=500, entry_multiple=10, exit_multiple=10,
    leverage_multiple=5, interest_rate=0.08, tax_rate=0.25,
))
print("\nLBO equity_amount:", l.equity_amount, "exit_equity:", l.exit_equity_value)
print("MOIC:", l.moic, "expected", l.exit_equity_value / l.equity_amount)
years = 5
print("IRR:", l.irr, "expected", (l.exit_equity_value / l.equity_amount) ** (1/years) - 1)
print("total_sources:", l.total_sources, "total_uses:", l.total_uses)
