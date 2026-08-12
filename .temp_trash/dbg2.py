import sys
sys.path.insert(0, "/Users/aavibharucha/Documents/octavian")
import pandas as pd
from mna_model_engine import MnAAssumptions, get_mna_engine


def make_mna(**kw):
    base = dict(
        acquirer_ticker="ACQ", target_ticker="TGT",
        acquirer_price=100, acquirer_eps=5, acquirer_shares=1000,
        target_price=50, target_eps=2, target_shares=200,
        offer_premium=0.30, percent_stock=0.5, percent_cash=0.5,
        cost_of_debt=0.06, tax_rate=0.21, pre_tax_synergies=100,
    )
    base.update(kw)
    return MnAAssumptions(**base)


mna_result = get_mna_engine().run_mna(make_mna())
contrib = mna_result.contribution_analysis
print("columns:", list(contrib.columns))
print("dtypes:\n", contrib.dtypes)
labels = set(contrib["Driver"])
print("labels:", labels)
print("'Synerg' in labels check:", any("Synerg" in l for l in labels))
for l in labels:
    print(repr(l))
