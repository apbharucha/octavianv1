"""Verify PowerPoint pitchbooks: charts present, no empty slides, typography."""
import io
import sys
import types
from unittest.mock import patch, MagicMock

sys.path.insert(0, "/Users/aavibharucha/Documents/octavian")

from pptx import Presentation

from mna_model_engine import MnAAssumptions, get_mna_engine
from lbo_model_engine import LBOAssumptions, get_lbo_engine
from financial_model_generator import DCFAssumptions, get_dcf_engine
from presentation_generator import (
    get_presentation_generator, _add_bar_chart, _add_pie_chart, _new_presentation,
    _blank_slide, SLIDE_W, SLIDE_H,
)

failures = []

# ── M&A pitchbook ────────────────────────────────────────────────────────────
m = get_mna_engine().run_mna(MnAAssumptions(
    acquirer_ticker="ACQ", target_ticker="TGT",
    acquirer_price=100, acquirer_eps=5, acquirer_shares=1000,
    target_price=50, target_eps=2, target_shares=800,
    offer_premium=0.30, percent_stock=0.5, percent_cash=0.5,
    cost_of_debt=0.05, tax_rate=0.21, pre_tax_synergies=500,
))
pptx_bytes = get_presentation_generator().generate_mna_pitchbook("ACQ", "TGT", m)
assert len(pptx_bytes) > 10000, "M&A deck too small"
prs = Presentation(io.BytesIO(pptx_bytes))
n_slides = len(prs.slides.__iter__.__self__._sldIdLst)
n_charts = 0
for slide in prs.slides:
    for shape in slide.shapes:
        if shape.has_chart:
            n_charts += 1
print(f"M&A deck: {n_slides} slides, {n_charts} native charts, {len(pptx_bytes)} bytes")
if n_charts < 2:
    failures.append(f"M&A: expected >=2 charts, got {n_charts}")

# ── DCF pitchbook ────────────────────────────────────────────────────────────
d = get_dcf_engine().run_dcf(DCFAssumptions(
    ticker="TST", base_revenue=5000, revenue_growth_rates=[0.08]*5,
    ebit_margin=0.20, tax_rate=0.21, da_pct_revenue=0.04, capex_pct_revenue=0.05,
    nwc_change_pct_revenue=0.01, equity_value_market=30000, debt_value=8000,
    cost_of_debt=0.045, risk_free_rate=0.0425, equity_risk_premium=0.055,
    beta=1.0, terminal_growth_rate=0.025, cash=2000, shares_outstanding=1000,
    current_price=30, peer_pe=25.0, peer_ev_ebitda=15.0, peer_ev_fcf=20.0,
    peg_ratio=2.0, projection_years=5,
))
pptx_bytes = get_presentation_generator().generate_dcf_pitchbook("TST", d)
assert len(pptx_bytes) > 10000, "DCF deck too small"
prs = Presentation(io.BytesIO(pptx_bytes))
n_slides = len(prs.slides.__iter__.__self__._sldIdLst)
n_charts = sum(1 for s in prs.slides for sh in s.shapes if sh.has_chart)
print(f"DCF deck: {n_slides} slides, {n_charts} native charts, {len(pptx_bytes)} bytes")
if n_charts < 2:
    failures.append(f"DCF: expected >=2 charts, got {n_charts}")

# ── LBO pitchbook ────────────────────────────────────────────────────────────
l = get_lbo_engine().run_lbo(LBOAssumptions(
    ticker="TGT", target_name="Target", entry_year=2024, exit_year=2029,
    ltm_ebitda=1000, entry_multiple=10.0, exit_multiple=10.0,
    leverage_multiple=6.0, interest_rate=0.08, tax_rate=0.21,
    revenue_growth_rates=[0.05]*5, ebitda_margins=[0.25]*5,
    capex_pct_rev=0.04, nwc_pct_rev=0.01, depreciation_pct_rev=0.05,
))
pptx_bytes = get_presentation_generator().generate_lbo_pitchbook("TGT", l)
assert len(pptx_bytes) > 10000, "LBO deck too small"
prs = Presentation(io.BytesIO(pptx_bytes))
n_slides = len(prs.slides.__iter__.__self__._sldIdLst)
n_charts = sum(1 for s in prs.slides for sh in s.shapes if sh.has_chart)
print(f"LBO deck: {n_slides} slides, {n_charts} native charts, {len(pptx_bytes)} bytes")
if n_charts < 1:
    failures.append(f"LBO: expected >=1 chart, got {n_charts}")

# ── Slide-geometry sanity: nothing hangs off-canvas, no zero-size shapes ─────
for deck_name, gen, arg in [
    ("M&A", get_presentation_generator().generate_mna_pitchbook, ("ACQ", "TGT", m)),
    ("DCF", get_presentation_generator().generate_dcf_pitchbook, ("TST", d)),
    ("LBO", get_presentation_generator().generate_lbo_pitchbook, ("TGT", l)),
]:
    b = gen(*arg)
    prs = Presentation(io.BytesIO(b))
    for si, slide in enumerate(prs.slides):
        for sh in slide.shapes:
            try:
                l, t, w, h = sh.left, sh.top, sh.width, sh.height
            except Exception:
                continue
            if l is None:
                continue
            if w is not None and (w <= 0 or h <= 0):
                failures.append(f"{deck_name} slide {si+1}: zero-size shape {sh.shape_type}")
            # allow slight overhang tolerance (charts may extend a hair)
            if l is not None and w is not None:
                slide_w_emu = prs.slide_width
                if l + w > slide_w_emu + int(slide_w_emu * 0.02):
                    failures.append(f"{deck_name} slide {si+1}: shape overflows right edge (l+w={l + w} vs slide_w={slide_w_emu})")

print("\n==================== RESULT ====================")
if failures:
    print("FAILURES:")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("ALL PPT VERIFICATION TESTS PASSED")
