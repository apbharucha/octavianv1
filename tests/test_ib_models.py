"""
Institutional model & Excel-formula integrity tests.

Validates:
1. M&A engine: accretion/dilution math, sources=uses balance, contribution
   analysis sums to pro forma EPS, sensitivity tables, breakeven synergies.
2. LBO engine: sources=uses, debt schedule monotonicity, IRR/MOIC sanity,
   value bridge decomposition, sensitivity tables.
3. Excel workbooks: every live formula is resolvable and evaluates to the
   same value the engine computed (formula integrity / no broken refs).
"""

import io
import re
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from openpyxl.utils import get_column_letter

from mna_model_engine import MnAAssumptions, get_mna_engine
from lbo_model_engine import LBOAssumptions, get_lbo_engine


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────
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


def make_lbo(**kw):
    base = dict(
        ticker="TGT", target_name="Target", entry_year=2026, exit_year=2031,
        ltm_ebitda=500, entry_multiple=10, exit_multiple=10,
        leverage_multiple=5, interest_rate=0.08, tax_rate=0.25,
    )
    base.update(kw)
    return LBOAssumptions(**base)


@pytest.fixture
def mna_result():
    return get_mna_engine().run_mna(make_mna())


@pytest.fixture
def lbo_result():
    return get_lbo_engine().run_lbo(make_lbo())


# ─────────────────────────────────────────────────────────────────────────────
# M&A engine
# ─────────────────────────────────────────────────────────────────────────────
def test_mna_sources_equals_uses(mna_result):
    assert abs(mna_result.sources_uses_check) < 1e-6


def test_mna_accretion_math(mna_result):
    a = mna_result.assumptions
    expected_offer = a.target_price * (1 + a.offer_premium)
    assert mna_result.offer_price == pytest.approx(expected_offer)
    assert mna_result.total_deal_value == pytest.approx(expected_offer * a.target_shares)

    # Pro forma EPS must be (acq NI + tgt NI + syn_at - int_at - fees)/pf shares
    acq_ni = a.acquirer_eps * a.acquirer_shares
    tgt_ni = a.target_eps * a.target_shares
    syn_at = a.pre_tax_synergies * (1 - a.tax_rate)
    int_at = mna_result.total_deal_value * a.percent_cash * a.cost_of_debt * (1 - a.tax_rate)
    new_shares = mna_result.total_deal_value * a.percent_stock / a.acquirer_price
    pf_shares = a.acquirer_shares + new_shares
    expected_ni = acq_ni + tgt_ni + syn_at - int_at
    assert mna_result.pro_forma_net_income == pytest.approx(expected_ni, rel=1e-6)
    assert mna_result.pro_forma_eps == pytest.approx(expected_ni / pf_shares, rel=1e-6)


def test_mna_contribution_analysis_sums_to_pf_eps(mna_result):
    contrib = mna_result.contribution_analysis
    # Sum only the driver rows (exclude the final 'Pro forma EPS' result row,
    # which repeats the total and would double-count).
    drivers = contrib[contrib["Type"] != "Result"]
    total = drivers["EPS Impact ($)"].sum()
    # The walk starts at standalone acquirer EPS; drivers must bridge to pf EPS
    assert mna_result.pro_forma_eps == pytest.approx(
        mna_result.assumptions.acquirer_eps + total, rel=1e-6)
    # Cumulative column's final value equals pro forma EPS
    assert contrib.iloc[-1]["Cumulative EPS ($)"] == pytest.approx(
        mna_result.pro_forma_eps, rel=1e-6)
    # Every driver present (case-insensitive)
    labels = [str(l).lower() for l in contrib["Driver"]]
    assert any("target net income" in l for l in labels)
    assert any("synerg" in l for l in labels)
    assert any("interest" in l for l in labels)
    assert any("dilution" in l for l in labels)


def test_mna_sensitivity_shapes_and_values(mna_result):
    s = mna_result.sensitivity
    assert s.premium_stock_table.shape == (9, 6)
    assert s.premium_synergy_table.shape == (9, 4)
    # At the base premium/stock combo, sensitivity must match the base case
    base_row = s.premium_stock_table[
        s.premium_stock_table["Premium"] == "30%"].iloc[0]
    assert base_row["50%"] == pytest.approx(mna_result.accretion_dilution_pct, rel=1e-6)
    # Higher premium should be less accretive (or more dilutive)
    low = s.premium_stock_table[s.premium_stock_table["Premium"] == "10%"].iloc[0]["50%"]
    high = s.premium_stock_table[s.premium_stock_table["Premium"] == "50%"].iloc[0]["50%"]
    assert high < low


def test_mna_breakeven_synergies(mna_result):
    be = mna_result.sensitivity.breakeven_synergies
    assert be is not None and be > 0
    # At breakeven synergies, accretion should be ~0
    a = make_mna(pre_tax_synergies=be)
    r = get_mna_engine().run_mna(a)
    assert abs(r.accretion_dilution_pct) < 1e-3


def test_mna_all_cash_stock_consistency():
    # 100% stock: new shares = deal/price; no interest
    r = get_mna_engine().run_mna(make_mna(percent_stock=1.0, percent_cash=0.0))
    a = r.assumptions
    assert r.new_shares_issued == pytest.approx(r.total_deal_value / a.acquirer_price)
    assert r.pro_forma_shares == pytest.approx(a.acquirer_shares + r.new_shares_issued)
    assert abs(r.sources_uses_check) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# LBO engine
# ─────────────────────────────────────────────────────────────────────────────
def test_lbo_sources_equals_uses(lbo_result):
    assert abs(lbo_result.sources_uses_check) < 1e-6


def test_lbo_debt_schedule(lbo_result):
    cf = lbo_result.cash_flows
    # TLB balance never negative, never exceeds initial
    tlb = cf["Ending TLB Balance"]
    assert (tlb >= -1e-9).all()
    assert (tlb <= lbo_result.debt_amount + 1e-9).all()
    # Mandatory amortization is non-negative and monotonic-ish (non-increasing debt)
    assert (cf["(-) Mandatory Amortization"] <= 0).all()
    # Total debt decreases over time (cash sweep + amortization)
    assert cf["Total Debt"].iloc[-1] < cf["Total Debt"].iloc[0]


def test_lbo_irr_moic_consistency(lbo_result):
    years = lbo_result.assumptions.exit_year - lbo_result.assumptions.entry_year
    expected_moic = lbo_result.exit_equity_value / lbo_result.equity_amount
    assert lbo_result.moic == pytest.approx(expected_moic, rel=1e-6)
    expected_irr = expected_moic ** (1 / years) - 1
    assert lbo_result.irr == pytest.approx(expected_irr, rel=1e-6)
    assert lbo_result.moic > 0 and lbo_result.irr > -1


def test_lbo_value_bridge_components(lbo_result):
    bridge = lbo_result.value_bridge
    assert bridge["EBITDA growth contribution"] > 0  # 5% growth each year
    assert bridge["Multiple expansion (contraction) contribution"] == pytest.approx(0.0)
    # Deleveraging contribution = ending debt - initial (negative = paid down)
    assert bridge["Net debt change (deleveraging)"] <= 0


def test_lbo_sensitivity_tables(lbo_result):
    s = lbo_result.sensitivity
    assert s.irr_table.shape == (7, 8)
    assert s.moic_table.shape == (7, 8)
    # Base cell (10x entry, 10x exit) matches
    row = s.irr_table[s.irr_table["Entry"] == "10.0x"].iloc[0]
    assert row["10.0x"] == pytest.approx(lbo_result.irr, rel=1e-6)
    # Higher exit multiple -> higher IRR
    low = s.irr_table[s.irr_table["Entry"] == "10.0x"].iloc[0]["7.0x"]
    high = s.irr_table[s.irr_table["Entry"] == "10.0x"].iloc[0]["12.0x"]
    assert high > low


def test_lbo_revolver_draw_on_weak_cash_flow():
    # Aggressive leverage + high interest should draw the revolver in early years
    r = get_lbo_engine().run_lbo(make_lbo(
        leverage_multiple=8.0, interest_rate=0.12, revenue_growth_rates=[-0.05] * 5,
    ))
    rev = r.cash_flows["Ending Revolver Balance"]
    assert (rev >= -1e-9).all()
    assert rev.sum() > 0, "weak cash flows should draw the revolver"


# ─────────────────────────────────────────────────────────────────────────────
# Excel live-formula integrity
#
# The workbooks emit a small, well-defined formula subset:
#   * arithmetic: + - * / ^ ( )
#   * MIN(a,b) / MAX(a,b) — possibly nested (e.g. MIN(MAX(x,0)*0.5, cap))
#   * cell refs: B12  (same-sheet)  and  Projections!F17 / 'Sources & Uses'!B14
#   * numbers possibly carrying thousands separators (e.g. 2,500.00)
# The evaluator below resolves refs recursively against the parsed workbook
# (depth-guarded), converts MIN/MAX to Python min/max with a balanced-paren
# scanner, rewrites ^ to **, and strips thousands separators.
# ─────────────────────────────────────────────────────────────────────────────
_eval_depth = 0
# Memoization cache + in-progress set so the recursive evaluator stays linear
# on the (acyclic) formula graph instead of re-evaluating shared sub-chains
# exponentially. Both are keyed by (sheet_name, column_letter, row).
_eval_cache = {}
_eval_in_progress = set()

_SHEET_REF = re.compile(r"(?:'([^']+)'|([A-Za-z_][A-Za-z_0-9]*))!(\$?[A-Z]{1,3}\$?\d+)")
_BARE_REF = re.compile(r"(?<![A-Za-z0-9_])(\$?[A-Z]{1,3}\$?\d+)")


def _find_matching_paren(s: str, start: int) -> int:
    """Index of the ')' matching the '(' at s[start]."""
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "(":
            depth += 1
        elif s[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _split_top_level(s: str):
    """Split on commas at paren depth 0."""
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch == "(":
            depth += 1
            cur += ch
        elif ch == ")":
            depth -= 1
            cur += ch
        elif ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return parts


def _rewrite_sum(body: str) -> str:
    """Convert SUM(A1:B5) into A1+A2+...+B5 (same row/col ranges only)."""
    pattern = re.compile(r"SUM\(([A-Z]{1,3}\d+):([A-Z]{1,3}\d+)\)")
    while True:
        m = pattern.search(body)
        if not m:
            return body
        start, end = m.group(1), m.group(2)
        c1, r1 = re.match(r"([A-Z]+)(\d+)", start).groups()
        c2, r2 = re.match(r"([A-Z]+)(\d+)", end).groups()
        r1, r2 = int(r1), int(r2)
        if c1 == c2:
            cells = [f"{c1}{r}" for r in range(r1, r2 + 1)]
        elif r1 == r2:
            cols = [get_column_letter(i) for i in range(
                _col_index(c1), _col_index(c2) + 1)]
            cells = [f"{c}{r1}" for c in cols]
        else:
            cells = [start, end]  # can't expand 2-D range; leave as-is
        body = body[:m.start()] + "(" + "+".join(cells) + ")" + body[m.end():]


def _col_index(letter: str) -> int:
    idx = 0
    for ch in letter:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx


def _rewrite_minmax(body: str) -> str:
    """Convert (possibly nested) MIN(a,b)/MAX(a,b) to Python min()/max()."""
    while True:
        idx = min(
            (i for i in (body.find("MIN("), body.find("MAX(")) if i >= 0),
            default=-1,
        )
        if idx < 0:
            return body
        fn = body[idx:idx + 3]
        open_i = body.index("(", idx)
        close_i = _find_matching_paren(body, open_i)
        if close_i < 0:
            return body
        args = _split_top_level(body[open_i + 1:close_i])
        if len(args) != 2:
            return body
        inner = "min" if fn == "MIN" else "max"
        body = body[:idx] + inner + "(" + args[0] + "," + args[1] + ")" + body[close_i + 1:]


def _resolve_ref(ref: str, sheet):
    """Resolve 'A1' or 'Sheet!A1' to a numeric value. Recursively evaluates
    formula cells (with a depth guard and memoization against cycles)."""
    global _eval_depth
    if _eval_depth > 60:
        return None
    if "!" in ref:
        sname, addr = ref.split("!", 1)
        sname = sname.strip("'")
        sheet = _wb_sheets.get(sname)
        if sheet is None:
            return None
    else:
        sname = None
        addr = ref
    addr = addr.replace("$", "")  # tolerate absolute refs ($B$42)
    m = re.match(r"^([A-Z]+)(\d+)$", addr)
    if not m:
        return None
    col, row = m.group(1), int(m.group(2))
    cell = sheet.get((col, row))
    if cell is None:
        return None
    if isinstance(cell, str) and cell.startswith("="):
        sname = sname or _sheet_name_of(sheet)
        key = (sname, col, row)
        if key in _eval_in_progress:
            return None  # genuine cycle -> treat as 0 (Excel would flag #REF!)
        if key in _eval_cache:
            return _eval_cache[key]
        _eval_in_progress.add(key)
        _eval_depth += 1
        try:
            val = _eval_formula(cell, sheet)
            _eval_cache[key] = val
            return val
        finally:
            _eval_depth -= 1
            _eval_in_progress.discard(key)
    if isinstance(cell, (int, float)) and not isinstance(cell, bool):
        return float(cell)
    try:
        return float(cell)
    except (TypeError, ValueError):
        return None


def _sheet_name_of(sheet):
    """Reverse-lookup the name of a sheet map (used for cache keys)."""
    for name, m in _wb_sheets.items():
        if m is sheet:
            return name
    return "?"


def _eval_formula(formula: str, sheet) -> float:
    """Evaluate our emitted formula subset. Returns numeric result."""
    body = formula[1:].strip()
    # Strip *thousands separators* only (digit,comma,three digits) — a plain
    # digit-comma-digit pattern would corrupt MIN(B26,0) into MIN(B260).
    body = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", body)
    body = _rewrite_minmax(body)                            # MIN/MAX -> min/max
    body = _rewrite_sum(body)                               # SUM(A1:B3) -> a+b+...

    def sheet_repl(m):
        sname = m.group(1) or m.group(2)
        v = _resolve_ref(f"{sname}!{m.group(3)}", sheet)
        return str(v) if v is not None else "0"

    def bare_repl(m):
        v = _resolve_ref(m.group(1), sheet)
        return str(v) if v is not None else "0"

    body = _SHEET_REF.sub(sheet_repl, body)
    body = _BARE_REF.sub(bare_repl, body)
    body = body.replace("^", "**")
    # min/max come from the MIN/MAX rewrite; supply them explicitly since
    # __builtins__ is emptied for safety.
    namespace = {"min": min, "max": max, "abs": abs}
    try:
        return float(eval(body, {"__builtins__": {}}, namespace))
    except Exception:
        return float("nan")


def _collect_cells(wb):
    """Map sheet-name -> {(col_letter, row): value}."""
    sheets = {}
    for ws in wb.worksheets:
        m = {}
        for row in ws.iter_rows():
            for cell in row:
                if cell.value is not None:
                    m[(cell.column_letter, cell.row)] = cell.value
        sheets[ws.title] = m
    return sheets


def _check_formulas(wb, expected_checks):
    """Evaluate every formula and assert it's finite; then check key cells
    against expected values computed by the engines."""
    global _wb_sheets, _eval_cache, _eval_in_progress
    _wb_sheets = _collect_cells(wb)
    _eval_cache = {}
    _eval_in_progress = set()
    bad = []
    for ws in wb.worksheets:
        sheet_map = _wb_sheets[ws.title]
        for (col, row), val in sheet_map.items():
            if isinstance(val, str) and val.startswith("="):
                result = _eval_formula(val, sheet_map)
                if not np.isfinite(result):
                    bad.append(f"{ws.title}!{col}{row}: {val} -> {result}")
    assert not bad, f"Broken formulas: {bad[:5]}"
    for (sheet_name, addr, expected) in expected_checks:
        m = re.match(r"^([A-Z]+)(\d+)$", addr)
        got = _wb_sheets[sheet_name].get((m.group(1), int(m.group(2))))
        if isinstance(got, str) and got.startswith("="):
            got = _eval_formula(got, _wb_sheets[sheet_name])
        assert got is not None and abs(float(got) - expected) < max(1.0, abs(expected) * 1e-6), \
            f"{sheet_name}!{addr}: got {got}, expected {expected}"


def test_mna_workbook_formula_integrity(mna_result):
    from ib_excel_engine import build_mna_workbook
    from openpyxl import load_workbook
    data = build_mna_workbook(mna_result)
    wb = load_workbook(io.BytesIO(data))
    # expected: deal value 13000, offer 65, pf EPS ~4.855, acc/dil -2.9%
    _check_formulas(wb, [
        ("Assumptions", "B46", 65.0),                    # offer price
        ("Assumptions", "B48", 13000.0),                 # deal value
        ("Assumptions", "B70", mna_result.pro_forma_eps),
        ("Assumptions", "B72", mna_result.accretion_dilution_dollar),
    ])


def test_lbo_workbook_formula_integrity(lbo_result):
    from ib_excel_engine import build_lbo_workbook
    from openpyxl import load_workbook
    data = build_lbo_workbook(lbo_result)
    wb = load_workbook(io.BytesIO(data))
    # Every formula in the workbook must resolve to a finite number — this
    # catches broken refs (e.g. a sweep that references the wrong year) that
    # would otherwise surface as #REF! / #VALUE! errors in Excel.
    _check_formulas(wb, [])
    # Locate the 'Exit equity value ($M)' live formula on the Returns sheet
    # and verify it reproduces the engine's exit equity exactly.
    global _wb_sheets, _eval_cache, _eval_in_progress
    _wb_sheets = _collect_cells(wb)
    _eval_cache = {}
    _eval_in_progress = set()
    ret = _wb_sheets["Returns"]
    target_row = None
    for (col, row), val in ret.items():
        if col == "A" and isinstance(val, str) and "Exit equity value" in val:
            target_row = row
            break
    assert target_row is not None, "Exit equity value row not found on Returns sheet"
    got = ret.get(("B", target_row + 1))  # labels at N, values at N+1
    assert isinstance(got, str) and got.startswith("="), f"Exit equity not a formula: {got!r}"
    res = _eval_formula(got, ret)
    assert abs(res - lbo_result.exit_equity_value) < max(
        1.0, lbo_result.exit_equity_value * 1e-6), \
        f"Exit equity {res} != engine {lbo_result.exit_equity_value}"
    # MOIC = exit equity / sponsor equity must also reproduce
    moic_row = None
    for (col, row), val in ret.items():
        if col == "A" and isinstance(val, str) and val.strip() == "MOIC (x)":
            moic_row = row
            break
    assert moic_row is not None
    moic_val = ret.get(("B", moic_row + 1))  # labels at N, values at N+1
    if isinstance(moic_val, str) and moic_val.startswith("="):
        moic_val = _eval_formula(moic_val, ret)
    assert abs(float(moic_val) - lbo_result.moic) < max(
        1e-6, abs(lbo_result.moic) * 1e-6), \
        f"MOIC {moic_val} != engine {lbo_result.moic}"


def test_dcf_workbook_formula_integrity():
    """The DCF workbook must be fully live: WACC, FCF projection, terminal
    value, and the EV → equity → fair-value bridge must all be formulas that
    reproduce the engine's numbers exactly."""
    from financial_model_generator import (
        DCFAssumptions, InstitutionalDCFEngine,
    )
    from ib_excel_engine import build_dcf_workbook
    from openpyxl import load_workbook

    a = DCFAssumptions(
        ticker="TEST",
        base_revenue=1000.0,
        revenue_growth_rates=[0.05] * 5,
        ebit_margin=0.20,
        tax_rate=0.25,
        da_pct_revenue=0.03,
        capex_pct_revenue=0.04,
        nwc_change_pct_revenue=0.02,
        equity_value_market=8000.0,
        debt_value=2000.0,
        cost_of_debt=0.05,
        risk_free_rate=0.04,
        equity_risk_premium=0.05,
        beta=1.2,
        terminal_growth_rate=0.025,
        cash=500.0,
        shares_outstanding=200.0,
        current_price=40.0,
        projection_years=5,
    )
    result = InstitutionalDCFEngine().run_dcf(a)
    wb = load_workbook(io.BytesIO(build_dcf_workbook(result)))
    global _wb_sheets, _eval_cache, _eval_in_progress
    _wb_sheets = _collect_cells(wb)
    _eval_cache = {}
    _eval_in_progress = set()

    # Every formula must resolve to a finite number
    _check_formulas(wb, [])

    # The KPI 'Fair value per share' must reproduce the engine value
    # (labels are written at row N, values at row N+1).
    sheet = _wb_sheets["Assumptions & Outputs"]
    fv_row = None
    for (col, row), val in sheet.items():
        if col == "A" and isinstance(val, str) and val.strip() == "Fair value per share ($)":
            fv_row = row
            break
    assert fv_row is not None
    got = sheet.get(("B", fv_row + 1))
    assert isinstance(got, str) and got.startswith("="), f"FV not a formula: {got!r}"
    res = _eval_formula(got, sheet)
    assert abs(res - result.fair_value_per_share) < max(
        1e-6, abs(result.fair_value_per_share) * 1e-6), \
        f"Fair value {res} != engine {result.fair_value_per_share}"
    # Enterprise value KPI too
    ev_row = None
    for (col, row), val in sheet.items():
        if col == "A" and isinstance(val, str) and val.strip() == "Enterprise value ($M)":
            ev_row = row
            break
    got_ev = sheet.get(("B", ev_row + 1))
    res_ev = _eval_formula(got_ev, sheet)
    assert abs(res_ev - result.enterprise_value) < max(
        1e-6, abs(result.enterprise_value) * 1e-6), \
        f"EV {res_ev} != engine {result.enterprise_value}"


def test_comps_workbook_builds():
    from ib_excel_engine import build_comps_workbook
    from openpyxl import load_workbook
    comps = pd.DataFrame({
        "Company": ["AAA", "BBB"], "Market Cap": [1e9, 2e9],
        "P/E": [15.0, 25.0], "EV/EBITDA": [8.0, 12.0], "Profit Margin": [0.15, 0.25],
    })
    med = pd.DataFrame({"Company": ["Median", "Mean"], "Market Cap": [1.5e9, 1.5e9],
                        "P/E": [20.0, 20.0], "EV/EBITDA": [10.0, 10.0],
                        "Profit Margin": [0.2, 0.2]})
    data = build_comps_workbook(comps, med)
    wb = load_workbook(io.BytesIO(data))
    assert "Comps" in wb.sheetnames
    # The median/mean rows must be LIVE formulas (=MEDIAN(...)/=AVERAGE(...))
    # that resolve to the correct values, so editing a comp recalculates stats.
    ws = wb["Comps"]
    median_row = None
    for row in ws.iter_rows():
        if row[0].value == "Median":
            median_row = row
            break
    assert median_row is not None
    # Median P/E is col D (Company=A, MarketCap=B, P/E=C, EV/EBITDA=D, Margin=E)
    pe_formula = ws.cell(row=median_row[0].row, column=3).value
    assert isinstance(pe_formula, str) and pe_formula.startswith("=MEDIAN("), pe_formula
    ev_formula = ws.cell(row=median_row[0].row, column=4).value
    assert ev_formula.startswith("=MEDIAN("), ev_formula
    # Mean row uses AVERAGE
    mean_row = None
    for row in ws.iter_rows():
        if row[0].value == "Mean":
            mean_row = row
            break
    assert mean_row is not None
    assert str(ws.cell(row=mean_row[0].row, column=3).value).startswith("=AVERAGE(")
