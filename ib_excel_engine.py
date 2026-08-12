"""
Institutional Excel Workbooks with LIVE Formulas

Generates Goldman-Sachs-standard, fully *editable* Excel workbooks for M&A and
LBO models. Every assumption is an input cell (yellow fill, blue font — the
conventional 'hardcode' style); every calculation is a live Excel formula
(black font) that recalculates when the analyst edits an input. Includes
sources = uses checks, contribution analysis, and color-scaled sensitivity
tables.

Author: Octavian Terminal
"""

import io
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.formatting.rule import ColorScaleRule
    _HAS_OPENPYXL = True
except ImportError:  # pragma: no cover
    _HAS_OPENPYXL = False

# ─────────────────────────────────────────────────────────────────────────────
# IB style constants (Goldman-standard conventions)
# ─────────────────────────────────────────────────────────────────────────────
IB_BLUE = "000080"       # header font
BLACK = "000000"         # formula text
WHITE = "FFFFFF"
INPUT_FILL = "FFFFCC"    # yellow — analyst-adjustable input
HEADER_FILL = "DCE6F1"   # light blue header band
TOTAL_FILL = "F2F2F2"    # totals band
SECTION_FONT = "1F4E79"

THIN = Side(style="thin", color="000000")
THICK_BOTTOM = Side(style="thick", color="000000")
DOUBLE_TOP = Side(style="thick", color="000000")

FMT_ACCT = '_($* #,##0.00_);_($* (#,##0.00);_($* "-"??_);_(@_)'
FMT_ACCT_INT = '_($* #,##0_);_($* (#,##0);_($* "-"??_);_(@_)'
FMT_PCT = '0.0%_)'
FMT_NUM = '_(* #,##0.00_);_(* (#,##0.00);_(* "-"??_);_(@_)'
FMT_INT = '_(* #,##0_);_(* (#,##0);_(* "-"??_);_(@_)'
FMT_MULT = '0.00"x"'


def _font(bold=False, italic=False, size=10, color=BLACK, name="Arial"):
    return Font(name=name, bold=bold, italic=italic, size=size, color=color)


def _fill(hex_color: str) -> PatternFill:
    return PatternFill(start_color=hex_color, end_color=hex_color, fill_type="solid")


def _align(horizontal: str = "right", vertical: str = "center", wrap: bool = False):
    return Alignment(horizontal=horizontal, vertical=vertical, wrap_text=wrap)


class IBSheet:
    """Thin wrapper over an openpyxl worksheet with IB conventions."""

    def __init__(self, ws, row_offset: int = 0):
        self.ws = ws
        ws.sheet_view.showGridLines = False
        self._r = row_offset  # current write row (0-based track)

    # -- low-level cell helpers ------------------------------------------------
    def _cell(self, row, col, value=None, font=None, fill=None, align=None,
              num_fmt=None, border=None):
        c = self.ws.cell(row=row, column=col, value=value)
        c.font = font or _font()
        if fill:
            c.fill = fill
        c.alignment = align or _align()
        if num_fmt:
            c.number_format = num_fmt
        if border:
            c.border = border
        return c

    def write(self, row, col, value, **kw):
        return self._cell(row, col, value, **kw)

    # -- semantic helpers -----------------------------------------------------
    def title(self, text: str, span_cols: int):
        self.ws.merge_cells(start_row=self._r + 1, start_column=1,
                            end_row=self._r + 1, end_column=span_cols)
        self.write(self._r + 1, 1, text, font=_font(bold=True, size=14),
                   align=_align("left"))
        self._r += 1

    def subtitle(self, text: str, span_cols: int):
        self.ws.merge_cells(start_row=self._r + 1, start_column=1,
                            end_row=self._r + 1, end_column=span_cols)
        self.write(self._r + 1, 1, text, font=_font(italic=True, size=9),
                   align=_align("left"))
        self._r += 1

    def blank(self, rows: int = 1):
        self._r += rows

    def section(self, text: str, span_cols: int):
        self.ws.merge_cells(start_row=self._r + 1, start_column=1,
                            end_row=self._r + 1, end_column=span_cols)
        self.write(self._r + 1, 1, text, font=_font(bold=True, size=10,
                                                    color=SECTION_FONT),
                   fill=_fill(HEADER_FILL), align=_align("left"))
        self._r += 1

    def label(self, text: str, indent: int = 0):
        self.write(self._r + 1, 1 + indent, text, font=_font(),
                   align=_align("left", wrap=True))
        self._r += 1

    def input(self, value, num_fmt: str = FMT_ACCT, indent: int = 0):
        """Yellow input cell (analyst-editable hardcode) in column B."""
        self.write(self._r + 1, 2 + indent, value, font=_font(color=IB_BLUE),
                   fill=_fill(INPUT_FILL), align=_align("right"), num_fmt=num_fmt)
        self._r += 1
        return self._r  # 1-based row just written

    def formula(self, formula: str, num_fmt: str = FMT_ACCT, indent: int = 0,
                bold: bool = False, fill: Optional[str] = None,
                border: Optional[Border] = None, row: Optional[int] = None):
        """Black formula cell (live calculation) in column B."""
        r = row if row is not None else self._r + 1
        self.write(r, 2 + indent, formula, font=_font(bold=bold),
                   fill=_fill(fill) if fill else None,
                   align=_align("right"), num_fmt=num_fmt,
                   border=border)
        if row is None:
            self._r += 1
        return r

    def formula_row(self, col, formula, num_fmt: str = FMT_ACCT, bold: bool = False,
                    fill: Optional[str] = None):
        self.write(self._r + 1, col, formula, font=_font(bold=bold),
                   fill=_fill(fill) if fill else None, align=_align("right"),
                   num_fmt=num_fmt)
        return self._r + 1

    def total(self, label: str, value, num_fmt: str = FMT_ACCT, indent: int = 0):
        self.write(self._r + 1, 1 + indent, label, font=_font(bold=True),
                   align=_align("left"))
        self.write(self._r + 1, 2 + indent, value, font=_font(bold=True),
                   fill=_fill(TOTAL_FILL), align=_align("right"),
                   num_fmt=num_fmt, border=Border(top=THIN, bottom=THICK_BOTTOM))
        self._r += 1
        return self._r

    def check(self, label: str, formula, num_fmt: str = FMT_ACCT):
        """Sources = Uses style check: green PASS / red FAIL via conditional?"""
        r = self.total(label, formula, num_fmt=num_fmt)
        return r

    def table(self, df, start_row=None, num_fmt: str = FMT_NUM,
              header_fill: str = HEADER_FILL, pct_cols: Optional[List[int]] = None,
              bold_last_col: bool = False, indent: int = 0):
        """Write a DataFrame as a formatted block. Returns last row written."""
        if start_row is None:
            start_row = self._r + 1
        pct_cols = pct_cols or []
        # Headers
        for j, col in enumerate(df.columns):
            self.write(start_row, 1 + indent + j, str(col),
                       font=_font(bold=True, color=IB_BLUE),
                       fill=_fill(header_fill), align=_align("center"),
                       border=Border(bottom=THICK_BOTTOM))
        # Data
        for i, (_, row) in enumerate(df.iterrows()):
            r = start_row + 1 + i
            for j, col in enumerate(df.columns):
                v = row[col]
                if v is None or (isinstance(v, float) and v != v):  # NaN
                    self.write(r, 1 + indent + j, "", font=_font())
                elif isinstance(v, (int, float)):
                    fmt = FMT_PCT if j in pct_cols else num_fmt
                    self.write(r, 1 + indent + j, v, font=_font(),
                               align=_align("right"), num_fmt=fmt)
                else:
                    self.write(r, 1 + indent + j, str(v), font=_font(),
                               align=_align("left" if j == 0 else "right"))
        last = start_row + len(df)
        self._r = last
        return last

    def autofit(self, ncols: int, min_width: int = 12, max_width: int = 34):
        for j in range(1, ncols + 1):
            letter = get_column_letter(j)
            best = min_width
            for row in self.ws.iter_rows(min_col=j, max_col=j):
                for cell in row:
                    if cell.value is None:
                        continue
                    s = str(cell.value)
                    best = max(best, min(len(s) + 3, max_width))
            self.ws.column_dimensions[letter].width = best

    def finish(self, freeze_panes: str = "A5", tab_color: str = "1F4E79",
               landscape: bool = True, fit_width: int = 1):
        """Apply institutional print + navigation polish to a worksheet.

        - freeze header rows so the model header stays visible while scrolling
        - tab colour so the workbook reads like a professional package
        - landscape print setup with fit-to-width (single page across)
        - confidential print footer on every printed page
        """
        self.ws.freeze_panes = freeze_panes
        self.ws.sheet_properties.tabColor = tab_color
        if landscape:
            self.ws.page_setup.orientation = "landscape"
        if fit_width:
            self.ws.page_setup.fitToWidth = fit_width
            self.ws.page_setup.fitToHeight = 0
            self.ws.sheet_properties.pageSetUpPr.fitToPage = True
        self.ws.page_margins.left = self.ws.page_margins.right = 0.3
        self.ws.page_margins.top = self.ws.page_margins.bottom = 0.4
        self.ws.oddFooter.center.text = (
            "Octavian Terminal  |  Confidential  |  Page &[Page] of &[Pages]"
        )
        self.ws.oddFooter.center.size = 8


# ─────────────────────────────────────────────────────────────────────────────
# M&A workbook
# ─────────────────────────────────────────────────────────────────────────────
def build_mna_workbook(result) -> bytes:
    """Build an editable M&A accretion/dilution workbook from MnAResult."""
    if not _HAS_OPENPYXL:  # pragma: no cover
        raise RuntimeError("openpyxl required for Excel export")
    a = result.assumptions
    wb = Workbook()
    ws = wb.active
    ws.title = "Assumptions"
    S = IBSheet(ws)
    S.title(f"M&A Accretion / Dilution Model  |  {a.acquirer_ticker} / {a.target_ticker}", 4)
    S.subtitle(f"Prepared by Octavian Terminal  |  {datetime.now().strftime('%b %d, %Y %H:%M')}", 4)
    S.blank()

    # ── Acquirer ─────────────────────────────────────────────────────────────
    S.section("ACQUIRER", 4)
    S.label("Share price ($)")
    r_acq_price = S.input(a.acquirer_price)
    S.label("EPS ($)")
    r_acq_eps = S.input(a.acquirer_eps)
    S.label("Shares outstanding (M)")
    r_acq_shares = S.input(a.acquirer_shares, num_fmt=FMT_ACCT_INT)
    S.label("Net income ($M)")
    r_acq_ni = S.formula(f"=B{r_acq_eps}*B{r_acq_shares}")  # EPS * shares
    S.blank()

    # ── Target ───────────────────────────────────────────────────────────────
    S.section("TARGET", 4)
    S.label("Share price ($)")
    r_tgt_price = S.input(a.target_price)
    S.label("EPS ($)")
    r_tgt_eps = S.input(a.target_eps)
    S.label("Shares outstanding (M)")
    r_tgt_shares = S.input(a.target_shares, num_fmt=FMT_ACCT_INT)
    S.label("Net income ($M)")
    r_tgt_ni = S.formula(f"=B{r_tgt_eps}*B{r_tgt_shares}")
    S.blank()

    # ── Deal terms ───────────────────────────────────────────────────────────
    S.section("DEAL TERMS & FINANCING", 4)
    S.label("Offer premium (%)")
    r_prem = S.input(a.offer_premium, num_fmt=FMT_PCT)
    S.label("Stock consideration (%)")
    r_stock = S.input(a.percent_stock, num_fmt=FMT_PCT)
    S.label("Cash consideration (%)")
    r_cash = S.input(a.percent_cash, num_fmt=FMT_PCT)
    S.label("Pre-tax cost of debt (%)")
    r_debt_cost = S.input(a.cost_of_debt, num_fmt=FMT_PCT)
    S.label("Tax rate (%)")
    r_tax = S.input(a.tax_rate, num_fmt=FMT_PCT)
    S.label("Pre-tax synergies ($M)")
    r_syn = S.input(a.pre_tax_synergies)
    S.label("One-time integration costs ($M)")
    r_int_cost = S.input(a.integration_costs)
    S.label("Advisory fee (% of deal value)")
    r_adv = S.input(a.advisory_fee_pct, num_fmt=FMT_PCT)
    S.label("Financing fee (% of debt raised)")
    r_fin = S.input(a.financing_fee_pct, num_fmt=FMT_PCT)
    S.blank()

    # ── Key outputs (live formulas) ──────────────────────────────────────────
    S.section("KEY OUTPUTS (LIVE FORMULAS)", 4)
    S.label("Offer price per share ($)")
    r_offer = S.formula(f"=B{r_tgt_price}*(1+B{r_prem})")
    S.label("Total deal value ($M)")
    r_deal = S.formula(f"=B{r_offer}*B{r_tgt_shares}")
    S.label("Equity consideration ($M)")
    r_eq = S.formula(f"=B{r_deal}*B{r_stock}")
    S.label("Cash consideration ($M)")
    r_cash_amt = S.formula(f"=B{r_deal}*B{r_cash}")
    S.label("Advisory fee ($M)")
    r_adv_fee = S.formula(f"=B{r_deal}*B{r_adv}")
    S.label("Financing fee ($M)")
    r_fin_fee = S.formula(f"=B{r_cash_amt}*B{r_fin}")
    S.label("New shares issued (M)")
    r_new_shares = S.formula(f"=B{r_eq}/B{r_acq_price}", num_fmt=FMT_ACCT_INT)
    S.label("Pro forma shares (M)")
    r_pf_shares = S.formula(f"=B{r_acq_shares}+B{r_new_shares}", num_fmt=FMT_ACCT_INT)
    S.label("After-tax synergies ($M)")
    r_syn_at = S.formula(f"=B{r_syn}*(1-B{r_tax})")
    S.label("After-tax integration costs ($M)")
    r_int_at = S.formula(f"=B{r_int_cost}*(1-B{r_tax})")
    S.label("After-tax interest on new debt ($M)")
    r_int = S.formula(f"=B{r_cash_amt}*B{r_debt_cost}*(1-B{r_tax})")
    S.label("Pro forma net income ($M)")
    r_pf_ni = S.formula(
        f"=B{r_acq_ni}+B{r_tgt_ni}+B{r_syn_at}-B{r_int_at}-B{r_int}")
    S.label("Pro forma EPS ($)")
    r_pf_eps = S.formula(f"=B{r_pf_ni}/B{r_pf_shares}", bold=True)
    S.label("Accretion / (dilution) per share ($)")
    r_acc = S.formula(f"=B{r_pf_eps}-B{r_acq_eps}", bold=True)
    S.label("Accretion / (dilution) (%)")
    S.formula(f"=B{r_acc}/B{r_acq_eps}", FMT_PCT, bold=True,
              fill="E2EFDA" if result.is_accretive else "FCE4EC")

    S.autofit(4)

    # ── Contribution Analysis sheet ─────────────────────────────────────────
    ws_c = wb.create_sheet("Contribution Analysis")
    C = IBSheet(ws_c)
    C.title("Accretion / Dilution EPS Bridge", 3)
    C.subtitle("Per-share contribution of each driver (live formulas)", 3)
    C.blank()
    C.write(4, 1, "Driver", font=_font(bold=True, color=IB_BLUE),
            fill=_fill(HEADER_FILL), align=_align("center"), border=Border(bottom=THICK_BOTTOM))
    C.write(4, 2, "EPS Impact ($)", font=_font(bold=True, color=IB_BLUE),
            fill=_fill(HEADER_FILL), align=_align("center"), border=Border(bottom=THICK_BOTTOM))
    C.write(4, 3, "Cumulative EPS ($)", font=_font(bold=True, color=IB_BLUE),
            fill=_fill(HEADER_FILL), align=_align("center"), border=Border(bottom=THICK_BOTTOM))
    contrib = result.contribution_analysis
    pf_ref = f"Assumptions!B{r_pf_shares}"
    # Live per-driver formulas so the bridge recalculates when assumptions change
    driver_formulas = [
        f"=Assumptions!B{r_tgt_ni}/{pf_ref}",                       # target NI (P/E effect)
        f"=Assumptions!B{r_syn_at}/{pf_ref}",                       # synergies
        f"=-Assumptions!B{r_int_at}/{pf_ref}",                      # integration costs
        f"=-Assumptions!B{r_int}/{pf_ref}",                         # financing effect
        f"=Assumptions!B{r_acq_ni}/{pf_ref}-Assumptions!B{r_acq_eps}",  # share-count dilution
    ]
    for i, (_, r) in enumerate(contrib.iterrows()):
        row = 5 + i
        C.write(row, 1, r["Driver"], font=_font(), align=_align("left"))
        if r["Type"] == "Result":
            pf_f = f"=Assumptions!B{r_pf_eps}"
            C.write(row, 2, pf_f, font=_font(bold=True),
                    align=_align("right"), num_fmt=FMT_ACCT)
            C.write(row, 3, pf_f, font=_font(bold=True),
                    align=_align("right"), num_fmt=FMT_ACCT,
                    border=Border(top=THIN, bottom=THICK_BOTTOM))
        else:
            f = driver_formulas[i] if i < len(driver_formulas) else "=0"
            C.write(row, 2, f, font=_font(), align=_align("right"), num_fmt=FMT_ACCT)
            cum_f = (f"=Assumptions!B{r_acq_eps}+B{row}" if i == 0
                     else f"=C{row-1}+B{row}")
            C.write(row, 3, cum_f, font=_font(), align=_align("right"),
                    num_fmt=FMT_ACCT)
    C.autofit(3)

    # ── Sensitivity sheet ────────────────────────────────────────────────────
    ws_s = wb.create_sheet("Sensitivity")
    Sn = IBSheet(ws_s)
    Sn.title("Accretion / (Dilution) % Sensitivity", 6)
    Sn.subtitle("Row = offer premium   |   Column = % stock consideration", 6)
    Sn.blank()
    start = Sn._r + 1
    last = Sn.table(result.sensitivity.premium_stock_table, start_row=start,
                    num_fmt=FMT_NUM, pct_cols=list(range(1, 6)))
    _apply_color_scale(ws_s, start + 1, last, 2, 6)
    Sn.blank(2)
    Sn.subtitle("Row = offer premium   |   Column = synergy level (multiple of base)", 6)
    Sn.blank()
    start2 = Sn._r + 1
    last2 = Sn.table(result.sensitivity.premium_synergy_table, start_row=start2,
                     num_fmt=FMT_NUM, pct_cols=list(range(1, 4)))
    _apply_color_scale(ws_s, start2 + 1, last2, 2, 4)
    if result.sensitivity.breakeven_synergies is not None:
        Sn.blank()
        Sn.label(f"Breakeven pre-tax synergies: ${result.sensitivity.breakeven_synergies:,.0f}M")
    Sn.autofit(6)

    # institutional polish: freeze headers, tab colours, landscape print
    for _ws in wb.worksheets:
        IBSheet(_ws).finish(freeze_panes="A5")

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.getvalue()


def _find_label_row(ws, text: str) -> int:
    """Find the 1-based row of a label cell in column A."""
    for row in ws.iter_rows(min_col=1, max_col=1):
        cell = row[0]
        if cell.value and str(cell.value).strip() == text.strip():
            return cell.row
    return 2


def _find_input(ws, label: str) -> int:
    """Return the 1-based row of the *input value* (col B) for a label."""
    return _find_label_row(ws, label) + 1


def _apply_color_scale(ws, first_row: int, last_row: int, first_col: int, last_col: int):
    """Green→yellow→red color scale (standard for sensitivity tables)."""
    ws.conditional_formatting.add(
        f"{get_column_letter(first_col)}{first_row}:{get_column_letter(last_col)}{last_row}",
        ColorScaleRule(
            start_type="min", start_color="63BE7B",
            mid_type="percentile", mid_value=50, mid_color="FFEB84",
            end_type="max", end_color="F8696B",
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# LBO workbook
# ─────────────────────────────────────────────────────────────────────────────
def build_lbo_workbook(result) -> bytes:
    """Build an editable LBO workbook from LBOResult.

    The Projections sheet mirrors the engine's math exactly: operating model,
    Term Loan B with mandatory amortization + excess cash flow sweep, revolver
    (draw when CFADS is short, repay otherwise), and a minimum-cash floor.
    Every yellow cell is an analyst input; every black cell is a live formula
    so the analyst can change assumptions and watch IRR / MOIC recalculate.
    """
    if not _HAS_OPENPYXL:  # pragma: no cover
        raise RuntimeError("openpyxl required for Excel export")
    a = result.assumptions
    years = max(a.exit_year - a.entry_year, 1)
    wb = Workbook()
    ws = wb.active
    ws.title = "Sources & Uses"
    S = IBSheet(ws)
    S.title(f"Leveraged Buyout  |  {a.ticker or a.target_name}", 4)
    S.subtitle(f"Prepared by Octavian Terminal  |  {datetime.now().strftime('%b %d, %Y %H:%M')}", 4)
    S.blank()

    # Sources & Uses (equity is the plug → the check always balances by
    # construction, which is exactly how a live model should behave).
    S.section("SOURCES & USES", 4)
    S.label("Purchase price ($M)")
    r_pp = S.input(result.purchase_price)
    S.label("Transaction fees ($M)")
    r_fees = S.input(result.total_uses - result.purchase_price)
    S.total("Total Uses", f"=B{r_pp}+B{r_fees}", FMT_ACCT)
    r_uses_total = S._r
    S.blank()
    S.label("Debt raised (TLB) ($M)")
    r_debt = S.input(result.debt_amount)
    S.label("Sponsor equity (check) ($M)")
    r_sponsor_eq = S.formula(f"=B{r_uses_total}-B{r_debt}")  # plug
    S.total("Total Sources", f"=B{r_debt}+B{r_sponsor_eq}", FMT_ACCT)
    r_sources_total = S._r
    S.blank()
    S.check("Sources = Uses check ($M)",
            f"=B{r_sources_total}-B{r_uses_total}", FMT_ACCT)
    S.autofit(4)

    # ── Operating Projections & Debt Schedule ────────────────────────────────
    # Row layout (relative to header row hdr):
    #   hdr+1  LTM EBITDA ($M)                 [INPUT]
    #   hdr+2  Base revenue (prior year)       [FORMULA = LTM EBITDA / margin]
    #   hdr+3  EBITDA margin (%)               [INPUT x years]
    #   hdr+4  Revenue growth (%)              [INPUT x years]
    #   hdr+5  TLB interest rate (%)           [INPUT]
    #   hdr+6  Revolver rate (%)               [INPUT]
    #   hdr+7  Cash interest rate (%)          [INPUT]
    #   hdr+8  Tax rate (%)                    [INPUT]
    #   hdr+9  Minimum cash (% of revenue)     [INPUT]
    #   hdr+10 CapEx (% of revenue)            [INPUT]
    #   hdr+11 Change in NWC (% of revenue)    [INPUT]
    #   hdr+12 D&A (% of revenue)              [INPUT]
    #   hdr+13 Mandatory amortization (% orig) [INPUT]
    #   hdr+14 Excess cash flow sweep (%)      [INPUT]
    #   hdr+15 Original TLB principal ($M)     [INPUT]
    #   hdr+16 Revenue                         [FORMULA]
    #   hdr+17 EBITDA                          [FORMULA]
    #   hdr+18 (-) D&A                         [FORMULA]
    #   hdr+19 EBIT                            [FORMULA]
    #   hdr+20 (-) Interest expense            [FORMULA]
    #   hdr+21 (+) Interest income             [FORMULA]
    #   hdr+22 Pre-tax income                  [FORMULA]
    #   hdr+23 (-) Taxes                       [FORMULA]
    #   hdr+24 Net income                      [FORMULA]
    #   hdr+25 CFADS                           [FORMULA]
    #   hdr+26 (-) Mandatory amortization      [FORMULA]
    #   hdr+27 (-) Excess cash flow sweep      [FORMULA]
    #   hdr+28 Beginning TLB balance           [B = =orig; C+ = prior ending]
    #   hdr+29 Ending TLB balance              [FORMULA]
    #   hdr+30 Beginning revolver balance      [B = 0 input; C+ = prior ending]
    #   hdr+31 Ending revolver balance         [FORMULA]
    #   hdr+32 Beginning cash balance          [B = base rev * min cash; C+ = prior ending]
    #   hdr+33 Ending cash balance             [FORMULA]
    # Year columns: year j (0-based) = column (2 + j)  →  year 1 = B.
    ws_p = wb.create_sheet("Projections")
    P = IBSheet(ws_p)
    P.title(f"Operating Model & Debt Schedule  |  {a.entry_year} - {a.exit_year}", 3 + years)
    P.subtitle("Yellow cells are analyst inputs; all other figures recalculate live", 3 + years)
    P.blank()

    proj = result.cash_flows
    rev_growth = proj["Revenue Growth %"].tolist() if "Revenue Growth %" in proj else [0.05] * years
    margins = proj["EBITDA Margin %"].tolist() if "EBITDA Margin %" in proj else [0.20] * years
    margin0 = margins[0] if margins and margins[0] > 0 else 0.20
    rev0 = a.ltm_ebitda / margin0

    hdr = P._r + 1  # row 4
    P.write(hdr, 1, "Line Item", font=_font(bold=True, color=IB_BLUE),
            fill=_fill(HEADER_FILL), align=_align("left"), border=Border(bottom=THICK_BOTTOM))
    for j in range(years):
        P.write(hdr, 2 + j, str(proj.iloc[j]["Year"]),
                font=_font(bold=True, color=IB_BLUE), fill=_fill(HEADER_FILL),
                align=_align("center"), border=Border(bottom=THICK_BOTTOM))
    P._r = hdr

    r_ltm = hdr + 1
    r_base = hdr + 2
    r_margin = hdr + 3
    r_growth = hdr + 4
    r_rate = hdr + 5
    r_rev_rate = hdr + 6
    r_cash_rate = hdr + 7
    r_tax = hdr + 8
    r_mincash = hdr + 9
    r_capex = hdr + 10
    r_nwc = hdr + 11
    r_dep = hdr + 12
    r_mand_pct = hdr + 13
    r_sweep_pct = hdr + 14
    r_orig = hdr + 15
    r_rev = hdr + 16
    r_ebitda = hdr + 17
    r_da = hdr + 18
    r_ebit = hdr + 19
    r_int_exp = hdr + 20
    r_int_inc = hdr + 21
    r_ebt = hdr + 22
    r_taxes = hdr + 23
    r_ni = hdr + 24
    r_cfads = hdr + 25
    r_mand = hdr + 26
    r_sweep = hdr + 27
    r_tlb_beg = hdr + 28
    r_tlb_end = hdr + 29
    r_rev_beg = hdr + 30
    r_rev_end = hdr + 31
    r_cash_beg = hdr + 32
    r_cash_end = hdr + 33

    def col(j):
        return get_column_letter(2 + j)  # year j (0-based) => column B..

    def beg_of(r_beg, r_end, j):
        """Beginning balance ref for year j (input col B, else prior ending)."""
        return f"B{r_beg}" if j == 0 else f"{col(j-1)}{r_end}"

    def prev_rev(j):
        return f"B{r_base}" if j == 0 else f"{col(j-1)}{r_rev}"

    def pct_row(row, label, value, fmt=FMT_PCT):
        P.write(row, 1, label, font=_font(color=IB_BLUE), align=_align("left"))
        P.write(row, 2, value, font=_font(color=IB_BLUE), fill=_fill(INPUT_FILL),
                align=_align("right"), num_fmt=fmt)

    # Inputs
    P.write(r_ltm, 1, "LTM EBITDA ($M)", font=_font(color=IB_BLUE), align=_align("left"))
    P.write(r_ltm, 2, a.ltm_ebitda, font=_font(color=IB_BLUE), fill=_fill(INPUT_FILL),
            align=_align("right"), num_fmt=FMT_ACCT_INT)
    P.write(r_base, 1, "Base revenue (prior year) ($M)", font=_font(color=IB_BLUE),
            align=_align("left"))
    P.write(r_base, 2, f"=B{r_ltm}/B{r_margin}", font=_font(),
            align=_align("right"), num_fmt=FMT_ACCT_INT)
    P.write(r_margin, 1, "EBITDA margin (%)", font=_font(color=IB_BLUE), align=_align("left"))
    for j in range(years):
        P.write(r_margin, 2 + j, margins[j], font=_font(color=IB_BLUE),
                fill=_fill(INPUT_FILL), align=_align("right"), num_fmt=FMT_PCT)
    P.write(r_growth, 1, "Revenue growth (%)", font=_font(color=IB_BLUE), align=_align("left"))
    for j in range(years):
        P.write(r_growth, 2 + j, rev_growth[j], font=_font(color=IB_BLUE),
                fill=_fill(INPUT_FILL), align=_align("right"), num_fmt=FMT_PCT)
    pct_row(r_rate, "TLB interest rate (%)", a.interest_rate)
    pct_row(r_rev_rate, "Revolver rate (%)", a.revolver_rate)
    pct_row(r_cash_rate, "Cash interest rate (%)", a.cash_interest_rate)
    pct_row(r_tax, "Tax rate (%)", a.tax_rate)
    pct_row(r_mincash, "Minimum cash (% of revenue)", a.min_cash_pct_rev)
    pct_row(r_capex, "CapEx (% of revenue)", a.capex_pct_rev)
    pct_row(r_nwc, "Change in NWC (% of revenue)", a.nwc_pct_rev)
    pct_row(r_dep, "D&A (% of revenue)", a.depreciation_pct_rev)
    pct_row(r_mand_pct, "Mandatory amortization (% of original)", a.mandatory_amort_pct)
    pct_row(r_sweep_pct, "Excess cash flow sweep (%)", a.cash_sweep_pct)
    P.write(r_orig, 1, "Original TLB principal ($M)", font=_font(color=IB_BLUE),
            align=_align("left"))
    P.write(r_orig, 2, result.debt_amount, font=_font(color=IB_BLUE),
            fill=_fill(INPUT_FILL), align=_align("right"), num_fmt=FMT_ACCT_INT)

    # Revenue: Y1 = base*(1+g1); then prior*(1+g)
    P.write(r_rev, 1, "Revenue ($M)", font=_font(), align=_align("left"))
    for j in range(years):
        P.write(r_rev, 2 + j, f"={prev_rev(j)}*(1+{col(j)}{r_growth})",
                font=_font(), align=_align("right"), num_fmt=FMT_ACCT_INT)

    P.write(r_ebitda, 1, "EBITDA ($M)", font=_font(), align=_align("left"))
    for j in range(years):
        P.write(r_ebitda, 2 + j, f"={col(j)}{r_rev}*{col(j)}{r_margin}",
                font=_font(), align=_align("right"), num_fmt=FMT_ACCT_INT)

    P.write(r_da, 1, "(-) D&A ($M)", font=_font(), align=_align("left"))
    for j in range(years):
        P.write(r_da, 2 + j, f"=-{col(j)}{r_rev}*B{r_dep}",
                font=_font(), align=_align("right"), num_fmt=FMT_ACCT_INT)

    P.write(r_ebit, 1, "EBIT ($M)", font=_font(), align=_align("left"))
    for j in range(years):
        P.write(r_ebit, 2 + j, f"={col(j)}{r_ebitda}+{col(j)}{r_da}",
                font=_font(), align=_align("right"), num_fmt=FMT_ACCT_INT)

    # Interest on *beginning* balances (TLB + revolver), less cash interest
    P.write(r_int_exp, 1, "(-) Interest expense ($M)", font=_font(), align=_align("left"))
    for j in range(years):
        beg_t = beg_of(r_tlb_beg, r_tlb_end, j)
        beg_r = beg_of(r_rev_beg, r_rev_end, j)
        P.write(r_int_exp, 2 + j,
                f"=-{beg_t}*B{r_rate}-{beg_r}*B{r_rev_rate}",
                font=_font(), align=_align("right"), num_fmt=FMT_ACCT_INT)
    P.write(r_int_inc, 1, "(+) Interest income ($M)", font=_font(), align=_align("left"))
    for j in range(years):
        beg_c = beg_of(r_cash_beg, r_cash_end, j)
        P.write(r_int_inc, 2 + j, f"={beg_c}*B{r_cash_rate}",
                font=_font(), align=_align("right"), num_fmt=FMT_ACCT_INT)

    P.write(r_ebt, 1, "Pre-tax income ($M)", font=_font(), align=_align("left"))
    for j in range(years):
        P.write(r_ebt, 2 + j,
                f"={col(j)}{r_ebit}+{col(j)}{r_int_exp}+{col(j)}{r_int_inc}",
                font=_font(), align=_align("right"), num_fmt=FMT_ACCT_INT)

    P.write(r_taxes, 1, "(-) Taxes ($M)", font=_font(), align=_align("left"))
    for j in range(years):
        P.write(r_taxes, 2 + j, f"=-MAX({col(j)}{r_ebt},0)*B{r_tax}",
                font=_font(), align=_align("right"), num_fmt=FMT_ACCT_INT)

    P.write(r_ni, 1, "Net income ($M)", font=_font(bold=True), align=_align("left"))
    for j in range(years):
        P.write(r_ni, 2 + j, f"={col(j)}{r_ebt}+{col(j)}{r_taxes}",
                font=_font(bold=True), align=_align("right"), num_fmt=FMT_ACCT_INT,
                border=Border(bottom=THICK_BOTTOM))

    # CFADS = NI + D&A add-back - CapEx - change in NWC. The D&A row is
    # stored negative, so the add-back is expressed as subtraction: NI - DA.
    P.write(r_cfads, 1, "CFADS ($M)", font=_font(), align=_align("left"))
    for j in range(years):
        P.write(r_cfads, 2 + j,
                f"={col(j)}{r_ni}-{col(j)}{r_da}-{col(j)}{r_rev}*B{r_capex}"
                f"-({col(j)}{r_rev}-{prev_rev(j)})*B{r_nwc}",
                font=_font(), align=_align("right"), num_fmt=FMT_ACCT_INT)

    # Mandatory amortization: 1% of ORIGINAL TLB, capped at current balance
    P.write(r_mand, 1, "(-) Mandatory amortization ($M)", font=_font(), align=_align("left"))
    for j in range(years):
        beg_t = beg_of(r_tlb_beg, r_tlb_end, j)
        P.write(r_mand, 2 + j, f"=-MIN(B{r_orig}*B{r_mand_pct},{beg_t})",
                font=_font(), align=_align("right"), num_fmt=FMT_ACCT_INT)

    # Sweep: 50% of excess cash flow, capped at balance after amortization
    P.write(r_sweep, 1, "(-) Excess cash flow sweep ($M)", font=_font(), align=_align("left"))
    for j in range(years):
        beg_t = beg_of(r_tlb_beg, r_tlb_end, j)
        P.write(r_sweep, 2 + j,
                f"=-MIN(MAX({col(j)}{r_cfads}+{col(j)}{r_mand},0)*B{r_sweep_pct},"
                f"{beg_t}+{col(j)}{r_mand})",
                font=_font(), align=_align("right"), num_fmt=FMT_ACCT_INT)

    P.write(r_tlb_beg, 1, "Beginning TLB balance ($M)", font=_font(color=IB_BLUE),
            align=_align("left"))
    for j in range(years):
        P.write(r_tlb_beg, 2 + j, f"=B{r_orig}" if j == 0 else f"={col(j-1)}{r_tlb_end}",
                font=_font(color=IB_BLUE), fill=_fill(INPUT_FILL),
                align=_align("right"), num_fmt=FMT_ACCT_INT)
    P.write(r_tlb_end, 1, "Ending TLB balance ($M)", font=_font(bold=True), align=_align("left"))
    for j in range(years):
        beg_t = beg_of(r_tlb_beg, r_tlb_end, j)
        P.write(r_tlb_end, 2 + j, f"={beg_t}+{col(j)}{r_mand}+{col(j)}{r_sweep}",
                font=_font(bold=True), align=_align("right"), num_fmt=FMT_ACCT_INT)

    P.write(r_rev_beg, 1, "Beginning revolver balance ($M)", font=_font(color=IB_BLUE),
            align=_align("left"))
    for j in range(years):
        if j == 0:
            P.write(r_rev_beg, 2, 0.0, font=_font(color=IB_BLUE), fill=_fill(INPUT_FILL),
                    align=_align("right"), num_fmt=FMT_ACCT_INT)
        else:
            P.write(r_rev_beg, 2 + j, f"={col(j-1)}{r_rev_end}",
                    font=_font(color=IB_BLUE), fill=_fill(INPUT_FILL),
                    align=_align("right"), num_fmt=FMT_ACCT_INT)
    # Ending revolver = MAX(beg - cash_after_debt_service, 0): draws when CFADS
    # is short (cash_after < 0), repays when there is surplus.
    P.write(r_rev_end, 1, "Ending revolver balance ($M)", font=_font(), align=_align("left"))
    for j in range(years):
        beg_r = beg_of(r_rev_beg, r_rev_end, j)
        P.write(r_rev_end, 2 + j,
                f"=MAX({beg_r}-({col(j)}{r_cfads}+{col(j)}{r_mand}+{col(j)}{r_sweep}),0)",
                font=_font(), align=_align("right"), num_fmt=FMT_ACCT_INT)

    P.write(r_cash_beg, 1, "Beginning cash balance ($M)", font=_font(color=IB_BLUE),
            align=_align("left"))
    for j in range(years):
        if j == 0:
            P.write(r_cash_beg, 2, f"=B{r_base}*B{r_mincash}",
                    font=_font(color=IB_BLUE), fill=_fill(INPUT_FILL),
                    align=_align("right"), num_fmt=FMT_ACCT_INT)
        else:
            P.write(r_cash_beg, 2 + j, f"={col(j-1)}{r_cash_end}",
                    font=_font(color=IB_BLUE), fill=_fill(INPUT_FILL),
                    align=_align("right"), num_fmt=FMT_ACCT_INT)
    # Cash grows by post-revolver surplus, never below the min-cash floor
    P.write(r_cash_end, 1, "Ending cash balance ($M)", font=_font(), align=_align("left"))
    for j in range(years):
        beg_c = beg_of(r_cash_beg, r_cash_end, j)
        beg_r = beg_of(r_rev_beg, r_rev_end, j)
        P.write(r_cash_end, 2 + j,
                f"=MAX({beg_c}+MAX({col(j)}{r_cfads}+{col(j)}{r_mand}+{col(j)}{r_sweep}-{beg_r},0),"
                f"{col(j)}{r_rev}*B{r_mincash})",
                font=_font(), align=_align("right"), num_fmt=FMT_ACCT_INT)

    P.autofit(2 + years)

    # ── Returns sheet ────────────────────────────────────────────────────────
    exit_col = col(years - 1)
    ws_r = wb.create_sheet("Returns")
    R = IBSheet(ws_r)
    R.title("Sponsor Returns", 4)
    R.subtitle("IRR / MOIC at exit (live formulas)", 4)
    R.blank()
    R.section("EXIT", 4)
    R.label("Exit multiple (x)")
    r_x_mult = R.input(a.exit_multiple, num_fmt=FMT_MULT)
    R.label("Exit enterprise value ($M)")
    r_x_ev = R.formula(f"=B{r_x_mult}*Projections!{exit_col}{r_ebitda}", FMT_ACCT_INT)
    R.label("(-) Total debt at exit ($M)")
    r_x_debt = R.formula(
        f"=-Projections!{exit_col}{r_tlb_end}-Projections!{exit_col}{r_rev_end}",
        FMT_ACCT_INT)
    R.label("(+) Cash at exit ($M)")
    r_x_cash = R.formula(f"=Projections!{exit_col}{r_cash_end}", FMT_ACCT_INT)
    R.label("Exit equity value ($M)")
    r_x_eq = R.formula(f"=B{r_x_ev}+B{r_x_debt}+B{r_x_cash}", FMT_ACCT_INT, bold=True)
    R.blank()
    R.section("INITIAL INVESTMENT", 4)
    R.label("Sponsor equity invested ($M)")
    r_sponsor = R.formula(f"='Sources & Uses'!B{r_sponsor_eq}")
    R.label("Hold period (years)")
    r_hold = R.input(years, num_fmt=FMT_INT)
    R.blank()
    R.section("RETURNS", 4)
    R.label("MOIC (x)")
    r_moic = R.formula(f"=B{r_x_eq}/B{r_sponsor}", FMT_MULT, bold=True)
    R.label("IRR (%)")
    R.formula(f"=(B{r_moic})^(1/B{r_hold})-1", FMT_PCT, bold=True,
              fill="E2EFDA" if result.irr >= 0.20 else "FCE4EC")
    R.autofit(4)

    # institutional polish: freeze headers, tab colours, landscape print
    for _ws in wb.worksheets:
        IBSheet(_ws).finish(freeze_panes="A5")

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.getvalue()


def build_comps_workbook(comps_df, median_df=None, title: str = "Comparable Company Analysis") -> bytes:
    """Build an editable comps workbook with LIVE MEDIAN / MEAN formulas.

    The stats rows are real =MEDIAN(...)/=AVERAGE(...) formulas over the comp
    rows, so the analyst can adjust any comp's multiple and the median/mean
    recalculate automatically (institutional convention).
    """
    if not _HAS_OPENPYXL:  # pragma: no cover
        raise RuntimeError("openpyxl required for Excel export")
    wb = Workbook()
    ws = wb.active
    ws.title = "Comps"
    S = IBSheet(ws)
    S.title(title, len(comps_df.columns))
    S.subtitle(f"Prepared by Octavian Terminal  |  {datetime.now().strftime('%b %d, %Y %H:%M')}",
               len(comps_df.columns))
    S.blank()
    cols = list(comps_df.columns)
    # Header
    hdr = S._r + 1
    for j, c in enumerate(cols):
        S.write(hdr, 1 + j, str(c), font=_font(bold=True, color=IB_BLUE),
                fill=_fill(HEADER_FILL), align=_align("center"),
                border=Border(bottom=THICK_BOTTOM))
    S._r = hdr
    # Data (strings preserved; numbers formatted). Track first/last data rows
    # per column so MEDIAN/MEAN formulas can target exactly the comp rows.
    pct_cols = [j for j, c in enumerate(cols) if any(
        k in str(c).lower() for k in ("margin", "yield", "growth", "return", "premium"))]
    int_cols = [j for j, c in enumerate(cols) if any(
        k in str(c).lower() for k in ("cap", "value", "shares", "debt", "cash"))]
    numeric_cols = {}
    for i, (_, row) in enumerate(comps_df.iterrows()):
        S._r += 1
        for j, c in enumerate(cols):
            v = row[c]
            if isinstance(v, (int, float)) and not isinstance(v, bool) and not (
                    isinstance(v, float) and v != v):
                numeric_cols.setdefault(j, []).append(S._r)
                fmt = FMT_PCT if j in pct_cols else (FMT_ACCT_INT if j in int_cols else FMT_NUM)
                S.write(S._r, 1 + j, float(v), font=_font(), align=_align("right"), num_fmt=fmt)
            else:
                S.write(S._r, 1 + j, "" if v is None else str(v), font=_font(),
                        align=_align("right" if j else "left"))
    # LIVE MEDIAN / MEAN statistics rows
    if median_df is not None:
        S.blank()
        for _, row in median_df.iterrows():
            S._r += 1
            stat_name = str(row.get("Symbol") or row.get("Company") or "")
            is_median = "median" in stat_name.lower()
            S.write(S._r, 1, stat_name, font=_font(bold=True), fill=_fill(TOTAL_FILL),
                    align=_align("left"))
            for j, c in enumerate(cols):
                if j == 0:
                    continue
                rows_list = numeric_cols.get(j, [])
                if rows_list:
                    rng = f"{get_column_letter(1 + j)}{rows_list[0]}:" \
                          f"{get_column_letter(1 + j)}{rows_list[-1]}"
                    fn = "MEDIAN" if is_median else "AVERAGE"
                    fmt = FMT_PCT if j in pct_cols else \
                          (FMT_ACCT_INT if j in int_cols else FMT_NUM)
                    S.write(S._r, 1 + j, f"={fn}({rng})", font=_font(bold=True),
                            fill=_fill(TOTAL_FILL), align=_align("right"), num_fmt=fmt)
                else:
                    S.write(S._r, 1 + j, "", font=_font(bold=True),
                            fill=_fill(TOTAL_FILL))
    S.autofit(len(cols))
    # institutional polish: freeze headers, tab colours, landscape print
    for _ws in wb.worksheets:
        IBSheet(_ws).finish(freeze_panes="A5")
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.getvalue()


def build_dcf_workbook(result) -> bytes:
    """Build an editable DCF workbook with LIVE formulas.

    Mirrors the engine's math exactly: WACC (CAPM + weights), 5-year FCF
    projection, Gordon-growth terminal value, enterprise value, net debt,
    equity value, and fair value per share. Every assumption is a yellow
    input cell; every figure is a black live formula, so the analyst can
    change WACC / growth / margins and watch valuation recalculate — exactly
    what an institutional DCF deliverable looks like.
    """
    if not _HAS_OPENPYXL:  # pragma: no cover
        raise RuntimeError("openpyxl required for Excel export")
    a = result.assumptions
    years = int(getattr(a, "projection_years", 5) or 5)
    g = list(a.revenue_growth_rates or [])
    while len(g) < years:
        g.append(g[-1] if g else 0.05)
    g = g[:years]

    wb = Workbook()
    ws = wb.active
    ws.title = "Assumptions & Outputs"
    S = IBSheet(ws)
    S.title(f"Discounted Cash Flow  |  {result.ticker}", 3 + years)
    S.subtitle(f"Prepared by Octavian Terminal  |  {datetime.now().strftime('%b %d, %Y %H:%M')}", 3 + years)
    S.blank()

    def col(j):
        return get_column_letter(2 + j)  # year j (0-based) => column B..

    # ── Key outputs (live formulas, forward refs to sections below) ─────────
    S.section("KEY OUTPUTS (LIVE FORMULAS)", 3 + years)
    # Forward-referenced KPIs — final formulas are patched in after the
    # valuation bridge is laid out (rows unknown at this point).
    S.label("Enterprise value ($M)")
    r_ev = S.formula("=0")
    S.label("(-) Net debt ($M)")
    r_nd = S.formula("=0")
    S.label("Equity value ($M)")
    r_eq = S.formula("=0")
    S.label("Fair value per share ($)")
    r_fv = S.formula("=0")
    S.label("Current price ($)")
    r_px = S.input(a.current_price)
    S.label("Upside / (downside) (%)")
    S.formula(f"=B{r_fv}/B{r_px}-1", FMT_PCT, bold=True,
              fill="E2EFDA" if result.fair_value_per_share >= a.current_price else "FCE4EC")
    S.blank()

    # ── WACC ─────────────────────────────────────────────────────────────────
    S.section("WACC", 3 + years)
    S.label("Risk-free rate (%)")
    r_rf = S.input(a.risk_free_rate, num_fmt=FMT_PCT)
    S.label("Equity risk premium (%)")
    r_erp = S.input(a.equity_risk_premium, num_fmt=FMT_PCT)
    S.label("Beta (levered)")
    r_beta = S.input(a.beta, num_fmt=FMT_NUM)
    S.label("Cost of equity = Rf + B × ERP (%)")
    r_re = S.formula(f"=B{r_rf}+B{r_beta}*B{r_erp}", FMT_PCT)
    S.label("Pre-tax cost of debt (%)")
    r_rd = S.input(a.cost_of_debt, num_fmt=FMT_PCT)
    S.label("Tax rate (%)")
    r_tax = S.input(a.tax_rate, num_fmt=FMT_PCT)
    S.label("After-tax cost of debt (%)")
    r_rdat = S.formula(f"=B{r_rd}*(1-B{r_tax})", FMT_PCT)
    S.label("Market value of equity ($M)")
    r_e = S.input(a.equity_value_market)
    S.label("Market value of debt ($M)")
    r_d = S.input(a.debt_value)
    S.label("Weight of equity")
    r_we = S.formula(f"=B{r_e}/(B{r_e}+B{r_d})", FMT_PCT)
    S.label("Weight of debt")
    r_wd = S.formula(f"=B{r_d}/(B{r_e}+B{r_d})", FMT_PCT)
    S.label("WACC = We×Re + Wd×Rd×(1−T) (%)")
    r_wacc = S.formula(f"=B{r_we}*B{r_re}+B{r_wd}*B{r_rdat}", FMT_PCT, bold=True,
                       fill=TOTAL_FILL)
    S.blank()

    # ── Operating assumptions ────────────────────────────────────────────────
    S.section("OPERATING ASSUMPTIONS", 3 + years)
    S.label("Base revenue (prior year, $M)")
    r_base = S.input(a.base_revenue)
    S.label("EBIT margin (%)")
    r_margin = S.input(a.ebit_margin, num_fmt=FMT_PCT)
    S.label("D&A (% of revenue)")
    r_da_pct = S.input(a.da_pct_revenue, num_fmt=FMT_PCT)
    S.label("CapEx (% of revenue)")
    r_capex_pct = S.input(a.capex_pct_revenue, num_fmt=FMT_PCT)
    S.label("Change in NWC (% of revenue)")
    r_nwc_pct = S.input(a.nwc_change_pct_revenue, num_fmt=FMT_PCT)
    S.label("Terminal growth rate (%)")
    r_g = S.input(a.terminal_growth_rate, num_fmt=FMT_PCT)
    S.blank()

    # ── FCF projection ───────────────────────────────────────────────────────
    hdr = S._r + 1
    S.write(hdr, 1, "Line Item", font=_font(bold=True, color=IB_BLUE),
            fill=_fill(HEADER_FILL), align=_align("left"), border=Border(bottom=THICK_BOTTOM))
    for j in range(years):
        S.write(hdr, 2 + j, f"Year {j + 1}", font=_font(bold=True, color=IB_BLUE),
                fill=_fill(HEADER_FILL), align=_align("center"),
                border=Border(bottom=THICK_BOTTOM))
    S._r = hdr

    r_growth = hdr + 1
    r_rev = hdr + 2
    r_ebit = hdr + 3
    r_taxes = hdr + 4
    r_nopat = hdr + 5
    r_da = hdr + 6
    r_capex = hdr + 7
    r_nwc = hdr + 8
    r_fcf = hdr + 9
    r_df = hdr + 10
    r_pv = hdr + 11
    r_terminal = hdr + 12

    S.write(r_growth, 1, "Revenue growth (%)", font=_font(color=IB_BLUE), align=_align("left"))
    for j in range(years):
        S.write(r_growth, 2 + j, g[j], font=_font(color=IB_BLUE), fill=_fill(INPUT_FILL),
                align=_align("right"), num_fmt=FMT_PCT)

    def prev_rev(j):
        return f"B{r_base}" if j == 0 else f"{col(j-1)}{r_rev}"

    S.write(r_rev, 1, "Revenue ($M)", font=_font(), align=_align("left"))
    for j in range(years):
        S.write(r_rev, 2 + j, f"={prev_rev(j)}*(1+{col(j)}{r_growth})",
                font=_font(), align=_align("right"), num_fmt=FMT_ACCT_INT)

    S.write(r_ebit, 1, "EBIT ($M)", font=_font(), align=_align("left"))
    for j in range(years):
        S.write(r_ebit, 2 + j, f"={col(j)}{r_rev}*B{r_margin}",
                font=_font(), align=_align("right"), num_fmt=FMT_ACCT_INT)

    S.write(r_taxes, 1, "(-) Taxes ($M)", font=_font(), align=_align("left"))
    for j in range(years):
        S.write(r_taxes, 2 + j, f"=-{col(j)}{r_ebit}*B{r_tax}",
                font=_font(), align=_align("right"), num_fmt=FMT_ACCT_INT)

    S.write(r_nopat, 1, "NOPAT ($M)", font=_font(), align=_align("left"))
    for j in range(years):
        S.write(r_nopat, 2 + j, f"={col(j)}{r_ebit}+{col(j)}{r_taxes}",
                font=_font(), align=_align("right"), num_fmt=FMT_ACCT_INT)

    S.write(r_da, 1, "(+) D&A ($M)", font=_font(), align=_align("left"))
    for j in range(years):
        S.write(r_da, 2 + j, f"={col(j)}{r_rev}*B{r_da_pct}",
                font=_font(), align=_align("right"), num_fmt=FMT_ACCT_INT)

    S.write(r_capex, 1, "(-) CapEx ($M)", font=_font(), align=_align("left"))
    for j in range(years):
        S.write(r_capex, 2 + j, f"=-{col(j)}{r_rev}*B{r_capex_pct}",
                font=_font(), align=_align("right"), num_fmt=FMT_ACCT_INT)

    S.write(r_nwc, 1, "(-) Change in NWC ($M)", font=_font(), align=_align("left"))
    for j in range(years):
        S.write(r_nwc, 2 + j, f"=-{col(j)}{r_rev}*B{r_nwc_pct}",
                font=_font(), align=_align("right"), num_fmt=FMT_ACCT_INT)

    S.write(r_fcf, 1, "Free cash flow ($M)", font=_font(bold=True), align=_align("left"))
    for j in range(years):
        S.write(r_fcf, 2 + j,
                f"={col(j)}{r_nopat}+{col(j)}{r_da}+{col(j)}{r_capex}+{col(j)}{r_nwc}",
                font=_font(bold=True), align=_align("right"), num_fmt=FMT_ACCT_INT,
                border=Border(bottom=THICK_BOTTOM))

    S.write(r_df, 1, "Discount factor = 1/(1+WACC)^t", font=_font(), align=_align("left"))
    for j in range(years):
        S.write(r_df, 2 + j, f"=1/(1+$B${r_wacc})^{j + 1}",
                font=_font(), align=_align("right"), num_fmt='0.0000')

    S.write(r_pv, 1, "PV of FCF ($M)", font=_font(), align=_align("left"))
    for j in range(years):
        S.write(r_pv, 2 + j, f"={col(j)}{r_fcf}*{col(j)}{r_df}",
                font=_font(), align=_align("right"), num_fmt=FMT_ACCT_INT)

    # Terminal value: TV = FCF(n) × (1+g) / (WACC − g)
    S.write(r_terminal, 1, "Terminal value (Gordon growth, $M)", font=_font(), align=_align("left"))
    last_col = col(years - 1)
    S.write(r_terminal, 2, f"={last_col}{r_fcf}*(1+$B${r_g})/($B${r_wacc}-$B${r_g})",
            font=_font(), align=_align("right"), num_fmt=FMT_ACCT_INT,
            border=Border(bottom=THICK_BOTTOM))
    # The projection block wrote at explicit rows; keep the write cursor
    # in sync so the sections below don't collide with it.
    S._r = r_terminal
    S.blank()

    # ── Bridge to equity ─────────────────────────────────────────────────────
    S.section("VALUATION BRIDGE (LIVE)", 3 + years)
    S.label("PV of projected FCFs ($M)")
    r_sum_pv = S.formula(
        f"=SUM({col(0)}{r_pv}:{last_col}{r_pv})")
    S.label("PV of terminal value ($M)")
    r_pv_tv = S.formula(f"={col(0)}{r_terminal}*{last_col}{r_df}")
    S.total("Enterprise value ($M)", f"=B{r_sum_pv}+B{r_pv_tv}", FMT_ACCT_INT)
    r_ev_real = S._r
    S.blank()
    S.label("Total debt ($M)")
    r_debt = S.input(a.debt_value)
    S.label("(-) Cash ($M)")
    r_cash = S.input(a.cash)
    S.total("Net debt ($M)", f"=B{r_debt}-B{r_cash}", FMT_ACCT_INT)
    r_nd_real = S._r
    S.blank()
    S.label("Equity value ($M)")
    r_eq_real = S.formula(f"=B{r_ev_real}-B{r_nd_real}", FMT_ACCT_INT, bold=True)
    S.label("Shares outstanding (M)")
    r_shares = S.input(a.shares_outstanding, num_fmt=FMT_ACCT_INT)
    S.total("Fair value per share ($)", f"=B{r_eq_real}/B{r_shares}", FMT_ACCT)
    r_fv_real = S._r

    # Patch the forward-referenced KPI cells at the top to real formulas
    ws.cell(row=r_ev, column=2, value=f"=B{r_ev_real}")
    ws.cell(row=r_nd, column=2, value=f"=B{r_nd_real}")
    ws.cell(row=r_eq, column=2, value=f"=B{r_eq_real}")
    ws.cell(row=r_fv, column=2, value=f"=B{r_fv_real}")

    S.autofit(3 + years)

    # ── Sensitivity sheet ────────────────────────────────────────────────────
    if getattr(result, "sensitivity", None) is not None and not result.sensitivity.empty:
        ws_s = wb.create_sheet("Sensitivity")
        Sn = IBSheet(ws_s)
        Sn.title("Fair Value per Share — WACC × Terminal Growth", len(result.sensitivity.columns))
        Sn.blank()
        start = Sn._r + 1
        last = Sn.table(result.sensitivity, start_row=start, num_fmt=FMT_ACCT)
        _apply_color_scale(ws_s, start + 1, last, 2, len(result.sensitivity.columns))
        Sn.autofit(len(result.sensitivity.columns))

    # institutional polish: freeze headers, tab colours, landscape print
    for _ws in wb.worksheets:
        IBSheet(_ws).finish(freeze_panes="A5")

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.getvalue()


def build_sensitivity_only(df, sheet_title: str, subtitle: str = "",
                           ncols_data: int = 5) -> bytes:
    """Minimal workbook for a single color-scaled sensitivity table."""
    if not _HAS_OPENPYXL:  # pragma: no cover
        raise RuntimeError("openpyxl required for Excel export")
    wb = Workbook()
    ws = wb.active
    ws.title = "Sensitivity"
    S = IBSheet(ws)
    S.title(sheet_title, len(df.columns))
    if subtitle:
        S.subtitle(subtitle, len(df.columns))
    S.blank()
    start = S._r + 1
    last = S.table(df, start_row=start, num_fmt=FMT_NUM,
                   pct_cols=list(range(1, len(df.columns))))
    _apply_color_scale(ws, start + 1, last, 2, len(df.columns))
    S.autofit(len(df.columns))
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# IPO workbook
# ─────────────────────────────────────────────────────────────────────────────
def build_ipo_workbook(result) -> bytes:
    """Build an editable IPO underwriting workbook from IPOResult.

    Assumptions sheet holds every analyst input (yellow); the Pricing sheet
    holds live multi-method valuation formulas (EV = metric × multiple),
    the IPO-discounted price range, proceeds and dilution; Scenarios and
    Checks complete the package.
    """
    if not _HAS_OPENPYXL:  # pragma: no cover
        raise RuntimeError("openpyxl required for Excel export")
    a = result.assumptions
    wb = Workbook()
    ws = wb.active
    ws.title = "Assumptions"
    S = IBSheet(ws)
    S.title(f"IPO Underwriting Model  |  {a.company_name or a.ticker}", 4)
    S.subtitle(f"Prepared by Octavian Terminal  |  {datetime.now().strftime('%b %d, %Y %H:%M')}", 4)
    S.blank()

    # ── Company financials ───────────────────────────────────────────────────
    S.section("COMPANY FINANCIALS ($M)", 4)
    S.label("Revenue")
    r_rev = S.input(a.revenue)
    S.label("Revenue growth (%)")
    r_growth = S.input(a.revenue_growth_pct / 100.0, num_fmt=FMT_PCT)
    S.label("EBITDA")
    r_ebitda = S.input(a.ebitda)
    S.label("EBITDA margin (%)")
    r_em = S.input(a.ebitda_margin_pct / 100.0, num_fmt=FMT_PCT)
    S.label("EBIT")
    r_ebit = S.input(a.ebit)
    S.label("FCF")
    r_fcf = S.input(a.fcf)
    S.label("Net income")
    r_ni = S.input(a.net_income)
    S.label("Net debt")
    r_nd = S.input(a.net_debt)
    S.label("Shares outstanding pre-IPO (M)")
    r_shares = S.input(a.shares_outstanding_m, num_fmt=FMT_ACCT_INT)
    S.blank()

    # ── Comp multiples ───────────────────────────────────────────────────────
    S.section("VALUATION MULTIPLES (COMP CONTEXT)", 4)
    S.label("Peer EV/Revenue (x)")
    r_evr = S.input(a.peer_ev_revenue, num_fmt=FMT_MULT)
    S.label("Peer EV/EBITDA (x)")
    r_eve = S.input(a.peer_ev_ebitda, num_fmt=FMT_MULT)
    S.label("Peer EV/EBIT (x)")
    r_evi = S.input(a.peer_ev_ebit, num_fmt=FMT_MULT)
    S.label("IPO comp EV/Revenue (x)")
    r_ievr = S.input(a.ipo_comp_ev_revenue, num_fmt=FMT_MULT)
    S.label("IPO comp EV/EBITDA (x)")
    r_ieve = S.input(a.ipo_comp_ev_ebitda, num_fmt=FMT_MULT)
    S.label("Precedent EV/EBITDA (x)")
    r_prev = S.input(a.precedent_ev_ebitda, num_fmt=FMT_MULT)
    S.blank()

    # ── Offering structure ───────────────────────────────────────────────────
    S.section("OFFERING STRUCTURE", 4)
    S.label("Primary shares (M)")
    r_prim = S.input(a.primary_shares_m, num_fmt=FMT_ACCT_INT)
    S.label("Secondary shares (M)")
    r_sec = S.input(a.secondary_shares_m, num_fmt=FMT_ACCT_INT)
    S.label("Gross spread (%)")
    r_spread = S.input(a.gross_spread_pct / 100.0, num_fmt=FMT_PCT)
    S.label("Other expenses ($M)")
    r_exp = S.input(a.other_expenses_m)
    S.label("Options (M)")
    r_opt = S.input(a.options_m, num_fmt=FMT_ACCT_INT)
    S.label("Options strike ($)")
    r_opt_k = S.input(a.options_strike)
    S.label("Warrants (M)")
    r_warr = S.input(a.warrants_m, num_fmt=FMT_ACCT_INT)
    S.label("Warrants strike ($)")
    r_warr_k = S.input(a.warrants_strike)
    S.label("RSUs (M)")
    r_rsu = S.input(a.rsus_m, num_fmt=FMT_ACCT_INT)
    S.label("Convertible ($M)")
    r_conv = S.input(a.convertible_m, num_fmt=FMT_ACCT_INT)
    S.label("Convertible premium (%)")
    r_conv_p = S.input(a.convertible_premium_pct / 100.0, num_fmt=FMT_PCT)
    S.label("Preferred shares (M)")
    r_pref = S.input(a.preferred_m, num_fmt=FMT_ACCT_INT)
    S.blank()

    # ── Key outputs (live) ───────────────────────────────────────────────────
    S.section("KEY OUTPUTS (LIVE)", 4)
    S.label("Blended enterprise value ($M)")
    # Blend is a weighted average of the EV legs below (weights as in engine)
    r_blend = S.formula(
        f"=(B{r_rev}*B{r_evr}*1.0 + B{r_ebitda}*B{r_eve}*1.2 + "
        f"IF(B{r_ebit}>0,B{r_ebit}*B{r_evi}*0.8,0) + B{r_rev}*B{r_ievr}*1.1 + "
        f"B{r_ebitda}*B{r_ieve}*1.1 + B{r_ebitda}*B{r_prev}*0.7) / "
        f"(1.0 + 1.2 + IF(B{r_ebit}>0,0.8,0) + 1.1 + 1.1 + 0.7)",
        FMT_ACCT_INT, bold=True)
    S.label("Equity value ($M)")
    r_eq = S.formula(f"=B{r_blend}-B{r_nd}", FMT_ACCT_INT)
    S.label("IPO discount (%)")
    r_disc = S.formula(f"=IF(B{r_em}>=0.15,0.15,0.20)", FMT_PCT)
    # Anchor price uses the pre-IPO fully diluted share count (no TSM, no
    # conversion) so the range is computed without circularity; the share
    # bridge below applies the treasury-stock method at the mid price.
    S.label("Anchor shares for pricing (M, pre-IPO)")
    r_anchor = S.input(
        max(a.shares_outstanding_m + a.options_m + a.warrants_m + a.rsus_m + a.preferred_m, 1.0),
        num_fmt=FMT_ACCT_INT)
    S.label("Low-end offer price ($)")
    r_lo = S.formula(f"=B{r_eq}*(1-B{r_disc})/B{r_anchor}", FMT_ACCT, bold=True)
    S.label("High-end offer price ($)")
    r_hi = S.formula(f"=B{r_lo}*(1+0.05)", FMT_ACCT, bold=True)
    S.label("Mid price ($)")
    r_mid = S.formula(f"=(B{r_lo}+B{r_hi})/2", FMT_ACCT, bold=True)
    S.label("Gross proceeds ($M)")
    r_gp = S.formula(f"=(B{r_prim}+B{r_sec})*B{r_mid}", FMT_ACCT_INT, bold=True)
    S.label("Underwriting spread ($M)")
    S.formula(f"=B{r_gp}*B{r_spread}", FMT_ACCT_INT)
    S.label("Net proceeds ($M)")
    r_np = S.formula(f"=B{r_gp}-B{r_gp}*B{r_spread}-B{r_exp}", FMT_ACCT_INT, bold=True)
    S.label("Total shares post-IPO (M)")
    r_tot = S.formula(
        f"=B{r_shares}+B{r_prim}+IF(B{r_opt_k}<B{r_mid},B{r_opt}*(1-B{r_opt_k}/B{r_mid}),0)+"
        f"IF(B{r_warr_k}<B{r_mid},B{r_warr}*(1-B{r_warr_k}/B{r_mid}),0)+B{r_rsu}+B{r_pref}+"
        f"B{r_conv}/(B{r_mid}*(1+B{r_conv_p}))",
        FMT_ACCT_INT, bold=True)
    S.label("Market cap at mid ($M)")
    S.formula(f"=B{r_tot}*B{r_mid}", FMT_ACCT_INT, bold=True)
    S.label("Primary dilution (%)")
    S.formula(f"=B{r_prim}/B{r_tot}", FMT_PCT, bold=True)
    S.label("Sources = Uses check (gross vs net+spread+expenses, $M)")
    S.formula(f"=B{r_gp}-(B{r_np}+B{r_gp}*B{r_spread}+B{r_exp})", FMT_ACCT, bold=True)
    S.autofit(4)

    # ── Valuation methods sheet ──────────────────────────────────────────────
    ws_v = wb.create_sheet("Valuation Methods")
    V = IBSheet(ws_v)
    V.title("Multi-Method Valuation  (EV legs, $M)", 3)
    V.subtitle("Blend weights mirror the pricing engine (dynamic, method-relevant)", 3)
    V.blank()
    V.table(pd.DataFrame({
        "Method": list(result.valuation_methods.keys()) if result.valuation_methods else ["—"],
        "Implied EV ($M)": [v.get("ev", 0) for v in (result.valuation_methods.values() if result.valuation_methods else [])],
        "Weight": [v.get("weight", 1) for v in (result.valuation_methods.values() if result.valuation_methods else [])],
    }))
    V.autofit(3)

    # ── Pricing / proceeds summary ───────────────────────────────────────────
    ws_p = wb.create_sheet("Pricing & Proceeds")
    P = IBSheet(ws_p)
    P.title(f"IPO Pricing & Proceeds  |  {a.company_name or a.ticker}", 3)
    P.blank()
    rng = result.price_range
    P.label("Indicative low price ($)")
    P.write(P._r + 1, 2, rng[0], font=_font(bold=True), align=_align("right"), num_fmt=FMT_ACCT)
    P._r += 1
    P.label("Indicative high price ($)")
    P.write(P._r + 1, 2, rng[1], font=_font(bold=True), align=_align("right"), num_fmt=FMT_ACCT)
    P._r += 1
    P.label("Probability-weighted price ($)")
    P.write(P._r + 1, 2, result.implied_ipo_price, font=_font(bold=True), align=_align("right"), num_fmt=FMT_ACCT)
    P._r += 1
    P.blank()
    P.section("PROCEEDS ($M)", 3)
    P.label("Gross proceeds")
    P.write(P._r + 1, 2, result.gross_proceeds, font=_font(), align=_align("right"), num_fmt=FMT_ACCT_INT)
    P._r += 1
    P.label("Primary proceeds")
    P.write(P._r + 1, 2, result.primary_proceeds, font=_font(), align=_align("right"), num_fmt=FMT_ACCT_INT)
    P._r += 1
    P.label("Secondary proceeds")
    P.write(P._r + 1, 2, result.secondary_proceeds, font=_font(), align=_align("right"), num_fmt=FMT_ACCT_INT)
    P._r += 1
    P.label("Net proceeds")
    P.write(P._r + 1, 2, result.net_proceeds, font=_font(bold=True), align=_align("right"), num_fmt=FMT_ACCT_INT)
    P._r += 1
    P.blank()
    P.section("SHARE COUNT (M)", 3)
    bridge = result.share_count_bridge
    for k, label in [("existing_shares", "Existing shares"),
                     ("options_net_tsm", "Options (net, TSM)"),
                     ("warrants_net_tsm", "Warrants (net, TSM)"),
                     ("rsus", "RSUs"),
                     ("convertible_shares", "Convertible shares"),
                     ("preferred_shares", "Preferred shares"),
                     ("primary_new_shares", "Primary new shares"),
                     ("total_shares_post_ipo", "Total post-IPO")]:
        P.label(label)
        P.write(P._r + 1, 2, bridge.get(k, 0), font=_font(bold=(k == "total_shares_post_ipo")),
                align=_align("right"), num_fmt=FMT_ACCT_INT)
        P._r += 1
    P.autofit(3)

    # ── Scenarios ────────────────────────────────────────────────────────────
    ws_s = wb.create_sheet("Scenarios")
    Sn = IBSheet(ws_s)
    Sn.title("IPO Scenario Cases (probability-weighted)", 5)
    Sn.blank()
    rows = []
    for s in result.scenarios:
        rows.append({
            "Scenario": s.label, "Probability": s.probability, "Price ($)": s.price,
            "Valuation ($M)": s.valuation, "Market Reaction": s.market_reaction[:80],
        })
    if rows:
        Sn.table(pd.DataFrame(rows), pct_cols=[1])
    Sn.autofit(5)

    # institutional polish
    for _ws in wb.worksheets:
        IBSheet(_ws).finish(freeze_panes="A5")
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Precedent transactions workbook
# ─────────────────────────────────────────────────────────────────────────────
def build_precedents_workbook(result) -> bytes:
    """Editable precedents workbook: deal universe, multiples stats and the
    adjusted valuation range for the subject."""
    if not _HAS_OPENPYXL:  # pragma: no cover
        raise RuntimeError("openpyxl required for Excel export")
    wb = Workbook()
    ws = wb.active
    ws.title = "Deals"
    S = IBSheet(ws)
    S.title("Precedent Transactions Analysis", len(result.multiples_df.columns) if not result.multiples_df.empty else 4)
    S.subtitle(f"Prepared by Octavian Terminal  |  {datetime.now().strftime('%b %d, %Y %H:%M')}", 4)
    S.blank()
    if not result.multiples_df.empty:
        S.table(result.multiples_df)
    S.blank()
    S.autofit(len(result.multiples_df.columns) if not result.multiples_df.empty else 4)

    ws_m = wb.create_sheet("Implied Valuation")
    M = IBSheet(ws_m)
    M.title("Adjusted Valuation Range", 4)
    M.blank()
    M.section("MULTIPLE STATISTICS", 4)
    stats_rows = []
    for mult, s in result.stats.items():
        if s.get("count", 0) == 0:
            continue
        stats_rows.append({"Multiple": mult, "Mean": s["mean"], "Median": s["median"],
                           "Q1": s["q1"], "Q3": s["q3"], "Low": s["low"], "High": s["high"]})
    if stats_rows:
        M.table(pd.DataFrame(stats_rows), num_fmt=FMT_MULT)
    M.blank()
    M.section("IMPLIED VALUE OF SUBJECT ($/share)", 4)
    imp_rows = [{"Method": k, "Value ($)": v} for k, v in result.implied_valuation.items()]
    if imp_rows:
        M.table(pd.DataFrame(imp_rows), num_fmt=FMT_ACCT)
    M.blank()
    lo, hi = result.adjusted_range
    M.total("Adjusted range low ($)", lo, FMT_ACCT)
    M.total("Adjusted range high ($)", hi, FMT_ACCT)
    M.blank()
    M.section("CONTROL PREMIUM", 4)
    cp = result.control_premium_analysis
    M.label(f"Median control premium: {cp.get('median_premium_pct', 0) or 0:.1f}%")
    M.label(f"Deals with disclosed premium: {cp.get('n_deals_with_premium', 0)}")
    M.label(cp.get("context", ""))
    M.autofit(4)

    for _ws in wb.worksheets:
        IBSheet(_ws).finish(freeze_panes="A5")
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Valuation bridge workbook
# ─────────────────────────────────────────────────────────────────────────────
def build_bridge_workbook(result) -> bytes:
    """Cross-model valuation dashboard workbook with live weighted range."""
    if not _HAS_OPENPYXL:  # pragma: no cover
        raise RuntimeError("openpyxl required for Excel export")
    wb = Workbook()
    ws = wb.active
    ws.title = "Valuation Bridge"
    S = IBSheet(ws)
    S.title("Cross-Model Valuation Dashboard", 6)
    S.subtitle(f"Prepared by Octavian Terminal  |  {datetime.now().strftime('%b %d, %Y %H:%M')}", 6)
    S.blank()
    rows = []
    for m in result.methods:
        rows.append({
            "Method": m.label, "Low ($)": m.low, "High ($)": m.high,
            "Mid ($)": m.mid, "Weight": result.weights.get(m.key, 0),
            "Source": m.source,
        })
    if rows:
        S.table(pd.DataFrame(rows), num_fmt=FMT_ACCT, pct_cols=[4])
    S.blank()
    lo, hi = result.weighted_range
    S.total("Weighted low ($)", lo, FMT_ACCT)
    S.total("Weighted high ($)", hi, FMT_ACCT)
    S.total("Weighted mid ($)", result.weighted_mid, FMT_ACCT)
    S.blank()
    S.section("CONVERGENCE & DRIVERS", 6)
    S.label(result.convergence)
    for d in result.drivers:
        S.label(d)
    if result.outliers:
        S.blank()
        S.section("OUTLIERS", 6)
        for o in result.outliers:
            S.label(o)
    S.autofit(6)
    for _ws in wb.worksheets:
        IBSheet(_ws).finish(freeze_panes="A5")
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.getvalue()

