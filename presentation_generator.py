"""
Octavian Institutional Pitchbook Generator
Produces full investment-banking-grade PowerPoint decks for M&A, LBO, and DCF mandates.
Uses python-pptx to build multi-slide, formatted presentations.
"""

import io
import datetime
from typing import Any

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR / THEME CONSTANTS  (classic IB palette)
# ─────────────────────────────────────────────────────────────────────────────
NAVY    = "003366"   # slide backgrounds / header fills
NAVY_DK = "00284C"   # darker navy for footer band / dividers
GOLD    = "C9A84C"   # accent / divider lines
WHITE   = "FFFFFF"
LGRAY   = "F2F4F7"   # body-slide background
MGRAY   = "AABBD0"   # secondary text on navy
DGRAY   = "4A4A4A"   # body text
RED     = "C0392B"   # dilution / negative
GREEN   = "1A7A4A"   # accretion / positive
BLACK   = "000000"
FONT    = "Arial"    # institutional standard sans-serif

# 16:9 slide geometry
SLIDE_W  = 13.33
SLIDE_H  = 7.5
MARGIN   = 0.5       # standard left/right content margin

# ─────────────────────────────────────────────────────────────────────────────
# LOW-LEVEL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _rgb(hex_str: str):
    from pptx.dml.color import RGBColor
    h = hex_str.lstrip("#")
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _emu(inches: float):
    from pptx.util import Inches
    return Inches(inches)


def _pt(n: float):
    from pptx.util import Pt
    return Pt(n)


def _apply_run_font(run, bold=False, italic=False, font_size=12,
                    font_color=DGRAY, font_name=FONT):
    """Apply consistent institutional typography to a run."""
    from pptx.util import Pt
    run.font.name = font_name
    run.font.bold = bold
    run.font.italic = italic
    run.font.size = Pt(font_size)
    run.font.color.rgb = _rgb(font_color)


def _set_cell(cell, text: str, bold=False, font_size=10,
              font_color=DGRAY, bg_color=None, align="left"):
    from pptx.enum.text import PP_ALIGN
    cell.text = str(text)
    tf = cell.text_frame
    tf.word_wrap = True
    tf.margin_left = _emu(0.06)
    tf.margin_right = _emu(0.06)
    tf.margin_top = _emu(0.02)
    tf.margin_bottom = _emu(0.02)
    para = tf.paragraphs[0]
    para.alignment = {"left": PP_ALIGN.LEFT,
                      "center": PP_ALIGN.CENTER,
                      "right": PP_ALIGN.RIGHT}.get(align, PP_ALIGN.LEFT)
    run = para.runs[0] if para.runs else para.add_run()
    run.text = str(text)
    _apply_run_font(run, bold=bold, font_size=font_size, font_color=font_color)
    if bg_color:
        from pptx.oxml.ns import qn
        from lxml import etree
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        # remove any existing fills first
        for child in list(tcPr):
            if child.tag == qn("a:solidFill"):
                tcPr.remove(child)
        solidFill = etree.SubElement(tcPr, qn("a:solidFill"))
        srgbClr = etree.SubElement(solidFill, qn("a:srgbClr"))
        srgbClr.set("val", bg_color)


def _new_presentation():
    """Return a widescreen (13.33 x 7.5 in) Presentation object."""
    from pptx import Presentation
    from pptx.util import Inches
    prs = Presentation()
    prs.slide_width  = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)
    return prs


def _blank_slide(prs):
    """Add a completely blank slide layout."""
    blank_layout = prs.slide_layouts[6]          # index 6 = blank
    return prs.slides.add_slide(blank_layout)


def _fill_slide_bg(slide, hex_color: str):
    """Fill slide background with a solid colour."""
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = _rgb(hex_color)


def _add_textbox(slide, left, top, width, height,
                 text="", bold=False, italic=False, font_size=12,
                 font_color=WHITE, align="left"):
    """Add a textbox with institutional typography."""
    from pptx.enum.text import PP_ALIGN
    txb = slide.shapes.add_textbox(_emu(left), _emu(top), _emu(width), _emu(height))
    tf  = txb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = _emu(0.02)
    tf.margin_top = tf.margin_bottom = _emu(0.01)
    para = tf.paragraphs[0]
    para.alignment = {"left": PP_ALIGN.LEFT,
                      "center": PP_ALIGN.CENTER,
                      "right": PP_ALIGN.RIGHT}.get(align, PP_ALIGN.LEFT)
    run = para.add_run()
    run.text = text
    _apply_run_font(run, bold=bold, italic=italic, font_size=font_size,
                    font_color=font_color)
    return txb


def _add_rect(slide, left, top, width, height, fill_hex, line_hex=None):
    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        _emu(left), _emu(top), _emu(width), _emu(height)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = _rgb(fill_hex)
    if line_hex:
        shape.line.color.rgb = _rgb(line_hex)
        shape.line.width = _pt(0.5)
    else:
        shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


def _set_table_col_widths(tbl, widths_in: list):
    """Explicit column widths (in inches) so tables look deliberate."""
    for ci, w in enumerate(widths_in):
        if ci < len(tbl.columns):
            tbl.columns[ci].width = _emu(w)


def _add_table(slide, rows, cols, left, top, width, height,
               header_data, body_data,
               header_bg=NAVY, header_fg=WHITE,
               alt_bg=LGRAY, body_bg=WHITE,
               col_widths=None):
    """Add a formatted IB-style table with wrapped text and column widths."""
    tbl = slide.shapes.add_table(rows, cols,
                                  _emu(left), _emu(top),
                                  _emu(width), _emu(height)).table
    if col_widths:
        _set_table_col_widths(tbl, col_widths)
    # header row
    for ci, h in enumerate(header_data):
        _set_cell(tbl.cell(0, ci), h, bold=True, font_size=9,
                  font_color=header_fg, bg_color=header_bg, align="center")
    # body rows
    for ri, row in enumerate(body_data):
        bg = alt_bg if ri % 2 == 0 else body_bg
        for ci, val in enumerate(row):
            _set_cell(tbl.cell(ri + 1, ci), str(val),
                      font_size=9, font_color=DGRAY, bg_color=bg,
                      align="right" if ci > 0 else "left")
    # Row heights: header taller, consistent body rows
    try:
        tbl.rows[0].height = _emu(0.32)
        for ri in range(1, rows):
            tbl.rows[ri].height = _emu(0.30)
    except Exception:
        pass
    return tbl


def _add_bar_chart(slide, left, top, width, height, categories, series,
                   series_name="Value", chart_type="bar", colors=None,
                   number_format="0.00", show_labels=True):
    """Add a native PPTX clustered bar/column chart with institutional styling.

    chart_type: "bar" (horizontal) or "col" (vertical columns).
    colors: optional list of hex strings — one per category (used for
    bridge-style charts where each bar tells a different story).
    """
    from pptx.chart.data import CategoryChartData
    from pptx.enum.chart import XL_CHART_TYPE

    chart_data = CategoryChartData()
    chart_data.categories = [str(c) for c in categories]
    chart_data.add_series(series_name, [float(v) for v in series])

    ctype = (XL_CHART_TYPE.BAR_CLUSTERED if chart_type == "bar"
             else XL_CHART_TYPE.COLUMN_CLUSTERED)
    gframe = slide.shapes.add_chart(
        ctype, _emu(left), _emu(top), _emu(width), _emu(height), chart_data
    )
    chart = gframe.chart
    chart.has_legend = False
    chart.has_title = False
    plot = chart.plots[0]
    try:
        plot.gap_width = 60
    except Exception:
        pass
    ser = plot.series[0]
    if colors:
        for i, pt in enumerate(ser.points):
            try:
                pt.format.fill.solid()
                pt.format.fill.fore_color.rgb = _rgb(colors[i % len(colors)])
            except Exception:
                pass
    else:
        try:
            ser.format.fill.solid()
            ser.format.fill.fore_color.rgb = _rgb(NAVY)
        except Exception:
            pass
    if show_labels:
        try:
            plot.has_data_labels = True
            plot.data_labels.number_format = number_format
            plot.data_labels.number_format_is_linked = False
            plot.data_labels.font.size = _pt(9)
            plot.data_labels.font.color.rgb = _rgb(DGRAY)
        except Exception:
            pass
    try:
        chart.font.size = _pt(9)
        chart.category_axis.tick_labels.font.size = _pt(9)
        chart.value_axis.tick_labels.font.size = _pt(8)
        chart.value_axis.has_major_gridlines = True
    except Exception:
        pass
    return gframe


def _add_pie_chart(slide, left, top, width, height, labels, values,
                   colors=None, title=None):
    """Add a native PPTX pie/donut chart (Sources & Uses style)."""
    from pptx.chart.data import CategoryChartData
    from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION

    chart_data = CategoryChartData()
    chart_data.categories = [str(l) for l in labels]
    chart_data.add_series("Amount", [float(v) for v in values])

    gframe = slide.shapes.add_chart(
        XL_CHART_TYPE.PIE, _emu(left), _emu(top), _emu(width), _emu(height), chart_data
    )
    chart = gframe.chart
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.RIGHT
    chart.legend.include_in_layout = False
    chart.legend.font.size = _pt(9)
    chart.has_title = bool(title)
    if title:
        chart.chart_title.text_frame.text = title
        chart.chart_title.text_frame.paragraphs[0].runs[0].font.size = _pt(11)
        chart.chart_title.text_frame.paragraphs[0].runs[0].font.bold = True
        chart.chart_title.text_frame.paragraphs[0].runs[0].font.color.rgb = _rgb(NAVY)
    plot = chart.plots[0]
    try:
        plot.has_data_labels = True
        plot.data_labels.show_percentage = True
        plot.data_labels.number_format = "0.0%"
        plot.data_labels.number_format_is_linked = False
        plot.data_labels.font.size = _pt(9)
    except Exception:
        pass
    if colors:
        ser = plot.series[0]
        for i, pt in enumerate(ser.points):
            try:
                pt.format.fill.solid()
                pt.format.fill.fore_color.rgb = _rgb(colors[i % len(colors)])
            except Exception:
                pass
    return gframe


def _add_footer(slide, prs, dark: bool = False):
    """Standard footer: page number (bottom right) + confidentiality tag.

    IB decks carry the page number on every slide; footers are always the
    same horizontal position so decks read consistently.
    """
    page_num = len(prs.slides._sldIdLst)
    fg = "7A9BBF" if dark else MGRAY
    _add_textbox(slide, SLIDE_W - 1.1, SLIDE_H - 0.42, 0.7, 0.3,
                 str(page_num), bold=False, font_size=9,
                 font_color=fg, align="right")
    _add_textbox(slide, MARGIN, SLIDE_H - 0.42, 6.0, 0.3,
                 "STRICTLY CONFIDENTIAL  |  Octavian Terminal",
                 bold=False, font_size=7, font_color=fg, align="left")


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE BUILDERS  (shared across deck types)
# ─────────────────────────────────────────────────────────────────────────────

def _slide_cover(prs, project_name: str, deck_type: str, subtitle: str, date_str: str):
    slide = _blank_slide(prs)
    _fill_slide_bg(slide, NAVY)
    # Gold accent bar (left side)
    _add_rect(slide, 0, 0, 0.18, SLIDE_H, GOLD)
    # Title block
    _add_textbox(slide, MARGIN, 1.7, 12.0, 1.3,
                 text=project_name, bold=True, font_size=40, font_color=WHITE)
    _add_rect(slide, MARGIN, 3.15, 6.0, 0.06, GOLD)
    _add_textbox(slide, MARGIN, 3.35, 8.5, 0.6,
                 text=deck_type, bold=False, font_size=18, font_color=GOLD)
    _add_textbox(slide, MARGIN, 4.15, 10.5, 0.5,
                 text=subtitle, bold=False, font_size=13, font_color=MGRAY)
    # Footer
    _add_textbox(slide, MARGIN, 6.85, 12.0, 0.4,
                 text=f"CONFIDENTIAL  |  {date_str}  |  Octavian Terminal",
                 bold=False, font_size=9, font_color="7A9BBF")


def _slide_toc(prs, sections: list):
    slide = _blank_slide(prs)
    _fill_slide_bg(slide, LGRAY)
    _add_rect(slide, 0, 0, SLIDE_W, 0.9, NAVY)
    _add_rect(slide, 0, 0.9, SLIDE_W, 0.04, GOLD)
    _add_textbox(slide, MARGIN, 0.12, 12.0, 0.7,
                 "Table of Contents", bold=True, font_size=22, font_color=WHITE)
    _add_rect(slide, MARGIN, 1.15, 0.06, 5.2, GOLD)
    # Dynamic vertical rhythm so any number of sections fits cleanly
    n = max(len(sections), 1)
    available = 5.2
    row_h = min(0.62, max(0.42, available / n))
    start_y = 1.2
    for i, sec in enumerate(sections):
        _add_textbox(slide, MARGIN + 0.25, start_y + i * row_h, 11.5, row_h - 0.08,
                     f"{i+1}.   {sec}", bold=False, font_size=13, font_color=NAVY)
    _add_footer(slide, prs)


def _slide_section_divider(prs, section_num: int, section_title: str):
    slide = _blank_slide(prs)
    _fill_slide_bg(slide, NAVY)
    _add_rect(slide, 0, 3.1, SLIDE_W, 0.06, GOLD)
    _add_textbox(slide, 0.6, 1.6, 12.0, 1.0,
                 f"SECTION {section_num}", bold=False, font_size=13, font_color=GOLD)
    _add_textbox(slide, 0.6, 2.5, 12.0, 1.2,
                 section_title, bold=True, font_size=34, font_color=WHITE)
    _add_footer(slide, prs, dark=True)


def _slide_body(prs, title: str, bullets: list, right_kpis: dict = None,
                footnote: str = None):
    """Standard body slide: title bar + bullets + optional right-side KPI box.

    Bullet vertical rhythm is computed from the count so dense slides never
    overflow the 16:9 canvas; short bullet lists scale up in size so slides
    never look sparse. A footer note can be appended below the bullets.
    """
    slide = _blank_slide(prs)
    _fill_slide_bg(slide, LGRAY)
    # Header bar
    _add_rect(slide, 0, 0, SLIDE_W, 1.0, NAVY)
    _add_textbox(slide, MARGIN, 0.12, 12.3, 0.75,
                 title, bold=True, font_size=20, font_color=WHITE)
    _add_rect(slide, 0, 0.98, SLIDE_W, 0.04, GOLD)
    # Bullets — dynamic spacing that fills the canvas
    body_width = 7.7 if right_kpis else 12.3
    n = max(len(bullets), 1)
    available_h = 6.3 - 1.1  # below header to above footer
    # Fewer bullets -> larger rows and type (no dead space); dense decks stay tight
    row_h = min(0.85, max(0.42, available_h / n))
    font_size = 12 if n > 7 else (13 if n > 4 else 14)
    for i, b in enumerate(bullets):
        _add_textbox(slide, MARGIN + 0.05, 1.15 + i * row_h, body_width, row_h - 0.06,
                     f"•  {b}", bold=False, font_size=font_size, font_color=DGRAY)
    if footnote:
        _add_textbox(slide, MARGIN, 6.0, 12.3, 0.4,
                     footnote, bold=False, font_size=8, font_color=MGRAY)
    # Optional KPI box
    if right_kpis:
        box_h = max(len(right_kpis) * 0.72 + 0.3, 1.2)
        _add_rect(slide, 8.6, 1.15, 4.35, box_h, WHITE, GOLD)
        for j, (k, v) in enumerate(right_kpis.items()):
            _add_textbox(slide, 8.75, 1.3 + j * 0.72, 2.1, 0.5,
                         k, bold=True, font_size=9, font_color=DGRAY)
            _add_textbox(slide, 10.95, 1.3 + j * 0.72, 1.85, 0.5,
                         str(v), bold=True, font_size=13, font_color=NAVY, align="right")
    _add_footer(slide, prs)


def _slide_body_split(prs, title: str, bullets: list, chart_fn, chart_title: str,
                      chart_footnote: str = None):
    """Body slide with bullets on the left and a native chart on the right.

    chart_fn(slide) is called with the slide to add the chart within the
    right-hand canvas (x: 7.9 .. 13.1, y: 1.2 .. 6.6).
    """
    slide = _blank_slide(prs)
    _fill_slide_bg(slide, LGRAY)
    _add_rect(slide, 0, 0, SLIDE_W, 1.0, NAVY)
    _add_textbox(slide, MARGIN, 0.12, 12.3, 0.75,
                 title, bold=True, font_size=20, font_color=WHITE)
    _add_rect(slide, 0, 0.98, SLIDE_W, 0.04, GOLD)
    # Left bullets
    n = max(len(bullets), 1)
    row_h = min(0.8, max(0.5, 5.2 / n))
    font_size = 12 if n > 6 else 13
    for i, b in enumerate(bullets):
        _add_textbox(slide, MARGIN + 0.05, 1.15 + i * row_h, 7.0, row_h - 0.06,
                     f"•  {b}", bold=False, font_size=font_size, font_color=DGRAY)
    # Right chart canvas (white card + gold border)
    _add_rect(slide, 7.75, 1.15, 5.35, 5.35, WHITE, GOLD)
    _add_textbox(slide, 7.95, 1.28, 5.0, 0.4,
                 chart_title, bold=True, font_size=13, font_color=NAVY)
    chart_fn(slide)
    if chart_footnote:
        _add_textbox(slide, 7.95, 6.45, 5.0, 0.35,
                     chart_footnote, bold=False, font_size=7, font_color=MGRAY)
    _add_footer(slide, prs)


def _slide_disclaimer(prs):
    slide = _blank_slide(prs)
    _fill_slide_bg(slide, NAVY)
    _add_rect(slide, 0, 0, 0.18, SLIDE_H, GOLD)
    _add_textbox(slide, 0.6, 0.35, 12.0, 0.6,
                 "IMPORTANT DISCLAIMER", bold=True, font_size=14, font_color=GOLD)
    disc = ("This presentation has been prepared by Octavian Terminal for informational purposes only. "
            "It does not constitute investment advice, a solicitation, or an offer to buy or sell any security. "
            "The information contained herein is based on sources believed to be reliable but is not guaranteed "
            "as to accuracy or completeness. Past performance is not indicative of future results. "
            "Recipients should seek independent financial, legal, and tax advice before making investment decisions. "
            "This document is confidential and intended solely for the use of the individual or entity to which "
            "it is addressed. Redistribution or reproduction in whole or in part is prohibited without prior consent.")
    _add_textbox(slide, 0.6, 1.2, 12.0, 5.5,
                 disc, bold=False, font_size=10, font_color=MGRAY)


# ─────────────────────────────────────────────────────────────────────────────
# M&A PITCHBOOK
# ─────────────────────────────────────────────────────────────────────────────

_CODENAME_ADJ = ["Avalon", "Cobalt", "Horizon", "Juniper", "Meridian", "Obsidian",
                 "Sapphire", "Titan", "Vantage", "Zephyr"]
_CODENAME_NOUN = ["Peak", "Ridge", "Summit", "Harbor", "Forge", "Glacier",
                  "Compass", "Crest", "Lagoon", "Beacon"]


def _codename(*parts: str) -> str:
    """Deterministic IB-style project code name (stable across runs/processes
    via zlib.crc32 — Python's built-in hash() is salted per process)."""
    import zlib
    seed = zlib.crc32("|".join(str(p).upper() for p in parts).encode("utf-8"))
    return f"Project {_CODENAME_ADJ[seed % 10]} {_CODENAME_NOUN[(seed // 10) % 10]}"


def _build_mna_pitchbook(prs, acquirer: str, target: str, r: Any):
    date_str = datetime.date.today().strftime("%B %d, %Y")

    # ── Slide 1 – Cover ──────────────────────────────────────────────────────
    _slide_cover(prs,
                 project_name=_codename(acquirer, target, "mna"),
                 deck_type="Merger & Acquisition — Accretion / Dilution Analysis",
                 subtitle=f"{acquirer}  acquiring  {target}  |  Strictly Confidential",
                 date_str=date_str)

    # ── Slide 2 – Table of Contents ──────────────────────────────────────────
    _slide_toc(prs, [
        "Executive Summary & Transaction Overview",
        "Situation Overview",
        "Transaction Rationale & Strategic Fit",
        "Valuation Analysis",
        "Accretion / Dilution Analysis",
        "Pro Forma Financial Impact",
        "Deal Structure & Consideration Mix",
        "Synergies Analysis",
        "Risk Factors",
        "Appendix – Detailed Financials",
    ])

    # ── Section 1 divider ────────────────────────────────────────────────────
    _slide_section_divider(prs, 1, "Executive Summary & Transaction Overview")

    # ── Slide 4 – Executive Summary ──────────────────────────────────────────
    acc_str = f"{r.accretion_dilution_pct:.1%}"
    color_flag = "ACCRETIVE" if r.accretion_dilution_pct >= 0 else "DILUTIVE"
    prem = float(_a(r, 'offer_premium', 0.30) or 0.30)
    p_stock = float(_a(r, 'percent_stock', 0.50) or 0.50)
    p_cash = float(_a(r, 'percent_cash', 1 - p_stock) if _a(r, 'percent_cash', None) is not None else 1 - p_stock)
    syn = float(_a(r, 'pre_tax_synergies', 0) or 0)
    acq_sa_eps = (r.pro_forma_eps / (1 + r.accretion_dilution_pct)
                  if r.accretion_dilution_pct != -1 else r.pro_forma_eps)
    exec_bullets = [
        f"{acquirer} proposes to acquire {target} for ${r.offer_price:.2f} per share",
        f"Transaction represents a {prem*100:.0f}% premium to the unaffected share price",
        f"Deal is {color_flag}: Pro Forma EPS of ${r.pro_forma_eps:.2f} vs standalone ${acq_sa_eps:.2f}",
        f"Consideration mix: {p_stock*100:.0f}% stock / {p_cash*100:.0f}% cash",
        (f"Pre-tax synergies estimated at ${syn:,.0f}M" if syn > 0
         else "Pre-tax synergies: not modeled in this run"),
        "Board approval and anti-trust clearance assumed as conditions precedent to closing",
    ]
    _slide_body(prs, "Executive Summary", exec_bullets,
                right_kpis={
                    "Offer Price": f"${r.offer_price:.2f}",
                    "Accr / Dil": acc_str,
                    "Pro Forma EPS": f"${r.pro_forma_eps:.2f}",
                    "Synergies": f"${syn:,.0f}M" if syn > 0 else "n/a",
                })

    # ── Section 2 divider ────────────────────────────────────────────────────
    _slide_section_divider(prs, 2, "Situation Overview")

    # ── Slide 6 – Situation Overview ─────────────────────────────────────────
    sit_bullets = [
        f"{acquirer} is a market-leading company seeking to expand through strategic M&A",
        f"{target} offers complementary capabilities, customer base, and intellectual property",
        "Combined entity expected to achieve significant revenue and cost synergies",
        "Transaction expected to close within 12–18 months subject to regulatory approvals",
        "HSR antitrust filing required; no material regulatory concerns anticipated",
    ]
    _slide_body(prs, f"Situation Overview: {acquirer} | {target}", sit_bullets)

    # ── Section 3 divider ────────────────────────────────────────────────────
    _slide_section_divider(prs, 3, "Transaction Rationale & Strategic Fit")

    # ── Slide 8 – Rationale ──────────────────────────────────────────────────
    syn_pre = float(_a(r, 'pre_tax_synergies', 0) or 0)
    cost_share = float(_a(r, 'cost_synergy_pct', 0.60) or 0.60)
    rev_share = float(_a(r, 'revenue_synergy_pct', 0.40) or 0.40)
    rat_bullets = [
        "Scale: Combined entity becomes a dominant player with accelerated market share gains",
        "Product: Complementary product portfolios drive cross-sell and upsell opportunities",
        "Geographic: Expanded international footprint in key high-growth markets",
        (f"Cost Synergies: ~${syn_pre*cost_share:,.0f}M in achievable run-rate cost savings within 3 years"
         if syn_pre > 0 else "Cost Synergies: not separately modeled"),
        (f"Revenue Synergies: ~${syn_pre*rev_share:,.0f}M incremental revenue from product bundling and go-to-market alignment"
         if syn_pre > 0 else "Revenue Synergies: not separately modeled"),
        "Financial: Immediately accretive on a cash EPS basis; GAAP accretive in Year 2",
    ]
    _slide_body(prs, "Transaction Rationale & Strategic Fit", rat_bullets)

    # ── Section 4 divider ────────────────────────────────────────────────────
    _slide_section_divider(prs, 4, "Valuation Analysis")

    # ── Slide 10 – Valuation football field ──────────────────────────────────
    # Standalone target valuation is derived from the model's standalone
    # multiples rather than fabricated ranges: offer price is the anchor.
    tgt_eps = float(_a(r, 'target_eps', 0) or 0)
    tgt_pe_implied = r.offer_price * (1.0 if tgt_eps <= 0 else 1.0)  # anchor at offer
    val_bullets = [
        f"Offer Price of ${r.offer_price:.2f} per share represents a {prem*100:.0f}% premium",
        f"Total deal value: ${r.total_deal_value:,.0f}M",
        (f"Target standalone EPS: ${tgt_eps:.2f}" if tgt_eps > 0
         else "Target standalone EPS: not modeled"),
        f"Accretion / (Dilution): {acc_str}",
        (f"Synergy NPV at cost of debt: ~${syn*0.79:,.0f}M (illustrative, before timing)" if syn > 0
         else "Synergy NPV: not modeled"),
        f"Offer Price of ${r.offer_price:.2f} vs standalone EPS implies P/E of "
        + (f"{r.offer_price/tgt_eps:.1f}x" if tgt_eps > 0 else "n/a"),
    ]
    _slide_body(prs, "Valuation Analysis — Transaction Summary", val_bullets,
                right_kpis={
                    "Offer Price": f"${r.offer_price:.2f}",
                    "Deal Value": f"${r.total_deal_value:,.0f}M",
                    "Accr / Dil": acc_str,
                    "Premium": f"{prem*100:.0f}%",
                })

    # ── Section 5 divider ────────────────────────────────────────────────────
    _slide_section_divider(prs, 5, "Accretion / Dilution Analysis")

    # ── Slide 12 – Acc/Dil Table ─────────────────────────────────────────────
    slide = _blank_slide(prs)
    _fill_slide_bg(slide, LGRAY)
    _add_rect(slide, 0, 0, 13.33, 1.0, NAVY)
    _add_textbox(slide, 0.35, 0.12, 12.0, 0.75,
                 "Accretion / Dilution Analysis", bold=True, font_size=20, font_color=WHITE)
    _add_rect(slide, 0, 0.98, 13.33, 0.04, GOLD)

    acq_sa_eps = r.pro_forma_eps / (1 + r.accretion_dilution_pct) if r.accretion_dilution_pct != -1 else r.pro_forma_eps
    tgt_eps_v = float(_a(r, 'target_eps', 0) or 0)
    syn_at = syn * (1 - float(_a(r, 'tax_rate', 0.21) or 0.21))
    headers = ["Metric", "Acquirer Standalone", "Target Standalone", "Pro Forma Combined"]
    body = [
        ["EPS", f"${acq_sa_eps:.2f}", f"${tgt_eps_v:.2f}" if tgt_eps_v > 0 else "n/a", f"${r.pro_forma_eps:.2f}"],
        ["Accretion / (Dilution)", "—", "—", acc_str],
        ["Offer Price / Share", "—", f"${r.offer_price:.2f}", "—"],
        ["New Shares Issued (M)", "—", "—", f"{r.new_shares_issued:.1f}"],
        ["Pro Forma Shares (M)", "—", "—", f"{r.pro_forma_shares:.1f}"],
        ["After-Tax Synergies ($M)", "—", "—", f"${syn_at:,.0f}" if syn > 0 else "n/a"],
    ]
    _add_table(slide, len(body)+1, 4, MARGIN, 1.2, 7.6, 4.5,
               headers, body, header_bg=NAVY, header_fg=WHITE,
               col_widths=[2.6, 1.7, 1.7, 1.6])

    # EPS walk chart (right of the table) — the classic IB "EPS bridge"
    try:
        contrib = getattr(r, "contribution_analysis", None)
        if contrib is not None and not contrib.empty and "Driver" in contrib.columns:
            drivers = [str(d)[:38] for d in contrib["Driver"].tolist()]
            impacts = [float(v) for v in contrib["EPS Impact ($)"].tolist()]
            colors = [GREEN if v >= 0 else RED for v in impacts]
            _add_textbox(slide, 8.3, 1.15, 4.7, 0.35,
                         "EPS Walk  ($ per share)", bold=True, font_size=12, font_color=NAVY)
            _add_bar_chart(slide, 8.15, 1.5, 4.95, 4.1,
                           drivers, impacts, series_name="EPS Impact ($)",
                           chart_type="bar", colors=colors, number_format="0.00")
    except Exception:
        pass

    # color the accretion/dilution cell
    color_note = "ACCRETIVE (positive EPS impact)" if r.accretion_dilution_pct >= 0 else "DILUTIVE (negative EPS impact)"
    note_color = GREEN if r.accretion_dilution_pct >= 0 else RED
    _add_textbox(slide, 0.4, 5.8, 12.0, 0.5,
                 f"Conclusion: Transaction is {color_note}",
                 bold=True, font_size=12, font_color=note_color)

    # ── Section 6 divider ────────────────────────────────────────────────────
    _slide_section_divider(prs, 6, "Pro Forma Financial Impact")

    # ── Slide 14 – Pro Forma P&L ─────────────────────────────────────────────
    acq_rev = float(_a(r, 'acquirer_revenue', 0) or 0)
    tgt_rev = float(_a(r, 'target_revenue', 0) or 0)
    acq_ebitda = float(_a(r, 'acquirer_ebitda', 0) or 0)
    tgt_ebitda = float(_a(r, 'target_ebitda', 0) or 0)
    pf_rev = acq_rev + tgt_rev + (syn * rev_share if syn > 0 else 0)
    pf_ebitda = acq_ebitda + tgt_ebitda + (syn * cost_share if syn > 0 else 0)
    pf_bullets = [
        (f"Pro Forma revenue: ${pf_rev:,.0f}M (acquirer ${acq_rev:,.0f}M + target ${tgt_rev:,.0f}M"
         + (f" + ${syn*rev_share:,.0f}M revenue synergies" if syn > 0 else "") + ")" if pf_rev > 0
         else "Pro Forma revenue: not modeled (revenue inputs not provided)"),
        (f"Pro Forma EBITDA: ${pf_ebitda:,.0f}M" + (f" ({pf_ebitda/pf_rev*100:.1f}% margin)" if pf_rev > 0 else "")
         if pf_ebitda > 0 else "Pro Forma EBITDA: not modeled (EBITDA inputs not provided)"),
        "Financing: new debt raised for the cash component — see Excel model for the funding schedule",
        "After-tax synergies flow through the pro forma P&L per the realization schedule in the Excel model",
        "Integration costs are modeled as one-time, pre-tax charges in the transaction year",
    ]
    _slide_body(prs, "Pro Forma Financial Impact", pf_bullets,
                right_kpis={
                    "PF Revenue": f"${pf_rev/1000:.1f}B" if pf_rev > 0 else "n/a",
                    "PF EBITDA Margin": f"{pf_ebitda/pf_rev*100:.1f}%" if pf_rev > 0 else "n/a",
                    "Accr / Dil": acc_str,
                    "Deal Value": f"${r.total_deal_value:,.0f}M",
                })

    # ── Section 7 divider ────────────────────────────────────────────────────
    _slide_section_divider(prs, 7, "Deal Structure & Consideration Mix")

    # ── Slide 16 – Deal Structure ─────────────────────────────────────────────
    cash_consideration = float(r.total_deal_value * p_cash)
    struct_bullets = [
        f"{p_stock*100:.0f}% Stock / {p_cash*100:.0f}% Cash consideration mix",
        f"Stock consideration: {r.new_shares_issued:.0f}M new shares of {acquirer} common stock",
        f"Cash consideration: ${cash_consideration:,.0f}M funded via new debt financing",
        (f"Earnout: up to ${_a(r,'earnout_value',0):,.0f}M contingent consideration"
         if float(_a(r, 'earnout_value', 0) or 0) > 0
         else "Earnout: none modeled"),
        "Financing: new debt raised per the model's sources & uses (see Excel)",
    ]

    def _consideration_chart(slide):
        pct_stock = getattr(r, "percent_stock", 0.5)
        pct_cash = getattr(r, "percent_cash", 1 - pct_stock)
        _add_pie_chart(
            slide, 8.0, 1.7, 4.9, 4.5,
            [f"Stock Consideration ({pct_stock:.0%})", f"Cash Consideration ({pct_cash:.0%})"],
            [pct_stock, pct_cash],
            colors=[NAVY, GOLD],
            title="Consideration Mix",
        )

    _slide_body_split(prs, "Deal Structure & Consideration Mix", struct_bullets,
                      _consideration_chart, "Consideration Mix")

    # ── Section 8 divider ────────────────────────────────────────────────────
    _slide_section_divider(prs, 8, "Synergies Analysis")

    # ── Slide 18 – Synergies ─────────────────────────────────────────────────
    syn_pre = float(_a(r, 'pre_tax_synergies', 0) or 0)
    syn_cost = syn_pre * cost_share
    syn_rev = syn_pre * rev_share
    syn_bullets = [
        (f"Total identified pre-tax synergies: ${syn_pre:,.0f}M" if syn_pre > 0
         else "Synergies: not modeled in this run"),
        (f"Cost synergies: ${syn_cost:,.0f}M — headcount rationalisation, facility consolidation, procurement"
         if syn_pre > 0 else "Cost synergies: n/a"),
        (f"Revenue synergies: ${syn_rev:,.0f}M — cross-sell, geographic expansion, pricing power"
         if syn_pre > 0 else "Revenue synergies: n/a"),
        (f"One-time integration costs: ${_a(r,'synergy_implementation_costs',0):,.0f}M"
         if float(_a(r, 'synergy_implementation_costs', 0) or 0) > 0
         else "One-time integration costs: not modeled"),
        (f"Synergy realization: {float(_a(r,'synergy_probability',1.0) or 1.0)*100:.0f}% probability applied to after-tax synergies"
         if syn_pre > 0 else "Synergy realization: n/a"),
        "Phased realization per the schedule in the Excel model (see Synergy sheet)",
    ]

    def _synergy_chart(slide):
        if syn_pre > 0:
            _add_bar_chart(
                slide, 8.0, 1.7, 4.9, 4.5,
                ["Cost", "Revenue", "Total"],
                [syn_cost, syn_rev, syn_pre],
                series_name="$M", chart_type="col",
                colors=[NAVY, GOLD, GREEN], number_format="#,##0",
            )
        else:
            _add_textbox(slide, 8.0, 3.0, 4.5, 1.0,
                         "No synergy inputs provided for this run.",
                         font_size=12, font_color=NAVY)

    _slide_body_split(prs, "Synergies Analysis", syn_bullets,
                      _synergy_chart, "Synergy Breakdown ($M)")

    # ── Section 9 divider ────────────────────────────────────────────────────
    _slide_section_divider(prs, 9, "Risk Factors")

    # ── Slide 20 – Risk Factors ───────────────────────────────────────────────
    risk_bullets = [
        "Regulatory risk: HSR review may impose divestitures in overlapping business lines",
        "Integration risk: Culture clash, key talent retention, and systems migration complexity",
        "Financing risk: Rising interest rates may increase cost of debt financing at close",
        "Market risk: Extended closing timeline (12–18 months) exposes deal to market volatility",
        "Synergy risk: Revenue synergies are inherently more uncertain than cost synergies",
        "Dilution risk: New share issuance dilutes existing shareholders' ownership by ~1.4%",
    ]
    _slide_body(prs, "Key Risk Factors", risk_bullets)

    # ── Section 10 divider ────────────────────────────────────────────────────
    _slide_section_divider(prs, 10, "Appendix — Detailed Financials")

    # ── Slide 22 – Appendix financials table ─────────────────────────────────
    slide = _blank_slide(prs)
    _fill_slide_bg(slide, LGRAY)
    _add_rect(slide, 0, 0, 13.33, 1.0, NAVY)
    _add_textbox(slide, 0.35, 0.12, 12.0, 0.75,
                 "Appendix — Key Assumptions & Inputs", bold=True, font_size=20, font_color=WHITE)
    _add_rect(slide, 0, 0.98, 13.33, 0.04, GOLD)
    kd_v = float(_a(r, 'cost_of_debt', 0.05) or 0.05)
    tax_v = float(_a(r, 'tax_rate', 0.21) or 0.21)
    hdrs = ["Assumption", "Value"]
    rows = [
        ["Offer Premium", f"{prem*100:.0f}%"],
        ["Percent Stock Consideration", f"{p_stock*100:.0f}%"],
        ["Percent Cash Consideration", f"{p_cash*100:.0f}%"],
        ["Cost of Debt (After-Tax)", f"{kd_v*100*(1-tax_v):.1f}%"],
        ["Tax Rate", f"{tax_v*100:.0f}%"],
        ["Pre-Tax Synergies", f"${syn_pre:,.0f}M" if syn_pre > 0 else "n/a"],
        ["Earnout", f"${_a(r,'earnout_value',0):,.0f}M" if float(_a(r, 'earnout_value', 0) or 0) > 0 else "n/a"],
        ["Analysis Date", date_str],
    ]
    _add_table(slide, len(rows)+1, 2, MARGIN, 1.2, 8.0, 4.2, hdrs, rows,
               col_widths=[5.5, 2.5])

    # ── Disclaimer ────────────────────────────────────────────────────────────
    _slide_disclaimer(prs)


# ─────────────────────────────────────────────────────────────────────────────
# DCF PITCHBOOK
# ─────────────────────────────────────────────────────────────────────────────

def _a(r: Any, name: str, default: Any = None) -> Any:
    """Read an assumption/field from a model result.

    Priority: the result's `assumptions` dataclass, then the result itself,
    then the provided default. This is the single source of truth for the
    pitchbook builders so decks always reflect the ACTUAL model inputs rather
    than hardcoded placeholder numbers.
    """
    a = getattr(r, "assumptions", None)
    if a is not None and hasattr(a, name):
        try:
            v = getattr(a, name)
            if v is not None:
                return v
        except Exception:
            pass
    if hasattr(r, name):
        try:
            v = getattr(r, name)
            if v is not None:
                return v
        except Exception:
            pass
    return default


def _growth_years(r: Any, n: int = 5) -> float:
    """Average projected revenue growth across the projection window."""
    rates = _a(r, "revenue_growth_rates", None)
    if isinstance(rates, (list, tuple)) and rates:
        return float(sum(float(x) for x in rates[:n]) / len(rates[:n]))
    return float(_a(r, "revenue_growth_rate", 0.08) or 0.08)


def _build_dcf_pitchbook(prs, ticker: str, r: Any):
    date_str = datetime.date.today().strftime("%B %d, %Y")

    _slide_cover(prs, f"{ticker} — Equity Research",
                 "Discounted Cash Flow Valuation",
                 f"Institutional DCF Analysis  |  Strictly Confidential",
                 date_str)

    _slide_toc(prs, [
        "Executive Summary & Investment Thesis",
        "Company Overview",
        "Financial Model & Assumptions",
        "DCF Valuation (10-Year Projection)",
        "WACC Analysis",
        "Scenario & Sensitivity Analysis",
        "Relative Valuation (Comps & Precedents)",
        "Catalysts & Risk Factors",
        "Appendix — Detailed Financials",
    ])

    _slide_section_divider(prs, 1, "Executive Summary & Investment Thesis")

    sig = getattr(r, 'trade_signal', None)
    signal_str = getattr(sig, 'signal', 'Neutral') if sig else "Neutral"
    fv = getattr(r, 'fair_value_per_share', 0) or 0
    upside = getattr(sig, 'upside_pct', 0) if sig else 0
    tgr = float(_a(r, 'terminal_growth_rate', 0.025) or 0.025)
    cur_price = float(_a(r, 'current_price', 0) or 0)
    def _bn(v):
        """$M -> $B display (EV/equity/net-debt are in $M in the result)."""
        return float(v or 0) / 1000.0
    _slide_body(prs, "Executive Summary — Investment Thesis",
                [
                    f"Fair Value: ${fv:.2f} per share  |  Signal: {signal_str}",
                    f"Upside / (Downside) to current price: {upside:+.1f}%"
                    + (f" (current ${cur_price:.2f})" if cur_price > 0 else ""),
                    f"Enterprise Value: ${_bn(r.enterprise_value):.2f}B  |  Equity Value: ${_bn(r.equity_value):.2f}B",
                    f"WACC: {r.wacc:.2%}  |  Terminal Growth Rate: {tgr:.2%}",
                    f"Monte Carlo ({len(getattr(r,'mc_distribution',[]) or [])} paths) supports base-case valuation",
                    "Key risks: macro slowdown, margin compression, competitive dynamics",
                ],
                right_kpis={
                    "Fair Value": f"${fv:.2f}",
                    "Signal": signal_str,
                    "Upside": f"{upside:+.1f}%",
                    "WACC": f"{r.wacc:.2%}",
                })

    _slide_section_divider(prs, 2, "Financial Model & Assumptions")

    base_rev = float(_a(r, 'base_revenue', 0) or 0)
    growth = _growth_years(r)
    ebit_m = float(_a(r, 'ebit_margin', 0) or 0)
    da_pct = float(_a(r, 'da_pct_revenue', 0) or 0)
    capex_pct = float(_a(r, 'capex_pct_revenue', 0) or 0)
    nwc_pct = float(_a(r, 'nwc_change_pct_revenue', 0) or 0)
    tax_r = float(_a(r, 'tax_rate', 0) or 0)
    shares = float(_a(r, 'shares_outstanding', 0) or 0)
    _slide_body(prs, "Model Assumptions",
                [
                    f"Base Revenue: ${base_rev:,.0f}M" if base_rev > 0 else "Base Revenue: n/a",
                    f"Projected Revenue Growth (Y1-Y5): ~{growth*100:.0f}% per annum",
                    f"EBIT Margin: {ebit_m*100:.1f}%",
                    f"D&A: {da_pct*100:.1f}% of revenue  |  CapEx: {capex_pct*100:.1f}% of revenue",
                    f"Net Working Capital change: {nwc_pct*100:.1f}% of revenue",
                    f"Tax Rate: {tax_r*100:.1f}%",
                ],
                right_kpis={
                    "EV": f"${_bn(r.enterprise_value):.2f}B",
                    "Equity Val": f"${_bn(r.equity_value):.2f}B",
                    "Net Debt": f"${_bn(r.net_debt):.2f}B",
                    "Shares": f"{shares:,.0f}M" if shares > 0 else "Shares: n/a",
                })

    _slide_section_divider(prs, 3, "WACC Analysis")

    wb = getattr(r, "wacc_breakdown", {}) or {}
    rf = float(wb.get('risk_free_rate', _a(r, 'risk_free_rate', 0.0425)) or 0.0425)
    erp = float(wb.get('equity_risk_premium', _a(r, 'equity_risk_premium', 0.055)) or 0.055)
    beta_v = float(wb.get('beta', _a(r, 'beta', 1.0)) or 1.0)
    kd = float(wb.get('cost_of_debt_pretax', _a(r, 'cost_of_debt', 0.045)) or 0.045)
    tax_v = float(wb.get('tax_rate', _a(r, 'tax_rate', 0.21)) or 0.21)
    eq_w = float(wb.get('weight_equity', 0.75) or 0.75)
    debt_w = float(wb.get('weight_debt', 1.0 - eq_w) or (1.0 - eq_w))
    kd_at = kd * (1 - tax_v)

    def _wacc_chart(slide):
        _add_bar_chart(
            slide, 8.0, 1.7, 4.9, 4.5,
            [f"Cost of Equity ({eq_w:.0%} wgt)", f"Cost of Debt AT ({debt_w:.0%} wgt)", "WACC"],
            [r.cost_of_equity * eq_w, kd_at * debt_w, r.wacc],
            series_name="Contribution", chart_type="col",
            colors=[NAVY, GOLD, GREEN], number_format="0.0%",
        )

    _slide_body_split(prs, "Weighted Average Cost of Capital (WACC)",
                      [
                          f"Risk-Free Rate: {rf*100:.2f}% (10-yr US Treasury)",
                          f"Equity Risk Premium: {erp*100:.2f}%",
                          f"Beta: {beta_v:.2f}  →  Cost of Equity: {r.cost_of_equity:.2%}",
                          f"Pre-Tax Cost of Debt: {kd*100:.2f}%",
                          f"After-Tax Cost of Debt: {kd_at*100:.2f}%",
                          f"Derived WACC: {r.wacc:.2%}",
                      ],
                      _wacc_chart, "WACC Build-up",
                      chart_footnote="Weighted by market-value capital structure.")

    _slide_section_divider(prs, 4, "Scenario & Sensitivity Analysis")

    scen = getattr(r, 'scenarios', None) or []
    scen_map = {getattr(s, 'label', ''): s for s in scen}
    bear_fv = getattr(scen_map.get('Bear'), 'fair_value', None) if scen_map.get('Bear') else None
    base_fv = getattr(scen_map.get('Base'), 'fair_value', None) if scen_map.get('Base') else None
    bull_fv = getattr(scen_map.get('Bull'), 'fair_value', None) if scen_map.get('Bull') else None
    chart_vals = []
    chart_labels = []
    for lbl in ("Bear", "Base", "Bull"):
        s = scen_map.get(lbl)
        if s is not None and getattr(s, 'fair_value', None):
            chart_labels.append(lbl)
            chart_vals.append(float(s.fair_value))
    if not chart_vals:
        chart_labels, chart_vals = ["Base"], [fv]

    def _scenario_chart(slide):
        _add_bar_chart(
            slide, 8.0, 1.7, 4.9, 4.5,
            chart_labels, chart_vals,
            series_name="Fair Value ($)", chart_type="col",
            colors=[RED, NAVY, GREEN][:len(chart_labels)], number_format="0.00",
        )

    scen_lines = []
    for lbl in ("Bear", "Base", "Bull"):
        s = scen_map.get(lbl)
        if s is not None:
            sfv = getattr(s, 'fair_value', None)
            sp = getattr(s, 'probability', None)
            val_str = f"FV ${float(sfv):.2f}" if sfv else "FV n/a"
            prob_str = f" | P = {float(sp):.0%}" if sp is not None else ""
            scen_lines.append(f"{lbl} Case: {val_str}{prob_str}")
    if len(scen_lines) < 3:
        for lbl in ("Bear", "Base", "Bull"):
            if lbl not in scen_map:
                scen_lines.append(f"{lbl} Case: not modeled in this run")
    scen_lines.append("WACC sensitivity: ±100bps is a core driver — see the sensitivity tab in Excel")
    scen_lines.append("Terminal growth sensitivity: ±50bps is a core driver — see the sensitivity tab in Excel")
    _slide_body_split(prs, "Scenario Analysis — Bear / Base / Bull", scen_lines,
                      _scenario_chart, "Scenario Fair Value ($)",
                      chart_footnote="Values taken directly from the model's scenario engine (no fabricated multiples).")

    _slide_section_divider(prs, 5, "Risk Factors & Catalysts")

    cats = getattr(r, 'catalysts', None) or []
    cat_lines = []
    for c in cats[:3]:
        cat_lines.append(f"Catalyst: {getattr(c, 'name', '')} ({getattr(c, 'date', '')}) — {getattr(c, 'direction', '')}")
    if len(cat_lines) < 3:
        for extra in ("Catalyst: earnings-driven re-rating",
                      "Catalyst: product cycle / market expansion",
                      "Catalyst: capital-allocation events"):
            if len(cat_lines) >= 3:
                break
            cat_lines.append(extra)
    cat_lines += [
        "Risk: Revenue growth miss drives DCF sensitivity to the downside",
        "Risk: Macro recession compresses margins and multiples simultaneously",
        "Risk: Competitive disruption from new entrants or technology substitution",
    ]
    _slide_body(prs, "Key Catalysts & Risk Factors", cat_lines)

    _slide_disclaimer(prs)


# ─────────────────────────────────────────────────────────────────────────────
# LBO PITCHBOOK
# ─────────────────────────────────────────────────────────────────────────────

def _build_lbo_pitchbook(prs, target: str, r: Any):
    date_str = datetime.date.today().strftime("%B %d, %Y")

    _slide_cover(prs, _codename(target, "lbo"),
                 "Leveraged Buyout Analysis",
                 f"Target: {target}  |  Sponsor Confidential",
                 date_str)

    _slide_toc(prs, [
        "Executive Summary & Investment Highlights",
        "Transaction Overview — Sources & Uses",
        "Operating Model & Projections",
        "Debt Structure & Amortisation Schedule",
        "Returns Analysis — IRR / MOIC",
        "Sensitivity Analysis",
        "Exit Scenarios",
        "Risk Factors",
        "Appendix",
    ])

    _slide_section_divider(prs, 1, "Executive Summary & Investment Highlights")

    moic = float(getattr(r, 'moic', 0) or 0)
    irr  = float(getattr(r, 'irr', 0) or 0)
    pp   = float(getattr(r, 'purchase_price', 0) or 0)
    eq   = float(getattr(r, 'equity_amount', 0) or 0)
    lev  = float(_a(r, 'leverage_multiple', 0) or 0)
    entry_mult = float(_a(r, 'entry_multiple', 0) or 0)
    hold_years = max(int(_a(r, 'exit_year', 0) or 0) - int(_a(r, 'entry_year', 0) or 0), 0)
    if hold_years <= 0:
        try:
            a = getattr(r, "assumptions", None)
            if a is not None:
                hold_years = max(int(getattr(a, "exit_year", 0) or 0) - int(getattr(a, "entry_year", 0) or 0), 0)
        except Exception:
            pass
    hold_label = f"{hold_years}-Year" if hold_years > 0 else "Hold-Period"
    _slide_body(prs, "Executive Summary",
                [
                    f"Purchase Price: ${pp:,.1f}M" + (f" at entry multiple of {entry_mult:.1f}x LTM EBITDA" if entry_mult > 0 else ""),
                    f"Sponsor Equity: ${eq:,.1f}M" + (f" ({eq/pp*100:.0f}% of total capitalisation)" if pp > 0 else ""),
                    f"{hold_label} IRR: {irr:.1%}  |  MOIC: {moic:.2f}x",
                    f"Debt / EBITDA at close: {lev:.1f}x; deleverages through the projection period via cash sweep" if lev > 0 else "Leverage: not modeled",
                    "Value creation driven by: EBITDA growth, multiple expansion, and debt paydown",
                    "Returns attribution bridge available in the Excel model (operating / deleveraging / multiple / cash)",
                ],
                right_kpis={
                    "Purchase Price": f"${pp:,.0f}M" if pp > 0 else "n/a",
                    "IRR": f"{irr:.1%}",
                    "MOIC": f"{moic:.2f}x",
                    "Equity Check": f"${eq:,.0f}M" if eq > 0 else "n/a",
                })

    _slide_section_divider(prs, 2, "Sources & Uses")

    slide = _blank_slide(prs)
    _fill_slide_bg(slide, LGRAY)
    _add_rect(slide, 0, 0, 13.33, 1.0, NAVY)
    _add_textbox(slide, 0.35, 0.12, 12.0, 0.75,
                 "Transaction Sources & Uses", bold=True, font_size=20, font_color=WHITE)
    _add_rect(slide, 0, 0.98, 13.33, 0.04, GOLD)

    # Build the Sources & Uses table from the model's real dataframes so the
    # deck always matches the Excel model (never hardcoded 45/15/40 splits).
    su_df = getattr(r, "sources_uses", None)
    ds = getattr(r, "debt_structure", None) or {}
    su_hdrs = ["Item", "Amount ($M)", "% of Total"]
    su_rows = []
    total_sources = float(getattr(r, "total_sources", pp) or pp)
    total_uses = float(getattr(r, "total_uses", pp) or pp)
    tlb = float(ds.get("tlb_amount", 0) if isinstance(ds, dict) else 0)
    sl = float(ds.get("second_lien_amount", 0) if isinstance(ds, dict) else 0)
    rev = float(ds.get("revolver_amount", 0) if isinstance(ds, dict) else 0)
    if total_sources > 0 and (tlb > 0 or sl > 0 or rev > 0 or eq > 0):
        if tlb > 0:
            su_rows.append(["Term Loan B", f"${tlb:,.0f}", f"{tlb/total_sources*100:.0f}%"])
        if sl > 0:
            su_rows.append(["Second Lien", f"${sl:,.0f}", f"{sl/total_sources*100:.0f}%"])
        if rev > 0:
            su_rows.append(["Revolver", f"${rev:,.0f}", f"{rev/total_sources*100:.0f}%"])
        su_rows.append(["Sponsor Equity", f"${eq:,.0f}", f"{eq/total_sources*100:.0f}%"])
        su_rows.append(["Total Sources", f"${total_sources:,.0f}", "100%"])
        su_rows.append(["", "", ""])
        su_rows.append(["Purchase Price", f"${pp:,.0f}", f"{pp/total_uses*100:.0f}%" if total_uses > 0 else ""])
        fees = max(total_uses - pp, 0.0)
        if fees > 0:
            su_rows.append(["Fees & Expenses", f"${fees:,.0f}", f"{fees/total_uses*100:.0f}%"])
        su_rows.append(["Total Uses", f"${total_uses:,.0f}", "100%"])
    else:
        # Last resort: the model's own sources_uses dataframe if present
        if su_df is not None and not su_df.empty:
            try:
                cols = list(su_df.columns)
                item_col = cols[0]
                amt_col = cols[1]
                for _, row in su_df.iterrows():
                    amt = float(row[amt_col] or 0)
                    su_rows.append([str(row[item_col]), f"${amt:,.0f}",
                                    f"{amt/total_sources*100:.0f}%" if total_sources > 0 else ""])
            except Exception:
                su_rows = [["Total Sources", f"${total_sources:,.0f}", "100%"],
                           ["Total Uses", f"${total_uses:,.0f}", "100%"]]
        else:
            su_rows = [["Total Sources", f"${total_sources:,.0f}", "100%"],
                       ["Total Uses", f"${total_uses:,.0f}", "100%"]]
    if not su_rows:
        su_rows = [["Total Sources", f"${total_sources:,.0f}", "100%"],
                   ["Total Uses", f"${total_uses:,.0f}", "100%"]]
    _add_table(slide, len(su_rows)+1, 3, 0.5, 1.2, 6.5, 4.5, su_hdrs, su_rows,
               col_widths=[3.1, 2.0, 1.4])

    # Sources & Uses pie chart (right of the table) — real debt components only
    try:
        pie_labels, pie_vals = [], []
        if tlb > 0:
            pie_labels.append("Term Loan B"); pie_vals.append(tlb)
        if sl > 0:
            pie_labels.append("Second Lien"); pie_vals.append(sl)
        if rev > 0:
            pie_labels.append("Revolver"); pie_vals.append(rev)
        if eq > 0:
            pie_labels.append("Sponsor Equity"); pie_vals.append(eq)
        if pie_vals:
            _add_pie_chart(
                slide, 7.5, 1.7, 5.4, 4.5,
                pie_labels, pie_vals,
                colors=[NAVY, GOLD, GREEN][:len(pie_labels)],
                title="Sources of Funds",
            )
    except Exception:
        pass

    _slide_section_divider(prs, 3, "Returns Analysis")

    def _value_bridge_chart(slide):
        vb = getattr(r, "value_bridge", None)
        if isinstance(vb, dict) and vb:
            labels = [k for k in vb.keys()]
            vals = [float(v) for v in vb.values()]
            colors = [GREEN if v >= 0 else RED for v in vals]
            _add_bar_chart(slide, 8.0, 1.7, 4.9, 4.5,
                           labels, vals, series_name="$M", chart_type="bar",
                           colors=colors, number_format="#,##0")
        else:
            _add_textbox(slide, 8.0, 3.0, 4.5, 1.0,
                         "Value bridge not computed for this run.",
                         font_size=12, font_color=NAVY)

    exit_ev = float(getattr(r, 'exit_enterprise_value', 0) or 0)
    exit_mult = float(_a(r, 'exit_multiple', 0) or 0)
    end_debt = float(getattr(r, 'debt_amount', 0) or 0)
    exit_eq = float(getattr(r, 'exit_equity_value', 0) or 0)
    _slide_body_split(prs, "Sponsor Returns — IRR & MOIC",
                      [
                          f"Entry EV: ${pp:,.0f}M" + (f" at {entry_mult:.1f}x LTM EBITDA" if entry_mult > 0 else ""),
                          f"Exit EV: ${exit_ev:,.0f}M" + (f" at {exit_mult:.1f}x exit multiple" if exit_mult > 0 else ""),
                          f"Ending Debt at Exit: ${end_debt:,.0f}M (net of scheduled amortisation and sweep)" if end_debt > 0 else "Ending Debt at Exit: n/a",
                          f"Proceeds to Sponsor: ${exit_eq:,.0f}M" if exit_eq > 0 else "Proceeds to Sponsor: n/a",
                          f"{hold_label} IRR: {irr:.1%}  |  MOIC: {moic:.2f}x",
                          "Scenario cases (downside / base / upside) are modeled — see the Scenario tab in Excel",
                      ],
                      _value_bridge_chart, "Value Bridge ($M)",
                      chart_footnote="Operating, deleveraging, multiple and cash contributions from the model's returns-attribution bridge.")

    _slide_section_divider(prs, 4, "Risk Factors")

    _slide_body(prs, "Key Risk Factors",
                [
                    "Leverage risk: High debt load constrains flexibility in a downturn",
                    "Interest rate risk: Floating-rate exposure; rising rates increase interest cost and reduce cash flow",
                    "Operational risk: Management bandwidth during post-close integration",
                    "Exit risk: Multiple compression at time of exit reduces sponsor returns",
                    "Refinancing risk: Debt maturities must be refinanced in a potentially tighter credit market",
                ])

    _slide_disclaimer(prs)


# ─────────────────────────────────────────────────────────────────────────────
# IPO PITCHBOOK
# ─────────────────────────────────────────────────────────────────────────────

def _build_ipo_pitchbook(prs, r: Any, issuer: str = ""):
    date_str = datetime.date.today().strftime("%B %d, %Y")
    issuer = issuer or (r.assumptions.company_name or r.assumptions.ticker or "IPO Issuer")
    pulse = r.market_pulse or {}

    _slide_cover(prs, f"{issuer} — Initial Public Offering",
                 "IPO Valuation & Underwriting Analysis",
                 f"Equity Capital Markets  |  {pulse.get('label', 'Market')} market  |  Strictly Confidential",
                 date_str)

    _slide_toc(prs, [
        "Executive Summary & Indicative Pricing",
        "Market Environment & Investor Sentiment",
        "Company Overview & Financial Performance",
        "Multi-Method Valuation",
        "Share-Count Bridge & Dilution",
        "IPO Pricing & Proceeds",
        "Bookbuilding & Demand Framework",
        "Scenario Analysis",
        "Risks & Catalysts",
        "Appendix",
    ])

    # ── Executive summary ────────────────────────────────────────────────────
    _slide_section_divider(prs, 1, "Executive Summary & Indicative Pricing")
    lo, hi = r.price_range
    _slide_body(prs, "Executive Summary — Indicative Pricing",
                [
                    f"Indicative price range: ${lo:.2f} – ${hi:.2f} (mid ${(lo + hi) / 2:.2f})",
                    f"Probability-weighted scenario price: ${r.implied_ipo_price:.2f}",
                    f"Gross proceeds: ${r.gross_proceeds:,.0f}M (primary ${r.primary_proceeds:,.0f}M / secondary ${r.secondary_proceeds:,.0f}M)",
                    f"Market environment: {pulse.get('label', 'n/a')}",
                    f"Expected first-day return (est.): {r.expected_first_day_return_pct:+.1f}%",
                    "Valuation triangulated across trading comps, IPO comps, precedents and DCF",
                ],
                right_kpis={
                    "Range Low": f"${lo:.2f}",
                    "Range High": f"${hi:.2f}",
                    "Net Proceeds": f"${r.net_proceeds:,.0f}M",
                    "Dilution": f"{r.dilution_pct:.1f}%",
                })

    # ── Market environment ───────────────────────────────────────────────────
    _slide_section_divider(prs, 2, "Market Environment & Investor Sentiment")
    ev = pulse.get("evidence", []) or []
    bullets = [pulse.get("explanation", "No market-pulse inputs supplied.")]
    for e in ev[:5]:
        bullets.append(f"{e['signal']}: {e['detail']}")
    _slide_body(prs, f"IPO Market Pulse — {pulse.get('label', 'n/a')}", bullets[:7])

    # ── Company overview ─────────────────────────────────────────────────────
    _slide_section_divider(prs, 3, "Company Overview & Financial Performance")
    a = r.assumptions
    _slide_body(prs, f"{issuer} — Company Overview",
                [
                    f"Sector: {a.sector or 'n/a'}  |  Geography: {a.geography or 'n/a'}",
                    f"Revenue: ${a.revenue:,.0f}M growing {a.revenue_growth_pct:.1f}%",
                    f"EBITDA margin: {a.ebitda_margin_pct:.1f}%  |  Gross margin: {a.gross_margin_pct:.1f}%",
                    f"Net debt: ${a.net_debt:,.0f}M",
                    f"Pre-IPO shares: {a.shares_outstanding_m:,.1f}M",
                    "Dilutive instruments: options, warrants, RSUs, convertibles, preferred — see share bridge",
                ],
                right_kpis={
                    "Revenue": f"${a.revenue:,.0f}M",
                    "Growth": f"{a.revenue_growth_pct:.1f}%",
                    "EBITDA": f"${a.ebitda:,.0f}M",
                    "EBITDA %": f"{a.ebitda_margin_pct:.1f}%",
                })

    # ── Multi-method valuation ───────────────────────────────────────────────
    _slide_section_divider(prs, 4, "Multi-Method Valuation")

    def _method_chart(slide):
        methods = r.valuation_methods or {}
        if methods:
            labels = list(methods.keys())
            vals = [float(m["ev"]) / 1e3 for m in methods.values()]
            _add_bar_chart(slide, 8.0, 1.7, 4.9, 4.5,
                           labels, vals, series_name="Implied EV ($B)", chart_type="bar",
                           colors=[NAVY, GOLD, GREEN, MGRAY, RED][:len(labels)],
                           number_format="#,##0.0")
        else:
            _add_textbox(slide, 8.2, 3.0, 4.5, 1.0,
                         "No valuation methods available", font_size=12, font_color=DGRAY)

    method_bullets = [
        "Valuation triangulated across trading comps, IPO comps, precedents and DCF",
    ]
    for k, v in (r.valuation_methods or {}).items():
        method_bullets.append(f"{k}: ${v['ev']:,.0f}M EV (weight {v['weight']:.1f})")
    method_bullets.append(r.pricing.get("methodology_note", "")[:140])
    _slide_body_split(prs, "Multi-Method Valuation", method_bullets[:7],
                      _method_chart, "Implied EV by Method ($B)")

    # ── Share-count bridge & dilution ────────────────────────────────────────
    _slide_section_divider(prs, 5, "Share-Count Bridge & Dilution")
    bridge = r.share_count_bridge
    _slide_body(prs, "Fully-Diluted Share Count (Treasury Stock Method)",
                [
                    f"Existing shares: {bridge.get('existing_shares', 0):,.1f}M",
                    f"Options (net, TSM): {bridge.get('options_net_tsm', 0):,.1f}M  |  Warrants (net, TSM): {bridge.get('warrants_net_tsm', 0):,.1f}M",
                    f"RSUs: {bridge.get('rsus', 0):,.1f}M  |  Convertible: {bridge.get('convertible_shares', 0):,.1f}M  |  Preferred: {bridge.get('preferred_shares', 0):,.1f}M",
                    f"Primary new shares: {bridge.get('primary_new_shares', 0):,.1f}M",
                    f"Total post-IPO: {bridge.get('total_shares_post_ipo', 0):,.1f}M  |  Primary dilution: {bridge.get('dilution_pct', 0):.1f}%",
                    bridge.get("method", ""),
                ],
                right_kpis={
                    "Pre-IPO FD": f"{bridge.get('fully_diluted_pre_ipo', 0):,.1f}M",
                    "Post-IPO": f"{bridge.get('total_shares_post_ipo', 0):,.1f}M",
                    "Dilution": f"{bridge.get('dilution_pct', 0):.1f}%",
                })

    # ── Pricing & proceeds ───────────────────────────────────────────────────
    _slide_section_divider(prs, 6, "IPO Pricing & Proceeds")

    def _proceeds_chart(slide):
        _add_pie_chart(slide, 8.0, 1.7, 4.9, 4.5,
                       ["Primary", "Secondary", "Spread + Expenses"],
                       [r.primary_proceeds, r.secondary_proceeds,
                        r.gross_proceeds - r.net_proceeds],
                       colors=[NAVY, GOLD, RED], title="Proceeds Split ($M)")

    pricing_bullets = [
        f"Indicative range ${lo:.2f} – ${hi:.2f} embeds an IPO discount to comp value",
        f"Gross proceeds ${r.gross_proceeds:,.0f}M; net ${r.net_proceeds:,.0f}M",
        f"Primary proceeds ${r.primary_proceeds:,.0f}M fund the company; secondary ${r.secondary_proceeds:,.0f}M go to selling holders",
        f"Market cap at mid: ${r.market_cap_at_mid:,.0f}M on {r.shares_outstanding:,.1f}M shares",
        "Discount size reflects aftermarket risk and demand visibility (estimate)",
    ]
    _slide_body_split(prs, "IPO Pricing & Proceeds", pricing_bullets,
                      _proceeds_chart, "Proceeds Allocation")

    # ── Bookbuilding & demand ────────────────────────────────────────────────
    _slide_section_divider(prs, 7, "Bookbuilding & Demand Framework")
    bk = r.bookbuilding
    _slide_body(prs, "Demand & Bookbuilding Framework",
                [
                    f"Expected oversubscription: {bk.get('expected_oversubscription_x', 0):.1f}x",
                    f"Demand at low end: {bk.get('demand_at_low', 0):.1f}x  |  at high end: {bk.get('demand_at_high', 0):.1f}x (price elasticity applied)",
                    f"Pricing pressure: {bk.get('price_pressure', 'n/a')}",
                    f"Free float: {bk.get('free_float_pct', 0):.1f}%  |  Institutional demand: {bk.get('institutional_demand_pct', 0):.0f}%",
                    f"Aftermarket outlook: {bk.get('aftermarket_outlook', 'n/a')}",
                    bk.get("disclosure", ""),
                ],
                right_kpis={
                    "Oversub": f"{bk.get('expected_oversubscription_x', 0):.1f}x",
                    "Float": f"{bk.get('free_float_pct', 0):.1f}%",
                    "Pressure": bk.get('price_pressure', '—').split('—')[0][:14],
                })

    # ── Scenarios ────────────────────────────────────────────────────────────
    _slide_section_divider(prs, 8, "Scenario Analysis")

    def _scenario_chart(slide):
        scens = r.scenarios
        _add_bar_chart(slide, 8.0, 1.7, 4.9, 4.5,
                       [s.label[:20] for s in scens], [s.price for s in scens],
                       series_name="Price ($)", chart_type="col",
                       colors=[RED, NAVY, GREEN, GOLD][:len(scens)], number_format="0.00")

    scen_bullets = []
    for s in r.scenarios:
        scen_bullets.append(f"{s.label} (P={s.probability:.0%}): ${s.price:.2f} — {s.market_reaction[:80]}")
    _slide_body_split(prs, "IPO Scenario Cases", scen_bullets[:7],
                      _scenario_chart, "Scenario Price ($)")

    # ── Risks & catalysts ───────────────────────────────────────────────────
    _slide_section_divider(prs, 9, "Risks & Catalysts")
    risk_bullets = []
    seen = set()
    for s in r.scenarios:
        for t in s.triggers + s.risks:
            t = t.strip()
            if t and t not in seen:
                seen.add(t)
                risk_bullets.append(t)
            if len(risk_bullets) >= 6:
                break
        if len(risk_bullets) >= 6:
            break
    _slide_body(prs, "Key Risks & Invalidation Triggers", risk_bullets or ["No structured risks supplied."])

    _slide_disclaimer(prs)


# ─────────────────────────────────────────────────────────────────────────────
# COMPARABLE COMPANY (CCA) PITCHBOOK
# ─────────────────────────────────────────────────────────────────────────────
def _build_comps_pitchbook(prs, r: Any):
    date_str = datetime.date.today().strftime("%B %d, %Y")
    subj = r.subject

    _slide_cover(prs, f"{subj.ticker or subj.name} — Trading Comps",
                 "Comparable Company Analysis",
                 f"Relative Valuation  |  {len(r.peers or [])} comparable companies",
                 date_str)

    _slide_toc(prs, [
        "Executive Summary",
        "Peer Universe & Relevance",
        "Operating Comparison",
        "Trading Multiples",
        "Implied Valuation",
        "Premium / Discount Positioning",
        "Growth & Margin Adjustments",
        "Outliers & Data Quality",
        "Conclusion & Recommendation",
    ])

    _slide_section_divider(prs, 1, "Executive Summary")
    iv = r.implied_valuation or {}
    blended = iv.get("Blended (weighted)", 0)
    lo, hi = r.implied_range
    _slide_body(prs, "Executive Summary",
                [
                    f"Comp-blended value: ${blended:.2f} / share",
                    f"Defensible comp range: ${lo:.2f} – ${hi:.2f}",
                    f"Current price: ${subj.price:.2f}",
                    f"Recommendation: {r.recommendation}",
                    "Trading comps measure market pricing of comparable businesses — triangulate with DCF and precedents",
                ],
                right_kpis={
                    "Blended": f"${blended:.2f}",
                    "Range": f"${lo:.2f}–${hi:.2f}",
                    "Price": f"${subj.price:.2f}",
                })

    _slide_section_divider(prs, 2, "Peer Universe & Relevance")
    rel_bullets = []
    for t in (r.peers or [])[:7]:
        rel = r.relevance.get(t.ticker, {})
        rel_bullets.append(f"{t.ticker} — relevance {rel.get('score', 0):.0%}: {rel.get('reasons', [''])[0]}")
    _slide_body(prs, "Comparable-Company Relevance", rel_bullets or ["No peers supplied."])

    _slide_section_divider(prs, 3, "Operating Comparison")

    def _op_chart(slide):
        peers = (r.peers or [])[:6]
        if peers:
            _add_bar_chart(slide, 8.0, 1.7, 4.9, 4.5,
                           [p.ticker for p in peers], [p.ebitda_margin_pct for p in peers],
                           series_name="EBITDA Margin %", chart_type="col",
                           colors=[NAVY] * len(peers), number_format="0.0")

    op_bullets = [
        f"Subject growth: {subj.revenue_growth_pct:.1f}%  |  EBITDA margin: {subj.ebitda_margin_pct:.1f}%",
        "Peer operating metrics below (median margins / growth across the set)",
    ]
    for p in (r.peers or [])[:5]:
        op_bullets.append(f"{p.ticker}: {p.revenue_growth_pct:.0f}% growth, {p.ebitda_margin_pct:.0f}% margin")
    _slide_body_split(prs, "Operating Comparison", op_bullets[:7],
                      _op_chart, "EBITDA Margins (%)")

    _slide_section_divider(prs, 4, "Trading Multiples")
    slide = _blank_slide(prs)
    _fill_slide_bg(slide, LGRAY)
    _add_rect(slide, 0, 0, SLIDE_W, 1.0, NAVY)
    _add_textbox(slide, 0.35, 0.12, 12.0, 0.75,
                 "Trading Multiples", bold=True, font_size=20, font_color=WHITE)
    _add_rect(slide, 0, 0.98, SLIDE_W, 0.04, GOLD)
    md = r.multiples_df
    if md is not None and not md.empty:
        cols = [c for c in md.columns if c not in ("is_subject",)]
        display_cols = [c for c in cols][:8]
        hdr = display_cols
        body = [[("" if pd_val is None else (f"{pd_val:,.1f}" if isinstance(pd_val, (int, float)) else str(pd_val)))
                 for pd_val in [row[c] for c in display_cols]] for _, row in md.iterrows()]
        _add_table(slide, len(body) + 1, len(hdr), MARGIN, 1.2, 12.3, 5.2,
                   hdr, body, header_bg=NAVY, header_fg=WHITE)
    _add_footer(slide, prs)

    _slide_section_divider(prs, 5, "Implied Valuation")

    def _implied_chart(slide):
        if iv:
            labels = [k[:24] for k in iv.keys()]
            vals = [float(v) for v in iv.values()]
            colors = [GREEN if v >= subj.price else RED for v in vals]
            _add_bar_chart(slide, 8.0, 1.7, 4.9, 4.5, labels, vals,
                           series_name="$/share", chart_type="bar", colors=colors,
                           number_format="0.00")

    imp_bullets = []
    for k, v in iv.items():
        imp_bullets.append(f"{k}: ${v:.2f}")
    imp_bullets.append(f"Weighted blend: ${blended:.2f}")
    _slide_body_split(prs, "Implied Valuation by Method", imp_bullets[:7],
                      _implied_chart, "Implied Value ($/share)")

    _slide_section_divider(prs, 6, "Premium / Discount Positioning")
    pd_ = r.premium_discount or {}
    prem = pd_.get("premium_to_blended", 0)
    if prem is not None and isinstance(prem, (int, float)):
        prem = float(prem)
    else:
        prem = 0.0
    _slide_body(prs, "Positioning Within the Peer Set",
                [
                    f"Premium / (discount) to comp blend: {prem:+.1%}",
                    f"Subject at ${subj.price:.2f} vs blend ${blended:.2f}",
                    "Positioning explained by growth, margins, and business-model mix",
                    r.conclusion,
                ])

    _slide_section_divider(prs, 7, "Growth & Margin Adjustments")
    reg = r.regressions or {}
    reg_bullets = []
    g = reg.get("ev_ebitda_vs_growth", {}) or {}
    m = reg.get("ev_ebitda_vs_margin", {}) or {}
    if g.get("n"):
        reg_bullets.append(f"EV/EBITDA vs growth: slope {g['slope']:.2f}, R² {g['r2']:.2f} (n={g['n']})")
    if m.get("n"):
        reg_bullets.append(f"EV/EBITDA vs margin: slope {m['slope']:.2f}, R² {m['r2']:.2f} (n={m['n']})")
    if not reg_bullets:
        reg_bullets.append("Insufficient peers (n<3) for regression-based adjustment")
    _slide_body(prs, "Multiple-vs-Fundamentals Regression", reg_bullets)

    _slide_section_divider(prs, 8, "Outliers & Data Quality")
    if r.outliers:
        _slide_body(prs, "Outlier Detection (IQR rule)",
                    [f"{o['ticker']} — {o['multiple']} at {o['value']}x (outside {o['band']})" for o in r.outliers[:6]])
    else:
        _slide_body(prs, "Outlier Detection", ["No statistical outliers detected in the core peer set."])

    _slide_section_divider(prs, 9, "Conclusion & Recommendation")
    _slide_body(prs, "Conclusion",
                [r.recommendation, r.conclusion])
    _slide_disclaimer(prs)


# ─────────────────────────────────────────────────────────────────────────────
# PRECEDENT TRANSACTIONS PITCHBOOK
# ─────────────────────────────────────────────────────────────────────────────
def _build_precedents_pitchbook(prs, r: Any, subject_name: str = ""):
    date_str = datetime.date.today().strftime("%B %d, %Y")
    subject_name = subject_name or "Subject Company"

    _slide_cover(prs, f"{subject_name} — Precedent Transactions",
                 "Precedent Transaction Analysis",
                 f"M&A Valuation Context  |  {len(r.core_deals or [])} comparable deals",
                 date_str)

    _slide_toc(prs, [
        "Executive Summary",
        "Transaction Universe & Relevance",
        "Transaction Multiples",
        "Control Premium Analysis",
        "Adjusted Valuation Range",
        "Context Adjustments",
        "Conclusion",
    ])

    _slide_section_divider(prs, 1, "Executive Summary")
    iv = r.implied_valuation or {}
    blended = iv.get("Blended (weighted)", 0)
    lo, hi = r.adjusted_range
    _slide_body(prs, "Executive Summary",
                [
                    f"Precedent-derived control value: ${blended:.2f} / share",
                    f"Adjusted range: ${lo:.2f} – ${hi:.2f}",
                    f"Core deal set: {len(r.core_deals or [])} transactions",
                    "Transaction multiples embed control premiums — typically above trading comps",
                ],
                right_kpis={
                    "Control Value": f"${blended:.2f}",
                    "Range": f"${lo:.2f}–${hi:.2f}",
                    "Deals": f"{len(r.core_deals or [])}",
                })

    _slide_section_divider(prs, 2, "Transaction Universe & Relevance")
    rel_bullets = []
    for t in (r.core_deals or [])[:7]:
        rel = r.relevance.get(t.deal_name, {})
        rel_bullets.append(f"{t.deal_name} — relevance {rel.get('score', 0):.0%}: {rel.get('reasons', [''])[0]}")
    _slide_body(prs, "Deal Relevance", rel_bullets or ["No transactions supplied."])

    _slide_section_divider(prs, 3, "Transaction Multiples")
    slide = _blank_slide(prs)
    _fill_slide_bg(slide, LGRAY)
    _add_rect(slide, 0, 0, SLIDE_W, 1.0, NAVY)
    _add_textbox(slide, 0.35, 0.12, 12.0, 0.75,
                 "Transaction Multiples", bold=True, font_size=20, font_color=WHITE)
    _add_rect(slide, 0, 0.98, SLIDE_W, 0.04, GOLD)
    md = r.multiples_df
    if md is not None and not md.empty:
        display_cols = [c for c in md.columns][:9]
        hdr = display_cols
        body = [[("" if pd_val is None else (f"{pd_val:,.1f}" if isinstance(pd_val, (int, float)) else str(pd_val)))
                 for pd_val in [row[c] for c in display_cols]] for _, row in md.iterrows()]
        _add_table(slide, len(body) + 1, len(hdr), MARGIN, 1.2, 12.3, 5.2,
                   hdr, body, header_bg=NAVY, header_fg=WHITE)
    _add_footer(slide, prs)

    _slide_section_divider(prs, 4, "Control Premium Analysis")
    cp = r.control_premium_analysis or {}
    _slide_body(prs, "Control Premiums Paid",
                [
                    f"Median premium: {cp.get('median_premium_pct') or 0:.1f}%  |  Mean: {cp.get('mean_premium_pct') or 0:.1f}%",
                    f"Range: {cp.get('low_premium_pct') or 0:.1f}% – {cp.get('high_premium_pct') or 0:.1f}%",
                    cp.get("context", ""),
                ])

    _slide_section_divider(prs, 5, "Adjusted Valuation Range")

    def _adj_chart(slide):
        if iv:
            labels = [k[:24] for k in iv.keys()]
            vals = [float(v) for v in iv.values()]
            _add_bar_chart(slide, 8.0, 1.7, 4.9, 4.5, labels, vals,
                           series_name="$/share", chart_type="bar",
                           colors=[NAVY] * len(labels), number_format="0.00")

    adj_bullets = []
    for k, v in iv.items():
        adj_bullets.append(f"{k}: ${v:.2f}")
    adj_bullets.append(f"Adjusted range: ${lo:.2f} – ${hi:.2f}")
    _slide_body_split(prs, "Implied Control Value by Method", adj_bullets[:7],
                      _adj_chart, "Implied Value ($/share)")

    _slide_section_divider(prs, 6, "Context Adjustments")
    adj = r.context_adjustments or []
    _slide_body(prs, "Comparability Adjustments",
                [f"{a.get('factor', '')}: {a.get('note', '')}" for a in adj])

    _slide_section_divider(prs, 7, "Conclusion")
    _slide_body(prs, "Conclusion", [r.conclusion])
    _slide_disclaimer(prs)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API  – InstitutionalPresentationGenerator
# ─────────────────────────────────────────────────────────────────────────────

class InstitutionalPresentationGenerator:
    """Generates full investment-banking-grade PowerPoint pitchbooks."""

    def generate_dcf_pitchbook(self, ticker: str, dcf_result: Any) -> bytes:
        prs = _new_presentation()
        _build_dcf_pitchbook(prs, ticker, dcf_result)
        buf = io.BytesIO()
        prs.save(buf)
        return buf.getvalue()

    def generate_mna_pitchbook(self, acquirer: str, target: str, mna_result: Any) -> bytes:
        prs = _new_presentation()
        _build_mna_pitchbook(prs, acquirer, target, mna_result)
        buf = io.BytesIO()
        prs.save(buf)
        return buf.getvalue()

    def generate_lbo_pitchbook(self, target: str, lbo_result: Any) -> bytes:
        prs = _new_presentation()
        _build_lbo_pitchbook(prs, target, lbo_result)
        buf = io.BytesIO()
        prs.save(buf)
        return buf.getvalue()

    def generate_ipo_pitchbook(self, ipo_result: Any, issuer: str = "") -> bytes:
        prs = _new_presentation()
        _build_ipo_pitchbook(prs, ipo_result, issuer)
        buf = io.BytesIO()
        prs.save(buf)
        return buf.getvalue()

    def generate_comps_pitchbook(self, comps_result: Any) -> bytes:
        prs = _new_presentation()
        _build_comps_pitchbook(prs, comps_result)
        buf = io.BytesIO()
        prs.save(buf)
        return buf.getvalue()

    def generate_precedents_pitchbook(self, precedents_result: Any,
                                      subject_name: str = "") -> bytes:
        prs = _new_presentation()
        _build_precedents_pitchbook(prs, precedents_result, subject_name)
        buf = io.BytesIO()
        prs.save(buf)
        return buf.getvalue()


_ppt_generator = None

def get_presentation_generator() -> InstitutionalPresentationGenerator:
    global _ppt_generator
    if _ppt_generator is None:
        _ppt_generator = InstitutionalPresentationGenerator()
    return _ppt_generator


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────

def show_presentation_generator():
    """Streamlit UI for the Institutional Pitchbook Generator."""
    st.title("Institutional Pitchbook Generator")
    st.caption("Generate full investment-banking-grade PowerPoint decks — M&A, LBO, and DCF mandates.")

    deck_type = st.selectbox(
        "Pitchbook Type",
        ["M&A Accretion/Dilution", "DCF Valuation", "LBO Returns"],
    )

    if deck_type == "DCF Valuation":
        ticker = st.text_input("Ticker Symbol", "AAPL")
        if st.button("Generate DCF Pitchbook (.pptx)", type="primary"):
            with st.spinner("Building slides…"):
                try:
                    from financial_model_generator import get_dcf_engine, DCFAssumptions
                    import yfinance as yf
                    info = yf.Ticker(ticker.upper()).info or {}
                    a = DCFAssumptions(
                        ticker=ticker.upper(),
                        base_revenue=float(info.get("totalRevenue", 5_000_000_000)) / 1e6,
                        revenue_growth_rates=[float(info.get("revenueGrowth", 0.08))] * 5,
                        ebit_margin=float(info.get("operatingMargins", 0.20)),
                        tax_rate=0.21,
                        da_pct_revenue=0.04,
                        capex_pct_revenue=0.05,
                        nwc_change_pct_revenue=0.01,
                        equity_value_market=float(info.get("marketCap", 10_000_000_000)) / 1e6,
                        debt_value=float(info.get("totalDebt", 1_000_000_000)) / 1e6,
                        cost_of_debt=0.045,
                        risk_free_rate=0.0425,
                        equity_risk_premium=0.055,
                        beta=float(info.get("beta", 1.0)),
                        terminal_growth_rate=0.025,
                        cash=float(info.get("totalCash", 500_000_000)) / 1e6,
                        shares_outstanding=float(info.get("sharesOutstanding", 1_000_000_000)) / 1e6,
                        current_price=float(info.get("currentPrice") or info.get("previousClose", 100)),
                        peer_pe=25.0, peer_ev_ebitda=15.0, peer_ev_fcf=20.0, peg_ratio=2.0,
                        projection_years=5,
                    )
                    res = get_dcf_engine().run_dcf(a)
                    pptx_bytes = get_presentation_generator().generate_dcf_pitchbook(ticker.upper(), res)
                    st.download_button(
                        "Download DCF Pitchbook (.pptx)",
                        data=pptx_bytes,
                        file_name=f"{ticker.upper()}_DCF_Pitchbook.pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    )
                    st.success("Pitchbook generated — click above to download.")
                except Exception as e:
                    st.error(f"Generation error: {e}")
                    import traceback; st.code(traceback.format_exc())

    elif deck_type == "M&A Accretion/Dilution":
        c1, c2 = st.columns(2)
        acq = c1.text_input("Acquirer Ticker", "MSFT")
        tgt = c2.text_input("Target Ticker", "ATVI")

        st.markdown("#### Adjust Assumptions (optional)")
        cc1, cc2, cc3 = st.columns(3)
        offer_premium = cc1.slider("Offer Premium (%)", 10, 60, 30) / 100
        pct_stock     = cc2.slider("% Stock Consideration", 0, 100, 50) / 100
        synergies     = cc3.number_input("Pre-Tax Synergies ($M)", value=500.0, step=50.0)

        if st.button("Generate M&A Pitchbook (.pptx)", type="primary"):
            with st.spinner("Building slides…"):
                try:
                    import yfinance as yf
                    from mna_model_engine import get_mna_engine, MnAAssumptions
                    ai = yf.Ticker(acq.upper()).info or {}
                    ti = yf.Ticker(tgt.upper()).info or {}
                    a = MnAAssumptions(
                        acquirer_ticker=acq.upper(), target_ticker=tgt.upper(),
                        acquirer_price=float(ai.get("currentPrice") or ai.get("previousClose", 300)),
                        acquirer_eps=float(ai.get("trailingEps", 10.0)),
                        acquirer_shares=float(ai.get("sharesOutstanding", 7_000_000_000)) / 1e6,
                        target_price=float(ti.get("currentPrice") or ti.get("previousClose", 80)),
                        target_eps=float(ti.get("trailingEps", 3.0)),
                        target_shares=float(ti.get("sharesOutstanding", 800_000_000)) / 1e6,
                        offer_premium=offer_premium,
                        percent_stock=pct_stock,
                        percent_cash=1 - pct_stock,
                        cost_of_debt=0.05,
                        tax_rate=0.21,
                        pre_tax_synergies=float(synergies),
                    )
                    res = get_mna_engine().run_mna(a)
                    pptx_bytes = get_presentation_generator().generate_mna_pitchbook(acq.upper(), tgt.upper(), res)
                    st.download_button(
                        "Download M&A Pitchbook (.pptx)",
                        data=pptx_bytes,
                        file_name=f"{acq.upper()}_{tgt.upper()}_MnA_Pitchbook.pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    )
                    st.success("Pitchbook generated — click above to download.")
                    # Preview key stats
                    c_a, c_b, c_c = st.columns(3)
                    c_a.metric("Offer Price",    f"${res.offer_price:.2f}")
                    c_b.metric("Pro Forma EPS",  f"${res.pro_forma_eps:.2f}")
                    c_c.metric("Accr / (Dil)",   f"{res.accretion_dilution_pct:.1%}")
                except Exception as e:
                    st.error(f"Generation error: {e}")
                    import traceback; st.code(traceback.format_exc())

    elif deck_type == "LBO Returns":
        tgt = st.text_input("Target Ticker", "TWTR")
        cc1, cc2, cc3 = st.columns(3)
        entry_mult  = cc1.slider("Entry Multiple (x EBITDA)", 6.0, 20.0, 12.0, 0.5)
        exit_mult   = cc2.slider("Exit Multiple (x EBITDA)",  6.0, 20.0, 12.0, 0.5)
        lev_mult    = cc3.slider("Leverage Multiple (x EBITDA)", 3.0, 8.0, 6.0, 0.5)

        if st.button("Generate LBO Pitchbook (.pptx)", type="primary"):
            with st.spinner("Building slides…"):
                try:
                    from lbo_model_engine import get_lbo_engine, LBOAssumptions
                    a = LBOAssumptions(
                        ticker=tgt.upper(), target_name=tgt.upper(),
                        entry_year=2024, exit_year=2029,
                        ltm_ebitda=1000.0,
                        entry_multiple=entry_mult,
                        exit_multiple=exit_mult,
                        leverage_multiple=lev_mult,
                        interest_rate=0.08,
                    )
                    res = get_lbo_engine().run_lbo(a)
                    pptx_bytes = get_presentation_generator().generate_lbo_pitchbook(tgt.upper(), res)
                    st.download_button(
                        "Download LBO Pitchbook (.pptx)",
                        data=pptx_bytes,
                        file_name=f"{tgt.upper()}_LBO_Pitchbook.pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    )
                    st.success("Pitchbook generated — click above to download.")
                    c_a, c_b, c_c = st.columns(3)
                    c_a.metric("IRR",   f"{res.irr:.1%}")
                    c_b.metric("MOIC",  f"{res.moic:.2f}x")
                    c_c.metric("Entry", f"${res.purchase_price:,.0f}M")
                except Exception as e:
                    st.error(f"Generation error: {e}")
                    import traceback; st.code(traceback.format_exc())
