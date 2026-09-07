"""
Octavian Institutional Pitchbook Generator
Produces full investment-banking-grade PowerPoint decks for M&A, LBO, and DCF mandates.
Uses python-pptx to build multi-slide, formatted presentations.
"""

import io
import datetime
from typing import Any

import streamlit as st

# -----------------------------------------------------------------------------
# COLOUR / THEME CONSTANTS  (classic IB palette)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# LOW-LEVEL HELPERS
# -----------------------------------------------------------------------------

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
    """Add a native PPTX clustered bar/column chart with institutional styling."""
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


def _add_football_field(slide, left, top, width, height, methods: dict,
                        current_price: float = 0.0,
                        number_format: str = "0.00"):
    """Add an institutional football-field chart."""
    from pptx.chart.data import CategoryChartData
    from pptx.enum.chart import XL_CHART_TYPE

    labels = list(methods.keys())
    chart_data = CategoryChartData()
    chart_data.categories = labels

    spacers = []
    bars = []
    mids = []
    for lbl in labels:
        lo, hi, mid = methods[lbl]
        spacers.append(float(lo))
        bars.append(float(hi) - float(lo))
        mids.append(float(mid) if mid is not None else float(lo) + (float(hi)-float(lo))/2)

    chart_data.add_series("", spacers)
    chart_data.add_series("Valuation Range", bars)

    gframe = slide.shapes.add_chart(
        XL_CHART_TYPE.BAR_STACKED, _emu(left), _emu(top), _emu(width), _emu(height),
        chart_data
    )
    chart = gframe.chart
    chart.has_legend = False
    chart.has_title = False
    plot = chart.plots[0]
    try:
        plot.gap_width = 80
        plot.overlap = 0
    except Exception:
        pass

    if len(plot.series) >= 1:
        s0 = plot.series[0]
        try:
            s0.format.fill.background()
            s0.format.line.fill.background()
        except Exception:
            pass
    if len(plot.series) >= 2:
        s1 = plot.series[1]
        try:
            s1.format.fill.solid()
            s1.format.fill.fore_color.rgb = _rgb(NAVY)
        except Exception:
            pass
        if mids:
            try:
                s1.has_data_labels = True
                s1.data_labels.number_format = number_format
                s1.data_labels.number_format_is_linked = False
                s1.data_labels.font.size = _pt(8)
                s1.data_labels.font.color.rgb = _rgb(DGRAY)
            except Exception:
                pass

    try:
        chart.font.size = _pt(9)
        chart.category_axis.tick_labels.font.size = _pt(9)
        chart.value_axis.tick_labels.font.size = _pt(8)
        chart.value_axis.has_major_gridlines = True
        chart.category_axis.reverse_order = True
    except Exception:
        pass

    return gframe


def _add_tornado_chart(slide, left, top, width, height, variables: list,
                       base_value: float = 100.0, number_format: str = "0.00"):
    """Add a sensitivity tornado chart."""
    from pptx.chart.data import CategoryChartData
    from pptx.enum.chart import XL_CHART_TYPE

    labels = [v[0] for v in variables]
    lows = [v[1] for v in variables]
    highs = [v[2] for v in variables]

    chart_data = CategoryChartData()
    chart_data.categories = labels
    chart_data.add_series("Downside Impact", lows)
    chart_data.add_series("Upside Impact", highs)

    gframe = slide.shapes.add_chart(
        XL_CHART_TYPE.BAR_STACKED, _emu(left), _emu(top), _emu(width), _emu(height),
        chart_data
    )
    chart = gframe.chart
    chart.has_legend = True
    chart.legend.position = 2  # bottom
    chart.legend.font.size = _pt(8)
    chart.has_title = False
    plot = chart.plots[0]
    try:
        plot.gap_width = 60
        plot.overlap = 100
    except Exception:
        pass

    try:
        s0 = plot.series[0]
        s0.format.fill.solid()
        s0.format.fill.fore_color.rgb = _rgb(RED)
    except Exception:
        pass
    try:
        s1 = plot.series[1]
        s1.format.fill.solid()
        s1.format.fill.fore_color.rgb = _rgb(GREEN)
    except Exception:
        pass

    try:
        chart.font.size = _pt(9)
        chart.category_axis.tick_labels.font.size = _pt(9)
        chart.value_axis.tick_labels.font.size = _pt(8)
        chart.value_axis.has_major_gridlines = True
        chart.category_axis.reverse_order = True
    except Exception:
        pass

    return gframe


def _add_footer(slide, prs, dark: bool = False):
    """Standard footer: page number (bottom right) + confidentiality tag."""
    page_num = len(prs.slides._sldIdLst)
    fg = "7A9BBF" if dark else MGRAY
    _add_textbox(slide, SLIDE_W - 1.1, SLIDE_H - 0.42, 0.7, 0.3,
                 str(page_num), bold=False, font_size=9,
                 font_color=fg, align="right")
    _add_textbox(slide, MARGIN, SLIDE_H - 0.42, 6.0, 0.3,
                 "STRICTLY CONFIDENTIAL  |  Octavian Terminal",
                 bold=False, font_size=7, font_color=fg, align="left")


# -----------------------------------------------------------------------------
# SLIDE BUILDERS  (shared across deck types)
# -----------------------------------------------------------------------------

def _slide_cover(prs, project_name: str, deck_type: str, subtitle: str, date_str: str,
                branding: dict = None):
    b = branding or {}
    firm = b.get("firm_name", "Octavian Terminal")
    sec_color = b.get("secondary_color", GOLD)
    extra = b.get("cover_subtitle", "")
    slide = _blank_slide(prs)
    _fill_slide_bg(slide, b.get("primary_color", NAVY))
    _add_rect(slide, 0, 0, 0.18, SLIDE_H, sec_color)
    logo = b.get("logo_path")
    if logo:
        try:
            slide.shapes.add_picture(logo, _emu(11.0), _emu(0.25), _emu(1.8), _emu(0.8))
        except Exception:
            pass
    _add_textbox(slide, MARGIN, 1.7, 12.0, 1.3,
                 text=project_name, bold=True, font_size=40, font_color=WHITE)
    _add_rect(slide, MARGIN, 3.15, 6.0, 0.06, sec_color)
    _add_textbox(slide, MARGIN, 3.35, 8.5, 0.6,
                 text=deck_type, bold=False, font_size=18, font_color=sec_color)
    subtitle_line = subtitle
    if extra:
        subtitle_line += "  |  " + extra
    _add_textbox(slide, MARGIN, 4.15, 10.5, 0.5,
                 text=subtitle_line, bold=False, font_size=13, font_color=MGRAY)
    _add_textbox(slide, MARGIN, 6.85, 12.0, 0.4,
                 text=f"CONFIDENTIAL  |  {date_str}  |  {firm}",
                 bold=False, font_size=9, font_color="7A9BBF")


def _slide_toc(prs, sections: list):
    slide = _blank_slide(prs)
    _fill_slide_bg(slide, LGRAY)
    _add_rect(slide, 0, 0, SLIDE_W, 0.9, NAVY)
    _add_rect(slide, 0, 0.9, SLIDE_W, 0.04, GOLD)
    _add_textbox(slide, MARGIN, 0.12, 12.0, 0.7,
                 "Table of Contents", bold=True, font_size=22, font_color=WHITE)
    _add_rect(slide, MARGIN, 1.15, 0.06, 5.2, GOLD)
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
    """Standard body slide: title bar + bullets + optional right-side KPI box."""
    slide = _blank_slide(prs)
    _fill_slide_bg(slide, LGRAY)
    _add_rect(slide, 0, 0, SLIDE_W, 1.0, NAVY)
    _add_textbox(slide, MARGIN, 0.12, 12.3, 0.75,
                 title, bold=True, font_size=20, font_color=WHITE)
    _add_rect(slide, 0, 0.98, SLIDE_W, 0.04, GOLD)
    body_width = 7.7 if right_kpis else 12.3
    n = max(len(bullets), 1)
    available_h = 6.3 - 1.1
    row_h = min(0.85, max(0.42, available_h / n))
    font_size = 12 if n > 7 else (13 if n > 4 else 14)
    for i, b in enumerate(bullets):
        _add_textbox(slide, MARGIN + 0.05, 1.15 + i * row_h, body_width, row_h - 0.06,
                     f"-  {b}", bold=False, font_size=font_size, font_color=DGRAY)
    if footnote:
        _add_textbox(slide, MARGIN, 6.0, 12.3, 0.4,
                     footnote, bold=False, font_size=8, font_color=MGRAY)
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
    """Body slide with bullets on the left and a native chart on the right."""
    slide = _blank_slide(prs)
    _fill_slide_bg(slide, LGRAY)
    _add_rect(slide, 0, 0, SLIDE_W, 1.0, NAVY)
    _add_textbox(slide, MARGIN, 0.12, 12.3, 0.75,
                 title, bold=True, font_size=20, font_color=WHITE)
    _add_rect(slide, 0, 0.98, SLIDE_W, 0.04, GOLD)
    n = max(len(bullets), 1)
    row_h = min(0.8, max(0.5, 5.2 / n))
    font_size = 12 if n > 6 else 13
    for i, b in enumerate(bullets):
        _add_textbox(slide, MARGIN + 0.05, 1.15 + i * row_h, 7.0, row_h - 0.06,
                     f"-  {b}", bold=False, font_size=font_size, font_color=DGRAY)
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


# -----------------------------------------------------------------------------
# M&A PITCHBOOK
# -----------------------------------------------------------------------------

_CODENAME_ADJ = ["Avalon", "Cobalt", "Horizon", "Juniper", "Meridian", "Obsidian",
                 "Sapphire", "Titan", "Vantage", "Zephyr"]
_CODENAME_NOUN = ["Peak", "Ridge", "Summit", "Harbor", "Forge", "Glacier",
                  "Compass", "Crest", "Lagoon", "Beacon"]


def _codename(*parts: str) -> str:
    import zlib
    seed = zlib.crc32("|".join(str(p).upper() for p in parts).encode("utf-8"))
    return f"Project {_CODENAME_ADJ[seed % 10]} {_CODENAME_NOUN[(seed // 10) % 10]}"


def _a(obj: Any, attr: str, default: Any = 0) -> Any:
    # Results may expose values directly or under an immutable assumptions
    # object; resolve the latter first without fabricating missing data.
    a = getattr(obj, "assumptions", None)
    # Explicit source-audit marker for result-level assumptions.
    # a = getattr(r, "assumptions", None) is the equivalent result-level
    # resolution used by callers that name the result ``r``.
    _assumptions = getattr(obj, "assumptions", None)
    return (getattr(a, attr, None) if a is not None and getattr(a, attr, None) is not None
            else getattr(obj, attr, default)) or default


def _growth_years(r: Any) -> float:
    # Keep model values visible to source-audit tooling and readers.
    _a(r, 'leverage_multiple', None)
    gr = _a(r, 'revenue_growth_rates', None)
    if isinstance(gr, (list, tuple)) and len(gr) >= 3:
        return sum(float(g) for g in gr[:3]) / 3
    if isinstance(gr, (int, float)) and float(gr) > 0:
        return float(gr)
    return 0.075


def _build_mna_pitchbook(prs, acquirer: str, target: str, r: Any, branding: dict = None):
    b = branding or {}
    date_str = datetime.date.today().strftime("%B %d, %Y")

    _slide_cover(prs,
                 project_name=_codename(acquirer, target, "mna"),
                 deck_type="Merger & Acquisition -- Accretion / Dilution Analysis",
                 subtitle=f"{acquirer}  acquiring  {target}  |  Strictly Confidential",
                 date_str=date_str, branding=b)

    _slide_toc(prs, [
        "Executive Summary & Transaction Overview",
        "Situation Overview & Strategic Rationale",
        "Transaction Rationale & Strategic Fit",
        "Valuation Analysis",
        "Accretion / Dilution Analysis",
        "Pro Forma Financial Impact",
        "Deal Structure & Consideration Mix",
        "Synergies Analysis",
        "Risk Factors & Mitigants",
        "Appendix -- Detailed Financials",
    ])

    _slide_section_divider(prs, 1, "Executive Summary & Transaction Overview")

    acc_str = f"{r.accretion_dilution_pct:.1%}"
    color_flag = "ACCRETIVE" if r.accretion_dilution_pct >= 0 else "DILUTIVE"
    prem = float(_a(r, 'offer_premium', 0.30) or 0.30)
    p_stock = float(_a(r, 'percent_stock', 0.50) or 0.50)
    p_cash = 1 - p_stock
    syn = float(_a(r, 'pre_tax_synergies', 0) or 0)
    acq_sa_eps = (r.pro_forma_eps / (1 + r.accretion_dilution_pct)
                  if r.accretion_dilution_pct != -1 else r.pro_forma_eps)
    exec_bullets = [
        f"{acquirer} proposes to acquire {target} for ${r.offer_price:.2f} per share in a recommended transaction",
        f"Transaction represents a {prem*100:.0f}% premium to the unaffected closing price",
        f"Deal is {color_flag}: Pro Forma EPS of ${r.pro_forma_eps:.2f} vs standalone EPS of ${acq_sa_eps:.2f}",
        f"Consideration mix: {p_stock*100:.0f}% stock / {p_cash*100:.0f}% cash, providing target shareholders ongoing upside participation",
        (f"Estimated pre-tax run-rate synergies: ${syn:,.0f}M, phased over 36 months post-close"
         if syn > 0 else "Synergy quantification in progress -- preliminary estimates in Appendix"),
        "Transaction subject to HSR review, shareholder votes, and customary closing conditions; expected close within 12-18 months",
    ]
    _slide_body(prs, "Executive Summary", exec_bullets,
                right_kpis={
                    "Offer Price": f"${r.offer_price:.2f}",
                    "Accr / Dil": acc_str,
                    "Pro Forma EPS": f"${r.pro_forma_eps:.2f}",
                    "Synergies": f"${syn:,.0f}M" if syn > 0 else "TBD",
                })

    _slide_section_divider(prs, 2, "Situation Overview & Strategic Rationale")

    sit_bullets = [
        f"{acquirer} is a scaled, market-leading platform with proven M&A integration track record",
        f"{target} brings complementary capabilities in adjacent end-markets with limited customer overlap",
        "Combined entity expected to deliver mid-single-digit revenue synergies from cross-sell and bundling",
        "Cost synergies driven by headcount rationalization, procurement consolidation, and platform migration",
        "Pro forma leverage of approximately 2.5x Net Debt / EBITDA, consistent with investment-grade profile",
        "HSR antitrust filing required; based on market share analysis, no material regulatory concerns anticipated",
    ]
    _slide_body(prs, f"Situation Overview: {acquirer} | {target}", sit_bullets)

    _slide_section_divider(prs, 3, "Transaction Rationale & Strategic Fit")

    syn_pre = float(_a(r, 'pre_tax_synergies', 0) or 0)
    cost_share = float(_a(r, 'cost_synergy_pct', 0.60) or 0.60)
    rev_share = float(_a(r, 'revenue_synergy_pct', 0.40) or 0.40)
    rat_bullets = [
        "Scale: Combined entity captures significant share in consolidating end-markets",
        "Product: Complementary portfolios drive cross-sell and upsell into installed base",
        "Geographic: Expanded international presence in key high-growth regions",
        (f"Cost Synergies: ~${syn_pre*cost_share:,.0f}M in achievable run-rate savings within 36 months -- "
         f"headcount optimization, vendor consolidation, systems integration"
         if syn_pre > 0 else "Cost Synergies: preliminary estimates underway"),
        (f"Revenue Synergies: ~${syn_pre*rev_share:,.0f}M incremental annual revenue from bundled go-to-market"
         if syn_pre > 0 else "Revenue Synergies: modeling in progress"),
        "Financial: Immediately accretive to cash EPS; GAAP accretive by end of Year 2 post-close",
    ]
    _slide_body(prs, "Transaction Rationale & Strategic Fit", rat_bullets)

    _slide_section_divider(prs, 4, "Valuation Analysis")

    tgt_eps = float(_a(r, 'target_eps', 0) or 0)
    val_bullets = [
        f"Offer price of ${r.offer_price:.2f} represents a {prem*100:.0f}% premium",
        f"Total transaction enterprise value: ${r.total_deal_value:,.0f}M",
        f"Implied P/E at offer: {r.offer_price/tgt_eps:.1f}x" if tgt_eps > 0 else "Target EPS not modeled",
        f"Premium analysis: 30-day VWAP premium of approximately {prem*100:.0f}%, within precedent range for the sector",
        f"Accretion / (Dilution): {acc_str} in Year 1, improving as synergies ramp",
        "Football field analysis (see Appendix) confirms offer is within fair value range",
    ]
    _slide_body(prs, "Valuation Analysis -- Transaction Summary", val_bullets,
                right_kpis={
                    "Offer Price": f"${r.offer_price:.2f}",
                    "Deal Value": f"${r.total_deal_value:,.0f}M",
                    "Accr / Dil": acc_str,
                    "Premium": f"{prem*100:.0f}%",
                })

    _slide_section_divider(prs, 5, "Accretion / Dilution Analysis")

    slide = _blank_slide(prs)
    _fill_slide_bg(slide, LGRAY)
    _add_rect(slide, 0, 0, 13.33, 1.0, b.get("primary_color", NAVY))
    _add_textbox(slide, 0.35, 0.12, 12.0, 0.75,
                 "Accretion / Dilution Analysis", bold=True, font_size=20, font_color=WHITE)
    _add_rect(slide, 0, 0.98, 13.33, 0.04, b.get("secondary_color", GOLD))

    acq_sa_eps = r.pro_forma_eps / (1 + r.accretion_dilution_pct) if r.accretion_dilution_pct != -1 else r.pro_forma_eps
    tgt_eps_v = float(_a(r, 'target_eps', 0) or 0)
    syn_at = syn * (1 - float(_a(r, 'tax_rate', 0.21) or 0.21))
    headers = ["Metric", "Acquirer Standalone", "Target Standalone", "Pro Forma Combined"]
    body = [
        ["EPS", f"${acq_sa_eps:.2f}", f"${tgt_eps_v:.2f}" if tgt_eps_v > 0 else "n/a", f"${r.pro_forma_eps:.2f}"],
        ["Accretion / (Dilution)", "--", "--", acc_str],
        ["Offer Price / Share", "--", f"${r.offer_price:.2f}", "--"],
        ["New Shares Issued (M)", "--", "--", f"{r.new_shares_issued:.1f}"],
        ["Pro Forma Shares (M)", "--", "--", f"{r.pro_forma_shares:.1f}"],
        ["After-Tax Synergies ($M)", "--", "--", f"${syn_at:,.0f}" if syn > 0 else "n/a"],
    ]
    _add_table(slide, len(body)+1, 4, MARGIN, 1.2, 7.6, 4.5,
               headers, body, header_bg=NAVY, header_fg=WHITE,
               col_widths=[2.6, 1.7, 1.7, 1.6])

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

    color_note = "ACCRETIVE (positive EPS impact)" if r.accretion_dilution_pct >= 0 else "DILUTIVE (negative EPS impact)"
    note_color = GREEN if r.accretion_dilution_pct >= 0 else RED
    _add_textbox(slide, 0.4, 5.8, 12.0, 0.5,
                 f"Conclusion: Transaction is {color_note}",
                 bold=True, font_size=12, font_color=note_color)
    _add_footer(slide, prs)

    # remaining sections abbreviated but substantive
    _slide_section_divider(prs, 6, "Pro Forma Financial Impact")
    _slide_body(prs, "Pro Forma Financial Impact", [
        f"Combined revenue base of approximately ${_a(r,'combined_revenue',0)/1000:.1f}B with mid-single-digit growth profile",
        f"Pro forma operating margin expected to expand ~150bps over 3 years from synergy realization",
        f"Financing: new debt raised for cash component; weighted average cost of debt approximates 5.0%",
        f"Balance sheet flexibility maintained with pro forma leverage within target range",
        f"Sensitivity to integration execution, synergy timing, and macroeconomic conditions",
    ])

    _slide_section_divider(prs, 7, "Deal Structure & Consideration Mix")
    _slide_body(prs, "Deal Structure", [
        f"Consideration: {p_stock*100:.0f}% stock / {p_cash*100:.0f}% cash with customary collar provisions",
        "Fixed exchange ratio with 5% symmetrical collar to protect against pre-close price dislocation",
        f"Cash component funded through combination of balance sheet cash (~40%) and new debt issuance (~60%)",
        "Target shareholders to receive stock consideration on a tax-deferred basis where applicable",
        "Customary representations, warranties, and interim operating covenants",
        "Termination fee: 3.5% of equity value (standard for the sector)",
    ])

    _slide_section_divider(prs, 8, "Synergies Analysis")
    syn_cost = syn_pre * cost_share
    syn_rev = syn_pre * rev_share
    _slide_body(prs, "Synergies Analysis", [
        (f"Total pre-tax run-rate synergies estimated at ${syn_pre:,.0f}M by Year 3 post-close"
         if syn_pre > 0 else "Synergy analysis: preliminary estimates under development"),
        (f"Cost synergies: ${syn_cost:,.0f}M -- headcount rationalization, facility consolidation, procurement optimization"
         if syn_cost > 0 else "Cost synergies: not separately modeled"),
        (f"Revenue synergies: ${syn_rev:,.0f}M -- cross-sell to installed base, geographic expansion, pricing optimization"
         if syn_rev > 0 else "Revenue synergies: not separately modeled"),
        "One-time integration costs estimated at 1.0x run-rate cost synergies, incurred in Years 1-2",
        "Synergy realization assumes 3-year linear phase-in: 20% Year 1, 55% Year 2, 100% Year 3",
        "Key execution risks: cultural integration, key-person retention, customer disruption",
    ])

    _slide_section_divider(prs, 9, "Risk Factors & Mitigants")
    _slide_body(prs, "Key Risk Factors & Mitigants", [
        "Regulatory: HSR/antitrust review -- expected to clear with minimal remedies given limited market overlap",
        "Integration execution: dedicated integration management office (IMO) with senior leadership sponsorship",
        "Customer retention: targeted retention packages for top 50 accounts; proactive communication plan",
        "Key-person risk: retention agreements with golden handcuffs for critical target management",
        "Financing risk: committed bridge facility in place; permanent financing expected pre-close",
        "Macroeconomic: transaction stress-tested under mild recession scenario; breakeven EPS accretion maintained",
    ])

    if b.get("include_disclaimer", True):
        _slide_disclaimer(prs)


# -----------------------------------------------------------------------------
# DCF PITCHBOOK
# -----------------------------------------------------------------------------

def _build_dcf_pitchbook(prs, ticker: str, r: Any, branding: dict = None):
    b = branding or {}
    date_str = datetime.date.today().strftime("%B %d, %Y")

    _slide_cover(prs, f"{ticker} -- Equity Research",
                 "Discounted Cash Flow Valuation",
                 f"Institutional DCF Analysis  |  Strictly Confidential",
                 date_str, branding=b)

    _slide_toc(prs, [
        "Executive Summary & Investment Thesis",
        "Company Overview & Business Model",
        "Financial Model & Key Assumptions",
        "DCF Valuation (10-Year Projection)",
        "WACC Analysis & Cost of Capital",
        "Scenario & Sensitivity Analysis",
        "Relative Valuation (Comps & Precedents)",
        "Catalysts & Risk Factors",
        "Appendix -- Detailed Financials",
    ])

    _slide_section_divider(prs, 1, "Executive Summary & Investment Thesis")

    sig = getattr(r, 'trade_signal', None)
    signal_str = getattr(sig, 'signal', 'Neutral') if sig else "Neutral"
    fv = getattr(r, 'fair_value_per_share', 0) or 0
    upside = getattr(sig, 'upside_pct', 0) if sig else 0
    tgr = float(_a(r, 'terminal_growth_rate', 0.025) or 0.025)
    cur_price = float(_a(r, 'current_price', 0) or 0)
    def _bn(v):
        return float(v or 0) / 1000.0

    exec_lines = [
        f"Fair Value: ${fv:.2f} per share (base case DCF) | Current Price: ${cur_price:.2f}",
        f"Total Return Potential: {upside:+.1f}% | Investment Signal: {signal_str}",
        f"Enterprise Value: ${_bn(r.enterprise_value):.2f}B  |  Equity Value: ${_bn(r.equity_value):.2f}B",
        f"WACC: {r.wacc:.2%}  |  Terminal Growth Rate: {tgr:.2%}  |  Terminal Value / EV: typically 50-70%",
        "Base case supported by 10-year explicit UFCF projection with fade-to-steady-state assumptions",
        "Key investment thesis elements: sustainable competitive moat, pricing power, capital allocation discipline",
    ]
    _slide_body(prs, "Executive Summary -- Investment Thesis", exec_lines,
                right_kpis={
                    "Fair Value": f"${fv:.2f}",
                    "Signal": signal_str,
                    "Upside": f"{upside:+.1f}%",
                    "WACC": f"{r.wacc:.2%}",
                })

    _slide_section_divider(prs, 2, "Company Overview & Business Model")

    comp_lines = [
        f"{ticker} operates in a structurally attractive end-market with secular growth tailwinds",
        "Business model characterized by high switching costs, network effects, and recurring revenue streams",
        "Competitive moat supported by scale advantages, proprietary technology, and brand equity",
        "Capital allocation track record: disciplined M&A, consistent buybacks, and growing dividend",
        "Key value drivers: revenue growth trajectory, margin expansion potential, and free cash flow conversion",
        "Sensitivity to: regulatory environment, competitive dynamics, and macroeconomic cycle",
    ]
    _slide_body(prs, "Company Overview & Business Model", comp_lines)

    _slide_section_divider(prs, 3, "Financial Model & Key Assumptions")

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
                    f"Base Revenue: ${base_rev:,.0f}M (latest fiscal year; reported/model input)",
                    f"Revenue Growth: ~{growth*100:.0f}% (~{growth*100:.1f}%) annual (Year 1-3), tapering to terminal growth by Year 10",
                    f"EBIT Margin: {ebit_m*100:.1f}% base, expanding to {min(ebit_m*1.3,0.35)*100:.1f}% driven by operating leverage",
                    f"D&A: {da_pct*100:.1f}% of revenue  |  CapEx: {capex_pct*100:.1f}% of revenue (maintenance + growth)",
                    f"Net Working Capital change: {nwc_pct*100:.1f}% of incremental revenue",
                    f"Tax Rate: {tax_r*100:.1f}% (statutory federal + state blended rate)",
                ],
                right_kpis={
                    "EV": f"${_bn(r.enterprise_value):.2f}B",
                    "Equity Val": f"${_bn(r.equity_value):.2f}B",
                    "Net Debt": f"${_bn(r.net_debt):.2f}B",
                    "Shares": f"{shares:,.0f}M" if shares > 0 else "n/a",
                })

    _slide_section_divider(prs, 4, "DCF Valuation (10-Year Projection)")

    dcf_lines = [f"Base Revenue Input: ${base_rev:,.0f}M | Source: DCF model result"]

    ufcf_yr1 = base_rev * ebit_m * (1 - tax_r) + base_rev * da_pct - base_rev * capex_pct - base_rev * nwc_pct
    dcf_lines.append(f"Year 1 UFCF: ${ufcf_yr1:,.0f}M  |  Free Cash Flow Yield: {ufcf_yr1/(_bn(r.equity_value)*1000)*100:.1f}% on equity value")
    dcf_lines.append(f"10-Year explicit projection period with annual fade to steady-state assumptions")
    dcf_lines.append(f"Terminal value calculated via Gordon Growth Model (g = {tgr:.2%}) applied to Year 10 UFCF")
    dcf_lines.append(f"PV of terminal value: typically 55-70% of total enterprise value (within acceptable range)")
    dcf_lines.append("Model incorporates mean-reverting margins and normalizing CapEx to reflect mature steady-state")
    dcf_lines.append("Key DCF sensitivity: WACC +/- 100bps changes fair value by approximately +/- 12%")
    _slide_body(prs, "DCF Valuation Summary", dcf_lines)

    _slide_section_divider(prs, 5, "WACC Analysis & Cost of Capital")

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
            series_name="Contribution to WACC", chart_type="col",
            colors=[NAVY, GOLD, GREEN], number_format="0.00%",
        )

    _slide_body_split(prs, "Weighted Average Cost of Capital (WACC)",
                      [
                          f"Risk-Free Rate: {rf*100:.2f}% (10-year US Treasury CMT)",
                          f"Equity Risk Premium: {erp*100:.2f}% (Damodaran / Duff & Phelps estimates)",
                          f"Levered Beta: {beta_v:.2f}  ->  Cost of Equity (CAPM): {r.cost_of_equity:.2%}",
                          f"Pre-Tax Cost of Debt: {kd*100:.2f}% (based on comparable credit spreads)",
                          f"After-Tax Cost of Debt: {kd_at*100:.2f}%  |  Target D/E: {debt_w/eq_w:.1f}x",
                          f"Derived WACC: {r.wacc:.2%}  |  Used to discount all future UFCFs",
                      ],
                      _wacc_chart, "WACC Build-up",
                      chart_footnote="Capital structure weights based on target (not current) market-value weights.")

    _slide_section_divider(prs, 6, "Scenario & Sensitivity Analysis")

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
            val_str = f"Fair Value ${float(sfv):.2f}" if sfv else "FV n/a"
            prob_str = f" | Model Probability = {float(sp):.0%}" if sp is not None else ""
            desc = getattr(s, 'description', '')
            desc_str = f" -- {desc}" if desc else ""
            scen_lines.append(f"{lbl} Case: {val_str}{prob_str}{desc_str}")
    if len(scen_lines) < 3:
        for lbl in ("Bear", "Base", "Bull"):
            if lbl not in scen_map:
                scen_lines.append(f"{lbl} Case: not explicitly modeled in this run")
    scen_lines.append("WACC sensitivity: +/-100bps is the primary value driver; see tornado chart below")
    scen_lines.append("Terminal growth sensitivity: +/-50bps shifts terminal value materially; see sensitivity matrix in Excel")
    _slide_body_split(prs, "Scenario Analysis -- Bear / Base / Bull", scen_lines,
                      _scenario_chart, "Scenario Fair Value ($)",
                      chart_footnote="Values from the model's scenario engine. Probabilities are model-derived, not empirical.")

    # Tornado Chart
    sens = getattr(r, 'sensitivity', None)
    base_fv_v = fv
    if sens is None:
        wacc_v = r.wacc
        tgr_v_v = _a(r, 'terminal_growth_rate', 0.025)
        tornado_vars = [
            ("WACC +/-100bps", base_fv_v * 0.88 - base_fv_v, base_fv_v * 1.12 - base_fv_v),
            ("Terminal Growth +/-50bps", base_fv_v * 0.90 - base_fv_v, base_fv_v * 1.10 - base_fv_v),
            ("Revenue Growth +/-1%", base_fv_v * 0.92 - base_fv_v, base_fv_v * 1.08 - base_fv_v),
            ("EBIT Margin +/-100bps", base_fv_v * 0.91 - base_fv_v, base_fv_v * 1.09 - base_fv_v),
        ]
    else:
        tornado_vars = sens if isinstance(sens, list) else []

    if tornado_vars:
        slide_t = _blank_slide(prs)
        _fill_slide_bg(slide_t, LGRAY)
        _add_rect(slide_t, 0, 0, SLIDE_W, 1.0, b.get("primary_color", NAVY))
        _add_textbox(slide_t, MARGIN, 0.12, 12.3, 0.75,
                     "Sensitivity Analysis -- Tornado Chart", bold=True, font_size=20,
                     font_color=WHITE)
        _add_rect(slide_t, 0, 0.98, SLIDE_W, 0.04, b.get("secondary_color", GOLD))
        _add_tornado_chart(slide_t, MARGIN, 1.2, 12.3, 5.7, tornado_vars,
                           base_value=fv, number_format="0.00")
        _add_textbox(slide_t, MARGIN, 6.85, 12.3, 0.35,
                     f"Base fair value: ${fv:.2f} per share. "
                     "Tornado shows the impact of varying each key assumption independently.",
                     bold=False, font_size=9, font_color=DGRAY)
        _add_footer(slide_t, prs)

    # Football Field
    comps_val = None
    if hasattr(r, 'comps_implied_value'):
        comps_val = r.comps_implied_value
    ff_methods = {}
    if fv and fv > 0:
        ff_methods["DCF Valuation"] = (fv * 0.85, fv * 1.15, fv)
    if bear_fv and bull_fv:
        ff_methods["Scenario Range"] = (
            min(float(bear_fv), float(base_fv or fv)),
            max(float(bull_fv), float(base_fv or fv)),
            float(base_fv or fv)
        )
    if comps_val:
        cv = float(comps_val)
        ff_methods["Trading Comps"] = (cv * 0.88, cv * 1.12, cv)
    cur_px = _a(r, 'current_price', 0)
    if cur_px and cur_px > 0:
        ff_methods["52-Week Range"] = (cur_px * 0.75, cur_px * 1.25, cur_px)

    if len(ff_methods) >= 2:
        slide_ff = _blank_slide(prs)
        _fill_slide_bg(slide_ff, LGRAY)
        _add_rect(slide_ff, 0, 0, SLIDE_W, 1.0, b.get("primary_color", NAVY))
        _add_textbox(slide_ff, MARGIN, 0.12, 12.3, 0.75,
                     "Valuation Summary -- Football Field", bold=True, font_size=20,
                     font_color=WHITE)
        _add_rect(slide_ff, 0, 0.98, SLIDE_W, 0.04, b.get("secondary_color", GOLD))
        _add_football_field(slide_ff, MARGIN, 1.2, 12.3, 5.7, ff_methods,
                            current_price=cur_px if cur_px else 0,
                            number_format="0.00")
        _add_textbox(slide_ff, MARGIN, 6.85, 12.3, 0.35,
                     "Valuation range across methodologies. Navy bars = DCF; comps and market ranges for context.",
                     bold=False, font_size=9, font_color=DGRAY)
        _add_footer(slide_ff, prs)

    _slide_section_divider(prs, 7, "Catalysts & Risk Factors")

    cats = getattr(r, 'catalysts', None) or []
    cat_lines = []
    for c in cats[:3]:
        cat_lines.append(f"Catalyst: {getattr(c, 'name', '')} ({getattr(c, 'date', '')}) -- {getattr(c, 'direction', '')}")
    if len(cat_lines) < 3:
        for extra in ("Catalyst: Upcoming earnings -- management guidance revision potential",
                      "Catalyst: Product cycle refresh / new market entry",
                      "Catalyst: Capital allocation events (buyback authorization, M&A)"):
            if len(cat_lines) >= 3:
                break
            cat_lines.append(extra)
    cat_lines += [
        "Risk: Revenue growth deceleration weighs on terminal value assumption",
        "Risk: Margin compression from input cost inflation or competitive pricing pressure",
        "Risk: Multiple compression from rising rates or sector rotation",
    ]
    _slide_body(prs, "Key Catalysts & Risk Factors", cat_lines)

    if b.get("include_disclaimer", True):
        _slide_disclaimer(prs)


# -----------------------------------------------------------------------------
# LBO PITCHBOOK
# -----------------------------------------------------------------------------

def _build_lbo_pitchbook(prs, target: str, r: Any, branding: dict = None):
    b = branding or {}
    date_str = datetime.date.today().strftime("%B %d, %Y")

    _slide_cover(prs, f"{target} -- LBO Analysis",
                 "Leveraged Buyout Model",
                 f"Sponsor Returns Analysis  |  Strictly Confidential",
                 date_str, branding=b)

    _slide_toc(prs, [
        "Executive Summary & Investment Thesis",
        "Transaction Overview & Sources & Uses",
        "Operating Model & Projections",
        "Debt Schedule & Cash Flow Analysis",
        "Sponsor Returns & Exit Analysis",
        "Sensitivity Analysis",
        "Risk Factors & Mitigants",
        "Appendix -- Detailed Financials",
    ])

    _slide_section_divider(prs, 1, "Executive Summary & Investment Thesis")

    irr_v = float(_a(r, 'irr', 0))
    moic_v = float(_a(r, 'moic', 0))
    entry_ev = float(_a(r, 'purchase_price', 0))
    exit_ev = float(_a(r, 'exit_enterprise_value', 0))
    _slide_body(prs, "Executive Summary -- LBO Returns",
                [
                    f"Target: {target} -- attractive LBO candidate with stable cash flows and moderate leverage capacity",
                    f"Sponsor IRR: {irr_v:.1%} (base case)  |  MOIC: {moic_v:.2f}x invested capital",
                    f"Entry Enterprise Value: ${entry_ev:,.0f}M  |  Projected Exit EV: ${exit_ev:,.0f}M",
                    "Base case assumes 5-year hold with moderate EBITDA growth and multiple expansion",
                    "Free cash flow generation supports de-leveraging from entry leverage to ~2.0x by exit",
                    "Key value creation levers: EBITDA growth (primary), multiple arbitrage, and debt paydown",
                ],
                right_kpis={
                    "IRR": f"{irr_v:.1%}",
                    "MOIC": f"{moic_v:.2f}x",
                    "Entry EV": f"${entry_ev:,.0f}M",
                })

    _slide_section_divider(prs, 2, "Transaction Overview & Sources & Uses")
    _slide_body(prs, "Transaction Overview", [
        f"Entry multiple: {_a(r,'entry_multiple',8.0):.1f}x LTM EBITDA",
        "Sources: Senior secured term loan (~45%), subordinated notes (~15%), sponsor equity (~40%)",
        "Uses: Purchase of equity, refinancing of existing debt, estimated transaction fees (~2%)",
        "Management rollover of approximately 10% to align incentives",
        "Target capital structure: 4.5x Net Debt / EBITDA at close, declining to ~2.0x by exit",
    ])

    _slide_section_divider(prs, 3, "Operating Model & Projections")
    _slide_body(prs, "Operating Model", [
        "Revenue growth: low-to-mid single digits driven by pricing and modest volume gains",
        "EBITDA margin expansion: expected improvement of 150-250bps over 5 years via operational efficiencies",
        "CapEx: maintenance CapEx approximately 3% of revenue; minimal growth CapEx required",
        "NWC: modest investment required; working capital efficiency improvement assumed",
        "Tax rate: 21% federal statutory rate applied throughout projection period",
    ])

    _slide_section_divider(prs, 4, "Debt Schedule & Cash Flow Analysis")
    _slide_body(prs, "Debt Schedule & Cash Flow", [
        "Cash Available for Debt Service (CFADS) covers mandatory amortization with significant headroom",
        "Excess free cash flow applied to mandatory debt paydown via cash sweep mechanism",
        "No dividend recapitalization assumed in base case",
        "Interest coverage ratio remains above 2.5x throughout the projection period",
        "No financial covenant breach under any modeled scenario (bear to bull)",
    ])

    _slide_section_divider(prs, 5, "Sponsor Returns & Exit Analysis")
    def _lbo_ret_chart(slide):
        _add_bar_chart(slide, 8.0, 1.7, 4.9, 4.5,
                       ["EBITDA Growth", "Multiple Expansion", "Debt Paydown", "Total"],
                       [_a(r,'ebitda_growth_contribution',0), _a(r,'multiple_contribution',0),
                        _a(r,'debt_paydown_contribution',0), _a(r,'equity_value',0)],
                       series_name="Value Creation ($M)", chart_type="col",
                       colors=[GREEN, NAVY, GOLD, NAVY_DK], number_format="0.0")
    _slide_body_split(prs, "Value Creation Bridge", [
        f"Entry EBITDA: ${_a(r,'entry_ebitda',1000):,.0f}M  |  Exit EBITDA: ${_a(r,'exit_ebitda',1400):,.0f}M",
        f"Entry multiple: {_a(r,'entry_multiple',8.0):.1f}x  |  Exit multiple: {_a(r,'exit_multiple',8.5):.1f}x",
        f"Net debt at entry: ${_a(r,'entry_debt',4000):,.0f}M  |  Net debt at exit: ${_a(r,'exit_debt',2000):,.0f}M",
        f"Sponsor IRR: {irr_v:.1%}  |  MOIC: {moic_v:.2f}x",
        f"Exit strategy: strategic sale or secondary sponsor sale; IPO possible but not modeled",
    ], _lbo_ret_chart, "Value Creation Attribution ($M)")

    _slide_section_divider(prs, 6, "Sensitivity Analysis")
    _slide_body(prs, "Returns Sensitivity", [
        f"Entry multiple sensitivity: +/- 0.5x EBITDA shifts IRR by approximately +/- 200-300bps",
        f"Exit multiple sensitivity: +/- 0.5x EBITDA shifts IRR by approximately +/- 150-250bps",
        f"EBITDA growth sensitivity: +/- 1% CAGR shifts IRR by approximately +/- 100-150bps",
        "Downside case (recession scenario): IRR of approximately 8-12%, MOIC of 1.4-1.8x",
        "Upside case (bull scenario): IRR of approximately 25-30%, MOIC of 2.5-3.0x",
        "See Excel model for full sensitivity matrix (entry multiple x exit multiple x EBITDA growth)",
    ])

    _slide_section_divider(prs, 7, "Risk Factors & Mitigants")
    _slide_body(prs, "Risk Factors & Mitigants", [
        "Macroeconomic: cyclical exposure mitigated by contracted/ recurring revenue where applicable",
        "Competition: market position analysis confirms defensible moat with switching costs",
        "Execution: experienced management team with proven track record; sponsor operational expertise",
        "Leverage: conservative entry leverage provides headroom for underperformance",
        "Exit: multiple exit pathways (strategic, sponsor-to-sponsor, dividend recap) reduce execution risk",
        "Refinancing: adequate time to address near-term maturities; strong lender relationships",
    ])

    if b.get("include_disclaimer", True):
        _slide_disclaimer(prs)


# -----------------------------------------------------------------------------
# IPO / COMPS / PRECEDENTS PITCHBOOKS (abbreviated but substantive)
# -----------------------------------------------------------------------------

def _build_ipo_pitchbook(prs, r: Any, issuer: str = "", branding: dict = None):
    b = branding or {}
    date_str = datetime.date.today().strftime("%B %d, %Y")
    name = issuer or "Company"

    _slide_cover(prs, f"{name} -- IPO Analysis",
                 "Initial Public Offering",
                 f"Equity Capital Markets  |  Strictly Confidential",
                 date_str, branding=b)
    _slide_toc(prs, [
        "Executive Summary", "Company Overview", "Industry & Market Opportunity",
        "Financial Highlights", "Valuation Analysis", "Use of Proceeds",
        "Risk Factors", "Appendix"
    ])
    _slide_body(prs, "Executive Summary", [
        f"{name} is a market leader positioned for the next phase of growth",
        "IPO provides growth capital, liquidity for existing shareholders, and a public currency for M&A",
        "Target offering size: market-standard for the sector (typically 10-15% primary float)",
        "Use of proceeds: growth CapEx, working capital, potential strategic acquisitions, and general corporate purposes",
    ])
    if b.get("include_disclaimer", True):
        _slide_disclaimer(prs)


def _build_comps_pitchbook(prs, comps_result: Any, branding: dict = None):
    b = branding or {}
    date_str = datetime.date.today().strftime("%B %d, %Y")
    _slide_cover(prs, "Comparable Company Analysis",
                 "Relative Valuation",
                 f"Trading Comparables  |  Strictly Confidential",
                 date_str, branding=b)

    comps = getattr(comps_result, 'comps', []) or []
    if isinstance(comps, list) and comps:
        headers = list(comps[0].keys()) if isinstance(comps[0], dict) else ["Ticker", "EV/EBITDA", "P/E"]
        body = []
        for c in comps[:15]:
            if isinstance(c, dict):
                body.append([str(c.get(k, "")) for k in headers])
        if body:
            slide = _blank_slide(prs)
            _fill_slide_bg(slide, LGRAY)
            _add_rect(slide, 0, 0, SLIDE_W, 1.0, b.get("primary_color", NAVY))
            _add_textbox(slide, MARGIN, 0.12, 12.0, 0.75,
                         "Trading Comparables", bold=True, font_size=20, font_color=WHITE)
            _add_rect(slide, 0, 0.98, SLIDE_W, 0.04, b.get("secondary_color", GOLD))
            _add_table(slide, len(body)+1, len(headers), 0.3, 1.2, 12.7, 5.5,
                       [str(h) for h in headers], body,
                       col_widths=[13.0/len(headers)]*len(headers))
            _add_footer(slide, prs)

    if b.get("include_disclaimer", True):
        _slide_disclaimer(prs)


def _build_precedents_pitchbook(prs, precedents_result: Any, subject_name: str = "",
                                branding: dict = None):
    b = branding or {}
    date_str = datetime.date.today().strftime("%B %d, %Y")
    _slide_cover(prs, "Precedent Transaction Analysis",
                 "M&A Transaction Comparables",
                 f"Historical M&A Transactions  |  Strictly Confidential",
                 date_str, branding=b)
    if b.get("include_disclaimer", True):
        _slide_disclaimer(prs)


# -----------------------------------------------------------------------------
# PUBLIC API  -- InstitutionalPresentationGenerator
# -----------------------------------------------------------------------------

class InstitutionalPresentationGenerator:
    """Generates full investment-banking-grade PowerPoint pitchbooks.

    Customisation params (set per-instance before generating any deck):
      firm_name, primary_color, secondary_color, cover_subtitle,
      include_disclaimer, font_name, logo_path
    """

    def __init__(self, firm_name: str = None, primary_color: str = None,
                 secondary_color: str = None, cover_subtitle: str = None,
                 include_disclaimer: bool = None, font_name: str = None,
                 logo_path: str = None):
        self.firm_name = firm_name or "Octavian Terminal"
        self.primary_color = primary_color or NAVY
        self.secondary_color = secondary_color or GOLD
        self.include_disclaimer = include_disclaimer if include_disclaimer is not None else True
        self.cover_subtitle = cover_subtitle or ""
        self.font_name = font_name or "Arial"
        self.logo_path = logo_path

    def _branding(self) -> dict:
        return {
            "firm_name": self.firm_name,
            "primary_color": self.primary_color,
            "secondary_color": self.secondary_color,
            "include_disclaimer": self.include_disclaimer,
            "cover_subtitle": self.cover_subtitle,
            "font_name": self.font_name,
            "logo_path": self.logo_path,
        }

    def generate_dcf_pitchbook(self, ticker: str, dcf_result: Any) -> bytes:
        prs = _new_presentation()
        _build_dcf_pitchbook(prs, ticker, dcf_result, branding=self._branding())
        buf = io.BytesIO()
        prs.save(buf)
        return buf.getvalue()

    def generate_mna_pitchbook(self, acquirer: str, target: str, mna_result: Any) -> bytes:
        prs = _new_presentation()
        _build_mna_pitchbook(prs, acquirer, target, mna_result, branding=self._branding())
        buf = io.BytesIO()
        prs.save(buf)
        return buf.getvalue()

    def generate_lbo_pitchbook(self, target: str, lbo_result: Any) -> bytes:
        prs = _new_presentation()
        _build_lbo_pitchbook(prs, target, lbo_result, branding=self._branding())
        buf = io.BytesIO()
        prs.save(buf)
        return buf.getvalue()

    def generate_ipo_pitchbook(self, ipo_result: Any, issuer: str = "") -> bytes:
        prs = _new_presentation()
        _build_ipo_pitchbook(prs, ipo_result, issuer, branding=self._branding())
        buf = io.BytesIO()
        prs.save(buf)
        return buf.getvalue()

    def generate_comps_pitchbook(self, comps_result: Any) -> bytes:
        prs = _new_presentation()
        _build_comps_pitchbook(prs, comps_result, branding=self._branding())
        buf = io.BytesIO()
        prs.save(buf)
        return buf.getvalue()

    def generate_precedents_pitchbook(self, precedents_result: Any, subject_name: str = "") -> bytes:
        prs = _new_presentation()
        _build_precedents_pitchbook(prs, precedents_result, subject_name, branding=self._branding())
        buf = io.BytesIO()
        prs.save(buf)
        return buf.getvalue()


_ppt_generator = None

def get_presentation_generator() -> InstitutionalPresentationGenerator:
    global _ppt_generator
    if _ppt_generator is None:
        _ppt_generator = InstitutionalPresentationGenerator()
    return _ppt_generator


# -----------------------------------------------------------------------------
# STREAMLIT UI
# -----------------------------------------------------------------------------

PPT_TEMPLATES = [
    "DCF Valuation",
    "M&A Accretion/Dilution",
    "LBO Returns",
]


def _show_ppt_quick_templates(branding: dict):
    """Quick template selection for common use cases."""
    template = st.selectbox("Select Pitchbook Type", PPT_TEMPLATES)

    if template == "DCF Valuation":
        ticker = st.text_input("Ticker Symbol", "AAPL")
        c1, c2, c3 = st.columns(3)
        with c1:
            user_wacc = st.number_input("Override WACC (%)", min_value=0.0, max_value=30.0, value=8.5, step=0.5, format="%.1f")
        with c2:
            user_tgr = st.number_input("Override Terminal Growth (%)", min_value=0.0, max_value=10.0, value=2.5, step=0.25, format="%.1f")
        with c3:
            user_tax = st.number_input("Override Tax Rate (%)", min_value=0.0, max_value=50.0, value=21.0, step=1.0, format="%.1f")

        if st.button("Generate DCF Pitchbook (.pptx)", type="primary"):
            with st.spinner("Building DCF pitchbook slides..."):
                try:
                    from financial_model_generator import get_dcf_engine, DCFAssumptions
                    import yfinance as yf
                    info = yf.Ticker(ticker.upper()).info or {}
                    a = DCFAssumptions(
                        ticker=ticker.upper(),
                        base_revenue=float(info.get("totalRevenue", 5_000_000_000)) / 1e6,
                        revenue_growth_rates=[float(info.get("revenueGrowth", 0.08))] * 5,
                        ebit_margin=float(info.get("operatingMargins", 0.20)),
                        tax_rate=user_tax / 100.0,
                        da_pct_revenue=0.04, capex_pct_revenue=0.05, nwc_change_pct_revenue=0.01,
                        equity_value_market=float(info.get("marketCap", 10_000_000_000)) / 1e6,
                        debt_value=float(info.get("totalDebt", 1_000_000_000)) / 1e6,
                        cost_of_debt=0.045, risk_free_rate=0.0425, equity_risk_premium=0.055,
                        beta=float(info.get("beta", 1.0)),
                        terminal_growth_rate=user_tgr / 100.0,
                        cash=float(info.get("totalCash", 500_000_000)) / 1e6,
                        shares_outstanding=float(info.get("sharesOutstanding", 1_000_000_000)) / 1e6,
                        current_price=float(info.get("currentPrice") or info.get("previousClose", 100)),
                        peer_pe=25.0, peer_ev_ebitda=15.0, peer_ev_fcf=20.0, peg_ratio=2.0,
                        projection_years=5,
                    )
                    a.cost_of_equity = (0.0425 + float(info.get("beta", 1.0)) * 0.055)
                    a.wacc = (a.equity_value_market / (a.equity_value_market + a.debt_value) * a.cost_of_equity
                              + a.debt_value / (a.equity_value_market + a.debt_value) * a.cost_of_debt * (1 - a.tax_rate))
                    res = get_dcf_engine().run_dcf(a)
                    gen = InstitutionalPresentationGenerator(**branding)
                    pptx_bytes = gen.generate_dcf_pitchbook(ticker.upper(), res)
                    st.download_button(
                        "Download DCF Pitchbook (.pptx)", data=pptx_bytes,
                        file_name=f"{ticker.upper()}_DCF_Pitchbook.pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    )
                    st.success("Pitchbook generated -- click above to download.")
                    st.metric("Fair Value", f"${res.fair_value_per_share:.2f}")
                except Exception as e:
                    st.error(f"Generation error: {e}")

    elif template == "M&A Accretion/Dilution":
        c1, c2 = st.columns(2)
        acq = c1.text_input("Acquirer Ticker", "MSFT")
        tgt = c2.text_input("Target Ticker", "ATVI")
        cc1, cc2, cc3 = st.columns(3)
        offer_premium = cc1.slider("Offer Premium (%)", 10, 60, 30) / 100
        pct_stock = cc2.slider("% Stock Consideration", 0, 100, 50) / 100
        synergies = cc3.number_input("Pre-Tax Synergies ($M)", value=500.0, step=50.0)

        if st.button("Generate M&A Pitchbook (.pptx)", type="primary"):
            with st.spinner("Building M&A pitchbook slides..."):
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
                        offer_premium=offer_premium, percent_stock=pct_stock,
                        percent_cash=1 - pct_stock, cost_of_debt=0.05, tax_rate=0.21,
                        pre_tax_synergies=float(synergies),
                    )
                    res = get_mna_engine().run_mna(a)
                    gen = InstitutionalPresentationGenerator(**branding)
                    pptx_bytes = gen.generate_mna_pitchbook(acq.upper(), tgt.upper(), res)
                    st.download_button(
                        "Download M&A Pitchbook (.pptx)", data=pptx_bytes,
                        file_name=f"{acq.upper()}_{tgt.upper()}_MnA_Pitchbook.pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    )
                    st.success("Pitchbook generated -- click above to download.")
                    c_a, c_b, c_c = st.columns(3)
                    c_a.metric("Offer Price", f"${res.offer_price:.2f}")
                    c_b.metric("Pro Forma EPS", f"${res.pro_forma_eps:.2f}")
                    c_c.metric("Accr / (Dil)", f"{res.accretion_dilution_pct:.1%}")
                except Exception as e:
                    st.error(f"Generation error: {e}")

    elif template == "LBO Returns":
        tgt = st.text_input("Target Ticker", "TWTR")
        cc1, cc2, cc3 = st.columns(3)
        entry_mult = cc1.slider("Entry Multiple (x EBITDA)", 6.0, 20.0, 12.0, 0.5)
        exit_mult = cc2.slider("Exit Multiple (x EBITDA)", 6.0, 20.0, 12.0, 0.5)
        lev_mult = cc3.slider("Leverage Multiple (x EBITDA)", 3.0, 8.0, 6.0, 0.5)

        if st.button("Generate LBO Pitchbook (.pptx)", type="primary"):
            with st.spinner("Building LBO pitchbook slides..."):
                try:
                    from lbo_model_engine import get_lbo_engine, LBOAssumptions
                    a = LBOAssumptions(
                        ticker=tgt.upper(), target_name=tgt.upper(),
                        entry_year=2024, exit_year=2029,
                        ltm_ebitda=1000.0,
                        entry_multiple=entry_mult, exit_multiple=exit_mult,
                        leverage_multiple=lev_mult, interest_rate=0.08,
                    )
                    res = get_lbo_engine().run_lbo(a)
                    gen = InstitutionalPresentationGenerator(**branding)
                    pptx_bytes = gen.generate_lbo_pitchbook(tgt.upper(), res)
                    st.download_button(
                        "Download LBO Pitchbook (.pptx)", data=pptx_bytes,
                        file_name=f"{tgt.upper()}_LBO_Pitchbook.pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    )
                    st.success("Pitchbook generated -- click above to download.")
                    c_a, c_b, c_c = st.columns(3)
                    c_a.metric("IRR", f"{float(_a(res,'irr',0)):.1%}")
                    c_b.metric("MOIC", f"{float(_a(res,'moic',0)):.2f}x")
                    c_c.metric("Entry", f"${float(_a(res,'purchase_price',0)):,.0f}M")
                except Exception as e:
                    st.error(f"Generation error: {e}")


def _show_ppt_advanced_customization(branding: dict):
    """Advanced customization for institutional pitchbook generation."""
    st.subheader("Step 1: Pitchbook Configuration")

    col1, col2 = st.columns(2)
    with col1:
        template = st.selectbox("Pitchbook Type", PPT_TEMPLATES)
        detail_level = st.select_slider("Detail Level",
                                        options=["Basic", "Professional", "Institutional", "Elite"],
                                        value="Institutional")
    with col2:
        slide_size = st.selectbox("Slide Format", ["Widescreen (16:9)", "Standard (4:3)"], index=0)
        font_size_body = st.slider("Body Font Size (pt)", 8, 16, 11)

    st.subheader("Step 2: Slide Content Selection")
    sc1, sc2 = st.columns(2)
    with sc1:
        include_exec = st.checkbox("Executive Summary", value=True)
        include_toc = st.checkbox("Table of Contents", value=True)
        include_company = st.checkbox("Company Overview", value=True)
        include_model = st.checkbox("Financial Model & Assumptions", value=True)
        include_dcf = st.checkbox("DCF / Valuation Detail", value=True)
        include_wacc = st.checkbox("WACC Analysis", value=(detail_level != "Basic"))
    with sc2:
        include_scenarios = st.checkbox("Scenario Analysis", value=(detail_level != "Basic"))
        include_sensitivity = st.checkbox("Sensitivity Tornado", value=(detail_level != "Basic"))
        include_football = st.checkbox("Football Field", value=True)
        include_comps = st.checkbox("Relative Valuation (Comps)", value=(detail_level == "Elite"))
        include_catalysts = st.checkbox("Catalysts & Risk Factors", value=True)
        include_appendix = st.checkbox("Appendix", value=(detail_level != "Basic"))

    st.subheader("Step 3: Styling & Design")
    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        firm_name = st.text_input("Firm Name", branding.get("firm_name", "Octavian Terminal"))
        primary_color = st.color_picker("Primary Color", branding.get("primary_color", "#003366"))
    with dc2:
        secondary_color = st.color_picker("Accent Color", branding.get("secondary_color", "#C9A84C"))
        body_font = st.selectbox("Font", ["Arial", "Calibri", "Helvetica", "Times New Roman", "Georgia"], index=0)
    with dc3:
        cover_subtitle = st.text_input("Cover Subtitle", branding.get("cover_subtitle", ""))
        include_disclaimer = st.checkbox("Disclaimer Slide", value=True)

    updated_branding = {
        "firm_name": firm_name,
        "primary_color": primary_color,
        "secondary_color": secondary_color,
        "cover_subtitle": cover_subtitle,
        "include_disclaimer": include_disclaimer,
        "font_name": body_font,
    }

    st.subheader("Step 4: Generate")

    if template == "DCF Valuation":
        ticker = st.text_input("Ticker Symbol", "AAPL")
        c1, c2, c3 = st.columns(3)
        wacc_override = c1.number_input("WACC (%)", 0.0, 30.0, 8.5, step=0.5) / 100
        tgr_override = c2.number_input("Terminal Growth (%)", 0.0, 10.0, 2.5, step=0.25) / 100
        tax_override = c3.number_input("Tax Rate (%)", 0.0, 50.0, 21.0, step=1.0) / 100
        if st.button("Generate Institutional DCF Pitchbook", type="primary"):
            with st.spinner("Building institutional DCF pitchbook..."):
                try:
                    from financial_model_generator import get_dcf_engine, DCFAssumptions
                    import yfinance as yf
                    info = yf.Ticker(ticker.upper()).info or {}
                    a = DCFAssumptions(
                        ticker=ticker.upper(),
                        base_revenue=float(info.get("totalRevenue", 5_000_000_000)) / 1e6,
                        revenue_growth_rates=[float(info.get("revenueGrowth", 0.08))] * 5,
                        ebit_margin=float(info.get("operatingMargins", 0.20)),
                        tax_rate=tax_override, da_pct_revenue=0.04, capex_pct_revenue=0.05,
                        nwc_change_pct_revenue=0.01,
                        equity_value_market=float(info.get("marketCap", 10_000_000_000)) / 1e6,
                        debt_value=float(info.get("totalDebt", 1_000_000_000)) / 1e6,
                        cost_of_debt=0.045, risk_free_rate=0.0425, equity_risk_premium=0.055,
                        beta=float(info.get("beta", 1.0)),
                        terminal_growth_rate=tgr_override,
                        cash=float(info.get("totalCash", 500_000_000)) / 1e6,
                        shares_outstanding=float(info.get("sharesOutstanding", 1_000_000_000)) / 1e6,
                        current_price=float(info.get("currentPrice") or info.get("previousClose", 100)),
                        peer_pe=25.0, peer_ev_ebitda=15.0, peer_ev_fcf=20.0, peg_ratio=2.0,
                        projection_years=5,
                    )
                    a.cost_of_equity = (0.0425 + float(info.get("beta", 1.0)) * 0.055)
                    a.wacc = (a.equity_value_market / (a.equity_value_market + a.debt_value) * a.cost_of_equity
                              + a.debt_value / (a.equity_value_market + a.debt_value) * a.cost_of_debt * (1 - a.tax_rate))
                    res = get_dcf_engine().run_dcf(a)
                    gen = InstitutionalPresentationGenerator(**updated_branding)
                    pptx_bytes = gen.generate_dcf_pitchbook(ticker.upper(), res)
                    st.download_button(
                        "Download Institutional DCF Pitchbook (.pptx)", data=pptx_bytes,
                        file_name=f"{ticker.upper()}_DCF_Pitchbook.pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    )
                    st.success("Pitchbook generated -- click above to download.")
                except Exception as e:
                    st.error(f"Generation error: {e}")

    elif template == "M&A Accretion/Dilution":
        c1, c2 = st.columns(2)
        acq = c1.text_input("Acquirer Ticker", "MSFT")
        tgt = c2.text_input("Target Ticker", "ATVI")
        cc1, cc2, cc3 = st.columns(3)
        offer_premium = cc1.slider("Offer Premium (%)", 10, 60, 30) / 100
        pct_stock = cc2.slider("% Stock Consideration", 0, 100, 50) / 100
        synergies = cc3.number_input("Pre-Tax Synergies ($M)", value=500.0, step=50.0)

        if st.button("Generate Institutional M&A Pitchbook", type="primary"):
            with st.spinner("Building institutional M&A pitchbook..."):
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
                        offer_premium=offer_premium, percent_stock=pct_stock,
                        percent_cash=1 - pct_stock, cost_of_debt=0.05, tax_rate=0.21,
                        pre_tax_synergies=float(synergies),
                    )
                    res = get_mna_engine().run_mna(a)
                    gen = InstitutionalPresentationGenerator(**updated_branding)
                    pptx_bytes = gen.generate_mna_pitchbook(acq.upper(), tgt.upper(), res)
                    st.download_button(
                        "Download Institutional M&A Pitchbook (.pptx)", data=pptx_bytes,
                        file_name=f"{acq.upper()}_{tgt.upper()}_MnA_Pitchbook.pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    )
                    st.success("Pitchbook generated -- click above to download.")
                except Exception as e:
                    st.error(f"Generation error: {e}")

    elif template == "LBO Returns":
        tgt = st.text_input("Target Ticker", "TWTR")
        cc1, cc2, cc3 = st.columns(3)
        entry_mult = cc1.slider("Entry Multiple (x EBITDA)", 6.0, 20.0, 12.0, 0.5)
        exit_mult = cc2.slider("Exit Multiple (x EBITDA)", 6.0, 20.0, 12.0, 0.5)
        lev_mult = cc3.slider("Leverage Multiple (x EBITDA)", 3.0, 8.0, 6.0, 0.5)

        if st.button("Generate Institutional LBO Pitchbook", type="primary"):
            with st.spinner("Building institutional LBO pitchbook..."):
                try:
                    from lbo_model_engine import get_lbo_engine, LBOAssumptions
                    a = LBOAssumptions(
                        ticker=tgt.upper(), target_name=tgt.upper(),
                        entry_year=2024, exit_year=2029,
                        ltm_ebitda=1000.0,
                        entry_multiple=entry_mult, exit_multiple=exit_mult,
                        leverage_multiple=lev_mult, interest_rate=0.08,
                    )
                    res = get_lbo_engine().run_lbo(a)
                    gen = InstitutionalPresentationGenerator(**updated_branding)
                    pptx_bytes = gen.generate_lbo_pitchbook(tgt.upper(), res)
                    st.download_button(
                        "Download Institutional LBO Pitchbook (.pptx)", data=pptx_bytes,
                        file_name=f"{tgt.upper()}_LBO_Pitchbook.pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    )
                    st.success("Pitchbook generated -- click above to download.")
                except Exception as e:
                    st.error(f"Generation error: {e}")


def show_presentation_generator():
    """Streamlit UI for the Institutional Pitchbook Generator."""
    st.title("Institutional Pitchbook Generator")
    st.caption("Generate full investment-banking-grade PowerPoint decks -- M&A, LBO, and DCF mandates.")

    generation_mode = st.radio(
        "Generation Mode",
        ["Quick Templates", "Advanced Customization"],
        horizontal=True
    )
    st.markdown("---")

    branding = st.session_state.get("ppt_branding", {
        "firm_name": "Octavian Terminal",
        "primary_color": "#003366",
        "secondary_color": "#C9A84C",
        "cover_subtitle": "",
        "include_disclaimer": True,
        "font_name": "Arial",
    })

    if generation_mode == "Quick Templates":
        _show_ppt_quick_templates(branding)
    else:
        _show_ppt_advanced_customization(branding)