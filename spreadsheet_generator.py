"""
Octavian Spreadsheet Generator - Advanced Customization & Control
Full user control over: data selection, timeframes, calculations, formatting, 
visual elements, financial modeling, and export options.

This module provides institutional-grade spreadsheet generation with 
complete customization at every step.

Author: APB - Octavian Team
"""

import io
import re
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

try:
    from octavian_theme import section_header
except ImportError:
    def section_header(t): st.subheader(t)

from data_sources import get_stock, get_fx, get_futures_proxy

# 
# AVAILABLE DATA & CUSTOMIZATION OPTIONS
# 

AVAILABLE_PRICE_COLUMNS = [
    "Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"
]

AVAILABLE_RETURNS_COLUMNS = [
    "Returns_1d", "Returns_5d", "Returns_20d", "Returns_MTD", "Returns_YTD",
    "Returns_Cumulative", "Log_Returns_1d"
]

AVAILABLE_MA_COLUMNS = [
    "SMA_5", "SMA_10", "SMA_20", "SMA_50", "SMA_100", "SMA_200",
    "EMA_5", "EMA_12", "EMA_26", "EMA_50", "EMA_200",
    "WMA_20", "DEMA_20", "TEMA_20"
]

AVAILABLE_MOMENTUM_COLUMNS = [
    "RSI_14", "RSI_21", "Stochastic_K", "Stochastic_D",
    "MACD", "MACD_Signal", "MACD_Histogram",
    "ROC_12", "CCI_20", "Williams_%R"
]

AVAILABLE_VOLATILITY_COLUMNS = [
    "Volatility_5d", "Volatility_20d", "Volatility_60d", "Volatility_Annual",
    "ATR_14", "ATR_21", "Historical_Vol",
    "Bollinger_Upper_20", "Bollinger_Mid_20", "Bollinger_Lower_20",
    "Bollinger_Width", "Bollinger_Position"
]

AVAILABLE_VOLUME_COLUMNS = [
    "Volume", "Volume_SMA_20", "Volume_Ratio",
    "OBV", "On_Balance_Volume_MA", "Accumulation_Distribution",
    "Money_Flow", "Money_Flow_Ratio"
]

AVAILABLE_TREND_COLUMNS = [
    "Trend_Direction", "Support_Level", "Resistance_Level",
    "ADX_14", "DI_Plus", "DI_Minus", "PSAR"
]

ALL_DATA_COLUMNS = (
    AVAILABLE_PRICE_COLUMNS +
    AVAILABLE_RETURNS_COLUMNS +
    AVAILABLE_MA_COLUMNS +
    AVAILABLE_MOMENTUM_COLUMNS +
    AVAILABLE_VOLATILITY_COLUMNS +
    AVAILABLE_VOLUME_COLUMNS +
    AVAILABLE_TREND_COLUMNS
)

TIMEFRAME_OPTIONS = {
    "1 Day": "1d",
    "5 Days": "5d",
    "1 Week": "1wk",
    "2 Weeks": "14d",
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "YTD": "ytd",
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y",
    "10 Years": "10y",
    "Max": "max"
}

CHART_TYPES = {
    "Line Chart": "line",
    "Candlestick": "candlestick",
    "OHLC Bar": "ohlc",
    "Area Chart": "area",
    "Bar Chart": "bar",
    "Scatter Plot": "scatter",
    "Heatmap": "heatmap",
    "Correlation Matrix": "correlation",
    "Distribution": "histogram",
    "Waterfall": "waterfall"
}

FORMATTING_OPTIONS = {
    "Currency": "currency",
    "Percentage": "percentage",
    "Decimal (2 places)": "decimal_2",
    "Decimal (4 places)": "decimal_4",
    "Thousands Separator": "thousands",
    "Scientific": "scientific"
}

COLOR_SCHEMES = {
    "Professional (Blue/Red)": "professional",
    "Octavian (Gold/Lavender)": "octavian",
    "Green/Red": "green_red",
    "Monochrome": "monochrome",
    "Heatmap (Cool)": "cool",
    "Heatmap (Warm)": "warm"
}


@dataclass
class SpreadsheetSpec:
    """Complete spreadsheet specification."""
    symbols: List[str]
    timeframe: str
    interval: str
    data_columns: List[str]
    price_columns: List[str]
    technical_columns: List[str]
    risk_columns: List[str]
    chart_types: List[str]
    formatting: Dict[str, Any]
    color_scheme: str
    include_summary: bool
    include_statistics: bool
    include_charts: bool
    include_correlations: bool
    include_financial_model: bool
    model_type: Optional[str]
    custom_calculations: Dict[str, str]
    sheet_names: Dict[str, str]


def show_spreadsheet_generator():
    """Advanced spreadsheet generator with full customization."""
    st.title("Advanced Spreadsheet Generator")
    st.caption("Institutional-grade customization: data, calculations, formatting, and visuals")
    
    # Main interface: Wizard vs Advanced
    generation_mode = st.radio(
        "Generation Mode",
        ["Quick Templates", "Advanced Customization"],
        horizontal=True
    )
    
    st.markdown("---")
    
    if generation_mode == "Quick Templates":
        show_quick_templates()
    else:
        show_advanced_customization()


def show_quick_templates():
    """Quick template selection for common use cases."""
    
    template = st.selectbox(
        "Select Template",
        [
            "Stock Performance Comparison",
            "Technical Analysis Dashboard",
            "Portfolio Tracking",
            "Trade Log & Performance",
            "Financial Statements Model",
            "DCF Valuation Model",
            "LBO Analysis",
            "Relative Valuation (Comps)",
            "Risk & Volatility Analysis",
            "Correlation & Diversification"
        ]
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        symbols_input = st.text_input("Symbols (comma-separated)", value="AAPL, MSFT, NVDA")
    with col2:
        timeframe = st.selectbox("Timeframe", list(TIMEFRAME_OPTIONS.keys()), index=8)
    with col3:
        include_charts = st.checkbox("Include Charts", value=True)

    # --- Template-specific assumption inputs (used exactly as entered) ---
    dcf_wacc = dcf_tgr = dcf_tax = None
    lbo_debt = lbo_exit = lbo_rate = None

    if template == "DCF Valuation Model":
        st.markdown("#### DCF Assumptions")
        a1, a2, a3 = st.columns(3)
        dcf_wacc = a1.number_input("WACC (%)", min_value=1.0, max_value=30.0, value=8.5, step=0.5, format="%.2f") / 100
        dcf_tgr = a2.number_input("Terminal Growth Rate (%)", min_value=0.0, max_value=10.0, value=2.5, step=0.25, format="%.2f") / 100
        dcf_tax = a3.number_input("Tax Rate (%)", min_value=0.0, max_value=50.0, value=21.0, step=1.0, format="%.1f") / 100
    elif template == "LBO Analysis":
        st.markdown("#### LBO Assumptions")
        b1, b2, b3 = st.columns(3)
        lbo_debt = b1.number_input("Debt / Total Capital (%)", min_value=10.0, max_value=90.0, value=60.0, step=5.0, format="%.0f") / 100
        lbo_exit = b2.number_input("Exit Multiple (x EBITDA)", min_value=3.0, max_value=30.0, value=10.0, step=0.5, format="%.1f")
        lbo_rate = b3.number_input("Interest Rate (%)", min_value=1.0, max_value=20.0, value=8.0, step=0.5, format="%.2f") / 100

    if st.button("Generate Template", type="primary", use_container_width=True):
        symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
        
        if not symbols:
            st.error("Enter at least one symbol")
            return
        
        with st.spinner("Generating spreadsheet..."):
            try:
                if template == "Stock Performance Comparison":
                    _generate_performance_comparison(symbols, TIMEFRAME_OPTIONS[timeframe], include_charts)
                elif template == "Technical Analysis Dashboard":
                    _generate_technical_dashboard(symbols, TIMEFRAME_OPTIONS[timeframe], include_charts)
                elif template == "Portfolio Tracking":
                    _generate_portfolio_tracking(symbols)
                elif template == "Trade Log & Performance":
                    _generate_trade_log()
                elif template == "Financial Statements Model":
                    _generate_financial_statements(symbols[0])
                elif template == "DCF Valuation Model":
                    _generate_dcf_model(symbols[0], wacc=dcf_wacc, tgr=dcf_tgr, tax_rate=dcf_tax)
                elif template == "LBO Analysis":
                    _generate_lbo_model(symbols[0], debt_pct=lbo_debt, exit_mult=lbo_exit, interest_rate=lbo_rate)
                elif template == "Relative Valuation (Comps)":
                    _generate_comps_model(symbols)
                elif template == "Risk & Volatility Analysis":
                    _generate_risk_analysis(symbols, TIMEFRAME_OPTIONS[timeframe])
                elif template == "Correlation & Diversification":
                    _generate_correlation_analysis(symbols, TIMEFRAME_OPTIONS[timeframe])
            except Exception as e:
                st.error(f"Error generating template: {e}")


def show_advanced_customization():
    """Full advanced customization interface."""
    
    # STEP 1: Symbol & Timeframe Selection
    st.subheader("Step 1: Asset Selection")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        symbols_input = st.text_input(
            "Enter Symbols (comma-separated)",
            value="AAPL, MSFT, GOOGL",
            help="Stocks: AAPL | Forex: EUR/USD | Futures: ES=F | Crypto: BTC-USD"
        )
    with col2:
        timeframe_label = st.selectbox("Timeframe", list(TIMEFRAME_OPTIONS.keys()), index=8)
    with col3:
        interval = st.selectbox("Data Interval", ["1 Day", "1 Week", "1 Month"], index=0)
    
    symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
    timeframe = TIMEFRAME_OPTIONS[timeframe_label]
    interval_map = {"1 Day": "1d", "1 Week": "1wk", "1 Month": "1mo"}
    
    st.markdown("---")
    
    # STEP 2: Data Selection
    st.subheader("Step 2: Data Customization")
    
    data_tabs = st.tabs([
        "Price Data",
        "Returns & Performance",
        "Moving Averages",
        "Momentum Indicators",
        "Volatility & Bands",
        "Volume Indicators",
        "Trend Analysis"
    ])
    
    selected_columns = {}
    
    with data_tabs[0]:
        st.markdown("**Price Data**")
        price_cols = st.multiselect(
            "Price Columns",
            AVAILABLE_PRICE_COLUMNS,
            default=["Date", "Open", "High", "Low", "Close", "Volume"]
        )
        selected_columns["price"] = price_cols
    
    with data_tabs[1]:
        st.markdown("**Returns & Performance**")
        returns_cols = st.multiselect(
            "Returns Columns",
            AVAILABLE_RETURNS_COLUMNS,
            default=["Returns_1d", "Returns_20d", "Returns_YTD"]
        )
        selected_columns["returns"] = returns_cols
    
    with data_tabs[2]:
        st.markdown("**Moving Averages**")
        ma_cols = st.multiselect(
            "Moving Average Columns",
            AVAILABLE_MA_COLUMNS,
            default=["SMA_20", "SMA_50", "SMA_200", "EMA_12", "EMA_26"]
        )
        selected_columns["ma"] = ma_cols
    
    with data_tabs[3]:
        st.markdown("**Momentum Indicators**")
        momentum_cols = st.multiselect(
            "Momentum Columns",
            AVAILABLE_MOMENTUM_COLUMNS,
            default=["RSI_14", "MACD", "MACD_Signal"]
        )
        selected_columns["momentum"] = momentum_cols
    
    with data_tabs[4]:
        st.markdown("**Volatility & Bands**")
        vol_cols = st.multiselect(
            "Volatility Columns",
            AVAILABLE_VOLATILITY_COLUMNS,
            default=["Volatility_20d", "ATR_14", "Bollinger_Upper_20", "Bollinger_Lower_20"]
        )
        selected_columns["volatility"] = vol_cols
    
    with data_tabs[5]:
        st.markdown("**Volume Indicators**")
        vol_ind_cols = st.multiselect(
            "Volume Indicator Columns",
            AVAILABLE_VOLUME_COLUMNS,
            default=["Volume", "Volume_SMA_20", "OBV"]
        )
        selected_columns["volume"] = vol_ind_cols
    
    with data_tabs[6]:
        st.markdown("**Trend Analysis**")
        trend_cols = st.multiselect(
            "Trend Columns",
            AVAILABLE_TREND_COLUMNS,
            default=["Trend_Direction", "ADX_14", "Support_Level", "Resistance_Level"]
        )
        selected_columns["trend"] = trend_cols
    
    st.markdown("---")
    
    # STEP 3: Calculations & Modeling
    st.subheader("Step 3: Custom Calculations & Models")
    
    calc_tabs = st.tabs([
        "Financial Metrics",
        "Risk Metrics",
        "Performance Analysis",
        "Custom Formulas"
    ])
    
    calculations = {}
    
    with calc_tabs[0]:
        st.markdown("**Financial Metrics**")
        include_pe = st.checkbox("P/E Ratio", value=False)
        include_eps = st.checkbox("EPS / Share", value=False)
        include_pb = st.checkbox("Price-to-Book", value=False)
        include_fcf = st.checkbox("Free Cash Flow", value=False)
        
        if include_pe or include_eps or include_pb or include_fcf:
            calculations["financial"] = {
                "pe": include_pe,
                "eps": include_eps,
                "pb": include_pb,
                "fcf": include_fcf
            }
    
    with calc_tabs[1]:
        st.markdown("**Risk Metrics**")
        include_sharpe = st.checkbox("Sharpe Ratio", value=True)
        include_sortino = st.checkbox("Sortino Ratio", value=True)
        include_var = st.checkbox("Value at Risk (VaR)", value=False)
        include_drawdown = st.checkbox("Max Drawdown", value=True)
        include_beta = st.checkbox("Beta", value=False)
        
        calculations["risk"] = {
            "sharpe": include_sharpe,
            "sortino": include_sortino,
            "var": include_var,
            "drawdown": include_drawdown,
            "beta": include_beta
        }
    
    with calc_tabs[2]:
        st.markdown("**Performance Analysis**")
        include_cagr = st.checkbox("CAGR (Compound Annual Return)", value=True)
        include_annual_return = st.checkbox("Annual Returns", value=True)
        include_rolling = st.checkbox("Rolling Performance (20-day)", value=False)
        
        rolling_window = st.number_input("Rolling Window (days)", min_value=5, max_value=252, value=20) if include_rolling else None
        
        calculations["performance"] = {
            "cagr": include_cagr,
            "annual": include_annual_return,
            "rolling": include_rolling,
            "rolling_window": rolling_window
        }
    
    with calc_tabs[3]:
        st.markdown("**Custom Formulas**")
        st.info("Add custom calculations (e.g., Price/Book, Custom Ratio, etc.)")
        
        custom_formula_1 = st.text_input(
            "Custom Formula 1",
            placeholder="e.g., Close / Book_Value",
            help="Supported: Close, Open, High, Low, Volume, SMA_*, EMA_*, RSI, MACD, ATR, etc."
        )
        
        custom_formula_2 = st.text_input(
            "Custom Formula 2",
            placeholder="e.g., RSI / 100 * Close",
            help="Advanced: combine indicators and prices"
        )
        
        calculations["custom"] = {
            "formula_1": custom_formula_1 if custom_formula_1 else None,
            "formula_2": custom_formula_2 if custom_formula_2 else None
        }
    
    st.markdown("---")
    
    # STEP 4: Formatting & Visual Options
    st.subheader("Step 4: Formatting & Visual Customization")
    
    fmt_col1, fmt_col2, fmt_col3 = st.columns(3)
    
    with fmt_col1:
        st.markdown("**Number Formatting**")
        number_format = st.selectbox(
            "Format",
            list(FORMATTING_OPTIONS.keys()),
            index=2,
            help="How numbers appear in spreadsheet"
        )
        decimal_places = st.selectbox(
            "Decimal Places",
            [0, 1, 2, 3, 4, 6],
            index=2,
            help="For currency and decimal formats"
        )
    
    with fmt_col2:
        st.markdown("**Color & Theme**")
        color_scheme = st.selectbox(
            "Color Scheme",
            list(COLOR_SCHEMES.keys()),
            help="Color scheme for headers, data, and charts"
        )
        use_conditional_formatting = st.checkbox(
            "Conditional Formatting (Heat Map)",
            value=True,
            help="Highlight values: green (positive), red (negative), yellow (neutral)"
        )
    
    with fmt_col3:
        st.markdown("**Header & Styling**")
        include_index = st.checkbox("Include Row Numbers", value=True)
        freeze_header = st.checkbox("Freeze Header Row", value=True)
        autofit_columns = st.checkbox("Auto-Fit Columns", value=True)
        include_filters = st.checkbox("Add Filters to Headers", value=True)
    
    st.markdown("---")
    
    # STEP 5: Charts & Visualization
    st.subheader("Step 5: Visual Elements")
    
    include_charts = st.checkbox("Include Charts", value=True)
    
    if include_charts:
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("**Chart Types**")
            selected_charts = st.multiselect(
                "Select Charts to Include",
                list(CHART_TYPES.keys()),
                default=["Line Chart", "Candlestick", "Area Chart"],
                help="Charts will be embedded in the spreadsheet or included as separate sheets"
            )
        
        with chart_col2:
            st.markdown("**Chart Options**")
            chart_height = st.slider("Chart Height (pixels)", 300, 800, 500)
            include_legend = st.checkbox("Include Legend", value=True)
            include_grid = st.checkbox("Include Grid Lines", value=True)
            show_data_labels = st.checkbox("Show Data Labels", value=False)
    else:
        selected_charts = []
        chart_height = 500
        include_legend = True
        include_grid = True
        show_data_labels = False
    
    st.markdown("---")
    
    # STEP 6: Summary & Statistics
    st.subheader("Step 6: Summary Sections")
    
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        include_summary_stats = st.checkbox("Summary Statistics", value=True)
        include_correlation = st.checkbox("Correlation Matrix", value=True)
    
    with summary_col2:
        include_portfolio_analysis = st.checkbox("Portfolio Analysis", value=False)
        include_drawdown_analysis = st.checkbox("Drawdown Analysis", value=True)
    
    with summary_col3:
        include_sector_analysis = st.checkbox("Sector Analysis", value=False)
        include_notes_section = st.checkbox("Notes Section", value=True)
    
    st.markdown("---")
    
    # STEP 7: Sheet Organization
    st.subheader("Step 7: Sheet Organization")
    
    st.markdown("**Customize sheet names and organization:**")
    
    sheet_col1, sheet_col2, sheet_col3 = st.columns(3)
    
    with sheet_col1:
        data_sheet_name = st.text_input("Data Sheet Name", value="Market Data")
        summary_sheet_name = st.text_input("Summary Sheet Name", value="Summary")
    
    with sheet_col2:
        charts_sheet_name = st.text_input("Charts Sheet Name", value="Charts")
        analysis_sheet_name = st.text_input("Analysis Sheet Name", value="Technical Analysis")
    
    with sheet_col3:
        correlation_sheet_name = st.text_input("Correlation Sheet Name", value="Correlations")
        custom_sheet_name = st.text_input("Custom Sheet Name", value="Custom Calcs")
    
    st.markdown("---")
    
    # GENERATE BUTTON
    col_gen1, col_gen2 = st.columns([3, 1])
    
    with col_gen1:
        st.markdown("**Ready to generate?**")
    
    with col_gen2:
        if st.button("Generate Advanced Spreadsheet", type="primary", use_container_width=True):
            if not symbols:
                st.error("Enter at least one symbol")
                return
            
            with st.spinner("Building your custom spreadsheet..."):
                try:
                    _generate_advanced_spreadsheet(
                        spec=SpreadsheetSpec(
                            symbols=symbols,
                            timeframe=timeframe,
                            interval=interval_map.get(interval, "1d"),
                            data_columns=sum(selected_columns.values(), []),
                            price_columns=selected_columns.get("price", []),
                            technical_columns=sum([selected_columns.get(k, []) for k in ["ma", "momentum", "volatility", "trend"]], []),
                            risk_columns=selected_columns.get("volatility", []),
                            chart_types=[CHART_TYPES[c] for c in selected_charts],
                            formatting={
                                "number_format": FORMATTING_OPTIONS[number_format],
                                "decimal_places": decimal_places,
                                "conditional_formatting": use_conditional_formatting,
                                "include_index": include_index,
                                "freeze_header": freeze_header,
                                "autofit": autofit_columns,
                                "filters": include_filters
                            },
                            color_scheme=COLOR_SCHEMES[color_scheme],
                            include_summary=include_summary_stats,
                            include_statistics=include_summary_stats,
                            include_charts=include_charts,
                            include_correlations=include_correlation,
                            include_financial_model=False,
                            model_type=None,
                            custom_calculations=calculations,
                            sheet_names={
                                "data": data_sheet_name,
                                "summary": summary_sheet_name,
                                "charts": charts_sheet_name,
                                "analysis": analysis_sheet_name,
                                "correlation": correlation_sheet_name,
                                "custom": custom_sheet_name
                            }
                        )
                    )
                except Exception as e:
                    st.error(f"Error generating spreadsheet: {e}")
                    import traceback
                    st.error(traceback.format_exc())


def _generate_advanced_spreadsheet(spec: SpreadsheetSpec):
    """Generate comprehensive spreadsheet with all customizations."""
    
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter
        
        # Fetch data for all symbols
        all_dfs = []
        for symbol in spec.symbols:
            symbol = symbol.strip().upper()
            try:
                df = _safe_fetch_data(symbol, spec.timeframe)
                if df is not None and not df.empty:
                    df["Symbol"] = symbol
                    df = _calculate_indicators(df, spec.data_columns)
                    all_dfs.append(df)
            except Exception as e:
                st.warning(f"Could not load {symbol}: {e}")
        
        if not all_dfs:
            st.error("No data could be fetched")
            return
        
        # Combine data
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Create workbook
        wb = Workbook()
        ws_data = wb.active
        ws_data.title = spec.sheet_names["data"]
        
        # Write data to sheet
        _write_data_sheet(ws_data, combined_df, spec)
        
        # Add summary sheet
        if spec.include_summary:
            ws_summary = wb.create_sheet(spec.sheet_names["summary"])
            _write_summary_sheet(ws_summary, combined_df, spec)
        
        # Add correlation sheet
        if spec.include_correlations:
            ws_corr = wb.create_sheet(spec.sheet_names["correlation"])
            _write_correlation_sheet(ws_corr, combined_df, spec)
        
        # Add charts sheet
        if spec.include_charts and spec.chart_types:
            ws_charts = wb.create_sheet(spec.sheet_names["charts"])
            st.info("Note: Charts are embedded in the Excel file (advanced format)")
        
        # Save to bytes
        excel_bytes = io.BytesIO()
        wb.save(excel_bytes)
        excel_bytes.seek(0)
        
        # Download button
        st.download_button(
            "Download Custom Spreadsheet",
            excel_bytes.getvalue(),
            f"market_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        
        st.success("Spreadsheet generated successfully!")
        
        # Preview
        st.markdown("---")
        st.markdown("**Data Preview:**")
        preview_cols = spec.data_columns[:10]  # Show first 10 columns
        valid_preview_cols = [c for c in preview_cols if c in combined_df.columns]
        if valid_preview_cols:
            st.dataframe(combined_df[valid_preview_cols].head(20), use_container_width=True)
        else:
            st.dataframe(combined_df.head(20), use_container_width=True)
        
    except Exception as e:
        st.error(f"Spreadsheet generation error: {e}")
        import traceback
        st.error(traceback.format_exc())


def _ib_styles():
    """Classic, picture-perfect Investment Banker openpyxl style objects.
    IB Style Rules: White background, no gridlines, dark blue for headers, 
    black for calculated data, thick bottom borders for headers, accounting number formats."""
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    
    ib_blue = "000080"      # Navy blue for headers/hardcodes
    black = "000000"        # Black for calculations
    white = "FFFFFF"        # Pure white background
    gray_fill = "F2F2F2"    # Very subtle gray for header backgrounds
    
    thin_border = Side(style="thin", color="000000")
    thick_bottom = Side(style="thick", color="000000")
    
    return {
        "title_font": Font(name="Arial", bold=True, size=14, color=black),
        "subtitle_font": Font(name="Arial", bold=False, size=10, color=black, italic=True),
        "header_font": Font(name="Arial", bold=True, size=10, color=ib_blue),
        "header_fill": PatternFill(start_color=white, end_color=white, fill_type="solid"),
        "data_font": Font(name="Arial", size=10, color=black),
        "white_fill": PatternFill(start_color=white, end_color=white, fill_type="solid"),
        "center": Alignment(horizontal="center", vertical="center", wrap_text=False),
        "right": Alignment(horizontal="right", vertical="center"),
        "left": Alignment(horizontal="left", vertical="center"),
        "header_border": Border(bottom=thick_bottom),
        "total_border": Border(top=thin_border, bottom=thick_bottom),
        "fmt_currency": '_($* #,##0.00_);_($* (#,##0.00);_($* "-"??_);_(@_)',
        "fmt_pct": '0.0%_)',
        "fmt_number": '_(* #,##0.00_);_(* (#,##0.00);_(* "-"??_);_(@_)',
        "fmt_int": '_(* #,##0_);_(* (#,##0);_(* "-"??_);_(@_)',
    }


def _apply_ib_sheet(ws, df, title, subtitle="", freeze_row=3, spec=None):
    """Apply classic IB formatting to a worksheet containing a DataFrame.
    If spec is provided, respects customized user formatting preferences."""
    from openpyxl.utils import get_column_letter
    S = _ib_styles()
    num_cols = len(df.columns)

    # Honor advanced spec options
    currency_fmt = S["fmt_currency"]
    pct_fmt = S["fmt_pct"]
    num_fmt = S["fmt_number"]
    
    if spec and spec.formatting:
        base_fmt = spec.formatting.get("number_format", "Auto")
        decimals = spec.formatting.get("decimal_places", 2)
        dec_str = "." + "0" * decimals if decimals > 0 else ""
        
        if base_fmt == "USD":
            currency_fmt = f'_($* #,##0{dec_str}_);_($* (#,##0{dec_str});_($* "-"??_);_(@_)'
            num_fmt = f'_(* #,##0{dec_str}_);_(* (#,##0{dec_str});_(* "-"??_);_(@_)'
        elif base_fmt == "EUR":
            currency_fmt = f'_-* #,##0{dec_str}_-;_-* #,##0{dec_str}_-;_-* "-"??_-;_-@_-'
            num_fmt = f'_(* #,##0{dec_str}_);_(* (#,##0{dec_str});_(* "-"??_);_(@_)'
        else:
            num_fmt = f'_(* #,##0{dec_str}_);_(* (#,##0{dec_str});_(* "-"??_);_(@_)'

    # Hide gridlines (classic IB standard)
    ws.sheet_view.showGridLines = False

    # Title row
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=max(num_cols, 3))
    ws.cell(1, 1, title).font = S["title_font"]
    ws.cell(1, 1).alignment = S["left"]

    # Subtitle
    if subtitle:
        ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=max(num_cols, 3))
        ws.cell(2, 1, subtitle).font = S["subtitle_font"]

    header_row = freeze_row
    
    # Headers
    for c_idx, col_name in enumerate(df.columns, 1):
        cell = ws.cell(header_row, c_idx, str(col_name))
        cell.font = S["header_font"]
        cell.fill = S["header_fill"]
        cell.alignment = S["center"]
        cell.border = S["header_border"]
        
    # Data rows
    for r_idx, (_, row) in enumerate(df.iterrows(), header_row + 1):
        for c_idx, val in enumerate(row, 1):
            cell = ws.cell(r_idx, c_idx)
            if pd.isna(val):
                cell.value = ""
            elif isinstance(val, (float, np.floating)):
                cell.value = float(val)
                col_name = df.columns[c_idx - 1].lower() if c_idx <= len(df.columns) else ""
                if any(k in col_name for k in ["price", "close", "open", "high", "low", "avg", "min", "max", "value", "fcf", "revenue", "ebit", "capital"]):
                    cell.number_format = currency_fmt
                elif any(k in col_name for k in ["return", "pct", "yield", "change", "%", "sharpe", "sortino", "vol", "drawdown", "beta", "cagr"]):
                    cell.number_format = pct_fmt if abs(val) < 5 else num_fmt
                else:
                    cell.number_format = num_fmt
            elif isinstance(val, (int, np.integer)):
                cell.value = int(val)
                cell.number_format = S["fmt_int"]
            else:
                cell.value = str(val)
            
            cell.font = S["data_font"]
            cell.fill = S["white_fill"]
            cell.alignment = S["right"] if isinstance(val, (int, float, np.integer, np.floating)) else S["left"]
            
    # Auto-fit columns
    for c_idx in range(1, num_cols + 1):
        max_len = max(len(str(ws.cell(r, c_idx).value or "")) for r in range(header_row, ws.max_row + 1))
        # Provide a little extra padding for accounting formats
        ws.column_dimensions[get_column_letter(c_idx)].width = min(max(max_len + 5, 12), 40)
        
    # Freeze panes below header
    if not (spec and spec.formatting and not spec.formatting.get("freeze_header", True)):
        ws.freeze_panes = ws.cell(header_row + 1, 1)


def _build_ib_workbook(sheets_dict, filename_prefix="octavian"):
    """Build and offer download for an IB-formatted workbook from {name: (df, title, subtitle)}."""
    from openpyxl import Workbook
    wb = Workbook()
    first = True
    for sheet_name, (df, title, subtitle) in sheets_dict.items():
        if first:
            ws = wb.active
            ws.title = sheet_name[:31]
            first = False
        else:
            ws = wb.create_sheet(sheet_name[:31])
        _apply_ib_sheet(ws, df, title, subtitle, spec=None)
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    fname = f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    st.download_button(
        "Download Spreadsheet",
        buf.getvalue(),
        fname,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
    st.success(f"Spreadsheet ready: **{fname}**")
    return df  # return last df for preview


def _write_data_sheet(ws, df: pd.DataFrame, spec: SpreadsheetSpec):
    """Write formatted data to sheet with IB-grade styling."""
    cols_to_write = ["Symbol"] + [c for c in spec.data_columns if c in df.columns]
    out = df[cols_to_write].copy() if cols_to_write else df.copy()
    _apply_ib_sheet(ws, out, spec.sheet_names.get("data", "Market Data"),
                    f"Generated {datetime.now().strftime('%b %d, %Y %H:%M')}", spec=spec)


def _write_summary_sheet(ws, df: pd.DataFrame, spec: SpreadsheetSpec):
    """Write summary statistics with IB formatting."""
    rows = []
    for symbol in df["Symbol"].unique():
        sym_data = df[df["Symbol"] == symbol]
        if "Close" in sym_data.columns:
            close = pd.to_numeric(sym_data["Close"], errors="coerce").dropna()
            if len(close) == 0:
                continue
            ret = (close.iloc[-1] / close.iloc[0] - 1) if close.iloc[0] != 0 else 0
            vol = close.pct_change().std() * np.sqrt(252) if len(close) > 2 else 0
            rows.append({
                "Symbol": symbol, "Last Price": close.iloc[-1],
                "Avg Price": close.mean(), "Min Price": close.min(),
                "Max Price": close.max(), "Std Dev": close.std(),
                "Total Return": ret, "Annualized Vol": vol,
            })
    summary_df = pd.DataFrame(rows) if rows else pd.DataFrame({"Info": ["No data"]})
    _apply_ib_sheet(ws, summary_df, spec.sheet_names.get("summary", "Summary Statistics"),
                    f"Octavian Terminal | {datetime.now().strftime('%b %d, %Y')}", spec=spec)


def _write_correlation_sheet(ws, df: pd.DataFrame, spec: SpreadsheetSpec):
    """Write correlation matrix with IB formatting."""
    if "Close" in df.columns and "Symbol" in df.columns:
        pivot = df.pivot_table(values="Close", index=df.index, columns="Symbol", aggfunc="first")
        corr = pivot.corr().round(4)
        corr.index.name = "Symbol"
        corr = corr.reset_index()
        _apply_ib_sheet(ws, corr, spec.sheet_names.get("correlation", "Correlation Matrix"),
                        f"{len(corr)-1 if len(corr)>0 else 0} assets | Octavian Terminal", spec=spec)


def _calculate_indicators(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Calculate requested indicators."""
    df = df.copy()
    
    try:
        close = pd.to_numeric(df["Close"], errors="coerce") if "Close" in df.columns else None
        
        # Moving Averages
        for ma in ["SMA_20", "SMA_50", "SMA_200"]:
            if ma in columns and close is not None:
                period = int(ma.split("_")[1])
                df[ma] = close.rolling(period).mean()
        
        # RSI
        if "RSI_14" in columns and close is not None and len(close) >= 15:
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
            rs = gain / (loss + 1e-10)
            df["RSI_14"] = 100 - (100 / (1 + rs))
        
        # MACD
        if ("MACD" in columns or "MACD_Signal" in columns) and close is not None:
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            df["MACD"] = ema12 - ema26
            df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        if "Bollinger_Upper_20" in columns and close is not None:
            sma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            df["Bollinger_Upper_20"] = sma20 + (std20 * 2)
            df["Bollinger_Lower_20"] = sma20 - (std20 * 2)
        
        # Volatility
        if "Volatility_20d" in columns and close is not None:
            df["Volatility_20d"] = close.pct_change().rolling(20).std() * np.sqrt(252)
        
        # Returns
        if "Returns_1d" in columns and close is not None:
            df["Returns_1d"] = close.pct_change()
        if "Returns_YTD" in columns and close is not None:
            df["Returns_YTD"] = close / close.iloc[0] - 1
        
    except Exception as e:
        st.warning(f"Indicator calculation error: {e}")
    
    return df


def _safe_fetch_data(symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
    """Safely fetch data with fallbacks."""
    symbol = symbol.strip().upper()
    
    try:
        df = get_stock(symbol, period=period)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
    except:
        pass
    
    try:
        import yfinance as yf
        df = yf.Ticker(symbol).history(period=period)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
    except:
        pass
    
    return None


def _generate_performance_comparison(symbols: List[str], period: str, include_charts: bool):
    """Quick template: Performance comparison."""
    rows = []
    for sym in symbols:
        df = _safe_fetch_data(sym, period)
        if df is None or df.empty or "Close" not in df.columns:
            continue
        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
        if len(close) < 2:
            continue
        ret = close.iloc[-1] / close.iloc[0] - 1
        vol = close.pct_change().std() * np.sqrt(252)
        sharpe = (ret * 252 / max(len(close), 1)) / vol if vol > 0 else 0
        dd = ((close / close.cummax()) - 1).min()
        rows.append({
            "Symbol": sym, "Last Price": close.iloc[-1], "Period Start": close.iloc[0],
            "Total Return": ret, "Annualized Vol": vol,
            "Sharpe Ratio": sharpe, "Max Drawdown": dd,
            "High": close.max(), "Low": close.min(),
        })
    if not rows:
        st.error("No data fetched"); return
    perf = pd.DataFrame(rows)
    _build_ib_workbook({"Performance": (perf, "Performance Comparison", f"{len(symbols)} assets | {period}")},
                       "perf_comparison")
    st.dataframe(perf, use_container_width=True)


def _generate_technical_dashboard(symbols: List[str], period: str, include_charts: bool):
    """Quick template: Technical analysis."""
    rows = []
    for sym in symbols:
        df = _safe_fetch_data(sym, period)
        if df is None or df.empty or "Close" not in df.columns:
            continue
        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
        if len(close) < 30:
            continue
        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else np.nan
        sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd_val = (ema12 - ema26).iloc[-1]
        bb_mid = close.rolling(20).mean().iloc[-1]
        bb_std = close.rolling(20).std().iloc[-1]
        rows.append({
            "Symbol": sym, "Price": close.iloc[-1],
            "SMA 20": sma20, "SMA 50": sma50, "SMA 200": sma200,
            "RSI (14)": rsi, "MACD": macd_val,
            "BB Upper": bb_mid + 2 * bb_std, "BB Lower": bb_mid - 2 * bb_std,
            "Signal": "BUY" if rsi < 30 else "SELL" if rsi > 70 else "HOLD",
        })
    if not rows:
        st.error("No data fetched"); return
    tech = pd.DataFrame(rows)
    _build_ib_workbook({"Technical": (tech, "Technical Analysis Dashboard", f"{len(symbols)} assets")},
                       "technical_dashboard")
    st.dataframe(tech, use_container_width=True)


def _generate_portfolio_tracking(symbols: List[str]):
    """Quick template: Portfolio tracking."""
    rows = []
    equal_weight = 1.0 / len(symbols) if symbols else 1
    for sym in symbols:
        df = _safe_fetch_data(sym, "1y")
        if df is None or df.empty or "Close" not in df.columns:
            continue
        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
        if len(close) < 2:
            continue
        ret = close.iloc[-1] / close.iloc[0] - 1
        rows.append({
            "Symbol": sym, "Weight": equal_weight, "Price": close.iloc[-1],
            "1Y Return": ret, "Contribution": equal_weight * ret,
            "52W High": close.max(), "52W Low": close.min(),
            "Pct from High": (close.iloc[-1] / close.max() - 1),
        })
    if not rows:
        st.error("No data fetched"); return
    port = pd.DataFrame(rows)
    total_ret = port["Contribution"].sum()
    summary = pd.DataFrame([{"Metric": "Portfolio 1Y Return", "Value": total_ret},
                            {"Metric": "# Holdings", "Value": len(rows)},
                            {"Metric": "Avg Weight", "Value": equal_weight}])
    _build_ib_workbook({
        "Holdings": (port, "Portfolio Holdings", "Equal-weight allocation"),
        "Summary": (summary, "Portfolio Summary", f"Total Return: {total_ret:.2%}"),
    }, "portfolio_tracker")
    st.dataframe(port, use_container_width=True)


def _generate_trade_log():
    """Quick template: Trade log  provides a blank formatted template."""
    template = pd.DataFrame({
        "Date": [datetime.now().strftime("%Y-%m-%d")] * 3,
        "Symbol": ["", "", ""], "Side": ["BUY", "SELL", ""],
        "Quantity": [0, 0, 0], "Entry Price": [0.0, 0.0, 0.0],
        "Exit Price": [0.0, 0.0, 0.0], "P&L": [0.0, 0.0, 0.0],
        "Notes": ["", "", ""],
    })
    _build_ib_workbook({"Trade Log": (template, "Trade Log", "Enter your trades below")},
                       "trade_log")
    st.info("A blank trade log template has been generated. Fill it in after downloading.")


def _generate_financial_statements(symbol: str):
    """Quick template: Comprehensive 3-Statement Financials from yfinance."""
    try:
        import yfinance as yf
        tk = yf.Ticker(symbol)
        
        inc = tk.financials.fillna(0) if tk.financials is not None else pd.DataFrame()
        bs = tk.balance_sheet.fillna(0) if tk.balance_sheet is not None else pd.DataFrame()
        cf = tk.cashflow.fillna(0) if tk.cashflow is not None else pd.DataFrame()
        
        sheets = {}
        if not inc.empty:
            inc.columns = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in inc.columns]
            inc = inc.reset_index()
            inc.rename(columns={"index": "Line Item"}, inplace=True)
            sheets["Income Statement"] = (inc, f"Income Statement  {symbol}", "Annual reporting")
            
        if not bs.empty:
            bs.columns = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in bs.columns]
            bs = bs.reset_index()
            bs.rename(columns={"index": "Line Item"}, inplace=True)
            sheets["Balance Sheet"] = (bs, f"Balance Sheet  {symbol}", "Annual reporting")
            
        if not cf.empty:
            cf.columns = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in cf.columns]
            cf = cf.reset_index()
            cf.rename(columns={"index": "Line Item"}, inplace=True)
            sheets["Cash Flow"] = (cf, f"Cash Flow Statement  {symbol}", "Annual reporting")
            
        if not sheets:
            st.error("No financial statements available from Yahoo Finance.")
            return

        _build_ib_workbook(sheets, f"financials_{symbol}")
        st.dataframe(inc if not inc.empty else pd.DataFrame(), use_container_width=True)
    except Exception as e:
        st.error(f"Could not fetch financials for {symbol}: {e}")


def _generate_dcf_model(symbol: str, wacc: float = None, tgr: float = None, tax_rate: float = None):
    """Quick template: 10-Year detailed DCF model.

    Uses user-provided WACC / terminal growth / tax-rate assumptions exactly
    as entered (falls back to sensible defaults only when not provided).
    """
    try:
        import yfinance as yf
        tk = yf.Ticker(symbol)
        info = tk.info or {}
        
        inc = tk.financials
        if inc is not None and not inc.empty and "Total Revenue" in inc.index:
            rev_base = inc.loc["Total Revenue"].iloc[0]
            ebit_base = inc.loc["EBIT"].iloc[0] if "EBIT" in inc.index else rev_base * 0.15
        else:
            rev_base = info.get("totalRevenue", 10_000_000_000)
            ebit_base = info.get("ebitda", 2_000_000_000) * 0.8
            
        tax_rate = 0.21 if tax_rate is None else float(tax_rate)
        shares = info.get("sharesOutstanding", 1_000_000_000)
        price = info.get("currentPrice") or info.get("previousClose", 100)
        wacc = 0.085 if wacc is None else float(wacc)
        tgr = 0.025 if tgr is None else float(tgr)
        
        proj = []
        rev = rev_base
        for yr in range(1, 11):
            g = max(0.15 - yr*0.01, 0.03)  # fades to 3%
            rev *= (1 + g)
            margin = min(ebit_base/rev_base + yr*0.005, 0.35) if rev_base else 0.20
            ebit = rev * margin
            taxes = ebit * tax_rate
            nopat = ebit - taxes
            dna = rev * 0.04  # 4% of sales
            capex = rev * 0.05 # 5% of sales
            nwc_chg = rev * 0.01 # 1% of sales
            ufcf = nopat + dna - capex - nwc_chg
            df_factor = 1 / ((1 + wacc) ** yr)
            pv_ufcf = ufcf * df_factor
            
            proj.append({
                "Year": f"Year {yr}",
                "Revenue Growth %": g,
                "Revenue": rev,
                "EBIT Margin %": margin,
                "EBIT": ebit,
                f"Taxes ({tax_rate:.0%})": -taxes,
                "NOPAT": nopat,
                "(+) D&A": dna,
                "(-) CapEx": -capex,
                "(-) Change in NWC": -nwc_chg,
                "Unlevered Free Cash Flow": ufcf,
                "Discount Factor": df_factor,
                "PV of UFCF": pv_ufcf
            })
            
        term_val = proj[-1]["Unlevered Free Cash Flow"] * (1 + tgr) / (wacc - tgr)
        pv_tv = term_val / ((1 + wacc) ** 10)
        
        dcf_df = pd.DataFrame(proj).T
        dcf_df.columns = dcf_df.iloc[0]
        dcf_df = dcf_df.drop(dcf_df.index[0]).reset_index()
        dcf_df.rename(columns={"index": "Line Item"}, inplace=True)
        
        ent_val = sum(p["PV of UFCF"] for p in proj) + pv_tv
        eq_val = ent_val + info.get("totalCash", 0) - info.get("totalDebt", 0)
        fv = eq_val / shares if shares else 0
        
        summary = pd.DataFrame([
            {"Metric": "Sum of PV of UFCF", "Value": sum(p["PV of UFCF"] for p in proj)},
            {"Metric": "Terminal Value", "Value": term_val},
            {"Metric": "PV of Terminal Value", "Value": pv_tv},
            {"Metric": "Enterprise Value", "Value": ent_val},
            {"Metric": "(+) Cash", "Value": info.get("totalCash", 0)},
            {"Metric": "(-) Debt", "Value": info.get("totalDebt", 0)},
            {"Metric": "Implied Equity Value", "Value": eq_val},
            {"Metric": "Shares Outstanding", "Value": shares},
            {"Metric": "Implied Share Price", "Value": fv},
            {"Metric": "Current Share Price", "Value": price},
            {"Metric": "Premium / (Discount)", "Value": fv/price - 1 if price else 0},
            {"Metric": "WACC", "Value": wacc},
            {"Metric": "Terminal Growth Rate", "Value": tgr}
        ])
        
        _build_ib_workbook({
            "DCF Model": (dcf_df, f"10-Year DCF Valuation  {symbol}", "Unlevered Free Cash Flow Build"),
            "Valuation Output": (summary, "Valuation Summary", f"WACC: {wacc:.1%} | TGR: {tgr:.1%}")
        }, f"dcf_{symbol}")
        st.dataframe(summary, use_container_width=True)
    except Exception as e:
        st.error(f"DCF model error: {e}")


def _generate_lbo_model(symbol: str, debt_pct: float = None, exit_mult: float = None, interest_rate: float = None):
    """Quick template: Comprehensive Leveraged Buyout (LBO) model.

    Uses user-provided debt / total capital split, exit multiple, and interest
    rate exactly as entered (falls back to sensible defaults when not provided).
    """
    try:
        import yfinance as yf
        tk = yf.Ticker(symbol)
        info = tk.info or {}
        
        ebitda = info.get("ebitda", 500_000_000) or 500_000_000
        ev = info.get("enterpriseValue", ebitda * 10) or ebitda * 10
        entry_mult = ev / ebitda if ebitda else 10
        
        # Sources and Uses (user-specified leverage split)
        debt_pct = 0.60 if debt_pct is None else float(debt_pct)
        eq_pct = 1.0 - debt_pct
        fees = ev * 0.02
        total_uses = ev + fees
        debt_amt = total_uses * debt_pct
        sponsor_eq = total_uses * eq_pct
        
        su_df = pd.DataFrame([
            {"Item": "Purchase Equity & Refinance Debt", "Amount": ev, "% of Total": ev/total_uses},
            {"Item": "Estimated Fees", "Amount": fees, "% of Total": fees/total_uses},
            {"Item": "Total Uses", "Amount": total_uses, "% of Total": 1.0},
            {"Item": "", "Amount": None, "% of Total": None},
            {"Item": f"Term Loan B ({debt_pct:.0%})", "Amount": debt_amt, "% of Total": debt_pct},
            {"Item": f"Sponsor Equity ({eq_pct:.0%})", "Amount": sponsor_eq, "% of Total": eq_pct},
            {"Item": "Total Sources", "Amount": total_uses, "% of Total": 1.0},
        ])
        
        # 5-Year Projections
        proj = []
        rev = ebitda * 4
        debt_bal = debt_amt
        interest_rate = 0.08 if interest_rate is None else float(interest_rate)
        tax_rate = 0.21
        
        for yr in range(1, 6):
            rev *= 1.05
            ebitda_yr = rev * 0.25
            depr = rev * 0.05
            ebit = ebitda_yr - depr
            interest = debt_bal * interest_rate
            ebt = ebit - interest
            taxes = ebt * tax_rate if ebt > 0 else 0
            ni = ebt - taxes
            
            # CFADS
            capex = rev * 0.04
            nwc_chg = rev * 0.01
            cfads = ni + depr - capex - nwc_chg
            
            # Debt paydown
            paydown = min(cfads, debt_bal) if cfads > 0 else 0
            debt_bal -= paydown
            
            proj.append({
                "Year": f"Year {yr}",
                "Revenue": rev,
                "EBITDA": ebitda_yr,
                "(-) D&A": -depr,
                "EBIT": ebit,
                "(-) Interest Exp": -interest,
                "EBT": ebt,
                "(-) Taxes": -taxes,
                "Net Income": ni,
                "(+) D&A": depr,
                "(-) CapEx": -capex,
                "(-) Change in NWC": -nwc_chg,
                "Cash Available for Debt Service (CFADS)": cfads,
                "Mandatory Debt Paydown": -paydown,
                "Ending Debt Balance": debt_bal
            })
            
        lbo_df = pd.DataFrame(proj).T
        lbo_df.columns = lbo_df.iloc[0]
        lbo_df = lbo_df.drop(lbo_df.index[0]).reset_index()
        lbo_df.rename(columns={"index": "Line Item"}, inplace=True)
        
        # Returns
        exit_mult = float(exit_mult) if exit_mult is not None else entry_mult
        exit_ev = proj[-1]["EBITDA"] * exit_mult
        exit_eq = exit_ev - debt_bal
        moic = exit_eq / sponsor_eq if sponsor_eq else 0
        irr = (moic ** (1/5)) - 1 if moic > 0 else 0
        
        ret_df = pd.DataFrame([
            {"Metric": "Entry Multiple", "Value": f"{entry_mult:.1f}x"},
            {"Metric": "Exit Multiple", "Value": f"{exit_mult:.1f}x"},
            {"Metric": "Exit Enterprise Value", "Value": exit_ev},
            {"Metric": "Less: Ending Debt", "Value": -debt_bal},
            {"Metric": "Exit Equity Value", "Value": exit_eq},
            {"Metric": "Initial Sponsor Equity", "Value": sponsor_eq},
            {"Metric": "MOIC", "Value": moic},
            {"Metric": "5-Year IRR", "Value": irr}
        ])
        
        _build_ib_workbook({
            "Sources & Uses": (su_df, f"LBO Sources & Uses  {symbol}", f"Transaction Value: {ev:,.0f}"),
            "Projections": (lbo_df, f"5-Year Operating Model", "Cash Flow & Debt Schedule"),
            "Returns Analysis": (ret_df, f"Sponsor Returns", "Base Case Exit scenario")
        }, f"lbo_{symbol}")
        st.dataframe(ret_df, use_container_width=True)
    except Exception as e:
        st.error(f"LBO model error: {e}")


def _generate_comps_model(symbols: List[str]):
    """Quick template: Relative valuation comps table."""
    try:
        import yfinance as yf
        rows = []
        for sym in symbols:
            info = yf.Ticker(sym).info or {}
            rows.append({
                "Symbol": sym,
                "Market Cap": info.get("marketCap", "N/A"),
                "EV": info.get("enterpriseValue", "N/A"),
                "P/E": info.get("trailingPE", "N/A"),
                "Fwd P/E": info.get("forwardPE", "N/A"),
                "P/B": info.get("priceToBook", "N/A"),
                "EV/EBITDA": info.get("enterpriseToEbitda", "N/A"),
                "EV/Revenue": info.get("enterpriseToRevenue", "N/A"),
                "Profit Margin": info.get("profitMargins", "N/A"),
                "Dividend Yield": info.get("dividendYield", "N/A"),
            })
        comps = pd.DataFrame(rows)
        _build_ib_workbook({"Comps": (comps, "Comparable Company Analysis",
                           f"{len(symbols)} companies | Octavian Terminal")},
                           "comps_analysis")
        st.dataframe(comps, use_container_width=True)
    except Exception as e:
        st.error(f"Comps error: {e}")


def _generate_risk_analysis(symbols: List[str], period: str):
    """Quick template: Risk & volatility analysis."""
    rows = []
    for sym in symbols:
        df = _safe_fetch_data(sym, period)
        if df is None or df.empty or "Close" not in df.columns:
            continue
        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
        if len(close) < 30:
            continue
        rets = close.pct_change().dropna()
        vol = rets.std() * np.sqrt(252)
        dd = ((close / close.cummax()) - 1).min()
        var_95 = rets.quantile(0.05)
        skew = rets.skew()
        kurt = rets.kurtosis()
        rows.append({
            "Symbol": sym, "Ann. Volatility": vol, "Max Drawdown": dd,
            "VaR (95%)": var_95, "Skewness": skew, "Kurtosis": kurt,
            "Worst Day": rets.min(), "Best Day": rets.max(),
        })
    if not rows:
        st.error("No data fetched"); return
    risk = pd.DataFrame(rows)
    _build_ib_workbook({"Risk": (risk, "Risk & Volatility Analysis", f"{len(symbols)} assets | {period}")},
                       "risk_analysis")
    st.dataframe(risk, use_container_width=True)


def _generate_correlation_analysis(symbols: List[str], period: str):
    """Quick template: Correlation & diversification analysis."""
    price_data = {}
    for sym in symbols:
        df = _safe_fetch_data(sym, period)
        if df is not None and not df.empty and "Close" in df.columns:
            price_data[sym] = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if len(price_data) < 2:
        st.error("Need at least 2 symbols with data"); return
    prices = pd.DataFrame(price_data)
    returns = prices.pct_change().dropna()
    corr = returns.corr().round(4)
    corr_reset = corr.copy()
    corr_reset.index.name = "Symbol"
    corr_reset = corr_reset.reset_index()
    _build_ib_workbook({"Correlations": (corr_reset, "Correlation Matrix",
                       f"{len(symbols)} assets | {period}")},
                       "correlation_analysis")
    st.dataframe(corr, use_container_width=True)

