# Bugfix Requirements Document

## Introduction

This document captures requirements for seven bugs and quality deficiencies across the Octavian platform. The issues span output quality (Presentation Generator, Spreadsheet Generator), broken feature loading (Comparative Analysis, Financial Model Generator), AI response quality (Chatbot), and portfolio analytics correctness and completeness (Portfolio Analyzer risk metrics, ETF asset type support).

Collectively these issues degrade the platform from its intended institutional-grade standard. The fixes must restore correct behavior for affected inputs while preserving all existing functionality for unaffected inputs.

---

## Bug Analysis

### Current Behavior (Defect)

**Bug 1: Presentation Generator — Massively Incomplete Output**

1.1 WHEN a user requests a full investment banking pitch deck (e.g., an M&A pitch deck), THEN the system generates a few plain-text lines of summary data (e.g., acquirer, target, offer price, pro forma EPS, accretion/dilution) with no slides, no sections, no formatting, and no substantive content

1.2 WHEN a pitch deck is generated, THEN the system produces output that is not structured as a presentation and does not include the sections, narrative, financial analysis tables, or supporting content expected of an investment banking deck

**Bug 2: Spreadsheet Generator — Extremely Basic Output**

2.1 WHEN a user requests a complex financial spreadsheet (e.g., an M&A model, LBO model, or multi-tab financial analysis), THEN the system generates an extremely small, basic spreadsheet that does not reflect the depth, structure, or analytical content expected from an investment banking analyst

2.2 WHEN a spreadsheet is generated for a complex financial task, THEN the output lacks the tabs, formulas, data tables, supporting schedules, and formatting that a Goldman Sachs, JP Morgan, or Morgan Stanley analyst-level output would contain

**Bug 3: Comparative Analysis Feature — Blank Screen**

3.1 WHEN a user navigates to the Comparative Analysis feature, THEN the system displays a blank screen with no content, controls, or error messaging

3.2 WHEN the Comparative Analysis page attempts to load, THEN the system fails silently without rendering any UI elements

**Bug 4: Financial Model Generator — Blank Screen**

4.1 WHEN a user navigates to the Financial Model Generator feature, THEN the system displays a blank screen with no content, controls, or error messaging

4.2 WHEN the Financial Model Generator page attempts to load, THEN the system fails silently without rendering any UI elements

**Bug 5: AI Chatbot — Shallow, Non-Responsive Analysis**

5.1 WHEN a user asks a directional or analytical financial question (e.g., "in what direction do you see crude oil futures heading?"), THEN the system returns a processing status message (e.g., "Octavian analysis complete! Processed 2 symbols for swing timeframe, generated 0 charts in 54376ms") with no actual directional insight, analysis, or reasoning

5.2 WHEN a chatbot query is processed, THEN the system generates 0 charts and provides no substantive content — only a completion status string

5.3 WHEN a user submits a financial query, THEN the response does not address the user's question and instead reports internal processing metadata

**Bug 6: Portfolio Analyzer — All Risk Metrics Show Zero**

6.1 WHEN a user loads a portfolio with real holdings and a non-zero portfolio value (e.g., $157,184.81 with +8.86% return), THEN the system displays all risk metrics as zero: Sharpe Ratio 0.00, Sortino Ratio 0.00, Max Drawdown 0.0%, Win Rate 0.0%, Beta to SPY 0.00, Annual Volatility 0.00%, and 95% Daily VaR $0.00

6.2 WHEN risk metrics are calculated for a portfolio with actual position data, THEN the system returns zeroed-out values rather than computed values derived from the portfolio's holdings and returns

6.3 WHEN the Portfolio Analyzer generates improvement suggestions, THEN the system produces generic placeholder text (e.g., "Reallocate weights based on model contributor deficits") that is not specific to the portfolio's actual composition or risk profile

**Bug 7: Portfolio Analyzer — ETF Asset Type Missing**

7.1 WHEN a user attempts to add a position to the Portfolio Analyzer, THEN the system does not offer ETF as an available asset type option

7.2 WHEN a user holds ETF positions (e.g., SPY, QQQ, IWM) and attempts to add them, THEN the system cannot represent them correctly due to the absence of an ETF asset type

---

### Expected Behavior (Correct)

**Bug 1: Presentation Generator**

2.1 WHEN a user requests a full investment banking pitch deck, THEN the system SHALL generate a comprehensive, fully-formatted pitch deck that includes all standard IB sections: executive summary, situation overview, transaction rationale, valuation analysis (DCF, comparable companies, precedent transactions), pro forma analysis, deal structure, synergies analysis, risk factors, and appendix

2.2 WHEN a pitch deck is generated, THEN the system SHALL produce clearly delineated slides or sections with headers, supporting narrative text, formatted financial tables, and data that is coherent and substantive — not a summary of a few raw data points

**Bug 2: Spreadsheet Generator**

2.3 WHEN a user requests a complex financial spreadsheet, THEN the system SHALL generate a fully built-out, multi-tab spreadsheet with appropriate formulas, linked schedules, structured data tables, and formatting consistent with analyst-level investment banking output

2.4 WHEN a spreadsheet is generated for a complex financial task, THEN the output SHALL include all relevant supporting tabs (e.g., assumptions, income statement, balance sheet, cash flow, debt schedule, returns analysis) with formulas driving calculations rather than hardcoded values

**Bug 3: Comparative Analysis Feature**

2.5 WHEN a user navigates to the Comparative Analysis feature, THEN the system SHALL render the full Comparative Analysis UI with all controls, input fields, and content sections visible and functional

2.6 WHEN the Comparative Analysis page loads, THEN the system SHALL display without errors, blank screens, or missing content

**Bug 4: Financial Model Generator**

2.7 WHEN a user navigates to the Financial Model Generator feature, THEN the system SHALL render the full Financial Model Generator UI with all controls, input fields, and content sections visible and functional

2.8 WHEN the Financial Model Generator page loads, THEN the system SHALL display without errors, blank screens, or missing content

**Bug 5: AI Chatbot**

2.9 WHEN a user asks a directional or analytical financial question, THEN the system SHALL provide a substantive response that directly addresses the question with analysis, directional insight, supporting reasoning, and relevant market context

2.10 WHEN a chatbot query is processed, THEN the system SHALL generate charts, visualizations, or at minimum a coherent written analysis — not solely a processing status message

2.11 WHEN a financial query is submitted, THEN the response SHALL be framed around the user's question and SHALL NOT consist solely of internal processing metadata

**Bug 6: Portfolio Analyzer Risk Metrics**

2.12 WHEN a portfolio with real holdings and return history is loaded, THEN the system SHALL compute and display accurate, non-zero risk metrics: Sharpe Ratio, Sortino Ratio, Max Drawdown, Win Rate, Beta to SPY, Annual Volatility, and 95% Daily VaR — each calculated from actual portfolio position and return data

2.13 WHEN risk metrics are computed for a portfolio, THEN the values SHALL reflect the portfolio's actual composition, return distribution, and market correlation rather than returning zeros

2.14 WHEN the Portfolio Analyzer generates improvement suggestions, THEN the system SHALL produce specific, actionable recommendations derived from the portfolio's actual holdings, risk profile, concentration, and performance characteristics

**Bug 7: Portfolio Analyzer — ETF Asset Type**

2.15 WHEN a user adds a position to the Portfolio Analyzer, THEN the system SHALL offer ETF as a selectable asset type alongside equity, option, crypto, and FX

2.16 WHEN a user adds an ETF position, THEN the system SHALL correctly represent and include it in portfolio calculations, P&L, and risk analysis

---

### Unchanged Behavior (Regression Prevention)

3.1 WHEN a user requests a simple or short-form presentation (e.g., a brief company overview), THEN the system SHALL CONTINUE TO generate output at the appropriate level of detail for the request

3.2 WHEN a user requests a simple spreadsheet (e.g., a single-tab income statement), THEN the system SHALL CONTINUE TO generate a correctly structured spreadsheet without unnecessary complexity

3.3 WHEN a user navigates to any platform feature other than Comparative Analysis and Financial Model Generator, THEN the system SHALL CONTINUE TO load and render those features correctly without regression

3.4 WHEN a user submits a non-directional or data-retrieval chatbot query (e.g., "what is the current price of AAPL?"), THEN the system SHALL CONTINUE TO handle it correctly and return the appropriate data response

3.5 WHEN a user submits a chatbot query that currently works correctly and returns substantive analysis, THEN the system SHALL CONTINUE TO return the same quality of analysis without regression

3.6 WHEN a portfolio is loaded with positions whose risk metrics are genuinely zero or near-zero (e.g., a portfolio with a single position and no return history), THEN the system SHALL CONTINUE TO display zero or near-zero metrics accurately and not artificially inflate them

3.7 WHEN a user adds an equity, option, crypto, or FX position to the Portfolio Analyzer, THEN the system SHALL CONTINUE TO correctly classify and process those position types without regression

3.8 WHEN the Portfolio Analyzer calculates total portfolio value and unrealized P&L, THEN the system SHALL CONTINUE TO compute those metrics correctly as changes are made to support ETF positions and fix risk metric calculations

3.9 WHEN a user interacts with any currently-working page or feature of the platform, THEN the system SHALL CONTINUE TO function correctly with no regressions introduced by these fixes
