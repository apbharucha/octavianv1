# Implementation Plan: Trading Analysis Enhancements

## Overview

Eight targeted enhancements across the Octavian trading system: fundamental analysis integration, per-asset sentiment, options chain fixes, enhanced options UI, paper trading account lookup fix, LM Studio first-pass interpretation, datetime import fix, and portfolio import/manual entry.

## Tasks

- [x] 1. Fix `datetime` Not Defined Error in Portfolio Analyzer
  - [x] 1.1 Add `from datetime import datetime` import to all portfolio analyzer files that reference `datetime` without importing it
    - Search for bare `datetime.now()`, `datetime.strptime()`, etc. in `analytics_dashboard.py` and any `portfolio_analyzer.py` / `portfolio_*.py` files
    - Add the missing import at the top of each affected file
    - _Requirements: 7.1, 7.2_
  - [x] 1.2 Wrap render-time date/time calls in a try/except that logs the error and shows a user-readable message instead of a raw traceback
    - _Requirements: 7.3, 7.4_

- [x] 2. Fix Options Chain Missing `vol` Column Error
  - [x] 2.1 Add a `_normalize_options_chain` helper in `data_sources.py` (or wherever `get_options_chain` lives) that checks for `vol`, `strike`, `lastPrice`, `bid`, `ask`, `impliedVolatility`, and `openInterest` columns
    - If `vol` is absent, insert a zero-filled column and log `"vol column missing for {symbol}; defaulting to 0"`
    - For any other missing numeric column, substitute 0 rather than raising `KeyError`
    - _Requirements: 3.1, 3.2, 3.3_
  - [x] 2.2 Ensure `get_options_chain` returns a consistent schema DataFrame regardless of data provider, and returns an empty DataFrame with a user-readable error string on total fetch failure
    - _Requirements: 3.4, 3.5_
  - [ ]* 2.3 Write unit tests for `_normalize_options_chain` covering: all columns present, `vol` missing, multiple columns missing, and empty input
    - _Requirements: 3.1, 3.2, 3.3_

- [x] 3. Fix Paper Trading Account Lookup Failure
  - [x] 3.1 Update `PaperTradingSystem.list_accounts` in `paper_trading_system.py` to use a case-insensitive `user_id` comparison (`LOWER(user_id) = LOWER(?)`)
    - _Requirements: 5.2, 5.3_
  - [x] 3.2 Add a `user_id` alias mapping so that `"trader_1"` is treated as `"default_user"` — either normalise on write in `create_account`, or normalise on read in `list_accounts` and `get_account`
    - _Requirements: 5.6_
  - [x] 3.3 Ensure `paper_trading_ui.py` always reads `user_id` from `st.session_state` with a `"default_user"` default and passes that consistently to all `PaperTradingSystem` calls
    - _Requirements: 5.5_
  - [x] 3.4 Wrap all `get_account` / `list_accounts` database calls in try/except; return `None` or `[]` on error and log the exception
    - _Requirements: 5.7_
  - [ ]* 3.5 Write unit tests for `list_accounts` covering: exact match, case-insensitive match, `trader_1` alias, and no accounts found
    - _Requirements: 5.1, 5.2, 5.3, 5.6_

- [ ] 4. Checkpoint — Ensure all tests pass, ask the user if questions arise.

- [x] 5. LM Studio as First-Pass Interpreter for AI Chatbot
  - [x] 5.1 Add a `_call_lm_studio_interpreter` method to `OctavianEnhancedChatbot` in `ai_chatbot.py` that sends a structured prompt to LM Studio via `financial_llm_engine._call_llm` and parses the response into `{intent, symbols, timeframe, analysis_type}`
    - The prompt must instruct LM Studio to identify: (a) true financial intent, (b) symbols/asset classes, (c) implied timeframe, (d) most relevant analysis type
    - Parse JSON or structured text response; return `None` on parse failure
    - _Requirements: 6.1, 6.2, 6.3_
  - [x] 5.2 Refactor `process_unbiased_query` (and `process_query` if it exists) so that `_call_lm_studio_interpreter` is called first, before `_extract_symbols` or `_detect_intent_and_timeframe`
    - If LM Studio returns a valid `Intent_Context`, use its `symbols` and `intent` as primary inputs; supplement with regex extraction
    - If LM Studio is unavailable or times out (60 s), fall back to existing regex pipeline without degrading functionality
    - _Requirements: 6.1, 6.3, 6.4, 6.7, 6.8, 6.10_
  - [x] 5.3 Inject the LM Studio narrative into the final response text as a "LM Studio Interpretation" section displayed before all other analysis sections
    - _Requirements: 6.5, 6.6_
  - [x] 5.4 Add a `check_llm_connectivity()` call at chatbot startup; if LM Studio is unreachable, set a flag that surfaces a non-blocking status indicator in the Streamlit UI: `"LM Studio offline — using built-in analysis engine"`
    - _Requirements: 6.9_
  - [ ]* 5.5 Write unit tests for `_call_lm_studio_interpreter` covering: valid JSON response, malformed response (fallback), timeout (fallback), and LM Studio offline (fallback)
    - _Requirements: 6.3, 6.8_

- [x] 6. Fundamental Analysis Integration
  - [x] 6.1 Create a `FundamentalAnalyzer` class (new file `fundamental_analyzer.py`) with a `fetch_fundamentals(symbol)` method that retrieves: trailing P/E, forward P/E, EPS (TTM), revenue growth YoY, profit margin, and debt-to-equity ratio via `yfinance` `.info`
    - Return `None` for non-equity symbols (FX, futures, crypto)
    - Cache results per symbol with a 3600-second TTL using a simple dict + timestamp
    - _Requirements: 1.1, 1.6, 1.8_
  - [x] 6.2 Add a `score_fundamentals(fundamentals_dict)` method that returns a composite fundamental score in `[-1, 1]` and a qualitative label (`"Undervalued"`, `"Fairly Valued"`, `"Overvalued"`)
    - _Requirements: 1.7_
  - [x] 6.3 In `OctavianEnhancedChatbot._analyze_symbol_unbiased`, run `FundamentalAnalyzer.fetch_fundamentals` concurrently with the existing technical analysis using `ThreadPoolExecutor`
    - _Requirements: 1.2_
  - [x] 6.4 Implement `_compute_composite_score(technical_score, fundamental_score, timeframe_scope)` in the chatbot that applies the correct weights:
    - Default equity: 0.60 fundamental / 0.40 technical
    - `SCALPING` or `INTRADAY`: 0.20 fundamental / 0.80 technical
    - `INVESTMENT`: 0.75 fundamental / 0.25 technical
    - Non-equity (fundamental is `None`): 100% technical
    - _Requirements: 1.3, 1.4, 1.5, 1.6_
  - [x] 6.5 Add a "Fundamental Snapshot" section to the chatbot response text whenever fundamental data is available, displaying P/E, EPS, revenue growth, and the qualitative label
    - _Requirements: 1.7_
  - [x]* 6.6 Write property test: for any equity symbol and any `TimeframeScope`, the sum of fundamental weight + technical weight always equals 1.0
    - **Property 1: Weight sum invariant**
    - **Validates: Requirements 1.3, 1.4, 1.5, 1.6**
  - [x]* 6.7 Write unit tests for `FundamentalAnalyzer` covering: equity with full data, equity with partial data, non-equity returns `None`, cache hit within TTL, cache miss after TTL
    - _Requirements: 1.1, 1.6, 1.8_

- [x] 7. Per-Asset Sentiment Analysis During Symbol Analysis
  - [x] 7.1 Add a `analyze_symbol_sentiment(symbol, timeout=10)` method to `AdvancedNewsProcessor` in `advanced_news_processor.py` that filters fetched articles to those explicitly mentioning `symbol`, computes a weighted aggregate sentiment score using `SourceCredibilityEngine` credibility scores, and returns `{score: float, label: str, article_count: int, top_headlines: List[str]}`
    - Score range: `[-1.0, 1.0]`; labels: `VERY_BEARISH`, `BEARISH`, `NEUTRAL`, `BULLISH`, `VERY_BULLISH`
    - If the 10-second timeout is exceeded, return `None`
    - _Requirements: 2.1, 2.2, 2.5, 2.6_
  - [x] 7.2 Call `analyze_symbol_sentiment` inside `_analyze_symbol_unbiased` (concurrently with technical and fundamental analysis) and attach the result to the symbol analysis dict
    - _Requirements: 2.1_
  - [x] 7.3 Add a "Sentiment Overview" section to the chatbot response text for every symbol analysis, displaying: score, label, article count, and top 3 headlines
    - If sentiment timed out, note "Sentiment data unavailable"
    - _Requirements: 2.3, 2.5_
  - [x] 7.4 Add a "Sentiment Divergence" warning to the response when `abs(sentiment_score - technical_direction_score) > 0.4`
    - Map technical signal direction to a numeric: BULLISH → +1, BEARISH → -1, NEUTRAL → 0; scale to compare with sentiment score
    - _Requirements: 2.4_
  - [ ]* 7.5 Write property test: sentiment score is always in `[-1.0, 1.0]` for any non-empty article list
    - **Property 2: Sentiment score bounds**
    - **Validates: Requirements 2.2**
  - [ ]* 7.6 Write unit tests for `analyze_symbol_sentiment` covering: articles mentioning symbol, no matching articles, timeout exceeded, credibility weighting applied
    - _Requirements: 2.1, 2.2, 2.5, 2.6_

- [ ] 8. Checkpoint — Ensure all tests pass, ask the user if questions arise.

- [ ] 9. Enhanced Options Features — Strategy Research Lab
  - [x] 9.1 Add an "Options Strategy Builder" tab to `strategy_research_lab.py` with a multi-leg strategy constructor supporting: covered call, protective put, bull call spread, bear put spread, iron condor, iron butterfly, straddle, strangle, calendar spread, diagonal spread
    - Use `OptionsEngine.get_strategy_pnl_map` for payoff data; display payoff diagram, breakeven prices, max profit, and max loss
    - _Requirements: 4.1_
  - [x] 9.2 Add IV rank and IV percentile display to the Strategy Research Lab, computed against a 252-trading-day lookback using the options chain IV data
    - _Requirements: 4.2_
  - [x] 9.3 Render a volatility surface heatmap (strike vs. days-to-expiry) in the Strategy Research Lab using `OptionsEngine.generate_greek_surface`
    - _Requirements: 4.3_
  - [x] 9.4 Add a "Greeks Dashboard" to the Strategy Research Lab showing Delta, Gamma, Theta, Vega, Rho, Vanna, and Charm per leg plus net portfolio Greeks
    - _Requirements: 4.4_

- [-] 10. Enhanced Options Features — Simulation Hub
  - [ ] 10.1 Add a Monte Carlo P&L simulation panel to `market_simulation_engine.py` / the Simulation Hub UI that simulates an options position's P&L evolution over time and displays the distribution of outcomes at expiration
    - _Requirements: 4.5_
  - [ ] 10.2 Compute and display probability of profit (PoP) from the Monte Carlo distribution in the Simulation Hub
    - _Requirements: 4.6_

- [-] 11. Enhanced Options Features — Quant Portal
  - [ ] 11.1 Add a live options chain table to the "Derivative Intelligence" tab in `quant_portal.py` for a user-specified symbol, with columns: strike, expiry, type, bid, ask, last price, volume, open interest, IV, Delta, Gamma, Theta, Vega
    - Use the normalized `get_options_chain` (from task 2) to populate the table
    - _Requirements: 4.7_
  - [ ] 11.2 Add an unusual options activity scanner to the Quant Portal that flags contracts where `volume / open_interest >= 2`
    - _Requirements: 4.8_
  - [ ] 11.3 Add a contract detail panel in the Quant Portal: when a user selects a contract from the chain table, display a full Greeks breakdown and a single-contract payoff diagram
    - _Requirements: 4.9_
  - [ ] 11.4 Add a Put/Call Ratio metric to the Quant Portal, computed from the fetched options chain
    - _Requirements: 4.10_

- [ ] 12. Checkpoint — Ensure all tests pass, ask the user if questions arise.

- [-] 13. Portfolio Import and Manual Entry in Portfolio Analyzer
  - [ ] 13.1 Create `portfolio_analyzer.py` (or extend the existing file) with a `show_portfolio_analyzer()` Streamlit function that presents two input modes via tabs or a toggle: "Import from Paper Trading Account" and "Manual Entry"
    - Import `datetime` at the top of the file
    - _Requirements: 8.1, 7.1_
  - [ ] 13.2 Implement the "Import from Paper Trading Account" mode: list all accounts for `st.session_state.user_id` via `PaperTradingSystem.list_accounts`, let the user select one, then load all open positions from that account
    - On failure (no accounts, DB error), display a clear error message and fall back to Manual Entry mode without crashing
    - _Requirements: 8.2, 8.8_
  - [ ] 13.3 Implement the "Manual Entry" mode: a form accepting symbol, quantity, entry price, optional current price (defaults to live market price), and asset type (equity, option, crypto, FX) per position
    - Allow adding multiple positions before submitting; provide a row-delete control
    - _Requirements: 8.3, 8.4_
  - [ ] 13.4 Implement portfolio metrics computation and display for both modes: total portfolio value, unrealized P&L per position and aggregate, percentage return per position and overall, and position weight breakdown
    - When live price cannot be fetched and no current price was supplied, display "Price unavailable" and exclude that position from aggregate P&L
    - _Requirements: 8.5, 8.6, 8.9_
  - [ ] 13.5 Add a CSV export button that downloads the currently loaded portfolio (imported or manual) as a CSV file
    - _Requirements: 8.7_
  - [ ]* 13.6 Write unit tests for portfolio metrics computation covering: all prices available, one price unavailable (excluded from aggregate), zero positions, and imported vs manual entry producing identical metric shapes
    - _Requirements: 8.5, 8.6, 8.9_

- [ ] 14. Final Checkpoint — Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Tasks 1–4 are pure bug fixes and should be completed first
- Tasks 5–8 are the AI/analysis enhancements
- Tasks 9–12 are the options UI features
- Task 13 is the portfolio analyzer feature
- Each task references specific requirements for traceability
- Property tests validate universal correctness properties; unit tests validate specific examples and edge cases
