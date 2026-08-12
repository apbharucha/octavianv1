# Requirements Document

## Introduction

This document defines requirements for a set of enhancements to the Octavian trading/AI chatbot system. The enhancements span eight areas: (1) integrating fundamental analysis alongside technical analysis in model learning and decision-making, (2) adding per-asset sentiment analysis during symbol analysis, (3) fixing a missing `vol` column crash in the options chain data pipeline, (4) expanding options-specific features in the Strategy Research Lab, Simulation Hub, and Quant Portal, (5) fixing a paper trading account lookup failure, (6) restructuring the AI chatbot to use LM Studio as a first-pass interpreter that leads the chatbot's decision process, (7) fixing a `datetime` not defined error in the portfolio analyzer, and (8) adding portfolio import and manual entry capabilities to the portfolio analyzer.

---

## Glossary

- **Chatbot**: The `OctavianEnhancedChatbot` class in `ai_chatbot.py` that processes user queries and generates responses.
- **LM_Studio**: The local LLM server running at `http://localhost:1234` exposing an OpenAI-compatible API, accessed via `financial_llm_engine._call_llm`.
- **Fundamental_Analyzer**: The component responsible for fetching and scoring fundamental data (earnings, revenue, P/E, EPS, balance sheet metrics) for a given symbol.
- **Technical_Analyzer**: The existing ML/indicator pipeline (`ml_analysis.py`, `indicators.py`, `unbiased_market_analyzer.py`) that produces technical signals.
- **Sentiment_Analyzer**: The component (backed by `advanced_news_processor.py` and `news_analysis_engine.py`) that produces a sentiment score for a specific asset.
- **Options_Chain**: The tabular dataset of option contracts for an underlying symbol, fetched via `data_sources.get_options_chain` or `yfinance`.
- **Paper_Trading_System**: The `PaperTradingSystem` class in `paper_trading_system.py` that manages paper trading accounts in SQLite.
- **Strategy_Research_Lab**: The Streamlit page rendered by `strategy_research_lab.py`.
- **Simulation_Hub**: The Streamlit page rendered by `market_simulation_engine.py` / `simulation_viewer.py`.
- **Quant_Portal**: The Streamlit page rendered by `quant_portal.py`.
- **Analysis_Weight**: A numeric coefficient (0.0–1.0) applied to a signal category when computing a composite score.
- **Intent_Context**: The structured interpretation of a user query produced by LM_Studio before the Chatbot processes it.
- **Portfolio_Analyzer**: The component (UI page and backing logic) that displays portfolio performance, P&L, and risk metrics for a set of positions.
- **Portfolio_Import**: The process of loading positions into the Portfolio_Analyzer from an existing paper trading account.
- **Manual_Portfolio_Entry**: The process of a user directly entering position data (symbol, quantity, entry price, asset type) into the Portfolio_Analyzer via a form.

---

## Requirements

### Requirement 1: Fundamental Analysis Integration in Model Learning

**User Story:** As a trader, I want the system to incorporate fundamental analysis data alongside technical analysis in its model learning and decision-making, so that signals are grounded in both price action and business fundamentals.

#### Acceptance Criteria

1. THE Fundamental_Analyzer SHALL fetch at least the following metrics for equity symbols: trailing P/E ratio, forward P/E ratio, EPS (trailing twelve months), revenue growth (year-over-year), profit margin, and debt-to-equity ratio.
2. WHEN a symbol analysis is requested, THE Chatbot SHALL run both Technical_Analyzer and Fundamental_Analyzer concurrently and combine their outputs into a single composite signal.
3. THE Chatbot SHALL apply a default Analysis_Weight of 0.60 to fundamental signals and 0.40 to technical signals when computing the composite score for equity symbols.
4. WHEN the query context indicates a short-term or intraday intent (e.g., `TimeframeScope.SCALPING` or `TimeframeScope.INTRADAY`), THE Chatbot SHALL reduce the fundamental Analysis_Weight to 0.20 and increase the technical Analysis_Weight to 0.80.
5. WHEN the query context indicates a long-term or investment intent (e.g., `TimeframeScope.INVESTMENT`), THE Chatbot SHALL apply a fundamental Analysis_Weight of 0.75 and a technical Analysis_Weight of 0.25.
6. IF fundamental data is unavailable for a symbol (e.g., non-equity assets such as FX, futures, or crypto), THEN THE Fundamental_Analyzer SHALL return a null result and THE Chatbot SHALL fall back to a 100% technical Analysis_Weight for that symbol.
7. THE Chatbot SHALL include a "Fundamental Snapshot" section in its response text whenever fundamental data is available, displaying at minimum: P/E ratio, EPS, revenue growth, and a qualitative assessment (e.g., "Undervalued", "Fairly Valued", "Overvalued").
8. THE Fundamental_Analyzer SHALL cache fetched fundamental data per symbol with a time-to-live of 3600 seconds to avoid redundant API calls.

---

### Requirement 2: Per-Asset Sentiment Analysis During Symbol Analysis

**User Story:** As a trader, I want the system to perform sentiment analysis specifically around the asset I am querying, so that I understand the current market mood and news narrative for that symbol.

#### Acceptance Criteria

1. WHEN a symbol analysis is requested, THE Sentiment_Analyzer SHALL fetch and score news articles, social posts, and RSS feed items that explicitly mention the queried symbol.
2. THE Sentiment_Analyzer SHALL produce a sentiment score in the range [-1.0, 1.0] and a categorical label: `VERY_BEARISH`, `BEARISH`, `NEUTRAL`, `BULLISH`, or `VERY_BULLISH`.
3. THE Chatbot SHALL include a "Sentiment Overview" section in its response text for every symbol analysis, displaying: the sentiment score, the categorical label, the number of articles analyzed, and the top 3 most impactful headlines.
4. WHEN the sentiment score diverges from the technical signal direction by more than 0.4 (e.g., technical is BULLISH but sentiment is BEARISH), THE Chatbot SHALL flag a "Sentiment Divergence" warning in the response.
5. THE Sentiment_Analyzer SHALL complete its analysis within 10 seconds per symbol; IF the timeout is exceeded, THEN THE Chatbot SHALL proceed with the response and note that sentiment data is unavailable.
6. THE Sentiment_Analyzer SHALL weight news sources according to the existing `SourceCredibilityEngine` credibility scores when computing the aggregate sentiment score.

---

### Requirement 3: Fix Options Chain Missing `vol` Column Error

**User Story:** As a trader, I want the options chain to load without errors, so that I can view and analyze options data without the application crashing.

#### Acceptance Criteria

1. WHEN the options chain is fetched for any symbol, THE Options_Chain pipeline SHALL check for the presence of a `vol` (volume) column before attempting to access it.
2. IF the `vol` column is absent from the raw options chain DataFrame, THEN THE Options_Chain pipeline SHALL substitute a column of zeros and log a warning message of the form `"vol column missing for {symbol}; defaulting to 0"`.
3. THE Options_Chain pipeline SHALL also validate the presence of the following columns before use: `strike`, `lastPrice`, `bid`, `ask`, `impliedVolatility`, `openInterest`; IF any are missing, THEN THE Options_Chain pipeline SHALL substitute sensible defaults (0 for numeric columns) rather than raising a `KeyError`.
4. WHEN the options chain is successfully fetched and normalized, THE Options_Chain pipeline SHALL return a DataFrame with a consistent schema regardless of which data provider supplied the raw data.
5. IF the options chain fetch fails entirely (network error or no data returned), THEN THE Options_Chain pipeline SHALL return an empty DataFrame and surface a user-readable error message: `"Options chain unavailable for {symbol}. Please try again later."`.

---

### Requirement 4: Enhanced Options Features in Strategy Research Lab, Simulation Hub, and Quant Portal

**User Story:** As an options trader, I want the Strategy Research Lab, Simulation Hub, and Quant Portal to provide deeper, more useful options-specific tools, so that I can research, simulate, and analyze options strategies with institutional-grade depth.

#### Acceptance Criteria

1. THE Strategy_Research_Lab SHALL provide an "Options Strategy Builder" tab that allows the user to construct multi-leg options strategies (covered call, protective put, bull call spread, bear put spread, iron condor, iron butterfly, straddle, strangle, calendar spread, diagonal spread) and display the resulting payoff diagram, breakeven prices, maximum profit, and maximum loss.
2. THE Strategy_Research_Lab SHALL display a live implied volatility (IV) rank and IV percentile for the selected underlying symbol, computed against a 252-trading-day lookback.
3. THE Strategy_Research_Lab SHALL render a volatility surface heatmap (strike vs. days-to-expiry) using the `OptionsEngine.generate_greek_surface` method.
4. THE Strategy_Research_Lab SHALL provide a "Greeks Dashboard" that displays Delta, Gamma, Theta, Vega, Rho, Vanna, and Charm for each leg of the constructed strategy, as well as net portfolio Greeks.
5. THE Simulation_Hub SHALL allow the user to simulate the P&L evolution of an options position over time using Monte Carlo price paths, displaying the distribution of outcomes at expiration.
6. THE Simulation_Hub SHALL display the probability of profit (PoP) for a simulated options strategy, computed from the Monte Carlo distribution.
7. THE Quant_Portal "Derivative Intelligence" tab SHALL display a live options chain table for a user-specified symbol, including columns: strike, expiry, type (call/put), bid, ask, last price, volume, open interest, IV, Delta, Gamma, Theta, and Vega.
8. THE Quant_Portal "Derivative Intelligence" tab SHALL provide an unusual options activity scanner that flags contracts where volume exceeds open interest by a factor of 2 or more, indicating potential institutional positioning.
9. WHEN a user selects a specific contract in the Quant_Portal options chain, THE Quant_Portal SHALL display a detailed Greeks breakdown and a payoff diagram for that single contract.
10. THE Quant_Portal SHALL provide a "Put/Call Ratio" metric for the selected underlying, computed from the fetched options chain data.

---

### Requirement 5: Fix Paper Trading Account Lookup Failure

**User Story:** As a trader, I want the paper trading system to correctly find my account, so that I can execute trades and view my portfolio without encountering "Account not found" errors.

#### Acceptance Criteria

1. WHEN `PaperTradingSystem.get_account` is called with a valid `account_id`, THE Paper_Trading_System SHALL return the corresponding `PaperTradingAccount` object.
2. WHEN `PaperTradingSystem.list_accounts` is called with a `user_id`, THE Paper_Trading_System SHALL return all accounts whose `user_id` column matches the provided value, using a case-insensitive comparison.
3. IF no accounts exist for a given `user_id`, THEN THE Paper_Trading_System SHALL return an empty list rather than raising an exception.
4. THE Paper_Trading_System SHALL ensure the `paper_trading_accounts` table is created with the correct schema before any account lookup is attempted, using `CREATE TABLE IF NOT EXISTS`.
5. WHEN the paper trading UI initializes, THE Paper_Trading_System SHALL use the `user_id` stored in `st.session_state` (defaulting to `"default_user"` if absent) consistently across all account creation, lookup, and listing operations.
6. IF a `user_id` of `"trader_1"` is used in any legacy data or test fixture, THEN THE Paper_Trading_System SHALL treat `"trader_1"` as equivalent to `"default_user"` for backward compatibility, OR the UI SHALL ensure it never passes `"trader_1"` as a `user_id` unless explicitly set by the user.
7. WHEN a database connection error occurs during account lookup, THE Paper_Trading_System SHALL log the error and return `None` (for single-account lookups) or an empty list (for multi-account lookups) rather than propagating the exception to the UI.

---

### Requirement 6: LM Studio as First-Pass Interpreter for AI Chatbot

**User Story:** As a trader, I want the AI chatbot to use LM Studio's response as the primary interpretation layer before generating its own analysis, so that the chatbot's output is directly correlated to what I actually asked and builds meaningfully on the LM Studio interpretation.

#### Acceptance Criteria

1. WHEN a user submits a query to the Chatbot, THE Chatbot SHALL call LM_Studio first, before any symbol extraction, intent detection, or market data fetching.
2. THE Chatbot SHALL send LM_Studio a structured prompt that instructs it to: (a) identify the user's true financial intent, (b) extract any mentioned symbols or asset classes, (c) identify the implied timeframe, and (d) suggest the most relevant analysis type (technical, fundamental, sentiment, macro, options, etc.).
3. THE Chatbot SHALL parse the LM_Studio response to extract: `intent`, `symbols`, `timeframe`, and `analysis_type` fields; IF LM_Studio is unavailable or returns an empty response, THEN THE Chatbot SHALL fall back to its existing regex-based intent detection without degrading functionality.
4. WHEN LM_Studio returns a valid Intent_Context, THE Chatbot SHALL use the extracted `symbols` and `intent` from LM_Studio as the primary inputs to its analysis pipeline, supplementing (not replacing) them with its own regex-based extraction.
5. THE Chatbot SHALL incorporate the full LM_Studio narrative response into its final output as a "LM Studio Interpretation" section, displayed before the Chatbot's own analysis sections.
6. THE Chatbot SHALL use the LM_Studio interpretation to elaborate and build upon the user's query — specifically, the Chatbot's analysis SHALL address the intent identified by LM_Studio rather than a literal keyword match of the raw query.
7. WHEN LM_Studio identifies an intent that differs from the Chatbot's own regex-based intent detection, THE Chatbot SHALL prefer the LM_Studio intent, provided LM_Studio's response is non-empty and parseable.
8. THE Chatbot SHALL complete the LM_Studio call within 60 seconds; IF the call times out, THEN THE Chatbot SHALL log a warning and proceed with its own intent detection.
9. THE Chatbot SHALL check LM_Studio connectivity using `check_llm_connectivity()` at startup; IF LM_Studio is not reachable, THEN THE Chatbot SHALL display a non-blocking status indicator in the UI noting "LM Studio offline — using built-in analysis engine".
10. FOR ALL valid user queries, the LM_Studio interpretation step SHALL occur before any market data is fetched, ensuring the data fetching is scoped to the correct symbols and analysis type identified by LM_Studio.

---

### Requirement 7: Fix `datetime` Not Defined Error in Portfolio Analyzer

**User Story:** As a trader, I want the portfolio analyzer to run without crashing, so that I can view my historical portfolio performance without encountering a `NameError`.

#### Acceptance Criteria

1. WHEN the Portfolio_Analyzer renders any view that references date or time values, THE Portfolio_Analyzer SHALL import `datetime` from the Python standard library at the top of the module.
2. THE Portfolio_Analyzer SHALL not use bare `datetime` references without the proper import; all date construction calls (e.g., `datetime.now()`, `datetime.strptime()`) SHALL resolve without a `NameError`.
3. WHEN the Portfolio_Analyzer encounters any other missing-import error at render time, THE Portfolio_Analyzer SHALL catch the exception, log it, and display a user-readable error message rather than a raw traceback.
4. THE fix SHALL be applied to all files in the portfolio analyzer module that reference `datetime` without importing it.

---

### Requirement 8: Portfolio Import and Manual Entry in Portfolio Analyzer

**User Story:** As a trader, I want to be able to load my portfolio into the analyzer either by importing it from my paper trading account or by manually entering positions, so that I can analyze any portfolio — simulated or real — without being locked into a single data source.

#### Acceptance Criteria

1. THE Portfolio_Analyzer SHALL provide two portfolio input modes: "Import from Paper Trading Account" and "Manual Entry", selectable via a toggle or tab control in the UI.
2. WHEN the user selects "Import from Paper Trading Account", THE Portfolio_Analyzer SHALL list all paper trading accounts available for the current `user_id` and allow the user to select one; upon selection, THE Portfolio_Analyzer SHALL load all open positions from that account into the analysis view.
3. WHEN the user selects "Manual Entry", THE Portfolio_Analyzer SHALL display a form that accepts the following fields per position: symbol (text), quantity (number), entry price (number), current price (optional number, defaults to live market price if omitted), and asset type (equity, option, crypto, FX).
4. THE Manual Entry form SHALL allow the user to add multiple positions before submitting, and SHALL provide a row-delete control to remove individual entries.
5. WHEN the user submits a Manual Entry portfolio, THE Portfolio_Analyzer SHALL compute and display: total portfolio value, unrealized P&L per position and in aggregate, percentage return per position and overall, and a position weight breakdown.
6. WHEN positions are imported from a paper trading account, THE Portfolio_Analyzer SHALL display the same metrics as Manual Entry (total value, unrealized P&L, percentage return, position weights).
7. THE Portfolio_Analyzer SHALL allow the user to export the currently loaded portfolio (whether imported or manually entered) as a CSV file.
8. IF the paper trading account import fails (e.g., no accounts found, database error), THE Portfolio_Analyzer SHALL display a clear error message and fall back to the Manual Entry mode without crashing.
9. WHEN a live market price cannot be fetched for a manually entered position, THE Portfolio_Analyzer SHALL use the user-supplied current price if provided, or display "Price unavailable" and exclude that position from aggregate P&L calculations.
