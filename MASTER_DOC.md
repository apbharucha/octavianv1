# OCTAVIAN TERMINAL — MASTER DOC

> **The complete, authoritative reference for the Octavian Terminal codebase.**
> Every module, every feature, every process — explained end to end.
>
> - Version covered: **v4.0.0** (streamlit entry: `main.py`)
> - Last updated: **2026-08-21**
> - Update policy: see [How to Keep This Document Current](#how-to-keep-this-document-current)

---

## Table of Contents

1. [How to Keep This Document Current](#how-to-keep-this-document-current)
2. [Platform Overview](#platform-overview)
3. [System Architecture](#system-architecture)
4. [Tech Stack & Dependencies](#tech-stack--dependencies)
5. [How to Run the Platform](#how-to-run-the-platform)
6. [Configuration & Secrets](#configuration--secrets)
7. [Data Layer](#data-layer)
   - [Market Data Sources (`data_sources.py`)](#market-data-sources-data_sourcespy)
   - [Ticker Universe (`ticker_universe.py`)](#ticker-universe-ticker_universepy)
   - [Comprehensive Ticker Universe](#comprehensive-ticker-universe)
   - [Databases & Persistence](#databases--persistence)
   - [Caching Strategy](#caching-strategy)
   - [Data Corrections Registry](#data-corrections-registry)
   - [Rate Limiting (`api_rate_limiter.py`)](#rate-limiting-api_rate_limiterpy)
8. [Application Entry Point (`main.py`)](#application-entry-point-mainpy)
   - [Startup Sequence](#startup-sequence)
   - [Sidebar Navigation](#sidebar-navigation)
   - [Background Task Plumbing](#background-task-plumbing)
   - [Page Router](#page-router)
9. [UI Foundation](#ui-foundation)
   - [Theme System (`octavian_theme.py`)](#theme-system-octavian_themepy)
   - [Background Task Manager (`background_tasks.py`)](#background-task-manager-background_taskspy)
10. [AI & NLP Layer](#ai--nlp-layer)
    - [Financial LLM Engine (`financial_llm_engine.py`)](#financial-llm-engine-financial_llm_enginepy)
    - [Tool Router (`tool_router.py`)](#tool-router-tool_routerpy)
    - [Deep-Dive Chart Engine (`deep_dive_charts.py`)](#deep-dive-chart-engine-deep_dive_chartspy)
    - [AI Chatbot (`ai_chatbot.py`)](#ai-chatbot-ai_chatbotpy)
    - [Intent Detection (`intent_detection_engine.py`)](#intent-detection-intent_detection_enginepy)
    - [Response Formatter (`response_formatter.py`)](#response-formatter-response_formatterpy)
    - [ML Analysis (`ml_analysis.py`)](#ml-analysis-ml_analysispy)
    - [Advanced ML Engine (`advanced_ml_engine.py`)](#advanced-ml-engine-advanced_ml_enginepy)
    - [Quant Ensemble Model (`quant_ensemble_model.py`)](#quant-ensemble-model-quant_ensemble_modelpy)
    - [Adaptive Reasoning (`adaptive_reasoning_engine.py`)](#adaptive-reasoning-adaptive_reasoning_enginepy)
    - [Personalization (`personalization_engine.py`)](#personalization-personalization_enginepy)
    - [Portfolio Chatbot Context (`portfolio_chatbot_context.py`)](#portfolio-chatbot-context-portfolio_chatbot_contextpy)
    - [Chatbot Evaluation Harness (`chatbot_eval/`)](#chatbot-evaluation-harness-chatbot_eval)
11. [News & Sentiment Layer](#news--sentiment-layer)
    - [News Analysis Engine (`news_analysis_engine.py`)](#news-analysis-engine-news_analysis_enginepy)
    - [Advanced News Processor (`advanced_news_processor.py`)](#advanced-news-processor-advanced_news_processorpy)
    - [Source Credibility (`source_credibility_engine.py`)](#source-credibility-source_credibility_enginepy)
    - [News Dashboard (`news_dashboard.py`)](#news-dashboard-news_dashboardpy)
    - [Narrative Generator (`narrative_generator.py`)](#narrative-generator-narrative_generatorpy)
12. [Market Analytics Engines](#market-analytics-engines)
    - [Unbiased Market Analyzer (`unbiased_market_analyzer.py`)](#unbiased-market-analyzer-unbiased_market_analyzerpy)
    - [Multi-Asset Analyzer (`multi_asset_analyzer.py`)](#multi-asset-analyzer-multi_asset_analyzerpy)
    - [Market Scanner (`market_scanner.py`)](#market-scanner-market_scannerpy)
    - [Market Movers (`market_movers.py`)](#market-movers-market_moverspy)
    - [Cross-Asset Transmission (`cross_asset_transmission.py`)](#cross-asset-transmission-cross_asset_transmissionpy)
    - [Macro Cross-Asset (`macro_cross_asset_engine.py`)](#macro-cross-asset-macro_cross_asset_enginepy)
    - [Daily Intelligence (`daily_intelligence.py`)](#daily-intelligence-daily_intelligencepy)
    - [Market Heartbeat (`market_heartbeat_system.py`)](#market-heartbeat-market_heartbeat_systempy)
    - [Regime Detection (`regime.py`)](#regime-detection-regimepy)
    - [Timeframe Analysis (`timeframe_analysis_engine.py`)](#timeframe-analysis-timeframe_analysis_enginepy)
    - [Counter-Trend Analyzer (`counter_trend_analyzer.py`)](#counter-trend-analyzer-counter_trend_analyzerpy)
    - [Factor Crowding (`factor_crowding_engine.py`)](#factor-crowding-factor_crowding_enginepy)
    - [Narrative Dislocation (`narrative_dislocation_engine.py`)](#narrative-dislocation-narrative_dislocation_enginepy)
    - [Target Probability (`target_probability_engine.py`)](#target-probability-target_probability_enginepy)
    - [Market Consensus (`market_consensus_engine.py`)](#market-consensus-market_consensus_enginepy)
    - [Cross-Sector Analyzer (`cross_sector_analyzer.py`)](#cross-sector-analyzer-cross_sector_analyzerpy)
    - [Sector Scanner (`sector_scanner.py`)](#sector-scanner-sector_scannerpy)
    - [FX Scanner (`fx_scanner.py`)](#fx-scanner-fx_scannerpy)
    - [Chart Image Analyzer (`chart_image_analyzer.py`)](#chart-image-analyzer-chart_image_analyzerpy)
    - [Graph Analysis (`graph_analysis.py`)](#graph-analysis-graph_analysispy)
13. [Quant Research & Strategy Lab](#quant-research--strategy-lab)
    - [Quant Portal (`quant_portal.py`)](#quant-portal-quant_portalpy)
    - [Quant Modeling Lab (`quant_modeling_lab.py`)](#quant-modeling-lab-quant_modeling_labpy)
    - [Strategy Research Lab (`strategy_research_lab.py`)](#strategy-research-lab-strategy_research_labpy)
    - [Genetic Strategy Engine (`genetic_strategy_engine.py`)](#genetic-strategy-engine-genetic_strategy_enginepy)
    - [Advanced Backtester (`advanced_backtester.py`)](#advanced-backtester-advanced_backtesterpy)
    - [Legacy Backtest (`backtest.py`)](#legacy-backtest-backtestpy)
14. [Algorithm Builder](#algorithm-builder)
    - [Engine (`algorithm_builder_engine.py`)](#engine-algorithm_builder_enginepy)
    - [UI (`algorithm_builder_ui.py`)](#ui-algorithm_builder_uipy)
15. [Trading Infrastructure](#trading-infrastructure)
    - [Paper Trading System (`paper_trading_system.py`)](#paper-trading-system-paper_trading_systempy)
    - [Paper Trading UI (`paper_trading_ui.py`)](#paper-trading-ui-paper_trading_uipy)
    - [Automated Trading Engine (`automated_trading_engine.py`)](#automated-trading-engine-automated_trading_enginepy)
    - [Risk Engine (`risk_engine.py`)](#risk-engine-risk_enginepy)
    - [Position Optimizer (`position_optimizer_engine.py`)](#position-optimizer-position_optimizer_enginepy)
    - [Portfolio Analyzer (`portfolio_analyzer.py`, `portfolio_analyzer_engine.py`)](#portfolio-analyzer)
    - [Trading System Subpackage (`trading_system/`)](#trading-system-subpackage-trading_system)
    - [Brokerage Engine (`brokerage_engine.py`)](#brokerage-engine-brokerage_enginepy)
    - [Breaking Trades Generator (`breaking_trades_generator.py`)](#breaking-trades-generator-breaking_trades_generatorpy)
    - [Trade Signal Overlay (`trade_signal_overlay.py`)](#trade-signal-overlay-trade_signal_overlaypy)
16. [Simulation Hub](#simulation-hub)
    - [Market Simulation Engine (`market_simulation_engine.py`)](#market-simulation-engine-market_simulation_enginepy)
    - [Simulation Universe (`market_simulation_universe.py`)](#simulation-universe-market_simulation_universepy)
    - [Simulation Viewer (`simulation_viewer.py`)](#simulation-viewer-simulation_viewerpy)
    - [Simulation Graders](#simulation-graders)
17. [Financial Modeling Suite](#financial-modeling-suite)
    - [Financial Model Generator (`financial_model_generator.py`)](#financial-model-generator-financial_model_generatorpy)
    - [Financial Model Generator UI](#financial-model-generator-ui)
    - [DCF / Valuation Engine](#dcf--valuation-engine)
    - [Comps Engine (`comps_engine.py`)](#comps-engine-comps_enginepy)
    - [LBO Engine (`lbo_model_engine.py`)](#lbo-engine-lbo_model_enginepy)
    - [M&A Model Engine (`mna_model_engine.py`)](#ma-model-engine-mna_model_enginepy)
    - [IPO Engine (`ipo_engine.py`)](#ipo-engine-ipo_enginepy)
    - [Precedent Transactions (`precedent_transactions_engine.py`)](#precedent-transactions-precedent_transactions_enginepy)
    - [Valuation Bridge (`valuation_bridge.py`)](#valuation-bridge-valuation_bridgepy)
    - [Investment Memo (`investment_memo.py`)](#investment-memo-investment_memopy)
    - [IB Excel Engine (`ib_excel_engine.py`)](#ib-excel-engine-ib_excel_enginepy)
    - [Spreadsheet Generator (`spreadsheet_generator.py`)](#spreadsheet-generator-spreadsheet_generatorpy)
    - [Presentation Generator (`presentation_generator.py`)](#presentation-generator-presentation_generatorpy)
    - [Model Audit (`model_audit.py`)](#model-audit-model_auditpy)
18. [Derivatives & Commodities](#derivatives--commodities)
    - [Options Engine (`options_engine.py`)](#options-engine-options_enginepy)
    - [Options Simulation Grader (`options_simulation_grader.py`)](#options-simulation-grader)
    - [Futures Engine (`futures_engine.py`)](#futures-engine-futures_enginepy)
    - [Futures Simulation Grader (`futures_simulation_grader.py`)](#futures-simulation-grader)
    - [Commodities Engine (`commodities_engine.py`)](#commodities-engine-commodities_enginepy)
19. [Institutional & Alternative Data](#institutional--alternative-data)
    - [SEC 13F Engine (`sec_13f_engine.py`)](#sec-13f-engine-sec_13f_enginepy)
    - [Institutional 13F UI (`institutional_13f_ui.py`)](#institutional-13f-ui-institutional_13f_uipy)
    - [Dark Pool Engine (`dark_pool_engine.py`)](#dark-pool-engine-dark_pool_enginepy)
    - [Dark Pool UI (`dark_pool_ui.py`)](#dark-pool-ui-dark_pool_uipy)
    - [Alternative Data Engine (`alternative_data_engine.py`)](#alternative-data-engine-alternative_data_enginepy)
    - [Institutional Analytics (`institutional_analytics_engine.py`)](#institutional-analytics-institutional_analytics_enginepy)
    - [Institutional Visualizations (`institutional_visualizations.py`)](#institutional-visualizations-institutional_visualizationspy)
    - [Discovery Engine (`octavian_discovery_engine.py`)](#discovery-engine-octavian_discovery_enginepy)
20. [Document & Chart Intelligence](#document--chart-intelligence)
    - [Document Analyzer (`document_analyzer.py`)](#document-analyzer-document_analyzerpy)
    - [Comparative Analysis (`comparative_analysis_engine.py`, `comparative_analysis_ui.py`)](#comparative-analysis)
    - [Fundamental Analyzer (`fundamental_analyzer.py`)](#fundamental-analyzer-fundamental_analyzerpy)
    - [Master Strategy (`master_strategy_engine.py`)](#master-strategy-master_strategy_enginepy)
    - [Strategy Intelligence (`strategy_intelligence_engine.py`)](#strategy-intelligence-strategy_intelligence_enginepy)
    - [Information Weighting (`information_weighting_engine.py`)](#information-weighting-information_weighting_enginepy)
21. [User & Personalization Layer](#user--personalization-layer)
    - [Trader Profile (`trader_profile.py`)](#trader-profile-trader_profilepy)
    - [Custom Dashboard (`custom_dashboard.py`)](#custom-dashboard-custom_dashboardpy)
    - [Watchlist Dashboard (`watchlist_dashboard.py`)](#watchlist-dashboard-watchlist_dashboardpy)
    - [Integrated Market System (`integrated_market_system.py`)](#integrated-market-system-integrated_market_systempy)
    - [Dashboard Intelligence (`dashboard_intelligence_engine.py`)](#dashboard-intelligence-dashboard_intelligence_enginepy)
    - [Auth Engine (`auth_engine.py`)](#auth-engine-auth_enginepy)
    - [Terms of Service (`terms_of_service.py`, `terms_of_service_ui.py`)](#terms-of-service)
    - [Notification Engine (`notification_engine.py`)](#notification-engine-notification_enginepy)
22. [Infrastructure & Services](#infrastructure--services)
    - [API Backend (`api_backend.py`)](#api-backend-api_backendpy)
    - [Real-Time Data Service (`realtime_data_service.py`)](#real-time-data-service-realtime_data_servicepy)
    - [WebSocket Engine (`websocket_engine.py`)](#websocket-engine-websocket_enginepy)
    - [Database Manager (`database_manager.py`)](#database-manager-database_managerpy)
    - [Analytics Dashboard (`analytics_dashboard.py`)](#analytics-dashboard-analytics_dashboardpy)
    - [Backend Package (`backend/`)](#backend-package-backend)
    - [Data Downloader (`data_downloader.py`)](#data-downloader-data_downloaderpy)
    - [Historical Data Engine (`historical_data_engine.py`)](#historical-data-engine-historical_data_enginepy)
    - [Market Data Cache (`market_data_cache.py`)](#market-data-cache-market_data_cachepy)
23. [Small Utility Modules (catch-all)](#small-utility-modules-catch-all)
24. [End-to-End Process Walkthroughs](#end-to-end-process-walkthroughs)
25. [Test Suite](#test-suite)
26. [Project Conventions & Gotchas](#project-conventions--gotchas)
27. [Changelog](#changelog)

---

## How to Keep This Document Current

This document is the **single source of truth** for the codebase. It must be
updated **whenever anything changes**:

1. **After every code change** (feature, fix, refactor), update the relevant
   section(s) of this document in the same working session, before the change
   is committed.
2. **When adding a new module or file**, add a new section (or subsection)
   describing: purpose, key classes/functions, data flow, and how it plugs
   into `main.py` navigation.
3. **When renaming / removing features**, edit the affected sections and
   update the **Changelog** entry at the end of the file.
4. **After test count changes**, update the Test Suite section.

The AGENTS.md workflow instructions reference this file so that future
sessions automatically keep it in sync with the code.

---

## Platform Overview

Octavian Terminal is a **Streamlit-based, institutional-grade multi-asset
financial intelligence platform**. It integrates:

- Real-time market data (Yahoo Finance / yfinance, OANDA FX, FINRA OTC,
  Alpha Vantage, Polygon, EODHD)
- A rule-based **financial LLM engine** (zero-API-key heuristic NLP) that
  answers natural-language market questions with provenance-tagged numbers
- A deep institutional **deep-dive memo** generator (20-section research memo
  with DCF, reverse DCF, scenario analysis, Monte Carlo, Bayesian updates,
  falsification, QC audit)
- A full **algorithm builder** that turns plain-language requests into
  backtested, deployable trading algorithms (10+ strategy archetypes,
  walk-forward CV, dynamic ensembles, per-trade reasoning)
- **Paper trading** with multiple accounts, positions, options, and an
  **automated trading engine** with configurable risk rules
- A **quant portal** (backtesting, factor models, alternative data),
  **strategy research lab** (pairs trading, genetic evolution, options
  strategies), **simulation hub** (Monte Carlo, market microstructure,
  crisis scenarios)
- Institutional tools: **SEC 13F tracking, dark pool intelligence,
  financial model generator** (DCF/LBO/M&A/Comps/IPO/Precedents), pitchbook
  **presentation generator**, **document analyzer** for SEC filings,
  **spreadsheet generator**, and more
- A professional dark navy/gold **theme** with no emojis, micro-animations,
  and glassmorphism cards

The whole app is a single Streamlit process routed from `main.py`; heavy
features run on **background worker threads** so the UI never blocks.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit App (main.py)                      │
│   sidebar nav → page router → feature render functions          │
└───────────────┬──────────────────────────────┬──────────────────┘
                │                              │
        ┌───────▼────────┐            ┌────────▼─────────┐
        │ Background     │            │  Feature modules │
        │ Task Manager   │            │  (engines + UI)  │
        │ (thread pool)  │            │  see TOC §9-§22  │
        └───────┬────────┘            └────────┬─────────┘
                │                              │
        ┌───────▼──────────────────────────────▼─────────┐
        │              Data & Service Layer              │
        │  data_sources.py   api_rate_limiter.py         │
        │  ticker_universe.py  database_manager.py       │
        │  historical_data_engine.py  websocket_engine   │
        └───────┬──────────────────────────────┬─────────┘
                │                              │
      ┌─────────▼─────────┐          ┌─────────▼─────────┐
      │ External vendors  │          │ Local persistence │
      │ yfinance (Yahoo)  │          │ *.db (SQLite)     │
      │ OANDA REST        │          │ *.json state      │
      │ FINRA OTC         │          │ data_cache/       │
      │ Alpha Vantage     │          │ __pycache__       │
      │ Polygon / EODHD   │          │                   │
      │ LM Studio (LLM)   │          │                   │
      └───────────────────┘          └───────────────────┘
```

Key architectural decisions:

- **Single Streamlit process, module-level singletons.** Streamlit reruns the
  script on every interaction, but all sessions share one Python process.
  Module-level state (singletons, caches, task registries) survives reruns and
  page changes — this is what lets background tasks keep running while the
  user navigates.
- **Engines are deterministic and offline-guarded.** Most analysis engines are
  pure functions of their inputs; live fetches are wrapped so that offline
  environments (tests, demos) fall back to synthetic or empty data.
- **Provenance-first data integrity.** The dark pool engine, deep-dive memo,
  and 13F engine tag every number with OBSERVED / MODELED / INFERENCE /
  REPORTED labels rather than fabricating data.
- **Lazy imports in `main.py`.** Heavy modules are imported inside the page
  branches, not at the top, keeping startup fast.

---

## Tech Stack & Dependencies

From `requirements.txt` (pinned):

| Area | Libraries |
|---|---|
| UI | `streamlit==1.53.1`, `plotly==6.5.2`, `altair`, `pydeck` |
| Data | `yfinance==1.1.0`, `requests`, `feedparser`, `beautifulsoup4`, `lxml` |
| ML | `torch==2.9.1`, `scikit-learn==1.8.0`, `xgboost==3.1.3`, `lightgbm==4.6.0`, `nltk`, `textblob`, `ta` |
| LLM | `llama_cpp_python==0.3.16`, `huggingface_hub`, `transformers`-free heuristic engine |
| Modeling | `numpy`, `pandas`, `scipy`, `sympy`, `openpyxl`, `xlsxwriter`, `python-pptx` |
| Backend | `bcrypt`, `peewee`, `schedule`, `tenacity`, `diskcache`, `websockets` |
| Testing | `pytest==9.0.3`, `hypothesis`, `pytest-asyncio` |

Python 3.12 is the reference interpreter (`.venv` present). The platform runs
entirely offline-capable for analysis (vendor keys are all optional; the app
degrades gracefully).

---

## How to Run the Platform

### Start the app

```bash
./start_streamlit.sh            # recommended: kills old processes, starts fresh
# or
streamlit run main.py --server.headless true
```

`start_streamlit.sh`:
1. Kills any running Streamlit (`pkill -9 -f "streamlit run"`).
2. Preserves caches by default; pass `--clean` to wipe `__pycache__`,
   `.streamlit/cache`, and `.pyc` files (forces a cold re-fetch — slower).
3. Verifies key modules import (`paper_trading_system`, `risk_manager`,
   `simulation_grader`, `breaking_trades_generator`).
4. Starts `streamlit run main.py --server.headless true`.

### Alternative entry points

| Entry | Purpose |
|---|---|
| `python api_backend.py` | Flask REST API backend (port 5000 default) |
| `python realtime_data_service.py` | Standalone real-time data service |
| `python -m chatbot_eval.run_eval` | Chatbot evaluation harness CLI |
| `python llm_stress_test.py` | LLM stress testing |
| `python data_downloader.py` | Download historical datasets |

### Running tests

```bash
python3 -m pytest tests/ -q          # full suite (758 tests)
```

The suite currently has **730 passing tests** covering engines, UI walkthroughs
(Streamlit `AppTest`), integration flows, and the deep-dive memo integrity.

---

## Configuration & Secrets

`config.py` is the central runtime configuration module:

- `_get_secret(name, default)` reads **environment variables first**, then
  **Streamlit secrets** (`st.secrets`). Never hardcodes keys.
- Vendor keys: `POLYGON_API_KEY`, `ALPHA_VANTAGE_KEY`, `OANDA_API_KEY`,
  `MASSIVE_API_KEY`, `EODHD_API_KEY`.
- `FLASK_SECRET_KEY`: from `SECRET_KEY` env var or a generated ephemeral key
  (sessions reset on restart if unset).
- `CORS_ORIGINS`: comma-separated allow-list (defaults to local Streamlit
  origins `http://localhost:8501,http://127.0.0.1:8501`).
- `API_BACKEND_KEY`: optional shared bearer token for the Flask backend's
  protected endpoints.
- `OANDA_BASE_URL`: defaults to `https://api-fxpractice.oanda.com` (practice
  account).

`.env.example` documents all optional variables; copy to `.env` and fill in.
All keys optional — the platform falls back gracefully.

---

## Data Layer

### Market Data Sources (`data_sources.py`)

The central data-fetching module (1,213 lines). All other modules should go
through it rather than calling yfinance directly.

**Data corrections registry** — known-bad data points are managed as data in
`data_corrections.json` (not hardcoded), so every correction is attributable,
reviewable, and removable without a code change. `_apply_data_corrections()`
overwrites bad OHLC rows and adjusts the previous close so daily changes
compute correctly.

**Thread safety** — yfinance's module-level `download()` writes into shared
module globals, so concurrent calls from multiple threads can cross-
contaminate results. All network fetches are serialized behind a reentrant
lock (`_YF_FETCH_LOCK`) so parallel callers (watchlist `ThreadPoolExecutor`,
scanners) never mix instruments.

**Cooldown** — if Yahoo starts returning invalid JSON, fetches are skipped
for a short cooldown window (`_YF_SKIP_UNTIL_TS`).

**Key functions** (as exposed to the rest of the app):

| Function | Purpose |
|---|---|
| `get_stock(symbol, period, interval)` | Historical OHLCV DataFrame, normalized |
| `get_realtime_quote(symbol)` | Fresh quote dict (price, prev_close, change, verified) |
| `get_fresh_quote(symbol)` | Same with freshness guarantee |
| `get_realtime_price(symbol)` | `(price, prev_close)` tuple |
| `get_realtime_prices_batch(symbols)` | Batched quotes for many symbols |
| `get_futures_data(symbol)` | Futures history (=F suffixes) |
| `get_fx_data(pair)` | FX pair history (X suffixes) |
| `get_crypto_data(symbol)` | Crypto history (-USD suffixes) |
| `get_options_chain(symbol)` | Options chains from yfinance |
| `get_company_info(symbol)` | Fundamentals snapshot |
| `search_symbols(query)` | Ticker search |
| `_normalize_ohlc()` | Handles yfinance MultiIndex columns, drops ticker level |
| `get_realtime_prices_batch` | Shared-cache batched quotes used by dashboards |

`_normalize_ohlc` handles yfinance's MultiIndex column shapes (both
`('Close','AAPL')` and `('AAPL','Close')` orientations) and flattens them to a
single-level OHLCV frame.

### Ticker Universe (`ticker_universe.py`)

The **single source of truth** for all tickers, sectors, aliases, and universe
helpers (2,984 lines). Every other module should import from here instead of
maintaining its own lists.

- `TickerUniverse` class (line 2466): curated symbol lists, sector maps,
  aliases, asset-type classification, and universe accessors.
- `_fetch_etf_holdings()` — pulls ETF constituent lists (up to 100 holdings).
- `_fetch_sp500_tickers()`, `_fetch_russell2000_sample()`,
  `_fetch_nasdaq100_tickers()` — index constituent fetchers.
- `_validate_ticker(sym)` — sanity check before use.
- `get_ticker_universe()` — singleton accessor.

Used by scanners, the algorithm builder's universe selection, quick-select
asset buttons in the quant portal, and market movers.

### Comprehensive Ticker Universe

`comprehensive_ticker_universe.py` — a 10,000+ symbol universe covering US
equities (all exchanges), ETFs, international, FX, futures, and crypto. All
symbols are yfinance-compatible. `ComprehensiveTickerUniverse` class provides
filtering by asset class, sector, and market cap tier.

### Databases & Persistence

| File | DB | Contents |
|---|---|---|
| `market_ai_system.db` | SQLite | Conversations, analytics, ML predictions, news articles (via `database_manager.py`) |
| `octavian_simulations.db` (+ dated archives) | SQLite | Simulations, backtests |
| `octavian_users.db` | SQLite | User accounts (via `db_manager.py`, bcrypt-hashed passwords) |
| `octavian_paper_trading_state.json` | JSON | Paper trading fallback state |
| `octavian_paper_trading_state_default.json` | JSON | Default state template |
| `dark_pool_state.json` | JSON | Dark pool settings + watchlists + alerts (gitignored) |
| `octavian_learned_params.json` | JSON | Simulation learning engine learned params |
| `manual_portfolios.json` | JSON | User-defined manual portfolios |
| `data_corrections.json` | JSON | Data corrections registry |
| `.edgar_filings_cache.json` | JSON | SEC 13F EDGAR response cache |
| `ticker_universe_cache.json` | JSON | Cached universe fetches |
| `data_cache/` | dir | Diskcache-backed cache storage |
| `market_ai_system.db` | SQLite | Main app DB (see `database_manager.py` section) |

### Caching Strategy

- **Streamlit caches**: `@st.cache_data(ttl=...)` for market data (e.g. index
  charts 5 min TTL, universe fetch TTL), `@st.cache_resource` for ML model
  singletons.
- **In-memory dict caches with TTL** in `data_sources.py` (`_CACHE`), news
  processors, and engine-level LRUs.
- **Diskcache** (`diskcache==5.6.3`) used by `market_data_cache.py`.
- **Module-level singletons** survive Streamlit reruns (engine instances, the
  background-task registry, the rate limiter).

### Data Corrections Registry

`data_corrections.json` — a JSON list of `{symbol, date, ohlc{...},
prev_close}` entries. Loaded lazily by `data_sources._load_corrections()` and
applied to every fetched frame. Corrections overwrite the bad row's OHLC and
rewrite the prior row's close so percentage changes are consistent.

### Rate Limiting (`api_rate_limiter.py`)

Thread-safe rate limiter + request handler with:

- **Sliding-window rate limits** per service (Reddit 60/min, Yahoo RSS
  30/min, generic RSS 100/min, etc.)
- **Exponential backoff with jitter** (3 retries, 2^n-second delays) to
  prevent thundering herd
- **In-memory response caching with TTL** (5-10 min) to cut API calls
- **Service-specific User-Agent headers** (fixes Reddit 403 / Yahoo 429)
- **Thread-safe** via locks; decorator pattern for easy application

---

## Application Entry Point (`main.py`)

`main.py` (1,817 lines) is the Streamlit entry point and routing hub. It
implements the classic flow: **Auth Gate → Onboarding → Sidebar Navigation →
Content Router**, plus the background-task plumbing that keeps long features
responsive.

### Startup Sequence

1. `matplotlib.use("Agg")` — headless backend so charts never open windows.
2. Lightweight always-needed imports: `octavian_theme` (COLORS, apply_theme,
   render_header, section_header) and `trader_profile` (personalized
   dashboard, profile settings, trader selection).
3. Optional `data_sources` realtime helpers with `HAS_REALTIME` flag.
4. `background_tasks` plumbing: `drain_completed`, `get_task`, `reopen`,
   `submit_task`, `tasks_for_session`.
5. `st.set_page_config(layout="wide", page_title="Octavian Terminal",
   page_icon="O")` then `apply_theme()`.
6. Sidebar logo (`logo.png`, fallback to styled text), caption, navigation
   radio, `show_trader_selection(key_suffix="_sidebar")`, and the background
   status panel.
7. Background-task check + idle poller fragment.
8. Content router — one `if selection == ...` branch per page.

### Sidebar Navigation

28 entries (no emojis, per project convention):

```
Dashboard · Watchlist · Market Scanner · Symbol Analysis · Chart Analysis ·
Intelligence Center · Market Heartbeat · Dark Pool Intelligence ·
Institutional 13F & SEC Filings · Financial Model Generator · Target
Probability · Presentation Generator · Document Analyzer · Comparative
Analysis · Daily Briefing · Quant Portal · Quant Modeling Lab · Strategy
Research Lab · Algorithm Builder · Portfolio Analyzer · Position Optimizer ·
Paper Trading · Simulation Hub · Spreadsheet Generator · Trader Profile ·
Notification Settings · Settings & Analytics · Terms of Service
```

### Background Task Plumbing

- `_bg_session_id()` — the Streamlit session id (falls back to `"default"`
  in bare mode) via `st.runtime.scriptrunner.get_script_run_ctx()`.
- `_launch_background(name, result_key, fn, *args, **kwargs)` — submits a
  task to the module-level thread pool and stores the task id in
  `st.session_state["_bg_<result_key>"]`. Returns immediately; the user can
  navigate anywhere.
- `_check_background_tasks(force_rerun=False)` — drains completed tasks for
  this session, publishes `result` into `st.session_state[result_key]`, and
  toasts "<name> complete — results are ready." On delivery failure it
  re-queues the task via `reopen()` so the result is never lost. When
  `force_rerun=True` it calls `st.rerun(scope="app")` so the page showing the
  result updates immediately.
- `_bg_poller()` — a `@st.fragment(run_every=15)` idle poller that calls
  `_check_background_tasks(force_rerun=True)` every 15 seconds, so the user
  is notified the moment any background process finishes even with zero
  interaction.
- `_render_background_status()` — compact sidebar list of this session's
  tasks (running with elapsed seconds / done / failed), newest last.

### Page Router

Every page is imported lazily inside its branch (keeps startup fast). The
routing map:

| Nav label | Render function | Module |
|---|---|---|
| Dashboard | `show_personalized_dashboard()` + Market Overview + Breaking Trades tabs | `trader_profile`, `main` |
| Watchlist | `show_watchlist_dashboard()` | `watchlist_dashboard` |
| Market Scanner | `show_market_scanner()` | `market_scanner` |
| Symbol Analysis | `show_custom_dashboard()` | `custom_dashboard` |
| Chart Analysis | `show_chart_analyzer()` | `chart_image_analyzer` |
| Intelligence Center | tabs: News & Sentiment / Octavian AI Assistant / Counter-Trend Signals | `news_dashboard`, `ai_chatbot`, `counter_trend_analyzer` |
| Paper Trading | `show_paper_trading_dashboard()` | `paper_trading_ui` |
| Simulation Hub | `render_simulation_viewer()` | `simulation_viewer` |
| Spreadsheet Generator | `show_spreadsheet_generator()` | `spreadsheet_generator` |
| Trader Profile | `show_profile_settings()` | `trader_profile` |
| Market Heartbeat | `show_market_heartbeat_tab()` | `market_heartbeat_system` |
| Dark Pool Intelligence | `show_dark_pool_dashboard()` | `dark_pool_ui` |
| Financial Model Generator | `show_financial_generator()` | `financial_model_generator_ui` |
| Daily Briefing | engine-driven briefing UI | `daily_intelligence` |
| Quant Portal | `render_quant_portal()` | `quant_portal` |
| Strategy Research Lab | `render_strategy_research_lab()` | `strategy_research_lab` |
| Algorithm Builder | `render_algorithm_builder()` | `algorithm_builder_ui` |
| Settings & Analytics | `show_analytics_dashboard()` | `analytics_dashboard` |
| Institutional 13F & SEC Filings | `render_13f_analysis_tab()` | `institutional_13f_ui` |
| Target Probability | `show_target_probability()` | `target_probability_engine` |
| Presentation Generator | `show_presentation_generator()` | `presentation_generator` |
| Document Analyzer | `show_document_analyzer()` | `document_analyzer` |
| Comparative Analysis | `render_comparative_analysis()` | `comparative_analysis_ui` |
| Quant Modeling Lab | `render_quant_modeling_lab()` | `quant_modeling_lab` |
| Portfolio Analyzer | `show_portfolio_analyzer()` | `portfolio_analyzer` |
| Position Optimizer | `render_position_optimizer_ui()` | `position_optimizer_engine` |
| Notification Settings | `show_notification_settings()` | `notification_settings_ui` |
| Terms of Service | `show_terms_of_service()` | `terms_of_service` |

**Dashboard specifics** — the Dashboard page hosts three tabs: "My View"
(personalized via `show_personalized_dashboard`), "Market Overview" (live
index tickers in a `@st.fragment(run_every=30)` auto-refreshing strip using
batched quotes, plus a cached 5-minute index chart), and "Breaking Trades"
(high-confidence setups via the breaking trades generator).

**Intelligence Center specifics** — three tabs: News & Sentiment
(`news_dashboard`), Octavian AI Assistant (`show_octavian_chatbot`), and
Counter-Trend Signals (narrative divergence tracker table + active
counter-trend trade signals with direction/strength/time-horizon cards).

**Footer** — sidebar footer "v4.0.0 | Octavian AI by APB".

---

## UI Foundation

### Theme System (`octavian_theme.py`)

Dark-mode institutional theme (1,495 lines):

- **Color palette** (`COLORS`): navy `#0a1628` / navy_light `#132240`
  backgrounds, gold accent `#c9a84c`, lavender `#9b8ec4`, text `#e8eaf0`,
  success `#4caf50`, danger `#ef5350`.
- `apply_theme()` — injects global CSS: Inter font, letter-spaced headers,
  glassmorphism cards, CSS keyframe micro-animations (slide-in, pulse,
  shimmer), custom metric cards, section headers, gradient banners.
- `render_header(title, subtitle)` — page header component.
- `section_header(text)` — section divider.
- No emojis anywhere (project convention — swept from UI and docs).

### Background Task Manager (`background_tasks.py`)

Module-level singleton task registry that lets long-running features keep
running in worker threads while the user navigates freely.

**Why a module-level singleton?** Streamlit reruns the whole script on every
interaction, but all reruns share one Python process. Module-level state
survives reruns and page changes, so a task submitted on the Dashboard keeps
running while the user browses anywhere else.

**Guarantees:**

- Never blocks the UI — bounded thread pool (`_MAX_WORKERS = 4`).
- Concurrent but bounded — one shared pool for all tasks.
- Session isolation — tasks keyed by Streamlit session id.
- Exactly-once notification — `drain_completed` atomically marks a finished
  task "notified" so the main rerun and the idle poller can never
  double-notify.
- Bounded memory — registry keeps the most recent `_MAX_TASKS = 300`
  records, evicting oldest finished/errored/queued first (never evicts a
  genuinely executing task unless flooded).
- Thread safe — every mutation under one re-entrant lock `_LOCK`.

**API:**

| Function | Purpose |
|---|---|
| `submit_task(name, fn, session_id, result_key, *args, **kwargs)` | Run on worker thread, return task id immediately |
| `get_task(task_id)` | Copy of a task record |
| `tasks_for_session(session_id, limit)` | Recent tasks (payload stripped) |
| `running_tasks(session_id)` | Currently-running tasks, optional filter |
| `drain_completed(session_id)` | Atomically return finished-but-unnotified tasks (payload released) |
| `mark_notified(session_id, task_id)` | Idempotent notify flag |
| `reopen(session_id, task_id)` | Re-queue a finished task for delivery |
| `count()` | Records held (tests/diagnostics) |
| `shutdown(wait)` | Release the worker pool (used by tests) |

Task record fields: `task_id`, `name`, `session_id`, `status`
(running/done/error), `result`, `error`, `result_key`, `notified`,
`submitted_at`, `finished_at`. The worker `_run()` stamps `_EXECUTING` on
start (so cleanup can distinguish queued from executing), runs the function,
and records done/error under the lock; exceptions are caught so a worker
thread never dies silently.

---

## AI & NLP Layer

### Financial LLM Engine (`financial_llm_engine.py`)

The centerpiece: a **fully self-contained, zero-API-key heuristic NLP engine**
(5,663 lines) that performs multi-agent financial reasoning via rule-based
text generation. It is called directly by the AI chatbot in the Intelligence
Center, and by the chatbot evaluation harness.

**Entity resolution firewall (top of file).** Three layers of token protection
stop the "Live snapshot for CAPEX / FCF / GPU" class of hallucination:

1. `_SEMANTIC_CONCEPTS` — finance jargon that must NEVER be tickers even in
   ALL CAPS: CAPEX, FCF, DCF, EBITDA, EPS, WACC, RSI, MACD, GPU, CUDA, ASIC,
   LLM, TTM, YTD, Q1–Q4, FY24–FY28, CONSENSUS, and ~80 more.
2. `_AMBIGUOUS_TICKER_CONCEPTS` — words that are BOTH real tickers and
   concepts (AI = C3.ai, MOAT = VanEck MOAT ETF, LOW = Lowe's, KEY =
   KeyCorp, CASH = Pathward, DASH = DoorDash, COST, TGT, NET, …). These are
   kept as tickers ONLY when the query frames them as securities ("MOAT
   ETF", "buy NET"), via `_is_equity_reference()`.
3. `_REGION_CODES` (EU/US/UK/CN/JP/…) and `_COMMON_ENGLISH_WORDS` (~250
   short words) — hard-blocked or security-only resolution for prose tokens.

**LLM connectivity layer.** `check_llm_connectivity()` probes LM Studio at
`localhost:1234`; `_call_llm(prompt, system)` calls it with a timeout,
falling back to heuristic generation when offline. `query_parser_llm` and
`generate_analysis_llm` use it to (optionally) parse queries and write
analysis prose — but the deterministic heuristic path always works without
any LLM.

**Query decomposition.** `expand_query_intents(query)` (700+ lines) expands
free text into structured intents and entities using the ticker universe.
Helpers classify: `_is_equity_reference`, `_try_verb_split` (verb+object
parsing), `_is_event_query`, `_is_market_instrument`, `_infer_deep_dive_subject`,
`_is_deep_dive_query`, `_is_focused_event_question`, `_mega_label`,
`_decompose_mega_query` (splits compound multi-part questions into parts).

**Main entry: `generate_financial_analysis(query, context_data=None)`**
(1,300+ lines). Routing ladder:

| Query class | Builder | Output |
|---|---|---|
| Mega / multi-part | `_build_mega_response` | Sectioned multi-topic brief |
| Deep dive ("institutional deep dive on X") | `_build_institutional_deep_dive` | 20-section research memo (see below) |
| Geopolitical briefing | `_build_geopolitical_briefing` | Regime-affected asset analysis |
| Current events | `_build_current_events_briefing` | Event → market impact |
| Probability question | `_build_probability_answer` | Probabilistic forecast |
| Trade setup | `_build_trade_setup_response` | Entry/stop/targets with risk-reward |
| Hedging strategy | `_build_hedging_strategy` | Hedge construction |
| Sector scan | `_build_sector_scan_response` | Sector ETF survey |
| FX outlook | `_build_fx_outlook_response` | Currency analysis |
| Valuation | `_build_valuation_answer` | Multi-method valuation |
| Macro analysis | `_build_macro_analysis` | Macro regime briefing |
| Single ticker | `_build_single_answer` | Per-symbol analysis |

Every builder renders sections with provenance tags (OBSERVED DATA / MODEL
CALCULATION / MODEL ASSUMPTION / DATA UNAVAILABLE — NO ESTIMATE SUBSTITUTED)
and uses live data via `_fetch_live_data_for_tickers` / `_fetch_series_for`.

**Institutional deep-dive memo** (`_build_institutional_deep_dive`, ~350
lines assembler + ~1,100 lines of section builders). Built to a 20-section
research-integrity spec with these helpers:

| Helper | Section | Notes |
|---|---|---|
| `_dd_provenance_legend` | Legend | Every number tagged with provenance |
| `_dd_data_gate` | §2 Data completeness | Availability matrix + score that DRIVES confidence |
| `_dd_segment_reconstruction` | §3 Segments | Per-segment reconstruction from reported data |
| `_dd_sensitivity_table` | §3b | Gross-margin bps → EPS% + operating leverage |
| `_dd_ai_infra_economics` | §4 (mandatory) | Capex→ROI chain, cycle classification with probabilities; "not applicable" variant for non-AI names |
| `_dd_moat_matrix` | §5 | 9-factor competitive moat matrix + share-loss sensitivity |
| `_dd_real_dcf` | §6 | Full DCF mechanics through the site's own `InstitutionalDCFEngine`: revenue→EBIT→NOPAT→FCF→PV→terminal→EV→equity→per-share, WACC×g grid, DCF-vs-market reconciliation. **Gated**: BLOCKED without reported income statement; PARTIAL (FCF = NOPAT − ASSUMED reinvestment, disclosed) when cash-flow statement is DATA UNAVAILABLE |
| `_dd_reverse_dcf` | §7 | Multi-variable solve: implied revenue CAGR / EBIT margin / FCF growth via deterministic bisection on the same DCF equation; coarse Gordon grid secondary. Explicit note that implied FCF growth ≠ implied revenue growth |
| `_dd_expectation_gap` | §8 | Company / consensus / model / market-implied gaps |
| `_dd_macro_transmission` | §9 | Per-variable earnings × multiple impact table; every range labeled SCENARIO ASSUMPTION; +100bps-WACC fair-value impact is the one MODEL CALCULATION |
| `_dd_sentiment_engine` | §10 | Sentiment NEVER inferred from price momentum; unavailable sources say DATA UNAVAILABLE |
| `_dd_risk_matrix` | §11 | Full matrix with leading indicator / mitigation / priced-in |
| `_dd_catalyst_table` | §12 | Known / Probable / Speculative × timeline buckets |
| `_dd_scenario_table` | §13 | Bear/Base/Bull (5 scenarios), probs sum EXACTLY 100%; margins move with growth via operating leverage (0.2pp/pp, clipped ±6pp) |
| `_dd_monte_carlo` | §14 | Empirical CDF, P10–P90 monotonic, >10/20/30/50% loss probs, P(outperform SPX) |
| `_dd_bayesian` | §15 | 10 evidence events with LR vectors, posteriors renormalized to 100%; Evidence-basis column + calibration caveat (LRs are analyst-set MODEL ASSUMPTIONS, a decision framework, not statistics) |
| `_dd_info_advantage` | §16 | 8 variables with measurement/proxy/bullish/bearish/availability |
| `_dd_falsification` | §17 | 10 ranked arguments, confirm/falsify per argument, single rating-reverser |
| `_dd_confidence` | §18 | Computed from completeness + fundamentals + scenario dispersion; <70% always; fixed disclosed weights (DQ×30% + MR×25% + FC×25% + RC×20%), formula printed so the number is reproducible |
| `_dd_final_committee` | §19 | Rating, fair value, 12m/24m returns, drawdown probs (single-source with §13/§14), reasons for/against, key variables, what changes rating |
| `_dd_qc_audit` | §20 | 21 checks incl. numerical reconciliation (weighted return, weighted price, P(dd>30/50) recomputed, DCF independently recomputed to 1e-6); any failure → "AUDIT PARTIAL PASS — N QUANTITATIVE CHECK(S) FAILED"; QC8 deliberately FAILS without cash-flow data |
| `_dd_scenario_engine` | §13 core | **Single source of truth** for all scenario math: rounds probabilities (largest-remainder, sum = exactly 100%) and returns (whole percent) to DISPLAY precision BEFORE computing weighted return / drawdown probs, so every displayed number recomputes exactly from the displayed table |
| `_dd_rating` | §19 | Mechanically derived rating: base notch from expected return (≥+20% Strong Buy … else Strong Sell) then risk-adjusted (P(>30% dd) ≥20% or confidence <50% → −1 notch); framework text printed with the rating |

**Review-fix iteration (Aug 2026):**
- **Scenario math:** probabilities/returns are rounded to display precision *upstream* (`_dd_scenario_engine` + `_dd_round_scenarios`), so the displayed weighted return, drawdown probabilities, fair value and rating all recompute exactly from the printed table (review: hidden precision may never produce a displayed result; QC3b enforces ≤0.05pp tolerance).
- **Same-variable discipline:** the expectation-gap (§8) and "largest market-vs-model disagreement" (§19) compare the market-implied REVENUE CAGR (multi-variable reverse solve) against the model's REVENUE growth — never FCF growth vs revenue growth. QC16 enforces this.
- **DCF labeling:** the DCF now prints its EXACT reinvestment formulas (D&A = 6%×Revenue, CAPEX = 8%×Revenue, **ΔNWC = 5%×ΔRevenue — the CHANGE in revenue, not total revenue**); status is **CONDITIONAL / ASSUMPTION-BASED** (not "PARTIAL") when cash flow is unavailable; every DCF fair value is labeled ILLUSTRATIVE. QC21 verifies the status.
- **Valuation separation:** §19 splits valuation into (A) reported-data, (B) assumption-based DCF, (C) market-implied, (D) scenario — provenances are never merged.
- **Directional validation (round-2 hardened):** reasons-to-own/reasons-not-to-own are built by `_dd_build_reasons` and scanned by `_dd_semantic_violations` — a negative expected return OR a below-market DCF anchor can never appear under "reasons to own" (a below-market anchor is a negative implication and goes under reasons-not-to-own), and a positive implication can never appear under "reasons not to own". QC18 runs the REAL scan (no longer hardcoded) and a violation prints a SEMANTIC QC FAILURES block and flips the audit to FAIL.
- **Confidence:** mechanically computed from disclosed components with fixed weights (30/25/25/20); the formula and weights are printed. QC17 recomputes it independently.
- **Bayesian:** labeled **SUBJECTIVE BAYESIAN FRAMEWORK / Bayesian-Style Scenario Update** — likelihood ratios are analyst-set, never presented as calibrated probability. QC12.
- **Fake precision removed:** every drawdown/loss probability is "modeled P(...) within defined scenarios" with empirical probabilities stated DATA UNAVAILABLE (§13/§14/§19).
- **Macro gating:** `_build_macro_analysis(..., current_data_available=False)` suppresses all current-claims macro commentary ("The Fed remains…", "here's what matters right now") when no timestamped macro dataset is attached; only the transmission framework + DATA UNAVAILABLE note are emitted. QC19.
- **LLM system prompt:** carries the 13-point QUANTITATIVE INTEGRITY & ANTI-HALLUCINATION FIREWALL (reproducible numbers, same-variable comparisons, ASSUMPTION-BASED labels, no out-of-model probabilities, mechanically derived confidence/rating, SUBJECTIVE Bayesian, macro gating, A/B/C/D valuation separation).

Price stamps: header + §19 carry `quote_date` + source (yahoo 5d daily,
delayed EOD). Fundamentals come from `_dd_fetch_fundamentals` →
`fetch_ticker_fundamentals` (REPORTED data, offline-guarded by
OCTAVIAN_OFFLINE). `_is_financial_metric_token` is the second firewall layer
so metric tokens never become deep-dive subjects.

**Deep-dive chart engine (`deep_dive_charts.py`, ~430 lines)** — REVIEW FIX 1
(the 4-chart problem): every deep-dive thesis gets EXACTLY FOUR charts, each
serving a DISTINCT analytical job built from the memo's own numbers — never
generic price charts:

1. **expectation_gap** — current price vs DCF bear/base/bull, scenario-weighted
   value, market-implied requirement. When the DCF is CONDITIONAL
   (assumption-based, cash-flow statement unavailable), the chart prints a
   prominent banner "CONDITIONAL DCF — NOT VERIFIED FCF VALUATION", labels the
   bars "Cond. DCF Bear/Base/Bull" (never a bare "DCF Bear"), and the
   provenance says so — review round 2, issue 1.
2. **scenario_distribution** — five scenarios: probability, implied price,
   expected-return contribution.
3. **valuation_sensitivity** — WACC × terminal-growth heatmap on the base DCF,
   computed by the same engine as memo §6. Carries the same CONDITIONAL DCF
   banner when applicable.
4. **risk_reward** — metrics TABLE tied to the memo text (review round 2,
   issue 2): probability-weighted return, probability-weighted price, current
   price, upside/downside scenario ranges, weighted upside/downside
   contributions, and modeled P(>30%/>50% drawdown) each labeled "within
   defined scenarios (empirical: DATA UNAVAILABLE)". Replaces the old
   abstract weighted-downside/upside bars that did not obviously reconcile
   with the headline return.

Every chart carries full metadata: `chart_id` (run_id + purpose), `purpose`,
`dataset`, `data_timestamp`, `provenance`, `calculation`, `run_id`, a
`dataset_hash` (sha256 of its inputs), and a `qc_status` computed by
`_chart_qc` — the visualization layer inherits the memo's reconciliation rules
(review round 2: "extend the integrity firewall to the visualization layer"),
so a chart can never display numbers the memo's audit would reject. A
process-wide `_CHART_HASH_REGISTRY` flags `reused_dataset` when an identical
(purpose, dataset) was already rendered — the spec is always regenerated from
the current run's data, never a cached figure. Unsupported charts return a
DATA UNAVAILABLE state instead of fabricating or recycling one.

Wired into `ai_chatbot.process_enhanced_query`: deep-dive queries are detected
via `_is_deep_dive_query` and get the four analytical charts; all other
warranted queries keep the price-chart path.

**Other public entry points:**

- `generate_portfolio_advice(portfolio_data)` — portfolio-level strategy
  advice via `_generate_heuristic_portfolio_strategy`.
- `generate_risk_narrative_llm(portfolio_greeks, iv_regime)` — options-risk
  prose.
- `generate_market_briefing_llm(macro_data, volatility_data)` — daily
  briefing via `_build_data_grounded_briefing`.
- `_df_to_markdown` — pandas→markdown table renderer (avoids the optional
  `tabulate` dependency).

### Tool Router (`tool_router.py`)

The **tool router** is the layer that outsources explicit tool asks in user
chat prompts to Octavian's own analytical engines instead of letting the LLM
improvise an answer from memory. When a user asks the chatbot to "run a DCF",
"use the Bayesian network", "fit a Markov model", "check dark pool flow",
"look up 13F positioning", "correlation between X and Y", "options greeks",
"factor crowding" or "the market regime", the router detects the explicit
tool vocabulary, invokes the REAL engine that powers that feature, and
repackages the engine's computed output as the chatbot's answer (with a
provenance footer stating what was computed, by which engine, and on what
data basis).

**Design rules**

- Fires ONLY on explicit tool vocabulary — generic asks ("what do you think
  about NVDA", "hedge my XOM position", "what options strategy for LOW",
  plain filler "hmm") keep their existing specialist builders untouched.
- Every runner is fully defensive: any missing data or engine failure
  returns `None`, so the normal pipeline is never degraded.
- Multi-tool queries run ALL matched engines and combine their sections into
  one answer.
- Reverse-DCF asks are matched only by the Reverse DCF tool (a negative
  lookbehind keeps "reverse DCF" from also firing the plain DCF tool).

**Tool registry (`_TOOLS`)** — each entry has trigger patterns (regex),
`needs_symbol`, and a runner:

| Tool | Triggers | Runner → engine |
| --- | --- | --- |
| DCF | `dcf`, discounted cash flow, dcf model | `_run_dcf` → `financial_model_generator` `InstitutionalDCFEngine` (`compute_wacc`, `project_fcf`, `compute_terminal_value`) with reported fundamentals |
| Reverse DCF | reverse dcf, what growth does the price imply | `_run_reverse_dcf` → `market_consensus_engine.reverse_dcf_expectations` (implied growth/margin/FCF) |
| Bayesian network | bayesian network, bayes net, bayesian analysis | `_run_bayesian_network` → `institutional_analytics_engine.BayesianNetwork` (shock propagation through the 3-layer DAG) |
| Markov / HMM regime | markov, hidden markov, hmm model/regime, gaussian mixture, regime detection | `_run_markov` → `hmm_engine.get_regime_detector()` fit on the symbol's price history |
| Dark pool | dark pool, off-exchange | `_run_dark_pool` → `dark_pool_engine.analyze_ticker` + `ai_insight` |
| Institutional 13F | 13f, sec filings, institutional holdings, smart money | `_run_sec_13f` → `sec_13f_engine.SEC13FEngine.get_global_smart_money_flow` |
| Correlation | correlation, correlation matrix | `_run_correlation` → `risk_engine.correlation_matrix` (needs 2+ symbols) |
| Options greeks | greeks, black-scholes, option pricing, delta/gamma | `_run_options_greeks` → `options_engine.OptionsEngine.black_scholes` |
| Factor crowding | crowding, crowded trades, hedge fund overlap | `_run_factor_crowding` → `factor_crowding_engine` `detect_crowded_trades` + `simulate_hf_overlap` |
| Market regime | risk-on/off, market regime, volatility regime | `_run_market_regime` → `regime.get_regime_context` (VIX + tape) |

**Public API**

- `detect_tool_requests(query) -> List[str]` — names of every matched tool.
- `route_tool_query(query, tickers, sectors, live_data) -> Optional[str]` —
  run all matched tools and join their repackaged sections; `None` when
  nothing matched or every runner failed.

**Wiring** (all three chat paths):

- `financial_llm_engine.generate_financial_analysis` — right after the
  deep-dive + mega-decomposition gates, before the specialist builders, so a
  single explicit tool ask returns the engine's answer (cached like every
  other response).
- `financial_llm_engine._build_single_answer` — same gate for each part of a
  decomposed mega query.
- `ai_chatbot.OctavianEnhancedChatbot.process_query` — the classic path
  returns the tool answer directly (no symbol-analysis pipeline).

Intentionally NOT routed here: "probability of reaching $X" (already handled
by the target-probability engine via the probability intent) and COT
(needs a live CFTC download that could hang a chat turn).

### AI Chatbot (`ai_chatbot.py`)

The Octavian Enhanced AI Market Chatbot (5,993 lines). It integrates the
unbiased analyzer, source-credibility weighting, timeframe analysis, trader
profiling, and the financial LLM engine.

**Main class: `OctavianEnhancedChatbot`** (line 111).

- `process_query(query, user_id)` — the primary entry point. Returns a dict
  with `user_text` (clean user-facing answer), `raw_analysis` (hidden full
  detail for the "Show AI Reasoning" expander), charts, suggestions, and
  metadata.
- Lazy singleton accessors: `analyzer`, `db_manager`, `news_engine`,
  `multi_asset_analyzer`, `advanced_news_processor`, `cross_sector_analyzer`,
  `narrative_generator`, `source_credibility_engine`, `timeframe_engine`,
  `intent_engine`, `unbiased_analyzer`, `simulation_engine`,
  `response_formatter`, `llm_agent` (the financial LLM engine).
- `_call_lm_studio_interpreter(prompt)` — optional live-LLM path via LM
  Studio; deterministic heuristic fallback when offline.
- `_detect_intent_and_timeframe(query)` / `_detect_intent(query)` — intent
  classification + timeframe scope.
- `_extract_symbols(query)` / `_extract_symbols_legacy` — robust ticker
  extraction (see entity firewall in the financial LLM engine).
- `_analyze_symbol_unbiased` / `_analyze_symbol_enhanced` /
  `_analyze_symbol_comprehensive` — per-symbol deep analysis pipelines.
- Section writers: `_generate_risk_analysis_section`,
  `_generate_profit_maximization_section`, `_generate_unbiased_recommendations`,
  `_generate_credibility_insights_section`, `_generate_timeframe_recommendations`,
  `_generate_octavian_signature_insights`, `_generate_comparative_insights`,
  `_generate_trading_recommendations`, `_generate_octavian_guidance`.
- Metrics/indicators: `_calculate_enhanced_metrics`,
  `_extract_timeframe_technical_signals`, `_calculate_key_metrics`,
  `_extract_technical_signals`, `_calculate_risk_metrics`,
  `_calculate_price_levels`, `_calculate_enhanced_anticipation_factors`.
- Data: `_get_real_time_data(symbol, period)` (two overloads),
  `_aggregate_news_analysis`, `_get_symbol_sectors`.
- Charts: `_create_advanced_price_chart`, `_create_prediction_chart`,
  `_create_comparison_chart`, `_create_volatility_chart`,
  `_generate_opportunity_charts`, `_generate_charts`,
  `_generate_enhanced_charts`, `_generate_intent_aware_charts`.
- Response assembly: `_generate_intent_aware_response`,
  `_generate_standard_intent_response`, `_generate_concise_response`,
  `_generate_comprehensive_response`, `_generate_symbol_analysis_text`,
  `_create_analysis_summary`, `_create_enhanced_analysis_summary`,
  `_extract_credibility_insights`, `_generate_followup_suggestions`,
  `_generate_helpful_response`, `_get_query_suggestions`.

**UI:** `show_advanced_chatbot()` (line 3594) and `show_octavian_chatbot()`
(line 5577) — the Intelligence Center assistant. The "Show AI Reasoning"
expander reveals the per-symbol raw analysis (indicators, ML scores,
sentiment, signal factors, market context, timeframe).

### Intent Detection (`intent_detection_engine.py`)

`DynamicIntentDetectionEngine` — richer intent layer used by the chatbot:

- Enums: `IntentCategory` (trade_setup, risk_analysis, news_sentiment,
  comparison, prediction, valuation, portfolio, macro, etc.), `ResponseFormat`
  (analysis, table, chart, scorecard), `DetailLevel`, `Urgency`.
- `analyze_intent(query, detected_symbols)` → `IntentAnalysis` (primary
  intent, intent hierarchy, confidence, response format, detail level,
  urgency, comparison flag + count, chart/table/calculation requirements,
  emotion, needs-explanation/validation/alternatives/risk-warning flags,
  response strategy, ambiguity score, keywords).
- Pattern scoring via `_score_all_intents` (keyword/regex patterns per
  category), `_determine_intent_hierarchy` (primary + secondary),
  `_detect_response_format`, `_detect_detail_level`, `_detect_urgency`,
  `_detect_comparison`, `_detect_emotion`, `_needs_explanation`,
  `_needs_validation`, `_needs_alternatives`, `_needs_risk_warning`,
  `_formulate_response_strategy`, `_calculate_ambiguity`, `_extract_keywords`.

### Response Formatter (`response_formatter.py`)

`ResponseFormatter` converts raw quantitative data into high-fidelity
institutional reports. Output is a `FormattedResponse` dataclass with
`user_text` (clean, conversational, direct answer) + `raw_analysis` (hidden
full detail) + metadata — the separation that powers the "Show AI Reasoning"
button.

- `format_analysis_response(analyses)` — single/multi-symbol responses;
  grouped bullish/bearish/neutral.
- `format_scan_response(opportunities)` — ranked market scans.
- `_format_single_symbol` — per-symbol prose with confidence expressed
  naturally (no jargon).
- `_generate_institutional_synthesis`, `_multi_asset_conclusion`,
  `_create_quantitative_breakdown`, `_calculate_aggregate_confidence`.

### ML Analysis (`ml_analysis.py`)

`MLMarketAnalyzer` — scikit-learn ensemble prediction model (Random Forest +
XGBoost + LightGBM + Logistic Regression + AdaBoost, weighted voting),
50+ technical/fundamental features, confidence-calibrated forecasts.
`ensure_ml_libraries()` guards optional imports; `get_analyzer()` singleton.

### Advanced ML Engine (`advanced_ml_engine.py`)

`advanced_ml_engine.py` (747 lines) — the "massive" ensemble:

- 5-model ensemble: **LSTM (25%), Transformer (25%), MLP (20%), Random
  Forest (15%), GBM (15%)** with dynamic weights.
- 8 input features: Close, High, Low, Volume, Volatility, RSI, MACD,
  Institutional Flow (13F-injected into training tensors).
- Online learning on recent data.
- PyTorch for the deep nets; scikit-learn for the trees.

### Quant Ensemble Model (`quant_ensemble_model.py`)

`QuantEnsembleModel` — unified signal model for the quant terminal/sim/lab
with options-aware augmentation. `QuantSignal` dataclass (signal, score,
confidence, direction, source); `get_quant_ensemble()` singleton.

### Adaptive Reasoning (`adaptive_reasoning_engine.py`)

`AdaptiveReasoningEngine` — adapts Octavian's models dynamically to current
market regimes (tracks `market_regime` state) and personalizes output to the
user's trader profile.

### Personalization (`personalization_engine.py`)

`AdaptivePersonaEngine` — tracks user behavior and learns preferences:
action logging (page views, searches, tool usage), interest detection
(symbol clusters, asset-class bias), and adaptive UI context (filtering and
prioritization). `get_adaptive_engine()` singleton.

### Portfolio Chatbot Context (`portfolio_chatbot_context.py`)

Gives the chatbot portfolio awareness:

- `detect_portfolio_intent(query)` — recognizes 6 portfolio query types
  (grade/health check, allocation, risk, performance, rebalance, position
  questions).
- `extract_ticker_from_fit_query(query)` — pulls the symbol from "is X a fit"
  questions.
- `load_all_portfolios(user_id)` — merges paper trading accounts + manual
  portfolios (`manual_portfolios.json`).
- `_get_all_holdings(context)` — flattens all holdings.
- `build_portfolio_context_string(context, max_length)` — composes the
  context snippet injected into chatbot prompts.

### Chatbot Evaluation Harness (`chatbot_eval/`)

Mass-scale prompt testing with persistent SQLite storage:

- `prompt_factory.py` — generates thousands of prompts across every asset
  class plus macro, geopolitics, current events, risk/hedging, and
  trade-setup categories; each carries structured **expectations**
  (primary, requires, requires_any, avoid) for objective scoring.
- `pipeline.py` — runs the real `financial_llm_engine` against a *mocked*
  deterministic data layer (seeded by symbol name) — thousands of queries in
  milliseconds, zero network.
- `rubric.py` — 8 criteria out of 10: ticker_cleanliness, data_grounding,
  task_fulfillment, relevance, depth, + more.
- `db.py` — SQLite persistence (evaluations, criteria_scores, issues).
- `run_eval.py` — orchestrator CLI (`--limit`, `--workers`, `--report`,
  `--rerun-failed`, `--sync-issues`).
- `resume_eval.py` — resume-capable bucket runner (same deterministic
  corpus, skips persisted queries, safe to re-invoke).

---

## News & Sentiment Layer

### News Analysis Engine (`news_analysis_engine.py`)

Core news aggregation + sentiment engine (1,229 lines).

- **Sources**: generic RSS feeds, Alpha Vantage news API, Reddit financial
  subreddits, Yahoo Finance RSS. All fetches go through `api_rate_limiter`
  (User-Agent headers, sliding-window limits, exponential backoff, response
  caching) to avoid Reddit 403 / Yahoo 429 errors.
- **NLTK**: `_ensure_nltk_data()` / `_lazy_ensure_nltk()` download required
  corpora (vader_lexicon, punkt, stopwords) into `nltk_data/` lazily.
- **Data model**: `NewsArticle` (source, title, summary, url, published,
  sentiment, market_impact, relevance, symbols, category),
  `MarketWhisper` (rumors/whispers with type), `SentimentAnalysis` (score,
  label, confidence, key_phrases, aspects).
- **Pipeline**: `start_background_processing()` (daemon thread, scheduled
  polling) → `fetch_and_process_news()` → `_process_articles()` (dedupe,
  sentiment, impact, relevance, symbol extraction, category tagging) →
  `_store_articles()` (persistence).
- **Sentiment**: `analyze_sentiment` (VADER via `textblob`/`nltk` with a
  `_simple_sentiment_analysis` fallback), `_score_to_category` maps to
  SentimentScore enum (BULLISH/BEARISH/NEUTRAL).
- **Symbol extraction**: `_extract_symbols` — ticker candidates from text
  with noise filtering.
- **Queries**: `get_market_sentiment(hours_back)`, `get_sentiment_for_symbol`,
  `get_market_whispers(symbol)` (with `_infer_whisper_type`), and
  `get_news_summary_for_symbol` — the data behind the news dashboard.
- **Metrics**: `get_metrics()` (articles processed, avg sentiment, source
  breakdown). `stop()` shuts the background loop.
- Singleton: `get_news_engine()`.

### Advanced News Processor (`advanced_news_processor.py`)

Deeper news intelligence (1,289 lines) used by the chatbot:

- `ProcessedArticle` / `MarketNarrative` data models.
- `_deduplicate_articles` — similarity-based dedupe across sources.
- `_analyze_sentiment_advanced` — financial-domain sentiment with
  `_financial_sentiment_analysis` (finance lexicon scoring).
- `_extract_symbols_advanced` — context-aware symbol extraction.
- `_extract_sectors` / `_extract_countries` — entity extraction.
- `_calculate_market_impact_advanced` — impact scoring given symbols and
  sectors.
- `_analyze_cross_sector_implications` — spillover analysis.
- `_extract_anticipation_factors` — forward-looking factor detection
  (expectations vs reality).
- `_extract_narrative_elements` / `_extract_decision_implications` —
  narrative + decision-relevance tagging.
- `analyze_symbol_sentiment(symbol, timeout)` — per-symbol deep sentiment
  (timeout-guarded).
- Singleton: `get_advanced_news_processor()`.

### Source Credibility (`source_credibility_engine.py`)

Weights news by source/author trustworthiness:

- `SourceTier` enum (tier-1 institutional, tier-2 mainstream, tier-3
  aggregator, tier-4 social, etc.) with per-source base credibility.
- `_initialize_source_weights()` — the source→credibility table.
- `calculate_weighted_news_score(news_item, ...)` — combines base source
  credibility, author score, time multiplier (recency decay), and session
  multiplier into one weighted score.
- `get_source_credibility(source)`, `get_author_credibility(author)`,
  `update_author_credibility(author, prediction_accuracy)` — reputation
  feedback loop.
- `get_weighted_news_summary(items, ...)` — aggregate view with
  `_get_source_breakdown`.

### News Dashboard (`news_dashboard.py`)

The Intelligence Center's News & Sentiment tab (935 lines):

- `show_news_dashboard()` — entry.
- `show_live_news_feed(engine, symbol_filter, time_range, ...)` — filterable
  feed with sentiment badges.
- `show_sentiment_analysis(...)` — sentiment over time + distribution.
- `show_market_whispers(engine, symbol_filter)` — rumor/whisper tracking.
- `show_sector_sentiment(engine)` — sector-level sentiment map.
- `show_news_impact_analysis(...)` — news→price impact correlation.
- `generate_sample_articles(...)` / `generate_market_sentiment()` —
  offline/demo fallbacks so the tab never crashes without network.
- `get_sentiment_label/emoji/color` — consistent rendering helpers.
- `export_sentiment_data(engine, symbol_filter)` — CSV export.

### Narrative Generator (`narrative_generator.py`)

`NarrativeGenerator` + `NarrativeComparison` — produces narrative prose
comparing AI analysis vs market sentiment for a symbol (used by the chatbot
to explain *why* the market and the model may disagree).

---

## Market Analytics Engines

### Unbiased Market Analyzer (`unbiased_market_analyzer.py`)

The "pure profit-maximizing" analyzer (1,681 lines) — analyzes ANY asset
without bias toward well-known tickers, producing model conclusions and
data-driven insights. It powers the automated trading engine's opportunity
scans and the chatbot's unbiased analysis. Provides per-symbol analysis with
profit-probability estimates, risk/reward, and model signals.

### Multi-Asset Analyzer (`multi_asset_analyzer.py`)

`MultiAssetAnalyzer` — cross-asset coverage: equities, FX, futures, crypto,
commodities, options, indices. Detects asset class from the symbol, fetches
appropriate data, and returns unified analysis dicts so downstream UI can
render any instrument uniformly.

### Market Scanner (`market_scanner.py`)

Sector/momentum screening page (1,007 lines):

- `show_market_scanner()` — entry with universe selection, filters
  (sector, market cap, min score), scan triggers.
- Signal computation (momentum, RSI, trend, volume) per symbol → composite
  Score column.
- `_render_heatmap(...)` — sector/asset heatmap.
- `_render_signal_table(df, show_cols)` — ranked table.
- `_render_momentum_bars(df, title, col)` / `_render_rsi_chart(df)` —
  charts.
- Results exportable; scanner draws from `ticker_universe`.

### Market Movers (`market_movers.py`)

Top gainers/losers/volume movers (1,261 lines):

- `show_market_movers()` — live movers board with VERIFIED / UNVERIFIED
  badges (fresh quote cross-check), change recomputed from displayed
  price/prev_close so math is always consistent.
- `show_symbol_search()` — ticker search + quick analysis.
- Used inside the Dashboard / watchlist flows; region tags (NYSE/NASDAQ)
  shown next to symbols.

### Cross-Asset Transmission (`cross_asset_transmission.py`)

Quantitative answer engine for questions like "what does the recent
volatility in oil futures affect other asset prices — provide quantitative
proof" (540 lines). Instead of canned boilerplate it:

1. Detects the **anchor asset** in the query (oil → CL=F, gold → GC=F, ...)
   via symbol/alias resolution.
2. Downloads anchor + candidate asset histories.
3. Computes **rolling correlations, beta, and transmission magnitudes**
   (anchor move → expected move in each candidate).
4. Renders a ranked transmission table with confidence, plus a caveat that
   correlation ≠ causation.

### Macro Cross-Asset (`macro_cross_asset_engine.py`)

Macro regime → asset-class transmission modeling (1,307 lines): yields,
currencies, commodities, equities, credit; scenario shocks (rate hike,
inflation surprise, recession) propagated through per-asset sensitivities;
regime labels; macro dashboard data. Used by the Daily Briefing and chatbot
macro answers.

### Daily Intelligence (`daily_intelligence.py`)

The Daily Briefing engine (1,300 lines):

- `_safe_yf_download` and per-domain fetchers: `_fetch_equity_indices`,
  `_fetch_yield_curve`, `_fetch_fx_rates`, `_fetch_commodities`,
  `_fetch_crypto`, `_fetch_volatility_surface`, `_fetch_sector_performance`,
  `_fetch_macro_indicators`, `_fetch_fed_signals`.
- Indicator helpers: `_pct_change_1d/_nd`, `_last_close`, `_annualized_vol`,
  `_rsi`, `_above_sma`, `_macd_signal`, `_bollinger_position`.
- `_assess_monetary_bias()` — Fed stance from signals.
- `_detect_cross_asset_signals(...)` — cross-asset divergence detection
  (828 lines of signal logic).
- `_get_regime_analysis()` — HMM regime classification.
- `_detect_narrative_dislocations(...)` — narrative vs price gaps.
- `_generate_trade_ideas(...)` — idea generation.
- `_recommend_posture(regime, vix, spread)` — risk posture.
- `DailyIntelligenceEngine` — assembles the full briefing:
  `_compile_key_risks`, `_compile_key_opportunities`.
- Singleton: `get_daily_engine()`.

### Market Heartbeat (`market_heartbeat_system.py`)

Real-time market "pulse" page (461 lines):

- `fetch_heartbeat_data()` — live index/rates/commodities data.
- `_fetch_sector_data()` — sector performance.
- `generate_market_heartbeat(g, u, timeframe, custom_g, custom_u)` —
  health-state prose given growth/uncertainty axes.
- `_render_gauge_chart`, `_render_sector_heatmap`, `_render_heartbeat_card`
  — visual components.
- `show_market_heartbeat_tab()` — assembles the page.

### Regime Detection (`regime.py`)

Lightweight regime utilities used across the platform (replaces a missing
`regime` module dependency):

- `volatility_regime(vix)` — LOW/MED/HIGH regimes.
- `risk_on_off(spy_change, vix)` — RISK_ON / RISK_OFF classification.
- `classify_regime(returns)` — trend/mean-reversion/volatility regimes from
  returns.
- `get_regime_context()` — combined regime dict for model inputs.

### Timeframe Analysis (`timeframe_analysis_engine.py`)

Advanced multi-timeframe analysis (706 lines): aligns signals across
intraday/daily/weekly/monthly, produces per-timeframe context-aware insights
and forward-looking projections; supports all asset classes. Used by the
chatbot (`timeframe_engine` accessor) to frame recommendations by horizon.

### Counter-Trend Analyzer (`counter_trend_analyzer.py`)

Identifies weaknesses/contradictions in mainstream economic narratives and
generates signals that profit from consensus mispricing (898 lines):

- `NarrativeStrength`, `CounterTrendSignal` data models.
- `CounterTrendAnalyzer` — narrative consensus vs fundamental score,
  divergence computation, `generate_counter_signals(divergence_threshold,
  min_strength)`; `get_all_narratives()` for the tracker table.
- `score_narrative_strength(...)`, `identify_macro_contradictions(...)`
  module helpers.
- Singleton: `get_counter_trend_analyzer()`.

### Factor Crowding (`factor_crowding_engine.py`)

Detects crowded trades, factor concentration risk, and hedge-fund holdings
overlap (571 lines): per-factor crowding scores, overlap analytics, and
crowding-adjusted recommendations ("avoid positions already saturated with
institutional capital").

### Narrative Dislocation (`narrative_dislocation_engine.py`)

Price-narrative divergence engine (898 lines): compares the market's
narrative (news/sentiment) with what prices imply; quantifies dislocation
and flags reversion opportunities. Consumed by the quant modeling lab's
narrative tab and the daily briefing.

### Target Probability (`target_probability_engine.py`)

Probability engine for price targets (1,173 lines): computes the probability
of reaching a target price within a horizon from historical vol, drift,
and distribution assumptions; `show_target_probability()` UI with
`_render_sensitivity` charts. Produces `TargetAnalysisResult` objects with
scenario probabilities.

### Market Consensus (`market_consensus_engine.py`)

Consensus-estimate and dislocation engine (693 lines):

- `DataProvenance`, `ConsensusEstimates`, `SentimentSnapshot`,
  `ImpliedExpectations`, `DislocationFinding`, `ConsensusDislocationReport`
  data models.
- `reverse_dcf_expectations(...)` — implied expectations from price.
- Used by the Financial Model Generator's consensus layer
  (`_render_consensus_dislocation` in the UI).

### Cross-Sector Analyzer (`cross_sector_analyzer.py`)

Cross-sector correlation + anticipation factor engine (836 lines): sector
correlation matrices, anticipation factors (predicting outcomes vs current
expectations), sector rotation/leadership identification, geopolitical
impact assessment, economic-indicator cross-asset impact.

### Sector Scanner (`sector_scanner.py`)

Sector display/scan helpers: `_SECTOR_DISPLAY` map (technology,
semiconductors, ...) and sector-symbol lists for scans.

### FX Scanner (`fx_scanner.py`)

Currency scanning (244 lines): `_fetch_pair_df`, `scan_fx(lookback, pairs)`
(momentum/RSI signals per pair), `get_fx_quote(pair_key)`,
`get_all_fx_pairs()`, `get_core_fx_pairs()` (majors list).

### Chart Image Analyzer (`chart_image_analyzer.py`)

Upload a chart screenshot → AI pattern recognition (947 lines): image
preprocessing, technical-pattern detection (trendlines, flags, head &
shoulders approximations), indicator overlay reading, and trade
recommendations with confidence. `show_chart_analyzer()` is the UI. Has an
optional full-vision-model path.

### Graph Analysis (`graph_analysis.py`)

Programmatic chart analysis on DataFrames (237 lines):
`analyze_price_chart(df, indicators_df)` — trend/pattern/level detection;
`analyze_rsi_chart(rsi_series)` — RSI shape analysis; `analyze_volume_chart`
— volume-price relationship. Used by the chart analyzer and document
analyzer.

---

## Quant Research & Strategy Lab

### Quant Portal (`quant_portal.py`)

The comprehensive quantitative research portal (1,255 lines) — a unified
interface combining Quant Terminal, Quant Modeling Lab, and Strategy
Research Lab capabilities. Entry: `render_quant_portal()`.

**8 tabs:**

1. **Multi-Asset Analysis** — symbol input with quick-select buttons
   (Stocks/Futures/FX/Crypto). The buttons are `on_click` callbacks that
   WRITE into `st.session_state["qp_symbol_input"]` (seeded via
   `setdefault("qp_symbol_input", "AAPL, MSFT, NVDA")`) so the whole portal
   sees the symbols and analysis auto-runs — this fixed the dead-button bug
   where a local var was discarded before the next rerun. Quick-select fills
   curated mega-liquid names first (AAPL/MSFT/NVDA; ES=F/NQ=F/CL=F;
   EURUSD=X; BTC-USD), padded from the live universe.
2. **Risk & Correlation** — `_calculate_advanced_metrics(returns)`
   (Sharpe, Sortino, max drawdown, vol, VaR, beta, alpha) + correlation
   matrix.
3. **Quant Signals & ML** — quant ensemble signals + ML predictions.
4. **Regime Detection** — regime classification (HMM/volatility).
5. **Strategy Evolution** — genetic strategy evolution (wired to
   `genetic_strategy_engine.evolve` with a 3-arg progress callback).
6. **Cross-Asset Macro** — macro cross-asset transmission.
7. **Advanced Backtesting** — wired to `AdvancedBacktester` (real metrics,
   trades, alpha/beta vs benchmark; synthetic fallback when data
   insufficient).
8. **Alternative Data** — fetches ALL alt-data sources at once and groups
   signals by real category with bull/bear counts + composite verdict;
   `_alt_data_what_it_means(sig)` gives in-depth per-signal explanations.

Helpers: `_apply_portal_css`, `_section`, `_metric_card`.

### Quant Modeling Lab (`quant_modeling_lab.py`)

Advanced quantitative modeling environment (561 lines):

- `_render_liquidity_tab()` — market microstructure/liquidity analytics.
- `_render_crowding_tab()` — factor crowding detection (via
  `factor_crowding_engine`).
- `_render_narrative_tab()` — narrative dislocation analysis.
- `render_quant_modeling_lab()` — entry with tabs (regime detection,
  factor crowding, ML framework RF/GBM/LSTM/RL, narrative).
- `_section`, `_metric`, `_dark_layout` rendering helpers.

### Strategy Research Lab (`strategy_research_lab.py`)

Comprehensive strategy design/testing/evolution environment (1,795 lines):

- `_render_pairs_tab()` — statistical arbitrage & pairs trading.
- `_render_volatility_lab_tab()` — vol trading strategies.
- `_render_factor_tab()` — factor investing (Fama-French + custom).
- `_render_genetic_tab()` + `_render_genetic_demo()` — genetic strategy
  evolution (wraps `genetic_strategy_engine`).
- `_render_alpha_signals_tab()` — alpha signal screening.
- `_render_backtest_tab()` — strategy backtesting.
- `_render_options_strategy_builder_tab()` — options strategy
  construction (via `options_engine`).
- `_render_options_alpha_lab()` — options alpha research.
- `_render_futures_commodities_lab()` — futures/commodities strategies.
- `_render_strategy_grader_tab()` — strategy grading (via
  `strategy_intelligence_engine`).
- `_render_strategy_suggester_tab()` — strategy suggestions.
- `render_strategy_research_lab()` — entry.

### Genetic Strategy Engine (`genetic_strategy_engine.py`)

Genetic-algorithm strategy optimizer (1,209 lines): evolves trading
strategies (parameter sets / rule combinations) over generations using
fitness = backtest performance; `evolve(...)` takes a
`progress_callback(gen, total, fitness)` signature (the portal fixed its
`_progress` to accept 3 args). Supports population, crossover, mutation,
elitism, and fitness history tracking.

### Advanced Backtester (`advanced_backtester.py`)

Professional-grade backtesting (434 lines):

- `BacktestTrade` / `BacktestResult` dataclasses.
- `AdvancedBacktester(initial_capital=100_000.0)` — constructor takes ONLY
  `initial_capital` (the portal's older call passing `symbol=` was fixed).
- `run_backtest(df, symbol, rebalance_every=10, ...)` — full loop:
  indicator computation, quant-ensemble signals, position sizing, trade
  log with entry/exit/exit_reason, rolling equity.
  **Lookback floor of 60 bars** (quant ensemble needs ≥ 50 bars warmup to
  emit directional signals — without the floor every signal was NEUTRAL and
  the backtest returned all zeros).
- `_compute_metrics(...)` — returns, Sharpe, Sortino, max drawdown, win
  rate, profit factor, alpha/beta vs benchmark, rolling analytics.
- `run(df, symbol, **kwargs)` — alias entry point (legacy callers).

### Legacy Backtest (`backtest.py`)

Tiny EMA20/EMA50 crossover walk-forward backtest used by early modules
(equity curve + `walk_forward(df, train, test)` splitter).

---

## Algorithm Builder

### Engine (`algorithm_builder_engine.py`)

A research-grounded algorithmic strategy generator (2,506 lines). Builds,
backtests, and exports quantitative trading algorithms from vague
natural-language requests (auto mode) or explicit structured constraints
(guided mode).

**Pipeline (`build_algorithms(...)`):**

1. Resolve families: auto mode → `parse_request(request, risk)` (keyword
   matching across archetype groups, interleaved round-robin 2-per-group,
   cap 4, so multi-topic requests get a diverse mix); guided mode → the
   `archetypes` list; empty → risk-based defaults.
2. Fetch data per symbol (universe capped at 6; each symbol retried once
   after a 1.5s pause for transient provider outages; requires ≥ 60 bars;
   fills missing OHLC columns; errors collected). No data → raise with the
   per-symbol error list.
3. For each family: run `runs_per_family` (default 3) searches with
   distinct seeds — "many backtests per algorithm". Each run: parameter
   sampling → `backtest_ohlcv` → `compute_metrics` → window breakdown →
   trade narratives → `AlgorithmResult`.
4. Walk-forward CV (optional, `wf_folds`): `walk_forward_evaluate` —
   multi-fold trailing-window OOS with mean/median/worst Sharpe,
   consistency, and residual autocorrelation.
5. Ensemble (optional): `build_ensemble(members, df, method)` —
   `"dynamic"` default: per-bar weights = wealth-adaptive mixing (fast
   universalization) TILTED by measured quality (Sharpe+Sortino, drawdown
   cap, win rate, trade robustness) and penalized for redundancy (pairwise
   correlation). `_ensemble_rationale` writes per-member
   strength/weakness/redundancy/weight prose. Members capped at top-6 by
   OOS Sharpe.
6. Failure diagnosis: families that find no viable strategy get
   `_error_result(..., suggestion=)`; `diagnose_failure` + `_data_character`
   profile the data (trend strength, lag-1 autocorr, mean-reversion index,
   vol) and produce concrete tweaks + better-suited families.

**Strategy archetypes** (registered via `_register`, generated Python is
compilable & runnable):

| Archetype | Idea |
|---|---|
| `_trend_ma` | Moving-average trend following (EMA/SMA cross + filters) |
| `_breakout` | Donchian-style breakout with volume confirmation |
| `_dual_momentum` | Absolute + relative momentum (MOM2-12 style) |
| `_rsi_meanrev` | RSI mean reversion |
| `_bollinger_meanrev` | Bollinger-band mean reversion |
| `_vol_target` | Volatility targeting / position scaling by vol |
| `_market_making` | Spread-capture market making on synthetic L2 |
| `_statarb_z` | Z-score statistical arbitrage on pairs/spreads |
| `_gap_fade` | Gap-fade counter-trend entries |
| `_stoch_williams` | Weekly-resampled Stochastic %K/%D + Williams %R timing, volume-surge scaling, weekly loss-cut (from Paik et al. low-frequency timing paper) |
| `_value_momentum` | Value (price-based proxy) + MOM2-12 momentum blend, both z-scored, tanh composite, absolute-momentum gate (Asness-style) |
| `_flag_breakout` | Strict bull/bear flag pattern (flagpole → tight consolidation → continuation) with strict bounds (near-zero false positives) |
| `online_ops` | Online portfolio-selection operations (FTRL, PAMR, OLMAR, CWMR, AntiCor steps via `_ftrl_step`/`_pamr_step`/`_olmar_step`/`_cwmr_step`/`_anticor_step`; `run_ops_basket`, `run_fast_universalization`) |

**Indicators**: `_sma`, `_ema`, `_rsi`, `_atr`, `_rolling_z`,
`_resample_weekly`, `_simplex_project` (projection onto simplex for
weights).

**Backtest loop (`backtest_ohlcv`)** — vectorized signal → positions →
trades with entry/exit timestamps, exit reasons (STOP_LOSS / TAKE_PROFIT /
TRAILING_STOP / MAX_HOLD / SIGNAL_FLAT / END_OF_DATA), fees, and slippage;
`compute_metrics` (Sharpe with ddof=1 mean/std · √252, Sortino, max
Drawdown, win rate, profit factor, exposure, `_lag1_autocorr` residual
check). `window_returns` / `window_metrics` give 5y/3y/2y/1y/6m/3m/1m
breakdowns (trades attributed by exit date; short histories leave longer
windows None). `rank_factors(dfs)` computes cross-sectional factor z-scores
(MOM2-12, value proxy, low-vol, trend, volume momentum) — WorldQuant-style
retail factor ranking with symbol as first-class column.

**Trade transparency**: `_stamp_trades` adds `symbol` + `asset_type`
(`classify_asset_type`: futures =F / FX =X / crypto -USD / index ^ / ETF
list / equity) to every trade; `trade_narratives` writes dynamic prose per
trade (entry context: 5d drift + vol; why it exited; P&L; hold).

**`AlgorithmResult`** carries: metrics, window_returns, window_metrics,
trades (stamped), equity curve, per-trade narratives, SWOT (computed in
UI), generated Python source, `strategy_spec()` (self-contained deployable
spec), `to_dict()`, `to_markdown()`.

**Deployment bridge**: `strategy_spec_to_signal(spec, df)` recomputes live
target exposure in [-1,1] from the spec on fresh OHLCV (long_only clips
shorts; None for online_ops); `strategy_spec_to_weights(spec, dfs)` gives
portfolio rebalance weights for online_ops. These feed the paper trading
deployment path (see Trading Infrastructure).

**Generated code**: `generate_python(archetype_name, params, ...)` emits a
standalone, runnable signal script (`_signal_body`, `_py_lit` for safe
literal formatting); tests verify generated scripts compile AND run on mock
data. `_default_backtest_params(risk, direction)` provides risk-aware
stops/targets/position sizing.

### UI (`algorithm_builder_ui.py`)

Streamlit UI (693 lines) with two build modes:

- **Auto** — describe what you want in plain language; the engine decides
  families, parameters, and (if requested) ensemble composition.
- **Guided** — pick strategy families, universe, risk profile, direction,
  and advanced execution settings explicitly.

Key components: `render_algorithm_builder()` (entry, mode toggle, sliders
for count / runs_per_family / wf_folds / trials, universe multiselect,
build button with progress); `_build(build_args, progress)` (runs the
engine, reports per-family progress); `_render_result(r, idx, expanded)`
(metric row, equity chart with unique key `eq_<id>` to avoid
DuplicateElementId, SWOT expander, trade-by-trade reasoning expander,
window breakdown dataframe, trade table, markdown download);
`_render_trade_table` (clean dataframe; best/worst/riskiest highlights
computed on NUMERIC pnl before stringifying); `_render_window_breakdown`;
`_render_factor_ranking`; `_render_methodology` (explains the research
basis); `_render_deploy_panel` ("Test in paper trading" button per result).

`_render_deploy_panel` deploys a built algorithm to a NEW paper trading
account or an EXISTING portfolio; existing-with-strategy shows a warning +
mandatory override checkbox before the deploy button enables.

---

## Trading Infrastructure

### Paper Trading System (`paper_trading_system.py`)

Multi-account paper trading (1,033 lines).

**Data models:**

- `TradeAction` enum: BUY, SELL, SHORT, COVER.
- `AccountStatus` enum: ACTIVE, PAUSED, CLOSED.
- `PaperTradingAccount` (id, user_id, name, balance, initial_balance,
  status, created_at).
- `Position` (symbol, quantity, avg_entry, side, realized_pnl, opened_at,
  asset_type, last_price).
- `OptionPosition` (underlying, contract, type CALL/PUT, strike, expiry,
  quantity, avg_premium).
- `Trade` (timestamp, symbol, action, quantity, price, strategy, reasoning,
  market_context JSON, pnl).

**Core methods:**

| Method | Purpose |
|---|---|
| `create_account(user_id, name, balance)` | New account with initial balance |
| `get_account(account_id)` / `list_accounts(user_id)` | Lookup |
| `delete_account(account_id)` | Close/remove an account |
| `execute_trade(account_id, symbol, action, qty, ...)` | Live-priced BUY/SELL/SHORT/COVER with balance validation, position averaging, realized P&L, full logging (reasoning + market context JSON) |
| `execute_option_trade(...)` | Options chain trades (underlying, contract, strike, expiry) |
| `execute_optimization_order(account_id, symbol, ...)` | Sizing-adjusted order (from Position Optimizer recommendations) |
| `get_positions(account_id)` / `get_option_positions(account_id)` | Open positions with live P&L |
| `get_trade_history(account_id, limit)` | Trade log |
| `update_automation_settings(account_id, enabled, ...)` | Toggle/configure automation |
| `_set_automation_config(account_id, config)` | Persist automation config |
| `deploy_strategy(account_id, strategy_spec)` | Deploy an algorithm-builder spec into the account's automation config (merges under `strategy_spec`, preserves other keys like risk rules; does NOT toggle automation) |
| `get_deployed_strategy(account_id)` / `remove_strategy(account_id)` | Deployed-strategy access/manage |

Persistence is SQLite via `db_manager` (tables `paper_trading_accounts`,
`paper_trading_positions`, `paper_trading_trades` with foreign keys and
indexes). `_get_current_price` uses live quotes; balance updates are atomic
under locks. `_normalize_user_id` gives string user ids. Singleton:
`get_paper_trading_system()`.

### Paper Trading UI (`paper_trading_ui.py`)

The full paper trading dashboard (1,360 lines), `show_paper_trading_dashboard()`:

- **Performance Dashboard & Grading** — large grade display (A+ to F via
  `SimulationGrader`), total P&L + return %, Sharpe, max drawdown, win
  rate, expandable score breakdown, risk status, recent trades with P&L.
- **5 tabs**: Breaking Trades (generator), Equity Trade, Options Trade,
  Open Positions, Trade History.
- `show_account_overview` — account management (create/switch, status
  indicators, metrics).
- `show_positions` — live positions with color-coded P&L, close buttons.
- `show_options_portfolio` — options positions (Greeks-aware display).
- `show_trade_history` — filterable trade log, expandable AI reasoning +
  market context, CSV export.
- `show_automation_controls` — Start/Pause/Resume/Stop buttons + status
  (Running/Paused/Stopped/Error) + performance metrics.
- `show_automation_mode_selection` — full-auto vs strategy-spec mode.
- `show_risk_configuration_dialog` — sliders for all risk params (max
  position 1-25%, max exposure 10-100%, max positions 1-20, min confidence
  50-95%, stop loss 1-20%, take profit 2-50%, max loss/trade 0.5-5%, R:R
  1-5x).
- `show_performance_charts` — cumulative P&L, action distribution pie,
  most-traded symbols bar.
- `show_manual_trade_dialog` — manual BUY/SELL/SHORT/COVER entry.

### Automated Trading Engine (`automated_trading_engine.py`)

AI-driven automated trading for paper accounts (1,159 lines).

**Data models:** `AutomationStatus` enum (STOPPED/PAUSED/RUNNING/ERROR),
`RiskManagementRules` (position sizing, exposure limits, max positions,
min confidence, stop loss/take profit, R:R, balance reserve, daily loss
limit; `is_full_auto()`, `get_effective_max_position_size(confidence)`,
`should_stop_trading(account_pnl_pct, daily_pnl_pct)`),
`AutomatedTradeDecision` (symbol, action, qty, reason, confidence, target
price, stop, take_profit).

**Lifecycle API:**

| Method | Purpose |
|---|---|
| `start_automation(account_id, risk_rules, strategy_spec=None, scan_interval=300)` | Spawn the per-account worker thread; strategy_spec routes to strategy mode (falls back to whatever is deployed on the account) |
| `pause_automation(account_id)` | Pause new trades, keep positions |
| `resume_automation(account_id)` | Continue |
| `stop_automation(account_id)` | Stop + close thread |
| `get_strategy_spec(account_id)` / `set_scan_interval` | Accessors |
| `get_automation_status(account_id)` | Real-time status + metrics (trades, opportunities, P&L, win rate, started) |

**Worker loop (`_automation_loop`):** every scan interval, either:

- **Full-auto mode**: scan market (via `unbiased_market_analyzer`),
  evaluate opportunities against risk rules, `_generate_trade_decision` →
  `_execute_automated_trade`.
- **Strategy mode** (algorithm-builder spec deployed):
  `_fetch_strategy_ohlcv` per symbol, `_strategy_signal` (recompute target
  exposure from the spec), position size via
  `_strategy_position_size(capital, price, spec)` (ATR-based risk budget,
  capped by risk rules), enter/exit per symbol; `_execute_ops_strategy_cycle`
  rebalances toward target weights for online_ops.

`_manage_existing_positions` enforces stops/targets on open positions;
`_check_risk_limits` / `_can_take_trade` validate every trade against risk
rules before execution; `_fetch_live_price` live quotes; `_log_activity`
records events. Thread-safe per-account locks; graceful error handling and
retry. Singleton: `get_automated_trading_engine()`.

### Risk Engine (`risk_engine.py`)

VaR, correlation, position sizing (274 lines): `_get_asset_data` (any asset
type), `correlation_matrix`, `portfolio_var`, `position_size`, and related
risk utilities used by `trading_system/risk_manager` and the portfolio
analyzer.

### Position Optimizer (`position_optimizer_engine.py`)

Grades existing user positions and suggests mathematically superior
optimizations (1,109 lines): Kelly-criterion sizing, leverage handling (50x
futures, 100x options), portfolio-correlation-aware sizing, per-position
grades with reasons, and optimization orders that feed back into paper
trading via `execute_optimization_order`. UI: `render_position_optimizer_ui()`.

### Portfolio Analyzer

Two modules:

- **`portfolio_analyzer.py`** (UI, 542 lines): dual input modes — import
  from a paper trading account or manual entry. `compute_portfolio_metrics`
  (returns, Sharpe/Sortino, max drawdown, VaR, exposure),
  `_position_multiplier(asset_type)` (per-asset contract multipliers),
  `_fetch_live_price`, `_coerce_float`; `show_portfolio_analyzer()` entry.
- **`portfolio_analyzer_engine.py`** (502 lines): `PortfolioAnalyzerEngine`
  — comprehensive analysis with past/present/predictive metrics, stress
  tests (Lehman, COVID, Vol-mageddon), cross-asset VaR at 95% confidence.

### Trading System Subpackage (`trading_system/`)

The older strategy simulation stack:

- `trading_data_types.py` — `TradeType` (SPOT/FX/FUTURES/OPTIONS/RATES/
  YIELD_SPREAD), `Direction`, `OrderType`, `PositionSpec`, `TradeAlert`.
- `trade_generator.py` — trade idea generation with alerts.
- `risk_manager.py` — `AdvancedRiskManager`: position sizing, risk limits,
  correlation checks, trade viability analysis (imports `risk_engine`).
- `simulation_grader.py` — `SimulationGrader`: weighted grading (Total PnL
  35%, PnL/trade 25%, Win Rate 20%, + more) → letter grade A+ to F.
- `confidence_score.py` — `ConfidenceScorer`: weighted 0-100 confidence
  (technical 30%, sentiment, fundamentals, momentum, risk, model).
- `interest_rate_manager.py` — yield curve / central bank data (^TNX/^FVX/
  ^IRX proxies).
- `macro_analysis.py` — `EconomyData` per-country macro dataclass and
  analysis.

### Brokerage Engine (`brokerage_engine.py`)

Optional Interactive Brokers integration via `ib_insync` (guarded import,
`_HAS_IBKR`): IB, Stock, Option, MarketOrder, LimitOrder, Contract; Smart
Order Routing (SOR) algorithm stub. Present but not the primary execution
path — paper trading is the default.

### Breaking Trades Generator (`breaking_trades_generator.py`)

High-confidence trade setups (786 lines): scans ~15 stocks, emits
`BreakingTradeSetup` objects with entry trigger price, stop loss, 3 take-
profit targets (scaling strategy), full reasoning, technical analysis
breakdown, position sizing recommendation, and risk/reward ratios (only
>55% confidence). `OptionLeg` / `OptionSetup` dataclasses for options
variants. Singleton: `get_breaking_trades_generator()`.

### Trade Signal Overlay (`trade_signal_overlay.py`)

Generates model entry/exit markers for chart display: `TradeMarker`
(timestamp, price, type) and overlay builders used by chart UIs.

---

## Simulation Hub

### Market Simulation Engine (`market_simulation_engine.py`)

The deep market simulation engine (2,031 lines):

- **Regime & news models**: `MarketRegime`, `NewsType` enums,
  `SimulatedNewsEvent`, `SimulatedMarketData`, `TradingDecision`,
  `SimulatedOptionPosition`, `SimulationResult`.
- **`SimulationLearningEngine`** (line 265) — learns from simulation runs,
  persisting learned params to `octavian_learned_params.json`
  (`_load_learned_params` / `_save_learned_params`).
- **`MarketSimulationEngine`** (line 864) — the simulation core:
  market microstructure (agent-based order book), portfolio evolution with
  drawdowns, scenario & crisis simulation (historical + hypothetical),
  options Monte Carlo (`render_options_monte_carlo_panel` at line 1764).
- Fallback analyzers (`_FallbackAnalysis`, `_FallbackUnbiasedAnalyzer`)
  keep simulation running offline.

### Simulation Universe (`market_simulation_universe.py`)

Synthetic financial world generator (1,170 lines): generates
universe/dataset for simulations — assets with realistic price dynamics,
regimes, correlations — so simulations can run on plausible synthetic
markets (verified by `tests/verify_simulation_universe.py`).

### Simulation Viewer (`simulation_viewer.py`)

The Simulation Hub page (871 lines), `render_simulation_viewer()` with
tabs: market microstructure, portfolio evolution, crisis simulation,
simulation universe generator, hyperdim, performance, Bayesian network,
macro analyzer, micro analyzer, scenarios, options analytics,
futures/commodity sim, derivative dynamics, comprehensive sim.

### Simulation Graders

- `trading_system/simulation_grader.py` — equity-grade (PnL-weighted).
- `options_simulation_grader.py` — `OptionsPerformanceGrade`
  (delta_management, theta_efficiency, convexity_utilization 0-100).
- `futures_simulation_grader.py` — futures-grade grading.

---

## Financial Modeling Suite

### Financial Model Generator (`financial_model_generator.py`)

Institutional financial modeling (2,325 lines) — the engine behind the
Financial Model Generator page.

**Runtime safety patches** (lines 20-281):

- `_apply_financial_runtime_safety_patch()` — startup patches.
- `_patch_alt_data_signal()` — fixes `AltDataSignal` attribute mismatches
  (the `signal_type` vs `name`/`category` bug class).
- `_patch_advanced_backtester_init()` — fixes `AdvancedBacktester.__init__`
  unexpected `symbol` kwarg.
- `apply_backward_compatibility_patches()` — applies all.

**DCF engine (`InstitutionalDCFEngine`, line 455):**

- `DCFAssumptions` (revenue, growth, margins, capex, working capital, tax,
  WACC, terminal growth, years, debt/cash/shares).
- `DCFResult` (fair values per scenario, sensitivity grid, reverse-DCF,
  revenue bridge, output tables).
- `ScenarioResult`, `CatalystEvent`, `TradeSignal` supporting models.
- `project_fcf(...)` — the 20-line FCF projection table.
- `compute_wacc(...)` — CAPM-based WACC (risk-free, ERP, beta, size
  premium) with cross-check.
- `_calculate_reverse_dcf` — NOTE: the scipy `fsolve` path is broken in
  this environment (prints "Quick DCF error: only 0-dimensional arrays…"
  and silently falls back to defaults), so the deep-dive memo uses its own
  deterministic bisection on the same equation instead.
- DCF mechanics: revenue → EBIT → NOPAT → FCF → PV → terminal value → EV →
  equity → per-share, WACC × g sensitivity grid, DCF-vs-market
  reconciliation.
- Bear/Base/Bull scenario generation.

**Fundamentals fetching:** `fetch_ticker_fundamentals(ticker)` (line 2175)
— REPORTED data from yfinance info/financials with offline guard
(OCTAVIAN_OFFLINE → {}). Helpers: `_safe_float`, `_extract_ebitda_millions`,
`_extract_eps`, `_resolve_live_price`, `_validate_ticker_fundamentals`,
`_safe_info_fetch`, `_derive_revenue_millions`.

Singletons: `get_dcf_engine()`, `get_financial_generator()`.

### Financial Model Generator UI

The UI (2,030 lines), `show_financial_generator()` with tabs:

| Tab | Renderer | Content |
|---|---|---|
| DCF | `_render_dcf_tab` | Assumptions form, scenario tables, sensitivity grids, charts, Excel export |
| LBO | `_render_lbo_tab` | LBO model (feeds `InstitutionalLBOEngine`) |
| M&A | `_render_mna_tab` | Accretion/dilution + synergies |
| Comps | `_render_comps_tab` | Trading comps tables + valuation football field |
| Precedents | `_render_precedents_tab` | Precedent transactions |
| IPO | `_render_ipo_tab` | IPO pricing + bookbuilding |
| Bridge | `_render_bridge_tab` | Valuation bridge across methods |

Shared helpers: `_render_qa` (QA report display),
`_render_consensus_dislocation` (market consensus layer),
`_render_memo_button` (investment memo download),
`_render_pptx_button` (pitchbook download).

### Comps Engine (`comps_engine.py`)

Trading comps beyond a multiples table (480 lines):

- `CompsCompany` (peer with relevance score + reason), `CompsInputs`,
  `CompsResult`.
- `CompsEngine`: peer universe with explicit relevance scores; full
  multiples set (EV/Rev, EV/EBITDA, EV/EBIT, P/E, P/FCF, EV/FCF);
  distributional stats (mean/median/quartiles/high/low); implied valuation
  of the subject from the comp set. Singleton `get_comps_engine()`.

### LBO Engine (`lbo_model_engine.py`)

`InstitutionalLBOEngine` (557 lines): LBOAssumptions (purchase multiple,
leverage, debt terms, exit multiple/horizon, revenue/margin paths),
LBOSensitivity, LBOResult (IRR, MOIC, debt paydown schedule, returns
bridge); `get_lbo_engine()` singleton. Excel via `ib_excel_engine`.

### M&A Model Engine (`mna_model_engine.py`)

`InstitutionalMnAEngine` (700 lines): MnAAssumptions (acquirer/target
financials, deal structure, synergies, financing), MnASensitivity,
MnAResult (accretion/dilution, EPS impact, premium analysis);
`get_mna_engine()`.

### IPO Engine (`ipo_engine.py`)

Full IPO suite (668 lines):

- `IPOAssumptions`, `EquityAward`, `build_share_count_bridge`,
  `_dcf_value` (internal DCF for IPO).
- `IPOPricingEngine` (valuation bands), `IPOBookbuildingEngine` (demand
  curves, allocation), `IPOScenarioEngine` (`IPOScenario`),
  `IPOEngine` (orchestrator) → `IPOResult`.
- Singleton `get_ipo_engine()`.

### Precedent Transactions (`precedent_transactions_engine.py`)

`PrecedentEngine` (367 lines): scores each transaction's *relevance* to the
subject (size, growth, margins, rationale, buyer type, structure, market
regime — not just industry), computes transaction multiples (EV/Rev,
EV/EBITDA, EV/EBIT, EV/FCF), stats, and implied value. Singleton
`get_precedent_engine()`.

### Valuation Bridge (`valuation_bridge.py`)

Unifies valuations from every model into one institutional dashboard (308
lines): DCF → CCA → Precedents → IPO → LBO → M&A implied. Never averages
blindly — each methodology is weighted by relevance and reliability for the
specific situation. `ValuationBridgeResult` + `ValuationBridgeEngine`;
`get_valuation_bridge()`.

### Investment Memo (`investment_memo.py`)

`InvestmentMemoGenerator` (281 lines): MemoContext + generator producing a
structured institutional investment memo (thesis, valuation, risks,
catalysts, recommendation) — downloadable from the model generator UI.

### IB Excel Engine (`ib_excel_engine.py`)

Investment-bank-grade Excel workbook builders using openpyxl (1,448 lines):

| Function | Workbook |
|---|---|
| `build_mna_workbook(result)` | M&A model with color scales, label rows |
| `build_lbo_workbook(result)` | LBO with debt schedule, returns |
| `build_comps_workbook(comps_df, median_df, title)` | Comps table |
| `build_dcf_workbook(result)` | DCF with scenarios + sensitivity |
| `build_sensitivity_only(df, ...)` | Standalone sensitivity grid |
| `build_ipo_workbook(result)` | IPO pricing + bookbuilding |
| `build_precedents_workbook(result)` | Precedent transactions |
| `build_bridge_workbook(result)` | Valuation bridge |

Styling helpers: `_font`, `_fill`, `_align`, `IBSheet` (formatted sheet
wrapper), `_find_label_row`/`_find_input` (input-cell detection for
sensitivity), `_apply_color_scale` (conditional color scales).

### Spreadsheet Generator (`spreadsheet_generator.py`)

Advanced customizable spreadsheet creation (1,383 lines): data selection,
timeframes, calculations, formatting, visual elements, financial modeling,
export options — full user control at every step. `show_spreadsheet_generator()`,
`show_quick_templates()`, `show_advanced_customization()`.

### Presentation Generator (`presentation_generator.py`)

Institutional pitchbook generator with python-pptx (1,806 lines):

- Low-level slide primitives: `_new_presentation`, `_blank_slide`,
  `_fill_slide_bg`, `_add_textbox`, `_add_rect`, `_add_table`,
  `_add_bar_chart`, `_add_pie_chart`, `_add_footer`.
- Deck skeletons: `_slide_cover`, `_slide_toc`, `_slide_section_divider`,
  `_slide_body`, `_slide_body_split`, `_slide_disclaimer`.
- Pitchbook builders: `_build_mna_pitchbook`, `_build_dcf_pitchbook`,
  `_build_lbo_pitchbook`, `_build_ipo_pitchbook`, `_build_comps_pitchbook`,
  `_build_precedents_pitchbook`.
- `InstitutionalPresentationGenerator` orchestrates; `get_presentation_generator()`
  singleton; `show_presentation_generator()` UI.
- `_codename` generates deal codenames for slide headers.

### Model Audit (`model_audit.py`)

Programmatic QA for financial models (346 lines):

- `QACheck` (label, status PASS/FAIL/WARN, message), `QAReport`.
- `ModelAuditor` — runs financial-model sanity checks (internal
  consistency, formula recomputation to 1e-6 via `_approx`, reasonableness
  gates).
- `run_model_qa(result, module)` — entry for any module.
- Used by the model generator UI's `_render_qa`.

---

## Derivatives & Commodities

### Options Engine (`options_engine.py`)

Neural + analytic options intelligence (1,180 lines).

- **Pricing models**: `black_scholes` (with Greeks),
  `bjerksund_stensland` (American approx), `monte_carlo_price`,
  `binomial_tree_price`, `finite_difference_american`, `longstaff_schwartz_american`,
  `merton_jump_diffusion_price`, `heston_price`, `barrier_option_price`,
  `asian_option_geometric`, `binary_option_price`, `lookback` (in futures
  engine), `price_variance_swap`.
- **Vol surfaces**: `sabr_vol`, `svi_vol`, `calibrate_svi_surface`,
  `estimate_skew`, `iv_rank_percentile`, `analyze_iv_term_structure`,
  `model_surface_dynamics`, `sticky_delta_adjustment`, `estimate_vix_implied`.
- **Greeks**: full surface generators (`generate_greeks_surface`,
  `generate_greek_surface`), `aggregate_portfolio_greeks`, `dual_delta`,
  `dual_gamma`, `calculate_skew_sensitivity`, `analyze_strategy_skew_risk`.
- **Strategies**: `construct_straddle`, `construct_professional_strategy`
  (iron butterfly etc.), `find_optimal_hedging`, `find_optimal_strike`,
  `get_strategy_pnl_map` (with `_breakevens`), `predict_option_edge`,
  `compute_option_market_metrics`.
- **Risk**: `calculate_var_cvar`, `historical_stress_test`,
  `comprehensive_stress_report`, `generate_hedging_signals`,
  `calculate_transaction_cost_impact`.
- **Neural augmentation**: `_ensure_nn` trains a small PyTorch net on a
  synthetic Black-Scholes training set (`_build_synth_train_set`) and
  `_predict_nn` blends its output with the analytic price (state cached to
  `_nn_cache_path()`).

Singleton: `get_options_engine()`.

### Options Simulation Grader

`options_simulation_grader.py` — `OptionsPerformanceGrade` (0-100 each):
delta_management, theta_efficiency, convexity_utilization + grading logic
for derivative-based performance.

### Futures Engine (`futures_engine.py`)

Comprehensive futures/commodities analytics (1,127 lines):

- Pricing: `black76`, `whaley_american_futures`, `implied_vol`,
  `lookback_futures_option_price`.
- Term structure: `schwartz_smith_term_structure`, `analyze_term_structure`,
  `simulate_term_structure`, `nelson_siegel_svensson`, `fit_nss_curve`,
  `physical_carry_model`, `implied_convenience_yield`,
  `theory_of_storage_analysis`.
- Positioning: `get_cot_positioning` (CFTC data), `cot_positioning_signal`,
  `get_seasonal_bias`.
- Spreads: `get_spread_analytics`, `optimize_spread_ratio`,
  `compute_roll_yield_optimization`, `get_calendar_spread_pnl`,
  `get_spread_pnl_map`, `analyze_complex_spread`, `analyze_energy_arbitrage`,
  `calculate_spark_spread`, `systematic_roll_optimization`.
- Risk/sizing: `estimate_margin`, `position_size_for_risk`,
  `calculate_portfolio_var`, `estimate_slippage_cost`, `physical_risk_overlay`,
  `carbon_intensity_adjustment`, `inflation_hedging_score`.
- Allocation: `black_litterman_allocation`, `generate_momentum_signals`,
  `generate_mean_reversion_signals`, `calculate_optimal_allocation`,
  `calculate_rebalance_trajectory`, `calculate_pnl_attribution`,
  `generate_factor_scores`.
- Execution: `compute_optimal_execution_trajectory`,
  `simulate_hjm_forward_rate_paths`, `calculate_shipping_impact`,
  `model_macro_sensitivity`, `full_commodity_analysis`,
  `run_engine_diagnostics`.

Singleton: `get_futures_engine()`.

### Futures Simulation Grader

`futures_simulation_grader.py` — futures-performance grading.

### Commodities Engine (`commodities_engine.py`)

Commodities analysis (335 lines): `CommodityAnalysis` (trend, momentum,
seasonality, RSI, vol, spreads), `SpreadAnalysis`;
`CommoditiesEngine` with `analyze_commodity`, `analyze_all(sector)`,
`analyze_spread`, `get_seasonal_calendar`, `get_universe`/`get_sectors`/
`get_symbols_by_sector`, `_calc_rsi`, `is_commodity`. Singleton
`get_commodities_engine()`.

---

## Institutional & Alternative Data

### SEC 13F Engine (`sec_13f_engine.py`)

Institutional holdings tracker (1,621 lines) covering equities, options
(calls/puts), bonds/fixed income, commodities, ETFs via SEC 13F filings.

**Data models:** `OptionPosition`, `BondPosition`, `CommodityPosition`,
`PositionChange`, `InstitutionalFiling` (with `provenance_label()`:
OBSERVED / SIMULATED / UNAVAILABLE).

**Pipeline:**

- `_fetch_from_edgar(cik)` — real EDGAR fetch (JSON, cached to
  `.edgar_filings_cache.json` via `_load_edgar_cache` / `_save_edgar_cache`).
- `_parse_infotable` — parses EDGAR info tables; `_resolve_symbol` resolves
  issuer/CUSIP/tickers → symbol with confidence; `_normalize_name`;
  `_load_company_ticker_map`.
- `_build_filing_from_edgar` — assembles an `InstitutionalFiling`.
- **Simulated fallbacks** (clearly labeled SIMULATED): `_mock_renaissance`,
  `_mock_citadel`, `_mock_bridgewater`, `_mock_berkshire`, `_mock_point72`,
  `_mock_two_sigma`, `_mock_millennium`, `_mock_de_shaw`, `_mock_tiger_global`,
  `_mock_appaloosa`, `_mock_generic` — deterministic mock filings for known
  funds when EDGAR is unreachable.
- `fetch_latest_filings(limit, allow_simulated)` — orchestrator;
  `_empty_unavailable_filing` for the no-data case.

**Analytics:** `generate_ai_insights(filing)`, `get_global_smart_money_flow`,
`get_cross_fund_options_flow`, `get_bond_market_positioning`,
`get_commodity_exposure`, `get_asset_class_flows`,
`get_options_sentiment_summary`. Singleton: `get_sec_13f_engine()`.

### Institutional 13F UI (`institutional_13f_ui.py`)

The 13F analysis tab (1,104 lines), `render_13f_analysis_tab()` with tabs:
`_render_tab_global_flow` (global smart-money flow), `_render_tab_options`
(cross-fund options flow), `_render_tab_bonds` (bond positioning),
`_render_tab_commodities`, `_render_tab_fund_deep_dive` (per-fund
holdings), `_render_tab_cross_fund` (overlap analysis).

### Dark Pool Engine (`dark_pool_engine.py`)

Institutional-grade off-exchange / dark-pool analytics (1,858 lines) with a
strict **data integrity model**: never fabricates exchange-reported data;
never claims modeled estimates are observed facts; every metric carries a
provenance (`Provenance` dataclass: source, method, observed flag,
confidence; `provenance()` factory).

**FINRA OTC integration:**

- `fetch_finra_otc(week_start)` — authoritative off-exchange volume from the
  FINRA OTC Transparency API (`_fetch_finra_uncached`, `_normalize_finra_df`,
  `_finra_partitions` for the weeklySummary dataset, `_latest_full_week` for
  week-partition discovery). API key configurable in-app (stored in
  `dark_pool_state.json`, gitignored): `get_finra_api_key`/`set_finra_api_key`,
  `get_settings`.
- `_finra_volume_map` — FINRA volume per symbol.
- `finra_metadata` — data quality/metadata.

**Modeled analytics (clearly labeled MODELED/INFERENCE):**

- `model_offexchange_share(df, sector)` — off-exchange share estimation with
  `offexchange_baseline(sector)` calibration.
- `analyze_ticker(symbol, period)` — full per-ticker report:
  `_market_cap`, `_live_quote`, `_model_largest_prints`, `_historical_context`,
  `_signal_engine` (signal generation: high_share, surge, accumulation,
  distribution, etc.), `_institutional_inference`, `_price_relationship`.
- `scan_market(limit)` / `get_scan_universe` — market-wide dark pool scan.
- `sector_analysis(limit)` — sector-level aggregation.
- `detect_regime` — off-exchange regime detection.
- `backtest_signal(symbol, condition)` — signal backtesting.
- `ai_insight(report)` — LLM-style insight text from report.
- `data_quality_report()` — integrity report; `methodology()` — the
  methodology disclosure string.

**Watchlists & alerts:** `get_watchlists`, `create_watchlist`,
`add_to_watchlist`, `remove_from_watchlist`, `delete_watchlist`,
`get_alerts`, `add_alert`, `remove_alert`, `evaluate_alerts` (with
`_evaluate_alert` and `_alert_key` dedup) — persisted in `dark_pool_state.json`.

Singleton: `get_dark_pool_engine()`.

### Dark Pool UI (`dark_pool_ui.py`)

The dark pool dashboard (1,600 lines), `show_dark_pool_dashboard()` (alias
`render_dark_pool_dashboard`). Design principle: data integrity over visual
polish — every number traceable to a source with an OBSERVED / MODELED /
INFERENCE label and confidence. Includes the FINRA key settings panel,
watchlists, alerts, per-ticker reports, market scan, sector analysis,
regime, backtests, and methodology disclosure.

### Alternative Data Engine (`alternative_data_engine.py`)

Non-traditional signal generation (1,445 lines):

- `AltDataSignal` (name, category, direction, strength, description — NOT
  `signal_type`; the portal display was fixed to use real fields),
  `SatelliteProxy`, `SocialSentimentSnapshot`.
- Deterministic seeded RNG per ticker (`_rng(ticker, salt)`) so signals are
  reproducible per symbol.
- Signal sources: `get_satellite_signals` (shipping / oil storage / retail
  parking proxies), `get_social_sentiment` + `get_social_signal` (WSB,
  Twitter/X mentions), `get_hiring_signal` (job-posting growth,
  `_estimate_job_categories`), `get_web_traffic_signal`, `get_credit_card_signal`,
  `get_esg_signal`, `get_dark_pool_signal`, `get_options_flow_signal`.
- Aggregates: `get_all_signals(ticker)` (all sources),
  `get_composite_score(ticker)` (weighted composite + verdict).
- Optional MASSIVE API path (`_get_close_from_massive`).

Singleton: `get_alt_data_engine()`.

### Institutional Analytics (`institutional_analytics_engine.py`)

Regime, macro/micro, Bayesian-network analytics (808 lines):

- `BayesianNode` / `BayesianNetwork` — custom Bayesian network with
  `propagate(shock_node, shock_value)` for shock transmission;
  `build_bayesian_network_from_data`, `get_layer_nodes`, `get_edges`.
- `run_macro_analysis(...)` → `MacroRegimeResult`; `run_micro_analysis(...)`
  → `MicroRegimeResult`; `detect_regime(...)` → `RegimeAnalysis`;
  `generate_market_scenarios(...)` → `MarketScenario` list;
  `get_institutional_summary()`.

### Institutional Visualizations (`institutional_visualizations.py`)

Plotly chart builders for institutional analytics (558 lines): regime
charts, scenario fan charts, Bayesian network diagrams, macro/micro
dashboards.

### Discovery Engine (`octavian_discovery_engine.py`)

High-velocity market scanning (312 lines): fans out its own fetches
internally (referenced by the background-task docs) to discover
opportunities across the universe.

---

## Document & Chart Intelligence

### Document Analyzer (`document_analyzer.py`)

Financial document intelligence — analyzes uploaded documents and SEC
filings as a strict, integrity-first data validation engine (1,265 lines):

- `FinancialDocumentParser` — parses uploaded financial documents
  (statements, reports, filings).
- `_fcf_locked(cfo, capex)` — FCF = CFO − capex (locked formula).
- `apply_numerical_sanity_gate(vals)` — validates parsed numbers, returns
  corrected values + warnings.
- `classify_real_financial_risks(vals, implied_fcf, text)` — risk
  classification from actual financials.
- `enforce_score_spreads(company_stats)` — score-spread enforcement.
- `DocumentIntelligenceEngine` — the main analysis engine (line 370):
  document → metrics → scores → report with AI-style narrative.
- `PortfolioAllocationEngine` (line 1022) — allocation recommendations
  from document contents.
- `_fetch_news_context` / `_fetch_market_context` — supplementary context.
- `show_document_analyzer()` — UI.

### Comparative Analysis

**Engine (`comparative_analysis_engine.py`, 783 lines):** institutional,
cross-asset comparison system. Supports Equities, Crypto, Commodities,
Forex, Options, Futures. `detect_asset_type(symbol)` auto-detects class;
`AssetScore` / `CrossAssetStrategy` / `ComparativeResult` models;
`_analyze_single_asset` (multi-model: quant/fundamentals/macro/technical);
`_calc_rsi`, `_calc_macd_signal`, `_calc_support_resistance`,
`_calc_technical_score`; `_build_correlation_matrix`; `_generate_strategies`
+ `_enrich_strategies_with_ai` (AI-written strategy theses via LM Studio);
`run_analysis(...)` orchestrator with parallel async fetches (2 retries).
Singleton `get_comparative_engine()`.

**UI (`comparative_analysis_ui.py`, 872 lines):** unlimited assets with
auto-detect + manual type override per symbol; custom model mode selection;
side-by-side metric cards (`_render_asset_cards`), strategy cards
(`_render_strategy_card`), `render_comparative_analysis()` entry.

### Fundamental Analyzer (`fundamental_analyzer.py`)

Fundamentals analysis (251 lines): ratios, growth, margins, returns from
yfinance info — used by the model generator and comparative analysis.

### Master Strategy (`master_strategy_engine.py`)

Dominant market outlook + regime detection (199 lines): synthesizes
multiple signals into one master view (outlook, regime, positioning
bias, key risks).

### Strategy Intelligence (`strategy_intelligence_engine.py`)

`StrategyIntelligenceEngine` — institutional strategy grader and suggester:
`grade_strategy(strategy_name, performance_data, parameters)` → letter grade
(A-F) with score and feedback; used by the Strategy Research Lab grader tab.

### Information Weighting (`information_weighting_engine.py`)

Weights information sources by reliability and recency for composite
signals (393 lines) — used to blend model/news/sentiment into unified
views.

---

## User & Personalization Layer

### Trader Profile (`trader_profile.py`)

The comprehensive personalization system (1,255 lines):

- Profile persistence: `save_profile` / `load_profile` (JSON per profile id,
  `_profile_file`), `get_trader_profile`, `list_profiles`.
- `get_asset_classes()`, `get_watchlist_presets()` — option sources.
- `learn_from_user_interaction(query, response)` — implicit learning.
- Context helpers: `get_timeframe_context_for_analysis`,
  `get_recommendation_style`, `get_watchlist`, `get_risk_params`,
  `get_preferred_assets`, `should_show_advanced`.
- UI: `show_trader_selection(key_suffix)` (sidebar selector),
  `_render_watchlist_editor`, `show_profile_settings()` (the Trader Profile
  page), `_render_profile_card`.
- `show_personalized_dashboard()` — the Dashboard "My View" tab:
  watchlist signal snapshot (`_fetch_watchlist_signal_snapshot`),
  `generate_market_aware_insights(watchlist_signals)` — dynamic, non-
  hardcoded takeaways computed from live data (aligned with profile style,
  risk tolerance, watchlist, timeframe; 15-min cache via
  `dashboard_intelligence_engine`).

### Custom Dashboard (`custom_dashboard.py`)

The Symbol Analysis page (1,022 lines): `show_custom_dashboard()` — enter
any symbol → multi-asset analysis with technical indicators, charts,
news, risk, ML prediction, and recommendations. Uses `get_fresh_quote`
when available (`_HAS_FRESH` guard).

### Watchlist Dashboard (`watchlist_dashboard.py`)

Live monitoring + deep-dive for the user's watchlist (778 lines):
`show_watchlist_dashboard()` — watchlist editor, live quote grid (batched
fetch via ThreadPoolExecutor), per-symbol deep-dive expanders, alerts.

### Integrated Market System (`integrated_market_system.py`)

`IntegratedMarketSystem` orchestrator + `show_integrated_dashboard()`
(505 lines): system overview (`show_system_overview`), advanced features
(`show_advanced_features`). The older all-in-one dashboard shell.

### Dashboard Intelligence (`dashboard_intelligence_engine.py`)

Personalized dashboard intelligence (411 lines): generates dynamic,
real-time, non-hardcoded market takeaways aligned with the user's trader
profile — all computed fresh from live market data per call, with a
15-minute cache.

### Auth Engine (`auth_engine.py`)

Authentication: bcrypt `hash_password` / `verify_password`; `authenticate_user`
(username/password against `octavian_users.db` via `db_manager`),
`show_auth_page()` (login), `show_onboarding_page()` (registration with
ToS acceptance).

### Terms of Service

- `terms_of_service.py` — `show_terms_of_service()`: informational-only for
  accounts created after the `tos_accepted` migration; for legacy accounts
  (tos_accepted = 0) shows a checkbox and `_persist_tos_acceptance(user_id)`
  writes acceptance immediately.
- `terms_of_service_ui.py` — `render_terms_of_service()`: full legal
  policy view (risk disclaimers, simulated-trading liability shield).

### Notification Engine (`notification_engine.py`)

Notification system (4771 bytes): in-app notifications for alerts,
background completions, and price events; `notification_settings_ui.py`
renders the settings page (`show_notification_settings`).

---

## Infrastructure & Services

### API Backend (`api_backend.py`)

Flask REST API (807 lines) — alternative non-Streamlit interface:

- Endpoints: `GET /health` (`health_check`), `GET /api/system/status`,
  `GET /api/market-data/<symbol>`, `POST /api/multiple-quotes`,
  `GET /api/analyze/<symbol>`, `POST /api/chat`, `GET /api/predictions/<symbol>`,
  `GET /api/predictions/advanced`, `GET/POST /api/user/profile`,
  `GET /api/conversations/<session>`, `GET /api/analytics/dashboard`,
  `POST /api/admin/cleanup`, `GET /api/admin/stats`.
- `require_api_key` decorator protects endpoints (bearer token from
  `API_BACKEND_KEY`); `log_api_request` records usage; error handlers
  (`not_found`, `rate_limit_exceeded`, `internal_error`) return JSON.
- CORS restricted to `CORS_ORIGINS`; sessions signed with `FLASK_SECRET_KEY`.

### Real-Time Data Service (`realtime_data_service.py`)

WebSocket + polling real-time service (638 lines):

- `PriceAlert` / `MarketEvent` models; `RealTimeDataService` with
  background tasks: `update_market_data` (polling loop),
  `check_price_alerts` (threshold + volume-spike + sentiment-change
  alerts), `detect_market_events`, `cleanup_cache`.
- `get_metrics()` (throughput, freshness), `stop()`; default watchlist via
  `_build_default_watchlist`. Singleton `get_realtime_service()`.

### WebSocket Engine (`websocket_engine.py`)

Optional `websockets`-based live price feed (`HAS_WEBSOCKETS` guard):
`WebSocketEngine` streaming prices; `get_websocket_price(symbol)` cached
lookup (`_LIVE_PRICE_CACHE`); `get_engine()` singleton.

### Database Manager (`database_manager.py`)

Advanced DB layer for `market_ai_system.db` (768 lines):

- `_initialize_databases` — full schema (market_data, technical_indicators,
  ml_predictions, conversations, system_metrics, query_analytics,
  user_activity, user_profiles, news_articles, data_quality).
- Store/query: `store_market_data`, `get_market_data`,
  `store_technical_indicators`, `store_ml_prediction`, `store_conversation`,
  `get_conversation_history`, `log_system_metric`, `log_query_analytics`,
  `get_analytics_dashboard`, `log_user_activity`, `get_user_activity`,
  `create_user_profile`, `get_user_profile`, `cleanup_old_data`,
  `get_database_stats`.
- Singleton `get_database_manager()`.

### Analytics Dashboard (`analytics_dashboard.py`)

The Settings & Analytics page (303 lines): `show_analytics_dashboard()`
with tabs — overview metrics, conversation analytics, ML performance,
data quality — all driven by `DatabaseManager.get_analytics_dashboard()`.

### Backend Package (`backend/`)

The newer backend scaffolding:

- `backend/core/config.py` — centralized config loaded from env vars with
  defaults.
- `backend/core/tiers.py` — subscription tier definitions (feature gates,
  rate limits, capabilities per tier) used by middleware and the frontend
  for upgrade prompts.
- `backend/api/`, `backend/auth/`, `backend/middleware/` — package
  structure for the REST API/auth/middleware layers.

### Data Downloader (`data_downloader.py`)

`DatasetDownloader` — downloads NASDAQ, Kaggle, and other free historical
datasets for offline training/testing (`download_datasets()`).

### Historical Data Engine (`historical_data_engine.py`)

Multi-vendor historical financials/prices (433 lines):

- `get_historical_financials(ticker, years=10)` — fundamentals history via
  EODHD (preferred) or yfinance (`_fetch_from_eodhd` / `_fetch_from_yfinance`).
- `get_historical_prices(ticker, years=10)` — prices via CSV, EODHD, or
  yfinance (`_fetch_price_from_csv` / `_fetch_price_from_eodhd` /
  `_fetch_price_from_yfinance`).
- `_determine_exchange`, `_cache_data` (diskcache), `_get_empty_historical`.
- Singleton `get_historical_engine()`.

### Market Data Cache (`market_data_cache.py`)

Thin diskcache-backed market data cache used by historical fetches.

---

## Small Utility Modules (catch-all)

A handful of small helper modules used across the platform:

| Module | Purpose |
|---|---|
| `indicators.py` | `add_indicators(df)` — EMAs, Bollinger Bands, RSI etc. appended to a price frame (the classic technical-indicator bag used by legacy backtests and charts). |
| `correlation.py` | `correlation_matrix(price_series_dict)` — return-correlation matrix from a dict of price series. |
| `cot.py` | `load_cot()` — CFTC Commitments of Traders disaggregated futures data (zip download from cftc.gov). |
| `hmm_engine.py` | `MarketRegimeEngine`-style GMM-based regime detection (Bull/Bear/Sideways/Volatile) — the `hmm_engine` used by regime analysis. |
| `futures_leaderboard.py` | `FUTURES` map (ES=F, NQ=F, CL=F, GC=F, ...) for futures leaderboards/quick lists. |
| `manual_portfolio_system.py` | `ManualPortfolioSystem` — user-defined manual portfolios persisted to `manual_portfolios.json` (consumed by the portfolio chatbot context). |
| `models.py` | `train_models(X, y)` — RandomForest + XGBoost classifiers (300 estimators, balanced) for legacy ML predictions. |
| `llm_stress_test.py` | LLM entity-resolution & decomposition stress harness: generates tens of thousands of prompts from combinatorial templates and verifies the financial LLM's entity-resolution firewall invariants; `main()` CLI. |
| `chart_image_analyzer` note | See §12 Chart Image Analyzer. |

Also present: `test_lm_studio_interpreter.py` (unit tests for the LM Studio interpreter fallback), `bot_eval/` companion scripts, and `ticker_universe_cache.json` / `.edgar_filings_cache.json` runtime caches.

---

## End-to-End Process Walkthroughs

This section traces the most important end-to-end flows through the
codebase — the actual journey of a user action from UI click to output.

### Walkthrough 1: Asking the AI Assistant a question

1. User opens **Intelligence Center → Octavian AI Assistant** and types a
   question like "Analyze NVDA and tell me if it's a buy".
2. `main.py` routes to `ai_chatbot.show_octavian_chatbot()` (line 5577).
3. The UI calls `OctavianEnhancedChatbot.process_query(query, user_id)`
   (line 2497).
4. `process_query`:
   - Runs `intent_engine.analyze_intent(query)` → `IntentAnalysis`
     (primary intent, format, detail level, urgency).
   - Extracts symbols via `_extract_symbols` (with the entity-resolution
     firewall protecting against CAPEX/FCF/GPU-style false tickers).
   - Determines timeframe scope via `_detect_intent_and_timeframe`.
   - For each symbol runs `_analyze_symbol_unbiased` /
     `_analyze_symbol_comprehensive` → fetches OHLCV via
     `_get_real_time_data` (through `data_sources`), computes enhanced
     metrics (technical signals, risk metrics, price levels), pulls news
     via `_aggregate_news_analysis` (through `news_analysis_engine`),
     sentiment, and the `unbiased_analyzer` profit probability.
   - Assembles the response: user-facing text via `response_formatter`
     (FormattedResponse.user_text) + `raw_analysis` (full per-symbol
     detail for the "Show AI Reasoning" expander).
5. UI renders charts (price + prediction via plotly), the formatted text,
   follow-up suggestions, and the expandable reasoning.

### Walkthrough 2: Institutional deep-dive memo

1. User asks for a deep dive (e.g. "institutional deep dive on NVDA" or
   any query matching `_is_deep_dive_query`).
2. `generate_financial_analysis` routes to `_build_institutional_deep_dive`
   (line 3392).
3. The assembler: fetches price + fundamentals (REPORTED data via
   `fetch_ticker_fundamentals`, offline-guarded), stamps the price with
   quote_date + source, computes the data-completeness gate, then calls the
   20 section builders in order (§1-§20), each returning provenance-tagged
   text/table sections.
4. Key numerics: `_dd_real_dcf` runs the site's own
   `InstitutionalDCFEngine.compute_wacc` + `project_fcf` and transports the
   20-line projection table into the memo; `_dd_reverse_dcf` bisects on
   the DCF equation for implied growth/margin; scenarios, Monte Carlo
   percentiles, Bayesian posteriors, and the QC audit all recompute
   independently and cross-check to 1e-6.
5. Result: a single markdown string rendered in the chat UI with the
   AUDIT verdict, DCF status (BLOCKED/PARTIAL/PASS), and confidence <70%.

### Walkthrough 3: Building and deploying an algorithm

1. User opens **Algorithm Builder**, picks Auto mode, types "mean
   reversion on tech stocks", hits Build.
2. `render_algorithm_builder()` → `_build(build_args, progress)` →
   `build_algorithms(request, mode="auto", ...)`.
3. Engine: `parse_request` maps keywords → families (rsi_meanrev,
   bollinger_meanrev, statarb_z, ...); data fetched per symbol (retry
   once on outage, ≥60 bars); each family searched `runs_per_family`
   times (distinct seeds) → `backtest_ohlcv` → `compute_metrics` →
   window breakdown → `trade_narratives` → `AlgorithmResult` (trades
   stamped with symbol + asset_type).
4. Optional: walk-forward CV (wf_folds) and/or dynamic ensemble
   (quality-tilted + redundancy-penalized weights). Failures get
   `_error_result` with `diagnose_failure` suggestions.
5. UI renders each result card (equity chart, metrics, SWOT, per-trade
   reasoning, window table, trade table, factor ranking).
6. **Deploy**: user clicks "Test in paper trading" → `_render_deploy_panel`
   → `paper_trading_system.deploy_strategy(account_id, spec)` (merges
   spec into automation_config; override checkbox required when a strategy
   already exists).
7. In **Paper Trading → Automation Controls**, user starts automation;
   `automated_trading_engine.start_automation(account_id, rr,
   strategy_spec=spec)` spawns the worker loop; `_strategy_signal`
   recomputes target exposure on fresh OHLCV; positions sized by ATR risk
   budget; `_manage_existing_positions` enforces stops/targets; all trades
   logged with reasoning.

### Walkthrough 4: Background task (Breaking Trades scan)

1. User triggers a scan on Dashboard or Paper Trading. The UI calls
   `_launch_background(name, result_key, fn, ...)` in `main.py`.
2. `background_tasks.submit_task` registers the record and submits to the
   4-worker pool; the script returns immediately.
3. The `@st.fragment(run_every=15)` `_bg_poller` runs every 15s calling
   `_check_background_tasks(force_rerun=True)`; `drain_completed` returns
   finished-but-unnotified tasks exactly once, publishes results into
   `st.session_state[result_key]`, toasts the user, and reruns the app so
   the page showing results updates immediately.
4. Sidebar `_render_background_status` shows running/done/failed tasks
   with elapsed time.

### Walkthrough 5: Dark pool analysis

1. User opens **Dark Pool Intelligence**, selects a ticker.
2. `dark_pool_ui.show_dark_pool_dashboard()` →
   `dark_pool_engine.analyze_ticker(symbol, period)`.
3. Engine fetches OHLCV (`_fetch_ohlc`), optionally pulls authoritative
   FINRA OTC volume (`fetch_finra_otc` via the configurable API key), then
   models off-exchange share (`model_offexchange_share` with sector
   baseline), generates signals (`_signal_engine`: high_share, surge,
   accumulation, distribution), infers institutional activity
   (`_institutional_inference`), and checks price relationships.
4. Every metric carries a `Provenance` (OBSERVED/MODELED/INFERENCE) and
   confidence; the UI labels each accordingly, and `methodology()` /
   `data_quality_report()` disclose exactly what is modeled vs observed.
5. Watchlists/alerts can be set; `evaluate_alerts` fires on new reports.

### Walkthrough 6: Financial model → Excel + pitchbook

1. User opens **Financial Model Generator → DCF tab**, enters assumptions.
2. `InstitutionalDCFEngine` projects FCF, computes WACC (CAPM), builds
   scenarios, sensitivity grid, reverse DCF.
3. `_render_qa` shows the `ModelAuditor` QA report; `_render_memo_button`
   exports an investment memo (`InvestmentMemoGenerator`);
   `_render_pptx_button` exports a pitchbook
   (`InstitutionalPresentationGenerator._build_dcf_pitchbook`); Excel
   via `ib_excel_engine.build_dcf_workbook`.
4. **Valuation Bridge** tab aggregates DCF + Comps + Precedents + IPO +
   LBO + M&A implied values, each weighted by relevance/reliability.

---

## Test Suite

Run: `python3 -m pytest tests/ -q` — **758 tests pass** (as of 2026-08-22
session). Coverage by file:

| Test file | Tests | Focus |
|---|---|---|
| `test_algorithm_builder.py` | 58 | Archetypes, ensembles, metrics math, trade stamps, walk-forward, generated-code compile+run, strategy_spec round-trip |
| `test_deep_dive_integrity.py` | 53 | All 20 memo sections, provenance, DCF gating + CONDITIONAL labeling, reverse DCF, exp-return/drawdown consistency, QC audit, determinism, fuzz matrix |
| `test_deep_dive_review_fixes.py` | 24 | Review-fix iteration: 4-chart engine (distinct analytical jobs, metadata, no-reuse hash), display-precision scenario math, same-variable disagreement, DCF formula labels, valuation A/B/C/D separation, directional validation, mechanical rating/confidence, SUBJECTIVE Bayesian, modeled-vs-empirical language, macro gating, QC16-21 firewall |
| `test_dark_pool_engine.py` | 45 | Provenance model, FINRA fetch, signals, alerts, backtest, methodology |
| `test_institutional_suite.py` | 32 | 13F, options, dark pool, institutional engines |
| `test_audit_regressions.py` | 29 | Model QA / audit regressions |
| `test_integration_flows.py` | 24 | End-to-end bridges (build→deploy→automate) |
| `test_fundamental_analyzer.py` | 20 | Fundamentals pipeline |
| `test_ib_models.py` | 16 | IB Excel workbook builders |
| `test_background_tasks.py` | 11 | Task registry, exactly-once drain, bounded memory |
| `test_financial_model.py` | 11 | DCF/LBO/M&A engine math |
| `test_chatbot_prompt_coverage.py` | 10 | Prompt corpus coverage |
| `test_normalize_options_chain.py` | 10 | Options chain normalization |
| `test_paper_trading_system.py` | 9 | Accounts, trades, P&L, strategy deployment |
| `test_ui_walkthrough.py` | 9 | Streamlit AppTest click-throughs (portal quick-select, alt-data, automation) |
| `test_advanced_news_processor.py` | 7 | News processing |
| `test_ai_chatbot.py` | 7 | Chatbot query paths |
| `test_llm_and_models_fixes.py` | 69 | LLM/model fix regressions |
| `test_tool_router.py` | 28 | Tool router: detection, runners, end-to-end chat wiring |
| `test_comparative_analysis.py` | 4 | Comparative engine |
| `test_portfolio_analyzer.py` | 4 | Portfolio metrics |
| `test_background_tasks_streamlit.py` | 3 | Streamlit integration |
| `verify_simulation_universe.py` | 3 | Simulation universe sanity |
| `test_before_start.py`, `test_fx_fix.py`, `verify_dynamic_system.py` | 0-3 | Startup/verification scripts |

Verification steps used on every change: full pytest run, boot HTTP 200
check, pyflakes clean of new issues.

---

## Project Conventions & Gotchas

**Conventions:**

- **No emojis** in the UI or docs (institutional aesthetic) — swept out of
  the site UI and docs.
- **Provenance-first data integrity** — never fabricate observed data;
  modeled estimates are labeled MODELED/INFERENCE with confidence.
- **Module-level singletons** for engines (`get_*_engine()` accessors) so
  state survives Streamlit reruns.
- **Lazy imports** inside `main.py` page branches for startup speed.
- **Deterministic engines** — no randomness in analysis; tests verify
  byte-identical output for identical inputs.
- **Secrets via env / Streamlit secrets only** (`config._get_secret`),
  never hardcoded.
- **Data corrections as data** (`data_corrections.json`), not code.

**Gotchas (learned the hard way):**

- Streamlit reruns the whole script: widgets with session-state values
  cannot also set `value=` (DuplicateElementId / write-after-instantiation
  errors); use `on_click` callbacks + `setdefault` for quick-select.
- `AdvancedBacktester.__init__` takes ONLY `initial_capital` (no `symbol=`).
- `AltDataSignal` fields are `name`/`category`/`direction`/`strength`/
  `description` (no `signal_type`).
- yfinance's module-level `download()` is NOT thread-safe — serialize via
  `data_sources._YF_FETCH_LOCK`.
- `scipy.optimize.fsolve` reverse-DCF in `InstitutionalDCFEngine` is broken
  in this env (0-dim array error, silent default fallback) — the deep-dive
  memo uses its own bisection.
- `pandas.DataFrame.to_markdown()` needs optional `tabulate` (not
  installed) — use `financial_llm_engine._df_to_markdown`.
- pandas boolean-indexing on string columns raises ("iLocation based
  boolean indexing on an integer type") — compute best/worst on numeric
  values first.
- Streamlit `use_container_width` is deprecated (warning only).
- Quant ensemble needs ≥50 bars warmup — backtest lookback floored at 60.
- `_progress` callbacks must accept 3 args (gen, total, fitness).
- Background-task delivery failures must `reopen()` the task or the result
  is lost silently.
- `datetime.utcnow()` is deprecated (pre-existing warnings in
  dark_pool/news engines — scheduled for cleanup).
- `re.match` needs the string arg; `out.splitlines()[1]` is a blank line,
  not the price line (memo price-stamp parsing).

---

## Changelog

| Date | Change | Sections touched |
|---|---|---|
| 2026-08-21 | Master Doc created (full codebase walkthrough); deep-dive memo review-fix iteration: new `deep_dive_charts.py` (4 distinct analytical charts with chart_id/purpose/dataset/timestamp/provenance/calculation/run_id/dataset_hash + no-reuse registry, wired into `process_enhanced_query`), display-precision scenario engine (weighted return recomputes exactly from printed table, QC3b), same-variable market-vs-model disagreement (QC16), DCF exact formula labels + CONDITIONAL/ASSUMPTION-BASED status + A/B/C/D valuation separation (QC21), directional validation of reasons to own/not (QC18), mechanical risk-adjusted rating, confidence with fixed disclosed weights (QC17), SUBJECTIVE Bayesian label, modeled-vs-empirical probability language, macro current-claims gating (QC19), 13-point quantitative-integrity firewall in the LLM system prompt. 695 tests → 719. | AI & NLP Layer; Chart Engine; Test Suite |
| 2026-08-22 | Tool router (`tool_router.py`, ~450 lines): user prompts that explicitly ask for a built-in analytical tool are outsourced to the REAL engine and repackaged as the chatbot's answer — DCF (InstitutionalDCFEngine), reverse DCF (market-consensus implied expectations), Bayesian network (3-layer propagation), Markov/HMM regime (hmm_engine), dark pool (analyze_ticker + ai_insight), 13F smart-money flow, correlation matrix, options greeks (Black-Scholes), factor crowding, market regime. Fires only on explicit tool vocabulary (no false positives on hedge/probability/options/hmm-filler); multi-tool queries combine sections; every runner degrades to None on missing data. Wired into all three chat paths: `generate_financial_analysis`, `_build_single_answer` (mega parts), and `ai_chatbot.process_query`. 730 tests → 758 (28 new in `test_tool_router.py`). | AI & NLP Layer; Test Suite |
| 2026-08-21 | Deep-dive retest review (round 2, ~8.3/10) fixes: (1) charts now carry an explicit "CONDITIONAL DCF — NOT VERIFIED FCF VALUATION" banner + "Cond. DCF" bar labels on expectation-gap and valuation-sensitivity charts; (2) risk/reward chart rebuilt as a metrics table that directly matches the memo text (weighted return, weighted price, current price, upside/downside ranges, modeled drawdown probs with empirical DATA UNAVAILABLE); (3) below-market DCF anchor moved from reasons-to-own to reasons-not-to-own (it was a negative implication listed as a reason to own); (4) QC18 no longer hardcoded — `_dd_build_reasons` + `_dd_semantic_violations` actually scan the reason lists and a violation flips QC18 to FAIL with a SEMANTIC QC FAILURES block; (5) every chart gains a `qc_status` computed by `_chart_qc` so the visualization inherits the memo's reconciliation/QC status; (6) eval prompts (`prompt_factory`, `llm_stress_test`) now request "MODELED probability … within defined scenarios" instead of bare drawdown probabilities. 719 tests → 730. | AI & NLP Layer; Chart Engine; Test Suite |
| 2026-08-19 | Deep-dive memo fixes (DCF gating + site-tool DCF, multi-variable reverse DCF, consistent expected-return/drawdown probs, operating-leverage scenarios, price timestamps, labeled macro, Bayesian basis, numeric QC audit); algorithm builder symbol/asset-type stamps, real ensemble trade counts, exact per-window Sharpe; portal quick-select buttons fill most-relevant assets. 694 tests → 695. | AI & NLP Layer; Algorithm Builder; Quant Portal; Test Suite |
| 2026-08-17 | Rebuilt institutional deep-dive memo to the 20-section research-integrity spec + extreme stress tests (43 → 48 tests). | AI & NLP Layer |
| 2026-08-15 | Quant portal error fixes; Algorithm Builder advanced backtests + dynamic ensemble; algorithm-builder → paper-trading deployment bridge. | Quant Portal; Algorithm Builder; Trading Infrastructure |

---

*End of Master Doc. This document is updated in the same session as every
code change — see [How to Keep This Document Current](#how-to-keep-this-document-current).*

