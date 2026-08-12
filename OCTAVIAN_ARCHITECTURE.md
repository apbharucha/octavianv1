# OCTAVIAN TERMINAL - Complete Architecture Document

> Institutional-Grade Multi-Asset Financial Intelligence Platform
> Version 4.0 | Built by APB

---

## Table of Contents

1. [Platform Overview](#platform-overview)
2. [Frontend Architecture](#frontend-architecture)
3. [Backend Engine Catalog](#backend-engine-catalog)
4. [Data Layer](#data-layer)
5. [ML & AI Systems](#ml--ai-systems)
6. [Trading Infrastructure](#trading-infrastructure)
7. [Visual Design System](#visual-design-system)
8. [Navigation & Routing](#navigation--routing)

---

## Platform Overview

Octavian Terminal is a Streamlit-based institutional-grade financial intelligence platform. It integrates real-time market data, deep learning predictive models, options/futures/commodities analytics, paper trading simulation, and AI-powered research into a unified terminal experience.

**Tech Stack:**
- **Framework:** Python 3.12 + Streamlit
- **ML:** PyTorch (LSTM, Transformer), Scikit-Learn (MLP, RF, GBM)
- **Data:** yfinance, OANDA API, Polygon API, Alpha Vantage
- **Visualization:** Plotly, Matplotlib
- **Database:** SQLite (local persistence)
- **LLM:** Local Mistral 7B via LM Studio (localhost:1234)

---

## Frontend Architecture

### Entry Point: `main.py`
- Central routing hub (~3,170 lines)
- Handles authentication, navigation, and page rendering via conditional imports
- Flow: Auth Gate -> Onboarding -> Sidebar Navigation -> Content Router

### Theme System: `octavian_theme.py`
- Dark mode, navy/gold palette, glassmorphism
- CSS keyframe micro-animations (slide-in, pulse, shimmer)
- Custom metric cards, section headers, gradient banners

### Navigation Pages (16 total):
1. Dashboard - Market overview, breaking trades, personalized view
2. Watchlist - Custom symbol tracking
3. Market Scanner - Sector/momentum screening
4. Symbol Analysis - Deep single-asset analysis
5. Chart Analysis - Image-based chart pattern recognition
6. Intelligence Center - AI chatbot + news dashboard
7. Market Heartbeat - Real-time pulse monitoring
8. Financial Model Generator - DCF, comparable analysis, Excel export
9. Daily Briefing - Automated morning intelligence report
10. Quant Portal - Backtesting, factor models, alternative data
11. Strategy Research Lab - Options strategies, genetic evolution, pairs trading
12. Paper Trading - Simulated execution environment
13. Simulation Hub - Monte Carlo, stress testing, scenario analysis
14. Spreadsheet Generator - Automated financial spreadsheet creation
15. Document Analyzer - SEC filing analysis with AI scoring
16. Position Optimizer - Kelly criterion, risk-adjusted sizing

---

## Backend Engine Catalog

### Core Engines

| Engine | File | Purpose |
|--------|------|---------|
| Master Strategy | `master_strategy_engine.py` | Dominant market outlook, regime detection |
| ML Ensemble | `advanced_ml_engine.py` | 5-model ensemble for price prediction |
| Unbiased Analyzer | `unbiased_market_analyzer.py` | Bias-free profit probability calculator |
| Options Engine | `options_engine.py` | Black-Scholes, Greeks, strategy builder |
| Futures Engine | `futures_engine.py` | Term structure, basis, roll yield |
| Commodities Engine | `commodities_engine.py` | Seasonal patterns, spreads, sector analysis |
| Risk Engine | `risk_engine.py` | VaR, correlation, position sizing |
| Financial LLM | `financial_llm_engine.py` | Local Mistral for NLP analysis |
| SEC 13F Engine | `sec_13f_engine.py` | Institutional flow tracking |
| Discovery Engine | `octavian_discovery_engine.py` | High-velocity market scanning |

### Analytics Engines

| Engine | File | Purpose |
|--------|------|---------|
| News Analysis | `news_analysis_engine.py` | Sentiment aggregation |
| Alternative Data | `alternative_data_engine.py` | Satellite, social, hiring |
| Counter-Trend | `counter_trend_analyzer.py` | Mean reversion signals |
| Institutional | `institutional_analytics_engine.py` | Regime, macro/micro |
| Factor Crowding | `factor_crowding_engine.py` | Factor exposure/risk |
| Narrative Dislocation | `narrative_dislocation_engine.py` | Price-narrative divergence |
| Source Credibility | `source_credibility_engine.py` | News source quality |
| Timeframe Analysis | `timeframe_analysis_engine.py` | Multi-TF alignment |

---

## Data Layer

### Sources
- **Equities:** Yahoo Finance (yfinance)
- **FX:** OANDA REST API
- **Futures/Commodities:** Yahoo Finance (=F suffix)
- **Crypto:** Yahoo Finance (-USD suffix)
- **Options:** Yahoo Finance options chain
- **News:** Multiple RSS/API feeds

### Databases
- `market_ai_system.db` - Conversations, analytics
- `octavian_simulations.db` - Simulations, backtests
- `octavian_users.db` - User accounts
- `octavian_paper_trading_state.json` - Paper positions

### Caching
- `@st.cache_data(ttl=600)` for market data
- `@st.cache_resource` for ML model singletons
- In-memory dict caches with TTL

---

## ML & AI Systems

### Ensemble (advanced_ml_engine.py)
- **8 Input Features:** Close, High, Low, Volume, Volatility, RSI, MACD, Institutional Flow
- **5 Models:** LSTM (25%), Transformer (25%), MLP (20%), RF (15%), GBM (15%)
- Online learning on recent data
- 13F institutional flow injected into training tensors

### LLM Pipeline (financial_llm_engine.py)
- Step 1: Query Parser - Mode + ticker extraction
- Step 2: Analysis Generator - Structured output
- Modes: MACRO_COMMODITIES, MACRO_GENERAL, SINGLE_STOCK, MULTI_ASSET, TRADE_IDEA

---

## Trading Infrastructure

### Paper Trading
- Virtual account with configurable capital
- Order types: Market, Limit, Stop-Loss, Trailing Stop
- Multi-asset: Equities, Options, Futures, Commodities, FX, Crypto
- Automated strategy execution from Research Lab

### Position Optimizer
- Kelly Criterion sizing
- Leverage handling (50x futures, 100x options)
- Portfolio correlation-aware sizing

### Portfolio Analyzer
- Sharpe, Sortino, Max Drawdown
- Cross-asset VaR (95% confidence)
- Stress tests (Lehman, COVID, Vol-mageddon)

---

## Visual Design System

### Color Palette
- Primary BG: `#0a1628` | Secondary BG: `#132240`
- Gold Accent: `#c9a84c` | Lavender: `#9b8ec4`
- Text: `#e8eaf0` | Success: `#4caf50` | Danger: `#ef5350`

### Principles
- No Emojis - Clean institutional typography
- Glassmorphism - Semi-transparent cards
- Micro-animations - CSS keyframes
- Dark Mode Only - Trading terminal aesthetic
- Inter font family with letter-spacing headers

---

*Document generated for Octavian Terminal v4.0 | May 2026*
