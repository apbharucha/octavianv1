# Octavian Platform Functionality Guide

## Purpose
This document explains what every major Octavian page does, how users interact with it, and what output to expect.

## User Flow (Top-Level)
1. User authenticates.
2. User chooses a page from sidebar navigation.
3. Page-specific engine loads and fetches required data.
4. UI presents metrics, charts, and actionable outputs.

## Page-by-Page Guide

### Dashboard
- **Goal:** real-time market context and high-level intelligence.
- **What user can do:**
  - View institutional outlook banner.
  - Refresh live index metrics.
  - Inspect market overview charts and movers.
- **Primary outputs:** index metrics, trend charts, summarized catalysts.

### Watchlist
- **Goal:** track chosen symbols quickly.
- **Actions:** view custom watchlist snapshots and updates.
- **Outputs:** symbol cards, movement and status summaries.

### Market Scanner
- **Goal:** discover opportunities from broad market scan logic.
- **Actions:** run scanners and filter ranked results.
- **Outputs:** ranked candidates with scores and supporting indicators.

### Symbol Analysis
- **Goal:** deep dive into one ticker.
- **Actions:** input symbol and trigger analysis.
- **Outputs:** multi-factor insights, technical/fundamental summaries.

### Chart Analysis
- **Goal:** visual pattern and chart signal interpretation.
- **Actions:** load chart data and request analysis.
- **Outputs:** chart overlays, indicator states, directional read.

### Intelligence Center
- **Goal:** integrated news + chat intelligence.
- **Actions:** browse news dashboard and query chatbot.
- **Outputs:** narrative summaries, symbol-level decision support.

### Market Heartbeat
- **Goal:** current macro/risk regime snapshot.
- **Actions:** inspect heartbeat status and context factors.
- **Outputs:** volatility regime, risk mode, environment classification.

### Financial Model Generator
- **Goal:** run valuation workflows (including DCF paths).
- **Actions:** input assumptions or auto-fetch values.
- **Outputs:** model outputs, valuation scenarios, assumptions panel.

### Daily Briefing
- **Goal:** consolidated daily summary for decision framing.
- **Actions:** generate/update briefing.
- **Outputs:** daily macro/market bullet summary.

### Quant Portal
- **Goal:** quantitative research and backtesting workspace.
- **Actions:** run strategy tests, risk analysis, and simulations.
- **Outputs:** equity curves, win/loss stats, VaR/CVaR diagnostics.

### Strategy Research Lab
- **Goal:** experiment with research ideas and framework combinations.
- **Actions:** configure strategy components and evaluate outcomes.
- **Outputs:** comparative strategy analytics and tradeoff summaries.

### Paper Trading
- **Goal:** simulate execution and track performance without live capital.
- **Actions:** place simulated trades, inspect account/positions.
- **Outputs:** PnL, account state, historical trade records.

### Simulation Hub
- **Goal:** run and view structured simulation sessions.
- **Actions:** select/view simulation runs and diagnostics.
- **Outputs:** scenario outcomes, viewer summaries.

### Spreadsheet Generator
- **Goal:** export analytics into structured spreadsheet artifacts.
- **Actions:** choose parameters and generate downloadable file.
- **Outputs:** tabular exports with configurable formatting.

### Document Analyzer
- **Goal:** parse financial text/documents into structured investment analysis.
- **Actions:**
  - Upload or paste one or multiple company documents.
  - Run integrity-first extraction/scoring pipeline.
- **Outputs:**
  - Validated financial summary
  - Parse errors vs real financial risk separation
  - FCF-locked scoring and conviction label
  - Multi-company relative allocation section

### Position Optimizer
- **Goal:** evaluate position quality and suggest optimized structures.
- **Actions:** enter symbol/entry/size/type, run optimizer.
- **Outputs:** alpha/risk/efficiency grades, transformation ideas, hedges.

### Portfolio Analyzer
- **Goal:** evaluate portfolio quality across performance, risk, and scenarios.
- **Actions:** run account analysis.
- **Outputs:** metrics (Sharpe, drawdown, VaR, diversification), scenario impacts, health grade.

### Institutional 13F Analysis
- **Goal:** inspect institutional holdings flow and signal implications.
- **Actions:** run filing/ownership analysis.
- **Outputs:** concentration changes, conviction interpretation.

### Trader Profile
- **Goal:** personalize behavior and profile-based preferences.
- **Actions:** tune profile + notification preferences.
- **Outputs:** updated strategy/style settings and alerts configuration.

### Terms of Service
- **Goal:** legal risk disclosure and usage acknowledgment.
- **Actions:** review terms and acknowledge checkbox.
- **Outputs:** session-level acknowledgment state.

### Settings & Analytics
- **Goal:** app-level diagnostics and analytics controls.
- **Actions:** inspect analytics dashboards and settings surfaces.
- **Outputs:** usage and system analytics views.

## Design / UX Principles
- Consistent dark institutional theme through `octavian_theme.py`.
- High-signal cards, compact metrics, and explicit sectioning.
- Risk-first language in outputs where financial decisions are implied.

## Backend/Frontend Boundary
- Frontend = Streamlit pages and controls.
- Backend = engine modules performing fetch/compute/scoring.
- Communication = direct Python function calls per route.

## Operational Notes
- Startup speed depends on lazy imports and cached resources.
- Optional dependencies are guarded; modules should degrade gracefully.
- API keys must be provided via environment or secrets, never hardcoded.
