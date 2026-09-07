# Octavian Terminal

Octavian Terminal is a Streamlit-based, multi-asset financial intelligence and research platform. It combines market data, technical and fundamental analysis, news and sentiment processing, quantitative research, financial modeling, institutional analytics, paper trading, simulations, and an AI-assisted research workflow in one local web application.

> **Important:** Octavian is a research and education tool, not an investment adviser, broker, or trading system for live capital. Analysis can be modeled, delayed, incomplete, or unavailable when a data provider cannot be reached. Review the in-app Terms of Service and Risk Disclosure before using the platform.

## What you get

- A professional Streamlit web interface with account registration, onboarding, trader profiles, watchlists, and notifications.
- Multi-asset coverage for equities, ETFs, crypto, FX, futures, and commodities where supported by the selected data source.
- Offline-safe behavior: most analysis engines can run without paid API keys and clearly label unavailable, modeled, inferred, and observed data.
- Optional NVIDIA hosted inference for AI-generated analysis, with a local LM Studio fallback and deterministic analysis paths when no LLM is available.
- Local SQLite/JSON persistence for users, profiles, conversations, simulations, paper-trading state, and application settings.

## Requirements

### Recommended local environment

- macOS, Linux, or Windows with a Bash-compatible shell
- Python **3.12** (the reference environment for the project)
- Git
- At least 4 GB of free memory; 8 GB or more is recommended for heavier quantitative and ML features
- Internet access for live Yahoo Finance, news, SEC, FINRA, and optional vendor data
- A modern browser such as Chrome, Firefox, Safari, or Edge

Python 3.13+ may work, but the pinned scientific/ML dependencies are tested against Python 3.12. The included Dockerfile uses Python 3.14 and is an optional container path; local Python 3.12 is the safer default.

## Quick start

```bash
git clone https://github.com/apbharucha/octavianv1.git
cd octavianv1

python3.12 -m venv .venv
source .venv/bin/activate                 # Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

cp .env.example .env                    # optional; add keys before the next step
./start_streamlit.sh
```

Open [http://localhost:8501](http://localhost:8501) in your browser. If the startup script is not available in your shell, run the app directly:

```bash
streamlit run main.py --server.headless true
```

To stop the app, press `Ctrl+C` in the terminal running Streamlit. Use `./start_streamlit.sh --clean` only when you need to clear local Python/Streamlit caches; normal starts preserve caches and are faster.

### Windows notes

Use Git Bash, WSL, or another Bash-compatible terminal for `start_streamlit.sh`. In PowerShell, the equivalent direct launch is:

```powershell
.venv\\Scripts\\Activate.ps1
python -m streamlit run main.py --server.headless true
```

If PowerShell blocks activation, run `python -m pip install -r requirements.txt` after activating the environment through your preferred Python tooling, or use Git Bash/WSL.

## Configuration and API keys

All provider keys are optional for a basic local/demo experience. The application reads environment variables first and can also read the same names from Streamlit secrets. Never commit `.env`, `secrets.toml`, API keys, passwords, or local database files.

### `.env` setup

Copy the supplied template and edit it locally:

```bash
cp .env.example .env
```

The available settings are:

| Variable | Required | Purpose |
|---|---:|---|
| `NVIDIA_API_KEY` | No | Enables NVIDIA hosted inference as the primary AI provider. |
| `NVIDIA_LLM_API_URL` | No | NVIDIA-compatible chat-completions URL; the template contains the default. |
| `NVIDIA_LLM_MODEL` | No | Hosted model name; the template contains the default. |
| `FINRA_API_KEY` | No | Authoritative weekly FINRA OTC Transparency data in Dark Pool Intelligence. Register through [FINRA Developer Center](https://developer.finra.org/). |
| `POLYGON_API_KEY` | No | Optional Polygon market-data integration. |
| `ALPHA_VANTAGE_KEY` | No | Optional Alpha Vantage market/news integration. |
| `OANDA_API_KEY` | No | Optional OANDA FX data. Keep the default practice URL unless you intentionally use another endpoint. |
| `OANDA_BASE_URL` | No | OANDA API base URL; defaults to the practice endpoint. |
| `MASSIVE_API_KEY` | No | Optional Massive/alternative-data integration. |
| `EODHD_API_KEY` | No | Optional EODHD historical data and financial statements. |
| `SECRET_KEY` | Production | Flask session signing key for the optional API backend. Use a long random value. |
| `API_BACKEND_KEY` | Production | Optional bearer token for protected API endpoints. |
| `CORS_ORIGINS` | Production | Comma-separated browser origins allowed to call the optional API backend. |

Without `NVIDIA_API_KEY`, the chatbot can use a local LM Studio OpenAI-compatible server at `http://localhost:1234` when one is running, then falls back to deterministic, rule-based analysis. Without vendor keys, the app attempts public/free sources and labels unavailable data rather than silently inventing it.

### Streamlit secrets alternative

Instead of `.env`, create `.streamlit/secrets.toml` and add only the keys you need:

```toml
NVIDIA_API_KEY = "your-key"
FINRA_API_KEY = "your-key"
```

`.streamlit/secrets.toml` is ignored by Git. Restart Streamlit after changing environment variables or secrets.

## First use

1. Start the app and open the local URL.
2. Select **Register** on the authentication page.
3. Create a username, email, and password, then accept the Terms of Service and Risk Disclosure.
4. Complete onboarding with your interests, risk tolerance, preferred timeframe, and in-app alert preference.
5. Use the left sidebar **Navigation** menu to open a research area.
6. Select a trader profile in the sidebar or refine it later in **Trader Profile**.

User accounts and profiles are stored in the local `octavian_users.db` SQLite database. Treat that file as private local state. This repository does not provide a hosted login service or a production identity-management deployment.

## Using the site

The primary entry point is `main.py`. The sidebar contains the following pages:

### Market discovery and analysis

- **Dashboard** — personalized view, market overview, live index strip, cached charts, and high-confidence Breaking Trades setups.
- **Watchlist** — monitor selected symbols with quotes, changes, charts, and watchlist-oriented analysis.
- **Market Scanner** — scan the supported universe for movers, technical conditions, and candidate opportunities.
- **Symbol Analysis** — inspect a selected symbol with price history, indicators, fundamentals, and model-derived trade context.
- **Chart Analysis** — upload a chart screenshot for pattern recognition and technical observations.
- **Market Heartbeat** — review macro market state, index conditions, volatility, and cross-asset observations.
- **Dark Pool Intelligence** — inspect off-exchange activity, FINRA OTC anchors when available, modeled signals, provenance, and methodology. Configure a FINRA key from the page’s settings area.
- **Institutional 13F & SEC Filings** — review institutional holdings and SEC filing-based smart-money analysis when source data is available.

### Intelligence Center

The **Intelligence Center** contains three tabs:

- **News & Sentiment** — live news feeds, source filtering, sentiment summaries, market whispers, sector sentiment, and news-impact analysis.
- **Octavian AI Assistant** — ask natural-language questions such as:
  - `Analyze AAPL with technical indicators`
  - `Compare MSFT and GOOGL`
  - `Give me a DCF view of NVDA`
  - `What is the risk profile of SPY?`
  - `Show me EUR/USD technical analysis`
  - `Explain the current macro regime and its impact on gold`
- **Counter-Trend Signals** — inspect macro narrative divergence, consensus versus fundamentals, crowding, catalysts, transmission paths, value-trap warnings, invalidation conditions, and optional developed trade setups. A high divergence score is not an automatic trade recommendation.

For stronger AI responses, include the symbol, asset type, timeframe, and the question you want answered. Treat any live-data timestamp and provenance label as part of the answer, not as decoration.

### Research, modeling, and strategy

- **Daily Briefing** — generate a broad market intelligence report with macro, cross-asset, narrative, risk, and opportunity sections. Longer jobs run in the background; you can navigate while they finish.
- **Quant Portal** — backtesting, factor analysis, alternative data, correlation, and quantitative strategy tools.
- **Quant Modeling Lab** — explore quantitative models, factor crowding, options, market regimes, and narrative dislocation.
- **Strategy Research Lab** — research pairs, options, factor, and other strategy families with risk-aware outputs.
- **Algorithm Builder** — describe a strategy in plain language, choose a universe, run backtests, inspect trade reasoning, and bridge a strategy into paper trading where supported.
- **Target Probability** — estimate modeled probability bands for a target and horizon using documented assumptions and available history.
- **Comparative Analysis** — compare symbols, sectors, performance, fundamentals, risk, and optional AI theses.

### Financial modeling and documents

- **Financial Model Generator** — build DCF, LBO, M&A accretion/dilution, comps, IPO, and precedent-transaction analyses. Review assumptions, data provenance, sensitivities, and model limitations.
- **Spreadsheet Generator** — export formatted workbooks, including formula-driven DCF/LBO templates, financial statements, indicators, charts, and sensitivity tables. Open exported files in Excel or another compatible spreadsheet application so formulas can recalculate.
- **Presentation Generator** — create institutional-style pitchbooks with optional branding, valuation ranges, football fields, tornado charts, disclaimers, and downloadable PowerPoint output.
- **Document Analyzer** — upload supported financial documents or filings for extraction and analysis.
- **Chart Analysis** — upload chart images for visual technical analysis; do not treat image interpretation as a substitute for source data.

### Portfolio, trading, and simulation

- **Portfolio Analyzer** — load or enter holdings and review allocation, performance, risk, and portfolio-level analytics.
- **Position Optimizer** — explore position sizing and portfolio construction under stated constraints.
- **Paper Trading** — create paper accounts, record simulated equity and options trades, review positions/history/performance, and configure automated paper-trading controls. This does not place live brokerage orders.
- **Simulation Hub** — run market simulations, scenario analysis, Monte Carlo-style experiments, parameter exploration, and simulation grading.

### Personalization and administration

- **Trader Profile** — update interests, risk tolerance, timeframe, and personalized dashboard preferences.
- **Notification Settings** — control in-app notification behavior and alert frequency.
- **Settings & Analytics** — inspect application, conversation, model, and data-quality metrics.
- **Terms of Service** — review the platform terms and risk disclosure at any time.

## Data behavior and provenance

Octavian uses a mixture of public, optional authenticated, cached, modeled, and user-entered data. Common sources include Yahoo Finance/yfinance, OANDA, Alpha Vantage, Polygon, EODHD, FINRA OTC Transparency, SEC/EDGAR, RSS/news sources, and local caches.

- Public market data can be delayed, rate-limited, incomplete, or temporarily unavailable.
- FINRA OTC Transparency data is weekly and has reporting lag; it should not be read as daily dark-pool volume.
- A `MODELED`, `INFERENCE`, `ASSUMED`, `MARKET-IMPLIED`, or `DATA UNAVAILABLE` label means the value is not a directly observed live fact.
- The institutional deep-dive and financial-model workflows separate reported-data valuation, assumption-based DCF, market-implied expectations, and scenario analysis.
- Never paste an API key into the chatbot. Put credentials in `.env` or Streamlit secrets only.
- Cached data and local state are not a substitute for validating a decision against primary sources.

## Optional API backend

The repository also contains `api_backend.py`, a Flask REST backend for market data, analysis, predictions, chat, profiles, history, and analytics. The main user site does **not** require the API backend; use the Streamlit interface unless you specifically need REST access.

To run it, first install the Flask backend dependencies required by your environment (they are not all included in the main Streamlit requirements file), configure `SECRET_KEY`, `API_BACKEND_KEY`, and `CORS_ORIGINS`, then run:

```bash
source .venv/bin/activate
python api_backend.py
```

The default port is `5000`; set `PORT` to change it. Check `http://localhost:5000/health`. Do not expose this development server directly to the public internet without production-grade authentication, HTTPS, a real WSGI server, and a reviewed CORS policy.

## Docker (optional)

The repository includes a `Dockerfile` for a containerized Streamlit launch:

```bash
docker build -t octavian-terminal .
docker run --rm -p 8501:8501 --env-file .env octavian-terminal
```

Open [http://localhost:8501](http://localhost:8501). For persistent users, paper-trading state, exports, and databases, mount a host directory at `/app` or use an explicit volume strategy after reviewing which local files you want to persist. Confirm the image builds successfully in your environment because some scientific/ML packages may require platform-specific wheels.

## Project layout

```text
main.py                         Streamlit entry point and page router
config.py                       Environment and Streamlit-secret configuration
data_sources.py                 Market-data and quote adapters
financial_llm_engine.py         Query routing, financial analysis, and deep dives
ai_chatbot.py                   Chatbot UI and conversation workflow
news_analysis_engine.py         News aggregation and sentiment analysis
analytical_context.py           Canonical security data and provenance context
analytical_integrity.py         Entity, semantic, economic, and confidence checks
rating_gate.py                  Investment-rating and template integrity gates
paper_trading_system.py         Local paper-trading engine
market_simulation_engine.py     Simulation and scenario engine
spreadsheet_generator.py        Workbook generation UI and exports
presentation_generator.py       PowerPoint pitchbook generation
api_backend.py                  Optional Flask REST backend
backend/                        Additional backend package components
tests/                          Automated unit, integration, and UI tests
MASTER_DOC.md                   Detailed architecture and module reference
.env.example                    Safe configuration template
```

See [MASTER_DOC.md](MASTER_DOC.md) for the authoritative technical reference, architecture notes, module APIs, data flow, known gotchas, and test inventory.

## Development and verification

Activate the environment, then run the full test suite:

```bash
source .venv/bin/activate
OCTAVIAN_OFFLINE=1 python -m pytest tests/ -q
```

`OCTAVIAN_OFFLINE=1` keeps tests deterministic and prevents optional network fetches. Run without it only when intentionally testing live-provider behavior. For a quick import check:

```bash
python -c "import main; print('main imported')"
```

When changing code:

1. Keep secrets and runtime artifacts out of Git.
2. Prefer the existing data-source adapters instead of calling providers directly.
3. Preserve provenance labels and explicit unavailable-data states.
4. Keep Streamlit-heavy imports lazy where possible.
5. Add or update tests for behavior changes.
6. Update `MASTER_DOC.md` in the same change, including its changelog.

## Troubleshooting

### The app does not start

- Confirm the virtual environment is active: `which python` and `python --version`.
- Reinstall dependencies with `python -m pip install -r requirements.txt`.
- Run `streamlit run main.py` directly to see the full error.
- If stale bytecode or Streamlit cache is suspected, use `./start_streamlit.sh --clean`.

### A chart or quote is empty

- Check internet access and the symbol format supported by the source.
- Yahoo and other public providers can rate-limit or temporarily return no data.
- Wait for the cache window to expire and retry; do not assume an empty result means the asset has no market data.
- Add an optional provider key if the feature supports one.

### AI responses are unavailable

- The deterministic analysis path should still work for many queries.
- Add `NVIDIA_API_KEY` and restart Streamlit for hosted inference.
- Alternatively run a compatible LM Studio server at `localhost:1234`.
- Check that `NVIDIA_LLM_API_URL` and `NVIDIA_LLM_MODEL` match the provider configuration.

### Login or local data problems

- User data is stored in `octavian_users.db` in the project directory.
- Ensure the project directory is writable.
- Stop Streamlit before moving or backing up the database.
- For a disposable local reset, stop the app and back up the database before replacing it; never do this on a production copy without a backup.

### Exports fail

- Confirm the output directory is writable and the relevant optional dependency is installed from `requirements.txt`.
- For PowerPoint or Excel files, open the generated file in a compatible desktop application and allow formulas to recalculate.
- Large analyses may finish in the background; check the sidebar Background Tasks panel.

## Security and operational notes

- This is primarily a local development/research application. Review authentication, secrets, CORS, rate limits, persistence, and reverse-proxy settings before any shared deployment.
- Do not expose SQLite databases, `.env`, Streamlit secrets, API responses, or exported files containing personal/financial information.
- Paper trading and automated trading features are simulated/local unless a separately reviewed brokerage integration is added; they do not authorize live orders by themselves.
- Provider terms, rate limits, and data licenses apply to every external source.

## Contributing

1. Create a branch for your change.
2. Make the smallest focused change that solves the problem.
3. Add regression coverage where appropriate.
4. Run the relevant tests and the full suite when practical.
5. Update `MASTER_DOC.md` for code or test changes.
6. Submit a pull request with a concise explanation and verification results.

## License and support

No `LICENSE` file is currently included in this repository. Confirm the intended licensing terms with the repository owner before redistributing or deploying the project.

For problems, search the existing documentation and GitHub issues first, then open an issue with the operating system, Python version, command used, traceback, and whether the failure reproduces with `OCTAVIAN_OFFLINE=1`. Remove API keys and other private data before posting logs.
