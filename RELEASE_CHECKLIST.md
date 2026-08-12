# Octavian Release Checklist

Use this before publishing/deploying any build.

## 1) Security and Secrets
- [ ] No API keys in source files.
- [ ] `.env` is not committed.
- [ ] `.streamlit/secrets.toml` is not committed.
- [ ] `config.py` reads secrets from env/secrets only.
- [ ] `rg "API_KEY|SECRET|token"` review completed for accidental leaks.

## 2) Runtime Stability
- [ ] `python -m py_compile` passes for core modules:
  - `main.py`
  - `ai_chatbot.py`
  - `document_analyzer.py`
  - `position_optimizer_engine.py`
  - `portfolio_analyzer_engine.py`
  - `alternative_data_engine.py`
  - `risk_engine.py`
- [ ] Streamlit app launches: `streamlit run main.py`
- [ ] Critical pages open without exceptions.

## 3) Performance
- [ ] Startup path uses lazy imports for heavy modules.
- [ ] Route render timing displayed and reviewed for major pages.
- [ ] Expensive resources cached where appropriate.

## 4) Legal and UX
- [ ] Terms of Service page is visible in navigation.
- [ ] Sensitive pages require ToS acknowledgment.
- [ ] Risk language is present on decision-support pages.

## 5) Data Vendor Integrations
- [ ] `MASSIVE_API_KEY` configured in target environment.
- [ ] Massive fallback path verified (falls back to yfinance gracefully).
- [ ] Session-only key control in Settings works.

## 6) Documentation
- [ ] `OCTAVIAN_ARCHITECTURE.md` current.
- [ ] `OCTAVIAN_PLATFORM_FUNCTIONALITY_GUIDE.md` current.
- [ ] README setup instructions match actual config flow (`.env` / secrets).

## 7) Final QA
- [ ] Smoke test main user paths end-to-end.
- [ ] Confirm no obvious UI regressions on themed pages.
- [ ] Confirm logs have no repeated tracebacks.
