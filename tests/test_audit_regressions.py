"""
Regression tests for the full-codebase audit.

Locks in the key fixes:
1. No hardcoded API secrets in source (all secrets come from env vars).
2. SEC 13F data defaults to real EDGAR data with explicit provenance labels.
3. get_latest_price() never presents 0.0 as a real market price.
4. Asset universes are powered by the dynamic ticker universe (no hardcoded lists).
"""
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _src(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def _iter_source_files():
    for p in ROOT.rglob("*.py"):
        if any(x in str(p) for x in (".venv", "frontend", "node_modules", "nltk_data",
                                     "data_cache", ".temp_trash", "__pycache__")):
            continue
        yield p


# ─────────────────────────────────────────────────────────────────────────────
# 1. SECURITY: no hardcoded secrets
# ─────────────────────────────────────────────────────────────────────────────
def test_eodhd_key_is_env_backed():
    """The EODHD key must be read from the environment, not a source literal."""
    cfg = _src("config.py")
    assert "EODHD_API_KEY = " in cfg
    # No 32+ char key literal should exist anywhere in the project source.
    key_like = re.compile(r"['\"]([A-Za-z0-9_]{32,})['\"]")
    for p in _iter_source_files():
        text = p.read_text(encoding="utf-8")
        for m in key_like.finditer(text):
            tok = m.group(1)
            low = tok.lower()
            if any(x in low for x in ("example", "placeholder", "your_", "dummy",
                                      "abcdef", "test", "changeme")):
                continue
            if re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-", low):
                continue  # UUIDs are identifiers, not secrets
            if not re.search(r"[0-9]", tok):
                continue  # pure-alphabetic tokens are field names, not keys
            pytest.fail(f"Possible hardcoded secret in {p.name}: {tok[:12]}...")


def test_flask_secret_key_not_static():
    """The Flask secret key must not be a hardcoded default string."""
    cfg = _src("config.py")
    assert "token_hex" in cfg  # ephemeral key generated at startup when unset


def test_env_example_documents_new_vars():
    env = _src(".env.example")
    for var in ("EODHD_API_KEY", "SECRET_KEY", "API_BACKEND_KEY", "CORS_ORIGINS"):
        assert var in env, f".env.example should document {var}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA PROVENANCE: SEC 13F must be real by default
# ─────────────────────────────────────────────────────────────────────────────
def test_13f_defaults_to_real_edgar():
    src = _src("sec_13f_engine.py")
    assert "def fetch_latest_filings(self, limit: int = 10, allow_simulated: bool = False)" in src
    assert "PROVENANCE_REAL" in src and "PROVENANCE_SIMULATED" in src
    assert "SEC EDGAR" in src


def test_13f_simulation_is_explicitly_labeled():
    src = _src("sec_13f_engine.py")
    # Simulated filings must be tagged with the simulated provenance constant.
    assert "PROVENANCE_SIMULATED" in src
    assert "SIMULATED ESTIMATE" in src


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATA QUALITY: get_latest_price() never returns 0.0
# ─────────────────────────────────────────────────────────────────────────────
class _FakeTicker:
    info = {}


def test_get_latest_price_never_returns_zero(monkeypatch):
    import yfinance as yf
    from data_sources import get_latest_price

    monkeypatch.setattr(yf, "Ticker", lambda s: _FakeTicker())

    # Provider returns a real price → returned as-is.
    monkeypatch.setattr("data_sources.get_realtime_price", lambda s: (152.3, None))
    assert get_latest_price("TEST") == 152.3

    # Provider returns 0.0 → must become None, never a fake 0 price.
    monkeypatch.setattr("data_sources.get_realtime_price", lambda s: (0.0, None))
    assert get_latest_price("TEST") is None

    # Provider returns None → stays None.
    monkeypatch.setattr("data_sources.get_realtime_price", lambda s: (None, None))
    assert get_latest_price("TEST") is None

    # Negative values are also invalid prices.
    monkeypatch.setattr("data_sources.get_realtime_price", lambda s: (-5.0, None))
    assert get_latest_price("TEST") is None


# ─────────────────────────────────────────────────────────────────────────────
# 4. DYNAMIC UNIVERSES (no hardcoded ticker lists in user-facing paths)
# ─────────────────────────────────────────────────────────────────────────────
def test_universe_is_multi_asset_and_live():
    from ticker_universe import get_ticker_universe

    u = get_ticker_universe()
    assert len(u.get_all_stocks()) > 100
    assert len(u.get_etfs()) > 10
    assert len(u.get_crypto()) > 5
    assert len(u.get_forex()) > 5
    assert len(u.get_futures()) > 5


def test_market_scanner_uses_dynamic_groups():
    src = _src("market_scanner.py")
    assert "def _get_watchlist_groups" in src
    assert "def _get_crypto_list" in src
    # The old hardcoded watchlist dict must be gone.
    assert "_WATCHLIST = {" not in src
    assert "_CRYPTO = [" not in src


def test_trader_profile_uses_dynamic_presets():
    src = _src("trader_profile.py")
    assert "def get_asset_classes" in src
    assert "def get_watchlist_presets" in src


def test_ai_chatbot_uses_dynamic_examples():
    src = _src("ai_chatbot.py")
    assert "def _get_asset_examples" in src
    # Only ONE guidance definition may exist (the duplicate was removed).
    assert src.count("def _generate_octavian_guidance") == 1
    # The hardcoded "Popular Stocks" list must be gone.
    assert "Popular Stocks" not in src


def test_market_movers_has_no_hardcoded_fallback():
    src = _src("market_movers.py")
    assert "Last-resort hardcoded list" not in src


def test_news_dashboard_has_no_fake_dates():
    src = _src("news_dashboard.py")
    assert "2024-02-04" not in src
    assert "2024-02-03" not in src


def test_main_breaking_trades_is_dynamic():
    src = _src("main.py")
    # Breaking trades scans the FULL dynamic universe, not a fixed sample,
    # and reuses the singleton generator (no per-click thread-pool leak).
    assert "tu.get_full_universe()" in src
    assert "get_breaking_trades_generator" in src
    assert "BreakingTradesGenerator(" not in src
    # The old hardcoded equity/commodity/fx/crypto lists are gone.
    assert '"EURUSD=X", "USDJPY=X", "GBPUSD=X", "AUDUSD=X"' not in src


def test_universe_has_no_duplicate_definitions():
    src = _src("ticker_universe.py")
    for name in ("_ETFS", "_FOREX", "_FUTURES"):
        assert len(re.findall(rf"^{name} = \[", src, re.M)) == 1, f"duplicate {name}"


# ─────────────────────────────────────────────────────────────────────────────
# 5. AMERICAN OPTIONS: Barone-Adesi-Whaley is a real implementation
# ─────────────────────────────────────────────────────────────────────────────
def test_whaley_american_is_not_a_placeholder():
    """The futures American option pricer must be a full BAW implementation
    (no 'return euro' placeholder) with correct boundary behavior."""
    src = _src("futures_engine.py")
    assert "whaley_american_futures" in src
    assert "# Placeholder for brevity" not in src
    assert "_newton" in src  # critical-price solver present


def test_whaley_american_boundaries_and_accuracy():
    """American futures options must be >= European, >= intrinsic, and match a
    high-step binomial tree within BAW's expected tolerance (~2%)."""
    import numpy as np
    from futures_engine import get_futures_engine

    eng = get_futures_engine()
    eng.rf_rate = 0.05
    r = 0.05

    def binomial_american(F, K, T, sigma, n=600, typ="call"):
        dt = T / n
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        p = (1.0 - d) / (u - d)
        disc = np.exp(-r * dt)
        S = F * u ** np.arange(-n, n + 1, 2)
        V = np.maximum(S - K, 0.0) if typ == "call" else np.maximum(K - S, 0.0)
        for i in range(n - 1, -1, -1):
            V = disc * (p * V[1:] + (1 - p) * V[:-1])
            S = F * u ** np.arange(-i, i + 1, 2)
            V = np.maximum(V, S - K) if typ == "call" else np.maximum(V, K - S)
        return V[0]

    cases = [
        (100, 100, 0.25, 0.2), (100, 100, 1.0, 0.25), (110, 100, 0.5, 0.2),
        (90, 100, 0.5, 0.2), (95, 100, 0.1, 0.35), (105, 100, 2.0, 0.15),
        (80, 100, 0.75, 0.3), (120, 100, 0.75, 0.3), (100, 95, 0.4, 0.22),
    ]
    max_rel_err = 0.0
    for F, K, T, sig in cases:
        for typ in ("call", "put"):
            bin_val = binomial_american(F, K, T, sig, typ=typ)
            wal = eng.whaley_american_futures(F, K, T, sig, typ)
            euro = eng.black76(F, K, T, sig, typ)["price"]
            intrinsic = max(0.0, F - K) if typ == "call" else max(0.0, K - F)
            # Boundary constraints
            assert wal >= euro - 1e-9, (F, K, T, sig, typ, wal, euro)
            assert wal >= intrinsic - 1e-9, (F, K, T, sig, typ, wal, intrinsic)
            # BAW accuracy vs binomial
            rel = abs(wal - bin_val) / max(bin_val, 1e-9)
            max_rel_err = max(max_rel_err, rel)
    assert max_rel_err < 0.03, f"BAW deviates {max_rel_err:.2%} from binomial tree"


def test_whaley_early_exercise_premium_present():
    """Deep ITM puts on futures must carry a real early-exercise premium over
    the European value (this is the bug that was previously placeholder)."""
    from futures_engine import get_futures_engine

    eng = get_futures_engine()
    eng.rf_rate = 0.05
    # Deep ITM put, high vol, longer maturity -> measurable early-exercise premium
    euro = eng.black76(80, 100, 1.0, 0.35, "put")["price"]
    am = eng.whaley_american_futures(80, 100, 1.0, 0.35, "put")
    assert am > euro + 0.05
    # ATM call at T=2y with vol 0.15 also carries a premium
    euro_c = eng.black76(105, 100, 2.0, 0.15, "call")["price"]
    am_c = eng.whaley_american_futures(105, 100, 2.0, 0.15, "call")
    assert am_c > euro_c + 0.01


# ─────────────────────────────────────────────────────────────────────────────
# 6. NO FABRICATED MARKET WHISPERS
# ─────────────────────────────────────────────────────────────────────────────
def test_market_whispers_are_not_fabricated():
    """get_market_whispers must derive signals from real fetched articles, not
    random templates with fake sources / mention counts / verification statuses."""
    src = _src("news_analysis_engine.py")
    # No invented "Institutional Desk" / "Alpha Scanner" sources
    assert "Institutional Desk" not in src
    assert "Alpha Scanner" not in src
    # No random template fabrication
    assert "tmpl['text'].format(sym=" not in src
    # Random verification-status selection is gone (honest corroboration instead)
    assert "verification_status=np.random.choice" not in src
    # Real data hooks present
    assert "self.fetch_and_process_news() or []" in src
    assert "real_coverage" in src


def test_market_whispers_empty_when_no_articles():
    """With no articles available, whispers must be empty (honest empty state)."""
    from news_analysis_engine import NewsAnalysisEngine

    eng = NewsAnalysisEngine.__new__(NewsAnalysisEngine)  # skip __init__

    def _no_news(self):
        return []

    eng.fetch_and_process_news = _no_news.__get__(eng)
    eng._infer_whisper_type = lambda a: "other"
    assert eng.get_market_whispers("AAPL") == []
    assert eng.get_market_whispers() == []


def test_market_whispers_use_real_article_data():
    """Whispers built from real articles must carry the article's real title,
    source and timestamp, and verification must reflect corroboration."""
    from datetime import datetime, timezone
    from news_analysis_engine import (
        NewsAnalysisEngine, NewsArticle, SentimentScore,
    )

    eng = NewsAnalysisEngine.__new__(NewsAnalysisEngine)
    now = datetime.now(timezone.utc)

    def _make(title, src, syms):
        return NewsArticle(
            title=title, summary="", url=f"http://x/{src}", source=src,
            published_at=now, symbols_mentioned=syms, sentiment_score=0.6,
            sentiment_category=SentimentScore.BULLISH, market_impact_score=0.5,
            relevance_score=0.8, article_id=title, tags=[],
        )

    # 3 distinct outlets on the same symbol -> 'verified' corroboration
    articles = [
        _make("AAPL beats earnings", "Reuters", ["AAPL"]),
        _make("AAPL guidance strong", "Bloomberg", ["AAPL"]),
        _make("Apple raises outlook", "CNBC", ["AAPL"]),
        _make("NVDA rally continues", "Reuters", ["NVDA"]),
    ]
    eng.fetch_and_process_news = lambda: articles
    eng._infer_whisper_type = lambda a: "earnings"

    whispers = eng.get_market_whispers("AAPL")
    assert whispers, "expected real whispers from 3-article AAPL coverage"
    aapl = [w for w in whispers if "AAPL" in w.symbols_mentioned]
    assert aapl
    top = aapl[0]
    assert "AAPL beats earnings" in top.content or "AAPL guidance strong" in top.content
    assert top.source in {"Reuters", "Bloomberg", "CNBC"}
    assert top.verification_status == "verified"
    assert top.social_mentions >= 3  # real coverage count


def test_quant_portal_crowding_is_not_random():
    """Factor crowding lives exclusively in the Quant Modeling Lab and must
    call the real engine, never np.random.uniform fake crowding scores."""
    src = _src("quant_modeling_lab.py")
    assert "get_crowding_engine()" in src
    assert "engine.build_dashboard" in src
    # The old fake-data block is gone
    assert "crowding = np.random.uniform(0, 100, len(factors))" not in src
    assert "factors = ['Momentum', 'Value', 'Size', 'Quality', 'Low Vol']" not in src


# ─────────────────────────────────────────────────────────────────────────────
# 7. INSTITUTIONAL PITCHBOOK AESTHETICS + NO FABRICATED DEAL FACTS
# ─────────────────────────────────────────────────────────────────────────────
def test_pitchbook_uses_institutional_typography():
    """Generated decks must use a consistent institutional font (Arial), carry
    page-number footers, and avoid fabricated deal facts (fake banks / fake
    board recommendations)."""
    src = _src("presentation_generator.py")
    assert 'FONT    = "Arial"' in src
    assert "_apply_run_font" in src          # centralized typography
    assert "_add_footer" in src              # page numbers on every slide
    assert "_set_table_col_widths" in src    # deliberate table geometry
    # Fabricated deal facts removed
    assert "J.P. Morgan, Goldman Sachs" not in src
    assert "unanimously recommended the transaction" not in src
    assert "Project Spartan" not in src
    assert "Project Horizon" not in src
    # Dynamic code names + real model fields
    assert "_codename(" in src
    # Model fields are read through the central _a() helper, which resolves
    # against the result's `assumptions` dataclass first (no hardcoded values).
    assert "_a(r, 'leverage_multiple'" in src
    assert 'a = getattr(r, "assumptions", None)' in src


def test_pitchbook_builds_all_deck_types():
    """All three pitchbook builders produce valid pptx with the right slide
    counts and a consistent font on every run."""
    import io
    from pptx import Presentation
    from presentation_generator import get_presentation_generator
    from mna_model_engine import MnAAssumptions, get_mna_engine
    from lbo_model_engine import LBOAssumptions, get_lbo_engine
    from financial_model_generator import DCFAssumptions, get_dcf_engine

    pg = get_presentation_generator()

    m = get_mna_engine().run_mna(MnAAssumptions(
        acquirer_ticker="MSFT", target_ticker="ATVI",
        acquirer_price=400, acquirer_eps=10, acquirer_shares=7000,
        target_price=80, target_eps=3, target_shares=800,
        offer_premium=0.30, percent_stock=0.5, percent_cash=0.5,
        cost_of_debt=0.05, tax_rate=0.21, pre_tax_synergies=500,
    ))
    decks = {
        "mna": pg.generate_mna_pitchbook("MSFT", "ATVI", m),
        "lbo": pg.generate_lbo_pitchbook("TWTR", get_lbo_engine().run_lbo(
            LBOAssumptions(
                ticker="TWTR", target_name="TWTR", entry_year=2024, exit_year=2029,
                ltm_ebitda=1000, entry_multiple=12, exit_multiple=12,
                leverage_multiple=6, interest_rate=0.08,
            ))),
        "dcf": pg.generate_dcf_pitchbook("AAPL", get_dcf_engine().run_dcf(
            DCFAssumptions(
                ticker="AAPL", base_revenue=5000, revenue_growth_rates=[0.05]*5,
                ebit_margin=0.25, tax_rate=0.21, da_pct_revenue=0.03,
                capex_pct_revenue=0.04, nwc_change_pct_revenue=0.02,
                equity_value_market=8000, debt_value=2000, cost_of_debt=0.045,
                risk_free_rate=0.0425, equity_risk_premium=0.055, beta=1.2,
                terminal_growth_rate=0.025, cash=500, shares_outstanding=200,
                current_price=40, projection_years=5,
            ))),
    }
    for name, b in decks.items():
        assert b and b.startswith(b"PK"), f"{name} deck not a valid zip/pptx"
        prs = Presentation(io.BytesIO(b))
        n = len(prs.slides._sldIdLst)
        assert n >= 10, f"{name} deck too short: {n} slides"
        fonts = set()
        for s in prs.slides:
            for sh in s.shapes:
                if sh.has_text_frame:
                    for p in sh.text_frame.paragraphs:
                        for r in p.runs:
                            if r.font.name:
                                fonts.add(r.font.name)
        assert fonts == {"Arial"}, f"{name} deck uses non-Arial fonts: {fonts}"


# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC UNIVERSE (no preset ticker lists)
# ─────────────────────────────────────────────────────────────────────────────

def test_breaking_trades_uses_singleton_no_thread_leak():
    """main.py must reuse the singleton generator (no per-click 20-thread leak)."""
    src = _src("main.py")
    assert "get_breaking_trades_generator()" in src
    assert "BreakingTradesGenerator(" not in src


def test_generator_has_lazy_executor_and_shutdown():
    """The generator must not spawn threads at construction, and must release
    them via shutdown() (thread exhaustion was the 'Python quit unexpectedly'
    crash)."""
    src = _src("breaking_trades_generator.py")
    assert "self._executor = None" in src
    assert "def shutdown(self)" in src
    assert "threading.Lock" in src


def test_universe_exposes_full_dynamic_universe():
    """get_full_universe() returns every known asset (no sampling) — the
    Breaking Trades scan analyzes ALL tickers, never a preset list."""
    src = _src("ticker_universe.py")
    assert "def get_full_universe(self)" in src


def test_sector_scanner_is_dynamic():
    """sector_scanner must not ship a hardcoded SECTOR_MAP anymore."""
    src = _src("sector_scanner.py")
    assert "SECTOR_MAP = {" not in src
    assert "def get_dynamic_sector_map" in src


def test_no_stale_sector_map_imports():
    """No consumer may import the removed hardcoded SECTOR_MAP."""
    for f in ("ai_chatbot.py", "news_dashboard.py", "market_scanner.py"):
        src = _src(f)
        assert "from sector_scanner import SECTOR_MAP" not in src, f


def test_discovery_batch_fetch_is_chunked():
    """Full-universe scans must chunk yfinance downloads (single giant calls
    fail/hang on thousands of symbols)."""
    src = _src("octavian_discovery_engine.py")
    assert "_CHUNK = 150" in src
