"""
Chatbot prompt-coverage harness.

Generates a large corpus of user prompts (hundreds-to-thousands) across
topic categories and asserts the extraction layer behaves correctly:

  - No English words are ever parsed as tickers (the AFFECT=X / OIL / PROOF
    / CLAIM class of bug).
  - Commodity names resolve to their futures anchors (oil -> CL=F).
  - Index/vol aliases resolve (VIX -> ^VIX, DXY -> DX-Y.NYB).
  - Explicit tickers are preserved.
  - Cross-asset transmission queries are detected and route to the
    quantitative engine (which never fabricates when data is missing).
"""

import re
from itertools import product

import pandas as pd
import pytest

from financial_llm_engine import expand_query_intents

# ─────────────────────────────────────────────────────────────────────────────
# Prompt generators — build a broad, realistic prompt corpus
# ─────────────────────────────────────────────────────────────────────────────

_ASSETS = [
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "AMD", "JPM",
    "GS", "BAC", "XOM", "CVX", "CAT", "BA", "DIS", "NFLX", "KO", "PEP", "WMT",
]

_SECTORS = ["technology", "biotech", "healthcare", "energy", "financials",
            "consumer", "industrials", "semiconductors", "defense", "utilities"]

_COMMODITIES = ["oil", "crude oil", "gold", "silver", "copper", "natural gas",
                "wheat", "corn", "soybeans", "brent", "gasoline"]

_TIMEFRAMES = ["today", "this week", "this month", "next quarter", "the next 6 months", "1 year"]

_DIRECTIONS = ["bullish", "bearish", "neutral", "positive", "negative"]

_TRANSMISSION_TEMPLATES = [
    "How does {anchor} affect {target}?",
    "What is the impact of {anchor} on {target}?",
    "How does recent {anchor} volatility move {target}?",
    "Does {anchor} predict {target}?",
    "What is the relationship between {anchor} and {target}?",
    "How do changes in {anchor} transmit to {target}?",
    "What happens to {target} when {anchor} rises?",
    "Provide quantitative proof of how {anchor} impacts {target}.",
    "Is {target} correlated with {anchor}? Show the evidence.",
    "How would a spike in {anchor} volatility affect {target}?",
]

_ANALYSIS_TEMPLATES = [
    "Is {asset} {direction} right now?",
    "What is the outlook for {asset}?",
    "Analyze {asset} for {tf}",
    "Should I buy {asset}?",
    "Compare {asset_a} and {asset_b}",
    "Which is better, {asset_a} or {asset_b}?",
    "What are the risks for {asset}?",
    "Earnings preview for {asset}",
    "Valuation of {asset}",
    "Technical analysis of {asset} over {tf}",
    "How is {sector} sector performing?",
    "Top {direction} stocks in {sector}",
    "Best {sector} stocks to watch",
    "Is {asset} overvalued?",
    "What drives {asset}?",
    "Give me a deep dive on {asset}",
    "Swing trade idea in {asset}",
    "Options strategy for {asset}",
]

_MACRO_TEMPLATES = [
    "What does the latest inflation print mean for the market?",
    "How do rising rates affect stocks?",
    "Is the economy heading into a recession?",
    "What is the Fed likely to do next?",
    "How does the dollar affect gold?",
    "What happens to bonds when inflation rises?",
    "Analysis of the yield curve",
    "How does the VIX relate to equities?",
    "What is driving the bond market?",
    "Cross-asset implications of higher rates",
]

_CRYPTO_TEMPLATES = [
    "Is bitcoin bullish this week?",
    "What is the outlook for ethereum?",
    "How does BTC correlate with tech stocks?",
    "Crypto market analysis",
    "Should I add BTC to my portfolio?",
]

_FX_TEMPLATES = [
    "Best forex pairs to trade this month",
    "EURUSD analysis",
    "How does the dollar index affect emerging markets?",
    "Outlook for GBP/USD",
    "USD/JPY technical view",
]

_BAD_TICKER_TRAP_QUERIES = [
    # Every one of these previously produced garbage tickers
    "what does the recent volatility relating to the price of oil futures affect other asset prices, provide quantitive proof and evidence to support your claim",
    "How does gold affect the stock market?",
    "Relationship between crude oil and the dollar",
    "Does rising copper predict emerging markets?",
    "How do rising rates impact growth stocks?",
    "Will rising oil hurt airlines and benefit energy?",
    "What is the outlook for AAPL stock?",
    "Compare NVDA and AMD",
    "Best biotech stocks to buy this month",
    "How does the VIX relate to equities?",
    "What happens to bonds when inflation rises?",
    "How does the dollar affect gold?",
    "What does the recent volatility of treasury yields mean for tech stocks?",
    "How would a spike in natural gas prices affect utilities?",
    "Evidence that rising silver prices lead gold",
    "Support the claim that copper leads the global economy",
    "Please prove that oil moves the dollar",
    "What are the implications of higher rates for high growth names?",
    "Show me data linking oil and the stock market",
    "Quantify the relationship between gold and real yields",
]


def _build_corpus():
    """Generate the full prompt corpus (hundreds-to-thousands)."""
    prompts = []

    # Transmission prompts: anchors x targets
    anchors = _COMMODITIES + ["rates", "the dollar", "the VIX", "bitcoin", "treasury yields", "inflation"]
    targets = ["equities", "stocks", "bonds", "the dollar", "gold", "oil stocks",
               "tech stocks", "emerging markets", "airlines", "high-yield credit",
               "growth stocks", "real estate", "banks", "commodities"]
    for tpl in _TRANSMISSION_TEMPLATES:
        for anchor, target in product(anchors[:8], targets[:8]):
            prompts.append(tpl.format(anchor=anchor, target=target))

    # Analysis prompts: assets x directions x timeframes
    for tpl in _ANALYSIS_TEMPLATES:
        if "{asset_a}" in tpl and "{asset_b}" in tpl:
            for a, b in zip(_ASSETS, _ASSETS[1:] + _ASSETS[:1]):
                prompts.append(tpl.format(asset_a=a, asset_b=b))
        elif "{asset}" in tpl and "{direction}" in tpl:
            for asset, direction in product(_ASSETS[:10], _DIRECTIONS):
                prompts.append(tpl.format(asset=asset, direction=direction))
        elif "{asset}" in tpl and "{tf}" in tpl:
            for asset, tf in product(_ASSETS[:10], _TIMEFRAMES[:4]):
                prompts.append(tpl.format(asset=asset, tf=tf))
        elif "{sector}" in tpl and "{direction}" in tpl:
            for sector, direction in product(_SECTORS[:6], _DIRECTIONS[:2]):
                prompts.append(tpl.format(sector=sector, direction=direction))
        elif "{sector}" in tpl:
            for sector in _SECTORS[:6]:
                prompts.append(tpl.format(sector=sector))
        elif "{asset}" in tpl:
            for asset in _ASSETS:
                prompts.append(tpl.format(asset=asset))

    prompts.extend(_MACRO_TEMPLATES)
    prompts.extend(_CRYPTO_TEMPLATES)
    prompts.extend(_FX_TEMPLATES)
    prompts.extend(_BAD_TICKER_TRAP_QUERIES)
    return list(dict.fromkeys(prompts))


CORPUS = _build_corpus()

# Words that must NEVER appear as extracted tickers (they are prose)
_NEVER_TICKERS = {
    "OIL", "CRUDE", "WTI", "GOLD", "SILVER", "COPPER", "PROOF", "CLAIM",
    "AFFECT", "AFFECTS", "EVIDENCE", "SUPPORT", "PROVIDE", "RELATING",
    "VOLATILITY", "VOLATILE", "FUTURES", "QUANTITATIVE", "QUANTITIVE",
    "BUY", "SELL", "MONTH", "WEEK", "YEAR", "RATES", "PRICES", "ASSET",
    "MARKET", "STOCKS", "BONDS", "SECTOR", "OUTLOOK", "ANALYZE", "ANALYSIS",
    "COMPARE", "BETTER", "SHOULD", "RIGHT", "PREVIEW", "VALUATION", "TECHNICAL",
    "PERFORMING", "WATCH", "DRIVES", "MEAN", "HAPPENS", "IMPACT", "IMPLICATIONS",
    "RISING", "HIGHER", "RELATIONSHIP", "BETWEEN", "DOES", "HOW", "WHAT", "WHEN",
    "SPIKE", "LEAD", "PROVE", "PREDICT", "MOVES", "HURT", "BENEFIT", "ENERGY",
}


@pytest.fixture(scope="module")
def corpus():
    return CORPUS


def test_corpus_is_large():
    """The harness must actually exercise hundreds-to-thousands of prompts."""
    assert len(CORPUS) >= 400, f"corpus too small: {len(CORPUS)}"
    print(f"\n[coverage] corpus size: {len(CORPUS)} prompts")


def test_no_english_words_as_tickers(corpus):
    """No prose word may survive as an extracted ticker."""
    bad = []
    for q in corpus:
        _, tickers, _ = expand_query_intents(q)
        for t in tickers:
            tu = str(t).upper()
            if tu in _NEVER_TICKERS:
                bad.append((q, t))
    assert not bad, f"prose words extracted as tickers: {bad[:8]}"


def test_no_fx_mangling_of_english_words(corpus):
    """6-letter English words (AFFECT, PROOF, CLAIM, ...) must never become
    =X pseudo-tickers."""
    bad = []
    for q in corpus:
        _, tickers, _ = expand_query_intents(q)
        for t in tickers:
            tu = str(t).upper()
            if tu.endswith("=X") and tu[:-2] in {
                "AFFECT", "PROOF", "CLAIM", "SUPPORT", "EVIDENCE", "MARKET",
                "SECTOR", "STOCKS", "BONDS", "FUTURE", "PRICES", "ASSETS",
                "BETTER", "RISING", "IMPACT", "HAPPEN", "DOES", "SHOULD",
            }:
                bad.append((q, t))
    assert not bad, f"FX-mangled English words: {bad[:8]}"


def test_commodity_names_map_to_futures():
    cases = {
        "oil": "CL=F",
        "crude oil": "CL=F",
        "gold": "GC=F",
        "silver": "SI=F",
        "copper": "HG=F",
        "natural gas": "NG=F",
        "wheat": "ZW=F",
    }
    for name, expected in cases.items():
        _, tickers, _ = expand_query_intents(f"How does {name} affect stocks?")
        assert expected in tickers, f"{name} -> expected {expected}, got {tickers}"


def test_index_and_vol_aliases_resolve():
    _, t1, _ = expand_query_intents("How does the VIX relate to equities?")
    assert "^VIX" in t1, t1
    _, t2, _ = expand_query_intents("DXY outlook")
    assert "DX-Y.NYB" in t2, t2


def test_explicit_tickers_preserved():
    for asset in _ASSETS[:8]:
        _, tickers, _ = expand_query_intents(f"What is the outlook for {asset}?")
        assert asset in tickers, f"{asset} not preserved: {tickers}"
    # Regression: COST (Costco) must not be blocked as a stopword
    _, t_cost, _ = expand_query_intents("Is COST undervalued?")
    assert "COST" in t_cost, f"COST blocked: {t_cost}"


def test_transmission_intent_detection():
    """Cross-asset transmission queries must be flagged."""
    trans_queries = [
        "How does oil affect stocks?",
        "What is the impact of gold on the dollar?",
        "How does recent oil volatility move equities?",
        "Does copper predict emerging markets?",
        "What happens to bonds when inflation rises?",
        "How do rising rates impact growth stocks?",
        "Will rising oil hurt airlines and benefit energy?",
        "Provide quantitative proof of how oil impacts the market.",
    ]
    non_trans = [
        "Is AAPL bullish right now?",
        "What is the outlook for MSFT?",
        "Compare NVDA and AMD",
        "Best biotech stocks to buy this month",
        "Earnings preview for TSLA",
        # Policy questions are NOT cross-asset transmission
        "Does the Fed cut rates next year?",
        "What will the FOMC decide at the next meeting?",
        "Is the Fed likely to pause?",
    ]
    for q in trans_queries:
        intents, _, _ = expand_query_intents(q)
        assert intents["transmission"], f"transmission not detected: {q!r}"
    for q in non_trans:
        intents, _, _ = expand_query_intents(q)
        assert not intents["transmission"], f"false transmission: {q!r}"


def test_transmission_engine_no_fabrication_without_data(monkeypatch):
    """If the anchor data is unavailable the engine returns None (never
    fabricates numbers)."""
    import cross_asset_transmission as cat

    class NoData:
        def get_returns(self, sym, period="2y"):
            return None
        def get_close(self, sym, period="2y"):
            return None

    ana = cat.CrossAssetTransmissionAnalyzer.__new__(cat.CrossAssetTransmissionAnalyzer)
    ana.data = NoData()
    ana._result_cache = {}
    ana._lock = __import__("threading").Lock()

    assert ana.analyze("CL=F", "WTI crude oil") is None


def test_transmission_engine_recovers_known_relationships():
    """Synthetic data with known betas must be recovered by the engine."""
    import numpy as np
    import threading
    import cross_asset_transmission as cat

    rng = np.random.default_rng(42)
    n = 520
    idx = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    oil_ret = rng.normal(0.0002, 0.02, n)
    data = {
        "CL=F": pd.Series(100 * np.exp(np.cumsum(oil_ret)), index=idx),
        "SPY": pd.Series(400 * np.exp(np.cumsum(0.3 * oil_ret + rng.normal(0.0002, 0.008, n))), index=idx),
        "XLE": pd.Series(80 * np.exp(np.cumsum(1.5 * oil_ret + rng.normal(0.0001, 0.01, n))), index=idx),
        "JETS": pd.Series(20 * np.exp(np.cumsum(-1.0 * oil_ret + rng.normal(0.0, 0.015, n))), index=idx),
        "GLD": pd.Series(180 * np.exp(np.cumsum(-0.1 * oil_ret + rng.normal(0.0, 0.009, n))), index=idx),
    }

    class FakeData:
        def get_returns(self, sym, period="2y"):
            s = data.get(sym)
            return s.pct_change().dropna() if s is not None else None
        def get_close(self, sym, period="2y"):
            return data.get(sym)

    ana = cat.CrossAssetTransmissionAnalyzer.__new__(cat.CrossAssetTransmissionAnalyzer)
    ana.data = FakeData()
    ana._result_cache = {}
    ana._lock = threading.Lock()

    res = ana.analyze("CL=F", "WTI crude oil")
    assert res is not None
    by_sym = {r["symbol"]: r for r in res["rows"]}
    assert by_sym["XLE"]["corr"] > 0.8, by_sym["XLE"]["corr"]
    assert by_sym["XLE"]["beta"] > 1.0, by_sym["XLE"]["beta"]
    assert by_sym["JETS"]["corr"] < -0.5, by_sym["JETS"]["corr"]
    assert by_sym["SPY"]["corr"] > 0.3, by_sym["SPY"]["corr"]

    report = ana.build_report("How does oil affect markets?", "CL=F", "WTI crude oil", res)
    assert "Cross-Asset Transmission" in report
    assert "Caveats" in report
    assert "XLE" in report
    # Every claim is backed by a measured number
    assert re.search(r"correlation\s*\*\*[+-]?[\d.]+", report)


def test_generate_cross_asset_transmission_none_without_anchor():
    """Queries without an anchor (single-stock etc.) must return None."""
    import cross_asset_transmission as cat
    assert cat.generate_cross_asset_transmission("What is the outlook for AAPL stock?") is None
    assert cat.generate_cross_asset_transmission("") is None
