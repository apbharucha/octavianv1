import io

tests = '''

# ---------------------------------------------------------------------------
# v12: stopword tickers (ARE/ALL/AM/CAN/...) resolve only when UPPERCASE +
# explicit security context; prose plurals and verb forms never do.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q,tok", [
    ("buy ARE", "ARE"),
    ("ARE stock analysis", "ARE"),
    ("outlook for ARE", "ARE"),
    ("Is ARE undervalued at current levels", "ARE"),
    ("Allstate ALL stock", "ALL"),
    ("buy ALL shares", "ALL"),
    ("ALL stock vs TGT", "ALL"),
    ("play RUN this week", "RUN"),
    ("ARE or NVDA which is better", "ARE"),
])
def test_v12_stopword_ticker_resolves_with_uppercase_security_context(q, tok):
    """Genuine universe tickers that are also core English words (ARE, ALL,
    RUN, ...) must resolve when written in UPPERCASE with explicit security
    context — previously they were unconditionally dropped by _STOPWORDS."""
    from financial_llm_engine import expand_query_intents
    _i, tickers, _s = expand_query_intents(q)
    assert tok in [str(t).upper() for t in tickers], (q, tickers)


@pytest.mark.parametrize("q", [
    "the are of investing",
    "all stocks are down",
    "can you analyze the market",
    "how are stocks doing today",
    "how much are prices going up",
    "Which technology stocks are undervalued?",
    "real estate is expensive",
    "buy are",          # lowercase ticker = prose, per case-sensitivity rule
    "there are no good buys today",
])
def test_v12_stopword_ticker_never_leaks_from_prose(q):
    """Prose uses of stopword tickers ('are', 'all', 'can') must never resolve
    — even in equity-adjacent phrasing like 'stocks are undervalued' or
    'all stocks' (plural). Lowercase tickers are prose by convention."""
    from financial_llm_engine import expand_query_intents
    _i, tickers, _s = expand_query_intents(q)
    leaked = [str(t) for t in tickers if str(t).upper() in ("ARE", "ALL", "AM", "CAN")]
    assert not leaked, (q, tickers)


def test_v12_gold_remains_commodity_not_equity():
    """'gold' (lowercase) must stay a commodity (GC=F) and never resolve as
    the GOLD equity ticker via the stopword escape."""
    from financial_llm_engine import expand_query_intents
    _i, tickers, _s = expand_query_intents("What happens to TGT and gold if the Fed cuts rates")
    uppers = [str(t).upper() for t in tickers]
    assert "TGT" in uppers, tickers
    assert "GOLD" not in uppers, tickers
    assert any("GC=F" == str(t) for t in tickers), tickers
'''

with io.open('tests/test_llm_and_models_fixes.py', 'a') as f:
    f.write(tests)
print("APPEND_OK")
