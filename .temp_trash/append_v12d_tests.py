import io

tests = '''

# ---------------------------------------------------------------------------
# v12d: 'setup for X' phrases must not leak into the ambiguous-token path —
# prose like 'the setup for key support levels' must never resolve KEY/AI/SUM.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q", [
    "what is the setup for key support levels",
    "the setup for AI infrastructure spend",
    "setup for SUM of the parts",
    "the setup for all users is ready",
])
def test_v12_setup_prose_never_resolves_ambiguous_tokens(q):
    """'setup for X' is security context ONLY on the strict stopword path;
    ambiguous concept tokens (KEY/AI/SUM/ALL) must never resolve from prose."""
    from financial_llm_engine import expand_query_intents
    _i, tickers, _s = expand_query_intents(q)
    leaked = [str(t) for t in tickers if str(t).upper() in ("KEY", "AI", "SUM", "ALL")]
    assert not leaked, (q, tickers)


@pytest.mark.parametrize("q,tok", [
    ("Give me a trade setup for KEY stock", "KEY"),
    ("Give me a trade setup for AI stock", "AI"),
])
def test_v12_ambiguous_ticker_still_resolves_with_own_phrase(q, tok):
    """Ambiguous tokens keep resolving via their own security phrases
    ('KEY stock', 'AI stock') — the setup gate must not break those."""
    from financial_llm_engine import expand_query_intents
    _i, tickers, _s = expand_query_intents(q)
    assert tok in [str(t).upper() for t in tickers], (q, tickers)
'''

with io.open('tests/test_llm_and_models_fixes.py', 'a') as f:
    f.write(tests)
print("APPEND_OK")
