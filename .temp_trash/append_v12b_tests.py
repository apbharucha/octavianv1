import io

tests = '''

@pytest.mark.parametrize("q,tok", [
    ("Give me a trade setup for ARE with entry, stop and target.", "ARE"),
    ("Give me a trade setup for ALL with entry, stop and target.", "ALL"),
    ("trade setup for CAN with entry and stop", "CAN"),
])
def test_v12_trade_setup_phrase_resolves_stopword_ticker(q, tok):
    """'trade setup for X' is unambiguous security context — stopword tickers
    (ARE/ALL/CAN) must resolve in it (this closed the last 13/100k stress
    failures)."""
    from financial_llm_engine import expand_query_intents
    _i, tickers, _s = expand_query_intents(q)
    assert tok in [str(t).upper() for t in tickers], (q, tickers)


def test_v12_trade_setup_prose_never_resolves():
    """'the setup for all users is ready' is prose — lowercase 'all' must not
    resolve even with the trade-setup phrase present."""
    from financial_llm_engine import expand_query_intents
    _i, tickers, _s = expand_query_intents("the setup for all users is ready")
    assert not [str(t) for t in tickers if str(t).upper() == "ALL"], tickers
'''

with io.open('tests/test_llm_and_models_fixes.py', 'a') as f:
    f.write(tests)
print("APPEND_OK")
