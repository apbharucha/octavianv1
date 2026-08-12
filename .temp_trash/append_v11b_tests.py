tests = '''

def test_v11_mega_distinct_instruments_same_label_both_kept():
    """Two 'outlook for X' asks under the same 'Analysis' label but with
    DIFFERENT instruments (VIX vs PYPL, GILD vs VIX, LI vs Nasdaq) must both
    be answered — template text overlap must never drop a distinct ask."""
    from financial_llm_engine import _decompose_mega_query, _build_mega_response
    for q in [
        "what is the outlook for the VIX?. Next, what is the outlook for PYPL?",
        "what is the outlook for LI?. And since we are on the topic, what is the outlook for the Nasdaq?",
        "what is the outlook for GILD?. Meanwhile, what is the outlook for the VIX?",
    ]:
        parts = _decompose_mega_query(q) or []
        assert len(parts) >= 2, (q, parts)
        out = _build_mega_response(q, parts, None) or ""
        assert out.count("### Part ") >= 2, (q, out)


def test_v11_mega_same_instrument_deduped():
    """Two asks about the SAME instrument under one label are a re-ask and
    must collapse to a single part."""
    from financial_llm_engine import _build_mega_response
    parts = ["what is the outlook for AAPL", "what is the outlook for AAPL next quarter"]
    out = _build_mega_response(" ".join(parts), parts, None) or ""
    assert out.count("### Part ") == 1, out
'''
open("tests/test_llm_and_models_fixes.py", "a").write(tests)
print("APPENDED v11b")
