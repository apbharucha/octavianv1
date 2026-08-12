tests = '''

def test_v11_play_and_protect_gains_extract_tickers():
    """'best way to play TGT' / 'protect gains in TGT' must resolve TGT/LOW
    as securities (verb+preposition framing), not drop them."""
    from financial_llm_engine import expand_query_intents
    for q, want in [
        ("What is the best way to play TGT this week?", ["TGT"]),
        ("What is the best way to protect gains in TGT?", ["TGT"]),
        ("What is the best way to protect gains in LOW?", ["LOW"]),
        ("How to play LOW into earnings?", ["LOW"]),
    ]:
        _, t, _ = expand_query_intents(q)
        assert sorted(t) == sorted(want), (q, t)


def test_v11_play_phrase_no_false_positive():
    """'play the market' / plain prose never resolves an ambiguous ticker."""
    from financial_llm_engine import _is_equity_reference
    assert not _is_equity_reference("What is the best way to play the market this week?", "TGT")
    assert not _is_equity_reference("gains in the sector are broad", "TGT")


def test_v11_mega_imperative_boundary_split():
    """'...target compare JNJ and GOOG give me the key support' must split into
    setup / comparison / technical parts without punctuation."""
    from financial_llm_engine import _decompose_mega_query, expand_query_intents
    parts = _decompose_mega_query(
        "what is the probability UBER reaches $41 in 24 months?. Meanwhile, give "
        "me a trade setup for BIIB with entry, stop and target compare JNJ and "
        "GOOG give me the key support and resistance") or []
    assert len(parts) >= 4, parts
    for p in parts:
        _, t, i = expand_query_intents(p)
        pl = p.lower()
        if "probability" in pl:
            assert "UBER" in t, (p, t)
        if "setup" in pl:
            assert "BIIB" in t, (p, t)
        if "compare" in pl:
            assert {"JNJ", "GOOG"} <= set(t), (p, t)
        if "key support" in pl:
            assert any(w in pl for w in ("support", "resistance")), p


def test_v11_mega_imperative_boundary_no_false_split():
    """Task words followed by plain prose never create bogus mega-parts."""
    from financial_llm_engine import _decompose_mega_query
    for q in [
        "entry, stop and target are the key levels to watch",
        "what is the best way to play the market this week?",
    ]:
        parts = _decompose_mega_query(q)
        assert parts is None, (q, parts)


def test_v11_mega_two_macro_parts_both_kept():
    """Two genuinely different macro asks with the same label must both be
    answered — the label de-dup only drops near-duplicate prose."""
    from financial_llm_engine import _decompose_mega_query, _build_mega_response
    q = "should i own long-duration bonds right now?. On a different note, what is the outlook for quantitative tightening?"
    parts = _decompose_mega_query(q) or []
    assert len(parts) >= 2, parts
    out = _build_mega_response(q, parts, None) or ""
    assert out.count("### Part ") >= 2, out
    assert "quantitative tightening" in out.lower(), out
    assert "bonds" in out.lower(), out


def test_v11_mega_near_duplicate_parts_still_deduped():
    """A re-asked/near-duplicate part under the same label IS still dropped."""
    from financial_llm_engine import _decompose_mega_query, _build_mega_response
    parts = [
        "what is the outlook for the economy right now",
        "what is the outlook for the economy at this moment",
    ]
    out = _build_mega_response(" ".join(parts), parts, None) or ""
    assert out.count("### Part ") == 1, out
'''
open("tests/test_llm_and_models_fixes.py", "a").write(tests)
print("APPENDED")
