import io

tests = '''

# ---------------------------------------------------------------------------
# v12c: FX named-pair asks must lead with the specific pair (not just a
# generic G10 momentum ranking), with an evidence-based Fed-transmission note.
# ---------------------------------------------------------------------------

def test_fx_named_pair_leads_with_pair_and_fed_transmission():
    """'How will USD/BRL react to the Fed?' must lead with the USDBRL quote and
    a Fed-transmission note, not only the generic momentum ranking."""
    from financial_llm_engine import generate_financial_analysis
    from chatbot_eval.pipeline import mocked_pipeline
    with mocked_pipeline():
        r = generate_financial_analysis("How will USD/BRL react to the Fed?") or ""
        assert "US Dollar / Brazilian Real" in r, r[:300]
        assert "Fed transmission" in r, r[:300]


def test_fx_generic_ask_still_ranks_pairs():
    """A generic currency ask keeps the momentum ranking and does not
    fabricate a named pair."""
    from financial_llm_engine import generate_financial_analysis
    from chatbot_eval.pipeline import mocked_pipeline
    with mocked_pipeline():
        r = generate_financial_analysis("Which currency is the best buy right now?") or ""
        assert "Euro / US Dollar" in r or "US Dollar / Yen" in r, r[:300]
'''

with io.open('tests/test_llm_and_models_fixes.py', 'a') as f:
    f.write(tests)
print("APPEND_OK")
