"""
Tests for the tool router (tool_router.py): user prompts that explicitly ask
for one of Octavian's built-in analytical engines (DCF, reverse DCF, Bayesian
network, Markov/HMM regime, dark pool, 13F, correlation, options greeks,
factor crowding, market regime) are outsourced to the real engine and
repackaged as the chatbot's answer.

Checks:
  * Explicit tool vocabulary triggers exactly the right tool(s).
  * Generic asks / existing specialist asks (hedge, probability, options
    strategy, plain "hmm") are NOT hijacked.
  * Each runner produces engine-grounded output (or None, never a crash).
  * route_tool_query returns None when nothing matches.
  * The router is wired into generate_financial_analysis so tool asks
    produce tool answers end-to-end.
"""

import os

import pytest

os.environ.setdefault("OCTAVIAN_OFFLINE", "1")


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q,expected", [
    ("run a DCF on NVDA", ["DCF"]),
    ("build a discounted cash flow model for AAPL", ["DCF"]),
    ("what does a reverse DCF imply for NVDA", ["Reverse DCF"]),
    ("use the Bayesian network to assess an inflation shock", ["Bayesian network"]),
    ("fit a Markov model to SPY returns", ["Markov / HMM regime"]),
    ("dark pool flow for AAPL", ["Dark pool"]),
    ("13F positioning on NVDA", ["Institutional 13F"]),
    ("correlation between AAPL and MSFT", ["Correlation"]),
    ("compute black-scholes greeks for NVDA", ["Options greeks"]),
    ("factor crowding on NVDA", ["Factor crowding"]),
    ("what is the current market regime", ["Market regime"]),
])
def test_detect_tool_requests(q, expected):
    from tool_router import detect_tool_requests
    assert detect_tool_requests(q) == expected, q


@pytest.mark.parametrize("q", [
    "what do you think about NVDA stock",
    "hedge my XOM position against a market correction",
    "what options strategy makes sense for LOW before earnings?",
    "hmm, what is the outlook for interest rates",
    "probability of NVDA reaching 300 in 6 months",
    "Is the dollar strengthening?",
    "Which technology stocks are undervalued?",
])
def test_detect_tool_requests_no_false_positives(q):
    from tool_router import detect_tool_requests
    assert detect_tool_requests(q) == [], q


def test_reverse_dcf_not_double_matched_as_dcf():
    """'reverse DCF' must route to the Reverse DCF tool only, never also to
    the plain DCF tool (which would run a second full valuation)."""
    from tool_router import detect_tool_requests
    assert detect_tool_requests("run a reverse DCF on NVDA") == ["Reverse DCF"]


def test_hmm_requires_model_context():
    """Bare 'hmm' (a filler word) must not trigger the Markov tool; only
    'hmm model/regime' does."""
    from tool_router import detect_tool_requests
    assert detect_tool_requests("hmm, what do you think?") == []
    assert detect_tool_requests("fit an hmm model to SPY") == ["Markov / HMM regime"]


# ---------------------------------------------------------------------------
# runners
# ---------------------------------------------------------------------------

def test_route_tool_query_none_when_no_match():
    from tool_router import route_tool_query
    assert route_tool_query("what do you think about NVDA", ["NVDA"], [], {}) is None
    assert route_tool_query("", ["NVDA"], [], {}) is None


def test_bayesian_network_runner():
    """The Bayesian network runner must produce a table of propagated
    states without any network dependency."""
    from tool_router import route_tool_query
    out = route_tool_query("use the Bayesian network to assess an inflation shock",
                           [], [], {})
    assert out is not None
    assert "Bayesian Network" in out
    assert "Inflation" in out and "Interest Rates" in out
    assert "Posterior" in out


def test_market_regime_runner():
    from tool_router import route_tool_query
    out = route_tool_query("what is the current market regime", [], [], {})
    assert out is not None
    assert "Market Regime Context" in out
    assert "Volatility regime" in out and "Risk mode" in out


def test_dcf_runner_offline_safe():
    """DCF must either produce engine output or None — never raise, and never
    fabricate when fundamentals are unavailable."""
    from tool_router import route_tool_query
    # No mocked pipeline here: fundamentals fetch will fail offline, so the
    # runner must degrade to None gracefully (normal pipeline takes over).
    out = route_tool_query("run a DCF on NVDA", ["NVDA"], [], {})
    assert out is None or "DCF Valuation" in out


# ---------------------------------------------------------------------------
# end-to-end wiring into generate_financial_analysis
# ---------------------------------------------------------------------------

def test_generate_financial_analysis_routes_dcf_tool():
    from chatbot_eval.pipeline import mocked_pipeline
    from financial_llm_engine import generate_financial_analysis
    with mocked_pipeline():
        r = generate_financial_analysis("run a DCF on NVDA") or ""
    assert "DCF Valuation" in r
    assert "InstitutionalDCFEngine" in r


def test_generate_financial_analysis_routes_bayesian():
    from chatbot_eval.pipeline import mocked_pipeline
    from financial_llm_engine import generate_financial_analysis
    with mocked_pipeline():
        r = generate_financial_analysis(
            "use the Bayesian network to assess an inflation shock") or ""
    assert "Bayesian Network" in r
    assert "Posterior" in r


def test_generate_financial_analysis_not_hijacked_for_generic():
    """Generic asks keep their normal pipeline output (executive summary),
    never a tool section."""
    from chatbot_eval.pipeline import mocked_pipeline
    from financial_llm_engine import generate_financial_analysis
    with mocked_pipeline():
        r = generate_financial_analysis("what do you think about NVDA stock") or ""
    assert "Executive Summary" in r
    assert "DCF Valuation" not in r


def test_generate_financial_analysis_multi_tool_combines():
    """A single query naming two tools runs both engines and combines their
    output into one answer."""
    from chatbot_eval.pipeline import mocked_pipeline
    from financial_llm_engine import generate_financial_analysis
    with mocked_pipeline():
        r = generate_financial_analysis(
            "use the Bayesian network and check the market regime") or ""
    assert "Bayesian Network" in r
    assert "Market Regime Context" in r
