"""Prompt corpus factory — generates thousands of test queries.

Every prompt carries structured *expectations* so the rubric can score task
fulfillment objectively:

    expectation keys
      primary            — symbol/entity the response must engage with
      requires           — substrings the response should contain
      requires_any       — at least one of these substrings
      avoid              — substrings the response must NOT contain (garbage)
      risk_query         — risk/hedging content is mandatory
      setup_query        — entry/stop/target content is mandatory
      transmission_query — correlation/beta/lead-lag evidence is mandatory
      category_flavor    — description for issue grouping

Deterministic: `random.Random(seed)` is used so corpus + run order reproduce.
"""

import random

# ---------------------------------------------------------------------------
# Value pools
# ---------------------------------------------------------------------------

TICKERS = [
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "GOOG", "META", "AMD",
    "INTC", "NFLX", "JPM", "BAC", "GS", "MS", "BLK", "SCHW", "WFC", "C",
    "XOM", "CVX", "COP", "OXY", "SLB", "HAL", "BKR", "XLE", "CAT", "DE",
    "BA", "GE", "LMT", "NOC", "RTX", "UNH", "JNJ", "PFE", "MRK", "ABBV",
    "LLY", "TMO", "WMT", "COST", "HD", "MCD", "SBUX", "NKE", "DIS", "KO",
    "PEP", "PG", "V", "MA", "AXP", "CRM", "ORCL", "ADBE", "CSCO", "QCOM",
    "MU", "TSM", "SHOP", "SQ", "PYPL", "UBER", "LYFT", "ABNB", "DAL",
    "UAL", "AAL", "JETS", "TGT", "LOW", "CVS", "GILD", "AMGN", "BIIB",
    "MRNA", "SPY", "QQQ", "DIA", "IWM", "GLD", "SLV", "TLT", "HYG", "LQD",
    "EEM", "VWO", "FXI", "EWZ", "ARKK", "COIN", "MSTR", "HOOD", "RIVN",
    "LCID", "NIO", "XPEV", "LI", "F", "GM", "STLA", "ROKU", "PINS", "SNAP",
    "XLK", "XLF", "XLV", "XLY", "XLP", "XLU", "XLI", "XLB", "XLRE", "XLC",
]

SECTORS = [
    "technology", "software", "semiconductors", "biotech", "healthcare",
    "pharma", "energy", "oil and gas", "financials", "banks", "consumer",
    "retail", "industrials", "defense", "aerospace", "materials", "mining",
    "utilities", "real estate", "communication", "media", "autos", "airlines",
]

COMMODITIES = [
    ("oil", "CL=F"), ("crude oil", "CL=F"), ("wti", "CL=F"),
    ("brent", "BZ=F"), ("natural gas", "NG=F"), ("gasoline", "RB=F"),
    ("heating oil", "HO=F"), ("gold", "GC=F"), ("silver", "SI=F"),
    ("copper", "HG=F"), ("platinum", "PL=F"), ("palladium", "PA=F"),
    ("wheat", "ZW=F"), ("corn", "ZC=F"), ("soybeans", "ZS=F"),
    ("coffee", "KC=F"), ("sugar", "SB=F"), ("cotton", "CT=F"), ("cocoa", "CC=F"),
]

FUTURES_SYMBOLS = [
    "CL=F", "BZ=F", "NG=F", "RB=F", "HO=F", "GC=F", "SI=F", "HG=F",
    "PL=F", "PA=F", "ZW=F", "ZC=F", "ZS=F", "KC=F", "SB=F", "CT=F", "CC=F",
    "ES=F", "NQ=F", "YM=F", "RTY=F", "ZN=F", "ZB=F", "ZF=F", "ZT=F",
]

FX_PAIRS = [
    "EUR/USD", "USD/JPY", "GBP/USD", "AUD/USD", "USD/CAD", "USD/CHF",
    "NZD/USD", "EUR/GBP", "EUR/JPY", "GBP/JPY", "AUD/JPY", "USD/CNY",
    "USD/INR", "USD/MXN", "USD/BRL", "EUR/JPY", "AUD/CAD", "GBP/AUD",
    "USD/SGD", "USD/KRW",
]

CRYPTO = [
    ("bitcoin", "BTC-USD"), ("ethereum", "ETH-USD"), ("solana", "SOL-USD"),
    ("dogecoin", "DOGE-USD"), ("xrp", "XRP-USD"), ("cardano", "ADA-USD"),
    ("polkadot", "DOT-USD"), ("litecoin", "LTC-USD"), ("chainlink", "LINK-USD"),
    ("avalanche", "AVAX-USD"), ("polygon", "MATIC-USD"), ("shiba", "SHIB-USD"),
    ("uniswap", "UNI-USD"), ("aave", "AAVE-USD"), ("aptos", "APT-USD"),
    ("sui", "SUI-USD"), ("bitcoin", "BTC-USD"),
]

INDICES = [
    ("S&P 500", "^GSPC"), ("S&P", "^GSPC"), ("SPX", "^GSPC"),
    ("Nasdaq", "^IXIC"), ("Dow", "^DJI"), ("Russell 2000", "^RUT"),
    ("VIX", "^VIX"), ("DXY", "DX-Y.NYB"), ("dollar index", "DX-Y.NYB"),
]

MACRO_TOPICS = [
    "inflation", "CPI", "PCE", "interest rates", "the Fed", "FOMC",
    "GDP growth", "unemployment", "the jobs report", "nonfarm payrolls",
    "a recession", "the yield curve", "PMI", "consumer confidence",
    "retail sales", "ISM manufacturing", "housing starts", "quantitative tightening",
    "quantitative easing", "the 10-year Treasury yield", "the money supply",
]

GEOPOLITICS = [
    "a war in the Middle East", "an escalation between Russia and Ukraine",
    "US-China tensions", "a conflict over Taiwan", "new sanctions on Russia",
    "new tariffs on Chinese goods", "a trade war", "the US presidential election",
    "a government shutdown", "OPEC+ cutting production", "OPEC+ raising production",
    "a nuclear deal with Iran", "attacks in the Red Sea", "a blockade in the Strait of Hormuz",
    "Brexit", "NATO tensions", "an EU energy crisis", "political instability in emerging markets",
]

SENTIMENT_WORDS = ["bullish", "bearish", "neutral", "positive", "negative"]

# Approximate reference prices used only to build *plausible target prices* in
# probability prompts (the engine itself always fetches real data).
REF_PRICES = {
    "AAPL": 310, "MSFT": 620, "NVDA": 175, "TSLA": 260, "AMZN": 235, "GOOGL": 205,
    "META": 640, "AMD": 130, "INTC": 21, "NFLX": 990, "JPM": 290, "BAC": 51,
    "GS": 610, "MS": 145, "BLK": 1080, "SCHW": 95, "XOM": 115, "CVX": 152,
    "COP": 102, "OXY": 48, "SLB": 43, "BKR": 62, "CAT": 355, "DE": 430,
    "BA": 185, "GE": 220, "LMT": 520, "NOC": 500, "UNH": 290, "JNJ": 165,
    "PFE": 26, "MRK": 102, "ABBV": 165, "LLY": 790, "WMT": 105, "COST": 930,
    "HD": 420, "MCD": 300, "SBUX": 85, "NKE": 82, "DIS": 118, "KO": 74,
    "PEP": 150, "PG": 175, "V": 350, "MA": 565, "AXP": 330, "CRM": 290,
    "ORCL": 175, "ADBE": 430, "CSCO": 68, "QCOM": 180, "MU": 95, "TSM": 235,
    "SHOP": 115, "SQ": 92, "PYPL": 96, "UBER": 82, "DAL": 62, "UAL": 95,
    "AAL": 16, "JETS": 23, "TGT": 135, "LOW": 250, "CVS": 62, "GILD": 92,
    "AMGN": 290, "BIIB": 135, "MRNA": 52, "SPY": 610, "QQQ": 550, "DIA": 450,
    "IWM": 225, "GLD": 245, "SLV": 29, "TLT": 90, "HYG": 78, "LQD": 109,
    "EEM": 45, "FXI": 32, "EWZ": 32, "ARKK": 55, "COIN": 320, "MSTR": 180,
    "HOOD": 95, "RIVN": 14, "LCID": 3.2, "NIO": 5.5, "F": 11, "GM": 55,
    "ROKU": 68, "PINS": 35, "SNAP": 11, "XOM": 115,
}


def _ref_price(sym: str) -> float:
    return REF_PRICES.get(sym, 100.0)


# ---------------------------------------------------------------------------
# Template library. Each entry: (category, subcategory, asset_class,
# expected_intent, templates, expectation_builder)
# ---------------------------------------------------------------------------

_TEMPLATES = []


def _reg(category, subcategory, asset_class, expected_intent, templates, exp_builder):
    _TEMPLATES.append((category, subcategory, asset_class, expected_intent,
                       templates, exp_builder))


# ── Equities ────────────────────────────────────────────────────────────────
def _exp_sym(sym):
    return {"primary": sym, "requires_any": [sym, sym.lower()]}


_reg("equity_outlook", "single stock", "equities", "single_stock", [
    "What is the outlook for {t}?",
    "Is {t} a good buy right now?",
    "Should I buy or sell {t}?",
    "Give me a {w} case for {t}",
    "What is your analysis of {t}?",
    "How should I think about {t} in my portfolio?",
], lambda s: _exp_sym(s["t"]))


_reg("equity_technical", "technical analysis", "equities", "technical", [
    "What are the key support and resistance levels for {t}?",
    "Is {t} in an uptrend or a downtrend?",
    "Analyze the price chart for {t}",
    "What does the momentum look like on {t}?",
    "Is {t} overbought or oversold?",
    "Should I buy the pullback in {t}?",
], lambda s: {**_exp_sym(s["t"]),
              "requires_any": [s["t"], "support", "resistance", "trend", "momentum"]})


_reg("equity_valuation", "valuation", "equities", "valuation", [
    "Is {t} overvalued or undervalued?",
    "What is a fair value for {t}?",
    "Analyze the valuation of {t}",
    "Is {t} expensive at current levels?",
    "What multiple should {t} trade at?",
], lambda s: {**_exp_sym(s["t"]),
              "requires_any": [s["t"], "valuation", "value", "multiple", "pe"]})


_reg("equity_earnings", "earnings", "equities", "earnings", [
    "When does {t} report earnings?",
    "Earnings preview for {t}",
    "Did {t} beat earnings expectations?",
    "What should I expect from {t} earnings?",
    "How will {t} react to its earnings report?",
], lambda s: {**_exp_sym(s["t"]),
              "requires_any": [s["t"], "earnings", "revenue", "eps"]})


_reg("equity_dividend", "dividends", "equities", "dividend", [
    "Does {t} pay a dividend?",
    "What is the dividend yield of {t}?",
    "Is {t} a good income stock?",
    "Has {t} been growing its dividend?",
], lambda s: {**_exp_sym(s["t"]),
              "requires_any": [s["t"], "dividend", "yield", "income"]})


# ── Comparison ──────────────────────────────────────────────────────────────
def _exp_pair(s):
    return {"primary": s["t1"],
            "requires": [s["t1"], s["t2"]],
            "category_flavor": "comparison"}


_reg("comparison_two", "pairwise comparison", "equities", "comparison", [
    "Compare {t1} and {t2}",
    "Which is the better investment: {t1} or {t2}?",
    "{t1} vs {t2} — which should I pick?",
    "How does {t1} compare to {t2}?",
    "Should I own {t1} or {t2}?",
], _exp_pair)


# ── Sectors ─────────────────────────────────────────────────────────────────
_reg("sector_scan", "sector scan", "equities", "sector_scan", [
    "Best {s} stocks to buy now",
    "Top {s} stocks for growth",
    "Which {s} stocks are undervalued?",
    "Scan the {s} sector for opportunities",
    "What is the outlook for the {s} sector?",
    "Give me {s} stocks with the best momentum",
], lambda s: {"primary": s["s"], "requires_any": [s["s"]],
              "category_flavor": "sector"})


_reg("sector_etf", "sector ETF", "equities", "sector_scan", [
    "How is the {etf} sector ETF performing?",
    "Is {etf} a good way to play the sector?",
    "What is inside {etf}?",
], lambda s: _exp_sym(s["etf"]))


# ── Macro ───────────────────────────────────────────────────────────────────
def _exp_macro(s):
    return {"primary": s["m"], "requires_any": [s["m"]],
            "category_flavor": "macro"}


_reg("macro_outlook", "macro outlook", "macro", "macro", [
    "What is the outlook for {m}?",
    "Where is {m} heading over the next year?",
    "What does the data say about {m}?",
    "Is {m} a risk to markets right now?",
    "Explain {m} and what it means for investors",
], _exp_macro)


# ── Cross-asset transmission ────────────────────────────────────────────────
def _exp_transmission(s):
    return {"primary": s.get("anchor"),
            "transmission_query": True,
            "requires_any": [s["anchor"]] if s.get("anchor") else None,
            "category_flavor": "transmission"}


_reg("transmission_oil", "oil transmission", "commodities", "transmission", [
    "How does the price of oil affect the stock market?",
    "What happens to airlines when oil prices rise?",
    "How does rising oil impact energy stocks?",
    "What is the relationship between oil and inflation?",
    "How do oil futures moves transmit to the broader market?",
    "Does oil predict the direction of the S&P 500?",
    "What is the correlation between oil and airline stocks?",
], lambda s: {**_exp_transmission(s), "primary": "CL=F"})


_reg("transmission_gold", "gold transmission", "commodities", "transmission", [
    "How does gold correlate with equities?",
    "What happens to gold when real rates rise?",
    "How does the dollar affect gold prices?",
    "Is gold a good hedge for stocks?",
    "What is the relationship between gold and bitcoin?",
], lambda s: {**_exp_transmission(s), "primary": "GC=F"})


_reg("transmission_rates", "rates transmission", "bonds", "transmission", [
    "How do rising rates impact growth stocks?",
    "What happens to bonds when inflation rises?",
    "How does the 10-year yield affect equities?",
    "What is the relationship between the Fed and the stock market?",
    "How do rate hikes transmit to housing stocks?",
], lambda s: {**_exp_transmission(s), "primary": "^TNX"})


_reg("transmission_dollar", "dollar transmission", "fx", "transmission", [
    "What does a stronger dollar mean for commodities?",
    "How does dollar strength affect emerging markets?",
    "What happens to US stocks when the dollar strengthens?",
    "How does the dollar index correlate with gold?",
], lambda s: {**_exp_transmission(s), "primary": "DX-Y.NYB"})


_reg("transmission_crypto", "crypto transmission", "crypto", "transmission", [
    "How does bitcoin react to Fed hikes?",
    "What is the link between crypto and inflation?",
    "Does a strong dollar hurt bitcoin?",
    "How do interest rates affect crypto valuations?",
], lambda s: {**_exp_transmission(s), "primary": "BTC-USD"})


# ── Futures / commodities forecasts ─────────────────────────────────────────
def _exp_commodity(s):
    return {"primary": s["f"], "requires_any": [s["f"], s["n"]],
            "category_flavor": "commodity"}


_reg("commodity_outlook", "commodity forecast", "commodities", "commodities", [
    "What is the price forecast for {n}?",
    "Where is {n} headed this year?",
    "Is {n} a good buy right now?",
    "What is driving {n} prices?",
    "Bull or bear case for {n}",
], _exp_commodity)


# ── Trade setups (the class the user flagged) ───────────────────────────────
def _exp_setup(s):
    return {"primary": s["t"],
            "setup_query": True,
            "risk_query": True,
            "requires_any": [s["t"]],
            "category_flavor": "trade_setup"}


_reg("setup_futures", "futures trade setup", "futures", "trade_setup", [
    "If I want to trade {t}, what is a good setup?",
    "Give me a trade plan for {t} futures",
    "What entry and stop should I use on {t}?",
    "How should I trade {t} right now?",
    "Trade idea for {t} with targets and stops",
    "What setup should I take on {t} given current volatility?",
], _exp_setup)


_reg("setup_equity", "equity trade setup", "equities", "trade_setup", [
    "Give me a trade setup for {t}",
    "What entry, stop, and target for {t}?",
    "I want to long {t} — how do I manage the risk?",
    "What is the best way to play {t} this week?",
    "Trade plan for {t} with a defined risk",
], _exp_setup)


_reg("setup_crypto", "crypto trade setup", "crypto", "trade_setup", [
    "Give me a trade setup for {t}",
    "How should I trade {t} with a stop loss?",
    "Entry and target for {t}?",
], _exp_setup)


# ── Hedging & risk management ───────────────────────────────────────────────
def _exp_hedge(s):
    return {"primary": s.get("t"),
            "risk_query": True,
            "requires_any": [s["t"]] if s.get("t") else None,
            "category_flavor": "hedging"}


_reg("hedging_portfolio", "portfolio hedging", "multi-asset", "hedging", [
    "How do I hedge my portfolio?",
    "What is the best hedge for a market downturn?",
    "How should I protect my portfolio given current volatility?",
    "What hedges work when the VIX is low?",
    "How much of my portfolio should be hedged?",
], _exp_hedge)


_reg("hedging_position", "position hedging", "equities", "hedging", [
    "How do I hedge my position in {t}?",
    "What is the best way to protect gains in {t}?",
    "Should I buy puts on {t} to hedge?",
    "How can I hedge {t} without selling?",
    "What hedge makes sense for a long {t} position?",
], _exp_hedge)


_reg("risk_management", "risk management", "multi-asset", "risk", [
    "How much should I risk per trade?",
    "What is a good position sizing rule?",
    "Where should I put my stop loss?",
    "How do I manage risk in a volatile market?",
    "What is a good risk/reward ratio to target?",
    "How should I size positions in {t}?",
], lambda s: {"primary": s.get("t"), "risk_query": True,
              "requires_any": [s.get("t", "")] if s.get("t") else None,
              "category_flavor": "risk"})


# ── Options ─────────────────────────────────────────────────────────────────
def _exp_options(s):
    return {"primary": s["t"], "requires_any": [s["t"]],
            "category_flavor": "options"}


_reg("options_strategy", "options strategy", "options", "options", [
    "What options strategy makes sense for {t} before earnings?",
    "Is a bull call spread on {t} a good idea?",
    "Should I sell covered calls on {t}?",
    "Would a straddle work on {t}?",
    "What is the best put structure to protect {t}?",
], _exp_options)


# ── FX ──────────────────────────────────────────────────────────────────────
_reg("fx_outlook", "fx outlook", "fx", "fx", [
    "What is the outlook for {fx}?",
    "Where is {fx} heading?",
    "Is the dollar strengthening?",
    "Which currency is the best buy right now?",
    "How will {fx} react to the Fed?",
], lambda s: {"primary": s.get("fx", "dollar"),
              "requires_any": [s.get("fx", "dollar")],
              "category_flavor": "fx"})


# ── Crypto ──────────────────────────────────────────────────────────────────
def _exp_crypto(s):
    return {"primary": s["f"], "requires_any": [s["f"], s["n"]],
            "category_flavor": "crypto"}


_reg("crypto_outlook", "crypto outlook", "crypto", "crypto", [
    "What is the price outlook for {n}?",
    "Is {n} a good investment?",
    "What is driving {n} right now?",
    "Bull or bear case for {n}",
    "Should I buy {n} at these levels?",
], _exp_crypto)


# ── Geopolitics / global events ─────────────────────────────────────────────
def _exp_geopol(s):
    return {"primary": s["g"], "requires_any": [s["g"]],
            "category_flavor": "geopolitics"}


_reg("geopolitics_markets", "geopolitics to markets", "macro", "geopolitics", [
    "What happens to markets if {g}?",
    "How would {g} affect oil prices?",
    "What is the market impact of {g}?",
    "How should investors position for {g}?",
    "What does {g} mean for stocks?",
    "How would {g} impact inflation and rates?",
], _exp_geopol)


_reg("geopolitics_assets", "geopolitics single asset", "equities", "geopolitics", [
    "How would {g} affect {t}?",
    "Is {t} a safe haven if {g}?",
    "How should I hedge {t} against {g}?",
], lambda s: {"primary": s["t"], "requires_any": [s["g"], s["t"]],
              "risk_query": True, "category_flavor": "geopolitics"})


# ── Current events ──────────────────────────────────────────────────────────
_reg("current_events_cpi", "CPI event", "macro", "current_events", [
    "What did the latest CPI report show?",
    "How will this week's CPI print move markets?",
    "CPI came in hot — what does it mean for stocks?",
    "Analyze the latest inflation data",
], lambda s: {"primary": "CPI", "requires_any": ["CPI", "inflation"],
              "category_flavor": "current_events"})


_reg("current_events_fomc", "FOMC event", "macro", "current_events", [
    "What did the Fed decide at the last FOMC meeting?",
    "FOMC decision preview — what to expect?",
    "How will the market react to the Fed's announcement?",
    "Did the Fed cut rates?",
], lambda s: {"primary": "Fed", "requires_any": ["Fed", "FOMC", "rate"],
              "category_flavor": "current_events"})


_reg("current_events_earnings", "earnings season", "equities", "earnings", [
    "Earnings season outlook — what to watch?",
    "Big tech earnings this week",
    "How are companies guiding for next quarter?",
], lambda s: {"primary": "earnings", "requires_any": ["earnings"],
              "category_flavor": "current_events"})


_reg("current_events_news", "news driven", "equities", "news", [
    "What is the latest news on {t}?",
    "Is the news flow bullish or bearish for {t}?",
    "What happened to {t} this week?",
], lambda s: _exp_sym(s["t"]))


# ── Portfolio / probability ─────────────────────────────────────────────────
_reg("portfolio_analysis", "portfolio analysis", "multi-asset", "portfolio", [
    "How should I analyze my portfolio?",
    "What is a good portfolio allocation for 2026?",
    "How do I rebalance my portfolio?",
    "What is the ideal stock/bond split right now?",
    "How diversified should my portfolio be?",
], lambda s: {"primary": "portfolio", "requires_any": ["portfolio", "allocat", "diversif"],
              "category_flavor": "portfolio"})


def _exp_prob(s):
    return {"primary": s["t"], "requires_any": [s["t"]],
            "category_flavor": "probability"}


_reg("probability_target", "target probability", "equities", "probability", [
    "What is the probability {t} reaches {p} within {h} months?",
    "What are the odds {t} hits {p} by the end of the year?",
    "How likely is {t} to reach {p} in {h} days?",
    "What is the chance {t} doubles in a year?",
], lambda s: {**_exp_prob(s),
              "target_price": s.get("p"), "horizon": s.get("h")})


_reg("probability_downside", "downside probability", "equities", "probability", [
    "What is the probability {t} falls to {p}?",
    "How likely is {t} to drop {k}% in the next {h} months?",
    "What are the chances of a drawdown in {t}?",
], lambda s: {**_exp_prob(s)})


# ── Indices / bonds / indicators ────────────────────────────────────────────
_reg("index_outlook", "index outlook", "indices", "macro", [
    "What is the outlook for the {idx}?",
    "Where will the {idx} be in 12 months?",
    "Is the {idx} overbought?",
    "What is driving the {idx} right now?",
], lambda s: {"primary": s["f"], "requires_any": [s["idx"], s["f"]],
              "category_flavor": "index"})


_reg("bonds_outlook", "bond markets", "bonds", "macro", [
    "What is the outlook for the 10-year Treasury yield?",
    "Is the yield curve still inverted?",
    "What happens to bond prices when the Fed cuts?",
    "Should I own long-duration bonds right now?",
    "How does the treasury market look?",
], lambda s: {"primary": "bonds", "requires_any": ["bond", "yield", "treasury"],
              "category_flavor": "bonds"})


_reg("indicators_outlook", "economic indicators", "macro", "macro", [
    "What is the outlook for GDP growth?",
    "What does the latest PMI data mean?",
    "How do housing starts affect the economy?",
    "What is consumer confidence telling us?",
], lambda s: {"primary": s.get("m", ""),
              "requires_any": [s["m"]] if s.get("m") else None,
              "category_flavor": "indicators"})


# ── Facts / knowledge ───────────────────────────────────────────────────────
_reg("company_facts", "company facts", "equities", "knowledge", [
    "What does {t} do?",
    "Tell me about {t} the company",
    "What sector is {t} in?",
    "Who are {t}'s main competitors?",
    "What is {t}'s business model?",
], lambda s: {**_exp_sym(s["t"]), "category_flavor": "knowledge"})


# ── MEGA PROMPTS (multi-task) ───────────────────────────────────────────────
# Each fragment: (query_template, expectation_template, flags). Placeholders
# are filled by `_fill`; expectation values are substituted from the same ctx.
_MEGA_FRAGMENTS = [
    ("what is the outlook for {t}?",
     {"requires_any": ["{t}"], "label": "outlook"}, {}),
    ("give me the key support and resistance levels for {t}",
     {"requires_any": ["{t}", "support", "resistance"], "label": "technicals"}, {}),
    ("is {t} overbought or oversold?",
     {"requires_any": ["{t}", "rsi", "overbought", "oversold", "momentum"],
      "label": "momentum"}, {}),
    ("is {t} overvalued or undervalued?",
     {"requires_any": ["{t}", "valuation", "value", "multiple", "fair value"],
      "label": "valuation"}, {}),
    ("earnings preview for {t}",
     {"requires_any": ["{t}", "earnings"], "label": "earnings"}, {}),
    ("does {t} pay a dividend?",
     {"requires_any": ["{t}", "dividend", "yield"], "label": "dividend"}, {}),
    ("what is the probability {t} reaches {p} in {h} months?",
     {"requires_any": ["probability", "odds", "chance", "likely"],
      "primary": "{t}", "label": "probability"}, {}),
    ("how likely is {t} to fall to {p} in {h} months?",
     {"requires_any": ["probability", "odds", "chance", "drop", "fall"],
      "primary": "{t}", "label": "downside"}, {}),
    ("how should i hedge a long {t} position?",
     {"requires_any": ["hedge", "hedging", "puts", "protection", "covered"],
      "primary": "{t}", "label": "hedge"}, {"risk_query": True}),
    ("give me a trade setup for {t} with entry, stop and target",
     {"requires_any": ["{t}", "entry", "stop", "target"],
      "label": "setup"}, {"setup_query": True}),
    ("what is the best options strategy for {t} before earnings?",
     {"requires_any": ["{t}", "call", "put", "straddle", "spread"],
      "label": "options"}, {}),
    ("should i sell covered calls on {t}?",
     {"requires_any": ["{t}", "call", "covered"], "label": "options"}, {}),
    ("compare {t1} and {t2}",
     {"requires": ["{t1}", "{t2}"], "label": "comparison"}, {}),
    ("which is the better investment, {t1} or {t2}?",
     {"requires": ["{t1}", "{t2}"], "label": "comparison"}, {}),
    ("what are the best {s} stocks to buy now?",
     {"requires_any": ["{s}"], "label": "sector"}, {}),
    ("what is the outlook for {m}?",
     {"requires_any": ["{m}"], "label": "macro"}, {}),
    ("how would {g} affect markets?",
     {"requires_any": ["{g}"], "label": "geopolitics"}, {}),
    ("what is the price forecast for {n}?",
     {"requires_any": ["{n}", "{f}"], "label": "commodity"}, {}),
    ("how does the price of {n} affect the stock market?",
     {"requires_any": ["{n}", "{f}"], "label": "transmission"},
     {"transmission_query": True}),
    ("what is the correlation between {n} and the S&P 500?",
     {"requires_any": ["{n}", "{f}", "correlat"], "label": "transmission"},
     {"transmission_query": True}),
    ("what is the price outlook for {cr}?",
     {"requires_any": ["{cr}", "{cf}"], "label": "crypto"}, {}),
    ("should i buy {cr} at these levels?",
     {"requires_any": ["{cr}", "{cf}"], "label": "crypto"}, {}),
    ("where is {fx} heading?",
     {"requires_any": ["{fx}"], "label": "fx"}, {}),
    ("what is the outlook for the {idx}?",
     {"requires_any": ["{idx}", "{f}"], "label": "index"}, {}),
    ("what is the outlook for the 10-year treasury yield?",
     {"requires_any": ["treasury", "yield", "bond"], "label": "bonds"}, {}),
    ("should i own long-duration bonds right now?",
     {"requires_any": ["bond", "duration", "yield"], "label": "bonds"}, {}),
]

# Chained mega prompts where later tasks explicitly build on earlier ones.
# Each: (query_template, [task expectation templates])
_MEGA_CHAINED = [
    ("Analyze the technicals for {t}, then if the trend is constructive give me a "
     "trade setup with entry, stop and target, and finally recommend how to hedge "
     "the position.",
     [{"requires_any": ["{t}", "support", "resistance", "trend", "momentum"],
       "label": "technicals"},
      {"requires_any": ["entry", "stop", "target", "setup"], "label": "setup",
       "setup_query": True},
      {"requires_any": ["hedge", "hedging", "puts", "protection"], "label": "hedge",
       "risk_query": True}]),
    ("Give me the outlook for {t}, then the probability it reaches {p} in {h} months, "
     "then the best options strategy if you expect a rally.",
     [{"requires_any": ["{t}"], "label": "outlook"},
      {"requires_any": ["probability", "odds", "chance", "likely"], "label": "probability"},
      {"requires_any": ["call", "put", "straddle", "spread", "option"], "label": "options"}]),
    ("Compare {t1} and {t2}, then tell me which has better momentum, then give me a "
     "covered call strategy on the winner.",
     [{"requires": ["{t1}", "{t2}"], "label": "comparison"},
      {"requires_any": ["momentum", "trend", "rsi"], "label": "momentum"},
      {"requires_any": ["call", "covered"], "label": "options"}]),
    ("What is the outlook for {n}? Then, how would {g} affect energy stocks? Then give "
     "me a futures setup on {f} with stops, and a hedge for my XOM position.",
     [{"requires_any": ["{n}", "{f}"], "label": "commodity"},
      {"requires_any": ["{g}"], "label": "geopolitics"},
      {"requires_any": ["entry", "stop", "target", "setup"], "label": "setup",
       "setup_query": True},
      {"requires_any": ["hedge", "hedging", "puts"], "label": "hedge",
       "risk_query": True}]),
    ("Explain {m}, then what the latest data means for the Fed path, then whether I "
     "should own TLT now, then how the 10-year yield correlates with equities.",
     [{"requires_any": ["{m}"], "label": "macro"},
      {"requires_any": ["fed", "fomc", "rate", "policy"], "label": "fed_path"},
      {"requires_any": ["tlt", "bond", "treasury"], "label": "bonds"},
      {"requires_any": ["yield", "correlat", "10-year"], "label": "transmission",
       "transmission_query": True}]),
    ("Scan the {s} sector for opportunities, then compare {t1} vs {t2} within it, then "
     "give me an earnings-play options strategy on the winner, then the probability "
     "{t1} doubles in {h} months.",
     [{"requires_any": ["{s}"], "label": "sector"},
      {"requires": ["{t1}", "{t2}"], "label": "comparison"},
      {"requires_any": ["earnings", "call", "put", "option"], "label": "options"},
      {"requires_any": ["probability", "odds", "chance", "double"], "label": "probability"}]),
    ("What is driving {t} right now, what is the probability it pulls back {k}% in the "
     "next {h} months, and how should I hedge my position?",
     [{"requires_any": ["{t}"], "label": "outlook"},
      {"requires_any": ["probability", "odds", "chance", "pullback", "drawdown"],
       "label": "downside"},
      {"requires_any": ["hedge", "hedging", "puts", "protection"], "label": "hedge",
       "risk_query": True}]),
    ("Give me a sector view on {s}, then the probability {t} reaches {p} in {h} months, "
     "then a trade setup for {t} with stops, and finally where {fx} is heading.",
     [{"requires_any": ["{s}"], "label": "sector"},
      {"requires_any": ["probability", "odds", "chance"], "primary": "{t}",
       "label": "probability"},
      {"requires_any": ["entry", "stop", "target", "setup"], "label": "setup",
       "setup_query": True},
      {"requires_any": ["{fx}"], "label": "fx"}]),
]

_MEGA_CONNECTORS = [
    ". ", ". ", " ", " ", ". Also, ", ". Then, ", ". After that, ",
    ". Additionally, ", ". Meanwhile, ", ". And relatedly, ", ". Separately, ",
    ". Now, ", ". Next, ", " Now: ", ". And since we are on the topic, ",
    ". In the same vein, ", ". On a different note, ",
]


def _sub_ctx(s, ctx):
    """Substitute {slot} placeholders from the fill context."""
    s = str(s)
    for k, v in ctx.items():
        s = s.replace("{" + k + "}", str(v))
    return s


def _fill_exp_tmpl(exp, ctx):
    """Substitute placeholders inside an expectation dict."""
    out = {}
    for k, v in exp.items():
        if isinstance(v, list):
            out[k] = [_sub_ctx(x, ctx) for x in v]
        elif isinstance(v, str):
            out[k] = _sub_ctx(v, ctx)
        else:
            out[k] = v
    return out


def _task_primary(exp):
    if exp.get("primary"):
        return exp["primary"]
    for key in ("requires", "requires_any"):
        vals = exp.get(key) or []
        if vals:
            return vals[0]
    return None


def _build_mega(rng, pools):
    """Build one mega prompt (2-5 tasks, chained or unrelated) + expectations."""
    if rng.random() < 0.35:
        qt, tasks_tmpl = rng.choice(_MEGA_CHAINED)
        query, ctx = _fill(qt, rng, pools)
        tasks = [_fill_exp_tmpl(et, ctx) for et in tasks_tmpl]
    else:
        k = rng.choice([2, 2, 3, 3, 3, 4, 4, 5])
        frags = rng.sample(_MEGA_FRAGMENTS, k)
        parts, tasks = [], []
        for qt, et, flags in frags:
            q, ctx = _fill(qt, rng, pools)
            exp = _fill_exp_tmpl(et, ctx)
            exp.update(flags)
            parts.append(q)
            tasks.append(exp)
        query = parts[0]
        for i in range(1, len(parts)):
            query += rng.choice(_MEGA_CONNECTORS) + parts[i]
    first_primary = _task_primary(tasks[0])
    expectations = {
        "mega_tasks": tasks,
        "primary": first_primary,
        "requires_any": tasks[0].get("requires_any") or [first_primary],
        "category_flavor": "mega",
    }
    return query, expectations


_reg("mega_prompt", "multi-task mega prompt", "multi-asset", "mega", [], lambda s: {})


# ---------------------------------------------------------------------------
# Deep-dive institutional thesis prompts (the "huge" bucket)
# ---------------------------------------------------------------------------
# These are 200-600 word institutional mandates of exactly the shape that used
# to regress: they name a company, build a thesis, and mention geopolitical
# risk / earnings / scenarios / options / probability as sections among many —
# the routing must answer the WHOLE ask (deep-dive memo), never collapse into
# a single specialist template (e.g. the "Geopolitical & Event Briefing").

_DEEP_DIVE_TEMPLATES = [
    # ── T1: The canonical NVDA-style institutional mandate ──
    "You are the lead investment strategist at a multi-strategy institutional asset manager. "
    "Conduct a full fundamental, quantitative, macroeconomic, market-structure and "
    "sentiment-driven investment assessment of {t} over the next {hrz} months. Build an "
    "evidence-based investment thesis that separates observable facts, model assumptions, "
    "market-implied expectations and your own inference. Reconstruct the revenue and earnings "
    "drivers by segment; identify the variables that determine forward earnings power; analyze "
    "pricing power, unit growth, gross margins, operating leverage, capex intensity, FCF "
    "conversion and capital allocation. Evaluate the competitive position against {peer} and "
    "emerging alternatives; quantify how much market-share loss the current valuation could "
    "tolerate. Build a DCF with at least three explicit scenarios, perform a reverse DCF to "
    "determine what the current price implies about long-run growth and margins, and compare "
    "EV/Revenue, EV/EBITDA, P/E and FCF yield against appropriate peers. Determine what the "
    "equity market is currently pricing in and identify expectation gaps. Analyze sensitivity "
    "to Fed policy, real rates, inflation, {macro} and geopolitical tensions such as "
    "U.S.-China technology restrictions. Analyze institutional positioning, options "
    "positioning, short interest and narrative momentum. Construct a comprehensive risk "
    "matrix covering probability, severity, time horizon and detectability. Identify "
    "catalysts over 0-3, 3-6, 6-12 and 12-24 months with probabilities and valuation impact. "
    "Construct {sc} scenarios with probabilities that sum to 100% and provide "
    "probability-weighted fair value, 12-month expected return, the MODELED probability "
    "of permanent capital impairment and of a >30% drawdown (each explicitly labeled "
    "'within defined scenarios'; empirical probability is DATA UNAVAILABLE if not "
    "measured) and a final rating. Falsify your "
    "own thesis: identify the strongest arguments against your conclusion and what would "
    "cause you to change your rating. Finally tell me: what would I have to believe for "
    "this investment to be dramatically mispriced?",

    # ── T2: Valuation / consensus / dislocation focus ──
    "Act as the head of equity research at an institutional asset manager. Write a complete "
    "research report on {t}. Cover: (1) the investment thesis and the two strongest "
    "counter-arguments; (2) a segment-level revenue build for the next three years with "
    "growth, margin and capital-intensity assumptions; (3) a full valuation using DCF, "
    "reverse DCF and trading multiples (EV/Revenue, EV/EBITDA, P/E) versus {peer}; (4) what "
    "the market already believes — the expectations embedded in the current price and where "
    "consensus could be wrong; (5) sensitivity to rates, inflation and {macro}; (6) a risk "
    "matrix with probability, severity and detection lead-times; (7) catalysts across the "
    "next four quarters with probabilities; (8) a scenario engine with probabilities that "
    "sum to 100% and probability-weighted fair value; (9) falsification — what data would "
    "prove the thesis wrong; (10) a final rating with a confidence decomposition. Do not "
    "present estimates as facts; label every assumption.",

    # ── T3: Growth / margin / competitive-dynamics focus ──
    "You are a senior portfolio manager evaluating {t} as a new core holding. Build a "
    "thorough investment memo: start with the thesis and the three questions that would "
    "invalidate it; reconstruct the growth algorithm (unit growth, ASPs, pricing power, "
    "market share) and the margin path (gross margin, operating leverage, cost structure); "
    "analyze the competitive dynamics against {peer} — moat durability, share-shift risk, "
    "and the market-share loss the valuation could absorb; value the company with a DCF and "
    "a reverse DCF, and show what the current price implies about long-run growth; compare "
    "multiples to the peer group; stress the thesis under {macro} and a geopolitical "
    "scenario such as new trade restrictions; build a risk matrix, a catalyst calendar, and "
    "at least {sc} scenarios with probabilities; give a probability-weighted fair value, a "
    "12-month expected return, the MODELED probability of a >30% drawdown (labeled "
    "'within defined scenarios'; empirical probability DATA UNAVAILABLE unless measured), "
    "and a final rating. Distinguish hard data from assumptions and inference throughout.",

    # ── T4: Macro-regime / risk-first framing ──
    "You are the CIO of a long-only fund. {t} is a candidate position over a 12-24 month "
    "horizon. Produce a full institutional analysis: (1) thesis and expected-return driver "
    "attribution; (2) fundamental economics — revenue segments, pricing power, margins, "
    "capex, FCF conversion, working capital, capital allocation; (3) competitive position "
    "and moat vs {peer}, with the tolerance of the current valuation to share loss; (4) "
    "valuation — DCF, reverse DCF, EV/EBITDA and FCF yield, and the growth the price "
    "implies; (5) market-implied expectations and the consensus blind spots; (6) macro "
    "regime analysis — sensitivity to {macro}, Fed policy, real rates and geopolitical "
    "tensions (including U.S.-China restrictions where relevant); (7) sentiment and "
    "positioning — institutional flows, short interest, options positioning; (8) risk "
    "matrix with probability, severity and valuation impact; (9) catalysts by quarter with "
    "probabilities; (10) scenario engine with probabilities summing to 100%; (11) "
    "falsification list; (12) final decision — fair value, expected return, drawdown "
    "probabilities, rating and confidence. Never present an assumption as an observed fact.",

    # ── T5: FCF / capital-return / balance-sheet focus ──
    "Build a comprehensive institutional thesis on {t} focused on cash generation and "
    "capital allocation. Answer in full: the investment thesis; the quality of the revenue "
    "base (recurrence, pricing power, segment mix); the FCF algorithm (EBITDA conversion, "
    "capex, working capital, taxes); balance-sheet and leverage considerations; the "
    "competitive position vs {peer} and the moat's durability; valuation via DCF, reverse "
    "DCF and FCF yield, with the growth the price implies; what the market expects versus "
    "what you can defend; sensitivity to rates, inflation and {macro}; a risk matrix; a "
    "catalyst calendar with probabilities; at least {sc} scenarios with transparent "
    "probabilities and probability-weighted fair value; what would falsify the thesis; and "
    "a final rating with confidence. Flag every estimate.",

    # ── T6: Bear / short-thesis framing ──
    "You are a short-biased analyst building a falsification-heavy case file on {t}. "
    "Produce the full institutional analysis: the bear thesis and the bull case it must "
    "defeat; segment-level revenue and margin analysis with the key variables; competitive "
    "pressure points against {peer} and the risk of market-share loss; valuation — DCF, "
    "reverse DCF and multiples, showing what the current price already assumes about "
    "growth and margins; the expectations embedded in price, options positioning and "
    "sentiment; sensitivity to {macro}, rates and geopolitical risk (tariffs, restrictions); "
    "a risk matrix including the short-squeeze and momentum risks; catalysts over the next "
    "12-24 months; {sc} scenarios with probabilities summing to 100% and the probability-"
    "weighted downside; what would invalidate the short; and a final rating with confidence.",

    # ── T7: Bull / long-thesis framing ──
    "You are the lead technology analyst at a growth-focused institutional fund. Build the "
    "complete long thesis on {t}: the investment thesis and its key assumptions; the "
    "segment-by-segment revenue build and the variables that drive earnings power; pricing "
    "power, margins and operating leverage; the moat versus {peer} and emerging "
    "alternatives; a DCF and reverse DCF showing the growth required versus achievable; "
    "peer multiples and growth-adjusted valuation; what the market is pricing and where it "
    "is too conservative; macro sensitivity ({macro}, Fed policy, geopolitical tensions); "
    "sentiment, positioning and crowding risk; a risk matrix; a catalyst calendar with "
    "probabilities; {sc} scenarios with probabilities and probability-weighted fair value; "
    "falsification — the evidence that would break the bull case; and a final rating with a "
    "confidence decomposition.",

    # ── T8: Compact-but-complete memo ──
    "Write a complete investment memo on {t} for the investment committee. Sections: (1) "
    "thesis; (2) fundamental drivers by segment (growth, pricing, margins, capex, FCF); (3) "
    "competitive dynamics vs {peer}; (4) valuation — DCF, reverse DCF, EV/EBITDA, FCF "
    "yield, and the growth the price implies; (5) market expectations and consensus gaps; "
    "(6) macro sensitivity to {macro} and geopolitical risk; (7) sentiment and positioning; "
    "(8) risk matrix with severity and probability; (9) catalysts with probabilities; (10) "
    "{sc} scenarios with probabilities summing to 100%; (11) falsification; (12) final "
    "decision — fair value, 12-month expected return, drawdown probabilities, rating, "
    "confidence. Label all assumptions; never fabricate data.",
]


def _exp_deep_dive(ctx):
    """Expectations for deep-dive theses: every core section must be present and
    no specialist-template leakage (e.g. the generic geopolitical briefing)."""
    return {
        "primary": ctx["t"],
        "requires": ["investment thesis", "valuation", "risk matrix", "catalysts",
                     "scenario", "falsif", "fair value", "rating"],
        "avoid": ["geopolitical & event briefing", "scenario: geopolitical",
                  "scenario: geopolitic"],
        "category_flavor": "deep_dive",
    }


_reg("deep_dive", "institutional deep-dive thesis", "equities", "deep_dive",
     _DEEP_DIVE_TEMPLATES, _exp_deep_dive)


# ---------------------------------------------------------------------------
# Corpus generation
# ---------------------------------------------------------------------------

def _fill(template, rng, pools):
    """Fill a template's {slots} with pool values; returns query + context."""
    ctx = {}

    def _pick(pool_key, exclude=None):
        pool = pools[pool_key]
        for _ in range(20):
            v = rng.choice(pool)
            if exclude is None or v not in exclude:
                return v
        return rng.choice(pool)

    t = template
    if "{t1}" in t or "{t2}" in t:
        ctx["t1"] = _pick("tickers")
        ctx["t2"] = _pick("tickers", exclude={ctx["t1"]})
        t = t.replace("{t1}", ctx["t1"]).replace("{t2}", ctx["t2"])
    if "{t}" in t:
        ctx["t"] = _pick("tickers")
        t = t.replace("{t}", ctx["t"])
    if "{s}" in t:
        ctx["s"] = _pick("sectors")
        t = t.replace("{s}", ctx["s"])
    if "{etf}" in t:
        ctx["etf"] = _pick("sector_etfs")
        t = t.replace("{etf}", ctx["etf"])
    if "{m}" in t:
        ctx["m"] = _pick("macro")
        t = t.replace("{m}", ctx["m"])
    if "{macro}" in t:
        ctx["macro"] = _pick("macro")
        t = t.replace("{macro}", ctx["macro"])
    if "{w}" in t:
        ctx["w"] = rng.choice(["bull", "bear"])
        t = t.replace("{w}", ctx["w"])
    if "{g}" in t:
        ctx["g"] = _pick("geopolitics")
        t = t.replace("{g}", ctx["g"])
    if "{n}" in t or "{f}" in t:
        name, fut = _pick("commodities")
        ctx["n"], ctx["f"] = name, fut
        t = t.replace("{n}", name).replace("{f}", fut)
    if "{cr}" in t or "{cf}" in t:
        name, sym = _pick("crypto")
        ctx["cr"], ctx["cf"] = name, sym
        t = t.replace("{cr}", name).replace("{cf}", sym)
    if "{h}" in t:  # probability horizon — units inferred from the template
        if "month" in t:
            ctx["h"] = rng.choice([1, 3, 6, 9, 12, 18, 24])
        elif "day" in t:
            ctx["h"] = rng.choice([5, 10, 30, 60, 90, 180, 365])
        else:
            ctx["h"] = rng.choice([3, 6, 12, 24, 30, 90, 180, 365])
        t = t.replace("{h}", str(ctx["h"]))
    if "{fx}" in t:
        ctx["fx"] = _pick("fx")
        t = t.replace("{fx}", ctx["fx"])
    if "{idx}" in t:
        name, fut = _pick("indices")
        ctx["idx"], ctx["f"] = name, fut
        t = t.replace("{idx}", name).replace("{f}", fut)
    if "{p}" in t:
        base = _ref_price(ctx.get("t", "SPY"))
        mult = rng.choice([0.5, 0.75, 0.9, 1.1, 1.25, 1.5, 2.0, 3.0])
        price = base * mult
        ctx["p"] = f"${price:,.0f}"
        t = t.replace("{p}", ctx["p"])
    if "{k}" in t:
        ctx["k"] = rng.choice([10, 15, 20, 25, 30, 40, 50])
        t = t.replace("{k}", str(ctx["k"]))  # template already carries the % sign
    if "{n_days}" in t:
        ctx["n_days"] = rng.choice([30, 90, 180, 365])
        t = t.replace("{n_days}", str(ctx["n_days"]))
    if "{n}" in t:  # months horizon (probability)
        ctx["n"] = rng.choice([3, 6, 12, 24])
        t = t.replace("{n}", str(ctx["n"]))
    # Deep-dive slots
    if "{peer}" in t:
        ctx["peer"] = _pick("tickers", exclude={ctx.get("t")})
        t = t.replace("{peer}", ctx["peer"])
    if "{sc}" in t:
        ctx["sc"] = rng.choice([5, 5, 5, 7])
        t = t.replace("{sc}", str(ctx["sc"]))
    if "{hrz}" in t:
        ctx["hrz"] = rng.choice(["12-24", "12\u201324", "12 to 24", "6-18", "24"])
        t = t.replace("{hrz}", ctx["hrz"])
    if "{sec}" in t:
        ctx["sec"] = rng.choice(["data center", "cloud software", "semiconductors",
                                 "consumer platforms", "energy transition",
                                 "financial technology", "healthcare technology",
                                 "industrial automation"])
        t = t.replace("{sec}", ctx["sec"])
    return t, ctx


def generate_corpus(seed: int = 42, max_per_category: int = 400,
                    include_categories=None):
    """Return a deterministic list of prompt dicts."""
    rng = random.Random(seed)

    pools = {
        "tickers": TICKERS,
        "sectors": SECTORS,
        "sector_etfs": ["XLK", "XLF", "XLV", "XLY", "XLP", "XLU", "XLI", "XLB",
                        "XLE", "XLRE", "XLC", "SMH", "IBB", "KRE", "XHB", "IYR"],
        "macro": MACRO_TOPICS,
        "geopolitics": GEOPOLITICS,
        "commodities": COMMODITIES,
        "fx": FX_PAIRS,
        "indices": INDICES,
        "futures": FUTURES_SYMBOLS,
        "crypto": CRYPTO,
    }

    corpus = []
    for (category, sub, asset_class, intent, templates, exp_builder) in _TEMPLATES:
        if include_categories and category not in include_categories:
            continue
        if category == "mega_prompt":
            # Mega prompts are built combinatorially, not from a template list.
            for _ in range(max_per_category):
                query, expectations = _build_mega(rng, pools)
                corpus.append({
                    "query": query,
                    "category": category,
                    "subcategory": "multi-task mega prompt",
                    "asset_class": "multi-asset",
                    "expected_intent": "mega",
                    "expectations": expectations,
                })
            continue
        # Dedupe by full template list order; shuffle for variety
        shuffled = templates[:]
        rng.shuffle(shuffled)
        count = 0
        for template in shuffled:
            # Generate multiple fills per template up to the category cap
            for _ in range(max_per_category // len(shuffled) + 1):
                query, ctx = _fill(template, rng, pools)
                expectations = exp_builder(ctx)
                corpus.append({
                    "query": query,
                    "category": category,
                    "subcategory": sub,
                    "asset_class": asset_class,
                    "expected_intent": intent,
                    "expectations": expectations,
                })
                count += 1
                if count >= max_per_category:
                    break
            if count >= max_per_category:
                break
    # Deterministic shuffle + dedupe preserving order
    rng.shuffle(corpus)
    seen = set()
    unique = []
    for p in corpus:
        if p["query"] not in seen:
            seen.add(p["query"])
            unique.append(p)
    return unique


# ---------------------------------------------------------------------------
# Medium-length prompts (15-90 words) for the "medium" bucket
# ---------------------------------------------------------------------------

_MEDIUM_TEMPLATES = [
    "Analyze the outlook for {t} over the next 12 months, including the key risks, earnings catalysts and a fair value estimate.",
    "How would {macro} and rising rates affect {t} over the next 6-12 months, and what does the valuation look like?",
    "Build a trade plan for {t}: entry zone, stop loss, take profit targets and risk/reward, given the recent momentum.",
    "What is the probability {t} reaches {p} within {h} months, and what are the main upside and downside risks?",
    "Compare {t1} and {t2} on growth, margins, valuation and balance sheet strength, and tell me which is the better long-term holding.",
    "Hedge my {t} position against a market correction: what put structures or collars make sense, and what do they cost in a low-vol regime?",
    "What does the latest inflation data and Fed policy path imply for growth stocks like {t} over the next quarter?",
    "Scan the {s} sector for the best opportunities right now and highlight the names with the strongest momentum and reasonable valuations.",
    "If {g}, how does that transmit to {t} and the broader market, and which sectors would outperform and underperform?",
    "How does the price of {n} affect airlines, industrials and {t}, and what is the historical correlation in different rate regimes?",
    "Is {t} overvalued or undervalued at current levels, and what multiple should it trade at versus {peer}?",
    "What is the outlook for {t} earnings this quarter, and how would a beat or a miss with weak guidance move the stock?",
    "Build a diversified portfolio with {t}, {t1} and {t2}: what weights, what risk controls, and what is the expected drawdown profile?",
    "What happens to {t} and gold if the Fed cuts rates aggressively next year while inflation stays sticky?",
    "Should I buy the pullback in {t} or wait for a lower entry, and what technical levels confirm the setup?",
    "Analyze {t}'s competitive moat versus {peer}: pricing power, switching costs, network effects, and what could erode the advantage.",
    "What is the probability of a recession in the next 12 months, and how should I position {t} defensively?",
    "How do rising yields transmit to technology and growth equities, and does duration explain most of the multiple compression in that group?",
    "What are the key catalysts for {t} over the next 6 months, and which news events would cause the biggest repricing?",
    "Assess the valuation of {t} using DCF logic: what growth and margin assumptions does the current price already embed?",
]


# Uppercase tokens that are macro/economic terms, not securities — never pick
# these as the medium-prompt primary entity ("CPI", "FOMC", currency codes...).
_MACRO_PRIMARY_STOP = frozenset({
    "CPI", "PCE", "PMI", "ISM", "NFP", "FOMC", "GDP", "ECB", "BOJ", "OPEC",
    "NATO", "FED", "VIX", "DXY", "SPX", "Q1", "Q2", "Q3", "Q4", "H1", "H2",
    "FY", "YTD", "EPS", "PE", "ROE", "ROA", "ROI", "DCF", "EBITDA", "EBIT",
    "FCF", "WACC", "CAGR", "RSI", "MACD", "CAPEX", "NOPAT", "ROIC", "IV", "PE",
    "USD", "EUR", "JPY", "GBP", "CHF", "AUD", "CAD", "NZD", "CNY", "INR",
    "MXN", "BRL", "KRW", "HKD", "SGD", "SEK", "NOK", "DKK", "PLN", "CZK",
    "HUF", "TRY", "ZAR", "THB", "PHP", "MYR", "IDR", "TWD", "VND", "ILS",
    "AED", "SAR", "RUB", "CLP", "ARS", "PKR", "ETC", "USDCNY", "USDJPY",
    # Common English words that can appear uppercase at sentence start / as
    # pronouns — never the primary entity ("What", "I", "US", "The"...)
    "WHAT", "HOW", "WHY", "WHEN", "WHERE", "WHICH", "WHO", "WHOM", "THE",
    "AND", "OR", "BUT", "IF", "I", "A", "AN", "US", "UK", "EU", "IS",
    "ARE", "DO", "DOES", "SHOULD", "WOULD", "WILL", "CAN", "COULD", "MY",
    "YOUR", "HE", "SHE", "IT", "THIS", "THAT", "THESE", "THOSE", "NOT",
    "NO", "YES", "SO", "AS", "IN", "ON", "AT", "BY", "TO", "OF", "FOR",
    "WITH", "FROM", "ME", "WE", "OUR", "THEY", "THEM", "HIM", "HER",
    "BUY", "SELL", "BEST", "TOP", "NEW", "MOST", "NEAR", "NEXT", "LAST",
})


def _exp_medium(query: str):
    """Expectations for a medium prompt, derived from the filled query text so
    setup / hedging / probability expectations line up with rubric criteria.
    The primary entity is the first SECURITY-like uppercase token (never a
    macro term like CPI or a currency code)."""
    import re as _re
    ql = query.lower()
    tks = [t for t in _re.findall(r"\b[A-Z]{1,6}\b", query)
           if t not in _MACRO_PRIMARY_STOP and not t.isdigit()]
    primary = tks[0] if tks else None
    exp = {"primary": primary, "category_flavor": "medium"}
    if primary:
        exp["requires_any"] = [primary, primary.lower()]
    # Transmission-flavored prompts are answered by the quantitative
    # transmission engine, which re-anchors on the measured relationships
    # (correlation / beta / lead-lag) — satisfying the ask with transmission
    # evidence is a pass even if the specific ticker is not named.
    if any(w in ql for w in ("transmit", "transmission", "correlation",
                             "affect", "impact", "relationship between",
                             "historical correlation")):
        exp["requires_any"] = [primary, "correlation", "transmission",
                                "relationship", "market"] if primary \
            else ["correlation", "transmission", "market"]
    if any(w in ql for w in ("entry", "stop loss", "take profit", "risk/reward",
                             "trade plan", "buy the pullback")):
        exp["setup_query"] = True
    if any(w in ql for w in ("hedge", "hedging", "collar", "protect", "buy puts",
                             "defensive", "position")):
        exp["risk_query"] = True
    if any(w in ql for w in ("probability", "chance", "odds", "likely")):
        exp["category_flavor"] = "probability"
    return exp


def generate_sized_corpus(bucket: str, n: int = 10000, seed: int = 42):
    """Deterministic corpus of exactly `n` prompts for one size bucket.

    bucket:
      small  — short questions (3-14 words)
      medium — 1-2 sentence questions (15-90 words)
      huge   — institutional deep-dive theses (200-600 words)
      mega   — multi-task chained prompts (2-5 tasks)

    All prompts carry rubric expectations and are deduplicated, so a bucket
    run produces `n` distinct queries stored in the eval database.
    """
    rng = random.Random(seed)
    pools = {
        "tickers": TICKERS,
        "sectors": SECTORS,
        "sector_etfs": ["XLK", "XLF", "XLV", "XLY", "XLP", "XLU", "XLI", "XLB",
                        "XLE", "XLRE", "XLC", "SMH", "IBB", "KRE", "XHB", "IYR"],
        "macro": MACRO_TOPICS,
        "geopolitics": GEOPOLITICS,
        "commodities": COMMODITIES,
        "fx": FX_PAIRS,
        "indices": INDICES,
        "futures": FUTURES_SYMBOLS,
        "crypto": CRYPTO,
    }
    out, seen = [], set()
    target = max(1, int(n))
    attempts = 0
    max_attempts = target * 250

    def _emit(category, subcategory, asset_class, intent, query, exp):
        nonlocal attempts
        if query in seen or exp is None:
            return False
        seen.add(query)
        out.append({
            "query": query, "category": category, "subcategory": subcategory,
            "asset_class": asset_class, "expected_intent": intent,
            "expectations": exp,
        })
        return True

    if bucket == "huge":
        while len(out) < target and attempts < max_attempts:
            attempts += 1
            q, ctx = _fill(rng.choice(_DEEP_DIVE_TEMPLATES), rng, pools)
            _emit("deep_dive", "institutional deep-dive thesis", "equities",
                  "deep_dive", q, _exp_deep_dive(ctx))
    elif bucket == "mega":
        while len(out) < target and attempts < max_attempts:
            attempts += 1
            q, exp = _build_mega(rng, pools)
            _emit("mega_prompt", "multi-task mega prompt", "multi-asset",
                  "mega", q, exp)
    elif bucket == "medium":
        while len(out) < target and attempts < max_attempts:
            attempts += 1
            q, ctx = _fill(rng.choice(_MEDIUM_TEMPLATES), rng, pools)
            exp = _exp_medium(q)
            if _emit("medium_prompt", "medium multi-clause question",
                     "multi-asset", "medium", q, exp):
                continue
    else:
        # small: derive from every non-mega template, filtered by word count
        # (3-14 words), so variety is maximal without new template bloat.
        sources = [(c, sub, ac, it, tmpl, eb)
                   for (c, sub, ac, it, tmpl, eb) in _TEMPLATES if c != "mega_prompt"]
        lo, hi = (3, 14)
        while len(out) < target and attempts < max_attempts:
            attempts += 1
            cat, sub, ac, it, tmpl, eb = rng.choice(sources)
            q, ctx = _fill(rng.choice(tmpl), rng, pools)
            wc = len(q.split())
            if not (lo <= wc <= hi):
                continue
            try:
                exp = eb(ctx)
            except Exception:
                exp = {"primary": ctx.get("t"),
                       "requires_any": [ctx.get("t")] if ctx.get("t") else None}
            _emit(cat, sub, ac, it, q, exp)
    return out


def categories():
    return sorted({c for (c, *_rest) in _TEMPLATES})
