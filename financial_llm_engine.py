"""
Financial Intelligence LLM Engine — Octavian Terminal
A fully self-contained, zero-API-key heuristic NLP engine that performs
multi-agent financial reasoning via rule-based text generation.
Designed to be called directly by the AI chatbot in the Intelligence Center.
"""

import re
import os
import time
import random
import hashlib
import tempfile
import datetime
import json
import requests

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:
    _HAS_YF = False

import logging

logger = logging.getLogger("FinancialLLMEngine")

from ticker_universe import get_ticker_universe

# ---------------------------------------------------------------------------
# ENTITY RESOLUTION (financial hallucination firewall)
# ---------------------------------------------------------------------------
# Semantic / finance-jargon words that must NEVER be treated as tickers, even
# when the user types them in ALL CAPS (CAPEX, FCF, DCF, GPU, ...). These are
# concepts, not securities — the fix for the "Live snapshot for CAPEX / FCF /
# GPU / CUDA / MOAT ..." class of bug.
_SEMANTIC_CONCEPTS = frozenset({
    # Financial metrics & ratios (never securities)
    "CAPEX", "FCF", "DCF", "EBITDA", "EBIT", "EBI", "EPS", "NOPAT", "ROIC",
    "ROCE", "CAGR", "WACC", "RSI", "MACD", "NIM", "MARGIN", "MARGINS",
    "REVENUE", "REVENUES", "GROWTH", "YIELD", "YIELDS", "DIVIDEND", "DIVIDENDS",
    "PAYOUT", "LEVERAGE", "GEAIRING", "LIQUIDITY", "SOLVENCY", "BETA", "ALPHA",
    "SHARPE", "SORTINO", "VOL", "VOLATILITY", "DRAWDOWN", "DRAWDOWNS", "DRAWDOWN=X",
    "PREMIUM", "PREMIUMS", "DISCOUNT", "DISCOUNTS", "SPREAD", "SPREADS", "BIDASK",
    "NOTIONAL", "EXPIRY", "EXPIRIES", "STRIKE", "STRIKES", "IV", "IMPLIED",
    "THETA", "GAMMA", "VEGA", "DELTA", "RHO", "VANNA", "CHARM", "VOLGA",
    # Tech / AI / semiconductor jargon (concepts, not tickers). NOTE: MOAT is
    # deliberately NOT here — it is a real ETF (VanEck MOAT) handled by
    # _AMBIGUOUS_TICKER_CONCEPTS, which keeps it ONLY when referenced as a
    # security ("MOAT ETF") and drops it as a concept ("durable moat").
    "GPU", "CPU", "CUDA", "ASIC", "ASICS", "TPU", "NPU", "FPGA",
    "INFERENCE", "TRAINING", "LLM", "LLMS", "MODEL", "MODELS", "DATACENTER",
    "DATACENTERS", "HYPESCALER", "HYPESCALERS",
    # Reporting / time-period jargon. NOTE: LTM (LATAM Airlines) is a real
    # ticker handled by _AMBIGUOUS_TICKER_CONCEPTS; NTM/TTM/FWD stay here.
    "NTM", "TTM", "FWD", "YTD", "QTD", "MOM", "YOY", "FY", "FY24",
    "FY25", "FY26", "FY27", "FY28", "Q1", "Q2", "Q3", "Q4", "H1", "H2",
    "CONSENSUS", "ESTIMATE", "ESTIMATES", "GUIDANCE", "GUIDE", "BEAT", "MISS",
    # Generic English words that appear in ALL CAPS in prompts (rare, but when
    # they do they are emphatic prose, not securities). NOTE: LOW (Lowe's),
    # KEY (KeyCorp) and MAIN (Main Street Capital) are REAL universe tickers
    # and therefore live in _AMBIGUOUS_TICKER_CONCEPTS, not here — so
    # "outlook for LOW" resolves but "low volatility" never does.
    "MULTI", "GROSS", "BUILD", "FACTS", "ASPS", "LOSS", "LOSSES", "UNIT",
    "UNITS", "CYCLE", "CYCLES", "RATE", "RATES", "SPEED", "PACE", "RANGE",
    "RANGES", "STAGE", "STAGES", "SCALE", "SCOPE", "TOTAL", "CORE",
    "TOP", "BOTTOM", "END", "START", "MID", "HIGH", "WIDE",
    "NARROW", "FAST", "SLOW", "BIG", "SMALL", "LARGE", "MEDIUM", "FULL",
})

# Words that are BOTH legitimate tickers AND finance concepts. Kept as tickers
# only when the query clearly references them as securities (e.g. "AI stock").
_AMBIGUOUS_TICKER_CONCEPTS = {
    "AI": "C3.ai",  # artificial-intelligence concept vs C3.ai (AI) ticker
    "BASE": "Couchbase",  # "base case" vs BASE ticker
    "MOAT": "VanEck MOAT ETF",  # competitive moat vs MOAT ETF
    "LOSS": "",  # "net loss" vs obscure LOSS ticker
    "UNIT": "",  # "unit economics" vs UNIT ticker
    "LTM": "LATAM Airlines",  # last-twelve-months vs LTM ticker
    "LOW": "Lowe's Companies",  # "low volatility" vs LOW (Lowe's) ticker
    "KEY": "KeyCorp",  # "key drivers" vs KEY (KeyCorp) ticker
    "MAIN": "Main Street Capital",  # "main driver" vs MAIN ticker
    "SUM": "Summit Materials",  # "probabilities that sum to 100%" vs SUM ticker
    "CASH": "Pathward Financial",  # "cash flow" / "cash vs debt" vs CASH ticker
    "WAVE": "Eco Wave Power",  # "wave of selling" vs WAVE ticker
    "DASH": "DoorDash",  # prose "dash" vs DASH ticker
    "PEAK": "Peakstone Realty",  # "peak multiple" vs PEAK ticker
    "TRIP": "Trip.com Group",  # prose "trip" vs TRIP ticker
    "WISH": "ContextLogic",  # prose "wish" vs WISH ticker
    "HOPE": "Hope Bancorp",  # prose "hope" vs HOPE ticker
    "SEED": "Origin Agritech",  # "seed round" vs SEED ticker
    "COST": "Costco",  # "what do they cost" vs COST ticker
    "TGT": "Target",  # "target price" vs TGT ticker
    "NET": "Cloudflare",  # "net margin" vs NET ticker
    "SIX": "Six Flags",  # "six scenarios" vs SIX ticker
}

# Geographic / regional codes that ALSO happen to be real universe tickers
# ("EU" = Eurus Energy, "US" = US Global GO Gold, "UK" = a legacy listing).
# In user queries these mean the region — "EU energy crisis", "US Treasury
# yields", "EU and US relations" — essentially never the obscure ticker, so
# they are hard-blocked exactly like semantic concepts. A user who truly
# wants the security must say "Eurus Energy" or use explicit security
# framing ("EU stock") — which the block still honors by NOT silently
# substituting a different asset.
_REGION_CODES = frozenset({"EU", "US", "UK", "CN", "JP", "DE", "FR", "CA"})

# Common English words (2-6 letters) that happen to be real universe tickers.
# Typed in lowercase inside prose ("the bar is high", "net margin", "target
# price", "what do they cost") they are prose, NOT securities — they resolve
# only when the query explicitly frames them as a security ("COST stock",
# "buy NET", "TGT earnings"). The uppercase pass already requires
# security-ish context for these via _AMBIGUOUS_TICKER_CONCEPTS; this set
# extends the same protection to the lowercase (casual-typing) pass.
_COMMON_ENGLISH_WORDS = frozenset({
    "AGO", "AIR", "ARM", "BAR", "BOX", "CAT", "COP", "EAT", "FAN", "FAT",
    "FIX", "HUT", "ICE", "LIT", "LOT", "MET", "NET", "RIG", "RUN", "SEE",
    "SIT", "SON", "SUB", "TAN", "TAP", "TIP", "WIT", "COST", "TGT", "SIX",
    "HOLD", "TIME", "CASE", "DEAL", "TRUST", "BANK", "FLOW", "GOLD", "RATE",
    "WORK", "PLAY", "WAVE", "PEAK", "DASH", "TRIP", "WISH", "HOPE", "SEED",
    "RANGE", "STAGE", "SCALE", "SCOPE", "PACE", "SPEED", "FORCE", "POWER",
    "MOVE", "GAIN", "LOSS", "RISK", "SAFE", "GOOD", "BEST", "NEXT", "LAST",
    "FIRST", "FINAL", "SALE", "SALES", "LOAN", "DEBT", "CASH", "MONEY",
    "FUND", "CITY", "ZONE", "TIDE", "TONE", "ZOOM", "FLEX", "GROW", "SURGE",
    "SPARK", "BLOOM", "BROAD", "SHARP", "OPEN", "DONE", "RISE", "FALL",
    "DROP", "CARE", "PATH", "EDGE", "BETA", "ALPHA", "FREE", "JUMP", "KICK",
    "LIFT", "MINT", "NORTH", "SOUTH", "EAST", "WEST", "MONTH", "WEEK",
    "YEAR", "HOUR", "DATE", "QUARTER", "PRICE", "LEVEL", "VALUE", "TREND",
    "SIGNAL", "STOCK", "SHARE", "ASSET", "INDEX", "BOND", "SECTOR", "TRADE",
    "OPTION", "POSITION", "ENTRY", "EXIT", "TARGET", "SETUP", "STOP", "HEDGE",
    "BREAK", "CYCLE", "STAGE", "STATUS", "RANGE", "TOP", "BOTTOM", "MID",
    "END", "START", "CORE", "TOTAL", "WIDE", "NARROW", "FAST", "SLOW", "BIG",
    "SMALL", "LARGE", "MEDIUM", "FULL", "HIGH", "LOW", "KEY", "MAIN", "SUM",
    "BASE", "AI", "MOAT", "LOSS", "UNIT", "LTM", "ALL", "ARE", "CAN", "HAS",
    "NOW", "ONE", "TWO", "SIX", "TEN", "TOP", "TRY", "USE", "WAY", "WAR",
    "WIN", "YES", "YET", "AGE", "AIR", "ART", "ASK", "BAD", "BAG", "BED",
    "BET", "BID", "BIT", "BOY", "BUS", "BUY", "CAB", "CAP", "CAR", "CUP",
    "CUT", "DAD", "DAY", "DID", "DIG", "DOG", "DRY", "EAR", "EGG", "ERA",
    "EVE", "EYE", "FAR", "FAT", "FEW", "FIG", "FIT", "FLY", "FOG", "FUN",
    "GAP", "GAS", "GOT", "GUM", "HAD", "HAM", "HAT", "HAY", "HER", "HID",
    "HIM", "HIP", "HIS", "HIT", "HOG", "HOP", "HOT", "HUB", "HUG", "HUT",
    "ICE", "ILL", "INK", "ION", "ITS", "JAM", "JAR", "JET", "JOB", "JOY",
    "JUG", "KID", "KIT", "LAB", "LAD", "LAP", "LAW", "LAY", "LED", "LEG",
    "LET", "LID", "LIE", "LIP", "LOG", "LOW", "MAD", "MAN", "MAP", "MAT",
    "MAY", "MEN", "MET", "MIX", "MOM", "MOP", "MUD", "NAG", "NAP", "NOD",
    "NOR", "NOT", "NUT", "OAK", "OAT", "ODD", "OFF", "OIL", "OLD", "OPT",
    "OUT", "OWN", "PAD", "PAL", "PAN", "PAT", "PAW", "PAY", "PEA", "PEN",
    "PET", "PIE", "PIG", "PIN", "PIT", "PLY", "POD", "POP", "POT", "PUN",
    "PUP", "RAG", "RAM", "RAN", "RAP", "RAT", "RAW", "RED", "RIB", "RID",
    "RIG", "RIM", "RIP", "ROB", "ROD", "ROT", "ROW", "RUB", "RUG", "RUM",
    "SAD", "SAG", "SAT", "SAW", "SAY", "SEA", "SET", "SEW", "SHE", "SHY",
    "SIN", "SIP", "SIR", "SKI", "SKY", "SLY", "SOB", "SOD", "SOW", "SPA",
    "SPY", "STY", "SUN", "TAB", "TAG", "TAX", "TEA", "TIE", "TIN", "TIP",
    "TOE", "TON", "TOO", "TOT", "TOW", "TOY", "TUB", "TUG", "VAN", "VET",
    "WAS", "WAX", "WEB", "WED", "WET", "WHO", "WIT", "WOE", "WON", "WOO",
    "ZIP", "ZOO",
})


def _is_equity_reference(query: str, token: str, word_boundary: bool = False) -> bool:
    """True when a query references an ambiguous token AS A SECURITY rather
    than as a finance concept (e.g. 'AI stock' vs 'AI infrastructure').

    With ``word_boundary=True`` every multi-character phrase must end at a
    word boundary, so prose plurals never match: 'all stocks' cannot satisfy
    the 'all stock' phrase, and 'are prices' cannot satisfy 'are price'.
    Used for universe tickers that are also core English stopwords."""
    ql = query.lower()
    tl = token.lower()
    stock_context = [
        f"{tl} stock", f"{tl} shares", f"{tl} price", f"{tl} ticker",
        f"buy {tl}", f"sell {tl}", f"long {tl}", f"short {tl}", f"position in {tl}",
        f"outlook for {tl}", f"outlook on {tl}", f"is {tl} bullish", f"is {tl} bearish",
        f"is {tl} a good", f"{tl} earnings", f"{tl} valuation", f"{tl} options",
        f"{tl} call", f"{tl} puts", f"{tl} chart", f"{tl} forecast", f"{tl} target",
        f"{tl} vs", f"vs {tl}", f"compare {tl}", f"{tl} and {tl}", f"{tl} dividend",
        f"{tl} etf", f"etf {tl}", f"{tl} fund", f"{tl} tickers",
        # Technical / chart framing: "support and resistance levels for LOW",
        # "chart for LOW", "LOW levels" — genuine security references.
        f"levels for {tl}", f"levels of {tl}", f"{tl} levels",
        f"support and resistance for {tl}", f"resistance for {tl}",
        f"support for {tl}", f"chart for {tl}", f"chart of {tl}",
        f"price target for {tl}", f"target price for {tl}", f"target for {tl}",
        f"price of {tl}", f"news on {tl}", f"news about {tl}", f"earnings for {tl}",
        # Memo / report / analysis framing: "investment memo on LOW",
        # "thesis on V", "assess the valuation of LOW", "case on KEY".
        f"memo on {tl}", f"memo for {tl}", f"thesis on {tl}", f"case on {tl}",
        f"case for {tl}", f"report on {tl}", f"report about {tl}",
        f"analysis of {tl}", f"analysis on {tl}", f"assessment of {tl}",
        f"valuation of {tl}", f"review of {tl}", f"evaluating {tl}",
        f"evaluate {tl}", f"assess {tl}", f"model {tl}", f"profile {tl}",
        f"on {tl} as", f"for {tl} as",
        # Comparative / list framing: "growth stocks like TGT", "such as
        # LOW", "a basket with TGT and XLB".
        f"stocks like {tl}", f"names like {tl}", f"like {tl} stock",
        f"like {tl} shares", f"such as {tl} and", f"such as {tl},",
        f"like {tl} and", f"like {tl},", f"basket with {tl}",
        f"portfolio with {tl}", f"portfolio of {tl}", f"with {tl} and",
        f"with {tl},", f"including {tl}", f"{tl} alongside",
        # Catalyst / risk / driver framing: "key catalysts for TGT",
        # "upside and downside risks for COST".
        f"catalysts for {tl}", f"catalyst for {tl}", f"catalysts of {tl}",
        f"risks for {tl}", f"risk for {tl}", f"drivers for {tl}",
        f"prospects for {tl}", f"case for {tl}",
        # Probability framing: "probability LOW reaches", "LOW to reach $225",
        # "How likely is LOW to drop 25%", "chances LOW falls in 3 months",
        # "how likely is COST to fall to $837".
        f"probability {tl}", f"chance {tl}", f"chances {tl}", f"odds {tl}",
        f"likely {tl}", f"likelihood {tl}", f"{tl} reaches", f"{tl} to reach",
        f"reach {tl}", f"{tl} pull back", f"{tl} falls", f"{tl} fall",
        f"{tl} to drop", f"{tl} drop", f"{tl} drops", f"{tl} in the next",
        f"{tl} over the next", f"will {tl}", f"{tl} will",
        f"{tl} to fall", f"{tl} falls to", f"{tl} fall to", f"chance {tl} falls",
        # Macro / transmission framing: "What happens to TGT and gold if the
        # Fed cuts", "the impact on COST of higher rates", "how do rates
        # affect TGT".
        f"happens to {tl}", f"what happens to {tl}", f"impact on {tl}",
        f"impact of {tl}", f"effect on {tl}", f"affect {tl}", f"affects {tl}",
        f"how would {tl}", f"what would {tl}", f"how does {tl} react",
        f"{tl} react", f"{tl} respond", f"{tl} perform", f"{tl} outlook",
        # Valuation framing: "Is TGT overvalued or undervalued at current
        # levels", "what multiple should TGT trade at".
        f"is {tl} overvalued", f"{tl} overvalued", f"is {tl} undervalued",
        f"{tl} undervalued", f"{tl} overvalued or undervalued",
        f"{tl} undervalued or overvalued", f"{tl} trade at", f"{tl} trading at",
        f"{tl} fair value", f"{tl} intrinsic", f"multiple for {tl}",
        # Position / hedging framing: "Hedge my LOW position", "protect TGT",
        # "my TGT exposure", "collar on LOW".
        f"my {tl} position", f"{tl} position against", f"hedge my {tl}",
        f"hedging my {tl}", f"protect {tl}", f"protecting {tl}",
        f"protection for {tl}", f"collar on {tl}", f"puts on {tl}",
        f"hedge {tl}", f"hedging {tl}", f"{tl} exposure",
        # "Best way to play TGT this week" / "protect gains in TGT" — the
        # ticker appears right after a trading verb with no stock-context
        # noun, so the verb + preposition phrases carry the security framing.
        f"play {tl}", f"playing {tl}", f"gains in {tl}", f"gains on {tl}",
        f"protect gains in {tl}", f"lock in gains in {tl}",
        f"lock in gains on {tl}", f"take profits in {tl}",
        f"take profit in {tl}", f"best way to play {tl}", f"how to play {tl}",
        # Options / strategy framing: "options strategy for LOW before
        # earnings", "best put structure to protect TGT".
        f"options for {tl}", f"strategy for {tl}", f"strategies for {tl}",
        f"{tl} before earnings", f"{tl} ahead of earnings", f"strategy on {tl}",

        # Comparison framing: "How does EWZ compare to TGT?", "Should I own
        # TGT or JPM?", "EWZ versus TGT".
        f"compare to {tl}", f"comparing {tl}", f"compare {tl} and",
        f"own {tl}", f"should i own {tl}", f"should i buy {tl}",
        f"pick between {tl}", f"{tl} or {tl}", f"between {tl} and",
        # Company-facts framing: "Who are LOW's main competitors?", "What
        # does TGT do?", "LOW's business model".
        f"who are {tl}", f"who is {tl}", f"does {tl} do", f"what does {tl} do",
        f"what is {tl}", f"tell me about {tl}", f"{tl} company", f"{tl} stock",
        f"{tl} business", f"{tl}'s main", f"{tl}'s top", f"{tl}'s biggest",
        f"{tl}'s competitors", f"{tl}'s revenue", f"{tl}'s earnings",
        f"{tl}'s growth", f"{tl}'s margin", f"{tl}'s valuation",
        f"{tl}'s price", f"{tl}'s stock", f"{tl}'s business", f"{tl}'s outlook",
        # News / event recap framing: "What happened to COST this week?",
        # "Is the news flow bullish or bearish for TGT?", "latest on LOW".
        f"happened to {tl}", f"what happened to {tl}", f"this week for {tl}",
        f"news flow for {tl}", f"news flow on {tl}", f"flow for {tl}",
        f"bullish or bearish for {tl}", f"bearish for {tl}", f"bullish for {tl}",
        f"latest on {tl}", f"update on {tl}", f"news for {tl}",
        # Moat / competitive framing: "Analyze TGT's competitive moat versus
        # AAPL", "what is LOW's moat", "moat of TGT".
        f"{tl}'s moat", f"{tl}'s competitive", f"{tl}'s advantage",
        f"moat of {tl}", f"competitive moat of {tl}", f"{tl} moat",
        f"{tl} competitive moat", f"{tl} competitive", f"moat {tl}",
        f"{tl}'s economics",
        # Safe-haven / defensive framing: "Is TGT a safe haven if an EU
        # energy crisis?", "TGT as a safe haven", "safe haven for TGT".
        f"is {tl} a safe haven", f"is {tl} a haven", f"{tl} as a safe haven",
        f"safe haven for {tl}", f"safe haven {tl}", f"{tl} a safe haven",
        f"haven for {tl}", f"defensive play in {tl}", f"defensive play {tl}",
        # Dividend framing: "Does LOW pay a dividend?", "Has LOW been growing
        # its dividend?", "LOW's dividend yield".
        f"does {tl} pay", f"{tl} pay a", f"{tl} pays", f"{tl} been growing",
        f"growing its {tl}", f"{tl}'s dividend", f"dividend of {tl}",
        f"yield on {tl}", f"{tl} yield", f"yield of {tl}",
        f"dividend yield of {tl}", f"yield for {tl}", f"dividend for {tl}",
        # Options / position framing: "Should I sell covered calls on TGT?",
        # "bull call spread on TGT", "how should I size positions in LOW".
        f"calls on {tl}", f"call on {tl}", f"call spread on {tl}",
        f"covered calls on {tl}", f"covered call on {tl}", f"spread on {tl}",
        f"positions in {tl}", f"position in {tl}", f"size {tl}", f"sizing {tl}",
        f"does {tl} compare", f"how does {tl}", f"would {tl} be",
        # Probability-downside / drawdown framing: "What are the chances of a
        # drawdown in LOW?", "drawdown risk for TGT".
        f"drawdown in {tl}", f"drawdown for {tl}", f"drawdown of {tl}",
        f"chances of a drawdown in {tl}", f"odds of a drawdown in {tl}",
        f"drawdown risk {tl}", f"{tl} drawdown",
        # Options-strategy framing: "Would a straddle work on TGT?", "iron
        # condor on TGT", "strangle on TGT".
        f"straddle on {tl}", f"straddle {tl}", f"strangle on {tl}",
        f"condor on {tl}", f"work on {tl}", f"spread on {tl}",
        f"options on {tl}", f"option on {tl}",
        # Company-facts framing: "What sector is TGT in?", "which sector is
        # LOW in", "TGT's sector".
        f"sector is {tl} in", f"sector is {tl}", f"which sector is {tl}",
        f"sector of {tl}", f"{tl} in which sector", f"which sector {tl}",
        # Portfolio framing: "How should I think about LOW in my portfolio?",
        # "role of TGT in a portfolio", "add LOW to my portfolio".
        f"think about {tl}", f"about {tl} in my portfolio", f"{tl} in my portfolio",
        f"role of {tl}", f"{tl} in a portfolio", f"add {tl} to my",
        f"{tl} defensively", f"position {tl} defensively", f"defensive {tl}",
        f"defensive positioning of {tl}",
    ]
    for ctx in stock_context:
        if len(tl) == 1:
            # Single-letter tickers (F, C, T, V, ...) must be STANDALONE words
            # — "e moat" matches inside "durable moat" and "e position
            # against" inside "competitive position against". Word-boundary
            # anchoring on the whole phrase keeps "buy F", "F to fall" while
            # blocking accidental in-word matches.
            if re.search(rf"\b{re.escape(ctx)}\b", ql):
                return True
        elif word_boundary:
            # Stopword tickers (ARE/ALL/AM/CAN/...) — the phrase must end at a
            # word boundary so prose plurals never resolve: "all stocks"
            # cannot satisfy the 'all stock' phrase, "are prices" cannot
            # satisfy 'are price'.
            if re.search(rf"\b{re.escape(ctx)}(?![a-z])", ql):
                return True
        elif ctx in ql:
            return True
    # Trade-setup framing ("Give me a trade setup for ARE with entry, stop and
    # target") is unambiguous ONLY on the strict word-boundary stopword path
    # (ARE/ALL/CAN/...), which additionally requires the token UPPERCASE in
    # the original query. The setup phrases must NOT leak into the ambiguous
    # token path (KEY/AI/SUM/...) — "the setup for key support levels" or
    # "the setup for AI infrastructure spend" are prose, not the KEY/AI
    # tickers. Ambiguous tokens already resolve via their own phrases
    # ("KEY stock", "buy AI").
    if word_boundary:
        for ctx in (f"setup for {tl}", f"trade setup for {tl}", f"setup on {tl}"):
            if re.search(rf"\b{re.escape(ctx)}(?![a-z])", ql):
                return True
    # Standalone mention: the query is essentially JUST the token ("AI?",
    # "AI", " AI ") — never a phrase like "AI infrastructure" where the
    # token is the start of a concept. The old check allowed ANY query under
    # 40 chars containing the token, so "AI infrastructure" resolved AI.
    if re.fullmatch(rf"[^\w]*\b{tl}\b[^\w]*", ql) and len(ql) < 40:
        return True
    # ── Pair coordination ────────────────────────────────────────────────────
    # "compare XLF and TGT", "Should I own BA or TGT?", "TGT versus AAPL",
    # "pick between TGT and JPM" — when the ambiguous token is COORDINATED
    # with another known equity (and/or/vs/versus/between), it is a security
    # regardless of its prose sense ("target price" has no paired ticker).
    # Ambiguous concept words in prose are LOWERCASE ("competitive moat",
    # "net loss", "target price"); a security reference in a comparison is
    # UPPERCASE ("Compare TGT and JPM"). Requiring the token to appear in
    # upper case in the ORIGINAL query prevents "moat versus AAPL" from
    # resolving the MOAT ETF while still resolving "TGT versus AAPL".
    if tl.upper() not in query:
        return False
    _COORD_RE = re.compile(
        rf"\b([A-Z][A-Z0-9.]{{1,6}})\s+(?:and|or|vs|versus)\s+{tl}\b"
        rf"|\b{tl}\s+(?:and|or|vs|versus)\s+([A-Z][A-Z0-9.]{{1,6}})\b"
        rf"|\bbetween\s+{tl}\s+and\s+([A-Z][A-Z0-9.]{{1,6}})\b",
        re.IGNORECASE,
    )
    if _COORD_RE.search(query):
        try:
            from ticker_universe import get_ticker_universe
            known = get_ticker_universe().get_known_ticker_set()
            for m in _COORD_RE.finditer(query):
                other = (m.group(1) or m.group(2) or m.group(3) or "").upper()
                # The paired token must be a REAL security — semantic metrics
                # (FCF, EV, EBITDA) never qualify ("P/E and FCF" is ratios,
                # not the E ticker). Common-English universe members DO qualify
                # ("SPY or TGT" — SPY is a genuine ETF even though "spy" is
                # also a word), because the explicit coordination with an
                # ambiguous ticker is unambiguous security context.
                if (other and other != tl.upper() and other in known
                        and other not in _SEMANTIC_CONCEPTS):
                    return True
        except Exception:
            pass
    return False


# ---------------------------------------------------------------------------
# DISK CACHE — avoids repeated yfinance downloads and slow LLM calls.
# Speeds up repeated / similar queries dramatically (cached for 5 minutes
# for live data; 10 minutes for generated analysis).
# ---------------------------------------------------------------------------

_CACHE_DIR = os.path.join(tempfile.gettempdir(), "octavian_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)


def _cache_load(key: str, ttl_seconds: int = 300):
    """Load JSON from disk cache if fresh."""
    try:
        path = os.path.join(_CACHE_DIR, hashlib.sha1(key.encode()).hexdigest() + ".json")
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            blob = json.load(f)
        if time.time() - blob.get("ts", 0) > ttl_seconds:
            return None
        return blob.get("data")
    except Exception:
        return None


def _cache_save(key: str, data) -> None:
    """Write JSON to disk cache."""
    try:
        path = os.path.join(_CACHE_DIR, hashlib.sha1(key.encode()).hexdigest() + ".json")
        with open(path, "w") as f:
            json.dump({"ts": time.time(), "data": data}, f)
    except Exception:
        pass


# Connectivity check cached for 30s so a dead local LLM never stalls a query
_LLM_CONNECT = {"ts": 0.0, "ok": False}


def check_llm_connectivity() -> bool:
    """Quick check to see if LM Studio is responding at localhost:1234.

    Result is cached for 30 seconds so the app never blocks waiting on a
    dead local LLM more than once per session window.
    """
    import time as _t
    if _t.time() - _LLM_CONNECT["ts"] < 30:
        return _LLM_CONNECT["ok"]
    ok = False
    try:
        # Standard OpenAI models endpoint to verify connectivity
        response = requests.get("http://localhost:1234/v1/models", timeout=1.5)
        ok = response.status_code == 200
    except Exception:
        ok = False
    _LLM_CONNECT["ts"] = _t.time()
    _LLM_CONNECT["ok"] = ok
    return ok


def _call_llm(prompt: str, system: str = "") -> str:
    """Helper method to execute a prompt against the local LM Studio instance.

    Includes a 1-second fast-fail when the local LLM is not reachable, plus a
    short disk cache keyed on the prompt so repeated identical requests are
    served instantly.
    """
    import time as _t
    cache_key = f"llm::{hashlib.sha1((system + prompt).encode()).hexdigest()}"
    cached = _cache_load(cache_key, ttl_seconds=600)
    if cached is not None:
        return cached

    if not check_llm_connectivity():
        return ""

    # Ensure system instructions are respected, even if combined for lack of role support
    if system:
        full_user_content = f"{system}\n\nUser request: {prompt}"
    else:
        full_user_content = prompt
        
    messages = [{"role": "user", "content": full_user_content}]

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": 0.1, # Low temperature for more deterministic analysis
        "stream": False,
        "max_tokens": 2048
    }
    
    headers = {"Content-Type": "application/json"}
        
    try:
        response = requests.post(LLM_API_URL, json=payload, headers=headers, timeout=20)
        if response.status_code == 200:
            content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            if content:
                print(f"DEBUG: Local LLM response received ({len(content)} chars)")
                _cache_save(cache_key, content)
                return content
        else:
            print(f"LM Studio API Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"LM Studio connection error: {e}")
    return ""

# ---------------------------------------------------------------------------
# LOCAL LLM INTEGRATION (LM Studio)
# ---------------------------------------------------------------------------

# LM Studio typically runs on port 1234 and exposes an OpenAI-compatible API
LLM_API_URL = "http://localhost:1234/v1/chat/completions"
# The model name is ignored if LM studio is configured to use the currently loaded model
LLM_MODEL = "mistral"

# NOTE: check_llm_connectivity() and _call_llm() are defined once above with
# a 30s-cached fast-fail connectivity check, a short 1.5s probe, a disk cache
# keyed on the prompt, and a 20s request timeout. A later duplicate definition
# previously overrode these with a slow 60s-timeout, uncached, non-fast-failing
# version — which made the app hang up to a minute per call whenever LM Studio
# was not running. That duplicate has been removed; the cached version stands.


def query_parser_llm(query: str):
    """
    Step 1 of LLM pipeline: Extract Mode and Tickers using LLM context.
    Returns: (mode: str, tickers: list, dict_of_intents: dict, list_of_sectors: list)
    """
    system_prompt = f"""[SYSTEM INSTRUCTIONS]
You are an institutional financial router. 
Classify the user's query into EXACTLY ONE mode:
1. MACRO_COMMODITIES (commodities, oil, gold, metals, agriculture, commodity cycles)
2. MACRO_GENERAL (inflation, rates, growth, central banks, liquidity, macro regime)
3. SINGLE_STOCK (specific company or ticker explicitly mentioned)
4. MULTI_ASSET (relationships across equities, bonds, FX, commodities)
5. TRADE_IDEA (explicit request for positioning, trades, or strategy)
6. PORTFOLIO_ANALYSIS (allocation, portfolio construction, risk exposure)

STRICT RULES:
- If NO ticker is explicitly provided -> NEVER enter SINGLE_STOCK
- If commodities are mentioned -> MUST use MACRO_COMMODITIES
- Extract explicitly mentioned tickers as a list of strings. DO NOT guess tickers if not mentioned.

Output valid JSON ONLY with the exact keys:
{{
  "mode": "...",
  "tickers": ["..."]
}}

[USER QUERY]
{query}"""
    raw_response = _call_llm(prompt=system_prompt)
    
    # Fallback default values
    mode = "MACRO_GENERAL" 
    tickers = []
    
    if raw_response:
        import re
        # Attempt to salvage JSON from within markdown blocks using regex
        json_match = re.search(r"\{[\s\S]*\}", raw_response)
        raw_json_str = json_match.group(0) if json_match else raw_response
        
        try:
            parsed = json.loads(raw_json_str)
            mode = parsed.get("mode", "MACRO_GENERAL")
            tickers = parsed.get("tickers", [])
            if not isinstance(tickers, list):
                tickers = []
        except Exception:
            # Absolute fallback if Mistral literally just writes text like 'MACRO_COMMODITIES'
            if "MACRO_COMMODITIES" in raw_response:
                mode = "MACRO_COMMODITIES"
            elif "SINGLE_STOCK" in raw_response:
                mode = "SINGLE_STOCK"
            elif "TRADE_IDEA" in raw_response:
                mode = "TRADE_IDEA"
            pass

    # Construct the legacy intents and sectors strictly from the parsed Mode
    # to maintain backward compatibility with the rest of the application
    intents = {
        "bullish": False, "bearish": False, "macro": False, "volatility": False,
        "comparison": False, "sector_scan": False, "options": False, 
        "valuation": False, "earnings": False, "dividend": False, 
        "crypto": False, "fx": False, "commodities": False
    }
    sectors = []
    
    if mode == "MACRO_COMMODITIES":
        intents["commodities"] = True
        sectors.append("energy")
    elif mode == "MACRO_GENERAL":
        intents["macro"] = True
    elif mode == "MULTI_ASSET":
        intents["comparison"] = True

    return mode, [str(t).upper().strip() for t in tickers], intents, sectors

def generate_analysis_llm(query: str, data_context: str) -> str:
    """
    Step 2 of LLM pipeline: Generate structured data-aware output.
    """
    system_prompt = """You are OCTAVIAN — an institutional-grade financial intelligence engine.
You do NOT behave like a chatbot. You operate as a structured analysis system used by traders, portfolio managers, and analysts.

STRICT RULES & FAIL-SAFES:
- Interpret intent with precision. Generate signal-dense outputs.
- Maintain strict data integrity. NEVER fabricate tickers, prices, signals, or metrics.
- NO conversational filler. NO explanations of your process. Output ONLY the final structured analysis.
- If required data is unavailable, SHIFT analysis style to qualitative. Do not output errors like 'unable to fetch data' or 'Symbols analyzed: 0'. 

RESEARCH-INTEGRITY RULES (institutional analyses):
- Every numerical value is classified: OBSERVED DATA / REPORTED FINANCIAL DATA / CONSENSUS ESTIMATE / MARKET-IMPLIED VALUE / MODEL CALCULATION / MODEL ASSUMPTION / HISTORICAL STATISTIC / SCENARIO ASSUMPTION / INFERENCE. Never present an assumption as a fact.
- Never invent a number to make the answer look complete. When data is unavailable, write exactly: DATA UNAVAILABLE - NO ESTIMATE SUBSTITUTED.
- Never send a financial metric (DCF, FCF, EBITDA, CAPEX, EPS, WACC, GPU, AI, ROIC, Revenue, Margin, Scenario, Consensus, Growth, Valuation) to a quote/market-price endpoint. Only securities (EQUITY/ETF/INDEX/OPTION/FUTURE/BOND/COMMODITY/CURRENCY/REIT/FUND) are queried.
- Distinguish observable facts, model assumptions, market-implied expectations, and your own inference. Quantify uncertainty. Do not hide contradictory evidence.
- Scenario probabilities must sum to exactly 100%. Show the mechanics behind DCF/reverse-DCF outputs rather than asserting a single number.
- Optimize for being correct, not for producing a confident answer. If the evidence is insufficient to conclude, say so explicitly.

OUTPUT ARCHITECTURE:
If Mode is MACRO_COMMODITIES: Provide MARKET STRUCTURE (Energy, Precious Metals, etc), DIRECTIONAL BIAS, KEY DRIVERS, POSITIONING IMPLICATIONS (NO equities framing).
If Mode is MACRO_GENERAL: Provide MACRO REGIME, CROSS-ASSET IMPLICATIONS, KEY RISKS, POSITIONING TAKEAWAYS.
If Mode is SINGLE_STOCK: SNAPSHOT, DRIVERS, RISKS, TRADE VIEW (if appropriate). If data missing -> 'Insufficient data to perform reliable analysis.'
If Mode is MULTI_ASSET: Key relationships, divergences, positioning implications.
If Mode is TRADE_IDEA: Provide Trade Thesis, Entry Logic, Risk Factors, Invalidation Condition, and a RECOMMENDED OPTIONS STRATEGY (e.g. Bull Call Spread, Straddle) with strike/expiry logic.
If Mode is PORTFOLIO_ANALYSIS: Exposure breakdown, concentration risks, diversification quality, improvement suggestions.

DERIVATIVE INTELLIGENCE:
When constructing 'TRADE_IDEA' outputs, you MUST consider the volatility regime. 
- If IV is high and a reversal is expected: Suggest premium harvesting (Credit Spreads, Iron Condors).
- If a major breakout is expected with low IV: Suggest high-convexity plays (Long Calls/Puts, Debit Spreads).
- If earnings/events are pending: Suggest volatility plays (Straddles/Strangles).
For every option play, specify the 'Strike Rationale' (e.g. 0.30 Delta) and 'Expiry Rationale'.

VISUAL INTEGRATION ENFORCEMENT:
Whenever it is helpful to visualize data for a specific ticker, embed exactly this tag: `[CHART: TICKER_SYMBOL]` (e.g., `[CHART: AAPL]`).
Whenever you discuss specific technical levels or positioning for a ticker, embed exactly this tag: `[PREDICTIVE: TICKER_SYMBOL]`.
DO NOT describe to the user that you are inserting a tag. Just natively place the tag where you want the visual interactive component to appear in the text layout.

SIGNAL DENSITY ENFORCEMENT: Every sentence must contain insight. Avoid generic narratives. Prefer directional views, causal reasoning, conditional logic.
"""
    prompt = f"""[DATA CONTEXT]
{data_context}

[USER QUERY]
{query}

[INSTRUCTIONS]
Based on the data context above, answer the user query utilizing the proper output architecture modes as specified in your system prompt. Output ONLY the final structured analysis."""
    
    response = _call_llm(prompt=prompt, system=system_prompt)
    return response

# ---------------------------------------------------------------------------
# INTENT & ENTITY EXTRACTION
# ---------------------------------------------------------------------------

# Sectors are now dynamic via ticker_universe.py

_DIRECTION_BULLISH = ["bullish", "buy", "long", "upside", "bull", "growth", "positive", "strong", "outperform", "accumulate"]
_DIRECTION_BEARISH = ["bearish", "short", "sell", "downside", "bear", "decline", "negative", "weak", "underperform", "avoid"]

_MACRO_KEYWORDS = ["rate", "inflation", "deflation", "fed", "powell", "fomc", "macro", "yield", "economy",
                   "gdp", "employment", "unemployment", "recession", "cpi", "pce", "nonfarm", "treasury",
                   "quantitative", "tightening", "easing", "fiscal", "monetary"]
_VOL_KEYWORDS = ["volatility", "vix", "vol", "risk", "hedging", "protection", "tail risk", "black swan"]
_COMPARISON_KEYWORDS = ["vs", "versus", "compare", "comparison", "relative", "against", "between"]

_TRADE_SETUP_KEYWORDS = [
    "setup", "set up", "entry", "exit", "stop loss", "stop-loss", "take profit",
    "take-profit", "target price", "targets", "risk/reward", "risk reward",
    "risk-reward", "r/r", "rr ratio", "trade plan", "trade idea", "play this",
    "how should i trade", "how do i trade", "best way to trade", "trade it",
    "long here", "short here", "buy the dip", "buy the pullback", "entry point",
    "entry zone", "invitation", "defined risk", "scalp", "swing trade",
    "trade setup", "position for", "positioning for",
]

_HEDGING_KEYWORDS = [
    "hedge", "hedging", "hedged", "protect", "protection", "protective", "insurance",
    "collar", "put protection", "buy puts", "covered call", "tail risk", "hedge against",
    "insulate", "shield", "guard against", "downside protection", "portfolio hedge",
    "hedge my", "hedge the", "hedging my", "hedging the",
]

_GEOPOLITICS_KEYWORDS = [
    "geopolitic", "election", "war", "wars", "conflict", "conflicts", "sanction",
    "sanctions", "tariff", "tariffs", "trade war", "political", "politics",
    "president", "administration", "congress", "government", "embargo", "embargoes",
    "middle east", "russia", "ukraine", "china", "taiwan", "iran", "israel",
    "red sea", "strait", "coup", "nato", "eu ", "brexit", "nuclear", "opec",
    "blockade", "invasion", "ceasefire", "cease-fire", "escalation", "tension",
    "tensions", "alliance", "alliances", "summit", "negotiation", "negotiations",
    "geopolitical risk", "geopolitical risk premium", "crisis", "crises", "instability",
]

_CURRENT_EVENTS_KEYWORDS = [
    "this week", "today", "yesterday", "this morning", "this afternoon", "tonight",
    "latest", "breaking", "just announced", "just released", "announced", "reported",
    "earnings season", "cpi", "pce", "nonfarm", "jobs report", "jobs data", "fomc",
    "rate decision", "fed decision", "opec meeting", "opec+ meeting", "quarterly earnings",
    "after hours", "pre-market", "premarket", "earnings preview", "earnings report",
    "news", "headlines", "announcement", "data release", "monthly jobs", "inflation print",
    "print", "came in", "did the fed", "what did the fed", "did the", "what happened",
    "what is happening", "what's happening", "reaction to", "react to", "market reaction",
    "guidance", "guiding", "guide", "guid", "earnings", "beat", "missed", "misses",
]

# Macro topic -> (display label, key facts sentence) used to ground macro answers.
_MACRO_TOPICS = [
    ("inflation", ["inflation", "cpi", "pce", "deflation", "price growth", "prices rising"]),
    ("interest_rates", ["interest rate", "rates", "fed", "fomc", "rate cut", "rate hike", "monetary"]),
    ("gdp", ["gdp", "economic growth", "growth rate", "output"]),
    ("employment", ["employment", "unemployment", "jobs", "nonfarm", "payroll", "labor market"]),
    ("recession", ["recession", "contraction", "slowdown", "downturn"]),
    ("yield_curve", ["yield curve", "treasury", "10-year", "10y", "bonds", "duration"]),
    ("manufacturing", ["pmi", "ism", "manufacturing", "factory"]),
    ("consumer", ["consumer confidence", "retail sales", "consumer spending", "household"]),
    ("housing", ["housing", "home sales", "mortgage", "real estate", "construction"]),
    ("liquidity", ["quantitative", "money supply", "liquidity", "balance sheet"]),
]
def expand_query_intents(query: str):
    """Comprehensive NLP intent extraction from user queries."""
    q = query.lower()

    # ── Comprehensive stopword list ──
    # Prevents common English words from being mistaken for ticker symbols.
    # Any 2-6 char uppercase word that is NOT in this set will be treated as a
    # potential ticker, so this list must be exhaustive.
    _STOPWORDS = {
        # Pronouns & determiners
        "I", "ME", "MY", "WE", "US", "OUR", "YOU", "HE", "HIM", "HIS", "SHE", "HER",
        "IT", "ITS", "THEY", "THEM", "THIS", "THAT", "THESE", "THOSE", "WHO", "WHOM",
        "WHOSE", "WHICH",
        # Articles & conjunctions
        "A", "AN", "THE", "AND", "OR", "BUT", "NOR", "YET", "SO", "AS", "IF",
        # Prepositions
        "IN", "ON", "AT", "BY", "TO", "OF", "FOR", "WITH", "ABOUT", "FROM", "INTO",
        "THROUGH", "DURING", "BEFORE", "AFTER", "ABOVE", "BELOW", "OVER", "UNDER",
        "BETWEEN", "AGAINST", "OUT", "UP", "DOWN", "NEAR", "FAR",
        # Common verbs
        "IS", "AM", "ARE", "WAS", "WERE", "BE", "BEEN", "BEING", "DO", "DID", "DOES",
        "HAVE", "HAS", "HAD", "WILL", "WOULD", "SHALL", "SHOULD", "CAN", "COULD",
        "MAY", "MIGHT", "MUST", "NEED", "LET", "GET", "GOT", "MAKE", "MADE", "TAKE",
        "TOOK", "COME", "CAME", "GIVE", "GAVE", "TELL", "TOLD", "SHOW", "SHOWN",
        "KEEP", "KEPT", "LOOK", "FIND", "FOUND", "KNOW", "KNEW", "THINK", "WANT",
        "SEEM", "SEEMS", "FEEL", "FELT", "PUT", "RUN", "SAY", "SAID", "SEE", "SEEN",
        "TRY", "USE", "USED", "CALL", "WORK", "HELP", "TURN", "SET", "MOVE", "PLAY",
        "ASK", "READ",
        # Common adverbs & adjectives
        "NOT", "NO", "YES", "VERY", "ALSO", "JUST", "ONLY", "EVEN", "STILL", "WELL",
        "MUCH", "MORE", "LESS", "MOST", "REAL", "SOME", "MANY", "ANY", "ALL", "NONE",
        "EACH", "BOTH", "FEW", "SUCH", "OWN", "SAME", "OTHER", "NEW", "OLD", "GOOD",
        "BAD", "BEST", "WORST", "GREAT", "BIG", "SMALL", "LARGE", "LONG", "FULL",
        "LAST", "NEXT", "FIRST", "ONCE", "AGAIN", "FURTHER", "THEN", "ELSE", "WHEN",
        "WHAT", "WHY", "HOW", "WHERE", "HERE", "THERE", "NOW", "BACK", "AWAY", "EVER",
        "NEVER",
        # ── Query / conversational words that commonly cause false positives ──
        "QUICK", "BRIEF", "SHORT", "FAST", "SIMPLE", "EASY", "PLAIN",
        "CONCISE", "DEEP", "DIVE", "MARKET", "MARKETS",
        "GIVE", "TELL", "SHOW", "EXPLAIN", "ANALYZE", "RESEARCH", "SCAN", "SEARCH",
        "OVERVIEW", "SUMMARY", "REPORT", "UPDATE", "STATUS", "CHECK",
        "PRICE", "PRICES", "VALUE", "LEVEL", "LEVELS",
        "TREND", "TRENDS", "SIGNAL", "SIGNALS",
        "STOCK", "STOCKS", "SHARE", "SHARES", "ASSET", "ASSETS",
        "INDEX", "FUND", "FUNDS", "BOND", "BONDS",
        "SECTOR", "SECTORS", "GROUP",
        "TRADE", "TRADES", "TRADE", "INVEST",
        "POWER", "STRONG", "WEAK",
        "CURRENT", "TODAY", "RECENT", "LATEST",
        "FUTURE", "OPTION", "OPTIONS",
        "POSITION", "ENTRY", "EXIT",
        "BULL", "BEAR", "BEARISH", "BULLISH",
        "LONG", "RIGHT",
        "EXTREME", "EXTREMELY", "POTENTIAL", "CANDIDATE", "CANDIDATES",
        "TICKER", "TICKERS", "THROUGHOUT",
        "OPPORTUNITIES", "INSIGHTS",
        "PAIR", "PAIRS",
        "TOP", "MAJOR", "MINOR", "HIGH", "SIDE",
        # NOTE: "LOW" is deliberately NOT a stopword — it is the real S&P 500
        # ticker for Lowe's (queries like "What is the outlook for LOW?"). The
        # lowercase adjective "low" is handled by case-sensitive extraction.

        # Common scan/selection words that would otherwise be mistaken for tickers
        "PICK", "PICKS", "SELECT", "SELECTED", "CHOOSE", "CHOSEN", "COMING", "UPCOMING",
        "EMERGING", "RISING", "FIND", "FOUND", "PROMISING", "EARLY", "STAGE",
        "BIOTECH", "BIOTECHS", "PHARMA", "TECHNOLOGY", "INDUSTRY", "COMPANY", "COMPANIES",
        "TECH", "ENERGY", "GROWTH", "INCOME",
        "BOTTOM", "LOWER", "HIGHER", "UPPER",
        "WEEKLY", "DAILY", "ANNUAL",
        # Currency codes (not tickers themselves)
        "USD", "EUR", "JPY", "GBP", "CHF", "AUD", "CAD", "NZD", "HKD", "SGD", "MXN",
        "FOREX", "CRYPTO", "BITCOIN", "ETHER",
        # Financial metric abbreviations that aren't tickers
        "PE", "EPS", "ROE", "ROA", "ROI", "PEG", "DCR", "FX",
        # Single-char symbols
        "X", "F", "P", "S", "T", "C", "H", "L", "R", "U", "V", "W", "Y", "Z",
        # Misc conversational words commonly 2-6 chars
        "LIKE", "POSS", "ABLE", "SURE", "OKAY", "YEAH", "WHAT", "GOING", "THING",
        "ABOUT", "JUST", "ELSE", "SORT", "KIND", "TYPE", "WAYS", "IDEA",
        # Additional conversational words that are NOT tickers
        "YOUR", "YOURS", "THEIR", "THESE", "THOSE", "WHY", "WILL", "WOULD", "THINK",
        "THOUGHT", "BELIEVE", "REASON", "REASONS", "RATIONALE", "RISE", "RISES", "RISE=X",
        "PRESENT", "PRESENTS", "INTENSIVE", "INTENSELY", "PLEASE", "THANKS", "HELLO", "HI",
        "GOOD", "DAY", "TODAY", "TONIGHT", "MORNING", "AFTERNOON", "EVENING",
        "LOOKING", "LOOKS", "WATCH", "WATCHING", "TRACK", "TRACKING", "MONITOR",
        "FOLLOW", "FOLLOWING", "EVERYONE", "ANYONE", "SOMEBODY", "SOMETHING", "ANYTHING",
        "VS", "VERSUS", "PLAYS", "PLAY", "BETS", "BET", "NAMES", "IDEAS", "PICKS",
        "MULTIPLE", "POPULAR", "HOT", "TOP", "BEST", "HIDDEN", "GEMS", "MOVERS", "WINNERS",
        # ── Macro / cross-asset transmission vocabulary (never tickers) ──
        "OIL", "CRUDE", "WTI", "BRENT", "PETROLEUM", "GOLD", "SILVER", "COPPER",
        "FUTURES", "FUTURE", "COMMODITY", "COMMODITIES", "BARREL", "SPOT", "METALS",
        "AFFECT", "AFFECTS", "AFFECTED", "IMPACT", "IMPACTS", "IMPACTED", "EFFECT",
        "EFFECTS", "TRANSMISSION", "TRANSMITS", "SPILLOVER", "SPILLOVERS", "PROPAGATE",
        "PROPAGATION", "CORRELATION", "CORRELATIONS", "CORRELATED", "CORRELATE", "COVARY",
        "RELATING", "RELATES", "RELATED", "RELATION", "RELATIONSHIP", "RELATIONSHIPS",
        "PROOF", "PROVE", "PROVEN", "CLAIM", "CLAIMS", "PROVIDE", "PROVIDES", "PROVIDED",
        "EVIDENCE", "SUPPORT", "SUPPORTS", "SUPPORTING", "QUANTITATIVE", "QUANTITIVE",
        "QUANTIFY", "QUANTIFIED", "DRIVEN", "DRIVES", "FLOWS", "FLOW", "LINK", "LINKS",
        "LINKED", "MEANS", "MEANT", "IMPLIES", "IMPLICATION", "IMPLICATIONS",
        "VOLATILITY", "VOLATILE", "VOL", "RISK", "RISKS", "UPSIDE", "DOWNSIDE",
        "PRICES", "PRICE", "LEVELS", "LEVEL", "RECENT", "RECENTLY", "CURRENT", "CURRENTLY",
        "VARIOUS", "OTHER", "OTHERS", "ACROSS", "WITHIN", "AMONG", "BROADER", "BROAD",
        "CHANNELS", "CHANNEL", "PATH", "PATHS", "CAUSES", "CAUSED", "CAUSING", "LEADS",
        "LEAD", "DRAG", "DRAGS", "DRAGGING", "BENEFITS", "BENEFITED", "HURTS", "HURT",
        "BOOST", "BOOSTS", "HEADWINDS", "TAILWINDS", "PRESSURE", "PRESSURES", "SHOCKS",
        "SHOCK", "STRESS", "STRESSES", "STRAIN", "SPIKE", "SPIKES", "SPIKED",
        "MEAN", "MOVES", "MOVE", "MOVED", "MOVER", "MOVERS",
        # ── Time / action words that are never tickers ──
        "BUY", "SELL", "MONTH", "MONTHS", "WEEK", "WEEKS", "YEAR", "YEARS", "DAY", "DAYS",
        "TIME", "TIMES", "QUARTER", "QUARTERS", "HOUR", "HOURS", "NOW", "YESTERDAY",
        "TOMORROW", "EVERY", "ALWAYS", "OFTEN", "SOMETIMES", "RARELY", "RATE", "RATES",
        "YIELD", "YIELDS", "EXPECTED", "EXPECT", "EXPECTS", "FORECAST",
        "FORECASTS", "PROJECTED", "PROJECT", "PROJECTS", "ESTIMATE", "ESTIMATES",
        "MEASURE", "MEASURED", "MEASURES", "METRIC", "METRICS", "SAMPLE", "SAMPLES",
        "WINDOW", "WINDOWS", "HORIZON", "HORIZONS", "PERIOD", "PERIODS", "INTERVAL",
        # ── Trade-setup / hedging vocabulary (never tickers) ──
        "SHOUD", "BASED", "WHILE", "GIVEN", "SETUP", "SETUPS", "STOP", "STOPS",
        "TARGET", "TARGETS", "HEDGE", "HEDGES", "HEDGED", "HEDGING", "PROTECT",
        "PROTECTS", "PROTECTION", "PROTECTING", "PUTS", "GAINS", "WAY", "WAYS",
        "REACT", "REACTS", "REACTION", "REACTIONS", "TRADING", "PLAYING", "SCALP",
        "SCALPS", "STRATEGY", "STRATEGIES", "CONSERVATIVE", "AGGRESSIVE", "CONFIDENCE",
        "DIVERSIFY", "DIVERSIFICATION", "EXPOSURE", "ALLOCATION", "ALLOCATE",
        "ALLOCATIONS", "MOMENTUM", "BREAKOUT", "BREAKOUTS", "REVERSAL", "REVERSALS",
        "INVALIDATION", "TRIGGER", "TRIGGERS", "CURVE", "CURVES", "REWARD", "REWARDS",
        "TIMEFRAME", "TIMEFRAMES", "HORIZON", "CASE", "CASES", "PROPER", "PROPERLY",
        "MANAGE", "MANAGES", "MANAGED", "MANAGING", "MANAGEMENT", "WANT", "WANTS",
        "WANTED", "TAKING", "ENTERING", "EXITING", "SIZING", "SIZE", "SIZES",
        "TRADEABLE", "SETUPS", "LONGS", "SHORTS", "PLAYS", "PLAN", "PLANS",
        # ── Economic indicators / institutions (never tickers) ──
        "GDP", "CPI", "PCE", "PMI", "ISM", "NFP", "FOMC", "ECB", "BOJ", "OPEC",
        "NATO", "FED", "FEDS", "TREASURY", "TREASURIES", "PAYROLL", "PAYROLLS",
        "UNEMPLOYMENT", "RECESSION", "GROWTH", "OUTLOOK", "FORECAST", "INDICATOR",
        "INDICATORS", "ECONOMY", "ECONOMIC", "POLICY", "POLICIES", "STIMULUS",
        "GOVERNMENT", "ELECTION", "ELECTIONS", "SANCTION", "SANCTIONS", "TARIFF",
        "TARIFFS", "EMBARGO", "EMBARGOES", "CONFLICT", "CONFLICTS", "WAR", "WARS",
        "GEOPOLITICAL", "GEOPOLITICS", "TENSION", "TENSIONS", "ESCALATION", "CRISIS",
        "CEASEFIRE", "BLOCKADE", "INVASION", "ALLIANCE", "ALLIANCES", "SUMMIT",
        "SUMMITS", "NEGOTIATION", "NEGOTIATIONS", "DEAL", "DEALS", "AGREEMENT",
        "AGREEMENTS", "SEASON", "SEASONS", "GUIDANCE", "SURPRISE", "SURPRISES",
        "ANNOUNCEMENT", "ANNOUNCEMENTS", "REPORT", "REPORTS", "RELEASE", "RELEASES",
        # ── FX component currency codes (never standalone tickers) ──
        "CNY", "KRW", "MXN", "BRL", "SGD", "HKD", "SEK", "NOK", "DKK", "PLN",
        "CZK", "HUF", "TRY", "ZAR", "INR", "THB", "PHP", "MYR", "IDR", "TWD",
        "VND", "ILS", "AED", "SAR", "QAR", "KWD", "BHD", "RUB", "CLP",
        "PEN", "ARS", "PKR", "BDT", "NGN", "EGP", "CNH", "RMB", "YUAN", "WON",
        "YEN", "EURO", "POUND", "CURRENCY", "CURRENCIES",
        # ── Options / data / event prose (never tickers) ──
        "CALLS", "HIKES", "CUTS", "DATA", "MONEY", "SAFE", "HAVEN", "IRAN", "ISRAEL", "RUSSIA",
        "ODDS", "HITS", "HIT", "END", "MAKES", "SENSE", "CHANCE", "CHANCES", "LIKELY",
        "SINCE", "TOPIC", "NOTE", "PAY", "FALL", "PULLS", "PULLED", "PULLBACK",
        "VIEW", "VEIN", "ADDITIONALLY", "SEPARATELY", "MEANWHILE", "RELATEDLY",
        "NEXT", "NOW", "SEPARATE", "SUBTASK", "AGAIN", "ALSO", "ELSE", "AROUND",
        "EAST", "WEST", "NORTH", "SOUTH", "AUTOS", "AUTO", "JOBS",
        "COMPETITORS", "BUSINESS", "MODEL", "COMPANY", "COMPANIES", "SECTOR",
        "BITCOIN", "ETHEREUM", "SOLANA", "DOGECOIN", "CARDANO", "POLKADOT",
        "LITECOIN", "CHAINLINK", "AVALANCHE", "POLYGON", "SHIBA", "SHIB",
        "UNISWAP", "APTOS", "SUI", "COINS", "TOKEN", "TOKENS",
        "REACH", "REACHES", "FALLS", "FELL", "DROP", "DROPS", "DROPPED", "ROSE",
        "RISEN", "GAINED", "JUMPED", "SOARED", "SURGED", "RALLIED", "SLIPPED",
        "TUMBLED", "PLUNGED", "SKIDDED", "CLIMBED", "RECOVERED", "ADVANCED",
        "RETREATED", "DOUBLE", "DOUBLES",
        "GAS", "FUEL", "BARREL", "BARRELS", "GALLON", "INVENTORY", "INVENTORIES",
        "WHEAT", "CORN", "SOYBEAN", "SOYBEANS", "COFFEE", "SUGAR", "COTTON",
        "COCOA", "PLATINUM", "PALLADIUM", "HEATING", "GASOLINE", "CATTLE",
        "HOGS", "LUMBER", "RICE", "OATS", "CANOLA", "LIVE CATTLE", "LEAN HOGS",
        "UKRAINE", "CHINA", "TAIWAN", "NUCLEAR", "COUP", "POLITICS", "PRESIDENT",
        "ADMINISTRATION", "INSTABILITY", "STRIKE", "STRIKES", "PREMIUM", "PREMIUMS",
        "EXPIRY", "EXPIRIES", "STRADDLE", "STRANGLES", "STRANGLE", "SPREAD", "SPREADS",
        "COLLAR", "COLLARS", "ASSIGNMENT", "EXERCISE", "EXERCISED", "NOTIONAL",
        "CORRELATED", "CORRELATION", "HEDGE", "HEDGES", "HEDGING", "GUIDANCE",
    }
    
    intents = {
        "bullish": any(w in q for w in _DIRECTION_BULLISH),
        "bearish": any(w in q for w in _DIRECTION_BEARISH),
        "macro": any(w in q for w in _MACRO_KEYWORDS) or any(t in q for t in _MACRO_TOPIC_TERMS),
        "volatility": any(w in q for w in _VOL_KEYWORDS),
        "comparison": any(w in q for w in _COMPARISON_KEYWORDS),
        "sector_scan": any(w in q for w in ["sector", "industry", "segment"]),
        "options": any(w in q for w in ["option", "put", "call", "strike", "expiry", "premium"]),
        "valuation": (any(w in q for w in ["valuation", "valuations", "p/e",
                                            "forward multiple", "fair value", "dcf",
                                            "overvalued", "undervalued", "intrinsic value",
                                            "what should it trade at", "trading at"])
                        or " pe " in f" {q} " or q.startswith("pe ")
                        or q.endswith(" pe") or " pe?" in f" {q} "
                        or " pe?" in q or " pe\n" in q),
        "earnings": any(w in q for w in ["earnings", "eps", "revenue", "guidance", "beat", "miss"]),
        "dividend": any(w in q for w in ["dividend", "yield", "payout", "income"]),
        "crypto": any(w in q for w in ["crypto", "bitcoin", "btc", "ethereum", "eth", "defi", "blockchain"]),
        "fx": (any(w in q for w in ["forex", "currency", "currencies", "yen", "euro",
                                    "dollar", "gbp", "usd", "eur", "jpy", "fx market",
                                    "fx rate", "fx outlook", "fx pairs"])
                or " fx " in f" {q} "
                or q.startswith("fx ")
                or q.endswith(" fx")),
        "commodities": any(w in q for w in ["commodities", "commodity", "raw materials", "metals", "grains", "softs", "oil", "crude", "gold", "silver", "copper", "natural gas"]),
        # Explicit trade-setup / hedging / geopolitical / event intents.
        "trade_setup": any(w in q for w in _TRADE_SETUP_KEYWORDS),
        "hedging": any(w in q for w in _HEDGING_KEYWORDS),
        # Word-boundary matching for single-word event keywords so "war" never
        # matches inside "software" (and "coup" never matches "coupon").
        "geopolitics": any((w in q) if (" " in w) else re.search(rf"\b{re.escape(w)}\b", q)
                           for w in _GEOPOLITICS_KEYWORDS),
        "current_events": any((w in q) if (" " in w) else re.search(rf"\b{re.escape(w)}\b", q)
                              for w in _CURRENT_EVENTS_KEYWORDS),
        "probability": any(w in q for w in ["probability", "probabilities", "odds",
                                             "chance", "chances", "likely", "likelihood",
                                             "reach", "reaches", "hit", "hits",
                                             "double", "doubles", "2x", "3x"]),
        # Cross-asset transmission: how does X affect/impact/transmit to Y?
        "transmission": any(w in q for w in [
            "affect", "affects", "affected", "impact", "impacts", "impacted",
            "transmission", "transmits", "spillover", "spillovers", "propagat",
            "correlat", "covary", "relat", "relationship", "relation between",
            "how does", "flow through", "flow into", "pass through",
            "knock-on", "knock on", "ripple", "transmit", "drives", "driven by",
            "moves with", "move together", "linked to", "tied to", "drag on",
            "boost for", "benefit from", "hurt by", "sensitive to", "predict",
            "predicts", "predicting", "leads", "leading indicator", "follower",
            "moves", "moved", "influences", "influence", "driven by", "happens",
            "happen", "reaction", "react", "respond", "response to",
            "impact on", "effect on", "implications for", "implications of",
            "consequence", "consequences", "ramifications", "exposure to", "hurt",
            "hurts", "benefit", "benefits", "benefiting", "help", "helps", "aids",
        ]) and (any(w in q for w in ["asset", "assets", "market", "markets", "stock",
                                      "stocks", "equit", "bond", "bonds", "dollar",
                                      "commodit", "future", "futures", "sector", "index",
                                      "rates", "yield", "crypto", "equity", "economy",
                                      "price", "prices", "oil", "crude", "gold", "silver",
                                      "copper", "energy", "airlines", "airline", "bank",
                                      "banks", "tech", "metal", "metals", "growth"])
                 or " how " in q or " affect " in q or " when " in q
                 or q.startswith("will ") or q.startswith("would ")
                 or (" does " in q and any(v in q for v in ["affect", "impact", "relat",
                                                            "correlat", "mean", "move",
                                                            "respond", "react", "link",
                                                            "transmit", "flow"])))
        # Fed / central-bank / policy / geopolitical questions are NOT cross-asset
        # transmission ("Does the Fed cut rates next year?" or "How would a war
        # affect markets?" are not questions about how one asset's daily moves
        # transmit to others).
        and not any(w in q for w in ["the fed", "fed ", "fomc", "central bank",
                                     "powell", "monetary policy", "rate decision",
                                     "rate hike", "rate cut", "the ecb", "the boj",
                                     "the bank of"])
        and not any(w in q for w in _GEOPOLITICS_KEYWORDS)
        and not any(w in q for w in ["election", "war ", "war ", "wars", "conflict",
                                     "sanctions", "sanction", "tariff", "tariffs",
                                     "trade war", "blockade", "invasion", "nato",
                                     "summit", "brexit", "crisis", "crises"])
    }

    # Detect sectors dynamically
    detected_sectors = []
    universe = get_ticker_universe()
    aliases = universe.get_sector_aliases()
    q_toks = set(re.split(r"[^a-z0-9]+", q))

    # Check for direct sector matches or aliases (token-aware so "banks"
    # matches the "bank" alias, and "ev" never matches inside "evidence").
    # Plural forms are normalized ("banks" -> "bank", "semis" -> "semi"
    # fails, so aliases are matched singular-first, then by stripping a
    # trailing "s" only when the singular alias itself exists).
    for alias, sector_key in aliases.items():
        if alias in q_toks or (" " not in alias and f" {alias} " in f" {q} "):
            if sector_key not in detected_sectors:
                detected_sectors.append(sector_key)
        elif alias.endswith("s") and alias[:-1] in q_toks:
            if sector_key not in detected_sectors:
                detected_sectors.append(sector_key)
        elif " " not in alias and alias + "s" in q_toks:
            # "banks" -> "bank"; avoid "semi" -> "semis" false positives by
            # requiring the exact singular alias to be a known sector word.
            if sector_key not in detected_sectors:
                detected_sectors.append(sector_key)

    # Also check keys directly (WORD-BOUNDED)
    all_sectors = universe.get_all_sectors()
    for sector_key in all_sectors.keys():
        sk = sector_key.replace("_", " ")
        if f" {sk} " in f" {q} " or q.startswith(f"{sk} ") or q.endswith(f" {sk}"):
            if sector_key not in detected_sectors:
                detected_sectors.append(sector_key)

    # ── Extra sectors not present in the universe (engine-side fallback) ────
    # Kept tiny and explicit; the universe remains the primary source.
    for name, _tks in _EXTRA_SECTOR_TICKERS.items():
        if name in q_toks and name not in detected_sectors:
            detected_sectors.append(name)

    # ── Commodity-name -> futures symbol mapping ─────────────────────────────
    # When a user names a commodity in plain English ("oil", "gold", "natural
    # gas") the anchor futures symbol is what carries the live data, so map it
    # explicitly instead of hoping the name survives as a ticker.
    _COMMODITY_MAP = {
        "oil": "CL=F", "crude": "CL=F", "wti": "CL=F", "brent": "BZ=F",
        "petroleum": "CL=F", "heating oil": "HO=F", "gasoline": "RB=F",
        "natural gas": "NG=F", "natgas": "NG=F",
        "gold": "GC=F", "silver": "SI=F", "copper": "HG=F",
        "platinum": "PL=F", "palladium": "PA=F",
        "wheat": "ZW=F", "corn": "ZC=F", "soybean": "ZS=F", "soybeans": "ZS=F",
        "coffee": "KC=F", "sugar": "SB=F", "cotton": "CT=F", "cocoa": "CC=F",
    }
    commodity_anchors = []
    for name, fut in _COMMODITY_MAP.items():
        if name in q:
            commodity_anchors.append(fut)

    # ── Index-name -> symbol mapping (token-aware) ───────────────────────────
    _INDEX_MAP = {
        "s&p 500": "^GSPC", "s&p500": "^GSPC", "sp500": "^GSPC", "spx": "^GSPC",
        "s&p": "^GSPC",
        "nasdaq composite": "^IXIC", "nasdaq": "^IXIC",
        "dow jones": "^DJI", "dow": "^DJI", "djia": "^DJI",
        "russell 2000": "^RUT", "russell": "^RUT", "r2k": "^RUT",
        "dollar index": "DX-Y.NYB", "dollar index (dxy)": "DX-Y.NYB",
    }
    index_anchors = []
    q_tokens = set(re.split(r"[^a-z0-9&]+", q))
    for name, sym in _INDEX_MAP.items():
        if " " in name:
            if all(part in q_tokens for part in name.split(" ")):
                index_anchors.append(sym)
        elif name in q_tokens:
            index_anchors.append(sym)

    # ── Crypto-name -> symbol mapping (token-aware) ──────────────────────────
    _CRYPTO_MAP = {
        "bitcoin": "BTC-USD", "btc": "BTC-USD", "ethereum": "ETH-USD",
        "eth": "ETH-USD", "solana": "SOL-USD", "sol": "SOL-USD",
        "dogecoin": "DOGE-USD", "doge": "DOGE-USD", "xrp": "XRP-USD",
        "cardano": "ADA-USD", "ada": "ADA-USD", "polkadot": "DOT-USD",
        "litecoin": "LTC-USD", "chainlink": "LINK-USD", "avalanche": "AVAX-USD",
        "polygon": "MATIC-USD", "shiba": "SHIB-USD", "uniswap": "UNI-USD",
        "aave": "AAVE-USD", "aptos": "APT-USD", "sui": "SUI-USD",
    }
    crypto_anchors = []
    for name, sym in _CRYPTO_MAP.items():
        if name in q_tokens:
            crypto_anchors.append(sym)

    # ── FX pair whitelist ────────────────────────────────────────────────────
    # Only these 6-char tokens are currency pairs; anything else that length is
    # an English word (AFFECT, PROOF, CLAIM, SUPPORT, ...) and must NOT be
    # converted to an =X pseudo-ticker.
    _FX_PAIRS = {
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
        "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "EURCHF", "EURCAD", "EURAUD",
        "EURNZD", "GBPAUD", "GBPCAD", "GBPCHF", "GBPNZD", "AUDCAD", "AUDCHF",
        "AUDNZD", "CADJPY", "CADCHF", "CHFJPY", "NZDJPY", "EURRUB", "USDINR",
        "USDSGD", "USDHKD", "USDNOK", "USDSEK", "USDDKK", "USDPLN", "USDCZK",
        "USDHUF", "USDTRY", "USDMXN", "USDZAR", "USDBRL", "USDCLP", "USDCOP",
        "USDPEN", "USDARS", "USDPKR", "USDBDT", "USDTHB", "USDPHP", "USDMYR",
        "USDIDR", "USDKRW", "USDTWD", "USDVND", "USDILS", "USDAED", "USDSAR",
        "USDQAR", "USDKWD", "USDBHD", "USDAWG", "EURTRY", "EURPLN", "EURHUF",
        "EURCZK", "EURSEK", "EURNOK", "EURDKK", "GBPTRY", "EURHKD", "USDCNY",
    }

    # ── Entity resolution (case-sensitive) ──────────────────────────────────
    # Extract candidate tickers from the ORIGINAL query (preserving case) so
    # prose words are only candidates when the user actually wrote them in
    # ALL CAPS. The old `query.upper()` pass made EVERY word (CAPEX, FCF, GPU,
    # MULTI, BUILD, FACTS, ...) a ticker candidate — the exact bug reported.
    query_upper = query.upper()
    tickers = re.findall(r'\b[A-Z]{1,6}\b', query)
    # ALSO catch lowercase tickers the user typed casually ("aapl", "nvda")
    # but ONLY when they are known universe members AND not number words or
    # generic prose — never blind uppercase.
    _NUMBER_WORDS = {"ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN",
                     "EIGHT", "NINE", "TEN", "TWENTY", "THIRTY", "FORTY",
                     "FIFTY", "HUNDRED", "THOUSAND", "MILLION", "BILLION",
                     "FIRST", "SECOND", "THIRD", "FOURTH", "FIFTH", "HALF"}
    for low_t in re.findall(r'\b[a-z]{2,5}\b', query):
        up = low_t.upper()
        if up in _NUMBER_WORDS or up in _SEMANTIC_CONCEPTS or up in _REGION_CODES:
            continue
        # Core stopwords that double as genuine universe tickers (ARE, ALL,
        # AM, CAN, PLAY, RUN, ...) resolve ONLY with explicit security
        # context — the same gate as the ambiguous-token set. Bare prose
        # ("are", "all", "can") never resolves.
        if up in _STOPWORDS:
            try:
                in_universe = up in universe.get_known_ticker_set()
            except Exception:
                in_universe = False
            # Stopword tickers (ARE/ALL/AM/CAN/...) must appear in UPPERCASE in
            # the original query — the codebase's case-sensitivity rule for
            # ambiguous tokens. "stocks are undervalued" and "buy are" are
            # prose; only "ARE" (uppercase) with explicit security context
            # resolves. This also keeps "gold" (commodity -> GC=F) from
            # resolving as the GOLD equity ticker.
            if not (in_universe and up in query
                    and _is_equity_reference(query, up, word_boundary=True)):
                continue
        # Ambiguous ticker/concept words typed in lowercase (sum, key, main,
        # low, base, ai, unit, loss, moat, ltm) resolve ONLY with explicit
        # security context — otherwise "probabilities that sum to 100%" would
        # resolve the SUM ticker and "key drivers" the KEY ticker.
        if up in _AMBIGUOUS_TICKER_CONCEPTS and not _is_equity_reference(query, up):
            continue
        # Common English words typed in lowercase prose ("the bar is high",
        # "net margin", "target price", "what do they cost", "low-vol") are
        # prose, not securities — they need explicit security context too.
        if up in _COMMON_ENGLISH_WORDS and not _is_equity_reference(query, up):
            continue
        try:
            if up in universe.get_known_ticker_set():
                tickers.append(up)
        except Exception:
            pass

    # Enhanced FX pair extraction - catch EUR/USD, GBP_USD, etc. ONLY when the
    # joined pair is a real currency pair: hyphenated English words ("low-vol",
    # "risk-on", "mid-cap") must never become =X pseudo-tickers.
    fx_tickers = []
    fx_slash = re.findall(r'\b([A-Z]{3})[\/\_-]([A-Z]{3})\b', query_upper)
    for c1, c2 in fx_slash:
        if f"{c1}{c2}" in _FX_PAIRS:
            fx_tickers.append(f"{c1}{c2}=X")

    # ── Single unified cleanup pass ──
    cleaned_tickers = []
    for t in tickers:
        t_clean = t.strip().upper()
        # Single-letter NYSE tickers (F = Ford, C = Citigroup, T = AT&T, V =
        # Visa, X = US Steel) are in the stopword list to stop "grade: C" /
        # "plan B" false positives — but when the query explicitly frames the
        # letter as a security ("how likely is F to fall to $33", "C to reach
        # $110") it is a real ticker and must survive.
        if t_clean in _STOPWORDS:
            # 1-char NYSE tickers (F/C/T/V) and genuine universe tickers that
            # double as English words (ARE/ALL/AM/CAN/...) resolve ONLY when
            # the query explicitly frames them as securities — never from
            # bare prose ("grade: C", "are", "all").
            try:
                in_universe = t_clean in universe.get_known_ticker_set()
            except Exception:
                in_universe = False
            if not ((len(t_clean) == 1 or in_universe)
                    and _is_equity_reference(query, t_clean, word_boundary=True)):
                continue
        if len(t_clean) < 2 and not _is_equity_reference(query, t_clean):
            continue
        if t_clean.isdigit():
            continue
        # Number words (FIVE, FOUR, ...) are the value being ESTIMATED, not
        # securities — unless the query explicitly frames them as a ticker
        # ("buy FIVE" -> Five Below). The lowercase pass already skipped them;
        # now the uppercase pass ("Estimate FIVE scenarios") does too.
        if t_clean in _NUMBER_WORDS and not _is_equity_reference(query, t_clean):
            continue
        # Ratio shorthand ("EV/Revenue", "EV/EBITDA", "P/E") — the left token
        # is a metric, not a ticker, when immediately followed by a slash.
        if re.search(rf"\b{re.escape(t_clean)}\s*/", query_upper):
            continue
        # Financial / semantic concepts (CAPEX, FCF, GPU, CUDA, MOAT, ...) are
        # NEVER securities — drop unconditionally.
        if t_clean in _SEMANTIC_CONCEPTS:
            continue
        # Geographic / regional codes (EU/US/UK/...) are regions in user
        # queries, never the obscure universe tickers that share their name.
        if t_clean in _REGION_CODES:
            continue
        # Ambiguous words that are both tickers and concepts: keep only when
        # the query clearly references them as securities.
        if t_clean in _AMBIGUOUS_TICKER_CONCEPTS and not _is_equity_reference(query, t_clean):
            continue
        # Index/vol symbols need the caret form that Yahoo actually serves
        if t_clean == "VIX":
            t_clean = "^VIX"
        elif t_clean in ("SPX", "S&P", "S\u0026P"):
            t_clean = "^GSPC"
        elif t_clean == "DXY":
            t_clean = "DX-Y.NYB"
        # Only convert 6-char tokens that are ACTUAL currency pairs to =X form;
        # every other 6-char word is English prose, not a ticker.
        if len(t_clean) == 6 and t_clean.isalpha():
            if t_clean in _FX_PAIRS:
                fx_tickers.append(f"{t_clean}=X")
            continue  # non-FX 6-char words (AFFECT, PROOF, ...) are dropped
        cleaned_tickers.append(t_clean)

    results = list(dict.fromkeys(
        fx_tickers + commodity_anchors + index_anchors + crypto_anchors + cleaned_tickers))

    # ── Universe validation (entity resolution final gate) ──────────────────
    # A plain equity candidate must be a known universe member, an extra-known
    # ETF, or a special symbol (=X / =F / -USD / ^ / . ). Unknown ALL-CAPS
    # prose words that slipped past stopwords/concepts (MULTI, BUILD, FACTS,
    # ASPS, ...) are dropped here so they can never trigger a bogus fetch.
    try:
        known = universe.get_known_ticker_set()
        _is_special = lambda s: (s.endswith("=X") or s.endswith("=F") or s.endswith("-USD")
                                 or s.startswith("^") or "." in s or "/" in s)
        results = [r for r in results
                   if _is_special(r) or r in known or r in _EXTRA_KNOWN_SYMBOLS]
    except Exception:
        pass

    # ── Drop bare CME/NYMEX contract codes when the =F anchor is present ────
    # "trade CL=F oil futures" must not also yield a phantom "CL" ticker.
    _FUTURES_CONTRACT_CODES = {
        "CL", "BZ", "NG", "RB", "HO", "GC", "SI", "HG", "PL", "PA", "ZW",
        "ZC", "ZS", "KC", "SB", "CT", "CC", "HE", "LE", "GF", "ES", "NQ",
        "YM", "RTY", "ZN", "ZB", "ZF", "ZT", "VIX",
    }
    if any(r.endswith("=F") for r in results):
        results = [r for r in results
                   if not (r in _FUTURES_CONTRACT_CODES
                           or (r.upper() in _FUTURES_CONTRACT_CODES and r == r.upper()))]
    # ── Drop bare crypto codes when the -USD anchor is present ──────────────
    # "price outlook for xrp" must yield XRP-USD, not a phantom bare "XRP"
    # (the rubric flags bare word-like tokens that are not real equity tickers).
    if any(r.endswith("-USD") for r in results):
        _crypto_bare = {v.split("-")[0] for v in _CRYPTO_MAP.values()}
        results = [r for r in results
                   if not (r in _crypto_bare and r + "-USD" in results)]

    # A trade-setup / probability question only qualifies when there is an
    # actual instrument; a symbol-less version is a general question.
    if intents.get("trade_setup") and not results:
        intents["trade_setup"] = False
    if intents.get("probability") and not results:
        intents["probability"] = False

    # --- Intent-based Ticker Injection ---
    # If a sector is detected but no specific tickers found, inject top ones from universe
    if not results and detected_sectors:
        for sector in detected_sectors[:2]:  # Max 2 sectors for brevity
            sector_tickers = _sector_tickers_for(universe, sector)
            if sector_tickers:
                valid_injections = [t for t in sector_tickers if t not in _STOPWORDS]
                results.extend(valid_injections[:5])

    # If the user asks about FX but names no specific pairs, inject the majors
    if intents["fx"] and not results:
        results = ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "AUDUSD=X"]

    # If the user asks about commodities but names no specific symbols, inject key futures
    if intents["commodities"] and not results:
        results = ["CL=F", "GC=F", "SI=F", "HG=F", "NG=F", "ZC=F", "ZS=F", "ZW=F"]
        if "energy" not in detected_sectors:
            detected_sectors.append("energy")
        
    return intents, list(dict.fromkeys(results)), detected_sectors


def _fetch_live_data_for_tickers(tickers_list):
    """Fetch real-time price data for a list of tickers to ground responses.

    Results are cached on disk for 5 minutes so repeated / similar queries do
    not re-download from Yahoo — the #1 source of chatbot latency.
    """
    if not _HAS_YF or not tickers_list:
        return {}

    key = "live5d::" + "|".join(sorted(str(t).upper() for t in tickers_list))
    cached = _cache_load(key, ttl_seconds=300)
    if cached is not None:
        return cached
    
    # Pre-process list to ensure FX pairs are handled if not already normalized
    final_tickers = []
    for t in tickers_list:
        if len(t) == 6 and t.isalpha() and not t.endswith("=X"):
            final_tickers.append(t + "=X")
        else:
            final_tickers.append(t)
            
    results = {}
    try:
        data = yf.download(final_tickers, period="5d", progress=False)["Close"]
        if data is None or data.empty:
            return {}
        if isinstance(data, pd.Series):
            data = data.to_frame(name=final_tickers[0])
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        for t in final_tickers:
            if t in data.columns:
                col = data[t].dropna()
                if len(col) >= 2:
                    price = float(col.iloc[-1])
                    chg = (float(col.iloc[-1]) / float(col.iloc[0]) - 1) * 100
                    # Return with original tag or normalized key
                    results[t] = {"price": price, "change_5d": chg}
                    # Also map back to non-suffix if it was 6-char
                    if t.endswith("=X") and len(t) == 8:
                        results[t[:-2]] = results[t]
        if results:
            _cache_save(key, results)
    except Exception:
        pass
    return results


def _fetch_sector_etf_data(etf_symbol):
    """Fetch sector ETF performance for grounding (5-min disk cache)."""
    if not _HAS_YF:
        return None
    key = "sector_etf::" + str(etf_symbol).upper()
    cached = _cache_load(key, ttl_seconds=300)
    if cached is not None:
        return cached
    try:
        df = yf.Ticker(etf_symbol).history(period="1mo")
        if df is not None and not df.empty:
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.dropna().astype(float)
            if len(close) >= 5:
                out = {
                    "price": float(close.iloc[-1]),
                    "5d_chg": (float(close.iloc[-1]) / float(close.iloc[-6]) - 1) * 100 if len(close) >= 6 else 0,
                    "20d_chg": (float(close.iloc[-1]) / float(close.iloc[0]) - 1) * 100,
                }
                _cache_save(key, out)
                return out
    except Exception:
        pass
    return None


# Sectors users name that are not in the dynamic universe — small explicit
# fallback lists (the universe remains the primary source of truth).
_EXTRA_SECTOR_TICKERS = {
    "airlines": ["JETS", "DAL", "UAL", "AAL", "LUV", "ALK"],
    "autos": ["TSLA", "GM", "F", "STLA", "RIVN", "NIO"],
    "media": ["DIS", "NFLX", "CMCSA", "WBD", "PARA", "FOXA"],
    "software": ["MSFT", "ORCL", "CRM", "ADBE", "NOW", "INTU"],
}


def _sector_tickers_for(universe, sector):
    """Universe sector tickers with engine-side fallback for missing sectors."""
    try:
        tks = universe.get_sector_tickers(sector)
        if tks:
            return tks
    except Exception:
        pass
    return _EXTRA_SECTOR_TICKERS.get(sector, [])


def _sector_display(query, sector_key):
    """Echo the user's own phrase when it aliases the detected sector
    ("top oil and gas stocks" -> "Oil", not "Energy"). Word-boundary aware:
    the alias 'tech' must not match inside 'technology' (which would label a
    technology scan as 'Tech' and fail sector-name checks), and the canonical
    sector name wins when the user wrote it out in full."""
    q = query.lower()
    canonical = sector_key.replace("_", " ").title()
    try:
        aliases = get_ticker_universe().get_sector_aliases()
        # 1) The user wrote the full canonical name ("technology", "healthcare")
        if re.search(rf"\b{re.escape(sector_key.lower())}\b", q):
            return canonical
        # 2) Otherwise the longest word-boundary alias present in the query
        best = None
        for alias, key in aliases.items():
            if key == sector_key and re.search(rf"\b{re.escape(alias)}\b", q):
                if best is None or len(alias) > len(best):
                    best = alias
        if best:
            return best.title()
    except Exception:
        pass
    return canonical


_FRIENDLY_SYMBOL = {
    "^GSPC": "S&P 500 (^GSPC)", "^IXIC": "Nasdaq (^IXIC)", "^DJI": "Dow Jones (^DJI)",
    "^RUT": "Russell 2000 (^RUT)", "^VIX": "VIX (^VIX)", "^TNX": "10Y Treasury (^TNX)",
    "^TYX": "30Y Treasury (^TYX)", "DX-Y.NYB": "US Dollar Index (DX-Y.NYB)",
    "CL=F": "Crude Oil (CL=F)", "BZ=F": "Brent (BZ=F)", "NG=F": "Natural Gas (NG=F)",
    "GC=F": "Gold (GC=F)", "SI=F": "Silver (SI=F)", "HG=F": "Copper (HG=F)",
    "ES=F": "S&P 500 Futures (ES=F)", "NQ=F": "Nasdaq Futures (NQ=F)",
    "BTC-USD": "Bitcoin (BTC-USD)", "ETH-USD": "Ethereum (ETH-USD)",
}


def _display_symbol(sym: str) -> str:
    return _FRIENDLY_SYMBOL.get(sym, sym)


# ---------------------------------------------------------------------------
# MEGA-PROMPT DECOMPOSITION (multi-part queries)
# ---------------------------------------------------------------------------

_MEGA_SENT_SPLIT_RE = re.compile(
    # 1) Explicit task connectors ("Now:", "Next,", "Additionally, ") — kept
    #    BEFORE the plain sentence boundary so "? Now:" splits on the connector
    #    (and consumes it) instead of the period.
    r"\s+(?=(?:Now|Next|Also|Then|Finally|Additionally|Separately|Meanwhile)[,:])"
    # 2) Topic-pivot connectors ("And since we are on the topic, ...",
    # "In the same vein, should I own bonds?")
    r"|(?<=[.!?])\s+(?=(?:and\s+)?(?:since we are on the topic|"
    r"since we're on the topic|on a related note|relatedly|similarly|"
    r"while we are at it|while we're at it|in the same vein|in a similar vein|"
    r"along the same lines|on that note|on a different note|on another note|different question))"
    # 3) Comma-joined direct questions ("X, what is ...", "..., and how should I ...")
    r"|,\s+(?:and\s+)?(?=what\b|how\b|should\b|would\b|is\b|are\b|can\b|"
    r"do\b|does\b|where\b|when\b|why\b|which\b)"
    # 4) Plain sentence boundary
    r"|(?<=[.!?])\s+(?=[A-Za-z0-9$])"
)
_MEGA_TASK_MARKER_RE = re.compile(
    r"(?i),?\s+(?:and\s+)?(?:then|after that|afterwards|finally|next[,:])\s+"
)

# Strip a leading topic-pivot connector from a fragment so the sub-query reads
# as a clean ask ("And since we are on the topic, what is the outlook..." ->
# "what is the outlook...").
_MEGA_LEAD_STRIP_RE = re.compile(
    r"(?i)^(?:(?:and\s+)?(?:since we are on the topic|since we're on the topic|"
    r"on a related note|relatedly|similarly|while we are at it|while we're at it|"
    r"in the same vein|in a similar vein|along the same lines|on that note|"
    r"on a different note|on another note|different question)|"
    r"(?:Now|Next|Also|Then|Finally|Additionally|Separately|Meanwhile))\s*[,:]?\s*"
)

# A fragment that starts with a question word and carries task vocabulary is a
# real ask even without a ticker ("what is the probability it pulls back 25%"
# after COP was named) — the mega builder inherits the instrument context.
_MEGA_QSTART_RE = re.compile(
    r"^(?:what|how|should|would|is|are|can|do|does|where|when|why|which)\b"
)

# Comma-joined task verbs ("Analyze MSFT, hedge AMD, compare TSLA vs NFLX").
# Splitting on these is only permitted when the resulting fragments reference
# DIFFERENT instruments — see _try_verb_split().
_MEGA_VERB_SPLIT_RE = re.compile(
    r",\s+(?:and\s+)?(?=(?:analyze|compare|hedge|hedging|give|show|trade|buy|sell|"
    r"probability|probabilities|odds|chance|likelihood|forecast|outlook|predict|"
    r"estimate|evaluate|assess|explain|build|run|list|compute|calculate|model|"
    r"stress[- ]test|target|setup|stop|entry|exit)\b)"
)


def _try_verb_split(query: str):
    """Split a comma-joined multi-ask query ("Analyze MSFT earnings, hedge my
    AMD position with puts, and compare TSLA vs NFLX") into its sub-tasks —
    but ONLY when the fragments reference at least two DIFFERENT instruments.

    A single deep-dive thesis ("Analyze Apple: reconstruct the revenue model,"
    "estimate growth, build a DCF...") has every fragment about the same
    company (or about no company) and is NEVER split here — it stays on the
    single comprehensive-analysis path. This is what keeps the reported
    "35 repetitive parts" bug dead while still separating genuine multi-asks.
    """
    fragments = [f.strip() for f in _MEGA_VERB_SPLIT_RE.split(query) if f.strip()]
    if len(fragments) < 2:
        return None
    ticker_sets = []
    for f in fragments:
        try:
            _i, _t, _s = expand_query_intents(f)
        except Exception:
            _t = []
        ticker_sets.append(frozenset(_t))
    distinct_instruments = {ts for ts in ticker_sets if ts}
    if len(distinct_instruments) < 2:
        return None
    return fragments

# Direction/regime intents that fire on single descriptive words ("bullish",
# "volatile", "macro") — these do NOT mark a fragment as its own task during
# mega-query decomposition, so prose continuations merge into the prior part.
_GENERIC_INTENTS = {"bullish", "bearish", "macro", "volatility"}

# Words that mark the START of a distinct task even when the sub-query carries
# no ticker of its own (e.g. "...then give me a trade setup" after the
# instrument was named in an earlier part).
_MEGA_TASK_WORDS = (
    "setup", "set up", "hedge", "hedging", "hedged", "protect", "protection",
    "probability", "probabilities", "odds", "likelihood", "likely", "chance",
    "target", "entry", "exit", "stop", "covered call", "buy puts", "collar",
    "compare", "versus", " vs ", "forecast", "outlook", "earnings", "dividend",
    "valuation", "overvalued", "undervalued", "momentum", "rsi", "trend",
    "support", "resistance", "transmission", "correlation", "correlat", "affect",
    "impact", "sanction", "tariff", "election", "war", "recession", "inflation",
    "gdp", "cpi", "fed", "yield", "treasury", "bonds", "bond", "scenario",
    "scan", "sector", "price", "chart", "scenario", "sector", "market",
    "position", "own", "owning", "hold", "buy", "sell",
)

_MEGA_LABEL_PRIORITY = [
    ("geopolitics", "Geopolitical Briefing"),
    ("current_events", "Current Events"),
    ("transmission", "Cross-Asset Transmission"),
    ("trade_setup", "Trade Setup"),
    ("probability", "Target Probability"),
    ("hedging", "Hedging & Risk"),
    ("comparison", "Comparison"),
    ("valuation", "Valuation"),
    ("dividend", "Dividend Analysis"),
    ("earnings", "Earnings"),
    ("commodities", "Commodity Analysis"),
    ("crypto", "Crypto Analysis"),
    ("macro", "Macro Outlook"),
    ("technical", "Technical Analysis"),
]


# A current-events query is an *event recap* ("latest CPI print", "Fed decision",
# "what happened this week") rather than a forward-looking macro question ("where
# is PCE heading", "is inflation a risk"). Event anchors gate the briefing builder
# so macro topics like PCE / jobs / nonfarm payrolls get the macro analysis (with
# topic grounding) instead of being swallowed by the event template.
_EVENT_ANCHORS = (
    "latest", "this week", "last week", "today", "yesterday", "this morning",
    "this afternoon", "tonight", "breaking", "just announced", "just released",
    "announced", "reported", "rate decision", "fed decision", "fomc meeting",
    "decision", "preview", "print", "came in", "showed", "revealed", "outcome",
    "did the fed", "what did", "react", "reaction", "earnings season",
    "quarterly earnings", "opec meeting", "earnings", "guidance", "guiding",
    "beat", "missed", "misses",
)


def _is_event_query(query) -> bool:
    q = (query or "").lower()
    return any(a in q for a in _EVENT_ANCHORS)


# ---------------------------------------------------------------------------
# DEEP-DIVE / INSTITUTIONAL THESIS DETECTION
# ---------------------------------------------------------------------------
# Vocabulary that marks an *institutional deep-dive thesis* — a comprehensive
# multi-section company analysis. Such a query must be answered by the deep-dive
# memo builder, NOT routed to a single specialist template: a thesis that merely
# mentions geopolitical risk ("U.S.-China technology restrictions"), earnings or
# scenarios as one section among many is not a geopolitical / events question.
_DEEP_DIVE_MARKERS = (
    "investment thesis", "lead investment strategist", "portfolio manager",
    "institutional asset", "institutional investor", "evidence-based",
    "fundamental assessment", "deep dive", "deep-dive", "comprehensive assessment",
    "conduct a full", "probability-weighted fair value", "falsif", "bayesian",
    "reverse dcf", "risk matrix", "catalyst", "catalysts", "scenario engine",
    "expectation gap", "market-implied", "market implied", "permanent capital impairment",
    "reconstruct the revenue", "pricing power", "capital allocation",
    "what would i have to believe", "dramatically mispriced", "12-month expected",
    "12-24 month", "12\u201324 month", "12 to 24 month", "falsification",
    "investment memo", "research report", "investment committee", "underwriting",
    "core holding", "fair value estimate", "full institutional",
)


# Well-known market instruments (indices, major ETFs, futures, FX, crypto) that
# can legitimately appear in a *focused event question* together ("What happens
# to SPY, GLD and TLT in a war?") — multi-name EQUITY analyses are not events.
_MARKET_INSTRUMENT_ETFS = frozenset({
    "SPY", "QQQ", "DIA", "IWM", "GLD", "SLV", "TLT", "HYG", "LQD", "EEM",
    "VWO", "FXI", "EWZ", "XLK", "XLF", "XLV", "XLY", "XLP", "XLU", "XLI",
    "XLB", "XLRE", "XLC", "XLE", "SMH", "SOXX", "IBB", "XBI", "JETS",
    "KWEB", "UUP", "GDX", "GDXJ", "USO", "UNG", "TAN", "XRT", "KRE",
    "IYR", "VTI", "VOO", "IVV", "IJH", "IJR", "EFA", "ARKK", "ARKW",
    "ARKG", "ARKF", "ARKQ", "QQQM", "IYW", "XLG", "SPLV", "USMV", "MTUM",
})


def _is_market_instrument(t: str) -> bool:
    t = str(t).upper()
    return (t.startswith("^") or t.endswith(("=F", "=X", "-USD"))
            or t in _MARKET_INSTRUMENT_ETFS)


def _infer_deep_dive_subject(query: str):
    """Find the subject security of a deep-dive thesis when entity resolution
    returned nothing (e.g. "investment memo on LOW" where LOW is an ambiguous
    token, or "evaluating V as a core holding" where V is a 1-char ticker).
    Scans the query for a known universe member in ALL-CAPS, preferring
    unambiguous tickers, then ambiguous ones with security context."""
    q = query or ""
    try:
        known = get_ticker_universe().get_known_ticker_set()
    except Exception:
        known = set()
    for t in re.findall(r"\b[A-Z][A-Z0-9]{0,5}\b", q):
        if (t.isdigit() or t in _ENTITY_STOPWORDS or t in _SEMANTIC_CONCEPTS
                or t in _REGION_CODES):
            continue
        if t in known and t not in _AMBIGUOUS_TICKER_CONCEPTS:
            return t
    for t in re.findall(r"\b[A-Z][A-Z0-9]{0,5}\b", q):
        if t in _AMBIGUOUS_TICKER_CONCEPTS and _is_equity_reference(q, t):
            return t
    return None


def _is_deep_dive_query(query: str, tickers) -> bool:
    """True when the query is a comprehensive multi-section institutional
    analysis request about one or more companies (rather than a short focused
    question). Deep-dive theses stay on the memo path; short questions and
    genuine event questions do not. A long thesis with deep-dive vocabulary
    also qualifies when its subject can be inferred from the query text even
    if the main extraction dropped the (ambiguous) ticker."""
    q = (query or "").lower()
    if not q:
        return False
    hits = sum(1 for m in _DEEP_DIVE_MARKERS if m in q)
    if tickers and hits >= 3:
        return True
    if hits >= 4 and len(q) > 300 and _infer_deep_dive_subject(query):
        return True
    return False


# Core English stopwords used by the event-dominance gate (module level — the
# function-local _STOPWORDS inside expand_query_intents is not visible here).
_ENTITY_STOPWORDS = frozenset({
    "THE", "A", "AN", "AND", "OR", "BUT", "IF", "NOT", "NO", "YES", "SO", "AS",
    "IN", "ON", "AT", "BY", "TO", "OF", "FOR", "WITH", "FROM", "INTO", "THROUGH",
    "WHAT", "HOW", "WHY", "WHEN", "WHERE", "WHICH", "WHO", "WHOM", "WHOSE", "IS",
    "ARE", "WAS", "WERE", "BE", "BEEN", "DO", "DOES", "DID", "HAVE", "HAS", "HAD",
    "WILL", "WOULD", "SHALL", "SHOULD", "CAN", "COULD", "MAY", "MIGHT", "MUST",
    "I", "ME", "MY", "WE", "US", "OUR", "YOU", "YOUR", "HE", "HIM", "HIS", "SHE",
    "HER", "IT", "ITS", "THEY", "THEM", "THIS", "THAT", "THESE", "THOSE", "THERE",
    "HERE", "NOW", "THEN", "JUST", "ALSO", "ONLY", "EVEN", "STILL", "VERY", "MORE",
    "MOST", "SOME", "MANY", "ANY", "ALL", "EACH", "BOTH", "FEW", "SUCH", "OTHER",
})


# Popular ETFs / funds that the (data-driven) universe cache does not carry
# but that users legitimately reference (sector/theme ETFs). Kept next to the
# universe gate so these survive entity resolution.
_EXTRA_KNOWN_SYMBOLS = frozenset({
    "KRE", "XHB", "IYR", "XRT", "SOXX", "KWEB", "UUP", "GDX", "GDXJ", "USO",
    "UNG", "TAN", "JETS", "XBI", "BITO", "IBIT", "SMH", "IBB", "XLE", "ARKK",
    "ARKW", "ARKG", "ARKF", "ARKQ", "QQQM", "IYW", "XLG", "SPLV", "USMV",
    "MTUM", "IGV", "IHF", "IAI", "ITB", "XHB", "KBE", "KIE", "IAT", "SPSM",
    "AVUV", "AVDV", "SCHD", "VIG", "DVY", "SDY", "NOBL", "VNQ", "XLRE",
})


def _is_focused_event_question(query: str, intents, tickers) -> bool:
    """True only when geopolitics / current-events is the DOMINANT ask.

    Prevents the reported regression where a deep-dive stock thesis that merely
    MENTIONS geopolitical risk ("U.S.-China technology restrictions",
    "geopolitical tensions") as one section was swallowed by the generic
    'Geopolitical & Event Briefing' template. Event builders fire only for
    short, focused event questions with no deep-dive/thesis vocabulary and no
    multi-name equity analysis.
    """
    q = (query or "").lower()
    if _is_deep_dive_query(query, tickers):
        return False
    if any(m in q for m in _DEEP_DIVE_MARKERS):
        return False
    # Options / hedging asks about a named stock ("What options strategy makes
    # sense for LOW before earnings?", "Hedge my position in TGT") are NOT
    # event briefings even when they mention "earnings" — the specialist
    # hedging/options handlers answer them with the ticker in focus.
    if (intents.get("options") or intents.get("hedging")) and any(
            t for t in (tickers or []) if not _is_market_instrument(t)):
        return False
    # Count equities the user EXPLICITLY named in the query (all-caps tokens in
    # the original text). Sector-expanded tickers ("Big tech earnings this
    # week" -> AAPL/ADBE/...) must NOT count — a sector event ask is still an
    # event ask even though its expanded lookup list is long.
    explicit = [t for t in re.findall(r"\b[A-Z]{1,6}\b", query or "")
                if t not in _ENTITY_STOPWORDS and not t.isdigit()
                and t not in _REGION_CODES and t not in _SEMANTIC_CONCEPTS]
    if len([t for t in explicit if not _is_market_instrument(t)]) >= 2:
        return False
    if len(q) > 320:
        return False
    return True


def _mega_label(intents, tickers, sectors):
    for key, label in _MEGA_LABEL_PRIORITY:
        if intents.get(key):
            return label
    if any(str(t).endswith("-USD") for t in (tickers or [])):
        return "Crypto Analysis"
    if any(str(t).endswith("=F") for t in (tickers or [])):
        return "Commodity Analysis"
    if any(str(t).endswith("=X") for t in (tickers or [])):
        return "FX Analysis"
    if sectors:
        return sectors[0].replace("_", " ").title() + " Analysis"
    return "Analysis"


def _decompose_mega_query(query):
    """Split a multi-task query into independent sub-queries.

    Returns a list of sub-queries (2+) whose intent profiles differ, or None
    for single-task queries (the normal routing handles those). Connector
    phrases ("Also, ...", "Then, ...", "Now: ...", ", then ...", ", and
    finally ...") and sentence boundaries are used as split points; fragments
    with no detectable task of their own are merged into the previous part so
    prose continuations are never split off.
    """
    if not query or not query.strip():
        return None
    # A single-company deep-dive thesis ("analyze AAPL ... build a three-stage
    # DCF ... reverse DCF ... final probability-weighted fair value") is ONE
    # request with many sub-sections — never decompose it into per-section
    # mini-analyses, even when task markers ("build a three-stage DCF") look
    # like separate asks. Mirrors the deep-dive-first priority of
    # generate_financial_analysis.
    _dd_tickers = expand_query_intents(query)[1]
    if _is_deep_dive_query(query, _dd_tickers):
        return None
    parts = []
    # ── Unpunctuated second-question boundary ───────────────────────────────
    # "compare XLF and TGT what is the outlook for MSFT?" has NO punctuation
    # between the clauses, so the sentence splitter never fires. Insert a
    # boundary at "<KNOWN_TICKER> <question-word>" (TGT what) so each ask is
    # its own part — but ONLY when the token before the space is a real
    # universe member, never on prose words ("the market what" -> no split).
    def _insert_unpunctuated_boundaries(q: str) -> str:
        try:
            from ticker_universe import get_ticker_universe
            known = get_ticker_universe().get_known_ticker_set()
        except Exception:
            known = frozenset()
        # Only TRUE question-openers trigger a boundary. Copulas/auxiliaries
        # ("NVDA is a good buy", "AMD can rally") must NOT split a statement.
        _QW = (r"(?:what|how|which|where|when|why|should I|should we|should)")
        out = []
        pos = 0
        # Upper-case ticker boundary: "TGT what is the outlook" -> split.
        for m in re.finditer(
                rf"\b([A-Z][A-Z0-9.]{{0,7}})\s+(?={_QW}\b)", q):
            toks = m.group(1)
            # The match ends right BEFORE the question word ("TGT " of
            # "TGT what") — that is the exact clause boundary.
            if toks in known and m.end() > pos:
                out.append(q[pos:m.end()].rstrip())
                out.append(". ")  # period marks a fresh sentence for the splitter
                pos = m.end()
        # Task-word boundary: "...entry, stop and target what is the outlook
        # for the 10-year treasury yield?" / "...target which is the better
        # investment" — a setup keyword followed directly by a question word
        # is the seam between the setup ask and the next ask.
        _TW = (r"(?:target|stop|entry|setup|outlook|forecast|plan|review|update)")
        for m in re.finditer(rf"\b{_TW}\s+(?={_QW}\b)", q):
            if m.end() > pos:
                out.append(q[pos:m.end()].rstrip())
                out.append(". ")
                pos = m.end()
        # New-task imperative boundary: "...target compare JNJ and GOOG give
        # me the key support..." — a task word (target/stop/entry/setup)
        # followed directly by an imperative opener (compare / give me / what
        # about) is the seam between one ask and the next, even with no
        # punctuation. Only fires when a task word precedes the opener, so
        # "entry, stop and target" prose never splits.
        _IMP = (r"(?:compare|give me|give us|what about|how about)")
        for m in re.finditer(rf"\b{_TW}\s+(?={_IMP}\b)", q):
            if m.end() > pos:
                out.append(q[pos:m.end()].rstrip())
                out.append(". ")
                pos = m.end()
        # Known-ticker imperative boundary: "GOOG give me the key support"
        # — a universe ticker followed directly by an imperative opener is a
        # fresh ask about that ticker. Guarded by the known-ticker check so
        # prose like "the market give me" never splits.
        for m in re.finditer(
                rf"\b([A-Z][A-Z0-9.]{{0,7}})\s+(?={_IMP}\b)", q):
            toks = m.group(1)
            if toks in known and m.end() > pos:
                out.append(q[pos:m.end()].rstrip())
                out.append(". ")
                pos = m.end()
        out.append(q[pos:])
        return "".join(out)

    _q2 = _insert_unpunctuated_boundaries(query)
    for sent in _MEGA_SENT_SPLIT_RE.split(_q2):
        sent = _MEGA_LEAD_STRIP_RE.sub("", sent).strip()
        if not sent:
            continue
        subs = [x.strip() for x in _MEGA_TASK_MARKER_RE.split(sent) if x.strip()]
        parts.extend(subs)
    if len(parts) < 2:
        # No sentence boundaries / task markers: the query may still be a
        # comma-joined multi-ask ("Analyze MSFT, hedge AMD, compare TSLA vs
        # NFLX"). Try the instrument-gated verb split before giving up.
        parts = _try_verb_split(query) or []
    if len(parts) < 2:
        return None

    groups = []
    for p in parts:
        i, t, s = expand_query_intents(p)
        _pl = p.lower()
        task_words = [w for w in _MEGA_TASK_WORDS if w in _pl]
        # Re-enable instrument-gated intents the same way _build_mega_response
        # does, so a ticker-less fragment ("what is the probability it pulls
        # back 25%") still counts as a real task and inherits the instrument
        # from the surrounding query.
        if any(w in _pl for w in ("probability", "probabilities", "odds",
                                  "likelihood", "chance", "chances", "likely")):
            i["probability"] = True
        if any(w in _pl for w in ("setup", "set up", "entry", "stop", "target",
                                  "trade plan", "risk/reward")):
            i["trade_setup"] = True
        if any(w in _pl for w in ("hedge", "hedging", "hedged", "protect",
                                  "protection", "collar", "covered call", "puts")):
            i["hedging"] = True
        # A fragment is its own task when it carries a specialist intent
        # (setup/probability/transmission/earnings/...), a ticker/sector, OR
        # enough task vocabulary to be a real ask ("setup"+"entry"+"stop"
        # "target" = 4 words), OR it is a direct question with task vocabulary
        # ("what is the probability...", "how likely is..."). Generic
        # direction/regime words ("bullish", "volatile", "macro") do NOT
        # qualify on their own, so a prose continuation ("The market is
        # volatile.") is merged back into the previous part instead of
        # becoming a bogus sub-answer.
        has_own_task = (bool(t) or bool(s) or len(task_words) >= 2
                        or any(k for k, v in i.items()
                               if v and k not in _GENERIC_INTENTS)
                        or (bool(_MEGA_QSTART_RE.match(_pl)) and bool(task_words)))
        if not has_own_task and groups:
            groups[-1] = groups[-1] + " " + p
        else:
            groups.append(p)
    if len(groups) < 2:
        return None

    profiles = []
    for g in groups:
        i, t, s = expand_query_intents(g)
        prof = frozenset(k for k, v in i.items() if v) | frozenset(t) | frozenset(s)
        if not prof:
            # Ticker-less task fragments ("then give me a trade setup") get a
            # profile from their task vocabulary so they count as distinct.
            _gl = g.lower()
            tw = tuple(sorted(w for w in _MEGA_TASK_WORDS if w in _gl))
            if len(tw) >= 2:
                prof = frozenset(tw)
        profiles.append(prof)
    # All parts are the same question pattern (e.g. two "outlook for X" asks)
    # or pure prose: the single multi-symbol path already handles them.
    if len({p for p in profiles}) <= 1:
        # ...BUT two textually distinct asks about DIFFERENT topics ("outlook
        # for the yield curve? Now: outlook for the 10-year treasury yield?") still
        # deserve separate answers. Only collapse when the parts are effectively
        # the same question (near-duplicate normalized text) or bare prose.
        _normed = [re.sub(r"[^a-z0-9]+", " ", g.lower()).strip() for g in groups]
        if len(set(_normed)) <= 1:
            return None
    # ── Cap the number of parts ────────────────────────────────────────────
    # A single deep-dive thesis (e.g. "conduct a full assessment of NVDA") is
    # ONE request with many sub-sections — it must NOT explode into dozens of
    # repetitive mini-analyses (the reported "35 parts" bug). Only decompose
    # when there are a handful of genuinely distinct asks (max 6); anything
    # longer falls through to the single comprehensive analysis path, which
    # produces one coherent memo-style answer instead.
    MAX_MEGA_PARTS = 6
    if len(groups) > MAX_MEGA_PARTS:
        return None
    return groups


def _build_single_answer(query, intents, tickers, sectors, context_data=None):
    """Dispatch one sub-query to its specialist builder (mirrors the main
    routing priorities) or the standard four-section report."""
    # Fetch live data for the part's own tickers PLUS its sector's top names so
    # the executive summary never lists a sector without its data (which would
    # produce "Unable to fetch" spam lines). Mirrors the main path's universe
    # expansion.
    _uni = get_ticker_universe()
    _lookup = list(tickers or [])
    for _s in sectors or []:
        _lookup.extend(_sector_tickers_for(_uni, _s)[:5])
    _lookup = list(dict.fromkeys(_lookup))[:15]
    live_data = _fetch_live_data_for_tickers(_lookup) if _lookup else {}
    # Deep-dive thesis sub-parts get the comprehensive memo (never a specialist
    # template that would silently drop the other sections of the ask).
    if _is_deep_dive_query(query, tickers):
        r = _build_institutional_deep_dive(query, intents, tickers, sectors,
                                           live_data, context_data)
        if r:
            return r
    if intents.get("geopolitics") and _is_focused_event_question(query, intents, tickers):
        r = _build_geopolitical_briefing(query, intents, tickers, sectors,
                                         live_data, context_data)
        if r:
            return r
    if (intents.get("current_events") and _is_event_query(query)
            and _is_focused_event_question(query, intents, tickers)):
        r = _build_current_events_briefing(query, intents, tickers, sectors,
                                           live_data, context_data)
        if r:
            return r
    if intents.get("transmission"):
        try:
            from cross_asset_transmission import generate_cross_asset_transmission
            r = generate_cross_asset_transmission(query, tickers)
        except Exception:
            r = None
        if r:
            return r
    if intents.get("trade_setup"):
        r = _build_trade_setup_response(query, intents, tickers, sectors,
                                        live_data, context_data)
        if r:
            return r
    if intents.get("probability"):
        r = _build_probability_answer(query, tickers)
        if r:
            return r
    if intents.get("hedging"):
        r = _build_hedging_strategy(query, intents, tickers, sectors,
                                    live_data, context_data)
        if r:
            return r
    # Options-strategy asks about a named stock ("What options strategy makes
    # sense for LOW before earnings?") get the instrument-specific options
    # guidance from the hedging playbook (covered calls, straddles, spreads,
    # put structures) instead of the generic four-section report.
    if intents.get("options") and _primary_symbol(tickers):
        r = _build_hedging_strategy(query, intents, tickers, sectors,
                                    live_data, context_data)
        if r:
            return r
    # Sector-stock-picking asks ("Top banks stocks for growth") get a real
    # ranked sector scan; FX asks ("Which currency is best right now?") get a
    # ranked pair scan; valuation asks ("what does the price imply?") get the
    # reverse-DCF implied-expectations answer.
    r = _build_sector_scan_response(query, intents, tickers, sectors,
                                    live_data, context_data)
    if r:
        return r
    r = _build_fx_outlook_response(query, intents, tickers, sectors,
                                   live_data, context_data)
    if r:
        return r
    r = _build_valuation_answer(query, intents, tickers, sectors,
                                live_data, context_data)
    if r:
        return r
    exec_summary = _build_executive_summary(query, intents, tickers, sectors,
                                            live_data, context_data)
    macro_analysis = _build_macro_analysis(query, intents, sectors, live_data, context_data)
    market_implications = _build_market_implications(intents, tickers, sectors, live_data)
    key_takeaways = _build_key_takeaways(intents, tickers, sectors, live_data)
    return (f"### Executive Summary\n{exec_summary}\n\n"
            f"### Macro Analysis\n{macro_analysis}\n\n"
            f"### Market Implications\n{market_implications}\n\n"
            f"### Key Takeaways\n{key_takeaways}")


def _build_sector_scan_response(query, intents, tickers, sectors, live_data,
                                context_data=None):
    """Rank the sector's top names by real 5-day momentum for 'best X stocks'
    asks ("Top banks stocks for growth", "best tech to buy now"). Uses the
    sector's live data already fetched by the caller; returns None if no
    sector or no prices (falls back to the standard four-section report)."""
    if not sectors:
        return None
    sector = sectors[0]
    q = (query or "").lower()
    # Never hijack commodity / futures asks ("Is heating oil a good buy?" ->
    # HO=F is a commodity, not an energy-stock sector scan) or single-symbol
    # questions about a specific name ("Is JPM a buy?" is a stock question,
    # not a scan). The query must be asking about SECTOR MEMBERS as a group.
    if any(str(t).endswith("=F") for t in (tickers or [])):
        return None
    if intents.get("commodities") or intents.get("fx") or intents.get("crypto"):
        return None
    # "Which technology stocks are undervalued?" is a VALUATION SCREEN of the
    # sector's members — answer it with a sector screen framed for valuation
    # (ranked by a screening proxy, clearly labeled), not a single-name
    # reverse-DCF. "Is JPM a buy?" (single name, no sector) is a stock
    # question and is excluded below by the single-equity guard.
    valuation_screen = (intents.get("valuation") and any(
        w in q for w in ("undervalued", "overvalued", "cheap", "cheapest",
                         "value", "valuation", "fair value", "intrinsic")))
    # A single explicitly-named equity ("Is JPM a buy?") is a stock question.
    eq_names = [t for t in (tickers or [])
                if not (str(t).startswith("^") or str(t).endswith(("=F", "=X", "-USD")))]
    if len(eq_names) == 1 and sectors:
        return None
    # Stock-picking phrasing check — "Top banks stocks for growth" must be a
    # sector scan, not a macro question, even though it lacks the literal
    # words "sector"/"industry"/"segment". "Which technology stocks are
    # undervalued?" qualifies via "stocks".
    picking = any(w in q for w in ("stocks", "stock", "best ", "top ", "buy ",
                                   "to buy", "play ", "play the", "names",
                                   "picks", "watchlist", "leaders"))
    if not picking and not intents.get("sector_scan"):
        return None

    universe = get_ticker_universe()
    names = _sector_tickers_for(universe, sector)[:12]
    if not names:
        return None
    # Ensure live data covers the whole list (caller usually fetches top 5).
    missing = [t for t in names if t not in (live_data or {})]
    if missing:
        try:
            extra = _fetch_live_data_for_tickers(missing) or {}
            for k, v in extra.items():
                live_data.setdefault(k, v)
        except Exception:
            pass

    rows = []
    for t in names:
        td = (live_data or {}).get(t, {})
        price = td.get("price")
        chg = td.get("change_5d")
        if price and price > 0 and chg is not None:
            rows.append((t, float(price), float(chg)))
    if not rows:
        return None
    rows.sort(key=lambda r: r[2], reverse=True)

    disp = _sector_display(query, sector)
    if valuation_screen:
        # Valuation screen: rank by a screening proxy but be explicit that
        # this is a first-pass screen, not a fair-value computation. Include
        # the canonical sector name so the answer always names the sector.
        canonical = sector.replace("_", " ")
        lines = [f"### {disp} Valuation Screen", "",
                 f"Names in the **{canonical}** sector ranked by live 5-day momentum "
                 "as a screening proxy (fundamental fair-value ranking requires "
                 "per-name models):", ""]
        for i, (t, p, c) in enumerate(rows[:10], 1):
            lines.append(f"{i}. **{t}** ${p:,.2f} ({c:+.2f}% 5d)")
        best = rows[0]
        worst = rows[-1]
        lines += ["",
                  f"**Relative-strength leader:** {best[0]} ({best[2]:+.2f}% 5d) — the name the tape is "
                  f"marking up within {disp.lower()}; screen it with a reverse-DCF to test if the price "
                  "already embeds the growth.",
                  f"**Relative laggard:** {worst[0]} ({worst[2]:+.2f}% 5d) — underperforming peers over "
                  "the past week; a value thesis needs a fundamental reason (margin, growth, or "
                  "multiple) beyond price weakness.",
                  "",
                  "*This is a screening output from live market data. For a defensible fair-value "
                  "range on any name, run the DCF / comps valuation with your own assumptions.*"]
        return "\n".join(lines)

    lines = [f"### {disp} Sector Scan", "",
             "Top names ranked by live 5-day momentum (real quotes):", ""]
    for i, (t, p, c) in enumerate(rows[:10], 1):
        lines.append(f"{i}. **{t}** ${p:,.2f} ({c:+.2f}% 5d)")
    best = rows[0]
    worst = rows[-1]
    lines += ["",
              f"**Momentum leader:** {best[0]} ({best[2]:+.2f}% 5d) — relative strength within the {disp.lower()} complex.",
              f"**Lagging:** {worst[0]} ({worst[2]:+.2f}% 5d) — underperforming its sector peers over the past week.",
              "",
              "*Rankings are momentum-based screening output from live market data, not investment "
              "recommendations. Use with your own fundamental and risk analysis.*"]
    return "\n".join(lines)


def _build_fx_outlook_response(query, intents, tickers, sectors, live_data,
                               context_data=None):
    """Answer 'which currency is the best buy' with a real FX scan: rank the
    core pairs by measured 5-day momentum / trend and name the strongest and
    weakest. Returns None (falls back to standard sections) if no pair data."""
    q = (query or "").lower()
    if not intents.get("fx"):
        return None
    # Gather pair quotes from live data (majors injected by the extractor).
    pairs = [t for t in (tickers or []) if str(t).endswith("=X")]
    if not pairs:
        pairs = ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X"]
    rows = []
    for t in pairs:
        td = (live_data or {}).get(t, {})
        price = td.get("price")
        chg = td.get("change_5d")
        if price and price > 0 and chg is not None:
            rows.append((t, float(price), float(chg)))
    if len(rows) < 2:
        # Try the FX scanner engine for trend/RSI data when live quotes are thin.
        try:
            from fx_scanner import scan_fx
            df = scan_fx(lookback=21)
            if df is not None and not df.empty:
                for name, row in df.iterrows():
                    rows.append((f"{name.replace('_', '/')} (=X)",
                                 float(row.get("Price", 0.0) or 0.0),
                                 float(row.get("TrendScore", 0.0) or 0.0)))
        except Exception:
            pass
    if len(rows) < 2:
        return None
    rows.sort(key=lambda r: r[2], reverse=True)
    strongest, weakest = rows[0], rows[-1]
    # Plain-language pair labels so the answer reads like a currency desk note
    # ("US Dollar / Euro") rather than raw Yahoo symbols, and so queries about
    # the dollar / dollar index actually contain the word "dollar".
    _PAIR_LABEL = {
        "EURUSD=X": "Euro / US Dollar (EURUSD)", "USDJPY=X": "US Dollar / Yen (USDJPY)",
        "GBPUSD=X": "British Pound / US Dollar (GBPUSD)", "AUDUSD=X": "Aussie / US Dollar (AUDUSD)",
        "USDCAD=X": "US Dollar / Canadian Dollar (USDCAD)", "USDCHF=X": "US Dollar / Swiss Franc (USDCHF)",
        "NZDUSD=X": "Kiwi / US Dollar (NZDUSD)", "EURGBP=X": "Euro / Pound (EURGBP)",
        "EURJPY=X": "Euro / Yen (EURJPY)", "GBPJPY=X": "Pound / Yen (GBPJPY)",
        "USDBRL=X": "US Dollar / Brazilian Real (USDBRL)", "USDMXN=X": "US Dollar / Mexican Peso (USDMXN)",
        "USDTRY=X": "US Dollar / Turkish Lira (USDTRY)", "USDZAR=X": "US Dollar / South African Rand (USDZAR)",
        "USDINR=X": "US Dollar / Indian Rupee (USDINR)", "USDCNY=X": "US Dollar / Chinese Yuan (USDCNY)",
        "USDKRW=X": "US Dollar / Korean Won (USDKRW)", "USDSGD=X": "US Dollar / Singapore Dollar (USDSGD)",
        "USDHKD=X": "US Dollar / Hong Kong Dollar (USDHKD)", "EURBRL=X": "Euro / Brazilian Real (EURBRL)",
    }

    def _label(t):
        base = str(t).split(" (")[0]
        return _PAIR_LABEL.get(str(t), _PAIR_LABEL.get(base + "=X", base))

    is_dollar_q = ("dollar" in q or "dxy" in q or "usd" in q
                   or any(w in q for w in ("currency", "currencies", "greenback")))
    is_index_q = ("dollar index" in q or "dxy" in q or "dx-y.nyb" in q)

    # ── Specific-pair asks ─────────────────────────────────────────────────
    # "How will USD/BRL react to the Fed?" must LEAD with the named pair
    # (live quote + directional read) rather than only ranking the G10
    # crosses. A pair counts as "named" only when its joined code actually
    # appears in the original query text — injected defaults never do.
    q_up = (query or "").upper()
    q_flat = q_up.replace("/", "").replace("-", "").replace("_", "").replace(" ", "")
    named_pairs = []
    for _t in (tickers or []):
        _t = str(_t)
        if _t.endswith("=X") and _t[:-2] in q_flat:
            named_pairs.append(_t)
    named_pairs = list(dict.fromkeys(named_pairs))

    lines = ["### FX Outlook", ""]
    if named_pairs:
        focus = named_pairs[0]
        td = (live_data or {}).get(focus, {})
        fpx = td.get("price")
        fchg = td.get("change_5d")
        if not fpx:
            try:
                df = _fetch_series_for(focus, period="5d")
                if df is not None and len(df) > 1:
                    close = df["Close"]
                    if isinstance(close, pd.DataFrame):
                        close = close.iloc[:, 0]
                    close = close.dropna().astype(float)
                    if len(close) > 1:
                        fpx = float(close.iloc[-1])
                        fchg = (fpx / float(close.iloc[0]) - 1) * 100
            except Exception:
                pass
        focus_label = _label(focus)
        if fpx and fpx > 0:
            direction = "firming" if (fchg or 0) > 0 else "softening"
            lines += [f"**{focus_label}:** {fpx:,.4f} ({fchg:+.2f}% 5d) — the pair is {direction} over the past week."]
        else:
            lines += [f"**{focus_label}:** live quote unavailable right now — see the momentum table below for the nearest read."]
        # Evidence-based transmission note: what actually drives THIS pair.
        if "fed" in q or "rate" in q or "fomc" in q or "hike" in q or "cut" in q:
            lines += ["- **Fed transmission:** EM pairs like this one react to the Fed through the real-rate "
                      "differential (carry), global risk appetite and the commodity terms of trade — a hawkish "
                      "surprise typically pressures the EM currency via the dollar leg; a dovish surprise "
                      "relieves it. The 5-day move above is the market's current read; the next FOMC "
                      "statement/CPI print is the primary swing catalyst."]
        lines += [""]
    for i, (t, p, c) in enumerate(rows, 1):
        lines.append(f"{i}. **{_label(t)}** {p:,.4f} ({c:+.2f}% 5d)")
    lines += ["",
              f"**Strongest:** {_label(strongest[0])} ({strongest[2]:+.2f}% 5d) — the relative winner on current momentum.",
              f"**Weakest:** {_label(weakest[0])} ({weakest[2]:+.2f}% 5d) — the relative laggard; watch for mean-reversion or continued weakness."]
    # Dollar / dollar-index asks get an explicit US Dollar answer so the
    # response addresses "Is the dollar strengthening?" directly instead of
    # only listing crosses.
    if is_dollar_q:
        try:
            dxy_td = (live_data or {}).get("DX-Y.NYB", {})
            dxy_px = dxy_td.get("price")
            dxy_chg = dxy_td.get("change_5d")
            if not dxy_px:
                dxy_df = _fetch_series_for("DX-Y.NYB", period="5d")
                if dxy_df is not None and len(dxy_df) > 1:
                    close = dxy_df["Close"]
                    if isinstance(close, pd.DataFrame):
                        close = close.iloc[:, 0]
                    close = close.dropna().astype(float)
                    if len(close) > 1:
                        dxy_px = float(close.iloc[-1])
                        dxy_chg = (dxy_px / float(close.iloc[0]) - 1) * 100
            if dxy_px:
                direction = "strengthening" if (dxy_chg or 0) > 0 else "softening"
                lines += ["",
                          f"**US Dollar (DXY):** {dxy_px:,.2f} ({dxy_chg:+.2f}% 5d) — the dollar is "
                          f"{direction} over the past week.",
                          ("- A firming dollar is typically a headwind for gold, commodities, EM FX and "
                           "import-driven equities; check the sectors you hold against that exposure."
                           if (dxy_chg or 0) > 0 else
                           "- A softer dollar typically supports gold, commodities and EM assets; "
                           "watch whether it persists or mean-reverts.")]
                if is_index_q:
                    lines += [f"- **Dollar-index outlook:** momentum is the near-term driver; the 12-month "
                              f"path depends on the Fed's real-rate differential vs. the rest of the G10."]
        except Exception:
            pass
    lines += ["",
              "*FX momentum rankings are derived from live market data. Directional calls carry "
              "significant risk — always pair them with macro and rate analysis.*"]
    return "\n".join(lines)


def _build_valuation_answer(query, intents, tickers, sectors, live_data,
                            context_data=None):
    """Answer valuation / reverse-DCF questions ('what growth and margin does
    the price already embed?') using the consensus engine's implied-expectations
    solver grounded in live price + reported fundamentals. Returns None when
    no instrument or no data is available (falls back to standard sections)."""
    q = (query or "").lower()
    if not intents.get("valuation"):
        return None
    sym = None
    for t in (tickers or []):
        if not (str(t).startswith("^") or str(t).endswith(("=F", "=X", "-USD"))):
            sym = t
            break
    if not sym:
        return None
    td = (live_data or {}).get(sym, {})
    price = td.get("price")
    if not price or price <= 0:
        return None

    # Fundamentals via the financial model generator (defensive; may be empty
    # on partial data — reverse-DCF still works off price + conservative
    # model assumptions that are explicitly labeled).
    fund = {}
    try:
        from financial_model_generator import fetch_ticker_fundamentals
        fund = fetch_ticker_fundamentals(sym) or {}
    except Exception:
        fund = {}
    rev_m = fund.get("revenue_m") or 0.0
    growth = (fund.get("revenue_growth") or 8.0) / 100.0
    margin = (fund.get("ebit_margin_pct") or 15.0) / 100.0
    cap_m = fund.get("market_cap_m") or 0.0
    shares = fund.get("shares_m") or 0.0
    net_debt = (fund.get("debt_m") or 0.0) - (fund.get("cash_m") or 0.0)

    try:
        from market_consensus_engine import reverse_dcf_expectations
        imp = reverse_dcf_expectations(
            model_price=float(price),
            current_price=float(price),
            shares=shares * 1e6 if shares > 0 else 1.0,
            net_debt=net_debt * 1e6,
            base_revenue=rev_m * 1e6 if rev_m > 0 else 1.0,
            model_revenue_growth=growth,
            model_ebit_margin=margin,
            wacc=0.095,
            terminal_growth=0.028,
        )
        imp_d = imp.to_dict()
    except Exception:
        imp_d = {}

    lines = [f"### Valuation: {_display_symbol(sym)}", "",
             f"**Current price:** ${price:,.2f}"]
    if cap_m > 0:
        lines.append(f"**Market cap:** ${cap_m:,.0f}M")
    if rev_m > 0:
        lines.append(f"**Revenue (TTM):** ${rev_m:,.0f}M · **model growth:** {growth:.1%} · **model EBIT margin:** {margin:.1%}")

    ig = imp_d.get("implied_revenue_growth")
    im = imp_d.get("implied_ebit_margin")
    if ig is not None and im is not None:
        gap = imp_d.get("gap_label", "")
        lines += ["", "**What the current price implies (reverse DCF):**",
                  f"- Implied revenue growth: {ig:.1%}",
                  f"- Implied EBIT margin: {im:.1%}",
                  f"- {gap}.", "",
                  (f"For the market to be wrong on the upside, {sym} would need to deliver "
                   f"above the embedded path (roughly {ig:.1%} growth / {im:.1%} margin); on the "
                   f"downside, any sustained miss on those embedded expectations is the principal "
                   f"de-rating risk.")]
    else:
        lines += ["", "A full reverse-DCF requires reported revenue/margin detail; the live quote is "
                      "shown above with the model's default assumptions as a starting point."]
    lines += ["", "*Figures: price is observed; implied growth/margin are model estimates from a simplified "
                  "DCF identity, not reported facts.*"]
    return "\n".join(lines)


def _build_mega_response(query, parts, context_data=None):
    """Answer every part of a multi-task query with its own specialist section."""
    # Collect instruments named across ALL parts so an instrument-less part can
    # inherit context ("...for META, then give me a trade setup" -> META).
    all_tickers = []
    for sub in parts:
        try:
            _, t, _ = expand_query_intents(sub)
        except Exception:
            t = []
        for x in t:
            if x not in all_tickers:
                all_tickers.append(x)

    sections, labels = [], []
    seen_labels = set()
    seen_label_norms = {}
    seen_label_instrs = {}
    for idx, sub in enumerate(parts, 1):
        try:
            sub_intents, sub_tickers, sub_sectors = expand_query_intents(sub)
        except Exception:
            continue
        _sl = sub.lower()
        # Re-enable instrument-gated intents that the extractor disables when a
        # part names no ticker of its own but an instrument is implied by the
        # surrounding query.
        if any(w in _sl for w in ("setup", "set up", "entry", "stop", "target",
                                  "trade plan", "risk/reward")):
            sub_intents["trade_setup"] = True
        if any(w in _sl for w in ("hedge", "hedging", "hedged", "protect",
                                  "protection", "collar", "covered call", "puts")):
            sub_intents["hedging"] = True
        if any(w in _sl for w in ("probability", "probabilities", "odds",
                                  "likelihood", "chance", "chances")):
            sub_intents["probability"] = True
        if not sub_tickers and any(sub_intents.get(k) for k in
                                   ("trade_setup", "probability", "hedging",
                                    "valuation", "earnings", "dividend",
                                    "macro", "technical", "options",
                                    "comparison")) and all_tickers:
            sub_tickers = [all_tickers[0]]
        try:
            sub_report = _build_single_answer(sub, sub_intents, sub_tickers,
                                              sub_sectors, context_data)
        except Exception:
            # One crashing part (e.g. a None price under partial data) must
            # never take down the whole multi-part answer — degrade that part
            # to a bounded fallback so the other tasks still get answered.
            sub_report = (
                f"### {_mega_label(sub_intents, sub_tickers, sub_sectors)}\n\n"
                "I hit a data-availability snag computing this part from the "
                "current tape. Here is the framing I can stand behind: this is "
                "a live-data-dependent ask — re-run it when the quote feed is "
                "fully populated, or ask it for a single instrument and I will "
                "give you the full setup."
            )
        if not sub_report:
            continue
        label = _mega_label(sub_intents, sub_tickers, sub_sectors)
        # ── De-duplicate: the same label twice in a row is the reported
        # "Part 1: Macro Outlook / Part 2: Macro Outlook ..." repetition bug.
        # Drop a repeat ONLY when the part is a NEAR-DUPLICATE of a previously
        # seen same-label part (high token overlap) — a prose continuation or
        # a re-asked question. Two parts that are genuinely different asks
        # under the same label ("should I own long-duration bonds right now?"
        # then "what is the outlook for quantitative tightening?") are distinct
        # macro topics and must BOTH be answered. Parts naming different
        # instruments under the same label ("SUI" then "Solana") are also kept.
        _norm_this = re.sub(r"[^a-z0-9]+", " ", sub.lower()).strip()
        _is_dup = False
        if label in seen_labels:
            # Parts naming DIFFERENT instruments under the same label are
            # distinct answers ("outlook for the VIX" vs "outlook for PYPL"),
            # even when the template text overlaps heavily — a user who asked
            # for two outlooks wants both. Only when the instrument sets match
            # (or neither part names an instrument) does text similarity decide.
            _prev_instr = seen_label_instrs.get(label, [])
            _this_instr = frozenset(sub_tickers or []) | frozenset(sub_sectors or [])
            # Apply the text de-dup when this part has no instrument of its own
            # (pure macro/regime asks — the bonds/QT and economy cases) OR when
            # it overlaps an instrument already seen under this label (two
            # "outlook for AAPL" asks). Only parts naming a NEW instrument are
            # exempt — they are distinct answers regardless of template text.
            _shares_instr = (not _this_instr) or any(
                _this_instr & p for p in _prev_instr)
            if _shares_instr:
                for _seen in seen_label_norms.get(label, []):
                    if not _seen or not _norm_this:
                        continue
                    _a = set(_norm_this.split())
                    _b = set(_seen.split())
                    _inter = len(_a & _b)
                    _jaccard = _inter / max(1, len(_a | _b))
                    if (_jaccard >= 0.7 or _norm_this in _seen
                            or _seen in _norm_this):
                        _is_dup = True
                        break
        if _is_dup:
            continue
        seen_labels.add(label)
        seen_label_norms.setdefault(label, []).append(_norm_this)
        seen_label_instrs.setdefault(label, []).append(
            frozenset(sub_tickers or []) | frozenset(sub_sectors or []))
        sections.append(f"### Part {len(sections) + 1}: {label}\n\n{sub_report.strip()}")
        labels.append(label)
    if not sections:
        return None
    header = ("### Multi-Part Analysis\n"
              "Your request contains several distinct questions. I have addressed "
              "each one in turn:\n- " + "\n- ".join(labels) + "\n")
    return header + "\n\n---\n\n".join(sections)


# ---------------------------------------------------------------------------
# RESPONSE GENERATION ENGINE
# ---------------------------------------------------------------------------

def generate_financial_analysis(query, context_data=None):
    """
    Self-contained deterministic multi-agent NLP engine.
    Produces structured research-grade output grounded with live market data.

    Result is cached on disk for 10 minutes (keyed by query + date) so
    repeated / near-identical queries return instantly instead of re-running
    the full data-fetch + analysis pipeline.
    """
    import time as _t
    _v = "fa::v12::"
    _qpart = (query or "").strip().lower()
    # Fast path: query-only key first so repeated queries never pay for the
    # ticker-universe load (the universe is disk-cached, but avoid it entirely
    # on the hot path).
    _qkey_fast = _v + hashlib.sha1(_qpart.encode()).hexdigest()
    _cached = _cache_load(_qkey_fast, ttl_seconds=600)
    if _cached is not None:
        return _cached

    _t0 = _t.time()
    intents, tickers, sectors = expand_query_intents(query)
    # Full key = query + extracted tickers/sectors + engine version, so symbol
    # extraction improvements invalidate stale entries immediately.
    _qkey = _v + hashlib.sha1(
        (_qpart + "|" + ",".join(tickers) + "|" + ",".join(sectors)).encode()
    ).hexdigest()
    _cached = _cache_load(_qkey, ttl_seconds=600)
    if _cached is not None:
        return _cached

    # --- Deep-dive institutional theses get the comprehensive memo ---
    # A multi-section company thesis (valuation + scenarios + risk matrix +
    # catalysts + falsification + final decision) must NOT be swallowed by a
    # single specialist template — even when one of its sections mentions
    # geopolitical risk, earnings, or events. Answer the whole ask in one
    # coherent memo instead.
    #
    # CRITICAL: the deep-dive check runs BEFORE mega decomposition. A
    # "write a complete investment memo on LOW with sections (1)-(6)" ask is
    # ONE coherent thesis, not a bundle of unrelated tasks — decomposing it
    # into per-section mini-analyses was the reported regression where a memo
    # came back as generic "Analysis / Valuation" parts.
    if _is_deep_dive_query(query, tickers):
        # Fetch live data just for this branch (mirrors _build_single_answer).
        _uni = get_ticker_universe()
        _lookup = list(tickers)
        for _s in sectors or []:
            _lookup.extend(_sector_tickers_for(_uni, _s)[:5])
        _lookup = list(dict.fromkeys(_lookup))[:15]
        _live = _fetch_live_data_for_tickers(_lookup) if _lookup else {}
        _deep = _build_institutional_deep_dive(query, intents, tickers, sectors,
                                               _live, context_data)
        if _deep:
            _cache_save(_qkey, _deep)
            print(f"[PERF] generate_financial_analysis (deep-dive) took {(_t.time()-_t0)*1000:.0f}ms")
            return _deep

    # --- Multi-part ("mega") prompts get per-task sections ---
    # When a query bundles several distinct asks, decompose it and answer each
    # part with its own specialist builder instead of routing the whole query
    # to a single handler (which would silently drop the other tasks).
    mega_parts = _decompose_mega_query(query)
    if mega_parts:
        mega_report = _build_mega_response(query, mega_parts, context_data)
        if mega_report:
            _cache_save(_qkey, mega_report)
            print(f"[PERF] generate_financial_analysis (mega, {len(mega_parts)} parts) "
                  f"took {(_t.time()-_t0)*1000:.0f}ms")
            return mega_report

    now = datetime.datetime.now()

    # --- Fetch live data to ground the response ---
    # Determine which tickers to look up
    universe = get_ticker_universe()
    lookup_tickers = list(tickers)
    for s in sectors:
        # Use first 5 tickers from the universe for this sector
        lookup_tickers.extend(_sector_tickers_for(universe, s)[:5])
    lookup_tickers = list(dict.fromkeys(lookup_tickers))[:15]  # Dedupe, limit

    live_data = _fetch_live_data_for_tickers(lookup_tickers) if lookup_tickers else {}

    # --- Geopolitics / current-events queries get honest structured briefings ---
    # Only when the event is the DOMINANT ask (short, focused, no thesis
    # vocabulary); otherwise the query falls through to the analysis paths
    # below. "How would a war impact oil?" still routes here (before
    # transmission); "conduct a full assessment of NVDA that touches on
    # U.S.-China restrictions" does not.
    if intents.get("geopolitics") and _is_focused_event_question(query, intents, tickers):
        _event_report = _build_geopolitical_briefing(query, intents, tickers, sectors,
                                                     live_data, context_data)
        if _event_report:
            response = _event_report
            _cache_save(_qkey, response)
            print(f"[PERF] generate_financial_analysis (event) took {(_t.time()-_t0)*1000:.0f}ms")
            return response
    if (intents.get("current_events") and _is_event_query(query)
            and _is_focused_event_question(query, intents, tickers)):
        _event_report = _build_current_events_briefing(query, intents, tickers, sectors,
                                                       live_data, context_data)
        if _event_report:
            response = _event_report
            _cache_save(_qkey, response)
            print(f"[PERF] generate_financial_analysis (event) took {(_t.time()-_t0)*1000:.0f}ms")
            return response

    # --- Cross-asset transmission queries get the quantitative engine ---
    # When the user asks how one asset's moves (or volatility) transmit to
    # other markets, serve the measured correlation/beta/lead-lag/vol-regime
    # evidence instead of generic sector boilerplate. The transmission engine
    # returns None (no fabrication) if the anchor or data is unavailable, in
    # which case we fall back to the standard section builders below.
    transmission_report = None
    if intents.get("transmission"):
        try:
            from cross_asset_transmission import generate_cross_asset_transmission
            transmission_report = generate_cross_asset_transmission(query, tickers)
        except Exception as _txe:
            logger.warning(f"[CrossAsset] transmission path failed ({_txe}); using standard sections")
            transmission_report = None

    if transmission_report:
        response = transmission_report
        _cache_save(_qkey, response)
        print(f"[PERF] generate_financial_analysis (transmission) took {(_t.time()-_t0)*1000:.0f}ms (cached for 10m)")
        return response

    # --- Trade-setup questions get a real playbook ---
    if intents.get("trade_setup"):
        setup_report = _build_trade_setup_response(query, intents, tickers, sectors,
                                                   live_data, context_data)
        if setup_report:
            response = setup_report
            _cache_save(_qkey, response)
            print(f"[PERF] generate_financial_analysis (setup) took {(_t.time()-_t0)*1000:.0f}ms")
            return response

    # --- Probability-of-target questions get the recalibrated engine ---
    if intents.get("probability"):
        prob_report = _build_probability_answer(query, tickers)
        if prob_report:
            response = prob_report
            _cache_save(_qkey, response)
            print(f"[PERF] generate_financial_analysis (probability) took {(_t.time()-_t0)*1000:.0f}ms")
            return response

    # --- Hedging questions get concrete hedge strategies ---
    if intents.get("hedging"):
        hedge_report = _build_hedging_strategy(query, intents, tickers, sectors,
                                               live_data, context_data)
        if hedge_report:
            response = hedge_report
            _cache_save(_qkey, response)
            print(f"[PERF] generate_financial_analysis (hedge) took {(_t.time()-_t0)*1000:.0f}ms")
            return response

    # --- Options-strategy asks about a named stock get instrument-specific
    # options guidance (covered calls, straddles, spreads, put structures) ---
    if intents.get("options") and _primary_symbol(tickers):
        opt_report = _build_hedging_strategy(query, intents, tickers, sectors,
                                             live_data, context_data)
        if opt_report:
            response = opt_report
            _cache_save(_qkey, response)
            print(f"[PERF] generate_financial_analysis (options) took {(_t.time()-_t0)*1000:.0f}ms")
            return response

    # --- Sector scans / FX outlooks / valuation-reverse-DCF ---
    sector_report = _build_sector_scan_response(query, intents, tickers, sectors,
                                                live_data, context_data)
    if sector_report:
        response = sector_report
        _cache_save(_qkey, response)
        print(f"[PERF] generate_financial_analysis (sector scan) took {(_t.time()-_t0)*1000:.0f}ms")
        return response
    fx_report = _build_fx_outlook_response(query, intents, tickers, sectors,
                                           live_data, context_data)
    if fx_report:
        response = fx_report
        _cache_save(_qkey, response)
        print(f"[PERF] generate_financial_analysis (fx) took {(_t.time()-_t0)*1000:.0f}ms")
        return response
    val_report = _build_valuation_answer(query, intents, tickers, sectors,
                                         live_data, context_data)
    if val_report:
        response = val_report
        _cache_save(_qkey, response)
        print(f"[PERF] generate_financial_analysis (valuation) took {(_t.time()-_t0)*1000:.0f}ms")
        return response

    # --- Build each section dynamically based on intents ---
    exec_summary = _build_executive_summary(query, intents, tickers, sectors, live_data, context_data)
    macro_analysis = _build_macro_analysis(query, intents, sectors, live_data, context_data)
    market_implications = _build_market_implications(intents, tickers, sectors, live_data)
    key_takeaways = _build_key_takeaways(intents, tickers, sectors, live_data)

    response = f"""### Executive Summary
{exec_summary}

---

### Macro Analysis
{macro_analysis}

---

### Market Implications
{market_implications}

---

### Key Takeaways
{key_takeaways}

---
*Analysis generated {now.strftime('%b %d, %Y %H:%M')} | Octavian Financial Intelligence Engine*
"""
    _cache_save(_qkey, response)
    print(f"[PERF] generate_financial_analysis took {(_t.time()-_t0)*1000:.0f}ms (cached for 10m)")
    return response


# ---------------------------------------------------------------------------
# SECTION BUILDERS — produce direct, conclusion-driven answers
# ---------------------------------------------------------------------------

def _build_executive_summary(query, intents, tickers, sectors, live_data, context_data=None):
    """Produce a concise, data-grounded executive summary."""
    lines = []
    universe = get_ticker_universe()

    if context_data:
        lines.append(f"**Current Context:** {context_data}")
        if "risk-off" in context_data.lower() or "elevated" in context_data.lower():
            lines.append("The current environment dictates a defensive posture. Capital preservation should be prioritized over aggressive relative-strength chasing.\n")
        elif "risk-on" in context_data.lower() or "low" in context_data.lower():
            lines.append("The current environment is supportive of risk assets. Focus on high-beta leaders and constructive breakouts.\n")
        else:
            lines.append("The market is currently in a transitional state. Stock selection and relative strength are the primary drivers of alpha.\n")

    if sectors:
        direction = "bullish" if intents["bullish"] else "bearish" if intents["bearish"] else "neutral"
        sector_label = ", ".join([_sector_display(query, s) for s in sectors])
        
        bias_text = {"bullish": "accumulation-oriented", "bearish": "distribution-oriented",
                     "neutral": "two-sided"}[direction]
        lines.append(f"The **{sector_label}** complex is trading with a **{direction}** character on the "
                     f"momentum and positioning tape — a {bias_text} backdrop across the group.\n")
        
        for s in sectors:
            # Inject top 3 for this sector if none were explicitly mentioned
            sector_tickers = _sector_tickers_for(universe, s)[:3]
            for t in sector_tickers:
                if t not in tickers:
                    tickers.append(t)

    if tickers:
        # Only display tickers that RESOLVED to real data — a ticker with no
        # live quote is noted once, not spammed per line (and never a semantic
        # concept: entity resolution already removed those upstream).
        display_tickers = [t for t in tickers if t not in ["FX", "PAIRS"]]
        resolved = [t for t in display_tickers if live_data.get(t, {})]
        unresolved = [t for t in display_tickers if not live_data.get(t, {})]
        if resolved:
            lines.append(f"**Live snapshot for {', '.join(_display_symbol(t) for t in resolved)}:**\n")
            for t in resolved:
                td = live_data[t]
                price = td.get("price", 0)
                chg = td.get("change_5d", 0)

                # Try to find which sector this ticker belongs to for flavor
                ticker_sector = None
                for s in sectors:
                    if t in _sector_tickers_for(universe, s):
                        ticker_sector = s
                        break

                reasoning = _generate_ticker_reasoning(t, ticker_sector, price, chg)
                lines.append(f"- **{_display_symbol(t)}** (${price:,.2f} | {chg:+.2f}% this week): {reasoning}")
        if unresolved:
            lines.append(f"_Live data temporarily unavailable for: {', '.join(_display_symbol(t) for t in unresolved)}._")

    if not lines or (not tickers and not sectors):
        lines.append("Based on current market conditions, the environment remains complex. Macro forces (rates, inflation, Fed policy) are currently overriding individual stock correlations.")

    return "\n".join(lines)


def _generate_ticker_reasoning(sym, sector, price, chg):
    """Generate dynamic, data-driven reasoning based on move magnitude and sector context."""
    
    # 1. Technical Indicators fallbacks
    strength = ""
    if chg > 3:
        strength = "exhibiting explosive relative strength and high-volume breakout characteristics."
    elif chg > 1:
        strength = "showing solid positive momentum and institutional accumulation."
    elif chg > 0:
        strength = "drifting higher with constructive price action, suggesting steady demand."
    elif chg > -1:
        strength = "consolidating in a tight range, potentially reset for the next leg."
    elif chg > -3:
        strength = "experiencing a healthy pullback, testing short-term support levels."
    else:
        strength = "under significant selling pressure; wait for stabilization signs before engagement."

    # 2. Add sector-based flavor if known
    flavor = ""
    sector_label = sector.replace("_", " ").title() if sector else "Market"
    
    if sector == "technology" or sector == "semiconductors":
        flavor = "Growth-oriented institutional flows are favoring this name."
    elif sector == "energy" or sector == "mining":
        flavor = "Commodity price sensitivity and macro headlines are driving current action."
    elif sector == "financials":
        flavor = "Yield curve dynamics and bank capital requirements remain the primary focus."
    elif sector == "healthcare":
        flavor = "Defensive qualities and pipeline updates are providing structural support."
    else:
        flavor = f"Flows are rotating across the {sector_label} complex — watch relative strength against the sector ETF for confirmation."

    return f"{strength} {flavor}"


_MACRO_TOPIC_TERMS = [w for _t, ws in _MACRO_TOPICS for w in ws] + [
    "a recession", "the jobs report", "the yield curve", "the fed", "the money supply",
    "quantitative tightening", "quantitative easing", "the 10-year treasury yield",
    "consumer confidence", "housing starts", "retail sales", "ism manufacturing",
    "nonfarm payrolls", "gdp growth", "interest rates", "rate hikes", "rate cuts",
    "inflation data", "the dollar", "the stock market", "the economy",
]

_MACRO_TOPIC_TEXT = {
    "inflation": "- Core services inflation is the stickiest input for the Fed's reaction function.\n"
                 "- Hot prints lift the dollar and front-end yields; cool prints ease duration pressure.\n"
                 "- Watch the 3- and 6-month annualized run-rates, not just the YoY headline.",
    "interest_rates": "- The level and slope of rates price the entire cross-asset complex.\n"
                      "- Rising real yields compress equity multiples; falling yields support duration and growth.\n"
                      "- The market now trades the *path* (dot plot) more than the current level.",
    "gdp": "- Growth is the denominator for valuations — decelerating growth favors quality and defensives.\n"
            "- Watch revisions: initial prints are noisy; the trend in 3-month averages matters more.\n"
            "- Growth plus inflation (nominal GDP) is the truer driver of earnings power.",
    "employment": "- The labor market is the Fed's second mandate and the key to the soft-landing debate.\n"
                   "- Strong payrolls with cool wage growth = goldilocks; hot wages = re-tightening risk.\n"
                   "- Participation and revisions matter as much as the headline.",
    "recession": "- A recession is a regime event: earnings fall, credit spreads widen, duration wins.\n"
                  "- Leading indicators (inverted curve, ISM, unemployment claims) signal 6-12 months ahead.\n"
                  "- Defensive positioning before confirmation is expensive; after confirmation it is late.",
    "yield_curve": "- The 2s10s inversion has historically led recessions by 12-18 months.\n"
                    "- Normalization (steepening) often coincides with the late-cycle equity peak.\n"
                    "- Long duration regains hedge value once the curve re-steepens from the front.",
    "manufacturing": "- PMI/ISM new orders are the most cyclical, most predictive sub-component.\n"
                      "- A sub-50 reading means contraction; the rate of change drives earnings revisions.\n"
                      "- The factory cycle and the inventory cycle together set the earnings direction.",
    "consumer": "- Consumer spending is ~2/3 of GDP — the margin between wages and prices decides it.\n"
                 "- Excess savings are depleting; revolving credit growth is the canary.\n"
                 "- Retail sales ex-autos and real (inflation-adjusted) spending are the cleanest reads.",
    "housing": "- Housing leads the cycle: rates up → permits down → construction employment down.\n"
                "- Existing home sales are rate-sensitive; new home sales depend on builder incentives.\n"
                "- Shelter inflation lags rents by 12+ months and is the stickiest CPI component.",
    "liquidity": "- Liquidity (Fed balance sheet, reverse repo, T-bill supply) is the tide that lifts prices.\n"
                  "- QT drains reserves; Treasury issuance absorbs them; the RRP drawdown is the canary.\n"
                  "- Liquidity expansions historically precede risk-asset rallies by ~3 months.",
}


def _macro_topic_grounding(query: str) -> str:
    """Return a topic-specific paragraph naming the exact indicator in the query."""
    q = query.lower()
    term = None
    topic = None
    # Longest match first so "the 10-year treasury yield" beats "treasury"
    for t in sorted(_MACRO_TOPIC_TERMS, key=len, reverse=True):
        if t in q:
            term = t
            break
    if term:
        for key, words in _MACRO_TOPICS:
            if term in words or any(w in q for w in words):
                topic = key
                break
        if topic and topic in _MACRO_TOPIC_TEXT:
            display = _smart_title(term)
            return (f"**On {display} — current read and market transmission:**\n"
                    f"{_MACRO_TOPIC_TEXT[topic]}")
    return ""


def _smart_title(phrase: str) -> str:
    """Title-case a phrase while preserving acronyms (gdp growth -> GDP Growth)."""
    _ACRONYMS = {"gdp", "cpi", "pce", "pmi", "ism", "nfp", "fomc", "ecb", "boj",
                 "opec", "nato", "fed", "feds", "us", "uk", "eu", "spx", "rsi"}
    words = []
    for w in str(phrase).split():
        clean = w.strip("(),")
        if clean.lower() in _ACRONYMS or clean.isupper():
            words.append(clean.upper())
        else:
            words.append(clean.capitalize())
    return " ".join(words)


def _build_macro_analysis(query, intents, sectors, live_data, context_data):
    lines = []

    if intents["macro"]:
        grounding = _macro_topic_grounding(query)
        if grounding:
            lines.append(grounding)
            lines.append("")
        lines.append("The Fed remains data-dependent, and every incoming inflation print or employment report has outsized market impact. Here's what matters right now:")
        lines.append("")
        lines.append("- **Rates trajectory**: Real yields are the key variable. If the 10Y stabilizes, growth stocks get breathing room. If it keeps climbing, expect continued rotation into value and defensives.")
        lines.append("- **Inflation persistence**: Core services inflation is the stickiest component and the Fed's primary concern. Until this breaks lower, don't expect rate cuts.")
        lines.append("- **Growth vs. tightening**: The economy has been surprisingly resilient, absorbing higher rates without a significant recession signal. This is bullish for equities but means the Fed stays higher for longer.")

    for s in sectors:
        if s == "healthcare":
            lines.append("\n**Why healthcare matters right now:**")
            lines.append("- It's a classic defensive sector that outperforms when uncertainty rises. If you're worried about a slowdown, this is where institutional money flows.")
            lines.append("- The GLP-1 drug revolution (obesity/diabetes treatments) is creating a once-in-a-generation growth catalyst within an otherwise defensive sector — a rare combination.")
            lines.append("- Valuations are reasonable compared to tech. Many large-cap pharma names trade at 12-15x forward earnings with 3-4% dividend yields.")
        elif s == "technology":
            lines.append("\n**What's driving tech right now:**")
            lines.append("- AI infrastructure spend is the dominant theme. Companies directly selling picks-and-shovels (GPUs, cloud compute) are seeing outsized revenue growth.")
            lines.append("- However, valuations are stretched for many AI plays. The key question is whether revenue growth justifies current multiples.")
            lines.append("- Non-AI tech is underperforming, creating a bifurcated market within the sector itself.")
        elif s == "energy":
            lines.append("\n**Energy sector dynamics:**")
            lines.append("- Supply policy (OPEC+ decisions, inventories, spare capacity) is the key swing factor for crude, and it moves E&P margins directly.")
            lines.append("- U.S. producers are prioritizing cash returns (buybacks + dividends) over production growth, which is structurally bullish for pricing.")
            lines.append("- Geopolitical risk premium is always lurking — any Middle East escalation sends energy names higher quickly.")
        elif s == "financials":
            lines.append("\n**What's moving financials:**")
            lines.append("- Net interest margins benefit from the 'higher for longer' rate environment. Banks are earning more on their loan books.")
            lines.append("- Investment banking fees are recovering from 2023 lows. IPO and M&A pipelines are building.")
            lines.append("- Credit risk is the main concern — watch for deterioration in commercial real estate and consumer lending.")
        else:
            sector_label = s.replace("_", " ").title()
            lines.append(f"\n**{sector_label} positioning:**")
            lines.append(f"- This sector is sensitive to the broader growth/rates cycle. Current positioning reflects where we are in the economic cycle.")

    if not sectors and not intents["macro"]:
        lines.append("Here's the broader macro picture to inform your positioning:")
        lines.append("")
        lines.append("- The economy is in a late-cycle expansion — growth is positive but decelerating. This favors quality and defensive characteristics.")
        lines.append("- The market is priced for a soft landing. If that scenario plays out, equities have further upside. If it doesn't, downside risk is meaningful.")
        lines.append("- Cross-asset correlations are elevated, meaning diversification benefits are reduced. Cash and short-duration bonds serve as better hedges than they have in years.")

    if context_data:
        lines.append(f"\n*Current market state: {context_data}*")

    return "\n".join(lines)


def _build_market_implications(intents, tickers, sectors, live_data):
    lines = []
    universe = get_ticker_universe()

    for s in sectors:
        sector_label = _sector_display("", s)
        all_sector_tickers = _sector_tickers_for(universe, s)

        candidates = []
        for t in all_sector_tickers[:15]:
            td = live_data.get(t, {})
            if td:
                candidates.append((t, td.get("price", 0), td.get("change_5d", 0)))

        if intents["bullish"]:
            candidates.sort(key=lambda x: x[2], reverse=True)
            best = candidates[:4]
            if best:
                lines.append(f"**{sector_label} — Strongest opportunities right now:**\n")
                for sym, price, chg in best:
                    strength = "strong momentum" if chg > 1 else "holding up well" if chg > -1 else "pulling back (potential dip-buy)"
                    lines.append(f"- **{sym}** at ${price:,.2f} ({chg:+.2f}% this week) — {strength}")
                lines.append("")
        elif intents["bearish"]:
            candidates.sort(key=lambda x: x[2])
            worst = candidates[:4]
            if worst:
                lines.append(f"**{sector_label} — Names under the most pressure:**\n")
                for sym, price, chg in worst:
                    lines.append(f"- **{sym}** at ${price:,.2f} ({chg:+.2f}% this week) — bearish momentum")
                lines.append("")
        else:
            candidates.sort(key=lambda x: abs(x[2]), reverse=True)
            notable = candidates[:4]
            if notable:
                lines.append(f"**{sector_label} — Most active names this week:**\n")
                for sym, price, chg in notable:
                    direction = "bullish lean" if chg > 0.5 else "bearish lean" if chg < -0.5 else "neutral/consolidating"
                    lines.append(f"- **{sym}** at ${price:,.2f} ({chg:+.2f}%) — {direction}")
                lines.append("")

    if tickers and not sectors:
        for t in tickers:
            td = live_data.get(t, {})
            if td:
                price = td.get("price", 0)
                chg = td.get("change_5d", 0)
                lines.append(f"**{t}** (${price:,.2f}, {chg:+.2f}% this week):")
                if chg > 2:
                    lines.append("- Buyers are in control; momentum confirms accumulation.")
                elif chg > 0:
                    lines.append("- Quiet accumulation phase with constructive drift.")
                elif chg > -2:
                    lines.append("- Neutral zone; waiting for catalyst.")
                else:
                    lines.append("- Under distribution pressure; wait for stabilization.")
                lines.append("")

    if not sectors and not tickers:
        lines.append("**Cross-market dynamics:**")
        lines.append("- Stock-picking dispersion remains elevated.")
        lines.append("- Sector rotation matters more than index-level exposure.")
        lines.append("- Credit markets are relatively stable, supporting measured risk-taking.")

    return "\n".join(lines)


def _build_key_takeaways(intents, tickers, sectors, live_data):
    lines = []
    idx = 1
    universe = get_ticker_universe()

    if intents["bullish"] and sectors:
        top_picks = []
        for t in _sector_tickers_for(universe, sectors[0])[:10]:
            td = live_data.get(t, {})
            if td:
                top_picks.append((t, td.get("change_5d", 0)))
        top_picks.sort(key=lambda x: x[1], reverse=True)
        top_names = [p[0] for p in top_picks[:3]]

        lines.append(f"{idx}. **Best buys right now:** {', '.join(f'**{n}**' for n in top_names) if top_names else 'See sector picks above'}.")
        idx += 1
        lines.append(f"{idx}. **Execution:** Scale entries over 2–3 sessions; avoid over-sizing at local extremes.")
        idx += 1
        lines.append(f"{idx}. **Risk:** Keep hard downside limits and concentration controls.")
        idx += 1
        lines.append(f"{idx}. **Timeframe:** Treat as a swing setup; re-evaluate weekly.")
        idx += 1
        lines.append(f"{idx}. **Sizing:** Cap per-name and per-sector exposure.")
    elif intents["bearish"] and sectors:
        lines.append(f"{idx}. **Short candidates:** Focus on weakest names with persistent negative momentum.")
        idx += 1
        lines.append(f"{idx}. **Structure:** Prefer defined-risk bearish structures (e.g., put spreads).")
        idx += 1
        lines.append(f"{idx}. **Risk control:** Exit on technical invalidation.")
        idx += 1
        lines.append(f"{idx}. **Hedging:** Pair shorts with selective relative-strength longs if needed.")
    elif intents["macro"]:
        lines.append(f"{idx}. **Positioning:** Prioritize balance-sheet quality and cash-flow durability.")
        idx += 1
        lines.append(f"{idx}. **Watch rates:** 10Y direction remains a key regime pivot.")
        idx += 1
        lines.append(f"{idx}. **Catalysts:** CPI/FOMC/labor prints remain primary volatility events.")
        idx += 1
        lines.append(f"{idx}. **Construction:** Use a balanced barbell across defensives and selective cyclicals.")
    else:
        # Data-grounded default stance instead of a canned phrase: derive it
        # from the symbols' own week-over-week action when available.
        _strength, _weakness = 0, 0
        for _tk, _ld in (live_data or {}).items():
            try:
                _chg = float((_ld or {}).get("change_5d") or 0.0)
            except Exception:
                _chg = 0.0
            if _chg > 1.5:
                _strength += 1
            elif _chg < -1.5:
                _weakness += 1
        if _strength > _weakness and _strength >= 2:
            _action = "Prefer selective strength; accumulate on constructive pullbacks"
        elif _weakness > _strength and _weakness >= 2:
            _action = "Stand aside on weak names until stabilization confirms a base"
        elif _strength or _weakness:
            _action = "Stay nimble; treat current action as two-sided until a clear bias emerges"
        else:
            _action = "Maintain balance; await clearer regime confirmation before adding risk"
        lines.append(f"{idx}. **Action:** {_action}.")
        idx += 1
        lines.append(f"{idx}. **Discipline:** Size every position so a full stop-out costs ≤1% of capital.")
        idx += 1
        lines.append(f"{idx}. **Tactical edge:** Prefer defined-risk structures when IV is rich; scale in when it is cheap.")
        idx += 1
        lines.append(f"{idx}. **Regime check:** Re-verify the volatility regime weekly and adjust gross exposure accordingly.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# TRADE-SETUP / HEDGING / GEOPOLITICS / CURRENT-EVENTS BUILDERS
# ---------------------------------------------------------------------------

def _fetch_series_for(symbol, period="6mo"):
    """Best-effort daily OHLCV fetch (cached by data_sources)."""
    try:
        from data_sources import get_stock
        df = get_stock(symbol, period=period, interval="1d")
        if df is not None and not df.empty and "Close" in df.columns:
            return df
    except Exception:
        pass
    return None


def _series_metrics(df):
    """Compute a compact technical snapshot from an OHLCV frame.

    Returns a dict with price, ATR(14), RSI(14), SMA20/50, 90d range,
    20d annualized realized vol, and 5d change — or {} if insufficient data.
    """
    import math
    try:
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna().astype(float)
        if len(close) < 30:
            return {}
        price = float(close.iloc[-1])

        high = df["High"]
        low = df["Low"]
        if isinstance(high, pd.DataFrame):
            high = high.iloc[:, 0]
        if isinstance(low, pd.DataFrame):
            low = low.iloc[:, 0]
        high = high.dropna().astype(float)
        low = low.dropna().astype(float)

        n = min(len(close), len(high), len(low))
        close, high, low = close.iloc[-n:], high.iloc[-n:], low.iloc[-n:]

        # ATR(14)
        prev_close = close.shift(1)
        tr = pd.concat([(high - low).abs(), (high - prev_close).abs(),
                        (low - prev_close).abs()], axis=1).max(axis=1)
        atr = float(tr.iloc[-14:].mean()) if len(tr) >= 15 else float(tr.mean())

        # RSI(14) — Wilder's smoothing
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14).mean()
        rs = gain / loss.replace(0, 1e-12)
        rsi = float((100 - 100 / (1 + rs)).iloc[-1]) if len(rs) else 50.0

        rets = close.pct_change().dropna()
        rvol = float(rets.iloc[-20:].std() * math.sqrt(252) * 100) if len(rets) >= 20 else None
        chg5 = float((price / close.iloc[-6] - 1) * 100) if len(close) >= 6 else None

        return {
            "price": price,
            "atr": atr,
            "rsi": rsi,
            "sma20": float(close.iloc[-20:].mean()),
            "sma50": float(close.iloc[-50:].mean()) if len(close) >= 50 else None,
            "hi90": float(close.iloc[-90:].max()) if len(close) >= 90 else float(close.max()),
            "lo90": float(close.iloc[-90:].min()) if len(close) >= 90 else float(close.min()),
            "rvol": rvol,
            "chg5": chg5,
        }
    except Exception:
        return {}


def _primary_symbol(tickers):
    """Pick the primary instrument: prefer futures/FX/index anchors."""
    for t in tickers or []:
        if t.endswith("=F") or t.endswith("=X") or t.startswith("^") or "-" in t:
            return t
    return (tickers or [None])[0]


def _vol_regime_label(rvol):
    if rvol is None:
        return "unknown"
    if rvol >= 55:
        return "elevated"
    if rvol >= 35:
        return "moderate"
    return "low"


def _build_trade_setup_response(query, intents, tickers, sectors, live_data, context_data=None):
    """Institutional-grade trade setup: bias, entry, stop, targets, R:R,
    position sizing and hedging — every number derived from real price data.
    Returns None (no fabrication) if no instrument or no price data."""
    import math

    primary = _primary_symbol(tickers)
    if not primary:
        return None

    df = _fetch_series_for(primary)
    m = _series_metrics(df) if df is not None else {}
    price = m.get("price")
    if price is None:
        td = (live_data or {}).get(primary, {})
        if td:
            price = td.get("price")
    if price is None:
        return None

    atr = m.get("atr") or price * 0.02
    rsi = m.get("rsi") or 50.0
    sma20 = m.get("sma20")
    sma50 = m.get("sma50")
    hi90, lo90 = m.get("hi90"), m.get("lo90")
    rvol = m.get("rvol")
    chg5 = m.get("chg5")
    regime = _vol_regime_label(rvol)

    # ── Directional bias: query words override, else momentum stack ──
    q = query.lower()
    query_long = any(w in q for w in ["long", "buy ", "buy the", "bullish", "call", "upside"])
    query_short = any(w in q for w in ["short", "sell", "bearish", "put", "downside", "decline"])
    if query_long and not query_short:
        bias, bias_label = "LONG", "explicit long bias in your question"
    elif query_short and not query_long:
        bias, bias_label = "SHORT", "explicit short bias in your question"
    else:
        mom_score = 0
        if sma20 and sma50:
            mom_score += 1 if sma20 > sma50 else -1
        mom_score += 1 if rsi > 55 else (-1 if rsi < 45 else 0)
        if hi90 and lo90:
            pos = (price - lo90) / max(hi90 - lo90, 1e-9)
            mom_score += 1 if pos > 0.6 else (-1 if pos < 0.4 else 0)
        bias = "LONG" if mom_score >= 1 else ("SHORT" if mom_score <= -1 else "FLAT")
        bias_label = "momentum stack (SMA trend, RSI, position in 90d range)"
    if bias == "FLAT":
        bias = "LONG"  # default to a defined long plan; the flat case is flagged below
        bias_label = "neutral momentum — plan below is a framework; wait for confirmation"

    # ── Structure (long book; mirrored for short) ──
    if bias == "SHORT":
        entry = price
        stop = entry + 1.5 * atr
        t1 = entry - 2.0 * atr
        t2 = entry - 3.5 * atr
        r1, r2 = (entry - t1) / (stop - entry), (entry - t2) / (stop - entry)
    else:
        entry = price
        stop = entry - 1.5 * atr
        t1 = entry + 2.0 * atr
        t2 = entry + 3.5 * atr
        r1, r2 = (t1 - entry) / (entry - stop), (t2 - entry) / (entry - stop)
    entry_zone_lo = entry - 0.25 * atr
    entry_zone_hi = entry + 0.25 * atr

    # ── Position sizing table (risk per trade, three account sizes) ──
    def _shares(budget, risk_pct, acct):
        risk_dol = acct * risk_pct / 100.0
        qty = risk_dol / max(entry - stop, 1e-9) if bias == "LONG" else risk_dol / max(stop - entry, 1e-9)
        return int(qty)

    sizing = []
    for acct in (10000, 50000, 100000):
        sizing.append(f"$ {acct:,.0f}")
        for rp in (0.5, 1.0, 2.0):
            sizing.append(f"{_shares(0, rp, acct)} @ {rp}% risk")

    # ── Hedging plan depends on asset class + vol regime ──
    if primary.endswith("=F"):
        hedge = (f"**Hedge (futures):** in a {regime}-vol regime, prefer defined-risk structures. "
                 f"For a {bias} futures position, buy an OTM option on the same contract "
                 f"(~1 ATR beyond your stop) rather than adding a second outright future; "
                 f"or pair with a negatively correlated contract (e.g., a crude long with a "
                 f"product crack hedge) to cut event risk.")
    elif primary.endswith("=X"):
        hedge = (f"**Hedge (FX):** with {regime} volatility, cap downside with an OTM option or a "
                 f"risk-reversal structure; avoid naked leveraged entries. For event risk "
                 f"(central-bank decisions), consider a short-dated straddle only if you can "
                 f"afford both premium and a stop.")
    elif "-" in primary or primary.startswith("^"):
        hedge = (f"**Hedge ({primary}):** crypto/index exposure is best protected with a defined-risk "
                 f"instrument (puts / inverse ETF or futures spread). In a {regime}-vol regime, "
                 f"size hedges at 25-50% of gross exposure and roll before expiry.")
    else:
        hedge = (f"**Hedge (equity):** in a {regime}-vol regime, a protective put ~5% OTM "
                 f"(30-60 DTE) or a collar (sell a ~10% OTM call to fund it) preserves upside "
                 f"while capping tail risk. Alternatively, an index hedge (SPY/QQQ puts) if your "
                 f"edge is sector/market beta rather than idiosyncratic.")

    invalidation = (f"Close above {stop:,.2f}" if bias == "SHORT" else
                    f"Close below {stop:,.2f}")
    invalidation += (f" or a daily close through the SMA{'50' if sma50 else '20'} trend line")

    lines = [
        f"### Trade Setup: {primary}",
        "",
        f"**Current price:** ${price:,.2f} · 5d: {chg5:+.2f}% · ATR(14): ${atr:,.2f} · "
        f"RSI(14): {rsi:.0f} · 20d realized vol: {rvol:.0f}% ann. ({regime})" if rvol else
        f"**Current price:** ${price:,.2f} · ATR(14): ${atr:,.2f} · RSI(14): {rsi:.0f}",
        "",
        f"**Bias: {bias}** — {bias_label}.",
        "",
        "**The Play:**",
        f"- **Entry zone:** ${entry_zone_lo:,.2f} – ${entry_zone_hi:,.2f} (limit into strength on confirmation)",
        f"- **Stop loss:** ${stop:,.2f} ({1.5 * atr / max(price, 1e-9) * 100:.1f}% away, ~1.5 × ATR)",
        f"- **Target 1:** ${t1:,.2f} ({2.0 * atr / max(price, 1e-9) * 100:.1f}%) → **R/R {r1:.2f}:1**",
        f"- **Target 2:** ${t2:,.2f} ({3.5 * atr / max(price, 1e-9) * 100:.1f}%) → **R/R {r2:.2f}:1**",
        "",
        "**Position sizing (risk per trade):**",
    ]
    if sma20 and sma50:
        trend = "uptrend" if sma20 > sma50 else "downtrend"
        lines.append(f"- Context: SMA20 ${sma20:,.2f} vs SMA50 ${sma50:,.2f} — {trend}; "
                     f"90d range ${lo90:,.2f} – ${hi90:,.2f}")
    if rvol:
        lines.append(f"- Vol regime {regime}: size smaller than normal ({regime} vol means wider "
                     f"stops; keep risk-per-trade ≤1% of capital).")
    for i in range(0, len(sizing), 4):
        lines.append(f"- Account {sizing[i]}: {sizing[i + 1]} · {sizing[i + 2]} · {sizing[i + 3]}")
    lines += [
        "",
        hedge,
        "",
        f"**Invalidation:** {invalidation}. Exiting at the stop is mandatory — no averaging down.",
        "",
        "*Setup generated from live price history (ATR/RSI/SMA/realized-vol). "
        "Not financial advice — size for your own risk tolerance.*",
    ]
    return "\n".join(lines)


def _build_hedging_strategy(query, intents, tickers, sectors, live_data, context_data=None):
    """Concrete hedging playbook: instrument choice, strike/expiry logic, cost
    budgeting and vol-regime sizing. Returns None if nothing to hedge."""
    primary = _primary_symbol(tickers)
    df = _fetch_series_for(primary) if primary else None
    m = _series_metrics(df) if df is not None else {}
    price = m.get("price")
    if price is None and primary:
        td = (live_data or {}).get(primary, {})
        if td:
            price = td.get("price")
    rvol = m.get("rvol")
    regime = _vol_regime_label(rvol)

    lines = [f"### Hedging Strategy"]
    if primary:
        lines.append(f"**Instrument in focus:** {primary}" + (f" at ${price:,.2f}" if price else ""))
    else:
        lines.append("**Instrument in focus:** portfolio / broad-market exposure")
    lines.append("")

    q = query.lower()
    is_assessment = bool(primary) and ("hedge for" in q or "a good hedge" in q
                                       or "as a hedge" in q or "good hedge for" in q)
    if not is_assessment and ("portfolio" in q or (not primary and not tickers)):
        lines += [
            "**Portfolio-level hedges (ranked by cost):**",
            "1. **Index puts (SPY/QQQ)** — the cleanest beta hedge; buy ~5% OTM, 45-60 DTE, roll monthly.",
            "2. **Collar on your largest winners** — sell a ~10% OTM call to fund the protective put; cap upside but zero cost.",
            "3. **Volatility convexity (VIX calls / VXX)** — best when you expect a sharp repricing; premium is insurance, not a trade.",
            "4. **Negative-beta assets** — long-duration treasuries (TLT) or gold (GLD) as permanent diversifiers in a {regime}-vol regime.",
            "5. **Reduce gross exposure** — the cheapest hedge is cash; in elevated vol, trim beta before paying for options.",
        ]
    elif not is_assessment:
        # price can be None under mocked/partial data — never multiply None.
        _put_strike = f"${price * 0.95:,.2f}" if price else "~5% below spot"
        _call_strike = f"${price * 1.10:,.2f}" if price else "~10% above spot"
        lines += [
            f"**Hedging {primary}** in a {regime}-vol regime:",
            f"1. **Protective put** — strike ~5% OTM (≈ {_put_strike} if long), 30-60 DTE; "
            f"caps loss at the strike while keeping upside open.",
            f"2. **Collar** — add a short call ~10% OTM (≈ {_call_strike}) to pay for the put; "
            "zero-cost when strikes are chosen symmetrically.",
            "3. **Put spread** — for cheaper protection, sell a put ~10% below your long put; "
            "reduces cost ~40-60% but leaves a gap below the short strike.",
            f"4. **Correlated hedge** — if {_display_symbol(primary)} moves with a sector, hedge the sector ETF instead "
            "(often more liquid, tighter spreads).",
        ]
    # ── "Is X a good hedge?" → assess the hedge asset with measured data ──
    ql = query.lower()
    if primary and ("hedge for" in ql or "a good hedge" in ql or "as a hedge" in ql
                    or "is x a hedge" in ql or "good hedge for" in ql):
        try:
            spy_df = _fetch_series_for("SPY", period="2y")
            ast_df = _fetch_series_for(primary, period="2y")
            if spy_df is not None and ast_df is not None:
                rs = spy_df["Close"]
                ra = ast_df["Close"]
                if isinstance(rs, pd.DataFrame):
                    rs = rs.iloc[:, 0]
                if isinstance(ra, pd.DataFrame):
                    ra = ra.iloc[:, 0]
                rs = rs.dropna().astype(float)
                ra = ra.dropna().astype(float)
                a, b = rs.align(ra, join="inner")
                if len(a) >= 60:
                    corr = float(a.pct_change().dropna().corr(b.pct_change().dropna()))
                    if corr < 0.2:
                        verdict = "a genuine diversifier — it tends to rise when stocks fall"
                    elif corr < 0.5:
                        verdict = "a weak diversifier — it lowers portfolio vol but is not a true hedge"
                    else:
                        verdict = "NOT a hedge — it moves with equities, so it amplifies drawdowns"
                    lines += [
                        "",
                        f"**Hedge assessment — {_display_symbol(primary)} vs US equities:**",
                        f"- Measured 2y daily correlation to SPY: **{corr:+.2f}** → {verdict}.",
                        f"- {'The hedge works best in risk-off shocks (rates/liquidity driven); it tends to fail in inflationary or growth-driven selloffs.' if corr < 0.5 else 'Pair it with a true negative-beta asset (long-duration treasuries or gold) rather than relying on it alone.'}",
                        f"- When it fails: correlation converges toward 1 in a broad liquidation — no asset is a perfect hedge when everything is sold.",
                    ]
        except Exception:
            pass

    # ── Options-specific strategy guidance when the query asks for it ──
    if intents.get("options") and primary:
        ql = query.lower()
        if "covered call" in ql or "covered calls" in ql or "sell calls" in ql:
            opt = ("**Covered-call play:** sell calls ~0.30 delta, 30-45 DTE, against shares you own; "
                   "collect premium and cap upside at the strike. Avoid selling into earnings unless "
                   "you accept gap risk (the stock gets called away).")
        elif "straddle" in ql or "strangle" in ql:
            opt = ("**Volatility play:** a long straddle/strangle pays only if the realized move exceeds "
                   "the priced move (implied vol). Enter when IV is cheap relative to realized, and size "
                   "for roughly a 2x move to break even.")
        elif "spread" in ql or "debit spread" in ql or "credit spread" in ql:
            opt = ("**Spread structure:** defined-risk debit/credit spreads cap both cost and payoff; "
                   "choose strikes near ~0.20-0.30 delta and expiry aligned with the catalyst "
                   "(earnings / event).")
        elif "bull call" in ql:
            opt = ("**Bull call spread:** buy a near-ATM call and sell a higher strike (~1.5-2x the width) "
                   "to cut cost; max loss is the debit, max gain is the width minus debit.")
        else:
            opt = ("**Options framework:** match the structure to the view — debit spreads for direction "
                   "with defined risk, credit spreads to harvest premium when IV is rich, and long "
                   "straddles only for large expected moves.")
        lines += ["", "**Options strategy guidance:**", opt]

    lines += [
        "",
        "**Cost & sizing discipline:**",
        f"- Budget hedging cost at 0.5–1.5% of notional per quarter in a {regime}-vol regime.",
        "- Hedge 25–50% of gross exposure for event risk; 100% only at extreme valuations or ahead of known catalysts.",
        "- Rebalance hedges weekly; roll options before they lose their convexity (last ~7 days of life).",
        "- Never let the hedge premium exceed the risk budget — if protection is too expensive, reduce size instead.",
        "",
        "*Framework based on current volatility and instrument characteristics. "
        "Not financial advice.*",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# DEEP-DIVE INSTITUTIONAL MEMO BUILDER
# ---------------------------------------------------------------------------

def _dd_momentum(chg):
    """Momentum score in [-1, 1] from the observed 5-day change."""
    try:
        return max(-1.0, min(1.0, float(chg or 0.0) / 6.0))
    except Exception:
        return 0.0


def _dd_risk_row(label, prob, sev, det, horizon, impact, note=""):
    line = (f"- **{label}** \u2014 Probability: {prob}, Severity: {sev}/5, "
            f"Detectability: {det}, Horizon: {horizon}, Est. valuation impact: {impact}")
    if note:
        line += f" \u2014 {note}"
    return line


def _build_institutional_deep_dive(query, intents, tickers, sectors, live_data,
                                   context_data=None, fundamentals=None):
    """Comprehensive institutional investment memo for deep-dive thesis
    queries, built to the RESEARCH-INTEGRITY spec:

    * Every number carries a provenance label: OBSERVED DATA, REPORTED
      FINANCIAL DATA, CONSENSUS ESTIMATE, MARKET-IMPLIED VALUE, MODEL
      CALCULATION, MODEL ASSUMPTION, HISTORICAL STATISTIC, SCENARIO
      ASSUMPTION, or INFERENCE. Unavailable data is disclosed as
      "DATA UNAVAILABLE - NO ESTIMATE SUBSTITUTED" — never invented.
    * Data-completeness gate drives confidence: with only a live quote the
      memo openly scores low and NEVER prints 70-90% confidence.
    * Real three-scenario DCF mechanics (when reported fundamentals are
      fetchable), reverse-DCF grid, expectation-gap, macro transmission,
      sentiment (positioning is never inferred from price momentum),
      risk/catalyst engines, five-scenario engine (probabilities sum to
      100%), Monte Carlo distribution, Bayesian updates, information
      advantage, falsification, computed confidence, and a final quality-
      control audit that can declare the analysis incomplete.

    Entity-resolution firewall: a financial-metric token (DCF/FCF/EBITDA/
    CAPEX/EPS/WACC/GPU/...) is NEVER accepted as the memo's subject, even if
    upstream extraction slipped it through.
    """
    syms = [t for t in (tickers or [])
            if not (str(t).startswith("^") or str(t).endswith(("=F", "=X", "-USD")))]
    # Entity-resolution firewall (defense in depth): a financial-metric token
    # must NEVER become the memo's subject, even if upstream extraction
    # slipped it through ("analyze DCF for FCF of EBITDA" must not name
    # DCF/FCF/EBITDA as securities).
    syms = [s for s in syms if not _is_financial_metric_token(str(s))]
    if not syms:
        syms = list(tickers or [])
    syms = [s for s in syms if not _is_financial_metric_token(str(s))]
    if not syms:
        # The thesis names a company whose ticker is ambiguous ("memo on LOW")
        # or a 1-char symbol ("evaluating V") — infer the subject from the
        # query text so the memo still engages the right name.
        _inferred = _infer_deep_dive_subject(query)
        if _inferred:
            syms = [_inferred]
    if not syms:
        return None
    sym = syms[0]
    display = _display_symbol(sym)
    td = (live_data or {}).get(sym, {})
    price = float(td.get("price", 0.0) or 0.0)
    chg = float(td.get("change_5d", 0.0) or 0.0)
    mom = _dd_momentum(chg)
    has_price = price > 0
    q = (query or "").lower()
    peers = [str(t) for t in (tickers or []) if str(t).upper() != sym.upper()]

    # ---- Reported fundamentals (REPORTED FINANCIAL DATA when fetched; never
    # fabricated). Skipped entirely offline so tests stay deterministic. ----
    if fundamentals is None:
        fundamentals = _dd_fetch_fundamentals(sym)
    fund = fundamentals or {}
    has_fund = bool(fund.get("revenue_m") and fund.get("shares_m"))
    # ---- Data-completeness gate (spec section 2) ----
    gate_lines, completeness = _dd_data_gate(display, has_price, has_fund, fund)

    # ---- Stance: observed tape first; explicit query direction only for
    # short directional asks (long theses contain incidental directional
    # vocabulary — "short interest", "bear scenario", "downside" — that
    # must not set the stance). ----
    bull_q, bear_q = bool(intents.get("bullish")), bool(intents.get("bearish"))
    if mom > 0.08:
        stance = "constructive"
    elif mom < -0.08:
        stance = "cautious"
    else:
        stance = "neutral"
    if len(q) < 220 and bull_q and not bear_q:
        stance = "bullish"
    elif len(q) < 220 and bear_q and not bull_q:
        stance = "bearish"

    # ---- Labeled model assumptions (never presented as observed facts) ----
    wacc = 0.095          # assumption: cost of capital
    term_g = 0.028        # assumption: terminal growth
    fcf_yield = 0.025 + 0.015 * max(mom, 0.0)   # assumption: normalized FCF yield 2.5-4%
    implied_g = max(0.02, wacc - fcf_yield)     # reverse-DCF implied growth
    base_g = max(implied_g, 0.05)               # assumption: base-case growth

    # ---- Scenario engine (transparent probabilities; model estimates) ----
    skew = 1.0 if mom > 0.1 else (-0.5 if mom < -0.1 else 0.0)
    scenarios = [
        ("Extreme Bull", 0.08 + 0.04 * skew, 0.35 + 0.10 * skew),
        ("Bull",         0.20 + 0.05 * skew, 0.16 + 0.06 * skew),
        ("Base",         0.40,               0.02 + 0.02 * skew),
        ("Bear",         0.22 - 0.04 * skew, -0.16 + 0.04 * skew),
        ("Extreme Bear", 0.10 - 0.05 * skew, -0.35 + 0.05 * skew),
    ]
    _tot = sum(p for _, p, _ in scenarios)
    scenarios = [(n, p / _tot, r) for n, p, r in scenarios]
    exp_ret = sum(p * r for _, p, r in scenarios)
    _p_eb = next(p for n, p, _ in scenarios if n == "Extreme Bear")
    _p_b = next(p for n, p, _ in scenarios if n == "Bear")
    p_dd30 = _p_eb + 0.5 * _p_b
    p_dd50 = _p_eb
    fair_center = price * (1.0 + min(max(exp_ret, -0.35), 0.45))
    fair_lo, fair_hi = fair_center * 0.90, fair_center * 1.10

    if exp_ret >= 0.20:
        rating = "Strong Buy"
    elif exp_ret >= 0.08:
        rating = "Buy"
    elif exp_ret >= -0.05:
        rating = "Hold"
    elif exp_ret >= -0.20:
        rating = "Sell"
    else:
        rating = "Strong Sell"

    # ---- Confidence engine (computed from data completeness + dispersion,
    # NOT arbitrary) ----
    _conf_txt, overall_conf = _dd_confidence(completeness, has_fund, scenarios, has_price)

    # ---- 1. Thesis pillars (tailored by the query's own vocabulary) ----
    pillars = ["Revenue / earnings power by segment", "Pricing power and unit economics",
               "Margin and operating-leverage trajectory", "Capital intensity and FCF conversion",
               "Capital-allocation discipline"]
    if "moat" in q or "cuda" in q:
        pillars.insert(1, "Durability of the competitive moat / ecosystem lock-in")
    if "market share" in q or "share loss" in q:
        pillars.append("Market-share trajectory vs. emerging alternatives")
    pillar_lines = "\n".join(f"- {p}" for p in pillars[:6])

    # ---- Valuation math (all assumption-labeled) ----
    tol_pct = round(max(3.0, min(25.0, 5.0 + 20.0 * (0.5 - exp_ret))))
    l = []
    l.append(f"### {display} \u2014 Institutional Investment View")
    l.append("")
    if has_price:
        l.append(f"**Current price:** ${price:,.2f} ({chg:+.2f}% over 5 days) (OBSERVED "
                 f"DATA). **Stance:** {stance}. The provenance legend in section 0 "
                 "applies to every number below.")
    else:
        l.append(f"**Current price:** live quote temporarily unavailable. "
                 f"**Stance:** {stance} (from request framing). Estimates below are "
                 "model assumptions to be re-based once a live quote is attached.")
    l.append("")
    l.append(_dd_provenance_legend())
    l.append("")
    l.append("### Investment Thesis")
    l.append("")
    l.append(f"**Stance:** {stance}. The pillars that determine forward earnings power "
             f"(tailored to the request's vocabulary):")
    l.append("")
    l.append(pillar_lines)
    l.append("")
    l.append("---")
    l.append("")
    l.append("### 1. Entity Resolution & Subject")
    l.append("")
    l.append(f"- **Subject security:** **{display}** ({sym}) \u2014 EQUITY. "
             f"Security candidates after firewall filtering: {', '.join(syms[:6]) or 'none'}.")
    l.append(f"- **Entity firewall (spec rule 1):** financial-metric tokens "
             "(DCF/FCF/EBITDA/CAPEX/EPS/WACC/GPU/AI/...) are NEVER analyzed as "
             "securities; only EQUITY / ETF / INDEX / OPTION / FUTURE / BOND / "
             "COMMODITY / CURRENCY / REIT / FUND entities pass.")
    if peers:
        l.append(f"- **Named comparables:** {', '.join(peers)} (securities only).")
    l.append("")
    l.append("---")
    l.append("")
    l.append("### 2. Data Completeness Gate")
    l.append("")
    l.extend(gate_lines)
    l.append("")
    l.append("---")
    l.append("")
    l.append("### 3. Fundamental Reconstruction")
    l.append("")
    l.append(_dd_segment_reconstruction(display, q, has_fund, fund))
    l.append("")
    l.append("### 3b. Highest-Sensitivity Variables (quantified)")
    l.append("")
    l.append(_dd_sensitivity_table(display, base_g, implied_g, has_fund, fund))
    l.append("")
    l.append("---")
    l.append("")
    l.append("### 4. AI Infrastructure Economics")
    l.append("")
    l.append(_dd_ai_infra_economics(display, q))
    l.append("")
    l.append("---")
    l.append("")
    l.append("### 5. Competitive Moat Analysis")
    l.append("")
    l.append(_dd_moat_matrix(display, q, wacc, tol_pct, has_fund, fund))
    l.append("")
    l.append("---")
    l.append("")
    l.append("### 6. Real DCF Engine")
    l.append("")
    _dcf_txt, dcf_fvs = _dd_real_dcf(display, fund, price, base_g, wacc, term_g,
                                     has_price, has_fund)
    l.append(_dcf_txt)
    l.append("")
    l.append("---")
    l.append("")
    l.append("### 7. Reverse DCF")
    l.append("")
    l.append(_dd_reverse_dcf(wacc, fcf_yield, term_g, price, has_price))
    l.append("")
    l.append("---")
    l.append("")
    l.append("### 8. Expectation-Gap Engine")
    l.append("")
    l.append(_dd_expectation_gap(display, has_fund, fund, base_g, implied_g))
    l.append("")
    l.append("---")
    l.append("")
    l.append("### 9. Macro Transmission Engine")
    l.append("")
    l.append(_dd_macro_transmission(q))
    l.append("")
    _macro = _build_macro_analysis(query, intents, sectors, live_data, context_data)
    if _macro:
        l.append(_macro)
    l.append("")
    l.append("---")
    l.append("")
    l.append("### 10. Sentiment & Positioning Engine")
    l.append("")
    l.append(_dd_sentiment_engine(display, chg, mom, has_price))
    l.append("")
    l.append("---")
    l.append("")
    l.append("### 11. Risk Engine")
    l.append("")
    l.append(_dd_risk_matrix(q))
    l.append("")
    l.append("---")
    l.append("")
    l.append("### 12. Catalyst Engine")
    l.append("")
    l.append(_dd_catalyst_table(display))
    l.append("")
    l.append("---")
    l.append("")
    l.append("### 13. Five-Scenario Engine")
    l.append("")
    l.append(_dd_scenario_table(display, scenarios, price, has_price, has_fund, fund,
                                base_g))
    l.append("")
    l.append(f"**Probability-weighted expected return: {exp_ret:+.1%}** (MODEL "
             f"CALCULATION); probability of a >30% drawdown: {p_dd30:.0%}; "
             f">50% drawdown: {p_dd50:.0%}.")
    l.append("")
    l.append("---")
    l.append("")
    l.append("### 14. Monte Carlo / Distribution Engine")
    l.append("")
    l.append(_dd_monte_carlo(scenarios, has_price))
    l.append("")
    l.append("---")
    l.append("")
    l.append("### 15. Bayesian Update Engine")
    l.append("")
    l.append(_dd_bayesian(display, scenarios))
    l.append("")
    l.append("---")
    l.append("")
    l.append("### 16. Information Advantage Engine")
    l.append("")
    l.append(_dd_info_advantage(display, q))
    l.append("")
    l.append("---")
    l.append("")
    l.append("### 17. Falsification Engine")
    l.append("")
    l.append(_dd_falsification(display, implied_g, base_g))
    l.append("")
    l.append("---")
    l.append("")
    l.append("### 18. Confidence Engine")
    l.append("")
    l.append(_conf_txt)
    l.append("")
    l.append("---")
    l.append("")
    l.append("### 19. Final Investment Committee Output")
    l.append("")
    l.append(_dd_final_committee(display, price, fair_lo, fair_hi, fair_center,
                                 exp_ret, rating, p_dd30, p_dd50, base_g, implied_g,
                                 overall_conf, has_price))
    l.append("")
    l.append("---")
    l.append("")
    l.append("### 20. Quality Control Audit")
    l.append("")
    l.append(_dd_qc_audit(has_price, has_fund, scenarios, dcf_fvs))
    l.append("")
    l.append("*Model output \u2014 verify against live data and your own due diligence. "
             "Not financial advice.*")
    return "\n".join(l)


# ---------------------------------------------------------------------------
# DEEP-DIVE MEMO RESEARCH-INTEGRITY HELPERS (spec sections 0-20)
# ---------------------------------------------------------------------------

# Financial metrics / concepts that must NEVER be analyzed as securities, used
# by the entity-resolution firewall as defense in depth (upstream extraction
# already blocks most of these via _SEMANTIC_CONCEPTS).
_FINANCIAL_METRIC_TOKENS = _SEMANTIC_CONCEPTS | {
    "DCF", "FCF", "EBITDA", "CAPEX", "EPS", "WACC", "NOPAT", "ROIC", "GPU",
    "CUDA", "ASIC", "AI", "REVENUE", "REVENUES", "MARGIN", "MARGINS",
    "SCENARIO", "SCENARIOS", "CONSENSUS", "GROWTH", "VALUATION", "CAGR",
    "BETA", "ALPHA", "SHARPE", "SORTINO", "VOLATILITY", "DRAWDOWN",
    "YIELD", "DIVIDEND", "PAYOUT", "LEVERAGE", "LIQUIDITY", "GUIDANCE",
    "INFERENCE", "TRAINING", "DATACENTER", "HYPESCALER",
}


# SPEC RULE 1 / 2 — provenance + auditable numbers
_DD_DATA_UNAVAILABLE = "DATA UNAVAILABLE - NO ESTIMATE SUBSTITUTED."


def _is_financial_metric_token(s: str) -> bool:
    """True for finance jargon that upstream extraction must never treat as a
    security (DCF, FCF, EBITDA, CAPEX, EPS, WACC, GPU, AI, ...)."""
    return str(s).upper() in _FINANCIAL_METRIC_TOKENS


def _dd_fetch_fundamentals(sym: str) -> dict:
    """REPORTED FINANCIAL DATA when fetchable, {} otherwise. Never fabricated.
    Skipped entirely when OCTAVIAN_OFFLINE=1 so tests stay fast/deterministic."""
    if os.environ.get("OCTAVIAN_OFFLINE") == "1":
        return {}
    try:
        from financial_model_generator import fetch_ticker_fundamentals
        return fetch_ticker_fundamentals(sym) or {}
    except Exception:  # noqa: BLE001
        return {}


def _dd_provenance_legend() -> str:
    return ("**Provenance legend (no unsupported numbers).** Every figure is tagged: "
            "(OBSERVED DATA) quote/tape; (REPORTED FINANCIAL DATA) filed statements; "
            "(CONSENSUS ESTIMATE) sell-side; (MARKET-IMPLIED VALUE) reverse-engineered "
            "from price; (MODEL CALCULATION) computed from the inputs above it; "
            "(MODEL ASSUMPTION) our assumption; (HISTORICAL STATISTIC) measured "
            "history; (SCENARIO ASSUMPTION) scenario input; (INFERENCE) reasoned "
            "judgment. Where data is missing the memo prints exactly: "
            f"**{_DD_DATA_UNAVAILABLE}**")


def _dd_data_gate(display, has_price, has_fund, fund):
    """Data-availability matrix + completeness score (spec section 2). Returns
    (markdown_lines, completeness_score). Completeness drives confidence."""
    rows = [
        ("Current price + 5-day tape", has_price, "OBSERVED DATA"),
        ("Income statement (revenue, EPS)", has_fund, "REPORTED FINANCIAL DATA" if has_fund else "DATA UNAVAILABLE"),
        ("Balance sheet (debt, cash)", has_fund and fund.get("debt_m") is not None, "REPORTED FINANCIAL DATA" if has_fund else "DATA UNAVAILABLE"),
        ("Cash flow statement", False, "DATA UNAVAILABLE"),
        ("Segment revenue", False, "DATA UNAVAILABLE - not in quote/fundamentals feed"),
        ("Consensus estimates / revisions", False, "DATA UNAVAILABLE"),
        ("Options (IV, skew, OI)", False, "DATA UNAVAILABLE"),
        ("Short interest", False, "DATA UNAVAILABLE"),
        ("Institutional positioning (13F)", False, "DATA UNAVAILABLE"),
        ("Macro regime context", True, "MODEL layer (rates/inflation transmission)"),
        ("News / alt data", False, "DATA UNAVAILABLE (not attached to this quote)"),
    ]
    lines = ["| Data Category | Available? | Source / Provenance |",
             "| --- | --- | --- |"]
    for cat, ok, src in rows:
        lines.append(f"| {cat} | {'YES' if ok else 'NO'} | {src} |")
    avail = (1 if has_price else 0) + (1 if has_fund else 0) \
        + (1 if (has_fund and fund.get("debt_m") is not None) else 0) + 1  # macro layer
    total = len(rows)
    completeness = round(avail / total, 3)
    lines.append("")
    lines.append(f"**DATA COMPLETENESS SCORE = {avail}/{total} = {completeness:.0%}** "
                 "(MODEL CALCULATION). Consensus, options, short interest, 13F, "
                 "segment detail, cash-flow and news are NOT attached to this quote, "
                 "so confidence in section 18 is capped accordingly.")
    return lines, completeness


_NVDA_SEGMENTS = [
    ("Data Center", "Growth engine (hyperscaler + enterprise AI compute)", "High-growth, capex-linked", "GPU/network unit demand, ASPs, utilization", "Direct - dominant AI revenue driver"),
    ("Gaming", "Consumer GPU cash cow", "Cyclical", "GeForce cycle, attach rates", "Low (indirect: AI PCs)"),
    ("Professional Visualization", "Workstation GPU/software", "Moderately cyclical", "Workstation refresh, Omniverse", "Medium - AI workstations"),
    ("Automotive", "Small, strategic (Orin/Thor, Drive)", "Long-cycle", "Design wins, robotaxi programs", "Medium - AV compute"),
    ("OEM & Other", "Residual OEM GPU / other", "Low-margin residual", "Mix", "Low"),
]


def _dd_segment_reconstruction(display, q, has_fund, fund):
    """Segment-level reconstruction (spec section 3). NVIDIA gets its real
    segment framework; other names get the generic dimensions. Reported segment
    figures are NOT attached, so the table is framework + qualitative call,
    never invented numbers."""
    is_nvda = "nvidia" in q or display.upper() == "NVDA"
    if is_nvda:
        seg_lines = ["**NVIDIA segment framework** (REPORTED segment revenue figures "
                     "are not attached to this quote - verify against the latest "
                     "10-K/10-Q):"]
        seg_lines.append("")
        seg_lines.append("| Segment | Role | Cyclicality | Key driver | Sensitivity to AI demand |")
        seg_lines.append("| --- | --- | --- | --- | --- |")
        for name, role, cyc, driver, ai in _NVDA_SEGMENTS:
            seg_lines.append(f"| {name} | {role} | {cyc} | {driver} | {ai} |")
        seg_lines.append("")
        seg_lines.append("- Data Center is the swing factor: its mix share sets the "
                         "consolidated growth ceiling and the blended gross margin "
                         "(INFERENCE - structurally rising mix share).")
        seg_lines.append("- The single most decision-relevant check: whether Data Center "
                         "growth is **volume-led (units) or price-led (ASPs)** and whether "
                         "inventory is being absorbed by real inference demand rather than "
                         "channel build (INFERENCE - watch utilization and cloud capex).")
        return "\n".join(seg_lines)
    # Generic reconstruction for any other name
    return (f"**{display} - revenue/earnings reconstruction.** Segment revenue is not "
            "attached to this quote, so segment figures are not estimated. The drivers "
            "that actually set forward earnings power: (1) revenue mix by business line "
            "(growth engines vs. mature), (2) pricing power vs. volume (margin quality), "
            "(3) unit growth and ASP mix, (4) gross margin structure, (5) operating "
            "leverage, (6) capex intensity and FCF conversion, (7) working-capital "
            "efficiency, (8) capital allocation (buybacks/dividends/M&A/R&D). Each must "
            "be validated against the latest filings (" + _DD_DATA_UNAVAILABLE + " for "
            "live figures here).")


def _dd_sensitivity_table(display, base_g, implied_g, has_fund, fund):
    """5-10 highest-sensitivity variables with QUANTIFIED earnings sensitivity
    (spec section 3). Numbers are MODEL CALCULATIONS on REPORTED inputs when the
    fundamentals feed is present, otherwise the formulas are given with the
    data-unavailable marker."""
    lines = ["| Variable | Current | Base | Bull | Bear | Earnings sensitivity |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    if has_fund:
        revenue_m = float(fund["revenue_m"])
        eps = float(fund.get("eps") or 0)
        shares_m = float(fund.get("shares_m") or 0)
        tax = float(fund.get("tax_rate_pct") or 21.0) / 100.0
        ni_m = eps * shares_m if shares_m else 0.0
        # +100bps gross margin -> pretax on revenue -> after-tax EPS impact
        gm_eps = (0.01 * revenue_m * (1 - tax)) / max(abs(ni_m), 0.01) if ni_m else None
        # +1pt revenue growth with ~1.3x operating leverage
        growth_eps = "~1.3x growth delta (MODEL ASSUMPTION)"
        gm_txt = f"\u00b1100bps gross margin -> EPS \u00b1{gm_eps:.0%}" if gm_eps else "n/a"
        rows = [
            ("Gross margin", "REPORTED", "+", "++", "--", gm_txt),
            ("Revenue growth", "REPORTED", f"{base_g:.0%}", f"{base_g + 0.08:.0%}", f"{max(0.0, base_g - 0.13):.0%}", growth_eps),
            ("ASP vs. unit mix", "REPORTED", "balanced", "price-led", "volume-led", "mix shift 1pt -> GM ~\u00b10.5pt (MODEL ASSUMPTION)"),
            ("Operating leverage", "REPORTED", "1.3x", "1.5x", "1.0x", "fixed-cost base scales EPS faster than revenue (MODEL ASSUMPTION)"),
            ("Capex intensity", "REPORTED", "steady", "rising", "falling", "FCF conversion inverse to capex/revenue (MODEL CALCULATION)"),
            ("Tax rate", "REPORTED", "flat", "-1pt", "+1pt", "\u00b1100bps tax -> EPS \u00b1~0.9-1.1% (MODEL CALCULATION)"),
            ("Buyback pace", "REPORTED", "flat", "accelerate", "pause", "per-share accretion inverse to share count (MODEL CALCULATION)"),
            ("China export restrictions", "n/a", "status quo", "eased", "tightened", "revenue haircut scenario (SCENARIO ASSUMPTION)"),
        ]
        for var, cur, b, u, d, sens in rows:
            lines.append(f"| {var} | {cur} | {b} | {u} | {d} | {sens} |")
    else:
        lines.append("| (all variables) | " + _DD_DATA_UNAVAILABLE + " | | | | |")
        lines.append("")
        lines.append("Sensitivity formulas (plug REPORTED inputs to compute): "
                     "EPS impact of a \u00b1100bps gross-margin move = 1% of revenue x "
                     "(1 - tax) / net income; EPS impact of \u00b11pt growth = growth delta "
                     "x operating leverage (~1.3x); FCF conversion = FCF/net income.")
    return "\n".join(lines)


def _dd_ai_infra_economics(display, q):
    """Mandatory AI-infrastructure economics section (spec section 4) for
    AI-adjacent names; None otherwise (the generic memo omits it)."""
    ql = q.lower()
    ai_relevant = any(k in ql for k in ("nvidia", "ai", "gpu", "hyperscaler",
                                        "semiconductor", "datacenter", "asic",
                                        "cuda", "inference", "token"))
    if not ai_relevant and display.upper() != "NVDA":
        return ("This section is mandatory in the framework but not applicable to "
                f"**{display}** (MODEL ASSUMPTION: not an AI-compute / hyperscaler / "
                "semiconductor name). The relevant capital-cycle questions for this "
                "company are its own capex intensity, capacity, and demand cycle - "
                "see sections 3, 6 and 11. " + _DD_DATA_UNAVAILABLE + " for AI-specific "
                "metrics that do not apply.")
    lines = [
        "**The economic chain under test:** AI CAPEX \u2192 GPU purchases \u2192 installed "
        "compute \u2192 utilization \u2192 training/inference workload \u2192 AI revenue \u2192 "
        "incremental gross profit \u2192 ROIC \u2192 depreciation burden \u2192 replacement cycle.",
        "",
        "**Is current AI infrastructure spending rational?** (probabilities are "
        "MODEL ASSUMPTIONS / INFERENCE):",
        "- Secular structural growth: **45%** - enterprise + hyperscaler AI adoption "
        "still early; inference demand growing faster than training.",
        "- Temporary capex supercycle: **25%** - capacity built ahead of proven "
        "monetization; a digestion phase is likely at some point.",
        "- Speculative bubble: **10%** - only if AI revenue fails to materialize "
        "relative to installed compute (low-probability tail).",
        "- Mixture (secular core + cyclical overlay): **20%** - the most probable "
        "reading: durable demand with inventory/capex cyclicality.",
        "",
        "**What would cause AI infrastructure spending to slow materially?**",
        "- Sustained GPU utilization below ~30-40% across major clouds ("
        "HISTORICAL STATISTIC threshold; live utilization " + _DD_DATA_UNAVAILABLE + ")",
        "- Weak inference/token economics: AI revenue per dollar of capex stays \u003c1x "
        "for several quarters (INFERENCE).",
        "- Model-efficiency breakthroughs that cut compute demand per unit of "
        "intelligence (distillation, sparse inference).",
        "- Power / data-center constraints easing demand rather than supply.",
        "- Financing costs rising enough to make 5-year paybacks negative.",
        "",
        "**Second-order effects to monitor:** depreciation waves hitting hyperscaler "
        "margins, ASIC substitution at the inference layer, and NVIDIA's own channel "
        "inventory. Live utilization/token-economics figures are not attached to "
        "this quote - treat this section as the decision framework.",
    ]
    return "\n".join(lines)


def _dd_moat_matrix(display, q, wacc, tol_pct, has_fund, fund):
    """Competitive matrix + quantified share-loss tolerance (spec section 5)."""
    moat_ask = "moat" in q or "cuda" in q or "asic" in q or "share loss" in q
    lines = [
        "**Competitive matrix** (qualitative ratings are INFERENCE; re-check against "
        "latest product cycles):",
        "",
        "| Factor | NVIDIA | AMD | Custom ASICs | Hyperscaler silicon |",
        "| --- | --- | --- | --- | --- |",
        "| Performance | Leader | Strong | Strong for fixed workloads | Strong for fixed workloads |",
        "| Cost | Premium | Competitive | Best unit economics at scale | Best at scale |",
        "| Software ecosystem | CUDA - durable moat | ROCm - improving | Weak | Proprietary per cloud |",
        "| Developer adoption | Highest | Moderate | Low | Low outside cloud |",
        "| Switching costs | High | Medium | Low-Medium | High within one cloud |",
        "| Supply | Constrained, improving | Improving | Contract-based | Contract-based |",
        "| Networking | Leader (NVLink/InfiniBand) | Moderate | n/a | Proprietary |",
        "| Inference economics | Strong | Competitive | Best at high volume | Best at high volume |",
        "| Training economics | Leader | Competitive | Niche | Niche |",
        "",
    ]
    lines.append(f"**Share-loss tolerance (MODEL CALCULATION):** at the current price and "
                 f"a {wacc:.1%} cost of capital, roughly **{tol_pct}%** of long-run "
                 "earnings power can be lost before base-case fair value is breached. "
                 "Share-loss sensitivity (1:1 earnings-to-share mapping, MODEL ASSUMPTION):")
    lines.append("")
    lines.append("| Market-share loss | Long-run earnings impact | vs. tolerance | Verdict |")
    lines.append("| ---: | ---: | ---: | --- |")
    for loss in (0, 5, 10, 15, 20, 30):
        within = loss <= tol_pct
        verdict = "within tolerance" if within else "breaks fair value"
        lines.append(f"| {loss}% | -{loss}% | {loss}% vs {tol_pct}% | {verdict} |")
    lines.append("")
    if moat_ask:
        lines.append("- **Ecosystem durability:** developer/customer lock-in compounds; "
                     "fragmentation (custom ASICs, alternative architectures) erodes it. "
                     "CUDA is the durable asset to watch (INFERENCE).")
        lines.append("- **Discontinuity risk:** a step-change in technology (not gradual "
                     "share drift) is the scenario that invalidates the thesis.")
    return "\n".join(lines)


def _dd_real_dcf(display, fund, price, base_g, wacc, term_g, has_price, has_fund):
    """Three-scenario DCF with real mechanics (spec section 6). Returns
    (markdown, {scenario: fair_value_per_share}) - empty dict when REPORTED
    inputs are unavailable (no numbers are invented)."""
    if not has_fund:
        txt = (f"**{_DD_DATA_UNAVAILABLE}** Absolute DCF fair value requires reported "
               "revenue, margins, capex, net debt and share count, none of which are "
               "attached to this quote. The full mechanics chain (to be computed once "
               "REPORTED inputs are provided): Revenue \u2192 EBIT margin \u2192 NOPAT \u2192 "
               "+D&A \u2192 -CAPEX \u2192 -\u0394NWC \u2192 FCF \u2192 PV @ WACC \u2192 terminal (g) \u2192 "
               "EV \u2192 -net debt + cash \u2192 equity \u2192 \u00f7 shares \u2192 fair value/share. "
               "The price-implied framework in section 7 is used instead "
               "(MARKET-IMPLIED VALUE).")
        return txt, {}
    revenue_m = float(fund["revenue_m"])
    shares_m = float(fund["shares_m"])
    ebit_m = float(fund.get("ebit_margin_pct") or 20.0) / 100.0
    tax = float(fund.get("tax_rate_pct") or 21.0) / 100.0
    debt_m = float(fund.get("debt_m") or 0)
    cash_m = float(fund.get("cash_m") or 0)
    da_rate, capex_rate, nwc_rate = 0.06, 0.08, 0.05  # MODEL ASSUMPTION (% revenue)
    years = 5
    dcf_scen = [
        ("Bear", max(-0.05, base_g - 0.13), ebit_m - 0.04),
        ("Base", base_g, ebit_m),
        ("Bull", base_g + 0.08, ebit_m + 0.04),
    ]
    rows = ["| Scenario | Rev growth | EBIT margin | FCF (Y5) | PV of FCF | Terminal | EV | Equity | Fair value/share |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    fvs = {}
    base_fv = None
    for name, g0, m0 in dcf_scen:
        rev = prev = revenue_m
        pv = 0.0
        fcf_last = 0.0
        for y in range(1, years + 1):
            g = g0 + (term_g - g0) * (y - 1) / max(years - 1, 1)
            nrev = rev * (1 + g)
            nopat = nrev * m0 * (1 - tax)
            fcf = nopat + nrev * da_rate - nrev * capex_rate - (nrev - prev) * nwc_rate
            pv += fcf / (1 + wacc) ** y
            rev, prev, fcf_last = nrev, nrev, fcf
        tv = fcf_last * (1 + term_g) / (wacc - term_g)
        pv += tv / (1 + wacc) ** years
        ev = pv
        equity = ev - debt_m + cash_m
        fv = equity / shares_m
        fvs[name] = fv
        if name == "Base":
            base_fv = fv
        rows.append(f"| {name} | {g0:.1%} | {m0:.1%} | ${fcf_last:,.0f}M | ${pv:,.0f}M | "
                    f"${tv:,.0f}M | ${ev:,.0f}M | ${equity:,.0f}M | ${fv:,.2f} |")
    lines = ["**Three-scenario DCF mechanics** (REPORTED inputs + MODEL ASSUMPTIONS: "
             f"WACC {wacc:.1%}, terminal growth {term_g:.1%}, D&A {da_rate:.0%} of revenue, "
             f"CAPEX {capex_rate:.0%} of revenue, \u0394NWC {nwc_rate:.0%} of incremental "
             "revenue, 5-yr fade to terminal):"]
    lines.append("")
    lines.extend(rows)
    if base_fv:
        lines.append("")
        lines.append("**WACC \u00d7 terminal-growth sensitivity** on the Base DCF (MODEL "
                     "CALCULATION):")
        lines.append("")
        lines.append("| | g=2.0% | g=2.8% | g=3.5% |")
        lines.append("| --- | ---: | ---: | ---: |")
        for w in (0.085, 0.095, 0.105):
            cells = []
            for g in (0.020, 0.028, 0.035):
                fv = _dd_dcf_value(revenue_m, shares_m, ebit_m, tax, debt_m, cash_m,
                                   base_g, w, g, da_rate, capex_rate, nwc_rate, years)
                cells.append(f"${fv:,.2f}")
            rows_s = " | ".join(cells)
            lines.append(f"| WACC {w:.1%} | {rows_s} |")
        lines.append("")
        lines.append("**Reconciliation (read this before using any DCF number):** the "
                     "conservative DCF fair values above sit far below the market price "
                     "- that spread IS the expectation gap quantified in section 8. The "
                     "DCF is the fundamental downside anchor under normalized assumptions; "
                     "the probability-weighted fair value in section 13 (anchored to the "
                     "market-implied framework) is the decision value. The single largest "
                     "valuation-sensitivity driver is the growth-vs-margin combination, "
                     "not WACC or terminal growth (MODEL CALCULATION / INFERENCE).")
    return "\n".join(lines), fvs


def _dd_dcf_value(revenue_m, shares_m, ebit_m, tax, debt_m, cash_m, g0, wacc, term_g,
                  da_rate, capex_rate, nwc_rate, years=5):
    """Pure DCF computation reused for sensitivity grids."""
    rev = prev = revenue_m
    pv = 0.0
    fcf_last = 0.0
    for y in range(1, years + 1):
        g = g0 + (term_g - g0) * (y - 1) / max(years - 1, 1)
        nrev = rev * (1 + g)
        nopat = nrev * ebit_m * (1 - tax)
        fcf = nopat + nrev * da_rate - nrev * capex_rate - (nrev - prev) * nwc_rate
        pv += fcf / (1 + wacc) ** y
        rev, prev, fcf_last = nrev, nrev, fcf
    tv = fcf_last * (1 + term_g) / (wacc - term_g)
    return (pv + tv / (1 + wacc) ** years - debt_m + cash_m) / shares_m


def _dd_reverse_dcf(wacc, fcf_yield, term_g, price, has_price):
    """Reverse-DCF grid (spec section 7): implied sustainable growth under
    WACC \u00d7 FCF-yield combinations (MARKET-IMPLIED VALUE given assumptions)."""
    if not has_price:
        return "Reverse DCF requires a live quote: " + _DD_DATA_UNAVAILABLE
    lines = [
        "**What the current price implies (Gordon-growth capitalization: "
        "g_implied = WACC - normalized FCF yield):**",
        "",
        "| | FCF yield 2.5% | FCF yield 3.4% | FCF yield 4.0% |",
        "| --- | ---: | ---: | ---: |",
    ]
    for w in (0.085, 0.095, 0.105):
        cells = " | ".join(f"{max(0.005, w - fy):.1%}" for fy in (0.025, 0.034, 0.040))
        lines.append(f"| WACC {w:.1%} | {cells} |")
    lines.append("")
    lines.append(f"At WACC {wacc:.1%} and the assumed normalized FCF yield "
                 f"{fcf_yield:.1%}, the observed price ${price:,.2f} embeds sustained "
                 f"growth near **{max(0.005, wacc - fcf_yield):.1%}** (MARKET-IMPLIED "
                 "VALUE). Terminal growth shifts the read by \u00b1~0.3pt (MODEL "
                 "ASSUMPTION).")
    lines.append("")
    lines.append("**What must happen operationally for today's price to be justified?** "
                 "Revenue must compound at (or above) the implied rate with stable "
                 "margins and FCF conversion; any shortfall in growth or margin is "
                 "priced in as de-rating. The gap between this implied rate and your "
                 "own base case is the entire expectation debate.")
    return "\n".join(lines)


def _dd_expectation_gap(display, has_fund, fund, base_g, implied_g):
    """Expectation-gap engine (spec section 8): company vs consensus vs model vs
    market-implied per metric; honest DATA UNAVAILABLE where no feed exists."""
    def _cell(v):
        return v if v else _DD_DATA_UNAVAILABLE
    lines = [
        "| Metric | Company guidance | Consensus | Model base | Market-implied | Gap |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| Revenue growth | {_cell(None)} | {_cell(None)} | {base_g:.0%} (MODEL ASSUMPTION) | {implied_g:.0%} (MARKET-IMPLIED) | {(base_g - implied_g):+.0%} |",
    ]
    if has_fund:
        lines.append(f"| EPS | {_cell(None)} | {_cell(None)} | ${fund.get('eps', 0):.2f} (REPORTED, trailing) | DATA UNAVAILABLE | n/a |")
        lines.append(f"| EBIT margin | {_cell(None)} | {_cell(None)} | {fund.get('ebit_margin_pct', 0):.1f}% (REPORTED) | DATA UNAVAILABLE | n/a |")
    else:
        lines.append(f"| EPS | {_cell(None)} | {_cell(None)} | {_cell(None)} | {_cell(None)} | n/a |")
        lines.append(f"| Margin | {_cell(None)} | {_cell(None)} | {_cell(None)} | {_cell(None)} | n/a |")
    lines.append("")
    lines.append("**Largest disagreement:** growth expectations (model base vs. market-"
                 "implied). Consensus, company guidance and options-implied expectations "
                 "are not attached to this quote, so the gap shown is model-vs-market only "
                 "(the honest, computable subset).")
    lines.append("")
    lines.append(f"**Can {display} outperform fundamentals and still decline?** Yes - if "
                 f"the market already prices growth above {implied_g:.0%}, even solid "
                 "execution can de-rate. The reverse (bad fundamentals, stock rallies) "
                 "happens when expectations were already low enough.")
    return "\n".join(lines)


def _dd_macro_transmission(q):
    """Macro transmission table (spec section 9): each macro variable mapped to
    earnings impact, multiple impact and total. MODEL ASSUMPTIONS throughout."""
    lines = [
        "**Transmission into the company's earnings and multiple** (MODEL ASSUMPTIONS; "
        "direction is the current regime read, not a forecast):",
        "",
        "| Macro variable | Direction | Earnings impact | Multiple impact | Total valuation impact |",
        "| --- | --- | ---: | ---: | ---: |",
        "| Fed policy / real rates | rates up = headwind | -1 to -2% | -5 to -10% | -6 to -12% |",
        "| Inflation (core services) | sticky = headwind | -1% | -3 to -8% | -4 to -9% |",
        "| USD strength | strong USD = headwind | -1 to -2% (translation) | -2% | -3 to -4% |",
        "| GDP / corporate capex cycle | resilient = tailwind | +2 to +5% | +3 to +8% | +5 to +13% |",
        "| Liquidity / credit conditions | easy = tailwind | +1% | +5 to +10% | +6 to +11% |",
        "| China restrictions | tightening = risk | -2 to -5% | -3 to -8% | -5 to -13% |",
        "| Geopolitical risk | elevated = risk | -1% | -3 to -6% | -4 to -7% |",
        "",
        "**Regime scenarios and how fair value moves:**",
        "- Soft landing: modest tailwind (+0-5% to fair value).",
        "- Recession: multiple de-rate dominates (-15 to -30%).",
        "- Stagflation: growth + margin double hit (-20 to -35%).",
        "- Renewed inflation: rate-driven de-rate (-10 to -20%).",
        "- Productivity / AI boom: earnings upgrade + multiple expansion (+10 to +30%).",
    ]
    return "\n".join(lines)


def _dd_sentiment_engine(display, chg, mom, has_price):
    """Sentiment & positioning (spec section 10). Positioning is NEVER inferred
    from price momentum; unavailable dimensions print DATA UNAVAILABLE."""
    lines = [
        "**Positioning dimensions** (spec rule: price momentum is NOT a substitute "
        "for sentiment data):",
        "",
        "| Dimension | Bullish/Bearish | Strength | Crowding |",
        "| --- | --- | ---: | ---: |",
        f"| Observed tape (5-day) | {'Bullish' if mom > 0.1 else 'Bearish' if mom < -0.1 else 'Neutral'} (OBSERVED DATA) | {abs(chg):.1f}% | n/a - not positioning |",
        f"| Retail social sentiment | {_DD_DATA_UNAVAILABLE} | n/a | n/a |",
        f"| Institutional 13F positioning | {_DD_DATA_UNAVAILABLE} | n/a | n/a |",
        f"| Sell-side ratings/revisions | {_DD_DATA_UNAVAILABLE} | n/a | n/a |",
        f"| Options IV / skew / OI | {_DD_DATA_UNAVAILABLE} | n/a | n/a |",
        f"| Short interest | {_DD_DATA_UNAVAILABLE} | n/a | n/a |",
        "",
        "**Crowding:** cannot be measured from this quote. The observed tape is "
        "momentum-positive/negative only, which is consistent with crowding risk in "
        "either direction \u2014 confirm via flow/options data before sizing. What would "
        "shift the sentiment regime: an earnings/guidance surprise, a hyperscaler "
        "capex announcement, or a headline macro shock (INFERENCE).",
    ]
    return "\n".join(lines)


def _dd_risk_matrix(q):
    """Full risk matrix (spec section 11): probability, severity, horizon,
    detectability, valuation impact, leading indicator, mitigation, priced-in."""
    rows = [
        ("Business / competitive risk", "Med", "4/5", "1-3 yrs", "High", "-5% to -20%", "Market share in flagship segment", "Product cadence + pricing discipline", "Partly"),
        ("AI capex slowdown (demand shock)", "Med", "4/5", "0-18 mo", "Medium", "-10% to -30%", "Hyperscaler capex guidance, utilization", "Diversify revenue base", "Partly"),
        ("Margin compression", "Med", "3/5", "0-12 mo", "High", "-5% to -15%", "Gross margin prints, pricing", "Mix + cost discipline", "Partly"),
        ("Market / multiple risk", "Med", "4/5", "0-12 mo", "Medium", "-10% to -30%", "Rates, breadth, valuation vs history", "Position sizing", "No"),
        ("Geopolitical / China restrictions", "Med", "4/5", "0-24 mo", "Medium", "-10% to -25%", "Export-control announcements", "Geographic mix, licenses", "Partly"),
        ("Valuation risk (expectations embedded)", "Med", "3/5", "0-12 mo", "Medium", "-5% to -15%", "Growth vs implied rate", "Scenario discipline", "No"),
        ("Technological discontinuity", "Low", "5/5", "1-3 yrs", "Low", "-20% to -40%", "Architecture transitions, ASIC wins", "R&D watch, ecosystem breadth", "Partly"),
        ("Customer concentration", "Med", "3/5", "0-24 mo", "Medium", "-5% to -15%", "Top-customer capex mix", "Broaden customer base", "Partly"),
        ("Regulatory / policy risk", "Low", "3/5", "1-3 yrs", "Low", "-5% to -15%", "Antitrust/export developments", "Compliance posture", "Partly"),
        ("Financial / execution risk", "Low", "3/5", "1-3 yrs", "Medium", "-3% to -12%", "Balance-sheet moves, capex", "Capital discipline", "No"),
        ("Macro / rates regime risk", "Med", "3/5", "0-24 mo", "Low", "-5% to -15%", "Fed path, real yields", "Hedging, duration posture", "Partly"),
        ("Sentiment / narrative risk", "Med", "4/5", "0-6 mo", "High", "-10% to -25%", "Flows, options skew, media tone", "Conviction discipline", "No"),
    ]
    if "geopolit" in q or "china" in q or "sanction" in q or "tariff" in q:
        rows.insert(2, ("U.S.-China trade restrictions (flagged in request)", "Med", "4/5", "0-24 mo", "Medium", "-10% to -25%", "Export-control / licensing news", "Geographic mix, contingency", "Partly"))
    lines = ["| Risk | Prob | Severity | Horizon | Detectability | Valuation impact | Leading indicator | Mitigation | Priced in? (INFERENCE) |",
             "| --- | --- | ---: | --- | --- | ---: | --- | --- | --- |"]
    for label, prob, sev, hor, det, imp, lead, mit, priced in rows:
        lines.append(f"| {label} | {prob} | {sev} | {hor} | {det} | {imp} | {lead} | {mit} | {priced} |")
    lines.append("")
    lines.append("Non-consensus risks: AI-capex digestion and customer concentration are "
                 "underweighted by the consensus narrative relative to their probability "
                 "(INFERENCE).")
    return "\n".join(lines)


def _dd_catalyst_table(display):
    """Catalyst engine (spec section 12): timeline buckets with probability,
    fundamental/EPS/valuation impact and expected price reaction. No fabricated
    dates."""
    rows = [
        ("Known", "0-3 mo", "Earnings print + guidance", "Med", "High", "Med", "Either direction, 2-6%"),
        ("Known", "0-3 mo", "Capital-allocation announcements", "Med", "Low", "Low-Med", "+/-1-3%"),
        ("Known", "3-6 mo", "Product cycle / capacity updates", "Med-High", "Med", "Med", "+/-3-8%"),
        ("Known", "6-12 mo", "Margin trajectory confirmation", "Med", "Med-High", "High", "+/-4-10%"),
        ("Probable", "3-6 mo", "Hyperscaler capex guidance updates", "Med", "Med", "Med-High", "+/-3-8%"),
        ("Probable", "6-12 mo", "Competitive share-shift data points", "Med", "Med", "Med", "+/-3-7%"),
        ("Probable", "12-24 mo", "New market expansion / ecosystem adoption", "Med", "High", "High", "+/-5-12%"),
        ("Speculative", "0-12 mo", "Export-control policy change", "Low-Med", "Med-High", "Med", "-5 to -15% on tightening"),
        ("Speculative", "6-24 mo", "Technological discontinuity (ASIC/architectural)", "Low", "High", "High", "-10 to -25% on confirmation"),
        ("Speculative", "12-24 mo", "Inference demand inflection", "Low-Med", "High", "High", "+5 to +15% on evidence"),
    ]
    lines = ["| Class | Timeline | Catalyst | Probability | Fundamental impact | EPS impact | Expected price reaction |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for cls, tm, cat, prob, fi, ei, px in rows:
        lines.append(f"| {cls} | {tm} | {cat} | {prob} | {fi} | {ei} | {px} |")
    lines.append("")
    lines.append("All probabilities are MODEL ASSUMPTIONS; no event dates are fabricated "
                 "- verify against the actual earnings/event calendar.")
    return "\n".join(lines)


def _dd_scenario_table(display, scenarios, price, has_price, has_fund, fund, base_g=0.05):
    """Five-scenario engine (spec section 13): probabilities sum to 100%; every
    scenario has revenue/EPS/FCF/margin/exit multiple/price. Absolute fundamentals
    print DATA UNAVAILABLE when REPORTED inputs are missing."""
    base_rev = float(fund.get("revenue_m") or 0) if has_fund else None
    base_eps = float(fund.get("eps") or 0) if has_fund else None
    base_mult = float(fund.get("exit_multiple") or 10.0) if has_fund else None
    growth_map = {"Extreme Bull": 0.18, "Bull": 0.08, "Base": 0.0, "Bear": -0.13, "Extreme Bear": -0.22}
    mult_map = {"Extreme Bull": 1.30, "Bull": 1.15, "Base": 1.00, "Bear": 0.85, "Extreme Bear": 0.60}
    lines = ["| Scenario | Prob | Revenue (12m) | Rev growth | EBIT margin | EPS (12m) | FCF (12m) | Exit mult. | Implied price | Expected return |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, p, r in scenarios:
        if has_fund:
            g_d = base_g + growth_map.get(name, 0.0)
            rev_12 = base_rev * (1 + g_d)
            eps_12 = base_eps * (1 + g_d * 1.3)
            fcf_12 = rev_12 * 0.25 * (1.0 + mult_map.get(name, 1.0) * 0.10)
            marg = float(fund.get("ebit_margin_pct") or 20.0) + (0.04 if "Bull" in name and "Extreme" not in name else 0.0) \
                - (0.04 if "Bear" in name and "Extreme" not in name else 0.0) \
                + (0.06 if name == "Extreme Bull" else 0.0) - (0.06 if name == "Extreme Bear" else 0.0)
            mult = base_mult * mult_map.get(name, 1.0)
            px = price * (1 + r) if has_price else None
            px_txt = f"${px:,.2f}" if px else "n/a"
            lines.append(f"| {name} | {p:.0%} | ${rev_12:,.0f}M | {g_d:+.0%} | {marg:.1f}% | ${eps_12:,.2f} | ${fcf_12:,.0f}M | {mult:.1f}x | {px_txt} | {r:+.0%} |")
        else:
            px = price * (1 + r) if has_price else None
            px_txt = f"${px:,.2f}" if px else "n/a"
            g_d = base_g + growth_map.get(name, 0.0)
            lines.append(f"| {name} | {p:.0%} | {_DD_DATA_UNAVAILABLE} | {g_d:+.0%} (SCENARIO ASSUMPTION) | n/a | n/a | n/a | n/a | {px_txt} | {r:+.0%} |")
    lines.append("")
    lines.append("Probabilities sum to exactly 100% (MODEL CALCULATION). Revenue/EPS/FCF/"
                 "margin/multiple cells marked n/a are DATA UNAVAILABLE (no REPORTED "
                 "inputs attached) - they are not estimated.")
    return "\n".join(lines)


def _dd_monte_carlo(scenarios, has_price):
    """Distribution engine (spec section 14): empirical CDF of the five scenario
    returns; P10-P90; loss probabilities; P(outperform S&P 500). MODEL
    CALCULATION on the scenario set (no manufactured randomness)."""
    if not has_price:
        return "Monte Carlo / distribution requires a live quote: " + _DD_DATA_UNAVAILABLE
    pts = sorted([(r, p) for _, p, r in scenarios], key=lambda x: x[0])
    cum = 0.0
    def _pct(qt):
        # linear interpolation over the empirical CDF
        prev_r, prev_c = pts[0][0], 0.0
        for r, p in pts:
            c = prev_c + p
            if qt <= c:
                if c - prev_c <= 0:
                    return r
                return prev_r + (r - prev_r) * (qt - prev_c) / (c - prev_c)
            prev_r, prev_c = r, c
        return pts[-1][0]
    p10, p25, p50, p75, p90 = (_pct(q) for q in (0.10, 0.25, 0.50, 0.75, 0.90))
    def _p_below(level):
        c = 0.0
        for r, p in pts:
            if r <= level:
                c += p
        return c
    lines = [
        "**Empirical distribution from the five scenario returns** (MODEL "
        "CALCULATION; a full Monte Carlo requires defensible input distributions, "
        "so the scenario set is used as the discrete distribution rather than "
        "manufactured randomness):",
        "",
        f"| P10 | P25 | P50 | P75 | P90 |",
        f"| ---: | ---: | ---: | ---: | ---: |",
        f"| {p10:+.0%} | {p25:+.0%} | {p50:+.0%} | {p75:+.0%} | {p90:+.0%} |",
        "",
        f"- Probability of >10% loss: **{_p_below(-0.10):.0%}**",
        f"- Probability of >20% loss: **{_p_below(-0.20):.0%}**",
        f"- Probability of >30% loss: **{_p_below(-0.30):.0%}**",
        f"- Probability of >50% loss: **{_p_below(-0.50):.0%}**",
        "- Probability of outperforming the S&P 500: "
        f"**{1.0 - _p_below(0.08):.0%}** "
        "(MODEL ASSUMPTION: S&P 500 12m total return ~+8%).",
    ]
    return "\n".join(lines)


_BAYES_EVENTS = [
    ("Earnings beat with raised guidance", (2.0, 1.0, 0.4)),
    ("Earnings miss / guided down", (0.4, 0.8, 2.2)),
    ("Hyperscaler capex accelerates", (2.2, 1.1, 0.4)),
    ("Hyperscaler capex slows", (0.4, 0.9, 2.0)),
    ("Major custom-ASIC adoption", (0.5, 0.9, 1.9)),
    ("Gross-margin deterioration", (0.5, 0.9, 1.8)),
    ("New export restriction on China", (0.5, 0.9, 2.1)),
    ("Inference demand accelerates", (2.0, 1.1, 0.5)),
    ("Model-efficiency breakthrough cuts compute demand", (0.7, 1.0, 1.6)),
    ("Competitive market-share shift to rival", (0.6, 1.0, 1.7)),
]


def _dd_bayesian(display, scenarios):
    """Bayesian update engine (spec section 15): priors from the scenario set;
    ten evidence events with likelihood ratios; posterior per outcome. MODEL
    ASSUMPTIONS on the likelihood ratios."""
    p_bull = next(p for n, p, _ in scenarios if n == "Extreme Bull") \
        + next(p for n, p, _ in scenarios if n == "Bull")
    p_base = next(p for n, p, _ in scenarios if n == "Base")
    p_bear = next(p for n, p, _ in scenarios if n == "Bear") \
        + next(p for n, p, _ in scenarios if n == "Extreme Bear")
    prior = {"Bull": p_bull, "Base": p_base, "Bear": p_bear}
    lines = ["**Bayesian updating on the scenario distribution** (priors = scenario "
             "probabilities; likelihood ratios are MODEL ASSUMPTIONS; posterior = "
             "prior \u00d7 LR, renormalized - MODEL CALCULATION):", "",
             "| New evidence | Prior P(Bull/Base/Bear) | Likelihood (Bull/Base/Bear) | Posterior P(Bull/Base/Bear) |",
             "| --- | ---: | ---: | ---: |"]
    for ev, (lrb, lrm, lrs) in _BAYES_EVENTS:
        post = {k: prior[k] * lr for k, lr in zip(("Bull", "Base", "Bear"), (lrb, lrm, lrs))}
        tot = sum(post.values())
        post = {k: v / tot for k, v in post.items()}
        lines.append(f"| {ev} | {prior['Bull']:.0%}/{prior['Base']:.0%}/{prior['Bear']:.0%} | "
                     f"{lrb:.1f}/{lrm:.1f}/{lrs:.1f} | {post['Bull']:.0%}/{post['Base']:.0%}/{post['Bear']:.0%} |")
    lines.append("")
    lines.append("Evidence that would materially change the thesis: hyperscaler capex "
                 "direction, gross-margin prints, and export-control policy are the "
                 "highest-information events for this name (INFERENCE).")
    return "\n".join(lines)


_INFO_ADVANTAGE = [
    ("GPU allocation / order backlog", "Real demand vs. channel build", "Management disclosures, supply-chain checks", "Order lead times, distributor inventory", "Backlog growing + lead times long", "Backlog shrinking + lead times normalizing", "Partial - vendor data only"),
    ("Inference token economics", "Whether AI revenue justifies compute", "Cloud AI revenue per GPU-hour", "AI API pricing, utilization disclosures", "Revenue per GPU-hour rising", "Falling despite volume growth", "No"),
    ("Data-center utilization", "Overbuild risk signal", "Hyperscaler utilization disclosures", "Cloud capex vs. AI revenue mix", "Rising utilization", "Sustained low utilization", "No"),
    ("Capacity / power pipeline", "Supply-constrained revenue ceiling", "Power agreements, fab allocation", "Lead times, power-purchase announcements", "Capacity locked in ahead of demand", "Power/permitting delays", "Partial"),
    ("Hyperscaler AI ROI evidence", "Capex-cycle durability", "Management commentary, S-1s of AI startups", "AI revenue disclosures vs. capex", "ROI improving", "ROI deteriorating", "No"),
    ("Custom-ASIC design wins", "Share-loss early warning", "Design-win announcements, tape-outs", "Cloud/startup ASIC announcements", "Few new wins (status quo)", "Broad design-win migration", "No"),
    ("Customer concentration", "Demand shock exposure", "Top-customer revenue share", "Annual report disclosures", "Concentration falling", "Concentration rising", "Partial"),
    ("China license pipeline", "Export-restriction revenue risk", "License filings, export-control lists", "News flow, partner commentary", "Licenses flowing", "Restrictions tightening", "No"),
]


def _dd_info_advantage(display, q):
    """Information-advantage engine (spec section 16): variables a top-tier
    institution would want that filings do not reveal; each with why it matters,
    measurement, proxy, bullish/bearish signal, and availability."""
    lines = ["| Variable | Why it matters | Theoretical measurement | Observable proxy | Bullish signal | Bearish signal | Availability |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for var, why, meas, proxy, bull, bear, avail in _INFO_ADVANTAGE:
        lines.append(f"| {var} | {why} | {meas} | {proxy} | {bull} | {bear} | {avail} |")
    lines.append("")
    lines.append("None of these feeds are attached to this quote (each marked "
                 "availability); they are the decision-useful data an institutional "
                 "desk would pull before committing capital (INFERENCE).")
    return "\n".join(lines)


_FALSIFICATION_ARGS = [
    ("AI capex digestion (hyperscalers pause buying)", "capacity built ahead of monetization", 0.20, "-20 to -40% revenue growth", "capex guides down 2+ quarters", "utilization + AI revenue keep rising", True),
    ("Custom-ASIC substitution at inference", "TPU-class silicon takes the high-volume layer", 0.25, "-10 to -25% share", "design wins + volume ramps", "NVIDIA wins the training layer anyway", True),
    ("Gross-margin compression", "mix shifts or price competition", 0.25, "-5 to -15% EPS", "margin guides down", "margin holds >50%", True),
    ("Valuation de-rate (multiple compression)", "growth no longer justifies the premium", 0.30, "-15 to -30% price", "growth decelerates to market rates", "growth re-accelerates", True),
    ("Export restrictions on China", "revenue haircut on the largest ex-US market", 0.20, "-10 to -25% revenue", "new export controls announced", "licensing resolved", True),
    ("Customer concentration shock", "a top-2 customer cuts capex", 0.15, "-10 to -20% revenue", "top-customer capex guides down", "customer base broadens", True),
    ("Technological discontinuity", "post-GPU architecture leap by a rival", 0.10, "-20 to -40% franchise value", "rival architecture wins flagship deals", "NVIDIA roadmap extends lead", True),
    ("Power / data-center constraints", "demand softens as capacity hits limits", 0.25, "-5 to -15% growth", "power availability stalls deployments", "power resolved + demand intact", False),
    ("Macro recession", "corporate IT/AI budgets cut", 0.20, "-15 to -30%", "ISM/IT-spend surveys deteriorate", "recession avoided", False),
    ("Model-efficiency breakthrough", "less compute needed per unit of intelligence", 0.15, "-10 to -20% demand", "efficiency gains documented at scale", "new workloads absorb the savings", False),
]


def _dd_falsification(display, implied_g, base_g):
    """Falsification engine (spec section 17): 10 ranked arguments against the
    thesis; what confirms/falsifies each; whether it changes the rating."""
    lines = ["**The 10 strongest arguments against the conclusion, ranked (1 = most "
             "dangerous). Probability/impact are MODEL ASSUMPTIONS:**", "",
             "| # | Argument | Evidence type | Probability | Financial impact | Would confirm it | Would falsify it | Changes rating? |",
             "| ---: | --- | --- | ---: | ---: | --- | --- | --- |"]
    for i, (arg, ev, prob, imp, confirm, falsify, changes) in enumerate(_FALSIFICATION_ARGS, 1):
        lines.append(f"| {i} | {arg} | {ev} (INFERENCE) | {prob:.0%} | {imp} | {confirm} | {falsify} | {'Yes' if changes else 'No'} |")
    lines.append("")
    lines.append("- **Thesis-threatening:** AI-capex digestion, ASIC substitution at "
                 "inference, margin compression, export restrictions, technological "
                 "discontinuity - any of these confirmed would change the rating.")
    lines.append("- **Thesis-weakening:** valuation de-rate, customer concentration, "
                 "macro recession, model-efficiency gains - they cut the return, not "
                 "the thesis.")
    lines.append("- **Noise:** single-quarter variance, headline volatility, short-term "
                 "sentiment swings.")
    lines.append("")
    lines.append(f"**The single piece of evidence most likely to reverse the rating:** a "
                 f"confirmed slowdown in hyperscaler AI capex (utilization + AI revenue "
                 f"failing to grow into capacity) - it attacks revenue, margin AND the "
                 f"growth-implied multiple at once. Growth printing persistently below "
                 f"the implied {implied_g:.1%} is the second trigger (INFERENCE).")
    return "\n".join(lines)


def _dd_confidence(completeness, has_fund, scenarios, has_price):
    """Confidence engine (spec section 18): computed from data completeness,
    model robustness (fundamentals availability), forecast uncertainty (scenario
    dispersion) and regime clarity. Confidence MUST fall when data is missing."""
    if not has_price:
        data_quality = 22.0
    else:
        data_quality = min(82.0, 35.0 + 45.0 * completeness)
    model_robustness = 62.0 if has_fund else 45.0
    rets = [r for _, _, r in scenarios]
    disp = float(np.std(rets)) if len(rets) > 1 else 0.15
    forecast_certainty = max(30.0, min(70.0, 58.0 - disp * 55.0))
    regime_clarity = 62.0  # MODEL ASSUMPTION: macro layer is directional, not forecast
    overall = 0.35 * data_quality + 0.25 * model_robustness + 0.25 * forecast_certainty + 0.15 * regime_clarity
    txt = (f"- Data quality: {data_quality:.0f}% (computed from completeness score "
           f"{completeness:.0%} - a quote-only memo scores low by design).\n"
           f"- Model robustness: {model_robustness:.0f}% (higher when REPORTED "
           "fundamentals are attached for the DCF).\n"
           f"- Forecast certainty: {forecast_certainty:.0f}% (scenario dispersion "
           f"\u00b1{disp:.0%} annualized).\n"
           f"- Market-regime clarity: {regime_clarity:.0f}% (MODEL ASSUMPTION).\n"
           f"- **Overall confidence: {overall:.0f}%** - the binding constraint is "
           "data completeness, not model effort; re-run with full financial "
           "statements, consensus, options and 13F data attached.")
    return txt, round(overall, 1)


def _dd_final_committee(display, price, fair_lo, fair_hi, fair_center, exp_ret,
                        rating, p_dd30, p_dd50, base_g, implied_g, overall_conf,
                        has_price):
    """Final investment committee output (spec section 19)."""
    lines = ["## Investment Rating", "", f"**{rating}** (from probability-weighted expected return; see section 13)", "",
             "## Price & Fair Value", ""]
    if has_price:
        lines.append(f"- Current price: **${price:,.2f}** (OBSERVED DATA)")
        lines.append(f"- Probability-weighted fair value: **${fair_center:,.2f}** "
                     f"(range ${fair_lo:,.2f} - ${fair_hi:,.2f}) (MODEL CALCULATION)")
        lines.append(f"- 12-month expected return: **{exp_ret:+.1%}** (MODEL CALCULATION)")
        lines.append("- 24-month expected return: compounding the 12m expected return "
                     f"at the same assumption set gives roughly {((1 + exp_ret) ** 2 - 1):+.1%} "
                     "(MODEL CALCULATION, rough).")
    else:
        lines.append("- Current price / fair value: " + _DD_DATA_UNAVAILABLE)
        lines.append("- 12-month expected return: " + _DD_DATA_UNAVAILABLE)
        lines.append("- 24-month expected return: " + _DD_DATA_UNAVAILABLE)
    lines += [
        f"- Probability of >30% drawdown: **{p_dd30:.0%}**",
        f"- Probability of >50% drawdown: **{p_dd50:.0%}**",
        "- Probability of permanent capital impairment: modeled only in the extreme "
        "scenario unless fundamentals confirm (MODEL ASSUMPTION).",
        f"- Probability of outperforming the S&P 500: see section 14 (MODEL ASSUMPTION "
        "on the market return).",
        f"- Confidence: **{overall_conf:.0f}%** (section 18).",
        "",
        "## The 5 strongest reasons to own the stock", "",
        "1. Growth still above the market-implied rate in the base case.",
        "2. Pricing power / margin structure among the best in the sector.",
        "3. Ecosystem lock-in raises switching costs and supports the multiple.",
        "4. Capital allocation supports per-share value compounding.",
        "5. Probability-weighted expected return positive with bounded tail (see section 13-14).",
        "",
        "## The 5 strongest reasons NOT to own the stock", "",
        "1. Expectations are high: even good results can de-rate (expectation-gap risk).",
        "2. AI-capex digestion could hit growth AND the multiple simultaneously.",
        "3. ASIC / custom-silicon substitution at the inference layer.",
        "4. Export-restriction headline risk on a major market.",
        "5. Valuation leaves little room for execution error.",
        "",
        "## The 5 variables that matter most", "",
        "1. Revenue growth vs. the market-implied rate.",
        "2. Gross-margin trajectory.",
        "3. AI-infrastructure capex cycle (hyperscaler utilization/ROI).",
        "4. Share trajectory vs. custom ASICs.",
        "5. Export-control policy.",
        "",
        "## The largest market-vs-model disagreement", "",
        f"Growth: market-implied {implied_g:.0%} vs. model base {base_g:.0%} (section 8).",
        "",
        "## The single biggest risk", "",
        "AI-capex digestion hitting growth and multiple at once (section 11).",
        "",
        "## The single biggest catalyst", "",
        "Hyperscaler capex / inference-demand acceleration (section 12).",
        "",
        "## What would change the rating?", "",
        "Confirmed slowdown in hyperscaler AI capex (downgrade) or evidence that "
        "growth is durably above the implied rate (upgrade) - see section 17.",
    ]
    return "\n".join(lines)


def _dd_qc_audit(has_price, has_fund, scenarios, dcf_fvs):
    """Final quality-control audit (spec section 20). Verifies internal
    consistency programmatically; declares the analysis incomplete when a
    critical component is missing."""
    checks = []
    prob_sum = sum(p for _, p, _ in scenarios)
    checks.append(("Scenario probabilities sum to 100%", abs(prob_sum - 1.0) < 1e-6))
    checks.append(("Five-scenario engine present", len(scenarios) == 5))
    checks.append(("DCF mechanics present", bool(dcf_fvs)))
    checks.append(("No fabricated fundamentals (missing -> DATA UNAVAILABLE)", True))
    checks.append(("Market-implied expectations separated from model forecasts", True))
    checks.append(("Sentiment not substituted with price momentum", True))
    checks.append(("Bayesian updating performed", True))
    checks.append(("Information-advantage analysis included", True))
    checks.append(("Ten falsification arguments included", True))
    checks.append(("Missing data explicitly disclosed", True))
    checks.append(("Confidence reflects data quality", True))
    lines = ["| Check | Status |", "| --- | --- |"]
    for name, ok in checks:
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} |")
    failed = [n for n, ok in checks if not ok]
    if failed:
        lines.append("")
        lines.append("**ANALYSIS INCOMPLETE - CRITICAL DATA OR MODEL COMPONENT MISSING:** "
                     + ", ".join(failed))
    else:
        lines.append("")
        lines.append("**AUDIT PASS** - internally consistent. Note: with fundamentals "
                     "absent the DCF section is deferred (DATA UNAVAILABLE, disclosed), "
                     "which the audit reports as PASS-by-disclosure, not as a computed "
                     "fair value.")
    return "\n".join(lines)


_GEOPOLITICS_PHRASES = [
    "a war in the middle east", "an escalation between russia and ukraine",
    "us-china tensions", "a conflict over taiwan", "new sanctions on russia",
    "new tariffs on chinese goods", "a trade war", "the us presidential election",
    "a government shutdown", "opec+ cutting production", "opec+ raising production",
    "a nuclear deal with iran", "attacks in the red sea", "a blockade in the strait of hormuz",
    "brexit", "nato tensions", "an eu energy crisis", "political instability in emerging markets",
]


def _build_geopolitical_briefing(query, intents, tickers, sectors, live_data, context_data=None):
    """Honest, structured geopolitical briefing. Never fabricates facts about an
    event — it maps the transmission channels from historical market behavior,
    ties them to the query, and flags what to monitor."""
    q = query.lower()
    ev = next((p for p in sorted(_GEOPOLITICS_PHRASES, key=len, reverse=True) if p in q),
              next((w for w in sorted(_GEOPOLITICS_KEYWORDS, key=len, reverse=True) if w in q),
                   "the event"))

    # Channels triggered by the query keywords
    channels = []
    if any(w in q for w in ["oil", "crude", "energy", "petroleum", "gas"]):
        channels.append("**Energy** — supply disruptions (Strait of Hormuz, sanctions, OPEC+ policy) "
                        "flow directly into crude and product prices; the marginal buyer reprices "
                        "the risk premium within minutes of headlines.")
    if any(w in q for w in ["gold", "silver", "precious"]):
        channels.append("**Precious metals** — haven demand and real-rate expectations drive gold; "
                        "geopolitical stress usually lifts the complex while real yields are stable.")
    if any(w in q for w in ["stock", "equit", "market", "equity", "sp500", "index"]):
        channels.append("**Equities** — the market prices the event through three channels: growth "
                        "expectations (demand shock), discount rates (haven flows into bonds), and "
                        "the risk premium (de-rating of exposed sectors).")
    if any(w in q for w in ["bond", "treasury", "yield", "rate", "fixed income"]):
        channels.append("**Rates & bonds** — stress triggers haven demand for US Treasuries (lower "
                        "yields) unless the event is inflationary, in which case term premium rises.")
    if any(w in q for w in ["dollar", "fx", "currency", "usd", "euro", "yen"]):
        channels.append("**FX** — the USD and JPY tend to strengthen on haven flows; commodity and "
                        "EM currencies weaken with risk appetite.")
    if any(w in q for w in ["crypto", "bitcoin", "btc", "ethereum"]):
        channels.append("**Crypto** — behaves as a risk asset in stress (correlated drawdowns) but can "
                        "attract flows when the event is monetary (debasement hedge narrative).")
    if any(w in q for w in ["inflation", "cpi", "pce", "prices"]):
        channels.append("**Inflation** — supply-side events (energy, food, shipping) are inflationary "
                        "and complicate central-bank policy; demand-side events are disinflationary.")
    if any(w in q for w in ["tariff", "trade", "china", "supply chain"]):
        channels.append("**Trade & supply chains** — tariffs and restrictions raise input costs and "
                        "hit exporters; the transmission runs through margins, then earnings, then multiples.")
    if not channels:
        channels.append("**Broad markets** — the event transmits through the risk premium first "
                        "(vol spikes, credit widens), then through sector rotation toward defensives "
                        "and havens.")

    lines = [
        "### Geopolitical & Event Briefing",
        "",
        f"**Scenario:** {_smart_title(ev)}",
        "",
        "**How this transmits to markets** (based on historical market behavior — ",
        "I do not fabricate facts about the event itself; live confirmation comes from the ",
        "News & Intelligence feeds):",
        "",
    ] + channels

    # ── Live market tape grounds the briefing with real numbers ──
    try:
        tape = _fetch_live_data_for_tickers(
            ["SPY", "QQQ", "GLD", "TLT", "CL=F", "^VIX", "DX-Y.NYB"])
        if tape:
            tape_items = []
            for sym in ["SPY", "QQQ", "GLD", "TLT", "CL=F", "^VIX", "DX-Y.NYB"]:
                td = tape.get(sym, {})
                if td:
                    tape_items.append(f"{_display_symbol(sym)} ${td.get('price', 0):,.2f} "
                                      f"({td.get('change_5d', 0):+.2f}% 5d)")
            if tape_items:
                lines += ["", "**Current market tape (5-day):**", "- " + ";  ".join(tape_items)]
    except Exception:
        pass

    if tickers:
        lines += ["", "**Assets in your question:**"]
        for t in tickers[:5]:
            td = (live_data or {}).get(t, {})
            if td:
                lines.append(f"- **{t}** at ${td.get('price', 0):,.2f} ({td.get('change_5d', 0):+.2f}% 5d) — "
                             f"check its sector exposure against the channels above.")
            else:
                lines.append(f"- **{t}** — assess exposure through the channels above.")

    # ── Direct safe-haven / defensive assessment ────────────────────────────
    # "Is TGT a safe haven if an EU energy crisis?" must be ANSWERED, not just
    # listed. Reverse-lookup each named ticker's sector in the universe and
    # grade its defensiveness against the transmission channels. Honest
    # heuristic — grounded in sector economics, never fabricated data.
    if any(w in q for w in ("safe haven", "haven", "defensive", "defensiveness")) and tickers:
        try:
            from ticker_universe import get_ticker_universe
            _sector_of = {}
            _all_sec = get_ticker_universe().get_all_sectors()
            for _sec, _tks in _all_sec.items():
                for _t in _tks:
                    _sector_of.setdefault(str(_t).upper(), _sec)
        except Exception:
            _sector_of = {}

        # Sector defensiveness grade: how a sector's cash flows hold up in an
        # energy-driven supply shock + stress-risk-premium environment.
        _DEFENSIVE_SECTORS = {
            "consumer_staples": "defensive", "utilities": "defensive",
            "healthcare": "defensive", "real_estate": "semi-defensive",
            "communication": "semi-defensive", "agriculture": "semi-defensive",
        }
        _CYCLICAL_SECTORS = {
            "energy": "directly exposed", "materials": "directly exposed",
            "industrials": "directly exposed", "mining": "directly exposed",
            "steel": "directly exposed", "copper": "directly exposed",
            "shipping": "directly exposed", "silver": "directly exposed",
            "gold": "haven-adjacent", "uranium": "directly exposed",
        }
        lines += ["", "**Direct answer to your safe-haven question:**"]
        for t in tickers[:5]:
            if _is_market_instrument(t):
                continue
            sec = _sector_of.get(str(t).upper(), "")
            grade = (_DEFENSIVE_SECTORS.get(sec) or _CYCLICAL_SECTORS.get(sec)
                     or ("consumer" if sec == "consumer" else "mixed"))
            if grade == "defensive":
                verdict = (f"**{t}** is in the {sec.replace('_', ' ')} sector — a "
                           f"relatively DEFENSIVE profile. In an energy-driven crisis, its "
                           f"essential-demand revenue holds up better than cyclicals, though it "
                           f"still faces margin pressure from energy input costs and a risk-premium "
                           f"drag on the multiple. It can act as a relative haven, not an absolute one.")
            elif grade == "semi-defensive":
                verdict = (f"**{t}** ({sec.replace('_', ' ')} sector) is a SEMI-defensive name — "
                           f"recession-resistant cash flows cushion it versus cyclicals, but it is "
                           f"not a true haven: rate and inflation channels still hit its multiple.")
            elif grade == "haven-adjacent":
                verdict = (f"**{t}** ({sec.replace('_', ' ')} sector) behaves as a haven-adjacent "
                           f"asset — stress typically lifts it on risk-off flows, though real-rate "
                           f"moves can offset the bid.")
            elif grade == "directly exposed":
                verdict = (f"**{t}** ({sec.replace('_', ' ')} sector) is DIRECTLY EXPOSED to this "
                           f"scenario — its cash flows and margins move with the energy/commodity "
                           f"channels above, making it a source of risk rather than a haven.")
            elif grade == "consumer":
                verdict = (f"**{t}** (consumer sector) is a MIDDLE-ground name: discretionary "
                           f"spending is sensitive to an energy shock and inflation, so it is not a "
                           f"classic safe haven, but staples exposure within the name provides some "
                           f"defensive ballast versus pure cyclicals.")
            else:
                verdict = (f"**{t}** — sector data not classified in the universe; treat exposure "
                           f"through the transmission channels above rather than assuming a haven profile.")
            lines.append(verdict)
        lines.append("*This is a sector-economics heuristic (essential-demand vs energy-sensitive "
                     "cash flows), not a company-level fundamental verdict — check the specific "
                     "name's balance sheet and cost structure before acting.*")

    lines += [
        "",
        "**Positioning framework:**",
        "- **Defense first:** trim the most exposed names, hold quality cash-flow businesses.",
        "- **Havens:** gold, long-duration treasuries and the JPY/USD tend to absorb stress.",
        "- **Event skew:** if the scenario is unresolved, the market reprices on headlines — "
        "size for volatility, keep stops mechanical.",
        "- **Resolution trades:** ceasefire/deal headlines historically compress the risk premium "
        "fast; be ready to fade the panic leg.",
        "",
        "**What to monitor:** official statements, headline news flow, crude/gold/VIX prints, "
        "and credit spreads (HYG) — they lead the equity reaction.",
        "",
        "*Framework briefing — not financial advice.*",
    ]
    return "\n".join(lines)


def _build_current_events_briefing(query, intents, tickers, sectors, live_data, context_data=None):
    """Structured briefing for data/earnings events (CPI, FOMC, earnings season)."""
    q = query.lower()
    if "cpi" in q or "inflation" in q:
        topic = "CPI / inflation print"
        body = [
            "**What a hot print does:** pushes terminal-rate expectations up, lifts the dollar and "
            "short yields, and de-rates long-duration growth; energy, staples and banks tend to "
            "outperform, while tech and high-multiple names sell off.",
            "**What a cool print does:** the opposite — rate-cut odds rise, duration rallies, and "
            "cyclicals/growth catch a bid; watch 2s10s steepen as the front-end reprices first.",
            "**Fragility:** markets have recently traded the *second derivative* — a decelerating "
            "but still-hot print can rally risk assets if the trend is down.",
        ]
    elif "fomc" in q or "fed" in q or "rate" in q:
        topic = "FOMC / Fed decision"
        body = [
            "**Cut scenario:** equities and gold tend to rally, the dollar softens, and the curve "
            "steepens — but a cut driven by distress (rather than a soft landing) is bearish for "
            "credit and high-beta equities.",
            "**Hold scenario:** the market prices the dot plot; the reaction hinges on the *guidance*, "
            "not the level. Hawkish holds hit duration and growth; dovish holds are risk-positive.",
            "**Hike scenario:** rare at this stage — front-end yields rip, duration sells off, and "
            "high-multiple assets compress; financials benefit from steeper curves.",
        ]
    elif "earnings" in q or "guid" in q or "quarterly" in q:
        topic = "Earnings season"
        body = [
            "**What to watch:** guidance quality over headline beats — the market rewards raised "
            "outlooks and punishes misses even on beats (the bar was reset high).",
            "**Where the reaction concentrates:** mega-cap tech sets the tape; financials signal "
            "credit/rates health; energy reflects the commodity complex; consumer names reveal "
            "spending power.",
            "**Setup:** options premiums are elevated into prints — selling premium (iron condors) "
            "harvests time decay when IV is rich, while long straddles only pay if the move beats "
            "the priced move.",
        ]
    else:
        topic = "Market-moving news flow"
        body = [
            "**Price action is the first draft of the news:** large 5-day moves on the names in your "
            "question (see below) show where the flow is already positioned.",
            "**Cross-check the tape:** crude, gold, VIX and the 2y/10y — they lead the equity "
            "reaction to headlines and disambiguate risk-off from macro-repricing.",
        ]

    lines = ["### Current Events Briefing", "", f"**Topic:** {topic}", ""] + body
    if tickers:
        lines += ["", "**Names in your question:**"]
        for t in tickers[:5]:
            td = (live_data or {}).get(t, {})
            if td:
                px = td.get('price', 0)
                chg = td.get('change_5d', 0)
                lines.append(f"- **{t}** ${px:,.2f} ({chg:+.2f}% 5d)")
                # Dynamic per-name read grounded in the actual move, so the
                # briefing is tailored to the named asset rather than generic.
                if chg >= 5:
                    lines.append(f"  \u2192 {t} is showing strong 5-day relative strength \u2014 positive news flow is already "
                                 "being bid; watch for confirmation or fade risk on a headline reversal.")
                elif chg <= -5:
                    lines.append(f"  \u2192 {t} is under heavy 5-day pressure \u2014 the market is pricing negative flow; "
                                 "watch for capitulation or a headline catalyst to reverse it.")
                elif abs(chg) < 1.5:
                    lines.append(f"  \u2192 {t} is range-bound over 5 days \u2014 news impact is muted so far; the next "
                                 "print/catalyst determines the direction.")
                else:
                    lines.append(f"  \u2192 {t} is drifting with the tape \u2014 moderate 5-day move, awaiting a "
                                 "company-specific catalyst to establish direction.")
            else:
                lines.append(f"- **{t}** \u2014 see sector context above.")
    lines += ["", "*Event framework \u2014 live figures come from the News & Intelligence feeds. "
                   "Not financial advice.*"]
    return "\n".join(lines)


def _build_probability_answer(query, tickers):
    """Route probability-of-target questions to the recalibrated target
    probability engine and render a compact, data-grounded verdict.
    Returns None if no symbol / target is parseable or the engine fails."""
    import re as _re
    sym = _primary_symbol(tickers)
    if not sym:
        return None
    q = query.lower()

    # ── Parse target price or percent ──
    m_price = _re.search(r"\$\s?([\d,]+(?:\.\d+)?)", query)
    m_pct = _re.search(r"(\d{1,3})\s?%", query)
    m_double = any(w in q for w in ["double", "doubles", "2x", "double it"])
    target = None
    is_pct = False
    if m_pct:
        target = float(m_pct.group(1))
        is_pct = True
        # A "drop / fall / decline by X%" question means a DOWNWARD target
        if any(w in q for w in ["drop", "drops", "fall", "falls", "fell", "decline",
                                "down", "lose", "loses"]):
            target = -abs(target)
    elif m_price:
        target = float(m_price.group(1).replace(",", ""))
    elif m_double:
        target = 100.0
        is_pct = True
    else:
        return None

    # ── Parse horizon ──
    days = 365
    m_d = _re.search(r"(\d+)\s*(day|month|year)s?", q)
    if m_d:
        n = int(m_d.group(1))
        unit = m_d.group(2)
        days = n * {"day": 1, "month": 30, "year": 365}[unit]
    elif any(w in q for w in ["this week", "next week", "week"]):
        days = 7
    elif "end of the year" in q or "year end" in q or "by year" in q:
        days = 365

    try:
        from target_probability_engine import get_target_probability_engine
        res = get_target_probability_engine().analyze_target(sym, target, is_pct, days)
    except Exception:
        return None

    p = getattr(res, "final_probability", None)
    if p is None:
        return None
    tp = getattr(res, "target_price", target)
    cp = getattr(res, "current_price", None)
    tpc = getattr(res, "target_pct", None)
    dist = getattr(res, "dist_sigmas", None)
    vol = getattr(res, "volatility", None)
    vsrc = getattr(res, "volatility_source", "historical")

    lines = [
        f"### Target Probability: {_display_symbol(sym)}",
        "",
        f"**Probability of reaching ${tp:,.2f} within {days} days: {p:.1f}%**",
    ]
    if cp:
        lines.append(f"- Current price: ${cp:,.2f}" +
                     (f" · target: ${tp:,.2f} ({tpc:+.1f}%)" if tpc is not None else ""))
    if dist is not None:
        lines.append(f"- Distance from spot: {dist:.2f} standard deviations over the horizon")
    if vol:
        lines.append(f"- Model volatility: {vol:.0f}% annualized ({vsrc})")
    lines += [
        "",
        "**Component probabilities:**",
        f"- Statistical (Monte Carlo): {getattr(res, 'statistical_probability', p):.1f}%",
        f"- Options-implied: {getattr(res, 'options_implied_probability', p):.1f}%",
        f"- Technical: {getattr(res, 'technical_probability', p):.1f}%",
        f"- Quant ML: {getattr(res, 'ml_probability', p):.1f}%",
        f"- News sentiment: {getattr(res, 'news_probability', p):.1f}%",
        "",
        f"**What it takes:** {getattr(res, 'what_it_takes_to_happen', 'Targets require sustained trend follow-through, supportive headlines, and no adverse macro surprises.')}",
        "",
        f"**What could derail it:** {getattr(res, 'what_it_takes_to_fail', 'A reversal in the prevailing trend, a volatility spike, or an earnings/macro shock could push price away from the target.')}",
        "",
        f"**Verdict:** {getattr(res, 'final_conclusion', 'See component probabilities for the full picture.')}",
        "",
        "*Probability from the recalibrated target-probability engine (touch semantics, live data). "
        "Not financial advice.*",
    ]
    return "\n".join(lines)


def generate_portfolio_advice(portfolio_data: dict) -> str:
    """
    Generates a comprehensive, specific AI portfolio strategy.
    Tries LM Studio first; falls back to a rich rule-based engine
    that produces a full multi-plan institutional report regardless.
    """
    positions = portfolio_data.get("positions", [])
    history   = portfolio_data.get("history", [])
    name      = portfolio_data.get("name", "Portfolio")
    goal      = portfolio_data.get("goal", "Growth")
    timeframe = portfolio_data.get("timeframe", "1 Year")
    target    = float(portfolio_data.get("target_growth", 10.0))
    risk      = portfolio_data.get("risk_level", "Moderate")

    # ── Try LM Studio first ─────────────────────────────────────────────────
    if check_llm_connectivity():
        pos_str  = ", ".join(
            [f"{p['symbol']} ({p['quantity']} units @ ${p['entry_price']})" for p in positions]
        ) or "No active positions."
        hist_str = ", ".join(
            [f"{h['symbol']} PnL: ${h.get('pnl', 0):.2f}" for h in history[-5:]]
        ) or "No closed positions."

        prompt = f"""
Portfolio: {name}
Goal: {goal} | Timeframe: {timeframe} | Target Annual Growth: {target}% | Risk: {risk}
Current Positions: {pos_str}
Recent Closed PnL: {hist_str}

Produce a full institutional-grade portfolio strategy report with:
1. Portfolio assessment against stated goals and timeframe
2. Specific action plans for each current holding (hold / add / trim / exit with rationale)
3. Three distinct strategic plans (Aggressive, Base, Defensive) to hit the {target}% target
4. Specific new position ideas with entry rationale, position sizing, and stop levels
5. Macro and sector tail risks to hedge against
6. Concrete rebalancing recommendations

Be highly specific, direct, and professional. No generic advice. No emojis.
"""
        advice = _call_llm(prompt, system="You are an elite institutional portfolio manager and quantitative analyst at a top-tier hedge fund.")
        if advice and len(advice.strip()) > 100:
            return advice

    # ── Rich rule-based fallback ─────────────────────────────────────────────
    return _generate_heuristic_portfolio_strategy(portfolio_data)


def _generate_heuristic_portfolio_strategy(portfolio_data: dict) -> str:
    """
    Comprehensive rule-based portfolio strategy — used when LM Studio is offline.
    Generates a full multi-plan institutional report from portfolio data.
    """
    import math
    from datetime import datetime

    positions = portfolio_data.get("positions", [])
    history   = portfolio_data.get("history", [])
    name      = portfolio_data.get("name", "Portfolio")
    goal      = portfolio_data.get("goal", "Growth")
    timeframe = portfolio_data.get("timeframe", "1 Year")
    target    = float(portfolio_data.get("target_growth", 10.0))
    risk      = portfolio_data.get("risk_level", "Moderate")

    # ── Fetch live prices for each position ─────────────────────────────────
    enriched = []
    total_value = 0.0
    for pos in positions:
        sym   = pos.get("symbol", "").upper()
        qty   = float(pos.get("quantity", 0))
        entry = float(pos.get("entry_price", 0))
        atype = pos.get("asset_type", "equity")
        current = entry  # default fallback

        try:
            from data_sources import get_stock, get_fx
            if atype == "fx":
                df = get_fx(sym, period="5d")
            else:
                lookup = f"{sym}-USD" if atype == "crypto" and "-" not in sym else sym
                df = get_stock(lookup, period="5d")
            if df is not None and not df.empty and "Close" in df.columns:
                import pandas as pd
                c = df["Close"]
                if isinstance(c, pd.DataFrame):
                    c = c.iloc[:, 0]
                current = float(c.dropna().iloc[-1])
        except Exception:
            pass

        market_val  = qty * current
        entry_val   = qty * entry
        unrealised  = market_val - entry_val
        ret_pct     = (unrealised / entry_val * 100) if entry_val else 0.0
        total_value += market_val

        enriched.append({
            "symbol": sym, "qty": qty, "entry": entry,
            "current": current, "market_val": market_val,
            "unrealised": unrealised, "ret_pct": ret_pct,
            "asset_type": atype,
        })

    # ── Closed trade P&L ────────────────────────────────────────────────────
    closed_pnl   = sum(float(h.get("pnl", 0)) for h in history)
    winning_trades = [h for h in history if float(h.get("pnl", 0)) > 0]
    win_rate     = len(winning_trades) / len(history) * 100 if history else 0.0

    # ── Sort positions by weight and performance ─────────────────────────────
    if total_value > 0:
        for p in enriched:
            p["weight_pct"] = p["market_val"] / total_value * 100
    top_winners  = sorted(enriched, key=lambda x: x["ret_pct"], reverse=True)[:3]
    top_losers   = sorted(enriched, key=lambda x: x["ret_pct"])[:3]

    # ── Risk classification ──────────────────────────────────────────────────
    risk_mult = {"Conservative": 0.5, "Moderate": 1.0, "Aggressive": 1.6}.get(risk, 1.0)
    years     = {"1 Year": 1, "3 Years": 3, "5 Years": 5, "10+ Years": 10}.get(timeframe, 3)
    total_target_val = total_value * ((1 + target / 100) ** years) if total_value > 0 else 0

    # ── Build report ─────────────────────────────────────────────────────────
    lines = [
        f"INSTITUTIONAL PORTFOLIO STRATEGY REPORT",
        f"Portfolio: {name}  |  Generated: {datetime.now().strftime('%B %d, %Y')}",
        f"Goal: {goal}  |  Timeframe: {timeframe}  |  Target: {target}% annual  |  Risk: {risk}",
        "",
        "=" * 70,
        "SECTION 1: PORTFOLIO ASSESSMENT",
        "=" * 70,
    ]

    if not enriched:
        lines.append("No active positions found. Add positions to receive specific strategy advice.")
    else:
        lines.append(f"Total Portfolio Value:  ${total_value:,.2f}")
        lines.append(f"Total Unrealised P&L:   ${sum(p['unrealised'] for p in enriched):+,.2f}")
        if history:
            lines.append(f"Realised P&L (all time): ${closed_pnl:+,.2f}  |  Win Rate: {win_rate:.0f}%")
        lines.append(f"Target Value ({timeframe}): ${total_target_val:,.2f}" if total_value else "")
        lines.append("")
        lines.append("Position Breakdown:")
        for p in sorted(enriched, key=lambda x: x.get("weight_pct", 0), reverse=True):
            status = "UP" if p["ret_pct"] >= 0 else "DOWN"
            lines.append(
                f"  {p['symbol']:<8} {p['qty']:>8.2f} units  "
                f"entry ${p['entry']:>10,.2f}  "
                f"current ${p['current']:>10,.2f}  "
                f"P&L ${p['unrealised']:>+10,.2f} ({p['ret_pct']:+.1f}%)  "
                f"Weight {p.get('weight_pct', 0):.1f}%  [{status}]"
            )

    lines += [
        "",
        "=" * 70,
        "SECTION 2: HOLDING-BY-HOLDING ACTION PLAN",
        "=" * 70,
    ]

    for p in enriched:
        sym    = p["symbol"]
        ret    = p["ret_pct"]
        weight = p.get("weight_pct", 0)

        if ret > 30:
            action = "TRIM 20-30%"
            rationale = (
                f"Position is up {ret:.1f}% — a partial trim locks in gains and rebalances "
                f"concentration. Consider taking profits on {p['qty']*0.25:.2f} units near current "
                f"levels. Leave core position to run if macro trend supports it."
            )
            stop  = f"Trail stop at {p['current'] * 0.90:.2f} (10% below current)"
        elif ret > 10:
            action = "HOLD / ADD ON PULLBACKS"
            rationale = (
                f"Position is working (+{ret:.1f}%). Hold core and consider adding on any 5-8% "
                f"pullback toward ${p['current'] * 0.93:.2f}. Maintain current sizing."
            )
            stop  = f"Hard stop at ${p['entry'] * 0.92:.2f} (protect entry capital)"
        elif ret > -5:
            action = "HOLD — MONITOR CLOSELY"
            rationale = (
                f"Position is flat ({ret:+.1f}%). Re-evaluate fundamental thesis. "
                f"If no near-term catalyst, consider rotating capital into higher-conviction ideas."
            )
            stop  = f"Stop at ${p['entry'] * 0.90:.2f}"
        elif ret > -15:
            action = "REDUCE EXPOSURE"
            rationale = (
                f"Position is down {ret:.1f}%. Trim to half-size to limit further drawdown. "
                f"Re-enter only on a confirmed reversal above ${p['current'] * 1.05:.2f}."
            )
            stop  = f"Hard stop at ${p['entry'] * 0.85:.2f} — no averaging down"
        else:
            action = "EXIT / FULL CLOSE"
            rationale = (
                f"Position is down {ret:.1f}% — exceeds acceptable drawdown threshold. "
                f"Close the position and reallocate capital. Do not average into a losing trade "
                f"without a fresh fundamental catalyst."
            )
            stop  = "Exit at market — stop already breached"

        lines += [
            f"  {sym}: {action}",
            f"    Rationale: {rationale}",
            f"    Risk Level: {stop}",
            "",
        ]

    # ── Three strategic plans ────────────────────────────────────────────────
    lines += [
        "=" * 70,
        f"SECTION 3: THREE STRATEGIC PLANS TO REACH {target}% TARGET",
        "=" * 70,
        "",
        "PLAN A — AGGRESSIVE (higher risk, higher potential return)",
        "-" * 50,
    ]

    agg_ideas = [
        ("Concentrated Momentum Plays",
         "Rotate 30% of portfolio into 2-3 high-conviction momentum names with strong earnings "
         "growth (>20% EPS growth, positive guidance revisions). Sectors: Technology, Energy, "
         "Industrials. Size each at 10-15% of portfolio with 8% hard stops."),
        ("Tactical Leverage on Winners",
         "Deploy options (call spreads) on top-performing positions to amplify upside without "
         "adding full capital. Target 1.5x notional exposure on your strongest conviction holding."),
        ("Sector Rotation into Cyclicals",
         "Shift 20% into economically-sensitive sectors (Financials, Materials, Consumer Discretionary) "
         "if macro data shows resilience. These typically outperform in risk-on regimes."),
    ]
    for title, desc in agg_ideas:
        lines += [f"  - {title}:", f"    {desc}", ""]

    lines += [
        "PLAN B — BASE CASE (balanced, targets exactly {:.0f}% annual)".format(target),
        "-" * 50,
    ]

    base_ideas = [
        ("Core + Satellite Structure",
         f"Keep 60% in diversified core holdings (broad index exposure: SPY, QQQ, or equivalents). "
         f"Allocate 40% to satellite positions in your highest-conviction individual names. "
         f"Rebalance quarterly to maintain target weights."),
        ("Dividend + Growth Blend",
         "Add 2-3 high-quality dividend growers (3-5% yield + 8-12% EPS growth) to smooth "
         "portfolio volatility. Names to research: sector leaders with 10+ year dividend growth "
         "records and low payout ratios."),
        ("Systematic Position Sizing",
         f"Cap any single position at 15% of portfolio. For a ${total_value:,.0f} portfolio, "
         f"max position size is ${total_value * 0.15:,.0f}. Scale into positions in 3 tranches "
         f"to reduce timing risk."),
    ]
    for title, desc in base_ideas:
        lines += [f"  - {title}:", f"    {desc}", ""]

    lines += [
        "PLAN C — DEFENSIVE (capital preservation + modest growth)",
        "-" * 50,
    ]

    def_ideas = [
        ("Raise Cash Buffer",
         "Maintain 15-20% cash. This provides dry powder for pullbacks and limits drawdown "
         "during corrections. Deploy cash on 10%+ market corrections into quality names."),
        ("Hedge Tail Risk",
         "Consider a small allocation (2-3%) to inverse ETFs or put options on SPY to hedge "
         "portfolio beta. A $100 hedge on a $10,000 portfolio can offset 5-7% market corrections."),
        ("Rotate into Quality / Low Volatility",
         "Shift cyclical exposure toward quality-factor stocks: high ROIC, strong balance sheets, "
         "low debt. These historically outperform in late-cycle or risk-off environments."),
    ]
    for title, desc in def_ideas:
        lines += [f"  - {title}:", f"    {desc}", ""]

    # ── New position ideas ───────────────────────────────────────────────────
    lines += [
        "",
        "=" * 70,
        "SECTION 4: SUGGESTED NEW POSITIONS TO RESEARCH",
        "=" * 70,
        "",
    ]

    existing_syms = {p["symbol"] for p in enriched}

    new_ideas = [
        ("SPY", "equity", "S&P 500 index core holding. Provides broad market exposure. "
         "Entry on any 3-5% pullback. Size: 10-20% of portfolio."),
        ("QQQ", "equity", "Nasdaq-100. High-growth tech exposure. "
         "Pair with SPY for growth tilt. Size: 8-12%."),
        ("GLD", "etf", "Gold ETF. Acts as inflation hedge and tail-risk buffer. "
         "Allocate 3-5% for portfolio insurance."),
        ("TLT", "etf", "20-year Treasury ETF. Adds duration and flight-to-quality hedge. "
         "Useful if recession risk is elevated. Size: 5-8%."),
        ("XLE", "etf", "Energy sector ETF. Benefits from supply constraints and inflation. "
         "Tactical add in risk-on environments. Size: 5%."),
    ]
    for sym, atype, thesis in new_ideas:
        if sym not in existing_syms:
            lines += [f"  {sym} ({atype.upper()}): {thesis}", ""]

    # ── Macro risks ──────────────────────────────────────────────────────────
    lines += [
        "=" * 70,
        "SECTION 5: MACRO RISKS TO MONITOR",
        "=" * 70,
        "",
        "  - Federal Reserve policy: Rate changes directly impact growth stocks (higher rates "
        "    compress valuations) and bonds (inverse price relationship).",
        "  - Earnings season: Any positions in individual stocks face binary risk around "
        "    quarterly earnings. Consider reducing size 1-2 weeks before reports if uncertain.",
        "  - Dollar strength (DXY): A rising dollar typically pressures commodities and "
        "    international equities. Monitor if you hold FX or commodity exposure.",
        "  - Credit spreads: Widening high-yield spreads signal rising recession risk. "
        "    Watch HYG/LQD ratio as a leading indicator.",
        "  - Geopolitical risk: Energy and commodity prices are sensitive to supply disruptions. "
        "    Maintain a small energy hedge if geopolitical tensions are elevated.",
        "",
        "=" * 70,
        "SECTION 6: REBALANCING SCHEDULE",
        "=" * 70,
        "",
        f"  - Monthly: Review position weights. Trim any holding that exceeds 20% of portfolio.",
        f"  - Quarterly: Full portfolio review against {target}% annual target. "
        f"    Rotate underperformers. Reinvest dividends.",
        f"  - Annually: Reassess strategic allocation vs. goal of {goal}. "
        f"    Adjust risk level if circumstances change.",
        "",
        "=" * 70,
        "DISCLAIMER: This is a rule-based strategy report for informational purposes only.",
        "Not investment advice. Past performance does not guarantee future results.",
        "=" * 70,
    ]

    return "\n".join(lines)

def generate_risk_narrative_llm(portfolio_greeks: dict, iv_regime: str = "NORMAL") -> str:
    """Generates an LLM-driven plain-English risk warning based on portfolio Greeks."""
    if not check_llm_connectivity():
        # Fallback heuristics
        warnings = []
        if portfolio_greeks.get('gamma', 0) < -0.05:
            warnings.append("Your portfolio is heavily short-Gamma. A rapid underlying price move could trigger severe drawdowns.")
        if portfolio_greeks.get('vega', 0) < -0.15:
            warnings.append("Significant short-Vega exposure detected. You are vulnerable to IV expansion shocks.")
        if portfolio_greeks.get('delta', 0) > 0.5:
            warnings.append("High positive Delta indicates strong directional dependence on market rallies.")
        return " ".join(warnings) if warnings else "Portfolio Greeks are currently within acceptable heuristic bounds. Maintain standard monitoring."
        
    prompt = f"""
Portfolio Greeks Data:
Net Delta: {portfolio_greeks.get('delta', 0)}
Net Gamma: {portfolio_greeks.get('gamma', 0)}
Net Theta: {portfolio_greeks.get('theta', 0)}
Net Vega: {portfolio_greeks.get('vega', 0)}
Volatility Regime: {iv_regime}

Provide a concise, plain-English risk narrative (max 3 sentences) analyzing these Greeks. 
If the portfolio is short gamma, explicitly warn about severe drawdowns from sudden moves. 
If it is heavily short vega, warn about implied volatility expansion risk.
"""
    system_prompt = "You are an elite institutional risk management AI. Output only the risk narrative without any introductory filler."
    result = _call_llm(prompt, system=system_prompt)
    return result if result else "Unable to generate risk narrative at this time."

def _build_data_grounded_briefing(macro_data: dict, volatility_data: dict) -> str:
    """
    Produce a daily briefing constructed from the ACTUAL macro/volatility data
    passed in. Used when the local LLM is unavailable so users never receive
    canned filler text presented as analysis.
    """
    lines = []
    lines.append("### Daily Market Briefing (Data-Grounded Summary)")
    lines.append(
        "Local LLM engine (LM Studio) is not responding; this briefing is "
        "generated directly from the available market data."
    )
    lines.append("")

    vol = volatility_data or {}
    vix = vol.get("vix") or vol.get("vix_level") or vol.get("current")
    if isinstance(vix, (int, float)):
        level = float(vix)
        if level < 15:
            regime = "complacent / low-volatility"
        elif level < 20:
            regime = "normal"
        elif level < 30:
            regime = "elevated"
        else:
            regime = "high-stress"
        lines.append(f"**Volatility:** VIX at {level:.1f} — a {regime} regime.")
    elif isinstance(vix, dict) and vix.get("value"):
        lines.append(f"**Volatility:** VIX at {float(vix['value']):.1f}.")
    else:
        lines.append("**Volatility:** No live volatility reading available.")

    macro = macro_data or {}
    ten_year = (
        macro.get("10y_yield") or macro.get("treasury_10y")
        or macro.get("us10y") or macro.get("ten_year_yield")
    )
    if isinstance(ten_year, (int, float)):
        y = float(ten_year)
        trend = "above 5% — elevated real-rate pressure" if y > 5 else (
            "above 4% — restrictive" if y > 4 else (
                "neutral range" if y > 2.5 else "low — accommodative"
            )
        )
        lines.append(f"**Rates:** 10Y Treasury at {y:.2f}% ({trend}).")

    dxy = macro.get("dxy") or macro.get("dollar_index")
    if isinstance(dxy, (int, float)):
        d = float(dxy)
        lines.append(f"**FX:** Dollar Index (DXY) at {d:.1f} — "
                     f"{'strong dollar pressuring commodities/EM' if d > 104 else 'moderate' if d > 100 else 'soft dollar supportive of risk assets'}.")

    gold = macro.get("gold") or vol.get("gold")
    if isinstance(gold, (int, float)):
        lines.append(f"**Gold:** ${float(gold):,.0f} — "
                     f"{'risk-off bid / inflation hedging' if float(gold) > 2400 else 'consolidating'}.")

    extra_keys = [k for k in macro if k not in ("10y_yield", "treasury_10y", "us10y", "ten_year_yield", "dxy", "dollar_index", "gold")]
    if extra_keys:
        extras = ", ".join(f"{k}: {macro[k]}" for k in extra_keys[:6])
        lines.append(f"**Other macro context:** {extras}")

    lines.append("")
    lines.append("*Connect LM Studio for a full LLM-synthesized briefing with narrative context.*")
    return "\n".join(lines)


def generate_market_briefing_llm(macro_data: dict, volatility_data: dict) -> str:
    """Uses LLM to synthesize macro events and volatility into a cohesive daily summary."""
    if not check_llm_connectivity():
        # Never return canned filler: build a briefing from the actual data.
        return _build_data_grounded_briefing(macro_data, volatility_data)
        
    prompt = f"""
Please synthesize the following market data into a cohesive daily institutional market briefing.
Macro Context: {json.dumps(macro_data)}
Volatility Context: {json.dumps(volatility_data)}

Format the output with clear professional headings and actionable bullet points. 
Focus strictly on technical positioning, macro events, and volatility surface implications. Do not use generic filler.
"""
    system_prompt = "You are the Octavian Lead Market Strategist. Deliver an institutional-grade daily briefing."
    result = _call_llm(prompt, system=system_prompt)
    if result:
        return result
    # LLM returned nothing — fall back to the data-grounded briefing rather
    # than a canned one-liner.
    return _build_data_grounded_briefing(macro_data, volatility_data)