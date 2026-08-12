"""Rubric scoring — every response is scored on 8 criteria, each out of 10.

Criteria
--------
ticker_cleanliness    — no stopword leakage / garbage symbols in extraction
data_grounding        — response cites real numbers (prices, %, metrics)
task_fulfillment      — response contains what the query class demands
relevance             — response engages the primary symbol / entity
depth                 — substantive length, structure, specificity
risk_content          — risk/hedging/sizing content (mandatory for risk queries)
honesty_no_fabrication — no "unable to fetch" spam, no garbage ticker echoes
formatting            — clean markdown, no emoji/debug artifacts

Overall = weighted combination; a row passes when overall >= 6.5 AND
task_fulfillment >= 6 AND ticker_cleanliness >= 7.
"""

import re

CRITERIA = [
    "ticker_cleanliness",
    "data_grounding",
    "task_fulfillment",
    "relevance",
    "depth",
    "risk_content",
    "honesty_no_fabrication",
    "formatting",
    "visual_relevance",
]

WEIGHTS = {
    "task_fulfillment": 0.20,
    "relevance": 0.14,
    "data_grounding": 0.14,
    "ticker_cleanliness": 0.13,
    "depth": 0.11,
    "risk_content": 0.09,
    "honesty_no_fabrication": 0.05,
    "formatting": 0.04,
    "visual_relevance": 0.10,
}

# Symbols that are legitimate even if they look like words — never flag these.
# Built from the real ticker universe (all sectors) plus a static base set, so
# genuine instruments (AAL, XLE, ABNB, ...) are never penalized as "prose".
_BASE_OK_SYMBOLS = {
    "GOOG", "META", "SHOP", "COIN", "HOOD", "F", "C", "V", "MA", "GS",
    "MS", "DE", "BA", "GE", "MU", "LI", "PINS", "SNAP", "FXI", "EEM",
    "T", "X", "A", "B", "G", "H", "L", "P", "S", "U", "W", "Y", "Z",
    "PLAY", "WORK", "SIX", "MGM", "WYNN", "CROX", "SKX", "TREX", "COST",
    "CAT", "AMP", "ALL", "AIG", "AON", "APD", "ARE", "ATO", "AVB", "AWK",
    "AXP", "AZO", "BBY", "BKR", "BMY", "BRK", "BSX", "BXP", "CAG", "CAH",
    "CARR", "CB", "CBOE", "CBRE", "CCI", "CDW", "CE", "CFG", "CHRW", "CI",
    "CL", "CLX", "CMCSA", "CME", "CMG", "CNC", "CNP", "COF", "CPB", "CPRT",
    "CPT", "CRL", "CSX", "CTAS", "CTLT", "CTRA", "CTVA", "CVS", "D", "DAL",
    "DAY", "DD", "DELL", "DFS", "DG", "DGX", "DHI", "DHR", "DIS", "DLR",
    "DLTR", "DOC", "DOV", "DOW", "DPZ", "DRI", "DTE", "DUK", "DVA", "DVN",
    "DXCM", "EA", "EBAY", "ECL", "ED", "EFX", "EIX", "EL", "ELS", "EMN",
    "ENPH", "EOG", "EPAM", "EQIX", "EQR", "EQT", "ES", "ESS", "ETN", "ETR",
    "EW", "EXC", "EXPD", "EXPE", "EXR", "FANG", "FAST", "FCX", "FDS",
    "FDX", "FE", "FFIV", "FHN", "FITB", "FMC", "FOX", "FOXA", "FRT",
    "FSLR", "FTNT", "FTV", "GD", "GDDY", "GILD", "GIS", "GL", "GLW", "GM",
    "GNRC", "GOOGL", "GPC", "GPN", "GRMN", "GS", "GWW", "HAL", "HAS",
    "HBAN", "HCA", "HD", "HES", "HIG", "HII", "HOLX", "HON", "HPE", "HPQ",
    "HRL", "HSIC", "HST", "HSY", "HUBB", "HUM", "HWM", "IBM", "ICE", "IDXX",
    "IEX", "IFF", "INCY", "INTC", "IP", "IPG", "IQV", "IR", "IRM", "ISRG",
    "IT", "ITW", "IVZ", "J", "JBHT", "JBL", "JKHY", "JNJ", "JNPR", "JPM",
    "K", "KDP", "KEY", "KEYS", "KHC", "KIM", "KLAC", "KMB", "KMI", "KMX",
    "KO", "KR", "KVUE", "L", "LDOS", "LEN", "LH", "LHX", "LIN", "LKQ",
    "LMT", "LNT", "LOW", "LRCX", "LUV", "LVS", "LW", "LYB", "LYV", "M",
    "MAA", "MARA", "MAS", "MCD", "MCK", "MCO", "MDLZ", "MDT", "MET", "META",
    "MGM", "MHK", "MKC", "MKTX", "MLM", "MMC", "MMM", "MNST", "MO", "MOH",
    "MOS", "MPC", "MPWR", "MRNA", "MSFT", "MSI", "MTB", "MTCH", "MTD",
    "MU", "NCLH", "NDAQ", "NDSN", "NEE", "NEM", "NFLX", "NKE", "NOC", "NOW",
    "NRG", "NSC", "NTAP", "NTRS", "NUE", "NVDA", "NVR", "NWL", "NWS", "NWSA",
    "NXPI", "O", "ODFL", "OKE", "OMC", "ON", "ORCL", "ORLY", "OTIS", "OXY",
    # Popular ETFs / funds / crypto assets that are genuine instruments even
    # though they read like words (ARKK, XRP, SOL, ...) — never flag as prose.
    "ARKK", "ARKW", "ARKG", "ARKF", "ARKQ", "QQQ", "SPY", "IWM", "DIA",
    "TLT", "GLD", "SLV", "XLE", "XLF", "XLK", "XLV", "XLI", "XLY", "XLP",
    "XLB", "XLU", "XLRE", "SMH", "SOXX", "IBB", "XBI", "JETS", "KWEB",
    "FXI", "EEM", "VTI", "VOO", "IVV", "IJH", "IJR", "UUP", "HYG", "LQD",
    "EFA", "EWZ", "GDX", "GDXJ", "USO", "UNG", "TAN", "IBB", "XRT", "KRE",
    "BTC", "ETH", "SOL", "XRP", "ADA", "DOT", "LTC", "LINK", "AVAX",
    "MATIC", "SHIB", "UNI", "AAVE", "APT", "SUI", "DOGE", "FIL", "NEAR",
    "ATOM", "ALGO", "ETC", "XLM", "TRX", "ICP", "HBAR", "VET", "XTZ",
    "PANW", "PARA", "PAYX", "PCAR", "PCG", "PEG", "PEP", "PFE", "PFG", "PG",
    "PGR", "PH", "PHM", "PKG", "PLD", "PM", "PNC", "PNR", "PNW", "PODD",
    "POOL", "PPG", "PPL", "PRU", "PSA", "PSX", "PTC", "PWR", "PYPL", "QCOM",
    "QRVO", "RCL", "REG", "REGN", "RF", "RJF", "RL", "RMD", "ROK", "ROL",
    "ROP", "ROST", "RSG", "RTX", "RVTY", "SBAC", "SBUX", "SCHW", "SHW",
    "SJM", "SLB", "SNA", "SNPS", "SO", "SOLV", "SPG", "SPGI", "SRE", "STE",
    "STLD", "STT", "STX", "STZ", "SW", "SWK", "SWKS", "SYF", "SYK", "SYY",
    "TAP", "TDY", "TEL", "TER", "TFC", "TFX", "TGT", "TJX", "TMO", "TMUS",
    "TPR", "TRGP", "TROW", "TRV", "TSCO", "TSLA", "TSN", "TT", "TTWO",
    "TXN", "TXT", "TYL", "UAL", "UDR", "UHS", "ULTA", "UNH", "UNP", "UPS",
    "URI", "USB", "VICI", "VLO", "VLTO", "VMC", "VRSK", "VRSN", "VRTX",
    "VST", "VTR", "VTRS", "VZ", "WAB", "WAT", "WBA", "WBD", "WDC", "WEC",
    "WELL", "WFC", "WM", "WMB", "WMS", "WMT", "WRB", "WRK", "WST", "WTW",
    "WY", "XEL", "XOM", "XYL",    "YUM", "ZBH", "ZBRA", "ZTS", "AAL", "ABNB", "ACHR", "AGCO", "ALK", "AME",
    "AOS", "ALV", "AEO", "AJRD", "ALIT", "AMSC", "ARKX", "STLA", "XLE", "XBI",
    "IBB", "SMH", "KRE", "XHB", "IYR", "XLK", "XLF", "XLV", "XLY", "XLP",
    "XLU", "XLI", "XLB", "XLRE", "XLC", "GLD", "SLV", "TLT", "HYG", "LQD",
    "EEM", "VWO", "FXI", "EWZ", "QQQ", "SPY", "DIA", "IWM", "COIN", "MSTR",
    "HOOD", "RIVN", "LCID", "NIO", "XPEV", "LI", "ROKU", "PINS", "SNAP",
    "DAL", "UAL", "JETS", "CSCO", "QCOM", "MU", "TSM", "SHOP", "SQ", "PYPL",
    "UBER", "LYFT", "BKR", "OXY", "COP", "XOM", "CVX", "SLB", "HAL", "BLK",
    "SCHW", "WFC", "GS", "MS", "BAC", "JPM", "BLK", "TGT", "LOW", "CVS",
    "GILD", "AMGN", "BIIB", "MRNA", "NFLX", "AMD", "INTC", "META", "NVDA",
}


def _known_ok_symbols() -> frozenset:
    """Real tickers from the universe + base set (memoized)."""
    ok = set(_BASE_OK_SYMBOLS)
    try:
        from ticker_universe import get_ticker_universe
        uni = get_ticker_universe()
        for sector in uni.get_all_sectors():
            ok.update(uni.get_sector_tickers(sector))
    except Exception:
        pass
    return frozenset(ok)


KNOWN_OK_SYMBOLS = _known_ok_symbols()

# Substring patterns that indicate boilerplate / template leakage.
BOILERPLATE_MARKERS = [
    "landscape is evolving",
    "flow-of-funds suggests",
    "here's what the data shows for",
    "reflects broad thematic rotation",
    "OPEC+ supply discipline is keeping oil prices elevated",
    "stay patient and selective",
    "position sizing should dominate decision quality",
    "monitor regime",
]

EMOJI_RE = re.compile(
    r"[\U0001F000-\U0001FAFF\u2600-\u27BF\U0001F1E6-\U0001F1FF\u2B00-\u2BFF\u23E9-\u23FA]"
)
GARBAGE_TICKER_RE = re.compile(r"\b[A-Z]{1,6}=X\b")


def _clamp(x: float) -> float:
    return max(0.0, min(10.0, x))


def _in_response(term, rl: str) -> bool:
    """Check a required term against the (lowercased) response, tolerating
    symbol-form differences: 'USD/KRW' should match 'USDKRW=X', and
    'S&P 500' should match '^GSPC' or 'S&P 500 (^GSPC)'."""
    t = str(term).lower()
    if not t:
        return False
    if t in rl:
        return True
    t_norm = t.replace("/", "").replace(" ", "").replace("-", "")
    if t_norm and t_norm in rl.replace("/", "").replace(" ", "").replace("-", ""):
        return True
    if t_norm and (t_norm + "=x") in rl:
        return True
    # caret form: S&P 500 / spx -> ^gspc
    aliases = {
        "s&p 500": "^gspc", "spx": "^gspc", "s&p": "^gspc", "sp500": "^gspc",
        "nasdaq": "^ixic", "dow": "^dji", "dow jones": "^dji",
        "russell 2000": "^rut", "russell": "^rut", "vix": "^vix",
        "dollar index": "dx-y.nyb", "dxy": "dx-y.nyb",
    }
    for name, sym in aliases.items():
        if t == name and sym in rl:
            return True
    # Futures / crypto / index symbol aliases: a response that names the
    # underlying commodity or coin engages the instrument even when the Yahoo
    # symbol form never appears ("gold" satisfies GC=F, "bitcoin" BTC-USD).
    _SYM_ALIASES = {
        "gc=f": ["gold"], "cl=f": ["oil", "crude"], "bz=f": ["brent"],
        "ng=f": ["natural gas", "natgas"], "si=f": ["silver"], "hg=f": ["copper"],
        "pl=f": ["platinum"], "pa=f": ["palladium"], "zc=f": ["corn"],
        "zw=f": ["wheat"], "zs=f": ["soybean"], "kc=f": ["coffee"],
        "sb=f": ["sugar"], "ct=f": ["cotton"], "cc=f": ["cocoa"],
        "btc-usd": ["bitcoin", "btc"], "eth-usd": ["ethereum", "eth"],
        "sol-usd": ["solana"], "xrp-usd": ["xrp"], "ada-usd": ["cardano"],
        "dx-y.nyb": ["dollar index", "dxy"], "^vix": ["vix"],
        "^gspc": ["s&p 500", "s&p"], "^ixic": ["nasdaq"], "^dji": ["dow"],
        "^rut": ["russell 2000", "russell"], "es=f": ["s&p futures"],
        "nq=f": ["nasdaq futures"],
    }
    for sym, words in _SYM_ALIASES.items():
        if t == sym and any(w in rl for w in words):
            return True
    # Sector alias normalization: "oil and gas" acceptable if the response
    # engages the "energy" complex; "banks" if it engages "financials".
    _SECTOR_ALIASES = {
        "oil and gas": ["energy", "oil"], "banks": ["financials", "banking", "bank"],
        "software": ["technology", "software"], "media": ["communication", "media"],
        "autos": ["ev", "automotive", "autos"], "pharma": ["healthcare", "pharmaceutical"],
        "semis": ["semiconductors", "chip"], "airlines": ["airlines", "airline", "jets"],
    }
    if t in _SECTOR_ALIASES:
        return any(a in rl for a in _SECTOR_ALIASES[t])
    return False


def _mega_task_primary(t):
    if t.get("primary"):
        return t["primary"]
    for key in ("requires", "requires_any"):
        vals = t.get(key) or []
        if vals:
            return vals[0]
    return None


def score_response(query, expectations, intents, tickers, response_text,
                   visuals=None) -> dict:
    """Return {criterion: (score, reason)} for one query/response pair.

    `visuals` is an optional dict from the pipeline's visual check:
        expected: bool   — this query class warrants charts
        requested: bool  — user explicitly asked for a chart/visual
        generated: [sym] — charts actually produced
        candidates: [sym] — tickers available to chart
    """
    resp = response_text or ""
    rl = resp.lower()
    ql = query.lower()
    scores = {}

    # ── Template-leak guard ─────────────────────────────────────────────────
    # expectations['avoid'] lists substrings that must NEVER appear (e.g. a
    # deep-dive thesis response that is actually the generic "Geopolitical &
    # Event Briefing" template). Computed once, applied to both task
    # fulfillment and honesty below.
    avoid_hits = [a for a in (expectations.get("avoid") or []) if a in rl]

    # ── 1. ticker_cleanliness ────────────────────────────────────────────────
    penalty = 0.0
    reasons = []
    for t in tickers or []:
        tu = str(t).upper()
        if tu in KNOWN_OK_SYMBOLS:
            continue
        if tu.endswith("=F") or tu.endswith("=X") or tu.startswith("^") or "-USD" in tu:
            continue
        if tu.isalpha() and len(tu) >= 2:
            penalty += 1.5
            reasons.append(f"extracted '{tu}' looks like prose, not a ticker")
    # Response echoing garbage tickers is worse than just extraction
    for m in re.findall(r"\b[A-Z]{2,6}\b", resp):
        mu = m.upper()
        if mu in KNOWN_OK_SYMBOLS or mu.endswith("=X") or mu.endswith("=F") or mu.startswith("^"):
            continue
        if mu in ("CL", "GC", "SI", "NG", "HG", "BZ", "ES", "NQ", "YM", "RTY",
                  "ZW", "ZC", "ZS", "KC", "SB", "CT", "CC", "HO", "RB", "PL", "PA"):
            continue  # contract codes may legitimately appear in prose
        if mu in ("SHOUD", "BASED", "WHILE", "GIVEN", "AFFECT", "PROOF", "CLAIM",
                  "SUPPORT", "EVIDENCE", "PROVIDE", "SHOULD", "TRADE", "SETUP",
                  "HEDGE", "HEDGING", "TARGET", "STOP", "RISK", "MANAGE", "WANT"):
            penalty += 1.0
            reasons.append(f"response echoes prose-as-ticker '{mu}'")
    # =X tokens are legitimate Yahoo FX symbols (USDKRW=X, EURJPY=X, ...); only
    # flag them when the base is NOT a known currency or currency pair.
    _CURRENCY_CODES = {"USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD",
                       "CNY", "INR", "MXN", "BRL", "SGD", "KRW", "HKD", "SEK",
                       "NOK", "DKK", "PLN", "CZK", "HUF", "TRY", "ZAR", "THB",
                       "PHP", "MYR", "IDR", "TWD", "VND", "ILS", "AED", "SAR",
                       "RUB", "CLP", "COP", "PEN", "ARS", "PKR", "BDT", "NGN",
                       "EGP", "RON", "BGN", "ISK", "JOD", "KWD", "BHD", "QAR",
                       "OMR", "LKR", "UAH", "GHS", "KES", "MAD", "DZD", "TND",
                       "ETB", "MUR", "ZMW", "MZN", "NPR", "UYU", "PYG", "BOB",
                       "CRC", "DOP", "GTQ", "HNL", "NIO", "PAB", "TTD", "XCD",
                       "FJD", "PGK", "SBD", "TOP", "VUV", "WST", "ANG", "AWG",
                       "BMD", "BBD", "BZD", "GYD", "HTG", "JMD", "KYD", "SOS",
                       "TZS", "UGX", "XAF", "XOF", "GMD", "GNF", "LRD", "MRU",
                       "SCR", "SLL", "STN", "SDG", "SSP", "DJF", "ERN", "CVE",
                       "MOP", "MNT", "MMK", "KZT", "KGS", "TJS", "TMT", "AZN",
                       "GEL", "AMD", "BYN", "MDL", "RSD", "MKD", "ALL", "BAM",
                       "MGA", "RWF", "BIF", "CDF", "GIP", "IMP", "JEP", "GGP",
                       "FKP", "SHP", "SYP", "LBP", "IQD", "YER", "AFN", "IRR",
                       "MVR", "LSL", "NAD", "SZL", "AOA", "CUP", "CUC", "DZD",
                       "XPF", "KMF", "VES", "VED", "MRO", "ZWL", "BTC", "ETH",
                       "XRP", "LTC", "BCH", "ADA", "DOT", "SOL", "AVAX", "MATIC"}
    for m in GARBAGE_TICKER_RE.findall(resp):
        base = m.rstrip("=X").rstrip("=").upper()
        if base in _CURRENCY_CODES:
            continue
        if len(base) == 6 and base[:3] in _CURRENCY_CODES and base[3:] in _CURRENCY_CODES:
            continue
        penalty += 2.0
        reasons.append(f"fabricated =X pseudo-ticker '{m}' in response")
    scores["ticker_cleanliness"] = (_clamp(10.0 - penalty),
                                    "; ".join(reasons[:4]) or "clean symbol extraction")

    # ── 2. data_grounding ────────────────────────────────────────────────────
    numbers = re.findall(r"\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?", resp)
    dollar = len(re.findall(r"\$\d", resp))
    pct = len(re.findall(r"[\d.]+\s?%", resp))
    numeric = len([n for n in numbers if re.sub(r"[$,%]", "", n).replace(".", "").isdigit()])
    if numeric >= 8:
        g = 10.0
        reason = f"{numeric} numeric values cited"
    elif numeric >= 5:
        g = 8.0
        reason = f"{numeric} numeric values cited"
    elif numeric >= 3:
        g = 6.0
        reason = f"only {numeric} numeric values cited"
    else:
        g = 2.0
        reason = "response is prose-heavy with almost no numbers"
    if dollar and pct:
        g = min(10.0, g + 0.5)
        reason += "; includes $ and % figures"
    scores["data_grounding"] = (g, reason)

    # ── 3. task_fulfillment (incl. mega multi-task prompts) ──────────────────
    mega_tasks = expectations.get("mega_tasks")
    if mega_tasks:
        matched = 0
        detail = []
        for t in mega_tasks:
            reqs = t.get("requires") or []
            req_any = t.get("requires_any") or []
            if req_any:
                ok = any(_in_response(r, rl) for r in req_any)
            elif reqs:
                ok = all(_in_response(r, rl) for r in reqs)
            else:
                ok = True
            # setup tasks are satisfied by concrete trade-plan language
            if t.get("setup_query") and not ok:
                ok = any(x in rl for x in ("entry", "stop", "target", "risk/reward"))
            if t.get("risk_query") and not ok:
                ok = any(x in rl for x in ("hedge", "hedging", "risk", "puts"))
            matched += 1 if ok else 0
            detail.append(f"{t.get('label', '?')}:{'OK' if ok else 'MISS'}")
        task_score = 10.0 * matched / max(len(mega_tasks), 1)
        reason = (f"{matched}/{len(mega_tasks)} sub-tasks addressed "
                  f"({', '.join(detail)})")
    else:
        req = expectations.get("requires") or []
        req_any = expectations.get("requires_any") or []
        matched = sum(1 for r in req if _in_response(r, rl))
        if req_any:
            any_ok = any(_in_response(r, rl) for r in req_any)
            task_score = 10.0 if any_ok else 3.0
            reason = ("primary entity addressed" if any_ok
                      else f"failed to mention any of: {', '.join(map(str, req_any))[:80]}")
        elif req:
            task_score = 10.0 * matched / len(req)
            reason = f"matched {matched}/{len(req)} required elements"
        else:
            task_score = 7.0
            reason = "no explicit requirements"
        if avoid_hits:
            task_score = min(task_score, 3.0)
            reason = ("template leakage: "
                      + ", ".join(repr(a) for a in avoid_hits[:2]))
        if expectations.get("setup_query") and task_score < 8:
            for token in ("entry", "stop", "target", "risk/reward", "position size"):
                if token in rl:
                    task_score = max(task_score, 8.0)
                    reason = f"setup content present ({token})"
                    break
    scores["task_fulfillment"] = (_clamp(task_score), reason)

    # ── 4. relevance (avg over mega sub-tasks when present) ──────────────────
    if mega_tasks:
        rel_scores = []
        for t in mega_tasks:
            p = _mega_task_primary(t)
            if not p:
                continue
            if _in_response(p, rl):
                rel_scores.append(10.0)
            elif str(p).lower() in ql:
                rel_scores.append(5.0)
            else:
                rel_scores.append(3.0)
        if rel_scores:
            rel = sum(rel_scores) / len(rel_scores)
            reason = f"avg engagement across {len(rel_scores)} sub-task entities"
        else:
            rel = 8.0
            reason = "no per-task entities specified"
    else:
        primary = expectations.get("primary")
        if primary and _in_response(primary, rl):
            rel = 10.0
            reason = f"engages primary entity '{primary}'"
        elif primary and str(primary).lower() in ql:
            rel = 4.0
            reason = f"query mentions '{primary}' but response does not"
        elif primary:
            rel = 4.0
            reason = f"expected engagement with '{primary}', response generic"
        else:
            rel = 8.0
            reason = "no primary entity specified"
    scores["relevance"] = (rel, reason)

    # ── 5. depth ─────────────────────────────────────────────────────────────
    words = len(resp.split())
    headers = len(re.findall(r"^#{1,4} ", resp, re.M))
    if words >= 250 and headers >= 3:
        d = 10.0
    elif words >= 160 and headers >= 2:
        d = 8.0
    elif words >= 90:
        d = 6.0
    elif words >= 40:
        d = 4.0
    else:
        d = 2.0
    scores["depth"] = (_clamp(d), f"{words} words, {headers} sections")

    # ── 6. risk_content ──────────────────────────────────────────────────────
    risk_tokens = ["risk", "stop", "entry", "target", "hedge", "hedging",
                   "position size", "sizing", "reward", "drawdown", "volatility",
                   "exposure", "invalidation"]
    found = [tok for tok in risk_tokens if tok in rl]
    mega_tasks = expectations.get("mega_tasks")
    if mega_tasks and any(t.get("risk_query") or t.get("setup_query")
                          for t in mega_tasks):
        risk_query = True
    else:
        risk_query = expectations.get("risk_query") or expectations.get("setup_query")
    if risk_query:
        if "stop" in rl and "entry" in rl and ("target" in rl or "reward" in rl):
            rc = 10.0
        elif len(found) >= 4:
            rc = 8.0
        elif len(found) >= 2:
            rc = 5.0
        else:
            rc = 2.0
        reason = f"risk tokens found: {', '.join(found[:6]) or 'none'}"
    else:
        rc = 7.0 if found else 6.0
        reason = "risk content not mandatory for this query"
    scores["risk_content"] = (rc, reason)

    # ── 7. honesty / no fabrication ──────────────────────────────────────────
    hp = 0.0
    h_reasons = []
    unfetch = len(re.findall(r"unable to fetch|could not (?:load|fetch)", rl))
    if unfetch:
        hp += 1.5 * unfetch
        h_reasons.append(f"{unfetch} 'unable to fetch' lines")
    for bp in BOILERPLATE_MARKERS:
        if bp in rl:
            hp += 1.0
            h_reasons.append(f"boilerplate: '{bp[:40]}'")
    if "please verify the ticker symbol" in rl:
        hp += 1.5
    if "!=" in resp or "TODO" in resp or "FIXME" in resp:
        hp += 2.0
        h_reasons.append("debug/placeholder artifact")
    if avoid_hits:
        hp += 2.0 * len(avoid_hits)
        h_reasons.append("template leakage: " + ", ".join(repr(a) for a in avoid_hits[:2]))
    scores["honesty_no_fabrication"] = (_clamp(10.0 - hp),
                                        "; ".join(h_reasons[:4]) or "no fabrication markers")

    # ── 8. formatting ────────────────────────────────────────────────────────
    fp = 0.0
    f_reasons = []
    if EMOJI_RE.search(resp):
        fp += 3.0
        f_reasons.append("emoji present")
    if "DEBUG:" in resp or "\n\n\n\n" in resp:
        fp += 1.5
        f_reasons.append("debug/whitespace artifact")
    if headers == 0:
        fp += 1.0
        f_reasons.append("no markdown headers")
    if len(resp) > 6000:
        fp += 1.0
        f_reasons.append("overlong response")
    scores["formatting"] = (_clamp(10.0 - fp), "; ".join(f_reasons[:3]) or "clean formatting")

    # ── 9. visual_relevance ──────────────────────────────────────────────────
    vis = visuals or {}
    expected = bool(vis.get("expected"))
    requested = bool(vis.get("requested"))
    generated = vis.get("generated") or []
    candidates = vis.get("candidates") or []
    if not candidates:
        if generated:
            vs = 6.0
            v_reason = f"{len(generated)} charts generated with no ticker in query"
        else:
            vs = 10.0
            v_reason = "no tickers — no charts needed"
    elif expected or requested:
        if not generated:
            vs = 3.0
            v_reason = "query warrants visuals but none were generated"
        else:
            cov = sum(1 for g in generated if g in candidates) / max(len(candidates), 1)
            if cov >= 0.8:
                vs = 10.0
                v_reason = f"charts cover {len(generated)}/{len(candidates)} query symbols"
            elif cov >= 0.5:
                vs = 7.0
                v_reason = f"partial chart coverage ({cov:.0%} of query symbols)"
            else:
                vs = 5.0
                v_reason = f"charts for symbols not matching the query ({cov:.0%})"
    else:
        if generated:
            vs = 5.0
            v_reason = f"{len(generated)} charts generated although this query type does not need visuals"
        else:
            vs = 9.0
            v_reason = "no charts — appropriate for a non-visual query"
    scores["visual_relevance"] = (vs, v_reason)

    return scores


def overall_score(scores: dict) -> float:
    w = sum(WEIGHTS.values())
    return round(sum(scores[c][0] * WEIGHTS[c] for c in CRITERIA) / w, 2)


def passed(scores: dict) -> bool:
    ov = overall_score(scores)
    return (ov >= 6.5
            and scores["task_fulfillment"][0] >= 6.0
            and scores["ticker_cleanliness"][0] >= 7.0)


def failure_tags(scores: dict) -> list:
    tags = []
    for c in CRITERIA:
        sc = scores[c][0]
        if sc < 5.0:
            tags.append(f"{c}:{sc:.1f}")
    return tags
