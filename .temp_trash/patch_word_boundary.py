src = open('financial_llm_engine.py').read()

# --- Patch A: signature ---
old_sig = "def _is_equity_reference(query: str, token: str) -> bool:"
new_sig = ("def _is_equity_reference(query: str, token: str, word_boundary: bool = False) -> bool:")
assert src.count(old_sig) == 1
src = src.replace(old_sig, new_sig)

# --- Patch B: docstring mention ---
old_doc = """    \"\"\"True when a query references an ambiguous token AS A SECURITY rather
    than as a finance concept (e.g. 'AI stock' vs 'AI infrastructure').\"\"\""""
new_doc = """    \"\"\"True when a query references an ambiguous token AS A SECURITY rather
    than as a finance concept (e.g. 'AI stock' vs 'AI infrastructure').

    With ``word_boundary=True`` every multi-character phrase must end at a
    word boundary, so prose plurals never match: 'all stocks' cannot satisfy
    the 'all stock' phrase, and 'are prices' cannot satisfy 'are price'.
    Used for universe tickers that are also core English stopwords.\"\"\""""
assert src.count(old_doc) == 1
src = src.replace(old_doc, new_doc)

# --- Patch C: phrase loop ---
old_loop = """    for ctx in stock_context:
        if len(tl) == 1:
            # Single-letter tickers (F, C, T, V, ...) must be STANDALONE words
            # — \"e moat\" matches inside \"durable moat\" and \"e position
            # against\" inside \"competitive position against\". Word-boundary
            # anchoring on the whole phrase keeps \"buy F\", \"F to fall\" while
            # blocking accidental in-word matches.
            if re.search(rf\"\\b{re.escape(ctx)}\\b\", ql):
                return True
        elif ctx in ql:
            return True"""
new_loop = """    for ctx in stock_context:
        if len(tl) == 1:
            # Single-letter tickers (F, C, T, V, ...) must be STANDALONE words
            # — \"e moat\" matches inside \"durable moat\" and \"e position
            # against\" inside \"competitive position against\". Word-boundary
            # anchoring on the whole phrase keeps \"buy F\", \"F to fall\" while
            # blocking accidental in-word matches.
            if re.search(rf\"\\b{re.escape(ctx)}\\b\", ql):
                return True
        elif word_boundary:
            # Stopword tickers (ARE/ALL/AM/CAN/...) — the phrase must end at a
            # word boundary so prose plurals never resolve: \"all stocks\"
            # cannot satisfy the 'all stock' phrase, \"are prices\" cannot
            # satisfy 'are price'.
            if re.search(rf\"\\b{re.escape(ctx)}(?![a-z])\", ql):
                return True
        elif ctx in ql:
            return True"""
assert src.count(old_loop) == 1
src = src.replace(old_loop, new_loop)

# --- Patch D: lowercase stopword gate uses word_boundary ---
old_low = """        if up in _STOPWORDS:
            try:
                in_universe = up in universe.get_known_ticker_set()
            except Exception:
                in_universe = False
            if not (in_universe and _is_equity_reference(query, up)):
                continue"""
new_low = """        if up in _STOPWORDS:
            try:
                in_universe = up in universe.get_known_ticker_set()
            except Exception:
                in_universe = False
            if not (in_universe and _is_equity_reference(query, up, word_boundary=True)):
                continue"""
assert src.count(old_low) == 1
src = src.replace(old_low, new_low)

# --- Patch E: uppercase stopword gate uses word_boundary ---
old_up = """        if t_clean in _STOPWORDS:
            # 1-char NYSE tickers (F/C/T/V) and genuine universe tickers that
            # double as English words (ARE/ALL/AM/CAN/...) resolve ONLY when
            # the query explicitly frames them as securities — never from
            # bare prose (\"grade: C\", \"are\", \"all\").
            try:
                in_universe = t_clean in universe.get_known_ticker_set()
            except Exception:
                in_universe = False
            if not ((len(t_clean) == 1 or in_universe)
                    and _is_equity_reference(query, t_clean)):
                continue"""
new_up = """        if t_clean in _STOPWORDS:
            # 1-char NYSE tickers (F/C/T/V) and genuine universe tickers that
            # double as English words (ARE/ALL/AM/CAN/...) resolve ONLY when
            # the query explicitly frames them as securities — never from
            # bare prose (\"grade: C\", \"are\", \"all\").
            try:
                in_universe = t_clean in universe.get_known_ticker_set()
            except Exception:
                in_universe = False
            if not ((len(t_clean) == 1 or in_universe)
                    and _is_equity_reference(query, t_clean, word_boundary=True)):
                continue"""
assert src.count(old_up) == 1
src = src.replace(old_up, new_up)

open('financial_llm_engine.py', 'w').write(src)
print("PATCH_OK")
