import re

src = open('financial_llm_engine.py').read()

# --- Patch 1: lowercase pass ---
old1 = """        if up in _NUMBER_WORDS or up in _STOPWORDS or up in _SEMANTIC_CONCEPTS:
            continue"""
new1 = """        if up in _NUMBER_WORDS or up in _SEMANTIC_CONCEPTS:
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
            if not (in_universe and _is_equity_reference(query, up)):
                continue"""
assert src.count(old1) == 1, f"lowercase gate: {src.count(old1)} matches"
src = src.replace(old1, new1)

# --- Patch 2: uppercase pass ---
old2 = """        if t_clean in _STOPWORDS:
            if not (len(t_clean) == 1 and _is_equity_reference(query, t_clean)):
                continue"""
new2 = """        if t_clean in _STOPWORDS:
            # 1-char NYSE tickers (F/C/T/V) and genuine universe tickers that
            # double as English words (ARE/ALL/AM/CAN/...) resolve ONLY when
            # the query explicitly frames them as securities — never from
            # bare prose ("grade: C", "are", "all").
            try:
                in_universe = t_clean in universe.get_known_ticker_set()
            except Exception:
                in_universe = False
            if not ((len(t_clean) == 1 or in_universe)
                    and _is_equity_reference(query, t_clean)):
                continue"""
assert src.count(old2) == 1, f"uppercase gate: {src.count(old2)} matches"
src = src.replace(old2, new2)

open('financial_llm_engine.py', 'w').write(src)
print("PATCH_OK")
