src = open('financial_llm_engine.py').read()

old = """        if up in _STOPWORDS:
            try:
                in_universe = up in universe.get_known_ticker_set()
            except Exception:
                in_universe = False
            if not (in_universe and _is_equity_reference(query, up, word_boundary=True)):
                continue"""
new = """        if up in _STOPWORDS:
            try:
                in_universe = up in universe.get_known_ticker_set()
            except Exception:
                in_universe = False
            # Stopword tickers (ARE/ALL/AM/CAN/...) must appear in UPPERCASE in
            # the original query — the codebase's case-sensitivity rule for
            # ambiguous tokens. \"stocks are undervalued\" and \"buy are\" are
            # prose; only \"ARE\" (uppercase) with explicit security context
            # resolves. This also keeps \"gold\" (commodity -> GC=F) from
            # resolving as the GOLD equity ticker.
            if not (in_universe and up in query
                    and _is_equity_reference(query, up, word_boundary=True)):
                continue"""
assert src.count(old) == 1
src = src.replace(old, new)
open('financial_llm_engine.py', 'w').write(src)
print("PATCH_OK")
