src = open('financial_llm_engine.py').read()

# 1) Remove setup phrases from the shared list
old = """        # Trade-setup framing: \"Give me a trade setup for ARE with entry, stop
        # and target\" — a setup can only be for a security, so this is
        # unambiguous context (the uppercase requirement on stopword tickers
        # keeps \"setup for all users\" prose from resolving ALL).
        f\"setup for {tl}\", f\"trade setup for {tl}\", f\"setup on {tl}\","""
assert src.count(old) == 1
src = src.replace(old, "")

# 2) Add a gated setup-context check right after the phrase loop
old2 = """    for ctx in stock_context:
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
new2 = """    for ctx in stock_context:
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
            return True
    # Trade-setup framing (\"Give me a trade setup for ARE with entry, stop and
    # target\") is unambiguous ONLY when the security context is strict — the
    # word-boundary stopword path, or the token written in UPPERCASE. Without
    # that gate, lowercase prose like \"the setup for key support levels\" or
    # \"the setup for AI infrastructure\" would resolve KEY/AI.
    if word_boundary or tl.upper() in query:
        for ctx in (f\"setup for {tl}\", f\"trade setup for {tl}\", f\"setup on {tl}\"):
            if re.search(rf\"\\b{re.escape(ctx)}(?![a-z])\", ql):
                return True"""
assert src.count(old2) == 1
src = src.replace(old2, new2)

open('financial_llm_engine.py', 'w').write(src)
print("PATCH_OK")
