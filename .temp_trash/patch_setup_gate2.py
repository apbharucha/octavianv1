src = open('financial_llm_engine.py').read()

old = """    # Trade-setup framing (\"Give me a trade setup for ARE with entry, stop and
    # target\") is unambiguous ONLY when the security context is strict — the
    # word-boundary stopword path, or the token written in UPPERCASE. Without
    # that gate, lowercase prose like \"the setup for key support levels\" or
    # \"the setup for AI infrastructure\" would resolve KEY/AI.
    if word_boundary or tl.upper() in query:
        for ctx in (f\"setup for {tl}\", f\"trade setup for {tl}\", f\"setup on {tl}\"):
            if re.search(rf\"\\b{re.escape(ctx)}(?![a-z])\", ql):
                return True"""
new = """    # Trade-setup framing (\"Give me a trade setup for ARE with entry, stop and
    # target\") is unambiguous ONLY on the strict word-boundary stopword path
    # (ARE/ALL/CAN/...), which additionally requires the token UPPERCASE in
    # the original query. The setup phrases must NOT leak into the ambiguous
    # token path (KEY/AI/SUM/...) — \"the setup for key support levels\" or
    # \"the setup for AI infrastructure spend\" are prose, not the KEY/AI
    # tickers. Ambiguous tokens already resolve via their own phrases
    # (\"KEY stock\", \"buy AI\").
    if word_boundary:
        for ctx in (f\"setup for {tl}\", f\"trade setup for {tl}\", f\"setup on {tl}\"):
            if re.search(rf\"\\b{re.escape(ctx)}(?![a-z])\", ql):
                return True"""
assert src.count(old) == 1
src = src.replace(old, new)
open('financial_llm_engine.py', 'w').write(src)
print("PATCH_OK")
