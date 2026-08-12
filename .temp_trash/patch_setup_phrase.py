src = open('financial_llm_engine.py').read()

old = """        # Options / strategy framing: \"options strategy for LOW before
        # earnings\", \"best put structure to protect TGT\".
        f\"options for {tl}\", f\"strategy for {tl}\", f\"strategies for {tl}\",
        f\"{tl} before earnings\", f\"{tl} ahead of earnings\", f\"strategy on {tl}\","""
new = """        # Options / strategy framing: \"options strategy for LOW before
        # earnings\", \"best put structure to protect TGT\".
        f\"options for {tl}\", f\"strategy for {tl}\", f\"strategies for {tl}\",
        f\"{tl} before earnings\", f\"{tl} ahead of earnings\", f\"strategy on {tl}\",
        # Trade-setup framing: \"Give me a trade setup for ARE with entry, stop
        # and target\" — a setup can only be for a security, so this is
        # unambiguous context (the uppercase requirement on stopword tickers
        # keeps \"setup for all users\" prose from resolving ALL).
        f\"setup for {tl}\", f\"trade setup for {tl}\", f\"setup on {tl}\","""
assert src.count(old) == 1
src = src.replace(old, new)
open('financial_llm_engine.py', 'w').write(src)
print("PATCH_OK")
