src = open('financial_llm_engine.py').read()

old = """    q_up = (query or \"\").upper()
    named_pairs = []
    for _t in (tickers or []):
        _t = str(_t)
        if _t.endswith(\"=X\"):
            _base = _t[:-2]
            if _base in q_up.replace(\"/\", \"\").replace(\"-\", \"\").replace(\"_\", \"\").replace(\" \", \"\"):
                named_pairs.append(_t)
    named_pairs = list(dict.fromkeys(named_pairs))"""
new = """    q_up = (query or \"\").upper()
    q_flat = q_up.replace(\"/\", \"\").replace(\"-\", \"\").replace(\"_\", \"\").replace(\" \", \"\")
    named_pairs = []
    for _t in (tickers or []):
        _t = str(_t)
        if _t.endswith(\"=X\") and _t[:-2] in q_flat:
            named_pairs.append(_t)
    named_pairs = list(dict.fromkeys(named_pairs))"""
assert src.count(old) == 1
src = src.replace(old, new)
open('financial_llm_engine.py', 'w').write(src)
print("PATCH_OK")
