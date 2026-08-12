src = open('financial_llm_engine.py').read()
old = '_v = "fa::v11::"'
new = '_v = "fa::v12::"'
assert src.count(old) == 1
src = src.replace(old, new)
open('financial_llm_engine.py', 'w').write(src)
print("BUMP_OK")
