path = "financial_llm_engine.py"
src = open(path).read()
old = '_v = "fa::v10::"'
new = '_v = "fa::v11::"'
assert old in src, "version anchor not found"
src = src.replace(old, new, 1)
open(path, "w").write(src)
print("VERSION BUMPED TO v11")
