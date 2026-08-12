path = "financial_llm_engine.py"
src = open(path).read()

old = '''            _prev_instr = seen_label_instrs.get(label, [])
            _this_instr = frozenset(sub_tickers or []) | frozenset(sub_sectors or [])
            _shares_instr = any(_this_instr & p for p in _prev_instr) or not _prev_instr'''
new = '''            _prev_instr = seen_label_instrs.get(label, [])
            _this_instr = frozenset(sub_tickers or []) | frozenset(sub_sectors or [])
            # Apply the text de-dup when this part has no instrument of its own
            # (pure macro/regime asks — the bonds/QT and economy cases) OR when
            # it overlaps an instrument already seen under this label (two
            # "outlook for AAPL" asks). Only parts naming a NEW instrument are
            # exempt — they are distinct answers regardless of template text.
            _shares_instr = (not _this_instr) or any(
                _this_instr & p for p in _prev_instr)'''
assert old in src, "anchor not found"
src = src.replace(old, new, 1)
open(path, "w").write(src)
print("PATCH v11c OK")
