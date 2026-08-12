"""Refine the v11 mega de-dup: two parts under the same label that name
DIFFERENT instruments are distinct answers and must both be kept, even when
their template text overlaps heavily ("outlook for the VIX" vs "outlook for
PYPL" share 5/7 tokens). Text-based de-dup only applies when the instruments
match or neither part names one.
"""
path = "financial_llm_engine.py"
src = open(path).read()

old = '''        _norm_this = re.sub(r"[^a-z0-9]+", " ", sub.lower()).strip()
        _is_dup = False
        if label in seen_labels:
            for _seen in seen_label_norms.get(label, []):
                if not _seen or not _norm_this:
                    continue
                _a = set(_norm_this.split())
                _b = set(_seen.split())
                _inter = len(_a & _b)
                _jaccard = _inter / max(1, len(_a | _b))
                if _jaccard >= 0.7 or _norm_this in _seen or _seen in _norm_this:
                    _is_dup = True
                    break
        if _is_dup:
            continue
        seen_labels.add(label)
        seen_label_norms.setdefault(label, []).append(_norm_this)'''

new = '''        _norm_this = re.sub(r"[^a-z0-9]+", " ", sub.lower()).strip()
        _is_dup = False
        if label in seen_labels:
            # Parts naming DIFFERENT instruments under the same label are
            # distinct answers ("outlook for the VIX" vs "outlook for PYPL"),
            # even when the template text overlaps heavily — a user who asked
            # for two outlooks wants both. Only when the instrument sets match
            # (or neither part names an instrument) does text similarity decide.
            _prev_instr = seen_label_instrs.get(label, [])
            _this_instr = frozenset(sub_tickers or []) | frozenset(sub_sectors or [])
            _shares_instr = any(_this_instr & p for p in _prev_instr) or not _prev_instr
            if _shares_instr:
                for _seen in seen_label_norms.get(label, []):
                    if not _seen or not _norm_this:
                        continue
                    _a = set(_norm_this.split())
                    _b = set(_seen.split())
                    _inter = len(_a & _b)
                    _jaccard = _inter / max(1, len(_a | _b))
                    if (_jaccard >= 0.7 or _norm_this in _seen
                            or _seen in _norm_this):
                        _is_dup = True
                        break
        if _is_dup:
            continue
        seen_labels.add(label)
        seen_label_norms.setdefault(label, []).append(_norm_this)
        seen_label_instrs.setdefault(label, []).append(
            frozenset(sub_tickers or []) | frozenset(sub_sectors or []))'''

assert old in src, "anchor not found"
src = src.replace(old, new, 1)

# Add the seen_label_instrs tracking alongside seen_label_norms
old2 = '''    sections, labels = [], []
    seen_labels = set()
    seen_label_norms = {}
    for idx, sub in enumerate(parts, 1):'''
new2 = '''    sections, labels = [], []
    seen_labels = set()
    seen_label_norms = {}
    seen_label_instrs = {}
    for idx, sub in enumerate(parts, 1):'''
assert old2 in src, "anchor2 not found"
src = src.replace(old2, new2, 1)

open(path, "w").write(src)
print("PATCH v11b OK")
