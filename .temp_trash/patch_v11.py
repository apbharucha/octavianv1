"""Apply v11 fixes to financial_llm_engine.py:

1. _is_equity_reference: add "play {tl}" and "gains in {tl}" / "gains on {tl}"
   security-context phrases so "best way to play TGT" and "protect gains in
   TGT/LOW" resolve the ticker.
2. _insert_unpunctuated_boundaries: split at "<KNOWN_TICKER> give me|compare"
   and "<task-word> compare|give me" boundaries (the UBER mega case:
   "...target compare JNJ and GOOG give me the key support...").
3. _build_mega_response: replace label-only de-dup with near-duplicate-text
   de-dup so two distinct macro parts ("own long-duration bonds" then
   "outlook for quantitative tightening") are BOTH kept.
"""
import re

path = "financial_llm_engine.py"
src = open(path).read()

# ─────────────────────────────────────────────────────────────────────────
# FIX 1: security-context phrases
# ─────────────────────────────────────────────────────────────────────────
old1 = '''        f"my {tl} position", f"{tl} position against", f"hedge my {tl}",
        f"hedging my {tl}", f"protect {tl}", f"protecting {tl}",
        f"protection for {tl}", f"collar on {tl}", f"puts on {tl}",
        f"hedge {tl}", f"hedging {tl}", f"{tl} exposure",'''
new1 = '''        f"my {tl} position", f"{tl} position against", f"hedge my {tl}",
        f"hedging my {tl}", f"protect {tl}", f"protecting {tl}",
        f"protection for {tl}", f"collar on {tl}", f"puts on {tl}",
        f"hedge {tl}", f"hedging {tl}", f"{tl} exposure",
        # "Best way to play TGT this week" / "protect gains in TGT" — the
        # ticker appears right after a trading verb with no stock-context
        # noun, so the verb + preposition phrases carry the security framing.
        f"play {tl}", f"playing {tl}", f"gains in {tl}", f"gains on {tl}",
        f"protect gains in {tl}", f"lock in gains in {tl}",
        f"lock in gains on {tl}", f"take profits in {tl}",
        f"take profit in {tl}", f"best way to play {tl}", f"how to play {tl}",'''
assert old1 in src, "FIX1 anchor not found"
src = src.replace(old1, new1, 1)

# ─────────────────────────────────────────────────────────────────────────
# FIX 2: unpunctuated-boundary extension in _insert_unpunctuated_boundaries
# ─────────────────────────────────────────────────────────────────────────
old2 = '''        # Task-word boundary: "...entry, stop and target what is the outlook
        # for the 10-year treasury yield?" / "...target which is the better
        # investment" — a setup keyword followed directly by a question word
        # is the seam between the setup ask and the next ask.
        _TW = (r"(?:target|stop|entry|setup|outlook|forecast|plan|review|update)")
        for m in re.finditer(rf"\\b{_TW}\\s+(?={_QW}\\b)", q):
            if m.end() > pos:
                out.append(q[pos:m.end()].rstrip())
                out.append(". ")
                pos = m.end()
        out.append(q[pos:])
        return "".join(out)'''
new2 = '''        # Task-word boundary: "...entry, stop and target what is the outlook
        # for the 10-year treasury yield?" / "...target which is the better
        # investment" — a setup keyword followed directly by a question word
        # is the seam between the setup ask and the next ask.
        _TW = (r"(?:target|stop|entry|setup|outlook|forecast|plan|review|update)")
        for m in re.finditer(rf"\\b{_TW}\\s+(?={_QW}\\b)", q):
            if m.end() > pos:
                out.append(q[pos:m.end()].rstrip())
                out.append(". ")
                pos = m.end()
        # New-task imperative boundary: "...target compare JNJ and GOOG give
        # me the key support..." — a task word (target/stop/entry/setup)
        # followed directly by an imperative opener (compare / give me / what
        # about) is the seam between one ask and the next, even with no
        # punctuation. Only fires when a task word precedes the opener, so
        # "entry, stop and target" prose never splits.
        _IMP = (r"(?:compare|give me|give us|what about|how about)")
        for m in re.finditer(rf"\\b{_TW}\\s+(?={_IMP}\\b)", q):
            if m.end() > pos:
                out.append(q[pos:m.end()].rstrip())
                out.append(". ")
                pos = m.end()
        # Known-ticker imperative boundary: "GOOG give me the key support"
        # — a universe ticker followed directly by an imperative opener is a
        # fresh ask about that ticker. Guarded by the known-ticker check so
        # prose like "the market give me" never splits.
        for m in re.finditer(
                rf"\\b([A-Z][A-Z0-9.]{{0,7}})\\s+(?={_IMP}\\b)", q):
            toks = m.group(1)
            if toks in known and m.end() > pos:
                out.append(q[pos:m.end()].rstrip())
                out.append(". ")
                pos = m.end()
        out.append(q[pos:])
        return "".join(out)'''
assert old2 in src, "FIX2 anchor not found"
src = src.replace(old2, new2, 1)

# ─────────────────────────────────────────────────────────────────────────
# FIX 3: mega label de-dup -> near-duplicate-text de-dup
# ─────────────────────────────────────────────────────────────────────────
old3 = '''    sections, labels = [], []
    seen_labels = set()
    for idx, sub in enumerate(parts, 1):'''
new3 = '''    sections, labels = [], []
    seen_labels = set()
    seen_label_norms = {}
    for idx, sub in enumerate(parts, 1):'''
assert old3 in src, "FIX3 anchor A not found"
src = src.replace(old3, new3, 1)

old3b = '''        label = _mega_label(sub_intents, sub_tickers, sub_sectors)
        # ── De-duplicate: the same label twice in a row is the reported
        # "Part 1: Macro Outlook / Part 2: Macro Outlook ..." repetition bug.
        # Drop a repeat ONLY when the part carries no instrument of its own
        # (a prose continuation of the previous section). Two parts that name
        # DIFFERENT instruments under the same label ("SUI" then "Solana")
        # are distinct answers and must both be kept.
        if label in seen_labels and not (sub_tickers or sub_sectors):
            continue
        seen_labels.add(label)'''
new3b = '''        label = _mega_label(sub_intents, sub_tickers, sub_sectors)
        # ── De-duplicate: the same label twice in a row is the reported
        # "Part 1: Macro Outlook / Part 2: Macro Outlook ..." repetition bug.
        # Drop a repeat ONLY when the part is a NEAR-DUPLICATE of a previously
        # seen same-label part (high token overlap) — a prose continuation or
        # a re-asked question. Two parts that are genuinely different asks
        # under the same label ("should I own long-duration bonds right now?"
        # then "what is the outlook for quantitative tightening?") are distinct
        # macro topics and must BOTH be answered. Parts naming different
        # instruments under the same label ("SUI" then "Solana") are also kept.
        _norm_this = re.sub(r"[^a-z0-9]+", " ", sub.lower()).strip()
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
assert old3b in src, "FIX3 anchor B not found"
src = src.replace(old3b, new3b, 1)

open(path, "w").write(src)
print("PATCH OK")
