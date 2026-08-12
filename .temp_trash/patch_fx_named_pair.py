src = open('financial_llm_engine.py').read()

old = """    is_dollar_q = (\"dollar\" in q or \"dxy\" in q or \"usd\" in q
                   or any(w in q for w in (\"currency\", \"currencies\", \"greenback\")))
    is_index_q = (\"dollar index\" in q or \"dxy\" in q or \"dx-y.nyb\" in q)
    lines = [\"### FX Outlook\", \"\",
             \"Core pairs ranked by 5-day momentum (higher = stronger):\", \"\"]"""
new = """    is_dollar_q = (\"dollar\" in q or \"dxy\" in q or \"usd\" in q
                   or any(w in q for w in (\"currency\", \"currencies\", \"greenback\")))
    is_index_q = (\"dollar index\" in q or \"dxy\" in q or \"dx-y.nyb\" in q)

    # ── Specific-pair asks ─────────────────────────────────────────────────
    # \"How will USD/BRL react to the Fed?\" must LEAD with the named pair
    # (live quote + directional read) rather than only ranking the G10
    # crosses. A pair counts as \"named\" only when its joined code actually
    # appears in the original query text — injected defaults never do.
    q_up = (query or \"\").upper()
    named_pairs = []
    for _t in (tickers or []):
        _t = str(_t)
        if _t.endswith(\"=X\"):
            _base = _t[:-2]
            if _base in q_up.replace(\"/\", \"\").replace(\"-\", \"\").replace(\"_\", \"\").replace(\" \", \"\"):
                named_pairs.append(_t)
    named_pairs = list(dict.fromkeys(named_pairs))

    lines = [\"### FX Outlook\", \"\"]
    if named_pairs:
        focus = named_pairs[0]
        td = (live_data or {}).get(focus, {})
        fpx = td.get(\"price\")
        fchg = td.get(\"change_5d\")
        if not fpx:
            try:
                df = _fetch_series_for(focus, period=\"5d\")
                if df is not None and len(df) > 1:
                    close = df[\"Close\"]
                    if isinstance(close, pd.DataFrame):
                        close = close.iloc[:, 0]
                    close = close.dropna().astype(float)
                    if len(close) > 1:
                        fpx = float(close.iloc[-1])
                        fchg = (fpx / float(close.iloc[0]) - 1) * 100
            except Exception:
                pass
        focus_label = _label(focus)
        if fpx and fpx > 0:
            direction = \"firming\" if (fchg or 0) > 0 else \"softening\"
            lines += [f\"**{focus_label}:** {fpx:,.4f} ({fchg:+.2f}% 5d) — the pair is {direction} over the past week.\"]
        else:
            lines += [f\"**{focus_label}:** live quote unavailable right now — see the momentum table below for the nearest read.\"]
        # Evidence-based transmission note: what actually drives THIS pair.
        if \"fed\" in q or \"rate\" in q or \"fomc\" in q or \"hike\" in q or \"cut\" in q:
            lines += [\"- **Fed transmission:** EM pairs like this one react to the Fed through the real-rate \"
                      \"differential (carry), global risk appetite and the commodity terms of trade — a hawkish \"
                      \"surprise typically pressures the EM currency via the dollar leg; a dovish surprise \"
                      \"relieves it. The 5-day move above is the market's current read; the next FOMC \"
                      \"statement/CPI print is the primary swing catalyst.\"]
        lines += [\"\"]"""
assert src.count(old) == 1
src = src.replace(old, new)

open('financial_llm_engine.py', 'w').write(src)
print("PATCH_OK")
