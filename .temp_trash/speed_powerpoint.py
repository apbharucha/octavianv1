import subprocess, time

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.stdout + r.stderr

print("=" * 60)
print("A. financial_llm_engine LLM call timeout (slow path)")
print("=" * 60)
# check if it calls local LLM API
print(sh("grep -n 'timeout\\|requests.post' financial_llm_engine.py | head -5"))
# check _fetch_live_data_for_tickers cost (yf.download is expensive)
print(sh("grep -n 'yf.download' financial_llm_engine.py | head -5"))

print("=" * 60)
print("B. data_sources.py caching")
print("=" * 60)
print(sh("grep -n 'cache\\|functools\\|lru_cache\\|CACHE' data_sources.py | head -20"))

print("=" * 60)
print("C. presentation_generator full state (current line count)")
print("=" * 60)
print(sh("wc -l presentation_generator.py"))
print(sh("grep -n 'def _slide_cover\\|def _slide_body\\|def _slide_toc\\|def _build_mna_pitchbook\\|def _build_dcf_pitchbook\\|def _build_lbo_pitchbook' presentation_generator.py | head -10"))

print("=" * 60)
print("D. Does _build_mna/_dcf/_lbo pitchbook methods have visual aids (charts)?")
print("=" * 60)
print(sh("grep -n 'chart\\|plotly\\|matplotlib\\|Figure\\|image\\|png' presentation_generator.py | head -20"))

print("=" * 60)
print("E. main.py initial imports (lazy loading assessment)")
print("=" * 60)
# Count import lines in main.py
print(sh("grep -c '^import\\|^from ' main.py"))
# Count lazy imports (inside functions)
print(sh("grep -c '    import\\|    from ' main.py"))