import os, re, subprocess, sys

print("=" * 70)
print("1. EMOJIS REMAINING")
print("=" * 70)
emoji_re = re.compile(
    r"[\U0001F000-\U0001FAFF]|[\U00002600-\U000027BF]|[\U0001F1E6-\U0001F1FF]|[\uFE0F\u200D]|[⬀-⯿]"
)
left = []
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in (".git", "node_modules", ".venv", "__pycache__", "nltk_data", "data_cache", ".temp_trash", ".kiro", ".next")]
    for fn in files:
        if not fn.endswith((".py", ".ts", ".tsx", ".js", ".jsx", ".md", ".html", ".css", ".json", ".txt", ".sh")):
            continue
        p = os.path.join(root, fn)
        try:
            t = open(p, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        m = emoji_re.findall(t)
        if m:
            left.append((p, len(m)))
print("EMOJIS REMAINING:", left if left else "NONE")

print()
print("=" * 70)
print("2. COMPILE CHECK (key modules)")
print("=" * 70)
mods = [
    "ai_chatbot.py", "main.py", "quant_portal.py", "spreadsheet_generator.py",
    "data_downloader.py", "custom_dashboard.py", "financial_model_generator_ui.py",
    "futures_simulation_grader.py", "options_simulation_grader.py",
    "portfolio_chatbot_context.py", "terms_of_service.py",
    "notification_settings_ui.py", "document_analyzer.py", "news_analysis_engine.py",
    "presentation_generator.py", "ib_excel_engine.py", "options_engine.py",
    "unbiased_market_analyzer.py", "financial_model_generator.py",
    "mna_model_engine.py", "lbo_model_engine.py",
]
for f in mods:
    r = subprocess.run(["python3", "-m", "py_compile", f], capture_output=True, text=True)
    print(f"{f}: {'COMPILE_OK' if r.returncode == 0 else 'BROKEN'}")
    if r.returncode != 0:
        print("   ", r.stderr.strip().splitlines()[:3])

print()
print("=" * 70)
print("3. spreadsheet_generator.py STATE")
print("=" * 70)
t = open("spreadsheet_generator.py").read()
lines = t.splitlines()
print("lines:", len(lines))
print("def _generate count:", t.count("def _generate"))
stubs = [i + 1 for i, l in enumerate(lines) if re.match(r"^\s+pass\s*$", l)]
print("pass-stub lines:", stubs[:30], "total:", len(stubs))
print("build_comps_workbook present:", "build_comps_workbook" in t)
print("ib_excel_engine import:", "ib_excel_engine" in t)

print()
print("=" * 70)
print("4. quant_portal.py CROWDING STATE")
print("=" * 70)
t = open("quant_portal.py").read()
print("get_crowding_engine:", "get_crowding_engine" in t)
print("build_dashboard:", "build_dashboard" in t)
print("factor_crowding import:", "factor_crowding" in t)
print("np.random.uniform occurrences:", t.count("np.random.uniform"))

print()
print("=" * 70)
print("5. ai_chatbot.py KEY FUNCTIONS")
print("=" * 70)
t = open("ai_chatbot.py").read()
for fn in [
    "def process_enhanced_query", "def process_unbiased_query",
    "def show_octavian_chatbot", "def _generate_enhanced_charts",
    "_generate_opportunity_charts", "show_reasoning_available",
    "def _analyze_symbol_unbiased", "Octavian analysis complete",
]:
    print(f"  {fn}: {'present' if fn in t else 'MISSING'}")
print("lines:", len(t.splitlines()))

print()
print("=" * 70)
print("6. main.py KEY FEATURES")
print("=" * 70)
t = open("main.py").read()
for fn in [
    "_show_personalized_dashboard", "get_cached_outlook",
    "Institutional Strategy Outlook", "show_octavian_chatbot",
    "render_13f", "show_financial_generator", "render_quant_portal",
    "show_paper_trading_dashboard", "import options_engine",
    "get_options_engine", "render_simulation_viewer",
]:
    print(f"  {fn}: {'present' if fn in t else 'MISSING'}")
print("lines:", len(t.splitlines()))

print()
print("=" * 70)
print("7. news_analysis_engine.py WHISPER STATE")
print("=" * 70)
t = open("news_analysis_engine.py").read()
print("def get_market_whispers:", "def get_market_whispers" in t)
print("  real-coverage markers (coverage_count/corroborat/outlets):",
      ("coverage_count" in t) or ("corroborat" in t) or ("outlets" in t))
print("  fabricated 'Institutional Desk':", "Institutional Desk" in t)

print()
print("=" * 70)
print("8. custom_dashboard.py OPTIONS BUG")
print("=" * 70)
t = open("custom_dashboard.py").read()
bare = re.findall(r"(?<![\w.])get_options_engine\(\)", t)
print("bare get_options_engine() calls:", len(bare))
print("has 'import options_engine' or 'from options_engine':",
      ("import options_engine" in t) or ("from options_engine" in t))

print()
print("=" * 70)
print("9. RECOVERY SOURCES")
print("=" * 70)
# pyc files
for name in ["spreadsheet_generator", "quant_portal"]:
    import glob
    for p in glob.glob(f"__pycache__/{name}*"):
        st = os.stat(p)
        print(f"  pyc: {p}  size={st.st_size}  mtime={st.st_mtime}")
# stash
r = subprocess.run(["git", "stash", "list"], capture_output=True, text=True)
print("  git stash list:", r.stdout.strip() or "(empty)")
# fsck dangling
r = subprocess.run(["git", "fsck", "--lost-found"], capture_output=True, text=True)
dangling = [l for l in r.stdout.splitlines() if "dangling" in l]
print("  fsck dangling objects:", len(dangling))
for l in dangling[:15]:
    print("   ", l.strip())
# patch files referencing spreadsheet
for pf in ["main_diff.patch", "main_diff2.patch"]:
    if os.path.exists(pf):
        t2 = open(pf, errors="ignore").read()
        print(f"  {pf}: {len(t2.splitlines())} lines, spreadsheet refs: {t2.count('spreadsheet_generator')}")
# temp trash contents
print("  .temp_trash contains:", sorted(os.listdir(".temp_trash"))[:40])
# /tmp scripts of interest
for f in sorted(os.listdir("/tmp")):
    if f.endswith(".py") and ("spreadsheet" in f or "quant" in f or "check" in f or "excel" in f or "ib" in f):
        print(f"  /tmp/{f} size={os.path.getsize(os.path.join('/tmp', f))}")
