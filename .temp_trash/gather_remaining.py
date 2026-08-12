import subprocess

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.stdout + r.stderr

print("=" * 60)
print("A. quant_portal crowding section (np.random.uniform sites)")
print("=" * 60)
print(sh("grep -n 'np.random.uniform' quant_portal.py"))
print(sh("grep -n -A3 -B3 'np.random.uniform' quant_portal.py | head -60"))

print("=" * 60)
print("B. factor_crowding_engine public API")
print("=" * 60)
print(sh("grep -n 'def \\|class ' factor_crowding_engine.py | head -30"))
print(sh("grep -n 'def build_dashboard\\|def get_crowding_engine' factor_crowding_engine.py"))

print("=" * 60)
print("C. chatbot: the 0-charts bug - analyze_unbiased returns + _last_fetched_data")
print("=" * 60)
print(sh("grep -n '_last_fetched_data' unbiased_market_analyzer.py | head -10"))
print(sh("grep -n 'def analyze_unbiased' unbiased_market_analyzer.py"))
print(sh("grep -n -A40 'def analyze_unbiased' unbiased_market_analyzer.py | head -55"))

print("=" * 60)
print("D. chatbot: chart generation flow after scan")
print("=" * 60)
print(sh("grep -n 'generated .* charts\\|_generate_opportunity_charts' ai_chatbot.py | head -10"))
print(sh("grep -n -B5 -A30 'def _generate_opportunity_charts' ai_chatbot.py | head -60"))

print("=" * 60)
print("E. presentation_generator.py structure")
print("=" * 60)
print(sh("grep -n 'def \\|class ' presentation_generator.py | head -60"))
print(sh("wc -l presentation_generator.py"))
