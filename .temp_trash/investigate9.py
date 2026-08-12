import subprocess

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.stdout + r.stderr

print("=" * 60)
print("A. GeneticStrategyEngine evolve/run methods")
print("=" * 60)
print(sh("grep -n 'def evolve\\|def run\\|def step\\|def train' genetic_strategy_engine.py | head -10"))
print(sh("grep -n -A15 'def evolve' genetic_strategy_engine.py | head -25"))

print("=" * 60)
print("B. quant_portal crowding section full context (720-760)")
print("=" * 60)
print(sh("sed -n '715,765p' quant_portal.py"))

print("=" * 60)
print("C. quant_portal: what symbols variable exists in render_quant_portal")
print("=" * 60)
print(sh("grep -n 'symbols =\\|symbols=\\|symbols_input\\|watchlist\\|default_symbols' quant_portal.py | head -15"))

print("=" * 60)
print("D. quant_portal genetic section context (660-700)")
print("=" * 60)
print(sh("sed -n '650,700p' quant_portal.py"))

print("=" * 60)
print("E. ai_chatbot _create_price_chart signature (for reusing in opportunity charts)")
print("=" * 60)
print(sh("grep -n -A8 'def _create_price_chart' ai_chatbot.py | head -12"))
print(sh("grep -n -A8 'def _create_advanced_price_chart' ai_chatbot.py | head -12"))

print("=" * 60)
print("F. FactorCrowdingEngine build_dashboard return structure")
print("=" * 60)
print(sh("grep -n -A30 'def build_dashboard' factor_crowding_engine.py | head -40"))
