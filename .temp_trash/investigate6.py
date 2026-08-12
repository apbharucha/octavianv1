import subprocess

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.stdout + r.stderr

print("=" * 60)
print("A. generate_financial_analysis in financial_llm_engine")
print("=" * 60)
print(sh("grep -n 'def generate_financial_analysis\\|def expand_query_intents' financial_llm_engine.py"))
print(sh("grep -n -A40 'def generate_financial_analysis' financial_llm_engine.py | head -55"))

print("=" * 60)
print("B. Does financial_llm_engine fetch live data per symbol? (speed)")
print("=" * 60)
print(sh("grep -n 'get_stock\\|yf.\\|requests\\|urllib\\|fetch' financial_llm_engine.py | head -20"))

print("=" * 60)
print("C. show_octavian_chatbot - how queries are sent (around 5800-5870)")
print("=" * 60)
print(sh("grep -n 'def show_octavian_chatbot' ai_chatbot.py"))
print(sh("sed -n '5800,5870p' ai_chatbot.py"))

print("=" * 60)
print("D. _generate_enhanced_charts signature + body head (reusable chart builder)")
print("=" * 60)
print(sh("sed -n '4999,5060p' ai_chatbot.py"))

print("=" * 60)
print("E. _generate_intent_aware_charts signature")
print("=" * 60)
print(sh("sed -n '4187,4210p' ai_chatbot.py"))
