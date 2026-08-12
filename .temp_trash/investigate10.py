import subprocess, re

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.stdout + r.stderr

print("=" * 60)
print("A. show_octavian_chatbot call to chatbot")
print("=" * 60)
print(sh("sed -n '5551,5590p' ai_chatbot.py"))

print("=" * 60)
print("B. The async call pattern (asyncio.run / get_event_loop)")
print("=" * 60)
print(sh("grep -B2 -A5 'process_enhanced_query\\|process_unbiased_query' ai_chatbot.py | grep -v '_generate\\|def process' | head -30"))

print("=" * 60)
print("C. The scan path in _perform_unbiased_market_scan — does it use real data from opps?")
print("=" * 60)
print(sh("grep -n 'class UnbiasedAnalysis' unbiased_market_analyzer.py"))
print(sh("grep -n -A20 '@dataclass\\|class UnbiasedAnalysis:' unbiased_market_analyzer.py | head -30"))

print("=" * 60)
print("D. Does UnbiasedAnalysis have .data or .df field? (for charts)")
print("=" * 60)
print(sh("grep -n '\\.data\\|\\.df\\|\\.close\\|\\.prices' unbiased_market_analyzer.py | grep -v 'class\\|def '| head -10"))