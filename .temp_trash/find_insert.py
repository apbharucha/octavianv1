import subprocess

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.stdout + r.stderr

print("=" * 60)
print("A. _create_advanced_price_chart (2216) — around it for method insertion")
print("=" * 60)
print(sh("sed -n '2200,2250p' ai_chatbot.py"))

print("=" * 60)
print("B. _generate_charts at 4807 context")
print("=" * 60)
print(sh("sed -n '4807,4840p' ai_chatbot.py"))

print("=" * 60)
print("C. _generate_enhanced_charts at 4999 context")
print("=" * 60)
print(sh("sed -n '4999,5050p' ai_chatbot.py"))

print("=" * 60)
print("D. What's right before _perform_unbiased_market_scan (550-560)")
print("=" * 60)
print(sh("sed -n '540,558p' ai_chatbot.py"))