import subprocess

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.stdout + r.stderr

print("=" * 60)
print("A. process_unbiased_query definition start")
print("=" * 60)
print(sh("grep -n 'def process_unbiased_query' ai_chatbot.py"))

print("=" * 60)
print("B. Lines 430-560 (dispatch + symbol-based scan)")
print("=" * 60)
print(sh("sed -n '430,560p' ai_chatbot.py"))

print("=" * 60)
print("C. Lines 584-600 (the missing _generate_opportunity_charts call)")
print("=" * 60)
print(sh("sed -n '584,600p' ai_chatbot.py"))
