import subprocess

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.stdout + r.stderr

print("=" * 60)
print("A. All references to 'opportunity_charts' across codebase")
print("=" * 60)
print(sh("grep -rn 'opportunity_charts' --include='*.py' . | head -20"))

print("=" * 60)
print("B. Chatbot class definition and base classes")
print("=" * 60)
print(sh("grep -n 'class .*Chatbot\\|class Octavian' ai_chatbot.py | head -10"))

print("=" * 60)
print("C. custom_dashboard options error display (find the st.error string)")
print("=" * 60)
print(sh("grep -n 'Options analysis error\\|get_options_engine\\|options analysis' custom_dashboard.py"))

print("=" * 60)
print("D. _generate_unbiased_charts definition")
print("=" * 60)
print(sh("grep -n -A30 'def _generate_unbiased_charts' ai_chatbot.py | head -45"))

print("=" * 60)
print("E. How the scan query is dispatched (intent detection for scan)")
print("=" * 60)
print(sh("grep -n 'scan' ai_chatbot.py | grep -i 'intent\\|market_scan\\|unbiased' | head -15"))
