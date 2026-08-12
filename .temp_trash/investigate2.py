import subprocess

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.stdout + r.stderr

print("=" * 60)
print("A. Is _generate_opportunity_charts defined ANYWHERE in ai_chatbot?")
print("=" * 60)
print(sh("grep -n 'def _generate_opportunity_charts\\|def _generate.*chart' ai_chatbot.py | head -20"))

print("=" * 60)
print("B. Line 3110-3130 (options analysis error site)")
print("=" * 60)
print(sh("sed -n '3095,3135p' ai_chatbot.py"))

print("=" * 60)
print("C. Options engine import in ai_chatbot")
print("=" * 60)
print(sh("grep -n 'options_engine\\|OptionsEngine\\|HAS_OPTIONS' ai_chatbot.py | head -20"))

print("=" * 60)
print("D. How charts render in the UI (line 5918 area)")
print("=" * 60)
print(sh("sed -n '5870,5920p' ai_chatbot.py"))

print("=" * 60)
print("E. What chart-generation helpers exist")
print("=" * 60)
print(sh("grep -n 'def _generate_unbiased_charts\\|def _generate_enhanced_charts\\|def _generate_intent' ai_chatbot.py"))
