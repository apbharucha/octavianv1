import subprocess

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.stdout + r.stderr

print("=" * 60)
print("1. custom_dashboard HAS_OPTIONS check")
print("=" * 60)
print(sh("grep -n 'HAS_OPTIONS\\|try:\\|except' custom_dashboard.py | head -20"))
print(sh("sed -n '42,52p' custom_dashboard.py"))

print("=" * 60)
print("2. custom_dashboard options_engine usage at 883")
print("=" * 60)
print(sh("sed -n '877,892p' custom_dashboard.py"))

print("=" * 60)
print("3. quant_portal HAS_FACTOR or factor_crowding_engine import block")
print("=" * 60)
print(sh("sed -n '65,82p' quant_portal.py"))

print("=" * 60)
print("4. quant_portal VAAR context: where do symbols come from?")
print("=" * 60)
print(sh("sed -n '250,320p' quant_portal.py | head -40"))

print("=" * 60)
print("5. ai_chatbot: first process_unbiased_query vs second")
print("=" * 60)
print(sh("grep -n 'def process_unbiased_query' ai_chatbot.py"))