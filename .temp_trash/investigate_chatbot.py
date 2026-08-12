import subprocess

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.stdout + r.stderr

print("=" * 60)
print("A. Options analysis error in ai_chatbot")
print("=" * 60)
print(sh("grep -n 'Options analysis error\\|options analysis\\|Options Analysis\\|get_options_engine' ai_chatbot.py | head -20"))

print("=" * 60)
print("B. _generate_opportunity_charts definition and usage")
print("=" * 60)
print(sh("grep -n '_generate_opportunity_charts' ai_chatbot.py"))
print(sh("grep -n -A40 'def _generate_opportunity_charts' ai_chatbot.py | head -60"))

print("=" * 60)
print("C. charts_count tracking")
print("=" * 60)
print(sh("grep -n 'charts_count' ai_chatbot.py | head -20"))

print("=" * 60)
print("D. The scan pipeline (line 584 area - process_unbiased_query / scan)")
print("=" * 60)
print(sh("sed -n '500,600p' ai_chatbot.py"))

print("=" * 60)
print("E. How top_opportunities are produced (the scan)")
print("=" * 60)
print(sh("grep -n 'def scan_entire_market\\|def find_opportunities\\|def _perform_unbiased_market_scan' ai_chatbot.py | head"))
