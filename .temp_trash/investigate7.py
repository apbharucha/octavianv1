import subprocess

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.stdout + r.stderr

print("=" * 60)
print("A. scan_entire_market in unbiased_market_analyzer")
print("=" * 60)
print(sh("grep -n -A35 'def scan_entire_market' unbiased_market_analyzer.py | head -45"))

print("=" * 60)
print("B. _fetch_live_data_for_tickers in financial_llm_engine")
print("=" * 60)
print(sh("sed -n '377,415p' financial_llm_engine.py"))

print("=" * 60)
print("C. presentation_generator body slide builder (263-330)")
print("=" * 60)
print(sh("sed -n '263,330p' presentation_generator.py"))

print("=" * 60)
print("D. quant_portal genetic algo context (680-710)")
print("=" * 60)
print(sh("sed -n '675,715p' quant_portal.py"))

print("=" * 60)
print("E. quant_portal VaR context (855-890)")
print("=" * 60)
print(sh("sed -n '850,890p' quant_portal.py"))

print("=" * 60)
print("F. _slide_body and _slide_toc details (233-296)")
print("=" * 60)
print(sh("sed -n '233,296p' presentation_generator.py"))
