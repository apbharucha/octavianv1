import subprocess

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.stdout + r.stderr

for meth in ["_generate_unbiased_charts", "_generate_opportunity_charts",
             "_generate_unbiased_profit_response", "_create_unbiased_analysis_summary",
             "_extract_unbiased_insights", "_extract_profit_opportunities",
             "_generate_profit_suggestions", "_get_simulation_learnings",
             "_create_opportunity_summary", "_extract_scan_insights",
             "_format_profit_opportunities", "_generate_scan_suggestions"]:
    cnt = open("ai_chatbot.py").read().count(f"def {meth}")
    refs = open("ai_chatbot.py").read().count(f"self.{meth}")
    print(f"{meth}: defined={cnt}, referenced={refs}")