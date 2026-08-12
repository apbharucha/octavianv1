import subprocess

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.stdout + r.stderr

print("=" * 60)
print("A. Do scan-path methods exist in ai_chatbot?")
print("=" * 60)
for m in ["_generate_unbiased_profit_response", "_create_unbiased_analysis_summary",
          "_extract_unbiased_insights", "_extract_profit_opportunities",
          "_generate_profit_suggestions", "_get_simulation_learnings",
          "_create_opportunity_summary", "_extract_scan_insights",
          "_format_profit_opportunities", "_generate_scan_suggestions",
          "_create_price_chart", "_create_advanced_price_chart",
          "_create_prediction_chart", "_create_volatility_chart"]:
    print(f"  {m}: {'YES' if m in open('ai_chatbot.py').read() else 'NO'}")

print("=" * 60)
print("B. GeneticStrategyEngine API")
print("=" * 60)
print(sh("grep -n 'def \\|class ' genetic_strategy_engine.py | head -25"))

print("=" * 60)
print("C. quant_portal: what symbols/data available near VaR (800-870)")
print("=" * 60)
print(sh("sed -n '790,855p' quant_portal.py"))

print("=" * 60)
print("D. quant_portal imports + HAS_GENETIC")
print("=" * 60)
print(sh("grep -n 'import\\|HAS_GENETIC\\|GeneticStrategy' quant_portal.py | head -40"))

print("=" * 60)
print("E. financial_llm_engine LLM call (lines 30-70)")
print("=" * 60)
print(sh("sed -n '30,70p' financial_llm_engine.py"))
