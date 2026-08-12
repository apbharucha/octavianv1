import sys
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')
import chatbot_eval.pipeline as pipeline
import chatbot_eval.rubric as rubric
import chatbot_eval.prompt_factory as pf
from financial_llm_engine import generate_financial_analysis, expand_query_intents

rng = pf._rng if hasattr(pf, '_rng') else __import__('random').Random(42)

# Find the actual expectations for these queries by re-building corpus with same seed
# Simpler: run the queries through pipeline + rubric with best-effort expectations
QS = [
    'what is the price outlook for sui?. should i buy solana at these levels?',
    'how does the price of coffee affect the stock market? how likely is COST to fall to $837 in 18 months?',
    'is XLK overbought or oversold? what is the outlook for the VIX?',
]

with pipeline.mocked_pipeline():
    for q in QS:
        intents, tickers, sectors, text, elapsed = pipeline.run_query(q)
        visuals = pipeline.visual_check(q, tickers, 'mega_prompt')
        # build generic mega expectations: requires_any first entity
        exp = {
            'mega_tasks': [{'label': 'task1', 'requires_any': [tickers[0]] if tickers else []},
                           {'label': 'task2', 'requires_any': [tickers[-1]] if len(tickers) > 1 else []}],
            'primary': tickers[0] if tickers else '',
            'requires_any': [tickers[0]] if tickers else [],
            'category_flavor': 'mega',
        }
        scores = rubric.score_response(q, exp, intents, tickers, text, visuals)
        ov = rubric.overall_score(scores)
        ok = rubric.passed(scores)
        tags = rubric.failure_tags(scores)
        print('=' * 90)
        print('Q:', q[:110])
        print('tickers:', tickers, '| passed:', ok, '| score:', ov, '| tags:', tags)
        print('sections:', list(scores.keys()))
        for k in ('task_fulfillment', 'relevance', 'ticker_cleanliness'):
            if k in scores:
                print(f'  {k}: {scores[k]}')
