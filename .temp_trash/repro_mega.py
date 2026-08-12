import sys
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')
from chatbot_eval.pipeline import mocked_pipeline
from financial_llm_engine import generate_financial_analysis

QS = [
    'what is the price outlook for sui?. should i buy solana at these levels?',
    'how does the price of coffee affect the stock market? how likely is COST to fall to $837 in 18 months?',
    'is XLK overbought or oversold? what is the outlook for the VIX?',
]

with mocked_pipeline():
    for q in QS:
        out = generate_financial_analysis(q)
        print('=' * 100)
        print('Q:', q[:120])
        parts = [l for l in out.split('\n') if l.strip().startswith('### Part')]
        print('PARTS:', parts)
        print('HEAD:', out[:200].replace('\n', ' | '))
        # Check sub-task coverage
        rl = out.lower()
        for term in ['sui', 'solana', 'coffee', 'cost', 'xlk', 'vix', 'stock market']:
            if term in rl:
                print(f'  mentions {term}')
