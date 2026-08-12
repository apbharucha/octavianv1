import sys
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')
from chatbot_eval.pipeline import mocked_pipeline
from financial_llm_engine import generate_financial_analysis, expand_query_intents

QS = [
    'Assess the valuation of LOW using DCF logic: what growth and margin assumptions does the current price already embed?',
    'What is the probability TGT reaches $122 within 3 months, and what are the main upside and downside risks?',
    'What does the latest inflation data and Fed policy path imply for growth stocks like TGT over the next quarter?',
    'What are the key catalysts for TGT over the next 6 months, and which news events would cause the biggest repricing?',
    'Build a diversified portfolio with TGT, BKR and XLB: what weights, what risk controls, and what is the expected drawdown profile?',
]

with mocked_pipeline():
    for q in QS:
        i, t, s = expand_query_intents(q)
        print('=' * 100)
        print('Q:', q[:115])
        print('tickers:', t, '| sectors:', s)
        out = generate_financial_analysis(q)
        print('HEAD:', out[:300].replace('\n', ' | '))
        rl = out.lower()
        for term in ['target probability', 'tgt', 'bkr', 'xlb', 'valuation', 'dcf', 'catalyst']:
            if term in rl:
                print(f'  mentions {term}')
