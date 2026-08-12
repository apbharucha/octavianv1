import sys
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')
from chatbot_eval.pipeline import mocked_pipeline
from financial_llm_engine import (
    generate_financial_analysis,
    expand_query_intents,
    _is_deep_dive_query,
    _is_event_query,
    _is_focused_event_question,
)

QS = [
    'Write a complete investment memo on LOW for the investment committee. Sections: (1) thesis; (2) fundamental drivers by segment (growth, margins, cash generation); (3) valuation with DCF and comps; (4) risks; (5) catalysts; (6) final recommendation.',
    'Assess the valuation of LOW using DCF logic: what growth and margin assumptions does the current price already embed?',
    'what is the price outlook for sui?. Now, how likely is PINS to fall to $38 in 6 months?. Meanwhile, what is the best options strategy for earnings?',
    'Which currency is the best buy right now?',
    'Top banks stocks for growth',
    'How likely is LOW to reach $225 in 365 days?',
    'You are the lead investment strategist at a multi-strategy institutional asset manager. Conduct a full fundamental, quantitative, macroeconomic, market-structure, and sentiment-driven investment assessment of NVIDIA (NVDA) over the next 12-24 months. Build an evidence-based investment thesis. Reconstruct the revenue drivers by segment. Evaluate competitive position against AMD and custom ASICs. Determine whether CUDA represents a durable moat. Build a DCF with three scenarios, perform a reverse DCF. Determine what the equity market is pricing in. Analyze sensitivity to Fed policy and China restrictions. Build a risk matrix, catalyst timeline, five scenarios with probabilities, falsification, and final decision.',
]

with mocked_pipeline():
    for q in QS:
        i, t, s = expand_query_intents(q)
        print('=' * 100)
        print('Q:', q[:110])
        print('tickers:', t, '| sectors:', s)
        print('deep_dive:', _is_deep_dive_query(q, t), '| event_q:', _is_event_query(q),
              '| focused:', _is_focused_event_question(q, i, t))
        out = generate_financial_analysis(q)
        head = out[:600].replace('\n', ' | ')
        print('OUT:', head)
        # check for generic template leak
        leak = 'geopolitical' in out.lower() or 'risk-on / constructive' in out.lower()
        print('TEMPLATE_LEAK:', leak)
