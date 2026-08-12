import sys
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')
from chatbot_eval.pipeline import mocked_pipeline
from financial_llm_engine import generate_financial_analysis

QS = [
    'Which technology stocks are undervalued?',
    'Which currency is the best buy right now?',
    'How does XPEV compare to UAL?',
    'Is heating oil a good buy right now?',
    'Is the dollar strengthening?',
    "Analyze NFLX's competitive moat versus FXI: pricing power, switching costs, network effects, and what could erode the moat over the next 5 years?",
    'Compare DIS and FXI on growth, margins, valuation and balance sheet strength, and tell me which is the better long-term investment',
    'Is FXI overvalued or undervalued at current levels, and what multiple should it trade at versus DIA?',
    'What are the key catalysts for FXI over the next 6 months, and which news events would cause the biggest repricing?',
    'Should I buy natural gas at these levels?',
]

with mocked_pipeline():
    for q in QS:
        out = generate_financial_analysis(q)
        head = out[:150].replace('\n', ' | ')
        print('Q:', q[:75])
        print('  ->', head)
