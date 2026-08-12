import sys
sys.path.insert(0, '.')
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')
from financial_llm_engine import expand_query_intents

QS = [
    "Analyze NFLX's competitive moat versus FXI: pricing power, switching costs, network effects, and what could erode the moat over the next 5 years?",
    "Compare DIS and FXI on growth, margins, valuation and balance sheet strength, and tell me which is the better long-term investment",
    "What are the key catalysts for FXI over the next 6 months?",
    "Is FXI overvalued or undervalued at current levels?",
    "Is heating oil a good buy right now?",
    "Which technology stocks are undervalued?",
    "How does XPEV compare to UAL?",
]

for q in QS:
    i, t, s = expand_query_intents(q)
    print('Q:', q[:70])
    print('  tickers:', t, '| sectors:', s, '| fx:', i.get('fx'), '| valuation:', i.get('valuation'),
          '| comparison:', i.get('comparison'), '| commodities:', i.get('commodities'))
