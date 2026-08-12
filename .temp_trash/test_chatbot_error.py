import sys
import os
os.environ["STREAMLIT_SERVER_PORT"] = "8502"
import streamlit as st
import asyncio
from ai_chatbot import get_chatbot

if "rt_data" not in st.session_state:
    st.session_state["rt_data"] = {}
if "pt_portfolio" not in st.session_state:
    st.session_state["pt_portfolio"] = {"positions": {}, "cash": 100000}

async def main():
    try:
        bot = get_chatbot()
        res = await bot.process_enhanced_query("what are some bearish stocks today?", "test_user")
        print("Success:")
        print(res)
    except Exception as e:
        import traceback
        traceback.print_exc()

asyncio.run(main())
